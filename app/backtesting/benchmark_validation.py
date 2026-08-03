"""Cross-asset validation for simple daily strategies with realistic execution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np

from app.backtesting.walk_forward import (
    Candidate,
    _driver_kwargs,
    build_walk_forward_folds,
    compound_returns,
    load_daily_data,
)
from app.trading.trader_driver import TraderDriver
from app.core.config import CURS


EXECUTION_FRICTION_RATE = 0.02
INITIAL_CAPITAL = 10_000.0
TARGET_STRATEGY_NAMES = (
    "cash",
    "buy_hold",
    "sma200_trend",
    "dual_ma_50_200",
    "volatility_target",
    "regime_switch",
)
STRATEGY_NAMES = (*TARGET_STRATEGY_NAMES, "ma_boll_fixed")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        result[index] = float(np.std(values[index - window + 1 : index + 1]))
    return result


def build_strategy_targets(data: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Create close-derived targets; every target is executed at the next open."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    sma20 = _rolling_mean(closes, 20)
    sma50 = _rolling_mean(closes, 50)
    sma200 = _rolling_mean(closes, 200)
    returns = np.zeros(len(closes))
    returns[1:] = np.diff(closes) / closes[:-1]
    vol20 = _rolling_std(returns, 20)
    price_std20 = _rolling_std(closes, 20)

    cash = np.zeros(len(closes))
    buy_hold = np.ones(len(closes))
    sma_trend = np.where(np.isnan(sma200), 0.0, (closes > sma200).astype(float))
    dual_ma = np.where(
        np.isnan(sma200) | np.isnan(sma50), 0.0, (sma50 > sma200).astype(float)
    )
    daily_target_vol = 0.50 / np.sqrt(365.0)
    vol_target = np.where(
        np.isnan(vol20) | (vol20 <= 0),
        0.0,
        np.clip(daily_target_vol / vol20, 0.0, 1.0),
    )

    regime_target = np.zeros(len(closes))
    regimes = np.full(len(closes), "WARMUP", dtype=object)
    previous_target = 0.0
    for index in range(len(closes)):
        if index < 219 or np.isnan(sma200[index]):
            continue
        slope = sma200[index] / sma200[index - 20] - 1.0
        if closes[index] > sma200[index] and slope > 0:
            regimes[index] = "BULL"
            previous_target = 1.0
        elif closes[index] < sma200[index] and slope < 0:
            regimes[index] = "BEAR"
            previous_target = 0.0
        else:
            regimes[index] = "RANGE"
            lower = sma20[index] - 2.0 * price_std20[index]
            upper = sma20[index] + 2.0 * price_std20[index]
            if closes[index] < lower:
                previous_target = 1.0
            elif closes[index] > upper:
                previous_target = 0.0
        regime_target[index] = previous_target

    return {
        "targets": {
            "cash": cash,
            "buy_hold": buy_hold,
            "sma200_trend": sma_trend,
            "dual_ma_50_200": dual_ma,
            "volatility_target": vol_target,
            "regime_switch": regime_target,
        },
        "regimes": regimes,
    }


def simulate_target_strategy(
    data: Sequence[Sequence[Any]],
    targets: Sequence[float],
    test_start: int,
    test_end: int,
    slippage_bps: float = 10.0,
    rebalance_threshold: float = 0.10,
) -> dict[str, float | int]:
    """Trade previous-close targets at the next open with 2% friction and slippage."""
    if slippage_bps < 0:
        raise ValueError("slippage_bps cannot be negative")
    cash = INITIAL_CAPITAL
    coin = 0.0
    trades = 0
    turnover = 0.0
    equity_curve = []
    exposures = []
    slip = slippage_bps / 10_000.0

    for index in range(test_start, test_end):
        close_price = float(data[index][0])
        open_price = float(data[index][2])
        desired = float(np.clip(targets[index - 1], 0.0, 1.0))
        open_equity = cash + coin * open_price
        current = 0.0 if open_equity <= 0 else coin * open_price / open_equity

        if abs(desired - current) >= rebalance_threshold:
            if desired > current:
                spend = min(cash, (desired - current) * open_equity)
                if spend > 0:
                    execution_price = open_price * (1.0 + slip)
                    coin += spend * (1.0 - EXECUTION_FRICTION_RATE) / execution_price
                    cash -= spend
                    turnover += spend / open_equity
                    trades += 1
            else:
                sell_value = min(coin * open_price, (current - desired) * open_equity)
                if sell_value > 0:
                    execution_price = open_price * (1.0 - slip)
                    sell_coin = min(coin, sell_value / open_price)
                    coin -= sell_coin
                    cash += (
                        sell_coin * execution_price * (1.0 - EXECUTION_FRICTION_RATE)
                    )
                    turnover += sell_value / open_equity
                    trades += 1

        equity = cash + coin * close_price
        equity_curve.append(equity)
        exposures.append(0.0 if equity <= 0 else coin * close_price / equity)

    peaks = np.maximum.accumulate(np.asarray(equity_curve))
    drawdowns = (peaks - np.asarray(equity_curve)) / peaks
    return {
        "return": (equity_curve[-1] / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_drawdown": float(np.max(drawdowns)) * 100.0,
        "transactions": trades,
        "turnover": turnover,
        "mean_exposure": mean(exposures),
    }


def simulate_fixed_ma_boll(
    data: Sequence[Sequence[Any]],
    test_start: int,
    test_end: int,
    slippage_bps: float = 10.0,
    warmup_days: int = 35,
) -> dict[str, float | int | None]:
    """Run one shared MA-BOLL configuration without per-asset tuning."""
    candidate = Candidate(
        strategy="MA-BOLL-BANDS",
        tol_pct=0.1,
        buy_pct=0.5,
        sell_pct=0.5,
        bollinger_sigma=2.0,
    )
    warmup_start = max(0, test_start - warmup_days)
    combined = list(data[warmup_start:test_end])
    warmup_points = test_start - warmup_start
    driver = TraderDriver(**_driver_kwargs("MA_BOLL_FIXED", candidate, slippage_bps))
    driver.feed_data(combined, warmup_points=warmup_points)
    trader = driver.traders[0]
    return {
        "return": float(trader.rate_of_return),
        "max_drawdown": float(trader.max_drawdown) * 100.0,
        "transactions": int(trader.num_transaction),
        "turnover": None,
        "mean_exposure": None,
    }


def validate_symbol(
    symbol: str,
    data: Sequence[Sequence[Any]],
    train_days: int = 365,
    validation_days: int = 90,
    test_days: int = 30,
    purge_days: int = 7,
    slippage_bps: float = 10.0,
) -> dict[str, Any]:
    features = build_strategy_targets(data)
    folds = build_walk_forward_folds(
        len(data), train_days, validation_days, test_days, purge_days
    )
    fold_results = []
    for fold in folds:
        strategies = {
            name: simulate_target_strategy(
                data,
                features["targets"][name],
                fold.test_start,
                fold.test_end,
                slippage_bps,
            )
            for name in TARGET_STRATEGY_NAMES
        }
        strategies["ma_boll_fixed"] = simulate_fixed_ma_boll(
            data,
            fold.test_start,
            fold.test_end,
            slippage_bps,
        )
        market_return = (
            float(data[fold.test_end - 1][0]) / float(data[fold.test_start][0]) - 1.0
        ) * 100.0
        benchmark_return = strategies["buy_hold"]["return"]
        for metrics in strategies.values():
            metrics["excess_vs_buy_hold"] = metrics["return"] - benchmark_return
        signal_regimes = features["regimes"][fold.test_start - 1 : fold.test_end - 1]
        fold_results.append(
            {
                "fold": fold.fold,
                "test_start": data[fold.test_start][1],
                "test_end": data[fold.test_end - 1][1],
                "market_return": market_return,
                "regime_counts": dict(Counter(signal_regimes)),
                "strategies": strategies,
            }
        )

    oos_start = folds[0].test_start
    oos_end = folds[-1].test_end
    continuous_results = {
        name: simulate_target_strategy(
            data,
            features["targets"][name],
            oos_start,
            oos_end,
            slippage_bps,
        )
        for name in TARGET_STRATEGY_NAMES
    }
    continuous_results["ma_boll_fixed"] = simulate_fixed_ma_boll(
        data,
        oos_start,
        oos_end,
        slippage_bps,
    )
    continuous_benchmark = continuous_results["buy_hold"]["return"]

    summaries = {}
    for name in STRATEGY_NAMES:
        returns = [fold["strategies"][name]["return"] for fold in fold_results]
        excess = [
            fold["strategies"][name]["excess_vs_buy_hold"] for fold in fold_results
        ]
        summaries[name] = {
            "continuous_oos_return": continuous_results[name]["return"],
            "continuous_excess_vs_buy_hold": (
                continuous_results[name]["return"] - continuous_benchmark
            ),
            "fold_compounded_return": compound_returns(returns),
            "mean_fold_return": mean(returns),
            "median_fold_return": median(returns),
            "mean_excess_vs_buy_hold": mean(excess),
            "excess_win_rate": sum(value > 0 for value in excess) / len(excess),
            "continuous_transactions": continuous_results[name]["transactions"],
            "continuous_turnover": continuous_results[name]["turnover"],
            "continuous_max_drawdown": continuous_results[name]["max_drawdown"],
            "mean_fold_max_drawdown": mean(
                fold["strategies"][name]["max_drawdown"] for fold in fold_results
            ),
        }

    return {
        "symbol": symbol,
        "data_start": data[0][1],
        "data_end": data[-1][1],
        "data_rows": len(data),
        "folds": fold_results,
        "summary": summaries,
    }


def aggregate_assets(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for name in STRATEGY_NAMES:
        returns = [
            result["summary"][name]["continuous_oos_return"] for result in results
        ]
        excess = [
            result["summary"][name]["continuous_excess_vs_buy_hold"]
            for result in results
        ]
        aggregate[name] = {
            "median_asset_return": median(returns),
            "median_asset_excess_vs_buy_hold": median(excess),
            "positive_return_assets": sum(value > 0 for value in returns),
            "positive_excess_assets": sum(value > 0 for value in excess),
            "asset_count": len(results),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-asset daily strategy validation"
    )
    parser.add_argument("--symbols", nargs="+", default=CURS)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--purge-days", type=int, default=7)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        data = load_daily_data(symbol)
        result = validate_symbol(
            symbol.upper(),
            data,
            args.train_days,
            args.validation_days,
            args.test_days,
            args.purge_days,
            args.slippage_bps,
        )
        results.append(result)
        regime = result["summary"]["regime_switch"]
        print(
            f"{symbol.upper()}: regime={regime['continuous_oos_return']:.2f}%, "
            f"excess={regime['continuous_excess_vs_buy_hold']:.2f}%"
        )

    report = {
        "configuration": {
            "execution_friction_rate": EXECUTION_FRICTION_RATE,
            "slippage_bps": args.slippage_bps,
            "signal_time": "daily_close",
            "execution_time": "next_daily_open",
            "options_enabled": False,
            "leverage_enabled": False,
            "parameters_shared_across_assets": True,
            "regime_parameters": {
                "trend_ma_days": 200,
                "trend_slope_days": 20,
                "range_bollinger_days": 20,
                "range_bollinger_sigma": 2.0,
            },
            "fixed_ma_boll_parameters": {
                "tol_pct": 0.1,
                "buy_pct": 0.5,
                "sell_pct": 0.5,
                "bollinger_sigma": 2.0,
            },
        },
        "assets": results,
        "cross_asset_summary": aggregate_assets(results),
    }
    output = args.output or Path("artifacts/backtests") / (
        f"cross_asset_benchmarks_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
