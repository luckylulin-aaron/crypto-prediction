"""Validate the frozen COIN and MSFT defensive strategies against buy-and-hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    simulate_target_strategy,
)
from app.backtesting.tencent_defensive_validation import load_stock_daily_data
from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    COIN_BTC_SMA200_DEFENSIVE_PARAMETERS,
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
    EMA_LENGTHS,
    MA_LENGTHS,
    MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS,
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
    SELL_STAS,
    STOCK_SLIPPAGE_BPS,
)
from app.trading.trader_driver import TraderDriver


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def build_coin_btc_targets(
    coin_data: Sequence[Sequence[Any]],
    btc_data: Sequence[Sequence[Any]],
) -> np.ndarray:
    """Use only a completed, prior-calendar-day BTC close for every COIN signal."""
    params = COIN_BTC_SMA200_DEFENSIVE_PARAMETERS
    closes = np.asarray([float(row[0]) for row in btc_data], dtype=float)
    average = _rolling_mean(closes, 200)
    context_targets = np.ones(len(closes))
    active = True
    for index in range(199, len(closes)):
        if closes[index] > average[index] * (1.0 + params["entry_band_pct"]):
            active = True
        elif closes[index] < average[index] * (1.0 - params["exit_band_pct"]):
            active = False
        context_targets[index] = float(active)

    context_by_date = {
        str(row[1])[:10]: float(context_targets[index])
        for index, row in enumerate(btc_data)
    }
    lag_days = int(params["context_lag_days"])
    targets = []
    for row in coin_data:
        signal_date = np.datetime64(str(row[1])[:10]) - np.timedelta64(lag_days, "D")
        targets.append(context_by_date.get(str(signal_date), 1.0))
    return np.asarray(targets, dtype=float)


def build_msft_breakout_targets(
    data: Sequence[Sequence[Any]],
) -> np.ndarray:
    """Bootstrap invested, stop 10% below the peak, and re-enter on a 20-day high."""
    params = MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    targets = np.ones(len(closes))
    in_position = True
    peak = closes[0]
    for index in range(1, len(closes)):
        close = closes[index]
        if in_position:
            peak = max(peak, close)
            if close < peak * (1.0 - params["trailing_stop_pct"]):
                in_position = False
        elif index >= params["lookback_days"]:
            prior = closes[index - params["lookback_days"] : index]
            if close > float(np.max(prior)):
                in_position = True
                peak = close
        targets[index] = float(in_position)
    return targets


def _compare(data, targets, start, end):
    strategy = simulate_target_strategy(
        data, targets, start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    baseline = simulate_target_strategy(
        data, np.ones(len(data)), start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    return {
        "start": data[start][1],
        "end": data[end - 1][1],
        "strategy": strategy,
        "buy_and_hold": baseline,
        "excess_return": strategy["return"] - baseline["return"],
    }


def _registered_runtime(symbol: str, strategy: str, data, btc_data=None):
    driver = TraderDriver(
        name=symbol,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[strategy],
        tol_pcts=[0.0],
        ma_lengths=MA_LENGTHS,
        ema_lengths=EMA_LENGTHS,
        bollinger_mas=BOLLINGER_MAS,
        bollinger_tols=[2],
        buy_pcts=[1.0],
        sell_pcts=[1.0],
        buy_stas=BUY_STAS,
        sell_stas=SELL_STAS,
        enable_options=False,
        btc_data_stream=btc_data,
    )
    driver.feed_data(data)
    trader = driver.traders[0]
    trades = trader.all_history_trade_only
    return {
        "return": (float(trader.portfolio_value) / 10_000.0 - 1.0) * 100.0,
        "max_drawdown": float(trader.max_drawdown) * 100.0,
        "transactions": len(trades),
        "final_value": float(trader.portfolio_value),
        "parameters": trader.trading_strategy,
        "trades": [
            {
                "date": str(item["date"]),
                "action": item["action"],
                "price": float(item["price"]),
            }
            for item in trades
        ],
    }


def _asset_report(symbol, strategy, data, targets, btc_data=None):
    periods = {
        "selection": _compare(data, targets, 1, 455),
        "validation": _compare(data, targets, 365, 455),
        "test": _compare(data, targets, 462, len(data)),
        "full": _compare(data, targets, 1, len(data)),
    }
    runtime = _registered_runtime(symbol, strategy, data, btc_data)
    runtime["excess_return"] = (
        runtime["return"] - periods["full"]["buy_and_hold"]["return"]
    )
    runtime_matches = bool(
        abs(runtime["return"] - periods["full"]["strategy"]["return"]) < 1e-9
        and runtime["transactions"] == periods["full"]["strategy"]["transactions"]
    )
    passes_profit_gate = bool(
        periods["test"]["excess_return"] > 0.0
        and periods["full"]["excess_return"] > 0.0
        and periods["full"]["strategy"]["max_drawdown"]
        < periods["full"]["buy_and_hold"]["max_drawdown"]
        and runtime_matches
    )
    return {
        "configuration": {
            "asset": symbol,
            "strategy": strategy,
            "data_start": data[0][1],
            "data_end": data[-1][1],
            "data_rows": len(data),
            "execution_friction_rate": EXECUTION_FRICTION_RATE,
            "slippage_bps": STOCK_SLIPPAGE_BPS,
            "execution": "prior close signal, next stock-session open fill",
        },
        **periods,
        "registered_runtime": runtime,
        "runtime_matches_offline": runtime_matches,
        "passes_profit_gate": passes_profit_gate,
    }


def run_validation() -> dict[str, Any]:
    coin = load_stock_daily_data("COIN")
    btc = load_stock_daily_data(
        str(COIN_BTC_SMA200_DEFENSIVE_PARAMETERS["context_symbol"])
    )
    msft = load_stock_daily_data("MSFT")
    return {
        "COIN": _asset_report(
            "COIN",
            COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
            coin,
            build_coin_btc_targets(coin, btc),
            btc,
        ),
        "MSFT": _asset_report(
            "MSFT",
            MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
            msft,
            build_msft_breakout_targets(msft),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/backtests/held_stock_defensive_validation.json"),
    )
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for symbol, result in report.items():
        print(
            f"{result['configuration']['strategy']}: gate={result['passes_profit_gate']}"
        )
        for split in ("selection", "validation", "test", "full"):
            period = result[split]
            print(
                f"  {split}: strategy={period['strategy']['return']:.2f}%, "
                f"buy_hold={period['buy_and_hold']['return']:.2f}%, "
                f"excess={period['excess_return']:.2f}%"
            )
        runtime = result["registered_runtime"]
        print(
            f"  runtime: return={runtime['return']:.2f}%, "
            f"drawdown={runtime['max_drawdown']:.2f}%, "
            f"transactions={runtime['transactions']}, "
            f"matches={result['runtime_matches_offline']}"
        )
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
