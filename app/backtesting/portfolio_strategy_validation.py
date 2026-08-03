"""Offline validation for cross-asset trend rotation strategies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    INITIAL_CAPITAL,
)
from app.backtesting.walk_forward import load_daily_data
from app.core.config import BTC_SMA200_DEFENSIVE_STRATEGY, CURS


@dataclass(frozen=True)
class RotationParameters:
    momentum_days: int
    top_n: int
    rebalance_days: int
    inverse_volatility: bool
    bollinger_pullback: bool


@dataclass(frozen=True)
class TrendParameters:
    entry_band_pct: float
    exit_band_pct: float
    bollinger_entry_sigma: float | None


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


def load_aligned_assets(
    symbols: Sequence[str],
) -> tuple[list[str], dict[str, list[list[Any]]]]:
    data_by_asset = {symbol: load_daily_data(symbol) for symbol in symbols}
    if any(not data for data in data_by_asset.values()):
        missing = [symbol for symbol, data in data_by_asset.items() if not data]
        raise ValueError(f"Missing daily data for: {', '.join(missing)}")

    reference_symbol = symbols[0]
    reference_dates = [row[1] for row in data_by_asset[reference_symbol]]
    for symbol, data in data_by_asset.items():
        dates = [row[1] for row in data]
        if dates != reference_dates:
            raise ValueError(f"Daily dates are not aligned for {symbol}")
    return reference_dates, data_by_asset


def _feature_matrices(
    symbols: Sequence[str], data_by_asset: dict[str, list[list[Any]]]
) -> dict[str, np.ndarray]:
    closes = np.column_stack(
        [[float(row[0]) for row in data_by_asset[symbol]] for symbol in symbols]
    )
    opens = np.column_stack(
        [[float(row[2]) for row in data_by_asset[symbol]] for symbol in symbols]
    )
    sma20 = np.column_stack(
        [_rolling_mean(closes[:, index], 20) for index in range(len(symbols))]
    )
    std20 = np.column_stack(
        [_rolling_std(closes[:, index], 20) for index in range(len(symbols))]
    )
    sma200 = np.column_stack(
        [_rolling_mean(closes[:, index], 200) for index in range(len(symbols))]
    )
    returns = np.zeros_like(closes)
    returns[1:] = closes[1:] / closes[:-1] - 1.0
    vol20 = np.column_stack(
        [_rolling_std(returns[:, index], 20) for index in range(len(symbols))]
    )
    return {
        "closes": closes,
        "opens": opens,
        "sma20": sma20,
        "std20": std20,
        "sma200": sma200,
        "vol20": vol20,
    }


def _hysteresis_states(
    closes: np.ndarray,
    sma200: np.ndarray,
    entry_band_pct: float = 0.05,
    exit_band_pct: float = 0.05,
) -> np.ndarray:
    states = np.zeros_like(closes, dtype=bool)
    active = np.zeros(closes.shape[1], dtype=bool)
    for index in range(len(closes)):
        valid = ~np.isnan(sma200[index])
        active[valid & (closes[index] > sma200[index] * (1.0 + entry_band_pct))] = True
        active[valid & (closes[index] < sma200[index] * (1.0 - exit_band_pct))] = False
        active[~valid] = False
        states[index] = active
    return states


def build_equal_weight_targets(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    btc_gate: bool,
) -> np.ndarray:
    closes = features["closes"]
    states = _hysteresis_states(closes, features["sma200"])
    targets = np.zeros_like(closes)
    btc_index = symbols.index("BTC")
    for index in range(len(closes)):
        eligible = states[index].copy()
        if btc_gate and not states[index, btc_index]:
            eligible[:] = False
        count = int(np.sum(eligible))
        if count:
            targets[index, eligible] = 1.0 / count
    return targets


def build_rotation_targets(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    params: RotationParameters,
) -> np.ndarray:
    closes = features["closes"]
    sma20 = features["sma20"]
    std20 = features["std20"]
    vol20 = features["vol20"]
    states = _hysteresis_states(closes, features["sma200"])
    targets = np.zeros_like(closes)
    btc_index = symbols.index("BTC")
    selected: list[int] = []
    selected_weights: dict[int, float] = {}

    for index in range(len(closes)):
        if not states[index, btc_index]:
            selected = []
            selected_weights = {}
            continue

        selected = [asset for asset in selected if states[index, asset]]
        selected_weights = {
            asset: weight
            for asset, weight in selected_weights.items()
            if asset in selected
        }
        scheduled = index % params.rebalance_days == 0
        if scheduled and index >= max(200, params.momentum_days):
            scores = []
            for asset in range(len(symbols)):
                if not states[index, asset]:
                    continue
                past = closes[index - params.momentum_days, asset]
                volatility = vol20[index, asset]
                if past <= 0 or np.isnan(volatility) or volatility <= 0:
                    continue
                momentum = closes[index, asset] / past - 1.0
                score = momentum / volatility
                if params.bollinger_pullback and asset not in selected:
                    upper_entry = sma20[index, asset] + 0.5 * std20[index, asset]
                    if np.isnan(upper_entry) or closes[index, asset] > upper_entry:
                        continue
                scores.append((score, asset))
            scores.sort(reverse=True)
            selected = [asset for _, asset in scores[: params.top_n]]
            selected_weights = {}

        if not selected:
            continue
        if not selected_weights and params.inverse_volatility:
            inverse_vol = np.asarray(
                [
                    1.0 / vol20[index, asset] if vol20[index, asset] > 0 else 0.0
                    for asset in selected
                ]
            )
            if np.sum(inverse_vol) <= 0:
                continue
            weights = inverse_vol / np.sum(inverse_vol)
            selected_weights = {
                asset: float(weight) for asset, weight in zip(selected, weights)
            }
        elif not selected_weights:
            selected_weights = {asset: 1.0 / len(selected) for asset in selected}
        else:
            total_weight = sum(selected_weights.values())
            selected_weights = {
                asset: weight / total_weight
                for asset, weight in selected_weights.items()
            }
        for asset, weight in selected_weights.items():
            targets[index, asset] = weight
    return targets


def build_single_asset_trend_targets(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    symbol: str,
    params: TrendParameters,
) -> np.ndarray:
    closes = features["closes"]
    sma200 = features["sma200"]
    sma20 = features["sma20"]
    std20 = features["std20"]
    asset = symbols.index(symbol)
    targets = np.zeros_like(closes)
    active = False

    for index in range(len(closes)):
        long_average = sma200[index, asset]
        if np.isnan(long_average):
            active = False
            continue
        close = closes[index, asset]
        if active and close < long_average * (1.0 - params.exit_band_pct):
            active = False
        elif not active and close > long_average * (1.0 + params.entry_band_pct):
            if params.bollinger_entry_sigma is None:
                active = True
            else:
                entry_ceiling = (
                    sma20[index, asset]
                    + params.bollinger_entry_sigma * std20[index, asset]
                )
                if not np.isnan(entry_ceiling) and close <= entry_ceiling:
                    active = True
        if active:
            targets[index, asset] = 1.0
    return targets


def simulate_portfolio(
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    test_start: int,
    test_end: int,
    slippage_bps: float = 10.0,
) -> dict[str, Any]:
    if test_start <= 0 or test_end <= test_start:
        raise ValueError("Invalid test range")
    opens = features["opens"]
    closes = features["closes"]
    asset_count = closes.shape[1]
    cash = INITIAL_CAPITAL
    quantities = np.zeros(asset_count)
    trades = 0
    turnover = 0.0
    equity_curve = []
    exposure_curve = []
    executed_target = np.zeros(asset_count)
    slip = slippage_bps / 10_000.0

    for index in range(test_start, test_end):
        desired = np.asarray(targets[index - 1], dtype=float)
        if not np.allclose(desired, executed_target, atol=1e-10):
            open_prices = opens[index]
            starting_equity = cash + float(np.dot(quantities, open_prices))
            desired_values = desired * starting_equity
            current_values = quantities * open_prices

            for asset in range(asset_count):
                excess_value = current_values[asset] - desired_values[asset]
                if excess_value <= 1e-8:
                    continue
                sell_quantity = min(
                    quantities[asset], excess_value / open_prices[asset]
                )
                execution_price = open_prices[asset] * (1.0 - slip)
                quantities[asset] -= sell_quantity
                cash += (
                    sell_quantity * execution_price * (1.0 - EXECUTION_FRICTION_RATE)
                )
                turnover += excess_value / starting_equity
                trades += 1

            current_values = quantities * open_prices
            deficits = np.maximum(0.0, desired_values - current_values)
            total_deficit = float(np.sum(deficits))
            scale = min(1.0, cash / total_deficit) if total_deficit > 0 else 0.0
            for asset in range(asset_count):
                spend = deficits[asset] * scale
                if spend <= 1e-8:
                    continue
                execution_price = open_prices[asset] * (1.0 + slip)
                quantities[asset] += (
                    spend * (1.0 - EXECUTION_FRICTION_RATE) / execution_price
                )
                cash -= spend
                turnover += spend / starting_equity
                trades += 1
            executed_target = desired.copy()

        equity = cash + float(np.dot(quantities, closes[index]))
        equity_curve.append(equity)
        invested = float(np.dot(quantities, closes[index]))
        exposure_curve.append(invested / equity if equity > 0 else 0.0)

    curve = np.asarray(equity_curve)
    curve_with_initial_capital = np.concatenate(([INITIAL_CAPITAL], curve))
    peaks = np.maximum.accumulate(curve_with_initial_capital)
    drawdowns = (peaks - curve_with_initial_capital) / peaks
    return {
        "return": (curve[-1] / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_drawdown": float(np.max(drawdowns)) * 100.0,
        "transactions": trades,
        "turnover": turnover,
        "mean_exposure": float(np.mean(exposure_curve)),
        "final_value": float(curve[-1]),
    }


def _score(metrics: dict[str, Any]) -> float:
    return float(metrics["return"]) - 0.25 * float(metrics["max_drawdown"])


def run_validation(
    symbols: Sequence[str],
    train_days: int = 365,
    validation_days: int = 90,
    purge_days: int = 7,
    slippage_bps: float = 10.0,
) -> dict[str, Any]:
    dates, data_by_asset = load_aligned_assets(symbols)
    features = _feature_matrices(symbols, data_by_asset)
    validation_end = train_days + validation_days
    test_start = validation_end + purge_days
    if len(dates) <= test_start:
        raise ValueError("Not enough rows for the requested split")

    cash_targets = np.zeros_like(features["closes"])
    buy_hold_targets = np.full_like(features["closes"], 1.0 / len(symbols), dtype=float)
    equal_targets = build_equal_weight_targets(features, symbols, btc_gate=False)
    gated_equal_targets = build_equal_weight_targets(features, symbols, btc_gate=True)

    candidates = [
        RotationParameters(momentum, top_n, rebalance, inverse_vol, bollinger)
        for momentum, top_n, rebalance, inverse_vol, bollinger in product(
            (60, 90, 120),
            (2, 3),
            (14, 30),
            (False, True),
            (False, True),
        )
    ]
    candidate_results = []
    for candidate in candidates:
        targets = build_rotation_targets(features, symbols, candidate)
        train_metrics = simulate_portfolio(
            features, targets, 200, train_days, slippage_bps
        )
        validation_metrics = simulate_portfolio(
            features, targets, train_days, validation_end, slippage_bps
        )
        candidate_results.append(
            {
                "parameters": candidate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

    shortlist = sorted(
        candidate_results, key=lambda item: _score(item["train"]), reverse=True
    )[:12]
    selected_result = max(shortlist, key=lambda item: _score(item["validation"]))
    selected = selected_result["parameters"]
    selected_targets = build_rotation_targets(features, symbols, selected)

    trend_candidates = [
        TrendParameters(entry_band, exit_band, bollinger_sigma)
        for entry_band, exit_band, bollinger_sigma in product(
            (0.0, 0.03, 0.05),
            (0.0, 0.03, 0.05),
            (None, 0.5, 1.0),
        )
    ]
    trend_candidate_results = []
    for candidate in trend_candidates:
        targets = build_single_asset_trend_targets(features, symbols, "BTC", candidate)
        trend_candidate_results.append(
            {
                "parameters": candidate,
                "train": simulate_portfolio(
                    features, targets, 200, train_days, slippage_bps
                ),
                "validation": simulate_portfolio(
                    features, targets, train_days, validation_end, slippage_bps
                ),
            }
        )
    trend_shortlist = sorted(
        trend_candidate_results,
        key=lambda item: _score(item["train"]),
        reverse=True,
    )[:9]
    selected_trend_result = max(
        trend_shortlist, key=lambda item: _score(item["validation"])
    )
    selected_trend = selected_trend_result["parameters"]
    selected_trend_targets = build_single_asset_trend_targets(
        features, symbols, "BTC", selected_trend
    )

    fixed_rotation = RotationParameters(90, 2, 30, False, False)
    fixed_bollinger = RotationParameters(90, 2, 30, False, True)
    fixed_btc_defensive = TrendParameters(0.05, 0.05, None)
    btc_defensive_targets = build_single_asset_trend_targets(
        features, symbols, "BTC", fixed_btc_defensive
    )
    btc_buy_hold_targets = np.zeros_like(features["closes"])
    btc_buy_hold_targets[:, symbols.index("BTC")] = 1.0
    strategy_targets = {
        "cash": cash_targets,
        "equal_buy_hold": buy_hold_targets,
        "sma5_equal": equal_targets,
        "btc_gate_sma5_equal": gated_equal_targets,
        "btc_gate_top2_momentum": build_rotation_targets(
            features, symbols, fixed_rotation
        ),
        "btc_gate_top2_momentum_bollinger": build_rotation_targets(
            features, symbols, fixed_bollinger
        ),
        "selected_rotation": selected_targets,
        "selected_btc_trend": selected_trend_targets,
        "btc_sma5_defensive": btc_defensive_targets,
        "btc_buy_hold": btc_buy_hold_targets,
    }
    test_results = {
        name: simulate_portfolio(
            features, targets, test_start, len(dates), slippage_bps
        )
        for name, targets in strategy_targets.items()
    }

    long_horizon_results = {
        "btc_sma5_defensive": simulate_portfolio(
            features, btc_defensive_targets, 200, len(dates), slippage_bps
        ),
        "btc_buy_hold": simulate_portfolio(
            features, btc_buy_hold_targets, 200, len(dates), slippage_bps
        ),
    }

    return {
        "configuration": {
            "symbols": list(symbols),
            "data_start": dates[0],
            "data_end": dates[-1],
            "data_rows": len(dates),
            "train": {"start": dates[0], "end": dates[train_days - 1]},
            "validation": {
                "start": dates[train_days],
                "end": dates[validation_end - 1],
            },
            "purge_days": purge_days,
            "test": {"start": dates[test_start], "end": dates[-1]},
            "execution_friction_rate": EXECUTION_FRICTION_RATE,
            "slippage_bps": slippage_bps,
            "signal_time": "daily_close",
            "execution_time": "next_daily_open",
        },
        "selected_parameters": asdict(selected),
        "selected_train": selected_result["train"],
        "selected_validation": selected_result["validation"],
        "selected_btc_trend_parameters": asdict(selected_trend),
        "selected_btc_trend_train": selected_trend_result["train"],
        "selected_btc_trend_validation": selected_trend_result["validation"],
        "recommended_strategy": {
            "name": BTC_SMA200_DEFENSIVE_STRATEGY,
            "parameters": asdict(fixed_btc_defensive),
            "rationale": (
                "Fixed low-turnover BTC trend filter; rotation and Bollinger-entry "
                "ablations failed under the same cost model."
            ),
        },
        "long_horizon_results": long_horizon_results,
        "btc_trend_candidate_results": [
            {
                "parameters": asdict(item["parameters"]),
                "train": item["train"],
                "validation": item["validation"],
            }
            for item in trend_candidate_results
        ],
        "candidate_results": [
            {
                "parameters": asdict(item["parameters"]),
                "train": item["train"],
                "validation": item["validation"],
            }
            for item in candidate_results
        ],
        "test_results": test_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=CURS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_validation(args.symbols)
    output = args.output or Path(
        "artifacts/backtests/portfolio_strategy_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected rotation parameters:", report["selected_parameters"])
    print(
        "Selected BTC trend parameters:",
        report["selected_btc_trend_parameters"],
    )
    for name, metrics in report["test_results"].items():
        print(
            f"{name}: return={metrics['return']:.2f}%, "
            f"drawdown={metrics['max_drawdown']:.2f}%, "
            f"trades={metrics['transactions']}"
        )
    print("Long-horizon comparison:")
    for name, metrics in report["long_horizon_results"].items():
        print(
            f"{name}: return={metrics['return']:.2f}%, "
            f"drawdown={metrics['max_drawdown']:.2f}%, "
            f"trades={metrics['transactions']}"
        )
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
