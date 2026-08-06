"""Validate the frozen NFLX monthly SMA100 strategy against same-cost buy-and-hold."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.backtesting.aapl_defensive_research import run_research
from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    simulate_target_strategy,
)
from app.backtesting.tencent_defensive_validation import load_stock_daily_data
from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    EMA_LENGTHS,
    MA_LENGTHS,
    NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS,
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
    SELL_STAS,
    STOCK_SLIPPAGE_BPS,
)
from app.trading.trader_driver import TraderDriver

VALIDATION_DAYS = 252
PURGE_DAYS = 10
FINAL_TEST_DAYS = 750
MINIMUM_ANNUALIZED_RETURN_PCT = 10.0


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def build_nflx_monthly_sma_targets(
    data: Sequence[Sequence[Any]],
) -> np.ndarray:
    """Bootstrap invested and review a buffered SMA only once per calendar month."""
    params = NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS
    window_days = int(params["window_days"])
    band_pct = float(params["band_pct"])
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    average = _rolling_mean(closes, window_days)
    targets = np.ones(len(closes))
    active = True
    previous_month = str(data[0][1])[:7]

    for index in range(1, len(closes)):
        month = str(data[index][1])[:7]
        if index >= window_days - 1 and month != previous_month:
            if closes[index] > average[index] * (1.0 + band_pct):
                active = True
            elif closes[index] < average[index] * (1.0 - band_pct):
                active = False
        targets[index] = float(active)
        previous_month = month
    return targets


def _annualized_return(return_pct: float, start: str, end: str) -> float:
    elapsed_days = (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days
    if elapsed_days <= 0:
        raise ValueError("Annualized return requires a positive calendar span")
    years = elapsed_days / 365.25
    return ((1.0 + return_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def _compare(data, targets, start: int, end: int) -> dict[str, Any]:
    strategy = simulate_target_strategy(
        data, targets, start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    baseline = simulate_target_strategy(
        data, np.ones(len(data)), start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    start_date = str(data[start][1])[:10]
    end_date = str(data[end - 1][1])[:10]
    return {
        "start": start_date,
        "end": end_date,
        "strategy": {
            **strategy,
            "annualized_return": _annualized_return(
                float(strategy["return"]), start_date, end_date
            ),
        },
        "buy_and_hold": {
            **baseline,
            "annualized_return": _annualized_return(
                float(baseline["return"]), start_date, end_date
            ),
        },
        "excess_return": float(strategy["return"]) - float(baseline["return"]),
    }


def _registered_runtime(
    data,
    *,
    warmup_points: int = 0,
    performance_start_index: int = 0,
) -> dict[str, Any]:
    driver = TraderDriver(
        name="NFLX",
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY],
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
    )
    driver.feed_data(
        data,
        warmup_points=warmup_points,
        performance_start_index=performance_start_index,
    )
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


def run_validation() -> dict[str, Any]:
    data = load_stock_daily_data("NFLX")
    test_start = len(data) - FINAL_TEST_DAYS
    validation_end = test_start - PURGE_DAYS
    train_end = validation_end - VALIDATION_DAYS
    if not 200 < train_end < validation_end < test_start < len(data):
        raise ValueError("NFLX history is too short for the configured split")

    selection = run_research("NFLX", reveal_test=False)
    frozen_candidate = {
        "family": "monthly_sma",
        "parameters": {
            "window_days": int(NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS["window_days"]),
            "band_pct": float(NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS["band_pct"]),
        },
    }
    if selection["selected"]["candidate"] != frozen_candidate:
        raise AssertionError(
            "Frozen NFLX parameters no longer match pre-test selection"
        )

    targets = build_nflx_monthly_sma_targets(data)
    periods = {
        "train": _compare(data, targets, 1, train_end),
        "validation": _compare(data, targets, train_end, validation_end),
        "test": _compare(data, targets, test_start, len(data)),
        "full": _compare(data, targets, 1, len(data)),
    }
    runtime = _registered_runtime(data)
    runtime["excess_return"] = (
        runtime["return"] - periods["full"]["buy_and_hold"]["return"]
    )
    full_runtime_matches = bool(
        abs(runtime["return"] - periods["full"]["strategy"]["return"]) < 1e-9
        and runtime["transactions"] == periods["full"]["strategy"]["transactions"]
    )
    test_runtime = _registered_runtime(
        data,
        warmup_points=test_start - 1,
        performance_start_index=test_start,
    )
    test_runtime["excess_return"] = (
        test_runtime["return"] - periods["test"]["buy_and_hold"]["return"]
    )
    test_runtime_matches = bool(
        abs(test_runtime["return"] - periods["test"]["strategy"]["return"]) < 1e-9
        and test_runtime["transactions"] == periods["test"]["strategy"]["transactions"]
    )
    test_objective = bool(
        periods["test"]["excess_return"] > 0.0
        or periods["test"]["strategy"]["annualized_return"]
        >= MINIMUM_ANNUALIZED_RETURN_PCT
    )

    return {
        "configuration": {
            "asset": "NFLX",
            "strategy": NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
            "parameters": NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS,
            "data_start": data[0][1],
            "data_end": data[-1][1],
            "data_rows": len(data),
            "train_end": data[train_end - 1][1],
            "validation_end": data[validation_end - 1][1],
            "purge_days": PURGE_DAYS,
            "test_start": data[test_start][1],
            "candidate_count": selection["configuration"]["candidate_count"],
            "selection_rule": selection["configuration"]["selection_rule"],
            "execution_friction_rate": EXECUTION_FRICTION_RATE,
            "slippage_bps": STOCK_SLIPPAGE_BPS,
            "execution": "prior close signal, next stock-session open fill",
            "profit_objective": (
                "final-test excess return > 0 or annualized return >= 10%"
            ),
        },
        "selection": selection["selected"],
        "selection_evidence": selection["selection_evidence"],
        **periods,
        "registered_runtime": runtime,
        "registered_test_runtime": test_runtime,
        "full_runtime_matches_offline": full_runtime_matches,
        "test_runtime_matches_offline": test_runtime_matches,
        "runtime_matches_offline": full_runtime_matches and test_runtime_matches,
        "passes_return_objective": test_objective,
        "passes_profit_gate": bool(
            test_objective and full_runtime_matches and test_runtime_matches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/backtests/nflx_monthly_sma100_defensive_validation.json"
        ),
    )
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{report['configuration']['strategy']}: "
        f"gate={report['passes_profit_gate']}, "
        f"runtime_match={report['runtime_matches_offline']}"
    )
    for split in ("train", "validation", "test", "full"):
        period = report[split]
        print(
            f"  {split}: strategy={period['strategy']['return']:.2f}%, "
            f"cagr={period['strategy']['annualized_return']:.2f}%, "
            f"buy_hold={period['buy_and_hold']['return']:.2f}%, "
            f"excess={period['excess_return']:.2f} pp"
        )


if __name__ == "__main__":
    main()
