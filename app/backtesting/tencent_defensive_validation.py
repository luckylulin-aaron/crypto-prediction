"""Validate the frozen TCEHY regime-defensive strategy against buy and hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sqlalchemy import select

from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    simulate_target_strategy,
)
from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    EMA_LENGTHS,
    MA_LENGTHS,
    SELL_STAS,
    STOCK_SLIPPAGE_BPS,
    TCEHY_REGIME_DEFENSIVE_PARAMETERS,
    TCEHY_REGIME_DEFENSIVE_STRATEGY,
)
from app.db.database import HistoricalData, SessionLocal
from app.trading.trader_driver import TraderDriver


def load_stock_daily_data(symbol: str = "TCEHY") -> list[list[Any]]:
    """Load a stock's complete daily history from the local database."""
    with SessionLocal() as session:
        records = session.scalars(
            select(HistoricalData)
            .where(HistoricalData.symbol == symbol.upper())
            .order_by(HistoricalData.date.asc())
        ).all()
    if not records:
        raise ValueError(f"No local daily data found for {symbol}")
    return [
        [
            row.close_price,
            row.date.strftime("%Y-%m-%d"),
            row.open_price,
            row.low_price,
            row.high_price,
            row.volume,
        ]
        for row in records
    ]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def build_regime_targets(
    data: Sequence[Sequence[Any]], parameters: dict[str, float | int]
) -> np.ndarray:
    """Build close-derived targets that the runtime executes at the next open."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    trend_days = int(parameters["trend_ma_days"])
    slope_days = int(parameters["trend_slope_days"])
    range_days = int(parameters["range_ma_days"])
    range_sigma = float(parameters["range_sigma"])
    trend_ma = _rolling_mean(closes, trend_days)
    range_ma = _rolling_mean(closes, range_days)
    targets = np.ones(len(closes))
    previous_target = 1.0

    first_signal = trend_days + slope_days - 1
    for index in range(first_signal, len(closes)):
        slope = trend_ma[index] / trend_ma[index - slope_days] - 1.0
        if closes[index] > trend_ma[index] and slope > 0.0:
            previous_target = 1.0
        elif closes[index] < trend_ma[index] and slope < 0.0:
            previous_target = 0.0
        else:
            recent = closes[index - range_days + 1 : index + 1]
            range_std = float(np.std(recent))
            lower = range_ma[index] - range_sigma * range_std
            upper = range_ma[index] + range_sigma * range_std
            if closes[index] < lower:
                previous_target = 1.0
            elif closes[index] > upper:
                previous_target = 0.0
        targets[index] = previous_target
    return targets


def _compare_period(
    data: Sequence[Sequence[Any]],
    targets: np.ndarray,
    start: int,
    end: int,
    slippage_bps: float,
) -> dict[str, Any]:
    strategy = simulate_target_strategy(
        data, targets, start, end, slippage_bps=slippage_bps
    )
    buy_and_hold = simulate_target_strategy(
        data, np.ones(len(data)), start, end, slippage_bps=slippage_bps
    )
    return {
        "start": data[start][1],
        "end": data[end - 1][1],
        "strategy": strategy,
        "buy_and_hold": buy_and_hold,
        "excess_return": strategy["return"] - buy_and_hold["return"],
    }


def _registered_runtime(data: list[list[Any]]) -> dict[str, Any]:
    driver = TraderDriver(
        name="TCEHY",
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[TCEHY_REGIME_DEFENSIVE_STRATEGY],
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
    driver.feed_data(data)
    trader = driver.traders[0]
    trades = trader.all_history_trade_only
    return {
        "return": float(trader.rate_of_return),
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


def run_validation(
    train_end: int = 365,
    validation_end: int = 455,
    purge_days: int = 7,
    slippage_bps: float = STOCK_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Evaluate the frozen strategy without tuning on validation or test rows."""
    data = load_stock_daily_data("TCEHY")
    parameters = TCEHY_REGIME_DEFENSIVE_PARAMETERS.copy()
    targets = build_regime_targets(data, parameters)
    regime_start = int(parameters["trend_ma_days"]) + int(
        parameters["trend_slope_days"]
    )
    full_start = 1
    test_start = validation_end + purge_days
    if not regime_start < train_end < validation_end < test_start < len(data):
        raise ValueError("TCEHY history is too short for the configured split")

    periods = {
        "train": _compare_period(data, targets, regime_start, train_end, slippage_bps),
        "validation": _compare_period(
            data, targets, train_end, validation_end, slippage_bps
        ),
        "test": _compare_period(data, targets, test_start, len(data), slippage_bps),
        "full": _compare_period(data, targets, full_start, len(data), slippage_bps),
    }
    runtime = _registered_runtime(data)
    runtime["excess_return"] = (
        runtime["return"] - periods["full"]["buy_and_hold"]["return"]
    )
    passes_profit_gate = bool(
        periods["validation"]["excess_return"] >= -1e-9
        and periods["test"]["excess_return"] > 0.0
        and periods["full"]["excess_return"] > 0.0
        and runtime["excess_return"] > 0.0
        and periods["full"]["strategy"]["max_drawdown"]
        < periods["full"]["buy_and_hold"]["max_drawdown"]
    )

    return {
        "configuration": {
            "strategy": TCEHY_REGIME_DEFENSIVE_STRATEGY,
            "asset": "TCEHY",
            "parameters": parameters,
            "data_start": data[0][1],
            "data_end": data[-1][1],
            "data_rows": len(data),
            "full_start": data[full_start][1],
            "regime_start": data[regime_start][1],
            "train_end": data[train_end - 1][1],
            "validation_end": data[validation_end - 1][1],
            "purge_days": purge_days,
            "test_start": data[test_start][1],
            "execution_friction_rate": EXECUTION_FRICTION_RATE,
            "slippage_bps": slippage_bps,
            "execution": "prior daily close signal, next daily open fill",
        },
        **periods,
        "registered_runtime": runtime,
        "passes_profit_gate": passes_profit_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/backtests/tcehy_regime_defensive_validation.json"),
    )
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{TCEHY_REGIME_DEFENSIVE_STRATEGY}: "
        f"profit_gate={report['passes_profit_gate']}"
    )
    for split in ("train", "validation", "test", "full"):
        result = report[split]
        print(
            f"  {split}: strategy={result['strategy']['return']:.2f}%, "
            f"buy_hold={result['buy_and_hold']['return']:.2f}%, "
            f"excess={result['excess_return']:.2f}%"
        )
    runtime = report["registered_runtime"]
    print(
        f"  runtime: return={runtime['return']:.2f}%, "
        f"excess={runtime['excess_return']:.2f}%, "
        f"drawdown={runtime['max_drawdown']:.2f}%, "
        f"transactions={runtime['transactions']}"
    )
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
