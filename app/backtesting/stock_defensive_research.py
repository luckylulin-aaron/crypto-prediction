"""Research a small, auditable set of long/cash stock strategies."""

from __future__ import annotations

import argparse
import itertools
from typing import Any, Callable, Sequence

import numpy as np

from app.backtesting.benchmark_validation import simulate_target_strategy
from app.backtesting.tencent_defensive_validation import load_stock_daily_data
from app.core.config import STOCK_SLIPPAGE_BPS


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def _sma_targets(data: Sequence[Sequence[Any]], window: int, band: float) -> np.ndarray:
    closes = np.asarray([float(row[0]) for row in data])
    average = _rolling_mean(closes, window)
    targets = np.ones(len(closes))
    previous = 1.0
    for index in range(window - 1, len(closes)):
        if closes[index] > average[index] * (1.0 + band):
            previous = 1.0
        elif closes[index] < average[index] * (1.0 - band):
            previous = 0.0
        targets[index] = previous
    return targets


def _dual_ma_targets(
    data: Sequence[Sequence[Any]], short: int, long: int, band: float
) -> np.ndarray:
    closes = np.asarray([float(row[0]) for row in data])
    short_ma = _rolling_mean(closes, short)
    long_ma = _rolling_mean(closes, long)
    targets = np.ones(len(closes))
    previous = 1.0
    for index in range(long - 1, len(closes)):
        ratio = short_ma[index] / long_ma[index] - 1.0
        if ratio > band:
            previous = 1.0
        elif ratio < -band:
            previous = 0.0
        targets[index] = previous
    return targets


def _breakout_targets(
    data: Sequence[Sequence[Any]], lookback: int, trailing_stop: float
) -> np.ndarray:
    closes = np.asarray([float(row[0]) for row in data])
    targets = np.ones(len(closes))
    in_position = True
    peak = closes[0]
    for index in range(1, len(closes)):
        close = closes[index]
        if in_position:
            peak = max(peak, close)
            if close < peak * (1.0 - trailing_stop):
                in_position = False
        elif index >= lookback and close > np.max(closes[index - lookback : index]):
            in_position = True
            peak = close
        targets[index] = float(in_position)
    return targets


def _donchian_targets(
    data: Sequence[Sequence[Any]], entry: int, exit_: int
) -> np.ndarray:
    closes = np.asarray([float(row[0]) for row in data])
    targets = np.ones(len(closes))
    in_position = True
    for index in range(1, len(closes)):
        if in_position and index >= exit_:
            if closes[index] < np.min(closes[index - exit_ : index]):
                in_position = False
        elif not in_position and index >= entry:
            if closes[index] > np.max(closes[index - entry : index]):
                in_position = True
        targets[index] = float(in_position)
    return targets


def _candidate_targets(
    data: Sequence[Sequence[Any]],
) -> list[tuple[str, dict, np.ndarray]]:
    candidates = []
    for window, band in itertools.product(
        (20, 50, 100, 150, 200), (0.0, 0.02, 0.05, 0.10, 0.15)
    ):
        params = {"window": window, "band": band}
        candidates.append(("sma", params, _sma_targets(data, **params)))
    for short, long in (
        (20, 50),
        (20, 100),
        (50, 100),
        (50, 150),
        (50, 200),
        (100, 200),
    ):
        for band in (0.0, 0.02, 0.05):
            params = {"short": short, "long": long, "band": band}
            candidates.append(("dual_ma", params, _dual_ma_targets(data, **params)))
    for lookback, trailing_stop in itertools.product(
        (20, 30, 60, 90, 120), (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    ):
        params = {"lookback": lookback, "trailing_stop": trailing_stop}
        candidates.append(("breakout", params, _breakout_targets(data, **params)))
    for entry, exit_ in itertools.product((20, 50, 100, 150), (10, 20, 50, 100)):
        if exit_ <= entry:
            params = {"entry": entry, "exit_": exit_}
            candidates.append(("donchian", params, _donchian_targets(data, **params)))
    return candidates


def _compare(data, targets, start, end):
    strategy = simulate_target_strategy(
        data, targets, start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    baseline = simulate_target_strategy(
        data, np.ones(len(data)), start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    return {
        "return": strategy["return"],
        "excess": strategy["return"] - baseline["return"],
        "drawdown": strategy["max_drawdown"],
        "transactions": strategy["transactions"],
        "baseline_return": baseline["return"],
        "baseline_drawdown": baseline["max_drawdown"],
    }


def research(symbol: str) -> list[dict[str, Any]]:
    data = load_stock_daily_data(symbol)
    results = []
    for family, parameters, targets in _candidate_targets(data):
        selection = _compare(data, targets, 1, 455)
        validation = _compare(data, targets, 365, 455)
        test = _compare(data, targets, 462, len(data))
        full = _compare(data, targets, 1, len(data))
        results.append(
            {
                "family": family,
                "parameters": parameters,
                "selection": selection,
                "validation": validation,
                "test": test,
                "full": full,
            }
        )
    results.sort(
        key=lambda item: (
            min(item["selection"]["excess"], item["validation"]["excess"]),
            item["selection"]["excess"],
            -item["validation"]["transactions"],
        ),
        reverse=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+", choices=("COIN", "MSFT"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    for symbol in args.symbols:
        print(f"\n{symbol}")
        for result in research(symbol)[: args.top]:
            print(result)


if __name__ == "__main__":
    main()
