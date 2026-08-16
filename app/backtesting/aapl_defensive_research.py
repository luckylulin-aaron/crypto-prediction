"""Select one explainable AAPL long/cash rule without inspecting final-test results."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from app.backtesting.benchmark_validation import simulate_target_strategy
from app.backtesting.tencent_defensive_validation import load_stock_daily_data
from app.core.config import STOCK_SLIPPAGE_BPS


VALIDATION_DAYS = 252
PURGE_DAYS = 10
FINAL_TEST_DAYS = 750


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, float | int]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def build_breakout_targets(
    data: Sequence[Sequence[Any]], lookback_days: int, trailing_stop_pct: float
) -> np.ndarray:
    """Bootstrap invested, trail the peak, then re-enter above a prior high."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    targets = np.ones(len(closes))
    in_position = True
    peak = closes[0]
    for index in range(1, len(closes)):
        close = closes[index]
        if in_position:
            peak = max(peak, close)
            if close < peak * (1.0 - trailing_stop_pct):
                in_position = False
        elif index >= lookback_days:
            prior_high = float(np.max(closes[index - lookback_days : index]))
            if close > prior_high:
                in_position = True
                peak = close
        targets[index] = float(in_position)
    return targets


def build_sma_band_targets(
    data: Sequence[Sequence[Any]], window_days: int, band_pct: float
) -> np.ndarray:
    """Bootstrap invested and switch only beyond a symmetric SMA band."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    average = _rolling_mean(closes, window_days)
    targets = np.ones(len(closes))
    active = True
    for index in range(window_days - 1, len(closes)):
        if closes[index] > average[index] * (1.0 + band_pct):
            active = True
        elif closes[index] < average[index] * (1.0 - band_pct):
            active = False
        targets[index] = float(active)
    return targets


def build_sma_exit_breakout_reentry_targets(
    data: Sequence[Sequence[Any]],
    window_days: int,
    exit_band_pct: float,
    reentry_days: int,
) -> np.ndarray:
    """Exit below a long trend band, but re-enter promptly on a shorter high."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    average = _rolling_mean(closes, window_days)
    targets = np.ones(len(closes))
    active = True
    for index in range(window_days - 1, len(closes)):
        if active:
            if closes[index] < average[index] * (1.0 - exit_band_pct):
                active = False
        elif index >= reentry_days:
            prior_high = float(np.max(closes[index - reentry_days : index]))
            if closes[index] > prior_high:
                active = True
        targets[index] = float(active)
    return targets


def build_monthly_sma_targets(
    data: Sequence[Sequence[Any]], window_days: int, band_pct: float
) -> np.ndarray:
    """Evaluate a symmetric SMA band only on the first stock session each month."""
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


def build_donchian_targets(
    data: Sequence[Sequence[Any]], entry_days: int, exit_days: int
) -> np.ndarray:
    """Bootstrap invested; exit below a prior low and re-enter above a prior high."""
    closes = np.asarray([float(row[0]) for row in data], dtype=float)
    targets = np.ones(len(closes))
    in_position = True
    for index in range(1, len(closes)):
        if in_position and index >= exit_days:
            if close_below_prior_low(closes, index, exit_days):
                in_position = False
        elif not in_position and index >= entry_days:
            prior_high = float(np.max(closes[index - entry_days : index]))
            if closes[index] > prior_high:
                in_position = True
        targets[index] = float(in_position)
    return targets


def close_below_prior_low(closes: np.ndarray, index: int, window: int) -> bool:
    return bool(closes[index] < float(np.min(closes[index - window : index])))


def candidates() -> list[Candidate]:
    """A deliberately small, auditable candidate set."""
    result = [
        Candidate(
            "breakout",
            {"lookback_days": lookback, "trailing_stop_pct": stop},
        )
        for lookback in (20, 60, 120)
        for stop in (0.08, 0.12, 0.16, 0.20)
    ]
    result.extend(
        Candidate("sma_band", {"window_days": window, "band_pct": band})
        for window in (100, 150, 200)
        for band in (0.0, 0.03, 0.05)
    )
    result.extend(
        Candidate("donchian", {"entry_days": entry, "exit_days": exit_})
        for entry, exit_ in ((20, 10), (60, 20), (120, 20), (120, 60))
    )
    result.extend(
        Candidate(
            "sma_exit_breakout_reentry",
            {
                "window_days": 200,
                "exit_band_pct": exit_band,
                "reentry_days": reentry,
            },
        )
        for exit_band in (0.0, 0.03, 0.05)
        for reentry in (10, 20, 60)
    )
    result.extend(
        Candidate("monthly_sma", {"window_days": window, "band_pct": band})
        for window in (100, 150, 200)
        for band in (0.0, 0.03, 0.05)
    )

    return result


def build_targets(data: Sequence[Sequence[Any]], candidate: Candidate) -> np.ndarray:
    if candidate.family == "breakout":
        return build_breakout_targets(data, **candidate.parameters)
    if candidate.family == "sma_band":
        return build_sma_band_targets(data, **candidate.parameters)
    if candidate.family == "donchian":
        return build_donchian_targets(data, **candidate.parameters)
    if candidate.family == "sma_exit_breakout_reentry":
        return build_sma_exit_breakout_reentry_targets(data, **candidate.parameters)
    if candidate.family == "monthly_sma":
        return build_monthly_sma_targets(data, **candidate.parameters)
    raise ValueError(f"Unsupported AAPL candidate family: {candidate.family}")


def compare_period(data, targets, start: int, end: int) -> dict[str, Any]:
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


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float, int]:
    train_excess = float(result["train"]["excess_return"])
    validation_excess = float(result["validation"]["excess_return"])
    return (
        min(train_excess, validation_excess),
        validation_excess,
        train_excess,
        -int(result["validation"]["strategy"]["transactions"]),
    )


def run_research(symbol: str = "AAPL", reveal_test: bool = True) -> dict[str, Any]:
    data = load_stock_daily_data(symbol)
    test_start = len(data) - FINAL_TEST_DAYS
    validation_end = test_start - PURGE_DAYS
    train_end = validation_end - VALIDATION_DAYS
    if not 200 < train_end < validation_end < test_start < len(data):
        raise ValueError("AAPL history is too short for the configured split")

    evaluated = []
    for candidate in candidates():
        targets = build_targets(data, candidate)
        evaluated.append(
            {
                "candidate": asdict(candidate),
                "train": compare_period(data, targets, 1, train_end),
                "validation": compare_period(data, targets, train_end, validation_end),
            }
        )

    selected = max(evaluated, key=_selection_score)
    frozen = Candidate(**selected["candidate"])
    frozen_targets = build_targets(data, frozen)
    selected_report = dict(selected)
    if reveal_test:
        selected_report["test"] = compare_period(
            data, frozen_targets, test_start, len(data)
        )
        selected_report["full"] = compare_period(data, frozen_targets, 1, len(data))

    return {
        "configuration": {
            "asset": symbol,
            "data_start": data[0][1],
            "data_end": data[-1][1],
            "data_rows": len(data),
            "train_end": data[train_end - 1][1],
            "validation_end": data[validation_end - 1][1],
            "purge_days": PURGE_DAYS,
            "test_start": data[test_start][1],
            "candidate_count": len(evaluated),
            "selection_rule": "maximize minimum train/validation excess on long pre-test history",
        },
        "selected": selected_report,
        "selection_evidence": sorted(evaluated, key=_selection_score, reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()
    report = run_research(reveal_test=not args.no_test)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
