"""Cross-asset, leakage-resistant validation for a small SMA200 variant set."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from app.backtesting.walk_forward import (
    Candidate,
    _evaluate_candidate,
    load_daily_data,
)
from app.core.config import SMA200_VARIANTS

SMA200_BASELINE_VARIANT = {
    "entry_band_pct": 0.0,
    "exit_band_pct": 0.0,
    "min_hold_days": 0,
}


def _sma_candidate(variant: dict[str, Any]) -> Candidate:
    return Candidate(
        strategy="SMA200",
        tol_pct=0.0,
        buy_pct=1.0,
        sell_pct=1.0,
        bollinger_sigma=2.0,
        sma200_entry_band_pct=float(variant["entry_band_pct"]),
        sma200_exit_band_pct=float(variant["exit_band_pct"]),
        sma200_min_hold_days=int(variant["min_hold_days"]),
    )


def _fixed_ma_boll_candidate() -> Candidate:
    return Candidate(
        strategy="MA-BOLL-BANDS",
        tol_pct=0.1,
        buy_pct=0.5,
        sell_pct=0.5,
        bollinger_sigma=2.0,
    )


def _evaluate_period(
    symbol: str,
    candidate: Candidate,
    data: Sequence[Sequence[Any]],
    start: int,
    end: int,
    warmup_days: int,
    slippage_bps: float,
) -> dict[str, float | int]:
    warmup_start = max(0, start - warmup_days)
    return _evaluate_candidate(
        symbol,
        candidate,
        data[start:end],
        data[warmup_start:start],
        slippage_bps,
    )


def _aggregate(metrics_by_asset: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    returns = [float(item["strategy_return"]) for item in metrics_by_asset.values()]
    excess = [float(item["excess_return"]) for item in metrics_by_asset.values()]
    transactions = [int(item["transactions"]) for item in metrics_by_asset.values()]
    return {
        "median_return": median(returns),
        "median_excess_return": median(excess),
        "positive_return_assets": sum(value > 0 for value in returns),
        "positive_excess_assets": sum(value > 0 for value in excess),
        "median_transactions": median(transactions),
        "asset_count": len(metrics_by_asset),
    }


def run_validation(
    symbols: Sequence[str],
    train_days: int = 365,
    validation_days: int = 90,
    purge_days: int = 7,
    warmup_days: int = 200,
    slippage_bps: float = 10.0,
) -> dict[str, Any]:
    data_by_asset = {symbol: load_daily_data(symbol) for symbol in symbols}
    minimum_rows = min(len(data) for data in data_by_asset.values())
    test_start = train_days + validation_days + purge_days
    if minimum_rows <= test_start:
        raise ValueError("Not enough data for train/validation/purge/test split")

    candidates = [_sma_candidate(variant) for variant in SMA200_VARIANTS]
    candidate_results = []
    for candidate in candidates:
        train_by_asset = {}
        validation_by_asset = {}
        for symbol, data in data_by_asset.items():
            train_by_asset[symbol] = _evaluate_period(
                symbol,
                candidate,
                data,
                0,
                train_days,
                warmup_days,
                slippage_bps,
            )
            validation_by_asset[symbol] = _evaluate_period(
                symbol,
                candidate,
                data,
                train_days,
                train_days + validation_days,
                warmup_days,
                slippage_bps,
            )
        candidate_results.append(
            {
                "candidate": candidate,
                "train": _aggregate(train_by_asset),
                "validation": _aggregate(validation_by_asset),
                "train_by_asset": train_by_asset,
                "validation_by_asset": validation_by_asset,
            }
        )

    train_shortlist = sorted(
        candidate_results,
        key=lambda item: (
            item["train"]["median_excess_return"],
            item["train"]["median_return"],
        ),
        reverse=True,
    )[:3]
    selected = max(
        train_shortlist,
        key=lambda item: (
            item["validation"]["median_excess_return"],
            item["validation"]["median_return"],
            -item["validation"]["median_transactions"],
        ),
    )["candidate"]

    baseline = _sma_candidate(SMA200_BASELINE_VARIANT)
    fixed_ma_boll = _fixed_ma_boll_candidate()
    test_results = {}
    for symbol, data in data_by_asset.items():
        end = len(data)
        test_results[symbol] = {
            "data_start": data[0][1],
            "data_end": data[-1][1],
            "data_rows": len(data),
            "test_start": data[test_start][1],
            "test_end": data[-1][1],
            "sma200_selected": _evaluate_period(
                symbol,
                selected,
                data,
                test_start,
                end,
                warmup_days,
                slippage_bps,
            ),
            "sma200_baseline": _evaluate_period(
                symbol,
                baseline,
                data,
                test_start,
                end,
                warmup_days,
                slippage_bps,
            ),
            "ma_boll_fixed": _evaluate_period(
                symbol,
                fixed_ma_boll,
                data,
                test_start,
                end,
                warmup_days,
                slippage_bps,
            ),
        }

    aggregate_test = {
        name: _aggregate(
            {symbol: result[name] for symbol, result in test_results.items()}
        )
        for name in ("sma200_selected", "sma200_baseline", "ma_boll_fixed")
    }
    return {
        "configuration": {
            "symbols": list(symbols),
            "train_days": train_days,
            "validation_days": validation_days,
            "purge_days": purge_days,
            "warmup_days": warmup_days,
            "test_days": minimum_rows - test_start,
            "execution_friction_rate": 0.02,
            "slippage_bps": slippage_bps,
            "signal_time": "daily_close",
            "execution_time": "next_daily_open",
            "options_enabled": False,
            "selection_scope": "one shared variant across all assets",
        },
        "selected_sma200_candidate": asdict(selected),
        "candidate_selection": [
            {
                "candidate": asdict(item["candidate"]),
                "train": item["train"],
                "validation": item["validation"],
                "train_by_asset": item["train_by_asset"],
                "validation_by_asset": item["validation_by_asset"],
            }
            for item in candidate_results
        ],
        "test_by_asset": test_results,
        "aggregate_test": aggregate_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_validation(args.symbols)
    output = args.output or Path("artifacts/backtests/sma200_variants_btc_eth_sol.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Selected SMA200:", report["selected_sma200_candidate"])
    for symbol, result in report["test_by_asset"].items():
        selected = result["sma200_selected"]
        baseline = result["sma200_baseline"]
        ma_boll = result["ma_boll_fixed"]
        print(
            f"{symbol}: selected={selected['strategy_return']:.2f}%, "
            f"baseline={baseline['strategy_return']:.2f}%, "
            f"ma_boll={ma_boll['strategy_return']:.2f}%"
        )
    print("Aggregate:", report["aggregate_test"])
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
