"""Strict walk-forward backtesting using daily candles stored in PostgreSQL."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import select

from app.core.config import (
    BTC_SMA200_DEFENSIVE_STRATEGY,
    BOLLINGER_MAS,
    BOLLINGER_TOLS,
    BUY_PCTS,
    BUY_STAS,
    CRYPTO_STRATEGIES,
    CURS,
    EMA_LENGTHS,
    KDJ_OVERBOUGHT_THRESHOLDS,
    KDJ_OVERSOLD_THRESHOLDS,
    MA_LENGTHS,
    RSI_OVERBOUGHT_THRESHOLDS,
    RSI_OVERSOLD_THRESHOLDS,
    RSI_PERIODS,
    SELL_PCTS,
    SELL_STAS,
    TOL_PCTS,
    crypto_strategies_for_asset,
)
from app.db.database import HistoricalData, SessionLocal
from app.trading.trader_driver import TraderDriver


@dataclass(frozen=True)
class WalkForwardFold:
    """Half-open row-index boundaries for a strict walk-forward fold."""

    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    purge_start: int
    purge_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class Candidate:
    strategy: str
    tol_pct: float
    buy_pct: float
    sell_pct: float
    bollinger_sigma: float
    rsi_period: int | None = None
    rsi_oversold: float | None = None
    rsi_overbought: float | None = None
    kdj_oversold: float | None = None
    kdj_overbought: float | None = None
    sma200_entry_band_pct: float = 0.0
    sma200_exit_band_pct: float = 0.0
    sma200_min_hold_days: int = 0


def build_walk_forward_folds(
    data_length: int,
    train_days: int = 365,
    validation_days: int = 90,
    test_days: int = 30,
    purge_days: int = 7,
    step_days: int | None = None,
) -> list[WalkForwardFold]:
    """Build rolling folds with non-overlapping tests and a validation/test purge gap."""
    if min(data_length, train_days, validation_days, test_days) <= 0:
        raise ValueError("data_length and window sizes must be positive")
    if purge_days < 0:
        raise ValueError("purge_days cannot be negative")

    step = test_days if step_days is None else step_days
    if step < test_days:
        raise ValueError(
            "step_days must be >= test_days so test windows do not overlap"
        )

    required = train_days + validation_days + purge_days + test_days
    folds = []
    offset = 0
    while offset + required <= data_length:
        train_end = offset + train_days
        validation_end = train_end + validation_days
        purge_end = validation_end + purge_days
        folds.append(
            WalkForwardFold(
                fold=len(folds) + 1,
                train_start=offset,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                purge_start=validation_end,
                purge_end=purge_end,
                test_start=purge_end,
                test_end=purge_end + test_days,
            )
        )
        offset += step
    return folds


def compound_returns(returns_pct: Iterable[float]) -> float:
    """Compound percentage returns and return the result as a percentage."""
    growth = math.prod(1.0 + value / 100.0 for value in returns_pct)
    return (growth - 1.0) * 100.0


def load_daily_data(symbol: str) -> list[list[Any]]:
    """Load every stored daily candle for a symbol in chronological order."""
    cache_key = symbol.upper()
    if not cache_key.endswith("USDT"):
        cache_key = f"{cache_key}USDT"
    cache_key = f"{cache_key}__1d"

    with SessionLocal() as session:
        records = session.scalars(
            select(HistoricalData)
            .where(HistoricalData.symbol == cache_key)
            .order_by(HistoricalData.date.asc())
        ).all()

    return [
        [
            row.close_price,
            row.date.strftime("%Y-%m-%d %H:%M:%S"),
            row.open_price,
            row.low_price,
            row.high_price,
            row.volume,
        ]
        for row in records
    ]


def _candidate_from_trader(trader: Any) -> Candidate:
    return Candidate(
        strategy=trader.high_strategy,
        tol_pct=float(trader.tol_pct),
        buy_pct=float(trader.buy_pct),
        sell_pct=float(trader.sell_pct),
        bollinger_sigma=float(trader.bollinger_sigma),
        rsi_period=getattr(trader, "rsi_period", None),
        rsi_oversold=getattr(trader, "rsi_oversold", None),
        rsi_overbought=getattr(trader, "rsi_overbought", None),
        kdj_oversold=getattr(trader, "kdj_oversold", None),
        kdj_overbought=getattr(trader, "kdj_overbought", None),
        sma200_entry_band_pct=float(
            getattr(trader, "sma200_entry_band_pct", 0.0)
        ),
        sma200_exit_band_pct=float(
            getattr(trader, "sma200_exit_band_pct", 0.0)
        ),
        sma200_min_hold_days=int(getattr(trader, "sma200_min_hold_days", 0)),
    )


def _metrics(trader: Any, data: Sequence[Sequence[Any]]) -> dict[str, float | int]:
    strategy_return = float(trader.rate_of_return)
    first_close = float(data[0][0])
    last_close = float(data[-1][0])
    buy_hold_return = (last_close / first_close - 1.0) * 100.0
    return {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": strategy_return - buy_hold_return,
        "max_drawdown": float(trader.max_drawdown) * 100.0,
        "transactions": int(trader.num_transaction),
    }


def _driver_kwargs(
    name: str, candidate: Candidate | None = None, slippage_bps: float = 10.0
) -> dict[str, Any]:
    return {
        "name": name,
        "init_amount": 10_000,
        "cur_coin": 0.0,
        "overall_stats": [candidate.strategy] if candidate else CRYPTO_STRATEGIES,
        "tol_pcts": [candidate.tol_pct] if candidate else TOL_PCTS,
        "ma_lengths": MA_LENGTHS,
        "ema_lengths": EMA_LENGTHS,
        "bollinger_mas": BOLLINGER_MAS,
        "bollinger_tols": [candidate.bollinger_sigma] if candidate else BOLLINGER_TOLS,
        "buy_pcts": [candidate.buy_pct] if candidate else BUY_PCTS,
        "sell_pcts": [candidate.sell_pct] if candidate else SELL_PCTS,
        "buy_stas": BUY_STAS,
        "sell_stas": SELL_STAS,
        "rsi_periods": (
            [candidate.rsi_period]
            if candidate and candidate.rsi_period
            else RSI_PERIODS
        ),
        "rsi_oversold_thresholds": (
            [candidate.rsi_oversold]
            if candidate and candidate.rsi_oversold is not None
            else RSI_OVERSOLD_THRESHOLDS
        ),
        "rsi_overbought_thresholds": (
            [candidate.rsi_overbought]
            if candidate and candidate.rsi_overbought is not None
            else RSI_OVERBOUGHT_THRESHOLDS
        ),
        "kdj_oversold_thresholds": (
            [candidate.kdj_oversold]
            if candidate and candidate.kdj_oversold is not None
            else KDJ_OVERSOLD_THRESHOLDS
        ),
        "kdj_overbought_thresholds": (
            [candidate.kdj_overbought]
            if candidate and candidate.kdj_overbought is not None
            else KDJ_OVERBOUGHT_THRESHOLDS
        ),
        "mode": "normal",
        "execute_on_next_open": True,
        "slippage_bps": slippage_bps,
        "enable_options": False,
        "sma200_variants": (
            [
                {
                    "entry_band_pct": candidate.sma200_entry_band_pct,
                    "exit_band_pct": candidate.sma200_exit_band_pct,
                    "min_hold_days": candidate.sma200_min_hold_days,
                }
            ]
            if candidate
            and candidate.strategy in {"SMA200", BTC_SMA200_DEFENSIVE_STRATEGY}
            else None
        ),
    }


def _train_candidates(
    name: str,
    data: Sequence[Sequence[Any]],
    top_k: int,
    slippage_bps: float,
) -> list[tuple[Candidate, dict[str, float | int]]]:
    driver = TraderDriver(**_driver_kwargs(name, slippage_bps=slippage_bps))
    driver.feed_data(list(data))
    ranked = [
        (_candidate_from_trader(trader), _metrics(trader, data))
        for trader in driver.traders
    ]
    ranked.sort(
        key=lambda item: (item[1]["excess_return"], item[1]["strategy_return"]),
        reverse=True,
    )
    return ranked[:top_k]


def _evaluate_candidate(
    name: str,
    candidate: Candidate,
    data: Sequence[Sequence[Any]],
    warmup_data: Sequence[Sequence[Any]] | None = None,
    slippage_bps: float = 10.0,
) -> dict[str, float | int]:
    warmup_data = warmup_data or []
    combined_data = [*warmup_data, *data]
    driver = TraderDriver(**_driver_kwargs(name, candidate, slippage_bps))
    driver.feed_data(combined_data, warmup_points=len(warmup_data))
    return _metrics(driver.traders[0], data)


def run_symbol_walk_forward(
    symbol: str,
    data: Sequence[Sequence[Any]],
    train_days: int = 365,
    validation_days: int = 90,
    test_days: int = 30,
    purge_days: int = 7,
    top_k: int = 10,
    warmup_days: int = 200,
    slippage_bps: float = 10.0,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Tune only on train/validation data, then evaluate frozen parameters on test data."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup_days < 0:
        raise ValueError("warmup_days cannot be negative")

    folds = build_walk_forward_folds(
        len(data), train_days, validation_days, test_days, purge_days
    )
    if max_folds is not None:
        folds = folds[:max_folds]
    if not folds:
        raise ValueError(
            f"Insufficient data for walk-forward windows: {len(data)} rows"
        )

    fold_results = []
    for fold in folds:
        train_data = data[fold.train_start : fold.train_end]
        validation_data = data[fold.validation_start : fold.validation_end]
        test_data = data[fold.test_start : fold.test_end]
        validation_warmup = data[
            max(
                fold.validation_start - warmup_days, fold.train_start
            ) : fold.validation_start
        ]
        test_warmup = data[
            max(fold.test_start - warmup_days, fold.train_start) : fold.test_start
        ]

        training_candidates = _train_candidates(symbol, train_data, top_k, slippage_bps)
        validation_results = []
        for candidate, train_metrics in training_candidates:
            validation_metrics = _evaluate_candidate(
                symbol,
                candidate,
                validation_data,
                validation_warmup,
                slippage_bps,
            )
            validation_results.append((candidate, train_metrics, validation_metrics))
        candidate, train_metrics, validation_metrics = max(
            validation_results,
            key=lambda item: (
                item[2]["excess_return"],
                item[2]["strategy_return"],
                item[1]["excess_return"],
            ),
        )

        test_metrics = _evaluate_candidate(
            symbol, candidate, test_data, test_warmup, slippage_bps
        )
        fold_results.append(
            {
                "fold": fold.fold,
                "train": {
                    "start": train_data[0][1],
                    "end": train_data[-1][1],
                    **train_metrics,
                },
                "validation": {
                    "start": validation_data[0][1],
                    "end": validation_data[-1][1],
                    **validation_metrics,
                },
                "purge": {
                    "start": data[fold.purge_start][1] if purge_days else None,
                    "end": data[fold.purge_end - 1][1] if purge_days else None,
                    "days": purge_days,
                },
                "indicator_warmup_days": warmup_days,
                "test": {
                    "start": test_data[0][1],
                    "end": test_data[-1][1],
                    **test_metrics,
                },
                "frozen_parameters": asdict(candidate),
            }
        )
        print(
            f"{symbol} fold {fold.fold}/{len(folds)}: "
            f"test={test_data[0][1][:10]}..{test_data[-1][1][:10]}, "
            f"strategy={test_metrics['strategy_return']:.2f}%, "
            f"buy_hold={test_metrics['buy_hold_return']:.2f}%, "
            f"excess={test_metrics['excess_return']:.2f}%"
        )

    strategy_returns = [item["test"]["strategy_return"] for item in fold_results]
    benchmark_returns = [item["test"]["buy_hold_return"] for item in fold_results]
    excess_returns = [item["test"]["excess_return"] for item in fold_results]
    compounded_strategy = compound_returns(strategy_returns)
    compounded_benchmark = compound_returns(benchmark_returns)
    return {
        "symbol": symbol,
        "data_start": data[0][1],
        "data_end": data[-1][1],
        "data_rows": len(data),
        "folds": fold_results,
        "summary": {
            "fold_count": len(fold_results),
            "compounded_strategy_return": compounded_strategy,
            "compounded_buy_hold_return": compounded_benchmark,
            "compounded_excess_return": compounded_strategy - compounded_benchmark,
            "mean_excess_return": sum(excess_returns) / len(excess_returns),
            "excess_win_rate": sum(value > 0 for value in excess_returns)
            / len(excess_returns),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict daily walk-forward backtest")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[symbol for symbol in CURS if crypto_strategies_for_asset(symbol)],
    )
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--purge-days", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup-days", type=int, default=200)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        data = load_daily_data(symbol)
        results.append(
            run_symbol_walk_forward(
                symbol=symbol.upper(),
                data=data,
                train_days=args.train_days,
                validation_days=args.validation_days,
                test_days=args.test_days,
                purge_days=args.purge_days,
                top_k=args.top_k,
                warmup_days=args.warmup_days,
                slippage_bps=args.slippage_bps,
                max_folds=args.max_folds,
            )
        )

    output = args.output or Path("artifacts/backtests") / (
        f"walk_forward_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
