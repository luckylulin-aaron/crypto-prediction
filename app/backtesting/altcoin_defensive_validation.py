"""Validate independent low-frequency defensive strategies for ETH and SOL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import (
    ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
    SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
)
from app.trading.trader_driver import TraderDriver

from app.backtesting.portfolio_strategy_validation import (
    _feature_matrices,
    _hysteresis_states,
    load_aligned_assets,
    simulate_portfolio,
)


@dataclass(frozen=True)
class BreakoutParameters:
    lookback_days: int
    trailing_stop_pct: float
    require_btc_regime: bool
    require_sma200: bool


def _single_asset_targets(
    asset_index: int, active: np.ndarray, asset_count: int
) -> np.ndarray:
    targets = np.zeros((len(active), asset_count), dtype=float)
    targets[:, asset_index] = active.astype(float)
    return targets


def build_btc_regime_follower_targets(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    traded_symbol: str,
) -> np.ndarray:
    """Hold the selected asset only while BTC is in its fixed defensive regime."""
    btc_index = symbols.index("BTC")
    traded_index = symbols.index(traded_symbol)
    btc_active = _hysteresis_states(features["closes"], features["sma200"], 0.05, 0.05)[
        :, btc_index
    ]
    return _single_asset_targets(traded_index, btc_active, len(symbols))


def build_breakout_targets(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    traded_symbol: str,
    params: BreakoutParameters,
) -> np.ndarray:
    """Enter prior-high breakouts and exit via a trailing stop or regime failure."""
    closes = features["closes"]
    sma200 = features["sma200"]
    asset = symbols.index(traded_symbol)
    btc = symbols.index("BTC")
    btc_active = _hysteresis_states(closes, sma200, 0.05, 0.05)[:, btc]
    active = np.zeros(len(closes), dtype=bool)
    in_position = False
    peak = 0.0

    for index in range(len(closes)):
        close = closes[index, asset]
        own_sma = sma200[index, asset]
        btc_ok = not params.require_btc_regime or btc_active[index]
        own_trend_ok = not params.require_sma200 or (
            not np.isnan(own_sma) and close > own_sma
        )

        if in_position:
            peak = max(peak, close)
            stopped = close < peak * (1.0 - params.trailing_stop_pct)
            regime_failed = params.require_btc_regime and not btc_active[index]
            trend_failed = params.require_sma200 and (
                np.isnan(own_sma) or close < own_sma * 0.95
            )
            if stopped or regime_failed or trend_failed:
                in_position = False
                peak = 0.0

        if not in_position and index >= params.lookback_days:
            prior_high = float(
                np.max(closes[index - params.lookback_days : index, asset])
            )
            if close > prior_high and btc_ok and own_trend_ok:
                in_position = True
                peak = close

        active[index] = in_position

    return _single_asset_targets(asset, active, len(symbols))


def _strategy_results(
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    dates: Sequence[str],
    train_end: int,
    validation_end: int,
    test_start: int,
    slippage_bps: float,
) -> dict[str, Any]:
    results = {
        "train": simulate_portfolio(features, targets, 200, train_end, slippage_bps),
        "validation": simulate_portfolio(
            features, targets, train_end, validation_end, slippage_bps
        ),
        "test": simulate_portfolio(
            features, targets, test_start, len(dates), slippage_bps
        ),
        "full": simulate_portfolio(features, targets, 200, len(dates), slippage_bps),
    }
    results["passes_profit_gate"] = bool(
        results["full"]["return"] > 0.0
        and results["test"]["return"] > 0.0
        and results["validation"]["return"] >= 0.0
    )
    return results


def _buy_and_hold_results(
    features: dict[str, np.ndarray],
    symbols: Sequence[str],
    symbol: str,
    dates: Sequence[str],
    test_start: int,
    slippage_bps: float,
) -> dict[str, Any]:
    active = np.ones(len(dates), dtype=bool)
    targets = _single_asset_targets(symbols.index(symbol), active, len(symbols))
    return {
        "test": simulate_portfolio(
            features, targets, test_start, len(dates), slippage_bps
        ),
        "full": simulate_portfolio(features, targets, 200, len(dates), slippage_bps),
    }


def _reset_segments(
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    dates: Sequence[str],
    slippage_bps: float,
    segment_days: int = 180,
) -> list[dict[str, Any]]:
    segments = []
    for start in range(200, len(dates), segment_days):
        end = min(start + segment_days, len(dates))
        if end <= start + 1:
            continue
        segments.append(
            {
                "start": dates[start],
                "end": dates[end - 1],
                **simulate_portfolio(features, targets, start, end, slippage_bps),
            }
        )
    return segments


def _registered_runtime_result(
    symbol: str,
    strategy: str,
    data_by_asset: dict[str, list],
) -> dict[str, Any]:
    driver = TraderDriver(
        name=symbol,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[strategy],
        tol_pcts=[0.1],
        ma_lengths=[6],
        ema_lengths=[6],
        bollinger_mas=[6],
        bollinger_tols=[2],
        buy_pcts=[1.0],
        sell_pcts=[1.0],
        enable_options=False,
        btc_data_stream=(data_by_asset["BTC"] if symbol == "SOL" else None),
    )
    driver.feed_data(data_by_asset[symbol])
    trader = driver.traders[0]
    trades = trader.all_history_trade_only
    return {
        "return": float(trader.rate_of_return),
        "max_drawdown": float(trader.max_drawdown) * 100.0,
        "transactions": len(trades),
        "final_value": float(trader.portfolio_value),
        "data_start": str(data_by_asset[symbol][0][1]),
        "data_end": str(data_by_asset[symbol][-1][1]),
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
    symbols: Sequence[str],
    train_end: int = 365,
    validation_end: int = 455,
    purge_days: int = 7,
    slippage_bps: float = 10.0,
) -> dict[str, Any]:
    """Validate frozen ETH and SOL rules without tuning on the test period."""
    dates, data_by_asset = load_aligned_assets(symbols)
    features = _feature_matrices(symbols, data_by_asset)
    test_start = validation_end + purge_days
    if len(dates) <= test_start:
        raise ValueError("Not enough aligned daily data")

    strategies = {
        "eth_120d_breakout_defensive": (
            ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
            "ETH",
            BreakoutParameters(120, 0.05, False, False),
        ),
        "sol_30d_breakout_defensive": (
            SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
            "SOL",
            BreakoutParameters(30, 0.10, True, False),
        ),
    }
    results: dict[str, Any] = {}
    for key, (registered_name, symbol, parameters) in strategies.items():
        targets = build_breakout_targets(features, symbols, symbol, parameters)
        result = {
            "registered_name": registered_name,
            "asset": symbol,
            "parameters": asdict(parameters),
            "registered_runtime": _registered_runtime_result(
                symbol, registered_name, data_by_asset
            ),
            **_strategy_results(
                features,
                targets,
                dates,
                train_end,
                validation_end,
                test_start,
                slippage_bps,
            ),
            "buy_and_hold": _buy_and_hold_results(
                features, symbols, symbol, dates, test_start, slippage_bps
            ),
            "reset_180_day_segments": _reset_segments(
                features, targets, dates, slippage_bps
            ),
        }
        result["passes_profit_gate"] = bool(
            result["passes_profit_gate"]
            and result["registered_runtime"]["return"] > 0.0
        )
        results[key] = result

    return {
        "configuration": {
            "symbols": list(symbols),
            "data_start": dates[0],
            "data_end": dates[-1],
            "tradable_start_after_200_day_warmup": dates[200],
            "train": {"start": dates[200], "end": dates[train_end - 1]},
            "validation": {
                "start": dates[train_end],
                "end": dates[validation_end - 1],
            },
            "purge_days": purge_days,
            "test": {"start": dates[test_start], "end": dates[-1]},
            "execution_friction_rate": 0.02,
            "slippage_bps": slippage_bps,
            "execution": "prior daily close signal, next daily open fill",
            "profit_gate": (
                "validation >= 0, test > 0, offline full > 0, "
                "and registered runtime full > 0"
            ),
        },
        **results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ETH", "BTC", "SOL", "UNI", "LTC", "ETC", "DOGE", "AAVE"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/backtests/altcoin_defensive_validation.json"),
    )
    args = parser.parse_args()

    report = run_validation(args.symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in (
        "eth_120d_breakout_defensive",
        "sol_30d_breakout_defensive",
    ):
        result = report[name]
        print(name, result.get("parameters", {}))
        runtime = result["registered_runtime"]
        print(
            f"  registered runtime: return={runtime['return']:.2f}%, "
            f"drawdown={runtime['max_drawdown']:.2f}%, "
            f"trades={runtime['transactions']}"
        )
        for split in ("train", "validation", "test", "full"):
            metrics = result[split]
            print(
                f"  {split}: return={metrics['return']:.2f}%, "
                f"drawdown={metrics['max_drawdown']:.2f}%, "
                f"trades={metrics['transactions']}"
            )
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
