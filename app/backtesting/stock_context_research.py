"""Research pre-specified market-context gates for held stock positions."""

from __future__ import annotations

import numpy as np

from app.backtesting.benchmark_validation import simulate_target_strategy
from app.backtesting.stock_defensive_research import (
    _breakout_targets,
    _dual_ma_targets,
)
from app.backtesting.tencent_defensive_validation import load_stock_daily_data
from app.core.config import STOCK_SLIPPAGE_BPS


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window) / window, mode="valid"
        )
    return result


def _aligned_closes(stock_data, context_data) -> np.ndarray:
    context_by_date = {str(row[1])[:10]: float(row[0]) for row in context_data}
    missing = [row[1] for row in stock_data if str(row[1])[:10] not in context_by_date]
    if missing:
        raise ValueError(f"Context is missing {len(missing)} stock dates")
    return np.asarray(
        [context_by_date[str(row[1])[:10]] for row in stock_data], dtype=float
    )


def _hysteresis_targets(
    closes: np.ndarray,
    window: int = 200,
    entry_band: float = 0.05,
    exit_band: float = 0.05,
) -> np.ndarray:
    average = _rolling_mean(closes, window)
    targets = np.ones(len(closes))
    active = True
    for index in range(window - 1, len(closes)):
        if closes[index] > average[index] * (1.0 + entry_band):
            active = True
        elif closes[index] < average[index] * (1.0 - exit_band):
            active = False
        targets[index] = float(active)
    return targets


def _calendar_context_targets(stock_data, context_data) -> np.ndarray:
    """Compute the 200-calendar-day context state, then align it to stock dates."""
    context_closes = np.asarray([float(row[0]) for row in context_data], dtype=float)
    context_targets = _hysteresis_targets(context_closes)
    target_by_date = {
        str(row[1])[:10]: float(context_targets[index])
        for index, row in enumerate(context_data)
    }
    # A completed prior-calendar-day BTC candle is always known before a US stock
    # close. The COIN signal then executes at the following stock-session open.
    targets = []
    for row in stock_data:
        stock_date = np.datetime64(str(row[1])[:10])
        prior_date = str(stock_date - np.timedelta64(1, "D"))
        targets.append(target_by_date.get(prior_date, 1.0))
    return np.asarray(targets, dtype=float)


def _metrics(data, targets, start, end):
    result = simulate_target_strategy(
        data, targets, start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    baseline = simulate_target_strategy(
        data, np.ones(len(data)), start, end, slippage_bps=STOCK_SLIPPAGE_BPS
    )
    return {
        "return": result["return"],
        "baseline": baseline["return"],
        "excess": result["return"] - baseline["return"],
        "drawdown": result["max_drawdown"],
        "baseline_drawdown": baseline["max_drawdown"],
        "transactions": result["transactions"],
    }


def main() -> None:
    coin = load_stock_daily_data("COIN")
    btc = load_stock_daily_data("BTC-USD")
    btc_targets = _calendar_context_targets(coin, btc)
    coin_targets = _dual_ma_targets(coin, short=100, long=200, band=0.05)

    msft = load_stock_daily_data("MSFT")
    msft_targets = _breakout_targets(msft, lookback=20, trailing_stop=0.10)

    candidates = {
        "COIN-BTC-SMA200": (coin, btc_targets),
        "COIN-DUAL-AND-BTC": (coin, np.minimum(coin_targets, btc_targets)),
        "MSFT-BREAKOUT20-STOP10": (msft, msft_targets),
    }
    for name, (data, targets) in candidates.items():
        print(f"\n{name}")
        for label, start, end in (
            ("selection", 1, 455),
            ("validation", 365, 455),
            ("test", 462, len(data)),
            ("full", 1, len(data)),
        ):
            print(label, _metrics(data, targets, start, end))


if __name__ == "__main__":
    main()
