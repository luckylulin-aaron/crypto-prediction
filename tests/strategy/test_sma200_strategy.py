from datetime import datetime, timedelta

import pytest

from app.core.config import (
    BUY_SIGNAL,
    SELL_SIGNAL,
    SMA200_VARIANTS,
    SUPPORTED_STRATEGIES,
)
from app.trading.strat_trader import StratTrader
from app.trading.strategies import STRATEGY_REGISTRY
from app.trading.trader_driver import TraderDriver


def _trader(**overrides):
    kwargs = {
        "name": "BTC",
        "init_amount": 10_000,
        "stat": "SMA200",
        "tol_pct": 0.1,
        "ma_lengths": [6, 12, 30],
        "ema_lengths": [6, 12, 26, 30],
        "bollinger_mas": [6, 12],
        "bollinger_sigma": 2,
        "buy_pct": 0.25,
        "sell_pct": 0.25,
    }
    kwargs.update(overrides)
    return StratTrader(**kwargs)


def _add_day(trader, index, close, open_price=None, execute_strategy=True):
    day = datetime(2024, 1, 1) + timedelta(days=index)
    open_price = close if open_price is None else open_price
    trader.add_new_day(
        new_p=close,
        d=day,
        misc_p={
            "open": open_price,
            "low": min(open_price, close) - 1,
            "high": max(open_price, close) + 1,
            "volume": 1000,
        },
        execute_strategy=execute_strategy,
    )


def test_sma200_is_registered_and_adds_required_average():
    trader = _trader()

    assert SMA200_VARIANTS == [
        {"entry_band_pct": 0.05, "exit_band_pct": 0.05, "min_hold_days": 0}
    ]
    assert "SMA200" in SUPPORTED_STRATEGIES
    assert STRATEGY_REGISTRY["SMA200"] is not None
    assert "200" in trader.moving_averages
    assert trader.buy_pct == 1.0
    assert trader.sell_pct == 1.0


def test_trader_driver_creates_one_fixed_sma200_candidate():
    grid = {
        "overall_stats": ["MA-BOLL-BANDS", "SMA200"],
        "tol_pcts": [0.1, 0.2],
        "buy_pcts": [0.5, 1.0],
        "sell_pcts": [0.5, 1.0],
        "bollinger_tols": [2, 3],
        "rsi_periods": [14],
        "rsi_oversold_thresholds": [30],
        "rsi_overbought_thresholds": [70],
        "kdj_oversold_thresholds": [20],
        "kdj_overbought_thresholds": [80],
        "sma200_variants": [SMA200_VARIANTS[0]],
    }

    specs = list(TraderDriver._iter_trader_specs(**grid))
    sma_specs = [spec for spec in specs if spec["stat"] == "SMA200"]

    assert len(specs) == 17
    assert len(sma_specs) == 1
    assert TraderDriver.expected_trader_count(**grid) == 17


def test_sma200_goes_all_in_above_average_and_exits_below_it():
    trader = _trader()
    for index in range(199):
        _add_day(trader, index, 100, execute_strategy=False)

    _add_day(trader, 199, 110)

    assert trader.moving_averages["200"][-1] == pytest.approx(100.05)
    assert trader.cash == 0
    assert trader.cur_coin == pytest.approx(10_000 * 0.98 / 110)
    assert trader.trade_history[-1]["action"] == BUY_SIGNAL

    _add_day(trader, 200, 90)

    assert trader.cur_coin == 0
    assert trader.trade_history[-1]["action"] == SELL_SIGNAL


def test_sma200_confirmation_band_filters_small_crosses():
    trader = _trader(
        sma200_entry_band_pct=0.03,
        sma200_exit_band_pct=0.03,
    )
    for index in range(199):
        _add_day(trader, index, 100, execute_strategy=False)

    _add_day(trader, 199, 102)
    assert trader.cur_coin == 0

    _add_day(trader, 200, 104)
    assert trader.cur_coin > 0


def test_sma200_minimum_holding_period_delays_exit():
    trader = _trader(sma200_min_hold_days=14)
    for index in range(199):
        _add_day(trader, index, 100, execute_strategy=False)

    _add_day(trader, 199, 110)
    for index in range(200, 213):
        _add_day(trader, index, 90)
        assert trader.cur_coin > 0

    _add_day(trader, 213, 90)
    assert trader.cur_coin == 0


def test_sma200_signal_executes_at_next_open_when_enabled():
    trader = _trader(execute_on_next_open=True, slippage_bps=10)
    for index in range(199):
        _add_day(trader, index, 100, execute_strategy=False)

    _add_day(trader, 199, 110)

    assert trader.cur_coin == 0
    assert trader.pending_order["action"] == BUY_SIGNAL

    _add_day(trader, 200, 110, open_price=105)

    expected_quantity = 10_000 * 0.98 / 105.105
    assert trader.cash == 0
    assert trader.cur_coin == pytest.approx(expected_quantity)
    buy_trade = next(
        item for item in trader.trade_history if item["action"] == BUY_SIGNAL
    )
    assert buy_trade["price"] == pytest.approx(105.105)
