from datetime import datetime, timedelta

import pytest

from app.core.config import BUY_SIGNAL, SELL_SIGNAL, SUPPORTED_STRATEGIES
from app.trading.strat_trader import StratTrader
from app.trading.strategies import STRATEGY_REGISTRY


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

    assert "SMA200" in SUPPORTED_STRATEGIES
    assert STRATEGY_REGISTRY["SMA200"] is not None
    assert "200" in trader.moving_averages
    assert trader.buy_pct == 1.0
    assert trader.sell_pct == 1.0


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
