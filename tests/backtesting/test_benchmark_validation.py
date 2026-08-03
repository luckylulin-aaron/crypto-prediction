from datetime import datetime

import pytest

from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    build_strategy_targets,
    simulate_target_strategy,
)
from app.trading.strat_trader import StratTrader
from app.trading.strategies import apply_signal_option_leverage


def _row(close, day, open_price=None):
    open_price = close if open_price is None else open_price
    return [
        close,
        f"2024-01-{day:02d} 00:00:00",
        open_price,
        close - 1,
        close + 1,
        1000,
    ]


def _trader(**overrides):
    kwargs = {
        "name": "BTC",
        "init_amount": 10_000,
        "stat": "MA-BOLL-BANDS",
        "tol_pct": 0.1,
        "ma_lengths": [6, 12, 30],
        "ema_lengths": [6, 12, 26, 30],
        "bollinger_mas": [6, 12],
        "bollinger_sigma": 2,
        "buy_pct": 0.5,
        "sell_pct": 0.5,
    }
    kwargs.update(overrides)
    return StratTrader(**kwargs)


def test_target_strategy_executes_previous_signal_at_next_open():
    data = [_row(100, 1), _row(110, 2, open_price=100)]
    result = simulate_target_strategy(
        data,
        targets=[1.0, 1.0],
        test_start=1,
        test_end=2,
        slippage_bps=10,
    )

    quantity = 10_000 * (1 - EXECUTION_FRICTION_RATE) / 100.1
    expected_return = (quantity * 110 / 10_000 - 1) * 100
    assert result["return"] == pytest.approx(expected_return)
    assert result["transactions"] == 1


def test_strat_trader_defers_signal_until_next_open():
    trader = _trader(execute_on_next_open=True, slippage_bps=10)

    assert trader._execute_one_buy("by_percentage", 100) is True
    assert trader.cur_coin == 0
    assert trader.pending_order["action"] == "BUY"

    trader.add_new_day(
        new_p=110,
        d=datetime(2024, 1, 2),
        misc_p={"open": 100, "low": 99, "high": 111, "volume": 1000},
    )

    expected_quantity = 5_000 * (1 - EXECUTION_FRICTION_RATE) / 100.1
    assert trader.cur_coin == pytest.approx(expected_quantity)
    assert trader.trade_history[0]["action"] == "BUY"
    assert trader.trade_history[0]["price"] == pytest.approx(100.1)


def test_options_can_be_disabled_for_backtests():
    trader = _trader(enable_options=False)
    cash_before = trader.cash

    result = apply_signal_option_leverage(
        trader=trader,
        signal="BUY",
        price=100,
        today=datetime(2024, 1, 1),
    )

    assert result is None
    assert trader.cash == cash_before
    assert trader.option_history == []


def test_regime_targets_use_shared_fixed_rules():
    data = []
    for index in range(260):
        price = 100 + index * 0.5
        data.append(
            [price, f"2024-01-01 00:00:{index:02d}", price, price - 1, price + 1, 1000]
        )

    features = build_strategy_targets(data)

    assert features["regimes"][-1] == "BULL"
    assert features["targets"]["regime_switch"][-1] == 1.0
    assert set(features["targets"]) == {
        "cash",
        "buy_hold",
        "sma200_trend",
        "dual_ma_50_200",
        "volatility_target",
        "regime_switch",
    }
