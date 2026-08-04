from datetime import datetime, timedelta

import pytest

from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    EMA_LENGTHS,
    MA_LENGTHS,
    SELL_STAS,
    STOCK_EXECUTE_ON_NEXT_OPEN,
    STOCK_SLIPPAGE_BPS,
    TCEHY_REGIME_DEFENSIVE_PARAMETERS,
    TCEHY_REGIME_DEFENSIVE_STRATEGY,
    stock_strategies_for_asset,
)
from app.trading.strategies import STRATEGY_REGISTRY
from app.trading.trader_driver import TraderDriver


def _driver(name: str = "TCEHY") -> TraderDriver:
    return TraderDriver(
        name=name,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[TCEHY_REGIME_DEFENSIVE_STRATEGY],
        tol_pcts=[0.1],
        ma_lengths=MA_LENGTHS,
        ema_lengths=EMA_LENGTHS,
        bollinger_mas=BOLLINGER_MAS,
        bollinger_tols=[2],
        buy_pcts=[0.5],
        sell_pcts=[0.5],
        buy_stas=BUY_STAS,
        sell_stas=SELL_STAS,
        enable_options=False,
    )


def test_tcehy_strategy_is_registered_enabled_and_asset_isolated():
    assert STRATEGY_REGISTRY[TCEHY_REGIME_DEFENSIVE_STRATEGY] is not None
    assert stock_strategies_for_asset("TCEHY") == [TCEHY_REGIME_DEFENSIVE_STRATEGY]
    assert TCEHY_REGIME_DEFENSIVE_STRATEGY not in stock_strategies_for_asset("AAPL")

    with pytest.raises(ValueError, match="restricted to TCEHY"):
        _driver("AAPL")


def test_tcehy_strategy_forces_frozen_execution_configuration():
    trader = _driver().traders[0]
    params = TCEHY_REGIME_DEFENSIVE_PARAMETERS

    assert trader.buy_pct == pytest.approx(1.0)
    assert trader.sell_pct == pytest.approx(1.0)
    assert trader.execute_on_next_open is STOCK_EXECUTE_ON_NEXT_OPEN
    assert trader.slippage_bps == pytest.approx(STOCK_SLIPPAGE_BPS)
    assert trader.regime_trend_ma_days == params["trend_ma_days"]
    assert trader.regime_trend_slope_days == params["trend_slope_days"]
    assert trader.regime_range_ma_days == params["range_ma_days"]
    assert trader.regime_range_sigma == pytest.approx(params["range_sigma"])
    assert str(params["trend_ma_days"]) in trader.moving_averages
    assert str(params["range_ma_days"]) in trader.moving_averages


def test_tcehy_bull_signal_executes_at_next_open_with_cost_and_slippage():
    driver = _driver()
    start = datetime(2023, 1, 1)
    closes = [100.0] * 200 + [float(value) for value in range(101, 121)]
    rows = []
    for index, close in enumerate(closes):
        date = (start + timedelta(days=index)).strftime("%Y-%m-%d")
        rows.append([close, date, close, close - 1.0, close + 1.0, 1_000.0])

    next_open = 115.0
    next_date = (start + timedelta(days=len(rows))).strftime("%Y-%m-%d")
    rows.append([121.0, next_date, next_open, 114.0, 122.0, 1_000.0])
    driver.feed_data(rows, warmup_points=219)

    trader = driver.traders[0]
    execution_price = next_open * (1.0 + STOCK_SLIPPAGE_BPS / 10_000.0)
    expected_quantity = 10_000 * 0.98 / execution_price
    assert trader.cur_coin == pytest.approx(expected_quantity)
    assert trader.cash == pytest.approx(0.0)
    assert trader.all_history_trade_only[0]["date"].strftime("%Y-%m-%d") == next_date
    assert trader.all_history_trade_only[0]["price"] == pytest.approx(execution_price)
