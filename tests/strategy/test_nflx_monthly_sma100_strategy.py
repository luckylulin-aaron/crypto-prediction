from datetime import datetime, timedelta

import pytest

from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    EMA_LENGTHS,
    MA_LENGTHS,
    NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS,
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
    SELL_STAS,
    STOCK_EXECUTE_ON_NEXT_OPEN,
    STOCK_SLIPPAGE_BPS,
    stock_strategies_for_asset,
)
from app.trading.strategies import STRATEGY_REGISTRY
from app.trading.trader_driver import TraderDriver


def _driver(symbol: str = "NFLX") -> TraderDriver:
    return TraderDriver(
        name=symbol,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY],
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


def _row(close: float, day: datetime, open_price: float | None = None):
    open_price = close if open_price is None else open_price
    return [
        float(close),
        day.strftime("%Y-%m-%d"),
        float(open_price),
        float(min(close, open_price) - 1.0),
        float(max(close, open_price) + 1.0),
        1_000.0,
    ]


def test_nflx_strategy_is_registered_frozen_and_asset_isolated():
    assert STRATEGY_REGISTRY[NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY] is not None
    assert stock_strategies_for_asset("NFLX") == [
        NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY
    ]
    assert NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS == {
        "window_days": 100,
        "band_pct": 0.05,
    }

    with pytest.raises(ValueError, match="restricted to NFLX"):
        _driver("MSFT")


def test_nflx_strategy_forces_one_full_allocation_next_open_candidate():
    trader = _driver().traders[0]

    assert len(_driver().traders) == 1
    assert trader.buy_pct == pytest.approx(1.0)
    assert trader.sell_pct == pytest.approx(1.0)
    assert trader.brokerage_pct == pytest.approx(0.02)
    assert trader.execute_on_next_open is STOCK_EXECUTE_ON_NEXT_OPEN
    assert trader.slippage_bps == pytest.approx(STOCK_SLIPPAGE_BPS)
    assert trader.monthly_sma_window_days == 100
    assert trader.monthly_sma_band_pct == pytest.approx(0.05)


def test_nflx_bootstrap_buy_and_monthly_exit_execute_at_following_open():
    start = datetime(2024, 1, 22)
    rows = [_row(100.0, start + timedelta(days=index)) for index in range(100)]
    rows.extend(
        [
            _row(90.0, datetime(2024, 5, 1), open_price=90.0),
            _row(90.0, datetime(2024, 5, 2), open_price=80.0),
        ]
    )

    driver = _driver()
    driver.feed_data(rows)
    trader = driver.traders[0]
    trades = trader.all_history_trade_only

    assert [trade["action"] for trade in trades] == ["BUY", "SELL"]
    expected_buy = 100.0 * (1.0 + STOCK_SLIPPAGE_BPS / 10_000.0)
    expected_sell = 80.0 * (1.0 - STOCK_SLIPPAGE_BPS / 10_000.0)
    assert trades[0]["price"] == pytest.approx(expected_buy)
    assert trades[0]["date"].strftime("%Y-%m-%d") == "2024-01-23"
    assert trades[1]["price"] == pytest.approx(expected_sell)
    assert trades[1]["date"].strftime("%Y-%m-%d") == "2024-05-02"


def test_nflx_does_not_react_to_intramonth_band_crossing():
    start = datetime(2024, 1, 22)
    rows = [_row(100.0, start + timedelta(days=index)) for index in range(100)]
    rows.append(_row(90.0, datetime(2024, 4, 30), open_price=90.0))

    driver = _driver()
    driver.feed_data(rows)
    trades = driver.traders[0].all_history_trade_only

    assert [trade["action"] for trade in trades] == ["BUY"]
