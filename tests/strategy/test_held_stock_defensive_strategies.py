from datetime import datetime, timedelta

import pytest

from app.core.config import (
    BOLLINGER_MAS,
    BUY_STAS,
    COIN_BTC_SMA200_DEFENSIVE_PARAMETERS,
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
    EMA_LENGTHS,
    MA_LENGTHS,
    MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS,
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
    SELL_STAS,
    STOCK_EXECUTE_ON_NEXT_OPEN,
    STOCK_SLIPPAGE_BPS,
    STOCKS,
    stock_strategies_for_asset,
)
from app.trading.strategies import STRATEGY_REGISTRY
from app.trading.trader_driver import TraderDriver


def _driver(symbol: str, strategy: str, btc_data_stream=None) -> TraderDriver:
    return TraderDriver(
        name=symbol,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[strategy],
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
        btc_data_stream=btc_data_stream,
    )


def _rows(closes, start=datetime(2024, 1, 1), opens=None):
    opens = closes if opens is None else opens
    return [
        [
            float(close),
            (start + timedelta(days=index)).strftime("%Y-%m-%d"),
            float(opens[index]),
            float(min(close, opens[index]) - 1.0),
            float(max(close, opens[index]) + 1.0),
            1_000.0,
        ]
        for index, close in enumerate(closes)
    ]


def test_held_stock_strategies_are_registered_enabled_and_asset_isolated():
    assert STRATEGY_REGISTRY[COIN_BTC_SMA200_DEFENSIVE_STRATEGY] is not None
    assert STRATEGY_REGISTRY[MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY] is not None
    assert stock_strategies_for_asset("COIN") == [COIN_BTC_SMA200_DEFENSIVE_STRATEGY]
    assert stock_strategies_for_asset("MSFT") == [MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY]
    assert "COIN" in STOCKS

    with pytest.raises(ValueError, match="restricted to COIN"):
        _driver("MSFT", COIN_BTC_SMA200_DEFENSIVE_STRATEGY)
    with pytest.raises(ValueError, match="restricted to MSFT"):
        _driver("COIN", MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY)


@pytest.mark.parametrize(
    ("symbol", "strategy"),
    [
        ("COIN", COIN_BTC_SMA200_DEFENSIVE_STRATEGY),
        ("MSFT", MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY),
    ],
)
def test_held_stock_strategies_force_one_full_allocation_candidate(symbol, strategy):
    driver = _driver(symbol, strategy)
    trader = driver.traders[0]

    assert len(driver.traders) == 1
    assert trader.buy_pct == pytest.approx(1.0)
    assert trader.sell_pct == pytest.approx(1.0)
    assert trader.brokerage_pct == pytest.approx(0.02)
    assert trader.execute_on_next_open is STOCK_EXECUTE_ON_NEXT_OPEN
    assert trader.slippage_bps == pytest.approx(STOCK_SLIPPAGE_BPS)


def test_coin_requires_context_and_uses_only_the_completed_prior_btc_day():
    stock_rows = _rows([100.0, 101.0], start=datetime(2024, 7, 19))
    with pytest.raises(ValueError, match="requires a BTC daily data stream"):
        _driver("COIN", COIN_BTC_SMA200_DEFENSIVE_STRATEGY).feed_data(stock_rows)

    # Day 200 turns the BTC state off; day 201 turns it back on. COIN day 201
    # must still see day 200 (off), and only schedules a buy on COIN day 202.
    btc_closes = [100.0] * 199 + [90.0, 120.0, 120.0]
    btc_rows = _rows(btc_closes)
    stock_rows = _rows([100.0, 101.0], start=datetime(2024, 7, 19))
    driver = _driver(
        "COIN", COIN_BTC_SMA200_DEFENSIVE_STRATEGY, btc_data_stream=btc_rows
    )
    driver.feed_data(stock_rows)
    trader = driver.traders[0]

    assert trader.market_context["btc_defensive_ready"] is True
    assert trader.market_context["btc_defensive_active"] is True
    assert trader.all_history_trade_only == []
    assert trader.pending_order == {"action": "BUY", "method": "by_percentage"}
    assert trader.sma200_entry_band_pct == pytest.approx(
        COIN_BTC_SMA200_DEFENSIVE_PARAMETERS["entry_band_pct"]
    )


def test_msft_bootstrap_buy_executes_at_next_open_with_cost_and_slippage():
    driver = _driver("MSFT", MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY)
    rows = _rows([100.0, 101.0], opens=[100.0, 110.0])
    driver.feed_data(rows)
    trader = driver.traders[0]

    execution_price = 110.0 * (1.0 + STOCK_SLIPPAGE_BPS / 10_000.0)
    assert trader.cur_coin == pytest.approx(10_000 * 0.98 / execution_price)
    assert trader.cash == pytest.approx(0.0)
    assert trader.all_history_trade_only[0]["price"] == pytest.approx(execution_price)
    assert (
        trader.breakout_lookback_days
        == MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS["lookback_days"]
    )
    assert trader.breakout_trailing_stop_pct == pytest.approx(
        MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS["trailing_stop_pct"]
    )
