import datetime

import pytest

from app.backtesting.portfolio_strategy_validation import load_aligned_assets
from app.core.config import (
    BTC_SMA200_DEFENSIVE_STRATEGY,
    CRYPTO_STRATEGIES,
    CURS,
    ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
    SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
    crypto_strategies_for_asset,
)
from app.trading.strategies import STRATEGY_REGISTRY
from app.trading.trader_driver import TraderDriver


def _driver(name, strategy, btc_data_stream=None):
    return TraderDriver(
        name=name,
        init_amount=10_000,
        cur_coin=0.0,
        overall_stats=[strategy],
        tol_pcts=[0.1],
        ma_lengths=[6],
        ema_lengths=[6],
        bollinger_mas=[6],
        bollinger_tols=[2],
        buy_pcts=[0.5],
        sell_pcts=[0.5],
        btc_data_stream=btc_data_stream,
        enable_options=False,
    )


def _stream(closes, opens=None):
    opens = closes if opens is None else opens
    start = datetime.datetime(2024, 1, 1)
    return [
        (close, start + datetime.timedelta(days=index), open_, close, close)
        for index, (close, open_) in enumerate(zip(closes, opens))
    ]


def test_new_strategies_are_enabled_with_strict_asset_isolation():
    assert ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY in STRATEGY_REGISTRY
    assert SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY in STRATEGY_REGISTRY
    assert CRYPTO_STRATEGIES == [
        BTC_SMA200_DEFENSIVE_STRATEGY,
        ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
        SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
    ]
    assert crypto_strategies_for_asset("BTC") == [BTC_SMA200_DEFENSIVE_STRATEGY]
    assert crypto_strategies_for_asset("ETH") == [ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY]
    assert crypto_strategies_for_asset("SOL") == [SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY]
    assert crypto_strategies_for_asset("UNI") == []
    assert [asset for asset in CURS if crypto_strategies_for_asset(asset)] == [
        "ETH",
        "BTC",
        "SOL",
    ]


@pytest.mark.parametrize(
    ("asset", "strategy"),
    [
        ("BTC", ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY),
        ("ETH", SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY),
    ],
)
def test_fixed_strategies_reject_the_wrong_asset(asset, strategy):
    with pytest.raises(ValueError, match="not allowed"):
        _driver(asset, strategy)


def test_eth_breakout_executes_on_the_next_open_with_fixed_costs():
    closes = [100.0] * 120 + [101.0, 101.0]
    opens = [100.0] * 121 + [102.0]
    driver = _driver("ETH", ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY)

    driver.feed_data(_stream(closes, opens))

    trader = driver.traders[0]
    buys = [item for item in trader.trade_history if item["action"] == "BUY"]
    assert len(buys) == 1
    assert buys[0]["date"] == datetime.datetime(2024, 5, 1)
    assert buys[0]["price"] == pytest.approx(102.0 * 1.001)
    assert trader.cur_coin == pytest.approx(10_000 * 0.98 / (102.0 * 1.001))


def test_sol_strategy_requires_an_aligned_btc_stream():
    driver = _driver("SOL", SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY)

    with pytest.raises(ValueError, match="requires a BTC daily data stream"):
        driver.feed_data(_stream([10.0] * 32))


def test_sol_breakout_only_enters_when_btc_defensive_regime_is_active():
    sol = [10.0] * 200 + [11.0, 11.0]
    sol_opens = [10.0] * 201 + [12.0]
    btc = [100.0] * 199 + [110.0, 110.0, 110.0]
    btc_stream = _stream(btc)
    driver = _driver(
        "SOL", SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY, btc_data_stream=btc_stream
    )

    driver.feed_data(_stream(sol, sol_opens))

    trader = driver.traders[0]
    buys = [item for item in trader.trade_history if item["action"] == "BUY"]
    assert len(buys) == 1
    assert buys[0]["price"] == pytest.approx(12.0 * 1.001)


def test_registered_runtime_strategies_are_profitable_on_local_three_year_data():
    _, data = load_aligned_assets(["ETH", "BTC", "SOL"])
    eth_driver = _driver("ETH", ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY)
    sol_driver = _driver(
        "SOL",
        SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
        btc_data_stream=data["BTC"],
    )

    eth_driver.feed_data(data["ETH"])
    sol_driver.feed_data(data["SOL"])

    eth = eth_driver.traders[0]
    sol = sol_driver.traders[0]
    assert eth.rate_of_return == pytest.approx(21.716, abs=0.001)
    assert len(eth.all_history_trade_only) == 16
    assert sol.rate_of_return == pytest.approx(32.548, abs=0.001)
    assert len(sol.all_history_trade_only) == 20
