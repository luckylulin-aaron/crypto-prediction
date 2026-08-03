from datetime import datetime, timedelta

import pytest

from app.backtesting.walk_forward import build_walk_forward_folds, compound_returns
from app.trading.strat_trader import StratTrader


def test_folds_have_purge_gap_and_non_overlapping_tests():
    folds = build_walk_forward_folds(
        data_length=600,
        train_days=180,
        validation_days=60,
        test_days=30,
        purge_days=7,
    )

    assert folds
    assert folds[0].train_start == 0
    assert folds[0].train_end == folds[0].validation_start
    assert folds[0].validation_end == folds[0].purge_start
    assert folds[0].purge_end == folds[0].test_start
    assert folds[0].purge_end - folds[0].purge_start == 7
    assert all(
        left.test_end <= right.test_start for left, right in zip(folds, folds[1:])
    )


def test_overlapping_test_windows_are_rejected():
    with pytest.raises(ValueError, match="do not overlap"):
        build_walk_forward_folds(
            data_length=600,
            train_days=180,
            validation_days=60,
            test_days=30,
            purge_days=7,
            step_days=15,
        )


def test_compound_returns():
    assert compound_returns([10.0, -10.0]) == pytest.approx(-1.0)


def test_indicator_warmup_does_not_trade():
    trader = StratTrader(
        name="BTC",
        init_amount=10_000,
        stat="MA-BOLL-BANDS",
        tol_pct=0.1,
        ma_lengths=[6, 12, 30],
        ema_lengths=[6, 12, 26, 30],
        bollinger_mas=[6, 12],
        bollinger_sigma=2,
        buy_pct=0.5,
        sell_pct=0.5,
    )

    for day in range(40):
        price = 100 + day
        trader.add_new_day(
            price,
            datetime(2024, 1, 1) + timedelta(days=day),
            {"open": price, "low": price - 1, "high": price + 1, "volume": 1000},
            execute_strategy=False,
        )

    assert len(trader.crypto_prices) == 40
    assert trader.trade_history == []
    assert trader.cash == 10_000
    assert trader.cur_coin == 0
