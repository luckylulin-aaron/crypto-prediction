#!/usr/bin/env python3
"""
Test script for MA-BOLL-BANDS (MA trend + Bollinger Bands) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_ma_boll_bands


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()

    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()

    trader.strat_dct = {"MA-BOLL-BANDS": []}
    trader.moving_averages = {"6": []}
    trader.cash = 10000.0
    trader.cur_coin = 0.0

    return trader


class TestMABollBandsStrategy(unittest.TestCase):
    """Test cases for MA-BOLL-BANDS strategy."""

    def setUp(self):
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.queue_name = "6"
        self.tol_pct = 0.01
        self.buy_pct = 0.3
        self.sell_pct = 0.3
        self.bollinger_sigma = 2

    def test_bull_buy_on_ma_touch(self):
        """Volatility mode: buy when price is sufficiently below lower band (even in an uptrend)."""
        # Uptrend slope exists, but entry is driven by band extreme/z-score + bandwidth filter.
        self.trader.moving_averages["6"] = [100, 101, 102, 103, 104, 105, 106]
        new_price = 98.0  # Below lower band for this window -> should buy

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
        )

        self.assertTrue(buy)
        self.assertFalse(sell)
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, BUY_SIGNAL)

    def test_bull_sell_on_upper_band_break(self):
        """Bull mode: do not sell immediately on the first upper-band break; sell after reversal."""
        self.trader.moving_averages["6"] = [100, 101, 102, 103, 104, 105, 106]
        self.trader.cur_coin = 1.0
        self.trader.cash = 0.0
        # First spike: arms blow-off, but does not sell immediately.
        new_price = 112.0

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
            cooldown_days=0,
        )

        self.assertFalse(buy)
        self.assertFalse(sell)

        # Second day: reversal triggers trailing-stop sell.
        next_day = self.today + datetime.timedelta(days=1)
        reversal_price = 106.0  # ~5% below 112 peak -> should sell
        buy2, sell2 = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            reversal_price,
            next_day,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
            cooldown_days=0,
        )
        self.assertFalse(buy2)
        self.assertTrue(sell2)
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", reversal_price)
        self.trader._record_history.assert_called_with(reversal_price, next_day, SELL_SIGNAL)

    def test_bear_sell_on_ma_touch(self):
        """Volatility mode: no sell if we don't hold a position, even in a downtrend."""
        self.trader.moving_averages["6"] = [106, 105, 104, 103, 102, 101, 100]
        new_price = 101.0

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
        )

        self.assertFalse(buy)
        self.assertFalse(sell)
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)

    def test_bear_buy_below_lower_band(self):
        """Volatility mode: buy when price goes below lower Bollinger band."""
        self.trader.moving_averages["6"] = [106, 105, 104, 103, 102, 101, 100]
        new_price = 98.0  # Below lower band for this sequence with sigma=2

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
        )

        self.assertTrue(buy)
        self.assertFalse(sell)
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, BUY_SIGNAL)

    def test_no_action_on_flat_ma(self):
        """No action in low-volatility regime (bandwidth too small)."""
        self.trader.moving_averages["6"] = [100, 100, 100, 100, 100, 100, 100]
        new_price = 100.0

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
        )

        self.assertFalse(buy)
        self.assertFalse(sell)
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_no_action_when_insufficient_ma_for_trend(self):
        """No action when there isn't enough MA data to infer trend."""
        self.trader.moving_averages["6"] = [100.0]
        new_price = 100.0

        buy, sell = strategy_ma_boll_bands(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.bollinger_sigma,
        )

        self.assertFalse(buy)
        self.assertFalse(sell)
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )


if __name__ == "__main__":
    unittest.main()


