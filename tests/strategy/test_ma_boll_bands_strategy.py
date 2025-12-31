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
        """Bull regime: buy when price touches the MA."""
        # Uptrend: last_ma > prev_ma
        # Provide 7 points so the strategy can compute bands excluding today's MA.
        self.trader.moving_averages["6"] = [100, 101, 102, 103, 104, 105, 106]
        new_price = 105.0  # Touch yesterday MA (strategy avoids lookahead)

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
        """Bull regime: sell when price exceeds upper Bollinger band."""
        self.trader.moving_averages["6"] = [100, 101, 102, 103, 104, 105, 106]
        new_price = 106.5  # Exceeds upper band for this sequence with sigma=2

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
        self.assertTrue(sell)
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, SELL_SIGNAL)

    def test_bear_sell_on_ma_touch(self):
        """Bear regime: sell when price touches the MA."""
        # Downtrend: last_ma < prev_ma
        self.trader.moving_averages["6"] = [106, 105, 104, 103, 102, 101, 100]
        new_price = 101.0  # Touch yesterday MA (strategy avoids lookahead)

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
        self.assertTrue(sell)
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, SELL_SIGNAL)

    def test_bear_buy_below_lower_band(self):
        """Bear regime: buy when price goes below lower Bollinger band."""
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
        """Neutral regime: no action when MA is flat."""
        self.trader.moving_averages["6"] = [100, 100, 100, 100, 100, 100]
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


