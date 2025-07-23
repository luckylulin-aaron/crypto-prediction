#!/usr/bin/env python3
"""
Test script for DOUBLE-MA (Double Moving Average Crossover) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_double_moving_averages


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()

    # Mock wallet
    trader.wallet = {"USD": 10000, "crypto": 100}

    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()

    # Mock strategy dictionary
    trader.strat_dct = {"DOUBLE-MA": []}

    # Mock moving averages for short and long periods
    trader.moving_averages = {
        "10": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],  # Short-term MA
        "20": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],  # Long-term MA
    }

    return trader


class TestDoubleMAStrategy(unittest.TestCase):
    """Test cases for DOUBLE-MA strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.shorter_queue_name = "10"
        self.longer_queue_name = "20"
        self.new_price = 100.0

    def test_golden_cross_buy_signal(self):
        """Test buy signal when golden cross occurs (short MA > long MA)."""
        # Set up golden cross: short MA (105) > long MA (104)
        self.trader.moving_averages["10"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        self.trader.moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 104.0]

        buy, sell = strategy_double_moving_averages(
            self.trader,
            self.shorter_queue_name,
            self.longer_queue_name,
            self.new_price,
            self.today,
        )

        self.assertTrue(buy, "Should execute buy when golden cross occurs")
        self.assertFalse(sell, "Should not execute sell when golden cross occurs")
        self.trader._execute_one_buy.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, BUY_SIGNAL
        )

    def test_death_cross_sell_signal(self):
        """Test sell signal when death cross occurs (long MA > short MA)."""
        # Set up death cross: long MA (105) > short MA (104)
        self.trader.moving_averages["10"] = [100.0, 101.0, 102.0, 103.0, 104.0, 104.0]
        self.trader.moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

        buy, sell = strategy_double_moving_averages(
            self.trader,
            self.shorter_queue_name,
            self.longer_queue_name,
            self.new_price,
            self.today,
        )

        self.assertFalse(buy, "Should not execute buy when death cross occurs")
        self.assertTrue(sell, "Should execute sell when death cross occurs")
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, SELL_SIGNAL
        )

    def test_no_action_when_ma_equal(self):
        """Test no action when moving averages are equal."""
        # Set up equal MAs
        self.trader.moving_averages["10"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        self.trader.moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

        buy, sell = strategy_double_moving_averages(
            self.trader,
            self.shorter_queue_name,
            self.longer_queue_name,
            self.new_price,
            self.today,
        )

        self.assertFalse(buy, "Should not execute buy when MAs are equal")
        self.assertFalse(sell, "Should not execute sell when MAs are equal")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False

        # Set up golden cross
        self.trader.moving_averages["10"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        self.trader.moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 104.0]

        buy, sell = strategy_double_moving_averages(
            self.trader,
            self.shorter_queue_name,
            self.longer_queue_name,
            self.new_price,
            self.today,
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        # Set up death cross
        self.trader.moving_averages["10"] = [100.0, 101.0, 102.0, 103.0, 104.0, 104.0]
        self.trader.moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

        buy, sell = strategy_double_moving_averages(
            self.trader,
            self.shorter_queue_name,
            self.longer_queue_name,
            self.new_price,
            self.today,
        )

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn("DOUBLE-MA", STRATEGY_REGISTRY, "DOUBLE-MA not in registry")
        self.assertTrue(
            callable(STRATEGY_REGISTRY["DOUBLE-MA"]), "DOUBLE-MA not callable"
        )


if __name__ == "__main__":
    unittest.main()
