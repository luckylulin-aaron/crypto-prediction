#!/usr/bin/env python3
"""
Test script for MACD (Moving Average Convergence Divergence) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_macd


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
    trader.strat_dct = {"MACD": []}

    # Mock MACD data
    trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]
    trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    return trader


class TestMACDStrategy(unittest.TestCase):
    """Test cases for MACD strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0

    def test_buy_signal_on_bullish_crossover_in_bullish_regime(self):
        """Buy on bullish histogram crossover (<=0 -> >0) in bullish regime."""
        # hist_prev = 0.1 - 0.2 = -0.1, hist_cur = 0.3 - 0.2 = 0.1 (cross up)
        self.trader.macd_diff = [0.1, 0.3]
        self.trader.macd_dea = [0.2, 0.2]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertTrue(buy, "Should execute buy on bullish crossover")
        self.assertFalse(
            sell, "Should not execute sell on bullish crossover"
        )
        self.trader._execute_one_buy.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, BUY_SIGNAL
        )

    def test_sell_signal_on_bearish_crossover_when_holding_position(self):
        """Sell on bearish histogram crossover (>=0 -> <0) when holding a position."""
        # hist_prev = 0.3 - 0.2 = 0.1, hist_cur = 0.1 - 0.2 = -0.1 (cross down)
        self.trader.macd_diff = [0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertFalse(buy, "Should not execute buy on bearish crossover")
        self.assertTrue(sell, "Should execute sell on bearish crossover when holding a position")
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, SELL_SIGNAL
        )

    def test_no_action_when_no_crossover(self):
        """No action when there is no histogram crossover."""
        # hist stays positive (no cross)
        self.trader.macd_diff = [0.3, 0.31]
        self.trader.macd_dea = [0.2, 0.2]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertFalse(buy, "Should not execute buy without a crossover")
        self.assertFalse(sell, "Should not execute sell without a crossover")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_no_action_when_macd_dea_none(self):
        """Test no action when MACD DEA is None."""
        # Set up None MACD DEA
        self.trader.macd_diff = [0.1, 0.3]
        self.trader.macd_dea = [0.2, None]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertFalse(buy, "Should not execute buy when MACD DEA is None")
        self.assertFalse(sell, "Should not execute sell when MACD DEA is None")

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False

        # Set up bullish crossover
        self.trader.macd_diff = [0.1, 0.3]
        self.trader.macd_dea = [0.2, 0.2]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        # Set up bearish crossover
        self.trader.macd_diff = [0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2]

        buy, sell = strategy_macd(self.trader, self.new_price, self.today)

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn("MACD", STRATEGY_REGISTRY, "MACD not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["MACD"]), "MACD not callable")


if __name__ == "__main__":
    unittest.main()
