#!/usr/bin/env python3
"""
Test script for RSI (Relative Strength Index) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_rsi


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()

    # Mock wallet
    trader.wallet = {"USD": 10000, "crypto": 100}

    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()
    trader.compute_rsi = Mock(return_value=50.0)  # Default RSI value

    # Mock strategy dictionary
    trader.strat_dct = {"RSI": []}

    return trader


class TestRSIStrategy(unittest.TestCase):
    """Test cases for RSI strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0

    def test_buy_signal_when_rsi_oversold(self):
        """Test buy signal when RSI is oversold (below threshold)."""
        # Set up oversold RSI (20 < 30)
        self.trader.compute_rsi.return_value = 20.0

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertTrue(buy, "Should execute buy when RSI is oversold")
        self.assertFalse(sell, "Should not execute sell when RSI is oversold")
        self.trader._execute_one_buy.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, BUY_SIGNAL
        )

    def test_sell_signal_when_rsi_overbought(self):
        """Test sell signal when RSI is overbought (above threshold)."""
        # Set up overbought RSI (80 > 70)
        self.trader.compute_rsi.return_value = 80.0

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertFalse(buy, "Should not execute buy when RSI is overbought")
        self.assertTrue(sell, "Should execute sell when RSI is overbought")
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, SELL_SIGNAL
        )

    def test_no_action_when_rsi_neutral(self):
        """Test no action when RSI is in neutral zone."""
        # Set up neutral RSI (50 between 30 and 70)
        self.trader.compute_rsi.return_value = 50.0

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertFalse(buy, "Should not execute buy when RSI is neutral")
        self.assertFalse(sell, "Should not execute sell when RSI is neutral")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_no_action_when_rsi_none(self):
        """Test no action when RSI computation returns None."""
        # Set up None RSI (insufficient data)
        self.trader.compute_rsi.return_value = None

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertFalse(buy, "Should not execute buy when RSI is None")
        self.assertFalse(sell, "Should not execute sell when RSI is None")

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        self.trader.compute_rsi.return_value = 20.0  # Oversold

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        self.trader.compute_rsi.return_value = 80.0  # Overbought

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=70,
            oversold=30,
        )

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")

    def test_custom_thresholds(self):
        """Test strategy with custom overbought/oversold thresholds."""
        # Test with custom thresholds (40/60 instead of 30/70)
        self.trader.compute_rsi.return_value = 35.0  # Below custom oversold (40)

        buy, sell = strategy_rsi(
            self.trader,
            self.new_price,
            self.today,
            period=14,
            overbought=60,
            oversold=40,
        )

        self.assertTrue(buy, "Should execute buy with custom oversold threshold")
        self.assertFalse(sell, "Should not execute sell with custom oversold threshold")

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn("RSI", STRATEGY_REGISTRY, "RSI not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["RSI"]), "RSI not callable")


if __name__ == "__main__":
    unittest.main()
