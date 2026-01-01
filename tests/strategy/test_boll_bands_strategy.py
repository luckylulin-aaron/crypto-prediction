#!/usr/bin/env python3
"""
Test script for BOLL-BANDS (Bollinger Bands) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_bollinger_bands


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
    trader.strat_dct = {"BOLL-BANDS": []}

    # Mock moving averages (Bollinger Bands uses MA data)
    trader.moving_averages = {"12": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}

    return trader


class TestBollingerBandsStrategy(unittest.TestCase):
    """Test cases for Bollinger Bands strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.ma_length = "12"
        self.bollinger_sigma = 2

    def test_buy_signal_when_price_below_lower_band(self):
        """Test buy signal when price is below lower Bollinger Band."""
        # Set up moving averages that will create lower band around 90
        # With sigma=2, std=5, mean=100: lower_band = 100 - 2*5 = 90
        # Provide 13 values so strategy can avoid lookahead by excluding today's MA.
        self.trader.moving_averages["12"] = [
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            100,
            100,
        ]
        new_price = 85.0  # Below lower band (90)

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertTrue(buy, "Should execute buy when price is below lower band")
        self.assertFalse(sell, "Should not execute sell when price is below lower band")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(
            new_price, self.today, BUY_SIGNAL
        )

    def test_sell_signal_when_price_above_upper_band(self):
        """Test sell signal when price is above upper Bollinger Band."""
        # Set up moving averages that will create upper band around 110
        # With sigma=2, std=5, mean=100: upper_band = 100 + 2*5 = 110
        # Provide 13 values so strategy can avoid lookahead by excluding today's MA.
        self.trader.moving_averages["12"] = [
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            100,
            100,
        ]
        # Require a small breakout beyond upper band (default 0.2%)
        new_price = 116.0

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertFalse(buy, "Should not execute buy when price is above upper band")
        self.assertTrue(sell, "Should execute sell when price is above upper band")
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", new_price
        )
        self.trader._record_history.assert_called_with(
            new_price, self.today, SELL_SIGNAL
        )

    def test_no_action_when_price_within_bands(self):
        """Test no action when price is within Bollinger Bands."""
        # Set up moving averages that will create bands around 100
        # With sigma=2, std=5, mean=100: bands = 90 to 110
        # Provide 13 values so strategy can avoid lookahead by excluding today's MA.
        self.trader.moving_averages["12"] = [
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            100,
            100,
        ]
        new_price = 100.0  # Within bands (90-110)

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertFalse(buy, "Should not execute buy when price is within bands")
        self.assertFalse(sell, "Should not execute sell when price is within bands")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_no_action_when_insufficient_data(self):
        """Test no action when insufficient moving average data."""
        # Set up insufficient data (less than queue length)
        self.trader.moving_averages["12"] = [100.0, 101.0, 102.0]  # Only 3 values for length 12
        new_price = 100.0

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertFalse(buy, "Should not execute buy when insufficient data")
        self.assertFalse(sell, "Should not execute sell when insufficient data")

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False

        # Set up moving averages for buy signal
        # Provide 13 values so strategy can avoid lookahead by excluding today's MA.
        self.trader.moving_averages["12"] = [
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            100,
            100,
        ]
        new_price = 85.0

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        # Set up moving averages for sell signal
        # Provide 13 values so strategy can avoid lookahead by excluding today's MA.
        self.trader.moving_averages["12"] = [
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            100,
            100,
        ]
        new_price = 116.0

        buy, sell = strategy_bollinger_bands(
            self.trader, self.ma_length, new_price, self.today, self.bollinger_sigma
        )

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn("BOLL-BANDS", STRATEGY_REGISTRY, "BOLL-BANDS not in registry")
        self.assertTrue(
            callable(STRATEGY_REGISTRY["BOLL-BANDS"]), "BOLL-BANDS not callable"
        )


if __name__ == "__main__":
    unittest.main()
