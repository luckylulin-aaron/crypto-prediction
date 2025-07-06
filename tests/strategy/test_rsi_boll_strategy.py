#!/usr/bin/env python3
"""
Test script for RSI-BOLL (Combined RSI and Bollinger Bands) strategy.
"""

import os
import sys
import datetime
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.trading.strategies import strategy_rsi_bollinger_combined
from app.core.config import BUY_SIGNAL, SELL_SIGNAL, NO_ACTION_SIGNAL


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()
    
    # Mock wallet
    trader.wallet = {'USD': 10000, 'crypto': 100}
    
    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()
    trader.compute_rsi = Mock(return_value=50.0)  # Default RSI value
    
    # Mock strategy dictionary
    trader.strat_dct = {
        "RSI-BOLL": []
    }
    
    # Mock moving averages (for Bollinger Bands calculation)
    trader.moving_averages = {
        "12": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    }
    
    return trader


class TestRSIBollStrategy(unittest.TestCase):
    """Test cases for RSI-Bollinger Bands combined strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.queue_name = "12"
        self.bollinger_sigma = 2
        self.new_price = 100.0
        
    def test_buy_signal_when_rsi_oversold_and_price_below_lower_band(self):
        """Test buy signal when RSI is oversold and price is below lower band."""
        # Set up RSI oversold
        self.trader.compute_rsi.return_value = 20.0
        
        # Set up price below lower Bollinger Band
        self.trader.moving_averages["12"] = [95.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        new_price = 85.0  # Below lower band
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertTrue(buy, "Should execute buy when RSI oversold and price below lower band")
        self.assertFalse(sell, "Should not execute sell when both signals are positive")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_when_rsi_overbought_and_price_above_upper_band(self):
        """Test sell signal when RSI is overbought and price is above upper band."""
        # Set up RSI overbought
        self.trader.compute_rsi.return_value = 80.0
        
        # Set up price above upper Bollinger Band
        self.trader.moving_averages["12"] = [115.0, 115.0, 115.0, 115.0, 115.0, 115.0]
        new_price = 125.0  # Above upper band
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when both signals are negative")
        self.assertTrue(sell, "Should execute sell when RSI overbought and price above upper band")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_rsi_oversold_but_price_within_bands(self):
        """Test no action when RSI is oversold but price is within bands."""
        # Set up RSI oversold
        self.trader.compute_rsi.return_value = 20.0
        
        # Set up price within bands
        self.trader.moving_averages["12"] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        new_price = 100.0  # Within bands
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when signals conflict")
        self.assertFalse(sell, "Should not execute sell when signals conflict")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_rsi_overbought_but_price_within_bands(self):
        """Test no action when RSI is overbought but price is within bands."""
        # Set up RSI overbought
        self.trader.compute_rsi.return_value = 80.0
        
        # Set up price within bands
        self.trader.moving_averages["12"] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        new_price = 100.0  # Within bands
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when signals conflict")
        self.assertFalse(sell, "Should not execute sell when signals conflict")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_rsi_neutral_and_price_within_bands(self):
        """Test no action when both RSI and price are neutral."""
        # Set up RSI neutral
        self.trader.compute_rsi.return_value = 50.0
        
        # Set up price within bands
        self.trader.moving_averages["12"] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        new_price = 100.0  # Within bands
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when both signals are neutral")
        self.assertFalse(sell, "Should not execute sell when both signals are neutral")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_rsi_none(self):
        """Test no action when RSI computation returns None."""
        # Set up None RSI
        self.trader.compute_rsi.return_value = None
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, self.new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when RSI is None")
        self.assertFalse(sell, "Should not execute sell when RSI is None")
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Set up both positive signals
        self.trader.compute_rsi.return_value = 20.0
        self.trader.moving_averages["12"] = [95.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        new_price = 85.0
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        # Set up both negative signals
        self.trader.compute_rsi.return_value = 80.0
        self.trader.moving_averages["12"] = [115.0, 115.0, 115.0, 115.0, 115.0, 115.0]
        new_price = 125.0
        
        buy, sell = strategy_rsi_bollinger_combined(
            self.trader, self.queue_name, new_price, self.today,
            self.bollinger_sigma, rsi_period=14, rsi_overbought=70, rsi_oversold=30
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("RSI-BOLL", STRATEGY_REGISTRY, "RSI-BOLL not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["RSI-BOLL"]), "RSI-BOLL not callable")


if __name__ == "__main__":
    unittest.main() 