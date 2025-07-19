#!/usr/bin/env python3
"""
Test script for MACD-KDJ (Combined MACD and KDJ) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_macd_kdj_combined


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()
    
    # Mock wallet
    trader.wallet = {'USD': 10000, 'crypto': 100}
    
    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()
    
    # Mock strategy dictionary
    trader.strat_dct = {
        "MACD-KDJ": []
    }
    
    # Mock MACD data
    trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
    trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Mock KDJ data
    trader.kdj_dct = {
        "K": [50.0, 45.0, 40.0, 35.0, 30.0, 25.0],
        "D": [55.0, 50.0, 45.0, 40.0, 35.0, 30.0],
        "J": [45.0, 40.0, 35.0, 30.0, 25.0, 20.0]
    }
    
    return trader


class TestMACDKDJStrategy(unittest.TestCase):
    """Test cases for MACD-KDJ combined strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0
        
    def test_buy_signal_when_both_macd_and_kdj_positive(self):
        """Test buy signal when both MACD and KDJ signals are positive."""
        # Set up MACD buy signal (positive difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up KDJ buy signal (oversold)
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 20.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertTrue(buy, "Should execute buy when both MACD and KDJ signals are positive")
        self.assertFalse(sell, "Should not execute sell when both signals are positive")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_when_both_macd_and_kdj_negative(self):
        """Test sell signal when both MACD and KDJ signals are negative."""
        # Set up MACD sell signal (negative difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up KDJ sell signal (overbought)
        self.trader.kdj_dct["K"] = [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when both signals are negative")
        self.assertTrue(sell, "Should execute sell when both MACD and KDJ signals are negative")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_macd_positive_kdj_negative(self):
        """Test no action when MACD is positive but KDJ is negative."""
        # Set up MACD buy signal
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up KDJ sell signal
        self.trader.kdj_dct["K"] = [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when signals conflict")
        self.assertFalse(sell, "Should not execute sell when signals conflict")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_macd_negative_kdj_positive(self):
        """Test no action when MACD is negative but KDJ is positive."""
        # Set up MACD sell signal
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up KDJ buy signal
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 20.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when signals conflict")
        self.assertFalse(sell, "Should not execute sell when signals conflict")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_kdj_data_missing(self):
        """Test no action when KDJ data is missing."""
        # Set up MACD buy signal
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up missing KDJ data
        self.trader.kdj_dct = {}
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when KDJ data is missing")
        self.assertFalse(sell, "Should not execute sell when KDJ data is missing")
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Set up both positive signals
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 20.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        # Set up both negative signals
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.trader.kdj_dct["K"] = [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]
        
        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("MACD-KDJ", STRATEGY_REGISTRY, "MACD-KDJ not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["MACD-KDJ"]), "MACD-KDJ not callable")


if __name__ == "__main__":
    unittest.main() 