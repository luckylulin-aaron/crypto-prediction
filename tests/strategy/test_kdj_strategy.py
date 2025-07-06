#!/usr/bin/env python3
"""
Test script for KDJ (Stochastic Oscillator) strategy.
"""

import os
import sys
import datetime
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.trading.strategies import strategy_kdj
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
    
    # Mock strategy dictionary
    trader.strat_dct = {
        "KDJ": []
    }
    
    # Mock KDJ data
    trader.kdj_dct = {
        "K": [50.0, 45.0, 40.0, 35.0, 30.0, 25.0],  # K values
        "D": [55.0, 50.0, 45.0, 40.0, 35.0, 30.0],  # D values
        "J": [45.0, 40.0, 35.0, 30.0, 25.0, 20.0]   # J values
    }
    
    return trader


class TestKDJStrategy(unittest.TestCase):
    """Test cases for KDJ strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0
        
    def test_buy_signal_when_kdj_oversold(self):
        """Test buy signal when KDJ is oversold (below threshold)."""
        # Set up oversold KDJ (20 < 30)
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 20.0]
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertTrue(buy, "Should execute buy when KDJ is oversold")
        self.assertFalse(sell, "Should not execute sell when KDJ is oversold")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_when_kdj_overbought(self):
        """Test sell signal when KDJ is overbought (above threshold)."""
        # Set up overbought KDJ (80 > 70)
        self.trader.kdj_dct["K"] = [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when KDJ is overbought")
        self.assertTrue(sell, "Should execute sell when KDJ is overbought")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_kdj_neutral(self):
        """Test no action when KDJ is in neutral zone."""
        # Set up neutral KDJ (50 between 30 and 70)
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 50.0]
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when KDJ is neutral")
        self.assertFalse(sell, "Should not execute sell when KDJ is neutral")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_kdj_data_missing(self):
        """Test no action when KDJ data is missing."""
        # Set up missing KDJ data
        self.trader.kdj_dct = {}
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when KDJ data is missing")
        self.assertFalse(sell, "Should not execute sell when KDJ data is missing")
        
    def test_no_action_when_k_values_empty(self):
        """Test no action when K values are empty."""
        # Set up empty K values
        self.trader.kdj_dct["K"] = []
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when K values are empty")
        self.assertFalse(sell, "Should not execute sell when K values are empty")
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Set up oversold KDJ
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 20.0]
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        # Set up overbought KDJ
        self.trader.kdj_dct["K"] = [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        
    def test_custom_thresholds(self):
        """Test strategy with custom overbought/oversold thresholds."""
        # Test with custom thresholds (40/60 instead of 30/70)
        self.trader.kdj_dct["K"] = [50.0, 45.0, 40.0, 35.0, 30.0, 35.0]  # Below custom oversold (40)
        
        buy, sell = strategy_kdj(
            self.trader, self.new_price, self.today, oversold=40, overbought=60
        )
        
        self.assertTrue(buy, "Should execute buy with custom oversold threshold")
        self.assertFalse(sell, "Should not execute sell with custom oversold threshold")
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("KDJ", STRATEGY_REGISTRY, "KDJ not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["KDJ"]), "KDJ not callable")


if __name__ == "__main__":
    unittest.main() 