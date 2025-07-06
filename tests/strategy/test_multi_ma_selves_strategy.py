#!/usr/bin/env python3
"""
Test script for MULTI-MA-SELVES (Multi-timeframe Moving Average) strategy.
"""

import os
import sys
import datetime
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.trading.strategies import strategy_multi_timeframe_ma_selves
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
        "MULTI-MA-SELVES": []
    }
    
    # Mock moving averages for multiple timeframes
    trader.moving_averages = {
        "10": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],  # Short-term
        "20": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],  # Medium-term
        "50": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]   # Long-term
    }
    
    return trader


class TestMultiMASelvesStrategy(unittest.TestCase):
    """Test cases for MULTI-MA-SELVES strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.short_queue = "10"
        self.medium_queue = "20"
        self.long_queue = "50"
        self.tol_pct = 0.1
        self.buy_pct = 0.2
        self.sell_pct = 0.3
        self.confirmation_required = 2
        
    def test_buy_signal_with_multiple_confirmations(self):
        """Test buy signal when multiple timeframes confirm."""
        # Price at 94.5 (10% below MA of 105) - should trigger buy on multiple timeframes
        new_price = 94.5  # (1 - 0.1) * 105 = 94.5
        
        buy, sell = strategy_multi_timeframe_ma_selves(
            self.trader, self.short_queue, self.medium_queue, self.long_queue,
            new_price, self.today, self.tol_pct, self.buy_pct, self.sell_pct,
            self.confirmation_required
        )
        
        self.assertTrue(buy, "Should execute buy when multiple timeframes confirm")
        self.assertFalse(sell, "Should not execute sell when multiple timeframes confirm buy")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_with_multiple_confirmations(self):
        """Test sell signal when multiple timeframes confirm."""
        # Price at 115.5 (10% above MA of 105) - should trigger sell on multiple timeframes
        new_price = 115.5  # (1 + 0.1) * 105 = 115.5
        
        buy, sell = strategy_multi_timeframe_ma_selves(
            self.trader, self.short_queue, self.medium_queue, self.long_queue,
            new_price, self.today, self.tol_pct, self.buy_pct, self.sell_pct,
            self.confirmation_required
        )
        
        self.assertFalse(buy, "Should not execute buy when multiple timeframes confirm sell")
        self.assertTrue(sell, "Should execute sell when multiple timeframes confirm")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_insufficient_confirmations(self):
        """Test no action when insufficient timeframes confirm."""
        # Price at 105 (exactly at MA) - no clear signal
        new_price = 105.0
        
        buy, sell = strategy_multi_timeframe_ma_selves(
            self.trader, self.short_queue, self.medium_queue, self.long_queue,
            new_price, self.today, self.tol_pct, self.buy_pct, self.sell_pct,
            self.confirmation_required
        )
        
        self.assertFalse(buy, "Should not execute buy when insufficient confirmations")
        self.assertFalse(sell, "Should not execute sell when insufficient confirmations")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        new_price = 94.5  # Below tolerance
        
        buy, sell = strategy_multi_timeframe_ma_selves(
            self.trader, self.short_queue, self.medium_queue, self.long_queue,
            new_price, self.today, self.tol_pct, self.buy_pct, self.sell_pct,
            self.confirmation_required
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        new_price = 115.5  # Above tolerance
        
        buy, sell = strategy_multi_timeframe_ma_selves(
            self.trader, self.short_queue, self.medium_queue, self.long_queue,
            new_price, self.today, self.tol_pct, self.buy_pct, self.sell_pct,
            self.confirmation_required
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("MULTI-MA-SELVES", STRATEGY_REGISTRY, "MULTI-MA-SELVES not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["MULTI-MA-SELVES"]), "MULTI-MA-SELVES not callable")
        
    def test_config_integration(self):
        """Test that strategy is in the config."""
        from app.core.config import STRATEGIES
        
        self.assertIn("MULTI-MA-SELVES", STRATEGIES, "MULTI-MA-SELVES not in config")


if __name__ == "__main__":
    unittest.main() 