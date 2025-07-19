#!/usr/bin/env python3
"""
Test script for VOLUME-BREAKOUT strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_volume_breakout


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
        "VOLUME-BREAKOUT": []
    }
    
    # Mock volume history
    trader.volume_history = [1000, 1100, 1200, 1300, 1400, 1500]
    
    return trader


class TestVolumeBreakoutStrategy(unittest.TestCase):
    """Test cases for Volume Breakout strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0
        
    def test_buy_signal_when_volume_above_threshold(self):
        """Test buy signal when volume is above threshold."""
        # Set up high volume (2000 > 1.5 * average of ~1250)
        self.trader.volume_history = [1000, 1100, 1200, 1300, 1400, 2000]
        
        buy, sell = strategy_volume_breakout(
            self.trader, self.new_price, self.today, 
            volume_period=20, volume_threshold=1.5
        )
        
        self.assertTrue(buy, "Should execute buy when volume is above threshold")
        self.assertFalse(sell, "Should not execute sell when volume is above threshold")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, BUY_SIGNAL)
        
    def test_no_action_when_volume_below_threshold(self):
        """Test no action when volume is below threshold."""
        # Set up normal volume (1200 < 1.5 * average of ~1250)
        self.trader.volume_history = [1000, 1100, 1200, 1300, 1400, 1200]
        
        buy, sell = strategy_volume_breakout(
            self.trader, self.new_price, self.today, 
            volume_period=20, volume_threshold=1.5
        )
        
        self.assertFalse(buy, "Should not execute buy when volume is below threshold")
        self.assertFalse(sell, "Should not execute sell when volume is below threshold")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_insufficient_volume_data(self):
        """Test no action when insufficient volume data."""
        # Set up insufficient data (less than volume_period)
        self.trader.volume_history = [1000, 1100, 1200]  # Only 3 values for period 20
        
        buy, sell = strategy_volume_breakout(
            self.trader, self.new_price, self.today, 
            volume_period=20, volume_threshold=1.5
        )
        
        self.assertFalse(buy, "Should not execute buy when insufficient volume data")
        self.assertFalse(sell, "Should not execute sell when insufficient volume data")
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Set up high volume
        self.trader.volume_history = [1000, 1100, 1200, 1300, 1400, 2000]
        
        buy, sell = strategy_volume_breakout(
            self.trader, self.new_price, self.today, 
            volume_period=20, volume_threshold=1.5
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_custom_volume_threshold(self):
        """Test strategy with custom volume threshold."""
        # Set up volume that's above custom threshold (2.0) but below default (1.5)
        self.trader.volume_history = [1000, 1100, 1200, 1300, 1400, 1800]  # 1.8x average
        
        buy, sell = strategy_volume_breakout(
            self.trader, self.new_price, self.today, 
            volume_period=20, volume_threshold=2.0
        )
        
        self.assertFalse(buy, "Should not execute buy when volume below custom threshold")
        self.assertFalse(sell, "Should not execute sell when volume below custom threshold")
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("VOLUME-BREAKOUT", STRATEGY_REGISTRY, "VOLUME-BREAKOUT not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["VOLUME-BREAKOUT"]), "VOLUME-BREAKOUT not callable")


if __name__ == "__main__":
    unittest.main() 