#!/usr/bin/env python3
"""
Test script for TRIPLE-SIGNAL (MA + MACD + RSI) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_triple_signal


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
        "TRIPLE-SIGNAL": []
    }
    
    # Mock moving averages
    trader.moving_averages = {
        "12": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    }
    
    # Mock MACD data
    trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
    trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    return trader


class TestTripleSignalStrategy(unittest.TestCase):
    """Test cases for TRIPLE-SIGNAL strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.queue_name = "12"
        self.tol_pct = 0.1
        self.buy_pct = 0.3
        self.sell_pct = 0.3
        
    def test_buy_signal_when_two_out_of_three_signals_positive(self):
        """Test buy signal when 2 out of 3 signals are positive."""
        # Set up MA buy signal (price below MA - tolerance)
        self.trader.moving_averages["12"] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
        new_price = 94.0  # Below MA (105) - tolerance (10.5) = 94.5
        
        # Set up MACD buy signal (positive difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up RSI neutral (not oversold)
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertTrue(buy, "Should execute buy when 2 out of 3 signals are positive")
        self.assertFalse(sell, "Should not execute sell when 2 out of 3 signals are positive")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_when_two_out_of_three_signals_negative(self):
        """Test sell signal when 2 out of 3 signals are negative."""
        # Set up MA sell signal (price above MA + tolerance)
        self.trader.moving_averages["12"] = [95.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        new_price = 106.0  # Above MA (95) + tolerance (9.5) = 104.5
        
        # Set up MACD sell signal (negative difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up RSI neutral (not overbought)
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should not execute buy when 2 out of 3 signals are negative")
        self.assertTrue(sell, "Should execute sell when 2 out of 3 signals are negative")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_only_one_signal_positive(self):
        """Test no action when only 1 out of 3 signals is positive."""
        # Set up MA buy signal
        self.trader.moving_averages["12"] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
        new_price = 94.0
        
        # Set up MACD neutral (zero difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.2]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up RSI neutral
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should not execute buy when only 1 signal is positive")
        self.assertFalse(sell, "Should not execute sell when only 1 signal is positive")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_all_signals_neutral(self):
        """Test no action when all signals are neutral."""
        # Set up neutral MA (price within tolerance)
        self.trader.moving_averages["12"] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        new_price = 100.0
        
        # Set up neutral MACD (zero difference)
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.2]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up neutral RSI
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should not execute buy when all signals are neutral")
        self.assertFalse(sell, "Should not execute sell when all signals are neutral")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_rsi_none(self):
        """Test no action when RSI computation returns None."""
        # Set up MA buy signal
        self.trader.moving_averages["12"] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
        new_price = 94.0
        
        # Set up MACD buy signal
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Set up None RSI
        self.trader.compute_rsi.return_value = None
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should not execute buy when RSI is None")
        self.assertFalse(sell, "Should not execute sell when RSI is None")
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Set up 2 positive signals
        self.trader.moving_averages["12"] = [105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
        new_price = 94.0
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.3]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        # Set up 2 negative signals
        self.trader.moving_averages["12"] = [95.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        new_price = 106.0
        self.trader.macd_diff = [0.5, 0.3, 0.1, -0.1, -0.3, 0.1]
        self.trader.macd_dea = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.trader.compute_rsi.return_value = 50.0
        
        buy, sell = strategy_triple_signal(
            self.trader, self.queue_name, new_price, self.today,
            self.tol_pct, self.buy_pct, self.sell_pct
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("TRIPLE-SIGNAL", STRATEGY_REGISTRY, "TRIPLE-SIGNAL not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["TRIPLE-SIGNAL"]), "TRIPLE-SIGNAL not callable")


if __name__ == "__main__":
    unittest.main() 