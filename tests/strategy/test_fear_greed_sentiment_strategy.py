#!/usr/bin/env python3
"""
Test script for FEAR-GREED-SENTIMENT (Fear & Greed Index) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_fear_greed_sentiment


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
        "FEAR-GREED-SENTIMENT": []
    }
    
    # Mock fear & greed data
    trader.fear_greed_data = [
        {
            'timestamp': '2024-01-01',
            'value': 25,  # Fear
            'classification': 'Fear'
        },
        {
            'timestamp': '2024-01-02',
            'value': 75,  # Greed
            'classification': 'Greed'
        }
    ]
    
    return trader


class TestFearGreedSentimentStrategy(unittest.TestCase):
    """Test cases for FEAR-GREED-SENTIMENT strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0
        self.buy_thresholds = [20.0, 30.0]
        self.sell_thresholds = [70.0, 80.0]
        
    def test_buy_signal_when_fear_extreme(self):
        """Test buy signal when fear & greed index indicates extreme fear."""
        # Mock fear & greed data with extreme fear (value = 15)
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 15,  # Extreme Fear
                'classification': 'Extreme Fear'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertTrue(buy, "Should execute buy when fear & greed index indicates extreme fear")
        self.assertFalse(sell, "Should not execute sell when fear & greed index indicates extreme fear")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, BUY_SIGNAL)
        
    def test_buy_signal_when_fear_moderate(self):
        """Test buy signal when fear & greed index indicates moderate fear."""
        # Mock fear & greed data with moderate fear (value = 25)
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 25,  # Fear
                'classification': 'Fear'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertTrue(buy, "Should execute buy when fear & greed index indicates moderate fear")
        self.assertFalse(sell, "Should not execute sell when fear & greed index indicates moderate fear")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, BUY_SIGNAL)
        
    def test_sell_signal_when_greed_extreme(self):
        """Test sell signal when fear & greed index indicates extreme greed."""
        # Mock fear & greed data with extreme greed (value = 85)
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 85,  # Extreme Greed
                'classification': 'Extreme Greed'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should not execute buy when fear & greed index indicates extreme greed")
        self.assertTrue(sell, "Should execute sell when fear & greed index indicates extreme greed")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, SELL_SIGNAL)
        
    def test_sell_signal_when_greed_moderate(self):
        """Test sell signal when fear & greed index indicates moderate greed."""
        # Mock fear & greed data with moderate greed (value = 75)
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 75,  # Greed
                'classification': 'Greed'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should not execute buy when fear & greed index indicates moderate greed")
        self.assertTrue(sell, "Should execute sell when fear & greed index indicates moderate greed")
        self.trader._execute_one_sell.assert_called_once_with("by_percentage", self.new_price)
        self.trader._record_history.assert_called_with(self.new_price, self.today, SELL_SIGNAL)
        
    def test_no_action_when_neutral_sentiment(self):
        """Test no action when fear & greed index indicates neutral sentiment."""
        # Mock fear & greed data with neutral sentiment (value = 50)
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 50,  # Neutral
                'classification': 'Neutral'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should not execute buy when fear & greed index indicates neutral sentiment")
        self.assertFalse(sell, "Should not execute sell when fear & greed index indicates neutral sentiment")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_no_action_when_no_fear_greed_data(self):
        """Test no action when no fear & greed data is available."""
        # Mock empty fear & greed data
        self.trader.fear_greed_data = []
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should not execute buy when no fear & greed data available")
        self.assertFalse(sell, "Should not execute sell when no fear & greed data available")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False
        
        # Mock fear & greed data with extreme fear
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 15,  # Extreme Fear
                'classification': 'Extreme Fear'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False
        
        # Mock fear & greed data with extreme greed
        self.trader.fear_greed_data = [
            {
                'timestamp': '2024-01-01',
                'value': 85,  # Extreme Greed
                'classification': 'Extreme Greed'
            }
        ]
        
        buy, sell = strategy_fear_greed_sentiment(
            self.trader, self.new_price, self.today,
            self.buy_thresholds, self.sell_thresholds
        )
        
        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(self.new_price, self.today, NO_ACTION_SIGNAL)
        
    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn("FEAR-GREED-SENTIMENT", STRATEGY_REGISTRY, "FEAR-GREED-SENTIMENT not in registry")
        self.assertTrue(callable(STRATEGY_REGISTRY["FEAR-GREED-SENTIMENT"]), "FEAR-GREED-SENTIMENT not callable")
        
    def test_config_integration(self):
        """Test that strategy is in the config."""
        from app.core.config import STRATEGIES
        
        self.assertIn("FEAR-GREED-SENTIMENT", STRATEGIES, "FEAR-GREED-SENTIMENT not in config")


if __name__ == "__main__":
    unittest.main() 