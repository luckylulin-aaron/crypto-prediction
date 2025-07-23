#!/usr/bin/env python3
"""
Test script for TREND-MA-SELVES (Trend-aware Moving Average) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_trend_aware_ma_selves


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
    trader.strat_dct = {"TREND-MA-SELVES": []}

    # Mock moving averages with trend data
    trader.moving_averages = {
        "20": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    }

    return trader


class TestTrendMASelvesStrategy(unittest.TestCase):
    """Test cases for TREND-MA-SELVES strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.queue_name = "20"
        self.tol_pct = 0.1
        self.buy_pct = 0.2
        self.sell_pct = 0.3
        self.trend_period = 50
        self.trend_threshold = 0.02

    def test_buy_signal_with_uptrend(self):
        """Test buy signal when price is below tolerance in uptrend."""
        # Price at 94.5 (10% below MA of 105) with uptrend
        new_price = 94.5  # (1 - 0.1) * 105 = 94.5

        buy, sell = strategy_trend_aware_ma_selves(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.trend_period,
            self.trend_threshold,
        )

        self.assertTrue(buy, "Should execute buy when price below tolerance in uptrend")
        self.assertFalse(
            sell, "Should not execute sell when price below tolerance in uptrend"
        )
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(
            new_price, self.today, BUY_SIGNAL
        )

    def test_sell_signal_with_downtrend(self):
        """Test sell signal when price is above tolerance in downtrend."""
        # Price at 115.5 (10% above MA of 105) with downtrend
        new_price = 115.5  # (1 + 0.1) * 105 = 115.5

        buy, sell = strategy_trend_aware_ma_selves(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.trend_period,
            self.trend_threshold,
        )

        self.assertFalse(
            buy, "Should not execute buy when price above tolerance in downtrend"
        )
        self.assertTrue(
            sell, "Should execute sell when price above tolerance in downtrend"
        )
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", new_price
        )
        self.trader._record_history.assert_called_with(
            new_price, self.today, SELL_SIGNAL
        )

    def test_no_action_when_price_within_tolerance(self):
        """Test no action when price is within tolerance range."""
        # Price at 105 (exactly at MA)
        new_price = 105.0

        buy, sell = strategy_trend_aware_ma_selves(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.trend_period,
            self.trend_threshold,
        )

        self.assertFalse(buy, "Should not execute buy when price within tolerance")
        self.assertFalse(sell, "Should not execute sell when price within tolerance")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False

        new_price = 94.5  # Below tolerance

        buy, sell = strategy_trend_aware_ma_selves(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.trend_period,
            self.trend_threshold,
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        new_price = 115.5  # Above tolerance

        buy, sell = strategy_trend_aware_ma_selves(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            self.trend_period,
            self.trend_threshold,
        )

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn(
            "TREND-MA-SELVES", STRATEGY_REGISTRY, "TREND-MA-SELVES not in registry"
        )
        self.assertTrue(
            callable(STRATEGY_REGISTRY["TREND-MA-SELVES"]),
            "TREND-MA-SELVES not callable",
        )

    def test_config_integration(self):
        """Test that strategy is in the config."""
        from app.core.config import STRATEGIES

        self.assertIn("TREND-MA-SELVES", STRATEGIES, "TREND-MA-SELVES not in config")


if __name__ == "__main__":
    unittest.main()
