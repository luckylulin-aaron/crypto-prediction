#!/usr/bin/env python3
"""
Test script for EXP-MA-SELVES (Exponential Moving Average with Tolerance) strategy.
"""

import datetime
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import strategy_exponential_moving_average_w_tolerance


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()

    # Mock wallet
    trader.wallet = {"USD": 10000, "crypto": 100}
    trader.cash = 10000.0
    trader.cur_coin = 0.0

    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()

    # Mock strategy dictionary
    trader.strat_dct = {"EXP-MA-SELVES": []}

    # Mock exponential moving averages
    trader.exp_moving_averages = {"20": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}

    # Mock volume history (always present in pipeline)
    trader.volume_history = [100.0] * 30 + [100.0]

    return trader


class TestExpMASelvesStrategy(unittest.TestCase):
    """Test cases for EXP-MA-SELVES strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.queue_name = "20"
        self.tol_pct = 0.1
        self.buy_pct = 0.2
        self.sell_pct = 0.3

    def test_buy_signal_when_price_below_tolerance(self):
        """Buy in bullish regime on a pullback below EMA (easier buy)."""
        # Bullish slope; pullback crosses below EMA
        self.trader.exp_moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        # Seed prev rel (previous price above EMA)
        setattr(self.trader, "_exp_ma_prev_rel:20", 0.01)
        new_price = 104.5  # slightly below EMA(105)

        buy, sell = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=120.0,
        )

        self.assertTrue(buy, "Should execute buy when price below tolerance")
        self.assertFalse(sell, "Should not execute sell when price below tolerance")
        self.trader._execute_one_buy.assert_called_once_with("by_percentage", new_price)
        self.trader._record_history.assert_called_with(
            new_price, self.today, BUY_SIGNAL
        )

    def test_bearish_regime_easier_sell(self):
        """In bearish regime, sell is easier (smaller deviation above EMA)."""
        # Bearish slope
        self.trader.exp_moving_averages["20"] = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0]
        self.trader.cur_coin = 1.0
        self.trader.cash = 0.0
        new_price = 111.0  # above EMA by ~5.7% => should sell with bear-easy threshold

        buy, sell = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=120.0,
        )

        self.assertFalse(buy, "Should not execute buy when price above tolerance")
        self.assertTrue(sell, "Should execute sell when price above tolerance")
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", new_price
        )
        self.trader._record_history.assert_called_with(
            new_price, self.today, SELL_SIGNAL
        )

    def test_bearish_regime_harder_buy_requires_volume_and_reversal(self):
        """In bearish regime, buys are harder: require volume spike + reversal confirmation."""
        self.trader.exp_moving_averages["20"] = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0]
        self.trader.cur_coin = 0.0
        self.trader.cash = 10000.0
        # Far below EMA, but volume is not a spike and candle not a reversal -> no buy
        new_price = 90.0
        buy, sell = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=100.0,
            open_p=95.0,
            low_p=88.0,
            high_p=96.0,
        )

        self.assertFalse(buy, "Should not execute buy when price within tolerance")
        self.assertFalse(sell, "Should not execute sell when price within tolerance")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_bullish_regime_harder_sell_uses_trailing_stop(self):
        """In bullish regime, don't sell on first spike; sell after reversal via trailing stop."""
        self.trader.exp_moving_averages["20"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        self.trader.cur_coin = 1.0
        self.trader.cash = 0.0

        # Arm blowoff
        spike_price = 114.0  # +8.6% above EMA, arms state
        buy1, sell1 = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            spike_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=150.0,
            cooldown_days=0,
        )
        self.assertFalse(buy1)
        self.assertFalse(sell1)

        # Reversal triggers trailing stop (~6% below 114 -> <= 107.16)
        next_day = self.today + datetime.timedelta(days=1)
        reversal_price = 107.0
        buy2, sell2 = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            reversal_price,
            next_day,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=150.0,
            cooldown_days=0,
        )
        self.assertFalse(buy2)
        self.assertTrue(sell2)

    def test_buy_execution_failure(self):
        """Test behavior when buy execution fails."""
        self.trader._execute_one_buy.return_value = False

        # Bull pullback buy attempt
        setattr(self.trader, "_exp_ma_prev_rel:20", 0.01)
        new_price = 104.5

        buy, sell = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=120.0,
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        # Bearish easy sell attempt
        self.trader.exp_moving_averages["20"] = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0]
        self.trader.cur_coin = 1.0
        self.trader.cash = 0.0
        new_price = 111.0

        buy, sell = strategy_exponential_moving_average_w_tolerance(
            self.trader,
            self.queue_name,
            new_price,
            self.today,
            self.tol_pct,
            self.buy_pct,
            self.sell_pct,
            volume=120.0,
            cooldown_days=0,
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
            "EXP-MA-SELVES", STRATEGY_REGISTRY, "EXP-MA-SELVES not in registry"
        )
        self.assertTrue(
            callable(STRATEGY_REGISTRY["EXP-MA-SELVES"]), "EXP-MA-SELVES not callable"
        )

    def test_config_integration(self):
        """Test that strategy is in the config."""
        from app.core.config import SUPPORTED_STRATEGIES

        self.assertIn(
            "EXP-MA-SELVES", SUPPORTED_STRATEGIES, "EXP-MA-SELVES not supported in config"
        )


if __name__ == "__main__":
    unittest.main()
