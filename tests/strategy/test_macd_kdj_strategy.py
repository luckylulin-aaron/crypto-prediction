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
    trader.wallet = {"USD": 10000, "crypto": 100}
    # Also set explicit fields used by StratTrader (position-awareness)
    trader.cash = 10000.0
    trader.cur_coin = 1.0

    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()

    # Mock strategy dictionary
    trader.strat_dct = {"MACD-KDJ": []}

    # Mock MACD/KDJ data (will be overridden per test)
    trader.macd_diff = [0.1, 0.3]
    trader.macd_dea = [0.2, 0.2]
    trader.kdj_dct = {
        "K": [20.0, 30.0],
        "D": [55.0, 50.0, 45.0, 40.0, 35.0, 30.0],
        "J": [45.0, 40.0, 35.0, 30.0, 25.0, 20.0],
    }

    # Mock volume history (always present in pipeline)
    trader.volume_history = [100.0] * 30 + [120.0]

    return trader


class TestMACDKDJStrategy(unittest.TestCase):
    """Test cases for MACD-KDJ combined strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.trader = create_mock_trader()
        self.today = datetime.datetime(2024, 1, 1)
        self.new_price = 100.0

    def test_bullish_regime_easier_buy(self):
        """In bullish regime, buy should be easier (MACD bull cross + KDJ not overbought)."""
        # Bullish regime (diff/dea >= 0) and histogram crosses up:
        # hist_prev = 0.15 - 0.2 = -0.05, hist_cur = 0.25 - 0.2 = 0.05
        self.trader.macd_diff = [0.15, 0.25]
        self.trader.macd_dea = [0.2, 0.2]
        # Not oversold, but <= bull_kdj_buy_max (default 60)
        self.trader.kdj_dct["K"] = [45.0, 55.0]
        self.trader.volume_history = [100.0] * 30 + [110.0]

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertTrue(
            buy, "Should execute buy when both MACD and KDJ signals are positive"
        )
        self.assertFalse(sell, "Should not execute sell when both signals are positive")
        self.trader._execute_one_buy.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, BUY_SIGNAL
        )

    def test_bullish_regime_harder_sell_requires_kdj_cross(self):
        """In bullish regime, sell should be harder (requires MACD bear cross AND KDJ sell-cross)."""
        # Bullish regime (diff/dea >=0), but histogram crosses down:
        # hist_prev = 0.25 - 0.2 = 0.05, hist_cur = 0.15 - 0.2 = -0.05
        self.trader.macd_diff = [0.25, 0.15]
        self.trader.macd_dea = [0.2, 0.2]
        # KDJ sell-cross: prev > 70, cur <= 70
        self.trader.kdj_dct["K"] = [80.0, 69.0]
        self.trader.volume_history = [100.0] * 30 + [150.0]  # pass stricter sell volume

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertFalse(buy, "Should not execute buy when both signals are negative")
        self.assertTrue(
            sell, "Should execute sell when both MACD and KDJ signals are negative"
        )
        self.trader._execute_one_sell.assert_called_once_with(
            "by_percentage", self.new_price
        )
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, SELL_SIGNAL
        )

    def test_bullish_regime_sell_not_triggered_without_kdj_cross(self):
        """Bullish regime: MACD bear cross alone should NOT sell (harder exits)."""
        self.trader.macd_diff = [0.25, 0.15]
        self.trader.macd_dea = [0.2, 0.2]
        # No KDJ sell-cross (stays under overbought)
        self.trader.kdj_dct["K"] = [60.0, 65.0]
        self.trader.volume_history = [100.0] * 30 + [200.0]

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertFalse(buy, "Should not execute buy when signals conflict")
        self.assertFalse(sell, "Should not execute sell when signals conflict")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_bearish_regime_harder_buy_requires_kdj_cross_and_volume_spike(self):
        """Bearish regime: buy requires KDJ oversold cross + MACD bull cross + higher volume."""
        # Bearish regime (diff/dea < 0) and histogram crosses up
        self.trader.macd_diff = [-0.25, -0.15]  # dea will be -0.2 to keep regime bearish
        self.trader.macd_dea = [-0.2, -0.2]
        # KDJ oversold cross: prev < 30, cur >= 30 and cur <= bear_kdj_buy_max (35)
        self.trader.kdj_dct["K"] = [20.0, 30.0]
        # Volume NOT high enough for bear buy (needs >= 1.3x avg); avg=100, cur=110
        self.trader.volume_history = [100.0] * 30 + [110.0]

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertFalse(buy, "Should not buy in bearish regime without required volume spike")
        self.assertFalse(sell, "Should not sell on buy setup")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_bearish_regime_easier_sell_can_trigger_on_macd_cross(self):
        """Bearish regime: sell can trigger on MACD bear cross alone (easier exits)."""
        self.trader.cash = 0.0
        self.trader.cur_coin = 1.0
        # Bearish regime: keep diff/dea < 0
        self.trader.macd_diff = [-0.15, -0.25]
        self.trader.macd_dea = [-0.2, -0.2]
        # hist_prev = -0.15 - (-0.2)=0.05, hist_cur=-0.25-(-0.2)=-0.05 (cross down)
        self.trader.kdj_dct["K"] = [50.0, 55.0]  # no sell-cross, but bearish allows MACD-only sell
        self.trader.volume_history = [100.0] * 30 + [100.0]

        buy, sell = strategy_macd_kdj_combined(self.trader, self.new_price, self.today)
        self.assertFalse(buy)
        self.assertTrue(sell)

    def test_cooldown_blocks_repeated_trades(self):
        """Cooldown should prevent rapid repeat trades even if signals re-trigger."""
        # Set a recent BUY yesterday
        yesterday = self.today - datetime.timedelta(days=1)
        self.trader.strat_dct["MACD-KDJ"] = [(yesterday, BUY_SIGNAL)]
        self.trader.macd_diff = [0.15, 0.25]
        self.trader.macd_dea = [0.2, 0.2]
        self.trader.kdj_dct["K"] = [45.0, 55.0]
        self.trader.volume_history = [100.0] * 30 + [200.0]

        buy, sell = strategy_macd_kdj_combined(self.trader, self.new_price, self.today, cooldown_days=2)
        self.assertFalse(buy)
        self.assertFalse(sell)

    def test_no_action_when_kdj_data_missing(self):
        """Test no action when KDJ data is missing."""
        # Set up MACD cross candidates (won't matter due to missing KDJ)
        self.trader.macd_diff = [0.15, 0.25]
        self.trader.macd_dea = [0.2, 0.2]

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

        # Set up bullish regime buy setup
        self.trader.macd_diff = [0.15, 0.25]
        self.trader.macd_dea = [0.2, 0.2]
        self.trader.kdj_dct["K"] = [45.0, 55.0]
        self.trader.volume_history = [100.0] * 30 + [200.0]

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertFalse(buy, "Should return False when buy execution fails")
        self.assertFalse(sell, "Should not execute sell when buy fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_sell_execution_failure(self):
        """Test behavior when sell execution fails."""
        self.trader._execute_one_sell.return_value = False

        # Set up bullish regime sell setup (MACD bear cross + KDJ sell-cross)
        self.trader.macd_diff = [0.25, 0.15]
        self.trader.macd_dea = [0.2, 0.2]
        self.trader.kdj_dct["K"] = [80.0, 69.0]
        self.trader.volume_history = [100.0] * 30 + [200.0]

        buy, sell = strategy_macd_kdj_combined(
            self.trader, self.new_price, self.today, oversold=30, overbought=70
        )

        self.assertFalse(buy, "Should not execute buy when sell fails")
        self.assertFalse(sell, "Should return False when sell execution fails")
        self.trader._record_history.assert_called_with(
            self.new_price, self.today, NO_ACTION_SIGNAL
        )

    def test_strategy_registry_integration(self):
        """Test that strategy is properly registered."""
        from app.trading.strategies import STRATEGY_REGISTRY

        self.assertIn("MACD-KDJ", STRATEGY_REGISTRY, "MACD-KDJ not in registry")
        self.assertTrue(
            callable(STRATEGY_REGISTRY["MACD-KDJ"]), "MACD-KDJ not callable"
        )


if __name__ == "__main__":
    unittest.main()
