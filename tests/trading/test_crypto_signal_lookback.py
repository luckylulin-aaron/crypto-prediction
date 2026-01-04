#!/usr/bin/env python3
"""
Tests for crypto trade signal lookback behavior.
"""

import datetime
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL


class DummyTrader:
    def __init__(self):
        self.buy_pct = 0.3
        self.sell_pct = 0.3
        self.crypto_name = "ETH"  # not a stock
        self.trade_history = []

    # Import the method from StratTrader by mixin-style delegation
    from app.trading.strat_trader import StratTrader as _ST

    get_trade_signal = _ST.get_trade_signal


class TestCryptoSignalLookback(unittest.TestCase):
    def test_returns_recent_buy_within_48h(self):
        t = DummyTrader()
        now = datetime.datetime.now()
        t.trade_history = [
            {"date": now - datetime.timedelta(hours=60), "action": BUY_SIGNAL},
            {"date": now - datetime.timedelta(hours=10), "action": NO_ACTION_SIGNAL},
            {"date": now - datetime.timedelta(hours=2), "action": BUY_SIGNAL},
        ]
        sig = t.get_trade_signal(lag_intervals=0, lookback_hours=24)
        self.assertEqual(sig["action"], "BUY")

    def test_skips_no_action_and_picks_recent_sell(self):
        t = DummyTrader()
        now = datetime.datetime.now()
        t.trade_history = [
            {"date": now - datetime.timedelta(hours=3), "action": NO_ACTION_SIGNAL},
            {"date": now - datetime.timedelta(hours=2), "action": SELL_SIGNAL},
        ]
        sig = t.get_trade_signal(lag_intervals=0, lookback_hours=24)
        self.assertEqual(sig["action"], "SELL")

    def test_no_recent_signal_returns_no_action(self):
        t = DummyTrader()
        now = datetime.datetime.now()
        t.trade_history = [
            {"date": now - datetime.timedelta(hours=49), "action": BUY_SIGNAL},
        ]
        sig = t.get_trade_signal(lag_intervals=0, lookback_hours=24)
        self.assertEqual(sig["action"], NO_ACTION_SIGNAL)


if __name__ == "__main__":
    unittest.main()


