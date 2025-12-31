#!/usr/bin/env python3
"""
Regression test: DATA_INTERVAL_HOURS must not affect stock (1d) trading granularity.
"""

import datetime
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import NO_ACTION_SIGNAL, STOCKS
from app.trading.strat_trader import StratTrader


class TestStockIntervalGranularity(unittest.TestCase):
    def test_stock_trade_signal_uses_daily_interval(self):
        """
        Ensure `StratTrader.get_trade_signal()` uses 24h intervals for stocks, not DATA_INTERVAL_HOURS.
        """
        stock = STOCKS[0] if STOCKS else "AAPL"
        t = StratTrader(
            name=stock,
            init_amount=1000,
            stat="MA-BOLL-BANDS",
            tol_pct=0.1,
            ma_lengths=[6, 12, 30],
            ema_lengths=[6, 12, 26, 30],
            bollinger_mas=[6, 12],
            bollinger_sigma=2,
            buy_pct=0.1,
            sell_pct=0.1,
            cur_coin=0.0,
            mode="normal",
        )

        # Simulate a signal from ~20 hours ago (should be considered "fresh" for stocks since 24h cadence).
        ts = datetime.datetime.now() - datetime.timedelta(hours=20)
        t.trade_history.append(
            {
                "action": NO_ACTION_SIGNAL,
                "price": 100.0,
                "date": ts,
                "coin": 0.0,
                "cash": 1000.0,
                "portfolio": 1000.0,
            }
        )

        sig = t.get_trade_signal(lag_intervals=0)
        self.assertIn("action", sig)
        self.assertEqual(sig["action"], NO_ACTION_SIGNAL)


if __name__ == "__main__":
    unittest.main()


