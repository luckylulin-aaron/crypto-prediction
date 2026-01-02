#!/usr/bin/env python3
"""
LIVE integration tests (real API calls) to verify volume is returned by all data clients.

These are intentionally opt-in because they:
- require network access
- can be flaky / rate-limited
- may require credentials (Coinbase Advanced Trade)

Enable by setting:
  LIVE_API_TESTS=1

Optionally for Coinbase:
  COINBASE_API_KEY=...
  COINBASE_API_SECRET=...
"""

import os
import sys
import unittest

# Ensure both `app.*` imports and legacy `core.*` imports work in tests.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))


def _live_enabled() -> bool:
    return os.getenv("LIVE_API_TESTS", "").strip().lower() in ("1", "true", "yes", "y", "on")


@unittest.skipUnless(_live_enabled(), "Set LIVE_API_TESTS=1 to run live API integration tests")
class TestClientsLiveReturnVolume(unittest.TestCase):
    def test_binance_live_returns_volume(self):
        # Binance spot public klines typically work without API keys.
        from app.trading.binance_client import BinanceClient
        import app.trading.binance_client as mod

        # Keep the test light: don't fetch a huge history window.
        mod.TIMESPAN = int(os.getenv("LIVE_TIMESPAN_DAYS", "7"))

        c = BinanceClient(api_key="", api_secret="")
        rows = c.get_historic_data("BTCUSDT", use_cache=False)
        print('BTC-USD rows from Binance:', rows[:2])
        self.assertTrue(rows, "Expected Binance to return at least one row")
        self.assertGreaterEqual(len(rows[0]), 6, "Expected OHLCV row with volume as 6th element")
        self.assertIsInstance(rows[0][5], float)
        self.assertGreaterEqual(rows[0][5], 0.0)

    def test_coinbase_live_returns_volume(self):
        # Coinbase client may require credentials depending on library behavior/env.
        if not (os.getenv("COINBASE_API_KEY") and os.getenv("COINBASE_API_SECRET")):
            self.skipTest("Missing COINBASE_API_KEY/COINBASE_API_SECRET for live Coinbase test")

        from app.trading.cbpro_client import CBProClient
        import app.trading.cbpro_client as mod

        mod.TIMESPAN = int(os.getenv("LIVE_TIMESPAN_DAYS", "7"))

        c = CBProClient(key=os.getenv("COINBASE_API_KEY"), secret=os.getenv("COINBASE_API_SECRET"))
        rows = c.get_historic_data("BTC-USD", use_cache=False)
        print('BTC-USD rows from Coinbase:', rows[:2])
        self.assertTrue(rows, "Expected Coinbase to return at least one row")
        self.assertEqual(len(rows[0]), 6, "Expected [close, dt, open, low, high, volume]")
        self.assertIsInstance(rows[0][5], float)
        self.assertGreaterEqual(rows[0][5], 0.0)

    def test_us_stock_live_returns_volume(self):
        # yfinance requires network
        from app.trading.us_stock_client import USStockClient

        c = USStockClient(tickers=["AAPL"])
        rows = c.get_historic_data("AAPL", use_cache=False)
        print('AAPL rows:', rows[:2])
        self.assertTrue(rows, "Expected yfinance to return at least one row")
        self.assertEqual(len(rows[0]), 6, "Expected [close, date, open, low, high, volume]")
        # yfinance volume should generally be > 0 on trading days, but be tolerant.
        self.assertTrue(isinstance(rows[0][5], (int, float)))
        self.assertGreaterEqual(float(rows[0][5]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


