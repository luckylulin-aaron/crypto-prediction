#!/usr/bin/env python3
"""
Verify that all 3 data clients (Binance, Coinbase Advanced Trade, yfinance stocks)
return OHLCV rows where volume is present as the 6th element.

These tests mock upstream API calls to avoid network dependencies.
"""

import os
import sys
import unittest
import warnings
from unittest.mock import Mock

# Ensure both `app.*` imports and legacy `core.*` imports work in tests.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))

# Silence noisy warnings during client imports.
try:
    from sqlalchemy.exc import MovedIn20Warning

    warnings.filterwarnings("ignore", category=MovedIn20Warning)
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*The ``declarative_base\(\)`` function is now available.*",
)


class TestClientsReturnVolume(unittest.TestCase):
    def test_binance_client_returns_volume(self):
        try:
            from app.trading.binance_client import BinanceClient
        except Exception as e:
            self.skipTest(f"Binance client import failed (dependency missing?): {e}")

        c = BinanceClient(api_key="", api_secret="")
        c.client = Mock()

        # Binance kline format: [
        #   open_time_ms, open, high, low, close, volume, ...
        # ]
        c.client.klines.return_value = [
            [
                946684800000,  # 2000-01-01 00:00:00 UTC
                "100.0",  # open
                "110.0",  # high
                "90.0",  # low
                "105.0",  # close
                "123.45",  # volume
            ]
        ]

        rows = c.get_historic_data("BTCUSDT", use_cache=False)
        self.assertTrue(rows, "Expected at least one kline row")
        self.assertGreaterEqual(len(rows[0]), 6)
        self.assertAlmostEqual(rows[0][5], 123.45, places=6)

    def test_coinbase_client_returns_volume(self):
        try:
            from app.trading.cbpro_client import CBProClient
        except Exception as e:
            self.skipTest(f"Coinbase client import failed (dependency missing?): {e}")

        c = CBProClient(key="", secret="")
        c.rest_client = Mock()
        c.rest_client.get_candles.return_value = {
            "candles": [
                {
                    "close": "105.0",
                    "start": 946684800,  # seconds
                    "open": "100.0",
                    "low": "90.0",
                    "high": "110.0",
                    "volume": "123.45",
                }
            ]
        }

        rows = c.get_historic_data("BTC-USD", use_cache=False)
        self.assertTrue(rows, "Expected at least one candle row")
        self.assertEqual(len(rows[0]), 6)
        self.assertAlmostEqual(rows[0][5], 123.45, places=6)

    def test_us_stock_client_returns_volume(self):
        try:
            import pandas as pd
            from app.trading.us_stock_client import USStockClient
            import app.trading.us_stock_client as mod
        except Exception as e:
            self.skipTest(f"US stock client import failed (dependency missing?): {e}")

        # Mock yfinance download output
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [110.0],
                "Low": [90.0],
                "Close": [105.0],
                "Volume": [12345],
            },
            index=[pd.Timestamp("2020-01-01")],
        )
        mod.yf.download = Mock(return_value=df)

        c = USStockClient(tickers=["AAPL"])
        rows = c.get_historic_data("AAPL", start="2019-12-31", end="2020-01-02", use_cache=False)
        self.assertTrue(rows, "Expected at least one OHLCV row")
        self.assertEqual(len(rows[0]), 6)
        self.assertEqual(rows[0][5], 12345)


if __name__ == "__main__":
    unittest.main()


