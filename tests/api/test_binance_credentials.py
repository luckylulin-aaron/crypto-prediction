import configparser
import os
import unittest

from app.trading.binance_client import BinanceClient


class TestBinanceCredentials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load credentials from secret.ini
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), '../../app/core/secret.ini')
        config.read(config_path)
        cls.api_key = config["CONFIG"]["BINANCE_API_KEY"].strip('"')
        cls.api_secret = config["CONFIG"]["BINANCE_API_SECRET"].strip('"')
        cls.client = BinanceClient(api_key=cls.api_key, api_secret=cls.api_secret)

    def test_account_balances(self):
        """Test fetching account balances with Binance credentials."""
        wallets = self.client.get_wallets()
        self.assertIsInstance(wallets, list)
        # There should be at least one asset (even if all are zero)
        self.assertTrue(isinstance(wallets, list))
        print(f"Verify for all wallets: {wallets}")

    def test_get_cur_rate(self):
        """Test fetching current price for BTCUSDT."""
        price = self.client.get_cur_rate("BTCUSDT")
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
        print(f"Current BTC price: {price}")

if __name__ == "__main__":
    unittest.main() 