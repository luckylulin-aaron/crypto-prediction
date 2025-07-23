import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))
import unittest

from trading.us_stock_client import USStockClient


class TestUSStockClient(unittest.TestCase):
    def setUp(self):
        self.client = USStockClient(tickers=["AAPL"])

    def test_get_historic_data(self):
        data = self.client.get_historic_data("AAPL")
        print(f"\nAAPL Data Length: {len(data)}")
        print(f"First 3 records: {data[:3]}")
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        # Check structure: [close, date, open, low, high, volume]
        first = data[0]
        print(f"First record structure: {first}")
        self.assertEqual(len(first), 6)
        self.assertIsInstance(first[0], float)  # close
        self.assertIsInstance(first[1], str)    # date
        self.assertIsInstance(first[2], float)  # open
        self.assertIsInstance(first[3], float)  # low
        self.assertIsInstance(first[4], float)  # high
        self.assertIsInstance(first[5], float)  # volume

    def test_get_all_historic_data(self):
        all_data = self.client.get_all_historic_data()
        self.assertIn("AAPL", all_data)
        self.assertIsInstance(all_data["AAPL"], list)

if __name__ == "__main__":
    unittest.main() 