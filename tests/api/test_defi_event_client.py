import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))

import unittest

from app.data.defi_event_client import DefiEventClient


class TestDefiEventClient(unittest.TestCase):
    def setUp(self):
        self.client = DefiEventClient()

    def test_fetch_protocols(self):
        protocols = self.client.fetch_protocols()
        if protocols is None:
            self.fail("fetch_protocols() returned None")
            return
        self.assertIsInstance(protocols, list)
        self.assertGreater(len(protocols), 0)
        # Check at least one protocol has both TVL and market cap
        found = False
        for p in protocols:
            tvl = p.get("tvl")
            mcap = p.get("mcap")
            if tvl is not None and mcap is not None and tvl > 0:
                r = mcap / tvl
                print(f"Protocol: {p.get('name')} ({p.get('symbol')}) | TVL: {tvl} | MCAP: {mcap} | R: {r:.3f}")
                found = True
                break
        self.assertTrue(found, "No protocol with both TVL and market cap found.")

if __name__ == "__main__":
    unittest.main() 