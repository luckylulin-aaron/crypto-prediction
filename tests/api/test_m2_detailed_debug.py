#!/usr/bin/env python3
"""
Detailed debug script for M2 data.
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from data.economic_indicators_client import EconomicIndicatorsClient

def debug_m2_detailed():
    """Detailed debug of M2 data retrieval."""
    print("🔍 Detailed M2 Debug...")
    
    client = EconomicIndicatorsClient()
    
    # Test M2 with more detailed logging
    print("\n📊 Testing M2 with detailed logging...")
    m2_data = client.get_m2_money_supply(days_back=365)
    
    print(f"M2 data type: {type(m2_data)}")
    print(f"M2 data empty: {m2_data.empty}")
    print(f"M2 data shape: {m2_data.shape}")
    
    if not m2_data.empty:
        print(f"M2 data columns: {m2_data.columns.tolist()}")
        print(f"M2 data head:\n{m2_data.head()}")
    else:
        print("M2 data is empty")

if __name__ == "__main__":
    debug_m2_detailed()
