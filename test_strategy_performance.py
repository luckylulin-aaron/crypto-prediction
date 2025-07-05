#!/usr/bin/env python3
"""
Test script for strategy performance visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from trading.trader_driver import TraderDriver
from core.config import *

def test_strategy_performance():
    """Test the strategy performance collection."""
    
    # Create a simple trader driver with minimal parameters
    t_driver = TraderDriver(
        name="SOL",
        init_amount=10000,
        overall_stats=["MA-SELVES"],  # Just one strategy
        cur_coin=10.0,
        tol_pcts=[0.1],  # Just one value
        ma_lengths=[20],  # Just one value
        ema_lengths=[12],  # Just one value
        bollinger_mas=[20],  # Just one value
        bollinger_tols=[2],  # Just one value
        buy_pcts=[0.5],  # Just one value
        sell_pcts=[0.5],  # Just one value
        mode="normal",
    )
    
    print(f"Created trader driver with {len(t_driver.traders)} traders")
    
    # Create some dummy data
    dummy_data = [
        (100.0, "2024-01-01", 100.0, 99.0, 101.0),
        (101.0, "2024-01-02", 101.0, 100.0, 102.0),
        (102.0, "2024-01-03", 102.0, 101.0, 103.0),
    ]
    
    # Feed data
    t_driver.feed_data(dummy_data)
    
    print("Fed data to traders")
    
    # Try to get strategy performance
    try:
        strategy_performance = t_driver.get_all_strategy_performance()
        print(f"Successfully collected {len(strategy_performance)} strategy performance records")
        if strategy_performance:
            print(f"First record: {strategy_performance[0]}")
    except Exception as e:
        print(f"Error collecting strategy performance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_performance() 