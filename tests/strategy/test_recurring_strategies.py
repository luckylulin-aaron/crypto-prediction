#!/usr/bin/env python3
"""
Test script for recurring investment strategies.
"""

import datetime
import os
import sys
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL
from app.trading.strategies import (
    strategy_simple_recurring_investment,
    strategy_weighted_recurring_investment,
)


def create_mock_trader():
    """Create a mock trader object for testing."""
    trader = Mock()

    # Mock wallet
    trader.wallet = {"USD": 10000, "crypto": 100}

    # Mock methods
    trader._execute_one_buy = Mock(return_value=True)
    trader._execute_one_sell = Mock(return_value=True)
    trader._record_history = Mock()

    # Mock strategy dictionary
    trader.strat_dct = {"SIMPLE-RECURRING": [], "WEIGHTED-RECURRING": []}

    # Mock moving averages for weighted strategy
    trader.moving_averages = {"20": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}

    # Mock initial investment value
    trader.initial_investment_value = 10000

    return trader


def test_simple_recurring_strategy():
    """Test the simple recurring investment strategy."""
    print("Testing Simple Recurring Investment Strategy...")

    base_date = datetime.datetime(2024, 1, 1)

    # Test 1: First investment (should buy)
    print("  Test 1: First investment")
    trader1 = create_mock_trader()
    trader1.wallet = {"USD": 10000, "crypto": 0}  # Only USD, no crypto
    buy, sell = strategy_simple_recurring_investment(
        trader1,
        100.0,
        base_date,
        investment_interval_days=7,
        profit_threshold=0.10,
        loss_threshold=0.10,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == True, "First investment should buy"
    assert sell == False, "First investment should not sell"

    # Test 2: Investment too soon (should not invest)
    print("  Test 2: Investment too soon")
    trader2 = create_mock_trader()
    trader2.wallet = {"USD": 9000, "crypto": 1}  # Simulate after first buy
    trader2.last_investment_date = base_date  # Set last investment date
    buy, sell = strategy_simple_recurring_investment(
        trader2,
        100.0,
        base_date + datetime.timedelta(days=3),
        investment_interval_days=7,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == False, "Investment too soon should not buy"
    assert sell == False, "Investment too soon should not sell"

    # Test 3: Investment after interval (should buy)
    print("  Test 3: Investment after interval")
    trader3 = create_mock_trader()
    trader3.wallet = {"USD": 9000, "crypto": 1}  # Simulate after first buy
    trader3.last_investment_date = base_date  # Set last investment date
    buy, sell = strategy_simple_recurring_investment(
        trader3,
        100.0,
        base_date + datetime.timedelta(days=7),
        investment_interval_days=7,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == True, "Investment after interval should buy"
    assert sell == False, "Investment after interval should not sell"

    print("  ✓ Simple recurring strategy tests passed!")


def test_weighted_recurring_strategy():
    """Test the weighted recurring investment strategy."""
    print("Testing Weighted Recurring Investment Strategy...")

    base_date = datetime.datetime(2024, 1, 1)

    # Test 1: First investment (should buy)
    print("  Test 1: First investment")
    trader1 = create_mock_trader()
    buy, sell = strategy_weighted_recurring_investment(
        trader1, "20", 100.0, base_date, investment_interval_days=7
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == True, "First investment should buy"
    assert sell == False, "First investment should not sell"

    # Test 2: Investment too soon (should not invest)
    print("  Test 2: Investment too soon")
    trader2 = create_mock_trader()
    trader2.last_weighted_investment_date = base_date  # Set last investment date
    buy, sell = strategy_weighted_recurring_investment(
        trader2,
        "20",
        100.0,
        base_date + datetime.timedelta(days=3),
        investment_interval_days=7,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == False, "Investment too soon should not buy"
    assert sell == False, "Investment too soon should not sell"

    # Test 3: Investment after interval (should buy)
    print("  Test 3: Investment after interval")
    trader3 = create_mock_trader()
    trader3.last_weighted_investment_date = base_date  # Set last investment date
    buy, sell = strategy_weighted_recurring_investment(
        trader3,
        "20",
        100.0,
        base_date + datetime.timedelta(days=7),
        investment_interval_days=7,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == True, "Investment after interval should buy"
    assert sell == False, "Investment after interval should not sell"

    # Test 4: Price below MA (should still buy, but with different weighting)
    print("  Test 4: Price below MA")
    trader4 = create_mock_trader()
    trader4.last_weighted_investment_date = base_date  # Set last investment date
    buy, sell = strategy_weighted_recurring_investment(
        trader4,
        "20",
        95.0,
        base_date + datetime.timedelta(days=14),
        investment_interval_days=7,
    )
    print(f"    Result: buy={buy}, sell={sell}")
    assert buy == True, "Price below MA should still buy"
    assert sell == False, "Price below MA should not sell"

    print("  ✓ Weighted recurring strategy tests passed!")


def test_strategy_registry():
    """Test that strategies are properly registered."""
    print("Testing Strategy Registry...")

    from app.trading.strategies import STRATEGY_REGISTRY

    # Check if strategies are in registry
    assert "SIMPLE-RECURRING" in STRATEGY_REGISTRY, "SIMPLE-RECURRING not in registry"
    assert (
        "WEIGHTED-RECURRING" in STRATEGY_REGISTRY
    ), "WEIGHTED-RECURRING not in registry"

    # Check if strategies are callable
    assert callable(
        STRATEGY_REGISTRY["SIMPLE-RECURRING"]
    ), "SIMPLE-RECURRING not callable"
    assert callable(
        STRATEGY_REGISTRY["WEIGHTED-RECURRING"]
    ), "WEIGHTED-RECURRING not callable"

    print("  ✓ Strategy registry tests passed!")


def test_config_integration():
    """Test that strategies are in the config."""
    print("Testing Config Integration...")

    from app.core.config import STRATEGIES

    # Check if strategies are in config
    assert "SIMPLE-RECURRING" in STRATEGIES, "SIMPLE-RECURRING not in config"
    assert "WEIGHTED-RECURRING" in STRATEGIES, "WEIGHTED-RECURRING not in config"

    print("  ✓ Config integration tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Recurring Investment Strategies")
    print("=" * 60)

    try:
        test_simple_recurring_strategy()
        test_weighted_recurring_strategy()
        test_strategy_registry()
        test_config_integration()

        print("\n" + "=" * 60)
        print(
            "🎉 All tests passed! Recurring investment strategies are working correctly."
        )
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
