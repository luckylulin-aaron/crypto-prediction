#!/usr/bin/env python3
"""
Simple test to verify FRED API key works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from data.economic_indicators_client import EconomicIndicatorsClient

def test_fred_api():
    """Test FRED API with the configured key."""
    print("🧪 Testing FRED API...")
    
    try:
        # Initialize client
        client = EconomicIndicatorsClient()
        print(f"✅ Client initialized")
        print(f"🔑 API Key: {client.fred_api_key[:8]}...")
        
        # Test M2 data retrieval
        print("\n📊 Testing M2 Money Supply data...")
        m2_data = client.get_m2_money_supply(days_back=365)  # Use 1 year for monthly data
        if not m2_data.empty:
            print(f"✅ Successfully retrieved {len(m2_data)} M2 data points")
            print(f"📅 Latest date: {m2_data['date'].iloc[-1].strftime('%Y-%m-%d')}")
            print(f"💰 Latest value: ${m2_data['value'].iloc[-1]:,.0f}B")
        else:
            print("❌ Failed to retrieve M2 data")
        
        # Test ON RRP data retrieval
        print("\n🏦 Testing Overnight Reverse Repo data...")
        rrp_data = client.get_overnight_reverse_repo(days_back=7)
        if not rrp_data.empty:
            print(f"✅ Successfully retrieved {len(rrp_data)} ON RRP data points")
            print(f"📅 Latest date: {rrp_data['date'].iloc[-1].strftime('%Y-%m-%d')}")
            print(f"💰 Latest value: ${rrp_data['value'].iloc[-1]:,.0f}B")
        else:
            print("❌ Failed to retrieve ON RRP data")
        
        # Test liquidity indicators
        print("\n📈 Testing liquidity indicators...")
        indicators = client.get_liquidity_indicators()
        if indicators:
            print(f"✅ Successfully calculated {len(indicators)} indicators")
            for key, value in indicators.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ Failed to calculate liquidity indicators")
        
        # Test trading signals
        print("\n🎯 Testing trading signals...")
        signals = client.get_trading_signals()
        if signals:
            print(f"✅ Successfully generated trading signals")
            for key, value in signals.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Failed to generate trading signals")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fred_api()
