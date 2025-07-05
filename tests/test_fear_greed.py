#!/usr/bin/env python3
"""
Test script for Fear & Greed Index client.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.data.fear_greed_client import FearGreedClient


def test_fear_greed_client():
    """Test the Fear & Greed Index client functionality."""
    
    print("🧪 Testing Fear & Greed Index Client...")
    
    # Initialize client
    client = FearGreedClient()
    
    # Test 1: Get current fear & greed index
    print("\n1. Testing current fear & greed index...")
    current = client.get_current_fear_greed()
    if current:
        print(f"✅ Current Fear & Greed Index: {current['value']} ({current['value_classification']})")
    else:
        print("❌ Failed to get current fear & greed index")
    
    # Test 2: Get historical data (last 30 days)
    print("\n2. Testing historical fear & greed data...")
    historical = client.get_historical_fear_greed(days=30)
    if historical:
        print(f"✅ Retrieved {len(historical)} days of historical data")
        print(f"   Date range: {historical[-1]['date']} to {historical[0]['date']}")
        print(f"   Sample values:")
        for i, record in enumerate(historical[:5]):
            print(f"     {record['date']}: {record['value']} ({record['value_classification']})")
    else:
        print("❌ Failed to get historical fear & greed data")
    
    # Test 3: Get data for specific date
    print("\n3. Testing specific date lookup...")
    if historical:
        test_date = historical[0]['date']
        specific = client.get_fear_greed_by_date(test_date)
        if specific:
            print(f"✅ Found data for {test_date}: {specific['value']} ({specific['value_classification']})")
        else:
            print(f"❌ No data found for {test_date}")
    
    # Test 4: Test color mapping
    print("\n4. Testing color mapping...")
    classifications = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    for classification in classifications:
        color = client.get_classification_color(classification)
        print(f"   {classification}: {color}")
    
    # Test 5: Test description mapping
    print("\n5. Testing description mapping...")
    values = [10, 30, 50, 70, 90]
    for value in values:
        desc = client.get_classification_description(value)
        print(f"   Value {value}: {desc}")
    
    print("\n✅ Fear & Greed Index client test completed!")

if __name__ == "__main__":
    test_fear_greed_client() 