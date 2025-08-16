#!/usr/bin/env python3
"""
Debug script to test M2 data retrieval.
"""

import sys
import os
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from data.economic_indicators_client import EconomicIndicatorsClient

def debug_m2_api():
    """Debug M2 API call."""
    print("🔍 Debugging M2 API call...")
    
    client = EconomicIndicatorsClient()
    api_key = client.fred_api_key
    
    # Test different API calls
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Test 1: Simple call
    print("\n1. Testing simple M2 call...")
    params1 = {
        'series_id': 'M2SL',
        'api_key': api_key,
        'file_type': 'json',
        'limit': 5
    }
    
    try:
        response1 = requests.get(base_url, params=params1)
        print(f"Status: {response1.status_code}")
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"Observations: {len(data1.get('observations', []))}")
            if data1.get('observations'):
                print(f"Latest: {data1['observations'][0]}")
        else:
            print(f"Error: {response1.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 2: With date range
    print("\n2. Testing M2 call with date range...")
    params2 = {
        'series_id': 'M2SL',
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '2024-01-01',
        'observation_end': '2024-12-31'
    }
    
    try:
        response2 = requests.get(base_url, params=params2)
        print(f"Status: {response2.status_code}")
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"Observations: {len(data2.get('observations', []))}")
            if data2.get('observations'):
                print(f"Latest: {data2['observations'][0]}")
        else:
            print(f"Error: {response2.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 3: Check series info
    print("\n3. Testing series info...")
    series_url = "https://api.stlouisfed.org/fred/series"
    params3 = {
        'series_id': 'M2SL',
        'api_key': api_key,
        'file_type': 'json'
    }
    
    try:
        response3 = requests.get(series_url, params=params3)
        print(f"Status: {response3.status_code}")
        if response3.status_code == 200:
            data3 = response3.json()
            series = data3.get('seriess', [])
            if series:
                print(f"Series info: {series[0]}")
        else:
            print(f"Error: {response3.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    debug_m2_api()
