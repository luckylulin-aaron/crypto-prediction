#!/usr/bin/env python3
"""
Test script to verify FRED API key configuration.
Run this script to check if your FRED API key is properly set up.
"""

import os
import sys
import configparser

# Add app directory to path (now in tests/api/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

def test_fred_api_key():
    """Test FRED API key configuration from different sources."""
    print("🔍 Testing FRED API Key Configuration...")
    print("=" * 50)
    
    api_key = None
    source = None
    
    # Test 1: Environment variable
    print("1. Checking environment variable...")
    env_key = os.environ.get('FRED_API_KEY')
    if env_key:
        api_key = env_key
        source = "Environment variable"
        print(f"   ✅ Found API key in environment variable")
    else:
        print(f"   ❌ No API key in environment variable")
    
    # Test 2: secret.ini file
    if not api_key:
        print("2. Checking secret.ini file...")
        try:
            config = configparser.ConfigParser()
            config_path = os.path.join('..', '..', 'app', 'core', 'secret.ini')
            if config.read(config_path):
                ini_key = config.get('CONFIG', 'FRED_API_KEY', fallback=None)
                if ini_key and ini_key != "your_fred_api_key_here":
                    # Remove quotes if present
                    api_key = ini_key.strip().strip('"').strip("'")
                    source = "secret.ini file"
                    print(f"   ✅ Found API key in secret.ini")
                else:
                    print(f"   ❌ No valid API key in secret.ini (using placeholder)")
            else:
                print(f"   ❌ Could not read secret.ini file")
        except Exception as e:
            print(f"   ❌ Error reading secret.ini: {e}")
    
    # Test 3: Economic Indicators Client
    print("3. Testing Economic Indicators Client...")
    try:
        from data.economic_indicators_client import EconomicIndicatorsClient
        client = EconomicIndicatorsClient()
        if client.fred_api_key:
            print(f"   ✅ Client successfully initialized with API key")
            print(f"   📍 Source: {source}")
            
            # Test API call
            print("4. Testing API connection...")
            try:
                # Try a simple API call
                import requests
                test_url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': 'M2SL',
                    'api_key': client.fred_api_key,
                    'file_type': 'json',
                    'limit': 1
                }
                response = requests.get(test_url, params=params, timeout=10)
                if response.status_code == 200:
                    print(f"   ✅ API connection successful")
                    data = response.json()
                    if 'observations' in data:
                        print(f"   📊 Successfully retrieved M2 data")
                    else:
                        print(f"   ⚠️  API response format unexpected")
                else:
                    print(f"   ❌ API connection failed: {response.status_code}")
                    print(f"   📝 Response: {response.text[:200]}...")
            except Exception as e:
                print(f"   ❌ API test failed: {e}")
        else:
            print(f"   ❌ Client initialized without API key")
    except Exception as e:
        print(f"   ❌ Error initializing client: {e}")
    
    print("=" * 50)
    
    if api_key:
        print("🎉 Configuration looks good!")
        print(f"📋 API Key source: {source}")
        print(f"🔑 API Key (first 8 chars): {api_key[:8]}...")
        print("\n💡 Next steps:")
        print("   - Run your trading bot to test economic indicators")
        print("   - Check logs for economic data retrieval")
    else:
        print("❌ No FRED API key found!")
        print("\n🔧 To fix this:")
        print("   1. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   2. Add it to app/core/secret.ini (from project root):")
        print("      [CONFIG]")
        print("      FRED_API_KEY = \"your_actual_api_key_here\"")
        print("   3. Or set environment variable: export FRED_API_KEY=\"your_api_key_here\"")

if __name__ == "__main__":
    test_fred_api_key()
