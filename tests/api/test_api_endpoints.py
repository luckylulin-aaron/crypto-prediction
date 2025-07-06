#!/usr/bin/env python3
"""
Test script for buy and sell API endpoints.

This script demonstrates how to test buy and sell endpoints
that could be added to the Flask API server.
"""

import requests
import json
import time
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from app.core.config import CURS


class APITester:
    """Class to test buy and sell API endpoints."""
    
    def __init__(self, base_url="http://localhost:5000"):
        """Initialize the API tester.
        
        Args:
            base_url (str): Base URL of the API server.
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_server_health(self):
        """Test if the server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Server is running")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Is it running?")
            return False
    
    def test_get_status(self):
        """Test the status endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Server status: {status.get('status', 'unknown')}")
                return status
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error getting status: {e}")
            return None
    
    def test_buy_order(self, currency="SOL", amount_usdt=10.0):
        """Test a buy order endpoint (if it exists).
        
        Args:
            currency (str): Currency to buy.
            amount_usdt (float): Amount in USDT to spend.
        """
        print(f"\n=== Testing BUY Order API ===")
        print(f"Currency: {currency}")
        print(f"Amount: {amount_usdt} USDT")
        
        # This would be the endpoint if it existed
        endpoint = f"{self.base_url}/api/buy"
        
        payload = {
            "currency": currency,
            "amount_usdt": amount_usdt,
            "commit": True
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Buy order successful!")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"❌ Buy order failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server")
            return None
        except Exception as e:
            print(f"❌ Error placing buy order: {e}")
            return None
    
    def test_sell_order(self, currency="SOL", amount_crypto=0.1):
        """Test a sell order endpoint (if it exists).
        
        Args:
            currency (str): Currency to sell.
            amount_crypto (float): Amount in cryptocurrency to sell.
        """
        print(f"\n=== Testing SELL Order API ===")
        print(f"Currency: {currency}")
        print(f"Amount: {amount_crypto} {currency}")
        
        # This would be the endpoint if it existed
        endpoint = f"{self.base_url}/api/sell"
        
        payload = {
            "currency": currency,
            "amount_crypto": amount_crypto,
            "commit": True
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Sell order successful!")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"❌ Sell order failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server")
            return None
        except Exception as e:
            print(f"❌ Error placing sell order: {e}")
            return None
    
    def test_get_balances(self):
        """Test getting account balances."""
        print(f"\n=== Testing Balance API ===")
        
        # This would be the endpoint if it existed
        endpoint = f"{self.base_url}/api/balances"
        
        try:
            response = self.session.get(endpoint)
            
            if response.status_code == 200:
                balances = response.json()
                print("✅ Balances retrieved successfully!")
                print(f"Balances: {json.dumps(balances, indent=2)}")
                return balances
            else:
                print(f"❌ Balance check failed: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server")
            return None
        except Exception as e:
            print(f"❌ Error getting balances: {e}")
            return None
    
    def test_get_prices(self):
        """Test getting current prices."""
        print(f"\n=== Testing Price API ===")
        
        # This would be the endpoint if it existed
        endpoint = f"{self.base_url}/api/prices"
        
        try:
            response = self.session.get(endpoint)
            
            if response.status_code == 200:
                prices = response.json()
                print("✅ Prices retrieved successfully!")
                print(f"Prices: {json.dumps(prices, indent=2)}")
                return prices
            else:
                print(f"❌ Price check failed: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server")
            return None
        except Exception as e:
            print(f"❌ Error getting prices: {e}")
            return None


def main():
    """Main function to run API tests."""
    print("🚀 Starting API Endpoint Tests")
    print("=" * 50)
    
    # Initialize API tester
    tester = APITester()
    
    # Test server health
    if not tester.test_server_health():
        print("\n❌ Server is not running. Please start the server first.")
        print("Run: python start_server.py")
        return
    
    # Test basic endpoints
    tester.test_get_status()
    tester.test_get_balances()
    tester.test_get_prices()
    
    # Test buy/sell endpoints (these would need to be implemented in the server)
    print("\n" + "="*50)
    print("NOTE: Buy/Sell endpoints are not yet implemented in the server.")
    print("The following tests will fail until the endpoints are added.")
    print("="*50)
    
    # Test buy order
    buy_result = tester.test_buy_order(currency="SOL", amount_usdt=10.0)
    
    if buy_result:
        print("\n⏳ Waiting 5 seconds before testing sell order...")
        time.sleep(5)
        
        # Test sell order
        sell_result = tester.test_sell_order(currency="SOL", amount_crypto=0.1)
        
        if sell_result:
            print("\n✅ Both buy and sell API tests completed successfully!")
        else:
            print("\n❌ Sell API test failed")
    else:
        print("\n❌ Buy API test failed")
    
    print("\n🏁 API test script completed")


if __name__ == "__main__":
    main() 