#!/usr/bin/env python3
"""
Test script for executing actual buy and sell orders with minimum amounts.

This script tests the place_buy_order and place_sell_order methods
with real API calls using minimum order sizes for SOL cryptocurrency.

WARNING: This script will execute real trades. Use with caution!
"""

import sys
import os
import configparser
import time

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from app.trading.cbpro_client import CBProClient
from app.core.config import CURS, COMMIT


def load_api_credentials():
    """Load API credentials from secret.ini file."""
    try:
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'core', 'secret.ini')
        config.read(config_path)
        
        api_key = config["CONFIG"]["COINBASE_API_KEY"].strip('"')
        api_secret = config["CONFIG"]["COINBASE_API_SECRET"].strip('"')
        
        return api_key, api_secret
    except Exception as e:
        print(f"Error loading API credentials: {e}")
        return None, None


def get_wallet_info(client, currency):
    """Get wallet information for the specified currency."""
    try:
        wallets = client.get_wallets(cur_names=[currency])
        for wallet in wallets:
            if wallet["currency"] == currency and wallet["type"] == "ACCOUNT_TYPE_CRYPTO":
                return {
                    "wallet_id": wallet["uuid"],
                    "balance": float(wallet["available_balance"]["value"])
                }
        return None
    except Exception as e:
        print(f"Error getting wallet info: {e}")
        return None


def test_buy_order(client, wallet_id, currency="SOL"):
    """Test a buy order with minimum amount."""
    print(f"\n=== Testing BUY Order for {currency} ===")
    
    try:
        # Get current price
        current_price = client.get_cur_rate(f"{currency}-USD")
        print(f"Current {currency} price: ${current_price:.2f}")
        
        # Calculate minimum USD amount for 0.1 SOL
        min_sol_amount = 0.1
        min_usd_amount = min_sol_amount * current_price
        
        # Ensure minimum USD amount is at least $10
        if min_usd_amount < 10.0:
            min_usd_amount = 10.0
            min_sol_amount = min_usd_amount / current_price
        
        print(f"Minimum order: {min_sol_amount:.6f} {currency} (${min_usd_amount:.2f})")
        
        # Check USD balance
        usd_balance = client.get_account_balance("USD")
        print(f"USD balance: ${usd_balance:.2f}")
        
        if usd_balance < min_usd_amount:
            print(f"❌ Insufficient USD balance. Need ${min_usd_amount:.2f}, have ${usd_balance:.2f}")
            return False
        
        # Place buy order
        print(f"Placing buy order for ${min_usd_amount:.2f} worth of {currency}...")
        order = client.place_buy_order(
            wallet_id=wallet_id,
            amount=min_usd_amount,
            currency=currency,
            commit=True  # Set to True for actual execution
        )
        
        print(f"✅ Buy order placed successfully!")
        print(f"Order details: {order}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error placing buy order: {e}")
        return False


def test_sell_order(client, wallet_id, currency="SOL"):
    """Test a sell order with minimum amount."""
    print(f"\n=== Testing SELL Order for {currency} ===")
    
    try:
        # Get current price
        current_price = client.get_cur_rate(f"{currency}-USD")
        print(f"Current {currency} price: ${current_price:.2f}")
        
        # Check crypto balance
        crypto_balance = client.get_account_balance(currency)
        print(f"{currency} balance: {crypto_balance:.6f}")
        
        # Use minimum amount (0.1 SOL)
        min_sol_amount = 0.1
        
        if crypto_balance < min_sol_amount:
            print(f"❌ Insufficient {currency} balance. Need {min_sol_amount}, have {crypto_balance:.6f}")
            return False
        
        # Place sell order
        print(f"Placing sell order for {min_sol_amount} {currency}...")
        order = client.place_sell_order(
            wallet_id=wallet_id,
            amount=min_sol_amount,
            currency=currency,
            commit=True  # Set to True for actual execution
        )
        
        print(f"✅ Sell order placed successfully!")
        print(f"Order details: {order}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error placing sell order: {e}")
        return False


def main():
    """Main function to run the buy/sell tests."""
    print("🚀 Starting Buy/Sell Order Test Script")
    print("=" * 50)
    
    # Load API credentials
    api_key, api_secret = load_api_credentials()
    if not api_key or not api_secret:
        print("❌ Failed to load API credentials")
        return
    
    # Initialize client
    try:
        client = CBProClient(key=api_key, secret=api_secret)
        print("✅ Coinbase client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Test currency
    currency = "SOL"
    
    # Get wallet information
    wallet_info = get_wallet_info(client, currency)
    if not wallet_info:
        print(f"❌ Failed to get wallet info for {currency}")
        return
    
    print(f"Wallet ID: {wallet_info['wallet_id']}")
    print(f"Current {currency} balance: {wallet_info['balance']:.6f}")
    
    # Get initial portfolio value
    try:
        crypto_value, stablecoin_value = client.portfolio_value
        print(f"Initial portfolio - Crypto: ${crypto_value:.2f}, Stablecoin: ${stablecoin_value:.2f}")
    except Exception as e:
        print(f"Warning: Could not get portfolio value: {e}")
    
    # Confirm before proceeding
    print(f"\n⚠️  WARNING: This script will execute real trades!")
    print(f"Currency: {currency}")
    print(f"Minimum amounts will be used for testing")
    
    response = input("\nDo you want to proceed? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("Test cancelled by user")
        return
    
    # Test buy order
    buy_success = test_buy_order(client, wallet_info['wallet_id'], currency)
    
    if buy_success:
        print("\n⏳ Waiting 5 seconds before testing sell order...")
        time.sleep(5)
        
        # Test sell order
        sell_success = test_sell_order(client, wallet_info['wallet_id'], currency)
        
        if sell_success:
            print("\n✅ Both buy and sell tests completed successfully!")
        else:
            print("\n❌ Sell test failed")
    else:
        print("\n❌ Buy test failed")
    
    # Get final portfolio value
    try:
        crypto_value, stablecoin_value = client.portfolio_value
        print(f"\nFinal portfolio - Crypto: ${crypto_value:.2f}, Stablecoin: ${stablecoin_value:.2f}")
    except Exception as e:
        print(f"Warning: Could not get final portfolio value: {e}")
    
    print("\n🏁 Test script completed")


if __name__ == "__main__":
    main() 