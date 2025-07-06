# Buy/Sell Endpoint Testing

This directory contains test files for testing the buy and sell order functionality of the cryptocurrency trading bot.

## Test Files Overview

### 1. `../test_buy_sell_endpoints.py` - Unit Tests
- **Purpose**: Unit tests for the `place_buy_order` and `place_sell_order` methods
- **Type**: Mocked tests (no real API calls)
- **Usage**: Safe for development and CI/CD

### 2. `test_buy_sell_execution.py` - Real Trading Tests
- **Purpose**: Execute actual buy and sell orders with minimum amounts
- **Type**: Real API calls (executes actual trades)
- **Usage**: For testing real trading functionality (use with caution!)

### 3. `test_api_endpoints.py` - API Endpoint Tests
- **Purpose**: Test buy/sell endpoints that could be added to the Flask server
- **Type**: HTTP API tests
- **Usage**: For testing API endpoints (requires server to be running)

## Minimum Order Requirements

For SOL cryptocurrency:
- **Minimum SOL amount**: 0.1 SOL
- **Minimum USD amount**: $10.00 (or equivalent in SOL at current price)

## Running the Tests

### 1. Unit Tests (Safe - No Real Trading)

```bash
cd crypto-prediction
python tests/test_buy_sell_endpoints.py
```

This will run comprehensive unit tests with mocked API calls.

### 2. Real Trading Tests (Dangerous - Real Money)

⚠️ **WARNING**: This will execute real trades with real money!

```bash
cd crypto-prediction
python tests/api/test_buy_sell_execution.py
```

**Prerequisites:**
- Valid API credentials in `app/core/secret.ini`
- Sufficient USD balance for buy orders
- Sufficient SOL balance for sell orders
- Understanding that real money will be spent

**Safety Features:**
- Confirms with user before executing trades
- Uses minimum order sizes
- Shows current balances before trading
- Displays portfolio values before and after

### 3. API Endpoint Tests (Requires Server)

```bash
# First start the server
cd crypto-prediction
python start_server.py

# In another terminal, run the API tests
python tests/api/test_api_endpoints.py
```

**Note**: The buy/sell API endpoints are not yet implemented in the server. This test demonstrates the expected API structure.

## Test Configuration

### Currency
- **Default**: SOL (Solana)
- **Minimum SOL order**: 0.1 SOL
- **Minimum USD order**: $10.00

### Order Types
- **Buy orders**: Market orders with immediate-or-cancel (IOC)
- **Sell orders**: Market orders with immediate-or-cancel (IOC)

### Validation
- Balance checks before placing orders
- Order size validation
- Currency validation
- Error handling and logging

## Expected Test Results

### Unit Tests
- ✅ All tests should pass
- ✅ No real API calls made
- ✅ Comprehensive coverage of edge cases

### Real Trading Tests
- ✅ Buy order placed successfully
- ✅ Sell order placed successfully
- ✅ Order details logged
- ✅ Portfolio values updated

### API Tests
- ❌ Currently fails (endpoints not implemented)
- ✅ Demonstrates expected API structure
- ✅ Shows how to implement the endpoints

## Adding Buy/Sell Endpoints to Server

To add buy/sell endpoints to the Flask server, you would need to add these routes to `app/api/server.py`:

```python
@app.route("/api/buy", methods=["POST"])
def place_buy_order():
    """Place a buy order."""
    data = request.get_json()
    currency = data.get("currency", "SOL")
    amount_usd = data.get("amount_usd", 10.0)
    commit = data.get("commit", False)
    
    try:
        # Get wallet info
        wallet_info = get_wallet_info(client, currency)
        if not wallet_info:
            return jsonify({"error": f"No wallet found for {currency}"}), 400
        
        # Place order
        order = client.place_buy_order(
            wallet_id=wallet_info["wallet_id"],
            amount=amount_usd,
            currency=currency,
            commit=commit
        )
        
        return jsonify({"success": True, "order": order})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/sell", methods=["POST"])
def place_sell_order():
    """Place a sell order."""
    data = request.get_json()
    currency = data.get("currency", "SOL")
    amount_crypto = data.get("amount_crypto", 0.1)
    commit = data.get("commit", False)
    
    try:
        # Get wallet info
        wallet_info = get_wallet_info(client, currency)
        if not wallet_info:
            return jsonify({"error": f"No wallet found for {currency}"}), 400
        
        # Place order
        order = client.place_sell_order(
            wallet_id=wallet_info["wallet_id"],
            amount=amount_crypto,
            currency=currency,
            commit=commit
        )
        
        return jsonify({"success": True, "order": order})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
```

## Safety Recommendations

1. **Always test with small amounts first**
2. **Use simulation mode (commit=False) for development**
3. **Verify API credentials before running real tests**
4. **Check account balances before placing orders**
5. **Monitor order execution and confirmations**
6. **Keep logs of all trading activities**

## Troubleshooting

### Common Issues

1. **"Cannot connect to server"**
   - Ensure the Flask server is running
   - Check the server URL in the test script

2. **"Insufficient balance"**
   - Check your account balances
   - Verify the minimum order requirements

3. **"Invalid currency"**
   - Ensure the currency is supported (currently only SOL)
   - Check the CURS configuration in `app/core/config.py`

4. **"API credentials error"**
   - Verify `app/core/secret.ini` exists and contains valid credentials
   - Check API key permissions

### Getting Help

If you encounter issues:
1. Check the logs for detailed error messages
2. Verify your API credentials and permissions
3. Ensure sufficient account balances
4. Review the minimum order requirements 