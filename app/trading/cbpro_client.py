# built-in packages
import datetime
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional

# third-party packages
import numpy as np
from coinbase.rest import RESTClient

# customized packages
from core.config import CURS, STABLECOIN, TIMESPAN
from core.logger import get_logger
from db.database import db_manager
from utils.util import timer


class CBProClient:
    def __init__(self, key: str = "", secret: str = ""):
        self.logger = get_logger(__name__)
        # Create the RESTClient from coinbase-advanced-py
        if len(key) > 0 and len(secret) > 0:
            self.rest_client = RESTClient(api_key=key, api_secret=secret)
        else:
            self.rest_client = RESTClient()  # Will use env vars if set

    @timer
    def get_cur_rate(self, name: str):
        """For a given currency name, return its latest price using Advanced Trade API."""
        try:
            trades = self.rest_client.get_market_trades(
                name, limit=1
            )  # The first trade is the most recent
            res = trades["trades"][0]["price"]
        except Exception as e:
            raise ConnectionError(f"cannot get current rate for {name}: {e}")
        return float(res)

    @timer
    def get_historic_data(self, name: str, use_cache: bool = True):
        """Get historic rates using Advanced Trade API with database caching.

        Args:
            name (str): The product_id or currency pair (e.g., 'BTC-USD').
            use_cache (bool): Whether to use cached data if available and fresh.

        Returns:
            list: Each element is [closing_price (float), date (str, 'YYYY-MM-DD'), open_price (float), low (float), high (float), volume (float)].

        Raises:
            ConnectionError: If the API call fails or the response is invalid.
        """
        # Try to get data from cache first
        if use_cache:
            cached_data = db_manager.get_historical_data(name, TIMESPAN)
            if cached_data and db_manager.is_data_fresh(name, max_age_hours=24):
                self.logger.info(
                    f"Using cached data for {name} ({len(cached_data)} records)"
                )
                return cached_data
            elif cached_data:
                self.logger.info(
                    f"Cached data for {name} is stale, fetching fresh data"
                )
            else:
                self.logger.info(f"No cached data found for {name}, fetching from API")

        # Fetch fresh data from API
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=TIMESPAN)
            res = self.rest_client.get_candles(
                name,
                granularity="ONE_DAY",
                start=int(start.timestamp()),
                end=int(end.timestamp()),
            )  # res: <class 'coinbase.rest.types.product_types.GetProductCandlesResponse'>
            # processing
            candles = res["candles"]
            parsed = []
            for item in candles:
                # item fields are strings, convert as needed
                closing_price = float(item["close"])
                fmt_dt_str = datetime.utcfromtimestamp(int(item["start"])).strftime(
                    "%Y-%m-%d"
                )
                open_price = float(item["open"])
                low = float(item["low"])
                high = float(item["high"])
                # Try to get volume data, but handle cases where it's not available
                try:
                    volume = float(item["volume"])
                except (KeyError, ValueError, TypeError):
                    volume = 0.0  # Default to 0 if volume data is not available
                # append
                parsed.append(
                    [closing_price, fmt_dt_str, open_price, low, high, volume]
                )
            # Sort by date ascending
            parsed = sorted(parsed, key=lambda x: x[1])
            # Remove today's data if present
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            parsed = [x for x in parsed if x[1] != today_str]

            # Store data in database for future use
            if use_cache and parsed:
                db_manager.store_historical_data(name, parsed)

            return parsed

        except Exception as e:
            raise ConnectionError(f"cannot get historic rates for {name}: {e}")

    @property
    def portfolio_value(self):
        """
        Computes portfolio value for an account using Advanced Trade API, where
        v_crypto is sum of CURS, v_stable is sum of STABLECOIN.

        Returns:
            tuple: (v_crypto, v_stable): Portfolio value in USDT for crypto and stablecoin.

        Raises:
        """
        wallets = self.get_wallets()
        # self.logger.info(f'wallets: {wallets}')
        v_crypto = v_stable = 0

        for item in wallets:
            try:
                delta_v = float(item["available_balance"]["value"])
                if item["currency"] in CURS:
                    coin_cur_rate = self.get_cur_rate(name=item["currency"] + "-USD")
                    v_crypto += coin_cur_rate * delta_v
                elif item["currency"] in STABLECOIN:
                    v_stable += delta_v
            except Exception as e:
                self.logger.error(
                    f"Exception processing wallet item: {item}, error: {e}"
                )
                continue

        return np.round(v_crypto, 2), np.round(v_stable, 2)

    def get_wallets(self, cur_names: Optional[List[str]] = None):
        """
        Retrieve wallet information for currency names using Advanced Trade API.

        Args:
            cur_names (list): List of currency names to retrieve. If None, returns all accounts.

        Returns:
            list: List of wallet dictionaries.
        """
        try:
            accounts = self.rest_client.get_accounts()
            all_currencies = sorted([x["currency"] for x in accounts.accounts])
            self.logger.info(f"All available currencies in account: {all_currencies}")

            if cur_names is None:
                # Return all accounts if no specific currencies requested
                wallets = accounts.accounts
                self.logger.info(f"Returning all {len(wallets)} accounts")
            else:
                # Filter by requested currencies
                wallets = [x for x in accounts.accounts if x["currency"] in cur_names]
                self.logger.info(
                    f"Filtered to {len(wallets)} accounts for currencies: {cur_names}"
                )

            # Log details of returned wallets
            for wallet in wallets:
                balance = float(wallet["available_balance"]["value"])
                if balance > 0:  # Only log wallets with non-zero balance
                    self.logger.info(
                        f"Wallet: {wallet['currency']} - Balance: {balance:.3f}"
                    )

            return wallets
        except Exception as e:
            self.logger.error(f"Exception when calling get_accounts: {e}")
            return []

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get the status of an order.

        Args:
            order_id (str): The order ID to check.

        Returns:
            dict: Order status information or None if error.

        Raises:
            Exception: If there is an error getting the order status.
        """
        try:
            order = self.rest_client.get_order(order_id)
            return order
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.

        Args:
            order_id (str): The order ID to cancel.

        Returns:
            bool: True if order was cancelled successfully, False otherwise.

        Raises:
            Exception: If there is an error cancelling the order.
        """
        try:
            self.rest_client.cancel_order(order_id)
            self.logger.info(f"Successfully cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_account_balance(self, currency: str) -> float:
        """Get the available balance for a specific currency.

        Args:
            currency (str): The currency to get balance for.

        Returns:
            float: Available balance for the currency.

        Raises:
            ValueError: If currency is invalid.
            Exception: If there is an error getting the balance.
        """
        try:
            if not currency:
                raise ValueError("Currency cannot be empty")
            
            wallets = self.get_wallets(cur_names=[currency])
            
            for wallet in wallets:
                if wallet["currency"] == currency:
                    return float(wallet["available_balance"]["value"])
            
            # If no wallet found for the currency, return 0
            self.logger.warning(f"No wallet found for currency: {currency}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting balance for {currency}: {e}")
            raise e

    def validate_order_size(self, amount: float, currency: str, side: str) -> bool:
        """Validate if the order size meets minimum requirements.

        Args:
            amount (float): The order amount.
            currency (str): The currency being traded.
            side (str): 'BUY' or 'SELL'.

        Returns:
            bool: True if order size is valid, False otherwise.
        """
        try:
            # Minimum order sizes (these are approximate and may vary)
            min_order_sizes = {
                "BTC": 0.001,  # Minimum 0.001 BTC
                "ETH": 0.01,   # Minimum 0.01 ETH
                "SOL": 0.1,    # Minimum 0.1 SOL
                "LTC": 0.1,    # Minimum 0.1 LTC
                "BCH": 0.01,   # Minimum 0.01 BCH
                "ADA": 1.0,    # Minimum 1 ADA
                "DOT": 0.1,    # Minimum 0.1 DOT
                "LINK": 0.1,   # Minimum 0.1 LINK
            }
            
            # For USDT amounts (buy orders), convert to crypto amount first
            if side == "BUY":
                product_id = f"{currency}-USDT"
                current_price = self.get_cur_rate(product_id)
                crypto_amount = amount / current_price
            else:
                crypto_amount = amount
            
            min_size = min_order_sizes.get(currency, 0.01)  # Default minimum
            
            if crypto_amount < min_size:
                self.logger.warning(
                    f"Order size {crypto_amount} {currency} is below minimum {min_size} {currency}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order size: {e}")
            return False

    def place_buy_order(
        self, wallet_id: str, amount: float, currency: str, commit: bool = False
    ):
        """Place and (optionally) execute a buy order.

        Args:
            wallet_id (str): The wallet ID to use for the order.
            amount (float): The amount to buy in USDT.
            currency (str): The cryptocurrency to buy (e.g., 'BTC', 'ETH').
            commit (bool): Whether to commit this transaction.

        Returns:
            dict: Order response from the API.

        Raises:
            ValueError: If input parameters are invalid.
            ConnectionError: If there is a network or API error.
            Exception: If there is any other error placing the order.
        """
        try:
            if not commit:
                # For simulation mode, just log the order
                self.logger.info(
                    f"Would place BUY order: {amount} USDT for {currency} from wallet {wallet_id}"
                )

                # Mock order response for simulation
                order = {
                    "amount": {"amount": str(amount), "currency": currency},
                    "total": {"amount": str(amount), "currency": "USDT"},
                    "unit_price": {"amount": "1.00", "currency": "USDT"},
                    "status": "simulated",
                }
            else:
                # Validate inputs
                if amount <= 0:
                    raise ValueError("Amount must be positive")
                if not currency or currency not in CURS:
                    raise ValueError(f"Invalid currency: {currency}")
                
                # Check if we have enough USDT balance
                usdt_balance = self.get_account_balance("USDT")
                if usdt_balance < amount:
                    raise ValueError(f"Insufficient USDT balance. Required: {amount}, Available: {usdt_balance}")
                
                # Validate order size
                if not self.validate_order_size(amount, currency, "BUY"):
                    raise ValueError(f"Order size {amount} USDT for {currency} is below minimum requirements")
                
                # Get current price for the currency pair
                product_id = f"{currency}-USDT"
                current_price = self.get_cur_rate(product_id)
                
                # Calculate the quantity to buy based on USDT amount
                quantity = amount / current_price
                
                # Round USDT amount to 2 decimal places to avoid precision errors
                rounded_amount = round(amount, 2)
                
                # Create the order using Coinbase Advanced Trade API
                # Using market order with immediate-or-cancel (IOC) for immediate execution
                import uuid
                client_order_id = f"buy_{currency}_{uuid.uuid4().hex[:8]}"
                
                order = self.rest_client.create_order(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    side="BUY",
                    order_configuration={
                        "market_market_ioc": {
                            "quote_size": str(rounded_amount)
                        }
                    }
                )
                
                # Check if the order was successful
                if hasattr(order, 'success') and order.success:
                    self.logger.info(
                        f"Placed BUY order: {quantity:.6f} {currency} for {rounded_amount} USDT at ~{current_price:.2f} USDT"
                    )
                else:
                    error_msg = getattr(order, 'error_response', 'Unknown error')
                    if hasattr(error_msg, 'message'):
                        error_msg = error_msg.message
                    raise Exception(f"Buy order failed: {error_msg}")

        except ValueError as e:
            self.logger.error(f"Validation error in buy order: {e}")
            raise e
        except ConnectionError as e:
            self.logger.error(f"Connection error in buy order: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            raise e

        return order

    def place_sell_order(
        self, wallet_id: str, amount: float, currency: str, commit: bool = False
    ):
        """Place and (optionally) execute a sell order.

        Args:
            wallet_id (str): The wallet ID to use for the order.
            amount (float): The amount to sell in cryptocurrency units.
            currency (str): The cryptocurrency to sell (e.g., 'BTC', 'ETH').
            commit (bool): Whether to commit this transaction.

        Returns:
            dict: Order response from the API.

        Raises:
            ValueError: If input parameters are invalid.
            ConnectionError: If there is a network or API error.
            Exception: If there is any other error placing the order.
        """
        try:
            if not commit:
                # For simulation mode, just log the order
                self.logger.info(
                    f"Would place SELL order: {amount} {currency} from wallet {wallet_id}"
                )

                # Mock order response for simulation
                order = {
                    "amount": {"amount": str(amount), "currency": currency},
                    "subtotal": {"amount": str(amount), "currency": "USDT"},
                    "unit_price": {"amount": "1.00", "currency": "USDT"},
                    "status": "simulated",
                }
            else:
                # Validate inputs
                if amount <= 0:
                    raise ValueError("Amount must be positive")
                if not currency or currency not in CURS:
                    raise ValueError(f"Invalid currency: {currency}")
                
                # Check if we have enough crypto balance
                crypto_balance = self.get_account_balance(currency)
                if crypto_balance < amount:
                    raise ValueError(f"Insufficient {currency} balance. Required: {amount}, Available: {crypto_balance}")
                
                # Validate order size
                if not self.validate_order_size(amount, currency, "SELL"):
                    raise ValueError(f"Order size {amount} {currency} is below minimum requirements")
                
                # Get current price for the currency pair
                product_id = f"{currency}-USDT"
                current_price = self.get_cur_rate(product_id)
                
                # Create the order using Coinbase Advanced Trade API
                # Using market order with immediate-or-cancel (IOC) for immediate execution
                import uuid
                client_order_id = f"sell_{currency}_{uuid.uuid4().hex[:8]}"
                
                order = self.rest_client.create_order(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    side="SELL",
                    order_configuration={
                        "market_market_ioc": {
                            "base_size": str(amount)
                        }
                    }
                )
                
                # Check if the order was successful
                if hasattr(order, 'success_response'):
                    estimated_usdt_value = amount * current_price
                    self.logger.info(
                        f"Placed SELL order: {amount:.6f} {currency} for ~{estimated_usdt_value:.2f} USDT at ~{current_price:.2f} USDT"
                    )
                else:
                    error_msg = getattr(order, 'error_response', 'Unknown error')
                    if hasattr(error_msg, 'message'):
                        error_msg = error_msg.message
                    raise Exception(f"Sell order failed: {error_msg}")

        except ValueError as e:
            self.logger.error(f"Validation error in sell order: {e}")
            raise e
        except ConnectionError as e:
            self.logger.error(f"Connection error in sell order: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            raise e

        return order
