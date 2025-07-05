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

    def place_buy_order(
        self, wallet_id: str, amount: float, currency: str, commit: bool = False
    ):
        """Place and (optionally) execute a buy order.

        :argument
            wallet_id (str):
            amount (float):
            currency (str):
            commit (boolean): Whether to commit this transaction.

        :return
            order (dictionary):

        :raise
            (Exception):
        """
        try:
            # For now, just log the order since COMMIT is False
            self.logger.info(
                f"Would place BUY order: {amount} {currency} from wallet {wallet_id}"
            )

            # Mock order response for simulation
            order = {
                "amount": {"amount": str(amount), "currency": currency},
                "total": {"amount": str(amount), "currency": "USD"},
                "unit_price": {"amount": "1.00", "currency": "USD"},
                "status": "simulated",
            }

            # TODO: Implement actual Coinbase Advanced Trade API call
            # The correct method might be something like:
            # order = self.rest_client.create_order(...)

        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            raise e

        return order

    def place_sell_order(
        self, wallet_id: str, amount: float, currency: str, commit: bool = False
    ):
        """Place and (optionally) execute a sell order.

        :argument
            wallet_id (str):
            amount (float):
            currency (str):
            commit (boolean): Whether to commit this transaction.

        :return
            order (dictionary):

        :raise
            (Exception):
        """
        try:
            # For now, just log the order since COMMIT is False
            self.logger.info(
                f"Would place SELL order: {amount} {currency} from wallet {wallet_id}"
            )

            # Mock order response for simulation
            order = {
                "amount": {"amount": str(amount), "currency": currency},
                "subtotal": {"amount": str(amount), "currency": "USD"},
                "unit_price": {"amount": "1.00", "currency": "USD"},
                "status": "simulated",
            }

            # TODO: Implement actual Coinbase Advanced Trade API call
            # The correct method might be something like:
            # order = self.rest_client.create_order(...)

        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            raise e

        return order
