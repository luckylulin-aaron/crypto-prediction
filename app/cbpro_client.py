# built-in packages
import datetime
import time
from datetime import datetime, timedelta
from typing import Any, List

# third-party packages
import numpy as np
from coinbase.rest import RESTClient

# customized packages
from config import CURS, STABLECOIN, TIMESPAN
from logger import get_logger
from util import timer


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
    def get_historic_data(self, name: str):
        """Get historic rates using Advanced Trade API.

        Args:
            name (str): The product_id or currency pair (e.g., 'BTC-USD').

        Returns:
            list: Each element is [closing_price (float), date (str, 'YYYY-MM-DD'), open_price (float), low (float), high (float)].

        Raises:
            ConnectionError: If the API call fails or the response is invalid.
        """
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
                parsed.append([closing_price, fmt_dt_str, open_price, low, high])
            # Sort by date ascending
            parsed = sorted(parsed, key=lambda x: x[1])
            # Remove today's data if present
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            parsed = [x for x in parsed if x[1] != today_str]
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

    def get_wallets(self, cur_names: List[str] = [*CURS, *STABLECOIN]):
        """
        Retrieve wallet information for pre-defined currency names using Advanced Trade API.

        Args:
            cur_names (list): List of currency names to retrieve.

        Returns:
            list: List of wallet dictionaries.
        """
        try:
            accounts = self.rest_client.get_accounts()
            wallets = [x for x in accounts.accounts if x["currency"] in cur_names]
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
            order = self.rest_client.buy(
                wallet_id, amount=str(amount), currency=currency, commit=commit
            )
        except Exception as e:
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
            order = self.rest_client.sell(
                wallet_id, amount=str(amount), currency=currency, commit=commit
            )
        except Exception as e:
            raise e

        return order
