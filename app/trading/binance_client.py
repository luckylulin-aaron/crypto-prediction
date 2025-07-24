# Requires: pip install binance-connector
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
from binance.spot import Spot
from core.config import CURS, STABLECOIN, TIMESPAN
from core.logger import get_logger
from db.database import db_manager
from utils.util import timer


class BinanceClient:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.logger = get_logger(__name__)
        if api_key and api_secret:
            self.client = Spot(api_key=api_key, api_secret=api_secret)
        else:
            self.client = Spot()

    def get_cur_rate(self, symbol: str) -> float:
        """
        Get the latest price for a given symbol (e.g., 'BTCUSDT').

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        Returns:
            float: Latest price.
        Raises:
            ConnectionError: If the API call fails.
        """
        try:
            res = self.client.ticker_price(symbol)
            return float(res["price"])
        except Exception as e:
            raise ConnectionError(f"Cannot get current rate for {symbol}: {e}")

    @timer
    def get_historic_data(self, symbol: str, use_cache: bool = True) -> list:
        """
        Get historical daily klines for a symbol, with database caching.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            use_cache (bool): Whether to use cached data if available and fresh.
        Returns:
            list: Each element is [closing_price, date, open_price, low, high, volume].
        Raises:
            ConnectionError: If the API call fails.
        """
        if use_cache:
            cached_data = db_manager.get_historical_data(symbol, TIMESPAN)
            if cached_data and db_manager.is_data_fresh(symbol, max_age_hours=72):
                self.logger.info(
                    f"Using cached data for {symbol} ({len(cached_data)} records)"
                )
                return cached_data
            elif cached_data:
                self.logger.info(
                    f"Cached data for {symbol} is stale, fetching fresh data"
                )
            else:
                self.logger.info(
                    f"No cached data found for {symbol}, fetching from API"
                )

        try:
            end = int(time.time() * 1000)
            start = int(
                (datetime.utcnow() - timedelta(days=TIMESPAN)).timestamp() * 1000
            )
            klines = self.client.klines(symbol, "1d", startTime=start, endTime=end)
            parsed = []
            for k in klines:
                open_time = k[0] // 1000
                fmt_dt_str = datetime.utcfromtimestamp(open_time).strftime("%Y-%m-%d")
                open_price = float(k[1])
                high = float(k[2])
                low = float(k[3])
                closing_price = float(k[4])
                volume = float(k[5])
                parsed.append(
                    [closing_price, fmt_dt_str, open_price, low, high, volume]
                )
            parsed = sorted(parsed, key=lambda x: x[1])
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            parsed = [x for x in parsed if x[1] != today_str]
            if use_cache and parsed:
                db_manager.store_historical_data(symbol, parsed)
            return parsed
        except Exception as e:
            raise ConnectionError(f"Cannot get historic rates for {symbol}: {e}")

    @property
    def portfolio_value(self):
        """
        Computes portfolio value for the Binance account.

        Returns:
            tuple: (v_crypto, v_stable): Portfolio value in USDT for crypto and stablecoin.
        Raises:
            Exception: If API call fails.
        """
        wallets = self.get_wallets()
        v_crypto = v_stable = 0
        for item in wallets:
            try:
                free = float(item["free"])
                asset = item["asset"]
                asset_name_clean = asset.replace("LD", "")
                if asset_name_clean in CURS:
                    symbol = asset_name_clean + "USDT"
                    coin_cur_rate = self.get_cur_rate(symbol=symbol)
                    v_crypto += coin_cur_rate * free
                elif asset_name_clean in STABLECOIN:
                    v_stable += free
            except Exception as e:
                self.logger.error(
                    f"Exception processing wallet item: {item}, error: {e}"
                )
                continue
        return np.round(v_crypto, 2), np.round(v_stable, 2)

    def get_wallets(self, asset_names: Optional[List[str]] = None) -> list:
        """
        Retrieve wallet information for asset names.

        Args:
            asset_names (list): List of asset names to retrieve. If None, returns all assets.
        Returns:
            list: List of asset balance dictionaries.
        Raises:
            Exception: If API call fails.
        """
        try:
            account = self.client.account()
            balances = account["balances"]
            if asset_names is None:
                wallets = [b for b in balances if float(b["free"]) > 0]
            else:
                wallets = [
                    b
                    for b in balances
                    if b["asset"] in asset_names and float(b["free"]) > 0
                ]
            for wallet in wallets:
                asset_name_clean = wallet["asset"].replace("LD", "")
                asset_balance = float(wallet["free"])
                # log it
                if asset_balance > 0.05:
                    self.logger.info(
                        f"Wallet: {asset_name_clean} - Balance: {asset_balance:.3f}"
                    )
            return wallets
        except Exception as e:
            self.logger.error(f"Exception when calling get_wallets: {e}")
            return []

    def get_account_balance(self, asset: str) -> float:
        """
        Get the available balance for a specific asset.

        Args:
            asset (str): The asset to get balance for.
        Returns:
            float: Available balance for the asset.
        Raises:
            Exception: If API call fails.
        """
        try:
            wallets = self.get_wallets(asset_names=[asset])
            for wallet in wallets:
                if wallet["asset"] == asset:
                    return float(wallet["free"])
            self.logger.warning(f"No wallet found for asset: {asset}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting account balance for {asset}: {e}")
            return 0.0
