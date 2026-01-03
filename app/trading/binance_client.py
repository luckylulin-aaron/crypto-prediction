import time
from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
from binance.spot import Spot
from core.config import CURS, DATA_INTERVAL_HOURS, STABLECOIN, TIMESPAN
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
        Get historical klines for a symbol with configurable interval, with database caching.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            use_cache (bool): Whether to use cached data if available and fresh.
        Returns:
            list: Each element is [closing_price, datetime_str, open_price, low, high, volume].
        Raises:
            ConnectionError: If the API call fails.
        """
        # NOTE ABOUT GRANULARITY + CACHING:
        # The current DB schema uses (symbol, date) as the uniqueness key. If we store multiple
        # granularities (e.g., 12h vs 30m) under the same symbol, rows will collide/overwrite.
        # To avoid corrupting cached data, we only use DB cache for the original hour-based intervals.
        interval_minutes = int(round(float(DATA_INTERVAL_HOURS) * 60))

        def _binance_interval_str(minutes: int) -> str:
            # Binance spot kline interval strings (common subset)
            if minutes < 1:
                raise ValueError(f"interval_minutes must be >= 1, got {minutes}")
            if minutes % 60 == 0:
                hours = minutes // 60
                if hours == 1:
                    return "1h"
                if hours == 2:
                    return "2h"
                if hours == 4:
                    return "4h"
                if hours == 6:
                    return "6h"
                if hours == 8:
                    return "8h"
                if hours == 12:
                    return "12h"
                if hours == 24:
                    return "1d"
            # minute-based
            if minutes in (1, 3, 5, 15, 30):
                return f"{minutes}m"
            raise ValueError(
                f"Unsupported Binance kline interval: {minutes} minutes. "
                f"Try 30m directly, or use 5m/15m and resample."
            )

        # 10m is not a standard Binance interval; we can fetch 5m and resample.
        wants_resample_10m = interval_minutes == 10
        fetch_interval_minutes = 5 if wants_resample_10m else interval_minutes
        interval_str = _binance_interval_str(fetch_interval_minutes)

        cache_allowed = fetch_interval_minutes in (60, 360, 720, 1440) and not wants_resample_10m
        # IMPORTANT: DB uniqueness is (symbol, date). If DATA_INTERVAL_HOURS changes (e.g. 12h -> 6h),
        # cached rows can silently mismatch the requested granularity. Use an interval-specific cache key.
        cache_key = f"{symbol}__{interval_str}"

        if use_cache and cache_allowed:
            cached_data = db_manager.get_historical_data(cache_key, TIMESPAN)
            if cached_data and db_manager.is_data_fresh(cache_key, max_age_hours=72):
                self.logger.info(
                    f"Using cached data for {symbol} ({len(cached_data)} records, interval={interval_str})"
                )
                return cached_data
            elif cached_data:
                self.logger.info(
                    f"Cached data for {symbol} is stale, fetching fresh data (interval={interval_str})"
                )
            else:
                self.logger.info(
                    f"No cached data found for {symbol}, fetching from API (interval={interval_str})"
                )
        elif use_cache and not cache_allowed:
            self.logger.info(
                f"DB cache disabled for {symbol} at interval={interval_minutes}m "
                f"(to avoid DB collisions). Fetching from API."
            )

        try:
            end = int(time.time() * 1000)
            start = int(
                (datetime.utcnow() - timedelta(days=TIMESPAN)).timestamp() * 1000
            )
            self.logger.info(f"Fetching klines for {symbol} from {datetime.utcfromtimestamp(start/1000).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.utcfromtimestamp(end/1000).strftime('%Y-%m-%d %H:%M:%S')}")

            # Binance returns limited klines per request; paginate to cover long timespans.
            def _fetch_klines_paginated(
                sym: str,
                itv: str,
                start_ms: int,
                end_ms: int,
                limit: int = 1000,
            ) -> List[list]:
                out: List[list] = []
                cur_start = start_ms
                while True:
                    batch = self.client.klines(
                        sym, itv, startTime=cur_start, endTime=end_ms, limit=limit
                    )
                    if not batch:
                        break
                    out.extend(batch)
                    last_open_time = batch[-1][0]
                    # Move start forward; +1ms to avoid infinite loop on identical last_open_time.
                    next_start = last_open_time + 1
                    if next_start <= cur_start:
                        break
                    cur_start = next_start
                    # If we got less than the limit, we’re likely done.
                    if len(batch) < limit:
                        break
                    # Be gentle on rate limits
                    time.sleep(0.05)
                return out

            klines = _fetch_klines_paginated(symbol, interval_str, start, end, limit=1000)
            self.logger.info(
                f"Received {len(klines)} klines from API for {symbol} (interval: {interval_str})"
            )
            
            if not klines:
                self.logger.warning(f"No klines returned from API for {symbol}")
                return []

            # Parse into [close, datetime_str, open, low, high, volume]
            raw = []
            for k in klines:
                open_time_s = k[0] // 1000
                fmt_dt_str = datetime.utcfromtimestamp(open_time_s).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                open_price = float(k[1])
                high = float(k[2])
                low = float(k[3])
                closing_price = float(k[4])
                volume = float(k[5])
                raw.append([closing_price, fmt_dt_str, open_price, low, high, volume])
            raw = sorted(raw, key=lambda x: x[1])

            # Resample 5m -> 10m if requested
            parsed = raw
            if wants_resample_10m:
                grouped = {}
                for close_p, dt_str, open_p, low_p, high_p, vol in raw:
                    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    # floor to 10-minute boundary
                    floored_minute = (dt_obj.minute // 10) * 10
                    bucket = dt_obj.replace(minute=floored_minute, second=0, microsecond=0)
                    key = bucket.strftime("%Y-%m-%d %H:%M:%S")
                    if key not in grouped:
                        grouped[key] = {
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": vol,
                        }
                    else:
                        grouped[key]["high"] = max(grouped[key]["high"], high_p)
                        grouped[key]["low"] = min(grouped[key]["low"], low_p)
                        grouped[key]["close"] = close_p
                        grouped[key]["volume"] += vol
                keys = sorted(grouped.keys())
                parsed = [
                    [
                        grouped[k]["close"],
                        k,
                        grouped[k]["open"],
                        grouped[k]["low"],
                        grouped[k]["high"],
                        grouped[k]["volume"],
                    ]
                    for k in keys
                ]

            # Filter out data from the current (possibly incomplete) interval
            now = datetime.utcnow()
            cutoff_time = now - timedelta(minutes=interval_minutes)
            cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")
            parsed = [x for x in parsed if x[1] < cutoff_str]
            
            self.logger.info(f"Processed {len(parsed)} data points for {symbol} after filtering")
            
            if use_cache and cache_allowed and parsed:
                db_manager.store_historical_data(cache_key, parsed)
            return parsed
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
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
            # Wallet balances logging removed to reduce redundancy
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
