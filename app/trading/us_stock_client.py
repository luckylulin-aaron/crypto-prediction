from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

try:
    from ..core.config import STOCK_HISTORY_LOOKBACK_DAYS, STOCKS
    from ..db.database import db_manager
except ImportError:
    from core.config import STOCK_HISTORY_LOOKBACK_DAYS, STOCKS
    from db.database import db_manager


class USStockClient:
    """Fetch US stock daily candles from Yahoo Finance with database caching."""

    _CACHE_COVERAGE_TOLERANCE_DAYS = 7

    def __init__(self, tickers: Optional[List[str]] = None):
        self.tickers = tickers if tickers is not None else STOCKS

    @staticmethod
    def _row_value(row: pd.Series, column: str) -> float:
        """Return a scalar from either regular or yfinance MultiIndex rows."""
        value = row[column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value)

    @staticmethod
    def _filter_to_range(
        data: List[list], start: datetime, end: datetime
    ) -> List[list]:
        """Filter cached rows to yfinance's inclusive-start/exclusive-end range."""
        filtered = []
        for row in data:
            row_date = datetime.strptime(str(row[1])[:10], "%Y-%m-%d")
            if start <= row_date < end:
                filtered.append(row)
        return filtered

    def _cache_covers_range(
        self, data: List[list], start: datetime, end: datetime
    ) -> bool:
        """Allow for weekends/holidays while rejecting a fresh but short cache."""
        if not data:
            return False

        first_date = datetime.strptime(str(data[0][1])[:10], "%Y-%m-%d")
        last_date = datetime.strptime(str(data[-1][1])[:10], "%Y-%m-%d")
        tolerance = timedelta(days=self._CACHE_COVERAGE_TOLERANCE_DAYS)
        latest_expected = min(end - timedelta(days=1), datetime.now())
        return (
            first_date <= start + tolerance and last_date >= latest_expected - tolerance
        )

    def get_historic_data(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        use_cache: bool = True,
    ) -> List[list]:
        """
        Fetch daily OHLCV data for one stock.

        Yahoo Finance treats start as inclusive and end as exclusive. If no
        range is supplied, the most recent three calendar years are requested.
        """
        end_dt = (
            datetime.strptime(end, "%Y-%m-%d") if end is not None else datetime.now()
        )
        start_dt = (
            datetime.strptime(start, "%Y-%m-%d")
            if start is not None
            else end_dt - timedelta(days=STOCK_HISTORY_LOOKBACK_DAYS)
        )
        if start_dt >= end_dt:
            raise ValueError("start must be earlier than end")

        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

        if use_cache:
            requested_days = (end_dt - start_dt).days
            cached_data = db_manager.get_historical_data(
                ticker,
                days=requested_days + self._CACHE_COVERAGE_TOLERANCE_DAYS,
            )
            cached_data = self._filter_to_range(cached_data or [], start_dt, end_dt)
            historical_range = end_dt < datetime.now() - timedelta(days=7)
            cache_is_current = historical_range or db_manager.is_data_fresh(
                ticker, max_age_hours=72
            )
            if cache_is_current and self._cache_covers_range(
                cached_data, start_dt, end_dt
            ):
                return cached_data

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        parsed = []
        for idx, row in df.iterrows():
            parsed.append(
                [
                    self._row_value(row, "Close"),
                    idx.strftime("%Y-%m-%d"),
                    self._row_value(row, "Open"),
                    self._row_value(row, "Low"),
                    self._row_value(row, "High"),
                    self._row_value(row, "Volume"),
                ]
            )

        if use_cache and parsed:
            db_manager.store_historical_data(ticker, parsed)
        return parsed

    def get_all_historic_data(self, start: str = None, end: str = None) -> dict:
        """Fetch daily OHLCV data for all configured tickers."""
        all_data = {}
        for ticker in self.tickers:
            try:
                all_data[ticker] = self.get_historic_data(ticker, start, end)
            except Exception:
                all_data[ticker] = []
        return all_data
