import logging
import math
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

try:
    from ..core.config import STOCK_HISTORY_LOOKBACK_DAYS, STOCKS
    from ..db.database import db_manager
    from .stock_data_sources import (
        StockDataSourceUnavailable,
        default_stock_data_sources,
    )
except ImportError:
    from core.config import STOCK_HISTORY_LOOKBACK_DAYS, STOCKS
    from db.database import db_manager
    from trading.stock_data_sources import (
        StockDataSourceUnavailable,
        default_stock_data_sources,
    )


logger = logging.getLogger(__name__)


class USStockClient:
    """Fetch US stock daily candles through a resilient source chain.

    Source priority is Alpaca, AKShare, then Yahoo Finance. Existing SQLite
    rows are immutable here: refreshes insert only dates not already cached.
    """

    _CACHE_COVERAGE_TOLERANCE_DAYS = 7
    _TAIL_FETCH_OVERLAP_DAYS = 7

    def __init__(self, tickers: Optional[List[str]] = None, sources=None):
        self.tickers = tickers if tickers is not None else STOCKS
        self.sources = sources if sources is not None else default_stock_data_sources()

    @staticmethod
    def _filter_to_range(
        data: List[list], start: datetime, end: datetime
    ) -> List[list]:
        filtered = []
        for row in data:
            if len(row) < 6:
                continue
            try:
                required_values = [float(row[index]) for index in (0, 2, 3, 4)]
                row_date = datetime.strptime(str(row[1])[:10], "%Y-%m-%d")
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) and value > 0 for value in required_values):
                continue
            if start <= row_date < end:
                filtered.append(row)
        return sorted(filtered, key=lambda row: str(row[1])[:10])

    def _cache_covers_range(
        self, data: List[list], start: datetime, end: datetime
    ) -> bool:
        """Allow weekends/holidays while rejecting a materially short cache."""
        if not data:
            return False
        first_date = datetime.strptime(str(data[0][1])[:10], "%Y-%m-%d")
        last_date = datetime.strptime(str(data[-1][1])[:10], "%Y-%m-%d")
        tolerance = timedelta(days=self._CACHE_COVERAGE_TOLERANCE_DAYS)
        latest_expected = min(end - timedelta(days=1), datetime.now())
        return (
            first_date <= start + tolerance and last_date >= latest_expected - tolerance
        )

    @staticmethod
    def _cache_has_latest_business_day(data: List[list], end: datetime) -> bool:
        if not data:
            return False
        last_date = datetime.strptime(str(data[-1][1])[:10], "%Y-%m-%d")
        expected = (
            pd.Timestamp(end.date()) - pd.tseries.offsets.BDay(1)
        ).to_pydatetime()
        return last_date >= expected

    def _fetch_from_sources(
        self, ticker: str, start: datetime, end: datetime
    ) -> List[list]:
        errors = []
        for source in self.sources:
            try:
                rows = self._filter_to_range(
                    source.fetch(ticker, start, end), start, end
                )
                if rows:
                    logger.info(
                        "%s supplied %d daily rows for %s (%s through %s)",
                        source.name,
                        len(rows),
                        ticker,
                        rows[0][1],
                        rows[-1][1],
                    )
                    return rows
                errors.append(f"{source.name}: empty response")
            except StockDataSourceUnavailable as exc:
                logger.info("Skipping %s for %s: %s", source.name, ticker, exc)
                errors.append(f"{source.name}: unavailable")
            except Exception as exc:
                logger.warning("%s refresh failed for %s: %s", source.name, ticker, exc)
                errors.append(f"{source.name}: {type(exc).__name__}: {exc}")
        if errors and all(
            "empty response" in error or "unavailable" in error for error in errors
        ):
            raise ValueError(f"No data found for ticker: {ticker}")
        raise RuntimeError(
            f"All stock data sources failed for {ticker}: " + "; ".join(errors)
        )

    def get_historic_data(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        use_cache: bool = True,
    ) -> List[list]:
        """Fetch daily OHLCV using inclusive-start/exclusive-end semantics."""
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
        start_dt = (
            datetime.strptime(start, "%Y-%m-%d")
            if start
            else end_dt - timedelta(days=STOCK_HISTORY_LOOKBACK_DAYS)
        )
        if start_dt >= end_dt:
            raise ValueError("start must be earlier than end")

        cached_data = []
        if use_cache:
            requested_days = (end_dt - start_dt).days
            cached_data = db_manager.get_historical_data(
                ticker, days=requested_days + self._CACHE_COVERAGE_TOLERANCE_DAYS
            )
            cached_data = self._filter_to_range(cached_data or [], start_dt, end_dt)
            historical_range = end_dt < datetime.now() - timedelta(days=7)
            cache_is_current = historical_range or db_manager.is_data_fresh(
                ticker, max_age_hours=72
            )
            if (
                cache_is_current
                and self._cache_covers_range(cached_data, start_dt, end_dt)
                and (
                    historical_range
                    or self._cache_has_latest_business_day(cached_data, end_dt)
                )
            ):
                return cached_data

        fetch_start = start_dt
        if use_cache and cached_data:
            last_cached = datetime.strptime(str(cached_data[-1][1])[:10], "%Y-%m-%d")
            fetch_start = max(
                start_dt, last_cached - timedelta(days=self._TAIL_FETCH_OVERLAP_DAYS)
            )

        try:
            fetched = self._fetch_from_sources(ticker, fetch_start, end_dt)
        except Exception:
            if use_cache and self._cache_covers_range(cached_data, start_dt, end_dt):
                logger.warning(
                    "All stock refresh sources failed for %s; using %d cached rows through %s",
                    ticker,
                    len(cached_data),
                    cached_data[-1][1],
                )
                return cached_data
            raise

        if not use_cache:
            return fetched

        existing_dates = {str(row[1])[:10] for row in cached_data}
        missing_rows = [
            row for row in fetched if str(row[1])[:10] not in existing_dates
        ]
        if missing_rows:
            if not db_manager.store_historical_data(ticker, missing_rows):
                raise RuntimeError(f"database store operation failed for {ticker}")
            logger.info(
                "Inserted %d missing SQLite rows for %s (%s through %s)",
                len(missing_rows),
                ticker,
                missing_rows[0][1],
                missing_rows[-1][1],
            )

        merged = {str(row[1])[:10]: row for row in cached_data}
        for row in missing_rows:
            merged[str(row[1])[:10]] = row
        return self._filter_to_range(list(merged.values()), start_dt, end_dt)

    def get_all_historic_data(self, start: str = None, end: str = None) -> dict:
        all_data = {}
        for ticker in self.tickers:
            try:
                all_data[ticker] = self.get_historic_data(ticker, start, end)
            except Exception:
                all_data[ticker] = []
        return all_data
