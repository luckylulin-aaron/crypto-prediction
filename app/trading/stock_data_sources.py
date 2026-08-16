"""Normalized daily-bar sources for US stocks.

Every source returns rows in the repository's canonical format:
``[close, YYYY-MM-DD, open, low, high, volume]``.  Credentials are loaded
without logging their values, and unavailable sources are skipped by the
caller.
"""

from __future__ import annotations

import configparser
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import yfinance as yf


class StockDataSourceUnavailable(RuntimeError):
    """Raised when a configured source cannot be used in this environment."""


def _valid_price(value) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid price: {value!r}")
    return result


def _valid_volume(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0 else 0.0


def _frame_to_rows(df: pd.DataFrame, date_column: Optional[str] = None) -> List[list]:
    rows = []
    for index, row in df.iterrows():
        try:

            def scalar(column: str):
                value = row[column]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                return value

            close = _valid_price(scalar("close" if "close" in row else "Close"))
            open_price = _valid_price(scalar("open" if "open" in row else "Open"))
            low = _valid_price(scalar("low" if "low" in row else "Low"))
            high = _valid_price(scalar("high" if "high" in row else "High"))
            volume_key = "volume" if "volume" in row else "Volume"
            volume = _valid_volume(scalar(volume_key))
            raw_date = scalar(date_column) if date_column else index
            date = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        rows.append([close, date, open_price, low, high, volume])
    return rows


class AlpacaStockDataSource:
    name = "alpaca"

    @staticmethod
    def _credentials() -> tuple[Optional[str], Optional[str]]:
        key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret = (
            os.getenv("ALPACA_SECRET_KEY")
            or os.getenv("ALPACA_API_SECRET")
            or os.getenv("APCA_API_SECRET_KEY")
        )
        if key and secret:
            return key, secret

        parser = configparser.ConfigParser()
        secret_path = Path(__file__).resolve().parents[1] / "core" / "secret.ini"
        if secret_path.exists():
            parser.read(secret_path, encoding="utf-8")
            section = parser["CONFIG"] if parser.has_section("CONFIG") else {}
            key = section.get("ALPACA_API_KEY") or section.get("APCA_API_KEY_ID")
            secret = (
                section.get("ALPACA_SECRET_KEY")
                or section.get("ALPACA_API_SECRET")
                or section.get("APCA_API_SECRET_KEY")
            )
        return key, secret

    def fetch(self, ticker: str, start: datetime, end: datetime) -> List[list]:
        key, secret = self._credentials()
        if not key or not secret:
            raise StockDataSourceUnavailable("Alpaca credentials are not configured")

        from alpaca.data.enums import Adjustment
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(key, secret)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
        )
        response = client.get_stock_bars(request)
        bars = response.data.get(ticker, [])
        return [
            [
                _valid_price(bar.close),
                pd.Timestamp(bar.timestamp).strftime("%Y-%m-%d"),
                _valid_price(bar.open),
                _valid_price(bar.low),
                _valid_price(bar.high),
                _valid_volume(bar.volume),
            ]
            for bar in bars
        ]


class AKShareStockDataSource:
    name = "akshare"

    def fetch(self, ticker: str, start: datetime, end: datetime) -> List[list]:
        import akshare as ak

        df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
        if df is None or df.empty:
            return []
        dates = pd.to_datetime(df["date"])
        filtered = df.loc[(dates >= start) & (dates < end)]
        return _frame_to_rows(filtered, date_column="date")


class YahooStockDataSource:
    name = "yfinance"

    def __init__(self, downloader: Optional[Callable] = None):
        self._downloader = downloader

    def fetch(self, ticker: str, start: datetime, end: datetime) -> List[list]:
        downloader = self._downloader or yf.download
        df = downloader(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return []
        return _frame_to_rows(df)


def default_stock_data_sources() -> list:
    """Return the persistent priority order used by daily simulations."""
    return [
        AlpacaStockDataSource(),
        AKShareStockDataSource(),
        YahooStockDataSource(),
    ]
