from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf
from core.config import STOCKS


class USStockClient:
    """
    Client for fetching US stock data using yfinance.
    """
    def __init__(self, tickers: Optional[List[str]] = None):
        """
        Args:
            tickers: List of stock tickers to fetch. If None, uses config.STOCKS.
        """
        self.tickers = tickers if tickers is not None else STOCKS

    def get_historic_data(self, ticker: str, start: str = None, end: str = None) -> List[list]:
        """
        Fetch daily OHLCV data for a given stock ticker.
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            start: Start date in 'YYYY-MM-DD' format (default: 1 year ago).
            end: End date in 'YYYY-MM-DD' format (default: today).
        Returns:
            List of [close, date, open, low, high, volume] for each day.
        Raises:
            ValueError: If ticker is invalid or data is unavailable.
        """
        if start is None:
            start = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start, end=end, interval='1d', progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        parsed = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            parsed.append([
                row['Close'].item(),
                date_str,
                row['Open'].item(),
                row['Low'].item(),
                row['High'].item(),
                row['Volume'].item()
            ])
        return parsed

    def get_all_historic_data(self, start: str = None, end: str = None) -> dict:
        """
        Fetch daily OHLCV data for all configured tickers.
        Args:
            start: Start date in 'YYYY-MM-DD' format (default: 1 year ago).
            end: End date in 'YYYY-MM-DD' format (default: today).
        Returns:
            Dict mapping ticker to list of [close, date, open, low, high, volume].
        """
        all_data = {}
        for ticker in self.tickers:
            try:
                all_data[ticker] = self.get_historic_data(ticker, start, end)
            except Exception as e:
                all_data[ticker] = []
        return all_data 