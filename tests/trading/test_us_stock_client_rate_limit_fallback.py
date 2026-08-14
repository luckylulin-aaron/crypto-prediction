from unittest.mock import patch

import pandas as pd
import pytest

import app.trading.us_stock_client as stock_module
from app.trading.stock_data_sources import YahooStockDataSource
from app.trading.us_stock_client import USStockClient


def _row(date: str) -> list:
    return [100.0, date, 100.0, 99.0, 101.0, 1000.0]


def test_empty_yahoo_response_falls_back_to_complete_recent_cache():
    cached = [_row("2023-08-16"), _row("2026-08-10")]
    client = USStockClient(["MSFT"], sources=[YahooStockDataSource()])

    with (
        patch.object(
            stock_module.db_manager, "get_historical_data", return_value=cached
        ),
        patch.object(stock_module.db_manager, "is_data_fresh", return_value=False),
        patch.object(stock_module.yf, "download", return_value=pd.DataFrame()),
    ):
        rows = client.get_historic_data("MSFT", start="2023-08-15", end="2026-08-14")

    assert rows == cached


def test_yahoo_exception_falls_back_to_complete_recent_cache():
    cached = [_row("2023-08-16"), _row("2026-08-10")]
    client = USStockClient(["MSFT"], sources=[YahooStockDataSource()])

    with (
        patch.object(
            stock_module.db_manager, "get_historical_data", return_value=cached
        ),
        patch.object(stock_module.db_manager, "is_data_fresh", return_value=False),
        patch.object(stock_module.yf, "download", side_effect=RuntimeError("429")),
    ):
        rows = client.get_historic_data("MSFT", start="2023-08-15", end="2026-08-14")

    assert rows == cached


def test_incomplete_cache_is_not_used_as_rate_limit_fallback():
    incomplete = [_row("2025-08-01"), _row("2026-08-10")]
    client = USStockClient(["MSFT"], sources=[YahooStockDataSource()])

    with (
        patch.object(
            stock_module.db_manager,
            "get_historical_data",
            return_value=incomplete,
        ),
        patch.object(stock_module.db_manager, "is_data_fresh", return_value=False),
        patch.object(stock_module.yf, "download", return_value=pd.DataFrame()),
    ):
        with pytest.raises(ValueError, match="No data found"):
            client.get_historic_data("MSFT", start="2023-08-15", end="2026-08-14")
