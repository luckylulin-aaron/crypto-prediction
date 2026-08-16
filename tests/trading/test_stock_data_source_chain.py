from unittest.mock import patch

from app.trading.stock_data_sources import StockDataSourceUnavailable
from app.trading.us_stock_client import USStockClient


def _row(date: str, close: float = 100.0) -> list:
    return [close, date, close, close - 1, close + 1, 1000.0]


class _Source:
    def __init__(self, name, rows=None, error=None):
        self.name = name
        self.rows = rows or []
        self.error = error
        self.calls = 0

    def fetch(self, ticker, start, end):
        self.calls += 1
        if self.error:
            raise self.error
        return self.rows


def test_source_chain_skips_unconfigured_alpaca_and_uses_akshare():
    alpaca = _Source("alpaca", error=StockDataSourceUnavailable("credentials missing"))
    akshare = _Source("akshare", rows=[_row("2026-08-13")])
    yahoo = _Source("yfinance", rows=[_row("2026-08-13")])
    client = USStockClient(["MSFT"], sources=[alpaca, akshare, yahoo])

    rows = client.get_historic_data(
        "MSFT", start="2026-08-12", end="2026-08-14", use_cache=False
    )

    assert rows == [_row("2026-08-13")]
    assert (alpaca.calls, akshare.calls, yahoo.calls) == (1, 1, 0)


def test_cache_refresh_inserts_only_dates_missing_from_sqlite():
    cached = [_row("2026-08-10", 100.0)]
    source = _Source(
        "akshare",
        rows=[
            _row("2026-08-10", 999.0),
            _row("2026-08-11", 101.0),
            _row("2026-08-12", 102.0),
            _row("2026-08-13", 103.0),
        ],
    )
    client = USStockClient(["MSFT"], sources=[source])

    with (
        patch(
            "app.trading.us_stock_client.db_manager.get_historical_data",
            return_value=cached,
        ),
        patch(
            "app.trading.us_stock_client.db_manager.is_data_fresh",
            return_value=False,
        ),
        patch(
            "app.trading.us_stock_client.db_manager.store_historical_data",
            return_value=True,
        ) as store,
    ):
        rows = client.get_historic_data("MSFT", start="2026-08-10", end="2026-08-14")

    store.assert_called_once_with(
        "MSFT",
        [
            _row("2026-08-11", 101.0),
            _row("2026-08-12", 102.0),
            _row("2026-08-13", 103.0),
        ],
    )
    assert rows[0] == cached[0]
    assert [row[1] for row in rows] == [
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
    ]
