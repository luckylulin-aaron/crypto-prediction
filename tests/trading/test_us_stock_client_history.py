from datetime import datetime
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

import app.trading.us_stock_client as stock_module
from app.core.config import STOCK_HISTORY_LOOKBACK_DAYS
from app.trading.us_stock_client import USStockClient


def _row(date: str, price: float = 100.0) -> list:
    return [price, date, price, price, price, 1000.0]


class USStockClientHistoryTests(TestCase):
    def setUp(self):
        self.client = USStockClient(tickers=["AAPL"])

    @staticmethod
    def _download_frame(date: str = "2019-01-02") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [100.0],
                "High": [110.0],
                "Low": [90.0],
                "Close": [105.0],
                "Volume": [12345],
            },
            index=[pd.Timestamp(date)],
        )

    def test_fresh_but_short_cache_does_not_replace_requested_range(self):
        short_cache = [_row("2021-01-04"), _row("2021-12-31")]
        with (
            patch.object(
                stock_module.db_manager,
                "get_historical_data",
                return_value=short_cache,
            ),
            patch.object(stock_module.db_manager, "is_data_fresh", return_value=True),
            patch.object(
                stock_module.db_manager,
                "store_historical_data",
                return_value=True,
            ),
            patch.object(
                stock_module.yf,
                "download",
                return_value=self._download_frame(),
            ) as download,
        ):
            rows = self.client.get_historic_data(
                "AAPL", start="2019-01-01", end="2022-01-03"
            )

        download.assert_called_once()
        self.assertEqual(rows[0][1], "2019-01-02")

    def test_complete_historical_cache_is_used(self):
        complete_cache = [_row("2019-01-02"), _row("2021-12-31")]
        with (
            patch.object(
                stock_module.db_manager,
                "get_historical_data",
                return_value=complete_cache,
            ),
            patch.object(stock_module.yf, "download") as download,
        ):
            rows = self.client.get_historic_data(
                "AAPL", start="2019-01-01", end="2022-01-03"
            )

        download.assert_not_called()
        self.assertEqual(rows, complete_cache)

    def test_default_request_spans_three_calendar_years(self):
        with patch.object(
            stock_module.yf,
            "download",
            return_value=self._download_frame(),
        ) as download:
            self.client.get_historic_data("AAPL", use_cache=False)

        call = download.call_args.kwargs
        start = datetime.strptime(call["start"], "%Y-%m-%d")
        end = datetime.strptime(call["end"], "%Y-%m-%d")
        self.assertEqual((end - start).days, STOCK_HISTORY_LOOKBACK_DAYS)

    def test_download_drops_incomplete_non_finite_candles(self):
        frame = pd.DataFrame(
            {
                "Open": [100.0, float("nan")],
                "High": [110.0, float("nan")],
                "Low": [90.0, float("nan")],
                "Close": [105.0, float("nan")],
                "Volume": [12345.0, float("nan")],
            },
            index=[pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02")],
        )

        with patch.object(stock_module.yf, "download", return_value=frame):
            rows = self.client.get_historic_data(
                "AAPL",
                start="2026-08-01",
                end="2026-08-03",
                use_cache=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "2026-08-01")
        self.assertEqual(rows[0][0], 105.0)
