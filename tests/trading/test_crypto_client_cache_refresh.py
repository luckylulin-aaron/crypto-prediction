import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


sys.path.insert(0, os.path.abspath("app"))

import trading.binance_client as binance_module
import trading.cbpro_client as coinbase_module
from trading.binance_client import BinanceClient
from trading.cbpro_client import CBProClient


def _cached_row(timestamp: datetime):
    return ("BTCUSDT__1d", timestamp.isoformat(), 100, 110, 90, 105, 123)


def _as_utc(timestamp: str) -> datetime:
    value = datetime.fromisoformat(timestamp)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def test_binance_refreshes_recent_cache_when_latest_daily_candle_is_missing():
    client = BinanceClient()
    completed_open = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    stale_row = _cached_row(completed_open - timedelta(days=3))
    api_kline = [
        int(completed_open.timestamp() * 1000),
        "100",
        "110",
        "90",
        "105",
        "123",
    ]

    with (
        patch.object(
            binance_module.db_manager,
            "get_historical_data",
            return_value=[stale_row],
        ),
        patch.object(
            binance_module.db_manager, "is_data_fresh", return_value=True
        ),
        patch.object(
            binance_module.db_manager, "store_historical_data", return_value=True
        ),
        patch.object(client.client, "klines", return_value=[api_kline]) as api_call,
    ):
        rows = client.get_historic_data(
            "BTCUSDT", interval_hours=24, lookback_days=1095
        )

    api_call.assert_called_once()
    assert _as_utc(rows[-1][1]) == completed_open


def test_coinbase_refreshes_recent_cache_when_latest_daily_candle_is_missing():
    client = CBProClient()
    completed_open = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    stale_row = _cached_row(completed_open - timedelta(days=3))
    api_candle = {
        "start": str(int(completed_open.timestamp())),
        "open": "100",
        "high": "110",
        "low": "90",
        "close": "105",
        "volume": "123",
    }

    with (
        patch.object(
            coinbase_module.db_manager,
            "get_historical_data",
            return_value=[stale_row],
        ),
        patch.object(
            coinbase_module.db_manager, "is_data_fresh", return_value=True
        ),
        patch.object(
            coinbase_module.db_manager, "store_historical_data", return_value=True
        ),
        patch.object(
            client.rest_client,
            "get_candles",
            return_value={"candles": [api_candle]},
        ) as api_call,
    ):
        rows = client.get_historic_data(
            "BTC-USD", interval_hours=24, lookback_days=1095
        )

    api_call.assert_called_once()
    assert _as_utc(rows[-1][1]) == completed_open
