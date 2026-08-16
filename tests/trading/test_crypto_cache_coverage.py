from datetime import datetime, timezone

from app.trading.cache_coverage import covers_latest_completed_utc_interval


def _row(timestamp: str) -> list:
    return [100.0, timestamp, 100.0, 100.0, 100.0, 1.0]


def test_daily_cache_requires_latest_completed_utc_candle():
    now = datetime(2026, 8, 13, 0, 45, tzinfo=timezone.utc)

    assert not covers_latest_completed_utc_interval(
        [_row("2026-08-09")], 24, now=now
    )
    assert covers_latest_completed_utc_interval(
        [_row("2026-08-12")], 24, now=now
    )


def test_intraday_cache_uses_completed_interval_boundary():
    now = datetime(2026, 8, 13, 7, 15, tzinfo=timezone.utc)

    assert not covers_latest_completed_utc_interval(
        [_row("2026-08-12 18:00:00")], 6, now=now
    )
    assert covers_latest_completed_utc_interval(
        [_row("2026-08-13 00:00:00")], 6, now=now
    )


def test_empty_or_invalid_cache_is_not_current():
    now = datetime(2026, 8, 13, 0, 45, tzinfo=timezone.utc)

    assert not covers_latest_completed_utc_interval([], 24, now=now)
    assert not covers_latest_completed_utc_interval(
        [_row("not-a-date")], 24, now=now
    )
