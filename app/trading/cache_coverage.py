"""Helpers for deciding whether market-data caches cover completed candles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence


def covers_latest_completed_utc_interval(
    cached_data: Sequence[Sequence],
    interval_hours: float,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return whether the cache includes the latest fully completed UTC candle."""
    if not cached_data or interval_hours <= 0:
        return False

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    interval_seconds = int(round(float(interval_hours) * 3600))
    current_open_seconds = (
        int(current_time.timestamp()) // interval_seconds * interval_seconds
    )
    expected_latest_open = datetime.fromtimestamp(
        current_open_seconds - interval_seconds, tz=timezone.utc
    )
    try:
        latest_value = str(cached_data[-1][1]).replace("Z", "+00:00")
        latest_open = datetime.fromisoformat(latest_value)
    except (IndexError, TypeError, ValueError):
        return False
    if latest_open.tzinfo is None:
        latest_open = latest_open.replace(tzinfo=timezone.utc)
    else:
        latest_open = latest_open.astimezone(timezone.utc)
    return latest_open >= expected_latest_open
