from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database


@pytest.fixture
def ledger_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", session_factory)
    manager = database.DatabaseManager()
    yield manager, session_factory
    database.Base.metadata.drop_all(engine)
    engine.dispose()


def _record(manager, latest, events):
    return manager.record_signal_events(
        asset_type="CRYPTO",
        exchange="BINANCE",
        asset="BTC",
        strategy="BTC-SMA200-DEFENSIVE",
        strategy_version="v1",
        latest_candle_date=latest,
        events=events,
        bootstrap_days=2,
    )


def test_signal_is_idempotent_and_delivery_is_retryable(ledger_db):
    manager, session_factory = ledger_db
    event = {
        "date": datetime(2026, 8, 9),
        "action": "BUY",
        "buy_percentage": 1.0,
        "sell_percentage": 1.0,
    }

    assert _record(manager, datetime(2026, 8, 10), [event]) == 1
    assert _record(manager, datetime(2026, 8, 10), [event]) == 0

    pending = manager.get_pending_signals()
    assert len(pending) == 1
    signal_id = pending[0]["id"]

    assert manager.mark_signal_delivery(
        [signal_id], success=False, error="smtp unavailable"
    ) == 1
    failed = manager.get_pending_signals()
    assert len(failed) == 1
    assert failed[0]["delivery_attempts"] == 1
    assert failed[0]["last_error"] == "smtp unavailable"

    assert manager.mark_signal_delivery([signal_id], success=True) == 1
    assert manager.get_pending_signals() == []

    with session_factory() as session:
        row = session.query(database.SignalLedger).one()
        assert row.delivery_status == "delivered"
        assert row.delivery_attempts == 2
        assert row.delivered_at is not None


def test_checkpoint_catches_signal_after_multi_day_downtime(ledger_db):
    manager, session_factory = ledger_db

    # First successful evaluation establishes the checkpoint without inventing
    # old historical notifications.
    assert _record(manager, datetime(2026, 8, 10), []) == 0

    delayed_event = {
        "date": datetime(2026, 8, 12),
        "action": "SELL",
        "buy_percentage": 1.0,
        "sell_percentage": 1.0,
    }
    # The next run is five days later. This event is older than the former 48h
    # reminder window, but is still found because it is newer than the checkpoint.
    assert _record(manager, datetime(2026, 8, 15), [delayed_event]) == 1
    assert _record(manager, datetime(2026, 8, 15), [delayed_event]) == 0

    pending = manager.get_pending_signals()
    assert [(row["action"], row["signal_date"]) for row in pending] == [
        ("SELL", datetime(2026, 8, 12))
    ]

    with session_factory() as session:
        checkpoint = session.query(database.SignalLedgerCheckpoint).one()
        assert checkpoint.last_evaluated_candle == datetime(2026, 8, 15)
