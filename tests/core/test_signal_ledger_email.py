from datetime import datetime

from app.core import main


def test_pending_ledger_signal_drives_email_with_original_signal_date(
    tmp_path, monkeypatch
):
    log_file = tmp_path / "log.txt"
    log_file.write_text("Finish job at time 2026-08-15 08:50:00\n", encoding="utf-8")
    sent_messages = []

    def fake_send_email(**kwargs):
        sent_messages.append(kwargs)
        return True

    monkeypatch.setitem(
        main.send_daily_recommendations_email.__globals__, "send_email", fake_send_email
    )
    pending = [
        {
            "id": 7,
            "asset_type": "CRYPTO",
            "exchange": "BINANCE",
            "asset": "BTC",
            "strategy": "BTC-SMA200-DEFENSIVE",
            "strategy_version": "v1",
            "signal_date": datetime(2026, 8, 12),
            "action": "BUY",
            "buy_percentage": 1.0,
            "sell_percentage": 1.0,
        }
    ]

    delivered = main.send_daily_recommendations_email(
        log_file,
        ["admin@example.com"],
        "bot@example.com",
        "app-password",
        pending_signals=pending,
    )

    assert delivered is True
    assert len(sent_messages) == 1
    assert "2026-08-12 00:00:00" in sent_messages[0]["body"]
    assert "BTC" in sent_messages[0]["body"]
    assert "BUY" in sent_messages[0]["body"]


def test_admin_send_failure_keeps_delivery_unsuccessful(tmp_path, monkeypatch):
    log_file = tmp_path / "log.txt"
    log_file.write_text("", encoding="utf-8")
    monkeypatch.setitem(
        main.send_daily_recommendations_email.__globals__,
        "send_email",
        lambda **kwargs: False,
    )

    delivered = main.send_daily_recommendations_email(
        log_file,
        ["admin@example.com"],
        "bot@example.com",
        "app-password",
        pending_signals=[{"asset": "TCEHY", "exchange": "STOCK", "action": "SELL"}],
    )

    assert delivered is False
