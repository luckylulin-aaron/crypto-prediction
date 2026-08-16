from datetime import datetime
from types import SimpleNamespace

from app.services.market_client_factory import MarketClientFactory
from app.services.trader_driver_factory import TraderDriverFactory
from app.services.notification_service import NotificationService
from app.services.simulation_service import SimulationService


class LoggerSpy:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)


class LedgerSpy:
    def __init__(self):
        self.record_calls = []
        self.mark_calls = []
        self.pending_rows = []

    def record_trader_signals(self, **kwargs):
        self.record_calls.append(kwargs)
        return 1

    def pending(self):
        return list(self.pending_rows)

    def mark_delivery(self, signal_ids, **kwargs):
        self.mark_calls.append((list(signal_ids), kwargs))
        return len(signal_ids)


def test_simulation_service_is_shared_selection_and_ledger_boundary():
    ledger = LedgerSpy()
    logger = LoggerSpy()
    trader = SimpleNamespace(
        high_strategy="BTC-SMA200-DEFENSIVE",
        trade_signal={"action": "NO ACTION"},
        get_trade_signal=lambda **kwargs: {"action": "BUY"},
    )
    driver = SimpleNamespace(
        best_trader_info={"trader_index": 0},
        traders=[trader],
    )
    service = SimulationService(
        ledger, strategy_version="v1", bootstrap_days=2, logger=logger
    )

    result = service.select_and_record(
        trader_driver=driver,
        asset_type="CRYPTO",
        exchange="Binance",
        asset="BTC",
        data_stream=[[100.0, datetime(2026, 8, 4)]],
    )

    assert result.trader is trader
    assert result.signal == {"action": "BUY"}
    assert result.best_info == {"trader_index": 0}
    assert ledger.record_calls[0]["asset"] == "BTC"
    assert ledger.record_calls[0]["strategy_version"] == "v1"


def test_notification_service_marks_only_successful_admin_delivery():
    ledger = LedgerSpy()
    ledger.pending_rows = [{"id": 11}, {"id": 12}]
    logger = LoggerSpy()
    renderer_calls = []

    def renderer(*args, **kwargs):
        renderer_calls.append((args, kwargs))
        return True

    service = NotificationService(ledger, renderer, logger)
    assert service.send_daily(
        log_file="runtime/logs/trading-bot.log",
        recipient_list=["admin@example.com"],
        from_email="bot@example.com",
        app_password="secret",
    )
    assert renderer_calls[0][1]["pending_signals"] == ledger.pending_rows
    assert ledger.mark_calls == [([11, 12], {"success": True, "error": None})]


def test_notification_service_keeps_failed_delivery_pending():
    ledger = LedgerSpy()
    ledger.pending_rows = [{"id": 21}]
    logger = LoggerSpy()
    service = NotificationService(ledger, lambda *args, **kwargs: False, logger)

    assert not service.send_daily(
        log_file="runtime/logs/trading-bot.log",
        recipient_list=["admin@example.com"],
        from_email="bot@example.com",
        app_password="secret",
    )
    assert ledger.mark_calls == [
        (
            [21],
            {"success": False, "error": "admin email delivery failed"},
        )
    ]


def test_trader_driver_factory_merges_common_and_asset_parameters():
    calls = []

    class Driver:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    factory = TraderDriverFactory(Driver, {"tol_pcts": [0.1], "mode": "normal"})
    factory.create(
        name="ETH",
        initial_cash=10_000,
        initial_coin=0,
        strategies=["ETH-120D-BREAKOUT-DEFENSIVE"],
        buy_pcts=[1.0],
        sell_pcts=[1.0],
        execute_on_next_open=True,
    )

    assert calls == [
        {
            "name": "ETH",
            "init_amount": 10_000,
            "cur_coin": 0,
            "overall_stats": ["ETH-120D-BREAKOUT-DEFENSIVE"],
            "buy_pcts": [1.0],
            "sell_pcts": [1.0],
            "btc_data_stream": None,
            "tol_pcts": [0.1],
            "mode": "normal",
            "execute_on_next_open": True,
        }
    ]


def test_market_client_factory_constructs_explicit_dependencies():
    class Coinbase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Binance:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Stocks:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    factory = MarketClientFactory(Coinbase, Binance, Stocks)
    clients = factory.crypto(
        coinbase_key="cb-key",
        coinbase_secret="cb-secret",
        binance_key="bn-key",
        binance_secret="bn-secret",
    )
    stocks = factory.stocks(("MSFT", "TCEHY", "COIN"))

    assert clients.coinbase.kwargs == {"key": "cb-key", "secret": "cb-secret"}
    assert clients.binance.kwargs == {
        "api_key": "bn-key",
        "api_secret": "bn-secret",
    }
    assert stocks.kwargs == {"tickers": ["MSFT", "TCEHY", "COIN"]}
