"""Persistence boundary for durable strategy signal notifications."""

from typing import Any, Dict, Iterable, List, Optional


class SignalLedgerRepository:
    def __init__(self, database_manager):
        self._database_manager = database_manager

    def record_trader_signals(
        self,
        *,
        asset_type: str,
        exchange: str,
        asset: str,
        trader,
        data_stream: list,
        strategy_version: str,
        bootstrap_days: int,
    ) -> int:
        if not data_stream:
            return 0
        signal_events = getattr(trader, "signal_history", None) or []
        event_source = signal_events or (getattr(trader, "trade_history", None) or [])
        events = [
            {
                "date": event.get("date"),
                "action": event.get("action"),
                "buy_percentage": trader.buy_pct,
                "sell_percentage": trader.sell_pct,
            }
            for event in event_source
            if str(event.get("action", "")).upper() in ("BUY", "SELL")
        ]
        return self._database_manager.record_signal_events(
            asset_type=asset_type,
            exchange=exchange,
            asset=asset,
            strategy=trader.high_strategy,
            strategy_version=strategy_version,
            latest_candle_date=data_stream[-1][1],
            events=events,
            bootstrap_days=bootstrap_days,
        )

    def pending(self) -> List[Dict[str, Any]]:
        return self._database_manager.get_pending_signals()

    def mark_delivery(
        self,
        signal_ids: Iterable[int],
        *,
        success: bool,
        error: Optional[str] = None,
    ) -> int:
        return self._database_manager.mark_signal_delivery(
            list(signal_ids), success=success, error=error
        )
