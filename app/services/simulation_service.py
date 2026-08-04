"""Shared post-simulation workflow for stocks and crypto assets."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SimulationSelection:
    trader: Any
    signal: Dict[str, Any]

    best_info: Dict[str, Any]


class SimulationService:
    def __init__(
        self,
        signal_repository,
        *,
        strategy_version: str,
        bootstrap_days: int,
        logger,
    ):
        self._signal_repository = signal_repository
        self._strategy_version = strategy_version
        self._bootstrap_days = bootstrap_days
        self._logger = logger

    def select_and_record(
        self,
        *,
        trader_driver,
        asset_type: str,
        exchange: str,
        asset: str,
        data_stream: list,
        lookback_hours: int = 48,
    ) -> SimulationSelection:
        best_info = trader_driver.best_trader_info
        trader = trader_driver.traders[best_info["trader_index"]]
        try:
            signal = trader.get_trade_signal(
                lag_intervals=0, lookback_hours=lookback_hours
            )
        except Exception:
            signal = trader.trade_signal

        try:
            inserted = self._signal_repository.record_trader_signals(
                asset_type=asset_type,
                exchange=exchange,
                asset=asset,
                trader=trader,
                data_stream=data_stream,
                strategy_version=self._strategy_version,
                bootstrap_days=self._bootstrap_days,
            )
            self._logger.info(
                f"Signal ledger checkpoint updated for {asset}/{trader.high_strategy}; "
                f"new pending signals={inserted}"
            )
        except Exception as exc:
            self._logger.error(f"Failed to update signal ledger for {asset}: {exc}")
        return SimulationSelection(trader=trader, signal=signal, best_info=best_info)
