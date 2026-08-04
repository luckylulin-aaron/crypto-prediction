from pathlib import Path

from app.core import main


class StockRunnerSpy:
    def __init__(self):
        self.calls = []

    def run(self, all_actions, best_summaries):
        self.calls.append((all_actions, best_summaries))


def test_main_stock_wrapper_delegates_to_runner(monkeypatch):
    runner = StockRunnerSpy()
    monkeypatch.setattr(main, "stock_simulation_runner", runner)
    actions = []
    summaries = []

    main._run_stock_simulation(actions, summaries)

    assert runner.calls == [(actions, summaries)]


def test_main_reexports_crypto_fetch_helpers():
    assert callable(main.fetch_historical_data_with_fallback)
    assert callable(main.fetch_intraday_data_with_fallback)
    assert main.fetch_historical_data_with_fallback.__module__.endswith(
        "crypto_simulation_runner"
    )


def test_main_is_kept_as_a_thin_orchestration_entrypoint():
    main_path = Path(main.__file__)
    assert len(main_path.read_text(encoding="utf-8").splitlines()) <= 500


def test_runner_dependencies_are_explicitly_injected():
    assert (
        main.stock_simulation_runner._market_client_factory
        is main.market_client_factory
    )
    assert (
        main.stock_simulation_runner._trader_driver_factory
        is main.trader_driver_factory
    )
    assert main.stock_simulation_runner._simulation_service is main.simulation_service
    assert (
        main.crypto_simulation_runner._trader_driver_factory
        is main.trader_driver_factory
    )
    assert main.crypto_simulation_runner._simulation_service is main.simulation_service
