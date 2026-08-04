import pytest

from app.backtesting.held_stock_defensive_validation import run_validation


@pytest.mark.parametrize("symbol", ["COIN", "MSFT"])
def test_registered_held_stock_strategy_beats_same_cost_buy_and_hold(symbol):
    report = run_validation()[symbol]

    assert report["configuration"]["data_rows"] >= 700
    assert report["configuration"]["execution_friction_rate"] == pytest.approx(0.02)
    assert report["passes_profit_gate"] is True
    assert report["runtime_matches_offline"] is True
    assert report["test"]["excess_return"] > 0.0
    assert report["full"]["excess_return"] > 0.0
    assert report["registered_runtime"]["excess_return"] > 0.0
    assert (
        report["full"]["strategy"]["max_drawdown"]
        < report["full"]["buy_and_hold"]["max_drawdown"]
    )
