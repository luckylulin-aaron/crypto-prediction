import pytest

from app.backtesting.tencent_defensive_validation import run_validation


def test_registered_tcehy_strategy_beats_buy_and_hold_on_frozen_test_and_full_data():
    report = run_validation()

    assert report["configuration"]["data_rows"] >= 700
    assert report["configuration"]["execution_friction_rate"] == pytest.approx(0.02)
    assert report["passes_profit_gate"] is True
    assert report["test"]["excess_return"] > 0.0
    assert report["full"]["excess_return"] > 0.0
    assert report["registered_runtime"]["excess_return"] > 0.0
    assert report["registered_runtime"]["return"] == pytest.approx(
        report["full"]["strategy"]["return"], abs=0.001
    )
    assert (
        report["full"]["strategy"]["max_drawdown"]
        < report["full"]["buy_and_hold"]["max_drawdown"]
    )
