import pytest

from app.backtesting.nflx_defensive_validation import run_validation


def test_nflx_frozen_strategy_passes_final_test_objective_and_runtime_parity():
    report = run_validation()

    assert report["configuration"]["data_rows"] >= 2_500
    assert report["configuration"]["candidate_count"] == 43
    assert report["configuration"]["execution_friction_rate"] == pytest.approx(0.02)
    assert report["selection"]["candidate"] == {
        "family": "monthly_sma",
        "parameters": {"window_days": 100, "band_pct": 0.05},
    }
    assert report["test"]["excess_return"] > 0.0
    assert report["test"]["strategy"]["annualized_return"] >= 10.0
    assert (
        report["test"]["strategy"]["max_drawdown"]
        < report["test"]["buy_and_hold"]["max_drawdown"]
    )
    assert report["runtime_matches_offline"] is True
    assert report["full_runtime_matches_offline"] is True
    assert report["test_runtime_matches_offline"] is True
    assert report["registered_test_runtime"]["return"] == pytest.approx(
        report["test"]["strategy"]["return"]
    )
    assert (
        report["registered_test_runtime"]["transactions"]
        == report["test"]["strategy"]["transactions"]
    )
    assert report["passes_return_objective"] is True
    assert report["passes_profit_gate"] is True
