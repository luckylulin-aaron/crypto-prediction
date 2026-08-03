import numpy as np
import pytest

from app.backtesting.benchmark_validation import (
    EXECUTION_FRICTION_RATE,
    INITIAL_CAPITAL,
)
from app.backtesting.portfolio_strategy_validation import (
    RotationParameters,
    TrendParameters,
    _feature_matrices,
    build_equal_weight_targets,
    build_rotation_targets,
    build_single_asset_trend_targets,
    simulate_portfolio,
)


def _features(closes, opens=None, sma200=None, vol20=None):
    closes = np.asarray(closes, dtype=float)
    opens = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    shape = closes.shape
    return {
        "closes": closes,
        "opens": opens,
        "sma20": closes.copy(),
        "std20": np.ones(shape),
        "sma200": (
            np.full(shape, 100.0) if sma200 is None else np.asarray(sma200, dtype=float)
        ),
        "vol20": (
            np.full(shape, 0.02) if vol20 is None else np.asarray(vol20, dtype=float)
        ),
    }


def test_portfolio_executes_previous_close_target_at_next_open():
    features = _features([[100.0], [110.0]], opens=[[100.0], [100.0]])
    targets = np.asarray([[1.0], [1.0]])

    result = simulate_portfolio(features, targets, 1, 2, slippage_bps=10)

    quantity = INITIAL_CAPITAL * (1.0 - EXECUTION_FRICTION_RATE) / (100.0 * 1.001)
    expected_return = (quantity * 110.0 / INITIAL_CAPITAL - 1.0) * 100.0
    assert result["return"] == pytest.approx(expected_return)
    assert result["transactions"] == 1


def test_initial_execution_cost_is_included_in_drawdown():
    features = _features([[100.0], [100.0]], opens=[[100.0], [100.0]])
    targets = np.asarray([[1.0], [1.0]])

    result = simulate_portfolio(features, targets, 1, 2, slippage_bps=10)

    assert result["return"] < -2.0
    assert result["max_drawdown"] == pytest.approx(-result["return"])


def test_btc_gate_moves_all_assets_to_cash_after_exit_threshold():
    closes = np.asarray([[106.0, 106.0], [110.0, 110.0], [94.0, 110.0]])
    features = _features(closes, sma200=np.full_like(closes, 100.0))

    targets = build_equal_weight_targets(features, ["BTC", "ETH"], btc_gate=True)

    assert targets[1].tolist() == pytest.approx([0.5, 0.5])
    assert targets[2].tolist() == pytest.approx([0.0, 0.0])


def test_inverse_volatility_weights_are_frozen_between_rebalances():
    days = 240
    closes = np.column_stack(
        (np.linspace(106.0, 220.0, days), np.linspace(106.0, 300.0, days))
    )
    vol20 = np.column_stack(
        (np.linspace(0.01, 0.04, days), np.linspace(0.04, 0.01, days))
    )
    features = _features(closes, sma200=np.full_like(closes, 100.0), vol20=vol20)
    params = RotationParameters(60, 2, 30, True, False)

    targets = build_rotation_targets(features, ["BTC", "ETH"], params)

    assert targets[210].sum() == pytest.approx(1.0)
    assert targets[211].tolist() == pytest.approx(targets[210].tolist())


def test_future_prices_do_not_change_past_rotation_targets():
    days = 260
    symbols = ["BTC", "ETH", "SOL"]

    def data(future_multiplier):
        result = {}
        for asset_index, symbol in enumerate(symbols):
            rows = []
            for index in range(days):
                close = 100.0 + index * (asset_index + 1) * 0.2
                if index >= 230:
                    close *= future_multiplier ** (asset_index + 1)
                rows.append([close, str(index), close, close, close, 1_000.0])
            result[symbol] = rows
        return result

    base_features = _feature_matrices(symbols, data(1.0))
    changed_features = _feature_matrices(symbols, data(2.0))
    params = RotationParameters(90, 2, 30, False, True)

    base_targets = build_rotation_targets(base_features, symbols, params)
    changed_targets = build_rotation_targets(changed_features, symbols, params)

    assert changed_targets[:230] == pytest.approx(base_targets[:230])


def test_bollinger_entry_waits_for_pullback_without_using_future_prices():
    closes = np.asarray(
        [
            [106.0, 100.0],
            [110.0, 100.0],
            [104.0, 100.0],
            [94.0, 100.0],
        ]
    )
    features = _features(closes, sma200=np.full_like(closes, 100.0))
    features["sma20"][:, 0] = 103.0
    features["std20"][:, 0] = 2.0
    params = TrendParameters(0.05, 0.05, 0.5)

    targets = build_single_asset_trend_targets(features, ["BTC", "ETH"], "BTC", params)

    assert targets[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])

    closes[2, 0] = 106.0
    features["closes"] = closes
    features["sma20"][2, 0] = 106.0
    targets = build_single_asset_trend_targets(features, ["BTC", "ETH"], "BTC", params)
    assert targets[:, 0].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
