import numpy as np

from app.backtesting.altcoin_defensive_validation import (
    BreakoutParameters,
    build_breakout_targets,
    run_validation,
)


SYMBOLS = ["ETH", "BTC", "SOL"]


def _features(eth, btc, sol, btc_sma=None):
    closes = np.asarray([eth, btc, sol], dtype=float).T
    sma200 = np.full_like(closes, np.nan)
    if btc_sma is not None:
        sma200[:, 1] = np.asarray(btc_sma, dtype=float)
    return {
        "closes": closes,
        "sma200": sma200,
        "opens": closes.copy(),
    }


def test_breakout_uses_only_prior_closes_and_trailing_stop():
    features = _features(
        eth=[10, 11, 12, 14, 13, 12],
        btc=[100] * 6,
        sol=[20] * 6,
    )
    params = BreakoutParameters(3, 0.10, False, False)

    active = build_breakout_targets(features, SYMBOLS, "ETH", params)[:, 0]

    assert active.tolist() == [0, 0, 0, 1, 1, 0]


def test_sol_breakout_requires_btc_defensive_regime():
    features = _features(
        eth=[10] * 7,
        btc=[100, 106, 107, 106, 94, 93, 92],
        sol=[10, 11, 12, 13, 14, 15, 16],
        btc_sma=[100] * 7,
    )
    params = BreakoutParameters(2, 0.10, True, False)

    active = build_breakout_targets(features, SYMBOLS, "SOL", params)[:, 2]

    assert active.tolist() == [0, 0, 1, 1, 0, 0, 0]


def test_frozen_strategies_pass_profit_gate_on_local_three_year_data():
    report = run_validation(
        ["ETH", "BTC", "SOL", "UNI", "LTC", "ETC", "DOGE", "AAVE"]
    )

    eth = report["eth_120d_breakout_defensive"]
    sol = report["sol_30d_breakout_defensive"]
    assert eth["passes_profit_gate"] is True
    assert sol["passes_profit_gate"] is True
    assert eth["full"]["return"] > 0
    assert eth["test"]["return"] > 0
    assert sol["full"]["return"] > 0
    assert sol["test"]["return"] > 0
