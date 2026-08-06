from app.core.config import (
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
    CRYPTO_SIGNAL_LOOKBACK_DAYS,
    CRYPTO_SIMULATION_INITIAL_CASH,
    CRYPTO_SIMULATION_INITIAL_COIN,
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
    NFLX_RUNTIME_EVALUATION_ROWS,
    NFLX_RUNTIME_HISTORY_LOOKBACK_DAYS,
    STOCK_SIMULATION_ASSETS,
    TCEHY_REGIME_DEFENSIVE_STRATEGY,
    stock_strategies_for_asset,
)


def test_crypto_daily_simulation_matches_frozen_validation_capital_and_history():
    assert CRYPTO_SIGNAL_LOOKBACK_DAYS == 3 * 365
    assert CRYPTO_SIMULATION_INITIAL_CASH == 10_000.0
    assert CRYPTO_SIMULATION_INITIAL_COIN == 0.0


def test_daily_stock_simulation_is_limited_to_fixed_asset_strategies():
    assert STOCK_SIMULATION_ASSETS == ["MSFT", "TCEHY", "COIN", "NFLX"]
    assert NFLX_RUNTIME_HISTORY_LOOKBACK_DAYS == 10 * 365
    assert NFLX_RUNTIME_EVALUATION_ROWS == 750
    assert stock_strategies_for_asset("MSFT") == [MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY]
    assert stock_strategies_for_asset("TCEHY") == [TCEHY_REGIME_DEFENSIVE_STRATEGY]
    assert stock_strategies_for_asset("COIN") == [COIN_BTC_SMA200_DEFENSIVE_STRATEGY]
    assert stock_strategies_for_asset("NFLX") == [
        NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY
    ]
