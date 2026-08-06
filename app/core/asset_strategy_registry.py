"""Asset-specific strategy registrations and frozen parameters."""

from typing import List

BTC_SMA200_DEFENSIVE_STRATEGY = "BTC-SMA200-DEFENSIVE"
ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY = "ETH-120D-BREAKOUT-DEFENSIVE"
SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY = "SOL-30D-BREAKOUT-DEFENSIVE"
TCEHY_REGIME_DEFENSIVE_STRATEGY = "TCEHY-REGIME-DEFENSIVE"
COIN_BTC_SMA200_DEFENSIVE_STRATEGY = "COIN-BTC-SMA200-DEFENSIVE"
MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY = "MSFT-20D-BREAKOUT-DEFENSIVE"
NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY = "NFLX-MONTHLY-SMA100-DEFENSIVE"

BTC_SMA200_DEFENSIVE_ASSETS = frozenset({"BTC"})
BTC_SMA200_DEFENSIVE_PARAMETERS = {
    "entry_band_pct": 0.05,
    "exit_band_pct": 0.05,
    "min_hold_days": 0,
}
ETH_120D_BREAKOUT_DEFENSIVE_PARAMETERS = {
    "lookback_days": 120,
    "trailing_stop_pct": 0.05,
    "require_btc_regime": False,
}
SOL_30D_BREAKOUT_DEFENSIVE_PARAMETERS = {
    "lookback_days": 30,
    "trailing_stop_pct": 0.10,
    "require_btc_regime": True,
}
TCEHY_REGIME_DEFENSIVE_PARAMETERS = {
    "trend_ma_days": 200,
    "trend_slope_days": 20,
    "range_ma_days": 20,
    "range_sigma": 2.0,
}
COIN_BTC_SMA200_DEFENSIVE_PARAMETERS = {
    "context_symbol": "BTC-USD",
    "entry_band_pct": 0.05,
    "exit_band_pct": 0.05,
    "context_lag_days": 1,
}
MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS = {
    "lookback_days": 20,
    "trailing_stop_pct": 0.10,
}
NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS = {
    "window_days": 100,
    "band_pct": 0.05,
}

CRYPTO_STRATEGIES = [
    BTC_SMA200_DEFENSIVE_STRATEGY,
    ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
    SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
]
CRYPTO_STRATEGY_ASSET_ALLOWLIST = {
    BTC_SMA200_DEFENSIVE_STRATEGY: BTC_SMA200_DEFENSIVE_ASSETS,
    ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY: frozenset({"ETH"}),
    SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY: frozenset({"SOL"}),
}

STOCK_STRATEGIES = [
    TCEHY_REGIME_DEFENSIVE_STRATEGY,
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
    "MA-SELVES",
    "DOUBLE-MA",
    "MA-BOLL-BANDS",
    "RSI",
    "KDJ",
    "ADAPTIVE-MA-SELVES",
]
STOCK_STRATEGY_ASSET_ALLOWLIST = {
    TCEHY_REGIME_DEFENSIVE_STRATEGY: frozenset({"TCEHY"}),
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY: frozenset({"COIN"}),
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY: frozenset({"MSFT"}),
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY: frozenset({"NFLX"}),
}
STOCK_EXCLUSIVE_STRATEGIES_BY_ASSET = {
    "TCEHY": (TCEHY_REGIME_DEFENSIVE_STRATEGY,),
    "COIN": (COIN_BTC_SMA200_DEFENSIVE_STRATEGY,),
    "MSFT": (MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,),
    "NFLX": (NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,),
}


def normalize_crypto_asset(asset: str) -> str:
    normalized = str(asset).replace("\\", "/").rsplit("/", 1)[-1].upper()
    return normalized.replace("-USD", "").replace("USDT", "")


def normalize_stock_symbol(asset: str) -> str:
    return str(asset).replace("\\", "/").rsplit("/", 1)[-1].upper()


def crypto_strategies_for_asset(asset: str) -> List[str]:
    normalized = normalize_crypto_asset(asset)
    return [
        strategy
        for strategy in CRYPTO_STRATEGIES
        if normalized in CRYPTO_STRATEGY_ASSET_ALLOWLIST.get(strategy, {normalized})
    ]


def stock_strategies_for_asset(asset: str) -> List[str]:
    normalized = normalize_stock_symbol(asset)
    exclusive = STOCK_EXCLUSIVE_STRATEGIES_BY_ASSET.get(normalized)
    if exclusive is not None:
        return list(exclusive)
    return [
        strategy
        for strategy in STOCK_STRATEGIES
        if normalized in STOCK_STRATEGY_ASSET_ALLOWLIST.get(strategy, {normalized})
    ]


def is_crypto_strategy_allowed_for_asset(strategy: str, asset: str) -> bool:
    normalized = normalize_crypto_asset(asset)
    allowed = CRYPTO_STRATEGY_ASSET_ALLOWLIST.get(strategy)
    return allowed is None or normalized in allowed


def is_stock_strategy_allowed_for_asset(strategy: str, asset: str) -> bool:
    normalized = normalize_stock_symbol(asset)
    allowed = STOCK_STRATEGY_ASSET_ALLOWLIST.get(strategy)
    return allowed is None or normalized in allowed
