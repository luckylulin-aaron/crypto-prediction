# built-in packages
import datetime
import math
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Optional

# third-party packages
import numpy as np

# customized packages
from app.core.config import (
    BTC_SMA200_DEFENSIVE_PARAMETERS,
    BTC_SMA200_DEFENSIVE_STRATEGY,
    COIN_BTC_SMA200_DEFENSIVE_PARAMETERS,
    COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
    ETH_120D_BREAKOUT_DEFENSIVE_PARAMETERS,
    ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
    KDJ_OVERBOUGHT_THRESHOLDS,
    KDJ_OVERSOLD_THRESHOLDS,
    MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS,
    MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
    NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS,
    NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
    ROUND_PRECISION,
    RSI_OVERBOUGHT_THRESHOLDS,
    RSI_OVERSOLD_THRESHOLDS,
    RSI_PERIODS,
    SMA200_VARIANTS,
    SOL_30D_BREAKOUT_DEFENSIVE_PARAMETERS,
    SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
    TCEHY_REGIME_DEFENSIVE_PARAMETERS,
    TCEHY_REGIME_DEFENSIVE_STRATEGY,
    is_crypto_strategy_allowed_for_asset,
    is_stock_strategy_allowed_for_asset,
)
from app.trading.strat_trader import StratTrader
from app.utils.util import timer

try:
    from app.core.logger import get_logger
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.logger import get_logger

logger = get_logger(__name__)


class TraderDriver:
    """A wrapper class on top of any of trader classes."""

    @staticmethod
    def expected_trader_count(
        overall_stats: List[str],
        tol_pcts: List[float],
        buy_pcts: List[float],
        sell_pcts: List[float],
        bollinger_tols: List[int],
        rsi_periods: List[int],
        rsi_oversold_thresholds: List[float],
        rsi_overbought_thresholds: List[float],
        kdj_oversold_thresholds: List[float],
        kdj_overbought_thresholds: List[float],
        sma200_variants: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Compute the expected number of StratTrader instances given parameter grids.

        This is useful for sanity-checking narrowed grids (e.g., moving-window auto-tuning).
        """

        # Adjust for strategies that expand the grid (RSI/KDJ)
        # Our unified grid counts RSI/KDJ once each; replace that "1" with their extra grid sizes.
        rsi_extra = (
            len(rsi_periods)
            * len(rsi_oversold_thresholds)
            * len(rsi_overbought_thresholds)
        )
        kdj_extra = len(kdj_oversold_thresholds) * len(kdj_overbought_thresholds)

        n_stats = len(overall_stats)
        if n_stats == 0:
            return 0

        # For each of bollinger_sigma/tol/buy/sell combination, total traders equals:
        # sum over strategies of (extra_grid_size_for_strategy)
        per_combo = 0
        fixed_strategy_count = 0
        configured_sma200_variants = sma200_variants or SMA200_VARIANTS
        for s in overall_stats:
            if s in {
                BTC_SMA200_DEFENSIVE_STRATEGY,
                ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
                SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
                TCEHY_REGIME_DEFENSIVE_STRATEGY,
                COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
                MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
                NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
            }:
                fixed_strategy_count += 1
            elif s == "SMA200":
                fixed_strategy_count += len(configured_sma200_variants)
            elif s == "RSI":
                per_combo += max(1, rsi_extra)
            elif s == "KDJ":
                per_combo += max(1, kdj_extra)
            else:
                per_combo += 1

        combos_without_strategy = (
            len(bollinger_tols) * len(tol_pcts) * len(buy_pcts) * len(sell_pcts)
        )
        return fixed_strategy_count + combos_without_strategy * per_combo

    @staticmethod
    def _strategy_extra_param_grid(
        strategy_name: str,
        rsi_periods: List[int],
        rsi_oversold_thresholds: List[float],
        rsi_overbought_thresholds: List[float],
        kdj_oversold_thresholds: List[float],
        kdj_overbought_thresholds: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Return a list of extra parameter dicts for a given strategy.

        This keeps trader creation logic uniform: every strategy uses the same core grid
        (tol_pct/buy_pct/sell_pct/bollinger_sigma), plus an optional strategy-specific grid.
        """
        if strategy_name == "RSI":
            return [
                {
                    "rsi_period": period,
                    "rsi_oversold": oversold,
                    "rsi_overbought": overbought,
                }
                for period, oversold, overbought in product(
                    rsi_periods, rsi_oversold_thresholds, rsi_overbought_thresholds
                )
            ]
        if strategy_name == "KDJ":
            return [
                {"kdj_oversold": oversold, "kdj_overbought": overbought}
                for oversold, overbought in product(
                    kdj_oversold_thresholds, kdj_overbought_thresholds
                )
            ]
        return [{}]

    @classmethod
    def _iter_trader_specs(
        cls,
        overall_stats: List[str],
        tol_pcts: List[float],
        buy_pcts: List[float],
        sell_pcts: List[float],
        bollinger_tols: List[int],
        rsi_periods: List[int],
        rsi_oversold_thresholds: List[float],
        rsi_overbought_thresholds: List[float],
        kdj_oversold_thresholds: List[float],
        kdj_overbought_thresholds: List[float],
        sma200_variants: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield dicts of parameters that define a unique StratTrader instance.

        This is intentionally strategy-agnostic: strategies are handled by providing an
        (optional) extra param grid per strategy.
        """
        configured_sma200_variants = sma200_variants or SMA200_VARIANTS
        for stat in overall_stats:
            if stat == BTC_SMA200_DEFENSIVE_STRATEGY:
                variant = BTC_SMA200_DEFENSIVE_PARAMETERS
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                    "sma200_entry_band_pct": float(variant["entry_band_pct"]),
                    "sma200_exit_band_pct": float(variant["exit_band_pct"]),
                    "sma200_min_hold_days": int(variant["min_hold_days"]),
                }
                continue
            if stat == COIN_BTC_SMA200_DEFENSIVE_STRATEGY:
                variant = COIN_BTC_SMA200_DEFENSIVE_PARAMETERS
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                    "sma200_entry_band_pct": float(variant["entry_band_pct"]),
                    "sma200_exit_band_pct": float(variant["exit_band_pct"]),
                    "sma200_min_hold_days": 0,
                }
                continue
            if stat == MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY:
                params = MSFT_20D_BREAKOUT_DEFENSIVE_PARAMETERS
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                    "breakout_lookback_days": int(params["lookback_days"]),
                    "breakout_trailing_stop_pct": float(params["trailing_stop_pct"]),
                    "breakout_require_btc_regime": False,
                }
                continue
            if stat == NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY:
                params = NFLX_MONTHLY_SMA100_DEFENSIVE_PARAMETERS
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                    "monthly_sma_window_days": int(params["window_days"]),
                    "monthly_sma_band_pct": float(params["band_pct"]),
                }
                continue
            if stat == TCEHY_REGIME_DEFENSIVE_STRATEGY:
                params = TCEHY_REGIME_DEFENSIVE_PARAMETERS
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": float(params["range_sigma"]),
                    "regime_trend_ma_days": int(params["trend_ma_days"]),
                    "regime_trend_slope_days": int(params["trend_slope_days"]),
                    "regime_range_ma_days": int(params["range_ma_days"]),
                    "regime_range_sigma": float(params["range_sigma"]),
                }
                continue
            if stat in {
                ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY,
                SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
            }:
                params = (
                    ETH_120D_BREAKOUT_DEFENSIVE_PARAMETERS
                    if stat == ETH_120D_BREAKOUT_DEFENSIVE_STRATEGY
                    else SOL_30D_BREAKOUT_DEFENSIVE_PARAMETERS
                )
                yield {
                    "stat": stat,
                    "tol_pct": 0.0,
                    "buy_pct": 1.0,
                    "sell_pct": 1.0,
                    "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                    "breakout_lookback_days": int(params["lookback_days"]),
                    "breakout_trailing_stop_pct": float(params["trailing_stop_pct"]),
                    "breakout_require_btc_regime": bool(params["require_btc_regime"]),
                }
                continue
            if stat == "SMA200":
                for variant in configured_sma200_variants:
                    yield {
                        "stat": stat,
                        "tol_pct": 0.0,
                        "buy_pct": 1.0,
                        "sell_pct": 1.0,
                        "bollinger_sigma": bollinger_tols[0] if bollinger_tols else 2,
                        "sma200_entry_band_pct": float(variant["entry_band_pct"]),
                        "sma200_exit_band_pct": float(variant["exit_band_pct"]),
                        "sma200_min_hold_days": int(variant["min_hold_days"]),
                    }
                continue

            for bollinger_sigma, tol_pct, buy_pct, sell_pct in product(
                bollinger_tols, tol_pcts, buy_pcts, sell_pcts
            ):
                extras = cls._strategy_extra_param_grid(
                    stat,
                    rsi_periods=rsi_periods,
                    rsi_oversold_thresholds=rsi_oversold_thresholds,
                    rsi_overbought_thresholds=rsi_overbought_thresholds,
                    kdj_oversold_thresholds=kdj_oversold_thresholds,
                    kdj_overbought_thresholds=kdj_overbought_thresholds,
                )
                for extra in extras:
                    yield {
                        "stat": stat,
                        "tol_pct": tol_pct,
                        "buy_pct": buy_pct,
                        "sell_pct": sell_pct,
                        "bollinger_sigma": bollinger_sigma,
                        **extra,
                    }

    def __init__(
        self,
        name: str,
        init_amount: int,
        cur_coin: float,
        overall_stats: List[str],
        tol_pcts: List[float],
        ma_lengths: List[int],
        ema_lengths: List[int],
        bollinger_mas: List[int],
        bollinger_tols: List[int],
        buy_pcts: List[float],
        sell_pcts: List[float],
        buy_stas: List[str] = ["by_percentage"],
        sell_stas: List[str] = ["by_percentage"],
        rsi_periods: List[int] = [14],
        rsi_oversold_thresholds: List[float] = [30],
        rsi_overbought_thresholds: List[float] = [70],
        kdj_oversold_thresholds: List[float] = [20],
        kdj_overbought_thresholds: List[float] = [80],
        zoom_in: bool = False,
        zoom_in_min_move_pct: float = 0.003,
        ma_boll_simplify: bool = True,
        execute_on_next_open: bool = False,
        slippage_bps: float = 0.0,
        enable_options: bool = True,
        sma200_variants: Optional[List[Dict[str, Any]]] = None,
        btc_data_stream: Optional[List[tuple]] = None,
        mode: str = "normal",
    ):
        """
        Initialize a TraderDriver instance.

        Args:
            name (str): Name of the trader.
            init_amount (int): Initial cash amount.
            cur_coin (float): Initial coin amount.
            overall_stats (List[str]): List of strategy names.
            tol_pcts (List[float]): List of tolerance percentages.
            ma_lengths (List[int]): List of moving average lengths.
            ema_lengths (List[int]): List of exponential moving average lengths.
            bollinger_mas (List[int]): List of Bollinger MA lengths.
            bollinger_tols (List[int]): List of Bollinger tolerances.
            buy_pcts (List[float]): List of buy percentages.
            sell_pcts (List[float]): List of sell percentages.
            buy_stas (List[str], optional): Buy strategies. Defaults to ['by_percentage'].
            sell_stas (List[str], optional): Sell strategies. Defaults to ['by_percentage'].
            zoom_in (bool, optional): Enable MA-BOLL-BANDS zoom-in refinement. Defaults to False.
            zoom_in_min_move_pct (float, optional): Min intraday move to treat as trending. Defaults to 0.003.
            ma_boll_simplify (bool, optional): Enable simplified MA-BOLL-BANDS logic. Defaults to False.
            mode (str, optional): Mode. Defaults to 'normal'.

        Returns:
            None
        """
        disallowed = []
        fixed_stock_strategies = {
            TCEHY_REGIME_DEFENSIVE_STRATEGY,
            COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
            MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY,
            NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY,
        }
        for strategy in overall_stats:
            if strategy in fixed_stock_strategies:
                allowed = is_stock_strategy_allowed_for_asset(strategy, name)
            else:
                allowed = is_crypto_strategy_allowed_for_asset(strategy, name)
            if not allowed:
                disallowed.append(strategy)
        if BTC_SMA200_DEFENSIVE_STRATEGY in disallowed:
            raise ValueError(
                f"{BTC_SMA200_DEFENSIVE_STRATEGY} is restricted to BTC; received {name}"
            )
        if TCEHY_REGIME_DEFENSIVE_STRATEGY in disallowed:
            raise ValueError(
                f"{TCEHY_REGIME_DEFENSIVE_STRATEGY} is restricted to TCEHY; "
                f"received {name}"
            )
        if COIN_BTC_SMA200_DEFENSIVE_STRATEGY in disallowed:
            raise ValueError(
                f"{COIN_BTC_SMA200_DEFENSIVE_STRATEGY} is restricted to COIN; "
                f"received {name}"
            )
        if MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY in disallowed:
            raise ValueError(
                f"{MSFT_20D_BREAKOUT_DEFENSIVE_STRATEGY} is restricted to MSFT; "
                f"received {name}"
            )
        if NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY in disallowed:
            raise ValueError(
                f"{NFLX_MONTHLY_SMA100_DEFENSIVE_STRATEGY} is restricted to NFLX; "
                f"received {name}"
            )
        if disallowed:
            raise ValueError(f"Strategies {disallowed} are not allowed for {name}")
        self.name = name
        self.btc_data_stream = btc_data_stream
        self.init_amount, self.init_coin = init_amount, cur_coin
        self.mode = mode
        self.traders = []
        for spec in self._iter_trader_specs(
            overall_stats=overall_stats,
            tol_pcts=tol_pcts,
            buy_pcts=buy_pcts,
            sell_pcts=sell_pcts,
            bollinger_tols=bollinger_tols,
            rsi_periods=rsi_periods,
            rsi_oversold_thresholds=rsi_oversold_thresholds,
            rsi_overbought_thresholds=rsi_overbought_thresholds,
            kdj_oversold_thresholds=kdj_oversold_thresholds,
            kdj_overbought_thresholds=kdj_overbought_thresholds,
            sma200_variants=sma200_variants,
        ):
            t = StratTrader(
                name=name,
                init_amount=init_amount,
                stat=spec["stat"],
                tol_pct=spec["tol_pct"],
                ma_lengths=ma_lengths,
                ema_lengths=ema_lengths,
                bollinger_mas=bollinger_mas,
                bollinger_sigma=spec["bollinger_sigma"],
                buy_pct=spec["buy_pct"],
                sell_pct=spec["sell_pct"],
                cur_coin=cur_coin,
                buy_stas=buy_stas,
                sell_stas=sell_stas,
                mode=mode,
                # optional extras (only used by certain strategies)
                rsi_period=spec.get("rsi_period"),
                rsi_oversold=spec.get("rsi_oversold"),
                rsi_overbought=spec.get("rsi_overbought"),
                kdj_oversold=spec.get("kdj_oversold"),
                kdj_overbought=spec.get("kdj_overbought"),
                zoom_in=zoom_in,
                zoom_in_min_move_pct=zoom_in_min_move_pct,
                ma_boll_simplify=ma_boll_simplify,
                execute_on_next_open=execute_on_next_open,
                slippage_bps=slippage_bps,
                enable_options=enable_options,
                sma200_entry_band_pct=spec.get("sma200_entry_band_pct", 0.0),
                sma200_exit_band_pct=spec.get("sma200_exit_band_pct", 0.0),
                sma200_min_hold_days=spec.get("sma200_min_hold_days", 0),
                breakout_lookback_days=spec.get("breakout_lookback_days", 0),
                breakout_trailing_stop_pct=spec.get("breakout_trailing_stop_pct", 0.0),
                breakout_require_btc_regime=spec.get(
                    "breakout_require_btc_regime", False
                ),
                regime_trend_ma_days=spec.get("regime_trend_ma_days", 200),
                regime_trend_slope_days=spec.get("regime_trend_slope_days", 20),
                regime_range_ma_days=spec.get("regime_range_ma_days", 20),
                regime_range_sigma=spec.get("regime_range_sigma", 2.0),
                monthly_sma_window_days=spec.get("monthly_sma_window_days", 100),
                monthly_sma_band_pct=spec.get("monthly_sma_band_pct", 0.05),
            )
            self.traders.append(t)

        expected = self.expected_trader_count(
            overall_stats=overall_stats,
            tol_pcts=tol_pcts,
            buy_pcts=buy_pcts,
            sell_pcts=sell_pcts,
            bollinger_tols=bollinger_tols,
            rsi_periods=rsi_periods,
            rsi_oversold_thresholds=rsi_oversold_thresholds,
            rsi_overbought_thresholds=rsi_overbought_thresholds,
            kdj_oversold_thresholds=kdj_oversold_thresholds,
            kdj_overbought_thresholds=kdj_overbought_thresholds,
            sma200_variants=sma200_variants,
        )
        if len(self.traders) != expected:
            logger.warning(
                f"[{self.name}] Trader count mismatch: expected={expected}, actual={len(self.traders)}. "
                f"(This can happen if a strategy ignores certain params, or grids were modified.)"
            )

        # Sanity check: ensure we created at least one trader.
        if len(self.traders) == 0:
            raise ValueError(
                "No traders were created. Check your parameter grids and overall_stats."
            )
        # unknown, without data
        self.best_trader = None

    def set_fear_greed_data(self, fear_greed_data: List[dict]):
        """
        Set fear & greed index data for all traders.

        Args:
            fear_greed_data (List[dict]): List of fear & greed index data dictionaries.

        Returns:
            None
        """
        for trader in self.traders:
            trader.fear_greed_data = fear_greed_data

    @timer
    def feed_data(
        self,
        data_stream: List[tuple],
        intraday_stream: Optional[List[tuple]] = None,
        btc_data_stream: Optional[List[tuple]] = None,
        intraday_interval_hours: int = 1,
        warmup_points: int = 0,
        performance_start_index: Optional[int] = None,
    ):
        """
        Feed in historic data, where data_stream consists of tuples of (price, date, open, low, high).
        Date can be either a datetime object or a string that will be parsed.

        Args:
            data_stream (List[tuple]): List of tuples with price and date info.
            intraday_stream (Optional[List[tuple]]): Optional lower-granularity candles (e.g., 1h)
                to support zoom-in logic. Same tuple format as data_stream.
            intraday_interval_hours (int): Expected intraday interval in hours. Defaults to 1.
            warmup_points (int): Leading rows used only for indicators; trading is disabled.

        Returns:
            None

        Raises:
            ValueError: If data_stream is empty or has insufficient data for simulation.
        """
        if self.mode == "verbose":
            print("running simulation...")

        # Validate data stream
        if not data_stream:
            raise ValueError(
                "Data stream is empty - no historical data available for simulation"
            )

        if len(data_stream) < 2:
            raise ValueError(
                f"Data stream has insufficient data points ({len(data_stream)}). Need at least 2 data points for simulation."
            )
        if warmup_points < 0 or warmup_points >= len(data_stream):
            raise ValueError("warmup_points must be >= 0 and smaller than data_stream")
        if performance_start_index is None:
            performance_start_index = 0
        if not 0 <= performance_start_index < len(data_stream):
            raise ValueError(
                "performance_start_index must reference a row in data_stream"
            )

        # Log data feed details
        logger.info(
            f"[{self.name}] Feeding {len(data_stream)} data points to {len(self.traders)} traders"
        )
        if len(data_stream) > 0:
            try:
                start_date = data_stream[0][1]
                end_date = data_stream[-1][1]
                logger.info(f"[{self.name}] Data range: {start_date} to {end_date}")
                # Calculate price range
                prices = [item[0] for item in data_stream]
                if prices:
                    min_price = min(prices)
                    max_price = max(prices)
                    logger.info(
                        f"[{self.name}] Price range: ${min_price:.2f} - ${max_price:.2f}"
                    )
            except Exception as e:
                logger.debug(f"[{self.name}] Could not extract data range info: {e}")

        # Helper function to convert date string to datetime if needed
        def parse_date(date_input):
            """Convert date string to datetime object if needed."""
            if isinstance(date_input, datetime.datetime):
                return date_input
            # Try parsing with datetime format first, then date-only format
            date_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]
            for fmt in date_formats:
                try:
                    return datetime.datetime.strptime(date_input, fmt)
                except (ValueError, TypeError):
                    continue
            # If all parsing fails, raise an error
            raise ValueError(f"Unable to parse date: {date_input}")

        requires_btc_regime = any(
            trader.high_strategy
            in {
                SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY,
                COIN_BTC_SMA200_DEFENSIVE_STRATEGY,
            }
            for trader in self.traders
        )
        uses_lagged_btc_context = any(
            trader.high_strategy == COIN_BTC_SMA200_DEFENSIVE_STRATEGY
            for trader in self.traders
        )
        btc_regime_by_date: Dict[datetime.date, bool] = {}
        btc_regime_ready_by_date: Dict[datetime.date, bool] = {}
        if requires_btc_regime:
            effective_btc_stream = (
                btc_data_stream if btc_data_stream is not None else self.btc_data_stream
            )
            required_strategy = (
                COIN_BTC_SMA200_DEFENSIVE_STRATEGY
                if uses_lagged_btc_context
                else SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY
            )
            if not effective_btc_stream:
                raise ValueError(
                    f"{required_strategy} requires a BTC daily data stream"
                )
            parsed_btc = sorted(
                (
                    (parse_date(item[1]), float(item[0]))
                    for item in effective_btc_stream
                ),
                key=lambda item: item[0],
            )
            btc_closes: List[float] = []
            # COIN bootstraps buy-and-hold until the first decisive BTC band signal.
            # SOL keeps its existing fail-closed warmup behavior.
            btc_active = uses_lagged_btc_context
            for btc_date, btc_close in parsed_btc:
                btc_closes.append(btc_close)
                btc_ready = len(btc_closes) >= 200
                if btc_ready:
                    btc_sma200 = float(np.mean(btc_closes[-200:]))
                    if btc_close > btc_sma200 * 1.05:
                        btc_active = True
                    elif btc_close < btc_sma200 * 0.95:
                        btc_active = False
                btc_regime_by_date[btc_date.date()] = btc_active
                btc_regime_ready_by_date[btc_date.date()] = btc_ready

            context_lag_days = (
                int(COIN_BTC_SMA200_DEFENSIVE_PARAMETERS["context_lag_days"])
                if uses_lagged_btc_context
                else 0
            )
            first_btc_date = parsed_btc[0][0].date()
            missing_dates = []
            for item in data_stream:
                stock_date = parse_date(item[1]).date()
                context_date = stock_date - datetime.timedelta(days=context_lag_days)
                if context_date not in btc_regime_by_date and not (
                    uses_lagged_btc_context and context_date < first_btc_date
                ):
                    missing_dates.append(context_date)
            if missing_dates:
                stream_name = "COIN" if uses_lagged_btc_context else "SOL"
                raise ValueError(
                    f"BTC daily data is not aligned with the {stream_name} stream; "
                    f"first missing date: {missing_dates[0]}"
                )

        def market_context_for(date_value: datetime.datetime) -> Dict[str, bool]:
            if not requires_btc_regime:
                return {}
            lag_days = (
                int(COIN_BTC_SMA200_DEFENSIVE_PARAMETERS["context_lag_days"])
                if uses_lagged_btc_context
                else 0
            )
            context_date = date_value.date() - datetime.timedelta(days=lag_days)
            return {
                "btc_defensive_active": btc_regime_by_date.get(context_date, False),
                "btc_defensive_ready": btc_regime_ready_by_date.get(
                    context_date, False
                ),
            }

        max_final_p = -math.inf
        num_traders = len(self.traders)

        # Pre-parse intraday candles if provided
        intraday_items = []
        if intraday_stream:
            for item in intraday_stream:
                try:
                    dt = parse_date(item[1])
                except Exception:
                    dt = None
                if dt is None:
                    continue
                intraday_items.append((dt, item))
            intraday_items.sort(key=lambda x: x[0])

        logger.debug(
            f"[{self.name}] Processing {num_traders} trading strategies across {len(data_stream)} data points"
        )

        for index, t in enumerate(self.traders):
            t.performance_start_index = int(performance_start_index)
            trader_start_time = time.perf_counter()

            intraday_idx = 0
            prev_dt = None

            # compute initial value
            date_obj = parse_date(data_stream[0][1])
            intraday_slice = []
            if intraday_items:
                while intraday_idx < len(intraday_items):
                    dt, item = intraday_items[intraday_idx]
                    if dt <= date_obj:
                        intraday_slice.append(item)
                        intraday_idx += 1
                    else:
                        break
            t.add_new_day(
                new_p=data_stream[0][0],
                d=date_obj,
                misc_p={
                    "open": data_stream[0][2],
                    "low": data_stream[0][3],
                    "high": data_stream[0][4],
                    "intraday_candles": intraday_slice,
                    "intraday_interval_hours": intraday_interval_hours,
                    "market_context": market_context_for(date_obj),
                    **(
                        {"volume": data_stream[0][5]}
                        if isinstance(data_stream[0], (list, tuple))
                        and len(data_stream[0]) > 5
                        else {}
                    ),
                },
                execute_strategy=warmup_points == 0,
            )
            prev_dt = date_obj
            # run simulation
            for i in range(1, len(data_stream)):
                p = data_stream[i][0]
                d = parse_date(data_stream[i][1])  # [cur_price,date,open,low,high]
                intraday_slice = []
                if intraday_items:
                    while intraday_idx < len(intraday_items):
                        dt, item = intraday_items[intraday_idx]
                        if dt <= d:
                            if prev_dt is None or dt > prev_dt:
                                intraday_slice.append(item)
                            intraday_idx += 1
                        else:
                            break
                misc_p = {
                    "open": data_stream[i][2],
                    "low": data_stream[i][3],
                    "high": data_stream[i][4],
                    "intraday_candles": intraday_slice,
                    "intraday_interval_hours": intraday_interval_hours,
                    "market_context": market_context_for(d),
                    **(
                        {"volume": data_stream[i][5]}
                        if isinstance(data_stream[i], (list, tuple))
                        and len(data_stream[i]) > 5
                        else {}
                    ),
                }
                t.add_new_day(
                    p,
                    d,
                    misc_p,
                    execute_strategy=i >= warmup_points,
                )
                prev_dt = d
            # decide best trader while we loop, by comparing all traders final portfolio value
            # sometimes a trader makes no trade at all
            if len(t.all_history) > 0:
                tmp_final_p = t.all_history[-1]["portfolio"]
            # o/w, compute it
            else:
                tmp_final_p = (t.crypto_prices[-1][0] * t.cur_coin) + t.cash
            """
            try:
                tmp_final_p = t.all_history[-1]['portfolio']
            except IndexError as e:
                print('Found error!', t.high_strategy)
            """
            trader_process_time = time.perf_counter() - trader_start_time

            if not math.isfinite(float(tmp_final_p)):
                logger.warning(
                    f"[{self.name}] Ignoring non-finite final portfolio value "
                    f"for strategy {t.high_strategy}: {tmp_final_p}"
                )
                continue

            # Log trader performance summary (every 10 traders to avoid too much output)
            if (index + 1) % 10 == 0 or index == num_traders - 1:
                logger.debug(
                    f"[{self.name}] Processed {index + 1}/{num_traders} traders "
                    f"(strategy: {t.high_strategy}, final_value: ${tmp_final_p:.2f}, "
                    f"time: {trader_process_time:.3f}s)"
                )

            if tmp_final_p >= max_final_p:
                max_final_p = tmp_final_p
                self.best_trader = t

        if self.best_trader is None:
            raise ValueError(
                f"[{self.name}] No strategy produced a finite final portfolio value"
            )

        logger.info(
            f"[{self.name}] Completed feed_data: Best trader strategy={self.best_trader.high_strategy}, "
            f"final_value=${max_final_p:.2f}"
        )

    @property
    def best_trader_info(self):
        """
        Find the best trading strategy for a given crypto-currency.

        Args:
            None

        Returns:
            dict: Information about the best trader and its performance.
        """
        best_trader = self.best_trader

        # compute init value once again, in case no single trade is made
        init_v = (
            best_trader.init_coin * best_trader.crypto_prices[0][0]
            + best_trader.init_cash
        )

        extra = {
            "init_value": np.round(init_v, ROUND_PRECISION),
            "max_final_value": np.round(best_trader.portfolio_value, ROUND_PRECISION),
            "rate_of_return": str(best_trader.rate_of_return) + "%",
            "baseline_rate_of_return": str(best_trader.baseline_rate_of_return) + "%",
            "coin_rate_of_return": str(best_trader.coin_rate_of_return) + "%",
        }

        return {
            **best_trader.trading_strategy,
            **extra,
            "trader_index": self.traders.index(self.best_trader),
        }

    def get_all_strategy_performance(self):
        """
        Collect performance data for all strategies tested.

        Args:
            None

        Returns:
            List[Dict]: List of dictionaries containing strategy performance data.
        """
        strategy_performance = []

        for index, trader in enumerate(self.traders):
            try:
                if trader is None:
                    continue

                # compute init value once again, in case no single trade is made
                if (
                    hasattr(trader, "init_coin")
                    and hasattr(trader, "crypto_prices")
                    and hasattr(trader, "init_cash")
                ):
                    init_v = (
                        trader.init_coin * trader.crypto_prices[0][0] + trader.init_cash
                    )
                else:
                    init_v = 0

                # Extract rate of return as float for sorting
                rate_of_return_str = getattr(trader, "rate_of_return", "0%")
                if isinstance(rate_of_return_str, (int, float, np.number)):
                    rate_of_return_float = float(rate_of_return_str)
                    rate_of_return_str = f"{rate_of_return_float:.3f}%"
                else:
                    rate_of_return_float = float(
                        str(rate_of_return_str).replace("%", "")
                    )

                performance_data = {
                    "strategy": getattr(trader, "high_strategy", f"Strategy_{index}"),
                    "rate_of_return": rate_of_return_float,
                    "rate_of_return_str": rate_of_return_str,
                    "init_value": np.round(init_v, ROUND_PRECISION),
                    "max_final_value": np.round(
                        getattr(trader, "portfolio_value", 0), ROUND_PRECISION
                    ),
                    "baseline_rate_of_return": getattr(
                        trader, "baseline_rate_of_return", 0
                    ),
                    "coin_rate_of_return": getattr(trader, "coin_rate_of_return", 0),
                    "max_drawdown": getattr(trader, "max_drawdown", 0) * 100,
                    "num_transactions": getattr(trader, "num_transaction", 0),
                    "num_buys": getattr(trader, "num_buy_action", 0),
                    "num_sells": getattr(trader, "num_sell_action", 0),
                    "trader_index": index,
                }

                # Add trading strategy parameters if available
                if hasattr(trader, "trading_strategy"):
                    performance_data.update(trader.trading_strategy)

                strategy_performance.append(performance_data)

            except Exception as e:
                continue

        # Sort by rate of return in descending order
        strategy_performance.sort(key=lambda x: x["rate_of_return"], reverse=True)

        return strategy_performance
