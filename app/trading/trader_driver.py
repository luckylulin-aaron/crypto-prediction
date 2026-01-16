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
    KDJ_OVERBOUGHT_THRESHOLDS,
    KDJ_OVERSOLD_THRESHOLDS,
    ROUND_PRECISION,
    RSI_OVERBOUGHT_THRESHOLDS,
    RSI_OVERSOLD_THRESHOLDS,
    RSI_PERIODS,
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
    ) -> int:
        """
        Compute the expected number of StratTrader instances given parameter grids.

        This is useful for sanity-checking narrowed grids (e.g., moving-window auto-tuning).
        """
        base = (
            len(bollinger_tols)
            * len(overall_stats)
            * len(tol_pcts)
            * len(buy_pcts)
            * len(sell_pcts)
        )

        # Adjust for strategies that expand the grid (RSI/KDJ)
        # Our unified grid counts RSI/KDJ once each; replace that "1" with their extra grid sizes.
        rsi_extra = len(rsi_periods) * len(rsi_oversold_thresholds) * len(rsi_overbought_thresholds)
        kdj_extra = len(kdj_oversold_thresholds) * len(kdj_overbought_thresholds)

        n_stats = len(overall_stats)
        if n_stats == 0:
            return 0

        # For each of bollinger_sigma/tol/buy/sell combination, total traders equals:
        # sum over strategies of (extra_grid_size_for_strategy)
        per_combo = 0
        for s in overall_stats:
            if s == "RSI":
                per_combo += max(1, rsi_extra)
            elif s == "KDJ":
                per_combo += max(1, kdj_extra)
            else:
                per_combo += 1

        combos_without_strategy = len(bollinger_tols) * len(tol_pcts) * len(buy_pcts) * len(sell_pcts)
        return combos_without_strategy * per_combo

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
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield dicts of parameters that define a unique StratTrader instance.

        This is intentionally strategy-agnostic: strategies are handled by providing an
        (optional) extra param grid per strategy.
        """
        for bollinger_sigma, stat, tol_pct, buy_pct, sell_pct in product(
            bollinger_tols, overall_stats, tol_pcts, buy_pcts, sell_pcts
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
            mode (str, optional): Mode. Defaults to 'normal'.

        Returns:
            None
        """
        self.name = name
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
        )
        if len(self.traders) != expected:
            logger.warning(
                f"[{self.name}] Trader count mismatch: expected={expected}, actual={len(self.traders)}. "
                f"(This can happen if a strategy ignores certain params, or grids were modified.)"
            )

        # Sanity check: ensure we created at least one trader.
        if len(self.traders) == 0:
            raise ValueError("No traders were created. Check your parameter grids and overall_stats.")
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
        intraday_interval_hours: int = 1,
    ):
        """
        Feed in historic data, where data_stream consists of tuples of (price, date, open, low, high).
        Date can be either a datetime object or a string that will be parsed.

        Args:
            data_stream (List[tuple]): List of tuples with price and date info.
            intraday_stream (Optional[List[tuple]]): Optional lower-granularity candles (e.g., 1h)
                to support zoom-in logic. Same tuple format as data_stream.
            intraday_interval_hours (int): Expected intraday interval in hours. Defaults to 1.

        Returns:
            None

        Raises:
            ValueError: If data_stream is empty or has insufficient data for simulation.
        """
        if self.mode == "verbose":
            print("running simulation...")

        # Validate data stream
        if not data_stream:
            raise ValueError("Data stream is empty - no historical data available for simulation")
        
        if len(data_stream) < 2:
            raise ValueError(f"Data stream has insufficient data points ({len(data_stream)}). Need at least 2 data points for simulation.")

        # Log data feed details
        logger.info(f"[{self.name}] Feeding {len(data_stream)} data points to {len(self.traders)} traders")
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
                    logger.info(f"[{self.name}] Price range: ${min_price:.2f} - ${max_price:.2f}")
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
        
        logger.debug(f"[{self.name}] Processing {num_traders} trading strategies across {len(data_stream)} data points")

        for index, t in enumerate(self.traders):
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
                    **(
                        {"volume": data_stream[0][5]}
                        if isinstance(data_stream[0], (list, tuple)) and len(data_stream[0]) > 5
                        else {}
                    ),
                },
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
                    **(
                        {"volume": data_stream[i][5]}
                        if isinstance(data_stream[i], (list, tuple)) and len(data_stream[i]) > 5
                        else {}
                    ),
                }
                t.add_new_day(p, d, misc_p)
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
            
            # Log trader performance summary (every 10 traders to avoid too much output)
            if (index + 1) % 10 == 0 or index == num_traders - 1:
                logger.debug(f"[{self.name}] Processed {index + 1}/{num_traders} traders "
                           f"(strategy: {t.high_strategy}, final_value: ${tmp_final_p:.2f}, "
                           f"time: {trader_process_time:.3f}s)")
            
            if tmp_final_p >= max_final_p:
                max_final_p = tmp_final_p
                self.best_trader = t
        
        logger.info(f"[{self.name}] Completed feed_data: Best trader strategy={self.best_trader.high_strategy}, "
                    f"final_value=${max_final_p:.2f}")

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
