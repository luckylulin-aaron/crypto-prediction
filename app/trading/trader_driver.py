# built-in packages
import datetime
import math
import time
from typing import Any, List

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
            mode (str, optional): Mode. Defaults to 'normal'.

        Returns:
            None
        """
        self.name = name
        self.init_amount, self.init_coin = init_amount, cur_coin
        self.mode = mode
        self.traders = []
        for bollinger_sigma in bollinger_tols:
            for stat in overall_stats:
                for tol_pct in tol_pcts:
                    for buy_pct in buy_pcts:
                        for sell_pct in sell_pcts:
                            # For RSI strategy, iterate through RSI parameters
                            if stat == "RSI":
                                for rsi_period in rsi_periods:
                                    for oversold in rsi_oversold_thresholds:
                                        for overbought in rsi_overbought_thresholds:
                                            t = StratTrader(
                                                name=name,
                                                init_amount=init_amount,
                                                stat=stat,
                                                tol_pct=tol_pct,
                                                ma_lengths=ma_lengths,
                                                ema_lengths=ema_lengths,
                                                bollinger_mas=bollinger_mas,
                                                bollinger_sigma=bollinger_sigma,
                                                buy_pct=buy_pct,
                                                sell_pct=sell_pct,
                                                cur_coin=cur_coin,
                                                buy_stas=buy_stas,
                                                sell_stas=sell_stas,
                                                rsi_period=rsi_period,
                                                rsi_oversold=oversold,
                                                rsi_overbought=overbought,
                                                mode=mode,
                                            )
                                            self.traders.append(t)
                            elif stat == "KDJ":
                                for oversold in kdj_oversold_thresholds:
                                    for overbought in kdj_overbought_thresholds:
                                        t = StratTrader(
                                            name=name,
                                            init_amount=init_amount,
                                            stat=stat,
                                            tol_pct=tol_pct,
                                            ma_lengths=ma_lengths,
                                            ema_lengths=ema_lengths,
                                            bollinger_mas=bollinger_mas,
                                            bollinger_sigma=bollinger_sigma,
                                            buy_pct=buy_pct,
                                            sell_pct=sell_pct,
                                            cur_coin=cur_coin,
                                            buy_stas=buy_stas,
                                            sell_stas=sell_stas,
                                            kdj_oversold=oversold,
                                            kdj_overbought=overbought,
                                            mode=mode,
                                        )
                                        self.traders.append(t)
                            else:
                                # For non-RSI/KDJ strategies, use default parameters
                                t = StratTrader(
                                    name=name,
                                    init_amount=init_amount,
                                    stat=stat,
                                    tol_pct=tol_pct,
                                    ma_lengths=ma_lengths,
                                    ema_lengths=ema_lengths,
                                    bollinger_mas=bollinger_mas,
                                    bollinger_sigma=bollinger_sigma,
                                    buy_pct=buy_pct,
                                    sell_pct=sell_pct,
                                    cur_coin=cur_coin,
                                    buy_stas=buy_stas,
                                    sell_stas=sell_stas,
                                    mode=mode,
                                )
                                self.traders.append(t)

        # check
        expected_traders = 0

        # Calculate for each strategy type
        for stat in overall_stats:
            if stat == "RSI":
                # RSI: tol_pcts * buy_pcts * sell_pcts * bollinger_tols * rsi_periods * rsi_oversold * rsi_overbought
                rsi_combinations = (
                    len(tol_pcts)
                    * len(buy_pcts)
                    * len(sell_pcts)
                    * len(bollinger_tols)
                    * len(rsi_periods)
                    * len(rsi_oversold_thresholds)
                    * len(rsi_overbought_thresholds)
                )
                expected_traders += rsi_combinations
            elif stat == "KDJ":
                # KDJ: tol_pcts * buy_pcts * sell_pcts * bollinger_tols * kdj_oversold * kdj_overbought
                kdj_combinations = (
                    len(tol_pcts)
                    * len(buy_pcts)
                    * len(sell_pcts)
                    * len(bollinger_tols)
                    * len(kdj_oversold_thresholds)
                    * len(kdj_overbought_thresholds)
                )
                expected_traders += kdj_combinations
            else:
                # Other strategies: tol_pcts * buy_pcts * sell_pcts * bollinger_tols
                other_combinations = (
                    len(tol_pcts) * len(buy_pcts) * len(sell_pcts) * len(bollinger_tols)
                )
                expected_traders += other_combinations

        if len(self.traders) != expected_traders:
            raise ValueError(
                f"trader creation is wrong! Expected {expected_traders}, got {len(self.traders)}"
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
    def feed_data(self, data_stream: List[tuple]):
        """
        Feed in historic data, where data_stream consists of tuples of (price, date, open, low, high).
        Date can be either a datetime object or a string that will be parsed.

        Args:
            data_stream (List[tuple]): List of tuples with price and date info.

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
        
        logger.debug(f"[{self.name}] Processing {num_traders} trading strategies across {len(data_stream)} data points")

        for index, t in enumerate(self.traders):
            trader_start_time = time.perf_counter()
            
            # compute initial value
            date_obj = parse_date(data_stream[0][1])
            t.add_new_day(
                new_p=data_stream[0][0],
                d=date_obj,
                misc_p={
                    "open": data_stream[0][2],
                    "low": data_stream[0][3],
                    "high": data_stream[0][4],
                },
            )
            # run simulation
            for i in range(1, len(data_stream)):
                p = data_stream[i][0]
                d = parse_date(data_stream[i][1])  # [cur_price,date,open,low,high]
                misc_p = {
                    "open": data_stream[i][2],
                    "low": data_stream[i][3],
                    "high": data_stream[i][4],
                }
                t.add_new_day(p, d, misc_p)
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
