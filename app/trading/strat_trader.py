# built-in packages
import collections
import copy
import datetime
import time
from typing import Any, List

# third-party packages
import numpy as np

# customized packages
from app.core.config import (
    BUY_SIGNAL,
    DEA_NUM_OF_DAYS,
    DEPOSIT_CST,
    NO_ACTION_SIGNAL,
    ROUND_PRECISION,
    SELL_SIGNAL,
    STRATEGIES,
    WITHDRAW_CST,
)
from app.trading.strategies import STRATEGY_REGISTRY
from app.utils.util import ema_helper, max_drawdown_helper


class StratTrader:

    """Moving Average Trader, a strategic-trading implementation that focus and relies on moving averages."""

    def __init__(
        self,
        name: str,
        init_amount: float,
        stat: str,
        tol_pct: float,
        ma_lengths: List[int],
        ema_lengths: List[int],
        bollinger_mas: List[int],
        bollinger_sigma: int,
        buy_pct: float,
        sell_pct: float,
        cur_coin: float = 0.0,
        buy_stas: List[str] = ["by_percentage"],
        sell_stas: List[str] = ["by_percentage"],
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        kdj_oversold: float = 20,
        kdj_overbought: float = 80,
        mode: str = "normal",
    ):
        """
        Initialize a StratTrader instance.

        Args:
            name (str): Name of the cryptocurrency
            init_amount (float): Initial amount of cash
            stat (str): Strategy type (MA-SELVES, RSI, KDJ, etc.)
            tol_pct (float): Tolerance percentage for moving averages
            ma_lengths (List[int]): List of moving average lengths
            ema_lengths (List[int]): List of exponential moving average lengths
            bollinger_mas (List[int]): List of Bollinger Bands moving average lengths
            bollinger_sigma (int): Bollinger Bands standard deviation multiplier
            buy_pct (float): Percentage of cash to buy
            sell_pct (float): Percentage of crypto to sell
            cur_coin (float): Current amount of cryptocurrency
            buy_stas (List[str]): List of buy execution strategies
            sell_stas (List[str]): List of sell execution strategies
            rsi_period (int): RSI calculation period
            rsi_oversold (float): RSI oversold threshold
            rsi_overbought (float): RSI overbought threshold
            kdj_oversold (float): KDJ oversold threshold
            kdj_overbought (float): KDJ overbought threshold
            mode (str): Trading mode (normal, verbose)

        Returns:
            None
        """
        # check
        if stat not in STRATEGIES:
            raise ValueError("Unknown high-level trading strategy!")
        if not all([x in ma_lengths for x in bollinger_mas]):
            raise ValueError(
                "cannot initialize Bollinger Band strategy if some of moving averages are not available!"
            )
        # 1. Basic
        # crypto-currency name, tolerance percentage
        self.crypto_name, self.tol_pct = name, tol_pct
        # initial and current bitcoin at hand, and cash available
        self.init_coin, self.init_cash = cur_coin, init_amount
        self.cur_coin, self.cash = cur_coin, init_amount

        # 2. Transactions
        self.trade_history = []
        self.crypto_prices = []
        self.volume_history = []  # Track volume for volume-based strategies
        self.price_history = []  # Track price history for volatility calculations
        # moving average (type: dictionary)
        self.moving_averages = dict(
            zip([str(x) for x in ma_lengths], [[] for _ in range(len(ma_lengths))])
        )
        # exponential moving average queues and related items (type: dictionary)
        self.exp_moving_averages = dict(
            zip([str(x) for x in ema_lengths], [[] for _ in range(len(ema_lengths))])
        )
        # MACD_DIFF = EMA12- EMA26; MACD_DEA = 9_DAY_EMA_OF_MACD_DIFF
        self.macd_diff, self.macd_dea = [], []
        # KDJ index related
        self.nine_day_rsv = []
        self.kdj_dct = collections.defaultdict(list)
        # number of MAs considered in Bollinger Band strategy
        self.bollinger_mas = [str(x) for x in bollinger_mas]
        self.bollinger_sigma = bollinger_sigma

        # 3. Trading Strategy
        # buy percentage (how much you want to invest) of your cash
        # sell percentage (how much you want to sell off) from your coin
        self.buy_pct, self.sell_pct = buy_pct, sell_pct
        self.strategies = {"buy": buy_stas, "sell": sell_stas}
        # high-level strategy
        self.high_strategy = stat
        # debug mode
        self.mode = mode  # normal, verbose
        # brokerage percentage
        self.broker_pct = 0.02
        # Track last buy price for stop loss and take profit strategies
        self.last_buy_price = None
        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        # KDJ parameters
        self.kdj_oversold = kdj_oversold
        self.kdj_overbought = kdj_overbought
        # strategy signal lists
        self.strat_dct = collections.defaultdict(list)

    def add_new_day(self, new_p: float, d: datetime.datetime, misc_p: dict):
        """
        Add a new day's crypto-currency price, find out a computed transaction for today.

        Args:
            new_p (float): New price.
            d (datetime.datetime): Date.
            misc_p (dict): Miscellaneous price info (open, low, high).

        Returns:
            None
        """
        open, low, high = misc_p["open"], misc_p["low"], misc_p["high"]

        self.crypto_prices.append((new_p, d, open, low, high))
        self.price_history.append(new_p)  # Track price for volatility calculations

        # Track volume if available in misc_p
        if "volume" in misc_p:
            self.volume_history.append(misc_p["volume"])
        else:
            # Use a default volume if not available
            self.volume_history.append(1000.0)

        # add new moving averages and exponential moving averages, get_historic_data respectively
        for queue_name in self.moving_averages:
            self.add_new_moving_averages(queue_name=queue_name, new_p=new_p)
        for queue_name in self.exp_moving_averages:
            self.add_new_exponential_moving_averages(queue_name=queue_name, new_p=new_p)

        # compute KDJ related arrays
        self.compute_kdj_related(d=d, low=low, high=high, open=open, close=new_p)

        # Execute trading strategy using the strategy registry
        if self.high_strategy in STRATEGY_REGISTRY:
            strategy_func = STRATEGY_REGISTRY[self.high_strategy]

            if self.high_strategy == "MA-SELVES":
                for queue_name in self.moving_averages:
                    # if we don't have a moving average yet, skip
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                    )

            elif self.high_strategy == "EXP-MA-SELVES":
                for queue_name in self.exp_moving_averages:
                    # if we don't have an exponential moving average yet, skip
                    if self.exp_moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                    )

            elif self.high_strategy == "DOUBLE-MA":
                mas = [int(x) for x in list(self.moving_averages.keys())]
                mas = sorted(mas, reverse=False)

                for i in range(0, len(mas)):
                    for j in range(i + 1, len(mas)):
                        # if too close, skip
                        if j - i <= 2:
                            continue
                        s_qn, l_qn = str(mas[i]), str(mas[j])
                        # if either of the two MAs does not have enough lengths, respectively, skip
                        if (
                            self.moving_averages[s_qn][-1] is None
                            or self.moving_averages[l_qn][-1] is None
                        ):
                            continue
                        strategy_func(
                            trader=self,
                            shorter_queue_name=s_qn,
                            longer_queue_name=l_qn,
                            new_p=new_p,
                            today=d,
                        )

            elif self.high_strategy == "MACD":
                self.compute_macd_related()
                strategy_func(trader=self, new_p=new_p, today=d)

            elif self.high_strategy == "BOLL-BANDS":
                for queue_name in self.bollinger_mas:
                    # if we don't have a moving average yet, skip
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        bollinger_sigma=self.bollinger_sigma,
                    )

            elif self.high_strategy == "RSI":
                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d,
                    period=self.rsi_period,
                    oversold=self.rsi_oversold,
                    overbought=self.rsi_overbought,
                )

            elif self.high_strategy == "KDJ":
                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d,
                    oversold=self.kdj_oversold,
                    overbought=self.kdj_overbought,
                )

            elif self.high_strategy == "MULTI-MA-SELVES":
                # Use short, medium, and long MAs for confirmation
                ma_lengths_int = [int(x) for x in self.moving_averages.keys()]
                short_queue = str(min(ma_lengths_int))
                medium_queue = str(sorted(ma_lengths_int)[len(ma_lengths_int) // 2])
                long_queue = str(max(ma_lengths_int))

                strategy_func(
                    trader=self,
                    short_queue=short_queue,
                    medium_queue=medium_queue,
                    long_queue=long_queue,
                    new_p=new_p,
                    today=d,
                    tol_pct=self.tol_pct,
                    buy_pct=self.buy_pct,
                    sell_pct=self.sell_pct,
                    confirmation_required=2,
                )

            elif self.high_strategy == "TREND-MA-SELVES":
                for queue_name in self.moving_averages.keys():
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                        trend_period=50,
                        trend_threshold=0.02,
                    )

            elif self.high_strategy == "VOLUME-MA-SELVES":
                for queue_name in self.moving_averages.keys():
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                        volume_period=20,
                        volume_threshold=1.5,
                    )

            elif self.high_strategy == "ADAPTIVE-MA-SELVES":
                for queue_name in self.moving_averages.keys():
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        base_tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                        volatility_period=20,
                        volatility_multiplier=1.0,
                    )

            elif self.high_strategy == "MOMENTUM-MA-SELVES":
                for queue_name in self.moving_averages.keys():
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        tol_pct=self.tol_pct,
                        buy_pct=self.buy_pct,
                        sell_pct=self.sell_pct,
                        momentum_period=14,
                        momentum_threshold=0.01,
                    )

            elif self.high_strategy == "FEAR-GREED-SENTIMENT":
                # Import config for Fear & Greed thresholds
                from app.core.config import (
                    FEAR_GREED_BUY_THRESHOLDS,
                    FEAR_GREED_SELL_THRESHOLDS,
                )

                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d,
                    buy_thresholds=FEAR_GREED_BUY_THRESHOLDS,
                    sell_thresholds=FEAR_GREED_SELL_THRESHOLDS,
                )

    def add_new_moving_averages(self, queue_name: str, new_p: float):
        """
        Compute and append a new moving average price.

        Args:
            queue_name (str): Name of the MA queue.
            new_p (float): New price.

        Returns:
            None
        """
        max_l = int(queue_name)
        if len(self.crypto_prices) <= max_l:
            self.moving_averages[queue_name].append(None)
            return
        # compute new moving average, add it
        # note: grab -max because we already add today's price
        prices = [x[0] for x in self.crypto_prices[-max_l:]]
        new_ma = sum(prices) / max_l
        self.moving_averages[queue_name].append(new_ma)

    def add_new_exponential_moving_averages(self, queue_name: str, new_p: float):
        """
        Compute and append a new exponential moving average price.

        Args:
            queue_name (str): Name of the EMA queue.
            new_p (float): New price.

        Returns:
            None
        """
        max_l = int(queue_name)
        if len(self.crypto_prices) <= max_l:
            self.exp_moving_averages[queue_name].append(None)
            return

        # when the lengths are just equal, EMA = MA
        if len(self.exp_moving_averages[queue_name]) == max_l:
            prices = [x[0] for x in self.crypto_prices[-max_l:]]
            new_ema = sum(prices) / max_l
            self.exp_moving_averages[queue_name].append(new_ema)
        else:
            # compute new exponential moving average, append it
            old_ema = self.exp_moving_averages[queue_name][-1]
            new_ema = ema_helper(new_price=new_p, old_ema=old_ema, num_of_days=max_l)
            self.exp_moving_averages[queue_name].append(new_ema)

    def compute_macd_related(self):
        """
        Compute MACD-related values (DIF, DEA).

        Args:
            None

        Returns:
            None
        """
        # when shorter than required, simply add None
        ema12, ema26 = (
            self.exp_moving_averages["12"][-1],
            self.exp_moving_averages["26"][-1],
        )
        if ema12 is None or ema26 is None:
            self.macd_diff.append(None)
            self.macd_dea.append(None)
            return

        # now, self.macd_diff and self.macd_dea have at least 26 elements
        macd_diff = ema12 - ema26
        self.macd_diff.append(macd_diff)
        # compute macd_dea case by case
        non_empty_diff = list(filter(lambda x: x is not None, self.macd_diff))

        if len(non_empty_diff) < DEA_NUM_OF_DAYS + 1:
            self.macd_dea.append(None)
        elif len(non_empty_diff) == DEA_NUM_OF_DAYS + 1:
            self.macd_dea.append(self.macd_diff[-1])
        else:
            old_dea = self.macd_dea[-1]
            new_dea = ema_helper(
                new_price=macd_diff, old_ema=old_dea, num_of_days=DEA_NUM_OF_DAYS
            )
            self.macd_dea.append(new_dea)

    def compute_kdj_related(
        self, d: datetime.datetime, low: float, high: float, open: float, close: float
    ):
        """
        Compute KDJ related arrays with dates, prices of lows, highs, opens and closes.

        Args:
            d (datetime.datetime): Date.
            low (float): Low price.
            high (float): High price.
            open (float): Open price.
            close (float): Close price.

        Returns:
            None
        """
        assert hasattr(self, "nine_day_rsv") and hasattr(
            self, "kdj_dct"
        ), "Incorrect initialization, KDJ related attributes missing!"

        def_KnD = 50
        # if only have at most 8 days' data, then append 0 to nine_days_rsv,
        # and use 50 for K and D
        if len(self.crypto_prices) <= 8:
            self.nine_day_rsv.append(0)
            self.kdj_dct["K"].append(def_KnD)
            self.kdj_dct["D"].append(def_KnD)
        else:
            highest_within_9 = max(x[4] for x in self.crypto_prices[-9:])
            lowest_within_9 = min(x[3] for x in self.crypto_prices[-9:])
            new_rsv = (
                100 * (close - lowest_within_9) / (highest_within_9 - lowest_within_9)
            )
            self.nine_day_rsv.append(new_rsv)

    def compute_rsi(self, period: int = 14):
        """
        Compute the Relative Strength Index (RSI) for the given period.

        Args:
            period (int, optional): RSI period. Defaults to 14.

        Returns:
            float or None: RSI value or None if not enough data.
        """
        if len(self.crypto_prices) < period + 1:
            return None
        closes = [x[0] for x in self.crypto_prices]
        deltas = np.diff(closes[-(period + 1) :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # basic functionality
    def _execute_one_buy(self, method: str, new_p: float):
        """
        Execute a buy action via a specific method.

        Args:
            method (str): Which strategy to use.
            new_p (float): Signals a new day's price of a currency.

        Returns:
            bool: Whether a buy action is executed.
        """
        if method not in self.strategies["buy"]:
            raise ValueError("unknown buy strategy!")
        if self.cash <= 0:
            if self.mode == "verbose":
                print("no more cash left, cannot buy anymore!")
            return False

        # execution body
        if method == "by_percentage":
            # Use percentage of available cash
            out_cash = self.cash * self.sell_pct
            self.cur_coin += out_cash * (1 - self.broker_pct) / new_p
            self.cash -= out_cash

        elif method == "fixed_amount":
            # Use fixed amount (buy_pct represents fixed USD amount)
            fixed_amount = (
                self.buy_pct
            )  # In this context, buy_pct is the fixed USD amount
            if fixed_amount > self.cash:
                fixed_amount = self.cash  # Can't spend more than we have
            self.cur_coin += fixed_amount * (1 - self.broker_pct) / new_p
            self.cash -= fixed_amount

        elif method == "market_order":
            # Market order - use all available cash
            out_cash = self.cash
            self.cur_coin += out_cash * (1 - self.broker_pct) / new_p
            self.cash = 0  # Spend all cash

        # Track the buy price for stop loss and take profit strategies
        if method in ["by_percentage", "fixed_amount", "market_order"]:
            self.last_buy_price = new_p

        return True

    def _execute_one_sell(self, method: str, new_p: float):
        """
        Execute a sell action via a specific method.

        Args:
            method (str): Which strategy to use.
            new_p (float): Signals a new day's price of a currency.

        Returns:
            bool: Whether a sell action is executed.
        """
        if method not in self.strategies["sell"]:
            raise ValueError("unknown sell strategy!")
        if self.cur_coin <= 0:
            if self.mode == "verbose":
                print("no coin left, cannot sell anymore!")
            return False

        # execution body
        if method == "by_percentage":
            # Use percentage of available crypto
            out_coin = self.cur_coin * self.sell_pct
            self.cur_coin -= out_coin
            self.cash += out_coin * new_p * (1 - self.broker_pct)

        elif method == "stop_loss":
            # Stop loss - sell if price drops below threshold
            # sell_pct represents the stop loss percentage below purchase price
            if hasattr(self, "last_buy_price") and self.last_buy_price:
                stop_loss_price = self.last_buy_price * (1 - self.sell_pct)
                if new_p <= stop_loss_price:
                    # Trigger stop loss - sell all crypto
                    out_coin = self.cur_coin
                    self.cur_coin = 0
                    self.cash += out_coin * new_p * (1 - self.broker_pct)
                else:
                    return False  # No sell action
            else:
                # No previous buy price, use percentage sell
                out_coin = self.cur_coin * self.sell_pct
                self.cur_coin -= out_coin
                self.cash += out_coin * new_p * (1 - self.broker_pct)

        elif method == "take_profit":
            # Take profit - sell if price rises above threshold
            # sell_pct represents the take profit percentage above purchase price
            if hasattr(self, "last_buy_price") and self.last_buy_price:
                take_profit_price = self.last_buy_price * (1 + self.sell_pct)
                if new_p >= take_profit_price:
                    # Trigger take profit - sell all crypto
                    out_coin = self.cur_coin
                    self.cur_coin = 0
                    self.cash += out_coin * new_p * (1 - self.broker_pct)
                else:
                    return False  # No sell action
            else:
                # No previous buy price, use percentage sell
                out_coin = self.cur_coin * self.sell_pct
                self.cur_coin -= out_coin
                self.cash += out_coin * new_p * (1 - self.broker_pct)

        return True

    def _record_history(self, new_p: float, d: datetime.datetime, action: str):
        """
        Record an event into trade history.

        Args:
            new_p (float): New price.
            d (datetime.datetime): Date.
            action (str): Action taken.

        Returns:
            None
        """
        item = {
            "action": action,
            "price": new_p,
            "date": d,
            "coin": self.cur_coin,
            "cash": self.cash,
            "portfolio": self.portfolio_value,
        }
        self.trade_history.append(item)

        if self.mode == "verbose":
            print(item)

    def deposit(self, c: float, new_p: float, d: datetime.datetime):
        """
        Deposit more cash to our wallet.

        Args:
            c (float): Amount to deposit.
            new_p (float): New price.
            d (datetime.datetime): Date.

        Returns:
            None
        Raises:
            ValueError: If c <= 0.
        """
        if c <= 0:
            raise ValueError("Should deposit a positive amount!")

        self.cash += c
        self.record_history(new_p, d, DEPOSIT_CST)

    def withdraw(self, c: float, new_p: float, d: datetime.datetime):
        """
        Withdraw some cash out of our wallet.

        Args:
            c (float): Amount to withdraw.
            new_p (float): New price.
            d (datetime.datetime): Date.

        Returns:
            None
        Raises:
            ValueError: If c > self.cash.
        """
        if c > self.cash:
            raise ValueError("Not enough cash to withdraw!")

        self.cash -= c
        self._record_history(new_p, d, WITHDRAW_CST)

    # properties, built upon functions defined above
    @property
    def all_history(self):
        """Returns all histories in complete fashion."""
        if len(self.trade_history) == 0:
            return []

        tmps = copy.deepcopy(self.trade_history)
        # for logs on the same day, if all logs are 'no action' take only 1; o/w, return all actions
        day_to_action = collections.defaultdict(list)
        for t in tmps:
            day_to_action[t["date"]].append(t)

        all_hist = []
        for d in day_to_action:
            if all([x["action"] == NO_ACTION_SIGNAL for x in day_to_action[d]]):
                all_hist.append(day_to_action[d][-1])
            else:
                today_trades = list(
                    filter(lambda x: x["action"] != NO_ACTION_SIGNAL, day_to_action[d])
                )
                all_hist.extend(today_trades)

        return all_hist

    @property
    def all_history_trade_only(self):
        """Returns all histories, filter out NO_ACTION items."""
        all_hist = self.all_history
        return list(filter(lambda x: x["action"] != NO_ACTION_SIGNAL, all_hist))

    @property
    def num_transaction(self):
        return len(self.all_history_trade_only)

    @property
    def num_buy_action(self):
        trades_only = self.all_history_trade_only
        return len([x for x in trades_only if x["action"] == BUY_SIGNAL])

    @property
    def num_sell_action(self):
        trades_only = self.all_history_trade_only
        return len([x for x in trades_only if x["action"] == SELL_SIGNAL])

    @property
    def max_drawdown(self):
        """Utilizes history to compute max draw-down."""
        all_hist = []
        if len(self.all_history) > 0:
            all_hist = list(map(lambda x: x["portfolio"], self.all_history))
        else:
            # no transaction happened at all
            # free to grab all this currency's data, re-compute
            for price_entry in self.crypto_prices:
                price = price_entry[0]
                port_value = self.cur_coin * price + self.cash
                all_hist.append(port_value)

        return max_drawdown_helper(all_hist)

    @property
    def baseline_rate_of_return(self):
        """Computes for baseline gain percentage (i.e. hold all coins, have 0 transaction)."""
        if len(self.all_history) == 0:
            # No trades made, compute baseline return using price data
            if len(self.crypto_prices) < 2:
                return 0.0  # Not enough price data
            init_price = self.crypto_prices[0][0]
            final_price = self.crypto_prices[-1][0]
            init_p = self.init_coin * init_price + self.init_cash
            final_p = self.init_coin * final_price + self.init_cash
            
            # Handle division by zero
            if init_p == 0:
                if final_p == 0:
                    return 0.0  # No change if both initial and final are zero
                else:
                    return float('inf') if final_p > 0 else float('-inf')  # Infinite return if starting from zero
            
            return np.round(100 * (final_p - init_p) / init_p, ROUND_PRECISION)
        else:
            init_p = self.all_history[0]["portfolio"]
            final_p = self.init_coin * self.all_history[-1]["price"] + self.init_cash
            
            # Handle division by zero
            if init_p == 0:
                if final_p == 0:
                    return 0.0  # No change if both initial and final are zero
                else:
                    return float('inf') if final_p > 0 else float('-inf')  # Infinite return if starting from zero
            
            return np.round(100 * (final_p - init_p) / init_p, ROUND_PRECISION)

    @property
    def rate_of_return(self):
        """Computes for the rate of return = (V_final - V_init) / V_init."""
        # compute init portfolio value
        init_p = self.init_coin * self.crypto_prices[0][0] + self.init_cash
        # compute final portfolio value
        final_p = self.cur_coin * self.crypto_prices[-1][0] + self.cash

        # Handle division by zero
        if init_p == 0:
            if final_p == 0:
                return 0.0  # No change if both initial and final are zero
            else:
                return float('inf') if final_p > 0 else float('-inf')  # Infinite return if starting from zero
        
        return np.round(100 * (final_p - init_p) / init_p, ROUND_PRECISION)

    @property
    def coin_rate_of_return(self):
        """How much the price of the currency has gone up."""
        if len(self.all_history) == 0:
            # No trades made, compute return using price data
            if len(self.crypto_prices) < 2:
                return 0.0  # Not enough price data
            init_price = self.crypto_prices[0][0]
            final_price = self.crypto_prices[-1][0]
            
            # Handle division by zero
            if init_price == 0:
                if final_price == 0:
                    return 0.0  # No change if both initial and final are zero
                else:
                    return float('inf') if final_price > 0 else float('-inf')  # Infinite return if starting from zero
            
            return np.round(100 * (final_price - init_price) / init_price, ROUND_PRECISION)
        else:
            init_price, final_price = (
                self.all_history[0]["price"],
                self.all_history[-1]["price"],
            )
            
            # Handle division by zero
            if init_price == 0:
                if final_price == 0:
                    return 0.0  # No change if both initial and final are zero
                else:
                    return float('inf') if final_price > 0 else float('-inf')  # Infinite return if starting from zero
            
            return np.round(100 * (final_price - init_price) / init_price, ROUND_PRECISION)

    @property
    def trade_signal(self):
        """Given a new day\'s price, determine if there\'s a suggested transaction.

        :argument

        :return
            (str)

        :raise
        """
        res = {
            "action": NO_ACTION_SIGNAL,
            "buy_percentage": self.buy_pct,
            "sell_percentage": self.sell_pct,
        }
        # find last date on which a transaction occurs
        last_evt = self.trade_history[-1] if len(self.trade_history) > 0 else {}
        last_date_str = last_evt.get("date", None)
        if last_date_str is None:
            return res

        # Support both date-only and datetime formats
        fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]
        last_date_dt_obj = None
        for fmt in fmts:
            try:
                last_date_dt_obj = datetime.datetime.strptime(last_date_str, fmt)
                break
            except:
                pass
        
        if last_date_dt_obj is None:
            return res

        # find today's date, and time delta
        td = datetime.datetime.now()
        diff = td - last_date_dt_obj

        # For sub-daily intervals, check if within the last interval period
        # Default to 6 hours if DATA_INTERVAL_HOURS is not available
        try:
            from core.config import DATA_INTERVAL_HOURS
            interval_hours = DATA_INTERVAL_HOURS
        except ImportError:
            interval_hours = 6
        
        # Check if within the last interval (e.g., 6 hours for 6h granularity)
        if diff.total_seconds() <= interval_hours * 3600:
            res["action"] = last_evt["action"]

        return res

    @property
    def portfolio_value(self):
        """Returns your latest portfolio value."""
        latest_price = self.crypto_prices[-1][0]
        p = latest_price * self.cur_coin + self.cash
        if p < 0:
            raise ValueError("portfolio value cannot be negative!")
        return p

    @property
    def brokerage_pct(self):
        """Returns brokerage percentage."""
        return self.broker_pct

    @property
    def trading_strategy(self):
        """Returns this object\'s trading strategy."""
        basic = {
            "buy_pct": self.buy_pct,
            "sell_pct": self.sell_pct,
            "tol_pct": self.tol_pct,
            "bollinger_sigma": self.bollinger_sigma,
        }
        return {**basic, **self.strategies}
