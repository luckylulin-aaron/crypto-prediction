# built-in packages
import copy
import datetime
import collections
import time

from typing import List, Any

# third-party packages
import numpy as np

# customized packages
from config import (BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL,
    DEPOSIT_CST, WITHDRAW_CST, STRATEGIES, ROUND_PRECISION, DEA_NUM_OF_DAYS)
from util import (max_drawdown_helper, ema_helper)
from strategies import STRATEGY_REGISTRY

class MATrader:

    '''Moving Average Trader, a strategic-trading implementation that focus and relies on moving averages.'''

    def __init__(self, name: str,
            init_amount: float,
            stat: str,
            tol_pct: float,
            ma_lengths: List[int],
            ema_lengths: List[int],
            bollinger_mas: List[int],
            bollinger_sigma: int,
            buy_pct: float,
            sell_pct: float,
            cur_coin: float=0.0,
            buy_stas: List[str]=['by_percentage'],
            sell_stas: List[str]=['by_percentage'],
            mode: str='normal'):
        """
        Initialize a MATrader instance.

        Args:
            name (str): Name of the crypto.
            init_amount (float): Initial cash amount.
            stat (str): Strategy name.
            tol_pct (float): Tolerance percentage.
            ma_lengths (List[int]): Moving average lengths.
            ema_lengths (List[int]): Exponential moving average lengths.
            bollinger_mas (List[int]): Bollinger MA lengths.
            bollinger_sigma (int): Bollinger sigma value.
            buy_pct (float): Buy percentage.
            sell_pct (float): Sell percentage.
            cur_coin (float, optional): Initial coin amount. Defaults to 0.0.
            buy_stas (List[str], optional): Buy strategies. Defaults to ['by_percentage'].
            sell_stas (List[str], optional): Sell strategies. Defaults to ['by_percentage'].
            mode (str, optional): Mode. Defaults to 'normal'.

        Returns:
            None
        """
        # check
        if stat not in STRATEGIES:
            raise ValueError('Unknown high-level trading strategy!')
        if not all([x in ma_lengths for x in bollinger_mas]):
            raise ValueError('cannot initialize Bollinger Band strategy if some of moving averages are not available!')
        # 1. Basic
        # crypto-currency name, tolerance percentage
        self.crypto_name, self.tol_pct = name, tol_pct
        # initial and current bitcoin at hand, and cash available
        self.init_coin, self.init_cash = cur_coin, init_amount
        self.cur_coin, self.cash = cur_coin, init_amount

        # 2. Transactions
        self.trade_history = []
        self.crypto_prices = []
        # moving average (type: dictionary)
        self.moving_averages = dict(zip([str(x) for x in ma_lengths],
                                        [[] for _ in range(len(ma_lengths))]))
        # exponential moving average queues and related items (type: dictionary)
        self.exp_moving_averages = dict(zip([str(x) for x in ema_lengths],
                                            [[] for _ in range(len(ema_lengths))]))
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
        self.strategies = {'buy': buy_stas, 'sell': sell_stas}
        # high-level strategy
        self.high_strategy = stat
        # debug mode
        self.mode = mode # normal, verbose
        # brokerage percentage
        self.broker_pct = 0.02
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
        open, low, high = misc_p['open'], misc_p['low'], misc_p['high']

        self.crypto_prices.append((new_p, d, open, low, high))

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
            
            if self.high_strategy == 'MA-SELVES':
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
                        sell_pct=self.sell_pct
                    )

            elif self.high_strategy == 'DOUBLE-MA':
                mas = [int(x) for x in list(self.moving_averages.keys())]
                mas = sorted(mas, reverse=False)

                for i in range(0, len(mas)):
                    for j in range(i+1, len(mas)):
                        # if too close, skip
                        if j - i <= 2:
                            continue
                        s_qn, l_qn = str(mas[i]), str(mas[j])
                        # if either of the two MAs does not have enough lengths, respectively, skip
                        if (self.moving_averages[s_qn][-1] is None or
                            self.moving_averages[l_qn][-1] is None):
                            continue
                        strategy_func(
                            trader=self,
                            shorter_queue_name=s_qn,
                            longer_queue_name=l_qn,
                            new_p=new_p,
                            today=d
                        )

            elif self.high_strategy == 'MACD':
                self.compute_macd_related()
                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d
                )

            elif self.high_strategy == 'BOLL-BANDS':
                for queue_name in self.bollinger_mas:
                    # if we don't have a moving average yet, skip
                    if self.moving_averages[queue_name][-1] is None:
                        continue
                    strategy_func(
                        trader=self,
                        queue_name=queue_name,
                        new_p=new_p,
                        today=d,
                        bollinger_sigma=self.bollinger_sigma
                    )

            elif self.high_strategy == 'RSI':
                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d
                )

            elif self.high_strategy == 'KDJ':
                strategy_func(
                    trader=self,
                    new_p=new_p,
                    today=d
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
        ema12, ema26 = self.exp_moving_averages['12'][-1], self.exp_moving_averages['26'][-1]
        if (ema12 is None or
            ema26 is None):
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
            new_dea = ema_helper(new_price=macd_diff, old_ema=old_dea, num_of_days=DEA_NUM_OF_DAYS)
            self.macd_dea.append(new_dea)

    def compute_kdj_related(self,
        d: datetime.datetime,
        low: float,
        high: float,
        open: float,
        close: float):
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
        assert hasattr(self, 'nine_day_rsv') and hasattr(self, 'kdj_dct'), \
            "Incorrect initialization, KDJ related attributes missing!"

        def_KnD = 50
        # if only have at most 8 days' data, then append 0 to nine_days_rsv,
        # and use 50 for K and D
        if len(self.crypto_prices) <= 8:
            self.nine_day_rsv.append(0)
            self.kdj_dct['K'].append(def_KnD)
            self.kdj_dct['D'].append(def_KnD)
        else:
            highest_within_9 = max(x[4] for x in self.crypto_prices[-9:])
            lowest_within_9 = min(x[3] for x in self.crypto_prices[-9:])
            new_rsv = 100 * (close - lowest_within_9) / (highest_within_9 - lowest_within_9)
            self.nine_day_rsv.append(new_rsv)

    def compute_rsi(self, period: int=14):
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
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Core Section: TRADING STRATEGY
    # Strategy implementations have been moved to strategies.py for better maintainability

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
        if method not in self.strategies['buy']:
            raise ValueError('unknown buy strategy!')
        if self.cash <= 0:
            if self.mode == 'verbose':
                print('no more cash left, cannot buy anymore!')
            return False
        # execution body
        if method == 'by_percentage':
            out_cash = self.cash * self.sell_pct
            self.cur_coin += out_cash * (1 - self.broker_pct) / new_p
            self.cash -= out_cash
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
        if method not in self.strategies['sell']:
            raise ValueError('unknown sell strategy!')
        if self.cur_coin <= 0:
            if self.mode == 'verbose':
                print('no coin left, cannot sell anymore!')
            return False
        # execution body
        if method == 'by_percentage':
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
            'action': action,
            'price': new_p,
            'date': d,
            'coin': self.cur_coin,
            'cash': self.cash,
            'portfolio': self.portfolio_value
        }
        self.trade_history.append(item)

        if self.mode == 'verbose':
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
            raise ValueError('Should deposit a positive amount!')

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
            raise ValueError('Not enough cash to withdraw!')

        self.cash -= c
        self._record_history(new_p, d, WITHDRAW_CST)

    # properties, built upon functions defined above
    @property
    def all_history(self):
        '''Returns all histories in complete fashion.'''
        if len(self.trade_history) == 0:
            return []

        tmps = copy.deepcopy(self.trade_history)
        # for logs on the same day, if all logs are 'no action' take only 1; o/w, return all actions
        day_to_action = collections.defaultdict(list)
        for t in tmps:
            day_to_action[t['date']].append(t)

        all_hist = []
        for d in day_to_action:
            if all([x['action'] == NO_ACTION_SIGNAL for x in day_to_action[d]]):
                all_hist.append(day_to_action[d][-1])
            else:
                today_trades = list(filter(lambda x: x['action'] != NO_ACTION_SIGNAL, day_to_action[d]))
                all_hist.extend(today_trades)

        return all_hist

    @property
    def all_history_trade_only(self):
        '''Returns all histories, filter out NO_ACTION items.'''
        all_hist = self.all_history
        return list(filter(lambda x: x['action'] != NO_ACTION_SIGNAL, all_hist))

    @property
    def num_transaction(self):
        return len(self.all_history_trade_only)

    @property
    def num_buy_action(self):
        trades_only = self.all_history_trade_only
        return len([x for x in trades_only if x['action'] == BUY_SIGNAL])

    @property
    def num_sell_action(self):
        trades_only = self.all_history_trade_only
        return len([x for x in trades_only if x['action'] == SELL_SIGNAL])

    @property
    def max_drawdown(self):
        '''Utilizes history to compute max draw-down.'''
        all_hist = []
        if len(self.all_history) > 0:
            all_hist = list(map(lambda x: x['portfolio'], self.all_history))
        else:
            # no transaction happened at all
            # free to grab all this currency's data, re-compute
            for (price, _) in self.crypto_prices:
                port_value = self.cur_coin * price + self.cash
                all_hist.append(port_value)

        return max_drawdown_helper(all_hist)

    @property
    def baseline_rate_of_return(self):
        '''Computes for baseline gain percentage (i.e. hold all coins, have 0 transaction).'''
        init_p = self.all_history[0]['portfolio']
        final_p = self.init_coin * self.all_history[-1]['price'] + self.init_cash
        return np.round(100 * (final_p - init_p) / init_p, ROUND_PRECISION)

    @property
    def rate_of_return(self):
        '''Computes for the rate of return = (V_final - V_init) / V_init.'''
        # compute init portfolio value
        init_p = self.init_coin * self.crypto_prices[0][0] + self.init_cash
        # compute final portfolio value
        final_p = self.cur_coin * self.crypto_prices[-1][0] + self.cash

        return np.round(100 * (final_p - init_p) / init_p, ROUND_PRECISION)

    @property
    def coin_rate_of_return(self):
        '''How much the price of the currency has gone up.'''
        init_price, final_price = self.all_history[0]['price'], self.all_history[-1]['price']
        return np.round(100 * (final_price - init_price) / init_price, ROUND_PRECISION)

    @property
    def trade_signal(self):
        '''Given a new day\'s price, determine if there\'s a suggested transaction.

        :argument

        :return
            (str)

        :raise
        '''
        res = {
            'action': NO_ACTION_SIGNAL,
            'buy_percentage': self.buy_pct,
            'sell_percentage': self.sell_pct
        }
        # find last date on which a transaction occurs
        last_evt = self.trade_history[-1] if len(self.trade_history) > 0 else {}
        last_date_str = last_evt.get('date', None)
        if last_date_str is None:
            return res

        fmts = ['%Y-%m-%d', '%m/%d/%Y']
        for fmt in fmts:
            try:
                last_date_dt_obj = datetime.datetime.strptime(last_date_str, fmt)
            except:
                pass

        # find today's date, and time delta
        td = datetime.datetime.now()
        diff = td - last_date_dt_obj

        if diff.days <= 1:
            res['action'] = last_evt['action']

        return res

    @property
    def portfolio_value(self):
        '''Returns your latest portfolio value.'''
        latest_price = self.crypto_prices[-1][0]
        p = latest_price * self.cur_coin + self.cash
        if p < 0:
            raise ValueError('portfolio value cannot be negative!')
        return p

    @property
    def brokerage_pct(self):
        '''Returns brokerage percentage.'''
        return self.broker_pct

    @property
    def trading_strategy(self):
        '''Returns this object\'s trading strategy.'''
        basic = {
            'buy_pct': self.buy_pct,
            'sell_pct': self.sell_pct,
            'tol_pct': self.tol_pct,
            'bollinger_sigma': self.bollinger_sigma
        }
        return {**basic, **self.strategies}