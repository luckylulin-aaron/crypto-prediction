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
        '''Add a new day\'s crypto-currency price, find out a computed transaction for today.'''
        open, low, high = misc_p['open'], misc_p['low'], misc_p['high']

        self.crypto_prices.append((new_p, d, open, low, high))

        # add new moving averages and exponential moving averages, get_historic_data respectively
        for queue_name in self.moving_averages:
            self.add_new_moving_averages(queue_name=queue_name, new_p=new_p)
        for queue_name in self.exp_moving_averages:
            self.add_new_exponential_moving_averages(queue_name=queue_name, new_p=new_p)

        # compute KDJ related arrays
        self.compute_kdj_related((d,low,high,open,new_p))

        # Execute trading strategy
        # [1] MA w/ itself
        if self.high_strategy == 'MA-SELVES':
            for queue_name in self.moving_averages:
                # if we don't have a moving average yet, skip
                if self.moving_averages[queue_name][-1] is None:
                    continue
                self.strategy_moving_average_w_tolerance(
                    queue_name=queue_name,
                    new_p=new_p,
                    today=d
                )

        # [2] pair-wise MAs
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
                    self.strategy_double_moving_averages(
                        shorter_queue_name=s_qn,
                        longer_queue_name=l_qn,
                        new_p=new_p,
                        today=d
                    )

        # [3] MACD divergence/convergence approach
        elif self.high_strategy == 'MACD':
            self.compute_macd_related()
            self.strategy_macd(new_p=new_p, today=d)

        # [4] Bollinger Band approach
        elif self.high_strategy == 'BOLL-BANDS':
            for queue_name in self.bollinger_mas:
                # if we don't have a moving average yet, skip
                if self.moving_averages[queue_name][-1] is None:
                    continue
                self.strategy_bollinger_bands(
                    queue_name=queue_name,
                    new_p=new_p,
                    today=d
                )

    def add_new_moving_averages(self, queue_name: str, new_p: float):
        '''Compute and append a new moving average price.'''
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
        '''Compute and append a new exponential moving average price.'''
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
        '''Compute macd_diff: 差离值（DIF）的计算： DIF = EMA12 - EMA26
         and macd_dea: 差离值(above)的9日的EMA.'''

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
        low: float, high: float,
        open: float, close: float):
        '''Compute KDJ related arrays with dates, prices of lows, highs, opens and closes.
        Reference: https://www.zcaijing.com/tzzjygs/28735.html.
        '''
        # checking
        assert hasattr(MATrader, 'nine_day_rsv') and hasattr(MATrader, 'kdj_dct'), \
            'Incorrect initialization, KDJ related attributes missing!'

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

    # Core Section: TRADING STRATEGY
    def strategy_moving_average_w_tolerance(self, queue_name: str,
            new_p: float, today: datetime.datetime):
        '''For a new day's price, find if beyond tolerance level, execute a buy or sell action.

        Args:
            queue_name (str): The name of the queue.
            new_p (float): Today's new price of a currency.
            today (datetime.datetime):
        '''
        strat_name = 'MA-SELVES'
        assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

        # retrieve the most recent moving average
        last_ma = self.moving_averages[queue_name][-1]
        # (1) if too high, we do a sell
        r_sell = False
        if new_p >= (1 + self.tol_pct) * last_ma:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
                self.strat_dct[strat_name].append((today, SELL_SIGNAL))
        # (2) if too low, we do a buy
        r_buy = False
        if new_p <= (1 - self.tol_pct) * last_ma:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
                self.strat_dct[strat_name].append((today, BUY_SIGNAL))
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    def strategy_double_moving_averages(self, shorter_queue_name: str, longer_queue_name,
            new_p: float, today: datetime.datetime):
        '''For a new day's price, if the shorter MA is greater than the longer MA's, execute a buy;
        otherwise, execute a sell.

        Args:
            shorter_queue_name (str): The name of the queue with shorter interval.
            longer_queue_name (str): The name of the queue with longer interval.
            new_p (float): Today's new price of a currency.
            today (datetime.datetime):
        '''
        strat_name = 'DOUBLE-MA'
        assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

        # compute the moving averages for shorter queue and longer queue, respectively
        shorter_p = self.moving_averages[shorter_queue_name][-1]
        longer_p = self.moving_averages[longer_queue_name][-1]

        # (1) if a 'death-cross', sell
        r_sell = False
        if longer_p > shorter_p:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
                self.strat_dct[strat_name].append((today, SELL_SIGNAL))
        # (2) if a 'golden-cross', buy
        r_buy = False
        if shorter_p > longer_p:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
                self.strat_dct[strat_name].append((today, BUY_SIGNAL))
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    def strategy_macd(self, new_p, today: datetime.datetime):
        '''By default, only have 12 and 26 as keys in self.exp_moving_averages, and use 9 for computing
        the EMA for MACD diff.

        Args:
            new_p (float): Today's new price of a currency.
            today (datetime.datetime):
        '''
        strat_name = 'MACD'
        assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

        if self.macd_dea[-1] is None:
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return

        diff = self.macd_diff[-1] - self.macd_dea[-1]
        # (1) if (MACD-DIFF - MACD-DEA) is negative, sell
        r_sell = False
        if diff < 0:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
                self.strat_dct[strat_name].append((today, SELL_SIGNAL))
        # (2) if (MACD-DIFF - MACD-DEA) is, buy
        r_buy = False
        if diff > 0:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
                self.strat_dct[strat_name].append((today, BUY_SIGNAL))
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    def strategy_bollinger_bands(self, queue_name: str,
            new_p, today: datetime.datetime):
        '''Trading with Bollinger Band strategy.
        Buy signal: if today's price <= X's MA - sigma * standard_deviation_of_the_X's_MA;
        Sell signal: if today's price >= X's MA + sigma * standard_deviation_of_the_X's_MA.

        Args:
            queue_name (str): The name of the queue.
            new_p (float): Today's new price of a currency.
            today (datetime.datetime):
        '''
        strat_name = 'BOLL-BANDS'
        assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

        # if the moving average queue is not long enough to consider, skip
        not_null_values = list(filter(lambda x: x is not None, self.moving_averages[queue_name]))
        if len(not_null_values) < int(queue_name):
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return

        # want the last X number of items
        mas = not_null_values[-int(queue_name):]
        # compute mean and standard deviation
        ma_mean, ma_std = np.mean(mas), np.std(mas)

        # compute upper and lower bound
        boll_upper = ma_mean + self.bollinger_sigma * ma_std
        boll_lower = ma_mean - self.bollinger_sigma * ma_std

        # (1) if the price exceeds the upper bound, sell
        r_sell = False
        if new_p >= boll_upper:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
                self.strat_dct[strat_name].append((today, SELL_SIGNAL))
        # (2) if the price drops below the lower bound, buy
        r_buy = False
        if new_p <= boll_lower:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
                self.strat_dct[strat_name].append((today, BUY_SIGNAL))
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)
            self.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    # basic functionality
    def _execute_one_buy(self, method: str, new_p: float):
        '''Execute a buy action via a specific method.

        Args:
            method (str): Which strategy to use.
            new_p (float): Signals a new day's price of a currency.

        Returns:
            (bool): Whether a buy action is executed, given the
        '''
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
        '''Execute a sell action via a specific method.'''
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
        '''Record an event into trade history.'''
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
        '''Deposit more cash to our wallet.'''
        if c <= 0:
            raise ValueError('Should deposit a positive amount!')

        self.cash += c
        self.record_history(new_p, d, DEPOSIT_CST)

    def withdraw(self, c: float, new_p: float, d: datetime.datetime):
        '''Withdraw some cash out of our wallet.'''
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