import copy
import collections
import datetime
import numpy as np
import time

from typing import List, Any

from config import (BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL,
    DEPOSIT_CST, WITHDRAW_CST, STRATEGIES)
from util import (max_drawdown_helper)

class MATrader:

    '''Moving Average Trader, a strategic-trading implementation that focus and relies on moving averages.'''

    def __init__(self, name: str,
            init_amount: float,
            stat: str,
            tol_pct: float,
            ma_lengths: List[int],
            buy_pct: float,
            sell_pct: float,
            cur_coin: float=0.0,
            buy_stas: List[str]=['by_percentage'],
            sell_stas: List[str]=['by_percentage'],
            mode: str='normal'):
        # check
        if stat not in STRATEGIES:
            raise ValueError('Unknown high-level trading strategy!')
        # 1. Basic
        # crypto-currency name, tolerance percentage
        self.crypto_name, self.tol_pct = name, tol_pct
        # current bitcoin at hand, and cash available
        self.cur_coin, self.cash = cur_coin, init_amount
        # 2. Transactions
        self.trade_history = []
        self.crypto_prices = collections.deque()
        # moving average queues
        self.moving_averages = dict()
        for ma_l in ma_lengths:
            self.moving_averages[str(ma_l)] = collections.deque()
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

    def add_new_day(self, new_p: float, d: datetime.datetime):
        '''Add a new day\'s crypto-currency price, find out a computed transaction for today.'''
        self.crypto_prices.append((new_p, d))

        # add new moving averages
        for queue_name in self.moving_averages:
            self.compute_new_moving_averages(queue_name=queue_name, new_p=new_p, today=d)

        # execute trading strategy
        # [1] MA w/ itself
        if self.high_strategy == 'MA-SELVES':
            for queue_name in self.moving_averages:
                ma_len = int(queue_name)
                # if MA does not have enough length, skip
                if len(self.moving_averages[queue_name]) < ma_len:
                    continue
                self.strategy_moving_average_w_tolerance(
                    queue_name=queue_name,
                    new_p=new_p,
                    today=d,
                    max_l=ma_len
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
                    if (len(self.moving_averages[s_qn]) < int(s_qn) or
                        len(self.moving_averages[l_qn]) < int(l_qn)):
                        continue
                    self.strategy_double_moving_averages(
                        shorter_queue_name=s_qn,
                        longer_queue_name=l_qn,
                        new_p=new_p,
                        today=d
                    )

    def compute_new_moving_averages(self, queue_name: str, new_p: float, today: datetime.datetime):
        max_l = int(queue_name)
        cur_l = len(self.moving_averages[queue_name])
        if cur_l < max_l:
            self.moving_averages[queue_name].append(new_p)
            return

    # Core Section: TRADING STRATEGY
    def strategy_moving_average_w_tolerance(self, queue_name: str,
            new_p: float, today: datetime.datetime, max_l: int):
        '''For a new day's price, find if beyond tolerance level, execute a buy or sell action.

        Args:
            queue_name (str): The name of the queue.
            new_p (float): Today's new price of a currency.
            today (datetime.datetime):
            max_l (int): The length of the moving average. For example, for MA5, max_l = 5.
        '''
        # o/w, we can do sth
        cur_avg = np.mean(self.moving_averages[queue_name])
        # (1) if too high, we do a sell
        r_sell = False
        if new_p >= (1 + self.tol_pct) * cur_avg:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
        # (2) if too low, we do a buy
        r_buy = False
        if new_p <= (1 - self.tol_pct) * cur_avg:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
        # add this new price
        self.moving_averages[queue_name].append(new_p)
        # chop from the left
        if len(self.moving_averages[queue_name]) > max_l:
            self.moving_averages[queue_name].popleft()
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)

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
        shorter_p = self.moving_averages[shorter_queue_name][-1]
        longer_p = self.moving_averages[longer_queue_name][-1]

        # (1) if a 'death-cross', sell
        r_sell = False
        if longer_p > shorter_p:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, today, SELL_SIGNAL)
        # (2) if a 'golden-cross', buy
        r_buy = False
        if shorter_p > longer_p:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, today, BUY_SIGNAL)
        # add history as well if nothing happens
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, today, NO_ACTION_SIGNAL)

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
            'price': new_p, 'date': d,
            'coin': self.cur_coin, 'cash': self.cash,
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
        all_hist = self.all_history
        return list(filter(lambda x: x['action'] != NO_ACTION_SIGNAL, all_hist))

    @property
    def num_transaction(self):
        return len(self.all_history_trade_only)

    @property
    def max_drawdown(self):
        '''Utilizes history to compute max draw-down.'''
        all_hist = []
        if len(self.all_history) > 0:
            all_hist = list(map(lambda x: x['portfolio'], self.all_history))
        else:
            # need to grab crypto-currency historical data, re-compute
            for (price, _) in self.crypto_prices:
                port_value = self.cur_coin * price + self.cash
                all_hist.append(port_value)

        return max_drawdown_helper(all_hist)

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
        '''Returns your current portfolio value.'''
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
            'tol_pct': self.tol_pct
        }
        return {**basic, **self.strategies}