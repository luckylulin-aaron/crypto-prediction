import copy
import collections
import datetime
import numpy as np

from typing import List, Any

from config import (BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL)


class MATrader:

    '''Moving Average Trader, a strategic-trading implementation that focus and relies on moving averages.'''

    def __init__(self, name: str, init_amount: float, tol_pct: float, ma_lengths: List[int],
            buy_pct: float, sell_pct: float, cur_coin: float=0.0,
            buy_stas: List[str]=['by_percentage'], sell_stas: List[str]=['by_percentage'],
            mode: str='normal'):
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
        # debug mode
        self.mode = mode # normal, verbose
        # brokerage percentage
        self.broker_pct = 0.02

    def add_new_day(self, new_p: float, d: datetime.datetime):
        '''Add a new day\'s crypto-currency price, see what we should do.'''
        self.crypto_prices.append((new_p, d))
        for q in self.moving_averages:
            self.process_queues(q, new_p, d)

    def process_queues(self, queue_name: str, new_p: float, d: datetime.datetime):
        '''For a new day\'s price, find if beyond tolerance level, execute a buy or sell action.'''
        # if too short the queue, simply add the new price and leave
        max_l = int(queue_name)
        cur_l = len(self.moving_averages[queue_name])
        if cur_l < max_l:
            self.moving_averages[queue_name].append(new_p)
            self._record_history(new_p, d, NO_ACTION_SIGNAL)
            return

        # o/w, we can do sth
        cur_avg = np.mean(self.moving_averages[queue_name])
        # (1) if too high, we do a sell
        r_sell = False
        if new_p >= (1 + self.tol_pct) * cur_avg:
            r_sell = self._execute_one_sell('by_percentage', new_p)
            if r_sell is True:
                self._record_history(new_p, d, SELL_SIGNAL)
        # (2) if too low, we do a buy
        r_buy = False
        if new_p <= (1 - self.tol_pct) * cur_avg:
            r_buy = self._execute_one_buy('by_percentage', new_p)
            if r_buy is True:
                self._record_history(new_p, d, BUY_SIGNAL)
        # add this new price
        self.moving_averages[queue_name].append(new_p)
        # chop from the left
        if len(self.moving_averages[queue_name]) > max_l:
            self.moving_averages[queue_name].popleft()
        # add history
        if (r_buy is False and
            r_sell is False):
            self._record_history(new_p, d, NO_ACTION_SIGNAL)

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
        self.cash += c
        self.record_history(new_p, d, 'deposit')

    @property
    def max_drawdown(self):
        '''Compute max draw-down of trader. This metric gives us insight on the risks.'''
        raise NotImplementedError('pending')

    @property
    def all_history(self):
        '''Returns all histories in complete fashion.'''
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
        all_hist = self.all_history
        value_cur, value_max_pre = all_hist[0]['portfolio'], all_hist[0]['portfolio']

        l = []
        for hist in all_hist[1:]:
            value_max_pre = max(value_max_pre, hist['portfolio'])
            value_cur = hist['portfolio']
            l.append(1 - value_cur / value_max_pre)

        return np.round(max(l), 4)

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
        last_date_str = self.trade_history[-1].get('date', None)
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