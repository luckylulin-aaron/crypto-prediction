import collections
import datetime
import numpy as np

from typing import List, Any


class MATrader:

    def __init__(self, name: str, init_amount: int, tol_pct: float, ma_lengths: List[int],
            buy_pct: float, sell_pct: float, cur_coin: float=0.0,
            buy_stas: List[str]=['by_percentage'], sell_stas: List[str]=['by_percentage']):
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
            return

        # o/w, we can do sth
        cur_avg = np.mean(self.moving_averages[queue_name])
        # (1) if too high, we do a sell
        if new_p >= (1 + self.tol_pct) * cur_avg:
            r = self.execute_one_sell('by_percentage', new_p)
            if r is True:
                self.record_history(new_p, d, 'sell')
        # (2) if too low, we do a buy
        if new_p <= (1 - self.tol_pct) * cur_avg:
            r = self.execute_one_buy('by_percentage', new_p)
            if r is True:
                self.record_history(new_p, d, 'buy')
        # add this new price
        self.moving_averages[queue_name].append(new_p)
        # chop from the left
        if len(self.moving_averages[queue_name]) > max_l:
            self.moving_averages[queue_name].popleft()

    def execute_one_buy(self, method: str, new_p: float):
        '''Execute a buy action via a specific method.'''
        if method not in self.strategies['buy']:
            raise ValueError('unknown buy strategy!')
        if self.cash <= 0:
            #print('no more cash left, cannot buy anymore!')
            return False
        # execution body
        if method == 'by_percentage':
            out_cash = self.cash * self.sell_pct
            self.cur_coin += out_cash / new_p
            self.cash -= out_cash
        return True

    def execute_one_sell(self, method: str, new_p: float):
        '''Execute a sell action via a specific method.'''
        if method not in self.strategies['sell']:
            raise ValueError('unknown sell strategy!')
        if self.cur_coin <= 0:
            #print('no coin left, cannot sell anymore!')
            return False
        # execution body
        if method == 'by_percentage':
            out_coin = self.cur_coin * self.sell_pct
            self.cur_coin -= out_coin
            self.cash += out_coin * new_p
        return True

    def record_history(self, new_p: float, d: datetime.datetime, action: str):
        '''Record an event into trade history.'''
        item = {
            'action': action,
            'price': new_p, 'date': d,
            'coin': self.cur_coin, 'cash': self.cash,
            'porfolio': self.portfolio_value
        }
        print(item)
        self.trade_history.append(item)

    def deposit(self, c: float, new_p: float, d: datetime.datetime):
        '''Deposit more cash to our wallent.'''
        self.cash += c
        self.record_history(new_p, d, 'deposit')

    @property
    def portfolio_value(self):
        '''Returns your current portfolio value.'''
        latest_price = self.crypto_prices[-1][0]
        p = latest_price * self.cur_coin + self.cash
        if p < 0:
            raise ValueError('portfolio value cannot be negative!')
        return p
