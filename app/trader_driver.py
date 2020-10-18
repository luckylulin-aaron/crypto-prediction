import numpy as np

from typing import List, Any

from ma_trader import MATrader
from util import timer

class TraderDriver:

    '''A wrapper class on top of any of trader classes.'''

    def __init__(self, name: str, init_amount: int, cur_coin: float,
            tol_pcts: List[float],
            ma_lengths: List[int],
            buy_pcts: List[float],
            sell_pcts: List[float],
            buy_stas: List[str] = ['by_percentage'],
            sell_stas: List[str] = ['by_percentage'],
            mode: str='normal'):

        self.init_amount, self.init_coin = init_amount, cur_coin
        self.mode = mode
        self.traders = []
        for tol_pct in tol_pcts:
            for buy_pct in buy_pcts:
                for sell_pct in sell_pcts:
                    t = MATrader(
                            name=name,
                            init_amount=init_amount,
                            tol_pct=tol_pct,
                            ma_lengths=ma_lengths,
                            buy_pct=buy_pct,
                            sell_pct=sell_pct,
                            cur_coin=cur_coin,
                            buy_stas=buy_stas,
                            sell_stas=sell_stas,
                            mode=mode
                        )
                    self.traders.append(t)

        # check
        if len(self.traders) != len(tol_pcts) * len(buy_pcts) * len(sell_pcts):
            raise ValueError('trader creation is wrong!')

    @timer
    def feed_data(self, data_stream: List[tuple]):
        '''Feed in historic data, where data_stream consists of tuples of (price, date).'''
        if self.mode == 'verbose':
            print('running simulation...')

        self.trader_indNgain = []

        for index,t in enumerate(self.traders):
            # compute initial value
            t.add_new_day(data_stream[0][0], data_stream[0][1])
            self.init_val = t.portfolio_value
            # run simulation
            for i in range(1, len(data_stream)):
                p,d = data_stream[i]
                t.add_new_day(p,d)
            # compute gain
            self.trader_indNgain.append((index, t.portfolio_value - self.init_val))

        # store some critical prices for later use
        self.first_p, self.last_p = data_stream[0][0], data_stream[-1][0]

    @property
    def best_trader_info(self):
        '''Find the best trading strategy for a given crypto-currency.'''
        best_t_ind, max_gain = self.trader_indNgain[0]
        # compute for baseline model (no transaction at all)
        bsl_gain = self.last_p * self.init_coin
        # sorely the currency's fluctuation
        bsl_delta_coin_pct = (self.last_p - self.first_p) / self.first_p

        for i in range(1, len(self.trader_indNgain)):
            t_ind,g = self.trader_indNgain[i]
            if g >= max_gain:
                max_gain = g
                best_t_ind = t_ind

        precision = 3
        extra = {
            'init_value': np.round(self.init_val, precision),
            'max_final_value': np.round(max_gain + self.init_val, precision),
            'gain_pct': str(np.round(max_gain / self.init_val * 100, precision)) + '%',
            'bsl_gain_pct': str(np.round(bsl_gain / self.init_val * 100, precision)) + '%',
            'bsl_coin_pct': str(np.round(bsl_delta_coin_pct * 100, precision)) + '%'
        }

        return {**self.traders[best_t_ind].trading_strategy, **extra, 'trader_index': best_t_ind}