# built-in packages
import math

from typing import List, Any

# third-party packages
import numpy as np

# customized packages
from config import ROUND_PRECISION
from ma_trader import MATrader
from util import timer


class TraderDriver:

    '''A wrapper class on top of any of trader classes.'''

    def __init__(self,
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
            buy_stas: List[str] = ['by_percentage'],
            sell_stas: List[str] = ['by_percentage'],
            mode: str='normal'):

        self.init_amount, self.init_coin = init_amount, cur_coin
        self.mode = mode
        self.traders = []
        for bollinger_sigma in bollinger_tols:
            for stat in overall_stats:
                for tol_pct in tol_pcts:
                    for buy_pct in buy_pcts:
                        for sell_pct in sell_pcts:
                            t = MATrader(
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
                                    mode=mode
                                )
                            self.traders.append(t)

        # check
        if len(self.traders) != (len(tol_pcts) * len(buy_pcts) *
                                 len(sell_pcts) * len(overall_stats) *
                                 len(bollinger_tols)):
            raise ValueError('trader creation is wrong!')
        # unknown, without data
        self.best_trader = None

    @timer
    def feed_data(self, data_stream: List[tuple]):
        '''Feed in historic data, where data_stream consists of tuples of (price, date).'''
        if self.mode == 'verbose':
            print('running simulation...')

        max_final_p = -math.inf

        for index,t in enumerate(self.traders):
            # compute initial value
            t.add_new_day(
                new_p=data_stream[0][0],
                d=data_stream[0][1]
            )
            # run simulation
            for i in range(1, len(data_stream)):
                p,d = data_stream[i]
                t.add_new_day(p,d)
            # decide best trader while we loop, by comparing all traders final portfolio value
            # sometimes a trader makes no trade at all
            if len(t.all_history) > 0:
                tmp_final_p = t.all_history[-1]['portfolio']
            # o/w, compute it
            else:
                tmp_final_p = (t.crypto_prices[-1][0] * t.cur_coin) + t.cash
            '''
            try:
                tmp_final_p = t.all_history[-1]['portfolio']
            except IndexError as e:
                print('Found error!', t.high_strategy)
            '''
            if tmp_final_p >= max_final_p:
                max_final_p = tmp_final_p
                self.best_trader = t

    @property
    def best_trader_info(self):
        '''Find the best trading strategy for a given crypto-currency.'''
        best_trader = self.best_trader

        # compute init value once again, in case no single trade is made
        init_v = best_trader.init_coin * best_trader.crypto_prices[0][0] + best_trader.init_cash

        extra = {
            'init_value': np.round(init_v, ROUND_PRECISION),
            'max_final_value': np.round(best_trader.portfolio_value, ROUND_PRECISION),
            'rate_of_return': str(best_trader.rate_of_return) + '%',
            'baseline_rate_of_return': str(best_trader.baseline_rate_of_return) + '%',
            'coin_rate_of_return': str(best_trader.coin_rate_of_return) + '%'
        }

        return {**best_trader.trading_strategy, **extra, 'trader_index': self.traders.index(self.best_trader)}