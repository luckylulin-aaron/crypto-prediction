# built-in packages
import sys
import unittest

# third-party packages
import datetime

# customized packages
sys.path.insert(0, '../../app')

from config import *
from credentials import (CB_API_KEY, CB_API_SECRET)
from cbpro_client import CBProClient
from trader_driver import TraderDriver
from util import *


class TestMADriver(unittest.TestCase):

    def test_regular_load_from_local(self):
        '''Test loading data stream from local file.'''

        name, l = load_csv(csv_fn='../fixtures/BTC_HISTORY.csv')
        l = l[-120:]

        #print('debug:', type(l[0][0]), type(l[0][1]), type(l[0]))

        init_amount = 3000
        cur_coin = 5
        mode = 'normal'

        t_driver = TraderDriver(
            name=name,
            init_amount=init_amount,
            overall_stats=STRATEGIES,
            cur_coin=cur_coin,
            tol_pcts=TOL_PCTS,
            ma_lengths=MA_LENGTHS,
            ema_lengths=EMA_LENGTHS,
            bollinger_mas=BOLLINGER_MAS,
            bollinger_tols=BOLLINGER_TOLS,
            buy_pcts=BUY_PCTS,
            sell_pcts=SELL_PCTS,
            buy_stas=BUY_STAS,
            sell_stas=SELL_STAS,
            mode=mode
        )

        t_driver.feed_data(data_stream=l)

        info = t_driver.best_trader_info
        # best trader
        best_t = t_driver.traders[info['trader_index']]
        # for a new price, find the trade signal
        signal = best_t.trade_signal

        # check
        print('examine best trader performance:', info)
        self.assertNotEqual(info['trader_index'], None), 'incorrect best trader index!'

    def test_run_with_cbpro(self):
        '''Test running with CBProClient that requires internet connection.'''
        return
        client = CBProClient()

        for cur_name in CURS:
            r1 = client.get_cur_rate(cur_name + '-USD')
            data_stream = client.get_historic_data(cur_name + '-USD', SECONDS_IN_ONE_DAY)

            init_amount = 3000
            cur_coin = 5
            mode = 'normal'

            t_driver = TraderDriver(
                name=cur_name,
                init_amount=init_amount,
                overall_stats=STRATEGIES,
                cur_coin=cur_coin,
                tol_pcts=TOL_PCTS,
                ma_lengths=MA_LENGTHS,
                ema_lengths=EMA_LENGTHS,
                bollinger_mas=BOLLINGER_MAS,
                bollinger_tols=BOLLINGER_TOLS,
                buy_pcts=BUY_PCTS,
                sell_pcts=SELL_PCTS,
                buy_stas=BUY_STAS,
                sell_stas=SELL_STAS,
                mode=mode
            )

            t_driver.feed_data(data_stream=data_stream)

            info = t_driver.best_trader_info
            # best trader
            best_t = t_driver.traders[info['trader_index']]

            # for a new price, find the trade signal
            new_p = client.get_cur_rate(cur_name + '-USD')
            today = datetime.datetime.now().strftime('%m/%d/%Y')
            # add it and compute
            best_t.add_new_day(new_p, today)
            signal = best_t.trade_signal

            # check
            self.assertEqual(best_t.crypto_name, cur_name), 'incorrect crypto name!'
            self.assertIsInstance(signal, dict), 'trading signal must be of type dictionary!'
            print('examine best trader performance:', info)
            print(f'for crypto={best_t.crypto_name}, today\'s signal={signal}')


if __name__ == '__main__':

    unittest.main()
