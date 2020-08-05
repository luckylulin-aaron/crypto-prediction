import datetime
import numpy as np
import os
import pandas as pd

from cbpro_client import CBProClient
from trader_driver import TraderDriver
from config import *
from util import load_csv


def main():
    '''Test cbpro functionality.'''
    client = CBProClient()

    cur_name = 'ETH'
    assert cur_name in CURS, 'undefined currency name!'

    r1 = client.get_cur_rate(cur_name + '-USD')
    data_stream = client.get_historic_data(cur_name + '-USD', SECONDS_IN_ONE_DAY)

    init_amount = 900
    cur_coin = 0.05
    mode = 'normal'

    t_driver = TraderDriver(
        name=cur_name,
        init_amount=init_amount,
        cur_coin=cur_coin,
        tol_pcts=TOL_PCTS,
        ma_lengths=MA_LENGTHS,
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

    print('DONE:', best_t.crypto_name, signal)

if __name__ == '__main__':

    main()