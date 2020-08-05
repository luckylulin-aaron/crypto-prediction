import numpy as np
import os
import pandas as pd

from trader_driver import TraderDriver
from util import load_csv


def main():

    name,l = load_csv(csv_fn='ETH_HISTORY.csv')
    l = l[-200:]

    # configuration
    name = name
    init_amount = 5000
    tol_pcts = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ma_lengths = [10, 20, 30]
    buy_pcts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sell_pcts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cur_coin = 0.5
    buy_stas = ('by_percentage')
    sell_stas = ('by_percentage')
    mode = 'normal'

    t_driver = TraderDriver(
        name=name,
        init_amount=init_amount,
        cur_coin=cur_coin,
        tol_pcts=tol_pcts,
        ma_lengths=ma_lengths,
        buy_pcts=buy_pcts,
        sell_pcts=sell_pcts,
        buy_stas=buy_stas,
        sell_stas=sell_stas,
        mode=mode
    )

    t_driver.feed_data(data_stream=l)
    info = t_driver.best_trader_info

    # for a new price, find the trade signal
    signal = t_driver.traders[info['trader_index']].trade_signal
    print(signal)


if __name__ == '__main__':

    main()