import numpy as np
import os
import pandas as pd

from ma_trader import MATrader
from util import load_csv

def main():

    name,l = load_csv(csv_fn='ETH_HISTORY.csv')
    l = l[-365:]

    trader = MATrader(
        name=name,
        init_amount=5000,
        tol_pct=0.3,
        ma_lengths=[10,20,30],
        buy_pct=0.4,
        sell_pct=0.4,
        cur_coin=1.64,
        buy_stas=('by_percentage'),
        sell_stas=('by_percentage')
    )
    # init with first day
    trader.add_new_day(l[0][0], l[0][1])
    print('init value: {:.2f}'.format(trader.portfolio_value))

    for ind in range(1,len(l)):
        p,d = l[ind]
        trader.add_new_day(p,d)

    print('final value: {:.2f}'.format(trader.portfolio_value))

if __name__ == '__main__':

    main()