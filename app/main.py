import datetime
import json
import numpy as np
import os
import pandas as pd
import schedule
import time

from cbpro_client import CBProClient
from credentials import (CB_API_KEY, CB_API_SECRET)
from trader_driver import TraderDriver
from config import *
from util import (display_port_msg, load_csv)


def main():
    '''Run simulation and make trades.'''
    log_file = './log.txt'

    # cbpro client to interact with coinbase
    client = CBProClient(
        key=CB_API_KEY,
        secret=CB_API_SECRET
    )

    fake_cur_coin, fake_cash = 5, 3000
    mode = 'normal'

    # before
    v_c1, v_f1 = client.portfolio_value
    display_port_msg(v_c=v_c1, v_f=v_f1, before=True)

    for index,cur_name in enumerate(CURS):
        print(f'\n[{index+1}] processing for currency={cur_name}...')
        cur_rate = client.get_cur_rate(cur_name + '-USD')
        data_stream = client.get_historic_data(cur_name + '-USD', SECONDS_IN_ONE_DAY)

        # initial cash amount
        _, cash = client.portfolio_value
        # initial coin at hand
        wallet = client.get_wallets(cur_names=[cur_name])
        cur_coin, wallet_id = None, None
        for item in wallet:
            if item['currency'] == cur_name and item['type'] == 'wallet':
                cur_coin, wallet_id = float(item['balance']['amount']), item['id']

        assert cur_coin and wallet_id, f'cannot find relevant wallet for {cur_name}!'

        # run simulation
        t_driver = TraderDriver(
            name=cur_name,
            init_amount=fake_cash,
            cur_coin=fake_cur_coin,
            tol_pcts=TOL_PCTS,
            ma_lengths=MA_LENGTHS,
            buy_pcts=BUY_PCTS,
            sell_pcts=SELL_PCTS,
            buy_stas=BUY_STAS,
            sell_stas=SELL_STAS,
            mode=mode
        )

        t_driver.feed_data(data_stream)

        info = t_driver.best_trader_info
        # best trader
        best_t = t_driver.traders[info['trader_index']]

        # for a new price, find the trade signal
        new_p = client.get_cur_rate(cur_name + '-USD')
        today = datetime.datetime.now().strftime('%m/%d/%Y')
        # add it and compute
        best_t.add_new_day(new_p, today)
        signal = best_t.trade_signal

        print('examine best trader performance:', info)
        print(f'for crypto={best_t.crypto_name}, today\'s signal={signal}')

        # if too less, we leave
        if cash <= EP_CASH:
            print('[warning] too less cash, discard further actions.')
            continue
        elif cur_coin <= EP_COIN or cur_coin * new_p <= EP_CASH:
            print('[warning] too less crypto, discard further actions.')
            continue

        # otherwise, execute a transaction
        if signal['action'] == BUY_SIGNAL:
            buy_pct = signal['buy_percentage']
            order = client.place_buy_order(
                wallet_id=wallet_id,
                amount=buy_pct * cash,
                currency='USD',
                commit=COMMIT
            )
            print('bought {:.5f} {}, used {:.2f} USD, at unit price={}'.format(
                float(order['amount']['amount']), cur_name,
                float(order['total']['amount']), float(order['unit_price']['amount'])
            ))

        elif signal['action'] == SELL_SIGNAL:
            sell_pct = signal['sell_percentage']
            order = client.place_sell_order(
                wallet_id=wallet_id,
                amount=sell_pct * cur_coin,
                currency=cur_name,
                commit=COMMIT
            )
            print('sold {:.5f} {}, cash out {:.2f} USD, at unit price={}'.format(
                float(order['amount']['amount']), cur_name,
                float(order['subtotal']['amount']), float(order['unit_price']['amount'])
            ))
        elif signal['action'] == NO_ACTION_SIGNAL:
            if mode == 'verbose':
                print('no action performed as simulation suggests.')

    # after
    v_c2, v_f2 = client.portfolio_value
    display_port_msg(v_c=v_c2, v_f=v_f2, before=False)

    # write to log file
    now = datetime.datetime.now()
    with open(log_file, 'a') as outfile:
        outfile.write('Finish job at time {}'.format(str(now)))

if __name__ == '__main__':

    # Run one-time
    #main()

    # Run as a cron-job
    print('\nI am awake...')

    schedule.every().day.at('10:30').do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)