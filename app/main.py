# built-in packages
import configparser
from curses import nocbreak
import datetime
import json
import os
import sys
import time

# third-party packages
import numpy as np
import pandas as pd
import schedule
# customized packages
from cbpro_client import CBProClient
from config import *
from trader_driver import TraderDriver
from util import display_port_msg, load_csv
from logger import get_logger

logger = get_logger(__name__)

# Read API credentials from screte.ini
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'screte.ini'))

CB_API_KEY = config['CONFIG']['COINBASE_API_KEY'].strip('"')
CB_API_SECRET = config['CONFIG']['COINBASE_API_SECRET'].strip('"')


def main():
    """
    Run simulation and make trades.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: If there is an error during portfolio value retrieval or trading simulation.
    """

    logger.info(f'COMMIT is set to {COMMIT}')
    log_file = './log.txt'

    # cbpro client to interact with coinbase
    client = CBProClient(
        key=CB_API_KEY,
        secret=CB_API_SECRET
    )

    fake_cur_coin, fake_cash = 5, 3000
    mode = 'normal'

    # before
    try:
        logger.info('Getting portfolio value...')
        v_c1, v_s1 = client.portfolio_value
        logger.info(f'Portfolio value before, crypto={v_c1}, stablecoin={v_s1}')
    except Exception as e:
        logger.error(f'Error getting portfolio value: {e}')
        return

    return
    display_port_msg(v_c=v_c1, v_s=v_s1, before=True)

    for index,cur_name in enumerate(CURS):
        logger.info(f'[{index+1}] processing for currency={cur_name}...')
        cur_rate = client.get_cur_rate(name=cur_name + '-USD')
        data_stream = client.get_historic_data(name=cur_name + '-USD')

        # cut-off, only want the last X days of data
        data_stream = data_stream[-TIMESPAN:]
        logger.info(f'only want the latest {TIMESPAN} days of data!')

        # initial cash amount
        _, cash = client.portfolio_value
        # initial coin at hand
        wallet = client.get_wallets(cur_names=[cur_name])
        logger.info(f'wallet: {wallet}')
        time.sleep(30)

        cur_coin, wallet_id = None, None
        for item in wallet:
            if item['currency'] == cur_name and item['type'] == 'ACCOUNT_TYPE_CRYPTO':
                cur_coin, wallet_id = float(item['available_balance']['value']), item['uuid']

        assert cur_coin is not None and wallet_id is not None, f'cannot find relevant wallet for {cur_name}!'

        # run simulation
        t_driver = TraderDriver(
            name=cur_name,
            init_amount=fake_cash,
            overall_stats=STRATEGIES,
            cur_coin=fake_cur_coin,
            tol_pcts=TOL_PCTS,
            ma_lengths=MA_LENGTHS,
            ema_lengths=EMA_LENGTHS,
            bollinger_mas=BOLLINGER_MAS,
            bollinger_tols=BOLLINGER_TOLS,
            buy_pcts=BUY_PCTS,
            sell_pcts=SELL_PCTS,
            buy_stas=list(BUY_STAS) if isinstance(BUY_STAS, (list, tuple)) else [BUY_STAS],
            sell_stas=list(SELL_STAS) if isinstance(SELL_STAS, (list, tuple)) else [SELL_STAS],
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

        logger.info(f'examine best trader performance: {info}')
        logger.info(f'maximum drawdown / 最大回撤: {best_t.max_drawdown}')
        logger.info(f'number of transaction / 总的交易数: {best_t.num_transaction}')
        logger.info(f'high-level trading strategy / 高级交易策略: {best_t.high_strategy}')
        logger.info(f'number of buy action / 买入次数: {best_t.num_buy_action}')
        logger.info(f'number of sell action / 卖出次数: {best_t.num_sell_action}')

        logger.info(f'for crypto={best_t.crypto_name}, today\'s signal={signal}')

        # if too less, we leave
        if cash <= EP_CASH and signal == BUY_SIGNAL:
            logger.warning('too less cash, cannot execute a buy, discard further actions.')
            continue
        elif (cur_coin <= EP_COIN or cur_coin * new_p <= EP_CASH) and signal == SELL_SIGNAL:
            logger.warning('too less crypto, cannot execute a sell, discard further actions.')
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
            logger.info('bought {:.5f} {}, used {:.2f} USD, at unit price={}'.format(
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
            logger.info('sold {:.5f} {}, cash out {:.2f} USD, at unit price={}'.format(
                float(order['amount']['amount']), cur_name,
                float(order['subtotal']['amount']), float(order['unit_price']['amount'])
            ))
        elif signal['action'] == NO_ACTION_SIGNAL:
            if mode == 'verbose':
                logger.info('no action performed as simulation suggests.')

    # after
    v_c2, v_s2 = client.portfolio_value
    display_port_msg(v_c=v_c2, v_s=v_s2, before=False)

    # write to log file
    now = datetime.datetime.now()
    with open(log_file, 'a') as outfile:
        outfile.write('Finish job at time {}\n'.format(str(now)))

if __name__ == '__main__':

    # Run one-time
    main()

    # Run as a cron-job
    '''
    schedule.every().day.at('22:53').do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
        sys.stdout.flush()
    '''