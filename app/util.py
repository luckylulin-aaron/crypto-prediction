# built-in packages
import functools
import datetime
import math
import time

from typing import List, Any

# third-party packages
import numpy as np
import pandas as pd


def timer(func):
    """Print the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        # do something...
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("--> executed={} using time={:.2f} seconds.".format(func.__name__, run_time))
        return value

    return wrapper_timer

def load_csv(csv_fn='../models/BTC_HISTORY.csv'):
    '''Load a csv file and return dates and prices lists.'''
    df = pd.read_csv(csv_fn)
    dates, prices = df['Date'].tolist(), df['Closing Price (USD)'].tolist()

    name, l = csv_fn.split('.')[0].split('_')[0], []
    for ind in range(len(dates)):
        l.append([prices[ind], dates[ind]])

    return name, l

def display_port_msg(v_c: float, v_f: float, before: bool=True):
    '''Display portfolio message in certain format.'''
    now = datetime.datetime.now()
    stage = 'before' if before else 'after'
    s = v_c + v_f

    print('\n{} transaction, by {}, crypto_value={:.2f}, fiat_value={:.2f}, amount to {:.2f}'.format(
        stage, now, v_c, v_f, s
    ))

def max_drawdown_helper(hist_l: List[float]):
    '''Compute max drawdown for a given portfolio value history.'''
    res = -math.inf
    value_cur, value_max_pre = hist_l[0], hist_l[0]
    for hist in hist_l[1:]:
        value_max_pre = max(value_max_pre, hist)
        value_cur = hist
        res = max(res, 1 - value_cur / value_max_pre)

    return np.round(res, 4)

def ema_helper(new_price: float, old_ema: float, num_of_days: int):
    '''Compute a new EMA. According to formula:
        - EMA = (today’s closing price * K) + (Previous EMA * (1 – K));
        - Smoothing Factor = 2/(N+1), N: number of days.
    '''
    smoothing_factor = 2.0 / (num_of_days + 1)
    ema = new_price * smoothing_factor + old_ema * (1 - smoothing_factor)

    return ema