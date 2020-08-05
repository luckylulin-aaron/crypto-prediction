import functools
import pandas as pd
import time

def timer(func):
    """Print the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        # do something...
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("--> executed={} using time={:.2f} seconds.\n".format(func.__name__, run_time))
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