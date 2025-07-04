"""
Trading strategy implementations for cryptocurrency trading bot.
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL, STRATEGIES


def strategy_moving_average_w_tolerance(
    trader,
    queue_name: str,
    new_p: float, 
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float
) -> Tuple[bool, bool]:
    """
    For a new day's price, if beyond tolerance level, execute a buy or sell action.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'MA-SELVES'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

    # retrieve the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # (1) if too high, we do a sell
    r_sell = False
    if new_p >= (1 + tol_pct) * last_ma:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if too low, we do a buy
    r_buy = False
    if new_p <= (1 - tol_pct) * last_ma:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if (r_buy is False and r_sell is False):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_double_moving_averages(
    trader,
    shorter_queue_name: str, 
    longer_queue_name: str,
    new_p: float, 
    today: datetime.datetime
) -> Tuple[bool, bool]:
    """
    For a new day's price, if the shorter MA is greater than the longer MA's, execute a buy; otherwise, execute a sell.

    Args:
        trader: The trader instance with necessary methods and attributes.
        shorter_queue_name (str): The name of the queue with shorter interval.
        longer_queue_name (str): The name of the queue with longer interval.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'DOUBLE-MA'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

    # compute the moving averages for shorter queue and longer queue, respectively
    shorter_p = trader.moving_averages[shorter_queue_name][-1]
    longer_p = trader.moving_averages[longer_queue_name][-1]

    # (1) if a 'death-cross', sell
    r_sell = False
    if longer_p > shorter_p:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if a 'golden-cross', buy
    r_buy = False
    if shorter_p > longer_p:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if (r_buy is False and r_sell is False):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_macd(
    trader,
    new_p: float, 
    today: datetime.datetime
) -> Tuple[bool, bool]:
    """
    MACD-based trading strategy.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'MACD'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

    if trader.macd_dea[-1] is None:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    diff = trader.macd_diff[-1] - trader.macd_dea[-1]
    
    # (1) if (MACD-DIFF - MACD-DEA) is negative, sell
    r_sell = False
    if diff < 0:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if (MACD-DIFF - MACD-DEA) is positive, buy
    r_buy = False
    if diff > 0:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if (r_buy is False and r_sell is False):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_bollinger_bands(
    trader,
    queue_name: str,
    new_p: float, 
    today: datetime.datetime,
    bollinger_sigma: int
) -> Tuple[bool, bool]:
    """
    Trading with Bollinger Band strategy.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        bollinger_sigma (int): Bollinger sigma value.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'BOLL-BANDS'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

    # if the moving average queue is not long enough to consider, skip
    not_null_values = list(filter(lambda x: x is not None, trader.moving_averages[queue_name]))
    if len(not_null_values) < int(queue_name):
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # want the last X number of items
    mas = not_null_values[-int(queue_name):]
    # compute mean and standard deviation
    ma_mean, ma_std = np.mean(mas), np.std(mas)

    # compute upper and lower bound
    boll_upper = ma_mean + bollinger_sigma * ma_std
    boll_lower = ma_mean - bollinger_sigma * ma_std

    # (1) if the price exceeds the upper bound, sell
    r_sell = False
    if new_p >= boll_upper:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if the price drops below the lower bound, buy
    r_buy = False
    if new_p <= boll_lower:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if (r_buy is False and r_sell is False):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_rsi(
    trader,
    new_p: float, 
    today: datetime.datetime, 
    period: int = 14, 
    overbought: float = 70, 
    oversold: float = 30
) -> Tuple[bool, bool]:
    """
    RSI-based trading strategy: buy if oversold, sell if overbought.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        period (int, optional): RSI period. Defaults to 14.
        overbought (float, optional): Overbought threshold. Defaults to 70.
        oversold (float, optional): Oversold threshold. Defaults to 30.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'RSI'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'

    rsi = trader.compute_rsi(period)
    if rsi is None:
        return False, False  # Not enough data yet
    
    trader.strat_dct['RSI'].append(rsi)
    
    # Buy signal: RSI below oversold threshold
    if rsi < oversold:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
        return r_buy, False
    
    # Sell signal: RSI above overbought threshold
    elif rsi > overbought:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
        return False, r_sell
    
    else:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        return False, False


def strategy_kdj(
    trader,
    new_p: float, 
    today: datetime.datetime, 
    oversold: float=30, 
    overbought: float=70) -> Tuple[bool, bool]:
    """
    KDJ-based trading strategy: buy if KDJ below oversold threshold, sell if above overbought threshold.
    
    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        oversold (float, optional): Oversold threshold. Defaults to 20.
        overbought (float, optional): Overbought threshold. Defaults to 80.
        
    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = 'KDJ'
    assert strat_name in STRATEGIES, 'Unknown trading strategy name!'
    
    # Check if we have enough KDJ data
    if not hasattr(trader, 'kdj_dct') or 'K' not in trader.kdj_dct or len(trader.kdj_dct['K']) == 0:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Get the latest K value (KDJ indicator)
    current_k = trader.kdj_dct['K'][-1]
    
    # Buy signal: KDJ below oversold threshold
    if current_k < oversold:
        r_buy = trader._execute_one_buy('by_percentage', new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        return r_buy, False
    
    # Sell signal: KDJ above overbought threshold
    elif current_k > overbought:
        r_sell = trader._execute_one_sell('by_percentage', new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        return False, r_sell
    
    # No action if KDJ is in neutral zone
    else:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False


# Strategy registry for easy lookup
STRATEGY_REGISTRY = {
    'MA-SELVES': strategy_moving_average_w_tolerance,
    'DOUBLE-MA': strategy_double_moving_averages,
    'MACD': strategy_macd,
    'BOLL-BANDS': strategy_bollinger_bands,
    'RSI': strategy_rsi,
    'KDJ': strategy_kdj,
} 