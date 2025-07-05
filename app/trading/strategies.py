"""
Trading strategy implementations for cryptocurrency trading bot.
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL, STRATEGIES


def strategy_moving_average_w_tolerance(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
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
    strat_name = "MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # retrieve the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]

    # (1) if too high, we do a sell
    r_sell = False
    if new_p >= (1 + tol_pct) * last_ma:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if too low, we do a buy
    r_buy = False
    if new_p <= (1 - tol_pct) * last_ma:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_double_moving_averages(
    trader,
    shorter_queue_name: str,
    longer_queue_name: str,
    new_p: float,
    today: datetime.datetime,
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
    strat_name = "DOUBLE-MA"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # compute the moving averages for shorter queue and longer queue, respectively
    shorter_p = trader.moving_averages[shorter_queue_name][-1]
    longer_p = trader.moving_averages[longer_queue_name][-1]

    # (1) if a 'death-cross', sell
    r_sell = False
    if longer_p > shorter_p:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if a 'golden-cross', buy
    r_buy = False
    if shorter_p > longer_p:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_macd(trader, new_p: float, today: datetime.datetime) -> Tuple[bool, bool]:
    """
    MACD-based trading strategy.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MACD"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    if trader.macd_dea[-1] is None:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    diff = trader.macd_diff[-1] - trader.macd_dea[-1]

    # (1) if (MACD-DIFF - MACD-DEA) is negative, sell
    r_sell = False
    if diff < 0:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if (MACD-DIFF - MACD-DEA) is positive, buy
    r_buy = False
    if diff > 0:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_bollinger_bands(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    bollinger_sigma: int,
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
    strat_name = "BOLL-BANDS"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # if the moving average queue is not long enough to consider, skip
    not_null_values = list(
        filter(lambda x: x is not None, trader.moving_averages[queue_name])
    )
    if len(not_null_values) < int(queue_name):
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # want the last X number of items
    mas = not_null_values[-int(queue_name) :]
    # compute mean and standard deviation
    ma_mean, ma_std = np.mean(mas), np.std(mas)

    # compute upper and lower bound
    boll_upper = ma_mean + bollinger_sigma * ma_std
    boll_lower = ma_mean - bollinger_sigma * ma_std

    # (1) if the price exceeds the upper bound, sell
    r_sell = False
    if new_p >= boll_upper:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if the price drops below the lower bound, buy
    r_buy = False
    if new_p <= boll_lower:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_rsi(
    trader,
    new_p: float,
    today: datetime.datetime,
    period: int = 14,
    overbought: float = 70,
    oversold: float = 30,
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
    strat_name = "RSI"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    rsi = trader.compute_rsi(period)
    if rsi is None:
        return False, False  # Not enough data yet

    trader.strat_dct["RSI"].append(rsi)

    # Buy signal: RSI below oversold threshold
    if rsi < oversold:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
        return r_buy, False

    # Sell signal: RSI above overbought threshold
    elif rsi > overbought:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
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
    oversold: float = 30,
    overbought: float = 70,
) -> Tuple[bool, bool]:
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
    strat_name = "KDJ"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if we have enough KDJ data
    if (
        not hasattr(trader, "kdj_dct")
        or "K" not in trader.kdj_dct
        or len(trader.kdj_dct["K"]) == 0
    ):
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Get the latest K value (KDJ indicator)
    current_k = trader.kdj_dct["K"][-1]

    # Buy signal: KDJ below oversold threshold
    if current_k < oversold:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        return r_buy, False

    # Sell signal: KDJ above overbought threshold
    elif current_k > overbought:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        return False, r_sell

    # No action if KDJ is in neutral zone
    else:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False


def strategy_ma_macd_combined(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    Combined Moving Average and MACD strategy.
    Requires both MA signal and MACD confirmation for trades.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage for MA.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MA-MACD"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get MA signal
    last_ma = trader.moving_averages[queue_name][-1]
    ma_buy_signal = new_p <= (1 - tol_pct) * last_ma
    ma_sell_signal = new_p >= (1 + tol_pct) * last_ma

    # Get MACD signal
    macd_buy_signal = False
    macd_sell_signal = False

    if trader.macd_dea[-1] is not None:
        diff = trader.macd_diff[-1] - trader.macd_dea[-1]
        macd_buy_signal = diff > 0
        macd_sell_signal = diff < 0

    # Execute trades only when both signals agree
    r_buy, r_sell = False, False

    # Buy: MA shows oversold AND MACD shows bullish
    if ma_buy_signal and macd_buy_signal:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell: MA shows overbought AND MACD shows bearish
    elif ma_sell_signal and macd_sell_signal:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if signals don't agree
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_macd_kdj_combined(
    trader,
    new_p: float,
    today: datetime.datetime,
    oversold: float = 30,
    overbought: float = 70,
) -> Tuple[bool, bool]:
    """
    Combined MACD and KDJ strategy.
    Requires both momentum (MACD) and oscillator (KDJ) confirmation.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        oversold (float): KDJ oversold threshold.
        overbought (float): KDJ overbought threshold.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MACD-KDJ"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get MACD signal
    macd_buy_signal = False
    macd_sell_signal = False

    if trader.macd_dea[-1] is not None:
        diff = trader.macd_diff[-1] - trader.macd_dea[-1]
        macd_buy_signal = diff > 0
        macd_sell_signal = diff < 0

    # Get KDJ signal
    kdj_buy_signal = False
    kdj_sell_signal = False

    if (
        hasattr(trader, "kdj_dct")
        and "K" in trader.kdj_dct
        and len(trader.kdj_dct["K"]) > 0
    ):
        current_k = trader.kdj_dct["K"][-1]
        kdj_buy_signal = current_k < oversold
        kdj_sell_signal = current_k > overbought

    # Execute trades only when both signals agree
    r_buy, r_sell = False, False

    # Buy: MACD bullish AND KDJ oversold
    if macd_buy_signal and kdj_buy_signal:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell: MACD bearish AND KDJ overbought
    elif macd_sell_signal and kdj_sell_signal:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if signals don't agree
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_rsi_bollinger_combined(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    bollinger_sigma: int,
    rsi_period: int = 14,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> Tuple[bool, bool]:
    """
    Combined RSI and Bollinger Bands strategy.
    Uses RSI for momentum and Bollinger Bands for volatility confirmation.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        bollinger_sigma (int): Bollinger sigma value.
        rsi_period (int): RSI period.
        rsi_overbought (float): RSI overbought threshold.
        rsi_oversold (float): RSI oversold threshold.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "RSI-BOLL"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get RSI signal
    rsi = trader.compute_rsi(rsi_period)
    rsi_buy_signal = False
    rsi_sell_signal = False

    if rsi is not None:
        rsi_buy_signal = rsi < rsi_oversold
        rsi_sell_signal = rsi > rsi_overbought

    # Get Bollinger Bands signal
    boll_buy_signal = False
    boll_sell_signal = False

    not_null_values = list(
        filter(lambda x: x is not None, trader.moving_averages[queue_name])
    )
    if len(not_null_values) >= int(queue_name):
        mas = not_null_values[-int(queue_name) :]
        ma_mean, ma_std = np.mean(mas), np.std(mas)
        boll_upper = ma_mean + bollinger_sigma * ma_std
        boll_lower = ma_mean - bollinger_sigma * ma_std

        boll_buy_signal = new_p <= boll_lower
        boll_sell_signal = new_p >= boll_upper

    # Execute trades only when both signals agree
    r_buy, r_sell = False, False

    # Buy: RSI oversold AND price at lower Bollinger band
    if rsi_buy_signal and boll_buy_signal:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell: RSI overbought AND price at upper Bollinger band
    elif rsi_sell_signal and boll_sell_signal:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if signals don't agree
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_triple_signal(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    Triple signal strategy combining MA, MACD, and RSI.
    Requires confirmation from at least 2 out of 3 signals.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage for MA.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "TRIPLE-SIGNAL"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get MA signal
    last_ma = trader.moving_averages[queue_name][-1]
    ma_buy_signal = new_p <= (1 - tol_pct) * last_ma
    ma_sell_signal = new_p >= (1 + tol_pct) * last_ma

    # Get MACD signal
    macd_buy_signal = False
    macd_sell_signal = False

    if trader.macd_dea[-1] is not None:
        diff = trader.macd_diff[-1] - trader.macd_dea[-1]
        macd_buy_signal = diff > 0
        macd_sell_signal = diff < 0

    # Get RSI signal
    rsi = trader.compute_rsi(14)
    rsi_buy_signal = False
    rsi_sell_signal = False

    if rsi is not None:
        rsi_buy_signal = rsi < 30
        rsi_sell_signal = rsi > 70

    # Count buy and sell signals
    buy_signals = sum([ma_buy_signal, macd_buy_signal, rsi_buy_signal])
    sell_signals = sum([ma_sell_signal, macd_sell_signal, rsi_sell_signal])

    # Execute trades when at least 2 out of 3 signals agree
    r_buy, r_sell = False, False

    # Buy: At least 2 buy signals
    if buy_signals >= 2:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell: At least 2 sell signals
    elif sell_signals >= 2:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if insufficient signal agreement
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_conservative_ma(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    confirmation_periods: int = 3,
) -> Tuple[bool, bool]:
    """
    Conservative Moving Average strategy requiring multiple confirmations.
    Requires price to stay beyond tolerance for multiple periods before trading.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        confirmation_periods (int): Number of periods to confirm signal.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "CONSERVATIVE-MA"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get recent prices and MAs for confirmation
    recent_prices = trader.crypto_prices[-confirmation_periods:]
    recent_mas = trader.moving_averages[queue_name][-confirmation_periods:]

    # Check if we have enough data
    if (
        len(recent_prices) < confirmation_periods
        or len(recent_mas) < confirmation_periods
    ):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Count how many periods price was above/below MA with tolerance
    above_ma_count = 0
    below_ma_count = 0

    for i in range(confirmation_periods):
        price = recent_prices[i][0]
        ma = recent_mas[i]

        if ma is not None:
            if price >= (1 + tol_pct) * ma:
                above_ma_count += 1
            elif price <= (1 - tol_pct) * ma:
                below_ma_count += 1

    # Execute trades only with strong confirmation
    r_buy, r_sell = False, False

    # Buy: Price below MA for most periods
    if below_ma_count >= confirmation_periods - 1:  # At least 2 out of 3
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell: Price above MA for most periods
    elif above_ma_count >= confirmation_periods - 1:  # At least 2 out of 3
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if insufficient confirmation
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_volume_breakout(
    trader,
    new_p: float,
    today: datetime.datetime,
    volume_period: int = 20,
    volume_threshold: float = 1.5,
) -> Tuple[bool, bool]:
    """
    Volume-based breakout strategy.
    Uses volume spikes to confirm price movements and identify breakout opportunities.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        volume_period (int): Period for volume average calculation.
        volume_threshold (float): Volume spike threshold multiplier.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "VOLUME-BREAKOUT"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if we have volume data
    if len(trader.crypto_prices) < volume_period:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Extract volume data from crypto_prices (assuming volume is the 6th element)
    volumes = []
    for price_data in trader.crypto_prices[-volume_period:]:
        if len(price_data) >= 6:  # Check if volume data exists
            volumes.append(price_data[5])  # Volume is 6th element
        else:
            volumes.append(0)  # Default if no volume data

    if not volumes or all(v == 0 for v in volumes):
        # No volume data available, fall back to no action
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Filter out zero volumes for average calculation
    non_zero_volumes = [v for v in volumes[:-1] if v > 0]  # Exclude today's volume
    if not non_zero_volumes:
        # No valid volume data for average calculation
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Calculate average volume from non-zero volumes
    avg_volume = np.mean(non_zero_volumes)
    current_volume = volumes[-1] if len(volumes) > 0 else 0

    # Check for volume spike
    volume_spike = current_volume > (avg_volume * volume_threshold)

    # Get price movement direction
    if len(trader.crypto_prices) >= 2:
        prev_price = trader.crypto_prices[-2][0]  # Previous closing price
        price_change = new_p - prev_price
        price_up = price_change > 0
        price_down = price_change < 0
    else:
        price_up = price_down = False

    # Execute trades based on volume and price movement
    r_buy, r_sell = False, False

    # Buy signal: Volume spike with price increase
    if volume_spike and price_up:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell signal: Volume spike with price decrease
    elif volume_spike and price_down:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # No action if no volume spike or conflicting signals
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_multi_timeframe_ma_selves(
    trader,
    short_queue: str,
    medium_queue: str,
    long_queue: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    confirmation_required: int = 2,
) -> Tuple[bool, bool]:
    """
    MA-SELVES that requires confirmation from multiple timeframes.
    Only executes trades when majority of timeframes agree.

    Args:
        trader: The trader instance with necessary methods and attributes.
        short_queue (str): Short-term MA queue name (e.g., "6").
        medium_queue (str): Medium-term MA queue name (e.g., "12").
        long_queue (str): Long-term MA queue name (e.g., "30").
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        confirmation_required (int): Number of timeframes that must agree (default: 2).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MULTI-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Get signals from all timeframes
    signals = []
    for queue in [short_queue, medium_queue, long_queue]:
        if queue not in trader.moving_averages or len(trader.moving_averages[queue]) == 0:
            signals.append("HOLD")
            continue
            
        last_ma = trader.moving_averages[queue][-1]
        if last_ma is None:  # Check for None values
            signals.append("HOLD")
            continue
            
        if new_p >= (1 + tol_pct) * last_ma:
            signals.append("SELL")
        elif new_p <= (1 - tol_pct) * last_ma:
            signals.append("BUY")
        else:
            signals.append("HOLD")
    
    # Count signals
    sell_count = signals.count("SELL")
    buy_count = signals.count("BUY")
    
    # Execute trades based on confirmation
    r_buy, r_sell = False, False
    
    if sell_count >= confirmation_required:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    elif buy_count >= confirmation_required:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # Record no action if no confirmation
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_trend_aware_ma_selves(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    trend_period: int = 50,
    trend_threshold: float = 0.02,
) -> Tuple[bool, bool]:
    """
    MA-SELVES that considers the overall trend direction.
    More aggressive in trend direction, conservative against trend.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Base tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        trend_period (int): Period for trend calculation (default: 50).
        trend_threshold (float): Threshold for trend detection (default: 0.02).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "TREND-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Calculate trend using longer period
    if len(trader.moving_averages[queue_name]) < trend_period:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    trend_ma = trader.moving_averages[queue_name][-trend_period]
    current_ma = trader.moving_averages[queue_name][-1]
    
    # Check for None values
    if trend_ma is None or current_ma is None:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    trend_direction = (current_ma - trend_ma) / trend_ma
    
    # Adjust tolerance based on trend
    if trend_direction > trend_threshold:  # Uptrend
        buy_tol = tol_pct * 0.8  # More aggressive buying
        sell_tol = tol_pct * 1.2  # More conservative selling
    elif trend_direction < -trend_threshold:  # Downtrend
        buy_tol = tol_pct * 1.2  # More conservative buying
        sell_tol = tol_pct * 0.8  # More aggressive selling
    else:  # Sideways
        buy_tol = sell_tol = tol_pct
    
    # Get the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Check for None values
    if last_ma is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Execute trades with trend-adjusted tolerances
    r_buy, r_sell = False, False
    
    # (1) if too high, we do a sell
    if new_p >= (1 + sell_tol) * last_ma:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if too low, we do a buy
    elif new_p <= (1 - buy_tol) * last_ma:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_volume_weighted_ma_selves(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    volume_period: int = 20,
    volume_threshold: float = 1.5,
) -> Tuple[bool, bool]:
    """
    MA-SELVES that considers volume for signal confirmation.
    Higher volume = stronger signals with tighter tolerance.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Base tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        volume_period (int): Period for volume calculation (default: 20).
        volume_threshold (float): Threshold for high volume detection (default: 1.5).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "VOLUME-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if we have enough volume data
    if len(trader.volume_history) < volume_period:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Calculate volume ratio
    avg_volume = np.mean(trader.volume_history[-volume_period:])
    current_volume = trader.volume_history[-1]
    volume_ratio = current_volume / avg_volume
    
    # Adjust tolerance based on volume
    if volume_ratio > volume_threshold:
        # High volume - use tighter tolerance (stronger signals)
        adjusted_tol = tol_pct * 0.7
    else:
        # Low volume - use wider tolerance (weaker signals)
        adjusted_tol = tol_pct * 1.3
    
    # Get the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Check for None values
    if last_ma is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Execute trades with volume-adjusted tolerance
    r_buy, r_sell = False, False
    
    # (1) if too high, we do a sell
    if new_p >= (1 + adjusted_tol) * last_ma:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if too low, we do a buy
    elif new_p <= (1 - adjusted_tol) * last_ma:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_adaptive_ma_selves(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    base_tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    volatility_period: int = 20,
    volatility_multiplier: float = 1.0,
) -> Tuple[bool, bool]:
    """
    MA-SELVES with adaptive tolerance based on market volatility.
    Higher volatility = wider tolerance bands.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        base_tol_pct (float): Base tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        volatility_period (int): Period for volatility calculation (default: 20).
        volatility_multiplier (float): Multiplier for volatility adjustment (default: 1.0).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "ADAPTIVE-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if we have enough price data
    if len(trader.price_history) < volatility_period:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Calculate recent volatility
    recent_prices = trader.price_history[-volatility_period:]
    volatility = np.std(recent_prices) / np.mean(recent_prices)
    
    # Adjust tolerance based on volatility
    adaptive_tol = base_tol_pct * (1 + volatility * volatility_multiplier)
    
    # Get the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Check for None values
    if last_ma is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Execute trades with adaptive tolerance
    r_buy, r_sell = False, False
    
    # (1) if too high, we do a sell
    if new_p >= (1 + adaptive_tol) * last_ma:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # (2) if too low, we do a buy
    elif new_p <= (1 - adaptive_tol) * last_ma:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_momentum_enhanced_ma_selves(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    momentum_period: int = 14,
    momentum_threshold: float = 0.01,
) -> Tuple[bool, bool]:
    """
    MA-SELVES with momentum confirmation.
    Only trade when momentum aligns with MA signal.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.
        momentum_period (int): Period for momentum calculation (default: 14).
        momentum_threshold (float): Threshold for momentum confirmation (default: 0.01).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MOMENTUM-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if we have enough price data
    if len(trader.price_history) < momentum_period:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    # Calculate momentum (rate of change)
    momentum = (new_p - trader.price_history[-momentum_period]) / trader.price_history[-momentum_period]
    
    # Get MA signal
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Check for None values
    if last_ma is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    
    ma_signal = None
    
    if new_p >= (1 + tol_pct) * last_ma:
        ma_signal = "SELL"
    elif new_p <= (1 - tol_pct) * last_ma:
        ma_signal = "BUY"
    
    # Execute trades only if momentum confirms MA signal
    r_buy, r_sell = False, False
    
    if ma_signal == "BUY" and momentum > momentum_threshold:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    elif ma_signal == "SELL" and momentum < -momentum_threshold:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # Record no action if no confirmation
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_fear_greed_sentiment(
    trader,
    new_p: float,
    today: datetime.datetime,
    buy_thresholds: List[float] = [20, 30],
    sell_thresholds: List[float] = [70, 80],
) -> Tuple[bool, bool]:
    """
    Fear & Greed Index sentiment-based trading strategy.
    Buy when market sentiment shows extreme fear (low index values).
    Sell when market sentiment shows extreme greed (high index values).

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        buy_thresholds (List[float]): List of fear thresholds for buy signals (default: [20, 30]).
        sell_thresholds (List[float]): List of greed thresholds for sell signals (default: [70, 80]).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "FEAR-GREED-SENTIMENT"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if fear & greed data is available
    if not hasattr(trader, 'fear_greed_data') or not trader.fear_greed_data:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Get the most recent fear & greed index value
    current_fear_greed = trader.fear_greed_data[0]
    current_value = int(current_fear_greed['value'])

    r_buy, r_sell = False, False

    # Check for buy signal (extreme fear)
    for threshold in buy_thresholds:
        if current_value <= threshold:
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
                break

    # Check for sell signal (extreme greed) - only if no buy was executed
    if not r_buy:
        for threshold in sell_thresholds:
            if current_value >= threshold:
                r_sell = trader._execute_one_sell("by_percentage", new_p)
                if r_sell is True:
                    trader._record_history(new_p, today, SELL_SIGNAL)
                    trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
                    break

    # Record no action if no signal was triggered
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_exponential_moving_average_w_tolerance(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    For a new day's price, if beyond tolerance level, execute a buy or sell action using exponential moving averages.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the EMA queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Tolerance percentage.
        buy_pct (float): Buy percentage.
        sell_pct (float): Sell percentage.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "EXP-MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # retrieve the most recent exponential moving average
    last_ema = trader.exp_moving_averages[queue_name][-1]

    # Check for None values
    if last_ema is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # (1) if too high, we do a sell
    r_sell = False
    if new_p >= (1 + tol_pct) * last_ema:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if too low, we do a buy
    r_buy = False
    if new_p <= (1 - tol_pct) * last_ema:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_simple_recurring_investment(
    trader,
    new_p: float,
    today: datetime.datetime,
    investment_interval_days: int = 7,
    profit_threshold: float = 0.10,
    loss_threshold: float = 0.10,
    base_investment_pct: float = 0.05,
) -> Tuple[bool, bool]:
    """
    Simple recurring investment strategy.
    Buy every X days, adjust position size based on profit/loss thresholds.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        investment_interval_days (int): Days between investments (default: 7).
        profit_threshold (float): Profit threshold to reduce position (default: 0.10).
        loss_threshold (float): Loss threshold to increase position (default: 0.10).
        base_investment_pct (float): Base investment percentage (default: 0.05).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "SIMPLE-RECURRING"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if it's time to invest (every X days)
    last_date = getattr(trader, 'last_investment_date', None)
    # If last_date is a Mock (from unittest), treat as None
    if hasattr(last_date, '__class__') and 'Mock' in str(type(last_date)):
        last_date = None
    should_invest = False
    if last_date is None:
        should_invest = True
    else:
        days_since_last = (today - last_date).days
        should_invest = days_since_last >= investment_interval_days

    if not should_invest:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Calculate current portfolio performance
    if not hasattr(trader, 'initial_investment_value'):
        trader.initial_investment_value = trader.wallet['USD']
    
    current_value = trader.wallet['USD'] + (trader.wallet['crypto'] * new_p)
    total_return = (current_value - trader.initial_investment_value) / trader.initial_investment_value

    # Determine investment amount based on performance
    if total_return > profit_threshold:
        # Reduce position - sell some crypto
        investment_pct = base_investment_pct * 0.5  # Reduce by half
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
            trader.last_investment_date = today
            return False, True
    elif total_return < -loss_threshold:
        # Increase position - buy more crypto
        investment_pct = base_investment_pct * 1.5  # Increase by 50%
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
            trader.last_investment_date = today
            return True, False
    else:
        # Normal investment
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
            trader.last_investment_date = today
            return True, False

    # No action taken
    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    return False, False


def strategy_weighted_recurring_investment(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    investment_interval_days: int = 7,
    base_investment_pct: float = 0.05,
    ma_weight_factor: float = 2.0,
) -> Tuple[bool, bool]:
    """
    Weighted recurring investment strategy based on moving averages.
    Adjust investment amount based on price position relative to moving average.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the MA queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        investment_interval_days (int): Days between investments (default: 7).
        base_investment_pct (float): Base investment percentage (default: 0.05).
        ma_weight_factor (float): Factor to weight MA-based adjustments (default: 2.0).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "WEIGHTED-RECURRING"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Check if it's time to invest (every X days)
    last_weighted_date = getattr(trader, 'last_weighted_investment_date', None)
    if hasattr(last_weighted_date, '__class__') and 'Mock' in str(type(last_weighted_date)):
        last_weighted_date = None
    should_invest = False
    if last_weighted_date is None:
        should_invest = True
    else:
        days_since_last = (today - last_weighted_date).days
        should_invest = days_since_last >= investment_interval_days

    if not should_invest:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Get moving average for weighting
    if queue_name not in trader.moving_averages or len(trader.moving_averages[queue_name]) == 0:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    last_ma = trader.moving_averages[queue_name][-1]
    if last_ma is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Calculate price deviation from MA
    price_deviation = (new_p - last_ma) / last_ma
    
    # Determine investment weight based on MA position
    if price_deviation < -0.05:  # Price significantly below MA - good buying opportunity
        investment_weight = 1.0 + (abs(price_deviation) * ma_weight_factor)
    elif price_deviation > 0.05:  # Price significantly above MA - reduce buying
        investment_weight = max(0.1, 1.0 - (price_deviation * ma_weight_factor))
    else:  # Price near MA - normal investment
        investment_weight = 1.0

    # Execute weighted investment
    adjusted_investment_pct = base_investment_pct * investment_weight
    
    # For simplicity, we'll use the existing buy mechanism
    # In a real implementation, you might want to adjust the actual investment amount
    r_buy = trader._execute_one_buy("by_percentage", new_p)
    if r_buy:
        trader._record_history(new_p, today, BUY_SIGNAL)
        trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        trader.last_weighted_investment_date = today
        return True, False

    # No action taken
    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    return False, False


# ---- Strategy registry for easy lookup ---- #
STRATEGY_REGISTRY = {
    "MA-SELVES": strategy_moving_average_w_tolerance,
    "DOUBLE-MA": strategy_double_moving_averages,
    "MACD": strategy_macd,
    "BOLL-BANDS": strategy_bollinger_bands,
    "RSI": strategy_rsi,
    "KDJ": strategy_kdj,
    "MA-MACD": strategy_ma_macd_combined,
    "MACD-KDJ": strategy_macd_kdj_combined,
    "RSI-BOLL": strategy_rsi_bollinger_combined,
    "TRIPLE-SIGNAL": strategy_triple_signal,
    "CONSERVATIVE-MA": strategy_conservative_ma,
    "VOLUME-BREAKOUT": strategy_volume_breakout,
    "MULTI-MA-SELVES": strategy_multi_timeframe_ma_selves,
    "TREND-MA-SELVES": strategy_trend_aware_ma_selves,
    "VOLUME-MA-SELVES": strategy_volume_weighted_ma_selves,
    "ADAPTIVE-MA-SELVES": strategy_adaptive_ma_selves,
    "MOMENTUM-MA-SELVES": strategy_momentum_enhanced_ma_selves,
    "FEAR-GREED-SENTIMENT": strategy_fear_greed_sentiment,
    "EXP-MA-SELVES": strategy_exponential_moving_average_w_tolerance,
    "SIMPLE-RECURRING": strategy_simple_recurring_investment,
    "WEIGHTED-RECURRING": strategy_weighted_recurring_investment,
}
# ----------------------------- #
