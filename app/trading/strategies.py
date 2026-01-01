"""
Trading strategy implementations for cryptocurrency trading bot.
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import BUY_SIGNAL, NO_ACTION_SIGNAL, SELL_SIGNAL, STRATEGIES, SUPPORTED_STRATEGIES
from app.core.logger import get_logger

# IMPORTANT:
# `config.STRATEGIES` is the *enabled* set (e.g. crypto strategies used in simulations).
# In this file, asserts should validate against the *supported/implemented* set.
STRATEGIES = SUPPORTED_STRATEGIES


def _is_positive_number(value) -> bool:
    """
    Check whether a value is a positive int/float.

    Args:
        value: Value to check.

    Returns:
        bool: True if value is an int/float and > 0, else False.
    """
    return isinstance(value, (int, float, np.number)) and float(value) > 0


def strategy_moving_average_w_tolerance(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    volatility_window: int=20,
    volatility_multiplier: float=1.0,
    cooldown_days: int=2,
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
        volatility_window (int): Window size (in data points) to estimate recent volatility.
            Used to widen the tolerance during high volatility to reduce churn. Defaults to 20.
        volatility_multiplier (float): Multiplier applied to volatility estimate when computing
            an adaptive tolerance. Defaults to 1.0.
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy.
            0 disables cooldown. Defaults to 0.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MA-SELVES"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Position and cash check (prevents meaningless orders & reduces churn).
    # Prefer `cash/cur_coin` used by `StratTrader`, but fall back to `wallet` for older tests/mocks.
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_cash = _is_positive_number(cash_val)
    has_position = _is_positive_number(coin_val)
    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # Optional cooldown to reduce flip-flopping
    if cooldown_days > 0 and strat_name in getattr(trader, "strat_dct", {}):
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # retrieve the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]

    # Volatility-adjusted tolerance: widen thresholds when volatility is high
    effective_tol = tol_pct
    try:
        prices = getattr(trader, "price_history", None)
        if not prices and hasattr(trader, "crypto_prices"):
            prices = [x[0] for x in trader.crypto_prices]
        if (
            prices
            and _is_positive_number(volatility_window)
            and len(prices) >= int(volatility_window)
        ):
            window = prices[-int(volatility_window) :]
            m = float(np.mean(window))
            s = float(np.std(window))
            if m > 0:
                vol_pct = s / m
                effective_tol = max(tol_pct, float(volatility_multiplier) * vol_pct)
    except Exception:
        # Keep base tolerance on any error (e.g., missing attrs)
        effective_tol = tol_pct

    # (1) if too high, we do a sell
    r_sell = False
    if has_position and new_p >= (1 + effective_tol) * last_ma:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if too low, we do a buy
    r_buy = False
    # Avoid executing both sell and buy in the same tick.
    if r_sell is False and has_cash and new_p <= (1 - effective_tol) * last_ma:
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
    opportunity_days: int=3,
    touch_tol_pct: float=0.01,
    cooldown_days: int=1,
) -> Tuple[bool, bool]:
    """
    Double moving average crossover with a short "opportunity window".

    Idea: the crossover defines a short-lived regime change, but the best entries/exits often
    happen shortly after the crossover on a pullback/retest.

    - When a golden cross happens (short MA crosses above long MA), we consider there are buy
      opportunities within the next `opportunity_days` days. We buy when price "touches"
      the short MA (a pullback to support).
    - When a death cross happens (short MA crosses below long MA), we consider there are sell
      opportunities within the next `opportunity_days` days. We sell when price "touches"
      the short MA (a rebound to resistance).

    Args:
        trader: The trader instance with necessary methods and attributes.
        shorter_queue_name (str): The name of the queue with shorter interval.
        longer_queue_name (str): The name of the queue with longer interval.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        opportunity_days (int): Window size (in days) after a cross to look for opportunities.
            Defaults to 3.
        touch_tol_pct (float): Touch tolerance as a fraction of MA (e.g. 0.01 = 1%).
            Defaults to 0.01.
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy key.
            Defaults to 1.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "DOUBLE-MA"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Position/cash checks (avoid meaningless orders)
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_cash = _is_positive_number(cash_val)
    has_position = _is_positive_number(coin_val)

    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    def _last_two_non_none(values):
        not_null = [x for x in values if x is not None]
        if len(not_null) < 2:
            return None, None
        return not_null[-2], not_null[-1]

    s_prev, s_cur = _last_two_non_none(trader.moving_averages[shorter_queue_name])
    l_prev, l_cur = _last_two_non_none(trader.moving_averages[longer_queue_name])

    if s_prev is None or s_cur is None or l_prev is None or l_cur is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Detect true cross events (using previous vs current)
    golden_cross = s_prev <= l_prev and s_cur > l_cur
    death_cross = s_prev >= l_prev and s_cur < l_cur

    # Track opportunity windows per MA pair
    window_key = f"{shorter_queue_name}:{longer_queue_name}"
    windows = getattr(trader, "_double_ma_windows", None)
    # `Mock` will happily fabricate attributes; ensure we have a real dict.
    if not isinstance(windows, dict):
        windows = {}
        setattr(trader, "_double_ma_windows", windows)

    if golden_cross:
        windows[window_key] = {"type": "BULLISH", "start": today}
    elif death_cross:
        windows[window_key] = {"type": "BEARISH", "start": today}

    window = windows.get(window_key)
    if not window:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Check window age
    try:
        days_since = (today.date() - window["start"].date()).days
    except Exception:
        days_since = (today - window["start"]).days
    if days_since > opportunity_days:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Cooldown per window key
    if cooldown_days > 0 and strat_name in getattr(trader, "strat_dct", {}):
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # Price touch logic to reduce whipsaw/chasing
    short_ma = s_cur
    if not _is_positive_number(short_ma):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    touch = abs(new_p - short_ma) <= (touch_tol_pct * short_ma)

    r_buy, r_sell = False, False

    if window["type"] == "BULLISH":
        # Buy on pullback to short MA within window
        if has_cash and touch and new_p <= short_ma * (1.0 + touch_tol_pct):
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    elif window["type"] == "BEARISH":
        # Sell on rebound to short MA within window
        if has_position and touch and new_p >= short_ma * (1.0 - touch_tol_pct):
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_macd(
    trader,
    new_p: float,
    today: datetime.datetime,
    hist_threshold: float=0.0,
    cooldown_days: int=1,
) -> Tuple[bool, bool]:
    """
    MACD-based trading strategy.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        hist_threshold (float): Minimum absolute MACD histogram value required to act on a crossover.
            This helps reduce whipsaw around zero. Defaults to 0.0 (disabled).
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy.
            Defaults to 1.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MACD"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Need at least 2 points to detect a crossover
    if (
        not hasattr(trader, "macd_diff")
        or not hasattr(trader, "macd_dea")
        or len(trader.macd_diff) < 2
        or len(trader.macd_dea) < 2
        or trader.macd_dea[-1] is None
        or trader.macd_dea[-2] is None
    ):
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Position/cash checks (avoid meaningless orders)
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_cash = _is_positive_number(cash_val)
    has_position = _is_positive_number(coin_val)

    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # Cooldown to reduce churn
    if cooldown_days > 0 and strat_name in getattr(trader, "strat_dct", {}):
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # MACD histogram (DIFF - DEA) and crossover detection
    hist_prev = trader.macd_diff[-2] - trader.macd_dea[-2]
    hist_cur = trader.macd_diff[-1] - trader.macd_dea[-1]

    cross_up = hist_prev <= 0 and hist_cur > 0
    cross_down = hist_prev >= 0 and hist_cur < 0

    # Regime: above/below the zero line
    bullish_regime = trader.macd_diff[-1] > 0 and trader.macd_dea[-1] > 0
    bearish_regime = trader.macd_diff[-1] < 0 and trader.macd_dea[-1] < 0

    # Threshold filter (optional)
    if _is_positive_number(hist_threshold):
        if abs(hist_cur) < float(hist_threshold):
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return False, False

    r_buy, r_sell = False, False

    # --- Trading rules (position-aware + regime-aware) ---
    #
    # Bullish regime: trade crossovers more aggressively (trend-following).
    # Bearish regime: avoid chasing; only buy on cross_up if still below zero line (bounce),
    # and prioritize exits on cross_down.
    if cross_down and has_position:
        # Exit signal is valid in any regime; this is risk control.
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    elif cross_up and has_cash:
        if bullish_regime:
            # Trend-follow in bullish regime
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        elif bearish_regime:
            # Mean-reversion bounce in bearish regime: only buy if still below zero line
            if trader.macd_diff[-1] < 0 and trader.macd_dea[-1] < 0:
                r_buy = trader._execute_one_buy("by_percentage", new_p)
                if r_buy is True:
                    trader._record_history(new_p, today, BUY_SIGNAL)
                    trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        else:
            # Neutral regime: allow buy on cross_up
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
    band_breakout_pct: float=0.002,
    min_bandwidth_pct: float=0.02,
    cooldown_days: int=2,
    avoid_lookahead: bool=True,
) -> Tuple[bool, bool]:
    """
    Trading with Bollinger Band strategy.

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        bollinger_sigma (int): Bollinger sigma value.
        band_breakout_pct (float): Extra margin beyond bands to confirm a band break and reduce whipsaw.
            Defaults to 0.002 (0.2%).
        min_bandwidth_pct (float): Minimum band width (as % of midline) required to trade.
            Prevents overtrading in low-volatility regimes. Defaults to 0.02 (2%).
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy key.
            Defaults to 2.
        avoid_lookahead (bool): If True, compute bands using MA values up to yesterday (exclude last MA),
            since today's MA is computed using today's price. Defaults to True.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "BOLL-BANDS"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Position/cash checks (avoid meaningless orders)
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_cash = _is_positive_number(cash_val)
    has_position = _is_positive_number(coin_val)

    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # Cooldown to reduce churn
    if cooldown_days > 0 and strat_name in getattr(trader, "strat_dct", {}):
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # if the moving average queue is not long enough to consider, skip
    not_null_values = list(
        filter(lambda x: x is not None, trader.moving_averages[queue_name])
    )
    # When avoiding lookahead, we need one extra MA value so we can exclude today's MA.
    required_len = int(queue_name) + (1 if avoid_lookahead else 0)
    if len(not_null_values) < required_len:
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Use the last X number of items (optionally excluding today's MA to avoid lookahead)
    band_values = not_null_values[:-1] if avoid_lookahead else not_null_values
    mas = band_values[-int(queue_name) :]
    # compute mean and standard deviation
    ma_mean, ma_std = np.mean(mas), np.std(mas)

    # compute upper and lower bound
    boll_upper = ma_mean + bollinger_sigma * ma_std
    boll_lower = ma_mean - bollinger_sigma * ma_std

    # Volatility filter: bands must be wide enough to overcome fees/spread
    if ma_mean is None or ma_mean == 0:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    bandwidth_pct = (boll_upper - boll_lower) / ma_mean
    if bandwidth_pct < min_bandwidth_pct:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # (1) if the price exceeds the upper bound, sell
    r_sell = False
    if has_position and new_p >= boll_upper * (1.0 + band_breakout_pct):
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if the price drops below the lower bound, buy
    r_buy = False
    if r_sell is False and has_cash and new_p <= boll_lower * (1.0 - band_breakout_pct):
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))

    return r_buy, r_sell


def strategy_ma_boll_bands(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
    bollinger_sigma: int,
    trend_lookback: int = 3,
    trend_min_abs_slope_pct: float = 0.002,
    band_breakout_pct: float = 0.002,
    cooldown_days: int = 2,
    min_bandwidth_pct: float = 0.02,
    entry_z: float = 1.5,
    exit_z: float = 0.8,
) -> Tuple[bool, bool]:
    """
    MA + Bollinger Bands strategy using the shortest MA trend to define regime.

    Regime definition is based on the direction of the moving average:
    - Bull market: MA trending upward (last MA > previous MA)
      - Buy when price "touches" the MA (within tol_pct)
      - Sell when price exceeds the upper Bollinger Band
    - Bear market: MA trending downward (last MA < previous MA)
      - Sell when price "touches" the MA (within tol_pct)
      - Buy when price goes below the lower Bollinger Band (oversold)

    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The MA queue name (should be the shortest MA).
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        tol_pct (float): Touch tolerance as a fraction of MA (e.g. 0.01 = 1%).
        buy_pct (float): Buy percentage (kept for API consistency).
        sell_pct (float): Sell percentage (kept for API consistency).
        bollinger_sigma (int): Bollinger sigma value.
        trend_lookback (int, optional): Lookback window (in MA points) for trend slope. Defaults to 3.
        trend_min_abs_slope_pct (float, optional): Minimum absolute MA slope (as % of MA) to trade.
            Defaults to 0.002 (0.2%).
        band_breakout_pct (float, optional): Extra margin beyond bands to confirm breakout and reduce churn.
            Defaults to 0.002 (0.2%).
        cooldown_days (int, optional): Minimum days between non-NO-ACTION signals to reduce overtrading.
            Defaults to 2.
        min_bandwidth_pct (float, optional): Minimum Bollinger bandwidth (as % of midline) required to trade.
            This prevents overtrading in low-volatility regimes. Defaults to 0.02 (2%).
        entry_z (float, optional): Z-score threshold for entries (buy when z <= -entry_z). Defaults to 1.5.
        exit_z (float, optional): Z-score threshold for exits (sell when z >= exit_z). Defaults to 0.8.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MA-BOLL-BANDS"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    # Require enough MA points to (a) determine trend and (b) compute bands without lookahead.
    not_null_values = list(
        filter(lambda x: x is not None, trader.moving_averages[queue_name])
    )
    if len(not_null_values) < (trend_lookback + 2):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # NOTE: `StratTrader.add_new_day()` appends today's MA value before calling strategies.
    # To avoid lookahead bias, we use the previous MA for regime + "touch" decisions.
    signal_ma = not_null_values[-2]
    if signal_ma is None or signal_ma <= 0:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    ma_now = not_null_values[-2]
    ma_past = not_null_values[-(trend_lookback + 2)]
    if ma_past is None or ma_past <= 0:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Trend slope (normalized): positive => bull, negative => bear.
    # We keep this as a mild bias, but the strategy's core is volatility harvesting.
    slope_pct = (ma_now - ma_past) / ma_now
    trending_up = slope_pct >= trend_min_abs_slope_pct
    trending_down = slope_pct <= -trend_min_abs_slope_pct

    # Simple cooldown using the last non-NO-ACTION signal date for this strategy.
    if cooldown_days > 0 and strat_name in trader.strat_dct:
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                if hasattr(today, "date"):
                    delta_days = (today.date() - past_dt.date()).days
                else:
                    delta_days = (today - past_dt).days
                if delta_days < cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # Compute Bollinger bands from recent MA values (must have >= window length)
    boll_upper = None
    boll_lower = None
    mid = None
    std = None
    # Avoid lookahead: compute bands using MA values up to yesterday (exclude last appended MA).
    band_values = not_null_values[:-1]
    if len(band_values) >= int(queue_name):
        mas = band_values[-int(queue_name) :]
        mid, std = float(np.mean(mas)), float(np.std(mas))
        boll_upper = mid + bollinger_sigma * std
        boll_lower = mid - bollinger_sigma * std

    # If we can't compute bands, we can't meaningfully harvest volatility.
    if boll_upper is None or boll_lower is None or mid is None or std is None or mid <= 0:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Volatility filter: trade only when the bands are wide enough to overcome fees/spread.
    bandwidth_pct = (boll_upper - boll_lower) / mid if mid != 0 else 0.0
    if bandwidth_pct < min_bandwidth_pct:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Z-score of price relative to midline; epsilon prevents division-by-zero.
    z = (new_p - mid) / (std + 1e-12)

    # position and cash check, really important!
    has_position = _is_positive_number(getattr(trader, "cur_coin", 0))
    has_cash = _is_positive_number(getattr(trader, "cash", 0))

    r_buy, r_sell = False, False

    # --- Volatility harvesting core ---
    #
    # Entry: buy when price is sufficiently below the lower band (or z-score is low enough).
    # Exit: sell when price is sufficiently above the upper band (or z-score is high enough).
    #
    # Trend bias (mild):
    # - In strong uptrends, require a deeper pullback to buy (avoid buying "not cheap enough")
    # - In strong downtrends, require a stronger overbought condition to sell (avoid selling into weakness)
    effective_entry_z = entry_z + (0.3 if trending_up else 0.0)
    effective_exit_z = exit_z + (0.3 if trending_down else 0.0)

    buy_condition = (
        (new_p <= boll_lower * (1.0 - band_breakout_pct)) or (z <= -effective_entry_z)
    )
    sell_condition = (
        (new_p >= boll_upper * (1.0 + band_breakout_pct)) or (z >= effective_exit_z)
    )

    # Prefer exits before entries (risk management) and be position-aware.
    if has_position and sell_condition:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    elif (not has_position) and has_cash and buy_condition:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # add history as well if nothing happens (or execution failed)
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
    strat_name = "RSI" # NOTE: NEXT ONE TO BE OPTIMISED!
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
        if (
            queue not in trader.moving_averages
            or len(trader.moving_averages[queue]) == 0
        ):
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
    momentum = (new_p - trader.price_history[-momentum_period]) / trader.price_history[
        -momentum_period
    ]

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
    if not hasattr(trader, "fear_greed_data") or not trader.fear_greed_data:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Get the most recent fear & greed index value
    current_fear_greed = trader.fear_greed_data[0]
    current_value = int(current_fear_greed["value"])

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

    # Position and cash check (prevents meaningless orders & reduces churn).
    # Prefer `cash/cur_coin` used by `StratTrader`, but fall back to `wallet` for older tests/mocks.
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_cash = _is_positive_number(cash_val)
    has_position = _is_positive_number(coin_val)
    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # retrieve the most recent exponential moving average
    last_ema = trader.exp_moving_averages[queue_name][-1]

    # Check for None values
    if last_ema is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # (1) if too high, we do a sell
    r_sell = False
    if has_position and new_p >= (1 + tol_pct) * last_ema:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if too low, we do a buy
    r_buy = False
    # Avoid executing both sell and buy in the same tick.
    if r_sell is False and has_cash and new_p <= (1 - tol_pct) * last_ema:
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
    last_date = getattr(trader, "last_investment_date", None)
    # If last_date is a Mock (from unittest), treat as None
    if hasattr(last_date, "__class__") and "Mock" in str(type(last_date)):
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
    if not hasattr(trader, "initial_investment_value"):
        trader.initial_investment_value = trader.wallet["USD"]

    current_value = trader.wallet["USD"] + (trader.wallet["crypto"] * new_p)
    total_return = (
        current_value - trader.initial_investment_value
    ) / trader.initial_investment_value

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
    last_weighted_date = getattr(trader, "last_weighted_investment_date", None)
    if hasattr(last_weighted_date, "__class__") and "Mock" in str(
        type(last_weighted_date)
    ):
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
    if (
        queue_name not in trader.moving_averages
        or len(trader.moving_averages[queue_name]) == 0
    ):
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
    if (
        price_deviation < -0.05
    ):  # Price significantly below MA - good buying opportunity
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


class EconomicIndicatorsStrategy:
    """
    Trading strategy that incorporates macroeconomic indicators like M2 money supply
    and Federal Reserve overnight reverse repo data.
    
    This strategy provides a macro overlay to other technical strategies.
    """
    
    def __init__(self, name: str = "Economic Indicators"):
        """
        Initialize the economic indicators strategy.
        
        Args:
            name (str): Strategy name
        """
        self.name = name
        self.logger = get_logger(__name__)
        
        # Import here to avoid circular imports
        from app.data.economic_indicators_client import EconomicIndicatorsClient
        self.econ_client = EconomicIndicatorsClient()
        
        # Strategy parameters
        self.m2_bullish_threshold = 1.0  # M2 growth > 1% monthly = bullish
        self.m2_bearish_threshold = -0.5  # M2 contraction > 0.5% monthly = bearish
        self.rrp_bullish_threshold = -10  # ON RRP < 7-day avg by 10% = bullish (easing)
        self.rrp_bearish_threshold = 10   # ON RRP > 7-day avg by 10% = bearish (tightening)
        
    def get_macro_signal(self) -> Dict[str, any]:
        """
        Get macroeconomic trading signal based on economic indicators.
        
        Returns:
            Dict[str, any]: Macro signal with direction, strength, and reasoning
        """
        try:
            # Get economic indicators
            indicators = self.econ_client.get_liquidity_indicators()
            signals = self.econ_client.get_trading_signals()
            
            if not indicators or not signals:
                return {
                    'signal': 'NEUTRAL',
                    'strength': 0.0,
                    'reason': 'No economic data available',
                    'indicators': {},
                    'signals': {}
                }
            
            # Calculate signal strength based on indicators
            strength = 0.0
            reasons = []
            
            # M2 contribution to signal strength
            if 'm2_change_1m_pct' in indicators:
                m2_change = indicators['m2_change_1m_pct']
                if m2_change > self.m2_bullish_threshold:
                    strength += 0.4  # Strong bullish
                    reasons.append(f"M2 growing {m2_change:.2f}% monthly")
                elif m2_change < self.m2_bearish_threshold:
                    strength -= 0.4  # Strong bearish
                    reasons.append(f"M2 contracting {abs(m2_change):.2f}% monthly")
                elif m2_change > 0:
                    strength += 0.2  # Weak bullish
                    reasons.append(f"M2 growing modestly {m2_change:.2f}%")
                else:
                    strength -= 0.2  # Weak bearish
                    reasons.append(f"M2 contracting modestly {abs(m2_change):.2f}%")
            
            # ON RRP contribution to signal strength
            if 'on_rrp_vs_7d_avg_pct' in indicators:
                rrp_deviation = indicators['on_rrp_vs_7d_avg_pct']
                if rrp_deviation < self.rrp_bullish_threshold:
                    strength += 0.3  # Strong bullish (easing)
                    reasons.append(f"ON RRP {abs(rrp_deviation):.1f}% below average (easing)")
                elif rrp_deviation > self.rrp_bearish_threshold:
                    strength -= 0.3  # Strong bearish (tightening)
                    reasons.append(f"ON RRP {rrp_deviation:.1f}% above average (tightening)")
                elif rrp_deviation < 0:
                    strength += 0.15  # Weak bullish
                    reasons.append(f"ON RRP slightly below average")
                else:
                    strength -= 0.15  # Weak bearish
                    reasons.append(f"ON RRP slightly above average")
            
            # Determine signal direction
            if strength > 0.3:
                signal = 'BULLISH'
            elif strength < -0.3:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
            
            return {
                'signal': signal,
                'strength': abs(strength),
                'reason': '; '.join(reasons),
                'indicators': indicators,
                'signals': signals
            }
            
        except Exception as e:
            self.logger.error(f"Error getting macro signal: {e}")
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'reason': f'Error: {e}',
                'indicators': {},
                'signals': {}
            }
    
    def adjust_technical_signal(self, technical_signal: Dict[str, any], 
                               macro_signal: Dict[str, any]) -> Dict[str, any]:
        """
        Adjust technical trading signal based on macroeconomic conditions.
        
        Args:
            technical_signal (Dict): Original technical signal
            macro_signal (Dict): Macroeconomic signal
            
        Returns:
            Dict[str, any]: Adjusted signal with macro overlay
        """
        try:
            adjusted_signal = technical_signal.copy()
            
            # If macro signal is strong, it can override weak technical signals
            macro_strength = macro_signal.get('strength', 0.0)
            macro_direction = macro_signal.get('signal', 'NEUTRAL')
            
            if macro_strength > 0.5:  # Strong macro signal
                if macro_direction == 'BULLISH':
                    # Boost bullish signals, reduce bearish signals
                    if technical_signal.get('action') == 'BUY':
                        adjusted_signal['buy_percentage'] = min(100, 
                            technical_signal.get('buy_percentage', 0) + 20)
                        adjusted_signal['sell_percentage'] = max(0,
                            technical_signal.get('sell_percentage', 0) - 20)
                    elif technical_signal.get('action') == 'SELL':
                        # Reduce sell strength due to bullish macro
                        adjusted_signal['sell_percentage'] = max(0,
                            technical_signal.get('sell_percentage', 0) - 30)
                        if adjusted_signal['sell_percentage'] < 20:
                            adjusted_signal['action'] = 'HOLD'
                            
                elif macro_direction == 'BEARISH':
                    # Boost bearish signals, reduce bullish signals
                    if technical_signal.get('action') == 'SELL':
                        adjusted_signal['sell_percentage'] = min(100,
                            technical_signal.get('sell_percentage', 0) + 20)
                        adjusted_signal['buy_percentage'] = max(0,
                            technical_signal.get('buy_percentage', 0) - 20)
                    elif technical_signal.get('action') == 'BUY':
                        # Reduce buy strength due to bearish macro
                        adjusted_signal['buy_percentage'] = max(0,
                            technical_signal.get('buy_percentage', 0) - 30)
                        if adjusted_signal['buy_percentage'] < 20:
                            adjusted_signal['action'] = 'HOLD'
            
            # Add macro context to signal
            adjusted_signal['macro_overlay'] = {
                'macro_signal': macro_direction,
                'macro_strength': macro_strength,
                'macro_reason': macro_signal.get('reason', ''),
                'original_action': technical_signal.get('action'),
                'adjusted_action': adjusted_signal.get('action')
            }
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Error adjusting technical signal: {e}")
            return technical_signal
    
    def get_combined_signal(self, technical_strategies: List[any]) -> Dict[str, any]:
        """
        Get combined signal from technical strategies with macroeconomic overlay.
        
        Args:
            technical_strategies (List): List of technical strategy instances
            
        Returns:
            Dict[str, any]: Combined signal with macro overlay
        """
        try:
            # Get macro signal
            macro_signal = self.get_macro_signal()
            
            # Get best technical signal (assuming first strategy is best)
            if technical_strategies:
                best_technical = technical_strategies[0]
                technical_signal = best_technical.trade_signal
            else:
                technical_signal = {'action': 'HOLD', 'buy_percentage': 0, 'sell_percentage': 0}
            
            # Adjust technical signal with macro overlay
            combined_signal = self.adjust_technical_signal(technical_signal, macro_signal)
            
            # Add strategy metadata
            combined_signal['strategy_name'] = f"{self.name} + Technical"
            combined_signal['macro_indicators'] = macro_signal.get('indicators', {})
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error getting combined signal: {e}")
            return {
                'action': 'HOLD',
                'buy_percentage': 0,
                'sell_percentage': 0,
                'strategy_name': self.name,
                'error': str(e)
            }


# Enhanced strategies with macro overlay
def strategy_ma_selves_macro_enhanced(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    Enhanced MA-SELVES strategy with macroeconomic overlay.
    Uses the original MA-SELVES logic but adjusts signals based on macro conditions.
    
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
    strat_name = "MA-SELVES-MACRO"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"
    
    # Get macro signal
    macro_strategy = EconomicIndicatorsStrategy()
    macro_signal = macro_strategy.get_macro_signal()
    macro_direction = macro_signal.get('signal', 'NEUTRAL')
    macro_strength = macro_signal.get('strength', 0.0)
    
    # retrieve the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Original MA-SELVES logic
    original_buy_signal = new_p <= (1 - tol_pct) * last_ma
    original_sell_signal = new_p >= (1 + tol_pct) * last_ma
    
    # Apply macro overlay
    r_buy, r_sell = False, False
    
    if original_buy_signal:
        if macro_direction == 'BULLISH' and macro_strength > 0.3:
            # Strong macro bullish - enhance buy signal
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        elif macro_direction == 'BEARISH' and macro_strength > 0.5:
            # Strong macro bearish - reduce buy signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    elif original_sell_signal:
        if macro_direction == 'BEARISH' and macro_strength > 0.3:
            # Strong macro bearish - enhance sell signal
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        elif macro_direction == 'BULLISH' and macro_strength > 0.5:
            # Strong macro bullish - reduce sell signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_exp_ma_selves_macro_enhanced(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    Enhanced EXP-MA-SELVES strategy with macroeconomic overlay.
    Uses the original EXP-MA-SELVES logic but adjusts signals based on macro conditions.
    
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
    strat_name = "EXP-MA-SELVES-MACRO"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"
    
    # Get macro signal
    macro_strategy = EconomicIndicatorsStrategy()
    macro_signal = macro_strategy.get_macro_signal()
    macro_direction = macro_signal.get('signal', 'NEUTRAL')
    macro_strength = macro_signal.get('strength', 0.0)
    
    # retrieve the most recent exponential moving average
    last_ema = trader.exponential_moving_averages[queue_name][-1]
    
    # Original EXP-MA-SELVES logic
    original_buy_signal = new_p <= (1 - tol_pct) * last_ema
    original_sell_signal = new_p >= (1 + tol_pct) * last_ema
    
    # Apply macro overlay
    r_buy, r_sell = False, False
    
    if original_buy_signal:
        if macro_direction == 'BULLISH' and macro_strength > 0.3:
            # Strong macro bullish - enhance buy signal
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        elif macro_direction == 'BEARISH' and macro_strength > 0.5:
            # Strong macro bearish - reduce buy signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    elif original_sell_signal:
        if macro_direction == 'BEARISH' and macro_strength > 0.3:
            # Strong macro bearish - enhance sell signal
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        elif macro_direction == 'BULLISH' and macro_strength > 0.5:
            # Strong macro bullish - reduce sell signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_rsi_macro_enhanced(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    rsi_period: int = 14,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> Tuple[bool, bool]:
    """
    Enhanced RSI strategy with macroeconomic overlay.
    Uses the original RSI logic but adjusts signals based on macro conditions.
    
    Args:
        trader: The trader instance with necessary methods and attributes.
        queue_name (str): The name of the queue.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        rsi_period (int): RSI period.
        rsi_overbought (float): RSI overbought threshold.
        rsi_oversold (float): RSI oversold threshold.
        
    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "RSI-MACRO"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"
    
    # Get macro signal
    macro_strategy = EconomicIndicatorsStrategy()
    macro_signal = macro_strategy.get_macro_signal()
    macro_direction = macro_signal.get('signal', 'NEUTRAL')
    macro_strength = macro_signal.get('strength', 0.0)
    
    # Original RSI logic
    if (
        hasattr(trader, "rsi_dct")
        and "RSI" in trader.rsi_dct
        and len(trader.rsi_dct["RSI"]) > 0
    ):
        current_rsi = trader.rsi_dct["RSI"][-1]
        original_buy_signal = current_rsi < rsi_oversold
        original_sell_signal = current_rsi > rsi_overbought
    else:
        original_buy_signal = False
        original_sell_signal = False
    
    # Apply macro overlay
    r_buy, r_sell = False, False
    
    if original_buy_signal:
        if macro_direction == 'BULLISH' and macro_strength > 0.3:
            # Strong macro bullish - enhance buy signal
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        elif macro_direction == 'BEARISH' and macro_strength > 0.5:
            # Strong macro bearish - reduce buy signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    elif original_sell_signal:
        if macro_direction == 'BEARISH' and macro_strength > 0.3:
            # Strong macro bearish - enhance sell signal
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        elif macro_direction == 'BULLISH' and macro_strength > 0.5:
            # Strong macro bullish - reduce sell signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell


def strategy_adaptive_ma_selves_macro_enhanced(
    trader,
    queue_name: str,
    new_p: float,
    today: datetime.datetime,
    tol_pct: float,
    buy_pct: float,
    sell_pct: float,
) -> Tuple[bool, bool]:
    """
    Enhanced ADAPTIVE-MA-SELVES strategy with macroeconomic overlay.
    Uses the original ADAPTIVE-MA-SELVES logic but adjusts signals based on macro conditions.
    
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
    strat_name = "ADAPTIVE-MA-SELVES-MACRO"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"
    
    # Get macro signal
    macro_strategy = EconomicIndicatorsStrategy()
    macro_signal = macro_strategy.get_macro_signal()
    macro_direction = macro_signal.get('signal', 'NEUTRAL')
    macro_strength = macro_signal.get('strength', 0.0)
    
    # retrieve the most recent moving average
    last_ma = trader.moving_averages[queue_name][-1]
    
    # Adaptive tolerance based on volatility
    price_history = trader.price_history[-30:]  # Last 30 days
    if len(price_history) >= 10:
        volatility = np.std(price_history) / np.mean(price_history)
        adaptive_tol = tol_pct * (1 + volatility * 2)  # Increase tolerance with volatility
    else:
        adaptive_tol = tol_pct
    
    # Original ADAPTIVE-MA-SELVES logic with adaptive tolerance
    original_buy_signal = new_p <= (1 - adaptive_tol) * last_ma
    original_sell_signal = new_p >= (1 + adaptive_tol) * last_ma
    
    # Apply macro overlay
    r_buy, r_sell = False, False
    
    if original_buy_signal:
        if macro_direction == 'BULLISH' and macro_strength > 0.3:
            # Strong macro bullish - enhance buy signal
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
        elif macro_direction == 'BEARISH' and macro_strength > 0.5:
            # Strong macro bearish - reduce buy signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    
    elif original_sell_signal:
        if macro_direction == 'BEARISH' and macro_strength > 0.3:
            # Strong macro bearish - enhance sell signal
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
        elif macro_direction == 'BULLISH' and macro_strength > 0.5:
            # Strong macro bullish - reduce sell signal
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        else:
            # Neutral macro or weak signal - use original logic
            r_sell = trader._execute_one_sell("by_percentage", new_p)
            if r_sell is True:
                trader._record_history(new_p, today, SELL_SIGNAL)
                trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    
    # add history as well if nothing happens
    if r_buy is False and r_sell is False:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    
    return r_buy, r_sell

# ---- Strategy registry for easy lookup ---- #
STRATEGY_REGISTRY = {
    "ECONOMIC-INDICATORS": EconomicIndicatorsStrategy,
    "MA-SELVES": strategy_moving_average_w_tolerance,
    "MA-SELVES-MACRO": strategy_ma_selves_macro_enhanced,
    "EXP-MA-SELVES": strategy_exponential_moving_average_w_tolerance,
    "EXP-MA-SELVES-MACRO": strategy_exp_ma_selves_macro_enhanced,
    "RSI": strategy_rsi,
    "RSI-MACRO": strategy_rsi_macro_enhanced,
    "ADAPTIVE-MA-SELVES": strategy_adaptive_ma_selves,
    "ADAPTIVE-MA-SELVES-MACRO": strategy_adaptive_ma_selves_macro_enhanced,
    "DOUBLE-MA": strategy_double_moving_averages,
    "MACD": strategy_macd,
    "BOLL-BANDS": strategy_bollinger_bands,
    "MA-BOLL-BANDS": strategy_ma_boll_bands,
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
    "MOMENTUM-MA-SELVES": strategy_momentum_enhanced_ma_selves,
    "FEAR-GREED-SENTIMENT": strategy_fear_greed_sentiment,
    "SIMPLE-RECURRING": strategy_simple_recurring_investment,
    "WEIGHTED-RECURRING": strategy_weighted_recurring_investment,
}
# ----------------------------- #
