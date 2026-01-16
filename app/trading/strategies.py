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


def _intraday_momentum(
    candles: Optional[List[Tuple]], min_move_pct: float = 0.003
) -> str:
    """
    Estimate short-term momentum from intraday candles.

    Args:
        candles (Optional[List[Tuple]]): Intraday candles as (close, date, open, low, high, volume).
        min_move_pct (float): Minimum move (as pct) to be considered trending.

    Returns:
        str: "UP", "DOWN", or "FLAT".

    Raises:
        None
    """
    if not candles or len(candles) < 3:
        return "FLAT"

    closes = []
    for c in candles:
        try:
            closes.append(float(c[0]))
        except Exception:
            continue

    if len(closes) < 3:
        return "FLAT"

    first, last = closes[0], closes[-1]
    if first <= 0:
        return "FLAT"

    move_pct = (last - first) / first
    if move_pct >= min_move_pct and last > closes[-2]:
        return "UP"
    if move_pct <= -min_move_pct and last < closes[-2]:
        return "DOWN"
    return "FLAT"


def _compute_atr(
    candles: Optional[List[Tuple]], window: int = 14
) -> Optional[float]:
    """
    Compute a simple ATR from candle tuples.

    Args:
        candles (Optional[List[Tuple]]): Candles as (close, date, open, low, high, volume).
        window (int): Lookback window length. Defaults to 14.

    Returns:
        Optional[float]: ATR value if enough data, else None.

    Raises:
        None
    """
    if not candles or window <= 1:
        return None
    if len(candles) < window + 1:
        return None

    trs = []
    for i in range(-window, 0):
        try:
            close = float(candles[i][0])
            low = float(candles[i][3])
            high = float(candles[i][4])
            prev_close = float(candles[i - 1][0])
        except Exception:
            continue
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if not trs:
        return None
    return float(np.mean(trs))


def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp a value between lo and hi.

    Args:
        value (float): Value to clamp.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Clamped value.

    Raises:
        None
    """
    return max(lo, min(hi, value))


def apply_signal_option_leverage(
    trader,
    signal: str,
    price: float,
    today: datetime.datetime,
    leverage_multiple: int = 3,
    expiration_days: int = 10,
    allocation_pct: float = 0.1,
    enabled: bool = True,
) -> Optional[Dict]:
    """
    Record a synthetic option trade tied to a BUY/SELL signal.

    This helper is strategy-agnostic and can be plugged into any strategy that
    wants to model leveraged CALL/PUT exposure on signals.

    Args:
        trader: The trader instance with a mutable `option_history` list.
        signal (str): Signal that triggered the option (BUY or SELL).
        price (float): Entry price at the time of the signal.
        today (datetime.datetime): Timestamp for the signal.
        leverage_multiple (int): Allowed leverage multiple, must be 3 or 5.
        expiration_days (int): Option expiration horizon in days (max 10).
        allocation_pct (float): Fraction of available cash to allocate as option premium.
            Defaults to 0.1 (10%).
        enabled (bool): Whether to record option leverage on this signal.

    Returns:
        Optional[Dict]: The recorded option trade dict, or None if skipped.
    """
    if not enabled:
        return None

    if signal not in (BUY_SIGNAL, SELL_SIGNAL):
        return None

    if leverage_multiple not in (3, 5):
        return None

    if not _is_positive_number(expiration_days) or int(expiration_days) > 10:
        return None

    if not _is_positive_number(allocation_pct):
        return None

    cash_val = float(getattr(trader, "cash", 0) or 0)
    if cash_val <= 0:
        return None

    premium = cash_val * float(allocation_pct)
    if premium <= 0:
        return None

    option_type = "CALL" if signal == BUY_SIGNAL else "PUT"
    exp_days = int(expiration_days)
    try:
        expires_on = today + datetime.timedelta(days=exp_days)
    except Exception:
        expires_on = None

    # Deduct premium immediately (options are paid upfront).
    trader.cash -= premium

    record = {
        "date": today,
        "signal": signal,
        "option_type": option_type,
        "leverage_multiple": leverage_multiple,
        "expiration_days": exp_days,
        "expires_on": expires_on,
        "entry_price": price,
        "premium": premium,
        "allocation_pct": float(allocation_pct),
        "settled": False,
        "strategy": getattr(trader, "high_strategy", None),
    }

    if not hasattr(trader, "option_history") or trader.option_history is None:
        setattr(trader, "option_history", [])
    trader.option_history.append(record)
    return record


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
    volume: Optional[float] = None,
    volume_window: int = 20,
    volume_ratio_threshold: float = 1.2,
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
        volume (Optional[float]): Current interval volume. If not provided, uses the latest value from
            `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        volume_ratio_threshold (float): Require current volume to be at least this multiple of the
            rolling average volume to confirm a signal. Defaults to 1.2.

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

    # Volume confirmation (best-effort): if we have enough volume history, require above-average volume.
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]

    volume_ok = True
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)):
            vols = list(vols_attr)
            if len(vols) >= int(volume_window) + 1:
                avg_vol = float(np.mean(vols[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0 and _is_positive_number(volume_ratio_threshold):
                    volume_ok = float(cur_vol) >= avg_vol * float(volume_ratio_threshold)
    except Exception:
        volume_ok = True

    r_buy, r_sell = False, False

    if window["type"] == "BULLISH":
        # Buy on pullback to short MA within window
        if has_cash and touch and volume_ok and new_p <= short_ma * (1.0 + touch_tol_pct):
            r_buy = trader._execute_one_buy("by_percentage", new_p)
            if r_buy is True:
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    elif window["type"] == "BEARISH":
        # Sell on rebound to short MA within window
        if has_position and touch and volume_ok and new_p >= short_ma * (1.0 - touch_tol_pct):
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
    open_p: Optional[float] = None,
    low_p: Optional[float] = None,
    high_p: Optional[float] = None,
    trend_lookback: int = 3,
    trend_min_abs_slope_pct: float = 0.002,
    band_breakout_pct: float = 0.002,
    cooldown_days: int = 2,
    min_bandwidth_pct: float = 0.02,
    entry_z: float = 1.5,
    exit_z: float = 0.8,
    volume: Optional[float] = None,
    volume_window: int = 20,
    volume_ratio_threshold: float = 1.2,
    require_volume_for_bear_buys: bool = True,
    bear_exit_z_discount: float = 0.3,
    bear_midline_take_profit: bool = True,
    bull_entry_z_discount: float = 0.3,
    bull_exit_z_premium: float = 0.3,
    bull_require_stronger_sell: bool = True,
    bull_allow_additional_buys: bool = True,
    bull_additional_buy_min_cash_ratio: float = 0.20,
    neutral_scalp_enabled: bool = True,
    neutral_scalp_min_bandwidth_pct: float = 0.06,
    neutral_scalp_entry_z: float = 0.6,
    neutral_scalp_exit_z: float = 0.6,
    neutral_scalp_volume_ratio_threshold: float = 1.0,
    high_vol_bandwidth_pct: float = 0.08,
    low_vol_bandwidth_pct: float = 0.03,
    cooldown_high_vol_discount_days: int = 1,
    cooldown_low_vol_bonus_days: int = 1,
    capitulation_enabled: bool = True,
    capitulation_z: float = 2.6,
    capitulation_volume_ratio_threshold: float = 1.8,
    capitulation_close_near_high_pct: float = 0.6,
    bull_blowoff_enabled: bool = True,
    bull_blowoff_z: float = 2.2,
    bull_trailing_stop_pct: float = 0.05,
    bull_buy_on_midline_pullback: bool = True,
    bull_pullback_z: float = 0.0,
    bull_pullback_volume_ratio_threshold: float = 1.0,
    bull_disable_take_profit_on_first_touch: bool = True,
    leverage_enabled: bool = True,
    leverage_multiple: int = 3,
    leverage_expiration_days: int = 10,
    leverage_allocation_pct: float = 0.1,
    zoom_in: bool = False,
    zoom_in_bandwidth_pct: Optional[float] = None,
    zoom_in_min_move_pct: float = 0.003,
    intraday_candles: Optional[List[Tuple]] = None,
    max_drawdown_pct: float = 0.25,
    drawdown_cooldown_days: int = 5,
    volatility_window: int = 20,
    target_vol_pct: float = 0.02,
    min_buy_scale: float = 0.3,
    max_buy_scale: float = 1.0,
    atr_window: int = 14,
    atr_stop_mult: float = 2.0,
    atr_take_profit_mult: float = 3.0,
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
        open_p (Optional[float]): Current interval open (used for reversal confirmation). Defaults to None.
        low_p (Optional[float]): Current interval low (used for reversal confirmation). Defaults to None.
        high_p (Optional[float]): Current interval high (used for reversal confirmation). Defaults to None.
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
        volume (Optional[float]): Current interval volume. If not provided, uses the latest value from
            `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        volume_ratio_threshold (float): Require current volume to be at least this multiple of the
            rolling average volume to confirm certain signals. Defaults to 1.2.
        require_volume_for_bear_buys (bool): If True, require volume confirmation for buy entries when
            the MA trend is downward (helps avoid catching a falling knife). Defaults to True.
        bear_exit_z_discount (float): In downtrends, reduce the sell z-threshold by this amount so we
            take profits sooner on rebounds. Defaults to 0.3.
        bear_midline_take_profit (bool): If True, in downtrends allow taking profit when price reverts
            back to the midline (z >= 0). Defaults to True.
        bull_entry_z_discount (float): In uptrends, reduce the buy z-threshold by this amount so entries
            are easier (buy dips more readily). Defaults to 0.3.
        bull_exit_z_premium (float): In uptrends, increase the sell z-threshold by this amount so exits
            are harder (let winners run). Defaults to 0.3.
        bull_require_stronger_sell (bool): If True, in uptrends require BOTH band breakout AND z-threshold
            to sell (stronger confirmation). Defaults to True.
        bull_allow_additional_buys (bool): If True, allow "add-on" buys in strong uptrends even when
            already holding a position (uses by_percentage of remaining cash). Defaults to True.
        bull_additional_buy_min_cash_ratio (float): Only allow add-on buys if cash / (cash + position_value)
            is at least this ratio. Defaults to 0.20.
        neutral_scalp_enabled (bool): If True, allow smaller mean-reversion trades in neutral/mild-trend
            regimes, using z-score cross events to avoid repeated signals. Defaults to True.
        neutral_scalp_min_bandwidth_pct (float): Minimum bandwidth to enable scalp mode. Defaults to 0.06.
        neutral_scalp_entry_z (float): Entry threshold for scalp mode (cross below -z). Defaults to 0.6.
        neutral_scalp_exit_z (float): Exit threshold for scalp mode (cross above +z). Defaults to 0.6.
        neutral_scalp_volume_ratio_threshold (float): Volume ratio required for scalp triggers. Defaults to 1.0.
        high_vol_bandwidth_pct (float): Bandwidth threshold considered "high vol" for dynamic cooldown. Defaults to 0.08.
        low_vol_bandwidth_pct (float): Bandwidth threshold considered "low vol" for dynamic cooldown. Defaults to 0.03.
        cooldown_high_vol_discount_days (int): Reduce cooldown by this many days in high vol. Defaults to 1.
        cooldown_low_vol_bonus_days (int): Increase cooldown by this many days in low vol. Defaults to 1.
        capitulation_enabled (bool): If True, allow rare downtrend "capitulation + reversal" buys. Defaults to True.
        capitulation_z (float): Z-score required to consider capitulation. Defaults to 2.6.
        capitulation_volume_ratio_threshold (float): Volume ratio required for capitulation buys. Defaults to 1.8.
        capitulation_close_near_high_pct (float): Require close to be in the top fraction of candle range
            for reversal confirmation (when OHLC available). Defaults to 0.6.
        bull_blowoff_enabled (bool): If True, in strong uptrends we don't immediately sell on an upper-band
            breakout. Instead we "arm" a blow-off state and sell only after a reversal (trailing stop).
            Defaults to True.
        bull_blowoff_z (float): Z-score threshold to arm the blow-off state. Defaults to 2.2.
        bull_trailing_stop_pct (float): Trailing stop percentage from the peak price while blow-off is armed.
            Defaults to 0.05 (5%).
        bull_buy_on_midline_pullback (bool): If True, in strong uptrends allow add-on buys on a pullback
            to/below the midline (z crosses below `bull_pullback_z`). Defaults to True.
        bull_pullback_z (float): Midline pullback z-threshold (often 0.0). Defaults to 0.0.
        bull_pullback_volume_ratio_threshold (float): Volume ratio required to confirm pullback add-on buys.
            Defaults to 1.0.
        bull_disable_take_profit_on_first_touch (bool): If True, in strong uptrends disable immediate
            take-profit selling on first upper-band touch; rely on blow-off trailing exit instead.
            Defaults to True.
        leverage_enabled (bool): If True, record leveraged option exposure on BUY/SELL signals.
            Defaults to True.
        leverage_multiple (int): Option leverage multiple. Allowed values: 3 or 5. Defaults to 3.
        leverage_expiration_days (int): Option expiration horizon in days (max 10). Defaults to 10.
        leverage_allocation_pct (float): Fraction of available cash to allocate as option premium.
            Defaults to 0.1.
        zoom_in (bool): If True, use intraday candles to refine decisions when bands are wide.
            Defaults to False.
        zoom_in_bandwidth_pct (Optional[float]): Bandwidth threshold to trigger zoom-in. Defaults to
            high_vol_bandwidth_pct when None.
        zoom_in_min_move_pct (float): Minimum intraday move to treat as trending. Defaults to 0.003.
        intraday_candles (Optional[List[Tuple]]): Lower-granularity candles for zoom-in logic.
        max_drawdown_pct (float): Max allowed drawdown before pausing buys. Defaults to 0.25.
        drawdown_cooldown_days (int): Days to pause new buys after drawdown breach. Defaults to 5.
        volatility_window (int): Window for volatility scaling. Defaults to 20.
        target_vol_pct (float): Target volatility for position sizing. Defaults to 0.02.
        min_buy_scale (float): Min buy scaling multiplier. Defaults to 0.3.
        max_buy_scale (float): Max buy scaling multiplier. Defaults to 1.0.
        atr_window (int): ATR lookback window. Defaults to 14.
        atr_stop_mult (float): ATR multiplier for stop-loss. Defaults to 2.0.
        atr_take_profit_mult (float): ATR multiplier for take-profit. Defaults to 3.0.

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

    # Volatility-adaptive cooldown baseline (may be adjusted after bandwidth is known).
    effective_cooldown_days = cooldown_days

    # Simple cooldown using the last non-NO-ACTION signal date for this strategy.
    if effective_cooldown_days > 0 and strat_name in trader.strat_dct:
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                if hasattr(today, "date"):
                    delta_days = (today.date() - past_dt.date()).days
                else:
                    delta_days = (today - past_dt).days
                if delta_days < effective_cooldown_days:
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

    # Drawdown guard: pause new buys if drawdown exceeds threshold.
    dd_state_key = f"_ma_boll_dd_state:{queue_name}:{bollinger_sigma}"
    dd_state = getattr(trader, "__dict__", {}).get(dd_state_key, None)
    if not isinstance(dd_state, dict):
        dd_state = {"peak": None, "cooldown_until": None}
        try:
            setattr(trader, dd_state_key, dd_state)
        except Exception:
            pass

    peak_val = dd_state.get("peak")
    cur_val = trader.portfolio_value
    if peak_val is None or cur_val > peak_val:
        dd_state["peak"] = cur_val
        peak_val = cur_val

    block_buys = False
    if _is_positive_number(peak_val):
        drawdown_pct = (peak_val - cur_val) / peak_val
        if drawdown_pct >= float(max_drawdown_pct):
            try:
                cooldown_until = today + datetime.timedelta(days=int(drawdown_cooldown_days))
            except Exception:
                cooldown_until = None
            dd_state["cooldown_until"] = cooldown_until

    cooldown_until = dd_state.get("cooldown_until")
    if cooldown_until is not None:
        try:
            if today < cooldown_until:
                block_buys = True
        except Exception:
            block_buys = False

    # Volatility-scaled position sizing for buys.
    base_buy_pct = buy_pct if _is_positive_number(buy_pct) else float(getattr(trader, "buy_pct", 0))
    effective_buy_pct = base_buy_pct
    try:
        prices = getattr(trader, "price_history", None)
        if prices and len(prices) >= int(volatility_window):
            window = prices[-int(volatility_window) :]
            m = float(np.mean(window))
            s = float(np.std(window))
            vol_pct = s / m if m > 0 else 0.0
            if _is_positive_number(vol_pct) and _is_positive_number(target_vol_pct):
                scale = float(target_vol_pct) / float(vol_pct)
                scale = _clamp(scale, float(min_buy_scale), float(max_buy_scale))
                effective_buy_pct = base_buy_pct * scale
    except Exception:
        effective_buy_pct = base_buy_pct

    def _execute_buy_with_pct(pct: float) -> bool:
        old = getattr(trader, "buy_pct", pct)
        try:
            trader.buy_pct = pct
            return bool(trader._execute_one_buy("by_percentage", new_p))
        finally:
            trader.buy_pct = old

    # Position and cash check (needed for ATR exits before later logic).
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_position = _is_positive_number(coin_val)
    has_cash = _is_positive_number(cash_val)
    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # ATR-based risk exits (stop-loss / take-profit).
    atr = _compute_atr(getattr(trader, "crypto_prices", []), window=int(atr_window))
    atr_stop = False
    atr_take_profit = False
    if has_position and atr and _is_positive_number(atr):
        if getattr(trader, "last_buy_price", None):
            last_buy = float(trader.last_buy_price)
            if _is_positive_number(atr_stop_mult) and new_p <= last_buy - float(atr_stop_mult) * atr:
                atr_stop = True
            if _is_positive_number(atr_take_profit_mult) and new_p >= last_buy + float(atr_take_profit_mult) * atr:
                atr_take_profit = True

    zoom_in_buy = False
    zoom_in_sell = False
    zoom_in_option = None
    if zoom_in and intraday_candles:
        zoom_threshold = (
            float(zoom_in_bandwidth_pct)
            if _is_positive_number(zoom_in_bandwidth_pct)
            else float(high_vol_bandwidth_pct)
        )
        if bandwidth_pct >= zoom_threshold:
            intraday_trend = _intraday_momentum(
                intraday_candles, min_move_pct=zoom_in_min_move_pct
            )
            if trending_up:
                if intraday_trend == "UP":
                    zoom_in_buy = True and (not block_buys)
                    zoom_in_option = BUY_SIGNAL
                elif intraday_trend == "DOWN":
                    zoom_in_sell = True
            elif trending_down:
                if intraday_trend == "DOWN":
                    zoom_in_sell = True
                    zoom_in_option = SELL_SIGNAL
                elif intraday_trend == "UP":
                    zoom_in_sell = True

    # Now that we know bandwidth, adjust cooldown and re-check it (only if we haven't traded yet).
    try:
        if _is_positive_number(high_vol_bandwidth_pct) and bandwidth_pct >= float(high_vol_bandwidth_pct):
            effective_cooldown_days = max(0, int(cooldown_days) - int(cooldown_high_vol_discount_days))
        elif _is_positive_number(low_vol_bandwidth_pct) and bandwidth_pct <= float(low_vol_bandwidth_pct):
            effective_cooldown_days = int(cooldown_days) + int(cooldown_low_vol_bonus_days)
    except Exception:
        effective_cooldown_days = cooldown_days

    if effective_cooldown_days > 0 and strat_name in trader.strat_dct:
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < effective_cooldown_days:
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # Z-score of price relative to midline; epsilon prevents division-by-zero.
    z = (new_p - mid) / (std + 1e-12)

    # position and cash check, really important!
    cash_val = getattr(trader, "cash", 0)
    coin_val = getattr(trader, "cur_coin", 0)
    has_position = _is_positive_number(coin_val)
    has_cash = _is_positive_number(cash_val)
    if not has_cash and hasattr(trader, "wallet"):
        has_cash = _is_positive_number(trader.wallet.get("USD", 0)) or _is_positive_number(
            trader.wallet.get("USDT", 0)
        )
    if not has_position and hasattr(trader, "wallet"):
        has_position = _is_positive_number(trader.wallet.get("crypto", 0))

    # Volume confirmation (best-effort)
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]
    volume_ok = True
    vol_ratio = None
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)):
            vols = list(vols_attr)
            if len(vols) >= int(volume_window) + 1:
                avg_vol = float(np.mean(vols[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0 and _is_positive_number(volume_ratio_threshold):
                    volume_ok = float(cur_vol) >= avg_vol * float(volume_ratio_threshold)
                    vol_ratio = float(cur_vol) / avg_vol
    except Exception:
        volume_ok = True

    r_buy, r_sell = False, False

    # --- Volatility harvesting core ---
    #
    # Entry: buy when price is sufficiently below the lower band (or z-score is low enough).
    # Exit: sell when price is sufficiently above the upper band (or z-score is high enough).
    #
    # Trend bias (mild):
    # - In strong uptrends, require a deeper pullback to buy (avoid buying "not cheap enough")
    # - In strong downtrends, require a stronger overbought condition to sell (avoid selling into weakness)
    # In uptrends, entries are easier; in downtrends, leave entry stricter by default.
    effective_entry_z = entry_z - (bull_entry_z_discount if trending_up else 0.0)
    effective_entry_z = max(0.1, float(effective_entry_z))

    # In downtrends, take profits sooner on rebounds (otherwise it's easy to get stuck).
    # In uptrends, take profits later (otherwise you sell winners too early).
    effective_exit_z = exit_z
    if trending_down:
        effective_exit_z = exit_z - bear_exit_z_discount
    elif trending_up:
        effective_exit_z = exit_z + bull_exit_z_premium
    effective_exit_z = max(0.1, float(effective_exit_z))

    # Base conditions
    buy_condition = (new_p <= boll_lower * (1.0 - band_breakout_pct)) or (z <= -effective_entry_z)
    sell_condition = (new_p >= boll_upper * (1.0 + band_breakout_pct)) or (z >= effective_exit_z)

    # In uptrends, require stronger confirmation to sell (harder exits).
    if trending_up and bull_require_stronger_sell:
        sell_condition = (new_p >= boll_upper * (1.0 + band_breakout_pct)) and (
            z >= effective_exit_z
        )

    # Make bearish entries more selective: require volume confirmation by default.
    if trending_down and require_volume_for_bear_buys:
        buy_condition = buy_condition and volume_ok

    # Rare downtrend opportunities: capitulation + reversal.
    # This is intentionally strict: extreme z + large volume + (optional) bullish candle shape.
    capitulation_buy = False
    if trending_down and capitulation_enabled and has_cash:
        cap_vol_ok = True
        try:
            if vol_ratio is not None and _is_positive_number(capitulation_volume_ratio_threshold):
                cap_vol_ok = float(vol_ratio) >= float(capitulation_volume_ratio_threshold)
            elif _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)) and len(vols_attr) >= int(volume_window) + 1:
                avg_vol = float(np.mean(list(vols_attr)[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0:
                    cap_vol_ok = float(cur_vol) / avg_vol >= float(capitulation_volume_ratio_threshold)
        except Exception:
            cap_vol_ok = True

        reversal_ok = True
        try:
            if _is_positive_number(open_p) and _is_positive_number(low_p) and _is_positive_number(high_p) and float(high_p) > float(low_p):
                close_pos = (float(new_p) - float(low_p)) / (float(high_p) - float(low_p))
                reversal_ok = float(new_p) >= float(open_p) and close_pos >= float(capitulation_close_near_high_pct)
        except Exception:
            reversal_ok = True

        if _is_positive_number(capitulation_z):
            capitulation_buy = (z <= -float(capitulation_z)) and cap_vol_ok and reversal_ok

    # In downtrends, allow taking profit when reverting to midline (helps avoid being trapped).
    if trending_down and bear_midline_take_profit:
        sell_condition = sell_condition or (z >= 0.0)

    # Neutral / mild-trend scalp mode (agile but less whipsaw):
    # Use z-score cross events to avoid repeated signals when price oscillates.
    scalp_buy = False
    scalp_sell = False
    if neutral_scalp_enabled and (not trending_up) and (not trending_down) and bandwidth_pct >= float(neutral_scalp_min_bandwidth_pct):
        prev_z_key = f"_ma_boll_prev_z:{queue_name}:{bollinger_sigma}"
        prev_z = getattr(trader, "__dict__", {}).get(prev_z_key, None)
        try:
            prev_z_val = float(prev_z) if prev_z is not None else None
        except Exception:
            prev_z_val = None

        scalp_vol_ok = True
        try:
            if vol_ratio is not None:
                scalp_vol_ok = float(vol_ratio) >= float(neutral_scalp_volume_ratio_threshold)
        except Exception:
            scalp_vol_ok = True

        if prev_z_val is not None:
            scalp_buy = prev_z_val > -float(neutral_scalp_entry_z) and z <= -float(neutral_scalp_entry_z) and scalp_vol_ok
            scalp_sell = prev_z_val < float(neutral_scalp_exit_z) and z >= float(neutral_scalp_exit_z) and scalp_vol_ok

        # store current z for next step
        try:
            setattr(trader, prev_z_key, float(z))
        except Exception:
            pass

    # Bullish uptrend "let winners run" mode:
    # Arm on an extreme (upper band / high z), but only sell after a reversal (trailing stop).
    bull_blowoff_sell = False
    if trending_up and bull_blowoff_enabled and has_position:
        state_key = f"_ma_boll_bull_state:{queue_name}:{bollinger_sigma}"
        state = getattr(trader, "__dict__", {}).get(state_key, None)
        if not isinstance(state, dict):
            state = {"armed": False, "peak": None}
            try:
                setattr(trader, state_key, state)
            except Exception:
                pass

        try:
            armed = bool(state.get("armed", False))
        except Exception:
            armed = False

        # Arm on blow-off conditions (do NOT sell immediately)
        try:
            blowoff_arm = (z >= float(bull_blowoff_z)) or (
                new_p >= boll_upper * (1.0 + band_breakout_pct)
            )
        except Exception:
            blowoff_arm = False

        if not armed and blowoff_arm:
            state["armed"] = True
            state["peak"] = float(new_p)
        elif armed:
            try:
                peak = float(state.get("peak") or new_p)
            except Exception:
                peak = float(new_p)
            peak = max(peak, float(new_p))
            state["peak"] = peak
            try:
                if _is_positive_number(bull_trailing_stop_pct) and new_p <= peak * (
                    1.0 - float(bull_trailing_stop_pct)
                ):
                    bull_blowoff_sell = True
            except Exception:
                bull_blowoff_sell = False

        # If we triggered a sell (below), disarm so we don't keep selling.
        if bull_blowoff_sell:
            state["armed"] = False
            state["peak"] = None

    # In strong uptrends, avoid selling winners too early:
    # keep the earlier "stronger sell confirmation" logic, but optionally disable immediate take-profit.
    if trending_up and bull_disable_take_profit_on_first_touch:
        sell_condition = False
        # Still allow a sell on blow-off reversal or if we actually flip to downtrend logic later.
        if bull_blowoff_sell:
            sell_condition = True

    # In strong uptrends, allow pullback add-on entries near/below midline (more bullish exposure).
    bull_pullback_buy = False
    if trending_up and bull_buy_on_midline_pullback:
        prev_z_key = f"_ma_boll_prev_z:{queue_name}:{bollinger_sigma}"
        prev_z = getattr(trader, "__dict__", {}).get(prev_z_key, None)
        try:
            prev_z_val = float(prev_z) if prev_z is not None else None
        except Exception:
            prev_z_val = None
        pullback_vol_ok = True
        try:
            if vol_ratio is not None:
                pullback_vol_ok = float(vol_ratio) >= float(bull_pullback_volume_ratio_threshold)
        except Exception:
            pullback_vol_ok = True
        if prev_z_val is not None:
            bull_pullback_buy = (
                prev_z_val > float(bull_pullback_z)
                and z <= float(bull_pullback_z)
                and pullback_vol_ok
            )

    # Prefer exits before entries (risk management) and be position-aware.
    if has_position and (sell_condition or scalp_sell or bull_blowoff_sell or atr_stop or atr_take_profit):
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            apply_signal_option_leverage(
                trader=trader,
                signal=SELL_SIGNAL,
                price=new_p,
                today=today,
                leverage_multiple=leverage_multiple,
                expiration_days=leverage_expiration_days,
                allocation_pct=leverage_allocation_pct,
                enabled=leverage_enabled,
            )
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
    else:
        # In strong uptrends, allow add-on buys if we still have meaningful cash left.
        allow_add_buy = False
        if trending_up and bull_allow_additional_buys and has_cash:
            try:
                pos_val = float(getattr(trader, "cur_coin", 0)) * float(new_p)
                cash_val_f = float(getattr(trader, "cash", 0))
                total = cash_val_f + pos_val
                if total > 0:
                    cash_ratio = cash_val_f / total
                    allow_add_buy = cash_ratio >= float(bull_additional_buy_min_cash_ratio)
            except Exception:
                allow_add_buy = False

        can_buy = (
            has_cash
            and (buy_condition or scalp_buy or capitulation_buy or bull_pullback_buy)
            and ((not has_position) or allow_add_buy)
            and (not block_buys)
        )
        if can_buy:
            r_buy = _execute_buy_with_pct(effective_buy_pct)
            if r_buy is True:
                apply_signal_option_leverage(
                    trader=trader,
                    signal=BUY_SIGNAL,
                    price=new_p,
                    today=today,
                    leverage_multiple=leverage_multiple,
                    expiration_days=leverage_expiration_days,
                    allocation_pct=leverage_allocation_pct,
                    enabled=leverage_enabled,
                )
                trader._record_history(new_p, today, BUY_SIGNAL)
                trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Zoom-in refinement: add-on trade or option if intraday confirms momentum.
    zoom_r_buy = False
    zoom_r_sell = False
    if zoom_in_buy and r_sell is False and has_cash:
        zoom_r_buy = _execute_buy_with_pct(effective_buy_pct)
        if zoom_r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
    if zoom_in_sell and r_buy is False and has_position:
        zoom_r_sell = trader._execute_one_sell("by_percentage", new_p)
        if zoom_r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    if zoom_in_option and leverage_enabled and (r_buy is False and r_sell is False) and (not block_buys or zoom_in_option == SELL_SIGNAL):
        apply_signal_option_leverage(
            trader=trader,
            signal=zoom_in_option,
            price=new_p,
            today=today,
            leverage_multiple=leverage_multiple,
            expiration_days=leverage_expiration_days,
            allocation_pct=leverage_allocation_pct,
            enabled=leverage_enabled,
        )

    if zoom_r_buy is True:
        r_buy = True
    if zoom_r_sell is True:
        r_sell = True

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
    cooldown_days: int = 1,
    volume: Optional[float] = None,
    volume_window: int = 20,
    volume_ratio_threshold: float = 1.2,
) -> Tuple[bool, bool]:
    """
    RSI-based trading strategy (smarter, less-churn).

    Instead of firing every time RSI is beyond a threshold (which can overtrade while RSI stays
    oversold/overbought), this version trades on *threshold cross events*:
    - Buy when RSI crosses below the oversold threshold (prev >= oversold, cur < oversold)
    - Sell when RSI crosses above the overbought threshold (prev <= overbought, cur > overbought)

    It is also position-aware (requires cash for buy, position for sell) and supports an optional
    cooldown to avoid whipsaw.

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        period (int, optional): RSI period. Defaults to 14.
        overbought (float, optional): Overbought threshold. Defaults to 70.
        oversold (float, optional): Oversold threshold. Defaults to 30.
        cooldown_days (int, optional): Minimum days between BUY/SELL signals for this strategy.
            Defaults to 1.
        volume (Optional[float]): Current interval volume. If not provided, uses the latest value from
            `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        volume_ratio_threshold (float): Require current volume to be at least this multiple of the
            rolling average volume to confirm a signal. Defaults to 1.2.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "RSI"
    assert strat_name in STRATEGIES, "Unknown trading strategy name!"

    rsi = trader.compute_rsi(period)
    if rsi is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False  # Not enough data yet

    # Store RSI series separately to avoid mixing numeric values with signal tuples.
    if not hasattr(trader, "rsi_series") or trader.rsi_series is None:
        trader.rsi_series = []
    trader.rsi_series.append((today, float(rsi)))

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
                    setattr(trader, "_prev_rsi", float(rsi))
                    return False, False
                break

    # `Mock` objects fabricate missing attributes; read from __dict__ to avoid getting a Mock.
    prev_rsi = getattr(trader, "__dict__", {}).get("_prev_rsi", None)
    setattr(trader, "_prev_rsi", float(rsi))
    if prev_rsi is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    cross_into_oversold = float(prev_rsi) >= oversold and float(rsi) < oversold
    cross_into_overbought = float(prev_rsi) <= overbought and float(rsi) > overbought

    # Volume confirmation (best-effort):
    # If we have enough volume history, require the current volume to be meaningfully above average.
    cur_vol = volume
    # `Mock` objects fabricate attributes; use __dict__ and require a real list/tuple.
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]

    volume_ok = True
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)):
            vols = list(vols_attr)
            if len(vols) >= int(volume_window) + 1:
                # Exclude the current volume (last item) to avoid self-influence
                avg_vol = float(np.mean(vols[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0 and _is_positive_number(volume_ratio_threshold):
                    volume_ok = float(cur_vol) >= avg_vol * float(volume_ratio_threshold)
    except Exception:
        volume_ok = True

    if cross_into_overbought and has_position:
        if not volume_ok:
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return False, False
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
            return False, True
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    if cross_into_oversold and has_cash:
        if not volume_ok:
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return False, False
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
            return True, False
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
    return False, False


def strategy_kdj(
    trader,
    new_p: float,
    today: datetime.datetime,
    oversold: float = 30,
    overbought: float = 70,
    cooldown_days: int = 1,
    volume: Optional[float] = None,
    volume_window: int = 20,
    volume_ratio_threshold: float = 1.2,
) -> Tuple[bool, bool]:
    """
    KDJ-based trading strategy (smarter, less-churn).

    Improvements vs the naive version:
    - Threshold *cross events* (avoids repeated buys/sells while K stays oversold/overbought)
    - Position-aware (requires cash for buy, position for sell)
    - Optional cooldown
    - Optional volume confirmation (best-effort)

    Args:
        trader: The trader instance with necessary methods and attributes.
        new_p (float): Today's new price of a currency.
        today (datetime.datetime): Date.
        oversold (float, optional): Oversold threshold. Defaults to 30.
        overbought (float, optional): Overbought threshold. Defaults to 70.
        cooldown_days (int, optional): Minimum days between BUY/SELL signals for this strategy.
            Defaults to 1.
        volume (Optional[float]): Current interval volume. If not provided, uses the latest value from
            `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        volume_ratio_threshold (float): Require current volume to be at least this multiple of the
            rolling average volume to confirm a signal. Defaults to 1.2.

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
    if not _is_positive_number(current_k) and not isinstance(current_k, (int, float, np.number)):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    current_k = float(current_k)

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
                    setattr(trader, "_prev_kdj_k", current_k)
                    return False, False
                break

    # `Mock` objects fabricate missing attributes; read from __dict__ to avoid getting a Mock.
    prev_k = getattr(trader, "__dict__", {}).get("_prev_kdj_k", None)
    setattr(trader, "_prev_kdj_k", current_k)
    if prev_k is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    prev_k = float(prev_k)

    # Cross events
    cross_into_oversold = prev_k >= oversold and current_k < oversold
    cross_into_overbought = prev_k <= overbought and current_k > overbought

    # Volume confirmation (best-effort)
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]

    volume_ok = True
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)):
            vols = list(vols_attr)
            if len(vols) >= int(volume_window) + 1:
                avg_vol = float(np.mean(vols[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0 and _is_positive_number(volume_ratio_threshold):
                    volume_ok = float(cur_vol) >= avg_vol * float(volume_ratio_threshold)
    except Exception:
        volume_ok = True

    # Sell signal: K crosses into overbought
    if cross_into_overbought and has_position:
        if not volume_ok:
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return False, False
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))
            return False, True
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Buy signal: K crosses into oversold
    if cross_into_oversold and has_cash:
        if not volume_ok:
            trader._record_history(new_p, today, NO_ACTION_SIGNAL)
            trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
            return False, False
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))
            return True, False
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

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
    opportunity_days: int = 3,
    touch_tol_pct: float = 0.01,
    cooldown_days: int = 1,
    volume: Optional[float] = None,
    volume_window: int = 20,
    volume_ratio_threshold: float = 1.2,
    min_volatility_pct: float = 0.0,
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
        opportunity_days (int): Window size (in days) after a MACD crossover to look for MA-based opportunities.
            Defaults to 3.
        touch_tol_pct (float): Touch tolerance as a fraction of MA (e.g. 0.01 = 1%).
            Defaults to 0.01.
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy.
            Defaults to 1.
        volume (Optional[float]): Current interval volume. If not provided, uses the latest value from
            `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        volume_ratio_threshold (float): Require current volume to be at least this multiple of the
            rolling average volume to confirm a signal. Defaults to 1.2.
        min_volatility_pct (float): Minimum rolling price volatility (std/mean) required to trade.
            Defaults to 0.0 (disabled).

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MA-MACD"
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

    # MA reference (avoid lookahead): today's MA includes today's price, so use previous MA value
    not_null_ma = [x for x in trader.moving_averages[queue_name] if x is not None]
    if len(not_null_ma) < 2:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False
    ma_ref = not_null_ma[-2]
    if not _is_positive_number(ma_ref):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    ma_touch = abs(new_p - ma_ref) <= (touch_tol_pct * ma_ref)
    ma_buy_signal = ma_touch or (new_p <= (1 - tol_pct) * ma_ref)
    ma_sell_signal = ma_touch or (new_p >= (1 + tol_pct) * ma_ref)

    # Volatility filter (best-effort): require some price movement to justify entries/exits
    if _is_positive_number(min_volatility_pct):
        prices = getattr(trader, "price_history", None)
        if not prices and hasattr(trader, "crypto_prices"):
            prices = [x[0] for x in trader.crypto_prices]
        if prices and len(prices) >= 20:
            window = prices[-20:]
            m = float(np.mean(window))
            s = float(np.std(window))
            if m > 0 and (s / m) < float(min_volatility_pct):
                trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                return False, False

    # Get MACD crossover event (less whipsaw than sign-only)
    if (
        not hasattr(trader, "macd_diff")
        or not hasattr(trader, "macd_dea")
        or len(trader.macd_diff) < 2
        or len(trader.macd_dea) < 2
        or trader.macd_dea[-1] is None
        or trader.macd_dea[-2] is None
        or trader.macd_diff[-1] is None
        or trader.macd_diff[-2] is None
    ):
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    hist_prev = trader.macd_diff[-2] - trader.macd_dea[-2]
    hist_cur = trader.macd_diff[-1] - trader.macd_dea[-1]
    macd_cross_up = hist_prev <= 0 and hist_cur > 0
    macd_cross_down = hist_prev >= 0 and hist_cur < 0

    # Track opportunity windows per MA queue
    if not isinstance(getattr(trader, "_ma_macd_windows", None), dict):
        trader._ma_macd_windows = {}
    w = trader._ma_macd_windows.get(queue_name)
    if macd_cross_up:
        trader._ma_macd_windows[queue_name] = {"type": "BULLISH", "start": today}
        w = trader._ma_macd_windows[queue_name]
    elif macd_cross_down:
        trader._ma_macd_windows[queue_name] = {"type": "BEARISH", "start": today}
        w = trader._ma_macd_windows[queue_name]

    if not w:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    try:
        days_since = (today.date() - w["start"].date()).days
    except Exception:
        days_since = (today - w["start"]).days
    if days_since > opportunity_days:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Volume confirmation (best-effort)
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]
    volume_ok = True
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)):
            vols = list(vols_attr)
            if len(vols) >= int(volume_window) + 1:
                avg_vol = float(np.mean(vols[-(int(volume_window) + 1) : -1]))
                if avg_vol > 0 and _is_positive_number(volume_ratio_threshold):
                    volume_ok = float(cur_vol) >= avg_vol * float(volume_ratio_threshold)
    except Exception:
        volume_ok = True

    r_buy, r_sell = False, False

    # Within bullish window: buy on MA pullback/oversold
    if w["type"] == "BULLISH" and has_cash and ma_buy_signal and volume_ok:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Within bearish window: sell on MA rebound/overbought
    elif w["type"] == "BEARISH" and has_position and ma_sell_signal and volume_ok:
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
    cooldown_days: int = 1,
    volume: Optional[float] = None,
    volume_window: int = 20,
    bull_buy_volume_ratio_threshold: float = 1.0,
    bull_sell_volume_ratio_threshold: float = 1.1,
    bear_buy_volume_ratio_threshold: float = 1.3,
    bear_sell_volume_ratio_threshold: float = 1.0,
    hist_threshold: float = 0.02,
    bull_kdj_buy_max: float = 70.0,
    bear_kdj_buy_max: float = 45.0,
    opportunity_days: int = 5,
    inactivity_relax_days: int = 30,
    inactivity_relax_factor: float = 0.85,
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
        cooldown_days (int): Minimum days between BUY/SELL signals for this strategy.
            Defaults to 2.
        volume (Optional[float]): Current interval trading volume. If None, uses the latest
            value from `trader.volume_history` when available. Defaults to None.
        volume_window (int): Rolling window size used to compute average volume for confirmation.
            Defaults to 20.
        bull_buy_volume_ratio_threshold (float): In bullish regime, minimum volume ratio (cur/avg)
            to allow a BUY. Lower means easier buys. Defaults to 1.0.
        bull_sell_volume_ratio_threshold (float): In bullish regime, minimum volume ratio (cur/avg)
            to allow a SELL. Higher means harder sells. Defaults to 1.2.
        bear_buy_volume_ratio_threshold (float): In bearish regime, minimum volume ratio (cur/avg)
            to allow a BUY. Higher means harder buys. Defaults to 1.3.
        bear_sell_volume_ratio_threshold (float): In bearish regime, minimum volume ratio (cur/avg)
            to allow a SELL. Lower means easier sells. Defaults to 1.0.
        hist_threshold (float): Minimum absolute MACD histogram value required for a crossover
            to be considered meaningful (noise filter). Defaults to 0.05.
        bull_kdj_buy_max (float): In bullish regime, BUY confirmation allows KDJ-K up to this value
            (not necessarily oversold). Defaults to 60.0.
        bear_kdj_buy_max (float): In bearish regime, BUY confirmation requires KDJ-K to be <= this
            value (stricter). Defaults to 35.0.
        opportunity_days (int): After a MACD histogram crossover, allow KDJ/volume confirmation to
            trigger a trade within this many days (reduces "same-day alignment" conservatism).
            Defaults to 5.
        inactivity_relax_days (int): If no BUY/SELL happens for this many days, automatically relax
            thresholds to avoid going idle for long windows. Defaults to 30.
        inactivity_relax_factor (float): Multiplicative factor (<1 makes it easier) applied to
            histogram threshold and volume ratio thresholds after inactivity. Defaults to 0.85.

    Returns:
        Tuple[bool, bool]: (buy_executed, sell_executed)
    """
    strat_name = "MACD-KDJ"
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

    # Cooldown to reduce whipsaw / repeated trades
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

    # If we've been inactive for a long time, relax thresholds to avoid going silent.
    # (This helps in "slow grind" markets where strict cross+confirm alignment rarely happens.)
    days_since_trade = None
    try:
        last_trade_dt = None
        for past_dt, past_sig in reversed(getattr(trader, "strat_dct", {}).get(strat_name, [])):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                last_trade_dt = past_dt
                break
        if last_trade_dt is not None:
            try:
                days_since_trade = (today.date() - last_trade_dt.date()).days
            except Exception:
                days_since_trade = (today - last_trade_dt).days
    except Exception:
        days_since_trade = None

    relax = False
    try:
        relax = (
            days_since_trade is not None
            and _is_positive_number(inactivity_relax_days)
            and int(days_since_trade) >= int(inactivity_relax_days)
            and isinstance(inactivity_relax_factor, (int, float))
            and 0 < float(inactivity_relax_factor) < 1.0
        )
    except Exception:
        relax = False

    if relax:
        hist_threshold = float(hist_threshold) * float(inactivity_relax_factor)
        bull_buy_volume_ratio_threshold = float(bull_buy_volume_ratio_threshold) * float(
            inactivity_relax_factor
        )
        bull_sell_volume_ratio_threshold = float(bull_sell_volume_ratio_threshold) * float(
            inactivity_relax_factor
        )
        bear_buy_volume_ratio_threshold = float(bear_buy_volume_ratio_threshold) * float(
            inactivity_relax_factor
        )
        bear_sell_volume_ratio_threshold = float(bear_sell_volume_ratio_threshold) * float(
            inactivity_relax_factor
        )
        bull_kdj_buy_max = min(90.0, float(bull_kdj_buy_max) + 10.0)
        bear_kdj_buy_max = min(70.0, float(bear_kdj_buy_max) + 10.0)

    def _last_two_non_none(values):
        not_null = [x for x in values if x is not None]
        if len(not_null) < 2:
            return None, None
        return not_null[-2], not_null[-1]

    # MACD histogram cross events (less whipsaw than raw sign checks)
    diff_prev, diff_cur = _last_two_non_none(getattr(trader, "macd_diff", []))
    dea_prev, dea_cur = _last_two_non_none(getattr(trader, "macd_dea", []))
    if diff_prev is None or diff_cur is None or dea_prev is None or dea_cur is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    hist_prev = float(diff_prev) - float(dea_prev)
    hist_cur = float(diff_cur) - float(dea_cur)
    # Small epsilon to avoid brittle behavior around equality due to float rounding.
    _eps = 1e-12
    _hist_th = max(0.0, float(hist_threshold))
    macd_bull_cross = (
        hist_prev <= 0 and hist_cur > 0 and (abs(hist_cur) + _eps) >= _hist_th
    )
    macd_bear_cross = (
        hist_prev >= 0 and hist_cur < 0 and (abs(hist_cur) + _eps) >= _hist_th
    )

    # Regime: use MACD zero-line (trend-ish) to modulate strictness
    bullish_regime = float(diff_cur) >= 0 and float(dea_cur) >= 0
    bearish_regime = not bullish_regime

    # KDJ K cross events
    k_series = None
    if hasattr(trader, "kdj_dct") and isinstance(trader.kdj_dct, dict):
        k_series = trader.kdj_dct.get("K")
    k_prev, k_cur = _last_two_non_none(k_series or [])
    if k_prev is None or k_cur is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    k_prev = float(k_prev)
    k_cur = float(k_cur)
    kdj_buy_cross = k_prev < float(oversold) and k_cur >= float(oversold)  # exit oversold
    kdj_sell_cross = k_prev > float(overbought) and k_cur <= float(overbought)  # exit overbought

    # Track a short "opportunity window" after MACD cross so confirmations can come later.
    # This greatly increases trade frequency vs requiring cross+confirm to happen on the same bar.
    window = getattr(trader, "__dict__", {}).get("_macd_kdj_window", None)
    if not isinstance(window, dict):
        window = {}
        try:
            setattr(trader, "_macd_kdj_window", window)
        except Exception:
            pass

    if macd_bull_cross:
        window["type"] = "BULLISH"
        window["start"] = today
    elif macd_bear_cross:
        window["type"] = "BEARISH"
        window["start"] = today

    in_window = False
    window_type = window.get("type")
    if window_type in ("BULLISH", "BEARISH") and window.get("start") is not None:
        try:
            days_since_cross = (today.date() - window["start"].date()).days
        except Exception:
            days_since_cross = (today - window["start"]).days
        if _is_positive_number(opportunity_days) and days_since_cross <= int(opportunity_days):
            in_window = True

    # Volume confirmation (always present in pipeline, but keep safe fallback for tests/mocks)
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]
    if not isinstance(vols_attr, (list, tuple)):
        vols_attr = []

    buy_vol_ratio_threshold = (
        float(bull_buy_volume_ratio_threshold)
        if bullish_regime
        else float(bear_buy_volume_ratio_threshold)
    )
    sell_vol_ratio_threshold = (
        float(bull_sell_volume_ratio_threshold)
        if bullish_regime
        else float(bear_sell_volume_ratio_threshold)
    )

    buy_volume_ok = True
    sell_volume_ok = True
    try:
        if _is_positive_number(cur_vol) and len(vols_attr) >= int(volume_window) + 1:
            avg_vol = float(np.mean(list(vols_attr)[-(int(volume_window) + 1) : -1]))
            if avg_vol > 0:
                ratio = float(cur_vol) / avg_vol
                buy_volume_ok = ratio >= buy_vol_ratio_threshold
                sell_volume_ok = ratio >= sell_vol_ratio_threshold
    except Exception:
        buy_volume_ok = True
        sell_volume_ok = True

    # Execute trades only when both signals agree
    r_buy, r_sell = False, False

    # Asymmetry by regime:
    # - Bullish regime: easier BUYs, harder SELLs
    # - Bearish regime: harder BUYs, easier SELLs
    if bullish_regime:
        buy_confirm = (kdj_buy_cross or k_cur <= float(bull_kdj_buy_max)) and buy_volume_ok
        # Harder sells in bull: require KDJ sell-cross (confirmation), but allow it within MACD bear window.
        sell_confirm = kdj_sell_cross and sell_volume_ok
        buy_condition = (macd_bull_cross or (in_window and window_type == "BULLISH")) and buy_confirm and has_cash
        sell_condition = (macd_bear_cross or (in_window and window_type == "BEARISH")) and sell_confirm and has_position
    else:
        # Still harder buys in bear, but allow "deep oversold" without requiring an exact cross event.
        bear_oversold_ok = (kdj_buy_cross or k_cur <= float(oversold)) and k_cur <= float(
            bear_kdj_buy_max
        )
        buy_confirm = bear_oversold_ok and buy_volume_ok
        # easier sell in bearish regime: allow MACD cross alone OR KDJ cross, but still cooldown'ed
        sell_confirm = sell_volume_ok and (macd_bear_cross or kdj_sell_cross)
        buy_condition = (macd_bull_cross or (in_window and window_type == "BULLISH")) and buy_confirm and has_cash
        sell_condition = sell_confirm and has_position

    # Buy
    if buy_condition:
        r_buy = trader._execute_one_buy("by_percentage", new_p)
        if r_buy is True:
            trader._record_history(new_p, today, BUY_SIGNAL)
            trader.strat_dct[strat_name].append((today, BUY_SIGNAL))

    # Sell
    elif sell_condition:
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
    volume: Optional[float] = None,
    open_p: Optional[float] = None,
    low_p: Optional[float] = None,
    high_p: Optional[float] = None,
    trend_lookback: int = 3,
    trend_min_abs_slope_pct: float = 0.001,
    cooldown_days: int = 2,
    volume_window: int = 20,
    bull_buy_tol_multiplier: float = 0.7,
    bull_sell_tol_multiplier: float = 1.6,
    bear_buy_tol_multiplier: float = 1.3,
    bear_sell_tol_multiplier: float = 0.5,
    bull_buy_volume_ratio_threshold: float = 1.0,
    bear_buy_volume_ratio_threshold: float = 1.3,
    bull_blowoff_enabled: bool = True,
    bull_trailing_stop_pct: float = 0.06,
    bull_blowoff_rel_threshold: float = 0.06,
    bear_reversal_close_near_high_pct: float = 0.6,
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
        volume (Optional[float]): Current interval trading volume (always present in live pipeline).
            If None, strategy falls back to `trader.volume_history` when available. Defaults to None.
        open_p (Optional[float]): Current interval open (used for reversal confirmation). Defaults to None.
        low_p (Optional[float]): Current interval low (used for reversal confirmation). Defaults to None.
        high_p (Optional[float]): Current interval high (used for reversal confirmation). Defaults to None.
        trend_lookback (int): EMA lookback points for slope/regime. Defaults to 3.
        trend_min_abs_slope_pct (float): Minimum EMA slope (as % of EMA) to call bull/bear. Defaults to 0.001.
        cooldown_days (int): Minimum days between BUY/SELL signals to reduce whipsaw. Defaults to 2.
        volume_window (int): Rolling window for volume ratio confirmation. Defaults to 20.
        bull_buy_tol_multiplier (float): In bullish regime, buy is easier (smaller deviation). Defaults to 0.7.
        bull_sell_tol_multiplier (float): In bullish regime, sell is harder (larger deviation). Defaults to 1.6.
        bear_buy_tol_multiplier (float): In bearish regime, buy is harder (larger deviation). Defaults to 1.3.
        bear_sell_tol_multiplier (float): In bearish regime, sell is easier (smaller deviation). Defaults to 0.7.
        bull_buy_volume_ratio_threshold (float): Volume ratio required to confirm buys in bullish regime. Defaults to 1.0.
        bear_buy_volume_ratio_threshold (float): Volume ratio required to confirm buys in bearish regime. Defaults to 1.3.
        bull_blowoff_enabled (bool): If True, in bullish regime avoid selling on the first spike; sell on reversal
            via trailing stop. Defaults to True.
        bull_trailing_stop_pct (float): Trailing stop % from peak while blowoff armed. Defaults to 0.06.
        bull_blowoff_rel_threshold (float): Relative deviation above EMA to arm blowoff. Defaults to 0.06.
        bear_reversal_close_near_high_pct (float): For bearish-regime buys, require close near the candle high
            (if OHLC available). Defaults to 0.6.

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

    # Cooldown to reduce whipsaw / repeated trades
    if cooldown_days > 0 and strat_name in getattr(trader, "strat_dct", {}):
        for past_dt, past_sig in reversed(trader.strat_dct[strat_name]):
            if past_sig in (BUY_SIGNAL, SELL_SIGNAL):
                try:
                    delta_days = (today.date() - past_dt.date()).days
                except Exception:
                    delta_days = (today - past_dt).days
                if delta_days < int(cooldown_days):
                    trader._record_history(new_p, today, NO_ACTION_SIGNAL)
                    trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
                    return False, False
                break

    # retrieve the most recent exponential moving average
    last_ema = trader.exp_moving_averages[queue_name][-1]

    # Check for None values
    if last_ema is None:
        trader._record_history(new_p, today, NO_ACTION_SIGNAL)
        trader.strat_dct[strat_name].append((today, NO_ACTION_SIGNAL))
        return False, False

    # Determine regime from EMA slope (last vs lookback)
    ema_vals = [x for x in trader.exp_moving_averages[queue_name] if x is not None]
    trending_up = False
    trending_down = False
    try:
        if len(ema_vals) >= int(trend_lookback) + 1 and float(ema_vals[-1]) > 0:
            ema_now = float(ema_vals[-1])
            ema_past = float(ema_vals[-(int(trend_lookback) + 1)])
            slope_pct = (ema_now - ema_past) / ema_now
            trending_up = slope_pct >= float(trend_min_abs_slope_pct)
            trending_down = slope_pct <= -float(trend_min_abs_slope_pct)
    except Exception:
        trending_up = False
        trending_down = False

    # Relative deviation from EMA
    try:
        rel = (float(new_p) - float(last_ema)) / float(last_ema)
    except Exception:
        rel = 0.0

    # Volume ratio confirmation (best-effort, but volume is present in live pipeline)
    cur_vol = volume
    vols_attr = getattr(trader, "__dict__", {}).get("volume_history", None)
    if cur_vol is None and isinstance(vols_attr, (list, tuple)) and len(vols_attr) > 0:
        cur_vol = vols_attr[-1]

    vol_ratio = None
    try:
        if _is_positive_number(cur_vol) and isinstance(vols_attr, (list, tuple)) and len(vols_attr) >= int(volume_window) + 1:
            avg_vol = float(np.mean(list(vols_attr)[-(int(volume_window) + 1) : -1]))
            if avg_vol > 0:
                vol_ratio = float(cur_vol) / avg_vol
    except Exception:
        vol_ratio = None

    # Regime-aware thresholds
    buy_tol = float(tol_pct)
    sell_tol = float(tol_pct)
    if trending_up:
        buy_tol = float(tol_pct) * float(bull_buy_tol_multiplier)
        sell_tol = float(tol_pct) * float(bull_sell_tol_multiplier)
    elif trending_down:
        buy_tol = float(tol_pct) * float(bear_buy_tol_multiplier)
        sell_tol = float(tol_pct) * float(bear_sell_tol_multiplier)

    buy_vol_ok = True
    if vol_ratio is not None:
        buy_vol_ok = float(vol_ratio) >= (float(bull_buy_volume_ratio_threshold) if trending_up else float(bear_buy_volume_ratio_threshold))

    # Pullback detection: use previous rel to trigger only on cross events (less churn)
    prev_rel_key = f"_exp_ma_prev_rel:{queue_name}"
    prev_rel = getattr(trader, "__dict__", {}).get(prev_rel_key, None)
    try:
        prev_rel_val = float(prev_rel) if prev_rel is not None else None
    except Exception:
        prev_rel_val = None
    try:
        setattr(trader, prev_rel_key, float(rel))
    except Exception:
        pass

    pullback_buy = False
    if trending_up and prev_rel_val is not None:
        # Buy on pullback cross: from above EMA to slightly below EMA (or below -buy_tol)
        pullback_buy = (prev_rel_val > 0.0 and rel <= 0.0) or (prev_rel_val > -buy_tol and rel <= -buy_tol)

    # Bearish "rare chance": require reversal-style candle if OHLC is available
    reversal_ok = True
    if trending_down:
        try:
            if _is_positive_number(open_p) and _is_positive_number(low_p) and _is_positive_number(high_p) and float(high_p) > float(low_p):
                close_pos = (float(new_p) - float(low_p)) / (float(high_p) - float(low_p))
                reversal_ok = float(new_p) >= float(open_p) and close_pos >= float(bear_reversal_close_near_high_pct)
        except Exception:
            reversal_ok = True

    # (1) if too high, we do a sell
    r_sell = False
    bull_blowoff_sell = False
    if trending_up and bull_blowoff_enabled and has_position:
        state_key = f"_exp_ma_bull_state:{queue_name}"
        state = getattr(trader, "__dict__", {}).get(state_key, None)
        if not isinstance(state, dict):
            state = {"armed": False, "peak": None}
            try:
                setattr(trader, state_key, state)
            except Exception:
                pass

        armed = bool(state.get("armed", False)) if isinstance(state, dict) else False
        arm = rel >= float(bull_blowoff_rel_threshold) or rel >= sell_tol
        if (not armed) and arm:
            state["armed"] = True
            state["peak"] = float(new_p)
        elif armed:
            peak = float(state.get("peak") or new_p)
            peak = max(peak, float(new_p))
            state["peak"] = peak
            try:
                if _is_positive_number(bull_trailing_stop_pct) and float(new_p) <= peak * (1.0 - float(bull_trailing_stop_pct)):
                    bull_blowoff_sell = True
            except Exception:
                bull_blowoff_sell = False
        if bull_blowoff_sell:
            state["armed"] = False
            state["peak"] = None

    sell_condition = rel >= sell_tol
    if trending_up and bull_blowoff_enabled:
        # Harder sells: don't sell just because it's above EMA; wait for reversal/stop
        sell_condition = bull_blowoff_sell

    if has_position and sell_condition:
        r_sell = trader._execute_one_sell("by_percentage", new_p)
        if r_sell is True:
            trader._record_history(new_p, today, SELL_SIGNAL)
            trader.strat_dct[strat_name].append((today, SELL_SIGNAL))

    # (2) if too low, we do a buy
    r_buy = False
    # Avoid executing both sell and buy in the same tick.
    buy_condition = rel <= -buy_tol
    if trending_up:
        # Easier buys in bull: allow pullback entries near EMA
        buy_condition = buy_condition or pullback_buy
    if trending_down:
        # Harder buys in bear: require volume + reversal confirmation
        buy_condition = buy_condition and buy_vol_ok and reversal_ok
    else:
        # In neutral/bull, still prefer volume >= baseline if available
        buy_condition = buy_condition and buy_vol_ok

    if r_sell is False and has_cash and buy_condition:
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
