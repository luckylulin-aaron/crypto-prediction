# built-in packages
import datetime
import functools
import math
import time
from typing import Any, Dict, List, Tuple

# third-party packages
import numpy as np
import pandas as pd

try:
    from core.logger import get_logger
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.logger import get_logger

logger = get_logger(__name__)


def compute_option_signal_win_rates(
    trade_history: List[Dict[str, Any]],
    hold_days: int,
) -> Dict[str, Any]:
    """
    Compute win rates for "options opened on signals" using historical BUY/SELL events.

    Heuristic definition:
    - Call (opened on BUY): find the first SELL within `hold_days` after the BUY.
      It's a win if sell_price > buy_price.
    - Put (opened on SELL): find the first BUY within `hold_days` after the SELL.
      It's a win if buy_price < sell_price.

    Notes:
    - We only count trials where a matching exit signal exists within the horizon.
      (No exit within horizon => not counted, to avoid arbitrary assumptions.)

    Args:
        trade_history (List[Dict[str, Any]]): Trader's full trade history (events with at least
            `action`, `price`, `date`).
        hold_days (int): Maximum holding horizon in days.

    Returns:
        Dict[str, Any]: {
            "call_win_rate_pct": float,
            "put_win_rate_pct": float,
            "call_trials": int,
            "put_trials": int,
            "call_wins": int,
            "put_wins": int,
        }

    Raises:
        ValueError: If hold_days is negative.
    """
    if hold_days < 0:
        raise ValueError("hold_days must be >= 0")

    horizon_seconds = float(hold_days) * 86400.0

    # Normalize + filter to BUY/SELL only, preserve order
    events: List[Dict[str, Any]] = []
    for evt in (trade_history or []):
        try:
            act = str(evt.get("action", "")).upper()
        except Exception:
            continue
        if act not in ("BUY", "SELL"):
            continue

        d0 = evt.get("date")
        dt0 = None
        if isinstance(d0, datetime.datetime):
            dt0 = d0
        else:
            # best-effort parsing (supports stored datetime or string)
            try:
                dt0 = datetime.datetime.fromisoformat(str(d0).replace("Z", "+00:00"))
            except Exception:
                try:
                    dt0 = datetime.datetime.strptime(str(d0), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt0 = None
        if dt0 is None:
            continue

        try:
            p0 = float(evt.get("price"))
        except Exception:
            continue

        events.append({"action": act, "date": dt0, "price": p0})

    call_trials = call_wins = 0
    put_trials = put_wins = 0

    for i, e in enumerate(events):
        if e["action"] == "BUY":
            # find first SELL within horizon
            for j in range(i + 1, len(events)):
                nxt = events[j]
                if (nxt["date"] - e["date"]).total_seconds() > horizon_seconds:
                    break
                if nxt["action"] == "SELL":
                    call_trials += 1
                    if nxt["price"] > e["price"]:
                        call_wins += 1
                    break
        elif e["action"] == "SELL":
            # find first BUY within horizon
            for j in range(i + 1, len(events)):
                nxt = events[j]
                if (nxt["date"] - e["date"]).total_seconds() > horizon_seconds:
                    break
                if nxt["action"] == "BUY":
                    put_trials += 1
                    if nxt["price"] < e["price"]:
                        put_wins += 1
                    break

    call_win_rate_pct = 100.0 * call_wins / call_trials if call_trials > 0 else 0.0
    put_win_rate_pct = 100.0 * put_wins / put_trials if put_trials > 0 else 0.0

    return {
        "call_win_rate_pct": round(call_win_rate_pct, 2),
        "put_win_rate_pct": round(put_win_rate_pct, 2),
        "call_trials": int(call_trials),
        "put_trials": int(put_trials),
        "call_wins": int(call_wins),
        "put_wins": int(put_wins),
    }


def timer(func: Any) -> Any:
    """
    Decorator to print the runtime of the decorated function.

    Args:
        func (Any): The function to be decorated.

    Returns:
        Any: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        # do something...
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(
            "--> executed={} using time={:.2f} seconds.".format(func.__name__, run_time)
        )
        return value

    return wrapper_timer


def load_csv(csv_fn="../models/BTC_HISTORY.csv") -> tuple:
    """
    Load a csv file and return dates and prices lists.

    Args:
        csv_fn (str): Path to the CSV file.

    Returns:
        tuple: (name (str), l (List[Any]))
    """
    df = pd.read_csv(csv_fn)
    dates, prices = df["Date"].tolist(), df["Closing Price (USD)"].tolist()

    name, l = csv_fn.split(".")[0].split("_")[0], []
    for ind in range(len(dates)):
        l.append([prices[ind], dates[ind]])

    return name, l


def display_port_msg(v_c: float, v_s: float, before: bool = True) -> None:
    """
    Display portfolio message in a certain format.

    Args:
        v_c (float): Crypto value.
        v_s (float): Stablecoin value.
        before (bool, optional): Whether this is before or after a transaction. Defaults to True.

    Returns:
        None
    """
    now = datetime.datetime.now()
    stage = "before" if before else "after"
    s = v_c + v_s

    print(
        "\n{} transaction, by {}, crypto_value={:.2f}, stablecoin_value={:.2f}, amount to {:.2f}".format(
            stage, now, v_c, v_s, s
        )
    )


def max_drawdown_helper(hist_l: List[float]) -> float:
    """
    Compute max drawdown (最大回撤) for a given portfolio value history.

    Args:
        hist_l (List[float]): Portfolio value history.

    Returns:
        float: Computed maximum drawdown in percentage to the total portfolio value.
    """
    if not hist_l or len(hist_l) < 2:
        return 0.0
    
    res = -math.inf
    value_cur, value_max_pre = hist_l[0], hist_l[0]
    for hist in hist_l[1:]:
        value_max_pre = max(value_max_pre, hist)
        value_cur = hist
        
        # Handle division by zero
        if value_max_pre == 0:
            if value_cur == 0:
                continue  # Skip if both are zero (no change)
            else:
                res = max(res, 1.0)  # 100% drawdown if max was zero and current is not
        else:
            res = max(res, 1 - value_cur / value_max_pre)

    return np.round(res, 4)


def ema_helper(new_price: float, old_ema: float, num_of_days: int) -> float:
    """
    Helper function to compute a new EMA.

    Args:
        new_price (float): Today's closing price.
        old_ema (float): Previous EMA value.
        num_of_days (int): Number of days for smoothing.

    Returns:
        float: The new EMA value.
    """
    k = 2 / (num_of_days + 1)
    return (new_price * k) + (old_ema * (1 - k))


def calculate_simulation_amounts(
    actual_cash: float,
    actual_coin: float,
    method: str = "PORTFOLIO_SCALED",
    base_amount: float = 10000,
    percentage: float = 0.1,
) -> tuple[float, float]:
    """
    Calculate simulation amounts using different methods to eliminate bias.

    Args:
        actual_cash: Actual cash amount in portfolio
        actual_coin: Actual coin amount in portfolio
        method: Simulation method ("FIXED", "PORTFOLIO_SCALED", "PERCENTAGE_BASED")
        base_amount: Base amount for scaling (default: 10000)
        percentage: Percentage of portfolio to use (default: 0.1 = 10%)

    Returns:
        tuple: (sim_cash, sim_coin) - Simulation amounts
    """
    if method == "FIXED":
        # Legacy method - fixed amounts (biased)
        return 3000, 5

    elif method == "PORTFOLIO_SCALED":
        # Scale actual portfolio to standard amount
        actual_portfolio_value = actual_cash + (
            actual_coin * 1
        )  # Assuming $1 per coin for scaling
        if actual_portfolio_value <= 0:
            return base_amount, base_amount * 0.001  # Default if no portfolio

        simulation_ratio = base_amount / actual_portfolio_value
        sim_cash = actual_cash * simulation_ratio
        sim_coin = actual_coin * simulation_ratio
        return sim_cash, sim_coin

    elif method == "PERCENTAGE_BASED":
        # Use fixed percentage of actual portfolio
        sim_cash = actual_cash * percentage
        sim_coin = actual_coin * percentage
        return sim_cash, sim_coin

    else:
        raise ValueError(f"Unknown simulation method: {method}")


def create_moving_windows(
    data_stream: List[Tuple], window_size: int, step_size: int = 1
) -> List[List[Tuple]]:
    """
    Create overlapping moving windows from historical data.

    Args:
        data_stream: List of tuples containing (price, date, open, low, high, volume)
        window_size: Size of each window in number of data points
        step_size: Step size for moving window (1 = overlapping, window_size = non-overlapping)

    Returns:
        List of windows, where each window is a list of data tuples
    """
    if not data_stream:
        return []
    
    if window_size > len(data_stream):
        # If window size is larger than data, return single window with all data
        return [data_stream]
    
    windows = []
    for i in range(0, len(data_stream) - window_size + 1, step_size):
        window = data_stream[i : i + window_size]
        windows.append(window)
    
    return windows


def run_moving_window_simulation(
    trader_driver_class,
    data_stream: List[Tuple],
    window_size: int,
    step_size: int = 1,
    **trader_driver_kwargs,
) -> Dict[str, Any]:
    """
    Run simulation using moving windows and aggregate results.

    Args:
        trader_driver_class: TraderDriver class to use for simulation
        data_stream: Full historical data stream
        window_size: Size of each moving window
        step_size: Step size for moving window
        **trader_driver_kwargs: Keyword arguments to pass to TraderDriver

    Returns:
        Dictionary containing aggregated performance metrics and best strategy info
    """
    from collections import defaultdict
    try:
        # Preferred when running as a package
        from app.core.config import (
            MOVING_WINDOW_BEST_STRATEGY_METRIC,
            MOVING_WINDOW_BEST_STRATEGY_MIN_WINS,
            MOVING_WINDOW_AUTO_TUNE_PARAMS,
            MOVING_WINDOW_TUNE_NEIGHBORHOOD,
        )
    except Exception:
        try:
            # Fallback used by some scripts/tests
            from core.config import (
                MOVING_WINDOW_BEST_STRATEGY_METRIC,
                MOVING_WINDOW_BEST_STRATEGY_MIN_WINS,
                MOVING_WINDOW_AUTO_TUNE_PARAMS,
                MOVING_WINDOW_TUNE_NEIGHBORHOOD,
            )
        except Exception:
            MOVING_WINDOW_BEST_STRATEGY_METRIC = "risk_adjusted_return"
            MOVING_WINDOW_BEST_STRATEGY_MIN_WINS = 2
            MOVING_WINDOW_AUTO_TUNE_PARAMS = False
            MOVING_WINDOW_TUNE_NEIGHBORHOOD = 1

    def _neighbors(values, target, k: int):
        """Return a small neighborhood around target in an ordered list of values."""
        if not values:
            return []
        try:
            vals = list(values)
            vals_sorted = sorted(vals)
            idx = vals_sorted.index(target) if target in vals_sorted else None
            if idx is None:
                return [target]
            lo = max(0, idx - int(k))
            hi = min(len(vals_sorted), idx + int(k) + 1)
            return vals_sorted[lo:hi]
        except Exception:
            return [target]

    def _best_params_by_strategy(driver) -> Dict[str, dict]:
        """Pick the best trader (by final portfolio value) per strategy and return its params."""
        best = {}
        for t in getattr(driver, "traders", []) or []:
            s = getattr(t, "high_strategy", None)
            if not s:
                continue
            pv = getattr(t, "portfolio_value", None)
            try:
                score = float(pv)
            except Exception:
                continue
            if s not in best or score > best[s]["score"]:
                best[s] = {
                    "score": score,
                    "tol_pct": getattr(t, "tol_pct", None),
                    "buy_pct": getattr(t, "buy_pct", None),
                    "sell_pct": getattr(t, "sell_pct", None),
                    "bollinger_sigma": getattr(t, "bollinger_sigma", None),
                    "rsi_period": getattr(t, "rsi_period", None),
                    "rsi_oversold": getattr(t, "rsi_oversold", None),
                    "rsi_overbought": getattr(t, "rsi_overbought", None),
                    "kdj_oversold": getattr(t, "kdj_oversold", None),
                    "kdj_overbought": getattr(t, "kdj_overbought", None),
                }
        # remove score field
        for s in list(best.keys()):
            best[s].pop("score", None)
        return best

    def _fmt_list(values: list, max_items: int = 10) -> str:
        """Format a list for logs with truncation."""
        try:
            vals = list(values)
        except Exception:
            return str(values)
        if len(vals) <= max_items:
            return str(vals)
        head = vals[:max_items]
        return f"{head} ... (+{len(vals) - max_items} more)"

    # Log interval configuration details
    asset_name = trader_driver_kwargs.get("name", "UNKNOWN")
    logger.info(f"[{asset_name}] Moving window simulation configuration:")
    logger.info(f"  - Total data points: {len(data_stream)}")
    logger.info(f"  - Window size: {window_size} data points")
    logger.info(f"  - Step size: {step_size} data points")
    if len(data_stream) > 0:
        # Try to calculate time span of data
        try:
            start_date = data_stream[0][1]
            end_date = data_stream[-1][1]
            logger.info(f"  - Data range: {start_date} to {end_date}")
            # Calculate approximate interval between data points
            if len(data_stream) > 1:
                if isinstance(start_date, str):
                    from datetime import datetime as dt
                    try:
                        start_dt = dt.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                        end_dt = dt.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                        total_hours = (end_dt - start_dt).total_seconds() / 3600
                        avg_interval_hours = total_hours / (len(data_stream) - 1)
                        logger.info(f"  - Average interval between data points: ~{avg_interval_hours:.1f} hours")
                    except:
                        pass
        except:
            pass

    # Create moving windows
    windows = create_moving_windows(data_stream, window_size, step_size)
    
    if not windows:
        raise ValueError("No windows created from data stream")
    
    logger.info(f"[{asset_name}] Created {len(windows)} windows for processing")
    
    # Store results for each window
    window_results = []
    strategy_performances = defaultdict(list)  # strategy -> list of metrics
    window_chart_data = []  # per-window price + buy/sell markers for visualization

    # Walk-forward parameter tuning: keep last best parameters per strategy and narrow the next window's grid.
    last_best_params: Dict[str, dict] = {}
    
    for window_idx, window_data in enumerate(windows):
        window_start_time = time.perf_counter()
        
        # Log window processing details
        window_start_date = window_data[0][1] if window_data else None
        window_end_date = window_data[-1][1] if window_data else None
        logger.info(f"[{asset_name}] Processing window {window_idx + 1}/{len(windows)}: "
                    f"{window_start_date} to {window_end_date} ({len(window_data)} data points)")
        
        # Create a new TraderDriver for this window.
        # If auto-tuning is enabled, narrow the search space around last best parameters (no leakage).
        td_kwargs = dict(trader_driver_kwargs)
        if MOVING_WINDOW_AUTO_TUNE_PARAMS and window_idx > 0 and last_best_params:
            k = int(MOVING_WINDOW_TUNE_NEIGHBORHOOD)
            # Base lists (full search space)
            tol_pcts = list(td_kwargs.get("tol_pcts", []))
            buy_pcts = list(td_kwargs.get("buy_pcts", []))
            sell_pcts = list(td_kwargs.get("sell_pcts", []))
            boll_tols = list(td_kwargs.get("bollinger_tols", []))
            rsi_periods = list(td_kwargs.get("rsi_periods", []))
            rsi_os = list(td_kwargs.get("rsi_oversold_thresholds", []))
            rsi_ob = list(td_kwargs.get("rsi_overbought_thresholds", []))
            kdj_os = list(td_kwargs.get("kdj_oversold_thresholds", []))
            kdj_ob = list(td_kwargs.get("kdj_overbought_thresholds", []))

            # Union of neighborhoods around each strategy's last best params (still small).
            tuned_tol = set()
            tuned_buy = set()
            tuned_sell = set()
            tuned_boll = set()
            tuned_rsi_p = set()
            tuned_rsi_os = set()
            tuned_rsi_ob = set()
            tuned_kdj_os = set()
            tuned_kdj_ob = set()

            for s, p in last_best_params.items():
                if p.get("tol_pct") is not None:
                    tuned_tol.update(_neighbors(tol_pcts, p["tol_pct"], k))
                if p.get("buy_pct") is not None:
                    tuned_buy.update(_neighbors(buy_pcts, p["buy_pct"], k))
                if p.get("sell_pct") is not None:
                    tuned_sell.update(_neighbors(sell_pcts, p["sell_pct"], k))
                if p.get("bollinger_sigma") is not None:
                    tuned_boll.update(_neighbors(boll_tols, p["bollinger_sigma"], k))
                if p.get("rsi_period") is not None:
                    tuned_rsi_p.update(_neighbors(rsi_periods, p["rsi_period"], k))
                if p.get("rsi_oversold") is not None:
                    tuned_rsi_os.update(_neighbors(rsi_os, p["rsi_oversold"], k))
                if p.get("rsi_overbought") is not None:
                    tuned_rsi_ob.update(_neighbors(rsi_ob, p["rsi_overbought"], k))
                if p.get("kdj_oversold") is not None:
                    tuned_kdj_os.update(_neighbors(kdj_os, p["kdj_oversold"], k))
                if p.get("kdj_overbought") is not None:
                    tuned_kdj_ob.update(_neighbors(kdj_ob, p["kdj_overbought"], k))

            if tuned_tol:
                td_kwargs["tol_pcts"] = sorted(tuned_tol)
            if tuned_buy:
                td_kwargs["buy_pcts"] = sorted(tuned_buy)
            if tuned_sell:
                td_kwargs["sell_pcts"] = sorted(tuned_sell)
            if tuned_boll:
                td_kwargs["bollinger_tols"] = sorted(tuned_boll)
            if tuned_rsi_p:
                td_kwargs["rsi_periods"] = sorted(tuned_rsi_p)
            if tuned_rsi_os:
                td_kwargs["rsi_oversold_thresholds"] = sorted(tuned_rsi_os)
            if tuned_rsi_ob:
                td_kwargs["rsi_overbought_thresholds"] = sorted(tuned_rsi_ob)
            if tuned_kdj_os:
                td_kwargs["kdj_oversold_thresholds"] = sorted(tuned_kdj_os)
            if tuned_kdj_ob:
                td_kwargs["kdj_overbought_thresholds"] = sorted(tuned_kdj_ob)

            logger.info(
                f"[{asset_name}] Auto-tuning enabled: narrowed search space for window {window_idx + 1} "
                f"(neighborhood=±{k})"
            )
            logger.info(
                f"[{asset_name}] Tuned grids: "
                f"tol_pcts={_fmt_list(td_kwargs.get('tol_pcts', []))}, "
                f"buy_pcts={_fmt_list(td_kwargs.get('buy_pcts', []))}, "
                f"sell_pcts={_fmt_list(td_kwargs.get('sell_pcts', []))}, "
                f"bollinger_tols={_fmt_list(td_kwargs.get('bollinger_tols', []))}"
            )
            # Only log RSI/KDJ grids if those strategies are enabled in this run
            try:
                enabled_stats = td_kwargs.get("overall_stats", []) or []
            except Exception:
                enabled_stats = []
            if "RSI" in enabled_stats:
                logger.info(
                    f"[{asset_name}] Tuned RSI grids: "
                    f"periods={_fmt_list(td_kwargs.get('rsi_periods', []))}, "
                    f"oversold={_fmt_list(td_kwargs.get('rsi_oversold_thresholds', []))}, "
                    f"overbought={_fmt_list(td_kwargs.get('rsi_overbought_thresholds', []))}"
                )
            if "KDJ" in enabled_stats:
                logger.info(
                    f"[{asset_name}] Tuned KDJ grids: "
                    f"oversold={_fmt_list(td_kwargs.get('kdj_oversold_thresholds', []))}, "
                    f"overbought={_fmt_list(td_kwargs.get('kdj_overbought_thresholds', []))}"
                )

        driver = trader_driver_class(**td_kwargs)
        driver.feed_data(window_data)
        
        window_process_time = time.perf_counter() - window_start_time
        
        # Get best trader info for this window
        best_info = driver.best_trader_info
        best_trader = driver.traders[best_info["trader_index"]]
        
        # Extract metrics
        rate_of_return_str = best_info.get("rate_of_return", "0%")
        rate_of_return_float = float(str(rate_of_return_str).replace("%", ""))
        
        # Get baseline return (asset uplift %)
        baseline_return = best_trader.baseline_rate_of_return
        
        # Get transaction details (dates and actions) - limit to top 20 most recent
        trade_history = best_trader.all_history_trade_only
        transaction_details = []
        
        # Sort transactions by date (handle both datetime objects and strings)
        def get_trade_date(trade):
            trade_date = trade.get("date", "")
            if isinstance(trade_date, datetime.datetime):
                return trade_date
            elif isinstance(trade_date, str):
                # Try to parse common date formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
                    try:
                        return datetime.datetime.strptime(trade_date, fmt)
                    except:
                        continue
            return datetime.datetime.min  # Fallback for unparseable dates
        
        # Sort by date and take the most recent 20
        sorted_trades = sorted(trade_history, key=get_trade_date)
        recent_trades = sorted_trades[-20:] if len(sorted_trades) > 20 else sorted_trades
        
        for trade in recent_trades:
            trade_date = trade.get("date", "")
            trade_action = trade.get("action", "")
            # Format date if it's a datetime object
            if isinstance(trade_date, datetime.datetime):
                trade_date_str = trade_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                trade_date_str = str(trade_date)
            transaction_details.append(f"{trade_date_str}:{trade_action}")
        
        # Format transaction details string
        if transaction_details:
            transactions_str = " | ".join(transaction_details)
            if len(sorted_trades) > 20:
                transactions_str += f" (showing 20 of {len(sorted_trades)} total)"
        else:
            transactions_str = "No transactions"
        
        logger.info(f"[{asset_name}] Window {window_idx + 1} completed in {window_process_time:.2f}s - "
                    f"Best strategy: {best_trader.high_strategy}, "
                    f"Return: {rate_of_return_float:.2f}%, "
                    f"Baseline return: {baseline_return:.2f}%, "
                    f"Transactions: {best_trader.num_transaction} | "
                    f"Transaction details: {transactions_str}")

        # Prepare per-window chart data (price series + buy/sell markers)
        # Deduplicate actions by timestamp to avoid multiple markers on the same date.
        action_priority = {"SELL": 2, "BUY": 1}
        date_to_trade = {}
        for evt in getattr(best_trader, "trade_history", []) or []:
            action = evt.get("action")
            if action not in ("BUY", "SELL"):
                continue
            d = evt.get("date")
            key = str(d)
            prev = date_to_trade.get(key)
            if prev is None or action_priority[action] > action_priority.get(prev.get("action"), 0):
                date_to_trade[key] = {"date": d, "action": action, "price": evt.get("price")}

        dedup_trade_history = list(date_to_trade.values())

        window_chart_data.append(
            {
                "window_idx": window_idx,
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "window_size": len(window_data),
                "best_strategy": best_trader.high_strategy,
                "rate_of_return": rate_of_return_float,
                "baseline_rate_of_return": float(baseline_return),
                "num_transactions": best_trader.num_transaction,
                "crypto_prices": getattr(best_trader, "crypto_prices", window_data),
                "trade_history": dedup_trade_history,
            }
        )
        
        window_result = {
            "window_idx": window_idx,
            "window_start_date": window_start_date,
            "window_end_date": window_end_date,
            "window_size": len(window_data),
            "best_strategy": best_trader.high_strategy,
            "rate_of_return": rate_of_return_float,
            "baseline_rate_of_return": float(
                str(best_info.get("baseline_rate_of_return", "0%")).replace("%", "")
            ),
            "coin_rate_of_return": float(
                str(best_info.get("coin_rate_of_return", "0%")).replace("%", "")
            ),
            "max_drawdown": best_trader.max_drawdown * 100,
            "num_transactions": best_trader.num_transaction,
            "num_buys": best_trader.num_buy_action,
            "num_sells": best_trader.num_sell_action,
            "final_value": best_info.get("max_final_value", 0),
            "init_value": best_info.get("init_value", 0),
            "trader_index": best_info.get("trader_index", -1),
        }
        
        window_results.append(window_result)

        # Update last_best_params for next window (learned from this window only; no future leakage).
        try:
            last_best_params = _best_params_by_strategy(driver)
        except Exception:
            last_best_params = last_best_params or {}
        
        # Aggregate by strategy
        strategy_key = best_trader.high_strategy
        strategy_performances[strategy_key].append({
            "rate_of_return": rate_of_return_float,
            "baseline_rate_of_return": window_result["baseline_rate_of_return"],
            "coin_rate_of_return": window_result["coin_rate_of_return"],
            "max_drawdown": window_result["max_drawdown"],
            "num_transactions": window_result["num_transactions"],
            "final_value": window_result["final_value"],
            "init_value": window_result["init_value"],
        })
    
    # Compute aggregated statistics for each strategy
    aggregated_strategies = {}
    for strategy, performances in strategy_performances.items():
        rates_of_return = [p["rate_of_return"] for p in performances]
        baseline_rates = [p["baseline_rate_of_return"] for p in performances]
        coin_rates = [p["coin_rate_of_return"] for p in performances]
        drawdowns = [p["max_drawdown"] for p in performances]
        transactions = [p["num_transactions"] for p in performances]
        
        aggregated_strategies[strategy] = {
            "mean_rate_of_return": np.mean(rates_of_return),
            "std_rate_of_return": np.std(rates_of_return),
            "min_rate_of_return": np.min(rates_of_return),
            "max_rate_of_return": np.max(rates_of_return),
            "median_rate_of_return": np.median(rates_of_return),
            "mean_baseline_rate": np.mean(baseline_rates),
            "mean_coin_rate": np.mean(coin_rates),
            "mean_drawdown": np.mean(drawdowns),
            "mean_transactions": np.mean(transactions),
            "num_windows": len(performances),
            "win_rate": sum(1 for r in rates_of_return if r > 0) / len(rates_of_return) if rates_of_return else 0,
            # Risk-adjusted return (mean - std for conservative estimate)
            "risk_adjusted_return": np.mean(rates_of_return) - np.std(rates_of_return) if rates_of_return else 0,
        }
    
    # Find best strategy across windows.
    # Default: risk_adjusted_return (mean - std) with a minimum-win-count guard.
    metric = str(MOVING_WINDOW_BEST_STRATEGY_METRIC or "risk_adjusted_return")
    min_wins = int(MOVING_WINDOW_BEST_STRATEGY_MIN_WINS or 0)

    eligible = {
        s: m for s, m in aggregated_strategies.items() if int(m.get("num_windows", 0)) >= min_wins
    }
    if not eligible:
        eligible = aggregated_strategies

    def _score(s: str):
        m = eligible[s]
        primary = float(m.get(metric, m.get("risk_adjusted_return", m.get("mean_rate_of_return", 0.0))))
        # tie-breaker: prefer higher mean return
        secondary = float(m.get("mean_rate_of_return", 0.0))
        return (primary, secondary)

    best_strategy_name = max(eligible.keys(), key=_score)
    logger.info(
        f"[{asset_name}] Aggregated best strategy selection: metric={metric}, "
        f"min_wins={min_wins}, best={best_strategy_name}"
    )
    
    # Get the best window result for the best strategy
    best_window_results = [
        r for r in window_results if r["best_strategy"] == best_strategy_name
    ]
    best_window_result = max(
        best_window_results, key=lambda r: r["rate_of_return"]
    ) if best_window_results else window_results[0]
    
    return {
        "num_windows": len(windows),
        "window_size": window_size,
        "step_size": step_size,
        "best_strategy": best_strategy_name,
        "aggregated_strategies": aggregated_strategies,
        "best_strategy_metrics": aggregated_strategies[best_strategy_name],
        "window_results": window_results,
        "best_window_result": best_window_result,
        "window_chart_data": window_chart_data,
    }
