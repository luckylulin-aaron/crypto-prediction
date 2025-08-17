# built-in packages
import datetime
import functools
import math
import time
from typing import Any, List

# third-party packages
import numpy as np
import pandas as pd


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
