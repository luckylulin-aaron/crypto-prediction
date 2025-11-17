# built-in packages
import datetime
import functools
import math
import time
from typing import Any, Dict, List, Tuple

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

    # Create moving windows
    windows = create_moving_windows(data_stream, window_size, step_size)
    
    if not windows:
        raise ValueError("No windows created from data stream")
    
    # Store results for each window
    window_results = []
    strategy_performances = defaultdict(list)  # strategy -> list of metrics
    
    for window_idx, window_data in enumerate(windows):
        # Create a new TraderDriver for this window
        driver = trader_driver_class(**trader_driver_kwargs)
        driver.feed_data(window_data)
        
        # Get best trader info for this window
        best_info = driver.best_trader_info
        best_trader = driver.traders[best_info["trader_index"]]
        
        # Extract metrics
        rate_of_return_str = best_info.get("rate_of_return", "0%")
        rate_of_return_float = float(str(rate_of_return_str).replace("%", ""))
        
        window_result = {
            "window_idx": window_idx,
            "window_start_date": window_data[0][1] if window_data else None,
            "window_end_date": window_data[-1][1] if window_data else None,
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
    
    # Find best strategy based on mean rate of return (can be changed to risk_adjusted_return)
    best_strategy_name = max(
        aggregated_strategies.keys(),
        key=lambda s: aggregated_strategies[s]["mean_rate_of_return"],
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
    }
