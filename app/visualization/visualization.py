"""
Visualization functions for cryptocurrency trading bot simulation results.
Uses Plotly for interactive charts.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_portfolio_value_chart(
    trade_history: List[Dict],
    title: str = "Portfolio Value Over Time with Buy/Sell Signals",
    execution_strategies: str = "",
) -> go.Figure:
    """
    Create an interactive portfolio value chart with buy/sell markers.

    Args:
        trade_history: List of trade history dictionaries from MATrader
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not trade_history:
        return go.Figure()

    # Convert to DataFrame
    df = pd.DataFrame(trade_history)
    # Handle multiple date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Create figure
    fig = go.Figure()

    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["portfolio"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Portfolio:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # Add buy signals
    buy_signals = df[df["action"] == "BUY"]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals["date"],
                y=buy_signals["portfolio"],
                mode="markers",
                name="Buy Signal",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                hovertemplate="<b>BUY</b><br><b>Date:</b> %{x}<br><b>Portfolio:</b> $%{y:,.2f}<br><b>Price:</b> $%{text}<extra></extra>",
                text=buy_signals["price"].round(2),
            )
        )

    # Add sell signals
    sell_signals = df[df["action"] == "SELL"]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals["date"],
                y=sell_signals["portfolio"],
                mode="markers",
                name="Sell Signal",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                hovertemplate="<b>SELL</b><br><b>Date:</b> %{x}<br><b>Portfolio:</b> $%{y:,.2f}<br><b>Price:</b> $%{text}<extra></extra>",
                text=sell_signals["price"].round(2),
            )
        )

    # Update layout
    if execution_strategies:
        full_title = f"{title}<br><sub>{execution_strategies}</sub>"
    else:
        full_title = title

    fig.update_layout(
        title=full_title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_asset_allocation_chart(
    trade_history: List[Dict], title: str = "Asset Allocation Over Time"
) -> go.Figure:
    """
    Create a stacked area chart showing crypto vs stablecoin allocation.

    Args:
        trade_history: List of trade history dictionaries
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not trade_history:
        return go.Figure()

    df = pd.DataFrame(trade_history)
    # Handle multiple date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Calculate crypto and cash values
    df["crypto_value"] = df["coin"] * df["price"]
    df["cash_value"] = df["cash"]

    fig = go.Figure()

    # Add crypto allocation
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["crypto_value"],
            mode="lines",
            fill="tonexty",
            name="Crypto Value",
            line=dict(color="orange", width=0),
            hovertemplate="<b>Date:</b> %{x}<br><b>Crypto:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # Add cash allocation
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cash_value"],
            mode="lines",
            fill="tonexty",
            name="Cash Value",
            line=dict(color="lightblue", width=0),
            hovertemplate="<b>Date:</b> %{x}<br><b>Cash:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_drawdown_chart(
    trade_history: List[Dict], title: str = "Drawdown Over Time"
) -> go.Figure:
    """
    Create a drawdown chart showing portfolio decline from peak.

    Args:
        trade_history: List of trade history dictionaries
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not trade_history:
        return go.Figure()

    df = pd.DataFrame(trade_history)
    # Handle multiple date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Calculate running maximum and drawdown
    df["running_max"] = df["portfolio"].expanding().max()
    df["drawdown"] = (df["portfolio"] - df["running_max"]) / df["running_max"] * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            mode="lines",
            fill="tonexty",
            name="Drawdown %",
            line=dict(color="red", width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(range=[df["drawdown"].min() - 5, 5]),
    )

    return fig


def create_price_and_signals_chart(
    crypto_prices: List[Tuple],
    trade_history: List[Dict],
    title: str = "Price History with Trading Signals",
) -> go.Figure:
    """
    Create a candlestick-like chart with price and trading signals.

    Args:
        crypto_prices: List of (price, date, open, low, high) tuples
        trade_history: List of trade history dictionaries
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not crypto_prices:
        return go.Figure()

    # Convert crypto_prices to DataFrame
    price_df = pd.DataFrame(
        crypto_prices, columns=["price", "date", "open", "low", "high"]
    )
    # Handle multiple date formats
    price_df["date"] = pd.to_datetime(price_df["date"], format="mixed", dayfirst=False)

    # Convert trade history to DataFrame
    trade_df = pd.DataFrame(trade_history)
    if not trade_df.empty:
        # Handle multiple date formats
        trade_df["date"] = pd.to_datetime(
            trade_df["date"], format="mixed", dayfirst=False
        )

    fig = go.Figure()

    # Add price line
    fig.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["price"],
            mode="lines",
            name="Price",
            line=dict(color="black", width=1),
            hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>",
        )
    )

    # Add buy signals
    if not trade_df.empty:
        buy_signals = trade_df[trade_df["action"] == "BUY"]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["date"],
                    y=buy_signals["price"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", size=12, symbol="triangle-up"),
                    hovertemplate="<b>BUY</b><br><b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>",
                )
            )

        # Add sell signals
        sell_signals = trade_df[trade_df["action"] == "SELL"]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["date"],
                    y=sell_signals["price"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", size=12, symbol="triangle-down"),
                    hovertemplate="<b>SELL</b><br><b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_performance_comparison_chart(
    trade_history: List[Dict],
    baseline_prices: List[Tuple],
    title: str = "Strategy Performance vs Buy & Hold",
) -> go.Figure:
    """
    Create a performance comparison chart between strategy and buy & hold.

    Args:
        trade_history: List of trade history dictionaries
        baseline_prices: List of (price, date) tuples for baseline calculation
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not trade_history or not baseline_prices:
        return go.Figure()

    # Strategy performance
    df_strategy = pd.DataFrame(trade_history)
    # Handle multiple date formats
    df_strategy["date"] = pd.to_datetime(
        df_strategy["date"], format="mixed", dayfirst=False
    )

    # Baseline performance (buy & hold)
    df_baseline = pd.DataFrame(baseline_prices, columns=["price", "date"])
    # Handle multiple date formats
    df_baseline["date"] = pd.to_datetime(
        df_baseline["date"], format="mixed", dayfirst=False
    )

    # Calculate baseline portfolio value (assuming same initial investment)
    initial_investment = (
        df_strategy["portfolio"].iloc[0] if not df_strategy.empty else 1000
    )
    initial_price = df_baseline["price"].iloc[0]
    df_baseline["portfolio"] = (
        df_baseline["price"] / initial_price
    ) * initial_investment

    fig = go.Figure()

    # Add strategy performance
    fig.add_trace(
        go.Scatter(
            x=df_strategy["date"],
            y=df_strategy["portfolio"],
            mode="lines",
            name="Strategy",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Strategy</b><br><b>Date:</b> %{x}<br><b>Portfolio:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # Add baseline performance
    fig.add_trace(
        go.Scatter(
            x=df_baseline["date"],
            y=df_baseline["portfolio"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="gray", width=2, dash="dash"),
            hovertemplate="<b>Buy & Hold</b><br><b>Date:</b> %{x}<br><b>Portfolio:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_returns_distribution_chart(
    trade_history: List[Dict], title: str = "Daily Returns Distribution"
) -> go.Figure:
    """
    Create a histogram of daily returns.

    Args:
        trade_history: List of trade history dictionaries
        title: Chart title

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not trade_history:
        return go.Figure()

    df = pd.DataFrame(trade_history)
    # Handle multiple date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

    # Calculate daily returns
    df["daily_return"] = df["portfolio"].pct_change() * 100

    # Remove NaN values
    returns = df["daily_return"].dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=30,
            name="Daily Returns",
            marker_color="lightblue",
            opacity=0.7,
            hovertemplate="<b>Return:</b> %{x:.2f}%<br><b>Count:</b> %{y}<extra></extra>",
        )
    )

    # Add mean line
    mean_return = returns.mean()
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.2f}%",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    return fig


def create_comprehensive_dashboard(
    trader_instance, save_html: bool = True, filename: str = "trading_dashboard.html"
) -> go.Figure:
    """
    Create a comprehensive dashboard with all charts.

    Args:
        trader_instance: MATrader instance with trade history and crypto prices
        save_html: Whether to save as HTML file
        filename: HTML filename

    Returns:
        plotly.graph_objects.Figure: Dashboard figure
    """
    # Get strategy name for subplot titles
    strategy_name = getattr(trader_instance, "high_strategy", "Unknown Strategy")

    # Get execution strategies
    buy_strategies = getattr(trader_instance, "strategies", {}).get("buy", ["Unknown"])
    sell_strategies = getattr(trader_instance, "strategies", {}).get(
        "sell", ["Unknown"]
    )

    # Format execution strategies for display
    buy_strat_display = (
        ", ".join(buy_strategies)
        if isinstance(buy_strategies, list)
        else str(buy_strategies)
    )
    sell_strat_display = (
        ", ".join(sell_strategies)
        if isinstance(sell_strategies, list)
        else str(sell_strategies)
    )

    # Create execution strategy subtitle
    execution_subtitle = f"Buy: {buy_strat_display} | Sell: {sell_strat_display}"

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"Portfolio Value Over Time - {strategy_name}",
            f"Asset Allocation - {strategy_name}",
            f"Price History with Signals - {strategy_name}",
            f"Drawdown Analysis - {strategy_name}",
            f"Performance Comparison - {strategy_name}",
            f"Returns Distribution - {strategy_name}",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Get data from trader instance
    trade_history = trader_instance.trade_history
    crypto_prices = trader_instance.crypto_prices

    if not trade_history:
        return fig

    # 1. Portfolio Value Chart
    portfolio_fig = create_portfolio_value_chart(
        trade_history,
        title=f"Portfolio Value Over Time - {strategy_name}",
        execution_strategies=execution_subtitle,
    )
    for trace in portfolio_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # 2. Asset Allocation Chart
    allocation_fig = create_asset_allocation_chart(
        trade_history, title=f"Asset Allocation - {strategy_name}"
    )
    for trace in allocation_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # 3. Price and Signals Chart
    price_fig = create_price_and_signals_chart(
        crypto_prices,
        trade_history,
        title=f"Price History with Signals - {strategy_name}",
    )
    for trace in price_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # 4. Drawdown Chart
    drawdown_fig = create_drawdown_chart(
        trade_history, title=f"Drawdown Analysis - {strategy_name}"
    )
    for trace in drawdown_fig.data:
        fig.add_trace(trace, row=2, col=2)

    # 5. Performance Comparison (simplified - just strategy line)
    df = pd.DataFrame(trade_history)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["portfolio"],
            mode="lines",
            name=f"Strategy Performance ({strategy_name})",
            line=dict(color="blue", width=2),
        ),
        row=3,
        col=1,
    )

    # 6. Returns Distribution
    returns_fig = create_returns_distribution_chart(
        trade_history, title=f"Returns Distribution - {strategy_name}"
    )
    for trace in returns_fig.data:
        fig.add_trace(trace, row=3, col=2)

    # Update layout with strategy name and execution strategies
    strategy_name = getattr(trader_instance, "high_strategy", "Unknown Strategy")
    fig.update_layout(
        title=f"Trading Strategy Dashboard - {trader_instance.crypto_name} ({strategy_name})<br><sub>{execution_subtitle}</sub>",
        height=1200,
        showlegend=True,
        template="plotly_white",
    )

    # Save to HTML if requested
    if save_html:
        fig.write_html(filename)
        print(f"Dashboard saved to {filename}")

    return fig


def save_chart_as_html(fig: go.Figure, filename: str) -> None:
    """
    Save a Plotly chart as an interactive HTML file.

    Args:
        fig: Plotly figure object
        filename: Output filename
    """
    fig.write_html(filename)
    print(f"Chart saved to {filename}")


def display_chart(fig: go.Figure) -> None:
    """
    Display a Plotly chart in a Jupyter notebook or browser.

    Args:
        fig: Plotly figure object

    Returns:
        None
    """
    fig.show()


def create_strategy_performance_chart(
    strategy_performance: List[Dict],
    title: str = "Strategy Performance Comparison",
    top_n: int = 20,
) -> go.Figure:
    """
    Create a bar chart showing the best performing strategies in descending order.

    Args:
        strategy_performance: List of dictionaries containing strategy performance data
        title: Chart title
        top_n: Number of top strategies to display (default: 20)

    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not strategy_performance:
        print("No strategy performance data provided")
        return go.Figure()

    try:
        # Take top N strategies
        top_strategies = strategy_performance[:top_n]
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(top_strategies)
        
        # Create simple strategy labels
        strategy_labels = []
        for i, row in df.iterrows():
            strategy = row.get('strategy', f'Strategy {i}')
            rate = row.get('rate_of_return', 0)
            strategy_labels.append(f"{strategy} ({rate:.2f}%)")
        
        # Create the figure
        fig = go.Figure()
        
        # Add bars for rate of return
        fig.add_trace(
            go.Bar(
                x=strategy_labels,
                y=df['rate_of_return'],
                name='Rate of Return (%)',
                marker_color='lightblue',
                hovertemplate="<b>%{x}</b><br>" +
                             "Rate of Return: %{y:.2f}%<br>" +
                             "Max Drawdown: " + df['max_drawdown'].astype(str) + "%<br>" +
                             "Transactions: " + df['num_transactions'].astype(str) + "<br>" +
                             "Buys: " + df['num_buys'].astype(str) + ", Sells: " + df['num_sells'].astype(str) + "<br>" +
                             "Final Value: $" + df['max_final_value'].astype(str) + "<extra></extra>",
            )
        )
        
        # Add a horizontal line for baseline performance (buy & hold) if available
        if len(df) > 0 and 'baseline_rate_of_return' in df.columns:
            baseline_return = df.iloc[0]['baseline_rate_of_return']
            fig.add_hline(
                y=baseline_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Baseline (Buy & Hold): {baseline_return:.2f}%",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Strategy",
            yaxis_title="Rate of Return (%)",
            xaxis_tickangle=-45,
            showlegend=True,
            template="plotly_white",
            height=600,
            margin=dict(b=150),  # Increase bottom margin for rotated labels
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_strategy_performance_chart: {e}")
        print(f"Strategy performance data length: {len(strategy_performance)}")
        if strategy_performance:
            print(f"First record keys: {list(strategy_performance[0].keys())}")
        return go.Figure()
