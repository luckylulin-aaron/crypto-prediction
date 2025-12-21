# built-in packages
import configparser
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# third-party packages
import numpy as np
import pandas as pd
import schedule

# customized packages
try:
    from core.config import *
    from core.logger import get_logger
    from data.defi_event_client import DefiEventClient
    from data.fear_greed_client import FearGreedClient
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from trading.us_stock_client import USStockClient
    from utils.email_util import send_email
    from utils.util import (
        calculate_simulation_amounts,
        display_port_msg,
        load_csv,
        run_moving_window_simulation,
    )
    from visualization.visualization import (
        create_comprehensive_dashboard,
        create_portfolio_value_chart,
        create_strategy_performance_chart,
    )
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.config import *
    from core.logger import get_logger
    from data.defi_event_client import DefiEventClient
    from data.fear_greed_client import FearGreedClient
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from trading.us_stock_client import USStockClient
    from utils.email_util import send_email
    from utils.util import (
        calculate_simulation_amounts,
        display_port_msg,
        load_csv,
        run_moving_window_simulation,
    )
    from visualization.visualization import (
        create_comprehensive_dashboard,
        create_portfolio_value_chart,
        create_strategy_performance_chart,
    )

logger = get_logger(__name__)

# Read API credentials from secret.ini
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "secret.ini"))

CB_API_KEY = config["CONFIG"]["COINBASE_API_KEY"].strip('"')
CB_API_SECRET = config["CONFIG"]["COINBASE_API_SECRET"].strip('"')
BINANCE_API_KEY = config["CONFIG"]["BINANCE_API_KEY"].strip('"')
BINANCE_API_SECRET = config["CONFIG"]["BINANCE_API_SECRET"].strip('"')
GMAIL_ADDRESS = config["CONFIG"].get("GMAIL_ADDRESS", "").strip('"')
GMAIL_APP_PASSWORD = config["CONFIG"].get("GMAIL_APP_PASSWORD", "").strip('"')
GMAIL_RECIPIENTS = config["CONFIG"].get("GMAIL_RECIPIENTS", "").strip('"')
RECIPIENT_LIST = [
    email.strip() for email in GMAIL_RECIPIENTS.split(",") if email.strip()
]


def send_daily_recommendations_email(
    log_file, recipient_list, from_email, app_password
):
    """
    Send daily recommendations email from log.txt for today's actions.
    
    Sends different content based on recipient:
    - First recipient (you): Gets all recommendations (crypto + stocks)
    - Other recipients: Gets only crypto recommendations (stocks filtered out)

    Args:
        log_file: path to the log file
        recipient_list: list of email addresses to send the email to
        from_email: email address to send the email from
        app_password: app password for the email account

    Returns:
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    buy_sell_lines = []
    no_action_entries = []

    with open(log_file, "r") as infile:
        for line in infile:
            if line.strip() and line[:10] == today_str:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) == 6:
                    time, exch, asset, action, buy, sell = parts
                    action_val = action.replace("Action: ", "")
                    buy_val = buy.replace("Buy %: ", "")
                    sell_val = sell.replace("Sell %: ", "")
                    if action_val.upper() in ("BUY", "SELL"):
                        buy_sell_lines.append(
                            (time, exch, asset, action_val, buy_val, sell_val)
                        )
                    else:
                        no_action_entries.append((exch, asset))
                else:
                    # fallback: treat as no action
                    no_action_entries.append(("?", "?"))

    if not buy_sell_lines and not no_action_entries:
        logger.info("No trading actions found for today, skipping email notification.")
        return

    # Separate crypto and stock recommendations
    crypto_lines = []
    stock_lines = []
    crypto_no_action = []
    stock_no_action = []
    
    # Filter buy/sell lines
    for line in buy_sell_lines:
        time, exch, asset, action, buy, sell = line
        if exch == "STOCK":
            stock_lines.append(line)
        else:
            crypto_lines.append(line)
    
    # Filter no action entries
    for exch, asset in no_action_entries:
        if exch == "STOCK":
            stock_no_action.append((exch, asset))
        else:
            crypto_no_action.append((exch, asset))

    # Send different emails to different recipients
    for i, recipient in enumerate(recipient_list):
        body = ""
        
        if i == 0:
            # First recipient (you) - gets everything
            all_lines = crypto_lines + stock_lines
            all_no_action = crypto_no_action + stock_no_action
            
            if all_lines:
                header = f"{'Time':<19} | {'Exchange':<8} | {'Asset':<8} | {'Action':<10} | {'Buy %':<6} | {'Sell %':<6}"
                sep = "-" * len(header)
                formatted_lines = [
                    f"{t:<19} | {e:<8} | {a:<8} | {ac:<10} | {b:<6} | {s:<6}"
                    for t, e, a, ac, b, s in all_lines
                ]
                body += f"{header}\n{sep}\n" + "\n".join(formatted_lines) + "\n"
                body += (
                    "\nBuy %: Recommended proportion of available funds to use for buying this asset.\n"
                    "Sell %: Recommended proportion of current holdings of this asset to sell.\n\n"
                )

            if all_no_action:
                from collections import defaultdict
                exch_assets = defaultdict(list)
                for exch, asset in all_no_action:
                    exch_assets[exch].append(asset)
                body += "No action recommended for the following assets today:\n"
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    body += f"- {exch}: {asset_list}\n"
                    
        else:
            # Other recipients - crypto only
            if crypto_lines:
                header = f"{'Time':<19} | {'Exchange':<8} | {'Asset':<8} | {'Action':<10} | {'Buy %':<6} | {'Sell %':<6}"
                sep = "-" * len(header)
                formatted_lines = [
                    f"{t:<19} | {e:<8} | {a:<8} | {ac:<10} | {b:<6} | {s:<6}"
                    for t, e, a, ac, b, s in crypto_lines
                ]
                body += f"{header}\n{sep}\n" + "\n".join(formatted_lines) + "\n"
                body += (
                    "\nBuy %: Recommended proportion of available stablecoin to use for buying this asset.\n"
                    "Sell %: Recommended proportion of current holdings of this asset to sell.\n\n"
                )

            if crypto_no_action:
                from collections import defaultdict
                exch_assets = defaultdict(list)
                for exch, asset in crypto_no_action:
                    exch_assets[exch].append(asset)
                body += "No action recommended for the following assets today:\n"
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    body += f"- {exch}: {asset_list}\n"

        # Send email to this specific recipient
        subject = f"Daily Trading Bot Recommendations ({today_str})"
        send_email(
            subject=subject,
            body=body.strip(),
            to_emails=[recipient],  # Send to individual recipient
            from_email=from_email,
            app_password=app_password,
        )
        
    return


def main_defi():
    """Send the DEFI asset valuation report email (runs only on Sundays)."""
    cfg = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "secret.ini")
    cfg.read(config_path)
    section = "CONFIG"

    def get_secret(key: str, env_fallback: Optional[str] = None) -> str:
        if cfg.has_option(section, key):
            return cfg.get(section, key).strip('"')
        return os.environ.get(env_fallback or key, "")

    to_emails_raw = get_secret("DEFI_REPORT_TO_EMAILS")
    from_email = get_secret("DEFI_REPORT_FROM_EMAIL")
    app_password = get_secret("DEFI_REPORT_APP_PASSWORD")

    to_emails = [e.strip() for e in to_emails_raw.split(",") if e.strip()]
    if DEBUG and to_emails:
        to_emails = to_emails[:1]

    if to_emails and from_email and app_password:
        logger.info(f"Sending DEFI event client email to {to_emails}")
        DefiEventClient().run_and_email(to_emails, from_email, app_password, top_n=3)
    else:
        logger.warning("DEFI event client email not sent: missing credentials in secret.ini")


def fetch_historical_data_with_fallback(asset: str, binance_client: BinanceClient, coinbase_client: CBProClient, exchange_configs: list):
    """
    Fetch historical data for an asset, trying Binance first, then falling back to Coinbase.
    
    Args:
        asset (str): The asset symbol (e.g., 'BTC', 'ETH')
        binance_client (BinanceClient): Binance client instance
        coinbase_client (CBProClient): Coinbase client instance
        exchange_configs (list): List of exchange configurations
        
    Returns:
        tuple: (data_stream, source_exchange_name) or (None, None) if both fail
    """
    def validate_and_format_data(data_stream):
        """
        Validate and format data to ensure it matches the expected format for trader_driver.
        Expected format: (price, date, open, low, high)
        """
        if not data_stream:
            return None
            
        formatted_data = []
        for item in data_stream:
            if len(item) >= 5:
                # Take only the first 5 elements: (price, date, open, low, high)
                formatted_item = (item[0], item[1], item[2], item[3], item[4])
                formatted_data.append(formatted_item)
            else:
                logger.warning(f"Skipping data point with insufficient elements: {item}")
                
        return formatted_data if formatted_data else None
    
    # Try Binance first
    binance_config = next((config for config in exchange_configs if config["name"] == ExchangeName.BINANCE), None)
    if binance_config:
        try:
            binance_symbol = binance_config["symbol_format"](asset)
            logger.info(f"Attempting to fetch {asset} data from Binance using symbol: {binance_symbol}")
            data_stream = binance_client.get_historic_data(binance_symbol)
            
            # Validate and format the data
            formatted_data = validate_and_format_data(data_stream)
            if formatted_data and len(formatted_data) >= 2:
                logger.info(f"Successfully fetched {len(formatted_data)} data points from Binance for {asset}")
                return formatted_data, ExchangeName.BINANCE
            else:
                logger.warning(f"Binance returned insufficient data for {asset}: {len(formatted_data) if formatted_data else 0} points")
        except Exception as e:
            logger.warning(f"Failed to fetch {asset} data from Binance: {e}")
    
    # Fall back to Coinbase
    coinbase_config = next((config for config in exchange_configs if config["name"] == ExchangeName.COINBASE), None)
    if coinbase_config:
        try:
            coinbase_symbol = coinbase_config["symbol_format"](asset)
            logger.info(f"Attempting to fetch {asset} data from Coinbase using symbol: {coinbase_symbol}")
            data_stream = coinbase_client.get_historic_data(coinbase_symbol)
            
            # Validate and format the data
            formatted_data = validate_and_format_data(data_stream)
            if formatted_data and len(formatted_data) >= 2:
                logger.info(f"Successfully fetched {len(formatted_data)} data points from Coinbase for {asset}")
                return formatted_data, ExchangeName.COINBASE
            else:
                logger.warning(f"Coinbase returned insufficient data for {asset}: {len(formatted_data) if formatted_data else 0} points")
        except Exception as e:
            logger.warning(f"Failed to fetch {asset} data from Coinbase: {e}")
    
    logger.error(f"Both Binance and Coinbase failed to provide data for {asset}")
    return None, None


def main():
    """
    Run simulation and make trades.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: If there is an error during portfolio value retrieval or trading simulation.
    """

    logger.info(f"COMMIT is set to {COMMIT}")

    # initialise different clients
    coinbase_client = CBProClient(key=CB_API_KEY, secret=CB_API_SECRET)
    binance_client = BinanceClient(
        api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET
    )

    # Fear & Greed Index client for market sentiment data
    fear_greed_client = FearGreedClient()

    # Get actual portfolio values for simulation
    try:
        logger.info("Getting portfolio value (Coinbase)...")
        portfolio_result = coinbase_client.portfolio_value
        
        if isinstance(portfolio_result, tuple) and len(portfolio_result) == 2:
            coinbase_crypto_value, coinbase_stablecoin_value = portfolio_result
            logger.info(f"Coinbase portfolio: crypto=${coinbase_crypto_value}, stable=${coinbase_stablecoin_value}")
        else:
            logger.error(f"Unexpected portfolio result format")
            coinbase_crypto_value, coinbase_stablecoin_value = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error getting portfolio value: {e}")
        coinbase_crypto_value, coinbase_stablecoin_value = 0.0, 0.0

    display_port_msg(
        v_c=coinbase_crypto_value, v_s=coinbase_stablecoin_value, before=True
    )

    # Get Binance portfolio value
    try:
        logger.info("Getting portfolio value (Binance)...")
        binance_portfolio_result = binance_client.portfolio_value
        if isinstance(binance_portfolio_result, tuple) and len(binance_portfolio_result) == 2:
            binance_crypto_value, binance_stablecoin_value = binance_portfolio_result
            logger.info(f"Binance portfolio: crypto=${binance_crypto_value}, stable=${binance_stablecoin_value}")
        else:
            logger.error(f"Unexpected Binance portfolio result format")
            binance_crypto_value, binance_stablecoin_value = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error getting Binance portfolio value: {e}")
        binance_crypto_value, binance_stablecoin_value = 0.0, 0.0

    display_port_msg(
        v_c=binance_crypto_value, v_s=binance_stablecoin_value, before=True
    )

    # Fetch valid symbols for each exchange
    # Binance
    try:
        binance_info = binance_client.client.exchange_info()
        binance_symbols = set(s["symbol"] for s in binance_info["symbols"])
    except Exception as e:
        logger.error(f"Failed to fetch Binance symbols: {e}")
        binance_symbols = set()
    # Coinbase
    try:
        coinbase_products = coinbase_client.rest_client.get_products()
        coinbase_symbols = set(p["product_id"] for p in coinbase_products["products"])
    except Exception as e:
        logger.error(f"Failed to fetch Coinbase products: {e}")
        coinbase_symbols = set()

    # Map exchange name to valid symbol set
    valid_symbols_map = {
        ExchangeName.BINANCE: binance_symbols,
        ExchangeName.COINBASE: coinbase_symbols,
    }

    # --- Unified simulation for both exchanges ---
    # Explicit mapping by exchange name
    exchange_client_map = {
        ExchangeName.COINBASE: coinbase_client,
        ExchangeName.BINANCE: binance_client,
    }
    exchange_crypto_value_map = {
        ExchangeName.COINBASE: coinbase_crypto_value,
        ExchangeName.BINANCE: binance_crypto_value,
    }
    exchange_stablecoin_value_map = {
        ExchangeName.COINBASE: coinbase_stablecoin_value,
        ExchangeName.BINANCE: binance_stablecoin_value,
    }
    exchanges = []
    for config in EXCHANGE_CONFIGS:
        name = config["name"]
        exch = dict(config)  # shallow copy
        exch["client"] = exchange_client_map[name]
        exch["crypto_value"] = exchange_crypto_value_map[name]
        exch["stablecoin_value"] = exchange_stablecoin_value_map[name]
        exchanges.append(exch)

    simulated_assets = set()
    all_actions = []

    # Display portfolio information for both exchanges
    logger.info("=== Portfolio Overview ===")
    for exchange in exchanges:
        logger.info(f"--- {exchange['name'].value} Portfolio ---")
        display_port_msg(
            v_c=exchange["crypto_value"], v_s=exchange["stablecoin_value"], before=True
        )

    asset_list = CURS[:1] if DEBUG else CURS # only test 1 asset for debugging purposes
    for asset in asset_list:
        # Only simulate each asset once, regardless of exchange
        if asset in simulated_assets:
            logger.info(f"Skipping duplicate simulation for asset: {asset}")
            continue
        simulated_assets.add(asset)

        logger.info(
            f"\n\n# --- Simulating for asset: {asset} --- #"
        )
        
        # Use fallback approach: try Binance first, then Coinbase
        data_stream, source_exchange = fetch_historical_data_with_fallback(
            asset, binance_client, coinbase_client, EXCHANGE_CONFIGS
        )
        
        if data_stream is None:
            logger.error(f"Failed to fetch historical data for {asset} from both exchanges")
            continue
            
        logger.info(f"Using data from {source_exchange.value} for {asset}")
        
        # Use the source exchange for wallet and portfolio data
        source_exchange_config = next((config for config in exchanges if config["name"] == source_exchange), None)
        if not source_exchange_config:
            logger.error(f"Could not find configuration for {source_exchange.value}")
            continue

        wallet = source_exchange_config["client"].get_wallets()
        coin_amount = 0.0

        for item in wallet:
            if isinstance(item, dict):
                asset_name = item.get(source_exchange_config["asset_key"])
                if asset_name == asset:
                    if source_exchange_config["coin_value_key"]:
                        coin_amount = float(
                            item[source_exchange_config["coin_key"]][source_exchange_config["coin_value_key"]]
                        )
                    else:
                        coin_amount = float(item[source_exchange_config["coin_key"]])
            else:
                asset_name = getattr(item, source_exchange_config["asset_key"], None)
                if asset_name == asset:
                    balance = getattr(item, source_exchange_config["coin_key"])
                    if source_exchange_config["coin_value_key"]:
                        coin_amount = float(balance[source_exchange_config["coin_value_key"]])
                    else:
                        coin_amount = float(balance)
        if coin_amount == 0.0:
            logger.warning(f"No {asset} found in {source_exchange.value} wallet.")
            # Set a default initial amount for simulation purposes
            sim_coin_amount = DEFAULT_SIMULATION_COIN_AMOUNT
            logger.info(f"Using simulation amount of {sim_coin_amount} {asset} for testing")
        else:
            sim_coin_amount = coin_amount

        # Run simulation
        # simulation configuration
        if DEBUG:
            SIM_BUY_PCTS = [BUY_PCTS[0]]
            SIM_SELL_PCTS = [SELL_PCTS[0]]
        else:
            SIM_BUY_PCTS = BUY_PCTS
            SIM_SELL_PCTS = SELL_PCTS
        try:
            # Validate data stream before creating trader driver
            if not data_stream:
                logger.error(f"No historical data available for {asset}")
                continue
            
            if len(data_stream) < 2:
                logger.error(f"Insufficient historical data for {asset}: only {len(data_stream)} data points available")
                continue
            
            # Convert window size from days to data points based on DATA_INTERVAL_HOURS
            # Data points per day = 24 hours / DATA_INTERVAL_HOURS
            data_points_per_day = 24 / DATA_INTERVAL_HOURS
            window_size_data_points = int(MOVING_WINDOW_DAYS * data_points_per_day)
            step_size_data_points = int(MOVING_WINDOW_STEP * data_points_per_day)
            
            logger.info(
                f"Starting moving window simulation for {asset} using {source_exchange.value} data "
                f"with {len(data_stream)} data points (window size: {MOVING_WINDOW_DAYS} days = {window_size_data_points} data points, step: {MOVING_WINDOW_STEP} days = {step_size_data_points} data points)"
            )
            logger.info(
                f"Data interval configuration: {DATA_INTERVAL_HOURS}h intervals, "
                f"~{data_points_per_day:.2f} data points per day, "
                f"total data span covers ~{len(data_stream) / data_points_per_day:.1f} days"
            )
            
            # Run moving window simulation
            moving_window_results = run_moving_window_simulation(
                trader_driver_class=TraderDriver,
                data_stream=data_stream,
                window_size=window_size_data_points,
                step_size=step_size_data_points,
                name=asset,
                init_amount=source_exchange_config["stablecoin_value"],
                cur_coin=sim_coin_amount,
                # only test 1 strategy for debugging purposes
                overall_stats=STRATEGIES if DEBUG is not True else STRATEGIES[:5],
                tol_pcts=TOL_PCTS,
                ma_lengths=MA_LENGTHS,
                ema_lengths=EMA_LENGTHS,
                bollinger_mas=BOLLINGER_MAS,
                bollinger_tols=BOLLINGER_TOLS,
                buy_pcts=SIM_BUY_PCTS,
                sell_pcts=SIM_SELL_PCTS,
                buy_stas=BUY_STAS,
                sell_stas=SELL_STAS,
                rsi_periods=RSI_PERIODS,
                rsi_oversold_thresholds=RSI_OVERSOLD_THRESHOLDS,
                rsi_overbought_thresholds=RSI_OVERBOUGHT_THRESHOLDS,
                kdj_oversold_thresholds=KDJ_OVERSOLD_THRESHOLDS,
                kdj_overbought_thresholds=KDJ_OVERBOUGHT_THRESHOLDS,
                mode="normal",
            )
            
            # Get aggregated metrics for best strategy
            best_strategy = moving_window_results["best_strategy"]
            best_metrics = moving_window_results["best_strategy_metrics"]
            best_window_result = moving_window_results["best_window_result"]
            
            # Create a TraderDriver with full data to get the current signal
            # Use the most recent data for signal generation
            trader_driver = TraderDriver(
                name=asset,
                init_amount=source_exchange_config["stablecoin_value"],
                cur_coin=sim_coin_amount,
                # only test 1 strategy for debugging purposes
                overall_stats=STRATEGIES if DEBUG is not True else STRATEGIES[:5],
                tol_pcts=TOL_PCTS,
                ma_lengths=MA_LENGTHS,
                ema_lengths=EMA_LENGTHS,
                bollinger_mas=BOLLINGER_MAS,
                bollinger_tols=BOLLINGER_TOLS,
                buy_pcts=SIM_BUY_PCTS,
                sell_pcts=SIM_SELL_PCTS,
                buy_stas=BUY_STAS,
                sell_stas=SELL_STAS,
                rsi_periods=RSI_PERIODS,
                rsi_oversold_thresholds=RSI_OVERSOLD_THRESHOLDS,
                rsi_overbought_thresholds=RSI_OVERBOUGHT_THRESHOLDS,
                kdj_oversold_thresholds=KDJ_OVERSOLD_THRESHOLDS,
                kdj_overbought_thresholds=KDJ_OVERBOUGHT_THRESHOLDS,
                mode="normal",
            )
            trader_driver.feed_data(data_stream)
            best_info = trader_driver.best_trader_info
            best_t = trader_driver.traders[best_info["trader_index"]]
            signal = best_t.trade_signal

            # Log aggregated best trader summary with exchange name
            logger.info(
                f"\n{'★'*10} MOVING WINDOW SIMULATION RESULTS ({source_exchange.value}) {'★'*10}\n"
                f"Total windows analyzed: {moving_window_results['num_windows']}\n"
                f"Window size: {MOVING_WINDOW_DAYS} days ({window_size_data_points} data points at {DATA_INTERVAL_HOURS}h intervals)\n"
                f"Best strategy (aggregated): {best_strategy}\n"
                f"\n--- Aggregated Performance Metrics ---\n"
                f"Mean rate of return: {best_metrics['mean_rate_of_return']:.2f}%\n"
                f"Std dev of return: {best_metrics['std_rate_of_return']:.2f}%\n"
                f"Min rate of return: {best_metrics['min_rate_of_return']:.2f}%\n"
                f"Max rate of return: {best_metrics['max_rate_of_return']:.2f}%\n"
                f"Median rate of return: {best_metrics['median_rate_of_return']:.2f}%\n"
                f"Risk-adjusted return (mean - std): {best_metrics['risk_adjusted_return']:.2f}%\n"
                f"Win rate: {best_metrics['win_rate']*100:.1f}%\n"
                f"Mean baseline rate: {best_metrics['mean_baseline_rate']:.2f}%\n"
                f"Mean coin rate: {best_metrics['mean_coin_rate']:.2f}%\n"
                f"Mean max drawdown: {best_metrics['mean_drawdown']:.2f}%\n"
                f"Mean transactions: {best_metrics['mean_transactions']:.1f}\n"
                f"\n--- Best Window Performance ---\n"
                f"Best window period: {best_window_result['window_start_date']} to {best_window_result['window_end_date']}\n"
                f"Best window rate of return: {best_window_result['rate_of_return']:.2f}%\n"
                f"Today's signal: {signal} for crypto={best_t.crypto_name}\n"
                f"{'★'*50}\n"
            )

            # Save visualizations
            strategy_performance = trader_driver.get_all_strategy_performance()
            dashboard_filename = f"app/visualization/plots/trading_dashboard_{asset}_{source_exchange.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            create_comprehensive_dashboard(
                trader_instance=best_t,
                save_html=True,
                filename=dashboard_filename,
                strategy_performance=strategy_performance,
            )

            # Gather recommended action for email/log
            action_line = f"{datetime.now()} | {source_exchange.value} | {asset} | Action: {signal['action']} | Buy %: {signal.get('buy_percentage', '')} | Sell %: {signal.get('sell_percentage', '')}"
            all_actions.append(action_line)

            # Log recommended action to log.txt
            with open(LOG_FILE, "a") as outfile:
                outfile.write(action_line + "\n")

        except ValueError as e:
            logger.error(
                f"Data validation failed for {asset} using {source_exchange.value}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Simulation failed for {asset} using {source_exchange.value}: {e}"
            )

    # after
    try:
        portfolio_result_after = coinbase_client.portfolio_value
        if isinstance(portfolio_result_after, tuple) and len(portfolio_result_after) == 2:
            coinbase_crypto_value_after, coinbase_stablecoin_value_after = portfolio_result_after
        else:
            logger.error(f"Unexpected portfolio result format after: {portfolio_result_after}")
            coinbase_crypto_value_after, coinbase_stablecoin_value_after = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error getting portfolio value after: {e}")
        coinbase_crypto_value_after, coinbase_stablecoin_value_after = 0.0, 0.0
        
    display_port_msg(
        v_c=coinbase_crypto_value_after,
        v_s=coinbase_stablecoin_value_after,
        before=False,
    )

    try:
        binance_portfolio_result_after = binance_client.portfolio_value
        if isinstance(binance_portfolio_result_after, tuple) and len(binance_portfolio_result_after) == 2:
            binance_crypto_value_after, binance_stablecoin_value_after = binance_portfolio_result_after
        else:
            logger.error(f"Unexpected Binance portfolio result format after: {binance_portfolio_result_after}")
            binance_crypto_value_after, binance_stablecoin_value_after = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error getting Binance portfolio value after: {e}")
        binance_crypto_value_after, binance_stablecoin_value_after = 0.0, 0.0
        
    display_port_msg(
        v_c=binance_crypto_value_after, v_s=binance_stablecoin_value_after, before=False
    )

    # --- Stock Trading Simulation ---
    logger.info("\n" + "="*50)
    logger.info("STARTING STOCK TRADING SIMULATION")
    logger.info("="*50)
    
    # Check if it's a weekend day (Sunday or Monday) to skip stock simulation
    # US stock market is closed on weekends, and there's a one-day delay in data
    current_weekday = datetime.now().weekday()  # Monday=0, Sunday=6
    if current_weekday in [6, 0]:  # Sunday (6) or Monday (0)
        logger.info(f"Skipping stock simulation - current day is {'Sunday' if current_weekday == 6 else 'Monday'}")
        logger.info("US stock market is closed on weekends, and data has one-day delay")
    else:
        logger.info(f"Proceeding with stock simulation - current day is {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_weekday]}")
        
        # Initialize US Stock Client
        stock_client = USStockClient(tickers=STOCKS)
        
        # only 1 stock for debugging purposes
        stock_list = STOCKS[:1] if DEBUG else STOCKS

        # Simulate stock trading for each stock
        for stock in stock_list:
            logger.info(f"\n\n# --- Simulating for Stock: {stock} --- #")
            
            try:
                # Get historical data for the stock using TIMESPAN
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=TIMESPAN)).strftime('%Y-%m-%d')
                data_stream = stock_client.get_historic_data(stock, start=start_date, end=end_date)
                logger.info(f"Retrieved {len(data_stream)} data points for {stock} (last {TIMESPAN} days)")
                
                # Validate data stream before creating trader driver
                if not data_stream:
                    logger.error(f"No historical data available for stock {stock}")
                    continue
                
                if len(data_stream) < 2:
                    logger.error(f"Insufficient historical data for stock {stock}: only {len(data_stream)} data points available")
                    continue
                
                # For stock simulation, we'll use a fixed initial amount
                # You can modify this based on your stock portfolio value
                initial_stock_amount = 10000  # $10,000 initial investment
                current_stock_amount = 0  # Assume no current holdings for simulation
                
                # Run simulation for stocks
                if DEBUG:
                    SIM_BUY_PCTS = [BUY_PCTS[0]]
                    SIM_SELL_PCTS = [SELL_PCTS[0]]
                else:
                    SIM_BUY_PCTS = BUY_PCTS
                    SIM_SELL_PCTS = SELL_PCTS
                    
                trader_driver = TraderDriver(
                    name=stock,
                    init_amount=initial_stock_amount,
                    cur_coin=current_stock_amount,
                    # only test 1 strategy for debugging purposes
                    overall_stats=STRATEGIES if DEBUG is not True else STRATEGIES[:5],
                    tol_pcts=TOL_PCTS,
                    ma_lengths=MA_LENGTHS,
                    ema_lengths=EMA_LENGTHS,
                    bollinger_mas=BOLLINGER_MAS,
                    bollinger_tols=BOLLINGER_TOLS,
                    buy_pcts=SIM_BUY_PCTS,
                    sell_pcts=SIM_SELL_PCTS,
                    buy_stas=BUY_STAS,
                    sell_stas=SELL_STAS,
                    rsi_periods=RSI_PERIODS,
                    rsi_oversold_thresholds=RSI_OVERSOLD_THRESHOLDS,
                    rsi_overbought_thresholds=RSI_OVERBOUGHT_THRESHOLDS,
                    kdj_oversold_thresholds=KDJ_OVERSOLD_THRESHOLDS,
                    kdj_overbought_thresholds=KDJ_OVERBOUGHT_THRESHOLDS,
                    mode="normal",
                )
                trader_driver.feed_data(data_stream)
                best_info = trader_driver.best_trader_info
                best_t = trader_driver.traders[best_info["trader_index"]]
                signal = best_t.trade_signal

                # Log best trader summary for stock
                init_value = best_info.get('init_value', 0)
                max_final_value = best_info.get('max_final_value', 0)
                logger.info(
                    f"\n{'★'*10} BEST TRADER SUMMARY (STOCK: {stock}) {'★'*10}\n"
                    f"Best trader performance:\n"
                    f"  Strategy Parameters:\n"
                    f"    - Buy percentage: {best_info.get('buy_pct', 'N/A')}\n"
                    f"    - Sell percentage: {best_info.get('sell_pct', 'N/A')}\n"
                    f"    - Tolerance percentage: {best_info.get('tol_pct', 'N/A')}\n"
                    f"    - Bollinger sigma: {best_info.get('bollinger_sigma', 'N/A')}\n"
                    f"    - Buy strategy: {best_info.get('buy', 'N/A')}\n"
                    f"    - Sell strategy: {best_info.get('sell', 'N/A')}\n"
                    f"  Performance Metrics:\n"
                    f"    - Initial value: ${init_value:,.2f}\n"
                    f"    - Final value: ${max_final_value:,.2f}\n"
                    f"    - Rate of return: {best_info.get('rate_of_return', 'N/A')}\n"
                    f"    - Baseline rate of return: {best_info.get('baseline_rate_of_return', 'N/A')}\n"
                    f"    - Coin rate of return: {best_info.get('coin_rate_of_return', 'N/A')}\n"
                    f"  Trading Statistics:\n"
                    f"    - Max drawdown: {best_t.max_drawdown * 100:.2f}%\n"
                    f"    - Transactions: {best_t.num_transaction}\n"
                    f"    - Buys: {best_t.num_buy_action}, Sells: {best_t.num_sell_action}\n"
                    f"    - Strategy: {best_t.high_strategy}\n"
                    f"    - Today's signal: {signal} for stock={best_t.crypto_name}\n"
                    f"{'★'*36}\n"
                )

                # Save visualizations for stock
                strategy_performance = trader_driver.get_all_strategy_performance()
                dashboard_filename = f"app/visualization/plots/trading_dashboard_{stock}_STOCK_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                create_comprehensive_dashboard(
                    trader_instance=best_t,
                    save_html=True,
                    filename=dashboard_filename,
                    strategy_performance=strategy_performance,
                )

                # Gather recommended action for email/log
                action_line = f"{datetime.now()} | STOCK | {stock} | Action: {signal['action']} | Buy %: {signal.get('buy_percentage', '')} | Sell %: {signal.get('sell_percentage', '')}"
                all_actions.append(action_line)

                # Log recommended action to log.txt
                with open(LOG_FILE, "a") as outfile:
                    outfile.write(action_line + "\n")

            except ValueError as e:
                logger.error(f"Data validation failed for stock {stock}: {e}")
                continue
            except Exception as e:
                logger.error(f"Stock simulation failed for {stock}: {e}")
                continue

    # Send daily recommendations email if not in debug mode
    if DEBUG is False:
        send_daily_recommendations_email(
            LOG_FILE, RECIPIENT_LIST, GMAIL_ADDRESS, GMAIL_APP_PASSWORD
        )

    # Send DEFI report email based on configuration
    if DEFI_MONITORING_ENABLED:
        current_day = datetime.now().weekday()  # Monday=0, Sunday=6
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if current_day in DEFI_MONITORING_DAYS:
            logger.info(f"DEFI monitoring day detected ({day_names[current_day]}) - running DEFI monitoring")
            main_defi()
        else:
            logger.info(f"DEFI monitoring skipped - today is {day_names[current_day]} (runs on: {[day_names[d] for d in DEFI_MONITORING_DAYS]})")
    else:
        logger.info("DEFI monitoring disabled in configuration")

    # write to log file
    now = datetime.now()
    with open(LOG_FILE, "a") as outfile:
        outfile.write("Finish job at time {}\n\n".format(str(now)))


def run_trading_job():
    main()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cronjob":
        logger.info("Starting trading bot in schedule-based cronjob mode...")
        # Schedule the job for 1:00 PM UTC (9:00 PM SGT)
        schedule.every().day.at("13:00").do(run_trading_job)
        logger.info("Trading bot scheduled to run daily at 9:00 PM SGT (1:00 PM UTC)")
        logger.info("Press Ctrl+C to stop the bot")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")

    elif any(
        arg.startswith("--sendEmail=") and arg.split("=", 1)[1].lower() == "true"
        for arg in sys.argv[1:]
    ):
        logger.info(
            "Sending daily trading recommendations email only (no simulation)..."
        )
        send_daily_recommendations_email(
            LOG_FILE, RECIPIENT_LIST, GMAIL_ADDRESS, GMAIL_APP_PASSWORD
        )

    else:
        logger.info("Starting trading bot in one-time mode...")
        main()
