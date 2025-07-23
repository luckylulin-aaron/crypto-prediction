# built-in packages
import configparser
import os
import sys
import time
from datetime import datetime, timedelta

# third-party packages
import numpy as np
import pandas as pd
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# customized packages
try:
    from core.config import *
    from core.logger import get_logger
    from data.fear_greed_client import FearGreedClient
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg, load_csv
    from utils.email_util import send_email
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
    from data.fear_greed_client import FearGreedClient
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg, load_csv
    from utils.email_util import send_email
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
RECIPIENT_LIST = [email.strip() for email in GMAIL_RECIPIENTS.split(",") if email.strip()]


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
    log_file = "./log.txt"

    # initialise different clients
    coinbase_client = CBProClient(key=CB_API_KEY, secret=CB_API_SECRET)
    binance_client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

    # Fear & Greed Index client for market sentiment data
    fear_greed_client = FearGreedClient()

    # Get actual portfolio values for simulation
    try:
        logger.info("Getting portfolio value (Coinbase)...")
        coinbase_crypto_value, coinbase_stablecoin_value = coinbase_client.portfolio_value
        logger.info(f"Portfolio value before, crypto={coinbase_crypto_value}, stablecoin={coinbase_stablecoin_value}")
    except Exception as e:
        logger.error(f"Error getting portfolio value: {e}")
        return

    display_port_msg(v_c=coinbase_crypto_value, v_s=coinbase_stablecoin_value, before=True)

    # Get Binance portfolio value
    try:
        logger.info("Getting portfolio value (Binance)...")
        binance_crypto_value, binance_stablecoin_value = binance_client.portfolio_value
        logger.info(f"Binance portfolio value, crypto={binance_crypto_value}, stablecoin={binance_stablecoin_value}")
    except Exception as e:
        logger.error(f"Error getting Binance portfolio value: {e}")

    display_port_msg(v_c=binance_crypto_value, v_s=binance_stablecoin_value, before=True)

    # Fetch valid symbols for each exchange
    # Binance
    try:
        binance_info = binance_client.client.exchange_info()
        binance_symbols = set(s['symbol'] for s in binance_info['symbols'])
    except Exception as e:
        logger.error(f"Failed to fetch Binance symbols: {e}")
        binance_symbols = set()
    # Coinbase
    try:
        coinbase_products = coinbase_client.rest_client.get_products()
        coinbase_symbols = set(p['product_id'] for p in coinbase_products['products'])
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

    all_actions = []

    for exchange in exchanges:
        logger.info(f"=== {exchange['name'].value} Portfolio ===")
        display_port_msg(v_c=exchange['crypto_value'], v_s=exchange['stablecoin_value'], before=True)
        
        asset_list = CURS[:1] if DEBUG else CURS
        for asset in asset_list:
            logger.info(f"\n\n# --- Simulating for {exchange['name'].value} asset: {asset} --- #")
            symbol = exchange['symbol_format'](asset)
            # Check if symbol is valid for this exchange
            if symbol not in valid_symbols_map[exchange['name']]:
                logger.warning(f"Skipping {asset} on {exchange['name'].value}: {symbol} not available.")
                continue
            try:
                data_stream = exchange['client'].get_historic_data(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {asset} on {exchange['name'].value}: {e}")
                continue

            wallet = exchange['client'].get_wallets()
            coin_amount = 0.0

            for item in wallet:
                if isinstance(item, dict):
                    asset_name = item.get(exchange['asset_key'])
                    if asset_name == asset:
                        if exchange['coin_value_key']:
                            coin_amount = float(item[exchange['coin_key']][exchange['coin_value_key']])
                        else:
                            coin_amount = float(item[exchange['coin_key']])
                else:
                    asset_name = getattr(item, exchange['asset_key'], None)
                    if asset_name == asset:
                        balance = getattr(item, exchange['coin_key'])
                        if exchange['coin_value_key']:
                            coin_amount = float(balance[exchange['coin_value_key']])
                        else:
                            coin_amount = float(balance)
            if coin_amount == 0.0:
                logger.warning(f"No {asset} found in {exchange['name']} wallet.")

            # Run simulation
            try:
                trader_driver = TraderDriver(
                    name=asset,
                    init_amount=exchange['stablecoin_value'],
                    cur_coin=coin_amount,
                    # only test 1 strategy for debugging purposes
                    overall_stats=STRATEGIES if DEBUG is not True else STRATEGIES[:5],
                    tol_pcts=TOL_PCTS,
                    ma_lengths=MA_LENGTHS,
                    ema_lengths=EMA_LENGTHS,
                    bollinger_mas=BOLLINGER_MAS,
                    bollinger_tols=BOLLINGER_TOLS,
                    buy_pcts=BUY_PCTS,
                    sell_pcts=SELL_PCTS,
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

                # Log best trader summary with exchange name
                logger.info(
                    f"\n{'★'*10} BEST TRADER SUMMARY ({exchange['name'].value}) {'★'*10}\n"
                    f"Best trader performance: {best_info}\n"
                    f"Max drawdown: {best_t.max_drawdown * 100:.2f}%\n"
                    f"Transactions: {best_t.num_transaction}, "
                    f"Buys: {best_t.num_buy_action}, Sells: {best_t.num_sell_action}\n"
                    f"Strategy: {best_t.high_strategy}\n"
                    f"Today's signal: {signal} for crypto={best_t.crypto_name}\n"
                    f"{'★'*36}\n"
                )

                # Save visualizations
                dashboard_filename = f"app/visualization/plots/trading_dashboard_{asset}_{exchange['name'].value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                create_comprehensive_dashboard(trader_instance=best_t, save_html=True, filename=dashboard_filename)
                portfolio_filename = f"app/visualization/plots/portfolio_chart_{asset}_{exchange['name'].value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                portfolio_chart = create_portfolio_value_chart(trade_history=best_t.trade_history, title=f"Portfolio Value - {asset} ({best_t.high_strategy})")
                portfolio_chart.write_html(portfolio_filename)
                strategy_performance = trader_driver.get_all_strategy_performance()
                strategy_filename = f"app/visualization/plots/strategy_performance_{asset}_{exchange['name'].value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                strategy_chart = create_strategy_performance_chart(strategy_performance=strategy_performance, title=f"Strategy Performance Comparison - {asset}", top_n=20)
                strategy_chart.write_html(strategy_filename)

                # Gather recommended action for email/log
                action_line = f"{datetime.now()} | {exchange['name'].value} | {asset} | Action: {signal['action']} | Buy %: {signal.get('buy_percentage', '')} | Sell %: {signal.get('sell_percentage', '')}"
                all_actions.append(action_line)

                # Log recommended action to log.txt
                with open(log_file, "a") as outfile:
                    outfile.write(action_line + "\n")

            except Exception as e:
                logger.error(f"Simulation failed for {asset} on {exchange['name']}: {e}")

    # after
    coinbase_crypto_value_after, coinbase_stablecoin_value_after = coinbase_client.portfolio_value
    display_port_msg(v_c=coinbase_crypto_value_after, v_s=coinbase_stablecoin_value_after, before=False)

    binance_crypto_value_after, binance_stablecoin_value_after = binance_client.portfolio_value
    display_port_msg(v_c=binance_crypto_value_after, v_s=binance_stablecoin_value_after, before=False)

    # Send email with all actions
    subject = "Daily Trading Bot Recommendations"
    body = "\n".join(all_actions)
    send_email(subject, body, to_emails=recipient_list, from_email=GMAIL_ADDRESS, app_password=GMAIL_APP_PASSWORD)

    # write to log file
    now = datetime.now()
    with open(log_file, "a") as outfile:
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
    else:
        logger.info("Starting trading bot in one-time mode...")
        main()
