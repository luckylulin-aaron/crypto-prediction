# built-in packages
import configparser
import datetime
import json
import os
import sys
import time
from curses import nocbreak

# third-party packages
import numpy as np
import pandas as pd
import schedule

# customized packages
try:
    from core.config import *
    from core.logger import get_logger
    from data.fear_greed_client import FearGreedClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg, load_csv
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
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg, load_csv
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

    # cbpro client to interact with coinbase
    client = CBProClient(key=CB_API_KEY, secret=CB_API_SECRET)

    # Fear & Greed Index client for market sentiment data
    fear_greed_client = FearGreedClient()

    # Get actual portfolio values for simulation
    try:
        logger.info("Getting portfolio value...")
        v_c1, v_s1 = client.portfolio_value
        logger.info(f"Portfolio value before, crypto={v_c1}, stablecoin={v_s1}")
    except Exception as e:
        logger.error(f"Error getting portfolio value: {e}")
        return

    display_port_msg(v_c=v_c1, v_s=v_s1, before=True)

    for index, cur_name in enumerate(CURS):
        logger.info(f"\n\n[{index+1}] processing for currency={cur_name}...")
        cur_rate = client.get_cur_rate(name=cur_name + "-USD")
        data_stream = client.get_historic_data(name=cur_name + "-USD")

        # cut-off, only want the last X days of data
        data_stream = data_stream[-TIMESPAN:]
        logger.info(f"only want the latest {TIMESPAN} days of data!")

        # Fetch Fear & Greed Index data for the same time period
        try:
            fear_greed_data = fear_greed_client.get_historical_fear_greed(days=TIMESPAN)
            logger.info(f"Fetched {len(fear_greed_data)} days of Fear & Greed Index data")
            
            # Log current fear & greed sentiment
            if fear_greed_data:
                current_sentiment = fear_greed_data[0]
                logger.info(f"Current Fear & Greed Index: {current_sentiment['value']} ({current_sentiment['value_classification']})")
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index data: {e}")
            fear_greed_data = []

        # Get actual wallet balances for this currency
        wallet = client.get_wallets(cur_names=[cur_name])
        cur_coin, wallet_id = None, None
        for item in wallet:
            if item["currency"] == cur_name and item["type"] == "ACCOUNT_TYPE_CRYPTO":
                cur_coin, wallet_id = (
                    float(item["available_balance"]["value"]),
                    item["uuid"],
                )
        # check if the wallet is found
        if cur_coin is None or wallet_id is None:
            logger.error(f"cannot find relevant wallet for {cur_name}!")
            return

        logger.info("cur_coin={:.3f}, wallet_id={}".format(cur_coin, wallet_id))

        # Calculate simulation amounts using configurable method
        from core.config import (
            KDJ_OVERBOUGHT_THRESHOLDS,
            KDJ_OVERSOLD_THRESHOLDS,
            RSI_OVERBOUGHT_THRESHOLDS,
            RSI_OVERSOLD_THRESHOLDS,
            RSI_PERIODS,
            SIMULATION_BASE_AMOUNT,
            SIMULATION_METHOD,
            SIMULATION_PERCENTAGE,
        )

        sim_cash, sim_coin = calculate_simulation_amounts(
            actual_cash=v_s1,
            actual_coin=cur_coin,
            method=SIMULATION_METHOD,
            base_amount=SIMULATION_BASE_AMOUNT,
            percentage=SIMULATION_PERCENTAGE,
        )

        logger.info(
            f"Simulation setup ({SIMULATION_METHOD}): cash=${sim_cash:.2f}, coin={sim_coin:.2f}"
        )

        # run simulation with scaled values
        t_driver = TraderDriver(
            name=cur_name,
            init_amount=int(sim_cash),
            overall_stats=STRATEGIES,
            cur_coin=sim_coin,
            tol_pcts=TOL_PCTS,
            ma_lengths=MA_LENGTHS,
            ema_lengths=EMA_LENGTHS,
            bollinger_mas=BOLLINGER_MAS,
            bollinger_tols=BOLLINGER_TOLS,
            buy_pcts=BUY_PCTS,
            sell_pcts=SELL_PCTS,
            buy_stas=list(BUY_STAS)
            if isinstance(BUY_STAS, (list, tuple))
            else [BUY_STAS],
            sell_stas=list(SELL_STAS)
            if isinstance(SELL_STAS, (list, tuple))
            else [SELL_STAS],
            rsi_periods=RSI_PERIODS,
            rsi_oversold_thresholds=RSI_OVERSOLD_THRESHOLDS,
            rsi_overbought_thresholds=RSI_OVERBOUGHT_THRESHOLDS,
            kdj_oversold_thresholds=KDJ_OVERSOLD_THRESHOLDS,
            kdj_overbought_thresholds=KDJ_OVERBOUGHT_THRESHOLDS,
            mode="normal",
        )

        # Set fear & greed data for the trader driver if available
        if fear_greed_data:
            t_driver.set_fear_greed_data(fear_greed_data)

        t_driver.feed_data(data_stream)

        info = t_driver.best_trader_info
        # best trader
        best_t = t_driver.traders[info["trader_index"]]

        # for a new price, find the trade signal
        new_p = client.get_cur_rate(cur_name + "-USDT")
        today = datetime.datetime.now().strftime("%m/%d/%Y")
        # add it and compute
        best_t.add_new_day(
            new_p=new_p, d=today, misc_p={"open": new_p, "low": new_p, "high": new_p}
        )
        signal = best_t.trade_signal

        logger.info(
            f"\n{'★'*10} BEST TRADER SUMMARY {'★'*10}\n"
            f"Best trader performance: {info}\n"
            f"Max drawdown: {best_t.max_drawdown * 100:.2f}%\n"
            f"Transactions: {best_t.num_transaction}, "
            f"Buys: {best_t.num_buy_action}, Sells: {best_t.num_sell_action}\n"
            f"Strategy: {best_t.high_strategy}\n"
            f"Today's signal: {signal} for crypto={best_t.crypto_name}\n"
            f"{'★'*36}\n"
        )

        # Log Fear & Greed Index information
        if fear_greed_data:
            current_sentiment = fear_greed_data[0]
            sentiment_desc = fear_greed_client.get_classification_description(int(current_sentiment['value']))
            logger.info(
                f"\n{'='*10} MARKET SENTIMENT {'='*10}\n"
                f"Fear & Greed Index: {current_sentiment['value']} ({current_sentiment['value_classification']})\n"
                f"Sentiment: {sentiment_desc}\n"
                f"{'='*30}\n"
            )

        # Generate visualizations for the best trader
        try:
            logger.info("Generating trading visualizations...")

            # Create comprehensive dashboard
            dashboard_filename = f"app/visualization/plots/trading_dashboard_{cur_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            dashboard = create_comprehensive_dashboard(
                trader_instance=best_t, save_html=True, filename=dashboard_filename
            )
            logger.info(f"Dashboard saved to: {dashboard_filename}")

            # Create individual portfolio chart
            portfolio_filename = f"app/visualization/plots/portfolio_chart_{cur_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            portfolio_chart = create_portfolio_value_chart(
                trade_history=best_t.trade_history,
                title=f"Portfolio Value - {cur_name} ({best_t.high_strategy})",
            )
            portfolio_chart.write_html(portfolio_filename)
            logger.info(f"Portfolio chart saved to: {portfolio_filename}")

            # Create strategy performance comparison chart
            strategy_performance = t_driver.get_all_strategy_performance()
            strategy_filename = f"app/visualization/plots/strategy_performance_{cur_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            strategy_chart = create_strategy_performance_chart(
                strategy_performance=strategy_performance,
                title=f"Strategy Performance Comparison - {cur_name}",
                top_n=20,
            )
            strategy_chart.write_html(strategy_filename)
            logger.info(f"Strategy performance chart saved to: {strategy_filename}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        # Get current portfolio values for actual trading
        _, current_usdt = client.portfolio_value

        # if too less, we leave
        if current_usdt <= EP_CASH and signal == BUY_SIGNAL:
            logger.warning(
                "too less USDT, cannot execute a buy, discard further actions."
            )
            continue
        elif (
            cur_coin <= EP_COIN or cur_coin * new_p <= EP_CASH
        ) and signal == SELL_SIGNAL:
            logger.warning(
                "too less crypto, cannot execute a sell, discard further actions."
            )
            continue

        # otherwise, execute a transaction
        if signal["action"] == BUY_SIGNAL:
            buy_pct = signal["buy_percentage"]
            buy_amount_usdt = buy_pct * current_usdt
            order = client.place_buy_order(
                wallet_id=wallet_id,
                amount=buy_amount_usdt,
                currency=cur_name,
                commit=COMMIT,
            )
            
            # Handle different response formats (simulated vs real)
            if hasattr(order, 'success_response'):
                # Real order response
                logger.info(
                    f"bought {cur_name} using {buy_amount_usdt:.2f} USDT (order_id: {order.success_response.order_id})"
                )
            else:
                # Simulated order response
                logger.info(
                    "bought {:.5f} {}, used {:.2f} USDT, at unit price={}".format(
                        float(order["amount"]["amount"]),
                        cur_name,
                        float(order["total"]["amount"]),
                        float(order["unit_price"]["amount"]),
                    )
                )

        elif signal["action"] == SELL_SIGNAL:
            sell_pct = signal["sell_percentage"]
            # Use actual crypto balance, not simulation balance
            sell_amount_crypto = sell_pct * cur_coin
            order = client.place_sell_order(
                wallet_id=wallet_id,
                amount=sell_amount_crypto,
                currency=cur_name,
                commit=COMMIT,
            )
            
            # Handle different response formats (simulated vs real)
            if hasattr(order, 'success_response'):
                # Real order response
                estimated_usdt_value = sell_amount_crypto * new_p
                logger.info(
                    f"sold {sell_amount_crypto:.6f} {cur_name} for ~{estimated_usdt_value:.2f} USDT (order_id: {order.success_response.order_id})"
                )
            else:
                # Simulated order response
                logger.info(
                    "sold {:.5f} {}, cash out {:.2f} USDT, at unit price={}".format(
                        float(order["amount"]["amount"]),
                        cur_name,
                        float(order["subtotal"]["amount"]),
                        float(order["unit_price"]["amount"]),
                    )
                )
        elif signal["action"] == NO_ACTION_SIGNAL:
            logger.info("no action performed as simulation suggests.")

    # after
    v_c2, v_s2 = client.portfolio_value
    display_port_msg(v_c=v_c2, v_s=v_s2, before=False)

    # write to log file
    now = datetime.datetime.now()
    with open(log_file, "a") as outfile:
        outfile.write("Finish job at time {}\n".format(str(now)))


if __name__ == "__main__":
    # Check if running as cronjob or one-time
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cronjob":
        # Run as a daily cronjob with following command:
        # ```python app/core/main.py --cronjob```
        logger.info("Starting trading bot in cronjob mode...")
        
        # Schedule daily execution at 9:00 PM SGT (1:00 PM UTC)
        schedule.every().day.at('13:00').do(main)
        
        logger.info("Trading bot scheduled to run daily at 9:00 PM SGT (1:00 PM UTC)")
        logger.info("Press Ctrl+C to stop the bot")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                sys.stdout.flush()
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
    else:
        # Run one-time
        logger.info("Starting trading bot in one-time mode...")
        main()
