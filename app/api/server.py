"""
Flask web server for cryptocurrency trading bot.
Provides API endpoints for monitoring, configuration, and execution.
"""

import configparser
import datetime
import json
import os
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request, send_from_directory

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import trading bot components
try:
    from core.config import *
    from core.logger import get_logger
    from db.database import db_manager
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg
    from visualization.visualization import (
        create_comprehensive_dashboard,
        create_strategy_performance_chart,
    )
except ImportError:
    # Fallback for when running as script
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.config import *
    from core.logger import get_logger
    from db.database import db_manager
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import calculate_simulation_amounts, display_port_msg
    from visualization.visualization import (
        create_comprehensive_dashboard,
        create_strategy_performance_chart,
    )

app = Flask(__name__)
logger = get_logger(__name__)

# Global state
trading_state = {
    "is_running": False,
    "last_run": None,
    "current_status": "idle",
    "error_message": None,
    "results": {},
    "config": {},
}

# Global client instance
client = None


def initialize_client():
    """Initialize the Coinbase client with API credentials."""
    global client

    try:
        # Read API credentials from secret.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "..", "core", "secret.ini"))

        CB_API_KEY = config["CONFIG"]["COINBASE_API_KEY"].strip('"')
        CB_API_SECRET = config["CONFIG"]["COINBASE_API_SECRET"].strip('"')

        client = CBProClient(key=CB_API_KEY, secret=CB_API_SECRET)
        logger.info("Coinbase client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        trading_state["error_message"] = str(e)
        return False


def run_trading_simulation():
    """Run the trading simulation in a separate thread."""
    global trading_state, client

    if trading_state["is_running"]:
        logger.warning("Trading simulation already running")
        return

    trading_state["is_running"] = True
    trading_state["current_status"] = "running"
    trading_state["error_message"] = None
    trading_state["results"] = {}

    try:
        logger.info("Starting trading simulation...")

        # Get portfolio value
        try:
            logger.info("Getting portfolio value...")
            v_c1, v_s1 = client.portfolio_value
            logger.info(f"Portfolio value before, crypto={v_c1}, stablecoin={v_s1}")
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            trading_state["error_message"] = f"Portfolio error: {e}"
            return

        display_port_msg(v_c=v_c1, v_s=v_s1, before=True)

        results = {}

        for index, cur_name in enumerate(CURS):
            logger.info(f"[{index+1}] processing for currency={cur_name}...")

            try:
                cur_rate = client.get_cur_rate(name=cur_name + "-USD")
                data_stream = client.get_historic_data(name=cur_name + "-USD")

                # cut-off, only want the last X days of data
                data_stream = data_stream[-TIMESPAN:]
                logger.info(f"only want the latest {TIMESPAN} days of data!")

                # initial cash amount
                _, cash = client.portfolio_value
                # initial coin at hand
                wallet = client.get_wallets(cur_names=[cur_name])

                cur_coin, wallet_id = None, None
                for item in wallet:
                    if (
                        item["currency"] == cur_name
                        and item["type"] == "ACCOUNT_TYPE_CRYPTO"
                    ):
                        cur_coin, wallet_id = (
                            float(item["available_balance"]["value"]),
                            item["uuid"],
                        )

                logger.info("cur_coin={:.3f}, wallet_id={}".format(cur_coin, wallet_id))
                assert (
                    cur_coin is not None and wallet_id is not None
                ), f"cannot find relevant wallet for {cur_name}!"

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
                    percentage=SIMULATION_PERCENTAGE
                )
                
                logger.info(f"Simulation setup ({SIMULATION_METHOD}): cash=${sim_cash:.2f}, coin={sim_coin:.6f}")

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

                t_driver.feed_data(data_stream)

                info = t_driver.best_trader_info
                # best trader
                best_t = t_driver.traders[info["trader_index"]]

                # for a new price, find the trade signal
                new_p = client.get_cur_rate(cur_name + "-USD")
                today = datetime.datetime.now().strftime("%m/%d/%Y")
                # add it and compute
                best_t.add_new_day(
                    new_p=new_p,
                    d=today,
                    misc_p={"open": new_p, "low": new_p, "high": new_p},
                )
                signal = best_t.trade_signal

                # Generate visualizations
                try:
                    logger.info("Generating trading visualizations...")

                    # Ensure plots directory exists
                    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Create comprehensive dashboard (relative to project root)
                    dashboard_filename = f"app/visualization/plots/trading_dashboard_{cur_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    dashboard = create_comprehensive_dashboard(
                        trader_instance=best_t,
                        save_html=True,
                        filename=dashboard_filename,
                    )
                    logger.info(f"Dashboard saved to: {dashboard_filename}")

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

                # Store results
                results[cur_name] = {
                    "best_trader_info": info,
                    "max_drawdown": best_t.max_drawdown * 100,
                    "num_transaction": best_t.num_transaction,
                    "num_buy_action": best_t.num_buy_action,
                    "num_sell_action": best_t.num_sell_action,
                    "high_strategy": best_t.high_strategy,
                    "signal": signal,
                    "dashboard_file": dashboard_filename
                    if "dashboard_filename" in locals()
                    else None,
                }

            except Exception as e:
                logger.error(f"Error processing {cur_name}: {e}")
                results[cur_name] = {"error": str(e)}

        # Final portfolio value
        try:
            v_c2, v_s2 = client.portfolio_value
            display_port_msg(v_c=v_c2, v_s=v_s2, before=False)
            results["final_portfolio"] = {"crypto": v_c2, "stablecoin": v_s2}
        except Exception as e:
            logger.error(f"Error getting final portfolio: {e}")

        trading_state["results"] = results
        trading_state["last_run"] = datetime.datetime.now().isoformat()
        trading_state["current_status"] = "completed"

        logger.info("Trading simulation completed successfully")

    except Exception as e:
        logger.error(f"Trading simulation failed: {e}")
        trading_state["error_message"] = str(e)
        trading_state["current_status"] = "failed"
        traceback.print_exc()

    finally:
        trading_state["is_running"] = False


# API Routes


@app.route("/")
def index():
    """Main dashboard page."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Trading Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.running { background: #d4edda; color: #155724; }
            .status.completed { background: #d1ecf1; color: #0c5460; }
            .status.failed { background: #f8d7da; color: #721c24; }
            .status.idle { background: #e2e3e5; color: #383d41; }
            .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .button:hover { background: #0056b3; }
            .button:disabled { background: #6c757d; cursor: not-allowed; }
            .results { margin-top: 20px; }
            .currency-result { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Crypto Trading Bot Dashboard</h1>
                <p>Monitor and control your automated trading strategies</p>
            </div>
            
            <div class="status {{ trading_state.current_status }}">
                <h3>Status: {{ trading_state.current_status.upper() }}</h3>
                {% if trading_state.is_running %}
                    <p>🔄 Trading simulation is currently running...</p>
                {% elif trading_state.last_run %}
                    <p>📅 Last run: {{ trading_state.last_run }}</p>
                {% endif %}
                {% if trading_state.error_message %}
                    <p class="error">❌ Error: {{ trading_state.error_message }}</p>
                {% endif %}
            </div>
            
            <div>
                <button class="button" onclick="startTrading()" {% if trading_state.is_running %}disabled{% endif %}>
                    🚀 Start Trading Simulation
                </button>
                <button class="button" onclick="getStatus()">
                    🔄 Refresh Status
                </button>
                <button class="button" onclick="getResults()">
                    📊 View Results
                </button>
            </div>
            
            <div id="results" class="results"></div>
        </div>
        
        <script>
            function startTrading() {
                fetch('/api/start', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        setTimeout(getStatus, 1000);
                    });
            }
            
            function getStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        location.reload();
                    });
            }
            
            function getResults() {
                fetch('/api/results')
                    .then(response => response.json())
                    .then(data => {
                        displayResults(data);
                    });
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                if (!data.results || Object.keys(data.results).length === 0) {
                    resultsDiv.innerHTML = '<p>No results available yet.</p>';
                    return;
                }
                
                let html = '<h3>📈 Trading Results</h3>';
                
                for (const [currency, result] of Object.entries(data.results)) {
                    if (currency === 'final_portfolio') continue;
                    
                    html += '<div class="currency-result">';
                    html += `<h4>${currency}</h4>`;
                    
                    if (result.error) {
                        html += `<p class="error">Error: ${result.error}</p>`;
                    } else {
                        html += `<p><strong>Strategy:</strong> ${result.high_strategy}</p>`;
                        html += `<p><strong>Max Drawdown:</strong> ${result.max_drawdown.toFixed(2)}%</p>`;
                        html += `<p><strong>Transactions:</strong> ${result.num_transaction}</p>`;
                        html += `<p><strong>Buys:</strong> ${result.num_buy_action}, <strong>Sells:</strong> ${result.num_sell_action}</p>`;
                        html += `<p><strong>Current Signal:</strong> ${result.signal.action}</p>`;
                        if (result.dashboard_file) {
                            html += `<p><a href="/plots/${result.dashboard_file.split('/').pop()}" target="_blank">📊 View Dashboard</a></p>`;
                        }
                    }
                    html += '</div>';
                }
                
                if (data.results.final_portfolio) {
                    html += '<div class="currency-result">';
                    html += '<h4>💰 Final Portfolio</h4>';
                    html += `<p><strong>Crypto:</strong> $${data.results.final_portfolio.crypto}</p>`;
                    html += `<p><strong>Stablecoin:</strong> $${data.results.final_portfolio.stablecoin}</p>`;
                    html += '</div>';
                }
                
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, trading_state=trading_state)


@app.route("/api/status")
def get_status():
    """Get current trading bot status."""
    return jsonify(trading_state)


@app.route("/api/start", methods=["POST"])
def start_trading():
    """Start trading simulation."""
    if trading_state["is_running"]:
        return jsonify({"message": "Trading simulation already running"}), 400

    if not client:
        return jsonify({"message": "Client not initialized"}), 500

    # Start trading in a separate thread
    thread = threading.Thread(target=run_trading_simulation)
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Trading simulation started"})


@app.route("/api/results")
def get_results():
    """Get trading results."""
    return jsonify(trading_state)


@app.route("/api/config")
def get_config():
    """Get current configuration."""
    config = {
        "currencies": CURS,
        "strategies": STRATEGIES,
        "timespan": TIMESPAN,
        "commit": COMMIT,
    }
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update configuration (basic implementation)."""
    data = request.get_json()
    # Note: This is a basic implementation. In production, you'd want to
    # properly validate and persist configuration changes.
    return jsonify({"message": "Configuration update not implemented yet"})


@app.route("/plots/<filename>")
def serve_plot(filename):
    """Serve plot files."""
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "visualization", "plots")
    return send_from_directory(plots_dir, filename)


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "client_initialized": client is not None,
        }
    )


@app.route("/api/database/stats")
def get_database_stats():
    """Get database statistics."""
    try:
        stats = db_manager.get_data_statistics()
        return jsonify({
            "status": "success",
            "data": [
                {
                    "symbol": symbol,
                    "record_count": count,
                    "last_updated": last_updated.isoformat() if last_updated else None
                }
                for symbol, count, last_updated in stats
            ]
        })
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/database/clear", methods=["POST"])
def clear_old_data():
    """Clear old data from database."""
    try:
        data = request.get_json() or {}
        days = data.get("days", 365)
        deleted_count = db_manager.clear_old_data(days)
        return jsonify({
            "status": "success",
            "message": f"Deleted {deleted_count} records older than {days} days"
        })
    except Exception as e:
        logger.error(f"Error clearing old data: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def main():
    """Main function for Poetry script entry point."""
    # Initialize client on startup
    if initialize_client():
        logger.info("Server starting...")
        app.run(host="0.0.0.0", port=8000, debug=False)
    else:
        logger.error("Failed to initialize client. Server not started.")
        sys.exit(1)


if __name__ == "__main__":
    main()
