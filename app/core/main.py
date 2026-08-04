# built-in packages
import configparser
import os
import sys
import time
from datetime import datetime
from typing import Optional

# third-party packages
import schedule

# customized packages
try:
    from core.config import *
    from core.logger import get_logger
    from data.defi_event_client import DefiEventClient
    from data.fear_greed_client import FearGreedClient
    from db.database import db_manager
    from repositories.signal_ledger_repository import SignalLedgerRepository
    from services.daily_recommendation_renderer import send_daily_recommendations_email
    from services.market_client_factory import MarketClientFactory
    from services.trader_driver_factory import TraderDriverFactory
    from runners.stock_simulation_runner import StockSimulationRunner
    from runners.crypto_simulation_runner import (  # noqa: F401
        CryptoSimulationRunner,
        fetch_historical_data_with_fallback,  # noqa: F401 - compatibility export
        fetch_intraday_data_with_fallback,  # noqa: F401 - compatibility export
    )
    from services.notification_service import NotificationService
    from services.simulation_service import SimulationService
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from trading.us_stock_client import USStockClient
    from utils.util import display_port_msg
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.config import *
    from core.logger import get_logger
    from data.defi_event_client import DefiEventClient
    from data.fear_greed_client import FearGreedClient
    from db.database import db_manager
    from repositories.signal_ledger_repository import SignalLedgerRepository
    from services.daily_recommendation_renderer import send_daily_recommendations_email
    from services.market_client_factory import MarketClientFactory
    from services.trader_driver_factory import TraderDriverFactory
    from runners.stock_simulation_runner import StockSimulationRunner
    from runners.crypto_simulation_runner import (  # noqa: F401
        CryptoSimulationRunner,
        fetch_historical_data_with_fallback,  # noqa: F401 - compatibility export
        fetch_intraday_data_with_fallback,  # noqa: F401 - compatibility export
    )
    from services.notification_service import NotificationService
    from services.simulation_service import SimulationService
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from trading.us_stock_client import USStockClient
    from utils.util import display_port_msg

logger = get_logger(__name__)

signal_ledger_repository = SignalLedgerRepository(db_manager)
simulation_service = SimulationService(
    signal_ledger_repository,
    strategy_version=SIGNAL_LEDGER_STRATEGY_VERSION,
    bootstrap_days=SIGNAL_LEDGER_BOOTSTRAP_DAYS,
    logger=logger,
)
market_client_factory = MarketClientFactory(CBProClient, BinanceClient, USStockClient)
trader_driver_factory = TraderDriverFactory(
    TraderDriver,
    {
        "tol_pcts": TOL_PCTS,
        "ma_lengths": MA_LENGTHS,
        "ema_lengths": EMA_LENGTHS,
        "bollinger_mas": BOLLINGER_MAS,
        "bollinger_tols": BOLLINGER_TOLS,
        "buy_stas": BUY_STAS,
        "sell_stas": SELL_STAS,
        "rsi_periods": RSI_PERIODS,
        "rsi_oversold_thresholds": RSI_OVERSOLD_THRESHOLDS,
        "rsi_overbought_thresholds": RSI_OVERBOUGHT_THRESHOLDS,
        "kdj_oversold_thresholds": KDJ_OVERSOLD_THRESHOLDS,
        "kdj_overbought_thresholds": KDJ_OVERBOUGHT_THRESHOLDS,
        "mode": "normal",
    },
)

stock_simulation_runner = StockSimulationRunner(
    logger=logger,
    market_client_factory=market_client_factory,
    trader_driver_factory=trader_driver_factory,
    simulation_service=simulation_service,
)

crypto_simulation_runner = CryptoSimulationRunner(
    logger=logger,
    trader_driver_factory=trader_driver_factory,
    simulation_service=simulation_service,
)

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

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


def _send_daily_recommendations_with_ledger(best_summaries=None) -> bool:
    """Send pending ledger signals and persist the admin delivery outcome."""
    service = NotificationService(
        signal_ledger_repository, send_daily_recommendations_email, logger
    )
    return service.send_daily(
        log_file=LOG_FILE,
        recipient_list=RECIPIENT_LIST,
        from_email=GMAIL_ADDRESS,
        app_password=GMAIL_APP_PASSWORD,
        best_summaries=best_summaries,
    )


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
        logger.warning(
            "DEFI event client email not sent: missing credentials in secret.ini"
        )


def _run_stock_simulation(
    all_actions: list, best_summaries: Optional[list] = None
) -> None:
    """Backward-compatible wrapper around StockSimulationRunner."""
    return stock_simulation_runner.run(all_actions, best_summaries)


def main(asset: str = "all", send_email: bool = True):
    """
    Run simulation and make trades.

    Args:
        asset (str): "crypto" | "stock" | "all" (default "all").
        send_email (bool): Whether to send recommendation email after simulation.

    Returns:
        None

    Raises:
        Exception: If there is an error during portfolio value retrieval or trading simulation.
    """

    asset_mode = (asset or "all").strip().lower()
    if asset_mode not in ("crypto", "stock", "all"):
        raise ValueError(f"Unknown asset mode: {asset!r}. Use crypto|stock|all.")

    logger.info(f"COMMIT is set to {COMMIT}")
    logger.info(f"Asset mode: {asset_mode}")

    # Collect recommended actions for email/log
    all_actions = []
    best_summaries = []

    # Stock-only mode: skip crypto clients entirely.
    if asset_mode == "stock":
        _run_stock_simulation(all_actions, best_summaries)

        # Send daily recommendations email if not in debug mode
        if DEBUG is False and send_email:
            _send_daily_recommendations_with_ledger(best_summaries)

        # write to log file
        now = datetime.now()
        with open(LOG_FILE, "a") as outfile:
            outfile.write("Finish job at time {}\n\n".format(str(now)))
        return

    # initialise different clients
    market_clients = market_client_factory.crypto(
        coinbase_key=CB_API_KEY,
        coinbase_secret=CB_API_SECRET,
        binance_key=BINANCE_API_KEY,
        binance_secret=BINANCE_API_SECRET,
    )
    coinbase_client, binance_client = market_clients.coinbase, market_clients.binance

    # Fear & Greed Index client for market sentiment data
    fear_greed_client = FearGreedClient()

    # Get actual portfolio values for simulation
    try:
        logger.info("Getting portfolio value (Coinbase)...")
        portfolio_result = coinbase_client.portfolio_value

        if isinstance(portfolio_result, tuple) and len(portfolio_result) == 2:
            coinbase_crypto_value, coinbase_stablecoin_value = portfolio_result
            logger.info(
                f"Coinbase portfolio: crypto=${coinbase_crypto_value}, stable=${coinbase_stablecoin_value}"
            )
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
        if (
            isinstance(binance_portfolio_result, tuple)
            and len(binance_portfolio_result) == 2
        ):
            binance_crypto_value, binance_stablecoin_value = binance_portfolio_result
            logger.info(
                f"Binance portfolio: crypto=${binance_crypto_value}, stable=${binance_stablecoin_value}"
            )
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

    crypto_simulation_runner.run(
        all_actions=all_actions,
        best_summaries=best_summaries,
        binance_client=binance_client,
        coinbase_client=coinbase_client,
        exchanges=exchanges,
    )

    # after
    try:
        portfolio_result_after = coinbase_client.portfolio_value
        if (
            isinstance(portfolio_result_after, tuple)
            and len(portfolio_result_after) == 2
        ):
            coinbase_crypto_value_after, coinbase_stablecoin_value_after = (
                portfolio_result_after
            )
        else:
            logger.error(
                f"Unexpected portfolio result format after: {portfolio_result_after}"
            )
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
        if (
            isinstance(binance_portfolio_result_after, tuple)
            and len(binance_portfolio_result_after) == 2
        ):
            binance_crypto_value_after, binance_stablecoin_value_after = (
                binance_portfolio_result_after
            )
        else:
            logger.error(
                f"Unexpected Binance portfolio result format after: {binance_portfolio_result_after}"
            )
            binance_crypto_value_after, binance_stablecoin_value_after = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error getting Binance portfolio value after: {e}")
        binance_crypto_value_after, binance_stablecoin_value_after = 0.0, 0.0

    display_port_msg(
        v_c=binance_crypto_value_after, v_s=binance_stablecoin_value_after, before=False
    )

    # --- Stock Trading Simulation ---
    if asset_mode in ("all", "stock"):
        _run_stock_simulation(all_actions, best_summaries)

    # Send daily recommendations email if not in debug mode
    if DEBUG is False and send_email:
        _send_daily_recommendations_with_ledger(best_summaries)

    # Send DEFI report email based on configuration
    if DEFI_MONITORING_ENABLED:
        current_day = datetime.now().weekday()  # Monday=0, Sunday=6
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        if current_day in DEFI_MONITORING_DAYS:
            logger.info(
                f"DEFI monitoring day detected ({day_names[current_day]}) - running DEFI monitoring"
            )
            main_defi()
        else:
            logger.info(
                f"DEFI monitoring skipped - today is {day_names[current_day]} (runs on: {[day_names[d] for d in DEFI_MONITORING_DAYS]})"
            )
    else:
        logger.info("DEFI monitoring disabled in configuration")

    # write to log file
    now = datetime.now()
    with open(LOG_FILE, "a") as outfile:
        outfile.write("Finish job at time {}\n\n".format(str(now)))


def run_trading_job():
    # Default behavior: run both crypto + stocks.
    main(asset="all")


if __name__ == "__main__":
    import sys

    asset_mode = "all"
    send_email = "--no-email" not in sys.argv[1:]
    for arg in sys.argv[1:]:
        if arg.startswith("--asset="):
            asset_mode = arg.split("=", 1)[1].strip().lower()

    if len(sys.argv) > 1 and sys.argv[1] == "--cronjob":
        logger.info("Starting trading bot in schedule-based cronjob mode...")
        # Schedule the job for 1:00 PM UTC (9:00 PM SGT)
        schedule.every().day.at("13:00").do(
            lambda: main(asset=asset_mode, send_email=send_email)
        )
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
        _send_daily_recommendations_with_ledger()

    else:
        logger.info("Starting trading bot in one-time mode...")
        main(asset=asset_mode, send_email=send_email)
