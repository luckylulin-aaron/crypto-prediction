from enum import Enum

# numerical constants
SECONDS_IN_ONE_DAY = 86400

# whether commit transaction
COMMIT = False  # Set to True for actual trading deployment

# debug mode, if True, only test 1 strategy
DEBUG = False

# --- Strategy lists ---
# We distinguish:
# - SUPPORTED_STRATEGIES: strategies implemented by the codebase (used for validation/asserts)
# - CRYPTO_STRATEGIES: enabled strategies for crypto simulations/trading
# - STOCK_STRATEGIES: enabled strategies for stock simulations/trading (daily candles only)
#
# NOTE: We keep `STRATEGIES` as a backward-compatible alias to `CRYPTO_STRATEGIES` because many scripts/tests
# pass it into `TraderDriver(overall_stats=...)`.
SUPPORTED_STRATEGIES = [
    "MA-SELVES",
    "MA-SELVES-MACRO",
    "EXP-MA-SELVES",
    "EXP-MA-SELVES-MACRO",
    "RSI",
    "RSI-MACRO",
    "ADAPTIVE-MA-SELVES",
    "ADAPTIVE-MA-SELVES-MACRO",
    "DOUBLE-MA",
    "MACD",
    "BOLL-BANDS",
    "MA-BOLL-BANDS",
    "KDJ",
    "MA-MACD",
    "MACD-KDJ",
    "RSI-BOLL",
    "TRIPLE-SIGNAL",
    "CONSERVATIVE-MA",
    "VOLUME-BREAKOUT",
    "MULTI-MA-SELVES",
    "TREND-MA-SELVES",
    "VOLUME-MA-SELVES",
    "MOMENTUM-MA-SELVES",
    "FEAR-GREED-SENTIMENT",
    "SIMPLE-RECURRING",
    "WEIGHTED-RECURRING",
    "ECONOMIC-INDICATORS",
]

# Enabled strategies (crypto)
CRYPTO_STRATEGIES = [
    "MA-BOLL-BANDS",
]

# Enabled strategies (stocks) - daily candles only; keep separate from crypto
STOCK_STRATEGIES = [
    "MA-SELVES",  # Original moving average strategy
    "DOUBLE-MA", # Double Moving Average Crossover strategy
    "MA-BOLL-BANDS", # MA + Bollinger Bands strategy
    "RSI", # RSI strategy
    "KDJ", # KDJ strategy
    # --------- #
    "ADAPTIVE-MA-SELVES",  # Adaptive tolerance MA strategy
]


# Backward-compatible alias (treat as enabled crypto strategies)
STRATEGIES = CRYPTO_STRATEGIES

# simulation configuration
TOL_PCTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
BUY_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9]
SELL_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9]

MA_LENGTHS = [6, 12, 30]
EMA_LENGTHS = [6, 12, 26, 30]
BOLLINGER_MAS = [6, 12]
BOLLINGER_TOLS = [2, 3, 4, 5]

# --- Moving-window best strategy selection (aggregated across windows) ---
# Recommended: "risk_adjusted_return" (mean - std) for stability.
# Other options: "mean_rate_of_return" (more aggressive).
MOVING_WINDOW_BEST_STRATEGY_METRIC = "risk_adjusted_return"
# Minimum number of windows a strategy must win to be eligible; prevents picking a strategy that
# happened to win only once due to noise.
MOVING_WINDOW_BEST_STRATEGY_MIN_WINS = 2

# --- Moving-window parameter auto-tuning (walk-forward) ---
# If True, for window i we narrow the parameter search space around the best parameters observed in
# previous window(s). This lets strategies "adapt" over time without leaking future information.
MOVING_WINDOW_AUTO_TUNE_PARAMS = True
# How many adjacent values (on each side) to include around the last best parameter in the search space.
MOVING_WINDOW_TUNE_NEIGHBORHOOD = 1

# RSI Strategy Configuration
RSI_PERIODS = [7, 14, 21]  # RSI calculation periods
RSI_OVERSOLD_THRESHOLDS = [10, 20, 30]  # Oversold thresholds (buy signals)
RSI_OVERBOUGHT_THRESHOLDS = [70, 80, 90]  # Overbought thresholds (sell signals)

# KDJ Strategy Configuration
KDJ_OVERSOLD_THRESHOLDS = [10, 20, 30]  # Oversold thresholds (buy signals)
KDJ_OVERBOUGHT_THRESHOLDS = [70, 80, 90]  # Overbought thresholds (sell signals)

# Fear & Greed Index Strategy Configuration
FEAR_GREED_BUY_THRESHOLDS = [
    20,
    30,
]  # Buy when index is below these values (fear/extreme fear)
FEAR_GREED_SELL_THRESHOLDS = [
    70,
    80,
]  # Sell when index is above these values (greed/extreme greed)
FEAR_GREED_STRATEGY_NAME = "FEAR-GREED-SENTIMENT"  # Strategy name for identification

# Recurring Investment Strategy Configuration
RECURRING_INVESTMENT_INTERVALS = [3, 7, 14, 30]  # Days between investments
RECURRING_PROFIT_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
]  # Profit thresholds for position reduction
RECURRING_LOSS_THRESHOLDS = [0.05, 0.10, 0.15]  # Loss thresholds for position increase
RECURRING_BASE_INVESTMENT_PCTS = [0.03, 0.05, 0.10]  # Base investment percentages
RECURRING_MA_WEIGHT_FACTORS = [1.0, 2.0, 3.0]  # MA weight factors for weighted strategy

# Trading execution strategies
BUY_STAS = ["by_percentage", "fixed_amount", "market_order"]
SELL_STAS = ["by_percentage", "stop_loss", "take_profit"]

# currencies (crptocurrency + stablecoin + fiat)
# Updated to match actual account holdings
CURS = [
    "ETH",
    "BTC",
    "SOL",
    "UNI",
    "LTC",
    "ETC",
    "DOGE",
    "AAVE",
]  # Tradeable cryptocurrencies with balance
# Note: ETH2 exists in account but ETH2-USDT is not a valid trading pair
# LTC and ADA are not in the account
FIAT = ["USD", "SGD"]  # Fiat currencies (not used for trading)
STABLECOIN = ["USDC", "USDT"]  # Stablecoins for trading with cryptos
# Primary trading currency: USDT

# signals
NO_ACTION_SIGNAL = "NO ACTION"
BUY_SIGNAL = "BUY"
SELL_SIGNAL = "SELL"

# --- Options backtest heuristics (signal-based) ---
# For each BUY signal, we assume "open a call option" and consider it a win if we later see a SELL
# within this horizon at a higher price than the BUY price. Symmetric for SELL->BUY (put option).
# This is a simple, execution-free proxy to evaluate whether your signals are directionally useful.
OPTION_SIGNAL_HOLD_DAYS = 20

# constants
DEPOSIT_CST = "DEPOSIT"
WITHDRAW_CST = "WITHDRAWAL"
ROUND_PRECISION = 3
DEA_NUM_OF_DAYS = 9

# epilson
EP_COIN = 10e-3
EP_CASH = 5

# last X days of data to be considered; time span
TIMESPAN = 90

# Data interval granularity (in hours)
# Controls the frequency of data points fetched from exchanges
# Options: 1 (hourly), 6 (6-hourly), 12 (12-hourly), 24 (daily), etc.
DATA_INTERVAL_HOURS = 6 # 12-hour intervals for finer granularity

# Moving window configuration for simulation
# For a given TIMESPAN, use overlapping windows of MOVING_WINDOW_DAYS to run multiple simulations
# and aggregate results for more robust performance evaluation
MOVING_WINDOW_DAYS = 60  # Size of each moving window (should be <= TIMESPAN)
MOVING_WINDOW_STEP = 3  # Step size for moving window (1 = overlapping windows, MOVING_WINDOW_DAYS = non-overlapping)

# Simulation Configuration
SIMULATION_METHOD = (
    "PORTFOLIO_SCALED"  # Options: "FIXED", "PORTFOLIO_SCALED", "PERCENTAGE_BASED"
)
SIMULATION_BASE_AMOUNT = 10000  # Standard simulation amount for scaling
SIMULATION_PERCENTAGE = 0.1  # Use 10% of actual portfolio for percentage-based method
DEFAULT_SIMULATION_COIN_AMOUNT = 1.0  # Default coin amount to use for simulation when asset is not found in wallet

# Log file path
LOG_FILE = "./log.txt"

# DEFI Monitoring Configuration
# Configure which days of the week DEFI monitoring should run
# 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
# Example: [6] = Sundays only, [0, 6] = Mondays and Sundays, [0,1,2,3,4,5,6] = Every day
DEFI_MONITORING_DAYS = [6]  # Days of week to run DEFI monitoring (0=Monday, 6=Sunday)
DEFI_MONITORING_ENABLED = True  # Enable/disable DEFI monitoring


class ExchangeName(Enum):
    COINBASE = "Coinbase"
    BINANCE = "Binance"


# --- Exchange configuration for unified simulation --- #
EXCHANGE_CONFIGS = [
    {
        "name": ExchangeName.COINBASE,
        "symbol_format": lambda asset: f"{asset}-USD",
        "wallet_func": lambda client, asset: client.get_wallets(cur_names=[asset]),
        "coin_key": "available_balance",
        "coin_value_key": "value",
        "asset_key": "currency",
    },
    {
        "name": ExchangeName.BINANCE,
        "symbol_format": lambda asset: f"{asset}USDT",
        "wallet_func": lambda client, asset: client.get_wallets(asset_names=[asset]),
        "coin_key": "free",
        "coin_value_key": None,
        "asset_key": "asset",
    },
]

# List of US stock tickers to fetch via yfinance
STOCKS = [
    'AAPL',  # Apple
    'TSLA',  # Tesla
    'JD', # JD.com
    'ORCL', # Oracle,
    'META', # Meta,
    'MSFT', # Microsoft
    'NVDA', # Nvidia
    'AMZN', # Amazon
    'GOOGL', # Google
    'NFLX', # Netflix
    'SNAP', # Snapchat
    'BIDU', # Baidu
    'BABA', # Alibaba
    'TECHY', # Tencent
    'UBER', # Uber
    'DASH' # Doordash
]
