from enum import Enum

# numerical constants
SECONDS_IN_ONE_DAY = 86400

# whether commit transaction
COMMIT = False  # Set to True for actual trading deployment

# debug mode, if True, only test 1 strategy
DEBUG = True

# strategies
# MA-SELVES: use moving averages with lengths equal to MA_LENGTHS, trades would be made using comparisons with themselves (no cross-MA comparison)
# DOUBLE-MA: create several moving averages; trades are made via pair-wise comparison
# Advanced composite strategies combine multiple signals for better decision-making
STRATEGIES = [
    "MA-SELVES",  # Original moving average strategy
    "EXP-MA-SELVES",  # Exponential moving average strategy
    "DOUBLE-MA",  # Double moving average crossover
    "MACD",  # MACD momentum indicator
    "BOLL-BANDS",  # Bollinger Bands volatility strategy
    "RSI",  # Relative Strength Index
    "KDJ",  # KDJ oscillator
    "MA-MACD",  # Combined MA + MACD (requires both signals)
    "MACD-KDJ",  # Combined MACD + KDJ (momentum + oscillator)
    "RSI-BOLL",  # Combined RSI + Bollinger Bands
    "TRIPLE-SIGNAL",  # MA + MACD + RSI (requires 2/3 agreement)
    "CONSERVATIVE-MA",  # MA with multiple period confirmation
    "VOLUME-BREAKOUT",  # Volume-based breakout strategy
    "MULTI-MA-SELVES",  # Multi-timeframe MA confirmation strategy
    "TREND-MA-SELVES",  # Trend-aware MA strategy
    "VOLUME-MA-SELVES",  # Volume-weighted MA strategy
    "ADAPTIVE-MA-SELVES",  # Adaptive tolerance MA strategy
    "MOMENTUM-MA-SELVES",  # Momentum-enhanced MA strategy
    "FEAR-GREED-SENTIMENT",  # Fear & Greed Index sentiment strategy
    "SIMPLE-RECURRING",  # Simple recurring investment strategy
    "WEIGHTED-RECURRING",  # Weighted recurring investment based on MA
]

# simulation configuration
TOL_PCTS = [0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
BUY_PCTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SELL_PCTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

MA_LENGTHS = [6, 12, 30]
EMA_LENGTHS = [6, 12, 26, 30]
BOLLINGER_MAS = [6, 12]
BOLLINGER_TOLS = [2, 3, 4, 5]

# RSI Strategy Configuration
RSI_PERIODS = [7, 14, 21]  # RSI calculation periods
RSI_OVERSOLD_THRESHOLDS = [10, 20, 30]  # Oversold thresholds (buy signals)
RSI_OVERBOUGHT_THRESHOLDS = [70, 80, 90]  # Overbought thresholds (sell signals)

# KDJ Strategy Configuration
KDJ_OVERSOLD_THRESHOLDS = [10, 20, 30]  # Oversold thresholds (buy signals)
KDJ_OVERBOUGHT_THRESHOLDS = [70, 80, 90]  # Overbought thresholds (sell signals)

# Fear & Greed Index Strategy Configuration
FEAR_GREED_BUY_THRESHOLDS = [20, 30]  # Buy when index is below these values (fear/extreme fear)
FEAR_GREED_SELL_THRESHOLDS = [70, 80]  # Sell when index is above these values (greed/extreme greed)
FEAR_GREED_STRATEGY_NAME = "FEAR-GREED-SENTIMENT"  # Strategy name for identification

# Recurring Investment Strategy Configuration
RECURRING_INVESTMENT_INTERVALS = [3, 7, 14, 30]  # Days between investments
RECURRING_PROFIT_THRESHOLDS = [0.05, 0.10, 0.15]  # Profit thresholds for position reduction
RECURRING_LOSS_THRESHOLDS = [0.05, 0.10, 0.15]  # Loss thresholds for position increase
RECURRING_BASE_INVESTMENT_PCTS = [0.03, 0.05, 0.10]  # Base investment percentages
RECURRING_MA_WEIGHT_FACTORS = [1.0, 2.0, 3.0]  # MA weight factors for weighted strategy

# Trading execution strategies
BUY_STAS = ["by_percentage", "fixed_amount", "market_order"]
SELL_STAS = ["by_percentage", "stop_loss", "take_profit"]

# currencies (crptocurrency + stablecoin + fiat)
# Updated to match actual account holdings
CURS = [
    "SOL", "UNI", "LTC", "ETH", "ETC", "DOGE", "AAVE",
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

# constants
DEPOSIT_CST = "DEPOSIT"
WITHDRAW_CST = "WITHDRAWAL"
ROUND_PRECISION = 3
DEA_NUM_OF_DAYS = 9

# epilson
EP_COIN = 10e-3
EP_CASH = 5

# last X days of data to be considered; time span
TIMESPAN = 120

# Simulation Configuration
SIMULATION_METHOD = (
    "PORTFOLIO_SCALED"  # Options: "FIXED", "PORTFOLIO_SCALED", "PERCENTAGE_BASED"
)
SIMULATION_BASE_AMOUNT = 10000  # Standard simulation amount for scaling
SIMULATION_PERCENTAGE = 0.1  # Use 10% of actual portfolio for percentage-based method

# Log file path
LOG_FILE = "./log.txt"

class ExchangeName(Enum):
    COINBASE = "Coinbase"
    BINANCE = "Binance"

# --- Exchange configuration for unified simulation --- # 
EXCHANGE_CONFIGS = [
    {
        "name":             ExchangeName.COINBASE,
        "symbol_format":    lambda asset: f"{asset}-USD",
        "wallet_func":      lambda client, asset: client.get_wallets(cur_names=[asset]),
        "coin_key":         "available_balance",
        "coin_value_key":   "value",
        "asset_key":        "currency",
    },
    {
        "name":             ExchangeName.BINANCE,
        "symbol_format":    lambda asset: f"{asset}USDT",
        "wallet_func":      lambda client, asset: client.get_wallets(asset_names=[asset]),
        "coin_key":         "free",
        "coin_value_key":   None,
        "asset_key":        "asset",
    },
]