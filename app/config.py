# numerical constants
SECONDS_IN_ONE_DAY = 86400

# whether commit transaction
COMMIT = False

# strategies
# MA-SELVES: use moving averages with lengths equal to MA_LENGTHS, trades would be made using comparisons with themselves (no cross-MA comparison)
# DOUBLE-MA: create several moving averages; trades are made via pair-wise comparison
STRATEGIES = ['MA-SELVES', 'DOUBLE-MA', 'MACD', 'BOLL-BANDS', 'COMPOUND', 'KDJ']

# simulation configuration
TOL_PCTS = [0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
BUY_PCTS = [0.3, 0.4, 0.5, 0.6, 0.7]
SELL_PCTS = [0.3, 0.4, 0.5, 0.6, 0.7]

MA_LENGTHS = [10, 20, 30]
EMA_LENGTHS = [12, 26]
BOLLINGER_MAS = [10]
BOLLINGER_TOLS = [2, 3, 4]

BUY_STAS = ('by_percentage')
SELL_STAS = ('by_percentage')

# currencies (crptocurrency + stablecoin + fiat)
CURS = ['SOL', 'ETH']
FIAT = ['USD', 'SGD']
STABLECOIN = ['USDC', 'USDT']  # Stablecoins for trading with cryptos

# signals
NO_ACTION_SIGNAL = 'NO ACTION'
BUY_SIGNAL = 'BUY'
SELL_SIGNAL = 'SELL'

# constants
DEPOSIT_CST = 'DEPOSIT'
WITHDRAW_CST = 'WITHDRAWAL'
ROUND_PRECISION = 3
DEA_NUM_OF_DAYS = 9

# epilson
EP_COIN = 10e-3
EP_CASH = 5

# last X days of data to be considered; time span
TIMESPAN = 30