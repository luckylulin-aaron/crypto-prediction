# Crypto Trading Bot

A sophisticated cryptocurrency trading bot with web interface, built with Python and Flask. The bot uses multiple trading strategies including RSI, KDJ, and Moving Averages to make automated trading decisions on Coinbase Advanced Trade.

## 🚀 Features

- **Multiple Trading Strategies**: 13 strategies including 5 advanced composite strategies and Fear & Greed Index sentiment strategy
- **Bias-Free Simulation**: Configurable simulation methods to eliminate trading bias
- **Web Dashboard**: Interactive Flask-based web interface
- **Real-time Monitoring**: Live status updates and performance tracking
- **Enhanced Visualizations**: Plotly-based charts with winning strategy names and execution strategies
- **API Integration**: Coinbase Advanced Trade API
- **Fear & Greed Index Integration**: Market sentiment analysis using Bitcoin Fear & Greed Index
- **Automated Trading**: Configurable trading parameters and risk management
- **Comprehensive Logging**: Detailed logging and error tracking
- **PostgreSQL Database**: Cached historical data to avoid redundant API calls
- **Data Persistence**: Efficient storage and retrieval of trading data

## 📋 Prerequisites

- Python 3.8 or higher
- Coinbase Advanced Trade API credentials
- Poetry (will be installed automatically)
- PostgreSQL (optional, for data caching)

## 🛠️ Installation

### Quick Start (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd crypto-prediction

# Run the installation script
./install.sh
```

### Manual Installation

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   **Note**: If you see a warning about `pyproject.toml` changes, run:
   ```bash
   poetry lock --no-update
   poetry install
   ```

3. **Configure API credentials**:
   ```bash
   # Copy the template and add your credentials
   cp app/core/secret.ini.template app/core/secret.ini
   ```
   
   Then edit `app/core/secret.ini` with your Coinbase and Binance API credentials:
   ```ini
   [CONFIG]
   COINBASE_API_KEY = "your_actual_api_key_here"
   COINBASE_API_SECRET = "your_actual_api_secret_here"
   BINANCE_API_KEY = "your_binance_api_key_here"
   BINANCE_API_SECRET = "your_binance_api_secret_here"
   ```
   
   ⚠️ **IMPORTANT**: Never commit your actual `secret.ini` file to Git!

4. **Set up PostgreSQL (Optional)**:
   ```bash
   # Option A: Using Docker Compose (if Docker is installed)
   docker-compose up -d postgres
   
   # Option B: Install PostgreSQL locally
   # macOS: brew install postgresql
   # Ubuntu: sudo apt-get install postgresql postgresql-contrib
   # Windows: Download from https://www.postgresql.org/download/windows/
   
   # Then set DATABASE_URL environment variable
   export DATABASE_URL="postgresql://username:password@localhost:5432/crypto_trading"
   ```

## 🎯 Usage

### Virtual Environment Management (Poetry 2.x)

```bash
# Activate the virtual environment
poetry env activate

# Deactivate when done
exit

# Run commands without activating
poetry run <command>
```

### Running the Trading Bot

```bash
# Run the trading bot directly
poetry run python app/core/main.py

# Or activate the virtual environment first
poetry env activate
python app/core/main.py
```

### Starting the Web Server

```bash
# Start the web server
poetry run python start_server.py

# Or manually with activated environment
poetry env activate
python start_server.py
```

The web dashboard will be available at: http://localhost:8000

### Web Dashboard Features

- **Real-time Status**: Monitor trading bot status
- **Interactive Controls**: Start/stop trading simulations
- **Results Display**: View detailed trading results
- **Visualization Links**: Access generated charts and dashboards
- **API Endpoints**: Programmatic access to bot functionality

## 📊 API Endpoints

### Status Endpoints
- `GET /api/status` - Get current bot status
- `GET /health` - Health check

### Control Endpoints
- `POST /api/start` - Start trading simulation
- `GET /api/results` - Get trading results
- `GET /api/config` - Get configuration

### File Serving
- `GET /plots/<filename>` - Serve visualization files

## 🔧 Configuration

### Simulation Methods

The bot supports multiple simulation methods to eliminate trading bias. The original system used fixed values (`fake_cash=3000`, `fake_cur_coin=5`) which created significant biases in performance evaluation.

#### Available Methods

**1. PORTFOLIO_SCALED (Recommended)**
- **Description**: Scales actual portfolio to a standard simulation amount
- **Method**: Calculates ratio: `standard_amount / actual_portfolio_value`, then scales both cash and coin
- **Advantages**:
  - Eliminates bias from arbitrary starting values
  - Maintains realistic portfolio proportions
  - Results are comparable across different portfolio sizes
  - Standardized simulation amount ($10,000 by default)

**2. PERCENTAGE_BASED**
- **Description**: Uses a fixed percentage of actual portfolio
- **Method**: Uses `SIMULATION_PERCENTAGE` of actual cash and coin (default: 10%)
- **Advantages**: Simple, maintains proportions, results scale with portfolio
- **Disadvantages**: Results vary with portfolio size, harder to compare across users

**3. FIXED (Legacy - Biased)**
- **Description**: Uses fixed amounts regardless of actual portfolio
- **Values**: $3,000 cash, 5 coins
- **Problem**: Results are biased by arbitrary starting values
- **Use Case**: Only for backward compatibility

#### Configuration

```python
# Simulation Configuration
SIMULATION_METHOD = "PORTFOLIO_SCALED"  # Options: "FIXED", "PORTFOLIO_SCALED", "PERCENTAGE_BASED"
SIMULATION_BASE_AMOUNT = 10000  # Standard simulation amount for scaling
SIMULATION_PERCENTAGE = 0.1  # Use 10% of actual portfolio for percentage-based method
```

#### Example Scenarios

**Small Portfolio ($1,000 total)**
- **Actual**: $800 cash, 0.2 BTC
- **PORTFOLIO_SCALED**: $8,000 cash, 2 BTC (scaled to $10,000)
- **PERCENTAGE_BASED**: $80 cash, 0.02 BTC (10% of actual)

**Large Portfolio ($100,000 total)**
- **Actual**: $50,000 cash, 5 BTC
- **PORTFOLIO_SCALED**: $5,000 cash, 0.5 BTC (scaled to $10,000)
- **PERCENTAGE_BASED**: $5,000 cash, 0.5 BTC (10% of actual)

#### Usage

The simulation method is automatically applied in both `main.py` and `server.py`:

```python
from util import calculate_simulation_amounts
from config import SIMULATION_METHOD, SIMULATION_BASE_AMOUNT, SIMULATION_PERCENTAGE

# Calculate simulation amounts
sim_cash, sim_coin = calculate_simulation_amounts(
    actual_cash=v_s1,
    actual_coin=cur_coin,
    method=SIMULATION_METHOD,
    base_amount=SIMULATION_BASE_AMOUNT,
    percentage=SIMULATION_PERCENTAGE
)
```

#### Implementation Details

The `calculate_simulation_amounts()` function in `util.py` handles all three methods:

```python
def calculate_simulation_amounts(
    actual_cash: float,
    actual_coin: float,
    method: str = "PORTFOLIO_SCALED",
    base_amount: float = 10000,
    percentage: float = 0.1
) -> tuple[float, float]:
    """
    Calculate simulation amounts using different methods to eliminate bias.
    """
    if method == "FIXED":
        return 3000, 5
    
    elif method == "PORTFOLIO_SCALED":
        actual_portfolio_value = actual_cash + (actual_coin * 1)
        if actual_portfolio_value <= 0:
            return base_amount, base_amount * 0.001
        
        simulation_ratio = base_amount / actual_portfolio_value
        sim_cash = actual_cash * simulation_ratio
        sim_coin = actual_coin * simulation_ratio
        return sim_cash, sim_coin
    
    elif method == "PERCENTAGE_BASED":
        sim_cash = actual_cash * percentage
        sim_coin = actual_coin * percentage
        return sim_cash, sim_coin
    
    else:
        raise ValueError(f"Unknown simulation method: {method}")
```

#### Recommendations

1. **Use PORTFOLIO_SCALED** for most cases:
   - Provides consistent, comparable results
   - Eliminates bias from arbitrary starting values
   - Maintains realistic portfolio proportions

2. **Use PERCENTAGE_BASED** if you want results to scale with portfolio size:
   - Good for personal use where you want to see actual impact
   - Results grow/shrink with your portfolio

3. **Avoid FIXED** method:
   - Only use for backward compatibility
   - Results are biased and not meaningful

#### Benefits

- **Eliminates Bias**: Results no longer depend on arbitrary starting values
- **Comparable Results**: Different portfolio sizes can be compared fairly
- **Realistic Proportions**: Simulation maintains actual cash/coin ratios
- **Configurable**: Easy to switch between methods based on needs
- **Backward Compatible**: Old fixed method still available if needed

### Trading Parameters

Edit `app/config.py` to customize:

```python
# Currencies to trade
CURS = ['BTC', 'ETH', 'SOL']

# Trading strategies
STRATEGIES = ['RSI', 'KDJ', 'MA']

# Time span for historical data
TIMESPAN = 60

# Risk management
EP_CASH = 10  # Minimum cash threshold
EP_COIN = 0.001  # Minimum coin threshold
```

### Server Configuration

The web server runs on:
- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `8000`
- **Debug**: `False` (production mode)

## 🧪 Development

### Code Quality Tools

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy .

# Run tests
poetry run pytest

# Or activate environment first
poetry env activate
black .
isort .
mypy .
pytest
```

### Project Structure

```
crypto-prediction/
├── app/                    # Main application code
│   ├── api/               # API and web server
│   │   ├── __init__.py
│   │   └── server.py      # Flask web server
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration settings
│   │   ├── logger.py      # Logging utilities
│   │   ├── main.py        # Main trading bot logic
│   │   ├── log.txt        # Application logs
│   │   └── secret.ini     # API credentials (not in git)
│   ├── trading/           # Trading logic
│   │   ├── __init__.py
│   │   ├── cbpro_client.py    # Coinbase API client
│   │   ├── binance_client.py # Binance API client
│   │   ├── ma_trader.py       # Trading strategies
│   │   ├── strategies.py      # Strategy implementations
│   │   └── trader_driver.py   # Trading engine
│   ├── db/                # Database management
│   │   ├── __init__.py
│   │   ├── database.py        # SQLAlchemy models and DB manager
│   │   └── db_management.py   # DB CLI tools
│   ├── visualization/     # Visualization and charts
│   │   ├── __init__.py
│   │   ├── visualization.py  # Chart generation
│   │   └── plots/            # Generated visualizations
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── util.py           # Utility functions
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── pyproject.toml         # Poetry configuration
├── install.sh             # Installation script
├── start_server.py        # Server startup script
├── docker-compose.yml     # Docker setup for PostgreSQL
├── init.sql              # Database initialization
└── README.md              # This file
```

## 🗄️ PostgreSQL Database Integration

### Overview

The trading bot now includes PostgreSQL database integration to cache historical trading data and avoid redundant API calls to Coinbase. This significantly improves performance and reduces API rate limit usage.

### Features

- **Data Caching**: Stores historical OHLCV data for each trading pair
- **Smart Cache Management**: Automatically checks data freshness (24-hour default)
- **Upsert Operations**: Updates existing data or inserts new records
- **Database Statistics**: Track cached data and usage patterns
- **Automatic Cleanup**: Remove old data to manage storage

### Database Setup

The trading bot automatically uses SQLite for development (no setup required) and can be configured to use PostgreSQL for production.

#### Quick Start (SQLite - Default)

The bot automatically creates a SQLite database file (`crypto_trading.db`) in the project root. No additional setup required!

```bash
# The database will be created automatically when you first run the bot
poetry run python app/core/main.py --symbol BTC-USD --strategy MA-SELVES --days 30
```

#### Option 1: PostgreSQL with Docker Compose (Production)

```bash
# Start PostgreSQL database
docker-compose up -d postgres

# Set environment variable for PostgreSQL
export DATABASE_URL="postgresql://postgres:password@localhost:5432/crypto_trading"

# Initialize database tables
poetry run python app/db/db_management.py init

# Test database connection
poetry run python app/db/db_management.py test
```

#### Option 2: Local PostgreSQL Installation

1. **Install PostgreSQL**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   
   # Windows
   # Download from https://www.postgresql.org/download/windows/
   ```

2. **Create Database**:
   ```sql
   CREATE DATABASE crypto_trading;
   CREATE USER crypto_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE crypto_trading TO crypto_user;
   ```

3. **Set Environment Variable**:
   ```bash
   export DATABASE_URL="postgresql://crypto_user:your_password@localhost:5432/crypto_trading"
   ```

### Database Management

#### Command Line Tools

```bash
# Initialize database tables
poetry run python app/db/db_management.py init

# Show database statistics
poetry run python app/db/db_management.py stats

# Clear old data (older than 365 days)
poetry run python app/db/db_management.py clear --days 365

# Test database connection
poetry run python app/db/db_management.py test

# Drop all tables (use with caution)
poetry run python app/db/db_management.py drop
```

#### Web Interface

The web dashboard includes database management endpoints:

- `GET /api/database/stats` - Get database statistics
- `POST /api/database/clear` - Clear old data

### Database Schema

#### Historical Data Table

```sql
CREATE TABLE historical_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date TIMESTAMP NOT NULL,
    open_price FLOAT NOT NULL,
    high_price FLOAT NOT NULL,
    low_price FLOAT NOT NULL,
    close_price FLOAT NOT NULL,
    volume FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_historical_data_symbol_date ON historical_data (symbol, date);
CREATE INDEX idx_historical_data_date ON historical_data (date);
```

#### Cache Metadata Table

```sql
CREATE TABLE data_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    data_count INTEGER NOT NULL DEFAULT 0,
    cache_key VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_cache_symbol ON data_cache (symbol);
```

### How It Works

1. **Data Retrieval**: When `get_historic_data()` is called:
   - First checks database for cached data
   - If data exists and is fresh (< 24 hours old), returns cached data
   - If data is stale or missing, fetches from Coinbase API
   - Stores new data in database for future use

2. **Cache Freshness**: Data is considered fresh for 24 hours by default
   - Configurable via `max_age_hours` parameter
   - Stale data triggers fresh API call

3. **Upsert Logic**: Prevents duplicate records
   - Updates existing records if date already exists
   - Inserts new records for missing dates

### Configuration

#### Environment Variables

```bash
# Database connection string (optional)
# If not set, defaults to SQLite: "sqlite:///./crypto_trading.db"
DATABASE_URL="postgresql://username:password@localhost:5432/crypto_trading"

# Default: sqlite:///./crypto_trading.db (SQLite for development)
```

#### Cache Settings

```python
# In cbpro_client.py
def get_historic_data(self, name: str, use_cache: bool = True):
    # Set use_cache=False to bypass database and always fetch from API
```

### Performance Benefits

- **Reduced API Calls**: 90%+ reduction in API requests for historical data
- **Faster Response**: Database queries are 10-100x faster than API calls
- **Rate Limit Protection**: Avoids hitting Coinbase API rate limits
- **Offline Capability**: Can analyze historical data without internet connection

### Monitoring

#### Database Statistics

```bash
# View cached data statistics
poetry run python app/db/db_management.py stats

# Example output:
# === Database Statistics ===
# Symbol         Records    Last Updated
# --------------------------------------------------
# BTC-USD        90         2024-01-15 14:30:00
# ETH-USD        90         2024-01-15 14:30:00
# SOL-USD        90         2024-01-15 14:30:00
# 
# Total symbols: 3
```

#### Web Dashboard

The web interface shows database statistics and allows data management through API endpoints.

### Maintenance

#### Regular Cleanup

```bash
# Clear data older than 1 year (recommended monthly)
poetry run python app/db/db_management.py clear --days 365

# Clear data older than 6 months (more aggressive)
poetry run python app/db/db_management.py clear --days 180
```

#### Backup

```bash
# Backup database
pg_dump crypto_trading > backup_$(date +%Y%m%d).sql

# Restore database
psql crypto_trading < backup_20240115.sql
```

### Troubleshooting

#### Common Issues

1. **Connection Errors**
   ```bash
   # Test connection
   poetry run python app/db/db_management.py test
   
   # Check PostgreSQL status
   sudo systemctl status postgresql
   ```

2. **Permission Errors**
   ```sql
   -- Grant necessary permissions
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
   ```

3. **Performance Issues**
   ```sql
   -- Check table sizes
   SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
   FROM pg_tables WHERE schemaname = 'public';
   
   -- Analyze table statistics
   ANALYZE historical_data;
   ```

## 📈 Trading Strategies

### Basic Strategies

**RSI Strategy**
The RSI strategy uses the Relative Strength Index to identify overbought and oversold conditions:
- **Buy Signal**: When RSI falls below the oversold threshold (default: 30)
- **Sell Signal**: When RSI rises above the overbought threshold (default: 70)
- **Parameters**: 
  - RSI periods: [7, 14, 21]
  - Oversold thresholds: [10, 20, 30]
  - Overbought thresholds: [70, 80, 90]

**KDJ Strategy**
The KDJ strategy uses the KDJ oscillator to identify momentum and trend changes:
- **Buy Signal**: When KDJ falls below the oversold threshold (default: 20)
- **Sell Signal**: When KDJ rises above the overbought threshold (default: 80)
- **Parameters**:
  - Oversold thresholds: [10, 20, 30]
  - Overbought thresholds: [70, 80, 90]

**Fear & Greed Index Strategy (FEAR-GREED-SENTIMENT)**
The Fear & Greed Index strategy uses market sentiment to make trading decisions:
- **Buy Signal**: When Fear & Greed Index is below buy thresholds (extreme fear/fear)
- **Sell Signal**: When Fear & Greed Index is above sell thresholds (extreme greed/greed)
- **Parameters**:
  - Buy thresholds: [20, 30] (buy when index ≤ 20 or ≤ 30)
  - Sell thresholds: [70, 80] (sell when index ≥ 70 or ≥ 80)
- **Logic**: 
  - **Extreme Fear (0-25)**: Market panic, good buying opportunity
  - **Fear (26-45)**: Market uncertainty, potential buying opportunity
  - **Neutral (46-55)**: No clear sentiment, no action
  - **Greed (56-75)**: Market optimism, potential selling opportunity
  - **Extreme Greed (76-100)**: Market euphoria, good selling opportunity
- **Advantage**: Contrarian approach based on market psychology and sentiment

**Moving Average Strategy (MA-SELVES)**
- **Buy Signal**: Price < MA * (1 - tolerance)
- **Sell Signal**: Price > MA * (1 + tolerance)
- **Parameters**: Configurable MA lengths and tolerance percentages

**Exponential Moving Average Strategy (EXP-MA-SELVES)**
- **Buy Signal**: Price < EMA * (1 - tolerance)
- **Sell Signal**: Price > EMA * (1 + tolerance)
- **Parameters**: Configurable EMA lengths and tolerance percentages
- **Advantage**: EMA gives more weight to recent prices, making it more responsive to recent price changes compared to simple moving averages

**Bollinger Bands Strategy**
- **Buy Signal**: Price touches lower band
- **Sell Signal**: Price touches upper band
- **Parameters**: MA period and standard deviation

### Advanced Composite Strategies

**MA-MACD Strategy**
- **Description**: Combines Moving Average and MACD signals
- **Logic**: Requires both MA signal AND MACD confirmation
- **Buy**: MA shows oversold AND MACD shows bullish
- **Sell**: MA shows overbought AND MACD shows bearish
- **Advantage**: Reduces false signals by requiring dual confirmation

**MACD-KDJ Strategy**
- **Description**: Combines momentum (MACD) and oscillator (KDJ) signals
- **Logic**: Requires both momentum and oscillator confirmation
- **Buy**: MACD bullish AND KDJ oversold
- **Sell**: MACD bearish AND KDJ overbought
- **Advantage**: Captures both trend and reversal signals

**RSI-BOLL Strategy**
- **Description**: Combines RSI momentum with Bollinger Bands volatility
- **Logic**: Uses RSI for momentum and Bollinger Bands for volatility confirmation
- **Buy**: RSI oversold AND price at lower Bollinger band
- **Sell**: RSI overbought AND price at upper Bollinger band
- **Advantage**: Confirms momentum with volatility breakout

**TRIPLE-SIGNAL Strategy**
- **Description**: Combines MA, MACD, and RSI signals
- **Logic**: Requires confirmation from at least 2 out of 3 signals
- **Buy**: At least 2 buy signals from MA, MACD, RSI
- **Sell**: At least 2 sell signals from MA, MACD, RSI
- **Advantage**: Maximum signal confirmation, reduces false positives

**CONSERVATIVE-MA Strategy**
- **Description**: Moving Average with multiple period confirmation
- **Logic**: Requires price to stay beyond tolerance for multiple periods
- **Buy**: Price below MA for most periods (2 out of 3)
- **Sell**: Price above MA for most periods (2 out of 3)
- **Advantage**: Reduces whipsaws and false breakouts

**VOLUME-BREAKOUT Strategy**
- **Description**: Volume-based breakout strategy using trading volume
- **Logic**: Uses volume spikes to confirm price movements
- **Buy**: Volume spike (>1.5x average) with price increase
- **Sell**: Volume spike (>1.5x average) with price decrease
- **Advantage**: Confirms price movements with volume validation

**Enhanced MA-SELVES Strategies**

**Multi-Timeframe MA-SELVES**
- **Concept**: Requires confirmation from multiple timeframes (short, medium, long)
- **Buy Signal**: Majority of timeframes show buy signal
- **Sell Signal**: Majority of timeframes show sell signal
- **Advantage**: Reduces false signals through confirmation

**Trend-Aware MA-SELVES**
- **Concept**: Adjusts tolerance based on overall trend direction
- **Uptrend**: More aggressive buying, conservative selling
- **Downtrend**: More aggressive selling, conservative buying
- **Advantage**: Adapts to market conditions

**Volume-Weighted MA-SELVES**
- **Concept**: Adjusts tolerance based on trading volume
- **High Volume**: Tighter tolerance (stronger signals)
- **Low Volume**: Wider tolerance (weaker signals)
- **Advantage**: Considers market participation

**Adaptive MA-SELVES**
- **Concept**: Adjusts tolerance based on market volatility
- **High Volatility**: Wider tolerance bands
- **Low Volatility**: Tighter tolerance bands
- **Advantage**: Adapts to market volatility

**Momentum-Enhanced MA-SELVES**
- **Concept**: Only trades when momentum confirms MA signal
- **Buy Signal**: MA buy signal + positive momentum
- **Sell Signal**: MA sell signal + negative momentum
- **Advantage**: Filters out false signals

### Recurring Investment Strategies

**Simple Recurring Investment Strategy (SIMPLE-RECURRING)**
- **Description**: Invests a fixed amount every X days (e.g., every 7 days). If the portfolio profit exceeds a threshold (e.g., 10%), reduces position (sells). If the loss exceeds a threshold (e.g., -10%), increases position (buys more). Otherwise, performs a normal recurring buy.
- **Buy Signal**: Every X days, or if loss threshold is exceeded
- **Sell Signal**: If profit threshold is exceeded
- **Parameters**:
  - `investment_interval_days`: Days between investments (default: 7)
  - `profit_threshold`: Profit threshold to reduce position (default: 10%)
  - `loss_threshold`: Loss threshold to increase position (default: 10%)
  - `base_investment_pct`: Base investment percentage (default: 5%)
- **Advantage**: Implements a simple, robust dollar-cost averaging (DCA) approach with adaptive position management based on performance.

**Weighted Recurring Investment Strategy (WEIGHTED-RECURRING)**
- **Description**: Invests every X days, but the investment amount is weighted based on the price's position relative to its moving average. If the price is significantly below the MA, increases investment; if above, reduces investment; if near, invests the base amount.
- **Buy Signal**: Every X days, with amount weighted by MA deviation
- **Parameters**:
  - `investment_interval_days`: Days between investments (default: 7)
  - `base_investment_pct`: Base investment percentage (default: 5%)
  - `ma_weight_factor`: Factor to weight MA-based adjustments (default: 2.0)
- **Advantage**: Combines the simplicity of recurring investment with basic technical analysis for smarter DCA.

## 🎯 Trading Execution Strategies

The bot supports multiple execution strategies for buy and sell actions, significantly expanding the trading algorithm search space:

### **Buy Execution Strategies**

**by_percentage** (Default)
- **Description**: Use a percentage of available cash
- **Logic**: `buy_pct` represents the percentage of cash to spend
- **Example**: 70% means spend 70% of available cash
- **Advantage**: Conservative, maintains cash reserves

**fixed_amount**
- **Description**: Use a fixed USD amount for buying
- **Logic**: `buy_pct` represents the fixed USD amount to spend
- **Example**: 100 means spend exactly $100 USD
- **Advantage**: Predictable position sizing

**market_order**
- **Description**: Use all available cash for buying
- **Logic**: Spends entire cash balance on each buy signal
- **Example**: If you have $1000 cash, spend all $1000
- **Advantage**: Aggressive entry, maximizes position size

### **Sell Execution Strategies**

**by_percentage** (Default)
- **Description**: Use a percentage of available crypto
- **Logic**: `sell_pct` represents the percentage of crypto to sell
- **Example**: 70% means sell 70% of available crypto
- **Advantage**: Gradual profit taking, maintains position

**stop_loss**
- **Description**: Sell all crypto when price drops below threshold
- **Logic**: `sell_pct` represents the loss percentage below buy price
- **Example**: 10% means sell all if price drops 10% below buy price
- **Advantage**: Automatic risk management, limits losses

**take_profit**
- **Description**: Sell all crypto when price rises above threshold
- **Logic**: `sell_pct` represents the profit percentage above buy price
- **Example**: 20% means sell all if price rises 20% above buy price
- **Advantage**: Automatic profit taking, locks in gains

### **Strategy Configuration**

Execution strategies are configured in `app/core/config.py`:

```python
# Trading execution strategies
BUY_STAS = ["by_percentage", "fixed_amount", "market_order"]
SELL_STAS = ["by_percentage", "stop_loss", "take_profit"]
```

### **Strategy Combinations**

The bot automatically tests all combinations of:
- **Trading Strategies** (MA-SELVES, RSI, MACD, etc.)
- **Buy Execution Strategies** (by_percentage, fixed_amount, market_order)
- **Sell Execution Strategies** (by_percentage, stop_loss, take_profit)
- **Parameters** (percentages, tolerances, periods)

This creates a comprehensive search space of **6,075+ strategy combinations** for each cryptocurrency!

## 📊 Visualization Features

### **Interactive Dashboards**
The bot generates comprehensive interactive dashboards showing:

- **Portfolio Value Over Time**: Track portfolio performance with buy/sell signals
- **Asset Allocation**: Visualize crypto vs cash allocation changes
- **Price History with Signals**: See price movements and trading signals
- **Drawdown Analysis**: Monitor portfolio decline from peak values
- **Performance Comparison**: Compare strategy vs buy & hold performance
- **Returns Distribution**: Analyze daily return patterns

### **Execution Strategy Display**
All visualizations now include the selected execution strategies in their titles:

- **Buy Strategies**: Shows which buy execution method was selected (by_percentage, fixed_amount, market_order)
- **Sell Strategies**: Shows which sell execution method was selected (by_percentage, stop_loss, take_profit)
- **Strategy Combination**: Displays the complete strategy configuration used

**Example Title**: 
```
Trading Strategy Dashboard - SOL (MA-SELVES)
Buy: by_percentage, fixed_amount, market_order | Sell: by_percentage, stop_loss, take_profit
```

### **Chart Types**
- **Interactive HTML**: All charts are saved as interactive HTML files
- **Plotly Integration**: Professional-quality interactive visualizations
- **Real-time Updates**: Charts update with each trading simulation
- **Export Ready**: Charts can be easily shared or embedded in reports

## 🔒 Security

### API Security
- Store API credentials in `app/secret.ini` (not committed to git)
- Use environment variables in production
- Implement rate limiting for production use

### Best Practices
- Never commit API credentials
- Use HTTPS in production
- Implement authentication for web interface
- Regular security updates

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT**: This software is for educational and research purposes only. Cryptocurrency trading involves significant risk of loss. Never invest more than you can afford to lose.

- Past performance does not guarantee future results
- Always test strategies thoroughly before live trading
- Monitor the bot regularly
- Keep API credentials secure
- Understand the risks involved

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API credentials in `secret.ini`
   - Check Coinbase API status
   - Ensure proper permissions

2. **Port Already in Use**
   - Change port in `start_server.py`
   - Kill existing process: `lsof -ti:8000 | xargs kill`

3. **Import Errors**
   - Ensure Poetry environment is activated: `poetry env activate`
   - Reinstall dependencies: `poetry install`

4. **Visualization Errors**
   - Check plots directory exists: `mkdir -p app/plots`
   - Verify Plotly installation: `poetry show plotly`

### Debug Mode

For debugging, modify `start_server.py`:
```python
app.run(host='0.0.0.0', port=8000, debug=True)
```

## 📝 Logging

Logs are written to:
- `app/log.txt` - Application logs
- Console output - Real-time status

Log levels:
- `INFO` - General information
- `WARNING` - Non-critical issues
- `ERROR` - Errors that need attention
- `DEBUG` - Detailed debugging information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `poetry run pytest`
5. Format code: `poetry run black .`
6. Commit your changes: `git commit -m 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in `app/log.txt`
3. Check the web dashboard for error messages
4. Open an issue on GitHub

## 🔄 Updates

To update the project:
```bash
git pull origin main
poetry install
```

## 📚 Additional Resources

- [Coinbase Advanced Trade API Documentation](https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-overview)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Plotly Documentation](https://plotly.com/python/)
- [Poetry Documentation](https://python-poetry.org/docs/)

### Strategy Parameters
- **RSI Strategy**: Configure periods, oversold, and overbought thresholds
- **KDJ Strategy**: Configure oversold and overbought thresholds
- **Moving Averages**: Configure MA lengths, EMA lengths, and Bollinger Bands parameters
- **Tolerance Percentages**: Fine-tune buy/sell decision thresholds 

## 📧 Email Notifications

- To receive daily trading recommendations by email, add your Gmail credentials and recipient list to `app/core/secret.ini`:

  ```ini
  GMAIL_ADDRESS = "your_gmail@gmail.com"
  GMAIL_APP_PASSWORD = "your_gmail_app_password"
  GMAIL_RECIPIENTS = "your_gmail@gmail.com,another_email@example.com"
  ```
- You can list multiple recipients, separated by commas. All recipients will receive the same email (BCC for privacy).
- The script will automatically send a summary of today's trading actions to all listed emails after each run.
- You must use a Gmail App Password (not your main password) if you have 2-Step Verification enabled. 
