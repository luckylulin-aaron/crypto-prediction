# Crypto Trading Bot

A sophisticated cryptocurrency trading bot with web interface, built with Python and Flask. The bot uses multiple trading strategies including RSI, KDJ, and Moving Averages to make automated trading decisions on Coinbase Advanced Trade.

## 🚀 Features

- **Multiple Trading Strategies**: 11 strategies including 5 advanced composite strategies
- **Bias-Free Simulation**: Configurable simulation methods to eliminate trading bias
- **Web Dashboard**: Interactive Flask-based web interface
- **Real-time Monitoring**: Live status updates and performance tracking
- **Enhanced Visualizations**: Plotly-based charts with winning strategy names
- **API Integration**: Coinbase Advanced Trade API
- **Automated Trading**: Configurable trading parameters and risk management
- **Comprehensive Logging**: Detailed logging and error tracking

## 📋 Prerequisites

- Python 3.8 or higher
- Coinbase Advanced Trade API credentials
- Poetry (will be installed automatically)

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

3. **Configure API credentials**:
   Create `app/screte.ini` with your Coinbase API credentials:
   ```ini
   [CONFIG]
   COINBASE_API_KEY = "your_api_key_here"
   COINBASE_API_SECRET = "your_api_secret_here"
   ```

## 🎯 Usage

### Running the Trading Bot

```bash
# Run the trading bot directly
poetry run trading-bot

# Or activate the virtual environment
poetry shell
python app/main.py
```

### Starting the Web Server

```bash
# Start the web server
poetry run start-server

# Or manually
poetry shell
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
```

### Project Structure

```
crypto-prediction/
├── app/                    # Main application code
│   ├── __init__.py
│   ├── cbpro_client.py    # Coinbase API client
│   ├── config.py          # Configuration settings
│   ├── logger.py          # Logging utilities
│   ├── main.py            # Main trading bot logic
│   ├── ma_trader.py       # Trading strategies
│   ├── server.py          # Flask web server
│   ├── strategies.py      # Strategy implementations
│   ├── trader_driver.py   # Trading engine
│   ├── util.py            # Utility functions
│   ├── visualization.py   # Chart generation
│   ├── plots/             # Generated visualizations
│   └── screte.ini         # API credentials (not in git)
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── pyproject.toml         # Poetry configuration
├── install.sh             # Installation script
├── start_server.py        # Server startup script
└── README.md              # This file
```

## 📈 Trading Strategies

### Basic Strategies

**RSI Strategy**
- **Buy Signal**: RSI < 30 (oversold)
- **Sell Signal**: RSI > 70 (overbought)
- **Parameters**: Configurable thresholds

**KDJ Strategy**
- **Buy Signal**: KDJ < 20 (oversold)
- **Sell Signal**: KDJ > 80 (overbought)
- **Parameters**: Configurable thresholds

**Moving Average Strategy (MA-SELVES)**
- **Buy Signal**: Price crosses above MA
- **Sell Signal**: Price crosses below MA
- **Parameters**: Multiple MA periods

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

## 🔒 Security

### API Security
- Store API credentials in `app/screte.ini` (not committed to git)
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
   - Verify API credentials in `screte.ini`
   - Check Coinbase API status
   - Ensure proper permissions

2. **Port Already in Use**
   - Change port in `start_server.py`
   - Kill existing process: `lsof -ti:8000 | xargs kill`

3. **Import Errors**
   - Ensure Poetry environment is activated: `poetry shell`
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