# Crypto Trading Bot

A sophisticated cryptocurrency trading bot with web interface, built with Python and Flask. The bot uses multiple trading strategies including RSI, KDJ, and Moving Averages to make automated trading decisions on Coinbase Advanced Trade.

## 🚀 Features

- **Multiple Trading Strategies**: RSI, KDJ, Moving Averages, Bollinger Bands
- **Web Dashboard**: Interactive Flask-based web interface
- **Real-time Monitoring**: Live status updates and performance tracking
- **Interactive Visualizations**: Plotly-based charts and dashboards
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

### RSI Strategy
- **Buy Signal**: RSI < 30 (oversold)
- **Sell Signal**: RSI > 70 (overbought)
- **Parameters**: Configurable thresholds

### KDJ Strategy
- **Buy Signal**: KDJ < 20 (oversold)
- **Sell Signal**: KDJ > 80 (overbought)
- **Parameters**: Configurable thresholds

### Moving Average Strategy
- **Buy Signal**: Price crosses above MA
- **Sell Signal**: Price crosses below MA
- **Parameters**: Multiple MA periods

### Bollinger Bands Strategy
- **Buy Signal**: Price touches lower band
- **Sell Signal**: Price touches upper band
- **Parameters**: MA period and standard deviation

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