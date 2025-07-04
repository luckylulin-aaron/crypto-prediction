# Crypto Trading Bot Web Server

This document describes how to run and use the cryptocurrency trading bot as a web server.

## Overview

The trading bot has been refactored to run as a Flask web server, providing:

- **Web Dashboard**: Interactive web interface for monitoring and controlling the bot
- **REST API**: Programmatic access to bot functionality
- **Real-time Status**: Live updates on trading simulation status
- **Visualization**: Interactive charts and dashboards
- **Threaded Execution**: Non-blocking trading simulations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

Ensure your `app/screte.ini` file contains valid Coinbase API credentials:

```ini
[CONFIG]
COINBASE_API_KEY = "your_api_key_here"
COINBASE_API_SECRET = "your_api_secret_here"
```

### 3. Start the Server

```bash
python start_server.py
```

The server will start on `http://localhost:5000`

## Web Dashboard

### Main Dashboard (`/`)

The main dashboard provides:

- **Status Overview**: Current bot status (idle/running/completed/failed)
- **Control Buttons**: Start trading simulations
- **Results Display**: View trading results and performance metrics
- **Error Reporting**: Display any errors that occur during execution

### Features

- **Real-time Updates**: Status automatically refreshes
- **Interactive Controls**: Start/stop trading simulations
- **Results Visualization**: View detailed trading results
- **Dashboard Links**: Direct links to generated visualization files

## API Endpoints

### Status Endpoints

#### `GET /api/status`
Get current trading bot status.

**Response:**
```json
{
  "is_running": false,
  "last_run": "2024-01-15T10:30:00",
  "current_status": "completed",
  "error_message": null,
  "results": {...},
  "config": {...}
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "client_initialized": true
}
```

### Control Endpoints

#### `POST /api/start`
Start a new trading simulation.

**Response:**
```json
{
  "message": "Trading simulation started"
}
```

**Error Response (400):**
```json
{
  "message": "Trading simulation already running"
}
```

### Data Endpoints

#### `GET /api/results`
Get trading simulation results.

**Response:**
```json
{
  "is_running": false,
  "last_run": "2024-01-15T10:30:00",
  "current_status": "completed",
  "results": {
    "BTC": {
      "best_trader_info": {...},
      "max_drawdown": 15.5,
      "num_transaction": 25,
      "num_buy_action": 12,
      "num_sell_action": 13,
      "high_strategy": "RSI Strategy",
      "signal": {"action": "BUY", "confidence": 0.8},
      "dashboard_file": "plots/trading_dashboard_BTC_20240115_103000.html"
    },
    "final_portfolio": {
      "crypto": 5000.0,
      "stablecoin": 2500.0
    }
  }
}
```

#### `GET /api/config`
Get current bot configuration.

**Response:**
```json
{
  "currencies": ["BTC", "ETH", "LTC"],
  "strategies": ["RSI", "KDJ", "MA"],
  "timespan": 30,
  "commit": true
}
```

### File Serving

#### `GET /plots/<filename>`
Serve generated visualization files.

## Usage Examples

### Using cURL

```bash
# Check status
curl http://localhost:5000/api/status

# Start trading simulation
curl -X POST http://localhost:5000/api/start

# Get results
curl http://localhost:5000/api/results

# Health check
curl http://localhost:5000/health
```

### Using Python Requests

```python
import requests

# Start trading simulation
response = requests.post('http://localhost:5000/api/start')
print(response.json())

# Monitor status
while True:
    status = requests.get('http://localhost:5000/api/status').json()
    if status['current_status'] == 'completed':
        results = requests.get('http://localhost:5000/api/results').json()
        print("Trading completed:", results['results'])
        break
    time.sleep(5)
```

## Architecture

### Components

1. **Flask App**: Main web server (`app/server.py`)
2. **Trading Engine**: Core trading logic (existing modules)
3. **Threading**: Non-blocking execution
4. **State Management**: Global state tracking
5. **Error Handling**: Comprehensive error reporting

### Threading Model

- Trading simulations run in separate threads
- Main Flask thread remains responsive
- Global state is thread-safe for reading
- Multiple simulations cannot run simultaneously

### Error Handling

- API errors return appropriate HTTP status codes
- Detailed error messages in responses
- Graceful degradation on failures
- Comprehensive logging

## Configuration

### Server Configuration

The server runs on:
- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `5000`
- **Debug**: `False` (production mode)

### Trading Configuration

Trading parameters are configured in `app/config.py`:
- Currencies to trade
- Trading strategies
- Time spans
- Risk parameters

## Monitoring

### Logs

The server uses the existing logging system:
- All operations are logged
- Errors are captured and displayed
- Debug information available

### Metrics

Available metrics include:
- Trading simulation status
- Execution time
- Success/failure rates
- Portfolio performance

## Security Considerations

### API Security

- No authentication implemented (add for production)
- API endpoints are public
- Consider rate limiting for production

### Credential Management

- API credentials stored in `screte.ini`
- File should be secured and not committed to git
- Consider environment variables for production

## Production Deployment

### Recommended Setup

1. **Use WSGI Server**: Gunicorn or uWSGI
2. **Add Authentication**: Implement API key or OAuth
3. **Add Rate Limiting**: Prevent abuse
4. **Use HTTPS**: Secure communications
5. **Add Monitoring**: Application performance monitoring
6. **Backup Strategy**: Regular data backups

### Example Gunicorn Command

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.server:app
```

## Troubleshooting

### Common Issues

1. **Client Initialization Failed**
   - Check API credentials in `screte.ini`
   - Verify Coinbase API access

2. **Port Already in Use**
   - Change port in `start_server.py`
   - Kill existing process

3. **Import Errors**
   - Ensure all dependencies installed
   - Check Python path

4. **Trading Errors**
   - Check API rate limits
   - Verify account permissions
   - Review error logs

### Debug Mode

For debugging, modify `start_server.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

## Development

### Adding New Endpoints

1. Add route decorator to `app/server.py`
2. Implement handler function
3. Add error handling
4. Update documentation

### Extending Functionality

- Add new trading strategies
- Implement real-time data feeds
- Add user management
- Create mobile app interface

## Support

For issues or questions:
1. Check the logs in `app/log.txt`
2. Review error messages in the web dashboard
3. Verify API credentials and permissions
4. Test with minimal configuration 