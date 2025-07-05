#!/bin/bash

# Crypto Trading Bot - Poetry Installation Script

set -e  # Exit on any error

echo "🚀 Setting up Crypto Trading Bot with Poetry..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | sed 's/Python \([0-9]\+\.[0-9]\+\).*/\1/')
required_version="3.8"

# Compare versions (works on both macOS and Linux)
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo "✅ Poetry installed successfully"
else
    echo "✅ Poetry is already installed"
fi

# Install project dependencies
echo "📚 Installing project dependencies..."
poetry install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p app/plots

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    poetry run pre-commit install
else
    echo "ℹ️  pre-commit not found. Install it with: pip install pre-commit"
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure your API credentials in app/screte.ini"
echo "2. Run the trading bot: poetry run trading-bot"
echo "3. Start the web server: poetry run start-server"
echo "4. Access the dashboard at: http://localhost:8000"
echo ""
echo "🔧 Development commands:"
echo "- Format code: poetry run black ."
echo "- Sort imports: poetry run isort ."
echo "- Type checking: poetry run mypy ."
echo "- Run tests: poetry run pytest"
echo ""
echo "📚 For more information, see the README.md file" 