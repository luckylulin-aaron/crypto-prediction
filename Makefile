.PHONY: help install install-dev clean format lint type-check test run-server run-bot docker-build docker-run

# Default target
help:
	@echo "Crypto Trading Bot - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  clean        Clean up generated files"
	@echo ""
	@echo "Development:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting checks"
	@echo "  type-check   Run type checking with mypy"
	@echo "  test         Run tests"
	@echo ""
	@echo "Running:"
	@echo "  run-server   Start the web server"
	@echo "  run-bot      Run the trading bot"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

# Installation
install:
	@echo "Installing production dependencies..."
	poetry install --only main

install-dev:
	@echo "Installing all dependencies (including dev)..."
	poetry install

clean:
	@echo "Cleaning up generated files..."
	rm -rf app/plots/*.html
	rm -f app/log.txt
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Code Quality
format:
	@echo "Formatting code..."
	poetry run black .
	poetry run isort .

lint:
	@echo "Running linting checks..."
	poetry run flake8 app/ tests/
	poetry run black --check .
	poetry run isort --check-only .

type-check:
	@echo "Running type checking..."
	poetry run mypy app/

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v

# Running the application
run-server:
	@echo "Starting web server..."
	poetry run start-server

run-bot:
	@echo "Running trading bot..."
	poetry run trading-bot

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t crypto-trading-bot .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --env-file .env crypto-trading-bot

# Development workflow
dev-setup: install-dev
	@echo "Setting up development environment..."
	mkdir -p app/plots
	@echo "Development setup complete!"

check-all: format lint type-check test
	@echo "All checks completed!"

# Quick start
quick-start: install-dev
	@echo "Quick start setup complete!"
	@echo "Next steps:"
	@echo "1. Configure API credentials in app/secret.ini"
	@echo "2. Run: make run-server"
	@echo "3. Open: http://localhost:8000" 