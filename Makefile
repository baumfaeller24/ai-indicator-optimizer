# Nautilus AI Trading System Makefile

.PHONY: help install test lint format clean run-backtest run-live

help:
	@echo "Available commands:"
	@echo "  install     - Install all dependencies"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean cache and temp files"
	@echo "  run-backtest - Run backtest example"
	@echo "  run-live    - Run live trading (sandbox)"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=nautilus_trading --cov-report=html

lint:
	flake8 nautilus_trading/ strategies/ tests/
	mypy nautilus_trading/ strategies/

format:
	black nautilus_trading/ strategies/ tests/
	isort nautilus_trading/ strategies/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

run-backtest:
	python -m strategies.ai_strategies.example_backtest

run-live:
	python -m strategies.ai_strategies.example_live
