# Freqtrade Project Overview

## Project Purpose

Freqtrade is a free and open source cryptocurrency trading bot written in Python. The bot is designed to:

- Support all major cryptocurrency exchanges
- Be controlled via Telegram or webUI
- Provide backtesting and plotting capabilities
- Include money management tools
- Offer strategy optimization through machine learning (FreqAI)

## Main Features

- Automated cryptocurrency trading
- Support for spot and futures trading (experimental)
- Multiple exchange support (Binance, Bybit, Gate.io, HTX, Hyperliquid, Kraken, OKX, etc.)
- Strategy backtesting and optimization
- Risk management and protections
- Web UI and Telegram bot integration
- Machine learning integration for strategy optimization

## Target Users

- Users with coding and Python knowledge
- Educational purposes (with strong disclaimer about financial risks)
- Cryptocurrency traders looking for automated solutions

## Version

Current version: 2025.9-dev (development branch)

## License

GPLv3 - GNU General Public License v3

## Main Entry Points

- Main executable: `freqtrade` (defined in pyproject.toml)
- Core bot logic: `freqtrade.freqtradebot.FreqtradeBot` class
- Main function: `freqtrade.main:main`
