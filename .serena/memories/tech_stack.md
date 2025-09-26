# Technology Stack

## Core Language

- **Python 3.11+** (supports 3.11, 3.12, 3.13)

## Build System

- **setuptools** (>=64.0.0) with wheel
- **pyproject.toml** for project configuration

## Key Dependencies

### Trading & Exchange

- **ccxt** (>=4.5.4) - Cryptocurrency exchange abstraction
- **SQLAlchemy** (>=2.0.6) - Database ORM
- **python-telegram-bot** (>=20.1) - Telegram bot integration

### Data Processing

- **pandas** (>=2.2.0,<3.0) - Data manipulation
- **numpy** (>2.0,<3.0) - Numerical computing
- **TA-Lib** (<0.7) - Technical analysis library
- **ft-pandas-ta** - Extended technical analysis
- **technical** - Additional technical indicators

### Web & API

- **fastapi** - Modern web framework for APIs
- **pydantic** (>=2.2.0) - Data validation
- **uvicorn** - ASGI server
- **websockets** - WebSocket support
- **httpx** (>=0.24.1) - HTTP client
- **aiohttp** - Async HTTP client/server

### Optional Modules

- **plot**: plotly for visualization
- **hyperopt**: scipy, scikit-learn, optuna for optimization  
- **freqai**: ML libraries (catboost, lightgbm, xgboost, tensorboard)
- **freqai_rl**: Reinforcement learning (torch, gymnasium, stable-baselines3)

### Development Tools

- **pytest** - Testing framework
- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **isort** - Import sorting
- **pre-commit** - Git hooks

## Supported Platforms

- Linux
- macOS
- Unix-like systems
