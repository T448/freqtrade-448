# Codebase Structure

## Top-Level Structure

```
freqtrade/
├── freqtrade/           # Main package
├── tests/               # Test suite
├── docs/                # Documentation
├── user_data/           # User configuration and data
├── config_examples/     # Example configurations
├── scripts/             # Utility scripts
├── build_helpers/       # Build and development utilities
├── ft_client/           # Client library
└── docker/              # Docker-related files
```

## Main Package Structure (`freqtrade/`)

```
freqtrade/
├── __init__.py          # Package initialization, version info
├── main.py              # Entry point
├── freqtradebot.py      # Core bot logic (FreqtradeBot class)
├── worker.py            # Worker process management
├── constants.py         # Application constants
├── exceptions.py        # Custom exceptions
├── wallets.py           # Wallet management
├── misc.py              # Miscellaneous utilities
├── commands/            # CLI command implementations
├── configuration/       # Configuration management
├── data/                # Data handling and storage
├── enums/               # Enumerations and constants
├── exchange/            # Exchange abstraction layer
├── freqai/              # Machine learning components
├── ft_types/            # Type definitions
├── leverage/            # Leverage trading support
├── loggers/             # Logging configuration
├── mixins/              # Mixin classes
├── optimize/            # Backtesting and optimization
├── persistence/         # Database models and persistence
├── plot/                # Plotting and visualization
├── plugins/             # Plugin system (pairlists, etc.)
├── resolvers/           # Strategy and plugin resolvers
├── rpc/                 # RPC/API interfaces (Telegram, REST)
├── strategy/            # Strategy framework
├── system/              # System utilities
├── templates/           # Code templates
├── util/                # General utilities
└── vendor/              # Vendored dependencies
```

## Key Components

### Core Classes

- **FreqtradeBot** (`freqtradebot.py`): Main trading bot logic
- **Worker** (`worker.py`): Process management and lifecycle
- **Strategy** (`strategy/`): Trading strategy framework

### Data Management

- **Exchange** (`exchange/`): Exchange API abstractions
- **Persistence** (`persistence/`): Database models and ORM
- **Data** (`data/`): Historical data management

### User Interface

- **RPC** (`rpc/`): Telegram bot, REST API, WebUI
- **Commands** (`commands/`): CLI command implementations

### Analysis & Optimization

- **Optimize** (`optimize/`): Backtesting and hyperparameter optimization  
- **FreqAI** (`freqai/`): Machine learning integration
- **Plot** (`plot/`): Visualization tools

## Test Structure (`tests/`)

- Mirrors the main package structure
- Each module has corresponding test files
- `conftest.py` contains test fixtures
- `testdata/` contains sample data for testing

## Configuration

- **User data**: `user_data/` (runtime configs, strategies, data)
- **Examples**: `config_examples/` (sample configurations)
- **Project config**: `pyproject.toml` (build, tools configuration)

## Entry Points

- **CLI**: `freqtrade` command (via `main.py`)
- **Module**: `python -m freqtrade`
- **Programmatic**: Import `freqtrade` package
