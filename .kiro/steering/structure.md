# Project Structure

## Root Directory Organization

```
freqtrade/
├── freqtrade/           # Main application package
├── tests/               # Comprehensive test suite (mirrors main package structure)
├── docs/                # Documentation (Markdown files for MkDocs)
├── user_data/           # User-specific configuration and strategies
├── config_examples/     # Sample configuration files
├── build_helpers/       # Build automation and utility scripts
├── ft_client/           # Standalone client library for REST API
├── scripts/             # Development and utility scripts
├── docker/              # Docker configuration files
├── .github/             # GitHub Actions workflows and templates
├── .claude/             # Claude Code configuration and commands
├── .kiro/               # Kiro spec-driven development files
└── .serena/             # Serena coding agent memory files
```

## Subdirectory Structures

### Main Package (`freqtrade/`)

```
freqtrade/
├── __init__.py          # Package initialization with version management
├── main.py              # CLI entry point and argument parsing
├── freqtradebot.py      # Core FreqtradeBot class (2600+ lines)
├── worker.py            # Worker process management and lifecycle
├── constants.py         # Application-wide constants and defaults
├── exceptions.py        # Custom exception classes
├── wallets.py           # Wallet and balance management
├── misc.py              # Miscellaneous utility functions
│
├── commands/            # CLI command implementations
│   ├── trade_commands.py     # Trading-related commands
│   ├── optimize_commands.py  # Backtesting and optimization
│   ├── data_commands.py      # Data download and management
│   ├── list_commands.py      # List exchanges, pairs, strategies
│   ├── build_config_commands.py  # Interactive configuration builder
│   └── [...other command modules]
│
├── configuration/       # Configuration management
│   ├── configuration.py      # Main config loading and validation
│   ├── config_validation.py  # Schema validation
│   ├── load_config.py        # Config file loading utilities
│   ├── environment_vars.py   # Environment variable handling
│   └── timerange.py          # Time range parsing and validation
│
├── data/               # Data handling and storage
│   ├── history/             # Historical data management
│   ├── converter/           # Data format conversion utilities
│   ├── btanalysis/          # Backtest result analysis
│   ├── dataprovider.py      # Main data provider interface
│   └── entryexitanalysis.py # Entry/exit signal analysis
│
├── exchange/           # Exchange abstraction layer
│   ├── exchange.py          # Base exchange class
│   ├── binance.py           # Binance-specific implementation
│   ├── bybit.py             # Bybit-specific implementation
│   ├── [...other exchanges]
│   ├── exchange_utils.py    # Common exchange utilities
│   └── exchange_ws.py       # WebSocket functionality
│
├── strategy/           # Strategy framework
│   ├── interface.py         # Base strategy interface (IStrategy)
│   ├── strategy_wrapper.py  # Strategy execution wrapper
│   ├── hyper.py             # Hyperparameter optimization support
│   ├── informative_decorator.py  # Cross-timeframe data handling
│   └── parameters.py        # Strategy parameter definitions
│
├── optimize/           # Backtesting and optimization
│   ├── backtesting.py       # Main backtesting engine
│   ├── hyperopt/            # Hyperparameter optimization
│   ├── optimize_reports/    # Report generation
│   ├── analysis/            # Advanced analysis tools
│   └── space/               # Parameter space definitions
│
├── rpc/                # Remote procedure call interfaces
│   ├── rpc.py              # Core RPC functionality
│   ├── telegram.py         # Telegram bot integration
│   ├── api_server/         # REST API and WebSocket server
│   ├── webhook.py          # Webhook notifications
│   └── external_message_consumer.py  # External message handling
│
├── persistence/        # Database models and ORM
│   ├── models.py           # SQLAlchemy database models
│   ├── trade_model.py      # Trade-specific model definitions
│   ├── migrations.py       # Database migration utilities
│   └── key_value_store.py  # Key-value configuration storage
│
├── plugins/            # Plugin system
│   ├── pairlist/           # Pair selection plugins
│   ├── protections/        # Risk protection plugins
│   ├── pairlistmanager.py  # Pair list management
│   └── protectionmanager.py  # Protection management
│
├── freqai/             # Machine Learning integration
│   ├── base_models/        # Base ML model classes
│   ├── prediction_models/  # Specific ML implementations
│   ├── data_kitchen.py     # Data preprocessing
│   ├── data_drawer.py      # Data storage and retrieval
│   └── freqai_interface.py # Main FreqAI interface
│
├── plot/               # Visualization and plotting
├── leverage/           # Futures trading utilities
├── enums/              # Enumeration definitions
├── ft_types/           # Type definitions
├── util/               # General utility functions
├── mixins/             # Mixin classes
├── loggers/            # Custom logging implementations
├── system/             # System-level utilities
└── vendor/             # Vendored third-party code
```

### Test Structure (`tests/`)

The test directory mirrors the main package structure:

```
tests/
├── conftest.py              # Shared test fixtures and configuration
├── test_main.py             # Main module tests
├── freqtradebot/           # FreqtradeBot class tests
├── strategy/               # Strategy framework tests
├── exchange/               # Exchange-specific tests
├── commands/               # CLI command tests
├── optimize/               # Backtesting and optimization tests
├── rpc/                    # RPC interface tests
├── data/                   # Data handling tests
├── persistence/            # Database and ORM tests
├── freqai/                 # Machine learning tests
└── [...mirrors main structure]
```

## Code Organization Patterns

### Module Naming Conventions

- **snake_case**: All Python modules and packages use snake_case naming
- **PascalCase**: Class names use PascalCase (e.g., `FreqtradeBot`, `IStrategy`)
- **UPPER_CASE**: Constants and enums use UPPER_CASE with underscores
- **_private_methods**: Private methods/attributes prefixed with single underscore

### Class Organization

- **Interface Classes**: Abstract base classes prefixed with 'I' (e.g., `IStrategy`, `IDataHandler`)
- **Implementation Classes**: Concrete implementations without prefix
- **Mixin Classes**: Utility mixins in `mixins/` directory
- **Exception Classes**: Custom exceptions in `exceptions.py`

### File Naming Conventions

- **Single Class Files**: File named after primary class (snake_case conversion)
- **Multiple Classes**: Descriptive module name covering functionality
- **Utility Modules**: Plural names for collections of utilities (e.g., `formatters.py`)
- **Test Files**: Prefixed with `test_` and mirror source structure

## Import Organization

### Import Order (per isort configuration)

1. **Standard Library**: Python built-in modules
2. **Third-party**: External packages (pandas, numpy, etc.)
3. **Local Imports**: Freqtrade modules
4. **Relative Imports**: Same-package relative imports (discouraged)

### Import Style Guidelines

- **Absolute Imports**: Preferred over relative imports
- **Specific Imports**: Import specific classes/functions rather than entire modules when possible
- **Module Aliases**: Standard aliases for common libraries (`pd` for pandas, `np` for numpy)
- **Type Imports**: Use `from __future__ import annotations` for forward references

## Key Architectural Principles

### Separation of Concerns

- **Business Logic**: Core trading logic isolated in `FreqtradeBot` and strategy classes
- **Data Access**: Database operations encapsulated in persistence layer
- **External Interfaces**: Exchange APIs abstracted through exchange layer
- **User Interfaces**: RPC, CLI, and web interfaces kept separate from core logic

### Plugin Architecture

- **Exchange Plugins**: Each exchange implements common interface with exchange-specific optimizations
- **Strategy Plugins**: User strategies inherit from `IStrategy` base class
- **Pairlist Plugins**: Configurable pair selection through plugin system
- **Protection Plugins**: Risk management through pluggable protection mechanisms

### Dependency Injection

- **Configuration**: Centralized configuration passed to components
- **Exchange Instance**: Single exchange instance shared across components
- **Database Session**: SQLAlchemy sessions managed at application level
- **Logger**: Structured logging configuration propagated to all modules

### Error Handling Strategy

- **Custom Exceptions**: Domain-specific exceptions for different error types
- **Graceful Degradation**: Non-critical failures don't stop trading operations
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Retry Logic**: Automatic retry for transient failures (network, exchange API)

### Testing Patterns

- **Fixture-Based**: Extensive use of pytest fixtures for test data
- **Mock External Services**: Exchange APIs and external dependencies mocked
- **Property-Based Testing**: Some tests use hypothesis for property-based testing
- **Integration Tests**: End-to-end tests for critical trading workflows

### Configuration Management

- **Layered Configuration**: File-based config with environment variable overrides
- **Schema Validation**: JSON schema validation for configuration files
- **Type Safety**: Pydantic models for configuration validation and type hints
- **Secret Management**: Sensitive data handling with encryption support

### Performance Considerations

- **Lazy Loading**: Database relationships loaded on-demand
- **Caching**: Strategic caching of exchange data and calculations
- **Async Operations**: Non-blocking I/O for external API calls
- **Memory Management**: Efficient pandas DataFrame operations and cleanup
