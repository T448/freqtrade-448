# Technology Stack

## Architecture

Freqtrade follows a modular, event-driven architecture built around the core `FreqtradeBot` class. The system uses a plugin-based design with separate modules for exchange abstraction, strategy execution, data management, and user interfaces.

### System Design Principles

- **Modular Architecture**: Clear separation between trading logic, data management, and external interfaces
- **Plugin System**: Extensible components for exchanges, strategies, pairlists, and protections
- **Event-Driven**: Asynchronous processing for real-time market data and trade execution
- **Database-Centric**: SQLAlchemy ORM for persistent trade and configuration storage
- **API-First**: RESTful API design for external integrations and user interfaces

## Core Technology Stack

### Primary Language

- **Python 3.11+** (supports 3.11, 3.12, 3.13)
- **Minimum Version**: Python 3.11 required for latest language features and performance improvements

### Backend Framework

- **FastAPI**: Modern, high-performance web framework for APIs
- **Pydantic**: Data validation and settings management using Python type annotations
- **Uvicorn**: Lightning-fast ASGI server implementation
- **SQLAlchemy**: Python SQL toolkit and Object-Relational Mapping (ORM) library

### Data Processing

- **Pandas**: Powerful data manipulation and analysis library (>=2.2.0, <3.0)
- **NumPy**: Fundamental package for scientific computing (>2.0, <3.0)
- **TA-Lib**: Technical Analysis Library for financial market data
- **ft-pandas-ta**: Extended pandas technical analysis indicators
- **technical**: Additional technical indicator library

### Exchange Integration

- **CCXT**: Unified cryptocurrency exchange API (>=4.5.4)
- **WebSocket Libraries**: Real-time market data streaming
- **aiohttp**: Asynchronous HTTP client/server framework
- **httpx**: Next-generation HTTP client (>=0.24.1)

### Machine Learning (Optional)

- **scikit-learn**: General-purpose machine learning library
- **LightGBM**: Gradient boosting framework for ML models
- **XGBoost**: Optimized distributed gradient boosting library
- **CatBoost**: Gradient boosting on decision trees (not available on aarch64)
- **TensorBoard**: Visualization toolkit for machine learning experimentation

### Communication & Monitoring

- **python-telegram-bot**: Telegram Bot API integration (>=20.1)
- **Rich**: Rich text and beautiful formatting in terminal
- **Jinja2**: Template engine for configuration and reporting
- **WebSockets**: Real-time bidirectional communication

## Development Environment

### Build System

- **setuptools**: Python package build system (>=64.0.0)
- **wheel**: Python wheel packaging standard
- **pyproject.toml**: Modern Python project configuration

### Code Quality Tools

- **Ruff**: Fast Python linter and code formatter (replaces Black + Flake8)
- **mypy**: Static type checker for Python
- **isort**: Python import statement organizer
- **pre-commit**: Git hook framework for code quality

### Testing Framework

- **pytest**: Testing framework with extensive plugin ecosystem
- **pytest-cov**: Coverage reporting for pytest
- **pytest-asyncio**: Async test support
- **pytest-xdist**: Distributed testing for parallel execution
- **pytest-mock**: Mock object support for tests

## Common Commands

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Install all features
pip install -e .[all]

# Install specific feature sets
pip install -e .[plot,hyperopt,freqai]
```

### Code Quality

```bash
# Linting and formatting
ruff check .              # Check code quality
ruff format .             # Format code
ruff check --fix .        # Auto-fix issues

# Type checking
mypy freqtrade/

# Import sorting
isort .
```

### Testing

```bash
# Run all tests
pytest --random-order --cov=freqtrade --cov-config=.coveragerc tests/

# Run specific tests
pytest tests/test_main.py

# Run tests in parallel
pytest -n auto tests/
```

### Application Execution

```bash
# Main command-line interface
freqtrade --help

# Direct module execution
python -m freqtrade

# Development mode with config
freqtrade --config user_data/config.json trade
```

## Environment Variables

### Core Configuration

- **FREQTRADE_CONFIG**: Path to main configuration file
- **FREQTRADE_USERDIR**: User data directory location
- **FREQTRADE_LOG_LEVEL**: Logging verbosity level

### Database Configuration

- **DATABASE_URL**: Database connection string (SQLite default)
- **DB_URL**: Alternative database URL format

### API Configuration

- **FREQTRADE_API_USERNAME**: REST API authentication username
- **FREQTRADE_API_PASSWORD**: REST API authentication password
- **FREQTRADE_JWT_SECRET_KEY**: JWT token signing secret

### External Service Integration

- **TELEGRAM_TOKEN**: Telegram bot token for notifications
- **WEBHOOK_URL**: External webhook endpoint for notifications

## Port Configuration

### Default Service Ports

- **8080**: Web UI and REST API server (configurable)
- **WebSocket**: Same port as REST API for real-time data streaming

### Development Ports

- **8888**: Jupyter notebook server (when using jupyter optional dependency)
- **6006**: TensorBoard visualization server (FreqAI module)

## Database Architecture

### Default Configuration

- **SQLite**: Default database for development and small deployments
- **PostgreSQL**: Recommended for production deployments with multiple instances
- **MySQL/MariaDB**: Supported alternative for production use

### Key Tables

- **trades**: Individual trade records with entry/exit details
- **orders**: Order tracking and execution history
- **pairlocks**: Pair-level trading restrictions
- **pairlist_cache**: Cached pair selection data

## Security Considerations

### API Security

- **JWT Authentication**: Secure token-based API access
- **CORS Configuration**: Cross-origin request security
- **Rate Limiting**: Built-in request throttling

### Data Protection

- **Configuration Encryption**: Sensitive API keys can be encrypted
- **Secure Communication**: HTTPS/WSS for all external communications
- **Audit Logging**: Comprehensive trade and system event logging

## Performance Optimization

### Caching Strategy

- **In-Memory Caching**: Frequently accessed data caching
- **Database Query Optimization**: Efficient SQLAlchemy queries
- **Async Processing**: Non-blocking I/O for better concurrency

### Resource Management

- **Memory Optimization**: Efficient pandas DataFrame operations
- **CPU Utilization**: Parallel processing for backtesting and optimization
- **Network Efficiency**: Connection pooling and request batching
