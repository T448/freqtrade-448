# Suggested Development Commands

## Essential Commands

### Testing

```bash
# Run all tests
pytest --random-order --cov=freqtrade --cov-config=.coveragerc tests/

# Run specific test
pytest tests/test_main.py

# Run with coverage report
pytest --cov=freqtrade tests/
```

### Code Quality

```bash
# Run linter and formatter (Ruff)
ruff check .                # Lint check
ruff format .               # Format code
ruff check --fix .          # Auto-fix issues

# Type checking
mypy freqtrade/            # Type check with mypy
pyright                    # Alternative type checker

# Import sorting
isort .
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff
```

### Running Freqtrade

```bash
# Main executable
freqtrade --help

# Run directly from module
python -m freqtrade --help

# Run with specific config
freqtrade --config user_data/config.json
```

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Install all optional dependencies
pip install -e .[all]

# Install specific feature sets
pip install -e .[plot,hyperopt,freqai]
```

### Git Hooks (Lefthook)

- **lefthook** is configured for commit message formatting
- Automatically adds issue numbers to commit messages

## File Locations

- **Main config**: `user_data/config.json`
- **Tests**: `tests/`
- **Coverage config**: `.coveragerc`
- **Type config**: `pyproject.toml` (mypy/pyright sections)
- **Pre-commit config**: `.pre-commit-config.yaml`

## System Commands (Linux)

Standard Linux commands are available:

- `ls`, `cd`, `grep`, `find`
- `git` for version control
- `docker` and `docker-compose` for containerization
