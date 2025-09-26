# Code Style and Conventions

## Code Formatting

- **Line length**: 100 characters (defined in pyproject.toml)
- **Formatter**: Ruff (replaces Black)
- **Import sorting**: isort with profile="black"

## Code Quality Tools

- **Linter**: Ruff with extensive rule set
- **Type checker**: mypy + pyright
- **Import organization**: isort
- **Security**: bandit (via ruff S rules)

## Ruff Configuration

### Enabled Rules

- **C90**: mccabe complexity
- **B**: bugbear
- **F**: pyflakes
- **E, W**: pycodestyle
- **UP**: pyupgrade
- **I**: isort
- **A**: flake8-builtins
- **TID**: flake8-tidy-imports
- **YTT**: flake8-2020
- **S**: flake8-bandit (security)
- **PTH**: flake8-use-pathlib
- **RUF**: ruff-specific rules
- **ASYNC**: flake8-async
- **NPY**: numpy-specific rules

### Complexity Limits

- **Max complexity**: 12 (mccabe)

## Type Checking

- **mypy**: Enabled with SQLAlchemy plugin
- **pyright**: Used for additional type checking
- **Target Python**: 3.11+
- Type stubs for external libraries included in dev dependencies

## Testing Standards

- **Framework**: pytest
- **Coverage**: pytest-cov with .coveragerc config
- **Async support**: pytest-asyncio
- **Random order**: pytest-random-order
- **Parallel execution**: pytest-xdist
- **Mocking**: pytest-mock

## File Organization

- Main package: `freqtrade/`
- Tests: `tests/` (mirrors freqtrade structure)
- User data: `user_data/`
- Documentation: `docs/`
- Build helpers: `build_helpers/`

## Naming Conventions

- Follow standard Python PEP 8 conventions
- Private methods/attributes: prefix with underscore `_`
- Constants: ALL_CAPS
- Classes: PascalCase
- Functions/methods: snake_case

## Docstring Style

- Not explicitly specified, but appears to follow standard Python conventions
- Type hints preferred over docstring type information
