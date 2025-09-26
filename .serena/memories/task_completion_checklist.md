# Task Completion Checklist

When completing any development task in the Freqtrade codebase, follow these steps:

## Code Quality Checks (Required)

1. **Linting**: Run `ruff check .` to check for code issues
2. **Formatting**: Run `ruff format .` to ensure consistent formatting  
3. **Type Checking**: Run `mypy freqtrade/` for type validation
4. **Import Sorting**: Run `isort .` to organize imports

## Testing (Required)

1. **Unit Tests**: Run `pytest tests/` for full test suite
2. **Specific Tests**: Run `pytest tests/test_[module].py` for targeted testing
3. **Coverage**: Ensure coverage doesn't decrease significantly
4. **New Features**: Add unit tests for any new functionality

## Pre-commit Validation

1. **Run Pre-commit**: `pre-commit run --all-files`
2. **Fix Issues**: Address any failures before committing
3. **Manual Review**: Double-check automated fixes

## Documentation (When Applicable)

1. **Code Comments**: Add docstrings for new public methods/classes
2. **Type Hints**: Ensure all new functions have proper type annotations
3. **README Updates**: Update relevant documentation if behavior changes

## Git Workflow

1. **Commit Messages**: Use clear, descriptive commit messages
2. **Issue References**: Include issue numbers (handled by lefthook)
3. **Branch Strategy**: Work on `develop` branch, not `stable`
4. **PR Requirements**: Ensure CI passes before requesting review

## Performance Considerations

1. **Complexity**: Keep method complexity under 12 (mccabe limit)
2. **Dependencies**: Don't add unnecessary external dependencies
3. **Memory Usage**: Consider memory implications for long-running processes
4. **Database**: Optimize queries and consider migration impacts

## Security Checklist

1. **Credentials**: Never commit API keys or secrets
2. **Input Validation**: Validate all external inputs
3. **SQL Injection**: Use SQLAlchemy ORM properly
4. **Path Traversal**: Use pathlib for file operations

## Final Validation Commands

```bash
# Quick validation suite
ruff check . && ruff format . && mypy freqtrade/ && pytest tests/

# Full pre-commit check
pre-commit run --all-files
```

## Notes

- All these tools are configured in `pyproject.toml`
- Pre-commit hooks will run automatically on commit
- CI will verify all these checks pass
- Failed CI means the PR cannot be merged
