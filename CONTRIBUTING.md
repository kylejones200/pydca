# Contributing to Decline Curve Analysis

Thank you for your interest in contributing to the Decline Curve Analysis library! This guide will help you get started.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/decline-curve.git
   cd decline-curve
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🧪 Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=decline_curve --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests in parallel
pytest -n auto
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with Black
black decline_curve/ tests/

# Sort imports with isort
isort decline_curve/ tests/

# Lint with flake8
flake8 decline_curve/ tests/

# Type check with mypy
mypy decline_curve/
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before committing:

```bash
pip install pre-commit
pre-commit install
```

## 📝 Contribution Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Add type hints to all functions
- Write descriptive docstrings (Google style)

### Example Docstring

```python
def forecast(
    series: pd.Series,
    model: str = "arps",
    horizon: int = 12
) -> pd.Series:
    """Generate production forecast using specified model.

    Args:
        series: Historical production time series
        model: Forecasting model ('arps', 'arima', 'timesfm', 'chronos')
        horizon: Number of periods to forecast

    Returns:
        Forecasted production series

    Raises:
        ValueError: If series is empty or model is invalid

    Example:
        >>> series = pd.Series([100, 90, 80], index=pd.date_range('2020-01', periods=3, freq='MS'))
        >>> forecast = dca.forecast(series, model='arps', horizon=6)
    """
```

### Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Include edge cases and error conditions

```python
def test_forecast_with_valid_data():
    """Test forecast generates correct output with valid input."""
    series = pd.Series([100, 90, 80], index=pd.date_range('2020-01', periods=3, freq='MS'))
    result = dca.forecast(series, model='arps', horizon=6)

    assert len(result) == 9  # 3 historical + 6 forecast
    assert all(result > 0)  # All values positive
    assert result.index.freq == 'MS'  # Correct frequency
```

### Documentation

- Update docstrings for modified functions
- Add examples to documentation
- Update CHANGELOG.md with your changes
- Build docs locally to verify:
  ```bash
  cd docs/
  make html
  open _build/html/index.html
  ```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, package version
6. **Error Messages**: Full traceback if applicable

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
```python
# Minimal code to reproduce
import decline_curve as dca
# ...
```

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.11]
- Package version: [e.g., 0.1.2]

**Additional context**
Any other relevant information.
```

## ✨ Feature Requests

We welcome feature requests! Please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions you've considered
4. **Examples**: Code examples of how it would be used

## 🔄 Pull Request Process

1. **Update Tests**: Ensure all tests pass and add new tests for your changes
2. **Update Documentation**: Add docstrings and update relevant docs
3. **Update CHANGELOG**: Add entry under "Unreleased" section
4. **Code Quality**: Ensure all quality checks pass
5. **Commit Messages**: Use clear, descriptive commit messages
6. **PR Description**: Explain what changes you made and why

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code formatted with Black and isort
- [ ] Type hints added where applicable
- [ ] Docstrings updated
- [ ] CHANGELOG.md updated
- [ ] Documentation built successfully
- [ ] No merge conflicts with main branch

### Commit Message Format

```
type(scope): brief description

Longer description if needed

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(forecast): add support for seasonal ARIMA`
- `fix(models): correct hyperbolic decline calculation`
- `docs(readme): update installation instructions`

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- **New forecasting models** (e.g., LSTM, Transformer, additional time series models)
- **Performance optimizations** (vectorization, caching)
- **Additional evaluation metrics** (MASE, WAPE, etc.)
- **Enhanced visualizations** (interactive plots, dashboards)

### Medium Priority
- **Data connectors** (APIs for production data sources)
- **Export functionality** (Excel, PDF reports)
- **Configuration files** (YAML/TOML for analysis parameters)
- **Parallel processing** (multi-well batch analysis)

### Documentation
- **Tutorial notebooks** (real-world examples)
- **Video tutorials** (screencasts)
- **Blog posts** (case studies)
- **API documentation** (improve existing docs)

## 💬 Getting Help

- **Documentation**: [https://decline-analysis.readthedocs.io/](https://decline-analysis.readthedocs.io/)
- **GitHub Issues**: [https://github.com/yourusername/decline-curve/issues](https://github.com/yourusername/decline-curve/issues)
- **Discussions**: [https://github.com/yourusername/decline-curve/discussions](https://github.com/yourusername/decline-curve/discussions)

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to Decline Curve Analysis! 🎉**
