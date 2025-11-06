# Decline Curve Analysis (DCA)

[![PyPI version](https://badge.fury.io/py/decline-curve.svg)](https://badge.fury.io/py/decline-curve)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://decline-analysis.readthedocs.io/)

A modern, open-source Python package for **decline curve analysis** and **production forecasting** in petroleum engineering. Combines traditional Arps decline models with advanced machine learning approaches (ARIMA, Chronos, TimesFM) for accurate oil and gas production forecasting.

## 🎯 Key Features

- **📉 Arps Decline Models**: Exponential, harmonic, and hyperbolic decline curve analysis
- **🤖 ML Forecasting**: ARIMA, Chronos (Amazon), and TimesFM (Google) foundation models
- **💰 Economic Analysis**: NPV, cash flow, and payback period calculations
- **📊 Sensitivity Analysis**: Parameter sensitivity with tornado plots and 3D visualizations
- **🔍 Reserves Estimation**: EUR (Estimated Ultimate Recovery) calculations
- **📈 Professional Plotting**: Publication-ready Tufte-style visualizations
- **⚡ Batch Processing**: Multi-well benchmarking and analysis
- **🌐 Data Integration**: Built-in NDIC (North Dakota) production data scraper

## 🚀 Quick Start

### Installation

```bash
pip install decline-curve
```

**Note**: The package name is `decline-curve` on PyPI, but imports as `decline_analysis`.

For development installation:
```bash
git clone https://github.com/kylejones200/pydca.git
cd pydca
pip install -e .
```

### Basic Usage

```python
import pandas as pd
import numpy as np
from decline_analysis import dca

# Create sample production data
dates = pd.date_range('2020-01-01', periods=24, freq='MS')
production = 1000 * np.exp(-0.05 * np.arange(24))  # Exponential decline
series = pd.Series(production, index=dates, name='oil_bbl')

# Generate 12-month forecast using hyperbolic Arps model
forecast = dca.forecast(series, model='arps', kind='hyperbolic', horizon=12)

# Evaluate forecast accuracy
metrics = dca.evaluate(series, forecast)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")

# Create visualization
dca.plot(series, forecast, title='Production Forecast', filename='forecast.png')
```

### Economic Analysis

```python
from decline_analysis.models import ArpsParams
from decline_analysis import dca

# Define well parameters
params = ArpsParams(qi=1200, di=0.15, b=0.4)  # Initial rate, decline rate, b-factor

# Calculate reserves and economics
reserves = dca.reserves(params, t_max=240, econ_limit=10.0)
economics = dca.economics(
    production=pd.Series(reserves['q_valid']),
    price=70.0,      # $/bbl
    opex=20.0,       # $/bbl
    discount_rate=0.10
)

print(f"EUR: {reserves['eur']:.0f} bbl")
print(f"NPV: ${economics['npv']:,.0f}")
print(f"Payback: {economics['payback_month']} months")
```

### Multi-Well Benchmarking

```python
# Load field data
df = pd.read_csv('field_production.csv')  # Columns: well_id, date, oil_bbl

# Benchmark multiple models across top 10 wells
results = dca.benchmark(
    df,
    model='arps',
    kind='hyperbolic',
    horizon=12,
    top_n=10,
    verbose=True
)

print(results[['well_id', 'rmse', 'mae', 'smape']])
```

### Command Line Interface

```bash
# Forecast a single well
decline --csv production.csv --well WELL_001 --model arps --horizon 12

# Benchmark multiple wells
decline --csv production.csv --benchmark --model arima --top_n 20
```

## 📚 Documentation

Full documentation is available at [Read the Docs](https://decline-analysis.readthedocs.io/) including:

- **[Tutorial](https://decline-analysis.readthedocs.io/tutorial.html)**: Step-by-step guide
- **[API Reference](https://decline-analysis.readthedocs.io/api/dca.html)**: Complete function documentation
- **[Examples](https://decline-analysis.readthedocs.io/examples.html)**: Real-world use cases
- **[Models](https://decline-analysis.readthedocs.io/models.html)**: Theory and implementation details

## 🔬 Advanced Features

### Sensitivity Analysis

```python
# Define parameter grid
param_grid = [
    (1000, 0.10, 0.3),  # (qi, di, b)
    (1200, 0.15, 0.4),
    (1500, 0.20, 0.5),
]
prices = [50, 60, 70, 80]

# Run sensitivity analysis
sensitivity = dca.sensitivity_analysis(
    param_grid=param_grid,
    prices=prices,
    opex=20.0,
    discount_rate=0.10
)

print(sensitivity[['qi', 'di', 'b', 'price', 'EUR', 'NPV']])
```

### Foundation Model Forecasting

```python
# Use Google's TimesFM model
forecast_timesfm = dca.forecast(series, model='timesfm', horizon=24)

# Use Amazon's Chronos model
forecast_chronos = dca.forecast(series, model='chronos', horizon=24)

# Compare with traditional ARIMA
forecast_arima = dca.forecast(series, model='arima', horizon=24)
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/pydca.git
cd pydca

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=decline_analysis --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black decline_analysis/ tests/

# Sort imports
isort decline_analysis/ tests/

# Lint
flake8 decline_analysis/ tests/

# Type checking
mypy decline_analysis/
```

## 📊 Models Supported

| Model | Type | Best For | Speed |
|-------|------|----------|-------|
| **Exponential Arps** | Physics-based | Mature wells, constant decline | ⚡⚡⚡ |
| **Harmonic Arps** | Physics-based | Low decline rate wells | ⚡⚡⚡ |
| **Hyperbolic Arps** | Physics-based | Unconventional wells, shale | ⚡⚡⚡ |
| **ARIMA** | Statistical | Complex patterns, seasonality | ⚡⚡ |
| **TimesFM** | Foundation model | Long-term forecasts | ⚡ |
| **Chronos** | Foundation model | Zero-shot forecasting | ⚡ |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- New forecasting models
- Additional evaluation metrics
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{jones2025dca,
  author = {Jones, Kyle T.},
  title = {Decline Curve Analysis: Production Forecasting for Oil and Gas Wells},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pydca},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Arps decline curve theory from J.J. Arps (1945)
- TimesFM model by Google Research
- Chronos model by Amazon Science
- North Dakota Industrial Commission for public production data

## 📞 Support

- **Documentation**: [https://decline-analysis.readthedocs.io/](https://decline-analysis.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pydca/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pydca/discussions)

---

**Made with ❤️ for the petroleum engineering community**
