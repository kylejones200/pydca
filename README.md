# Decline Curve Analysis

A Python library for forecasting oil and gas production using traditional decline models and modern machine learning.

## What this is

Decline curve analysis is essential for petroleum engineers, reservoir analysts, and energy economists who need to predict future production from oil and gas wells. This library provides both classical Arps decline models and state-of-the-art forecasting methods in a single, easy-to-use package.

The library helps you forecast production rates, estimate reserves, and evaluate economic performance. It supports exponential, harmonic, and hyperbolic decline curves, plus machine learning models like ARIMA, Chronos, and TimesFM for more complex patterns. Monte Carlo simulation provides probabilistic forecasts with P10/P50/P90 ranges for risk assessment.

Whether you're analyzing a single well or benchmarking hundreds across a field, the library handles data processing, model fitting, and visualization automatically. It's designed for both quick analyses and production workflows that need reproducibility and uncertainty quantification.

## Highlights

The library combines traditional petroleum engineering methods with modern forecasting techniques. It covers the full evaluation workflow from raw production data to reserves, economics, and risk — without stitching together multiple packages.

## Feature comparison

| Capability | decline-curve | petbox-dca | DCApy |
|---|:---:|:---:|:---:|
| Arps (exp / harmonic / hyperbolic) | ✓ | ✓ | ✓ |
| Modified Hyperbolic (SEC standard) | ✓ | ✓ | — |
| Duong model | ✓ | ✓ | — |
| Power-Law Exponential (PLE / Ilk) | ✓ | ✓ | — |
| Stretched Exponential (SEPD) | ✓ | ✓ | — |
| EUR — closed form (SEPD) | ✓ | — | — |
| Model auto-recommender | ✓ | — | — |
| Full-cycle economics (royalty, WI, taxes) | ✓ | — | — |
| CAPEX / IRR / payout / ROI | ✓ | — | — |
| Breakeven price (Brent's method) | ✓ | — | — |
| Price scenario analysis (low/base/high) | ✓ | — | — |
| PRMS 1P/2P/3P reserves classification | ✓ | — | — |
| Type curve normalization (Fetkovich) | ✓ | — | — |
| Probabilistic P10/P50/P90 | ✓ | — | — |
| NPV risk report (VaR, CVaR) | ✓ | — | — |
| ARIMA / TimesFM / Chronos | ✓ | — | — |
| Multi-phase (oil + gas + water) | ✓ | — | — |
| Numba JIT acceleration | ✓ | — | — |
| Batch / portfolio analysis | ✓ | — | — |
| CLI (`dca fit …`) | ✓ | — | — |

## Installation

```bash
pip install decline-curve
```

### Optional Dependencies

For additional features, install optional extras:

```bash
# Statistical models (ARIMA)
pip install decline-curve[stats]

# Large language models (Chronos, TimesFM)
pip install decline-curve[llm]

# Spatial analysis (kriging)
pip install decline-curve[spatial]

# Ecosystem integrations (pygeomodeling, geosuite)
pip install decline-curve[integrations]

# All optional features
pip install decline-curve[all]
```

## Quick start

```python
import pandas as pd
import numpy as np
from decline_curve import dca

# Create sample production data
dates = pd.date_range('2020-01-01', periods=24, freq='MS')
production = 1000 * np.exp(-0.05 * np.arange(24))
series = pd.Series(production, index=dates, name='oil_bbl')

# Generate forecast
forecast = dca.forecast(series, model='arps', kind='hyperbolic', horizon=12)
```

Command line interface:

```bash
dca fit production.csv --well WELL_001 --model arps
```

## Examples

See `examples/01_basic_dca_analysis.ipynb` for a complete workflow and `examples/02_economic_evaluation.ipynb` for economic analysis.

## Project status

The library is in active development with core functionality stable. Current focus is on expanding machine learning model support and improving uncertainty quantification methods.

## Roadmap

We're adding support for more foundation models and ensemble forecasting methods. Data quality tools and automated segmentation are being enhanced. The batch processing pipeline is being optimized for larger field datasets.

## Ecosystem Integration

This library integrates with other tools in the ecosystem:

- **signalplot**: Minimalist plotting design for publication-ready figures (used by default)
- **pygeomodeling**: Advanced Gaussian Process Regression and Kriging for spatial EUR interpolation
- **geosuite**: Professional subsurface analysis tools for petrophysics and geomechanics

All plots use signalplot's minimalist aesthetic by default when available. See `examples/10_ecosystem_integration_example.py` for integration examples.

## Contributing

We welcome contributions. Please see `CONTRIBUTING.md` for guidelines.

## License

Licensed under the Apache License 2.0. See `LICENSE` for details.
