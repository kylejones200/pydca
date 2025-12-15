# Decline Curve Analysis

A Python library for forecasting oil and gas production using traditional decline models and modern machine learning.

## What this is

Decline curve analysis is essential for petroleum engineers, reservoir analysts, and energy economists who need to predict future production from oil and gas wells. This library provides both classical Arps decline models and state-of-the-art forecasting methods in a single, easy-to-use package.

The library helps you forecast production rates, estimate reserves, and evaluate economic performance. It supports exponential, harmonic, and hyperbolic decline curves, plus machine learning models like ARIMA, Chronos, and TimesFM for more complex patterns. Monte Carlo simulation provides probabilistic forecasts with P10/P50/P90 ranges for risk assessment.

Whether you're analyzing a single well or benchmarking hundreds across a field, the library handles data processing, model fitting, and visualization automatically. It's designed for both quick analyses and production workflows that need reproducibility and uncertainty quantification.

## Highlights

The library combines traditional petroleum engineering methods with modern forecasting techniques, providing probabilistic forecasts, economic analysis, and batch processing capabilities. It uses Numba JIT compilation and parallel processing for fast execution on large datasets, making it practical for field-wide analysis and sensitivity studies.

## Quick start

```bash
pip install decline-curve
```

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

```bash
dca fit production.csv --well WELL_001 --model arps
```

## Examples

See [examples/01_basic_dca_analysis.ipynb](examples/01_basic_dca_analysis.ipynb) for a complete workflow and [examples/02_economic_evaluation.ipynb](examples/02_economic_evaluation.ipynb) for economic analysis.

## Project status

The library is in active development with core functionality stable. Current focus is on expanding machine learning model support and improving uncertainty quantification methods.

## Roadmap

We're adding support for more foundation models and ensemble forecasting methods. Data quality tools and automated segmentation are being enhanced. The batch processing pipeline is being optimized for larger field datasets.

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
