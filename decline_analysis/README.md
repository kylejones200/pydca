---
title: 'DCA: Decline Curve Analysis and Economic Evaluation for Oil and Gas Wells in Python'
tags:
  - Python
  - decline curve analysis
  - oil and gas
  - reserves estimation
  - production forecasting
  - economic modeling
  - Arps model
authors:
  - name: Kyle Jones
    orcid: 0000-0002-4756-3973  
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-07-25
---

# decline-analysis

Forecast and evaluate oil production using Arps and LLM-based models.

## Install

```bash
pip install .
```




# Summary

Decline curve analysis (DCA) is one of the most widely used methods in petroleum engineering for estimating remaining recoverable reserves and forecasting future production based on historical data. Though simple in form, DCA provides critical inputs for reserve classification, field development planning, asset valuation, and economic decision-making.

`DCA` is a modern, open-source Python package that implements the Arps family of decline models—exponential, harmonic, and hyperbolic—and extends them with tools for economic evaluation and scenario analysis. It enables users to fit decline models to oil, gas, and water production data; forecast future rates and cumulative volumes; estimate economically recoverable reserves; and compute net present value (NPV) and payback time under various price and cost assumptions. 

The library also includes built-in utilities for downloading and preprocessing real-world well data from the North Dakota Department of Mineral Resources (NDIC), and for visualizing results with tornado plots and 3D surface plots.

# Statement of Need

Many petroleum engineers and analysts rely on proprietary tools for decline curve analysis and economic forecasting. Software such as Aries, PHDWin, or Harmony provides powerful features but is often expensive, closed-source, and difficult to integrate into reproducible workflows. Meanwhile, open-source alternatives either focus only on statistical forecasting or lack the domain-specific assumptions and structures required for reservoir performance and financial modeling.

`DCApy` addresses these gaps by providing a clean, extensible Python implementation of classical decline models with additional support for economic evaluation and sensitivity analysis. It is particularly well-suited for:

- Research on well performance in unconventional plays
- Reserve and resource estimation
- Economic scenario testing
- Batch analysis of multiple wells
- Reproducible academic or consulting workflows

All core functions use standard Python data structures (`numpy`, `pandas`) and follow an object-oriented architecture that facilitates extension and testing.

# Functionality

The `dcapy` package is organized into six modules:

### `models.py`

Implements the Arps decline models:

- Exponential decline: `q(t) = qᵢ exp(–Dᵢ t)`
- Harmonic decline: `q(t) = qᵢ / (1 + Dᵢ t)`
- Hyperbolic decline: `q(t) = qᵢ / (1 + b Dᵢ t)^{1/b}`

Includes `fit_arps()` for model fitting with nonlinear least squares and fallback logic for edge cases (e.g., flat production, short time series). Returns a typed `ArpsParams` dataclass (`qi`, `di`, `b`) and supports all three decline types.

### `reserves.py`

Calculates forecasted rates and economically recoverable reserves by integrating the fitted rate curve over time, subject to an economic cutoff rate (e.g., 10 bbl/day). Returns full forecast vectors (`t`, `q`) and filtered vectors (`t_valid`, `q_valid`) used for EUR computation.

### `economics.py`

Computes:

- Cash flows: `rate × (price – opex)`
- NPV using monthly-discounted cash flows
- Payback time (first month when cumulative cash flow > 0)

Supports separate evaluation for oil, gas, and water. Discount rates are handled monthly, but inputs are given in annual terms for ease of use.

### `sensitivity.py`

Performs grid-based sensitivity analysis on:

- Initial rate `qi`
- Nominal decline rate `di`
- Hyperbolic exponent `b`
- Price assumptions

Returns a `pandas.DataFrame` with `EUR`, `NPV`, and payback for each parameter-price combination. Supports full batch evaluation and export.

### `visuals.py`

Includes:

- **Tornado charts** showing top and bottom contributing scenarios to a chosen metric (e.g., NPV).
- **3D surface plots** of parameter sensitivity (e.g., `qi` vs `di` on NPV), filtered by fixed `b` and `price`.

All plots follow clean, minimalist defaults and save as PNG with `matplotlib`.

### `ndic_scraper.py`

Downloads monthly production data from the NDIC GIS Map Server using a simple looped URL template. Extracts Excel files, rewrites date formats, and aggregates into training and test sets based on `datetime.now()`. Used to produce long-range well histories suitable for DCA model fitting.

# Example

```python
from dcapy.models import fit_arps
from dcapy.reserves import forecast_and_reserves
from dcapy.economics import economic_metrics

params = fit_arps(t, q, kind="hyperbolic")
forecast = forecast_and_reserves(params, t_max=240)
econ = economic_metrics(forecast["q_valid"], price=70, opex=20)

print(f"EUR: {forecast['eur']:.0f} bbl")
print(f"NPV: ${econ['npv']:,.0f}")
print(f"Payback: {econ['payback_month']} months")
