# Additional Methods Summary

This document summarizes all methods found in the `wip/` folder that have been integrated or are available for integration into the main `decline_curve` package.

## Integrated Methods

### Statistical Forecasting (`decline_curve/forecast_statistical.py`)

The following statistical forecasting methods have been integrated:

1. **`simple_exponential_smoothing()`**
   - Simple exponential smoothing forecast
   - Parameters: `alpha` (smoothing parameter), `horizon`
   - Use case: Quick baseline forecasts

2. **`moving_average_forecast()`**
   - Moving average forecast
   - Parameters: `window` (periods for average), `horizon`
   - Use case: Simple trend-following forecasts

3. **`linear_trend_forecast()`**
   - Linear trend extrapolation
   - Parameters: `horizon`
   - Use case: Trend-based forecasts

4. **`holt_winters_forecast()`**
   - Holt-Winters exponential smoothing with seasonality
   - Parameters: `horizon`, `seasonal_periods` (optional)
   - Use case: Seasonal production patterns
   - Requires: `statsmodels`

5. **`calculate_confidence_intervals()`**
   - Calculate confidence intervals for forecasts
   - Parameters: `method` ('naive', 'residual_based'), `confidence` (0-1)
   - Returns: `(lower_bound, upper_bound)` Series

### EUR Estimation (`decline_curve/eur_estimation.py`)

1. **`calculate_eur_from_production()`**
   - Calculate EUR from production data by fitting decline curve
   - Parameters: `months`, `production`, `model_type`, `t_max`, `econ_limit`
   - Returns: Dictionary with EUR and parameters

2. **`calculate_eur_batch()`**
   - Calculate EUR for multiple wells from DataFrame
   - Parameters: DataFrame with production data, column names, `model_type`, `min_months`
   - Returns: DataFrame with EUR results per well

## Available in Main API

All integrated methods are available through the main `dca.forecast()` function:

```python
from decline_curve import dca

# Statistical methods
forecast = dca.forecast(series, model='exponential_smoothing', alpha=0.3, horizon=12)
forecast = dca.forecast(series, model='moving_average', window=6, horizon=12)
forecast = dca.forecast(series, model='linear_trend', horizon=12)
forecast = dca.forecast(series, model='holt_winters', horizon=12, seasonal_periods=12)

# EUR estimation
from decline_curve.eur_estimation import calculate_eur_from_production, calculate_eur_batch
```

## Methods Not Yet Integrated

### Research/Experimental Methods

The following methods in `wip/` are research code and may be integrated later:

1. **Interference Analysis** (`wip/injection_interference/interference_analysis.py`)
   - Cross-correlation analysis
   - Spatial interference detection
   - Use case: Well interference studies

2. **Reservoir Property Mapping** (`wip/reservoir_property_mapping/reservoir_mapping.py`)
   - Gaussian Process regression for spatial interpolation
   - Production data as proxy for reservoir properties
   - Use case: Spatial mapping of reservoir characteristics

3. **Granite Foundation Model** (`wip/time_series_forecasting/production_forecasting_enhanced.py`)
   - Granite (LagLlama) foundation model forecasting
   - Requires: `granite-timeseries` package
   - Status: Available in wip, not yet integrated

## Usage Examples

### Statistical Forecasting

```python
import pandas as pd
from decline_curve import dca

# Create sample data
dates = pd.date_range('2020-01-01', periods=24, freq='MS')
production = 1000 * (0.95 ** np.arange(24))
series = pd.Series(production, index=dates)

# Exponential smoothing
forecast = dca.forecast(series, model='exponential_smoothing', alpha=0.3, horizon=12)

# Moving average
forecast = dca.forecast(series, model='moving_average', window=6, horizon=12)

# Linear trend
forecast = dca.forecast(series, model='linear_trend', horizon=12)

# Holt-Winters (with seasonality)
forecast = dca.forecast(series, model='holt_winters', horizon=12, seasonal_periods=12)
```

### EUR Estimation

```python
from decline_curve.eur_estimation import calculate_eur_batch
import pandas as pd

# Load production data
df = pd.read_csv('production_data.csv')

# Calculate EUR for all wells
eur_results = calculate_eur_batch(
    df,
    well_id_col='API_WELLNO',
    date_col='Date',
    value_col='Oil',
    model_type='hyperbolic',
    min_months=12
)

print(eur_results[['well_id', 'eur', 'qi', 'di', 'b']])
```

### Confidence Intervals

```python
from decline_curve.forecast_statistical import calculate_confidence_intervals
from decline_curve import dca

# Generate forecast
forecast = dca.forecast(series, model='arps', horizon=12)

# Calculate confidence intervals
lower, upper = calculate_confidence_intervals(
    series, forecast, method='naive', confidence=0.95
)
```

## Benchmark Factory Integration

The benchmark factory now supports all integrated statistical methods:

```yaml
models:
  - "arps"
  - "arima"
  - "exponential_smoothing"
  - "moving_average"
  - "linear_trend"
  - "holt_winters"
```

## Testing Status

- [x] Statistical forecasting methods integrated
- [x] EUR estimation methods integrated
- [x] Main API updated
- [x] Benchmark factory updated
- [x] Tests for new methods
  - [x] `test_forecast_statistical.py` - Comprehensive tests for all statistical forecasting methods
  - [x] `test_eur_estimation.py` - Tests for EUR calculation from production data
  - [x] `test_panel_analysis.py` - Tests for panel data analysis functions
  - [x] `test_panel_analysis_sweep.py` - Tests for parameter sweep system
- [x] Documentation updated
