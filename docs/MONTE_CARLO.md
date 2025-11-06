# Monte Carlo Simulation Guide

## Overview

Monte Carlo simulation enables probabilistic forecasting and uncertainty quantification for decline curve analysis. Instead of single deterministic forecasts, you get probability distributions showing the range of possible outcomes.

## Key Benefits

- **Probabilistic forecasts**: P10/P50/P90 instead of single values
- **Uncertainty quantification**: Understand forecast reliability
- **Risk assessment**: Probability of hitting targets
- **Better decisions**: Account for uncertainty in planning
- **Fast execution**: Uses Numba JIT + joblib parallelization

## Installation

Monte Carlo simulation is included in the base package:

```bash
pip install decline-curve
```

## Quick Start

### Basic Monte Carlo Simulation

```python
from decline_analysis.monte_carlo import (
    monte_carlo_forecast,
    MonteCarloParams,
    DistributionParams,
    plot_monte_carlo_results
)

# Define parameter distributions
mc_params = MonteCarloParams(
    qi_dist=DistributionParams('lognormal', mean=1200, std=0.3),
    di_dist=DistributionParams('uniform', min=0.10, max=0.30),
    b_dist=DistributionParams('triangular', min=0.3, mode=0.5, max=0.8),
    price_dist=DistributionParams('normal', mean=70, std=15),
    n_simulations=1000
)

# Run simulation
results = monte_carlo_forecast(mc_params, verbose=True)

# Access results
print(f"P50 EUR: {results.p50_eur:,.0f} bbl")
print(f"P50 NPV: ${results.p50_npv:,.0f}")

# Plot results
plot_monte_carlo_results(results, title="Well 123 Probabilistic Forecast")
```

## Distribution Types

### 1. Normal Distribution

Best for: Parameters with symmetric uncertainty around a mean value.

```python
price_dist = DistributionParams(
    dist_type='normal',
    mean=70,      # $/bbl
    std=15        # Standard deviation
)
```

### 2. Lognormal Distribution

Best for: Physical parameters that can't be negative (qi, di, permeability).

```python
qi_dist = DistributionParams(
    dist_type='lognormal',
    mean=1200,    # Initial rate (bbl/month)
    std=0.3       # Coefficient of variation
)
```

### 3. Uniform Distribution

Best for: Equal probability across a range (pessimistic/optimistic bounds).

```python
di_dist = DistributionParams(
    dist_type='uniform',
    min=0.10,     # Minimum decline rate
    max=0.30      # Maximum decline rate
)
```

### 4. Triangular Distribution

Best for: Expert judgment with most likely value (min/mode/max).

```python
b_dist = DistributionParams(
    dist_type='triangular',
    min=0.3,      # Conservative
    mode=0.5,     # Most likely
    max=0.8       # Optimistic
)
```

## Understanding P10/P50/P90

### Definition

- **P90**: 90% probability of exceeding this value (conservative)
- **P50**: 50% probability of exceeding this value (median, best estimate)
- **P10**: 10% probability of exceeding this value (optimistic)

### Industry Usage

- **Reserves booking**: P90 for proved, P50 for proved+probable, P10 for proved+probable+possible
- **Economic analysis**: Use P50 for base case, P90 for downside
- **Risk assessment**: Range between P10-P90 shows uncertainty

### Example Interpretation

```python
results = monte_carlo_forecast(mc_params)

# EUR interpretation
print(f"P90 EUR: {results.p90_eur:,.0f} bbl")  # Conservative estimate
print(f"P50 EUR: {results.p50_eur:,.0f} bbl")  # Best estimate
print(f"P10 EUR: {results.p10_eur:,.0f} bbl")  # Optimistic estimate

# What this means:
# - 90% chance EUR exceeds P90 value
# - 50% chance EUR exceeds P50 value
# - 10% chance EUR exceeds P10 value
```

## Advanced Features

### Correlated Parameters

Real wells often show parameter correlations (e.g., high qi often means high di):

```python
import numpy as np

# Define correlation matrix [qi, di, b, price]
correlation_matrix = np.array([
    [1.0,  0.6,  0.0,  0.0],  # qi: positive correlation with di
    [0.6,  1.0,  0.0,  0.0],  # di
    [0.0,  0.0,  1.0,  0.0],  # b: independent
    [0.0,  0.0,  0.0,  1.0],  # price: independent
])

mc_params = MonteCarloParams(
    qi_dist=DistributionParams('lognormal', mean=1200, std=0.3),
    di_dist=DistributionParams('lognormal', mean=0.15, std=0.2),
    b_dist=DistributionParams('triangular', min=0.3, mode=0.5, max=0.8),
    price_dist=DistributionParams('normal', mean=70, std=15),
    n_simulations=1000,
    correlation_matrix=correlation_matrix
)

results = monte_carlo_forecast(mc_params)
```

### Risk Analysis

Compute additional risk metrics:

```python
from decline_analysis.monte_carlo import risk_analysis

# Compute risk metrics
risk_metrics = risk_analysis(results, threshold=0)

print(f"EUR Std Dev: {risk_metrics['eur_std']:,.0f}")
print(f"Coefficient of Variation: {risk_metrics['eur_cv']:.1%}")
print(f"Prob(NPV > 0): {risk_metrics['prob_positive_npv']:.1%}")
print(f"Value at Risk (5%): ${risk_metrics['value_at_risk_npv']:,.0f}")

# Probability of hitting specific targets
target_eur = 150000
prob_exceed = np.mean(results.eur_samples > target_eur)
print(f"Prob(EUR > {target_eur:,}): {prob_exceed:.1%}")
```

### From Sensitivity to Monte Carlo

Convert sensitivity analysis ranges to Monte Carlo distributions:

```python
from decline_analysis.monte_carlo import sensitivity_to_monte_carlo

# Define ranges from sensitivity analysis
mc_params = sensitivity_to_monte_carlo(
    base_qi=1200,
    base_di=0.15,
    base_b=0.5,
    qi_range=(1000, 1400),
    di_range=(0.10, 0.20),
    b_range=(0.3, 0.7),
    price_range=(50, 90),
    n_simulations=1000
)

results = monte_carlo_forecast(mc_params)
```

## Visualization

### Comprehensive Plot

```python
from decline_analysis.monte_carlo import plot_monte_carlo_results

plot_monte_carlo_results(
    results,
    title="Well 123 Probabilistic Forecast",
    filename="well_123_monte_carlo.png"
)
```

This creates a multi-panel plot showing:
1. **Fan chart**: P10/P50/P90 production forecasts over time
2. **EUR distribution**: Histogram with percentiles
3. **NPV distribution**: Histogram with percentiles
4. **Correlation heatmap**: Parameter correlations
5. **Summary table**: Key risk metrics

### Custom Plots

```python
import matplotlib.pyplot as plt

# Plot EUR distribution
plt.figure(figsize=(10, 6))
plt.hist(results.eur_samples, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(results.p90_eur, color='r', linestyle='--', label='P90')
plt.axvline(results.p50_eur, color='g', linestyle='-', label='P50')
plt.axvline(results.p10_eur, color='b', linestyle='--', label='P10')
plt.xlabel('EUR (bbl)')
plt.ylabel('Frequency')
plt.title('EUR Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Access raw data
forecasts = results.forecasts  # All forecast realizations
eur_samples = results.eur_samples  # All EUR values
npv_samples = results.npv_samples  # All NPV values
parameters = results.parameters  # DataFrame with sampled parameters
```

## Performance

Monte Carlo simulation is optimized for speed:

- **Numba JIT**: Compiles numerical functions to machine code
- **Joblib parallelization**: Distributes simulations across CPU cores
- **Expected performance**: 1000 simulations in 5-10 seconds on typical hardware

### Performance Tips

```python
# Use all CPU cores (default)
results = monte_carlo_forecast(mc_params, n_jobs=-1)

# Force sequential (for debugging)
results = monte_carlo_forecast(mc_params, n_jobs=1)

# Adjust number of simulations
mc_params.n_simulations = 10000  # More precise, slower
mc_params.n_simulations = 500    # Faster, less precise
```

## Best Practices

### 1. Choose Appropriate Distributions

- **Normal**: Symmetric uncertainty (prices, costs)
- **Lognormal**: Physical parameters that can't be negative (qi, di, k)
- **Uniform**: No preferred value within range
- **Triangular**: Expert judgment with most likely value

### 2. Set Realistic Ranges

```python
# Too wide (unrealistic)
qi_dist = DistributionParams('uniform', min=500, max=5000)  # ❌

# Reasonable (±20-30% from mean)
qi_dist = DistributionParams('lognormal', mean=1200, std=0.25)  # ✓
```

### 3. Use Enough Simulations

- **Quick analysis**: 500-1000 simulations
- **Standard analysis**: 1000-5000 simulations
- **High precision**: 10000+ simulations

### 4. Include Correlations

Real parameters often correlate:
- High qi often means high di (rapid early decline)
- Price and costs may correlate (inflation)
- Consider field analogues for correlation estimates

### 5. Validate Results

```python
# Check parameter ranges are reasonable
print(results.parameters.describe())

# Verify correlations are applied
print(results.parameters.corr())

# Sanity check forecasts
assert results.p90_eur < results.p50_eur < results.p10_eur
```

## Common Use Cases

### 1. Reserves Booking

```python
# Conservative (P90) for proved reserves
mc_params = MonteCarloParams(
    qi_dist=DistributionParams('lognormal', mean=1200, std=0.25),
    di_dist=DistributionParams('lognormal', mean=0.15, std=0.20),
    b_dist=DistributionParams('triangular', min=0.3, mode=0.5, max=0.8),
    price_dist=DistributionParams('normal', mean=70, std=10),
    n_simulations=1000
)

results = monte_carlo_forecast(mc_params)
proved_reserves = results.p90_eur
print(f"Proved Reserves (P90): {proved_reserves:,.0f} bbl")
```

### 2. Economic Decision Making

```python
# Evaluate probability of positive NPV
results = monte_carlo_forecast(mc_params)
prob_economic = np.mean(results.npv_samples > 0)
print(f"Probability of positive NPV: {prob_economic:.1%}")

if prob_economic > 0.8:
    print("Recommendation: Develop")
elif prob_economic > 0.5:
    print("Recommendation: Further analysis")
else:
    print("Recommendation: Do not develop")
```

### 3. Portfolio Optimization

```python
# Run Monte Carlo for multiple wells
well_results = {}
for well_id, params in well_parameters.items():
    results = monte_carlo_forecast(params)
    well_results[well_id] = {
        'p50_npv': results.p50_npv,
        'p90_npv': results.p90_npv,
        'prob_positive': np.mean(results.npv_samples > 0)
    }

# Rank wells by P50 NPV
ranked_wells = sorted(well_results.items(),
                     key=lambda x: x[1]['p50_npv'],
                     reverse=True)
```

### 4. Uncertainty Reduction Analysis

```python
# Compare uncertainty before and after additional data

# Before: High uncertainty
before_params = MonteCarloParams(
    qi_dist=DistributionParams('lognormal', mean=1200, std=0.4),
    # ... wide distributions
    n_simulations=1000
)
results_before = monte_carlo_forecast(before_params)

# After: Reduced uncertainty from well test
after_params = MonteCarloParams(
    qi_dist=DistributionParams('lognormal', mean=1200, std=0.2),
    # ... tighter distributions
    n_simulations=1000
)
results_after = monte_carlo_forecast(after_params)

# Compare uncertainty
eur_range_before = results_before.p10_eur - results_before.p90_eur
eur_range_after = results_after.p10_eur - results_after.p90_eur
reduction = (eur_range_before - eur_range_after) / eur_range_before

print(f"Uncertainty reduction: {reduction:.1%}")
```

## Troubleshooting

### Issue: Unrealistic Results

**Cause**: Distribution parameters too wide or incorrect type

**Solution**:
- Review parameter ranges
- Use lognormal for positive-only parameters
- Check correlation matrix is valid

### Issue: Slow Execution

**Cause**: Too many simulations or n_jobs=1

**Solution**:
```python
# Use parallelization
results = monte_carlo_forecast(mc_params, n_jobs=-1)

# Reduce simulations for testing
mc_params.n_simulations = 500
```

### Issue: P10 < P90

**Cause**: Incorrect percentile interpretation

**Solution**: This is correct! P10 (optimistic) should be higher than P90 (conservative). The "P" refers to probability of exceedance.

## Integration with Other Features

### With Sensitivity Analysis

```python
from decline_analysis import dca
from decline_analysis.monte_carlo import monte_carlo_forecast

# Traditional sensitivity
sensitivity = dca.sensitivity_analysis(
    param_grid=[(1000, 0.15, 0.5), (1200, 0.15, 0.5), (1500, 0.15, 0.5)],
    prices=[60, 70, 80],
    opex=20.0
)

# Monte Carlo (probabilistic)
mc_params = MonteCarloParams(
    qi_dist=DistributionParams('uniform', min=1000, max=1500),
    di_dist=DistributionParams('uniform', min=0.10, max=0.20),
    b_dist=DistributionParams('uniform', min=0.3, max=0.7),
    price_dist=DistributionParams('uniform', min=60, max=80),
    n_simulations=1000
)
results = monte_carlo_forecast(mc_params)
```

### With Multi-Phase Forecasting

Monte Carlo can be extended to multi-phase forecasts by sampling GOR and water cut trends.

### With Machine Learning Models

Use Monte Carlo to quantify uncertainty in ML model predictions.

## References

- [Wikipedia: Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- SPE Monograph: "Petroleum Reserves and Resources Classification, Reporting, and Management"
- [Risk Analysis in Petroleum Exploration](https://www.onepetro.org)

## See Also

- `examples/monte_carlo_example.py` - Complete working examples
- `decline_analysis/monte_carlo.py` - Source code
- `docs/PERFORMANCE.md` - Performance optimization guide
- `docs/sensitivity.rst` - Sensitivity analysis documentation
