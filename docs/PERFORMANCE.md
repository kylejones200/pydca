# Performance Optimization Guide

This document explains the performance optimizations in the decline curve analysis package and how to use them effectively.

## Overview

The package includes three major performance optimizations:

1. **Numba JIT Compilation**: 10-100x speedup for numerical computations
2. **Joblib Parallelization**: Near-linear scaling with CPU cores
3. **Line Profiler Integration**: Identify bottlenecks in your code

## Installation

Install the package with performance dependencies:

```bash
pip install decline-curve

# Or with development dependencies (includes line_profiler)
pip install decline-curve[dev]
```

## 1. Numba JIT Compilation

### What is Numba?

Numba is a just-in-time (JIT) compiler that translates Python functions to optimized machine code. The package uses Numba to accelerate:

- Arps decline curve predictions (`predict_arps`)
- Exponential and hyperbolic decline calculations
- Numerical integration for reserves estimation

### How it Works

Numba compiles functions on first use. Subsequent calls are 10-100x faster:

```python
from decline_curve.models import predict_arps, ArpsParams
import numpy as np

params = ArpsParams(qi=1200, di=0.15, b=0.5)
t = np.arange(0, 240)

# First call includes compilation overhead (~1 second)
q1 = predict_arps(t, params)

# Subsequent calls are very fast (~0.01 seconds)
q2 = predict_arps(t, params)
```

### Fallback Behavior

If Numba is not installed, functions run in pure Python mode without errors. Performance will be slower but functionality is identical.

### Tips

- Expect 1-2 second warmup on first function call
- Numba caches compiled functions across sessions
- Use `cache=True` in decorators (already enabled)
- Works best with NumPy arrays

## 2. Joblib Parallelization

### What is Joblib?

Joblib provides easy parallelization for embarrassingly parallel problems. The package uses Joblib for:

- Multi-well benchmarking (`dca.benchmark()`)
- Sensitivity analysis (`dca.sensitivity_analysis()`)

### Usage

Both functions accept an `n_jobs` parameter:

```python
from decline_curve import dca

# Use all CPU cores (default)
results = dca.benchmark(df, model='arps', n_jobs=-1)

# Use specific number of cores
results = dca.benchmark(df, model='arps', n_jobs=4)

# Force sequential execution
results = dca.benchmark(df, model='arps', n_jobs=1)
```

### Example: Multi-Well Benchmarking

```python
import pandas as pd
from decline_curve import dca

# Load data with multiple wells
df = pd.read_csv('production_data.csv')

# Parallel processing (8-core CPU = ~7x faster)
results = dca.benchmark(
    df,
    model='arps',
    kind='hyperbolic',
    horizon=12,
    top_n=100,
    n_jobs=-1,  # Use all cores
    verbose=True
)

print(results[['well_id', 'rmse', 'mae']])
```

### Example: Sensitivity Analysis

```python
from decline_curve import dca

# Define parameter grid
param_grid = [
    (qi, di, b)
    for qi in [1000, 1200, 1500]
    for di in [0.10, 0.15, 0.20]
    for b in [0.3, 0.5, 0.7]
]
prices = [50, 60, 70, 80, 90]

# Parallel execution (8-core CPU = ~8x faster)
sensitivity = dca.sensitivity_analysis(
    param_grid=param_grid,
    prices=prices,
    opex=20.0,
    discount_rate=0.10,
    n_jobs=-1  # Use all cores
)

print(sensitivity[['qi', 'price', 'EUR', 'NPV']])
```

### Performance Expectations

| Operation | Sequential | Parallel (8 cores) | Speedup |
|-----------|-----------|-------------------|---------|
| 100 wells | 60s | 8s | 7.5x |
| 1000 sensitivity cases | 45s | 6s | 7.5x |

### Fallback Behavior

If Joblib is not installed, functions run sequentially without errors.

## 3. Line Profiler Integration

### What is line_profiler?

Line profiler shows time spent on each line of code, helping identify bottlenecks.

### Installation

```bash
pip install decline-curve[dev]
```

### Basic Usage

```python
from decline_curve.profiling import profile, print_stats

@profile
def my_analysis():
    # Your code here
    pass

# Run your function
my_analysis()

# Print profiling results
print_stats()
```

### Context Manager

```python
from decline_curve.profiling import profile_context

with profile_context("My expensive operation") as timer:
    # Your code here
    result = expensive_function()

print(f"Took {timer['elapsed']:.2f} seconds")
```

### Time Decorator

```python
from decline_curve.profiling import time_function

@time_function
def forecast_all_wells(df):
    # Your code
    pass

# Prints: "forecast_all_wells took 12.345 seconds"
forecast_all_wells(df)
```

### Example: Profile Your Analysis

```python
from decline_curve import dca
from decline_curve.profiling import profile, print_stats
import pandas as pd

@profile
def run_analysis(df):
    """Profile this function to find bottlenecks."""
    results = []
    for well_id in df['well_id'].unique()[:10]:
        well_data = df[df['well_id'] == well_id]
        series = well_data.set_index('date')['oil_bbl']
        forecast = dca.forecast(series, model='arps', horizon=12)
        results.append(forecast)
    return results

# Run analysis
df = pd.read_csv('production_data.csv')
results = run_analysis(df)

# Print line-by-line profiling
print_stats()
```

### Save Profiling Results

```python
from decline_curve.profiling import save_stats

# After running profiled functions
save_stats("profiling_results.txt")
```

## Benchmarking

Run the included benchmark script to test performance on your system:

```bash
cd examples
python performance_benchmark.py
```

This will:
1. Generate synthetic well data
2. Compare sequential vs parallel processing
3. Measure Numba JIT compilation impact
4. Show expected speedups on your hardware

## Best Practices

### 1. Use Parallel Processing by Default

```python
# Good: Use all cores
results = dca.benchmark(df, n_jobs=-1)

# Only use n_jobs=1 if you need reproducibility or debugging
results = dca.benchmark(df, n_jobs=1)
```

### 2. Warm Up Numba for Benchmarking

```python
# Run once to compile functions
_ = dca.forecast(sample_series, model='arps', horizon=12)

# Now benchmark
start = time.time()
for well in wells:
    forecast = dca.forecast(well, model='arps', horizon=12)
elapsed = time.time() - start
```

### 3. Profile Before Optimizing

```python
from decline_curve.profiling import profile, print_stats

# Profile your code first
@profile
def my_analysis():
    # Your code
    pass

my_analysis()
print_stats()

# Identify bottlenecks, then optimize
```

### 4. Choose the Right Tool

| Task | Best Tool | Expected Speedup |
|------|-----------|------------------|
| Arps predictions | Numba (automatic) | 10-100x |
| Multi-well processing | Joblib `n_jobs=-1` | 4-8x (on typical CPU) |
| Large parameter grids | Joblib `n_jobs=-1` | 4-8x |
| Finding bottlenecks | Line profiler | N/A |

## Troubleshooting

### Numba Installation Issues

If you encounter Numba installation problems:

```bash
# Try installing with conda
conda install numba

# Or use pip with specific version
pip install numba==0.58.0
```

The package will still work without Numba, just slower.

### Parallel Processing Not Faster

Common reasons:
1. **Small dataset**: Parallelization overhead dominates (use `n_jobs=1`)
2. **I/O bound**: If reading/writing files, parallelization won't help much
3. **Already using all cores**: Check CPU usage during execution
4. **Memory bandwidth**: Multiple cores may saturate memory bandwidth

### Memory Issues with Parallel Processing

If you run out of memory:

```python
# Reduce number of parallel jobs
results = dca.benchmark(df, n_jobs=4)  # Instead of -1

# Process in batches
results = []
for batch in batches:
    batch_results = dca.benchmark(batch, n_jobs=4)
    results.append(batch_results)
```

## Performance Comparison

### Before Optimization (Pure Python)

```python
# 100 wells, sequential
results = old_benchmark(df, n_wells=100)
# Time: ~120 seconds
```

### After Optimization (Numba + Joblib)

```python
# 100 wells, parallel with Numba JIT
results = dca.benchmark(df, top_n=100, n_jobs=-1)
# Time: ~8 seconds (15x faster!)
```

### Breakdown

- Numba JIT: 10x faster per-well
- Joblib parallelization: 8 cores ≈ 7x faster
- Combined: ~70x faster (but 15x in practice due to overhead)

## Advanced: Custom Optimization

### Add Numba to Your Functions

```python
import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def my_custom_calculation(data):
    """Numba-optimized calculation."""
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = data[i] ** 2 + np.log(data[i] + 1)
    return result
```

### Parallelize Custom Code

```python
from joblib import Parallel, delayed

def process_well(well_id):
    # Your processing code
    return result

# Parallel execution
results = Parallel(n_jobs=-1)(
    delayed(process_well)(wid)
    for wid in well_ids
)
```

## References

- [Numba Documentation](https://numba.pydata.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [Line Profiler Documentation](https://kernprof.readthedocs.io/)

## Support

For performance-related questions or issues:
1. Check this document first
2. Run `examples/performance_benchmark.py` to verify setup
3. Open an issue on GitHub with benchmark results
