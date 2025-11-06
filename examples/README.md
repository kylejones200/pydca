# Examples

This directory contains practical examples demonstrating the use of the Decline Curve Analysis package.

## 📓 Jupyter Notebooks

1. **[01_basic_dca_analysis.ipynb](01_basic_dca_analysis.ipynb)** - Introduction to decline curve analysis with Arps models
2. **[02_economic_evaluation.ipynb](02_economic_evaluation.ipynb)** - Economic analysis, NPV, and reserves estimation
3. **[03_real_bakken_well_analysis.ipynb](03_real_bakken_well_analysis.ipynb)** - ⭐ Real Bakken shale well production analysis
4. **[04_advanced_data_processing.ipynb](04_advanced_data_processing.ipynb)** - ⭐ Advanced data cleaning and Prophet forecasting
5. **[05_multiphase_forecasting.ipynb](05_multiphase_forecasting.ipynb)** - ⭐⭐ Multi-phase forecasting (oil + gas + water)
6. **[06_multi_well_benchmarking.ipynb](06_multi_well_benchmarking.ipynb)** - Comparing models across multiple wells (coming soon)
7. **[07_ml_forecasting.ipynb](07_ml_forecasting.ipynb)** - Using ARIMA and foundation models for forecasting (coming soon)

## 🚀 Python Scripts

- **[performance_benchmark.py](performance_benchmark.py)** - ⭐⭐ NEW! Comprehensive performance benchmarking
- **[profiling_example.py](profiling_example.py)** - ⭐⭐ NEW! How to profile and optimize your code
- **[monte_carlo_example.py](monte_carlo_example.py)** - ⭐⭐⭐ NEW! Probabilistic forecasting and uncertainty quantification

## 📊 Sample Data

- **[sample_well_data.csv](data/sample_well_data.csv)** - Synthetic single well production history
- **[field_production_data.csv](data/field_production_data.csv)** - Synthetic multi-well field data
- **[bakken_well_production.csv](data/bakken_well_production.csv)** - ⭐ Real Bakken well production (NYSTUEN 14B-35HS)

## 🚀 Getting Started

### Install Jupyter

```bash
pip install jupyter notebook
```

### Run Notebooks

```bash
cd examples
jupyter notebook
```

Then open any `.ipynb` file in your browser.

## 📝 Example Descriptions

### 01 - Basic DCA Analysis
Learn the fundamentals:
- Loading production data
- Fitting Arps decline models (exponential, harmonic, hyperbolic)
- Generating forecasts
- Evaluating model performance
- Creating visualizations

### 02 - Economic Evaluation
Perform economic analysis:
- Calculate EUR (Estimated Ultimate Recovery)
- Compute NPV and cash flows
- Determine payback periods
- Run sensitivity analysis
- Create tornado plots

### 03 - Real Bakken Well Analysis
Work with real production data:
- Load and clean real Bakken well data
- Handle data quality issues
- Analyze oil, water, and gas production
- Calculate decline rates
- Compare multiple models on real data
- Generate realistic economic forecasts

### 04 - Advanced Data Processing
Master data preparation techniques:
- Clean and filter production data
- Calculate derived metrics (water cut, GOR, days online)
- Detect production anomalies
- Use convenience functions for data prep
- Calculate initial production rate (qi)
- Optional: Prophet forecasting for seasonal patterns

### 05 - Multi-Phase Forecasting ⭐⭐ NEW!
Forecast oil, gas, and water simultaneously:
- Create multi-phase data structures
- Coupled vs independent forecasting
- Maintain physical relationships (GOR, water cut)
- Evaluate multi-phase accuracy
- Check forecast consistency
- Compare forecasting approaches
- Multi-phase economic analysis

### 06 - Multi-Well Benchmarking (Coming Soon)
Analyze multiple wells:
- Load field-level data
- Benchmark different models
- Compare well performance
- Aggregate statistics
- Batch processing workflows

### 07 - ML Forecasting (Coming Soon)
Advanced forecasting techniques:
- ARIMA time series modeling
- Foundation model integration (TimesFM, Chronos)
- Model comparison
- Uncertainty quantification
- Ensemble forecasting

### Performance Benchmark Script ⭐⭐ NEW!
Measure and compare performance improvements:
- Numba JIT compilation speedup (10-100x)
- Joblib parallelization (4-8x on typical CPUs)
- Sequential vs parallel benchmarking
- Multi-well processing performance
- Sensitivity analysis performance
- Generate performance reports

**Run it:**
```bash
cd examples
python performance_benchmark.py
```

### Profiling Example Script ⭐⭐ NEW!
Learn how to profile and optimize your code:
- Using @profile decorator for line-by-line profiling
- Using profile_context() for timing blocks
- Using @time_function for quick timing
- Interpreting profiling results
- Finding bottlenecks in your analysis
- Saving profiling reports

**Run it:**
```bash
cd examples
python profiling_example.py
```

### Monte Carlo Example Script ⭐⭐⭐ NEW!
Probabilistic forecasting and uncertainty quantification:
- Setting up parameter distributions (normal, lognormal, uniform, triangular)
- Running Monte Carlo simulations
- Generating P10/P50/P90 forecasts
- Computing risk metrics and probabilities
- Visualizing uncertainty with fan charts
- Correlated parameters
- Converting sensitivity ranges to Monte Carlo
- Comparing deterministic vs probabilistic approaches

**Run it:**
```bash
cd examples
python monte_carlo_example.py
```

## 💡 Tips

- All notebooks use the sample data provided in the `data/` directory
- Notebooks are self-contained and can be run independently
- Modify parameters to experiment with different scenarios
- Check the main documentation for detailed API reference

## 🔗 Additional Resources

- [Full Documentation](https://decline-analysis.readthedocs.io/)
- [API Reference](https://decline-analysis.readthedocs.io/api/dca.html)
- [Tutorial](https://decline-analysis.readthedocs.io/tutorial.html)
