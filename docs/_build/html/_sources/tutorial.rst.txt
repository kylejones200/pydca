Tutorial
========

This tutorial provides a comprehensive guide to using the Decline Curve Analysis library for petroleum production forecasting.

Overview
--------

The Decline Curve Analysis library provides four main forecasting approaches:

1. **Arps Decline Curves**: Traditional petroleum engineering models (exponential, harmonic, hyperbolic)
2. **ARIMA**: Statistical time series modeling with automatic parameter selection
3. **TimesFM**: Google's foundation model for time series forecasting
4. **Chronos**: Amazon's probabilistic time series foundation model

Getting Started
---------------

Data Preparation
~~~~~~~~~~~~~~~~

Your production data should be a pandas Series with a DatetimeIndex:

.. code-block:: python

   import pandas as pd
   import numpy as np
   import decline_analysis as dca
   
   # Create sample monthly production data
   dates = pd.date_range('2020-01-01', periods=36, freq='MS')
   
   # Simulate realistic decline curve data
   qi = 1200  # Initial production rate
   di = 0.08  # Decline rate
   b = 0.4    # Hyperbolic exponent
   
   t = np.arange(len(dates))
   production = qi / (1 + b * di * t) ** (1 / b)
   
   # Add some realistic noise
   np.random.seed(42)
   noise = np.random.normal(0, production * 0.05)
   production = np.maximum(production + noise, 0)
   
   series = pd.Series(production, index=dates, name='oil_production_bbl')
   print(series.head())

Working with Arps Models
-------------------------

Exponential Decline
~~~~~~~~~~~~~~~~~~~

Best for wells with constant percentage decline:

.. code-block:: python

   # Fit exponential decline
   exp_forecast = dca.forecast(series, model="arps", kind="exponential", horizon=12)
   
   # Evaluate performance
   metrics = dca.evaluate(series, exp_forecast)
   print(f"Exponential RMSE: {metrics['rmse']:.2f}")
   
   # Plot results
   dca.plot(series, exp_forecast, title="Exponential Decline Forecast")

Harmonic Decline
~~~~~~~~~~~~~~~~

Suitable for wells with decreasing decline rates:

.. code-block:: python

   # Fit harmonic decline
   harm_forecast = dca.forecast(series, model="arps", kind="harmonic", horizon=12)
   
   # Compare with exponential
   exp_metrics = dca.evaluate(series, exp_forecast)
   harm_metrics = dca.evaluate(series, harm_forecast)
   
   print(f"Exponential RMSE: {exp_metrics['rmse']:.2f}")
   print(f"Harmonic RMSE: {harm_metrics['rmse']:.2f}")

Hyperbolic Decline
~~~~~~~~~~~~~~~~~~

Most flexible, often best fit for real data:

.. code-block:: python

   # Fit hyperbolic decline
   hyp_forecast = dca.forecast(series, model="arps", kind="hyperbolic", horizon=12)
   
   # This is often the best performing traditional model
   hyp_metrics = dca.evaluate(series, hyp_forecast)
   print(f"Hyperbolic RMSE: {hyp_metrics['rmse']:.2f}")
   
   # Plot with metrics displayed
   dca.plot(series, hyp_forecast, title="Hyperbolic Decline Forecast")

Advanced Time Series Models
----------------------------

ARIMA Modeling
~~~~~~~~~~~~~~

Automatic parameter selection:

.. code-block:: python

   # ARIMA with automatic parameter selection
   arima_forecast = dca.forecast(series, model="arima", horizon=12)
   arima_metrics = dca.evaluate(series, arima_forecast)
   print(f"ARIMA RMSE: {arima_metrics['rmse']:.2f}")

Manual ARIMA configuration:

.. code-block:: python

   from decline_analysis.forecast_arima import forecast_arima
   
   # Specify ARIMA order manually
   manual_arima = forecast_arima(series, horizon=12, order=(2, 1, 1))
   
   # Include seasonal components
   seasonal_arima = forecast_arima(
       series, 
       horizon=12, 
       seasonal=True, 
       seasonal_period=12
   )

Foundation Models
~~~~~~~~~~~~~~~~~

TimesFM and Chronos provide state-of-the-art forecasting:

.. code-block:: python

   # TimesFM (Google's foundation model)
   timesfm_forecast = dca.forecast(series, model="timesfm", horizon=12)
   timesfm_metrics = dca.evaluate(series, timesfm_forecast)
   
   # Chronos (Amazon's foundation model)
   chronos_forecast = dca.forecast(series, model="chronos", horizon=12)
   chronos_metrics = dca.evaluate(series, chronos_forecast)
   
   print(f"TimesFM RMSE: {timesfm_metrics['rmse']:.2f}")
   print(f"Chronos RMSE: {chronos_metrics['rmse']:.2f}")

Model Comparison
----------------

Compare all models systematically:

.. code-block:: python

   models = ["arps", "arima", "timesfm", "chronos"]
   results = {}
   
   for model in models:
       if model == "arps":
           forecast = dca.forecast(series, model=model, kind="hyperbolic", horizon=12)
       else:
           forecast = dca.forecast(series, model=model, horizon=12)
       
       metrics = dca.evaluate(series, forecast)
       results[model] = metrics
       
       print(f"{model.upper()} - RMSE: {metrics['rmse']:.2f}, "
             f"MAE: {metrics['mae']:.2f}, SMAPE: {metrics['smape']:.2f}%")

Multi-Well Analysis
-------------------

Prepare Multi-Well Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create sample multi-well dataset
   wells = ['WELL_001', 'WELL_002', 'WELL_003', 'WELL_004', 'WELL_005']
   well_data = []
   
   np.random.seed(42)
   for well_id in wells:
       # Different initial rates and decline parameters
       qi = np.random.uniform(800, 1500)
       di = np.random.uniform(0.05, 0.12)
       b = np.random.uniform(0.2, 0.8)
       
       dates = pd.date_range('2020-01-01', periods=30, freq='MS')
       t = np.arange(len(dates))
       production = qi / (1 + b * di * t) ** (1 / b)
       
       # Add noise
       noise = np.random.normal(0, production * 0.08)
       production = np.maximum(production + noise, 0)
       
       for date, prod in zip(dates, production):
           well_data.append({
               'well_id': well_id,
               'date': date,
               'oil_bbl': prod
           })
   
   df = pd.DataFrame(well_data)
   print(f"Dataset shape: {df.shape}")
   print(df.head())

Run Benchmark Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare models across all wells
   arps_results = dca.benchmark(df, model="arps", kind="hyperbolic", 
                                horizon=12, top_n=5, verbose=True)
   
   arima_results = dca.benchmark(df, model="arima", horizon=12, 
                                 top_n=5, verbose=True)
   
   print("\\nArps Results:")
   print(arps_results)
   
   print("\\nARIMA Results:")
   print(arima_results)

Visualize Benchmark Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_analysis.plot import plot_benchmark_results
   
   # Plot RMSE comparison
   plot_benchmark_results(arps_results, metric='rmse', 
                         title="Arps Model Performance - RMSE")
   
   plot_benchmark_results(arima_results, metric='rmse', 
                         title="ARIMA Model Performance - RMSE")

Advanced Usage Patterns
------------------------

Custom Forecaster Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_analysis.forecast import Forecaster
   
   # Create forecaster instance
   forecaster = Forecaster(series)
   
   # Try different models
   models_to_test = [
       ("arps", {"kind": "exponential"}),
       ("arps", {"kind": "harmonic"}),
       ("arps", {"kind": "hyperbolic"}),
       ("arima", {}),
   ]
   
   best_model = None
   best_rmse = float('inf')
   
   for model, kwargs in models_to_test:
       forecast = forecaster.forecast(model=model, horizon=12, **kwargs)
       
       # Use first 24 points for evaluation
       eval_data = series.iloc[:24]
       metrics = forecaster.evaluate(eval_data)
       
       print(f"{model} {kwargs}: RMSE = {metrics['rmse']:.2f}")
       
       if metrics['rmse'] < best_rmse:
           best_rmse = metrics['rmse']
           best_model = (model, kwargs)
   
   print(f"\\nBest model: {best_model[0]} {best_model[1]} (RMSE: {best_rmse:.2f})")

Production Forecasting Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete workflow for production forecasting:

.. code-block:: python

   def analyze_well_production(series, well_name="Unknown"):
       """Complete production analysis workflow."""
       
       print(f"\\n=== Analysis for {well_name} ===")
       print(f"Data period: {series.index[0]} to {series.index[-1]}")
       print(f"Data points: {len(series)}")
       print(f"Average production: {series.mean():.2f}")
       print(f"Latest production: {series.iloc[-1]:.2f}")
       
       # Test multiple models
       models = {
           'Hyperbolic Arps': ('arps', {'kind': 'hyperbolic'}),
           'ARIMA': ('arima', {}),
           'TimesFM': ('timesfm', {}),
       }
       
       results = {}
       forecasts = {}
       
       for name, (model, kwargs) in models.items():
           try:
               forecast = dca.forecast(series, model=model, horizon=12, **kwargs)
               metrics = dca.evaluate(series, forecast)
               
               results[name] = metrics
               forecasts[name] = forecast
               
               print(f"{name}: RMSE={metrics['rmse']:.2f}, "
                     f"MAE={metrics['mae']:.2f}, SMAPE={metrics['smape']:.2f}%")
               
           except Exception as e:
               print(f"{name}: Failed - {e}")
       
       # Find best model
       if results:
           best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
           print(f"\\nBest model: {best_model}")
           
           # Create visualization
           best_forecast = forecasts[best_model]
           dca.plot(series, best_forecast, 
                   title=f"{well_name} - {best_model} Forecast")
           
           return best_forecast, results[best_model]
       
       return None, None
   
   # Use the workflow
   forecast, metrics = analyze_well_production(series, "Example Well")

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

1. **Ensure regular time intervals**: Monthly data works best
2. **Handle missing values**: Remove or interpolate gaps
3. **Check for outliers**: Extreme values can skew results
4. **Sufficient history**: At least 12-24 data points recommended

Model Selection
~~~~~~~~~~~~~~~

1. **Start with Arps hyperbolic**: Often best for oil/gas wells
2. **Try ARIMA for complex patterns**: Good for irregular decline
3. **Use foundation models for difficult cases**: When traditional methods fail
4. **Cross-validate results**: Split data for out-of-sample testing

Evaluation
~~~~~~~~~~

1. **Use multiple metrics**: RMSE, MAE, SMAPE provide different insights
2. **Consider business context**: Some errors are more costly than others
3. **Validate on holdout data**: Don't evaluate on training data
4. **Monitor forecast uncertainty**: Consider confidence intervals

Next Steps
----------

* Explore the :doc:`examples` for real-world applications
* Check the :doc:`api/dca` for detailed API documentation
* Review the :doc:`models` section for theoretical background
