Examples
========

This section provides practical examples of using the Decline Curve Analysis library for real-world petroleum production forecasting scenarios.

Example 1: Single Well Analysis
--------------------------------

Complete analysis of a single oil well with multiple forecasting models.

.. code-block:: python

   import decline_analysis as dca
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load production data (replace with your actual data)
   dates = pd.date_range('2020-01-01', periods=30, freq='MS')
   
   # Simulate realistic Bakken shale oil production
   qi = 1200  # Initial rate (bbl/month)
   di = 0.15  # High initial decline
   b = 0.3    # Hyperbolic exponent
   
   t = np.arange(len(dates))
   production = qi / (1 + b * di * t) ** (1 / b)
   
   # Add realistic noise and operational variations
   np.random.seed(42)
   noise = np.random.normal(0, production * 0.08)
   operational_effects = np.random.choice([0.9, 1.0, 1.1], len(dates), p=[0.2, 0.6, 0.2])
   production = np.maximum(production * operational_effects + noise, 0)
   
   well_data = pd.Series(production, index=dates, name='oil_production_bbl')
   
   print("=== Well Production Analysis ===")
   print(f"Period: {well_data.index[0].strftime('%Y-%m')} to {well_data.index[-1].strftime('%Y-%m')}")
   print(f"Peak production: {well_data.max():.0f} bbl/month")
   print(f"Current production: {well_data.iloc[-1]:.0f} bbl/month")
   print(f"Cumulative production: {well_data.sum():.0f} bbl")
   
   # Compare different forecasting approaches
   models = {
       'Exponential Arps': ('arps', {'kind': 'exponential'}),
       'Harmonic Arps': ('arps', {'kind': 'harmonic'}),
       'Hyperbolic Arps': ('arps', {'kind': 'hyperbolic'}),
       'ARIMA': ('arima', {}),
   }
   
   forecasts = {}
   metrics = {}
   
   for name, (model, kwargs) in models.items():
       forecast = dca.forecast(well_data, model=model, horizon=24, **kwargs)
       metric = dca.evaluate(well_data, forecast)
       
       forecasts[name] = forecast
       metrics[name] = metric
       
       print(f"\n{name}:")
       print(f"  RMSE: {metric['rmse']:.2f}")
       print(f"  MAE: {metric['mae']:.2f}")
       print(f"  SMAPE: {metric['smape']:.2f}%")
   
   # Find best model
   best_model = min(metrics.keys(), key=lambda x: metrics[x]['rmse'])
   print(f"\nBest performing model: {best_model}")
   
   # Create comprehensive plot
   dca.plot(well_data, forecasts[best_model], 
           title=f"Well Production Forecast - {best_model}")

Example 2: Multi-Well Benchmark Study
--------------------------------------

Comparing model performance across multiple wells in a field.

.. code-block:: python

   # Create synthetic field data with 10 wells
   np.random.seed(123)
   field_data = []
   
   well_params = {
       'WELL_001': {'qi': 1500, 'di': 0.12, 'b': 0.4},
       'WELL_002': {'qi': 1200, 'di': 0.18, 'b': 0.3},
       'WELL_003': {'qi': 900, 'di': 0.10, 'b': 0.5},
       'WELL_004': {'qi': 1100, 'di': 0.15, 'b': 0.35},
       'WELL_005': {'qi': 800, 'di': 0.08, 'b': 0.6},
       'WELL_006': {'qi': 1300, 'di': 0.20, 'b': 0.25},
       'WELL_007': {'qi': 1000, 'di': 0.14, 'b': 0.4},
       'WELL_008': {'qi': 1400, 'di': 0.16, 'b': 0.3},
       'WELL_009': {'qi': 950, 'di': 0.11, 'b': 0.45},
       'WELL_010': {'qi': 1250, 'di': 0.13, 'b': 0.35},
   }
   
   dates = pd.date_range('2020-01-01', periods=36, freq='MS')
   
   for well_id, params in well_params.items():
       t = np.arange(len(dates))
       production = params['qi'] / (1 + params['b'] * params['di'] * t) ** (1 / params['b'])
       
       # Add well-specific noise and operational variations
       noise = np.random.normal(0, production * 0.1)
       production = np.maximum(production + noise, 0)
       
       for date, prod in zip(dates, production):
           field_data.append({
               'well_id': well_id,
               'date': date,
               'oil_bbl': prod
           })
   
   df = pd.DataFrame(field_data)
   print(f"Field dataset: {len(df)} records, {df['well_id'].nunique()} wells")
   
   # Run comprehensive benchmark
   models_to_test = ['arps', 'arima']
   results = {}
   
   for model in models_to_test:
       print(f"\n=== Benchmarking {model.upper()} Model ===")
       
       if model == 'arps':
           result = dca.benchmark(df, model=model, kind='hyperbolic', 
                                horizon=12, top_n=10, verbose=True)
       else:
           result = dca.benchmark(df, model=model, horizon=12, 
                                top_n=10, verbose=True)
       
       results[model] = result
       
       print(f"\n{model.upper()} Summary:")
       print(f"Average RMSE: {result['rmse'].mean():.2f}")
       print(f"Average MAE: {result['mae'].mean():.2f}")
       print(f"Average SMAPE: {result['smape'].mean():.2f}%")
   
   # Compare model performance
   print("\n=== Model Comparison ===")
   for model, result in results.items():
       print(f"{model.upper()}: Avg RMSE = {result['rmse'].mean():.2f}")
   
   # Visualize results
   from decline_analysis.plot import plot_benchmark_results
   
   for model, result in results.items():
       plot_benchmark_results(result, metric='rmse', 
                            title=f"{model.upper()} Model Performance")

Example 3: Seasonal Production Analysis
----------------------------------------

Analyzing wells with seasonal production patterns.

.. code-block:: python

   # Create data with seasonal patterns (e.g., gas wells with winter peaks)
   dates = pd.date_range('2019-01-01', periods=48, freq='MS')
   base_decline = 1000 * np.exp(-0.05 * np.arange(len(dates)))
   
   # Add seasonal component
   seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
   seasonal_production = base_decline + seasonal
   
   # Add noise
   np.random.seed(456)
   noise = np.random.normal(0, seasonal_production * 0.05)
   seasonal_production = np.maximum(seasonal_production + noise, 0)
   
   seasonal_series = pd.Series(seasonal_production, index=dates, name='gas_production_mcf')
   
   print("=== Seasonal Production Analysis ===")
   print(f"Data range: {seasonal_series.index[0]} to {seasonal_series.index[-1]}")
   print(f"Seasonal variation: {seasonal_series.std():.2f}")
   
   # Compare models for seasonal data
   seasonal_models = {
       'Hyperbolic Arps': ('arps', {'kind': 'hyperbolic'}),
       'ARIMA (auto)': ('arima', {}),
   }
   
   # Also test ARIMA with explicit seasonal parameters
   from decline_analysis.forecast_arima import forecast_arima
   
   seasonal_arima = forecast_arima(seasonal_series, horizon=12, 
                                  seasonal=True, seasonal_period=12)
   
   print("\nModel Performance on Seasonal Data:")
   
   for name, (model, kwargs) in seasonal_models.items():
       forecast = dca.forecast(seasonal_series, model=model, horizon=12, **kwargs)
       metric = dca.evaluate(seasonal_series, forecast)
       
       print(f"{name}: RMSE={metric['rmse']:.2f}, SMAPE={metric['smape']:.2f}%")
   
   # Evaluate seasonal ARIMA separately
   seasonal_metric = dca.evaluate(seasonal_series, 
                                 pd.concat([seasonal_series, seasonal_arima]))
   print(f"Seasonal ARIMA: RMSE={seasonal_metric['rmse']:.2f}, "
         f"SMAPE={seasonal_metric['smape']:.2f}%")
   
   # Plot seasonal analysis
   dca.plot(seasonal_series, 
           pd.concat([seasonal_series, seasonal_arima]),
           title="Seasonal Gas Production Forecast")

Example 4: Uncertainty Analysis
--------------------------------

Analyzing forecast uncertainty using multiple models.

.. code-block:: python

   # Create well data
   dates = pd.date_range('2020-01-01', periods=24, freq='MS')
   qi, di, b = 1000, 0.12, 0.4
   t = np.arange(len(dates))
   base_production = qi / (1 + b * di * t) ** (1 / b)
   
   np.random.seed(789)
   noise = np.random.normal(0, base_production * 0.08)
   production = np.maximum(base_production + noise, 0)
   
   well_series = pd.Series(production, index=dates, name='oil_production')
   
   print("=== Forecast Uncertainty Analysis ===")
   
   # Generate multiple forecasts
   forecast_horizon = 18
   models = ['arps', 'arima']
   
   all_forecasts = {}
   
   for model in models:
       if model == 'arps':
           # Try all three Arps types
           for kind in ['exponential', 'harmonic', 'hyperbolic']:
               name = f"{model}_{kind}"
               forecast = dca.forecast(well_series, model=model, kind=kind, 
                                     horizon=forecast_horizon)
               all_forecasts[name] = forecast
       else:
           forecast = dca.forecast(well_series, model=model, 
                                 horizon=forecast_horizon)
           all_forecasts[model] = forecast
   
   # Calculate forecast statistics
   forecast_only = {}
   for name, forecast in all_forecasts.items():
       # Extract only the forecast portion
       forecast_only[name] = forecast.iloc[len(well_series):]
   
   # Create ensemble statistics
   forecast_df = pd.DataFrame(forecast_only)
   
   ensemble_stats = pd.DataFrame({
       'mean': forecast_df.mean(axis=1),
       'median': forecast_df.median(axis=1),
       'std': forecast_df.std(axis=1),
       'min': forecast_df.min(axis=1),
       'max': forecast_df.max(axis=1),
       'q25': forecast_df.quantile(0.25, axis=1),
       'q75': forecast_df.quantile(0.75, axis=1),
   })
   
   print(f"\nForecast Uncertainty (month {len(well_series)+1}):")
   print(f"Mean forecast: {ensemble_stats['mean'].iloc[0]:.2f}")
   print(f"Standard deviation: {ensemble_stats['std'].iloc[0]:.2f}")
   print(f"Range: {ensemble_stats['min'].iloc[0]:.2f} - {ensemble_stats['max'].iloc[0]:.2f}")
   print(f"Interquartile range: {ensemble_stats['q25'].iloc[0]:.2f} - {ensemble_stats['q75'].iloc[0]:.2f}")
   
   # Plot uncertainty bands
   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(figsize=(12, 8))
   
   # Plot historical data
   ax.plot(well_series.index, well_series.values, 'o-', 
          color='blue', label='Historical', linewidth=2, markersize=4)
   
   # Plot individual forecasts
   colors = ['red', 'green', 'orange', 'purple', 'brown']
   for i, (name, forecast) in enumerate(all_forecasts.items()):
       forecast_part = forecast.iloc[len(well_series):]
       ax.plot(forecast_part.index, forecast_part.values, '--', 
              color=colors[i % len(colors)], label=name, alpha=0.7)
   
   # Plot ensemble mean and uncertainty bands
   forecast_dates = forecast_df.index
   ax.plot(forecast_dates, ensemble_stats['mean'], 'k-', 
          linewidth=3, label='Ensemble Mean')
   
   ax.fill_between(forecast_dates, 
                  ensemble_stats['q25'], ensemble_stats['q75'],
                  alpha=0.3, color='gray', label='IQR (25-75%)')
   
   ax.fill_between(forecast_dates, 
                  ensemble_stats['min'], ensemble_stats['max'],
                  alpha=0.1, color='gray', label='Full Range')
   
   ax.set_xlabel('Date')
   ax.set_ylabel('Production Rate')
   ax.set_title('Production Forecast with Uncertainty Bands')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Example 5: Real-Time Forecast Updates
--------------------------------------

Simulating how forecasts update as new production data becomes available.

.. code-block:: python

   # Simulate a well with 36 months of data
   full_dates = pd.date_range('2020-01-01', periods=36, freq='MS')
   qi, di, b = 1200, 0.10, 0.45
   t = np.arange(len(full_dates))
   true_production = qi / (1 + b * di * t) ** (1 / b)
   
   np.random.seed(999)
   noise = np.random.normal(0, true_production * 0.06)
   full_series = pd.Series(np.maximum(true_production + noise, 0), 
                          index=full_dates, name='production')
   
   print("=== Real-Time Forecast Updates ===")
   
   # Simulate monthly forecast updates
   forecast_history = []
   
   for i in range(12, 30, 3):  # Update every 3 months from month 12 to 30
       # Use data up to month i
       current_data = full_series.iloc[:i]
       
       # Generate 12-month forecast
       forecast = dca.forecast(current_data, model='arps', kind='hyperbolic', 
                             horizon=12)
       
       # Evaluate against actual future data (if available)
       future_data = full_series.iloc[i:i+12]
       if len(future_data) > 0:
           # Only evaluate on available future data
           eval_length = min(len(future_data), 12)
           forecast_eval = forecast.iloc[i:i+eval_length]
           actual_eval = future_data.iloc[:eval_length]
           
           if len(forecast_eval) > 0 and len(actual_eval) > 0:
               metrics = dca.evaluate(actual_eval, forecast_eval)
               
               forecast_history.append({
                   'update_month': i,
                   'data_points': len(current_data),
                   'forecast_date': current_data.index[-1],
                   'rmse': metrics['rmse'],
                   'mae': metrics['mae'],
                   'smape': metrics['smape']
               })
               
               print(f"Month {i}: Using {len(current_data)} data points, "
                     f"RMSE={metrics['rmse']:.2f}")
   
   # Analyze forecast improvement over time
   history_df = pd.DataFrame(forecast_history)
   
   print(f"\nForecast Accuracy Improvement:")
   print(f"Early forecasts (12-18 months): Avg RMSE = {history_df.iloc[:3]['rmse'].mean():.2f}")
   print(f"Later forecasts (24-30 months): Avg RMSE = {history_df.iloc[-3:]['rmse'].mean():.2f}")
   
   # Plot forecast accuracy over time
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
   
   # Plot 1: RMSE over time
   ax1.plot(history_df['update_month'], history_df['rmse'], 'o-', 
           color='red', linewidth=2, markersize=6)
   ax1.set_xlabel('Update Month')
   ax1.set_ylabel('RMSE')
   ax1.set_title('Forecast Accuracy Improvement Over Time')
   ax1.grid(True, alpha=0.3)
   
   # Plot 2: Production data and final forecast
   final_data = full_series.iloc[:30]
   final_forecast = dca.forecast(final_data, model='arps', kind='hyperbolic', 
                               horizon=6)
   
   ax2.plot(full_series.index, full_series.values, 'o-', 
           color='blue', label='Actual Production', linewidth=2)
   ax2.plot(final_forecast.index, final_forecast.values, '--', 
           color='red', label='Final Forecast', linewidth=2)
   ax2.axvline(x=final_data.index[-1], color='gray', linestyle=':', 
              label='Forecast Start')
   
   ax2.set_xlabel('Date')
   ax2.set_ylabel('Production Rate')
   ax2.set_title('Production History and Final Forecast')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Best Practices Summary
----------------------

Based on these examples, here are key best practices:

1. **Data Quality**: Ensure consistent time intervals and handle outliers
2. **Model Selection**: Start with hyperbolic Arps, then try ARIMA for complex patterns
3. **Validation**: Always evaluate on holdout data or use cross-validation
4. **Uncertainty**: Consider multiple models and ensemble approaches
5. **Updates**: Regularly update forecasts as new data becomes available
6. **Context**: Consider operational factors and field-specific characteristics

For more detailed information, see the :doc:`tutorial` and :doc:`api/dca` documentation.
