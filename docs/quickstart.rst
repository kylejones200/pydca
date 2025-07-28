Quick Start Guide
==================

This guide will help you get started with the Decline Curve Analysis library quickly.

Installation
------------

Install the library using pip:

.. code-block:: bash

   pip install decline-analysis

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/decline-analysis.git
   cd decline-analysis
   pip install -e .

Basic Usage
-----------

Import the Library
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_analysis as dca
   import pandas as pd
   import numpy as np

Prepare Your Data
~~~~~~~~~~~~~~~~~

Your production data should be a pandas Series with a DatetimeIndex:

.. code-block:: python

   # Create sample production data
   dates = pd.date_range('2020-01-01', periods=24, freq='MS')
   production = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 
                 500, 450, 400, 350, 300, 250, 200, 180, 160, 140, 
                 120, 100, 80, 60]
   
   series = pd.Series(production, index=dates, name='oil_production')

Generate Forecasts
~~~~~~~~~~~~~~~~~~~

Use different models to forecast production:

.. code-block:: python

   # Arps decline curve (traditional)
   arps_forecast = dca.forecast(series, model="arps", kind="hyperbolic", horizon=12)
   
   # ARIMA time series model
   arima_forecast = dca.forecast(series, model="arima", horizon=12)
   
   # Advanced AI models (with fallbacks if not available)
   timesfm_forecast = dca.forecast(series, model="timesfm", horizon=12)
   chronos_forecast = dca.forecast(series, model="chronos", horizon=12)

Evaluate Performance
~~~~~~~~~~~~~~~~~~~~

Compare different models:

.. code-block:: python

   # Evaluate forecast accuracy
   metrics = dca.evaluate(series, arps_forecast)
   print(f"RMSE: {metrics['rmse']:.2f}")
   print(f"MAE: {metrics['mae']:.2f}")
   print(f"SMAPE: {metrics['smape']:.2f}%")

Create Visualizations
~~~~~~~~~~~~~~~~~~~~~

Generate professional plots:

.. code-block:: python

   # Plot forecast with historical data
   dca.plot(series, arps_forecast, title="Well Production Forecast")
   
   # Save plot to file
   dca.plot(series, arps_forecast, title="Well Production Forecast", 
            filename="forecast.png")

Multi-Well Analysis
~~~~~~~~~~~~~~~~~~~

Benchmark models across multiple wells:

.. code-block:: python

   # Prepare multi-well dataset
   well_data = pd.DataFrame({
       'well_id': ['WELL_001'] * 24 + ['WELL_002'] * 24,
       'date': pd.date_range('2020-01-01', periods=24, freq='MS').tolist() * 2,
       'oil_bbl': production * 2  # Sample data for two wells
   })
   
   # Run benchmark analysis
   results = dca.benchmark(
       well_data, 
       model="arps", 
       kind="hyperbolic",
       horizon=12,
       top_n=10
   )
   
   print(results)

Advanced Usage
--------------

Using the Forecaster Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, use the Forecaster class directly:

.. code-block:: python

   from decline_analysis.forecast import Forecaster
   
   # Create forecaster instance
   forecaster = Forecaster(series)
   
   # Generate forecast
   forecast = forecaster.forecast(model="arps", kind="exponential", horizon=6)
   
   # Evaluate against actual data
   actual_data = series.iloc[:18]  # Use first 18 points as "actual"
   metrics = forecaster.evaluate(actual_data)
   
   # Plot results
   forecaster.plot(title="Custom Forecast Analysis")

ARIMA Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Customize ARIMA parameters:

.. code-block:: python

   from decline_analysis.forecast_arima import forecast_arima
   
   # Use automatic parameter selection
   auto_forecast = forecast_arima(series, horizon=12)
   
   # Specify ARIMA order manually
   manual_forecast = forecast_arima(series, horizon=12, order=(2, 1, 1))
   
   # Include seasonal components
   seasonal_forecast = forecast_arima(
       series, 
       horizon=12, 
       seasonal=True, 
       seasonal_period=12
   )

Working with Arps Models
~~~~~~~~~~~~~~~~~~~~~~~~

Direct access to Arps decline curve functions:

.. code-block:: python

   from decline_analysis.models import fit_arps, predict_arps
   import numpy as np
   
   # Prepare time and production arrays
   t = np.arange(len(series))
   q = series.values
   
   # Fit Arps model
   params = fit_arps(t, q, kind="hyperbolic")
   print(f"Initial rate (qi): {params.qi:.2f}")
   print(f"Decline rate (di): {params.di:.4f}")
   print(f"Hyperbolic exponent (b): {params.b:.3f}")
   
   # Generate predictions
   future_t = np.arange(len(series) + 12)
   predictions = predict_arps(future_t, params)

Next Steps
----------

* Read the :doc:`tutorial` for detailed examples
* Explore the :doc:`examples` section for real-world use cases
* Check the :doc:`models` documentation for model-specific details
* Browse the :doc:`api/dca` for complete API reference
