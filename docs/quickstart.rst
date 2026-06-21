Quick Start Guide
==================

This guide will help you get started with the Decline Curve Analysis library quickly.

Installation
------------

Install the library using pip:

.. code-block:: bash

   pip install decline-curve

Or install from source:

.. code-block:: bash

   git clone https://github.com/kylejones200/decline-curve.git
   cd decline-curve
   pip install -e .

Basic Usage
-----------

Import the Library
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_curve as dca
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

Shale / Unconventional Wells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For shale wells, Arps b > 1 is physically unsound at long time horizons. Use the
Duong, PLE, or SEPD models instead:

.. code-block:: python

   from decline_curve import fit_duong, predict_duong, eur_duong
   import numpy as np

   # Fit Duong model to historical data
   t = np.arange(len(series))
   q = series.values
   params = fit_duong(t, q)
   print(f"Duong params: q1={params.q1:.0f}, a={params.a:.3f}, m={params.m:.3f}")

   # Forecast 120 months
   future_t = np.arange(120)
   forecast = predict_duong(future_t, params)

   # Estimated Ultimate Recovery
   eur = eur_duong(params, t_max=360, econ_limit=5.0)
   print(f"EUR = {eur:,.0f} BOE")

   # Or use the main API dispatcher
   duong_forecast = dca.forecast(series, model="arps", kind="duong", horizon=24)

Full-Cycle Economics
~~~~~~~~~~~~~~~~~~~~~

The economics module handles royalties, taxes, CAPEX, OPEX, and all standard PE metrics:

.. code-block:: python

   from decline_curve.economics import WellEconomics, cashflow, npv, irr, payout, breakeven_price

   econ = WellEconomics(
       capex=5_000_000,          # $5MM drill & complete
       price=70.0,               # $/BOE
       royalty_rate=0.1875,      # 3/16 landowner royalty
       severance_tax_rate=0.046, # Texas default
       ad_valorem_rate=0.02,
       opex_fixed=5_000,         # $/month fixed LOE
       opex_variable=8.0,        # $/BOE variable LOE
       discount_rate=0.10,
   )

   result = cashflow(forecast.values, econ)
   print(f"NPV (10%): ${npv(result):>12,.0f}")
   print(f"IRR:       {irr(result):.1%}")
   print(f"Payout:    month {payout(result)}")
   print(f"Breakeven: ${breakeven_price(forecast.values, econ):.2f}/BOE")

Advanced Usage
--------------

Using the Forecaster Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, use the Forecaster class directly:

.. code-block:: python

   from decline_curve.forecast import Forecaster

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

   from decline_curve.forecast_arima import forecast_arima

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

   from decline_curve.models import fit_arps, predict_arps
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
