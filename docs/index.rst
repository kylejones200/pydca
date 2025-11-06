Decline Curve Analysis Documentation
=====================================

A comprehensive Python library for petroleum decline curve analysis with traditional Arps models, modern time series forecasting (ARIMA, TimesFM, Chronos), economic evaluation, sensitivity analysis, and advanced evaluation metrics.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

Features
--------

* **Multi-Phase Forecasting**: Simultaneous oil, gas, and water forecasting ⭐ NEW
* **Arps Decline Curves**: Exponential, harmonic, and hyperbolic models
* **Time Series Forecasting**: ARIMA, Prophet, TimesFM, and Chronos models
* **Data Processing Utilities**: Production-ready data cleaning and preparation ⭐ NEW
* **Economic Analysis**: NPV, cash flow, and payback calculations
* **Sensitivity Analysis**: Parameter and price sensitivity studies
* **Reserve Estimation**: EUR calculations with economic limits
* **Data Loading**: Automated NDIC and regulatory data access
* **Evaluation Metrics**: RMSE, MAE, SMAPE, MAPE, R²
* **Visualization**: Professional decline curve and forecast plots
* **Benchmarking**: Multi-well performance comparison

Quick Start
-----------

.. code-block:: python

   import decline_analysis as dca
   import pandas as pd

   # Load your production data
   dates = pd.date_range('2020-01-01', periods=24, freq='MS')
   production = pd.Series([1000, 950, 900, 850, 800] +
                         list(range(800, 200, -25)), index=dates)

   # Forecast with Arps decline curve
   forecast = dca.forecast(production, model='arps', horizon=12)

   # Calculate economics
   economics = dca.economics(forecast, price=60, opex=15)
   print(f"NPV: ${economics['npv']:,.2f}")

   # Plot results
   dca.plot(production, forecast, title='Production Forecast')

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/multiphase
   api/data_processing
   models
   source/sensitivity
   source/economics
   source/reserves
   source/data_loader

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
