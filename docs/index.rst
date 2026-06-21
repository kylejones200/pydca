Decline Curve Analysis Documentation
=====================================

A comprehensive Python library for petroleum decline curve analysis. Covers traditional
Arps models, shale-era decline variants (Duong, PLE, SEPD), full-cycle economics
(royalties, taxes, CAPEX, IRR, breakeven), probabilistic forecasting, time series
models (ARIMA, TimesFM, Chronos), and batch analysis tools.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

Features
--------

* **Arps Decline Curves**: Exponential, harmonic, and hyperbolic models with Numba JIT
* **Modified Hyperbolic**: SEC-standard hyperbolic-to-exponential switch at terminal decline
* **Shale Variants**: Duong, Power-Law Exponential (PLE/Ilk), and SEPD — physically correct for unconventionals
* **Model Auto-Selection**: ``recommend_model()`` picks the right decline model from data shape and drive mechanism
* **Full-Cycle Economics**: Royalties, WI, severance tax, ad valorem, CAPEX, IRR, payout, ROI, breakeven price
* **Scenario Analysis**: Low/base/high price deck comparison with full-cycle metrics per scenario
* **PRMS Reserves Classification**: SPE-PRMS 1P/2P/3P via Monte Carlo EUR simulation
* **Type Curve Normalization**: Fetkovich-style dimensionless matching across b-value families
* **Probabilistic Forecasting**: Monte Carlo P10/P50/P90 with NPV risk metrics (VaR, CVaR)
* **Multi-Phase Forecasting**: Simultaneous oil, gas, and water forecasting
* **Time Series Forecasting**: ARIMA, TimesFM, and Chronos models
* **Data Processing Utilities**: Production-ready data cleaning and preparation
* **Sensitivity Analysis**: Parameter and price sensitivity studies
* **Reserve Estimation**: EUR calculations with economic limits (closed-form for SEPD)
* **Data Loading**: Automated NDIC and regulatory data access
* **Evaluation Metrics**: RMSE, MAE, SMAPE, MAPE, R²
* **Visualization**: Professional decline curve and forecast plots
* **Benchmarking**: Multi-well performance comparison

Quick Start
-----------

.. code-block:: python

   import decline_curve as dca
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
   :caption: Workflows:

   workflows/single_well_workflow
   workflows/field_analysis
   workflows/state_scale_workflow

.. toctree::
   :maxdepth: 2
   :caption: Cookbook:

   cookbook/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/multiphase
   api/data_processing
   models
   source/economics
   source/scenarios
   source/prms
   source/type_curve_normalization
   source/recommend_model
   source/sensitivity
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
