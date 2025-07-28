Plotting (plot)
================

The plotting module provides professional visualization capabilities with Tufte-style aesthetics.

.. automodule:: decline_analysis.plot
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

plot_forecast
~~~~~~~~~~~~~

.. autofunction:: decline_analysis.plot.plot_forecast

plot_decline_curve
~~~~~~~~~~~~~~~~~~

.. autofunction:: decline_analysis.plot.plot_decline_curve

plot_benchmark_results
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: decline_analysis.plot.plot_benchmark_results

tufte_style
~~~~~~~~~~~

.. autofunction:: decline_analysis.plot.tufte_style

Styling and Aesthetics
----------------------

The plotting module follows Edward Tufte's principles of data visualization:

* **Minimize chart junk**: Remove unnecessary visual elements
* **Maximize data-ink ratio**: Focus on the data, not decoration
* **Use subtle grid lines**: Provide reference without distraction
* **Clean typography**: Clear, readable fonts and labels
* **Meaningful colors**: Use color purposefully to convey information

Example Usage
-------------

Basic Forecast Plot
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_analysis as dca
   import pandas as pd
   
   # Create sample data
   dates = pd.date_range('2020-01-01', periods=24, freq='MS')
   production = [1000 - i*30 for i in range(24)]
   series = pd.Series(production, index=dates)
   
   # Generate forecast
   forecast = dca.forecast(series, model="arps", horizon=12)
   
   # Create plot
   dca.plot(series, forecast, title="Well Production Forecast")

Decline Curve Analysis Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_analysis.models import fit_arps
   from decline_analysis.plot import plot_decline_curve
   import numpy as np
   
   # Fit Arps model
   t = np.arange(len(series))
   q = series.values
   params = fit_arps(t, q, kind="hyperbolic")
   
   # Plot decline curve
   plot_decline_curve(t, q, params, title="Hyperbolic Decline Analysis")

Benchmark Results Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run benchmark analysis
   results = dca.benchmark(well_data, model="arps", top_n=10)
   
   # Plot results
   from decline_analysis.plot import plot_benchmark_results
   plot_benchmark_results(results, metric='rmse', title="Model Performance")

Customization Options
---------------------

The plotting functions accept various parameters for customization:

* **title**: Plot title
* **filename**: Save plot to file
* **show_metrics**: Display evaluation metrics on plot
* **colors**: Custom color schemes
* **figure_size**: Control plot dimensions

All plots use matplotlib as the backend, allowing for further customization using standard matplotlib commands.
