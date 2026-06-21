Model Auto-Selection
====================

``dca.recommend_model`` analyzes a production history and returns the name of the
decline model most likely to give a good fit. It removes the guesswork from choosing
between Arps exponential, hyperbolic, modified hyperbolic, and shale-era models.

.. autofunction:: decline_curve.dca.recommend_model

Decision Logic
--------------

The recommender runs four sequential checks:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Condition
     - Interpretation
     - Recommendation
   * - ``drive_mechanism='shale'``
     - Caller signals unconventional reservoir
     - ``'modified_hyperbolic'``
   * - ``drive_mechanism='water_drive'`` or ``'compaction'``
     - Strong pressure support → near-constant D
     - ``'exponential'``
   * - ``drive_mechanism='solution_gas'``
     - Volumetric depletion
     - ``'hyperbolic'``
   * - CV(D) < 0.10 (near-constant D)
     - Decline rate barely changes → exponential behavior
     - ``'exponential'``
   * - Fast early drop (>65 % in 6 months) AND high CV(D) > 0.35
     - Fracture-dominated transient with erratic early rates
     - ``'duong'``
   * - b-estimate > 1.2 AND steep initial decline (D > 8 %/mo)
     - Super-hyperbolic transient — SEC terminal decline needed
     - ``'modified_hyperbolic'``
   * - Late-period D < 50 % of early-period D, b-estimate > 0.8
     - D falls over time but not steeply
     - ``'hyperbolic'``
   * - Late-period D < 50 % of early-period D, b-estimate ≤ 0.8
     - Mild curvature
     - ``'harmonic'``
   * - Default
     - Safe fallback for any other pattern
     - ``'hyperbolic'``

Usage
-----

Basic usage::

   import pandas as pd
   import numpy as np
   import decline_curve as dca

   dates = pd.date_range('2021-01-01', periods=36, freq='MS')
   q = 1800 * (1 + 1.4 * 0.07 * np.arange(36)) ** (-1 / 1.4)
   series = pd.Series(q, index=dates)

   model = dca.recommend_model(series)
   print(model)   # 'modified_hyperbolic'

   # Use the recommendation directly in a forecast
   forecast = dca.forecast(series, model='arps', kind=model, horizon=120)

With drive mechanism hint::

   # Waterflood well — override data-driven logic
   model = dca.recommend_model(series, drive_mechanism='water_drive')
   # Returns 'exponential' regardless of data shape

Integrate with ``classify_reserves``::

   model = dca.recommend_model(series)
   rc = dca.classify_reserves(series, kind=model, horizon=360, n_draws=500)
   print(rc.to_series())

.. note::

   ``recommend_model`` is a fast heuristic, not a rigorous model selection test.
   For production workflows, confirm the recommendation by comparing AIC/BIC across
   models using :func:`~decline_curve.model_comparison.compare_models`.
