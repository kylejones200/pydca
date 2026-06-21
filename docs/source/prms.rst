PRMS Reserves Classification
============================

``decline_curve.prms`` provides SPE-PRMS 2018 compliant 1P/2P/3P reserves
classification using DCA-based Monte Carlo simulation. It is the first
open-source implementation of probabilistic PRMS classification built
directly on Arps decline curves — no material balance or volumetric
estimate required.

Background
----------

The **Petroleum Resources Management System (SPE-PRMS 2018)** defines three
reserve categories based on the probability of achieving or exceeding a
given production volume:

.. list-table::
   :header-rows: 1
   :widths: 10 15 20 55

   * - Category
     - PRMS label
     - Probability
     - Meaning
   * - 1P
     - Proved
     - ≥ 90 % (P90)
     - Conservative — 90 % of Monte Carlo draws exceed this EUR
   * - 2P
     - Proved + Probable
     - ≥ 50 % (P50)
     - Best estimate — median EUR
   * - 3P
     - Proved + Probable + Possible
     - ≥ 10 % (P10)
     - Optimistic — only 10 % of draws exceed this EUR

A key convention: **P90** in PRMS means *90 % probability of achieving at
least this amount*, which corresponds to the **10th percentile** of the EUR
distribution (90 % of draws are above it). This is the *opposite* of the
common "P90 = pessimistic" shorthand used in some forecasting contexts.

.. note::

   pydca uses the correct SPE-PRMS convention throughout:
   ``P1 = np.percentile(eur_draws, 10)`` — conservative; 90 % of draws exceed this.

Method
------

``classify_reserves`` generates probabilistic reserves in three steps:

1. **Monte Carlo EUR draws** — calls :func:`~decline_curve.probabilistic_forecast.probabilistic_forecast`
   to generate *n_draws* Arps parameter sets by resampling the fitted covariance
   matrix. Each parameter set is propagated to a *horizon*-month forecast.
2. **EUR per draw** — integrates each forecast (summing monthly volumes) down
   to the economic limit rate.
3. **PRMS percentiles** — maps the 10th, 50th, and 90th percentiles of the EUR
   distribution to 1P, 2P, and 3P.

Data Classes
------------

.. autoclass:: decline_curve.prms.ReservesClassification
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: decline_curve.prms.classify_reserves

.. autofunction:: decline_curve.prms.classify_reserves

Usage Example
-------------

Single-well classification::

   import pandas as pd
   import numpy as np
   import decline_curve as dca

   # 36 months of history
   dates = pd.date_range('2021-01-01', periods=36, freq='MS')
   q = 2500 * np.exp(-0.04 * np.arange(36))
   series = pd.Series(q, index=dates)

   # Classify with 500 Monte Carlo draws, 30-year horizon, 5 BOE/month econ limit
   rc = dca.classify_reserves(
       series,
       kind='modified_hyperbolic',
       horizon=360,
       n_draws=500,
       econ_limit=5.0,
       seed=42,
   )
   print(rc.to_series())
   # 1P (P90)     18,432
   # 2P (P50)     23,710
   # 3P (P10)     29,085
   # Name: reserves_boe, dtype: float64

   print(f'Uncertainty ratio (P3/P1): {rc.uncertainty_ratio:.2f}x')

Inspect the full EUR distribution::

   import matplotlib.pyplot as plt

   plt.hist(rc.eur_distribution, bins=40, color='steelblue', edgecolor='white')
   plt.axvline(rc.p1, color='green',  ls='--', label='1P (P90)')
   plt.axvline(rc.p2, color='orange', ls='--', label='2P (P50)')
   plt.axvline(rc.p3, color='red',    ls='--', label='3P (P10)')
   plt.xlabel('EUR (BOE)')
   plt.ylabel('Draw count')
   plt.legend()
   plt.title('PRMS EUR Distribution')
   plt.tight_layout()

Using the ``recommend_model`` helper to auto-select the decline model::

   model = dca.recommend_model(series)          # e.g. 'modified_hyperbolic'
   rc = dca.classify_reserves(series, kind=model, n_draws=500)

.. seealso::

   :doc:`recommend_model` — auto-select the right decline model before classifying.

   :doc:`../cookbook/reserves_classification` — end-to-end PRMS workflow.
