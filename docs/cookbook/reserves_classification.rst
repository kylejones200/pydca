PRMS Reserves Classification
============================

This recipe classifies a well's reserves into PRMS 1P/2P/3P categories using
Monte Carlo EUR simulation. The workflow takes ~10 seconds for 500 draws and a
30-year horizon.

Setup
-----

::

   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import decline_curve as dca

   # 48 months of production history
   dates = pd.date_range('2020-01-01', periods=48, freq='MS')
   q = 3500 * (1 + 1.4 * 0.09 * np.arange(48)) ** (-1 / 1.4)
   series = pd.Series(q, index=dates, name='oil_bbl')

Step 1 — Auto-select the decline model
---------------------------------------

::

   model = dca.recommend_model(series)
   print(f'Recommended model: {model!r}')
   # → 'modified_hyperbolic'

Step 2 — Classify reserves
---------------------------

::

   rc = dca.classify_reserves(
       series,
       kind=model,          # use the recommended model
       horizon=360,         # 30-year forecast window
       n_draws=500,         # Monte Carlo samples
       econ_limit=5.0,      # abandon below 5 BOE/month
       seed=42,
   )

   print(rc.to_series())

Expected output::

   1P (P90)    21,847.0
   2P (P50)    26,104.0
   3P (P10)    31,822.0
   Name: reserves_boe, dtype: float64

   print(f'Uncertainty ratio: {rc.uncertainty_ratio:.2f}x')
   # → 1.46x

Step 3 — Inspect the EUR distribution
---------------------------------------

::

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.hist(rc.eur_distribution / 1000, bins=40,
           color='steelblue', edgecolor='white', alpha=0.8)
   for label, val, color in [
       ('1P (P90)', rc.p1, 'green'),
       ('2P (P50)', rc.p2, 'orange'),
       ('3P (P10)', rc.p3, 'crimson'),
   ]:
       ax.axvline(val / 1000, color=color, ls='--', lw=1.5, label=f'{label}: {val/1000:,.1f} MBOE')
   ax.set_xlabel('EUR (MBOE)')
   ax.set_ylabel('Draw count')
   ax.set_title('PRMS EUR Distribution — Monte Carlo (n=500)')
   ax.legend()
   plt.tight_layout()

Step 4 — Export to dict / DataFrame
-------------------------------------

::

   # Simple dict for reporting
   print(rc.to_dict())

   # DataFrame row for a multi-well summary table
   import pandas as pd
   well_df = pd.DataFrame([{
       'well_id': 'SMITH_1H',
       **rc.to_dict(),
   }])
   print(well_df[['well_id', 'p1', 'p2', 'p3', 'uncertainty_ratio']])

Multi-well loop
---------------

::

   results = []
   for well_id, s in production_dict.items():
       kind = dca.recommend_model(s)
       rc = dca.classify_reserves(s, kind=kind, horizon=360, n_draws=300, seed=0)
       results.append({'well_id': well_id, **rc.to_dict()})

   reserves_df = pd.DataFrame(results).set_index('well_id')
   print(reserves_df[['p1', 'p2', 'p3', 'uncertainty_ratio']].sort_values('p2', ascending=False))

Notes
-----

* ``n_draws=500`` is adequate for portfolio work; use 1000–2000 for regulatory filings.
* ``econ_limit`` should match your production threshold for booking proved reserves.
* For tight oil wells with b > 1, ``kind='modified_hyperbolic'`` gives the most conservative 1P
  estimate because the terminal exponential tail caps the EUR.
* ``uncertainty_ratio`` (P3/P1) above 2.5× typically signals a new or data-poor well.

.. seealso::

   :doc:`../source/prms` — PRMS module API reference.

   :doc:`../source/recommend_model` — model auto-selection reference.
