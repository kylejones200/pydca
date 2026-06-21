Type Curve Normalization Workflow
==================================

This recipe normalizes a group of wells onto a common dimensionless axis,
matches each to the Arps type curve family, and identifies which *b*-factor
best characterizes the group.

Setup
-----

::

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import decline_curve as dca
   from decline_curve.models import ArpsParams, predict_arps

Single-well match
-----------------

::

   # Synthetic well — true params: qi=2000, Di=0.08, b=1.3
   t = np.arange(48, dtype=float)
   q = predict_arps(t, ArpsParams(qi=2000, di=0.08, b=1.3))

   # Normalize (use observed peak rate and estimated initial decline)
   qi_ref, di_ref = q[0], 0.08
   t_norm, q_norm, factors = dca.normalize_production_data(t, q, qi_ref, di_ref)

   # Match against b family from 0 to 2 (step 0.1)
   match = dca.match_type_curve(t_norm, q_norm, b_values=np.arange(0.0, 2.1, 0.1))
   params = dca.denormalize_match(match, factors)

   print(f'Best b     = {match.matched_params.b:.2f}')  # → 1.30
   print(f'RMSE       = {match.match_error:.6f}')
   print(f'Correlation= {match.correlation:.4f}')
   print(f'qi (actual)= {params.qi:.0f}')              # → 2000
   print(f'Di (actual)= {params.di:.4f}/mo')           # → 0.0800

Plot normalized data against matched curve::

   fig, ax = plt.subplots(figsize=(7, 4))
   ax.scatter(t_norm, q_norm, s=18, label='Observed (normalized)', zorder=3)
   ax.plot(t_norm, match.matched_curve, color='crimson', label=f'Match  b={match.matched_params.b:.2f}')
   ax.set_xlabel('Dimensionless time $t_D$')
   ax.set_ylabel('Dimensionless rate $q_D$')
   ax.set_title('Type Curve Match')
   ax.legend()
   plt.tight_layout()

Multi-well overlay
------------------

Compare several wells on a single normalized axis to see which *b*-family
they fall into::

   wells = {
       'A': ArpsParams(qi=1800, di=0.07, b=1.1),
       'B': ArpsParams(qi=3200, di=0.10, b=1.4),
       'C': ArpsParams(qi=1200, di=0.05, b=0.9),
   }
   t = np.arange(60, dtype=float)
   b_family = np.arange(0.0, 2.1, 0.5)

   fig, ax = plt.subplots(figsize=(8, 5))

   # Draw type curve family in background
   t_bg = np.linspace(0.01, 20, 300)
   for b in b_family:
       _, q_tc = dca.generate_arps_type_curve(1.0, 1.0, b, t_bg)
       ax.plot(t_bg, q_tc, color='lightgray', lw=0.8)
       ax.text(t_bg[-1] * 1.02, q_tc[-1], f'b={b:.1f}', fontsize=7, va='center')

   # Overlay each well
   for name, p in wells.items():
       q = predict_arps(t, p)
       t_n, q_n, factors = dca.normalize_production_data(t, q, p.qi, p.di)
       match = dca.match_type_curve(t_n, q_n, b_values=b_family)
       ax.scatter(t_n, q_n, s=12, label=f'{name}  (match b={match.matched_params.b:.1f})')

   ax.set_xscale('log')
   ax.set_yscale('log')
   ax.set_xlabel('$t_D$')
   ax.set_ylabel('$q_D$')
   ax.set_title('Multi-well type curve comparison')
   ax.legend(fontsize=8)
   plt.tight_layout()

Group b-factor statistics
--------------------------

::

   b_matches = []
   for name, p in wells.items():
       q = predict_arps(t, p)
       t_n, q_n, factors = dca.normalize_production_data(t, q, p.qi, p.di)
       match = dca.match_type_curve(t_n, q_n, b_values=np.arange(0.0, 2.1, 0.1))
       b_matches.append({'well': name, 'b': match.matched_params.b, 'rmse': match.match_error})

   df_b = pd.DataFrame(b_matches)
   print(f"Mean b = {df_b['b'].mean():.2f}  (std {df_b['b'].std():.2f})")

Using the matched b for a group type curve
-------------------------------------------

::

   b_p50 = float(df_b['b'].median())
   qi_p50 = np.median([p.qi for p in wells.values()])
   di_p50 = np.median([p.di for p in wells.values()])

   t_tc = np.arange(120, dtype=float)
   group_curve = predict_arps(t_tc, ArpsParams(qi=qi_p50, di=di_p50, b=b_p50))
   print(f'Group type curve: qi={qi_p50:.0f}  Di={di_p50:.4f}/mo  b={b_p50:.2f}')

.. seealso::

   :doc:`../source/type_curve_normalization` — module API reference.

   :doc:`shale_variants` — choosing the right decline model for each type curve group.
