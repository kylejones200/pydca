Type Curve Normalization
========================

``decline_curve.type_curve_normalization`` implements **Fetkovich-style** type curve
analysis — the standard technique for comparing wells with different initial rates and
decline parameters on a single log-log plot.

Background
----------

A **type curve** is a dimensionless decline curve. By normalizing production data
to reference qi and Di values, wells of very different scale can be overlaid and
matched to a family of theoretical Arps curves parameterized by the shape factor
*b*. Once the best-fit *b* is found in normalized space, the parameters are
de-normalized back to the original production units.

Normalization conventions used here follow Fetkovich (1980):

.. math::

   t_D = D_i \cdot t \qquad q_D = q / q_i

so that all wells start at :math:`(t_D = 0,\; q_D = 1)` and diverge according to
their *b* factor.

Data Classes
------------

.. autoclass:: decline_curve.type_curve_normalization.TypeCurveMatch
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: decline_curve.type_curve_normalization.generate_arps_type_curve

.. autofunction:: decline_curve.type_curve_normalization.normalize_production_data

.. autofunction:: decline_curve.type_curve_normalization.match_type_curve

.. autofunction:: decline_curve.type_curve_normalization.denormalize_match

Workflow
--------

The standard four-step workflow::

   import numpy as np
   import decline_curve as dca
   from decline_curve.models import ArpsParams, predict_arps

   # Step 1 — build (or load) production data
   t = np.arange(48, dtype=float)                          # 48 months
   q = predict_arps(t, ArpsParams(qi=1500, di=0.07, b=1.1))

   # Step 2 — normalize to dimensionless time and rate
   qi_ref = q[0]          # use observed peak rate as reference
   di_ref = 0.07          # use estimated initial decline
   t_norm, q_norm, factors = dca.normalize_production_data(t, q, qi_ref, di_ref)

   # Step 3 — match against a b-value family
   match = dca.match_type_curve(
       t_norm,
       q_norm,
       b_values=np.arange(0.0, 2.1, 0.1),   # sweep b from 0 to 2
   )
   print(f'Best b = {match.matched_params.b:.2f}  RMSE = {match.match_error:.4f}')

   # Step 4 — de-normalize to recover well-scale parameters
   params = dca.denormalize_match(match, factors)
   print(f'qi = {params.qi:.0f}  Di = {params.di:.4f}/mo  b = {params.b:.2f}')

Free-*b* fitting (let the optimizer determine b simultaneously with qi and Di)::

   match = dca.match_type_curve(t_norm, q_norm)   # b_values=None → free-b fit
   params = dca.denormalize_match(match, factors)

Generating type curve families for plotting::

   import matplotlib.pyplot as plt

   t_plot = np.linspace(0, 15, 200)
   fig, ax = plt.subplots()
   for b in [0.0, 0.5, 1.0, 1.5, 2.0]:
       _, q_tc = dca.generate_arps_type_curve(1.0, 1.0, b, t_plot)
       ax.plot(t_plot, q_tc, label=f'b = {b:.1f}')

   ax.set_xscale('log')
   ax.set_yscale('log')
   ax.set_xlabel('Dimensionless time $t_D$')
   ax.set_ylabel('Dimensionless rate $q_D$')
   ax.set_title('Arps Type Curve Family')
   ax.legend()
   plt.tight_layout()

.. seealso::

   :doc:`../cookbook/type_curve_workflow` — multi-well type curve comparison.

   :class:`~decline_curve.models.ArpsParams` — Arps parameter dataclass.
