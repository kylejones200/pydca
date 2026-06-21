Duong vs. Arps: Choosing the Right Model for Shale Wells
==========================================================

Arps hyperbolic with b > 1 is common for early shale data but overpredicts EUR
because b physically can't stay above 1 indefinitely. This recipe compares Arps
to the Duong and SEPD models on the same synthetic shale well.

Setup
-----

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from decline_curve.models import fit_arps, predict_arps
   from decline_curve import fit_duong, predict_duong, eur_duong
   from decline_curve import fit_sepd, predict_sepd, eur_sepd

   # Synthetic shale well: Duong-like decline
   np.random.seed(42)
   t_hist = np.arange(36)
   q_true = 2000 * (t_hist + 1) ** -1.1 * np.exp(
       1.5 / (1 - 1.1) * ((t_hist + 1) ** -0.1 - 1)
   )
   q_noisy = q_true * (1 + 0.05 * np.random.randn(36))
   q_noisy = np.maximum(q_noisy, 0)

Fit All Three Models
--------------------

.. code-block:: python

   t_fit = t_hist

   # Arps hyperbolic
   arps = fit_arps(t_fit, q_noisy, kind="hyperbolic")

   # Duong
   duong = fit_duong(t_fit, q_noisy)

   # SEPD
   sepd = fit_sepd(t_fit, q_noisy)

   print("Arps:  qi={:.0f}  di={:.4f}  b={:.2f}".format(arps.qi, arps.di, arps.b))
   print("Duong: q1={:.0f}  a={:.3f}  m={:.3f}".format(duong.q1, duong.a, duong.m))
   print("SEPD:  qi={:.0f}  tau={:.1f}  n={:.3f}".format(sepd.qi, sepd.tau, sepd.n))

Forecast 30 Years
-----------------

.. code-block:: python

   t_fc = np.arange(360)
   fc_arps  = predict_arps(t_fc, arps)
   fc_duong = predict_duong(t_fc, duong)
   fc_sepd  = predict_sepd(t_fc, sepd)

   # EUR
   eur_arps  = np.trapz(fc_arps,  t_fc)
   eur_duong_ = eur_duong(duong, t_max=360, econ_limit=5.0)
   eur_sepd_  = eur_sepd(sepd)          # closed form

   print(f"Arps EUR:  {eur_arps:>10,.0f} BOE  ← often too optimistic at b>1")
   print(f"Duong EUR: {eur_duong_:>10,.0f} BOE")
   print(f"SEPD EUR:  {eur_sepd_:>10,.0f} BOE  ← closed form via Γ(1/n)")

Plot Comparison
---------------

.. code-block:: python

   fig, ax = plt.subplots(figsize=(9, 4))
   ax.scatter(t_hist, q_noisy, s=18, color="black", label="Historical", zorder=5)
   ax.plot(t_fc, fc_arps,  label=f"Arps hyp  EUR={eur_arps/1e3:.0f} MBOE")
   ax.plot(t_fc, fc_duong, label=f"Duong     EUR={eur_duong_/1e3:.0f} MBOE")
   ax.plot(t_fc, fc_sepd,  label=f"SEPD      EUR={eur_sepd_/1e3:.0f} MBOE")
   ax.axvline(36, color="gray", linestyle="--", linewidth=0.8, label="Forecast start")
   ax.set_xlabel("Month")
   ax.set_ylabel("Rate (BOE/month)")
   ax.set_title("Arps vs. Duong vs. SEPD — 30-year forecast")
   ax.legend()
   plt.tight_layout()
   plt.savefig("shale_comparison.png", dpi=150)

When to Use Each Model
----------------------

**Arps hyperbolic** (b < 1): conventional wells and shale wells that have already
transitioned to boundary-dominated flow. Safe EUR if b is constrained ≤ 1.

**Duong**: shale wells still in fracture-dominated transient flow (first 2–5 years of
production). Parameters have physical meaning tied to fracture geometry.

**SEPD**: heterogeneous unconventional reservoirs where anomalous diffusion governs
flow. Closed-form EUR is a significant computational advantage in Monte Carlo runs.

**PLE (Ilk)**: tight gas wells; explicitly models the transient-to-BDF transition via
a power-law loss-ratio.

See Also
--------

* :doc:`full_economics` — Add royalties, CAPEX, and IRR to these forecasts
* :doc:`../models` — Mathematical formulations for all models
* :doc:`../source/economics` — Full economics API reference
