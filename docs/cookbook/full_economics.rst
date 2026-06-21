Full-Cycle Well Economics
=========================

This recipe walks through a complete economic evaluation: build a production forecast,
set up well economics with royalties and taxes, generate a cashflow table, and compute
NPV at three discount rates, IRR, payout, and breakeven price.

Setup
-----

.. code-block:: python

   import numpy as np
   import pandas as pd
   from decline_curve.models import fit_arps, predict_arps
   from decline_curve.economics import (
       WellEconomics, cashflow, npv, irr, payout, roi, breakeven_price,
   )

   # 36-month history, then forecast to 120 months
   np.random.seed(0)
   t_hist = np.arange(36)
   q_hist = 800 * np.exp(-0.028 * t_hist) + 5 * np.random.randn(36)

   params = fit_arps(t_hist, np.maximum(q_hist, 0), kind="hyperbolic")
   q_fc = predict_arps(np.arange(120), params)

Define Well Economics
---------------------

.. code-block:: python

   econ = WellEconomics(
       capex=6_000_000,           # $6 MM D&C
       price=72.0,                # $/BOE realized
       price_escalation=0.02,     # 2% annual price escalation
       royalty_rate=0.1875,       # 3/16 landowner (standard TX)
       working_interest=1.0,      # 100% WI
       severance_tax_rate=0.046,  # Texas Production Tax
       ad_valorem_rate=0.02,      # local property tax
       opex_fixed=6_000,          # $/month base LOE
       opex_variable=9.0,         # $/BOE workover + chemicals
       opex_escalation=0.02,      # 2% annual OPEX escalation
       discount_rate=0.10,        # 10% hurdle rate
   )

Cashflow Table
--------------

.. code-block:: python

   result = cashflow(q_fc, econ)

   df = pd.DataFrame(result.to_dict())
   df.index = pd.date_range("2024-01-01", periods=120, freq="MS")
   df.index.name = "Date"

   cols = ["production", "gross_revenue", "royalty", "net_revenue",
           "severance_tax", "ad_valorem", "opex", "ebitda",
           "capex_schedule", "net_cashflow", "cumulative_cashflow"]
   print(df[cols].head(12).to_string(float_format="${:,.0f}".format))

Economic Metrics
----------------

.. code-block:: python

   print("=" * 40)
   print(f"NPV  @ 8%:   ${npv(result, discount_rate=0.08):>12,.0f}")
   print(f"NPV  @ 10%:  ${npv(result, discount_rate=0.10):>12,.0f}")
   print(f"NPV  @ 12%:  ${npv(result, discount_rate=0.12):>12,.0f}")
   print(f"IRR:          {irr(result):.1%}")
   print(f"Payout:       month {payout(result)}")
   print(f"ROI:          {roi(result):.2f}x")
   print(f"Breakeven:   ${breakeven_price(q_fc, econ):.2f}/BOE")

Cumulative Cashflow Plot
------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

   months = np.arange(120)
   ax1.bar(months, result.net_cashflow / 1e3, color="steelblue", width=0.9)
   ax1.axhline(0, color="black", linewidth=0.7)
   ax1.set_ylabel("Net Cash Flow ($K/month)")
   ax1.set_title("Well Economics — Monthly and Cumulative")

   ax2.plot(months, result.cumulative_cashflow / 1e6, color="darkgreen")
   ax2.axhline(0, color="black", linewidth=0.7)
   ax2.set_xlabel("Month")
   ax2.set_ylabel("Cumulative NCF ($MM)")

   plt.tight_layout()
   plt.savefig("cashflow.png", dpi=150)

Price Sensitivity
-----------------

.. code-block:: python

   prices = [50, 60, 70, 80, 90]
   rows = []
   for p in prices:
       e = WellEconomics(**{**econ.__dict__, "price": p})
       r = cashflow(q_fc, e)
       rows.append({
           "Price ($/BOE)": p,
           "NPV ($MM)": npv(r) / 1e6,
           "IRR": f"{irr(r):.1%}",
           "Payout (mo)": payout(r),
       })
   print(pd.DataFrame(rows).to_string(index=False))

Breakeven Sensitivity
---------------------

The breakeven price already uses :func:`~decline_curve.economics.breakeven_price`
under the hood, but you can also run a price sweep to see the NPV profile:

.. code-block:: python

   sweep_prices = np.linspace(20, 120, 50)
   npvs = []
   for p in sweep_prices:
       e = WellEconomics(**{**econ.__dict__, "price": p})
       npvs.append(npv(cashflow(q_fc, e)) / 1e6)

   fig, ax = plt.subplots(figsize=(7, 3))
   ax.plot(sweep_prices, npvs, color="steelblue")
   ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
   ax.set_xlabel("Oil Price ($/BOE)")
   ax.set_ylabel("NPV ($MM @ 10%)")
   ax.set_title("NPV vs. Price")
   plt.tight_layout()
   plt.savefig("npv_vs_price.png", dpi=150)

See Also
--------

* :doc:`shale_variants` — Use Duong/SEPD instead of Arps for shale forecasts
* :doc:`../source/economics` — Full API reference for the economics module
* :doc:`../models` — Model selection guide
