Economics
=========

The economics module provides full-cycle financial evaluation for oil and gas wells.
It goes well beyond simple NPV/payback: royalties, working interest, severance tax,
ad valorem, fixed and variable OPEX, CAPEX, IRR, breakeven price, and probabilistic
NPV distributions are all first-class citizens.

.. automodule:: decline_curve.economics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The evaluation workflow is:

1. Build a production forecast (Arps, Duong, PLE, SEPD, or probabilistic)
2. Describe the well economics with :class:`~decline_curve.economics.WellEconomics`
3. Call :func:`~decline_curve.economics.cashflow` to get a month-by-month
   :class:`~decline_curve.economics.CashflowResult`
4. Compute metrics: :func:`~decline_curve.economics.npv`,
   :func:`~decline_curve.economics.irr`, :func:`~decline_curve.economics.payout`,
   :func:`~decline_curve.economics.roi`,
   :func:`~decline_curve.economics.breakeven_price`

Revenue Model
-------------

.. math::

   \text{Gross Revenue}_t = Q_t \times P_t \times WI

.. math::

   \text{Net Revenue}_t = \text{Gross Revenue}_t \times (1 - r_{\text{royalty}})

.. math::

   \text{EBITDA}_t = \text{Net Revenue}_t
       - \text{Severance Tax}_t
       - \text{Ad Valorem}_t
       - \text{OPEX}_t

Where :math:`P_t = P_0 \cdot (1 + \delta_{\text{price}})^{t/12}` with optional price
escalation.

.. math::

   \text{Net Cash Flow}_t = \text{EBITDA}_t - \text{CAPEX}_t

CAPEX enters as a single outflow at :math:`t = 0`.

Discounting
-----------

NPV uses a **compound monthly rate** derived from the annual rate:

.. math::

   r_m = (1 + r_{\text{annual}})^{1/12} - 1

.. math::

   NPV = \sum_{t=0}^{T} \frac{NCF_t}{(1 + r_m)^t}

The legacy :func:`~decline_curve.economics.economic_metrics` function preserves the
original simple :math:`r_m = r/12` convention for backwards compatibility.

Quick-Start Example
-------------------

.. code-block:: python

   import numpy as np
   from decline_curve.models import fit_arps, predict_arps
   from decline_curve.economics import WellEconomics, cashflow, npv, irr, payout, roi

   # 1. Production forecast
   t_hist = np.arange(24)
   q_hist = 1000 * np.exp(-0.025 * t_hist)
   params = fit_arps(t_hist, q_hist, kind="hyperbolic")
   q_forecast = predict_arps(np.arange(120), params)

   # 2. Define economics
   econ = WellEconomics(
       capex=5_000_000,          # $5 MM CAPEX
       price=70.0,               # $/BOE
       royalty_rate=0.1875,      # 3/16 landowner royalty
       working_interest=1.0,
       severance_tax_rate=0.046, # Texas rate
       ad_valorem_rate=0.02,
       opex_fixed=5_000,         # $5,000/month fixed LOE
       opex_variable=8.0,        # $8/BOE variable LOE
       discount_rate=0.10,
   )

   # 3. Cashflow
   result = cashflow(q_forecast, econ)

   # 4. Metrics
   print(f"NPV (10%): ${npv(result):>12,.0f}")
   print(f"IRR:       {irr(result):.1%}")
   print(f"Payout:    month {payout(result)}")
   print(f"ROI:       {roi(result):.2f}x")

Breakeven Price
---------------

Find the commodity price at which the well breaks even (NPV = 0):

.. code-block:: python

   from decline_curve.economics import breakeven_price

   be = breakeven_price(q_forecast, econ)
   print(f"Breakeven: ${be:.2f}/BOE")

The function uses Brent's method over the range [$0, $500].

Cashflow Table
--------------

The :class:`~decline_curve.economics.CashflowResult` exposes every line item as a
NumPy array and can be converted to a DataFrame:

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame(result.to_dict())
   df.index = pd.date_range("2024-01-01", periods=len(q_forecast), freq="MS")
   print(df[["production", "net_revenue", "opex", "ebitda", "net_cashflow"]].head(12))

NPV Sensitivity to Discount Rate
---------------------------------

.. code-block:: python

   for rate in [0.08, 0.10, 0.12, 0.15, 0.20]:
       print(f"  {rate:.0%}  →  ${npv(result, discount_rate=rate):>12,.0f}")

Probabilistic Economics
-----------------------

Pair with :func:`~decline_curve.probabilistic_forecast.probabilistic_forecast` to get
P10/P50/P90 NPV distributions:

.. code-block:: python

   import pandas as pd
   from decline_curve.probabilistic_forecast import probabilistic_forecast
   from decline_curve.risk_report import calculate_risk_metrics

   dates = pd.date_range("2020-01-01", periods=36, freq="MS")
   prod = pd.Series(1000 * np.exp(-0.03 * np.arange(36)), index=dates)

   forecast = probabilistic_forecast(
       prod, model="arps", kind="hyperbolic", horizon=24,
       n_draws=500, price=70.0, opex=15.0, seed=42,
   )
   risk = calculate_risk_metrics(forecast, npv_threshold=1_000_000)
   print(f"P(NPV > 0):     {risk.prob_positive_npv:.1%}")
   print(f"Expected NPV:   ${risk.expected_npv:,.0f}")
   print(f"VaR (P90):      ${risk.value_at_risk_90:,.0f}")

WellEconomics Field Reference
------------------------------

+------------------------+-------------------+-----------------------------------------------+
| Field                  | Default           | Description                                   |
+========================+===================+===============================================+
| ``capex``              | (required)        | One-time CAPEX at t=0, in $                   |
+------------------------+-------------------+-----------------------------------------------+
| ``price``              | (required)        | Commodity price, $/BOE                        |
+------------------------+-------------------+-----------------------------------------------+
| ``price_escalation``   | 0.0               | Annual price escalation rate                  |
+------------------------+-------------------+-----------------------------------------------+
| ``royalty_rate``       | 0.1875            | Royalty fraction (3/16 landowner standard)    |
+------------------------+-------------------+-----------------------------------------------+
| ``working_interest``   | 1.0               | WI fraction                                   |
+------------------------+-------------------+-----------------------------------------------+
| ``severance_tax_rate`` | 0.046             | Severance tax on net revenue (TX default)     |
+------------------------+-------------------+-----------------------------------------------+
| ``ad_valorem_rate``    | 0.02              | Ad valorem tax on net revenue                 |
+------------------------+-------------------+-----------------------------------------------+
| ``opex_fixed``         | 0.0               | Fixed LOE, $/month                            |
+------------------------+-------------------+-----------------------------------------------+
| ``opex_variable``      | 0.0               | Variable LOE, $/BOE                           |
+------------------------+-------------------+-----------------------------------------------+
| ``opex_escalation``    | 0.0               | Annual OPEX escalation rate                   |
+------------------------+-------------------+-----------------------------------------------+
| ``discount_rate``      | 0.10              | Annual discount rate for NPV (10% = 0.10)     |
+------------------------+-------------------+-----------------------------------------------+
| ``econ_limit``         | 0.0               | Abandon threshold, $/month net (0 = run end)  |
+------------------------+-------------------+-----------------------------------------------+

See Also
--------

* :doc:`../models` — Decline model reference
* :doc:`../cookbook/full_economics` — Full worked example with cashflow table
* :doc:`../cookbook/shale_variants` — Duong vs Arps comparison with economics
