Price Scenario Analysis
=======================

``decline_curve.scenarios`` provides multi-scenario economic analysis using the
full-cycle :class:`~decline_curve.economics.WellEconomics` engine. Each scenario
can carry its own royalty structure, CAPEX, and tax rates — not just a price change.

Data Classes
------------

.. autoclass:: decline_curve.scenarios.PriceScenario
   :members:
   :undoc-members:

.. autoclass:: decline_curve.scenarios.ScenarioResult
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: decline_curve.scenarios.run_price_scenarios

.. autofunction:: decline_curve.scenarios.run_multi_phase_scenarios

.. autofunction:: decline_curve.scenarios.compare_scenarios

Quick Reference
---------------

Single-phase scenario sweep::

   import numpy as np
   import decline_curve as dca

   # 60 months of production
   t = np.arange(60, dtype=float)
   q = 2000 / (1 + 1.4 * 0.08 * t) ** (1 / 1.4)

   scenarios = [
       dca.PriceScenario('low',  oil_price=50.0, capex=6_000_000, royalty_rate=0.1875),
       dca.PriceScenario('base', oil_price=70.0, capex=6_000_000, royalty_rate=0.1875),
       dca.PriceScenario('high', oil_price=90.0, capex=6_000_000, royalty_rate=0.1875),
   ]

   df = dca.run_price_scenarios(q, scenarios)
   print(df[['scenario', 'npv', 'irr', 'payout_month', 'roi', 'breakeven_price']])

Typical output::

   scenario          npv       irr  payout_month       roi  breakeven_price
        low  -2,341,204      -0.38            -1      0.39            64.21
       base     412,880       0.14            47      1.07            64.21
       high   3,167,214       0.57            24      1.74            64.21

Compare scenarios against a base case::

   cmp = dca.compare_scenarios(df)
   print(cmp[['scenario', 'npv_vs_base', 'npv_pct_change', 'payback_vs_base']])

Multi-phase (oil + gas + water)::

   oil_q   = 1200 * np.exp(-0.04 * t)
   gas_q   =  800 * np.exp(-0.03 * t)   # MCF/month
   water_q =  400 * np.exp(-0.02 * t)

   scenarios_mp = [
       dca.PriceScenario('base', oil_price=65.0, gas_price=3.0,
                         water_price=-1.50, capex=7_000_000),
   ]

   df_mp = dca.run_multi_phase_scenarios(
       oil_q, scenarios_mp,
       gas_production=gas_q,
       water_production=water_q,
   )

Economics Engine
----------------

All scenario functions delegate to the same engine as :func:`~decline_curve.economics.cashflow`:

.. math::

   \text{Gross Revenue} &= q \cdot P \cdot WI \\
   \text{Net Revenue}   &= \text{Gross} \cdot (1 - r) \\
   \text{EBITDA}        &= \text{Net} \cdot (1 - \tau_{sev} - \tau_{av}) - \text{OPEX} \\
   \text{NCF}           &= \text{EBITDA} - \text{CAPEX}_{t=0}

NPV uses compound monthly discounting:

.. math::

   r_m = (1 + r_{annual})^{1/12} - 1

.. seealso::

   :doc:`economics` — full cashflow engine reference.

   :doc:`../cookbook/test_price_deck` — cookbook for multi-scenario price deck testing.
