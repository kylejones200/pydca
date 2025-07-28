Sensitivity Analysis
===================

The sensitivity analysis module provides tools for evaluating how changes in Arps decline parameters and economic conditions affect well performance and profitability.

Overview
--------

Sensitivity analysis is crucial in petroleum engineering for:

* Understanding parameter uncertainty impacts
* Risk assessment and decision making
* Economic optimization
* Portfolio evaluation

Functions
---------

.. automodule:: decline_analysis.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_analysis as dca
   
   # Define parameter grid to test
   param_grid = [
       (1000, 0.05, 0.3),  # (qi, di, b)
       (1000, 0.10, 0.5),
       (1000, 0.15, 0.7),
       (800, 0.05, 0.3),
       (800, 0.10, 0.5),
   ]
   
   # Define price scenarios
   prices = [40, 50, 60, 70, 80]  # $/bbl
   
   # Run sensitivity analysis
   results = dca.sensitivity_analysis(
       param_grid=param_grid,
       prices=prices,
       opex=15.0,  # $/bbl operating cost
       discount_rate=0.10,
       t_max=240,  # 20 years
       econ_limit=10.0  # bbl/month
   )
   
   print(results.head())
   print(f"Total scenarios: {len(results)}")

Advanced Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter results for economic viability
   viable_wells = results[results['NPV'] > 0]
   
   # Find best case scenario
   best_case = results.loc[results['NPV'].idxmax()]
   print(f"Best NPV: ${best_case['NPV']:,.2f}")
   print(f"Best EUR: {best_case['EUR']:,.0f} bbls")
   
   # Analyze price sensitivity
   price_sensitivity = results.groupby('price').agg({
       'NPV': ['mean', 'std', 'min', 'max'],
       'EUR': 'mean',
       'Payback_month': 'mean'
   })
   
   print(price_sensitivity)

Mathematical Background
----------------------

The sensitivity analysis evaluates combinations of:

**Arps Decline Parameters:**

* :math:`q_i` - Initial production rate
* :math:`D_i` - Initial decline rate  
* :math:`b` - Decline exponent

**Economic Parameters:**

* Oil/gas price ($/unit)
* Operating expenses ($/unit)
* Discount rate (annual)
* Economic limit (minimum rate)

**Calculated Metrics:**

* **EUR (Estimated Ultimate Recovery):**
  
  .. math::
     EUR = \int_0^{t_{econ}} q(t) \, dt

* **NPV (Net Present Value):**
  
  .. math::
     NPV = \sum_{t=0}^{T} \frac{CF_t}{(1+r)^{t/12}}

* **Payback Period:** Time to positive cumulative cash flow

Results Interpretation
---------------------

The sensitivity analysis returns a DataFrame with columns:

* ``qi``, ``di``, ``b`` - Arps parameters tested
* ``price`` - Oil/gas price scenario
* ``EUR`` - Estimated Ultimate Recovery (bbls or Mcf)
* ``NPV`` - Net Present Value ($)
* ``Payback_month`` - Payback period (months, -1 if never)

Use these results to:

1. **Identify robust parameters** that perform well across scenarios
2. **Assess price risk** by comparing NPV distributions
3. **Optimize development** by finding best parameter combinations
4. **Portfolio planning** by ranking opportunities

Best Practices
--------------

* **Parameter Ranges:** Use realistic ranges based on field data
* **Price Scenarios:** Include both optimistic and pessimistic cases
* **Economic Limits:** Set appropriate minimum rates for your field
* **Time Horizons:** Consider typical well life (10-30 years)
* **Discount Rates:** Use company-specific hurdle rates

See Also
--------

* :doc:`economics` - Economic evaluation functions
* :doc:`reserves` - Reserve estimation utilities
* :doc:`models` - Arps decline curve models
