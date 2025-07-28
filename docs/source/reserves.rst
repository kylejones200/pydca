Reserves
========

The reserves module provides tools for estimating ultimate recovery (EUR) from decline curve analysis and production forecasting with economic limits.

Overview
--------

Reserve estimation is fundamental to:

* Asset valuation and portfolio management
* Development planning and optimization
* Regulatory reporting and compliance
* Investment decision making

Functions
---------

.. automodule:: decline_analysis.reserves
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic Reserve Estimation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_analysis as dca
   
   # Define Arps parameters
   params = dca.ArpsParams(qi=1000, di=0.10, b=0.5)
   
   # Calculate reserves with economic limit
   reserves_result = dca.reserves(
       params=params,
       t_max=240,      # 20 years
       dt=1.0,         # monthly time steps
       econ_limit=10.0 # minimum economic rate
   )
   
   print(f"EUR: {reserves_result['eur']:,.0f} bbls")
   print(f"Economic life: {len(reserves_result['t_valid'])} months")

Advanced Reserve Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Compare different economic limits
   econ_limits = [5, 10, 15, 20, 25]  # bbl/month
   results = []
   
   for limit in econ_limits:
       res = dca.reserves(params, econ_limit=limit)
       results.append({
           'econ_limit': limit,
           'eur': res['eur'],
           'economic_life': len(res['t_valid'])
       })
   
   import pandas as pd
   comparison = pd.DataFrame(results)
   print(comparison)
   
   # Plot production profile with economic limit
   res = dca.reserves(params, econ_limit=10)
   
   plt.figure(figsize=(10, 6))
   plt.plot(res['t'], res['q'], 'b-', label='Production forecast')
   plt.axhline(y=10, color='r', linestyle='--', label='Economic limit')
   plt.fill_between(res['t_valid'], res['q_valid'], alpha=0.3, 
                    label=f'Economic reserves: {res["eur"]:,.0f} bbls')
   plt.xlabel('Time (months)')
   plt.ylabel('Production Rate (bbl/month)')
   plt.title('Production Forecast with Economic Limit')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Sensitivity to Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze sensitivity to decline parameters
   base_params = dca.ArpsParams(qi=1000, di=0.10, b=0.5)
   
   # Vary initial production
   qi_values = [800, 900, 1000, 1100, 1200]
   qi_results = []
   
   for qi in qi_values:
       params = dca.ArpsParams(qi=qi, di=0.10, b=0.5)
       res = dca.reserves(params)
       qi_results.append({'qi': qi, 'eur': res['eur']})
   
   # Vary decline rate
   di_values = [0.05, 0.08, 0.10, 0.12, 0.15]
   di_results = []
   
   for di in di_values:
       params = dca.ArpsParams(qi=1000, di=di, b=0.5)
       res = dca.reserves(params)
       di_results.append({'di': di, 'eur': res['eur']})
   
   print("Initial Rate Sensitivity:")
   print(pd.DataFrame(qi_results))
   print("\nDecline Rate Sensitivity:")
   print(pd.DataFrame(di_results))

Mathematical Background
----------------------

Reserve Calculation
~~~~~~~~~~~~~~~~~~

**Estimated Ultimate Recovery (EUR):**

For production above economic limit:

.. math::
   EUR = \int_0^{t_{econ}} q(t) \, dt

Where :math:`t_{econ}` is when :math:`q(t) = q_{limit}`

**Arps Decline Equations:**

* **Exponential** (:math:`b = 0`):
  
  .. math::
     q(t) = q_i e^{-D_i t}
     
  .. math::
     EUR = \frac{q_i}{D_i}

* **Harmonic** (:math:`b = 1`):
  
  .. math::
     q(t) = \frac{q_i}{1 + D_i t}
     
  .. math::
     EUR = \frac{q_i}{D_i} \ln\left(\frac{q_i}{q_{limit}}\right)

* **Hyperbolic** (:math:`0 < b < 1`):
  
  .. math::
     q(t) = \frac{q_i}{(1 + b D_i t)^{1/b}}
     
  .. math::
     EUR = \frac{q_i^b}{D_i(1-b)} \left[q_i^{1-b} - q_{limit}^{1-b}\right]

Economic Limit Determination
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The economic limit is typically set based on:

* **Operating costs** vs. revenue
* **Facility constraints** (minimum throughput)
* **Regulatory requirements**
* **Technical limitations**

Common approaches:

.. math::
   q_{limit} = \frac{OPEX}{P - VAR}

Where:
- :math:`OPEX` = Fixed operating expenses
- :math:`P` = Product price
- :math:`VAR` = Variable costs per unit

Reserve Categories
-----------------

**Proved Reserves (1P):**
- High confidence (≥90% probability)
- Conservative decline parameters
- Current economic conditions

**Probable Reserves (2P):**
- Best estimate (≥50% probability)  
- Most likely decline parameters
- Reasonable price assumptions

**Possible Reserves (3P):**
- Low estimate (≥10% probability)
- Optimistic decline parameters
- Favorable economic conditions

Quality Control
--------------

Reserve estimates should be validated by:

1. **Comparison with analogs** in similar fields
2. **Material balance** calculations
3. **Multiple decline curve** methods
4. **Economic sensitivity** analysis
5. **Independent review** by qualified evaluator

Common Issues
~~~~~~~~~~~~

* **Insufficient data** for reliable curve fitting
* **Changing operating conditions** affecting decline
* **Price volatility** impacting economic limits
* **Regulatory changes** affecting development
* **Technology improvements** extending well life

Best Practices
--------------

Data Requirements
~~~~~~~~~~~~~~~~

* Minimum 6-12 months of production data
* Consistent measurement and reporting
* Account for downtime and curtailment
* Consider seasonal variations

Parameter Selection
~~~~~~~~~~~~~~~~~~

* Use field-specific decline trends
* Consider completion and stimulation effects
* Account for depletion drive mechanisms
* Validate with offset well performance

Economic Assumptions
~~~~~~~~~~~~~~~~~~~

* Use conservative price forecasts
* Include all operating costs
* Consider facility sharing and synergies
* Account for abandonment costs

Uncertainty Assessment
~~~~~~~~~~~~~~~~~~~~~

* Perform sensitivity analysis
* Use probabilistic methods (P10/P50/P90)
* Consider correlation between parameters
* Document assumptions and limitations

See Also
--------

* :doc:`models` - Arps decline curve models
* :doc:`economics` - Economic evaluation
* :doc:`sensitivity` - Parameter sensitivity analysis
