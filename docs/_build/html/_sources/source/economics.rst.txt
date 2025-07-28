Economics
==========

The economics module provides financial evaluation tools for oil and gas wells, including NPV calculations, cash flow analysis, and payback period determination.

Overview
--------

Economic evaluation is essential for:

* Investment decision making
* Project ranking and portfolio optimization
* Risk assessment and sensitivity analysis
* Performance monitoring and optimization

Functions
---------

.. automodule:: decline_analysis.economics
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic Economic Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_analysis as dca
   import pandas as pd
   import numpy as np
   
   # Create production forecast
   dates = pd.date_range('2024-01-01', periods=60, freq='MS')
   production = pd.Series([1000, 950, 900, 850, 800] + 
                         list(np.linspace(800, 200, 55)), 
                         index=dates)
   
   # Calculate economics
   econ_results = dca.economics(
       production=production,
       price=60.0,  # $/bbl
       opex=15.0,   # $/bbl operating cost
       discount_rate=0.10  # 10% annual
   )
   
   print(f"NPV: ${econ_results['npv']:,.2f}")
   print(f"Payback: {econ_results['payback_month']} months")

Advanced Economic Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare different price scenarios
   prices = [40, 50, 60, 70, 80]
   results = []
   
   for price in prices:
       econ = dca.economics(production, price=price, opex=15.0)
       results.append({
           'price': price,
           'npv': econ['npv'],
           'payback': econ['payback_month']
       })
   
   econ_df = pd.DataFrame(results)
   print(econ_df)
   
   # Calculate break-even price
   break_even_price = None
   for price in np.arange(10, 100, 1):
       econ = dca.economics(production, price=price, opex=15.0)
       if econ['npv'] > 0:
           break_even_price = price
           break
   
   print(f"Break-even price: ${break_even_price}/bbl")

Cash Flow Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detailed cash flow analysis
   econ = dca.economics(production, price=60, opex=15)
   cash_flow = econ['cash_flow']
   
   # Monthly cash flow
   monthly_cf = pd.DataFrame({
       'date': production.index,
       'production': production.values,
       'revenue': production.values * 60,
       'opex': production.values * 15,
       'net_cf': cash_flow
   })
   
   # Cumulative cash flow
   monthly_cf['cum_cf'] = monthly_cf['net_cf'].cumsum()
   
   print(monthly_cf.head(10))

Mathematical Background
----------------------

Economic Calculations
~~~~~~~~~~~~~~~~~~~~

**Net Present Value (NPV):**

.. math::
   NPV = \sum_{t=0}^{T} \frac{CF_t}{(1+r)^{t/12}}

Where:
- :math:`CF_t` = Net cash flow in month t
- :math:`r` = Annual discount rate
- :math:`T` = Project life in months

**Monthly Cash Flow:**

.. math::
   CF_t = (P - OPEX) \times Q_t

Where:
- :math:`P` = Oil/gas price ($/unit)
- :math:`OPEX` = Operating expense ($/unit)
- :math:`Q_t` = Production in month t

**Payback Period:**

The month when cumulative undiscounted cash flow becomes positive:

.. math::
   \text{Payback} = \min(t) \text{ such that } \sum_{i=0}^{t} CF_i > 0

Discount Rate Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The discount rate should reflect:

* **Risk-free rate** (government bonds)
* **Risk premium** for oil/gas investments
* **Company cost of capital**
* **Project-specific risks**

Typical ranges:
- Low risk: 8-12%
- Medium risk: 12-15%
- High risk: 15-20%

Economic Indicators
------------------

NPV Interpretation
~~~~~~~~~~~~~~~~~

* **NPV > 0:** Project creates value, should proceed
* **NPV = 0:** Project breaks even at discount rate
* **NPV < 0:** Project destroys value, should reject

Payback Period Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

* **< 2 years:** Excellent payback
* **2-4 years:** Good payback
* **4-6 years:** Acceptable payback
* **> 6 years:** Poor payback, high risk

Sensitivity Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

Key variables affecting economics:

1. **Oil/gas prices** (highest impact)
2. **Operating costs** (medium impact)
3. **Production decline** (medium impact)
4. **Discount rate** (low-medium impact)

Best Practices
--------------

Price Assumptions
~~~~~~~~~~~~~~~~

* Use conservative price forecasts
* Consider price volatility and cycles
* Include multiple price scenarios
* Account for price differentials

Cost Estimates
~~~~~~~~~~~~~~

* Include all operating costs (LOE, taxes, etc.)
* Account for cost inflation
* Consider economies of scale
* Include abandonment costs

Risk Assessment
~~~~~~~~~~~~~~

* Perform sensitivity analysis on key variables
* Use Monte Carlo simulation for uncertainty
* Consider correlation between variables
* Evaluate downside scenarios

See Also
--------

* :doc:`sensitivity` - Sensitivity analysis tools
* :doc:`reserves` - Reserve estimation
* :doc:`models` - Production forecasting
