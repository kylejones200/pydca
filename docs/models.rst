Models and Methods
==================

This section provides detailed information about the forecasting models and methods available in the Decline Curve Analysis library.

Arps Decline Curves
--------------------

The Arps decline curves are the foundation of petroleum production forecasting, developed by J.J. Arps in 1945. These empirical models describe the relationship between production rate and time.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

All Arps models follow the general form:

.. math::

   q(t) = \frac{q_i}{(1 + b \cdot d_i \cdot t)^{1/b}}

Where:
- :math:`q(t)` = production rate at time t
- :math:`q_i` = initial production rate
- :math:`d_i` = initial decline rate
- :math:`b` = hyperbolic exponent
- :math:`t` = time

Exponential Decline (b = 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   q(t) = q_i \cdot e^{-d_i \cdot t}

**Characteristics:**
- Constant percentage decline rate
- Most conservative forecast
- Suitable for mature wells with steady decline
- Simple to fit and interpret

**When to Use:**
- Wells with long production history showing steady decline
- Conservative estimates required
- Regulatory or financial planning scenarios

Harmonic Decline (b = 1)
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   q(t) = \frac{q_i}{1 + d_i \cdot t}

**Characteristics:**
- Decreasing decline rate over time
- More optimistic than exponential
- Suitable for wells with improving reservoir conditions
- Asymptotic approach to zero

**When to Use:**
- Wells showing decreasing decline rates
- Pressure maintenance or enhanced recovery operations
- Long-term production planning

Hyperbolic Decline (0 < b < 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   q(t) = \frac{q_i}{(1 + b \cdot d_i \cdot t)^{1/b}}

**Characteristics:**
- Most flexible model
- Transitions from high initial decline to lower long-term decline
- Best fit for most unconventional wells
- Requires careful parameter estimation

**When to Use:**
- Unconventional wells (shale, tight gas)
- Wells with complex decline behavior
- When historical data shows hyperbolic characteristics

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

The library uses non-linear least squares optimization to fit Arps parameters:

.. code-block:: python

   from decline_curve.models import fit_arps, predict_arps
   import numpy as np

   # Prepare data
   t = np.arange(24)  # 24 months
   q = production_data.values

   # Fit hyperbolic model
   params = fit_arps(t, q, kind="hyperbolic")

   print(f"qi (initial rate): {params.qi:.2f}")
   print(f"di (decline rate): {params.di:.4f}")
   print(f"b (hyperbolic exponent): {params.b:.3f}")

   # Generate predictions
   future_t = np.arange(36)  # Extend to 36 months
   predictions = predict_arps(future_t, params)

ARIMA Models
------------

AutoRegressive Integrated Moving Average (ARIMA) models are powerful statistical tools for time series forecasting that can capture complex temporal patterns.

Model Structure
~~~~~~~~~~~~~~~

ARIMA(p,d,q) models combine three components:

- **AR(p)**: AutoRegressive terms - dependence on past values
- **I(d)**: Integration - differencing to achieve stationarity
- **MA(q)**: Moving Average terms - dependence on past forecast errors

.. math::

   (1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p)(1-L)^d X_t = (1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q)\epsilon_t

Where:
- :math:`L` is the lag operator
- :math:`\phi_i` are autoregressive parameters
- :math:`\theta_i` are moving average parameters
- :math:`\epsilon_t` is white noise

Parameter Selection
~~~~~~~~~~~~~~~~~~~

The library uses statsmodels for ARIMA forecasting. You can specify the order manually or use default parameters:

.. code-block:: python

   from decline_curve.forecast_arima import forecast_arima

   # Use default ARIMA(1,1,1) order
   forecast = forecast_arima(series, horizon=12)

   # Manual parameter specification
   forecast = forecast_arima(series, horizon=12, order=(2, 1, 1))

   # Seasonal ARIMA
   forecast = forecast_arima(series, horizon=12, seasonal=True, seasonal_period=12)

**Advantages:**
- Captures complex temporal patterns
- Handles trend and seasonality
- Statistical foundation with confidence intervals
- Automatic parameter selection available

**Limitations:**
- Requires stationary data (handled automatically)
- May need substantial historical data
- Can be sensitive to outliers

Foundation Models
-----------------

The library integrates with state-of-the-art foundation models for time series forecasting.

TimesFM (Google)
~~~~~~~~~~~~~~~~

TimesFM is Google's foundation model for time series forecasting, pre-trained on diverse time series data.

**Features:**
- Zero-shot forecasting capability
- Handles various time series patterns
- Robust to missing data and irregularities
- No parameter tuning required

**Implementation:**

.. code-block:: python

   # TimesFM forecasting (with automatic fallback)
   forecast = dca.forecast(series, model="timesfm", horizon=12)

**Note:** The library includes intelligent fallbacks when TimesFM is not available, using exponential smoothing methods.

Chronos (Amazon)
~~~~~~~~~~~~~~~~

Chronos is Amazon's probabilistic foundation model for time series forecasting.

**Features:**
- Probabilistic forecasts with uncertainty quantification
- Pre-trained on large-scale time series data
- Handles multiple forecast horizons
- Robust performance across domains

**Implementation:**

.. code-block:: python

   # Chronos forecasting
   forecast = dca.forecast(series, model="chronos", horizon=12)

   # Probabilistic forecasting
   from decline_curve.forecast_chronos import forecast_chronos_probabilistic
   quantiles = forecast_chronos_probabilistic(series, horizon=12,
                                            quantiles=[0.1, 0.5, 0.9])

Shale / Unconventional Decline Models
--------------------------------------

The three models below were developed specifically for shale and tight-reservoir wells
where Arps b > 1 is physically unsound at long time scales. Each captures the
fracture-dominated transient flow that drives early unconventional production.

Model Comparison Table
~~~~~~~~~~~~~~~~~~~~~~

+--------------------+-------------------------------+-------------------------------+-------------------+
| Model              | Best For                      | EUR Closed Form?              | Key Parameters    |
+====================+===============================+===============================+===================+
| Arps exponential   | Conventional, mature wells    | Yes                           | qi, di            |
+--------------------+-------------------------------+-------------------------------+-------------------+
| Arps harmonic      | Pressure-maintenance wells    | Yes                           | qi, di            |
+--------------------+-------------------------------+-------------------------------+-------------------+
| Arps hyperbolic    | Most wells, b ∈ (0,1)         | Yes                           | qi, di, b         |
+--------------------+-------------------------------+-------------------------------+-------------------+
| **Duong**          | Shale / transient flow        | No (numerical)                | q1, a, m          |
+--------------------+-------------------------------+-------------------------------+-------------------+
| **PLE (Ilk)**      | Tight gas / loss-ratio        | No (numerical)                | qi, D_inf, D1, n  |
+--------------------+-------------------------------+-------------------------------+-------------------+
| **SEPD**           | Unconventionals, anomalous    | Yes (via Γ function)          | qi, tau, n        |
+--------------------+-------------------------------+-------------------------------+-------------------+

Duong Model
~~~~~~~~~~~

The Duong model (Duong 2011) captures transient linear/bilinear flow in hydraulically
fractured horizontal wells. Unlike Arps, it is physically grounded in fracture-matrix
interaction theory.

.. math::

   q(t) = q_1 \cdot t^{-m} \cdot \exp\!\left(\frac{a}{1-m}\left(t^{1-m} - 1\right)\right)

Where :math:`q_1` is the production rate at :math:`t = 1`, :math:`a` controls the
time-rate of fracture interference, and :math:`m` governs long-term decline steepness.

**Special case:** at :math:`m = 1`, L'Hôpital's rule gives :math:`q(t) = q_1 \cdot t^{-(1+a)}`.

**Parameters:**

- :math:`q_1 > 0` — production rate at month 1 (BOE/month)
- :math:`a > 0` — fracture interference parameter (typical 0.5–2.5)
- :math:`m > 0` — decline exponent (typical 0.9–1.5)

.. code-block:: python

   import numpy as np
   from decline_curve import fit_duong, predict_duong, eur_duong

   t = np.arange(60)  # months
   q = 2000 * (t + 1) ** -0.9 * np.exp(1.2 / (1 - 0.9) * ((t + 1) ** 0.1 - 1))

   params = fit_duong(t, q)
   forecast = predict_duong(np.arange(120), params)
   eur = eur_duong(params, t_max=360, econ_limit=5.0)
   print(f"EUR = {eur:,.0f} BOE")

Power-Law Exponential (PLE / Ilk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Power-Law Exponential model (Ilk et al. 2008) is based on the loss-ratio approach.
It explicitly models the transition from transient to boundary-dominated flow by allowing
the instantaneous decline rate to vary as a power law.

.. math::

   q(t) = q_i \cdot \exp\!\left(-D_\infty t - \frac{D_1 - D_\infty}{n} \cdot t^n\right)

The instantaneous decline rate is:

.. math::

   D(t) = D_\infty + (D_1 - D_\infty) \cdot t^{-n}

**Parameters:**

- :math:`q_i > 0` — initial production rate
- :math:`D_\infty \geq 0` — terminal (boundary-dominated) decline rate (1/month)
- :math:`D_1 > D_\infty` — early-time decline rate at :math:`t = 1`
- :math:`n \in (0, 1)` — decline exponent (typically 0.3–0.7 for tight gas)

.. code-block:: python

   from decline_curve import fit_ple, predict_ple, eur_ple

   params = fit_ple(t, q)
   print(f"qi={params.qi:.0f}  D_inf={params.D_inf:.4f}  n={params.n:.3f}")
   eur = eur_ple(params, t_max=360, econ_limit=5.0)

Stretched Exponential (SEPD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stretched Exponential Production Decline model captures anomalous diffusion in
heterogeneous porous media. Its closed-form EUR makes it attractive for fast probabilistic
analysis.

.. math::

   q(t) = q_i \cdot \exp\!\left[-\left(\frac{t}{\tau}\right)^n\right]

**Closed-form EUR** (no numerical integration required):

.. math::

   \text{EUR} = \frac{q_i \cdot \tau}{n} \cdot \Gamma\!\left(\frac{1}{n}\right)

**Parameters:**

- :math:`q_i > 0` — initial production rate
- :math:`\tau > 0` — characteristic decline time (months)
- :math:`n \in (0, 1]` — stretching exponent

.. code-block:: python

   from decline_curve import fit_sepd, predict_sepd, eur_sepd

   params = fit_sepd(t, q)
   eur = eur_sepd(params)          # closed form — no t_max needed
   print(f"EUR = {eur:,.0f} BOE")

Using Shale Variants Through the Main API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three models are available through the standard ``forecast()`` dispatcher:

.. code-block:: python

   import decline_curve as dca

   forecast_duong = dca.forecast(series, model="arps", kind="duong", horizon=24)
   forecast_ple   = dca.forecast(series, model="arps", kind="ple",   horizon=24)
   forecast_sepd  = dca.forecast(series, model="arps", kind="sepd",  horizon=24)

Model Selection Guidelines
--------------------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with Arps Hyperbolic**
   - Best for most oil and gas wells
   - Industry standard with physical interpretation
   - Good baseline for comparison

2. **Use ARIMA for Complex Patterns**
   - When Arps models show poor fit
   - Seasonal or cyclical patterns present
   - Need statistical confidence intervals

3. **Try Foundation Models for Difficult Cases**
   - Limited historical data
   - Irregular production patterns
   - When traditional methods fail

Model Comparison Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_models(series, horizon=12):
       """Compare all available models."""

       models = {
           'Exponential': ('arps', {'kind': 'exponential'}),
           'Harmonic': ('arps', {'kind': 'harmonic'}),
           'Hyperbolic': ('arps', {'kind': 'hyperbolic'}),
           'ARIMA': ('arima', {}),
           'TimesFM': ('timesfm', {}),
           'Chronos': ('chronos', {}),
       }

       results = {}

       for name, (model, kwargs) in models.items():
           try:
               forecast = dca.forecast(series, model=model, horizon=horizon, **kwargs)
               metrics = dca.evaluate(series, forecast)
               results[name] = {
                   'forecast': forecast,
                   'rmse': metrics['rmse'],
                   'mae': metrics['mae'],
                   'smape': metrics['smape']
               }
           except Exception as e:
               print(f"{name} failed: {e}")

       # Rank by RMSE
       ranked = sorted(results.items(), key=lambda x: x[1]['rmse'])

       print("Model Performance Ranking (by RMSE):")
       for i, (name, result) in enumerate(ranked, 1):
           print(f"{i}. {name}: RMSE={result['rmse']:.2f}, "
                 f"MAE={result['mae']:.2f}, SMAPE={result['smse']:.2f}%")

       return results

Validation and Testing
----------------------

Cross-Validation
~~~~~~~~~~~~~~~~

Proper model validation is crucial for reliable forecasts:

.. code-block:: python

   def time_series_cv(series, model, n_splits=5, horizon=6):
       """Time series cross-validation."""

       results = []
       total_length = len(series)
       test_size = total_length // (n_splits + 1)

       for i in range(n_splits):
           # Split data
           split_point = test_size * (i + 2)
           train = series.iloc[:split_point-horizon]
           test = series.iloc[split_point-horizon:split_point]

           if len(train) < 12 or len(test) == 0:
               continue

           # Generate forecast
           if model == 'arps':
               forecast = dca.forecast(train, model=model, kind='hyperbolic',
                                     horizon=len(test))
           else:
               forecast = dca.forecast(train, model=model, horizon=len(test))

           # Evaluate
           forecast_part = forecast.iloc[len(train):len(train)+len(test)]
           metrics = dca.evaluate(test, forecast_part)
           results.append(metrics)

       # Average results
       avg_metrics = {
           'rmse': np.mean([r['rmse'] for r in results]),
           'mae': np.mean([r['mae'] for r in results]),
           'smape': np.mean([r['smape'] for r in results])
       }

       return avg_metrics

Out-of-Sample Testing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def out_of_sample_test(series, model, test_months=6):
       """Hold out last N months for testing."""

       train = series.iloc[:-test_months]
       test = series.iloc[-test_months:]

       # Generate forecast
       if model == 'arps':
           forecast = dca.forecast(train, model=model, kind='hyperbolic',
                                 horizon=test_months)
       else:
           forecast = dca.forecast(train, model=model, horizon=test_months)

       # Extract forecast portion
       forecast_part = forecast.iloc[len(train):]

       # Evaluate
       metrics = dca.evaluate(test, forecast_part)

       print(f"Out-of-sample performance ({test_months} months):")
       print(f"RMSE: {metrics['rmse']:.2f}")
       print(f"MAE: {metrics['mae']:.2f}")
       print(f"SMAPE: {metrics['smape']:.2f}%")

       return metrics
