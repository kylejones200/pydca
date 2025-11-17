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

   from decline_analysis.models import fit_arps, predict_arps
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

   from decline_analysis.forecast_arima import forecast_arima

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
   from decline_analysis.forecast_chronos import forecast_chronos_probabilistic
   quantiles = forecast_chronos_probabilistic(series, horizon=12,
                                            quantiles=[0.1, 0.5, 0.9])

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

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Ensure Regular Intervals**: Monthly data is typically best
2. **Handle Missing Values**: Remove or interpolate gaps carefully
3. **Check for Outliers**: Extreme values can skew model fits
4. **Sufficient History**: Minimum 12-24 data points recommended

Model Fitting
~~~~~~~~~~~~~

1. **Start Simple**: Begin with exponential Arps, then increase complexity
2. **Check Residuals**: Look for patterns in forecast errors
3. **Validate Parameters**: Ensure physically reasonable values
4. **Consider Uncertainty**: Use multiple models or confidence intervals

Forecast Interpretation
~~~~~~~~~~~~~~~~~~~~~~

1. **Understand Limitations**: All models are approximations
2. **Consider Business Context**: Some errors are more costly than others
3. **Update Regularly**: Incorporate new data as it becomes available
4. **Communicate Uncertainty**: Provide ranges, not just point estimates

Common Pitfalls
~~~~~~~~~~~~~~~

1. **Overfitting**: Complex models may not generalize well
2. **Extrapolation**: Be cautious with long-term forecasts
3. **Parameter Instability**: Check sensitivity to data changes
4. **Ignoring Physics**: Ensure forecasts make physical sense

For practical examples, see the :doc:`examples` section. For detailed API information, refer to the :doc:`api/models` documentation.
