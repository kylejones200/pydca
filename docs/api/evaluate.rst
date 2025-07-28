Evaluation (evaluate)
======================

The evaluation module provides metrics for assessing forecast accuracy.

.. automodule:: decline_analysis.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

rmse
~~~~

.. autofunction:: decline_analysis.evaluate.rmse

mae
~~~

.. autofunction:: decline_analysis.evaluate.mae

smape
~~~~~

.. autofunction:: decline_analysis.evaluate.smape

mape
~~~~

.. autofunction:: decline_analysis.evaluate.mape

r2_score
~~~~~~~~

.. autofunction:: decline_analysis.evaluate.r2_score

evaluate_forecast
~~~~~~~~~~~~~~~~~

.. autofunction:: decline_analysis.evaluate.evaluate_forecast

Evaluation Metrics
------------------

Root Mean Square Error (RMSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

Mean Absolute Error (MAE)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|

Symmetric Mean Absolute Percentage Error (SMAPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   SMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}

Mean Absolute Percentage Error (MAPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|

R-squared (R²)
~~~~~~~~~~~~~~

.. math::

   R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

Where:
- :math:`y_i` are the actual values
- :math:`\hat{y}_i` are the predicted values
- :math:`\bar{y}` is the mean of actual values
- :math:`n` is the number of observations
