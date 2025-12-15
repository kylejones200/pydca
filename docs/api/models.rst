Models (models)
================

The models module provides Arps decline curve fitting and prediction functionality.

.. automodule:: decline_curve.models
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

ArpsParams
~~~~~~~~~~

.. autoclass:: decline_curve.models.ArpsParams
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

fit_arps
~~~~~~~~

.. autofunction:: decline_curve.models.fit_arps

predict_arps
~~~~~~~~~~~~

.. autofunction:: decline_curve.models.predict_arps

q_exp
~~~~~

.. autofunction:: decline_curve.models.q_exp

q_hyp
~~~~~

.. autofunction:: decline_curve.models.q_hyp

Mathematical Background
-----------------------

Arps Decline Curves
~~~~~~~~~~~~~~~~~~~

The Arps decline curves are fundamental models in petroleum engineering for forecasting production decline:

**Exponential Decline** (b = 0):

.. math::

   q(t) = q_i e^{-d_i t}

**Harmonic Decline** (b = 1):

.. math::

   q(t) = \frac{q_i}{1 + d_i t}

**Hyperbolic Decline** (0 < b < 1):

.. math::

   q(t) = \frac{q_i}{(1 + b d_i t)^{1/b}}

Where:
- :math:`q(t)` is the production rate at time t
- :math:`q_i` is the initial production rate
- :math:`d_i` is the initial decline rate
- :math:`b` is the hyperbolic exponent
