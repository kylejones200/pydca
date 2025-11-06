Multi-Phase Forecasting
========================

The ``multiphase`` module provides functionality for simultaneous forecasting of oil, gas, and water production.

.. automodule:: decline_analysis.multiphase
   :members:
   :undoc-members:
   :show-inheritance:

MultiPhaseData
--------------

.. autoclass:: decline_analysis.multiphase.MultiPhaseData
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: calculate_ratios
   .. automethod:: to_dataframe

MultiPhaseForecaster
--------------------

.. autoclass:: decline_analysis.multiphase.MultiPhaseForecaster
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: forecast
   .. automethod:: evaluate
   .. automethod:: calculate_consistency_metrics

Helper Functions
----------------

.. autofunction:: decline_analysis.multiphase.create_multiphase_data_from_dataframe

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from decline_analysis.multiphase import (
        MultiPhaseData,
        MultiPhaseForecaster,
        create_multiphase_data_from_dataframe
    )
    
    # Load production data
    df = pd.read_csv('production.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create multi-phase data
    data = create_multiphase_data_from_dataframe(
        df,
        oil_column='Oil',
        gas_column='Gas',
        water_column='Wtr',
        date_column='date'
    )
    
    # Initialize forecaster
    forecaster = MultiPhaseForecaster()
    
    # Generate coupled forecast
    forecasts = forecaster.forecast(
        data,
        horizon=24,
        model='arps',
        kind='hyperbolic',
        enforce_ratios=True
    )
    
    # Evaluate accuracy
    metrics = forecaster.evaluate(data, forecasts)
    print(f"Oil RMSE: {metrics['oil']['rmse']:.0f}")
    print(f"Gas RMSE: {metrics['gas']['rmse']:.0f}")
    print(f"Water RMSE: {metrics['water']['rmse']:.0f}")

Calculating Production Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate GOR and water cut
    ratios = data.calculate_ratios()
    
    print(f"Average GOR: {ratios['gor'].mean():.1f} mcf/bbl")
    print(f"Average Water Cut: {ratios['water_cut'].mean():.1f}%")
    print(f"Final Water Cut: {ratios['water_cut'].iloc[-1]:.1f}%")

Comparing Coupled vs Independent Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Coupled forecasting (maintains physical relationships)
    coupled = forecaster.forecast(
        data, horizon=24, enforce_ratios=True
    )
    
    # Independent forecasting (traditional approach)
    independent = forecaster.forecast(
        data, horizon=24, enforce_ratios=False
    )
    
    # Check consistency
    coupled_consistency = forecaster.calculate_consistency_metrics(coupled)
    independent_consistency = forecaster.calculate_consistency_metrics(independent)
    
    print(f"Coupled GOR stability: {coupled_consistency['gor_stability']:.3f}")
    print(f"Independent GOR stability: {independent_consistency['gor_stability']:.3f}")

See Also
--------

* :doc:`../tutorial` - Complete tutorial on multi-phase forecasting
* :doc:`../examples` - More examples and use cases
* :doc:`dca` - Single-phase forecasting functions
