Data Processing Utilities
==========================

The ``data_processing`` module provides utilities for cleaning, filtering, and preparing production data for decline curve analysis.

.. automodule:: decline_analysis.utils.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

Data Cleaning
-------------

.. autofunction:: decline_analysis.utils.data_processing.remove_nan_and_zeroes

.. autofunction:: decline_analysis.utils.data_processing.filter_wells_by_date_range

Time Calculations
-----------------

.. autofunction:: decline_analysis.utils.data_processing.calculate_days_online

.. autofunction:: decline_analysis.utils.data_processing.get_grouped_min_max

Production Metrics
------------------

.. autofunction:: decline_analysis.utils.data_processing.calculate_cumulative_production

.. autofunction:: decline_analysis.utils.data_processing.normalize_production_to_daily

.. autofunction:: decline_analysis.utils.data_processing.calculate_water_cut

.. autofunction:: decline_analysis.utils.data_processing.calculate_gor

.. autofunction:: decline_analysis.utils.data_processing.get_max_initial_production

Quality Control
---------------

.. autofunction:: decline_analysis.utils.data_processing.detect_production_anomalies

Convenience Functions
---------------------

.. autofunction:: decline_analysis.utils.data_processing.prepare_well_data_for_dca

Examples
--------

Data Cleaning
~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from decline_analysis.utils import data_processing as dp
    
    # Load raw production data
    df = pd.read_csv('production.csv')
    
    # Remove invalid data
    df_clean = dp.remove_nan_and_zeroes(df, 'oil')
    
    print(f"Records before: {len(df)}")
    print(f"Records after: {len(df_clean)}")

Calculating Production Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate water cut
    df['water_cut'] = dp.calculate_water_cut(df, 'Oil', 'Wtr')
    
    # Calculate GOR
    df['gor'] = dp.calculate_gor(df, 'Gas', 'Oil')
    
    # Calculate cumulative production
    df['cum_oil'] = dp.calculate_cumulative_production(df, 'Oil', 'well_id')
    
    print(f"Average water cut: {df['water_cut'].mean():.1f}%")
    print(f"Average GOR: {df['gor'].mean():.1f} mcf/bbl")

Quick Data Preparation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Prepare well data for DCA in one line
    series = dp.prepare_well_data_for_dca(
        df,
        well_id='WELL_001',
        well_column='API_WELLNO',
        date_column='ReportDate',
        production_column='Oil',
        remove_zeros=True
    )
    
    # Ready for forecasting
    from decline_analysis import dca
    forecast = dca.forecast(series, model='arps', horizon=12)

Anomaly Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Detect production anomalies
    anomalies = dp.detect_production_anomalies(
        oil_series,
        threshold_std=3.0
    )
    
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly dates: {oil_series[anomalies].index.tolist()}")

Calculating Initial Production Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get maximum production from first 3 months
    # (handles ramp-up period)
    qi = dp.get_max_initial_production(
        df,
        n_months=3,
        production_column='Oil',
        date_column='ReportDate'
    )
    
    print(f"Initial production rate (qi): {qi:.0f} bbl/month")

Days Online Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate days online for each record
    df['Online_Date'] = dp.get_grouped_min_max(
        df, 'well_id', 'ReportDate', 'min'
    )
    
    df['Days_Online'] = dp.calculate_days_online(
        df, 'ReportDate', 'Online_Date'
    )
    
    print(f"Well has been online for {df['Days_Online'].max()} days")

See Also
--------

* :doc:`../tutorial` - Complete tutorial on data preparation
* :doc:`../examples` - More examples and use cases
* :doc:`multiphase` - Multi-phase forecasting
