Data Loader
===========

The data loader utility provides tools for accessing and processing production data from various sources, including the North Dakota Industrial Commission (NDIC) database.

Overview
--------

Data loading capabilities include:

* Automated web scraping of regulatory databases
* Excel file processing and data extraction
* Data cleaning and standardization
* Batch processing of multiple time periods

Functions
---------

.. automodule:: decline_curve.utils.data_loader
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic NDIC Data Loading
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import decline_curve as dca

   # Define months to download
   months = ['2023-01', '2023-02', '2023-03', '2023-04']

   # Load NDIC production data
   production_data = dca.load_ndic_data(
       months_list=months,
       output_dir='ndic_data'  # Directory to save raw files
   )

   print(f"Loaded {len(production_data)} records")
   print(production_data.head())

Advanced Data Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   # Load multiple years of data
   months = []
   for year in [2022, 2023]:
       for month in range(1, 13):
           months.append(f"{year}-{month:02d}")

   # Load data with error handling
   try:
       data = dca.load_ndic_data(months)
       print(f"Successfully loaded {len(data)} records")
   except Exception as e:
       print(f"Error loading data: {e}")

   # Basic data cleaning
   if not data.empty:
       # Remove invalid dates
       data = data[data['Date'].notna()]

       # Filter for oil wells only
       oil_wells = data[data['Product'] == 'Oil']

       # Group by well and calculate totals
       well_summary = oil_wells.groupby('Well_Name').agg({
           'Production': 'sum',
           'Date': ['min', 'max']
       }).round(2)

       print(well_summary.head())

Data Quality Checks
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Perform data quality assessment
   def assess_data_quality(df):
       """Assess quality of loaded production data."""
       quality_report = {
           'total_records': len(df),
           'missing_dates': df['Date'].isna().sum(),
           'missing_production': df['Production'].isna().sum(),
           'negative_production': (df['Production'] < 0).sum(),
           'date_range': (df['Date'].min(), df['Date'].max()),
           'unique_wells': df['Well_Name'].nunique() if 'Well_Name' in df.columns else 0
       }
       return quality_report

   # Load and assess data
   data = dca.load_ndic_data(['2023-06'])
   quality = assess_data_quality(data)

   for key, value in quality.items():
       print(f"{key}: {value}")

Data Sources
-----------

North Dakota Industrial Commission (NDIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NDIC provides monthly production reports for all oil and gas wells in North Dakota:

* **URL Pattern:** ``https://www.dmr.nd.gov/oilgas/mpr/{YYYY-MM}.xlsx``
* **Format:** Excel files with standardized columns
* **Update Frequency:** Monthly (typically 2-3 months lag)
* **Coverage:** All producing wells in North Dakota

**Typical Data Fields:**

* Well identification and location
* Monthly oil production (bbls)
* Monthly gas production (Mcf)
* Monthly water production (bbls)
* Operator information
* Completion dates

Data Processing Pipeline
-----------------------

The data loader follows this processing pipeline:

1. **Download:** Fetch Excel files from NDIC servers
2. **Parse:** Extract data using xlrd library
3. **Clean:** Convert date formats and handle missing values
4. **Standardize:** Apply consistent column naming
5. **Combine:** Merge multiple months into single DataFrame
6. **Save:** Cache raw files locally for future use

Error Handling
~~~~~~~~~~~~~

The loader includes robust error handling for:

* **Network timeouts** during file downloads
* **Invalid Excel formats** or corrupted files
* **Missing data fields** in source files
* **Date parsing errors** from inconsistent formats
* **Memory limitations** with large datasets

Best Practices
--------------

Data Management
~~~~~~~~~~~~~~

* **Incremental loading:** Only download new months
* **Local caching:** Save raw files to avoid re-downloading
* **Data validation:** Check for completeness and consistency
* **Version control:** Track data source versions and updates

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

* **Batch processing:** Load multiple months in single call
* **Parallel downloads:** Use threading for multiple files
* **Memory management:** Process large datasets in chunks
* **Compression:** Store cached data in compressed formats

Data Quality
~~~~~~~~~~~

* **Outlier detection:** Identify unrealistic production values
* **Completeness checks:** Ensure all required fields present
* **Consistency validation:** Check for logical relationships
* **Temporal continuity:** Verify sequential monthly data

Legal and Ethical Considerations
-------------------------------

Web Scraping Guidelines
~~~~~~~~~~~~~~~~~~~~~~

* **Respect robots.txt:** Follow website scraping policies
* **Rate limiting:** Avoid overwhelming servers with requests
* **Terms of service:** Comply with data provider terms
* **Attribution:** Properly credit data sources

Data Usage Rights
~~~~~~~~~~~~~~~~

* **Public domain:** NDIC data is generally public information
* **Commercial use:** Verify licensing for commercial applications
* **Redistribution:** Understand restrictions on data sharing
* **Privacy:** Protect any personally identifiable information

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Connection Errors:**
- Check internet connectivity
- Verify NDIC website availability
- Consider proxy or firewall restrictions

**File Format Changes:**
- NDIC may update Excel file formats
- Monitor for column name changes
- Implement flexible parsing logic

**Memory Errors:**
- Process data in smaller batches
- Use efficient data types (e.g., categories)
- Clear unused variables from memory

**Date Parsing Issues:**
- Handle various date formats
- Account for timezone differences
- Validate date ranges for reasonableness

Example Troubleshooting Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging

   # Set up logging for debugging
   logging.basicConfig(level=logging.INFO)

   # Test connection and data availability
   def test_ndic_connection():
       """Test NDIC data availability."""
       test_months = ['2023-01']  # Recent month

       try:
           data = dca.load_ndic_data(test_months)
           if len(data) > 0:
               print("✓ NDIC connection successful")
               print(f"✓ Loaded {len(data)} records")
               return True
           else:
               print("⚠ No data returned")
               return False
       except Exception as e:
           print(f"✗ Connection failed: {e}")
           return False

   # Run connection test
   test_ndic_connection()

See Also
--------

* :doc:`api` - Main API functions
* :doc:`examples` - Complete usage examples
* :doc:`contributing` - Guidelines for adding new data sources
