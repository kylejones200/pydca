"""Panel data analysis with company fixed effects and spatial controls.

This module provides tools for analyzing production data as panel data,
controlling for company ownership and using location data for spatial analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .eur_estimation import calculate_eur_batch
from .logging_config import get_logger

logger = get_logger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")


def prepare_panel_data(
    df: pd.DataFrame,
    well_id_col: str = "API_WELLNO",
    date_col: str = "ReportDate",
    value_col: str = "Oil",
    company_col: str = "Company",
    lat_col: str = "Lat",
    long_col: str = "Long",
    county_col: str = "County",
    field_col: str = "FieldName",
) -> pd.DataFrame:
    """Prepare production data as panel data with company and location info.

    Args:
        df: Raw production DataFrame
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        company_col: Column name for company
        lat_col: Column name for latitude
        long_col: Column name for longitude
        county_col: Column name for county
        field_col: Column name for field name

    Returns:
        Panel DataFrame with well_id, date, production, company, location
    """
    panel = df[[well_id_col, date_col, value_col]].copy()

    # Add company information (first company for each well)
    if company_col in df.columns:
        company_map = df.groupby(well_id_col)[company_col].first().to_dict()
        panel["company"] = panel[well_id_col].map(company_map)
        logger.info(
            f"Added company information for {panel['company'].notna().sum()} records"
        )

    # Add location information
    if lat_col in df.columns and long_col in df.columns:
        location_map = (
            df.groupby(well_id_col)[[lat_col, long_col]].first().to_dict("index")
        )
        panel["lat"] = panel[well_id_col].map(
            lambda x: location_map.get(x, {}).get(lat_col)
        )
        panel["long"] = panel[well_id_col].map(
            lambda x: location_map.get(x, {}).get(long_col)
        )
        n_with_location = panel[["lat", "long"]].notna().all(axis=1).sum()
        logger.info(f"Added location information for {n_with_location} records")

    # Add county and field information
    if county_col in df.columns:
        county_map = df.groupby(well_id_col)[county_col].first().to_dict()
        panel["county"] = panel[well_id_col].map(county_map)

    if field_col in df.columns:
        field_map = df.groupby(well_id_col)[field_col].first().to_dict()
        panel["field"] = panel[well_id_col].map(field_map)

    # Convert date
    panel[date_col] = pd.to_datetime(panel[date_col])

    # Sort by well and date
    panel = panel.sort_values([well_id_col, date_col])

    # Calculate months since first production
    panel["months_since_start"] = panel.groupby(well_id_col)[date_col].transform(
        lambda x: (x - x.min()).dt.days / 30.44
    )

    return panel


def calculate_spatial_features(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    basin_center_lat: float = 47.5,
    basin_center_long: float = -103.0,
) -> pd.DataFrame:
    """Calculate spatial features from location data.

    Note: This function calculates well density using all wells in the dataset.
    This is appropriate for panel data regression analysis (not forecasting),
    where spatial features are used as controls rather than predictors.

    Args:
        df: DataFrame with lat/long columns
        lat_col: Column name for latitude
        long_col: Column name for longitude
        basin_center_lat: Latitude of basin center
        basin_center_long: Longitude of basin center

    Returns:
        DataFrame with additional spatial features
    """
    spatial = df.copy()

    if lat_col in df.columns and long_col in df.columns:
        # Calculate distance from center of basin
        spatial["distance_from_center"] = np.sqrt(
            (spatial[lat_col] - basin_center_lat) ** 2
            + (spatial[long_col] - basin_center_long) ** 2
        )

        # Calculate well density (neighbors within 5 km) - vectorized
        # Note: Uses all wells in dataset - appropriate for panel regression,
        # not for time-series forecasting where this would cause leakage
        # Approximate: 1 degree ≈ 111 km
        distance_threshold = (5 / 111) ** 2

        # Vectorized calculation using broadcasting
        valid_mask = spatial[[lat_col, long_col]].notna().all(axis=1)
        if valid_mask.sum() > 0:
            valid_lat = spatial.loc[valid_mask, lat_col].values
            valid_long = spatial.loc[valid_mask, long_col].values

            # Calculate pairwise distances using broadcasting
            lat_diff = valid_lat[:, np.newaxis] - valid_lat[np.newaxis, :]
            long_diff = valid_long[:, np.newaxis] - valid_long[np.newaxis, :]
            distances_sq = lat_diff**2 + long_diff**2

            # Count neighbors (excluding self)
            neighbor_counts = (distances_sq < distance_threshold).sum(axis=1) - 1

            # Initialize with NaN
            spatial["well_density"] = np.nan
            spatial.loc[valid_mask, "well_density"] = neighbor_counts
        else:
            spatial["well_density"] = np.nan

        logger.info("Calculated spatial features")

    return spatial


def eur_with_company_controls(
    panel_df: pd.DataFrame,
    well_id_col: str = "API_WELLNO",
    date_col: str = "ReportDate",
    value_col: str = "Oil",
    company_col: str = "company",
    county_col: str = "county",
    model_type: str = "hyperbolic",
    min_months: int = 12,
) -> pd.DataFrame:
    """Calculate EUR with company and county information.

    Args:
        panel_df: Panel DataFrame with production data
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        company_col: Column name for company
        county_col: Column name for county
        model_type: Type of decline model
        min_months: Minimum months of data required

    Returns:
        DataFrame with EUR results, company, and county information
    """
    # Calculate EUR
    eur_results = calculate_eur_batch(
        panel_df,
        well_id_col=well_id_col,
        date_col=date_col,
        value_col=value_col,
        model_type=model_type,
        min_months=min_months,
    )

    if len(eur_results) == 0:
        logger.warning("No EUR calculations succeeded")
        return pd.DataFrame()

    # Merge with company and county info (optimized: single groupby)
    if company_col in panel_df.columns or county_col in panel_df.columns:
        merge_cols = []
        if company_col in panel_df.columns:
            merge_cols.append(company_col)
        if county_col in panel_df.columns:
            merge_cols.append(county_col)

        static_info = panel_df.groupby(well_id_col)[merge_cols].first()
        # Rename columns to match expected output names
        rename_map = {}
        if company_col in merge_cols:
            rename_map[company_col] = "company"
        if county_col in merge_cols:
            rename_map[county_col] = "county"
        static_info = static_info.rename(columns=rename_map)

        eur_results = eur_results.merge(
            static_info, left_on=well_id_col, right_index=True, how="left"
        )

        if "company" in eur_results.columns:
            n_wells = len(eur_results)
            n_companies = eur_results["company"].nunique()
            logger.info(
                f"EUR calculated for {n_wells} wells across {n_companies} companies"
            )
        if "county" in eur_results.columns:
            n_with_county = eur_results["county"].notna().sum()
            n_counties = eur_results["county"].nunique()
            logger.info(
                f"County information added for {n_with_county} wells "
                f"across {n_counties} counties"
            )

    return eur_results


def company_fixed_effects_regression(
    eur_results: pd.DataFrame,
    dependent_var: str = "eur",
    company_col: Optional[str] = "company",
    county_col: Optional[str] = "county",
    control_vars: Optional[list] = None,
) -> Optional[Dict]:
    """Run regression with company and county fixed effects.

    Args:
        eur_results: DataFrame with EUR and company information
        dependent_var: Dependent variable name
        company_col: Column name for company
        county_col: Column name for county (optional)
        control_vars: List of additional control variable column names

    Returns:
        Dictionary with regression results, or None if statsmodels unavailable
    """
    if not HAS_STATSMODELS:
        logger.warning("statsmodels not available for regression analysis")
        return None

    # Prepare data
    df_vars = [dependent_var]
    if company_col and company_col in eur_results.columns:
        df_vars.append(company_col)
    if county_col and county_col in eur_results.columns:
        df_vars.append(county_col)

    if len(df_vars) == 1:
        logger.warning("No control variables specified (company or county)")
        return None

    df = eur_results[df_vars].copy()
    df = df.dropna()

    if len(df) == 0:
        logger.warning("No valid data for regression")
        return None

    # Add control variables if specified
    if control_vars:
        for var in control_vars:
            if var in eur_results.columns:
                df[var] = eur_results[var]
                df = df.dropna(subset=[var])  # Remove rows with missing control vars

    # Create company dummy variables
    companies = []
    if company_col and company_col in df.columns:
        companies = list(df[company_col].dropna().unique())
        for company in companies:
            df[f"company_{company}"] = (df[company_col] == company).astype(int)

    # Create county dummy variables if county column exists
    counties = []
    if county_col and county_col in df.columns:
        counties = list(df[county_col].dropna().unique())
        for county in counties:
            df[f"county_{county}"] = (df[county_col] == county).astype(int)

    # Prepare regression
    y = df[dependent_var]
    X_vars = [f"company_{c}" for c in companies]

    if len(counties) > 0:
        X_vars.extend([f"county_{c}" for c in counties])

    if control_vars:
        X_vars.extend([v for v in control_vars if v in df.columns])

    X = df[X_vars]
    X = sm.add_constant(X)

    # Run regression
    try:
        model = OLS(y, X).fit()
        logger.info("Company and County Fixed Effects Regression Results:")
        logger.info(f"R-squared: {model.rsquared:.4f}")
        logger.info(f"Number of observations: {model.nobs}")
        logger.info(f"Number of companies: {len(companies)}")
        if counties:
            logger.info(f"Number of counties: {len(counties)}")

        return {
            "model": model,
            "rsquared": model.rsquared,
            "nobs": model.nobs,
            "n_companies": len(companies),
            "n_counties": len(counties) if counties else 0,
            "summary": str(model.summary()),
        }
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        return None


def spatial_eur_analysis(
    panel_df: pd.DataFrame,
    eur_results: pd.DataFrame,
    well_id_col: str = "API_WELLNO",
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    """Analyze spatial patterns in EUR.

    Args:
        panel_df: Panel DataFrame with location data
        eur_results: DataFrame with EUR results
        well_id_col: Column name for well identifier
        lat_col: Column name for latitude
        long_col: Column name for longitude

    Returns:
        DataFrame with EUR and spatial features
    """
    # Get location for each well (optimized: use merge instead of map)
    location_df = panel_df.groupby(well_id_col)[[lat_col, long_col]].first()
    eur_results = eur_results.merge(
        location_df, left_on=well_id_col, right_index=True, how="left"
    )

    # Calculate spatial features
    spatial_eur = calculate_spatial_features(eur_results)

    # Correlation analysis
    if "distance_from_center" in spatial_eur.columns:
        corr_distance = spatial_eur["eur"].corr(spatial_eur["distance_from_center"])
        logger.info(
            f"Correlation between EUR and distance from center: {corr_distance:.3f}"
        )

    if "well_density" in spatial_eur.columns:
        corr_density = spatial_eur["eur"].corr(spatial_eur["well_density"])
        logger.info(f"Correlation between EUR and well density: {corr_density:.3f}")

    return spatial_eur


def analyze_by_company(
    eur_results: pd.DataFrame, company_col: str = "company"
) -> pd.DataFrame:
    """Analyze EUR results by company.

    Args:
        eur_results: DataFrame with EUR results
        company_col: Column name for company

    Returns:
        DataFrame with company-level statistics
    """
    if company_col not in eur_results.columns:
        logger.warning(f"Company column {company_col} not found")
        return pd.DataFrame()

    # Company-level statistics
    company_stats = (
        eur_results.groupby(company_col)
        .agg(
            {
                "eur": ["count", "mean", "median", "std"],
                "qi": "mean",
                "di": "mean",
                "b": "mean",
            }
        )
        .round(2)
    )

    logger.info("Company-level EUR statistics:")
    logger.info(f"{company_stats}")

    return company_stats
