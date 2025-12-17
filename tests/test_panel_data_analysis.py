"""Panel data analysis with company fixed effects and spatial controls.

This module tests panel data analysis methods for production forecasting
with company ownership controls and spatial location data.
"""

import numpy as np
import pandas as pd
import pytest

from decline_curve.eur_estimation import calculate_eur_batch
from decline_curve.logging_config import get_logger

logger = get_logger(__name__)


def prepare_panel_data(
    df: pd.DataFrame,
    well_id_col: str = "API_WELLNO",
    date_col: str = "ReportDate",
    value_col: str = "Oil",
    company_col: str = "Company",
    lat_col: str = "Lat",
    long_col: str = "Long",
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

    Returns:
        Panel DataFrame with well_id, date, production, company, location
    """
    panel = df[[well_id_col, date_col, value_col]].copy()

    # Add company information (first company for each well)
    if company_col in df.columns:
        company_map = df.groupby(well_id_col)[company_col].first().to_dict()
        panel["company"] = panel[well_id_col].map(company_map)

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
    df: pd.DataFrame, lat_col: str = "lat", long_col: str = "long"
) -> pd.DataFrame:
    """Calculate spatial features from location data.

    Args:
        df: DataFrame with lat/long columns
        lat_col: Column name for latitude
        long_col: Column name for longitude

    Returns:
        DataFrame with additional spatial features
    """
    spatial = df.copy()

    if lat_col in df.columns and long_col in df.columns:
        # Calculate distance from center of basin (approximate Bakken center)
        # Bakken center approximately: 47.5°N, 103.0°W
        basin_center_lat = 47.5
        basin_center_long = -103.0

        spatial["distance_from_center"] = np.sqrt(
            (spatial[lat_col] - basin_center_lat) ** 2
            + (spatial[long_col] - basin_center_long) ** 2
        )

        # Calculate well density (neighbors within 5 km)
        # Approximate: 1 degree ≈ 111 km
        spatial["well_density"] = spatial.apply(
            lambda row: (
                (
                    (
                        (spatial[lat_col] - row[lat_col]) ** 2
                        + (spatial[long_col] - row[long_col]) ** 2
                    )
                    < (5 / 111) ** 2
                ).sum()
                if pd.notna(row[lat_col]) and pd.notna(row[long_col])
                else np.nan
            ),
            axis=1,
        )

    return spatial


def panel_data_with_company_effects(
    panel_df: pd.DataFrame,
    well_id_col: str = "API_WELLNO",
    date_col: str = "ReportDate",
    value_col: str = "Oil",
    company_col: str = "company",
) -> pd.DataFrame:
    """Add company fixed effects to panel data.

    Args:
        panel_df: Panel DataFrame
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        company_col: Column name for company

    Returns:
        DataFrame with company fixed effects indicators
    """
    result = panel_df.copy()

    if company_col in panel_df.columns:
        # Create company dummy variables
        companies = panel_df[company_col].dropna().unique()
        for company in companies:
            result[f"company_{company}"] = (panel_df[company_col] == company).astype(
                int
            )

        # Add company size (number of wells per company)
        company_sizes = panel_df.groupby(company_col)[well_id_col].nunique().to_dict()
        result["company_size"] = panel_df[company_col].map(company_sizes)

    return result


def test_prepare_panel_data():
    """Test panel data preparation."""
    # Create sample data
    data = []
    for well_id in ["WELL_001", "WELL_002"]:
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        for i, date in enumerate(dates):
            data.append(
                {
                    "API_WELLNO": well_id,
                    "ReportDate": date,
                    "Oil": 1000 * (0.95**i),
                    "Company": "Company_A" if well_id == "WELL_001" else "Company_B",
                    "Lat": 47.5 + (0.1 if well_id == "WELL_001" else 0.2),
                    "Long": -103.0 + (0.1 if well_id == "WELL_001" else 0.2),
                }
            )

    df = pd.DataFrame(data)
    panel = prepare_panel_data(df)

    assert "company" in panel.columns
    assert "lat" in panel.columns
    assert "long" in panel.columns
    assert "months_since_start" in panel.columns
    assert len(panel) == 24
    assert panel["company"].nunique() == 2


def test_calculate_spatial_features():
    """Test spatial feature calculation."""
    data = []
    for i, well_id in enumerate(["WELL_001", "WELL_002", "WELL_003"]):
        data.append(
            {
                "API_WELLNO": well_id,
                "lat": 47.5 + i * 0.1,
                "long": -103.0 + i * 0.1,
            }
        )

    df = pd.DataFrame(data)
    spatial = calculate_spatial_features(df)

    assert "distance_from_center" in spatial.columns
    assert "well_density" in spatial.columns
    assert all(spatial["distance_from_center"] >= 0)


def test_panel_data_with_company_effects():
    """Test company fixed effects."""
    data = []
    for well_id in ["WELL_001", "WELL_002"]:
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        for date in dates:
            data.append(
                {
                    "API_WELLNO": well_id,
                    "ReportDate": date,
                    "Oil": 1000,
                    "company": "Company_A" if well_id == "WELL_001" else "Company_B",
                }
            )

    df = pd.DataFrame(data)
    result = panel_data_with_company_effects(df)

    assert "company_Company_A" in result.columns
    assert "company_Company_B" in result.columns
    assert "company_size" in result.columns
    assert all(result["company_size"] > 0)


def test_real_data_panel_preparation():
    """Test panel data preparation with real data."""
    try:
        df = pd.read_parquet("data/north_dakota_production.parquet")
        # Sample for testing
        sample_wells = df["API_WELLNO"].unique()[:100]
        df_sample = df[df["API_WELLNO"].isin(sample_wells)].copy()

        panel = prepare_panel_data(df_sample)

        assert len(panel) > 0
        assert "months_since_start" in panel.columns

        if "Company" in df_sample.columns:
            assert "company" in panel.columns
            logger.info(
                f"Panel data prepared with {panel['company'].nunique()} companies"
            )

        if "Lat" in df_sample.columns and "Long" in df_sample.columns:
            assert "lat" in panel.columns
            assert "long" in panel.columns
            n_with_location = panel[["lat", "long"]].notna().all(axis=1).sum()
            logger.info(f"Location data available for {n_with_location} records")

    except FileNotFoundError:
        pytest.skip("Real data file not found")
    except ImportError:
        pytest.skip("Parquet support (pyarrow or fastparquet) not available")


def test_eur_with_company_controls():
    """Test EUR calculation with company information."""
    data = []
    for well_id in ["WELL_001", "WELL_002"]:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        months = np.arange(len(dates))
        production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))

        for date, prod in zip(dates, production):
            data.append(
                {
                    "API_WELLNO": well_id,
                    "ReportDate": date,
                    "Oil": max(0, prod),
                    "Company": "Company_A" if well_id == "WELL_001" else "Company_B",
                }
            )

    df = pd.DataFrame(data)
    panel = prepare_panel_data(df)

    # Calculate EUR
    eur_results = calculate_eur_batch(
        panel,
        well_id_col="API_WELLNO",
        date_col="ReportDate",
        value_col="Oil",
        model_type="hyperbolic",
        min_months=12,
    )

    assert len(eur_results) == 2
    assert "eur" in eur_results.columns

    # Merge with company info
    company_map = panel.groupby("API_WELLNO")["company"].first().to_dict()
    eur_results["company"] = eur_results["API_WELLNO"].map(company_map)

    # Check company grouping
    assert eur_results["company"].nunique() == 2
    n_wells = len(eur_results)
    n_companies = eur_results["company"].nunique()
    logger.info(f"EUR calculated for {n_wells} wells across {n_companies} companies")
