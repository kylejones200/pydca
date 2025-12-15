"""Utilities for loading and preparing real production data for deep learning.

This module provides functions to load the North Dakota production dataset
and prepare it for training deep learning models.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_north_dakota_production(
    data_path: Union[str, Path],
    max_wells: Optional[int] = None,
    min_months: int = 12,
    max_months: int = 120,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    phases: list[str] = ["oil", "gas", "water"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare North Dakota production data for deep learning.

    This function loads the real production dataset with thousands of wells
    and prepares it in the format required for deep learning training.

    Args:
        data_path: Path to the north_dakota_production.csv file
        max_wells: Maximum number of wells to load (None = all)
        min_months: Minimum production history required per well
        max_months: Maximum production history to use per well
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        phases: List of phases to include ('oil', 'gas', 'water')

    Returns:
        Tuple of (production_df, static_features_df):
        - production_df: DataFrame with columns: well_id, date, and phase columns
        - static_features_df: DataFrame with well_id and static features

    Example:
        >>> from decline_curve.utils.real_data_loader import (
        ...     load_north_dakota_production
        ... )
        >>>
        >>> prod_df, static_df = load_north_dakota_production(
        ...     data_path='path/to/north_dakota_production.csv',
        ...     max_wells=100,
        ...     min_months=24
        ... )
        >>>
        >>> print(f"Loaded {prod_df['well_id'].nunique()} wells")
        >>> print(f"Total records: {len(prod_df)}")
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Production data not found at {data_path}")

    from ..logging_config import get_logger

    logger = get_logger(__name__)

    # Load raw data (CSV or Parquet)
    logger.info(f"Loading production data from {data_path}")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Standardize column names
    column_mapping = {
        "API_WELLNO": "well_id",
        "ReportDate": "date",
        "Oil": "oil",
        "Gas": "gas",
        "Wtr": "water",
    }

    # Rename columns that exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Ensure well_id and date columns exist
    if "well_id" not in df.columns:
        raise ValueError("Dataset must contain API_WELLNO column")
    if "date" not in df.columns:
        raise ValueError("Dataset must contain ReportDate column")

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Filter by date range if specified
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    # Filter to wells with sufficient production history
    well_counts = df.groupby("well_id").size()
    valid_wells = well_counts[(well_counts >= min_months) & (well_counts <= max_months)]

    if max_wells:
        valid_wells = valid_wells.head(max_wells)

    df_filtered = df[df["well_id"].isin(valid_wells.index)].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No wells found matching criteria (min_months={min_months}, "
            f"max_months={max_months}, max_wells={max_wells})"
        )

    # Sort by well and date
    df_filtered = df_filtered.sort_values(["well_id", "date"])

    # Prepare phase columns
    phase_columns = []
    for phase in phases:
        phase_col = phase.lower()
        if phase_col in df_filtered.columns:
            # Fill missing values and ensure non-negative
            df_filtered[phase_col] = df_filtered[phase_col].fillna(0).clip(lower=0)
            phase_columns.append(phase_col)
        else:
            logger.warning(f"{phase} column not found in dataset")

    if not phase_columns:
        raise ValueError("No valid phase columns found in dataset")

    # Create production DataFrame (standard format for deep learning)
    production_df = df_filtered[["well_id", "date"] + phase_columns].copy()

    # Create static features DataFrame
    static_features_dict = {"well_id": df_filtered.groupby("well_id").first().index}

    # Extract available static features
    static_candidates = [
        ("Lat", "lat"),
        ("Long", "long"),
        ("County", "county"),
        ("FieldName", "field"),
        ("Pool", "pool"),
        ("Company", "company"),
    ]

    for old_col, new_col in static_candidates:
        if old_col in df_filtered.columns:
            static_features_dict[new_col] = (
                df_filtered.groupby("well_id")[old_col].first().values
            )

    static_features_df = pd.DataFrame(static_features_dict)

    # Normalize numeric static features
    if "lat" in static_features_df.columns:
        static_features_df["lat_norm"] = (
            static_features_df["lat"] - static_features_df["lat"].mean()
        ) / static_features_df["lat"].std()

    if "long" in static_features_df.columns:
        static_features_df["long_norm"] = (
            static_features_df["long"] - static_features_df["long"].mean()
        ) / static_features_df["long"].std()

    logger.info(
        "Loaded production data",
        extra={
            "n_wells": production_df["well_id"].nunique(),
            "n_records": len(production_df),
            "date_min": str(production_df["date"].min()),
            "date_max": str(production_df["date"].max()),
            "phases": phase_columns,
        },
    )

    return production_df, static_features_df


def prepare_data_for_lstm(
    production_df: pd.DataFrame,
    static_features_df: Optional[pd.DataFrame] = None,
    sequence_length: int = 24,
    horizon: int = 12,
    validation_split: float = 0.2,
    test_split: float = 0.1,
) -> dict:
    """
    Prepare production data for LSTM training with train/val/test splits.

    Args:
        production_df: Production DataFrame with well_id, date, and phase columns
        static_features_df: Optional static features DataFrame
        sequence_length: Input sequence length (months)
        horizon: Forecast horizon (months)
        validation_split: Fraction for validation set
        test_split: Fraction for test set

    Returns:
        Dictionary with train/val/test splits for production and static features
    """
    wells = production_df["well_id"].unique()
    n_wells = len(wells)

    # Split wells into train/val/test
    n_test = int(n_wells * test_split)
    n_val = int(n_wells * validation_split)

    test_wells = wells[:n_test]
    val_wells = wells[n_test : n_test + n_val]
    train_wells = wells[n_test + n_val :]

    # Split production data
    train_prod = production_df[production_df["well_id"].isin(train_wells)].copy()
    val_prod = production_df[production_df["well_id"].isin(val_wells)].copy()
    test_prod = production_df[production_df["well_id"].isin(test_wells)].copy()

    # Split static features if provided
    train_static = None
    val_static = None
    test_static = None

    if static_features_df is not None:
        train_static = static_features_df[
            static_features_df["well_id"].isin(train_wells)
        ].copy()
        val_static = static_features_df[
            static_features_df["well_id"].isin(val_wells)
        ].copy()
        test_static = static_features_df[
            static_features_df["well_id"].isin(test_wells)
        ].copy()

    return {
        "train": {"production": train_prod, "static_features": train_static},
        "validation": {"production": val_prod, "static_features": val_static},
        "test": {"production": test_prod, "static_features": test_static},
        "train_wells": train_wells,
        "val_wells": val_wells,
        "test_wells": test_wells,
    }
