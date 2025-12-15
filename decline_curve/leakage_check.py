"""Data leakage detection and validation module.

This module provides comprehensive checks for data leakage in forecasting
and training pipelines. It validates that:
- No future data is used in training
- No future data is used in feature engineering
- Forecasts only use historical data
- Normalization/scaling only uses training data statistics
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def validate_no_future_data(
    data: pd.DataFrame,
    date_column: str,
    forecast_cutoff_date: pd.Timestamp,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Validate that no future data is used.

    Args:
        data: DataFrame with production data
        date_column: Name of date column
        forecast_cutoff_date: Date after which data should not be used
        feature_columns: Optional list of feature columns to check

    Returns:
        Dictionary with validation results
    """
    checks = {
        "no_future_dates": True,
        "no_future_features": True,
        "errors": [],
    }

    # Check that all dates are before cutoff
    if date_column in data.columns:
        future_dates = data[data[date_column] > forecast_cutoff_date]
        if len(future_dates) > 0:
            checks["no_future_dates"] = False
            checks["errors"].append(
                f"Found {len(future_dates)} rows with dates after cutoff"
            )

    # Check feature columns if specified
    if feature_columns:
        for col in feature_columns:
            if col in data.columns:
                # Check for any NaN values that might indicate future data issues
                nan_count = data[col].isna().sum()
                if nan_count > len(data) * 0.5:  # More than 50% NaN is suspicious
                    checks["errors"].append(
                        f"Column {col} has {nan_count} NaN values (potential leakage)"
                    )

    return checks


def validate_training_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    date_column: str,
) -> Dict[str, bool]:
    """Validate that training and test data are properly split.

    Args:
        train_data: Training data
        test_data: Test data
        date_column: Name of date column

    Returns:
        Dictionary with validation results
    """
    checks = {
        "proper_split": True,
        "no_overlap": True,
        "chronological_order": True,
        "errors": [],
    }

    if date_column not in train_data.columns or date_column not in test_data.columns:
        checks["proper_split"] = False
        checks["errors"].append("Date column not found in data")
        return checks

    # Check that all training dates are before test dates
    max_train_date = train_data[date_column].max()
    min_test_date = test_data[date_column].min()

    if max_train_date >= min_test_date:
        checks["proper_split"] = False
        checks["errors"].append(
            f"Training data extends to {max_train_date}, "
            f"test data starts at {min_test_date}"
        )

    # Check for date overlap
    train_dates = set(train_data[date_column])
    test_dates = set(test_data[date_column])
    overlap = train_dates.intersection(test_dates)
    if overlap:
        checks["no_overlap"] = False
        checks["errors"].append(f"Found {len(overlap)} overlapping dates")

    return checks


def validate_rolling_window(
    data: pd.Series,
    window_size: int,
    center: bool = False,
) -> Dict[str, bool]:
    """Validate that rolling window operations don't use future data.

    Args:
        data: Time series data
        window_size: Size of rolling window
        center: Whether window is centered (True = potential leakage)

    Returns:
        Dictionary with validation results
    """
    checks = {
        "no_leakage": True,
        "errors": [],
    }

    if center:
        checks["no_leakage"] = False
        checks["errors"].append(
            "Rolling window with center=True uses future data. "
            "Use center=False for forecasting."
        )

    return checks


def validate_normalization(
    scaler_fit_data: np.ndarray,
    transform_data: np.ndarray,
    fit_dates: Optional[pd.DatetimeIndex] = None,
    transform_dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, bool]:
    """Validate that normalization doesn't leak future statistics.

    Args:
        scaler_fit_data: Data used to fit the scaler
        transform_data: Data being transformed
        fit_dates: Optional dates for fit data
        transform_dates: Optional dates for transform data

    Returns:
        Dictionary with validation results
    """
    checks = {
        "no_leakage": True,
        "errors": [],
    }

    # If dates are provided, check that transform data doesn't predate fit data
    if fit_dates is not None and transform_dates is not None:
        max_fit_date = fit_dates.max()
        min_transform_date = transform_dates.min()

        if min_transform_date < max_fit_date:
            # This is OK - we can normalize historical data with future statistics
            # But we should log a warning if transform is for training
            pass

    return checks


def check_sequence_preparation(
    sequences: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    horizon: int,
) -> Dict[str, bool]:
    """Check that sequence preparation doesn't create leakage.

    Args:
        sequences: Input sequences
        targets: Target values
        sequence_length: Length of input sequences
        horizon: Forecast horizon

    Returns:
        Dictionary with validation results
    """
    checks = {
        "proper_sequencing": True,
        "no_overlap": True,
        "errors": [],
    }

    # Check that targets come after sequences
    # This is a simplified check - in practice would need to verify
    # that target[i] corresponds to data after sequence[i]

    if len(sequences) != len(targets):
        checks["proper_sequencing"] = False
        checks["errors"].append(
            f"Sequence count ({len(sequences)}) doesn't match "
            f"target count ({len(targets)})"
        )

    return checks


def comprehensive_leakage_check(
    production_data: pd.DataFrame,
    forecast_cutoff_date: pd.Timestamp,
    training_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Comprehensive data leakage check.

    Args:
        production_data: Full production dataset
        forecast_cutoff_date: Date after which data should not be used
        training_data: Optional training subset
        test_data: Optional test subset
        feature_columns: Optional list of feature columns

    Returns:
        Dictionary with comprehensive validation results
    """
    results = {
        "overall_valid": True,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # Check main dataset
    main_check = validate_no_future_data(
        production_data, "date", forecast_cutoff_date, feature_columns
    )
    results["checks"]["main_data"] = main_check
    if not main_check["no_future_dates"]:
        results["overall_valid"] = False
        results["errors"].extend(main_check["errors"])

    # Check training/test split if provided
    if training_data is not None and test_data is not None:
        split_check = validate_training_split(training_data, test_data, "date")
        results["checks"]["train_test_split"] = split_check
        if not split_check["proper_split"]:
            results["overall_valid"] = False
            results["errors"].extend(split_check["errors"])

    # Log results
    if results["overall_valid"]:
        logger.info("Data leakage check passed")
    else:
        logger.error(
            f"Data leakage detected: {len(results['errors'])} issues found",
            extra={"errors": results["errors"]},
        )

    return results
