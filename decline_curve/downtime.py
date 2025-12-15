"""Downtime and allocation support for production data.

This module handles operational data like uptime, hours on, and allocated
volumes. It supports rate reconstruction from volume and uptime, and flags
cases where uptime data is missing or inconsistent.

Features:
- Rate reconstruction from volume and uptime
- Uptime validation and flagging
- Allocation adjustment support
- Downtime detection
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DowntimeResult:
    """Result of downtime analysis.

    Attributes:
        has_uptime_data: Whether uptime data is present
        uptime_coverage: Fraction of records with uptime data
        missing_uptime_count: Number of records missing uptime
        reconstructed_rates: Whether rates were reconstructed
        warnings: List of warning messages
    """

    has_uptime_data: bool
    uptime_coverage: float
    missing_uptime_count: int
    reconstructed_rates: bool
    warnings: list[str]


def reconstruct_rate_from_uptime(
    df: pd.DataFrame,
    volume_column: str,
    uptime_column: str = "uptime",
    hours_on_column: Optional[str] = None,
    rate_column: Optional[str] = None,
    time_period_days: float = 30.4375,
) -> pd.DataFrame:
    """Reconstruct production rate from volume and uptime.

    Rate = Volume / (Uptime * TimePeriod)

    Args:
        df: DataFrame with production data
        volume_column: Name of volume column
        uptime_column: Name of uptime column (fraction 0-1 or hours)
        hours_on_column: Optional hours_on column (alternative to uptime)
        rate_column: Name for output rate column (auto-generated if None)
        time_period_days: Time period in days (default: average month)

    Returns:
        DataFrame with reconstructed rate column

    Example:
        >>> df = pd.DataFrame({
        ...     'oil_bbl': [3000, 2700, 2400],
        ...     'uptime': [0.95, 0.90, 0.85]  # 95%, 90%, 85% uptime
        ... })
        >>> df = reconstruct_rate_from_uptime(df, 'oil_bbl')
        >>> # Creates 'oil_rate' column with rates adjusted for uptime
    """
    df_result = df.copy()

    if volume_column not in df_result.columns:
        raise ValueError(f"Volume column '{volume_column}' not found")

    # Determine uptime source
    if hours_on_column and hours_on_column in df_result.columns:
        # Use hours_on
        hours_in_period = time_period_days * 24
        uptime = df_result[hours_on_column] / hours_in_period
        uptime_source = hours_on_column
    elif uptime_column in df_result.columns:
        # Use uptime (assume fraction 0-1, or convert from hours if > 1)
        uptime = df_result[uptime_column].copy()
        # If values > 1, assume they're hours and convert
        if uptime.max() > 1.0:
            hours_in_period = time_period_days * 24
            uptime = uptime / hours_in_period
        uptime_source = uptime_column
    else:
        raise ValueError(
            f"Neither '{uptime_column}' nor '{hours_on_column}' found in DataFrame"
        )

    # Generate rate column name
    if rate_column is None:
        rate_column = volume_column.replace("_bbl", "_rate").replace("_mcf", "_rate")
        if rate_column == volume_column:
            rate_column = f"{volume_column}_rate"

    # Reconstruct rate: Rate = Volume / (Uptime * TimePeriod)
    # Only where uptime is available and > 0
    valid_uptime = (uptime > 0) & (uptime <= 1.0) & uptime.notna()

    df_result[rate_column] = np.nan
    df_result.loc[valid_uptime, rate_column] = df_result.loc[
        valid_uptime, volume_column
    ] / (uptime.loc[valid_uptime] * time_period_days)

    # Log reconstruction
    n_reconstructed = valid_uptime.sum()
    logger.info(
        f"Reconstructed {rate_column} from {volume_column} and {uptime_source}",
        extra={"n_reconstructed": n_reconstructed, "n_total": len(df_result)},
    )

    return df_result


def validate_uptime_data(
    df: pd.DataFrame,
    uptime_column: str = "uptime",
    hours_on_column: Optional[str] = None,
    well_id_column: Optional[str] = "well_id",
) -> DowntimeResult:
    """Validate uptime data quality.

    Args:
        df: DataFrame with production data
        uptime_column: Name of uptime column
        hours_on_column: Optional hours_on column
        well_id_column: Name of well ID column

    Returns:
        DowntimeResult with validation findings
    """
    warnings: list[str] = []
    has_uptime = False
    uptime_coverage = 0.0
    missing_count = 0

    # Check for uptime data
    if hours_on_column and hours_on_column in df.columns:
        has_uptime = True
        hours_on = df[hours_on_column]
        valid_count = hours_on.notna().sum()
        missing_count = hours_on.isna().sum()
        uptime_coverage = valid_count / len(df) if len(df) > 0 else 0.0

        # Check for reasonable values
        if valid_count > 0:
            max_hours = (
                df[hours_on_column].max() if df[hours_on_column].notna().any() else 0
            )
            if max_hours > 744:  # More than 31 days * 24 hours
                warnings.append(
                    f"hours_on values exceed 744 hours (31 days) - max: {max_hours:.1f}"
                )
            min_hours = (
                df[hours_on_column].min() if df[hours_on_column].notna().any() else 0
            )
            if min_hours < 0:
                warnings.append("Found negative hours_on values")

    elif uptime_column in df.columns:
        has_uptime = True
        uptime = df[uptime_column]
        valid_count = uptime.notna().sum()
        missing_count = uptime.isna().sum()
        uptime_coverage = valid_count / len(df) if len(df) > 0 else 0.0

        # Check for reasonable values (should be 0-1 for fraction, or hours)
        if valid_count > 0:
            max_val = df[uptime_column].max() if df[uptime_column].notna().any() else 0
            min_val = df[uptime_column].min() if df[uptime_column].notna().any() else 0

            if max_val > 1.0 and max_val < 1000:
                # Likely hours, not fraction
                warnings.append(
                    f"uptime values appear to be hours (max: {max_val:.1f}), "
                    f"not fraction"
                )
            elif max_val > 1.0:
                warnings.append(f"uptime values exceed 1.0 (max: {max_val:.2f})")

            if min_val < 0:
                warnings.append("Found negative uptime values")
    else:
        warnings.append("No uptime data found (uptime or hours_on columns missing)")

    # Check coverage per well if well_id available
    if well_id_column and well_id_column in df.columns and has_uptime:
        for well_id, group in df.groupby(well_id_column):
            well_coverage = (
                group[uptime_column].notna().sum() / len(group)
                if uptime_column in group.columns
                else (
                    group[hours_on_column].notna().sum() / len(group)
                    if hours_on_column and hours_on_column in group.columns
                    else 0
                )
            )
            if well_coverage < 0.5:
                warnings.append(
                    f"Well {well_id} has <50% uptime coverage ({well_coverage:.1%})"
                )

    return DowntimeResult(
        has_uptime_data=has_uptime,
        uptime_coverage=uptime_coverage,
        missing_uptime_count=missing_count,
        reconstructed_rates=False,
        warnings=warnings,
    )


def detect_downtime_periods(
    df: pd.DataFrame,
    uptime_column: str = "uptime",
    hours_on_column: Optional[str] = None,
    threshold: float = 0.5,
    min_duration_days: int = 1,
) -> pd.DataFrame:
    """Detect downtime periods from uptime data.

    Args:
        df: DataFrame with production data
        uptime_column: Name of uptime column
        hours_on_column: Optional hours_on column
        threshold: Uptime threshold below which is considered downtime (fraction)
        min_duration_days: Minimum duration to flag as downtime period

    Returns:
        DataFrame with downtime periods marked
    """
    df_result = df.copy()

    # Determine uptime values
    if hours_on_column and hours_on_column in df_result.columns:
        hours_in_period = 30.4375 * 24  # Average month
        uptime = df_result[hours_on_column] / hours_in_period
    elif uptime_column in df_result.columns:
        uptime = df_result[uptime_column].copy()
        # Convert hours to fraction if needed
        if uptime.max() > 1.0:
            hours_in_period = 30.4375 * 24
            uptime = uptime / hours_in_period
    else:
        logger.warning("No uptime data found for downtime detection")
        df_result["is_downtime"] = False
        return df_result

    # Mark downtime periods
    df_result["is_downtime"] = (uptime < threshold) & uptime.notna()

    # Filter to periods meeting minimum duration
    if "date" in df_result.columns and min_duration_days > 0:
        # Group consecutive downtime periods
        df_result = df_result.sort_values("date")
        downtime_groups = (
            df_result["is_downtime"] != df_result["is_downtime"].shift()
        ).cumsum()

        for group_id in downtime_groups.unique():
            group_mask = downtime_groups == group_id
            group = df_result[group_mask]

            if group["is_downtime"].iloc[0] and len(group) > 0:
                # Check duration
                if "date" in group.columns:
                    duration = (group["date"].max() - group["date"].min()).days
                    if duration < min_duration_days:
                        # Too short, don't flag
                        df_result.loc[group_mask, "is_downtime"] = False

    return df_result


def apply_allocation_adjustment(
    df: pd.DataFrame,
    rate_columns: list[str],
    allocation_column: str = "allocated_volume",
    method: str = "replace",
) -> pd.DataFrame:
    """Apply allocation adjustments to rates.

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names to adjust
        allocation_column: Name of allocation adjustment column
        method: 'replace' (use allocation) or 'multiply' (multiply by allocation)

    Returns:
        DataFrame with adjusted rates
    """
    df_result = df.copy()

    if allocation_column not in df_result.columns:
        logger.warning(f"Allocation column '{allocation_column}' not found")
        return df_result

    for col in rate_columns:
        if col not in df_result.columns:
            continue

        if method == "replace":
            # Replace rate with allocated value where available
            allocated_mask = df_result[allocation_column].notna()
            df_result.loc[allocated_mask, col] = df_result.loc[
                allocated_mask, allocation_column
            ]
        elif method == "multiply":
            # Multiply rate by allocation factor
            allocated_mask = df_result[allocation_column].notna()
            df_result.loc[allocated_mask, col] = (
                df_result.loc[allocated_mask, col]
                * df_result.loc[allocated_mask, allocation_column]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    logger.info(
        f"Applied allocation adjustment to {len(rate_columns)} rate columns",
        extra={"method": method},
    )

    return df_result
