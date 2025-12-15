"""Calendar logic for monthly production data.

This module handles the conversion of monthly production volumes to daily
rates, with support for different placement strategies (mid-month, end-month)
and day count weighting. This is critical for public data workflows where
monthly totals are reported.

References:
- Common industry practice for monthly data placement
- SPEE REP 6 guidance on time units
"""

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CalendarConfig:
    """Configuration for calendar/date handling.

    Attributes:
        placement: Where to place monthly volumes ('mid_month' or 'end_month')
        use_day_count_weighting: Whether to weight by actual days in month
        store_placement_choice: Whether to store placement in provenance
    """

    placement: Literal["mid_month", "end_month"] = "mid_month"
    use_day_count_weighting: bool = True
    store_placement_choice: bool = True


def place_monthly_data(
    df: pd.DataFrame,
    date_column: str = "date",
    volume_columns: Optional[list[str]] = None,
    placement: Literal["mid_month", "end_month"] = "mid_month",
    use_day_count_weighting: bool = True,
) -> pd.DataFrame:
    """Place monthly production data at specific dates and convert to daily rates.

    Monthly production data typically represents total production for the month.
    This function:
    1. Places the data point at mid-month or end-month
    2. Converts monthly volumes to daily rates
    3. Optionally weights by actual days in month

    Args:
        df: DataFrame with monthly production data
        date_column: Name of date column (should be month-start dates)
        volume_columns: List of volume column names (auto-detect if None)
        placement: 'mid_month' or 'end_month'
        use_day_count_weighting: If True, divide by actual days in month

    Returns:
        DataFrame with placed dates and daily rates

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=3, freq='MS'),
        ...     'oil_bbl': [3000, 2700, 2400]  # Monthly totals
        ... })
        >>> df_placed = place_monthly_data(df, volume_columns=['oil_bbl'])
        >>> # Dates moved to mid-month, volumes converted to daily rates
    """
    df_result = df.copy()

    # Auto-detect volume columns if not provided
    if volume_columns is None:
        volume_candidates = ["oil", "gas", "water", "oil_bbl", "gas_mcf", "water_bbl"]
        volume_columns = [col for col in volume_candidates if col in df_result.columns]

    if date_column not in df_result.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column])

    # Place dates
    if placement == "mid_month":
        # Move to mid-month (15th day, or actual mid-point)
        df_result[date_column] = (
            df_result[date_column] + pd.offsets.MonthBegin(-1) + pd.Timedelta(days=14)
        )
    elif placement == "end_month":
        # Move to end of month
        df_result[date_column] = df_result[date_column] + pd.offsets.MonthEnd(0)
    else:
        raise ValueError(f"Unknown placement: {placement}")

    # Convert volumes to daily rates
    for col in volume_columns:
        if col not in df_result.columns:
            continue

        # Calculate days in each month
        if use_day_count_weighting:
            # Get actual days in month for each date
            days_in_month = df_result[date_column].dt.days_in_month
        else:
            # Use average days per month
            days_in_month = pd.Series([30.4375] * len(df_result), index=df_result.index)

        # Convert monthly volume to daily rate
        rate_col = col.replace("_bbl", "_rate").replace("_mcf", "_rate")
        if rate_col == col:
            rate_col = f"{col}_rate"

        df_result[rate_col] = df_result[col] / days_in_month

        logger.debug(
            f"Converted {col} to {rate_col} using {placement} placement",
            extra={
                "placement": placement,
                "day_count_weighting": use_day_count_weighting,
            },
        )

    return df_result


def calculate_days_in_period(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Calculate actual days between two dates.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Number of days (inclusive)
    """
    return (end_date - start_date).days + 1


def get_month_day_counts(dates: pd.DatetimeIndex) -> pd.Series:
    """Get actual day counts for each month in dates.

    Args:
        dates: DatetimeIndex with dates

    Returns:
        Series with day counts for each date
    """
    return pd.Series(dates.days_in_month, index=dates)


def weight_monthly_volumes_by_days(
    df: pd.DataFrame,
    volume_columns: list[str],
    date_column: str = "date",
) -> pd.DataFrame:
    """Weight monthly volumes by actual days in month.

    This is important because months have different numbers of days,
    and production volumes should be normalized accordingly.

    Args:
        df: DataFrame with monthly data
        volume_columns: List of volume column names
        date_column: Name of date column

    Returns:
        DataFrame with day-weighted volumes
    """
    df_result = df.copy()

    if date_column not in df_result.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column])

    # Get days in month
    days_in_month = df_result[date_column].dt.days_in_month
    avg_days_per_month = 30.4375

    # Weight volumes
    for col in volume_columns:
        if col in df_result.columns:
            # Normalize to average month
            df_result[col] = df_result[col] * (avg_days_per_month / days_in_month)

    return df_result


def convert_monthly_to_daily(
    monthly_volumes: pd.Series,
    dates: pd.DatetimeIndex,
    placement: Literal["mid_month", "end_month"] = "mid_month",
    use_day_count_weighting: bool = True,
) -> pd.Series:
    """Convert monthly volumes to daily rates.

    Convenience function for converting a single series.

    Args:
        monthly_volumes: Series of monthly production volumes
        dates: DatetimeIndex with month dates
        placement: Where to place the data point
        use_day_count_weighting: Whether to use actual days in month

    Returns:
        Series of daily rates
    """
    if len(monthly_volumes) != len(dates):
        raise ValueError("Volumes and dates must have same length")

    # Calculate days in each month
    if use_day_count_weighting:
        days_in_month = dates.days_in_month
    else:
        days_in_month = pd.Series([30.4375] * len(dates), index=dates)

    # Convert to daily rates
    daily_rates = monthly_volumes / days_in_month

    return daily_rates


def create_daily_index_from_monthly(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    placement: Literal["mid_month", "end_month"] = "mid_month",
) -> pd.DatetimeIndex:
    """Create daily date index from monthly data range.

    Useful for interpolating monthly data to daily for forecasting.

    Args:
        start_date: Start date (month start)
        end_date: End date (month start)
        placement: Placement strategy

    Returns:
        DatetimeIndex with daily dates
    """
    if placement == "mid_month":
        # Create monthly mid-points
        monthly_dates = pd.date_range(start_date, end_date, freq="MS")
        mid_dates = monthly_dates + pd.Timedelta(days=14)
        return mid_dates
    else:
        # Create monthly end dates
        monthly_dates = pd.date_range(start_date, end_date, freq="MS")
        end_dates = monthly_dates + pd.offsets.MonthEnd(0)
        return end_dates
