"""
Data processing utilities for decline curve analysis.

This module contains helper functions for cleaning, filtering, and preparing
production data for decline curve analysis.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def remove_nan_and_zeroes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Filter out NaN values and zeroes from a specified column.

    Args:
        df: Input DataFrame
        column: Name of column to filter

    Returns:
        Filtered DataFrame with NaN's and zeroes removed from specified column

    Example:
        >>> df_clean = remove_nan_and_zeroes(df, 'oil_bbl')
    """
    filtered_df = df[(df[column].notnull()) & (df[column] > 0)]
    return filtered_df


def calculate_days_online(
    df: pd.DataFrame, date_column: str, online_date_column: str
) -> pd.Series:
    """
    Calculate cumulative days online for each record.

    Creates a column showing the time delta between when the well came online
    and each production record date.

    Args:
        df: Input DataFrame
        date_column: Name of column with record dates (datetime)
        online_date_column: Name of column with well online date (datetime)

    Returns:
        Series containing days online for each record

    Example:
        >>> df['days_online'] = calculate_days_online(df, 'ReportDate', 'Online_Date')
    """
    return (df[date_column] - df[online_date_column]).dt.days


def get_grouped_min_max(
    df: pd.DataFrame, group_column: str, calc_column: str, calc_type: str = "min"
) -> pd.Series:
    """
    Get min or max value for a column with grouping applied.

    Useful for finding earliest/latest dates or min/max production by well.

    Args:
        df: Input DataFrame
        group_column: Column name to group by (e.g., 'well_id')
        calc_column: Column name to calculate min/max for
        calc_type: Either 'min' or 'max'

    Returns:
        Series with min/max values for each group

    Example:
        >>> # Get earliest date for each well
        >>> df['first_date'] = get_grouped_min_max(df, 'well_id', 'date', 'min')
    """
    if calc_type not in ["min", "max"]:
        raise ValueError("calc_type must be 'min' or 'max'")

    return df.groupby(group_column)[calc_column].transform(calc_type)


def get_max_initial_production(
    df: pd.DataFrame, n_months: int, production_column: str, date_column: str
) -> float:
    """
    Get maximum production from first N months (handles ramp-up period).

    Useful for determining initial production rate (qi) when wells have
    a ramp-up period before reaching peak production.

    Args:
        df: Input DataFrame (should be for single well)
        n_months: Number of initial months to consider
        production_column: Name of production column (e.g., 'oil_bbl')
        date_column: Name of date column

    Returns:
        Maximum production value from first N months

    Example:
        >>> qi = get_max_initial_production(well_df, 3, 'oil_bbl', 'date')
    """
    df_sorted = df.sort_values(by=date_column)
    df_initial = df_sorted.head(n_months)
    max_value = df_initial[production_column].max()
    return float(max_value) if pd.notna(max_value) else float("nan")


def calculate_cumulative_production(
    df: pd.DataFrame, production_column: str, group_column: Optional[str] = None
) -> pd.Series:
    """
    Calculate cumulative production over time.

    Args:
        df: Input DataFrame
        production_column: Name of production column
        group_column: Optional column to group by (for multiple wells)

    Returns:
        Series with cumulative production

    Example:
        >>> df['cum_oil'] = calculate_cumulative_production(df, 'oil_bbl', 'well_id')
    """
    if group_column:
        return df.groupby(group_column)[production_column].cumsum()
    else:
        return df[production_column].cumsum()


def normalize_production_to_daily(
    df: pd.DataFrame,
    production_column: str,
    days_column: str,
    output_column: Optional[str] = None,
) -> pd.Series:
    """
    Convert monthly production totals to daily rates.

    Args:
        df: Input DataFrame
        production_column: Name of column with monthly production
        days_column: Name of column with days in period
        output_column: Optional name for output column

    Returns:
        Series with daily production rates

    Example:
        >>> df['oil_rate'] = normalize_production_to_daily(df, 'Oil', 'Days')
    """
    if output_column is None:
        output_column = f"{production_column}_daily"

    return df[production_column] / df[days_column]


def filter_wells_by_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    well_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter wells that came online within a date range.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        start_date: Start of date range
        end_date: End of date range
        well_column: Optional well identifier column

    Returns:
        Filtered DataFrame

    Example:
        >>> df_2016 = filter_wells_by_date_range(
        ...     df, 'Online_Date', '2016-01-01', '2016-12-31'
        ... )
    """
    df_filtered = df[
        (df[date_column] >= pd.to_datetime(start_date))
        & (df[date_column] <= pd.to_datetime(end_date))
    ]
    return df_filtered


def calculate_water_cut(
    df: pd.DataFrame, oil_column: str = "Oil", water_column: str = "Wtr"
) -> pd.Series:
    """
    Calculate water cut percentage.

    Water cut = Water / (Oil + Water) * 100

    Args:
        df: Input DataFrame
        oil_column: Name of oil production column
        water_column: Name of water production column

    Returns:
        Series with water cut percentages

    Example:
        >>> df['water_cut'] = calculate_water_cut(df, 'Oil', 'Wtr')
    """
    total_liquid = df[oil_column] + df[water_column]
    water_cut = (df[water_column] / total_liquid * 100).fillna(0)
    return water_cut


def calculate_gor(
    df: pd.DataFrame, gas_column: str = "Gas", oil_column: str = "Oil"
) -> pd.Series:
    """
    Calculate Gas-Oil Ratio (GOR).

    GOR = Gas / Oil (typically in scf/bbl or mcf/bbl)

    Args:
        df: Input DataFrame
        gas_column: Name of gas production column
        oil_column: Name of oil production column

    Returns:
        Series with GOR values

    Example:
        >>> df['gor'] = calculate_gor(df, 'Gas', 'Oil')
    """
    gor = (df[gas_column] / df[oil_column]).replace([np.inf, -np.inf], np.nan)
    return gor


def prepare_well_data_for_dca(
    df: pd.DataFrame,
    well_id: Union[str, int],
    well_column: str = "API_WELLNO",
    date_column: str = "ReportDate",
    production_column: str = "Oil",
    remove_zeros: bool = True,
) -> pd.Series:
    """
    Prepare single well data for decline curve analysis.

    Convenience function that combines filtering, sorting, and formatting
    steps to create a clean time series ready for DCA.

    Args:
        df: Input DataFrame with multiple wells
        well_id: Identifier for the specific well
        well_column: Name of well identifier column
        date_column: Name of date column
        production_column: Name of production column
        remove_zeros: Whether to remove zero/null values

    Returns:
        Time series indexed by date with production values

    Example:
        >>> series = prepare_well_data_for_dca(
        ...     df, '33023013930000', production_column='Oil'
        ... )
        >>> forecast = dca.forecast(series, model='arps', horizon=12)
    """
    # Filter to specific well
    well_df = df[df[well_column] == well_id].copy()

    # Convert date column if needed
    if not pd.api.types.is_datetime64_any_dtype(well_df[date_column]):
        well_df[date_column] = pd.to_datetime(well_df[date_column])

    # Sort by date
    well_df = well_df.sort_values(date_column)

    # Remove zeros/nulls if requested
    if remove_zeros:
        well_df = remove_nan_and_zeroes(well_df, production_column)

    # Create time series
    series = well_df.set_index(date_column)[production_column]
    series.name = production_column.lower()

    return series


def detect_production_anomalies(
    series: pd.Series, threshold_std: float = 3.0
) -> pd.Series:
    """
    Detect anomalous production values using median absolute deviation (MAD).

    Flags values that are more than threshold_std MAD from the rolling median.
    Uses MAD instead of std for robustness to outliers.

    Args:
        series: Production time series
        threshold_std: Number of MAD units for threshold (roughly equivalent to std)

    Returns:
        Boolean series indicating anomalies

    Example:
        >>> anomalies = detect_production_anomalies(oil_series, threshold_std=3.0)
        >>> print(f"Found {anomalies.sum()} anomalies")
    """
    # Use rolling median and MAD (center=False to avoid future data leakage)
    rolling_median = series.rolling(window=3, center=False).median()

    # Calculate MAD (Median Absolute Deviation)
    mad = series.rolling(window=3, center=False).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # Avoid division by zero
    mad = mad.replace(0, 1e-10)

    # Modified z-score using MAD (factor of 1.4826 makes it consistent with std)
    modified_z_score = 0.6745 * np.abs((series - rolling_median) / mad)
    anomalies = modified_z_score > threshold_std

    return anomalies.fillna(False)
