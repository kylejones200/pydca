"""Data quality assurance module.

This module provides comprehensive quality checks for production data,
detecting issues like unit mixes, date gaps, rate resets, and sensor
noise floors. These checks help ensure data quality before fitting
decline curves.

Quality checks:
- Unit mix detection
- Date gap detection
- Duplicate row detection
- Negative rate detection
- Rate reset detection
- Sensor noise floor detection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_contract import validate_production_data
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QAResult:
    """Result of data quality assurance checks.

    Attributes:
        passed: Whether all critical checks passed
        issues: Dictionary mapping issue types to details
        warnings: List of warning messages
        recommendations: List of recommended actions
    """

    passed: bool
    issues: Dict[str, List[Dict]]
    warnings: List[str]
    recommendations: List[str]

    def __str__(self) -> str:
        """Return string representation of QA result."""
        if self.passed:
            return "QA checks passed"
        else:
            issue_count = sum(len(details) for details in self.issues.values())
            return f"QA checks failed: {issue_count} issues found"


def detect_unit_mixes(
    df: pd.DataFrame, rate_columns: List[str], threshold: float = 2.0
) -> List[Dict]:
    """Detect potential unit mixing in rate columns.

    Checks for sudden jumps or discontinuities that might indicate
    unit mixing (e.g., bbl/day vs bbl/month). Only flags increases
    or very large decreases (not normal decline).

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names to check
        threshold: Threshold for detecting jumps (multiple of mean change)

    Returns:
        List of issue dictionaries with details
    """
    issues = []

    for col in rate_columns:
        if col not in df.columns:
            continue

        rates = df[col].dropna()
        if len(rates) < 3:
            continue

        # Compute rate changes
        rate_changes = rates.diff()
        mean_abs_change = rate_changes.abs().mean()

        if mean_abs_change == 0:
            continue

        # Detect large jumps (increases or very large decreases)
        # Normal decline should have consistent negative changes
        # Unit mixing shows as sudden increases or very large decreases
        for i in range(1, len(rates)):
            change = rate_changes.iloc[i]
            abs_change = abs(change)

            # Flag if:
            # 1. Large increase (suspicious)
            # 2. Very large decrease compared to typical decline (threshold * mean)
            if change > 0 or abs_change > (threshold * mean_abs_change):
                if abs_change > (threshold * mean_abs_change):
                    idx = rates.index[i]
                    issues.append(
                        {
                            "column": col,
                            "index": idx,
                            "rate_before": rates.iloc[i - 1],
                            "rate_after": rates.iloc[i],
                            "change_ratio": (
                                abs_change / mean_abs_change
                                if mean_abs_change > 0
                                else 0
                            ),
                            "issue_type": "unit_mix",
                        }
                    )
                    if len(issues) >= 5:  # Limit to first 5
                        break

    return issues


def detect_date_gaps(
    df: pd.DataFrame,
    date_column: str = "date",
    well_id_column: Optional[str] = "well_id",
    max_gap_days: int = 90,
) -> List[Dict]:
    """Detect large gaps in date sequences.

    Args:
        df: DataFrame with production data
        date_column: Name of date column
        well_id_column: Name of well ID column (None if single well)
        max_gap_days: Maximum allowed gap in days

    Returns:
        List of gap issue dictionaries
    """
    issues = []

    if date_column not in df.columns:
        return issues

    if well_id_column and well_id_column in df.columns:
        # Check gaps per well
        for well_id, group in df.groupby(well_id_column):
            group = group.sort_values(date_column)
            gaps = group[date_column].diff().dt.days

            large_gaps = gaps[gaps > max_gap_days]
            if len(large_gaps) > 0:
                for idx in large_gaps.index[:5]:  # Limit to first 5
                    gap_days = gaps.loc[idx]
                    issues.append(
                        {
                            "well_id": well_id,
                            "index": idx,
                            "gap_days": int(gap_days),
                            "date_before": (
                                group.loc[
                                    group.index[group.index < idx][-1], date_column
                                ]
                                if len(group.index[group.index < idx]) > 0
                                else None
                            ),
                            "date_after": group.loc[idx, date_column],
                            "issue_type": "date_gap",
                        }
                    )
    else:
        # Single well
        df_sorted = df.sort_values(date_column)
        gaps = df_sorted[date_column].diff().dt.days

        large_gaps = gaps[gaps > max_gap_days]
        if len(large_gaps) > 0:
            for idx in large_gaps.index[:5]:
                gap_days = gaps.loc[idx]
                issues.append(
                    {
                        "index": idx,
                        "gap_days": int(gap_days),
                        "date_before": (
                            df_sorted.loc[
                                df_sorted.index[df_sorted.index < idx][-1], date_column
                            ]
                            if len(df_sorted.index[df_sorted.index < idx]) > 0
                            else None
                        ),
                        "date_after": df_sorted.loc[idx, date_column],
                        "issue_type": "date_gap",
                    }
                )

    return issues


def detect_rate_resets(
    df: pd.DataFrame, rate_columns: List[str], threshold: float = 0.5
) -> List[Dict]:
    """Detect rate resets (sudden increases after decline).

    Rate resets can indicate workovers, recompletions, or data errors.

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names to check
        threshold: Minimum increase ratio to flag as reset

    Returns:
        List of reset issue dictionaries
    """
    issues = []

    for col in rate_columns:
        if col not in df.columns:
            continue

        rates = df[col].dropna()
        if len(rates) < 3:
            continue

        # Compute rate changes
        rate_changes = rates.diff()

        # Find increases after decreases (potential resets)
        decreasing = rate_changes < 0
        increasing = rate_changes > 0

        # Look for pattern: decrease followed by large increase
        for i in range(1, len(rates) - 1):
            if decreasing.iloc[i] and increasing.iloc[i + 1]:
                increase_ratio = rate_changes.iloc[i + 1] / rates.iloc[i]
                if increase_ratio > threshold:
                    issues.append(
                        {
                            "column": col,
                            "index": rates.index[i + 1],
                            "rate_before": rates.iloc[i],
                            "rate_after": rates.iloc[i + 1],
                            "increase_ratio": increase_ratio,
                            "issue_type": "rate_reset",
                        }
                    )

    return issues


def detect_sensor_noise_floor(
    df: pd.DataFrame,
    rate_columns: List[str],
    noise_threshold: Optional[float] = None,
) -> Tuple[List[Dict], Optional[float]]:
    """Detect sensor noise floor (very low rates that may be noise).

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names to check
        noise_threshold: Optional explicit noise threshold (auto-detect if None)

    Returns:
        Tuple of (issues list, detected noise threshold)
    """
    issues = []
    detected_threshold = noise_threshold

    for col in rate_columns:
        if col not in df.columns:
            continue

        rates = df[col].dropna()
        if len(rates) == 0:
            continue

        # Auto-detect noise floor if not provided
        if detected_threshold is None:
            # Use 1% of median rate as noise floor estimate
            median_rate = rates.median()
            if median_rate > 0:
                detected_threshold = median_rate * 0.01
            else:
                detected_threshold = 0.0

        # Find rates below noise floor
        low_rates = rates[rates < detected_threshold]

        if len(low_rates) > 0:
            issues.append(
                {
                    "column": col,
                    "count": len(low_rates),
                    "noise_threshold": detected_threshold,
                    "min_rate": float(rates.min()),
                    "max_rate": float(rates.max()),
                    "median_rate": float(rates.median()),
                    "issue_type": "sensor_noise",
                }
            )

    return issues, detected_threshold


def detect_duplicate_rows(
    df: pd.DataFrame,
    date_column: str = "date",
    well_id_column: Optional[str] = "well_id",
) -> List[Dict]:
    """Detect duplicate rows.

    Args:
        df: DataFrame with production data
        date_column: Name of date column
        well_id_column: Name of well ID column

    Returns:
        List of duplicate issue dictionaries
    """
    issues = []

    # Check for exact duplicates
    duplicates = df.duplicated()
    if duplicates.any():
        duplicate_count = duplicates.sum()
        issues.append(
            {
                "count": int(duplicate_count),
                "issue_type": "duplicate_rows",
            }
        )

    # Check for duplicate well-date combinations
    if well_id_column and well_id_column in df.columns and date_column in df.columns:
        well_date_duplicates = df.duplicated(subset=[well_id_column, date_column])
        if well_date_duplicates.any():
            duplicate_count = well_date_duplicates.sum()
            issues.append(
                {
                    "count": int(duplicate_count),
                    "issue_type": "duplicate_well_dates",
                }
            )

    return issues


def detect_negative_rates(df: pd.DataFrame, rate_columns: List[str]) -> List[Dict]:
    """Detect negative rate values.

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names to check

    Returns:
        List of negative rate issue dictionaries
    """
    issues = []

    for col in rate_columns:
        if col not in df.columns:
            continue

        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            issues.append(
                {
                    "column": col,
                    "count": int(negative_count),
                    "min_value": float(df[col].min()),
                    "issue_type": "negative_rates",
                }
            )

    return issues


def run_data_qa(
    df: pd.DataFrame,
    date_column: str = "date",
    well_id_column: Optional[str] = "well_id",
    rate_columns: Optional[List[str]] = None,
    max_gap_days: int = 90,
    noise_threshold: Optional[float] = None,
    check_unit_mixes: bool = True,
    check_rate_resets: bool = True,
) -> QAResult:
    """Run comprehensive data quality assurance checks.

    Args:
        df: DataFrame with production data
        date_column: Name of date column
        well_id_column: Name of well ID column
        rate_columns: List of rate column names (auto-detect if None)
        max_gap_days: Maximum allowed date gap in days
        noise_threshold: Optional noise floor threshold
        check_unit_mixes: Whether to check for unit mixing
        check_rate_resets: Whether to check for rate resets

    Returns:
        QAResult with all QA findings

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
        ...     'oil_rate': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        ... })
        >>> qa_result = run_data_qa(df, well_id_column=None)
        >>> if qa_result.passed:
        ...     print("Data quality is good")
        ... else:
        ...     print(f"Issues found: {qa_result.issues}")
    """
    issues: Dict[str, List[Dict]] = {}
    warnings: List[str] = []
    recommendations: List[str] = []

    # Auto-detect rate columns if not provided
    if rate_columns is None:
        rate_candidates = [
            "rate",
            "oil_rate",
            "gas_rate",
            "water_rate",
            "oil",
            "gas",
            "water",
        ]
        rate_columns = [col for col in rate_candidates if col in df.columns]

    if not rate_columns:
        warnings.append("No rate columns found for QA checks")
        return QAResult(
            passed=False,
            issues={"no_rate_columns": []},
            warnings=warnings,
            recommendations=["Specify rate columns or ensure standard column names"],
        )

    # Run basic validation first
    validation_result = validate_production_data(
        df, date_column, well_id_column, rate_columns, strict=False
    )

    if not validation_result.is_valid:
        issues["validation"] = [{"reason_codes": validation_result.reason_codes}]
        warnings.extend(validation_result.warnings)

    # Check for duplicate rows
    duplicate_issues = detect_duplicate_rows(df, date_column, well_id_column)
    if duplicate_issues:
        issues["duplicates"] = duplicate_issues
        recommendations.append("Remove duplicate rows before analysis")

    # Check for negative rates
    negative_issues = detect_negative_rates(df, rate_columns)
    if negative_issues:
        issues["negative_rates"] = negative_issues
        recommendations.append("Review and correct negative rate values")

    # Check for date gaps
    gap_issues = detect_date_gaps(df, date_column, well_id_column, max_gap_days)
    if gap_issues:
        issues["date_gaps"] = gap_issues
        warnings.append(
            f"Found {len(gap_issues)} large date gaps (> {max_gap_days} days)"
        )

    # Check for sensor noise floor
    noise_issues, detected_threshold = detect_sensor_noise_floor(
        df, rate_columns, noise_threshold
    )
    if noise_issues:
        issues["sensor_noise"] = noise_issues
        if detected_threshold:
            recommendations.append(
                f"Consider filtering rates below {detected_threshold:.2f} (noise floor)"
            )

    # Check for unit mixes
    if check_unit_mixes:
        unit_mix_issues = detect_unit_mixes(df, rate_columns)
        if unit_mix_issues:
            issues["unit_mixes"] = unit_mix_issues
            recommendations.append(
                "Review data for potential unit mixing (e.g., bbl/day vs bbl/month)"
            )

    # Check for rate resets
    if check_rate_resets:
        reset_issues = detect_rate_resets(df, rate_columns)
        if reset_issues:
            issues["rate_resets"] = reset_issues
            warnings.append(
                f"Found {len(reset_issues)} potential rate resets "
                f"(workovers/recompletions)"
            )

    # Determine if passed (no critical issues)
    critical_issues = ["validation", "duplicates", "negative_rates"]
    has_critical_issues = any(key in issues for key in critical_issues)

    passed = not has_critical_issues

    return QAResult(
        passed=passed,
        issues=issues,
        warnings=warnings,
        recommendations=recommendations,
    )


def apply_rate_cut(
    df: pd.DataFrame,
    rate_columns: List[str],
    cutoff: float,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply rate cutoff to remove values below noise floor.

    Args:
        df: DataFrame with production data
        rate_columns: List of rate column names
        cutoff: Rate cutoff value (values below will be set to NaN)
        inplace: If True, modify DataFrame in place

    Returns:
        DataFrame with rates below cutoff removed
    """
    if not inplace:
        df = df.copy()

    for col in rate_columns:
        if col in df.columns:
            df.loc[df[col] < cutoff, col] = np.nan

    return df
