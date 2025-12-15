"""Data contract for production data.

This module defines the canonical schema for production data and provides
validation to ensure data quality before processing.

The canonical schema enforces:
- Required columns: date, rate (or phase-specific rates)
- Optional columns: cumulative, uptime, hours_on
- Sorted dates
- One row per well per date
- Validator returns reason codes for issues
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        is_valid: Whether data passes all validation checks
        reason_codes: List of reason codes for any issues found
        warnings: List of warning messages
        errors: List of error messages
    """

    is_valid: bool
    reason_codes: List[str]
    warnings: List[str]
    errors: List[str]

    def __str__(self) -> str:
        """Return string representation of validation result."""
        if self.is_valid:
            return "Validation passed"
        else:
            return f"Validation failed: {', '.join(self.reason_codes)}"


# Canonical schema definition
CANONICAL_SCHEMA = {
    "required": {
        "date": "datetime64[ns]",
        # At least one rate column required
        "rate": "float64",  # Generic rate, OR
        "oil_rate": "float64",  # Phase-specific rates
        "gas_rate": "float64",
        "water_rate": "float64",
    },
    "optional": {
        "well_id": "object",  # String identifier
        "cumulative": "float64",
        "cum_oil": "float64",
        "cum_gas": "float64",
        "cum_water": "float64",
        "uptime": "float64",  # Fraction (0-1) or hours
        "hours_on": "float64",
        "allocated_volume": "float64",
    },
}


class ProductionDataValidator:
    """Validator for production data against canonical schema."""

    def __init__(
        self,
        date_column: str = "date",
        well_id_column: Optional[str] = "well_id",
        rate_columns: Optional[List[str]] = None,
        strict: bool = True,
    ):
        """Initialize validator.

        Args:
            date_column: Name of date column
            well_id_column: Name of well ID column (None if single well)
            rate_columns: List of rate column names (auto-detect if None)
            strict: If True, errors fail validation. If False, warnings only.
        """
        self.date_column = date_column
        self.well_id_column = well_id_column
        self.rate_columns = rate_columns
        self.strict = strict

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame against canonical schema.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and reason codes
        """
        reason_codes: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        # Check 1: Required columns
        missing_cols = self._check_required_columns(df)
        if missing_cols:
            reason_codes.append("MISSING_REQUIRED_COLUMNS")
            errors.append(f"Missing required columns: {missing_cols}")

        # Check 2: Date column exists and is datetime
        date_issue = self._check_date_column(df)
        if date_issue:
            reason_codes.append("INVALID_DATE_COLUMN")
            errors.append(date_issue)

        # Check 3: Sorted dates
        if not date_issue and self.date_column in df.columns:
            if not self._check_sorted_dates(df):
                reason_codes.append("UNSORTED_DATES")
                if self.strict:
                    errors.append("Dates must be sorted")
                else:
                    warnings.append("Dates are not sorted")

        # Check 4: One row per well per date
        if self.well_id_column and self.well_id_column in df.columns:
            duplicates = self._check_duplicate_well_dates(df)
            if duplicates:
                reason_codes.append("DUPLICATE_WELL_DATES")
                if self.strict:
                    errors.append(
                        f"Found {duplicates} duplicate well-date combinations"
                    )
                else:
                    warnings.append(
                        f"Found {duplicates} duplicate well-date combinations"
                    )

        # Check 5: Rate columns valid
        rate_issues = self._check_rate_columns(df)
        if rate_issues:
            reason_codes.extend(rate_issues["codes"])
            errors.extend(rate_issues["errors"])
            warnings.extend(rate_issues["warnings"])

        # Check 6: No negative rates
        negative_rates = self._check_negative_rates(df)
        if negative_rates:
            reason_codes.append("NEGATIVE_RATES")
            warnings.append(f"Found {negative_rates} negative rate values")

        # Determine if valid
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            reason_codes=reason_codes,
            warnings=warnings,
            errors=errors,
        )

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for required columns."""
        missing = []

        # Date column is always required
        if self.date_column not in df.columns:
            missing.append(self.date_column)

        # At least one rate column required
        rate_cols = self._detect_rate_columns(df)
        if not rate_cols:
            missing.append("rate (or oil_rate/gas_rate/water_rate)")

        return missing

    def _check_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Check date column is valid datetime."""
        if self.date_column not in df.columns:
            return None  # Already caught by required columns check

        col = df[self.date_column]

        # Check if datetime type
        if not pd.api.types.is_datetime64_any_dtype(col):
            # Try to convert
            try:
                pd.to_datetime(col)
            except Exception as e:
                return (
                    f"Date column '{self.date_column}' "
                    f"cannot be converted to datetime: {e}"
                )

        # Check for null dates
        null_count = col.isna().sum()
        if null_count > 0:
            return f"Date column '{self.date_column}' has {null_count} null values"

        return None

    def _check_sorted_dates(self, df: pd.DataFrame) -> bool:
        """Check if dates are sorted."""
        if self.well_id_column and self.well_id_column in df.columns:
            # Check sorted per well
            for well_id, group in df.groupby(self.well_id_column):
                if not group[self.date_column].is_monotonic_increasing:
                    return False
            return True
        else:
            # Single well, check overall
            return df[self.date_column].is_monotonic_increasing

    def _check_duplicate_well_dates(self, df: pd.DataFrame) -> int:
        """Check for duplicate well-date combinations."""
        if not self.well_id_column or self.well_id_column not in df.columns:
            return 0

        duplicates = df.duplicated(subset=[self.well_id_column, self.date_column]).sum()
        return int(duplicates)

    def _detect_rate_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect rate columns in DataFrame."""
        if self.rate_columns:
            return [col for col in self.rate_columns if col in df.columns]

        # Auto-detect
        rate_candidates = [
            "rate",
            "oil_rate",
            "gas_rate",
            "water_rate",
            "oil",
            "gas",
            "water",
        ]
        return [col for col in rate_candidates if col in df.columns]

    def _check_rate_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check rate columns for validity."""
        rate_cols = self._detect_rate_columns(df)
        codes: List[str] = []
        errors: List[str] = []
        warnings: List[str] = []

        if not rate_cols:
            codes.append("NO_RATE_COLUMNS")
            errors.append("No rate columns found")
            return {"codes": codes, "errors": errors, "warnings": warnings}

        # Check each rate column
        for col in rate_cols:
            # Check for all null
            if df[col].isna().all():
                codes.append(f"ALL_NULL_RATES_{col.upper()}")
                warnings.append(f"Rate column '{col}' is entirely null")

            # Check data type
            if not pd.api.types.is_numeric_dtype(df[col]):
                codes.append(f"NON_NUMERIC_RATES_{col.upper()}")
                errors.append(f"Rate column '{col}' is not numeric")

        return {"codes": codes, "errors": errors, "warnings": warnings}

    def _check_negative_rates(self, df: pd.DataFrame) -> int:
        """Check for negative rate values."""
        rate_cols = self._detect_rate_columns(df)
        negative_count = 0

        for col in rate_cols:
            negative_count += (df[col] < 0).sum()

        return int(negative_count)


def validate_production_data(
    df: pd.DataFrame,
    date_column: str = "date",
    well_id_column: Optional[str] = "well_id",
    rate_columns: Optional[List[str]] = None,
    strict: bool = True,
) -> ValidationResult:
    """Validate production data against canonical schema.

    Convenience function for validating production data.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        well_id_column: Name of well ID column (None if single well)
        rate_columns: List of rate column names (auto-detect if None)
        strict: If True, errors fail validation. If False, warnings only.

    Returns:
        ValidationResult with validation status and reason codes

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=10),
        ...     'oil_rate': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        ... })
        >>> result = validate_production_data(df)
        >>> if result.is_valid:
        ...     print("Data is valid")
        ... else:
        ...     print(f"Issues: {result.reason_codes}")
    """
    validator = ProductionDataValidator(
        date_column=date_column,
        well_id_column=well_id_column,
        rate_columns=rate_columns,
        strict=strict,
    )
    return validator.validate(df)


def normalize_to_canonical_schema(
    df: pd.DataFrame,
    date_column: str = "date",
    well_id_column: Optional[str] = "well_id",
    rate_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize DataFrame to canonical schema.

    Renames columns and ensures proper types to match canonical schema.

    Args:
        df: DataFrame to normalize
        date_column: Name of date column in input
        well_id_column: Name of well ID column in input
        rate_mapping: Dictionary mapping input rate columns to canonical names
                     (e.g., {'Oil': 'oil_rate', 'Gas': 'gas_rate'})

    Returns:
        Normalized DataFrame with canonical column names
    """
    df_norm = df.copy()

    # Rename date column if needed
    if date_column != "date" and date_column in df_norm.columns:
        df_norm = df_norm.rename(columns={date_column: "date"})

    # Ensure date is datetime
    if "date" in df_norm.columns:
        df_norm["date"] = pd.to_datetime(df_norm["date"])

    # Rename well_id if needed
    if (
        well_id_column
        and well_id_column != "well_id"
        and well_id_column in df_norm.columns
    ):
        df_norm = df_norm.rename(columns={well_id_column: "well_id"})

    # Apply rate mapping
    if rate_mapping:
        df_norm = df_norm.rename(columns=rate_mapping)

    # Sort by date (and well_id if present)
    if "well_id" in df_norm.columns:
        df_norm = df_norm.sort_values(["well_id", "date"])
    else:
        df_norm = df_norm.sort_values("date")

    return df_norm
