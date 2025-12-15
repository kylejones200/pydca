"""Tests for data contract module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.data_contract import (
    ProductionDataValidator,
    ValidationResult,
    normalize_to_canonical_schema,
    validate_production_data,
)


class TestProductionDataValidator:
    """Test ProductionDataValidator class."""

    def test_valid_single_well_data(self):
        """Test validation of valid single well data."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )

        validator = ProductionDataValidator(well_id_column=None)
        result = validator.validate(df)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_multi_well_data(self):
        """Test validation of valid multi-well data."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        df = pd.DataFrame(
            {
                "well_id": ["WELL_001"] * 5 + ["WELL_002"] * 5,
                "date": list(dates) * 2,
                "oil_rate": [100, 90, 80, 70, 60] * 2,
            }
        )

        validator = ProductionDataValidator()
        result = validator.validate(df)

        assert result.is_valid

    def test_missing_date_column(self):
        """Test validation fails with missing date column."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
            }
        )

        validator = ProductionDataValidator(well_id_column=None)
        result = validator.validate(df)

        assert not result.is_valid
        assert "MISSING_REQUIRED_COLUMNS" in result.reason_codes

    def test_missing_rate_column(self):
        """Test validation fails with missing rate column."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            }
        )

        validator = ProductionDataValidator(well_id_column=None)
        result = validator.validate(df)

        assert not result.is_valid
        assert "NO_RATE_COLUMNS" in result.reason_codes

    def test_unsorted_dates(self):
        """Test validation detects unsorted dates."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
                "oil_rate": [100, 90, 80],
            }
        )

        validator = ProductionDataValidator(well_id_column=None, strict=True)
        result = validator.validate(df)

        assert not result.is_valid
        assert "UNSORTED_DATES" in result.reason_codes

    def test_unsorted_dates_warning_mode(self):
        """Test unsorted dates as warning in non-strict mode."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
                "oil_rate": [100, 90, 80],
            }
        )

        validator = ProductionDataValidator(well_id_column=None, strict=False)
        result = validator.validate(df)

        # Should still be valid (warning only)
        assert result.is_valid
        assert "UNSORTED_DATES" in result.reason_codes
        assert len(result.warnings) > 0

    def test_duplicate_well_dates(self):
        """Test validation detects duplicate well-date combinations."""
        df = pd.DataFrame(
            {
                "well_id": ["WELL_001", "WELL_001", "WELL_001"],
                "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
                "oil_rate": [100, 90, 80],
            }
        )

        validator = ProductionDataValidator(strict=True)
        result = validator.validate(df)

        assert not result.is_valid
        assert "DUPLICATE_WELL_DATES" in result.reason_codes

    def test_negative_rates(self):
        """Test validation detects negative rates."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "oil_rate": [100, 90, -10, 70, 60],  # Negative value
            }
        )

        validator = ProductionDataValidator(well_id_column=None)
        result = validator.validate(df)

        # Should still be valid but with warning
        assert result.is_valid
        assert "NEGATIVE_RATES" in result.reason_codes
        assert len(result.warnings) > 0

    def test_invalid_date_type(self):
        """Test validation detects invalid date type."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "invalid"],
                "oil_rate": [100, 90, 80],
            }
        )

        validator = ProductionDataValidator(well_id_column=None)
        result = validator.validate(df)

        assert not result.is_valid
        assert "INVALID_DATE_COLUMN" in result.reason_codes


class TestValidateProductionData:
    """Test validate_production_data convenience function."""

    def test_convenience_function(self):
        """Test convenience function works."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )

        result = validate_production_data(df, well_id_column=None)

        assert result.is_valid
        assert isinstance(result, ValidationResult)


class TestNormalizeToCanonicalSchema:
    """Test normalize_to_canonical_schema function."""

    def test_normalize_column_names(self):
        """Test normalization of column names."""
        df = pd.DataFrame(
            {
                "ReportDate": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "Oil": [100, 90, 80, 70, 60],
                "WellID": ["WELL_001"] * 5,
            }
        )

        df_norm = normalize_to_canonical_schema(
            df,
            date_column="ReportDate",
            well_id_column="WellID",
            rate_mapping={"Oil": "oil_rate"},
        )

        assert "date" in df_norm.columns
        assert "well_id" in df_norm.columns
        assert "oil_rate" in df_norm.columns

    def test_normalize_sorts_dates(self):
        """Test normalization sorts by date."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
                "oil_rate": [100, 90, 80],
            }
        )

        df_norm = normalize_to_canonical_schema(df)

        assert df_norm["date"].is_monotonic_increasing

    def test_normalize_converts_date_type(self):
        """Test normalization converts date to datetime."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "oil_rate": [100, 90, 80],
            }
        )

        df_norm = normalize_to_canonical_schema(df)

        assert pd.api.types.is_datetime64_any_dtype(df_norm["date"])
