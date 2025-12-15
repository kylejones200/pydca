"""Tests for downtime and allocation module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.downtime import (
    DowntimeResult,
    apply_allocation_adjustment,
    detect_downtime_periods,
    reconstruct_rate_from_uptime,
    validate_uptime_data,
)


class TestReconstructRateFromUptime:
    """Test rate reconstruction from uptime."""

    def test_reconstruct_with_uptime_fraction(self):
        """Test reconstruction with uptime as fraction."""
        df = pd.DataFrame(
            {
                "oil_bbl": [3000, 2700, 2400],
                "uptime": [0.95, 0.90, 0.85],  # 95%, 90%, 85% uptime
            }
        )

        df_result = reconstruct_rate_from_uptime(df, "oil_bbl")

        # Should have rate column
        assert "oil_rate" in df_result.columns

        # Rate should be higher than volume/30 (accounting for uptime)
        # 3000 / (0.95 * 30.4375) ≈ 103.7
        assert df_result["oil_rate"].iloc[0] > 100

    def test_reconstruct_with_hours_on(self):
        """Test reconstruction with hours_on."""
        df = pd.DataFrame(
            {
                "oil_bbl": [3000, 2700],
                "hours_on": [700, 650],  # Hours in month
            }
        )

        df_result = reconstruct_rate_from_uptime(
            df, "oil_bbl", hours_on_column="hours_on"
        )

        assert "oil_rate" in df_result.columns

    def test_missing_uptime_handling(self):
        """Test handling of missing uptime values."""
        df = pd.DataFrame(
            {
                "oil_bbl": [3000, 2700, 2400],
                "uptime": [0.95, np.nan, 0.85],
            }
        )

        df_result = reconstruct_rate_from_uptime(df, "oil_bbl")

        # Middle value should be NaN
        assert pd.isna(df_result["oil_rate"].iloc[1])
        # Other values should be computed
        assert not pd.isna(df_result["oil_rate"].iloc[0])
        assert not pd.isna(df_result["oil_rate"].iloc[2])


class TestValidateUptimeData:
    """Test uptime data validation."""

    def test_validate_with_uptime(self):
        """Test validation with uptime column."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
                "uptime": [0.95, 0.90, 0.85],
            }
        )

        result = validate_uptime_data(df)

        assert result.has_uptime_data
        assert result.uptime_coverage == 1.0
        assert result.missing_uptime_count == 0

    def test_validate_with_missing_uptime(self):
        """Test validation with missing uptime."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
                "uptime": [0.95, np.nan, 0.85],
            }
        )

        result = validate_uptime_data(df)

        assert result.has_uptime_data
        assert result.uptime_coverage < 1.0
        assert result.missing_uptime_count == 1

    def test_validate_without_uptime(self):
        """Test validation without uptime column."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
            }
        )

        result = validate_uptime_data(df)

        assert not result.has_uptime_data
        assert len(result.warnings) > 0


class TestDetectDowntimePeriods:
    """Test downtime period detection."""

    def test_detect_downtime(self):
        """Test detection of downtime periods."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60],
                "uptime": [0.95, 0.90, 0.30, 0.85, 0.80],  # Low uptime at index 2
            }
        )

        df_result = detect_downtime_periods(df, threshold=0.5, min_duration_days=0)

        assert "is_downtime" in df_result.columns
        assert df_result["is_downtime"].iloc[2] == True  # Low uptime period

    def test_no_downtime(self):
        """Test with no downtime."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60],
                "uptime": [0.95, 0.90, 0.85, 0.80, 0.75],  # All above threshold
            }
        )

        df_result = detect_downtime_periods(df, threshold=0.5)

        assert df_result["is_downtime"].sum() == 0


class TestApplyAllocationAdjustment:
    """Test allocation adjustment."""

    def test_replace_method(self):
        """Test allocation with replace method."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
                "allocated_volume": [np.nan, 95, np.nan],
            }
        )

        df_result = apply_allocation_adjustment(df, ["oil_rate"], method="replace")

        # Second value should be replaced
        assert df_result["oil_rate"].iloc[1] == 95
        # Others unchanged
        assert df_result["oil_rate"].iloc[0] == 100
        assert df_result["oil_rate"].iloc[2] == 80

    def test_multiply_method(self):
        """Test allocation with multiply method."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80],
                "allocated_volume": [np.nan, 1.1, np.nan],  # 10% adjustment
            }
        )

        df_result = apply_allocation_adjustment(df, ["oil_rate"], method="multiply")

        # Second value should be multiplied
        assert np.isclose(df_result["oil_rate"].iloc[1], 99.0)  # 90 * 1.1
