"""Tests for data QA module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.data_qa import (
    QAResult,
    apply_rate_cut,
    detect_date_gaps,
    detect_duplicate_rows,
    detect_negative_rates,
    detect_rate_resets,
    detect_sensor_noise_floor,
    detect_unit_mixes,
    run_data_qa,
)


class TestDetectDateGaps:
    """Test date gap detection."""

    def test_no_gaps(self):
        """Test with no gaps."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [100] * 10,
            }
        )

        issues = detect_date_gaps(df, max_gap_days=90)

        assert len(issues) == 0

    def test_large_gap(self):
        """Test detection of large gap."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS").tolist()
        dates.append(pd.Timestamp("2020-12-01"))  # Large gap
        df = pd.DataFrame(
            {
                "date": dates,
                "oil_rate": [100] * 6,
            }
        )

        issues = detect_date_gaps(df, max_gap_days=90)

        assert len(issues) > 0
        assert issues[0]["issue_type"] == "date_gap"

    def test_multi_well_gaps(self):
        """Test gap detection with multiple wells."""
        dates1 = pd.date_range("2020-01-01", periods=3, freq="MS").tolist()
        dates1.append(pd.Timestamp("2020-12-01"))  # Gap
        dates2 = pd.date_range("2020-01-01", periods=4, freq="MS")

        df = pd.DataFrame(
            {
                "well_id": ["WELL_001"] * 4 + ["WELL_002"] * 4,
                "date": dates1 + dates2.tolist(),
                "oil_rate": [100] * 8,
            }
        )

        issues = detect_date_gaps(df, well_id_column="well_id", max_gap_days=90)

        assert len(issues) > 0


class TestDetectNegativeRates:
    """Test negative rate detection."""

    def test_no_negative_rates(self):
        """Test with no negative rates."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 70],
            }
        )

        issues = detect_negative_rates(df, ["oil_rate"])

        assert len(issues) == 0

    def test_negative_rates_detected(self):
        """Test detection of negative rates."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, -10, 70],
            }
        )

        issues = detect_negative_rates(df, ["oil_rate"])

        assert len(issues) > 0
        assert issues[0]["issue_type"] == "negative_rates"
        assert issues[0]["count"] == 1


class TestDetectRateResets:
    """Test rate reset detection."""

    def test_rate_reset_detected(self):
        """Test detection of rate reset."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 150, 140],  # Reset at index 3
            }
        )

        issues = detect_rate_resets(df, ["oil_rate"], threshold=0.5)

        assert len(issues) > 0
        assert issues[0]["issue_type"] == "rate_reset"

    def test_no_reset(self):
        """Test with no resets."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 70, 60],  # Declining
            }
        )

        issues = detect_rate_resets(df, ["oil_rate"])

        assert len(issues) == 0


class TestDetectSensorNoiseFloor:
    """Test sensor noise floor detection."""

    def test_noise_floor_detection(self):
        """Test auto-detection of noise floor."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 0.5, 0.3, 0.1],  # Low values at end
            }
        )

        issues, threshold = detect_sensor_noise_floor(df, ["oil_rate"])

        assert threshold is not None
        assert threshold > 0

    def test_explicit_threshold(self):
        """Test with explicit threshold."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 0.5, 0.3],
            }
        )

        issues, threshold = detect_sensor_noise_floor(
            df, ["oil_rate"], noise_threshold=1.0
        )

        assert threshold == 1.0
        assert len(issues) > 0


class TestDetectDuplicateRows:
    """Test duplicate row detection."""

    def test_exact_duplicates(self):
        """Test detection of exact duplicates."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "oil_rate": [100, 90, 80],
            }
        )
        df = pd.concat([df, df.iloc[[0]]])  # Add duplicate

        issues = detect_duplicate_rows(df)

        assert len(issues) > 0
        assert any(issue["issue_type"] == "duplicate_rows" for issue in issues)

    def test_duplicate_well_dates(self):
        """Test detection of duplicate well-date combinations."""
        df = pd.DataFrame(
            {
                "well_id": ["WELL_001", "WELL_001", "WELL_002"],
                "date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),  # Duplicate
                    pd.Timestamp("2020-01-01"),
                ],
                "oil_rate": [100, 90, 80],
            }
        )

        issues = detect_duplicate_rows(df, well_id_column="well_id")

        assert len(issues) > 0


class TestRunDataQA:
    """Test comprehensive QA function."""

    def test_clean_data_passes(self):
        """Test that clean data passes QA."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )

        result = run_data_qa(df, well_id_column=None)

        assert result.passed
        assert len(result.issues) == 0

    def test_data_with_issues(self):
        """Test QA detects issues."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [
                    100,
                    90,
                    -10,
                    70,
                    60,
                    50,
                    40,
                    30,
                    20,
                    10,
                ],  # Negative value
            }
        )

        result = run_data_qa(df, well_id_column=None)

        assert not result.passed
        assert "negative_rates" in result.issues

    def test_recommendations_provided(self):
        """Test that recommendations are provided."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "oil_rate": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )
        df = pd.concat([df, df.iloc[[0]]])  # Add duplicate

        result = run_data_qa(df, well_id_column=None)

        assert len(result.recommendations) > 0


class TestApplyRateCut:
    """Test rate cutoff application."""

    def test_apply_cutoff(self):
        """Test applying rate cutoff."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 90, 80, 5, 3, 1],  # Low values at end
            }
        )

        df_cut = apply_rate_cut(df, ["oil_rate"], cutoff=10.0, inplace=False)

        # Values below 10 should be NaN
        assert pd.isna(df_cut.loc[3, "oil_rate"])
        assert pd.isna(df_cut.loc[4, "oil_rate"])
        assert pd.isna(df_cut.loc[5, "oil_rate"])
        # Values above 10 should remain
        assert df_cut.loc[0, "oil_rate"] == 100

    def test_inplace_modification(self):
        """Test inplace modification."""
        df = pd.DataFrame(
            {
                "oil_rate": [100, 5, 3],
            }
        )

        df_original = df.copy()
        apply_rate_cut(df, ["oil_rate"], cutoff=10.0, inplace=True)

        # Original should be modified
        assert pd.isna(df.loc[1, "oil_rate"])
        assert pd.isna(df.loc[2, "oil_rate"])
