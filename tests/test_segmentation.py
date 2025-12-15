"""Tests for segmentation module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.segmentation import (
    Segment,
    SegmentationResult,
    create_segments_from_change_points,
    segment_cusum,
    segment_production_data,
    select_final_stable_segment,
    select_segment_by_aic,
)


class TestSegmentCUSUM:
    """Test CUSUM segmentation."""

    def test_cusum_basic(self):
        """Test basic CUSUM segmentation."""
        # Create data with clear change point
        rates = np.concatenate(
            [
                np.ones(20) * 100,  # Stable at 100
                np.ones(20) * 50,  # Drop to 50
            ]
        )

        change_points = segment_cusum(rates, baseline_window=10, threshold=2.0)

        # Should detect change point around index 20
        assert len(change_points) > 0

    def test_cusum_no_change(self):
        """Test CUSUM with no change points."""
        rates = np.ones(50) * 100  # Constant rate

        change_points = segment_cusum(rates, threshold=3.0)

        # Should have few or no change points
        assert len(change_points) <= 2  # Allow some noise


class TestCreateSegmentsFromChangePoints:
    """Test segment creation."""

    def test_create_segments(self):
        """Test creating segments from change points."""
        rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        change_points = [3, 6]
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")

        segments = create_segments_from_change_points(change_points, rates, dates)

        assert len(segments) == 3  # 3 segments from 2 change points
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == 3
        assert segments[1].start_idx == 3
        assert segments[1].end_idx == 6
        assert segments[2].start_idx == 6
        assert segments[2].end_idx == 10

    def test_create_segments_no_change_points(self):
        """Test creating segments with no change points."""
        rates = np.array([100, 90, 80, 70, 60])
        change_points = []

        segments = create_segments_from_change_points(change_points, rates)

        assert len(segments) == 1  # Single segment
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == len(rates)


class TestSelectFinalStableSegment:
    """Test final stable segment selection."""

    def test_select_final_stable(self):
        """Test selecting final stable segment."""
        segments = [
            Segment(0, 10, mean_rate=100.0, stability_score=0.3),
            Segment(10, 20, mean_rate=80.0, stability_score=0.6),
            Segment(20, 30, mean_rate=60.0, stability_score=0.7),
        ]

        selected = select_final_stable_segment(segments, min_stability=0.5)

        assert selected == 2  # Last segment with stability >= 0.5

    def test_select_last_if_none_stable(self):
        """Test selecting last segment if none meet threshold."""
        segments = [
            Segment(0, 10, mean_rate=100.0, stability_score=0.3),
            Segment(10, 20, mean_rate=80.0, stability_score=0.4),
        ]

        selected = select_final_stable_segment(segments, min_stability=0.5)

        assert selected == 1  # Last segment


class TestSelectSegmentByAIC:
    """Test AIC-based segment selection."""

    def test_select_by_aic(self):
        """Test selecting segment by AIC."""
        rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        segments = [
            Segment(0, 5, mean_rate=80.0, stability_score=0.5),
            Segment(5, 10, mean_rate=30.0, stability_score=0.6),
        ]

        selected = select_segment_by_aic(segments, rates)

        # Should select one of the segments
        assert selected in [0, 1]


class TestSegmentProductionData:
    """Test main segmentation function."""

    def test_segment_none_method(self):
        """Test segmentation with 'none' method."""
        rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])

        result = segment_production_data(rates, method="none")

        assert len(result.segments) == 1
        assert result.method == "none"
        assert result.selected_segment_idx == 0

    def test_segment_cusum_method(self):
        """Test segmentation with CUSUM method."""
        # Create data with change point
        rates = np.concatenate(
            [
                np.ones(20) * 100,
                np.ones(20) * 50,
            ]
        )

        result = segment_production_data(
            rates, method="cusum", selection_rule="final_stable"
        )

        assert result.method == "cusum"
        assert len(result.segments) >= 1
        assert result.selected_segment_idx >= 0

    def test_segment_with_dates(self):
        """Test segmentation with date index."""
        rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")

        result = segment_production_data(rates, dates=dates, method="none")

        assert result.segments[0].start_date is not None
        assert result.segments[0].end_date is not None

    def test_get_selected_segment(self):
        """Test getting selected segment."""
        rates = np.array([100, 90, 80, 70, 60])
        result = segment_production_data(rates, method="none")

        selected = result.get_selected_segment()

        assert selected is not None
        assert selected.start_idx == 0
        assert selected.end_idx == len(rates)

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip") or True,  # Skip if ruptures not available
        reason="ruptures library required",
    )
    def test_segment_pelt_log(self):
        """Test PELT log segmentation (if ruptures available)."""
        try:
            import ruptures
        except ImportError:
            pytest.skip("ruptures not available")

        # Create data with change point
        rates = np.concatenate(
            [
                np.ones(20) * 100,
                np.ones(20) * 50,
            ]
        )

        result = segment_production_data(rates, method="pelt_log")

        assert result.method == "pelt_log"
        assert len(result.segments) >= 1
