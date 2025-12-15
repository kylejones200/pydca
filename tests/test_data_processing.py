"""
Unit tests for data processing utilities.
"""

import numpy as np
import pandas as pd
import pytest

from decline_curve.utils import data_processing as dp


class TestRemoveNanAndZeroes:
    """Test remove_nan_and_zeroes function."""

    def test_remove_zeros(self):
        """Test removing zero values."""
        df = pd.DataFrame(
            {"oil": [100, 0, 200, 0, 300], "gas": [150, 100, 250, 100, 350]}
        )

        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 3
        assert all(result["oil"] > 0)

    def test_remove_nans(self):
        """Test removing NaN values."""
        df = pd.DataFrame(
            {"oil": [100, np.nan, 200, np.nan, 300], "gas": [150, 100, 250, 100, 350]}
        )

        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 3
        assert result["oil"].notna().all()

    def test_remove_both(self):
        """Test removing both zeros and NaNs."""
        df = pd.DataFrame(
            {
                "oil": [100, 0, 200, np.nan, 300, 0],
                "gas": [150, 100, 250, 100, 350, 200],
            }
        )

        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 3
        assert all(result["oil"] > 0)
        assert result["oil"].notna().all()

    def test_preserve_other_columns(self):
        """Test that other columns are preserved."""
        df = pd.DataFrame(
            {"oil": [100, 0, 200], "gas": [150, 100, 250], "water": [50, 75, 100]}
        )

        result = dp.remove_nan_and_zeroes(df, "oil")
        assert "gas" in result.columns
        assert "water" in result.columns
        assert len(result.columns) == 3


class TestCalculateDaysOnline:
    """Test calculate_days_online function."""

    def test_basic_calculation(self):
        """Test basic days online calculation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "online_date": pd.Timestamp("2020-01-01"),
            }
        )

        days = dp.calculate_days_online(df, "date", "online_date")

        assert days.iloc[0] == 0  # First day
        assert days.iloc[-1] > days.iloc[0]  # Increasing
        assert all(days >= 0)  # All non-negative

    def test_with_different_online_dates(self):
        """Test with different online dates per row."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        df = pd.DataFrame(
            {"date": dates, "online_date": [pd.Timestamp("2020-01-01")] * 5}
        )

        days = dp.calculate_days_online(df, "date", "online_date")

        # Days should be increasing
        assert all(days.diff().dropna() > 0)


class TestGetGroupedMinMax:
    """Test get_grouped_min_max function."""

    def test_get_min_by_group(self):
        """Test getting minimum value by group."""
        df = pd.DataFrame(
            {
                "well_id": ["A", "A", "A", "B", "B", "B"],
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "production": [100, 90, 80, 200, 190, 180],
            }
        )

        result = dp.get_grouped_min_max(df, "well_id", "production", "min")

        assert result.iloc[0] == 80  # Min for well A
        assert result.iloc[3] == 180  # Min for well B

    def test_get_max_by_group(self):
        """Test getting maximum value by group."""
        df = pd.DataFrame(
            {
                "well_id": ["A", "A", "A", "B", "B", "B"],
                "production": [100, 90, 80, 200, 190, 180],
            }
        )

        result = dp.get_grouped_min_max(df, "well_id", "production", "max")

        assert result.iloc[0] == 100  # Max for well A
        assert result.iloc[3] == 200  # Max for well B

    def test_invalid_calc_type(self):
        """Test that invalid calc_type raises error."""
        df = pd.DataFrame({"well_id": ["A"], "production": [100]})

        with pytest.raises(ValueError, match="calc_type must be"):
            dp.get_grouped_min_max(df, "well_id", "production", "invalid")


class TestGetMaxInitialProduction:
    """Test get_max_initial_production function."""

    def test_basic_max_initial(self):
        """Test getting max from first N months."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "oil": [100, 150, 120, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            }
        )

        qi = dp.get_max_initial_production(df, 3, "oil", "date")

        assert qi == 150  # Max of first 3 months

    def test_with_ramp_up(self):
        """Test with production ramp-up."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "oil": [50, 100, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60],
            }
        )

        qi = dp.get_max_initial_production(df, 4, "oil", "date")

        assert qi == 150  # Max of first 4 months (handles ramp-up)


class TestCalculateCumulativeProduction:
    """Test calculate_cumulative_production function."""

    def test_basic_cumulative(self):
        """Test basic cumulative calculation."""
        df = pd.DataFrame({"oil": [100, 100, 100, 100]})

        cum = dp.calculate_cumulative_production(df, "oil")

        assert cum.iloc[0] == 100
        assert cum.iloc[1] == 200
        assert cum.iloc[2] == 300
        assert cum.iloc[3] == 400

    def test_cumulative_by_group(self):
        """Test cumulative with grouping."""
        df = pd.DataFrame(
            {"well_id": ["A", "A", "B", "B"], "oil": [100, 100, 200, 200]}
        )

        cum = dp.calculate_cumulative_production(df, "oil", "well_id")

        assert cum.iloc[0] == 100  # Well A first
        assert cum.iloc[1] == 200  # Well A second
        assert cum.iloc[2] == 200  # Well B first (resets)
        assert cum.iloc[3] == 400  # Well B second


class TestNormalizeToDailyRate:
    """Test normalize_production_to_daily function."""

    def test_basic_normalization(self):
        """Test basic daily rate calculation."""
        df = pd.DataFrame(
            {"Oil": [3000, 2800, 3100], "Days": [30, 28, 31]}  # Monthly totals
        )

        daily = dp.normalize_production_to_daily(df, "Oil", "Days")

        assert daily.iloc[0] == 100  # 3000/30
        assert daily.iloc[1] == 100  # 2800/28
        assert daily.iloc[2] == 100  # 3100/31


class TestCalculateWaterCut:
    """Test calculate_water_cut function."""

    def test_basic_water_cut(self):
        """Test basic water cut calculation."""
        df = pd.DataFrame({"Oil": [100, 100, 100], "Wtr": [100, 200, 300]})

        wc = dp.calculate_water_cut(df, "Oil", "Wtr")

        assert wc.iloc[0] == 50.0  # 100/(100+100)*100
        assert wc.iloc[1] == pytest.approx(66.67, rel=0.01)
        assert wc.iloc[2] == 75.0

    def test_zero_liquid(self):
        """Test water cut with zero liquid."""
        df = pd.DataFrame({"Oil": [0, 100], "Wtr": [0, 100]})

        wc = dp.calculate_water_cut(df, "Oil", "Wtr")

        assert wc.iloc[0] == 0  # Should handle 0/0 gracefully
        assert wc.iloc[1] == 50.0


class TestCalculateGOR:
    """Test calculate_gor function."""

    def test_basic_gor(self):
        """Test basic GOR calculation."""
        df = pd.DataFrame({"Gas": [1500, 1400, 1300], "Oil": [1000, 1000, 1000]})

        gor = dp.calculate_gor(df, "Gas", "Oil")

        assert gor.iloc[0] == 1.5
        assert gor.iloc[1] == 1.4
        assert gor.iloc[2] == 1.3

    def test_zero_oil(self):
        """Test GOR with zero oil."""
        df = pd.DataFrame({"Gas": [1500, 1400], "Oil": [0, 1000]})

        gor = dp.calculate_gor(df, "Gas", "Oil")

        # Should handle division by zero
        assert pd.isna(gor.iloc[0]) or np.isinf(gor.iloc[0])
        assert gor.iloc[1] == 1.4


class TestPrepareWellDataForDCA:
    """Test prepare_well_data_for_dca function."""

    def test_basic_preparation(self):
        """Test basic data preparation."""
        df = pd.DataFrame(
            {
                "well_id": ["A", "A", "A", "B", "B"],
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "oil": [100, 90, 80, 200, 190],
            }
        )

        series = dp.prepare_well_data_for_dca(df, "A", "well_id", "date", "oil")

        assert isinstance(series, pd.Series)
        assert len(series) == 3
        assert all(series > 0)
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_remove_zeros_option(self):
        """Test remove_zeros parameter."""
        df = pd.DataFrame(
            {
                "well_id": ["A", "A", "A"],
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "oil": [100, 0, 80],
            }
        )

        # With remove_zeros=True
        series_filtered = dp.prepare_well_data_for_dca(
            df, "A", "well_id", "date", "oil", remove_zeros=True
        )
        assert len(series_filtered) == 2

        # With remove_zeros=False
        series_all = dp.prepare_well_data_for_dca(
            df, "A", "well_id", "date", "oil", remove_zeros=False
        )
        assert len(series_all) == 3

    def test_date_conversion(self):
        """Test automatic date conversion."""
        df = pd.DataFrame(
            {
                "well_id": ["A", "A", "A"],
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "oil": [100, 90, 80],
            }
        )

        series = dp.prepare_well_data_for_dca(df, "A", "well_id", "date", "oil")

        assert isinstance(series.index, pd.DatetimeIndex)


class TestDetectProductionAnomalies:
    """Test detect_production_anomalies function."""

    def test_no_anomalies(self):
        """Test with smooth declining production."""
        series = pd.Series(np.linspace(1000, 500, 20))
        anomalies = dp.detect_production_anomalies(series, threshold_std=3.0)

        # Smooth decline should have few/no anomalies
        assert anomalies.sum() <= 2  # Allow for edge effects

    def test_with_outlier(self):
        """Test detection of clear outlier."""
        data = [100, 90, 80, 500, 70, 60, 50]  # 500 is outlier
        series = pd.Series(data)

        anomalies = dp.detect_production_anomalies(series, threshold_std=2.0)

        # Should detect the outlier
        assert anomalies.sum() >= 1
        assert anomalies.iloc[3]  # The outlier at index 3

    def test_threshold_sensitivity(self):
        """Test that lower threshold detects more anomalies."""
        series = pd.Series([100, 90, 80, 120, 70, 60, 50])

        anomalies_strict = dp.detect_production_anomalies(series, threshold_std=3.0)
        anomalies_loose = dp.detect_production_anomalies(series, threshold_std=1.0)

        # Looser threshold should detect more
        assert anomalies_loose.sum() >= anomalies_strict.sum()


class TestFilterWellsByDateRange:
    """Test filter_wells_by_date_range function."""

    def test_basic_filtering(self):
        """Test basic date range filtering."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "oil": range(12),
            }
        )

        filtered = dp.filter_wells_by_date_range(df, "date", "2020-03-01", "2020-06-01")

        assert len(filtered) == 4  # March, April, May, June
        assert filtered["date"].min() >= pd.Timestamp("2020-03-01")
        assert filtered["date"].max() <= pd.Timestamp("2020-06-01")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"oil": []})
        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 0

    def test_all_zeros(self):
        """Test DataFrame with all zeros."""
        df = pd.DataFrame({"oil": [0, 0, 0]})
        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 0

    def test_all_nans(self):
        """Test DataFrame with all NaNs."""
        df = pd.DataFrame({"oil": [np.nan, np.nan, np.nan]})
        result = dp.remove_nan_and_zeroes(df, "oil")
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
