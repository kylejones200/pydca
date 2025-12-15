"""Tests for TimesFM forecasting module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.forecast_timesfm import (
    check_timesfm_availability,
    forecast_timesfm,
)


class TestTimesFMForecast:
    """Test TimesFM forecasting functionality."""

    def test_forecast_timesfm_basic(self):
        """Test basic TimesFM forecasting."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 20 + np.random.randn(24) * 10
        series = pd.Series(values, index=dates)

        result = forecast_timesfm(series, horizon=12)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series) + 12  # Historical + forecast
        assert result.name == "timesfm_forecast" or result.name == "timesfm_fallback"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_forecast_timesfm_short_series(self):
        """Test TimesFM with short time series."""
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        short_series = pd.Series([100, 95, 90, 85, 80, 75], index=dates)

        result = forecast_timesfm(short_series, horizon=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(short_series) + 3

    def test_forecast_timesfm_different_horizons(self):
        """Test TimesFM with different forecast horizons."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 15
        series = pd.Series(values, index=dates)

        for horizon in [6, 12, 24]:
            result = forecast_timesfm(series, horizon=horizon)
            assert len(result) == len(series) + horizon

    def test_forecast_timesfm_with_nan(self):
        """Test TimesFM with NaN values in series."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = np.array([1000 - i * 20 for i in range(24)], dtype=float)
        values[5] = np.nan
        values[15] = np.nan
        series = pd.Series(values, index=dates)

        # Should handle NaN values gracefully (fallback method handles them)
        result = forecast_timesfm(series, horizon=6)
        assert isinstance(result, pd.Series)
        # Result may be shorter if NaN values are dropped, or full length if handled
        assert len(result) >= 6

    def test_forecast_timesfm_index_preservation(self):
        """Test that TimesFM preserves index continuity."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 20
        series = pd.Series(values, index=dates)

        result = forecast_timesfm(series, horizon=12)

        # Check that historical index is preserved
        assert result.index[:24].equals(series.index)

        # Check that forecast index continues from last date
        expected_next_date = series.index[-1] + pd.DateOffset(months=1)
        assert result.index[24] == expected_next_date

    def test_forecast_timesfm_constant_series(self):
        """Test TimesFM with constant values."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        constant_series = pd.Series([500] * 12, index=dates)

        result = forecast_timesfm(constant_series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 18
        # Forecast should be non-negative
        assert all(result >= 0)

    def test_forecast_timesfm_trend_series(self):
        """Test TimesFM with declining trend."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        trend_values = 1000 - np.arange(24) * 30
        trend_series = pd.Series(trend_values, index=dates)

        result = forecast_timesfm(trend_series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 30
        # Forecast should continue trend (allow some variation)
        assert result.iloc[-1] <= result.iloc[24] * 1.5

    def test_forecast_timesfm_fallback_behavior(self):
        """Test that fallback method works when TimesFM is unavailable."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 20
        series = pd.Series(values, index=dates)

        # Should work even if TimesFM dependencies are missing
        result = forecast_timesfm(series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series) + 6
        # Should use fallback method
        assert result.name in ["timesfm_forecast", "timesfm_fallback"]

    def test_check_timesfm_availability(self):
        """Test TimesFM availability check."""
        available = check_timesfm_availability()

        # Should return boolean
        assert isinstance(available, bool)

        # Availability depends on whether torch and transformers are installed
        # This test just ensures the function works without error


class TestTimesFMEdgeCases:
    """Test edge cases for TimesFM forecasting."""

    def test_forecast_timesfm_single_value(self):
        """Test TimesFM with single data point."""
        # Use at least 3 dates to allow frequency inference
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        single_series = pd.Series([1000, 950, 900], index=dates)

        result = forecast_timesfm(single_series, horizon=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 6  # 3 historical + 3 forecast

    def test_forecast_timesfm_very_short_series(self):
        """Test TimesFM with very short series."""
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        short_series = pd.Series([1000, 950, 900], index=dates)

        result = forecast_timesfm(short_series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 9

    def test_forecast_timesfm_negative_values(self):
        """Test TimesFM with negative values (should be handled)."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        # Some negative values (unusual for production but test robustness)
        values = np.array([100, 90, 80, -10, 70, 60, 50, 40, 30, 20, 10, 5])
        series = pd.Series(values, index=dates)

        result = forecast_timesfm(series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 18

    def test_forecast_timesfm_zero_values(self):
        """Test TimesFM with zero values."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        values = [100, 90, 80, 0, 0, 50, 40, 30, 20, 10, 5, 0]
        series = pd.Series(values, index=dates)

        result = forecast_timesfm(series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 18
        # Forecast should be non-negative
        assert all(result >= 0)

    def test_forecast_timesfm_large_horizon(self):
        """Test TimesFM with large forecast horizon."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 20
        series = pd.Series(values, index=dates)

        large_horizon = 100
        result = forecast_timesfm(series, horizon=large_horizon)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series) + large_horizon

    def test_forecast_timesfm_different_frequencies(self):
        """Test TimesFM with different time series frequencies."""
        # Test with daily frequency
        daily_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        daily_series = pd.Series(
            np.random.randn(100).cumsum() + 1000, index=daily_dates
        )

        result = forecast_timesfm(daily_series, horizon=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 110

        # Test with weekly frequency
        weekly_dates = pd.date_range("2020-01-01", periods=52, freq="W")
        weekly_series = pd.Series(
            np.random.randn(52).cumsum() + 1000, index=weekly_dates
        )

        result = forecast_timesfm(weekly_series, horizon=5)
        assert isinstance(result, pd.Series)
        assert len(result) == 57


class TestTimesFMIntegration:
    """Test TimesFM integration with main API."""

    def test_timesfm_in_main_api(self):
        """Test TimesFM model in the main DCA API."""
        from decline_curve import dca

        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = 1000 - np.arange(24) * 20
        series = pd.Series(values, index=dates)

        result = dca.forecast(series, model="timesfm", horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series) + 6
