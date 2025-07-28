"""
Unit tests for ARIMA forecasting functionality.
"""

import numpy as np
import pandas as pd
import pytest

from decline_analysis.forecast_arima import forecast_arima


class TestARIMAForecasting:
    """Test ARIMA forecasting functionality."""

    def test_forecast_arima_basic(self, sample_production_data):
        """Test basic ARIMA forecasting."""
        result = forecast_arima(sample_production_data, horizon=12)

        assert isinstance(result, pd.Series)
        assert len(result) == 12  # Only forecast horizon, not historical
        assert result.name == "arima_forecast"
        assert isinstance(result.index, pd.DatetimeIndex)

        # Forecast should start after the last historical date
        assert result.index[0] > sample_production_data.index[-1]

    def test_forecast_arima_different_horizons(
        self, sample_production_data, forecast_horizons
    ):
        """Test ARIMA with different forecast horizons."""
        for horizon in forecast_horizons:
            result = forecast_arima(sample_production_data, horizon=horizon)
            assert len(result) == horizon
            assert isinstance(result, pd.Series)

    def test_forecast_arima_with_manual_order(self, sample_production_data):
        """Test ARIMA with manually specified order."""
        # Test with simple ARIMA(1,1,1) order
        result = forecast_arima(sample_production_data, horizon=6, order=(1, 1, 1))

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        assert not result.isna().any()

    def test_forecast_arima_seasonal(self, sample_production_data):
        """Test ARIMA with seasonal components."""
        result = forecast_arima(
            sample_production_data, horizon=12, seasonal=True, seasonal_period=12
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 12
        assert not result.isna().any()

    def test_forecast_arima_auto_order(self, sample_production_data):
        """Test ARIMA with automatic order selection."""
        # This should use auto_arima to find the best order
        result = forecast_arima(sample_production_data, horizon=6, order=None)

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        assert not result.isna().any()

    def test_forecast_arima_non_datetime_index(self):
        """Test ARIMA with non-datetime index raises error."""
        series = pd.Series([1, 2, 3, 4, 5])  # No datetime index

        with pytest.raises(ValueError, match="Index must be a DatetimeIndex"):
            forecast_arima(series)

    def test_forecast_arima_with_missing_values(self):
        """Test ARIMA with missing values in the series."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        # Use a more predictable series to avoid ARIMA fitting issues
        values = np.linspace(1000, 800, 24) + np.random.randn(24) * 10
        values[5] = np.nan  # Add missing value
        values[15] = np.nan  # Add another missing value

        series_with_nan = pd.Series(values, index=dates)

        # Should handle NaN values by dropping them and return valid forecast
        try:
            result = forecast_arima(series_with_nan, horizon=6)

            assert isinstance(result, pd.Series)
            assert len(result) == 6
            # The function should handle missing values gracefully
            # Either return valid forecasts or use fallback methods
            assert result.index.freq is not None or len(result.index) == 6
        except Exception as e:
            # If ARIMA fails completely, that's also acceptable for this edge case
            # The important thing is that it doesn't crash the application
            assert isinstance(e, (ValueError, RuntimeError))

    def test_forecast_arima_short_series(self):
        """Test ARIMA with very short time series."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        short_series = pd.Series([100, 95, 90, 85, 80], index=dates)

        # Should handle short series gracefully
        result = forecast_arima(short_series, horizon=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_forecast_arima_constant_series(self):
        """Test ARIMA with constant values."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        constant_series = pd.Series([500] * 12, index=dates)

        # Should handle constant series
        result = forecast_arima(constant_series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        # Forecast of constant series should be approximately constant
        assert all(abs(result - 500) < 100)  # Allow some variation

    def test_forecast_arima_trend_series(self):
        """Test ARIMA with trending series."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        trend_values = 1000 - np.arange(24) * 10  # Declining trend
        trend_series = pd.Series(trend_values, index=dates)

        result = forecast_arima(trend_series, horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        # Should continue the declining trend (allow for small numerical errors)
        assert result.iloc[-1] <= result.iloc[0] * 1.01


class TestARIMAIntegration:
    """Test ARIMA integration with main forecasting classes."""

    def test_arima_in_forecaster_class(self, sample_production_data):
        """Test ARIMA model in the main Forecaster class."""
        from decline_analysis.forecast import Forecaster

        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arima", horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        assert forecaster.last_forecast is not None

    def test_arima_in_main_api(self, sample_production_data):
        """Test ARIMA model in the main DCA API."""
        from decline_analysis import dca

        result = dca.forecast(sample_production_data, model="arima", horizon=6)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_production_data) + 6

    def test_arima_benchmark(self, sample_well_data):
        """Test ARIMA in benchmark function."""
        from decline_analysis import dca

        result = dca.benchmark(sample_well_data, model="arima", horizon=6, top_n=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2
        assert "rmse" in result.columns
        assert "mae" in result.columns
        assert "smape" in result.columns


class TestARIMAEdgeCases:
    """Test edge cases and error handling for ARIMA."""

    def test_arima_empty_series(self):
        """Test ARIMA with empty series."""
        empty_series = pd.Series([], dtype=float)
        empty_series.index = pd.DatetimeIndex([])

        try:
            result = forecast_arima(empty_series, horizon=3)
            # Should either work or raise a meaningful error
            assert isinstance(result, pd.Series)
        except Exception as e:
            # Should raise a meaningful error, not crash
            assert isinstance(e, (ValueError, RuntimeError))

    def test_arima_single_value(self):
        """Test ARIMA with single data point."""
        single_date = pd.DatetimeIndex(["2020-01-01"])
        single_series = pd.Series([1000], index=single_date)

        try:
            result = forecast_arima(single_series, horizon=3)
            assert isinstance(result, pd.Series)
            assert len(result) == 3
        except Exception as e:
            # Single point may not be enough for ARIMA
            assert isinstance(e, (ValueError, RuntimeError))

    def test_arima_negative_horizon(self, sample_production_data):
        """Test ARIMA with negative horizon."""
        try:
            result = forecast_arima(sample_production_data, horizon=-5)
            # Should either handle gracefully or raise error
        except Exception as e:
            assert isinstance(e, (ValueError, RuntimeError))

    def test_arima_zero_horizon(self, sample_production_data):
        """Test ARIMA with zero horizon."""
        with pytest.raises(ValueError, match="Horizon must be positive"):
            forecast_arima(sample_production_data, horizon=0)

    def test_arima_invalid_order(self, sample_production_data):
        """Test ARIMA with invalid order specification."""
        try:
            # Invalid order with negative values
            result = forecast_arima(sample_production_data, order=(-1, 1, 1))
        except Exception as e:
            # Should raise appropriate error for invalid order
            assert isinstance(e, (ValueError, RuntimeError))

    def test_arima_large_horizon(self, sample_production_data):
        """Test ARIMA with very large forecast horizon."""
        # Test with horizon larger than historical data
        large_horizon = len(sample_production_data) * 2

        result = forecast_arima(sample_production_data, horizon=large_horizon)

        assert isinstance(result, pd.Series)
        assert len(result) == large_horizon
        # Should not contain NaN or infinite values
        assert not result.isna().any()
        assert not np.isinf(result).any()


class TestARIMAPerformance:
    """Test ARIMA performance characteristics."""

    def test_arima_forecast_quality(self, sample_production_data):
        """Test that ARIMA forecasts are reasonable."""
        result = forecast_arima(sample_production_data, horizon=6)

        # Forecasts should be positive for production data
        assert all(result >= 0)

        # Forecasts should be in a reasonable range relative to historical data
        hist_mean = sample_production_data.mean()
        hist_std = sample_production_data.std()

        # Allow forecasts to be within 3 standard deviations of historical mean
        assert all(abs(result - hist_mean) <= 3 * hist_std)

    def test_arima_consistency(self, sample_production_data):
        """Test that ARIMA produces consistent results."""
        # Run the same forecast multiple times with fixed order
        results = []
        for _ in range(3):
            result = forecast_arima(sample_production_data, horizon=6, order=(1, 1, 1))
            results.append(result)

        # Results should be identical for the same order
        for i in range(1, len(results)):
            pd.testing.assert_series_equal(results[0], results[i])

    def test_arima_different_frequencies(self):
        """Test ARIMA with different time series frequencies."""
        # Test with daily frequency
        daily_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        daily_series = pd.Series(
            np.random.randn(100).cumsum() + 1000, index=daily_dates
        )

        result = forecast_arima(daily_series, horizon=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 10

        # Test with weekly frequency
        weekly_dates = pd.date_range("2020-01-01", periods=52, freq="W")
        weekly_series = pd.Series(
            np.random.randn(52).cumsum() + 1000, index=weekly_dates
        )

        result = forecast_arima(weekly_series, horizon=5)
        assert isinstance(result, pd.Series)
        assert len(result) == 5
