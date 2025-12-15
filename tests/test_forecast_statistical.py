"""Tests for statistical forecasting methods."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.forecast_statistical import (
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)


class TestSimpleExponentialSmoothing:
    """Test simple exponential smoothing forecast."""

    def test_basic_forecast(self):
        """Test basic exponential smoothing."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast = simple_exponential_smoothing(series, alpha=0.3, horizon=6)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 6
        assert all(forecast >= 0)

    def test_different_alpha(self):
        """Test with different alpha values."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast_low = simple_exponential_smoothing(series, alpha=0.1, horizon=6)
        forecast_high = simple_exponential_smoothing(series, alpha=0.9, horizon=6)

        assert len(forecast_low) == 6
        assert len(forecast_high) == 6

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            simple_exponential_smoothing(series, horizon=6)


class TestMovingAverageForecast:
    """Test moving average forecast."""

    def test_basic_forecast(self):
        """Test basic moving average."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast = moving_average_forecast(series, window=6, horizon=6)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 6
        assert all(forecast >= 0)

    def test_different_windows(self):
        """Test with different window sizes."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast_short = moving_average_forecast(series, window=3, horizon=6)
        forecast_long = moving_average_forecast(series, window=12, horizon=6)

        assert len(forecast_short) == 6
        assert len(forecast_long) == 6


class TestLinearTrendForecast:
    """Test linear trend forecast."""

    def test_basic_forecast(self):
        """Test basic linear trend."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 - 10 * np.arange(24)  # Linear decline
        series = pd.Series(production, index=dates)

        forecast = linear_trend_forecast(series, horizon=6)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 6
        assert all(forecast >= 0)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        series = pd.Series(
            [1000], index=pd.date_range("2020-01-01", periods=1, freq="MS")
        )
        with pytest.raises(ValueError):
            linear_trend_forecast(series, horizon=6)


class TestHoltWintersForecast:
    """Test Holt-Winters forecast."""

    def test_basic_forecast(self):
        """Test basic Holt-Winters (if statsmodels available)."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 + 50 * np.sin(np.arange(24) * 2 * np.pi / 12)  # Seasonal
        series = pd.Series(production, index=dates)

        forecast = holt_winters_forecast(series, horizon=6)

        if forecast is not None:
            assert isinstance(forecast, pd.Series)
            assert len(forecast) == 6
            assert all(forecast >= 0)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        production = 1000 * (0.95 ** np.arange(6))
        series = pd.Series(production, index=dates)

        forecast = holt_winters_forecast(series, horizon=6)
        assert forecast is None


class TestConfidenceIntervals:
    """Test confidence interval calculation."""

    def test_basic_intervals(self):
        """Test basic confidence interval calculation."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast = simple_exponential_smoothing(series, horizon=6)
        lower, upper = calculate_confidence_intervals(series, forecast, method="naive")

        assert lower is not None
        assert upper is not None
        assert len(lower) == len(forecast)
        assert len(upper) == len(forecast)
        assert all(lower <= upper)
        assert all(lower >= 0)  # Non-negative

    def test_insufficient_data(self):
        """Test with insufficient data."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        production = 1000 * (0.95 ** np.arange(5))
        series = pd.Series(production, index=dates)

        forecast = simple_exponential_smoothing(series, horizon=6)
        lower, upper = calculate_confidence_intervals(series, forecast)

        assert lower is None
        assert upper is None

    def test_different_confidence_levels(self):
        """Test with different confidence levels."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * (0.95 ** np.arange(24))
        series = pd.Series(production, index=dates)

        forecast = simple_exponential_smoothing(series, horizon=6)

        lower_90, upper_90 = calculate_confidence_intervals(
            series, forecast, confidence=0.90
        )
        lower_95, upper_95 = calculate_confidence_intervals(
            series, forecast, confidence=0.95
        )

        if lower_90 is not None and lower_95 is not None:
            # 95% intervals should be wider than 90%
            assert (upper_95 - lower_95).mean() > (upper_90 - lower_90).mean()
