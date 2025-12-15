"""Tests for Chronos forecasting."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.forecast_chronos import (
    _fallback_chronos_forecast,
    _generate_chronos_forecast,
    forecast_chronos,
)


class TestChronosForecast:
    """Test Chronos forecasting functions."""

    def test_forecast_chronos_basic(self):
        """Test basic Chronos forecast."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * np.exp(-0.05 * np.arange(24))
        series = pd.Series(production, index=dates, name="oil")

        forecast = forecast_chronos(series, horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 36  # 24 historical + 12 forecast
        assert all(forecast >= 0)  # Non-negative

    def test_forecast_chronos_short_series(self):
        """Test Chronos forecast with short series."""
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        production = [1000, 950, 900, 850, 800, 750]
        series = pd.Series(production, index=dates, name="oil")

        forecast = forecast_chronos(series, horizon=6)

        assert len(forecast) == 12  # 6 historical + 6 forecast
        assert all(forecast >= 0)

    def test_generate_chronos_forecast(self):
        """Test internal Chronos forecast generation."""
        values = np.array([1000, 950, 900, 850, 800, 750], dtype=np.float32)
        device = "cpu"

        forecast = _generate_chronos_forecast(values, horizon=6, device=device)

        assert len(forecast) == 6
        assert all(forecast >= 0)
        assert isinstance(forecast, np.ndarray)

    def test_generate_chronos_forecast_single_value(self):
        """Test Chronos forecast with single value."""
        values = np.array([1000], dtype=np.float32)
        device = "cpu"

        forecast = _generate_chronos_forecast(values, horizon=6, device=device)

        assert len(forecast) == 6
        assert all(forecast >= 0)

    def test_fallback_chronos_forecast(self):
        """Test fallback Chronos forecast method."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * np.exp(-0.05 * np.arange(24))
        series = pd.Series(production, index=dates, name="oil")

        forecast = _fallback_chronos_forecast(series, horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 36
        assert all(forecast >= 0)

    def test_forecast_chronos_with_nan(self):
        """Test Chronos forecast handling NaN values."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = 1000 * np.exp(-0.05 * np.arange(24))
        production[5] = np.nan  # Add NaN
        series = pd.Series(production, index=dates, name="oil")

        # Should handle NaN gracefully (fallback should interpolate or skip)
        forecast = forecast_chronos(series, horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) >= 24

    def test_forecast_chronos_index_preservation(self):
        """Test that forecast preserves date index."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = [1000 - i * 30 for i in range(24)]
        series = pd.Series(production, index=dates, name="oil")

        forecast = forecast_chronos(series, horizon=12)

        assert isinstance(forecast.index, pd.DatetimeIndex)
        assert forecast.index[0] == dates[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
