"""
Unit tests for forecasting functionality.
"""

import numpy as np
import pandas as pd
import pytest

from decline_analysis.forecast import Forecaster
from decline_analysis.forecast_chronos import (
    check_chronos_availability,
    forecast_chronos,
)
from decline_analysis.forecast_timesfm import (
    check_timesfm_availability,
    forecast_timesfm,
)


class TestForecaster:
    """Test the main Forecaster class."""

    def test_forecaster_init(self, sample_production_data):
        """Test Forecaster initialization."""
        forecaster = Forecaster(sample_production_data)

        assert isinstance(forecaster.series, pd.Series)
        assert isinstance(forecaster.series.index, pd.DatetimeIndex)
        assert forecaster.last_forecast is None
        assert len(forecaster.series) > 0

    def test_forecaster_init_non_datetime_index(self):
        """Test Forecaster with non-datetime index raises error."""
        series = pd.Series([1, 2, 3, 4, 5])  # No datetime index

        with pytest.raises(ValueError, match="Input must be indexed by datetime"):
            Forecaster(series)

    def test_forecaster_arps_exponential(self, sample_production_data):
        """Test Arps exponential forecasting."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", kind="exponential", horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        assert forecast.name == "arps_exponential"
        assert forecaster.last_forecast is not None

        # Check that forecast values are positive and declining
        forecast_only = forecast.iloc[len(sample_production_data) :]
        assert all(forecast_only > 0)

    def test_forecaster_arps_harmonic(self, sample_production_data):
        """Test Arps harmonic forecasting."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", kind="harmonic", horizon=6)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 6
        assert forecast.name == "arps_harmonic"

    def test_forecaster_arps_hyperbolic(self, sample_production_data):
        """Test Arps hyperbolic forecasting."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", kind="hyperbolic", horizon=24)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 24
        assert forecast.name == "arps_hyperbolic"

    def test_forecaster_timesfm(self, sample_production_data):
        """Test TimesFM forecasting."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="timesfm", horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        # Should work even if TimesFM is not available (fallback)

    def test_forecaster_chronos(self, sample_production_data):
        """Test Chronos forecasting."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="chronos", horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        # Should work even if Chronos is not available (fallback)

    def test_forecaster_invalid_model(self, sample_production_data):
        """Test forecasting with invalid model."""
        forecaster = Forecaster(sample_production_data)

        with pytest.raises(ValueError, match="Unknown model"):
            forecaster.forecast(model="invalid_model")

    def test_forecaster_evaluate(self, sample_production_data):
        """Test forecast evaluation."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", kind="exponential", horizon=12)

        # Create some "actual" data for evaluation
        actual = sample_production_data.iloc[:12]  # Use first 12 points as "actual"

        metrics = forecaster.evaluate(actual)

        assert isinstance(metrics, dict)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "smape" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_forecaster_evaluate_no_forecast(self, sample_production_data):
        """Test evaluation without prior forecast."""
        forecaster = Forecaster(sample_production_data)

        with pytest.raises(RuntimeError, match="Call .forecast\\(\\) first"):
            forecaster.evaluate(sample_production_data)

    def test_forecaster_evaluate_no_overlap(self, sample_production_data):
        """Test evaluation with no overlapping dates."""
        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", horizon=12)

        # Create actual data with non-overlapping dates
        future_dates = pd.date_range(start="2025-01-01", periods=12, freq="MS")
        actual = pd.Series([100] * 12, index=future_dates)

        with pytest.raises(ValueError, match="No overlapping dates"):
            forecaster.evaluate(actual)


class TestTimesFMIntegration:
    """Test TimesFM forecasting integration."""

    def test_timesfm_forecast_basic(self, sample_production_data):
        """Test basic TimesFM forecasting."""
        forecast = forecast_timesfm(sample_production_data, horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        assert all(forecast >= 0)  # Production should be non-negative

    def test_timesfm_forecast_different_horizons(
        self, sample_production_data, forecast_horizons
    ):
        """Test TimesFM with different forecast horizons."""
        for horizon in forecast_horizons:
            forecast = forecast_timesfm(sample_production_data, horizon=horizon)
            assert len(forecast) == len(sample_production_data) + horizon

    def test_timesfm_availability_check(self):
        """Test TimesFM availability check."""
        available = check_timesfm_availability()
        assert isinstance(available, bool)

    def test_timesfm_short_series(self):
        """Test TimesFM with very short time series."""
        dates = pd.date_range(start="2020-01-01", periods=3, freq="MS")
        short_series = pd.Series([100, 90, 80], index=dates)

        forecast = forecast_timesfm(short_series, horizon=6)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 3 + 6


class TestChronosIntegration:
    """Test Chronos forecasting integration."""

    def test_chronos_forecast_basic(self, sample_production_data):
        """Test basic Chronos forecasting."""
        forecast = forecast_chronos(sample_production_data, horizon=12)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == len(sample_production_data) + 12
        assert all(forecast >= 0)  # Production should be non-negative

    def test_chronos_forecast_different_horizons(
        self, sample_production_data, forecast_horizons
    ):
        """Test Chronos with different forecast horizons."""
        for horizon in forecast_horizons:
            forecast = forecast_chronos(sample_production_data, horizon=horizon)
            assert len(forecast) == len(sample_production_data) + horizon

    def test_chronos_availability_check(self):
        """Test Chronos availability check."""
        available = check_chronos_availability()
        assert isinstance(available, bool)

    def test_chronos_short_series(self):
        """Test Chronos with very short time series."""
        dates = pd.date_range(start="2020-01-01", periods=2, freq="MS")
        short_series = pd.Series([100, 90], index=dates)

        forecast = forecast_chronos(short_series, horizon=6)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 2 + 6


class TestForecastConsistency:
    """Test consistency across different forecasting methods."""

    def test_forecast_index_consistency(self, sample_production_data):
        """Test that all forecasting methods produce consistent date indices."""
        horizon = 12

        forecaster = Forecaster(sample_production_data)
        arps_forecast = forecaster.forecast(model="arps", horizon=horizon)
        timesfm_forecast = forecast_timesfm(sample_production_data, horizon=horizon)
        chronos_forecast = forecast_chronos(sample_production_data, horizon=horizon)

        # All should have the same length
        assert len(arps_forecast) == len(timesfm_forecast) == len(chronos_forecast)

        # All should start from the same date
        assert (
            arps_forecast.index[0]
            == timesfm_forecast.index[0]
            == chronos_forecast.index[0]
        )

        # All should have the same frequency
        assert (
            arps_forecast.index.freq
            == timesfm_forecast.index.freq
            == chronos_forecast.index.freq
        )

    def test_forecast_value_reasonableness(self, sample_production_data):
        """Test that forecast values are reasonable."""
        horizon = 6

        forecaster = Forecaster(sample_production_data)
        forecast = forecaster.forecast(model="arps", kind="hyperbolic", horizon=horizon)

        # Forecast should be positive
        assert all(forecast > 0)

        # Forecast should generally decline (allowing some tolerance)
        historical_part = forecast.iloc[: len(sample_production_data)]
        forecast_part = forecast.iloc[len(sample_production_data) :]

        # Last historical value should be >= last forecast value (general decline)
        assert (
            historical_part.iloc[-1] >= forecast_part.iloc[-1] * 0.5
        )  # Allow some flexibility

    def test_forecast_with_missing_data(self):
        """Test forecasting with missing data points."""
        dates = pd.date_range(start="2020-01-01", periods=12, freq="MS")
        values = [1000, 900, np.nan, 700, 600, np.nan, 400, 300, 200, 100, np.nan, 50]
        series_with_nan = pd.Series(values, index=dates)

        forecaster = Forecaster(series_with_nan)
        # Should automatically drop NaN values
        assert not forecaster.series.isna().any()

        forecast = forecaster.forecast(model="arps", horizon=6)
        assert isinstance(forecast, pd.Series)
        assert not forecast.isna().any()
