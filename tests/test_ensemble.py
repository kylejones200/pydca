"""Tests for ensemble forecasting methods."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.ensemble import (
    EnsembleForecaster,
    EnsembleWeights,
    ensemble_forecast,
)


class TestEnsembleWeights:
    """Test EnsembleWeights class."""

    def test_create_weights(self):
        """Test creating ensemble weights."""
        weights = EnsembleWeights(arps=0.5, lstm=0.3, deepar=0.2)
        assert weights.arps == 0.5
        assert weights.lstm == 0.3
        assert weights.deepar == 0.2

    def test_normalize_weights(self):
        """Test normalizing weights."""
        weights = EnsembleWeights(arps=1.0, lstm=1.0, deepar=1.0)
        normalized = weights.normalize()
        assert abs(normalized.arps + normalized.lstm + normalized.deepar - 1.0) < 1e-6


class TestEnsembleForecaster:
    """Test EnsembleForecaster class."""

    def _create_sample_data(self):
        """Create sample production data."""
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        production = 1000 * np.exp(-0.05 * np.arange(36))
        series = pd.Series(production, index=dates, name="oil")
        return series

    def test_ensemble_with_arps_only(self):
        """Test ensemble with Arps only."""
        series = self._create_sample_data()

        forecaster = EnsembleForecaster(models=["arps"], method="weighted")
        forecast = forecaster.forecast(series, horizon=12)

        assert len(forecast) == 12
        assert all(forecast >= 0)  # Non-negative

    def test_ensemble_with_multiple_models(self):
        """Test ensemble with multiple models (Arps + ARIMA)."""
        series = self._create_sample_data()

        forecaster = EnsembleForecaster(models=["arps", "arima"], method="weighted")
        forecast = forecaster.forecast(series, horizon=12)

        assert len(forecast) == 12
        assert all(forecast >= 0)

    def test_weighted_average_method(self):
        """Test weighted average ensemble method."""
        series = self._create_sample_data()

        weights = EnsembleWeights(arps=0.7, lstm=0.0, deepar=0.3)
        forecaster = EnsembleForecaster(
            models=["arps", "arima"], weights=weights, method="weighted"
        )
        forecast = forecaster.forecast(series, horizon=12)

        assert len(forecast) == 12

    def test_confidence_based_method(self):
        """Test confidence-based ensemble method."""
        series = self._create_sample_data()

        forecaster = EnsembleForecaster(models=["arps", "arima"], method="confidence")
        forecast = forecaster.forecast(series, horizon=12)

        assert len(forecast) == 12

    def test_ensemble_forecast_function(self):
        """Test ensemble_forecast convenience function."""
        series = self._create_sample_data()

        forecast = ensemble_forecast(
            series, models=["arps", "arima"], method="weighted", horizon=12
        )

        assert len(forecast) == 12
        assert all(forecast >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
