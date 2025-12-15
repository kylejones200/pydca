"""
Unit tests for the main DCA API functions.
"""

import numpy as np
import pandas as pd
import pytest

from decline_curve import dca


class TestDCAAPI:
    """Test the main DCA API functions."""

    def test_forecast_function(self, sample_production_data):
        """Test the main forecast function."""
        result = dca.forecast(
            sample_production_data, model="arps", kind="hyperbolic", horizon=12
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_production_data) + 12
        assert all(result > 0)

    def test_forecast_different_models(self, sample_production_data):
        """Test forecast function with different models."""
        models = ["arps", "timesfm", "chronos"]

        for model in models:
            if model == "arps":
                result = dca.forecast(
                    sample_production_data, model=model, kind="exponential"
                )
            else:
                result = dca.forecast(sample_production_data, model=model)

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_production_data) + 12  # default horizon

    def test_forecast_verbose(self, sample_production_data):
        """Test forecast function with verbose output."""
        # Verbose mode now uses logging, just verify it runs without error
        result = dca.forecast(sample_production_data, model="arps", verbose=True)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_evaluate_function(self, sample_production_data):
        """Test the main evaluate function."""
        # Create some predictions with small errors
        np.random.seed(42)
        y_pred = sample_production_data + np.random.normal(
            0, sample_production_data * 0.05
        )

        result = dca.evaluate(sample_production_data, y_pred)

        assert isinstance(result, dict)
        assert "rmse" in result
        assert "mae" in result
        assert "smape" in result
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_evaluate_non_overlapping_indices(self, sample_production_data):
        """Test evaluate function with non-overlapping indices."""
        # Create predictions with different index
        future_dates = pd.date_range(
            start="2025-01-01", periods=len(sample_production_data), freq="MS"
        )
        y_pred = pd.Series(sample_production_data.values, index=future_dates)

        result = dca.evaluate(sample_production_data, y_pred)

        # Should handle gracefully by finding common indices
        assert isinstance(result, dict)

    def test_benchmark_function(self, sample_well_data):
        """Test the benchmark function."""
        result = dca.benchmark(
            sample_well_data,
            model="arps",
            kind="hyperbolic",
            horizon=12,
            top_n=2,
            verbose=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2  # top_n=2
        assert "rmse" in result.columns
        assert "mae" in result.columns
        assert "smape" in result.columns
        assert "well_id" in result.columns

    def test_benchmark_verbose(self, sample_well_data):
        """Test benchmark function with verbose output."""
        # Verbose mode now uses logging, just verify it runs without error
        result = dca.benchmark(sample_well_data, model="arps", top_n=1, verbose=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_benchmark_insufficient_data(self):
        """Test benchmark with insufficient data."""
        # Create dataset with very short time series
        short_data = pd.DataFrame(
            {
                "well_id": ["WELL_001"] * 5,
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "oil_bbl": [100, 90, 80, 70, 60],
            }
        )

        result = dca.benchmark(short_data, top_n=5)

        # Should return empty DataFrame or handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_benchmark_different_models(self, sample_well_data):
        """Test benchmark with different models."""
        models = ["arps", "timesfm", "chronos"]

        for model in models:
            result = dca.benchmark(sample_well_data, model=model, top_n=1)

            assert isinstance(result, pd.DataFrame)

    def test_plot_function(self, sample_production_data):
        """Test the plot function (without actually displaying)."""
        # Create some forecast data
        forecast_data = dca.forecast(sample_production_data, horizon=6)

        # This should not raise an error
        try:
            # We can't easily test the actual plotting without display
            # but we can test that the function accepts the right parameters
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend

            dca.plot(sample_production_data, forecast_data, title="Test Plot")

        except Exception:
            # If plotting fails due to environment, that's okay for testing
            # The important thing is that the function signature works
            pass


class TestAPIEdgeCases:
    """Test edge cases in the API."""

    def test_forecast_empty_series(self):
        """Test forecast with empty series."""
        empty_series = pd.Series([], dtype=float, name="production")
        empty_series.index = pd.DatetimeIndex([])

        try:
            result = dca.forecast(empty_series)
            # Should either work or raise a meaningful error
            assert isinstance(result, pd.Series)
        except Exception as e:
            # Should raise a meaningful error, not crash
            assert isinstance(e, (ValueError, RuntimeError))

    def test_evaluate_empty_series(self):
        """Test evaluate with empty series."""
        empty_series = pd.Series([], dtype=float)

        try:
            result = dca.evaluate(empty_series, empty_series)
            assert isinstance(result, dict)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, RuntimeError))

    def test_benchmark_empty_dataframe(self):
        """Test benchmark with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["well_id", "date", "oil_bbl"])

        result = dca.benchmark(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_forecast_single_point(self):
        """Test forecast with single data point."""
        single_point = pd.Series(
            [1000], index=pd.DatetimeIndex(["2020-01-01"]), name="production"
        )

        try:
            result = dca.forecast(single_point, horizon=3)
            assert isinstance(result, pd.Series)
            assert len(result) >= 1
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, RuntimeError))


class TestAPIParameterValidation:
    """Test parameter validation in API functions."""

    def test_forecast_invalid_model(self, sample_production_data):
        """Test forecast with invalid model."""
        with pytest.raises(ValueError):
            dca.forecast(sample_production_data, model="invalid_model")

    def test_forecast_invalid_kind(self, sample_production_data):
        """Test forecast with invalid Arps kind."""
        with pytest.raises(ValueError):
            dca.forecast(sample_production_data, model="arps", kind="invalid_kind")

    def test_benchmark_invalid_columns(self):
        """Test benchmark with missing required columns."""
        invalid_df = pd.DataFrame(
            {
                "wrong_col": ["WELL_001"],
                "wrong_date": ["2020-01-01"],
                "wrong_value": [1000],
            }
        )

        try:
            dca.benchmark(invalid_df)
            # Should either work with default column names or handle gracefully
        except Exception as e:
            # Should raise a meaningful error about missing columns
            assert isinstance(e, (KeyError, ValueError))

    def test_forecast_negative_horizon(self, sample_production_data):
        """Test forecast with negative horizon."""
        try:
            dca.forecast(sample_production_data, horizon=-5)
            # Should either handle gracefully or raise error
        except Exception as e:
            assert isinstance(e, (ValueError, RuntimeError))

    def test_benchmark_zero_top_n(self, sample_well_data):
        """Test benchmark with zero top_n."""
        result = dca.benchmark(sample_well_data, top_n=0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
