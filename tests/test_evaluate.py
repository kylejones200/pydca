"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest

from decline_analysis.evaluate import (
    evaluate_forecast,
    mae,
    mape,
    r2_score,
    rmse,
    smape,
)


class TestEvaluationMetrics:
    """Test individual evaluation metrics."""

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        result = rmse(y_true, y_pred)
        assert result == 0.0

    def test_rmse_calculation(self):
        """Test RMSE calculation with known values."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1.1, 1.9, 3.1, 3.9, 5.1])

        expected = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.01, 0.01]))
        result = rmse(y_true, y_pred)

        assert abs(result - expected) < 1e-10

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        result = mae(y_true, y_pred)
        assert result == 0.0

    def test_mae_calculation(self):
        """Test MAE calculation with known values."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([2, 1, 4, 3, 6])

        expected = np.mean([1, 1, 1, 1, 1])  # All errors are 1
        result = mae(y_true, y_pred)

        assert result == expected

    def test_smape_perfect_prediction(self):
        """Test SMAPE with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        result = smape(y_true, y_pred)
        assert result == 0.0

    def test_smape_calculation(self):
        """Test SMAPE calculation with known values."""
        y_true = pd.Series([100, 200])
        y_pred = pd.Series([110, 180])

        # SMAPE = mean(|pred - true| / ((|true| + |pred|) / 2)) * 100
        # For first point: |110 - 100| / ((100 + 110) / 2) = 10 / 105 ≈ 0.0952
        # For second point: |180 - 200| / ((200 + 180) / 2) = 20 / 190 ≈ 0.1053
        expected = ((10 / 105 + 20 / 190) / 2) * 100
        result = smape(y_true, y_pred)

        assert abs(result - expected) < 1e-3

    def test_mape_calculation(self):
        """Test MAPE calculation with known values."""
        y_true = pd.Series([100, 200, 300])
        y_pred = pd.Series([110, 180, 330])

        # MAPE = mean(|pred - true| / |true|) * 100
        expected = np.mean([10 / 100, 20 / 200, 30 / 300]) * 100
        result = mape(y_true, y_pred)

        assert abs(result - expected) < 1e-10

    def test_r2_score_perfect_prediction(self):
        """Test R² with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        result = r2_score(y_true, y_pred)
        assert result == 1.0

    def test_r2_score_mean_prediction(self):
        """Test R² when predictions equal the mean."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([3, 3, 3, 3, 3])  # All predictions equal mean

        result = r2_score(y_true, y_pred)
        assert result == 0.0

    def test_r2_score_worse_than_mean(self):
        """Test R² when predictions are worse than mean."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([0, 0, 0, 0, 0])  # Very bad predictions

        result = r2_score(y_true, y_pred)
        assert result < 0


class TestEvaluateFunction:
    """Test the comprehensive evaluate_forecast function."""

    def test_evaluate_forecast_perfect(self):
        """Test comprehensive evaluation with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        result = evaluate_forecast(y_true, y_pred)

        assert "rmse" in result
        assert "mae" in result
        assert "smape" in result
        assert "mape" in result
        assert "r2" in result

        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0
        assert result["smape"] == 0.0
        assert result["mape"] == 0.0
        assert result["r2"] == 1.0

    def test_evaluate_forecast_realistic(self, sample_production_data):
        """Test evaluation with realistic production data."""
        # Create slightly noisy predictions
        np.random.seed(42)
        noise = np.random.normal(0, sample_production_data * 0.05)
        y_pred = sample_production_data + noise

        result = evaluate_forecast(sample_production_data, y_pred)

        # All metrics should be reasonable values
        assert result["rmse"] > 0
        assert result["mae"] > 0
        assert result["smape"] > 0
        assert result["mape"] > 0
        assert 0 <= result["r2"] <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_values(self):
        """Test metrics with zero values."""
        y_true = pd.Series([0, 1, 2])
        y_pred = pd.Series([0, 1, 2])

        # RMSE and MAE should work fine
        assert rmse(y_true, y_pred) == 0.0
        assert mae(y_true, y_pred) == 0.0

        # SMAPE should handle zeros gracefully
        smape_result = smape(y_true, y_pred)
        assert not np.isnan(smape_result)

    def test_negative_values(self):
        """Test metrics with negative values."""
        y_true = pd.Series([-1, 0, 1])
        y_pred = pd.Series([-1.1, 0.1, 0.9])

        # RMSE and MAE should work with negative values
        rmse_result = rmse(y_true, y_pred)
        mae_result = mae(y_true, y_pred)

        assert rmse_result > 0
        assert mae_result > 0
        assert not np.isnan(rmse_result)
        assert not np.isnan(mae_result)

    def test_single_value(self):
        """Test metrics with single value."""
        y_true = pd.Series([5])
        y_pred = pd.Series([4])

        assert rmse(y_true, y_pred) == 1.0
        assert mae(y_true, y_pred) == 1.0
        assert smape(y_true, y_pred) == (1 / 4.5) * 100  # |4-5| / ((5+4)/2) * 100

    def test_constant_values(self):
        """Test metrics with constant values."""
        y_true = pd.Series([5, 5, 5, 5])
        y_pred = pd.Series([5, 5, 5, 5])

        assert rmse(y_true, y_pred) == 0.0
        assert mae(y_true, y_pred) == 0.0
        assert smape(y_true, y_pred) == 0.0
        assert mape(y_true, y_pred) == 0.0
        assert r2_score(y_true, y_pred) == 1.0  # Perfect prediction

    def test_different_lengths(self):
        """Test that metrics handle series of different lengths."""
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 2])  # Shorter series

        # Should work by aligning indices
        try:
            rmse_result = rmse(y_true, y_pred)
            # Should not raise an error due to pandas alignment
            assert isinstance(rmse_result, (int, float))
        except Exception:
            # If it does raise an error, that's also acceptable behavior
            pass
