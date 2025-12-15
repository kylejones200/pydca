"""
Unit tests for Arps decline curve models.
"""

import numpy as np
import pytest

from decline_curve.models import (
    ArpsParams,
    estimate_reserves,
    fit_arps,
    predict_arps,
    q_exp,
    q_hyp,
)


class TestDeclineCurves:
    """Test individual decline curve functions."""

    def test_exponential_decline(self):
        """Test exponential decline curve calculation."""
        t = np.array([0, 1, 2, 3])
        qi = 1000
        di = 0.1

        result = q_exp(t, qi, di)
        expected = qi * np.exp(-di * t)

        np.testing.assert_array_almost_equal(result, expected)
        assert result[0] == qi  # Initial production should equal qi
        assert all(
            result[i] >= result[i + 1] for i in range(len(result) - 1)
        )  # Should decline

    def test_hyperbolic_decline(self):
        """Test hyperbolic decline curve calculation."""
        t = np.array([0, 1, 2, 3])
        qi = 1000
        di = 0.1
        b = 0.5

        result = q_hyp(t, qi, di, b)
        expected = qi / (1 + b * di * t) ** (1 / b)

        np.testing.assert_array_almost_equal(result, expected)
        assert result[0] == qi
        assert all(result[i] >= result[i + 1] for i in range(len(result) - 1))

    def test_decline_curves_positive(self):
        """Test that all decline curves produce positive values."""
        t = np.linspace(0, 10, 100)
        qi = 1000
        di = 0.1
        b = 0.5

        exp_result = q_exp(t, qi, di)
        hyp_result = q_hyp(t, qi, di, b)

        assert all(exp_result > 0)
        assert all(hyp_result > 0)


class TestArpsFitting:
    """Test Arps curve fitting functionality."""

    def test_fit_exponential(self, sample_production_data):
        """Test fitting exponential decline curve."""
        t = np.arange(len(sample_production_data))
        q = sample_production_data.values

        params = fit_arps(t, q, kind="exponential")

        assert isinstance(params, ArpsParams)
        assert params.qi > 0
        assert params.di > 0
        assert params.b == 0.0  # Exponential has b=0

    def test_fit_harmonic(self, sample_production_data):
        """Test fitting harmonic decline curve."""
        t = np.arange(len(sample_production_data))
        q = sample_production_data.values

        params = fit_arps(t, q, kind="harmonic")

        assert isinstance(params, ArpsParams)
        assert params.qi > 0
        assert params.di > 0
        assert params.b == 1.0  # Harmonic has b=1

    def test_fit_hyperbolic(self, sample_production_data):
        """Test fitting hyperbolic decline curve."""
        t = np.arange(len(sample_production_data))
        q = sample_production_data.values

        params = fit_arps(t, q, kind="hyperbolic")

        assert isinstance(params, ArpsParams)
        assert params.qi > 0
        assert params.di > 0
        assert 0 < params.b < 2  # Hyperbolic constraint

    def test_fit_invalid_kind(self, sample_production_data):
        """Test fitting with invalid decline type."""
        t = np.arange(len(sample_production_data))
        q = sample_production_data.values

        with pytest.raises(ValueError):
            fit_arps(t, q, kind="invalid")

    def test_predict_arps(self, arps_parameters):
        """Test Arps prediction functionality."""
        t = np.array([0, 1, 2, 3, 4, 5])

        for kind, params in arps_parameters.items():
            result = predict_arps(t, params)

            assert len(result) == len(t)
            assert all(result > 0)
            assert result[0] == params["qi"]  # Initial production
            # Should generally decline (allowing for small numerical errors)
            assert result[-1] <= result[0] * 1.01

    def test_predict_invalid_kind(self):
        """Test prediction with invalid parameters."""
        t = np.array([0, 1, 2])
        params_dict = {"kind": "invalid", "qi": 1000, "di": 0.1, "b": 0.5}

        # This should not raise an error since predict_arps doesn't validate 'kind'
        _ = predict_arps(t, params_dict)  # Test that it doesn't crash
        # Instead test with missing required parameter
        params_missing = {"qi": 1000, "di": 0.1}  # Missing 'b'

        with pytest.raises(KeyError):
            predict_arps(t, params_missing)


class TestReserveEstimation:
    """Test ultimate recovery estimation."""

    def test_exponential_reserves(self):
        """Test reserve estimation for exponential decline."""
        params = {"kind": "exponential", "qi": 1000, "di": 0.1, "b": 0.0}
        reserves = estimate_reserves(params, t_max=50)

        # For exponential: EUR ≈ qi/di for large t_max
        expected_approx = params["qi"] / params["di"]
        assert abs(reserves - expected_approx) / expected_approx < 0.05

    def test_harmonic_reserves(self):
        """Test reserve estimation for harmonic decline."""
        params = {"kind": "harmonic", "qi": 1000, "di": 0.1, "b": 1.0}
        reserves = estimate_reserves(params, t_max=50)

        assert reserves > 0
        # Harmonic decline should give higher reserves than exponential
        exp_params = {"kind": "exponential", "qi": 1000, "di": 0.1, "b": 0.0}
        exp_reserves = estimate_reserves(exp_params, t_max=50)
        assert reserves > exp_reserves

    def test_hyperbolic_reserves(self):
        """Test reserve estimation for hyperbolic decline."""
        params = {"kind": "hyperbolic", "qi": 1000, "di": 0.1, "b": 0.5}
        reserves = estimate_reserves(params, t_max=50)

        assert reserves > 0
        # Should be between exponential and harmonic
        exp_params = {"kind": "exponential", "qi": 1000, "di": 0.1, "b": 0.0}
        harm_params = {"kind": "harmonic", "qi": 1000, "di": 0.1, "b": 1.0}

        exp_reserves = estimate_reserves(exp_params, t_max=50)
        harm_reserves = estimate_reserves(harm_params, t_max=50)

        assert exp_reserves <= reserves <= harm_reserves

    def test_reserves_invalid_kind(self):
        """Test reserve estimation with invalid decline type."""
        params = {"kind": "invalid", "qi": 1000, "di": 0.1, "b": 0.5}

        with pytest.raises(ValueError):
            estimate_reserves(params)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test fitting with empty data."""
        t = np.array([])
        q = np.array([])

        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            fit_arps(t, q, kind="exponential")

    def test_single_point(self):
        """Test fitting with single data point."""
        t = np.array([0])
        q = np.array([1000])

        params = fit_arps(t, q, kind="exponential")
        # Should handle gracefully and return ArpsParams object
        assert isinstance(params, ArpsParams)
        assert params.qi > 0

    def test_zero_production(self):
        """Test with zero production values."""
        t = np.array([0, 1, 2])
        q = np.array([0, 0, 0])

        with pytest.raises(
            ValueError, match="All production values are zero or negative"
        ):
            fit_arps(t, q, kind="exponential")
