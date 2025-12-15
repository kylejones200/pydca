"""Tests for fitting module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.fitting import (
    CurveFitFitter,
    FitResult,
    FitSpec,
    RobustLeastSquaresFitter,
    compute_initial_guess,
    compute_metrics,
    compute_weights,
    validate_fit_result,
)
from decline_curve.models_arps import ExponentialArps, HyperbolicArps


class TestFitSpec:
    """Test FitSpec class."""

    def test_fit_spec_creation(self):
        """Test creating FitSpec."""
        model = ExponentialArps()
        spec = FitSpec(
            model=model,
            loss="huber",
            min_points=5,
        )

        assert spec.model == model
        assert spec.loss == "huber"
        assert spec.min_points == 5

    def test_fit_spec_with_fixed_params(self):
        """Test FitSpec with fixed parameters."""
        model = HyperbolicArps()
        spec = FitSpec(
            model=model,
            fixed_params={"qi": 1000.0},
        )

        assert spec.fixed_params["qi"] == 1000.0


class TestComputeWeights:
    """Test weight computation."""

    def test_uniform_weights(self):
        """Test uniform weights."""
        weights = compute_weights(10, "uniform")

        assert len(weights) == 10
        assert np.allclose(weights, 1.0)

    def test_recent_weights(self):
        """Test recent-favoring weights."""
        weights = compute_weights(10, "recent")

        assert len(weights) == 10
        # Last weight should be larger than first
        assert weights[-1] > weights[0]

    def test_custom_weights(self):
        """Test custom weight array."""
        custom_weights = np.array([1.0, 2.0, 3.0])
        weights = compute_weights(3, custom_weights)

        assert np.allclose(weights, custom_weights)


class TestComputeMetrics:
    """Test metrics computation."""

    def test_compute_metrics(self):
        """Test computing fit metrics."""
        q_obs = np.array([100, 90, 80, 70, 60])
        q_pred = np.array([100, 91, 79, 71, 59])  # Small errors

        metrics = compute_metrics(q_obs, q_pred)

        assert "r_squared" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "residuals" in metrics
        assert metrics["r_squared"] > 0.9  # Good fit


class TestComputeInitialGuess:
    """Test initial guess computation."""

    def test_basic_initial_guess(self):
        """Test basic initial guess."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 10)
        q = 1000 * np.exp(-0.1 * t)

        spec = FitSpec(model=model)
        guess = compute_initial_guess(t, q, model, spec)

        assert "qi" in guess
        assert "di" in guess
        assert guess["qi"] > 0
        assert guess["di"] > 0

    def test_ramp_aware_qi(self):
        """Test ramp-aware qi guess."""
        model = ExponentialArps()
        # Create ramp-up then decline
        t = np.linspace(0, 200, 20)
        q = np.concatenate(
            [
                np.linspace(0, 1000, 10),  # Ramp up
                1000 * np.exp(-0.1 * (t[10:] - t[10])),  # Decline
            ]
        )

        spec = FitSpec(model=model, ramp_aware_qi=True, ramp_window_months=6)
        guess = compute_initial_guess(t, q, model, spec)

        # Should use max rate from first 6 months
        assert guess["qi"] >= 1000


class TestCurveFitFitter:
    """Test CurveFitFitter."""

    def test_fit_exponential(self):
        """Test fitting exponential model."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q_true = 1000 * np.exp(-0.1 * t)
        q = q_true + np.random.normal(0, 10, len(t))  # Add noise

        spec = FitSpec(model=model)
        fitter = CurveFitFitter()
        result = fitter.fit(t, q, spec)

        assert result.success
        assert "qi" in result.params
        assert "di" in result.params
        assert result.r_squared > 0.8

    def test_fit_insufficient_points(self):
        """Test fitting with insufficient points."""
        model = ExponentialArps()
        t = np.array([0, 1, 2])
        q = np.array([100, 90, 80])

        spec = FitSpec(model=model, min_points=5)
        fitter = CurveFitFitter()
        result = fitter.fit(t, q, spec)

        assert not result.success
        assert "Insufficient" in result.message

    def test_fit_with_fixed_params(self):
        """Test fitting with fixed parameters."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q = 1000 * np.exp(-0.1 * t)

        spec = FitSpec(model=model, fixed_params={"qi": 1000.0})
        fitter = CurveFitFitter()
        result = fitter.fit(t, q, spec)

        assert result.success
        assert np.isclose(result.params["qi"], 1000.0, rtol=0.01)


class TestRobustLeastSquaresFitter:
    """Test RobustLeastSquaresFitter."""

    def test_fit_with_huber_loss(self):
        """Test fitting with Huber loss."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q_true = 1000 * np.exp(-0.1 * t)
        q = q_true + np.random.normal(0, 10, len(t))

        spec = FitSpec(model=model, loss="huber")
        fitter = RobustLeastSquaresFitter()
        result = fitter.fit(t, q, spec)

        assert result.success
        assert result.r_squared > 0.8

    def test_fit_with_soft_l1_loss(self):
        """Test fitting with soft L1 loss."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q = 1000 * np.exp(-0.1 * t)

        spec = FitSpec(model=model, loss="soft_l1")
        fitter = RobustLeastSquaresFitter()
        result = fitter.fit(t, q, spec)

        assert result.success


class TestValidateFitResult:
    """Test fit result validation."""

    def test_validate_good_fit(self):
        """Test validation of good fit."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q = 1000 * np.exp(-0.1 * t)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        warnings = validate_fit_result(result, t, q)

        # Should have few or no warnings for good fit
        assert len(warnings) <= 1

    def test_validate_out_of_bounds(self):
        """Test validation detects out-of-bounds parameters."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q = 1000 * np.exp(-0.1 * t)

        # Create result with out-of-bounds parameter
        result = FitResult(
            params={"qi": -100.0, "di": 0.1},  # Negative qi
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        warnings = validate_fit_result(result, t, q)

        # Should warn about out-of-bounds
        assert len(warnings) > 0
        assert any("bounds" in w.lower() for w in warnings)
