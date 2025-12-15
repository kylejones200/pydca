"""Tests for fit diagnostics module."""

import numpy as np
import pytest

from decline_curve.fit_diagnostics import (
    DiagnosticsResult,
    check_quality,
    compute_cumulative_metrics,
    compute_grade,
    compute_information_criteria,
    compute_rate_metrics,
    diagnose_fit,
)
from decline_curve.fitting import FitResult
from decline_curve.models_arps import ExponentialArps


class TestComputeRateMetrics:
    """Test rate metrics computation."""

    def test_perfect_fit(self):
        """Test metrics for perfect fit."""
        q_obs = np.array([100, 90, 80, 70, 60])
        q_pred = q_obs.copy()  # Perfect match

        metrics = compute_rate_metrics(q_obs, q_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r_squared"] == 1.0

    def test_poor_fit(self):
        """Test metrics for poor fit."""
        q_obs = np.array([100, 90, 80, 70, 60])
        q_pred = np.array([50, 45, 40, 35, 30])  # Half the values

        metrics = compute_rate_metrics(q_obs, q_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r_squared"] < 1.0


class TestComputeCumulativeMetrics:
    """Test cumulative metrics computation."""

    def test_cumulative_metrics(self):
        """Test cumulative metrics computation."""
        cum_obs = np.array([100, 190, 270, 340, 400])
        cum_pred = np.array([100, 190, 270, 340, 400])  # Perfect match

        metrics = compute_cumulative_metrics(cum_obs, cum_pred)

        assert metrics["rmse_cumulative"] == 0.0
        assert metrics["relative_error_end"] == 0.0

    def test_missing_cumulative(self):
        """Test with missing cumulative."""
        metrics = compute_cumulative_metrics(None, None)

        assert len(metrics) == 0


class TestComputeInformationCriteria:
    """Test information criteria computation."""

    def test_aic_bic(self):
        """Test AIC and BIC computation."""
        n = 20
        residuals = np.random.normal(0, 1, n)
        n_params = 3

        metrics = compute_information_criteria(n, residuals, n_params)

        assert "aic" in metrics
        assert "bic" in metrics
        assert metrics["bic"] > metrics["aic"]  # BIC should be larger


class TestCheckQuality:
    """Test quality checks."""

    def test_non_monotone_check(self):
        """Test detection of non-monotone decline."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 10)

        # Create result with non-monotone prediction
        result = FitResult(
            params={"qi": 1000.0, "di": -0.01},  # Negative di = increasing
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        q_obs = 1000 * np.exp(-0.1 * t)
        flags = check_quality(result, t, q_obs)

        # Should detect non-monotone (though this is a contrived case)
        assert "non_monotone" in flags

    def test_parameter_on_bound(self):
        """Test detection of parameters on bounds."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 10)

        # Create result with parameter at bound
        constraints = model.constraints()
        qi_lower = constraints["qi"][0]

        result = FitResult(
            params={"qi": qi_lower, "di": 0.1},  # qi at lower bound
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        q_obs = 1000 * np.exp(-0.1 * t)
        flags = check_quality(result, t, q_obs)

        assert "parameter_on_bound" in flags


class TestComputeGrade:
    """Test grade computation."""

    def test_grade_a(self):
        """Test A grade assignment."""
        metrics = {"r_squared": 0.98, "mape": 5.0}
        quality_flags = {}

        grade, score, reason_codes = compute_grade(metrics, quality_flags)

        assert grade == "A"
        assert score >= 90

    def test_grade_f(self):
        """Test F grade assignment."""
        metrics = {"r_squared": 0.3, "mape": 60.0}
        quality_flags = {"fit_failed": True, "non_monotone": True}

        grade, score, reason_codes = compute_grade(metrics, quality_flags)

        assert grade == "F"
        assert score < 60
        assert len(reason_codes) > 0


class TestDiagnoseFit:
    """Test main diagnostics function."""

    def test_diagnose_good_fit(self):
        """Test diagnostics for good fit."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q_obs = 1000 * np.exp(-0.1 * t)

        # Create a good fit result
        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            residuals=q_obs - model.rate(t, {"qi": 1000.0, "di": 0.1}),
            r_squared=0.99,
            rmse=1.0,
            mae=0.8,
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        diagnostics = diagnose_fit(result, t, q_obs)

        assert diagnostics.grade in ["A", "B"]
        assert diagnostics.numeric_score >= 80
        assert len(diagnostics.metrics) > 0

    def test_diagnose_failed_fit(self):
        """Test diagnostics for failed fit."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q_obs = 1000 * np.exp(-0.1 * t)

        result = FitResult(
            params={},
            model=model,
            success=False,
            message="Fitting failed",
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        diagnostics = diagnose_fit(result, t, q_obs)

        assert diagnostics.grade == "F"
        assert diagnostics.numeric_score == 0.0
        assert "fit_failed" in diagnostics.reason_codes

    def test_diagnose_with_cumulative(self):
        """Test diagnostics with cumulative data."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)
        q_obs = 1000 * np.exp(-0.1 * t)
        cum_obs = np.cumsum(q_obs) * (t[1] - t[0]) if len(t) > 1 else q_obs * t[0]

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            residuals=q_obs - model.rate(t, {"qi": 1000.0, "di": 0.1}),
            r_squared=0.99,
            rmse=1.0,
            mae=0.8,
            fit_start_idx=0,
            fit_end_idx=len(t),
        )

        diagnostics = diagnose_fit(result, t, q_obs, cum_obs=cum_obs)

        assert "rmse_cumulative" in diagnostics.metrics
