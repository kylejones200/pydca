"""Tests for uncertainty quantification module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.fitting import FitResult
from decline_curve.models_arps import ExponentialArps
from decline_curve.uncertainty import (
    UncertaintyResult,
    block_bootstrap_residuals,
    quantify_uncertainty,
    quantify_uncertainty_bootstrap,
    quantify_uncertainty_covariance,
    sample_from_covariance,
)


class TestSampleFromCovariance:
    """Test parameter sampling from covariance."""

    def test_sample_parameters(self):
        """Test sampling parameters from covariance matrix."""
        params = {"qi": 1000.0, "di": 0.1}
        param_names = ["qi", "di"]
        covariance = np.array([[10000, 10], [10, 0.001]])  # Small covariance

        samples = sample_from_covariance(
            params, covariance, param_names, n_draws=100, seed=42
        )

        assert samples.shape == (100, 2)
        # Mean should be close to original parameters
        assert np.isclose(samples.mean(axis=0)[0], params["qi"], rtol=0.1)
        assert np.isclose(samples.mean(axis=0)[1], params["di"], rtol=0.1)


class TestBlockBootstrapResiduals:
    """Test block bootstrap residual sampling."""

    def test_bootstrap_residuals(self):
        """Test block bootstrap on residuals."""
        residuals = np.array([1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5])

        bootstrap_samples = block_bootstrap_residuals(
            residuals, block_size=2, n_draws=10, seed=42
        )

        assert bootstrap_samples.shape == (10, len(residuals))
        # All values should come from original residuals
        assert np.all(np.isin(bootstrap_samples, residuals))


class TestQuantifyUncertaintyCovariance:
    """Test covariance-based uncertainty quantification."""

    def test_covariance_method(self):
        """Test uncertainty quantification using covariance."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        param_names = ["qi", "di"]
        covariance = np.array([[10000, 10], [10, 0.001]])

        uncertainty = quantify_uncertainty_covariance(
            result, t, covariance, param_names, n_draws=100, seed=42
        )

        assert uncertainty.method == "covariance"
        assert uncertainty.n_draws == 100
        assert len(uncertainty.rate_p50) == len(t)
        assert uncertainty.rate_p10 is not None
        assert uncertainty.rate_p90 is not None
        # P10 should be higher than P90 (optimistic vs conservative)
        assert np.all(uncertainty.rate_p10 >= uncertainty.rate_p90)


class TestQuantifyUncertaintyBootstrap:
    """Test bootstrap-based uncertainty quantification."""

    def test_bootstrap_method(self):
        """Test uncertainty quantification using bootstrap."""
        model = ExponentialArps()
        t_obs = np.linspace(0, 50, 10)
        t_forecast = np.linspace(0, 100, 20)
        q_obs = 1000 * np.exp(-0.1 * t_obs)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        uncertainty = quantify_uncertainty_bootstrap(
            result, t_forecast, t_obs, q_obs, n_draws=100, seed=42
        )

        assert uncertainty.method == "bootstrap"
        assert uncertainty.n_draws == 100
        assert len(uncertainty.rate_p50) == len(t_forecast)


class TestQuantifyUncertainty:
    """Test main uncertainty quantification function."""

    def test_auto_select_covariance(self):
        """Test auto-selection of covariance method."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        param_names = ["qi", "di"]
        covariance = np.array([[10000, 10], [10, 0.001]])

        uncertainty = quantify_uncertainty(
            result,
            t,
            covariance=covariance,
            param_names=param_names,
            n_draws=50,
            seed=42,
        )

        assert uncertainty.method == "covariance"

    def test_auto_select_bootstrap(self):
        """Test auto-selection of bootstrap method."""
        model = ExponentialArps()
        t_obs = np.linspace(0, 50, 10)
        t_forecast = np.linspace(0, 100, 20)
        q_obs = 1000 * np.exp(-0.1 * t_obs)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        uncertainty = quantify_uncertainty(
            result, t_forecast, t_obs=t_obs, q_obs=q_obs, n_draws=50, seed=42
        )

        assert uncertainty.method == "bootstrap"

    def test_with_econ_limit(self):
        """Test uncertainty with economic limit for EUR."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 20)

        result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        param_names = ["qi", "di"]
        covariance = np.array([[10000, 10], [10, 0.001]])

        uncertainty = quantify_uncertainty(
            result,
            t,
            covariance=covariance,
            param_names=param_names,
            n_draws=50,
            seed=42,
            econ_limit=10.0,
        )

        assert uncertainty.eur_p50 is not None
        assert uncertainty.eur_p10 is not None
        assert uncertainty.eur_p90 is not None
        # P10 EUR should be higher than P90 EUR
        assert uncertainty.eur_p10 >= uncertainty.eur_p90


class TestNativeDuongPLEProbabilistic:
    """Gap 2: Duong and PLE now have native parameter resamplers."""

    @staticmethod
    def _shale_series(n=36):
        """Synthetic shale-like transient production (Duong-ish shape)."""
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        t = np.arange(1, n + 1, dtype=float)
        q = 800.0 * t ** (-0.9) * np.exp(1.2 / (1.0 - 1.1) * (t ** (1.0 - 1.1) - 1.0))
        q = np.maximum(q, 1.0)
        return pd.Series(q, index=dates)

    def test_duong_probabilistic_returns_valid_bands(self):
        from decline_curve.probabilistic_forecast import probabilistic_forecast

        series = self._shale_series()
        pf = probabilistic_forecast(series, kind="duong", n_draws=100, horizon=24, seed=7)

        assert pf.p10 is not None
        assert pf.p50 is not None
        assert pf.p90 is not None
        assert len(pf.p50) == len(series) + 24
        # P10 >= P50 >= P90 at each step
        np.testing.assert_array_less(pf.p90.values - 1, pf.p50.values + 1)

    def test_duong_metadata_records_kind(self):
        from decline_curve.parameter_resample import _duong_resample

        series = self._shale_series()
        draws = _duong_resample(series, n_draws=50, seed=3, horizon=12)
        assert draws.metadata["kind"] == "duong"
        assert draws.draws.shape == (50, len(series) + 12)

    def test_ple_probabilistic_returns_valid_bands(self):
        from decline_curve.probabilistic_forecast import probabilistic_forecast

        series = self._shale_series()
        pf = probabilistic_forecast(series, kind="ple", n_draws=100, horizon=24, seed=11)

        assert len(pf.p50) == len(series) + 24
        np.testing.assert_array_less(pf.p90.values - 1, pf.p50.values + 1)

    def test_ple_metadata_records_kind(self):
        from decline_curve.parameter_resample import _ple_resample

        series = self._shale_series()
        draws = _ple_resample(series, n_draws=50, seed=5, horizon=12)
        assert draws.metadata["kind"] == "ple"
        assert draws.draws.shape == (50, len(series) + 12)
