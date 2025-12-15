"""Tests for Monte Carlo simulation module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.monte_carlo import (
    DistributionParams,
    MonteCarloParams,
    MonteCarloResults,
    monte_carlo_forecast,
)


class TestDistributionParams:
    """Test DistributionParams class."""

    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        dist = DistributionParams("normal", mean=100, std=10)
        samples = dist.sample(n=1000, seed=42)

        assert len(samples) == 1000
        assert np.mean(samples) == pytest.approx(100, abs=5)
        assert np.std(samples) == pytest.approx(10, abs=2)

    def test_lognormal_distribution(self):
        """Test lognormal distribution sampling."""
        dist = DistributionParams("lognormal", mean=100, std=0.3)
        samples = dist.sample(n=1000, seed=42)

        assert len(samples) == 1000
        assert all(samples > 0)  # Lognormal always positive

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist = DistributionParams("uniform", min=10, max=100)
        samples = dist.sample(n=1000, seed=42)

        assert len(samples) == 1000
        assert all(samples >= 10)
        assert all(samples <= 100)

    def test_triangular_distribution(self):
        """Test triangular distribution sampling."""
        dist = DistributionParams("triangular", min=10, mode=50, max=100)
        samples = dist.sample(n=1000, seed=42)

        assert len(samples) == 1000
        assert all(samples >= 10)
        assert all(samples <= 100)

    def test_invalid_distribution(self):
        """Test invalid distribution type."""
        dist = DistributionParams("invalid", mean=100, std=10)
        with pytest.raises(ValueError, match="Unknown distribution"):
            dist.sample(n=10)

    def test_missing_params_normal(self):
        """Test normal distribution with missing parameters."""
        dist = DistributionParams("normal", mean=100)
        with pytest.raises(ValueError, match="requires mean and std"):
            dist.sample(n=10)


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation functions."""

    def test_run_single_simulation(self):
        """Test single simulation iteration via monte_carlo_forecast."""
        # Test via the public API instead of private function
        mc_params = MonteCarloParams(
            qi_dist=DistributionParams("normal", mean=1000, std=50),
            di_dist=DistributionParams("uniform", min=0.08, max=0.12),
            b_dist=DistributionParams("uniform", min=0.4, max=0.6),
            price_dist=DistributionParams("normal", mean=70, std=5),
            opex_dist=DistributionParams("normal", mean=20, std=2),
            n_simulations=1,  # Single simulation
            seed=42,
        )

        results = monte_carlo_forecast(
            mc_params,
            t_max=120,
            dt=1.0,
            econ_limit=10,
            discount_rate=0.10,
            n_jobs=1,
            verbose=False,
        )

        assert len(results.forecasts) == 1
        assert results.p50_eur > 0
        assert results.p50_npv is not None

    def test_monte_carlo_forecast(self):
        """Test full Monte Carlo forecast."""
        mc_params = MonteCarloParams(
            qi_dist=DistributionParams("lognormal", mean=1200, std=0.2),
            di_dist=DistributionParams("uniform", min=0.08, max=0.15),
            b_dist=DistributionParams("triangular", min=0.3, mode=0.5, max=0.8),
            price_dist=DistributionParams("normal", mean=70, std=10),
            opex_dist=DistributionParams("normal", mean=20, std=2),
            n_simulations=100,
            seed=42,
        )

        results = monte_carlo_forecast(
            mc_params,
            t_max=120,
            dt=1.0,
            econ_limit=10,
            discount_rate=0.10,
            n_jobs=1,  # Sequential for testing
            verbose=False,
        )

        assert isinstance(results, MonteCarloResults)
        assert len(results.forecasts) == 100
        assert results.p50_eur > 0
        assert results.p50_npv is not None
        assert len(results.parameters) == 100

    def test_monte_carlo_quantiles(self):
        """Test that quantiles are correctly ordered."""
        mc_params = MonteCarloParams(
            qi_dist=DistributionParams("normal", mean=1000, std=100),
            di_dist=DistributionParams("uniform", min=0.1, max=0.2),
            b_dist=DistributionParams("uniform", min=0.4, max=0.6),
            price_dist=DistributionParams("normal", mean=70, std=5),
            n_simulations=100,
            seed=42,
        )

        results = monte_carlo_forecast(mc_params, n_jobs=1, verbose=False)

        # P10 should be >= P50 (for EUR/NPV, higher is better)
        # But for forecasts at each timestep, check ordering
        assert results.p50_eur >= results.p90_eur  # P50 >= P90 (P90 is conservative)
        assert results.p10_eur >= results.p50_eur  # P10 >= P50 (P10 is optimistic)
        assert results.p50_npv >= results.p90_npv
        assert results.p10_npv >= results.p50_npv

    def test_monte_carlo_with_correlation(self):
        """Test Monte Carlo with correlation matrix."""
        # Create simple correlation matrix (qi and di positively correlated)
        corr_matrix = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],  # qi correlated with di
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        mc_params = MonteCarloParams(
            qi_dist=DistributionParams("normal", mean=1000, std=100),
            di_dist=DistributionParams("normal", mean=0.15, std=0.02),
            b_dist=DistributionParams("uniform", min=0.4, max=0.6),
            price_dist=DistributionParams("normal", mean=70, std=5),
            n_simulations=100,
            correlation_matrix=corr_matrix,
            seed=42,
        )

        results = monte_carlo_forecast(mc_params, n_jobs=1, verbose=False)

        assert isinstance(results, MonteCarloResults)
        assert len(results.forecasts) == 100

    def test_economic_limit(self):
        """Test that economic limit is properly applied."""
        mc_params = MonteCarloParams(
            qi_dist=DistributionParams("normal", mean=100, std=10),  # Low qi
            di_dist=DistributionParams("uniform", min=0.2, max=0.3),  # High decline
            b_dist=DistributionParams("uniform", min=0.4, max=0.6),
            price_dist=DistributionParams("normal", mean=70, std=5),
            n_simulations=50,
            seed=42,
        )

        results = monte_carlo_forecast(
            mc_params,
            econ_limit=50,  # High economic limit
            n_jobs=1,
            verbose=False,
        )

        # Some scenarios may have EUR=0 if they never exceed economic limit
        assert results.p50_eur >= 0
        assert all(results.eur_samples >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
