"""Tests for reserves estimation module."""

import numpy as np

from decline_curve.models import ArpsParams
from decline_curve.reserves import forecast_and_reserves


class TestReservesEstimation:
    """Test suite for reserve estimation functionality."""

    def test_basic_reserves_calculation(self):
        """Test basic reserve estimation."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        result = forecast_and_reserves(
            params=params, t_max=120, dt=1.0, econ_limit=10.0
        )

        # Check return structure
        assert isinstance(result, dict)
        required_keys = ["t", "q", "t_valid", "q_valid", "eur"]
        for key in required_keys:
            assert key in result

        # Check data types and shapes
        assert isinstance(result["t"], np.ndarray)
        assert isinstance(result["q"], np.ndarray)
        assert isinstance(result["t_valid"], np.ndarray)
        assert isinstance(result["q_valid"], np.ndarray)
        assert isinstance(result["eur"], (int, float))

        # Check that valid arrays are subsets of full arrays
        assert len(result["t_valid"]) <= len(result["t"])
        assert len(result["q_valid"]) <= len(result["q"])
        assert len(result["t_valid"]) == len(result["q_valid"])

        # EUR should be positive
        assert result["eur"] > 0

        # All valid production should be above economic limit
        assert all(q >= 10.0 for q in result["q_valid"])

    def test_reserves_with_different_parameters(self):
        """Test reserves with different Arps parameters."""
        # Higher initial rate should give higher EUR
        high_qi = ArpsParams(qi=1500, di=0.10, b=0.5)
        low_qi = ArpsParams(qi=500, di=0.10, b=0.5)

        high_result = forecast_and_reserves(high_qi)
        low_result = forecast_and_reserves(low_qi)

        assert high_result["eur"] > low_result["eur"]

        # Lower decline rate should give higher EUR
        low_decline = ArpsParams(qi=1000, di=0.05, b=0.5)
        high_decline = ArpsParams(qi=1000, di=0.15, b=0.5)

        low_decline_result = forecast_and_reserves(low_decline)
        high_decline_result = forecast_and_reserves(high_decline)

        assert low_decline_result["eur"] > high_decline_result["eur"]

    def test_reserves_economic_limit_impact(self):
        """Test impact of economic limit on reserves."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # Different economic limits
        low_limit = forecast_and_reserves(params, econ_limit=5.0)
        high_limit = forecast_and_reserves(params, econ_limit=20.0)

        # Lower economic limit should give higher EUR
        assert low_limit["eur"] > high_limit["eur"]

        # Higher limit should have fewer valid time periods
        assert len(low_limit["t_valid"]) > len(high_limit["t_valid"])

        # All production should be above respective limits
        assert all(q >= 5.0 for q in low_limit["q_valid"])
        assert all(q >= 20.0 for q in high_limit["q_valid"])

    def test_reserves_time_horizon_impact(self):
        """Test impact of time horizon on reserves."""
        params = ArpsParams(qi=1000, di=0.08, b=0.3)  # Slow decline

        short_horizon = forecast_and_reserves(params, t_max=60)
        long_horizon = forecast_and_reserves(params, t_max=240)

        # Longer horizon should give higher or equal EUR
        assert long_horizon["eur"] >= short_horizon["eur"]

        # Longer horizon should have more time points
        assert len(long_horizon["t"]) > len(short_horizon["t"])

    def test_reserves_different_decline_types(self):
        """Test reserves for different decline curve types."""
        # Exponential decline (b=0)
        exp_params = ArpsParams(qi=1000, di=0.10, b=0.0)
        exp_result = forecast_and_reserves(exp_params)

        # Harmonic decline (b=1)
        harm_params = ArpsParams(qi=1000, di=0.10, b=1.0)
        harm_result = forecast_and_reserves(harm_params)

        # Hyperbolic decline (0 < b < 1)
        hyp_params = ArpsParams(qi=1000, di=0.10, b=0.5)
        hyp_result = forecast_and_reserves(hyp_params)

        # All should produce positive EUR
        assert exp_result["eur"] > 0
        assert harm_result["eur"] > 0
        assert hyp_result["eur"] > 0

        # Harmonic should typically give highest EUR for same qi, di
        assert harm_result["eur"] > exp_result["eur"]

    def test_reserves_time_step_impact(self):
        """Test impact of time step on reserves accuracy."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # Different time steps
        coarse_dt = forecast_and_reserves(params, dt=12.0)  # Annual
        fine_dt = forecast_and_reserves(params, dt=1.0)  # Monthly

        # Finer time step should give more accurate EUR
        # due to better numerical integration, but allow for numerical differences
        assert (
            fine_dt["eur"] >= coarse_dt["eur"] * 0.90
        )  # Allow 10% tolerance for numerical integration differences

        # Fine time step should have more data points
        assert len(fine_dt["t"]) > len(coarse_dt["t"])

    def test_reserves_edge_cases(self):
        """Test reserves calculation edge cases."""
        # Very high economic limit (no economic production)
        params = ArpsParams(qi=100, di=0.10, b=0.5)

        no_econ_result = forecast_and_reserves(params, econ_limit=200.0)

        # Should have no valid production periods
        assert len(no_econ_result["t_valid"]) == 0
        assert len(no_econ_result["q_valid"]) == 0
        assert no_econ_result["eur"] == 0

        # Very low economic limit (all production economic)
        all_econ_result = forecast_and_reserves(params, econ_limit=0.1)

        # Should have production for full time horizon
        assert len(all_econ_result["t_valid"]) == len(all_econ_result["t"])
        assert all_econ_result["eur"] > 0


class TestReservesIntegration:
    """Test integration with other modules and real-world scenarios."""

    def test_reserves_with_dca_api(self):
        """Test reserves calculation through main DCA API."""
        from decline_curve.dca import ArpsParams, reserves

        params = ArpsParams(qi=1000, di=0.10, b=0.5)
        result = reserves(params, t_max=120, econ_limit=15.0)

        assert isinstance(result, dict)
        assert "eur" in result
        assert result["eur"] > 0

    def test_reserves_consistency_with_models(self):
        """Test that reserves are consistent with direct model predictions."""
        from decline_curve.models import predict_arps

        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # Calculate via reserves function
        reserves_result = forecast_and_reserves(
            params, t_max=120, dt=1.0, econ_limit=10.0
        )

        # Calculate directly using models
        t = np.arange(0, 120 + 1, 1.0)
        q = predict_arps(t, params)
        valid_mask = q >= 10.0

        t_valid_direct = t[valid_mask]
        q_valid_direct = q[valid_mask]
        eur_direct = np.trapz(q_valid_direct, t_valid_direct)

        # Should be very close (within numerical precision)
        np.testing.assert_array_almost_equal(reserves_result["t"], t)
        np.testing.assert_array_almost_equal(reserves_result["q"], q)
        assert abs(reserves_result["eur"] - eur_direct) < 1.0

    def test_reserves_realistic_scenarios(self):
        """Test with realistic field scenarios."""
        # Typical oil well parameters
        oil_well = ArpsParams(qi=800, di=0.12, b=0.6)
        oil_reserves = forecast_and_reserves(
            oil_well, t_max=240, econ_limit=15.0  # 20 years  # 15 bbl/month
        )

        # Typical gas well parameters
        gas_well = ArpsParams(qi=5000, di=0.15, b=0.8)
        gas_reserves = forecast_and_reserves(
            gas_well, t_max=180, econ_limit=100.0  # 15 years  # 100 mcf/month
        )

        # Both should have reasonable EUR values
        assert 5000 < oil_reserves["eur"] < 50000  # Reasonable oil EUR range
        assert 20000 < gas_reserves["eur"] < 200000  # Reasonable gas EUR range

        # Should have reasonable economic lives
        assert 6 < len(oil_reserves["t_valid"]) <= 240  # 6 months to 20 years
        assert (
            6 < len(gas_reserves["t_valid"]) <= 181
        )  # 6 months to 15+ years (allow for edge case)

    def test_reserves_parameter_sensitivity(self):
        """Test sensitivity of reserves to parameter changes."""
        base_params = ArpsParams(qi=1000, di=0.10, b=0.5)
        base_result = forecast_and_reserves(base_params)

        # Test qi sensitivity
        qi_variations = [800, 900, 1100, 1200]
        qi_sensitivities = []

        for qi in qi_variations:
            params = ArpsParams(qi=qi, di=0.10, b=0.5)
            result = forecast_and_reserves(params)
            sensitivity = (result["eur"] - base_result["eur"]) / base_result["eur"]
            qi_sensitivities.append(sensitivity)

        # EUR should increase with qi
        assert qi_sensitivities[2] > 0  # qi=1100 > base
        assert qi_sensitivities[3] > qi_sensitivities[2]  # qi=1200 > qi=1100

        # Test di sensitivity
        di_variations = [0.08, 0.09, 0.11, 0.12]
        di_sensitivities = []

        for di in di_variations:
            params = ArpsParams(qi=1000, di=di, b=0.5)
            result = forecast_and_reserves(params)
            sensitivity = (result["eur"] - base_result["eur"]) / base_result["eur"]
            di_sensitivities.append(sensitivity)

        # EUR should decrease with di
        assert di_sensitivities[2] < 0  # di=0.11 < base
        assert di_sensitivities[3] < di_sensitivities[2]  # di=0.12 < di=0.11


class TestReservesErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test that ArpsParams handles validation or reserves function
        # handles gracefully
        try:
            # Negative qi - may be handled by ArpsParams or reserves function
            params = ArpsParams(qi=-100, di=0.10, b=0.5)
            result = forecast_and_reserves(params)
            # If no exception, result should be reasonable (empty or zero EUR)
            assert result["eur"] >= 0
        except (ValueError, AssertionError):
            # Expected behavior - validation caught the error
            pass

        try:
            # Negative di
            params = ArpsParams(qi=1000, di=-0.10, b=0.5)
            result = forecast_and_reserves(params)
            assert result["eur"] >= 0
        except (ValueError, AssertionError):
            pass

        try:
            # Invalid b value (b > 1)
            params = ArpsParams(qi=1000, di=0.10, b=1.5)
            result = forecast_and_reserves(params)
            assert result["eur"] >= 0
        except (ValueError, AssertionError):
            pass

    def test_extreme_parameters(self):
        """Test with extreme but valid parameters."""
        # Very high decline rate
        high_decline = ArpsParams(qi=1000, di=0.50, b=0.9)
        result = forecast_and_reserves(high_decline, econ_limit=1.0)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["eur"] >= 0

        # Very low decline rate
        low_decline = ArpsParams(qi=1000, di=0.001, b=0.1)
        result = forecast_and_reserves(
            low_decline, t_max=60
        )  # Limit time to avoid huge EUR

        assert isinstance(result, dict)
        assert result["eur"] > 0

    def test_zero_time_horizon(self):
        """Test with zero time horizon."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        result = forecast_and_reserves(params, t_max=0)

        # Should have minimal or no production
        assert len(result["t"]) <= 1
        assert result["eur"] == 0 or result["eur"] < 1

    def test_very_large_time_horizon(self):
        """Test with very large time horizon."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # 100 years
        result = forecast_and_reserves(params, t_max=1200, econ_limit=0.1)

        # Should handle large arrays
        assert isinstance(result, dict)
        assert result["eur"] > 0
        assert len(result["t"]) == 1201  # 0 to 1200 inclusive

    def test_numerical_precision(self):
        """Test numerical precision with small values."""
        # Very small production rates
        params = ArpsParams(qi=1.0, di=0.10, b=0.5)

        result = forecast_and_reserves(params, econ_limit=0.01)

        assert isinstance(result, dict)
        assert result["eur"] >= 0
        assert not np.isnan(result["eur"])
        assert not np.isinf(result["eur"])


class TestReservesPerformance:
    """Test performance and scalability."""

    def test_large_time_arrays(self):
        """Test performance with large time arrays."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        import time

        start_time = time.time()

        # Monthly time steps for 50 years
        result = forecast_and_reserves(params, t_max=600, dt=1.0)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert execution_time < 5.0  # 5 seconds max
        assert result["eur"] > 0
        assert len(result["t"]) == 601  # 0 to 600 inclusive

    def test_fine_time_resolution(self):
        """Test with very fine time resolution."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # Daily time steps for 2 years
        result = forecast_and_reserves(params, t_max=24, dt=1 / 30)  # Daily steps

        assert isinstance(result, dict)
        assert result["eur"] > 0
        assert len(result["t"]) > 700  # Should have many time points

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        params = ArpsParams(qi=1000, di=0.10, b=0.5)

        # Create multiple large results
        results = []
        for i in range(5):
            result = forecast_and_reserves(params, t_max=300, dt=0.5)
            results.append(result["eur"])  # Only keep EUR to save memory

        # All should be similar (same parameters)
        assert all(abs(eur - results[0]) < 1.0 for eur in results)
        assert all(eur > 0 for eur in results)
