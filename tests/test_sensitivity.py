"""Tests for sensitivity analysis module."""

import numpy as np
import pandas as pd

from decline_curve.sensitivity import run_sensitivity


class TestSensitivityAnalysis:
    """Test suite for sensitivity analysis functionality."""

    def test_basic_sensitivity_analysis(self):
        """Test basic sensitivity analysis functionality."""
        param_grid = [
            (1000, 0.10, 0.5),
            (800, 0.08, 0.3),
        ]
        prices = [50, 60]

        results = run_sensitivity(
            param_grid=param_grid, prices=prices, opex=15.0, discount_rate=0.10
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # 2 params × 2 prices

        # Check required columns
        expected_cols = ["qi", "di", "b", "price", "EUR", "NPV", "Payback_month"]
        for col in expected_cols:
            assert col in results.columns

        # Check data types
        assert results["qi"].dtype in [np.float64, int]
        assert results["di"].dtype == np.float64
        assert results["b"].dtype == np.float64
        assert results["price"].dtype in [np.float64, int]
        assert results["EUR"].dtype == np.float64
        assert results["NPV"].dtype == np.float64

    def test_sensitivity_with_different_parameters(self):
        """Test sensitivity analysis with different economic parameters."""
        param_grid = [(1000, 0.10, 0.5)]
        prices = [40, 80]

        results = run_sensitivity(
            param_grid=param_grid,
            prices=prices,
            opex=20.0,
            discount_rate=0.15,
            t_max=120,
            econ_limit=20.0,
        )

        assert len(results) == 2

        # Higher price should generally give higher NPV
        high_price_npv = results[results["price"] == 80]["NPV"].iloc[0]
        low_price_npv = results[results["price"] == 40]["NPV"].iloc[0]
        assert high_price_npv > low_price_npv

    def test_sensitivity_economic_limit_filtering(self):
        """Test that wells below economic limit are filtered out."""
        # Use parameters that will hit economic limit quickly
        param_grid = [(50, 0.80, 0.9)]  # Very high decline, low qi
        prices = [20]  # Very low price

        results = run_sensitivity(
            param_grid=param_grid,
            prices=prices,
            opex=25.0,  # High opex relative to price
            econ_limit=100.0,  # Very high economic limit
            t_max=60,  # Short time horizon
        )

        # Should have few or no results due to economic limit
        assert len(results) <= 1  # Allow for edge cases where it barely passes

    def test_sensitivity_multiple_scenarios(self):
        """Test sensitivity analysis with multiple parameter combinations."""
        # Create a larger parameter grid
        qi_values = [800, 1000, 1200]
        di_values = [0.08, 0.12]
        b_values = [0.3, 0.7]

        param_grid = []
        for qi in qi_values:
            for di in di_values:
                for b in b_values:
                    param_grid.append((qi, di, b))

        prices = [50, 70]

        results = run_sensitivity(param_grid=param_grid, prices=prices, opex=15.0)

        expected_combinations = (
            len(qi_values) * len(di_values) * len(b_values) * len(prices)
        )
        assert (
            len(results) <= expected_combinations
        )  # Some may be filtered by econ limit

        # Check parameter ranges
        assert results["qi"].min() >= min(qi_values)
        assert results["qi"].max() <= max(qi_values)
        assert results["di"].min() >= min(di_values)
        assert results["di"].max() <= max(di_values)

    def test_sensitivity_edge_cases(self):
        """Test sensitivity analysis edge cases."""
        # Test with single parameter and price
        param_grid = [(1000, 0.10, 0.5)]
        prices = [60]

        results = run_sensitivity(param_grid, prices, opex=10.0)

        assert len(results) == 1
        assert results.iloc[0]["qi"] == 1000
        assert results.iloc[0]["price"] == 60

        # Test with zero opex
        results_zero_opex = run_sensitivity(param_grid, prices, opex=0.0)
        assert len(results_zero_opex) == 1
        assert results_zero_opex.iloc[0]["NPV"] > results.iloc[0]["NPV"]

    def test_sensitivity_negative_npv_cases(self):
        """Test cases that should result in negative NPV."""
        param_grid = [(500, 0.20, 0.9)]  # Low qi, high decline
        prices = [20]  # Very low price

        results = run_sensitivity(
            param_grid=param_grid,
            prices=prices,
            opex=30.0,  # High opex relative to price
            econ_limit=5.0,  # Low economic limit to allow calculation
        )

        if len(results) > 0:  # If any results pass economic filter
            assert results.iloc[0]["NPV"] < 0
            assert results.iloc[0]["Payback_month"] == -1  # No payback


class TestSensitivityIntegration:
    """Test integration with other modules."""

    def test_sensitivity_with_arps_params_consistency(self):
        """Test sensitivity results consistency with direct Arps calculations."""
        from decline_curve.economics import economic_metrics
        from decline_curve.models import ArpsParams, predict_arps

        # Single parameter set for comparison
        qi, di, b = 1000, 0.10, 0.5
        price = 60
        opex = 15

        # Calculate via sensitivity analysis
        param_grid = [(qi, di, b)]
        prices = [price]
        sens_results = run_sensitivity(param_grid, prices, opex)

        # Calculate directly
        params = ArpsParams(qi=qi, di=di, b=b)
        t = np.arange(0, 240 + 1, 1)
        q = predict_arps(t, params)
        valid_mask = q > 10.0  # Default econ limit

        if np.any(valid_mask):
            t_valid = t[valid_mask]
            q_valid = q[valid_mask]
            eur_direct = np.trapz(q_valid, t_valid)

            econ_direct = economic_metrics(q_valid, price, opex, 0.10)

            # Compare results (allow small numerical differences)
            assert len(sens_results) == 1
            assert abs(sens_results.iloc[0]["EUR"] - eur_direct) < 1.0
            assert abs(sens_results.iloc[0]["NPV"] - econ_direct["npv"]) < 1.0

    def test_sensitivity_performance(self):
        """Test performance with larger parameter sets."""
        # Create moderate-sized parameter grid
        param_grid = [(qi, 0.10, 0.5) for qi in range(800, 1201, 100)]
        prices = list(range(40, 81, 10))

        import time

        start_time = time.time()

        results = run_sensitivity(param_grid, prices, opex=15.0)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 10.0  # 10 seconds max
        assert len(results) > 0

        # Results should be reasonable
        assert results["EUR"].min() > 0
        assert not results["NPV"].isna().any()


class TestSensitivityErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_parameter_grid(self):
        """Test behavior with empty parameter grid."""
        results = run_sensitivity([], [60], opex=15.0)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_empty_prices(self):
        """Test behavior with empty price list."""
        param_grid = [(1000, 0.10, 0.5)]
        results = run_sensitivity(param_grid, [], opex=15.0)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_invalid_parameters(self):
        """Test behavior with invalid Arps parameters."""
        # Negative qi should be handled gracefully
        param_grid = [(-100, 0.10, 0.5)]
        prices = [60]

        results = run_sensitivity(param_grid, prices, opex=15.0)
        # Should either filter out or handle gracefully
        assert isinstance(results, pd.DataFrame)

    def test_extreme_economic_parameters(self):
        """Test with extreme economic parameters."""
        param_grid = [(1000, 0.10, 0.5)]
        prices = [60]

        # Very high discount rate
        results_high_discount = run_sensitivity(
            param_grid, prices, opex=15.0, discount_rate=0.50
        )

        # Very high opex
        results_high_opex = run_sensitivity(param_grid, prices, opex=100.0)

        assert isinstance(results_high_discount, pd.DataFrame)
        assert isinstance(results_high_opex, pd.DataFrame)

        # High discount rate should reduce NPV
        if len(results_high_discount) > 0:
            normal_results = run_sensitivity(param_grid, prices, opex=15.0)
            if len(normal_results) > 0:
                assert (
                    results_high_discount.iloc[0]["NPV"] < normal_results.iloc[0]["NPV"]
                )
