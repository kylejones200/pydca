import numpy as np
import pandas as pd

from decline_curve.economics import economic_metrics


class TestEconomicMetrics:
    """Test suite for economic evaluation functionality."""

    def test_basic_economic_calculation(self):
        """Test basic economic metrics calculation."""
        # Simple production profile
        q = np.array([100, 90, 80, 70, 60])
        price = 60.0
        opex = 15.0
        discount_rate = 0.10

        result = economic_metrics(q, price, opex, discount_rate)

        # Check return structure
        assert isinstance(result, dict)
        assert "npv" in result
        assert "cash_flow" in result
        assert "payback_month" in result

        # Check data types
        assert isinstance(result["npv"], (int, float))
        assert isinstance(result["cash_flow"], np.ndarray)
        assert isinstance(result["payback_month"], (int, np.integer))

        # Check cash flow calculation
        expected_cf = (price - opex) * q
        np.testing.assert_array_almost_equal(result["cash_flow"], expected_cf)

        # NPV should be positive for profitable scenario
        assert result["npv"] > 0

        # Payback should be reasonable (first month for positive cash flow)
        assert result["payback_month"] >= 0

    def test_economic_metrics_negative_cash_flow(self):
        """Test economics with negative cash flow scenario."""
        q = np.array([50, 40, 30, 20, 10])
        price = 20.0  # Low price
        opex = 30.0  # High opex

        result = economic_metrics(q, price, opex, 0.10)

        # Should have negative cash flows
        assert all(cf < 0 for cf in result["cash_flow"])

        # NPV should be negative
        assert result["npv"] < 0

        # No payback period (should be -1)
        assert result["payback_month"] == -1

    def test_economic_metrics_break_even(self):
        """Test economics at break-even point."""
        q = np.array([100, 90, 80, 70, 60])
        price = 25.0
        opex = 25.0  # Break-even price

        result = economic_metrics(q, price, opex, 0.10)

        # Cash flows should be zero
        np.testing.assert_array_almost_equal(result["cash_flow"], np.zeros_like(q))

        # NPV should be zero (or very close)
        assert abs(result["npv"]) < 1e-10

        # No payback since no positive cash flow
        assert result["payback_month"] == -1

    def test_different_discount_rates(self):
        """Test impact of different discount rates."""
        q = np.array([100, 90, 80, 70, 60])
        price = 60.0
        opex = 15.0

        # Calculate NPV with different discount rates
        low_rate = economic_metrics(q, price, opex, 0.05)
        high_rate = economic_metrics(q, price, opex, 0.20)

        # Higher discount rate should result in lower NPV
        assert low_rate["npv"] > high_rate["npv"]

        # Cash flows should be the same
        np.testing.assert_array_equal(low_rate["cash_flow"], high_rate["cash_flow"])

        # Payback periods should be the same (undiscounted)
        assert low_rate["payback_month"] == high_rate["payback_month"]

    def test_payback_calculation(self):
        """Test payback period calculation logic."""
        # Design scenario with known payback
        q = np.array([10, 20, 30, 40, 50])  # Increasing production
        price = 60.0
        opex = 10.0

        result = economic_metrics(q, price, opex, 0.10)

        # Calculate expected cumulative cash flow
        cash_flow = (price - opex) * q
        cum_cf = np.cumsum(cash_flow)

        # Find first positive cumulative cash flow
        expected_payback = np.argmax(cum_cf > 0)

        assert result["payback_month"] == expected_payback

    def test_single_period_economics(self):
        """Test economics with single time period."""
        q = np.array([100])
        price = 60.0
        opex = 15.0

        result = economic_metrics(q, price, opex, 0.10)

        assert len(result["cash_flow"]) == 1
        assert result["cash_flow"][0] == (price - opex) * q[0]
        assert (
            result["npv"] == result["cash_flow"][0]
        )  # No discounting for single period
        assert result["payback_month"] == 0  # Immediate payback

    def test_zero_production(self):
        """Test economics with zero production."""
        q = np.array([0, 0, 0])
        price = 60.0
        opex = 15.0

        result = economic_metrics(q, price, opex, 0.10)

        # All cash flows should be zero
        np.testing.assert_array_equal(result["cash_flow"], np.zeros_like(q))
        assert result["npv"] == 0
        assert result["payback_month"] == -1

    def test_large_production_profile(self):
        """Test economics with large production profile."""
        # 10 years of monthly production
        months = 120
        q = np.linspace(1000, 100, months)  # Declining production
        price = 50.0
        opex = 12.0

        result = economic_metrics(q, price, opex, 0.10)

        assert len(result["cash_flow"]) == months
        assert result["npv"] > 0  # Should be profitable
        assert 0 <= result["payback_month"] < months


class TestEconomicsIntegration:
    """Test integration with pandas Series and real-world scenarios."""

    def test_economics_with_pandas_series(self):
        """Test economics calculation with pandas Series input."""
        from decline_curve.dca import economics

        # Create production series
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        production = pd.Series(
            [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450], index=dates
        )

        result = economics(production, price=60, opex=15)

        assert isinstance(result, dict)
        assert "npv" in result
        assert result["npv"] > 0
        assert len(result["cash_flow"]) == len(production)

    def test_economics_realistic_scenarios(self):
        """Test with realistic oil and gas scenarios."""
        # Typical oil well production profile
        oil_production = np.array([800, 720, 650, 580, 520, 470, 420, 380, 340, 300])

        # Oil economics
        oil_result = economic_metrics(
            oil_production, price=70.0, opex=20.0, discount_rate=0.12  # $/bbl  # $/bbl
        )

        # Gas well production profile (higher volumes, lower price)
        gas_production = np.array(
            [5000, 4500, 4000, 3600, 3200, 2900, 2600, 2300, 2100, 1900]
        )

        # Gas economics
        gas_result = economic_metrics(
            gas_production, price=3.50, opex=0.80, discount_rate=0.12  # $/mcf  # $/mcf
        )

        # Both should be profitable
        assert oil_result["npv"] > 0
        assert gas_result["npv"] > 0

        # Oil should have higher per-unit margins
        oil_margin = 70.0 - 20.0
        gas_margin = 3.50 - 0.80
        assert oil_margin > gas_margin

    def test_economics_price_sensitivity(self):
        """Test economics sensitivity to price changes."""
        q = np.array([1000, 900, 800, 700, 600])
        opex = 15.0

        prices = [40, 50, 60, 70, 80]
        npvs = []

        for price in prices:
            result = economic_metrics(q, price, opex, 0.10)
            npvs.append(result["npv"])

        # NPV should increase with price
        for i in range(1, len(npvs)):
            assert npvs[i] > npvs[i - 1]

        # Should have break-even point
        break_even_found = False
        for i, npv in enumerate(npvs):
            if npv > 0:
                break_even_found = True
                break

        assert (
            break_even_found or prices[0] > opex
        )  # Either found break-even or all prices profitable


class TestEconomicsErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_production_array(self):
        """Test behavior with empty production array."""
        q = np.array([])

        result = economic_metrics(q, 60.0, 15.0, 0.10)

        assert len(result["cash_flow"]) == 0
        assert result["npv"] == 0
        assert result["payback_month"] == -1

    def test_negative_production_values(self):
        """Test handling of negative production values."""
        q = np.array(
            [100, -50, 80]
        )  # Negative production (shouldn't happen in reality)

        result = economic_metrics(q, 60.0, 15.0, 0.10)

        # Should handle gracefully
        assert len(result["cash_flow"]) == 3
        assert (
            result["cash_flow"][1] < 0
        )  # Negative production gives negative cash flow

    def test_extreme_discount_rates(self):
        """Test with extreme discount rates."""
        q = np.array([100, 90, 80, 70, 60])
        price = 60.0
        opex = 15.0

        # Very high discount rate
        high_discount = economic_metrics(q, price, opex, 0.99)

        # Very low discount rate
        low_discount = economic_metrics(q, price, opex, 0.01)

        # Should handle both cases
        assert isinstance(high_discount["npv"], (int, float))
        assert isinstance(low_discount["npv"], (int, float))

        # Low discount should give higher NPV
        assert low_discount["npv"] > high_discount["npv"]

    def test_zero_discount_rate(self):
        """Test with zero discount rate."""
        q = np.array([100, 90, 80])

        result = economic_metrics(q, 60.0, 15.0, 0.0)

        # With zero discount, NPV should equal sum of cash flows
        expected_npv = sum((60.0 - 15.0) * q)
        assert abs(result["npv"] - expected_npv) < 1e-10

    def test_very_long_production_profile(self):
        """Test with very long production profiles."""
        # 50 years of monthly production
        months = 600
        q = np.linspace(1000, 10, months)

        result = economic_metrics(q, 50.0, 12.0, 0.10)

        # Should handle large arrays
        assert len(result["cash_flow"]) == months
        assert isinstance(result["npv"], (int, float))
        assert not np.isnan(result["npv"])
        assert not np.isinf(result["npv"])


class TestEconomicsNumericalAccuracy:
    """Test numerical accuracy and precision."""

    def test_npv_calculation_accuracy(self):
        """Test NPV calculation against manual calculation."""
        q = np.array([100, 90, 80])
        price = 60.0
        opex = 15.0
        discount_rate = 0.10

        result = economic_metrics(q, price, opex, discount_rate)

        # Manual NPV calculation
        monthly_rate = discount_rate / 12
        cash_flows = (price - opex) * q

        manual_npv = 0
        for i, cf in enumerate(cash_flows):
            manual_npv += cf / ((1 + monthly_rate) ** i)

        # Should match within numerical precision
        assert abs(result["npv"] - manual_npv) < 1e-10

    def test_cumulative_cash_flow_accuracy(self):
        """Test that payback calculation uses correct cumulative cash flow."""
        q = np.array([50, 100, 150, 200])  # Increasing production
        price = 60.0
        opex = 40.0  # High opex for delayed payback

        result = economic_metrics(q, price, opex, 0.10)

        # Manual cumulative calculation
        cash_flows = (price - opex) * q
        cum_cf = np.cumsum(cash_flows)

        # Find first positive cumulative cash flow
        payback_indices = np.where(cum_cf > 0)[0]
        expected_payback = payback_indices[0] if len(payback_indices) > 0 else -1

        assert result["payback_month"] == expected_payback
