"""Tests for yield models module."""

import numpy as np
import pytest

from decline_curve.yield_models import (
    ConstantYield,
    DecliningYield,
    HyperbolicYield,
    TimeBasedYield,
    YieldModel,
    YieldModelAttachment,
    create_gor_model,
    create_water_cut_model,
)


class TestConstantYield:
    """Test ConstantYield model."""

    def test_constant_yield(self):
        """Test constant yield computation."""
        model = ConstantYield()
        t = np.linspace(0, 100, 100)
        primary_rate = np.ones(100) * 100.0  # Constant 100 bbl/day
        params = {"yield_ratio": 0.5}

        secondary_rate = model.yield_rate(t, primary_rate, params)

        # Should be constant at 50
        assert np.allclose(secondary_rate, 50.0)

    def test_missing_params(self):
        """Test error with missing parameters."""
        model = ConstantYield()
        t = np.linspace(0, 10, 10)
        primary_rate = np.ones(10) * 100.0

        with pytest.raises(ValueError, match="params required"):
            model.yield_rate(t, primary_rate, None)

    def test_missing_primary_rate(self):
        """Test error with missing primary rate."""
        model = ConstantYield()
        t = np.linspace(0, 10, 10)
        params = {"yield_ratio": 0.5}

        with pytest.raises(ValueError, match="primary_rate required"):
            model.yield_rate(t, None, params)

    def test_constraints(self):
        """Test parameter constraints."""
        model = ConstantYield()
        constraints = model.constraints()

        assert "yield_ratio" in constraints
        assert constraints["yield_ratio"][0] == 0.0


class TestDecliningYield:
    """Test DecliningYield model."""

    def test_declining_yield(self):
        """Test declining yield computation."""
        model = DecliningYield()
        t = np.linspace(0, 100, 100)
        primary_rate = np.ones(100) * 100.0
        params = {"yield_initial": 1.0, "decline_rate": 0.01}

        secondary_rate = model.yield_rate(t, primary_rate, params)

        # Should decline over time
        assert secondary_rate[0] > secondary_rate[-1]
        # First value should be approximately yield_initial * primary_rate
        assert np.isclose(secondary_rate[0], 100.0, rtol=0.01)


class TestHyperbolicYield:
    """Test HyperbolicYield model."""

    def test_hyperbolic_yield(self):
        """Test hyperbolic yield computation."""
        model = HyperbolicYield()
        t = np.linspace(0, 100, 100)
        primary_rate = np.ones(100) * 100.0
        params = {"yield_initial": 1.0, "decline_rate": 0.01, "b": 0.5}

        secondary_rate = model.yield_rate(t, primary_rate, params)

        # Should decline over time
        assert secondary_rate[0] > secondary_rate[-1]

    def test_exponential_limit(self):
        """Test exponential limit (b=0)."""
        model = HyperbolicYield()
        t = np.linspace(0, 100, 100)
        primary_rate = np.ones(100) * 100.0
        params = {"yield_initial": 1.0, "decline_rate": 0.01, "b": 0.0}

        secondary_rate = model.yield_rate(t, primary_rate, params)

        # Should behave like exponential
        assert secondary_rate[0] > secondary_rate[-1]


class TestTimeBasedYield:
    """Test TimeBasedYield model."""

    def test_time_based_yield(self):
        """Test time-based yield computation."""
        model = TimeBasedYield()
        t = np.linspace(0, 100, 100)
        params = {"qi": 100.0, "di": 0.01}

        secondary_rate = model.yield_rate(t, None, params)

        # Should decline exponentially
        assert secondary_rate[0] > secondary_rate[-1]
        assert np.isclose(secondary_rate[0], 100.0, rtol=0.01)


class TestYieldModelAttachment:
    """Test YieldModelAttachment."""

    def test_attachment_computation(self):
        """Test computing secondary forecast from attachment."""
        yield_model = ConstantYield()
        attachment = YieldModelAttachment(
            yield_model=yield_model,
            params={"yield_ratio": 0.5},
            phase_name="gas",
        )

        t = np.linspace(0, 100, 100)
        primary_rate = np.ones(100) * 100.0

        secondary_rate = attachment.compute_secondary_forecast(t, primary_rate)

        assert np.allclose(secondary_rate, 50.0)
        assert attachment.phase_name == "gas"


class TestConvenienceFunctions:
    """Test convenience functions for creating yield models."""

    def test_create_gor_model_constant(self):
        """Test creating constant GOR model."""
        gor_model = create_gor_model(1000.0, model_type="constant")

        assert isinstance(gor_model, YieldModelAttachment)
        assert gor_model.phase_name == "gas"
        assert gor_model.params["yield_ratio"] == 1000.0

    def test_create_gor_model_declining(self):
        """Test creating declining GOR model."""
        gor_model = create_gor_model(
            1000.0, gor_decline_rate=0.001, model_type="declining"
        )

        assert isinstance(gor_model, YieldModelAttachment)
        assert gor_model.params["yield_initial"] == 1000.0
        assert gor_model.params["decline_rate"] == 0.001

    def test_create_water_cut_model(self):
        """Test creating water cut model."""
        water_model = create_water_cut_model(0.1, model_type="constant")

        assert isinstance(water_model, YieldModelAttachment)
        assert water_model.phase_name == "water"
        assert water_model.params["yield_ratio"] == 0.1

    def test_invalid_model_type(self):
        """Test error with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_gor_model(1000.0, model_type="invalid")


class TestYieldModelIntegration:
    """Integration tests for yield models with primary models."""

    def test_gor_with_oil_forecast(self):
        """Test GOR model with oil production forecast."""
        from decline_curve.models_arps import HyperbolicArps

        # Create oil forecast
        oil_model = HyperbolicArps()
        t = np.linspace(0, 100, 100)
        oil_params = {"qi": 1000.0, "di": 0.1, "b": 0.5}
        oil_rate = oil_model.rate(t, oil_params)

        # Create GOR model
        gor_model = create_gor_model(1000.0, model_type="constant")
        gas_rate = gor_model.compute_secondary_forecast(t, oil_rate)

        # Gas rate should be proportional to oil rate
        assert len(gas_rate) == len(oil_rate)
        assert np.allclose(gas_rate, oil_rate * 1000.0)

    def test_water_cut_with_oil_forecast(self):
        """Test water cut model with oil production forecast."""
        from decline_curve.models_arps import ExponentialArps

        # Create oil forecast
        oil_model = ExponentialArps()
        t = np.linspace(0, 100, 100)
        oil_params = {"qi": 1000.0, "di": 0.1}
        oil_rate = oil_model.rate(t, oil_params)

        # Create water cut model
        water_model = create_water_cut_model(0.2, model_type="constant")
        water_rate = water_model.compute_secondary_forecast(t, oil_rate)

        # Water rate should be 20% of oil rate
        assert len(water_rate) == len(oil_rate)
        assert np.allclose(water_rate, oil_rate * 0.2)
