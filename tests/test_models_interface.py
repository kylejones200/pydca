"""Tests for the Model interface and Arps implementations."""

import numpy as np
import pytest

from decline_curve.models_arps import (
    ExponentialArps,
    HarmonicArps,
    HyperbolicArps,
    HyperbolicToExponentialSwitch,
)
from decline_curve.models_base import Model


class TestModelInterface:
    """Test that all models implement the interface correctly."""

    def test_exponential_implements_interface(self):
        """Test ExponentialArps implements Model interface."""
        model = ExponentialArps()
        assert isinstance(model, Model)
        assert model.name == "ExponentialArps"

    def test_harmonic_implements_interface(self):
        """Test HarmonicArps implements Model interface."""
        model = HarmonicArps()
        assert isinstance(model, Model)
        assert model.name == "HarmonicArps"

    def test_hyperbolic_implements_interface(self):
        """Test HyperbolicArps implements Model interface."""
        model = HyperbolicArps()
        assert isinstance(model, Model)
        assert model.name == "HyperbolicArps"

    def test_switch_implements_interface(self):
        """Test HyperbolicToExponentialSwitch implements Model interface."""
        model = HyperbolicToExponentialSwitch()
        assert isinstance(model, Model)
        assert model.name == "HyperbolicToExponentialSwitch"


class TestExponentialArps:
    """Test ExponentialArps model."""

    def test_rate_calculation(self):
        """Test exponential rate calculation."""
        model = ExponentialArps()
        t = np.array([0, 1, 2, 3])
        params = {"qi": 100.0, "di": 0.1}

        q = model.rate(t, params)

        # At t=0, q should equal qi
        assert np.isclose(q[0], 100.0)
        # At t=1, q should be qi * exp(-di)
        assert np.isclose(q[1], 100.0 * np.exp(-0.1))

    def test_cumulative_analytic(self):
        """Test analytic cumulative calculation."""
        model = ExponentialArps()
        t = np.array([0, 1, 2])
        params = {"qi": 100.0, "di": 0.1}

        cum = model.cum(t, params)

        # At t=0, cumulative should be 0
        assert np.isclose(cum[0], 0.0)
        # At t=1, cumulative should be (qi/di) * (1 - exp(-di))
        expected = (100.0 / 0.1) * (1 - np.exp(-0.1))
        assert np.isclose(cum[1], expected, rtol=1e-6)

    def test_constraints(self):
        """Test parameter constraints."""
        model = ExponentialArps()
        constraints = model.constraints()

        assert "qi" in constraints
        assert "di" in constraints
        assert constraints["qi"][0] == 0.0
        assert constraints["di"][0] == 0.0

    def test_initial_guess(self):
        """Test initial guess generation."""
        model = ExponentialArps()
        t = np.array([0, 30, 60, 90])
        q = np.array([100.0, 80.0, 60.0, 40.0])

        guess = model.initial_guess(t, q)

        assert "qi" in guess
        assert "di" in guess
        assert guess["qi"] > 0
        assert guess["di"] > 0

    def test_validate(self):
        """Test parameter validation."""
        model = ExponentialArps()

        # Valid parameters
        valid, warnings = model.validate({"qi": 100.0, "di": 0.1})
        assert valid is True

        # Invalid: negative qi
        valid, warnings = model.validate({"qi": -10.0, "di": 0.1})
        assert valid is False


class TestHarmonicArps:
    """Test HarmonicArps model."""

    def test_rate_calculation(self):
        """Test harmonic rate calculation."""
        model = HarmonicArps()
        t = np.array([0, 1, 2])
        params = {"qi": 100.0, "di": 0.1}

        q = model.rate(t, params)

        # At t=0, q should equal qi
        assert np.isclose(q[0], 100.0)
        # At t=1, q should be qi / (1 + di)
        assert np.isclose(q[1], 100.0 / (1 + 0.1))

    def test_cumulative_analytic(self):
        """Test analytic cumulative calculation."""
        model = HarmonicArps()
        t = np.array([0, 1, 2])
        params = {"qi": 100.0, "di": 0.1}

        cum = model.cum(t, params)

        # At t=0, cumulative should be 0
        assert np.isclose(cum[0], 0.0)
        # At t=1, cumulative should be (qi/di) * ln(1 + di)
        expected = (100.0 / 0.1) * np.log(1 + 0.1)
        assert np.isclose(cum[1], expected, rtol=1e-6)


class TestHyperbolicArps:
    """Test HyperbolicArps model."""

    def test_rate_calculation(self):
        """Test hyperbolic rate calculation."""
        model = HyperbolicArps()
        t = np.array([0, 1, 2])
        params = {"qi": 100.0, "di": 0.1, "b": 0.5}

        q = model.rate(t, params)

        # At t=0, q should equal qi
        assert np.isclose(q[0], 100.0)
        # At t=1, q should be qi / (1 + b*di)^(1/b)
        expected = 100.0 / np.power(1 + 0.5 * 0.1, 1 / 0.5)
        assert np.isclose(q[1], expected)

    def test_cumulative_analytic(self):
        """Test analytic cumulative for b < 1."""
        model = HyperbolicArps()
        t = np.array([0, 1, 2])
        params = {"qi": 100.0, "di": 0.1, "b": 0.5}

        cum = model.cum(t, params)

        # At t=0, cumulative should be 0
        assert np.isclose(cum[0], 0.0)
        # Verify cumulative is monotonic
        assert np.all(np.diff(cum) >= 0)

    def test_edge_cases(self):
        """Test edge cases (b=0, b=1)."""
        model = HyperbolicArps()
        t = np.array([0, 1, 2])

        # b=0 should behave like exponential
        params_exp = {"qi": 100.0, "di": 0.1, "b": 0.0}
        q_exp = model.rate(t, params_exp)
        expected_exp = 100.0 * np.exp(-0.1 * t)
        assert np.allclose(q_exp, expected_exp)

        # b=1 should behave like harmonic
        params_harm = {"qi": 100.0, "di": 0.1, "b": 1.0}
        q_harm = model.rate(t, params_harm)
        expected_harm = 100.0 / (1 + 0.1 * t)
        assert np.allclose(q_harm, expected_harm)


class TestHyperbolicToExponentialSwitch:
    """Test HyperbolicToExponentialSwitch model."""

    def test_rate_calculation(self):
        """Test rate calculation with switch."""
        model = HyperbolicToExponentialSwitch()
        t = np.array([0, 5, 10, 15, 20])
        params = {
            "qi": 100.0,
            "di": 0.1,
            "b": 0.5,
            "t_switch": 10.0,
            "di_exp": 0.05,
        }

        q = model.rate(t, params)

        # Before switch, should follow hyperbolic
        assert q[0] > q[1] > q[2]  # Declining
        # After switch, should follow exponential
        assert q[2] > q[3] > q[4]  # Declining

    def test_cumulative_calculation(self):
        """Test cumulative calculation with switch."""
        model = HyperbolicToExponentialSwitch()
        t = np.array([0, 5, 10, 15, 20])
        params = {
            "qi": 100.0,
            "di": 0.1,
            "b": 0.5,
            "t_switch": 10.0,
            "di_exp": 0.05,
        }

        cum = model.cum(t, params)

        # Cumulative should be monotonic
        assert np.all(np.diff(cum) >= 0)
        # At t=0, cumulative should be 0
        assert np.isclose(cum[0], 0.0)


class TestCumulativeMonotonicity:
    """Test that cumulative production is always monotonic."""

    @pytest.mark.parametrize(
        "model_class,params",
        [
            (ExponentialArps, {"qi": 100.0, "di": 0.1}),
            (HarmonicArps, {"qi": 100.0, "di": 0.1}),
            (HyperbolicArps, {"qi": 100.0, "di": 0.1, "b": 0.5}),
            (
                HyperbolicToExponentialSwitch,
                {
                    "qi": 100.0,
                    "di": 0.1,
                    "b": 0.5,
                    "t_switch": 10.0,
                    "di_exp": 0.05,
                },
            ),
        ],
    )
    def test_cumulative_monotonic(self, model_class, params):
        """Test that cumulative is always increasing."""
        model = model_class()
        t = np.linspace(0, 100, 100)

        cum = model.cum(t, params)

        # Cumulative should be non-decreasing
        assert np.all(np.diff(cum) >= -1e-10)  # Allow small numerical errors


class TestRateNonNegative:
    """Test that production rates are always non-negative."""

    @pytest.mark.parametrize(
        "model_class,params",
        [
            (ExponentialArps, {"qi": 100.0, "di": 0.1}),
            (HarmonicArps, {"qi": 100.0, "di": 0.1}),
            (HyperbolicArps, {"qi": 100.0, "di": 0.1, "b": 0.5}),
            (
                HyperbolicToExponentialSwitch,
                {
                    "qi": 100.0,
                    "di": 0.1,
                    "b": 0.5,
                    "t_switch": 10.0,
                    "di_exp": 0.05,
                },
            ),
        ],
    )
    def test_rate_non_negative(self, model_class, params):
        """Test that rates are always non-negative."""
        model = model_class()
        t = np.linspace(0, 100, 100)

        q = model.rate(t, params)

        # Rates should be non-negative
        assert np.all(q >= -1e-10)  # Allow small numerical errors


class TestNumericVsAnalyticCumulative:
    """Test that numeric cumulative matches analytic cumulative where applicable."""

    def test_exponential_numeric_matches_analytic(self):
        """Test exponential: numeric integration matches analytic."""
        model = ExponentialArps()
        t = np.linspace(0, 10, 100)
        params = {"qi": 100.0, "di": 0.1}

        # Analytic cumulative
        cum_analytic = model.cum(t, params)

        # Numeric cumulative (integrate rate using numpy trapz)
        q = model.rate(t, params)
        cum_numeric = np.array(
            [np.trapz(q[: i + 1], t[: i + 1]) for i in range(len(t))]
        )

        # Should match within tolerance
        assert np.allclose(cum_analytic, cum_numeric, rtol=1e-4)

    def test_harmonic_numeric_matches_analytic(self):
        """Test harmonic: numeric integration matches analytic."""
        model = HarmonicArps()
        t = np.linspace(0, 10, 100)
        params = {"qi": 100.0, "di": 0.1}

        # Analytic cumulative
        cum_analytic = model.cum(t, params)

        # Numeric cumulative
        q = model.rate(t, params)
        cum_numeric = np.array(
            [np.trapz(q[: i + 1], t[: i + 1]) for i in range(len(t))]
        )

        # Should match within tolerance
        assert np.allclose(cum_analytic, cum_numeric, rtol=1e-4)

    def test_hyperbolic_numeric_matches_analytic(self):
        """Test hyperbolic (b < 1): numeric integration matches analytic."""
        model = HyperbolicArps()
        t = np.linspace(0, 10, 100)
        params = {"qi": 100.0, "di": 0.1, "b": 0.5}

        # Analytic cumulative
        cum_analytic = model.cum(t, params)

        # Numeric cumulative
        q = model.rate(t, params)
        cum_numeric = np.array(
            [np.trapz(q[: i + 1], t[: i + 1]) for i in range(len(t))]
        )

        # Should match within tolerance
        assert np.allclose(cum_analytic, cum_numeric, rtol=1e-3)
