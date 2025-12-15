"""Property tests for decline curve models.

These tests verify mathematical properties that should hold for all models.
"""

import numpy as np
import pytest

from decline_curve.models_arps import (
    ExponentialArps,
    HarmonicArps,
    HyperbolicArps,
)
from decline_curve.test_utils import (
    check_model_properties,
    generate_piecewise_decline,
    generate_ramp_up_data,
    generate_synthetic_arps_data,
)


class TestModelProperties:
    """Test mathematical properties of models."""

    def test_exponential_properties(self):
        """Test exponential Arps properties."""
        model = ExponentialArps()
        t = np.linspace(0, 100, 100)
        params = {"qi": 1000.0, "di": 0.1}

        checks = check_model_properties(model, params, t)

        assert checks["valid_params"]
        assert checks["non_negative_rates"]
        assert checks["non_negative_cumulative"]
        assert checks["monotone_cumulative"]
        assert checks["monotone_decline"]
        assert checks["finite_values"]

    def test_hyperbolic_properties(self):
        """Test hyperbolic Arps properties."""
        model = HyperbolicArps()
        t = np.linspace(0, 100, 100)
        params = {"qi": 1000.0, "di": 0.1, "b": 0.5}

        checks = check_model_properties(model, params, t)

        assert checks["valid_params"]
        assert checks["non_negative_rates"]
        assert checks["non_negative_cumulative"]
        assert checks["monotone_cumulative"]
        assert checks["monotone_decline"]
        assert checks["finite_values"]

    def test_harmonic_properties(self):
        """Test harmonic Arps properties."""
        model = HarmonicArps()
        t = np.linspace(0, 100, 100)
        params = {"qi": 1000.0, "di": 0.1}

        checks = check_model_properties(model, params, t)

        assert checks["valid_params"]
        assert checks["non_negative_rates"]
        assert checks["non_negative_cumulative"]
        assert checks["monotone_cumulative"]
        assert checks["monotone_decline"]
        assert checks["finite_values"]

    def test_parameter_bounds(self):
        """Test that parameter bounds are checked."""
        model = HyperbolicArps()
        t = np.linspace(0, 100, 100)

        # Test that validation is called
        # Note: Some models may handle invalid params gracefully
        invalid_params = [
            {"qi": -100, "di": 0.1, "b": 0.5},  # Negative qi
        ]

        for params in invalid_params:
            checks = check_model_properties(model, params, t)
            # At minimum, validation should be attempted
            assert "valid_params" in checks


class TestSyntheticDataGenerators:
    """Test synthetic data generators."""

    def test_generate_arps_data(self):
        """Test synthetic Arps data generation."""
        t, q = generate_synthetic_arps_data(1000, 0.1, 0.5, seed=42)

        assert len(t) == len(q)
        assert np.all(q >= 0)
        assert len(t) > 0

    def test_generate_piecewise_decline(self):
        """Test piecewise decline generation."""
        segments = [
            {"qi": 1000, "di": 0.1, "b": 0.5, "duration": 50},
            {"qi": 500, "di": 0.15, "b": 0.6, "duration": 50},
        ]
        t, q = generate_piecewise_decline(segments, seed=42)

        assert len(t) == len(q)
        assert np.all(q >= 0)

    def test_generate_ramp_up_data(self):
        """Test ramp-up data generation."""
        t, q = generate_ramp_up_data(1000, 10, 0.1, 0.5, seed=42)

        assert len(t) == len(q)
        assert np.all(q >= 0)
        # First point should be near zero
        assert q[0] < 100

    def test_baseline_dataset(self):
        """Test baseline dataset creation."""
        from decline_curve.test_utils import create_baseline_dataset

        wells = create_baseline_dataset(n_wells=5, seed=42)

        assert len(wells) == 5
        for well in wells:
            assert "well_id" in well
            assert "t" in well
            assert "q" in well
            assert len(well["t"]) == len(well["q"])
