"""Tests for diagnostic functions module."""

import numpy as np
import pytest

from decline_curve.diagnostics import (
    DiagnosticCurves,
    compute_b_diagnostic,
    compute_beta_diagnostic,
    compute_bourdet_derivative,
    compute_decline_rate_diagnostic,
    compute_diagnostic_curves,
    identify_decline_type,
)


class TestDeclineRateDiagnostic:
    """Test decline rate diagnostic computation."""

    def test_exponential_decline(self):
        """Test D diagnostic for exponential decline (constant D)."""
        t = np.linspace(0, 100, 100)
        di = 0.1
        q = 100.0 * np.exp(-di * t)

        D = compute_decline_rate_diagnostic(t, q)

        # D should be approximately constant at di
        D_valid = D[D > 0]
        assert len(D_valid) > 0
        assert np.allclose(D_valid, di, rtol=0.1)  # Allow 10% tolerance

    def test_harmonic_decline(self):
        """Test D diagnostic for harmonic decline."""
        t = np.linspace(0.1, 100, 100)  # Start > 0 to avoid issues
        di = 0.1
        q = 100.0 / (1 + di * t)

        D = compute_decline_rate_diagnostic(t, q)

        # D should decrease over time
        D_valid = D[D > 0]
        assert len(D_valid) > 0
        # First value should be higher than last
        assert D_valid[0] > D_valid[-1]

    def test_empty_input(self):
        """Test with empty input."""
        t = np.array([])
        q = np.array([])

        D = compute_decline_rate_diagnostic(t, q)

        assert len(D) == 0

    def test_zero_rates(self):
        """Test with zero rates."""
        t = np.linspace(0, 10, 10)
        q = np.zeros(10)

        D = compute_decline_rate_diagnostic(t, q)

        assert np.all(D == 0)


class TestBDiagnostic:
    """Test b-factor diagnostic computation."""

    def test_exponential_b_should_be_zero(self):
        """Test b diagnostic for exponential decline (b ≈ 0)."""
        t = np.linspace(0.1, 100, 100)
        di = 0.1
        q = 100.0 * np.exp(-di * t)

        b = compute_b_diagnostic(t, q)

        # b should be approximately 0 for exponential
        b_valid = b[(b != 0) & np.isfinite(b)]
        if len(b_valid) > 0:
            assert np.abs(np.mean(b_valid)) < 0.2  # Should be close to 0

    def test_harmonic_b_should_be_one(self):
        """Test b diagnostic for harmonic decline (b ≈ 1)."""
        t = np.linspace(0.1, 100, 100)
        di = 0.1
        q = 100.0 / (1 + di * t)

        b = compute_b_diagnostic(t, q)

        # b should be approximately 1 for harmonic
        b_valid = b[(b != 0) & np.isfinite(b)]
        if len(b_valid) > 0:
            assert np.abs(np.mean(b_valid) - 1.0) < 0.3  # Should be close to 1


class TestBetaDiagnostic:
    """Test beta diagnostic computation."""

    def test_harmonic_beta_should_be_negative_one(self):
        """Test beta diagnostic for harmonic decline (beta ≈ -1)."""
        t = np.linspace(1, 100, 100)  # Start at 1 for log
        di = 0.1
        q = 100.0 / (1 + di * t)

        beta = compute_beta_diagnostic(t, q)

        # Beta should be approximately -1 for harmonic
        beta_valid = beta[(beta != 0) & np.isfinite(beta)]
        if len(beta_valid) > 0:
            assert np.abs(np.mean(beta_valid) + 1.0) < 0.3


class TestBourdetDerivative:
    """Test Bourdet derivative computation."""

    def test_basic_derivative(self):
        """Test basic derivative computation."""
        t = np.linspace(0, 10, 100)
        q = 100.0 * np.exp(-0.1 * t)

        dq = compute_bourdet_derivative(t, q, L=0.0)

        # Derivative should be negative (declining)
        dq_valid = dq[dq != 0]
        assert len(dq_valid) > 0
        assert np.all(dq_valid < 0)

    def test_with_smoothing(self):
        """Test derivative with smoothing."""
        t = np.linspace(0, 10, 100)
        q = 100.0 * np.exp(-0.1 * t)

        dq_smooth = compute_bourdet_derivative(t, q, L=0.1)

        # Should still be negative
        dq_valid = dq_smooth[dq_smooth != 0]
        assert len(dq_valid) > 0


class TestDiagnosticCurves:
    """Test complete diagnostic curves computation."""

    def test_compute_all_diagnostics(self):
        """Test computing all diagnostics."""
        t = np.linspace(0.1, 100, 100)
        q = 100.0 * np.exp(-0.1 * t)

        diagnostics = compute_diagnostic_curves(t, q)

        assert isinstance(diagnostics, DiagnosticCurves)
        assert len(diagnostics.time) == len(t)
        assert len(diagnostics.rate) == len(q)
        assert len(diagnostics.d) == len(t)
        assert len(diagnostics.b) == len(t)
        assert len(diagnostics.beta) == len(t)
        assert len(diagnostics.derivative) == len(t)

    def test_with_provided_cumulative(self):
        """Test with pre-computed cumulative."""
        t = np.linspace(0.1, 100, 100)
        q = 100.0 * np.exp(-0.1 * t)
        cum = np.cumsum(q) * (t[1] - t[0])  # Simple cumulative

        diagnostics = compute_diagnostic_curves(t, q, cum=cum)

        assert np.allclose(diagnostics.cumulative, cum)

    def test_mismatched_lengths(self):
        """Test error with mismatched array lengths."""
        t = np.linspace(0, 10, 10)
        q = np.linspace(100, 50, 5)  # Different length

        with pytest.raises(ValueError, match="same length"):
            compute_diagnostic_curves(t, q)


class TestIdentifyDeclineType:
    """Test decline type identification."""

    def test_identify_exponential(self):
        """Test identifying exponential decline."""
        t = np.linspace(0.1, 100, 100)
        q = 100.0 * np.exp(-0.1 * t)

        diagnostics = compute_diagnostic_curves(t, q)
        result = identify_decline_type(diagnostics)

        assert "type" in result
        assert "confidence" in result
        assert "mean_b" in result
        # Should identify as exponential (or at least have low b)
        assert result["mean_b"] < 0.3

    def test_identify_harmonic(self):
        """Test identifying harmonic decline."""
        t = np.linspace(0.1, 100, 100)
        q = 100.0 / (1 + 0.1 * t)

        diagnostics = compute_diagnostic_curves(t, q)
        result = identify_decline_type(diagnostics)

        # Should have b close to 1
        assert result["mean_b"] > 0.7

    def test_identify_hyperbolic(self):
        """Test identifying hyperbolic decline."""
        t = np.linspace(0.1, 100, 100)
        b = 0.5
        di = 0.1
        q = 100.0 / np.power(1 + b * di * t, 1 / b)

        diagnostics = compute_diagnostic_curves(t, q)
        result = identify_decline_type(diagnostics)

        # Should have b between 0 and 1
        assert 0.1 < result["mean_b"] < 0.9


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics with models."""

    def test_diagnostics_with_exponential_model(self):
        """Test diagnostics match exponential model expectations."""
        from decline_curve.models_arps import ExponentialArps

        model = ExponentialArps()
        t = np.linspace(0.1, 100, 100)
        params = {"qi": 100.0, "di": 0.1}

        q = model.rate(t, params)
        diagnostics = compute_diagnostic_curves(t, q)

        # D should be approximately constant at di
        D_valid = diagnostics.d[diagnostics.d > 0]
        if len(D_valid) > 0:
            assert np.abs(np.mean(D_valid) - 0.1) < 0.05

    def test_diagnostics_with_hyperbolic_model(self):
        """Test diagnostics match hyperbolic model expectations."""
        from decline_curve.models_arps import HyperbolicArps

        model = HyperbolicArps()
        t = np.linspace(0.1, 100, 100)
        params = {"qi": 100.0, "di": 0.1, "b": 0.5}

        q = model.rate(t, params)
        diagnostics = compute_diagnostic_curves(t, q)

        # b diagnostic should be approximately constant
        b_valid = diagnostics.b[(diagnostics.b != 0) & np.isfinite(diagnostics.b)]
        if len(b_valid) > 0:
            # Should have some variation but be in reasonable range
            assert 0 < np.mean(b_valid) < 1
