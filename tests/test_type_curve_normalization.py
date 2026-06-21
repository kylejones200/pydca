"""Tests for type curve normalization (Fetkovich-style)."""

import numpy as np
import pytest

from decline_curve.models import ArpsParams, predict_arps
from decline_curve.type_curve_normalization import (
    TypeCurveMatch,
    denormalize_match,
    generate_arps_type_curve,
    match_type_curve,
    normalize_production_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arps_data(qi=1000.0, di=0.08, b=1.2, n=36):
    t = np.arange(n, dtype=float)
    params = ArpsParams(qi=qi, di=di, b=b)
    q = predict_arps(t, params)
    return t, q


# ---------------------------------------------------------------------------
# generate_arps_type_curve
# ---------------------------------------------------------------------------


class TestGenerateArpsTypeCurve:
    def test_returns_two_arrays(self):
        t = np.linspace(0, 10, 50)
        t_out, q_out = generate_arps_type_curve(1.0, 0.5, 1.2, t)
        assert len(t_out) == 50
        assert len(q_out) == 50

    def test_starts_at_qi(self):
        t = np.linspace(0, 10, 50)
        _, q = generate_arps_type_curve(1.0, 0.5, 1.2, t)
        assert q[0] == pytest.approx(1.0, rel=1e-4)

    def test_monotone_declining(self):
        t = np.linspace(0, 10, 50)
        _, q = generate_arps_type_curve(1.0, 0.5, 1.2, t)
        assert np.all(np.diff(q) <= 0)

    def test_exponential_b_zero(self):
        t = np.linspace(0, 5, 30)
        _, q = generate_arps_type_curve(1.0, 0.3, 0.0, t)
        expected = np.exp(-0.3 * t)
        np.testing.assert_allclose(q, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# normalize_production_data
# ---------------------------------------------------------------------------


class TestNormalizeProductionData:
    def test_output_shapes(self):
        t = np.arange(24, dtype=float)
        q = np.ones(24) * 500
        t_n, q_n, factors = normalize_production_data(t, q, qi_ref=500, di_ref=0.1)
        assert len(t_n) == 24
        assert len(q_n) == 24

    def test_normalized_rate_at_qi_ref(self):
        t = np.arange(10, dtype=float)
        q = np.full(10, 1000.0)
        _, q_n, _ = normalize_production_data(t, q, qi_ref=1000.0, di_ref=0.1)
        assert q_n[0] == pytest.approx(1.0)

    def test_factors_dict_keys(self):
        t = np.arange(5, dtype=float)
        q = np.ones(5) * 500
        _, _, factors = normalize_production_data(t, q, 500, 0.05)
        assert "qi_ref" in factors
        assert "di_ref" in factors

    def test_invalid_qi_ref_raises(self):
        with pytest.raises(ValueError, match="qi_ref must be positive"):
            normalize_production_data(np.arange(5.0), np.ones(5), qi_ref=0, di_ref=0.1)

    def test_invalid_di_ref_raises(self):
        with pytest.raises(ValueError, match="di_ref must be positive"):
            normalize_production_data(np.arange(5.0), np.ones(5), qi_ref=100, di_ref=-0.1)

    def test_time_scaling(self):
        t = np.array([0.0, 1.0, 2.0])
        q = np.ones(3) * 200
        t_n, _, _ = normalize_production_data(t, q, qi_ref=200, di_ref=0.2)
        np.testing.assert_allclose(t_n, np.array([0.0, 0.2, 0.4]))


# ---------------------------------------------------------------------------
# match_type_curve
# ---------------------------------------------------------------------------


class TestMatchTypeCurve:
    def test_returns_type_curve_match(self):
        t, q = _arps_data()
        match = match_type_curve(t, q, b_values=np.arange(0.5, 2.1, 0.5))
        assert isinstance(match, TypeCurveMatch)

    def test_matched_curve_same_length_as_input(self):
        t, q = _arps_data(n=24)
        match = match_type_curve(t, q, b_values=[1.0, 1.2, 1.5])
        assert len(match.matched_curve) == 24

    def test_b_sweep_low_rmse(self):
        t, q = _arps_data(qi=1000, di=0.08, b=1.2, n=48)
        match = match_type_curve(t, q, b_values=np.arange(0.3, 2.0, 0.1))
        # Should recover close to original data
        assert match.match_error < 0.15 * q.mean()

    def test_b_sweep_recovers_approximate_b(self):
        t, q = _arps_data(qi=1000, di=0.08, b=1.2, n=48)
        match = match_type_curve(t, q, b_values=np.arange(0.5, 2.1, 0.1))
        # b should be within 0.3 of true b=1.2
        assert abs(match.matched_params.b - 1.2) < 0.3

    def test_free_b_fit(self):
        t, q = _arps_data(qi=800, di=0.06, b=0.9, n=36)
        match = match_type_curve(t, q, b_values=None)
        assert isinstance(match, TypeCurveMatch)
        assert match.matched_params.qi > 0
        assert match.matched_params.di > 0

    def test_correlation_in_minus1_to_1(self):
        t, q = _arps_data()
        match = match_type_curve(t, q, b_values=[0.8, 1.0, 1.2, 1.5])
        assert -1.0 <= match.correlation <= 1.0

    def test_high_correlation_for_clean_data(self):
        t, q = _arps_data(qi=1000, di=0.07, b=1.0, n=36)
        match = match_type_curve(t, q, b_values=np.arange(0.5, 2.0, 0.25))
        assert match.correlation > 0.95

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3 data points"):
            match_type_curve(np.array([0.0, 1.0]), np.array([100.0, 90.0]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            match_type_curve(np.arange(5.0), np.ones(4))


# ---------------------------------------------------------------------------
# denormalize_match
# ---------------------------------------------------------------------------


class TestDenormalizeMatch:
    def test_denormalize_restores_qi(self):
        qi_ref, di_ref = 1000.0, 0.1
        t_true = np.arange(24, dtype=float)
        q_true = predict_arps(t_true, ArpsParams(qi=qi_ref, di=di_ref, b=1.2))

        t_n, q_n, factors = normalize_production_data(t_true, q_true, qi_ref, di_ref)
        match = match_type_curve(t_n, q_n, b_values=np.arange(0.5, 2.0, 0.25))
        restored = denormalize_match(match, factors)

        assert restored.qi == pytest.approx(match.matched_params.qi * qi_ref)
        assert restored.di == pytest.approx(match.matched_params.di * di_ref)
        assert restored.b == pytest.approx(match.matched_params.b)

    def test_roundtrip_qi_close_to_true(self):
        qi_ref, di_ref = 1500.0, 0.08
        t = np.arange(36, dtype=float)
        q = predict_arps(t, ArpsParams(qi=qi_ref, di=di_ref, b=1.1))

        t_n, q_n, factors = normalize_production_data(t, q, qi_ref, di_ref)
        match = match_type_curve(t_n, q_n, b_values=np.arange(0.5, 2.0, 0.2))
        restored = denormalize_match(match, factors)

        assert abs(restored.qi - qi_ref) / qi_ref < 0.10  # within 10%


# ---------------------------------------------------------------------------
# Top-level import check
# ---------------------------------------------------------------------------


def test_top_level_imports():
    import decline_curve as dca
    assert hasattr(dca, "TypeCurveMatch")
    assert hasattr(dca, "normalize_production_data")
    assert hasattr(dca, "match_type_curve")
    assert hasattr(dca, "denormalize_match")
    assert hasattr(dca, "generate_arps_type_curve")
