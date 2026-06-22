"""Tests for PRMS 1P/2P/3P reserves classification."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.prms import (
    ReservesClassification,
    _eur_from_forecast_array,
    classify_reserves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(qi=1000.0, d=0.025, n=36):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    q = np.array([qi * np.exp(-d * t) for t in range(n)])
    return pd.Series(q, index=dates)


_SERIES = _make_series()


# ---------------------------------------------------------------------------
# _eur_from_forecast_array
# ---------------------------------------------------------------------------


class TestEurFromForecastArray:
    def test_no_econ_limit(self):
        q = np.array([100.0, 90.0, 80.0, 70.0])
        eur = _eur_from_forecast_array(q)
        assert eur == pytest.approx(340.0)

    def test_econ_limit_truncates(self):
        q = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
        eur = _eur_from_forecast_array(q, econ_limit=75.0)
        # Only months where q >= 75: indices 0,1,2 → sum = 270
        assert eur == pytest.approx(270.0)

    def test_econ_limit_above_all_returns_zero(self):
        q = np.array([50.0, 40.0, 30.0])
        eur = _eur_from_forecast_array(q, econ_limit=200.0)
        assert eur == pytest.approx(0.0)

    def test_empty_array_returns_zero(self):
        eur = _eur_from_forecast_array(np.array([]), econ_limit=0.0)
        assert eur == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ReservesClassification dataclass
# ---------------------------------------------------------------------------


class TestReservesClassification:
    def _make(self):
        dist = np.linspace(1000, 5000, 100)
        return ReservesClassification(p1=2000, p2=3000, p3=4000, eur_distribution=dist)

    def test_uncertainty_ratio(self):
        rc = self._make()
        assert rc.uncertainty_ratio == pytest.approx(4000 / 2000)

    def test_uncertainty_ratio_zero_p1(self):
        rc = ReservesClassification(p1=0, p2=100, p3=200, eur_distribution=np.array([]))
        assert np.isnan(rc.uncertainty_ratio)

    def test_to_dict(self):
        rc = self._make()
        d = rc.to_dict()
        assert "p1" in d and "p2" in d and "p3" in d
        assert d["p1"] == pytest.approx(2000)

    def test_to_series(self):
        rc = self._make()
        s = rc.to_series()
        assert isinstance(s, pd.Series)
        assert "1P (P90)" in s.index


# ---------------------------------------------------------------------------
# classify_reserves
# ---------------------------------------------------------------------------


class TestClassifyReserves:
    def test_returns_classification(self):
        rc = classify_reserves(_SERIES, kind="hyperbolic", horizon=120, n_draws=50, seed=42)
        assert isinstance(rc, ReservesClassification)

    def test_p1_le_p2_le_p3(self):
        # P90 ≤ P50 ≤ P10  (conservative ≤ best ≤ optimistic)
        rc = classify_reserves(_SERIES, kind="hyperbolic", horizon=120, n_draws=100, seed=1)
        assert rc.p1 <= rc.p2 <= rc.p3

    def test_all_positive(self):
        rc = classify_reserves(_SERIES, kind="hyperbolic", horizon=120, n_draws=50, seed=42)
        assert rc.p1 > 0
        assert rc.p2 > 0
        assert rc.p3 > 0

    def test_distribution_has_correct_length(self):
        rc = classify_reserves(_SERIES, kind="hyperbolic", horizon=60, n_draws=80, seed=7)
        assert len(rc.eur_distribution) == 80

    def test_distribution_is_sorted(self):
        rc = classify_reserves(_SERIES, kind="hyperbolic", horizon=60, n_draws=80, seed=7)
        assert np.all(np.diff(rc.eur_distribution) >= 0)

    def test_econ_limit_reduces_eur(self):
        rc_no_limit = classify_reserves(
            _SERIES, kind="hyperbolic", horizon=120, n_draws=50, econ_limit=0.0, seed=42
        )
        rc_with_limit = classify_reserves(
            _SERIES, kind="hyperbolic", horizon=120, n_draws=50, econ_limit=50.0, seed=42
        )
        assert rc_with_limit.p2 <= rc_no_limit.p2

    def test_longer_horizon_larger_eur(self):
        rc_short = classify_reserves(
            _SERIES, kind="hyperbolic", horizon=60, n_draws=50, seed=42
        )
        rc_long = classify_reserves(
            _SERIES, kind="hyperbolic", horizon=240, n_draws=50, seed=42
        )
        assert rc_long.p2 >= rc_short.p2

    def test_metadata_stored(self):
        rc = classify_reserves(
            _SERIES, model="arps", kind="hyperbolic", horizon=60, n_draws=30, seed=1
        )
        assert rc.model == "arps"
        assert rc.kind == "hyperbolic"
        assert rc.n_draws == 30

    def test_seed_reproducibility(self):
        rc1 = classify_reserves(_SERIES, kind="hyperbolic", horizon=60, n_draws=50, seed=99)
        rc2 = classify_reserves(_SERIES, kind="hyperbolic", horizon=60, n_draws=50, seed=99)
        assert rc1.p1 == pytest.approx(rc2.p1)
        assert rc1.p2 == pytest.approx(rc2.p2)
        assert rc1.p3 == pytest.approx(rc2.p3)

    def test_series_from_top_level_import(self):
        import decline_curve as dca
        rc = dca.classify_reserves(_SERIES, horizon=60, n_draws=30, seed=0)
        assert isinstance(rc, dca.ReservesClassification)


class TestMinHistoryGuard:
    """Gap 3: classify_reserves enforces minimum history length."""

    @staticmethod
    def _series(n):
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        q = 1000.0 * np.exp(-0.03 * np.arange(n))
        return pd.Series(q, index=dates)

    def test_raises_below_min_months(self):
        short = self._series(6)
        with pytest.raises(ValueError, match="at least 12 months"):
            classify_reserves(short, horizon=60, n_draws=30)

    def test_custom_min_months_override(self):
        """Caller can lower the bar; 6 months should pass with min_months=6."""
        short = self._series(6)
        rc = classify_reserves(short, horizon=60, n_draws=30, seed=0, min_months=6)
        assert rc.p1 > 0

    def test_warn_below_18_months(self):
        """14 months passes the 12-month floor but should trigger a UserWarning."""
        borderline = self._series(14)
        with pytest.warns(UserWarning, match="14 months"):
            classify_reserves(borderline, horizon=60, n_draws=30, seed=0)

    def test_no_warn_at_18_months(self):
        """18+ months must not trigger the low-history warning."""
        good = self._series(24)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            classify_reserves(good, horizon=60, n_draws=30, seed=0)
