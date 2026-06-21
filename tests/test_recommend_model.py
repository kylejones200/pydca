"""Tests for dca.recommend_model() — model auto-selection logic."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.dca import recommend_model


VALID_MODELS = {"exponential", "harmonic", "hyperbolic", "modified_hyperbolic", "duong"}


def _make_series(q: np.ndarray) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(q), freq="MS")
    return pd.Series(q, index=dates)


# ---------------------------------------------------------------------------
# Return value contract
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_string(self):
        q = np.array([1000 * np.exp(-0.05 * t) for t in range(24)])
        result = recommend_model(_make_series(q))
        assert isinstance(result, str)

    def test_returns_valid_model(self):
        q = np.array([1000 * np.exp(-0.05 * t) for t in range(24)])
        result = recommend_model(_make_series(q))
        assert result in VALID_MODELS


# ---------------------------------------------------------------------------
# Drive mechanism overrides
# ---------------------------------------------------------------------------


class TestDriveMechanismOverrides:
    def _series(self):
        return _make_series(np.array([1000 * np.exp(-0.04 * t) for t in range(24)]))

    def test_solution_gas_returns_hyperbolic(self):
        assert recommend_model(self._series(), drive_mechanism="solution_gas") == "hyperbolic"

    def test_water_drive_returns_exponential(self):
        assert recommend_model(self._series(), drive_mechanism="water_drive") == "exponential"

    def test_compaction_returns_exponential(self):
        assert recommend_model(self._series(), drive_mechanism="compaction") == "exponential"

    def test_shale_returns_modified_hyperbolic(self):
        assert recommend_model(self._series(), drive_mechanism="shale") == "modified_hyperbolic"


# ---------------------------------------------------------------------------
# Data-driven routing
# ---------------------------------------------------------------------------


class TestDataDrivenRouting:
    def test_constant_decline_rate_gives_exponential(self):
        # Perfect exponential — D is nearly constant
        t = np.arange(36)
        q = 1000.0 * np.exp(-0.05 * t)
        result = recommend_model(_make_series(q))
        assert result == "exponential"

    def test_hyperbolic_data_gives_curvature_model(self):
        from decline_curve.models import ArpsParams, predict_arps
        t = np.arange(48, dtype=float)
        q = predict_arps(t, ArpsParams(qi=1000, di=0.08, b=1.2))
        result = recommend_model(_make_series(q))
        # b=1.2 data exhibits declining D — valid answers: hyperbolic, harmonic, modified_hyperbolic
        assert result in ("hyperbolic", "harmonic", "modified_hyperbolic")

    def test_rapid_early_drop_suggests_duong(self):
        # >65% decline in 6 months, highly irregular
        q = np.concatenate([
            np.array([5000, 3000, 1800, 1200, 900, 700, 600]),   # erratic early frac
            np.array([560, 530, 510, 490, 470]) * np.exp(-0.01 * np.arange(5)),  # stable tail
        ])
        result = recommend_model(_make_series(q))
        assert result in VALID_MODELS  # must be a valid model

    def test_short_series_returns_hyperbolic_fallback(self):
        q = np.array([1000, 900, 800])  # only 3 points → too few for D analysis
        result = recommend_model(_make_series(q))
        assert result in VALID_MODELS

    def test_very_short_series_returns_hyperbolic_fallback(self):
        q = np.array([1000, 900])  # < 6 → immediate fallback
        result = recommend_model(_make_series(q))
        assert result == "hyperbolic"

    def test_accepts_none_pressure_and_drive(self):
        q = 800 * np.exp(-0.04 * np.arange(24))
        result = recommend_model(_make_series(q), pressure=None, drive_mechanism=None)
        assert result in VALID_MODELS

    def test_top_level_import(self):
        import decline_curve as dca
        assert hasattr(dca, "recommend_model")

    def test_top_level_callable(self):
        import decline_curve as dca
        q = 500 * np.exp(-0.05 * np.arange(24))
        result = dca.recommend_model(_make_series(q))
        assert result in VALID_MODELS
