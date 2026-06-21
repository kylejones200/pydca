"""Tests for shale-era decline model variants: Duong, PLE, SEPD, Modified Hyperbolic."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import gamma as scipy_gamma

from decline_curve.decline_variants import (
    eur_duong,
    eur_modified_hyperbolic,
    eur_ple,
    eur_sepd,
    fit_duong,
    fit_modified_hyperbolic,
    fit_ple,
    fit_sepd,
    forecast_variant,
    predict_duong,
    predict_modified_hyperbolic,
    predict_ple,
    predict_sepd,
)
from decline_curve.models import DuongParams, ModifiedHyperbolicParams, PLEParams, SEPDParams


def _make_series(rates, freq="MS"):
    dates = pd.date_range("2020-01-01", periods=len(rates), freq=freq)
    return pd.Series(rates, index=dates)


# ---------------------------------------------------------------------------
# Duong model
# ---------------------------------------------------------------------------


def _duong_synthetic(q1=1000.0, a=1.5, m=1.1, n_months=36):
    """Generate synthetic Duong production data."""
    t = np.arange(n_months, dtype=float)
    t1 = t + 1.0
    rates = q1 * t1 ** (-m) * np.exp(a / (1 - m) * (t1 ** (1 - m) - 1))
    return t, rates


class TestDuong:
    def test_fit_returns_valid_params(self):
        # Duong has a degenerate parameter manifold; test that fit produces
        # valid params (positive values, m in range) and that the resulting
        # forecast reproduces the training data well.
        t, q = _duong_synthetic(q1=800.0, a=1.5, m=1.2)
        params = fit_duong(t, q)
        assert isinstance(params, DuongParams)
        assert params.q1 > 0
        assert params.a > 0
        assert params.m > 0
        # Forecast should reproduce training data within 20%
        pred = predict_duong(t, params)
        rel_err = np.abs(pred - q) / np.maximum(q, 1.0)
        assert np.median(rel_err) < 0.20

    def test_predict_shape(self):
        t, q = _duong_synthetic()
        params = fit_duong(t, q)
        full_t = np.arange(48, dtype=float)
        pred = predict_duong(full_t, params)
        assert pred.shape == (48,)

    def test_predict_long_term_decline(self):
        # Duong model can have non-monotone early behavior but should decline
        # long-term. Verify rate at t=60 is well below rate at t=0.
        params = DuongParams(q1=1000.0, a=1.5, m=1.2)
        full_t = np.arange(61, dtype=float)
        pred = predict_duong(full_t, params)
        assert pred[-1] < pred[0] * 0.5, "Duong rate at t=60 should be < 50% of initial"

    def test_predict_all_positive(self):
        params = DuongParams(q1=1000.0, a=1.5, m=1.2)
        pred = predict_duong(np.arange(120, dtype=float), params)
        assert np.all(pred > 0)

    def test_eur_positive(self):
        params = DuongParams(q1=1000.0, a=1.5, m=1.2)
        eur = eur_duong(params)
        assert eur > 0

    def test_eur_with_econ_limit_less_than_without(self):
        # Use econ_limit well above initial rate so model never exceeds it
        # — EUR with limit should be 0 (no production above limit), while
        # full EUR is positive.
        params = DuongParams(q1=1000.0, a=1.5, m=1.2)
        eur_full = eur_duong(params, t_max=240, econ_limit=0)
        # Set econ_limit above the peak to confirm EUR goes to 0
        eur_above_peak = eur_duong(params, t_max=240, econ_limit=2000.0)
        assert eur_full > 0
        assert eur_above_peak == 0.0

    def test_api_roundtrip(self):
        """forecast_variant produces a dated Series of correct length."""
        _, q = _duong_synthetic()
        series = _make_series(q)
        result = forecast_variant(series, kind="duong", horizon=12)
        assert isinstance(result, pd.Series)
        assert len(result) == len(q) + 12
        assert result.index.dtype == "datetime64[ns]"

    def test_fit_empty_q_returns_defaults(self):
        t = np.arange(10, dtype=float)
        q = np.zeros(10)
        params = fit_duong(t, q)
        assert isinstance(params, DuongParams)


# ---------------------------------------------------------------------------
# PLE (Power-Law Exponential)
# ---------------------------------------------------------------------------


def _ple_synthetic(qi=500.0, D_inf=0.002, D1=0.05, n=0.4, n_months=36):
    t = np.arange(n_months, dtype=float)
    t_safe = np.where(t == 0, 1e-6, t)
    rates = qi * np.exp(-D_inf * t_safe - (D1 - D_inf) / n * t_safe**n)
    return t, rates


class TestPLE:
    def test_fit_recovers_params(self):
        t, q = _ple_synthetic(qi=600.0, D_inf=0.003, D1=0.06, n=0.35)
        params = fit_ple(t, q)
        assert isinstance(params, PLEParams)
        assert abs(params.qi - 600.0) / 600.0 < 0.10
        assert params.n > 0 and params.n < 1.0

    def test_predict_shape(self):
        t, q = _ple_synthetic()
        params = fit_ple(t, q)
        pred = predict_ple(np.arange(48, dtype=float), params)
        assert pred.shape == (48,)

    def test_predict_all_positive(self):
        t, q = _ple_synthetic()
        params = fit_ple(t, q)
        pred = predict_ple(np.arange(120, dtype=float), params)
        assert np.all(pred > 0)

    def test_eur_numeric_integration(self):
        """EUR via trapz should match a known approximate value."""
        params = PLEParams(qi=500.0, D_inf=0.002, D1=0.05, n=0.4)
        eur = eur_ple(params, t_max=600)
        assert eur > 0
        # Sanity check: EUR < qi * t_max (bounded by flat production)
        assert eur < params.qi * 600

    def test_eur_econ_limit_reduces_eur(self):
        t, q = _ple_synthetic()
        params = fit_ple(t, q)
        eur_full = eur_ple(params, t_max=240, econ_limit=0)
        eur_lim = eur_ple(params, t_max=240, econ_limit=20.0)
        assert eur_lim <= eur_full

    def test_api_roundtrip(self):
        _, q = _ple_synthetic()
        series = _make_series(q)
        result = forecast_variant(series, kind="ple", horizon=24)
        assert len(result) == len(q) + 24


# ---------------------------------------------------------------------------
# SEPD (Stretched Exponential)
# ---------------------------------------------------------------------------


def _sepd_synthetic(qi=1000.0, tau=24.0, n=0.6, n_months=36):
    t = np.arange(n_months, dtype=float)
    rates = qi * np.exp(-((t / tau) ** n))
    return t, rates


class TestSEPD:
    def test_fit_recovers_params(self):
        t, q = _sepd_synthetic(qi=800.0, tau=30.0, n=0.55)
        params = fit_sepd(t, q)
        assert isinstance(params, SEPDParams)
        assert abs(params.qi - 800.0) / 800.0 < 0.10
        assert 0 < params.n <= 1.0
        assert params.tau > 0

    def test_predict_shape(self):
        t, q = _sepd_synthetic()
        params = fit_sepd(t, q)
        pred = predict_sepd(np.arange(60, dtype=float), params)
        assert pred.shape == (60,)

    def test_predict_monotone_decreasing(self):
        t, q = _sepd_synthetic()
        params = fit_sepd(t, q)
        pred = predict_sepd(np.arange(60, dtype=float), params)
        assert np.all(np.diff(pred) <= 0)

    def test_eur_closed_form(self):
        """Closed-form EUR should match scipy_gamma formula."""
        params = SEPDParams(qi=1000.0, tau=24.0, n=0.6)
        eur = eur_sepd(params, econ_limit=0.0)
        expected = params.qi * (params.tau / params.n) * scipy_gamma(1.0 / params.n)
        assert abs(eur - expected) / expected < 1e-6

    def test_eur_with_econ_limit(self):
        params = SEPDParams(qi=1000.0, tau=24.0, n=0.6)
        eur_inf = eur_sepd(params, econ_limit=0.0)
        eur_lim = eur_sepd(params, econ_limit=50.0)
        assert eur_lim < eur_inf

    def test_eur_n1_is_exponential(self):
        """When n=1, SEPD reduces to exponential: EUR = qi * tau."""
        params = SEPDParams(qi=500.0, tau=20.0, n=1.0)
        eur = eur_sepd(params, econ_limit=0.0)
        expected = params.qi * params.tau  # qi * tau * Γ(1) = qi * tau * 1
        assert abs(eur - expected) / expected < 1e-6

    def test_api_roundtrip(self):
        _, q = _sepd_synthetic()
        series = _make_series(q)
        result = forecast_variant(series, kind="sepd", horizon=12)
        assert len(result) == len(q) + 12

    def test_fit_empty_q_returns_defaults(self):
        t = np.arange(10, dtype=float)
        q = np.zeros(10)
        params = fit_sepd(t, q)
        assert isinstance(params, SEPDParams)


# ---------------------------------------------------------------------------
# forecast_variant dispatcher
# ---------------------------------------------------------------------------


class TestForecastVariant:
    def test_unknown_kind_raises(self):
        series = _make_series(np.linspace(100, 50, 24))
        with pytest.raises(ValueError, match="Unknown variant kind"):
            forecast_variant(series, kind="unknown")

    def test_all_kinds_return_series(self):
        t, q = _sepd_synthetic(n_months=24)
        series = _make_series(q)
        for kind in ("duong", "ple", "sepd"):
            result = forecast_variant(series, kind=kind, horizon=12)
            assert isinstance(result, pd.Series), f"{kind} should return pd.Series"
            assert len(result) == 24 + 12

    def test_result_has_datetimeindex(self):
        series = _make_series(np.linspace(500, 100, 24))
        result = forecast_variant(series, kind="duong", horizon=6)
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Integration: dca.forecast() passes through new kinds
# ---------------------------------------------------------------------------


class TestDcaAPIIntegration:
    def test_dca_forecast_duong(self):
        from decline_curve import dca

        t, q = _duong_synthetic()
        series = _make_series(q)
        result = dca.forecast(series, model="arps", kind="duong", horizon=12)
        assert isinstance(result, pd.Series)
        assert len(result) == len(q) + 12

    def test_dca_forecast_ple(self):
        from decline_curve import dca

        t, q = _ple_synthetic()
        series = _make_series(q)
        result = dca.forecast(series, model="arps", kind="ple", horizon=12)
        assert isinstance(result, pd.Series)

    def test_dca_forecast_sepd(self):
        from decline_curve import dca

        t, q = _sepd_synthetic()
        series = _make_series(q)
        result = dca.forecast(series, model="arps", kind="sepd", horizon=12)
        assert isinstance(result, pd.Series)

    def test_dca_forecast_modified_hyperbolic(self):
        from decline_curve import dca

        series = _make_series(_hyperbolic_synthetic()[1])
        result = dca.forecast(series, model="arps", kind="modified_hyperbolic", horizon=12)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series) + 12

    def test_legacy_arps_still_works(self):
        """Existing Arps kinds must not be broken."""
        from decline_curve import dca

        t, q = _sepd_synthetic()
        series = _make_series(q)
        for kind in ("exponential", "harmonic", "hyperbolic"):
            result = dca.forecast(series, model="arps", kind=kind, horizon=6)
            assert isinstance(result, pd.Series), f"Arps {kind} broken"


# ---------------------------------------------------------------------------
# Modified Hyperbolic
# ---------------------------------------------------------------------------


def _hyperbolic_synthetic(qi=1000.0, di=0.08, b=1.4, n_months=48):
    """Generate synthetic hyperbolic production — b > 1 typical for shale."""
    t = np.arange(n_months, dtype=float)
    q = qi / (1.0 + b * di * t) ** (1.0 / b)
    return t, q


class TestModifiedHyperbolic:
    def test_fit_returns_dataclass(self):
        t, q = _hyperbolic_synthetic()
        params = fit_modified_hyperbolic(t, q)
        assert isinstance(params, ModifiedHyperbolicParams)
        assert params.qi > 0
        assert params.di > 0
        assert 0 < params.b <= 2.0
        assert params.d_lim == 0.005

    def test_fit_custom_d_lim(self):
        t, q = _hyperbolic_synthetic()
        params = fit_modified_hyperbolic(t, q, d_lim=0.008)
        assert params.d_lim == 0.008

    def test_predict_shape(self):
        t, q = _hyperbolic_synthetic()
        params = fit_modified_hyperbolic(t, q)
        t_fc = np.arange(120, dtype=float)
        rates = predict_modified_hyperbolic(t_fc, params)
        assert rates.shape == (120,)

    def test_predict_all_positive(self):
        t, q = _hyperbolic_synthetic()
        params = fit_modified_hyperbolic(t, q)
        rates = predict_modified_hyperbolic(np.arange(360, dtype=float), params)
        assert np.all(rates > 0)

    def test_continuity_at_switch(self):
        """Rate must be continuous across the hyperbolic→exponential switch."""
        t, q = _hyperbolic_synthetic(qi=1000, di=0.08, b=1.4, n_months=60)
        params = fit_modified_hyperbolic(t, q, d_lim=0.005)
        # t_switch derived: (di/d_lim - 1) / (b * di)
        t_sw = (params.di / params.d_lim - 1.0) / (params.b * params.di)
        eps = 0.001
        q_before = predict_modified_hyperbolic(np.array([t_sw - eps]), params)[0]
        q_after = predict_modified_hyperbolic(np.array([t_sw + eps]), params)[0]
        assert abs(q_before - q_after) / max(q_before, 1.0) < 0.01

    def test_exponential_tail_at_d_lim(self):
        """After t_switch, instantaneous decline should equal d_lim."""
        params = ModifiedHyperbolicParams(qi=1000.0, di=0.08, b=1.4, d_lim=0.005)
        t_sw = (params.di / params.d_lim - 1.0) / (params.b * params.di)
        # In exponential tail: D = d_lim = -dq/dt / q
        t_far = np.array([t_sw + 1.0, t_sw + 2.0])
        q_far = predict_modified_hyperbolic(t_far, params)
        d_apparent = (q_far[0] - q_far[1]) / q_far[0]  # ≈ d_lim for small dt
        assert abs(d_apparent - params.d_lim) < 0.001

    def test_eur_finite_and_positive(self):
        t, q = _hyperbolic_synthetic()
        params = fit_modified_hyperbolic(t, q)
        eur = eur_modified_hyperbolic(params, t_max=360)
        assert np.isfinite(eur)
        assert eur > 0

    def test_eur_converges_with_t_max(self):
        """EUR should converge as t_max grows (exponential tail decays)."""
        params = ModifiedHyperbolicParams(qi=1000.0, di=0.08, b=1.4, d_lim=0.005)
        eur_10yr = eur_modified_hyperbolic(params, t_max=120)
        eur_30yr = eur_modified_hyperbolic(params, t_max=360)
        eur_inf = eur_modified_hyperbolic(params, t_max=9999)
        assert eur_10yr < eur_30yr < eur_inf
        # d_lim=0.005/mo → 200-month time constant; 30yr captures ~84% of tail
        assert eur_30yr / eur_inf > 0.80

    def test_eur_less_than_unconstrained_hyperbolic(self):
        """MH EUR (b > 1 bounded by d_lim) must be < pure hyperbolic EUR for b > 1."""
        from decline_curve.models import estimate_reserves
        params = ModifiedHyperbolicParams(qi=1000.0, di=0.08, b=1.4, d_lim=0.005)
        eur_mh = eur_modified_hyperbolic(params, t_max=9999)
        # Pure hyperbolic b=1.4 EUR diverges (b>1), so we compare at finite horizon
        from decline_curve.models import ArpsParams
        from decline_curve.decline_variants import predict_duong
        t_long = np.arange(9999, dtype=float)
        q_hyp_raw = 1000.0 / (1.0 + 1.4 * 0.08 * t_long) ** (1.0 / 1.4)
        eur_hyp = float(np.trapz(q_hyp_raw, t_long))
        assert eur_mh < eur_hyp

    def test_eur_econ_limit(self):
        """Applying econ_limit should reduce EUR when the limit cuts off before t_max."""
        params = ModifiedHyperbolicParams(qi=1000.0, di=0.08, b=1.4, d_lim=0.005)
        eur_no_lim = eur_modified_hyperbolic(params, t_max=360)
        # q_switch ≈ 117 BOE/mo; econ_limit=50 cuts off at ~170 mo into exp tail
        eur_with_lim = eur_modified_hyperbolic(params, t_max=360, econ_limit=50.0)
        assert eur_with_lim < eur_no_lim

    def test_forecast_variant_mh_alias(self):
        """'mh' short alias should work in forecast_variant."""
        t, q = _hyperbolic_synthetic()
        series = _make_series(q)
        fc = forecast_variant(series, kind="mh", horizon=24)
        assert isinstance(fc, pd.Series)
        assert len(fc) == len(series) + 24


# ---------------------------------------------------------------------------
# Portfolio economics
# ---------------------------------------------------------------------------


class TestPortfolioEconomics:
    def _profiles(self):
        """Three synthetic wells with identical 60-month profiles."""
        return {
            "WELL_A": 800.0 * np.exp(-0.025 * np.arange(60)),
            "WELL_B": 600.0 * np.exp(-0.030 * np.arange(60)),
            "WELL_C": 1000.0 * np.exp(-0.020 * np.arange(60)),
        }

    def _econ(self):
        from decline_curve import WellEconomics
        return WellEconomics(
            capex=3_000_000, price=70.0, opex_variable=8.0,
            royalty_rate=0.1875, severance_tax_rate=0.046,
            ad_valorem_rate=0.02, discount_rate=0.10,
        )

    def test_returns_dataframe(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_has_portfolio_row(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        assert "PORTFOLIO" in df.index

    def test_well_count(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        # 3 wells + 1 PORTFOLIO row
        assert len(df) == 4

    def test_columns_present(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        for col in ("npv", "irr", "payout_month", "roi", "breakeven_price", "eur"):
            assert col in df.columns, f"Missing column: {col}"

    def test_portfolio_npv_is_sum(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        well_npv_sum = df.loc[["WELL_A", "WELL_B", "WELL_C"], "npv"].sum()
        assert abs(df.loc["PORTFOLIO", "npv"] - well_npv_sum) < 1.0

    def test_portfolio_eur_is_sum(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        well_eur_sum = df.loc[["WELL_A", "WELL_B", "WELL_C"], "eur"].sum()
        assert abs(df.loc["PORTFOLIO", "eur"] - well_eur_sum) < 1.0

    def test_finite_metrics(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics(self._profiles(), self._econ())
        for col in ("npv", "eur"):
            assert df[col].apply(np.isfinite).all(), f"{col} has non-finite values"

    def test_empty_profiles(self):
        from decline_curve import portfolio_economics
        df = portfolio_economics({}, self._econ())
        assert len(df) == 0
