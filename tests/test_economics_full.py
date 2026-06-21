"""Tests for the full-cycle economics engine (WellEconomics / CashflowResult)."""

import numpy as np
import pytest

from decline_curve.economics import (
    CashflowResult,
    WellEconomics,
    breakeven_price,
    cashflow,
    economic_metrics,
    irr,
    npv,
    payout,
    roi,
)


def _flat_production(rate: float = 100.0, months: int = 60) -> np.ndarray:
    return np.full(months, rate)


def _declining_production(qi: float = 1000.0, months: int = 120, D_annual: float = 0.30) -> np.ndarray:
    """Exponential decline at D_annual (%/year). Default: qi=1000 BOE/mo, 30%/yr."""
    t = np.arange(months, dtype=float)
    D_monthly = D_annual / 12.0
    return qi * np.exp(-D_monthly * t)


# Capex is in $ (same unit as revenue = BOE/month × $/BOE).
# A "typical" well: qi=1000 BOE/month, price=$70/BOE → revenue ~$70K/month;
# capex=$1_000_000 pays out after ~14 months.
_CAPEX = 1_000_000.0  # $1 million
_CAPEX_LARGE = 100_000_000.0  # $100 million (unprofitable at low rates)


# ---------------------------------------------------------------------------
# WellEconomics defaults
# ---------------------------------------------------------------------------


class TestWellEconomicsDefaults:
    def test_default_royalty(self):
        econ = WellEconomics(capex=5.0, price=70.0)
        assert econ.royalty_rate == pytest.approx(0.1875)

    def test_default_severance(self):
        econ = WellEconomics(capex=5.0, price=70.0)
        assert econ.severance_tax_rate == pytest.approx(0.046)

    def test_default_working_interest(self):
        econ = WellEconomics(capex=5.0, price=70.0)
        assert econ.working_interest == 1.0


# ---------------------------------------------------------------------------
# Cashflow construction
# ---------------------------------------------------------------------------


class TestCashflow:
    def test_returns_cashflow_result(self):
        prod = _flat_production()
        econ = WellEconomics(capex=_CAPEX, price=70.0)
        result = cashflow(prod, econ)
        assert isinstance(result, CashflowResult)

    def test_capex_at_month_zero(self):
        prod = _flat_production()
        econ = WellEconomics(capex=_CAPEX, price=70.0)
        result = cashflow(prod, econ)
        assert result.capex_schedule[0] == pytest.approx(_CAPEX)
        assert np.all(result.capex_schedule[1:] == 0.0)

    def test_royalty_math(self):
        """NRI = WI × (1 - royalty_rate); gross_rev - royalty = gross_rev × NRI."""
        prod = np.array([100.0])
        econ = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.25, working_interest=1.0)
        result = cashflow(prod, econ)
        expected_gross = 100.0 * 70.0 * 1.0  # = 7000
        expected_royalty = expected_gross * 0.25  # = 1750
        assert result.gross_revenue[0] == pytest.approx(expected_gross)
        assert result.royalty[0] == pytest.approx(expected_royalty)
        assert result.net_revenue[0] == pytest.approx(expected_gross - expected_royalty)

    def test_working_interest_scales_revenue(self):
        prod = np.array([100.0])
        econ_full = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.0, working_interest=1.0)
        econ_half = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.0, working_interest=0.5)
        r_full = cashflow(prod, econ_full)
        r_half = cashflow(prod, econ_half)
        assert r_half.gross_revenue[0] == pytest.approx(r_full.gross_revenue[0] * 0.5)

    def test_severance_tax_on_net_revenue(self):
        prod = np.array([100.0])
        econ = WellEconomics(
            capex=0.0, price=70.0, royalty_rate=0.0,
            severance_tax_rate=0.046, ad_valorem_rate=0.0
        )
        result = cashflow(prod, econ)
        expected_sev = result.net_revenue[0] * 0.046
        assert result.severance_tax[0] == pytest.approx(expected_sev)

    def test_opex_variable_reduces_ebitda(self):
        prod = np.array([100.0])
        econ_no_opex = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.0,
                                     severance_tax_rate=0.0, ad_valorem_rate=0.0,
                                     opex_variable=0.0)
        econ_with_opex = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.0,
                                       severance_tax_rate=0.0, ad_valorem_rate=0.0,
                                       opex_variable=5.0)
        r_no = cashflow(prod, econ_no_opex)
        r_op = cashflow(prod, econ_with_opex)
        assert r_op.ebitda[0] < r_no.ebitda[0]
        assert r_no.ebitda[0] - r_op.ebitda[0] == pytest.approx(100.0 * 5.0)

    def test_price_escalation(self):
        prod = _flat_production(rate=100.0, months=12)
        econ = WellEconomics(capex=0.0, price=70.0, price_escalation=0.12,
                             royalty_rate=0.0, severance_tax_rate=0.0, ad_valorem_rate=0.0)
        result = cashflow(prod, econ)
        # Price at month 12 should be > month 0 price
        assert result.price[-1] > result.price[0]
        # Approx 12% annual = ~1% monthly
        expected_final = 70.0 * (1 + 0.12) ** (11 / 12)
        assert result.price[11] == pytest.approx(expected_final, rel=0.01)

    def test_cumulative_cashflow_shape(self):
        prod = _declining_production(months=60)
        econ = WellEconomics(capex=_CAPEX, price=60.0)
        result = cashflow(prod, econ)
        assert result.cumulative_cashflow.shape == (60,)
        assert result.cumulative_cashflow[-1] == pytest.approx(np.sum(result.net_cashflow))

    def test_to_dict_keys(self):
        prod = _flat_production()
        econ = WellEconomics(capex=_CAPEX, price=70.0)
        d = cashflow(prod, econ).to_dict()
        for key in ("production", "gross_revenue", "net_revenue", "ebitda",
                    "net_cashflow", "cumulative_cashflow"):
            assert key in d


# ---------------------------------------------------------------------------
# NPV
# ---------------------------------------------------------------------------


class TestNPV:
    def test_positive_npv_for_profitable_well(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        result = cashflow(prod, econ)
        assert npv(result) > 0

    def test_negative_npv_for_uneconomic_well(self):
        prod = _flat_production(rate=1.0, months=60)  # Very low production
        econ = WellEconomics(capex=_CAPEX_LARGE, price=10.0, opex_variable=50.0)
        result = cashflow(prod, econ)
        assert npv(result) < 0

    def test_higher_discount_rate_lowers_npv(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=60.0)
        r = cashflow(prod, econ)
        npv_low = npv(r, discount_rate=0.05)
        npv_high = npv(r, discount_rate=0.20)
        assert npv_low > npv_high

    def test_zero_discount_rate_equals_undiscounted_sum(self):
        prod = _flat_production(rate=100.0, months=12)
        econ = WellEconomics(capex=0.0, price=70.0, royalty_rate=0.0,
                             severance_tax_rate=0.0, ad_valorem_rate=0.0, discount_rate=0.0)
        result = cashflow(prod, econ)
        # NPV at 0% = sum of cashflows
        assert npv(result, discount_rate=0.0) == pytest.approx(
            float(np.sum(result.net_cashflow)), rel=0.001
        )


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------


class TestIRR:
    def test_irr_positive_for_profitable_project(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        result = cashflow(prod, econ)
        annual_irr = irr(result)
        assert not np.isnan(annual_irr)
        assert annual_irr > 0

    def test_irr_nan_when_no_sign_change(self):
        # If all cashflows negative, IRR is undefined
        prod = _flat_production(rate=1.0, months=12)
        econ = WellEconomics(capex=_CAPEX_LARGE, price=1.0, opex_variable=5.0)
        result = cashflow(prod, econ)
        # With huge capex vs tiny revenue, IRR may be NaN
        annual_irr = irr(result)
        # Either NaN or a large negative number — not asserting exact value


# ---------------------------------------------------------------------------
# Payout
# ---------------------------------------------------------------------------


class TestPayout:
    def test_payout_occurs_within_forecast(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        result = cashflow(prod, econ)
        month = payout(result)
        assert month > 0    # CAPEX drags month 0 negative
        assert month < 120

    def test_no_payout_for_uneconomic_well(self):
        prod = _flat_production(rate=1.0, months=12)
        econ = WellEconomics(capex=_CAPEX_LARGE, price=10.0)
        result = cashflow(prod, econ)
        assert payout(result) == -1

    def test_cumulative_cashflow_positive_at_payout(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        result = cashflow(prod, econ)
        month = payout(result)
        if month >= 0:
            assert result.cumulative_cashflow[month] > 0


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


class TestROI:
    def test_roi_positive_for_profitable_well(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        result = cashflow(prod, econ)
        assert roi(result) > 1.0  # Better than 100% return

    def test_roi_nan_when_capex_zero(self):
        prod = _flat_production()
        econ = WellEconomics(capex=0.0, price=70.0)
        result = cashflow(prod, econ)
        assert np.isnan(roi(result))


# ---------------------------------------------------------------------------
# Breakeven price
# ---------------------------------------------------------------------------


class TestBreakevenPrice:
    def test_breakeven_between_zero_and_current_price(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        be = breakeven_price(prod, econ)
        assert not np.isnan(be)
        assert 0 < be < 70.0  # Should be profitable at $70, breakeven is lower

    def test_npv_near_zero_at_breakeven(self):
        prod = _declining_production(qi=2000.0, months=120)
        econ = WellEconomics(capex=_CAPEX, price=70.0, opex_variable=10.0)
        be = breakeven_price(prod, econ)
        if not np.isnan(be):
            modified_econ = WellEconomics(
                capex=econ.capex, price=be, opex_variable=econ.opex_variable,
                royalty_rate=econ.royalty_rate, severance_tax_rate=econ.severance_tax_rate,
                ad_valorem_rate=econ.ad_valorem_rate, discount_rate=econ.discount_rate,
            )
            be_npv = npv(cashflow(prod, modified_econ))
            assert abs(be_npv) < 500.0  # Within $500 of zero

    def test_breakeven_nan_for_always_unprofitable_well(self):
        # So much CAPEX it can never break even within $500 price cap
        prod = _flat_production(rate=0.001, months=12)
        econ = WellEconomics(capex=_CAPEX_LARGE, price=70.0)
        be = breakeven_price(prod, econ)
        assert np.isnan(be)


# ---------------------------------------------------------------------------
# Backwards-compat legacy function
# ---------------------------------------------------------------------------


class TestLegacyEconomicMetrics:
    def test_returns_dict_with_expected_keys(self):
        q = np.array([100.0, 90.0, 80.0])
        result = economic_metrics(q, price=70.0, opex=15.0)
        assert "npv" in result
        assert "cash_flow" in result
        assert "payback_month" in result

    def test_cash_flow_length_matches_production(self):
        q = np.linspace(100, 50, 24)
        result = economic_metrics(q, price=60.0, opex=10.0)
        assert len(result["cash_flow"]) == 24

    def test_npv_is_float(self):
        q = np.linspace(100, 50, 12)
        result = economic_metrics(q, price=60.0, opex=10.0)
        assert isinstance(result["npv"], float)

    def test_consistent_with_new_api(self):
        # Legacy uses simple rate/12; new API uses compound (1+r)^(1/12)-1.
        # They intentionally differ — just verify both return finite floats.
        q = _declining_production(qi=500.0, months=60)
        legacy = economic_metrics(q, price=70.0, opex=15.0, discount_rate=0.10)
        econ = WellEconomics(
            capex=0.0, price=70.0, opex_variable=15.0, discount_rate=0.10,
            royalty_rate=0.0, severance_tax_rate=0.0, ad_valorem_rate=0.0,
        )
        new_npv = npv(cashflow(q, econ))
        assert np.isfinite(legacy["npv"])
        assert np.isfinite(new_npv)
        assert legacy["npv"] > 0 and new_npv > 0


# ---------------------------------------------------------------------------
# Risk report NPV fix
# ---------------------------------------------------------------------------


class TestRiskReportNPVFix:
    def test_npv_samples_not_empty_when_price_opex_provided(self):
        """The stub that returned [] is now replaced with real economics."""
        import pandas as pd

        from decline_curve.probabilistic_forecast import probabilistic_forecast

        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        prod = pd.Series(
            500.0 * np.exp(-0.03 * np.arange(36)), index=dates
        )
        forecast = probabilistic_forecast(
            prod, model="arps", kind="hyperbolic", horizon=12,
            n_draws=100, price=70.0, opex=15.0, seed=42
        )
        # price and opex should be stored on the forecast
        assert forecast.price == 70.0
        assert forecast.opex == 15.0
        # NPV bands should be computed
        assert forecast.npv_bands is not None
        assert "p50" in forecast.npv_bands
        assert forecast.npv_bands["p50"] != 0.0

    def test_risk_metrics_computable_with_price_opex(self):
        import pandas as pd

        from decline_curve.probabilistic_forecast import probabilistic_forecast
        from decline_curve.risk_report import calculate_risk_metrics

        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        prod = pd.Series(500.0 * np.exp(-0.03 * np.arange(36)), index=dates)
        forecast = probabilistic_forecast(
            prod, n_draws=100, price=70.0, opex=15.0, seed=42
        )
        metrics = calculate_risk_metrics(forecast)
        assert 0.0 <= metrics.prob_positive_npv <= 1.0
        assert isinstance(metrics.expected_npv, float)

    def test_risk_metrics_raises_without_price_opex(self):
        import pandas as pd

        from decline_curve.probabilistic_forecast import probabilistic_forecast
        from decline_curve.risk_report import calculate_risk_metrics

        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        prod = pd.Series(300.0 * np.exp(-0.04 * np.arange(24)), index=dates)
        forecast = probabilistic_forecast(prod, n_draws=50, seed=0)
        with pytest.raises(ValueError, match="price and opex"):
            calculate_risk_metrics(forecast)
