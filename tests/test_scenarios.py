"""Tests for price scenario analysis (upgraded full-cycle WellEconomics engine)."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.scenarios import (
    PriceScenario,
    compare_scenarios,
    run_multi_phase_scenarios,
    run_price_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 60

def _prod(qi=500.0, d=0.02, n=_N):
    return np.array([qi * np.exp(-d * t) for t in range(n)])


_BASE_Q = _prod()


# ---------------------------------------------------------------------------
# PriceScenario dataclass
# ---------------------------------------------------------------------------


class TestPriceScenario:
    def test_create_scenario(self):
        scenario = PriceScenario("base", oil_price=70.0, opex=15.0)
        assert scenario.name == "base"
        assert scenario.oil_price == 70.0
        assert scenario.opex == 15.0
        assert scenario.discount_rate == pytest.approx(0.10)

    def test_default_full_cycle_fields(self):
        s = PriceScenario("base", oil_price=70.0)
        assert s.royalty_rate == pytest.approx(0.1875)
        assert s.working_interest == pytest.approx(1.0)
        assert s.severance_tax_rate == pytest.approx(0.046)
        assert s.ad_valorem_rate == pytest.approx(0.02)
        assert s.capex == pytest.approx(0.0)

    def test_scenario_validation(self):
        with pytest.raises(ValueError, match="Oil price must be non-negative"):
            PriceScenario("low", oil_price=-10.0)

        with pytest.raises(ValueError, match="Gas price must be non-negative"):
            PriceScenario("low", oil_price=50.0, gas_price=-5.0)

        with pytest.raises(ValueError, match="Operating cost must be non-negative"):
            PriceScenario("low", oil_price=50.0, opex=-5.0)

    def test_to_well_economics(self):
        from decline_curve.economics import WellEconomics
        s = PriceScenario("test", oil_price=80.0, capex=5_000_000, royalty_rate=0.25)
        we = s.to_well_economics()
        assert isinstance(we, WellEconomics)
        assert we.price == pytest.approx(80.0)
        assert we.capex == pytest.approx(5_000_000)
        assert we.royalty_rate == pytest.approx(0.25)

    def test_to_well_economics_price_override(self):
        s = PriceScenario("test", oil_price=70.0)
        we = s.to_well_economics(price=50.0)
        assert we.price == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# run_price_scenarios
# ---------------------------------------------------------------------------


class TestRunPriceScenarios:
    def test_single_scenario(self):
        production = np.array([1000, 900, 800, 700, 600])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1
        assert results.iloc[0]["scenario"] == "base"
        assert "npv" in results.columns
        assert "payout_month" in results.columns

    def test_multiple_scenarios(self):
        production = np.array([1000, 900, 800, 700, 600])
        scenarios = [
            PriceScenario("low",  oil_price=50.0, opex=15.0),
            PriceScenario("base", oil_price=70.0, opex=15.0),
            PriceScenario("high", oil_price=90.0, opex=15.0),
        ]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 3
        assert set(results["scenario"]) == {"low", "base", "high"}
        assert (
            results[results["scenario"] == "high"]["npv"].iloc[0]
            > results[results["scenario"] == "base"]["npv"].iloc[0]
            > results[results["scenario"] == "low"]["npv"].iloc[0]
        )

    def test_has_full_cycle_columns(self):
        scenarios = [PriceScenario("base", oil_price=70.0, capex=2_000_000)]
        df = run_price_scenarios(_BASE_Q, scenarios)
        for col in ("scenario", "npv", "irr", "payout_month", "roi", "breakeven_price",
                    "total_revenue", "total_opex"):
            assert col in df.columns, f"Missing column: {col}"

    def test_pandas_series_input(self):
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        production = pd.Series([1000 - i * 50 for i in range(12)], index=dates)
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1

    def test_different_phases(self):
        production = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=3.0, opex=15.0)]

        results_oil = run_price_scenarios(production, scenarios, phase="oil")
        assert len(results_oil) == 1

        results_gas = run_price_scenarios(production, scenarios, phase="gas")
        assert len(results_gas) == 1

    def test_invalid_phase(self):
        production = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        with pytest.raises(ValueError, match="Unknown phase"):
            run_price_scenarios(production, scenarios, phase="invalid")

    def test_capex_reduces_npv(self):
        no_capex  = [PriceScenario("no_capex",   oil_price=70.0, capex=0)]
        with_capex = [PriceScenario("with_capex", oil_price=70.0, capex=5_000_000)]
        df_no = run_price_scenarios(_BASE_Q, no_capex)
        df_wi = run_price_scenarios(_BASE_Q, with_capex)
        assert df_no.iloc[0]["npv"] > df_wi.iloc[0]["npv"]

    def test_npv_matches_economics_engine(self):
        from decline_curve.economics import WellEconomics, cashflow, npv
        s = [PriceScenario("x", oil_price=70.0, capex=1_000_000, opex=10.0)]
        df = run_price_scenarios(_BASE_Q, s)
        econ = WellEconomics(capex=1_000_000, price=70.0, opex_variable=10.0)
        cf = cashflow(_BASE_Q, econ)
        assert df.iloc[0]["npv"] == pytest.approx(npv(cf), rel=1e-6)

    def test_irr_is_finite_or_nan(self):
        scenarios = [PriceScenario("base", oil_price=70.0, capex=1_000_000)]
        df = run_price_scenarios(_BASE_Q, scenarios)
        v = df.iloc[0]["irr"]
        assert np.isfinite(v) or np.isnan(v)

    def test_breakeven_finite_positive(self):
        scenarios = [PriceScenario("base", oil_price=70.0, capex=1_000_000)]
        df = run_price_scenarios(_BASE_Q, scenarios)
        bep = df.iloc[0]["breakeven_price"]
        if np.isfinite(bep):
            assert bep >= 0.0


# ---------------------------------------------------------------------------
# run_multi_phase_scenarios
# ---------------------------------------------------------------------------


class TestRunMultiPhaseScenarios:
    def test_oil_only(self):
        oil_prod = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]
        results = run_multi_phase_scenarios(oil_prod, scenarios)
        assert len(results) == 1
        assert results.iloc[0]["scenario"] == "base"

    def test_oil_and_gas(self):
        oil_prod = np.array([1000, 900, 800])
        gas_prod = np.array([5000, 4500, 4000])
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=3.0, opex=15.0)]
        results = run_multi_phase_scenarios(oil_prod, scenarios, gas_production=gas_prod)
        assert len(results) == 1
        assert results.iloc[0]["total_revenue"] > 0

    def test_gas_adds_total_revenue(self):
        oil_prod = _prod(qi=400)
        gas_prod = _prod(qi=200)
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=3.0, capex=3_000_000)]
        df_oil = run_multi_phase_scenarios(oil_prod, scenarios)
        df_multi = run_multi_phase_scenarios(oil_prod, scenarios, gas_production=gas_prod)
        # Gas adds gross revenue regardless of OPEX allocation
        assert df_multi.iloc[0]["total_revenue"] > df_oil.iloc[0]["total_revenue"]

    def test_gas_npv_higher_when_gas_price_exceeds_opex(self):
        # Gas price = $20/MCF >> opex = $2/unit → adding gas improves NPV
        oil_prod = _prod(qi=400)
        gas_prod = _prod(qi=200)
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=20.0, capex=3_000_000, opex=2.0)]
        df_oil = run_multi_phase_scenarios(oil_prod, scenarios)
        df_multi = run_multi_phase_scenarios(oil_prod, scenarios, gas_production=gas_prod)
        assert df_multi.iloc[0]["npv"] > df_oil.iloc[0]["npv"]

    def test_all_phases(self):
        oil_prod = np.array([1000, 900, 800])
        gas_prod = np.array([5000, 4500, 4000])
        water_prod = np.array([500, 450, 400])
        scenarios = [
            PriceScenario("base", oil_price=70.0, gas_price=3.0, water_price=-2.0, opex=15.0)
        ]
        results = run_multi_phase_scenarios(
            oil_prod, scenarios, gas_production=gas_prod, water_production=water_prod
        )
        assert len(results) == 1
        assert results.iloc[0]["total_revenue"] > 0

    def test_breakeven_is_nan_for_multiphase(self):
        scenarios = [PriceScenario("base", oil_price=70.0)]
        df = run_multi_phase_scenarios(_prod(qi=400), scenarios)
        assert df["breakeven_price"].isna().all()

    def test_has_irr_column(self):
        scenarios = [PriceScenario("base", oil_price=70.0, capex=2_000_000)]
        df = run_multi_phase_scenarios(_BASE_Q, scenarios)
        assert "irr" in df.columns


# ---------------------------------------------------------------------------
# compare_scenarios
# ---------------------------------------------------------------------------


class TestCompareScenarios:
    def test_compare_with_base(self):
        scenario_results = pd.DataFrame(
            {
                "scenario": ["low", "base", "high"],
                "npv": [100000, 200000, 300000],
                "payout_month": [24, 18, 12],
            }
        )
        comparison = compare_scenarios(scenario_results)
        assert "npv_vs_base" in comparison.columns
        assert "npv_pct_change" in comparison.columns
        assert "payback_vs_base" in comparison.columns

        base_row = comparison[comparison["scenario"] == "base"].iloc[0]
        assert base_row["npv_vs_base"] == pytest.approx(0.0, abs=1.0)

    def test_compare_without_base(self):
        scenario_results = pd.DataFrame(
            {
                "scenario": ["scenario1", "scenario2"],
                "npv": [100000, 200000],
                "payout_month": [24, 18],
            }
        )
        comparison = compare_scenarios(scenario_results)
        assert "npv_vs_base" in comparison.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_production(self):
        production = np.array([])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]
        results = run_price_scenarios(production, scenarios)
        assert isinstance(results, pd.DataFrame)

    def test_zero_production(self):
        production = np.array([0.0, 0.0, 0.0])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]
        results = run_price_scenarios(production, scenarios)
        assert len(results) == 1
        assert results.iloc[0]["npv"] <= 0

    def test_very_short_series(self):
        production = np.array([1000.0])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]
        results = run_price_scenarios(production, scenarios)
        assert len(results) == 1

    def test_negative_production_handled(self):
        production = np.array([1000, -100, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]
        results = run_price_scenarios(production, scenarios)
        assert len(results) == 1
