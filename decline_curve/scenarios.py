"""Price scenario analysis for economic evaluation.

This module provides tools for running multiple price scenarios (low, base, high)
and comparing economic results across scenarios. Uses the full-cycle
:class:`~decline_curve.economics.WellEconomics` engine (royalties, taxes, CAPEX).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .economics import WellEconomics, breakeven_price, cashflow, irr, npv, payout, roi
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PriceScenario:
    """Price scenario definition for full-cycle economic evaluation.

    All fields with defaults match the :class:`~decline_curve.economics.WellEconomics`
    conventions so scenarios map directly to that engine.

    Attributes:
        name: Scenario name (e.g., 'low', 'base', 'high').
        oil_price: Oil price ($/bbl).
        gas_price: Gas price ($/mcf); used by multi-phase runner.
        water_price: Water disposal cost ($/bbl, typically negative).
        opex: Variable oil lifting cost ($/bbl).
        gas_opex: Variable gas lifting cost ($/mcf).
        water_opex: Water disposal cost ($/bbl) — charged to opex, not revenue.
        fixed_opex: Fixed operating expenses ($/month).
        capex: One-time capital expenditure at t=0 ($).
        royalty_rate: Royalty fraction (0.1875 = standard 3/16).
        working_interest: Working interest fraction (1.0 = 100% WI).
        severance_tax_rate: Severance/production tax on net revenue.
        ad_valorem_rate: Ad valorem/property tax on net revenue.
        discount_rate: Annual discount rate for NPV (0.10 = 10%).
    """

    name: str
    oil_price: float
    gas_price: Optional[float] = None
    water_price: Optional[float] = None
    opex: float = 15.0
    gas_opex: float = 0.0
    water_opex: float = 0.0
    fixed_opex: float = 0.0
    capex: float = 0.0
    royalty_rate: float = 0.1875
    working_interest: float = 1.0
    severance_tax_rate: float = 0.046
    ad_valorem_rate: float = 0.02
    discount_rate: float = 0.10

    def __post_init__(self):
        if self.oil_price < 0:
            raise ValueError("Oil price must be non-negative")
        if self.gas_price is not None and self.gas_price < 0:
            raise ValueError("Gas price must be non-negative")
        if self.opex < 0:
            raise ValueError("Operating cost must be non-negative")

    def to_well_economics(self, price: float | None = None) -> WellEconomics:
        """Build a :class:`~decline_curve.economics.WellEconomics` from this scenario."""
        return WellEconomics(
            capex=self.capex,
            price=price if price is not None else self.oil_price,
            royalty_rate=self.royalty_rate,
            working_interest=self.working_interest,
            severance_tax_rate=self.severance_tax_rate,
            ad_valorem_rate=self.ad_valorem_rate,
            opex_fixed=self.fixed_opex,
            opex_variable=self.opex,
            gas_opex=self.gas_opex,
            water_opex=self.water_opex,
            discount_rate=self.discount_rate,
        )


@dataclass
class ScenarioResult:
    """Result from a single price scenario.

    Attributes:
        scenario_name: Name of the scenario.
        npv: Net present value using compound monthly discounting ($).
        irr: Annual internal rate of return.
        payout_month: Month when cumulative cashflow turns positive (-1 if never).
        roi: Return on investment = (cumulative NCF + CAPEX) / CAPEX.
        breakeven_price: Commodity price at which NPV = 0 ($/BOE).
        cash_flow: Monthly net cashflow array.
        cumulative_cash_flow: Cumulative sum of cash_flow.
        total_revenue: Undiscounted sum of gross revenue.
        total_opex: Undiscounted sum of all operating costs.
    """

    scenario_name: str
    npv: float
    irr: float
    payout_month: int
    roi: float
    breakeven_price: float
    cash_flow: np.ndarray
    cumulative_cash_flow: np.ndarray
    total_revenue: float
    total_opex: float


def run_price_scenarios(
    production: pd.Series | np.ndarray,
    scenarios: list[PriceScenario],
    phase: str = "oil",
) -> pd.DataFrame:
    """Run multiple price scenarios on a production forecast.

    Uses the full-cycle economics engine (royalties, taxes, CAPEX, compound
    discounting). Each :class:`PriceScenario` is converted to a
    :class:`~decline_curve.economics.WellEconomics` instance.

    Args:
        production: Monthly production forecast (BOE/month or MCF/month).
        scenarios: List of :class:`PriceScenario` objects.
        phase: Production phase (``'oil'``, ``'gas'``, ``'water'``).

    Returns:
        DataFrame with columns: ``scenario``, ``npv``, ``irr``, ``payout_month``,
        ``roi``, ``breakeven_price``, ``total_revenue``, ``total_opex``.

    Example:
        >>> scenarios = [
        ...     PriceScenario('low',  oil_price=50.0, capex=5_000_000),
        ...     PriceScenario('base', oil_price=70.0, capex=5_000_000),
        ...     PriceScenario('high', oil_price=90.0, capex=5_000_000),
        ... ]
        >>> results = run_price_scenarios(production_forecast, scenarios)
        >>> print(results[['scenario', 'npv', 'irr', 'payout_month']])
    """
    q = production.values if isinstance(production, pd.Series) else np.asarray(production)

    phase_price_map = {
        "oil":   lambda s: s.oil_price,
        "gas":   lambda s: s.gas_price if s.gas_price is not None else 0.0,
        "water": lambda s: s.water_price if s.water_price is not None else 0.0,
    }
    if phase not in phase_price_map:
        raise ValueError(f"Unknown phase: {phase!r}. Must be one of: {list(phase_price_map)}")

    rows = []
    for scenario in scenarios:
        price = phase_price_map[phase](scenario)
        econ = scenario.to_well_economics(price=price)
        cf = cashflow(q, econ)

        rows.append({
            "scenario":        scenario.name,
            "npv":             npv(cf),
            "irr":             irr(cf),
            "payout_month":    payout(cf),
            "roi":             roi(cf),
            "breakeven_price": breakeven_price(q, econ),
            "total_revenue":   float(cf.gross_revenue.sum()),
            "total_opex":      float(cf.opex.sum()),
        })

    return pd.DataFrame(rows)


def run_multi_phase_scenarios(
    oil_production: pd.Series | np.ndarray,
    scenarios: list[PriceScenario],
    gas_production: Optional[pd.Series | np.ndarray] = None,
    water_production: Optional[pd.Series | np.ndarray] = None,
) -> pd.DataFrame:
    """Run price scenarios on multi-phase production (oil + gas + water).

    Because the full-cycle engine is per-BOE, multi-phase cash flow is computed
    directly here with compound monthly discounting consistent with
    :func:`~decline_curve.economics.npv`.

    Args:
        oil_production: Monthly oil production (BOE/month).
        scenarios: List of :class:`PriceScenario` objects.
        gas_production: Optional monthly gas production (MCF/month).
        water_production: Optional monthly water production (BBL/month).

    Returns:
        DataFrame with scenario economics (same columns as :func:`run_price_scenarios`).
    """
    oil_q = oil_production.values if isinstance(oil_production, pd.Series) else np.asarray(oil_production)
    n = len(oil_q)
    gas_q  = (gas_production.values  if isinstance(gas_production,   pd.Series) else np.asarray(gas_production))  if gas_production   is not None else np.zeros(n)
    water_q = (water_production.values if isinstance(water_production, pd.Series) else np.asarray(water_production)) if water_production is not None else np.zeros(n)

    rows = []
    for scenario in scenarios:
        oil_rev   = oil_q   * scenario.oil_price
        gas_rev   = gas_q   * (scenario.gas_price   or 0.0)
        water_rev = water_q * (scenario.water_price  or 0.0)

        gross_rev  = oil_rev + gas_rev + water_rev
        royalty    = gross_rev * scenario.royalty_rate
        net_rev    = gross_rev - royalty
        sev_tax    = net_rev * scenario.severance_tax_rate
        ad_val     = net_rev * scenario.ad_valorem_rate
        var_opex   = oil_q * scenario.opex + gas_q * scenario.gas_opex + water_q * scenario.water_opex
        fix_opex   = np.full(n, scenario.fixed_opex)
        total_opex = var_opex + fix_opex
        ebitda     = net_rev - sev_tax - ad_val - total_opex

        capex_arr  = np.zeros(n)
        if n > 0:
            capex_arr[0] = scenario.capex
        cf_arr = ebitda - capex_arr

        # Compound monthly discounting (matches economics.npv)
        monthly_r = (1.0 + scenario.discount_rate) ** (1.0 / 12.0) - 1.0
        periods   = np.arange(n, dtype=float)
        npv_val   = float(np.sum(cf_arr / (1.0 + monthly_r) ** periods))
        cum_cf    = np.cumsum(cf_arr)
        payout_m  = int(np.argmax(cum_cf > 0)) if np.any(cum_cf > 0) else -1
        roi_val   = float((cum_cf[-1] + scenario.capex) / scenario.capex) if scenario.capex else float("nan")

        # IRR via numpy_financial
        try:
            import numpy_financial as npf
            monthly_irr = npf.irr(cf_arr)
            irr_val = float((1.0 + monthly_irr) ** 12 - 1.0) if monthly_irr is not None and not np.isnan(monthly_irr) else float("nan")
        except Exception:
            irr_val = float("nan")

        rows.append({
            "scenario":        scenario.name,
            "npv":             npv_val,
            "irr":             irr_val,
            "payout_month":    payout_m,
            "roi":             roi_val,
            "breakeven_price": float("nan"),   # not defined for multi-phase
            "total_revenue":   float(gross_rev.sum()),
            "total_opex":      float(total_opex.sum()),
        })

    return pd.DataFrame(rows)


def compare_scenarios(scenario_results: pd.DataFrame) -> pd.DataFrame:
    """Compare multiple scenarios and show differences from the base case.

    Args:
        scenario_results: DataFrame from :func:`run_price_scenarios` or
            :func:`run_multi_phase_scenarios`.

    Returns:
        DataFrame with additional ``npv_vs_base``, ``npv_pct_change``, and
        ``payback_vs_base`` columns.
    """
    vals = scenario_results["scenario"].values
    base_candidates = ["base", "Base", "BASE", "baseline"]
    mask = np.isin(vals, base_candidates)
    base = vals[mask][0] if mask.any() else vals[0]
    if not mask.any():
        logger.warning("No 'base' scenario found, using %r as reference", base)

    base_npv     = scenario_results.loc[scenario_results["scenario"] == base, "npv"].iloc[0]
    base_payout  = scenario_results.loc[scenario_results["scenario"] == base, "payout_month"].iloc[0]

    comparison = scenario_results.copy()
    npv_diff = comparison["npv"] - base_npv
    comparison["npv_vs_base"]    = npv_diff
    comparison["npv_pct_change"] = np.where(base_npv != 0, (npv_diff / abs(base_npv)) * 100, 0.0)
    comparison["payback_vs_base"] = comparison["payout_month"] - base_payout
    return comparison
