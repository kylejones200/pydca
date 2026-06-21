"""Full-cycle well economics for production forecasts.

Provides a proper cashflow model covering:
- Gross/net revenue with royalty and working interest
- Severance tax and ad valorem tax
- Fixed and variable operating costs with escalation
- CAPEX (one-time at t=0)
- NPV, IRR, payout period, ROI, and breakeven price

The legacy ``economic_metrics()`` function is preserved for backwards
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy_financial as npf
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WellEconomics:
    """Full-cycle economic parameters for a single well.

    Attributes:
        capex: Capital expenditure in $ (one-time at t=0, positive value).
        price: Commodity price in $/BBL or $/MCF.
        price_escalation: Annual price escalation rate (0.02 = 2% per year).
        royalty_rate: Royalty fraction (0.1875 = standard 3/16 landowner royalty).
        working_interest: Working interest fraction (1.0 = 100% WI).
        severance_tax_rate: Severance/production tax rate on gross revenue.
        ad_valorem_rate: Ad valorem/property tax rate on gross revenue.
        opex_fixed: Fixed operating cost in $/month.
        opex_variable: Variable operating cost in $/BOE.
        opex_escalation: Annual opex escalation rate.
        discount_rate: Annual discount rate for NPV (0.10 = 10%).
        econ_limit: Abandon when net revenue falls below this value in $/month (0 = run to end).
    """

    capex: float
    price: float
    price_escalation: float = 0.0
    royalty_rate: float = 0.1875
    working_interest: float = 1.0
    severance_tax_rate: float = 0.046
    ad_valorem_rate: float = 0.02
    opex_fixed: float = 0.0
    opex_variable: float = 0.0
    opex_escalation: float = 0.0
    discount_rate: float = 0.10
    econ_limit: float = 0.0


@dataclass
class CashflowResult:
    """Monthly cashflow arrays from a well economic model.

    All monetary values are in $/month unless noted. Units match revenue:
    production (BOE/month) × price ($/BOE) → $/month.

    Attributes:
        production: Monthly production volumes (BOE or MCF).
        price: Realized price each month ($/BOE).
        gross_revenue: production × price × working_interest.
        royalty: Royalty deduction (gross_revenue × royalty_rate).
        net_revenue: gross_revenue − royalty.
        severance_tax: net_revenue × severance_tax_rate.
        ad_valorem: net_revenue × ad_valorem_rate.
        opex: Total operating cost (fixed + variable) each month.
        ebitda: net_revenue − severance_tax − ad_valorem − opex.
        capex_schedule: CAPEX array (capex at index 0, 0 elsewhere).
        net_cashflow: ebitda − capex_schedule.
        cumulative_cashflow: Cumulative sum of net_cashflow.
        econ: The WellEconomics used to produce this result.
    """

    production: np.ndarray
    price: np.ndarray
    gross_revenue: np.ndarray
    royalty: np.ndarray
    net_revenue: np.ndarray
    severance_tax: np.ndarray
    ad_valorem: np.ndarray
    opex: np.ndarray
    ebitda: np.ndarray
    capex_schedule: np.ndarray
    net_cashflow: np.ndarray
    cumulative_cashflow: np.ndarray
    econ: WellEconomics = field(repr=False)

    def to_dict(self) -> dict:
        """Return all arrays as a plain dict (useful for DataFrames)."""
        return {
            "production": self.production,
            "price": self.price,
            "gross_revenue": self.gross_revenue,
            "royalty": self.royalty,
            "net_revenue": self.net_revenue,
            "severance_tax": self.severance_tax,
            "ad_valorem": self.ad_valorem,
            "opex": self.opex,
            "ebitda": self.ebitda,
            "capex_schedule": self.capex_schedule,
            "net_cashflow": self.net_cashflow,
            "cumulative_cashflow": self.cumulative_cashflow,
        }


# ---------------------------------------------------------------------------
# Core cashflow builder
# ---------------------------------------------------------------------------


def cashflow(production: np.ndarray, econ: WellEconomics) -> CashflowResult:
    """Build a monthly cashflow from a production forecast and well economics.

    The cashflow calculation follows standard petroleum engineering conventions:
    - NRI (Net Revenue Interest) = WI × (1 − royalty_rate)
    - Gross revenue = production × price × WI (before royalty)
    - Net revenue = gross_revenue × (1 − royalty_rate)
    - Taxes applied to net revenue
    - CAPEX is a one-time outflow at month 0

    Args:
        production: Monthly production volumes (BOE/month or MCF/month).
        econ: Well economic parameters.

    Returns:
        CashflowResult with per-month arrays.
    """
    n = len(production)
    months = np.arange(n, dtype=float)

    # Price escalation (monthly compounding from annual rate)
    monthly_price_esc = (1.0 + econ.price_escalation) ** (1.0 / 12.0) - 1.0
    realized_price = econ.price * (1.0 + monthly_price_esc) ** months

    # Revenue
    gross_rev = production * realized_price * econ.working_interest
    royalty = gross_rev * econ.royalty_rate
    net_rev = gross_rev - royalty

    # Taxes (applied to net revenue)
    sev_tax = net_rev * econ.severance_tax_rate
    ad_val = net_rev * econ.ad_valorem_rate

    # OPEX with escalation
    monthly_opex_esc = (1.0 + econ.opex_escalation) ** (1.0 / 12.0) - 1.0
    opex_fixed_arr = econ.opex_fixed * (1.0 + monthly_opex_esc) ** months
    opex_var_arr = production * econ.opex_variable
    total_opex = opex_fixed_arr + opex_var_arr

    # EBITDA
    ebitda_arr = net_rev - sev_tax - ad_val - total_opex

    # CAPEX schedule
    capex_arr = np.zeros(n)
    if n > 0:
        capex_arr[0] = econ.capex

    # Net cashflow
    ncf = ebitda_arr - capex_arr
    cum_ncf = np.cumsum(ncf)

    return CashflowResult(
        production=production,
        price=realized_price,
        gross_revenue=gross_rev,
        royalty=royalty,
        net_revenue=net_rev,
        severance_tax=sev_tax,
        ad_valorem=ad_val,
        opex=total_opex,
        ebitda=ebitda_arr,
        capex_schedule=capex_arr,
        net_cashflow=ncf,
        cumulative_cashflow=cum_ncf,
        econ=econ,
    )


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------


def npv(result: CashflowResult, discount_rate: Optional[float] = None) -> float:
    """Net Present Value of a cashflow at the given discount rate.

    Args:
        result: CashflowResult from :func:`cashflow`.
        discount_rate: Annual discount rate (overrides econ.discount_rate if given).

    Returns:
        NPV in $M.
    """
    rate = discount_rate if discount_rate is not None else result.econ.discount_rate
    monthly_rate = (1.0 + rate) ** (1.0 / 12.0) - 1.0
    return float(npf.npv(monthly_rate, result.net_cashflow))


def irr(result: CashflowResult) -> float:
    """Internal Rate of Return (annual) of a cashflow.

    Returns NaN if IRR cannot be computed (e.g., no sign change in cashflow).

    Args:
        result: CashflowResult from :func:`cashflow`.

    Returns:
        Annual IRR as a decimal (0.25 = 25%) or NaN.
    """
    monthly_irr = npf.irr(result.net_cashflow)
    if monthly_irr is None or np.isnan(monthly_irr):
        return float("nan")
    # Convert monthly to annual
    return float((1.0 + monthly_irr) ** 12 - 1.0)


def payout(result: CashflowResult) -> int:
    """Month index when cumulative cashflow first turns positive.

    Args:
        result: CashflowResult from :func:`cashflow`.

    Returns:
        Month index (0-based) of payout, or -1 if the well never pays out.
    """
    positive = np.where(result.cumulative_cashflow > 0)[0]
    return int(positive[0]) if len(positive) > 0 else -1


def roi(result: CashflowResult) -> float:
    """Return on investment: (cumulative NCF + CAPEX) / CAPEX.

    Args:
        result: CashflowResult from :func:`cashflow`.

    Returns:
        ROI as a decimal (1.5 = 150% return on invested capital) or NaN if
        CAPEX is zero.
    """
    if result.econ.capex == 0.0:
        return float("nan")
    return float((result.cumulative_cashflow[-1] + result.econ.capex) / result.econ.capex)


def breakeven_price(production: np.ndarray, econ: WellEconomics) -> float:
    """Find the commodity price at which NPV = 0.

    Uses Brent's method to solve for the breakeven price. Returns NaN if
    a breakeven cannot be found within the search range.

    Args:
        production: Monthly production volumes.
        econ: Well economic parameters (price field is ignored; solved for).

    Returns:
        Breakeven price in $/BOE or NaN.
    """
    from scipy.optimize import brentq

    def _npv_at_price(p: float) -> float:
        modified_econ = WellEconomics(
            capex=econ.capex,
            price=p,
            price_escalation=econ.price_escalation,
            royalty_rate=econ.royalty_rate,
            working_interest=econ.working_interest,
            severance_tax_rate=econ.severance_tax_rate,
            ad_valorem_rate=econ.ad_valorem_rate,
            opex_fixed=econ.opex_fixed,
            opex_variable=econ.opex_variable,
            opex_escalation=econ.opex_escalation,
            discount_rate=econ.discount_rate,
        )
        return npv(cashflow(production, modified_econ))

    try:
        # Search from $0 to $500/BOE
        return float(brentq(_npv_at_price, 0.0, 500.0, xtol=0.01, maxiter=100))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Backwards-compatible legacy function
# ---------------------------------------------------------------------------


def economic_metrics(
    q: np.ndarray,
    price: float,
    opex: float,
    discount_rate: float = 0.10,
    time_step_months: float = 1.0,
) -> dict:
    """Legacy economic metrics — kept for backwards compatibility.

    Prefer :func:`cashflow` + :func:`npv` / :func:`payout` for new code.

    Args:
        q: Production forecast (monthly volumes).
        price: Unit price ($/BOE or $/MCF).
        opex: Operating cost per unit ($/BOE).
        discount_rate: Annual discount rate.
        time_step_months: Unused; retained for API compatibility.

    Returns:
        Dict with ``cash_flow`` array, ``npv`` float, ``payback_month`` int.
    """
    # Preserve the original simple rate/12 convention for backwards compat.
    # New code should use cashflow() + npv() which uses compound monthly rate.
    monthly_rate = discount_rate / 12.0
    net_revenue = (price - opex) * q
    cash_flow = net_revenue
    npv_val = float(npf.npv(monthly_rate, cash_flow))
    cum_cf = np.cumsum(cash_flow)
    payback_month = int(np.argmax(cum_cf > 0)) if np.any(cum_cf > 0) else -1
    return {"npv": npv_val, "cash_flow": cash_flow, "payback_month": payback_month}


# ---------------------------------------------------------------------------
# Portfolio economics
# ---------------------------------------------------------------------------


def portfolio_economics(
    production_profiles: Dict[str, np.ndarray],
    econ: WellEconomics,
) -> pd.DataFrame:
    """Apply full-cycle economics to a portfolio of wells.

    Evaluates :func:`cashflow` for every well, computes NPV, IRR, payout, ROI,
    and breakeven price, then appends a PORTFOLIO summary row.

    Args:
        production_profiles: Mapping of ``{well_id: monthly_production_array}``.
            Each value is a 1-D NumPy array of BOE/month production volumes.
        econ: Shared :class:`WellEconomics` applied to every well.  To use
            per-well economics, call :func:`cashflow` + helpers directly.

    Returns:
        DataFrame indexed by well_id with columns
        ``npv``, ``irr``, ``payout_month``, ``roi``, ``breakeven_price``,
        ``eur`` (cumulative production), plus a final ``PORTFOLIO`` row with
        summed NPV/EUR and mean IRR/ROI.

    Example:
        >>> profiles = {"WELL_A": q_a, "WELL_B": q_b, "WELL_C": q_c}
        >>> econ = WellEconomics(capex=5_000_000, price=70.0, opex_variable=8.0)
        >>> df = portfolio_economics(profiles, econ)
        >>> print(df[["npv", "irr", "payout_month"]].to_string())
    """
    if not production_profiles:
        return pd.DataFrame(
            columns=["npv", "irr", "payout_month", "roi", "breakeven_price", "eur"]
        )

    rows = []
    for well_id, q in production_profiles.items():
        q = np.asarray(q, dtype=float)
        cf = cashflow(q, econ)
        rows.append(
            {
                "well_id": well_id,
                "npv": npv(cf),
                "irr": irr(cf),
                "payout_month": payout(cf),
                "roi": roi(cf),
                "breakeven_price": breakeven_price(q, econ),
                "eur": float(q.sum()),
            }
        )

    df = pd.DataFrame(rows).set_index("well_id")

    if len(df) > 0:
        portfolio_row = pd.DataFrame(
            [
                {
                    "well_id": "PORTFOLIO",
                    "npv": df["npv"].sum(),
                    "irr": df["irr"].mean(),
                    "payout_month": int(df["payout_month"].median()),
                    "roi": df["roi"].mean(),
                    "breakeven_price": df["breakeven_price"].mean(),
                    "eur": df["eur"].sum(),
                }
            ]
        ).set_index("well_id")
        df = pd.concat([df, portfolio_row])

    return df
