"""Economic analysis and financial metrics for production forecasts."""

import numpy as np
from numpy_financial import npv


def economic_metrics(
    q: np.ndarray,
    price: float,
    opex: float,
    discount_rate: float = 0.10,
    time_step_months: float = 1.0,
) -> dict:
    """
    Calculate economics from forecasted production.

    Args:
        q: Production forecast (monthly).
        price: Unit price.
        opex: Operating cost.
        discount_rate: Annual discount rate.
        time_step_months: Length of time step in months.

    Returns:
        Dict with cash flow, NPV, payback.
    """
    monthly_rate = discount_rate / 12
    net_revenue = (price - opex) * q
    cash_flow = net_revenue
    npv_val = npv(monthly_rate, cash_flow)
    cum_cf = np.cumsum(cash_flow)
    payback_month = int(np.argmax(cum_cf > 0)) if np.any(cum_cf > 0) else -1
    return {"npv": npv_val, "cash_flow": cash_flow, "payback_month": payback_month}
