"""Reserves estimation and production forecasting."""

import numpy as np

from .models import ArpsParams, predict_arps


def forecast_and_reserves(
    params: ArpsParams, t_max: float = 240, dt: float = 1.0, econ_limit: float = 10.0
) -> dict:
    """
    Generate forecast and compute EUR.

    Args:
        params: ArpsParams for decline model.
        t_max: Time horizon in months.
        dt: Time step in months.
        econ_limit: Minimum economic production rate.

    Returns:
        Dict with forecast, time, and EUR.
    """
    t = np.arange(0, t_max + dt, dt)
    q = predict_arps(t, params)
    valid = q > econ_limit
    t_valid, q_valid = t[valid], q[valid]
    eur = np.trapz(q_valid, t_valid)
    return {"t": t, "q": q, "t_valid": t_valid, "q_valid": q_valid, "eur": eur}
