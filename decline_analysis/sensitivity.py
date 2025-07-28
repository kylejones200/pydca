from typing import List, Tuple

import numpy as np
import pandas as pd

from .economics import economic_metrics
from .models import ArpsParams, predict_arps


def run_sensitivity(
    param_grid: List[Tuple[float, float, float]],
    prices: List[float],
    opex: float,
    discount_rate: float = 0.10,
    t_max: float = 240,
    econ_limit: float = 10.0,
    dt: float = 1.0,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across Arps parameters and prices.

    Args:
        param_grid: List of (qi, di, b) tuples.
        prices: List of oil/gas prices to test.
        opex: Operating cost per unit.
        discount_rate: Annual discount rate.
        t_max: Time horizon in months.
        econ_limit: Minimum economic production rate.
        dt: Time step in months.

    Returns:
        DataFrame with qi, di, b, price, EUR, NPV, payback.
    """
    results = []

    for price in prices:
        for qi, di, b in param_grid:
            p = ArpsParams(qi=qi, di=di, b=b)
            t = np.arange(0, t_max + dt, dt)
            q = predict_arps(t, p)
            mask = q > econ_limit
            if not np.any(mask):
                continue

            t_valid = t[mask]
            q_valid = q[mask]
            eur = np.trapz(q_valid, t_valid)

            econ = economic_metrics(q_valid, price, opex, discount_rate)

            results.append(
                {
                    "qi": qi,
                    "di": di,
                    "b": b,
                    "price": price,
                    "EUR": eur,
                    "NPV": econ["npv"],
                    "Payback_month": econ["payback_month"],
                }
            )

    return pd.DataFrame(results)
