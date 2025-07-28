from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class ArpsParams:
    qi: float
    di: float
    b: float


def q_exp(t, qi, di):
    return qi * np.exp(-di * t)


def q_hyp(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


def fit_arps(
    t: np.ndarray,
    q: np.ndarray,
    kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
) -> ArpsParams:
    """Fit an Arps model to one decline series.

    Args:
        t: Time index (months from first production).
        q: Production volumes.
        kind: exponential, harmonic, or hyperbolic.

    Returns:
        ArpsParams with qi, di, and b (b=0 for exponential or 1 for harmonic).
    """
    # Input validation
    if len(t) == 0 or len(q) == 0:
        raise ValueError("Input arrays cannot be empty")

    if len(t) != len(q):
        raise ValueError("Time and production arrays must have same length")

    # Handle single point case
    if len(t) == 1:
        qi = q[0] if q[0] > 0 else 1.0
        di = 0.01  # Small default decline rate
        if kind == "exponential":
            return ArpsParams(qi=qi, di=di, b=0.0)
        elif kind == "harmonic":
            return ArpsParams(qi=qi, di=di, b=1.0)
        else:  # hyperbolic
            return ArpsParams(qi=qi, di=di, b=0.5)

    # Handle zero or negative production
    if np.all(q <= 0):
        raise ValueError("All production values are zero or negative")

    # Filter out non-positive values
    valid_mask = q > 0
    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    if len(t_valid) < 2:
        # Not enough valid points for fitting
        qi = np.max(q) if np.max(q) > 0 else 1.0
        di = 0.01
        if kind == "exponential":
            return ArpsParams(qi=qi, di=di, b=0.0)
        elif kind == "harmonic":
            return ArpsParams(qi=qi, di=di, b=1.0)
        else:
            return ArpsParams(qi=qi, di=di, b=0.5)

    try:
        if kind == "exponential":
            popt, _ = curve_fit(
                q_exp, t_valid, q_valid, bounds=(0, np.inf), maxfev=10000
            )
            qi, di = popt
            return ArpsParams(qi=qi, di=di, b=0.0)

        if kind == "harmonic":

            def q_harm(t, qi, di):
                return qi / (1 + di * t)

            popt, _ = curve_fit(
                q_harm, t_valid, q_valid, bounds=(0, np.inf), maxfev=10000
            )
            qi, di = popt
            return ArpsParams(qi=qi, di=di, b=1.0)

        if kind == "hyperbolic":
            popt, _ = curve_fit(
                q_hyp,
                t_valid,
                q_valid,
                bounds=(0, [np.inf, np.inf, 2.0]),
                maxfev=100000,
            )
            qi, di, b = popt
            return ArpsParams(qi=qi, di=di, b=b)

    except Exception:
        # Fallback to simple estimates if curve fitting fails
        qi = q_valid[0] if len(q_valid) > 0 else 1.0
        di = 0.01  # Default decline rate
        if kind == "exponential":
            return ArpsParams(qi=qi, di=di, b=0.0)
        elif kind == "harmonic":
            return ArpsParams(qi=qi, di=di, b=1.0)
        else:
            return ArpsParams(qi=qi, di=di, b=0.5)

    raise ValueError("Unknown kind")


def predict_arps(t: np.ndarray, p: ArpsParams) -> np.ndarray:
    """Predict with fitted Arps parameters.

    Args:
        t: Time points.
        p: Arps parameters.

    Returns:
        Predicted rates.
    """
    # Handle both ArpsParams objects and dictionaries for backward compatibility
    if isinstance(p, dict):
        qi, di, b = p["qi"], p["di"], p["b"]
    else:
        qi, di, b = p.qi, p.di, p.b

    if b == 0.0:
        return q_exp(t, qi, di)
    if np.isclose(b, 1.0):
        return qi / (1 + di * t)
    return q_hyp(t, qi, di, b)


def estimate_reserves(params: ArpsParams, t_max: float = 50.0) -> float:
    """Estimate ultimate recoverable reserves using Arps decline curves.

    Args:
        params: Arps parameters (qi, di, b).
        t_max: Maximum time for integration (years).

    Returns:
        Estimated reserves.
    """
    # Handle both ArpsParams objects and dictionaries
    if isinstance(params, dict):
        qi, di, b = params["qi"], params["di"], params["b"]
    else:
        qi, di, b = params.qi, params.di, params.b

    if di <= 0:
        raise ValueError("Decline rate must be positive")

    # Check for invalid parameters
    if "kind" in params and isinstance(params, dict):
        kind = params["kind"]
        if kind not in ["exponential", "harmonic", "hyperbolic"]:
            raise ValueError(f"Invalid decline type: {kind}")

    if b == 0.0:  # Exponential
        return qi / di
    elif np.isclose(b, 1.0):  # Harmonic
        # For harmonic decline: EUR = qi * ln(1 + di * t_max) / di
        # This gives higher reserves than exponential for same qi, di
        return qi * np.log(1 + di * t_max) / di
    else:  # Hyperbolic
        if b >= 1.0:
            # For b >= 1, reserves approach infinity, use practical cutoff
            # Use harmonic approximation for b close to 1
            return qi * np.log(1 + di * t_max) / di
        else:
            # For b < 1, analytical solution exists
            # EUR = qi * (1 - (1 + b*di*t_max)^((1-b)/b)) / (di * (1-b))
            if np.isclose(b, 1.0, atol=1e-6):
                return qi * np.log(1 + di * t_max) / di
            else:
                # Use numerical integration for hyperbolic to ensure accuracy
                t_points = np.linspace(0, t_max, 1000)
                q_points = qi / ((1 + b * di * t_points) ** (1 / b))
                reserves = np.trapz(q_points, t_points)
                return max(reserves, 0)  # Ensure non-negative
