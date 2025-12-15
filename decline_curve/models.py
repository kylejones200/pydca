"""Arps decline curve models and parameter fitting."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional, Union, cast

import numpy as np
from scipy.optimize import curve_fit

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def _jit_noop(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    numba = cast(Any, SimpleNamespace(jit=_jit_noop))


@dataclass
class ArpsParams:
    """Arps decline curve parameters.

    Attributes:
        qi: Initial production rate.
        di: Initial decline rate.
        b: Decline exponent (b-factor).
    """

    qi: float
    di: float
    b: float


ArpsParamsLike = Union[ArpsParams, Mapping[str, Any]]


@numba.jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda f: f
def q_exp(t, qi, di):
    """Exponential decline rate function.

    Args:
        t: Time array.
        qi: Initial production rate.
        di: Initial decline rate.

    Returns:
        Production rate at time t.
    """
    return qi * np.exp(-di * t)


@numba.jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda f: f
def q_hyp(t, qi, di, b):
    """Hyperbolic decline rate function.

    Args:
        t: Time array.
        qi: Initial production rate.
        di: Initial decline rate.
        b: Decline exponent.

    Returns:
        Production rate at time t.
    """
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


def predict_arps(t: np.ndarray, p: ArpsParamsLike) -> np.ndarray:
    """Predict with fitted Arps parameters.

    Args:
        t: Time points.
        p: Arps parameters.

    Returns:
        Predicted rates.
    """
    # Handle both ArpsParams objects and dictionaries for backward compatibility
    if isinstance(p, Mapping):
        qi = float(p["qi"])
        di = float(p["di"])
        b = float(p["b"])
    else:
        qi, di, b = p.qi, p.di, p.b

    # Call optimized version
    return cast(np.ndarray, _predict_arps_numba(t, qi, di, b))


@numba.jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda f: f
def _predict_arps_numba(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Numba-optimized prediction function."""
    if b == 0.0:
        return q_exp(t, qi, di)
    if abs(b - 1.0) < 1e-9:
        return qi / (1 + di * t)
    return q_hyp(t, qi, di, b)


def estimate_reserves(params: ArpsParamsLike, t_max: float = 50.0) -> float:
    """Estimate ultimate recoverable reserves using Arps decline curves.

    Args:
        params: Arps parameters (qi, di, b).
        t_max: Maximum time for integration (years).

    Returns:
        Estimated reserves.
    """
    kind: Optional[str] = None
    if isinstance(params, Mapping):
        qi = float(params["qi"])
        di = float(params["di"])
        b = float(params["b"])
        kind = cast(Optional[str], params.get("kind"))
    else:
        qi, di, b = params.qi, params.di, params.b

    if di <= 0:
        raise ValueError("Decline rate must be positive")

    if kind is not None and kind not in ["exponential", "harmonic", "hyperbolic"]:
        raise ValueError(f"Invalid decline type: {kind}")

    if b == 0.0:  # Exponential
        return float(qi / di)
    elif np.isclose(b, 1.0):  # Harmonic
        # For harmonic decline: EUR = qi * ln(1 + di * t_max) / di
        # This gives higher reserves than exponential for same qi, di
        return float(qi * np.log(1 + di * t_max) / di)
    else:  # Hyperbolic
        if b >= 1.0:
            # For b >= 1, reserves approach infinity, use practical cutoff
            # Use harmonic approximation for b close to 1
            return float(qi * np.log(1 + di * t_max) / di)
        else:
            # For b < 1, analytical solution exists
            # EUR = qi * (1 - (1 + b*di*t_max)^((1-b)/b)) / (di * (1-b))
            if np.isclose(b, 1.0, atol=1e-6):
                return float(qi * np.log(1 + di * t_max) / di)
            else:
                # Use numerical integration for hyperbolic to ensure accuracy
                t_points = np.linspace(0, t_max, 1000)
                q_points = qi / ((1 + b * di * t_points) ** (1 / b))
                reserves = float(np.trapz(q_points, t_points))
                return float(max(reserves, 0))  # Ensure non-negative
