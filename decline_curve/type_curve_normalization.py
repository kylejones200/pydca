"""Type curve normalization for Arps decline curve matching.

Implements Fetkovich-style type curve normalization: production data is
normalized by reference qi/Di values, matched against a family of theoretical
Arps curves (varying b), and de-normalized to obtain well-specific parameters.

Ported and adapted from ressmith type_curves.py, replacing the ressmith
``arps_hyperbolic`` dependency with pydca's own ``predict_arps`` / ``q_hyp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from .models import ArpsParams, predict_arps


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TypeCurveMatch:
    """Result of matching production data to a type curve.

    Attributes:
        matched_params: Best-fit :class:`~decline_curve.models.ArpsParams`.
        match_error: Root-mean-squared error on the normalized rate axis.
        correlation: Pearson correlation coefficient between data and matched curve.
        matched_curve: Normalized type curve at the matched parameters (same
            length as the input time array).
    """

    matched_params: ArpsParams
    match_error: float
    correlation: float
    matched_curve: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def generate_arps_type_curve(
    qi_normalized: float,
    di_normalized: float,
    b: float,
    time_normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a normalized Arps type curve.

    Computes the hyperbolic (or exponential/harmonic) decline at normalized
    parameter values. The result is the unit-dimensionless production curve
    against normalized time.

    Args:
        qi_normalized: Normalized initial rate (dimensionless, typically 1.0).
        di_normalized: Normalized initial decline rate (1/time).
        b: Arps curvature factor (0 = exponential, 1 = harmonic, >1 transient).
        time_normalized: Normalized time array (same units as ``di_normalized``).

    Returns:
        Tuple of ``(t_norm, q_norm)`` — the time and rate arrays.

    Example:
        >>> t = np.linspace(0, 10, 100)
        >>> t_norm, q_norm = generate_arps_type_curve(1.0, 0.5, 1.2, t)
    """
    params = ArpsParams(qi=qi_normalized, di=di_normalized, b=b)
    q_norm = predict_arps(time_normalized, params)
    return time_normalized.copy(), q_norm


def normalize_production_data(
    time: np.ndarray,
    rate: np.ndarray,
    qi_ref: float,
    di_ref: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalize production data to dimensionless time and rate.

    Uses the reference qi and Di values to create dimensionless variables
    suitable for type curve matching on a log-log plot.

    Args:
        time: Time array (months or any consistent unit).
        rate: Production rate array (BOE/month or any consistent unit).
        qi_ref: Reference initial rate for normalization.
        di_ref: Reference initial decline rate for normalization.

    Returns:
        Tuple of ``(t_normalized, q_normalized, factors)`` where ``factors``
        is a dict with ``qi_ref`` and ``di_ref`` for de-normalization.

    Raises:
        ValueError: If ``qi_ref`` or ``di_ref`` are non-positive.

    Example:
        >>> t_norm, q_norm, factors = normalize_production_data(t, q, qi_ref=1000, di_ref=0.1)
        >>> # De-normalize: q_actual = q_norm * factors['qi_ref']
    """
    if qi_ref <= 0:
        raise ValueError(f"qi_ref must be positive, got {qi_ref}")
    if di_ref <= 0:
        raise ValueError(f"di_ref must be positive, got {di_ref}")

    t_norm = np.asarray(time, dtype=float) * di_ref
    q_norm = np.asarray(rate, dtype=float) / qi_ref

    factors = {"qi_ref": qi_ref, "di_ref": di_ref}
    return t_norm, q_norm, factors


def match_type_curve(
    time: np.ndarray,
    rate: np.ndarray,
    b_values: Optional[np.ndarray | list[float]] = None,
    initial_guess: Optional[tuple[float, float, float]] = None,
) -> TypeCurveMatch:
    """Match production data to an Arps type curve family.

    Searches a family of b-values (or fits b freely) to find the Arps
    parameters that best reproduce the observed production decline.

    The matching procedure:

    1. For each b in ``b_values``, fix b and fit (qi, Di) via nonlinear
       least-squares (``scipy.curve_fit``).
    2. Select the b that minimises RMSE on the rate axis.
    3. If ``b_values`` is None, fit (qi, Di, b) jointly with an initial guess.

    Args:
        time: Time array (months, 0-based).
        rate: Observed production rate array.
        b_values: Optional array of b-factors to search. If ``None``, b is
            fitted freely alongside qi and Di.
        initial_guess: Initial ``(qi, di, b)`` for the free-b fit.
            Defaults to ``(rate[0], 0.1, 1.0)`` if not provided.

    Returns:
        :class:`TypeCurveMatch` with best-fit parameters and diagnostics.

    Raises:
        ValueError: If ``time`` and ``rate`` have different lengths or fewer
            than 3 points.

    Example:
        >>> match = match_type_curve(time, rate, b_values=np.arange(0.5, 2.1, 0.1))
        >>> print(match.matched_params)
        >>> print(f"RMSE={match.match_error:.2f}  R={match.correlation:.3f}")
    """
    time = np.asarray(time, dtype=float)
    rate = np.asarray(rate, dtype=float)

    if len(time) != len(rate):
        raise ValueError("time and rate must have the same length")
    if len(time) < 3:
        raise ValueError("Need at least 3 data points to match a type curve")

    if b_values is None:
        # Free-b fit
        qi0, di0, b0 = initial_guess or (float(rate[0]), 0.1, 1.0)

        def _model_free(t, qi, di, b):
            p = ArpsParams(qi=qi, di=di, b=b)
            return predict_arps(t, p)

        try:
            popt, _ = curve_fit(
                _model_free,
                time,
                rate,
                p0=[qi0, di0, b0],
                bounds=([0, 1e-6, 0], [1e9, 10.0, 2.0]),
                maxfev=5000,
            )
            best_qi, best_di, best_b = popt
        except RuntimeError:
            best_qi, best_di, best_b = float(rate[0]), 0.1, 1.0

        best_params = ArpsParams(qi=best_qi, di=best_di, b=best_b)
        q_hat = predict_arps(time, best_params)
        rmse = float(np.sqrt(np.mean((rate - q_hat) ** 2)))
        corr = float(np.corrcoef(rate, q_hat)[0, 1]) if len(rate) > 1 else 0.0
        return TypeCurveMatch(
            matched_params=best_params,
            match_error=rmse,
            correlation=corr,
            matched_curve=q_hat,
        )

    # Sweep over b-values
    b_values = np.asarray(b_values, dtype=float)
    best_rmse = float("inf")
    best_result: TypeCurveMatch | None = None

    for b_candidate in b_values:
        b_fixed = float(b_candidate)

        def _model_fixed_b(t, qi, di, b=b_fixed):
            p = ArpsParams(qi=qi, di=di, b=b)
            return predict_arps(t, p)

        qi0 = float(rate[0]) if rate[0] > 0 else 1.0
        di0 = 0.1
        try:
            popt, _ = curve_fit(
                _model_fixed_b,
                time,
                rate,
                p0=[qi0, di0],
                bounds=([0, 1e-6], [1e9, 10.0]),
                maxfev=3000,
            )
            qi_fit, di_fit = popt
        except RuntimeError:
            qi_fit, di_fit = qi0, di0

        params = ArpsParams(qi=qi_fit, di=di_fit, b=b_fixed)
        q_hat = predict_arps(time, params)
        rmse = float(np.sqrt(np.mean((rate - q_hat) ** 2)))

        if rmse < best_rmse:
            best_rmse = rmse
            corr = float(np.corrcoef(rate, q_hat)[0, 1]) if len(rate) > 1 else 0.0
            best_result = TypeCurveMatch(
                matched_params=params,
                match_error=rmse,
                correlation=corr,
                matched_curve=q_hat,
            )

    if best_result is None:
        # Fallback: hyperbolic b=1
        params = ArpsParams(qi=float(rate[0]), di=0.1, b=1.0)
        q_hat = predict_arps(time, params)
        rmse = float(np.sqrt(np.mean((rate - q_hat) ** 2)))
        corr = float(np.corrcoef(rate, q_hat)[0, 1]) if len(rate) > 1 else 0.0
        best_result = TypeCurveMatch(
            matched_params=params,
            match_error=rmse,
            correlation=corr,
            matched_curve=q_hat,
        )

    return best_result


def denormalize_match(
    match: TypeCurveMatch,
    factors: dict,
) -> ArpsParams:
    """Convert a normalized type curve match back to well-scale parameters.

    Reverses the normalization applied by :func:`normalize_production_data`.

    Args:
        match: Result from :func:`match_type_curve` fitted on normalized data.
        factors: Normalization factors dict (``qi_ref``, ``di_ref``) returned
            by :func:`normalize_production_data`.

    Returns:
        :class:`~decline_curve.models.ArpsParams` in original production units.

    Example:
        >>> t_norm, q_norm, factors = normalize_production_data(t, q, 1000, 0.1)
        >>> match = match_type_curve(t_norm, q_norm, b_values=np.linspace(0.5, 2, 16))
        >>> actual_params = denormalize_match(match, factors)
    """
    qi_ref = factors["qi_ref"]
    di_ref = factors["di_ref"]
    return ArpsParams(
        qi=match.matched_params.qi * qi_ref,
        di=match.matched_params.di * di_ref,
        b=match.matched_params.b,
    )
