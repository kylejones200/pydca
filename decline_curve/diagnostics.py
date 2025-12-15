"""Diagnostic functions for decline curve analysis.

This module provides diagnostic curves and functions that help assess
the quality and characteristics of decline curve fits. These diagnostics
are essential for research-grade workflows and quality gates beyond RMSE.

References:
- Bourdet et al. (1983) - Derivative analysis
- SPEE REP 6 - Decline curve analysis standards
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DiagnosticCurves:
    """Diagnostic curves computed from production data.

    Attributes:
        time: Time array (days)
        rate: Production rate array
        cumulative: Cumulative production array
        d: Decline rate diagnostic (D = -d(ln q)/dt)
        b: Decline exponent diagnostic (b = d(ln D)/d(ln q))
        beta: Beta diagnostic (beta = d(ln q)/d(ln t))
        log_rate: Log of production rate
        log_time: Log of time
        derivative: Bourdet-style derivative (rate change)
    """

    time: np.ndarray
    rate: np.ndarray
    cumulative: np.ndarray
    d: np.ndarray
    b: np.ndarray
    beta: np.ndarray
    log_rate: np.ndarray
    log_time: np.ndarray
    derivative: np.ndarray


def compute_decline_rate_diagnostic(
    t: np.ndarray, q: np.ndarray, method: str = "log_derivative"
) -> np.ndarray:
    """Compute decline rate diagnostic D = -d(ln q)/dt.

    The decline rate D is the instantaneous decline rate. For Arps models:
    - Exponential: D = di (constant)
    - Hyperbolic: D = di / (1 + b*di*t)
    - Harmonic: D = di / (1 + di*t)

    Args:
        t: Time array (days)
        q: Production rate array
        method: Method for computing derivative
            ('log_derivative' or 'finite_difference')

    Returns:
        Decline rate diagnostic array D(t)

    Example:
        >>> t = np.linspace(0, 100, 100)
        >>> q = 100 * np.exp(-0.1 * t)  # Exponential decline
        >>> D = compute_decline_rate_diagnostic(t, q)
        >>> # D should be approximately constant at 0.1
    """
    if len(t) < 2 or len(q) < 2:
        return np.array([])

    # Filter out zero or negative rates
    valid_mask = q > 0
    if not np.any(valid_mask):
        return np.zeros_like(t)

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]
    log_q = np.log(q_valid)

    if method == "log_derivative":
        # D = -d(ln q)/dt using numerical derivative
        d_log_q = np.gradient(log_q, t_valid)
        D = -d_log_q
    else:  # finite_difference
        # D = -(1/q) * dq/dt
        dq = np.gradient(q_valid, t_valid)
        D = -(dq / q_valid)

    # Map back to original time array
    D_full = np.zeros_like(t)
    D_full[valid_mask] = D

    return D_full


def compute_b_diagnostic(
    t: np.ndarray, q: np.ndarray, D: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute decline exponent diagnostic b = d(ln D)/d(ln q).

    The b-factor diagnostic shows how the decline rate changes with rate.
    For Arps models:
    - Exponential: b = 0 (constant decline rate)
    - Hyperbolic: b = constant (0 < b < 1 typically)
    - Harmonic: b = 1 (decline rate proportional to rate)

    Args:
        t: Time array (days)
        q: Production rate array
        D: Optional pre-computed decline rate (if None, will compute)

    Returns:
        Decline exponent diagnostic array b(t)

    Example:
        >>> t = np.linspace(0, 100, 100)
        >>> q = 100 / (1 + 0.1 * t)  # Harmonic decline (b=1)
        >>> b = compute_b_diagnostic(t, q)
        >>> # b should be approximately constant at 1.0
    """
    if len(t) < 2 or len(q) < 2:
        return np.array([])

    # Filter out zero or negative rates
    valid_mask = q > 0
    if not np.any(valid_mask):
        return np.zeros_like(t)

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    # Compute D if not provided
    if D is None:
        D = compute_decline_rate_diagnostic(t, q)

    D_valid = D[valid_mask]

    # Filter out zero or negative decline rates
    valid_D_mask = D_valid > 0
    if not np.any(valid_D_mask):
        return np.zeros_like(t)

    t_final = t_valid[valid_D_mask]
    q_final = q_valid[valid_D_mask]
    D_final = D_valid[valid_D_mask]

    # b = d(ln D)/d(ln q)
    log_D = np.log(D_final)
    log_q = np.log(q_final)

    # Use gradient for derivative
    d_log_D = np.gradient(log_D, log_q)
    b = d_log_D

    # Map back to original time array
    b_full = np.zeros_like(t)
    b_full[valid_mask] = np.interp(t_valid, t_final, b)

    return b_full


def compute_beta_diagnostic(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute beta diagnostic beta = d(ln q)/d(ln t).

    The beta diagnostic shows the power-law relationship between rate and time.
    For Arps models:
    - Exponential: beta = -di*t (approximately linear in log-log space)
    - Hyperbolic: beta = -1/b (constant in log-log space)
    - Harmonic: beta = -1 (constant)

    Args:
        t: Time array (days)
        q: Production rate array

    Returns:
        Beta diagnostic array beta(t)

    Example:
        >>> t = np.linspace(1, 100, 100)  # Start at 1 to avoid log(0)
        >>> q = 100 / (1 + 0.1 * t)  # Harmonic decline
        >>> beta = compute_beta_diagnostic(t, q)
        >>> # beta should be approximately constant at -1.0
    """
    if len(t) < 2 or len(q) < 2:
        return np.array([])

    # Filter out zero or negative rates and times
    valid_mask = (q > 0) & (t > 0)
    if not np.any(valid_mask):
        return np.zeros_like(t)

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    log_t = np.log(t_valid)
    log_q = np.log(q_valid)

    # beta = d(ln q)/d(ln t)
    beta = np.gradient(log_q, log_t)

    # Map back to original time array
    beta_full = np.zeros_like(t)
    beta_full[valid_mask] = beta

    return beta_full


def compute_bourdet_derivative(
    t: np.ndarray, q: np.ndarray, L: float = 0.0
) -> np.ndarray:
    """Compute Bourdet-style derivative with smoothing.

    The Bourdet derivative uses a logarithmic spacing and smoothing to
    reduce noise in derivative calculations. This is particularly useful
    for noisy production data.

    Args:
        t: Time array (days)
        q: Production rate array
        L: Smoothing parameter (0 = no smoothing, higher = more smoothing)

    Returns:
        Bourdet derivative array dq/dt

    References:
        Bourdet, D., Ayoub, J.A., & Pirard, Y.M. (1989). Use of Pressure
        Derivative in Well-Test Interpretation. SPE Formation Evaluation.
    """
    if len(t) < 3 or len(q) < 3:
        return np.zeros_like(t)

    # Filter out zero or negative rates
    valid_mask = q > 0
    if not np.any(valid_mask):
        return np.zeros_like(t)

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    # Compute derivative
    dq = np.gradient(q_valid, t_valid)

    # Apply smoothing if requested
    if L > 0:
        # Simple moving average smoothing
        window_size = max(1, int(L * len(dq)))
        if window_size > 1:
            dq = np.convolve(dq, np.ones(window_size) / window_size, mode="same")

    # Map back to original time array
    dq_full = np.zeros_like(t)
    dq_full[valid_mask] = dq

    return dq_full


def compute_diagnostic_curves(
    t: np.ndarray,
    q: np.ndarray,
    cum: Optional[np.ndarray] = None,
    compute_cumulative: bool = True,
) -> DiagnosticCurves:
    """Compute all diagnostic curves from production data.

    This is the main entry point for diagnostic analysis. It computes
    all standard diagnostic curves used in decline curve analysis.

    Args:
        t: Time array (days)
        q: Production rate array
        cum: Optional cumulative production array (if None, will compute)
        compute_cumulative: Whether to compute cumulative if not provided

    Returns:
        DiagnosticCurves object with all computed diagnostics

    Example:
        >>> t = np.linspace(0, 100, 100)
        >>> q = 100 * np.exp(-0.1 * t)  # Exponential decline
        >>> diagnostics = compute_diagnostic_curves(t, q)
        >>> # Access diagnostics
        >>> D = diagnostics.d  # Decline rate
        >>> b = diagnostics.b  # Decline exponent
    """
    if len(t) != len(q):
        raise ValueError("Time and rate arrays must have same length")

    # Compute cumulative if needed
    if cum is None and compute_cumulative:
        # Simple trapezoidal integration
        cum = np.zeros_like(q)
        for i in range(1, len(q)):
            cum[i] = cum[i - 1] + np.trapz(q[i - 1 : i + 1], t[i - 1 : i + 1])
    elif cum is None:
        cum = np.zeros_like(q)

    # Filter for valid data
    valid_mask = (q > 0) & (t > 0)

    # Compute diagnostics
    D = compute_decline_rate_diagnostic(t, q)
    b = compute_b_diagnostic(t, q, D)
    beta = compute_beta_diagnostic(t, q)
    derivative = compute_bourdet_derivative(t, q)

    # Log transforms
    log_rate = np.zeros_like(q)
    log_time = np.zeros_like(t)
    log_rate[valid_mask] = np.log(q[valid_mask])
    log_time[valid_mask] = np.log(t[valid_mask])

    return DiagnosticCurves(
        time=t,
        rate=q,
        cumulative=cum,
        d=D,
        b=b,
        beta=beta,
        log_rate=log_rate,
        log_time=log_time,
        derivative=derivative,
    )


def identify_decline_type(diagnostics: DiagnosticCurves) -> Dict[str, float]:
    """Identify decline type from diagnostic curves.

    Analyzes diagnostic curves to determine which Arps model type
    best describes the data based on the behavior of D and b.

    Args:
        diagnostics: DiagnosticCurves object

    Returns:
        Dictionary with:
        - 'type': 'exponential', 'hyperbolic', 'harmonic', or 'unknown'
        - 'confidence': Confidence score (0-1)
        - 'mean_b': Mean b value
        - 'std_b': Standard deviation of b
        - 'mean_D': Mean D value
        - 'std_D': Standard deviation of D
    """
    # Filter valid data
    valid_mask = (diagnostics.b != 0) & (diagnostics.d > 0) & np.isfinite(diagnostics.b)

    if not np.any(valid_mask):
        return {
            "type": "unknown",
            "confidence": 0.0,
            "mean_b": 0.0,
            "std_b": 0.0,
            "mean_D": 0.0,
            "std_D": 0.0,
        }

    b_valid = diagnostics.b[valid_mask]
    D_valid = diagnostics.d[valid_mask]

    mean_b = np.mean(b_valid)
    std_b = np.std(b_valid)
    mean_D = np.mean(D_valid)
    std_D = np.std(D_valid)

    # Classify based on b value
    # Exponential: b ≈ 0
    # Hyperbolic: 0 < b < 1
    # Harmonic: b ≈ 1

    if std_b < 0.1:  # Low variation in b
        if abs(mean_b) < 0.1:
            decline_type = "exponential"
            confidence = 1.0 - min(std_b / 0.1, 1.0)
        elif 0.1 < mean_b < 0.9:
            decline_type = "hyperbolic"
            confidence = 1.0 - min(std_b / 0.1, 1.0)
        elif abs(mean_b - 1.0) < 0.1:
            decline_type = "harmonic"
            confidence = 1.0 - min(std_b / 0.1, 1.0)
        else:
            decline_type = "unknown"
            confidence = 0.5
    else:
        # High variation - could be transitional or noisy
        decline_type = "hyperbolic"  # Most common case
        confidence = 0.5

    return {
        "type": decline_type,
        "confidence": confidence,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_D": mean_D,
        "std_D": std_D,
    }
