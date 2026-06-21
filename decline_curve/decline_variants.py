"""Additional decline curve variants for practical applications.

This module provides decline variants that matter in practice:
- Fixed terminal decline: Transition to a fixed terminal decline rate
- Time to boundary: Constraints that prevent absurd tail behavior
- Duong: Shale/unconventional transient-flow model
- PLE (Power-Law Exponential): Tight gas loss-ratio model (Ilk et al. 2008)
- SEPD (Stretched Exponential): Anomalous diffusion in unconventionals
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .logging_config import get_logger
from .models import (
    ArpsParams,
    DuongParams,
    ModifiedHyperbolicParams,
    PLEParams,
    SEPDParams,
    fit_arps,
    predict_arps,
)

logger = get_logger(__name__)


def fixed_terminal_decline(
    series: pd.Series,
    kind: str = "hyperbolic",
    terminal_decline_rate: float = 0.05,
    transition_criteria: str = "rate",
    transition_value: Optional[float] = None,
) -> tuple[pd.Series, dict]:
    """
    Fit decline curve with transition to fixed terminal decline rate.

    This variant prevents unrealistic long-term forecasts by transitioning
    to a fixed terminal decline rate (typically 5-10% per year) when the
    hyperbolic decline rate becomes too low.

    Args:
        series: Historical production time series
        kind: Initial decline type ('exponential', 'harmonic', 'hyperbolic')
        terminal_decline_rate: Fixed annual decline rate for terminal phase
            (e.g., 0.05 = 5% per year)
        transition_criteria: When to transition:
            - 'rate': Transition when decline rate reaches threshold
            - 'time': Transition at fixed time
            - 'cumulative': Transition at cumulative volume
        transition_value: Threshold value for transition
            - If 'rate': minimum decline rate before transition (default: terminal_decline_rate)
            - If 'time': time in months (default: 60 months)
            - If 'cumulative': cumulative volume (default: None, uses rate-based)

    Returns:
        Tuple of (forecast_series, params_dict) where params_dict contains:
        - initial_params: ArpsParams for initial phase
        - terminal_decline_rate: Terminal decline rate
        - transition_time: Time of transition
        - transition_rate: Rate at transition

    Example:
        >>> import pandas as pd
        >>> from decline_curve.decline_variants import fixed_terminal_decline
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> forecast, params = fixed_terminal_decline(
        ...     production,
        ...     kind='hyperbolic',
        ...     terminal_decline_rate=0.06  # 6% per year
        ... )
    """
    if transition_criteria == "rate":
        if transition_value is None:
            transition_value = terminal_decline_rate
    elif transition_criteria == "time":
        if transition_value is None:
            transition_value = 60.0  # 5 years default
    elif transition_criteria == "cumulative":
        if transition_value is None:
            transition_value = None  # Will use rate-based
    else:
        raise ValueError(
            f"Unknown transition_criteria: {transition_criteria}. "
            "Must be 'rate', 'time', or 'cumulative'"
        )

    # Fit initial decline
    t = np.arange(len(series))
    q = series.values
    initial_params = fit_arps(t, q, kind=kind)

    # Convert terminal decline rate from annual to monthly
    terminal_decline_monthly = terminal_decline_rate / 12.0

    # Determine transition point
    if transition_criteria == "time":
        transition_time = transition_value
    elif transition_criteria == "rate":
        # Find when decline rate reaches threshold
        # For hyperbolic: D(t) = di / (1 + b*di*t)
        # Solve for t when D(t) = transition_value
        di = initial_params.di
        b = initial_params.b
        if b > 0:
            transition_time = (di / transition_value - 1) / (b * di)
            transition_time = max(0, min(transition_time, len(series) * 2))
        else:
            # Exponential: constant decline, transition immediately
            transition_time = 0.0
    else:  # cumulative
        # Use rate-based transition if cumulative not specified
        di = initial_params.di
        b = initial_params.b
        if b > 0:
            transition_time = (di / terminal_decline_monthly - 1) / (b * di)
        else:
            transition_time = 0.0

    # Calculate rate at transition
    t_transition = np.array([transition_time])
    q_transition = predict_arps(t_transition, initial_params)[0]

    # Generate forecast
    horizon = 240  # 20 years
    t_full = np.arange(0, len(series) + horizon, 1.0)

    forecast_rates = np.zeros_like(t_full)

    for i, t_val in enumerate(t_full):
        if t_val <= transition_time:
            # Initial phase
            forecast_rates[i] = predict_arps(np.array([t_val]), initial_params)[0]
        else:
            # Terminal phase: exponential decline from transition point
            t_terminal = t_val - transition_time
            forecast_rates[i] = q_transition * np.exp(
                -terminal_decline_monthly * t_terminal
            )

    # Create forecast series
    dates = pd.date_range(
        series.index[0], periods=len(forecast_rates), freq=series.index.freq or "MS"
    )
    forecast_series = pd.Series(forecast_rates, index=dates, name="forecast")

    params_dict = {
        "initial_params": initial_params,
        "terminal_decline_rate": terminal_decline_rate,
        "transition_time": transition_time,
        "transition_rate": q_transition,
        "kind": kind,
    }

    return forecast_series, params_dict


# ---------------------------------------------------------------------------
# Duong Model  (shale / transient flow)
# q(t) = q1 * t^(-m) * exp(a/(1-m) * (t^(1-m) - 1))
# t is 1-indexed (t=1 at first production month)
# ---------------------------------------------------------------------------


def _duong_fn(t1: np.ndarray, q1: float, a: float, m: float) -> np.ndarray:
    """Evaluate Duong rate at 1-indexed time array.

    When m→1 the formula q1*t^(-m)*exp(a/(1-m)*(t^(1-m)-1)) has a removable
    singularity; via L'Hôpital the limit is q1*t^(-(1+a)).
    """
    if abs(m - 1.0) < 1e-6:
        return q1 * t1 ** (-(1.0 + a))
    return q1 * t1 ** (-m) * np.exp(a / (1.0 - m) * (t1 ** (1.0 - m) - 1.0))


def fit_duong(t: np.ndarray, q: np.ndarray) -> DuongParams:
    """Fit Duong model to production data.

    Uses multi-start optimization to avoid local minima in the non-convex
    Duong parameter space (the a/(1-m) term creates a degenerate manifold
    near m=1).

    Args:
        t: 0-indexed time array (months from first production).
        q: Production rates.

    Returns:
        DuongParams with q1, a, m.
    """
    t1 = t.astype(float) + 1.0
    valid = q > 0
    if not valid.any():
        return DuongParams(q1=1.0, a=1.0, m=0.9)

    q_valid, t1_valid = q[valid], t1[valid]
    qi_guess = float(q_valid[0])

    # Multi-start: try several initial (a, m) pairs to escape local minima
    start_points = [
        [qi_guess, 1.5, 1.2],
        [qi_guess, 1.0, 0.9],
        [qi_guess, 2.0, 1.5],
        [qi_guess, 0.5, 0.7],
    ]
    best_residual = np.inf
    best_params = DuongParams(q1=qi_guess, a=1.0, m=0.9)

    for p0 in start_points:
        try:
            popt, _ = curve_fit(
                _duong_fn,
                t1_valid,
                q_valid,
                p0=p0,
                bounds=([0.0, 1e-6, 0.3], [np.inf, 10.0, 3.0]),
                maxfev=10000,
            )
            pred = _duong_fn(t1_valid, *popt)
            residual = float(np.sum((pred - q_valid) ** 2))
            if residual < best_residual:
                best_residual = residual
                best_params = DuongParams(
                    q1=float(popt[0]), a=float(popt[1]), m=float(popt[2])
                )
        except Exception:
            continue

    if best_residual == np.inf:
        logger.debug("Duong fit failed for all starting points; using defaults")
    return best_params


def predict_duong(t: np.ndarray, params: DuongParams) -> np.ndarray:
    """Predict Duong model rates.

    Args:
        t: 0-indexed time array (months from first production).
        params: DuongParams.

    Returns:
        Predicted production rates.
    """
    t1 = t.astype(float) + 1.0
    return _duong_fn(t1, params.q1, params.a, params.m)


def eur_duong(
    params: DuongParams,
    t_max: int = 240,
    econ_limit: float = 0.0,
) -> float:
    """Estimate EUR for Duong model via numerical integration.

    Args:
        params: DuongParams.
        t_max: Maximum forecast months (default 240 = 20 years).
        econ_limit: Abandon rate; integrate only while rate >= econ_limit.

    Returns:
        EUR in same units as production rates * months.
    """
    t = np.arange(t_max, dtype=float)
    rates = predict_duong(t, params)
    if econ_limit > 0:
        # Find the last index where rate is at or above econ_limit to handle
        # the non-monotone early-time behavior of the Duong model.
        above = np.where(rates >= econ_limit)[0]
        stop = int(above[-1]) + 1 if len(above) > 0 else 0
        rates = rates[:stop]
    return float(np.trapz(rates))


# ---------------------------------------------------------------------------
# Power-Law Exponential (PLE / Ilk et al. 2008)
# q(t) = qi * exp(-D_inf*t - (D1 - D_inf)/n * t^n)
# ---------------------------------------------------------------------------


def _ple_fn(t: np.ndarray, qi: float, D_inf: float, D1: float, n: float) -> np.ndarray:
    """Evaluate PLE rate."""
    return qi * np.exp(-D_inf * t - (D1 - D_inf) / n * t**n)


def fit_ple(t: np.ndarray, q: np.ndarray) -> PLEParams:
    """Fit Power-Law Exponential model.

    Args:
        t: 0-indexed time array (months).
        q: Production rates.

    Returns:
        PLEParams with qi, D_inf, D1, n.
    """
    t_f = t.astype(float)
    valid = q > 0
    if not valid.any():
        return PLEParams(qi=1.0, D_inf=0.001, D1=0.1, n=0.4)

    qi_guess = float(q[valid][0])

    # Guard t=0 for initial D1 guess (loss ratio at t=1)
    t_safe = np.where(t_f[valid] == 0, 1e-6, t_f[valid])

    try:
        popt, _ = curve_fit(
            _ple_fn,
            t_safe,
            q[valid],
            p0=[qi_guess, 0.001, 0.05, 0.4],
            bounds=([0.0, 0.0, 0.0, 0.01], [np.inf, 1.0, 2.0, 1.0]),
            maxfev=20000,
        )
        return PLEParams(
            qi=float(popt[0]),
            D_inf=float(popt[1]),
            D1=float(popt[2]),
            n=float(popt[3]),
        )
    except Exception:
        logger.debug("PLE fit failed; returning default parameters")
        return PLEParams(qi=qi_guess, D_inf=0.001, D1=0.05, n=0.4)


def predict_ple(t: np.ndarray, params: PLEParams) -> np.ndarray:
    """Predict PLE model rates.

    Args:
        t: 0-indexed time array.
        params: PLEParams.

    Returns:
        Predicted production rates.
    """
    t_f = np.where(t.astype(float) == 0, 1e-6, t.astype(float))
    return _ple_fn(t_f, params.qi, params.D_inf, params.D1, params.n)


def eur_ple(
    params: PLEParams,
    t_max: int = 240,
    econ_limit: float = 0.0,
) -> float:
    """Estimate EUR for PLE via numerical integration.

    Args:
        params: PLEParams.
        t_max: Maximum forecast months.
        econ_limit: Abandon rate.

    Returns:
        EUR in same units as production rates * months.
    """
    t = np.arange(t_max, dtype=float)
    rates = predict_ple(t, params)
    if econ_limit > 0:
        stop = np.searchsorted(-rates, -econ_limit)
        rates = rates[:stop]
    return float(np.trapz(rates))


# ---------------------------------------------------------------------------
# Stretched Exponential Production Decline (SEPD)
# q(t) = qi * exp(-(t/tau)^n)
# EUR = qi * (tau/n) * Γ(1/n)  [closed form]
# ---------------------------------------------------------------------------


def _sepd_fn(t: np.ndarray, qi: float, tau: float, n: float) -> np.ndarray:
    """Evaluate SEPD rate."""
    return qi * np.exp(-((t / tau) ** n))


def fit_sepd(t: np.ndarray, q: np.ndarray) -> SEPDParams:
    """Fit Stretched Exponential Production Decline model.

    Args:
        t: 0-indexed time array (months).
        q: Production rates.

    Returns:
        SEPDParams with qi, tau, n.
    """
    t_f = t.astype(float)
    valid = q > 0
    if not valid.any():
        return SEPDParams(qi=1.0, tau=24.0, n=0.5)

    qi_guess = float(q[valid][0])
    # Estimate tau from where rate drops to ~37% of initial
    half_life_idx = np.searchsorted(-q[valid], -qi_guess * 0.37)
    tau_guess = float(t_f[valid][half_life_idx]) if half_life_idx < len(t_f[valid]) else 24.0

    try:
        popt, _ = curve_fit(
            _sepd_fn,
            t_f[valid],
            q[valid],
            p0=[qi_guess, max(tau_guess, 1.0), 0.5],
            bounds=([0.0, 1e-3, 0.01], [np.inf, 1e6, 1.0]),
            maxfev=10000,
        )
        return SEPDParams(qi=float(popt[0]), tau=float(popt[1]), n=float(popt[2]))
    except Exception:
        logger.debug("SEPD fit failed; returning default parameters")
        return SEPDParams(qi=qi_guess, tau=max(tau_guess, 1.0), n=0.5)


def predict_sepd(t: np.ndarray, params: SEPDParams) -> np.ndarray:
    """Predict SEPD model rates.

    Args:
        t: 0-indexed time array.
        params: SEPDParams.

    Returns:
        Predicted production rates.
    """
    return _sepd_fn(t.astype(float), params.qi, params.tau, params.n)


def eur_sepd(params: SEPDParams, econ_limit: float = 0.0) -> float:
    """Estimate EUR for SEPD.

    Uses the closed-form solution when econ_limit=0:
        EUR = qi * (tau/n) * Γ(1/n)

    Falls back to numerical integration when econ_limit > 0.

    Args:
        params: SEPDParams.
        econ_limit: Abandon rate (BOE/month); 0 means integrate to infinity.

    Returns:
        EUR in same units as production rates * months.
    """
    from scipy.special import gamma as _gamma

    if econ_limit <= 0.0:
        return float(params.qi * (params.tau / params.n) * _gamma(1.0 / params.n))

    # Find economic-limit time and integrate numerically
    from scipy.integrate import quad
    from scipy.optimize import brentq

    try:
        t_econ = brentq(
            lambda t: _sepd_fn(np.array([t]), params.qi, params.tau, params.n)[0]
            - econ_limit,
            0.0,
            params.tau * 1000.0,
        )
    except ValueError:
        t_econ = params.tau * 1000.0

    result, _ = quad(
        lambda t: float(_sepd_fn(np.array([t]), params.qi, params.tau, params.n)[0]),
        0.0,
        t_econ,
    )
    return float(result)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Modified Hyperbolic (SEC / industry standard for shale)
# ---------------------------------------------------------------------------


def _mh_t_switch(params: ModifiedHyperbolicParams) -> float:
    """Derive the switch time from di falling to d_lim.

    D(t) = di / (1 + b * di * t) = d_lim  →  t_switch = (di/d_lim - 1) / (b * di)
    Returns 0.0 if di is already at or below d_lim.
    """
    if params.di <= params.d_lim or params.b <= 0:
        return 0.0
    return (params.di / params.d_lim - 1.0) / (params.b * params.di)


def fit_modified_hyperbolic(
    t: np.ndarray,
    q: np.ndarray,
    d_lim: float = 0.005,
) -> ModifiedHyperbolicParams:
    """Fit Modified Hyperbolic (hyperbolic-to-exponential) model.

    Fits a standard hyperbolic curve (qi, di, b) and attaches the terminal
    decline floor D_lim. The switch time is derived, not fitted, so the model
    has the same 3 degrees of freedom as plain hyperbolic — preventing the
    over-parameterisation of a 5-param free switch.

    Args:
        t: 0-indexed time array (months from first production).
        q: Production rates (BOE/month).
        d_lim: Terminal decline rate floor (1/month). Default 0.005 = 6%/year.
            Common choices: 0.004 (5%/yr), 0.005 (6%/yr), 0.008 (10%/yr).

    Returns:
        ModifiedHyperbolicParams.
    """
    arps = fit_arps(t, q, kind="hyperbolic")
    return ModifiedHyperbolicParams(qi=arps.qi, di=arps.di, b=arps.b, d_lim=d_lim)


def predict_modified_hyperbolic(
    t: np.ndarray,
    params: ModifiedHyperbolicParams,
) -> np.ndarray:
    """Predict Modified Hyperbolic rates.

    Args:
        t: 0-indexed time array (months).
        params: ModifiedHyperbolicParams.

    Returns:
        Production rates.
    """
    t = t.astype(float)
    t_sw = _mh_t_switch(params)
    q = np.empty_like(t)

    # Hyperbolic phase
    mask_h = t < t_sw
    if mask_h.any():
        th = t[mask_h]
        q[mask_h] = params.qi / np.power(1.0 + params.b * params.di * th, 1.0 / params.b)

    # Exponential phase
    mask_e = ~mask_h
    if mask_e.any():
        if t_sw > 0:
            q_sw = params.qi / (1.0 + params.b * params.di * t_sw) ** (1.0 / params.b)
        else:
            q_sw = params.qi
        q[mask_e] = q_sw * np.exp(-params.d_lim * (t[mask_e] - t_sw))

    return q


def eur_modified_hyperbolic(
    params: ModifiedHyperbolicParams,
    t_max: int = 360,
    econ_limit: float = 0.0,
) -> float:
    """EUR for Modified Hyperbolic via analytic closed form.

    For T → ∞: EUR = Np_hyp(t_switch) + q_switch / d_lim (exact).
    When econ_limit > 0 or t_max terminates in the exponential tail, the
    exponential contribution is truncated accordingly.

    Args:
        params: ModifiedHyperbolicParams.
        t_max: Forecast horizon in months. Use 9999 for ~infinite.
        econ_limit: Abandon rate; production below this is excluded.

    Returns:
        EUR in BOE.
    """
    t_sw = _mh_t_switch(params)
    b, di, qi, d_lim = params.b, params.di, params.qi, params.d_lim

    # Cumulative hyperbolic to t_switch
    if t_sw <= 0:
        np_hyp = 0.0
        q_sw = qi
        t_sw = 0.0
    else:
        if b == 0.0:
            np_hyp = (qi / di) * (1.0 - np.exp(-di * t_sw))
        elif abs(b - 1.0) < 1e-9:
            np_hyp = (qi / di) * np.log(1.0 + di * t_sw)
        else:
            np_hyp = (qi / (di * (1.0 - b))) * (
                1.0 - (1.0 + b * di * t_sw) ** ((b - 1.0) / b)
            )
        q_sw = qi / (1.0 + b * di * t_sw) ** (1.0 / b)

    # Exponential tail: truncate at t_max or econ_limit
    if econ_limit > 0 and q_sw > econ_limit:
        # Time within exponential phase when rate hits econ_limit
        t_econ = np.log(q_sw / econ_limit) / d_lim
    else:
        t_econ = float("inf")

    t_end_exp = min(t_max - t_sw, t_econ) if t_max > t_sw else 0.0
    t_end_exp = max(t_end_exp, 0.0)

    np_exp = (q_sw / d_lim) * (1.0 - np.exp(-d_lim * t_end_exp))
    return float(np_hyp + np_exp)


# Convenience dispatcher for use in Forecaster
# ---------------------------------------------------------------------------


def forecast_variant(
    series: pd.Series,
    kind: str,
    horizon: int = 12,
    d_lim: float = 0.005,
) -> pd.Series:
    """Forecast using a shale-era or modified decline variant.

    Dispatches to Duong, PLE, SEPD, or Modified Hyperbolic based on *kind*.

    Args:
        series: Historical production time series with DatetimeIndex.
        kind: One of 'duong', 'ple', 'sepd', 'modified_hyperbolic'.
        horizon: Number of future periods to forecast.
        d_lim: Terminal decline floor for 'modified_hyperbolic' (1/month).

    Returns:
        Series spanning history + forecast with same frequency as input.
    """
    t = np.arange(len(series), dtype=float)
    q = series.to_numpy()
    full_t = np.arange(len(series) + horizon, dtype=float)

    if kind == "duong":
        params = fit_duong(t, q)
        yhat = predict_duong(full_t, params)
    elif kind == "ple":
        params = fit_ple(t, q)
        yhat = predict_ple(full_t, params)
    elif kind == "sepd":
        params = fit_sepd(t, q)
        yhat = predict_sepd(full_t, params)
    elif kind in ("modified_hyperbolic", "mh"):
        params = fit_modified_hyperbolic(t, q, d_lim=d_lim)
        yhat = predict_modified_hyperbolic(full_t, params)
    else:
        raise ValueError(
            f"Unknown variant kind: {kind!r}. "
            "Expected 'duong', 'ple', 'sepd', or 'modified_hyperbolic'."
        )

    freq = series.index.freq or pd.infer_freq(series.index) or "MS"
    idx = pd.date_range(series.index[0], periods=len(yhat), freq=freq)
    return pd.Series(yhat, index=idx, name=f"arps_{kind}")
def time_to_boundary(
    series: pd.Series,
    kind: str = "hyperbolic",
    max_time: float = 240.0,
    min_rate: float = 1.0,
    enforce_bounds: bool = True,
) -> tuple[pd.Series, dict]:
    """
    Fit decline curve with time-to-boundary constraints.

    This variant prevents absurd tail behavior by:
    1. Limiting forecast to maximum time horizon
    2. Setting minimum economic rate threshold
    3. Enforcing reasonable parameter bounds

    Args:
        series: Historical production time series
        kind: Decline type ('exponential', 'harmonic', 'hyperbolic')
        max_time: Maximum forecast time in months (default: 240 = 20 years)
        min_rate: Minimum economic production rate (default: 1.0)
        enforce_bounds: If True, enforce parameter bounds to prevent unrealistic forecasts

    Returns:
        Tuple of (forecast_series, params_dict) where params_dict contains:
        - params: Fitted ArpsParams (with bounds enforced if requested)
        - max_time: Maximum time used
        - min_rate: Minimum rate threshold
        - time_to_min_rate: Time when forecast reaches min_rate

    Example:
        >>> import pandas as pd
        >>> from decline_curve.decline_variants import time_to_boundary
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> forecast, params = time_to_boundary(
        ...     production,
        ...     max_time=300,  # 25 years
        ...     min_rate=5.0   # 5 bbl/month minimum
        ... )
    """
    # Fit decline
    t = np.arange(len(series))
    q = series.values
    params = fit_arps(t, q, kind=kind)

    # Enforce bounds if requested
    if enforce_bounds:
        # Reasonable bounds for parameters
        params.qi = max(1.0, min(params.qi, 1e6))  # qi between 1 and 1M
        params.di = max(
            0.001, min(params.di, 10.0)
        )  # di between 0.1% and 1000% per month
        if kind == "hyperbolic":
            params.b = max(0.0, min(params.b, 2.0))  # b between 0 and 2

    # Find time when rate reaches minimum
    # For hyperbolic: q(t) = qi / (1 + b*di*t)^(1/b) = min_rate
    # Solve: (1 + b*di*t) = (qi/min_rate)^b
    # t = ((qi/min_rate)^b - 1) / (b*di)
    if kind == "exponential":
        # q(t) = qi * exp(-di*t) = min_rate
        # t = -ln(min_rate/qi) / di
        if params.qi > min_rate and params.di > 0:
            time_to_min_rate = -np.log(min_rate / params.qi) / params.di
        else:
            time_to_min_rate = max_time
    elif kind == "harmonic":
        # q(t) = qi / (1 + di*t) = min_rate
        # t = (qi/min_rate - 1) / di
        if params.qi > min_rate and params.di > 0:
            time_to_min_rate = (params.qi / min_rate - 1) / params.di
        else:
            time_to_min_rate = max_time
    else:  # hyperbolic
        if params.b > 0 and params.qi > min_rate and params.di > 0:
            time_to_min_rate = ((params.qi / min_rate) ** params.b - 1) / (
                params.b * params.di
            )
        else:
            time_to_min_rate = max_time

    # Use minimum of max_time and time_to_min_rate
    effective_max_time = min(max_time, time_to_min_rate)

    # Generate forecast up to effective max time
    t_forecast = np.arange(
        0, min(len(series) + int(effective_max_time), len(series) + 240), 1.0
    )
    forecast_rates = predict_arps(t_forecast, params)

    # Clip to minimum rate
    forecast_rates = np.maximum(forecast_rates, min_rate)

    # Create forecast series
    dates = pd.date_range(
        series.index[0], periods=len(forecast_rates), freq=series.index.freq or "MS"
    )
    forecast_series = pd.Series(forecast_rates, index=dates, name="forecast")

    params_dict = {
        "params": params,
        "max_time": max_time,
        "min_rate": min_rate,
        "time_to_min_rate": time_to_min_rate,
        "effective_max_time": effective_max_time,
    }

    return forecast_series, params_dict
