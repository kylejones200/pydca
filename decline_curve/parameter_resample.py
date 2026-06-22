"""Fast parameter resampling for Arps with approximate posteriors.

This module implements fast parameter resampling using simple approximate
posteriors (normal or lognormal) around the point estimate with scale tied
to residual error. This provides a fast alternative to full Bayesian inference.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models import ArpsParams, fit_arps, predict_arps
from .uncertainty_core import ForecastDraws, ParameterDistribution

logger = get_logger(__name__)


def fast_arps_resample(
    series: pd.Series,
    kind: str = "hyperbolic",
    n_draws: int = 1000,
    seed: Optional[int] = None,
    method: str = "residual_based",
    horizon: int = 12,
) -> ForecastDraws:
    """
    Fast parameter resampling for Arps models with approximate posteriors.

    Uses simple approximate posteriors (normal or lognormal) around the point
    estimate with scale tied to residual error. This is much faster than
    full Bayesian inference but provides reasonable uncertainty estimates.

    Args:
        series: Historical production time series
        kind: Arps decline type ('exponential', 'harmonic', 'hyperbolic')
        n_draws: Number of parameter samples
        seed: Random seed for reproducibility
        method: Resampling method:
            - 'residual_based': Scale uncertainty from residual error
            - 'fixed_scale': Use fixed scale factors (faster, less accurate)
        horizon: Forecast horizon

    Returns:
        ForecastDraws with uncertainty quantification

    Example:
        >>> import pandas as pd
        >>> from decline_curve.parameter_resample import fast_arps_resample
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> draws = fast_arps_resample(production, kind='hyperbolic', n_draws=1000)
        >>> print(f"P50 forecast: {draws.p50.iloc[-1]:.2f}")
        >>> print(f"P10-P90 range: {draws.p10.iloc[-1]:.2f} - {draws.p90.iloc[-1]:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Fit point estimate
    t = np.arange(len(series))
    q = series.values
    params = fit_arps(t, q, kind=kind)

    # Calculate residual error
    q_pred = predict_arps(t, params)
    residuals = q - q_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    # Estimate parameter uncertainty from residual error
    if method == "residual_based":
        # Scale uncertainty based on residual error
        # Higher residual error -> higher parameter uncertainty
        relative_error = rmse / np.mean(q) if np.mean(q) > 0 else 0.1

        # Parameter uncertainty scales with relative error
        qi_std = params.qi * relative_error * 0.5  # Conservative scaling
        di_std = params.di * relative_error * 0.5
        b_std = params.b * relative_error * 0.3 if kind == "hyperbolic" else 0.0

    elif method == "fixed_scale":
        # Fixed scale factors (faster, less accurate)
        qi_std = params.qi * 0.15  # 15% uncertainty
        di_std = params.di * 0.20  # 20% uncertainty
        b_std = params.b * 0.10 if kind == "hyperbolic" else 0.0  # 10% uncertainty
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sample parameters from approximate posteriors
    # Use lognormal for qi and di (must be positive), normal for b
    qi_samples = np.random.lognormal(
        np.log(max(params.qi, 1e-6)), qi_std / params.qi, n_draws
    )
    di_samples = np.random.lognormal(
        np.log(max(params.di, 1e-6)), di_std / params.di, n_draws
    )

    if kind == "hyperbolic":
        # b-factor: use truncated normal (0 to 2)
        b_samples = np.random.normal(params.b, b_std, n_draws)
        b_samples = np.clip(b_samples, 0.0, 2.0)
    elif kind == "exponential":
        b_samples = np.zeros(n_draws)
    else:  # harmonic
        b_samples = np.ones(n_draws)

    # Generate forecast for each parameter sample
    n_periods = len(series) + horizon
    draws = np.zeros((n_draws, n_periods))

    t_full = np.arange(n_periods)

    for i in range(n_draws):
        try:
            sample_params = ArpsParams(
                qi=qi_samples[i], di=di_samples[i], b=b_samples[i]
            )
            forecast = predict_arps(t_full, sample_params)
            draws[i] = forecast
        except Exception as e:
            logger.warning(f"Failed to generate forecast for sample {i}: {e}")
            # Use point estimate as fallback
            draws[i] = predict_arps(t_full, params)

    # Create date index
    dates = pd.date_range(
        series.index[0], periods=n_periods, freq=series.index.freq or "MS"
    )

    return ForecastDraws(
        draws=draws,
        dates=dates,
        metadata={
            "method": method,
            "kind": kind,
            "n_draws": n_draws,
            "seed": seed,
            "point_estimate": {
                "qi": params.qi,
                "di": params.di,
                "b": params.b,
            },
            "rmse": rmse,
            "mae": mae,
        },
    )


def _duong_resample(
    series: pd.Series,
    n_draws: int = 1000,
    seed: Optional[int] = None,
    horizon: int = 12,
) -> ForecastDraws:
    """Fast parameter resampling for the Duong transient-flow model.

    Samples q1 and a from lognormal distributions; m from a truncated normal
    clipped to the valid Duong range [0.3, 3.0].  Scale is tied to relative
    RMSE from the point-estimate fit, following the same convention as
    :func:`fast_arps_resample`.

    Args:
        series: Historical production time series.
        n_draws: Number of parameter samples.
        seed: Random seed for reproducibility.
        horizon: Forecast horizon in months beyond history.

    Returns:
        ForecastDraws with P10/P50/P90 curves.
    """
    from .decline_variants import DuongParams, fit_duong, predict_duong

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(len(series))
    q = series.values
    params = fit_duong(t, q)

    q_pred = predict_duong(t, params)
    residuals = q - q_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    relative_error = rmse / np.mean(q) if np.mean(q) > 0 else 0.1

    q1_samples = np.random.lognormal(
        np.log(max(params.q1, 1e-6)), relative_error * 0.5, n_draws
    )
    a_samples = np.random.lognormal(
        np.log(max(params.a, 1e-6)), relative_error * 0.4, n_draws
    )
    m_samples = np.clip(
        np.random.normal(params.m, params.m * relative_error * 0.2, n_draws), 0.3, 3.0
    )

    n_periods = len(series) + horizon
    t_full = np.arange(n_periods)
    draws = np.zeros((n_draws, n_periods))

    for i in range(n_draws):
        try:
            sample = DuongParams(q1=q1_samples[i], a=a_samples[i], m=m_samples[i])
            draws[i] = predict_duong(t_full, sample)
        except Exception as e:
            logger.warning(f"Duong sample {i} failed: {e}")
            draws[i] = predict_duong(t_full, params)

    dates = pd.date_range(
        series.index[0], periods=n_periods, freq=series.index.freq or "MS"
    )

    return ForecastDraws(
        draws=draws,
        dates=dates,
        metadata={
            "method": "residual_based",
            "kind": "duong",
            "n_draws": n_draws,
            "seed": seed,
            "point_estimate": {"q1": params.q1, "a": params.a, "m": params.m},
            "rmse": rmse,
        },
    )


def _ple_resample(
    series: pd.Series,
    n_draws: int = 1000,
    seed: Optional[int] = None,
    horizon: int = 12,
) -> ForecastDraws:
    """Fast parameter resampling for the PLE (Power-Law Exponential) model.

    All four parameters (qi, D_inf, D1, n) are sampled from approximate
    lognormal posteriors except n, which uses a truncated normal clipped to
    [0.01, 1.0].  Scale is tied to relative RMSE.

    Args:
        series: Historical production time series.
        n_draws: Number of parameter samples.
        seed: Random seed for reproducibility.
        horizon: Forecast horizon in months beyond history.

    Returns:
        ForecastDraws with P10/P50/P90 curves.
    """
    from .decline_variants import PLEParams, fit_ple, predict_ple

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(len(series))
    q = series.values
    params = fit_ple(t, q)

    q_pred = predict_ple(t, params)
    residuals = q - q_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    relative_error = rmse / np.mean(q) if np.mean(q) > 0 else 0.1

    qi_samples = np.random.lognormal(
        np.log(max(params.qi, 1e-6)), relative_error * 0.5, n_draws
    )
    # D_inf is small and highly uncertain — give it more spread
    D_inf_samples = np.random.lognormal(
        np.log(max(params.D_inf, 1e-9)), relative_error * 0.6, n_draws
    )
    D1_samples = np.random.lognormal(
        np.log(max(params.D1, 1e-9)), relative_error * 0.4, n_draws
    )
    n_samples = np.clip(
        np.random.normal(params.n, params.n * relative_error * 0.2, n_draws), 0.01, 1.0
    )

    n_periods = len(series) + horizon
    t_full = np.arange(n_periods)
    draws = np.zeros((n_draws, n_periods))

    for i in range(n_draws):
        try:
            sample = PLEParams(
                qi=qi_samples[i], D_inf=D_inf_samples[i], D1=D1_samples[i], n=n_samples[i]
            )
            draws[i] = predict_ple(t_full, sample)
        except Exception as e:
            logger.warning(f"PLE sample {i} failed: {e}")
            draws[i] = predict_ple(t_full, params)

    dates = pd.date_range(
        series.index[0], periods=n_periods, freq=series.index.freq or "MS"
    )

    return ForecastDraws(
        draws=draws,
        dates=dates,
        metadata={
            "method": "residual_based",
            "kind": "ple",
            "n_draws": n_draws,
            "seed": seed,
            "point_estimate": {
                "qi": params.qi, "D_inf": params.D_inf, "D1": params.D1, "n": params.n
            },
            "rmse": rmse,
        },
    )


def approximate_posterior(
    params: ArpsParams,
    residuals: np.ndarray,
    kind: str = "hyperbolic",
) -> ParameterDistribution:
    """
    Create approximate posterior distribution from point estimate and residuals.

    This is a fast approximation that assumes:
    - qi and di follow lognormal distributions (must be positive)
    - b follows truncated normal distribution (0 to 2)
    - Scale is tied to residual error

    Args:
        params: Point estimate parameters
        residuals: Residual errors from fit
        kind: Decline type

    Returns:
        ParameterDistribution object
    """
    # Calculate residual error
    rmse = np.sqrt(np.mean(residuals**2))
    relative_error = rmse / np.mean(np.abs(residuals) + 1e-6)

    # Estimate parameter uncertainty
    qi_std = params.qi * relative_error * 0.5
    di_std = params.di * relative_error * 0.5
    b_std = params.b * relative_error * 0.3 if kind == "hyperbolic" else 0.0

    # Create distributions
    qi_dist = {
        "type": "lognormal",
        "mean": params.qi,
        "std": qi_std / params.qi,  # Coefficient of variation
    }

    di_dist = {
        "type": "lognormal",
        "mean": params.di,
        "std": di_std / params.di,
    }

    if kind == "hyperbolic":
        b_dist = {
            "type": "normal",
            "mean": params.b,
            "std": b_std,
            "min": 0.0,
            "max": 2.0,
        }
    elif kind == "exponential":
        b_dist = {"type": "normal", "mean": 0.0, "std": 0.0}
    else:  # harmonic
        b_dist = {"type": "normal", "mean": 1.0, "std": 0.0}

    return ParameterDistribution(
        qi_dist=qi_dist, di_dist=di_dist, b_dist=b_dist, correlation=None
    )
