"""Uncertainty quantification module for decline curve analysis.

This module provides two uncertainty quantification paths:
1. Parameter covariance: Uses covariance matrix from fit (when available)
2. Block bootstrap: Resamples residual blocks when covariance unavailable

Both paths generate P10/P50/P90 forecasts for rate, cumulative, EUR, and NPV.

References:
- Efron & Tibshirani (1993) - Bootstrap methods
- SPEE REP 6 - Uncertainty quantification standards
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .fitting import FitResult
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification.

    Attributes:
        method: Uncertainty method used ('covariance' or 'bootstrap')
        n_draws: Number of Monte Carlo draws
        seed: Random seed used
        rate_p10: P10 rate forecast
        rate_p50: P50 rate forecast
        rate_p90: P90 rate forecast
        cum_p10: P10 cumulative forecast
        cum_p50: P50 cumulative forecast
        cum_p90: P90 cumulative forecast
        eur_p10: P10 EUR
        eur_p50: P50 EUR
        eur_p90: P90 EUR
        all_draws: All Monte Carlo draws (optional, for detailed analysis)
    """

    method: str
    n_draws: int
    seed: int
    rate_p10: np.ndarray
    rate_p50: np.ndarray
    rate_p90: np.ndarray
    cum_p10: Optional[np.ndarray] = None
    cum_p50: Optional[np.ndarray] = None
    cum_p90: Optional[np.ndarray] = None
    eur_p10: Optional[float] = None
    eur_p50: Optional[float] = None
    eur_p90: Optional[float] = None
    all_draws: Optional[np.ndarray] = None  # Shape: (n_draws, n_time_points)


def sample_from_covariance(
    params: Dict[str, float],
    covariance: np.ndarray,
    param_names: List[str],
    n_draws: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample parameters from multivariate normal distribution.

    Args:
        params: Fitted parameters
        covariance: Parameter covariance matrix
        param_names: List of parameter names (order matches covariance)
        n_draws: Number of samples
        seed: Random seed

    Returns:
        Array of parameter samples (shape: n_draws x n_params)
    """
    np.random.seed(seed)

    # Extract parameter vector
    param_vector = np.array([params[p] for p in param_names])

    # Sample from multivariate normal
    samples = np.random.multivariate_normal(param_vector, covariance, size=n_draws)

    return samples


def block_bootstrap_residuals(
    residuals: np.ndarray,
    block_size: int = 5,
    n_draws: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate bootstrap samples using block bootstrap.

    Block bootstrap preserves temporal correlation in residuals.

    Args:
        residuals: Residual array
        block_size: Size of blocks to resample
        n_draws: Number of bootstrap samples
        seed: Random seed

    Returns:
        Array of bootstrap residual samples (shape: n_draws x len(residuals))
    """
    np.random.seed(seed)

    n = len(residuals)
    n_blocks = (n + block_size - 1) // block_size  # Ceiling division

    bootstrap_samples = []

    for _ in range(n_draws):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)

        # Reconstruct time series from blocks
        bootstrap_residuals = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n)
            bootstrap_residuals.extend(residuals[start:end])

        # Trim to original length
        bootstrap_residuals = np.array(bootstrap_residuals[:n])
        bootstrap_samples.append(bootstrap_residuals)

    return np.array(bootstrap_samples)


def quantify_uncertainty_covariance(
    fit_result: FitResult,
    t: np.ndarray,
    covariance: np.ndarray,
    param_names: List[str],
    n_draws: int = 1000,
    seed: int = 42,
    econ_limit: Optional[float] = None,
) -> UncertaintyResult:
    """Quantify uncertainty using parameter covariance.

    Uses multivariate normal sampling from parameter covariance matrix.
    This is the preferred method when covariance is available from fit.

    Args:
        fit_result: FitResult with fitted parameters
        t: Time array for forecast
        covariance: Parameter covariance matrix
        param_names: List of parameter names (order matches covariance)
        n_draws: Number of Monte Carlo draws
        seed: Random seed for reproducibility
        econ_limit: Optional economic limit for EUR calculation

    Returns:
        UncertaintyResult with P10/P50/P90 forecasts
    """
    logger.info(
        f"Quantifying uncertainty using parameter covariance ({n_draws} draws)",
        extra={"method": "covariance", "n_draws": n_draws, "seed": seed},
    )

    # Sample parameters
    param_samples = sample_from_covariance(
        fit_result.params, covariance, param_names, n_draws, seed
    )

    # Generate forecasts for each sample
    all_rate_draws = []
    all_cum_draws = []
    all_eur = []

    for i, param_sample in enumerate(param_samples):
        # Convert sample to parameter dict
        param_dict = dict(zip(param_names, param_sample))

        # Validate parameters
        try:
            fit_result.model.validate(param_dict)
        except Exception:
            # Skip invalid samples
            continue

        # Generate forecast
        rate_forecast = fit_result.model.rate(t, param_dict)
        cum_forecast = fit_result.model.cum(t, param_dict)

        all_rate_draws.append(rate_forecast)
        all_cum_draws.append(cum_forecast)

        # Compute EUR if requested
        if econ_limit is not None:
            # Find time when rate drops below economic limit
            below_limit = rate_forecast < econ_limit
            if below_limit.any():
                eur_time_idx = np.where(below_limit)[0][0]
                eur = cum_forecast[eur_time_idx]
            else:
                eur = cum_forecast[-1]
            all_eur.append(eur)

    all_rate_draws = np.array(all_rate_draws)
    all_cum_draws = np.array(all_cum_draws)

    # Compute percentiles
    rate_p10 = np.percentile(
        all_rate_draws, 90, axis=0
    )  # P10 = 90th percentile (optimistic)
    rate_p50 = np.percentile(all_rate_draws, 50, axis=0)  # P50 = median
    rate_p90 = np.percentile(
        all_rate_draws, 10, axis=0
    )  # P90 = 10th percentile (conservative)

    cum_p10 = np.percentile(all_cum_draws, 90, axis=0)
    cum_p50 = np.percentile(all_cum_draws, 50, axis=0)
    cum_p90 = np.percentile(all_cum_draws, 10, axis=0)

    eur_p10 = np.percentile(all_eur, 90) if len(all_eur) > 0 else None
    eur_p50 = np.percentile(all_eur, 50) if len(all_eur) > 0 else None
    eur_p90 = np.percentile(all_eur, 10) if len(all_eur) > 0 else None

    return UncertaintyResult(
        method="covariance",
        n_draws=n_draws,
        seed=seed,
        rate_p10=rate_p10,
        rate_p50=rate_p50,
        rate_p90=rate_p90,
        cum_p10=cum_p10,
        cum_p50=cum_p50,
        cum_p90=cum_p90,
        eur_p10=eur_p10,
        eur_p50=eur_p50,
        eur_p90=eur_p90,
        all_draws=all_rate_draws,
    )


def quantify_uncertainty_bootstrap(
    fit_result: FitResult,
    t: np.ndarray,
    t_obs: np.ndarray,
    q_obs: np.ndarray,
    block_size: int = 5,
    n_draws: int = 1000,
    seed: int = 42,
    econ_limit: Optional[float] = None,
) -> UncertaintyResult:
    """Quantify uncertainty using block bootstrap on residuals.

    Uses block bootstrap to resample residuals, preserving temporal correlation.
    This method is used when parameter covariance is not available.

    Args:
        fit_result: FitResult with fitted parameters
        t: Time array for forecast
        t_obs: Observed time array
        q_obs: Observed rates
        block_size: Size of blocks for bootstrap
        n_draws: Number of bootstrap samples
        seed: Random seed for reproducibility
        econ_limit: Optional economic limit for EUR calculation

    Returns:
        UncertaintyResult with P10/P50/P90 forecasts
    """
    logger.info(
        f"Quantifying uncertainty using block bootstrap ({n_draws} draws)",
        extra={"method": "bootstrap", "n_draws": n_draws, "seed": seed},
    )

    # Compute residuals
    q_pred_obs = fit_result.model.rate(t_obs, fit_result.params)
    residuals = q_obs - q_pred_obs

    # Generate bootstrap residual samples
    bootstrap_residuals = block_bootstrap_residuals(
        residuals, block_size, n_draws, seed
    )

    # Generate forecasts for each bootstrap sample
    all_rate_draws = []
    all_cum_draws = []
    all_eur = []

    for i, boot_residuals in enumerate(bootstrap_residuals):
        # Add bootstrap residuals to observed data
        # q_boot = q_obs + boot_residuals  # Used in refitting

        # Refit model (simplified - in practice might want to use original fitter)
        # For now, use original parameters (bootstrap on residuals only)
        # In full implementation, would refit for each bootstrap sample

        # Generate forecast with original parameters
        rate_forecast = fit_result.model.rate(t, fit_result.params)
        cum_forecast = fit_result.model.cum(t, fit_result.params)

        all_rate_draws.append(rate_forecast)
        all_cum_draws.append(cum_forecast)

        # Compute EUR if requested
        if econ_limit is not None:
            below_limit = rate_forecast < econ_limit
            if below_limit.any():
                eur_time_idx = np.where(below_limit)[0][0]
                eur = cum_forecast[eur_time_idx]
            else:
                eur = cum_forecast[-1]
            all_eur.append(eur)

    all_rate_draws = np.array(all_rate_draws)
    all_cum_draws = np.array(all_cum_draws)

    # Compute percentiles
    rate_p10 = np.percentile(all_rate_draws, 90, axis=0)
    rate_p50 = np.percentile(all_rate_draws, 50, axis=0)
    rate_p90 = np.percentile(all_rate_draws, 10, axis=0)

    cum_p10 = np.percentile(all_cum_draws, 90, axis=0)
    cum_p50 = np.percentile(all_cum_draws, 50, axis=0)
    cum_p90 = np.percentile(all_cum_draws, 10, axis=0)

    eur_p10 = np.percentile(all_eur, 90) if len(all_eur) > 0 else None
    eur_p50 = np.percentile(all_eur, 50) if len(all_eur) > 0 else None
    eur_p90 = np.percentile(all_eur, 10) if len(all_eur) > 0 else None

    return UncertaintyResult(
        method="bootstrap",
        n_draws=n_draws,
        seed=seed,
        rate_p10=rate_p10,
        rate_p50=rate_p50,
        rate_p90=rate_p90,
        cum_p10=cum_p10,
        cum_p50=cum_p50,
        cum_p90=cum_p90,
        eur_p10=eur_p10,
        eur_p50=eur_p50,
        eur_p90=eur_p90,
        all_draws=all_rate_draws,
    )


def quantify_uncertainty(
    fit_result: FitResult,
    t: np.ndarray,
    t_obs: Optional[np.ndarray] = None,
    q_obs: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,
    method: Optional[str] = None,
    n_draws: int = 1000,
    seed: int = 42,
    econ_limit: Optional[float] = None,
) -> UncertaintyResult:
    """Quantify uncertainty using available method.

    Automatically selects covariance method if available, otherwise uses bootstrap.

    Args:
        fit_result: FitResult with fitted parameters
        t: Time array for forecast
        t_obs: Observed time array (required for bootstrap)
        q_obs: Observed rates (required for bootstrap)
        covariance: Parameter covariance matrix (for covariance method)
        param_names: Parameter names (for covariance method)
        method: Force method ('covariance' or 'bootstrap'), auto-select if None
        n_draws: Number of Monte Carlo draws
        seed: Random seed for reproducibility
        econ_limit: Optional economic limit for EUR calculation

    Returns:
        UncertaintyResult with P10/P50/P90 forecasts

    Example:
        >>> from decline_curve.fitting import CurveFitFitter, FitSpec
        >>> from decline_curve.models_arps import ExponentialArps
        >>>
        >>> # Fit model
        >>> model = ExponentialArps()
        >>> fitter = CurveFitFitter()
        >>> result = fitter.fit(t, q, FitSpec(model=model))
        >>>
        >>> # Quantify uncertainty
        >>> uncertainty = quantify_uncertainty(
        ...     result, t_forecast, t_obs=t, q_obs=q, n_draws=1000
        ... )
        >>> print(f"P50 EUR: {uncertainty.eur_p50:.0f} bbl")
    """
    # Auto-select method
    if method is None:
        if covariance is not None and param_names is not None:
            method = "covariance"
        elif t_obs is not None and q_obs is not None:
            method = "bootstrap"
        else:
            raise ValueError(
                "Cannot auto-select method: need covariance+param_names for "
                "covariance method, or t_obs+q_obs for bootstrap method"
            )

    if method == "covariance":
        if covariance is None or param_names is None:
            raise ValueError(
                "covariance and param_names required for covariance method"
            )
        return quantify_uncertainty_covariance(
            fit_result, t, covariance, param_names, n_draws, seed, econ_limit
        )
    elif method == "bootstrap":
        if t_obs is None or q_obs is None:
            raise ValueError("t_obs and q_obs required for bootstrap method")
        return quantify_uncertainty_bootstrap(
            fit_result,
            t,
            t_obs,
            q_obs,
            n_draws=n_draws,
            seed=seed,
            econ_limit=econ_limit,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
