"""Fit quality diagnostics and scoring module.

This module provides comprehensive quality assessment for decline curve fits,
including grading, quality checks, metrics computation, and operations-grade
quality scoring.

Features:
- Fit grade system (A-F or numeric score)
- Quality checks (monotonicity, bounds, residuals, etc.)
- Comprehensive metrics (RMSE, MAE, MAPE, R², AIC, BIC)
- Operations-grade quality score
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fitting import FitResult
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DiagnosticsResult:
    """Result of fit quality diagnostics.

    Attributes:
        grade: Fit grade (A, B, C, D, F) or numeric score (0-100)
        numeric_score: Numeric score (0-100, higher is better)
        reason_codes: List of reason codes for issues
        warnings: List of warning messages
        metrics: Dictionary of computed metrics
        quality_flags: Dictionary of quality check results
    """

    grade: str
    numeric_score: float
    reason_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return string representation of diagnostics result."""
        return f"Grade: {self.grade} (Score: {self.numeric_score:.1f})"


def compute_rate_metrics(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute rate-based error metrics.

    Args:
        q_obs: Observed rates
        q_pred: Predicted rates

    Returns:
        Dictionary of metrics (RMSE, MAE, MAPE, R²)
    """
    residuals = q_obs - q_pred

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # MAE
    mae = np.mean(np.abs(residuals))

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = q_obs > 0
    if mask.any():
        mape = np.mean(np.abs(residuals[mask] / q_obs[mask])) * 100
    else:
        mape = np.inf

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((q_obs - q_obs.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r_squared": float(r_squared),
    }


def compute_cumulative_metrics(
    cum_obs: Optional[np.ndarray],
    cum_pred: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute cumulative-based error metrics.

    Args:
        cum_obs: Observed cumulative (optional)
        cum_pred: Predicted cumulative (optional)

    Returns:
        Dictionary of metrics
    """
    if cum_obs is None or cum_pred is None:
        return {}

    residuals = cum_obs - cum_pred

    # RMSE on cumulative
    rmse_cum = np.sqrt(np.mean(residuals**2))

    # MAE on cumulative
    mae_cum = np.mean(np.abs(residuals))

    # Relative error at end
    if len(cum_obs) > 0 and cum_obs[-1] > 0:
        rel_error_end = abs(residuals[-1] / cum_obs[-1]) * 100
    else:
        rel_error_end = np.inf

    return {
        "rmse_cumulative": float(rmse_cum),
        "mae_cumulative": float(mae_cum),
        "relative_error_end": float(rel_error_end),
    }


def compute_information_criteria(
    n: int,
    residuals: np.ndarray,
    n_params: int,
) -> Dict[str, float]:
    """Compute AIC and BIC information criteria.

    Args:
        n: Number of data points
        residuals: Residuals array
        n_params: Number of fitted parameters

    Returns:
        Dictionary with AIC and BIC
    """
    # Compute log-likelihood (assuming normal distribution)
    variance = np.var(residuals)
    if variance > 0:
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * variance) + 1)
    else:
        log_likelihood = 0.0

    # AIC = -2*log_likelihood + 2*k
    aic = -2 * log_likelihood + 2 * n_params

    # BIC = -2*log_likelihood + k*log(n)
    bic = -2 * log_likelihood + n_params * np.log(n)

    return {
        "aic": float(aic),
        "bic": float(bic),
    }


def check_quality(
    fit_result: FitResult,
    t: np.ndarray,
    q_obs: np.ndarray,
    cum_obs: Optional[np.ndarray] = None,
    tail_window: int = 3,
) -> Dict[str, bool]:
    """Perform quality checks on fit result.

    Args:
        fit_result: FitResult to check
        t: Time array
        q_obs: Observed rates
        cum_obs: Observed cumulative (optional)
        tail_window: Window size for tail checks

    Returns:
        Dictionary of quality flags (True = issue detected)
    """
    flags = {}

    if not fit_result.success:
        flags["fit_failed"] = True
        return flags

    # Compute predictions
    q_pred = fit_result.model.rate(t, fit_result.params)

    # Check 1: Non-monotone rate after fit start
    if fit_result.fit_start_idx < len(q_pred) - 1:
        tail_rates = q_pred[fit_result.fit_start_idx :]
        if len(tail_rates) > 1:
            increases = np.diff(tail_rates) > 0
            flags["non_monotone"] = increases.any()
        else:
            flags["non_monotone"] = False
    else:
        flags["non_monotone"] = False

    # Check 2: Rate spikes near tail
    if len(q_pred) >= tail_window:
        tail_rates = q_pred[-tail_window:]
        if len(tail_rates) > 1:
            # Check for large increases in tail
            tail_changes = np.diff(tail_rates)
            max_increase = tail_changes.max() if len(tail_changes) > 0 else 0
            mean_rate = np.mean(tail_rates)
            flags["tail_spike"] = (
                max_increase > 0.1 * mean_rate if mean_rate > 0 else False
            )
        else:
            flags["tail_spike"] = False
    else:
        flags["tail_spike"] = False

    # Check 3: Parameter hits on bounds
    constraints = fit_result.model.constraints()
    flags["parameter_on_bound"] = False
    for param, value in fit_result.params.items():
        if param in constraints:
            lower, upper = constraints[param]
            tolerance = (upper - lower) * 0.01  # 1% tolerance
            if abs(value - lower) < tolerance or abs(value - upper) < tolerance:
                flags["parameter_on_bound"] = True
                break

    # Check 4: Poor match on cumulative
    if cum_obs is not None:
        cum_pred = fit_result.model.cum(t, fit_result.params)
        if len(cum_obs) > 0 and cum_obs[-1] > 0:
            rel_error = abs((cum_obs[-1] - cum_pred[-1]) / cum_obs[-1])
            flags["poor_cumulative_match"] = rel_error > 0.1  # 10% error threshold
        else:
            flags["poor_cumulative_match"] = False
    else:
        flags["poor_cumulative_match"] = False

    # Check 5: Excessive curvature (high second derivative)
    if len(q_pred) > 2:
        # Compute second derivative
        dq = np.diff(q_pred)
        d2q = np.diff(dq)
        # Normalize by mean rate
        mean_rate = np.mean(q_pred)
        if mean_rate > 0:
            max_curvature = np.abs(d2q).max() / mean_rate
            flags["excessive_curvature"] = max_curvature > 0.5  # Threshold
        else:
            flags["excessive_curvature"] = False
    else:
        flags["excessive_curvature"] = False

    # Check 6: Large recent residuals
    if fit_result.residuals is not None and len(fit_result.residuals) >= tail_window:
        recent_residuals = fit_result.residuals[-tail_window:]
        mean_abs_residual = np.mean(np.abs(recent_residuals))
        mean_rate = np.mean(q_obs[-tail_window:])
        if mean_rate > 0:
            flags["large_recent_residuals"] = (
                mean_abs_residual > 0.2 * mean_rate
            )  # 20% threshold
        else:
            flags["large_recent_residuals"] = False
    else:
        flags["large_recent_residuals"] = False

    return flags


def compute_grade(
    metrics: Dict[str, float],
    quality_flags: Dict[str, bool],
    r_squared_thresholds: Tuple[float, float, float, float] = (0.95, 0.85, 0.70, 0.50),
    mape_thresholds: Tuple[float, float, float, float] = (10.0, 20.0, 30.0, 50.0),
) -> Tuple[str, float, List[str]]:
    """Compute fit grade from metrics and quality flags.

    Args:
        metrics: Dictionary of computed metrics
        quality_flags: Dictionary of quality check results
        r_squared_thresholds: R² thresholds for A, B, C, D (default: 0.95, 0.85, 0.70, 0.50)
        mape_thresholds: MAPE thresholds for A, B, C, D (default: 10%, 20%, 30%, 50%)

    Returns:
        Tuple of (grade, numeric_score, reason_codes)
    """
    reason_codes = []
    score_deductions = 0.0

    # Start with perfect score
    base_score = 100.0

    # Check R²
    r_squared = metrics.get("r_squared", 0.0)
    if r_squared >= r_squared_thresholds[0]:
        pass  # A grade
    elif r_squared >= r_squared_thresholds[1]:
        score_deductions += 10.0
        reason_codes.append("moderate_r_squared")
    elif r_squared >= r_squared_thresholds[2]:
        score_deductions += 20.0
        reason_codes.append("low_r_squared")
    elif r_squared >= r_squared_thresholds[3]:
        score_deductions += 30.0
        reason_codes.append("very_low_r_squared")
    else:
        score_deductions += 40.0
        reason_codes.append("extremely_low_r_squared")

    # Check MAPE
    mape = metrics.get("mape", np.inf)
    if mape <= mape_thresholds[0]:
        pass  # A grade
    elif mape <= mape_thresholds[1]:
        score_deductions += 5.0
        if "moderate_mape" not in reason_codes:
            reason_codes.append("moderate_mape")
    elif mape <= mape_thresholds[2]:
        score_deductions += 10.0
        if "high_mape" not in reason_codes:
            reason_codes.append("high_mape")
    elif mape <= mape_thresholds[3]:
        score_deductions += 15.0
        if "very_high_mape" not in reason_codes:
            reason_codes.append("very_high_mape")
    else:
        score_deductions += 20.0
        if "extremely_high_mape" not in reason_codes:
            reason_codes.append("extremely_high_mape")

    # Check quality flags
    critical_flags = ["fit_failed", "non_monotone", "poor_cumulative_match"]
    warning_flags = [
        "tail_spike",
        "parameter_on_bound",
        "excessive_curvature",
        "large_recent_residuals",
    ]

    for flag in critical_flags:
        if quality_flags.get(flag, False):
            score_deductions += 15.0
            reason_codes.append(flag)

    for flag in warning_flags:
        if quality_flags.get(flag, False):
            score_deductions += 5.0
            if flag not in reason_codes:
                reason_codes.append(flag)

    # Compute final score
    numeric_score = max(0.0, base_score - score_deductions)

    # Assign letter grade
    if numeric_score >= 90:
        grade = "A"
    elif numeric_score >= 80:
        grade = "B"
    elif numeric_score >= 70:
        grade = "C"
    elif numeric_score >= 60:
        grade = "D"
    else:
        grade = "F"

    return grade, numeric_score, reason_codes


def diagnose_fit(
    fit_result: FitResult,
    t: np.ndarray,
    q_obs: np.ndarray,
    cum_obs: Optional[np.ndarray] = None,
    include_cumulative_metrics: bool = True,
    include_ic: bool = True,
) -> DiagnosticsResult:
    """Perform comprehensive fit quality diagnostics.

    Main entry point for fit diagnostics. Computes all metrics, performs
    quality checks, and assigns a grade.

    Args:
        fit_result: FitResult to diagnose
        t: Time array
        q_obs: Observed rates
        cum_obs: Observed cumulative (optional)
        include_cumulative_metrics: Whether to compute cumulative metrics
        include_ic: Whether to compute information criteria (AIC, BIC)

    Returns:
        DiagnosticsResult with grade, metrics, and quality flags

    Example:
        >>> from decline_curve.fitting import CurveFitFitter, FitSpec
        >>> from decline_curve.models_arps import ExponentialArps
        >>>
        >>> # Fit model
        >>> model = ExponentialArps()
        >>> fitter = CurveFitFitter()
        >>> spec = FitSpec(model=model)
        >>> result = fitter.fit(t, q, spec)
        >>>
        >>> # Diagnose fit
        >>> diagnostics = diagnose_fit(result, t, q)
        >>> print(f"Grade: {diagnostics.grade}, Score: {diagnostics.numeric_score:.1f}")
    """
    metrics = {}
    warnings = []

    if not fit_result.success:
        return DiagnosticsResult(
            grade="F",
            numeric_score=0.0,
            reason_codes=["fit_failed"],
            warnings=[fit_result.message],
            metrics={},
            quality_flags={"fit_failed": True},
        )

    # Compute rate metrics
    q_pred = fit_result.model.rate(t, fit_result.params)
    rate_metrics = compute_rate_metrics(q_obs, q_pred)
    metrics.update(rate_metrics)

    # Compute cumulative metrics if requested
    if include_cumulative_metrics and cum_obs is not None:
        cum_pred = fit_result.model.cum(t, fit_result.params)
        cum_metrics = compute_cumulative_metrics(cum_obs, cum_pred)
        metrics.update(cum_metrics)

    # Compute information criteria if requested
    if include_ic and fit_result.residuals is not None:
        n_params = len(fit_result.params)
        ic_metrics = compute_information_criteria(
            len(t), fit_result.residuals, n_params
        )
        metrics.update(ic_metrics)

    # Perform quality checks
    quality_flags = check_quality(fit_result, t, q_obs, cum_obs)

    # Generate warnings from quality flags
    for flag, is_issue in quality_flags.items():
        if is_issue:
            if flag == "non_monotone":
                warnings.append("Model predicts non-monotonic decline")
            elif flag == "tail_spike":
                warnings.append("Rate spike detected near tail")
            elif flag == "parameter_on_bound":
                warnings.append("One or more parameters at bounds")
            elif flag == "poor_cumulative_match":
                warnings.append("Poor match on cumulative production")
            elif flag == "excessive_curvature":
                warnings.append("Excessive curvature in predicted rates")
            elif flag == "large_recent_residuals":
                warnings.append("Large residuals in recent data")

    # Compute grade
    grade, numeric_score, reason_codes = compute_grade(metrics, quality_flags)

    logger.info(
        f"Fit diagnostics complete: Grade {grade} (Score: {numeric_score:.1f})",
        extra={"grade": grade, "score": numeric_score, "n_issues": len(reason_codes)},
    )

    return DiagnosticsResult(
        grade=grade,
        numeric_score=numeric_score,
        reason_codes=reason_codes,
        warnings=warnings,
        metrics=metrics,
        quality_flags=quality_flags,
    )
