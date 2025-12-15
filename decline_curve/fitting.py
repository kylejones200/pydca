"""Fitting engine for decline curve analysis.

This module provides a unified interface for fitting decline curve models
to production data, with support for multiple backends and robust fitting
strategies.

Fitters:
- CurveFitFitter: Wrapper around scipy.optimize.curve_fit
- RobustLeastSquaresFitter: Uses scipy.optimize.least_squares with robust loss

Features:
- Parameter bounds and initial guesses
- Robust loss functions (Huber, soft L1)
- Weight policies (favor recent points)
- Fixed qi mode for ramp-up scenarios
- Post-fit validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models_base import Model

logger = get_logger(__name__)

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(default, **kwargs):
        """Field function for when pydantic is not available."""
        return default

    logger.warning(
        "pydantic not available. Install with: pip install pydantic. "
        "FitSpec validation will be limited."
    )


@dataclass
class FitResult:
    """Result of decline curve fitting.

    Attributes:
        params: Fitted parameters dictionary
        model: Model instance used for fitting
        success: Whether fitting succeeded
        message: Status message
        n_iterations: Number of iterations
        cost: Final cost value
        residuals: Residuals (observed - predicted)
        r_squared: R² score
        rmse: Root mean squared error
        mae: Mean absolute error
        fit_start_idx: Index where fitting started
        fit_end_idx: Index where fitting ended
        warnings: List of warning messages
    """

    params: Dict[str, float]
    model: Model
    success: bool
    message: str
    n_iterations: int = 0
    cost: float = 0.0
    residuals: Optional[np.ndarray] = None
    r_squared: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    fit_start_idx: int = 0
    fit_end_idx: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert FitResult to dictionary."""
        return {
            "params": self.params,
            "model_name": self.model.name,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "cost": self.cost,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mae": self.mae,
            "fit_start_idx": self.fit_start_idx,
            "fit_end_idx": self.fit_end_idx,
            "warnings": self.warnings,
        }


if PYDANTIC_AVAILABLE:

    class FitSpec(BaseModel):
        """Specification for decline curve fitting.

        Attributes:
            model: Model instance to fit
            param_bounds: Parameter bounds (overrides model defaults)
            initial_guess: Initial parameter guess (auto if None)
            loss: Loss function type
            weights: Weight array or policy
            min_points: Minimum points required for fitting
            fixed_params: Parameters to fix (not fit)
            ramp_aware_qi: If True, use max rate in first N months for qi guess
            ramp_window_months: Window for ramp-aware qi (default: 6)
        """

        model: Any  # Model instance
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        initial_guess: Optional[Dict[str, float]] = None
        loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "linear"
        weights: Optional[Union[np.ndarray, Literal["recent", "uniform"]]] = "uniform"
        min_points: int = 3
        fixed_params: Optional[Dict[str, float]] = None
        ramp_aware_qi: bool = False
        ramp_window_months: int = 6

        class Config:
            """Pydantic configuration."""

            arbitrary_types_allowed = True

else:
    # Fallback FitSpec without Pydantic validation
    @dataclass
    class FitSpec:
        """Specification for decline curve fitting (without Pydantic validation)."""

        model: Model
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        initial_guess: Optional[Dict[str, float]] = None
        loss: str = "linear"
        weights: Optional[Any] = "uniform"
        min_points: int = 3
        fixed_params: Optional[Dict[str, float]] = None
        ramp_aware_qi: bool = False
        ramp_window_months: int = 6


class Fitter(ABC):
    """Abstract base class for decline curve fitters."""

    @abstractmethod
    def fit(
        self,
        t: np.ndarray,
        q: np.ndarray,
        fit_spec: FitSpec,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> FitResult:
        """Fit decline curve model to data.

        Args:
            t: Time array (days)
            q: Rate array
            fit_spec: Fitting specification
            dates: Optional date index

        Returns:
            FitResult with fitted parameters and diagnostics
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return fitter name."""
        pass


def compute_initial_guess(
    t: np.ndarray,
    q: np.ndarray,
    model: Model,
    fit_spec: FitSpec,
) -> Dict[str, float]:
    """Compute initial parameter guess.

    Args:
        t: Time array
        q: Rate array
        model: Model instance
        fit_spec: Fitting specification

    Returns:
        Initial parameter guess dictionary
    """
    # Use provided guess if available
    if fit_spec.initial_guess:
        guess = fit_spec.initial_guess.copy()
    else:
        # Use model's initial guess method
        guess = model.initial_guess(t, q)

    # Apply ramp-aware qi if requested
    if fit_spec.ramp_aware_qi and "qi" in guess:
        # Find max rate in first N months
        if len(t) > 0:
            # Convert months to days
            ramp_window_days = fit_spec.ramp_window_months * 30.4375
            mask = t <= ramp_window_days
            if mask.any():
                max_rate = q[mask].max()
                guess["qi"] = max_rate
                logger.debug(
                    f"Ramp-aware qi: using max rate {max_rate:.2f} from first "
                    f"{fit_spec.ramp_window_months} months"
                )

    # Apply fixed parameters
    if fit_spec.fixed_params:
        for param, value in fit_spec.fixed_params.items():
            guess[param] = value

    return guess


def compute_weights(
    n: int,
    weights: Optional[Union[np.ndarray, Literal["recent", "uniform"]]],
) -> np.ndarray:
    """Compute weight array for fitting.

    Args:
        n: Number of data points
        weights: Weight specification

    Returns:
        Weight array
    """
    if isinstance(weights, np.ndarray):
        if len(weights) != n:
            raise ValueError(f"Weight array length {len(weights)} != data length {n}")
        return weights
    elif weights is None or weights == "uniform":
        return np.ones(n)
    elif weights == "recent":
        # Favor recent points with exponential decay
        w = np.exp(np.linspace(-2, 0, n))
        return w / w.sum() * n  # Normalize
    else:
        raise ValueError(f"Unknown weight specification: {weights}")


def compute_metrics(
    q_obs: np.ndarray,
    q_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute fit quality metrics.

    Args:
        q_obs: Observed rates
        q_pred: Predicted rates

    Returns:
        Dictionary of metrics
    """
    residuals = q_obs - q_pred

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((q_obs - q_obs.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # MAE
    mae = np.mean(np.abs(residuals))

    return {
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "residuals": residuals,
    }


def validate_fit_result(
    result: FitResult,
    t: np.ndarray,
    q: np.ndarray,
) -> List[str]:
    """Validate fit result and return warnings.

    Args:
        result: FitResult to validate
        t: Time array
        q: Rate array

    Returns:
        List of warning messages
    """
    warnings = []

    # Check parameter bounds
    constraints = result.model.constraints()
    for param, value in result.params.items():
        if param in constraints:
            lower, upper = constraints[param]
            if value < lower or value > upper:
                warnings.append(
                    f"Parameter {param}={value:.4f} "
                    f"outside bounds [{lower:.4f}, {upper:.4f}]"
                )

    # Check for non-negative rates
    if result.success:
        q_pred = result.model.rate(t, result.params)
        if (q_pred < 0).any():
            warnings.append("Model predicts negative rates")

    # Check monotonicity (should decline after fit start)
    if result.success and len(t) > 1:
        q_pred = result.model.rate(t, result.params)
        # Check if rate increases after fit start
        if result.fit_start_idx < len(q_pred) - 1:
            tail_rates = q_pred[result.fit_start_idx :]
            if len(tail_rates) > 1:
                increases = np.diff(tail_rates) > 0
                if increases.any():
                    warnings.append(
                        "Model predicts rate increases (non-monotonic decline)"
                    )

    return warnings


class CurveFitFitter(Fitter):
    """Fitter using scipy.optimize.curve_fit.

    Simple wrapper around curve_fit, suitable for standard least squares
    fitting with bounds support.
    """

    @property
    def name(self) -> str:
        """Return fitter name."""
        return "curve_fit"

    def fit(
        self,
        t: np.ndarray,
        q: np.ndarray,
        fit_spec: FitSpec,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> FitResult:
        """Fit using curve_fit."""
        from scipy.optimize import curve_fit

        # Check minimum points
        if len(t) < fit_spec.min_points:
            return FitResult(
                params={},
                model=fit_spec.model,
                success=False,
                message=f"Insufficient data points: {len(t)} < {fit_spec.min_points}",
                fit_start_idx=0,
                fit_end_idx=len(t),
            )

        # Get parameter names from model constraints
        constraints = fit_spec.model.constraints()
        param_names = list(constraints.keys())

        # Prepare bounds
        bounds_lower = [constraints[p][0] for p in param_names]
        bounds_upper = [constraints[p][1] for p in param_names]
        bounds = (bounds_lower, bounds_upper)

        # Override with fit_spec bounds if provided
        if fit_spec.param_bounds:
            for param, (lower, upper) in fit_spec.param_bounds.items():
                if param in param_names:
                    idx = param_names.index(param)
                    bounds[0][idx] = lower
                    bounds[1][idx] = upper

        # Compute initial guess
        initial_guess = compute_initial_guess(t, q, fit_spec.model, fit_spec)
        p0 = [
            initial_guess.get(p, (bounds[0][i] + bounds[1][i]) / 2)
            for i, p in enumerate(param_names)
        ]

        # Handle fixed parameters
        fixed_mask = np.array([p in (fit_spec.fixed_params or {}) for p in param_names])
        if fixed_mask.any():
            # Remove fixed parameters
            free_param_names = [
                p for i, p in enumerate(param_names) if not fixed_mask[i]
            ]
            free_p0 = [p0[i] for i in range(len(param_names)) if not fixed_mask[i]]
            free_bounds_lower = [
                bounds[0][i] for i in range(len(param_names)) if not fixed_mask[i]
            ]
            free_bounds_upper = [
                bounds[1][i] for i in range(len(param_names)) if not fixed_mask[i]
            ]
            free_bounds = (free_bounds_lower, free_bounds_upper)

            # Create wrapper function
            def model_func(t_arr, *free_params):
                all_params = {}
                free_idx = 0
                for i, p in enumerate(param_names):
                    if fixed_mask[i]:
                        all_params[p] = fit_spec.fixed_params[p]
                    else:
                        all_params[p] = free_params[free_idx]
                        free_idx += 1
                return fit_spec.model.rate(t_arr, all_params)

        else:
            free_param_names = param_names
            free_p0 = p0
            free_bounds = bounds

            def model_func(t_arr, *params):
                param_dict = dict(zip(param_names, params))
                return fit_spec.model.rate(t_arr, param_dict)

        # Compute weights
        weights = compute_weights(len(t), fit_spec.weights)

        # Fit
        try:
            popt, pcov = curve_fit(
                model_func,
                t,
                q,
                p0=free_p0,
                bounds=free_bounds,
                sigma=1.0 / weights if weights is not None else None,
                maxfev=10000,
            )

            # Reconstruct full parameter dict
            if fixed_mask.any():
                params = {}
                free_idx = 0
                for i, p in enumerate(param_names):
                    if fixed_mask[i]:
                        params[p] = fit_spec.fixed_params[p]
                    else:
                        params[p] = popt[free_idx]
                        free_idx += 1
            else:
                params = dict(zip(param_names, popt))

            # Validate parameters
            fit_spec.model.validate(params)

            # Compute predictions and metrics
            q_pred = fit_spec.model.rate(t, params)
            metrics = compute_metrics(q, q_pred)

            # Generate warnings
            result = FitResult(
                params=params,
                model=fit_spec.model,
                success=True,
                message="Fit converged",
                cost=metrics["rmse"],
                residuals=metrics["residuals"],
                r_squared=metrics["r_squared"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                fit_start_idx=0,
                fit_end_idx=len(t),
            )
            result.warnings = validate_fit_result(result, t, q)

            return result

        except Exception as e:
            logger.warning(f"Fitting failed: {e}")
            return FitResult(
                params={},
                model=fit_spec.model,
                success=False,
                message=f"Fitting failed: {str(e)}",
                fit_start_idx=0,
                fit_end_idx=len(t),
            )


class RobustLeastSquaresFitter(Fitter):
    """Fitter using scipy.optimize.least_squares with robust loss.

    Supports robust loss functions (Huber, soft L1) for handling outliers.
    """

    @property
    def name(self) -> str:
        return "robust_least_squares"

    def fit(
        self,
        t: np.ndarray,
        q: np.ndarray,
        fit_spec: FitSpec,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> FitResult:
        """Fit using least_squares with robust loss."""
        from scipy.optimize import least_squares

        # Check minimum points
        if len(t) < fit_spec.min_points:
            return FitResult(
                params={},
                model=fit_spec.model,
                success=False,
                message=f"Insufficient data points: {len(t)} < {fit_spec.min_points}",
                fit_start_idx=0,
                fit_end_idx=len(t),
            )

        # Get parameter names
        constraints = fit_spec.model.constraints()
        param_names = list(constraints.keys())

        # Prepare bounds
        bounds_lower = np.array([constraints[p][0] for p in param_names])
        bounds_upper = np.array([constraints[p][1] for p in param_names])

        # Override with fit_spec bounds
        if fit_spec.param_bounds:
            for param, (lower, upper) in fit_spec.param_bounds.items():
                if param in param_names:
                    idx = param_names.index(param)
                    bounds_lower[idx] = lower
                    bounds_upper[idx] = upper

        # Compute initial guess
        initial_guess = compute_initial_guess(t, q, fit_spec.model, fit_spec)
        p0 = np.array(
            [
                initial_guess.get(p, (bounds_lower[i] + bounds_upper[i]) / 2)
                for i, p in enumerate(param_names)
            ]
        )

        # Handle fixed parameters
        fixed_mask = np.array([p in (fit_spec.fixed_params or {}) for p in param_names])
        if fixed_mask.any():
            free_param_names = [
                p for i, p in enumerate(param_names) if not fixed_mask[i]
            ]
            free_p0 = p0[~fixed_mask]
            free_bounds_lower = bounds_lower[~fixed_mask]
            free_bounds_upper = bounds_upper[~fixed_mask]

            def residual_func(free_params):
                all_params = {}
                free_idx = 0
                for i, p in enumerate(param_names):
                    if fixed_mask[i]:
                        all_params[p] = fit_spec.fixed_params[p]
                    else:
                        all_params[p] = free_params[free_idx]
                        free_idx += 1
                q_pred = fit_spec.model.rate(t, all_params)
                residuals = (q - q_pred) * compute_weights(len(t), fit_spec.weights)
                return residuals

        else:
            free_param_names = param_names
            free_p0 = p0
            free_bounds_lower = bounds_lower
            free_bounds_upper = bounds_upper

            def residual_func(params):
                param_dict = dict(zip(param_names, params))
                q_pred = fit_spec.model.rate(t, param_dict)
                residuals = (q - q_pred) * compute_weights(len(t), fit_spec.weights)
                return residuals

        # Fit
        try:
            result_ls = least_squares(
                residual_func,
                free_p0,
                bounds=(free_bounds_lower, free_bounds_upper),
                loss=fit_spec.loss,
                max_nfev=10000,
            )

            # Reconstruct full parameter dict
            if fixed_mask.any():
                params = {}
                free_idx = 0
                for i, p in enumerate(param_names):
                    if fixed_mask[i]:
                        params[p] = fit_spec.fixed_params[p]
                    else:
                        params[p] = result_ls.x[free_idx]
                        free_idx += 1
            else:
                params = dict(zip(param_names, result_ls.x))

            # Validate parameters
            fit_spec.model.validate(params)

            # Compute predictions and metrics
            q_pred = fit_spec.model.rate(t, params)
            metrics = compute_metrics(q, q_pred)

            # Generate warnings
            result = FitResult(
                params=params,
                model=fit_spec.model,
                success=result_ls.success,
                message=result_ls.message,
                n_iterations=result_ls.nfev,
                cost=result_ls.cost,
                residuals=metrics["residuals"],
                r_squared=metrics["r_squared"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                fit_start_idx=0,
                fit_end_idx=len(t),
            )
            result.warnings = validate_fit_result(result, t, q)

            return result

        except Exception as e:
            logger.warning(f"Fitting failed: {e}")
            return FitResult(
                params={},
                model=fit_spec.model,
                success=False,
                message=f"Fitting failed: {str(e)}",
                fit_start_idx=0,
                fit_end_idx=len(t),
            )
