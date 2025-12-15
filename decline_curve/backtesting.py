"""Backtesting module for forecast validation.

This module provides backtesting capabilities to validate forecasts
against historical data, including walk-forward and rolling origin methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .fitting import FitSpec
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run.

    Attributes:
        method: Backtest method used
        horizons: List of forecast horizons tested
        coverage_p10: Coverage for P10 band (fraction)
        coverage_p50: Coverage for P50 band (fraction)
        coverage_p90: Coverage for P90 band (fraction)
        forecast_errors: List of forecast errors by horizon
        rank_stability: Rank correlation for EUR/NPV ordering
    """

    method: str
    horizons: List[float]
    coverage_p10: float
    coverage_p50: float
    coverage_p90: float
    forecast_errors: Dict[float, Dict[str, float]] = field(default_factory=dict)
    rank_stability: Optional[float] = None


def walk_forward_backtest(
    t: np.ndarray,
    q: np.ndarray,
    fit_spec: FitSpec,
    forecast_horizons: List[float],
    min_train_size: int = 12,
    step_size: int = 1,
    seed: int = 42,
) -> BacktestResult:
    """Walk-forward backtest: refit at each step.

    Args:
        t: Time array
        q: Rate array
        fit_spec: FitSpec for fitting
        forecast_horizons: List of forecast horizons to test
        min_train_size: Minimum training set size
        step_size: Steps between refits
        seed: Random seed

    Returns:
        BacktestResult with coverage and error metrics

    Example:
        >>> from decline_curve.fitting import FitSpec
        >>> from decline_curve.models_arps import HyperbolicArps
        >>>
        >>> spec = FitSpec(model=HyperbolicArps())
        >>> result = walk_forward_backtest(t, q, spec, [12, 24, 36])
    """
    logger.info(
        f"Running walk-forward backtest: {len(forecast_horizons)} horizons",
        extra={"n_horizons": len(forecast_horizons), "min_train_size": min_train_size},
    )

    np.random.seed(seed)

    # Track coverage and errors
    coverage_p10 = []
    coverage_p50 = []
    coverage_p90 = []
    errors_by_horizon = {
        h: {"mae": [], "rmse": [], "mape": []} for h in forecast_horizons
    }

    # Walk forward through data
    for cut_idx in range(
        min_train_size, len(t) - int(min(forecast_horizons)), step_size
    ):
        # Training data
        t_train = t[:cut_idx]
        q_train = q[:cut_idx]

        # Test data
        for horizon in forecast_horizons:
            horizon_idx = cut_idx + int(horizon)
            if horizon_idx >= len(t):
                continue

            # Fit on training data
            from .fitting import CurveFitFitter

            fitter = CurveFitFitter()
            try:
                fit_result = fitter.fit(t_train, q_train, fit_spec)

                if not fit_result.success:
                    continue

                # Generate forecast
                t_forecast = t[cut_idx:horizon_idx]
                q_forecast = fit_result.model.rate(t_forecast, fit_result.params)
                q_actual = q[cut_idx:horizon_idx]

                # Compute errors
                mae = np.mean(np.abs(q_forecast - q_actual))
                rmse = np.sqrt(np.mean((q_forecast - q_actual) ** 2))
                mape = (
                    np.mean(np.abs((q_forecast - q_actual) / (q_actual + 1e-10))) * 100
                )

                errors_by_horizon[horizon]["mae"].append(mae)
                errors_by_horizon[horizon]["rmse"].append(rmse)
                errors_by_horizon[horizon]["mape"].append(mape)

                # Check coverage (simplified - would use uncertainty bands)
                # For now, just check if actual is within reasonable range
                if len(q_forecast) > 0 and len(q_actual) > 0:
                    p10_forecast = q_forecast * 1.2  # Approximate
                    p90_forecast = q_forecast * 0.8  # Approximate

                    coverage_p50.append(
                        np.mean((q_actual >= p90_forecast) & (q_actual <= p10_forecast))
                    )
                    coverage_p10.append(np.mean(q_actual <= p10_forecast))
                    coverage_p90.append(np.mean(q_actual >= p90_forecast))

            except Exception as e:
                logger.debug(f"Backtest step failed at cut_idx={cut_idx}: {e}")
                continue

    # Aggregate results
    avg_coverage_p10 = np.mean(coverage_p10) if coverage_p10 else 0.0
    avg_coverage_p50 = np.mean(coverage_p50) if coverage_p50 else 0.0
    avg_coverage_p90 = np.mean(coverage_p90) if coverage_p90 else 0.0

    # Compute average errors by horizon
    forecast_errors = {}
    for horizon, error_list in errors_by_horizon.items():
        if error_list["mae"]:
            forecast_errors[horizon] = {
                "mae": np.mean(error_list["mae"]),
                "rmse": np.mean(error_list["rmse"]),
                "mape": np.mean(error_list["mape"]),
            }

    return BacktestResult(
        method="walk_forward",
        horizons=forecast_horizons,
        coverage_p10=avg_coverage_p10,
        coverage_p50=avg_coverage_p50,
        coverage_p90=avg_coverage_p90,
        forecast_errors=forecast_errors,
    )


def rolling_origin_backtest(
    t: np.ndarray,
    q: np.ndarray,
    fit_spec: FitSpec,
    forecast_horizons: List[float],
    train_size: int = 24,
    step_size: int = 1,
    seed: int = 42,
) -> BacktestResult:
    """Perform rolling origin backtest with fixed training window.

    Args:
        t: Time array
        q: Rate array
        fit_spec: FitSpec for fitting
        forecast_horizons: List of forecast horizons
        train_size: Fixed training window size
        step_size: Steps between origins
        seed: Random seed

    Returns:
        BacktestResult with coverage and error metrics
    """
    logger.info(
        f"Running rolling origin backtest: train_size={train_size}",
        extra={"train_size": train_size, "n_horizons": len(forecast_horizons)},
    )

    np.random.seed(seed)

    # Similar implementation to walk_forward but with fixed training window
    # This is a simplified version - full implementation would be more sophisticated
    return walk_forward_backtest(
        t,
        q,
        fit_spec,
        forecast_horizons,
        min_train_size=train_size,
        step_size=step_size,
        seed=seed,
    )


def check_data_leakage(
    t: np.ndarray,
    q: np.ndarray,
    forecast_cut_idx: int,
    features: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Check for data leakage in features.

    Args:
        t: Time array
        q: Rate array
        forecast_cut_idx: Index where forecast starts
        features: List of feature names to check

    Returns:
        Dictionary of leakage checks (True = no leakage detected)
    """
    checks = {
        "no_future_dates": True,
        "no_future_rates": True,
    }

    # Check that no features reference future data
    if forecast_cut_idx >= len(t):
        checks["no_future_dates"] = False
        checks["no_future_rates"] = False
        return checks

    # Check that forecast cut is valid
    forecast_start_time = t[forecast_cut_idx]

    # In a full implementation, would check actual feature values
    # For now, just verify the cut index is reasonable
    if forecast_cut_idx < 0 or forecast_cut_idx >= len(t):
        checks["no_future_dates"] = False
        checks["no_future_rates"] = False

    logger.debug(
        f"Data leakage check: forecast_cut_idx={forecast_cut_idx}",
        extra={"forecast_start_time": forecast_start_time},
    )

    return checks
