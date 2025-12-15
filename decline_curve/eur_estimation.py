"""EUR (Estimated Ultimate Recovery) estimation from production data.

This module provides functions for calculating EUR from production decline curves,
including batch processing for multiple wells.
"""

from typing import Dict, Optional

import pandas as pd

from .fitting import CurveFitFitter, FitSpec
from .logging_config import get_logger
from .models_arps import HyperbolicArps
from .reserves import forecast_and_reserves

logger = get_logger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def calculate_eur_from_production(
    months: pd.Series,
    production: pd.Series,
    model_type: str = "hyperbolic",
    t_max: float = 360.0,
    econ_limit: float = 10.0,
) -> Optional[Dict[str, float]]:
    """Calculate EUR from production data by fitting decline curve.

    Args:
        months: Months since production start
        production: Production rates
        model_type: Type of decline model ('hyperbolic', 'exponential', 'harmonic')
        t_max: Maximum time for EUR calculation (months)
        econ_limit: Economic limit (minimum production rate)

    Returns:
        Dictionary with EUR and parameters, or None if fitting fails
    """
    if len(months) < 6:
        logger.warning("Insufficient data points for EUR calculation (need at least 6)")
        return None

    try:
        # Use existing fitting infrastructure
        if model_type == "hyperbolic":
            model = HyperbolicArps()
        else:
            from .models_arps import ExponentialArps, HarmonicArps

            if model_type == "exponential":
                model = ExponentialArps()
            elif model_type == "harmonic":
                model = HarmonicArps()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Fit model
        fit_spec = FitSpec(model=model)
        fitter = CurveFitFitter()
        t_array = months.values
        q_array = production.values

        fit_result = fitter.fit(t_array, q_array, fit_spec)

        if not fit_result.success:
            logger.warning(f"Fitting failed: {fit_result.message}")
            return None

        # Calculate EUR using reserves module
        from .models import ArpsParams

        # Ensure b parameter exists (default to 0 for exponential)
        fit_params = fit_result.params.copy()
        if "b" not in fit_params:
            fit_params["b"] = 0.0

        params = ArpsParams(**fit_params)
        reserves_result = forecast_and_reserves(
            params, t_max=t_max, econ_limit=econ_limit
        )

        return {
            "eur": reserves_result["eur"],
            "qi": fit_result.params.get("qi", 0),
            "di": fit_result.params.get("di", 0),
            "b": fit_result.params.get("b", 0),
            "model_type": model_type,
            "rmse": fit_result.rmse if hasattr(fit_result, "rmse") else None,
        }

    except Exception as e:
        logger.warning(f"EUR calculation failed: {e}")
        return None


def _calculate_eur_single_well(
    well_id: str,
    well_data: pd.DataFrame,
    well_id_col: str,
    date_col: str,
    value_col: str,
    model_type: str,
) -> Optional[Dict]:
    """Calculate EUR for a single well (helper for parallelization)."""
    well_data = well_data.sort_values(date_col)

    # Calculate months since start
    well_data[date_col] = pd.to_datetime(well_data[date_col])
    months_since_start = (
        well_data[date_col] - well_data[date_col].min()
    ).dt.days / 30.44

    eur_result = calculate_eur_from_production(
        months_since_start, well_data[value_col], model_type=model_type
    )

    if eur_result:
        eur_result[well_id_col] = well_id
        return eur_result
    return None


def calculate_eur_batch(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil_bbl",
    model_type: str = "hyperbolic",
    min_months: int = 6,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Calculate EUR for multiple wells from production DataFrame.

    Args:
        df: DataFrame with production data
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        model_type: Type of decline model
        min_months: Minimum months of data required
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)

    Returns:
        DataFrame with EUR results for each well
    """
    # Filter wells with sufficient data first
    well_counts = df.groupby(well_id_col).size()
    valid_wells = well_counts[well_counts >= min_months].index

    if len(valid_wells) == 0:
        logger.warning("No wells with sufficient data")
        return pd.DataFrame()

    # Prepare well data groups
    well_groups = [(well_id, df[df[well_id_col] == well_id]) for well_id in valid_wells]

    # Parallel or sequential processing
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(well_groups) > 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_eur_single_well)(
                well_id, well_data, well_id_col, date_col, value_col, model_type
            )
            for well_id, well_data in well_groups
        )
    else:
        results = [
            _calculate_eur_single_well(
                well_id, well_data, well_id_col, date_col, value_col, model_type
            )
            for well_id, well_data in well_groups
        ]

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        logger.warning("No EUR calculations succeeded")
        return pd.DataFrame()

    return pd.DataFrame(results)
