"""Default configuration for decline curve analysis.

This module centralizes all default values used throughout the library.
Treat defaults as part of the API - changes should be documented in changelog.
"""

from typing import Any, Dict

# Default fitting parameters
DEFAULT_FIT_SPEC: Dict[str, Any] = {
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "method": "curve_fit",
    "weights": None,
    "fixed_params": {},
}

# Default segmentation parameters
DEFAULT_SEGMENT_SPEC: Dict[str, Any] = {
    "method": "pelt",
    "min_segment_size": 6,
    "penalty": "BIC",
}

# Default outlier detection parameters
DEFAULT_OUTLIER_SPEC: Dict[str, Any] = {
    "method": "hampel",
    "threshold": 3.0,
    "window_size": 5,
    "tail_retention": True,
}

# Default uncertainty parameters
DEFAULT_UNCERTAINTY_SPEC: Dict[str, Any] = {
    "n_draws": 1000,
    "seed": 42,
    "method": "auto",  # auto-select covariance or bootstrap
}

# Default economic parameters
DEFAULT_ECON_SPEC: Dict[str, Any] = {
    "oil_price": 70.0,
    "gas_price": 3.0,
    "water_price": -2.0,
    "fixed_opex": 5000.0,
    "variable_opex": 5.0,
    "discount_rate": 0.10,
    "royalty_rate": 0.125,
    "tax_rate": 0.0,
}

# Default batch processing parameters
DEFAULT_BATCH_SPEC: Dict[str, Any] = {
    "n_jobs": -1,  # Use all cores
    "seed": 42,
    "output_format": "parquet",
}

# Default diagnostic thresholds
DEFAULT_DIAGNOSTIC_THRESHOLDS: Dict[str, float] = {
    "rmse_warning": 0.2,  # 20% of mean rate
    "mape_warning": 15.0,  # 15% MAPE
    "r_squared_min": 0.7,  # Minimum R² for good fit
    "monotonicity_tolerance": 1e-6,
}

# Default backtesting parameters
DEFAULT_BACKTEST_SPEC: Dict[str, Any] = {
    "method": "walk_forward",
    "forecast_horizons": [12, 24, 36],  # months
    "min_train_size": 12,
    "step_size": 1,
    "seed": 42,
}

# Default report parameters
DEFAULT_REPORT_SPEC: Dict[str, Any] = {
    "format": "html",
    "include_plots": True,
    "include_diagnostics": True,
    "include_provenance": True,
}


def get_defaults() -> Dict[str, Dict[str, Any]]:
    """Get all default configurations.

    Returns:
        Dictionary of all default specs
    """
    return {
        "fit": DEFAULT_FIT_SPEC,
        "segment": DEFAULT_SEGMENT_SPEC,
        "outlier": DEFAULT_OUTLIER_SPEC,
        "uncertainty": DEFAULT_UNCERTAINTY_SPEC,
        "econ": DEFAULT_ECON_SPEC,
        "batch": DEFAULT_BATCH_SPEC,
        "diagnostic": DEFAULT_DIAGNOSTIC_THRESHOLDS,
        "backtest": DEFAULT_BACKTEST_SPEC,
        "report": DEFAULT_REPORT_SPEC,
    }


def get_default(spec_type: str, key: str, default: Any = None) -> Any:
    """Get a specific default value.

    Args:
        spec_type: Type of spec ('fit', 'segment', etc.)
        key: Key within the spec
        default: Default value if not found

    Returns:
        Default value
    """
    defaults = get_defaults()
    if spec_type in defaults:
        return defaults[spec_type].get(key, default)
    return default
