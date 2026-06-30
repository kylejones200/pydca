"""Decline Curve Analysis package.

A comprehensive Python library for decline curve analysis with multi-phase
forecasting, data utilities, and ML models.
"""

# numpy 1.x/2.x compat: numpy>=2.0 removed the long-deprecated np.trapz in favour
# of np.trapezoid. Several internal modules still call np.trapz; restore the alias
# once at import time so the kernel runs on modern numpy. (Done before submodule
# imports below, which execute that code.)
import numpy as _np  # noqa: E402

if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid  # type: ignore[attr-defined]

from . import (  # noqa: F401
    catalog,
    config,
    dca,
    decline_variants,
    eur_estimation,
    history_matching,
    integrations,
    ipr,
    material_balance,
    model_comparison,
    model_interface,
    model_registry,
    monte_carlo,
    multiphase,
    multiphase_flow,
    panel_analysis,
    panel_analysis_sweep,
    parameter_resample,
    physics_informed,
    physics_reserves,
    portfolio,
    probabilistic_forecast,
    profiling,
    prms,
    pvt,
    risk_report,
    rta,
    runner,
    scenarios,
    schemas,
    segmented_decline,
    spatial_kriging,
    type_curve_normalization,
    uncertainty_core,
    vlp,
    well_test,
)
from .models import (  # noqa: F401
    ArpsParams,
    DuongParams,
    ModifiedHyperbolicParams,
    PLEParams,
    SEPDParams,
    fit_arps,
    predict_arps,
)
from .economics import (  # noqa: F401
    WellEconomics,
    CashflowResult,
    cashflow,
    npv,
    irr,
    payout,
    roi,
    breakeven_price,
    portfolio_economics,
)
from .decline_variants import (  # noqa: F401
    fit_duong,
    predict_duong,
    eur_duong,
    fit_ple,
    predict_ple,
    eur_ple,
    fit_sepd,
    predict_sepd,
    eur_sepd,
    fit_modified_hyperbolic,
    predict_modified_hyperbolic,
    eur_modified_hyperbolic,
    forecast_variant,
)
from .forecast_statistical import (  # noqa: F401
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from .logging_config import configure_logging, get_logger  # noqa: F401
from .prms import ReservesClassification, classify_reserves  # noqa: F401
from .type_curve_normalization import (  # noqa: F401
    TypeCurveMatch,
    generate_arps_type_curve,
    normalize_production_data,
    match_type_curve,
    denormalize_match,
)
from .dca import recommend_model  # noqa: F401
from .scenarios import (  # noqa: F401
    PriceScenario,
    ScenarioResult,
    run_price_scenarios,
    run_multi_phase_scenarios,
    compare_scenarios,
)

try:
    from . import deep_learning  # noqa: F401
except ImportError:
    # Deep learning module requires PyTorch
    pass

try:
    from . import ensemble, forecast_deepar, forecast_tft  # noqa: F401
except ImportError:
    # DeepAR, ensemble, and TFT modules require PyTorch
    pass

from .utils import data_processing  # noqa: F401

try:
    from . import benchmark_factory  # noqa: F401
except ImportError:
    # Benchmark factory requires optional dependencies
    pass

# Public API contract. Other libraries (e.g. ressmith) delegate to these names;
# treat additions as backward-compatible and removals/renames as breaking.
__all__ = [
    # version
    "__version__",
    # Arps + decline param models and core fit/predict (the kernel math)
    "ArpsParams", "DuongParams", "ModifiedHyperbolicParams", "PLEParams", "SEPDParams",
    "fit_arps", "predict_arps",
    # decline variants (shale + modified-hyperbolic w/ terminal switch)
    "fit_duong", "predict_duong", "eur_duong",
    "fit_ple", "predict_ple", "eur_ple",
    "fit_sepd", "predict_sepd", "eur_sepd",
    "fit_modified_hyperbolic", "predict_modified_hyperbolic", "eur_modified_hyperbolic",
    "forecast_variant",
    # economics (effective-annual DCF — the canonical convention)
    "WellEconomics", "CashflowResult", "cashflow", "npv", "irr", "payout", "roi",
    "breakeven_price", "portfolio_economics",
    # reserves / PRMS classification
    "ReservesClassification", "classify_reserves",
    # type-curve normalization
    "TypeCurveMatch", "generate_arps_type_curve", "normalize_production_data",
    "match_type_curve", "denormalize_match",
    # scenarios
    "PriceScenario", "ScenarioResult", "run_price_scenarios",
    "run_multi_phase_scenarios", "compare_scenarios",
    # statistical forecasters
    "calculate_confidence_intervals", "holt_winters_forecast", "linear_trend_forecast",
    "moving_average_forecast", "simple_exponential_smoothing",
    # misc
    "recommend_model", "configure_logging", "get_logger",
]

__version__ = "0.6.0"
