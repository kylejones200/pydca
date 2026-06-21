"""Decline Curve Analysis package.

A comprehensive Python library for decline curve analysis with multi-phase
forecasting, data utilities, and ML models.
"""

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

__version__ = "0.6.0"
