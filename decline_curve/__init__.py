"""Decline Curve Analysis package.

A comprehensive Python library for decline curve analysis with multi-phase
forecasting, data utilities, and ML models.
"""

from . import (  # noqa: F401
    dca,
    eur_estimation,
    monte_carlo,
    multiphase,
    panel_analysis,
    panel_analysis_sweep,
    profiling,
)
from .forecast_statistical import (  # noqa: F401
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from .logging_config import configure_logging, get_logger  # noqa: F401

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

__version__ = "0.2.0"
