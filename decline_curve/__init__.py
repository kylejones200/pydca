"""Decline Curve Analysis package.

A comprehensive Python library for decline curve analysis with multi-phase
forecasting, data utilities, and ML models.
"""

from . import dca, monte_carlo, multiphase, profiling  # noqa: F401
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

__version__ = "0.2.0"
