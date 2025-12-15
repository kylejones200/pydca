"""Statistical forecasting methods for production time series.

This module provides additional statistical forecasting methods beyond
the core Arps and ML models, including exponential smoothing, moving averages,
and trend-based forecasts.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def simple_exponential_smoothing(
    series: pd.Series, alpha: float = 0.3, horizon: int = 6
) -> pd.Series:
    """Generate forecast using simple exponential smoothing.

    Args:
        series: Historical production time series
        alpha: Smoothing parameter (0-1), higher = more weight on recent values
        horizon: Number of periods to forecast

    Returns:
        Forecasted production series
    """
    if len(series) == 0:
        raise ValueError("Series must not be empty")

    forecast_values = []
    last_value = series.iloc[-1]
    mean_value = series.mean()

    for i in range(horizon):
        if i == 0:
            forecast_values.append(alpha * last_value + (1 - alpha) * mean_value)
        else:
            forecast_values.append(
                alpha * forecast_values[-1] + (1 - alpha) * forecast_values[-1]
            )

    # Create index for forecast
    if isinstance(series.index, pd.DatetimeIndex):
        freq = series.index.freq or pd.infer_freq(series.index) or "MS"
        forecast_index = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq=freq
        )
    else:
        forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + horizon)

    return pd.Series(
        forecast_values, index=forecast_index, name="exponential_smoothing"
    )


def moving_average_forecast(
    series: pd.Series, window: int = 6, horizon: int = 6
) -> pd.Series:
    """Generate forecast using moving average.

    Args:
        series: Historical production time series
        window: Number of periods for moving average
        horizon: Number of periods to forecast

    Returns:
        Forecasted production series
    """
    if len(series) == 0:
        raise ValueError("Series must not be empty")

    ma = series.rolling(window=window, center=False).mean().iloc[-1]
    if pd.isna(ma):
        ma = series.mean()

    forecast_values = [ma] * horizon

    # Create index for forecast
    if isinstance(series.index, pd.DatetimeIndex):
        freq = series.index.freq or pd.infer_freq(series.index) or "MS"
        forecast_index = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq=freq
        )
    else:
        forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + horizon)

    return pd.Series(forecast_values, index=forecast_index, name="moving_average")


def linear_trend_forecast(series: pd.Series, horizon: int = 6) -> pd.Series:
    """Generate forecast using linear trend extrapolation.

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast

    Returns:
        Forecasted production series
    """
    if len(series) < 2:
        raise ValueError("Series must have at least 2 points for trend")

    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, 1)
    future_x = np.arange(len(series), len(series) + horizon)
    forecast_values = np.polyval(coeffs, future_x)

    # Ensure non-negative
    forecast_values = np.maximum(forecast_values, 0)

    # Create index for forecast
    if isinstance(series.index, pd.DatetimeIndex):
        freq = series.index.freq or pd.infer_freq(series.index) or "MS"
        forecast_index = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq=freq
        )
    else:
        forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + horizon)

    return pd.Series(forecast_values, index=forecast_index, name="linear_trend")


def holt_winters_forecast(
    series: pd.Series, horizon: int = 6, seasonal_periods: Optional[int] = None
) -> Optional[pd.Series]:
    """Generate forecast using Holt-Winters exponential smoothing.

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast
        seasonal_periods: Seasonal period (auto-detected if None and data sufficient)

    Returns:
        Forecasted production series, or None if insufficient data or statsmodels unavailable  # noqa: E501
    """
    if not HAS_STATSMODELS:
        logger.warning(
            "statsmodels not available. Install with: pip install statsmodels"
        )
        return None

    if len(series) < 12:
        logger.warning("Insufficient data for Holt-Winters (need at least 12 points)")
        return None

    try:
        # Auto-detect seasonality if not provided
        if seasonal_periods is None:
            if len(series) >= 24:
                seasonal_periods = 12  # Assume monthly seasonality
            else:
                seasonal_periods = None

        if seasonal_periods and len(series) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                series, seasonal="add", seasonal_periods=seasonal_periods
            ).fit()
        else:
            model = ExponentialSmoothing(series, trend="add").fit()

        forecast_values = model.forecast(horizon).values

        # Create index for forecast
        if isinstance(series.index, pd.DatetimeIndex):
            freq = series.index.freq or pd.infer_freq(series.index) or "MS"
            forecast_index = pd.date_range(
                series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq=freq
            )
        else:
            forecast_index = pd.RangeIndex(
                start=len(series), stop=len(series) + horizon
            )

        return pd.Series(forecast_values, index=forecast_index, name="holt_winters")

    except Exception as e:
        logger.warning(f"Holt-Winters forecast failed: {e}")
        return None


def calculate_confidence_intervals(
    series: pd.Series,
    forecast: pd.Series,
    method: str = "naive",
    confidence: float = 0.95,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Calculate confidence intervals for forecast.

    Args:
        series: Historical production time series
        forecast: Forecasted production series
        method: Method for calculating intervals ('naive', 'residual_based')
        confidence: Confidence level (0-1), default 0.95

    Returns:
        Tuple of (lower_bound, upper_bound) Series, or (None, None) if insufficient data
    """
    if len(series) < 10:
        return None, None

    # Calculate z-score for confidence level
    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence) / 2)

    if method == "naive":
        # Use recent historical standard deviation
        recent_std = series.iloc[-12:].std() if len(series) >= 12 else series.std()
        if pd.isna(recent_std) or recent_std == 0:
            recent_std = series.std()

        margin = z_score * recent_std
        upper = forecast + margin
        lower = forecast - margin

    elif method == "residual_based":
        # Use historical residuals if available
        # For now, fall back to naive
        recent_std = series.iloc[-12:].std() if len(series) >= 12 else series.std()
        margin = z_score * recent_std
        upper = forecast + margin
        lower = forecast - margin

    else:
        logger.warning(f"Unknown confidence interval method: {method}")
        return None, None

    # Ensure non-negative
    lower = lower.clip(lower=0)

    return lower, upper
