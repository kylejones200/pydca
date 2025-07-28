from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

try:
    from pmdarima import auto_arima

    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    auto_arima = None


def forecast_arima(
    series: pd.Series,
    horizon: int = 12,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal: bool = False,
    seasonal_period: int = 12,
) -> pd.Series:
    """
    Forecast using ARIMA model.

    Args:
        series: Univariate time series with a DateTimeIndex.
        horizon: Forecast horizon.
        order: Optional ARIMA (p,d,q) order. If None, uses auto_arima.
        seasonal: Whether to include seasonal ARIMA terms.
        seasonal_period: The seasonal period if seasonal is True.

    Returns:
        Forecasted pandas Series with horizon steps.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    if horizon <= 0:
        raise ValueError("Horizon must be positive")

    # Store original frequency before dropping NaN values
    original_freq = series.index.freq or pd.infer_freq(series.index)

    y = series.dropna()

    if len(y) == 0:
        raise ValueError("Series contains no valid data after removing NaN values")

    # Handle insufficient data for seasonal models
    if seasonal and len(y) < 2 * seasonal_period:
        seasonal = False  # Disable seasonal component
        seasonal_period = 1

    # Handle very short series
    if len(y) < 3:
        # For very short series, use simple exponential smoothing approach
        last_value = y.iloc[-1]
        trend = 0
        if len(y) >= 2:
            trend = y.iloc[-1] - y.iloc[-2]

        forecast_values = [last_value + trend * i for i in range(1, horizon + 1)]

        # Generate future index
        freq = y.index.freq or pd.infer_freq(y.index)
        if freq is None:
            # Fallback to daily frequency if can't infer
            freq = "D"

        try:
            future_idx = pd.date_range(
                y.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq,
            )
        except Exception:
            # If date range generation fails, use simple integer offset
            future_idx = pd.date_range(y.index[-1], periods=horizon + 1, freq="D")[1:]

        return pd.Series(forecast_values, index=future_idx, name="arima_forecast")

    try:
        # Ensure we have enough data after dropping NaN values
        if len(y) < 3:
            # Use simple trend-based forecast for very short series
            if len(y) >= 2:
                trend = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
                forecast_values = [
                    y.iloc[-1] + trend * i for i in range(1, horizon + 1)
                ]
            else:
                forecast_values = [y.iloc[-1]] * horizon

            # Generate future index using original frequency
            freq = original_freq or "MS"
            try:
                last_date = series.index[-1]
                future_idx = pd.date_range(
                    last_date + pd.tseries.frequencies.to_offset(freq),
                    periods=horizon,
                    freq=freq,
                )
            except Exception:
                last_date = series.index[-1]
                future_idx = pd.date_range(last_date, periods=horizon + 1, freq="MS")[
                    1:
                ]

            return pd.Series(forecast_values, index=future_idx, name="arima_forecast")

        if order is None:
            if not PMDARIMA_AVAILABLE:
                # Fallback to simple ARIMA(1,1,1) when pmdarima is not available
                order = (1, 1, 1)
            else:
                model = auto_arima(
                    y,
                    seasonal=seasonal,
                    m=seasonal_period if seasonal else 1,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                    max_p=min(2, len(y) // 4),
                    max_q=min(2, len(y) // 4),
                    max_d=min(1, len(y) // 6),
                )
                order = model.order

        # Validate order parameters for reduced dataset
        p, d, q = order
        if p + d + q > len(y) - 2:
            # Use very simple model for limited data
            order = (1, 0, 0)

        model = ARIMA(y, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)

        # Ensure forecast values are not NaN
        if np.isnan(forecast).any():
            # Fallback to simple trend forecast
            if len(y) >= 2:
                trend = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
                forecast = np.array(
                    [y.iloc[-1] + trend * i for i in range(1, horizon + 1)]
                )
            else:
                forecast = np.array([y.iloc[-1]] * horizon)

        # Generate future index using original frequency
        freq = original_freq or y.index.freq or pd.infer_freq(y.index)
        if freq is None:
            # Try to infer frequency from the original series or remaining data
            if len(series) >= 2:
                # Use original series to infer frequency
                time_diffs = series.index[1:] - series.index[:-1]
                most_common_diff = (
                    time_diffs.mode()[0]
                    if len(time_diffs.mode()) > 0
                    else time_diffs[0]
                )
                freq = pd.tseries.frequencies.to_offset(most_common_diff)
            elif len(y) >= 2:
                time_diff = y.index[1] - y.index[0]
                freq = pd.tseries.frequencies.to_offset(time_diff)
            else:
                freq = "MS"  # Default to monthly start

        try:
            # Use the last date from original series for continuity
            last_date = series.index[-1]
            future_idx = pd.date_range(
                last_date + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq,
            )
        except Exception:
            # Fallback: create index with inferred frequency
            last_date = series.index[-1]
            if len(series) >= 2:
                time_diffs = series.index[1:] - series.index[:-1]
                most_common_diff = (
                    time_diffs.mode()[0]
                    if len(time_diffs.mode()) > 0
                    else time_diffs[0]
                )
                future_idx = [
                    last_date + most_common_diff * (i + 1) for i in range(horizon)
                ]
                future_idx = pd.DatetimeIndex(future_idx)
            else:
                future_idx = pd.date_range(last_date, periods=horizon + 1, freq="MS")[
                    1:
                ]

        return pd.Series(forecast, index=future_idx, name="arima_forecast")

    except Exception as e:
        # Fallback to simple trend-based forecast
        if len(y) >= 2:
            trend = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
            forecast_values = [y.iloc[-1] + trend * i for i in range(1, horizon + 1)]
        else:
            forecast_values = [y.iloc[-1]] * horizon

        # Generate future index using original frequency
        freq = original_freq or "MS"
        try:
            last_date = series.index[-1]
            future_idx = pd.date_range(
                last_date + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq,
            )
        except Exception:
            last_date = series.index[-1]
            if len(series) >= 2:
                time_diffs = series.index[1:] - series.index[:-1]
                most_common_diff = (
                    time_diffs.mode()[0]
                    if len(time_diffs.mode()) > 0
                    else time_diffs[0]
                )
                future_idx = [
                    last_date + most_common_diff * (i + 1) for i in range(horizon)
                ]
                future_idx = pd.DatetimeIndex(future_idx)
            else:
                future_idx = pd.date_range(last_date, periods=horizon + 1, freq="MS")[
                    1:
                ]

        return pd.Series(forecast_values, index=future_idx, name="arima_forecast")

    # Final safety check - ensure no NaN values in the result
    if np.isnan(forecast).any():
        # Ultimate fallback: use last valid value
        last_valid = y.iloc[-1]
        forecast = np.full(horizon, last_valid)

    # Ensure the returned series has valid values
    result = pd.Series(forecast, index=future_idx, name="arima_forecast")

    # Replace any remaining NaN values with the last valid observation
    if result.isna().any():
        result = result.fillna(y.iloc[-1])

    return result
