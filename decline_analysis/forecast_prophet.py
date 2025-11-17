"""Prophet-based forecasting for decline curve analysis.

Facebook Prophet is a time series forecasting library that works well with
data that has strong seasonal patterns and handles missing data gracefully.
"""

import numpy as np
import pandas as pd

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet  # Legacy import

        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False


def forecast_prophet(
    series: pd.Series,
    horizon: int = 12,
    seasonality_mode: str = "multiplicative",
    yearly_seasonality: bool = False,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
    uncertainty_samples: int = 1000,
    include_history: bool = True,
) -> pd.Series:
    """
    Generate production forecast using Facebook Prophet.

    Prophet is particularly useful for production data with:
    - Seasonal patterns (e.g., winter gas demand)
    - Multiple trend changes (workovers, shutdowns)
    - Missing data or outliers

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast
        seasonality_mode: 'additive' or 'multiplicative'
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
        changepoint_prior_scale: Flexibility of trend changes (0.001-0.5)
        uncertainty_samples: Number of samples for uncertainty intervals
        include_history: Include historical data in output

    Returns:
        Forecasted production series (with or without history)

    Raises:
        ImportError: If Prophet is not installed

    Example:
        >>> forecast = forecast_prophet(oil_series, horizon=24, yearly_seasonality=True)
        >>> print(f"24-month forecast: {forecast.tail(24).sum():.0f} bbl")

    Note:
        Install Prophet with: pip install prophet
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        )

    # Prepare data in Prophet format (ds, y)
    df = pd.DataFrame({"ds": series.index, "y": series.values})

    # Initialize and configure Prophet model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        uncertainty_samples=uncertainty_samples,
    )

    # Fit model
    model.fit(df)

    # Create future dataframe
    if include_history:
        future = model.make_future_dataframe(periods=horizon, freq=series.index.freq)
    else:
        # Only forecast future periods
        future = model.make_future_dataframe(periods=horizon, freq=series.index.freq)
        future = future.tail(horizon)

    # Generate forecast
    forecast_df = model.predict(future)

    # Extract forecast values and create series
    result = pd.Series(
        forecast_df["yhat"].values,
        index=pd.DatetimeIndex(forecast_df["ds"]),
        name=series.name or "forecast",
    )

    # Ensure non-negative production
    result = result.clip(lower=0)

    return result


def forecast_prophet_with_uncertainty(
    series: pd.Series, horizon: int = 12, **kwargs
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generate Prophet forecast with uncertainty intervals.

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast
        **kwargs: Additional arguments passed to Prophet

    Returns:
        Tuple of (forecast, lower_bound, upper_bound) series

    Example:
        >>> forecast, lower, upper = forecast_prophet_with_uncertainty(
        ...     oil_series, horizon=12
        ... )
        >>> uncertainty = (upper.iloc[-1] - lower.iloc[-1]) / 2
        >>> print(f"Forecast: {forecast.iloc[-1]:.0f} ± {uncertainty:.0f}")
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        )

    # Prepare data
    df = pd.DataFrame({"ds": series.index, "y": series.values})

    # Initialize model with defaults
    model_kwargs = {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05,
        "uncertainty_samples": 1000,
    }
    model_kwargs.update(kwargs)

    model = Prophet(**model_kwargs)
    model.fit(df)

    # Generate forecast
    future = model.make_future_dataframe(periods=horizon, freq=series.index.freq)
    forecast_df = model.predict(future)

    # Extract forecast and bounds
    forecast = pd.Series(
        forecast_df["yhat"].values,
        index=pd.DatetimeIndex(forecast_df["ds"]),
        name="forecast",
    ).clip(lower=0)

    lower_bound = pd.Series(
        forecast_df["yhat_lower"].values,
        index=pd.DatetimeIndex(forecast_df["ds"]),
        name="lower",
    ).clip(lower=0)

    upper_bound = pd.Series(
        forecast_df["yhat_upper"].values,
        index=pd.DatetimeIndex(forecast_df["ds"]),
        name="upper",
    ).clip(lower=0)

    return forecast, lower_bound, upper_bound


def add_custom_seasonality(
    series: pd.Series,
    horizon: int = 12,
    seasonality_name: str = "monthly",
    period: float = 30.5,
    fourier_order: int = 5,
) -> pd.Series:
    """
    Add custom seasonality to Prophet model.

    Useful for production data with specific seasonal patterns
    (e.g., winter heating demand, summer cooling).

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast
        seasonality_name: Name for the custom seasonality
        period: Period of seasonality in days
        fourier_order: Number of Fourier terms (higher = more flexible)

    Returns:
        Forecasted series with custom seasonality

    Example:
        >>> # Add quarterly seasonality
        >>> forecast = add_custom_seasonality(gas_series, period=91.25, fourier_order=3)
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        )

    df = pd.DataFrame({"ds": series.index, "y": series.values})

    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
    )

    # Add custom seasonality
    model.add_seasonality(
        name=seasonality_name, period=period, fourier_order=fourier_order
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq=series.index.freq)
    forecast_df = model.predict(future)

    result = pd.Series(
        forecast_df["yhat"].values,
        index=pd.DatetimeIndex(forecast_df["ds"]),
        name=series.name or "forecast",
    ).clip(lower=0)

    return result


def detect_changepoints(
    series: pd.Series, n_changepoints: int = 25, changepoint_range: float = 0.8
) -> pd.DataFrame:
    """
    Detect significant trend changes in production data.

    Useful for identifying:
    - Workovers or stimulation treatments
    - Equipment failures
    - Production optimization events

    Args:
        series: Historical production time series
        n_changepoints: Number of potential changepoints
        changepoint_range: Proportion of history for changepoints (0-1)

    Returns:
        DataFrame with changepoint dates and magnitudes

    Example:
        >>> changepoints = detect_changepoints(oil_series)
        >>> print(f"Found {len(changepoints)} significant trend changes")
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        )

    df = pd.DataFrame({"ds": series.index, "y": series.values})

    model = Prophet(n_changepoints=n_changepoints, changepoint_range=changepoint_range)
    model.fit(df)

    # Get changepoints
    changepoints = model.changepoints

    # Calculate changepoint magnitudes
    deltas = model.params["delta"].mean(axis=0)

    changepoint_df = pd.DataFrame({"date": changepoints, "delta": deltas})

    # Filter to significant changepoints
    threshold = np.abs(deltas).std()
    significant = changepoint_df[np.abs(changepoint_df["delta"]) > threshold]

    return significant.sort_values("date")


# Convenience function for backward compatibility
def prophet_forecast(series: pd.Series, horizon: int = 12, **kwargs) -> pd.Series:
    """
    Alias for forecast_prophet for backward compatibility.

    Args:
        series: Historical production time series
        horizon: Number of periods to forecast
        **kwargs: Additional Prophet parameters

    Returns:
        Forecasted production series
    """
    return forecast_prophet(series, horizon=horizon, **kwargs)
