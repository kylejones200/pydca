"""Forecasting module for decline curve analysis."""

from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import mae, rmse, smape
from .forecast_chronos import forecast_chronos
from .forecast_statistical import (
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from .forecast_timesfm import forecast_timesfm
from .models import fit_arps, predict_arps
from .plot import _range_markers, tufte_style

try:
    from .forecast_arima import forecast_arima

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

    def forecast_arima(
        series: pd.Series,
        horizon: int = 12,
        order: Optional[tuple[int, int, int]] = None,
        seasonal: bool = False,
        seasonal_period: int = 12,
    ) -> pd.Series:
        """Raise error when ARIMA module is not available."""
        raise ImportError("ARIMA forecasting is not available due to dependency issues")


def forecast_arps(
    series: pd.Series,
    kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
    horizon: int = 12,
) -> pd.Series:
    """Generate forecast using Arps decline model.

    Args:
        series: Historical production time series
        kind: Type of Arps decline (exponential, harmonic, hyperbolic)
        horizon: Number of periods to forecast

    Returns:
        Forecasted production series
    """
    t = np.arange(len(series))
    q = series.to_numpy()
    params = fit_arps(t, q, kind=kind)
    full_t = np.arange(len(series) + horizon)
    yhat = predict_arps(full_t, params)
    idx = pd.date_range(series.index[0], periods=len(yhat), freq=series.index.freq)
    return pd.Series(yhat, index=idx, name=f"arps_{kind}")


class Forecaster:
    """Forecaster for production time series."""

    def __init__(self, series: pd.Series):
        """Initialize forecaster with historical series."""
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Input must be indexed by datetime")
        if not series.index.freq:
            series = series.asfreq(pd.infer_freq(series.index))
        self.series = series.dropna().copy()
        self.last_forecast = None

    def forecast(
        self,
        model: Literal[
            "arps",
            "timesfm",
            "chronos",
            "arima",
            "deepar",
            "ensemble",
            "exponential_smoothing",
            "moving_average",
            "linear_trend",
            "holt_winters",
        ],
        kind: Optional[Literal["exponential", "harmonic", "hyperbolic"]] = "hyperbolic",
        horizon: int = 12,
        **kwargs,
    ) -> pd.Series:
        """Generate forecast using specified model.

        Args:
            model: Forecasting model to use
            kind: Arps decline type (if using arps model)
            horizon: Number of periods to forecast
            **kwargs: Additional model-specific arguments

        Returns:
            Forecasted production series
        """
        if model == "arps":
            t = np.arange(len(self.series))
            q = self.series.to_numpy()
            arps_kind: Literal["exponential", "harmonic", "hyperbolic"] = (
                "hyperbolic"
                if kind is None
                else cast(Literal["exponential", "harmonic", "hyperbolic"], kind)
            )
            params = fit_arps(t, q, kind=arps_kind)
            full_t = np.arange(len(self.series) + horizon)
            yhat = predict_arps(full_t, params)
            idx = pd.date_range(
                self.series.index[0], periods=len(yhat), freq=self.series.index.freq
            )
            forecast = pd.Series(yhat, index=idx, name=f"arps_{kind}")

        elif model == "timesfm":
            forecast = forecast_timesfm(self.series, horizon=horizon)

        elif model == "chronos":
            forecast = forecast_chronos(self.series, horizon=horizon)

        elif model == "arima":
            forecast_part = forecast_arima(self.series, horizon=horizon)
            # Combine historical and forecast data
            full_index = pd.date_range(
                self.series.index[0],
                periods=len(self.series) + horizon,
                freq=self.series.index.freq,
            )
            full_forecast = pd.concat([self.series, forecast_part])
            forecast = pd.Series(
                full_forecast.values, index=full_index, name="arima_forecast"
            )

        elif model == "exponential_smoothing":
            alpha = kwargs.get("alpha", 0.3)
            forecast = simple_exponential_smoothing(
                self.series, alpha=alpha, horizon=horizon
            )

        elif model == "moving_average":
            window = kwargs.get("window", 6)
            forecast = moving_average_forecast(
                self.series, window=window, horizon=horizon
            )

        elif model == "linear_trend":
            forecast = linear_trend_forecast(self.series, horizon=horizon)

        elif model == "holt_winters":
            seasonal_periods = kwargs.get("seasonal_periods", None)
            forecast_result = holt_winters_forecast(
                self.series, horizon=horizon, seasonal_periods=seasonal_periods
            )
            if forecast_result is None:
                raise ValueError("Holt-Winters forecast failed")
            forecast = forecast_result

        else:
            raise ValueError(f"Unknown model: {model}")

        self.last_forecast = forecast
        return forecast

    def evaluate(self, actual: pd.Series) -> dict:
        """Evaluate forecast against actual values.

        Args:
            actual: Actual production values

        Returns:
            Dictionary with evaluation metrics (rmse, mae, smape)
        """
        if self.last_forecast is None:
            raise RuntimeError("Call .forecast() first.")
        common = self.last_forecast.index.intersection(actual.index)
        if len(common) == 0:
            raise ValueError("No overlapping dates to compare.")
        yhat = self.last_forecast.loc[common]
        ytrue = actual.loc[common]
        return {
            "rmse": rmse(ytrue, yhat),
            "mae": mae(ytrue, yhat),
            "smape": smape(ytrue, yhat),
        }

    def plot(self, title: str = "Forecast", filename: Optional[str] = None):
        """Plot forecast with historical data.

        Args:
            title: Plot title
            filename: Optional filename to save plot
        """
        if self.last_forecast is None:
            raise RuntimeError("Call .forecast() first.")
        tufte_style()
        fig, ax = plt.subplots()
        hist = self.series
        fcst = self.last_forecast

        ax.plot(hist.index, hist.values, lw=1.0, label="history")
        ax.plot(fcst.index, fcst.values, lw=1.2, label="forecast")

        _range_markers(ax, hist.values)
        ax.set_xlabel("Date")
        ax.set_ylabel("Production")
        ax.set_title(title)
        ax.legend()
        if filename:
            plt.savefig(filename, bbox_inches="tight")
        plt.show()
