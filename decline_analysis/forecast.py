from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .forecast_chronos import forecast_chronos
from .forecast_timesfm import forecast_timesfm
from .models import fit_arps, predict_arps

try:
    from .forecast_arima import forecast_arima

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

    def forecast_arima(*args, **kwargs):
        raise ImportError("ARIMA forecasting is not available due to dependency issues")


from .evaluate import mae, rmse, smape
from .plot import _range_markers, tufte_style


class Forecaster:
    def __init__(self, series: pd.Series):
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Input must be indexed by datetime")
        if not series.index.freq:
            series = series.asfreq(pd.infer_freq(series.index))
        self.series = series.dropna().copy()
        self.last_forecast = None

    def forecast(
        self,
        model: Literal["arps", "timesfm", "chronos", "arima"],
        kind: Optional[Literal["exponential", "harmonic", "hyperbolic"]] = "hyperbolic",
        horizon: Optional[int] = 12,
    ) -> pd.Series:
        if model == "arps":
            t = np.arange(len(self.series))
            q = self.series.to_numpy()
            params = fit_arps(t, q, kind=kind)
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

        else:
            raise ValueError(f"Unknown model: {model}")

        self.last_forecast = forecast
        return forecast

    def evaluate(self, actual: pd.Series) -> dict:
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
