"""First-class P10/P50/P90 forecast outputs.

This module exposes P10/P50/P90 forecasts as first-class outputs, providing
both curves and scalar summaries such as EUR bands and discounted cash flow bands.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .uncertainty_core import ForecastDraws

logger = get_logger(__name__)


@dataclass
class ProbabilisticForecast:
    """
    First-class probabilistic forecast with P10/P50/P90 outputs.

    This class provides both forecast curves and scalar summaries (EUR, NPV bands).

    Attributes:
        p10: P10 forecast series (optimistic)
        p50: P50 forecast series (median)
        p90: P90 forecast series (conservative)
        mean: Mean forecast series
        dates: Date index
        eur_bands: EUR bands (P10, P50, P90)
        npv_bands: NPV bands (P10, P50, P90) if economics provided
        draws: Underlying forecast draws (optional)
        price: Commodity price used for NPV bands ($/BOE or $/MCF)
        opex: Variable operating cost used for NPV bands ($/BOE)
        discount_rate: Annual discount rate used for NPV bands
    """

    p10: pd.Series
    p50: pd.Series
    p90: pd.Series
    mean: pd.Series
    dates: pd.DatetimeIndex
    eur_bands: Dict[str, float]
    npv_bands: Optional[Dict[str, float]] = None
    draws: Optional[ForecastDraws] = None
    price: Optional[float] = None
    opex: Optional[float] = None
    discount_rate: float = 0.10

    @classmethod
    def from_draws(
        cls,
        draws: ForecastDraws,
        econ_limit: Optional[float] = None,
        price: Optional[float] = None,
        opex: Optional[float] = None,
        discount_rate: float = 0.10,
    ) -> "ProbabilisticForecast":
        """
        Create probabilistic forecast from forecast draws.

        Args:
            draws: ForecastDraws object
            econ_limit: Economic limit for EUR calculation
            price: Unit price for NPV calculation
            opex: Operating cost per unit for NPV calculation
            discount_rate: Discount rate for NPV calculation

        Returns:
            ProbabilisticForecast object
        """
        # Calculate EUR bands
        eur_bands = _calculate_eur_bands(draws, econ_limit)

        # Calculate NPV bands if economics provided
        npv_bands = None
        if price is not None and opex is not None:
            npv_bands = _calculate_npv_bands(draws, price, opex, discount_rate)

        return cls(
            p10=draws.p10,
            p50=draws.p50,
            p90=draws.p90,
            mean=draws.mean,
            dates=draws.dates,
            eur_bands=eur_bands,
            npv_bands=npv_bands,
            draws=draws,
            price=price,
            opex=opex,
            discount_rate=discount_rate,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all quantiles.

        Returns:
            DataFrame with columns: p10, p50, p90, mean
        """
        return pd.DataFrame(
            {
                "p10": self.p10,
                "p50": self.p50,
                "p90": self.p90,
                "mean": self.mean,
            },
            index=self.dates,
        )

    def summary(self) -> Dict[str, any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "eur": self.eur_bands,
            "forecast_range": {
                "p10_max": float(self.p10.max()),
                "p50_max": float(self.p50.max()),
                "p90_max": float(self.p90.max()),
            },
        }

        if self.npv_bands:
            summary["npv"] = self.npv_bands

        return summary


def _calculate_eur_bands(
    draws: ForecastDraws, econ_limit: Optional[float] = None
) -> Dict[str, float]:
    """Calculate EUR bands from forecast draws.

    Args:
        draws: ForecastDraws object
        econ_limit: Economic limit (if None, uses full forecast)

    Returns:
        Dictionary with P10, P50, P90 EUR values
    """
    # Calculate cumulative production for each draw
    n_draws, n_periods = draws.draws.shape
    eur_samples = np.zeros(n_draws)

    for i in range(n_draws):
        forecast = draws.draws[i]
        if econ_limit is not None:
            # Find when rate drops below economic limit
            below_limit = forecast < econ_limit
            if below_limit.any():
                stop_idx = np.where(below_limit)[0][0]
                eur_samples[i] = np.sum(forecast[:stop_idx])
            else:
                eur_samples[i] = np.sum(forecast)
        else:
            # Use full forecast
            eur_samples[i] = np.sum(forecast)

    return {
        "p10": float(np.percentile(eur_samples, 90)),
        "p50": float(np.percentile(eur_samples, 50)),
        "p90": float(np.percentile(eur_samples, 10)),
        "mean": float(np.mean(eur_samples)),
    }


def _calculate_npv_bands(
    draws: ForecastDraws,
    price: float,
    opex: float,
    discount_rate: float = 0.10,
) -> Dict[str, float]:
    """Calculate NPV bands from forecast draws.

    Args:
        draws: ForecastDraws object
        price: Unit price
        opex: Operating cost per unit
        discount_rate: Annual discount rate

    Returns:
        Dictionary with P10, P50, P90 NPV values
    """
    from .economics import economic_metrics

    n_draws, n_periods = draws.draws.shape
    npv_samples = np.zeros(n_draws)

    # Monthly discount rate
    monthly_discount = (1 + discount_rate) ** (1 / 12) - 1

    for i in range(n_draws):
        forecast = draws.draws[i]
        # Calculate cash flow
        cash_flow = (forecast * price) - (forecast * opex)
        # Discount to present value
        discount_factors = (1 + monthly_discount) ** np.arange(n_periods)
        npv = np.sum(cash_flow / discount_factors)
        npv_samples[i] = npv

    return {
        "p10": float(np.percentile(npv_samples, 90)),
        "p50": float(np.percentile(npv_samples, 50)),
        "p90": float(np.percentile(npv_samples, 10)),
        "mean": float(np.mean(npv_samples)),
    }


def probabilistic_forecast(
    series: pd.Series,
    model: str = "arps",
    kind: str = "hyperbolic",
    horizon: int = 12,
    n_draws: int = 1000,
    seed: Optional[int] = None,
    econ_limit: Optional[float] = None,
    price: Optional[float] = None,
    opex: Optional[float] = None,
    discount_rate: float = 0.10,
) -> ProbabilisticForecast:
    """
    Generate probabilistic forecast with P10/P50/P90 outputs.

    This is the main entry point for probabilistic forecasting. It generates
    forecast draws and returns a ProbabilisticForecast object with both curves
    and scalar summaries.

    Args:
        series: Historical production time series
        model: Forecasting model ('arps' for now, can be extended)
        kind: Arps decline type
        horizon: Forecast horizon
        n_draws: Number of Monte Carlo draws
        seed: Random seed
        econ_limit: Economic limit for EUR calculation
        price: Unit price for NPV calculation
        opex: Operating cost per unit for NPV calculation
        discount_rate: Discount rate for NPV calculation

    Returns:
        ProbabilisticForecast object

    Example:
        >>> import pandas as pd
        >>> from decline_curve.probabilistic_forecast import probabilistic_forecast
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> forecast = probabilistic_forecast(
        ...     production,
        ...     kind='hyperbolic',
        ...     horizon=12,
        ...     price=70,
        ...     opex=15
        ... )
        >>> print(f"P50 EUR: {forecast.eur_bands['p50']:,.0f} bbl")
        >>> print(f"P50 NPV: ${forecast.npv_bands['p50']:,.0f}")
    """
    if kind == "duong":
        from .parameter_resample import _duong_resample

        draws = _duong_resample(series, n_draws=n_draws, seed=seed, horizon=horizon)
    elif kind == "ple":
        from .parameter_resample import _ple_resample

        draws = _ple_resample(series, n_draws=n_draws, seed=seed, horizon=horizon)
    elif kind in ("modified_hyperbolic", "mh"):
        from .parameter_resample import _modified_hyperbolic_resample

        draws = _modified_hyperbolic_resample(series, n_draws=n_draws, seed=seed, horizon=horizon)
    elif kind == "sepd":
        from .parameter_resample import _sepd_resample

        draws = _sepd_resample(series, n_draws=n_draws, seed=seed, horizon=horizon)
    elif model == "arps":
        from .parameter_resample import fast_arps_resample

        draws = fast_arps_resample(
            series, kind=kind, n_draws=n_draws, seed=seed, horizon=horizon
        )
    else:
        raise ValueError(
            f"Model '{model}' not yet supported for probabilistic forecasting"
        )

    return ProbabilisticForecast.from_draws(
        draws,
        econ_limit=econ_limit,
        price=price,
        opex=opex,
        discount_rate=discount_rate,
    )
