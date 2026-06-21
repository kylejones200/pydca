"""Risk report helper for forecast distributions.

This module provides a simple risk report helper that reads forecast distributions
and price assumptions and returns key risk metrics. For example probability that
NPV exceeds a threshold at well level and portfolio level.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .probabilistic_forecast import ProbabilisticForecast
from .uncertainty_core import ForecastDraws

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """
    Risk metrics for a single well or portfolio.

    Attributes:
        prob_positive_npv: Probability that NPV > 0
        prob_npv_above_threshold: Probability that NPV exceeds threshold
        expected_npv: Expected (mean) NPV
        value_at_risk_90: VaR at 90% confidence (P90 NPV)
        conditional_value_at_risk: CVaR (expected loss given VaR breach)
        eur_range: Range between P10 and P90 EUR
        npv_range: Range between P10 and P90 NPV
    """

    prob_positive_npv: float
    prob_npv_above_threshold: float
    expected_npv: float
    value_at_risk_90: float
    conditional_value_at_risk: float
    eur_range: float
    npv_range: float


def calculate_risk_metrics(
    forecast: ProbabilisticForecast,
    npv_threshold: float = 0.0,
) -> RiskMetrics:
    """
    Calculate risk metrics from probabilistic forecast.

    Args:
        forecast: ProbabilisticForecast object
        npv_threshold: NPV threshold for probability calculation

    Returns:
        RiskMetrics object

    Example:
        >>> from decline_curve.probabilistic_forecast import probabilistic_forecast
        >>> from decline_curve.risk_report import calculate_risk_metrics
        >>> forecast = probabilistic_forecast(series, price=70, opex=15)
        >>> metrics = calculate_risk_metrics(forecast, npv_threshold=1000000)
        >>> print(f"Probability of positive NPV: {metrics.prob_positive_npv:.1%}")
    """
    if forecast.draws is None:
        raise ValueError("Forecast draws not available for risk calculation.")

    if forecast.price is None or forecast.opex is None:
        raise ValueError(
            "price and opex must be provided to probabilistic_forecast() "
            "to compute risk metrics."
        )

    npv_samples = _calculate_npv_samples_from_draws(
        forecast.draws,
        price=forecast.price,
        opex=forecast.opex,
        discount_rate=forecast.discount_rate,
    )

    # Calculate risk metrics
    prob_positive_npv = float(np.mean(npv_samples > 0))
    prob_npv_above_threshold = float(np.mean(npv_samples > npv_threshold))
    expected_npv = float(np.mean(npv_samples))
    value_at_risk_90 = float(np.percentile(npv_samples, 10))  # P90 = 10th percentile
    conditional_value_at_risk = float(
        np.mean(npv_samples[npv_samples <= value_at_risk_90])
    )

    eur_range = forecast.eur_bands["p10"] - forecast.eur_bands["p90"]
    npv_range = forecast.npv_bands["p10"] - forecast.npv_bands["p90"]

    return RiskMetrics(
        prob_positive_npv=prob_positive_npv,
        prob_npv_above_threshold=prob_npv_above_threshold,
        expected_npv=expected_npv,
        value_at_risk_90=value_at_risk_90,
        conditional_value_at_risk=conditional_value_at_risk,
        eur_range=eur_range,
        npv_range=npv_range,
    )


def _calculate_npv_samples_from_draws(
    draws: ForecastDraws,
    price: float,
    opex: float,
    discount_rate: float = 0.10,
) -> np.ndarray:
    """Calculate per-draw NPV samples using the full economics engine.

    Args:
        draws: ForecastDraws with shape (n_draws, n_periods).
        price: Commodity price ($/BOE).
        opex: Variable operating cost ($/BOE).
        discount_rate: Annual discount rate.

    Returns:
        Array of NPV values, one per draw.
    """
    from .economics import WellEconomics, cashflow, npv as _npv

    econ = WellEconomics(
        capex=0.0,
        price=price,
        opex_variable=opex,
        discount_rate=discount_rate,
        royalty_rate=0.0,
        severance_tax_rate=0.0,
        ad_valorem_rate=0.0,
    )
    n_draws = draws.draws.shape[0]
    npv_samples = np.zeros(n_draws)
    for i in range(n_draws):
        result = cashflow(draws.draws[i], econ)
        npv_samples[i] = _npv(result)
    return npv_samples


def portfolio_risk_report(
    forecasts: Dict[str, ProbabilisticForecast],
    npv_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Generate portfolio-level risk report.

    Aggregates risk metrics across multiple wells to provide portfolio-level
    risk assessment.

    Args:
        forecasts: Dictionary mapping well_id to ProbabilisticForecast
        npv_threshold: NPV threshold for probability calculation

    Returns:
        DataFrame with risk metrics for each well and portfolio summary

    Example:
        >>> forecasts = {
        ...     'well_001': forecast1,
        ...     'well_002': forecast2,
        ... }
        >>> report = portfolio_risk_report(forecasts, npv_threshold=1000000)
        >>> print(report)
    """
    well_metrics = []

    for well_id, forecast in forecasts.items():
        try:
            metrics = calculate_risk_metrics(forecast, npv_threshold)
            well_metrics.append(
                {
                    "well_id": well_id,
                    "prob_positive_npv": metrics.prob_positive_npv,
                    "prob_npv_above_threshold": metrics.prob_npv_above_threshold,
                    "expected_npv": metrics.expected_npv,
                    "var_90": metrics.value_at_risk_90,
                    "cvar": metrics.conditional_value_at_risk,
                    "eur_p50": forecast.eur_bands["p50"],
                    "npv_p50": (
                        forecast.npv_bands["p50"] if forecast.npv_bands else None
                    ),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to calculate risk metrics for {well_id}: {e}")

    df = pd.DataFrame(well_metrics)

    # Add portfolio summary row
    if len(df) > 0:
        portfolio_summary = {
            "well_id": "PORTFOLIO",
            "prob_positive_npv": df["prob_positive_npv"].mean(),
            "prob_npv_above_threshold": df["prob_npv_above_threshold"].mean(),
            "expected_npv": df["expected_npv"].sum(),
            "var_90": df["var_90"].sum(),  # Portfolio VaR (simplified)
            "cvar": df["cvar"].sum(),  # Portfolio CVaR (simplified)
            "eur_p50": df["eur_p50"].sum(),
            "npv_p50": df["npv_p50"].sum() if "npv_p50" in df else None,
        }
        df = pd.concat([df, pd.DataFrame([portfolio_summary])], ignore_index=True)

    return df
