"""
Multi-phase production forecasting module.

This module enables simultaneous forecasting of oil, gas, and water production,
addressing a key limitation of traditional decline curve analysis which typically
handles only one phase at a time.

Based on research showing that coupled multi-target prediction yields more accurate
and consistent forecasts than independent single-phase models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class MultiPhaseData:
    """
    Container for multi-phase production data.

    Attributes:
        oil: Oil production time series (bbl/month)
        gas: Gas production time series (mcf/month)
        water: Water production time series (bbl/month)
        dates: Common date index
        well_id: Optional well identifier
    """

    oil: pd.Series
    gas: Optional[pd.Series] = None
    water: Optional[pd.Series] = None
    dates: Optional[pd.DatetimeIndex] = None
    well_id: Optional[str] = None

    def __post_init__(self):
        """Validate and align data."""
        if self.dates is None:
            self.dates = self.oil.index

        # Ensure all series have same index
        if self.gas is not None and not self.gas.index.equals(self.dates):
            raise ValueError("Gas series index must match oil series index")
        if self.water is not None and not self.water.index.equals(self.dates):
            raise ValueError("Water series index must match oil series index")

    @property
    def phases(self) -> list[str]:
        """Return list of available phases."""
        available = ["oil"]
        if self.gas is not None:
            available.append("gas")
        if self.water is not None:
            available.append("water")
        return available

    @property
    def length(self) -> int:
        """Return number of time steps."""
        return len(self.oil)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all phases."""
        df = pd.DataFrame({"oil": self.oil}, index=self.dates)
        if self.gas is not None:
            df["gas"] = self.gas
        if self.water is not None:
            df["water"] = self.water
        return df

    def calculate_ratios(self) -> dict[str, pd.Series]:
        """
        Calculate common production ratios.

        Returns:
            Dictionary with GOR, water_cut, and liquid_rate
        """
        ratios = {}

        # Gas-Oil Ratio (GOR)
        if self.gas is not None:
            ratios["gor"] = (self.gas / self.oil).replace([np.inf, -np.inf], np.nan)

        # Water Cut
        if self.water is not None:
            total_liquid = self.oil + self.water
            ratios["water_cut"] = (self.water / total_liquid * 100).fillna(0)

        # Total Liquid Rate
        if self.water is not None:
            ratios["liquid_rate"] = self.oil + self.water

        return ratios


class MultiPhaseForecaster:
    """
    Multi-phase production forecasting using coupled models.

    This class addresses the limitation of traditional DCA which forecasts
    only one phase at a time. By coupling oil, gas, and water forecasts,
    we can:
    1. Maintain physical relationships between phases
    2. Improve accuracy through shared information
    3. Ensure consistency across forecasts

    Example:
        >>> forecaster = MultiPhaseForecaster()
        >>> data = MultiPhaseData(oil=oil_series, gas=gas_series, water=water_series)
        >>> forecasts = forecaster.forecast(data, horizon=24, model='arps')
        >>> print(forecasts['oil'].tail())
    """

    def __init__(self):
        """Initialize multi-phase forecaster."""
        self.fitted_models = {}
        self.history = None

    def forecast(
        self,
        data: MultiPhaseData,
        horizon: int = 12,
        model: str = "arps",
        kind: Optional[str] = "hyperbolic",
        enforce_ratios: bool = True,
        **kwargs,
    ) -> dict[str, pd.Series]:
        """
        Generate multi-phase production forecast.

        Args:
            data: Multi-phase production data
            horizon: Number of periods to forecast
            model: Forecasting model ('arps', 'arima', 'prophet')
            kind: Arps model type if applicable
            enforce_ratios: Maintain physical relationships between phases
            **kwargs: Additional model parameters

        Returns:
            Dictionary with forecasts for each phase

        Example:
            >>> data = MultiPhaseData(oil=oil_series, gas=gas_series)
            >>> forecasts = forecaster.forecast(data, horizon=24)
            >>> oil_forecast = forecasts['oil']
            >>> gas_forecast = forecasts['gas']
        """
        from . import dca  # Import here to avoid circular dependency

        forecasts = {}

        # Forecast oil (primary phase)
        oil_forecast = dca.forecast(
            data.oil, model=model, kind=kind, horizon=horizon, **kwargs
        )
        forecasts["oil"] = oil_forecast

        # Forecast gas
        if data.gas is not None:
            if enforce_ratios:
                # Use GOR to derive gas from oil forecast
                gor = (data.gas / data.oil).replace([np.inf, -np.inf], np.nan)
                avg_gor = gor.tail(6).mean()  # Use recent average GOR

                # Apply GOR to oil forecast
                gas_forecast = oil_forecast * avg_gor
                gas_forecast.name = "gas"
                forecasts["gas"] = gas_forecast
            else:
                # Independent gas forecast
                gas_forecast = dca.forecast(
                    data.gas, model=model, kind=kind, horizon=horizon, **kwargs
                )
                forecasts["gas"] = gas_forecast

        # Forecast water
        if data.water is not None:
            if enforce_ratios:
                # Use water cut trend to derive water from oil forecast
                total_liquid = data.oil + data.water
                water_cut = (data.water / total_liquid * 100).fillna(0)

                # Fit trend to water cut (typically increasing)
                from scipy.stats import linregress

                x = np.arange(len(water_cut))
                slope, intercept, _, _, _ = linregress(x, water_cut.values)

                # Project water cut
                future_x = np.arange(len(data.oil), len(data.oil) + horizon)
                future_water_cut = slope * future_x + intercept
                future_water_cut = np.clip(future_water_cut, 0, 100)

                # Calculate water from oil forecast and projected water cut
                oil_forecast_only = oil_forecast.iloc[len(data.oil) :]
                water_forecast_values = (
                    oil_forecast_only.values
                    * future_water_cut
                    / (100 - future_water_cut)
                )

                water_forecast = pd.Series(
                    np.concatenate([data.water.values, water_forecast_values]),
                    index=oil_forecast.index,
                    name="water",
                )
                forecasts["water"] = water_forecast
            else:
                # Independent water forecast
                water_forecast = dca.forecast(
                    data.water, model=model, kind=kind, horizon=horizon, **kwargs
                )
                forecasts["water"] = water_forecast

        self.history = data
        return forecasts

    def evaluate(
        self, data: MultiPhaseData, forecasts: dict[str, pd.Series]
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate multi-phase forecast accuracy.

        Args:
            data: Actual multi-phase production data
            forecasts: Forecasted values for each phase

        Returns:
            Dictionary of metrics for each phase

        Example:
            >>> metrics = forecaster.evaluate(data, forecasts)
            >>> print(f"Oil RMSE: {metrics['oil']['rmse']:.2f}")
            >>> print(f"Gas RMSE: {metrics['gas']['rmse']:.2f}")
        """
        from . import dca

        metrics = {}

        for phase in data.phases:
            if phase in forecasts:
                actual = getattr(data, phase)
                forecast = forecasts[phase]
                metrics[phase] = dca.evaluate(actual, forecast)

        return metrics

    def calculate_consistency_metrics(
        self, forecasts: dict[str, pd.Series]
    ) -> dict[str, float]:
        """
        Calculate consistency metrics between phases.

        Checks if forecasted ratios (GOR, water cut) are physically reasonable.

        Args:
            forecasts: Forecasted values for each phase

        Returns:
            Dictionary with consistency metrics

        Example:
            >>> consistency = forecaster.calculate_consistency_metrics(forecasts)
            >>> print(f"GOR stability: {consistency['gor_stability']:.2f}")
        """
        metrics = {}

        # GOR stability
        if "oil" in forecasts and "gas" in forecasts:
            gor = forecasts["gas"] / forecasts["oil"]
            gor_std = gor.std()
            gor_mean = gor.mean()
            metrics["gor_stability"] = 1 - (gor_std / gor_mean)  # Higher is more stable
            metrics["avg_gor"] = gor_mean

        # Water cut trend
        if "oil" in forecasts and "water" in forecasts:
            total_liquid = forecasts["oil"] + forecasts["water"]
            water_cut = forecasts["water"] / total_liquid * 100

            # Check if water cut is monotonically increasing (expected behavior)
            is_increasing = (water_cut.diff().dropna() >= 0).mean()
            metrics["water_cut_monotonic"] = is_increasing
            metrics["final_water_cut"] = water_cut.iloc[-1]

        return metrics


def create_multiphase_data_from_dataframe(
    df: pd.DataFrame,
    oil_column: str = "Oil",
    gas_column: Optional[str] = "Gas",
    water_column: Optional[str] = "Wtr",
    date_column: str = "date",
    well_id_column: Optional[str] = None,
) -> MultiPhaseData:
    """
    Create MultiPhaseData from a DataFrame.

    Convenience function to convert standard production DataFrames
    to MultiPhaseData objects.

    Args:
        df: DataFrame with production data
        oil_column: Name of oil production column
        gas_column: Name of gas production column (optional)
        water_column: Name of water production column (optional)
        date_column: Name of date column
        well_id_column: Name of well identifier column (optional)

    Returns:
        MultiPhaseData object

    Example:
        >>> df = pd.read_csv('production.csv')
        >>> data = create_multiphase_data_from_dataframe(df)
        >>> forecaster = MultiPhaseForecaster()
        >>> forecasts = forecaster.forecast(data, horizon=12)
    """
    # Ensure date column is datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

    # Extract series
    oil = df[oil_column]
    gas = df[gas_column] if gas_column and gas_column in df.columns else None
    water = df[water_column] if water_column and water_column in df.columns else None

    # Get well ID if available
    well_id = None
    if well_id_column and well_id_column in df.columns:
        well_id = df[well_id_column].iloc[0]

    return MultiPhaseData(
        oil=oil, gas=gas, water=water, dates=df.index, well_id=well_id
    )
