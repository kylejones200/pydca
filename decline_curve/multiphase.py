"""Multi-phase production forecasting module.

This module enables simultaneous forecasting of oil, gas, and water production,
addressing a key limitation of traditional decline curve analysis which typically
handles only one phase at a time.

Based on research showing that coupled multi-target prediction yields more accurate
and consistent forecasts than independent single-phase models.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

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

    def calculate_phase_correlations(self) -> dict[str, float]:
        """
        Calculate correlation metrics between phases.

        Measures how strongly phases are related, which is important
        for multi-target prediction models.

        Returns:
            Dictionary with correlation coefficients between phases

        Example:
            >>> data = MultiPhaseData(oil=oil_series, gas=gas_series, water=water_series)
            >>> correlations = data.calculate_phase_correlations()
            >>> logger.info(f"Oil-Gas correlation: {correlations['oil_gas']:.3f}")
        """
        correlations = {}

        # Oil-Gas correlation
        if self.gas is not None:
            valid_mask = (self.oil > 0) & (self.gas > 0)
            if valid_mask.sum() > 1:
                oil_vals = self.oil[valid_mask].values
                gas_vals = self.gas[valid_mask].values
                correlations["oil_gas"] = float(np.corrcoef(oil_vals, gas_vals)[0, 1])

        # Oil-Water correlation
        if self.water is not None:
            valid_mask = (self.oil > 0) & (self.water >= 0)
            if valid_mask.sum() > 1:
                oil_vals = self.oil[valid_mask].values
                water_vals = self.water[valid_mask].values
                correlations["oil_water"] = float(
                    np.corrcoef(oil_vals, water_vals)[0, 1]
                )

        # Gas-Water correlation
        if self.gas is not None and self.water is not None:
            valid_mask = (self.gas > 0) & (self.water >= 0)
            if valid_mask.sum() > 1:
                gas_vals = self.gas[valid_mask].values
                water_vals = self.water[valid_mask].values
                correlations["gas_water"] = float(
                    np.corrcoef(gas_vals, water_vals)[0, 1]
                )

        return correlations


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

    def __init__(self, shared_model: bool = False):
        """
        Initialize multi-phase forecaster.

        Args:
            shared_model: If True, use shared model parameters across phases
                         for better consistency. If False, fit independent models.
        """
        self.fitted_models = {}
        self.history = None
        self.shared_model = shared_model

    def forecast(
        self,
        data: MultiPhaseData,
        horizon: int = 12,
        model: Literal["arps", "timesfm", "chronos", "arima"] = "arps",
        kind: Optional[Literal["exponential", "harmonic", "hyperbolic"]] = "hyperbolic",
        enforce_ratios: bool = True,
        **kwargs,
    ) -> dict[str, pd.Series]:
        """
        Generate multi-phase production forecast.

        Args:
            data: Multi-phase production data
            horizon: Number of periods to forecast
            model: Forecasting model ('arps', 'arima', 'timesfm', 'chronos')
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
        from .models import fit_arps, predict_arps

        forecasts = {}

        # If using shared model architecture, fit once and apply to all phases
        if self.shared_model and model == "arps" and kind is not None:
            # Fit shared Arps model to primary phase (oil)
            t = np.arange(len(data.oil))
            params = fit_arps(t, data.oil.values, kind=kind)

            # Generate forecast for oil
            forecast_t = np.arange(len(data.oil) + horizon)
            oil_forecast_values = predict_arps(forecast_t, params)
            oil_forecast = pd.Series(
                oil_forecast_values,
                index=pd.date_range(
                    start=data.dates[0], periods=len(forecast_t), freq="MS"
                ),
                name="oil",
            )
            forecasts["oil"] = oil_forecast

            # Apply shared decline pattern to other phases using ratios
            if data.gas is not None and enforce_ratios:
                gor = (data.gas / data.oil).replace([np.inf, -np.inf], np.nan)
                avg_gor = gor.tail(6).mean()
                gas_forecast = oil_forecast * avg_gor
                gas_forecast.name = "gas"
                forecasts["gas"] = gas_forecast

            if data.water is not None and enforce_ratios:
                total_liquid = data.oil + data.water
                water_cut = (data.water / total_liquid * 100).fillna(0)
                from scipy.stats import linregress

                x = np.arange(len(water_cut))
                slope, intercept, _, _, _ = linregress(x, water_cut.values)
                future_x = np.arange(len(data.oil), len(data.oil) + horizon)
                future_water_cut = np.clip(slope * future_x + intercept, 0, 100)
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

            self.history = data
            return forecasts

        # Standard forecasting (independent or ratio-based)
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
            Dictionary of metrics for each phase, including:
            - Individual phase metrics (rmse, mae, smape, r2)
            - Cross-phase consistency metrics
            - Overall multi-phase accuracy

        Example:
            >>> metrics = forecaster.evaluate(data, forecasts)
            >>> print(f"Oil RMSE: {metrics['oil']['rmse']:.2f}")
            >>> print(f"Gas RMSE: {metrics['gas']['rmse']:.2f}")
            >>> print(f"Overall accuracy: {metrics['overall']['rmse']:.2f}")
        """
        from . import dca

        metrics = {}

        # Individual phase metrics
        for phase in data.phases:
            if phase in forecasts:
                actual = getattr(data, phase)
                forecast = forecasts[phase]
                metrics[phase] = dca.evaluate(actual, forecast)

        # Cross-phase consistency checks
        consistency = self._calculate_cross_phase_consistency(data, forecasts)
        metrics["consistency"] = consistency

        # Overall multi-phase accuracy (weighted average)
        if len(metrics) > 1:
            phase_weights = {}
            total_production = 0
            for phase in data.phases:
                if phase in forecasts:
                    phase_prod = getattr(data, phase).sum()
                    phase_weights[phase] = phase_prod
                    total_production += phase_prod

            if total_production > 0:
                overall_rmse = sum(
                    metrics[phase]["rmse"] * phase_weights[phase] / total_production
                    for phase in data.phases
                    if phase in forecasts
                )
                overall_mae = sum(
                    metrics[phase]["mae"] * phase_weights[phase] / total_production
                    for phase in data.phases
                    if phase in forecasts
                )
                overall_smape = sum(
                    metrics[phase]["smape"] * phase_weights[phase] / total_production
                    for phase in data.phases
                    if phase in forecasts
                )
                metrics["overall"] = {
                    "rmse": overall_rmse,
                    "mae": overall_mae,
                    "smape": overall_smape,
                }

        return metrics

    def _calculate_cross_phase_consistency(
        self, data: MultiPhaseData, forecasts: dict[str, pd.Series]
    ) -> dict[str, float]:
        """
        Calculate cross-phase consistency metrics.

        Checks if forecasted relationships between phases match historical patterns.

        Args:
            data: Historical multi-phase production data
            forecasts: Forecasted values for each phase

        Returns:
            Dictionary with consistency metrics
        """
        consistency = {}

        # GOR consistency (compare forecast GOR to historical GOR)
        if "oil" in forecasts and "gas" in forecasts and data.gas is not None:
            # Historical GOR
            hist_gor = (data.gas / data.oil).replace([np.inf, -np.inf], np.nan)
            hist_gor_mean = hist_gor.tail(6).mean()

            # Forecast GOR
            forecast_gor = forecasts["gas"] / forecasts["oil"]
            forecast_gor_mean = forecast_gor.tail(6).mean()

            # Consistency score (1 - normalized difference)
            if hist_gor_mean > 0:
                gor_diff = abs(forecast_gor_mean - hist_gor_mean) / hist_gor_mean
                consistency["gor_consistency"] = max(0, 1 - gor_diff)
            else:
                consistency["gor_consistency"] = 0.0

        # Water cut consistency
        if "oil" in forecasts and "water" in forecasts and data.water is not None:
            # Historical water cut trend
            hist_total = data.oil + data.water
            hist_water_cut = (data.water / hist_total * 100).fillna(0)
            hist_trend = hist_water_cut.tail(6).mean()

            # Forecast water cut
            forecast_total = forecasts["oil"] + forecasts["water"]
            forecast_water_cut = (forecasts["water"] / forecast_total * 100).fillna(0)
            forecast_trend = forecast_water_cut.tail(6).mean()

            # Consistency score
            if hist_trend > 0:
                wc_diff = abs(forecast_trend - hist_trend) / hist_trend
                consistency["water_cut_consistency"] = max(0, 1 - wc_diff)
            else:
                consistency["water_cut_consistency"] = 0.0

        # Phase relationship preservation
        if len(data.phases) > 1:
            # Check if relative phase magnitudes are preserved
            hist_ratios = {}
            forecast_ratios = {}

            if data.gas is not None:
                hist_ratios["gas_oil"] = (data.gas / data.oil).mean()
                forecast_ratios["gas_oil"] = (
                    forecasts["gas"] / forecasts["oil"]
                ).mean()

            if data.water is not None:
                hist_ratios["water_oil"] = (data.water / data.oil).mean()
                forecast_ratios["water_oil"] = (
                    forecasts["water"] / forecasts["oil"]
                ).mean()

            # Calculate overall relationship preservation
            if hist_ratios:
                ratio_errors = []
                for key in hist_ratios:
                    if hist_ratios[key] > 0:
                        error = (
                            abs(forecast_ratios[key] - hist_ratios[key])
                            / hist_ratios[key]
                        )
                        ratio_errors.append(error)
                if ratio_errors:
                    consistency["phase_relationship_preservation"] = max(
                        0, 1 - np.mean(ratio_errors)
                    )

        return consistency

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


def load_multiphase_data(
    source: Union[str, pd.DataFrame],
    oil_column: str = "Oil",
    gas_column: Optional[str] = "Gas",
    water_column: Optional[str] = "Wtr",
    date_column: str = "date",
    well_id_column: Optional[str] = None,
    **kwargs,
) -> Union[MultiPhaseData, dict[str, MultiPhaseData]]:
    """
    Unified data loader for multi-phase production data.

    Loads production data from various sources (CSV, Excel, DataFrame)
    and converts to MultiPhaseData format. Supports both single-well
    and multi-well datasets.

    Args:
        source: File path (CSV/Excel) or DataFrame
        oil_column: Name of oil production column
        gas_column: Name of gas production column (optional)
        water_column: Name of water production column (optional)
        date_column: Name of date column
        well_id_column: Name of well identifier column (for multi-well data)
        **kwargs: Additional arguments passed to pd.read_csv or pd.read_excel

    Returns:
        MultiPhaseData object (single well) or dict mapping well_id to MultiPhaseData

    Example:
        >>> # Single well
        >>> data = load_multiphase_data('well_production.csv')
        >>>
        >>> # Multiple wells
        >>> multi_well_data = load_multiphase_data(
        ...     'field_production.csv',
        ...     well_id_column='well_id'
        ... )
        >>> for well_id, data in multi_well_data.items():
        ...     print(f"Well {well_id}: {len(data)} months")
    """
    # Load DataFrame if source is a string
    if isinstance(source, str):
        if source.endswith((".xlsx", ".xls")):
            df = pd.read_excel(source, **kwargs)
        else:
            df = pd.read_csv(source, **kwargs)
    else:
        df = source.copy()

    # Multi-well dataset
    if well_id_column and well_id_column in df.columns:
        result = {}
        for well_id in df[well_id_column].unique():
            well_df = df[df[well_id_column] == well_id].copy()
            result[well_id] = create_multiphase_data_from_dataframe(
                well_df,
                oil_column=oil_column,
                gas_column=gas_column,
                water_column=water_column,
                date_column=date_column,
                well_id_column=None,  # Already filtered by well_id
            )
        return result

    # Single well dataset
    return create_multiphase_data_from_dataframe(
        df,
        oil_column=oil_column,
        gas_column=gas_column,
        water_column=water_column,
        date_column=date_column,
        well_id_column=well_id_column,
    )


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
