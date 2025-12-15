"""Main API for decline curve analysis and forecasting."""

from typing import Any, Literal, Optional

import pandas as pd

from .economics import economic_metrics
from .evaluate import mae, rmse, smape
from .forecast import Forecaster
from .logging_config import get_logger
from .models import ArpsParams
from .plot import plot_forecast
from .reserves import forecast_and_reserves
from .sensitivity import run_sensitivity

logger = get_logger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def forecast(
    series: pd.Series,
    model: Literal[
        "arps",
        "timesfm",
        "chronos",
        "arima",
        "deepar",
        "tft",
        "ensemble",
        "exponential_smoothing",
        "moving_average",
        "linear_trend",
        "holt_winters",
    ] = "arps",
    kind: Optional[Literal["exponential", "harmonic", "hyperbolic"]] = "hyperbolic",
    horizon: int = 12,
    verbose: bool = False,
    # DeepAR parameters
    deepar_model: Optional[Any] = None,
    tft_model: Optional[Any] = None,
    production_data: Optional[pd.DataFrame] = None,
    well_id: Optional[str] = None,
    quantiles: Optional[list[float]] = [0.1, 0.5, 0.9],
    return_interpretation: bool = False,
    # Ensemble parameters
    ensemble_models: Optional[list[str]] = None,
    ensemble_weights: Optional[Any] = None,
    ensemble_method: Literal["weighted", "confidence", "stacking"] = "weighted",
    lstm_model: Optional[Any] = None,
) -> pd.Series:
    """Generate production forecast using specified model.

    Args:
        series: Historical production time series.
        model: Forecasting model to use.
            Options: 'arps', 'timesfm', 'chronos', 'arima', 'deepar', 'tft',
            'ensemble'.
        kind: Arps decline type ('exponential', 'harmonic', 'hyperbolic').
        horizon: Number of periods to forecast.
        verbose: Print forecast details.
        deepar_model: Pre-trained DeepAR model (required if model='deepar').
        tft_model: Pre-trained TFT model (required if model='tft').
        production_data: Production DataFrame (required for deepar/tft/ensemble
            with ML models).
        well_id: Well identifier (required for deepar/tft/ensemble with ML
            models).
        quantiles: Quantiles for DeepAR probabilistic forecasts
            (default [0.1, 0.5, 0.9]).
        return_interpretation: If True, return attention weights for TFT
            (default False).
        ensemble_models: List of models for ensemble
            (default ['arps', 'lstm', 'deepar']).
        ensemble_weights: Custom weights for ensemble (EnsembleWeights object).
        ensemble_method: Ensemble combination method
            ('weighted', 'confidence', 'stacking').
        lstm_model: Pre-trained LSTM model (for ensemble).

        Returns:
            Forecasted production series (or dict with quantiles for DeepAR
            probabilistic mode, or tuple (forecast, interpretation) for TFT
            with return_interpretation=True).

    Example:
        >>> # Arps forecast
        >>> forecast = dca.forecast(series, model='arps', horizon=12)
        >>>
        >>> # DeepAR probabilistic forecast (P50)
        >>> deepar_forecast = dca.forecast(
        ...     series, model='deepar', deepar_model=trained_model,
        ...     production_data=df, well_id='WELL_001', quantiles=[0.5]
        ... )
        >>>
        >>> # Ensemble forecast
        >>> ensemble = dca.forecast(
        ...     series, model='ensemble', ensemble_models=['arps', 'arima'],
        ...     ensemble_method='weighted'
        ... )
    """
    # Handle DeepAR model
    if model == "deepar":
        try:
            # DeepARForecaster is used via deepar_model parameter
            pass

            if deepar_model is None:
                raise ValueError(
                    "deepar_model required for DeepAR forecasting. "
                    "Train using DeepARForecaster.fit() first."
                )
            if production_data is None or well_id is None:
                raise ValueError(
                    "production_data and well_id required for DeepAR forecasting"
                )

            # Get P50 (median) forecast by default
            if quantiles is None:
                quantiles = [0.5]

            forecasts = deepar_model.predict_quantiles(
                well_id=well_id,
                production_data=production_data,
                quantiles=quantiles,
                horizon=horizon,
                n_samples=500,  # Reasonable default
            )

            # Return P50 if single quantile, otherwise return dict
            if quantiles == [0.5] or 0.5 in quantiles:
                phase = list(forecasts.keys())[0]
                return forecasts[phase].get(
                    "q50", forecasts[phase][list(forecasts[phase].keys())[0]]
                )
            else:
                # Return all quantiles as Series with MultiIndex or dict
                phase = list(forecasts.keys())[0]
                return forecasts[phase].get(
                    "q50", forecasts[phase][list(forecasts[phase].keys())[0]]
                )

        except ImportError:
            raise ImportError(
                "DeepAR requires PyTorch. Install with: pip install torch"
            )

    # Handle TFT model
    elif model == "tft":
        try:
            # TFTForecaster is used via tft_model parameter
            pass

            if tft_model is None:
                raise ValueError(
                    "tft_model required for TFT forecasting. "
                    "Train using TFTForecaster.fit() first."
                )
            if production_data is None or well_id is None:
                raise ValueError(
                    "production_data and well_id required for TFT forecasting"
                )

            # Generate forecast with optional interpretation
            result = tft_model.predict(
                well_id=well_id,
                production_data=production_data,
                horizon=horizon,
                return_interpretation=return_interpretation,
            )

            if return_interpretation:
                # Return tuple (forecast, interpretation)
                forecasts, interpretation = result
                # Return first phase as series for compatibility
                phase = list(forecasts.keys())[0]
                if verbose:
                    logger.debug(
                        f"TFT forecast with interpretation, horizon: {horizon}"
                    )
                return forecasts[phase], interpretation
            else:
                # Return first phase as series
                phase = list(result.keys())[0]
                if verbose:
                    logger.debug(f"TFT forecast, horizon: {horizon}")
                return result[phase]

        except ImportError:
            raise ImportError("TFT requires PyTorch. Install with: pip install torch")

    # Handle ensemble model
    elif model == "ensemble":
        try:
            from .ensemble import EnsembleForecaster

            forecaster = EnsembleForecaster(
                models=ensemble_models or ["arps", "arima"],
                weights=ensemble_weights,
                method=ensemble_method,
            )
            result = forecaster.forecast(
                series=series,
                horizon=horizon,
                arps_kind=kind or "hyperbolic",
                lstm_model=lstm_model,
                deepar_model=deepar_model,
                production_data=production_data,
                well_id=well_id,
                quantiles=quantiles,
                verbose=verbose,
            )
            if verbose:
                logger.debug(
                    f"Ensemble forecast ({ensemble_method}), horizon: {horizon}"
                )
            return result

        except ImportError:
            raise ImportError(
                "Ensemble forecasting requires PyTorch for ML models. "
                "Install with: pip install torch"
            )

    # Standard models (Arps, ARIMA, TimesFM, Chronos, statistical methods)
    else:
        fc = Forecaster(series)
        result = fc.forecast(model=model, kind=kind, horizon=horizon)
        if verbose:
            logger.debug(f"Forecast model: {model}, horizon: {horizon}")
        return result


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Evaluate forecast accuracy metrics.

    Args:
        y_true: Actual production values.
        y_pred: Predicted production values.

    Returns:
        Dictionary with RMSE, MAE, and SMAPE metrics.
    """
    common = y_true.index.intersection(y_pred.index)
    yt = y_true.loc[common]
    yp = y_pred.loc[common]
    return {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "smape": smape(yt, yp),
    }


def plot(
    y: pd.Series,
    yhat: pd.Series,
    title: str = "Forecast",
    filename: Optional[str] = None,
):
    """Plot forecast visualization.

    Args:
        y: Historical production series.
        yhat: Forecasted production series.
        title: Plot title.
        filename: Optional filename to save plot.
    """
    plot_forecast(y, yhat, title, filename)


def _benchmark_single_well(
    wid, df, well_col, date_col, value_col, model, kind, horizon, kwargs=None
):
    """Process a single well for benchmarking (used for parallel execution)."""
    try:
        wdf = df[df[well_col] == wid].copy()
        wdf = wdf[[date_col, value_col]].dropna()
        wdf[date_col] = pd.to_datetime(wdf[date_col])
        wdf = wdf.set_index(date_col).asfreq("MS")
        if len(wdf) < 24:
            return None

        y = wdf[value_col]
        forecast_kwargs = kwargs or {}
        yhat = forecast(y, model=model, kind=kind, horizon=horizon, **forecast_kwargs)
        metrics = evaluate(y, yhat)
        metrics[well_col] = wid
        return metrics
    except Exception as e:
        return {"error": str(e), well_col: wid}


def benchmark(
    df: pd.DataFrame,
    model: Literal[
        "arps",
        "timesfm",
        "chronos",
        "arima",
        "exponential_smoothing",
        "moving_average",
        "linear_trend",
        "holt_winters",
    ] = "arps",
    kind: Optional[str] = "hyperbolic",
    horizon: int = 12,
    well_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil_bbl",
    top_n: int = 10,
    verbose: bool = False,
    n_jobs: int = -1,  # -1 uses all available cores
    **kwargs,
) -> pd.DataFrame:
    """
    Benchmark forecasting models across multiple wells.

    Args:
        df: DataFrame with production data
        model: Forecasting model to use
        kind: Arps model type (if applicable)
        horizon: Forecast horizon in months
        well_col: Column name for well identifier
        date_col: Column name for dates
        value_col: Column name for production values
        top_n: Number of wells to process
        verbose: Print progress messages
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)

    Returns:
        DataFrame with metrics for each well
    """
    wells = df[well_col].unique()[:top_n]

    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel execution
        logger.info(
            "Processing wells in parallel",
            extra={"n_wells": len(wells), "n_jobs": n_jobs},
        )

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_benchmark_single_well)(
                wid, df, well_col, date_col, value_col, model, kind, horizon
            )
            for wid in wells
        )

        # Filter out None results and errors
        out = [r for r in results if r is not None and "error" not in r]
        errors = [r for r in results if r is not None and "error" in r]

        if errors:
            logger.warning(
                "Some wells failed during benchmark",
                extra={"successful": len(out), "failed": len(errors)},
            )
        else:
            logger.info(
                "Benchmark completed", extra={"successful": len(out), "failed": 0}
            )
    else:
        # Sequential execution (fallback)
        if not JOBLIB_AVAILABLE:
            logger.warning("joblib not available, running sequentially")

        out = []
        errors = []
        for wid in wells:
            result = _benchmark_single_well(
                wid, df, well_col, date_col, value_col, model, kind, horizon
            )
            if result is not None:
                if "error" not in result:
                    out.append(result)
                    logger.debug(f"Processed well: {wid}")
                else:
                    errors.append(result)
                    logger.warning(
                        f"Well failed: {wid}", extra={"error": result["error"]}
                    )

        if errors:
            logger.warning(
                "Some wells failed during benchmark",
                extra={"successful": len(out), "failed": len(errors)},
            )

    return pd.DataFrame(out)


def sensitivity_analysis(
    param_grid: list[tuple[float, float, float]],
    prices: list[float],
    opex: float,
    discount_rate: float = 0.10,
    t_max: float = 240,
    econ_limit: float = 10.0,
    dt: float = 1.0,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across Arps parameters and oil/gas prices.

    Args:
        param_grid: List of (qi, di, b) tuples to test
        prices: List of oil/gas prices to test
        opex: Operating cost per unit
        discount_rate: Annual discount rate (default 0.10)
        t_max: Time horizon in months (default 240)
        econ_limit: Minimum economic production rate (default 10.0)
        dt: Time step in months (default 1.0)

    Returns:
        DataFrame with sensitivity results including EUR, NPV, and payback
    """
    return run_sensitivity(
        param_grid, prices, opex, discount_rate, t_max, econ_limit, dt
    )


def economics(
    production: pd.Series, price: float, opex: float, discount_rate: float = 0.10
) -> dict:
    """
    Calculate economic metrics from production forecast.

    Args:
        production: Monthly production forecast
        price: Unit price ($/bbl or $/mcf)
        opex: Operating cost per unit
        discount_rate: Annual discount rate (default 0.10)

    Returns:
        Dictionary with NPV, cash flow, and payback period
    """
    return economic_metrics(production.values, price, opex, discount_rate)


def reserves(
    params: ArpsParams, t_max: float = 240, dt: float = 1.0, econ_limit: float = 10.0
) -> dict:
    """
    Generate production forecast and compute EUR (Estimated Ultimate Recovery).

    Args:
        params: Arps decline parameters (qi, di, b)
        t_max: Time horizon in months (default 240)
        dt: Time step in months (default 1.0)
        econ_limit: Minimum economic production rate (default 10.0)

    Returns:
        Dictionary with forecast, time arrays, and EUR
    """
    return forecast_and_reserves(params, t_max, dt, econ_limit)
