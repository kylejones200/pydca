"""Main API for decline curve analysis and forecasting."""

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from .economics import (
    economic_metrics,
    WellEconomics,
    cashflow as _cashflow,
    npv as _npv,
    irr as _irr,
    payout as _payout,
    roi as _roi,
    breakeven_price as _breakeven_price,
)
from .evaluate import mae, rmse, smape
from .forecast import Forecaster
from .logging_config import configure_logging, get_logger
from .models import ArpsParams, fit_arps, predict_arps
from .plot import plot_forecast
from .reserves import forecast_and_reserves
from .runner import FieldScaleRunner, RunnerConfig, WellResult
from .sensitivity import run_sensitivity as _run_sensitivity

try:
    from .config import BatchJobConfig, BenchmarkConfig, SensitivityConfig
except ImportError:
    BatchJobConfig = None
    BenchmarkConfig = None
    SensitivityConfig = None

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
        "material_balance",
        "pressure_decline",
    ] = "arps",
    kind: Optional[str] = "hyperbolic",
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
        kind: Decline type. Arps: 'exponential', 'harmonic', 'hyperbolic'.
            Shale variants: 'duong', 'ple', 'sepd'.
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

    # Handle physics-informed models
    elif model == "material_balance":
        from .physics_informed import material_balance_forecast

        result = material_balance_forecast(
            production_data=series,
            material_balance_params=None,  # Can be passed via kwargs in future
            horizon=horizon,
        )
        if verbose:
            logger.debug(f"Material balance forecast, horizon: {horizon}")
        return result

    elif model == "pressure_decline":
        # For pressure decline, need pressure data
        # If not provided, raise error
        raise ValueError(
            "pressure_decline model requires pressure data. "
            "Use physics_informed.pressure_decline_forecast() directly "
            "with pressure_data parameter."
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
    df: pd.DataFrame | None = None,
    config: str | Path | dict | Any = None,
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

    This function supports both config-driven and direct parameter usage.
    If config is provided, it takes precedence over individual parameters.

    Args:
        df: DataFrame with production data (required if config not provided)
        config: Path to config file, dict, or BenchmarkConfig (optional)
        model: Forecasting model to use (used if config not provided)
        kind: Arps model type (if applicable) (used if config not provided)
        horizon: Forecast horizon in months (used if config not provided)
        well_col: Column name for well identifier (used if config not provided)
        date_col: Column name for dates (used if config not provided)
        value_col: Column name for production values (used if config not provided)
        top_n: Number of wells to process (used if config not provided)
        verbose: Print progress messages
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
            (used if config not provided)

    Returns:
        DataFrame with metrics for each well

    Example:
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.benchmark(config='benchmark_config.toml')
        >>> # Using direct parameters
        >>> results = dca.benchmark(df, model='arps', top_n=10)
    """
    # If config provided, delegate to run_benchmark
    if config is not None:
        result = run_benchmark(config)
        return result["results"]  # Return just the results DataFrame

    # Otherwise use direct parameters
    if df is None:
        raise ValueError("Either df or config must be provided")

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
    param_grid: list[tuple[float, float, float]] | dict | None = None,
    prices: list[float] | None = None,
    opex: float | None = None,
    discount_rate: float = 0.10,
    t_max: float = 240,
    econ_limit: float = 10.0,
    dt: float = 1.0,
    config: str | Path | dict | Any = None,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across Arps parameters and oil/gas prices.

    This function supports both config-driven and direct parameter usage.
    If config is provided, it takes precedence over individual parameters.

    Args:
        param_grid: List of (qi, di, b) tuples to test or dict with ranges
            (used if config not provided)
        prices: List of oil/gas prices to test (used if config not provided)
        opex: Operating cost per unit (used if config not provided)
        discount_rate: Annual discount rate (default 0.10) (used if config not provided)
        t_max: Time horizon in months (default 240) (used if config not provided)
        econ_limit: Minimum economic production rate (default 10.0)
            (used if config not provided)
        dt: Time step in months (default 1.0) (used if config not provided)
        config: Path to config file, dict, or SensitivityConfig (optional)

    Returns:
        DataFrame with sensitivity results including EUR, NPV, and payback

    Example:
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.sensitivity_analysis(config='sensitivity_config.toml')
        >>> # Using direct parameters
        >>> results = dca.sensitivity_analysis(
        ...     param_grid=[(1000, 0.1, 0.5)], prices=[70, 80]
        ... )
    """
    # If config provided, delegate to run_sensitivity_analysis
    if config is not None:
        result = run_sensitivity_analysis(config)
        return result["results"]  # Return just the results DataFrame

    # Otherwise use direct parameters
    if param_grid is None or prices is None or opex is None:
        raise ValueError(
            "Either config or all of param_grid, prices, and opex must be provided"
        )

    return _run_sensitivity(
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


def well_economics(
    production: pd.Series,
    econ: WellEconomics,
) -> dict:
    """Full-cycle well economics: royalties, taxes, CAPEX, IRR, breakeven.

    This is the primary economic evaluation entry point. For the simpler
    legacy interface (price/opex only, no CAPEX), use :func:`economics`.

    Args:
        production: Monthly production forecast (BOE/month).
        econ: :class:`~decline_curve.economics.WellEconomics` with all cost
            and revenue parameters.

    Returns:
        Dict with keys: ``npv``, ``irr``, ``payout_month``, ``roi``,
        ``breakeven_price``, and ``cashflow``
        (:class:`~decline_curve.economics.CashflowResult`).

    Example:
        >>> from decline_curve import dca, WellEconomics
        >>> econ = WellEconomics(capex=5_000_000, price=70.0, opex_variable=8.0)
        >>> result = dca.well_economics(forecast, econ)
        >>> print(f"NPV: ${result['npv']:,.0f}  IRR: {result['irr']:.1%}")
    """
    q = production.values
    cf = _cashflow(q, econ)
    return {
        "npv": _npv(cf),
        "irr": _irr(cf),
        "payout_month": _payout(cf),
        "roi": _roi(cf),
        "breakeven_price": _breakeven_price(q, econ),
        "cashflow": cf,
    }


def recommend_model(
    series: pd.Series,
    pressure: "pd.Series | None" = None,
    drive_mechanism: str | None = None,
) -> str:
    """Recommend a decline model based on production data characteristics.

    Analyzes the decline rate behavior and b-factor to recommend the best-fit
    Arps or shale variant model. Extended from ressmith's
    ``identify_decline_type_from_physics`` to include shale-specific models.

    Args:
        series: Historical production time series (monthly, DatetimeIndex).
        pressure: Optional bottomhole/reservoir pressure data (unused currently
            — reserved for future physics-informed routing).
        drive_mechanism: Optional drive mechanism hint: ``'solution_gas'``,
            ``'water_drive'``, ``'gas_cap'``, ``'compaction'``, ``'shale'``.
            When supplied this can override the data-driven recommendation.

    Returns:
        One of: ``'exponential'``, ``'harmonic'``, ``'hyperbolic'``,
        ``'modified_hyperbolic'``, ``'duong'``.

    Example:
        >>> model = dca.recommend_model(series)
        >>> forecast = dca.forecast(series, model="arps", kind=model, horizon=120)
    """
    import numpy as np

    q = series.dropna().values
    if len(q) < 6:
        return "hyperbolic"

    # --- Drive mechanism shortcuts ---
    if drive_mechanism == "solution_gas":
        return "hyperbolic"
    if drive_mechanism in ("water_drive", "compaction"):
        return "exponential"
    if drive_mechanism == "shale":
        return "modified_hyperbolic"

    # --- Compute instantaneous decline rates ---
    q_safe = np.where(q > 0, q, np.nan)
    # D[i] = (q[i-1] - q[i]) / q[i-1]  (fractional per period)
    D = np.where(
        (q_safe[:-1] > 0) & (q_safe[1:] > 0),
        (q_safe[:-1] - q_safe[1:]) / q_safe[:-1],
        np.nan,
    )
    D = D[~np.isnan(D)]
    D = D[D > 0]  # only meaningful decline periods

    if len(D) < 3:
        return "hyperbolic"

    cv_D = float(np.std(D) / np.mean(D)) if np.mean(D) > 0 else 0.0

    # --- Early vs late decline ---
    mid = max(1, len(D) // 2)
    early_D = float(np.mean(D[:mid]))
    late_D = float(np.mean(D[mid:]))

    # --- Estimate b from log-log slope (simplified Fetkovich) ---
    # b = (d/dt)(1/D) ≈ ΔD / (D_early * D_late) for small ΔD
    b_estimate = 0.0
    if len(D) >= 4 and late_D > 0 and early_D > 0:
        b_estimate = float((early_D - late_D) / (early_D * late_D * len(D)))
        b_estimate = max(0.0, b_estimate)

    # --- Routing logic ---
    # Shale / tight: rapid early hyperbolic with very high initial rate,
    # then slow tail — best handled by Modified Hyperbolic or Duong.
    # Duong signature: CV in early decline is high (non-monotone start),
    #   and q[0] >> q[6] strongly (sharp frac-hit dominated peak).
    qi = float(q[0])
    q6 = float(q[min(6, len(q) - 1)])
    fast_early_drop = (qi > 0) and (q6 / qi < 0.35)  # >65% drop in 6 months

    if fast_early_drop and cv_D > 0.35:
        # Highly irregular early decline — Duong transient model
        return "duong"

    if b_estimate > 1.2 and early_D > 0.08:
        # High b-factor with steep initial decline → Modified Hyperbolic (SEC std)
        return "modified_hyperbolic"

    if cv_D < 0.10:
        # Near-constant decline rate → exponential
        return "exponential"

    if late_D < 0.50 * early_D:
        # Decline rate falling significantly over time → harmonic/hyperbolic
        if b_estimate > 0.8:
            return "hyperbolic"
        return "harmonic"

    return "hyperbolic"


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


def single_well(
    series: pd.Series,
    model: Literal["arps", "arima", "timesfm", "chronos"] = "arps",
    kind: Optional[str] = "hyperbolic",
    horizon: int = 12,
    return_params: bool = False,
) -> pd.Series | tuple[pd.Series, dict]:
    """
    Analyze a single well with Arps decline curve.

    This is the main entry point for single well analysis. It fits a decline
    curve to historical production and generates a forecast.

    Args:
        series: Historical production time series with DatetimeIndex
        model: Forecasting model ('arps', 'arima', 'timesfm', 'chronos')
        kind: Arps decline type ('exponential', 'harmonic', 'hyperbolic')
        horizon: Number of periods to forecast
        return_params: If True, also return fitted parameters

    Returns:
        Forecasted production series. If return_params=True, returns tuple
        (forecast, params_dict) where params_dict contains fitted parameters.

    Example:
        >>> import pandas as pd
        >>> import decline_curve as dca
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        >>> production = pd.Series([1000, 950, 900, ...], index=dates)
        >>> forecast = dca.single_well(production, model='arps', kind='hyperbolic')
        >>> forecast, params = dca.single_well(production, return_params=True)
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have DatetimeIndex")

    if model == "arps":
        # Use Arps directly for cleaner API
        import numpy as np

        t = np.arange(len(series))
        q = series.values
        params = fit_arps(t, q, kind=kind or "hyperbolic")

        # Generate forecast
        full_t = np.arange(len(series) + horizon)
        yhat = predict_arps(full_t, params)
        idx = pd.date_range(
            series.index[0], periods=len(yhat), freq=series.index.freq or "MS"
        )
        forecast_series = pd.Series(yhat, index=idx, name="forecast")

        if return_params:
            params_dict = {
                "qi": params.qi,
                "di": params.di,
                "b": params.b,
                "kind": kind or "hyperbolic",
            }
            return forecast_series, params_dict
        return forecast_series
    else:
        # Use existing forecast function for other models
        result = forecast(series, model=model, kind=kind, horizon=horizon)
        if return_params:
            # For non-Arps models, params may not be available
            return result, {}
        return result


def batch_jobs(
    df: pd.DataFrame | None = None,
    config: str | Path | dict | Any = None,
    well_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil_bbl",
    model: Literal["arps", "arima"] = "arps",
    kind: Optional[str] = "hyperbolic",
    horizon: int = 12,
    n_jobs: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run batch decline curve analysis across multiple wells.

    This function supports both config-driven and direct parameter usage.
    If config is provided, it takes precedence over individual parameters.

    Args:
        df: DataFrame with production data for multiple wells
            (required if config not provided)
        config: Path to config file, dict, or BatchJobConfig (optional)
        well_col: Column name for well identifier (used if config not provided)
        date_col: Column name for dates (used if config not provided)
        value_col: Column name for production values (used if config not provided)
        model: Forecasting model ('arps' or 'arima') (used if config not provided)
        kind: Arps decline type (if model='arps') (used if config not provided)
        horizon: Forecast horizon in months (used if config not provided)
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
            (used if config not provided)
        verbose: Print progress messages

    Returns:
        DataFrame with one row per well containing:
        - well_id
        - qi, di, b (Arps parameters if model='arps')
        - rmse, mae, smape (evaluation metrics)
        - forecast (as Series in column, if applicable)

    Example:
        >>> import pandas as pd
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.batch_jobs(config='config.toml')
        >>> # Using direct parameters
        >>> results = dca.batch_jobs(df, well_col='well_id', model='arps')
    """
    # If config provided, delegate to run_batch_job
    if config is not None:
        result = run_batch_job(config)
        return result["parameters"]  # Return just the parameters DataFrame

    # Otherwise use direct parameters
    if df is None:
        raise ValueError("Either df or config must be provided")
        raise ValueError(f"Column '{well_col}' not found in DataFrame")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")

    wells = df[well_col].unique()
    results = []

    def process_well(well_id: str) -> dict | None:
        """Process a single well."""
        try:
            well_df = df[df[well_col] == well_id].copy()
            well_df = well_df[[date_col, value_col]].dropna()
            well_df[date_col] = pd.to_datetime(well_df[date_col])
            well_df = well_df.set_index(date_col).sort_index()

            if len(well_df) < 6:  # Minimum data points
                if verbose:
                    logger.warning(
                        f"Well {well_id}: insufficient data ({len(well_df)} points)"
                    )
                return None

            # Ensure regular frequency
            if well_df.index.freq is None:
                well_df = well_df.asfreq("MS", method="ffill")

            series = well_df[value_col]

            # Fit and forecast
            if model == "arps":
                forecast_series, params = single_well(
                    series, model="arps", kind=kind, horizon=horizon, return_params=True
                )
                # Calculate metrics on historical fit
                hist_forecast = forecast_series[: len(series)]
                metrics = evaluate(series, hist_forecast)
                result = {
                    well_col: well_id,
                    "qi": params["qi"],
                    "di": params["di"],
                    "b": params["b"],
                    **metrics,
                }
            else:
                forecast_series = single_well(series, model=model, horizon=horizon)
                hist_forecast = forecast_series[: len(series)]
                metrics = evaluate(series, hist_forecast)
                result = {well_col: well_id, **metrics}

            return result
        except Exception as e:
            if verbose:
                logger.warning(f"Well {well_id} failed: {str(e)}")
            return {well_col: well_id, "error": str(e)}

    # Process wells
    if JOBLIB_AVAILABLE and n_jobs != 1:
        if verbose:
            logger.info(f"Processing {len(wells)} wells in parallel (n_jobs={n_jobs})")
        from joblib import Parallel, delayed

        results_list = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_well)(wid) for wid in wells
        )
        results = [r for r in results_list if r is not None]
    else:
        if not JOBLIB_AVAILABLE and verbose:
            logger.warning("joblib not available, running sequentially")
        for well_id in wells:
            result = process_well(well_id)
            if result is not None:
                results.append(result)
            if verbose and len(results) % 10 == 0:
                logger.info(f"Processed {len(results)}/{len(wells)} wells")

    if verbose:
        logger.info(
            f"Batch processing complete: {len(results)}/{len(wells)} wells successful"
        )

    return pd.DataFrame(results)


def type_curves(
    df: pd.DataFrame,
    grouping_col: Optional[str] = None,
    date_col: str = "date",
    value_col: str = "oil_bbl",
    kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
    method: Literal["mean", "median", "p50"] = "median",
    normalize: bool = True,
) -> pd.Series:
    """
    Generate type curves for groups of similar wells.

    A type curve is a representative decline curve for a group of wells with
    similar characteristics. This function fits Arps curves to each well in
    a group and returns an aggregated type curve.

    Args:
        df: DataFrame with production data
        grouping_col: Column to group wells by (e.g., 'formation', 'county').
            If None, all wells are treated as one group.
        date_col: Column name for dates
        value_col: Column name for production values
        kind: Arps decline type
        method: Aggregation method:
            - 'mean': Average of fitted parameters
            - 'median': Median of fitted parameters
            - 'p50': Use P50 parameters from distribution
        normalize: If True, normalize all wells to start at t=0 with q=qi.
            If False, use actual time alignment.

    Returns:
        Type curve as a pandas Series with DatetimeIndex starting from first
        production month. The curve represents the typical decline behavior
        for the group.

    Example:
        >>> import pandas as pd
        >>> import decline_curve as dca
        >>> # df has columns: well_id, date, oil_bbl, formation
        >>> type_curve = dca.type_curves(
        ...     df, grouping_col='formation', kind='hyperbolic'
        ... )
    """
    import numpy as np

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")

    # Group wells
    if grouping_col is None:
        groups = {"all": df}
    else:
        if grouping_col not in df.columns:
            raise ValueError(f"Column '{grouping_col}' not found in DataFrame")
        groups = {name: group_df for name, group_df in df.groupby(grouping_col)}

    # For now, process first group (can be extended to return multiple curves)
    group_name = list(groups.keys())[0]
    group_df = groups[group_name]

    # Extract well IDs (assuming there's a well_id column or using index)
    well_col = (
        "well_id" if "well_id" in group_df.columns else group_df.index.name or "well_id"
    )

    # Fit Arps to each well and collect parameters
    params_list = []
    for well_id, well_df in group_df.groupby(well_col):
        try:
            well_df = well_df[[date_col, value_col]].dropna().copy()
            well_df[date_col] = pd.to_datetime(well_df[date_col])
            well_df = well_df.set_index(date_col).sort_index()

            if len(well_df) < 6:
                continue

            series = well_df[value_col]
            t = np.arange(len(series))
            q = series.values
            params = fit_arps(t, q, kind=kind)
            params_list.append((params.qi, params.di, params.b))
        except Exception:
            continue

    if len(params_list) == 0:
        raise ValueError("No wells could be fitted successfully")

    # Aggregate parameters
    qi_values = [p[0] for p in params_list]
    di_values = [p[1] for p in params_list]
    b_values = [p[2] for p in params_list]

    if method == "mean":
        qi_agg = np.mean(qi_values)
        di_agg = np.mean(di_values)
        b_agg = np.mean(b_values)
    elif method == "median":
        qi_agg = np.median(qi_values)
        di_agg = np.median(di_values)
        b_agg = np.median(b_values)
    elif method == "p50":
        qi_agg = np.percentile(qi_values, 50)
        di_agg = np.percentile(di_values, 50)
        b_agg = np.percentile(b_values, 50)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Generate type curve
    type_params = ArpsParams(qi=qi_agg, di=di_agg, b=b_agg)
    t_max = 240  # 20 years
    t = np.arange(0, t_max, 1.0)
    q_curve = predict_arps(t, type_params)

    # Create index starting from first production
    dates = pd.date_range("2000-01-01", periods=len(q_curve), freq="MS")
    type_curve = pd.Series(q_curve, index=dates, name=f"type_curve_{group_name}")

    return type_curve


def run_batch_job(config_path: str | Path | dict | Any) -> dict:
    """
    Run a complete batch decline curve analysis from configuration.

    This is the main config-driven entry point. It loads configuration from a file
    or dict and runs the complete workflow: data loading, validation, forecasting,
    economic analysis, and output generation.

    Args:
        config_path: Path to TOML/YAML config file, dict, or BatchJobConfig instance

    Returns:
        Dictionary with:
        - 'parameters': DataFrame with fitted parameters per well
        - 'forecasts': DataFrame with forecasts per well
        - 'summary': Summary statistics
        - 'config': Configuration used

    Example:
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.run_batch_job('config.toml')
        >>> # Using dict
        >>> config_dict = {'data': {'path': 'data.csv', ...}, ...}
        >>> results = dca.run_batch_job(config_dict)
    """
    from pathlib import Path

    from .config import BatchJobConfig

    # Load configuration
    if isinstance(config_path, BatchJobConfig):
        config = config_path
    elif isinstance(config_path, (str, Path)):
        config = BatchJobConfig.from_file(config_path)
    elif isinstance(config_path, dict):
        config = BatchJobConfig.from_dict(config_path)
    else:
        raise ValueError(
            "config_path must be a file path (str/Path), dict, or BatchJobConfig"
        )

    # Configure logging
    configure_logging(
        level=getattr(__import__("logging"), config.log_level), log_file=config.log_file
    )
    logger.info(
        f"Running batch job from config: "
        f"{config_path if isinstance(config_path, (str, Path)) else 'dict'}"
    )

    # Load data
    logger.info(f"Loading data from: {config.data.path}")
    data_path = Path(config.data.path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {config.data.path}")

    if config.data.format == "csv":
        df = pd.read_csv(data_path)
    elif config.data.format == "parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported format: {config.data.format}")

    # Convert date column
    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])

    # Validate and standardize
    from .schemas import ProductionInputSchema, ProductionOutputSchema

    df_standard = ProductionInputSchema.standardize(df)
    # Validate will raise ValueError if invalid
    ProductionInputSchema.validate(df_standard)

    logger.info(
        f"Loaded {len(df_standard):,} records for "
        f"{df_standard[config.data.well_id_col].nunique():,} wells"
    )

    # Define forecast function from config
    def forecast_well_func(well_data: pd.DataFrame) -> WellResult:
        """Forecast function for a single well."""
        well_id = well_data[config.data.well_id_col].iloc[0]
        well_data = well_data.sort_values(config.data.date_col)

        # Create production series
        series = well_data.set_index(config.data.date_col)[config.data.value_col]
        series = series.asfreq("MS", fill_value=0)

        # Filter to valid production
        series = series[series > 0]

        if len(series) < 3:  # Minimum data requirement
            return WellResult(well_id=well_id, success=False, error="Insufficient data")

        try:
            model_params = getattr(config.model, "params", {})
            forecast_result = forecast(
                series,
                model=config.model.model,
                kind=config.model.kind if config.model.model == "arps" else None,
                horizon=config.model.horizon,
                **model_params,
            )

            # Calculate economic metrics if provided
            metrics = {}
            if config.economics.price and config.economics.opex:
                econ = economic_metrics(
                    forecast_result.values,
                    config.economics.price,
                    config.economics.opex,
                    config.economics.discount_rate,
                )
                metrics.update(econ)

            # Extract parameters for Arps models
            params = None
            if config.model.model == "arps":
                try:
                    import numpy as np

                    t = np.arange(len(series))
                    q = series.values
                    fitted_params = fit_arps(
                        t, q, kind=config.model.kind or "hyperbolic"
                    )
                    params = {
                        "qi": fitted_params.qi,
                        "di": fitted_params.di,
                        "b": fitted_params.b,
                        "kind": config.model.kind or "hyperbolic",
                    }
                except Exception:
                    pass

            return WellResult(
                well_id=well_id,
                success=True,
                forecast=forecast_result,
                params=params,
                metrics=metrics,
            )
        except Exception as e:
            logger.warning(f"Well {well_id} failed: {e}")
            return WellResult(well_id=well_id, success=False, error=str(e))

    # Run analysis using runner
    runner_config = RunnerConfig(
        n_jobs=config.n_jobs,
        chunk_size=config.chunk_size,
        max_retries=config.max_retries,
        log_level=config.log_level,
        log_file=config.log_file,
    )

    runner = FieldScaleRunner(runner_config)
    results = runner.run(
        df_standard,
        forecast_func=forecast_well_func,
        well_id_col=config.data.well_id_col,
    )

    # Convert to output schema
    output_schema = ProductionOutputSchema.from_runner_results(results)
    parameters_df, forecasts_df, events_df = output_schema.to_dataframes()

    # Save outputs
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if config.output.save_parameters:
        output_path = output_dir / f"parameters.{config.output.format}"
        if config.output.format == "csv":
            parameters_df.to_csv(output_path, index=False)
        elif config.output.format == "parquet":
            parameters_df.to_parquet(output_path, index=False)
        saved_files.append(str(output_path))
        logger.info(f"Saved parameters to {output_path}")

    if config.output.save_forecasts:
        output_path = output_dir / f"forecasts.{config.output.format}"
        if config.output.format == "csv":
            forecasts_df.to_csv(output_path, index=False)
        elif config.output.format == "parquet":
            forecasts_df.to_parquet(output_path, index=False)
        saved_files.append(str(output_path))
        logger.info(f"Saved forecasts to {output_path}")

    if config.output.save_reports:
        from .reports import generate_field_pdf_report

        well_results = parameters_df.to_dict("records")
        report_path = output_dir / "field_report.pdf"
        generate_field_pdf_report(well_results, output_path=report_path)
        saved_files.append(str(report_path))
        logger.info(f"Saved report to {report_path}")

    logger.info(
        f"Batch job complete: {results.summary.successful_wells} successful, "
        f"{results.summary.failed_wells} failed"
    )

    return {
        "parameters": parameters_df,
        "forecasts": forecasts_df,
        "events": events_df,
        "summary": results.summary,
        "config": config,
        "output_dir": str(output_dir),
        "saved_files": saved_files,
    }


def run_benchmark(config_path: str | Path | dict | Any) -> dict:
    """
    Run a complete benchmark analysis from configuration.

    This is the config-driven entry point for benchmarking workflows.

    Args:
        config_path: Path to TOML/YAML config file, dict, or BenchmarkConfig instance

    Returns:
        Dictionary with:
        - 'results': DataFrame with metrics for each well
        - 'config': Configuration used
        - 'output_dir': Output directory
        - 'saved_files': List of saved file paths

    Example:
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.run_benchmark('benchmark_config.toml')
        >>> # Using dict
        >>> config_dict = {'data': {'path': 'data.csv', ...}, ...}
        >>> results = dca.run_benchmark(config_dict)
    """
    from .config import BenchmarkConfig

    # Load configuration
    if isinstance(config_path, BenchmarkConfig):
        config = config_path
    elif isinstance(config_path, (str, Path)):
        config = BenchmarkConfig.from_file(config_path)
    elif isinstance(config_path, dict):
        config = BenchmarkConfig.from_dict(config_path)
    else:
        raise ValueError(
            "config_path must be a file path (str/Path), dict, or BenchmarkConfig"
        )

    # Configure logging
    configure_logging(
        level=getattr(__import__("logging"), config.log_level), log_file=config.log_file
    )
    logger.info(
        f"Running benchmark from config: "
        f"{config_path if isinstance(config_path, (str, Path)) else 'dict'}"
    )

    # Load data
    logger.info(f"Loading data from: {config.data.path}")
    data_path = Path(config.data.path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {config.data.path}")

    if config.data.format == "csv":
        df = pd.read_csv(data_path)
    elif config.data.format == "parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported format: {config.data.format}")

    # Convert date column
    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])

    # Run benchmark using direct parameters from config
    # (call internal function to avoid recursion)
    wells = df[config.data.well_id_col].unique()[: config.top_n]

    if JOBLIB_AVAILABLE and config.n_jobs != 1:
        from joblib import Parallel, delayed

        results_list = Parallel(n_jobs=config.n_jobs, verbose=0)(
            delayed(_benchmark_single_well)(
                wid,
                df,
                config.data.well_id_col,
                config.data.date_col,
                config.data.value_col,
                config.model.model,
                config.model.kind if config.model.model == "arps" else None,
                config.model.horizon,
            )
            for wid in wells
        )
        out = [r for r in results_list if r is not None and "error" not in r]
    else:
        out = []
        for wid in wells:
            result = _benchmark_single_well(
                wid,
                df,
                config.data.well_id_col,
                config.data.date_col,
                config.data.value_col,
                config.model.model,
                config.model.kind if config.model.model == "arps" else None,
                config.model.horizon,
            )
            if result is not None and "error" not in result:
                out.append(result)

    results_df = pd.DataFrame(out) if out else pd.DataFrame()

    # Save outputs
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if config.output.save_parameters:
        output_path = output_dir / f"benchmark_results.{config.output.format}"
        if config.output.format == "csv":
            results_df.to_csv(output_path, index=False)
        elif config.output.format == "parquet":
            results_df.to_parquet(output_path, index=False)
        saved_files.append(str(output_path))
        logger.info(f"Saved benchmark results to {output_path}")

    logger.info(f"Benchmark complete: {len(results_df)} wells processed")

    return {
        "results": results_df,
        "config": config,
        "output_dir": str(output_dir),
        "saved_files": saved_files,
    }


def run_sensitivity_analysis(config_path: str | Path | dict | Any) -> dict:
    """
    Run a complete sensitivity analysis from configuration.

    This is the config-driven entry point for sensitivity analysis workflows.

    Args:
        config_path: Path to TOML/YAML config file, dict, or SensitivityConfig instance

    Returns:
        Dictionary with:
        - 'results': DataFrame with sensitivity results
        - 'config': Configuration used
        - 'output_dir': Output directory
        - 'saved_files': List of saved file paths

    Example:
        >>> import decline_curve as dca
        >>> # Using config file
        >>> results = dca.run_sensitivity_analysis('sensitivity_config.toml')
        >>> # Using dict
        >>> config_dict = {'param_grid': [...], 'prices': [70, 80], ...}
        >>> results = dca.run_sensitivity_analysis(config_dict)
    """
    from .config import SensitivityConfig

    # Load configuration
    if isinstance(config_path, SensitivityConfig):
        config = config_path
    elif isinstance(config_path, (str, Path)):
        config = SensitivityConfig.from_file(config_path)
    elif isinstance(config_path, dict):
        config = SensitivityConfig.from_dict(config_path)
    else:
        raise ValueError(
            "config_path must be a file path (str/Path), dict, or SensitivityConfig"
        )

    # Configure logging
    configure_logging(
        level=getattr(__import__("logging"), config.log_level), log_file=config.log_file
    )
    logger.info(
        f"Running sensitivity analysis from config: "
        f"{config_path if isinstance(config_path, (str, Path)) else 'dict'}"
    )

    # Run sensitivity analysis using direct parameters from config
    # (call internal function to avoid recursion)
    results_df = _run_sensitivity(
        config.param_grid,
        config.prices,
        config.opex,
        config.discount_rate,
        config.t_max,
        config.econ_limit,
        config.dt,
    )

    # Save outputs
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if config.output.save_parameters:
        output_path = output_dir / f"sensitivity_results.{config.output.format}"
        if config.output.format == "csv":
            results_df.to_csv(output_path, index=False)
        elif config.output.format == "parquet":
            results_df.to_parquet(output_path, index=False)
        saved_files.append(str(output_path))
        logger.info(f"Saved sensitivity results to {output_path}")

    logger.info(f"Sensitivity analysis complete: {len(results_df)} combinations tested")

    return {
        "results": results_df,
        "config": config,
        "output_dir": str(output_dir),
        "saved_files": saved_files,
    }
