from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .economics import economic_metrics
from .evaluate import mae, rmse, smape
from .forecast import Forecaster
from .models import ArpsParams
from .plot import plot_forecast
from .reserves import forecast_and_reserves
from .sensitivity import run_sensitivity
from .utils.data_loader import scrape_ndic


def forecast(
    series: pd.Series,
    model: Literal["arps", "timesfm", "chronos", "arima"] = "arps",
    kind: Optional[Literal["exponential", "harmonic", "hyperbolic"]] = "hyperbolic",
    horizon: int = 12,
    verbose: bool = False,
) -> pd.Series:
    fc = Forecaster(series)
    result = fc.forecast(model=model, kind=kind, horizon=horizon)
    if verbose:
        print(f"Forecast model: {model}, horizon: {horizon}")
        print(result.head())
    return result


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
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
    plot_forecast(y, yhat, title, filename)


def _benchmark_single_well(
    wid, df, well_col, date_col, value_col, model, kind, horizon
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
        yhat = forecast(y, model=model, kind=kind, horizon=horizon)
        metrics = evaluate(y, yhat)
        metrics[well_col] = wid
        return metrics
    except Exception as e:
        return {"error": str(e), well_col: wid}


def benchmark(
    df: pd.DataFrame,
    model: Literal["arps", "timesfm", "chronos", "arima"] = "arps",
    kind: Optional[str] = "hyperbolic",
    horizon: int = 12,
    well_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil_bbl",
    top_n: int = 10,
    verbose: bool = False,
    n_jobs: int = -1,  # -1 uses all available cores
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
        if verbose:
            print(f"Processing {len(wells)} wells in parallel (n_jobs={n_jobs})...")

        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_benchmark_single_well)(
                wid, df, well_col, date_col, value_col, model, kind, horizon
            )
            for wid in wells
        )

        # Filter out None results and errors
        out = [r for r in results if r is not None and "error" not in r]

        if verbose:
            errors = [r for r in results if r is not None and "error" in r]
            print(f"Completed: {len(out)} successful, {len(errors)} failed")
    else:
        # Sequential execution (fallback)
        if verbose and not JOBLIB_AVAILABLE:
            print("joblib not available, running sequentially...")

        out = []
        for wid in wells:
            result = _benchmark_single_well(
                wid, df, well_col, date_col, value_col, model, kind, horizon
            )
            if result is not None:
                if "error" not in result:
                    out.append(result)
                    if verbose:
                        print(f"{wid}: {result}")
                elif verbose:
                    print(f"{wid} failed: {result['error']}")

    return pd.DataFrame(out)


def sensitivity_analysis(
    param_grid: List[Tuple[float, float, float]],
    prices: List[float],
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
) -> Dict:
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
) -> Dict:
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


def load_ndic_data(
    months_list: List[str], output_dir: str = "ndic_raw"
) -> pd.DataFrame:
    """
    Load North Dakota Industrial Commission (NDIC) production data.

    Args:
        months_list: List of month strings (e.g., ['2023-01', '2023-02'])
        output_dir: Directory to save raw data files (default 'ndic_raw')

    Returns:
        Combined DataFrame with production data
    """
    return scrape_ndic(months_list, output_dir)
