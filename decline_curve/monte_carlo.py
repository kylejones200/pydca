"""Monte Carlo simulation for probabilistic decline curve analysis.

This module enables uncertainty quantification and risk assessment by:
- Sampling from parameter distributions (qi, di, b, prices)
- Running multiple stochastic scenarios
- Generating P10/P50/P90 forecasts
- Computing probabilistic EUR and NPV
- Analyzing correlations between variables

Performance: Uses Numba JIT and joblib parallelization for fast execution.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Optional, cast

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def _jit_noop(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    numba = cast(Any, SimpleNamespace(jit=_jit_noop))

from .economics import economic_metrics
from .models import ArpsParams, predict_arps


@dataclass
class DistributionParams:
    """
    Parameters defining a probability distribution.

    Attributes:
        dist_type: Type of distribution ('normal', 'lognormal', 'uniform', 'triangular')
        mean: Mean value (for normal/lognormal)
        std: Standard deviation (for normal/lognormal)
        min: Minimum value (for uniform/triangular)
        max: Maximum value (for uniform/triangular)
        mode: Mode value (for triangular)
    """

    dist_type: Literal["normal", "lognormal", "uniform", "triangular"]
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mode: Optional[float] = None

    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Generate random samples from this distribution."""
        if seed is not None:
            np.random.seed(seed)

        if self.dist_type == "normal":
            if self.mean is None or self.std is None:
                raise ValueError("Normal distribution requires mean and std")
            return np.random.normal(self.mean, self.std, n)
        elif self.dist_type == "lognormal":
            if self.mean is None or self.std is None:
                raise ValueError("Lognormal distribution requires mean and std")
            return np.random.lognormal(np.log(self.mean), self.std, n)
        elif self.dist_type == "uniform":
            if self.min is None or self.max is None:
                raise ValueError("Uniform distribution requires min and max")
            return np.random.uniform(self.min, self.max, n)
        elif self.dist_type == "triangular":
            if self.min is None or self.max is None or self.mode is None:
                raise ValueError("Triangular distribution requires min, mode, and max")
            return np.random.triangular(self.min, self.mode, self.max, n)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


@dataclass
class MonteCarloParams:
    """
    Parameters for Monte Carlo simulation.

    Attributes:
        qi_dist: Initial production rate distribution
        di_dist: Decline rate distribution
        b_dist: b-factor distribution
        price_dist: Oil/gas price distribution
        opex_dist: Operating cost distribution
        n_simulations: Number of Monte Carlo iterations
        correlation_matrix: Optional correlation between parameters
        seed: Random seed for reproducibility
    """

    qi_dist: DistributionParams
    di_dist: DistributionParams
    b_dist: DistributionParams
    price_dist: DistributionParams
    opex_dist: Optional[DistributionParams] = None
    n_simulations: int = 1000
    correlation_matrix: Optional[np.ndarray] = None
    seed: Optional[int] = None


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo simulation.

    Attributes:
        forecasts: Array of all forecast realizations [n_sims, n_timesteps]
        eur_samples: Array of EUR values from all simulations
        npv_samples: Array of NPV values from all simulations
        p10_forecast: P10 (optimistic) forecast
        p50_forecast: P50 (median) forecast
        p90_forecast: P90 (conservative) forecast
        p10_eur: P10 EUR value
        p50_eur: P50 EUR value
        p90_eur: P90 EUR value
        p10_npv: P10 NPV value
        p50_npv: P50 NPV value
        p90_npv: P90 NPV value
        mean_eur: Mean EUR value
        mean_npv: Mean NPV value
        parameters: DataFrame with all sampled parameters
    """

    forecasts: np.ndarray
    eur_samples: np.ndarray
    npv_samples: np.ndarray
    p10_forecast: np.ndarray
    p50_forecast: np.ndarray
    p90_forecast: np.ndarray
    p10_eur: float
    p50_eur: float
    p90_eur: float
    p10_npv: float
    p50_npv: float
    p90_npv: float
    mean_eur: float
    mean_npv: float
    parameters: pd.DataFrame


def _run_single_simulation(
    qi: float,
    di: float,
    b: float,
    price: float,
    opex: float,
    t: np.ndarray,
    econ_limit: float,
    discount_rate: float,
) -> tuple[np.ndarray, float, float]:
    """
    Run a single Monte Carlo simulation iteration.

    Returns:
        Tuple of (forecast, EUR, NPV)
    """
    # Create Arps parameters
    params = ArpsParams(qi=qi, di=di, b=b)

    # Generate forecast
    forecast = predict_arps(t, params)

    # Apply economic limit
    valid_mask = forecast > econ_limit
    if not np.any(valid_mask):
        return forecast, 0.0, 0.0

    t_valid = t[valid_mask]
    q_valid = forecast[valid_mask]

    # Calculate EUR
    eur = np.trapz(q_valid, t_valid)

    # Calculate NPV
    try:
        econ = economic_metrics(q_valid, price, opex, discount_rate)
        npv = econ["npv"]
    except Exception:
        npv = 0.0

    return forecast, eur, npv


def monte_carlo_forecast(
    mc_params: MonteCarloParams,
    t_max: float = 240.0,
    dt: float = 1.0,
    econ_limit: float = 10.0,
    discount_rate: float = 0.10,
    n_jobs: int = -1,
    verbose: bool = False,
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation for probabilistic forecasting.

    Args:
        mc_params: Monte Carlo parameter distributions
        t_max: Maximum time horizon (months)
        dt: Time step (months)
        econ_limit: Economic production limit
        discount_rate: Annual discount rate
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Print progress messages

    Returns:
        MonteCarloResults with probabilistic forecasts and statistics

    Example:
        >>> from decline_curve.monte_carlo import (
        ...     monte_carlo_forecast, MonteCarloParams, DistributionParams
        ... )
        >>>
        >>> # Define parameter distributions
        >>> mc_params = MonteCarloParams(
        ...     qi_dist=DistributionParams('lognormal', mean=1200, std=0.3),
        ...     di_dist=DistributionParams('uniform', min=0.10, max=0.30),
        ...     b_dist=DistributionParams('triangular', min=0.3, mode=0.5, max=0.8),
        ...     price_dist=DistributionParams('normal', mean=70, std=15),
        ...     opex_dist=DistributionParams('normal', mean=20, std=3),
        ...     n_simulations=1000
        ... )
        >>>
        >>> # Run simulation
        >>> results = monte_carlo_forecast(mc_params)
        >>>
        >>> # Access results
        >>> print(f"P50 EUR: {results.p50_eur:.0f} bbl")
        >>> print(f"P50 NPV: ${results.p50_npv:,.0f}")
    """
    logger.info(
        "Running Monte Carlo simulation",
        extra={"n_simulations": mc_params.n_simulations},
    )

    # Generate time array
    t = np.arange(0, t_max + dt, dt)

    # Sample parameters
    logger.debug("Sampling parameter distributions")

    qi_samples = mc_params.qi_dist.sample(mc_params.n_simulations, mc_params.seed)
    di_samples = mc_params.di_dist.sample(mc_params.n_simulations, mc_params.seed)
    b_samples = mc_params.b_dist.sample(mc_params.n_simulations, mc_params.seed)
    price_samples = mc_params.price_dist.sample(mc_params.n_simulations, mc_params.seed)

    if mc_params.opex_dist is not None:
        opex_samples = mc_params.opex_dist.sample(
            mc_params.n_simulations, mc_params.seed
        )
    else:
        opex_samples = np.full(mc_params.n_simulations, 20.0)  # Default

    # Apply correlation if specified
    if mc_params.correlation_matrix is not None:
        logger.debug("Applying correlation matrix")
        # Use Cholesky decomposition to induce correlation
        params_uncorr = np.vstack([qi_samples, di_samples, b_samples, price_samples])
        L = np.linalg.cholesky(mc_params.correlation_matrix)
        params_corr = L @ params_uncorr
        qi_samples, di_samples, b_samples, price_samples = params_corr

    # Ensure parameters are within valid ranges
    qi_samples = np.maximum(qi_samples, 1.0)
    di_samples = np.clip(di_samples, 0.001, 2.0)
    b_samples = np.clip(b_samples, 0.0, 2.0)
    price_samples = np.maximum(price_samples, 1.0)
    opex_samples = np.maximum(opex_samples, 0.1)

    # Store parameters
    params_df = pd.DataFrame(
        {
            "qi": qi_samples,
            "di": di_samples,
            "b": b_samples,
            "price": price_samples,
            "opex": opex_samples,
        }
    )

    # Run simulations
    logger.debug("Running simulations")

    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel execution
        results_list = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_run_single_simulation)(
                qi_samples[i],
                di_samples[i],
                b_samples[i],
                price_samples[i],
                opex_samples[i],
                t,
                econ_limit,
                discount_rate,
            )
            for i in range(mc_params.n_simulations)
        )
    else:
        # Sequential execution
        results_list = [
            _run_single_simulation(
                qi_samples[i],
                di_samples[i],
                b_samples[i],
                price_samples[i],
                opex_samples[i],
                t,
                econ_limit,
                discount_rate,
            )
            for i in range(mc_params.n_simulations)
        ]

    # Extract results
    forecasts = np.array([r[0] for r in results_list])
    eur_samples = np.array([r[1] for r in results_list])
    npv_samples = np.array([r[2] for r in results_list])

    # Calculate percentiles
    logger.debug("Computing statistics")

    p10_forecast = np.percentile(
        forecasts, 90, axis=0
    )  # P10 is 90th percentile (optimistic)
    p50_forecast = np.percentile(forecasts, 50, axis=0)
    p90_forecast = np.percentile(
        forecasts, 10, axis=0
    )  # P90 is 10th percentile (conservative)

    p10_eur = np.percentile(eur_samples, 90)
    p50_eur = np.percentile(eur_samples, 50)
    p90_eur = np.percentile(eur_samples, 10)

    p10_npv = np.percentile(npv_samples, 90)
    p50_npv = np.percentile(npv_samples, 50)
    p90_npv = np.percentile(npv_samples, 10)

    mean_eur = np.mean(eur_samples)
    mean_npv = np.mean(npv_samples)

    logger.info(
        "Monte Carlo results",
        extra={
            "eur_p90": p90_eur,
            "eur_p50": p50_eur,
            "eur_p10": p10_eur,
            "eur_mean": mean_eur,
            "npv_p90": p90_npv,
            "npv_p50": p50_npv,
            "npv_p10": p10_npv,
            "npv_mean": mean_npv,
        },
    )

    return MonteCarloResults(
        forecasts=forecasts,
        eur_samples=eur_samples,
        npv_samples=npv_samples,
        p10_forecast=p10_forecast,
        p50_forecast=p50_forecast,
        p90_forecast=p90_forecast,
        p10_eur=p10_eur,
        p50_eur=p50_eur,
        p90_eur=p90_eur,
        p10_npv=p10_npv,
        p50_npv=p50_npv,
        p90_npv=p90_npv,
        mean_eur=mean_eur,
        mean_npv=mean_npv,
        parameters=params_df,
    )


def plot_monte_carlo_results(
    results: MonteCarloResults,
    t: Optional[np.ndarray] = None,
    title: str = "Monte Carlo Forecast",
    filename: Optional[str] = None,
):
    """
    Plot Monte Carlo simulation results.

    Creates a fan chart showing P10/P50/P90 forecasts and distributions.

    Args:
        results: Monte Carlo simulation results
        t: Time array (optional, will use months if not provided)
        title: Plot title
        filename: Optional filename to save plot

    Example:
        >>> from decline_curve.monte_carlo import plot_monte_carlo_results
        >>> plot_monte_carlo_results(results, title="Well 123 Probabilistic Forecast")
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Create time array if not provided
    if t is None:
        t = np.arange(len(results.p50_forecast))

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Fan chart for production forecast
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(
        t,
        results.p90_forecast,
        results.p10_forecast,
        alpha=0.3,
        color="blue",
        label="P10-P90 range",
    )
    ax1.plot(t, results.p50_forecast, "b-", linewidth=2, label="P50 (median)")
    ax1.plot(
        t, results.p10_forecast, "g--", linewidth=1, alpha=0.7, label="P10 (optimistic)"
    )
    ax1.plot(
        t,
        results.p90_forecast,
        "r--",
        linewidth=1,
        alpha=0.7,
        label="P90 (conservative)",
    )
    ax1.set_xlabel("Time (months)")
    ax1.set_ylabel("Production Rate")
    ax1.set_title(f"{title} - Fan Chart")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. EUR distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(results.eur_samples, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax2.axvline(
        results.p90_eur,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"P90: {results.p90_eur:,.0f}",
    )
    ax2.axvline(
        results.p50_eur,
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"P50: {results.p50_eur:,.0f}",
    )
    ax2.axvline(
        results.p10_eur,
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"P10: {results.p10_eur:,.0f}",
    )
    ax2.set_xlabel("EUR (bbl or mcf)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("EUR Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. NPV distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(results.npv_samples, bins=50, alpha=0.7, color="green", edgecolor="black")
    ax3.axvline(
        results.p90_npv,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"P90: ${results.p90_npv:,.0f}",
    )
    ax3.axvline(
        results.p50_npv,
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"P50: ${results.p50_npv:,.0f}",
    )
    ax3.axvline(
        results.p10_npv,
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"P10: ${results.p10_npv:,.0f}",
    )
    ax3.set_xlabel("NPV ($)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("NPV Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Parameter correlation heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    corr_matrix = results.parameters.corr()
    im = ax4.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45)
    ax4.set_yticklabels(corr_matrix.columns)
    plt.colorbar(im, ax=ax4, label="Correlation")
    ax4.set_title("Parameter Correlations")

    # 5. Risk metrics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Create summary table
    table_data = [
        ["Metric", "P90", "P50", "P10", "Mean"],
        [
            "EUR",
            f"{results.p90_eur:,.0f}",
            f"{results.p50_eur:,.0f}",
            f"{results.p10_eur:,.0f}",
            f"{results.mean_eur:,.0f}",
        ],
        [
            "NPV ($)",
            f"{results.p90_npv:,.0f}",
            f"{results.p50_npv:,.0f}",
            f"{results.p10_npv:,.0f}",
            f"{results.mean_npv:,.0f}",
        ],
    ]

    table = ax5.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax5.set_title("Risk Metrics Summary", pad=20)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        logger.debug(f"Plot saved to {filename}")

    plt.show()


def risk_analysis(
    results: MonteCarloResults, threshold: Optional[float] = None
) -> dict[str, float]:
    """
    Compute risk metrics from Monte Carlo results.

    Args:
        results: Monte Carlo simulation results
        threshold: Optional NPV threshold for probability calculations

    Returns:
        Dictionary with risk metrics

    Example:
        >>> risk_metrics = risk_analysis(results, threshold=0)
        >>> prob = risk_metrics['prob_positive_npv']
        >>> print(f"Probability of positive NPV: {prob:.1%}")
    """
    metrics = {
        "eur_mean": results.mean_eur,
        "eur_std": np.std(results.eur_samples),
        "eur_cv": np.std(results.eur_samples)
        / results.mean_eur,  # Coefficient of variation
        "npv_mean": results.mean_npv,
        "npv_std": np.std(results.npv_samples),
        "npv_cv": (
            np.std(results.npv_samples) / results.mean_npv
            if results.mean_npv != 0
            else np.inf
        ),
        "eur_range": results.p10_eur - results.p90_eur,
        "npv_range": results.p10_npv - results.p90_npv,
    }

    if threshold is not None:
        metrics["prob_npv_above_threshold"] = np.mean(results.npv_samples > threshold)
        metrics["prob_npv_below_threshold"] = np.mean(results.npv_samples < threshold)

    # Additional risk metrics
    metrics["prob_positive_npv"] = np.mean(results.npv_samples > 0)
    metrics["value_at_risk_npv"] = np.percentile(results.npv_samples, 5)  # 5% VaR

    return metrics


def sensitivity_to_monte_carlo(
    base_qi: float,
    base_di: float,
    base_b: float,
    qi_range: tuple[float, float],
    di_range: tuple[float, float],
    b_range: tuple[float, float],
    price_range: tuple[float, float],
    n_simulations: int = 1000,
) -> MonteCarloParams:
    """
    Convert sensitivity ranges to Monte Carlo distributions.

    Helper function to create Monte Carlo parameters from sensitivity analysis ranges.
    Assumes uniform distributions across specified ranges.

    Args:
        base_qi: Base initial production rate
        base_di: Base decline rate
        base_b: Base b-factor
        qi_range: (min, max) for qi
        di_range: (min, max) for di
        b_range: (min, max) for b
        price_range: (min, max) for price
        n_simulations: Number of Monte Carlo iterations

    Returns:
        MonteCarloParams object

    Example:
        >>> mc_params = sensitivity_to_monte_carlo(
        ...     base_qi=1200, base_di=0.15, base_b=0.5,
        ...     qi_range=(1000, 1400),
        ...     di_range=(0.10, 0.20),
        ...     b_range=(0.3, 0.7),
        ...     price_range=(50, 90)
        ... )
    """
    return MonteCarloParams(
        qi_dist=DistributionParams("uniform", min=qi_range[0], max=qi_range[1]),
        di_dist=DistributionParams("uniform", min=di_range[0], max=di_range[1]),
        b_dist=DistributionParams("uniform", min=b_range[0], max=b_range[1]),
        price_dist=DistributionParams(
            "uniform", min=price_range[0], max=price_range[1]
        ),
        n_simulations=n_simulations,
    )
