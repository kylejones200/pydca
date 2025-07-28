"""
Plotting utilities for decline curve analysis with Tufte-style aesthetics.
"""

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tufte_style():
    """Apply Tufte-inspired minimalist styling to matplotlib plots."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.linewidth": 0.5,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.bottom": True,
            "xtick.top": False,
            "ytick.left": True,
            "ytick.right": False,
            "axes.grid": True,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.7,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
        }
    )


def _range_markers(ax, values: np.ndarray):
    """Add subtle range markers to indicate data spread."""
    if len(values) == 0:
        return

    y_min, y_max = np.min(values), np.max(values)
    y_range = y_max - y_min

    if y_range > 0:
        # Add subtle horizontal lines at quartiles
        q25, q75 = np.percentile(values, [25, 75])
        ax.axhline(q25, color="gray", alpha=0.3, linewidth=0.5, linestyle="--")
        ax.axhline(q75, color="gray", alpha=0.3, linewidth=0.5, linestyle="--")


def plot_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Production Forecast",
    filename: Optional[str] = None,
    show_metrics: bool = True,
):
    """
    Plot historical production data with forecast.

    Parameters:
    - y_true: Historical production data
    - y_pred: Forecasted production data
    - title: Plot title
    - filename: Optional filename to save plot
    - show_metrics: Whether to display evaluation metrics
    """
    tufte_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Split historical and forecast data
    hist_end = y_true.index[-1]
    hist_data = y_pred[y_pred.index <= hist_end]
    forecast_data = y_pred[y_pred.index > hist_end]

    # Plot historical data
    ax.plot(
        y_true.index,
        y_true.values,
        "o-",
        color="#2E86AB",
        linewidth=2,
        markersize=4,
        label="Historical",
        alpha=0.8,
    )

    # Plot fitted curve on historical period
    if len(hist_data) > 0:
        ax.plot(
            hist_data.index,
            hist_data.values,
            "--",
            color="#A23B72",
            linewidth=2,
            label="Fitted",
            alpha=0.7,
        )

    # Plot forecast
    if len(forecast_data) > 0:
        ax.plot(
            forecast_data.index,
            forecast_data.values,
            "-",
            color="#F18F01",
            linewidth=2.5,
            label="Forecast",
            alpha=0.9,
        )

    # Add range markers
    _range_markers(ax, y_true.values)

    # Formatting
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Production Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)

    # Format x-axis dates
    if isinstance(y_true.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add metrics text box if requested
    if show_metrics and len(hist_data) > 0:
        from .evaluate import mae, rmse, smape

        common_idx = y_true.index.intersection(hist_data.index)
        if len(common_idx) > 0:
            y_common = y_true.loc[common_idx]
            pred_common = hist_data.loc[common_idx]

            metrics_text = f"RMSE: {rmse(y_common, pred_common):.1f}\n"
            metrics_text += f"MAE: {mae(y_common, pred_common):.1f}\n"
            metrics_text += f"SMAPE: {smape(y_common, pred_common):.1f}%"

            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
                ),
            )

    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_decline_curve(
    t: np.ndarray, q: np.ndarray, params: dict, title: str = "Decline Curve Analysis"
):
    """
    Plot decline curve with fitted parameters.

    Parameters:
    - t: time array
    - q: production rate array
    - params: fitted decline curve parameters
    - title: plot title
    """
    from .models import predict_arps

    tufte_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Linear plot
    t_extended = np.linspace(0, max(t) * 2, 100)
    q_fit = predict_arps(t_extended, params)

    ax1.plot(t, q, "o", color="#2E86AB", markersize=6, label="Data", alpha=0.7)
    ax1.plot(t_extended, q_fit, "-", color="#F18F01", linewidth=2, label="Fitted")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Production Rate")
    ax1.set_title("Linear Scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log plot
    ax2.semilogy(t, q, "o", color="#2E86AB", markersize=6, label="Data", alpha=0.7)
    ax2.semilogy(t_extended, q_fit, "-", color="#F18F01", linewidth=2, label="Fitted")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Production Rate (log scale)")
    ax2.set_title("Semi-log Scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add parameter text
    param_text = f"Model: {params['kind'].title()}\n"
    param_text += f"qi: {params['qi']:.1f}\n"
    param_text += f"di: {params['di']:.4f}\n"
    if "b" in params:
        param_text += f"b: {params['b']:.3f}\n"
    if "r2" in params:
        param_text += f"R²: {params['r2']:.3f}"

    ax1.text(
        0.02,
        0.98,
        param_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_benchmark_results(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Model Benchmark Results",
):
    """
    Plot benchmark results across multiple wells.

    Parameters:
    - results_df: DataFrame with benchmark results
    - metric: metric to plot ('rmse', 'mae', 'smape')
    - title: plot title
    """
    tufte_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    if metric in results_df.columns:
        results_sorted = results_df.sort_values(metric)
        bars = ax.bar(
            range(len(results_sorted)),
            results_sorted[metric],
            color="#2E86AB",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel("Well Index")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{title} - {metric.upper()}")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add summary statistics
        mean_val = results_sorted[metric].mean()
        median_val = results_sorted[metric].median()
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_val:.1f}",
        )
        ax.axhline(
            median_val,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Median: {median_val:.1f}",
        )

        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metric '{metric}' not found in results DataFrame")
