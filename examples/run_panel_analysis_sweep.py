#!/usr/bin/env python3
"""Run panel data analysis with parameter sweeps."""

from pathlib import Path

from decline_curve.logging_config import configure_logging, get_logger
from decline_curve.panel_analysis_sweep import PanelAnalysisSweep

logger = get_logger(__name__)
configure_logging()

# Load configuration
config_path = Path("decline_curve/panel_analysis_config_example.yaml")

if not config_path.exists():
    logger.error(f"Config file not found: {config_path}")
    logger.error("Please create a config file or use the example config.")
    exit(1)

logger.info("Loading configuration from %s", config_path)
sweep = PanelAnalysisSweep(config_path)

logger.info("Experiment: %s", sweep.base_config.experiment_name)
logger.info("Data path: %s", sweep.base_config.data_path)
logger.info("Parameter sweeps: %d", len(sweep.parameter_sweeps))

if sweep.parameter_sweeps:
    for i, ps in enumerate(sweep.parameter_sweeps):
        combos = ps.generate_combinations()
        logger.info(
            "Sweep %d: %s (%s) - %d combinations", i + 1, ps.name, ps.type, len(combos)
        )

logger.info("Running parameter sweep")
results_df = sweep.run_sweep(save_results=True)

if len(results_df) > 0:
    logger.info("Total runs: %d", len(results_df))
    logger.info("Columns: %s", ", ".join(results_df.columns))

    if "rsquared" in results_df.columns:
        logger.info(
            "R-squared statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            results_df["rsquared"].mean(),
            results_df["rsquared"].std(),
            results_df["rsquared"].min(),
            results_df["rsquared"].max(),
        )

    if "mean_eur" in results_df.columns:
        logger.info(
            "Mean EUR statistics: mean=%.0f bbl, std=%.0f bbl",
            results_df["mean_eur"].mean(),
            results_df["mean_eur"].std(),
        )

    if "rsquared" in results_df.columns:
        best_idx = results_df["rsquared"].idxmax()
        best_config = results_df.loc[best_idx]
        logger.info("Best configuration (highest R²):")
        for col in results_df.columns:
            if col not in [
                "rsquared",
                "mean_eur",
                "median_eur",
                "std_eur",
                "n_wells",
                "n_observations",
                "n_companies",
                "n_counties",
                "corr_distance",
                "corr_density",
                "sweep_name",
            ]:
                logger.info("  %s: %s", col, best_config[col])

    logger.info(
        "Results saved to: %s/panel_analysis_sweep_results.csv",
        sweep.base_config.output_dir,
    )
else:
    logger.warning("No results generated. Check logs for errors.")

logger.info("Sweep complete")
