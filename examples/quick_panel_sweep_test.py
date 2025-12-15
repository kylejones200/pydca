#!/usr/bin/env python3
"""Quick test of panel analysis sweep with small sample."""

from decline_curve.logging_config import configure_logging, get_logger
from decline_curve.panel_analysis_sweep import PanelAnalysisSweep

logger = get_logger(__name__)
configure_logging()

# Create a minimal config for testing
config_dict = {
    "experiment_name": "quick_test",
    "data_path": "data/north_dakota_production.parquet",
    "max_wells": 100,  # Small sample for quick test
    "min_months": 12,
    "mlflow_tracking": False,
    "include_company": True,
    "include_county": True,
    "parameter_sweeps": [
        {
            "name": "regression_specs",
            "type": "grid",
            "parameters": {
                "include_company": [True, False],
                "include_county": [True, False],
            },
        }
    ],
}

logger.info("Running quick panel analysis sweep test")
sweep = PanelAnalysisSweep(config_dict)
results_df = sweep.run_sweep(save_results=False)

logger.info("Results: %d runs", len(results_df))
if len(results_df) > 0:
    logger.info("Columns: %s", list(results_df.columns))
    if "rsquared" in results_df.columns:
        logger.info(
            "R² range: %.4f to %.4f",
            results_df["rsquared"].min(),
            results_df["rsquared"].max(),
        )
    logger.info("First few results:\n%s", results_df.head())

logger.info("Quick test complete")
