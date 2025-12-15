#!/usr/bin/env python3
"""Panel data analysis example with company fixed effects and spatial controls.

This script demonstrates how to:
1. Prepare production data as panel data
2. Control for company ownership (first company per well)
3. Use location data for spatial analysis
4. Calculate EUR with company and spatial controls
"""

from pathlib import Path

import pandas as pd

from decline_curve.logging_config import configure_logging, get_logger
from decline_curve.panel_analysis import (
    analyze_by_company,
    company_fixed_effects_regression,
    eur_with_company_controls,
    prepare_panel_data,
    spatial_eur_analysis,
)

logger = get_logger(__name__)
configure_logging()


def main():
    """Run panel data analysis."""
    data_path = Path("data/north_dakota_production.parquet")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info("Loading production data...")
    df = pd.read_parquet(data_path)

    # Sample for demonstration (use all data for full analysis)
    sample_wells = df["API_WELLNO"].unique()[:1000]  # First 1000 wells
    df_sample = df[df["API_WELLNO"].isin(sample_wells)].copy()

    logger.info(f"Analyzing {len(sample_wells)} wells with {len(df_sample)} records")

    # Prepare panel data
    logger.info("\nPreparing panel data...")
    panel = prepare_panel_data(df_sample)

    # Calculate EUR with company controls
    logger.info("\nCalculating EUR with company controls...")
    eur_results = eur_with_company_controls(
        panel,
        well_id_col="API_WELLNO",
        date_col="ReportDate",
        value_col="Oil",
        company_col="company",
        model_type="hyperbolic",
        min_months=12,
    )

    logger.info(f"Calculated EUR for {len(eur_results)} wells")

    # Analyze by company
    if "company" in eur_results.columns:
        logger.info("\nAnalyzing by company...")
        analyze_by_company(eur_results)

        # Company fixed effects regression
        logger.info("\nRunning company fixed effects regression...")
        company_fixed_effects_regression(
            eur_results, dependent_var="eur", company_col="company"
        )

    # Analyze spatial patterns
    if "lat" in panel.columns and "long" in panel.columns:
        logger.info("\nAnalyzing spatial patterns...")
        spatial_eur_analysis(panel, eur_results, well_id_col="API_WELLNO")

    # Summary statistics
    logger.info("\nOverall EUR statistics:")
    logger.info(f"Mean EUR: {eur_results['eur'].mean():,.0f} bbl")
    logger.info(f"Median EUR: {eur_results['eur'].median():,.0f} bbl")
    logger.info(f"Std EUR: {eur_results['eur'].std():,.0f} bbl")

    # Save results
    output_path = Path("outputs/panel_analysis_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    eur_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
