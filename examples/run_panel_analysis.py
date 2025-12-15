#!/usr/bin/env python3
"""Quick panel data analysis run."""

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

logger.info("Loading production data")
df = pd.read_parquet("data/north_dakota_production.parquet")

sample_wells = df["API_WELLNO"].unique()[:1000]
df_sample = df[df["API_WELLNO"].isin(sample_wells)].copy()

logger.info("Analyzing %d wells with %d records", len(sample_wells), len(df_sample))

logger.info("Preparing panel data")
panel = prepare_panel_data(df_sample)
logger.info(
    "Panel data: %d records, %d wells", len(panel), panel["API_WELLNO"].nunique()
)
if "company" in panel.columns:
    logger.info("Companies: %d", panel["company"].nunique())
if "county" in panel.columns:
    logger.info("Counties: %d", panel["county"].nunique())
if "lat" in panel.columns:
    logger.info(
        "Wells with location: %d", panel[["lat", "long"]].notna().all(axis=1).sum()
    )

logger.info("Calculating EUR with company and county controls")
eur_results = eur_with_company_controls(
    panel,
    well_id_col="API_WELLNO",
    date_col="ReportDate",
    value_col="Oil",
    company_col="company",
    county_col="county",
    model_type="hyperbolic",
    min_months=12,
)

logger.info("EUR calculated for %d wells", len(eur_results))

if len(eur_results) > 0:
    logger.info(
        "Overall EUR statistics: mean=%.0f bbl, median=%.0f bbl, "
        "std=%.0f bbl, min=%.0f bbl, max=%.0f bbl",
        eur_results["eur"].mean(),
        eur_results["eur"].median(),
        eur_results["eur"].std(),
        eur_results["eur"].min(),
        eur_results["eur"].max(),
    )

    if "company" in eur_results.columns:
        logger.info("Analyzing by company")
        company_stats = analyze_by_company(eur_results)
        top_companies = (
            eur_results.groupby("company")["eur"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        logger.info("Top 5 companies by mean EUR:")
        for company, mean_eur in top_companies.items():
            count = (eur_results["company"] == company).sum()
            logger.info("  %s: %.0f bbl (n=%d wells)", company, mean_eur, count)

        logger.info("Running company and county fixed effects regression")
        regression_results = company_fixed_effects_regression(
            eur_results,
            dependent_var="eur",
            company_col="company",
            county_col="county",
        )
        if regression_results:
            logger.info("R-squared: %.4f", regression_results["rsquared"])
            logger.info("Observations: %d", regression_results["nobs"])
            logger.info("Companies: %d", regression_results["n_companies"])
            logger.info("Counties: %d", regression_results["n_counties"])

    if "lat" in panel.columns and "long" in panel.columns:
        logger.info("Analyzing spatial patterns")
        spatial_eur = spatial_eur_analysis(panel, eur_results, well_id_col="API_WELLNO")
        if "distance_from_center" in spatial_eur.columns:
            corr = spatial_eur["eur"].corr(spatial_eur["distance_from_center"])
            logger.info("Correlation (EUR vs distance from center): %.3f", corr)
        if "well_density" in spatial_eur.columns:
            corr = spatial_eur["eur"].corr(spatial_eur["well_density"])
            logger.info("Correlation (EUR vs well density): %.3f", corr)

    output_path = Path("outputs/panel_analysis_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    eur_results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

logger.info("Analysis complete")
