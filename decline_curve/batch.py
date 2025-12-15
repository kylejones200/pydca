"""Batch processing module for decline curve analysis.

This module provides deterministic batch processing capabilities for
fitting and forecasting across many wells, with support for parallelization
and reproducible results.

Features:
- Deterministic batch runner with fixed seeds
- Joblib parallelism
- Manifest-based input
- Parquet output support
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .fitting import CurveFitFitter, FitSpec, RobustLeastSquaresFitter
from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning(
        "joblib not available. Install with: pip install joblib. "
        "Parallel batch processing will be unavailable."
    )


@dataclass
class BatchManifest:
    """Manifest for batch processing.

    Attributes:
        wells: List of well entries with paths and metadata
    """

    wells: List[Dict[str, Any]]

    @classmethod
    def from_file(cls, filepath: str) -> "BatchManifest":
        """Load manifest from YAML or JSON file."""
        import json

        import yaml

        path = Path(filepath)
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(wells=data.get("wells", []))

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {"wells": self.wells}


def process_single_well(
    well_id: str,
    data_path: str,
    fit_spec: FitSpec,
    fitter_name: str = "curve_fit",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Process a single well.

    Args:
        well_id: Well identifier
        data_path: Path to well data file
        fit_spec: FitSpec for fitting
        fitter_name: Name of fitter to use
        seed: Random seed for reproducibility

    Returns:
        Dictionary with well results
    """
    try:
        # Load data
        df = (
            pd.read_csv(data_path)
            if data_path.endswith(".csv")
            else pd.read_parquet(data_path)
        )

        # Extract time and rate (simplified - would use data contract)
        t = df["date"].values if "date" in df.columns else np.arange(len(df))
        q = df["oil_rate"].values if "oil_rate" in df.columns else df["rate"].values

        # Select fitter
        if fitter_name == "curve_fit":
            fitter = CurveFitFitter()
        elif fitter_name == "robust_least_squares":
            fitter = RobustLeastSquaresFitter()
        else:
            raise ValueError(f"Unknown fitter: {fitter_name}")

        # Fit
        result = fitter.fit(t, q, fit_spec)

        return {
            "well_id": well_id,
            "success": result.success,
            "params": result.params,
            "r_squared": result.r_squared,
            "rmse": result.rmse,
            "warnings": result.warnings,
        }
    except Exception as e:
        logger.warning(f"Failed to process well {well_id}: {e}")
        return {
            "well_id": well_id,
            "success": False,
            "error": str(e),
        }


def batch_fit(
    manifest: Union[BatchManifest, str],
    fit_spec: FitSpec,
    fitter_name: str = "curve_fit",
    output_dir: str = "batch_output",
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Run batch fitting on multiple wells.

    Args:
        manifest: BatchManifest or path to manifest file
        fit_spec: FitSpec for all wells
        fitter_name: Name of fitter to use
        output_dir: Output directory for results
        n_jobs: Number of parallel jobs (-1 for all cores)
        seed: Base random seed (each well gets seed + well_index)

    Returns:
        DataFrame with one row per well containing fit results
    """
    # Load manifest
    if isinstance(manifest, str):
        manifest = BatchManifest.from_file(manifest)

    logger.info(
        f"Starting batch fit: {len(manifest.wells)} wells",
        extra={"n_wells": len(manifest.wells), "n_jobs": n_jobs},
    )

    # Process wells
    if JOBLIB_AVAILABLE and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_well)(
                well["well_id"],
                well["data_path"],
                fit_spec,
                fitter_name,
                seed + i if seed else None,
            )
            for i, well in enumerate(manifest.wells)
        )
    else:
        results = [
            process_single_well(
                well["well_id"],
                well["data_path"],
                fit_spec,
                fitter_name,
                seed + i if seed else None,
            )
            for i, well in enumerate(manifest.wells)
        ]

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save to Parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "batch_results.parquet"
    df_results.to_parquet(output_file, index=False)

    n_successful = int(df_results["success"].sum())
    logger.info(
        f"Batch fit complete: {n_successful}/{len(df_results)} successful",
        extra={
            "n_successful": n_successful,
            "output_file": str(output_file),
        },
    )

    return df_results
