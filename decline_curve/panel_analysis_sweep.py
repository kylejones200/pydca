"""Panel data analysis with config-driven parameter sweeps.

This module provides a config-driven system for running panel data analysis
with different combinations of control variables and parameters.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .logging_config import get_logger
from .panel_analysis import (
    company_fixed_effects_regression,
    eur_with_company_controls,
    prepare_panel_data,
    spatial_eur_analysis,
)

logger = get_logger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    # Create dummy mlflow module
    class DummyMlflow:
        """Dummy MLflow class when mlflow is not installed."""

        @staticmethod
        def set_tracking_uri(uri):
            """Set tracking URI (no-op)."""
            pass

        @staticmethod
        def set_experiment(name):
            """Set experiment name (no-op)."""
            pass

        @staticmethod
        def start_run(run_name=None):
            """Start MLflow run (returns null context)."""
            return _null_context()

        @staticmethod
        def log_params(params):
            """Log parameters (no-op)."""
            pass

        @staticmethod
        def log_metrics(metrics):
            """Log metrics (no-op)."""
            pass

        @staticmethod
        def log_param(key, value):
            """Log single parameter (no-op)."""
            pass

    mlflow = DummyMlflow()
    MlflowClient = None


@dataclass
class PanelAnalysisConfig:
    """Configuration for panel data analysis sweep."""

    experiment_name: str
    data_path: str
    well_id_col: str = "API_WELLNO"
    date_col: str = "ReportDate"
    value_col: str = "Oil"
    company_col: str = "Company"
    county_col: str = "County"
    lat_col: str = "Lat"
    long_col: str = "Long"

    # Data sampling
    max_wells: Optional[int] = None
    min_months: int = 12
    random_seed: int = 42

    # EUR calculation
    model_type: str = "hyperbolic"
    t_max: float = 360.0
    econ_limit: float = 10.0

    # Regression specifications
    include_company: bool = True
    include_county: bool = True
    additional_controls: List[str] = field(default_factory=list)

    # Spatial analysis
    include_spatial: bool = True
    basin_center_lat: float = 47.5
    basin_center_long: float = -103.0

    # MLflow tracking
    mlflow_tracking: bool = True
    mlflow_uri: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class ParameterSweep:
    """Parameter sweep configuration for panel analysis."""

    name: str
    type: str  # 'grid', 'random', 'custom'
    parameters: Dict[str, Union[List[Any], Dict[str, Any]]]
    n_samples: Optional[int] = None  # For random search
    seed: Optional[int] = None

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on sweep type."""
        import itertools

        if self.type == "grid":
            keys = self.parameters.keys()
            values = self.parameters.values()
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        elif self.type == "random":
            if self.n_samples is None:
                raise ValueError("n_samples must be specified for random search")
            if self.seed is not None:
                np.random.seed(self.seed)

            combinations = []
            for _ in range(self.n_samples):
                combo = {}
                for param, spec in self.parameters.items():
                    if spec["type"] == "uniform":
                        combo[param] = np.random.uniform(spec["min"], spec["max"])
                    elif spec["type"] == "loguniform":
                        combo[param] = np.exp(
                            np.random.uniform(np.log(spec["min"]), np.log(spec["max"]))
                        )
                    elif spec["type"] == "choice":
                        combo[param] = np.random.choice(spec["values"])
                    elif spec["type"] == "boolean":
                        combo[param] = np.random.choice([True, False])
                    else:
                        raise ValueError(
                            f"Unknown random parameter type: {spec['type']}"
                        )
                combinations.append(combo)
            return combinations
        elif self.type == "custom":
            # Assume parameters is a list of dicts for custom
            return self.parameters  # type: ignore
        else:
            raise ValueError(f"Unknown sweep type: {self.type}")


class PanelAnalysisSweep:
    """Config-driven panel data analysis with parameter sweeps."""

    def __init__(self, config: Union[Path, str, Dict[str, Any]]):
        """Initialize the PanelAnalysisSweep.

        Args:
            config: Path to a YAML/JSON config file or a dictionary config.
        """
        if isinstance(config, (Path, str)):
            config_dict = self._load_config(config)
        else:
            config_dict = config.copy()

        # Extract parameter_sweeps before creating config
        parameter_sweeps_data = config_dict.pop("parameter_sweeps", [])

        self.base_config = self._dict_to_config(config_dict)
        self.parameter_sweeps = []
        for ps_dict in parameter_sweeps_data:
            self.parameter_sweeps.append(ParameterSweep(**ps_dict))

        self.mlflow_client: Optional[MlflowClient] = None
        if self.base_config.mlflow_tracking and MLFLOW_AVAILABLE:
            if self.base_config.mlflow_uri:
                mlflow.set_tracking_uri(self.base_config.mlflow_uri)
            mlflow.set_experiment(self.base_config.experiment_name)
            self.mlflow_client = MlflowClient()
        elif self.base_config.mlflow_tracking and not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow tracking requested but mlflow not installed. "
                "Install with: pip install mlflow"
            )

    def _load_config(self, config_path: Union[Path, str]) -> Dict[str, Any]:
        """Load config from YAML or JSON file."""
        config_path = Path(config_path)

        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path) as f:
                return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PanelAnalysisConfig:
        """Convert dict to PanelAnalysisConfig."""
        return PanelAnalysisConfig(**config_dict)

    def run_single_analysis(
        self, config: PanelAnalysisConfig, param_combo: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a single panel data analysis.

        Args:
            config: Panel analysis configuration
            param_combo: Optional parameter combination from sweep

        Returns:
            Dictionary with analysis results
        """
        # Merge parameter combo into config
        if param_combo:
            for key, value in param_combo.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Load data
        logger.info(f"Loading data from {config.data_path}")
        df = (
            pd.read_parquet(config.data_path)
            if config.data_path.endswith(".parquet")
            else pd.read_csv(config.data_path)
        )

        # Sample wells if specified
        if config.max_wells:
            np.random.seed(config.random_seed)
            sample_wells = np.random.choice(
                df[config.well_id_col].unique(), size=config.max_wells, replace=False
            )
            df = df[df[config.well_id_col].isin(sample_wells)].copy()

        # Prepare panel data
        panel = prepare_panel_data(
            df,
            well_id_col=config.well_id_col,
            date_col=config.date_col,
            value_col=config.value_col,
            company_col=config.company_col,
            lat_col=config.lat_col,
            long_col=config.long_col,
            county_col=config.county_col,
        )

        # Calculate EUR
        eur_results = eur_with_company_controls(
            panel,
            well_id_col=config.well_id_col,
            date_col=config.date_col,
            value_col=config.value_col,
            company_col="company",
            county_col="county",
            model_type=config.model_type,
            min_months=config.min_months,
        )

        if len(eur_results) == 0:
            logger.warning("No EUR calculations succeeded")
            return {"error": "No EUR calculations succeeded"}

        # Run regression
        regression_results = None
        if config.include_company or config.include_county:
            company_col_name = (
                "company"
                if config.include_company and "company" in eur_results.columns
                else None
            )
            county_col_name = (
                "county"
                if config.include_county and "county" in eur_results.columns
                else None
            )

            if company_col_name or county_col_name:
                regression_results = company_fixed_effects_regression(
                    eur_results,
                    dependent_var="eur",
                    company_col=company_col_name,
                    county_col=county_col_name,
                    control_vars=(
                        config.additional_controls
                        if config.additional_controls
                        else None
                    ),
                )

        # Spatial analysis
        spatial_results = None
        if config.include_spatial and "lat" in panel.columns:
            spatial_eur = spatial_eur_analysis(
                panel, eur_results, well_id_col=config.well_id_col
            )
            if "distance_from_center" in spatial_eur.columns:
                corr_distance = spatial_eur["eur"].corr(
                    spatial_eur["distance_from_center"]
                )
                spatial_results = {"corr_distance": corr_distance}
            if "well_density" in spatial_eur.columns:
                corr_density = spatial_eur["eur"].corr(spatial_eur["well_density"])
                if spatial_results:
                    spatial_results["corr_density"] = corr_density
                else:
                    spatial_results = {"corr_density": corr_density}

        # Compile results
        results = {
            "n_wells": len(eur_results),
            "mean_eur": float(eur_results["eur"].mean()),
            "median_eur": float(eur_results["eur"].median()),
            "std_eur": float(eur_results["eur"].std()),
        }

        if regression_results:
            results["rsquared"] = regression_results["rsquared"]
            results["n_observations"] = regression_results["nobs"]
            results["n_companies"] = regression_results["n_companies"]
            results["n_counties"] = regression_results["n_counties"]

        if spatial_results:
            results.update(spatial_results)

        # Add parameter combo to results
        if param_combo:
            results.update(param_combo)

        return results

    def run_sweep(self, save_results: bool = True) -> pd.DataFrame:
        """Run complete parameter sweep.

        Args:
            save_results: Whether to save results to disk

        Returns:
            DataFrame with all sweep results
        """
        all_results = []

        if not self.parameter_sweeps:
            # No sweeps, just run base config
            logger.info("No parameter sweeps specified, running base configuration")
            result = self.run_single_analysis(self.base_config)
            all_results.append(result)
        else:
            # Run each parameter sweep
            for sweep_idx, sweep in enumerate(self.parameter_sweeps):
                logger.info(f"Running parameter sweep: {sweep.name} ({sweep.type})")

                param_combos = sweep.generate_combinations()
                logger.info(f"Generated {len(param_combos)} parameter combinations")

                for combo_idx, param_combo in enumerate(param_combos):
                    logger.info(
                        f"  Running combination {combo_idx+1}/{len(param_combos)}: "
                        f"{param_combo}"
                    )

                    run_name = f"{sweep.name}_combo{combo_idx}"

                    with (
                        mlflow.start_run(run_name=run_name)
                        if self.base_config.mlflow_tracking and MLFLOW_AVAILABLE
                        else _null_context()
                    ):
                        # Log parameters
                        if self.base_config.mlflow_tracking and MLFLOW_AVAILABLE:
                            mlflow.log_params(param_combo)
                            mlflow.log_params(
                                {
                                    "sweep_name": sweep.name,
                                    "sweep_type": sweep.type,
                                    "model_type": self.base_config.model_type,
                                }
                            )

                        try:
                            # Run analysis
                            result = self.run_single_analysis(
                                self.base_config, param_combo
                            )

                            if "error" not in result:
                                # Log metrics to MLflow
                                if (
                                    self.base_config.mlflow_tracking
                                    and MLFLOW_AVAILABLE
                                ):
                                    mlflow.log_metrics(
                                        {
                                            "rsquared": result.get("rsquared", 0),
                                            "mean_eur": result.get("mean_eur", 0),
                                            "n_wells": result.get("n_wells", 0),
                                        }
                                    )

                                result["sweep_name"] = sweep.name
                                all_results.append(result)

                                rsquared = result.get("rsquared", 0)
                                mean_eur = result.get("mean_eur", 0)
                                logger.info(
                                    f"      Analysis complete: R²={rsquared:.4f}, "
                                    f"Mean EUR={mean_eur:,.0f}"
                                )
                            else:
                                logger.warning(f"Analysis failed: {result['error']}")

                        except Exception as e:
                            logger.error(f"Analysis failed for {run_name}: {e}")
                            if self.base_config.mlflow_tracking and MLFLOW_AVAILABLE:
                                mlflow.log_param("error", str(e))

        results_df = pd.DataFrame(all_results)

        # Save results
        if save_results and self.base_config.output_dir:
            output_path = Path(self.base_config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(
                output_path / "panel_analysis_sweep_results.csv", index=False
            )
            logger.info(
                f"Results saved to {output_path / 'panel_analysis_sweep_results.csv'}"
            )

        return results_df


class _null_context:
    """Null context manager for when MLflow is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
