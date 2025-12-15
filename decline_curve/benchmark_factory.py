"""Benchmarking factory for parameter sweeps and experiment tracking.

This module provides a config-driven factory for creating benchmarking scenarios
with parameter sweeps and MLflow integration for experiment tracking.
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import dca
from .logging_config import get_logger
from .test_utils import (
    generate_piecewise_decline,
    generate_ramp_up_data,
    generate_synthetic_arps_data,
)

logger = get_logger(__name__)


class _null_context:
    """Null context manager for when MLflow is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


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
class ParameterSweep:
    """Parameter sweep configuration.

    Supports grid search, random search, and custom parameter combinations.
    """

    name: str
    type: str  # 'grid', 'random', 'custom'
    parameters: Dict[str, Union[List[Any], Dict[str, Any]]]
    n_samples: Optional[int] = None  # For random search
    seed: Optional[int] = None

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on sweep type."""
        sweep_methods = {
            "grid": self._grid_search,
            "random": self._random_search,
            "custom": self._custom_combinations,
        }
        method = sweep_methods.get(self.type)
        if method is None:
            raise ValueError(f"Unknown sweep type: {self.type}")
        return method()

    def _grid_search(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        keys = list(self.parameters.keys())
        values = [self.parameters[k] for k in keys]
        combinations = []

        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _random_search(self) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        if self.n_samples is None:
            raise ValueError("n_samples required for random search")

        np.random.seed(self.seed)
        keys = list(self.parameters.keys())
        combinations = []

        distribution_samplers = {
            "uniform": lambda spec: np.random.uniform(spec["min"], spec["max"]),
            "normal": lambda spec: np.random.normal(spec["mean"], spec["std"]),
            "lognormal": lambda spec: np.random.lognormal(spec["mean"], spec["std"]),
            "choice": lambda spec: np.random.choice(spec["values"]),
        }

        for _ in range(self.n_samples):
            combo = {}
            for key in keys:
                param_spec = self.parameters[key]
                if isinstance(param_spec, dict):
                    dist_type = param_spec.get("type", "uniform")
                    sampler = distribution_samplers.get(dist_type)
                    if sampler is None:
                        raise ValueError(f"Unknown distribution type: {dist_type}")
                    combo[key] = sampler(param_spec)
                else:
                    combo[key] = np.random.choice(param_spec)

            combinations.append(combo)

        return combinations

    def _custom_combinations(self) -> List[Dict[str, Any]]:
        """Use custom parameter combinations."""
        if not isinstance(self.parameters, list):
            raise ValueError("Custom type requires parameters to be a list of dicts")
        return self.parameters


@dataclass
class WellConfig:
    """Configuration for generating synthetic well data or loading real data."""

    well_type: str  # 'hyperbolic', 'exponential', 'harmonic', 'piecewise', 'ramp_up', 'real_data'  # noqa: E501
    parameters: Dict[str, Any]
    noise_level: float = 0.05
    t_max: float = 100.0
    dt: float = 1.0
    n_months: Optional[int] = None  # Alternative to t_max/dt
    # For real data
    data_path: Optional[str] = None  # Path to CSV file with real data
    well_id_col: str = "API_WELLNO"  # Column name for well identifier
    date_col: str = "Date"  # Column name for date
    value_col: str = "Oil"  # Column name for production value
    max_wells: Optional[int] = None  # Limit number of wells from real data
    min_months: int = 12  # Minimum production history required


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    experiment_name: str
    well_configs: List[WellConfig]
    parameter_sweeps: List[ParameterSweep]
    models: List[str] = field(
        default_factory=lambda: ["arps"]
    )  # Can include: arps, arima, exponential_smoothing, moving_average, linear_trend, holt_winters  # noqa: E501
    horizons: List[int] = field(default_factory=lambda: [12])
    n_wells_per_config: int = 1
    seed: int = 42
    mlflow_tracking: bool = True
    mlflow_uri: Optional[str] = None
    output_dir: Optional[str] = None


class BenchmarkFactory:
    """Factory for creating and running benchmark experiments with parameter sweeps."""

    def __init__(self, config: Union[BenchmarkConfig, Dict[str, Any], Path, str]):
        """Initialize factory from config.

        Args:
            config: BenchmarkConfig instance, dict, or path to YAML/JSON config file
        """
        if isinstance(config, (Path, str)):
            config = self._load_config(config)
        if isinstance(config, dict):
            config = self._dict_to_config(config)

        self.config = config
        self.mlflow_client = None

        if not self.config.mlflow_tracking:
            return

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow tracking requested but mlflow not installed. "
                "Install with: pip install mlflow"
            )
            return

        if self.config.mlflow_uri:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        self.mlflow_client = MlflowClient()

    def _load_config(self, config_path: Union[Path, str]) -> Dict[str, Any]:
        """Load config from YAML or JSON file."""
        config_path = Path(config_path)

        loaders = {
            ".yaml": self._load_yaml,
            ".yml": self._load_yaml,
            ".json": self._load_json,
        }

        loader = loaders.get(config_path.suffix)
        if loader is None:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        return loader(config_path)

    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML config file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML configs. pip install pyyaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _load_json(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON config file."""
        with open(config_path) as f:
            return json.load(f)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> BenchmarkConfig:
        """Convert dict to BenchmarkConfig."""
        # Parse well configs
        well_configs = []
        for wc_dict in config_dict.get("well_configs", []):
            well_configs.append(WellConfig(**wc_dict))

        # Parse parameter sweeps
        parameter_sweeps = []
        for ps_dict in config_dict.get("parameter_sweeps", []):
            parameter_sweeps.append(ParameterSweep(**ps_dict))

        return BenchmarkConfig(
            experiment_name=config_dict["experiment_name"],
            well_configs=well_configs,
            parameter_sweeps=parameter_sweeps,
            models=config_dict.get("models", ["arps"]),
            horizons=config_dict.get("horizons", [12]),
            n_wells_per_config=config_dict.get("n_wells_per_config", 1),
            seed=config_dict.get("seed", 42),
            mlflow_tracking=config_dict.get("mlflow_tracking", True),
            mlflow_uri=config_dict.get("mlflow_uri"),
            output_dir=config_dict.get("output_dir"),
        )

    def generate_wells(
        self, well_config: WellConfig, param_combo: Dict[str, Any], n_wells: int
    ) -> pd.DataFrame:
        """Generate synthetic well data or load real data from config and parameters."""
        if well_config.well_type == "real_data":
            return self._load_real_wells(well_config, n_wells)

        np.random.seed(self.config.seed)
        data = []
        params = {**well_config.parameters, **param_combo}

        well_generators = {
            "hyperbolic": lambda i: generate_synthetic_arps_data(
                qi=params.get("qi", 1000),
                di=params.get("di", 0.1),
                b=params.get("b", 0.5),
                t_max=well_config.t_max,
                dt=well_config.dt,
                noise_level=well_config.noise_level,
                seed=self.config.seed + i,
                model_type="hyperbolic",
            ),
            "exponential": lambda i: generate_synthetic_arps_data(
                qi=params.get("qi", 1000),
                di=params.get("di", 0.1),
                b=0.0,
                t_max=well_config.t_max,
                dt=well_config.dt,
                noise_level=well_config.noise_level,
                seed=self.config.seed + i,
                model_type="exponential",
            ),
            "harmonic": lambda i: generate_synthetic_arps_data(
                qi=params.get("qi", 1000),
                di=params.get("di", 0.1),
                b=1.0,
                t_max=well_config.t_max,
                dt=well_config.dt,
                noise_level=well_config.noise_level,
                seed=self.config.seed + i,
                model_type="harmonic",
            ),
            "piecewise": lambda i: generate_piecewise_decline(
                segments=params.get("segments", []),
                t_max=well_config.t_max,
                dt=well_config.dt,
                noise_level=well_config.noise_level,
                seed=self.config.seed + i,
            ),
            "ramp_up": lambda i: generate_ramp_up_data(
                qi_final=params.get("qi_final", 1000),
                ramp_duration=params.get("ramp_duration", 10),
                decline_di=params.get("decline_di", 0.1),
                decline_b=params.get("decline_b", 0.5),
                t_max=well_config.t_max,
                dt=well_config.dt,
                noise_level=well_config.noise_level,
                seed=self.config.seed + i,
            ),
        }

        generator = well_generators.get(well_config.well_type)
        if generator is None:
            raise ValueError(f"Unknown well type: {well_config.well_type}")

        for i in range(n_wells):
            well_id = f"WELL_{len(data):06d}"
            t, q = generator(i)

            dates, q_sampled = self._prepare_time_series(t, q, well_config)

            for date, production in zip(dates, q_sampled):
                data.append(
                    {
                        "well_id": well_id,
                        "date": date,
                        "oil_bbl": max(0, production),
                    }
                )

        return pd.DataFrame(data)

    def _prepare_time_series(
        self, t: np.ndarray, q: np.ndarray, well_config: WellConfig
    ) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Prepare time series with appropriate frequency."""
        if well_config.n_months:
            dates = pd.date_range("2020-01-01", periods=well_config.n_months, freq="MS")
            q_sampled = (
                np.interp(np.linspace(0, len(t) - 1, len(dates)), np.arange(len(t)), q)
                if len(t) != len(dates)
                else q
            )
        else:
            dates = pd.date_range("2020-01-01", periods=len(t), freq="D")
            q_sampled = q

        return dates, q_sampled

    def _load_real_wells(self, well_config: WellConfig, n_wells: int) -> pd.DataFrame:
        """Load real well data from CSV file."""
        if well_config.data_path is None:
            raise ValueError("data_path required for real_data well_type")

        logger.info(
            f"Loading real data from {well_config.data_path} "
            f"(max_wells={n_wells or well_config.max_wells})"
        )

        data_path = Path(well_config.data_path)
        loaders = {
            ".parquet": pd.read_parquet,
            ".csv": pd.read_csv,
        }
        loader = loaders.get(data_path.suffix, pd.read_csv)
        df = loader(data_path)

        # Rename columns to standard format
        column_mapping = {
            well_config.well_id_col: "well_id",
            well_config.date_col: "date",
            well_config.value_col: "oil_bbl",
        }

        # Check which columns exist
        available_cols = set(df.columns)
        required_cols = set(column_mapping.keys())
        missing_cols = required_cols - available_cols

        if missing_cols:
            raise ValueError(
                f"Missing required columns in data file: {missing_cols}. "
                f"Available columns: {list(available_cols)}"
            )

        # Select and rename columns
        df = df[list(column_mapping.keys())].copy()
        df = df.rename(columns=column_mapping)

        # Convert date column
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Remove rows with missing dates or negative production
        df = df[df["date"].notna() & (df["oil_bbl"] >= 0)]

        # Filter by minimum months per well
        well_month_counts = df.groupby("well_id")["date"].nunique()
        valid_wells = well_month_counts[
            well_month_counts >= well_config.min_months
        ].index

        if len(valid_wells) == 0:
            raise ValueError(
                f"No wells found with at least {well_config.min_months} months of data"
            )

        max_wells = n_wells or well_config.max_wells
        if max_wells and len(valid_wells) > max_wells:
            np.random.seed(self.config.seed)
            valid_wells = np.random.choice(valid_wells, size=max_wells, replace=False)

        # Filter to selected wells
        df = df[df["well_id"].isin(valid_wells)]

        # Sort by well_id and date
        df = df.sort_values(["well_id", "date"]).reset_index(drop=True)

        logger.info(
            f"Loaded {df['well_id'].nunique()} wells with " f"{len(df)} total records"
        )

        return df

    def run_benchmark(self, save_results: bool = True) -> pd.DataFrame:
        """Run complete benchmark with all parameter sweeps.

        Args:
            save_results: Whether to save results to disk

        Returns:
            DataFrame with all benchmark results
        """
        all_results = []

        # Iterate over all parameter sweeps
        for sweep_idx, sweep in enumerate(self.config.parameter_sweeps):
            logger.info(f"Running parameter sweep: {sweep.name} ({sweep.type})")

            param_combos = sweep.generate_combinations()
            logger.info(f"Generated {len(param_combos)} parameter combinations")

            # Iterate over parameter combinations
            for combo_idx, param_combo in enumerate(param_combos):
                # Iterate over well configs
                for well_config in self.config.well_configs:
                    # Generate wells for this config
                    wells_df = self.generate_wells(
                        well_config, param_combo, self.config.n_wells_per_config
                    )

                    # Iterate over models and horizons
                    for model in self.config.models:
                        for horizon in self.config.horizons:
                            # Start MLflow run
                            run_name = (
                                f"{sweep.name}_combo{combo_idx}_"
                                f"{well_config.well_type}_{model}_h{horizon}"
                            )

                            mlflow_context = (
                                mlflow.start_run(run_name=run_name)
                                if self.config.mlflow_tracking and MLFLOW_AVAILABLE
                                else _null_context()
                            )

                            with mlflow_context:
                                if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                                    mlflow.log_params(param_combo)
                                    mlflow.log_params(
                                        {
                                            "well_type": well_config.well_type,
                                            "model": model,
                                            "horizon": str(horizon),
                                            "sweep_name": sweep.name,
                                            "sweep_type": sweep.type,
                                            "noise_level": str(well_config.noise_level),
                                        }
                                    )

                                try:
                                    arps_kind_map = {"arps": "hyperbolic"}
                                    results = dca.benchmark(
                                        wells_df,
                                        model=model,
                                        kind=arps_kind_map.get(model),
                                        horizon=horizon,
                                        verbose=False,
                                    )

                                    avg_metrics = {
                                        "rmse": results["rmse"].mean(),
                                        "mae": results["mae"].mean(),
                                        "smape": results["smape"].mean(),
                                    }

                                    if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                                        mlflow.log_metrics(avg_metrics)

                                    # Store results
                                    result_row = {
                                        "sweep_name": sweep.name,
                                        "sweep_idx": sweep_idx,
                                        "combo_idx": combo_idx,
                                        "well_type": well_config.well_type,
                                        "model": model,
                                        "horizon": horizon,
                                        "n_wells": len(results),
                                        **param_combo,
                                        **avg_metrics,
                                    }

                                    all_results.append(result_row)

                                    logger.info(
                                        f"Completed: {run_name} - "
                                        f"RMSE: {avg_metrics['rmse']:.2f}"
                                    )

                                except Exception as e:
                                    logger.error(
                                        "Benchmark failed for %s: %s", run_name, e
                                    )
                                    if self.config.mlflow_tracking and MLFLOW_AVAILABLE:
                                        try:
                                            mlflow.log_param("error", str(e))
                                        except Exception:
                                            pass

        results_df = pd.DataFrame(all_results)

        # Save results
        if save_results and self.config.output_dir:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path / "benchmark_results.csv", index=False)
            logger.info(f"Results saved to {output_path / 'benchmark_results.csv'}")

        return results_df
