"""Example usage of the BenchmarkFactory with MLflow tracking.

This example demonstrates:
1. Config-driven benchmark creation
2. Parameter sweeps (grid and random)
3. MLflow experiment tracking
4. Multiple well types and models
"""

import logging
from pathlib import Path

from decline_curve.benchmark_factory import BenchmarkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_from_config_file():
    """Run benchmark from YAML config file."""
    config_path = (
        Path(__file__).parent.parent / "decline_curve" / "benchmark_config_example.yaml"
    )

    factory = BenchmarkFactory(config_path)
    results = factory.run_benchmark(save_results=True)

    logger.info(f"Benchmark complete! Generated {len(results)} result rows")
    summary = results.groupby(["model", "horizon"])[["rmse", "mae"]].mean()
    logger.info(f"\nResults summary:\n{summary}")


def example_programmatic():
    """Create benchmark programmatically."""
    from decline_curve.benchmark_factory import (
        BenchmarkConfig,
        ParameterSweep,
        WellConfig,
    )

    config = BenchmarkConfig(
        experiment_name="programmatic_benchmark",
        well_configs=[
            WellConfig(
                well_type="hyperbolic",
                parameters={"qi": 1000, "di": 0.1, "b": 0.5},
                noise_level=0.05,
                n_months=36,
            )
        ],
        parameter_sweeps=[
            ParameterSweep(
                name="qi_sweep",
                type="grid",
                parameters={
                    "qi": [800, 1000, 1200],
                    "di": [0.08, 0.10, 0.12],
                },
            )
        ],
        models=["arps", "arima"],
        horizons=[12, 24],
        n_wells_per_config=3,
        seed=42,
        mlflow_tracking=True,
        output_dir="benchmark_results",
    )

    factory = BenchmarkFactory(config)
    results = factory.run_benchmark()

    logger.info(f"Results:\n{results.head()}")


if __name__ == "__main__":
    # Run from config file
    example_from_config_file()

    # Or create programmatically
    # example_programmatic()
