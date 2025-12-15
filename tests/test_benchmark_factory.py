"""Tests for benchmark factory."""

import tempfile
from pathlib import Path

import pandas as pd

from decline_curve.benchmark_factory import (
    BenchmarkConfig,
    BenchmarkFactory,
    ParameterSweep,
    WellConfig,
)


class TestParameterSweep:
    """Test parameter sweep generation."""

    def test_grid_search(self):
        """Test grid search parameter generation."""
        sweep = ParameterSweep(
            name="test_grid",
            type="grid",
            parameters={"qi": [1000, 1200], "di": [0.1, 0.15]},
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 4  # 2 * 2
        assert {"qi": 1000, "di": 0.1} in combos
        assert {"qi": 1200, "di": 0.15} in combos

    def test_random_search(self):
        """Test random search parameter generation."""
        sweep = ParameterSweep(
            name="test_random",
            type="random",
            n_samples=10,
            seed=42,
            parameters={
                "qi": {"type": "uniform", "min": 800, "max": 1500},
                "di": {"type": "uniform", "min": 0.05, "max": 0.20},
            },
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 10

        # Check bounds
        for combo in combos:
            assert 800 <= combo["qi"] <= 1500
            assert 0.05 <= combo["di"] <= 0.20

    def test_custom_combinations(self):
        """Test custom parameter combinations."""
        sweep = ParameterSweep(
            name="test_custom",
            type="custom",
            parameters=[
                {"qi": 1000, "di": 0.1},
                {"qi": 1200, "di": 0.15},
            ],
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 2
        assert combos[0] == {"qi": 1000, "di": 0.1}


class TestBenchmarkFactory:
    """Test benchmark factory."""

    def test_factory_from_dict(self):
        """Test factory initialization from dict."""
        config_dict = {
            "experiment_name": "test_exp",
            "well_configs": [
                {
                    "well_type": "hyperbolic",
                    "parameters": {"qi": 1000, "di": 0.1, "b": 0.5},
                    "n_months": 24,
                }
            ],
            "parameter_sweeps": [
                {
                    "name": "test_sweep",
                    "type": "grid",
                    "parameters": {"qi": [1000, 1200]},
                }
            ],
            "models": ["arps"],
            "horizons": [12],
            "n_wells_per_config": 1,
            "seed": 42,
            "mlflow_tracking": False,
        }

        factory = BenchmarkFactory(config_dict)
        assert factory.config.experiment_name == "test_exp"
        assert len(factory.config.well_configs) == 1
        assert len(factory.config.parameter_sweeps) == 1

    def test_factory_from_config_object(self):
        """Test factory initialization from BenchmarkConfig."""
        config = BenchmarkConfig(
            experiment_name="test_exp",
            well_configs=[
                WellConfig(
                    well_type="hyperbolic",
                    parameters={"qi": 1000, "di": 0.1, "b": 0.5},
                    n_months=24,
                )
            ],
            parameter_sweeps=[
                ParameterSweep(
                    name="test_sweep",
                    type="grid",
                    parameters={"qi": [1000, 1200]},
                )
            ],
            models=["arps"],
            horizons=[12],
            n_wells_per_config=1,
            seed=42,
            mlflow_tracking=False,
        )

        factory = BenchmarkFactory(config)
        assert factory.config.experiment_name == "test_exp"

    def test_generate_wells(self):
        """Test well generation."""
        config = BenchmarkConfig(
            experiment_name="test",
            well_configs=[
                WellConfig(
                    well_type="hyperbolic",
                    parameters={"qi": 1000, "di": 0.1, "b": 0.5},
                    n_months=24,
                )
            ],
            parameter_sweeps=[
                ParameterSweep(
                    name="test",
                    type="grid",
                    parameters={},
                )
            ],
            mlflow_tracking=False,
        )

        factory = BenchmarkFactory(config)
        wells_df = factory.generate_wells(config.well_configs[0], {}, n_wells=2)

        assert isinstance(wells_df, pd.DataFrame)
        assert "well_id" in wells_df.columns
        assert "date" in wells_df.columns
        assert "oil_bbl" in wells_df.columns
        assert wells_df["well_id"].nunique() == 2
        assert len(wells_df) == 2 * 24  # 2 wells * 24 months

    def test_run_benchmark_small(self):
        """Test running a small benchmark."""
        config = BenchmarkConfig(
            experiment_name="test_benchmark",
            well_configs=[
                WellConfig(
                    well_type="hyperbolic",
                    parameters={"qi": 1000, "di": 0.1, "b": 0.5},
                    n_months=24,
                )
            ],
            parameter_sweeps=[
                ParameterSweep(
                    name="small_sweep",
                    type="grid",
                    parameters={"qi": [1000]},
                )
            ],
            models=["arps"],
            horizons=[12],
            n_wells_per_config=1,
            seed=42,
            mlflow_tracking=False,
            output_dir=None,
        )

        factory = BenchmarkFactory(config)
        results = factory.run_benchmark(save_results=False)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "rmse" in results.columns
        assert "mae" in results.columns
        assert "smape" in results.columns

    def test_run_benchmark_with_output(self):
        """Test benchmark with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                experiment_name="test_benchmark",
                well_configs=[
                    WellConfig(
                        well_type="hyperbolic",
                        parameters={"qi": 1000, "di": 0.1, "b": 0.5},
                        n_months=24,
                    )
                ],
                parameter_sweeps=[
                    ParameterSweep(
                        name="small_sweep",
                        type="grid",
                        parameters={"qi": [1000]},
                    )
                ],
                models=["arps"],
                horizons=[12],
                n_wells_per_config=1,
                seed=42,
                mlflow_tracking=False,
                output_dir=tmpdir,
            )

            factory = BenchmarkFactory(config)
            results = factory.run_benchmark(save_results=True)

            output_file = Path(tmpdir) / "benchmark_results.csv"
            assert output_file.exists()
            assert len(results) > 0
