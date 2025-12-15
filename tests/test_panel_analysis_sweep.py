"""Tests for panel data analysis sweep system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from decline_curve.panel_analysis_sweep import (
    PanelAnalysisConfig,
    PanelAnalysisSweep,
    ParameterSweep,
)


class TestParameterSweep:
    """Test parameter sweep generation."""

    def test_grid_sweep(self):
        """Test grid search parameter sweep."""
        sweep = ParameterSweep(
            name="test_grid",
            type="grid",
            parameters={"include_company": [True, False], "max_wells": [100, 200]},
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 4  # 2 x 2 = 4 combinations
        assert {"include_company": True, "max_wells": 100} in combos
        assert {"include_company": False, "max_wells": 200} in combos

    def test_random_sweep(self):
        """Test random search parameter sweep."""
        sweep = ParameterSweep(
            name="test_random",
            type="random",
            n_samples=10,
            seed=42,
            parameters={
                "max_wells": {
                    "type": "choice",
                    "values": [100, 200, 500],
                },
                "include_company": {"type": "boolean"},
            },
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 10
        assert all("max_wells" in combo for combo in combos)
        assert all("include_company" in combo for combo in combos)

    def test_custom_sweep(self):
        """Test custom parameter sweep."""
        custom_params = [
            {"include_company": True, "include_county": False},
            {"include_company": False, "include_county": True},
        ]

        sweep = ParameterSweep(
            name="test_custom", type="custom", parameters=custom_params
        )

        combos = sweep.generate_combinations()
        assert len(combos) == 2
        assert combos == custom_params


class TestPanelAnalysisConfig:
    """Test panel analysis configuration."""

    def test_config_creation(self):
        """Test creating a configuration."""
        config = PanelAnalysisConfig(
            experiment_name="test",
            data_path="data/test.parquet",
            max_wells=100,
            include_company=True,
            include_county=True,
        )

        assert config.experiment_name == "test"
        assert config.data_path == "data/test.parquet"
        assert config.max_wells == 100
        assert config.include_company is True
        assert config.include_county is True


class TestPanelAnalysisSweep:
    """Test panel analysis sweep system."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        # Create temporary config file
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/test.parquet",
            "max_wells": 100,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name

        try:
            sweep = PanelAnalysisSweep(config_path)
            assert sweep.base_config.experiment_name == "test"
            assert sweep.base_config.data_path == "data/test.parquet"
            assert sweep.base_config.max_wells == 100
        finally:
            Path(config_path).unlink()

    def test_load_dict_config(self):
        """Test loading dictionary configuration."""
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/test.parquet",
            "max_wells": 100,
        }

        sweep = PanelAnalysisSweep(config_dict)
        assert sweep.base_config.experiment_name == "test"

    def test_sweep_without_parameter_sweeps(self):
        """Test running sweep without parameter sweeps (base config only)."""
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/north_dakota_production.parquet",
            "max_wells": 10,  # Small sample for testing
            "mlflow_tracking": False,
        }

        sweep = PanelAnalysisSweep(config_dict)

        # This will fail if data file doesn't exist, but we can test the structure
        assert sweep.base_config.experiment_name == "test"
        assert len(sweep.parameter_sweeps) == 0

    def test_sweep_with_parameter_sweeps(self):
        """Test sweep with parameter sweeps."""
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/north_dakota_production.parquet",
            "max_wells": 10,
            "mlflow_tracking": False,
            "parameter_sweeps": [
                {
                    "name": "test_sweep",
                    "type": "grid",
                    "parameters": {
                        "include_company": [True, False],
                    },
                }
            ],
        }

        sweep = PanelAnalysisSweep(config_dict)
        assert len(sweep.parameter_sweeps) == 1
        assert sweep.parameter_sweeps[0].name == "test_sweep"


class TestIntegration:
    """Integration tests for panel analysis sweep."""

    @pytest.mark.skipif(
        not Path("data/north_dakota_production.parquet").exists(),
        reason="Real data file not found",
    )
    def test_run_single_analysis(self):
        """Test running a single analysis."""
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/north_dakota_production.parquet",
            "max_wells": 50,  # Small sample for testing
            "mlflow_tracking": False,
            "include_company": True,
            "include_county": True,
        }

        sweep = PanelAnalysisSweep(config_dict)
        result = sweep.run_single_analysis(sweep.base_config)

        assert "n_wells" in result
        assert "mean_eur" in result
        assert result["n_wells"] > 0

    @pytest.mark.skipif(
        not Path("data/north_dakota_production.parquet").exists(),
        reason="Real data file not found",
    )
    def test_run_small_sweep(self):
        """Test running a small parameter sweep."""
        config_dict = {
            "experiment_name": "test",
            "data_path": "data/north_dakota_production.parquet",
            "max_wells": 50,
            "mlflow_tracking": False,
            "output_dir": None,
            "parameter_sweeps": [
                {
                    "name": "test_sweep",
                    "type": "grid",
                    "parameters": {
                        "include_company": [True, False],
                    },
                }
            ],
        }

        sweep = PanelAnalysisSweep(config_dict)
        results_df = sweep.run_sweep(save_results=False)

        assert len(results_df) > 0
        assert "rsquared" in results_df.columns or "error" in results_df.columns
        assert "include_company" in results_df.columns
