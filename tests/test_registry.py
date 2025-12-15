"""Tests for artifact registry module."""

import json
import tempfile
from pathlib import Path

import pytest

from decline_curve.artifacts import FitArtifact, create_fit_artifact
from decline_curve.fitting import FitResult, FitSpec
from decline_curve.models_arps import ExponentialArps
from decline_curve.registry import ArtifactRegistry


class TestArtifactRegistry:
    """Test ArtifactRegistry class."""

    def test_register_artifact(self):
        """Test registering an artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = str(Path(tmpdir) / "registry")
            registry = ArtifactRegistry(registry_path)

            model = ExponentialArps()
            spec = FitSpec(model=model)
            result = FitResult(
                params={"qi": 1000.0, "di": 0.1},
                model=model,
                success=True,
                message="OK",
                fit_start_idx=0,
                fit_end_idx=10,
            )

            artifact = create_fit_artifact("test_hash", spec, result)

            artifact_id = registry.register(artifact, well_id="WELL_001")

            assert artifact_id == artifact.metadata.run_id
            assert len(registry.index["artifacts"]) == 1

    def test_find_by_data_hash(self):
        """Test finding entries by data hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = str(Path(tmpdir) / "registry")
            registry = ArtifactRegistry(registry_path)

            model = ExponentialArps()
            spec = FitSpec(model=model)
            result = FitResult(
                params={"qi": 1000.0, "di": 0.1},
                model=model,
                success=True,
                message="OK",
                fit_start_idx=0,
                fit_end_idx=10,
            )

            artifact = create_fit_artifact("test_hash_123", spec, result)
            registry.register(artifact, well_id="WELL_001")

            prior = registry.find_prior_fit(data_hash="test_hash_123")

            assert prior is not None
            assert prior["data_hash"] == "test_hash_123"

    def test_compare_to_baseline(self):
        """Test comparing to baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = str(Path(tmpdir) / "registry")
            registry = ArtifactRegistry(registry_path)

            model = ExponentialArps()
            spec = FitSpec(model=model)

            result1 = FitResult(
                params={"qi": 1000.0, "di": 0.1},
                model=model,
                success=True,
                message="OK",
                fit_start_idx=0,
                fit_end_idx=10,
            )
            artifact1 = create_fit_artifact("hash1", spec, result1, run_id="run1")

            result2 = FitResult(
                params={"qi": 1200.0, "di": 0.1},
                model=model,
                success=True,
                message="OK",
                fit_start_idx=0,
                fit_end_idx=10,
            )
            artifact2 = create_fit_artifact("hash1", spec, result2, run_id="run2")

            # Register first artifact
            registry.register(artifact1, well_id="WELL_001")

            # Now compare second artifact to prior (should find run1)
            delta = registry.compare_to_prior(artifact2, well_id="WELL_001")

            assert delta is not None
            assert delta["new_artifact_id"] == "run2"
            assert delta["prior_artifact_id"] == "run1"
            assert len(delta["changes"]) > 0
