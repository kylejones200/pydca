"""Tests for artifacts module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decline_curve.artifacts import (
    ArtifactMetadata,
    FitArtifact,
    ForecastArtifact,
    compare_artifacts,
    compute_hash,
    create_fit_artifact,
    create_forecast_artifact,
    hash_array,
    hash_dataframe,
    load_artifact,
)
from decline_curve.fitting import FitResult
from decline_curve.models_arps import ExponentialArps


class TestHashFunctions:
    """Test hashing functions."""

    def test_compute_hash(self):
        """Test hash computation."""
        data = {"key": "value", "number": 42}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_dataframe(self):
        """Test DataFrame hashing."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash1 = hash_dataframe(df)
        hash2 = hash_dataframe(df)

        assert hash1 == hash2

    def test_hash_array(self):
        """Test array hashing."""
        arr = np.array([1, 2, 3, 4, 5])
        hash1 = hash_array(arr)
        hash2 = hash_array(arr)

        assert hash1 == hash2


class TestCreateFitArtifact:
    """Test fit artifact creation."""

    def test_create_fit_artifact(self):
        """Test creating fit artifact."""
        from decline_curve.fitting import FitSpec

        model = ExponentialArps()
        fit_spec = FitSpec(model=model)
        fit_result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        data_hash = "abc123"
        artifact = create_fit_artifact(data_hash, fit_spec, fit_result)

        assert artifact.data_hash == data_hash
        assert artifact.metadata.run_id is not None
        assert artifact.fit_spec_hash is not None
        assert "fit_result" in artifact.to_dict()

    def test_artifact_save_and_load(self):
        """Test saving and loading artifact."""
        from decline_curve.fitting import FitSpec

        model = ExponentialArps()
        fit_spec = FitSpec(model=model)
        fit_result = FitResult(
            params={"qi": 1000.0, "di": 0.1},
            model=model,
            success=True,
            message="OK",
            fit_start_idx=0,
            fit_end_idx=10,
        )

        artifact = create_fit_artifact("test_hash", fit_spec, fit_result)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            artifact.save(filepath)
            loaded = load_artifact(filepath)

            assert loaded["data_hash"] == artifact.data_hash
            assert loaded["metadata"]["run_id"] == artifact.metadata.run_id
        finally:
            Path(filepath).unlink()


class TestCompareArtifacts:
    """Test artifact comparison."""

    def test_compare_identical_artifacts(self):
        """Test comparing identical artifacts."""
        artifact1 = {
            "data_hash": "abc123",
            "fit_spec_hash": "def456",
            "metadata": {"package_version": "1.0.0"},
            "fit_result": {"params": {"qi": 1000.0, "di": 0.1}},
        }
        artifact2 = artifact1.copy()

        delta = compare_artifacts(artifact1, artifact2)

        assert not delta["data_changed"]
        assert not delta["fit_spec_changed"]
        assert len(delta["changes"]) == 0

    def test_compare_different_artifacts(self):
        """Test comparing different artifacts."""
        artifact1 = {
            "data_hash": "abc123",
            "fit_spec_hash": "def456",
            "metadata": {"package_version": "1.0.0"},
            "fit_result": {"params": {"qi": 1000.0, "di": 0.1}},
        }
        artifact2 = {
            "data_hash": "xyz789",
            "fit_spec_hash": "def456",
            "metadata": {"package_version": "1.0.0"},
            "fit_result": {"params": {"qi": 1200.0, "di": 0.1}},
        }

        delta = compare_artifacts(artifact1, artifact2)

        assert delta["data_changed"]
        assert len(delta["changes"]) > 0
