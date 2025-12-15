"""Artifact registry for tracking and comparing fits.

This module provides a registry system for storing fit artifacts with
stable IDs, enabling comparison of runs and delta reports.

Features:
- Store artifacts with stable IDs
- Compare new fit to prior fit
- Delta reports showing what changed
- Version tracking
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import FitArtifact, compare_artifacts, load_artifact
from .logging_config import get_logger

logger = get_logger(__name__)


class ArtifactRegistry:
    """Registry for storing and retrieving fit artifacts.

    Provides stable IDs and comparison capabilities for fits.
    """

    def __init__(self, registry_path: str = ".decline-curve_registry"):
        """Initialize artifact registry.

        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {"artifacts": []}

    def _save_index(self) -> None:
        """Save registry index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def register(self, artifact: FitArtifact, well_id: Optional[str] = None) -> str:
        """Register artifact in registry.

        Args:
            artifact: FitArtifact to register
            well_id: Optional well identifier

        Returns:
            Stable artifact ID
        """
        artifact_id = artifact.metadata.run_id

        # Save artifact file
        artifact_file = self.registry_path / f"{artifact_id}.json"
        artifact.save(str(artifact_file))

        # Add to index
        entry = {
            "artifact_id": artifact_id,
            "well_id": well_id,
            "data_hash": artifact.data_hash,
            "fit_spec_hash": artifact.fit_spec_hash,
            "package_version": artifact.metadata.package_version,
            "git_commit": artifact.metadata.git_commit,
            "timestamp": artifact.metadata.timestamp,
            "filepath": str(artifact_file),
        }

        self.index["artifacts"].append(entry)
        self._save_index()

        logger.info(
            f"Registered artifact: {artifact_id}",
            extra={"artifact_id": artifact_id, "well_id": well_id},
        )

        return artifact_id

    def find_prior_fit(
        self,
        well_id: Optional[str] = None,
        data_hash: Optional[str] = None,
    ) -> Optional[Dict]:
        """Find prior fit for well or data.

        Args:
            well_id: Well identifier
            data_hash: Data hash

        Returns:
            Prior artifact entry or None
        """
        for entry in reversed(self.index["artifacts"]):  # Most recent first
            if well_id and entry.get("well_id") == well_id:
                return entry
            if data_hash and entry.get("data_hash") == data_hash:
                return entry

        return None

    def compare_to_prior(
        self,
        new_artifact: FitArtifact,
        well_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Compare new artifact to prior fit.

        Args:
            new_artifact: New FitArtifact
            well_id: Optional well identifier

        Returns:
            Delta report or None if no prior fit found
        """
        prior_entry = self.find_prior_fit(
            well_id=well_id, data_hash=new_artifact.data_hash
        )

        if prior_entry is None:
            return None

        # Load prior artifact
        prior_artifact = load_artifact(prior_entry["filepath"])

        # Compare
        delta = compare_artifacts(prior_artifact, new_artifact.to_dict())
        delta["prior_artifact_id"] = prior_entry["artifact_id"]
        delta["new_artifact_id"] = new_artifact.metadata.run_id

        logger.info(
            f"Compared artifacts: {len(delta['changes'])} changes",
            extra={
                "prior_id": prior_entry["artifact_id"],
                "n_changes": len(delta["changes"]),
            },
        )

        return delta

    def list_artifacts(
        self,
        well_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """List artifacts in registry.

        Args:
            well_id: Filter by well ID
            limit: Maximum number to return

        Returns:
            List of artifact entries
        """
        artifacts = self.index["artifacts"]

        if well_id:
            artifacts = [a for a in artifacts if a.get("well_id") == well_id]

        if limit:
            artifacts = artifacts[-limit:]  # Most recent

        return artifacts
