"""Artifact and provenance system for reproducibility.

This module provides JSON artifact generation for fits and forecasts,
enabling full reproducibility from artifacts alone. Artifacts include:
- Input data hashes
- Spec hashes (FitSpec, SegmentSpec, OutlierSpec, etc.)
- Package version
- Git commit (when available)
- Random seeds
- All metadata needed for reproduction

References:
- Reproducible research best practices
- JSON Schema for artifact validation
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    import importlib.metadata

    PACKAGE_VERSION = importlib.metadata.version("decline-curve-analysis")
except Exception:
    try:
        import pkg_resources

        PACKAGE_VERSION = pkg_resources.get_distribution(
            "decline-curve-analysis"
        ).version
    except Exception:
        PACKAGE_VERSION = "unknown"


def compute_hash(data: Any) -> str:
    """Compute SHA256 hash of data.

    Args:
        data: Data to hash (will be serialized to JSON)

    Returns:
        Hexadecimal hash string
    """
    # Convert to JSON string for hashing
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_git_commit() -> Optional[str]:
    """Get current git commit hash.

    Returns:
        Git commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute hash of DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        Hexadecimal hash string
    """
    # Convert to JSON-serializable format
    data_dict = {
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
        "index": df.index.tolist(),
    }
    return compute_hash(data_dict)


def hash_array(arr: np.ndarray) -> str:
    """Compute hash of numpy array.

    Args:
        arr: Array to hash

    Returns:
        Hexadecimal hash string
    """
    # Convert to list for hashing
    data_dict = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tolist(),
    }
    return compute_hash(data_dict)


@dataclass
class ArtifactMetadata:
    """Metadata for artifact generation.

    Attributes:
        run_id: Stable run identifier
        timestamp: ISO format timestamp
        package_version: Package version
        git_commit: Git commit hash (if available)
        python_version: Python version
        platform: Platform information
    """

    run_id: str
    timestamp: str
    package_version: str
    git_commit: Optional[str] = None
    python_version: str = sys.version
    platform: str = sys.platform


@dataclass
class FitArtifact:
    """Artifact for a decline curve fit.

    Attributes:
        metadata: Artifact metadata
        data_hash: Hash of input data
        fit_spec: FitSpec as dictionary
        fit_spec_hash: Hash of FitSpec
        segment_spec: SegmentSpec as dictionary (optional)
        outlier_spec: OutlierSpec as dictionary (optional)
        fit_result: FitResult as dictionary
        diagnostics: DiagnosticsResult as dictionary (optional)
        warnings: List of warnings
        output_paths: Paths to output files
    """

    metadata: ArtifactMetadata
    data_hash: str
    fit_spec: Dict[str, Any]
    fit_spec_hash: str
    segment_spec: Optional[Dict[str, Any]] = None
    outlier_spec: Optional[Dict[str, Any]] = None
    fit_result: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert artifact to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        """Save artifact to JSON file.

        Args:
            filepath: Path to save artifact
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved fit artifact to {filepath}")


@dataclass
class ForecastArtifact:
    """Artifact for a forecast.

    Attributes:
        metadata: Artifact metadata
        fit_artifact_hash: Hash of associated fit artifact
        forecast_spec: ForecastSpec as dictionary
        forecast_spec_hash: Hash of ForecastSpec
        forecast_result: Forecast result as dictionary
        uncertainty: UncertaintyResult as dictionary (optional)
        output_paths: Paths to output files
    """

    metadata: ArtifactMetadata
    fit_artifact_hash: str
    forecast_spec: Dict[str, Any]
    forecast_spec_hash: str
    forecast_result: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[Dict[str, Any]] = None
    output_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert artifact to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        """Save artifact to JSON file.

        Args:
            filepath: Path to save artifact
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved forecast artifact to {filepath}")


def create_fit_artifact(
    data_hash: str,
    fit_spec: Any,
    fit_result: Any,
    segment_spec: Optional[Any] = None,
    outlier_spec: Optional[Any] = None,
    diagnostics: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> FitArtifact:
    """Create fit artifact from fit results.

    Args:
        data_hash: Hash of input data
        fit_spec: FitSpec object
        fit_result: FitResult object
        segment_spec: Optional SegmentSpec object
        outlier_spec: Optional OutlierSpec object
        diagnostics: Optional DiagnosticsResult object
        run_id: Optional run ID (auto-generated if None)

    Returns:
        FitArtifact object
    """
    # Generate run ID if not provided
    if run_id is None:
        run_id = f"fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create metadata
    metadata = ArtifactMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        package_version=PACKAGE_VERSION,
        git_commit=get_git_commit(),
    )

    # Convert specs to dictionaries
    try:
        from dataclasses import is_dataclass

        if is_dataclass(fit_spec):
            fit_spec_dict = asdict(fit_spec)
        elif hasattr(fit_spec, "__dict__"):
            fit_spec_dict = fit_spec.__dict__.copy()
        else:
            fit_spec_dict = fit_spec
    except Exception:
        if hasattr(fit_spec, "__dict__"):
            fit_spec_dict = fit_spec.__dict__.copy()
        else:
            fit_spec_dict = fit_spec

    # Handle model object in fit_spec
    if "model" in fit_spec_dict and hasattr(fit_spec_dict["model"], "name"):
        fit_spec_dict["model"] = {"name": fit_spec_dict["model"].name}

    fit_spec_hash = compute_hash(fit_spec_dict)

    segment_spec_dict = None
    if segment_spec is not None:
        try:
            from dataclasses import is_dataclass

            if is_dataclass(segment_spec):
                segment_spec_dict = asdict(segment_spec)
            elif hasattr(segment_spec, "__dict__"):
                segment_spec_dict = segment_spec.__dict__.copy()
            else:
                segment_spec_dict = segment_spec
        except Exception:
            if hasattr(segment_spec, "__dict__"):
                segment_spec_dict = segment_spec.__dict__.copy()
            else:
                segment_spec_dict = segment_spec

    outlier_spec_dict = None
    if outlier_spec is not None:
        try:
            from dataclasses import is_dataclass

            if is_dataclass(outlier_spec):
                outlier_spec_dict = asdict(outlier_spec)
            elif hasattr(outlier_spec, "__dict__"):
                outlier_spec_dict = outlier_spec.__dict__.copy()
            else:
                outlier_spec_dict = outlier_spec
        except Exception:
            if hasattr(outlier_spec, "__dict__"):
                outlier_spec_dict = outlier_spec.__dict__.copy()
            else:
                outlier_spec_dict = outlier_spec

    # Convert fit result to dictionary
    fit_result_dict = (
        fit_result.to_dict() if hasattr(fit_result, "to_dict") else fit_result
    )

    # Convert diagnostics to dictionary
    diagnostics_dict = None
    if diagnostics is not None:
        if hasattr(diagnostics, "__dict__"):
            diagnostics_dict = asdict(diagnostics)
        else:
            diagnostics_dict = diagnostics

    # Extract warnings
    warnings = []
    if hasattr(fit_result, "warnings"):
        warnings.extend(fit_result.warnings)
    if diagnostics is not None and hasattr(diagnostics, "warnings"):
        warnings.extend(diagnostics.warnings)

    artifact = FitArtifact(
        metadata=metadata,
        data_hash=data_hash,
        fit_spec=fit_spec_dict,
        fit_spec_hash=fit_spec_hash,
        segment_spec=segment_spec_dict,
        outlier_spec=outlier_spec_dict,
        fit_result=fit_result_dict,
        diagnostics=diagnostics_dict,
        warnings=warnings,
    )

    logger.info(
        f"Created fit artifact: {run_id}",
        extra={
            "run_id": run_id,
            "data_hash": data_hash[:8],
            "fit_spec_hash": fit_spec_hash[:8],
        },
    )

    return artifact


def create_forecast_artifact(
    fit_artifact_hash: str,
    forecast_spec: Any,
    forecast_result: Any,
    uncertainty: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> ForecastArtifact:
    """Create forecast artifact from forecast results.

    Args:
        fit_artifact_hash: Hash of associated fit artifact
        forecast_spec: ForecastSpec object
        forecast_result: Forecast result dictionary
        uncertainty: Optional UncertaintyResult object
        run_id: Optional run ID (auto-generated if None)

    Returns:
        ForecastArtifact object
    """
    # Generate run ID if not provided
    if run_id is None:
        run_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create metadata
    metadata = ArtifactMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        package_version=PACKAGE_VERSION,
        git_commit=get_git_commit(),
    )

    # Convert forecast spec to dictionary
    try:
        from dataclasses import is_dataclass

        if is_dataclass(forecast_spec):
            forecast_spec_dict = asdict(forecast_spec)
        elif hasattr(forecast_spec, "__dict__"):
            forecast_spec_dict = forecast_spec.__dict__.copy()
        else:
            forecast_spec_dict = forecast_spec
    except Exception:
        if hasattr(forecast_spec, "__dict__"):
            forecast_spec_dict = forecast_spec.__dict__.copy()
        else:
            forecast_spec_dict = forecast_spec

    forecast_spec_hash = compute_hash(forecast_spec_dict)

    # Convert uncertainty to dictionary
    uncertainty_dict = None
    if uncertainty is not None:
        if hasattr(uncertainty, "__dict__"):
            uncertainty_dict = asdict(uncertainty)
            # Convert numpy arrays to lists
            for key, value in uncertainty_dict.items():
                if isinstance(value, np.ndarray):
                    uncertainty_dict[key] = value.tolist()
        else:
            uncertainty_dict = uncertainty

    artifact = ForecastArtifact(
        metadata=metadata,
        fit_artifact_hash=fit_artifact_hash,
        forecast_spec=forecast_spec_dict,
        forecast_spec_hash=forecast_spec_hash,
        forecast_result=forecast_result,
        uncertainty=uncertainty_dict,
    )

    logger.info(
        f"Created forecast artifact: {run_id}",
        extra={"run_id": run_id, "fit_artifact_hash": fit_artifact_hash[:8]},
    )

    return artifact


def load_artifact(filepath: str) -> Dict[str, Any]:
    """Load artifact from JSON file.

    Args:
        filepath: Path to artifact file

    Returns:
        Artifact as dictionary
    """
    with open(filepath, "r") as f:
        artifact = json.load(f)

    logger.debug(f"Loaded artifact from {filepath}")
    return artifact


def compare_artifacts(
    artifact1: Dict[str, Any],
    artifact2: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two artifacts and generate delta report.

    Args:
        artifact1: First artifact (baseline)
        artifact2: Second artifact (comparison)

    Returns:
        Delta report dictionary
    """
    delta = {
        "data_changed": artifact1.get("data_hash") != artifact2.get("data_hash"),
        "fit_spec_changed": artifact1.get("fit_spec_hash")
        != artifact2.get("fit_spec_hash"),
        "package_version_changed": (
            artifact1.get("metadata", {}).get("package_version")
            != artifact2.get("metadata", {}).get("package_version")
        ),
        "git_commit_changed": (
            artifact1.get("metadata", {}).get("git_commit")
            != artifact2.get("metadata", {}).get("git_commit")
        ),
        "changes": [],
    }

    # Compare fit results
    if "fit_result" in artifact1 and "fit_result" in artifact2:
        params1 = artifact1["fit_result"].get("params", {})
        params2 = artifact2["fit_result"].get("params", {})

        for param in set(list(params1.keys()) + list(params2.keys())):
            val1 = params1.get(param)
            val2 = params2.get(param)
            if val1 != val2:
                delta["changes"].append(
                    {
                        "type": "parameter_change",
                        "parameter": param,
                        "old_value": val1,
                        "new_value": val2,
                    }
                )

    logger.info(
        f"Compared artifacts: {len(delta['changes'])} changes detected",
        extra={"n_changes": len(delta["changes"])},
    )

    return delta
