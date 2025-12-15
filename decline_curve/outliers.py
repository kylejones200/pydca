"""Outlier detection module for decline curve analysis.

This module provides methods to detect and handle outliers in production
data, with support for tail retention rules to preserve recent data.

Methods:
- IsolationForest (feature set: rate, delta rate, rolling median residual)
- Hampel filter (deterministic baseline)
- Z-score on log residual

Features:
- Tail retention rule (keep_last_k)
- Configurable detection thresholds
- Reason codes for outlier removals
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    logger.warning(
        "scikit-learn not available. Install with: pip install scikit-learn. "
        "IsolationForest outlier detection will be unavailable."
    )


@dataclass
class OutlierMask:
    """Outlier detection mask with reason codes.

    Attributes:
        is_outlier: Boolean array indicating outliers
        reason_codes: Array of reason codes for each point
        method: Detection method used
        n_outliers: Number of outliers detected
    """

    is_outlier: np.ndarray
    reason_codes: np.ndarray
    method: str
    n_outliers: int

    def apply_keep_last_k(self, keep_last_k: int = 0) -> "OutlierMask":
        """Apply tail retention rule.

        Never mark the last K points as outliers unless they fail
        hard constraints.

        Args:
            keep_last_k: Number of tail points to always keep

        Returns:
            New OutlierMask with tail retention applied
        """
        if keep_last_k <= 0:
            return self

        new_mask = self.is_outlier.copy()
        new_reason_codes = self.reason_codes.copy()

        # Unmark last K points unless reason is "hard_constraint"
        tail_start = len(new_mask) - keep_last_k
        if tail_start >= 0:
            for i in range(tail_start, len(new_mask)):
                if new_mask[i] and "hard_constraint" not in str(new_reason_codes[i]):
                    new_mask[i] = False
                    new_reason_codes[i] = "tail_retained"

        return OutlierMask(
            is_outlier=new_mask,
            reason_codes=new_reason_codes,
            method=self.method,
            n_outliers=int(new_mask.sum()),
        )


def detect_outliers_isolation_forest(
    rates: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
    window: int = 10,
) -> OutlierMask:
    """Detect outliers using IsolationForest.

    Uses feature set: rate, delta rate, rolling median residual.

    Args:
        rates: Production rate array
        contamination: Expected proportion of outliers
        random_state: Random seed for reproducibility
        window: Window size for rolling median

    Returns:
        OutlierMask with detected outliers
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for IsolationForest outlier detection")

    # Prepare features
    features = []

    # Feature 1: Rate (normalized)
    rate_norm = (rates - rates.mean()) / (rates.std() + 1e-10)
    features.append(rate_norm)

    # Feature 2: Delta rate (first difference)
    delta_rate = np.diff(rates, prepend=rates[0])
    delta_rate_norm = (delta_rate - delta_rate.mean()) / (delta_rate.std() + 1e-10)
    features.append(delta_rate_norm)

    # Feature 3: Rolling median residual (center=False to avoid future data leakage)
    rolling_median = (
        pd.Series(rates)
        .rolling(window=window, center=False)
        .median()
        .fillna(pd.Series(rates).median())
    )
    residual = rates - rolling_median.values
    residual_norm = (residual - residual.mean()) / (residual.std() + 1e-10)
    features.append(residual_norm)

    # Stack features
    X = np.column_stack(features)

    # Fit IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = iso_forest.fit_predict(X)

    # Convert to boolean (1 = inlier, -1 = outlier)
    is_outlier = predictions == -1

    # Generate reason codes
    reason_codes = np.array(["isolation_forest"] * len(rates))
    reason_codes[is_outlier] = "isolation_forest"

    logger.debug(
        f"IsolationForest detected {is_outlier.sum()} outliers",
        extra={"n_outliers": int(is_outlier.sum()), "contamination": contamination},
    )

    return OutlierMask(
        is_outlier=is_outlier,
        reason_codes=reason_codes,
        method="isolation_forest",
        n_outliers=int(is_outlier.sum()),
    )


def detect_outliers_hampel(
    rates: np.ndarray,
    window: int = 10,
    n_sigma: float = 3.0,
) -> OutlierMask:
    """Detect outliers using Hampel filter.

    Hampel filter uses median absolute deviation (MAD) to detect outliers
    relative to a rolling median baseline.

    Args:
        rates: Production rate array
        window: Window size for rolling median
        n_sigma: Number of standard deviations for threshold

    Returns:
        OutlierMask with detected outliers
    """
    # Compute rolling median (center=False to avoid future data leakage)
    rolling_median = (
        pd.Series(rates)
        .rolling(window=window, center=False)
        .median()
        .fillna(pd.Series(rates).median())
    )

    # Compute residuals
    residuals = rates - rolling_median.values

    # Compute MAD (Median Absolute Deviation)
    mad = np.median(np.abs(residuals - np.median(residuals)))

    # If MAD is zero or very small, use a small default threshold
    # This handles cases where data is perfectly smooth
    if mad < 1e-10:
        # Use a small fraction of the median rate as threshold
        threshold = n_sigma * 0.01 * np.median(np.abs(rates))
    else:
        # Threshold (using modified Z-score)
        threshold = (
            n_sigma * 1.4826 * mad
        )  # 1.4826 makes MAD comparable to std for normal dist

    # Detect outliers
    is_outlier = np.abs(residuals) > threshold

    # Generate reason codes
    reason_codes = np.array(["normal"] * len(rates))
    reason_codes[is_outlier] = "hampel_filter"

    logger.debug(
        f"Hampel filter detected {is_outlier.sum()} outliers",
        extra={"n_outliers": int(is_outlier.sum()), "threshold": threshold},
    )

    return OutlierMask(
        is_outlier=is_outlier,
        reason_codes=reason_codes,
        method="hampel_filter",
        n_outliers=int(is_outlier.sum()),
    )


def detect_outliers_zscore(
    rates: np.ndarray,
    window: int = 10,
    z_threshold: float = 3.0,
    use_log: bool = True,
) -> OutlierMask:
    """Detect outliers using Z-score on log residual.

    Args:
        rates: Production rate array
        window: Window size for rolling median baseline
        z_threshold: Z-score threshold
        use_log: If True, use log residual (better for multiplicative errors)

    Returns:
        OutlierMask with detected outliers
    """
    # Compute baseline (center=False to avoid future data leakage)
    rolling_median = (
        pd.Series(rates)
        .rolling(window=window, center=False)
        .median()
        .fillna(pd.Series(rates).median())
    )

    # Compute residuals
    if use_log:
        # Use log residual (multiplicative)
        rates_positive = np.maximum(
            rates, rates[rates > 0].min() if (rates > 0).any() else 1.0
        )
        baseline_positive = np.maximum(rolling_median.values, 1.0)
        residuals = np.log(rates_positive) - np.log(baseline_positive)
    else:
        # Use linear residual (additive)
        residuals = rates - rolling_median.values

    # Compute Z-scores
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    if residual_std > 0:
        z_scores = np.abs((residuals - residual_mean) / residual_std)
    else:
        z_scores = np.zeros_like(residuals)

    # Detect outliers
    is_outlier = z_scores > z_threshold

    # Generate reason codes
    reason_codes = np.array(["normal"] * len(rates))
    reason_codes[is_outlier] = "z_score"

    logger.debug(
        f"Z-score method detected {is_outlier.sum()} outliers",
        extra={"n_outliers": int(is_outlier.sum()), "z_threshold": z_threshold},
    )

    return OutlierMask(
        is_outlier=is_outlier,
        reason_codes=reason_codes,
        method="z_score",
        n_outliers=int(is_outlier.sum()),
    )


def detect_outliers_hard_constraints(
    rates: np.ndarray,
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None,
    max_increase: Optional[float] = None,
) -> OutlierMask:
    """Detect outliers using hard constraints.

    Hard constraints are always applied, even with tail retention.

    Args:
        rates: Production rate array
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
        max_increase: Maximum allowed increase from previous value

    Returns:
        OutlierMask with detected outliers
    """
    is_outlier = np.zeros(len(rates), dtype=bool)
    reason_codes = np.array(
        ["normal"] * len(rates), dtype="U50"
    )  # Allow longer strings

    # Initialize reason codes with proper dtype
    reason_codes = np.array(
        ["normal"] * len(rates), dtype="U50"
    )  # Allow longer strings

    # Check min/max rate
    if min_rate is not None:
        mask = rates < min_rate
        is_outlier[mask] = True
        reason_codes[mask] = "hard_constraint_min_rate"

    if max_rate is not None:
        mask = rates > max_rate
        is_outlier[mask] = True
        reason_codes[mask] = "hard_constraint_max_rate"

    # Check max increase
    if max_increase is not None:
        increases = np.diff(rates, prepend=rates[0])
        mask = increases > max_increase
        is_outlier[mask] = True
        reason_codes[mask] = "hard_constraint_max_increase"

    return OutlierMask(
        is_outlier=is_outlier,
        reason_codes=reason_codes,
        method="hard_constraints",
        n_outliers=int(is_outlier.sum()),
    )


def detect_outliers(
    rates: np.ndarray,
    method: Literal[
        "isolation_forest", "hampel", "z_score", "hard_constraints"
    ] = "hampel",
    keep_last_k: int = 0,
    **kwargs,
) -> OutlierMask:
    """Detect outliers in production data.

    Main entry point for outlier detection. Applies selected method and
    tail retention rule.

    Args:
        rates: Production rate array
        method: Detection method
        keep_last_k: Number of tail points to always keep
        **kwargs: Additional parameters for detection methods

    Returns:
        OutlierMask with detected outliers

    Example:
        >>> rates = np.array([100, 90, 80, 200, 70, 60, 50])  # Outlier at index 3
        >>> mask = detect_outliers(rates, method='hampel', keep_last_k=2)
        >>> print(f"Outliers: {mask.is_outlier}")
    """
    # Detect outliers
    if method == "isolation_forest":
        mask = detect_outliers_isolation_forest(
            rates,
            contamination=kwargs.get("contamination", 0.1),
            random_state=kwargs.get("random_state", 42),
            window=kwargs.get("window", 10),
        )
    elif method == "hampel":
        mask = detect_outliers_hampel(
            rates,
            window=kwargs.get("window", 10),
            n_sigma=kwargs.get("n_sigma", 3.0),
        )
    elif method == "z_score":
        mask = detect_outliers_zscore(
            rates,
            window=kwargs.get("window", 10),
            z_threshold=kwargs.get("z_threshold", 3.0),
            use_log=kwargs.get("use_log", True),
        )
    elif method == "hard_constraints":
        mask = detect_outliers_hard_constraints(
            rates,
            min_rate=kwargs.get("min_rate"),
            max_rate=kwargs.get("max_rate"),
            max_increase=kwargs.get("max_increase"),
        )
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # Apply tail retention
    if keep_last_k > 0:
        mask = mask.apply_keep_last_k(keep_last_k)

    logger.info(
        f"Outlier detection complete: {mask.n_outliers} outliers "
        f"detected using {method}",
        extra={
            "method": method,
            "n_outliers": mask.n_outliers,
            "keep_last_k": keep_last_k,
        },
    )

    return mask


def remove_outliers(
    rates: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    mask: Optional[OutlierMask] = None,
    **detect_kwargs,
) -> tuple[np.ndarray, Optional[pd.DatetimeIndex], OutlierMask]:
    """Remove outliers from production data.

    Convenience function that detects and removes outliers.

    Args:
        rates: Production rate array
        dates: Optional date index
        mask: Optional pre-computed outlier mask
        **detect_kwargs: Arguments for detect_outliers if mask not provided

    Returns:
        Tuple of (cleaned rates, cleaned dates, outlier mask)
    """
    if mask is None:
        mask = detect_outliers(rates, **detect_kwargs)

    # Remove outliers
    keep_mask = ~mask.is_outlier
    cleaned_rates = rates[keep_mask]

    if dates is not None:
        cleaned_dates = dates[keep_mask]
    else:
        cleaned_dates = None

    logger.info(
        f"Removed {mask.n_outliers} outliers, {len(cleaned_rates)} points remaining",
        extra={"n_removed": mask.n_outliers, "n_remaining": len(cleaned_rates)},
    )

    return cleaned_rates, cleaned_dates, mask
