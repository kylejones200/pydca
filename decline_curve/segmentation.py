"""Segmentation module for decline curve analysis.

This module provides methods to automatically detect regime changes in
production data, enabling selection of the most appropriate segment for
decline curve fitting.

Methods:
- PELT (Pruned Exact Linear Time) on log rate
- PELT on rate derivative
- CUSUM on residual of smooth baseline
- None (use full history)

References:
- ruptures library: https://github.com/deepcharles/ruptures
- CUSUM: Cumulative Sum Control Chart
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    import ruptures as rpt

    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    rpt = None
    logger.warning(
        "ruptures not available. Install with: pip install ruptures. "
        "PELT segmentation will be unavailable."
    )


@dataclass
class Segment:
    """A production segment identified by segmentation.

    Attributes:
        start_idx: Start index in original data
        end_idx: End index in original data (exclusive)
        start_date: Start date (if available)
        end_date: End date (if available)
        mean_rate: Mean rate in segment
        stability_score: Stability score (higher = more stable)
    """

    start_idx: int
    end_idx: int
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    mean_rate: float = 0.0
    stability_score: float = 0.0


@dataclass
class SegmentationResult:
    """Result of segmentation analysis.

    Attributes:
        segments: List of identified segments
        change_points: List of change point indices
        method: Method used for segmentation
        selected_segment_idx: Index of selected segment for fitting
    """

    segments: list[Segment]
    change_points: list[int]
    method: str
    selected_segment_idx: int = -1

    def get_selected_segment(self) -> Optional[Segment]:
        """Get the selected segment for fitting."""
        if self.selected_segment_idx >= 0 and self.selected_segment_idx < len(
            self.segments
        ):
            return self.segments[self.selected_segment_idx]
        return None


def segment_pelt_log_rate(
    rates: np.ndarray,
    min_size: int = 3,
    penalty: float = 10.0,
) -> list[int]:
    """Segment using PELT on log rate.

    PELT (Pruned Exact Linear Time) detects change points by minimizing
    a cost function. Using log rate helps detect multiplicative changes.

    Args:
        rates: Production rate array
        min_size: Minimum segment size
        penalty: Penalty parameter (higher = fewer change points)

    Returns:
        List of change point indices
    """
    if not RUPTURES_AVAILABLE:
        raise ImportError("ruptures library required for PELT segmentation")

    # Convert to log, handling zeros
    log_rates = np.log(
        np.maximum(rates, rates[rates > 0].min() if (rates > 0).any() else 1.0)
    )

    # Reshape for ruptures (needs 2D)
    log_rates_2d = log_rates.reshape(-1, 1)

    # Run PELT
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(log_rates_2d)
    change_points = algo.predict(pen=penalty)

    # Remove last point (always included)
    if len(change_points) > 0 and change_points[-1] == len(rates):
        change_points = change_points[:-1]

    return change_points.tolist()


def segment_pelt_derivative(
    rates: np.ndarray,
    min_size: int = 3,
    penalty: float = 10.0,
) -> list[int]:
    """Segment using PELT on rate derivative.

    Detects change points in the rate of change, useful for identifying
    transitions between decline regimes.

    Args:
        rates: Production rate array
        min_size: Minimum segment size
        penalty: Penalty parameter

    Returns:
        List of change point indices
    """
    if not RUPTURES_AVAILABLE:
        raise ImportError("ruptures library required for PELT segmentation")

    # Compute derivative (first difference)
    derivative = np.diff(rates)

    # Reshape for ruptures
    derivative_2d = derivative.reshape(-1, 1)

    # Run PELT
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(derivative_2d)
    change_points = algo.predict(pen=penalty)

    # Adjust indices (derivative is one shorter)
    change_points = change_points + 1

    # Remove last point if needed
    if len(change_points) > 0 and change_points[-1] >= len(rates):
        change_points = change_points[:-1]

    return change_points.tolist()


def segment_cusum(
    rates: np.ndarray,
    baseline_window: int = 10,
    threshold: float = 3.0,
) -> list[int]:
    """Segment using CUSUM on residual of smooth baseline.

    CUSUM (Cumulative Sum) detects changes by tracking cumulative deviations
    from a baseline. Detects change points by looking for significant shifts
    in the rate values themselves.

    Args:
        rates: Production rate array
        baseline_window: Window size for detecting changes
        threshold: Z-score threshold for change detection (lower = more sensitive)

    Returns:
        List of change point indices
    """
    if len(rates) < baseline_window * 2:
        return []

    change_points = []
    skip_until = 0

    # Slide a window and compare statistics before/after each point
    for i in range(baseline_window, len(rates) - baseline_window):
        # Skip if we're in a cooldown period after detecting a change
        if i < skip_until:
            continue

        # Get windows before and after this point
        before = rates[max(0, i - baseline_window) : i]
        after = rates[i : min(len(rates), i + baseline_window)]

        if len(before) > 0 and len(after) > 0:
            # Calculate means
            mean_before = np.mean(before)
            mean_after = np.mean(after)

            # Calculate pooled standard deviation
            std_before = np.std(before)
            std_after = np.std(after)
            pooled_std = np.sqrt(
                (std_before**2 + std_after**2) / 2 + 1e-10
            )  # Add small constant to avoid div by zero

            # Calculate z-score of difference in means
            z_score = abs(mean_before - mean_after) / pooled_std

            if z_score > threshold:
                change_points.append(i)
                # Skip ahead to avoid detecting the same change multiple times
                skip_until = i + baseline_window

    return change_points


def create_segments_from_change_points(
    change_points: list[int],
    rates: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> list[Segment]:
    """Create Segment objects from change points.

    Args:
        change_points: List of change point indices
        rates: Production rate array
        dates: Optional date index

    Returns:
        List of Segment objects
    """
    segments = []

    # Add start and end
    all_points = [0] + sorted(change_points) + [len(rates)]

    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]

        segment_rates = rates[start_idx:end_idx]
        mean_rate = np.mean(segment_rates) if len(segment_rates) > 0 else 0.0

        # Compute stability score (inverse of coefficient of variation)
        if mean_rate > 0 and len(segment_rates) > 1:
            cv = np.std(segment_rates) / mean_rate
            stability_score = 1.0 / (1.0 + cv)  # Higher = more stable
        else:
            stability_score = 0.0

        segment = Segment(
            start_idx=start_idx,
            end_idx=end_idx,
            start_date=(
                dates[start_idx]
                if dates is not None and start_idx < len(dates)
                else None
            ),
            end_date=(
                dates[end_idx - 1]
                if dates is not None and end_idx > 0 and end_idx <= len(dates)
                else None
            ),
            mean_rate=mean_rate,
            stability_score=stability_score,
        )
        segments.append(segment)

    return segments


def select_final_stable_segment(
    segments: list[Segment],
    min_stability: float = 0.5,
) -> int:
    """Select the final stable segment.

    Default selection rule: Choose the last segment that meets minimum
    stability threshold, or the last segment if none meet threshold.

    Args:
        segments: List of segments
        min_stability: Minimum stability score threshold

    Returns:
        Index of selected segment
    """
    # Start from the end and work backwards
    for i in range(len(segments) - 1, -1, -1):
        if segments[i].stability_score >= min_stability:
            return i

    # If none meet threshold, return last segment
    return len(segments) - 1


def select_segment_by_aic(
    segments: list[Segment],
    rates: np.ndarray,
    penalty_per_point: float = 2.0,
) -> int:
    """Select segment that minimizes AIC + penalty for short segments.

    Alternate selection rule: Choose segment with best AIC score,
    penalizing very short segments.

    Args:
        segments: List of segments
        rates: Production rate array
        penalty_per_point: Penalty per point for short segments

    Returns:
        Index of selected segment
    """
    best_idx = 0
    best_score = np.inf

    for i, segment in enumerate(segments):
        segment_rates = rates[segment.start_idx : segment.end_idx]

        if len(segment_rates) < 3:
            continue  # Skip very short segments

        # Simple AIC approximation: -2*log_likelihood + 2*k
        # Using variance as proxy for likelihood
        variance = np.var(segment_rates)
        if variance > 0:
            log_likelihood = (
                -0.5 * len(segment_rates) * (np.log(2 * np.pi * variance) + 1)
            )
            aic = -2 * log_likelihood + 2 * 3  # 3 parameters (qi, di, b)
        else:
            aic = 0

        # Penalty for short segments
        length_penalty = penalty_per_point * max(0, 10 - len(segment_rates))

        score = aic + length_penalty

        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def segment_production_data(
    rates: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    method: Literal["pelt_log", "pelt_derivative", "cusum", "none"] = "pelt_log",
    selection_rule: Literal["final_stable", "aic"] = "final_stable",
    **kwargs,
) -> SegmentationResult:
    """Segment production data to identify decline regimes.

    Main entry point for segmentation. Detects change points and selects
    the appropriate segment for decline curve fitting.

    Args:
        rates: Production rate array
        dates: Optional date index
        method: Segmentation method
        selection_rule: Segment selection rule
        **kwargs: Additional parameters for segmentation methods

    Returns:
        SegmentationResult with identified segments and selection

    Example:
        >>> rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        >>> result = segment_production_data(rates, method='pelt_log')
        >>> selected = result.get_selected_segment()
        >>> print(f"Selected segment: {selected.start_idx} to {selected.end_idx}")
    """
    # Run segmentation
    if method == "pelt_log":
        if not RUPTURES_AVAILABLE:
            logger.warning("ruptures not available, falling back to 'none'")
            method = "none"
        else:
            change_points = segment_pelt_log_rate(
                rates,
                min_size=kwargs.get("min_size", 3),
                penalty=kwargs.get("penalty", 10.0),
            )
    elif method == "pelt_derivative":
        if not RUPTURES_AVAILABLE:
            logger.warning("ruptures not available, falling back to 'none'")
            method = "none"
        else:
            change_points = segment_pelt_derivative(
                rates,
                min_size=kwargs.get("min_size", 3),
                penalty=kwargs.get("penalty", 10.0),
            )
    elif method == "cusum":
        change_points = segment_cusum(
            rates,
            baseline_window=kwargs.get("baseline_window", 10),
            threshold=kwargs.get("threshold", 3.0),
        )
    elif method == "none":
        change_points = []
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    # Create segments
    segments = create_segments_from_change_points(change_points, rates, dates)

    # Select segment
    if selection_rule == "final_stable":
        selected_idx = select_final_stable_segment(
            segments, min_stability=kwargs.get("min_stability", 0.5)
        )
    elif selection_rule == "aic":
        selected_idx = select_segment_by_aic(
            segments, rates, penalty_per_point=kwargs.get("penalty_per_point", 2.0)
        )
    else:
        raise ValueError(f"Unknown selection rule: {selection_rule}")

    logger.info(
        f"Segmentation complete: {len(segments)} segments identified, "
        f"selected segment {selected_idx}",
        extra={
            "method": method,
            "selection_rule": selection_rule,
            "n_segments": len(segments),
        },
    )

    return SegmentationResult(
        segments=segments,
        change_points=change_points,
        method=method,
        selected_segment_idx=selected_idx,
    )
