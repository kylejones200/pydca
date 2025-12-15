"""Test utilities for decline curve analysis.

This module provides synthetic data generators and property test helpers
for comprehensive testing of DCA models and pipelines.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .logging_config import get_logger
from .models_base import Model

logger = get_logger(__name__)


def generate_synthetic_arps_data(
    qi: float,
    di: float,
    b: float,
    t_max: float = 100.0,
    dt: float = 1.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    model_type: str = "hyperbolic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Arps decline data with noise.

    Args:
        qi: Initial rate
        di: Initial decline rate
        b: Decline exponent
        t_max: Maximum time
        dt: Time step
        noise_level: Relative noise level (fraction of rate)
        seed: Random seed for reproducibility
        model_type: Type of Arps model ('exponential', 'harmonic', 'hyperbolic')

    Returns:
        Tuple of (time array, rate array with noise)

    Example:
        >>> t, q = generate_synthetic_arps_data(1000, 0.1, 0.5, seed=42)
        >>> assert len(t) == len(q)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(0, t_max + dt, dt)

    # Generate true rates based on model type
    if model_type == "exponential":
        q_true = qi * np.exp(-di * t)
    elif model_type == "harmonic":
        q_true = qi / (1 + di * t)
    elif model_type == "hyperbolic":
        q_true = qi / (1 + b * di * t) ** (1 / b)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Add noise
    noise = np.random.normal(0, noise_level * q_true)
    q = np.maximum(q_true + noise, 0)  # Ensure non-negative

    return t, q


def generate_piecewise_decline(
    segments: List[Dict[str, float]],
    t_max: float = 100.0,
    dt: float = 1.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate piecewise decline with regime changes.

    Args:
        segments: List of segment dicts with keys: qi, di, b, duration
        t_max: Maximum time
        dt: Time step
        noise_level: Relative noise level
        seed: Random seed

    Returns:
        Tuple of (time array, rate array)

    Example:
        >>> segments = [
        ...     {"qi": 1000, "di": 0.1, "b": 0.5, "duration": 30},
        ...     {"qi": 500, "di": 0.15, "b": 0.6, "duration": 70}
        ... ]
        >>> t, q = generate_piecewise_decline(segments, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(0, t_max + dt, dt)
    q = np.zeros_like(t)

    t_current = 0.0
    q_current = segments[0]["qi"]

    for segment in segments:
        qi = segment["qi"]
        di = segment["di"]
        b = segment.get("b", 0.0)  # Default to exponential
        duration = segment["duration"]

        # Find indices for this segment
        t_start = t_current
        t_end = min(t_current + duration, t_max)
        idx_start = np.searchsorted(t, t_start)
        idx_end = np.searchsorted(t, t_end)

        # Generate rates for this segment
        t_seg = t[idx_start:idx_end] - t_start
        if b == 0:
            q_seg = qi * np.exp(-di * t_seg)
        else:
            q_seg = qi / (1 + b * di * t_seg) ** (1 / b)

        # Scale to match previous segment's end
        if idx_start > 0:
            scale = q[idx_start - 1] / q_seg[0] if q_seg[0] > 0 else 1.0
            q_seg = q_seg * scale

        q[idx_start:idx_end] = q_seg
        t_current = t_end
        q_current = q_seg[-1] if len(q_seg) > 0 else q_current

    # Add noise
    noise = np.random.normal(0, noise_level * q)
    q = np.maximum(q + noise, 0)

    return t, q


def generate_ramp_up_data(
    qi_final: float,
    ramp_duration: float,
    decline_di: float,
    decline_b: float,
    t_max: float = 100.0,
    dt: float = 1.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data with ramp-up period followed by decline.

    Args:
        qi_final: Final initial rate after ramp-up
        ramp_duration: Duration of ramp-up period
        decline_di: Decline rate after ramp-up
        decline_b: Decline exponent
        t_max: Maximum time
        dt: Time step
        noise_level: Relative noise level
        seed: Random seed

    Returns:
        Tuple of (time array, rate array)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(0, t_max + dt, dt)
    q = np.zeros_like(t)

    # Ramp-up: linear increase
    idx_ramp = t < ramp_duration
    q[idx_ramp] = qi_final * (t[idx_ramp] / ramp_duration)

    # Decline after ramp-up
    idx_decline = ~idx_ramp
    t_decline = t[idx_decline] - ramp_duration
    if decline_b == 0:
        q[idx_decline] = qi_final * np.exp(-decline_di * t_decline)
    else:
        q[idx_decline] = qi_final / (1 + decline_b * decline_di * t_decline) ** (
            1 / decline_b
        )

    # Add noise
    noise = np.random.normal(0, noise_level * q)
    q = np.maximum(q + noise, 0)

    return t, q


def check_model_properties(
    model: Model,
    params: Dict[str, float],
    t: np.ndarray,
) -> Dict[str, bool]:
    """Check mathematical properties of model output.

    Args:
        model: Model instance
        params: Model parameters
        t: Time array

    Returns:
        Dictionary of property checks (all should be True)
    """
    # Validate parameters
    try:
        model.validate(params)
    except Exception as e:
        return {"valid_params": False, "error": str(e)}

    # Generate rates and cumulative
    q = model.rate(t, params)
    cum = model.cum(t, params)

    checks = {
        "valid_params": True,
        "non_negative_rates": np.all(q >= 0),
        "non_negative_cumulative": np.all(cum >= 0),
        "monotone_cumulative": np.all(np.diff(cum) >= 0),
        "finite_values": np.all(np.isfinite(q)) and np.all(np.isfinite(cum)),
    }

    # Check monotone decline (after first point)
    if len(q) > 1:
        # Allow small numerical errors
        checks["monotone_decline"] = np.all(np.diff(q) <= 1e-10)

    return checks


def create_baseline_dataset(
    n_wells: int = 10,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Create baseline synthetic dataset for testing.

    Args:
        n_wells: Number of wells to generate
        seed: Random seed

    Returns:
        List of well dictionaries with 't', 'q', and metadata
    """
    np.random.seed(seed)

    wells = []

    # Generate variety of well types
    for i in range(n_wells):
        well_type = i % 4

        if well_type == 0:
            # Standard hyperbolic
            t, q = generate_synthetic_arps_data(
                qi=1000 + np.random.uniform(-200, 200),
                di=0.1 + np.random.uniform(-0.05, 0.05),
                b=0.5 + np.random.uniform(-0.2, 0.2),
                seed=seed + i,
            )
        elif well_type == 1:
            # Exponential decline
            t, q = generate_synthetic_arps_data(
                qi=800 + np.random.uniform(-150, 150),
                di=0.15 + np.random.uniform(-0.05, 0.05),
                b=0.0,  # Exponential
                model_type="exponential",
                seed=seed + i,
            )
        elif well_type == 2:
            # Piecewise decline
            segments = [
                {"qi": 1200, "di": 0.08, "b": 0.4, "duration": 40},
                {"qi": 600, "di": 0.12, "b": 0.6, "duration": 60},
            ]
            t, q = generate_piecewise_decline(segments, seed=seed + i)
        else:
            # Ramp-up
            t, q = generate_ramp_up_data(
                qi_final=900 + np.random.uniform(-100, 100),
                ramp_duration=10 + np.random.uniform(0, 10),
                decline_di=0.12 + np.random.uniform(-0.03, 0.03),
                decline_b=0.5 + np.random.uniform(-0.2, 0.2),
                seed=seed + i,
            )

        wells.append(
            {
                "well_id": f"WELL_{i:03d}",
                "t": t,
                "q": q,
                "well_type": ["hyperbolic", "exponential", "piecewise", "ramp_up"][
                    well_type
                ],
            }
        )

    return wells
