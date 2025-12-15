"""Yield models for secondary phase production.

This module provides interfaces and implementations for modeling secondary
phase production (GOR, CGR, water cut) as functions of primary phase production
or time. This enables forecasting associated gas from oil production, or
liquids from gas production.

References:
- SPEE REP 6 - Decline curve analysis standards
- Industry practices for GOR/CGR modeling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class YieldModel(ABC):
    """Abstract base class for yield models.

    Yield models compute secondary phase production rates from primary
    phase production or time. Examples include:
    - GOR (Gas-Oil Ratio): gas rate from oil rate
    - CGR (Condensate-Gas Ratio): liquids rate from gas rate
    - Water cut: water rate from oil rate
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the yield model name.

        Returns:
            Model name (e.g., 'ConstantGOR', 'DecliningGOR')
        """
        pass

    @abstractmethod
    def yield_rate(
        self,
        t: np.ndarray,
        primary_rate: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute secondary phase rate.

        Args:
            t: Time array (days)
            primary_rate: Optional primary phase rate array
            params: Model parameters

        Returns:
            Secondary phase rate array
        """
        pass

    @abstractmethod
    def constraints(self) -> Dict[str, tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds
        """
        pass


class ConstantYield(YieldModel):
    """Constant yield model (e.g., constant GOR).

    Secondary rate = primary_rate * yield_ratio

    This is the simplest yield model, appropriate when the yield
    ratio remains relatively constant over the production life.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "ConstantYield"

    def yield_rate(
        self,
        t: np.ndarray,
        primary_rate: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute secondary rate with constant yield ratio.

        Args:
            t: Time array (days)
            primary_rate: Primary phase rate array (required)
            params: Must contain 'yield_ratio'

        Returns:
            Secondary phase rate array
        """
        if params is None:
            raise ValueError("params required for ConstantYield")
        if primary_rate is None:
            raise ValueError("primary_rate required for ConstantYield")

        yield_ratio = params.get("yield_ratio", 0.0)
        return primary_rate * yield_ratio

    def constraints(self) -> Dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "yield_ratio": (0.0, np.inf),
        }


class DecliningYield(YieldModel):
    """Declining yield model (e.g., declining GOR over time).

    Secondary rate = primary_rate * yield_ratio(t)
    where yield_ratio(t) = yield_initial * exp(-decline_rate * t)

    This model accounts for yield decline over time, which is common
    in gas-oil systems as reservoir pressure declines.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "DecliningYield"

    def yield_rate(
        self,
        t: np.ndarray,
        primary_rate: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute secondary rate with declining yield ratio.

        Args:
            t: Time array (days)
            primary_rate: Primary phase rate array (required)
            params: Must contain 'yield_initial' and 'decline_rate'

        Returns:
            Secondary phase rate array
        """
        if params is None:
            raise ValueError("params required for DecliningYield")
        if primary_rate is None:
            raise ValueError("primary_rate required for DecliningYield")

        yield_initial = params.get("yield_initial", 0.0)
        decline_rate = params.get("decline_rate", 0.0)

        yield_ratio = yield_initial * np.exp(-decline_rate * t)
        return primary_rate * yield_ratio

    def constraints(self) -> Dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "yield_initial": (0.0, np.inf),
            "decline_rate": (0.0, np.inf),
        }


class HyperbolicYield(YieldModel):
    """Hyperbolic yield model.

    Secondary rate = primary_rate * yield_ratio(t)
    where yield_ratio(t) = yield_initial / (1 + b * decline_rate * t)^(1/b)

    This model provides more flexibility than exponential decline
    for yield ratios that decline hyperbolically.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "HyperbolicYield"

    def yield_rate(
        self,
        t: np.ndarray,
        primary_rate: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute secondary rate with hyperbolic yield decline.

        Args:
            t: Time array (days)
            primary_rate: Primary phase rate array (required)
            params: Must contain 'yield_initial', 'decline_rate', and 'b'

        Returns:
            Secondary phase rate array
        """
        if params is None:
            raise ValueError("params required for HyperbolicYield")
        if primary_rate is None:
            raise ValueError("primary_rate required for HyperbolicYield")

        yield_initial = params.get("yield_initial", 0.0)
        decline_rate = params.get("decline_rate", 0.0)
        b = params.get("b", 0.5)

        if b == 0.0:
            # Exponential limit
            yield_ratio = yield_initial * np.exp(-decline_rate * t)
        else:
            yield_ratio = yield_initial / np.power(1 + b * decline_rate * t, 1 / b)

        return primary_rate * yield_ratio

    def constraints(self) -> Dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "yield_initial": (0.0, np.inf),
            "decline_rate": (0.0, np.inf),
            "b": (0.0, 2.0),
        }


class TimeBasedYield(YieldModel):
    """Time-based yield model (independent of primary rate).

    Secondary rate = yield_rate(t)

    This model is useful when secondary phase production is primarily
    a function of time (e.g., water breakthrough timing) rather than
    primary phase rate.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "TimeBasedYield"

    def yield_rate(
        self,
        t: np.ndarray,
        primary_rate: Optional[np.ndarray] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute secondary rate as function of time only.

        Uses exponential decline: rate(t) = qi * exp(-di * t)

        Args:
            t: Time array (days)
            primary_rate: Not used (can be None)
            params: Must contain 'qi' and 'di'

        Returns:
            Secondary phase rate array
        """
        if params is None:
            raise ValueError("params required for TimeBasedYield")

        qi = params.get("qi", 0.0)
        di = params.get("di", 0.0)

        return qi * np.exp(-di * t)

    def constraints(self) -> Dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "qi": (0.0, np.inf),
            "di": (0.0, np.inf),
        }


@dataclass
class YieldModelAttachment:
    """Attachment of a yield model to a primary phase model.

    This class links a yield model to a primary phase decline model,
    enabling computation of secondary phase forecasts from primary
    phase forecasts.

    Attributes:
        yield_model: YieldModel instance
        params: Parameters for the yield model
        phase_name: Name of the secondary phase (e.g., 'gas', 'water')
    """

    yield_model: YieldModel
    params: Dict[str, float]
    phase_name: str

    def compute_secondary_forecast(
        self, t: np.ndarray, primary_rate: np.ndarray
    ) -> np.ndarray:
        """Compute secondary phase forecast from primary phase rate.

        Args:
            t: Time array (days)
            primary_rate: Primary phase rate array

        Returns:
            Secondary phase rate array
        """
        return self.yield_model.yield_rate(t, primary_rate, self.params)


def create_gor_model(
    gor_initial: float,
    gor_decline_rate: float = 0.0,
    model_type: str = "constant",
) -> YieldModelAttachment:
    """Create a GOR (Gas-Oil Ratio) yield model.

    Convenience function to create a GOR model with common configurations.

    Args:
        gor_initial: Initial GOR (mcf/bbl or scf/bbl)
        gor_decline_rate: GOR decline rate (1/day) - only used for declining models
        model_type: 'constant', 'declining', or 'hyperbolic'

    Returns:
        YieldModelAttachment configured for GOR

    Example:
        >>> # Constant GOR of 1000 scf/bbl
        >>> gor_model = create_gor_model(1000.0, model_type='constant')
        >>>
        >>> # Declining GOR
        >>> gor_model = create_gor_model(
        ...     1000.0, gor_decline_rate=0.001, model_type='declining'
        ... )
    """
    if model_type == "constant":
        model = ConstantYield()
        params = {"yield_ratio": gor_initial}
    elif model_type == "declining":
        model = DecliningYield()
        params = {
            "yield_initial": gor_initial,
            "decline_rate": gor_decline_rate,
        }
    elif model_type == "hyperbolic":
        model = HyperbolicYield()
        params = {
            "yield_initial": gor_initial,
            "decline_rate": gor_decline_rate,
            "b": 0.5,  # Default b-factor
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return YieldModelAttachment(yield_model=model, params=params, phase_name="gas")


def create_water_cut_model(
    water_cut_initial: float,
    water_cut_growth_rate: float = 0.0,
    model_type: str = "constant",
) -> YieldModelAttachment:
    """Create a water cut yield model.

    Convenience function to create a water cut model.

    Args:
        water_cut_initial: Initial water cut (fraction, 0-1)
        water_cut_growth_rate: Water cut growth rate (1/day) - only used for
            growing models
        model_type: 'constant', 'declining' (for water cut, declining means
            increasing), or 'hyperbolic'

    Returns:
        YieldModelAttachment configured for water cut

    Note:
        For water cut, "declining" yield actually means increasing water cut,
        which is the typical behavior as a well ages.
    """
    if model_type == "constant":
        model = ConstantYield()
        params = {"yield_ratio": water_cut_initial}
    elif model_type == "declining":
        # For water cut, we want it to increase, so we use negative decline
        model = DecliningYield()
        params = {
            "yield_initial": water_cut_initial,
            "decline_rate": -water_cut_growth_rate,  # Negative for growth
        }
    elif model_type == "hyperbolic":
        model = HyperbolicYield()
        params = {
            "yield_initial": water_cut_initial,
            "decline_rate": -water_cut_growth_rate,  # Negative for growth
            "b": 0.5,
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return YieldModelAttachment(yield_model=model, params=params, phase_name="water")
