"""Base model interface for decline curve models.

This module defines the strict interface that all decline curve models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class Model(ABC):
    """Abstract base class for decline curve models.

    All decline curve models must implement this interface to ensure
    consistency across the library.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name as a string.

        Returns:
            Model name (e.g., 'ExponentialArps', 'HyperbolicArps').
        """
        pass

    @abstractmethod
    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute production rate at given times.

        Args:
            t: Time array (in days from start).
            params: Model parameters as a dictionary.

        Returns:
            Production rate array (same units as input).
        """
        pass

    @abstractmethod
    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production at given times.

        Uses analytic formulas when available, otherwise numeric integration.

        Args:
            t: Time array (in days from start).
            params: Model parameters as a dictionary.

        Returns:
            Cumulative production array.
        """
        pass

    @abstractmethod
    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds and constraints.

        Returns:
            Dictionary mapping parameter names to (lower_bound, upper_bound) tuples.
            Use -np.inf and np.inf for unbounded parameters.
        """
        pass

    @abstractmethod
    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial parameter guess from data.

        Args:
            t: Time array (in days from start).
            q: Production rate array.

        Returns:
            Dictionary of parameter names to initial guess values.
        """
        pass

    def validate(self, params: Dict[str, float]) -> Tuple[bool, list[str]]:
        """Validate parameter values.

        Args:
            params: Model parameters as a dictionary.

        Returns:
            Tuple of (is_valid, list_of_warnings).
            Warnings are issued for parameters near bounds or unusual values.
        """
        warnings = []
        constraints = self.constraints()

        for param_name, value in params.items():
            if param_name in constraints:
                lower, upper = constraints[param_name]
                if value < lower or value > upper:
                    return False, [
                        f"{param_name}={value} is outside bounds [{lower}, {upper}]"
                    ]
                # Warn if near bounds
                if lower > -np.inf and value < lower * 1.1:
                    warnings.append(f"{param_name} is near lower bound")
                if upper < np.inf and value > upper * 0.9:
                    warnings.append(f"{param_name} is near upper bound")

        return True, warnings
