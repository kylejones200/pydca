"""Arps decline curve model implementations.

This module implements Exponential, Harmonic, Hyperbolic, and Hyperbolic-to-Exponential
switch models following the strict Model interface.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import quad

from .models_base import Model


class ExponentialArps(Model):
    """Exponential decline curve model (b=0).

    Rate equation: q(t) = qi * exp(-di * t)
    Cumulative: Np(t) = (qi / di) * (1 - exp(-di * t))
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "ExponentialArps"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute exponential decline rate.

        Args:
            t: Time array (in days).
            params: Must contain 'qi' and 'di'.

        Returns:
            Production rate array.
        """
        qi = params["qi"]
        di = params["di"]
        return qi * np.exp(-di * t)

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production analytically.

        Analytic formula: Np(t) = (qi / di) * (1 - exp(-di * t))

        Args:
            t: Time array (in days).
            params: Must contain 'qi' and 'di'.

        Returns:
            Cumulative production array.
        """
        qi = params["qi"]
        di = params["di"]

        if di <= 0:
            # Handle edge case: zero or negative decline
            return np.zeros_like(t)

        # Analytic cumulative for exponential decline
        return (qi / di) * (1 - np.exp(-di * t))

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            qi: (0, inf), di: (0, inf)
        """
        return {
            "qi": (0.0, np.inf),
            "di": (0.0, np.inf),
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from data.

        Uses ramp-aware heuristic: qi = max rate in first period.
        di estimated from first and last valid points.

        Args:
            t: Time array (in days).
            q: Production rate array.

        Returns:
            Dictionary with 'qi' and 'di'.
        """
        # Filter valid data
        valid_mask = q > 0
        if not np.any(valid_mask):
            return {"qi": 1.0, "di": 0.01}

        t_valid = t[valid_mask]
        q_valid = q[valid_mask]

        # Ramp-aware qi: max rate in first 30% of data or first 3 months
        first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
        qi = float(np.max(q_valid[:first_period]))

        # Estimate di from decline
        if len(q_valid) >= 2:
            # Use first and last point to estimate decline
            q0 = q_valid[0]
            q_last = q_valid[-1]
            t_span = t_valid[-1] - t_valid[0]
            if t_span > 0 and q_last > 0 and q0 > 0:
                di = -np.log(q_last / q0) / t_span
                di = max(0.001, min(di, 10.0))  # Reasonable bounds
            else:
                di = 0.01
        else:
            di = 0.01

        return {"qi": qi, "di": di}


class HarmonicArps(Model):
    """Harmonic decline curve model (b=1).

    Rate equation: q(t) = qi / (1 + di * t)
    Cumulative: Np(t) = (qi / di) * ln(1 + di * t)
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "HarmonicArps"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute harmonic decline rate.

        Args:
            t: Time array (in days).
            params: Must contain 'qi' and 'di'.

        Returns:
            Production rate array.
        """
        qi = params["qi"]
        di = params["di"]
        return qi / (1 + di * t)

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production analytically.

        Analytic formula: Np(t) = (qi / di) * ln(1 + di * t)

        Args:
            t: Time array (in days).
            params: Must contain 'qi' and 'di'.

        Returns:
            Cumulative production array.
        """
        qi = params["qi"]
        di = params["di"]

        if di <= 0:
            return np.zeros_like(t)

        # Analytic cumulative for harmonic decline
        return (qi / di) * np.log(1 + di * t)

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            qi: (0, inf), di: (0, inf)
        """
        return {
            "qi": (0.0, np.inf),
            "di": (0.0, np.inf),
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from data.

        Args:
            t: Time array (in days).
            q: Production rate array.

        Returns:
            Dictionary with 'qi' and 'di'.
        """
        # Filter valid data
        valid_mask = q > 0
        if not np.any(valid_mask):
            return {"qi": 1.0, "di": 0.01}

        t_valid = t[valid_mask]
        q_valid = q[valid_mask]

        # Ramp-aware qi
        first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
        qi = float(np.max(q_valid[:first_period]))

        # Estimate di from harmonic decline
        if len(q_valid) >= 2:
            q0 = q_valid[0]
            q_last = q_valid[-1]
            t_span = t_valid[-1] - t_valid[0]
            if t_span > 0 and q_last > 0 and q0 > 0:
                # From q = qi / (1 + di*t): di = (qi/q - 1) / t
                di = (q0 / q_last - 1) / t_span
                di = max(0.001, min(di, 10.0))
            else:
                di = 0.01
        else:
            di = 0.01

        return {"qi": qi, "di": di}


class HyperbolicArps(Model):
    """Hyperbolic decline curve model (0 < b < 1 typically).

    Rate equation: q(t) = qi / (1 + b * di * t)^(1/b)
    Cumulative: Np(t) = (qi / (di * (1-b))) * (1 - (1 + b*di*t)^((1-b)/b))
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "HyperbolicArps"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute hyperbolic decline rate.

        Args:
            t: Time array (in days).
            params: Must contain 'qi', 'di', and 'b'.

        Returns:
            Production rate array.
        """
        qi = params["qi"]
        di = params["di"]
        b = params["b"]

        # Handle edge cases
        if b == 0.0:
            # Exponential limit
            return qi * np.exp(-di * t)
        elif abs(b - 1.0) < 1e-9:
            # Harmonic limit
            return qi / (1 + di * t)
        else:
            # Standard hyperbolic
            return qi / np.power(1 + b * di * t, 1 / b)

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production analytically.

        Analytic formula for b < 1:
        Np(t) = (qi / (di * (1-b))) * (1 - (1 + b*di*t)^((1-b)/b))

        For b >= 1, uses numeric integration.

        Args:
            t: Time array (in days).
            params: Must contain 'qi', 'di', and 'b'.

        Returns:
            Cumulative production array.
        """
        qi = params["qi"]
        di = params["di"]
        b = params["b"]

        if di <= 0:
            return np.zeros_like(t)

        # Handle edge cases
        if b == 0.0:
            # Exponential
            return (qi / di) * (1 - np.exp(-di * t))
        elif abs(b - 1.0) < 1e-9:
            # Harmonic
            return (qi / di) * np.log(1 + di * t)
        elif b < 1.0:
            # Analytic cumulative for hyperbolic (b < 1)
            exponent = (1 - b) / b
            return (qi / (di * (1 - b))) * (1 - np.power(1 + b * di * t, -exponent))
        else:
            # b >= 1: use numeric integration
            return self._cum_numeric(t, params)

    def _cum_numeric(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative using numeric integration (for b >= 1).

        Args:
            t: Time array (in days).
            params: Model parameters.

        Returns:
            Cumulative production array.
        """
        qi = params["qi"]
        di = params["di"]
        b = params["b"]

        cum = np.zeros_like(t)
        for i, t_val in enumerate(t):
            if t_val == 0:
                cum[i] = 0.0
            else:
                # Numeric integration
                result, _ = quad(
                    lambda s: qi / np.power(1 + b * di * s, 1 / b),
                    0,
                    t_val,
                    limit=100,
                )
                cum[i] = result

        return cum

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            qi: (0, inf), di: (0, inf), b: (0, 2)
        """
        return {
            "qi": (0.0, np.inf),
            "di": (0.0, np.inf),
            "b": (0.0, 2.0),
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from data.

        Args:
            t: Time array (in days).
            q: Production rate array.

        Returns:
            Dictionary with 'qi', 'di', and 'b'.
        """
        # Filter valid data
        valid_mask = q > 0
        if not np.any(valid_mask):
            return {"qi": 1.0, "di": 0.01, "b": 0.5}

        t_valid = t[valid_mask]
        q_valid = q[valid_mask]

        # Ramp-aware qi
        first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
        qi = float(np.max(q_valid[:first_period]))

        # Estimate di and b
        if len(q_valid) >= 3:
            # Use first and last points to estimate b
            q0 = q_valid[0]
            q_last = q_valid[-1]
            t_mid = t_valid[len(t_valid) // 2]
            t_span = t_valid[-1] - t_valid[0]

            if t_span > 0 and t_mid > 0:
                # Rough estimate of b from decline pattern
                # More sophisticated methods could be used
                b = 0.5  # Default

                # Estimate di from first and last
                if q_last > 0 and q0 > 0:
                    # Use exponential as starting point
                    di_exp = -np.log(q_last / q0) / t_span
                    di = max(0.001, min(di_exp, 10.0))
                else:
                    di = 0.01
            else:
                di = 0.01
                b = 0.5
        else:
            di = 0.01
            b = 0.5

        return {"qi": qi, "di": di, "b": b}


class HyperbolicToExponentialSwitch(Model):
    """Hyperbolic decline switching to exponential at a transition time.

    Rate equation:
    - For t < t_switch: q(t) = qi / (1 + b * di * t)^(1/b)
    - For t >= t_switch: q(t) = q_switch * exp(-di_exp * (t - t_switch))

    where q_switch is the rate at t_switch from the hyperbolic curve.
    """

    @property
    def name(self) -> str:
        return "HyperbolicToExponentialSwitch"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute rate with hyperbolic-to-exponential switch.

        Args:
            t: Time array (in days).
            params: Must contain 'qi', 'di', 'b', 't_switch', 'di_exp'.

        Returns:
            Production rate array.
        """
        qi = params["qi"]
        di = params["di"]
        b = params["b"]
        t_switch = params["t_switch"]
        di_exp = params["di_exp"]

        q = np.zeros_like(t)

        # Hyperbolic phase
        mask_hyper = t < t_switch
        if np.any(mask_hyper):
            t_hyper = t[mask_hyper]
            if b == 0.0:
                q[mask_hyper] = qi * np.exp(-di * t_hyper)
            else:
                q[mask_hyper] = qi / np.power(1 + b * di * t_hyper, 1 / b)

        # Exponential phase
        mask_exp = t >= t_switch
        if np.any(mask_exp):
            # Compute rate at switch point
            if b == 0.0:
                q_switch = qi * np.exp(-di * t_switch)
            else:
                q_switch = qi / np.power(1 + b * di * t_switch, 1 / b)

            t_exp = t[mask_exp] - t_switch
            q[mask_exp] = q_switch * np.exp(-di_exp * t_exp)

        return q

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production.

        Uses analytic formulas for both phases.

        Args:
            t: Time array (in days).
            params: Must contain 'qi', 'di', 'b', 't_switch', 'di_exp'.

        Returns:
            Cumulative production array.
        """
        qi = params["qi"]
        di = params["di"]
        b = params["b"]
        t_switch = params["t_switch"]
        di_exp = params["di_exp"]

        cum = np.zeros_like(t)

        for i, t_val in enumerate(t):
            if t_val <= t_switch:
                # Only hyperbolic phase
                if b == 0.0:
                    cum[i] = (qi / di) * (1 - np.exp(-di * t_val))
                elif abs(b - 1.0) < 1e-9:
                    cum[i] = (qi / di) * np.log(1 + di * t_val)
                elif b < 1.0:
                    exponent = (1 - b) / b
                    cum[i] = (qi / (di * (1 - b))) * (
                        1 - np.power(1 + b * di * t_val, -exponent)
                    )
                else:
                    # Numeric for b >= 1
                    result, _ = quad(
                        lambda s: qi / np.power(1 + b * di * s, 1 / b),
                        0,
                        t_val,
                        limit=100,
                    )
                    cum[i] = result
            else:
                # Both phases: hyperbolic to t_switch, then exponential
                # Cumulative from hyperbolic phase
                if b == 0.0:
                    cum_hyper = (qi / di) * (1 - np.exp(-di * t_switch))
                elif abs(b - 1.0) < 1e-9:
                    cum_hyper = (qi / di) * np.log(1 + di * t_switch)
                elif b < 1.0:
                    exponent = (1 - b) / b
                    cum_hyper = (qi / (di * (1 - b))) * (
                        1 - np.power(1 + b * di * t_switch, -exponent)
                    )
                else:
                    result, _ = quad(
                        lambda s: qi / np.power(1 + b * di * s, 1 / b),
                        0,
                        t_switch,
                        limit=100,
                    )
                    cum_hyper = result

                # Rate at switch point
                if b == 0.0:
                    q_switch = qi * np.exp(-di * t_switch)
                else:
                    q_switch = qi / np.power(1 + b * di * t_switch, 1 / b)

                # Cumulative from exponential phase
                t_exp = t_val - t_switch
                cum_exp = (q_switch / di_exp) * (1 - np.exp(-di_exp * t_exp))

                cum[i] = cum_hyper + cum_exp

        return cum

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds.

        Returns:
            qi: (0, inf), di: (0, inf), b: (0, 2), t_switch: (0, inf), di_exp: (0, inf)
        """
        return {
            "qi": (0.0, np.inf),
            "di": (0.0, np.inf),
            "b": (0.0, 2.0),
            "t_switch": (0.0, np.inf),
            "di_exp": (0.0, np.inf),
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from data.

        Args:
            t: Time array (in days).
            q: Production rate array.

        Returns:
            Dictionary with all parameters.
        """
        # Start with hyperbolic guess
        hyper_model = HyperbolicArps()
        hyper_guess = hyper_model.initial_guess(t, q)

        # Set switch time to 2/3 of data range
        t_span = t[-1] - t[0] if len(t) > 1 else 1.0
        t_switch = t_span * 0.67

        # Use same di for exponential phase initially
        di_exp = hyper_guess["di"]

        return {
            "qi": hyper_guess["qi"],
            "di": hyper_guess["di"],
            "b": hyper_guess["b"],
            "t_switch": t_switch,
            "di_exp": di_exp,
        }
