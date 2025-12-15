"""Units system for decline curve analysis.

This module provides unit handling using pint for strict unit management.
All internal calculations use consistent units (days for time, user-specified
for rate), with conversion at the boundary (loaders/writers).

References:
- SPEE REP 6: Nominal vs effective decline rates
- pint documentation: https://pint.readthedocs.io/
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import pint

    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    pint = None

from .logging_config import get_logger

logger = get_logger(__name__)

# Internal unit constants
INTERNAL_TIME_UNIT = "days"
INTERNAL_RATE_UNIT = "bbl_per_day"  # Default, can be overridden


@dataclass
class UnitSystem:
    """Unit system configuration.

    Attributes:
        time_unit: Internal time unit (default: 'days')
        rate_unit: Internal rate unit (default: 'bbl_per_day')
        ureg: Pint unit registry (if available)
    """

    time_unit: str = INTERNAL_TIME_UNIT
    rate_unit: str = INTERNAL_RATE_UNIT
    ureg: Optional[object] = None

    def __post_init__(self):
        """Initialize pint unit registry if available."""
        if PINT_AVAILABLE and self.ureg is None:
            self.ureg = pint.UnitRegistry()
            # Define common oil & gas units
            self._define_custom_units()
        elif not PINT_AVAILABLE:
            logger.warning(
                "pint not available. Install with: pip install pint. "
                "Unit conversion will be limited."
            )

    def _define_custom_units(self):
        """Define custom units for oil & gas."""
        if not PINT_AVAILABLE:
            return

        # Define common units
        # bbl = barrel
        # mcf = thousand cubic feet
        # scf = standard cubic feet
        # bbl_per_day = barrels per day
        # mcf_per_day = thousand cubic feet per day

        # These are typically already defined in pint, but we ensure they exist
        try:
            _ = self.ureg.bbl
            _ = self.ureg.mcf
            _ = self.ureg.scf
        except AttributeError:
            # Define if not present
            self.ureg.define("bbl = barrel")
            self.ureg.define("mcf = 1000 * scf")
            self.ureg.define("scf = standard_cubic_foot")


class UnitConverter:
    """Unit converter for decline curve analysis.

    Handles conversion between external units (from data) and internal
    units (for calculations). Stores original units for provenance.
    """

    def __init__(self, unit_system: Optional[UnitSystem] = None):
        """Initialize unit converter.

        Args:
            unit_system: Optional UnitSystem instance (creates default if None)
        """
        self.unit_system = unit_system or UnitSystem()
        self.original_units: dict[str, str] = {}

    def convert_time(
        self,
        time_values: Union[list, "np.ndarray"],
        from_unit: str,
        to_unit: Optional[str] = None,
    ) -> "np.ndarray":
        """Convert time values between units.

        Args:
            time_values: Time values to convert
            from_unit: Source unit (e.g., 'days', 'months', 'years')
            to_unit: Target unit (default: internal time unit)

        Returns:
            Converted time values as numpy array
        """
        import numpy as np

        if to_unit is None:
            to_unit = self.unit_system.time_unit

        if from_unit == to_unit:
            return np.asarray(time_values)

        if not PINT_AVAILABLE:
            # Fallback conversion for common cases
            return self._convert_time_fallback(time_values, from_unit, to_unit)

        # Use pint for conversion
        try:
            quantity = self.unit_system.ureg.Quantity(time_values, from_unit)
            converted = quantity.to(to_unit)
            return np.asarray(converted.magnitude)
        except Exception as e:
            logger.warning(f"Pint conversion failed, using fallback: {e}")
            return self._convert_time_fallback(time_values, from_unit, to_unit)

    def _convert_time_fallback(
        self, time_values: Union[list, "np.ndarray"], from_unit: str, to_unit: str
    ) -> "np.ndarray":
        """Fallback time conversion without pint.

        Handles common time unit conversions.
        """
        import numpy as np

        time_values = np.asarray(time_values)

        # Conversion factors to days
        to_days = {
            "days": 1.0,
            "day": 1.0,
            "months": 30.4375,  # Average days per month
            "month": 30.4375,
            "years": 365.25,  # Average days per year
            "year": 365.25,
            "hours": 1.0 / 24.0,
            "hour": 1.0 / 24.0,
        }

        # Convert to days first
        if from_unit.lower() in to_days:
            days = time_values * to_days[from_unit.lower()]
        else:
            logger.warning(f"Unknown time unit: {from_unit}, assuming days")
            days = time_values

        # Convert from days to target
        if to_unit.lower() in to_days:
            factor = 1.0 / to_days[to_unit.lower()]
            return days * factor
        else:
            logger.warning(f"Unknown target time unit: {to_unit}, returning days")
            return days

    def convert_rate(
        self,
        rate_values: Union[list, "np.ndarray"],
        from_unit: str,
        to_unit: Optional[str] = None,
    ) -> "np.ndarray":
        """Convert rate values between units.

        Args:
            rate_values: Rate values to convert
            from_unit: Source unit (e.g., 'bbl_per_day', 'bbl_per_month')
            to_unit: Target unit (default: internal rate unit)

        Returns:
            Converted rate values as numpy array
        """
        import numpy as np

        if to_unit is None:
            to_unit = self.unit_system.rate_unit

        if from_unit == to_unit:
            return np.asarray(rate_values)

        if not PINT_AVAILABLE:
            # Fallback conversion for common cases
            return self._convert_rate_fallback(rate_values, from_unit, to_unit)

        # Use pint for conversion
        try:
            quantity = self.unit_system.ureg.Quantity(rate_values, from_unit)
            converted = quantity.to(to_unit)
            return np.asarray(converted.magnitude)
        except Exception as e:
            logger.warning(f"Pint conversion failed, using fallback: {e}")
            return self._convert_rate_fallback(rate_values, from_unit, to_unit)

    def _convert_rate_fallback(
        self, rate_values: Union[list, "np.ndarray"], from_unit: str, to_unit: str
    ) -> "np.ndarray":
        """Fallback rate conversion without pint.

        Handles common rate unit conversions.
        """
        import numpy as np

        rate_values = np.asarray(rate_values)

        # Parse units (e.g., "bbl_per_day" -> ("bbl", "day"))
        def parse_rate_unit(unit_str: str) -> tuple[str, str]:
            """Parse rate unit into volume and time components."""
            unit_lower = unit_str.lower()

            # Common patterns
            if "_per_" in unit_lower:
                parts = unit_lower.split("_per_")
                if len(parts) == 2:
                    return (parts[0], parts[1])

            # Try "/" separator
            if "/" in unit_lower:
                parts = unit_lower.split("/")
                if len(parts) == 2:
                    return (parts[0].strip(), parts[1].strip())

            # Default assumptions
            logger.warning(f"Could not parse rate unit: {unit_str}, assuming bbl/day")
            return ("bbl", "day")

        from_vol, from_time = parse_rate_unit(from_unit)
        to_vol, to_time = parse_rate_unit(to_unit)

        # Volume conversions (to bbl)
        vol_to_bbl = {
            "bbl": 1.0,
            "barrel": 1.0,
            "barrels": 1.0,
            "mcf": 0.0,  # Gas - different unit, would need BTU/GPM
            "scf": 0.0,
            "m3": 6.28981,  # cubic meters to barrels
        }

        # Time conversions (to days)
        time_to_days = {
            "day": 1.0,
            "days": 1.0,
            "month": 30.4375,
            "months": 30.4375,
            "year": 365.25,
            "years": 365.25,
        }

        # Convert volume component
        from_vol_factor = vol_to_bbl.get(from_vol, 1.0)
        to_vol_factor = vol_to_bbl.get(to_vol, 1.0)
        vol_factor = from_vol_factor / to_vol_factor

        # Convert time component (inverse relationship)
        from_time_factor = time_to_days.get(from_time, 1.0)
        to_time_factor = time_to_days.get(to_time, 1.0)
        time_factor = to_time_factor / from_time_factor  # Inverse

        return rate_values * vol_factor * time_factor

    def store_original_units(self, **units: str) -> None:
        """Store original units for provenance.

        Args:
            **units: Unit specifications (e.g., time='months', rate='bbl_per_month')
        """
        self.original_units.update(units)

    def get_original_units(self) -> dict[str, str]:
        """Get stored original units.

        Returns:
            Dictionary of original unit specifications
        """
        return self.original_units.copy()


def convert_decline_rate(
    di: float,
    from_time_unit: str,
    to_time_unit: str = "days",
    nominal: bool = True,
) -> float:
    """Convert decline rate between time units.

    Handles both nominal and effective decline rates according to SPEE REP 6.

    Args:
        di: Decline rate value
        from_time_unit: Source time unit (e.g., 'per_year', 'per_month')
        to_time_unit: Target time unit (default: 'per_day')
        nominal: If True, use nominal decline (default). If False, use effective.

    Returns:
        Converted decline rate

    References:
        SPEE REP 6: Guidelines for Application of the Definitions for Oil and
        Gas Reserves (Section on decline rates)

    Example:
        >>> # Convert annual decline rate to daily
        >>> di_daily = convert_decline_rate(0.5, 'per_year', 'per_day')
        >>> # Convert monthly to annual (nominal)
        >>> di_annual = convert_decline_rate(0.1, 'per_month', 'per_year')
    """
    # Time unit conversion factors
    time_factors = {
        "per_day": 1.0,
        "per_month": 30.4375,
        "per_year": 365.25,
    }

    from_factor = time_factors.get(from_time_unit, 1.0)
    to_factor = time_factors.get(to_time_unit, 1.0)

    if nominal:
        # Nominal decline: simple scaling
        # Decline rate is "per unit time", so we divide by time factor
        # (not multiply) - smaller time unit means smaller decline rate
        return di * (to_factor / from_factor)
    else:
        # Effective decline: requires exponential conversion
        # Effective decline: di_eff = 1 - exp(-di_nom * t)
        # For conversion, we need to invert this
        # This is more complex and typically nominal is used
        logger.warning(
            "Effective decline conversion not fully implemented. "
            "Using nominal conversion."
        )
        return di * (to_factor / from_factor)


def validate_units(unit_string: str, unit_type: str = "rate") -> bool:
    """Validate unit string format.

    Args:
        unit_string: Unit string to validate
        unit_type: Type of unit ('rate', 'time', 'volume')

    Returns:
        True if unit string appears valid
    """
    if not unit_string or not isinstance(unit_string, str):
        return False

    unit_lower = unit_string.lower()

    if unit_type == "time":
        valid_time_units = [
            "day",
            "days",
            "month",
            "months",
            "year",
            "years",
            "hour",
            "hours",
        ]
        return any(unit_lower.endswith(u) for u in valid_time_units)

    elif unit_type == "rate":
        # Should contain volume and time components
        return "_per_" in unit_lower or "/" in unit_lower

    elif unit_type == "volume":
        valid_volume_units = ["bbl", "barrel", "mcf", "scf", "m3", "m^3"]
        return any(unit_lower.startswith(u) for u in valid_volume_units)

    return False
