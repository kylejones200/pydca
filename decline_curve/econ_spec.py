"""Economics specifications for decline curve analysis.

This module provides EconSpec objects for configuring economic analysis,
including prices, costs, taxes, royalties, and economic limits.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(default, **kwargs):
        """Field function for when pydantic is not available."""
        return default


if PYDANTIC_AVAILABLE:

    class EconSpec(BaseModel):
        """Economics specification for decline curve analysis.

        Attributes:
            oil_price: Oil price ($/bbl)
            gas_price: Gas price ($/mcf)
            water_price: Water price ($/bbl, typically negative for disposal)
            fixed_opex: Fixed operating expenses ($/month)
            variable_opex: Variable operating expenses ($/bbl or $/mcf)
            capex: Capital expenditures ($)
            discount_rate: Annual discount rate (fraction)
            tax_rate: Tax rate (fraction)
            royalty_rate: Royalty rate (fraction)
            shrink: Shrink factor for gas (fraction)
            btu_factor: BTU conversion factor
            oil_differential: Oil price differential ($/bbl)
            gas_differential: Gas price differential ($/mcf)
            econ_limit_rate: Economic limit rate (bbl/day or mcf/day)
            econ_limit_cashflow: Economic limit cash flow ($/month)
        """

        oil_price: float = Field(70.0, description="Oil price ($/bbl)")
        gas_price: float = Field(3.0, description="Gas price ($/mcf)")
        water_price: float = Field(-2.0, description="Water disposal cost ($/bbl)")
        fixed_opex: float = Field(5000.0, description="Fixed opex ($/month)")
        variable_opex: float = Field(5.0, description="Variable opex ($/bbl)")
        capex: float = Field(0.0, description="Capital expenditures ($)")
        discount_rate: float = Field(0.10, description="Annual discount rate")
        tax_rate: float = Field(0.0, description="Tax rate")
        royalty_rate: float = Field(0.125, description="Royalty rate (12.5%)")
        shrink: float = Field(1.0, description="Gas shrink factor")
        btu_factor: float = Field(1.0, description="BTU conversion factor")
        oil_differential: float = Field(0.0, description="Oil price differential")
        gas_differential: float = Field(0.0, description="Gas price differential")
        econ_limit_rate: Optional[float] = Field(
            None, description="Economic limit rate"
        )
        econ_limit_cashflow: Optional[float] = Field(
            None, description="Economic limit cash flow"
        )

else:

    @dataclass
    class EconSpec:
        """Economics specification (without Pydantic validation)."""

        oil_price: float = 70.0
        gas_price: float = 3.0
        water_price: float = -2.0
        fixed_opex: float = 5000.0
        variable_opex: float = 5.0
        capex: float = 0.0
        discount_rate: float = 0.10
        tax_rate: float = 0.0
        royalty_rate: float = 0.125
        shrink: float = 1.0
        btu_factor: float = 1.0
        oil_differential: float = 0.0
        gas_differential: float = 0.0
        econ_limit_rate: Optional[float] = None
        econ_limit_cashflow: Optional[float] = None


@dataclass
class Period:
    """Time period with constraints and economic overrides.

    Attributes:
        start_date: Start date of period
        end_date: End date of period
        tag: Period tag/name
        facility_limit: Facility capacity limit (optional)
        choke_limit: Choke limit (optional)
        downtime: Expected downtime fraction (optional)
        min_uptime: Minimum uptime requirement (optional)
        econ_spec_override: Optional EconSpec overrides for this period
    """

    start_date: str
    end_date: str
    tag: str
    facility_limit: Optional[float] = None
    choke_limit: Optional[float] = None
    downtime: Optional[float] = None
    min_uptime: Optional[float] = None
    econ_spec_override: Optional[Dict[str, float]] = None


@dataclass
class Scenario:
    """Scenario for economic analysis.

    Attributes:
        name: Scenario name
        periods: List of Period objects
        well_ids: List of well IDs (empty for all wells)
        base_econ_spec: Base EconSpec for scenario
    """

    name: str
    periods: list[Period]
    well_ids: list[str] = field(default_factory=list)
    base_econ_spec: Optional[EconSpec] = None
