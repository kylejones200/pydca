"""Tests for units system module."""

import numpy as np
import pytest

from decline_curve.units import (
    UnitConverter,
    UnitSystem,
    convert_decline_rate,
    validate_units,
)


class TestUnitSystem:
    """Test UnitSystem class."""

    def test_default_initialization(self):
        """Test default unit system initialization."""
        system = UnitSystem()

        assert system.time_unit == "days"
        assert system.rate_unit == "bbl_per_day"

    def test_custom_initialization(self):
        """Test custom unit system initialization."""
        system = UnitSystem(time_unit="months", rate_unit="bbl_per_month")

        assert system.time_unit == "months"
        assert system.rate_unit == "bbl_per_month"


class TestUnitConverter:
    """Test UnitConverter class."""

    def test_time_conversion_days_to_months(self):
        """Test time conversion from days to months."""
        converter = UnitConverter()

        days = np.array([30, 60, 90])
        months = converter.convert_time(days, "days", "months")

        # 30 days ≈ 1 month
        assert np.isclose(months[0], 1.0, rtol=0.1)

    def test_time_conversion_months_to_days(self):
        """Test time conversion from months to days."""
        converter = UnitConverter()

        months = np.array([1, 2, 3])
        days = converter.convert_time(months, "months", "days")

        # 1 month ≈ 30 days
        assert np.isclose(days[0], 30.0, rtol=0.1)

    def test_time_conversion_same_unit(self):
        """Test time conversion with same unit (no-op)."""
        converter = UnitConverter()

        days = np.array([10, 20, 30])
        result = converter.convert_time(days, "days", "days")

        assert np.allclose(result, days)

    def test_rate_conversion_bbl_per_day_to_bbl_per_month(self):
        """Test rate conversion from bbl/day to bbl/month."""
        converter = UnitConverter()

        bbl_per_day = np.array([100, 200, 300])
        bbl_per_month = converter.convert_rate(
            bbl_per_day, "bbl_per_day", "bbl_per_month"
        )

        # Should be approximately 30x larger
        assert np.isclose(bbl_per_month[0] / bbl_per_day[0], 30.0, rtol=0.1)

    def test_rate_conversion_bbl_per_month_to_bbl_per_day(self):
        """Test rate conversion from bbl/month to bbl/day."""
        converter = UnitConverter()

        bbl_per_month = np.array([3000, 6000, 9000])
        bbl_per_day = converter.convert_rate(
            bbl_per_month, "bbl_per_month", "bbl_per_day"
        )

        # Should be approximately 30x smaller
        assert np.isclose(bbl_per_day[0] / bbl_per_month[0], 1.0 / 30.0, rtol=0.1)

    def test_store_and_retrieve_original_units(self):
        """Test storing and retrieving original units."""
        converter = UnitConverter()

        converter.store_original_units(time="months", rate="bbl_per_month")
        original = converter.get_original_units()

        assert original["time"] == "months"
        assert original["rate"] == "bbl_per_month"

    def test_original_units_accumulation(self):
        """Test that original units accumulate."""
        converter = UnitConverter()

        converter.store_original_units(time="months")
        converter.store_original_units(rate="bbl_per_month")

        original = converter.get_original_units()
        assert "time" in original
        assert "rate" in original


class TestConvertDeclineRate:
    """Test decline rate conversion."""

    def test_convert_annual_to_daily_nominal(self):
        """Test converting annual decline rate to daily (nominal)."""
        di_annual = 0.5  # 50% per year
        di_daily = convert_decline_rate(di_annual, "per_year", "per_day")

        # Daily should be much smaller
        assert di_daily < di_annual
        assert di_daily > 0

    def test_convert_monthly_to_annual_nominal(self):
        """Test converting monthly decline rate to annual (nominal)."""
        di_monthly = 0.1  # 10% per month
        di_annual = convert_decline_rate(di_monthly, "per_month", "per_year")

        # Annual should be larger (approximately 12x, but not exactly due to compounding)
        assert di_annual > di_monthly

    def test_convert_same_unit(self):
        """Test conversion with same unit."""
        di = 0.1
        result = convert_decline_rate(di, "per_month", "per_month")

        assert result == di


class TestValidateUnits:
    """Test unit validation."""

    def test_validate_time_units(self):
        """Test validation of time units."""
        assert validate_units("days", "time")
        assert validate_units("months", "time")
        assert validate_units("years", "time")
        assert not validate_units("invalid", "time")

    def test_validate_rate_units(self):
        """Test validation of rate units."""
        assert validate_units("bbl_per_day", "rate")
        assert validate_units("bbl/month", "rate")
        assert not validate_units("bbl", "rate")  # Missing time component

    def test_validate_volume_units(self):
        """Test validation of volume units."""
        assert validate_units("bbl", "volume")
        assert validate_units("mcf", "volume")
        assert not validate_units("invalid", "volume")

    def test_validate_invalid_input(self):
        """Test validation with invalid input."""
        assert not validate_units("", "rate")
        assert not validate_units(None, "rate")  # type: ignore
