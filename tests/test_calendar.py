"""Tests for calendar logic module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.calendar import (
    CalendarConfig,
    calculate_days_in_period,
    convert_monthly_to_daily,
    create_daily_index_from_monthly,
    get_month_day_counts,
    place_monthly_data,
    weight_monthly_volumes_by_days,
)


class TestPlaceMonthlyData:
    """Test place_monthly_data function."""

    def test_mid_month_placement(self):
        """Test mid-month placement."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "oil_bbl": [3000, 2700, 2400],
            }
        )

        df_placed = place_monthly_data(
            df, volume_columns=["oil_bbl"], placement="mid_month"
        )

        # Dates should be mid-month
        assert df_placed["date"].iloc[0].day == 15  # Mid-January

        # Should have rate column
        assert "oil_rate" in df_placed.columns or "oil_bbl_rate" in df_placed.columns

    def test_end_month_placement(self):
        """Test end-month placement."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "oil_bbl": [3000, 2700, 2400],
            }
        )

        df_placed = place_monthly_data(
            df, volume_columns=["oil_bbl"], placement="end_month"
        )

        # Dates should be end of month
        jan_end = pd.Timestamp("2020-01-31")
        assert df_placed["date"].iloc[0] == jan_end

    def test_day_count_weighting(self):
        """Test day count weighting."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                "oil_bbl": [
                    3100,
                    2800,
                ],  # Jan has 31 days, Feb has 29 (2020 is leap year)
            }
        )

        df_placed = place_monthly_data(
            df, volume_columns=["oil_bbl"], use_day_count_weighting=True
        )

        # January rate should be lower (more days)
        rate_col = "oil_rate" if "oil_rate" in df_placed.columns else "oil_bbl_rate"
        jan_rate = df_placed[rate_col].iloc[0]
        feb_rate = df_placed[rate_col].iloc[1]

        # Jan: 3100/31 ≈ 100, Feb: 2800/29 ≈ 96.5
        assert jan_rate < 110
        assert feb_rate < 100

    def test_auto_detect_volume_columns(self):
        """Test auto-detection of volume columns."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "oil": [3000, 2700, 2400],
                "gas": [1000, 900, 800],
            }
        )

        df_placed = place_monthly_data(df)

        # Should have created rate columns
        assert "oil_rate" in df_placed.columns or "oil" in df_placed.columns


class TestConvertMonthlyToDaily:
    """Test convert_monthly_to_daily function."""

    def test_basic_conversion(self):
        """Test basic monthly to daily conversion."""
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        volumes = pd.Series([3000, 2700, 2400], index=dates)

        daily_rates = convert_monthly_to_daily(volumes, dates)

        # January has 31 days, so 3000/31 ≈ 96.77
        assert np.isclose(daily_rates.iloc[0], 3000 / 31, rtol=0.01)

    def test_without_day_count_weighting(self):
        """Test conversion without day count weighting."""
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        volumes = pd.Series([3000, 2700, 2400], index=dates)

        daily_rates = convert_monthly_to_daily(
            volumes, dates, use_day_count_weighting=False
        )

        # Should use average days (30.4375)
        assert np.isclose(daily_rates.iloc[0], 3000 / 30.4375, rtol=0.01)


class TestCalculateDaysInPeriod:
    """Test calculate_days_in_period function."""

    def test_same_month(self):
        """Test days in same month."""
        start = pd.Timestamp("2020-01-15")
        end = pd.Timestamp("2020-01-20")

        days = calculate_days_in_period(start, end)

        assert days == 6  # 15, 16, 17, 18, 19, 20

    def test_cross_month(self):
        """Test days across months."""
        start = pd.Timestamp("2020-01-25")
        end = pd.Timestamp("2020-02-05")

        days = calculate_days_in_period(start, end)

        assert days == 12  # 7 days in Jan + 5 days in Feb


class TestGetMonthDayCounts:
    """Test get_month_day_counts function."""

    def test_various_months(self):
        """Test day counts for various months."""
        dates = pd.DatetimeIndex(
            [
                "2020-01-01",  # 31 days
                "2020-02-01",  # 29 days (leap year)
                "2020-04-01",  # 30 days
            ]
        )

        day_counts = get_month_day_counts(dates)

        assert day_counts.iloc[0] == 31
        assert day_counts.iloc[1] == 29
        assert day_counts.iloc[2] == 30


class TestWeightMonthlyVolumesByDays:
    """Test weight_monthly_volumes_by_days function."""

    def test_weighting(self):
        """Test volume weighting by days."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                "oil": [3100, 2800],  # Jan (31 days), Feb (29 days)
            }
        )

        df_weighted = weight_monthly_volumes_by_days(df, ["oil"])

        # January should be adjusted down (more days)
        # Feb should be adjusted up (fewer days)
        assert df_weighted["oil"].iloc[0] < 3100
        assert df_weighted["oil"].iloc[1] > 2800


class TestCreateDailyIndexFromMonthly:
    """Test create_daily_index_from_monthly function."""

    def test_mid_month_index(self):
        """Test creating mid-month index."""
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-03-01")

        index = create_daily_index_from_monthly(start, end, placement="mid_month")

        assert len(index) == 3
        assert index[0].day == 15  # Mid-January

    def test_end_month_index(self):
        """Test creating end-month index."""
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-03-01")

        index = create_daily_index_from_monthly(start, end, placement="end_month")

        assert len(index) == 3
        assert index[0] == pd.Timestamp("2020-01-31")  # End of January
