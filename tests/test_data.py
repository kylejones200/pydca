"""Tests for data loading and processing utilities."""

import os
import tempfile

import pandas as pd
import pytest

from decline_curve.data import (
    load_price_csv,
    load_production_csvs,
    make_panel,
    to_monthly,
)


class TestDataLoading:
    """Test data loading functions."""

    def test_load_production_csvs(self):
        """Test loading multiple production CSV files."""
        # Create temporary CSV files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            csv1 = os.path.join(tmpdir, "well1.csv")
            csv2 = os.path.join(tmpdir, "well2.csv")

            df1 = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "well_id": ["WELL_001"] * 12,
                    "oil_bbl": [1000 - i * 50 for i in range(12)],
                }
            )
            df1.to_csv(csv1, index=False)

            df2 = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "well_id": ["WELL_002"] * 12,
                    "oil_bbl": [800 - i * 40 for i in range(12)],
                }
            )
            df2.to_csv(csv2, index=False)

            # Load and test
            result = load_production_csvs([csv1, csv2])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 24
            assert "well_id" in result.columns
            assert "oil_bbl" in result.columns
            assert isinstance(result.index, pd.DatetimeIndex)

    def test_load_production_csvs_missing_cols(self):
        """Test error handling for missing columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame({"date": ["2020-01-01"], "oil_bbl": [1000]})
            df.to_csv(csv_file, index=False)

            with pytest.raises(ValueError, match="Missing columns"):
                load_production_csvs([csv_file])

    def test_to_monthly(self):
        """Test monthly aggregation."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "well_id": ["WELL_001"] * 60,
                "oil_bbl": [100] * 60,
            }
        ).set_index("date")

        result = to_monthly(df)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_make_panel(self):
        """Test panel creation with relative time."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "well_id": ["WELL_001"] * 12 + ["WELL_002"] * 12,
                "oil_bbl": [1000 - i * 50 for i in range(24)],
            }
        ).set_index("date")

        result = make_panel(df)

        assert "t" in result.columns
        assert result["t"].max() == 11  # 0-11 for 12 months per well

    def test_make_panel_first_n_months(self):
        """Test panel creation with truncation."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "well_id": ["WELL_001"] * 24,
                "oil_bbl": [1000 - i * 50 for i in range(24)],
            }
        ).set_index("date")

        result = make_panel(df, first_n_months=12)

        assert result["t"].max() == 11  # Truncated to 12 months

    def test_load_price_csv(self):
        """Test loading price CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "prices.csv")
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                    "price": [70 + i for i in range(12)],
                }
            )
            df.to_csv(csv_file, index=False)

            result = load_price_csv(csv_file)

            assert isinstance(result, pd.DataFrame)
            assert "price" in result.columns
            assert isinstance(result.index, pd.DatetimeIndex)
            assert len(result) == 12

    def test_load_price_csv_missing_cols(self):
        """Test error handling for missing columns in price CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "prices.csv")
            df = pd.DataFrame({"date": ["2020-01-01"]})
            df.to_csv(csv_file, index=False)

            with pytest.raises(ValueError, match="Missing columns"):
                load_price_csv(csv_file)

    def test_assert_cols_via_load(self):
        """Test column assertion via load functions."""
        # Test that missing columns are caught by load functions
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame({"date": ["2020-01-01"], "col2": [2]})  # Missing well_id
            df.to_csv(csv_file, index=False)

            # Should raise ValueError for missing columns
            with pytest.raises(ValueError, match="Missing columns"):
                load_production_csvs([csv_file])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
