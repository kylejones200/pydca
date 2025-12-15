"""Tests for EUR estimation module."""

import numpy as np
import pandas as pd

from decline_curve.eur_estimation import (
    calculate_eur_batch,
    calculate_eur_from_production,
)


class TestCalculateEURFromProduction:
    """Test EUR calculation from production data."""

    def test_hyperbolic_decline(self):
        """Test EUR calculation with hyperbolic decline."""
        months = pd.Series(np.arange(0, 36, 1))
        # Hyperbolic decline: qi=1000, di=0.1, b=0.5
        production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))
        production = pd.Series(production)

        result = calculate_eur_from_production(
            months, production, model_type="hyperbolic"
        )

        assert result is not None
        assert "eur" in result
        assert "qi" in result
        assert "di" in result
        assert "b" in result
        assert result["eur"] > 0
        assert result["model_type"] == "hyperbolic"

    def test_exponential_decline(self):
        """Test EUR calculation with exponential decline."""
        months = pd.Series(np.arange(0, 36, 1))
        # Exponential decline: qi=1000, di=0.1
        production = 1000 * np.exp(-0.1 * months)
        production = pd.Series(production)

        result = calculate_eur_from_production(
            months, production, model_type="exponential"
        )

        assert result is not None
        assert result["eur"] > 0
        assert result["model_type"] == "exponential"

    def test_insufficient_data(self):
        """Test with insufficient data."""
        months = pd.Series(np.arange(0, 3, 1))
        production = pd.Series([1000, 900, 800])

        result = calculate_eur_from_production(months, production)

        assert result is None

    def test_economic_limit(self):
        """Test EUR calculation with economic limit."""
        months = pd.Series(np.arange(0, 36, 1))
        production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))
        production = pd.Series(production)

        result_low = calculate_eur_from_production(months, production, econ_limit=5.0)
        result_high = calculate_eur_from_production(months, production, econ_limit=50.0)

        assert result_low is not None
        assert result_high is not None
        # Lower economic limit should give higher EUR
        assert result_low["eur"] > result_high["eur"]


class TestCalculateEURBatch:
    """Test batch EUR calculation."""

    def test_basic_batch(self):
        """Test basic batch calculation."""
        # Create sample data
        data = []
        for well_id in ["WELL_001", "WELL_002", "WELL_003"]:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            months = np.arange(len(dates))
            production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))

            for date, prod in zip(dates, production):
                data.append({"well_id": well_id, "date": date, "oil_bbl": max(0, prod)})

        df = pd.DataFrame(data)

        results = calculate_eur_batch(
            df,
            well_id_col="well_id",
            date_col="date",
            value_col="oil_bbl",
            model_type="hyperbolic",
            min_months=12,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "well_id" in results.columns
        assert "eur" in results.columns
        assert all(results["eur"] > 0)

    def test_insufficient_data_filtering(self):
        """Test that wells with insufficient data are filtered."""
        data = []
        # Well with enough data
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        for date in dates:
            data.append({"well_id": "WELL_001", "date": date, "oil_bbl": 1000})

        # Well with insufficient data
        dates_short = pd.date_range("2020-01-01", periods=3, freq="MS")
        for date in dates_short:
            data.append({"well_id": "WELL_002", "date": date, "oil_bbl": 1000})

        df = pd.DataFrame(data)

        results = calculate_eur_batch(df, min_months=12, model_type="hyperbolic")

        assert len(results) == 1
        assert results.iloc[0]["well_id"] == "WELL_001"

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["well_id", "date", "oil_bbl"])

        results = calculate_eur_batch(df)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
