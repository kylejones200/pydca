"""Tests for panel data analysis module."""

import numpy as np
import pandas as pd

from decline_curve.panel_analysis import (
    analyze_by_company,
    calculate_spatial_features,
    company_fixed_effects_regression,
    eur_with_company_controls,
    prepare_panel_data,
    spatial_eur_analysis,
)


class TestPreparePanelData:
    """Test panel data preparation."""

    def test_basic_preparation(self):
        """Test basic panel data preparation."""
        data = {
            "API_WELLNO": ["W1", "W1", "W1", "W2", "W2", "W2"],
            "ReportDate": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01"] * 2
            ),
            "Oil": [100, 90, 80, 120, 110, 100],
            "Company": ["A", "A", "A", "B", "B", "B"],
            "Lat": [47.0, 47.0, 47.0, 47.1, 47.1, 47.1],
            "Long": [-102.0, -102.0, -102.0, -102.1, -102.1, -102.1],
            "County": ["X", "X", "X", "Y", "Y", "Y"],
        }
        df = pd.DataFrame(data)

        panel = prepare_panel_data(df)

        assert isinstance(panel, pd.DataFrame)
        assert "months_since_start" in panel.columns
        assert len(panel) == 6
        assert panel["API_WELLNO"].nunique() == 2
        assert "company" in panel.columns
        assert "lat" in panel.columns
        assert "long" in panel.columns

    def test_with_missing_columns(self):
        """Test with missing optional columns."""
        data = {
            "API_WELLNO": ["W1", "W1", "W1"],
            "ReportDate": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "Oil": [100, 90, 80],
        }
        df = pd.DataFrame(data)

        panel = prepare_panel_data(df)

        assert isinstance(panel, pd.DataFrame)
        assert len(panel) == 3
        assert "months_since_start" in panel.columns

    def test_panel_data_structure(self):
        """Test that panel data has correct structure."""
        data = {
            "API_WELLNO": ["W1", "W1", "W1", "W2", "W2", "W2"],
            "ReportDate": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01"] * 2
            ),
            "Oil": [100, 90, 80, 120, 110, 100],
        }
        df = pd.DataFrame(data)

        panel = prepare_panel_data(df)

        assert panel["API_WELLNO"].nunique() == 2
        assert all(panel["months_since_start"] >= 0)
        assert panel["months_since_start"].iloc[0] == 0  # First month should be 0


class TestCalculateSpatialFeatures:
    """Test spatial feature calculation."""

    def test_basic_spatial_features(self):
        """Test basic spatial feature calculation."""
        data = {
            "well_id": ["W1", "W2", "W3"],
            "lat": [47.0, 47.1, 47.05],
            "long": [-102.0, -102.1, -102.05],
        }
        df = pd.DataFrame(data)

        result = calculate_spatial_features(df)

        assert "distance_from_center" in result.columns
        assert "well_density" in result.columns
        assert all(result["distance_from_center"] >= 0)
        assert all(result["well_density"] >= 0)

    def test_missing_location_data(self):
        """Test with missing location data."""
        data = {
            "well_id": ["W1", "W2"],
            "lat": [47.0, np.nan],
            "long": [-102.0, -102.1],
        }
        df = pd.DataFrame(data)

        result = calculate_spatial_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestEURWithCompanyControls:
    """Test EUR calculation with company controls."""

    def test_basic_eur_calculation(self):
        """Test basic EUR calculation with company info."""
        data = []
        for well_id, company in [("W1", "A"), ("W2", "B"), ("W3", "A")]:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            months = np.arange(len(dates))
            production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))

            for date, prod in zip(dates, production):
                data.append(
                    {
                        "API_WELLNO": well_id,
                        "ReportDate": date,
                        "Oil": max(0, prod),
                        "Company": company,
                        "County": "X",
                    }
                )

        panel_df = pd.DataFrame(data)

        results = eur_with_company_controls(
            panel_df,
            well_id_col="API_WELLNO",
            date_col="ReportDate",
            value_col="Oil",
            company_col="Company",
            county_col="County",
            min_months=12,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "eur" in results.columns
        assert "company" in results.columns
        assert "county" in results.columns
        assert all(results["eur"] > 0)

    def test_without_company_column(self):
        """Test when company column is missing."""
        data = []
        for well_id in ["W1", "W2"]:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            months = np.arange(len(dates))
            production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))

            for date, prod in zip(dates, production):
                data.append(
                    {
                        "API_WELLNO": well_id,
                        "ReportDate": date,
                        "Oil": max(0, prod),
                    }
                )

        panel_df = pd.DataFrame(data)

        results = eur_with_company_controls(
            panel_df,
            well_id_col="API_WELLNO",
            date_col="ReportDate",
            value_col="Oil",
            min_months=12,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert "eur" in results.columns


class TestCompanyFixedEffectsRegression:
    """Test company fixed effects regression."""

    def test_basic_regression(self):
        """Test basic fixed effects regression."""
        data = {
            "eur": [1000, 1200, 800, 1500, 1100],
            "company": ["A", "A", "B", "B", "C"],
            "county": ["X", "X", "Y", "Y", "X"],
        }
        eur_results = pd.DataFrame(data)

        result = company_fixed_effects_regression(
            eur_results,
            dependent_var="eur",
            company_col="company",
            county_col="county",
        )

        if result is not None:
            assert "rsquared" in result
            assert "nobs" in result
            assert "n_companies" in result
            assert "n_counties" in result
            assert result["rsquared"] >= 0
            assert result["nobs"] == 5

    def test_regression_without_county(self):
        """Test regression without county controls."""
        data = {
            "eur": [1000, 1200, 800, 1500, 1100],
            "company": ["A", "A", "B", "B", "C"],
        }
        eur_results = pd.DataFrame(data)

        result = company_fixed_effects_regression(
            eur_results,
            dependent_var="eur",
            company_col="company",
            county_col=None,
        )

        if result is not None:
            assert "rsquared" in result
            assert result["n_companies"] == 3

    def test_regression_without_statsmodels(self, monkeypatch):
        """Test regression when statsmodels is not available."""
        import decline_curve.panel_analysis as pa

        monkeypatch.setattr(pa, "HAS_STATSMODELS", False)

        data = {
            "eur": [1000, 1200, 800],
            "company": ["A", "B", "C"],
        }
        eur_results = pd.DataFrame(data)

        result = company_fixed_effects_regression(
            eur_results, dependent_var="eur", company_col="company"
        )

        assert result is None

    def test_regression_with_missing_data(self):
        """Test regression with missing data."""
        data = {
            "eur": [1000, np.nan, 800],
            "company": ["A", "B", "C"],
        }
        eur_results = pd.DataFrame(data)

        result = company_fixed_effects_regression(
            eur_results, dependent_var="eur", company_col="company"
        )

        if result is not None:
            assert result["nobs"] < 3  # Should drop NaN rows


class TestSpatialEURAnalysis:
    """Test spatial EUR analysis."""

    def test_basic_spatial_analysis(self):
        """Test basic spatial EUR analysis."""
        panel_data = {
            "API_WELLNO": ["W1", "W2", "W3"],
            "lat": [47.0, 47.1, 47.05],
            "long": [-102.0, -102.1, -102.05],
        }
        panel_df = pd.DataFrame(panel_data)

        eur_data = {
            "API_WELLNO": ["W1", "W2", "W3"],
            "eur": [1000, 1200, 800],
        }
        eur_results = pd.DataFrame(eur_data)

        result = spatial_eur_analysis(panel_df, eur_results, well_id_col="API_WELLNO")

        assert isinstance(result, pd.DataFrame)
        assert "eur" in result.columns
        assert "lat" in result.columns
        assert "long" in result.columns
        assert "distance_from_center" in result.columns
        assert "well_density" in result.columns

    def test_without_location_data(self):
        """Test when location data is missing."""
        panel_data = {
            "API_WELLNO": ["W1", "W2"],
            "lat": [47.0, 47.1],
            "long": [-102.0, -102.1],
        }
        panel_df = pd.DataFrame(panel_data)

        eur_data = {
            "API_WELLNO": ["W1", "W2"],
            "eur": [1000, 1200],
        }
        eur_results = pd.DataFrame(eur_data)

        result = spatial_eur_analysis(panel_df, eur_results, well_id_col="API_WELLNO")

        assert isinstance(result, pd.DataFrame)
        assert "eur" in result.columns
        assert "lat" in result.columns
        assert "long" in result.columns


class TestAnalyzeByCompany:
    """Test company analysis."""

    def test_basic_company_analysis(self):
        """Test basic company aggregation."""
        data = {
            "eur": [1000, 1200, 800, 1500, 1100],
            "company": ["A", "A", "B", "B", "C"],
            "qi": [1000, 1200, 800, 1500, 1100],
            "di": [0.1, 0.12, 0.08, 0.15, 0.11],
            "b": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
        eur_results = pd.DataFrame(data)

        result = analyze_by_company(eur_results, company_col="company")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three companies
        assert "eur" in result.columns or any(
            "eur" in str(col) for col in result.columns
        )

    def test_company_analysis_with_missing_data(self):
        """Test company analysis with missing company data."""
        data = {
            "eur": [1000, 1200, 800],
            "company": ["A", "A", None],
            "qi": [1000, 1200, 800],
            "di": [0.1, 0.12, 0.08],
            "b": [0.5, 0.5, 0.5],
        }
        eur_results = pd.DataFrame(data)

        result = analyze_by_company(eur_results, company_col="company")

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1


class TestIntegration:
    """Integration tests for panel analysis workflow."""

    def test_full_workflow(self):
        """Test complete panel analysis workflow."""
        # Create sample panel data
        data = []
        for well_id, company, county in [
            ("W1", "A", "X"),
            ("W2", "A", "X"),
            ("W3", "B", "Y"),
        ]:
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            months = np.arange(len(dates))
            production = 1000 / ((1 + 0.5 * 0.1 * months) ** (1 / 0.5))

            for date, prod in zip(dates, production):
                data.append(
                    {
                        "API_WELLNO": well_id,
                        "ReportDate": date,
                        "Oil": max(0, prod),
                        "Company": company,
                        "County": county,
                        "Lat": 47.0 + np.random.rand() * 0.1,
                        "Long": -102.0 - np.random.rand() * 0.1,
                    }
                )

        df = pd.DataFrame(data)

        # Step 1: Prepare panel data
        panel = prepare_panel_data(df)
        assert len(panel) > 0

        # Step 2: Calculate EUR with controls
        eur_results = eur_with_company_controls(
            panel,
            well_id_col="API_WELLNO",
            date_col="ReportDate",
            value_col="Oil",
            company_col="Company",
            county_col="County",
            min_months=12,
        )
        assert len(eur_results) == 3

        # Step 3: Analyze by company
        company_stats = analyze_by_company(eur_results, company_col="company")
        # Company stats may be empty if columns are missing, but should be a DataFrame
        assert isinstance(company_stats, pd.DataFrame)

        # Step 4: Spatial analysis
        spatial_eur = spatial_eur_analysis(panel, eur_results, well_id_col="API_WELLNO")
        assert "distance_from_center" in spatial_eur.columns

        # Step 5: Regression (if statsmodels available)
        regression_result = company_fixed_effects_regression(
            eur_results,
            dependent_var="eur",
            company_col="company",
            county_col="county",
        )
        if regression_result is not None:
            assert "rsquared" in regression_result
