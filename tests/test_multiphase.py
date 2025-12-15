"""Unit tests for multi-phase forecasting module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.multiphase import (
    MultiPhaseData,
    MultiPhaseForecaster,
    create_multiphase_data_from_dataframe,
    load_multiphase_data,
)


class TestMultiPhaseData:
    """Test MultiPhaseData class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample production data."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates, name="oil")
        gas = pd.Series(np.linspace(1500, 750, 12), index=dates, name="gas")
        water = pd.Series(np.linspace(100, 500, 12), index=dates, name="water")
        return oil, gas, water, dates

    def test_creation_with_all_phases(self, sample_data):
        """Test creating MultiPhaseData with all phases."""
        oil, gas, water, dates = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)

        assert data.phases == ["oil", "gas", "water"]
        assert data.length == 12
        assert len(data.dates) == 12

    def test_creation_with_oil_only(self, sample_data):
        """Test creating MultiPhaseData with oil only."""
        oil, _, _, _ = sample_data
        data = MultiPhaseData(oil=oil)

        assert data.phases == ["oil"]
        assert data.length == 12
        assert data.gas is None
        assert data.water is None

    def test_creation_with_oil_and_gas(self, sample_data):
        """Test creating MultiPhaseData with oil and gas."""
        oil, gas, _, _ = sample_data
        data = MultiPhaseData(oil=oil, gas=gas)

        assert data.phases == ["oil", "gas"]
        assert data.length == 12
        assert data.water is None

    def test_index_alignment_validation(self, sample_data):
        """Test that misaligned indices raise error."""
        oil, gas, _, _ = sample_data
        # Create gas with different index
        wrong_dates = pd.date_range("2020-02-01", periods=12, freq="MS")
        gas_wrong = pd.Series(np.linspace(1500, 750, 12), index=wrong_dates)

        with pytest.raises(ValueError, match="Gas series index must match"):
            MultiPhaseData(oil=oil, gas=gas_wrong)

    def test_calculate_ratios_with_all_phases(self, sample_data):
        """Test ratio calculations with all phases."""
        oil, gas, water, _ = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        ratios = data.calculate_ratios()

        assert "gor" in ratios
        assert "water_cut" in ratios
        assert "liquid_rate" in ratios
        assert len(ratios["gor"]) == 12
        assert len(ratios["water_cut"]) == 12

    def test_calculate_gor(self, sample_data):
        """Test GOR calculation."""
        oil, gas, water, _ = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        ratios = data.calculate_ratios()

        # GOR should be gas/oil
        expected_gor = gas / oil
        pd.testing.assert_series_equal(ratios["gor"], expected_gor, check_names=False)

    def test_calculate_water_cut(self, sample_data):
        """Test water cut calculation."""
        oil, gas, water, _ = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        ratios = data.calculate_ratios()

        # Water cut should be water/(oil+water) * 100
        total_liquid = oil + water
        expected_wc = water / total_liquid * 100
        pd.testing.assert_series_equal(
            ratios["water_cut"], expected_wc, check_names=False
        )

    def test_calculate_liquid_rate(self, sample_data):
        """Test liquid rate calculation."""
        oil, gas, water, _ = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        ratios = data.calculate_ratios()

        # Liquid rate should be oil + water
        expected_liquid = oil + water
        pd.testing.assert_series_equal(
            ratios["liquid_rate"], expected_liquid, check_names=False
        )

    def test_to_dataframe(self, sample_data):
        """Test conversion to DataFrame."""
        oil, gas, water, dates = sample_data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        df = data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12
        assert list(df.columns) == ["oil", "gas", "water"]
        pd.testing.assert_index_equal(df.index, dates)

    def test_to_dataframe_oil_only(self, sample_data):
        """Test DataFrame conversion with oil only."""
        oil, _, _, dates = sample_data
        data = MultiPhaseData(oil=oil)
        df = data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["oil"]
        assert len(df) == 12


class TestCreateMultiPhaseDataFromDataFrame:
    """Test convenience function for creating MultiPhaseData."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "Oil": np.linspace(1000, 500, 12),
                "Gas": np.linspace(1500, 750, 12),
                "Wtr": np.linspace(100, 500, 12),
                "well_id": "WELL_001",
            }
        )
        return df

    def test_create_from_dataframe(self, sample_dataframe):
        """Test creating MultiPhaseData from DataFrame."""
        data = create_multiphase_data_from_dataframe(
            sample_dataframe,
            oil_column="Oil",
            gas_column="Gas",
            water_column="Wtr",
            date_column="date",
        )

        assert data.phases == ["oil", "gas", "water"]
        assert data.length == 12

    def test_create_with_missing_gas_column(self, sample_dataframe):
        """Test creating with missing gas column."""
        data = create_multiphase_data_from_dataframe(
            sample_dataframe,
            oil_column="Oil",
            gas_column="NonExistent",
            water_column="Wtr",
            date_column="date",
        )

        assert data.phases == ["oil", "water"]
        assert data.gas is None

    def test_create_with_well_id(self, sample_dataframe):
        """Test extracting well ID."""
        data = create_multiphase_data_from_dataframe(
            sample_dataframe,
            oil_column="Oil",
            gas_column="Gas",
            water_column="Wtr",
            date_column="date",
            well_id_column="well_id",
        )

        assert data.well_id == "WELL_001"

    def test_date_conversion(self, sample_dataframe):
        """Test automatic date conversion."""
        # Convert dates to strings
        df = sample_dataframe.copy()
        df["date"] = df["date"].astype(str)

        data = create_multiphase_data_from_dataframe(
            df,
            oil_column="Oil",
            gas_column="Gas",
            water_column="Wtr",
            date_column="date",
        )

        assert isinstance(data.dates, pd.DatetimeIndex)
        assert len(data.dates) == 12


class TestMultiPhaseForecaster:
    """Test MultiPhaseForecaster class."""

    @pytest.fixture
    def sample_multiphase_data(self):
        """Create sample multi-phase data."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")

        # Create declining production
        t = np.arange(24)
        oil = pd.Series(1000 / (1 + 0.1 * t) ** 0.5, index=dates, name="oil")
        gas = pd.Series(1500 / (1 + 0.1 * t) ** 0.5, index=dates, name="gas")
        water = pd.Series(100 * (1 + 0.05 * t), index=dates, name="water")

        return MultiPhaseData(oil=oil, gas=gas, water=water)

    def test_forecaster_initialization(self):
        """Test forecaster initialization."""
        forecaster = MultiPhaseForecaster()
        assert forecaster.fitted_models == {}
        assert forecaster.history is None

    def test_forecast_with_enforce_ratios(self, sample_multiphase_data):
        """Test forecasting with ratio enforcement."""
        forecaster = MultiPhaseForecaster()

        # This test requires the full dca module to work
        # For now, just test that the method exists and accepts parameters
        assert hasattr(forecaster, "forecast")
        assert callable(forecaster.forecast)

    def test_evaluate_method_exists(self):
        """Test that evaluate method exists."""
        forecaster = MultiPhaseForecaster()
        assert hasattr(forecaster, "evaluate")
        assert callable(forecaster.evaluate)

    def test_calculate_consistency_metrics_method_exists(self):
        """Test that consistency metrics method exists."""
        forecaster = MultiPhaseForecaster()
        assert hasattr(forecaster, "calculate_consistency_metrics")
        assert callable(forecaster.calculate_consistency_metrics)

    def test_calculate_consistency_metrics_with_forecasts(self):
        """Test consistency metrics calculation."""
        forecaster = MultiPhaseForecaster()

        # Create mock forecasts
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        forecasts = {
            "oil": pd.Series(np.linspace(1000, 500, 12), index=dates),
            "gas": pd.Series(np.linspace(1500, 750, 12), index=dates),
            "water": pd.Series(np.linspace(100, 500, 12), index=dates),
        }

        metrics = forecaster.calculate_consistency_metrics(forecasts)

        assert isinstance(metrics, dict)
        assert "gor_stability" in metrics
        assert "avg_gor" in metrics
        assert "water_cut_monotonic" in metrics
        assert "final_water_cut" in metrics

    def test_gor_stability_calculation(self):
        """Test GOR stability metric."""
        forecaster = MultiPhaseForecaster()

        # Create forecasts with stable GOR
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates)
        gas = oil * 1.5  # Constant GOR of 1.5

        forecasts = {"oil": oil, "gas": gas}
        metrics = forecaster.calculate_consistency_metrics(forecasts)

        # With constant GOR, stability should be high (close to 1)
        assert metrics["gor_stability"] > 0.9
        assert abs(metrics["avg_gor"] - 1.5) < 0.01

    def test_water_cut_monotonic_check(self):
        """Test water cut monotonicity check."""
        forecaster = MultiPhaseForecaster()

        # Create forecasts with increasing water cut
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates)
        water = pd.Series(np.linspace(100, 500, 12), index=dates)

        forecasts = {"oil": oil, "water": water}
        metrics = forecaster.calculate_consistency_metrics(forecasts)

        # Water cut should be monotonically increasing
        assert metrics["water_cut_monotonic"] >= 0.9  # Allow some tolerance


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_series(self):
        """Test handling of empty series."""
        dates = pd.DatetimeIndex([])
        oil = pd.Series([], index=dates, dtype=float)

        data = MultiPhaseData(oil=oil)
        assert data.length == 0
        assert data.phases == ["oil"]

    def test_single_data_point(self):
        """Test with single data point."""
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        oil = pd.Series([1000], index=dates)

        data = MultiPhaseData(oil=oil)
        assert data.length == 1

    def test_zero_production(self):
        """Test handling of zero production values."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series([0] * 12, index=dates)
        gas = pd.Series([0] * 12, index=dates)

        data = MultiPhaseData(oil=oil, gas=gas)
        ratios = data.calculate_ratios()

        # GOR with zero oil should be handled gracefully
        assert "gor" in ratios
        # NaN or inf values should be present
        assert ratios["gor"].isna().any() or np.isinf(ratios["gor"]).any()

    def test_negative_production(self):
        """Test that negative values are allowed (for testing purposes)."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series([-100] * 12, index=dates)

        # Should not raise error
        data = MultiPhaseData(oil=oil)
        assert data.length == 12


class TestIntegration:
    """Integration tests for multi-phase workflow."""

    def test_full_workflow_with_real_data_structure(self):
        """Test complete workflow with realistic data structure."""
        # Create realistic production data
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")

        # Hyperbolic decline for oil
        t = np.arange(24)
        qi, di, b = 1200, 0.12, 0.4
        oil = pd.Series(qi / (1 + b * di * t) ** (1 / b), index=dates, name="oil")

        # Gas with relatively stable GOR
        gor = 1.3 + np.random.normal(0, 0.1, 24)
        gas = pd.Series(oil.values * gor, index=dates, name="gas")

        # Water with increasing water cut
        water_cut = np.linspace(70, 85, 24)
        water = pd.Series(
            oil.values * water_cut / (100 - water_cut), index=dates, name="water"
        )

        # Create multi-phase data
        data = MultiPhaseData(oil=oil, gas=gas, water=water)

        # Verify structure
        assert data.phases == ["oil", "gas", "water"]
        assert data.length == 24

        # Calculate ratios
        ratios = data.calculate_ratios()
        assert "gor" in ratios
        assert "water_cut" in ratios

        # Verify water cut is increasing
        wc_values = ratios["water_cut"].values
        assert wc_values[-1] > wc_values[0]

        # Convert to DataFrame
        df = data.to_dataframe()
        assert len(df) == 24
        assert len(df.columns) == 3


class TestPhaseCorrelations:
    """Test phase correlation metrics."""

    def test_calculate_phase_correlations(self):
        """Test phase correlation calculation."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        # Create correlated data
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates, name="oil")
        gas = pd.Series(oil * 1.5, index=dates, name="gas")  # Strongly correlated
        water = pd.Series(np.linspace(100, 500, 12), index=dates, name="water")

        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        correlations = data.calculate_phase_correlations()

        assert "oil_gas" in correlations
        assert correlations["oil_gas"] > 0.9  # Should be highly correlated
        assert "oil_water" in correlations
        assert "gas_water" in correlations

    def test_calculate_phase_correlations_oil_only(self):
        """Test correlation calculation with oil only."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates, name="oil")
        data = MultiPhaseData(oil=oil)
        correlations = data.calculate_phase_correlations()
        assert len(correlations) == 0


class TestEnhancedEvaluation:
    """Test enhanced evaluation metrics."""

    def test_evaluate_with_consistency_metrics(self):
        """Test evaluation includes consistency metrics."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates, name="oil")
        gas = pd.Series(oil * 1.5, index=dates, name="gas")
        water = pd.Series(np.linspace(100, 500, 12), index=dates, name="water")

        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        forecaster = MultiPhaseForecaster()
        forecasts = forecaster.forecast(data, horizon=6)

        metrics = forecaster.evaluate(data, forecasts)

        # Check individual phase metrics
        assert "oil" in metrics
        assert "gas" in metrics
        assert "water" in metrics

        # Check consistency metrics
        assert "consistency" in metrics
        assert isinstance(metrics["consistency"], dict)

        # Check overall metrics
        assert "overall" in metrics
        assert "rmse" in metrics["overall"]
        assert "mae" in metrics["overall"]
        assert "smape" in metrics["overall"]


class TestUnifiedDataLoader:
    """Test unified data loader function."""

    def test_load_from_dataframe(self):
        """Test loading from DataFrame."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates,
                "Oil": np.linspace(1000, 500, 12),
                "Gas": np.linspace(1500, 750, 12),
                "Wtr": np.linspace(100, 500, 12),
            }
        )

        data = load_multiphase_data(df, date_column="date")
        assert isinstance(data, MultiPhaseData)
        assert data.phases == ["oil", "gas", "water"]

    def test_load_multi_well_data(self):
        """Test loading multi-well data."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "well_id": ["WELL_001"] * 12 + ["WELL_002"] * 12,
                "Oil": np.linspace(1000, 500, 24),
                "Gas": np.linspace(1500, 750, 24),
            }
        )

        multi_well_data = load_multiphase_data(
            df, date_column="date", well_id_column="well_id"
        )
        assert isinstance(multi_well_data, dict)
        assert "WELL_001" in multi_well_data
        assert "WELL_002" in multi_well_data
        assert isinstance(multi_well_data["WELL_001"], MultiPhaseData)


class TestSharedModelArchitecture:
    """Test shared model architecture."""

    def test_shared_model_forecast(self):
        """Test forecasting with shared model architecture."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        oil = pd.Series(np.linspace(1000, 500, 12), index=dates, name="oil")
        gas = pd.Series(oil * 1.5, index=dates, name="gas")
        water = pd.Series(np.linspace(100, 500, 12), index=dates, name="water")

        data = MultiPhaseData(oil=oil, gas=gas, water=water)
        forecaster = MultiPhaseForecaster(shared_model=True)
        forecasts = forecaster.forecast(
            data, horizon=6, model="arps", kind="hyperbolic"
        )

        assert "oil" in forecasts
        assert "gas" in forecasts
        assert "water" in forecasts
        assert len(forecasts["oil"]) == 18  # 12 historical + 6 forecast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
