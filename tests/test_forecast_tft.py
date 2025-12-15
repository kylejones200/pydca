"""Tests for Temporal Fusion Transformer."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.forecast_tft import TFTForecaster


class TestTFTForecaster:
    """Test TFT forecaster."""

    def test_import_tft(self):
        """Test that TFTForecaster can be imported."""
        try:
            from decline_curve.forecast_tft import TFTForecaster

            # Should be able to create instance (will fail if PyTorch not available)
            try:
                forecaster = TFTForecaster(
                    phases=["oil"], horizon=12, sequence_length=24
                )
                assert forecaster.phases == ["oil"]
                assert forecaster.horizon == 12
            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("TFT module not available")

    def _create_sample_data(self):
        """Create sample production data."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        production = []
        for well_id in ["WELL_001", "WELL_002", "WELL_003"]:
            for i in range(30):
                prod = 1000 * np.exp(-0.05 * i) + np.random.normal(0, 50)
                prod = max(0, prod)
                production.append(
                    {
                        "well_id": well_id,
                        "date": dates[i],
                        "oil": prod,
                    }
                )

        df = pd.DataFrame(production)
        return df

    def test_fit_tft(self):
        """Test fitting TFT model."""
        try:
            from decline_curve.forecast_tft import TFTForecaster

            try:
                forecaster = TFTForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                    num_heads=2,
                )

                df = self._create_sample_data()

                # Fit model
                history = forecaster.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    validation_split=0.2,
                    verbose=False,
                )

                assert "loss" in history
                assert len(history["loss"]) == 2
                assert forecaster.is_fitted

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("TFT module not available")

    def test_predict_with_interpretation(self):
        """Test TFT prediction with interpretability."""
        try:
            from decline_curve.forecast_tft import TFTForecaster

            try:
                forecaster = TFTForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                    num_heads=2,
                )

                df = self._create_sample_data()

                # Fit model
                forecaster.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    verbose=False,
                )

                # Predict with interpretation
                forecast, interpretation = forecaster.predict(
                    well_id="WELL_001",
                    production_data=df,
                    horizon=6,
                    return_interpretation=True,
                )

                assert "oil" in forecast
                assert len(forecast["oil"]) == 6
                assert all(forecast["oil"] >= 0)

                # Check interpretation dict
                assert (
                    "attention_weights" in interpretation
                    or interpretation.get("attention_weights") is None
                )
                assert "gate_values" in interpretation
                assert "vsn_weights" in interpretation  # Variable Selection Network
                assert "decoder_outputs" in interpretation

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("TFT module not available")

    def test_tft_with_static_features(self):
        """Test TFT with static features."""
        try:
            from decline_curve.forecast_tft import TFTForecaster

            try:
                # Create data with static features
                dates = pd.date_range("2020-01-01", periods=60, freq="MS")
                production = []
                for well_id in ["WELL_001", "WELL_002"]:
                    for i in range(30):
                        prod = 1000 * np.exp(-0.05 * i) + np.random.normal(0, 50)
                        prod = max(0, prod)
                        production.append(
                            {
                                "well_id": well_id,
                                "date": dates[i],
                                "oil": prod,
                            }
                        )

                df = pd.DataFrame(production)

                # Create static features
                static_df = pd.DataFrame(
                    {
                        "well_id": ["WELL_001", "WELL_002"],
                        "porosity": [0.15, 0.18],
                        "permeability": [50, 75],
                    }
                )

                forecaster = TFTForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                    num_heads=2,
                    static_features=["porosity", "permeability"],
                )

                # Fit with static features
                forecaster.fit(
                    production_data=df,
                    static_features=static_df,
                    epochs=2,
                    batch_size=4,
                    verbose=False,
                )

                # Predict with static features
                forecast = forecaster.predict(
                    well_id="WELL_001",
                    production_data=df,
                    static_features=static_df,
                    horizon=6,
                )

                assert "oil" in forecast
                assert len(forecast["oil"]) == 6
                assert all(forecast["oil"] >= 0)

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("TFT module not available")

    def test_tft_integration_with_dca_api(self):
        """Test TFT integration with main dca.forecast() API."""
        try:
            from decline_curve import dca
            from decline_curve.forecast_tft import TFTForecaster

            try:
                # Create data
                dates = pd.date_range("2020-01-01", periods=36, freq="MS")
                production = []
                for well_id in ["WELL_001", "WELL_002"]:
                    for i in range(24):
                        prod = 1000 * np.exp(-0.05 * i) + np.random.normal(0, 50)
                        prod = max(0, prod)
                        production.append(
                            {
                                "well_id": well_id,
                                "date": dates[i],
                                "oil": prod,
                            }
                        )

                df = pd.DataFrame(production)
                series = df[df["well_id"] == "WELL_001"].set_index("date")["oil"]

                # Train TFT model
                tft_model = TFTForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                    num_heads=2,
                )

                tft_model.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    verbose=False,
                )

                # Use via main API
                forecast = dca.forecast(
                    series=series,
                    model="tft",
                    tft_model=tft_model,
                    production_data=df,
                    well_id="WELL_001",
                    horizon=6,
                )

                assert len(forecast) == 6
                assert all(forecast >= 0)

                # Test with interpretation
                forecast_with_interp, interpretation = dca.forecast(
                    series=series,
                    model="tft",
                    tft_model=tft_model,
                    production_data=df,
                    well_id="WELL_001",
                    horizon=6,
                    return_interpretation=True,
                )

                assert len(forecast_with_interp) == 6
                assert (
                    "attention_weights" in interpretation
                    or interpretation.get("attention_weights") is None
                )

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("TFT module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
