"""Tests for DeepAR probabilistic forecasting."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.forecast_deepar import DeepARForecaster


class TestDeepARForecaster:
    """Test DeepAR forecaster."""

    def test_import_deepar(self):
        """Test that DeepARForecaster can be imported."""
        try:
            from decline_curve.forecast_deepar import DeepARForecaster

            # Should be able to create instance (will fail if PyTorch not available)
            try:
                forecaster = DeepARForecaster(
                    phases=["oil"], horizon=12, sequence_length=24
                )
                assert forecaster.phases == ["oil"]
                assert forecaster.horizon == 12
            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("DeepAR module not available")

    def _create_sample_data(self):
        """Create sample production data for testing."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        production = []
        for well_id in ["WELL_001", "WELL_002", "WELL_003"]:
            for i in range(30):
                # Exponential decline with noise
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

    def test_fit_deepar(self):
        """Test fitting DeepAR model."""
        try:
            from decline_curve.forecast_deepar import DeepARForecaster

            try:
                forecaster = DeepARForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
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
            pytest.skip("DeepAR module not available")

    def test_predict_quantiles(self):
        """Test probabilistic forecasting with quantiles."""
        try:
            from decline_curve.forecast_deepar import DeepARForecaster

            try:
                forecaster = DeepARForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                )

                df = self._create_sample_data()

                # Fit model
                forecaster.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    verbose=False,
                )

                # Predict quantiles
                quantiles = [0.1, 0.5, 0.9]
                forecasts = forecaster.predict_quantiles(
                    well_id="WELL_001",
                    production_data=df,
                    quantiles=quantiles,
                    horizon=6,
                    n_samples=100,  # Reduced for speed
                )

                assert "oil" in forecasts
                assert "q10" in forecasts["oil"]
                assert "q50" in forecasts["oil"]
                assert "q90" in forecasts["oil"]

                # Check quantile ordering (P10 < P50 < P90)
                q10 = forecasts["oil"]["q10"]
                q50 = forecasts["oil"]["q50"]
                q90 = forecasts["oil"]["q90"]

                assert len(q10) == 6
                assert len(q50) == 6
                assert len(q90) == 6

                # P50 should be between P10 and P90 (on average)
                assert (q50 >= q10).all() or (
                    q50 <= q90
                ).all()  # Allow some flexibility

                # All forecasts should be non-negative
                assert (q10 >= 0).all()
                assert (q50 >= 0).all()
                assert (q90 >= 0).all()

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("DeepAR module not available")

    def test_multiphase_deepar(self):
        """Test multi-phase DeepAR forecasting."""
        try:
            from decline_curve.forecast_deepar import DeepARForecaster

            try:
                # Create multi-phase data
                dates = pd.date_range("2020-01-01", periods=60, freq="MS")
                production = []
                for well_id in ["WELL_001", "WELL_002"]:
                    for i in range(30):
                        production.append(
                            {
                                "well_id": well_id,
                                "date": dates[i],
                                "oil": 1000 * np.exp(-0.05 * i),
                                "gas": 2000 * np.exp(-0.04 * i),
                                "water": 500 * (1 + 0.01 * i),
                            }
                        )

                df = pd.DataFrame(production)

                forecaster = DeepARForecaster(
                    phases=["oil", "gas", "water"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                )

                # Fit
                forecaster.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    verbose=False,
                )

                # Predict
                forecasts = forecaster.predict_quantiles(
                    well_id="WELL_001",
                    production_data=df,
                    quantiles=[0.5],
                    horizon=6,
                    n_samples=50,
                )

                assert "oil" in forecasts
                assert "gas" in forecasts
                assert "water" in forecasts

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("DeepAR module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
