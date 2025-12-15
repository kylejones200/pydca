"""Tests for deep learning module."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.deep_learning import (
    ControlVariables,
    StaticFeatures,
)


class TestStaticFeatures:
    """Test StaticFeatures class."""

    def test_create_static_features(self):
        """Test creating StaticFeatures."""
        features = StaticFeatures(
            porosity=0.15,
            permeability=1.5,
            thickness=50.0,
            stages=30,
            clusters=5,
            proppant=5000000.0,
            spacing=640.0,
            artificial_lift_type="ESP",
            well_id="WELL_001",
        )

        assert features.porosity == 0.15
        assert features.permeability == 1.5
        assert features.stages == 30
        assert features.well_id == "WELL_001"

    def test_to_dict(self):
        """Test converting to dictionary."""
        features = StaticFeatures(
            porosity=0.15, permeability=1.5, stages=30, well_id="WELL_001"
        )
        d = features.to_dict()

        assert "porosity" in d
        assert d["porosity"] == 0.15
        assert "well_id" not in d  # Should be excluded

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "porosity": 0.15,
            "permeability": 1.5,
            "stages": 30,
            "well_id": "WELL_001",
        }
        features = StaticFeatures.from_dict(data)

        assert features.porosity == 0.15
        assert features.permeability == 1.5
        assert features.stages == 30
        assert features.well_id == "WELL_001"


class TestControlVariables:
    """Test ControlVariables class."""

    def test_create_control_variables(self):
        """Test creating ControlVariables."""
        control = ControlVariables(
            artificial_lift_install=12,
            artificial_lift_type="ESP",
            workover_month=24,
            choke_change={6: 0.5, 12: 0.75},
        )

        assert control.artificial_lift_install == 12
        assert control.artificial_lift_type == "ESP"
        assert control.workover_month == 24

    def test_to_dict(self):
        """Test converting to dictionary."""
        control = ControlVariables(
            artificial_lift_install=12,
            artificial_lift_type="ESP",
            other={"custom_var": 123},
        )
        d = control.to_dict()

        assert "artificial_lift_install" in d
        assert d["artificial_lift_type"] == "ESP"
        assert "custom_var" in d["other"]


class TestEncoderDecoderLSTM:
    """Test LSTM encoder-decoder (if PyTorch available)."""

    def test_import_encoder_decoder(self):
        """Test that EncoderDecoderLSTMForecaster can be imported."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            # Should be able to create instance (will fail if PyTorch not available)
            try:
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"], horizon=12, sequence_length=24
                )
                # If we get here, PyTorch is available
                assert forecaster.phases == ["oil"]
                assert forecaster.horizon == 12
            except ImportError:
                # PyTorch not available - that's okay
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def test_normalization_methods(self):
        """Test that different normalization methods can be set."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                # MinMax normalization
                forecaster_minmax = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=12,
                    sequence_length=24,
                    normalization_method="minmax",
                )
                assert forecaster_minmax.normalization_method == "minmax"

                # Standard normalization
                forecaster_std = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=12,
                    sequence_length=24,
                    normalization_method="standard",
                )
                assert forecaster_std.normalization_method == "standard"
            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def test_control_variable_encoding(self):
        """Test control variable encoding."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=12,
                    sequence_length=24,
                    control_variables=["artificial_lift_install", "workover_month"],
                )

                # Test encoding artificial lift install
                scenario = {"artificial_lift_install": 6}
                control_array = forecaster._encode_control_variables(
                    "WELL_001", scenario, horizon=12
                )

                assert control_array is not None
                assert control_array.shape == (12, 2)
                # Should be 1 after month 6
                assert control_array[6:, 0].sum() > 0
                assert control_array[:6, 0].sum() == 0

                # Test encoding workover
                scenario = {"workover_month": 8}
                control_array = forecaster._encode_control_variables(
                    "WELL_001", scenario, horizon=12
                )
                assert control_array[8, 1] == 1.0
                assert control_array[7, 1] == 0.0
                assert control_array[9, 1] == 0.0

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def _create_sample_data(self):
        """Create sample production data for testing."""
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        production = []
        for well_id in ["WELL_001", "WELL_002"]:
            for i in range(30):
                # Exponential decline
                prod = 1000 * np.exp(-0.05 * i)
                production.append(
                    {
                        "well_id": well_id,
                        "date": dates[i],
                        "oil": prod,
                    }
                )

        df = pd.DataFrame(production)
        return df

    def test_fit_and_predict(self):
        """Test fitting and prediction with sample data."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                )

                # Create sample data
                df = self._create_sample_data()

                # Fit model (with minimal epochs for speed)
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

                # Predict
                forecast = forecaster.predict(
                    well_id="WELL_001",
                    production_data=df,
                    horizon=6,
                )

                assert "oil" in forecast
                assert len(forecast["oil"]) == 6
                assert all(forecast["oil"] >= 0)  # Production should be non-negative

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading models."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                # Create and fit a model
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                )

                df = self._create_sample_data()
                forecaster.fit(
                    production_data=df,
                    epochs=2,
                    batch_size=4,
                    validation_split=0.2,
                    verbose=False,
                )

                # Save model
                model_path = tmp_path / "test_model.pt"
                forecaster.save_model(model_path)
                assert model_path.exists()

                # Load model
                loaded_forecaster = EncoderDecoderLSTMForecaster.load_model(model_path)

                assert loaded_forecaster.is_fitted
                assert loaded_forecaster.phases == forecaster.phases
                assert loaded_forecaster.horizon == forecaster.horizon
                assert loaded_forecaster.hidden_size == forecaster.hidden_size

                # Test prediction with loaded model
                forecast = loaded_forecaster.predict(
                    well_id="WELL_001",
                    production_data=df,
                    horizon=6,
                )
                assert "oil" in forecast
                assert len(forecast["oil"]) == 6

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def test_fine_tune(self):
        """Test fine-tuning functionality."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                # Create and fit initial model
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    hidden_size=32,
                    num_layers=1,
                )

                # Initial training data
                df1 = self._create_sample_data()

                forecaster.fit(
                    production_data=df1,
                    epochs=2,
                    batch_size=4,
                    validation_split=0.2,
                    verbose=False,
                )

                # Fine-tune on new data
                df2 = self._create_sample_data()  # Can use same or different data
                history = forecaster.fine_tune(
                    production_data=df2,
                    epochs=2,
                    batch_size=4,
                    freeze_encoder=False,
                )

                assert "loss" in history
                assert len(history["loss"]) == 2

                # Test prediction after fine-tuning
                forecast = forecaster.predict(
                    well_id="WELL_001",
                    production_data=df2,
                    horizon=6,
                )
                assert "oil" in forecast

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")

    def test_static_features(self):
        """Test static feature integration."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            try:
                forecaster = EncoderDecoderLSTMForecaster(
                    phases=["oil"],
                    horizon=6,
                    sequence_length=12,
                    static_features=["porosity", "stages"],
                    hidden_size=32,
                    num_layers=1,
                )

                # Create production data
                df = self._create_sample_data()

                # Create static features
                static_df = pd.DataFrame(
                    {
                        "well_id": ["WELL_001", "WELL_002"],
                        "porosity": [0.15, 0.18],
                        "stages": [30, 35],
                    }
                )

                # Fit with static features
                history = forecaster.fit(
                    production_data=df,
                    static_features=static_df,
                    epochs=2,
                    batch_size=4,
                    validation_split=0.2,
                    verbose=False,
                )

                assert forecaster.is_fitted

                # Predict with static features
                forecast = forecaster.predict(
                    well_id="WELL_001",
                    production_data=df,
                    static_features=static_df,
                    horizon=6,
                )
                assert "oil" in forecast

            except ImportError:
                pytest.skip("PyTorch not available")
        except ImportError:
            pytest.skip("Deep learning module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
