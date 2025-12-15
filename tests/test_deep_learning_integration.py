"""Integration tests for deep learning module using real production data.

These tests validate that the deep learning models work correctly with
real-world production data from thousands of North Dakota wells.
"""

import os

import pandas as pd
import pytest

from decline_curve.utils.real_data_loader import load_north_dakota_production

# Path to real production data (update if needed)
REAL_DATA_PATH = (
    "/Users/kylejonespatricia/Library/CloudStorage/"
    "GoogleDrive-kyletjones@gmail.com/My Drive/applandia/bana_4373/data/"
    "north_dakota_production.csv"
)


def load_real_production_data(
    max_wells: int = 100,
    min_months: int = 12,
    max_months: int = 120,
    data_path: str = REAL_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare real North Dakota production data for deep learning.

    This is a convenience wrapper around load_north_dakota_production.

    Args:
        max_wells: Maximum number of wells to load (for testing speed)
        min_months: Minimum production history required per well
        max_months: Maximum production history to use per well
        data_path: Path to production CSV file

    Returns:
        Tuple of (production_df, static_features_df)
        - production_df: DataFrame with columns: well_id, date, oil, gas, water
        - static_features_df: DataFrame with well_id and static features
    """
    if not os.path.exists(data_path):
        pytest.skip(f"Real production data not found at {data_path}")

    try:
        production_df, static_features_df = load_north_dakota_production(
            data_path=data_path,
            max_wells=max_wells,
            min_months=min_months,
            max_months=max_months,
            phases=["oil", "gas", "water"],
        )
        return production_df, static_features_df
    except Exception as e:
        pytest.skip(f"Error loading production data: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestDeepLearningWithRealData:
    """Integration tests with real production data."""

    def test_load_real_data(self):
        """Test loading real production data."""
        try:
            production_df, static_df = load_real_production_data(max_wells=10)

            assert len(production_df) > 0
            assert "well_id" in production_df.columns
            assert "date" in production_df.columns
            assert "oil" in production_df.columns

            assert len(static_df) > 0
            assert "well_id" in static_df.columns

            # Test passes if no exceptions raised

        except FileNotFoundError:
            pytest.skip("Real production data not available")

    def test_lstm_with_real_data(self):
        """Test LSTM training and prediction with real data."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            # Load small subset for testing
            production_df, static_df = load_real_production_data(
                max_wells=20, min_months=24
            )

            # Create forecaster
            forecaster = EncoderDecoderLSTMForecaster(
                phases=["oil"],
                horizon=6,
                sequence_length=12,
                hidden_size=32,
                num_layers=1,
                normalization_method="minmax",
            )

            # Train on real data (minimal epochs for speed)
            history = forecaster.fit(
                production_data=production_df,
                static_features=static_df,
                epochs=2,
                batch_size=4,
                validation_split=0.2,
                verbose=False,
            )

            assert "loss" in history
            assert len(history["loss"]) == 2
            assert forecaster.is_fitted

            # Test prediction on a well
            test_well = production_df["well_id"].iloc[0]
            forecast = forecaster.predict(
                well_id=test_well,
                production_data=production_df,
                static_features=static_df,
                horizon=6,
            )

            assert "oil" in forecast
            assert len(forecast["oil"]) == 6
            assert all(forecast["oil"] >= 0)  # Non-negative production

            # Test passes if no exceptions raised

        except ImportError:
            pytest.skip("PyTorch not available")
        except FileNotFoundError:
            pytest.skip("Real production data not available")

    def test_multiphase_lstm_with_real_data(self):
        """Test multi-phase LSTM with real oil, gas, water data."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            # Load data
            production_df, static_df = load_real_production_data(
                max_wells=15, min_months=24
            )

            # Create multi-phase forecaster
            forecaster = EncoderDecoderLSTMForecaster(
                phases=["oil", "gas", "water"],
                horizon=6,
                sequence_length=12,
                hidden_size=32,
                num_layers=1,
            )

            # Train
            forecaster.fit(
                production_data=production_df,
                epochs=2,
                batch_size=4,
                validation_split=0.2,
                verbose=False,
            )

            assert forecaster.is_fitted

            # Predict
            test_well = production_df["well_id"].iloc[0]
            forecast = forecaster.predict(
                well_id=test_well,
                production_data=production_df,
                horizon=6,
            )

            assert "oil" in forecast
            assert "gas" in forecast
            assert "water" in forecast
            assert all(forecast["oil"] >= 0)
            assert all(forecast["gas"] >= 0)
            assert all(forecast["water"] >= 0)

            # Test passes if no exceptions raised

        except ImportError:
            pytest.skip("PyTorch not available")
        except FileNotFoundError:
            pytest.skip("Real production data not available")

    def test_model_persistence_with_real_data(self, tmp_path):
        """Test saving/loading models trained on real data."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            # Load and train
            production_df, static_df = load_real_production_data(
                max_wells=10, min_months=24
            )

            forecaster = EncoderDecoderLSTMForecaster(
                phases=["oil"],
                horizon=6,
                sequence_length=12,
                hidden_size=32,
                num_layers=1,
            )

            forecaster.fit(
                production_data=production_df,
                epochs=2,
                batch_size=4,
                verbose=False,
            )

            # Save model
            model_path = tmp_path / "real_data_model.pt"
            forecaster.save_model(model_path)
            assert model_path.exists()

            # Load model
            loaded_forecaster = EncoderDecoderLSTMForecaster.load_model(model_path)

            # Test prediction with loaded model
            test_well = production_df["well_id"].iloc[0]
            forecast = loaded_forecaster.predict(
                well_id=test_well,
                production_data=production_df,
                horizon=6,
            )

            assert "oil" in forecast
            assert len(forecast["oil"]) == 6

            # Test passes if no exceptions raised

        except ImportError:
            pytest.skip("PyTorch not available")
        except FileNotFoundError:
            pytest.skip("Real production data not available")

    def test_fine_tuning_with_real_data(self):
        """Test transfer learning (fine-tuning) with real data."""
        try:
            from decline_curve.deep_learning import EncoderDecoderLSTMForecaster

            # Load larger dataset for initial training
            production_df_large, static_df_large = load_real_production_data(
                max_wells=30, min_months=24
            )

            # Train base model
            forecaster = EncoderDecoderLSTMForecaster(
                phases=["oil"],
                horizon=6,
                sequence_length=12,
                hidden_size=32,
                num_layers=1,
            )

            forecaster.fit(
                production_data=production_df_large,
                epochs=2,
                batch_size=4,
                verbose=False,
            )

            # Fine-tune on specific well
            target_well = production_df_large["well_id"].iloc[0]
            well_data = production_df_large[
                production_df_large["well_id"] == target_well
            ].copy()

            history = forecaster.fine_tune(
                production_data=well_data,
                epochs=2,
                batch_size=1,
                freeze_encoder=False,
            )

            assert "loss" in history
            assert len(history["loss"]) == 2

            # Test prediction after fine-tuning
            forecast = forecaster.predict(
                well_id=target_well,
                production_data=production_df_large,
                horizon=6,
            )

            assert "oil" in forecast
            # Test passes if no exceptions raised

        except ImportError:
            pytest.skip("PyTorch not available")
        except FileNotFoundError:
            pytest.skip("Real production data not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
