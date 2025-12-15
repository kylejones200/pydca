"""Deep learning models for production forecasting.

This module implements advanced neural network architectures for production
forecasting, including LSTM encoder-decoder models with support for static
features and control variables.

Based on research showing that deep learning models can outperform traditional
DCA methods, especially for wells with short production histories or complex
decline patterns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import PyTorch, but make it optional
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

# Try to import sklearn for normalization
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = None
    StandardScaler = None


@dataclass
class StaticFeatures:
    """
    Container for static well features.

    Attributes:
        porosity: Formation porosity (fraction)
        permeability: Formation permeability (mD)
        thickness: Net pay thickness (ft)
        stages: Number of completion stages
        clusters: Number of clusters per stage
        proppant: Proppant mass (lbs)
        spacing: Well spacing (acres)
        artificial_lift_type: Type of artificial lift (optional)
        well_id: Well identifier
    """

    porosity: Optional[float] = None
    permeability: Optional[float] = None
    thickness: Optional[float] = None
    stages: Optional[int] = None
    clusters: Optional[int] = None
    proppant: Optional[float] = None
    spacing: Optional[float] = None
    artificial_lift_type: Optional[str] = None
    well_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "well_id"
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticFeatures":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ControlVariables:
    """
    Container for control variables (known-in-advance operational changes).

    Attributes:
        artificial_lift_install: Month when artificial lift is installed
        artificial_lift_type: Type of artificial lift to install
        workover_month: Month when workover occurs
        choke_change: Choke size changes over time
        other: Other control variables as dictionary
    """

    artificial_lift_install: Optional[int] = None
    artificial_lift_type: Optional[str] = None
    workover_month: Optional[int] = None
    choke_change: Optional[dict[int, float]] = None
    other: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.artificial_lift_install is not None:
            result["artificial_lift_install"] = self.artificial_lift_install
        if self.artificial_lift_type is not None:
            result["artificial_lift_type"] = self.artificial_lift_type
        if self.workover_month is not None:
            result["workover_month"] = self.workover_month
        if self.choke_change is not None:
            result["choke_change"] = self.choke_change
        if self.other is not None:
            result["other"] = self.other
            result.update(self.other)
        return result


if TORCH_AVAILABLE:

    class ProductionDataset(Dataset):
        """PyTorch dataset for production forecasting."""

        def __init__(
            self,
            sequences: np.ndarray,
            targets: np.ndarray,
            static_features: Optional[np.ndarray] = None,
            control_variables: Optional[np.ndarray] = None,
        ):
            """
            Initialize dataset.

            Args:
                sequences: Input sequences (n_samples, seq_len, n_features)
                targets: Target sequences (n_samples, horizon, n_outputs)
                static_features: Static features (n_samples, n_static)
                control_variables: Control variables (n_samples, horizon, n_control)
            """
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
            self.static_features = (
                torch.FloatTensor(static_features)
                if static_features is not None
                else None
            )
            self.control_variables = (
                torch.FloatTensor(control_variables)
                if control_variables is not None
                else None
            )

        def __len__(self) -> int:
            """Return dataset size."""
            return len(self.sequences)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            """Get item by index."""
            item = {
                "sequence": self.sequences[idx],
                "target": self.targets[idx],
            }
            if self.static_features is not None:
                item["static_features"] = self.static_features[idx]
            if self.control_variables is not None:
                item["control_variables"] = self.control_variables[idx]
            return item

    class EncoderDecoderLSTM(nn.Module):
        """
        LSTM Encoder-Decoder model for production forecasting.

        This architecture uses an encoder to process historical production data
        and a decoder to generate multi-step forecasts directly, avoiding error
        accumulation from recursive prediction.
        """

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 1,
            horizon: int = 12,
            static_feature_size: int = 0,
            control_variable_size: int = 0,
            dropout: float = 0.1,
            use_attention: bool = False,
        ):
            """
            Initialize encoder-decoder LSTM.

            Args:
                input_size: Number of input features (phases)
                hidden_size: Hidden layer size
                num_layers: Number of LSTM layers
                output_size: Number of output features (phases)
                horizon: Forecast horizon
                static_feature_size: Number of static features
                control_variable_size: Number of control variables
                dropout: Dropout rate
                use_attention: Whether to use attention mechanism
            """
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.horizon = horizon
            self.use_attention = use_attention

            # Encoder LSTM
            self.encoder = nn.LSTM(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )

            # Static feature embedding
            if static_feature_size > 0:
                self.static_embedding = nn.Linear(static_feature_size, hidden_size)

            # Decoder LSTM
            decoder_input_size = hidden_size
            if control_variable_size > 0:
                decoder_input_size += control_variable_size

            self.decoder = nn.LSTM(
                decoder_input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
            )

            # Attention mechanism (optional)
            if use_attention:
                self.attention = nn.MultiheadAttention(
                    hidden_size, num_heads=4, batch_first=True
                )

            # Output projection
            self.output_projection = nn.Linear(hidden_size, output_size)

        def forward(
            self,
            sequence: torch.Tensor,
            static_features: Optional[torch.Tensor] = None,
            control_variables: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                sequence: Input sequence (batch, seq_len, input_size)
                static_features: Static features (batch, static_feature_size)
                control_variables: Control variables (batch, horizon, control_size)

            Returns:
                Forecast (batch, horizon, output_size)
            """
            # batch_size = sequence.size(0)  # Unused

            # Encode input sequence
            encoder_output, (hidden, cell) = self.encoder(sequence)

            # Incorporate static features
            if static_features is not None:
                static_embedding = self.static_embedding(static_features)
                # Add to hidden state
                hidden = hidden + static_embedding.unsqueeze(0)

            # Prepare decoder input
            if self.use_attention:
                # Use attention over encoder outputs
                decoder_input, _ = self.attention(
                    encoder_output, encoder_output, encoder_output
                )
                decoder_input = decoder_input[:, -1:, :]  # Use last timestep
            else:
                # Use last encoder output
                decoder_input = encoder_output[:, -1:, :]

            # Repeat for horizon
            decoder_input = decoder_input.repeat(1, self.horizon, 1)

            # Add control variables if available
            if control_variables is not None:
                decoder_input = torch.cat([decoder_input, control_variables], dim=-1)

            # Decode
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

            # Project to output
            output = self.output_projection(decoder_output)

            return output


class EncoderDecoderLSTMForecaster:
    """
    High-level interface for LSTM encoder-decoder forecasting.

    This class provides a user-friendly API for training and using LSTM models
    for production forecasting with support for static features and control variables.
    """

    def __init__(
        self,
        static_features: Optional[list[str]] = None,
        control_variables: Optional[list[str]] = None,
        phases: list[str] = ["oil"],
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 12,
        sequence_length: int = 24,
        dropout: float = 0.1,
        use_attention: bool = False,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        normalization_method: Literal["minmax", "standard"] = "minmax",
    ):
        """
        Initialize LSTM forecaster.

        Args:
            static_features: List of static feature names
            control_variables: List of control variable names
            phases: List of phases to forecast (e.g., ['oil', 'gas', 'water'])
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            horizon: Forecast horizon (months)
            sequence_length: Input sequence length (months)
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            learning_rate: Learning rate for training
            device: Device to use ('cpu' or 'cuda')
            normalization_method: 'minmax' or 'standard' scaling
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning models. "
                "Install with: pip install torch"
            )

        self.static_features = static_features or []
        self.control_variables = control_variables or []
        self.phases = phases
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.normalization_method = normalization_method

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model (will be created during fit)
        self.model: Optional[EncoderDecoderLSTM] = None
        self.scaler: Optional[Any] = None  # Will store scaler for normalization
        self.is_fitted = False
        self.control_variable_encoders: Optional[dict[str, Any]] = (
            None  # For encoding control vars
        )

    def fit(
        self,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        control_variables: Optional[dict[str, dict]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the LSTM model.

        Args:
            production_data: DataFrame with columns: well_id, date, and phase columns
            static_features: DataFrame with well_id and static feature columns
            control_variables: Dict mapping well_id to control variable dicts
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history (loss, val_loss)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        # Prepare data
        sequences, targets, static_feat_array, control_array = self._prepare_data(
            production_data, static_features, control_variables
        )

        # Fit scaler and normalize sequences/targets
        self.scaler = self._create_scaler()
        sequences_normalized = self._normalize_sequences(sequences, fit=True)
        targets_normalized = self._normalize_sequences(targets, fit=False)

        # Create model
        input_size = len(self.phases)
        output_size = len(self.phases)
        static_feature_size = (
            len(self.static_features) if static_feat_array is not None else 0
        )
        control_variable_size = (
            len(self.control_variables) if control_array is not None else 0
        )

        self.model = EncoderDecoderLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            horizon=self.horizon,
            static_feature_size=static_feature_size,
            control_variable_size=control_variable_size,
            dropout=self.dropout,
            use_attention=self.use_attention,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Split data
        n_train = int(len(sequences_normalized) * (1 - validation_split))
        train_dataset = ProductionDataset(
            sequences_normalized[:n_train],
            targets_normalized[:n_train],
            static_feat_array[:n_train] if static_feat_array is not None else None,
            control_array[:n_train] if control_array is not None else None,
        )
        val_dataset = ProductionDataset(
            sequences_normalized[n_train:],
            targets_normalized[n_train:],
            static_feat_array[n_train:] if static_feat_array is not None else None,
            control_array[n_train:] if control_array is not None else None,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                output = self.model(
                    batch["sequence"].to(self.device),
                    batch.get("static_features", None),
                    batch.get("control_variables", None),
                )
                loss = criterion(output, batch["target"].to(self.device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    output = self.model(
                        batch["sequence"].to(self.device),
                        batch.get("static_features", None),
                        batch.get("control_variables", None),
                    )
                    loss = criterion(output, batch["target"].to(self.device))
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        self.is_fitted = True
        return history

    def predict(
        self,
        well_id: str,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        scenario: Optional[dict[str, Any]] = None,
        horizon: Optional[int] = None,
    ) -> dict[str, pd.Series]:
        """
        Generate forecast for a specific well.

        Args:
            well_id: Well identifier
            production_data: Historical production data
            static_features: Static features DataFrame
            scenario: Scenario dict with control variables
            horizon: Forecast horizon (overrides default)

        Returns:
            Dictionary with forecasts for each phase
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        horizon = horizon or self.horizon

        # Prepare input sequence
        well_data = production_data[production_data["well_id"] == well_id].copy()
        if len(well_data) < self.sequence_length:
            raise ValueError(
                f"Insufficient data for well {well_id}. "
                f"Need at least {self.sequence_length} months, got {len(well_data)}"
            )

        # Get last sequence_length months
        well_data = well_data.tail(self.sequence_length)
        sequence = well_data[self.phases].values

        # Normalize using stored scaler
        sequence_normalized = self._normalize_sequences(
            sequence.reshape(1, *sequence.shape), fit=False
        )[0]

        # Get static features
        static_feat = None
        if static_features is not None and self.static_features:
            well_static = static_features[static_features["well_id"] == well_id]
            if len(well_static) > 0:
                static_feat = well_static[self.static_features].values[0]

        # Get control variables
        control_vars = None
        if scenario and self.control_variables:
            # Convert scenario to control variable array
            control_vars = self._encode_control_variables(
                well_id, scenario, horizon, production_data
            )

        # Predict
        self.model.eval()
        with torch.no_grad():
            seq_tensor = (
                torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            )
            static_tensor = (
                torch.FloatTensor(static_feat).unsqueeze(0).to(self.device)
                if static_feat is not None
                else None
            )
            control_tensor = (
                torch.FloatTensor(control_vars).unsqueeze(0).to(self.device)
                if control_vars is not None
                else None
            )

            output = self.model(seq_tensor, static_tensor, control_tensor)
            forecast_normalized = output.cpu().numpy()[0]

            # Denormalize forecast
            forecast = self._denormalize_sequences(
                forecast_normalized.reshape(1, *forecast_normalized.shape)
            )[0]

        # Convert to Series
        last_date = pd.to_datetime(well_data["date"].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        forecasts = {}
        for i, phase in enumerate(self.phases):
            # Clip to non-negative (production can't be negative)
            phase_forecast = np.clip(forecast[:, i], 0, None)
            forecasts[phase] = pd.Series(
                phase_forecast, index=forecast_dates, name=phase
            )

        return forecasts

    def _prepare_data(
        self,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame],
        control_variables: Optional[dict[str, dict]],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for training.

        Creates sliding windows of sequences and targets from production data.

        Returns:
            Tuple of (sequences, targets, static_features, control_variables)
        """
        sequences = []
        targets = []
        static_feat_list = []
        control_list = []

        # Group by well_id
        for well_id, well_data in production_data.groupby("well_id"):
            # Sort by date
            well_data = well_data.sort_values("date").reset_index(drop=True)

            # Extract phase data
            phase_data = well_data[self.phases].values

            # Create sliding windows
            for i in range(len(well_data) - self.sequence_length - self.horizon + 1):
                # Input sequence
                seq = phase_data[i : i + self.sequence_length]
                sequences.append(seq)

                # Target sequence
                target = phase_data[
                    i + self.sequence_length : i + self.sequence_length + self.horizon
                ]
                targets.append(target)

                # Static features
                if static_features is not None and self.static_features:
                    well_static = static_features[static_features["well_id"] == well_id]
                    if len(well_static) > 0:
                        static_feat_list.append(
                            well_static[self.static_features].values[0]
                        )
                    else:
                        static_feat_list.append(np.zeros(len(self.static_features)))
                elif static_features is not None:
                    static_feat_list.append(None)

                # Control variables
                if control_variables is not None and well_id in control_variables:
                    # Encode control variables for this training window
                    control_vars = self._encode_control_variables_for_training(
                        well_id, control_variables[well_id], i, well_data
                    )
                    control_list.append(control_vars)
                elif control_variables is not None:
                    # No control variables for this well
                    control_list.append(
                        np.zeros((self.horizon, len(self.control_variables)))
                    )
                else:
                    control_list.append(None)

        sequences = np.array(sequences)
        targets = np.array(targets)

        static_feat_array = (
            np.array(static_feat_list)
            if static_feat_list and static_feat_list[0] is not None
            else None
        )
        control_array = (
            np.array(control_list)
            if control_list and control_list[0] is not None
            else None
        )

        return sequences, targets, static_feat_array, control_array

    def _create_scaler(self) -> Any:
        """Create and return a scaler for normalization."""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple normalization
            return None

        if self.normalization_method == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.normalization_method == "standard":
            return StandardScaler()
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}"
            )

    def _normalize_sequences(
        self, sequences: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Normalize sequences using the scaler.

        Args:
            sequences: Array of shape (n_samples, seq_len, n_features) or (seq_len, n_features)
            fit: Whether to fit the scaler

        Returns:
            Normalized sequences with same shape
        """
        if self.scaler is None:
            # Simple min-max normalization without sklearn
            if len(sequences.shape) == 3:
                # (n_samples, seq_len, n_features)
                min_vals = sequences.min(axis=(0, 1), keepdims=True)
                max_vals = sequences.max(axis=(0, 1), keepdims=True)
                range_vals = max_vals - min_vals
                range_vals = np.where(
                    range_vals == 0, 1, range_vals
                )  # Avoid division by zero
                normalized = (sequences - min_vals) / range_vals
                return normalized
            else:
                # (seq_len, n_features)
                min_vals = sequences.min(axis=0, keepdims=True)
                max_vals = sequences.max(axis=0, keepdims=True)
                range_vals = max_vals - min_vals
                range_vals = np.where(range_vals == 0, 1, range_vals)
                normalized = (sequences - min_vals) / range_vals
                return normalized

        # Use sklearn scaler
        original_shape = sequences.shape
        if len(sequences.shape) == 3:
            # Reshape to (n_samples * seq_len, n_features) for scaling
            n_samples, seq_len, n_features = sequences.shape
            sequences_2d = sequences.reshape(-1, n_features)
        else:
            sequences_2d = sequences

        if fit:
            sequences_normalized = self.scaler.fit_transform(sequences_2d)
        else:
            sequences_normalized = self.scaler.transform(sequences_2d)

        # Reshape back to original shape
        if len(original_shape) == 3:
            sequences_normalized = sequences_normalized.reshape(original_shape)

        return sequences_normalized

    def _denormalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Denormalize sequences using the stored scaler."""
        if self.scaler is None:
            # If we used simple normalization, we can't denormalize properly
            # This should not happen if scaler was used
            return sequences

        original_shape = sequences.shape
        if len(sequences.shape) == 3:
            n_samples, seq_len, n_features = sequences.shape
            sequences_2d = sequences.reshape(-1, n_features)
        else:
            sequences_2d = sequences

        sequences_denormalized = self.scaler.inverse_transform(sequences_2d)

        if len(original_shape) == 3:
            sequences_denormalized = sequences_denormalized.reshape(original_shape)

        return sequences_denormalized

    def _encode_control_variables(
        self,
        well_id: str,
        scenario: dict[str, Any],
        horizon: int,
        production_data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Encode control variables from scenario dict into array format.

        Args:
            well_id: Well identifier
            scenario: Dictionary with control variable values
            horizon: Forecast horizon
            production_data: Optional production data for context

        Returns:
            Array of shape (horizon, n_control_variables)
        """
        if not self.control_variables:
            return None

        control_array = np.zeros((horizon, len(self.control_variables)))

        # Create mapping of control variable names to indices
        var_to_idx = {var: i for i, var in enumerate(self.control_variables)}

        # Encode each control variable
        for var_name, var_idx in var_to_idx.items():
            if var_name in scenario:
                value = scenario[var_name]

                # Handle different types of control variables
                if var_name == "artificial_lift_type":
                    # One-hot encode artificial lift type
                    lift_types = ["none", "rod_pump", "ESP", "gas_lift", "plunger"]
                    if isinstance(value, str):
                        for t in lift_types:
                            if t.lower() in value.lower():
                                value_idx = lift_types.index(t)
                                control_array[:, var_idx] = value_idx / len(lift_types)
                                break
                elif var_name == "artificial_lift_install":
                    # Binary: 1 after install month, 0 before
                    install_month = (
                        int(value) if isinstance(value, (int, float)) else horizon
                    )
                    control_array[install_month:, var_idx] = 1.0
                elif var_name == "workover_month":
                    # Binary: 1 in workover month, 0 otherwise
                    workover_month = (
                        int(value) if isinstance(value, (int, float)) else -1
                    )
                    if 0 <= workover_month < horizon:
                        control_array[workover_month, var_idx] = 1.0
                elif isinstance(value, (int, float)):
                    # Numeric value - constant across horizon or varying
                    control_array[:, var_idx] = float(value)
                elif isinstance(value, dict):
                    # Time-varying values (e.g., choke_change)
                    for month, val in value.items():
                        if 0 <= int(month) < horizon:
                            control_array[int(month), var_idx] = float(val)

        return control_array

    def _encode_control_variables_for_training(
        self,
        well_id: str,
        control_dict: dict[str, Any],
        window_start: int,
        well_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Encode control variables for a training window.

        Args:
            well_id: Well identifier
            control_dict: Control variables dictionary
            window_start: Starting index of the training window
            well_data: Well production data

        Returns:
            Array of shape (horizon, n_control_variables)
        """
        if not self.control_variables:
            return np.zeros((self.horizon, 1))

        control_array = np.zeros((self.horizon, len(self.control_variables)))
        var_to_idx = {var: i for i, var in enumerate(self.control_variables)}

        # For training, we need to map control variables to the forecast horizon
        # relative to the window start
        for var_name, var_idx in var_to_idx.items():
            if var_name in control_dict:
                value = control_dict[var_name]

                if isinstance(value, (int, float)):
                    # If it's a month index, adjust relative to window_start
                    month_idx = int(value) - window_start
                    if 0 <= month_idx < self.horizon:
                        if var_name in ["artificial_lift_install", "workover_month"]:
                            # Binary flags
                            control_array[month_idx, var_idx] = 1.0
                        else:
                            # Continuous value
                            control_array[month_idx:, var_idx] = float(value)
                elif isinstance(value, dict):
                    # Time-varying values
                    for month, val in value.items():
                        month_idx = int(month) - window_start
                        if 0 <= month_idx < self.horizon:
                            control_array[month_idx, var_idx] = float(val)

        return control_array

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "static_features": self.static_features,
            "control_variables": self.control_variables,
            "phases": self.phases,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "horizon": self.horizon,
            "sequence_length": self.sequence_length,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
            "normalization_method": self.normalization_method,
            "control_variable_encoders": self.control_variable_encoders,
        }

        torch.save(state, filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "EncoderDecoderLSTMForecaster":
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            EncoderDecoderLSTMForecaster instance with loaded weights
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load models")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load with weights_only=False to support sklearn scalers
        # This is safe as we're loading our own saved models
        state = torch.load(filepath, map_location="cpu", weights_only=False)

        # Recreate forecaster
        forecaster = cls(
            static_features=state["static_features"],
            control_variables=state["control_variables"],
            phases=state["phases"],
            hidden_size=state["hidden_size"],
            num_layers=state["num_layers"],
            horizon=state["horizon"],
            sequence_length=state["sequence_length"],
            dropout=state["dropout"],
            use_attention=state["use_attention"],
            normalization_method=state.get("normalization_method", "minmax"),
        )

        # Recreate model architecture
        input_size = len(state["phases"])
        output_size = len(state["phases"])
        static_feature_size = (
            len(state["static_features"]) if state["static_features"] else 0
        )
        control_variable_size = (
            len(state["control_variables"]) if state["control_variables"] else 0
        )

        forecaster.model = EncoderDecoderLSTM(
            input_size=input_size,
            hidden_size=state["hidden_size"],
            num_layers=state["num_layers"],
            output_size=output_size,
            horizon=state["horizon"],
            static_feature_size=static_feature_size,
            control_variable_size=control_variable_size,
            dropout=state["dropout"],
            use_attention=state["use_attention"],
        )

        # Load weights
        forecaster.model.load_state_dict(state["model_state_dict"])
        forecaster.model.eval()

        # Restore scaler and other state
        forecaster.scaler = state["scaler"]
        forecaster.control_variable_encoders = state.get("control_variable_encoders")
        forecaster.is_fitted = True

        return forecaster

    def fine_tune(
        self,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        control_variables: Optional[dict[str, dict]] = None,
        epochs: int = 10,
        learning_rate: Optional[float] = None,
        batch_size: int = 32,
        freeze_encoder: bool = False,
    ) -> dict[str, list[float]]:
        """
        Fine-tune the model on new well data.

        Useful for transfer learning - adapting a model trained on multiple wells
        to a specific well or region.

        Args:
            production_data: DataFrame with well production data
            static_features: Static features DataFrame
            control_variables: Control variables dict
            epochs: Number of fine-tuning epochs (typically fewer than initial training)
            learning_rate: Learning rate for fine-tuning (usually smaller,
                defaults to 0.1 * original)
            batch_size: Batch size
            freeze_encoder: If True, only fine-tune decoder (faster, less flexible)

        Returns:
            Training history dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before fine-tuning")

        # Use smaller learning rate for fine-tuning
        if learning_rate is None:
            learning_rate = self.learning_rate * 0.1

        # Optionally freeze encoder layers
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Prepare data
        sequences, targets, static_feat_array, control_array = self._prepare_data(
            production_data, static_features, control_variables
        )

        # Normalize using existing scaler (don't refit)
        sequences_normalized = self._normalize_sequences(sequences, fit=False)
        targets_normalized = self._normalize_sequences(targets, fit=False)

        # Create dataset
        dataset = ProductionDataset(
            sequences_normalized,
            targets_normalized,
            static_feat_array,
            control_array,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer with fine-tuning learning rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )
        criterion = nn.MSELoss()

        # Fine-tuning loop
        history = {"loss": []}
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                output = self.model(
                    batch["sequence"].to(self.device),
                    batch.get("static_features", None),
                    batch.get("control_variables", None),
                )
                loss = criterion(output, batch["target"].to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history["loss"].append(avg_loss)

        # Unfreeze if needed
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        return history


if not TORCH_AVAILABLE:
    # Fallback when PyTorch is not available
    class EncoderDecoderLSTMForecaster:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            """Initialize placeholder when PyTorch is not available."""
            raise ImportError(
                "PyTorch is required for deep learning models. "
                "Install with: pip install torch"
            )
