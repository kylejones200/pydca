"""Temporal Fusion Transformer (TFT) for production forecasting.

Temporal Fusion Transformer is an advanced transformer-based architecture
designed for time series forecasting with interpretability through attention
mechanisms.

Based on research showing TFT's ability to:
- Handle complex temporal patterns
- Provide interpretability via attention weights
- Incorporate static features and control variables
- Achieve state-of-the-art forecasting performance
"""

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
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


if TORCH_AVAILABLE:

    class TemporalFusionTransformer(nn.Module):
        """
        Temporal Fusion Transformer for production forecasting.

        Implements a simplified TFT architecture with:
        - Multi-head self-attention
        - Static feature encoders
        - Temporal feature processing
        - Interpretability through attention weights
        """

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            static_feature_size: int = 0,
            control_variable_size: int = 0,
            horizon: int = 12,
            dropout: float = 0.1,
        ):
            """
            Initialize TFT model.

            Args:
                input_size: Number of input features (phases)
                hidden_size: Hidden layer size
                num_heads: Number of attention heads
                num_layers: Number of transformer layers
                static_feature_size: Number of static features
                control_variable_size: Number of control variables
                horizon: Forecast horizon
                dropout: Dropout rate
            """
            super().__init__()

            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.horizon = horizon

            # Input projection
            self.input_projection = nn.Linear(input_size, hidden_size)

            # Static feature encoder
            if static_feature_size > 0:
                self.static_encoder = nn.Sequential(
                    nn.Linear(static_feature_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )

            # Temporal feature processing (LSTM)
            self.temporal_encoder = nn.LSTM(
                hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )

            # Multi-head self-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            # Control variable integration
            if control_variable_size > 0:
                self.control_encoder = nn.Linear(control_variable_size, hidden_size)

            # Variable Selection Networks (VSN) - Enhanced version
            self.vsn_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Softmax(dim=-1),
            )

            # Gating mechanism (GRN - Gated Residual Network)
            self.grn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            self.gate_norm = nn.LayerNorm(hidden_size)

            # Temporal decoder
            self.decoder = nn.LSTM(
                hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )

            # Output projection (multi-step)
            self.output_layers = nn.ModuleList(
                [nn.Linear(hidden_size, input_size) for _ in range(horizon)]
            )

            self.horizon = horizon
            self.output_size = input_size

        def forward(
            self,
            sequence: torch.Tensor,
            static_features: Optional[torch.Tensor] = None,
            control_variables: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """
            Forward pass.

            Args:
                sequence: Input sequence (batch, seq_len, input_size)
                static_features: Static features (batch, static_feature_size)
                control_variables: Control variables (batch, seq_len, control_size)

            Returns:
                Tuple of (forecast, attention_dict) where attention_dict contains
                attention weights for interpretability
            """
            batch_size, seq_len, _ = sequence.shape

            # Project input
            x = self.input_projection(sequence)  # (batch, seq_len, hidden_size)

            # Encode static features
            if static_features is not None:
                static_emb = self.static_encoder(
                    static_features
                )  # (batch, hidden_size)
                # Add to all timesteps
                static_emb = static_emb.unsqueeze(1).expand(-1, seq_len, -1)
                x = x + static_emb

            # Temporal encoding
            temporal_out, (hidden, cell) = self.temporal_encoder(x)

            # Self-attention
            attention_output = self.transformer(temporal_out)

            # Extract attention weights (from last layer)
            attention_weights = None
            if self.transformer.layers[-1].self_attn is not None:
                # Store attention for interpretation
                attn_output, attn_weights = self.transformer.layers[-1].self_attn(
                    attention_output,
                    attention_output,
                    attention_output,
                    average_attn_weights=False,
                )
                attention_weights = attn_weights.mean(dim=1)  # Average across heads

            # Variable Selection Network (VSN)
            vsn_weights = self.vsn_gate(
                attention_output
            )  # (batch, seq_len, hidden_size)
            vsn_output = attention_output * vsn_weights

            # Gated Residual Network (GRN)
            grn_out = self.grn(vsn_output)
            gate_values = torch.sigmoid(
                grn_out[:, :, : self.hidden_size]
            )  # Gating signal
            gated_output = vsn_output * gate_values + vsn_output  # Residual connection
            gated_output = self.gate_norm(gated_output)

            # Control variable integration
            if control_variables is not None:
                control_emb = self.control_encoder(control_variables)
                gated_output = gated_output + control_emb

            # Encoder output (last timestep)
            encoder_output = gated_output[:, -1:, :]  # (batch, 1, hidden_size)

            # Decoder: autoregressive decoding for multi-step forecast
            decoder_input = encoder_output
            decoder_outputs = []

            for step in range(self.horizon):
                decoder_out, (hidden, cell) = self.decoder(decoder_input)
                decoder_outputs.append(decoder_out)
                # Use output as next input (autoregressive)
                decoder_input = decoder_out

            # Stack decoder outputs: (batch, horizon, hidden_size)
            decoder_stacked = torch.cat(decoder_outputs, dim=1)

            # Multi-step output projection
            outputs = []
            for step in range(self.horizon):
                step_output = self.output_layers[step](decoder_stacked[:, step, :])
                outputs.append(step_output.unsqueeze(1))

            output = torch.cat(outputs, dim=1)  # (batch, horizon, output_size)

            # Store interpretation data
            interpretation = {
                "attention_weights": attention_weights,
                "vsn_weights": vsn_weights,
                "gate_values": gate_values,
                "static_embedding": static_emb if static_features is not None else None,
                "decoder_outputs": decoder_stacked,
            }

            return output, interpretation


class TFTForecaster:
    """
    Temporal Fusion Transformer forecaster for production data.

    Provides interpretable forecasting with attention mechanisms.
    """

    def __init__(
        self,
        phases: list[str] = ["oil"],
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        sequence_length: int = 24,
        horizon: int = 12,
        static_features: Optional[list[str]] = None,
        control_variables: Optional[list[str]] = None,
        dropout: float = 0.1,
        normalization_method: Literal["minmax", "standard"] = "minmax",
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Initialize TFT forecaster.

        Args:
            phases: List of phases to forecast
            hidden_size: Hidden layer size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            sequence_length: Input sequence length (months)
            horizon: Forecast horizon (months)
            static_features: List of static feature names
            control_variables: List of control variable names
            dropout: Dropout rate
            normalization_method: Normalization method
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TFT. Install with: pip install torch"
            )

        self.phases = phases
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.static_features = static_features or []
        self.control_variables = control_variables or []
        self.dropout = dropout
        self.normalization_method = normalization_method
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model (will be created during fit)
        self.model: Optional[TemporalFusionTransformer] = None
        self.scaler: Optional[Any] = None
        self.is_fitted = False

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
        Train the TFT model.

        Args:
            production_data: DataFrame with columns: well_id, date, and phase columns
            static_features: Optional DataFrame with well_id and static features
            control_variables: Optional control variables dict
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        # Prepare data (similar to LSTM)
        from .deep_learning import EncoderDecoderLSTMForecaster

        # Use LSTM's data preparation method
        lstm_helper = EncoderDecoderLSTMForecaster(
            phases=self.phases,
            horizon=self.horizon,
            sequence_length=self.sequence_length,
            static_features=self.static_features,
            control_variables=self.control_variables,
            normalization_method=self.normalization_method,
        )

        sequences, targets, static_feat_array, control_array = (
            lstm_helper._prepare_data(
                production_data, static_features, control_variables
            )
        )

        # Fit scaler and normalize
        self.scaler = lstm_helper._create_scaler()
        # Fit scaler on sequences
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        self.scaler.fit(sequences_reshaped)
        lstm_helper.scaler = self.scaler

        sequences_normalized = lstm_helper._normalize_sequences(sequences, fit=False)
        targets_normalized = lstm_helper._normalize_sequences(targets, fit=False)

        # Create model
        input_size = len(self.phases)
        static_feature_size = (
            len(self.static_features) if static_feat_array is not None else 0
        )
        control_variable_size = (
            len(self.control_variables) if control_array is not None else 0
        )

        self.model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            static_feature_size=static_feature_size,
            control_variable_size=control_variable_size,
            horizon=self.horizon,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create dataset
        class TFTDataset(Dataset):
            def __init__(self, seq, tgt, static, control):
                self.seq = torch.FloatTensor(seq)
                self.tgt = torch.FloatTensor(tgt)
                self.static = torch.FloatTensor(static) if static is not None else None
                self.control = (
                    torch.FloatTensor(control) if control is not None else None
                )

            def __len__(self):
                return len(self.seq)

            def __getitem__(self, idx):
                item = {"sequence": self.seq[idx], "target": self.tgt[idx]}
                if self.static is not None:
                    item["static_features"] = self.static[idx]
                if self.control is not None:
                    item["control_variables"] = self.control[idx]
                return item

        # Split data
        n_train = int(len(sequences_normalized) * (1 - validation_split))
        train_dataset = TFTDataset(
            sequences_normalized[:n_train],
            targets_normalized[:n_train],
            static_feat_array[:n_train] if static_feat_array is not None else None,
            control_array[:n_train] if control_array is not None else None,
        )
        val_dataset = TFTDataset(
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
                output, _ = self.model(
                    batch["sequence"].to(self.device),
                    batch.get("static_features"),
                    batch.get("control_variables"),
                )
                target = batch["target"].to(self.device)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    output, _ = self.model(
                        batch["sequence"].to(self.device),
                        batch.get("static_features"),
                        batch.get("control_variables"),
                    )
                    target = batch["target"].to(self.device)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        self.is_fitted = True
        return history

    def predict(
        self,
        well_id: str,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        control_variables: Optional[dict[str, Any]] = None,
        horizon: Optional[int] = None,
        return_interpretation: bool = False,
    ) -> Union[dict[str, pd.Series], tuple[dict[str, pd.Series], dict[str, Any]]]:
        """
        Generate forecast with optional interpretability.

        Args:
            well_id: Well identifier
            production_data: Historical production data
            static_features: Optional static features
            control_variables: Optional control variables
            horizon: Forecast horizon (overrides default)
            return_interpretation: If True, return attention weights and
                feature importance

        Returns:
            Dictionary with forecasts for each phase, optionally with interpretation dict
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        horizon = horizon or self.horizon

        # Prepare input (similar to LSTM)
        from .deep_learning import EncoderDecoderLSTMForecaster

        lstm_helper = EncoderDecoderLSTMForecaster(
            phases=self.phases,
            horizon=horizon,
            sequence_length=self.sequence_length,
            static_features=self.static_features,
            control_variables=self.control_variables,
            normalization_method=self.normalization_method,
        )
        lstm_helper.scaler = self.scaler

        # Get well data
        well_data = production_data[production_data["well_id"] == well_id].copy()
        if len(well_data) < self.sequence_length:
            raise ValueError(
                f"Insufficient data for well {well_id}. "
                f"Need at least {self.sequence_length} months, got {len(well_data)}"
            )

        well_data = well_data.tail(self.sequence_length)
        sequence = well_data[self.phases].values

        # Normalize
        sequence_normalized = lstm_helper._normalize_sequences(
            sequence.reshape(1, *sequence.shape), fit=False
        )[0]

        # Get static features
        static_feat = None
        if static_features is not None and self.static_features:
            well_static = static_features[static_features["well_id"] == well_id]
            if len(well_static) > 0:
                numeric_cols = [
                    c
                    for c in well_static.columns
                    if c in self.static_features
                    and well_static[c].dtype in [np.number, "int64", "float64"]
                ]
                if numeric_cols:
                    static_feat = well_static[numeric_cols].values[0]

        # Get control variables
        control_vars = None
        if control_variables is not None and self.control_variables:
            # Similar to LSTM encoding
            control_vars = np.zeros((self.sequence_length, len(self.control_variables)))
            # Simplified - full implementation would encode properly

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

            output, interpretation = self.model(
                seq_tensor, static_tensor, control_tensor
            )
            forecast_normalized = output.cpu().numpy()[0]

        # Denormalize
        forecast = lstm_helper._denormalize_sequences(
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
            phase_forecast = np.clip(forecast[:, i], 0, None)
            forecasts[phase] = pd.Series(
                phase_forecast, index=forecast_dates, name=phase
            )

        if return_interpretation:
            return forecasts, interpretation
        return forecasts

    def plot_attention_weights(
        self, interpretation: dict[str, Any], title: str = "Attention Weights"
    ):
        """
        Visualize attention weights for interpretability.

        Args:
            interpretation: Interpretation dict from predict()
            title: Plot title
        """
        import matplotlib.pyplot as plt

        if interpretation.get("attention_weights") is None:
            logger.warning("No attention weights available")
            return

        attn_weights = interpretation["attention_weights"].cpu().numpy()

        if len(attn_weights.shape) == 3:
            # (batch, seq_len, seq_len) - take first batch
            attn_weights = attn_weights[0]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_weights, cmap="viridis", aspect="auto")
        ax.set_xlabel("Query Position")
        ax.set_ylabel("Key Position")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(
        self, interpretation: dict[str, Any], title: str = "Feature Importance"
    ):
        """
        Visualize feature importance from gating mechanism and VSN.

        Args:
            interpretation: Interpretation dict from predict()
            title: Plot title
        """
        import matplotlib.pyplot as plt

        # Get VSN weights (Variable Selection Network weights)
        vsn_weights = interpretation.get("vsn_weights")
        gate_values = interpretation.get("gate_values")

        fig, axes = plt.subplots(
            1, 2 if vsn_weights is not None else 1, figsize=(14, 6)
        )
        if vsn_weights is None:
            axes = [axes]

        # Plot VSN weights (variable selection)
        if vsn_weights is not None:
            vsn = vsn_weights.cpu().numpy()
            if len(vsn.shape) == 3:
                vsn = vsn[0, -1, :]  # Last timestep, first batch

            axes[0].bar(range(len(vsn)), vsn)
            axes[0].set_xlabel("Feature Index")
            axes[0].set_ylabel("VSN Weight (Selection)")
            axes[0].set_title("Variable Selection Network Weights")
            axes[0].grid(True, alpha=0.3)

        # Plot gate values (gating mechanism)
        if gate_values is not None:
            gv = gate_values.cpu().numpy()
            if len(gv.shape) == 3:
                gv = gv[0, -1, : self.hidden_size]  # Last timestep, first batch

            ax_idx = 1 if vsn_weights is not None else 0
            axes[ax_idx].bar(range(len(gv)), gv)
            axes[ax_idx].set_xlabel("Feature Index")
            axes[ax_idx].set_ylabel("Gate Value (Importance)")
            axes[ax_idx].set_title("Gating Mechanism")
            axes[ax_idx].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


if not TORCH_AVAILABLE:
    # Fallback when PyTorch is not available
    class TFTForecaster:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            """Initialize placeholder when PyTorch is not available."""
            raise ImportError(
                "PyTorch is required for TFT. Install with: pip install torch"
            )
