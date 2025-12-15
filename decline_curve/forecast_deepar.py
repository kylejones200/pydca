"""DeepAR probabilistic forecasting for production data.

DeepAR is a probabilistic forecasting model that provides uncertainty
quantification through quantile predictions (P10, P50, P90).

Based on the DeepAR architecture from Amazon Research, adapted for
production forecasting with support for:
- Multi-well training
- Static features
- Probabilistic outputs
- Quantile predictions
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

    class DeepARModel(nn.Module):
        """
        DeepAR probabilistic forecasting model.

        Uses LSTM architecture with probabilistic output layer that predicts
        parameters of a distribution (e.g., mean and scale for negative binomial).
        """

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            distribution: Literal["normal", "negative_binomial"] = "normal",
        ):
            """
            Initialize DeepAR model.

            Args:
                input_size: Number of input features
                hidden_size: LSTM hidden layer size
                num_layers: Number of LSTM layers
                dropout: Dropout rate
                distribution: Output distribution type
            """
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.distribution = distribution

            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )

            # Output layers for distribution parameters
            if distribution == "normal":
                # Predict mean and scale (standard deviation)
                self.mean_layer = nn.Linear(hidden_size, input_size)
                self.scale_layer = nn.Linear(hidden_size, input_size)
            elif distribution == "negative_binomial":
                # Predict mean and dispersion
                self.mean_layer = nn.Linear(hidden_size, input_size)
                self.dispersion_layer = nn.Linear(hidden_size, input_size)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Input sequence (batch, seq_len, input_size)

            Returns:
                Tuple of (mean, scale) or (mean, dispersion) depending on distribution
            """
            # Encode sequence
            lstm_out, _ = self.lstm(x)

            # Use last timestep
            last_hidden = lstm_out[:, -1, :]

            # Predict distribution parameters
            mean = self.mean_layer(last_hidden)
            if self.distribution == "normal":
                scale = F.softplus(self.scale_layer(last_hidden)) + 1e-5
                return mean, scale
            else:  # negative_binomial
                dispersion = F.softplus(self.dispersion_layer(last_hidden)) + 1e-5
                return mean, dispersion

        def sample(self, mean: torch.Tensor, scale: torch.Tensor, n_samples: int = 100):
            """
            Sample from the predicted distribution.

            Args:
                mean: Predicted mean
                scale: Predicted scale (or dispersion)
                n_samples: Number of samples to generate

            Returns:
                Samples from the distribution
            """
            if self.distribution == "normal":
                # Sample from normal distribution
                samples = torch.normal(
                    mean.unsqueeze(0).expand(n_samples, -1, -1),
                    scale.unsqueeze(0).expand(n_samples, -1, -1),
                )
                return samples
            else:  # negative_binomial
                # For negative binomial, we'd need to implement sampling
                # For now, use normal approximation
                samples = torch.normal(
                    mean.unsqueeze(0).expand(n_samples, -1, -1),
                    scale.unsqueeze(0).expand(n_samples, -1, -1),
                )
                return samples

    class DeepARDataset(Dataset):
        """Dataset for DeepAR training."""

        def __init__(
            self,
            sequences: np.ndarray,
            targets: np.ndarray,
            static_features: Optional[np.ndarray] = None,
        ):
            """
            Initialize dataset.

            Args:
                sequences: Input sequences (n_samples, seq_len, n_features)
                targets: Target values (n_samples, n_features)
                static_features: Static features (n_samples, n_static)
            """
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
            self.static_features = (
                torch.FloatTensor(static_features)
                if static_features is not None
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
            return item


class DeepARForecaster:
    """
    DeepAR probabilistic forecaster for production data.

    Provides probabilistic forecasts with uncertainty quantification
    through quantile predictions (P10, P50, P90).
    """

    def __init__(
        self,
        phases: list[str] = ["oil"],
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 24,
        horizon: int = 12,
        dropout: float = 0.1,
        distribution: Literal["normal", "negative_binomial"] = "normal",
        normalization_method: Literal["minmax", "standard"] = "minmax",
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Initialize DeepAR forecaster.

        Args:
            phases: List of phases to forecast
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            sequence_length: Input sequence length (months)
            horizon: Forecast horizon (months)
            dropout: Dropout rate
            distribution: Output distribution type
            normalization_method: Normalization method
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DeepAR. Install with: pip install torch"
            )

        self.phases = phases
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.dropout = dropout
        self.distribution = distribution
        self.normalization_method = normalization_method
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model (will be created during fit)
        self.model: Optional[DeepARModel] = None
        self.scaler: Optional[Any] = None
        self.is_fitted = False

    def fit(
        self,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the DeepAR model.

        Args:
            production_data: DataFrame with columns: well_id, date, and phase columns
            static_features: Optional DataFrame with well_id and static features
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        # Prepare data
        sequences, targets, static_feat_array = self._prepare_data(
            production_data, static_features
        )

        # Fit scaler and normalize
        self.scaler = self._create_scaler()
        sequences_normalized = self._normalize_sequences(sequences, fit=True)
        targets_normalized = self._normalize_sequences(targets, fit=False)

        # Create model
        input_size = len(self.phases)
        self.model = DeepARModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            distribution=self.distribution,
        ).to(self.device)

        # Loss function (negative log-likelihood for probabilistic model)
        def nll_loss(mean, scale, target):
            """Negative log-likelihood for normal distribution."""
            dist = torch.distributions.Normal(mean, scale)
            return -dist.log_prob(target).mean()

        criterion = nll_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Split data
        n_train = int(len(sequences_normalized) * (1 - validation_split))
        train_dataset = DeepARDataset(
            sequences_normalized[:n_train],
            targets_normalized[:n_train],
            static_feat_array[:n_train] if static_feat_array is not None else None,
        )
        val_dataset = DeepARDataset(
            sequences_normalized[n_train:],
            targets_normalized[n_train:],
            static_feat_array[n_train:] if static_feat_array is not None else None,
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
                mean, scale = self.model(batch["sequence"].to(self.device))
                target = batch["target"].to(self.device)
                loss = criterion(mean, scale, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mean, scale = self.model(batch["sequence"].to(self.device))
                    target = batch["target"].to(self.device)
                    loss = criterion(mean, scale, target)
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

    def predict_quantiles(
        self,
        well_id: str,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        horizon: Optional[int] = None,
        n_samples: int = 1000,
    ) -> dict[str, pd.Series]:
        """
        Generate probabilistic forecast with quantiles.

        Args:
            well_id: Well identifier
            production_data: Historical production data
            static_features: Optional static features
            quantiles: List of quantiles to predict
                (e.g., [0.1, 0.5, 0.9] for P10, P50, P90)
            horizon: Forecast horizon (overrides default)
            n_samples: Number of samples for quantile estimation

        Returns:
            Dictionary with quantile forecasts for each phase
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

        # Normalize
        sequence_normalized = self._normalize_sequences(
            sequence.reshape(1, *sequence.shape), fit=False
        )[0]

        # Predict
        self.model.eval()
        with torch.no_grad():
            seq_tensor = (
                torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            )

            # Generate samples for each forecast step (autoregressive)
            all_samples = []
            current_sequence = seq_tensor

            for step in range(horizon):
                # Predict distribution parameters for this step
                mean, scale = self.model(current_sequence)

                # Sample from distribution (n_samples, n_phases)
                step_samples = torch.normal(
                    mean.expand(n_samples, -1), scale.expand(n_samples, -1)
                )
                all_samples.append(step_samples.cpu().numpy())

                # Use mean prediction as next input (autoregressive)
                # For each sample path, we'd use different values, but for simplicity
                # we use the mean for the next step
                next_input = mean.unsqueeze(1)  # (1, 1, n_phases)
                current_sequence = torch.cat([current_sequence, next_input], dim=1)[
                    :, 1:, :
                ]

            # Stack samples: (horizon, n_samples, n_phases)
            all_samples = np.array(all_samples)

        # Calculate quantiles for each phase
        forecasts = {}
        last_date = pd.to_datetime(well_data["date"].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        for i, phase in enumerate(self.phases):
            phase_samples = all_samples[:, :, i]  # (horizon, n_samples)

            # Calculate quantiles
            phase_quantiles = {}
            for q in quantiles:
                quantile_values = np.percentile(phase_samples, q * 100, axis=1)
                # Denormalize - need to handle single phase case
                if len(self.phases) == 1:
                    # Single phase: quantile_values is (horizon,)
                    quantile_values_2d = quantile_values.reshape(1, -1)
                    quantile_values_denorm = self._denormalize_sequences(
                        quantile_values_2d
                    )[0]
                else:
                    # Multi-phase: need to denormalize with all phases
                    # Create a dummy array with all phases, then extract the one we need
                    quantile_values_full = np.zeros((horizon, len(self.phases)))
                    quantile_values_full[:, i] = quantile_values
                    quantile_values_denorm_full = self._denormalize_sequences(
                        quantile_values_full
                    )
                    quantile_values_denorm = quantile_values_denorm_full[:, i]

                # Clip to non-negative
                quantile_values_denorm = np.clip(quantile_values_denorm, 0, None)
                phase_quantiles[f"q{int(q*100)}"] = pd.Series(
                    quantile_values_denorm,
                    index=forecast_dates,
                    name=f"{phase}_q{int(q*100)}",
                )

            forecasts[phase] = phase_quantiles

        return forecasts

    def _prepare_data(
        self,
        production_data: pd.DataFrame,
        static_features: Optional[pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare data for training."""
        sequences = []
        targets = []
        static_feat_list = []

        # Group by well_id
        for well_id, well_data in production_data.groupby("well_id"):
            well_data = well_data.sort_values("date").reset_index(drop=True)
            phase_data = well_data[self.phases].values

            # Create sliding windows
            for i in range(len(well_data) - self.sequence_length):
                seq = phase_data[i : i + self.sequence_length]
                target = phase_data[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)

                # Static features (if available)
                if static_features is not None:
                    well_static = static_features[static_features["well_id"] == well_id]
                    if len(well_static) > 0:
                        # Extract numeric columns only
                        numeric_cols = well_static.select_dtypes(
                            include=[np.number]
                        ).columns
                        numeric_cols = [c for c in numeric_cols if c != "well_id"]
                        if len(numeric_cols) > 0:
                            static_feat_list.append(well_static[numeric_cols].values[0])
                        else:
                            static_feat_list.append(None)
                    else:
                        static_feat_list.append(None)
                else:
                    static_feat_list.append(None)

        sequences = np.array(sequences)
        targets = np.array(targets)

        static_feat_array = (
            np.array(static_feat_list)
            if static_feat_list and static_feat_list[0] is not None
            else None
        )

        return sequences, targets, static_feat_array

    def _create_scaler(self) -> Any:
        """Create scaler for normalization."""
        if not SKLEARN_AVAILABLE:
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
        """Normalize sequences."""
        if self.scaler is None:
            # Simple min-max normalization
            if len(sequences.shape) == 3:
                min_vals = sequences.min(axis=(0, 1), keepdims=True)
                max_vals = sequences.max(axis=(0, 1), keepdims=True)
                range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
                return (sequences - min_vals) / range_vals
            else:
                min_vals = sequences.min(axis=0, keepdims=True)
                max_vals = sequences.max(axis=0, keepdims=True)
                range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
                return (sequences - min_vals) / range_vals

        original_shape = sequences.shape
        if len(sequences.shape) == 3:
            n_samples, seq_len, n_features = sequences.shape
            sequences_2d = sequences.reshape(-1, n_features)
        else:
            sequences_2d = sequences

        if fit:
            sequences_normalized = self.scaler.fit_transform(sequences_2d)
        else:
            sequences_normalized = self.scaler.transform(sequences_2d)

        if len(original_shape) == 3:
            sequences_normalized = sequences_normalized.reshape(original_shape)

        return sequences_normalized

    def _denormalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Denormalize sequences."""
        if self.scaler is None:
            return sequences

        original_shape = sequences.shape
        if len(sequences.shape) == 2:
            sequences_2d = sequences
        else:
            sequences_2d = sequences.reshape(-1, sequences.shape[-1])

        sequences_denormalized = self.scaler.inverse_transform(sequences_2d)

        if len(original_shape) == 2:
            return sequences_denormalized
        else:
            return sequences_denormalized.reshape(original_shape)


if not TORCH_AVAILABLE:
    # Fallback when PyTorch is not available
    class DeepARForecaster:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            """Initialize placeholder when PyTorch is not available."""
            raise ImportError(
                "PyTorch is required for DeepAR. Install with: pip install torch"
            )
