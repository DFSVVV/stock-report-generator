"""Inference module for stock prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from ..excel.models import StockDataBundle
from .data_processor import StockDataProcessor
from .model import StockLSTM


@dataclass
class PredictionResult:
    """Result of stock prediction."""

    next_day_return: float
    confidence: float
    trend: str
    attention_weights: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate trend value."""
        valid_trends = {"UP", "DOWN", "STABLE"}
        if self.trend not in valid_trends:
            raise ValueError(f"Invalid trend: {self.trend}")


class StockInference:
    """Inference engine for stock prediction."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        sequence_length: int = 60,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            sequence_length: Sequence length used during training
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sequence_length = sequence_length
        self.processor = StockDataProcessor(sequence_length=sequence_length)

        # Load model
        self.model = StockLSTM()
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, bundle: StockDataBundle) -> PredictionResult:
        """Run inference on StockDataBundle.

        Args:
            bundle: Stock data bundle

        Returns:
            PredictionResult with predicted return, confidence, and trend
        """
        if len(bundle.data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} days of data, "
                f"got {len(bundle.data)}"
            )

        # Prepare input
        prepared = self.processor.prepare_lstm_input(bundle, fit_normalizer=False)
        sequence = prepared["sequence"]

        # Take the last sequence_length days
        input_seq = sequence[-self.sequence_length :]
        input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output, attention = self.model.predict_with_features(input_tensor)

        predicted_return = output.item()

        # Calculate confidence based on prediction magnitude
        confidence = self._calculate_confidence(predicted_return, attention)

        # Determine trend
        if predicted_return > 0.5:
            trend = "UP"
        elif predicted_return < -0.5:
            trend = "DOWN"
        else:
            trend = "STABLE"

        return PredictionResult(
            next_day_return=predicted_return,
            confidence=confidence,
            trend=trend,
            attention_weights=attention.squeeze().cpu().numpy(),
        )

    def predict_with_uncertainty(
        self, bundle: StockDataBundle, num_samples: int = 30
    ) -> PredictionResult:
        """Run inference with Monte Carlo dropout for uncertainty estimation.

        Args:
            bundle: Stock data bundle
            num_samples: Number of forward passes with dropout

        Returns:
            PredictionResult with uncertainty estimates
        """
        if len(bundle.data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} days of data, "
                f"got {len(bundle.data)}"
            )

        # Prepare input
        prepared = self.processor.prepare_lstm_input(bundle, fit_normalizer=False)
        sequence = prepared["sequence"]
        input_seq = sequence[-self.sequence_length :]
        input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)

        # Enable dropout
        self.model.train()

        predictions = []
        attentions = []

        with torch.no_grad():
            for _ in range(num_samples):
                output, attention = self.model.predict_with_features(input_tensor)
                predictions.append(output.item())
                attentions.append(attention.squeeze().cpu().numpy())

        # Disable dropout
        self.model.eval()

        predictions = np.array(predictions)
        mean_return = float(np.mean(predictions))
        std_return = float(np.std(predictions))

        # Confidence based on prediction agreement
        confidence = 1.0 - min(std_return / 2.0, 1.0)

        # Trend based on mean
        if mean_return > 0.5:
            trend = "UP"
        elif mean_return < -0.5:
            trend = "DOWN"
        else:
            trend = "STABLE"

        # Average attention weights
        avg_attention = np.mean(attentions, axis=0)

        return PredictionResult(
            next_day_return=mean_return,
            confidence=confidence,
            trend=trend,
            attention_weights=avg_attention,
        )

    def _calculate_confidence(
        self, predicted_return: float, attention: torch.Tensor
    ) -> float:
        """Calculate prediction confidence.

        Args:
            predicted_return: Predicted return value
            attention: Attention weights from model

        Returns:
            Confidence value between 0 and 1
        """
        # Base confidence on prediction magnitude
        magnitude = abs(predicted_return)

        if magnitude > 3.0:
            base_confidence = 0.6
        elif magnitude > 2.0:
            base_confidence = 0.7
        elif magnitude > 1.0:
            base_confidence = 0.8
        elif magnitude > 0.5:
            base_confidence = 0.75
        else:
            base_confidence = 0.85

        # Adjust by attention concentration
        attention_std = float(torch.std(attention))
        attention_factor = 1.0 - min(attention_std * 2, 0.3)

        return base_confidence * attention_factor
