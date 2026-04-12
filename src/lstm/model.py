"""LSTM model for stock price prediction."""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """LSTM-based stock prediction model."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention layer for better context understanding
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Output layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Predicted values of shape (batch_size, 1)
        """
        # Input projection: (batch, seq_len, hidden_size)
        projected = torch.relu(self.input_projection(x))

        # LSTM: (batch, seq_len, hidden_size * directions)
        lstm_out, _ = self.lstm(projected)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        output = self.fc(context)
        return output

    def predict_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with intermediate features for analysis.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tuple of (prediction, attention_weights)
        """
        projected = torch.relu(self.input_projection(x))
        lstm_out, _ = self.lstm(projected)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        output = self.fc(context)

        return output, attention_weights
