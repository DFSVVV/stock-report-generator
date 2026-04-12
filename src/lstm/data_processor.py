"""Data processor for converting StockDataBundle to LSTM-ready sequences."""

import numpy as np
from typing import Optional

from ..excel.models import StockDataBundle
from .technical_indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_volume_ma,
)


class StockDataProcessor:
    """Processor for converting stock data to LSTM input format."""

    def __init__(self, sequence_length: int = 60):
        """Initialize processor.

        Args:
            sequence_length: Sequence length for LSTM window (default 60 trading days)
        """
        self.sequence_length = sequence_length
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def normalize(self, data: list[float]) -> np.ndarray:
        """Z-score normalization.

        Args:
            data: List of values to normalize

        Returns:
            Normalized numpy array
        """
        arr = np.array(data, dtype=np.float32)
        if self._mean is None:
            self._mean = np.mean(arr)
            self._std = np.std(arr)

        if self._std == 0:
            return np.zeros_like(arr)

        return (arr - self._mean) / self._std

    def compute_technical_features(
        self, bundle: StockDataBundle
    ) -> dict[str, np.ndarray]:
        """Compute all technical indicators for the stock data.

        Args:
            bundle: Stock data bundle

        Returns:
            Dictionary of feature arrays
        """
        closes = [d.close for d in bundle.data]
        volumes = [d.volume for d in bundle.data]
        turnover_rates = [d.turnover_rate for d in bundle.data]

        # Moving averages
        sma_5 = calculate_sma(closes, 5)
        sma_10 = calculate_sma(closes, 10)
        sma_20 = calculate_sma(closes, 20)

        # RSI
        rsi = calculate_rsi(closes, 14)

        # MACD
        macd_dif, macd_dea, macd_hist = calculate_macd(closes)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, 20)

        # Volume MA
        vol_ma = calculate_volume_ma(volumes, 5)

        return {
            "close": np.array(closes, dtype=np.float32),
            "volume": np.array(volumes, dtype=np.float32),
            "turnover_rate": np.array(turnover_rates, dtype=np.float32),
            "sma_5": np.array(sma_5, dtype=np.float32),
            "sma_10": np.array(sma_10, dtype=np.float32),
            "sma_20": np.array(sma_20, dtype=np.float32),
            "rsi": np.array(rsi, dtype=np.float32),
            "macd_dif": np.array(macd_dif, dtype=np.float32),
            "macd_dea": np.array(macd_dea, dtype=np.float32),
            "macd_hist": np.array(macd_hist, dtype=np.float32),
            "bb_upper": np.array(bb_upper, dtype=np.float32),
            "bb_middle": np.array(bb_middle, dtype=np.float32),
            "bb_lower": np.array(bb_lower, dtype=np.float32),
            "vol_ma": np.array(vol_ma, dtype=np.float32),
        }

    def prepare_lstm_input(
        self, bundle: StockDataBundle, fit_normalizer: bool = True
    ) -> dict[str, np.ndarray]:
        """Create feature matrix for LSTM.

        Args:
            bundle: Stock data bundle
            fit_normalizer: Whether to fit the normalizer on this data

        Returns:
            Dictionary containing:
                - sequence: (seq_len, num_features) feature matrix
                - close_prices: Original close prices for reference
                - dates: Trading dates
        """
        features = self.compute_technical_features(bundle)
        closes = features["close"]

        if fit_normalizer and self._mean is None:
            self._mean = float(np.mean(closes))
            self._std = float(np.std(closes))

        num_samples = len(closes)
        num_features = 10

        # Feature matrix columns:
        # 0: normalized close price
        # 1: normalized volume
        # 2: SMA5 / close - 1 (momentum)
        # 3: SMA10 / close - 1
        # 4: SMA20 / close - 1
        # 5: RSI / 100 (normalized)
        # 6: MACD histogram / close (normalized)
        # 7: BB_upper / close - 1
        # 8: BB_lower / close - 1
        # 9: normalized turnover rate

        feature_matrix = np.zeros((num_samples, num_features), dtype=np.float32)

        for i in range(num_samples):
            close = closes[i]

            # Feature 0: Normalized close price
            if self._std and self._std > 0:
                feature_matrix[i, 0] = (close - self._mean) / self._std
            else:
                feature_matrix[i, 0] = 0.0

            # Feature 1: Normalized volume
            vol_mean = np.mean(features["volume"])
            vol_std = np.std(features["volume"])
            if vol_std > 0:
                feature_matrix[i, 1] = (features["volume"][i] - vol_mean) / vol_std
            else:
                feature_matrix[i, 1] = 0.0

            # Feature 2-4: Moving average momentum
            sma5 = features["sma_5"][i]
            sma10 = features["sma_10"][i]
            sma20 = features["sma_20"][i]

            if not np.isnan(sma5) and close > 0:
                feature_matrix[i, 2] = sma5 / close - 1
            if not np.isnan(sma10) and close > 0:
                feature_matrix[i, 3] = sma10 / close - 1
            if not np.isnan(sma20) and close > 0:
                feature_matrix[i, 4] = sma20 / close - 1

            # Feature 5: Normalized RSI
            rsi_val = features["rsi"][i]
            if not np.isnan(rsi_val):
                feature_matrix[i, 5] = rsi_val / 100.0
            else:
                feature_matrix[i, 5] = 0.5

            # Feature 6: Normalized MACD histogram
            macd_hist = features["macd_hist"][i]
            if not np.isnan(macd_hist) and close > 0:
                feature_matrix[i, 6] = macd_hist / close
            else:
                feature_matrix[i, 6] = 0.0

            # Feature 7-8: Bollinger Bands position
            bb_upper = features["bb_upper"][i]
            bb_lower = features["bb_lower"][i]
            if not np.isnan(bb_upper) and not np.isnan(bb_lower) and close > 0:
                feature_matrix[i, 7] = bb_upper / close - 1
                feature_matrix[i, 8] = bb_lower / close - 1
            else:
                feature_matrix[i, 7] = 0.0
                feature_matrix[i, 8] = 0.0

            # Feature 9: Normalized turnover rate
            tr_mean = np.mean(features["turnover_rate"])
            tr_std = np.std(features["turnover_rate"])
            if tr_std > 0:
                feature_matrix[i, 9] = (features["turnover_rate"][i] - tr_mean) / tr_std
            else:
                feature_matrix[i, 9] = 0.0

        # Replace NaN with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "sequence": feature_matrix,
            "close_prices": closes,
            "dates": [d.trade_date for d in bundle.data],
        }

    def create_sequences(
        self, bundle: StockDataBundle, forecast_horizon: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM training.

        Args:
            bundle: Stock data bundle
            forecast_horizon: Number of days ahead to predict (default 1)

        Returns:
            Tuple of (sequences, labels)
                - sequences: (num_sequences, sequence_length, num_features)
                - labels: (num_sequences,) - next day return percentage
        """
        prepared = self.prepare_lstm_input(bundle)
        feature_matrix = prepared["sequence"]
        close_prices = prepared["close_prices"]

        num_total = len(feature_matrix)
        num_sequences = num_total - self.sequence_length - forecast_horizon + 1

        if num_sequences <= 0:
            raise ValueError(
                f"Not enough data for sequence length {self.sequence_length}. "
                f"Need at least {self.sequence_length + forecast_horizon} days, "
                f"got {num_total}."
            )

        sequences = np.zeros(
            (num_sequences, self.sequence_length, feature_matrix.shape[1]),
            dtype=np.float32,
        )
        labels = np.zeros(num_sequences, dtype=np.float32)

        for i in range(num_sequences):
            # Extract sequence
            sequences[i] = feature_matrix[i : i + self.sequence_length]

            # Calculate label: percentage change from last day in sequence to forecast_horizon days later
            last_close = close_prices[i + self.sequence_length - 1]
            target_idx = i + self.sequence_length + forecast_horizon - 1

            if target_idx < num_total and last_close > 0:
                target_close = close_prices[target_idx]
                labels[i] = (target_close - last_close) / last_close * 100
            else:
                labels[i] = 0.0

        return sequences, labels
