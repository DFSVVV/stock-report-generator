"""Tests for LSTM module."""

from datetime import date

import numpy as np
import pytest
import torch

from src.excel import DailyData, StockDataBundle
from src.lstm import (
    StockLSTM,
    StockDataProcessor,
    PredictionResult,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_volume_ma,
)


class TestTechnicalIndicators:
    """Test technical indicators calculations."""

    @pytest.fixture
    def sample_prices(self) -> list[float]:
        """Sample closing prices for testing."""
        return [
            10.0, 10.5, 10.3, 10.8, 11.0,
            11.2, 11.5, 11.3, 11.8, 12.0,
            12.2, 12.5, 12.3, 12.8, 13.0,
            13.2, 13.5, 13.3, 13.8, 14.0,
        ]

    def test_calculate_sma(self, sample_prices: list[float]) -> None:
        """Test SMA calculation."""
        sma_5 = calculate_sma(sample_prices, 5)

        assert len(sma_5) == len(sample_prices)
        assert np.isnan(sma_5[0])
        assert np.isnan(sma_5[1])
        assert np.isnan(sma_5[2])
        assert np.isnan(sma_5[3])

        expected_sma_5 = sum(sample_prices[0:5]) / 5
        assert abs(sma_5[4] - expected_sma_5) < 0.01

    def test_calculate_sma_short_data(self) -> None:
        """Test SMA with insufficient data."""
        prices = [10.0, 10.5, 10.3]
        sma = calculate_sma(prices, 5)

        assert len(sma) == 3
        assert all(np.isnan(v) for v in sma)

    def test_calculate_ema(self, sample_prices: list[float]) -> None:
        """Test EMA calculation."""
        ema_5 = calculate_ema(sample_prices, 5)

        assert len(ema_5) == len(sample_prices)
        assert np.isnan(ema_5[0])
        assert np.isnan(ema_5[1])
        assert np.isnan(ema_5[2])
        assert np.isnan(ema_5[3])
        assert not np.isnan(ema_5[4])

    def test_calculate_rsi(self, sample_prices: list[float]) -> None:
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_prices, 14)

        assert len(rsi) == len(sample_prices)
        assert all(np.isnan(v) for v in rsi[:14])
        assert not np.isnan(rsi[14])
        assert 0 <= rsi[14] <= 100

    def test_calculate_macd(self, sample_prices: list[float]) -> None:
        """Test MACD calculation."""
        dif, dea, hist = calculate_macd(sample_prices)

        assert len(dif) == len(sample_prices)
        assert len(dea) == len(sample_prices)
        assert len(hist) == len(sample_prices)

    def test_calculate_bollinger_bands(self, sample_prices: list[float]) -> None:
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(sample_prices, 20)

        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)

        for i in range(19, len(sample_prices)):
            assert upper[i] >= middle[i]
            assert lower[i] <= middle[i]

    def test_calculate_volume_ma(self) -> None:
        """Test volume MA calculation."""
        volumes = [1000, 1500, 1200, 1800, 2000, 1600, 1900, 2100, 1700, 2200]
        vol_ma = calculate_volume_ma(volumes, 5)

        assert len(vol_ma) == len(volumes)
        assert np.isnan(vol_ma[0])
        assert np.isnan(vol_ma[1])
        assert np.isnan(vol_ma[2])
        assert np.isnan(vol_ma[3])

        expected = sum(volumes[0:5]) / 5
        assert abs(vol_ma[4] - expected) < 0.01


class TestStockDataProcessor:
    """Test StockDataProcessor."""

    @pytest.fixture
    def sample_bundle(self) -> StockDataBundle:
        """Create sample stock data bundle."""
        data = []
        base_price = 10.0

        for i in range(100):
            d = DailyData(
                stock_code="000001",
                trade_date=date(2024, 1, 1).replace(day=(i % 28) + 1),
                open=base_price + i * 0.1,
                high=base_price + i * 0.1 + 0.5,
                low=base_price + i * 0.1 - 0.3,
                close=base_price + i * 0.1 + 0.1,
                volume=1000000 + i * 10000,
                amount=10000000 + i * 100000,
                turnover_rate=0.5 + (i % 10) * 0.1,
                change_pct=0.5 + (i % 5) * 0.2,
                change_amount=0.05 + (i % 3) * 0.02,
            )
            data.append(d)

        return StockDataBundle(stock_code="000001", data=data)

    def test_normalize(self) -> None:
        """Test Z-score normalization."""
        processor = StockDataProcessor(sequence_length=60)
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]

        normalized = processor.normalize(prices)

        assert len(normalized) == len(prices)
        assert abs(np.mean(normalized)) < 0.01

    def test_compute_technical_features(self, sample_bundle: StockDataBundle) -> None:
        """Test technical features computation."""
        processor = StockDataProcessor(sequence_length=60)
        features = processor.compute_technical_features(sample_bundle)

        assert "close" in features
        assert "volume" in features
        assert "sma_5" in features
        assert "sma_10" in features
        assert "sma_20" in features
        assert "rsi" in features
        assert "macd_dif" in features
        assert "macd_hist" in features
        assert "bb_upper" in features
        assert "bb_lower" in features

        assert len(features["close"]) == len(sample_bundle.data)

    def test_prepare_lstm_input(self, sample_bundle: StockDataBundle) -> None:
        """Test LSTM input preparation."""
        processor = StockDataProcessor(sequence_length=60)
        result = processor.prepare_lstm_input(sample_bundle)

        assert "sequence" in result
        assert "close_prices" in result
        assert "dates" in result

        assert result["sequence"].shape[1] == 10
        assert len(result["close_prices"]) == len(sample_bundle.data)
        assert len(result["dates"]) == len(sample_bundle.data)

    def test_create_sequences(self, sample_bundle: StockDataBundle) -> None:
        """Test sequence creation."""
        processor = StockDataProcessor(sequence_length=60)
        sequences, labels = processor.create_sequences(sample_bundle)

        expected_sequences = len(sample_bundle.data) - 60 - 1 + 1
        assert sequences.shape[0] == expected_sequences
        assert sequences.shape[1] == 60
        assert sequences.shape[2] == 10
        assert len(labels) == expected_sequences

    def test_create_sequences_insufficient_data(self) -> None:
        """Test sequence creation with insufficient data."""
        data = []
        for i in range(30):
            d = DailyData(
                stock_code="000001",
                trade_date=date(2024, 1, 1).replace(day=i + 1),
                open=10.0,
                high=10.5,
                low=9.5,
                close=10.2,
                volume=1000000,
                amount=10000000,
                turnover_rate=0.5,
                change_pct=0.5,
                change_amount=0.05,
            )
            data.append(d)

        bundle = StockDataBundle(stock_code="000001", data=data)
        processor = StockDataProcessor(sequence_length=60)

        with pytest.raises(ValueError, match="Not enough data"):
            processor.create_sequences(bundle)


class TestStockLSTM:
    """Test StockLSTM model."""

    def test_model_initialization(self) -> None:
        """Test model can be initialized."""
        model = StockLSTM(
            input_size=10,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )

        assert model is not None
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.bidirectional is True

    def test_model_forward(self) -> None:
        """Test model forward pass."""
        model = StockLSTM(input_size=10, hidden_size=64, num_layers=1, bidirectional=False)

        batch_size = 4
        seq_len = 60
        input_size = 10

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_model_predict_with_features(self) -> None:
        """Test model prediction with attention features."""
        model = StockLSTM(input_size=10, hidden_size=64, num_layers=1, bidirectional=True)

        x = torch.randn(2, 60, 10)
        output, attention = model.predict_with_features(x)

        assert output.shape == (2, 1)
        assert attention.shape == (2, 60, 1)


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_valid_trend(self) -> None:
        """Test valid trend values."""
        for trend in ["UP", "DOWN", "STABLE"]:
            result = PredictionResult(
                next_day_return=1.5,
                confidence=0.8,
                trend=trend,
            )
            assert result.trend == trend

    def test_invalid_trend(self) -> None:
        """Test invalid trend raises error."""
        with pytest.raises(ValueError, match="Invalid trend"):
            PredictionResult(
                next_day_return=1.5,
                confidence=0.8,
                trend="INVALID",
            )
