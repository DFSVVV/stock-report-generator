"""Tests for Excel data processing module."""

from datetime import date
from pathlib import Path

import pytest

from src.excel import (
    COLUMNS,
    ExcelReader,
    SHEET_NAME,
    DailyData,
    StockDataBundle,
    ValidationError,
)


class TestSchema:
    """Test schema definitions."""

    def test_sheet_name(self) -> None:
        """Verify sheet name is correct."""
        assert SHEET_NAME == "日线数据"

    def test_columns_defined(self) -> None:
        """Verify all required columns are defined."""
        required = ["stock_code", "trade_date", "close", "volume"]
        for col in required:
            assert col in COLUMNS

    def test_column_names(self) -> None:
        """Verify column Chinese names."""
        assert COLUMNS["stock_code"]["name"] == "股票代码"
        assert COLUMNS["trade_date"]["name"] == "交易日期"
        assert COLUMNS["volume"]["name"] == "成交量"


class TestDailyData:
    """Test DailyData model."""

    def test_valid_daily_data(self) -> None:
        """Test creating valid daily data."""
        data = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 2),
            open=10.5,
            high=10.8,
            low=10.4,
            close=10.7,
            volume=15000000,
            amount=160500000.0,
            turnover_rate=0.55,
            change_pct=0.8,
            change_amount=0.05,
        )
        assert data.stock_code == "000001"
        assert data.close == 10.7

    def test_negative_close_raises(self) -> None:
        """Test that negative close price raises error."""
        with pytest.raises(ValueError, match="Close price cannot be negative"):
            DailyData(
                stock_code="000001",
                trade_date=date(2024, 1, 2),
                open=10.5,
                high=10.8,
                low=10.4,
                close=-10.7,  # Invalid
                volume=15000000,
                amount=160500000.0,
                turnover_rate=0.55,
                change_pct=0.8,
                change_amount=0.05,
            )


class TestStockDataBundle:
    """Test StockDataBundle model."""

    def test_valid_bundle(self) -> None:
        """Test creating valid data bundle."""
        data1 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 2),
            open=10.5,
            high=10.8,
            low=10.4,
            close=10.7,
            volume=15000000,
            amount=160500000.0,
            turnover_rate=0.55,
            change_pct=0.8,
            change_amount=0.05,
        )
        data2 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 3),
            open=10.7,
            high=10.9,
            low=10.6,
            close=10.8,
            volume=16000000,
            amount=172800000.0,
            turnover_rate=0.58,
            change_pct=0.93,
            change_amount=0.1,
        )
        bundle = StockDataBundle(stock_code="000001", data=[data1, data2])
        assert bundle.stock_code == "000001"
        assert len(bundle.data) == 2

    def test_empty_bundle_raises(self) -> None:
        """Test that empty bundle raises error."""
        with pytest.raises(ValueError, match="must contain at least one"):
            StockDataBundle(stock_code="000001", data=[])

    def test_mixed_stock_codes_raises(self) -> None:
        """Test that mixed stock codes raises error."""
        data1 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 2),
            open=10.5,
            high=10.8,
            low=10.4,
            close=10.7,
            volume=15000000,
            amount=160500000.0,
            turnover_rate=0.55,
            change_pct=0.8,
            change_amount=0.05,
        )
        data2 = DailyData(
            stock_code="000002",  # Different stock
            trade_date=date(2024, 1, 3),
            open=10.7,
            high=10.9,
            low=10.6,
            close=10.8,
            volume=16000000,
            amount=172800000.0,
            turnover_rate=0.58,
            change_pct=0.93,
            change_amount=0.1,
        )
        with pytest.raises(ValueError, match="Stock code mismatch"):
            StockDataBundle(stock_code="000001", data=[data1, data2])

    def test_date_range(self) -> None:
        """Test date_range property."""
        data1 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 2),
            open=10.5,
            high=10.8,
            low=10.4,
            close=10.7,
            volume=15000000,
            amount=160500000.0,
            turnover_rate=0.55,
            change_pct=0.8,
            change_amount=0.05,
        )
        data2 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 15),
            open=10.7,
            high=10.9,
            low=10.6,
            close=10.8,
            volume=16000000,
            amount=172800000.0,
            turnover_rate=0.58,
            change_pct=0.93,
            change_amount=0.1,
        )
        bundle = StockDataBundle(stock_code="000001", data=[data1, data2])
        start, end = bundle.date_range
        assert start == date(2024, 1, 2)
        assert end == date(2024, 1, 15)

    def test_latest_close(self) -> None:
        """Test latest_close property."""
        data1 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 2),
            open=10.5,
            high=10.8,
            low=10.4,
            close=10.7,
            volume=15000000,
            amount=160500000.0,
            turnover_rate=0.55,
            change_pct=0.8,
            change_amount=0.05,
        )
        data2 = DailyData(
            stock_code="000001",
            trade_date=date(2024, 1, 15),
            open=10.7,
            high=10.9,
            low=10.6,
            close=10.8,
            volume=16000000,
            amount=172800000.0,
            turnover_rate=0.58,
            change_pct=0.93,
            change_amount=0.1,
        )
        bundle = StockDataBundle(stock_code="000001", data=[data1, data2])
        assert bundle.latest_close == 10.8


class TestExcelReader:
    """Test ExcelReader class."""

    @pytest.fixture
    def example_file(self) -> str:
        """Path to example Excel file."""
        return str(Path(__file__).parent.parent / "data" / "example_000001.xlsx")

    def test_validate_valid_file(self, example_file: str) -> None:
        """Test validation of valid Excel file."""
        reader = ExcelReader(example_file)
        result = reader.validate()
        assert result.valid, result.errors

    def test_validate_nonexistent_file(self) -> None:
        """Test validation of non-existent file."""
        reader = ExcelReader("nonexistent.xlsx")
        result = reader.validate()
        assert not result.valid
        assert "not found" in str(result.errors[0]).lower()

    def test_read_valid_file(self, example_file: str) -> None:
        """Test reading valid Excel file."""
        reader = ExcelReader(example_file)
        bundle = reader.read()
        assert bundle.stock_code == "000001"
        assert len(bundle.data) == 10
        assert bundle.data[0].trade_date == date(2024, 1, 2)

    def test_date_range(self, example_file: str) -> None:
        """Test date range calculation."""
        reader = ExcelReader(example_file)
        bundle = reader.read()
        start, end = bundle.date_range
        assert start == date(2024, 1, 2)
        assert end == date(2024, 1, 29)  # Jan 2 + 27 days = Jan 29
