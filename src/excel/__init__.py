"""Excel data processing module for stock report generator.

Format: Single stock per file with "日线数据" sheet.

Example usage:
    from src.excel import ExcelReader, StockDataBundle

    # Read stock data from Excel
    reader = ExcelReader("data/example_000001.xlsx")
    result = reader.validate()
    if result.valid:
        bundle = reader.read()
        print(f"Stock: {bundle.stock_code}")
        print(f"Date range: {bundle.date_range}")
        print(f"Latest close: {bundle.latest_close}")
"""

from .models import DailyData, StockDataBundle
from .reader import ExcelReader, ValidationError, ValidationResult
from .schema import COLUMNS, REQUIRED_COLUMNS, SHEET_NAME

__all__ = [
    "ExcelReader",
    "DailyData",
    "StockDataBundle",
    "ValidationError",
    "ValidationResult",
    "SHEET_NAME",
    "COLUMNS",
    "REQUIRED_COLUMNS",
]
