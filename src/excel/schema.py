"""Excel data format schema for stock report generator.

Format: Single stock per file (stock code in filename)
Sheet: "日线数据"

Columns:
    - stock_code: str, e.g., "000001"
    - trade_date: date, e.g., "2024-01-15"
    - open: float, 开盘价
    - high: float, 最高价
    - low: float, 最低价
    - close: float, 收盘价
    - volume: int, 成交量 (股数)
    - amount: float, 成交额 (元)
    - turnover_rate: float, 换手率 (%)
    - change_pct: float, 涨跌幅 (%)
    - change_amount: float, 涨跌额
"""

from typing import TypedDict


# Sheet name
SHEET_NAME = "日线数据"


class ColumnConfig(TypedDict):
    """Column configuration with Chinese name and expected type."""

    name: str
    dtype: str  # "str", "float", "int", "date"


# Column definitions: key -> (Chinese name, expected dtype)
COLUMNS: dict[str, ColumnConfig] = {
    "stock_code": {"name": "股票代码", "dtype": "str"},
    "trade_date": {"name": "交易日期", "dtype": "date"},
    "open": {"name": "开盘价", "dtype": "float"},
    "high": {"name": "最高价", "dtype": "float"},
    "low": {"name": "最低价", "dtype": "float"},
    "close": {"name": "收盘价", "dtype": "float"},
    "volume": {"name": "成交量", "dtype": "int"},
    "amount": {"name": "成交额", "dtype": "float"},
    "turnover_rate": {"name": "换手率", "dtype": "float"},
    "change_pct": {"name": "涨跌幅", "dtype": "float"},
    "change_amount": {"name": "涨跌额", "dtype": "float"},
}

# Minimum required columns for validation
REQUIRED_COLUMNS = ["stock_code", "trade_date", "close", "volume"]


def get_column_letter(index: int) -> str:
    """Convert 0-based column index to Excel column letter (A, B, C, ...).

    Args:
        index: 0-based column index

    Returns:
        Excel column letter (A, B, C, ..., Z, AA, AB, ...)
    """
    result = ""
    index += 1  # Convert to 1-based
    while index > 0:
        index -= 1
        result = chr(65 + index % 26) + result
        index //= 26
    return result


def get_column_index(letter: str) -> int:
    """Convert Excel column letter to 0-based column index.

    Args:
        letter: Excel column letter (A, B, C, ..., Z, AA, AB, ...)

    Returns:
        0-based column index
    """
    result = 0
    for char in letter.upper():
        result = result * 26 + (ord(char) - 64)
    return result - 1
