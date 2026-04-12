"""Data source module for stock data.

Supports multiple data sources:
- Excel: Read from local Excel files (existing ExcelReader)
- Akshare: Real-time data via akshare library (may have rate limiting issues)
- Sina: Direct API calls to Sina Finance (recommended, gentler rate limiting)
- EastMoney: Direct API calls to East Money (may be blocked by anti-scraping)
"""

from .akshare_source import AkshareDataSource
from .eastmoney_source import EastMoneyDataSource
from .sina_source import SinaDataSource

__all__ = [
    "AkshareDataSource",
    "EastMoneyDataSource",
    "SinaDataSource",
]
