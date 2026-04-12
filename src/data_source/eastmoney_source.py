"""East Money (东方财富) data source for stock data.

Directly fetches stock data from East Money API without akshare.
This avoids rate limiting issues during development.
"""

import json
import random
import time
from datetime import date, datetime
from typing import Optional

import requests

from ..excel.models import DailyData, StockDataBundle


class EastMoneyDataSource:
    """Data source using East Money API directly.

    Fetches Chinese A-share market data from East Money's official API,
    bypassing akshare to avoid rate limiting and IP blocking issues.

    Usage:
        from src.data_source import EastMoneyDataSource

        source = EastMoneyDataSource()
        bundle = source.fetch('000001', start_date='20240101', end_date='20240412')
        print(f"Fetched {len(bundle.data)} days of data for {bundle.stock_code}")
    """

    # Random User-Agent pool to avoid detection
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ]

    # East Money API URL for daily K-line data
    API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    # Request params template
    PARAMS = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",  # Daily K-line
        "fqt": "1",  # QFQ (front adjustment)
    }

    def __init__(self):
        """Initialize East Money data source."""
        self._session: Optional[requests.Session] = None
        self._last_request_time = 0.0
        self._min_request_interval = 2.0  # Minimum interval between requests (seconds)

    def _get_session(self) -> requests.Session:
        """Get or create a requests session with proper headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": random.choice(self.USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Referer": "https://quote.eastmoney.com/",
            })
        return self._session

    def _rotate_user_agent(self) -> None:
        """Rotate User-Agent to avoid detection."""
        if self._session:
            self._session.headers["User-Agent"] = random.choice(self.USER_AGENTS)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _random_delay(self, min_sec: float = 0.3, max_sec: float = 1.0) -> None:
        """Add small random delay.

        Args:
            min_sec: Minimum delay in seconds
            max_sec: Maximum delay in seconds
        """
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)

    def _get_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = 3,
    ) -> dict:
        """Execute request with retry logic.

        Args:
            url: API URL
            params: Request parameters
            max_retries: Maximum number of retries

        Returns:
            JSON response data

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()

                # Rotate User-Agent on retry
                if attempt > 0:
                    self._rotate_user_agent()

                session = self._get_session()
                response = session.get(url, params=params, timeout=10)
                response.raise_for_status()

                return response.json()

            except requests.RequestException as e:
                last_error = e
                # Exponential backoff on retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                    time.sleep(wait_time)
                continue

        raise Exception(f"获取数据失败，已重试 {max_retries} 次: {str(last_error)}")

    def fetch(
        self,
        stock_code: str,
        start_date: str = "20200101",
        end_date: str = "20241231",
        adjust: str = "qfq",
    ) -> StockDataBundle:
        """Fetch stock daily data from East Money API.

        Args:
            stock_code: Stock code (e.g., '000001' for 平安银行)
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            adjust: Price adjustment type ('qfq' for 前复权, 'hfq' for 后复权, '' for 不复权)

        Returns:
            StockDataBundle containing daily data

        Raises:
            ConnectionError: If fails to fetch data from network
            ValueError: If stock code is invalid or no data found
        """
        # Determine market code: Shanghai (6xx) = 1, Shenzhen (0xx, 3xx) = 0
        market_code = 1 if stock_code.startswith("6") else 0

        # Map adjust parameter to East Money API format
        adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
        adjust_code = adjust_dict.get(adjust, "1")

        # Build request parameters
        params = {
            **self.PARAMS,
            "secid": f"{market_code}.{stock_code}",
            "fqt": adjust_code,
            "beg": start_date,
            "end": end_date,
        }

        # Initial delay
        self._random_delay(0.5, 1.5)

        # Fetch data with retry
        data_json = self._get_with_retry(self.API_URL, params)

        # Check if data exists
        if not data_json.get("data") or not data_json["data"].get("klines"):
            raise ValueError(f"No data found for stock {stock_code}")

        # Parse klines data
        daily_data = []
        klines = data_json["data"]["klines"]

        for item in klines:
            parts = item.split(",")
            if len(parts) < 11:
                continue

            try:
                trade_date = self._parse_date(parts[0])
                open_price = float(parts[1])
                close_price = float(parts[2])
                high_price = float(parts[3])
                low_price = float(parts[4])
                volume = int(float(parts[5]))
                amount = float(parts[6])
                # parts[7] is amplitude (振幅)
                change_pct = float(parts[8]) if parts[8] else 0.0
                change_amount = float(parts[9]) if parts[9] else 0.0
                turnover_rate = float(parts[10]) if parts[10] else 0.0

                daily = DailyData(
                    stock_code=stock_code,
                    trade_date=trade_date,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    amount=amount,
                    turnover_rate=turnover_rate,
                    change_pct=change_pct,
                    change_amount=change_amount,
                )
                daily_data.append(daily)
            except (ValueError, IndexError) as e:
                # Skip invalid rows
                continue

        if not daily_data:
            raise ValueError(f"No valid data extracted for stock {stock_code}")

        # Sort by date
        daily_data.sort(key=lambda x: x.trade_date)

        # Small delay after successful fetch
        self._random_delay(0.2, 0.5)

        return StockDataBundle(stock_code=stock_code, data=daily_data)

    def fetch_with_name(
        self,
        stock_code: str,
        stock_name: str,
        start_date: str = "20200101",
        end_date: str = "20241231",
        adjust: str = "qfq",
    ) -> tuple[StockDataBundle, str]:
        """Fetch stock data and return with stock name."""
        bundle = self.fetch(stock_code, start_date, end_date, adjust)
        bundle.stock_name = stock_name
        return bundle, stock_name

    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object.

        Args:
            date_str: Date string in various formats

        Returns:
            date object

        Raises:
            ValueError: If date cannot be parsed
        """
        # East Money API returns dates in YYYY-MM-DD format
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
            try:
                return datetime.strptime(date_str[:10], fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")

    def get_stock_info(self, stock_code: str) -> dict:
        """Get basic stock information.

        Args:
            stock_code: Stock code

        Returns:
            Dictionary with stock information
        """
        market_code = 1 if stock_code.startswith("6") else 0
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": f"{market_code}.{stock_code}",
            "fields": "f58,f107,f43,f57,f58,f107,f29,f169",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
        }

        try:
            self._random_delay(0.5, 1.0)
            data_json = self._get_with_retry(url, params)

            if data_json.get("data"):
                data = data_json["data"]
                return {
                    "stock_code": stock_code,
                    "stock_name": data.get("f58", ""),
                    "current_price": data.get("f43", 0),
                    "change_pct": data.get("f169", 0),
                    "volume": data.get("f47", 0),
                }
        except Exception:
            pass

        return {"stock_code": stock_code}
