"""Akshare data source for getting real stock data."""

import random
import time
from datetime import date
from typing import Optional

from ..excel.models import DailyData, StockDataBundle


class AkshareDataSource:
    """Data source using akshare library to fetch real stock data.

    Akshare is a free, open-source financial data interface
    that provides access to Chinese A-share market data.

    Usage:
        from src.data_source import AkshareDataSource

        source = AkshareDataSource()
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
    ]

    def __init__(self):
        """Initialize akshare data source."""
        try:
            import akshare as ak

            self.ak = ak
        except ImportError:
            raise ImportError(
                "akshare is not installed. Please run: pip install akshare"
            )

        self._session_headers = {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    def _random_delay(self, min_sec: float = 0.5, max_sec: float = 2.5) -> None:
        """Add random delay to avoid rate limiting.

        Args:
            min_sec: Minimum delay in seconds
            max_sec: Maximum delay in seconds
        """
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)

    def _get_with_retry(
        self,
        func,
        *args,
        max_retries: int = 3,
        **kwargs
    ):
        """Execute function with retry logic and random delays.

        Args:
            func: Function to execute
            *args: Function arguments
            max_retries: Maximum number of retries
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Random delay before request (longer on retry)
                base_delay = 1.0 + attempt * 0.5
                self._random_delay(base_delay, base_delay + 2.0)

                # Update headers with new random User-Agent
                import requests
                session = requests.Session()
                session.headers.update(self._session_headers)

                # Call the function
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                # Exponential backoff on retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    time.sleep(wait_time)
                    # Get new User-Agent for retry
                    self._session_headers["User-Agent"] = random.choice(self.USER_AGENTS)
                continue

        raise Exception(f"获取数据失败，已重试 {max_retries} 次: {str(last_error)}")

    def fetch(
        self,
        stock_code: str,
        start_date: str = "20200101",
        end_date: str = "20241231",
        adjust: str = "qfq",
    ) -> StockDataBundle:
        """Fetch stock daily data.

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
        # Add initial random delay
        self._random_delay(1.0, 3.0)

        # Fetch data from akshare with retry logic
        def _fetch_data():
            return self.ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

        df = self._get_with_retry(_fetch_data)

        if df is None or len(df) == 0:
            raise ValueError(f"No data found for stock {stock_code}")

        daily_data = []

        for idx, row in df.iterrows():
            # Parse date
            date_str = str(row.iloc[0])
            trade_date = self._parse_date(date_str)

            # Parse numeric fields
            try:
                open_price = float(row.iloc[2])
                close_price = float(row.iloc[3])
                high_price = float(row.iloc[4])
                low_price = float(row.iloc[5])
                volume = int(float(row.iloc[6]))
                amount = float(row.iloc[7])
                turnover_rate = float(row.iloc[11])
                change_pct = float(row.iloc[9])
                change_amount = float(row.iloc[10])

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
            except (ValueError, IndexError):
                continue

        if not daily_data:
            raise ValueError(f"No valid data extracted for stock {stock_code}")

        # Sort by date
        daily_data.sort(key=lambda x: x.trade_date)

        # Add delay after successful fetch
        self._random_delay(0.5, 1.5)

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
        """Parse date string to date object."""
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
            try:
                from datetime import datetime
                return datetime.strptime(date_str[:10], fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")

    def get_stock_info(self, stock_code: str) -> dict:
        """Get basic stock information."""
        try:
            self._random_delay(1.0, 2.0)
            df = self.ak.stock_individual_info_em(symbol=stock_code)
            info = {}
            for _, row in df.iterrows():
                info[row.iloc[0]] = row.iloc[1]
            return info
        except Exception:
            return {"stock_code": stock_code}
