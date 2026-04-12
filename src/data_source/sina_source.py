"""Sina Finance data source for stock data.

Directly fetches stock data from Sina Finance API without akshare.
Sina has gentler rate limiting compared to East Money.
"""

import json
import os
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from ..excel.models import DailyData, StockDataBundle

# Local cache file for circulating shares data
CACHE_DIR = Path.home() / ".stock_report_generator"
CACHE_FILE = CACHE_DIR / "circulating_shares.json"


class SinaDataSource:
    """Data source using Sina Finance API directly.

    Fetches Chinese A-share market data from Sina Finance API.
    Sina has gentler rate limiting - IP blocks are released in minutes
    rather than hours like East Money.

    Usage:
        from src.data_source import SinaDataSource

        source = SinaDataSource()
        bundle = source.fetch('000001', start_date='20240101', end_date='20240412')
        print(f"Fetched {len(bundle.data)} days of data for {bundle.stock_code}")
    """

    # Random User-Agent pool
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
    ]

    # Sina Finance API URL
    API_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_kline_=/CN_MarketDataService.getKLineData"

    def __init__(self):
        """Initialize Sina data source."""
        self._session: Optional[requests.Session] = None
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # Minimum interval between requests (seconds)
        self._circulating_shares: Optional[dict] = None
        self._load_circulating_shares_cache()

    def _load_circulating_shares_cache(self) -> None:
        """Load cached circulating shares data from file."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._circulating_shares = json.load(f)
            except Exception:
                self._circulating_shares = {}

    def _save_circulating_shares_cache(self) -> None:
        """Save circulating shares data to file."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._circulating_shares, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_tencent_session(self) -> requests.Session:
        """Get or create a requests session for Tencent API."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": random.choice(self.USER_AGENTS),
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://finance.sina.com.cn/",
            })
        return self._session

    def _get_circulating_shares_from_tencent(self, stock_code: str) -> float:
        """Get circulating shares from Tencent API and cache it.

        Args:
            stock_code: Stock code like '000001'

        Returns:
            Circulating shares in shares (not in 100 shares units)
        """
        symbol = f"sz{stock_code}" if not stock_code.startswith(("6", "9")) else f"sh{stock_code}"

        url = f"https://qt.gtimg.cn/q={symbol}"
        session = self._get_tencent_session()

        try:
            response = session.get(url, timeout=10)
            text = response.text

            # Parse Tencent data: field 38 is turnover rate (%), field 36 is volume (hand = 100 shares)
            start = text.find('"')
            end = text.rfind('"')
            if start < 0 or end <= start:
                return 0.0

            fields = text[start+1:end].split('~')
            if len(fields) < 39:
                return 0.0

            volume_hand = int(fields[36]) if fields[36] else 0  # volume in 100 shares
            turnover_rate = float(fields[38]) if fields[38] else 0.0  # in percentage

            if turnover_rate > 0 and volume_hand > 0:
                # circulating_shares = volume * 100 / (turnover_rate / 100)
                # = volume * 100 * 100 / turnover_rate
                # = volume * 10000 / turnover_rate
                volume_shares = volume_hand * 100
                circulating_shares = volume_shares * 100 / turnover_rate
                return circulating_shares

        except Exception:
            pass

        return 0.0

    def get_circulating_shares(self, stock_code: str, force_update: bool = False) -> float:
        """Get circulating shares for a stock, using cache if available.

        Args:
            stock_code: Stock code
            force_update: If True, fetch fresh data from Tencent API

        Returns:
            Circulating shares in shares
        """
        if self._circulating_shares is None:
            self._circulating_shares = {}

        if not force_update and stock_code in self._circulating_shares:
            return self._circulating_shares[stock_code]

        # Fetch from Tencent API
        circulating = self._get_circulating_shares_from_tencent(stock_code)

        if circulating > 0:
            self._circulating_shares[stock_code] = circulating
            self._save_circulating_shares_cache()

        return circulating

    def _get_session(self) -> requests.Session:
        """Get or create a requests session with proper headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": random.choice(self.USER_AGENTS),
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Referer": "https://finance.sina.com.cn/",
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

    def _random_delay(self, min_sec: float = 0.2, max_sec: float = 0.8) -> None:
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
    ) -> list:
        """Execute request with retry logic.

        Args:
            url: API URL
            params: Request parameters
            max_retries: Maximum number of retries

        Returns:
            List of kline data

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

                # Parse JSONP response
                text = response.text
                start = text.find('[')
                end = text.rfind(']') + 1
                if start < 0 or end <= start:
                    raise ValueError(f"Invalid JSONP response: {text[:100]}")

                return json.loads(text[start:end])

            except requests.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    time.sleep(wait_time)
                continue
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    time.sleep(wait_time)
                continue

        raise Exception(f"获取数据失败，已重试 {max_retries} 次: {str(last_error)}")

    def _stock_prefix(self, stock_code: str) -> str:
        """Get Sina stock prefix for the code.

        Args:
            stock_code: Stock code like '000001'

        Returns:
            Sina symbol like 'sz000001' or 'sh600000'
        """
        if stock_code.startswith('6') or stock_code.startswith('9'):
            return f"sh{stock_code}"
        else:
            return f"sz{stock_code}"

    def fetch(
        self,
        stock_code: str,
        start_date: str = "20200101",
        end_date: str = "20241231",
        adjust: str = "qfq",
    ) -> StockDataBundle:
        """Fetch stock daily data from Sina Finance API.

        Sina API returns: day, open, high, low, close, volume, ma_price5, ma_volume5

        Args:
            stock_code: Stock code (e.g., '000001' for 平安银行)
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            adjust: Price adjustment type (only 'qfq' is supported for now)

        Returns:
            StockDataBundle containing daily data

        Raises:
            ConnectionError: If fails to fetch data from network
            ValueError: If stock code is invalid or no data found
        """
        symbol = self._stock_prefix(stock_code)

        # Calculate how many trading days to fetch (roughly)
        # Approximately 250 trading days per year
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        days_diff = (end_dt - start_dt).days
        # Add buffer since not every day is a trading day
        datalen = min(days_diff + 50, 500)  # Cap at 500 to avoid too large requests

        params = {
            "symbol": symbol,
            "scale": "240",  # Daily K-line
            "ma": "no",  # Don't need moving averages from API
            "datalen": str(datalen),
        }

        # Initial delay
        self._random_delay(0.3, 1.0)

        # Fetch data with retry
        data = self._get_with_retry(self.API_URL, params)

        if not data:
            raise ValueError(f"No data found for stock {stock_code}")

        # Try to get circulating shares for turnover rate calculation
        circulating_shares = self.get_circulating_shares(stock_code)

        daily_data = []
        prev_close = None

        for item in data:
            try:
                trade_date = self._parse_date(item['day'])

                # Skip data outside date range
                if trade_date < start_dt.date() or trade_date > end_dt.date():
                    continue

                open_price = float(item['open'])
                close_price = float(item['close'])
                high_price = float(item['high'])
                low_price = float(item['low'])
                volume = int(float(item['volume']))

                # Calculate change_pct and change_amount
                if prev_close is not None and prev_close > 0:
                    change_amount = close_price - prev_close
                    change_pct = (change_amount / prev_close) * 100
                else:
                    change_amount = 0.0
                    change_pct = 0.0

                # Calculate turnover rate from volume and circulating shares
                if circulating_shares > 0:
                    # turnover_rate = (volume / circulating_shares) * 100
                    turnover_rate = (volume / circulating_shares) * 100
                    turnover_rate = round(turnover_rate, 2)
                else:
                    turnover_rate = 0.0

                # Sina doesn't provide amount, estimate it
                amount = volume * close_price

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
                    change_pct=round(change_pct, 2),
                    change_amount=round(change_amount, 2),
                )
                daily_data.append(daily)
                prev_close = close_price

            except (ValueError, KeyError, IndexError) as e:
                # Skip invalid rows
                continue

        if not daily_data:
            raise ValueError(f"No valid data extracted for stock {stock_code}")

        # Sort by date
        daily_data.sort(key=lambda x: x.trade_date)

        # Small delay after successful fetch
        self._random_delay(0.1, 0.3)

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
            date_str: Date string in YYYY-MM-DD format

        Returns:
            date object
        """
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()

    def get_stock_info(self, stock_code: str) -> dict:
        """Get basic stock information.

        Args:
            stock_code: Stock code

        Returns:
            Dictionary with stock information
        """
        symbol = self._stock_prefix(stock_code)

        # Try to get stock name from Tencent API
        stock_name = self._get_stock_name_from_tencent(symbol)
        if stock_name:
            return {"stock_code": stock_code, "symbol": symbol, "stock_name": stock_name}

        return {"stock_code": stock_code, "symbol": symbol}

    def _get_stock_name_from_tencent(self, symbol: str) -> Optional[str]:
        """Get stock name from Tencent API.

        Args:
            symbol: Sina symbol like 'sz000001' or 'sh600000'

        Returns:
            Stock name or None if not found
        """
        url = f"https://qt.gtimg.cn/q={symbol}"
        session = self._get_session()

        try:
            response = session.get(url, timeout=10)
            text = response.text

            start = text.find('"')
            end = text.rfind('"')
            if start < 0 or end <= start:
                return None

            fields = text[start+1:end].split('~')
            if len(fields) > 1:
                name = fields[1].strip()
                # Remove extra spaces
                name = ' '.join(name.split())
                if name:
                    return name
        except Exception:
            pass

        return None

    def fetch_with_name(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        start_date: str = "20200101",
        end_date: str = "20241231",
        adjust: str = "qfq",
    ) -> tuple[StockDataBundle, str]:
        """Fetch stock data and return with stock name.

        If stock_name is not provided or equals stock_code, will try to fetch from Tencent API.
        """
        bundle = self.fetch(stock_code, start_date, end_date, adjust)

        # Try to get stock name if not provided or same as code
        if not stock_name or stock_name == stock_code:
            symbol = self._stock_prefix(stock_code)
            fetched_name = self._get_stock_name_from_tencent(symbol)
            if fetched_name:
                stock_name = fetched_name

        if not stock_name:
            stock_name = stock_code

        bundle.stock_name = stock_name
        return bundle, stock_name
