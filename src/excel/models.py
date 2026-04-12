"""Data models for stock report generator."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class DailyData:
    """Daily stock trading data.

    Attributes:
        stock_code: Stock code, e.g., "000001"
        trade_date: Trading date
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume (shares)
        amount: Trading amount (yuan)
        turnover_rate: Turnover rate (%)
        change_pct: Change percentage (%)
        change_amount: Change amount
    """

    stock_code: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float
    turnover_rate: float
    change_pct: float
    change_amount: float

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if self.close < 0:
            raise ValueError(f"Close price cannot be negative: {self.close}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass
class StockDataBundle:
    """Bundle of stock data with metadata.

    Attributes:
        stock_code: Stock code
        stock_name: Optional stock name
        data: List of daily data records
    """

    stock_code: str
    data: list[DailyData] = field(default_factory=list)
    stock_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate bundle after initialization."""
        if not self.data:
            raise ValueError("StockDataBundle must contain at least one DailyData record")
        # Verify all data belongs to the same stock
        for d in self.data:
            if d.stock_code != self.stock_code:
                raise ValueError(
                    f"Stock code mismatch: bundle is {self.stock_code}, "
                    f"but data contains {d.stock_code}"
                )

    @property
    def date_range(self) -> tuple[date, date]:
        """Get the date range of the data.

        Returns:
            Tuple of (earliest_date, latest_date)
        """
        dates = [d.trade_date for d in self.data]
        return min(dates), max(dates)

    @property
    def latest_close(self) -> float:
        """Get the latest closing price."""
        return self.data[-1].close

    @property
    def latest_volume(self) -> int:
        """Get the latest trading volume."""
        return self.data[-1].volume

    @property
    def avg_turnover_rate(self) -> float:
        """Calculate average turnover rate."""
        if not self.data:
            return 0.0
        return sum(d.turnover_rate for d in self.data) / len(self.data)
