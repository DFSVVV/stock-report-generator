"""Formatting utilities for report generation."""

from datetime import date
from typing import Union


def format_price(price: Union[float, int]) -> str:
    """Format price to 2 decimal places.

    Args:
        price: Price value

    Returns:
        Formatted price string
    """
    return f"{float(price):.2f}"


def format_percentage(value: Union[float, int], include_sign: bool = True) -> str:
    """Format as percentage with optional sign.

    Args:
        value: Percentage value
        include_sign: Whether to include + sign for positive values

    Returns:
        Formatted percentage string
    """
    val = float(value)
    if include_sign and val > 0:
        return f"+{val:.2f}%"
    return f"{val:.2f}%"


def format_volume(volume: Union[int, float]) -> str:
    """Format large volume numbers.

    Args:
        volume: Volume value

    Returns:
        Formatted volume string (e.g., "1500万", "1.5亿")
    """
    vol = float(volume)
    if vol >= 1e8:
        return f"{vol / 1e8:.2f}亿"
    elif vol >= 1e4:
        return f"{vol / 1e4:.0f}万"
    return str(int(vol))


def format_date(date_obj: date) -> str:
    """Format date as YYYY-MM-DD.

    Args:
        date_obj: Date object

    Returns:
        Formatted date string
    """
    return date_obj.strftime("%Y-%m-%d")


def format_date_chinese(date_obj: date) -> str:
    """Format date in Chinese style.

    Args:
        date_obj: Date object

    Returns:
        Formatted date string (e.g., "2024年1月15日")
    """
    return date_obj.strftime("%Y年%m月%d日")
