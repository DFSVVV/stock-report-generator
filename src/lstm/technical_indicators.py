"""Technical indicators calculation module for stock analysis."""

import numpy as np


def calculate_sma(prices: list[float], window: int) -> list[float]:
    """Calculate Simple Moving Average.

    Args:
        prices: List of price values
        window: Window size for SMA

    Returns:
        List of SMA values (same length as input, with NaN for first window-1 values)
    """
    if len(prices) < window:
        return [float("nan")] * len(prices)

    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(float("nan"))
        else:
            sma.append(sum(prices[i - window + 1 : i + 1]) / window)
    return sma


def calculate_ema(prices: list[float], window: int) -> list[float]:
    """Calculate Exponential Moving Average.

    Args:
        prices: List of price values
        window: Window size for EMA (span)

    Returns:
        List of EMA values
    """
    if len(prices) < window:
        return [float("nan")] * len(prices)

    ema = [float("nan")] * (window - 1)
    multiplier = 2 / (window + 1)

    # First EMA is SMA
    first_ema = sum(prices[:window]) / window
    ema.append(first_ema)

    for i in range(window, len(prices)):
        current_ema = (prices[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(current_ema)

    return ema


def calculate_rsi(prices: list[float], window: int = 14) -> list[float]:
    """Calculate Relative Strength Index.

    Args:
        prices: List of closing prices
        window: RSI period (default 14)

    Returns:
        List of RSI values (0-100)
    """
    if len(prices) < window + 1:
        return [float("nan")] * len(prices)

    rsi = [float("nan")] * window

    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # First average gain/loss
    gains = [c if c > 0 else 0 for c in changes[:window]]
    losses = [-c if c < 0 else 0 for c in changes[:window]]

    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window

    if avg_loss == 0:
        rsi.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - 100 / (1 + rs))

    # Subsequent RSI values using smoothed averages
    for i in range(window, len(changes)):
        gain = changes[i] if changes[i] > 0 else 0
        loss = -changes[i] if changes[i] < 0 else 0

        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window

        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - 100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: List of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Tuple of (dif_line, dea_line, macd_histogram)
    """
    if len(prices) < slow:
        return [float("nan")] * len(prices), [float("nan")] * len(prices), [float("nan")] * len(prices)

    # Calculate fast and slow EMAs
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    # DIF = Fast EMA - Slow EMA
    dif = [f - s if not (np.isnan(f) or np.isnan(s)) else float("nan") for f, s in zip(ema_fast, ema_slow)]

    # DEA = EMA of DIF with signal period
    valid_dif = [d for d in dif if not np.isnan(d)]
    if len(valid_dif) < signal:
        dea = [float("nan")] * len(dif)
    else:
        dea_ema = calculate_ema(valid_dif, signal)
        dea = [float("nan")] * (len(dif) - len(dea_ema)) + dea_ema

    # MACD Histogram = (DIF - DEA) * 2
    histogram = []
    for d, de in zip(dif, dea):
        if np.isnan(d) or np.isnan(de):
            histogram.append(float("nan"))
        else:
            histogram.append((d - de) * 2)

    return dif, dea, histogram


def calculate_bollinger_bands(
    prices: list[float], window: int = 20, num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """Calculate Bollinger Bands.

    Args:
        prices: List of closing prices
        window: Window size (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < window:
        return [float("nan")] * len(prices), [float("nan")] * len(prices), [float("nan")] * len(prices)

    middle = calculate_sma(prices, window)

    upper = []
    lower = []

    for i in range(len(prices)):
        if i < window - 1:
            upper.append(float("nan"))
            lower.append(float("nan"))
        else:
            segment = prices[i - window + 1 : i + 1]
            std = np.std(segment)
            mid = middle[i]
            upper.append(mid + num_std * std)
            lower.append(mid - num_std * std)

    return upper, middle, lower


def calculate_volume_ma(volumes: list[int], window: int = 5) -> list[float]:
    """Calculate Volume Moving Average.

    Args:
        volumes: List of trading volumes
        window: Window size for MA

    Returns:
        List of volume MA values
    """
    if len(volumes) < window:
        return [float("nan")] * len(volumes)

    result = []
    for i in range(len(volumes)):
        if i < window - 1:
            result.append(float("nan"))
        else:
            result.append(sum(volumes[i - window + 1 : i + 1]) / window)
    return result
