"""
analysis/market_data.py — Funding rate, Open Interest, and L/S ratio processing.

Processes raw market data from Binance into actionable filter decisions.
Each function returns both the value and a pass/fail determination
for the signal filter gates.
"""

from typing import Any, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def check_funding_rate_long(rate: Optional[float]) -> tuple[bool, float]:
    """
    Check if funding rate allows a LONG signal.

    Condition: Funding rate must be below +0.05% (not too positive).
    If too positive, longs are overcrowded — skip.

    Args:
        rate: Current funding rate (e.g., 0.0003 = 0.03%).

    Returns:
        Tuple of (passes_filter, rate_value).
        rate_value is 0.0 if rate is None.
    """
    if rate is None:
        # Conservative: skip if we can't get the data
        return False, 0.0
    return rate < settings.FUNDING_RATE_LONG_MAX, rate


def check_funding_rate_short(rate: Optional[float]) -> tuple[bool, float]:
    """
    Check if funding rate allows a SHORT signal.

    Condition: Funding rate must be above -0.05% (not too negative).
    If too negative, shorts are overcrowded — skip.

    Args:
        rate: Current funding rate.

    Returns:
        Tuple of (passes_filter, rate_value).
    """
    if rate is None:
        return False, 0.0
    return rate > settings.FUNDING_RATE_SHORT_MIN, rate


def check_oi_increasing(
    oi_data: list[dict[str, Any]],
    lookback_hours: int = settings.OI_LOOKBACK_HOURS,
) -> tuple[bool, float]:
    """
    Check if Open Interest has increased in the last N hours.

    Money entering the market = more conviction behind the move.

    Args:
        oi_data: OI history data points (from Binance API).
                 Each entry has 'sumOpenInterest' and 'timestamp'.
        lookback_hours: How far back to compare (default: 2 hours).

    Returns:
        Tuple of (is_increasing, percent_change).
        percent_change is 0.0 if insufficient data.
    """
    if not oi_data or len(oi_data) < 2:
        return False, 0.0

    try:
        # Data is sorted oldest -> newest by Binance
        current_oi = float(oi_data[-1].get("sumOpenInterest", 0))
        # Find OI from approximately lookback_hours ago
        # Each data point is usually 5m apart, so ~24 points per 2 hours
        target_idx = max(0, len(oi_data) - (lookback_hours * 12))
        past_oi = float(oi_data[target_idx].get("sumOpenInterest", 0))

        if past_oi <= 0:
            return False, 0.0

        pct_change = ((current_oi - past_oi) / past_oi) * 100
        return pct_change > 0, pct_change
    except (KeyError, ValueError, IndexError) as e:
        logger.warning(f"Error processing OI data: {e}")
        return False, 0.0


def check_ls_ratio_long(ratio: Optional[float]) -> tuple[bool, float]:
    """
    Check L/S ratio for contrarian LONG signal bonus.

    If ratio < 1.0, there are more shorts than longs → squeeze likely.
    This is a scoring bonus, not a mandatory filter.

    Args:
        ratio: Current long/short account ratio (1.0 = equal).

    Returns:
        Tuple of (is_contrarian, ratio_value).
    """
    if ratio is None:
        return False, 1.0
    return ratio < 1.0, ratio


def check_ls_ratio_short(ratio: Optional[float]) -> tuple[bool, float]:
    """
    Check L/S ratio for contrarian SHORT signal bonus.

    If ratio > 1.0, there are more longs than shorts → dump likely.

    Args:
        ratio: Current long/short account ratio.

    Returns:
        Tuple of (is_contrarian, ratio_value).
    """
    if ratio is None:
        return False, 1.0
    return ratio > 1.0, ratio


def is_funding_negative(rate: Optional[float]) -> bool:
    """
    Check if funding rate is negative (shorts paying longs = bullish).

    Used as a scoring bonus for LONG signals.

    Args:
        rate: Current funding rate.

    Returns:
        True if rate is negative.
    """
    if rate is None:
        return False
    return rate < 0


def is_funding_positive(rate: Optional[float]) -> bool:
    """
    Check if funding rate is positive (longs paying shorts = bearish).

    Used as a scoring bonus for SHORT signals.

    Args:
        rate: Current funding rate.

    Returns:
        True if rate is positive.
    """
    if rate is None:
        return False
    return rate > 0
