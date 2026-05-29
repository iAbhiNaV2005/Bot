"""
utils/time_utils.py — Candle close timing logic.

CRITICAL: Never run analysis on an open candle. Always wait for confirmed close.

This module provides functions to determine candle boundaries and check
whether a new candle has closed since the last analysis run.
"""

import time
from datetime import datetime, timezone

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def get_timeframe_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '4h').

    Returns:
        Number of seconds in the timeframe.

    Raises:
        ValueError: If timeframe is not recognized.
    """
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return TIMEFRAME_SECONDS[timeframe]


def get_current_candle_open_time(timeframe: str) -> int:
    """
    Get the open timestamp (unix seconds) of the currently active candle.

    This floors the current time to the nearest candle boundary.

    Args:
        timeframe: Timeframe string (e.g., '5m').

    Returns:
        Unix timestamp in seconds of the current candle's open.
    """
    tf_seconds = get_timeframe_seconds(timeframe)
    now = int(time.time())
    return (now // tf_seconds) * tf_seconds


def get_last_closed_candle_open_time(timeframe: str) -> int:
    """
    Get the open timestamp of the most recently CLOSED candle.

    Args:
        timeframe: Timeframe string (e.g., '5m').

    Returns:
        Unix timestamp in seconds of the last closed candle's open.
    """
    current_open = get_current_candle_open_time(timeframe)
    tf_seconds = get_timeframe_seconds(timeframe)
    return current_open - tf_seconds


def is_new_candle_closed(timeframe: str, last_processed_time: int) -> bool:
    """
    Check if a new candle has closed since the last processed time.

    Args:
        timeframe: Timeframe string (e.g., '5m').
        last_processed_time: Unix timestamp of the last candle we processed.

    Returns:
        True if a new candle has closed and is ready for analysis.
    """
    last_closed = get_last_closed_candle_open_time(timeframe)
    return last_closed > last_processed_time


def seconds_until_next_close(timeframe: str) -> float:
    """
    Calculate seconds remaining until the next candle close.

    Args:
        timeframe: Timeframe string (e.g., '5m').

    Returns:
        Seconds until the next candle closes.
    """
    tf_seconds = get_timeframe_seconds(timeframe)
    now = time.time()
    current_open = (int(now) // tf_seconds) * tf_seconds
    next_close = current_open + tf_seconds
    return max(0.0, next_close - now)


def wait_for_candle_close(timeframe: str) -> int:
    """
    Block until the current candle closes, plus a buffer.

    Uses the buffer from settings to account for data propagation delay.

    Args:
        timeframe: Timeframe string (e.g., '5m').

    Returns:
        Unix timestamp of the candle that just closed.
    """
    remaining = seconds_until_next_close(timeframe)
    total_wait = remaining + settings.CANDLE_CLOSE_BUFFER_SECONDS

    if total_wait > 0:
        logger.debug(
            f"Waiting {total_wait:.1f}s for {timeframe} candle close "
            f"({remaining:.1f}s + {settings.CANDLE_CLOSE_BUFFER_SECONDS}s buffer)"
        )
        time.sleep(total_wait)

    return get_last_closed_candle_open_time(timeframe)


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def timestamp_to_utc(ts: int | float) -> datetime:
    """
    Convert a unix timestamp to a UTC datetime.

    Args:
        ts: Unix timestamp in seconds.

    Returns:
        UTC datetime object.
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_trading_session() -> dict[str, any]:
    """
    Determine the current trading session based on UTC hour.

    Returns a dict with session name and score modifier:
      - Asian    (00:00–07:59 UTC): -2 (low quality)
      - London   (08:00–12:59 UTC): +2 (good quality)
      - Overlap  (13:00–16:59 UTC): +3 (London-NY overlap, best quality)
      - NY       (17:00–20:59 UTC): +1 (decent quality)
      - Late     (21:00–23:59 UTC): -1 (dead zone)

    Returns:
        Dict with 'session_name' (str) and 'session_score' (int).
    """
    hour = utc_now().hour

    if settings.SESSION_ASIAN_START_UTC <= hour <= settings.SESSION_ASIAN_END_UTC:
        return {"session_name": "Asian", "session_score": settings.SESSION_ASIAN_SCORE}
    elif settings.SESSION_LONDON_START_UTC <= hour <= settings.SESSION_LONDON_END_UTC:
        return {"session_name": "London", "session_score": settings.SESSION_LONDON_SCORE}
    elif settings.SESSION_OVERLAP_START_UTC <= hour <= settings.SESSION_OVERLAP_END_UTC:
        return {"session_name": "Overlap", "session_score": settings.SESSION_OVERLAP_SCORE}
    elif settings.SESSION_NY_START_UTC <= hour <= settings.SESSION_NY_END_UTC:
        return {"session_name": "NY", "session_score": settings.SESSION_NY_SCORE}
    else:
        return {"session_name": "Late", "session_score": settings.SESSION_LATE_SCORE}


def is_midnight_utc() -> bool:
    """Check if the current UTC hour is 0 (midnight)."""
    return utc_now().hour == 0
