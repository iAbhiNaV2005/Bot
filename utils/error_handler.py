"""
utils/error_handler.py — Centralized error handling with retry logic.

Provides:
- retry decorator for transient failures (API timeouts, rate limits)
- error counter to trigger Telegram alerts after N consecutive failures
- safe execution wrapper that catches and logs without crashing
"""

import asyncio
import functools
import traceback
from typing import Any, Callable, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorCounter:
    """
    Track consecutive errors per context key (e.g., per coin symbol).
    Triggers an alert callback when threshold is reached.
    """

    def __init__(self, threshold: int = settings.MAX_API_ERRORS_BEFORE_ALERT) -> None:
        self._counts: dict[str, int] = {}
        self._threshold = threshold
        self._alert_callback: Optional[Callable] = None

    def set_alert_callback(self, callback: Callable) -> None:
        """
        Set the async function to call when error threshold is reached.

        Args:
            callback: Async callable accepting (key: str, count: int).
        """
        self._alert_callback = callback

    def record_error(self, key: str) -> int:
        """
        Record an error for the given key. Returns current count.

        Args:
            key: Context identifier (e.g., symbol name or 'global').

        Returns:
            Current consecutive error count for this key.
        """
        self._counts[key] = self._counts.get(key, 0) + 1
        count = self._counts[key]

        if count >= self._threshold and self._alert_callback:
            logger.error(
                f"Error threshold reached for '{key}': {count} consecutive errors"
            )
            # Schedule alert — don't block
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._alert_callback(key, count))
            except RuntimeError:
                # No running loop — log only
                logger.warning(f"Cannot send alert (no event loop) for '{key}'")

        return count

    def reset(self, key: str) -> None:
        """Reset error count for a key after a successful operation."""
        self._counts.pop(key, None)

    def reset_all(self) -> None:
        """Reset all error counters."""
        self._counts.clear()


# Global error counter instance
error_counter = ErrorCounter()


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 2.0,
    exceptions: tuple = (Exception,),
    on_rate_limit_delay: float = 60.0,
) -> Callable:
    """
    Decorator that retries a function on failure.

    Args:
        max_attempts: Maximum number of retry attempts.
        delay_seconds: Delay between retries in seconds.
        exceptions: Tuple of exception types to catch and retry.
        on_rate_limit_delay: Extra delay when a rate limit (HTTP 429) is detected.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    is_rate_limit = "429" in error_msg or "rate" in error_msg

                    if attempt < max_attempts:
                        wait = on_rate_limit_delay if is_rate_limit else delay_seconds
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    is_rate_limit = "429" in error_msg or "rate" in error_msg

                    if attempt < max_attempts:
                        wait = on_rate_limit_delay if is_rate_limit else delay_seconds
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def safe_execute(func: Callable, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    Execute a function safely — catch all exceptions, log them, never crash.

    Args:
        func: Function to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Function result or None if an exception occurred.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(
            f"Safe execution failed for {func.__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        return None


async def async_safe_execute(func: Callable, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    Execute an async function safely — catch all exceptions, log them, never crash.

    Args:
        func: Async function to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Function result or None if an exception occurred.
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(
            f"Safe execution failed for {func.__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        return None
