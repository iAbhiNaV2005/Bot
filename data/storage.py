"""
data/storage.py — In-memory data store for live signal processing.

Stores OHLCV DataFrames, indicator values, SMC structures, and cooldown
timestamps for all tracked coins. Designed for speed — no database reads
during the hot path.

Also manages the rolling window of candle data (appending new candles,
trimming old ones).
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CoinData:
    """
    All in-memory data for a single coin across all timeframes.

    Attributes:
        ohlcv: Dict of timeframe -> OHLCV DataFrame.
        indicators: Dict of indicator name -> computed values.
        smc: Dict of SMC structure type -> list of detected structures.
        last_signal_time: Timestamp of the last signal sent (for cooldown).
        funding_rate: Current funding rate.
        oi_history: Open interest history data points.
        ls_ratio: Current long/short account ratio.
    """

    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    smc: dict[str, list[Any]] = field(default_factory=dict)
    last_signal_time: float = 0.0
    funding_rate: Optional[float] = None
    oi_history: list[dict[str, Any]] = field(default_factory=list)
    ls_ratio: Optional[float] = None


class DataStore:
    """
    Central in-memory storage for all coin data.

    Provides fast access to OHLCV, indicators, and SMC structures
    without any I/O overhead during the analysis pipeline.
    """

    def __init__(self) -> None:
        """Initialize empty data store."""
        self._data: dict[str, CoinData] = {}
        self._last_processed: dict[str, int] = {}  # timeframe -> unix timestamp

    def init_coin(self, symbol: str) -> None:
        """
        Initialize storage for a new coin.

        Args:
            symbol: Trading pair (e.g., 'ETH/USDT:USDT').
        """
        if symbol not in self._data:
            self._data[symbol] = CoinData()

    def remove_coin(self, symbol: str) -> None:
        """
        Remove a coin from storage.

        Args:
            symbol: Trading pair to remove.
        """
        self._data.pop(symbol, None)

    def get_coin(self, symbol: str) -> Optional[CoinData]:
        """
        Get all data for a coin.

        Args:
            symbol: Trading pair.

        Returns:
            CoinData instance or None if not tracked.
        """
        return self._data.get(symbol)

    @property
    def symbols(self) -> list[str]:
        """Get all tracked symbols."""
        return list(self._data.keys())

    # ─── OHLCV ────────────────────────────────────────────────────────

    def update_ohlcv(
        self, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> None:
        """
        Set or replace the OHLCV data for a coin/timeframe.

        Args:
            symbol: Trading pair.
            timeframe: Candle timeframe.
            df: OHLCV DataFrame (timestamp-indexed).
        """
        self.init_coin(symbol)
        self._data[symbol].ohlcv[timeframe] = df

    def append_candles(
        self, symbol: str, timeframe: str, new_df: pd.DataFrame, max_candles: int = 200
    ) -> None:
        """
        Append new candles to existing data, keeping only the latest N.

        Deduplicates by index (timestamp) and trims to max_candles.

        Args:
            symbol: Trading pair.
            timeframe: Candle timeframe.
            new_df: New candle data to append.
            max_candles: Maximum candles to retain.
        """
        self.init_coin(symbol)
        existing = self._data[symbol].ohlcv.get(timeframe)

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self._data[symbol].ohlcv[timeframe] = combined.tail(max_candles)
        else:
            self._data[symbol].ohlcv[timeframe] = new_df.tail(max_candles)

    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a coin/timeframe.

        Args:
            symbol: Trading pair.
            timeframe: Candle timeframe.

        Returns:
            OHLCV DataFrame or None.
        """
        coin = self._data.get(symbol)
        if coin is None:
            return None
        return coin.ohlcv.get(timeframe)

    # ─── Indicators ──────────────────────────────────────────────────

    def update_indicators(self, symbol: str, indicators: dict[str, Any]) -> None:
        """
        Update computed indicator values for a coin.

        Args:
            symbol: Trading pair.
            indicators: Dict of indicator name -> value/series.
        """
        self.init_coin(symbol)
        self._data[symbol].indicators.update(indicators)

    def get_indicators(self, symbol: str) -> dict[str, Any]:
        """
        Get all indicator values for a coin.

        Args:
            symbol: Trading pair.

        Returns:
            Dict of indicator values (empty dict if none computed).
        """
        coin = self._data.get(symbol)
        return coin.indicators if coin else {}

    # ─── SMC Structures ──────────────────────────────────────────────

    def update_smc(self, symbol: str, smc_type: str, structures: list[Any]) -> None:
        """
        Update detected SMC structures for a coin.

        Args:
            symbol: Trading pair.
            smc_type: Type key (e.g., 'swing_highs', 'order_blocks_bull').
            structures: List of detected structure objects.
        """
        self.init_coin(symbol)
        self._data[symbol].smc[smc_type] = structures

    def get_smc(self, symbol: str, smc_type: str) -> list[Any]:
        """
        Get SMC structures for a coin.

        Args:
            symbol: Trading pair.
            smc_type: Type key.

        Returns:
            List of structures (empty list if none).
        """
        coin = self._data.get(symbol)
        if coin is None:
            return []
        return coin.smc.get(smc_type, [])

    # ─── Market Data ─────────────────────────────────────────────────

    def update_funding_rate(self, symbol: str, rate: Optional[float]) -> None:
        """Update funding rate for a coin."""
        self.init_coin(symbol)
        self._data[symbol].funding_rate = rate

    def update_oi_history(self, symbol: str, oi_data: list[dict[str, Any]]) -> None:
        """Update open interest history for a coin."""
        self.init_coin(symbol)
        self._data[symbol].oi_history = oi_data

    def update_ls_ratio(self, symbol: str, ratio: Optional[float]) -> None:
        """Update long/short ratio for a coin."""
        self.init_coin(symbol)
        self._data[symbol].ls_ratio = ratio

    # ─── Cooldown ────────────────────────────────────────────────────

    def record_signal(self, symbol: str) -> None:
        """
        Record that a signal was sent for this coin (for cooldown enforcement).

        Args:
            symbol: Trading pair.
        """
        self.init_coin(symbol)
        self._data[symbol].last_signal_time = time.time()

    def is_in_cooldown(self, symbol: str) -> bool:
        """
        Check if a coin is in cooldown (signal sent too recently).

        Args:
            symbol: Trading pair.

        Returns:
            True if a signal was sent within SIGNAL_COOLDOWN_HOURS.
        """
        coin = self._data.get(symbol)
        if coin is None or coin.last_signal_time == 0.0:
            return False

        elapsed_hours = (time.time() - coin.last_signal_time) / 3600
        return elapsed_hours < settings.SIGNAL_COOLDOWN_HOURS

    # ─── Processing Tracker ──────────────────────────────────────────

    def set_last_processed(self, timeframe: str, timestamp: int) -> None:
        """Record the last processed candle timestamp for a timeframe."""
        self._last_processed[timeframe] = timestamp

    def get_last_processed(self, timeframe: str) -> int:
        """Get the last processed candle timestamp for a timeframe."""
        return self._last_processed.get(timeframe, 0)

    # ─── Bulk Operations ─────────────────────────────────────────────

    def sync_coin_list(self, new_symbols: list[str]) -> None:
        """
        Sync stored coins with a new list. Removes coins no longer tracked,
        initializes new ones.

        Args:
            new_symbols: Updated list of tracked symbols.
        """
        current = set(self._data.keys())
        incoming = set(new_symbols)

        # Remove delisted coins
        for symbol in current - incoming:
            self.remove_coin(symbol)
            logger.info(f"Removed {symbol} from data store")

        # Initialize new coins
        for symbol in incoming - current:
            self.init_coin(symbol)
            logger.info(f"Added {symbol} to data store")

    def clear(self) -> None:
        """Clear all stored data."""
        self._data.clear()
        self._last_processed.clear()


# Module-level singleton
data_store = DataStore()
