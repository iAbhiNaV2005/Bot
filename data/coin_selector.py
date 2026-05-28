"""
data/coin_selector.py — Top 50 volume-based coin selection engine.

Fetches all USDT perpetuals from Binance, sorts by 24h quote volume,
filters out stablecoins and coins with insufficient history,
and returns the top N tradeable coins.

Refreshes every 24 hours. Persists the list to disk for restart survival.
"""

import json
import os
import time
from typing import Any

from config import settings
from data.fetcher import fetcher
from utils.logger import get_logger

logger = get_logger(__name__)


class CoinSelector:
    """
    Selects and maintains the top N coins by 24h USDT futures volume.

    The selected coins are persisted to a JSON file so the list
    survives bot restarts without requiring an API call.
    """

    def __init__(self) -> None:
        """Initialize the coin selector with an empty list."""
        self._coins: list[str] = []
        self._last_refresh: float = 0.0
        self._coin_file: str = settings.COIN_LIST_FILE

    @property
    def coins(self) -> list[str]:
        """
        Get the current list of selected coins.

        Returns:
            List of ccxt-formatted symbol strings (e.g., 'ETH/USDT:USDT').
        """
        return self._coins

    def _is_stablecoin(self, symbol: str) -> bool:
        """
        Check if a symbol is a stablecoin pair to exclude.

        Args:
            symbol: Raw Binance symbol (e.g., 'USDCUSDT').

        Returns:
            True if the symbol should be excluded.
        """
        raw = symbol.replace("/", "").replace(":USDT", "")
        return raw in settings.STABLECOIN_SYMBOLS

    async def refresh(self) -> list[str]:
        """
        Refresh the top coin list from Binance.

        Steps:
        1. Fetch all active USDT perpetual futures
        2. Fetch 24h ticker for all symbols
        3. Sort by 24h quote volume descending
        4. Take top COIN_FETCH_POOL (60) symbols
        5. Remove stablecoins
        6. Return top TOP_N_COINS (50)

        Returns:
            List of selected symbol strings in ccxt format.
        """
        logger.info("Refreshing top coin list by 24h volume...")

        # Step 1: Get all futures markets
        markets = await fetcher.fetch_exchange_info()

        # Step 2: Get 24h tickers
        tickers = await fetcher.fetch_all_tickers()

        # Step 3: Build volume-sorted list
        volume_list: list[dict[str, Any]] = []
        for market in markets:
            symbol = market["symbol"]
            if symbol in tickers:
                ticker = tickers[symbol]
                quote_volume = float(ticker.get("quoteVolume", 0) or 0)
                volume_list.append({
                    "symbol": symbol,
                    "quoteVolume": quote_volume,
                })

        volume_list.sort(key=lambda x: x["quoteVolume"], reverse=True)

        # Step 4: Take top pool size
        candidates = volume_list[:settings.COIN_FETCH_POOL]

        # Step 5: Filter stablecoins
        filtered = [
            c["symbol"] for c in candidates
            if not self._is_stablecoin(c["symbol"])
        ]

        # Step 6: Take top N
        self._coins = filtered[:settings.TOP_N_COINS]
        self._last_refresh = time.time()

        # Persist to disk
        self._save_to_disk()

        logger.info(
            f"Selected {len(self._coins)} coins. "
            f"Top 5: {self._coins[:5]}"
        )
        return self._coins

    def needs_refresh(self) -> bool:
        """
        Check if the coin list needs refreshing.

        Returns:
            True if no coins loaded or refresh interval has elapsed.
        """
        if not self._coins:
            return True
        elapsed_hours = (time.time() - self._last_refresh) / 3600
        return elapsed_hours >= settings.COIN_REFRESH_INTERVAL_HOURS

    def _save_to_disk(self) -> None:
        """Save current coin list to JSON file for restart persistence."""
        os.makedirs(os.path.dirname(self._coin_file), exist_ok=True)
        data = {
            "coins": self._coins,
            "last_refresh": self._last_refresh,
            "count": len(self._coins),
        }
        with open(self._coin_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Coin list saved to {self._coin_file}")

    def load_from_disk(self) -> bool:
        """
        Load coin list from disk if available.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if not os.path.exists(self._coin_file):
            logger.info("No saved coin list found on disk.")
            return False

        try:
            with open(self._coin_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._coins = data.get("coins", [])
            self._last_refresh = data.get("last_refresh", 0.0)
            logger.info(
                f"Loaded {len(self._coins)} coins from disk "
                f"(last refresh: {self._last_refresh})"
            )
            return bool(self._coins)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load coin list from disk: {e}")
            return False


# Module-level singleton
coin_selector = CoinSelector()
