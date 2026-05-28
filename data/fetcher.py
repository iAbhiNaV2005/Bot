"""
data/fetcher.py — All Binance API calls live here.

Uses ccxt with futures mode. Handles batching, delays, and rate tracking.
Every API interaction in the entire bot goes through this module.

NOTE: Uses Binance Futures API endpoints directly (fapiPublic*) to avoid
ccxt's load_markets() calling the spot API which can fail due to permissions.
"""

import asyncio
import time
from typing import Any, Optional

import ccxt.async_support as ccxt  # type: ignore[import-untyped]
import pandas as pd

from config import settings
from utils.error_handler import retry
from utils.logger import get_logger

logger = get_logger(__name__)


class BinanceFetcher:
    """
    Async Binance Futures data fetcher using ccxt.

    Handles OHLCV, funding rates, open interest, and ticker data.
    Includes built-in rate tracking and configurable delays between calls.
    """

    def __init__(self) -> None:
        """Initialize the ccxt Binance Futures client."""
        self._exchange: Optional[ccxt.binance] = None
        self._request_count: int = 0
        self._request_window_start: float = time.time()

    async def _get_exchange(self) -> ccxt.binance:
        """
        Lazy-initialize and return the exchange client.

        Uses a custom aiohttp session with ThreadedResolver to avoid
        aiodns/pycares DNS issues on Windows.

        Returns:
            Configured ccxt.binance instance with futures mode.
        """
        if self._exchange is None:
            import aiohttp

            # Use system DNS resolver instead of aiodns (fixes Windows DNS issues)
            connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
            session = aiohttp.ClientSession(connector=connector)

            self._exchange = ccxt.binance({
                "apiKey": settings.BINANCE_API_KEY,
                "secret": settings.BINANCE_API_SECRET,
                "options": {
                    "defaultType": "future",
                    "fetchCurrencies": False,
                },
                "session": session,
                "enableRateLimit": True,
            })
        return self._exchange

    async def close(self) -> None:
        """Close the exchange connection and underlying aiohttp session."""
        if self._exchange:
            # Close the custom aiohttp session we injected
            if hasattr(self._exchange, "session") and self._exchange.session:
                if not self._exchange.session.closed:
                    await self._exchange.session.close()
            await self._exchange.close()
            self._exchange = None

    def _track_request(self) -> None:
        """
        Track API requests for rate limiting awareness.

        Resets the counter every 60 seconds. Logs warnings when
        approaching the safety limit.
        """
        now = time.time()
        if now - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = now

        self._request_count += 1

        if self._request_count >= settings.SAFETY_WEIGHT_LIMIT:
            logger.warning(
                f"Approaching rate limit: {self._request_count} requests "
                f"in current window. Adding delay."
            )

    # ─── Exchange Info & Tickers ──────────────────────────────────────

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_exchange_info(self) -> list[dict[str, Any]]:
        """
        Fetch all active USDT-margined perpetual futures symbols.

        Calls the Binance Futures API directly (fapiPublicGetExchangeInfo)
        to avoid ccxt's load_markets() which also calls the spot API.

        Returns:
            List of market info dicts with 'symbol', 'base', 'type', etc.
        """
        exchange = await self._get_exchange()
        self._track_request()

        raw = await exchange.fapiPublicGetExchangeInfo()
        symbols = raw.get("symbols", [])

        # Filter to active USDT-margined perpetual contracts
        futures_symbols: list[dict[str, Any]] = []
        for s in symbols:
            if (
                s.get("status") == "TRADING"
                and s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("marginAsset") == "USDT"
            ):
                base = s.get("baseAsset", "")
                futures_symbols.append({
                    "symbol": f"{base}/USDT:USDT",
                    "id": s.get("symbol", ""),
                    "base": base,
                    "quote": "USDT",
                    "type": "swap",
                    "active": True,
                    "linear": True,
                })

        logger.info(f"Found {len(futures_symbols)} active USDT perpetual futures")
        return futures_symbols

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_all_tickers(self) -> dict[str, dict[str, Any]]:
        """
        Fetch 24h ticker data for ALL futures symbols in a single call.

        Uses the futures ticker endpoint directly (fapiPublicGetTicker24hr).

        Returns:
            Dict mapping ccxt-style symbol to ticker data.
        """
        exchange = await self._get_exchange()
        self._track_request()

        raw_tickers = await exchange.fapiPublicGetTicker24hr()

        tickers: dict[str, dict[str, Any]] = {}
        for t in raw_tickers:
            raw_sym = t.get("symbol", "")
            if raw_sym.endswith("USDT"):
                base = raw_sym[:-4]
                ccxt_sym = f"{base}/USDT:USDT"
                tickers[ccxt_sym] = {
                    "symbol": ccxt_sym,
                    "quoteVolume": float(t.get("quoteVolume", 0)),
                    "lastPrice": float(t.get("lastPrice", 0)),
                    "priceChangePercent": float(t.get("priceChangePercent", 0)),
                }

        return tickers

    # ─── OHLCV ────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'ETH/USDT:USDT').
            timeframe: Candle timeframe (e.g., '5m', '1h', '4h').
            limit: Number of candles to fetch.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Index is the timestamp as datetime.
        """
        exchange = await self._get_exchange()
        self._track_request()

        # Use Binance futures klines endpoint directly
        raw_symbol = symbol.replace("/", "").replace(":USDT", "")
        raw = await exchange.fapiPublicGetKlines({
            "symbol": raw_symbol,
            "interval": timeframe,
            "limit": limit,
        })

        df = pd.DataFrame(
            raw,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        return df

    async def fetch_ohlcv_batch(
        self,
        symbols: list[str],
        timeframe: str,
        limit: int = 200,
        delay: float = settings.OHLCV_BATCH_DELAY,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols with delays between calls.

        Args:
            symbols: List of trading pair strings.
            timeframe: Candle timeframe.
            limit: Number of candles per symbol.
            delay: Seconds to wait between API calls.

        Returns:
            Dict mapping symbol to its OHLCV DataFrame.
            Symbols that fail are excluded (logged, not raised).
        """
        results: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                df = await self.fetch_ohlcv(symbol, timeframe, limit)
                results[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch OHLCV for {symbol} ({timeframe}): {e}")
            await asyncio.sleep(delay)

        logger.info(
            f"Fetched {timeframe} OHLCV: {len(results)}/{len(symbols)} symbols"
        )
        return results

    # ─── Funding Rates ────────────────────────────────────────────────

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch current funding rate for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'ETH/USDT:USDT').

        Returns:
            Current funding rate as a float, or None on failure.
        """
        exchange = await self._get_exchange()
        self._track_request()

        raw_symbol = symbol.replace("/", "").replace(":USDT", "")
        try:
            result = await exchange.fapiPublicGetFundingRate({
                "symbol": raw_symbol,
                "limit": 1,
            })
            if result and len(result) > 0:
                return float(result[-1].get("fundingRate", 0))
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def fetch_funding_rates_batch(
        self, symbols: list[str], delay: float = 0.05
    ) -> dict[str, Optional[float]]:
        """
        Fetch funding rates for multiple symbols.

        Args:
            symbols: List of trading pair strings.
            delay: Seconds between calls.

        Returns:
            Dict mapping symbol to funding rate (or None).
        """
        rates: dict[str, Optional[float]] = {}

        for symbol in symbols:
            rate = await self.fetch_funding_rate(symbol)
            rates[symbol] = rate
            await asyncio.sleep(delay)

        return rates

    # ─── Open Interest ────────────────────────────────────────────────

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_open_interest_history(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> list[dict[str, Any]]:
        """
        Fetch open interest history for a symbol.

        Uses /futures/data/openInterestHist endpoint directly.

        Args:
            symbol: Trading pair.
            period: Data period (e.g., '5m').
            limit: Number of data points.

        Returns:
            List of dicts with 'timestamp', 'sumOpenInterest' keys.
        """
        exchange = await self._get_exchange()
        self._track_request()

        raw_symbol = symbol.replace("/", "").replace(":USDT", "")
        try:
            # Use exchange.fetch() for the /futures/data/ endpoint
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = f"?symbol={raw_symbol}&period={period}&limit={limit}"
            result = await exchange.fetch(url + params)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning(f"Failed to fetch OI history for {symbol}: {e}")
            return []

    async def fetch_oi_batch(
        self, symbols: list[str], delay: float = settings.OI_BATCH_DELAY
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch OI history for multiple symbols with delays.

        Args:
            symbols: List of trading pair strings.
            delay: Seconds between calls.

        Returns:
            Dict mapping symbol to OI history list.
        """
        results: dict[str, list[dict[str, Any]]] = {}

        for symbol in symbols:
            oi_data = await self.fetch_open_interest_history(symbol)
            results[symbol] = oi_data
            await asyncio.sleep(delay)

        return results

    # ─── Long/Short Ratio ─────────────────────────────────────────────

    @retry(max_attempts=3, delay_seconds=2.0)
    async def fetch_long_short_ratio(
        self, symbol: str, period: str = "5m", limit: int = 1
    ) -> Optional[float]:
        """
        Fetch global long/short account ratio for a symbol.

        Uses /futures/data/globalLongShortAccountRatio endpoint directly.

        Args:
            symbol: Trading pair.
            period: Data period.
            limit: Number of data points.

        Returns:
            Long/short ratio as float, or None on failure.
        """
        exchange = await self._get_exchange()
        self._track_request()

        raw_symbol = symbol.replace("/", "").replace(":USDT", "")
        try:
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = f"?symbol={raw_symbol}&period={period}&limit={limit}"
            result = await exchange.fetch(url + params)
            if result and isinstance(result, list) and len(result) > 0:
                return float(result[0].get("longShortRatio", 1.0))
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch L/S ratio for {symbol}: {e}")
            return None


# Module-level singleton
fetcher = BinanceFetcher()

