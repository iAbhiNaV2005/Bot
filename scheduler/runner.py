"""
scheduler/runner.py — Main loop, timing, and orchestration.

This is the brain of the bot. It:
1. Checks for 5m candle closes every 60 seconds
2. Fetches data, runs indicators, runs SMC, checks signals
3. Processes coins in parallel via ThreadPoolExecutor
4. Manages periodic tasks (15m/1h/4h/24h refreshes)
5. Tracks errors and sends alerts when thresholds are reached
"""

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from analysis.indicators import calculate_all_indicators
from analysis.market_data import check_funding_rate_long, check_oi_increasing
from analysis.smc import run_smc_analysis
from config import settings
from data.coin_selector import coin_selector
from data.fetcher import fetcher
from data.storage import data_store
from database.signal_log import signal_logger
from delivery.telegram_bot import telegram
from signals.long_signal import Signal, check_long
from signals.short_signal import check_short
from utils.error_handler import error_counter
from utils.logger import get_logger
from utils.time_utils import (
    get_last_closed_candle_open_time,
    is_new_candle_closed,
    utc_now,
)

logger = get_logger(__name__)


class SignalRunner:
    """
    Main orchestration engine for the signal bot.

    Manages the lifecycle of data fetching, analysis, and signal delivery
    on a candle-close schedule.
    """

    def __init__(self) -> None:
        """Initialize the runner with tracking state."""
        self._running: bool = False
        self._last_5m_processed: int = 0
        self._last_15m_processed: int = 0
        self._last_1h_processed: int = 0
        self._last_4h_processed: int = 0
        self._last_daily_refresh: float = 0.0
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._signal_queue: list[tuple[Signal, Optional[Any]]] = []

    async def startup(self) -> None:
        """
        Run the startup sequence:
        1. Load or refresh coin list
        2. Fetch initial OHLCV data for all timeframes
        3. Send startup notification to Telegram
        """
        logger.info("=" * 60)
        logger.info("Bot starting up...")
        logger.info("=" * 60)

        # ── Set error alert callback ──
        error_counter.set_alert_callback(telegram.send_error_alert)

        # ── Load/refresh coin list ──
        loaded = coin_selector.load_from_disk()
        if not loaded or coin_selector.needs_refresh():
            await coin_selector.refresh()

        coins = coin_selector.coins
        if not coins:
            logger.error("No coins selected! Cannot start.")
            return

        data_store.sync_coin_list(coins)
        logger.info(f"Tracking {len(coins)} coins")

        # ── Fetch initial OHLCV data ──
        await self._fetch_all_timeframes(coins)

        # ── Send startup notification ──
        try:
            await telegram.send_startup()
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")

        logger.info("Startup complete. Entering main loop.")

    async def _fetch_all_timeframes(self, coins: list[str]) -> None:
        """
        Fetch OHLCV data for all timeframes for all coins.

        Args:
            coins: List of trading pair symbols.
        """
        timeframes = [
            (settings.MACRO_TF, settings.CANDLE_LIMITS[settings.MACRO_TF]),
            (settings.STRUCTURE_TF, settings.CANDLE_LIMITS[settings.STRUCTURE_TF]),
            (settings.SETUP_TF, settings.CANDLE_LIMITS[settings.SETUP_TF]),
            (settings.ENTRY_TF, settings.CANDLE_LIMITS[settings.ENTRY_TF]),
        ]

        for tf, limit in timeframes:
            logger.info(f"Fetching {tf} candles for {len(coins)} coins...")
            data = await fetcher.fetch_ohlcv_batch(coins, tf, limit)
            for symbol, df in data.items():
                data_store.update_ohlcv(symbol, tf, df)

        logger.info("Initial OHLCV fetch complete")

    async def main_loop_tick(self) -> None:
        """
        Single tick of the main loop — called every 60 seconds.

        Checks if a new 5m candle has closed, and if so, runs the
        full analysis pipeline.
        """
        try:
            coins = coin_selector.coins
            if not coins:
                return

            # ── Check 5m candle close ──
            if not is_new_candle_closed(settings.ENTRY_TF, self._last_5m_processed):
                return

            self._last_5m_processed = get_last_closed_candle_open_time(settings.ENTRY_TF)
            logger.info(f"New 5m candle closed. Running analysis cycle...")

            # ── Fetch latest 5m candles ──
            data_5m = await fetcher.fetch_ohlcv_batch(
                coins, settings.ENTRY_TF, settings.CANDLE_LIMITS[settings.ENTRY_TF]
            )
            for symbol, df in data_5m.items():
                data_store.update_ohlcv(symbol, settings.ENTRY_TF, df)

            # ── Check if 15m needs refresh ──
            if is_new_candle_closed(settings.SETUP_TF, self._last_15m_processed):
                self._last_15m_processed = get_last_closed_candle_open_time(settings.SETUP_TF)
                data_15m = await fetcher.fetch_ohlcv_batch(
                    coins, settings.SETUP_TF, settings.CANDLE_LIMITS[settings.SETUP_TF]
                )
                for symbol, df in data_15m.items():
                    data_store.update_ohlcv(symbol, settings.SETUP_TF, df)

            # ── Check if 1H needs refresh ──
            if is_new_candle_closed(settings.STRUCTURE_TF, self._last_1h_processed):
                self._last_1h_processed = get_last_closed_candle_open_time(settings.STRUCTURE_TF)
                data_1h = await fetcher.fetch_ohlcv_batch(
                    coins, settings.STRUCTURE_TF, settings.CANDLE_LIMITS[settings.STRUCTURE_TF]
                )
                for symbol, df in data_1h.items():
                    data_store.update_ohlcv(symbol, settings.STRUCTURE_TF, df)

            # ── Check if 4H needs refresh ──
            if is_new_candle_closed(settings.MACRO_TF, self._last_4h_processed):
                self._last_4h_processed = get_last_closed_candle_open_time(settings.MACRO_TF)
                data_4h = await fetcher.fetch_ohlcv_batch(
                    coins, settings.MACRO_TF, settings.CANDLE_LIMITS[settings.MACRO_TF]
                )
                for symbol, df in data_4h.items():
                    data_store.update_ohlcv(symbol, settings.MACRO_TF, df)

            # ── Fetch market data ──
            funding_rates = await fetcher.fetch_funding_rates_batch(coins)
            for symbol, rate in funding_rates.items():
                data_store.update_funding_rate(symbol, rate)

            oi_data = await fetcher.fetch_oi_batch(coins)
            for symbol, oi in oi_data.items():
                data_store.update_oi_history(symbol, oi)

            # ── Fetch L/S ratios (bonus data) ──
            for symbol in coins:
                try:
                    ratio = await fetcher.fetch_long_short_ratio(symbol)
                    data_store.update_ls_ratio(symbol, ratio)
                except Exception:
                    pass
                await asyncio.sleep(0.05)

            # ── Analyze each coin ──
            self._signal_queue.clear()

            for symbol in coins:
                try:
                    await self._analyze_coin(symbol)
                except Exception as e:
                    logger.error(
                        f"Analysis failed for {symbol}: {e}\n{traceback.format_exc()}"
                    )
                    error_counter.record_error(symbol)

            # ── Send queued signals ──
            for signal, ohlcv_15m in self._signal_queue:
                try:
                    await telegram.send_signal(signal, ohlcv_15m)
                    data_store.record_signal(signal.symbol)

                    signal_logger.log_signal(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry,
                        stop_loss=signal.stop_loss,
                        tp1=signal.tp1,
                        tp2=signal.tp2,
                        tp3=signal.tp3,
                        score=signal.score,
                        max_score=signal.max_score,
                        confidence=signal.confidence,
                        funding_rate=signal.funding_rate,
                        oi_change=signal.oi_change,
                        ls_ratio=signal.ls_ratio,
                    )
                    error_counter.reset(signal.symbol)
                except Exception as e:
                    logger.error(f"Failed to send signal for {signal.symbol}: {e}")

            logger.info(
                f"Cycle complete. Signals sent: {len(self._signal_queue)}. "
                f"Next check in {settings.MAIN_LOOP_INTERVAL_SECONDS}s."
            )

            # ── Update active signal statuses against current prices ──
            price_map: dict[str, float] = {}
            for symbol in coins:
                df = data_store.get_ohlcv(symbol, settings.ENTRY_TF)
                if df is not None and len(df) > 0:
                    price_map[symbol] = float(df["close"].iloc[-1])

            if price_map:
                signal_logger.check_and_update_statuses(price_map)
                signal_logger.expire_old_signals(max_age_hours=24)

        except Exception as e:
            logger.error(f"Main loop error: {e}\n{traceback.format_exc()}")
            error_counter.record_error("global")

    async def _analyze_coin(self, symbol: str) -> None:
        """
        Run the full analysis pipeline for a single coin.

        Args:
            symbol: Trading pair to analyze.
        """
        # ── Cooldown check ──
        if data_store.is_in_cooldown(symbol):
            return

        # ── Gather OHLCV ──
        ohlcv: dict[str, Any] = {}
        for tf in [settings.MACRO_TF, settings.STRUCTURE_TF, settings.SETUP_TF, settings.ENTRY_TF]:
            df = data_store.get_ohlcv(symbol, tf)
            if df is not None and len(df) > 0:
                ohlcv[tf] = df

        # Minimum data check
        if settings.MACRO_TF not in ohlcv or settings.ENTRY_TF not in ohlcv:
            return

        # ── Calculate indicators ──
        indicators = calculate_all_indicators(ohlcv)
        data_store.update_indicators(symbol, indicators)

        # ── Run SMC analysis ──
        atr_1h = indicators.get("1h_atr") if "1h_atr" in indicators else None
        atr_15m = indicators.get("15m_atr") if "15m_atr" in indicators else None
        smc = run_smc_analysis(ohlcv, atr_1h, atr_15m)

        # Store SMC results
        for key, value in smc.items():
            if isinstance(value, list):
                data_store.update_smc(symbol, key, value)

        # ── Build market data dict ──
        coin_data = data_store.get_coin(symbol)
        market: dict[str, Any] = {}
        if coin_data:
            market["funding_rate"] = coin_data.funding_rate
            oi_increasing, oi_change = check_oi_increasing(coin_data.oi_history)
            market["oi_increasing"] = oi_increasing
            market["oi_change"] = oi_change
            market["ls_ratio"] = coin_data.ls_ratio

        # ── Get current price ──
        df_5m = ohlcv.get(settings.ENTRY_TF)
        if df_5m is None or len(df_5m) == 0:
            return
        current_price = float(df_5m["close"].iloc[-1])

        # ── Check LONG ──
        long_signal = check_long(symbol, indicators, smc, market, current_price)
        if long_signal:
            ohlcv_15m = ohlcv.get(settings.SETUP_TF)
            self._signal_queue.append((long_signal, ohlcv_15m))
            return  # Don't check short if long fires

        # ── Check SHORT ──
        short_signal = check_short(symbol, indicators, smc, market, current_price)
        if short_signal:
            ohlcv_15m = ohlcv.get(settings.SETUP_TF)
            self._signal_queue.append((short_signal, ohlcv_15m))

    async def daily_refresh(self) -> None:
        """
        Daily tasks: refresh coin list and send daily report.
        Called once every 24 hours at midnight UTC.
        """
        logger.info("Running daily refresh...")

        # Refresh coin list
        await coin_selector.refresh()
        data_store.sync_coin_list(coin_selector.coins)

        # Fetch fresh data for any new coins
        await self._fetch_all_timeframes(coin_selector.coins)

        # Send daily report
        try:
            summary = signal_logger.get_daily_summary()
            await telegram.send_daily_report(
                total=summary.get("total_signals", 0),
                longs=summary.get("long_signals", 0),
                shorts=summary.get("short_signals", 0),
                high_conf=summary.get("high_confidence", 0),
                avg_score=summary.get("avg_score", 0) or 0,
                coins=len(coin_selector.coins),
            )
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown — close connections."""
        logger.info("Shutting down...")
        self._running = False
        await fetcher.close()
        self._executor.shutdown(wait=False)
        logger.info("Shutdown complete")


# Module-level singleton
runner = SignalRunner()
