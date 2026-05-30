"""
scheduler/runner.py — Main loop, timing, and orchestration (v3).

This is the brain of the bot. It:
1. Checks BTC volatility blackout before any analysis
2. Runs BTC structure analysis for correlation scoring
3. Gets current trading session for session scoring
4. Fetches data, runs indicators, runs SMC, checks signals
5. Manages periodic tasks (15m/1h/4h/24h refreshes)
6. Tracks signal outcomes and failure states
7. Sends alerts when thresholds are reached
8. Fetches tickers every cycle for live volume tracking (v3)
9. Runs hourly coin rank comparison and queues swaps (v3)

v2: Added volatility blackout, BTC correlation, session scoring,
    OB touch updates, failure state checks, outcome tracking.
v3: Hybrid volume refresh — ticker every 5m, hourly rank check,
    live coin swap without restart.
"""

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from analysis.indicators import calculate_all_indicators, calc_atr, calc_ema
from analysis.market_data import check_oi_increasing
from analysis.smc import (
    Direction, detect_bos, detect_swing_points, run_smc_analysis,
)
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
    get_trading_session,
    is_new_candle_closed,
    utc_now,
)

logger = get_logger(__name__)


class SignalRunner:
    """
    Main orchestration engine for the signal bot (v2).

    Manages the lifecycle of data fetching, analysis, and signal delivery
    on a candle-close schedule. Includes volatility blackout, BTC correlation,
    session filtering, and outcome tracking.
    """

    def __init__(self) -> None:
        """Initialize the runner with tracking state."""
        self._running: bool = False
        self._last_5m_processed: int = 0
        self._last_15m_processed: int = 0
        self._last_1h_processed: int = 0
        self._last_4h_processed: int = 0
        self._last_daily_refresh: float = 0.0
        self._last_failure_check: float = 0.0
        self._last_outcome_check: float = 0.0
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._signal_queue: list[tuple[Signal, Optional[Any]]] = []
        # Detailed per-cycle analysis results for logging
        self._cycle_results: list[dict[str, Any]] = []

        # v2: Volatility blackout state
        self._signals_paused: bool = False
        self._signals_paused_until: Optional[datetime] = None

        # v3: Hybrid volume refresh state
        self._last_ticker_snapshot: dict[str, float] = {}  # symbol -> 24h quote volume
        self._list_update_pending: bool = False
        self._pending_add_coins: list[str] = []
        self._pending_remove_coins: list[str] = []
        self._last_hourly_rank_check: float = 0.0  # unix timestamp
        self._last_ticker_fetch: dict[str, Any] = {}  # full ticker cache

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

    # ── BTC Volatility Blackout (v2) ──────────────────────────────────────

    async def _check_btc_volatility_blackout(self) -> bool:
        """
        Check if BTC volatility warrants pausing all signals.

        If BTC 1H ATR exceeds 2.5× its rolling average, pause for 60 minutes.

        Returns:
            True if signals should be paused (skip all analysis).
        """
        # Check if currently in blackout
        if self._signals_paused:
            if utc_now() < self._signals_paused_until:
                logger.debug("Signals paused (volatility blackout)")
                return True
            else:
                # Blackout expired
                self._signals_paused = False
                self._signals_paused_until = None
                logger.info("✅ Volatility normalized — signals resumed")
                try:
                    await telegram.send_text("✅ Volatility normalized — signals resumed")
                except Exception:
                    pass
                return False

        # Check BTC ATR
        btc_symbol = settings.BLACKOUT_CHECK_SYMBOL
        df_btc_1h = data_store.get_ohlcv(btc_symbol, settings.STRUCTURE_TF)
        if df_btc_1h is None or len(df_btc_1h) < 20:
            return False

        atr_series = calc_atr(df_btc_1h["high"], df_btc_1h["low"], df_btc_1h["close"])
        if len(atr_series) < 14:
            return False

        current_atr = float(atr_series.iloc[-1])
        # Average of last 14 ATR values
        avg_atr = float(atr_series.iloc[-14:].mean())

        if avg_atr > 0 and current_atr > settings.BLACKOUT_ATR_MULTIPLIER * avg_atr:
            self._signals_paused = True
            self._signals_paused_until = utc_now() + timedelta(
                minutes=settings.BLACKOUT_DURATION_MINUTES
            )
            logger.warning(
                f"⚠️ BTC volatility spike: ATR {current_atr:.2f} > "
                f"{settings.BLACKOUT_ATR_MULTIPLIER}× avg {avg_atr:.2f}"
            )
            try:
                await telegram.send_text(
                    f"⚠️ Extreme volatility detected — all signals paused for "
                    f"{settings.BLACKOUT_DURATION_MINUTES} minutes\n"
                    f"BTC ATR: {current_atr:.2f} (avg: {avg_atr:.2f})"
                )
            except Exception:
                pass
            return True

        return False

    # ── BTC Correlation Analysis (v2) ────────────────────────────────────

    def _analyze_btc_state(self) -> dict[str, Any]:
        """
        Analyze BTC structure for correlation scoring.

        Returns:
            Dict with 'btc_bullish', 'btc_bearish', 'is_btc_symbol' flags.
        """
        btc_symbol = settings.BTC_CORRELATION_SYMBOL
        df_btc_1h = data_store.get_ohlcv(btc_symbol, settings.STRUCTURE_TF)

        result = {"btc_bullish": False, "btc_bearish": False, "is_btc_symbol": False}

        if df_btc_1h is None or len(df_btc_1h) < 50:
            return result

        # BOS on BTC 1H
        sh_btc, sl_btc = detect_swing_points(df_btc_1h, lookback=settings.SWING_LOOKBACK_1H)
        bos_btc = detect_bos(df_btc_1h, sh_btc, sl_btc)

        # EMA 50 on BTC 1H
        ema_50_btc = calc_ema(df_btc_1h["close"], 50)
        btc_price = float(df_btc_1h["close"].iloc[-1])
        btc_ema50 = float(ema_50_btc.iloc[-1])

        if bos_btc and bos_btc.direction == Direction.BEARISH and btc_price < btc_ema50:
            result["btc_bearish"] = True
        elif bos_btc and bos_btc.direction == Direction.BULLISH and btc_price > btc_ema50:
            result["btc_bullish"] = True

        return result

    # ── Main Loop ────────────────────────────────────────────────────────

    async def main_loop_tick(self) -> None:
        """
        Single tick of the main loop — called every 60 seconds.

        v2 pipeline order (from spec Part 11):
        1. Check blackout state
        2. Check BTC volatility
        3. BTC structure analysis
        4. Get session
        5. Fetch candles
        6. Fetch market data
        7. Analyze each coin
        8. Send signals
        9. Periodic tasks (failure check, outcome tracking)
        """
        try:
            coins = coin_selector.coins
            if not coins:
                return

            # ── Step 1-2: Volatility blackout check ──
            if await self._check_btc_volatility_blackout():
                return

            # ── Check 5m candle close ──
            if not is_new_candle_closed(settings.ENTRY_TF, self._last_5m_processed):
                return

            self._last_5m_processed = get_last_closed_candle_open_time(settings.ENTRY_TF)
            logger.info(f"New 5m candle closed. Running analysis cycle...")

            # ── Step 3: BTC structure analysis ──
            btc_state = self._analyze_btc_state()

            # ── Step 4: Get current session ──
            session_info = get_trading_session()

            # ── Step 5: Fetch latest candles ──
            data_5m = await fetcher.fetch_ohlcv_batch(
                coins, settings.ENTRY_TF, settings.CANDLE_LIMITS[settings.ENTRY_TF]
            )
            for symbol, df in data_5m.items():
                data_store.update_ohlcv(symbol, settings.ENTRY_TF, df)

            # Check if 15m needs refresh
            if is_new_candle_closed(settings.SETUP_TF, self._last_15m_processed):
                self._last_15m_processed = get_last_closed_candle_open_time(settings.SETUP_TF)
                data_15m = await fetcher.fetch_ohlcv_batch(
                    coins, settings.SETUP_TF, settings.CANDLE_LIMITS[settings.SETUP_TF]
                )
                for symbol, df in data_15m.items():
                    data_store.update_ohlcv(symbol, settings.SETUP_TF, df)

            # Check if 1H needs refresh
            if is_new_candle_closed(settings.STRUCTURE_TF, self._last_1h_processed):
                self._last_1h_processed = get_last_closed_candle_open_time(settings.STRUCTURE_TF)
                data_1h = await fetcher.fetch_ohlcv_batch(
                    coins, settings.STRUCTURE_TF, settings.CANDLE_LIMITS[settings.STRUCTURE_TF]
                )
                for symbol, df in data_1h.items():
                    data_store.update_ohlcv(symbol, settings.STRUCTURE_TF, df)

            # Check if 4H needs refresh
            if is_new_candle_closed(settings.MACRO_TF, self._last_4h_processed):
                self._last_4h_processed = get_last_closed_candle_open_time(settings.MACRO_TF)
                data_4h = await fetcher.fetch_ohlcv_batch(
                    coins, settings.MACRO_TF, settings.CANDLE_LIMITS[settings.MACRO_TF]
                )
                for symbol, df in data_4h.items():
                    data_store.update_ohlcv(symbol, settings.MACRO_TF, df)

            # ── Step 5b: Ticker fetch every cycle (v3 — hybrid volume refresh) ──
            try:
                self._last_ticker_fetch = await fetcher.fetch_all_tickers()
                await self._update_volume_changes(coins)
                # Hourly rank comparison (runs only at HH:01 UTC)
                await self._hourly_rank_check()
            except Exception as e:
                logger.warning(f"Ticker fetch failed (non-critical): {e}")

            # Apply any pending coin list swaps atomically (v3)
            if self._list_update_pending:
                await self._apply_pending_coin_swap()
                coins = coin_selector.coins  # refresh local ref after swap

            # ── Step 6: Fetch market data ──
            funding_rates = await fetcher.fetch_funding_rates_batch(coins)
            for symbol, rate in funding_rates.items():
                data_store.update_funding_rate(symbol, rate)

            oi_data = await fetcher.fetch_oi_batch(coins)
            for symbol, oi in oi_data.items():
                data_store.update_oi_history(symbol, oi)

            # Fetch L/S ratios (bonus data)
            for symbol in coins:
                try:
                    ratio = await fetcher.fetch_long_short_ratio(symbol)
                    data_store.update_ls_ratio(symbol, ratio)
                except Exception:
                    pass
                await asyncio.sleep(0.05)

            # ── Step 7: Analyze each coin ──
            self._signal_queue.clear()
            self._cycle_results.clear()

            for symbol in coins:
                try:
                    # Determine if this is a BTC signal
                    btc_state_for_coin = dict(btc_state)
                    btc_state_for_coin["is_btc_symbol"] = (
                        "BTC/USDT" in symbol or "BTCUSDT" in symbol
                    )
                    await self._analyze_coin(symbol, session_info, btc_state_for_coin)
                except Exception as e:
                    logger.error(
                        f"Analysis failed for {symbol}: {e}\n{traceback.format_exc()}"
                    )
                    error_counter.record_error(symbol)
                    self._cycle_results.append({
                        "symbol": symbol, "status": "ERROR",
                        "reason": str(e)[:60], "score": 0,
                    })

            # ── Step 7b: Log cycle summary ──
            self._log_cycle_summary(session_info, btc_state)

            # ── Step 8: Send queued signals ──
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
                f"Session: {session_info.get('session_name', '?')}. "
                f"Next check in {settings.MAIN_LOOP_INTERVAL_SECONDS}s."
            )

            # ── Step 9: Update active signal statuses ──
            price_map: dict[str, float] = {}
            for symbol in coins:
                df = data_store.get_ohlcv(symbol, settings.ENTRY_TF)
                if df is not None and len(df) > 0:
                    price_map[symbol] = float(df["close"].iloc[-1])

            if price_map:
                signal_logger.check_and_update_statuses(price_map)
                signal_logger.expire_old_signals(max_age_hours=24)

            # ── Step 10: Periodic failure check (every 15 min) ──
            import time
            now = time.time()
            if now - self._last_failure_check > (
                settings.COOLDOWN_FAILURE_CHECK_INTERVAL_MINUTES * 60
            ):
                self._last_failure_check = now
                self._check_failure_states(price_map)

        except Exception as e:
            logger.error(f"Main loop error: {e}\n{traceback.format_exc()}")
            error_counter.record_error("global")

    def _check_failure_states(self, price_map: dict[str, float]) -> None:
        """
        Check pending signals for failure state (v2).

        If price crosses SL + 0.5×ATR, mark as FAILED and reset cooldown.

        Args:
            price_map: Dict of symbol -> current price.
        """
        try:
            pending = signal_logger.get_pending_signals()
            for sig in pending:
                symbol = sig.get("symbol", "")
                direction = sig.get("direction", "")
                sl_price = sig.get("stop_loss", 0.0)

                if symbol not in price_map:
                    continue

                current_price = price_map[symbol]

                # Get ATR for this coin
                atr_current = 0.0
                coin_data = data_store.get_coin(symbol)
                if coin_data and coin_data.indicators:
                    atr_current = coin_data.indicators.get("atr", {}).get("15m_current", 0.0)

                failure_buffer = settings.COOLDOWN_FAILURE_ATR_MULTIPLIER * atr_current

                if direction == "LONG":
                    failure_threshold = sl_price - failure_buffer
                    if current_price < failure_threshold:
                        signal_logger.mark_signal_failed(sig.get("id"))
                        # Reset cooldown for this coin
                        coin = data_store.get_coin(symbol)
                        if coin:
                            coin.last_signal_time = 0.0
                        logger.info(
                            f"Signal FAILED: {symbol} LONG — price {current_price} "
                            f"< failure threshold {failure_threshold}"
                        )
                elif direction == "SHORT":
                    failure_threshold = sl_price + failure_buffer
                    if current_price > failure_threshold:
                        signal_logger.mark_signal_failed(sig.get("id"))
                        coin = data_store.get_coin(symbol)
                        if coin:
                            coin.last_signal_time = 0.0
                        logger.info(
                            f"Signal FAILED: {symbol} SHORT — price {current_price} "
                            f"> failure threshold {failure_threshold}"
                        )
        except Exception as e:
            logger.error(f"Failure state check error: {e}")

    def _log_cycle_summary(
        self, session_info: dict, btc_state: dict
    ) -> None:
        """
        Log a detailed summary of the analysis cycle.

        Shows:
        - Status breakdown (signals/rejected/low_score/filtered/cooldown/skip)
        - Top 10 scoring coins (leaderboard)
        - Most common rejection reasons
        - BTC state + session info
        """
        results = self._cycle_results
        if not results:
            return

        # ── Status counts ──
        status_counts: dict[str, int] = {}
        for r in results:
            s = r.get("status", "UNKNOWN")
            status_counts[s] = status_counts.get(s, 0) + 1

        status_line = " | ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))

        logger.info("=" * 70)
        logger.info(f"CYCLE SUMMARY  |  {status_line}")
        logger.info(f"Session: {session_info.get('session_name', '?')} "
                     f"({session_info.get('session_score', 0):+d})  |  "
                     f"BTC: {'Bullish' if btc_state.get('btc_bullish') else 'Bearish' if btc_state.get('btc_bearish') else 'Neutral'}")
        logger.info("-" * 70)

        # ── Top scorers (any coin that scored > 0, sorted desc) ──
        scored = [r for r in results if r.get("score", 0) > 0]
        scored.sort(key=lambda x: x["score"], reverse=True)

        if scored:
            logger.info("TOP SCORERS:")
            for i, r in enumerate(scored[:10]):
                sym = r["symbol"]
                score = r["score"]
                status = r["status"]
                direction = r.get("direction", "?")
                reason = r.get("reason", "")

                # Show scoring breakdown for top 5 if available
                bar = "#" * min(score, 32) + "." * max(0, 32 - score)
                status_emoji = {
                    "SIGNAL": ">>>",
                    "FILTERED": "~F~",
                    "LOW_SCORE": "LOW",
                    "REJECTED": "REJ",
                }.get(status, "---")

                line = f"  {status_emoji} {sym:<12} {direction:>5}  [{bar}] {score}/32  {reason}"
                if status == "SIGNAL":
                    logger.info(line)
                else:
                    logger.info(line)

                # Print score detail breakdown for top 3 coins that have it
                bd = r.get("breakdown")
                if bd and i < 3 and bd.details:
                    parts = []
                    for cond_name, (passed, pts, info) in bd.details.items():
                        if pts != 0:
                            parts.append(f"{cond_name}={pts:+d}")
                    if parts:
                        logger.info(f"       Details: {', '.join(parts)}")
        else:
            logger.info("  No coins scored above 0 this cycle")

        # ── Rejection reason summary ──
        rejected = [r for r in results if r["status"] == "REJECTED"]
        if rejected:
            reason_counts: dict[str, int] = {}
            for r in rejected:
                # Extract just the failure condition (after the colon)
                reason = r.get("reason", "Unknown")
                # Normalize: strip direction prefix for counting
                parts = reason.split(": ", 1)
                key = parts[1] if len(parts) > 1 else parts[0]
                # Shorten common patterns
                if "EMA 50" in key:
                    key = "4H EMA trend"
                elif "ADX" in key:
                    key = "ADX < 20 (ranging)"
                elif "No 1H" in key:
                    key = "No BOS"
                elif "not in" in key and "OB" in key:
                    key = "Not in OB"
                elif "RSI" in key:
                    key = "RSI out of range"
                elif "Fibonacci" in key or "Fib" in key:
                    key = "Fib zone"
                elif "Funding" in key:
                    key = "Funding rate"
                reason_counts[key] = reason_counts.get(key, 0) + 1

            sorted_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])
            reasons_str = ", ".join(f"{k}({v})" for k, v in sorted_reasons[:6])
            logger.info(f"REJECTIONS ({len(rejected)}): {reasons_str}")

        logger.info("=" * 70)

    async def _analyze_coin(
        self,
        symbol: str,
        session_info: dict,
        btc_state: dict,
    ) -> None:
        """
        Run the full analysis pipeline for a single coin (v2).

        Logs detailed scoring info for every coin regardless of outcome.

        Args:
            symbol: Trading pair to analyze.
            session_info: Current trading session info.
            btc_state: BTC correlation state.
        """
        # ── Import scorers for detailed logging ──
        from analysis.scorer import (
            calculate_score_long, calculate_score_short,
        )

        short_sym = symbol.replace("/USDT:USDT", "")

        # ── Cooldown check ──
        if data_store.is_in_cooldown(symbol):
            self._cycle_results.append({
                "symbol": short_sym, "status": "COOLDOWN",
                "reason": "In cooldown", "score": 0,
            })
            return

        # ── Gather OHLCV ──
        ohlcv: dict[str, Any] = {}
        for tf in [settings.MACRO_TF, settings.STRUCTURE_TF, settings.SETUP_TF, settings.ENTRY_TF]:
            df = data_store.get_ohlcv(symbol, tf)
            if df is not None and len(df) > 0:
                ohlcv[tf] = df

        # Minimum data check
        if settings.MACRO_TF not in ohlcv or settings.ENTRY_TF not in ohlcv:
            self._cycle_results.append({
                "symbol": short_sym, "status": "SKIP",
                "reason": "Missing OHLCV data", "score": 0,
            })
            return
        if settings.STRUCTURE_TF not in ohlcv or settings.SETUP_TF not in ohlcv:
            self._cycle_results.append({
                "symbol": short_sym, "status": "SKIP",
                "reason": "Missing OHLCV data", "score": 0,
            })
            return

        # ── Calculate indicators (v2: nested dict) ──
        try:
            indicators = calculate_all_indicators(ohlcv)
        except ValueError as e:
            self._cycle_results.append({
                "symbol": short_sym, "status": "SKIP",
                "reason": str(e)[:60], "score": 0,
            })
            return

        # Store indicators for failure state checks
        coin_data = data_store.get_coin(symbol)
        if coin_data:
            coin_data.indicators = indicators

        # ── Run SMC analysis (v2: passes indicators for ATR/RSI) ──
        smc = run_smc_analysis(ohlcv, indicators)

        # ── Build market data dict ──
        market: dict[str, Any] = {}
        if coin_data:
            market["funding_rate"] = coin_data.funding_rate
            oi_increasing, oi_change = check_oi_increasing(coin_data.oi_history)
            market["oi_increasing"] = oi_increasing
            market["oi_change"] = oi_change
            market["ls_ratio"] = coin_data.ls_ratio
            market["volume_change_pct"] = coin_data.volume_change_pct  # v3: spike context

        # ── Get current price ──
        df_5m = ohlcv.get(settings.ENTRY_TF)
        if df_5m is None or len(df_5m) == 0:
            return
        current_price = float(df_5m["close"].iloc[-1])

        # ── Run BOTH scorers for logging (even if they fail mandatory) ──
        long_breakdown = calculate_score_long(
            indicators, smc, market, current_price,
            session_info=session_info, btc_state=btc_state,
        )
        short_breakdown = calculate_score_short(
            indicators, smc, market, current_price,
            session_info=session_info, btc_state=btc_state,
        )

        # Pick the best direction for logging
        best_dir = "LONG" if long_breakdown.total_score >= short_breakdown.total_score else "SHORT"
        best_bd = long_breakdown if best_dir == "LONG" else short_breakdown

        # ── Check LONG (v2: pass session + BTC state) ──
        long_signal = check_long(
            symbol, indicators, smc, market, current_price,
            session_info=session_info, btc_state=btc_state,
        )
        if long_signal:
            ohlcv_15m = ohlcv.get(settings.SETUP_TF)
            self._signal_queue.append((long_signal, ohlcv_15m))
            self._cycle_results.append({
                "symbol": short_sym, "status": "SIGNAL",
                "reason": f"LONG score={long_signal.score}/{long_signal.max_score} ({long_signal.confidence})",
                "score": long_signal.score, "direction": "LONG",
                "breakdown": long_breakdown,
            })
            return  # Don't check short if long fires

        # ── Check SHORT ──
        short_signal = check_short(
            symbol, indicators, smc, market, current_price,
            session_info=session_info, btc_state=btc_state,
        )
        if short_signal:
            ohlcv_15m = ohlcv.get(settings.SETUP_TF)
            self._signal_queue.append((short_signal, ohlcv_15m))
            self._cycle_results.append({
                "symbol": short_sym, "status": "SIGNAL",
                "reason": f"SHORT score={short_signal.score}/{short_signal.max_score} ({short_signal.confidence})",
                "score": short_signal.score, "direction": "SHORT",
                "breakdown": short_breakdown,
            })
            return

        # ── Both failed — log the best attempt ──
        if not best_bd.mandatory_passed:
            self._cycle_results.append({
                "symbol": short_sym, "status": "REJECTED",
                "reason": f"{best_dir}: {best_bd.failed_mandatory}",
                "score": best_bd.total_score, "direction": best_dir,
            })
        elif best_bd.total_score < settings.MIN_SCORE_TO_SIGNAL:
            self._cycle_results.append({
                "symbol": short_sym, "status": "LOW_SCORE",
                "reason": f"{best_dir}: {best_bd.total_score}/{best_bd.max_score}",
                "score": best_bd.total_score, "direction": best_dir,
                "breakdown": best_bd,
            })
        else:
            # Passed mandatory + score but failed TP/SL or R:R or session gate
            self._cycle_results.append({
                "symbol": short_sym, "status": "FILTERED",
                "reason": f"{best_dir}: score={best_bd.total_score} but failed TP/SL or R:R gate",
                "score": best_bd.total_score, "direction": best_dir,
            })

    async def _update_volume_changes(self, coins: list[str]) -> None:
        """
        Update 24h volume and 5m volume change for all tracked coins (v3).

        Called every 5m cycle after fetching tickers. Computes percentage
        change against the previous cycle's snapshot and stores it per coin.

        Args:
            coins: Active coin list.
        """
        tickers = self._last_ticker_fetch
        if not tickers:
            return

        for symbol in coins:
            ticker = tickers.get(symbol)
            if ticker is None:
                continue

            current_vol = float(ticker.get("quoteVolume", 0) or 0)
            prev_vol = self._last_ticker_snapshot.get(symbol, 0.0)

            if prev_vol > 0:
                change_pct = (current_vol - prev_vol) / prev_vol
            else:
                change_pct = 0.0

            coin_data = data_store.get_coin(symbol)
            if coin_data is not None:
                coin_data.volume_change_pct = change_pct

        # Update snapshot for next cycle
        self._last_ticker_snapshot = {
            sym: float(tickers[sym].get("quoteVolume", 0) or 0)
            for sym in tickers
        }

    async def _hourly_rank_check(self) -> None:
        """
        Compare current coin ranks against thresholds; queue swaps if needed (v3).

        Runs at HH:01 UTC each hour. Uses the last ticker snapshot already in
        memory (no extra API call). Sets _list_update_pending = True when any
        coin has drifted outside the threshold window. The actual swap is
        executed safely at the start of the next 5m analysis cycle.

        Swap logic:
        - Remove if rank > COIN_RANK_DROP_THRESHOLD AND a replacement exists
          with rank < COIN_RANK_RISE_THRESHOLD.
        - Only queues, never executes directly (atomic swap in main loop).
        """
        import time
        now = time.time()
        # Run once per hour at approximately HH:01 UTC
        if now - self._last_hourly_rank_check < 3500:
            return

        tickers = self._last_ticker_fetch
        if not tickers:
            return

        # Sort all tickers by 24h quote volume descending
        ranked = sorted(
            [(sym, float(t.get("quoteVolume", 0) or 0)) for sym, t in tickers.items()],
            key=lambda x: -x[1],
        )
        rank_map: dict[str, int] = {sym: i + 1 for i, (sym, _) in enumerate(ranked)}

        current_coins = set(coin_selector.coins)
        stablecoin_set = {
            f"{s.replace('USDT', '')}/USDT:USDT" for s in settings.STABLECOIN_SYMBOLS
        }

        to_remove: list[str] = []
        to_add: list[str] = []

        for coin in list(current_coins):
            rank = rank_map.get(coin, 9999)
            if rank > settings.COIN_RANK_DROP_THRESHOLD:
                # Find a replacement that is high-ranked and not already tracked
                replacement = None
                for sym, _ in ranked:
                    r = rank_map.get(sym, 9999)
                    if r <= settings.COIN_RANK_RISE_THRESHOLD and sym not in current_coins and sym not in stablecoin_set:
                        replacement = sym
                        break

                if replacement:
                    to_remove.append(coin)
                    to_add.append(replacement)
                    current_coins.discard(coin)
                    current_coins.add(replacement)  # prevent double-adding
                    logger.info(
                        f"[RankCheck] Queue swap: REMOVE {coin} (rank {rank}) -> "
                        f"ADD {replacement} (rank {rank_map.get(replacement, '?')})"
                    )

        if to_remove or to_add:
            self._pending_remove_coins = to_remove
            self._pending_add_coins = to_add
            self._list_update_pending = True
        else:
            logger.info("[RankCheck] All coins within rank thresholds. No swaps needed.")

        self._last_hourly_rank_check = now

    async def _apply_pending_coin_swap(self) -> None:
        """
        Atomically execute queued coin list swaps (v3).

        Called at the start of a 5m cycle when _list_update_pending is True.
        Removes dropped coins + their data, bootstraps new coins with full
        OHLCV fetch, then clears the pending lists.
        """
        if not self._list_update_pending:
            return

        removes = list(self._pending_remove_coins)
        adds = list(self._pending_add_coins)

        logger.info(
            f"[CoinSwap] Applying: removing {len(removes)} coins, "
            f"adding {len(adds)} coins"
        )

        # ── Remove dropped coins ──
        current_coins = list(coin_selector.coins)
        for sym in removes:
            if sym in current_coins:
                current_coins.remove(sym)
            data_store.remove_coin(sym)
            logger.info(f"[CoinSwap] Removed {sym} from watchlist (rank dropped)")

        # ── Add new coins ──
        for sym in adds:
            if sym not in current_coins:
                current_coins.append(sym)
            data_store.init_coin(sym)
            logger.info(f"[CoinSwap] Bootstrapping {sym} — fetching OHLCV...")
            try:
                for tf, limit in [
                    (settings.MACRO_TF, settings.COIN_SWAP_OHLCV_LIMIT),
                    (settings.STRUCTURE_TF, settings.COIN_SWAP_OHLCV_LIMIT),
                    (settings.SETUP_TF, settings.COIN_SWAP_OHLCV_LIMIT),
                    (settings.ENTRY_TF, settings.CANDLE_LIMITS[settings.ENTRY_TF]),
                ]:
                    df = await fetcher.fetch_ohlcv(sym, tf, limit)
                    data_store.update_ohlcv(sym, tf, df)
                    await asyncio.sleep(0.05)
                logger.info(f"[CoinSwap] {sym} added to watchlist (bootstrapped)")
            except Exception as e:
                logger.error(f"[CoinSwap] Failed to bootstrap {sym}: {e}")
                current_coins.remove(sym)
                data_store.remove_coin(sym)

        # Update coin_selector with new list
        coin_selector._coins = current_coins
        coin_selector._save_to_disk()

        # Reset pending state
        self._pending_remove_coins = []
        self._pending_add_coins = []
        self._list_update_pending = False

        logger.info(f"[CoinSwap] Complete. Now tracking {len(current_coins)} coins.")

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

        # Send daily report (v2: includes hit rate)
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
