"""
backtest/walk_forward.py — Phase 1: Candle-by-candle walk-forward simulation.

The simulation engine. For each coin it advances through every 5m candle,
updates higher timeframe indicators and SMC structures only when those
timeframes close, generates candidate signals with full look-ahead protection,
and tracks TP/SL outcomes in real-time.

CRITICAL RULE: When at 5m index i, ALL slices are df[0:i+1].
NEVER pass the full array to any function.

Parallelised across coins using ProcessPoolExecutor (max_workers=4).
"""

import bisect
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Import from live bot analysis modules (read-only) ──
# We add the project root to sys.path so these imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.indicators import (
    calc_ema, calc_rsi, calc_macd, calc_atr, calc_adx,
    calc_fibonacci_zones,
)
from analysis.smc import (
    Direction,
    detect_swing_points,
    detect_bos,
    detect_choch,
    detect_order_blocks,
    detect_fvgs,
    invalidate_fvgs,
    invalidate_order_blocks,
    update_ob_touch_counts,
    detect_liquidity_sweep,
    detect_rsi_divergence,
    detect_equal_levels,
    check_ob_rejection_confirmation,
    is_price_in_order_block,
)

from backtest import config
from backtest.candidate import CandidateSignal
from backtest.downloader import load_ohlcv, load_funding_rates
from backtest.state import SimulationState, PendingOutcome

# Session scoring table (matches live bot settings exactly)
_SESSION_SCORES = {
    "Asian":   -2,
    "London":   2,
    "Overlap":  3,
    "NY":       1,
    "Late":    -1,
}
_SESSION_ASIAN_MIN_SCORE = 12

# Live bot scoring constants (read from live settings)
from config import settings as live_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ms_to_dt(ts_ms: int) -> datetime:
    """Convert Unix milliseconds to UTC datetime."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _get_session(dt: datetime) -> str:
    """Return session name for a UTC datetime."""
    h = dt.hour
    if 0 <= h <= 7:
        return "Asian"
    elif 8 <= h <= 12:
        return "London"
    elif 13 <= h <= 16:
        return "Overlap"
    elif 17 <= h <= 20:
        return "NY"
    else:
        return "Late"


def _funding_at(funding_ts: np.ndarray, funding_rates: np.ndarray, ts_ms: int) -> float:
    """
    Binary-search for the most recent funding rate at or before ts_ms.

    Args:
        funding_ts:    Sorted numpy array of funding timestamps (ms ints).
        funding_rates: Corresponding funding rate floats.
        ts_ms:         Current candle timestamp in ms.

    Returns:
        Funding rate float, or 0.0 if no data available yet.
    """
    idx = bisect.bisect_right(funding_ts, ts_ms) - 1
    if idx < 0:
        return 0.0
    return float(funding_rates[idx])


def _volume_ratio(volume_series: pd.Series, period: int = 20) -> float:
    """Return ratio of last volume to rolling mean of previous N candles."""
    if len(volume_series) < period + 1:
        return 1.0
    avg = float(volume_series.iloc[-(period + 1):-1].mean())
    if avg <= 0:
        return 1.0
    return float(volume_series.iloc[-1]) / avg


def _macd_pre_crossover(hist_last3: list[float], direction: str) -> bool:
    """
    Check if MACD histogram shows a pre-crossover pattern.

    Long: histogram converging upward (all still negative).
    Short: histogram converging downward (all still positive).
    """
    if len(hist_last3) < 3:
        return False
    a, b, c = hist_last3[0], hist_last3[1], hist_last3[2]
    if direction == "long":
        return a < b < c and c < 0
    else:
        return a > b > c and c > 0


def _build_df_slice(arr: np.ndarray, cols: list[str], end_idx: int) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a numpy structured array slice.

    LOOK-AHEAD PROTECTION: always slices [0:end_idx+1].
    """
    slice_data = arr[:end_idx + 1]
    df = pd.DataFrame({col: slice_data[col] for col in cols})
    # Give it a timestamp-based index
    df.index = pd.to_datetime(slice_data["timestamp"], unit="ms", utc=True)
    return df


def _df_from_csv_arrays(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    end_idx: int,
) -> pd.DataFrame:
    """Build a slice DataFrame from pre-extracted numpy arrays."""
    n = end_idx + 1
    df = pd.DataFrame({
        "open":   opens[:n],
        "high":   highs[:n],
        "low":    lows[:n],
        "close":  closes[:n],
        "volume": volumes[:n],
    })
    df.index = pd.to_datetime(timestamps[:n], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# Per-coin simulation
# ---------------------------------------------------------------------------

def _simulate_coin(
    symbol: str,
    btc_bos_by_1h_idx: dict,  # {1h_idx: {'btc_bullish': bool, 'btc_bearish': bool}}
) -> list[CandidateSignal]:
    """
    Run the full walk-forward simulation for one coin.

    Args:
        symbol:           Binance symbol string (e.g. 'BTCUSDT').
        btc_bos_by_1h_idx: Pre-computed BTC BOS state at each 1H index.

    Returns:
        List of CandidateSignal objects for this coin.
    """
    # ── Load data ──
    try:
        df_4h = load_ohlcv(symbol, "4h")
        df_1h = load_ohlcv(symbol, "1h")
        df_15m = load_ohlcv(symbol, "15m")
        df_5m = load_ohlcv(symbol, "5m")
        df_fund = load_funding_rates(symbol)
    except FileNotFoundError as e:
        print(f"  [SKIP] {symbol}: {e}")
        return []

    # ── Convert to numpy for speed ──
    ts_5m = df_5m["timestamp"].to_numpy(dtype=np.int64)
    o_5m = df_5m["open"].to_numpy(dtype=np.float64)
    h_5m = df_5m["high"].to_numpy(dtype=np.float64)
    l_5m = df_5m["low"].to_numpy(dtype=np.float64)
    c_5m = df_5m["close"].to_numpy(dtype=np.float64)
    v_5m = df_5m["volume"].to_numpy(dtype=np.float64)

    ts_15m = df_15m["timestamp"].to_numpy(dtype=np.int64)
    o_15m = df_15m["open"].to_numpy(dtype=np.float64)
    h_15m = df_15m["high"].to_numpy(dtype=np.float64)
    l_15m = df_15m["low"].to_numpy(dtype=np.float64)
    c_15m = df_15m["close"].to_numpy(dtype=np.float64)
    v_15m = df_15m["volume"].to_numpy(dtype=np.float64)

    ts_1h = df_1h["timestamp"].to_numpy(dtype=np.int64)
    o_1h = df_1h["open"].to_numpy(dtype=np.float64)
    h_1h = df_1h["high"].to_numpy(dtype=np.float64)
    l_1h = df_1h["low"].to_numpy(dtype=np.float64)
    c_1h = df_1h["close"].to_numpy(dtype=np.float64)
    v_1h = df_1h["volume"].to_numpy(dtype=np.float64)

    ts_4h = df_4h["timestamp"].to_numpy(dtype=np.int64)
    o_4h = df_4h["open"].to_numpy(dtype=np.float64)
    h_4h = df_4h["high"].to_numpy(dtype=np.float64)
    l_4h = df_4h["low"].to_numpy(dtype=np.float64)
    c_4h = df_4h["close"].to_numpy(dtype=np.float64)
    v_4h = df_4h["volume"].to_numpy(dtype=np.float64)

    fund_ts = df_fund["timestamp"].to_numpy(dtype=np.int64) if len(df_fund) > 0 else np.array([], dtype=np.int64)
    fund_rates = df_fund["funding_rate"].to_numpy(dtype=np.float64) if len(df_fund) > 0 else np.array([], dtype=np.float64)

    n_5m = len(ts_5m)
    n_15m = len(ts_15m)
    n_1h = len(ts_1h)
    n_4h = len(ts_4h)

    state = SimulationState(coin=symbol)
    candidates: list[CandidateSignal] = []

    prev_idx_15m = -1
    prev_idx_1h = -1
    prev_idx_4h = -1

    # ── 200 4H warmup candles = first 200 × 48 = 9600 5m candles ──
    WARMUP_5M = config.WARMUP_4H_CANDLES * 48

    # ── Main 5m loop ──
    for i in range(n_5m):
        current_ts = int(ts_5m[i])
        current_dt = _ts_ms_to_dt(current_ts)

        # ─── STEP A: Update timeframe indices ───
        # Find current position in each higher timeframe array
        idx_4h = bisect.bisect_right(ts_4h, current_ts) - 1
        idx_1h = bisect.bisect_right(ts_1h, current_ts) - 1
        idx_15m = bisect.bisect_right(ts_15m, current_ts) - 1

        if idx_4h < 0: idx_4h = 0
        if idx_1h < 0: idx_1h = 0
        if idx_15m < 0: idx_15m = 0

        # Clamp to valid range
        idx_4h = min(idx_4h, n_4h - 1)
        idx_1h = min(idx_1h, n_1h - 1)
        idx_15m = min(idx_15m, n_15m - 1)

        state.idx_5m = i
        state.idx_4h = idx_4h
        state.idx_1h = idx_1h
        state.idx_15m = idx_15m

        # ─── STEP B: Warmup check ───
        if i < WARMUP_5M or idx_4h < config.WARMUP_4H_CANDLES:
            # Still in warmup — only update BTC state for later use
            if idx_4h != prev_idx_4h and idx_4h >= 1:
                df_4h_slice = _df_from_csv_arrays(ts_4h, o_4h, h_4h, l_4h, c_4h, v_4h, idx_4h)
                ema50 = calc_ema(df_4h_slice["close"], 50)
                ema200 = calc_ema(df_4h_slice["close"], 200)
                if len(ema50) > 0 and len(ema200) > 0:
                    state.ema_4h_50 = float(ema50.iloc[-1])
                    state.ema_4h_200 = float(ema200.iloc[-1])
                prev_idx_4h = idx_4h
            continue

        if not state.warmup_complete:
            state.warmup_complete = True

        # ─── STEP C: Higher timeframe updates (on close only) ───

        # 4H close
        if idx_4h != prev_idx_4h:
            df_4h_slice = _df_from_csv_arrays(ts_4h, o_4h, h_4h, l_4h, c_4h, v_4h, idx_4h)
            ema50 = calc_ema(df_4h_slice["close"], 50)
            ema200 = calc_ema(df_4h_slice["close"], 200)
            if len(ema50) > 0:
                state.ema_4h_50 = float(ema50.iloc[-1])
            if len(ema200) > 0:
                state.ema_4h_200 = float(ema200.iloc[-1])
            prev_idx_4h = idx_4h

        # 1H close
        if idx_1h != prev_idx_1h:
            df_1h_slice = _df_from_csv_arrays(ts_1h, o_1h, h_1h, l_1h, c_1h, v_1h, idx_1h)

            # EMA 21, 50 on 1H
            ema21 = calc_ema(df_1h_slice["close"], 21)
            ema50 = calc_ema(df_1h_slice["close"], 50)
            if len(ema21) > 0:
                state.ema_1h_21 = float(ema21.iloc[-1])
            if len(ema50) > 0:
                state.ema_1h_50 = float(ema50.iloc[-1])

            # ADX 1H
            adx_series = calc_adx(df_1h_slice["high"], df_1h_slice["low"], df_1h_slice["close"])
            if len(adx_series) > 0:
                state.adx_1h = float(adx_series.iloc[-1])

            # 4H slice for Fibonacci
            df_4h_for_fib = _df_from_csv_arrays(ts_4h, o_4h, h_4h, l_4h, c_4h, v_4h, idx_4h)

            # Swing points on 1H
            sh_1h, sl_1h = detect_swing_points(
                df_1h_slice, lookback=live_settings.SWING_LOOKBACK_1H
            )
            state.swing_highs_1h = sh_1h
            state.swing_lows_1h = sl_1h

            # BOS + CHoCH on 1H
            state.bos_1h = detect_bos(df_1h_slice, sh_1h, sl_1h)
            state.choch_1h = detect_choch(df_1h_slice, sh_1h, sl_1h)

            # Order Blocks on 1H
            atr_1h_series = calc_atr(df_1h_slice["high"], df_1h_slice["low"], df_1h_slice["close"])
            bull_obs, bear_obs = detect_order_blocks(df_1h_slice, atr_1h_series)
            bull_obs = invalidate_order_blocks(bull_obs, df_1h_slice)
            bear_obs = invalidate_order_blocks(bear_obs, df_1h_slice)
            bull_obs = update_ob_touch_counts(bull_obs, df_1h_slice)
            bear_obs = update_ob_touch_counts(bear_obs, df_1h_slice)
            state.order_blocks_bull_1h = bull_obs
            state.order_blocks_bear_1h = bear_obs

            # Fibonacci zones (from 4H swing points)
            sh_4h_pts, sl_4h_pts = detect_swing_points(df_4h_for_fib, lookback=live_settings.SWING_LOOKBACK_1H)
            current_price_for_fib = float(c_5m[i])
            try:
                state.fib_zones = calc_fibonacci_zones(sh_4h_pts, sl_4h_pts, current_price_for_fib)
            except Exception:
                state.fib_zones = None

            # Equal levels
            try:
                eq_result = detect_equal_levels(
                    swing_highs_1h=sh_1h,
                    swing_lows_1h=sl_1h,
                    df_1h=df_1h_slice,
                    atr_1h=atr_1h_series,
                    sweep_info=state.liquidity_sweep,
                    current_idx=len(df_1h_slice) - 1,
                )
                state.equal_highs_1h = eq_result.get("equal_highs", [])
                state.equal_lows_1h = eq_result.get("equal_lows", [])
            except Exception:
                state.equal_highs_1h = []
                state.equal_lows_1h = []

            # BTC correlation (from pre-computed BTC state)
            btc_state_at_1h = btc_bos_by_1h_idx.get(idx_1h, {})
            state.btc_bullish = btc_state_at_1h.get("btc_bullish", False)
            state.btc_bearish = btc_state_at_1h.get("btc_bearish", False)

            prev_idx_1h = idx_1h

        # 15m close
        if idx_15m != prev_idx_15m:
            df_15m_slice = _df_from_csv_arrays(
                ts_15m, o_15m, h_15m, l_15m, c_15m, v_15m, idx_15m
            )

            # ATR 15m
            atr_15m_series = calc_atr(
                df_15m_slice["high"], df_15m_slice["low"], df_15m_slice["close"]
            )
            if len(atr_15m_series) > 0:
                state.atr_15m = float(atr_15m_series.iloc[-1])

            # Volume ratio 15m
            state.volume_ratio_15m = _volume_ratio(df_15m_slice["volume"])

            # FVGs on 15m
            bull_fvgs, bear_fvgs = detect_fvgs(df_15m_slice)
            bull_fvgs = invalidate_fvgs(bull_fvgs, df_15m_slice)
            bear_fvgs = invalidate_fvgs(bear_fvgs, df_15m_slice)
            state.fvg_zones_bull_15m = bull_fvgs
            state.fvg_zones_bear_15m = bear_fvgs

            # Swing points on 15m (for sweep and divergence)
            sh_15m, sl_15m = detect_swing_points(
                df_15m_slice, lookback=live_settings.SWING_LOOKBACK_15M
            )
            state.swing_highs_15m = sh_15m
            state.swing_lows_15m = sl_15m

            # Liquidity sweep on 15m
            try:
                state.liquidity_sweep = detect_liquidity_sweep(
                    df_15m_slice, sl_15m, sh_15m, atr_15m_series
                )
            except Exception:
                state.liquidity_sweep = {}

            # RSI divergence on 15m
            try:
                rsi_15m_series = calc_rsi(df_15m_slice["close"])
                state.rsi_divergence = detect_rsi_divergence(
                    df_15m_slice, rsi_15m_series, sl_15m, sh_15m
                )
            except Exception:
                state.rsi_divergence = {"bullish_divergence": False, "bearish_divergence": False}

            # OB rejection confirmation (15m candle on 1H OB)
            for ob in state.order_blocks_bull_1h:
                try:
                    ob.confirmed_rejection = check_ob_rejection_confirmation(df_15m_slice, ob)
                except Exception:
                    pass
            for ob in state.order_blocks_bear_1h:
                try:
                    ob.confirmed_rejection = check_ob_rejection_confirmation(df_15m_slice, ob)
                except Exception:
                    pass

            prev_idx_15m = idx_15m

        # ─── STEP D: 5m indicators (every candle) ───
        df_5m_slice = _df_from_csv_arrays(ts_5m, o_5m, h_5m, l_5m, c_5m, v_5m, i)

        rsi_5m_series = calc_rsi(df_5m_slice["close"])
        if len(rsi_5m_series) >= 3:
            state.rsi_5m_current = float(rsi_5m_series.iloc[-1])
            state.rsi_5m_last3 = [
                float(rsi_5m_series.iloc[-3]),
                float(rsi_5m_series.iloc[-2]),
                float(rsi_5m_series.iloc[-1]),
            ]

        try:
            macd_fast = calc_ema(df_5m_slice["close"], live_settings.MACD_FAST)
            macd_slow = calc_ema(df_5m_slice["close"], live_settings.MACD_SLOW)
            macd_line = macd_fast - macd_slow
            signal_line = calc_ema(macd_line, live_settings.MACD_SIGNAL)
            histogram = macd_line - signal_line
            if len(histogram) >= 3:
                state.macd_histogram_last3 = [
                    float(histogram.iloc[-3]),
                    float(histogram.iloc[-2]),
                    float(histogram.iloc[-1]),
                ]
        except Exception:
            pass

        state.volume_ratio_5m = _volume_ratio(df_5m_slice["volume"])
        state.current_funding_rate = _funding_at(fund_ts, fund_rates, current_ts)

        current_price = float(c_5m[i])
        current_high = float(h_5m[i])
        current_low = float(l_5m[i])

        # ─── STEP E: Outcome tracking (BEFORE new signal generation) ───
        if state.pending_outcome is not None:
            po = state.pending_outcome
            ci = po.candidate_idx
            resolved = False

            if po.direction == "long":
                # MAE/MFE for long
                drawdown = po.entry - current_low
                runup = current_high - po.entry
                candidates[ci].max_adverse_excursion = max(
                    candidates[ci].max_adverse_excursion, max(drawdown, 0.0)
                )
                candidates[ci].max_favorable_excursion = max(
                    candidates[ci].max_favorable_excursion, max(runup, 0.0)
                )

                if current_high >= po.tp2:
                    candidates[ci].outcome = "TP2_HIT"
                    candidates[ci].tp2_hit = True
                    candidates[ci].tp1_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    resolved = True
                elif current_high >= po.tp1:
                    candidates[ci].outcome = "TP1_HIT"
                    candidates[ci].tp1_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    resolved = True
                elif current_low <= po.sl:
                    candidates[ci].outcome = "SL_HIT"
                    candidates[ci].sl_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    # Reset cooldown on SL hit
                    state.cooldown_active = False
                    state.cooldown_until = None
                    resolved = True

            else:  # short
                drawdown = current_high - po.entry
                runup = po.entry - current_low
                candidates[ci].max_adverse_excursion = max(
                    candidates[ci].max_adverse_excursion, max(drawdown, 0.0)
                )
                candidates[ci].max_favorable_excursion = max(
                    candidates[ci].max_favorable_excursion, max(runup, 0.0)
                )

                if current_low <= po.tp2:
                    candidates[ci].outcome = "TP2_HIT"
                    candidates[ci].tp2_hit = True
                    candidates[ci].tp1_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    resolved = True
                elif current_low <= po.tp1:
                    candidates[ci].outcome = "TP1_HIT"
                    candidates[ci].tp1_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    resolved = True
                elif current_high >= po.sl:
                    candidates[ci].outcome = "SL_HIT"
                    candidates[ci].sl_hit = True
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None
                    state.cooldown_active = False
                    state.cooldown_until = None
                    resolved = True

            # Check expiry if not yet resolved
            if not resolved and state.pending_outcome is not None:
                if i >= po.expires_at_idx:
                    candidates[ci].outcome = "EXPIRED"
                    candidates[ci].candles_to_outcome = i - po.signal_idx
                    state.pending_outcome = None

        # ─── STEP F: Cooldown check ───
        if state.cooldown_active:
            if state.cooldown_until is not None and current_dt >= state.cooldown_until:
                state.cooldown_active = False
                state.cooldown_until = None
            else:
                continue

        # Skip if another trade is still pending
        if state.pending_outcome is not None:
            continue

        # ─── STEP G: Signal generation ───
        # Try both long and short
        for direction in ("long", "short"):
            candidate = _try_generate_signal(
                direction=direction,
                symbol=symbol,
                i=i,
                current_price=current_price,
                current_ts=current_ts,
                current_dt=current_dt,
                state=state,
                atr_15m=state.atr_15m,
            )
            if candidate is not None:
                candidates.append(candidate)
                ci = len(candidates) - 1

                # Set up pending outcome tracking
                sl_price = candidate.sl_price
                tp1 = candidate.tp1_price
                tp2 = candidate.tp2_price
                risk = abs(current_price - sl_price)

                state.pending_outcome = PendingOutcome(
                    candidate_idx=ci,
                    entry=current_price,
                    tp1=tp1,
                    tp2=tp2,
                    sl=sl_price,
                    direction=direction,
                    signal_idx=i,
                    expires_at_idx=i + config.OUTCOME_WINDOW_CANDLES,
                )

                # Set cooldown
                state.cooldown_active = True
                state.cooldown_until = current_dt + timedelta(hours=config.COOLDOWN_HOURS)
                state.last_signal_direction = direction
                state.last_signal_sl = sl_price
                state.last_signal_entry = current_price

                # Only generate one signal per candle
                break

    print(f"  {symbol}: {len(candidates)} candidate signals")
    return candidates


# ---------------------------------------------------------------------------
# Mandatory gate evaluation (look-ahead safe — uses state only)
# ---------------------------------------------------------------------------

def _try_generate_signal(
    direction: str,
    symbol: str,
    i: int,
    current_price: float,
    current_ts: int,
    current_dt: datetime,
    state: SimulationState,
    atr_15m: float,
) -> Optional[CandidateSignal]:
    """
    Evaluate all mandatory gates and collect scoring inputs for one direction.

    Returns CandidateSignal if mandatory gates pass and TP/SL is valid,
    else None.
    """
    if atr_15m <= 0:
        return None

    # ── Gate 1: ADX ──
    adx = state.adx_1h
    gate_adx = adx >= live_settings.ADX_MIN_THRESHOLD

    # ── Gate 2: 4H EMA structure ──
    ema50 = state.ema_4h_50
    ema200 = state.ema_4h_200
    if direction == "long":
        gate_ema = ema50 > ema200 and ema200 > 0
    else:
        gate_ema = 0 < ema50 < ema200

    # ── Gate 3: 1H BOS ──
    bos = state.bos_1h
    if direction == "long":
        gate_bos = bos is not None and bos.direction == Direction.BULLISH
    else:
        gate_bos = bos is not None and bos.direction == Direction.BEARISH

    # ── Gate 4: OB present ──
    if direction == "long":
        obs = state.order_blocks_bull_1h
    else:
        obs = state.order_blocks_bear_1h

    active_ob = None
    for ob in obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break
    gate_ob = active_ob is not None

    # ── Gate 5: Funding rate ──
    fr = state.current_funding_rate
    if direction == "long":
        gate_funding = fr < live_settings.FUNDING_RATE_LONG_MAX
    else:
        gate_funding = fr > live_settings.FUNDING_RATE_SHORT_MIN

    # ── Gate 6: RSI range ──
    rsi = state.rsi_5m_current
    rsi3 = state.rsi_5m_last3
    if direction == "long":
        in_range = live_settings.RSI_LONG_MIN <= rsi <= live_settings.RSI_LONG_MAX
        trending = len(rsi3) == 3 and rsi3[0] < rsi3[1] < rsi3[2]
    else:
        in_range = live_settings.RSI_SHORT_MIN <= rsi <= live_settings.RSI_SHORT_MAX
        trending = len(rsi3) == 3 and rsi3[0] > rsi3[1] > rsi3[2]
    gate_rsi = in_range and trending

    # ── Gate 7: Fibonacci zone ──
    gate_fib = False
    fib_deep = False
    fib_zones = state.fib_zones
    if fib_zones is not None:
        trend = fib_zones.get("trend_direction")
        retrace = fib_zones.get("current_retrace_pct", 0.0)
        if direction == "long":
            gate_fib = (
                trend == "bullish"
                and live_settings.FIB_OPTIMAL_LONG_LOW <= retrace <= live_settings.FIB_DEEP_DISCOUNT_HIGH
            )
            fib_deep = (
                gate_fib
                and live_settings.FIB_OPTIMAL_LONG_HIGH <= retrace <= live_settings.FIB_DEEP_DISCOUNT_HIGH
            )
        else:
            gate_fib = (
                trend == "bearish"
                and live_settings.FIB_OPTIMAL_SHORT_LOW <= retrace <= live_settings.FIB_OPTIMAL_SHORT_HIGH
            )

    # ── All mandatory gates must pass ──
    if not all([gate_adx, gate_ema, gate_bos, gate_ob, gate_funding, gate_rsi, gate_fib]):
        return None

    # ── Calculate SL with default ATR multiplier ──
    if active_ob is None:
        return None

    if direction == "long":
        sl_raw_distance = live_settings.ATR_SL_MULTIPLIER * atr_15m
        sl_price = active_ob.bottom - sl_raw_distance
        risk = current_price - sl_price
    else:
        sl_raw_distance = live_settings.ATR_SL_MULTIPLIER * atr_15m
        sl_price = active_ob.top + sl_raw_distance
        risk = sl_price - current_price

    if risk <= 0:
        return None

    sl_pct = risk / current_price
    if sl_pct > live_settings.MAX_SL_PERCENTAGE:
        return None

    # R:R gate with default settings
    if direction == "long":
        tp1 = current_price + live_settings.TP1_RR * risk
        tp2 = current_price + live_settings.TP2_RR * risk
    else:
        tp1 = current_price - live_settings.TP1_RR * risk
        tp2 = current_price - live_settings.TP2_RR * risk

    rr = live_settings.TP1_RR  # by definition
    if rr < live_settings.MIN_RR_TO_SIGNAL:
        return None

    # ── Collect scoring inputs ──
    session = _get_session(current_dt)

    # FVG
    if direction == "long":
        fvg_present = any(fvg.top <= current_price for fvg in state.fvg_zones_bull_15m)
    else:
        fvg_present = any(fvg.bottom >= current_price for fvg in state.fvg_zones_bear_15m)

    # CHoCH
    choch = state.choch_1h
    if direction == "long":
        choch_present = choch is not None and choch.direction == Direction.BULLISH
    else:
        choch_present = choch is not None and choch.direction == Direction.BEARISH

    # MACD pre-crossover
    macd_trend = _macd_pre_crossover(state.macd_histogram_last3, direction)

    # EMA 200 gap
    ema200_gap = (current_price - ema200) / ema200 if ema200 > 0 else 0.0

    # Sweep
    sweep = state.liquidity_sweep
    if direction == "long":
        sweep_detected = sweep.get("bullish_sweep", False)
    else:
        sweep_detected = sweep.get("bearish_sweep", False)

    # RSI divergence
    div = state.rsi_divergence
    if direction == "long":
        rsi_div = div.get("bullish_divergence", False)
    else:
        rsi_div = div.get("bearish_divergence", False)

    # OB rejection + touch
    ob_rejection = active_ob.confirmed_rejection if active_ob else False
    ob_touch = active_ob.touch_count if active_ob else 0

    # BTC flags (don't apply to BTC itself)
    is_btc = symbol.upper().startswith("BTC")
    btc_bullish = state.btc_bullish if not is_btc else False
    btc_bearish = state.btc_bearish if not is_btc else False

    # Equal levels
    if direction == "long":
        eq_zones = [z for z in state.equal_lows_1h if not z.swept]
    else:
        eq_zones = [z for z in state.equal_highs_1h if not z.swept]

    eq_present = False
    eq_count = 0
    atr_1h_approx = state.atr_15m * 4  # rough 1H ATR from 15m ATR
    if atr_1h_approx > 0 and eq_zones:
        band = live_settings.EQUAL_LEVEL_PROXIMITY_ATR_MULT * atr_1h_approx
        for ez in eq_zones:
            if direction == "long":
                in_prox = ez.zone_price <= current_price <= ez.zone_price + band
            else:
                in_prox = ez.zone_price - band <= current_price <= ez.zone_price
            if in_prox:
                eq_present = True
                eq_count = ez.member_count
                break

    # OB volume quality
    ob_vol_quality = active_ob.volume_quality_score == 1 if active_ob else False

    # ── Build CandidateSignal ──
    return CandidateSignal(
        coin=symbol,
        direction=direction,
        timestamp=current_dt,
        candle_idx_5m=i,
        session=session,
        entry_price=current_price,
        sl_raw_distance=sl_raw_distance,
        ob_bottom=active_ob.bottom,
        ob_top=active_ob.top,
        sl_percentage=sl_pct,
        tp1_price=round(tp1, 8),
        tp2_price=round(tp2, 8),
        sl_price=round(sl_price, 8),
        # Gates
        gate_ema_4h_structure=gate_ema,
        gate_bos_1h=gate_bos,
        gate_ob_present=gate_ob,
        gate_adx_min=gate_adx,
        gate_fib_zone=gate_fib,
        gate_rsi_range=gate_rsi,
        gate_funding=gate_funding,
        # Scoring inputs
        fvg_present=fvg_present,
        volume_ratio_15m=state.volume_ratio_15m,
        volume_ratio_5m=state.volume_ratio_5m,
        oi_increasing=False,  # disabled in backtest
        choch_present=choch_present,
        macd_histogram_trend=macd_trend,
        ls_ratio=1.0,          # not available historically
        funding_rate=fr,
        ema200_gap_pct=ema200_gap,
        sweep_detected=sweep_detected,
        rsi_divergence=rsi_div,
        adx_value=adx,
        ob_rejection_candle=ob_rejection,
        ob_touch_count=ob_touch,
        fib_deep_discount=fib_deep,
        btc_bullish=btc_bullish,
        btc_bearish=btc_bearish,
        equal_level_present=eq_present,
        equal_level_count=eq_count,
        ob_volume_quality=ob_vol_quality,
    )


# ---------------------------------------------------------------------------
# BTC state pre-computation
# ---------------------------------------------------------------------------

def _precompute_btc_state(btc_symbol: str) -> dict[int, dict]:
    """
    Pre-compute BTC BOS state at every 1H index.

    This is computed ONCE and shared with all coin simulations, so we
    avoid re-computing BTC state inside every coin's loop.

    Args:
        btc_symbol: Binance symbol string for BTC (e.g. 'BTCUSDT').

    Returns:
        Dict mapping 1H index → {'btc_bullish': bool, 'btc_bearish': bool}.
    """
    try:
        df_1h = load_ohlcv(btc_symbol, "1h")
        df_4h = load_ohlcv(btc_symbol, "4h")
    except FileNotFoundError:
        return {}

    ts_1h = df_1h["timestamp"].to_numpy(dtype=np.int64)
    ts_4h = df_4h["timestamp"].to_numpy(dtype=np.int64)
    o_1h = df_1h["open"].to_numpy(dtype=np.float64)
    h_1h = df_1h["high"].to_numpy(dtype=np.float64)
    l_1h = df_1h["low"].to_numpy(dtype=np.float64)
    c_1h = df_1h["close"].to_numpy(dtype=np.float64)
    v_1h = df_1h["volume"].to_numpy(dtype=np.float64)
    c_4h = df_4h["close"].to_numpy(dtype=np.float64)

    result: dict[int, dict] = {}

    for idx in range(live_settings.SWING_LOOKBACK_1H * 2 + 5, len(ts_1h)):
        df_slice = _df_from_csv_arrays(ts_1h, o_1h, h_1h, l_1h, c_1h, v_1h, idx)

        sh, sl = detect_swing_points(df_slice, lookback=live_settings.SWING_LOOKBACK_1H)
        bos = detect_bos(df_slice, sh, sl)

        # EMA 50 on BTC 1H
        ema50 = calc_ema(df_slice["close"], 50)
        btc_price = float(c_1h[idx])
        btc_ema50 = float(ema50.iloc[-1]) if len(ema50) > 0 else 0.0

        btc_bullish = (
            bos is not None
            and bos.direction == Direction.BULLISH
            and btc_price > btc_ema50
        )
        btc_bearish = (
            bos is not None
            and bos.direction == Direction.BEARISH
            and btc_price < btc_ema50
        )
        result[idx] = {"btc_bullish": btc_bullish, "btc_bearish": btc_bearish}

    return result


# ---------------------------------------------------------------------------
# Phase 1 entry point
# ---------------------------------------------------------------------------

def _simulate_coin_wrapper(args: tuple) -> tuple[str, list]:
    """Wrapper for ProcessPoolExecutor (must be top-level picklable function)."""
    symbol, btc_bos = args
    try:
        candidates = _simulate_coin(symbol, btc_bos)
        return symbol, candidates
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}\n{traceback.format_exc()}")
        return symbol, []


def run(coins: list[str], phase1_only: bool = False) -> list[CandidateSignal]:
    """
    Phase 1 entry point.

    Runs the walk-forward simulation for all coins in parallel.

    Args:
        coins:       List of Binance symbol strings.
        phase1_only: If True, print summary and return without Phase 2.

    Returns:
        Master list of all CandidateSignal objects, sorted by timestamp.
    """
    print("\n=== Phase 1: Walk-Forward Simulation ===")

    # Pre-compute BTC state (used for correlation scoring in all coins)
    btc_symbol = next(
        (c for c in coins if c.upper().startswith("BTC")),
        coins[0] if coins else "BTCUSDT",
    )
    print(f"  Pre-computing BTC state from {btc_symbol}...")
    btc_bos_by_1h_idx = _precompute_btc_state(btc_symbol)
    print(f"  BTC state ready ({len(btc_bos_by_1h_idx)} 1H candles)")

    # Parallel simulation across all coins
    all_candidates: list[CandidateSignal] = []
    args_list = [(symbol, btc_bos_by_1h_idx) for symbol in coins]

    max_workers = min(4, len(coins))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_simulate_coin_wrapper, args): args[0] for args in args_list}
        for future in as_completed(futures):
            symbol, candidates = future.result()
            all_candidates.extend(candidates)

    # Sort by timestamp ascending
    all_candidates.sort(key=lambda c: c.timestamp)

    print(f"\n  Phase 1 complete. Total candidate signals: {len(all_candidates)}")
    per_coin: dict[str, int] = {}
    for c in all_candidates:
        per_coin[c.coin] = per_coin.get(c.coin, 0) + 1
    for sym, cnt in sorted(per_coin.items()):
        print(f"    {sym}: {cnt}")

    return all_candidates
