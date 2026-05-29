"""
analysis/indicators.py — Pure pandas indicator calculations.

Implements EMA, RSI, MACD, ATR, ADX, Volume SMA, and Fibonacci zones
without any external indicator library. All calculations use pandas
built-in ewm/rolling with correct Wilder's smoothing where required.

Public helpers are callable by any module (e.g., smc.py imports calc_atr).
The single entry-point calculate_all_indicators() packages everything
into a nested dict for consumers.

Smoothing rules:
  - EMA, MACD: standard EWM with span
  - RSI, ATR, ADX: Wilder's smoothing with alpha=1/period
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── EMA (Exponential Moving Average) ─────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Uses standard EWM (NOT Wilder's smoothing).

    Args:
        series: Price series (typically close prices).
        period: EMA period (e.g., 9, 21, 50, 200).

    Returns:
        EMA series with the same index as input.
    """
    return series.ewm(span=period, adjust=False).mean()


# ─── RSI (Relative Strength Index) ────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = settings.RSI_PERIOD) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing method.

    Uses alpha=1/period (NOT span=period) to match TradingView output.

    Args:
        series: Close price series.
        period: RSI period (default: 14).

    Returns:
        RSI series (values 0-100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ─── MACD (Moving Average Convergence Divergence) ─────────────────────────

def calc_macd(
    series: pd.Series,
    fast: int = settings.MACD_FAST,
    slow: int = settings.MACD_SLOW,
    signal: int = settings.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.

    All use standard EWM (not Wilder's). MACD is correct with span.

    Args:
        series: Close price series.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal EMA period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ─── ATR (Average True Range) ─────────────────────────────────────────────

def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = settings.ATR_PERIOD,
) -> pd.Series:
    """
    Calculate Average True Range using Wilder's smoothing.

    Uses alpha=1/period (NOT span=period) to match TradingView output.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period (default: 14).

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing = alpha = 1/period
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return atr


# ─── Volume SMA ───────────────────────────────────────────────────────────

def calc_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average of volume.

    Args:
        volume: Volume series.
        period: SMA period (default: 20).

    Returns:
        Volume SMA series.
    """
    return volume.rolling(window=period).mean()


def calc_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate the ratio of current volume to its SMA.

    A ratio > 1.5 indicates significant volume.

    Args:
        volume: Volume series.
        period: SMA period for the average.

    Returns:
        Volume ratio series (current / average).
    """
    avg = calc_volume_sma(volume, period)
    return volume / avg


# ─── ADX (Average Directional Index) ──────────────────────────────────────

def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = settings.ADX_PERIOD,
) -> dict[str, pd.Series]:
    """
    Calculate Average Directional Index using Wilder's smoothing.

    All smoothing uses alpha=1/period to match TradingView.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ADX period (default: 14).

    Returns:
        Dict with 'adx', 'di_plus', 'di_minus' as pd.Series.
    """
    # Step 1: True Range (same as ATR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Step 2: Raw Directional Movement
    dm_plus_raw = high.diff()
    dm_minus_raw = -low.diff()

    # DM+ valid only when > DM- AND > 0
    dm_plus = pd.Series(0.0, index=high.index)
    dm_minus = pd.Series(0.0, index=high.index)

    plus_mask = (dm_plus_raw > dm_minus_raw) & (dm_plus_raw > 0)
    minus_mask = (dm_minus_raw > dm_plus_raw) & (dm_minus_raw > 0)

    dm_plus = dm_plus_raw.where(plus_mask, 0.0)
    dm_minus = dm_minus_raw.where(minus_mask, 0.0)

    # Step 3: Wilder's smoothing
    smoothed_tr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_dm_plus = dm_plus.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_dm_minus = dm_minus.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Step 4: Directional Indicators
    # Handle division by zero: where smoothed_tr is 0, DI = 0
    di_plus = pd.Series(0.0, index=high.index)
    di_minus = pd.Series(0.0, index=high.index)

    valid_tr = smoothed_tr > 0
    di_plus = np.where(valid_tr, 100.0 * (smoothed_dm_plus / smoothed_tr), 0.0)
    di_minus = np.where(valid_tr, 100.0 * (smoothed_dm_minus / smoothed_tr), 0.0)

    di_plus = pd.Series(di_plus, index=high.index)
    di_minus = pd.Series(di_minus, index=high.index)

    # Step 5: DX and ADX
    di_sum = di_plus + di_minus
    dx = pd.Series(0.0, index=high.index)
    valid_sum = di_sum > 0
    dx = np.where(valid_sum, 100.0 * (di_plus - di_minus).abs() / di_sum, 0.0)
    dx = pd.Series(dx, index=high.index)

    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return {
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus,
    }


# ─── Fibonacci Zone Calculation ───────────────────────────────────────────

def calc_fibonacci_zones(
    swing_highs: list,
    swing_lows: list,
    current_price: float,
) -> dict[str, Any]:
    """
    Calculate Fibonacci retracement zones from 4H swing points.

    Replaces the old premium/discount 50% midpoint calculation.

    Args:
        swing_highs: List of SwingPoint objects (4H swing highs).
        swing_lows: List of SwingPoint objects (4H swing lows).
        current_price: Current market price.

    Returns:
        Dict with fib levels, zone boundaries, and trend direction.
        Returns None if insufficient swing data.
    """
    if not swing_highs or not swing_lows:
        return None

    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]

    sh_price = last_sh.price
    sl_price = last_sl.price

    # Determine trend by which swing came last
    if last_sh.index > last_sl.index:
        # Swing high is more recent → uptrend → price retracing down
        trend = "bullish"
        swing_range = sh_price - sl_price
        if swing_range <= 0:
            return None

        # Retracement levels: how far price has pulled back from the high
        fib_levels = {
            "swing_high": sh_price,
            "swing_low": sl_price,
            "fib_0": sl_price,                                    # 0% (swing low)
            "fib_236": sh_price - (0.236 * swing_range),          # 23.6%
            "fib_382": sh_price - (0.382 * swing_range),          # 38.2%
            "fib_500": sh_price - (0.500 * swing_range),          # 50.0%
            "fib_618": sh_price - (0.618 * swing_range),          # 61.8%
            "fib_786": sh_price - (0.786 * swing_range),          # 78.6%
            "fib_100": sh_price,                                  # 100% (swing high)
        }
    else:
        # Swing low is more recent → downtrend → price retracing up
        trend = "bearish"
        swing_range = sh_price - sl_price
        if swing_range <= 0:
            return None

        fib_levels = {
            "swing_high": sh_price,
            "swing_low": sl_price,
            "fib_0": sh_price,                                    # 0% (swing high)
            "fib_236": sl_price + (0.236 * swing_range),          # 23.6%
            "fib_382": sl_price + (0.382 * swing_range),          # 38.2%
            "fib_500": sl_price + (0.500 * swing_range),          # 50.0%
            "fib_618": sl_price + (0.618 * swing_range),          # 61.8%
            "fib_786": sl_price + (0.786 * swing_range),          # 78.6%
            "fib_100": sl_price,                                  # 100% (swing low)
        }

    fib_levels["trend_direction"] = trend

    # Zone boundaries for signal logic
    if trend == "bullish":
        fib_levels["long_optimal_zone"] = (fib_levels["fib_500"], fib_levels["fib_618"])
        fib_levels["long_deep_zone"] = (fib_levels["fib_618"], fib_levels["fib_786"])
        fib_levels["short_optimal_zone"] = (fib_levels["fib_382"], fib_levels["fib_500"])
    else:
        fib_levels["long_optimal_zone"] = (fib_levels["fib_500"], fib_levels["fib_618"])
        fib_levels["long_deep_zone"] = (fib_levels["fib_618"], fib_levels["fib_786"])
        fib_levels["short_optimal_zone"] = (fib_levels["fib_382"], fib_levels["fib_500"])

    # Determine current price position as retracement percentage
    if trend == "bullish" and swing_range > 0:
        retrace_pct = (sh_price - current_price) / swing_range
    elif trend == "bearish" and swing_range > 0:
        retrace_pct = (current_price - sl_price) / swing_range
    else:
        retrace_pct = 0.5

    fib_levels["current_retrace_pct"] = retrace_pct

    return fib_levels


# ─── Safe Float Extraction ────────────────────────────────────────────────

def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    """
    Safely get the last value of a Series as a float.

    Returns default if series is empty or last value is NaN.
    """
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    if pd.isna(val):
        return default
    return float(val)


def _safe_last_n(series: pd.Series, n: int, default: float = 0.0) -> list[float]:
    """
    Safely get the last N values of a Series as a list of floats.

    Returns list of defaults if insufficient data.
    """
    if series is None or len(series) < n:
        return [default] * n
    vals = series.iloc[-n:].tolist()
    return [default if pd.isna(v) else float(v) for v in vals]


# ─── Bulk Indicator Calculation ───────────────────────────────────────────

def calculate_all_indicators(
    ohlcv: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Calculate all indicators for a single coin across all timeframes.

    Returns a nested dict structure for clean access by consumers.

    Args:
        ohlcv: Dict of timeframe -> OHLCV DataFrame.
               Expected keys: '4h', '1h', '15m', '5m'.

    Returns:
        Nested dict with indicator results. See structure below.

    Raises:
        ValueError: If any required timeframe key is missing.
    """
    # ── Key validation ──
    required_keys = {settings.MACRO_TF, settings.STRUCTURE_TF,
                     settings.SETUP_TF, settings.ENTRY_TF}
    missing = required_keys - set(ohlcv.keys())
    if missing:
        raise ValueError(
            f"Missing required timeframe keys in ohlcv: {missing}. "
            f"Expected: {required_keys}"
        )

    df_4h = ohlcv[settings.MACRO_TF]
    df_1h = ohlcv[settings.STRUCTURE_TF]
    df_15m = ohlcv[settings.SETUP_TF]
    df_5m = ohlcv[settings.ENTRY_TF]

    results: dict[str, Any] = {
        "ema": {},
        "rsi": {},
        "macd": {},
        "atr": {},
        "volume": {},
        "adx": {},
        "price": {},
    }

    # ── 4H Indicators (Macro Bias) ──
    if len(df_4h) >= 200:
        ema_50_4h = calc_ema(df_4h["close"], 50)
        ema_200_4h = calc_ema(df_4h["close"], 200)
        results["ema"]["4h_50"] = _safe_last(ema_50_4h)
        results["ema"]["4h_200"] = _safe_last(ema_200_4h)
        results["ema"]["4h_50_series"] = ema_50_4h
        results["ema"]["4h_200_series"] = ema_200_4h
        results["price"]["4h_close"] = df_4h["close"]
        results["price"]["4h_high"] = df_4h["high"]
        results["price"]["4h_low"] = df_4h["low"]
        results["price"]["4h_open"] = df_4h["open"]
    else:
        logger.warning(f"4H data has only {len(df_4h)} candles (need 200)")
        results["ema"]["4h_50"] = 0.0
        results["ema"]["4h_200"] = 0.0

    # ── 1H Indicators (Structure) ──
    if len(df_1h) >= 50:
        results["ema"]["1h_21"] = _safe_last(calc_ema(df_1h["close"], 21))
        results["ema"]["1h_50"] = _safe_last(calc_ema(df_1h["close"], 50))
        results["price"]["1h_close"] = df_1h["close"]
        results["price"]["1h_high"] = df_1h["high"]
        results["price"]["1h_low"] = df_1h["low"]
        results["price"]["1h_open"] = df_1h["open"]

        # ATR on 1H (for OB detection)
        atr_1h = calc_atr(df_1h["high"], df_1h["low"], df_1h["close"])
        results["atr"]["1h_series"] = atr_1h
        results["atr"]["1h_current"] = _safe_last(atr_1h)

        # ADX on 1H (mandatory gate)
        adx_data = calc_adx(df_1h["high"], df_1h["low"], df_1h["close"])
        results["adx"]["1h_current"] = _safe_last(adx_data["adx"])
    else:
        logger.warning(f"1H data has only {len(df_1h)} candles (need 50)")
        results["ema"]["1h_21"] = 0.0
        results["ema"]["1h_50"] = 0.0
        results["adx"]["1h_current"] = 0.0

    # ── 15m Indicators (Setup) ──
    if len(df_15m) >= 50:
        results["ema"]["15m_9"] = _safe_last(calc_ema(df_15m["close"], 9))
        results["ema"]["15m_21"] = _safe_last(calc_ema(df_15m["close"], 21))

        atr_15m = calc_atr(df_15m["high"], df_15m["low"], df_15m["close"])
        results["atr"]["15m_current"] = _safe_last(atr_15m)
        results["atr"]["15m_series"] = atr_15m

        vol_sma_15m = calc_volume_sma(df_15m["volume"])
        vol_ratio_15m = calc_volume_ratio(df_15m["volume"])
        results["volume"]["15m_ratio"] = _safe_last(vol_ratio_15m)
        results["volume"]["15m_current"] = _safe_last(df_15m["volume"])

        # RSI on 15m — full series needed for divergence detection in smc.py
        rsi_15m = calc_rsi(df_15m["close"])
        results["rsi"]["15m_series"] = rsi_15m

        results["price"]["15m_close"] = df_15m["close"]
        results["price"]["15m_high"] = df_15m["high"]
        results["price"]["15m_low"] = df_15m["low"]
    else:
        logger.warning(f"15m data has only {len(df_15m)} candles (need 50)")
        results["ema"]["15m_9"] = 0.0
        results["ema"]["15m_21"] = 0.0
        results["atr"]["15m_current"] = 0.0
        results["volume"]["15m_ratio"] = 0.0
        results["volume"]["15m_current"] = 0.0

    # ── 5m Indicators (Entry Trigger) ──
    if len(df_5m) >= 30:
        results["ema"]["5m_9"] = _safe_last(calc_ema(df_5m["close"], 9))
        results["ema"]["5m_21"] = _safe_last(calc_ema(df_5m["close"], 21))

        rsi_5m = calc_rsi(df_5m["close"])
        results["rsi"]["5m_current"] = _safe_last(rsi_5m)
        results["rsi"]["5m_last3"] = _safe_last_n(rsi_5m, 3)

        _, _, histogram_5m = calc_macd(df_5m["close"])
        results["macd"]["histogram_last3"] = _safe_last_n(histogram_5m, 3)

        atr_5m = calc_atr(df_5m["high"], df_5m["low"], df_5m["close"])
        results["atr"]["5m_current"] = _safe_last(atr_5m)

        vol_sma_5m = calc_volume_sma(df_5m["volume"])
        vol_ratio_5m = calc_volume_ratio(df_5m["volume"])
        results["volume"]["5m_ratio"] = _safe_last(vol_ratio_5m)
        results["volume"]["5m_current"] = _safe_last(df_5m["volume"])

        results["price"]["5m_close"] = df_5m["close"]
        results["price"]["5m_high"] = df_5m["high"]
        results["price"]["5m_low"] = df_5m["low"]
        results["price"]["5m_open"] = df_5m["open"]
    else:
        logger.warning(f"5m data has only {len(df_5m)} candles (need 30)")
        results["rsi"]["5m_current"] = 0.0
        results["rsi"]["5m_last3"] = [0.0, 0.0, 0.0]
        results["macd"]["histogram_last3"] = [0.0, 0.0, 0.0]
        results["atr"]["5m_current"] = 0.0
        results["volume"]["5m_ratio"] = 0.0
        results["volume"]["5m_current"] = 0.0

    return results
