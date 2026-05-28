"""
analysis/indicators.py — Pure pandas indicator calculations.

Implements EMA, RSI, MACD, ATR, and Volume SMA without any external
indicator library. All calculations use pandas built-in ewm/rolling.

Every function is stateless: takes a Series or DataFrame in, returns
computed values out. No side effects.
"""

from typing import Any

import pandas as pd

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── EMA (Exponential Moving Average) ─────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

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
    Calculate Average True Range.

    True Range = max of:
      - High - Low
      - abs(High - Previous Close)
      - abs(Low - Previous Close)

    ATR = EMA of True Range over the specified period.

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

    atr = true_range.ewm(span=period, adjust=False).mean()

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


# ─── Bulk Indicator Calculation ───────────────────────────────────────────

def calculate_all_indicators(
    ohlcv: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Calculate all indicators for a single coin across all timeframes.

    Args:
        ohlcv: Dict of timeframe -> OHLCV DataFrame.
              Expected keys: '4h', '1h', '15m', '5m'.

    Returns:
        Dict of indicator results keyed by descriptive names.
        Example keys:
          '4h_ema_50', '4h_ema_200', '5m_rsi', '5m_macd_line',
          '15m_atr', '5m_volume_ratio', etc.
    """
    results: dict[str, Any] = {}

    # ── 4H Indicators (Macro Bias) ──
    df_4h = ohlcv.get(settings.MACRO_TF)
    if df_4h is not None and len(df_4h) >= 200:
        results["4h_ema_50"] = calc_ema(df_4h["close"], 50)
        results["4h_ema_200"] = calc_ema(df_4h["close"], 200)
        results["4h_close"] = df_4h["close"]
        results["4h_high"] = df_4h["high"]
        results["4h_low"] = df_4h["low"]

    # ── 1H Indicators (Structure) ──
    df_1h = ohlcv.get(settings.STRUCTURE_TF)
    if df_1h is not None and len(df_1h) >= 50:
        results["1h_ema_21"] = calc_ema(df_1h["close"], 21)
        results["1h_ema_50"] = calc_ema(df_1h["close"], 50)
        results["1h_close"] = df_1h["close"]
        results["1h_open"] = df_1h["open"]
        results["1h_high"] = df_1h["high"]
        results["1h_low"] = df_1h["low"]

    # ── 15m Indicators (Setup) ──
    df_15m = ohlcv.get(settings.SETUP_TF)
    if df_15m is not None and len(df_15m) >= 50:
        results["15m_ema_9"] = calc_ema(df_15m["close"], 9)
        results["15m_ema_21"] = calc_ema(df_15m["close"], 21)
        results["15m_atr"] = calc_atr(df_15m["high"], df_15m["low"], df_15m["close"])
        results["15m_atr_avg"] = results["15m_atr"].rolling(
            window=settings.ATR_PERIOD
        ).mean()
        results["15m_volume"] = df_15m["volume"]
        results["15m_volume_sma"] = calc_volume_sma(df_15m["volume"])
        results["15m_volume_ratio"] = calc_volume_ratio(df_15m["volume"])
        results["15m_close"] = df_15m["close"]
        results["15m_high"] = df_15m["high"]
        results["15m_low"] = df_15m["low"]

    # ── 5m Indicators (Entry Trigger) ──
    df_5m = ohlcv.get(settings.ENTRY_TF)
    if df_5m is not None and len(df_5m) >= 30:
        results["5m_ema_9"] = calc_ema(df_5m["close"], 9)
        results["5m_ema_21"] = calc_ema(df_5m["close"], 21)
        results["5m_rsi"] = calc_rsi(df_5m["close"])
        macd_line, signal_line, histogram = calc_macd(df_5m["close"])
        results["5m_macd_line"] = macd_line
        results["5m_macd_signal"] = signal_line
        results["5m_macd_histogram"] = histogram
        results["5m_atr"] = calc_atr(df_5m["high"], df_5m["low"], df_5m["close"])
        results["5m_volume"] = df_5m["volume"]
        results["5m_volume_sma"] = calc_volume_sma(df_5m["volume"])
        results["5m_volume_ratio"] = calc_volume_ratio(df_5m["volume"])
        results["5m_close"] = df_5m["close"]
        results["5m_open"] = df_5m["open"]
        results["5m_high"] = df_5m["high"]
        results["5m_low"] = df_5m["low"]

    return results
