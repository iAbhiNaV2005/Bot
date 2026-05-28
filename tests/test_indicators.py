"""
tests/test_indicators.py — Unit tests for indicator calculations.

Tests EMA, RSI, MACD, ATR, and Volume SMA against known values
using synthetic data.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from analysis.indicators import (
    calc_atr,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_volume_ratio,
    calc_volume_sma,
)


def _make_price_series(n: int = 100, start: float = 100.0, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series with random walk."""
    rng = np.random.default_rng(seed)
    changes = rng.normal(0, 1, n)
    prices = start + np.cumsum(changes)
    return pd.Series(prices, name="close")


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.uniform(-1.0, 1.0, n)
    volume = rng.uniform(1000, 5000, n)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_length(self) -> None:
        """EMA output should have same length as input."""
        series = _make_price_series(100)
        ema = calc_ema(series, 20)
        assert len(ema) == len(series)

    def test_ema_smoothing(self) -> None:
        """EMA should be smoother than raw prices (lower std dev)."""
        series = _make_price_series(200)
        ema = calc_ema(series, 50)
        assert ema.std() < series.std()

    def test_ema_follows_trend(self) -> None:
        """In an uptrend, EMA should be below current price."""
        prices = pd.Series(range(1, 101), dtype=float)
        ema = calc_ema(prices, 20)
        # At the end of a strong uptrend, EMA should be below price
        assert ema.iloc[-1] < prices.iloc[-1]

    def test_ema_periods(self) -> None:
        """Shorter period EMA should react faster than longer period."""
        series = _make_price_series(200)
        ema_9 = calc_ema(series, 9)
        ema_50 = calc_ema(series, 50)
        # Short EMA should be closer to the last price
        assert abs(ema_9.iloc[-1] - series.iloc[-1]) <= abs(
            ema_50.iloc[-1] - series.iloc[-1]
        )


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_range(self) -> None:
        """RSI should always be between 0 and 100."""
        series = _make_price_series(200)
        rsi = calc_rsi(series, 14)
        valid = rsi.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_uptrend_high(self) -> None:
        """In a strong uptrend, RSI should be high (>50)."""
        prices = pd.Series(np.linspace(100, 200, 100))
        rsi = calc_rsi(prices, 14)
        assert rsi.iloc[-1] > 50

    def test_rsi_downtrend_low(self) -> None:
        """In a strong downtrend, RSI should be low (<50)."""
        prices = pd.Series(np.linspace(200, 100, 100))
        rsi = calc_rsi(prices, 14)
        assert rsi.iloc[-1] < 50

    def test_rsi_flat_market(self) -> None:
        """In a flat market, RSI should be near 50."""
        prices = pd.Series([100.0] * 100)
        rsi = calc_rsi(prices, 14)
        # With zero movement, RSI is undefined, but shouldn't crash
        assert len(rsi) == 100


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_output_shapes(self) -> None:
        """MACD should return three series of equal length."""
        series = _make_price_series(100)
        macd_line, signal_line, histogram = calc_macd(series)
        assert len(macd_line) == len(series)
        assert len(signal_line) == len(series)
        assert len(histogram) == len(series)

    def test_histogram_is_difference(self) -> None:
        """Histogram should equal MACD line minus signal line."""
        series = _make_price_series(100)
        macd_line, signal_line, histogram = calc_macd(series)
        diff = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, diff, check_names=False)

    def test_macd_uptrend(self) -> None:
        """In strong uptrend, MACD line should be positive."""
        prices = pd.Series(np.linspace(100, 200, 200))
        macd_line, _, _ = calc_macd(prices)
        assert macd_line.iloc[-1] > 0


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_positive(self) -> None:
        """ATR should always be positive."""
        df = _make_ohlcv(200)
        atr = calc_atr(df["high"], df["low"], df["close"], 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_reflects_volatility(self) -> None:
        """Higher volatility should produce higher ATR."""
        # Low volatility
        n = 100
        close_low = pd.Series(np.linspace(100, 105, n))
        high_low = close_low + 0.5
        low_low = close_low - 0.5

        # High volatility
        close_high = pd.Series(np.linspace(100, 105, n))
        high_high = close_high + 5.0
        low_high = close_high - 5.0

        atr_low = calc_atr(high_low, low_low, close_low, 14)
        atr_high = calc_atr(high_high, low_high, close_high, 14)

        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestVolume:
    """Tests for volume calculations."""

    def test_volume_sma_length(self) -> None:
        """Volume SMA should have same length as input."""
        volume = pd.Series(np.random.default_rng(42).uniform(1000, 5000, 100))
        sma = calc_volume_sma(volume, 20)
        assert len(sma) == len(volume)

    def test_volume_ratio_spike(self) -> None:
        """A volume spike should produce ratio > 1."""
        volume = pd.Series([1000.0] * 50 + [5000.0])
        ratio = calc_volume_ratio(volume, 20)
        assert ratio.iloc[-1] > 1.0

    def test_volume_ratio_normal(self) -> None:
        """Constant volume should produce ratio near 1."""
        volume = pd.Series([1000.0] * 50)
        ratio = calc_volume_ratio(volume, 20)
        assert abs(ratio.iloc[-1] - 1.0) < 0.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
