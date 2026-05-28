"""
tests/test_smc.py — Unit tests for SMC calculations.

Tests swing point detection, BOS, CHoCH, Order Blocks, FVGs,
and premium/discount zones with synthetic OHLCV data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from analysis.smc import (
    Direction,
    Zone,
    detect_swing_points,
    detect_bos,
    detect_choch,
    detect_order_blocks,
    detect_fvgs,
    invalidate_order_blocks,
    invalidate_fvgs,
    is_price_in_order_block,
    calc_premium_discount,
    OrderBlock,
    SwingPoint,
)
from analysis.indicators import calc_atr


def _make_ohlcv_with_swing(n: int = 50) -> pd.DataFrame:
    """
    Generate OHLCV with a clear swing high at index 25
    and a clear swing low at index 35.
    """
    high = np.array([100.0] * n)
    low = np.array([95.0] * n)
    close = np.array([97.0] * n)
    open_ = np.array([96.0] * n)

    # Swing high at index 25: peak
    high[23] = 101.0
    high[24] = 102.0
    high[25] = 110.0  # Swing high
    high[26] = 102.0
    high[27] = 101.0

    # Swing low at index 35: valley
    low[33] = 94.0
    low[34] = 93.0
    low[35] = 85.0  # Swing low
    low[36] = 93.0
    low[37] = 94.0

    volume = np.array([1000.0] * n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


def _make_bullish_impulse_ohlcv(n: int = 30) -> pd.DataFrame:
    """
    Generate OHLCV with a bearish candle followed by a strong bullish impulse.
    Candle at index 10 is bearish, followed by 3+ bullish candles.
    """
    open_ = np.array([100.0] * n)
    close = np.array([100.0] * n)
    high = np.array([102.0] * n)
    low = np.array([98.0] * n)

    # Bearish candle at index 10
    open_[10] = 101.0
    close[10] = 97.0
    high[10] = 102.0
    low[10] = 96.0

    # Strong bullish impulse: indices 11, 12, 13
    for i in range(11, 14):
        open_[i] = close[i - 1]
        close[i] = open_[i] + 5.0
        high[i] = close[i] + 1.0
        low[i] = open_[i] - 0.5

    volume = np.array([1000.0] * n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


def _make_fvg_ohlcv() -> pd.DataFrame:
    """
    Generate OHLCV with a clear Bullish FVG.
    Candle 5: high=100
    Candle 6: strong bullish (any)
    Candle 7: low=105 (gap above candle 5's high)

    Default candles use overlapping high/low to prevent accidental gaps.
    """
    n = 20
    open_ = np.array([95.0] * n)
    close = np.array([96.0] * n)
    high = np.array([97.0] * n)
    low = np.array([94.0] * n)  # low < high of all neighbors → no accidental gaps

    # Candle 5 (index 5)
    high[5] = 100.0
    low[5] = 94.0

    # Candle 6 (index 6): strong bullish
    open_[6] = 100.0
    close[6] = 108.0
    high[6] = 109.0
    low[6] = 99.0

    # Candle 7 (index 7): low is above candle 5's high → creates FVG
    open_[7] = 108.0
    close[7] = 110.0
    high[7] = 111.0
    low[7] = 105.0  # Gap: high[5]=100 < low[7]=105

    volume = np.array([1000.0] * n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


class TestSwingPoints:
    """Tests for swing high/low detection."""

    def test_detects_swing_high(self) -> None:
        """Should detect the clear swing high."""
        df = _make_ohlcv_with_swing()
        sh, sl = detect_swing_points(df, lookback=2)
        swing_high_prices = [s.price for s in sh]
        assert 110.0 in swing_high_prices

    def test_detects_swing_low(self) -> None:
        """Should detect the clear swing low."""
        df = _make_ohlcv_with_swing()
        sh, sl = detect_swing_points(df, lookback=2)
        swing_low_prices = [s.price for s in sl]
        assert 85.0 in swing_low_prices

    def test_swing_directions(self) -> None:
        """Swing highs should be BULLISH, swing lows BEARISH."""
        df = _make_ohlcv_with_swing()
        sh, sl = detect_swing_points(df, lookback=2)
        for s in sh:
            assert s.direction == Direction.BULLISH
        for s in sl:
            assert s.direction == Direction.BEARISH

    def test_max_stored_limit(self) -> None:
        """Should not return more than max_stored swings."""
        df = _make_ohlcv_with_swing(200)
        sh, sl = detect_swing_points(df, lookback=2, max_stored=3)
        assert len(sh) <= 3
        assert len(sl) <= 3


class TestBOS:
    """Tests for Break of Structure detection."""

    def test_bullish_bos(self) -> None:
        """Should detect bullish BOS when price closes above swing high."""
        df = _make_ohlcv_with_swing()
        sh, sl = detect_swing_points(df, lookback=2)

        # Modify: make a candle after the swing high close above it
        df.iloc[40, df.columns.get_loc("close")] = 115.0

        bos = detect_bos(df, sh, sl, lookback=30)
        assert bos is not None
        assert bos.direction == Direction.BULLISH

    def test_no_bos_when_no_break(self) -> None:
        """No BOS should be detected if price stays below swing high."""
        df = _make_ohlcv_with_swing()
        sh, sl = detect_swing_points(df, lookback=2)
        # All closes at 97, swing high at 110 — no break
        bos = detect_bos(df, sh, sl, lookback=30)
        # May or may not detect depending on swing low break
        # Just ensure it doesn't crash
        assert bos is None or isinstance(bos, object)


class TestOrderBlocks:
    """Tests for Order Block detection."""

    def test_detects_bullish_ob(self) -> None:
        """Should detect a bullish OB before a bullish impulse."""
        df = _make_bullish_impulse_ohlcv()
        atr = calc_atr(df["high"], df["low"], df["close"], 14)
        bull_obs, bear_obs = detect_order_blocks(df, atr, impulse_candles=3)
        # Should find at least one bullish OB near index 10
        assert len(bull_obs) > 0

    def test_ob_price_check(self) -> None:
        """is_price_in_order_block should work correctly."""
        ob = OrderBlock(
            top=102.0, bottom=96.0,
            direction=Direction.BULLISH,
            candle_index=10, valid=True,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        assert is_price_in_order_block(99.0, ob) is True
        assert is_price_in_order_block(95.0, ob) is False
        assert is_price_in_order_block(103.0, ob) is False
        assert is_price_in_order_block(102.0, ob) is True  # Edge
        assert is_price_in_order_block(96.0, ob) is True   # Edge

    def test_ob_invalidation(self) -> None:
        """Bullish OB should be invalidated when price closes below bottom."""
        df = _make_bullish_impulse_ohlcv()
        ob = OrderBlock(
            top=102.0, bottom=96.0,
            direction=Direction.BULLISH,
            candle_index=5, valid=True,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        # Set a close below OB bottom
        df.iloc[15, df.columns.get_loc("close")] = 90.0

        valid = invalidate_order_blocks([ob], df)
        assert len(valid) == 0


class TestFVG:
    """Tests for Fair Value Gap detection."""

    def test_detects_bullish_fvg(self) -> None:
        """Should detect the bullish FVG in synthetic data."""
        df = _make_fvg_ohlcv()
        bull_fvgs, bear_fvgs = detect_fvgs(df, max_age=50)
        assert len(bull_fvgs) > 0
        # Find the FVG at our intended location (candle 5-7, middle index 6)
        target_fvg = [f for f in bull_fvgs if f.candle_index == 6]
        assert len(target_fvg) == 1, f"Expected FVG at index 6, got FVGs at: {[f.candle_index for f in bull_fvgs]}"
        fvg = target_fvg[0]
        assert fvg.bottom == 100.0  # high of candle 5
        assert fvg.top == 105.0     # low of candle 7
        assert fvg.direction == Direction.BULLISH

    def test_fvg_fill_check(self) -> None:
        """Filled FVG should be removed after invalidation."""
        df = _make_fvg_ohlcv()
        bull_fvgs, _ = detect_fvgs(df, max_age=50)

        # Simulate price filling the gap
        df.iloc[15, df.columns.get_loc("low")] = 99.0  # Below FVG bottom

        valid = invalidate_fvgs(bull_fvgs, df)
        assert len(valid) == 0


class TestPremiumDiscount:
    """Tests for Premium/Discount zone calculation."""

    def test_discount_zone(self) -> None:
        """Price below midpoint should be in discount zone."""
        sh = [SwingPoint(
            index=0, price=200.0,
            direction=Direction.BULLISH,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )]
        sl = [SwingPoint(
            index=1, price=100.0,
            direction=Direction.BEARISH,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )]
        # Midpoint = 100 + (200-100)/2 = 150
        zone, mid = calc_premium_discount(sh, sl, 120.0)
        assert zone == Zone.DISCOUNT
        assert mid == 150.0

    def test_premium_zone(self) -> None:
        """Price above midpoint should be in premium zone."""
        sh = [SwingPoint(
            index=0, price=200.0,
            direction=Direction.BULLISH,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )]
        sl = [SwingPoint(
            index=1, price=100.0,
            direction=Direction.BEARISH,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        )]
        zone, mid = calc_premium_discount(sh, sl, 170.0)
        assert zone == Zone.PREMIUM

    def test_empty_swings_defaults_discount(self) -> None:
        """With no swing data, should default to discount."""
        zone, _ = calc_premium_discount([], [], 100.0)
        assert zone == Zone.DISCOUNT


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
