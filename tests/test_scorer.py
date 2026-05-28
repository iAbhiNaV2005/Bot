"""
tests/test_scorer.py — Unit tests for the scoring system.

Tests mandatory condition checks and additive scoring logic
using mock data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from analysis.scorer import (
    check_mandatory_long,
    check_mandatory_short,
    calculate_score_long,
    calculate_score_short,
)
from analysis.smc import BOS, CHoCH, Direction, FVG, OrderBlock, Zone


def _make_bullish_indicators() -> dict:
    """Create indicator dict that passes all long mandatory conditions."""
    n = 200
    # 4H EMA 50 > EMA 200
    ema_50 = pd.Series([110.0] * n)
    ema_200 = pd.Series([100.0] * n)
    close_4h = pd.Series([115.0, 114.0, 116.0] + [115.0] * (n - 3))

    # 5m RSI between 28-45 and rising
    rsi_vals = [30.0] * (n - 3) + [33.0, 36.0, 40.0]
    rsi = pd.Series(rsi_vals)

    # 5m MACD
    macd_line = pd.Series([-2.0] * (n - 2) + [-1.5, -1.0])
    macd_signal = pd.Series([-1.0] * n)

    # 15m ATR
    atr_15m = pd.Series([2.0] * n)
    atr_avg = pd.Series([1.5] * n)

    # Volume
    vol_15m = pd.Series([1.5] * n)
    vol_5m = pd.Series([2.0] * n)

    return {
        "4h_ema_50": ema_50,
        "4h_ema_200": ema_200,
        "4h_close": close_4h,
        "5m_rsi": rsi,
        "5m_macd_line": macd_line,
        "5m_macd_signal": macd_signal,
        "15m_atr": atr_15m,
        "15m_atr_avg": atr_avg,
        "15m_volume_ratio": vol_15m,
        "5m_volume_ratio": vol_5m,
    }


def _make_bullish_smc() -> dict:
    """Create SMC dict that passes all long mandatory conditions."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return {
        "1h_bos": BOS(direction=Direction.BULLISH, candle_index=180, price=100.0, timestamp=ts),
        "1h_bull_obs": [
            OrderBlock(top=116.0, bottom=113.0, direction=Direction.BULLISH,
                       candle_index=170, valid=True, timestamp=ts),
        ],
        "1h_bear_obs": [],
        "15m_bull_fvgs": [
            FVG(top=114.0, bottom=112.0, direction=Direction.BULLISH,
                candle_index=190, filled=False, timestamp=ts),
        ],
        "15m_bear_fvgs": [],
        "zone": Zone.DISCOUNT,
        "zone_midpoint": 120.0,
        "1h_choch": CHoCH(direction=Direction.BULLISH, candle_index=175, price=99.0, timestamp=ts),
    }


def _make_bullish_market() -> dict:
    """Create market data dict for bullish scenario."""
    return {
        "funding_rate": -0.001,
        "oi_increasing": True,
        "oi_change": 4.2,
        "ls_ratio": 0.84,
    }


class TestMandatoryLong:
    """Tests for long signal mandatory conditions."""

    def test_all_pass(self) -> None:
        """All mandatory conditions should pass with correct data."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        passed, failed = check_mandatory_long(indicators, smc, market, 115.0)
        assert passed, f"Expected pass but failed on: {failed}"

    def test_fails_ema_cross(self) -> None:
        """Should fail when EMA 50 < EMA 200."""
        indicators = _make_bullish_indicators()
        indicators["4h_ema_50"] = pd.Series([90.0] * 200)  # Below 200
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        passed, failed = check_mandatory_long(indicators, smc, market, 115.0)
        assert not passed
        assert "EMA" in failed

    def test_fails_no_bos(self) -> None:
        """Should fail when no bullish BOS detected."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        smc["1h_bos"] = None
        market = _make_bullish_market()
        passed, failed = check_mandatory_long(indicators, smc, market, 115.0)
        assert not passed
        assert "BOS" in failed

    def test_fails_not_in_ob(self) -> None:
        """Should fail when price is outside Order Block."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        # Price 200 is way outside the OB [113, 116]
        passed, failed = check_mandatory_long(indicators, smc, market, 200.0)
        assert not passed

    def test_fails_funding_too_high(self) -> None:
        """Should fail when funding rate is too positive."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        market["funding_rate"] = 0.001  # +0.1%, above 0.05% limit
        passed, failed = check_mandatory_long(indicators, smc, market, 115.0)
        assert not passed
        assert "Funding" in failed

    def test_fails_rsi_out_of_range(self) -> None:
        """Should fail when RSI is outside 28-45 range."""
        indicators = _make_bullish_indicators()
        indicators["5m_rsi"] = pd.Series([60.0, 62.0, 65.0] + [60.0] * 197)
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        passed, failed = check_mandatory_long(indicators, smc, market, 115.0)
        assert not passed
        assert "RSI" in failed


class TestScoringLong:
    """Tests for long signal scoring."""

    def test_high_confidence_score(self) -> None:
        """Full bullish setup should achieve high confidence."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        breakdown = calculate_score_long(indicators, smc, market, 115.0)
        assert breakdown.mandatory_passed
        assert breakdown.total_score >= 8  # Minimum
        assert len(breakdown.details) > 0

    def test_score_breakdown_has_all_conditions(self) -> None:
        """Score breakdown should track all 10 scoring conditions."""
        indicators = _make_bullish_indicators()
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        breakdown = calculate_score_long(indicators, smc, market, 115.0)
        assert len(breakdown.details) == 10

    def test_failed_mandatory_returns_zero_score(self) -> None:
        """If mandatory fails, score should be 0."""
        indicators = _make_bullish_indicators()
        indicators["4h_ema_50"] = pd.Series([90.0] * 200)  # Fail EMA
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        breakdown = calculate_score_long(indicators, smc, market, 115.0)
        assert not breakdown.mandatory_passed
        assert breakdown.total_score == 0


class TestMandatoryShort:
    """Tests for short signal mandatory conditions."""

    def test_fails_when_bullish(self) -> None:
        """Short mandatory should fail when market is bullish."""
        indicators = _make_bullish_indicators()  # Bullish setup
        smc = _make_bullish_smc()
        market = _make_bullish_market()
        passed, failed = check_mandatory_short(indicators, smc, market, 115.0)
        assert not passed  # Should fail because EMA 50 > EMA 200


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
