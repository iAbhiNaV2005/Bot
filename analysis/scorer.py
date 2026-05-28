"""
analysis/scorer.py — Signal scoring logic.

Implements the 18-point scoring system from the strategy spec.
Separates MANDATORY conditions (instant discard if false) from
SCORING conditions (additive points).

A signal must pass ALL mandatory conditions AND score >= MIN_SCORE_TO_SIGNAL
to be sent.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from analysis.smc import (
    BOS, CHoCH, Direction, FVG, OrderBlock, Zone,
    is_price_in_order_block,
)
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoreBreakdown:
    """
    Detailed breakdown of a signal's score.

    Attributes:
        mandatory_passed: Whether all mandatory conditions passed.
        failed_mandatory: Name of the first failed mandatory condition (if any).
        total_score: Sum of all scoring points.
        max_score: Maximum possible score.
        details: Dict of condition name -> (passed: bool, points: int, info: str).
        confidence: 'HIGH' if score >= HIGH_CONFIDENCE_SCORE, else 'STANDARD'.
    """
    mandatory_passed: bool = False
    failed_mandatory: str = ""
    total_score: int = 0
    max_score: int = settings.MAX_POSSIBLE_SCORE
    details: dict[str, tuple[bool, int, str]] = field(default_factory=dict)
    confidence: str = "STANDARD"


def check_mandatory_long(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> tuple[bool, str]:
    """
    Check all MANDATORY conditions for a LONG signal.

    Every condition must be TRUE or the signal is instantly rejected.

    Args:
        indicators: Computed indicator values from indicators.py.
        smc: SMC analysis results from smc.py.
        market: Market data (funding rate, OI, etc.).
        current_price: Current market price.

    Returns:
        Tuple of (all_passed, failed_condition_name).
        If all_passed is True, failed_condition_name is empty.
    """
    # 1. 4H EMA 50 > EMA 200
    ema_50 = indicators.get("4h_ema_50")
    ema_200 = indicators.get("4h_ema_200")
    if ema_50 is None or ema_200 is None:
        return False, "4H EMA data missing"
    if ema_50.iloc[-1] <= ema_200.iloc[-1]:
        return False, "4H EMA 50 <= EMA 200"

    # 2. Current price above EMA 50 on 4H
    if current_price <= ema_50.iloc[-1]:
        return False, "Price below 4H EMA 50"

    # 3. Last 3 candles on 4H not all bearish
    close_4h = indicators.get("4h_close")
    open_4h = indicators.get("4h_close")  # We need open, use from ohlcv
    if close_4h is not None and len(close_4h) >= 3:
        # Check using close vs previous close as proxy
        last_3 = close_4h.iloc[-3:]
        all_bearish = all(last_3.diff().dropna() < 0)
        if all_bearish:
            return False, "4H last 3 candles all bearish"

    # 4. 1H Bullish BOS detected in last 20 candles
    bos: Optional[BOS] = smc.get("1h_bos")
    if bos is None or bos.direction != Direction.BULLISH:
        return False, "No 1H Bullish BOS"

    # 5. Price inside 1H Bullish Order Block
    bull_obs: list[OrderBlock] = smc.get("1h_bull_obs", [])
    in_ob = False
    active_ob: Optional[OrderBlock] = None
    for ob in bull_obs:
        if is_price_in_order_block(current_price, ob):
            in_ob = True
            active_ob = ob
            break
    if not in_ob:
        return False, "Price not in Bullish OB"

    # 6. Funding rate < +0.05%
    funding_rate = market.get("funding_rate")
    if funding_rate is not None and funding_rate >= settings.FUNDING_RATE_LONG_MAX:
        return False, "Funding rate too high for long"

    # 7. 5m RSI between 28-45 and rising
    rsi = indicators.get("5m_rsi")
    if rsi is None or len(rsi) < 3:
        return False, "5m RSI data missing"
    last_rsi = rsi.iloc[-1]
    if not (settings.RSI_LONG_MIN <= last_rsi <= settings.RSI_LONG_MAX):
        return False, f"5m RSI {last_rsi:.1f} outside 28-45 range"
    # RSI rising: last 3 values increasing
    rsi_rising = rsi.iloc[-3] < rsi.iloc[-2] < rsi.iloc[-1]
    if not rsi_rising:
        return False, "5m RSI not rising"

    return True, ""


def check_mandatory_short(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> tuple[bool, str]:
    """
    Check all MANDATORY conditions for a SHORT signal.

    Mirror of check_mandatory_long with reversed conditions.

    Args:
        indicators: Computed indicator values.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.

    Returns:
        Tuple of (all_passed, failed_condition_name).
    """
    # 1. 4H EMA 50 < EMA 200
    ema_50 = indicators.get("4h_ema_50")
    ema_200 = indicators.get("4h_ema_200")
    if ema_50 is None or ema_200 is None:
        return False, "4H EMA data missing"
    if ema_50.iloc[-1] >= ema_200.iloc[-1]:
        return False, "4H EMA 50 >= EMA 200"

    # 2. Price below EMA 50
    if current_price >= ema_50.iloc[-1]:
        return False, "Price above 4H EMA 50"

    # 3. Last 3 candles on 4H not all bullish
    close_4h = indicators.get("4h_close")
    if close_4h is not None and len(close_4h) >= 3:
        last_3 = close_4h.iloc[-3:]
        all_bullish = all(last_3.diff().dropna() > 0)
        if all_bullish:
            return False, "4H last 3 candles all bullish"

    # 4. 1H Bearish BOS
    bos: Optional[BOS] = smc.get("1h_bos")
    if bos is None or bos.direction != Direction.BEARISH:
        return False, "No 1H Bearish BOS"

    # 5. Price inside Bearish OB
    bear_obs: list[OrderBlock] = smc.get("1h_bear_obs", [])
    in_ob = False
    for ob in bear_obs:
        if is_price_in_order_block(current_price, ob):
            in_ob = True
            break
    if not in_ob:
        return False, "Price not in Bearish OB"

    # 6. Funding rate > -0.05%
    funding_rate = market.get("funding_rate")
    if funding_rate is not None and funding_rate <= settings.FUNDING_RATE_SHORT_MIN:
        return False, "Funding rate too negative for short"

    # 7. 5m RSI between 55-72 and falling
    rsi = indicators.get("5m_rsi")
    if rsi is None or len(rsi) < 3:
        return False, "5m RSI data missing"
    last_rsi = rsi.iloc[-1]
    if not (settings.RSI_SHORT_MIN <= last_rsi <= settings.RSI_SHORT_MAX):
        return False, f"5m RSI {last_rsi:.1f} outside 55-72 range"
    rsi_falling = rsi.iloc[-3] > rsi.iloc[-2] > rsi.iloc[-1]
    if not rsi_falling:
        return False, "5m RSI not falling"

    return True, ""


def calculate_score_long(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> ScoreBreakdown:
    """
    Calculate the full scoring breakdown for a LONG signal.

    Checks mandatory conditions first, then calculates additive score
    from scoring conditions.

    Args:
        indicators: Computed indicator values.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.

    Returns:
        ScoreBreakdown with full details.
    """
    breakdown = ScoreBreakdown()

    # ── Mandatory Check ──
    passed, failed = check_mandatory_long(indicators, smc, market, current_price)
    breakdown.mandatory_passed = passed
    breakdown.failed_mandatory = failed

    if not passed:
        return breakdown

    # ── Scoring Conditions ──
    score = 0
    details: dict[str, tuple[bool, int, str]] = {}

    # 1. 15m FVG present below price (+2)
    bull_fvgs: list[FVG] = smc.get("15m_bull_fvgs", [])
    fvg_present = any(fvg.top <= current_price for fvg in bull_fvgs)
    pts = settings.SCORE_FVG_PRESENT if fvg_present else 0
    score += pts
    details["15m_fvg"] = (fvg_present, pts, "Bullish FVG below price")

    # 2. Price in discount zone (+2)
    zone: Zone = smc.get("zone", Zone.PREMIUM)
    in_discount = zone == Zone.DISCOUNT
    pts = settings.SCORE_DISCOUNT_ZONE if in_discount else 0
    score += pts
    details["discount_zone"] = (in_discount, pts, "Price in discount zone")

    # 3. 15m volume above 20-period average (+1)
    vol_ratio_15m = indicators.get("15m_volume_ratio")
    vol_above = False
    if vol_ratio_15m is not None and len(vol_ratio_15m) > 0:
        vol_above = vol_ratio_15m.iloc[-1] > 1.0
    pts = settings.SCORE_15M_VOLUME if vol_above else 0
    score += pts
    details["15m_volume"] = (
        vol_above, pts,
        f"15m vol ratio: {vol_ratio_15m.iloc[-1]:.1f}x" if vol_ratio_15m is not None and len(vol_ratio_15m) > 0 else "N/A"
    )

    # 4. 5m volume 1.5x above average (+2)
    vol_ratio_5m = indicators.get("5m_volume_ratio")
    vol_spike = False
    if vol_ratio_5m is not None and len(vol_ratio_5m) > 0:
        vol_spike = vol_ratio_5m.iloc[-1] >= settings.VOLUME_RATIO_MIN
    pts = settings.SCORE_5M_VOLUME if vol_spike else 0
    score += pts
    details["5m_volume"] = (
        vol_spike, pts,
        f"5m vol ratio: {vol_ratio_5m.iloc[-1]:.1f}x" if vol_ratio_5m is not None and len(vol_ratio_5m) > 0 else "N/A"
    )

    # 5. OI increasing last 2 hours (+2)
    oi_increasing = market.get("oi_increasing", False)
    oi_change = market.get("oi_change", 0.0)
    pts = settings.SCORE_OI_INCREASING if oi_increasing else 0
    score += pts
    details["oi_increasing"] = (oi_increasing, pts, f"OI change: {oi_change:+.1f}%")

    # 6. 1H CHoCH detected (+3)
    choch: Optional[CHoCH] = smc.get("1h_choch")
    has_choch = choch is not None and choch.direction == Direction.BULLISH
    pts = settings.SCORE_CHOCH if has_choch else 0
    score += pts
    details["choch"] = (has_choch, pts, "1H Bullish CHoCH")

    # 7. 5m MACD pre-crossover (+1)
    macd_line = indicators.get("5m_macd_line")
    macd_signal = indicators.get("5m_macd_signal")
    macd_precross = False
    if macd_line is not None and macd_signal is not None and len(macd_line) >= 2:
        below_signal = macd_line.iloc[-1] < macd_signal.iloc[-1]
        gap_narrowing = abs(macd_line.iloc[-1] - macd_signal.iloc[-1]) < abs(
            macd_line.iloc[-2] - macd_signal.iloc[-2]
        )
        macd_precross = below_signal and gap_narrowing
    pts = settings.SCORE_MACD_PRECROSS if macd_precross else 0
    score += pts
    details["macd_precross"] = (macd_precross, pts, "MACD pre-crossover (bullish)")

    # 8. L/S ratio < 1.0 (contrarian) (+2)
    ls_ratio = market.get("ls_ratio")
    ls_contrarian = ls_ratio is not None and ls_ratio < 1.0
    pts = settings.SCORE_LS_RATIO if ls_contrarian else 0
    score += pts
    details["ls_ratio"] = (
        ls_contrarian, pts,
        f"L/S ratio: {ls_ratio:.2f}" if ls_ratio else "N/A"
    )

    # 9. Funding rate negative (+2)
    funding = market.get("funding_rate")
    funding_neg = funding is not None and funding < 0
    pts = settings.SCORE_FUNDING_BONUS if funding_neg else 0
    score += pts
    details["funding_bonus"] = (
        funding_neg, pts,
        f"Funding: {funding * 100:.4f}%" if funding else "N/A"
    )

    # 10. 4H price above EMA 200 by > 2% (+1)
    ema_200 = indicators.get("4h_ema_200")
    strong_bias = False
    if ema_200 is not None:
        pct_above = ((current_price - ema_200.iloc[-1]) / ema_200.iloc[-1]) * 100
        strong_bias = pct_above > 2.0
    pts = settings.SCORE_STRONG_BIAS if strong_bias else 0
    score += pts
    details["strong_bias"] = (strong_bias, pts, "4H strong bullish bias (>2% above EMA 200)")

    # ── Finalize ──
    breakdown.total_score = score
    breakdown.details = details
    breakdown.confidence = (
        "HIGH" if score >= settings.HIGH_CONFIDENCE_SCORE else "STANDARD"
    )

    return breakdown


def calculate_score_short(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> ScoreBreakdown:
    """
    Calculate the full scoring breakdown for a SHORT signal.

    Mirror of calculate_score_long with reversed conditions.

    Args:
        indicators: Computed indicator values.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.

    Returns:
        ScoreBreakdown with full details.
    """
    breakdown = ScoreBreakdown()

    # ── Mandatory Check ──
    passed, failed = check_mandatory_short(indicators, smc, market, current_price)
    breakdown.mandatory_passed = passed
    breakdown.failed_mandatory = failed

    if not passed:
        return breakdown

    # ── Scoring Conditions ──
    score = 0
    details: dict[str, tuple[bool, int, str]] = {}

    # 1. 15m Bearish FVG present above price (+2)
    bear_fvgs: list[FVG] = smc.get("15m_bear_fvgs", [])
    fvg_present = any(fvg.bottom >= current_price for fvg in bear_fvgs)
    pts = settings.SCORE_FVG_PRESENT if fvg_present else 0
    score += pts
    details["15m_fvg"] = (fvg_present, pts, "Bearish FVG above price")

    # 2. Price in premium zone (+2)
    zone: Zone = smc.get("zone", Zone.DISCOUNT)
    in_premium = zone == Zone.PREMIUM
    pts = settings.SCORE_DISCOUNT_ZONE if in_premium else 0
    score += pts
    details["premium_zone"] = (in_premium, pts, "Price in premium zone")

    # 3. 15m volume above average (+1)
    vol_ratio_15m = indicators.get("15m_volume_ratio")
    vol_above = False
    if vol_ratio_15m is not None and len(vol_ratio_15m) > 0:
        vol_above = vol_ratio_15m.iloc[-1] > 1.0
    pts = settings.SCORE_15M_VOLUME if vol_above else 0
    score += pts
    details["15m_volume"] = (
        vol_above, pts,
        f"15m vol ratio: {vol_ratio_15m.iloc[-1]:.1f}x" if vol_ratio_15m is not None and len(vol_ratio_15m) > 0 else "N/A"
    )

    # 4. 5m volume 1.5x above average (+2)
    vol_ratio_5m = indicators.get("5m_volume_ratio")
    vol_spike = False
    if vol_ratio_5m is not None and len(vol_ratio_5m) > 0:
        vol_spike = vol_ratio_5m.iloc[-1] >= settings.VOLUME_RATIO_MIN
    pts = settings.SCORE_5M_VOLUME if vol_spike else 0
    score += pts
    details["5m_volume"] = (
        vol_spike, pts,
        f"5m vol ratio: {vol_ratio_5m.iloc[-1]:.1f}x" if vol_ratio_5m is not None and len(vol_ratio_5m) > 0 else "N/A"
    )

    # 5. OI increasing (+2)
    oi_increasing = market.get("oi_increasing", False)
    oi_change = market.get("oi_change", 0.0)
    pts = settings.SCORE_OI_INCREASING if oi_increasing else 0
    score += pts
    details["oi_increasing"] = (oi_increasing, pts, f"OI change: {oi_change:+.1f}%")

    # 6. 1H Bearish CHoCH (+3)
    choch: Optional[CHoCH] = smc.get("1h_choch")
    has_choch = choch is not None and choch.direction == Direction.BEARISH
    pts = settings.SCORE_CHOCH if has_choch else 0
    score += pts
    details["choch"] = (has_choch, pts, "1H Bearish CHoCH")

    # 7. 5m MACD pre-crossover bearish (+1)
    macd_line = indicators.get("5m_macd_line")
    macd_signal = indicators.get("5m_macd_signal")
    macd_precross = False
    if macd_line is not None and macd_signal is not None and len(macd_line) >= 2:
        above_signal = macd_line.iloc[-1] > macd_signal.iloc[-1]
        gap_narrowing = abs(macd_line.iloc[-1] - macd_signal.iloc[-1]) < abs(
            macd_line.iloc[-2] - macd_signal.iloc[-2]
        )
        macd_precross = above_signal and gap_narrowing
    pts = settings.SCORE_MACD_PRECROSS if macd_precross else 0
    score += pts
    details["macd_precross"] = (macd_precross, pts, "MACD pre-crossover (bearish)")

    # 8. L/S ratio > 1.0 (contrarian for shorts) (+2)
    ls_ratio = market.get("ls_ratio")
    ls_contrarian = ls_ratio is not None and ls_ratio > 1.0
    pts = settings.SCORE_LS_RATIO if ls_contrarian else 0
    score += pts
    details["ls_ratio"] = (
        ls_contrarian, pts,
        f"L/S ratio: {ls_ratio:.2f}" if ls_ratio else "N/A"
    )

    # 9. Funding rate positive (+2)
    funding = market.get("funding_rate")
    funding_pos = funding is not None and funding > 0
    pts = settings.SCORE_FUNDING_BONUS if funding_pos else 0
    score += pts
    details["funding_bonus"] = (
        funding_pos, pts,
        f"Funding: {funding * 100:.4f}%" if funding else "N/A"
    )

    # 10. 4H price below EMA 200 by > 2% (+1)
    ema_200 = indicators.get("4h_ema_200")
    strong_bias = False
    if ema_200 is not None:
        pct_below = ((ema_200.iloc[-1] - current_price) / ema_200.iloc[-1]) * 100
        strong_bias = pct_below > 2.0
    pts = settings.SCORE_STRONG_BIAS if strong_bias else 0
    score += pts
    details["strong_bias"] = (strong_bias, pts, "4H strong bearish bias (>2% below EMA 200)")

    # ── Finalize ──
    breakdown.total_score = score
    breakdown.details = details
    breakdown.confidence = (
        "HIGH" if score >= settings.HIGH_CONFIDENCE_SCORE else "STANDARD"
    )

    return breakdown
