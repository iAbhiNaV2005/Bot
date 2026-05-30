"""
analysis/scorer.py — Signal scoring logic (v3: 37-point system).

Implements the updated scoring system with:
- 9 mandatory conditions (instant discard if false)
- 22 scoring conditions (additive points, max ~37)

New v2 additions: ADX gate, Fibonacci zone, max SL cap,
session scoring, liquidity sweep, RSI divergence, ADX tiers,
OB rejection, OB touch freshness, deep fib zone, BTC correlation.
New v3 additions: Equal Highs/Lows zone proximity (+2/+3),
Volume Quality OB bonus (+1).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from analysis.smc import (
    BOS, CHoCH, Direction, EqualLevel, FVG, OrderBlock,
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
    v2: Added ADX gate and Fibonacci zone check.

    Args:
        indicators: Nested indicator dict from calculate_all_indicators().
        smc: SMC analysis results from run_smc_analysis().
        market: Market data (funding rate, OI, etc.).
        current_price: Current market price.

    Returns:
        Tuple of (all_passed, failed_condition_name).
    """
    # 1. ADX gate — FIRST check (v2: no point calculating anything if ranging)
    adx_current = indicators.get("adx", {}).get("1h_current", 0.0)
    if adx_current < settings.ADX_MIN_THRESHOLD:
        return False, f"ADX {adx_current:.1f} < {settings.ADX_MIN_THRESHOLD} (ranging)"

    # 2. 4H EMA 50 > EMA 200
    ema_50 = indicators.get("ema", {}).get("4h_50", 0.0)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    if ema_50 == 0.0 or ema_200 == 0.0:
        return False, "4H EMA data missing"
    if ema_50 <= ema_200:
        return False, "4H EMA 50 <= EMA 200"

    # 3. 1H Bullish BOS detected in last 20 candles
    bos: Optional[BOS] = smc.get("1h_bos")
    if bos is None or bos.direction != Direction.BULLISH:
        return False, "No 1H Bullish BOS"

    # 4. Price inside 1H Bullish Order Block (body zone)
    bull_obs: list[OrderBlock] = smc.get("1h_bull_obs", [])
    in_ob = any(is_price_in_order_block(current_price, ob) for ob in bull_obs)
    if not in_ob:
        return False, "Price not in Bullish OB"

    # 5. Funding rate < +0.05%
    funding_rate = market.get("funding_rate")
    if funding_rate is not None and funding_rate >= settings.FUNDING_RATE_LONG_MAX:
        return False, "Funding rate too high for long"

    # 6. 5m RSI between 28-45 and rising
    rsi_current = indicators.get("rsi", {}).get("5m_current", 0.0)
    rsi_last3 = indicators.get("rsi", {}).get("5m_last3", [0.0, 0.0, 0.0])
    if not (settings.RSI_LONG_MIN <= rsi_current <= settings.RSI_LONG_MAX):
        return False, f"5m RSI {rsi_current:.1f} outside {settings.RSI_LONG_MIN}-{settings.RSI_LONG_MAX}"
    # RSI rising: last 3 values increasing
    if not (rsi_last3[0] < rsi_last3[1] < rsi_last3[2]):
        return False, "5m RSI not rising"

    # 7. Fibonacci zone check (v2: replaces 50% midpoint)
    fib_zones = smc.get("fib_zones")
    if fib_zones is not None:
        trend = fib_zones.get("trend_direction")
        retrace = fib_zones.get("current_retrace_pct", 0.0)
        if trend != "bullish":
            return False, "Fibonacci trend not bullish"
        if not (settings.FIB_OPTIMAL_LONG_LOW <= retrace <= settings.FIB_DEEP_DISCOUNT_HIGH):
            return False, f"Price retrace {retrace:.1%} outside Fib zone ({settings.FIB_OPTIMAL_LONG_LOW:.0%}-{settings.FIB_DEEP_DISCOUNT_HIGH:.0%})"

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
        indicators: Nested indicator dict.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.

    Returns:
        Tuple of (all_passed, failed_condition_name).
    """
    # 1. ADX gate
    adx_current = indicators.get("adx", {}).get("1h_current", 0.0)
    if adx_current < settings.ADX_MIN_THRESHOLD:
        return False, f"ADX {adx_current:.1f} < {settings.ADX_MIN_THRESHOLD} (ranging)"

    # 2. 4H EMA 50 < EMA 200
    ema_50 = indicators.get("ema", {}).get("4h_50", 0.0)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    if ema_50 == 0.0 or ema_200 == 0.0:
        return False, "4H EMA data missing"
    if ema_50 >= ema_200:
        return False, "4H EMA 50 >= EMA 200"

    # 3. 1H Bearish BOS
    bos: Optional[BOS] = smc.get("1h_bos")
    if bos is None or bos.direction != Direction.BEARISH:
        return False, "No 1H Bearish BOS"

    # 4. Price inside Bearish OB
    bear_obs: list[OrderBlock] = smc.get("1h_bear_obs", [])
    in_ob = any(is_price_in_order_block(current_price, ob) for ob in bear_obs)
    if not in_ob:
        return False, "Price not in Bearish OB"

    # 5. Funding rate > -0.05%
    funding_rate = market.get("funding_rate")
    if funding_rate is not None and funding_rate <= settings.FUNDING_RATE_SHORT_MIN:
        return False, "Funding rate too negative for short"

    # 6. 5m RSI between 55-72 and falling
    rsi_current = indicators.get("rsi", {}).get("5m_current", 0.0)
    rsi_last3 = indicators.get("rsi", {}).get("5m_last3", [0.0, 0.0, 0.0])
    if not (settings.RSI_SHORT_MIN <= rsi_current <= settings.RSI_SHORT_MAX):
        return False, f"5m RSI {rsi_current:.1f} outside {settings.RSI_SHORT_MIN}-{settings.RSI_SHORT_MAX}"
    if not (rsi_last3[0] > rsi_last3[1] > rsi_last3[2]):
        return False, "5m RSI not falling"

    # 7. Fibonacci zone check
    fib_zones = smc.get("fib_zones")
    if fib_zones is not None:
        trend = fib_zones.get("trend_direction")
        retrace = fib_zones.get("current_retrace_pct", 0.0)
        if trend != "bearish":
            return False, "Fibonacci trend not bearish"
        if not (settings.FIB_OPTIMAL_SHORT_LOW <= retrace <= settings.FIB_OPTIMAL_SHORT_HIGH):
            return False, f"Price retrace {retrace:.1%} outside Fib zone"

    return True, ""


def calculate_score_long(
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
    session_info: Optional[dict] = None,
    btc_state: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Calculate the full scoring breakdown for a LONG signal (v2: 32-point system).

    Args:
        indicators: Nested indicator dict.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.
        session_info: Trading session dict (name + score).
        btc_state: BTC correlation state (btc_bearish, btc_bullish flags).

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

    # --- ORIGINAL CONDITIONS (adjusted) ---

    # 1. 15m FVG present below price (+2)
    bull_fvgs: list[FVG] = smc.get("15m_bull_fvgs", [])
    fvg_present = any(fvg.top <= current_price for fvg in bull_fvgs)
    pts = settings.SCORE_FVG_PRESENT if fvg_present else 0
    score += pts
    details["15m_fvg"] = (fvg_present, pts, "Bullish FVG below price")

    # 2. 15m volume above 20-period average (+1)
    vol_ratio_15m = indicators.get("volume", {}).get("15m_ratio", 0.0)
    vol_above = vol_ratio_15m > 1.0
    pts = settings.SCORE_15M_VOLUME if vol_above else 0
    score += pts
    details["15m_volume"] = (vol_above, pts, f"15m vol ratio: {vol_ratio_15m:.1f}x")

    # 3. 5m volume 1.5x above average (+2)
    vol_ratio_5m = indicators.get("volume", {}).get("5m_ratio", 0.0)
    vol_spike = vol_ratio_5m >= settings.VOLUME_RATIO_MIN
    pts = settings.SCORE_5M_VOLUME if vol_spike else 0
    score += pts
    details["5m_volume"] = (vol_spike, pts, f"5m vol ratio: {vol_ratio_5m:.1f}x")

    # 4. OI increasing last 2 hours (+2)
    oi_increasing = market.get("oi_increasing", False)
    oi_change = market.get("oi_change", 0.0)
    pts = settings.SCORE_OI_INCREASING if oi_increasing else 0
    score += pts
    details["oi_increasing"] = (oi_increasing, pts, f"OI change: {oi_change:+.1f}%")

    # 5. 1H CHoCH detected (+3)
    choch: Optional[CHoCH] = smc.get("1h_choch")
    has_choch = choch is not None and choch.direction == Direction.BULLISH
    pts = settings.SCORE_CHOCH if has_choch else 0
    score += pts
    details["choch"] = (has_choch, pts, "1H Bullish CHoCH")

    # 6. 5m MACD histogram pre-crossover (+1) — v2: histogram convergence
    hist_last3 = indicators.get("macd", {}).get("histogram_last3", [0.0, 0.0, 0.0])
    macd_precross = (
        hist_last3[0] < hist_last3[1] < hist_last3[2] and  # converging upward
        hist_last3[2] < 0  # all still negative (below zero line)
    )
    pts = settings.SCORE_MACD_PRECROSS if macd_precross else 0
    score += pts
    details["macd_precross"] = (macd_precross, pts, "MACD histogram converging (bullish)")

    # 7. L/S ratio < 1.0 (contrarian) (+2)
    ls_ratio = market.get("ls_ratio")
    ls_contrarian = ls_ratio is not None and ls_ratio < 1.0
    pts = settings.SCORE_LS_RATIO if ls_contrarian else 0
    score += pts
    details["ls_ratio"] = (
        ls_contrarian, pts,
        f"L/S ratio: {ls_ratio:.2f}" if ls_ratio else "N/A"
    )

    # 8. Funding rate negative (+2)
    funding = market.get("funding_rate")
    funding_neg = funding is not None and funding < 0
    pts = settings.SCORE_FUNDING_BONUS if funding_neg else 0
    score += pts
    details["funding_bonus"] = (
        funding_neg, pts,
        f"Funding: {funding * 100:.4f}%" if funding else "N/A"
    )

    # 9. 4H price above EMA 200 by > 2% (+1)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    strong_bias = False
    if ema_200 > 0:
        pct_above = ((current_price - ema_200) / ema_200) * 100
        strong_bias = pct_above > 2.0
    pts = settings.SCORE_STRONG_BIAS if strong_bias else 0
    score += pts
    details["strong_bias"] = (strong_bias, pts, "4H strong bullish bias (>2% above EMA 200)")

    # --- NEW v2 CONDITIONS ---

    # 10. Session score (varies: -2 to +3)
    if session_info:
        session_score = session_info.get("session_score", 0)
        session_name = session_info.get("session_name", "Unknown")
        score += session_score
        details["session"] = (
            session_score > 0, session_score,
            f"Session: {session_name} ({session_score:+d})"
        )

    # 11. Liquidity sweep detected (+3)
    sweep = smc.get("liquidity_sweep", {})
    has_sweep = sweep.get("bullish_sweep", False)
    candles_since = sweep.get("candles_since_sweep")
    pts = settings.SCORE_SWEEP if has_sweep else 0
    score += pts
    details["sweep"] = (
        has_sweep, pts,
        f"Bullish sweep ({candles_since} candles ago)" if has_sweep else "No sweep"
    )

    # 12. RSI divergence on 15m (+3)
    divergence = smc.get("rsi_divergence", {})
    has_div = divergence.get("bullish_divergence", False)
    pts = settings.SCORE_RSI_DIVERGENCE if has_div else 0
    score += pts
    details["rsi_divergence"] = (has_div, pts, "Bullish RSI divergence (15m)")

    # 13. ADX scoring tiers (+1 if 25-40, +2 if >40)
    adx_val = indicators.get("adx", {}).get("1h_current", 0.0)
    if adx_val >= settings.ADX_SCORE_VERY_STRONG:
        adx_pts = settings.SCORE_ADX_VERY_STRONG
        adx_info = f"ADX: {adx_val:.1f} (very strong)"
    elif adx_val >= settings.ADX_SCORE_STRONG:
        adx_pts = settings.SCORE_ADX_STRONG
        adx_info = f"ADX: {adx_val:.1f} (strong)"
    else:
        adx_pts = 0
        adx_info = f"ADX: {adx_val:.1f} (weak)"
    score += adx_pts
    details["adx_tier"] = (adx_pts > 0, adx_pts, adx_info)

    # 14. OB rejection confirmation candle (+2)
    bull_obs: list[OrderBlock] = smc.get("1h_bull_obs", [])
    has_rejection = any(ob.confirmed_rejection for ob in bull_obs
                        if is_price_in_order_block(current_price, ob))
    pts = settings.SCORE_OB_REJECTION if has_rejection else 0
    score += pts
    details["ob_rejection"] = (has_rejection, pts, "OB rejection candle confirmed")

    # 15. OB touch freshness (+2 if first touch, +0 if second)
    active_ob = None
    for ob in bull_obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break
    if active_ob and active_ob.touch_count <= 1:
        pts = settings.SCORE_OB_FRESH
        ob_touch_info = f"OB 1st touch (fresh)"
    elif active_ob and active_ob.touch_count == 2:
        pts = 0
        ob_touch_info = f"OB 2nd touch"
    else:
        pts = 0
        ob_touch_info = "OB touch N/A"
    score += pts
    details["ob_fresh"] = (pts > 0, pts, ob_touch_info)

    # 16. Deep Fibonacci discount zone (+1)
    fib_zones = smc.get("fib_zones")
    deep_fib = False
    if fib_zones:
        retrace = fib_zones.get("current_retrace_pct", 0.0)
        deep_fib = settings.FIB_OPTIMAL_LONG_HIGH <= retrace <= settings.FIB_DEEP_DISCOUNT_HIGH
    pts = settings.SCORE_DEEP_FIB if deep_fib else 0
    score += pts
    details["deep_fib"] = (
        deep_fib, pts,
        f"Deep discount ({retrace:.1%})" if fib_zones else "N/A"
    )

    # 17. BTC correlation (altcoins only)
    if btc_state:
        is_btc = "BTC" in str(bull_obs[0].timestamp if bull_obs else "")  # won't match
        # Check the symbol via market context — BTC signals skip this
        btc_bullish = btc_state.get("btc_bullish", False)
        btc_bearish = btc_state.get("btc_bearish", False)
        is_btc_symbol = btc_state.get("is_btc_symbol", False)

        if not is_btc_symbol:
            if btc_bullish:
                pts = settings.SCORE_BTC_BULLISH
                score += pts
                details["btc_corr"] = (True, pts, "BTC structure bullish")
            elif btc_bearish:
                pts = settings.SCORE_BTC_BEARISH
                score += pts
                details["btc_corr"] = (False, pts, "BTC structure bearish")
            else:
                details["btc_corr"] = (False, 0, "BTC structure neutral")

    # 18. Equal Lows proximity (v3): price within 1.5x 1H ATR above an unswept EQL zone
    eq_lows: list[EqualLevel] = smc.get("equal_lows", [])
    atr_1h_val = indicators.get("atr", {}).get("1h_current", 0.0)
    eq_long_pts = 0
    eq_long_info = "No equal lows zone"
    if atr_1h_val > 0:
        proximity_band = settings.EQUAL_LEVEL_PROXIMITY_ATR_MULT * atr_1h_val
        for ez in eq_lows:
            if ez.swept:
                continue
            # Price must be ABOVE zone and within proximity_band
            if ez.zone_price <= current_price <= ez.zone_price + proximity_band:
                if ez.member_count >= 3:
                    eq_long_pts = settings.SCORE_EQUAL_TRIPLE
                    eq_long_info = f"Triple equal lows @ {ez.zone_price:.4f} ({ez.member_count} taps)"
                else:
                    eq_long_pts = settings.SCORE_EQUAL_DOUBLE
                    eq_long_info = f"Double equal lows @ {ez.zone_price:.4f}"
                break  # use highest-scoring zone found first
    score += eq_long_pts
    details["equal_lows"] = (eq_long_pts > 0, eq_long_pts, eq_long_info)

    # 19. Volume Quality OB (v3): active OB was formed on high-volume impulse (+1)
    ob_vol_pts = 0
    ob_vol_info = "OB vol quality N/A"
    if active_ob is not None and active_ob.volume_quality_score == 1:
        ob_vol_pts = settings.OB_VOLUME_QUALITY_SCORE
        ob_vol_info = f"High-vol OB ({active_ob.impulse_volume_ratio:.1f}x avg)"
    score += ob_vol_pts
    details["ob_vol_quality"] = (ob_vol_pts > 0, ob_vol_pts, ob_vol_info)

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
    session_info: Optional[dict] = None,
    btc_state: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Calculate the full scoring breakdown for a SHORT signal (v2).

    Mirror of calculate_score_long with reversed conditions.

    Args:
        indicators: Nested indicator dict.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.
        session_info: Trading session dict.
        btc_state: BTC correlation state.

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

    # 2. 15m volume above average (+1)
    vol_ratio_15m = indicators.get("volume", {}).get("15m_ratio", 0.0)
    vol_above = vol_ratio_15m > 1.0
    pts = settings.SCORE_15M_VOLUME if vol_above else 0
    score += pts
    details["15m_volume"] = (vol_above, pts, f"15m vol ratio: {vol_ratio_15m:.1f}x")

    # 3. 5m volume 1.5x above average (+2)
    vol_ratio_5m = indicators.get("volume", {}).get("5m_ratio", 0.0)
    vol_spike = vol_ratio_5m >= settings.VOLUME_RATIO_MIN
    pts = settings.SCORE_5M_VOLUME if vol_spike else 0
    score += pts
    details["5m_volume"] = (vol_spike, pts, f"5m vol ratio: {vol_ratio_5m:.1f}x")

    # 4. OI increasing (+2)
    oi_increasing = market.get("oi_increasing", False)
    oi_change = market.get("oi_change", 0.0)
    pts = settings.SCORE_OI_INCREASING if oi_increasing else 0
    score += pts
    details["oi_increasing"] = (oi_increasing, pts, f"OI change: {oi_change:+.1f}%")

    # 5. 1H Bearish CHoCH (+3)
    choch: Optional[CHoCH] = smc.get("1h_choch")
    has_choch = choch is not None and choch.direction == Direction.BEARISH
    pts = settings.SCORE_CHOCH if has_choch else 0
    score += pts
    details["choch"] = (has_choch, pts, "1H Bearish CHoCH")

    # 6. 5m MACD histogram pre-crossover bearish (+1)
    hist_last3 = indicators.get("macd", {}).get("histogram_last3", [0.0, 0.0, 0.0])
    macd_precross = (
        hist_last3[0] > hist_last3[1] > hist_last3[2] and  # converging downward
        hist_last3[2] > 0  # all still positive
    )
    pts = settings.SCORE_MACD_PRECROSS if macd_precross else 0
    score += pts
    details["macd_precross"] = (macd_precross, pts, "MACD histogram converging (bearish)")

    # 7. L/S ratio > 1.0 (contrarian for shorts) (+2)
    ls_ratio = market.get("ls_ratio")
    ls_contrarian = ls_ratio is not None and ls_ratio > 1.0
    pts = settings.SCORE_LS_RATIO if ls_contrarian else 0
    score += pts
    details["ls_ratio"] = (
        ls_contrarian, pts,
        f"L/S ratio: {ls_ratio:.2f}" if ls_ratio else "N/A"
    )

    # 8. Funding rate positive (+2)
    funding = market.get("funding_rate")
    funding_pos = funding is not None and funding > 0
    pts = settings.SCORE_FUNDING_BONUS if funding_pos else 0
    score += pts
    details["funding_bonus"] = (
        funding_pos, pts,
        f"Funding: {funding * 100:.4f}%" if funding else "N/A"
    )

    # 9. 4H price below EMA 200 by > 2% (+1)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    strong_bias = False
    if ema_200 > 0:
        pct_below = ((ema_200 - current_price) / ema_200) * 100
        strong_bias = pct_below > 2.0
    pts = settings.SCORE_STRONG_BIAS if strong_bias else 0
    score += pts
    details["strong_bias"] = (strong_bias, pts, "4H strong bearish bias (>2% below EMA 200)")

    # --- NEW v2 CONDITIONS ---

    # 10. Session score
    if session_info:
        session_score = session_info.get("session_score", 0)
        session_name = session_info.get("session_name", "Unknown")
        score += session_score
        details["session"] = (
            session_score > 0, session_score,
            f"Session: {session_name} ({session_score:+d})"
        )

    # 11. Liquidity sweep (bearish)
    sweep = smc.get("liquidity_sweep", {})
    has_sweep = sweep.get("bearish_sweep", False)
    candles_since = sweep.get("candles_since_sweep")
    pts = settings.SCORE_SWEEP if has_sweep else 0
    score += pts
    details["sweep"] = (
        has_sweep, pts,
        f"Bearish sweep ({candles_since} candles ago)" if has_sweep else "No sweep"
    )

    # 12. RSI divergence bearish
    divergence = smc.get("rsi_divergence", {})
    has_div = divergence.get("bearish_divergence", False)
    pts = settings.SCORE_RSI_DIVERGENCE if has_div else 0
    score += pts
    details["rsi_divergence"] = (has_div, pts, "Bearish RSI divergence (15m)")

    # 13. ADX tiers
    adx_val = indicators.get("adx", {}).get("1h_current", 0.0)
    if adx_val >= settings.ADX_SCORE_VERY_STRONG:
        adx_pts = settings.SCORE_ADX_VERY_STRONG
        adx_info = f"ADX: {adx_val:.1f} (very strong)"
    elif adx_val >= settings.ADX_SCORE_STRONG:
        adx_pts = settings.SCORE_ADX_STRONG
        adx_info = f"ADX: {adx_val:.1f} (strong)"
    else:
        adx_pts = 0
        adx_info = f"ADX: {adx_val:.1f} (weak)"
    score += adx_pts
    details["adx_tier"] = (adx_pts > 0, adx_pts, adx_info)

    # 14. OB rejection confirmation
    bear_obs: list[OrderBlock] = smc.get("1h_bear_obs", [])
    has_rejection = any(ob.confirmed_rejection for ob in bear_obs
                        if is_price_in_order_block(current_price, ob))
    pts = settings.SCORE_OB_REJECTION if has_rejection else 0
    score += pts
    details["ob_rejection"] = (has_rejection, pts, "OB rejection candle confirmed")

    # 15. OB touch freshness
    active_ob = None
    for ob in bear_obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break
    if active_ob and active_ob.touch_count <= 1:
        pts = settings.SCORE_OB_FRESH
        ob_touch_info = "OB 1st touch (fresh)"
    elif active_ob and active_ob.touch_count == 2:
        pts = 0
        ob_touch_info = "OB 2nd touch"
    else:
        pts = 0
        ob_touch_info = "OB touch N/A"
    score += pts
    details["ob_fresh"] = (pts > 0, pts, ob_touch_info)

    # 16. Fibonacci premium zone bonus for shorts (+1 if deep)
    fib_zones = smc.get("fib_zones")
    deep_fib = False
    retrace = 0.0
    if fib_zones:
        retrace = fib_zones.get("current_retrace_pct", 0.0)
        # For shorts: deeper into premium is better
        deep_fib = retrace <= settings.FIB_OPTIMAL_SHORT_LOW  # above 38.2% = strong premium
    pts = settings.SCORE_DEEP_FIB if deep_fib else 0
    score += pts
    details["deep_fib"] = (
        deep_fib, pts,
        f"Premium zone ({retrace:.1%})" if fib_zones else "N/A"
    )

    # 17. BTC correlation (altcoins only — penalty for short if BTC bullish)
    if btc_state:
        btc_bullish = btc_state.get("btc_bullish", False)
        btc_bearish = btc_state.get("btc_bearish", False)
        is_btc_symbol = btc_state.get("is_btc_symbol", False)

        if not is_btc_symbol:
            if btc_bearish:
                pts = settings.SCORE_BTC_BULLISH  # bonus for shorts when BTC bearish
                score += pts
                details["btc_corr"] = (True, pts, "BTC structure bearish (aligns with short)")
            elif btc_bullish:
                pts = settings.SCORE_BTC_BEARISH  # penalty for shorts when BTC bullish
                score += pts
                details["btc_corr"] = (False, pts, "BTC structure bullish (counter-trend short)")
            else:
                details["btc_corr"] = (False, 0, "BTC structure neutral")

    # 18. Equal Highs proximity (v3): price within 1.5x 1H ATR below an unswept EQH zone
    eq_highs: list[EqualLevel] = smc.get("equal_highs", [])
    atr_1h_val = indicators.get("atr", {}).get("1h_current", 0.0)
    eq_short_pts = 0
    eq_short_info = "No equal highs zone"
    if atr_1h_val > 0:
        proximity_band = settings.EQUAL_LEVEL_PROXIMITY_ATR_MULT * atr_1h_val
        for ez in eq_highs:
            if ez.swept:
                continue
            # Price must be BELOW zone and within proximity_band
            if ez.zone_price - proximity_band <= current_price <= ez.zone_price:
                if ez.member_count >= 3:
                    eq_short_pts = settings.SCORE_EQUAL_TRIPLE
                    eq_short_info = f"Triple equal highs @ {ez.zone_price:.4f} ({ez.member_count} taps)"
                else:
                    eq_short_pts = settings.SCORE_EQUAL_DOUBLE
                    eq_short_info = f"Double equal highs @ {ez.zone_price:.4f}"
                break
    score += eq_short_pts
    details["equal_highs"] = (eq_short_pts > 0, eq_short_pts, eq_short_info)

    # 19. Volume Quality OB (v3): active bearish OB was formed on high-volume impulse (+1)
    ob_vol_pts = 0
    ob_vol_info = "OB vol quality N/A"
    if active_ob is not None and active_ob.volume_quality_score == 1:
        ob_vol_pts = settings.OB_VOLUME_QUALITY_SCORE
        ob_vol_info = f"High-vol OB ({active_ob.impulse_volume_ratio:.1f}x avg)"
    score += ob_vol_pts
    details["ob_vol_quality"] = (ob_vol_pts > 0, ob_vol_pts, ob_vol_info)

    # ── Finalize ──
    breakdown.total_score = score
    breakdown.details = details
    breakdown.confidence = (
        "HIGH" if score >= settings.HIGH_CONFIDENCE_SCORE else "STANDARD"
    )

    return breakdown
