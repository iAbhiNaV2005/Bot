"""
signals/short_signal.py — Short signal condition checker (v2).

Mirror image of long_signal.py with all conditions reversed.
"""

from typing import Any, Optional

from analysis.scorer import calculate_score_short
from analysis.smc import Direction, OrderBlock, is_price_in_order_block
from config import settings
from signals.long_signal import Signal, _get_btc_context_str
from signals.tp_sl_calculator import calculate_short_tp_sl
from utils.logger import get_logger

logger = get_logger(__name__)


def check_short(
    symbol: str,
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
    session_info: Optional[dict] = None,
    btc_state: Optional[dict] = None,
) -> Optional[Signal]:
    """
    Evaluate a coin for a SHORT signal (v2).

    Full pipeline (mirror of check_long):
    1. Run scorer with session + BTC state
    2. If mandatory fails → None
    3. If score < minimum → None
    4. Apply Asian session post-score gate
    5. Find the active Bearish Order Block
    6. Calculate TP/SL (with max SL % cap)
    7. If TP/SL is None → None
    8. If R:R < minimum → None
    9. Build and return Signal

    Args:
        symbol: Trading pair.
        indicators: Nested indicator dict.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.
        session_info: Trading session dict (v2).
        btc_state: BTC correlation state (v2).

    Returns:
        Signal object if all conditions pass, None otherwise.
    """
    # ── Step 1: Score ──
    breakdown = calculate_score_short(
        indicators, smc, market, current_price,
        session_info=session_info, btc_state=btc_state,
    )

    if not breakdown.mandatory_passed:
        return None

    if breakdown.total_score < settings.MIN_SCORE_TO_SIGNAL:
        logger.debug(
            f"{symbol} SHORT: Score {breakdown.total_score}/{breakdown.max_score} "
            f"< min {settings.MIN_SCORE_TO_SIGNAL}"
        )
        return None

    # ── Step 2: Asian session post-score gate (v2) ──
    if session_info and session_info.get("session_name") == "Asian":
        if breakdown.total_score < settings.SESSION_ASIAN_MIN_SCORE:
            logger.debug(
                f"{symbol} SHORT: Asian session gate — score {breakdown.total_score} "
                f"< {settings.SESSION_ASIAN_MIN_SCORE}"
            )
            return None

    # ── Step 3: Find active Bearish Order Block ──
    bear_obs: list[OrderBlock] = smc.get("1h_bear_obs", [])
    active_ob: Optional[OrderBlock] = None
    for ob in bear_obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break

    if active_ob is None:
        return None

    # ── Step 4: ATR Filter (15m) ──
    if settings.ATR_FILTER_ACTIVE:
        atr_current = indicators.get("atr", {}).get("15m_current", 0.0)
        if atr_current <= 0:
            logger.debug(f"{symbol} SHORT: ATR filter failed (no ATR data)")
            return None

    # ── Step 5: Calculate TP/SL ──
    atr_value = indicators.get("atr", {}).get("15m_current", 0.0)

    # Strong trend check for TP3
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    strong_trend = False
    if ema_200 > 0:
        pct_below = ((ema_200 - current_price) / ema_200) * 100
        strong_trend = pct_below > 2.0

    tp_sl = calculate_short_tp_sl(current_price, active_ob, atr_value, strong_trend)

    # v2: TP/SL returns None if SL exceeds max cap
    if tp_sl is None:
        logger.debug(f"{symbol} SHORT: TP/SL calculation failed or SL cap exceeded")
        return None

    # ── Step 6: R:R Gate ──
    if not tp_sl.passes_min_rr:
        logger.debug(
            f"{symbol} SHORT: R:R {tp_sl.rr_ratio} < min {settings.MIN_RR_TO_SIGNAL}"
        )
        return None

    # ── Step 7: Build Signal ──
    smc_basis = _build_smc_basis_short(smc, current_price)
    indicator_basis = _build_indicator_basis_short(indicators)
    market_context = _build_market_context(market, session_info, btc_state, indicators)

    fib_zones = smc.get("fib_zones")
    fib_retrace = fib_zones.get("current_retrace_pct", 0.0) if fib_zones else 0.0
    adx_val = indicators.get("adx", {}).get("1h_current", 0.0)

    signal = Signal(
        symbol=symbol,
        direction="SHORT",
        entry=current_price,
        stop_loss=tp_sl.stop_loss,
        tp1=tp_sl.tp1,
        tp2=tp_sl.tp2,
        tp3=tp_sl.tp3,
        score=breakdown.total_score,
        max_score=breakdown.max_score,
        confidence=breakdown.confidence,
        rr_ratio=tp_sl.rr_ratio,
        score_breakdown=breakdown,
        smc_basis=smc_basis,
        indicator_basis=indicator_basis,
        market_context=market_context,
        funding_rate=market.get("funding_rate"),
        oi_change=market.get("oi_change", 0.0),
        ls_ratio=market.get("ls_ratio"),
        session_name=session_info.get("session_name", "") if session_info else "",
        session_score=session_info.get("session_score", 0) if session_info else 0,
        btc_context=_get_btc_context_str(btc_state),
        adx_value=adx_val,
        fib_retrace_pct=fib_retrace,
    )

    logger.info(
        f"🔴 SHORT SIGNAL: {symbol} | Score: {signal.score}/{signal.max_score} "
        f"| Confidence: {signal.confidence} | R:R: {signal.rr_ratio}"
    )

    return signal


def _build_smc_basis_short(smc: dict[str, Any], price: float) -> list[str]:
    """Build list of SMC conditions that passed for short signal (v2)."""
    basis = []

    if smc.get("1h_bos") and smc["1h_bos"].direction == Direction.BEARISH:
        basis.append("1H Bearish BOS ✅")

    bear_obs = smc.get("1h_bear_obs", [])
    for ob in bear_obs:
        if is_price_in_order_block(price, ob):
            touch_str = "1st touch 🔥" if ob.touch_count <= 1 else f"{ob.touch_count}nd touch"
            basis.append(f"Price at Bearish OB ({touch_str}) ✅")
            if ob.confirmed_rejection:
                basis.append("OB rejection candle ✅")
            break

    sweep = smc.get("liquidity_sweep", {})
    if sweep.get("bearish_sweep"):
        candles = sweep.get("candles_since_sweep", "?")
        basis.append(f"Liquidity sweep ({candles} candles ago) ✅")

    div = smc.get("rsi_divergence", {})
    if div.get("bearish_divergence"):
        basis.append("Bearish RSI divergence (15m) ✅")

    if smc.get("15m_bear_fvgs"):
        basis.append("15m Bearish FVG present ✅")

    fib = smc.get("fib_zones")
    if fib:
        retrace = fib.get("current_retrace_pct", 0.0)
        basis.append(f"Fibonacci premium zone ({retrace:.1%}) ✅")

    if smc.get("1h_choch") and smc["1h_choch"].direction == Direction.BEARISH:
        basis.append("1H CHoCH detected ✅")

    return basis


def _build_indicator_basis_short(indicators: dict[str, Any]) -> list[str]:
    """Build list of indicator conditions for short signal (v2)."""
    basis = []

    ema_50 = indicators.get("ema", {}).get("4h_50", 0.0)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    if 0 < ema_50 < ema_200:
        basis.append("4H EMA 50 < EMA 200 ✅")

    rsi_val = indicators.get("rsi", {}).get("5m_current", 0.0)
    rsi_last3 = indicators.get("rsi", {}).get("5m_last3", [0.0, 0.0, 0.0])
    falling = "↘" if rsi_last3[-1] < rsi_last3[-2] else "↗"
    basis.append(f"5m RSI: {rsi_val:.0f} {falling}")

    hist = indicators.get("macd", {}).get("histogram_last3", [0.0, 0.0, 0.0])
    if hist[0] > hist[1] > hist[2] and hist[2] > 0:
        basis.append("5m MACD: histogram converging ✅")

    vol_5m = indicators.get("volume", {}).get("5m_ratio", 0.0)
    if vol_5m >= 1.0:
        basis.append(f"Volume: {vol_5m:.1f}x avg ✅")

    return basis


def _build_market_context(
    market: dict[str, Any],
    session_info: Optional[dict],
    btc_state: Optional[dict],
    indicators: dict[str, Any],
) -> dict[str, str]:
    """Build market context for short signal Telegram message (v2)."""
    ctx: dict[str, str] = {}

    if session_info:
        name = session_info.get("session_name", "")
        score = session_info.get("session_score", 0)
        ctx["session"] = f"{name} ({score:+d})"

    if btc_state:
        ctx["btc_context"] = _get_btc_context_str(btc_state)

    fr = market.get("funding_rate")
    if fr is not None:
        status = "(shorts paying)" if fr < 0 else "(longs paying)" if fr > 0 else ""
        ctx["funding_rate"] = f"{fr * 100:.4f}% {status}"

    oi_change = market.get("oi_change", 0.0)
    direction = "↑" if oi_change > 0 else "↓"
    ctx["open_interest"] = f"{direction} {oi_change:+.1f}% (2h)"

    adx = indicators.get("adx", {}).get("1h_current", 0.0)
    ctx["adx"] = f"{adx:.1f}"

    ls = market.get("ls_ratio")
    if ls is not None:
        status = "(short-heavy)" if ls < 1.0 else "(long-heavy)" if ls > 1.0 else ""
        ctx["ls_ratio"] = f"{ls:.2f} {status}"

    return ctx
