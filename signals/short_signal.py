"""
signals/short_signal.py — Short signal condition checker.

Mirror image of long_signal.py with all conditions reversed.
"""

from typing import Any, Optional

from analysis.scorer import calculate_score_short
from analysis.smc import Direction, OrderBlock, Zone, is_price_in_order_block
from config import settings
from signals.long_signal import Signal
from signals.tp_sl_calculator import calculate_short_tp_sl
from utils.logger import get_logger

logger = get_logger(__name__)


def check_short(
    symbol: str,
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> Optional[Signal]:
    """
    Evaluate a coin for a SHORT signal.

    Full pipeline (mirror of check_long):
    1. Run mandatory + score checks
    2. If mandatory fails → None
    3. If score < minimum → None
    4. Find the active Bearish Order Block
    5. Calculate TP/SL
    6. If R:R < minimum → None
    7. Check ATR filter
    8. Build and return Signal

    Args:
        symbol: Trading pair.
        indicators: Computed indicator values.
        smc: SMC analysis results.
        market: Market data.
        current_price: Current market price.

    Returns:
        Signal object if all conditions pass, None otherwise.
    """
    # ── Step 1: Score ──
    breakdown = calculate_score_short(indicators, smc, market, current_price)

    if not breakdown.mandatory_passed:
        return None

    if breakdown.total_score < settings.MIN_SCORE_TO_SIGNAL:
        logger.debug(
            f"{symbol} SHORT: Score {breakdown.total_score}/{breakdown.max_score} "
            f"< min {settings.MIN_SCORE_TO_SIGNAL}"
        )
        return None

    # ── Step 2: Find active Bearish Order Block ──
    bear_obs: list[OrderBlock] = smc.get("1h_bear_obs", [])
    active_ob: Optional[OrderBlock] = None
    for ob in bear_obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break

    if active_ob is None:
        return None

    # ── Step 3: ATR Filter (15m) ──
    if settings.ATR_FILTER_ACTIVE:
        atr_15m = indicators.get("15m_atr")
        atr_avg = indicators.get("15m_atr_avg")
        if atr_15m is not None and atr_avg is not None:
            if len(atr_15m) > 0 and len(atr_avg) > 0:
                if atr_15m.iloc[-1] <= atr_avg.iloc[-1]:
                    logger.debug(f"{symbol} SHORT: ATR filter failed (market too quiet)")
                    return None

    # ── Step 4: Calculate TP/SL ──
    atr_value = 0.0
    atr_15m = indicators.get("15m_atr")
    if atr_15m is not None and len(atr_15m) > 0:
        atr_value = float(atr_15m.iloc[-1])

    # Strong trend check for TP3
    ema_200 = indicators.get("4h_ema_200")
    strong_trend = False
    if ema_200 is not None:
        pct_below = ((ema_200.iloc[-1] - current_price) / ema_200.iloc[-1]) * 100
        strong_trend = pct_below > 2.0

    tp_sl = calculate_short_tp_sl(current_price, active_ob, atr_value, strong_trend)

    # ── Step 5: R:R Gate ──
    if not tp_sl.passes_min_rr:
        logger.debug(
            f"{symbol} SHORT: R:R {tp_sl.rr_ratio} < min {settings.MIN_RR_TO_SIGNAL}"
        )
        return None

    # ── Step 6: Build Signal ──
    smc_basis = _build_smc_basis_short(smc)
    indicator_basis = _build_indicator_basis_short(indicators)
    market_context = _build_market_context(market)

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
    )

    logger.info(
        f"🔴 SHORT SIGNAL: {symbol} | Score: {signal.score}/{signal.max_score} "
        f"| Confidence: {signal.confidence} | R:R: {signal.rr_ratio}"
    )

    return signal


def _build_smc_basis_short(smc: dict[str, Any]) -> list[str]:
    """Build list of SMC conditions that passed for short signal."""
    basis = []
    if smc.get("1h_bos") and smc["1h_bos"].direction == Direction.BEARISH:
        basis.append("1H Bearish BOS ✅")
    if smc.get("1h_bear_obs"):
        basis.append("Price at Bearish Order Block ✅")
    if smc.get("15m_bear_fvgs"):
        basis.append("15m Bearish FVG present ✅")
    if smc.get("zone") == Zone.PREMIUM:
        basis.append("Premium Zone ✅")
    if smc.get("1h_choch") and smc["1h_choch"].direction == Direction.BEARISH:
        basis.append("1H CHoCH detected ✅")
    return basis


def _build_indicator_basis_short(indicators: dict[str, Any]) -> list[str]:
    """Build list of indicator conditions for short signal."""
    basis = []

    ema_50 = indicators.get("4h_ema_50")
    ema_200 = indicators.get("4h_ema_200")
    if ema_50 is not None and ema_200 is not None:
        if ema_50.iloc[-1] < ema_200.iloc[-1]:
            basis.append("4H EMA 50 < EMA 200 ✅")

    rsi = indicators.get("5m_rsi")
    if rsi is not None and len(rsi) > 0:
        rsi_val = rsi.iloc[-1]
        falling = "↘" if len(rsi) >= 2 and rsi.iloc[-1] < rsi.iloc[-2] else "↗"
        basis.append(f"5m RSI: {rsi_val:.0f} {falling}")

    macd = indicators.get("5m_macd_line")
    macd_sig = indicators.get("5m_macd_signal")
    if macd is not None and macd_sig is not None and len(macd) >= 2:
        if macd.iloc[-1] > macd_sig.iloc[-1]:
            gap_narrowing = abs(macd.iloc[-1] - macd_sig.iloc[-1]) < abs(
                macd.iloc[-2] - macd_sig.iloc[-2]
            )
            if gap_narrowing:
                basis.append("5m MACD: pre-crossover ✅")

    vol_5m = indicators.get("5m_volume_ratio")
    if vol_5m is not None and len(vol_5m) > 0:
        ratio = vol_5m.iloc[-1]
        if ratio >= 1.0:
            basis.append(f"Volume: {ratio:.1f}x avg ✅")

    return basis


def _build_market_context(market: dict[str, Any]) -> dict[str, str]:
    """Build market context for short signal Telegram message."""
    ctx: dict[str, str] = {}

    fr = market.get("funding_rate")
    if fr is not None:
        status = "(shorts paying)" if fr < 0 else "(longs paying)" if fr > 0 else ""
        ctx["funding_rate"] = f"{fr * 100:.4f}% {status}"

    oi_change = market.get("oi_change", 0.0)
    direction = "↑" if oi_change > 0 else "↓"
    ctx["open_interest"] = f"{direction} {oi_change:+.1f}% (2h)"

    ls = market.get("ls_ratio")
    if ls is not None:
        status = "(short-heavy)" if ls < 1.0 else "(long-heavy)" if ls > 1.0 else ""
        ctx["ls_ratio"] = f"{ls:.2f} {status}"

    return ctx
