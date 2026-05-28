"""
signals/long_signal.py — Long signal condition checker.

Orchestrates the full long signal evaluation:
1. Check all mandatory conditions via scorer
2. Calculate additive score
3. Calculate TP/SL
4. Enforce minimum R:R gate
5. Package into a Signal object if everything passes

Returns None if any gate fails.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from analysis.scorer import ScoreBreakdown, calculate_score_long
from analysis.smc import Direction, OrderBlock, is_price_in_order_block
from config import settings
from signals.tp_sl_calculator import TPSLResult, calculate_long_tp_sl
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """
    A complete trading signal ready for delivery.

    Attributes:
        symbol: Trading pair (e.g., 'ETH/USDT:USDT').
        direction: 'LONG' or 'SHORT'.
        entry: Entry price.
        stop_loss: Stop loss price.
        tp1: Take profit level 1.
        tp2: Take profit level 2.
        tp3: Take profit level 3 (optional).
        score: Signal score achieved.
        max_score: Maximum possible score.
        confidence: 'HIGH' or 'STANDARD'.
        rr_ratio: Risk-reward ratio.
        score_breakdown: Detailed scoring breakdown.
        smc_basis: List of SMC conditions that passed.
        indicator_basis: List of indicator conditions that passed.
        market_context: Dict of market data values.
        funding_rate: Current funding rate.
        oi_change: OI change percentage.
        ls_ratio: Long/short ratio.
    """
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    score: int
    max_score: int
    confidence: str
    rr_ratio: float
    score_breakdown: ScoreBreakdown
    smc_basis: list[str] = field(default_factory=list)
    indicator_basis: list[str] = field(default_factory=list)
    market_context: dict[str, str] = field(default_factory=dict)
    funding_rate: Optional[float] = None
    oi_change: float = 0.0
    ls_ratio: Optional[float] = None


def check_long(
    symbol: str,
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
) -> Optional[Signal]:
    """
    Evaluate a coin for a LONG signal.

    Full pipeline:
    1. Run mandatory + score checks
    2. If mandatory fails → None
    3. If score < minimum → None
    4. Find the active Order Block
    5. Calculate TP/SL
    6. If R:R < minimum → None
    7. Check ATR filter (market must be active)
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
    breakdown = calculate_score_long(indicators, smc, market, current_price)

    if not breakdown.mandatory_passed:
        return None

    if breakdown.total_score < settings.MIN_SCORE_TO_SIGNAL:
        logger.debug(
            f"{symbol} LONG: Score {breakdown.total_score}/{breakdown.max_score} "
            f"< min {settings.MIN_SCORE_TO_SIGNAL}"
        )
        return None

    # ── Step 2: Find active Order Block ──
    bull_obs: list[OrderBlock] = smc.get("1h_bull_obs", [])
    active_ob: Optional[OrderBlock] = None
    for ob in bull_obs:
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
                    logger.debug(f"{symbol} LONG: ATR filter failed (market too quiet)")
                    return None

    # ── Step 4: Calculate TP/SL ──
    atr_value = 0.0
    atr_15m = indicators.get("15m_atr")
    if atr_15m is not None and len(atr_15m) > 0:
        atr_value = float(atr_15m.iloc[-1])

    # Check for strong trend (TP3 eligibility)
    ema_200 = indicators.get("4h_ema_200")
    strong_trend = False
    if ema_200 is not None:
        pct_above = ((current_price - ema_200.iloc[-1]) / ema_200.iloc[-1]) * 100
        strong_trend = pct_above > 2.0

    tp_sl = calculate_long_tp_sl(current_price, active_ob, atr_value, strong_trend)

    # ── Step 5: R:R Gate ──
    if not tp_sl.passes_min_rr:
        logger.debug(
            f"{symbol} LONG: R:R {tp_sl.rr_ratio} < min {settings.MIN_RR_TO_SIGNAL}"
        )
        return None

    # ── Step 6: Build Signal ──
    smc_basis = _build_smc_basis_long(smc, current_price)
    indicator_basis = _build_indicator_basis_long(indicators, current_price)
    market_context = _build_market_context(market)

    signal = Signal(
        symbol=symbol,
        direction="LONG",
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
        f"🟢 LONG SIGNAL: {symbol} | Score: {signal.score}/{signal.max_score} "
        f"| Confidence: {signal.confidence} | R:R: {signal.rr_ratio}"
    )

    return signal


def _build_smc_basis_long(smc: dict[str, Any], price: float) -> list[str]:
    """Build list of SMC conditions that passed for the Telegram message."""
    basis = []
    if smc.get("1h_bos") and smc["1h_bos"].direction == Direction.BULLISH:
        basis.append("1H Bullish BOS ✅")
    if smc.get("1h_bull_obs"):
        basis.append("Price at Bullish Order Block ✅")
    if smc.get("15m_bull_fvgs"):
        basis.append("15m Bullish FVG present ✅")
    from analysis.smc import Zone
    if smc.get("zone") == Zone.DISCOUNT:
        basis.append("Discount Zone ✅")
    if smc.get("1h_choch") and smc["1h_choch"].direction == Direction.BULLISH:
        basis.append("1H CHoCH detected ✅")
    return basis


def _build_indicator_basis_long(indicators: dict[str, Any], price: float) -> list[str]:
    """Build list of indicator conditions that passed."""
    basis = []

    ema_50 = indicators.get("4h_ema_50")
    ema_200 = indicators.get("4h_ema_200")
    if ema_50 is not None and ema_200 is not None:
        if ema_50.iloc[-1] > ema_200.iloc[-1]:
            basis.append("4H EMA 50 > EMA 200 ✅")

    rsi = indicators.get("5m_rsi")
    if rsi is not None and len(rsi) > 0:
        rsi_val = rsi.iloc[-1]
        rising = "↗" if len(rsi) >= 2 and rsi.iloc[-1] > rsi.iloc[-2] else "↘"
        basis.append(f"5m RSI: {rsi_val:.0f} {rising}")

    macd = indicators.get("5m_macd_line")
    macd_sig = indicators.get("5m_macd_signal")
    if macd is not None and macd_sig is not None and len(macd) >= 2:
        if macd.iloc[-1] < macd_sig.iloc[-1]:
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
    """Build market context dict for the Telegram message."""
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
