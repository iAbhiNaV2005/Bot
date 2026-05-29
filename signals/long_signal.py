"""
signals/long_signal.py — Long signal condition checker (v2).

Orchestrates the full long signal evaluation:
1. Run scorer with session + BTC state
2. Check mandatory conditions
3. Calculate additive score
4. Apply Asian session post-score gate
5. Calculate TP/SL with max SL % cap
6. Enforce minimum R:R gate
7. Package into a Signal object if everything passes

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
    A complete trading signal ready for delivery (v2).

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
        session_name: Trading session name (v2).
        session_score: Session score modifier (v2).
        btc_context: BTC structure context string (v2).
        adx_value: Current ADX value (v2).
        fib_retrace_pct: Fibonacci retracement percentage (v2).
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
    session_name: str = ""
    session_score: int = 0
    btc_context: str = ""
    adx_value: float = 0.0
    fib_retrace_pct: float = 0.0


def check_long(
    symbol: str,
    indicators: dict[str, Any],
    smc: dict[str, Any],
    market: dict[str, Any],
    current_price: float,
    session_info: Optional[dict] = None,
    btc_state: Optional[dict] = None,
) -> Optional[Signal]:
    """
    Evaluate a coin for a LONG signal (v2).

    Full pipeline:
    1. Run scorer with session + BTC state
    2. If mandatory fails → None
    3. If score < minimum → None
    4. Apply Asian session post-score gate
    5. Find the active Order Block
    6. Calculate TP/SL (with max SL % cap)
    7. If TP/SL is None (SL too wide) → None
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
    breakdown = calculate_score_long(
        indicators, smc, market, current_price,
        session_info=session_info, btc_state=btc_state,
    )

    if not breakdown.mandatory_passed:
        return None

    if breakdown.total_score < settings.MIN_SCORE_TO_SIGNAL:
        logger.debug(
            f"{symbol} LONG: Score {breakdown.total_score}/{breakdown.max_score} "
            f"< min {settings.MIN_SCORE_TO_SIGNAL}"
        )
        return None

    # ── Step 2: Asian session post-score gate (v2) ──
    if session_info and session_info.get("session_name") == "Asian":
        if breakdown.total_score < settings.SESSION_ASIAN_MIN_SCORE:
            logger.debug(
                f"{symbol} LONG: Asian session gate — score {breakdown.total_score} "
                f"< {settings.SESSION_ASIAN_MIN_SCORE}"
            )
            return None

    # ── Step 3: Find active Order Block ──
    bull_obs: list[OrderBlock] = smc.get("1h_bull_obs", [])
    active_ob: Optional[OrderBlock] = None
    for ob in bull_obs:
        if is_price_in_order_block(current_price, ob):
            active_ob = ob
            break

    if active_ob is None:
        return None

    # ── Step 4: ATR Filter (15m) ──
    if settings.ATR_FILTER_ACTIVE:
        atr_current = indicators.get("atr", {}).get("15m_current", 0.0)
        # ATR filter: skip if ATR is too low (market too quiet)
        # Use a simple check: ATR must be > 0
        if atr_current <= 0:
            logger.debug(f"{symbol} LONG: ATR filter failed (no ATR data)")
            return None

    # ── Step 5: Calculate TP/SL ──
    atr_value = indicators.get("atr", {}).get("15m_current", 0.0)

    # Check for strong trend (TP3 eligibility)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    strong_trend = False
    if ema_200 > 0:
        pct_above = ((current_price - ema_200) / ema_200) * 100
        strong_trend = pct_above > 2.0

    tp_sl = calculate_long_tp_sl(current_price, active_ob, atr_value, strong_trend)

    # v2: TP/SL returns None if SL exceeds max cap
    if tp_sl is None:
        logger.debug(f"{symbol} LONG: TP/SL calculation failed or SL cap exceeded")
        return None

    # ── Step 6: R:R Gate ──
    if not tp_sl.passes_min_rr:
        logger.debug(
            f"{symbol} LONG: R:R {tp_sl.rr_ratio} < min {settings.MIN_RR_TO_SIGNAL}"
        )
        return None

    # ── Step 7: Build Signal ──
    smc_basis = _build_smc_basis_long(smc, current_price)
    indicator_basis = _build_indicator_basis_long(indicators, current_price)
    market_context = _build_market_context(market, session_info, btc_state, indicators)

    # v2: Extract new signal fields
    fib_zones = smc.get("fib_zones")
    fib_retrace = fib_zones.get("current_retrace_pct", 0.0) if fib_zones else 0.0
    adx_val = indicators.get("adx", {}).get("1h_current", 0.0)

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
        session_name=session_info.get("session_name", "") if session_info else "",
        session_score=session_info.get("session_score", 0) if session_info else 0,
        btc_context=_get_btc_context_str(btc_state),
        adx_value=adx_val,
        fib_retrace_pct=fib_retrace,
    )

    logger.info(
        f"🟢 LONG SIGNAL: {symbol} | Score: {signal.score}/{signal.max_score} "
        f"| Confidence: {signal.confidence} | R:R: {signal.rr_ratio}"
    )

    return signal


def _build_smc_basis_long(smc: dict[str, Any], price: float) -> list[str]:
    """Build list of SMC conditions that passed for the Telegram message (v2)."""
    basis = []

    if smc.get("1h_bos") and smc["1h_bos"].direction == Direction.BULLISH:
        basis.append("1H Bullish BOS ✅")

    # OB with touch info
    bull_obs = smc.get("1h_bull_obs", [])
    for ob in bull_obs:
        if is_price_in_order_block(price, ob):
            touch_str = "1st touch 🔥" if ob.touch_count <= 1 else f"{ob.touch_count}nd touch"
            basis.append(f"Price at Bullish OB ({touch_str}) ✅")
            if ob.confirmed_rejection:
                basis.append("OB rejection candle ✅")
            break

    # Liquidity sweep
    sweep = smc.get("liquidity_sweep", {})
    if sweep.get("bullish_sweep"):
        candles = sweep.get("candles_since_sweep", "?")
        basis.append(f"Liquidity sweep ({candles} candles ago) ✅")

    # RSI divergence
    div = smc.get("rsi_divergence", {})
    if div.get("bullish_divergence"):
        basis.append("Bullish RSI divergence (15m) ✅")

    if smc.get("15m_bull_fvgs"):
        basis.append("15m Bullish FVG present ✅")

    # Fibonacci zone
    fib = smc.get("fib_zones")
    if fib:
        retrace = fib.get("current_retrace_pct", 0.0)
        basis.append(f"Fibonacci discount zone ({retrace:.1%}) ✅")

    if smc.get("1h_choch") and smc["1h_choch"].direction == Direction.BULLISH:
        basis.append("1H CHoCH detected ✅")

    return basis


def _build_indicator_basis_long(indicators: dict[str, Any], price: float) -> list[str]:
    """Build list of indicator conditions that passed (v2)."""
    basis = []

    ema_50 = indicators.get("ema", {}).get("4h_50", 0.0)
    ema_200 = indicators.get("ema", {}).get("4h_200", 0.0)
    if ema_50 > ema_200 > 0:
        basis.append("4H EMA 50 > EMA 200 ✅")

    rsi_val = indicators.get("rsi", {}).get("5m_current", 0.0)
    rsi_last3 = indicators.get("rsi", {}).get("5m_last3", [0.0, 0.0, 0.0])
    rising = "↗" if rsi_last3[-1] > rsi_last3[-2] else "↘"
    basis.append(f"5m RSI: {rsi_val:.0f} {rising}")

    # MACD histogram
    hist = indicators.get("macd", {}).get("histogram_last3", [0.0, 0.0, 0.0])
    if hist[0] < hist[1] < hist[2] and hist[2] < 0:
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
    """Build market context dict for the Telegram message (v2)."""
    ctx: dict[str, str] = {}

    # Session
    if session_info:
        name = session_info.get("session_name", "")
        score = session_info.get("session_score", 0)
        ctx["session"] = f"{name} ({score:+d})"

    # BTC context
    if btc_state:
        ctx["btc_context"] = _get_btc_context_str(btc_state)

    # Funding rate
    fr = market.get("funding_rate")
    if fr is not None:
        status = "(shorts paying)" if fr < 0 else "(longs paying)" if fr > 0 else ""
        ctx["funding_rate"] = f"{fr * 100:.4f}% {status}"

    # OI
    oi_change = market.get("oi_change", 0.0)
    direction = "↑" if oi_change > 0 else "↓"
    ctx["open_interest"] = f"{direction} {oi_change:+.1f}% (2h)"

    # ADX
    adx = indicators.get("adx", {}).get("1h_current", 0.0)
    ctx["adx"] = f"{adx:.1f}"

    # L/S ratio
    ls = market.get("ls_ratio")
    if ls is not None:
        status = "(short-heavy)" if ls < 1.0 else "(long-heavy)" if ls > 1.0 else ""
        ctx["ls_ratio"] = f"{ls:.2f} {status}"

    return ctx


def _get_btc_context_str(btc_state: Optional[dict]) -> str:
    """Get BTC context as a display string."""
    if not btc_state:
        return "N/A"
    if btc_state.get("btc_bullish"):
        return "Bullish ✅"
    elif btc_state.get("btc_bearish"):
        return "Bearish ⚠️"
    return "Neutral"
