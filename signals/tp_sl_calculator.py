"""
signals/tp_sl_calculator.py — Take Profit, Stop Loss, and Risk-Reward calculation.

STOP LOSS:
  Long  → 1 ATR below the bottom of the Order Block
  Short → 1 ATR above the top of the Order Block

TAKE PROFIT:
  TP1 → 1.5x risk distance (1:1.5 R:R)
  TP2 → 3.0x risk distance (1:3.0 R:R)
  TP3 → 5.0x risk distance (1:5.0 R:R, optional for strong trends)

MINIMUM R:R GATE:
  If TP1 doesn't give at least 1:1.5 R:R, discard the signal.
"""

from dataclasses import dataclass
from typing import Optional

from analysis.smc import OrderBlock
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TPSLResult:
    """
    Calculated TP/SL levels for a signal.

    Attributes:
        entry: Suggested entry price.
        stop_loss: Stop loss price.
        tp1: Take Profit level 1 (1:1.5 R:R).
        tp2: Take Profit level 2 (1:3 R:R).
        tp3: Take Profit level 3 (1:5 R:R, optional).
        risk_distance: Absolute distance from entry to SL.
        rr_ratio: Actual R:R ratio for TP1.
        passes_min_rr: Whether the minimum R:R gate is satisfied.
    """
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    risk_distance: float
    rr_ratio: float
    passes_min_rr: bool


def calculate_long_tp_sl(
    entry_price: float,
    order_block: OrderBlock,
    atr_value: float,
    strong_trend: bool = False,
) -> Optional[TPSLResult]:
    """
    Calculate TP/SL for a LONG signal.

    v2: Returns None if SL distance exceeds MAX_SL_PERCENTAGE (3%).

    Args:
        entry_price: Suggested entry price (current price or OB midpoint).
        order_block: The Bullish Order Block used for the signal.
        atr_value: Current ATR value (14-period, 15m).
        strong_trend: If True, include TP3 at 5x risk.

    Returns:
        TPSLResult with all levels, or None if SL exceeds max cap.
    """
    # Stop Loss: 1 ATR below the bottom of the Order Block
    sl = order_block.bottom - (settings.ATR_SL_MULTIPLIER * atr_value)

    # Risk distance from entry to SL
    risk = entry_price - sl

    if risk <= 0:
        # Edge case: SL is at or above entry — invalid
        logger.warning(
            f"Invalid long SL: entry={entry_price}, sl={sl}, "
            f"ob_bottom={order_block.bottom}, atr={atr_value}"
        )
        return None

    # v2: Max SL percentage cap
    sl_percentage = risk / entry_price
    if sl_percentage > settings.MAX_SL_PERCENTAGE:
        logger.info(
            f"Signal discarded — SL distance {sl_percentage:.1%} "
            f"exceeds {settings.MAX_SL_PERCENTAGE:.0%} cap"
        )
        return None

    # Take Profit levels
    tp1 = entry_price + (settings.TP1_RR * risk)
    tp2 = entry_price + (settings.TP2_RR * risk)
    tp3 = entry_price + (settings.TP3_RR * risk) if strong_trend else None

    # Actual R:R ratio
    rr = (tp1 - entry_price) / risk

    # Minimum R:R gate
    passes = rr >= settings.MIN_RR_TO_SIGNAL

    return TPSLResult(
        entry=entry_price,
        stop_loss=round(sl, 6),
        tp1=round(tp1, 6),
        tp2=round(tp2, 6),
        tp3=round(tp3, 6) if tp3 else None,
        risk_distance=round(risk, 6),
        rr_ratio=round(rr, 2),
        passes_min_rr=passes,
    )


def calculate_short_tp_sl(
    entry_price: float,
    order_block: OrderBlock,
    atr_value: float,
    strong_trend: bool = False,
) -> Optional[TPSLResult]:
    """
    Calculate TP/SL for a SHORT signal.

    v2: Returns None if SL distance exceeds MAX_SL_PERCENTAGE (3%).

    Args:
        entry_price: Suggested entry price.
        order_block: The Bearish Order Block used for the signal.
        atr_value: Current ATR value (14-period, 15m).
        strong_trend: If True, include TP3 at 5x risk.

    Returns:
        TPSLResult with all levels, or None if SL exceeds max cap.
    """
    # Stop Loss: 1 ATR above the top of the Order Block
    sl = order_block.top + (settings.ATR_SL_MULTIPLIER * atr_value)

    # Risk distance
    risk = sl - entry_price

    if risk <= 0:
        logger.warning(
            f"Invalid short SL: entry={entry_price}, sl={sl}, "
            f"ob_top={order_block.top}, atr={atr_value}"
        )
        return None

    # v2: Max SL percentage cap
    sl_percentage = risk / entry_price
    if sl_percentage > settings.MAX_SL_PERCENTAGE:
        logger.info(
            f"Signal discarded — SL distance {sl_percentage:.1%} "
            f"exceeds {settings.MAX_SL_PERCENTAGE:.0%} cap"
        )
        return None

    # Take Profit levels (below entry for shorts)
    tp1 = entry_price - (settings.TP1_RR * risk)
    tp2 = entry_price - (settings.TP2_RR * risk)
    tp3 = entry_price - (settings.TP3_RR * risk) if strong_trend else None

    # Actual R:R
    rr = (entry_price - tp1) / risk

    passes = rr >= settings.MIN_RR_TO_SIGNAL

    return TPSLResult(
        entry=entry_price,
        stop_loss=round(sl, 6),
        tp1=round(tp1, 6),
        tp2=round(tp2, 6),
        tp3=round(tp3, 6) if tp3 else None,
        risk_distance=round(risk, 6),
        rr_ratio=round(rr, 2),
        passes_min_rr=passes,
    )
