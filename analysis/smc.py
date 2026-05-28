"""
analysis/smc.py — All Smart Money Concepts logic.

Implements exact definitions from the strategy spec:
- Swing High / Swing Low detection
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Order Block (OB) identification and invalidation
- Fair Value Gap (FVG) detection and fill-checking
- Premium / Discount zone calculation

Every structure is represented as a dataclass for type safety and clarity.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Enums & Data Classes ─────────────────────────────────────────────────

class Direction(Enum):
    """Trade or structure direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class Zone(Enum):
    """Price zone relative to the 4H range."""
    DISCOUNT = "DISCOUNT"
    PREMIUM = "PREMIUM"


@dataclass
class SwingPoint:
    """
    A confirmed swing high or swing low.

    Attributes:
        index: Integer position in the DataFrame.
        price: Price at the swing point (high for swing high, low for swing low).
        direction: BULLISH = swing high, BEARISH = swing low.
        timestamp: Datetime of the candle.
    """
    index: int
    price: float
    direction: Direction
    timestamp: pd.Timestamp


@dataclass
class BOS:
    """
    Break of Structure.

    Attributes:
        direction: BULLISH (broke above swing high) or BEARISH (broke below swing low).
        candle_index: Index of the candle that broke structure.
        price: Price of the broken swing level.
        timestamp: Datetime of the break.
    """
    direction: Direction
    candle_index: int
    price: float
    timestamp: pd.Timestamp


@dataclass
class CHoCH:
    """
    Change of Character — a structural shift in market direction.

    Attributes:
        direction: New direction after the character change.
        candle_index: Index where CHoCH occurred.
        price: Price of the significant swing that was broken.
        timestamp: Datetime of the CHoCH.
    """
    direction: Direction
    candle_index: int
    price: float
    timestamp: pd.Timestamp


@dataclass
class OrderBlock:
    """
    An Order Block zone.

    Attributes:
        top: Upper boundary of the OB (high of the source candle).
        bottom: Lower boundary of the OB (low of the source candle).
        direction: BULLISH or BEARISH OB.
        candle_index: Index of the source candle.
        valid: Whether the OB is still valid (not broken through).
        timestamp: Datetime of the source candle.
    """
    top: float
    bottom: float
    direction: Direction
    candle_index: int
    valid: bool
    timestamp: pd.Timestamp


@dataclass
class FVG:
    """
    Fair Value Gap.

    Attributes:
        top: Upper boundary of the gap.
        bottom: Lower boundary of the gap.
        direction: BULLISH (gap above) or BEARISH (gap below).
        candle_index: Index of the middle candle that created the gap.
        filled: Whether the gap has been filled by subsequent price action.
        timestamp: Datetime of the middle candle.
    """
    top: float
    bottom: float
    direction: Direction
    candle_index: int
    filled: bool
    timestamp: pd.Timestamp


# ─── Swing Point Detection ────────────────────────────────────────────────

def detect_swing_points(
    df: pd.DataFrame,
    lookback: int = settings.SWING_LOOKBACK,
    max_stored: int = settings.MAX_SWINGS_STORED,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Detect swing highs and swing lows in OHLCV data.

    A Swing High is a candle whose high is higher than the high of both
    the N candles before and N candles after it.

    A Swing Low is a candle whose low is lower than the low of both
    the N candles before and N candles after it.

    Args:
        df: OHLCV DataFrame with 'high' and 'low' columns.
        lookback: Number of candles on each side to confirm (default: 2).
        max_stored: Maximum number of each type to return (default: 10).

    Returns:
        Tuple of (swing_highs, swing_lows), each a list of SwingPoint.
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(lookback, len(df) - lookback):
        # Check swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(SwingPoint(
                index=i,
                price=float(highs[i]),
                direction=Direction.BULLISH,
                timestamp=timestamps[i],
            ))

        # Check swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(SwingPoint(
                index=i,
                price=float(lows[i]),
                direction=Direction.BEARISH,
                timestamp=timestamps[i],
            ))

    # Return only the most recent N
    return swing_highs[-max_stored:], swing_lows[-max_stored:]


# ─── Break of Structure (BOS) ─────────────────────────────────────────────

def detect_bos(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = settings.BOS_LOOKBACK,
) -> Optional[BOS]:
    """
    Detect the most recent Break of Structure.

    Bullish BOS: Price closes ABOVE the most recent Swing High.
    Bearish BOS: Price closes BELOW the most recent Swing Low.

    Only considers candles within the last `lookback` bars.

    Args:
        df: OHLCV DataFrame (1H timeframe).
        swing_highs: Detected swing highs.
        swing_lows: Detected swing lows.
        lookback: How far back to search (default: 20 candles).

    Returns:
        Most recent BOS, or None if no BOS detected.
    """
    if not swing_highs and not swing_lows:
        return None

    closes = df["close"].values
    timestamps = df.index
    search_start = max(0, len(df) - lookback)

    latest_bos: Optional[BOS] = None
    latest_bos_idx = -1

    # Check for bullish BOS (close above most recent swing high)
    if swing_highs:
        # Find the most recent swing high that is BEFORE the search window
        relevant_sh = [sh for sh in swing_highs if sh.index < len(df) - 1]
        if relevant_sh:
            last_sh = relevant_sh[-1]
            for i in range(max(search_start, last_sh.index + 1), len(df)):
                if closes[i] > last_sh.price:
                    if i > latest_bos_idx:
                        latest_bos = BOS(
                            direction=Direction.BULLISH,
                            candle_index=i,
                            price=last_sh.price,
                            timestamp=timestamps[i],
                        )
                        latest_bos_idx = i
                    break  # Only need the first break

    # Check for bearish BOS (close below most recent swing low)
    if swing_lows:
        relevant_sl = [sl for sl in swing_lows if sl.index < len(df) - 1]
        if relevant_sl:
            last_sl = relevant_sl[-1]
            for i in range(max(search_start, last_sl.index + 1), len(df)):
                if closes[i] < last_sl.price:
                    if i > latest_bos_idx:
                        latest_bos = BOS(
                            direction=Direction.BEARISH,
                            candle_index=i,
                            price=last_sl.price,
                            timestamp=timestamps[i],
                        )
                        latest_bos_idx = i
                    break

    return latest_bos


# ─── Change of Character (CHoCH) ──────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> Optional[CHoCH]:
    """
    Detect Change of Character — a stronger structural shift.

    Bullish CHoCH: Price was making Lower Highs + Lower Lows (downtrend),
    then breaks above the LAST swing high before the most recent Lower Low.

    Bearish CHoCH: Price was making Higher Highs + Higher Lows (uptrend),
    then breaks below the LAST swing low before the most recent Higher High.

    Args:
        df: OHLCV DataFrame (1H timeframe).
        swing_highs: Detected swing highs.
        swing_lows: Detected swing lows.

    Returns:
        CHoCH if detected, None otherwise.
    """
    closes = df["close"].values
    timestamps = df.index

    # ── Check for Bullish CHoCH ──
    # Need at least 2 swing highs and 2 swing lows to confirm a downtrend
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Check for downtrend: Lower Highs + Lower Lows
        last_two_sh = swing_highs[-2:]
        last_two_sl = swing_lows[-2:]

        lower_highs = last_two_sh[-1].price < last_two_sh[-2].price
        lower_lows = last_two_sl[-1].price < last_two_sl[-2].price

        if lower_highs and lower_lows:
            # Find the last swing high before the most recent lower low
            most_recent_ll = last_two_sl[-1]
            target_sh = None
            for sh in reversed(swing_highs):
                if sh.index < most_recent_ll.index:
                    target_sh = sh
                    break

            if target_sh:
                # Check if price has broken above this swing high
                for i in range(target_sh.index + 1, len(df)):
                    if closes[i] > target_sh.price:
                        return CHoCH(
                            direction=Direction.BULLISH,
                            candle_index=i,
                            price=target_sh.price,
                            timestamp=timestamps[i],
                        )

    # ── Check for Bearish CHoCH ──
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_two_sh = swing_highs[-2:]
        last_two_sl = swing_lows[-2:]

        higher_highs = last_two_sh[-1].price > last_two_sh[-2].price
        higher_lows = last_two_sl[-1].price > last_two_sl[-2].price

        if higher_highs and higher_lows:
            most_recent_hh = last_two_sh[-1]
            target_sl = None
            for sl in reversed(swing_lows):
                if sl.index < most_recent_hh.index:
                    target_sl = sl
                    break

            if target_sl:
                for i in range(target_sl.index + 1, len(df)):
                    if closes[i] < target_sl.price:
                        return CHoCH(
                            direction=Direction.BEARISH,
                            candle_index=i,
                            price=target_sl.price,
                            timestamp=timestamps[i],
                        )

    return None


# ─── Order Block (OB) Detection ───────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    atr: pd.Series,
    impulse_candles: int = settings.OB_IMPULSE_CANDLES,
    impulse_multiplier: float = settings.OB_IMPULSE_MULTIPLIER,
    max_stored: int = settings.MAX_OBS_STORED,
) -> tuple[list[OrderBlock], list[OrderBlock]]:
    """
    Detect Bullish and Bearish Order Blocks.

    Bullish OB: The last BEARISH candle before a STRONG BULLISH IMPULSE.
    Strong impulse = 3+ consecutive bullish candles moving >= 1.5x ATR.

    Bearish OB: The last BULLISH candle before a STRONG BEARISH IMPULSE.

    Args:
        df: OHLCV DataFrame.
        atr: ATR series (must align with df index).
        impulse_candles: Min consecutive candles in impulse (default: 3).
        impulse_multiplier: Impulse must move >= this * ATR (default: 1.5).
        max_stored: Maximum OBs to return per direction (default: 5).

    Returns:
        Tuple of (bullish_obs, bearish_obs).
    """
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index
    atr_vals = atr.values

    bullish_obs: list[OrderBlock] = []
    bearish_obs: list[OrderBlock] = []

    for i in range(1, len(df) - impulse_candles):
        # ── Bullish OB: bearish candle followed by bullish impulse ──
        if closes[i] < opens[i]:  # Current candle is bearish
            # Check if next N candles form a strong bullish impulse
            impulse_ok = True
            for j in range(1, impulse_candles + 1):
                idx = i + j
                if idx >= len(df):
                    impulse_ok = False
                    break
                if closes[idx] <= opens[idx]:  # Not bullish
                    impulse_ok = False
                    break

            if impulse_ok:
                # Check if total impulse move >= multiplier * ATR
                impulse_end = i + impulse_candles
                if impulse_end < len(df):
                    total_move = closes[impulse_end] - lows[i]
                    atr_at_point = atr_vals[i] if i < len(atr_vals) else 0
                    if atr_at_point > 0 and total_move >= impulse_multiplier * atr_at_point:
                        bullish_obs.append(OrderBlock(
                            top=float(highs[i]),
                            bottom=float(lows[i]),
                            direction=Direction.BULLISH,
                            candle_index=i,
                            valid=True,
                            timestamp=timestamps[i],
                        ))

        # ── Bearish OB: bullish candle followed by bearish impulse ──
        if closes[i] > opens[i]:  # Current candle is bullish
            impulse_ok = True
            for j in range(1, impulse_candles + 1):
                idx = i + j
                if idx >= len(df):
                    impulse_ok = False
                    break
                if closes[idx] >= opens[idx]:  # Not bearish
                    impulse_ok = False
                    break

            if impulse_ok:
                impulse_end = i + impulse_candles
                if impulse_end < len(df):
                    total_move = highs[i] - closes[impulse_end]
                    atr_at_point = atr_vals[i] if i < len(atr_vals) else 0
                    if atr_at_point > 0 and total_move >= impulse_multiplier * atr_at_point:
                        bearish_obs.append(OrderBlock(
                            top=float(highs[i]),
                            bottom=float(lows[i]),
                            direction=Direction.BEARISH,
                            candle_index=i,
                            valid=True,
                            timestamp=timestamps[i],
                        ))

    # Return most recent valid OBs
    return bullish_obs[-max_stored:], bearish_obs[-max_stored:]


def invalidate_order_blocks(
    obs: list[OrderBlock], df: pd.DataFrame
) -> list[OrderBlock]:
    """
    Check and invalidate Order Blocks that price has closed through.

    A Bullish OB is invalidated if price closes below its bottom.
    A Bearish OB is invalidated if price closes above its top.

    Args:
        obs: List of Order Blocks to check.
        df: Current OHLCV DataFrame.

    Returns:
        List with invalid OBs marked (valid=False), filtered to valid only.
    """
    closes = df["close"].values
    valid_obs: list[OrderBlock] = []

    for ob in obs:
        invalidated = False
        # Check all candles after the OB was formed
        for i in range(ob.candle_index + 1, len(df)):
            if ob.direction == Direction.BULLISH and closes[i] < ob.bottom:
                invalidated = True
                break
            elif ob.direction == Direction.BEARISH and closes[i] > ob.top:
                invalidated = True
                break

        if not invalidated:
            valid_obs.append(ob)

    return valid_obs


def is_price_in_order_block(price: float, ob: OrderBlock) -> bool:
    """
    Check if a price is touching or inside an Order Block zone.

    Args:
        price: Current price.
        ob: Order Block to check against.

    Returns:
        True if price is within the OB zone (inclusive).
    """
    return ob.bottom <= price <= ob.top


# ─── Fair Value Gap (FVG) Detection ───────────────────────────────────────

def detect_fvgs(
    df: pd.DataFrame,
    max_age: int = settings.FVG_MAX_AGE_CANDLES,
    max_stored: int = settings.MAX_FVGS_STORED,
) -> tuple[list[FVG], list[FVG]]:
    """
    Detect Bullish and Bearish Fair Value Gaps.

    Bullish FVG (3-candle pattern):
    - Low of candle 3 > High of candle 1
    - Gap = [High(candle 1), Low(candle 3)]

    Bearish FVG:
    - High of candle 3 < Low of candle 1
    - Gap = [High(candle 3), Low(candle 1)]

    Only considers FVGs within the last max_age candles.

    Args:
        df: OHLCV DataFrame (typically 15m).
        max_age: Maximum age in candles (default: 50).
        max_stored: Maximum FVGs per direction to return (default: 10).

    Returns:
        Tuple of (bullish_fvgs, bearish_fvgs).
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    bullish_fvgs: list[FVG] = []
    bearish_fvgs: list[FVG] = []

    start_idx = max(0, len(df) - max_age)

    for i in range(start_idx, len(df) - 2):
        candle1_high = highs[i]
        candle3_low = lows[i + 2]
        candle1_low = lows[i]
        candle3_high = highs[i + 2]

        # Bullish FVG: gap between candle 1 high and candle 3 low
        if candle3_low > candle1_high:
            bullish_fvgs.append(FVG(
                top=float(candle3_low),
                bottom=float(candle1_high),
                direction=Direction.BULLISH,
                candle_index=i + 1,  # middle candle
                filled=False,
                timestamp=timestamps[i + 1],
            ))

        # Bearish FVG: gap between candle 3 high and candle 1 low
        if candle3_high < candle1_low:
            bearish_fvgs.append(FVG(
                top=float(candle1_low),
                bottom=float(candle3_high),
                direction=Direction.BEARISH,
                candle_index=i + 1,
                filled=False,
                timestamp=timestamps[i + 1],
            ))

    return bullish_fvgs[-max_stored:], bearish_fvgs[-max_stored:]


def invalidate_fvgs(fvgs: list[FVG], df: pd.DataFrame) -> list[FVG]:
    """
    Check and remove FVGs that have been filled by price action.

    An FVG is filled when price trades through the entire gap.
    Bullish FVG filled: price went below the bottom.
    Bearish FVG filled: price went above the top.

    Args:
        fvgs: List of FVGs to check.
        df: Current OHLCV DataFrame.

    Returns:
        List of unfilled (valid) FVGs only.
    """
    lows = df["low"].values
    highs = df["high"].values
    valid_fvgs: list[FVG] = []

    for fvg in fvgs:
        filled = False
        for i in range(fvg.candle_index + 1, len(df)):
            if fvg.direction == Direction.BULLISH and lows[i] <= fvg.bottom:
                filled = True
                break
            elif fvg.direction == Direction.BEARISH and highs[i] >= fvg.top:
                filled = True
                break

        if not filled:
            valid_fvgs.append(fvg)

    return valid_fvgs


# ─── Premium / Discount Zone ──────────────────────────────────────────────

def calc_premium_discount(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    current_price: float,
) -> tuple[Zone, float]:
    """
    Determine if price is in premium or discount zone.

    Uses the last significant 4H swing high and swing low to define
    the range. Midpoint = swing_low + (swing_high - swing_low) / 2.

    Discount = below midpoint (good for longs).
    Premium = above midpoint (good for shorts).

    Args:
        swing_highs: 4H swing highs.
        swing_lows: 4H swing lows.
        current_price: Current market price.

    Returns:
        Tuple of (Zone enum, midpoint price).
    """
    if not swing_highs or not swing_lows:
        # Default to discount if no data — conservative
        return Zone.DISCOUNT, current_price

    sh_price = swing_highs[-1].price
    sl_price = swing_lows[-1].price

    midpoint = sl_price + (sh_price - sl_price) / 2

    if current_price < midpoint:
        return Zone.DISCOUNT, midpoint
    else:
        return Zone.PREMIUM, midpoint


# ─── Full SMC Analysis Pipeline ───────────────────────────────────────────

def run_smc_analysis(
    ohlcv: dict[str, pd.DataFrame],
    atr_1h: Optional[pd.Series] = None,
    atr_15m: Optional[pd.Series] = None,
) -> dict[str, any]:
    """
    Run the complete SMC analysis pipeline for a single coin.

    Detects swing points, BOS, CHoCH, Order Blocks, and FVGs across
    relevant timeframes.

    Args:
        ohlcv: Dict of timeframe -> OHLCV DataFrame.
        atr_1h: Pre-computed ATR for 1H (used for OB detection).
        atr_15m: Pre-computed ATR for 15m (used for OB detection on 15m).

    Returns:
        Dict of SMC analysis results keyed by structure type.
    """
    from analysis.indicators import calc_atr

    results: dict[str, any] = {}

    # ── 1H: Swing Points, BOS, CHoCH, Order Blocks ──
    df_1h = ohlcv.get(settings.STRUCTURE_TF)
    if df_1h is not None and len(df_1h) >= 20:
        sh_1h, sl_1h = detect_swing_points(df_1h)
        results["1h_swing_highs"] = sh_1h
        results["1h_swing_lows"] = sl_1h

        bos = detect_bos(df_1h, sh_1h, sl_1h)
        results["1h_bos"] = bos

        choch = detect_choch(df_1h, sh_1h, sl_1h)
        results["1h_choch"] = choch

        # Order Blocks on 1H
        if atr_1h is None:
            atr_1h = calc_atr(df_1h["high"], df_1h["low"], df_1h["close"])
        bull_obs, bear_obs = detect_order_blocks(df_1h, atr_1h)
        bull_obs = invalidate_order_blocks(bull_obs, df_1h)
        bear_obs = invalidate_order_blocks(bear_obs, df_1h)
        results["1h_bull_obs"] = bull_obs
        results["1h_bear_obs"] = bear_obs

    # ── 15m: FVGs ──
    df_15m = ohlcv.get(settings.SETUP_TF)
    if df_15m is not None and len(df_15m) >= 10:
        bull_fvgs, bear_fvgs = detect_fvgs(df_15m)
        bull_fvgs = invalidate_fvgs(bull_fvgs, df_15m)
        bear_fvgs = invalidate_fvgs(bear_fvgs, df_15m)
        results["15m_bull_fvgs"] = bull_fvgs
        results["15m_bear_fvgs"] = bear_fvgs

    # ── 4H: Swing Points for Premium/Discount ──
    df_4h = ohlcv.get(settings.MACRO_TF)
    if df_4h is not None and len(df_4h) >= 10:
        sh_4h, sl_4h = detect_swing_points(df_4h)
        results["4h_swing_highs"] = sh_4h
        results["4h_swing_lows"] = sl_4h

        current_price = float(df_4h["close"].iloc[-1])
        zone, midpoint = calc_premium_discount(sh_4h, sl_4h, current_price)
        results["zone"] = zone
        results["zone_midpoint"] = midpoint

    return results
