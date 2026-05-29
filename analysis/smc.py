"""
analysis/smc.py — All Smart Money Concepts logic.

Implements exact definitions from the strategy spec:
- Swing High / Swing Low detection (timeframe-specific lookback)
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Order Block (OB) identification (body-based) and invalidation
- Fair Value Gap (FVG) detection and fill-checking
- Liquidity Sweep detection
- RSI Divergence detection
- OB Rejection Confirmation
- OB Touch Count tracking

v2: Body-based OBs, timeframe-specific swings, liquidity sweeps,
    RSI divergence, OB rejection/touch tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

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
    An Order Block zone (v2: body-based with touch tracking).

    Attributes:
        top: Upper boundary of the OB (body = max(open, close)).
        bottom: Lower boundary of the OB (body = min(open, close)).
        wick_high: Full candle high (stored separately).
        wick_low: Full candle low (stored separately).
        direction: BULLISH or BEARISH OB.
        candle_index: Index of the source candle.
        valid: Whether the OB is still valid (not broken through).
        timestamp: Datetime of the source candle.
        touch_count: Number of times price has visited this OB zone.
        currently_inside: Whether price is currently inside the OB zone.
        confirmed_rejection: Whether a rejection candle has been detected.
    """
    top: float
    bottom: float
    wick_high: float
    wick_low: float
    direction: Direction
    candle_index: int
    valid: bool
    timestamp: pd.Timestamp
    touch_count: int = 0
    currently_inside: bool = False
    confirmed_rejection: bool = False


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
    lookback: int,
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
        lookback: Number of candles on each side to confirm.
                  Must be explicitly passed — no default.
                  Use settings.SWING_LOOKBACK_1H/15M/5M.
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
    Detect Bullish and Bearish Order Blocks (v2: body-based).

    Bullish OB: The last BEARISH candle before a STRONG BULLISH IMPULSE.
    Bearish OB: The last BULLISH candle before a STRONG BEARISH IMPULSE.

    OB zone is defined by the BODY (max/min of open/close), not the wick.
    Wicks are stored separately for reference.

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
                        # v2: Body-based OB zone
                        body_top = max(float(opens[i]), float(closes[i]))
                        body_bottom = min(float(opens[i]), float(closes[i]))
                        bullish_obs.append(OrderBlock(
                            top=body_top,
                            bottom=body_bottom,
                            wick_high=float(highs[i]),
                            wick_low=float(lows[i]),
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
                        body_top = max(float(opens[i]), float(closes[i]))
                        body_bottom = min(float(opens[i]), float(closes[i]))
                        bearish_obs.append(OrderBlock(
                            top=body_top,
                            bottom=body_bottom,
                            wick_high=float(highs[i]),
                            wick_low=float(lows[i]),
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
    Check and invalidate Order Blocks that price has CLOSED through.

    v2: Uses close-based invalidation (not wick-based).
    A Bullish OB is invalidated if price CLOSES below its bottom.
    A Bearish OB is invalidated if price CLOSES above its top.
    Wicking into the zone without closing through keeps OB valid.

    Args:
        obs: List of Order Blocks to check.
        df: Current OHLCV DataFrame.

    Returns:
        List of valid OBs only (invalid ones removed).
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


def update_ob_touch_counts(
    obs: list[OrderBlock], df: pd.DataFrame
) -> list[OrderBlock]:
    """
    Update touch counts for Order Blocks and invalidate on 3rd touch.

    A "touch" = price enters OB zone from outside (transition from
    currently_inside=False to True). Increment only on transition.

    Args:
        obs: List of Order Blocks to update.
        df: Current OHLCV DataFrame (use latest candle).

    Returns:
        List of valid OBs (3rd-touch OBs removed).
    """
    if len(df) == 0:
        return obs

    valid_obs: list[OrderBlock] = []

    for ob in obs:
        current_low = float(df["low"].iloc[-1])
        current_high = float(df["high"].iloc[-1])

        # Check if price is currently inside OB zone
        if ob.direction == Direction.BULLISH:
            price_in_zone = current_low <= ob.top and current_low >= ob.bottom
        else:
            price_in_zone = current_high >= ob.bottom and current_high <= ob.top

        if price_in_zone and not ob.currently_inside:
            # Transition: entered zone from outside
            ob.touch_count += 1
            ob.currently_inside = True
        elif not price_in_zone:
            ob.currently_inside = False

        # Invalidate on 3rd touch
        if ob.touch_count >= settings.OB_MAX_TOUCH_COUNT:
            continue  # Skip — OB exhausted

        valid_obs.append(ob)

    return valid_obs


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


# ─── Liquidity Sweep Detection (v2) ──────────────────────────────────────

def detect_liquidity_sweep(
    df_15m: pd.DataFrame,
    swing_lows: list[SwingPoint],
    swing_highs: list[SwingPoint],
    atr_15m: pd.Series,
) -> dict[str, Any]:
    """
    Detect liquidity sweeps on 15m timeframe.

    A Bullish Sweep: wick goes BELOW a swing low but CLOSES ABOVE it.
    A Bearish Sweep: wick goes ABOVE a swing high but CLOSES BELOW it.

    The sweep wick must be at least 0.1 × ATR beyond the swing level.

    Args:
        df_15m: 15m OHLCV DataFrame.
        swing_lows: Detected 15m swing lows.
        swing_highs: Detected 15m swing highs.
        atr_15m: ATR series for 15m (for minimum wick size check).

    Returns:
        Dict with 'bullish_sweep', 'bearish_sweep' (bool),
        'candles_since_sweep' (int or None).
    """
    result = {
        "bullish_sweep": False,
        "bearish_sweep": False,
        "candles_since_sweep": None,
    }

    if len(df_15m) < 10:
        return result

    lows = df_15m["low"].values
    highs = df_15m["high"].values
    closes = df_15m["close"].values
    current_idx = len(df_15m) - 1

    # Get current ATR for minimum wick check
    min_wick = 0.0
    if atr_15m is not None and len(atr_15m) > 0:
        current_atr = float(atr_15m.iloc[-1])
        min_wick = settings.SWEEP_MIN_WICK_ATR_MULT * current_atr

    # ── Bullish Sweep: wick below swing low, close above ──
    if swing_lows:
        for candle_offset in range(min(10, len(df_15m))):
            i = current_idx - candle_offset
            if i < 0:
                break
            candle_low = lows[i]
            candle_close = closes[i]

            for sl in swing_lows:
                if sl.index >= i:
                    continue  # Skip swing lows at or after this candle
                swing_price = sl.price
                wick_depth = swing_price - candle_low

                if (candle_low < swing_price and
                        candle_close > swing_price and
                        wick_depth >= min_wick):
                    candles_since = current_idx - i
                    if candles_since <= settings.SWEEP_SIGNAL_WINDOW_CANDLES:
                        result["bullish_sweep"] = True
                        result["candles_since_sweep"] = candles_since
                        break
            if result["bullish_sweep"]:
                break

    # ── Bearish Sweep: wick above swing high, close below ──
    if swing_highs:
        for candle_offset in range(min(10, len(df_15m))):
            i = current_idx - candle_offset
            if i < 0:
                break
            candle_high = highs[i]
            candle_close = closes[i]

            for sh in swing_highs:
                if sh.index >= i:
                    continue
                swing_price = sh.price
                wick_depth = candle_high - swing_price

                if (candle_high > swing_price and
                        candle_close < swing_price and
                        wick_depth >= min_wick):
                    candles_since = current_idx - i
                    if candles_since <= settings.SWEEP_SIGNAL_WINDOW_CANDLES:
                        result["bearish_sweep"] = True
                        result["candles_since_sweep"] = candles_since
                        break
            if result["bearish_sweep"]:
                break

    return result


# ─── RSI Divergence Detection (v2) ───────────────────────────────────────

def detect_rsi_divergence(
    df_15m: pd.DataFrame,
    rsi_series: pd.Series,
    swing_lows: list[SwingPoint],
    swing_highs: list[SwingPoint],
) -> dict[str, bool]:
    """
    Detect RSI divergence on 15m timeframe.

    Bullish divergence: price makes lower low, RSI makes higher low.
    Bearish divergence: price makes higher high, RSI makes lower high.

    Args:
        df_15m: 15m OHLCV DataFrame.
        rsi_series: Full RSI series for 15m (same length as df_15m).
        swing_lows: Detected 15m swing lows.
        swing_highs: Detected 15m swing highs.

    Returns:
        Dict with 'bullish_divergence' and 'bearish_divergence' (bool).
    """
    result = {"bullish_divergence": False, "bearish_divergence": False}

    if rsi_series is None or len(rsi_series) < settings.DIVERGENCE_SCAN_CANDLES:
        return result

    current_idx = len(df_15m) - 1
    scan_start = max(0, current_idx - settings.DIVERGENCE_SCAN_CANDLES)

    # ── Bullish Divergence ──
    # Find two most recent swing lows within scan window
    recent_lows = [sl for sl in swing_lows
                   if scan_start <= sl.index <= current_idx]
    if len(recent_lows) >= 2:
        swing_low_1 = recent_lows[-2]  # Older
        swing_low_2 = recent_lows[-1]  # More recent

        price_low_1 = swing_low_1.price
        price_low_2 = swing_low_2.price

        # Get RSI at each swing low
        idx1 = swing_low_1.index
        idx2 = swing_low_2.index

        if idx1 < len(rsi_series) and idx2 < len(rsi_series):
            rsi_at_low_1 = rsi_series.iloc[idx1]
            rsi_at_low_2 = rsi_series.iloc[idx2]

            if (not pd.isna(rsi_at_low_1) and not pd.isna(rsi_at_low_2)):
                # Divergence: price lower low, RSI higher low
                if (price_low_2 < price_low_1 and rsi_at_low_2 > rsi_at_low_1):
                    # Proximity check: must be within N candles of current
                    if (current_idx - idx2) <= settings.DIVERGENCE_SIGNAL_PROXIMITY:
                        result["bullish_divergence"] = True

    # ── Bearish Divergence ──
    recent_highs = [sh for sh in swing_highs
                    if scan_start <= sh.index <= current_idx]
    if len(recent_highs) >= 2:
        swing_high_1 = recent_highs[-2]
        swing_high_2 = recent_highs[-1]

        price_high_1 = swing_high_1.price
        price_high_2 = swing_high_2.price

        idx1 = swing_high_1.index
        idx2 = swing_high_2.index

        if idx1 < len(rsi_series) and idx2 < len(rsi_series):
            rsi_at_high_1 = rsi_series.iloc[idx1]
            rsi_at_high_2 = rsi_series.iloc[idx2]

            if (not pd.isna(rsi_at_high_1) and not pd.isna(rsi_at_high_2)):
                # Divergence: price higher high, RSI lower high
                if (price_high_2 > price_high_1 and rsi_at_high_2 < rsi_at_high_1):
                    if (current_idx - idx2) <= settings.DIVERGENCE_SIGNAL_PROXIMITY:
                        result["bearish_divergence"] = True

    return result


# ─── OB Rejection Confirmation (v2) ──────────────────────────────────────

def check_ob_rejection_confirmation(
    df_15m: pd.DataFrame,
    ob: OrderBlock,
) -> bool:
    """
    Check for a rejection candle at an Order Block.

    Bullish OB: wick touches/enters OB zone from above, close stays above OB top.
    Bearish OB: wick touches/enters OB zone from below, close stays below OB bottom.

    Only checks the last 3 candles on 15m.

    Args:
        df_15m: 15m OHLCV DataFrame.
        ob: Order Block to check rejection against.

    Returns:
        True if a rejection candle is found, False otherwise.
    """
    if len(df_15m) < 3:
        return False

    for i in range(-3, 0):
        try:
            candle_low = float(df_15m["low"].iloc[i])
            candle_high = float(df_15m["high"].iloc[i])
            candle_close = float(df_15m["close"].iloc[i])
        except (IndexError, KeyError):
            continue

        if ob.direction == Direction.BULLISH:
            # Wick touched or entered OB zone AND close is above OB top
            if candle_low <= ob.top and candle_close > ob.top:
                return True
        elif ob.direction == Direction.BEARISH:
            # Wick touched or entered OB zone AND close is below OB bottom
            if candle_high >= ob.bottom and candle_close < ob.bottom:
                return True

    return False


# ─── Full SMC Analysis Pipeline ───────────────────────────────────────────

def run_smc_analysis(
    ohlcv: dict[str, pd.DataFrame],
    indicators: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the complete SMC analysis pipeline for a single coin.

    v2: Timeframe-specific lookbacks, body-based OBs, liquidity sweeps,
    RSI divergence, OB rejection confirmation.

    Args:
        ohlcv: Dict of timeframe -> OHLCV DataFrame.
        indicators: Nested indicator dict from calculate_all_indicators().

    Returns:
        Dict of SMC analysis results keyed by structure type.
    """
    from analysis.indicators import calc_atr, calc_fibonacci_zones

    results: dict[str, Any] = {}

    # ── 4H: Swing Points for Fibonacci zones ──
    df_4h = ohlcv.get(settings.MACRO_TF)
    if df_4h is not None and len(df_4h) >= 10:
        sh_4h, sl_4h = detect_swing_points(df_4h, lookback=settings.SWING_LOOKBACK_1H)
        results["4h_swing_highs"] = sh_4h
        results["4h_swing_lows"] = sl_4h

        # Fibonacci zones (replaces premium/discount midpoint)
        current_price = float(df_4h["close"].iloc[-1])
        fib_zones = calc_fibonacci_zones(sh_4h, sl_4h, current_price)
        results["fib_zones"] = fib_zones

    # ── 1H: Swing Points, BOS, CHoCH, Order Blocks ──
    df_1h = ohlcv.get(settings.STRUCTURE_TF)
    if df_1h is not None and len(df_1h) >= 20:
        sh_1h, sl_1h = detect_swing_points(df_1h, lookback=settings.SWING_LOOKBACK_1H)
        results["1h_swing_highs"] = sh_1h
        results["1h_swing_lows"] = sl_1h

        bos = detect_bos(df_1h, sh_1h, sl_1h)
        results["1h_bos"] = bos

        choch = detect_choch(df_1h, sh_1h, sl_1h)
        results["1h_choch"] = choch

        # Order Blocks on 1H (v2: body-based)
        atr_1h = indicators.get("atr", {}).get("1h_series")
        if atr_1h is None:
            atr_1h = calc_atr(df_1h["high"], df_1h["low"], df_1h["close"])
        bull_obs, bear_obs = detect_order_blocks(df_1h, atr_1h)
        bull_obs = invalidate_order_blocks(bull_obs, df_1h)
        bear_obs = invalidate_order_blocks(bear_obs, df_1h)

        # Update touch counts
        bull_obs = update_ob_touch_counts(bull_obs, df_1h)
        bear_obs = update_ob_touch_counts(bear_obs, df_1h)

        results["1h_bull_obs"] = bull_obs
        results["1h_bear_obs"] = bear_obs

    # ── 15m: FVGs, Swing Points (for sweep/divergence), Sweep, Divergence ──
    df_15m = ohlcv.get(settings.SETUP_TF)
    if df_15m is not None and len(df_15m) >= 10:
        # FVGs
        bull_fvgs, bear_fvgs = detect_fvgs(df_15m)
        bull_fvgs = invalidate_fvgs(bull_fvgs, df_15m)
        bear_fvgs = invalidate_fvgs(bear_fvgs, df_15m)
        results["15m_bull_fvgs"] = bull_fvgs
        results["15m_bear_fvgs"] = bear_fvgs

        # 15m Swing Points (for sweep and divergence)
        sh_15m, sl_15m = detect_swing_points(df_15m, lookback=settings.SWING_LOOKBACK_15M)
        results["15m_swing_highs"] = sh_15m
        results["15m_swing_lows"] = sl_15m

        # Liquidity Sweep
        atr_15m = indicators.get("atr", {}).get("15m_series")
        sweep = detect_liquidity_sweep(df_15m, sl_15m, sh_15m, atr_15m)
        results["liquidity_sweep"] = sweep

        # RSI Divergence
        rsi_15m = indicators.get("rsi", {}).get("15m_series")
        if rsi_15m is not None:
            divergence = detect_rsi_divergence(df_15m, rsi_15m, sl_15m, sh_15m)
            results["rsi_divergence"] = divergence
        else:
            results["rsi_divergence"] = {"bullish_divergence": False, "bearish_divergence": False}

        # OB Rejection Confirmation (check each active OB)
        for obs_key in ["1h_bull_obs", "1h_bear_obs"]:
            for ob in results.get(obs_key, []):
                ob.confirmed_rejection = check_ob_rejection_confirmation(df_15m, ob)

    return results
