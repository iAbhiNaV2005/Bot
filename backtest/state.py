"""
backtest/state.py — SimulationState class for walk-forward simulation.

One SimulationState instance per coin. It carries everything that the
live bot would hold in memory for that coin, accumulated across time
without look-ahead.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class PendingOutcome:
    """
    Tracks an active open trade waiting for TP or SL resolution.

    Attributes:
        candidate_idx:   Index into the candidates list for this coin.
        entry:           Entry price of the trade.
        tp1:             Take profit level 1.
        tp2:             Take profit level 2.
        sl:              Stop loss price.
        direction:       'long' or 'short'.
        signal_idx:      5m candle index when signal was generated.
        expires_at_idx:  5m candle index after which the trade expires
                         (signal_idx + OUTCOME_WINDOW_CANDLES).
    """
    candidate_idx: int
    entry: float
    tp1: float
    tp2: float
    sl: float
    direction: str
    signal_idx: int
    expires_at_idx: int


class SimulationState:
    """
    Per-coin simulation state for the walk-forward loop.

    Carries all accumulated SMC structures, indicator state, and
    tracking variables. Updated incrementally as candles advance —
    never reset (except cooldown on SL hit).
    """

    def __init__(self, coin: str) -> None:
        """
        Initialise empty state for a coin.

        Args:
            coin: Symbol string (e.g. 'BTCUSDT').
        """
        self.coin: str = coin

        # ── Timeframe index tracking ──
        self.idx_5m: int = 0
        self.idx_15m: int = 0
        self.idx_1h: int = 0
        self.idx_4h: int = 0

        # Previous values to detect timeframe closes
        self.prev_idx_15m: int = -1
        self.prev_idx_1h: int = -1
        self.prev_idx_4h: int = -1

        # ── Warmup state ──
        self.warmup_complete: bool = False

        # ── 4H indicators ──
        self.ema_4h_50: float = 0.0
        self.ema_4h_200: float = 0.0

        # ── 1H SMC structures (accumulated, updated each 1H close) ──
        self.swing_highs_1h: list[Any] = []
        self.swing_lows_1h: list[Any] = []
        self.order_blocks_bull_1h: list[Any] = []
        self.order_blocks_bear_1h: list[Any] = []
        self.bos_1h: Any = None
        self.choch_1h: Any = None
        self.fib_zones: Optional[dict] = None
        self.equal_highs_1h: list[Any] = []
        self.equal_lows_1h: list[Any] = []

        # ── 1H indicators ──
        self.ema_1h_21: float = 0.0
        self.ema_1h_50: float = 0.0
        self.adx_1h: float = 0.0

        # ── 15m SMC structures ──
        self.fvg_zones_bull_15m: list[Any] = []
        self.fvg_zones_bear_15m: list[Any] = []
        self.swing_highs_15m: list[Any] = []
        self.swing_lows_15m: list[Any] = []
        self.liquidity_sweep: dict = {}
        self.rsi_divergence: dict = {"bullish_divergence": False, "bearish_divergence": False}

        # ── 15m indicators ──
        self.atr_15m: float = 0.0
        self.volume_ratio_15m: float = 0.0

        # ── 5m indicators (updated every candle) ──
        self.rsi_5m_current: float = 50.0
        self.rsi_5m_last3: list[float] = [50.0, 50.0, 50.0]
        self.macd_histogram_last3: list[float] = [0.0, 0.0, 0.0]
        self.volume_ratio_5m: float = 1.0
        self.current_funding_rate: float = 0.0

        # ── BTC state (updated from BTC coin's state each 1H close) ──
        self.btc_bullish: bool = False
        self.btc_bearish: bool = False

        # ── Cooldown tracking ──
        self.cooldown_active: bool = False
        self.cooldown_until: Optional[datetime] = None
        self.last_signal_direction: Optional[str] = None
        self.last_signal_sl: Optional[float] = None
        self.last_signal_entry: Optional[float] = None

        # ── Active trade tracking ──
        self.pending_outcome: Optional[PendingOutcome] = None
