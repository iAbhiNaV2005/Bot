"""
backtest/candidate.py — CandidateSignal dataclass.

This is the full record produced by Phase 1 (walk-forward simulation).
It stores every raw scoring input so Phase 2 can re-score without
re-running any SMC or indicator calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CandidateSignal:
    """
    A candidate trading signal from the walk-forward simulation.

    All raw scoring inputs are stored as measured values, NOT as boolean
    pass/fail. Phase 2 applies its own thresholds during re-scoring.

    IDENTITY
    --------
    coin:           Symbol string (e.g. 'BTCUSDT')
    direction:      'long' or 'short'
    timestamp:      UTC datetime of the 5m candle where signal was generated
    candle_idx_5m:  Integer index in the 5m array
    session:        Trading session name ('Asian', 'London', 'Overlap', 'NY', 'Late')

    PRICE
    -----
    entry_price:      Current close price at signal time
    sl_raw_distance:  ATR × 1.0 (raw SL offset — Phase 2 multiplies by its own mult)
    ob_bottom:        Bottom of the active bullish OB (or top of bearish OB)
    ob_top:           Top of the active bullish OB (or top of bearish OB)
    sl_percentage:    SL% using default multiplier (reference only)

    MANDATORY GATE RESULTS (all must be True in Phase 2)
    -----------------------------------------------------
    gate_ema_4h_structure:  4H EMA 50 > EMA 200 (long) or < EMA 200 (short)
    gate_bos_1h:            1H BOS in correct direction
    gate_ob_present:        Price inside 1H OB
    gate_adx_min:           ADX >= 20
    gate_fib_zone:          Price in correct Fibonacci zone
    gate_rsi_range:         5m RSI in correct range and trending correctly
    gate_funding:           Funding rate within acceptable bounds

    SCORING INPUTS (raw measured values)
    -------------------------------------
    fvg_present:             bool
    volume_ratio_15m:        actual ratio (e.g. 1.73)
    volume_ratio_5m:         actual ratio
    oi_increasing:           always False (disabled in backtest)
    choch_present:           bool
    macd_histogram_trend:    True if pre-crossover pattern valid
    ls_ratio:                actual value (e.g. 0.84)
    funding_rate:            actual float (e.g. -0.0008)
    ema200_gap_pct:          (price - ema200) / ema200
    sweep_detected:          bool
    rsi_divergence:          bool
    adx_value:               actual ADX float (e.g. 28.4)
    ob_rejection_candle:     bool
    ob_touch_count:          int (1 = fresh, 2 = second touch)
    fib_deep_discount:       bool (in 61.8–78.6% zone)
    btc_bullish:             bool
    btc_bearish:             bool
    equal_level_present:     bool
    equal_level_count:       int (2 or 3+)
    ob_volume_quality:       bool (impulse volume >= 2× avg)

    OUTCOME (filled during walk-forward as candles advance)
    -------------------------------------------------------
    outcome:                  'TP1_HIT', 'SL_HIT', 'TP2_HIT', 'EXPIRED', 'PENDING'
    tp1_price:                TP1 price used in the simulation (default ATR mult)
    tp2_price:                TP2 price
    sl_price:                 SL price used in the simulation
    tp1_hit:                  bool
    tp2_hit:                  bool
    sl_hit:                   bool
    candles_to_outcome:       None if PENDING, else int
    max_adverse_excursion:    worst price move against trade direction (distance)
    max_favorable_excursion:  best price move in trade direction (distance)
    """

    # ── Identity ──
    coin: str
    direction: str
    timestamp: datetime
    candle_idx_5m: int
    session: str

    # ── Price ──
    entry_price: float
    sl_raw_distance: float   # ATR × 1.0 at signal time
    ob_bottom: float
    ob_top: float
    sl_percentage: float
    tp1_price: float         # with default mult=1.0
    tp2_price: float

    # ── Mandatory gates ──
    gate_ema_4h_structure: bool = False
    gate_bos_1h: bool = False
    gate_ob_present: bool = False
    gate_adx_min: bool = False
    gate_fib_zone: bool = False
    gate_rsi_range: bool = False
    gate_funding: bool = False

    # ── Scoring inputs (raw values) ──
    fvg_present: bool = False
    volume_ratio_15m: float = 0.0
    volume_ratio_5m: float = 0.0
    oi_increasing: bool = False     # always False in backtest
    choch_present: bool = False
    macd_histogram_trend: bool = False
    ls_ratio: float = 1.0
    funding_rate: float = 0.0
    ema200_gap_pct: float = 0.0
    sweep_detected: bool = False
    rsi_divergence: bool = False
    adx_value: float = 0.0
    ob_rejection_candle: bool = False
    ob_touch_count: int = 0
    fib_deep_discount: bool = False
    btc_bullish: bool = False
    btc_bearish: bool = False
    equal_level_present: bool = False
    equal_level_count: int = 0
    ob_volume_quality: bool = False

    # ── Outcome ──
    outcome: str = "PENDING"
    tp1_hit: bool = False
    tp2_hit: bool = False
    sl_hit: bool = False
    candles_to_outcome: Optional[int] = None
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    sl_price: float = 0.0    # resolved SL used in simulation
