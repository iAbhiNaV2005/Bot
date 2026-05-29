"""
config/settings.py — Single source of truth for all configurable values.

Every number, threshold, and parameter in the bot references this file.
No hardcoded values anywhere else in the codebase.

v2: Added SMC improvements — ADX, Fibonacci, liquidity sweep, session filter,
    BTC correlation, volatility blackout, OB touch tracking, RSI divergence,
    OB rejection confirmation, max SL cap, outcome tracking.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# EXCHANGE SETTINGS
# ---------------------------------------------------------------------------
EXCHANGE: str = "binance"
MARKET_TYPE: str = "futures"  # USDT-margined perpetuals
BASE_URL: str = "https://fapi.binance.com"

BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

# ---------------------------------------------------------------------------
# COIN SETTINGS
# ---------------------------------------------------------------------------
TOP_N_COINS: int = 50
COIN_FETCH_POOL: int = 60          # fetch extra to account for failures
COIN_REFRESH_INTERVAL_HOURS: int = 24
MIN_CANDLE_HISTORY: int = 200      # discard coins with less history
STABLECOIN_SYMBOLS: list[str] = [
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT",
    "FDUSDUSDT", "USDPUSDT",
]
COIN_LIST_FILE: str = str(_PROJECT_ROOT / "data" / "top_coins.json")

# ---------------------------------------------------------------------------
# TIMEFRAME SETTINGS
# ---------------------------------------------------------------------------
MACRO_TF: str = "4h"
STRUCTURE_TF: str = "1h"
SETUP_TF: str = "15m"
ENTRY_TF: str = "5m"

# How many candles to fetch per timeframe
CANDLE_LIMITS: dict[str, int] = {
    "4h": 200,
    "1h": 200,
    "15m": 200,
    "5m": 100,
}

# ---------------------------------------------------------------------------
# INDICATOR SETTINGS
# ---------------------------------------------------------------------------
EMA_PERIODS: list[int] = [9, 21, 50, 200]
RSI_PERIOD: int = 14
RSI_LONG_MIN: float = 28.0
RSI_LONG_MAX: float = 45.0
RSI_SHORT_MIN: float = 55.0
RSI_SHORT_MAX: float = 72.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
MACD_PRECROSS_LOOKBACK: int = 3    # check last 3 histogram values for convergence
ATR_PERIOD: int = 14

# ---------------------------------------------------------------------------
# ADX SETTINGS (v2)
# ---------------------------------------------------------------------------
ADX_PERIOD: int = 14
ADX_TIMEFRAME: str = "1h"
ADX_MIN_THRESHOLD: float = 20.0     # mandatory gate — skip below this
ADX_SCORE_STRONG: float = 25.0      # score +1 if ADX between 25 and 40
ADX_SCORE_VERY_STRONG: float = 40.0  # score +2 if ADX above this

# ---------------------------------------------------------------------------
# SMC SETTINGS
# ---------------------------------------------------------------------------
# Swing lookback — timeframe specific (v2: replaces single SWING_LOOKBACK=2)
SWING_LOOKBACK_1H: int = 4          # 8 hours of confirmation each side
SWING_LOOKBACK_15M: int = 3         # 45 minutes each side
SWING_LOOKBACK_5M: int = 2          # 10 minutes each side

OB_IMPULSE_CANDLES: int = 3         # min consecutive candles in impulse move
OB_IMPULSE_MULTIPLIER: float = 1.5  # impulse must move >= 1.5x ATR
FVG_MAX_AGE_CANDLES: int = 50       # discard FVGs older than this
BOS_LOOKBACK: int = 20              # look for BOS in last N candles on 1H
MAX_OBS_STORED: int = 5             # keep last 5 valid OBs per coin
MAX_FVGS_STORED: int = 10           # keep last 10 valid FVGs per coin
MAX_SWINGS_STORED: int = 10         # keep last 10 swing highs/lows

# Order Block v2 settings
OB_USE_BODY_ONLY: bool = True       # use open/close for zone, not high/low
OB_MAX_TOUCH_COUNT: int = 3         # auto-invalidate on 3rd touch
OB_INVALIDATION_ON_CLOSE: bool = True  # only close through body invalidates OB

# ---------------------------------------------------------------------------
# FIBONACCI SETTINGS (v2 — replaces 50% midpoint)
# ---------------------------------------------------------------------------
FIB_TIMEFRAME: str = "4h"
FIB_OPTIMAL_LONG_LOW: float = 0.50    # price must be in 50–61.8% retracement zone
FIB_OPTIMAL_LONG_HIGH: float = 0.618
FIB_DEEP_DISCOUNT_HIGH: float = 0.786  # extra bonus if in deep discount
FIB_OPTIMAL_SHORT_LOW: float = 0.382   # price in 38.2–50% retracement zone for shorts
FIB_OPTIMAL_SHORT_HIGH: float = 0.50

# ---------------------------------------------------------------------------
# LIQUIDITY SWEEP SETTINGS (v2)
# ---------------------------------------------------------------------------
SWEEP_BONUS_SCORE: int = 3
SWEEP_SIGNAL_WINDOW_CANDLES: int = 5  # signal must come within 5 candles of sweep
SWEEP_MIN_WICK_ATR_MULT: float = 0.1  # minimum wick = 0.1 × ATR

# ---------------------------------------------------------------------------
# RSI DIVERGENCE SETTINGS (v2)
# ---------------------------------------------------------------------------
DIVERGENCE_SCAN_CANDLES: int = 30      # look back 30 candles on 15m
DIVERGENCE_BONUS_SCORE: int = 3
DIVERGENCE_SIGNAL_PROXIMITY: int = 5   # divergence swing within 5 candles of current

# ---------------------------------------------------------------------------
# SESSION FILTER SETTINGS (v2)
# ---------------------------------------------------------------------------
SESSION_ASIAN_START_UTC: int = 0
SESSION_ASIAN_END_UTC: int = 7         # inclusive, 00:00–07:59
SESSION_LONDON_START_UTC: int = 8
SESSION_LONDON_END_UTC: int = 12       # inclusive, 08:00–12:59
SESSION_OVERLAP_START_UTC: int = 13
SESSION_OVERLAP_END_UTC: int = 16      # inclusive, 13:00–16:59
SESSION_NY_START_UTC: int = 17
SESSION_NY_END_UTC: int = 20           # inclusive, 17:00–20:59
SESSION_ASIAN_MIN_SCORE: int = 12      # skip if Asian session AND score < this
SESSION_ASIAN_SCORE: int = -2
SESSION_LONDON_SCORE: int = 2
SESSION_OVERLAP_SCORE: int = 3
SESSION_NY_SCORE: int = 1
SESSION_LATE_SCORE: int = -1           # 21:00–23:59 UTC

# ---------------------------------------------------------------------------
# VOLATILITY BLACKOUT SETTINGS (v2)
# ---------------------------------------------------------------------------
BLACKOUT_ATR_MULTIPLIER: float = 2.5   # BTC ATR exceeds 2.5× average → blackout
BLACKOUT_DURATION_MINUTES: int = 60
BLACKOUT_CHECK_SYMBOL: str = "BTC/USDT:USDT"

# ---------------------------------------------------------------------------
# BTC CORRELATION SETTINGS (v2)
# ---------------------------------------------------------------------------
BTC_CORRELATION_SYMBOL: str = "BTC/USDT:USDT"
BTC_BEARISH_SCORE_PENALTY: int = -2    # applied to all altcoin long signals
BTC_BULLISH_SCORE_BONUS: int = 1       # applied to all altcoin long signals

# ---------------------------------------------------------------------------
# OB CONFIRMATION SETTINGS (v2)
# ---------------------------------------------------------------------------
OB_REJECTION_CONFIRMATION_SCORE: int = 2  # bonus if rejection candle detected

# ---------------------------------------------------------------------------
# FILTER SETTINGS
# ---------------------------------------------------------------------------
FUNDING_RATE_LONG_MAX: float = 0.0005    # +0.05%
FUNDING_RATE_SHORT_MIN: float = -0.0005  # -0.05%
OI_LOOKBACK_HOURS: int = 2
VOLUME_RATIO_MIN: float = 1.5
ATR_FILTER_ACTIVE: bool = True
SIGNAL_COOLDOWN_HOURS: int = 4

# Cooldown failure settings (v2)
COOLDOWN_FAILURE_ATR_MULTIPLIER: float = 0.5  # price crosses SL + 0.5×ATR = FAILED
COOLDOWN_FAILURE_CHECK_INTERVAL_MINUTES: int = 15

# ---------------------------------------------------------------------------
# SCORING SETTINGS (v2 — updated)
# ---------------------------------------------------------------------------
MIN_SCORE_TO_SIGNAL: int = 10       # raised from 8 — more conditions available
HIGH_CONFIDENCE_SCORE: int = 18     # raised from 13
MAX_POSSIBLE_SCORE: int = 32        # raised from 18

# Original scoring point values (kept)
SCORE_FVG_PRESENT: int = 2
SCORE_DISCOUNT_ZONE: int = 2        # now Fibonacci zone, same weight
SCORE_15M_VOLUME: int = 1
SCORE_5M_VOLUME: int = 2
SCORE_OI_INCREASING: int = 2
SCORE_CHOCH: int = 3
SCORE_MACD_PRECROSS: int = 1
SCORE_LS_RATIO: int = 2
SCORE_FUNDING_BONUS: int = 2
SCORE_STRONG_BIAS: int = 1

# New scoring point values (v2)
SCORE_SWEEP: int = 3                # liquidity sweep detected
SCORE_RSI_DIVERGENCE: int = 3       # RSI divergence at OB
SCORE_ADX_STRONG: int = 1           # ADX between 25 and 40
SCORE_ADX_VERY_STRONG: int = 2      # ADX above 40 (replaces SCORE_ADX_STRONG)
SCORE_OB_REJECTION: int = 2         # OB rejection candle confirmed
SCORE_OB_FRESH: int = 2             # OB touch_count == 1 (first touch)
SCORE_DEEP_FIB: int = 1             # price in deep discount (61.8–78.6%)
SCORE_BTC_BULLISH: int = 1          # BTC bullish structure (altcoin longs)
SCORE_BTC_BEARISH: int = -2         # BTC bearish structure (altcoin long penalty)

# ---------------------------------------------------------------------------
# TP / SL SETTINGS
# ---------------------------------------------------------------------------
ATR_SL_MULTIPLIER: float = 1.0      # SL = 1 ATR from OB edge
TP1_RR: float = 1.5
TP2_RR: float = 3.0
TP3_RR: float = 5.0
MIN_RR_TO_SIGNAL: float = 1.5
MAX_SL_PERCENTAGE: float = 0.03     # 3% maximum SL distance from entry (v2)

# ---------------------------------------------------------------------------
# TELEGRAM SETTINGS
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_CHART: bool = True
TELEGRAM_SEND_DELAY: float = 1.5    # seconds between consecutive sends

# ---------------------------------------------------------------------------
# SCHEDULING SETTINGS
# ---------------------------------------------------------------------------
MAIN_LOOP_INTERVAL_SECONDS: int = 60
CANDLE_CLOSE_BUFFER_SECONDS: int = 2
MAX_API_ERRORS_BEFORE_ALERT: int = 5

# ---------------------------------------------------------------------------
# OUTCOME TRACKING SETTINGS (v2)
# ---------------------------------------------------------------------------
OUTCOME_CHECK_INTERVAL_MINUTES: int = 30
OUTCOME_CHECK_DURATION_HOURS: int = 24

# ---------------------------------------------------------------------------
# RATE LIMITING
# ---------------------------------------------------------------------------
BINANCE_WEIGHT_LIMIT_PER_MINUTE: int = 1200
SAFETY_WEIGHT_LIMIT: int = 600      # stay at 50% capacity — not a hard rule
OHLCV_BATCH_DELAY: float = 0.05     # seconds between OHLCV calls
OI_BATCH_DELAY: float = 0.1         # seconds between OI calls

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
DB_PATH: str = str(_PROJECT_ROOT / "database" / "signals.db")
LOG_DIR: str = str(_PROJECT_ROOT / "logs")
CHART_TMP_DIR: str = str(_PROJECT_ROOT / "tmp")
