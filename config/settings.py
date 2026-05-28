"""
config/settings.py — Single source of truth for all configurable values.

Every number, threshold, and parameter in the bot references this file.
No hardcoded values anywhere else in the codebase.
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
ATR_PERIOD: int = 14

# ---------------------------------------------------------------------------
# SMC SETTINGS
# ---------------------------------------------------------------------------
SWING_LOOKBACK: int = 2           # candles on each side to confirm swing
OB_IMPULSE_CANDLES: int = 3       # min consecutive candles in impulse move
OB_IMPULSE_MULTIPLIER: float = 1.5  # impulse must move >= 1.5x ATR
FVG_MAX_AGE_CANDLES: int = 50     # discard FVGs older than this
BOS_LOOKBACK: int = 20            # look for BOS in last N candles on 1H
MAX_OBS_STORED: int = 5           # keep last 5 valid OBs per coin
MAX_FVGS_STORED: int = 10         # keep last 10 valid FVGs per coin
MAX_SWINGS_STORED: int = 10       # keep last 10 swing highs/lows

# ---------------------------------------------------------------------------
# FILTER SETTINGS
# ---------------------------------------------------------------------------
FUNDING_RATE_LONG_MAX: float = 0.0005    # +0.05%
FUNDING_RATE_SHORT_MIN: float = -0.0005  # -0.05%
OI_LOOKBACK_HOURS: int = 2
VOLUME_RATIO_MIN: float = 1.5
ATR_FILTER_ACTIVE: bool = True
SIGNAL_COOLDOWN_HOURS: int = 4

# ---------------------------------------------------------------------------
# SCORING SETTINGS
# ---------------------------------------------------------------------------
MIN_SCORE_TO_SIGNAL: int = 8
HIGH_CONFIDENCE_SCORE: int = 13
MAX_POSSIBLE_SCORE: int = 18

# Scoring point values
SCORE_FVG_PRESENT: int = 2
SCORE_DISCOUNT_ZONE: int = 2
SCORE_15M_VOLUME: int = 1
SCORE_5M_VOLUME: int = 2
SCORE_OI_INCREASING: int = 2
SCORE_CHOCH: int = 3
SCORE_MACD_PRECROSS: int = 1
SCORE_LS_RATIO: int = 2
SCORE_FUNDING_BONUS: int = 2
SCORE_STRONG_BIAS: int = 1

# ---------------------------------------------------------------------------
# TP / SL SETTINGS
# ---------------------------------------------------------------------------
ATR_SL_MULTIPLIER: float = 1.0    # SL = 1 ATR from OB edge
TP1_RR: float = 1.5
TP2_RR: float = 3.0
TP3_RR: float = 5.0
MIN_RR_TO_SIGNAL: float = 1.5

# ---------------------------------------------------------------------------
# TELEGRAM SETTINGS
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_CHART: bool = True
TELEGRAM_SEND_DELAY: float = 1.5  # seconds between consecutive sends

# ---------------------------------------------------------------------------
# SCHEDULING SETTINGS
# ---------------------------------------------------------------------------
MAIN_LOOP_INTERVAL_SECONDS: int = 60
CANDLE_CLOSE_BUFFER_SECONDS: int = 2
MAX_API_ERRORS_BEFORE_ALERT: int = 5

# ---------------------------------------------------------------------------
# RATE LIMITING
# ---------------------------------------------------------------------------
BINANCE_WEIGHT_LIMIT_PER_MINUTE: int = 1200
SAFETY_WEIGHT_LIMIT: int = 600    # stay at 50% capacity — not a hard rule
OHLCV_BATCH_DELAY: float = 0.05   # seconds between OHLCV calls
OI_BATCH_DELAY: float = 0.1       # seconds between OI calls

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
DB_PATH: str = str(_PROJECT_ROOT / "database" / "signals.db")
LOG_DIR: str = str(_PROJECT_ROOT / "logs")
CHART_TMP_DIR: str = str(_PROJECT_ROOT / "tmp")
