"""
backtest/config.py — All backtest-specific settings and parameter grid.

This is the single configuration file for the backtesting system.
It deliberately duplicates nothing from the live bot's config/settings.py —
it only defines parameters that are unique to the backtest.
"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_BACKTEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _BACKTEST_ROOT.parent

DATA_DIR: str = str(_BACKTEST_ROOT / "data")
RESULTS_DIR: str = str(_BACKTEST_ROOT / "results")
COIN_LIST_FILE: str = str(_BACKTEST_ROOT / "data" / "coin_list.txt")
DB_PATH: str = str(_BACKTEST_ROOT / "results" / "backtest_results.db")
HTML_REPORT_PATH: str = str(_BACKTEST_ROOT / "results" / "backtest_report.html")

# ---------------------------------------------------------------------------
# BACKTEST PERIOD
# ---------------------------------------------------------------------------
BACKTEST_PERIOD_MONTHS: int = 6
WARMUP_4H_CANDLES: int = 200          # no signals generated during warmup period

# ---------------------------------------------------------------------------
# COIN SELECTION
# ---------------------------------------------------------------------------
TOP_N_COINS: int = 10                 # use top 10 coins by 24h volume

# ---------------------------------------------------------------------------
# OUTCOME TRACKING
# ---------------------------------------------------------------------------
OUTCOME_WINDOW_CANDLES: int = 288     # 24h in 5m candles (12 × 24)
COOLDOWN_HOURS: int = 4               # same as live bot

# ---------------------------------------------------------------------------
# DISABLED FILTERS (data not available for backtesting)
# ---------------------------------------------------------------------------
USE_OI_FILTER: bool = False           # OI history not downloaded
USE_LIVE_LS_RATIO: bool = False       # L/S ratio not available historically

# ---------------------------------------------------------------------------
# DOWNLOAD SETTINGS
# ---------------------------------------------------------------------------
DOWNLOAD_DELAY_SECONDS: float = 0.2  # delay between API calls (safety)
KLINES_PER_CALL: int = 1500          # Binance max per call
BINANCE_FUTURES_BASE: str = "https://fapi.binance.com"

# Timeframe → interval string mapping for Binance API
TIMEFRAME_INTERVALS: dict[str, str] = {
    "4h": "4h",
    "1h": "1h",
    "15m": "15m",
    "5m": "5m",
}

# ---------------------------------------------------------------------------
# PARAMETER GRID (64 combinations = 4 × 4 × 2 × 2)
# ---------------------------------------------------------------------------
MIN_SCORE_OPTIONS: list[int] = [8, 10, 12, 14]
ATR_SL_MULT_OPTIONS: list[float] = [0.75, 1.0, 1.25, 1.5]
MIN_RR_OPTIONS: list[float] = [1.5, 2.0]
MAX_SL_PCT_OPTIONS: list[float] = [0.025, 0.03]   # 2.5% and 3.0%

# ---------------------------------------------------------------------------
# RANKING WEIGHTS (must sum to 1.0)
# ---------------------------------------------------------------------------
WIN_RATE_WEIGHT: float = 0.35
PROFIT_FACTOR_WEIGHT: float = 0.30
SHARPE_WEIGHT: float = 0.20
SIGNAL_COUNT_WEIGHT: float = 0.15

# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
MIN_SIGNALS_FOR_VALIDITY: int = 100   # warn if best combo has fewer
OVERFITTING_THRESHOLD: float = 0.30   # warn if best > 3rd by more than 30%
MAX_EXPIRY_RATE_WARN: float = 0.30    # warn if >30% signals expire


def generate_parameter_grid() -> list[dict[str, Any]]:
    """
    Generate all 64 parameter combinations.

    Returns:
        List of dicts, each with combo_id, min_score, atr_sl_mult,
        min_rr, max_sl_pct.
    """
    combos: list[dict[str, Any]] = []
    combo_id = 1
    for min_score, atr_mult, min_rr, max_sl_pct in product(
        MIN_SCORE_OPTIONS,
        ATR_SL_MULT_OPTIONS,
        MIN_RR_OPTIONS,
        MAX_SL_PCT_OPTIONS,
    ):
        combos.append({
            "combo_id": combo_id,
            "min_score": min_score,
            "atr_sl_mult": atr_mult,
            "min_rr": min_rr,
            "max_sl_pct": max_sl_pct,
        })
        combo_id += 1
    return combos
