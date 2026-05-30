"""
backtest/db.py — SQLite persistence for backtest results.

Saves all Phase 1 candidate signals and Phase 2 combination results
to a SQLite database. All tables are created fresh on each run
(existing DB is replaced).
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from backtest import config
from backtest.candidate import CandidateSignal
from backtest.rescorer import ComboResult, TradeResult


def _get_conn() -> sqlite3.Connection:
    """Open (or create) the backtest SQLite database."""
    db_path = Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db() -> None:
    """Create all tables, dropping existing ones for a clean run."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS signals_raw;
        DROP TABLE IF EXISTS combinations;
        DROP TABLE IF EXISTS trades;

        CREATE TABLE signals_raw (
            signal_id           INTEGER PRIMARY KEY,
            coin                TEXT,
            direction           TEXT,
            timestamp           TEXT,
            session             TEXT,
            candle_idx_5m       INTEGER,
            entry_price         REAL,
            sl_raw_distance     REAL,
            ob_bottom           REAL,
            ob_top              REAL,
            sl_percentage       REAL,
            tp1_price           REAL,
            tp2_price           REAL,
            sl_price            REAL,
            outcome             TEXT,
            tp1_hit             INTEGER,
            tp2_hit             INTEGER,
            sl_hit              INTEGER,
            candles_to_outcome  INTEGER,
            max_adverse_excursion  REAL,
            max_favorable_excursion REAL,
            -- gate results
            gate_ema_4h_structure  INTEGER,
            gate_bos_1h            INTEGER,
            gate_ob_present        INTEGER,
            gate_adx_min           INTEGER,
            gate_fib_zone          INTEGER,
            gate_rsi_range         INTEGER,
            gate_funding           INTEGER,
            -- scoring inputs
            fvg_present            INTEGER,
            volume_ratio_15m       REAL,
            volume_ratio_5m        REAL,
            oi_increasing          INTEGER,
            choch_present          INTEGER,
            macd_histogram_trend   INTEGER,
            ls_ratio               REAL,
            funding_rate           REAL,
            ema200_gap_pct         REAL,
            sweep_detected         INTEGER,
            rsi_divergence         INTEGER,
            adx_value              REAL,
            ob_rejection_candle    INTEGER,
            ob_touch_count         INTEGER,
            fib_deep_discount      INTEGER,
            btc_bullish            INTEGER,
            btc_bearish            INTEGER,
            equal_level_present    INTEGER,
            equal_level_count      INTEGER,
            ob_volume_quality      INTEGER
        );

        CREATE TABLE combinations (
            combo_id                INTEGER PRIMARY KEY,
            min_score               INTEGER,
            atr_sl_mult             REAL,
            min_rr                  REAL,
            max_sl_pct              REAL,
            total_signals           INTEGER,
            long_signals            INTEGER,
            short_signals           INTEGER,
            signals_per_month       REAL,
            tp1_hit_count           INTEGER,
            sl_hit_count            INTEGER,
            expired_count           INTEGER,
            win_rate                REAL,
            expiry_rate             REAL,
            profit_factor           REAL,
            sharpe_ratio            REAL,
            avg_r                   REAL,
            total_r                 REAL,
            max_r                   REAL,
            min_r                   REAL,
            r_std_dev               REAL,
            max_consecutive_losses  INTEGER,
            max_consecutive_wins    INTEGER,
            composite_score         REAL,
            rank                    INTEGER,
            per_coin_json           TEXT,
            per_session_json        TEXT,
            per_direction_json      TEXT,
            per_month_json          TEXT
        );

        CREATE TABLE trades (
            trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            combo_id    INTEGER REFERENCES combinations(combo_id),
            signal_id   INTEGER,
            coin        TEXT,
            direction   TEXT,
            timestamp   TEXT,
            session     TEXT,
            score       INTEGER,
            entry_price REAL,
            tp1_price   REAL,
            sl_price    REAL,
            rr_ratio    REAL,
            outcome     TEXT,
            r_achieved  REAL
        );
    """)

    conn.commit()
    conn.close()


def save_candidates(candidates: list[CandidateSignal]) -> None:
    """Save all Phase 1 candidate signals to signals_raw table."""
    conn = _get_conn()
    cur = conn.cursor()

    rows = []
    for idx, c in enumerate(candidates):
        rows.append((
            idx,
            c.coin, c.direction, c.timestamp.isoformat(), c.session,
            c.candle_idx_5m, c.entry_price, c.sl_raw_distance,
            c.ob_bottom, c.ob_top, c.sl_percentage,
            c.tp1_price, c.tp2_price, c.sl_price,
            c.outcome,
            int(c.tp1_hit), int(c.tp2_hit), int(c.sl_hit),
            c.candles_to_outcome,
            c.max_adverse_excursion, c.max_favorable_excursion,
            # gates
            int(c.gate_ema_4h_structure), int(c.gate_bos_1h),
            int(c.gate_ob_present), int(c.gate_adx_min),
            int(c.gate_fib_zone), int(c.gate_rsi_range), int(c.gate_funding),
            # scoring inputs
            int(c.fvg_present), c.volume_ratio_15m, c.volume_ratio_5m,
            int(c.oi_increasing), int(c.choch_present),
            int(c.macd_histogram_trend), c.ls_ratio, c.funding_rate,
            c.ema200_gap_pct, int(c.sweep_detected), int(c.rsi_divergence),
            c.adx_value, int(c.ob_rejection_candle), c.ob_touch_count,
            int(c.fib_deep_discount), int(c.btc_bullish), int(c.btc_bearish),
            int(c.equal_level_present), c.equal_level_count,
            int(c.ob_volume_quality),
        ))

    cur.executemany("""
        INSERT INTO signals_raw VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, rows)
    conn.commit()
    conn.close()


def save_results(combo_results: list[ComboResult]) -> None:
    """Save all Phase 2 combination results and their trades."""
    conn = _get_conn()
    cur = conn.cursor()

    for result in combo_results:
        cur.execute("""
            INSERT INTO combinations VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            result.combo_id,
            result.params["min_score"],
            result.params["atr_sl_mult"],
            result.params["min_rr"],
            result.params["max_sl_pct"],
            result.total_signals,
            result.long_signals,
            result.short_signals,
            result.signals_per_month,
            result.tp1_hit_count,
            result.sl_hit_count,
            result.expired_count,
            result.win_rate,
            result.expiry_rate,
            result.profit_factor,
            result.sharpe_ratio,
            result.avg_r_per_trade,
            result.total_r,
            result.max_r,
            result.min_r,
            result.r_std_dev,
            result.max_consecutive_losses,
            result.max_consecutive_wins,
            result.composite_score,
            result.rank,
            json.dumps(result.per_coin),
            json.dumps(result.per_session),
            json.dumps(result.per_direction),
            json.dumps(result.per_month),
        ))

        # Insert trades for this combo
        trade_rows = []
        for trade in result.trades:
            trade_rows.append((
                result.combo_id, trade.signal_id,
                trade.coin, trade.direction,
                trade.timestamp.isoformat(), trade.session,
                trade.score, trade.entry_price,
                trade.tp1_price, trade.sl_price,
                trade.rr_ratio, trade.outcome, trade.r_achieved,
            ))
        if trade_rows:
            cur.executemany("""
                INSERT INTO trades (
                    combo_id, signal_id, coin, direction, timestamp, session,
                    score, entry_price, tp1_price, sl_price, rr_ratio, outcome, r_achieved
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, trade_rows)

    conn.commit()
    conn.close()


def load_candidates() -> list[dict]:
    """Load all signals_raw rows as dicts (used by --phase2-only)."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM signals_raw ORDER BY timestamp ASC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
