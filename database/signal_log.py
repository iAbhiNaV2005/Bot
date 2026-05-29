"""
database/signal_log.py — SQLite read/write for signal history.

Stores all signals ever sent, daily performance logs, and error logs.
Auto-creates tables on first run.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalLogger:
    """
    Persistent signal storage using SQLite.

    Stores every signal sent to Telegram for historical analysis
    and cooldown enforcement backup.
    """

    def __init__(self, db_path: str = settings.DB_PATH) -> None:
        """
        Initialize the signal logger and create tables if needed.

        Args:
            db_path: Absolute path to the SQLite database file.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._create_tables()
        logger.info(f"Signal logger initialized at {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL,
                    score INTEGER NOT NULL,
                    max_score INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    funding_rate REAL,
                    oi_change REAL,
                    ls_ratio REAL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    current_price REAL,
                    pnl_pct REAL DEFAULT 0.0,
                    closed_at TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_signals INTEGER DEFAULT 0,
                    long_signals INTEGER DEFAULT 0,
                    short_signals INTEGER DEFAULT 0,
                    high_confidence INTEGER DEFAULT 0,
                    coins_scanned INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol
                    ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp
                    ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp
                    ON signals(symbol, timestamp);
            """)

            # Migrate old schemas first (adds status column if missing)
            self._migrate_schema(conn)

            # Create status index after migration ensures column exists
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_status
                ON signals(status)
            """)

            conn.commit()
        finally:
            conn.close()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables if they don't exist."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        migrations = [
            ("status", "TEXT NOT NULL DEFAULT 'ACTIVE'"),
            ("current_price", "REAL"),
            ("pnl_pct", "REAL DEFAULT 0.0"),
            ("closed_at", "TEXT"),
            # v2: Failure tracking
            ("failure_detected", "INTEGER DEFAULT 0"),
            ("failure_detected_at", "TEXT"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}")
                logger.info(f"Migrated signals table: added column '{col_name}'")

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: Optional[float],
        score: int,
        max_score: int,
        confidence: str,
        funding_rate: Optional[float] = None,
        oi_change: Optional[float] = None,
        ls_ratio: Optional[float] = None,
    ) -> int:
        """
        Log a signal to the database.

        Args:
            symbol: Trading pair (e.g., 'ETHUSDT').
            direction: 'LONG' or 'SHORT'.
            entry_price: Suggested entry price.
            stop_loss: Stop loss price.
            tp1: Take profit level 1.
            tp2: Take profit level 2.
            tp3: Take profit level 3 (optional).
            score: Signal score achieved.
            max_score: Maximum possible score.
            confidence: 'STANDARD' or 'HIGH'.
            funding_rate: Current funding rate.
            oi_change: OI change percentage.
            ls_ratio: Long/short ratio.

        Returns:
            Row ID of the inserted signal.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO signals
                   (symbol, direction, entry_price, stop_loss, tp1, tp2, tp3,
                    score, max_score, confidence, funding_rate, oi_change,
                    ls_ratio, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, direction, entry_price, stop_loss, tp1, tp2, tp3,
                    score, max_score, confidence, funding_rate, oi_change,
                    ls_ratio, now,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(
                f"Signal logged: {direction} {symbol} | Score: {score}/{max_score} | ID: {row_id}"
            )
            return row_id  # type: ignore[return-value]
        finally:
            conn.close()

    def log_error(
        self,
        module: str,
        error_type: str,
        message: str,
        stack_trace: Optional[str] = None,
    ) -> None:
        """
        Log an error to the database.

        Args:
            module: Module where the error occurred.
            error_type: Type of exception.
            message: Error message.
            stack_trace: Full stack trace string.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO errors (module, error_type, message, stack_trace)
                   VALUES (?, ?, ?, ?)""",
                (module, error_type, message, stack_trace),
            )
            conn.commit()
        finally:
            conn.close()

    def get_signals_since(self, symbol: str, hours: int) -> list[dict[str, Any]]:
        """
        Get all signals for a symbol within the last N hours.

        Args:
            symbol: Trading pair.
            hours: Number of hours to look back.

        Returns:
            List of signal dicts.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? AND timestamp > ? ORDER BY timestamp DESC",
                (symbol, cutoff),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_daily_summary(self, date_str: Optional[str] = None) -> dict[str, Any]:
        """
        Get signal summary for a specific date.

        Args:
            date_str: Date string in YYYY-MM-DD format. Defaults to today UTC.

        Returns:
            Dict with total_signals, long_signals, short_signals, etc.
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        conn = self._get_connection()
        try:
            row = conn.execute(
                """SELECT
                     COUNT(*) as total_signals,
                     SUM(CASE WHEN direction = 'LONG' THEN 1 ELSE 0 END) as long_signals,
                     SUM(CASE WHEN direction = 'SHORT' THEN 1 ELSE 0 END) as short_signals,
                     SUM(CASE WHEN confidence = 'HIGH' THEN 1 ELSE 0 END) as high_confidence,
                     AVG(score) as avg_score
                   FROM signals
                   WHERE DATE(timestamp) = ?""",
                (date_str,),
            ).fetchone()
            return dict(row) if row else {
                "total_signals": 0,
                "long_signals": 0,
                "short_signals": 0,
                "high_confidence": 0,
                "avg_score": 0,
            }
        finally:
            conn.close()

    def update_signal_status(
        self, signal_id: int, status: str, current_price: float, pnl_pct: float
    ) -> None:
        """
        Update signal status and current price.

        Args:
            signal_id: Row ID of the signal.
            status: New status (ACTIVE, TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, EXPIRED).
            current_price: Current market price.
            pnl_pct: Current P&L percentage.
        """
        closed_at = None
        if status != "ACTIVE":
            closed_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE signals
                   SET status = ?, current_price = ?, pnl_pct = ?, closed_at = ?
                   WHERE id = ?""",
                (status, current_price, pnl_pct, closed_at, signal_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_current_price(self, signal_id: int, current_price: float, pnl_pct: float) -> None:
        """
        Update only the current price and P&L for an active signal.

        Args:
            signal_id: Row ID of the signal.
            current_price: Current market price.
            pnl_pct: Current P&L percentage.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE signals SET current_price = ?, pnl_pct = ? WHERE id = ?",
                (current_price, pnl_pct, signal_id),
            )
            conn.commit()
        finally:
            conn.close()

    def check_and_update_statuses(self, price_map: dict[str, float]) -> None:
        """
        Check all active signals against current prices and update statuses.

        For LONG signals:
          - price <= SL → SL_HIT
          - price >= TP3 → TP3_HIT (if TP3 exists)
          - price >= TP2 → TP2_HIT
          - price >= TP1 → TP1_HIT
          - otherwise → update current_price and pnl

        For SHORT signals: reversed comparisons.

        Args:
            price_map: Dict mapping symbol to current price.
        """
        active = self.get_active_signals()

        for signal in active:
            symbol = signal["symbol"]
            if symbol not in price_map:
                continue

            current_price = price_map[symbol]
            entry = signal["entry_price"]
            sl = signal["stop_loss"]
            tp1 = signal["tp1"]
            tp2 = signal["tp2"]
            tp3 = signal["tp3"]
            direction = signal["direction"]
            signal_id = signal["id"]

            # Calculate P&L
            if direction == "LONG":
                pnl_pct = ((current_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current_price) / entry) * 100

            new_status = "ACTIVE"

            if direction == "LONG":
                if current_price <= sl:
                    new_status = "SL_HIT"
                elif tp3 is not None and current_price >= tp3:
                    new_status = "TP3_HIT"
                elif current_price >= tp2:
                    new_status = "TP2_HIT"
                elif current_price >= tp1:
                    new_status = "TP1_HIT"
            else:  # SHORT
                if current_price >= sl:
                    new_status = "SL_HIT"
                elif tp3 is not None and current_price <= tp3:
                    new_status = "TP3_HIT"
                elif current_price <= tp2:
                    new_status = "TP2_HIT"
                elif current_price <= tp1:
                    new_status = "TP1_HIT"

            if new_status != "ACTIVE":
                self.update_signal_status(signal_id, new_status, current_price, round(pnl_pct, 2))
                logger.info(
                    f"Signal #{signal_id} {direction} {symbol}: {new_status} "
                    f"(entry={entry}, current={current_price}, pnl={pnl_pct:+.2f}%)"
                )
            else:
                self.update_current_price(signal_id, current_price, round(pnl_pct, 2))

    def expire_old_signals(self, max_age_hours: int = 24) -> int:
        """
        Expire active signals older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before expiration.

        Returns:
            Number of signals expired.
        """
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """UPDATE signals SET status = 'EXPIRED',
                   closed_at = CURRENT_TIMESTAMP
                   WHERE status = 'ACTIVE' AND timestamp < ?""",
                (cutoff,),
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Expired {count} signals older than {max_age_hours}h")
            return count
        finally:
            conn.close()

    def get_active_signals(self) -> list[dict[str, Any]]:
        """
        Get all currently active signals.

        Returns:
            List of active signal dicts ordered by newest first.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE status = 'ACTIVE' ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all signals ordered by newest first.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List of signal dicts.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """
        Get overall signal statistics for the dashboard.

        Returns:
            Dict with total counts, win/loss stats, etc.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status LIKE 'TP%' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'SL_HIT' THEN 1 ELSE 0 END) as sl_hits,
                    SUM(CASE WHEN status = 'EXPIRED' THEN 1 ELSE 0 END) as expired,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN direction = 'LONG' THEN 1 ELSE 0 END) as longs,
                    SUM(CASE WHEN direction = 'SHORT' THEN 1 ELSE 0 END) as shorts,
                    AVG(CASE WHEN status != 'ACTIVE' THEN pnl_pct END) as avg_pnl,
                    SUM(CASE WHEN confidence = 'HIGH' THEN 1 ELSE 0 END) as high_conf
                FROM signals
            """).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()


    def get_pending_signals(self) -> list[dict[str, Any]]:
        """
        Get all active (pending) signals for failure checking.

        Returns:
            List of active signal dicts.
        """
        return self.get_active_signals()

    def mark_signal_failed(self, signal_id: int) -> None:
        """
        Mark a signal as FAILED (v2).

        Sets status to FAILED and records the failure timestamp.
        Used when price crosses SL + 0.5×ATR.

        Args:
            signal_id: Row ID of the signal to mark.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE signals
                   SET status = 'FAILED', failure_detected = 1,
                       failure_detected_at = ?, closed_at = ?
                   WHERE id = ?""",
                (now, now, signal_id),
            )
            conn.commit()
            logger.info(f"Signal #{signal_id} marked as FAILED")
        finally:
            conn.close()


# Module-level singleton
signal_logger = SignalLogger()
