"""Persistent Investor Notification and Alert Store backed by SQLite.

Provides:
- Persistent SQLite storage surviving server restarts and page refreshes.
- Duplicate prevention via deterministic cryptographic fingerprints.
- Unread notification tracking and bulk/single read marking.
- Pre-seeding of verified Form 4 institutional alerts.
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import ROOT

DEFAULT_DB_PATH = ROOT / "data/processed/notifications.db"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))


class NotificationStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    ticker TEXT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL, -- 'significant', 'requires_review', 'routine', 'info'
                    category TEXT NOT NULL, -- 'insider_selling', 'short_interest_spike', 'earnings', 'system'
                    metadata_json TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at DESC)")
            conn.commit()

    @staticmethod
    def generate_fingerprint(ticker: str, identifier: str, category: str = "insider_selling") -> str:
        """Create a deterministic SHA-256 fingerprint for deduplication."""
        raw = f"{category}:{ticker.upper().strip()}:{identifier.strip()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def create_notification(
        self,
        title: str,
        message: str,
        ticker: Optional[str] = None,
        severity: str = "requires_review",
        category: str = "insider_selling",
        metadata: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a notification if not already existing. Returns notification dict or None if duplicate."""
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_ticker = (ticker or "").strip().upper() or None
        meta_json = json.dumps(metadata or {}, default=str)

        if not fingerprint:
            fingerprint = self.generate_fingerprint(clean_ticker or "UNKNOWN", f"{title}:{message}", category)

        notif_id = f"notif_{fingerprint[:16]}"

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO notifications (
                        id, fingerprint, ticker, title, message, severity, category, metadata_json, is_read, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (notif_id, fingerprint, clean_ticker, title, message, severity, category, meta_json, now_iso))
                conn.commit()
                return self.get_notification_by_id(notif_id)
            except sqlite3.IntegrityError:
                # Duplicate fingerprint detected; reject insertion safely
                return None

    def get_notification_by_id(self, notif_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM notifications WHERE id = ?", (notif_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_notifications(self, limit: int = 50, unread_only: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM notifications"
        params: List[Any] = []
        if unread_only:
            query += " WHERE is_read = 0"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_dict(row) for row in cur.fetchall()]

    def get_unread_count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM notifications WHERE is_read = 0")
            return cur.fetchone()[0]

    def mark_as_read(self, notif_id: Optional[str] = None) -> int:
        """Mark a single notification or all notifications as read."""
        with self._get_connection() as conn:
            if notif_id:
                cur = conn.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))
            else:
                cur = conn.execute("UPDATE notifications SET is_read = 1 WHERE is_read = 0")
            conn.commit()
            return cur.rowcount

    def delete_notification(self, notif_id: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            if notif_id:
                cur = conn.execute("DELETE FROM notifications WHERE id = ?", (notif_id,))
            else:
                cur = conn.execute("DELETE FROM notifications")
            conn.commit()
            return cur.rowcount

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["is_read"] = bool(d["is_read"])
        try:
            d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
        except Exception:
            d["metadata"] = {}
        d.pop("metadata_json", None)
        return d

    def seed_initial_alerts(self, alerts: List[Dict[str, Any]]):
        """Seed verified high-priority alerts if database is fresh."""
        for alert in alerts:
            ticker = alert.get("ticker", "UNKNOWN")
            insider = alert.get("insider_name", "Corporate Insider")
            shares = alert.get("shares_sold", 0)
            date = alert.get("transaction_date", "")
            severity = alert.get("significance_category", "requires_review")
            val = alert.get("estimated_value")
            val_str = f"${val/1e6:.1f}M" if val and val >= 1e6 else (f"${val:,.0f}" if val else "")

            fingerprint = self.generate_fingerprint(ticker, f"{insider}:{date}:{shares}", "insider_selling")
            title = f"{ticker}: {insider} sold {shares:,.0f} shares ({val_str})"
            message = alert.get("investor_takeaway") or f"Form 4 filing analyzed with risk score {alert.get('risk_score', 0):.2f}."

            self.create_notification(
                title=title,
                message=message,
                ticker=ticker,
                severity=severity,
                category="insider_selling",
                metadata=alert,
                fingerprint=fingerprint
            )


# Singleton instance
_store: Optional[NotificationStore] = None

def get_notification_store() -> NotificationStore:
    global _store
    if _store is None:
        _store = NotificationStore()
    return _store

