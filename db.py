"""SQLite storage: raw items, their processing status, and the assessed events.

Item status lifecycle:
    pending   -> fetched, not finished yet (also: interrupted by budget/quota, retried next run)
    skipped   -> filtered out (by the routine-8-K rule or by LLM triage); final
    done      -> event saved; final
    error     -> failed for a non-retryable reason; retried next run up to MAX_ATTEMPTS
"""
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

MAX_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, ticker TEXT, source TEXT, title TEXT, url TEXT,
    published TEXT, relevant INTEGER, triage_reason TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, last_error TEXT
);
CREATE TABLE IF NOT EXISTS events (
    item_id TEXT PRIMARY KEY REFERENCES items(id), ticker TEXT,
    event_type TEXT, direction TEXT, severity INTEGER,
    summary TEXT, rationale TEXT, checked_filing INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    tools_used TEXT, filing_coverage REAL
);
"""

# Columns added after the first version; added in place so existing databases keep their data.
_MIGRATIONS = {
    "items": {"status": "TEXT DEFAULT 'pending'", "attempts": "INTEGER DEFAULT 0", "last_error": "TEXT"},
    "events": {"tools_used": "TEXT", "filing_coverage": "REAL"},
}


@contextmanager
def conn():
    """Open a connection, commit on success, always close."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    """Create tables, add new columns to old databases, and repair item statuses."""
    with conn() as c:
        c.executescript(SCHEMA)
        for table, cols in _MIGRATIONS.items():
            existing = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        # Repair rows from the first version: triaged as relevant but never analysed
        # (the "lost item" bug). Mark them pending so they are retried.
        c.execute("""UPDATE items SET status='pending'
                     WHERE relevant=1 AND id NOT IN (SELECT item_id FROM events)
                       AND (status IS NULL OR status='pending')""")
        c.execute("""UPDATE items SET status='done' WHERE id IN (SELECT item_id FROM events)""")
        c.execute("""UPDATE items SET status='skipped' WHERE relevant=0 AND status='pending'""")


def upsert_new(item: dict) -> None:
    """Record a fetched item as pending. Never overwrites an item we already know."""
    with conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO items (id,ticker,source,title,url,published,status) "
            "VALUES (?,?,?,?,?,?, 'pending')",
            (item["id"], item["ticker"], item["source"], item["title"], item["url"], item["published"]),
        )


def todo(tickers: list[str]) -> list[dict]:
    """Items still to process: pending, or errored fewer than MAX_ATTEMPTS times.
    Filings first (primary sources), then newest first."""
    q = ",".join("?" * len(tickers))
    with conn() as c:
        rows = c.execute(
            f"""SELECT id,ticker,source,title,url,published FROM items
                WHERE ticker IN ({q})
                  AND (status='pending' OR (status='error' AND attempts < ?))
                ORDER BY CASE source WHEN 'sec' THEN 0 ELSE 1 END, fetched_at DESC, published DESC""",
            (*tickers, MAX_ATTEMPTS),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_skipped(item_id: str, reason: str, relevant: bool | None = False):
    """Mark an item as filtered out (final)."""
    with conn() as c:
        c.execute("UPDATE items SET status='skipped', relevant=?, triage_reason=? WHERE id=?",
                  (None if relevant is None else int(relevant), reason, item_id))


def mark_error(item_id: str, error: str):
    """Record a failure; the item is retried until MAX_ATTEMPTS."""
    with conn() as c:
        c.execute("UPDATE items SET status='error', attempts=attempts+1, last_error=? WHERE id=?",
                  (error[:500], item_id))


def save_result(item_id: str, ticker: str, ev: dict, *, relevant: bool | None, triage_reason: str,
                tools_used: list[str], filing_read: bool, filing_coverage: float | None):
    """Save the event and mark the item done in one transaction, so nothing is half-saved."""
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO events (item_id,ticker,event_type,direction,severity,summary,"
            "rationale,checked_filing,created_at,tools_used,filing_coverage) "
            "VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?)",
            (item_id, ticker, ev["event_type"], ev["direction"], ev["severity"], ev["summary"],
             ev["rationale"], int(filing_read), ",".join(tools_used), filing_coverage),
        )
        c.execute("UPDATE items SET status='done', relevant=?, triage_reason=? WHERE id=?",
                  (None if relevant is None else int(relevant), triage_reason, item_id))


def previous_events(ticker: str, limit: int = 5, exclude_item: str | None = None) -> list[dict]:
    """Most recent events for an issuer, optionally excluding the current item."""
    with conn() as c:
        rows = c.execute(
            "SELECT e.created_at, e.event_type, e.direction, e.severity, e.summary "
            "FROM events e WHERE e.ticker=? AND e.item_id IS NOT ? ORDER BY e.created_at DESC LIMIT ?",
            (ticker, exclude_item, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def status_counts() -> dict[str, int]:
    """How many items are in each status, e.g. {'done': 3, 'pending': 7}."""
    with conn() as c:
        return {r["status"]: r["n"] for r in
                c.execute("SELECT status, COUNT(*) n FROM items GROUP BY status")}
