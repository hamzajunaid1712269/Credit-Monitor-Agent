"""SQLite storage: raw items and the agent's assessed events."""
import json
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, ticker TEXT, source TEXT, title TEXT, url TEXT,
    published TEXT, relevant INTEGER, triage_reason TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    item_id TEXT PRIMARY KEY REFERENCES items(id), ticker TEXT,
    event_type TEXT, direction TEXT, severity INTEGER,
    summary TEXT, rationale TEXT, checked_filing INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def is_new(item_id: str) -> bool:
    with conn() as c:
        return c.execute("SELECT 1 FROM items WHERE id=?", (item_id,)).fetchone() is None


def save_item(item: dict, relevant: bool, reason: str):
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO items (id,ticker,source,title,url,published,relevant,triage_reason) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (item["id"], item["ticker"], item["source"], item["title"], item["url"],
             item["published"], int(relevant), reason),
        )


def save_event(item_id: str, ticker: str, ev: dict):
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            (item_id, ticker, ev["event_type"], ev["direction"], ev["severity"],
             ev["summary"], ev["rationale"], int(ev["checked_filing"])),
        )


def previous_events(ticker: str, limit: int = 5) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT e.created_at, e.event_type, e.direction, e.severity, e.summary "
            "FROM events e WHERE e.ticker=? ORDER BY e.created_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return [dict(r) for r in rows]
