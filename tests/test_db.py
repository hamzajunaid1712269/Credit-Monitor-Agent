import sqlite3

import db

OLD_SCHEMA = """
CREATE TABLE items (id TEXT PRIMARY KEY, ticker TEXT, source TEXT, title TEXT, url TEXT,
    published TEXT, relevant INTEGER, triage_reason TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE events (item_id TEXT PRIMARY KEY, ticker TEXT, event_type TEXT, direction TEXT,
    severity INTEGER, summary TEXT, rationale TEXT, checked_filing INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
INSERT INTO items VALUES ('a','F','sec','8-K','u','p',1,'r',CURRENT_TIMESTAMP);
INSERT INTO items VALUES ('b','F','sec','10-Q','u','p',1,'r',CURRENT_TIMESTAMP);
INSERT INTO items VALUES ('c','F','news','x','u','p',0,'r',CURRENT_TIMESTAMP);
INSERT INTO events VALUES ('a','F','other','neutral',1,'s','r',1,CURRENT_TIMESTAMP);
"""

EV = {"event_type": "other", "direction": "neutral", "severity": 1, "summary": "s", "rationale": "r"}


def use_tmp_db(tmp_path):
    db.DB_PATH = str(tmp_path / "t.db")


def test_migration_recovers_lost_item(tmp_path):
    use_tmp_db(tmp_path)
    c = sqlite3.connect(db.DB_PATH)
    c.executescript(OLD_SCHEMA)
    c.close()
    db.init()
    assert db.status_counts() == {"done": 1, "pending": 1, "skipped": 1}
    assert [i["id"] for i in db.todo(["F"])] == ["b"]   # the lost 10-Q is back in the queue


def test_lifecycle(tmp_path):
    use_tmp_db(tmp_path)
    db.init()
    item = {"id": "n1", "ticker": "F", "source": "news", "title": "t", "url": "u", "published": "p"}
    db.upsert_new(item)
    db.upsert_new(item)  # duplicate fetch is ignored
    assert len(db.todo(["F"])) == 1
    for _ in range(db.MAX_ATTEMPTS):
        db.mark_error("n1", "boom")
    assert db.todo(["F"]) == []  # gives up after MAX_ATTEMPTS
    db.save_result("n1", "F", EV, relevant=True, triage_reason="r", tools_used=["analyst"],
                   filing_read=False, filing_coverage=None)
    assert db.status_counts() == {"done": 1}
    assert db.previous_events("F", exclude_item="n1") == []
    assert len(db.previous_events("F")) == 1
