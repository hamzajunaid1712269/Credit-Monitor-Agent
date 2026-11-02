"""Run the full monitor once:  python pipeline.py  [--tickers F CCL] [--dry-run]

ingest -> dedupe -> triage -> agent analysis -> store
"""
import argparse

import db
import ingest
from config import WATCHLIST


def collect(tickers: list[str]) -> list[dict]:
    items = []
    for t in tickers:
        try:
            items += ingest.fetch_filings(t)
        except Exception as e:
            print(f"[{t}] filings failed: {e}")
        try:
            items += ingest.fetch_news(t, WATCHLIST[t])
        except Exception as e:
            print(f"[{t}] news failed: {e}")
    return items


def run(tickers: list[str], dry_run: bool = False):
    db.init()
    items = [i for i in collect(tickers) if db.is_new(i["id"])]
    print(f"{len(items)} new items")
    if dry_run:
        for i in items:
            print(f"  [{i['ticker']}|{i['source']}] {i['title']}")
        return

    from agent import analyse, triage  # imported here so --dry-run needs no API key

    for i in items:
        t = triage(i)
        db.save_item(i, t.relevant, t.reason)
        if not t.relevant:
            print(f"  skip  [{i['ticker']}] {i['title'][:80]}")
            continue
        try:
            ev, tools_used = analyse(i)
        except Exception as e:
            print(f"  ERROR [{i['ticker']}] {e}")
            continue
        db.save_event(i["id"], i["ticker"], ev.model_dump(mode="json"))
        print(f"  {ev.direction.value:8} sev{ev.severity} [{i['ticker']}] {ev.summary[:90]}"
              f"  (tools: {', '.join(tools_used) or 'none'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=list(WATCHLIST))
    p.add_argument("--dry-run", action="store_true", help="fetch and list items, no LLM calls")
    a = p.parse_args()
    run(a.tickers, a.dry_run)
