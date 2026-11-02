"""Run the monitor:  python pipeline.py [--tickers F CCL] [--max-items 5] [--dry-run] [--no-fetch]

fetch -> store as pending -> for each pending item: run the graph -> save result

Built for a free-tier key: every run has an LLM call budget (LLM_MAX_CALLS_PER_RUN).
When the budget or the provider's daily quota runs out, the run stops cleanly; the item
in progress is checkpointed and the next run resumes exactly where this one stopped.
"""
import argparse
import sqlite3
import sys

import config
import db
import ingest
from llm_guard import BudgetReached, QuotaExhausted, guard


def collect(tickers: list[str]) -> int:
    """Fetch filings and news, store new ones as pending. Returns how many were new."""
    before = sum(db.status_counts().values())
    for t in tickers:
        for label, fetch in (("filings", lambda: ingest.fetch_filings(t)),
                             ("news", lambda: ingest.fetch_news(t, config.WATCHLIST[t]))):
            try:
                for item in fetch():
                    db.upsert_new(item)
            except Exception as e:
                print(f"[{t}] {label} failed: {e}")
    return sum(db.status_counts().values()) - before


def _checkpointer():
    """SQLite-backed LangGraph checkpointer, so stopped items can resume."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver(sqlite3.connect(config.CHECKPOINT_DB_PATH, check_same_thread=False))


def run(tickers: list[str], max_items: int | None, dry_run: bool, fetch: bool) -> int:
    """Fetch, then process the queue until done or the budget runs out. Returns an exit code."""
    db.init()
    if fetch:
        print(f"{collect(tickers)} new items fetched")
    queue = db.todo(tickers)
    print(f"{len(queue)} items to process | LLM budget this run: {guard.max_calls} calls, "
          f"{guard.min_interval:.0f}s apart")
    if dry_run:
        for i in queue:
            print(f"  [{i['ticker']}|{i['source']}] {i['title']}")
        return 0
    if max_items:
        queue = queue[:max_items]

    from graph import build_graph, default_deps, run_item  # needs API key; not for --dry-run

    checkpointer = _checkpointer()
    app = build_graph(default_deps(), checkpointer)
    done = skipped = errors = 0
    stopped = None

    for i in queue:
        print(f"- [{i['ticker']}|{i['source']}] {i['title'][:90]}")
        try:
            state = run_item(app, i)
        except (BudgetReached, QuotaExhausted) as e:
            stopped = e
            break
        except Exception as e:
            db.mark_error(i["id"], f"{type(e).__name__}: {e}")
            errors += 1
            print(f"    ERROR {type(e).__name__}: {str(e)[:150]}")
            continue

        if state.get("event"):
            ev = state["event"]
            db.save_result(i["id"], i["ticker"], ev, relevant=state.get("relevant"),
                           triage_reason=state.get("triage_reason", ""),
                           tools_used=state.get("tools_used", []),
                           filing_read=bool(state.get("filing_read")),
                           filing_coverage=state.get("filing_coverage"))
            done += 1
            cov = state.get("filing_coverage")
            print(f"    {ev['direction']:8} sev{ev['severity']} {ev['event_type']}: {ev['summary'][:100]}")
            print(f"    tools: {', '.join(state.get('tools_used') or []) or 'none'}"
                  f" | filing read: {'yes' if state.get('filing_read') else 'no'}"
                  + (f" ({cov:.0%} of filing)" if cov is not None else ""))
        else:
            db.mark_skipped(i["id"], state.get("triage_reason", ""), state.get("relevant"))
            skipped += 1
            print(f"    skip: {state.get('triage_reason', '')[:100]}")
        try:
            checkpointer.delete_thread(i["id"])  # finished: free the checkpoint
        except Exception:
            pass

    print(f"\nDone {done}, skipped {skipped}, errors {errors} | LLM calls used: {guard.calls_made}")
    if stopped:
        kind = "Daily quota reached" if isinstance(stopped, QuotaExhausted) else "Run budget reached"
        left = len(db.todo(tickers))
        print(f"{kind}. Progress is saved; {left} items left. Run again later to continue.")
        return 2
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=list(config.WATCHLIST))
    p.add_argument("--max-items", type=int, default=None, help="process at most N items this run")
    p.add_argument("--dry-run", action="store_true", help="fetch and list items, no LLM calls")
    p.add_argument("--no-fetch", action="store_true", help="only work through the existing queue")
    a = p.parse_args()
    unknown = [t for t in a.tickers if t not in config.WATCHLIST]
    if unknown:
        sys.exit(f"Not in WATCHLIST (config.py): {', '.join(unknown)}")
    sys.exit(run(a.tickers, a.max_items, a.dry_run, not a.no_fetch))
