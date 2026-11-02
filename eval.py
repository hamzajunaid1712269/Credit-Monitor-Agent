"""Measure triage quality against hand labels.

1. Run the pipeline once.
2. python eval.py export   -> writes labels.csv with all items and the model's decision
3. Fill the `human_relevant` column with 1/0 (and optionally `human_event_type`).
4. python eval.py score    -> prints precision / recall / accuracy
"""
import sqlite3
import sys

import pandas as pd

from config import DB_PATH


def export():
    """Write every item and the model's decision to labels.csv for hand labelling."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT i.id, i.ticker, i.source, i.title, i.relevant AS model_relevant, "
        "e.event_type AS model_event_type FROM items i LEFT JOIN events e ON e.item_id=i.id", con)
    df["human_relevant"] = ""
    df["human_event_type"] = ""
    df.to_csv("labels.csv", index=False)
    print(f"Wrote {len(df)} rows to labels.csv. Label them, then run: python eval.py score")


def score():
    # Only rows the triage step actually judged (news); SEC filings are always assessed.
    """Compare hand labels with model decisions and print the metrics."""
    df = pd.read_csv("labels.csv").dropna(subset=["human_relevant", "model_relevant"])
    y, p = df.human_relevant.astype(int), df.model_relevant.astype(int)
    tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
    print(f"Labelled items: {len(df)}")
    print(f"Triage accuracy:  {(y == p).mean():.0%}")
    print(f"Precision:        {tp / max(tp + fp, 1):.0%}   (of items flagged, how many mattered)")
    print(f"Recall:           {tp / max(tp + fn, 1):.0%}   (of items that mattered, how many caught)")
    et = df.dropna(subset=["human_event_type", "model_event_type"])
    if len(et):
        print(f"Event-type accuracy on {len(et)} items: {(et.human_event_type == et.model_event_type).mean():.0%}")


if __name__ == "__main__":
    {"export": export, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
