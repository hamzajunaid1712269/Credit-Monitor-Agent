"""Dashboard:  streamlit run app.py"""
import sqlite3

import pandas as pd
import streamlit as st

from config import DB_PATH, WATCHLIST

st.set_page_config(page_title="Credit Monitor", layout="wide")
st.title("Credit News & Filings Monitor")

try:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT e.*, i.title, i.url, i.source, i.published FROM events e JOIN items i ON i.id=e.item_id",
        con,
    )
    stats = pd.read_sql("SELECT COUNT(*) n, SUM(relevant) rel FROM items", con).iloc[0]
except Exception:
    st.info("No data yet. Run `python pipeline.py` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Items screened", int(stats.n or 0))
c2.metric("Credit-relevant", int(stats.rel or 0))
c3.metric("High severity (4-5)", int((df.severity >= 4).sum()))

with st.sidebar:
    tickers = st.multiselect("Issuers", list(WATCHLIST), default=list(WATCHLIST))
    min_sev = st.slider("Minimum severity", 1, 5, 1)
    directions = st.multiselect("Direction", ["negative", "neutral", "positive"],
                                default=["negative", "neutral", "positive"])

view = df[df.ticker.isin(tickers) & (df.severity >= min_sev) & df.direction.isin(directions)]
view = view.sort_values(["severity", "created_at"], ascending=[False, False])

st.subheader("Issuer overview")
if not view.empty:
    overview = view.groupby("ticker").agg(
        events=("item_id", "count"),
        max_severity=("severity", "max"),
        negative=("direction", lambda s: (s == "negative").sum()),
    )
    st.dataframe(overview, use_container_width=True)

st.subheader("Events")
icons = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
for _, r in view.iterrows():
    with st.expander(f"{icons[r.direction]} sev {r.severity} · {r.ticker} · {r.event_type} · {r.title[:90]}"):
        st.write(r.summary)
        st.caption(f"Rationale: {r.rationale}")
        st.caption(f"Source: {r.source} · {r.published} · filing read: {'yes' if r.checked_filing else 'no'}")
        st.markdown(f"[Open source]({r.url})")
