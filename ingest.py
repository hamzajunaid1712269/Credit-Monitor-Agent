"""Data ingestion: SEC EDGAR filings and Google News RSS. No AI here."""
import hashlib
import time
from urllib.parse import quote_plus

import feedparser
import requests

from config import SEC_USER_AGENT, FILING_FORMS, NEWS_PER_ISSUER, FILINGS_PER_ISSUER

HEADERS = {"User-Agent": SEC_USER_AGENT}
_cik_cache: dict[str, str] = {}

# Plain-language names for 8-K items, so titles (and triage) carry real information.
EIGHT_K_ITEMS = {
    "1.01": "material agreement", "1.02": "agreement terminated", "1.03": "bankruptcy",
    "2.01": "acquisition/disposal", "2.02": "results", "2.03": "new debt obligation",
    "2.04": "debt acceleration trigger", "2.05": "exit/restructuring costs", "2.06": "impairment",
    "3.01": "delisting notice", "3.03": "change to security holders' rights",
    "4.01": "auditor change", "4.02": "prior financials unreliable",
    "5.01": "change in control", "5.02": "officer/director change", "5.03": "bylaw change",
    "5.07": "shareholder vote", "5.08": "shareholder nominations",
    "7.01": "Reg FD disclosure", "8.01": "other events", "9.01": "exhibits",
}


def _item_id(*parts: str) -> str:
    """Stable short ID from the parts, used to deduplicate items."""
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def form_of(item: dict) -> str:
    """'8-K filed 2026-10-02 ...' -> '8-K'. Empty for news."""
    return item["title"].split(" ", 1)[0] if item.get("source") == "sec" else ""


def eight_k_items(item: dict) -> set[str]:
    """Item numbers stored in the title, e.g. '8-K filed ... [items 2.03, 9.01]'."""
    t = item.get("title", "")
    if "[items " not in t:
        return set()
    inner = t.split("[items ", 1)[1].split("]", 1)[0]
    return {p.strip().split(" ")[0] for p in inner.split(",") if p.strip()}


def get_cik(ticker: str) -> str:
    """Map a ticker to the 10-digit CIK EDGAR uses."""
    if not _cik_cache:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=20)
        r.raise_for_status()
        for row in r.json().values():
            _cik_cache[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return _cik_cache[ticker.upper()]


def fetch_filings(ticker: str, limit: int = FILINGS_PER_ISSUER) -> list[dict]:
    """Most recent credit-relevant filings for one issuer."""
    cik = get_cik(ticker)
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=20)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    n = len(recent["form"])
    out = []
    for form, acc, date, doc, items in zip(
        recent["form"], recent["accessionNumber"], recent["filingDate"],
        recent["primaryDocument"], recent.get("items", [""] * n),
    ):
        if form not in FILING_FORMS:
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        title = f"{form} filed {date}"
        if form == "8-K" and items:
            codes = [c.strip() for c in items.split(",") if c.strip()]
            title += " [items " + ", ".join(f"{c} {EIGHT_K_ITEMS.get(c, '')}".strip() for c in codes) + "]"
        out.append({
            "id": _item_id(ticker, acc),
            "ticker": ticker,
            "source": "sec",
            "title": title,
            "url": url,
            "published": date,
        })
        if len(out) >= limit:
            break
    time.sleep(0.2)  # SEC asks for <10 requests/second
    return out


def fetch_news(ticker: str, name: str, limit: int = NEWS_PER_ISSUER) -> list[dict]:
    """Recent headlines from Google News RSS."""
    q = quote_plus(f'"{name}" (debt OR bonds OR rating OR earnings OR lawsuit OR CEO)')
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    return [{
        "id": _item_id(ticker, e.link),
        "ticker": ticker,
        "source": "news",
        "title": e.title,
        "url": e.link,
        "published": e.get("published", ""),
    } for e in feed.entries[:limit]]
