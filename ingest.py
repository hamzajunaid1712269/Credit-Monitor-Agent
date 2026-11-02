"""Data ingestion: SEC EDGAR filings and Google News RSS. No AI here."""
import hashlib
import time
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

from config import SEC_USER_AGENT, FILING_FORMS, NEWS_PER_ISSUER, FILINGS_PER_ISSUER

HEADERS = {"User-Agent": SEC_USER_AGENT}
_cik_cache: dict[str, str] = {}


def _item_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


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
    out = []
    for form, acc, date, doc, desc in zip(
        recent["form"], recent["accessionNumber"], recent["filingDate"],
        recent["primaryDocument"], recent.get("primaryDocDescription", [""] * len(recent["form"])),
    ):
        if form not in FILING_FORMS:
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        out.append({
            "id": _item_id(ticker, acc),
            "ticker": ticker,
            "source": "sec",
            "title": f"{form} filed {date}" + (f" – {desc}" if desc else ""),
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


def fetch_filing_text(url: str, max_chars: int = 15000) -> str:
    """Download a filing and return plain text, truncated to keep LLM cost down."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    return text[:max_chars]
