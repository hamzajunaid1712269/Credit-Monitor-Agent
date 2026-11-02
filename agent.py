"""The AI layer.

1. triage(): a cheap single LLM call that filters noise.
2. analyst agent: a tool-using LangChain agent that decides what extra evidence
   it needs (read the filing, check other news, look at the issuer's history)
   before returning a structured CreditEvent.
"""
import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

import db
import ingest
from config import MODEL, WATCHLIST
from schemas import CreditEvent, Triage

llm = init_chat_model(MODEL, temperature=0)

# ---------- 1. Triage ----------

_triage_llm = llm.with_structured_output(Triage)

TRIAGE_PROMPT = """You screen items for a corporate credit analyst covering {name} ({ticker}).
Mark an item relevant if it could plausibly change the issuer's credit risk: debt issuance or
refinancing, rating actions, liquidity, covenants, earnings/guidance, M&A, major litigation,
leadership changes, restructuring. Routine filings with no new information, product reviews,
and stock-price chatter are NOT relevant.

Item source: {source}
Title: {title}"""


def triage(item: dict) -> Triage:
    return _triage_llm.invoke(TRIAGE_PROMPT.format(name=WATCHLIST[item["ticker"]], **item))


# ---------- 2. Tools for the analyst agent ----------

@tool
def read_filing(url: str) -> str:
    """Read the text of an SEC filing (first ~15k characters). Use for 8-K/10-Q/10-K items,
    or when a headline is vague and you need the primary source."""
    try:
        return ingest.fetch_filing_text(url)
    except Exception as e:
        return f"Could not fetch filing: {e}"


@tool
def search_recent_news(ticker: str) -> str:
    """Get the latest headlines for an issuer to corroborate or add context to an item."""
    try:
        items = ingest.fetch_news(ticker, WATCHLIST[ticker], limit=8)
        return "\n".join(f"- {i['published']}: {i['title']}" for i in items) or "No news found."
    except Exception as e:
        return f"News search failed: {e}"


@tool
def get_issuer_history(ticker: str) -> str:
    """Previously recorded credit events for this issuer. Use to judge whether
    an item is new information or a continuation of an existing story."""
    hist = db.previous_events(ticker)
    return json.dumps(hist, indent=1) if hist else "No prior events recorded."


SYSTEM_PROMPT = """You are a junior credit analyst at an asset manager. For each item, assess its
impact on the issuer's creditworthiness from a bondholder's perspective.

Work like an analyst:
- For SEC filings, read the filing before judging. Filing titles alone say little.
- For vague or dramatic headlines, corroborate with recent news or the filing.
- Check the issuer's history to tell new information from repeated news.
- Do not invent numbers. If evidence is thin, say so and keep severity low.
- Severity: 1 minor, 2 worth noting, 3 material, 4 likely rating pressure, 5 default/restructuring risk.
Use as few tool calls as needed (usually 0-3)."""

analyst = create_agent(
    model=llm,
    tools=[read_filing, search_recent_news, get_issuer_history],
    system_prompt=SYSTEM_PROMPT,
    response_format=CreditEvent,
)


def analyse(item: dict) -> tuple[CreditEvent, list[str]]:
    """Run the agent on one item. Returns the event and the names of tools it called."""
    msg = (f"Issuer: {WATCHLIST[item['ticker']]} ({item['ticker']})\n"
           f"Source: {item['source']}\nTitle: {item['title']}\nURL: {item['url']}\n"
           f"Published: {item['published']}")
    result = analyst.invoke({"messages": [{"role": "user", "content": msg}]})
    tools_used = [tc["name"] for m in result["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
    return result["structured_response"], tools_used
