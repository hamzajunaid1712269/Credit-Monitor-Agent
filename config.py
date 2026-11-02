"""Watchlist and settings. Edit WATCHLIST to change the issuers you monitor.

Every setting can be overridden in .env, so the code never needs editing to
switch provider or to fit a stricter free-tier quota.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    """Read an integer setting from the environment."""
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    """Read a float setting from the environment."""
    return float(os.getenv(name, default))


# ---------- Model ----------
MODEL = os.getenv("MODEL", "google_genai:gemini-3.8-flash")

# ---------- Data sources ----------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Credit Monitor demo contact@example.com")
DB_PATH = os.getenv("DB_PATH", "credit_monitor.db")
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")

# Issuers with SEC filings and active credit stories. Ticker -> display name.
WATCHLIST = {
    "F": "Ford Motor",
    "CCL": "Carnival Corporation",
    "BA": "Boeing",
    "INTC": "Intel",
    "AMC": "AMC Entertainment",
}

# Filing types that most often carry credit-relevant news
FILING_FORMS = {"8-K", "10-Q", "10-K"}

# 8-Ks that only contain these items are routine and skipped without an LLM call:
# 5.07 shareholder vote results, 5.08 shareholder nominations, 9.01 exhibits only.
ROUTINE_8K_ITEMS = {"5.07", "5.08", "9.01"}

NEWS_PER_ISSUER = _int("NEWS_PER_ISSUER", 10)
FILINGS_PER_ISSUER = _int("FILINGS_PER_ISSUER", 5)

# ---------- Free-tier survival ----------
# Seconds to wait between any two LLM calls (Gemini free tier allows only a few per minute).
LLM_MIN_INTERVAL = _float("LLM_MIN_INTERVAL", 12)
# Hard cap on LLM calls in one run. Keep it under your provider's daily quota.
LLM_MAX_CALLS_PER_RUN = _int("LLM_MAX_CALLS_PER_RUN", 18)
# Retries for timeouts / per-minute rate limits, with exponential backoff.
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 4)
LLM_BACKOFF_BASE = _float("LLM_BACKOFF_BASE", 15)  # 15s, 30s, 60s, 120s
# If the provider asks us to wait longer than this, treat it as "daily quota used up".
LLM_MAX_WAIT = _float("LLM_MAX_WAIT", 180)

# ---------- Agent ----------
MAX_TOOL_ROUNDS = _int("MAX_TOOL_ROUNDS", 3)
FILING_CHAR_BUDGET = _int("FILING_CHAR_BUDGET", 14000)
