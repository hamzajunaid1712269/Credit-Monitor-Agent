"""Watchlist and settings. Edit WATCHLIST to change the issuers you monitor."""
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-4.1-mini")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Credit Monitor demo contact@example.com")
DB_PATH = os.getenv("DB_PATH", "credit_monitor.db")

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

NEWS_PER_ISSUER = 10
FILINGS_PER_ISSUER = 5
