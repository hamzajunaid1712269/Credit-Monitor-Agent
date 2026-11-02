"""Structured outputs. These Pydantic models are what the LLM must return."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Kinds of credit event the agent can report."""
    rating_action = "rating_action"
    refinancing = "refinancing"          # new issuance, tender, exchange offer, maturity extension
    covenant_liquidity = "covenant_liquidity"  # covenant breach/waiver, liquidity stress, going concern
    earnings_guidance = "earnings_guidance"
    m_and_a = "m_and_a"
    management_change = "management_change"
    legal_regulatory = "legal_regulatory"
    restructuring_default = "restructuring_default"
    other = "other"


class Direction(str, Enum):
    """Effect on creditworthiness from a bondholder's view."""
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Triage(BaseModel):
    """Cheap first pass: is this item worth a credit analyst's time?"""
    relevant: bool = Field(description="True if the item could plausibly affect the issuer's credit risk")
    reason: str = Field(description="One short sentence explaining the decision")


class Assessment(BaseModel):
    """Submit your final credit assessment of the item. Call this once, when you are done."""
    event_type: EventType
    direction: Direction = Field(description="Effect on creditworthiness, from a bondholder's view")
    severity: int = Field(ge=1, le=5, description="1 = minor, 5 = could trigger rating action or default")
    summary: str = Field(description="One or two sentences, plain language, for a credit PM")
    rationale: str = Field(description="Why this direction and severity; name the evidence you used")


class CreditEvent(Assessment):
    """Assessment plus facts recorded by the system, not claimed by the model."""
    checked_filing: bool = Field(description="True only if filing text was actually retrieved")


# ---------- Tool argument schemas (bound to the model; executed by graph.tools_node) ----------

class ReadFilingSection(BaseModel):
    """Read sections of an SEC filing on one topic. Use it when the text you have does not
    cover what you need, e.g. liquidity or debt details in a 10-Q."""
    topic: Literal["liquidity", "debt", "risk", "results"] = Field(description="Which sections to read")
    filing: Literal["this_filing", "latest_10q"] = Field(
        default="this_filing",
        description="'this_filing' for the item being assessed (SEC items only); "
                    "'latest_10q' for the issuer's most recent quarterly report")


class SearchRecentNews(BaseModel):
    """Get the issuer's latest headlines, to corroborate a vague or dramatic item."""
    limit: int = Field(default=8, ge=1, le=15, description="How many headlines")


class GetIssuerHistory(BaseModel):
    """Credit events already recorded for this issuer. Use it to tell new information
    from a continuing story."""
    limit: int = Field(default=5, ge=1, le=20, description="How many past events")
