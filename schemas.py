"""Structured outputs. These Pydantic models are what the LLM must return."""
from enum import Enum
from pydantic import BaseModel, Field


class EventType(str, Enum):
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
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Triage(BaseModel):
    """Cheap first pass: is this item worth a credit analyst's time?"""
    relevant: bool = Field(description="True if the item could plausibly affect the issuer's credit risk")
    reason: str = Field(description="One short sentence explaining the decision")


class CreditEvent(BaseModel):
    """Final assessment produced by the agent."""
    event_type: EventType
    direction: Direction = Field(description="Effect on creditworthiness, from a bondholder's view")
    severity: int = Field(ge=1, le=5, description="1 = minor, 5 = could trigger rating action or default")
    summary: str = Field(description="One or two sentences, plain language, for a credit PM")
    rationale: str = Field(description="Why this direction and severity; mention evidence checked")
    checked_filing: bool = Field(description="True if the agent read the underlying filing")
