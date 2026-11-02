"""Keeps the pipeline alive on a free-tier API key.

Every LLM call in the project goes through `guard.call(...)`, which:
  1. spaces calls out (LLM_MIN_INTERVAL seconds apart),
  2. enforces a per-run budget, so a run stops cleanly before the daily quota,
  3. retries timeouts and per-minute rate limits with exponential backoff,
  4. recognises "daily quota used up" and stops instead of retrying for hours.

Provider-agnostic: errors are classified from their text, so it works the same for
Gemini, OpenAI, Anthropic or Groq. No LangChain import, so it is unit-testable offline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import config

T = TypeVar("T")


class BudgetReached(Exception):
    """This run used its LLM call budget. Progress is checkpointed; run again later."""


class QuotaExhausted(Exception):
    """The provider's daily quota is used up. Progress is checkpointed; run again later."""


_RATE_LIMIT = ("429", "resource_exhausted", "rate limit", "rate_limit", "too many requests")
_TRANSIENT = ("timeout", "timed out", "deadline_exceeded", "unavailable", "overloaded", "connection reset", "connection aborted",
              "internal error", "remote end closed")
_DAILY = ("per day", "perday", "daily")


def parse_retry_seconds(text: str) -> float | None:
    """Pull the provider's suggested wait out of an error message, if there is one.

    Handles Gemini's "Please retry in 13h29m8.59s" and "'retryDelay': '48548s'",
    and the generic "retry after 20 seconds".
    """
    t = text.lower()
    m = re.search(r"retrydelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s", t)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in\s+((?:\d+(?:\.\d+)?[hms]\s*)+)", t)
    if m:
        total = 0.0
        for num, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)):
            total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
        return total
    m = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", t)
    if m:
        return float(m.group(1))
    return None


def classify(err: Exception) -> str:
    """Return 'daily_quota', 'rate_limit', 'transient' or 'fatal'."""
    text = f"{type(err).__name__} {err}".lower()
    if any(k in text for k in _RATE_LIMIT):
        wait = parse_retry_seconds(text)
        if any(k in text for k in _DAILY) or (wait is not None and wait > config.LLM_MAX_WAIT):
            return "daily_quota"
        return "rate_limit"
    if any(k in text for k in _TRANSIENT) or re.search(r"\b(500|502|503|504)\b", text):
        return "transient"
    return "fatal"


@dataclass
class LLMGuard:
    """Wraps LLM calls with spacing, retries, a run budget and quota detection."""
    min_interval: float = config.LLM_MIN_INTERVAL
    max_calls: int = config.LLM_MAX_CALLS_PER_RUN
    max_retries: int = config.LLM_MAX_RETRIES
    backoff_base: float = config.LLM_BACKOFF_BASE
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    calls_made: int = 0
    log: list[str] = field(default_factory=list)
    _last_call: float | None = None

    @property
    def remaining(self) -> int:
        """LLM calls left in this run's budget."""
        return max(self.max_calls - self.calls_made, 0)

    def _wait_for_slot(self):
        """Sleep until min_interval has passed since the last call."""
        if self._last_call is not None:
            gap = self.clock() - self._last_call
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)

    def call(self, fn: Callable[[], T], label: str = "llm") -> T:
        """Run one LLM call under the spacing, budget and retry rules."""
        for attempt in range(self.max_retries + 1):
            if self.calls_made >= self.max_calls:
                raise BudgetReached(f"used {self.calls_made}/{self.max_calls} LLM calls this run")
            self._wait_for_slot()
            self.calls_made += 1  # failed attempts count too: the provider counts them
            self._last_call = self.clock()
            try:
                return fn()
            except Exception as e:
                kind = classify(e)
                if kind == "fatal":
                    raise
                if kind == "daily_quota":
                    raise QuotaExhausted(str(e)[:300]) from e
                if attempt == self.max_retries:
                    raise
                suggested = parse_retry_seconds(str(e))
                wait = max(suggested or 0, self.backoff_base * (2 ** attempt))
                self.log.append(f"{label}: {kind}, retry {attempt + 1} in {wait:.0f}s")
                print(f"    ... {label}: {kind}, waiting {wait:.0f}s before retry {attempt + 1}")
                self.sleep(wait)
        raise RuntimeError("unreachable")


# One shared guard per process, so every node draws from the same budget.
guard = LLMGuard()
