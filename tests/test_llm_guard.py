import pytest

from llm_guard import BudgetReached, LLMGuard, QuotaExhausted, classify, parse_retry_seconds

GEMINI_DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
                "limit: 20, description: Generate content requests per day limit exceeded on Free Tier. "
                "Please retry in 13h29m8.597835264s. 'retryDelay': '48548s'")
GEMINI_MINUTE = "429 RESOURCE_EXHAUSTED. Please retry in 21.5s. 'retryDelay': '21s'"


class FakeClock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make_guard(**kw):
    c = FakeClock()
    return LLMGuard(sleep=c.sleep, clock=c.clock, **{"min_interval": 10, "max_calls": 10,
                                                      "max_retries": 3, "backoff_base": 5, **kw}), c


def test_parse_retry_seconds():
    assert parse_retry_seconds(GEMINI_DAILY) == 48548
    assert parse_retry_seconds("Please retry in 1h2m3s") == 3723
    assert parse_retry_seconds("retry after 20 seconds") == 20
    assert parse_retry_seconds("nothing here") is None


def test_classify():
    assert classify(Exception(GEMINI_DAILY)) == "daily_quota"
    assert classify(Exception(GEMINI_MINUTE)) == "rate_limit"
    assert classify(TimeoutError("Read timed out")) == "transient"
    assert classify(Exception("503 UNAVAILABLE")) == "transient"
    assert classify(ValueError("invalid schema, limit 5000 chars")) == "fatal"


def test_spacing_between_calls():
    g, c = make_guard()
    g.call(lambda: 1)
    g.call(lambda: 2)
    assert c.slept == [10]


def test_retries_transient_then_succeeds():
    g, c = make_guard()
    attempts = iter([TimeoutError("timed out"), TimeoutError("timed out"), "ok"])

    def fn():
        r = next(attempts)
        if isinstance(r, Exception):
            raise r
        return r

    assert g.call(fn) == "ok"
    assert g.calls_made == 3
    assert 5 in c.slept and 10 in c.slept  # exponential backoff 5s, 10s


def test_rate_limit_uses_provider_delay_when_longer():
    g, c = make_guard(backoff_base=1)
    attempts = iter([Exception(GEMINI_MINUTE), "ok"])

    def fn():
        r = next(attempts)
        if isinstance(r, Exception):
            raise r
        return r

    assert g.call(fn) == "ok"
    assert 21 in c.slept


def test_daily_quota_stops_immediately():
    g, c = make_guard()
    with pytest.raises(QuotaExhausted):
        g.call(lambda: (_ for _ in ()).throw(Exception(GEMINI_DAILY)))
    assert g.calls_made == 1 and c.slept == []


def test_budget_stops_run():
    g, _ = make_guard(max_calls=2)
    g.call(lambda: 1)
    g.call(lambda: 1)
    with pytest.raises(BudgetReached):
        g.call(lambda: 1)


def test_fatal_error_not_retried():
    g, _ = make_guard()
    with pytest.raises(ValueError):
        g.call(lambda: (_ for _ in ()).throw(ValueError("bad request")))
    assert g.calls_made == 1
