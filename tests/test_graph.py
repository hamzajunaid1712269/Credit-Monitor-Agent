"""Graph tests with a scripted fake LLM: no API key, no network, no quota used.

Run these first after any change to graph.py:  pytest -q
"""
import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402

try:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
except ImportError:  # older langgraph
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # noqa: E402

from graph import Deps, build_graph, run_item  # noqa: E402
from llm_guard import BudgetReached, LLMGuard  # noqa: E402
from schemas import Assessment, Triage  # noqa: E402

ASSESS = {"event_type": "earnings_guidance", "direction": "neutral", "severity": 1,
          "summary": "Short supplier issue.", "rationale": "Within guidance."}

TEN_Q_TEXT = "\n".join(["FORM 10-Q", "x " * 9000, "Liquidity and Capital Resources",
                        "Liquidity was $46.1 billion.", "y " * 500, "Item 6. Exhibits"])


def tool_call(name, args, i="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])


class FakeLLM:
    """Implements just what graph.py uses: with_structured_output(...).invoke and bind_tools(...).invoke."""

    def __init__(self, triage=None, analyst=(), final=None):
        self.triage, self.analyst, self.final = triage, list(analyst), final
        self.calls = []

    def with_structured_output(self, schema):
        llm = self

        class S:
            def invoke(self, _input):
                llm.calls.append(schema.__name__)
                return llm.triage if schema is Triage else llm.final

        return S()

    def bind_tools(self, _tools):
        llm = self

        class T:
            def invoke(self, _msgs):
                llm.calls.append("analyst")
                return llm.analyst.pop(0)

        return T()


def make(llm, max_calls=50, checkpointer=None, filing_text=TEN_Q_TEXT):
    deps = Deps(
        llm=llm,
        fetch_filing_text=lambda url: filing_text,
        fetch_news=lambda t, n: [{"published": "today", "title": "Ford beats estimates"}],
        issuer_history=lambda t, n, ex: [],
        latest_filing_url=lambda t, form: "https://sec.example/10q.htm",
        guard=LLMGuard(min_interval=0, max_calls=max_calls, sleep=lambda s: None),
    )
    return build_graph(deps, checkpointer or InMemorySaver()), deps


def item(source="news", title="Ford cuts guidance", id="i1"):
    return {"id": id, "ticker": "F", "source": source, "title": title,
            "url": "https://sec.example/doc.htm", "published": "2026-10-02"}


def test_routine_8k_costs_no_llm_calls():
    llm = FakeLLM()
    app, _ = make(llm)
    s = run_item(app, item("sec", "8-K filed 2026-10-02 [items 5.07 shareholder vote, 9.01 exhibits]"))
    assert s["done"] and s.get("event") is None and llm.calls == []


def test_filing_is_prefetched_and_assessed_in_one_call():
    llm = FakeLLM(analyst=[tool_call("Assessment", ASSESS)])
    app, deps = make(llm)
    s = run_item(app, item("sec", "10-Q filed 2026-07-29"))
    assert llm.calls == ["analyst"]              # no title-only triage for filings
    assert s["event"]["checked_filing"] is True  # set from what ran, not from the model
    assert 0 < s["filing_coverage"] < 1
    assert "prefetch_filing" in s["tools_used"]
    assert deps.guard.calls_made == 1


def test_failed_filing_fetch_is_recorded_honestly():
    llm = FakeLLM(analyst=[tool_call("Assessment", ASSESS)])
    app, _ = make(llm, filing_text="")
    s = run_item(app, item("sec", "10-Q filed 2026-07-29"))
    assert s["event"]["checked_filing"] is False


def test_irrelevant_news_is_skipped_after_triage():
    llm = FakeLLM(triage=Triage(relevant=False, reason="stock chatter"))
    app, _ = make(llm)
    s = run_item(app, item())
    assert llm.calls == ["Triage"] and s.get("event") is None


def test_news_agent_uses_tools_then_assesses():
    llm = FakeLLM(triage=Triage(relevant=True, reason="guidance"),
                  analyst=[tool_call("SearchRecentNews", {"limit": 5}), tool_call("Assessment", ASSESS, "2")])
    app, _ = make(llm)
    s = run_item(app, item())
    assert llm.calls == ["Triage", "analyst", "analyst"]
    assert s["tools_used"] == ["SearchRecentNews"]
    assert s["event"]["checked_filing"] is False


def test_plain_text_answer_falls_back_to_structured_final():
    llm = FakeLLM(triage=Triage(relevant=True, reason="r"),
                  analyst=[AIMessage(content="I think it is minor.")], final=Assessment(**ASSESS))
    app, _ = make(llm)
    s = run_item(app, item())
    assert llm.calls[-1] == "Assessment" and s["event"]["severity"] == 1


def test_tool_rounds_are_capped():
    loops = [tool_call("GetIssuerHistory", {"limit": 5}, str(n)) for n in range(10)]
    llm = FakeLLM(triage=Triage(relevant=True, reason="r"), analyst=loops, final=Assessment(**ASSESS))
    app, _ = make(llm)
    s = run_item(app, item())
    assert s["tool_rounds"] == 3 and llm.calls[-1] == "Assessment"


def test_resume_after_budget_does_not_repeat_finished_calls():
    saver = InMemorySaver()
    llm = FakeLLM(triage=Triage(relevant=True, reason="r"), analyst=[tool_call("Assessment", ASSESS)])
    app, _ = make(llm, max_calls=1, checkpointer=saver)  # budget covers triage only
    with pytest.raises(BudgetReached):
        run_item(app, item())
    app2, _ = make(llm, max_calls=5, checkpointer=saver)  # "tomorrow": fresh budget
    s = run_item(app2, item())
    assert llm.calls == ["Triage", "analyst"]  # triage was not repeated
    assert s["event"]["summary"] == ASSESS["summary"]
