# Explaining the graph in 2 minutes

Practise saying this out loud with `graph.py` open. Use your own words; the points matter, not the script.

**1. What it is (15 s)**
"It's a credit monitoring agent. For each new SEC filing or headline about an issuer, it decides
whether it matters for bondholders, reads the evidence, and returns a structured assessment:
event type, direction, severity 1 to 5, and its reasoning."

**2. Why a graph (20 s)**
"I built it as an explicit LangGraph state graph rather than a prebuilt agent loop, so I control
the routing, I can see every step, and every step is checkpointed."

**3. Walk the routes (45 s)**
- "`screen` routes by source without an LLM. Routine 8-Ks, like shareholder vote results, cost nothing."
- "News goes through `triage`, one cheap call. Irrelevant items stop there."
- "Filings skip triage, because a title like '10-Q filed' tells the model nothing. Instead
  `prefetch_filing` pulls the liquidity, debt and risk sections, not the first page. The first
  version only saw about 8% of a 10-Q, mostly the table of contents."
- "The `analyst` node is the agentic part. It chooses: read another section, check recent news,
  check the issuer's history, or submit its assessment. Tool rounds are capped at three."
- "If it answers in plain text or runs out of rounds, `force_final` asks for the structured answer."

**4. Reliability (30 s)**
- "We run the tools ourselves, so 'filing read' is a fact the system records, not something the model claims."
- "It runs on a free key with about 20 calls a day. Every call goes through a guard that spaces
  calls out, retries timeouts with backoff, and stops the run cleanly at a budget."
- "Each item is a LangGraph thread with a SQLite checkpointer. If a run stops, the next one
  resumes at the node that failed, so no finished call is repeated, and no item is ever lost."

**5. Close (10 s)**
"The LLM and data sources are injected, so the tests run with a fake LLM and no network.
Next I'm adding retrieval over filings and an evaluation set with measured precision and recall."

## Questions to be ready for
- *Why not just use `create_agent`?* Explicit routing, recorded facts, checkpoint-level resume, cheaper.
- *What's the checkpointer for?* Resuming a stopped item without repeating LLM calls. Thread id = item id.
- *How do you know it's any good?* Today: one reviewed example. Next: ~100 hand labels and precision/recall.
- *What happens with prompt injection in a filing?* Text is wrapped as untrusted data and the prompt says
  never to follow it; tools are read-only, so the worst case is a wrong score, not an action.
- *Biggest weakness?* Headline-only news, heading-based section search, severity not yet calibrated.
