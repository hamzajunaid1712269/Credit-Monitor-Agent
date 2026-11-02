# Credit News & Filings Monitor

An agentic AI tool that watches a list of corporate bond issuers, screens SEC filings and news,
and flags events that could change credit risk, with a severity score and a written rationale.

## Why
Credit analysts cover many issuers and most incoming news is noise. This tool does the first pass:
filter, read the primary source when needed, compare against the issuer's history, and surface
only what a portfolio manager should look at.

## Architecture

```
EDGAR filings ─┐
               ├─> dedupe ─> triage (1 LLM call, structured) ─┬─> irrelevant: stored, skipped
Google News ───┘                                              └─> analyst agent
                                                                    tools: read_filing
                                                                           search_recent_news
                                                                           get_issuer_history
                                                                    output: CreditEvent (Pydantic)
                                                                         └─> SQLite ─> Streamlit
```

- **Two-stage design:** a cheap triage call filters noise so the more expensive agent only runs on
  items that matter.
- **Agent decides its own evidence:** it reads the filing for SEC items, corroborates vague
  headlines, and checks history to avoid re-flagging old stories.
- **Structured outputs everywhere:** Pydantic schemas (`schemas.py`) make results queryable and
  testable rather than free text.
- **Evaluation:** `eval.py` scores triage precision/recall against hand labels.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env           # add your API key and SEC contact
python pipeline.py --dry-run   # fetch only, no LLM cost
python pipeline.py --tickers F CCL
streamlit run app.py
```

## Limitations and next steps
- Google News RSS headlines only, no full article text.
- Severity is the model's judgement; should be calibrated against analyst labels.
- Next: rating agency press releases, European issuers (Bundesanzeiger), bond price moves as a
  second signal, scheduled runs with email/Slack alerts, and a human feedback loop on flags.
