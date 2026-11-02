# Glossary

Terms used in this project, in plain language.

## Credit and finance

| Term | Meaning |
|---|---|
| **Bond** | An IOU. Investors lend a company money; it pays interest and returns the money at a set date. |
| **Issuer** | The company that issued the bonds, i.e. the borrower. |
| **Bondholder** | The lender. Bondholders care about getting paid back, not about growth. |
| **Asset manager** | A firm that invests clients' money (pension funds, insurers) in bonds, stocks and more. |
| **Credit risk** | The risk that the issuer does not pay back. |
| **Creditworthiness** | How likely the issuer is to pay back. |
| **Credit analyst** | Watches issuers and judges their credit risk. |
| **Portfolio manager (PM)** | Decides what to buy and sell, using analysts' views. |
| **Default** | The issuer fails to pay interest or principal. |
| **Credit rating** | A grade from Moody's, S&P or Fitch. A **downgrade** signals higher risk. |
| **Maturity** | The date the bond must be paid back. |
| **Refinancing** | Replacing old debt with new debt, often to push the maturity back. |
| **Liquidity** | Cash and credit lines available right now to pay bills. |
| **Leverage** | Debt relative to earnings. Higher means riskier. |
| **Covenant** | A promise in a loan contract, e.g. "debt stays under 3x earnings". A **breach** lets lenders act. |
| **Guidance** | What the company says it expects to earn. A **guidance cut** is a warning. |
| **Going concern** | Auditors' doubt that the company can survive 12 months. A serious red flag. |
| **Restructuring** | Reorganising debt when a company is in trouble. |
| **Primary source** | The official document itself (a filing), as opposed to a headline about it. |

## SEC filings

| Term | Meaning |
|---|---|
| **SEC / EDGAR** | The US regulator and its public filing database, free to query. |
| **10-K** | Annual report. Long and detailed. |
| **10-Q** | Quarterly report. Includes the liquidity and debt sections. |
| **8-K** | "Something important just happened." Each lists **item numbers**, e.g. 2.03 = new debt obligation, 5.02 = officer change, 5.07 = shareholder vote. |
| **MD&A** | Management's Discussion and Analysis, the narrative part of 10-Q/10-K. |

## AI and engineering

| Term | Meaning |
|---|---|
| **LLM** | Large language model: the AI (Gemini, GPT, Claude). |
| **Agent** | An LLM that decides its next step and uses tools, within limits. |
| **Tool calling** | The LLM asks to run a function; the code runs it and returns the result. |
| **Structured output** | Forcing the LLM to answer in a fixed, validated format. |
| **LangGraph** | Library for building agents as graphs of steps (nodes) and transitions (edges). |
| **State** | The shared record passed between nodes: item, messages, tools used, result. |
| **Conditional edge / routing** | "If X go here, else go there." |
| **Checkpointer** | Saves the state after every node, so a stopped run can resume. |
| **Thread** | One checkpointed run; here, one per item. |
| **Bounded autonomy** | The agent chooses its tools, but within a cap (3 rounds). |
| **Rate limit / quota** | How many calls a provider allows per minute / per day. |
| **Exponential backoff** | Waiting 15s, 30s, 60s ... between retries. |
| **Idempotent** | Running it twice gives the same result; no duplicates. |
| **Prompt injection** | Hidden text trying to give the AI instructions. Defence: treat it as data. |
| **Context window / budget** | How much text the model can read at once; we choose what goes in. |
| **RAG** | Retrieval-augmented generation: search a document store and give the model only relevant chunks. |
| **Embedding / vector store** | Text turned into numbers so similar meaning can be searched (e.g. Chroma). |
| **Precision / recall** | Of what was flagged, how much mattered / of what mattered, how much was flagged. |
| **Dependency injection** | Passing the LLM and data sources in, so tests can swap in fakes. |
| **Mock / fake LLM** | A scripted stand-in for the model, so tests are free and repeatable. |
| **MCP** | Model Context Protocol: a standard way to expose tools to any AI assistant. |
| **Human-in-the-loop** | A person approves important AI decisions. |
