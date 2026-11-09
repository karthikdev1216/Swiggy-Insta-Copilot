# Customer Intelligence & Support Copilot 

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

##  outcomes
 (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/sql_analysis.py` : SQL requirements 
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

Q6 Ties cause RANK() to assign the same rank to multiple customers and skip subsequent rank numbers, whereas ROW_NUMBER() assigns a unique sequential number to every customer.

Q7 The strongest absolute correlation should be between order_amount and discount_pct, approximately −0.095. This is a very weak negative correlation and does not establish causation.

## Part B — Segments and lapse
RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes. The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV and a saved joblib model. The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

## Part C — Ticket triage and privacy

The ticket triage pipeline uses a privacy-first approach to classify customer support tickets into billing, delivery, technical, and fraud categories.

**Privacy and validation**
- Empty tickets and tickets exceeding 400 estimated input tokens are blocked from model calls.
- Email addresses, phone numbers, and order IDs are redacted before sending ticket text to the model.
- Assertions verify that no sensitive information remains in messages sent to the model.
- Transient model errors are retried using exponential backoff with jitter.
- Model responses are validated as JSON or, where applicable, classified using keyword extraction.

**Prompt comparison and cost**
- Zero-shot and few-shot prompts are evaluated on the same ticket dataset.
- The few-shot prompt includes labelled examples to improve classification accuracy.
- Report the actual total token usage and tokens per classified ticket for both approaches. The additional few-shot prompt tokens represent the cost of improved classification performance.

**Routing**
- Validated fraud tickets are routed to the security team with HIGH priority.
- Premium customers are routed to the premium desk with HIGH priority, except fraud tickets.
- Other valid classifications are routed to the support queue.
- Empty tickets, long tickets, invalid model responses, and budget-exhausted tickets are routed to human review.

**Outputs**
- `outputs/triage_budget_demo.csv`: demonstrates token-budget enforcement.
- `outputs/triage_results.csv`: final few-shot classification and routing results for all 164 tickets.

The JSON and keyword classifications are automatically accepted only after the specified validation checks. Tickets routed to human review are not treated as successfully classified.


## Part D — Policy retrieval and refund tools
Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences. Refund rules are implemented in Python tools backed by SQLite: completed orders delivered in over 45 minutes qualify for a full refund; orders over INR 1000 escalate instead of auto-issuing. This is a teaching simulation, not a payment integration.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.
- The Premium flag is a post-hoc, whole-window label derived from RFM segmentation. Since the RFM calculation includes the September period used to define the lapse outcome, the resulting Premium retention list is illustrative and is not a leakage-free evaluation of retention performance.

## Data Leakage Controls

All imputers, encoders and scalers are fitted only on training data through the model pipeline. No test rows are used during preprocessing.

No order data from 2025-09-01 onwards is included in the lapse model features.

Segment labels generated in Task 3 are not used as model features.

Oversampling is performed only on the training dataset. The original test dataset remains unchanged.

Evaluation observations:

Few-shot prompting improved classification accuracy from 85.71% to 100%.

Token consumption increased from 9,712 to 17,512 tokens.

Average token consumption increased from 66.07 to 119.13 tokens per classified ticket.

Few-shot prompting provided better classification accuracy at the cost of approximately 80.3% additional tokens per ticket.

<!-- PART D AUTO-REPORT START -->
## Part D — Policy RAG Assistant and Refund Agent

### Retrieval and grounding evaluation

| Chunking | Expected document (answerable) | Expected fact (answerable) | Fallback (unanswerable) |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, zero overlap | 12/12 | 10/12 | 2/2 |

TF-IDF explanation: Term Frequency (TF) measures how often a term occurs in a chunk; Inverse Document Frequency (IDF) downweights terms that occur in many chunks and gives more weight to terms that are relatively rare. A limitation is that TF-IDF is lexical: two phrases with the same meaning but different vocabulary may receive a low similarity score.

Zero-overlap boundary check: 
The first answerable failure is **What delivery time makes a completed order eligible for a full refund?** (`refund_policy.txt`). It fails the zero-overlap evaluation; inspect its source chunk boundary if a specific split-fact explanation is required.

### Refund-agent scenario results

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1012 |
| S2 | REFUND | 135.00 | 5 | 1012 |
| S3 | ESCALATE | 1999.00 | 5 | 999 |
| S4 | DENIED | 0.00 | 4 | 697 |
| S5 | DENIED | — | 3 | 406 |
| S6 | NOT_FOUND | — | 2 | 118 |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | 3138 |

### Safety-limit demonstrations

- S1 with `token_limit=200`: **FALLBACK_TOKEN_LIMIT**.
- S1 with `time_limit_seconds=0`: **FALLBACK_TIMEOUT**.
- S7 is expected to stop after six model calls with `FALLBACK_MAX_ITERATIONS` when the default `max_iterations=6` and `token_limit=6000` are used.

### Crash and resume

Tools executed across crash and resume: **get_order → search_policy → calculate_refund → issue_refund**.
Resumed final status: **REFUND**.

### ReAct versus plan-and-execute (under 250 words)

The measured ReAct runs use multiple model calls when the workflow needs several tool observations: S1 uses 5 calls, S5 uses 3, and S6 uses 2 (the exact token totals are shown in the scenario table). A plan-and-execute design could ask for a complete tool sequence in one initial call and replan only after an error, which may reduce calls and prompt overhead on predictable cases. However, its initial plan can become stale when an observation is unexpected. ReAct can inspect each result before choosing the next action: for example, it can stop immediately when `get_order` returns NOT_FOUND or when the order is cancelled/not completed, instead of blindly executing later refund steps. The plan-and-execute token cost was not benchmarked in this project, so a numeric token saving cannot be claimed. I would choose ReAct for this refund workflow because eligibility, delivery time, refund amount and escalation depend on database and policy observations, and step-by-step decisions reduce the risk of issuing an invalid refund.

All simulated requests are treated as having been raised within the policy's 7-day window, as specified in the assignment. The supplied order database does not contain a separate refund-request timestamp with which to independently validate that window.

<!-- PART D AUTO-REPORT END -->
