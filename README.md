# Customer Intelligence & Support Copilot

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or external network call is needed.

## Overview

This project demonstrates an end-to-end customer intelligence and support copilot workflow using synthetic Instamart-style data.

The project covers four major components:

- **Part A:** Data engineering, data quality, SQLite database loading, SQL analysis, and exploratory data analysis.
- **Part B:** RFM analysis, customer segmentation, PCA, and customer lapse prediction.
- **Part C:** Support ticket classification, privacy protection, token-budget guardrails, and ticket routing.
- **Part D:** Policy retrieval using TF-IDF and a bounded refund-support agent with crash recovery and resume functionality.

## Learning Outcomes

Students learn to:

1. Fetch and read CSV exports and audit data quality.
2. Clean and persist trusted tables in SQLite.
3. Write SQL queries and perform pandas analytics.
4. Segment customers and train a leakage-aware lapse prediction model.
5. Redact PII and triage support tickets with retries and output validation.
6. Retrieve policy passages and execute bounded database-backed tools.

---

## Reproduce

Python 3.11+ is recommended.

The project uses `generate_data.py` to create the synthetic input datasets and policy documents. The complete Parts A–D workflow is then executed through `run_project.py`.

### 1. Clone the repository

```bash
git clone https://github.com/karthikdev1216/Swiggy-Insta-Copilot.git
cd Swiggy-Insta-Copilot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate the input datasets

```bash
python generate_data.py
```

This generates deterministic synthetic CSV datasets and policy documents required by the project.

### 5. Execute the complete project (Parts A–D)

From the repository root, run:

```bash
python run_project.py
```

This is the main orchestration script and executes the complete project workflow.

The execution sequence is:

**Part A – Data Engineering and SQL Analysis**

1. Audits the raw datasets.
2. Cleans and transforms the datasets.
3. Validates the cleaned data.
4. Loads the data into SQLite.
5. Executes the seven SQL analysis patterns.
6. Performs exploratory data analysis.

**Part B – Customer Intelligence**

1. Performs RFM analysis.
2. Applies K-means customer segmentation.
3. Performs PCA and reports explained variance.
4. Builds and evaluates the customer lapse prediction model.
5. Generates the customer retention list.

**Part C – Support Ticket Triage**

1. Runs zero-shot and few-shot ticket classification.
2. Evaluates classification accuracy and token consumption.
3. Demonstrates token-budget guardrails.
4. Generates the final triage results.

**Part D – Policy RAG and Refund Agent**

1. Builds the policy retrieval system using TF-IDF.
2. Evaluates retrieval and grounding with overlapping and zero-overlap chunks.
3. Executes the refund-agent scenarios.
4. Demonstrates token-limit and timeout handling.
5. Demonstrates crash recovery and resume functionality.
6. Generates the Part D evaluation report inside `README.md`.

Generated datasets, analytical outputs, agent execution logs, refund records, and the SQLite database are written to their respective project directories.

### 6. Run Part D independently (optional)

To execute only the Part D Policy RAG and Refund Agent workflow:

```bash
python run_project.py --part-d-only
```

This skips Parts A–C and executes Part D using the existing project datasets and dependencies. Ensure the required input datasets and database are available before running this mode.

### 7. Run additional validation checks

```bash
python check_db.py
python check_scenario_dates.py
python test_part_d.py
```

These scripts provide additional checks for database integrity, scenario dates, and Part D functionality.

### 8. Inspect generated results

Review the following locations:

- `data/raw/` – generated synthetic input datasets.
- `data/clean/` – cleaned datasets.
- `data/instamart.db` – SQLite database.
- `outputs/` – analytical results, customer segments, model artifacts, triage results, agent runs, and refund ledger.
- `state/` – agent checkpoint and resume examples.

**README auto-report:** Part D automatically refreshes its generated evaluation section in this README between the markers `PART D AUTO-REPORT START` and `PART D AUTO-REPORT END`. Manually written sections outside these markers are retained.

---

## Repository Map

### Main execution

- `generate_data.py`: Generates deterministic fictional CSV datasets and policy documents.
- `run_project.py`: Main orchestration script for Parts A–D. Supports the optional `--part-d-only` execution mode.
- `mock_llm.py`: Provides an offline stand-in for classification, policy answers, and agent tool calls.

### Source code

- `src/data_pipeline.py`: Data auditing, cleaning, SQLite loading, and validation.
- `src/sql_analysis.py`: SQL analysis requirements and execution.
- `src/eda_analysis.py`: Exploratory data analysis.
- `src/customer_models.py`: RFM analysis, K-means clustering, PCA, and lapse prediction.
- `src/triage.py`: Ticket classification, token guardrails, PII redaction, retries, response validation, and routing.
- `src/policy_rag.py`: Policy document chunking, TF-IDF retrieval, and grounded policy evaluation.
- `src/refund_agent.py`: Order lookup, refund calculations, refund ledger management, and bounded agent tool execution.

### Data and outputs

- `data/raw/`: Raw fictional customer, product, order, ticket, policy evaluation, and agent scenario datasets.
- `data/clean/`: Cleaned datasets and RFM results.
- `data/policies/`: Policy documents used by the retrieval system.
- `data/instamart.db`: SQLite database containing the project tables.
- `outputs/`: Generated analytical artifacts, model outputs, triage results, agent execution logs, and refund records.
- `state/`: Agent checkpoint examples used for crash recovery and resume demonstrations.
- `notebooks/`: Customer intelligence and support copilot notebook.

### SQL

- `sql/queries.sql`: SQL queries required by the assignment.
- `sql/instamart_analysis.sql`: SQL learning patterns and analytical queries.

### Validation and configuration

- `check_db.py`: Database validation.
- `check_scenario_dates.py`: Agent scenario date validation.
- `test_part_d.py`: Part D tests.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes environment files and other generated or temporary files.

---

## How Students Should Learn the Data-Fetching Flow

1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys, and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

---

## Part A — Data Foundation

The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing, and imputes only completed-order delivery durations by product category.

Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

**SQL evaluation observations**

- **Q6:** Ties cause `RANK()` to assign the same rank to multiple customers and skip subsequent rank numbers, whereas `ROW_NUMBER()` assigns a unique sequential number to every customer.
- **Q7:** The strongest absolute correlation should be between `order_amount` and `discount_pct`, approximately −0.095. This is a very weak negative correlation and does not establish causation.

---

## Part B — Segments and Lapse

RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching.

Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes.

The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV, and a saved joblib model.

The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

---

## Part C — Ticket Triage and Privacy

The ticket triage pipeline uses a privacy-first approach to classify customer support tickets into billing, delivery, technical, and fraud categories.

### Privacy and validation

- Empty tickets and tickets exceeding 400 estimated input tokens are blocked from model calls.
- Email addresses, phone numbers, and order IDs are redacted before sending ticket text to the model.
- Assertions verify that no sensitive information remains in messages sent to the model.
- Transient model errors are retried using exponential backoff with jitter.
- Model responses are validated as JSON or, where applicable, classified using keyword extraction.

### Prompt comparison and cost

- Zero-shot and few-shot prompts are evaluated on the same ticket dataset.
- The few-shot prompt includes labelled examples to improve classification accuracy.
- Actual total token usage and tokens per classified ticket are reported for both approaches.
- Additional few-shot prompt tokens represent the cost of improved classification performance.

### Routing

- Validated fraud tickets are routed to the security team with HIGH priority.
- Premium customers are routed to the premium desk with HIGH priority, except fraud tickets.
- Other valid classifications are routed to the support queue.
- Empty tickets, long tickets, invalid model responses, and budget-exhausted tickets are routed to human review.

### Outputs

- `outputs/triage_budget_demo.csv`: Demonstrates token-budget enforcement.
- `outputs/triage_results.csv`: Final few-shot classification and routing results for all 164 tickets.

The JSON and keyword classifications are automatically accepted only after the specified validation checks. Tickets routed to human review are not treated as successfully classified.

---

## Part D — Policy Retrieval and Refund Tools

Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity.

The offline mock returns context-derived sentences.

Refund rules are implemented in Python tools backed by SQLite:

- Completed orders delivered in over 45 minutes qualify for a full refund.
- Orders over INR 1000 escalate instead of auto-issuing.

This is a teaching simulation, not a payment integration.

---

## Data Leakage Controls

- All imputers, encoders, and scalers are fitted only on training data through the model pipeline.
- No order data from 2025-09-01 onwards is included in the lapse model features.
- Segment labels generated in Task 3 are not used as model features.
- Oversampling is performed only on the training dataset.
- The original test dataset remains unchanged.

---

## Evaluation Observations

- Few-shot prompting improved classification accuracy from **85.71% to 100%**.
- Token consumption increased from **9,712 to 17,512 tokens**.
- Average token consumption increased from **66.07 to 119.13 tokens per classified ticket**.
- Few-shot prompting provided better classification accuracy at the cost of approximately **80.3% additional tokens per ticket**.

---

<!-- PART D AUTO-REPORT START -->

## Part D — Policy RAG Assistant and Refund Agent

### Retrieval and grounding evaluation

| Chunking | Expected document (answerable) | Expected fact (answerable) | Fallback (unanswerable) |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, zero overlap | 12/12 | 10/12 | 2/2 |

**TF-IDF explanation:** Term Frequency (TF) measures how often a term occurs in a chunk; Inverse Document Frequency (IDF) downweights terms that occur in many chunks and gives more weight to terms that are relatively rare. A limitation is that TF-IDF is lexical: two phrases with the same meaning but different vocabulary may receive a low similarity score.

**Zero-overlap boundary check:**

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

Tools executed across crash and resume:

`get_order → search_policy → calculate_refund → issue_refund`

Resumed final status: **REFUND**.

### ReAct versus plan-and-execute

The measured ReAct runs use multiple model calls when the workflow needs several tool observations: S1 uses 5 calls, S5 uses 3, and S6 uses 2 (the exact token totals are shown in the scenario table).

A plan-and-execute design could ask for a complete tool sequence in one initial call and replan only after an error, which may reduce calls and prompt overhead on predictable cases. However, its initial plan can become stale when an observation is unexpected.

ReAct can inspect each result before choosing the next action. For example, it can stop immediately when `get_order` returns NOT_FOUND or when the order is cancelled/not completed, instead of blindly executing later refund steps.

The plan-and-execute token cost was not benchmarked in this project, so a numeric token saving cannot be claimed.

I would choose ReAct for this refund workflow because eligibility, delivery time, refund amount, and escalation depend on database and policy observations, and step-by-step decisions reduce the risk of issuing an invalid refund.

All simulated requests are treated as having been raised within the policy's 7-day window, as specified in the assignment. The supplied order database does not contain a separate refund-request timestamp with which to independently validate that window.

<!-- PART D AUTO-REPORT END -->

---

## GitHub Suggestions

These public repositories and documentation projects are useful for learning concepts. Do not copy another student's capstone solution. Adapt the patterns and cite the source in your notes.

- [pandas documentation](https://github.com/pandas-dev/pandas) — CSV loading, cleaning, groupby, and pivot tables.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) — Pipelines, preprocessing, clustering, model selection, metrics, and TF-IDF.
- [Python tutorial](https://github.com/python/cpython/tree/main/Doc/tutorial) — Python fundamentals.
- [Jupyter Notebook](https://github.com/jupyter/notebook) — Notebook workflow.
- [SQLite](https://github.com/sqlite/sqlite) — SQL database engine (official mirror).
- [scikit-learn text feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — Text feature extraction and TF-IDF.

---

## Limitations and Next Steps

- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric. Compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is a deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks, and human approval workflows.
- The Premium flag is a post-hoc, whole-window label derived from RFM segmentation. Since the RFM calculation includes the September period used to define the lapse outcome, the resulting Premium retention list is illustrative and is not a leakage-free evaluation of retention performance.