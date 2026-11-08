
"""Part C: Privacy-first ticket triage, retries, validation and routing."""

from pathlib import Path
import json
import re
import random
import time

import pandas as pd
import mock_llm


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

CATEGORIES = {"billing", "delivery", "technical", "fraud"}


# ---------------------------------------------------------
# 2. PRIVACY REDACTION PATTERNS
# Exact order: email, phone, order ID
# ---------------------------------------------------------

PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")
]


# ---------------------------------------------------------
# 3. ZERO-SHOT AND FEW-SHOT PROMPTS
# ---------------------------------------------------------

ZERO = """
CLASSIFY the customer's issue into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}
"""

FEW = """
CLASSIFY the customer's issue into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}

EXAMPLES:
Text: Unauthorized transaction on my account.
-> {"category": "fraud", "reason": "unauthorized transaction"}

Text: My order arrived late.
-> {"category": "delivery", "reason": "late delivery"}

Text: The app crashes whenever I log in.
-> {"category": "technical", "reason": "app crash"}
"""


# ---------------------------------------------------------
# 4. REDACT PERSONAL INFORMATION
# ---------------------------------------------------------

def redact(text):
    """Redact email, phone and order ID in the required order."""

    counts = {}

    for name, pattern, replacement in PATTERNS:
        text, count = pattern.subn(replacement, text)
        counts[name] = count

    return text, counts


def verify_redaction(text):
    """Count any sensitive patterns remaining after redaction."""

    remaining = {}

    for name, pattern, _ in PATTERNS:
        remaining[name] = len(pattern.findall(text))

    return remaining


# ---------------------------------------------------------
# 5. RETRY FUNCTION
# Maximum 3 retries after the first attempt
# ---------------------------------------------------------

def retry_chat(messages, ticket_id):

    for retry_no in range(4):

        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:

            if retry_no >= 3:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_no)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"retry ticket={ticket_id} "
                f"retry={retry_no} wait={wait:.4f}s"
            )

            time.sleep(wait)


# ---------------------------------------------------------
# 6. OUTPUT VALIDATION
# ---------------------------------------------------------

def validate_response(content):
    """Validate JSON first, then use keyword fallback if JSON is invalid."""

    try:
        obj = json.loads(content)

        category = (
            obj.get("category")
            if isinstance(obj, dict)
            else None
        )

        if category in CATEGORIES:
            return category, "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):

        words = re.findall(r"[a-z]+", content.lower())

        found = set(words).intersection(CATEGORIES)

        if len(found) == 1:
            return found.pop(), "classified_keyword"

        return "", "flagged_invalid_output"


# ---------------------------------------------------------
# 7. PREMIUM FLAG CONVERSION
# ---------------------------------------------------------

def to_bool(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


# ---------------------------------------------------------
# 8. MAIN TRIAGE PIPELINE
# ---------------------------------------------------------

def run_pipeline(
    prompt=FEW,
    token_budget=100000,
    output_file=None
):

    # Read original tickets in their original order.
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        keep_default_na=False
    )

    # Check required input columns.
    required_columns = {
        "ticket_id",
        "customer_id",
        "text",
        "true_category"
    }

    missing_columns = required_columns.difference(tickets.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in tickets.csv: {missing_columns}"
        )

    # Load customer segments created in Part B.
    seg_path = OUT / "customer_segments.csv"

    if not seg_path.exists():
        raise FileNotFoundError(
            "outputs/customer_segments.csv not found. "
            "Run Part B before Part C."
        )

    segments = pd.read_csv(
        seg_path,
        keep_default_na=False
    )

    segments["customer_id"] = segments["customer_id"].astype(str)

    segmap = segments.set_index(
        "customer_id"
    ).to_dict("index")

    # Initialise tracking variables.
    redaction_totals = {
        "email": 0,
        "phone": 0,
        "order_id": 0
    }

    remaining_totals = {
        "email": 0,
        "phone": 0,
        "order_id": 0
    }

    redacted_texts = []
    rows = []

    model_call_ticket_ids = []
    blocked_ticket_ids = []

    total_tokens = 0

    # Reset mock immediately before each full run.
    mock_llm.reset()
    random.seed(42)

    # -----------------------------------------------------
    # PROCESS EACH TICKET IN ORIGINAL FILE ORDER
    # -----------------------------------------------------

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = str(ticket["customer_id"])
        original_text = str(ticket["text"])

        category = ""
        status = ""
        tokens_used = 0

        # Redact all tickets for overall redaction reporting.
        safe_text, counts = redact(original_text)

        redacted_texts.append(safe_text)

        for name, count in counts.items():
            redaction_totals[name] += count

        # -------------------------------------------------
        # INPUT GUARDRAILS
        # -------------------------------------------------

        if not original_text.strip():

            status = "skipped_empty"
            blocked_ticket_ids.append(ticket_id)

        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            status = "flagged_long"
            blocked_ticket_ids.append(ticket_id)

        elif total_tokens >= token_budget:

            status = "budget_exhausted"

        else:

            # -------------------------------------------------
            # PREPARE MODEL MESSAGES
            # -------------------------------------------------

            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": safe_text
                }
            ]

            # The user message must contain only redacted ticket text.
            assert messages[1]["content"] == safe_text

            # Verify that no sensitive information is in any prompt.
            for message in messages:

                for _, pattern, _ in PATTERNS:

                    assert pattern.search(message["content"]) is None, (
                        f"Sensitive information found in prompt "
                        f"for ticket {ticket_id}"
                    )

            # Record one model submission per ticket.
            model_call_ticket_ids.append(ticket_id)

            # Call the model. Retry function handles transient errors.
            response = retry_chat(messages, ticket_id)

            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

            content = response.choices[0].message.content or ""

            # Validate model response.
            category, status = validate_response(content)

        # -------------------------------------------------
        # CUSTOMER SEGMENT LOOKUP
        # -------------------------------------------------

        segment = segmap.get(customer_id, {})

        segment_name = segment.get("segment_name", "")

        is_premium = to_bool(
            segment.get("is_premium", False)
        )

        # -------------------------------------------------
        # ROUTING RULES
        # -------------------------------------------------

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        }:

            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":

            route = "security_team"
            priority = "HIGH"

        elif is_premium:

            route = "premium_desk"
            priority = "HIGH"

        else:

            route = "support_queue"
            priority = "NORMAL"

        # -------------------------------------------------
        # APPEND EXACTLY ONE ROW FOR EVERY TICKET
        # IMPORTANT: This must remain inside the ticket loop.
        # -------------------------------------------------

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    # -----------------------------------------------------
    # 9. PRIVACY ASSERTIONS AFTER PROCESSING ALL TICKETS
    # -----------------------------------------------------

    # Empty and long tickets must never be sent to the model.
    assert not set(blocked_ticket_ids).intersection(
        model_call_ticket_ids
    )

    print("\nPrivacy assertion passed.")

    print(
        "Tickets sent to model:",
        len(model_call_ticket_ids)
    )

    print(
        "Empty/long tickets blocked:",
        len(blocked_ticket_ids)
    )

    # Verify that no sensitive patterns remain after redaction.
    for text in redacted_texts:

        remaining = verify_redaction(text)

        for name, count in remaining.items():
            remaining_totals[name] += count

    assert sum(remaining_totals.values()) == 0, (
        "Sensitive information remains after redaction."
    )

    print("Sensitive-data redaction verification passed.")
    print("Remaining sensitive matches:", remaining_totals)

    # -----------------------------------------------------
    # 10. CREATE RESULT DATAFRAME
    # -----------------------------------------------------

    result = pd.DataFrame(rows)

    # Every input ticket must have exactly one output row.
    assert len(result) == len(tickets), (
        f"Expected {len(tickets)} output rows, "
        f"but found {len(result)}."
    )

    # Confirm that original ticket order is preserved.
    assert result["ticket_id"].tolist() == tickets[
        "ticket_id"
    ].tolist(), "Ticket order has changed."

    print("\n--- TRIAGE DEBUGGING ---")
    print("Input tickets:", len(tickets))
    print("Output rows:", len(result))
    print("Model submissions:", len(model_call_ticket_ids))
    print("Blocked empty/long tickets:", len(blocked_ticket_ids))

    # -----------------------------------------------------
    # 11. EVALUATION AGAINST TRUE CATEGORY
    # -----------------------------------------------------

    evaluation = result.copy()

    # Both DataFrames now have the same number of rows.
    evaluation["true_category"] = tickets[
        "true_category"
    ].values

    classified = evaluation[
        evaluation["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    classified_count = len(classified)

    accuracy = (
        (
            classified["category"] == classified["true_category"]
        ).mean()
        if classified_count
        else 0
    )

    # Confusion matrix for all four allowed categories.
    confusion = pd.crosstab(
        classified["true_category"],
        classified["category"]
    ).reindex(
        index=sorted(CATEGORIES),
        columns=sorted(CATEGORIES),
        fill_value=0
    )

    # -----------------------------------------------------
    # 12. SAVE OUTPUT CSV
    # -----------------------------------------------------

    OUT.mkdir(parents=True, exist_ok=True)

    if output_file:
        result.to_csv(
            OUT / output_file,
            index=False
        )

        print(f"\nOutput saved: {OUT / output_file}")

    # -----------------------------------------------------
    # 13. SUMMARY
    # -----------------------------------------------------

    summary = {
        "total_tickets": len(tickets),

        "classified_tickets": classified_count,

        "accuracy": accuracy,

        "total_tokens": total_tokens,

        "tokens_per_classified_ticket": (
            total_tokens / classified_count
            if classified_count
            else 0
        ),

        "redactions": redaction_totals,

        "redactions_total": sum(
            redaction_totals.values()
        ),

        "remaining_sensitive_matches": remaining_totals,

        "status_counts": result[
            "status"
        ].value_counts().to_dict(),

        "confusion_matrix": confusion
    }

    return result, summary
