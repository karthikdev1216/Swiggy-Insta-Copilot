
"""Part D: database-backed refund tools, ReAct agent and resumable execution."""

from pathlib import Path
import sys
import json
import sqlite3
import time
import uuid

from datetime import datetime, timezone

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm

from src.policy_rag import PolicyRAG


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"

rag = PolicyRAG()


# --------------------------------------------------
# TASK 5: REFUND TOOLS
# --------------------------------------------------

def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if not row:
        return {"error": "order not found"}

    return dict(zip(
        ["order_id", "status", "order_amount", "delivery_minutes"],
        row
    ))


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    amount = float(order_amount) if float(delivery_minutes) > 45 else 0.0

    return {"refund_inr": round(amount, 2)}


def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {"refund_inr": entry["refund_inr"]}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# JSON-style tool specifications

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order information using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant refund rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund based on order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an approved refund and append it to the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund request for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


# --------------------------------------------------
# CHECKPOINT HELPERS
# --------------------------------------------------

def save_checkpoint(run_id, scenario_id, messages, tokens,
                    step_number, executed, steps, llm_calls):

    STATE.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens,
        "step_number": step_number,
        "tools_executed": executed,
        "steps": steps,
        "llm_calls": llm_calls
    }

    path = STATE / f"{run_id}.json"

    path.write_text(
        json.dumps(checkpoint, indent=2, default=str),
        encoding="utf-8"
    )


def load_checkpoint(run_id):
    path = STATE / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")

    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------
# RESULT AND LOG HELPERS
# --------------------------------------------------

def make_result(status, final_answer, steps, llm_calls,
                tokens_used, run_id):

    return {
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "run_id": run_id
    }


def log_agent_run(scenario_id, result):
    OUT.mkdir(parents=True, exist_ok=True)

    record = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"]
    }

    with (OUT / "agent_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------
# TASK 6 AND 7: REACT AGENT
# --------------------------------------------------

def _continue_agent(
    run_id,
    scenario_id,
    messages,
    tokens,
    steps,
    executed,
    llm_calls,
    start_iteration,
    max_iterations,
    token_limit,
    time_limit_seconds
):

    start = time.monotonic()

    for iteration in range(start_iteration, max_iterations):

        # TIME LIMIT
        if time.monotonic() - start >= time_limit_seconds:
            return make_result(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Please hand this request to a human support agent.",
                steps, llm_calls, tokens, run_id
            )

        # MODEL CALL
        response = mock_llm.chat(messages)
        llm_calls += 1

        tokens += response.usage.total_tokens

        # TOKEN LIMIT: do not execute requested tool
        if tokens > token_limit:
            return make_result(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Token budget exceeded. Please hand this request to a human agent.",
                steps, llm_calls, tokens, run_id
            )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        # FINAL MODEL ANSWER
        if not tool_calls:
            content = msg.content or ""
            status = content.split(":", 1)[0].strip()

            return make_result(
                status,
                content,
                steps,
                llm_calls,
                tokens,
                run_id
            )

        # EXECUTE EACH REQUESTED TOOL
        for tc in tool_calls:

            name = tc.name
            args = tc.arguments

            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOL_MAP:
                raise ValueError(f"Unknown tool: {name}")

            # Append assistant tool-call message
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.id,
                    "name": name,
                    "arguments": args
                }]
            })

            # Execute tool
            observation = TOOL_MAP[name](**args)

            executed.append(name)

            step = {
                "tool": name,
                "arguments": args,
                "observation": observation
            }

            steps.append(step)

            # Append tool observation
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(observation)
            })

            # CHECKPOINT AFTER EVERY OBSERVATION
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens,
                iteration + 1,
                executed,
                steps,
                llm_calls
            )

    # MAXIMUM ITERATIONS
    return make_result(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Maximum iterations reached. Please hand this request to a human support agent.",
        steps,
        llm_calls,
        tokens,
        run_id
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None
):

    run_id = run_id or str(uuid.uuid4())

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund agent. "
                "Verify orders, consult policies and follow refund rules. "
                "Only completed orders are eligible. "
                "Refunds above INR 1000 require human escalation."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    result = _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        tokens=0,
        steps=[],
        executed=[],
        llm_calls=0,
        start_iteration=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )

    # Simulated crash happens after the second tool checkpoint.
    # The exception is raised inside _continue_agent's caller below
    # in the dedicated demonstration wrapper.

    if result["status"] != "FALLBACK_TIMEOUT" and crash_after is None:
        log_agent_run(scenario_id, result)

    return result


# --------------------------------------------------
# TASK 8: RESUME AGENT
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    cp = load_checkpoint(run_id)

    result = _continue_agent(
        run_id=run_id,
        scenario_id=cp["scenario_id"],
        messages=cp["messages"],
        tokens=cp["tokens_used"],
        steps=cp["steps"],
        executed=cp["tools_executed"],
        llm_calls=cp["llm_calls"],
        start_iteration=cp["step_number"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )

    if result["status"] != "FALLBACK_TIMEOUT":
        log_agent_run(cp["scenario_id"], result)

    return result


# --------------------------------------------------
# CRASH DEMONSTRATION
# --------------------------------------------------

def demonstrate_resume(user_message, scenario_id="S1"):

    run_id = str(uuid.uuid4())

    # First execution: deliberately crash after two tool executions.
    try:
        messages = [
            {
                "role": "system",
                "content": "AGENT: Follow the refund policy and use tools."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        messages_state = {
            "messages": messages,
            "tokens": 0,
            "steps": [],
            "executed": [],
            "llm_calls": 0
        }

        start = time.monotonic()

        for iteration in range(6):

            response = mock_llm.chat(messages_state["messages"])
            messages_state["llm_calls"] += 1
            messages_state["tokens"] += response.usage.total_tokens

            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            tc = msg.tool_calls[0]
            args = tc.arguments

            messages_state["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": args
                }]
            })

            observation = TOOL_MAP[tc.name](**args)
            messages_state["executed"].append(tc.name)

            messages_state["steps"].append({
                "tool": tc.name,
                "arguments": args,
                "observation": observation
            })

            messages_state["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(observation)
            })

            save_checkpoint(
                run_id,
                scenario_id,
                messages_state["messages"],
                messages_state["tokens"],
                iteration + 1,
                messages_state["executed"],
                messages_state["steps"],
                messages_state["llm_calls"]
            )

            if len(messages_state["executed"]) == 2:
                raise RuntimeError("Simulated crash after second tool")

    except RuntimeError as exc:
        if "Simulated crash" not in str(exc):
            raise

        print("Simulated crash:", exc)

    # Resume from the saved checkpoint
    result = resume_agent(run_id)

    print("Run ID:", run_id)
    print("Resumed result:", result)
    print("Tools executed:", [s["tool"] for s in result["steps"]])

    return result


# --------------------------------------------------
# RUN ALL SCENARIOS
# --------------------------------------------------

def run_all_scenarios():

    path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    df = pd.read_csv(path, keep_default_na=False)

    results = []

    for _, row in df.iterrows():

        result = run_agent(
            user_message=row["message"],
            scenario_id=row["scenario_id"]
        )

        results.append(result)

        print(
            row["scenario_id"],
            result["status"],
            result["final_answer"],
            result["llm_calls"],
            result["tokens_used"]
        )

    return results
