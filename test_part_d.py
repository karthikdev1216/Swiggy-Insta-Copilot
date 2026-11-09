
from src.refund_agent import (
    run_agent,
    demonstrate_resume,
    run_all_scenarios
)

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent


# Test 1: All seven scenarios
results = run_all_scenarios()


# Test 2: Token limit
df = pd.read_csv(
    ROOT / "data/raw/agent_scenarios.csv",
    keep_default_na=False
)

s1 = df[df["scenario_id"] == "S1"].iloc[0]

token_result = run_agent(
    s1["message"],
    "S1_TOKEN_TEST",
    token_limit=200
)

print("\nTOKEN TEST:", token_result["status"])


# Test 3: Timeout
timeout_result = run_agent(
    s1["message"],
    "S1_TIMEOUT_TEST",
    time_limit_seconds=0
)

print("\nTIMEOUT TEST:", timeout_result["status"])


# Test 4: Crash and resume
resume_result = demonstrate_resume(
    s1["message"],
    "S1_RESUME_TEST"
)

print("\nRESUME TEST:", resume_result["status"])
