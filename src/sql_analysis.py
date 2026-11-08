
"""Run SQL analysis queries from an external SQL file."""

import sqlite3
import pandas as pd
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "sql" / "instamart_analysis.sql"
DB = ROOT / "data" / "instamart.db"


def load_sql_queries():

    """Read and split SQL statements from the external SQL file."""

    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"SQL file not found: {SQL_FILE}"
        )

    sql_text = SQL_FILE.read_text(encoding="utf-8")

    statements = []
    current_statement = ""

    for line in sql_text.splitlines():
        current_statement += line + "\n"

        if sqlite3.complete_statement(current_statement):
            statement = current_statement.strip()

            if statement:
                statements.append(statement)

            current_statement = ""

    if current_statement.strip():
        raise ValueError(
            "SQL file contains an incomplete SQL statement."
        )

    return statements


def run_sql_analysis():

    """Execute SQL queries and return results as DataFrames."""

    results = {}

    queries = load_sql_queries()

    if len(queries) != 10:
        raise ValueError(
            f"Expected 10 SQL queries, found {len(queries)}"
        )

    result_names = [
        "tables",
        "revenue",
        "order_summary",
        "high_value_customers",
        "products_above_average",
        "customers_without_completed_orders",
        "above_average_spending",
        "ranking_differences",
        "top_customer_per_city",
        "cumulative_revenue"
    ]

    with sqlite3.connect(DB) as con:

        for index, (query, result_name) in enumerate(
            zip(queries, result_names), start=1
        ):

            print(f"\n--- QUERY {index}: {result_name.upper()} ---")

            df = pd.read_sql_query(query, con)

            results[result_name] = df

            print(df.to_string(index=False))

            # Display overall revenue
            if result_name == "revenue":

                grand_total = round(
                    df["revenue"].sum(), 2
                )

                print(
                    "SQL grand total INR:",
                    grand_total
                )

    return results
