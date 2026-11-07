
"""Exploratory Data Analysis for Instamart."""

import pandas as pd
import numpy as np
from pathlib import Path


def run_eda_analysis(sql_revenue):

    # Locate project folders
    ROOT = Path(__file__).resolve().parent.parent
    CLEAN = ROOT / "data" / "clean"

    print("\n=== EDA: LOADING CLEANED DATA ===")

    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    # 1. Filter completed orders
    completed_orders = orders[
        orders["status"] == "completed"
    ].copy()

    completed_orders["order_amount"] = pd.to_numeric(
        completed_orders["order_amount"],
        errors="coerce"
    )

    # 2. Merge orders with customer and product information
    analysis_df = (
        completed_orders
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="left"
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left"
        )
    )

    print("\n--- EDA 1: MERGED DATA ---")
    print(analysis_df.head().to_string(index=False))

    # 3. Revenue pivot table by city and category
    print("\n--- EDA 2: REVENUE PIVOT TABLE ---")

    pivot_table = pd.pivot_table(
        analysis_df,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Grand Total"
    )

    print(pivot_table.round(2).to_string())

    # 4. Cross-check Pandas revenue against SQL revenue
    pandas_total = analysis_df["order_amount"].sum()

    pivot_total = pivot_table.loc[
        "Grand Total", "Grand Total"
    ]

    sql_total = sql_revenue["revenue"].sum()

    print("\n--- EDA 3: SQL AND PANDAS REVENUE VALIDATION ---")
    print("Pandas total:", round(pandas_total, 2))
    print("Pivot table total:", round(pivot_total, 2))
    print("SQL Q2 total:", round(sql_total, 2))

    assert round(pandas_total, 2) == round(sql_total, 2), \
        "Pandas total does not match SQL total"

    assert round(pivot_total, 2) == round(sql_total, 2), \
        "Pivot table total does not match SQL total"

    print("Revenue validation successful!")

    # 5. Revenue summary by city
    print("\n--- EDA 4: CITY-WISE REVENUE SUMMARY ---")

    city_summary = (
        analysis_df
        .groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            order_count=("order_amount", "count")
        )
        .round(2)
        .sort_index()
    )

    print(city_summary.to_string())

    # 6. Mean, median and skewness
    print("\n--- EDA 5: ORDER AMOUNT STATISTICS ---")

    amount = completed_orders["order_amount"].dropna()

    eda_statistics = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Skewness"],
        "Value": [
            amount.mean(),
            amount.median(),
            amount.skew()
        ]
    })

    print(eda_statistics.round(2).to_string(index=False))

    # 7. IQR and outlier analysis
    print("\n--- EDA 6: IQR AND OUTLIER ANALYSIS ---")

    q1 = amount.quantile(0.25)
    median = amount.quantile(0.50)
    q3 = amount.quantile(0.75)

    iqr = q3 - q1

    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)

    outliers_below = (amount < lower_fence).sum()
    outliers_above = (amount > upper_fence).sum()

    iqr_summary = pd.DataFrame({
        "Metric": [
            "Q1 (25th percentile)",
            "Q2 (Median)",
            "Q3 (75th percentile)",
            "IQR",
            "Lower fence",
            "Upper fence",
            "Outliers below lower fence",
            "Outliers above upper fence"
        ],
        "Value": [
            q1,
            median,
            q3,
            iqr,
            lower_fence,
            upper_fence,
            outliers_below,
            outliers_above
        ]
    })

    print(iqr_summary.round(2).to_string(index=False))

    # 8. Correlation analysis
    print("\n--- EDA 7: CORRELATION ANALYSIS ---")

    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]

    for col in correlation_columns:
        completed_orders[col] = pd.to_numeric(
            completed_orders[col],
            errors="coerce"
        )

    correlation_matrix = completed_orders[
        correlation_columns
    ].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(3).to_string())

    # 9. Find strongest correlation pair
    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    correlation_pairs = upper_triangle.stack()

    if not correlation_pairs.empty:
        strongest_pair = correlation_pairs.abs().idxmax()
        strongest_value = correlation_pairs.loc[strongest_pair]

        print("\nStrongest pair:", strongest_pair)
        print("Correlation:", round(strongest_value, 3))
        print("Absolute correlation:", round(abs(strongest_value), 3))
    else:
        print("No valid correlation pairs found.")

    print("\nEDA analysis completed successfully!")

    # Return outputs for future reuse
    return {
        "analysis_df": analysis_df,
        "pivot_table": pivot_table,
        "city_summary": city_summary,
        "eda_statistics": eda_statistics,
        "iqr_summary": iqr_summary,
        "correlation_matrix": correlation_matrix
    }
