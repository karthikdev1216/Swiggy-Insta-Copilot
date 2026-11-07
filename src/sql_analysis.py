
"""Run all SQL analysis queries for the Instamart project."""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"; CLEAN=ROOT/"data"/"clean"; DB=ROOT/"data"/"instamart.db"

def run_sql_analysis():

    results = {}

    with sqlite3.connect(DB) as con:

        # 1. Display available database tables
        print("\n--- SQL TABLES ---")

        tables = pd.read_sql_query("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """, con)

        print(tables.to_string(index=False))

        # 2. Revenue by city and product category
        print("\n--- QUERY 2: REVENUE BY CITY AND CATEGORY ---")

        results["revenue"] = pd.read_sql_query("""
            SELECT
                c.city,
                p.category,
                ROUND(SUM(o.order_amount), 2) AS revenue
            FROM orders o
            JOIN customers c USING(customer_id)
            JOIN products p USING(product_id)
            WHERE o.status = 'completed'
            GROUP BY c.city, p.category
            ORDER BY c.city, p.category
        """, con)

        print(results["revenue"].to_string(index=False))
        print(
            "SQL grand total INR:",
            round(results["revenue"]["revenue"].sum(), 2)
        )

        # 3. Order summary
        print("\n--- QUERY 3: ORDER SUMMARY ---")

        results["order_summary"] = pd.read_sql_query("""
            SELECT
                COUNT(*) AS total_orders,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
                    AS completed_orders,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END)
                    AS cancelled_orders,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END)
                    AS pending_orders,
                SUM(CASE WHEN order_amount IS NULL THEN 1 ELSE 0 END)
                    AS missing_order_amounts,
                SUM(CASE WHEN delivery_minutes IS NULL
                    AND status = 'completed' THEN 1 ELSE 0 END)
                    AS missing_delivery_times_completed_orders,
                SUM(CASE WHEN delivery_minutes IS NULL
                    AND status = 'cancelled' THEN 1 ELSE 0 END)
                    AS missing_delivery_times_cancelled_orders
            FROM orders
        """, con)

        print(results["order_summary"].to_string(index=False))

        # 4. High-value customers
        print("\n--- QUERY 4: HIGH-VALUE CUSTOMERS ---")

        results["high_value_customers"] = pd.read_sql_query("""
            SELECT COUNT(*) AS high_value_customers
            FROM (
                SELECT c.customer_id
                FROM customers c
                JOIN orders o
                    ON c.customer_id = o.customer_id
                WHERE o.order_amount > 500
                    AND o.status = 'completed'
                GROUP BY c.customer_id
                HAVING SUM(o.order_amount) > 8000
            ) AS high_value_customers
        """, con)

        print(results["high_value_customers"].to_string(index=False))

        # 5. Products priced above average
        print("\n--- QUERY 5: PRODUCTS ABOVE AVERAGE PRICE ---")

        results["products_above_average"] = pd.read_sql_query("""
            SELECT
                COUNT(product_id) AS products_above_average_price
            FROM products p
            WHERE p.price_inr > (
                SELECT AVG(price_inr)
                FROM products
            )
        """, con)

        print(results["products_above_average"].to_string(index=False))

        # 6. Customers without completed orders
        print("\n--- QUERY 6: CUSTOMERS WITHOUT COMPLETED ORDERS ---")

        results["customers_without_completed_orders"] = pd.read_sql_query("""
            SELECT
                COUNT(*) AS customers_without_completed_orders
            FROM customers c
            LEFT JOIN orders o
                ON c.customer_id = o.customer_id
                AND o.status = 'completed'
            WHERE o.customer_id IS NULL
        """, con)

        print(
            results["customers_without_completed_orders"].to_string(index=False)
        )

        # 7. Customers spending above average
        print("\n--- QUERY 7: CUSTOMERS ABOVE AVERAGE SPENDING ---")

        results["above_average_spending"] = pd.read_sql_query("""
            WITH customer_total_spending AS (
                SELECT
                    c.customer_id,
                    SUM(o.order_amount) AS total_spent
                FROM customers c
                JOIN orders o
                    ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.customer_id
            ),
            customer_average_spending AS (
                SELECT AVG(total_spent) AS average_spent
                FROM customer_total_spending
            )
            SELECT
                ROUND(a.average_spent, 2) AS average_spending,
                COUNT(*) AS customers_above_average_spending
            FROM customer_total_spending c
            CROSS JOIN customer_average_spending a
            WHERE c.total_spent > a.average_spent
            GROUP BY a.average_spent
        """, con)

        print(results["above_average_spending"].to_string(index=False))

        # 8. Ranking ties using RANK and ROW_NUMBER
        print("\n--- QUERY 8: CUSTOMER RANKING DIFFERENCES ---")

        results["ranking_differences"] = pd.read_sql_query("""
            WITH customer_total_spending AS (
                SELECT
                    c.customer_id,
                    c.city,
                    SUM(o.order_amount) AS total_spend
                FROM customers c
                JOIN orders o
                    ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.customer_id, c.city
            ),
            customer_rankings AS (
                SELECT
                    customer_id,
                    city,
                    total_spend,

                    RANK() OVER (
                        PARTITION BY city
                        ORDER BY total_spend DESC
                    ) AS spending_rank,

                    DENSE_RANK() OVER (
                        PARTITION BY city
                        ORDER BY total_spend DESC
                    ) AS dense_spending_rank,

                    ROW_NUMBER() OVER (
                        PARTITION BY city
                        ORDER BY total_spend DESC
                    ) AS row_number_rank

                FROM customer_total_spending
            )
            SELECT *
            FROM customer_rankings
            WHERE spending_rank <> row_number_rank
            ORDER BY city, spending_rank, customer_id
        """, con)

        print(results["ranking_differences"].to_string(index=False))

        # 9. Top spending customer in each city
        print("\n--- QUERY 9: TOP CUSTOMER IN EACH CITY ---")

        results["top_customer_per_city"] = pd.read_sql_query("""
            WITH customer_total_spending AS (
                SELECT
                    c.customer_id,
                    c.city,
                    SUM(o.order_amount) AS total_spend
                FROM customers c
                JOIN orders o
                    ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.customer_id, c.city
            ),
            customer_rankings AS (
                SELECT
                    customer_id,
                    city,
                    total_spend,
                    RANK() OVER (
                        PARTITION BY city
                        ORDER BY total_spend DESC
                    ) AS spending_rank
                FROM customer_total_spending
            )
            SELECT
                customer_id,
                city,
                total_spend,
                spending_rank
            FROM customer_rankings
            WHERE spending_rank = 1
            ORDER BY city
        """, con)

        print(results["top_customer_per_city"].to_string(index=False))

        # 10. Daily and cumulative revenue
        print("\n--- QUERY 10: DAILY CUMULATIVE REVENUE ---")

        results["cumulative_revenue"] = pd.read_sql_query("""
            WITH daily_revenue AS (
                SELECT
                    SUBSTR(order_ts, 1, 10) AS day,
                    SUM(order_amount) AS revenue
                FROM orders
                WHERE status = 'completed'
                GROUP BY SUBSTR(order_ts, 1, 10)
            )
            SELECT
                day,
                ROUND(revenue, 2) AS revenue,
                ROUND(
                    SUM(revenue) OVER (
                        ORDER BY day
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND CURRENT ROW
                    ), 2
                ) AS cumulative_revenue
            FROM daily_revenue
            ORDER BY day
        """, con)

        print(results["cumulative_revenue"].to_string(index=False))

    return results
