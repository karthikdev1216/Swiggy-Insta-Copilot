
-- =====================================================
-- QUERY 1: DISPLAY AVAILABLE DATABASE TABLES
-- =====================================================

SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;


-- =====================================================
-- QUERY 2: REVENUE BY CITY AND PRODUCT CATEGORY
-- =====================================================

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c USING(customer_id)
JOIN products p USING(product_id)
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- =====================================================
-- QUERY 3: ORDER SUMMARY
-- =====================================================

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
FROM orders;


-- =====================================================
-- QUERY 4: HIGH-VALUE CUSTOMERS
-- =====================================================

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
) AS high_value_customers;


-- =====================================================
-- QUERY 5: PRODUCTS PRICED ABOVE AVERAGE
-- =====================================================

SELECT
    COUNT(product_id) AS products_above_average_price
FROM products p
WHERE p.price_inr > (
    SELECT AVG(price_inr)
    FROM products
);


-- =====================================================
-- QUERY 6: CUSTOMERS WITHOUT COMPLETED ORDERS
-- =====================================================

SELECT
    COUNT(*) AS customers_without_completed_orders
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.customer_id IS NULL;


-- =====================================================
-- QUERY 7: CUSTOMERS SPENDING ABOVE AVERAGE
-- =====================================================

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
GROUP BY a.average_spent;


-- =====================================================
-- QUERY 8: CUSTOMER RANKING DIFFERENCES
-- =====================================================

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
ORDER BY city, spending_rank, customer_id;


-- =====================================================
-- QUERY 9: TOP SPENDING CUSTOMER IN EACH CITY
-- =====================================================

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
ORDER BY city;


-- =====================================================
-- QUERY 10: DAILY AND CUMULATIVE REVENUE
-- =====================================================

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
ORDER BY day;
