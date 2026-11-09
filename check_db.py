import sqlite3

db_path = "data/instamart.db"

connection = sqlite3.connect(db_path)
connection.row_factory = sqlite3.Row

print("ORDER STATUS SUMMARY")
print("-" * 50)

statuses = connection.execute("""
    SELECT status, COUNT(*) AS total
    FROM orders
    GROUP BY status
""").fetchall()

for row in statuses:
    print(f"{row['status']}: {row['total']}")

print("\nCOMPLETED ORDERS WITH DELIVERY OVER 45 MINUTES")
print("-" * 50)

orders = connection.execute("""
    SELECT order_id, status, order_amount, delivery_minutes
    FROM orders
    WHERE LOWER(status) = 'completed'
      AND delivery_minutes > 45
    ORDER BY order_amount
""").fetchall()

for row in orders[:20]:
    print(
        f"ID: {row['order_id']} | "
        f"Amount: {row['order_amount']} | "
        f"Delivery: {row['delivery_minutes']} minutes"
    )

print("\nTARGET AMOUNT MATCHES")
print("-" * 50)

targets = [128.25, 135.00, 1999.00]

for target in targets:
    matches = connection.execute("""
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE ABS(order_amount - ?) < 0.01
    """, (target,)).fetchall()

    print(f"\nTarget: INR {target}")
    for row in matches:
        print(dict(row))

connection.close()