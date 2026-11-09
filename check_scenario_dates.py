import sqlite3
import os

print("SCRIPT STARTED")

db_path = "data/instamart.db"
print("Current folder:", os.getcwd())
print("Database exists:", os.path.exists(db_path))

connection = sqlite3.connect(db_path)

order_ids = [
    "ORD-100048",
    "ORD-100067",
    "ORD-100145",
    "ORD-107373",
    "ORD-100065"
]

for order_id in order_ids:
    print("\nChecking:", order_id)

    row = connection.execute("""
        SELECT order_id, order_ts, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
    """, (order_id,)).fetchone()

    print("Result:", row)

connection.close()

print("\nSCRIPT COMPLETED")