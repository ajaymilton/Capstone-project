"""Part A validation: assert the reference checkpoints against the clean CSVs and the SQLite database."""
import sqlite3

import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
con = sqlite3.connect("data/instamart.db")
db_orders = pd.read_sql_query("SELECT * FROM orders", con)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
done = o[o.status == "completed"]

checks = [
    ("orders has 12,309 rows", len(o) == 12309),
    ("order_id is unique", o["order_id"].is_unique),
    ("11,110 completed orders", (o.status == "completed").sum() == 11110),
    ("1,199 cancelled orders", (o.status == "cancelled").sum() == 1199),
    ("no pending rows", (o.status == "pending").sum() == 0),
    ("no missing order_amount", o["order_amount"].isna().sum() == 0),
    ("no missing delivery_minutes for completed", done["delivery_minutes"].isna().sum() == 0),
    ("1,199 missing delivery_minutes for cancelled", o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum() == 1199),
    ("completed revenue = 2,998,668.80", round(done["order_amount"].sum(), 2) == 2998668.80),
    ("customers: 1,200 rows, six cities", len(c) == 1200 and c["city"].nunique() == 6),
    ("city counts match", c["city"].value_counts().to_dict() ==
     {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts match", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("ages: none missing, all within 18-60", c["age"].notna().all() and c["age"].between(18, 60).all()),
    ("database has customers, products, orders", {"customers", "products", "orders"} <= tables),
    ("database orders match the clean CSV", len(db_orders) == len(o) and
     round(db_orders.loc[db_orders.status == "completed", "order_amount"].sum(), 2) == 2998668.80),
]
failed = [name for name, ok in checks if not ok]
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
