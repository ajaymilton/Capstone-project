"""Part A: audit the raw exports, clean them, write data/clean/*.csv and data/instamart.db."""
import os
import sqlite3

import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)


def audit_table(name, df):
    print(f"\n=== {name}: shape {df.shape} ===")
    miss = df.isna().sum()
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": miss,
                            "missing_pct": (miss / len(df) * 100).round(2)})
    print(summary.to_string())


prod = pd.read_csv(f"{RAW}/products.csv")
cust_raw = pd.read_csv(f"{RAW}/customers.csv")
# Orders are read as plain strings with keep_default_na=False: pandas would otherwise turn the planted "N/A"
# amounts into NaN on its own and the audit would under-count them. Blanks are turned into NaN explicitly.
ord_raw = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False).replace("", np.nan)
for c in ["quantity", "discount_pct"]:
    ord_raw[c] = pd.to_numeric(ord_raw[c])

print("##### 1. AUDIT #####")
for n, d in [("customers", cust_raw), ("products", prod), ("orders", ord_raw)]:
    audit_table(n, d)

amt_num = pd.to_numeric(ord_raw["order_amount"], errors="coerce")
non_numeric_amt = int(amt_num.isna().sum() - ord_raw["order_amount"].isna().sum())
deliv_num = pd.to_numeric(ord_raw["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(cust_raw["age"], errors="coerce")
gender_vals = cust_raw["gender"].dropna().unique()
audit = {
    "raw orders rows": len(ord_raw),
    "exact duplicate rows": int(ord_raw.duplicated().sum()),
    "rows with status pending": int((ord_raw["status"] == "pending").sum()),
    "distinct order_id": ord_raw["order_id"].nunique(),
    "non-numeric order_amount": non_numeric_amt,
    "negative delivery_minutes": int((deliv_num < 0).sum()),
    "blank delivery_minutes": int(deliv_num.isna().sum()),
    "blank age": int(age_num.isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": cust_raw["city"].nunique(),
    "distinct non-blank gender spellings": len(gender_vals),
    "blank gender": int(cust_raw["gender"].isna().sum()),
}
print("\n--- audit figures ---")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------- customers
print("\n##### 2. CLEAN CUSTOMERS #####")
cust = cust_raw.copy()
cust["city"] = cust["city"].str.strip().str.title()
print("cities after standardising:", cust["city"].nunique())

gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
cust["gender"] = cust["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")

cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
bad_age = (cust["age"] < 10) | (cust["age"] > 100)
print("ages set to missing (outside 10-100):", int(bad_age.sum()))
cust.loc[bad_age, "age"] = np.nan
n_missing_age = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["age"] = cust["age"].round().astype(int)
print("missing ages filled with city median:", n_missing_age)
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
print(cust["city"].value_counts().to_string())
print(cust["gender"].value_counts().to_string())
print("missing ages:", int(cust["age"].isna().sum()), "| age range:", cust["age"].min(), "-", cust["age"].max())

# ---------------------------------------------------------------- orders
print("\n##### 3. CLEAN ORDERS #####")
o = ord_raw.copy()
print("start rows:", len(o))
o["order_ts"] = pd.to_datetime(o["order_ts"])

before = len(o)
o = o.drop_duplicates()
print("rule 2 - exact duplicates removed:", before - len(o))

before = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
print("rule 3 - older records of the same order_id removed:", before - len(o))

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
miss_amt = o["order_amount"].isna()
price = o["product_id"].map(prod.set_index("product_id")["price_inr"])
calc = (price * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o.loc[miss_amt, "order_amount"] = calc[miss_amt]
print("rule 4 - amounts filled from product price:", int(miss_amt.sum()))

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
print("rule 5a - negative delivery times set to missing:", int(neg.sum()))
o["category"] = o["product_id"].map(prod.set_index("product_id")["category"])
done = o["status"] == "completed"
med = o["delivery_minutes"].where(done).groupby(o["category"]).transform("median")
fill_mask = done & o["delivery_minutes"].isna()
o.loc[fill_mask, "delivery_minutes"] = med[fill_mask]
print("rule 5b - completed orders filled with category median:", int(fill_mask.sum()))
o = o.drop(columns="category").sort_values("order_id").reset_index(drop=True)
o["delivery_minutes"] = o["delivery_minutes"].astype(float)

print("\nrows:", len(o), "| status counts:", o["status"].value_counts().to_dict())
print("missing order_amount:", int(o["order_amount"].isna().sum()))
print("missing delivery (completed):", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
      "| (cancelled):", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
print("completed revenue: INR", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---------------------------------------------------------------- store
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = cust.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    c_out.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("\nwritten: data/clean/*.csv and", DB)
