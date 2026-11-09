"""Part A: run the seven SQL queries against data/instamart.db and do the pandas cross-check / EDA."""
import re
import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
queries = {m.group(1): m.group(2).strip()
           for m in re.finditer(r"-- (Q\d+):[^\n]*\n(.*?)(?=\n-- Q\d+:|\Z)", text, re.S)}
res = {k: pd.read_sql_query(q, con) for k, q in queries.items()}

print("Q1 customers:", len(res["Q1"]))
q2 = res["Q2"]
print("Q2 rows:", len(q2), "| sum:", round(q2["revenue"].sum(), 2))
print("Q2 largest cell:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(res["Q3"]))
print("Q4 customers without a completed order:", len(res["Q4"]))
print("Q5 customers above average:", len(res["Q5"]), "| average spend:", res["Q5"]["avg_total"].iloc[0])
q6 = res["Q6"]
print("Q6 top customer per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print(q6[q6.rnk != q6.row_num].to_string(index=False))
q7 = res["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1])

# ------------------------------------------------------------ pandas cross-check
o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
done = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(
    p[["product_id", "category"]], on="product_id")
pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\npivot table (margins=True):")
print(pt.round(2).to_string())
print("pivot grand total:", round(pt.loc["All", "All"], 2), "| SQL total:", round(q2["revenue"].sum(), 2),
      "| equal:", abs(pt.loc["All", "All"] - q2["revenue"].sum()) < 0.01)

by_city = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                   n_orders=("order_amount", "count")).round(2)
print("\nper-city aggregation:")
print(by_city.to_string())

# ------------------------------------------------------------ EDA of completed order_amount
a = done["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\norder_amount: mean %.2f median %.2f skew %.2f" % (a.mean(), a.median(), a.skew()))
print("Q1 %.2f Q3 %.2f IQR %.2f | fences %.2f / %.2f" % (q1, q3, iqr, lo, hi))
print("outliers below:", int((a < lo).sum()), "above:", int((a > hi).sum()))
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
print("largest |r| pair:", pairs.abs().idxmax(), round(pairs[pairs.abs().idxmax()], 3))
