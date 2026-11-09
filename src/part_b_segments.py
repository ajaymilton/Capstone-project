"""Part B tasks 1-3: RFM table, K-means selection, segment profile/naming, PCA, outputs/customer_segments.csv."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 160)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o.status == "completed"]

# ---- task 1: RFM
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
print("customers with a completed order:", len(rfm))
print("recency  median %d min %d max %d" % (rfm.recency.median(), rfm.recency.min(), rfm.recency.max()))
print("frequency mean %.2f max %d" % (rfm.frequency.mean(), rfm.frequency.max()))
print("monetary  mean %.2f max %.2f" % (rfm.monetary.mean(), rfm.monetary.max()))

# ---- task 2: K-means for K = 2..8
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
ktab["inertia_drop_vs_prev"] = (-ktab["inertia"].diff()).round(1)
print("\n", ktab.to_string(index=False))

# ---- task 3: profile, name, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\nsegment profile (K=%d):" % FINAL_K)
print(prof.to_string())

# Names are assigned from the profile itself (not from the arbitrary cluster number), so they are stable.
names = {}
premium_id = prof["monetary"].idxmax()
names[premium_id] = "Premium Big-Basket"
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
names[loyal_id] = "Loyal Regulars"
rest = rest.drop(index=loyal_id)
dormant_id = rest["recency"].idxmax()
names[dormant_id] = "Fading / Dormant"
names[rest.drop(index=dormant_id).index[0]] = "Occasional Buyers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("\nnames:", {int(k): v for k, v in names.items()}, "| premium segment id:", int(premium_id))

pca = PCA(n_components=2).fit(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_.round(3), "total %.3f" % pca.explained_variance_ratio_.sum())

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
print("premium customers:", int(rfm.is_premium.sum()))
