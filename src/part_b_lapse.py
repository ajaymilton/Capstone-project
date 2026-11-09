"""Part B tasks 4-8: lapse label, leak-free features, tuned models, class-balance experiment, persistence, retention list."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 160)
CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id")

# ------------------------------------------------------------------ task 4: label and features
before = o[o.order_ts < CUTOFF]                      # the ONLY order rows the features may look at
done_b = before[before.status == "completed"]
g = done_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().index[0]),
})
final_b = before[before.status.isin(["completed", "cancelled"])]
feat["cancel_rate"] = (final_b.assign(x=final_b.status == "cancelled").groupby("customer_id")["x"].mean()
                       .reindex(feat.index).fillna(0))
feat = feat.reset_index().merge(c, on="customer_id", how="left")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days

sept = o[(o.status == "completed") & (o.order_ts >= CUTOFF) & (o.order_ts < END)]["customer_id"].unique()
feat["lapsed"] = (~feat["customer_id"].isin(sept)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
print("cohort:", len(feat), "| lapsed:", int(y.sum()), "(%.2f%%)" % (100 * y.mean()))
assert feat[NUM + CAT].isna().sum().sum() == 0
print("missing values in feature columns: 0 (asserted)")
assert before["order_ts"].max() < CUTOFF
print("latest order timestamp used for features:", before["order_ts"].max())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train", len(X_tr), "test", len(X_te), "| train lapsed", int(y_tr.sum()), "| test lapsed", int(y_te.sum()))


def report(name, y_true, pred):
    cm = confusion_matrix(y_true, pred)
    r = {"model": name, "accuracy": accuracy_score(y_true, pred),
         "precision": precision_score(y_true, pred, zero_division=0),
         "recall": recall_score(y_true, pred, zero_division=0), "f1": f1_score(y_true, pred, zero_division=0)}
    print(f"{name:34s} acc {r['accuracy']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f}"
          f"   confusion [[TN {cm[0,0]}, FP {cm[0,1]}], [FN {cm[1,0]}, TP {cm[1,1]}]]")
    return r


# ------------------------------------------------------------------ task 5: preprocessing inside the pipeline
def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


# demonstration of the scaler being fitted on training rows only
demo = make_pre().fit(X_tr)
scaler = demo.named_transformers_["num"]
print("\nscaler mean of frequency (train only): %.4f | full-cohort mean of frequency: %.4f"
      % (scaler.mean_[NUM.index("frequency")], feat["frequency"].mean()))

# ------------------------------------------------------------------ task 6: baseline, tuning, evaluation
print("\n--- baseline and tuned models on the untouched test set ---")
base_pred = np.zeros(len(y_te), dtype=int)
report("baseline: always not-lapsed", y_te, base_pred)

lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
print("\nLR best params:", lr_gs.best_params_, "| best CV F1 %.4f" % lr_gs.best_score_)

rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_tr, y_tr)
print("RF best params:", rf_gs.best_params_, "| best CV F1 %.4f" % rf_gs.best_score_)

print()
lr_best, rf_best = lr_gs.best_estimator_, rf_gs.best_estimator_
res_lr = report("logistic regression (tuned)", y_te, lr_best.predict(X_te))
res_rf = report("random forest (tuned)", y_te, rf_best.predict(X_te))

# the model is chosen on cross-validated F1 from the training data, not on the test scores
selected_name = "random forest" if rf_gs.best_score_ >= lr_gs.best_score_ else "logistic regression"
selected = rf_best if selected_name == "random forest" else lr_best
print("\nselected (higher CV F1):", selected_name)

# ------------------------------------------------------------------ task 7: over-sampling the training set only
lapsed_tr = X_tr[y_tr == 1]
n_extra = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = lapsed_tr.sample(n=n_extra, replace=True, random_state=42)
X_bal = pd.concat([X_tr, extra]); y_bal = pd.concat([y_tr, pd.Series(1, index=extra.index)])
print("\nbalanced training set:", dict(y_bal.value_counts().sort_index()), "(original %s)" % dict(y_tr.value_counts().sort_index()))
best_C = lr_gs.best_params_["clf__C"]
lr_bal = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, C=best_C))]).fit(X_bal, y_bal)
print("--- class-balance experiment (logistic regression, same C, same test set) ---")
res_unbal = report("LR on original training set", y_te, lr_best.predict(X_te))
res_bal = report("LR on over-sampled training set", y_te, lr_bal.predict(X_te))

# ------------------------------------------------------------------ task 8: interpretation, persistence, retention list
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\ntop 5 random-forest importances:")
print(imp.head(5).round(4).to_string())
lr_names = lr_best.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_names)
print("\nlogistic regression coefficients (standardised numeric features):")
print(coef.loc[["num__recency", "num__frequency", "num__cancel_rate", "num__monetary", "num__avg_delivery", "num__tenure_days"]].round(3).to_string())

joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(selected.predict(X_te), reloaded.predict(X_te)) and \
    np.allclose(selected.predict_proba(X_te), reloaded.predict_proba(X_te))
print("\nreloaded model reproduces the test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg.is_premium, "customer_id"])
te = feat.loc[X_te.index, ["customer_id"]].copy()
te["lapse_probability"] = selected.predict_proba(X_te)[:, 1].round(4)
te["actually_lapsed"] = y_te.values
ret = te[te.customer_id.isin(premium_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
prem_te = te[te.customer_id.isin(premium_ids)]
print("\npremium customers in test set:", len(prem_te), "| of them actually lapsed:", int(prem_te.actually_lapsed.sum()),
      "| highest predicted lapse probability among them: %.4f" % prem_te.lapse_probability.max())
print(ret.to_string(index=False))
print("of the %d on the retention list, %d truly lapsed" % (len(ret), ret.actually_lapsed.sum()))
