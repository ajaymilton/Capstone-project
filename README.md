# Instamart Customer Intelligence & Support Copilot

## Overview

This repository is my capstone submission: one small pipeline that (A) cleans the messy order exports and stores them in SQLite, (B) segments customers and predicts who will stop ordering, (C) triages support tickets safely, and (D) answers policy questions and decides refund requests with a tool-calling agent.

All data is synthetic (made by the supplied `generate_data.py`) and every language-model call goes through the supplied offline `mock_llm.py`. No network, API key or paid service is used anywhere. All money is in INR. `generate_data.py` and `mock_llm.py` are committed exactly as supplied.

Everything I report below was copied from the output of the scripts in this repository, run in the order of the Reproduce section.

```
generate_data.py, mock_llm.py     supplied files (unmodified)
src/part_a_clean.py               audit + cleaning + CSV/SQLite storage
src/validate_part_a.py            15 checkpoint assertions
src/part_a_sql.py                 runs sql/queries.sql, pandas cross-check, EDA
src/part_b_segments.py            RFM, K-means, PCA, customer_segments.csv
src/part_b_lapse.py               lapse label, features, models, retention list
src/part_c_triage.py              ticket triage pipeline
src/rag.py                        chunking, TF-IDF retrieval, answer()
src/part_d_rag_eval.py            policy question evaluation
src/agent.py                      tools, ReAct loop, limits, checkpoint/resume
src/part_d_agent.py               runs the 7 scenarios and the demos
sql/queries.sql                   the seven queries
data/clean, data/instamart.db     Part A output
outputs/, state/                  Part B, C, D output
```

## Reproduce

Tested on Python 3.12 (3.11+ is required). Windows users activate with `.venv\Scripts\activate` instead of `source`.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py`
6. `python src/validate_part_a.py` (last line: `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql.py`
8. `python src/part_b_segments.py`
9. `python src/part_b_lapse.py`
10. `python src/part_c_triage.py`
11. `python src/part_d_rag_eval.py`
12. `python src/part_d_agent.py`

Run them from the repository root. Part B needs Part A's clean files, Part C needs `outputs/customer_segments.csv` from step 8, and Part D's agent needs `data/instamart.db`. Steps 8 to 12 together take well under a minute. The retry sleeps in step 10 add only a few seconds.

`requirements.txt` pins the exact versions I ran with: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0 and their dependencies.

---

## Part A

### Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Missing values per column (blank cells only):

| table | column | dtype as read | missing | % missing |
|---|---|---|---|---|
| customers | age | float64 | 61 | 5.08 |
| customers | gender | str | 30 | 2.50 |
| customers | customer_id, city, signup_date | str | 0 | 0.00 |
| products | all four columns | str / int64 | 0 | 0.00 |
| orders | delivery_minutes | float64 | 2,541 | 19.42 |
| orders | all other eight columns | str / int64 | 0 | 0.00 |

`orders.order_amount` is text in the raw file because it contains the unparseable entries `N/A` and `error`. I therefore read `orders.csv` with `dtype=str, keep_default_na=False` and turn blanks into missing values myself. If I had used the pandas default, `N/A` would have silently become a missing value and the audit would have under-counted unparseable amounts.

| audit figure | found | reference |
|---|---|---|
| raw order rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status pending | 656 | 656 |
| distinct order_id | 12,309 | 12,309 |
| non-numeric order_amount | 120 | 120 |
| negative delivery_minutes | 106 | 106 |
| blank delivery_minutes | 2,541 | 2,541 |
| blank age | 61 | 61 |
| age outside 10–100 | 6 | 6 |
| distinct raw city spellings | 24 | 24 |
| distinct non-blank gender spellings | 10 | 10 |
| blank gender | 30 | 30 |

### Cleaning rules and how much each one changed

**Customers**

| rule | effect |
|---|---|
| city: strip spaces, title case | 24 spellings reduced to 6 cities |
| gender: m/male/M/MALE… to `male`, f/female… to `female`, blank to `unknown` | 10 spellings reduced to 3 labels; 30 blanks became `unknown` |
| age: values outside 10–100 set to missing | 6 values (280, -4, 0, 150, 999, 200) |
| age: fill missing with the city median (`groupby().transform("median")`) | 67 values filled (61 blank + 6 invalid) |
| signup_date to datetime | all 1,200 converted |

After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.

**Orders** (applied in this order)

| # | rule | effect | rows left |
|---|---|---|---|
| 1 | `order_ts` to datetime | 13,084 converted, 0 failures | 13,084 |
| 2 | drop exact duplicate rows | 119 removed | 12,965 |
| 3 | per `order_id` keep the record with the latest `order_ts` | 656 older records removed (they were exactly the pending rows) | 12,309 |
| 4 | `order_amount` to numeric; fill the unparseable ones with `round(price × quantity × (1 − discount/100), 2)` | 112 amounts filled (120 in the raw file, 8 of them sat on rows removed in steps 2 and 3) | 12,309 |
| 5a | negative `delivery_minutes` set to missing | 105 values (106 in the raw file, one was on a removed duplicate) | 12,309 |
| 5b | completed orders: fill missing `delivery_minutes` with the median of the order's product category among completed orders | 774 filled (105 negatives that just became missing + 669 blanks). Medians: Dairy 25, Electronics 25, Grocery 25, Snacks 25, Personal Care 25.5 | 12,309 |
| 5c | cancelled orders: `delivery_minutes` left missing on purpose | 1,199 stay missing | 12,309 |

Result: 12,309 rows, one per `order_id`; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing `order_amount`; 0 missing `delivery_minutes` among completed orders and 1,199 among cancelled ones; completed revenue INR 2,998,668.80. `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS` in both the CSV and the database.

### Store and validate

`src/part_a_clean.py` writes `data/clean/customers.csv`, `products.csv`, `orders.csv` and the tables `customers`, `products`, `orders` of `data/instamart.db`. `src/validate_part_a.py` asserts 15 checkpoints (row counts, status counts, revenue, city and gender counts, age range, tables present in the database, database matches the CSV) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| query | result | reference |
|---|---|---|
| Q1 WHERE then HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows, sum INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 | same |
| Q3 scalar subquery | 11 products above the average price | 11 |
| Q4 LEFT JOIN | 25 customers with no completed order | 25 |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 | 550 / 2,552.06 |
| Q6 ranking windows | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK ≠ ROW_NUMBER | same |
| Q7 running total | 183 daily rows, last cumulative value INR 2,998,668.80 | same |

Why RANK differs from ROW_NUMBER in Q6: when two customers in a city have exactly the same total spend, RANK gives both the same number, while ROW_NUMBER still counts 1, 2, 3… and gives the second one a higher number, so each tied customer after the first of its group (8 rows here, for example the two Pune customers at INR 285.00) has a different ROW_NUMBER.

### pandas cross-check

`pd.pivot_table(..., margins=True)` of completed revenue by city and category has a grand total of INR 2,998,668.80, identical to the SQL total from Q2.

Per city, completed `order_amount` (groupby with named aggregation):

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics of completed `order_amount`

| statistic | value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| fences (1.5×IQR) | lower −283.50, upper 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

**The four questions (137 words).** *What is the data about?* 11,110 completed orders from six cities, April to September 2025, mostly small grocery, dairy and snack baskets plus some electronics. *Distributions?* Order values are strongly right-skewed: the mean (269.91) is twice the median (135.00) and the skewness is 4.61. *Outliers?* 897 orders lie above the 645.30 fence and none below. 635 of them (71%) are Electronics, which make up only 9.2% of completed orders, so they look like genuine big purchases and I kept them. The real anomalies (duplicates, negative delivery times, text amounts) were handled in cleaning. *Relationships?* All correlations are weak. The largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087); delivery time is almost unrelated to everything. A correlation this small explains under 1% of the variance and does not establish causation.

---

## Part B

### RFM table

Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers have at least one completed order. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean 2,552.06, max 14,766.55. All identical to the references.

### K-means (RFM standardised with `StandardScaler`, `KMeans(n_init=10, random_state=42)`)

| K | inertia | silhouette | inertia drop from K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | |
| 3 | 1268.2 | 0.445 | 552.6 |
| 4 | 743.1 | 0.526 | 525.1 |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Choice: K = 4.** The elbow is at K=4: the drop in inertia is still large going from 3 to 4 (525) and then falls to 126 at K=5. K=4 also has the best silhouette score of all K (0.526), and K=5 is clearly lower (0.480). Four segments are also few enough for the retention team to give each one its own campaign.

Inertia for K=8 is always lower than for K=2 because every additional centroid can only shrink (never grow) the sum of squared distances to the nearest centre, and with one centroid per customer it would reach zero. A lower inertia therefore says nothing by itself about a better segmentation, which is why the elbow, the silhouette score and interpretability have to decide.

### Segment profile, names and actions (K = 4)

| segment_id | name | customers | mean recency (days) | mean frequency | mean monetary (INR) | recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big-Basket** (is_premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics offers, dedicated support; this is where the revenue per customer is. |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Reward habit with a loyalty streak or free delivery; do not over-discount. |
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Basket-building offers to lift order frequency. |
| 3 | Fading / Dormant | 182 | 109.04 | 2.20 | 685.99 | Win-back message with a one-time coupon; cheap to try. |

Cluster numbers are only labels produced by K-means and carry no order (segment 1 is not "better" than segment 0). The names are assigned from the profile in code (highest monetary = Premium, then highest frequency, then highest recency), not from the cluster number.

PCA on the standardised RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has the columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (1,175 rows, 138 Premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff = **1,156 customers**. Lapsed = no completed order from 2025-09-01 to 2025-09-30: **443 lapsed (38.32%)**. Split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train, 232 test** (354 lapsed in train, 89 in test). The latest order timestamp used for any feature is 2025-08-31 22:35:45.

Features (all from orders before the cutoff plus the clean customer table): recency, frequency, monetary, average discount_pct, cancellation rate (cancelled ÷ cancelled + completed), average delivery_minutes of completed orders, favourite category, city, age, gender, tenure in days. Baseline "never lapsed" on the test set: accuracy 61.6%, lapsed-class F1 0.

### Preprocessing without leakage

A single scikit-learn `Pipeline` holds a `ColumnTransformer` (`StandardScaler` for the eight numeric features, `OneHotEncoder(drop="first", handle_unknown="ignore")` for city, gender and favourite category) and the classifier. Inside `GridSearchCV` the pipeline is refitted on each training fold, so even the cross-validation folds never leak. I asserted in code that no feature column contains a missing value.

The mean of `frequency` learned by the scaler (training rows only) is **8.1147**, while the full-cohort mean is **8.0285**; they differ because the scaler only saw the 924 training customers and the cohort mean also contains the 232 test customers. (This `frequency` is counted before the cutoff, so it is not the 9.46 from the whole-window RFM.)

**No encoder, scaler or resampler saw test rows. No data from 2025-09-01 onwards entered the features. The segment labels from the K-means step were not used as features.** (They would have leaked: the "Fading / Dormant" segment, with mean recency 109 days at 1 October, contains customers who all lapsed, a 100% lapse rate.)

### Tuning and test results (lapsed class = positive)

| | best parameters | best CV F1 (5-fold) |
|---|---|---|
| Logistic regression, C ∈ {0.01, 0.1, 1, 10, 100} | C = 100 | 0.6436 |
| Random forest, max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20} (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.6831 |

Logistic regression chose the largest C in the grid (the weakest regularisation); I did not extend the grid, so I cannot say whether a larger C would do better.

Untouched test set (232 customers, 89 lapsed):

| model | accuracy | precision | recall | F1 | confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|
| baseline (never lapsed) | 0.616 | 0.000 | 0.000 | 0.000 | [[143, 0], [89, 0]] |
| logistic regression | 0.707 | 0.615 | 0.629 | 0.622 | [[108, 35], [33, 56]] |
| **random forest (selected)** | 0.733 | 0.648 | 0.663 | **0.656** | [[111, 32], [30, 59]] |

**Why the random forest.** I selected it on cross-validated F1 (0.683 against 0.644), before looking at the test scores, and the test set agrees: it has the higher precision, recall and F1 (0.656, above the required 0.55). For retention, missing a customer who is about to lapse costs the customer's future revenue, while wrongly targeting a loyal customer costs one coupon or message, so recall matters more than precision and the forest catches 59 of 89 lapsers against 56 for logistic regression, while also wasting fewer contacts (32 false positives against 35).

### Class-balance experiment

The training lapsed rows were over-sampled with replacement (`random_state=42`) from 354 to 570, matching the 570 non-lapsed rows; test set unchanged; logistic regression with the same C.

| | precision | recall | F1 | accuracy | [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|
| original training set | 0.615 | 0.629 | 0.622 | 0.707 | [[108, 35], [33, 56]] |
| over-sampled training set | 0.592 | 0.865 | 0.703 | 0.720 | [[90, 53], [12, 77]] |

Recall moved most (0.629 to 0.865); precision did not rise, it slipped slightly (0.615 to 0.592). For the retention team this means the model now catches 77 of the 89 lapsers instead of 56, but the price is that it contacts 130 customers instead of 91, so 53 loyal customers get an unnecessary offer instead of 35. That is worth it if the offer is cheap compared with the lost revenue, which is the usual case for a coupon.

### Interpretation

Five most important features of the best random forest:

| feature | importance | business reading |
|---|---|---|
| frequency | 0.284 | how often someone ordered before September is the strongest signal of staying |
| monetary | 0.275 | total spend carries similar information (it is moderately correlated with frequency, r = 0.41) |
| recency | 0.123 | days since the last order still helps beyond the two above |
| avg_delivery | 0.062 | slower average deliveries matter a little |
| cancel_rate | 0.048 | customers who cancel more are somewhat different, but this is a weak signal |

Logistic regression coefficients (standardised numeric features), sign and reading:

- frequency **−1.627**: more past orders means a lower chance of lapsing; clearly the dominant effect.
- recency **+0.042**: a longer gap since the last order raises the lapse risk, but the size is tiny because frequency and monetary already explain most of it.
- cancel_rate **−0.135**: negative, which is the opposite of what I expected. It is small and I would not read much into it; with this data it is probably noise.

Frequency and monetary are correlated (r = 0.41 before the cutoff), so a single coefficient such as monetary (+0.222) should not be read in isolation.

The selected random forest was saved with `joblib.dump` to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and the reloaded model reproduces the test predictions and probabilities exactly (checked in code).

### Retention list

40 of the 232 test customers are in the Premium segment. Ranking them by predicted lapse probability and keeping the top 10 gives `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). **0 of those 10 truly lapsed.** Across all 40 Premium test customers, 12 lapsed, and the highest predicted probability among them was only 0.4993, so the model considers none of them clearly likely to lapse. This list is illustrative and not a leakage-free evaluation: the Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc label.

---

## Part C

### Guardrails and redaction

4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, each estimated at 734 tokens, limit `MAX_INPUT_TOKENS = 400`) never reach the model. The CSV is read with `keep_default_na=False`, so empty texts stay empty strings.

Redaction with the three prescribed patterns in the prescribed order (email, phone, order id), counted over all 164 tickets:

| type | substitutions |
|---|---|
| email → `[EMAIL]` | 55 |
| phone → `[PHONE]` | 91 |
| order id → `[ORDER_ID]` | 132 |
| **total** | **278** |

Re-applying the three patterns to the redacted texts finds 0 matches (asserted). The user message sent to the model is exactly the redacted text, and `run_pipeline` asserts this before each call; across the 312 messages sent in the zero-shot and few-shot runs, none contains an email, phone number or order id.

### Prompts

Both prompts contain `CLASSIFY`, name the four categories and demand JSON only in the form `{"category": "...", "reason": "..."}`. The zero-shot prompt (45 estimated tokens) has no `EXAMPLES:`. The few-shot prompt (134 estimated tokens) has an `EXAMPLES:` line followed by four labelled examples, one per category, including a fraud one.

### Retries and validation

`call_with_retry` allows up to 3 retries per ticket with a wait of `0.05 × 2**n × random.uniform(0.9, 1.15)` seconds (seeded `random.Random(42)`), and writes one log line per retry (ticket id, retry number, wait) to the console and to `outputs/triage_retry.log`. `mock_llm.reset()` is called immediately before every full run. In a full run over the 156 eligible tickets there were exactly **19 TransientLLMErrors, all recovered**, no ticket ended with a permanent error, every retry was retry number 0 and every wait lay between 0.0453 and 0.0562 seconds.

Validation: valid JSON with an allowed category = `classified_json`; otherwise the lowercase words from `re.findall(r"[a-z]+", ...)` are checked and exactly one allowed category word gives `classified_keyword`; everything else (including valid JSON with a category like `shipping`) = `flagged_invalid_output` and goes to a human, never guessed.

| status | tickets (one full run, same for both prompts) |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
| **total** | **164** |

### Zero-shot against few-shot (no budget)

| run | classified | correct | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 11,428 | 77.7 |
| few-shot | 147 | 147 | 100.0% | 25,312 | 172.2 |

Zero-shot confusion (rows = true, columns = predicted), classified tickets only:

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 94 extra tokens per ticket (172.2 against 77.7, roughly 2.2 times the cost, 25,312 against 11,428 tokens for the full run), and in exchange removes all 21 zero-shot errors. Those errors are not random: 11 fraud tickets and 10 technical tickets are sent to billing. A fraud ticket sent to the billing queue misses the HIGH priority and the 2-hour security window in the escalation policy, so the extra tokens are clearly worth paying here; for a cheap, low-risk category they would not be. One caution: `mock_llm.py` is a keyword matcher whose keyword order changes when the system prompt contains `EXAMPLES:`, so this gain is partly a property of the mock, and with a real model I would measure it again before trusting a 100% figure.

### Budget demo and final output

Zero-shot run with `TOKEN_BUDGET = 6000` (the check happens before every model call, on the running total of `usage.total_tokens`): the run spent 6,021 tokens and **74 tickets ended `budget_exhausted`** (72 classified_json, 6 classified_keyword, 4 flagged_invalid_output, 4 empty, 4 long, 74 exhausted). Saved as `outputs/triage_budget_demo.csv` with the same routing columns. The count depends on my prompt length, so it is not comparable with other prompts.

`outputs/triage_results.csv` comes from a few-shot run with `TOKEN_BUDGET = 100000` (no ticket exhausted) and has 164 rows:

| route | priority | tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

Routing follows the brief: skipped_empty, flagged_long, flagged_invalid_output and budget_exhausted go to `human_review`; otherwise fraud goes to `security_team` (HIGH) first, then premium customers (from `is_premium` in Part B's segment file) go to `premium_desk` (HIGH), everything else to `support_queue`. A customer without a segment row counts as not premium (all 164 tickets' customers had a row in this data). 16 tickets belong to premium customers; 11 of them reached the premium desk, 3 went to the security team (fraud tickets take priority) and 2 went to human review.

**Rule-validated and human-handled.** 147 tickets were automatically classified and passed the output check (135 as valid JSON, 12 through the keyword rule). 17 tickets went to humans: 4 empty, 4 too long and 9 with an invalid model answer.

---

## Part D

### Chunking and retrieval

Each policy file is cut into 40-word chunks with an overlap of 10 words (a new chunk starts every 30 words, and the last chunk is the first one that reaches the end of the document). Result: **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2, each stored with its source file name.

Retrieval: `TfidfVectorizer()` fitted on the chunk texts, the query transformed with the same vectorizer, chunks ranked by cosine similarity, `retrieve(query, k=3)` returns dictionaries with `source`, `score`, `text`. For `refund policy late delivery` all three top chunks are from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk, IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, specific words count most. A limitation is that TF-IDF only matches the same words: "money back" and "refund", or "late" and "delayed", share no tokens, so a paraphrased question can miss the right chunk.

### Answering and evaluation

`answer(question)` retrieves three chunks and sends two messages: a system message with `POLICY ASSISTANT` that says to answer only from the context (or to say the information could not be found), and a user message `CONTEXT:\n[file]\nchunk ...\nQUESTION:\n...`. It returns the model text and the three sources.

| setting | chunks | answerable: expected_doc in top 3 | answerable: answer contains expected_fact | unanswerable: fallback "I could not find…" |
|---|---|---|---|---|
| 40 words, overlap 10 | 15 | 12 / 12 | 12 / 12 | 2 / 2 |
| 40 words, overlap 0 | 13 | 12 / 12 | **10 / 12** | 2 / 2 |

With overlap 0 two questions break:
- *"What delivery time makes a completed order eligible for a full refund?"* The right document is retrieved, but the answer is "receives a full refund of the order amount." The condition "more than 45 minutes" is in the previous chunk, because the sentence is cut by the chunk boundary.
- *"How soon is money returned when a payment fails but is deducted?"* The answer is cut after "returned automatically", and the "within 48 hours" part sits at the start of the next chunk.

In both cases the answer sentence is split across a boundary, which the 10-word overlap prevents.

### Tools

Five tools with names, one-sentence descriptions and parameter specs in `TOOLS_SPEC`, and a `TOOL_MAP` from name to Python function: `get_order` (reads `data/instamart.db`), `search_policy` (my `retrieve`, results under the key `results`), `calculate_refund` (refund 100% of the amount if delivery took more than 45 minutes, otherwise 0), `issue_refund` and `escalate_to_human` (each appends one JSON line with timestamp and tool name to `outputs/refund_ledger.jsonl`). I added `run_id`, `scenario_id`, `order_id` and `refund_inr` to each ledger line. The agent looks up the tool name the model asks for in `TOOL_MAP` and calls it with the arguments dictionary.

### ReAct loop: the seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` appends the assistant tool call and the tool result to the messages and loops until the model gives a final answer. Each run is one line of `outputs/agent_runs.jsonl`. Every status and refund amount matches `agent_scenarios.csv` (asserted in the script).

| scenario | status | refund (INR) | model calls | tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,177 |
| S2 | REFUND | 135.00 | 5 | 1,177 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,164 |
| S4 | DENIED | 0.00 | 4 | 829 |
| S5 | DENIED | 0.00 | 3 | 505 |
| S6 | NOT_FOUND | 0.00 | 2 | 184 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,336 |

S7 uses all six model calls (six `search_policy` calls in a row) and stays under the default `token_limit=6000`.

### Stop safely

| demonstration | status | what happened |
|---|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped after the 2nd model call at 214 tokens, before the tool it asked for (`search_policy`) was run; only `get_order` had been executed |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped at the top of the first iteration: 0 model calls, 0 tools |

All fallbacks return the steps done so far and a short hand-off text in `final_answer` (status, number of calls and tools run, and the customer request). These two demonstration runs are stored in `outputs/agent_demo_runs.jsonl` so that `agent_runs.jsonl` holds exactly the seven scenario runs.

### Crash and resume

After every tool observation the agent saves `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed, and also the steps so far and the limits). For S1 I raised a `SimulatedCrash` right after the second tool's checkpoint was saved; `resume_agent("demo_crash_S1")` then loaded that checkpoint and continued. Final status **REFUND, INR 128.25**.

Tools executed across the crash and the resume, in order (4 executions, none repeated):

1. `get_order`
2. `search_policy`  ← crash here, checkpoint saved
3. `calculate_refund`  ← first tool after the resume
4. `issue_refund`

### ReAct versus plan-and-execute (240 words)

My ReAct agent asks the model before every step, so the call count follows the case: 5 calls for S1 (1,177 tokens), 4 for the on-time order S4 (829), 3 for the cancelled order S5 (505) and 2 for the unknown order S6 (184). Each call re-sends the whole history, so tokens grow faster than calls, and S7 spent 3,336 tokens on six useless calls before the iteration limit caught it. A plan-and-execute design would ask the model once for the full step list and call it again only on error, which would need fewer model calls (I estimate about 2 for S1: one plan, one final answer; I could not measure this because the mock only supports the ReAct style, so this is an estimate and not a result). Its weakness is the unexpected observation: the refund amount depends on the order lookup, so the plan cannot contain the arguments of `calculate_refund` in advance, and for S6 (NOT_FOUND) or S5 (cancelled) the remaining steps are pointless and the executor needs an error rule and a replan call. ReAct simply stops, as in S5 and S6, using 3 and 2 calls. For this refund workflow I would keep ReAct: it has at most five steps, every step depends on the previous result, safety is easier to enforce per step (token, iteration and time limits, checkpoints), and the extra tokens are small. With longer, predictable workflows I would switch to plan-and-execute.

---

## Limitations and Next Steps

- **The retention list is illustrative, not a leakage-free evaluation.** The Premium flag is a post-hoc whole-window label that includes the September window used for the lapse label. In my run, 0 of the top 10 on the list lapsed, so it says nothing reliable about how well the model finds lapsing Premium customers. A better design would compute segments only from data before the cutoff.
- **The language model is a mock.** `mock_llm.py` is a keyword matcher with scripted failures. The Part C accuracies (85.7% and 100%) and the 135 / 12 / 9 counts show that the pipeline handles retries, bad output and budgets correctly; they do not predict how a real model would behave.
- **Synthetic data.** The weak correlations, the clean separation of the segments and the model scores are properties of the generator, not of a real business.
- **Imputed delivery times.** About 774 completed orders got a category median delivery time (almost always 25 minutes). This slightly dilutes the `avg_delivery` feature and the refund eligibility of those orders; the agent reads the cleaned value.
- **Logistic regression C = 100 is the edge of the grid**, and I did not extend the grid. The random forest only had 45 grid cells; I did not tune the decision threshold, although the balance experiment suggests that is where a retention team would gain most.
- **The refund ledger is not idempotent.** The crash-and-resume demonstration re-runs S1 as its own run, so `outputs/refund_ledger.jsonl` has two `issue_refund` lines for ORD-100048 (run ids `main_S1` and `demo_crash_S1`). A production system needs an idempotency key per order so a refund can never be paid twice, and the checkpoint should record the ledger write inside the same transaction as the tool result.
- **Retrieval is lexical.** TF-IDF will miss paraphrases; embeddings or a hybrid search would be the next step, and the chunk size and overlap were fixed by the brief rather than tuned.
- **`requirements.txt`** was written from the versions installed where I ran the project; if a newer Python or library breaks anything, those pins are the first place to look.
