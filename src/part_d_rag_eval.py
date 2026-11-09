"""Part D tasks 1-4: chunk counts, retrieval check, and the 14-question evaluation with overlap 10 and overlap 0."""
import collections
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import PolicyIndex, answer, chunk_text  # noqa: E402

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)
ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)


def evaluate(index, label):
    rows = []
    for q in ev.itertuples():
        ans, sources = answer(q.question, index)
        rows.append({"question": q.question, "expected_doc": q.expected_doc, "expected_fact": q.expected_fact,
                     "sources": sources, "answer": ans})
    df = pd.DataFrame(rows)
    a = df[df.expected_doc != "NONE"]
    doc_hit = a.apply(lambda r: r.expected_doc in r.sources, axis=1)
    fact_hit = a.apply(lambda r: r.expected_fact in r.answer, axis=1)
    u = df[df.expected_doc == "NONE"]
    fallback = u.answer.str.startswith("I could not find")
    print(f"\n[{label}] chunks: {len(index.chunks)} -> {dict(collections.Counter(c['source'] for c in index.chunks))}")
    print(f"  answerable: expected_doc in top-3 {int(doc_hit.sum())}/{len(a)} | answer contains expected_fact {int(fact_hit.sum())}/{len(a)}")
    print(f"  unanswerable: fallback answer {int(fallback.sum())}/{len(u)}")
    bad = a[~(doc_hit & fact_hit)]
    for r in bad.itertuples():
        print(f"  FAIL: {r.question}\n        expected doc {r.expected_doc} in {r.sources}? fact '{r.expected_fact}' in answer? -> answer: {r.answer}")
    return df


idx10, idx0 = PolicyIndex(overlap=10), PolicyIndex(overlap=0)
evaluate(idx10, "chunk 40 words, overlap 10")
evaluate(idx0, "chunk 40 words, overlap 0")

print("\nretrieve('refund policy late delivery') top 3 with overlap 10:")
for h in idx10.retrieve("refund policy late delivery"):
    print(f"  {h['source']:22s} score {h['score']:.4f}  {h['text'][:70]}...")

# where do the overlap-0 chunk boundaries fall inside the refund policy?
text = open("data/policies/refund_policy.txt").read()
print("\nrefund_policy.txt words:", len(text.split()))
for name, ov in [("overlap 10", 10), ("overlap 0", 0)]:
    print(name, "->", [len(c.split()) for c in chunk_text(text, 40, ov)], "words per chunk")
