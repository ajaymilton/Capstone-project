"""Part C: support-ticket triage with guardrails, redaction, retries, output validation, token budget and routing."""
import json
import logging
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

pd.set_option("display.width", 160)
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.FileHandler("outputs/triage_retry.log", mode="w"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("triage")

# ----------------------------------------------------------------- redaction (order matters: email, phone, order id)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text):
    counts = {}
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        counts[name] = n
    return text, counts


def has_pii(text):
    return any(pat.search(text) for _, pat, _ in PATTERNS)


# ----------------------------------------------------------------- prompts
JSON_RULE = ('Reply with JSON only, no other text, in exactly this form: {"category": "...", "reason": "..."}')
ZERO_SHOT = ("You are a support desk assistant. CLASSIFY the customer ticket into exactly one category: "
             "billing, delivery, technical or fraud. " + JSON_RULE)
FEW_SHOT = ("You are a support desk assistant. CLASSIFY the customer ticket into exactly one category: "
            "billing, delivery, technical or fraud. " + JSON_RULE + "\n"
            "EXAMPLES:\n"
            'Ticket: I was charged twice for [ORDER_ID], please check the bill. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The rider never reached me and [ORDER_ID] is very late. -> {"category": "delivery", "reason": "late order"}\n'
            'Ticket: The app freezes on the payment page and shows an error. -> {"category": "technical", "reason": "app problem"}\n'
            'Ticket: Someone used my wallet without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}')


# ----------------------------------------------------------------- resilient call
def call_with_retry(messages, ticket_id, rng, waits):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * rng.uniform(0.9, 1.15)
            waits.append((attempt, wait))
            log.info(f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s reason={e}")
            time.sleep(wait)


# ----------------------------------------------------------------- output validation
def validate_reply(content):
    """Return (status, category). Never guess: anything unclear is flagged for a human."""
    try:
        data = json.loads(content)
        cat = data.get("category") if isinstance(data, dict) else None
        if cat in ALLOWED:
            return "classified_json", cat
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        found = [w for w in set(re.findall(r"[a-z]+", (content or "").lower())) if w in ALLOWED]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None


# ----------------------------------------------------------------- the pipeline
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()                                   # fresh failure counters for every full run
    rng = random.Random(42)                            # seeded jitter so runs are reproducible
    waits, sent_prompts, rows, total_tokens = [], [], [], 0
    for t in tickets.itertuples():
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": None, "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():                                         # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:                    # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert messages[1]["content"] == clean and not has_pii(messages[1]["content"])
            sent_prompts.append(messages[1]["content"])
            resp = call_with_retry(messages, t.ticket_id, rng, waits)
            used = resp.usage.total_tokens
            total_tokens += used
            row["tokens_used"] = used
            row["status"], cat = validate_reply(resp.choices[0].message.content)
            row["category"] = cat or ""
        rows.append(row)
    df = pd.DataFrame(rows)
    # every retry wait must sit inside the jitter band for its retry number
    for n, w in waits:
        assert BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15
    print(f"\n[{label}] statuses: {df.status.value_counts().to_dict()} | retries logged: {len(waits)} "
          f"| total tokens: {total_tokens} | prompts sent: {len(sent_prompts)}")
    return df, waits, sent_prompts


def accuracy(df):
    ok = df[df.status.isin(["classified_json", "classified_keyword"])]
    return ok, int((ok.category == ok.true_category).sum())


def route(df, segments):
    seg = segments.set_index("customer_id")
    df = df.copy()
    df["segment_name"] = df.customer_id.map(seg["segment_name"]).fillna("")
    premium = df.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = df.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    df["route"] = "support_queue"; df["priority"] = "NORMAL"
    df.loc[~human & premium, ["route", "priority"]] = ["premium_desk", "HIGH"]
    df.loc[~human & (df.category == "fraud"), ["route", "priority"]] = ["security_team", "HIGH"]   # fraud wins over premium
    df.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    return df


if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)   # empty text stays ""
    segments = pd.read_csv("outputs/customer_segments.csv")
    print("tickets:", len(tickets))

    # ---- guardrail counts and redaction counts over all 164 tickets
    empty = int((tickets.text.str.strip() == "").sum())
    longs = int(tickets.text.map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum())
    print("empty tickets:", empty, "| long tickets:", longs,
          "| est_tokens of the long ones:", sorted(set(tickets.text.map(est_tokens)[tickets.text.map(est_tokens) > 400])))
    tot = {"email": 0, "phone": 0, "order_id": 0}
    redacted_all = []
    for txt in tickets.text:
        r, c = redact(txt)
        redacted_all.append(r)
        for k in tot:
            tot[k] += c[k]
    print("substitutions:", tot, "| total:", sum(tot.values()))
    left = sum(len(pat.findall(r)) for r in redacted_all for _, pat, _ in PATTERNS)
    print("matches left when the patterns are re-applied to the redacted texts:", left)
    assert left == 0

    # ---- prompt sanity
    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
    assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT
    print("system prompt size (est tokens): zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

    # ---- zero-shot vs few-shot, no budget
    zs, zs_waits, zs_prompts = run_pipeline(tickets, ZERO_SHOT, None, "zero-shot")
    fs, fs_waits, fs_prompts = run_pipeline(tickets, FEW_SHOT, None, "few-shot")
    assert not any(has_pii(p) for p in zs_prompts + fs_prompts)
    print("PII check: none of the", len(zs_prompts) + len(fs_prompts), "user messages sent contains an email, phone or order id")
    print("retry waits (zero-shot run): min %.4f max %.4f, retry numbers used: %s"
          % (min(w for _, w in zs_waits), max(w for _, w in zs_waits), sorted({n for n, _ in zs_waits})))
    for name, df in [("zero-shot", zs), ("few-shot", fs)]:
        ok, correct = accuracy(df)
        tot_tokens = int(df.tokens_used.sum())
        print(f"{name}: classified {len(ok)} | correct {correct} | accuracy {100*correct/len(ok):.1f}% "
              f"| total tokens {tot_tokens} | tokens per classified ticket {tot_tokens/len(ok):.1f}")
    ok, _ = accuracy(zs)
    print("\nzero-shot confusion (rows = true, columns = predicted):")
    print(pd.crosstab(ok.true_category, ok.category).to_string())

    # ---- budget demo (zero-shot) and final output (few-shot)
    BUDGET_DEMO = 6000
    bd, _, _ = run_pipeline(tickets, ZERO_SHOT, BUDGET_DEMO, f"zero-shot, TOKEN_BUDGET={BUDGET_DEMO}")
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    route(bd, segments)[cols].to_csv("outputs/triage_budget_demo.csv", index=False)
    print("budget demo: tokens spent", int(bd.tokens_used.sum()), "| budget_exhausted tickets:", int((bd.status == "budget_exhausted").sum()))

    final, _, _ = run_pipeline(tickets, FEW_SHOT, 100000, "few-shot, TOKEN_BUDGET=100000 (final)")
    out = route(final, segments)
    out[cols].to_csv("outputs/triage_results.csv", index=False)
    print("final rows:", len(out))
    print(out.groupby(["route", "priority"]).size().to_string())
    print("customers in tickets with no segment row:", int((out.segment_name == "").sum()), "tickets")
    print("tickets for premium customers:", int(out.customer_id.isin(segments.loc[segments.is_premium, 'customer_id']).sum()))
