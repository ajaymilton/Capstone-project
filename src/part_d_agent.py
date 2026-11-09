"""Part D: run the seven refund scenarios, the stop-safely demos and the crash-and-resume demo."""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
from agent import SimulatedCrash, resume_agent, run_agent  # noqa: E402

pd.set_option("display.width", 200)
RUNS, DEMOS = "outputs/agent_runs.jsonl", "outputs/agent_demo_runs.jsonl"
for p in [RUNS, DEMOS, agent.LEDGER] + glob.glob("state/*.json"):     # start from a clean slate so reruns are reproducible
    if os.path.exists(p):
        os.remove(p)
os.makedirs("state", exist_ok=True); os.makedirs("outputs", exist_ok=True)


def log(path, res, **extra):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": res["scenario_id"], "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"], **extra}) + "\n")


def refund_of(res):
    if res["status"] in ("REFUND", "ESCALATE"):
        calc = [s for s in res["steps"] if s["tool"] == "calculate_refund"]
        return calc[-1]["observation"]["refund_inr"] if calc else 0.0
    return 0.0


sc = pd.read_csv("data/raw/agent_scenarios.csv")
table = []
for s in sc.itertuples():
    res = run_agent(s.message, s.scenario_id, run_id=f"main_{s.scenario_id}")
    log(RUNS, res)
    exp = "FALLBACK_MAX_ITERATIONS" if s.expected_outcome == "FALLBACK" else s.expected_outcome
    amt = refund_of(res)
    assert res["status"] == exp, (s.scenario_id, res["status"], exp)
    assert abs(amt - s.expected_refund_inr) <= 0.01, (s.scenario_id, amt, s.expected_refund_inr)
    table.append({"scenario": s.scenario_id, "status": res["status"], "refund_inr": amt,
                  "model_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                  "tools": " > ".join(x["tool"] for x in res["steps"])})
print(pd.DataFrame(table).to_string(index=False))

# ---------------- stop-safely demos (all on scenario S1)
s1 = sc[sc.scenario_id == "S1"].iloc[0]
d_tok = run_agent(s1.message, "S1", token_limit=200, run_id="demo_token_limit_S1")
d_time = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="demo_timeout_S1")
log(DEMOS, d_tok, demo="token_limit=200"); log(DEMOS, d_time, demo="time_limit_seconds=0")
print("\nS1 with token_limit=200 ->", d_tok["status"], "| model calls", d_tok["llm_calls"], "| tokens", d_tok["tokens_used"],
      "| tools run:", [x["tool"] for x in d_tok["steps"]])
print("   handoff:", d_tok["final_answer"])
print("S1 with time_limit_seconds=0 ->", d_time["status"], "| model calls", d_time["llm_calls"], "| tools run:", [x["tool"] for x in d_time["steps"]])
print("   handoff:", d_time["final_answer"])

# ---------------- crash and resume (S1)
RID = "demo_crash_S1"
try:
    run_agent(s1.message, "S1", run_id=RID, crash_after_tools=2)
    raise SystemExit("the simulated crash did not happen")
except SimulatedCrash as e:
    print("\ncrash:", e)
ck = json.load(open(f"state/{RID}.json"))
print("checkpoint after the crash: step", ck["step"], "| tools executed:", ck["tools_executed"], "| tokens_used", ck["tokens_used"])
res = resume_agent(RID)
ck2 = json.load(open(f"state/{RID}.json"))
log(DEMOS, res, demo="crash_and_resume")
print("resumed -> status", res["status"], "| refund", refund_of(res), "| model calls", res["llm_calls"], "| tokens", res["tokens_used"])
print("tools executed across crash + resume:", ck2["tools_executed"])
assert ck2["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) < 0.01

print("\nledger:")
print(open(agent.LEDGER).read())
print("state files:", sorted(os.listdir("state")))
