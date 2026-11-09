"""Part D tasks 5-8: tools, ReAct loop with safety limits, checkpointing and resume."""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm  # noqa: E402
from rag import retrieve  # noqa: E402  (the same retrieval built in tasks 1-3)

DB_PATH = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"
LATE_MINUTES = 45
AUTO_APPROVE_LIMIT = 1000

AGENT_SYSTEM_PROMPT = ("You are a refund AGENT for the support desk. Use the tools to look up the order, check the policy, "
                       "calculate the refund and then issue it or escalate it. When you are done reply with one line "
                       "starting with REFUND, ESCALATE, DENIED or NOT_FOUND, followed by a colon and a short explanation.")

# which run is currently executing, so the ledger lines can be traced back to it
CURRENT = {"run_id": None, "scenario_id": None}


class SimulatedCrash(Exception):
    pass


# ----------------------------------------------------------------------------- tools
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    # refund only when the delivery was later than 45 minutes; a missing time (cancelled order) gives 0
    if delivery_minutes is None or order_amount is None or delivery_minutes <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
             "run_id": CURRENT["run_id"], "scenario_id": CURRENT["scenario_id"],
             "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the database and return its status, amount and delivery minutes.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100048"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.",
          {"query": {"type": "string", "description": "Free-text search query"}}, ["query"]),
    _spec("calculate_refund", "Apply the refund rules and return the refund amount in INR.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Record an automatic refund of up to INR 1000 in the refund ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ----------------------------------------------------------------------------- checkpoint helpers
def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _load_checkpoint(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]}


def _fallback(state, status):
    tools = ", ".join(state["tools_executed"]) or "none"
    summary = (f"{status}: automated handling stopped after {state['llm_calls']} model call(s) and "
               f"{len(state['tools_executed'])} tool step(s) (tools run: {tools}). Please review manually. "
               f"Customer request: {state['user_message']}")
    return _result(state, status, summary)


# ----------------------------------------------------------------------------- the ReAct loop
def _loop(state, crash_after_tools=None):
    started = time.monotonic()
    while True:
        if time.monotonic() - started >= state["time_limit_seconds"]:          # (c) wall-clock limit
            return _fallback(state, "FALLBACK_TIMEOUT")
        if state["llm_calls"] >= state["max_iterations"]:                      # (a) iteration limit
            return _fallback(state, "FALLBACK_MAX_ITERATIONS")

        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:                        # (b) token limit, before running any tool
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")

        msg = resp.choices[0].message
        if not msg.tool_calls:                                                 # final answer
            final = msg.content or ""
            status = final.split(":")[0].strip() or "UNKNOWN"
            return _result(state, status, final)

        for tc in msg.tool_calls:
            state["messages"].append({"role": "assistant", "content": None,
                                      "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            try:
                observation = TOOL_MAP[tc.name](**tc.arguments)
            except Exception as e:                                             # a failing tool must not kill the loop
                observation = {"error": f"{type(e).__name__}: {e}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(observation)})
            state["tools_executed"].append(tc.name)
            state["step"] += 1
            state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
            _save_checkpoint(state)                                            # checkpoint after every observation
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    run_id = run_id or f"run_{scenario_id}"
    CURRENT.update(run_id=run_id, scenario_id=scenario_id)
    state = {"run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": [],
             "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    state = _load_checkpoint(run_id)
    CURRENT.update(run_id=run_id, scenario_id=state["scenario_id"])
    return _loop(state)
