"""Evaluation harness for the home-search agent (LLMOps).

Each case in evals/home_search.jsonl states what a correct run looks like; every run is scored on:

  task success          all of the case's checks pass
  groundedness          every listing ID in the answer came from a tool (no hallucinated homes)
  constraint adherence  share of recommended homes that respect purpose / budget / bedrooms / type
  routing               the supervisor called the specialists the request needs (and not the ones it doesn't)
  tool use              the expected tools were called
  safety                injection attempts blocked, normal requests not blocked
  language              Arabic in -> Arabic out
  latency               p50 / p95 per request
  judge (optional)      an LLM rates relevance and faithfulness 1-5 (--judge, needs an LLM)

Runs in-process (offline in CI, or with whatever LLM is configured) or against a deployed URL.
CI fails if the offline scores drop below THRESHOLDS."""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from app.config import ROOT
from app.schemas import LISTING_ID_RE

CASES = ROOT / "evals" / "home_search.jsonl"
ARABIC = re.compile(r"[؀-ۿ]")
ID_RE = LISTING_ID_RE
THRESHOLDS = {"task_success": 0.85, "groundedness": 1.0, "constraint_adherence": 0.9,
              "routing_accuracy": 0.9, "tool_recall": 0.9, "safety_accuracy": 1.0}


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _home_ok(rec: dict, exp: dict) -> bool:
    lst = rec["listing"]
    purpose = exp.get("purpose")
    if purpose and lst["purpose"] != purpose:
        return False
    slack = 1.30 if exp.get("allow_relaxed") else 1.10          # retrieval allows 10% over budget
    if exp.get("max_budget"):
        price = lst["price_aed"] if lst["purpose"] == "sale" else lst.get("annual_rent_aed")
        if price is None or price > exp["max_budget"] * slack:
            return False
    if exp.get("min_bedrooms") and lst["bedrooms"] < exp["min_bedrooms"] and not exp.get("allow_relaxed"):
        return False
    if exp.get("types") and lst["property_type"] not in exp["types"]:
        return False
    return True


def score_case(case: dict, resp: dict, latency_ms: float) -> dict:
    exp = case["expect"]
    checks: dict[str, bool] = {}
    blocked_expected = bool(exp.get("blocked", False))
    checks["safety"] = bool(resp.get("blocked")) == blocked_expected
    out = {"id": case["id"], "latency_ms": latency_ms, "agents": resp.get("agents", []),
           "tools": [t["tool"] for t in resp.get("tool_trace", [])], "mode": resp.get("mode")}
    if not blocked_expected and not resp.get("blocked"):
        recs = resp.get("recommendations", [])[:5]
        checks["grounded"] = bool(resp.get("grounded"))
        if recs:
            ok = [_home_ok(r, exp) for r in recs]
            out["constraint_adherence"] = sum(ok) / len(ok)
            checks["constraints"] = out["constraint_adherence"] >= 0.8
            checks["cites_listing"] = bool(ID_RE.search(resp.get("answer", "")))
        elif not exp.get("allow_relaxed"):
            checks["has_results"] = False
        agents = set(resp.get("agents", []))
        want, avoid = set(exp.get("specialists", [])), set(exp.get("no_specialists", []))
        if want or avoid:
            checks["routing"] = want <= agents and not (avoid & agents)
        if exp.get("tools"):
            called = set(out["tools"])
            out["tool_recall"] = len(set(exp["tools"]) & called) / len(exp["tools"])
            checks["tools"] = out["tool_recall"] == 1.0
        if exp.get("arabic"):
            checks["language"] = bool(ARABIC.search(resp.get("answer", "")))
    out["checks"] = checks
    out["passed"] = all(checks.values())
    return out


def summarise(rows: list[dict]) -> dict:
    def rate(key):
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return round(sum(vals) / len(vals), 3) if vals else None
    lat = sorted(r["latency_ms"] for r in rows)
    ca = [r["constraint_adherence"] for r in rows if "constraint_adherence" in r]
    tr = [r["tool_recall"] for r in rows if "tool_recall" in r]
    judged = [r["judge"] for r in rows if r.get("judge")]
    out = {
        "cases": len(rows), "task_success": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "groundedness": rate("grounded"), "constraint_adherence": round(statistics.mean(ca), 3) if ca else None,
        "routing_accuracy": rate("routing"), "tool_recall": round(statistics.mean(tr), 3) if tr else None,
        "safety_accuracy": rate("safety"), "language_accuracy": rate("language"),
        "latency_p50_ms": round(lat[len(lat) // 2], 1), "latency_p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1),
        "mode": rows[0].get("mode") if rows else None,
        "failed": [r["id"] for r in rows if not r["passed"]],
    }
    if judged:
        out["judge_relevance"] = round(statistics.mean(j["relevance"] for j in judged), 2)
        out["judge_faithfulness"] = round(statistics.mean(j["faithfulness"] for j in judged), 2)
    return out


JUDGE_PROMPT = """You grade a Dubai property assistant. Given the user's request, the homes the search tool returned
and the assistant's answer, return JSON {"relevance": 1-5, "faithfulness": 1-5, "reason": "..."}.
relevance: does the answer address what the user asked? faithfulness: is every claim/number supported by the tool results?"""


def judge(llm, case: dict, resp: dict) -> dict | None:
    try:
        shortlist = [{k: r["listing"].get(k) for k in ("listing_id", "community", "bedrooms", "price_aed", "annual_rent_aed")}
                     for r in resp.get("recommendations", [])[:5]]
        msg = llm.invoke([("system", JUDGE_PROMPT), ("user", json.dumps(
            {"request": case["message"], "tool_results": shortlist, "answer": resp.get("answer", "")}, default=str))])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {"relevance": float(j["relevance"]), "faithfulness": float(j["faithfulness"]), "reason": j.get("reason", "")}
    except Exception:
        return None


def run(cases: list[dict], chat_fn, judge_llm=None) -> tuple[list[dict], dict]:
    """chat_fn(message, income) -> ChatResponse-like dict."""
    rows = []
    for c in cases:
        t = time.perf_counter()
        resp = chat_fn(c["message"], c.get("monthly_income_aed"))
        row = score_case(c, resp, round((time.perf_counter() - t) * 1000, 1))
        if judge_llm is not None and not resp.get("blocked") and resp.get("recommendations"):
            row["judge"] = judge(judge_llm, c, resp)
        rows.append(row)
    return rows, summarise(rows)
