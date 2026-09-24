"""Run the home-search agent evaluation.

    python -m scripts.run_evals                       # in-process, with whatever LLM .env configures (or offline)
    python -m scripts.run_evals --offline             # force the deterministic agents (what CI runs)
    python -m scripts.run_evals --judge               # + LLM-as-judge relevance / faithfulness scores
    python -m scripts.run_evals --url https://<app>   # against the deployed app
    python -m scripts.run_evals --engine classic      # compare with the single-loop agent

Writes evals/results/<timestamp>-<engine>-<mode>.json and prints a summary table.
Exit code 1 if any score is below the thresholds in app/evals.py (use it as a release gate)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from app.evals import THRESHOLDS, load_cases, run
from app.config import ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--engine", choices=["langgraph", "classic"])
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="don't fail on thresholds")
    a = ap.parse_args()
    if a.offline:
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
        os.environ["LLM_PROVIDER"] = "azure"
    if a.engine:
        os.environ["AGENT_ENGINE"] = a.engine

    judge_llm = None
    if a.url:
        import httpx
        http = httpx.Client(timeout=180)

        def chat(msg, income):
            return http.post(a.url.rstrip("/") + "/chat", json={"message": msg, "monthly_income_aed": income}).json()
        engine = "deployed"
    else:
        from app.config import get_settings
        from app.main import make_agent
        from app.recsys.pipeline import Recommender
        from app.schemas import ChatRequest
        from app.storage.feedback import get_store
        s = get_settings()
        agent = make_agent(Recommender(s, get_store(s)), s)

        def chat(msg, income):
            return agent.chat(ChatRequest(message=msg, monthly_income_aed=income)).model_dump()
        engine = s.agent_engine
        if a.judge:
            from app.agents.llm import chat_model
            judge_llm = chat_model(s)
    rows, summary = run(load_cases(), chat, judge_llm)

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{engine}-{(summary['mode'] or 'na').replace(':', '_')}.json"
    path.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, default=str))

    print(f"\nEngine: {engine}   mode: {summary['mode']}   cases: {summary['cases']}")
    print(f"{'metric':<24}{'score':>10}{'threshold':>12}")
    failed = []
    for k in ("task_success", "groundedness", "constraint_adherence", "routing_accuracy", "tool_recall",
              "safety_accuracy", "language_accuracy", "judge_relevance", "judge_faithfulness"):
        v = summary.get(k)
        if v is None:
            continue
        th = THRESHOLDS.get(k)
        flag = "" if th is None or v >= th else "  <-- below"
        if flag:
            failed.append(k)
        print(f"{k:<24}{v:>10}{(th if th is not None else ''):>12}{flag}")
    print(f"{'latency p50 / p95 (ms)':<24}{summary['latency_p50_ms']:>10} / {summary['latency_p95_ms']}")
    if summary["failed"]:
        print("failed cases:", ", ".join(summary["failed"]))
    print(f"saved {path.relative_to(ROOT)}")
    return 1 if failed and not a.no_gate else 0


if __name__ == "__main__":
    sys.exit(main())
