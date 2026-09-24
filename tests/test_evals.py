"""Release gate: the offline agent team must meet the evaluation thresholds on the eval set.
(With an LLM configured, run `python -m scripts.run_evals` - same cases, same scoring.)"""
from app.evals import THRESHOLDS, load_cases, run, score_case
from app.schemas import ChatRequest


def test_offline_agent_meets_eval_thresholds(client):
    from app.main import state
    agent = state["agent"]
    rows, summary = run(load_cases(), lambda m, inc: agent.chat(ChatRequest(message=m, monthly_income_aed=inc)).model_dump())
    print(summary)
    for metric, threshold in THRESHOLDS.items():
        assert summary[metric] is not None and summary[metric] >= threshold, (metric, summary)


def test_grounding_score_catches_hallucinated_real_dld_id():
    case = {"id": "x", "message": "m", "expect": {"purpose": "sale"}}
    resp = {"blocked": False, "grounded": False, "answer": "Buy [DLD-55E737F2]", "agents": [], "tool_trace": [],
            "recommendations": [{"listing": {"purpose": "sale", "price_aed": 1, "bedrooms": 1, "property_type": "apartment"}}]}
    assert not score_case(case, resp, 1.0)["passed"]


def test_listing_id_pattern_covers_synthetic_and_real_ids():
    from app.schemas import LISTING_ID_RE
    assert LISTING_ID_RE.findall("[DLD-55E737F2] and [DHM-00001], not DLD-xyz") == ["DLD-55E737F2", "DHM-00001"]
