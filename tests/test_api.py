def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["ranker_loaded"] and h["llm"] == "offline"


def test_recommend_respects_filters(client):
    r = client.post("/recommend", json={"purpose": "sale", "budget_aed": 1_500_000, "min_bedrooms": 2,
                                        "work_location": "DIFC"}).json()
    assert r["recommendations"], "expected results"
    ranked = [x for x in r["recommendations"] if x["source"] != "explore"]
    for x in ranked:
        assert x["listing"]["purpose"] == "sale" and x["listing"]["bedrooms"] >= 2
        assert x["listing"]["price_aed"] <= 1_500_000 * 1.3
    assert len({x["listing"]["listing_id"] for x in r["recommendations"]}) == len(r["recommendations"])


def test_chat_is_grounded_and_uses_tools(client):
    r = client.post("/chat", json={"message": "Family of 4, budget AED 2.5M, work in DIFC, Golden Visa please",
                                   "monthly_income_aed": 50_000}).json()
    assert r["grounded"] and r["recommendations"]
    tools = [t["tool"] for t in r["tool_trace"]]
    assert tools[0] == "search_homes" and "check_golden_visa" in tools and "check_affordability" in tools


def test_feedback_and_ab_metrics(client):
    r = client.post("/recommend?session_id=s1", json={"purpose": "rent", "budget_aed": 100_000}).json()
    lid = r["recommendations"][0]["listing"]["listing_id"]
    assert client.post("/feedback", json={"session_id": "s1", "listing_id": lid, "event": "save",
                                          "variant": r["variant"]}).json()["ok"]
    ab = client.get("/metrics/ab").json()
    assert ab[r["variant"]]["impressions"] >= 1 and ab[r["variant"]]["positives"] >= 1
    assert client.post("/feedback", json={"session_id": "s1", "listing_id": "DHM-99999", "event": "save"}).status_code == 404


def test_page_is_diverse(client):
    from collections import Counter
    r = client.post("/recommend", json={"purpose": "sale", "budget_aed": 2_000_000, "min_bedrooms": 2}).json()
    counts = Counter(x["listing"]["community"] for x in r["recommendations"] if x["source"] != "explore")
    assert max(counts.values()) <= 3
