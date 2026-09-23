"""Live Dubai Pulse refresh, against a fake API (no network)."""
from datetime import date, timedelta

import httpx
import numpy as np

from app.data import live
from app.data.live import DubaiPulseClient, LiveDataRefresher


def _rows(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    comms = [("Dubai Marina", "Marsa Dubai"), ("Jumeirah Village Circle", "Al Barsha South Fourth"),
             ("Business Bay", "Business Bay"), ("Dubai Hills Estate", "Hadaeq Sheikh Mohammed Bin Rashid")]
    out = []
    for i in range(n):
        m, a = comms[i % 4]
        beds = int(rng.integers(0, 4))
        size = [40, 75, 115, 165][beds] * float(rng.uniform(.9, 1.1))
        out.append({"transaction_id": i, "instance_date": (date.today() - timedelta(days=int(rng.integers(1, 300)))).isoformat(),
                    "trans_group_en": "Sales", "property_sub_type_en": "Flat", "property_usage_en": "Residential",
                    "reg_type_en": "Existing Properties", "area_name_en": a, "master_project_en": m,
                    "project_name_en": f"{m} P", "building_name_en": f"{m} Tower {i % 15}",
                    "rooms_en": "Studio" if beds == 0 else f"{beds} B/R", "procedure_area": round(size, 1),
                    "actual_worth": round(size * 10.76 * float(rng.uniform(1300, 2300)))})
    return out


def _fake_api(reject_filter=False):
    data = _rows()
    calls = {"token": 0, "pages": 0}

    def handler(req: httpx.Request):
        if "accesstoken" in str(req.url):
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": "t", "expires_in": "1799"})
        assert req.headers["Authorization"] == "Bearer t"
        if reject_filter and "filter" in req.url.params:
            return httpx.Response(400, json={"error": "bad filter"})
        off, lim = int(req.url.params["offset"]), int(req.url.params["limit"])
        calls["pages"] += 1
        return httpx.Response(200, json={"data": data[off:off + lim]})
    return handler, calls


def _client(handler):
    c = DubaiPulseClient("k", "s", page_size=500)
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_client_pages_and_falls_back_when_filter_rejected():
    handler, calls = _fake_api(reject_filter=True)
    df = _client(handler).fetch("https://api/x", "instance_date", date.today() - timedelta(days=365))
    assert len(df) == 1500 and calls["pages"] == 4 and calls["token"] == 1


def test_refresh_swaps_live_homes_into_recommender(client, monkeypatch):
    from app.main import state
    rec = state["rec"]
    before = rec.listings
    handler, _ = _fake_api()
    monkeypatch.setattr(live, "DubaiPulseClient", lambda *a, **k: _client(handler))
    s = rec.s.model_copy(update={"dubai_pulse_api_key": "k", "dubai_pulse_api_secret": "s", "dubai_pulse_rents_url": None})
    r = LiveDataRefresher(s, rec)
    try:
        r.refresh_once()
        assert r.status["state"] == "live" and r.status["homes"] >= 100
        assert (rec.listings["data_source"] == "DLD").all()
        resp = client.post("/recommend", json={"purpose": "sale", "budget_aed": 2_000_000, "min_bedrooms": 1}).json()
        assert resp["recommendations"] and resp["recommendations"][0]["listing"]["listing_id"].startswith("DLD-")
        assert client.get("/health").json()["data"]["source"] == "DLD (real)"
    finally:
        rec.swap_listings(before)                      # leave other tests on synthetic data
