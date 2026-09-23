"""Market Data Agent: daily append into the store (Cosmos stand-in: SQLite), Redis cache
(fakeredis), HTTP Range reads, guardrails, agent memory, LLM tool loop, and app hot-swap."""
import http.server
import json
import os
import threading
from datetime import date, timedelta
from types import SimpleNamespace as NS

import fakeredis
import numpy as np
import pandas as pd
import pytest

from app.config import get_settings
from app.data import incremental
from app.data.agent import DataTools, MarketDataAgent
from app.data.store import RedisCache, SQLiteMarketStore, current_version

COMMS = [("Dubai Marina", "Marsa Dubai"), ("Jumeirah Village Circle", "Al Barsha South Fourth"),
         ("Business Bay", "Business Bay"), ("Dubai Hills Estate", "Hadaeq Sheikh Mohammed Bin Rashid")]


def history(n=20_000, days=420, start_id=0, end_days_ago=1, seed=0, descending=False):
    """A DLD-shaped CSV table sorted by date, ending `end_days_ago` days ago."""
    rng = np.random.default_rng(seed)
    ago = np.sort(rng.integers(end_days_ago, days, n))[::-1]           # oldest first
    rows = []
    for i, a in enumerate(ago):
        m, area = COMMS[i % 4]
        beds = int(rng.integers(0, 4))
        size = [40, 75, 115, 165][beds] * float(rng.uniform(.9, 1.1))
        rows.append({"transaction_id": f"T{start_id + i}", "instance_date": (date.today() - timedelta(days=int(a))).isoformat(),
                     "trans_group_en": "Sales", "property_sub_type_en": "Flat", "property_usage_en": "Residential",
                     "reg_type_en": "Existing Properties", "area_name_en": area, "master_project_en": m,
                     "project_name_en": f"{m} P", "building_name_en": f"{m} Tower {i % 15}",
                     "rooms_en": "Studio" if beds == 0 else f"{beds} B/R", "procedure_area": round(size, 1),
                     "actual_worth": round(size * 10.76 * float(rng.uniform(1300, 2300))),
                     "some_unused_column": "x" * 40})
    df = pd.DataFrame(rows)
    return df.iloc[::-1] if descending else df


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with HTTP Range support, like a CDN."""
    def log_message(self, *a):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return None
        size = os.path.getsize(path)
        f = open(path, "rb")
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), size - 1)
            f.seek(a)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._left = b - a + 1
            return f
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._left = size
        return f

    def copyfile(self, source, outputfile):
        outputfile.write(source.read(self._left))


@pytest.fixture
def server(tmp_path):
    handler = lambda *a, **k: RangeHandler(*a, directory=str(tmp_path), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield tmp_path, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(incremental, "BLOCK", 128 << 10)       # small blocks so the test file is "big"
    monkeypatch.setattr(incremental, "PROBE", 32 << 10)
    s = get_settings().model_copy(update={"min_homes": 50, "min_communities": 3, "azure_openai_endpoint": None,
                                          "dubai_pulse_api_key": None})
    store = SQLiteMarketStore(tmp_path / "market.db")           # stands in for Cosmos DB
    cache = RedisCache.__new__(RedisCache)
    cache.r = fakeredis.FakeRedis()                             # stands in for Azure Cache for Redis
    return s, store, cache


def run(env, url, client=None):
    s, store, cache = env
    return MarketDataAgent(s, DataTools(s, store, cache, [url]), client=client).run()


def test_daily_append_downloads_only_the_new_tail(server, env):
    root, base = server
    hist = history()
    hist.to_csv(root / "tx.csv", index=False)
    size = os.path.getsize(root / "tx.csv")

    seed = run(env, f"{base}/tx.csv")                           # day 1: seed the rolling 12 months
    assert seed["outcome"] == "APPENDED" and seed["published"]
    s, store, cache = env
    n_seed = store.stats()["transactions"]
    v1 = current_version(store, cache)
    assert v1 and cache.get("baytak:homes:version").decode() == v1

    # day 2: DLD regenerates the file with 300 new deals from today at the end
    new = history(n=300, days=1, start_id=10**6, end_days_ago=0, seed=7)
    pd.concat([hist, new]).to_csv(root / "tx.csv", index=False)
    day2 = run(env, f"{base}/tx.csv")
    assert day2["outcome"] == "APPENDED" and day2["new_deals"] == 300
    assert store.stats()["transactions"] == n_seed + 300
    assert "tail" in day2["fetch"]["method"]
    assert day2["fetch"]["mb_downloaded"] * 1e6 < size / 5      # a sliver of the file, not all of it
    assert store.get_memory("watermark") == date.today().isoformat()
    assert current_version(store, cache) != v1                  # new homes published to Redis

    day2b = run(env, f"{base}/tx.csv")                          # same file again: nothing new
    assert day2b["outcome"] == "NO NEW DATA" and store.stats()["transactions"] == n_seed + 300
    runs = store.get_memory("runs")                             # agent memory keeps run history
    assert [r["outcome"] for r in runs] == ["APPENDED", "APPENDED", "NO NEW DATA"]


def test_bad_batch_is_not_appended_and_is_remembered(server, env):
    root, base = server
    hist = history()
    hist.to_csv(root / "tx.csv", index=False)
    run(env, f"{base}/tx.csv")
    s, store, cache = env
    wm, n, v = store.get_memory("watermark"), store.stats()["transactions"], current_version(store, cache)

    bad = history(n=300, days=1, start_id=10**6, end_days_ago=0, seed=7)
    bad["actual_worth"] *= 4                                    # prices x4 overnight = broken feed
    pd.concat([hist, bad]).to_csv(root / "tx.csv", index=False)
    r = run(env, f"{base}/tx.csv")
    assert r["outcome"] == "NOT APPENDED" and "off" in r["report"]
    assert store.get_memory("watermark") == wm and store.stats()["transactions"] == n
    assert current_version(store, cache) == v                   # live homes untouched
    assert "rejected" in store.get_memory("notes")[-1]["note"]  # remembered for next time


def test_newest_first_files_read_from_the_head(server, env):
    root, base = server
    history(descending=True).to_csv(root / "tx.csv", index=False)
    run(env, f"{base}/tx.csv")
    new = history(n=200, days=1, start_id=10**6, end_days_ago=0, seed=3, descending=True)
    pd.concat([new, history(descending=True)]).to_csv(root / "tx.csv", index=False)
    r = run(env, f"{base}/tx.csv")
    assert r["new_deals"] == 200 and "head" in r["fetch"]["method"]


def test_unsorted_files_fall_back_to_one_full_pass(server, env):
    root, base = server
    history().sample(frac=1, random_state=0).to_csv(root / "tx.csv", index=False)
    r = run(env, f"{base}/tx.csv")
    assert r["published"] and "not date-sorted" in r["fetch"]["method"]


class _FakeLLM:
    """Tries to append straight after fetching (skipping validation), then behaves."""
    def __init__(self):
        self.step = 0

    def create(self, **kw):
        plan = [("recall_memory", {}), ("fetch_new_deals", {}), ("append_deals", {}), ("validate_batch", {}),
                ("append_deals", {}), ("rebuild_homes", {}), ("remember", {"note": "first seed done"}),
                ("market_brief", {})]
        if self.step < len(plan):
            name, args = plan[self.step]
            self.step += 1
            call = NS(id=f"c{self.step}", type="function", function=NS(name=name, arguments=json.dumps(args)))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])
        return NS(choices=[NS(message=NS(content="APPENDED - test", tool_calls=None))])


def test_llm_agent_cannot_skip_validation(server, env):
    root, base = server
    history().to_csv(root / "tx.csv", index=False)
    r = run(env, f"{base}/tx.csv", client=NS(chat=NS(completions=_FakeLLM())))
    first_append = next(t for t in r["trace"] if t["tool"] == "append_deals")
    assert '"appended": false' in first_append["result"]         # refused in code
    assert r["published"] and r["mode"] == "azure-openai"
    assert env[1].get_memory("notes")[-1]["note"] == "first seed done"


def test_app_hot_swaps_published_homes(server, env, client):
    from app.data.watcher import StoreWatcher
    from app.main import state
    root, base = server
    history().to_csv(root / "tx.csv", index=False)
    run(env, f"{base}/tx.csv")
    s, store, cache = env
    rec = state["rec"]
    before = rec.listings
    try:
        w = StoreWatcher(s, rec, store, cache)
        assert w.check_once() and w.status["state"] == "live"
        assert (rec.listings["data_source"] == "DLD").all()
        assert not w.check_once()                                # same version: no reload
    finally:
        rec.swap_listings(before)
