"""CosmosMarketStore logic against an in-memory fake container (no Azure needed)."""
from datetime import date, timedelta

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from app.data.store import CosmosMarketStore


class FakeContainer:
    def __init__(self):
        self.items = {}

    def upsert_item(self, item):
        self.items[item["id"]] = item

    def read_item(self, item, partition_key):
        if item not in self.items:
            raise CosmosResourceNotFoundError(message="nope")
        return self.items[item]

    def query_items(self, q, parameters=None, partition_key=None, enable_cross_partition_query=False):
        p = {x["name"]: x["value"] for x in parameters or []}
        if "ARRAY_CONTAINS" in q:
            return [i["id"] for i in self.items.values() if i["id"] in p["@ids"] and i["ym"] == partition_key]
        if "c.row" in q:
            return [i["row"] for i in self.items.values() if i["date"] >= p["@since"]]
        if "COUNT" in q:
            return [len(self.items)]
        dates = [i["date"] for i in self.items.values()]
        return [min(dates) if "MIN" in q else max(dates)] if dates else []


def store():
    s = CosmosMarketStore.__new__(CosmosMarketStore)
    s.tx, s.mem = FakeContainer(), FakeContainer()
    return s


def doc(i, days_ago):
    d = date.today() - timedelta(days=days_ago)
    return {"id": f"T{i}", "date": d.isoformat(), "ym": d.strftime("%Y-%m"), "row": {"transaction_id": f"T{i}"}}


def test_upsert_counts_new_sets_ttl_and_skips_expired():
    s = store()
    r = s.upsert_transactions([doc(1, 5), doc(2, 10), doc(3, 500)])     # T3 is older than the window
    assert r == {"new": 2, "updated": 0} and set(s.tx.items) == {"T1", "T2"}
    ttl = s.tx.items["T1"]["ttl"]
    assert 390 * 86400 < ttl <= 400 * 86400                             # expires 400 days after the deal
    assert s.upsert_transactions([doc(1, 5), doc(4, 1)]) == {"new": 1, "updated": 1}
    assert s.stats()["transactions"] == 3 and len(s.load_transactions((date.today() - timedelta(days=7)).isoformat())) == 2


def test_large_memory_values_are_split_across_items():
    s = store()
    s.PART = 1000
    big = {"blob": "x" * 5000}
    s.set_memory("homes_snapshot", big)
    assert s.mem.items["homes_snapshot"]["parts"] > 1
    assert s.get_memory("homes_snapshot") == big
    assert s.get_memory("missing", "d") == "d"
