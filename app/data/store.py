"""Where market data and the agent's memory live.

MarketStore (durable)            Cosmos DB in Azure, SQLite locally
  transactions   one document per DLD deal (id = DLD transaction id). The daily agent only
                 APPENDS new deals. Each deal carries a TTL that ends 400 days after the DEAL
                 date, so Cosmos drops old deals by itself and the store stays a rolling window.
  memory         the agent's memory: watermark (last deal date ingested), run history,
                 notes (e.g. unmapped areas), and the published homes snapshot.

HotCache (fast)                  Azure Cache for Redis in Azure, in-process dict locally
  homes blob + version           the app polls the version (a cheap GET) and hot-swaps homes
  search results                 cached per (query, variant, data version) for a few minutes
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.data.dld import pick, to_dt

log = logging.getLogger(__name__)
TTL_SECONDS = 400 * 24 * 3600
HOMES_KEY = "baytak:homes"


# ---------------------------------------------------------------- helpers
def normalise_batch(raw: pd.DataFrame) -> list[dict]:
    """Raw DLD rows -> storable docs with a stable id, an ISO date and a month partition."""
    if raw.empty:
        return []
    id_col, date_col = pick(raw, "id"), pick(raw, "date")
    dates = to_dt(raw[date_col])
    docs = []
    for (_, row), d in zip(raw.iterrows(), dates):
        if pd.isna(d):
            continue
        body = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
        tid = str(row[id_col]) if id_col and pd.notna(row[id_col]) else \
            hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:20]
        docs.append({"id": re.sub(r"[/\\?#]", "_", tid), "date": d.date().isoformat(), "ym": d.strftime("%Y-%m"), "row": body})
    return docs


def homes_to_blob(homes: pd.DataFrame) -> bytes:
    h = homes.copy()
    h["amenities"] = h["amenities"].map(lambda a: "|".join(a) if isinstance(a, list) else (a or ""))
    return gzip.compress(h.to_csv(index=False).encode())


def blob_to_homes(blob: bytes) -> pd.DataFrame:
    h = pd.read_csv(io.BytesIO(gzip.decompress(blob)))
    h["amenities"] = h["amenities"].fillna("").map(lambda s: [a for a in str(s).split("|") if a])
    for c in ("off_plan", "furnished"):
        h[c] = h[c].astype(bool)
    return h


# ------------------------------------------------------------ SQLite (local)
class SQLiteMarketStore:
    kind = "sqlite"

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS tx (id TEXT PRIMARY KEY, date TEXT, body TEXT);
                CREATE INDEX IF NOT EXISTS tx_date ON tx(date);
                CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT);""")

    def upsert_transactions(self, docs: list[dict]) -> dict:
        with self.lock:
            ids = [d["id"] for d in docs]
            existing = set()
            for i in range(0, len(ids), 900):
                q = f"SELECT id FROM tx WHERE id IN ({','.join('?' * len(ids[i:i + 900]))})"
                existing |= {r[0] for r in self.db.execute(q, ids[i:i + 900])}
            self.db.executemany("INSERT OR REPLACE INTO tx VALUES (?,?,?)",
                                [(d["id"], d["date"], json.dumps(d["row"], default=str)) for d in docs])
            cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - TTL_SECONDS))
            self.db.execute("DELETE FROM tx WHERE date < ?", (cutoff,))       # same effect as Cosmos TTL
            self.db.commit()
        return {"new": len(set(ids) - existing), "updated": len(existing)}

    def load_transactions(self, since: str) -> pd.DataFrame:
        with self.lock:
            rows = self.db.execute("SELECT body FROM tx WHERE date >= ?", (since,)).fetchall()
        return pd.DataFrame([json.loads(r[0]) for r in rows])

    def stats(self) -> dict:
        with self.lock:
            n, lo, hi = self.db.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM tx").fetchone()
        return {"transactions": n, "from": lo, "to": hi}

    def get_memory(self, key: str, default=None):
        with self.lock:
            r = self.db.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_memory(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO memory VALUES (?,?)", (key, json.dumps(value, default=str)))
            self.db.commit()


# ----------------------------------------------------------- Cosmos (Azure)
class CosmosMarketStore:
    """Containers: `market_tx` (partition /ym, TTL 400 days) and `agent_memory` (partition /key)."""
    kind = "cosmos"
    PART = 1_500_000        # base64 chars per memory doc (Cosmos items max out at 2 MB)

    def __init__(self, settings: Settings):
        from azure.cosmos import CosmosClient, PartitionKey
        from azure.identity import DefaultAzureCredential

        client = CosmosClient(settings.cosmos_endpoint, credential=settings.cosmos_key or DefaultAzureCredential(),
                              connection_timeout=120)
        db = client.create_database_if_not_exists(settings.cosmos_database)
        # default_ttl=-1 turns TTL on without a default; each item sets its own `ttl`
        self.tx = db.create_container_if_not_exists(id="market_tx", partition_key=PartitionKey(path="/ym"),
                                                    default_ttl=-1)
        self.mem = db.create_container_if_not_exists(id="agent_memory", partition_key=PartitionKey(path="/key"))

    BATCH = 100             # Cosmos transactional batch limit (same partition key)
    WORKERS = 4

    def _write_batch(self, ym: str, items: list[dict]) -> None:
        """One round trip per 100 deals. Retries throttling/timeouts with backoff."""
        ops = [("upsert", (it,)) for it in items]
        for attempt in range(6):
            try:
                self.tx.execute_item_batch(ops, partition_key=ym)
                return
            except Exception as e:                   # 429 / 408 / network hiccup
                if attempt == 5:
                    raise
                wait = min(30, 2 ** attempt)
                log.warning("Cosmos batch (%s, %d items) failed: %s - retrying in %ss", ym, len(items),
                            type(e).__name__, wait)
                time.sleep(wait)

    def upsert_transactions(self, docs: list[dict]) -> dict:
        """Write only deals we don't have yet (deals are immutable), in batches of 100.
        Resumable: if a run dies half-way, the next run skips everything already written."""
        by_month: dict[str, list[dict]] = {}
        for d in docs:
            by_month.setdefault(d["ym"], []).append(d)
        now, jobs, skipped, expired = time.time(), [], 0, 0
        for ym, group in by_month.items():
            ids = [d["id"] for d in group]
            existing = set()
            for i in range(0, len(ids), 500):
                q = "SELECT VALUE c.id FROM c WHERE ARRAY_CONTAINS(@ids, c.id)"
                existing |= set(self.tx.query_items(q, parameters=[{"name": "@ids", "value": ids[i:i + 500]}],
                                                    partition_key=ym))
            todo = []
            for d in group:
                if d["id"] in existing:
                    skipped += 1
                    continue
                expires = pd.Timestamp(d["date"]).timestamp() + TTL_SECONDS
                if expires <= now:              # already outside the rolling window
                    expired += 1
                    continue
                todo.append({**d, "ttl": int(expires - now)})
            jobs += [(ym, todo[i:i + self.BATCH]) for i in range(0, len(todo), self.BATCH)]
        total, done = sum(len(j[1]) for j in jobs), 0
        if total:
            log.info("Writing %d new deals to Cosmos in %d batches (%d already stored)", total, len(jobs), skipped)
        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            for (ym, items), fut in zip(jobs, [pool.submit(self._write_batch, ym, items) for ym, items in jobs]):
                fut.result()
                done += len(items)
                if done % 20_000 < self.BATCH:
                    log.info("  %d / %d deals written", done, total)
        return {"new": total, "updated": skipped}

    def load_transactions(self, since: str) -> pd.DataFrame:
        q = "SELECT VALUE c.row FROM c WHERE c.date >= @since"
        rows = list(self.tx.query_items(q, parameters=[{"name": "@since", "value": since}],
                                        enable_cross_partition_query=True))
        return pd.DataFrame(rows)

    def stats(self) -> dict:
        def one(q):   # the SDK supports single-value aggregates across partitions
            r = list(self.tx.query_items(q, enable_cross_partition_query=True))
            return r[0] if r else None
        return {"transactions": one("SELECT VALUE COUNT(1) FROM c") or 0,
                "from": one("SELECT VALUE MIN(c.date) FROM c"), "to": one("SELECT VALUE MAX(c.date) FROM c")}

    def get_memory(self, key: str, default=None):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            item = self.mem.read_item(key, partition_key=key)
        except CosmosResourceNotFoundError:
            return default
        if item.get("parts"):                   # large value split across several items
            chunks = [self.mem.read_item(f"{key}__part{i}", partition_key=f"{key}__part{i}")["v"] for i in range(item["parts"])]
            return json.loads("".join(chunks))
        return item["v"]

    def set_memory(self, key: str, value) -> None:
        raw = json.dumps(value, default=str)
        if len(raw) <= self.PART:
            self.mem.upsert_item({"id": key, "key": key, "v": value})
            return
        chunks = [raw[i:i + self.PART] for i in range(0, len(raw), self.PART)]
        for i, c in enumerate(chunks):
            self.mem.upsert_item({"id": f"{key}__part{i}", "key": f"{key}__part{i}", "v": c})
        self.mem.upsert_item({"id": key, "key": key, "parts": len(chunks)})


# ------------------------------------------------------------------ caches
class MemoryCache:
    kind = "memory"

    def __init__(self):
        self.d: dict[str, tuple[float, bytes]] = {}

    def get(self, k):
        v = self.d.get(k)
        return None if v is None or (v[0] and v[0] < time.time()) else v[1]

    def set(self, k, v, ttl: int | None = None):
        self.d[k] = (time.time() + ttl if ttl else 0, v if isinstance(v, bytes) else str(v).encode())


class RedisCache:
    kind = "redis"

    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url, socket_timeout=5, socket_connect_timeout=5)

    def get(self, k):
        try:
            return self.r.get(k)
        except Exception:                        # cache must never take the app down
            return None

    def set(self, k, v, ttl: int | None = None):
        try:
            self.r.set(k, v, ex=ttl)
        except Exception:
            pass


# ------------------------------------------------------------ publishing
def publish_homes(store, cache, homes: pd.DataFrame, meta: dict) -> str:
    """Write the homes snapshot to durable memory and the hot cache; return the new version."""
    blob = homes_to_blob(homes)
    version = f"{meta.get('as_of')}-{hashlib.sha1(blob).hexdigest()[:8]}"
    payload = {"version": version, "meta": meta, "blob": base64.b64encode(blob).decode()}
    store.set_memory("homes_snapshot", payload)
    cache.set(f"{HOMES_KEY}:blob", blob)
    cache.set(f"{HOMES_KEY}:meta", json.dumps({"version": version, **meta}))
    cache.set(f"{HOMES_KEY}:version", version)          # written last: readers see a complete snapshot
    return version


def current_version(store, cache) -> str | None:
    v = cache.get(f"{HOMES_KEY}:version")
    if v:
        return v.decode()
    snap = store.get_memory("homes_snapshot")
    return snap["version"] if snap else None


def load_homes(store, cache) -> tuple[pd.DataFrame, dict] | None:
    blob, meta = cache.get(f"{HOMES_KEY}:blob"), cache.get(f"{HOMES_KEY}:meta")
    if blob and meta:
        return blob_to_homes(blob), json.loads(meta)
    snap = store.get_memory("homes_snapshot")
    if not snap:
        return None
    blob = base64.b64decode(snap["blob"])
    cache.set(f"{HOMES_KEY}:blob", blob)                 # warm the cache for the next reader
    cache.set(f"{HOMES_KEY}:meta", json.dumps({"version": snap["version"], **snap["meta"]}))
    cache.set(f"{HOMES_KEY}:version", snap["version"])
    return blob_to_homes(blob), {"version": snap["version"], **snap["meta"]}


def get_market_store(settings: Settings):
    if settings.use_cosmos:
        return CosmosMarketStore(settings)
    return SQLiteMarketStore(settings.data_dir / "market.db")


def get_cache(settings: Settings):
    return RedisCache(settings.redis_url) if settings.redis_url else MemoryCache()
