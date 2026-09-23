"""Feedback event store: Azure Cosmos DB in the cloud, SQLite locally."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from app.config import Settings


class SQLiteFeedbackStore:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, ts REAL, body TEXT)")
        self.conn.commit()

    def log(self, event: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO events VALUES (?, ?, ?)", (str(uuid.uuid4()), time.time(), json.dumps(event)))
            self.conn.commit()

    def events(self, limit: int = 100_000) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT body FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]


class CosmosFeedbackStore:
    """Partitioned by session_id. Use serverless Cosmos for a near-zero idle cost."""

    def __init__(self, settings: Settings):
        from azure.cosmos import CosmosClient, PartitionKey
        from azure.identity import DefaultAzureCredential

        cred = settings.cosmos_key or DefaultAzureCredential()
        client = CosmosClient(settings.cosmos_endpoint, credential=cred)
        db = client.create_database_if_not_exists(settings.cosmos_database)
        self.container = db.create_container_if_not_exists(
            id=settings.cosmos_container, partition_key=PartitionKey(path="/session_id"))

    def log(self, event: dict) -> None:
        self.container.create_item({"id": str(uuid.uuid4()), "ts": time.time(), **event})

    def events(self, limit: int = 100_000) -> list[dict]:
        q = f"SELECT TOP {int(limit)} * FROM c ORDER BY c.ts DESC"
        return list(self.container.query_items(q, enable_cross_partition_query=True))


def get_store(settings: Settings):
    if settings.use_cosmos:
        return CosmosFeedbackStore(settings)
    return SQLiteFeedbackStore(settings.data_dir / "feedback.db")
