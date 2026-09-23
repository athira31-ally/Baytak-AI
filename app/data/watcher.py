"""Web-app side of the daily refresh. Every few minutes, read the homes version from Redis
(a single cheap GET; Cosmos if there is no Redis). When the Market Data Agent has published a
new version, load the snapshot and hot-swap it into the recommender. No rebuild, no redeploy."""
from __future__ import annotations

import logging
import threading

from app.data import reference
from app.data.store import current_version, get_cache, get_market_store, load_homes

log = logging.getLogger(__name__)


class StoreWatcher:
    def __init__(self, settings, recommender, store=None, cache=None):
        self.s, self.rec = settings, recommender
        self.store = store if store is not None else get_market_store(settings)
        self.cache = cache if cache is not None else get_cache(settings)
        self.version: str | None = None
        self.status = {"state": "starting", "version": None, "error": None,
                       "store": self.store.kind, "cache": self.cache.kind}
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="store-watcher", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def check_once(self) -> bool:
        v = current_version(self.store, self.cache)
        if not v:
            self.status["state"] = "waiting for the first publish (serving built-in homes)"
            return False
        if v == self.version:
            return False
        loaded = load_homes(self.store, self.cache)
        if loaded is None:
            return False
        homes, meta = loaded
        reference.apply_price_overrides(meta.get("communities", {}))
        self.rec.swap_listings(homes)
        self.version = v
        self.status.update(state="live", version=v, error=None, homes=len(homes), as_of=meta.get("as_of"))
        log.info("Hot-swapped homes to version %s", v)
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception as e:
                log.exception("Homes sync failed; serving previous homes")
                self.status.update(state="error (serving previous homes)", error=str(e)[:300])
            self._stop.wait(self.s.homes_sync_seconds)
