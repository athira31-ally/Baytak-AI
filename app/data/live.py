"""LIVE Dubai Land Department data via the official Dubai Pulse API.

DLD updates its open transaction data daily. When DUBAI_PULSE_API_KEY / _SECRET are set,
the app fetches the latest 12 months of sales (and optionally Ejari rents) in the
background on startup and then every LIVE_REFRESH_HOURS, rebuilds the homes and swaps
them into the recommender without a restart. While a refresh runs, or if the API is
down, the app keeps serving the data it already has.

Access: register at dubaipulse.gov.ae, request the "dld_transactions-open-api" (and
"dld_rent_contracts-open-api") datasets, and you receive an API key + secret by email.
Check your setup with:  python -m scripts.check_dubai_pulse
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import date, timedelta

import httpx
import pandas as pd

from app.config import Settings
from app.data import reference
from app.data.dld import build

log = logging.getLogger(__name__)
TOKEN_URL = "https://api.dubaipulse.gov.ae/oauth/client_credential/accesstoken?grant_type=client_credentials"


class DubaiPulseClient:
    def __init__(self, key: str, secret: str, page_size: int = 1000, timeout: float = 60):
        self.key, self.secret, self.page_size = key, secret, page_size
        self.http = httpx.Client(timeout=timeout)
        self._token: str | None = None
        self._expires = 0.0

    def token(self) -> str:
        if self._token and time.time() < self._expires:
            return self._token
        r = self.http.post(TOKEN_URL, data={"client_id": self.key, "client_secret": self.secret},
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        # tokens last ~30 min; refresh a little early
        self._expires = time.time() + max(60, int(body.get("expires_in", 1800)) - 120)
        return self._token

    @staticmethod
    def _rows(resp: httpx.Response) -> list[dict]:
        if "csv" in resp.headers.get("content-type", ""):
            return pd.read_csv(io.StringIO(resp.text)).to_dict("records")
        body = resp.json()
        if isinstance(body, list):
            return body
        for key in ("data", "result", "results", "records", "value", "items"):
            if isinstance(body.get(key), list):
                return body[key]
            if isinstance(body.get(key), dict):          # e.g. {"result": {"records": [...]}}
                inner = body[key]
                for k2 in ("records", "data", "results"):
                    if isinstance(inner.get(k2), list):
                        return inner[k2]
        raise ValueError(f"Unrecognised response shape: keys={list(body)[:10]}")

    def fetch(self, url: str, date_col: str, since: date, max_rows: int = 300_000) -> pd.DataFrame:
        """Page through a dataset, newest filter first; falls back to local date filtering
        if the API rejects the server-side filter."""
        filters = [f"{date_col}>='{since.isoformat()}'", None]
        for flt in filters:
            try:
                return self._page_all(url, flt, max_rows, date_col, since)
            except httpx.HTTPStatusError as e:
                if flt is None or e.response.status_code not in (400, 422):
                    raise
                log.warning("Dubai Pulse rejected filter %r (%s); retrying without it", flt, e.response.status_code)
        raise RuntimeError("unreachable")

    def _page_all(self, url, flt, max_rows, date_col, since) -> pd.DataFrame:
        rows: list[dict] = []
        offset = 0
        while offset < max_rows:
            params = {"limit": self.page_size, "offset": offset}
            if flt:
                params["filter"] = flt
            r = self.http.get(url, params=params, headers={"Authorization": f"Bearer {self.token()}"})
            r.raise_for_status()
            page = self._rows(r)
            rows.extend(page)
            if len(page) < self.page_size:
                break
            offset += self.page_size
        df = pd.DataFrame(rows)
        if not flt and date_col in df:                 # local filter when the API can't
            d = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            df = df[d >= pd.Timestamp(since)]
        log.info("Dubai Pulse %s: %d rows (filter=%s)", url.rsplit("/", 1)[-1], len(df), bool(flt))
        return df


class LiveDataRefresher:
    """Fetch -> build homes -> swap into the recommender, on a timer. One worker process
    fetches (file lock); the others pick up its cached result."""

    def __init__(self, settings: Settings, recommender):
        self.s = settings
        self.rec = recommender
        self.cache = settings.data_dir / "live_homes.pkl"
        self.meta = settings.data_dir / "live_meta.json"
        self.lock = settings.data_dir / "live.lock"
        self.status: dict = {"enabled": True, "state": "starting", "as_of": None, "homes": None,
                             "last_refresh": None, "error": None}
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        threading.Thread(target=self._run, name="dld-live", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def refresh_once(self) -> None:
        client = DubaiPulseClient(self.s.dubai_pulse_api_key, self.s.dubai_pulse_api_secret)
        since = date.today() - timedelta(days=int(self.s.live_months * 30.5))
        sales = client.fetch(self.s.dubai_pulse_sales_url, "instance_date", since, self.s.live_max_rows)
        rents = None
        if self.s.dubai_pulse_rents_url:
            try:
                rents = client.fetch(self.s.dubai_pulse_rents_url, "contract_start_date", since, self.s.live_max_rows)
            except Exception as e:                      # rents are optional; don't lose the sales refresh
                log.warning("Rent contracts fetch failed, continuing with sales only: %s", e)
        homes, overrides = build(sales, rents, months=self.s.live_months, verbose=False)
        if len(homes) < 100:
            raise RuntimeError(f"Only {len(homes)} homes built from {len(sales)} rows - keeping previous data")
        homes["amenities"] = homes["amenities"].fillna("").map(lambda v: [a for a in str(v).split("|") if a])
        homes.to_pickle(self.cache)
        self.meta.write_text(json.dumps({"overrides": overrides, "fetched_at": time.time(),
                                         "rows": {"sales": len(sales), "rents": 0 if rents is None else len(rents)}}))
        self._apply(homes, overrides)

    # ----------------------------------------------------------------- private
    def _apply(self, homes: pd.DataFrame, overrides: dict) -> None:
        reference.apply_price_overrides(overrides.get("communities", {}))
        self.rec.swap_listings(homes)
        self.status.update(state="live", as_of=overrides.get("as_of"), homes=len(homes),
                           last_refresh=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), error=None)
        log.info("Live DLD data applied: %d homes, as of %s", len(homes), overrides.get("as_of"))

    def _cache_age_h(self) -> float | None:
        if not (self.cache.exists() and self.meta.exists()):
            return None
        return (time.time() - json.loads(self.meta.read_text())["fetched_at"]) / 3600

    def _load_cache(self) -> None:
        self._apply(pd.read_pickle(self.cache), json.loads(self.meta.read_text())["overrides"])

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                age = self._cache_age_h()
                if age is not None and age < self.s.live_refresh_hours:
                    if self.status["state"] != "live":
                        self._load_cache()
                else:
                    self._refresh_with_lock()
            except Exception as e:
                log.exception("Live DLD refresh failed; still serving previous data")
                self.status.update(state="error (serving previous data)", error=str(e)[:300])
            self._stop.wait(600 if self.status["state"] != "live" else 3600)

    def _refresh_with_lock(self) -> None:
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - self.lock.stat().st_mtime > 3600:   # stale lock from a crashed worker
                self.lock.unlink(missing_ok=True)
            self.status["state"] = "waiting for another worker's refresh"
            return
        try:
            self.status["state"] = "refreshing"
            self.refresh_once()
        finally:
            os.close(fd)
            self.lock.unlink(missing_ok=True)
