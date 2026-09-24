"""Market Data Agent: appends each day's new Dubai Land Department deals and keeps the app fresh.

Runs on a cron (Azure Container Apps Job, daily 07:00 Dubai). It never reloads the whole history.

  Durable memory  Cosmos DB  - `market_tx`: one item per deal (id = DLD transaction id); each item's
                               TTL ends 400 days after the deal date, so old deals age out by
                               themselves and the store stays a rolling window.
                             - `agent_memory`: watermark (newest deal stored), run history,
                               notes for future runs, published homes snapshot, market brief.
  Hot cache       Redis      - homes snapshot + version; the web app polls the version and
                               hot-swaps new homes within minutes (no rebuild, no redeploy).

Tools the LLM plans with (Azure OpenAI function calling):
  recall_memory    watermark, store size, typical daily volume, recent runs, notes
  fetch_new_deals  only deals since watermark - 3 days (late registrations). data.dubai files are
                   read with HTTP Range requests from the newest end: a few MB, not 1.1 GB.
                   With Dubai Pulse API keys, a server-side date filter is used instead.
  validate_batch   schema, no future dates, volume vs. history, prices vs. stored medians
  append_deals     upsert by transaction id + advance watermark   <- refused unless validated
  rebuild_homes    medians over the rolling 12 months in Cosmos -> publish to Cosmos + Redis
  remember         one-line note for future runs
  market_brief     numbers for the daily report
Guardrails are in code: the LLM cannot append or publish unvalidated data, whatever it asks.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import httpx
import pandas as pd

from app.config import Settings
from app.data.dld import BROWSER_UA, build
from app.data.store import get_cache, get_market_store, normalise_batch, publish_homes

log = logging.getLogger(__name__)
WINDOW_MONTHS = 12
DATASET_ID = 470061                     # "Real Estate Transactions" (Dubai Land Department) on data.dubai
DOWNLOAD_API = ("https://data.dubai/o/dda/data-services/dataset-download"
                f"?datasetId={DATASET_ID}&page=1&pageSize=30&sortDir=desc")
DATASET_PAGES = ["https://data.dubai/en/l/470061", "https://data.dubai/e/dataset-details-embed/35681/470061"]
FILE_RE = re.compile(r"""https?://[^"'\s<>]+?transactions_(\d{4}-\d{2}-\d{2})_[\d-]+_(\d{4})\.csv[^"'\s<>]*""")


def download_links(timeout: float = 30) -> dict[str, str]:
    """{file name: fresh signed CSV link} from data.dubai's public download API (no login)."""
    r = httpx.get(DOWNLOAD_API, headers={"User-Agent": BROWSER_UA, "Accept": "application/json"},
                  timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    out = {}
    for m in (r.json().get("data") or {}).get("metadata") or []:
        for f in m.get("files") or []:
            if f.get("file_extension") == "csv" and f.get("file_url"):
                out[f.get("file_name") or m.get("file_folder")] = f["file_url"]
    return out

SYSTEM_PROMPT = """You are the Market Data Agent for Baytak AI, a Dubai property recommender.
Each day you append the new Dubai Land Department deals to the database and refresh the homes.

1. recall_memory - learn the watermark, typical daily volume and notes from earlier runs.
2. fetch_new_deals - it only fetches deals since the watermark.
3. If it returned an "error": call remember with the error, then stop and report "NOT APPENDED" and the error.
   If it returned 0 rows and no error: stop and report "NO NEW DATA" with the watermark date.
4. validate_batch - read every check.
5. Only if it passed: append_deals. If append_deals reports 0 new deals, stop: "NO NEW DATA".
   Otherwise call rebuild_homes.
6. If something is worth knowing next time (e.g. an unusual volume, a failed check, an area
   with many unmatched deals), call remember with one short sentence.
7. Call market_brief, then finish with a report for the engineer:
   first line APPENDED / NOT APPENDED / NO NEW DATA and the newest deal date. Use NO NEW DATA only when
   the fetch succeeded with 0 rows or append_deals reported 0 new deals; any error means NOT APPENDED; then how many new
   deals and MB downloaded; then 3-5 bullets, each "label: value" (e.g. "Deals in store: 233,911"),
   using only numbers from the tools. Never invent numbers or print a bare number without its label."""


@dataclass
class Batch:
    raw: pd.DataFrame
    docs: list[dict]
    info: dict
    validated: bool = False
    checks: list[dict] = field(default_factory=list)


class DataTools:
    def __init__(self, settings: Settings, store=None, cache=None, file_urls: list[str] | None = None):
        self.s = settings
        self.store = store if store is not None else get_market_store(settings)
        self.cache = cache if cache is not None else get_cache(settings)
        self.file_urls = file_urls or [u for u in (settings.dld_file_urls or "").split(",") if u.strip()]
        self.batch: Batch | None = None
        self.append_result: dict | None = None
        self.rebuild_result: dict | None = None
        snap = self.store.get_memory("homes_snapshot")
        self.prev_meta = snap["meta"] if snap else None            # what's live before this run

    # ------------------------------------------------------------------ tools
    def recall_memory(self) -> dict:
        runs = self.store.get_memory("runs", [])
        # daily top-ups only: the one-off 12-month seed load would make "typical" look like 200k+
        vols = [r["new_deals"] for r in runs if r.get("new_deals") and r["new_deals"] <= self.s.max_new_deals_per_run]
        return {"watermark": self.store.get_memory("watermark"), "store": self.store.stats(),
                "typical_new_deals_per_run": int(pd.Series(vols).median()) if vols else None,
                "recent_runs": [{k: r.get(k) for k in ("at", "outcome", "new_deals", "watermark")} for r in runs[-5:]],
                "notes": self.store.get_memory("notes", [])[-10:]}

    def fetch_new_deals(self) -> dict:
        from app.data.incremental import fetch_since_files
        wm = self.store.get_memory("watermark")
        if wm:
            since, mode = pd.Timestamp(wm) - pd.Timedelta(days=self.s.refetch_overlap_days), "incremental"
        else:
            since, mode = pd.Timestamp.today().normalize() - pd.DateOffset(months=WINDOW_MONTHS), "first run: seeding 12 months"
        t0 = time.time()
        if self.s.use_live_data:                          # official API: server-side date filter
            from app.data.live import DubaiPulseClient
            c = DubaiPulseClient(self.s.dubai_pulse_api_key, self.s.dubai_pulse_api_secret)
            raw = c.fetch(self.s.dubai_pulse_sales_url, "instance_date", since.date(), self.s.live_max_rows)
            how = {"source": "Dubai Pulse API", "method": "date filter", "mb_downloaded": None}
        else:
            found = self._find_files()
            if not found["files"]:
                return {"rows": 0, "error": found["error"], "details": found.get("details")}
            raw, stats = fetch_since_files(found["files"], since, resolve=found.get("resolve"))
            how = {"source": found.get("source"), "method": stats.mode,
                   "mb_downloaded": round(stats.bytes_read / 1e6, 1), "http_requests": stats.requests}
        docs = normalise_batch(raw)
        dates = [d["date"] for d in docs]
        info = {"mode": mode, "since": since.date().isoformat(), "rows": len(docs), **how,
                "date_range": [min(dates), max(dates)] if dates else None, "seconds": round(time.time() - t0, 1)}
        self.batch = Batch(raw, docs, info)
        return info

    def validate_batch(self) -> dict:
        b = self.batch
        if b is None:
            return {"passed": False, "checks": [{"check": "batch", "ok": False, "detail": "call fetch_new_deals first"}]}
        checks = []

        def add(name, ok, detail, blocking=True):
            checks.append({"check": name, "ok": bool(ok), "blocking": blocking, "detail": detail})

        add("not empty", len(b.docs) > 0, f"{len(b.docs)} rows")
        if b.docs:
            newest = pd.Timestamp(max(d["date"] for d in b.docs))
            add("no future dates", newest <= pd.Timestamp.today().normalize() + pd.Timedelta(days=1), f"newest {newest.date()}")
            homes = None
            try:
                homes, _ = build(b.raw, None, months=WINDOW_MONTHS, min_deals=1, verbose=False)
                add("schema", True, "required DLD columns present")
            except Exception as e:
                add("schema", False, str(e)[:200])
            wm = self.store.get_memory("watermark")
            if wm:
                add("batch size", len(b.docs) <= self.s.max_new_deals_per_run,
                    f"{len(b.docs)} rows (max {self.s.max_new_deals_per_run}; more suggests a format/duplicate problem)")
                typical = self.recall_memory()["typical_new_deals_per_run"]
                if typical:
                    ratio = len(b.docs) / typical
                    add("volume vs history", 0.05 <= ratio <= 20, f"{len(b.docs)} rows vs typical {typical} ({ratio:.1f}x)",
                        blocking=False)
            if homes is not None and len(homes) and self.prev_meta:
                prev = self.prev_meta.get("communities", {})
                sale = homes[homes["purpose"] == "sale"]
                psf = (sale["price_aed"] / sale["size_sqft"]).groupby(sale["community"]).median()
                limit = self.s.max_price_drift * 2          # a single day's deals are noisier than a 12-month median
                bad = {c: f"{v / prev[c]['price_psf'] - 1:+.0%}" for c, v in psf.items()
                       if c in prev and prev[c]["price_psf"] and abs(v / prev[c]["price_psf"] - 1) > limit}
                add("prices vs stored medians", not bad, f"communities > {limit:.0%} off: {bad or 'none'}")
        b.checks = checks
        b.validated = all(c["ok"] for c in checks if c["blocking"])
        return {"passed": b.validated, "checks": checks}

    def append_deals(self) -> dict:
        b = self.batch
        if b is None or not b.validated:                                 # hard guard
            return {"appended": False, "reason": "validate_batch has not passed on this batch"}
        res = self.store.upsert_transactions(b.docs)
        newest = max(d["date"] for d in b.docs)
        wm = self.store.get_memory("watermark")
        wm = max(newest, wm) if wm else newest
        self.store.set_memory("watermark", wm)
        self.append_result = {"appended": True, "new": res["new"], "already_stored": res["updated"], "watermark": wm}
        return {**self.append_result, "store": self.store.stats()}

    def rebuild_homes(self) -> dict:
        if not self.append_result:                                        # hard guard
            return {"rebuilt": False, "reason": "append_deals has not run in this session"}
        wm = pd.Timestamp(self.append_result["watermark"])
        raw = self.store.load_transactions((wm - pd.DateOffset(months=WINDOW_MONTHS)).date().isoformat())
        homes, meta = build(raw, None, months=WINDOW_MONTHS, verbose=False)
        if len(homes) < self.s.min_homes or homes["community"].nunique() < self.s.min_communities:
            return {"rebuilt": False, "reason": f"only {len(homes)} homes in {homes['community'].nunique()} communities"}
        version = publish_homes(self.store, self.cache, homes, meta)
        self.rebuild_result = {"rebuilt": True, "version": version, "homes": len(homes), "deals_used": len(raw),
                               "data_as_of": meta.get("as_of"), "communities": int(homes["community"].nunique()),
                               "meta": meta}
        return {k: v for k, v in self.rebuild_result.items() if k != "meta"}

    def remember(self, note: str) -> dict:
        notes = self.store.get_memory("notes", [])
        notes.append({"at": time.strftime("%Y-%m-%d"), "note": str(note)[:300]})
        self.store.set_memory("notes", notes[-50:])
        return {"saved": True, "notes": len(notes)}

    def market_brief(self) -> dict:
        out = {"watermark": self.store.get_memory("watermark"), "store": self.store.stats()}
        b = self.batch
        if b is not None and b.docs:
            try:
                h, _ = build(b.raw, None, months=WINDOW_MONTHS, min_deals=1, verbose=False)
                s = h[h["purpose"] == "sale"]
                out["new_deals_by_community"] = s.groupby("community")["n_transactions"].sum().sort_values(ascending=False).head(5).to_dict()
                out["new_deals_median_price_by_bedrooms"] = s.groupby("bedrooms")["price_aed"].median().round(-3).astype(int).to_dict()
            except Exception as e:
                out["batch_note"] = str(e)[:200]
        if self.rebuild_result:
            psf = {k: v["price_psf"] for k, v in self.rebuild_result["meta"].get("communities", {}).items()}
            prev = (self.prev_meta or {}).get("communities", {})
            out["rolling_12m_price_psf_top3"] = dict(sorted(psf.items(), key=lambda kv: -kv[1])[:3])
            moves = {k: v / prev[k]["price_psf"] - 1 for k, v in psf.items() if k in prev and prev[k]["price_psf"]}
            out["price_psf_change_since_last_publish"] = {k: f"{v:+.1%}" for k, v in
                                                          sorted(moves.items(), key=lambda kv: -abs(kv[1]))[:5]}
        return out

    # ---------------------------------------------------------------- helpers
    def _find_files(self) -> dict:
        """Where today's DLD export lives. data.dubai hands out signed links that expire after
        10 minutes, so we return file NAMES plus a resolver that asks for a fresh link right
        before each file is downloaded (the two ~550 MB parts can take longer than 10 min)."""
        if self.file_urls:
            return {"files": self.file_urls, "source": "DLD_FILE_URLS"}
        errors = []
        try:
            links = download_links()
            if links:
                snap = sorted(links)[0].split("_")[1]
                return {"files": sorted(links), "resolve": lambda name: download_links()[name],
                        "source": f"data.dubai API snapshot {snap}"}
            errors.append(f"{DOWNLOAD_API}: no CSV files listed")
        except Exception as e:
            errors.append(f"{DOWNLOAD_API}: {type(e).__name__}: {e}"[:300])
        for page in DATASET_PAGES:                      # fallback: links embedded in the page HTML
            try:
                html = httpx.get(page, headers={"User-Agent": BROWSER_UA}, timeout=30, follow_redirects=True).text
            except Exception as e:
                errors.append(f"{page}: {e}")
                continue
            found: dict = {}
            for m in FILE_RE.finditer(html.replace("\\/", "/")):
                found.setdefault(m.group(1), {})[m.group(2)] = m.group(0)
            if found:
                latest = max(found)
                return {"files": [found[latest][k] for k in sorted(found[latest])], "source": f"data.dubai snapshot {latest}"}
            errors.append(f"{page}: no transaction CSV links found (JavaScript-rendered or blocked)")
        return {"files": [], "error": "Could not get the DLD download links from data.dubai; set DLD_FILE_URLS "
                                      "or DUBAI_PULSE_API_KEY/SECRET.", "details": errors}

    def call(self, name: str, args: dict) -> dict:
        fn = {"recall_memory": self.recall_memory, "fetch_new_deals": self.fetch_new_deals,
              "validate_batch": self.validate_batch, "append_deals": self.append_deals,
              "rebuild_homes": self.rebuild_homes, "remember": self.remember,
              "market_brief": self.market_brief}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:
            log.exception("tool %s failed", name)
            return {"error": f"{type(e).__name__}: {e}"[:500]}


def _spec(name, desc, props=None, required=None):
    params = {"type": "object", "properties": props or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


TOOL_SPECS = [
    _spec("recall_memory", "Watermark, store size, typical volume, recent runs and notes from memory."),
    _spec("fetch_new_deals", "Fetch only the DLD deals since the watermark (small overlap for late registrations)."),
    _spec("validate_batch", "Data-quality checks on the fetched batch."),
    _spec("append_deals", "Upsert the validated batch into Cosmos DB and advance the watermark."),
    _spec("rebuild_homes", "Recompute homes from the rolling 12 months stored and publish them (Cosmos + Redis)."),
    _spec("remember", "Save one short note for future runs.", {"note": {"type": "string"}}, ["note"]),
    _spec("market_brief", "Numbers for the daily report."),
]


class MarketDataAgent:
    mode_name = "azure-openai"

    def __init__(self, settings: Settings, tools: DataTools | None = None, client=None):
        self.s = settings
        self.tools = tools or DataTools(settings)
        if client is False:                       # subclass supplies its own client
            client = None
        elif client is None and settings.use_azure_openai:
            from app.agents.llm import azure_openai_client
            client = azure_openai_client(settings)
        self.client = client
        self.trace: list[dict] = []

    def _call(self, name, args):
        t = time.time()
        out = self.tools.call(name, args)
        self.trace.append({"tool": name, "args": args, "seconds": round(time.time() - t, 1),
                           "result": json.dumps(out, default=str)[:600]})
        return out

    def run(self) -> dict:
        t0 = time.time()
        report, mode = None, "offline"
        if self.client is not None:
            try:
                report, mode = self._llm_run(), self.mode_name
            except Exception as e:
                log.exception("LLM agent failed; falling back to fixed pipeline: %s", e)
        if report is None:
            report = self._offline_run()
        t = self.tools
        app, reb = t.append_result or {}, t.rebuild_result or {}
        outcome = "APPENDED" if reb.get("rebuilt") else ("NO NEW DATA" if (t.batch and not t.batch.docs) or
                                                         (app and not app.get("new")) else "NOT APPENDED")
        run = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode, "outcome": outcome,
               "new_deals": app.get("new", 0), "watermark": t.store.get_memory("watermark"),
               "version": reb.get("version"), "fetch": t.batch.info if t.batch else None,
               "seconds": round(time.time() - t0, 1), "report": report[:2000]}
        runs = t.store.get_memory("runs", [])
        t.store.set_memory("runs", (runs + [{k: v for k, v in run.items() if k != "report"}])[-90:])
        if reb.get("rebuilt"):
            t.store.set_memory("market_brief", {"at": run["at"], "data_as_of": reb.get("data_as_of"), "text": report})
        return {**run, "published": bool(reb.get("rebuilt")), "trace": self.trace}

    def _llm_run(self) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Run today's refresh. Today is {pd.Timestamp.today().date()}."}]
        for _ in range(12):
            kw = dict(model=self.s.azure_openai_chat_deployment, messages=msgs, tools=TOOL_SPECS, tool_choice="auto")
            try:
                resp = self.client.chat.completions.create(reasoning_effort=self.s.llm_reasoning_effort, **kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                resp = self.client.chat.completions.create(**kw)
            m = resp.choices[0].message
            if not m.tool_calls:
                return m.content or ""
            msgs.append({"role": "assistant", "content": m.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in m.tool_calls]})
            for c in m.tool_calls:
                out = self._call(c.function.name, json.loads(c.function.arguments or "{}"))
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out, default=str)[:8000]})
        return "Stopped after too many steps."

    def _offline_run(self) -> str:
        mem = self._call("recall_memory", {})
        got = self._call("fetch_new_deals", {})
        if got.get("error"):
            return f"NOT APPENDED - {got['error']}"
        if not got.get("rows"):
            return f"NO NEW DATA - newest deal stored {mem.get('watermark')}"
        val = self._call("validate_batch", {})
        if not val.get("passed"):
            failed = "; ".join(c["detail"] for c in val["checks"] if not c["ok"] and c["blocking"])
            self._call("remember", {"note": f"Batch rejected: {failed}"[:300]})
            return f"NOT APPENDED - failed checks: {failed}"
        app = self._call("append_deals", {})
        if app.get("error") or not app.get("appended"):
            return (f"NOT APPENDED - writing to the database failed: {app.get('error') or app.get('reason')}. "
                    "Safe to re-run: deals already written are skipped.")
        if not app.get("new"):
            return f"NO NEW DATA - {got['rows']:,} recent deals re-read, all already stored (newest {app.get('watermark')})"
        reb = self._call("rebuild_homes", {})
        brief = self._call("market_brief", {})
        mb = got.get("mb_downloaded")
        lines = [f"{'APPENDED' if reb.get('rebuilt') else 'NOT APPENDED'} - newest deal {app.get('watermark')}",
                 f"- {app['new']:,} new deals appended ({app['already_stored']:,} re-sent deals ignored)"
                 + (f"; {mb} MB downloaded ({got['method']})" if mb is not None else f"; {got['method']}"),
                 f"- Cosmos now holds {app['store']['transactions']:,} deals ({app['store']['from']} to {app['store']['to']})",
                 f"- Homes: {reb.get('homes', 0):,} " + (f"published as {reb['version']}" if reb.get("rebuilt") else f"not rebuilt: {reb.get('reason')}")]
        if brief.get("new_deals_by_community"):
            lines.append("- Busiest in this batch: " + ", ".join(list(brief["new_deals_by_community"])[:3]))
        if brief.get("price_psf_change_since_last_publish"):
            lines.append("- Price/sq ft moves: " + ", ".join(f"{k} {v}" for k, v in
                                                             list(brief["price_psf_change_since_last_publish"].items())[:3]))
        return "\n".join(lines)
