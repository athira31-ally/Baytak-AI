"""Baytak AI API."""
from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import mcp_server, observability
from app.agents.orchestrator import Agent
from app.agents.tools import Toolbox
from app.config import ROOT, get_settings
from app.recsys.pipeline import Recommender, ab_summary, assign_variant
from app.schemas import (ChatRequest, ChatResponse, FeedbackEvent, Listing, Recommendation,
                         RecommendResponse, UserQuery)
from app.storage.feedback import get_store

settings = get_settings()
observability.setup(settings)


def make_agent(rec, s):
    """Home-search agent: LangGraph multi-agent team (default) or the classic single tool loop."""
    if s.agent_engine == "langgraph":
        from app.agents.graph import GraphAgent
        return GraphAgent(rec, s)
    return Agent(rec, s)
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    store = get_store(settings)
    rec = Recommender(settings, store)
    state.update(store=store, rec=rec, agent=make_agent(rec, settings))
    # The daily Market Data Agent publishes homes to Cosmos + Redis; this hot-swaps them in.
    from app.data.watcher import StoreWatcher
    state["live"] = StoreWatcher(settings, rec)
    state["live"].start()
    # MCP server shares the live recommender / agent / market store with the web app
    mcp_server.bind(lambda: state.get("rec"), settings, get_agent=lambda: state.get("agent"),
                    get_store=lambda: state["live"].store if "live" in state else None)
    async with mcp_server.mcp.session_manager.run():
        yield
    if "live" in state:
        state["live"].stop()
    state.clear()


app = FastAPI(title="Baytak AI", version="0.1.0", lifespan=lifespan,
              description="Agentic property recommender for Dubai - Azure OpenAI + AI Search + LightGBM")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
app.mount("/mcp", mcp_server.http_app(settings.mcp_api_key))   # Model Context Protocol (streamable HTTP)


def _log_impressions(session_id: str, variant: str, recs: list[Recommendation]) -> None:
    for r in recs:
        state["store"].log(FeedbackEvent(session_id=session_id, listing_id=r.listing.listing_id, event="impression",
                                         variant=variant, position=r.rank, source=r.source).model_dump())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "listings": len(state["rec"].listings), "ranker_loaded": state["rec"].ranker.ready,
            "llm": settings.llm_label if settings.use_llm else "offline",
            "agent_engine": settings.agent_engine,
            "retrieval": {**state["rec"].search_status,
                          "last_query": getattr(state["rec"].retriever, "last_backend", "local")},
            "store": "cosmos" if settings.use_cosmos else "sqlite", "data": data_status()}


def data_status() -> dict:
    listings = state["rec"].listings
    src = "DLD (real)" if "data_source" in listings and (listings["data_source"] == "DLD").all() else "synthetic"
    out = {"source": src, "homes": len(listings)}
    if "last_transaction_date" in listings:
        out["latest_deal"] = listings["last_transaction_date"].max()
    if "live" in state:
        out["live"] = state["live"].status
    return out


@app.post("/recommend", response_model=RecommendResponse)
def recommend(q: UserQuery, session_id: str | None = None):
    # Redis-cached per (data version, A/B variant, query): repeat searches skip retrieval + ranking
    session_id = session_id or str(uuid.uuid4())
    variant = assign_variant(session_id, settings.ab_ranker_share)
    live = state.get("live")
    key = f"baytak:rec:{live.version if live else 'base'}:{variant}:" + \
        hashlib.sha1(q.model_dump_json().encode()).hexdigest()
    cached = live.cache.get(key) if live else None
    if cached:
        resp = RecommendResponse.model_validate_json(cached).model_copy(update={"session_id": session_id, "latency_ms": 0.0})
    else:
        resp = state["rec"].recommend(q, session_id=session_id, variant=variant)
        if live:
            live.cache.set(key, resp.model_dump_json(), ttl=600)
    _log_impressions(resp.session_id, resp.variant, resp.recommendations)
    observability.events.info("recommend", extra={"custom_dimensions": {
        "variant": resp.variant, "latency_ms": resp.latency_ms, "n": len(resp.recommendations)}})
    return resp


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    resp = state["agent"].chat(req)
    if resp.recommendations:
        variant = assign_variant(resp.session_id, settings.ab_ranker_share)
        _log_impressions(resp.session_id, variant, resp.recommendations)
    observability.events.info("chat", extra={"custom_dimensions": {
        "mode": resp.mode, "grounded": resp.grounded, "tools": [t.tool for t in resp.tool_trace],
        "latency_ms": resp.latency_ms}})
    return resp


@app.post("/feedback")
def feedback(ev: FeedbackEvent):
    rec = state["rec"]
    if rec.get(ev.listing_id) is None:
        raise HTTPException(404, "Unknown listing")
    if ev.variant is None:  # the variant is a pure function of the session, so the client needn't send it
        ev.variant = assign_variant(ev.session_id, settings.ab_ranker_share)
    state["store"].log(ev.model_dump())
    if ev.source == "explore":
        rec.bandit.update(rec.by_id.loc[ev.listing_id, "community"], ev.event)
    return {"ok": True}


@app.get("/listings/{listing_id}", response_model=Listing)
def get_listing(listing_id: str):
    listing = state["rec"].get(listing_id)
    if listing is None:
        raise HTTPException(404, "Unknown listing")
    return listing


@app.get("/listings/{listing_id}/details")
def listing_details(listing_id: str, monthly_income_aed: float | None = None):
    """Everything the 'View' panel shows: the listing, its community, and the money side."""
    listing = state["rec"].get(listing_id)
    if listing is None:
        raise HTTPException(404, "Unknown listing")
    tools = Toolbox(state["rec"], settings)
    out = {"listing": listing.model_dump(), "community": tools.community_profile(listing.community),
           "price_per_sqft": round(listing.price_aed / listing.size_sqft), "data_note": (
               f"Real Dubai Land Department data: median of {listing.n_transactions} registered "
               f"{'sales' if listing.purpose == 'sale' else 'Ejari rent contracts'} for this building and unit type "
               f"(latest {listing.last_transaction_date}). It shows what such homes actually cost - it is not a live advert."
               if listing.data_source == "DLD" else
               "Synthetic demo listing generated from illustrative Dubai market figures - not a real property.")}
    if listing.purpose == "sale":
        out["golden_visa"] = tools.check_golden_visa(listing.price_aed, off_plan=listing.off_plan)
        out["mortgage"] = tools.check_affordability(listing.price_aed, monthly_income_aed or 0,
                                                    off_plan=listing.off_plan)
    return out


@app.get("/market-brief")
def market_brief():
    """Latest daily note written by the Market Data Agent, plus its recent runs (agent memory)."""
    store = state["live"].store
    runs = store.get_memory("runs", [])
    return {"brief": store.get_memory("market_brief"), "watermark": store.get_memory("watermark"),
            "recent_runs": [{k: r.get(k) for k in ("at", "outcome", "new_deals", "watermark", "mode")} for r in runs[-7:]]}


@app.get("/metrics/offline")
def offline_metrics():
    p = settings.data_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/metrics/ab")
def ab_metrics():
    return ab_summary(state["store"].events())


@app.get("/metrics/bandit")
def bandit_state():
    return state["rec"].bandit.posterior()
