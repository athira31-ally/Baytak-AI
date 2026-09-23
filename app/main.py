"""Baytak AI API."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import observability
from app.agents.orchestrator import Agent
from app.config import ROOT, get_settings
from app.recsys.pipeline import Recommender, ab_summary, assign_variant
from app.schemas import (ChatRequest, ChatResponse, FeedbackEvent, Listing, Recommendation,
                         RecommendResponse, UserQuery)
from app.storage.feedback import get_store

settings = get_settings()
observability.setup(settings)
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    store = get_store(settings)
    rec = Recommender(settings, store)
    state.update(store=store, rec=rec, agent=Agent(rec, settings))
    yield
    state.clear()


app = FastAPI(title="Baytak AI", version="0.1.0", lifespan=lifespan,
              description="Agentic property recommender for Dubai - Azure OpenAI + AI Search + LightGBM")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


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
            "llm": "azure-openai" if settings.use_azure_openai else "offline",
            "retrieval": "azure-ai-search" if settings.use_azure_search else "local",
            "store": "cosmos" if settings.use_cosmos else "sqlite"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(q: UserQuery, session_id: str | None = None):
    resp = state["rec"].recommend(q, session_id=session_id)
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
