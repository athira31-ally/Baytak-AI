"""Retrieve -> rank -> explore -> explain. Also owns A/B assignment."""
from __future__ import annotations

import hashlib
import logging
import time
import uuid

import numpy as np
import pandas as pd

from app.config import Settings
from app.data.reference import COMMUNITY_BY_NAME, commute_minutes, resolve_hub
from app.recsys.bandit import CommunityThompsonBandit
from app.recsys.embeddings import get_embedder
from app.recsys.features import build_features, explain
from app.recsys.ranker import LTRRanker
from app.recsys.retrieval import AzureSearchRetriever, LocalRetriever
from app.schemas import Listing, Recommendation, RecommendResponse, UserQuery

log = logging.getLogger(__name__)
MAX_PER_COMMUNITY = 3


def assign_variant(session_id: str, ranker_share: float) -> str:
    """Deterministic hash bucketing: a session always sees the same variant."""
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "ranker" if bucket < ranker_share else "baseline"


def to_listing(row: pd.Series) -> Listing:
    d = row.to_dict()
    for k in ("annual_rent_aed", "handover_year"):
        v = d.get(k)
        d[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
    d["amenities"] = list(d.get("amenities", []))
    return Listing(**{k: d[k] for k in Listing.model_fields if k in d})


class Recommender:
    def __init__(self, settings: Settings, store):
        self.s = settings
        self.store = store
        self.listings = pd.read_pickle(settings.data_dir / "listings.pkl")
        emb = np.load(settings.data_dir / "embeddings.npz", allow_pickle=True)
        self.embedder = get_embedder(settings)
        if str(emb["embedder"]) != self.embedder.name:
            raise RuntimeError(f"Listing vectors were built with '{emb['embedder']}' but the app is configured for "
                               f"'{self.embedder.name}'. Re-run `python -m scripts.bootstrap`.")
        if settings.use_azure_search:
            self.retriever = AzureSearchRetriever(settings, self.listings, self.embedder)
        else:
            self.retriever = LocalRetriever(self.listings, emb["vectors"], self.embedder)
        self.ranker = LTRRanker(settings.data_dir / "ranker.txt")
        self.bandit = CommunityThompsonBandit()
        self.by_id = self.listings.set_index("listing_id", drop=False)
        self.refresh_bandit()

    def refresh_bandit(self) -> None:
        try:
            self.bandit.rebuild(self.store.events(), dict(zip(self.listings.listing_id, self.listings.community)))
        except Exception as e:  # never block serving on analytics
            log.warning("Bandit rebuild failed: %s", e)

    def get(self, listing_id: str) -> Listing | None:
        return to_listing(self.by_id.loc[listing_id]) if listing_id in self.by_id.index else None

    def recommend(self, q: UserQuery, session_id: str | None = None, variant: str | None = None,
                  k: int | None = None) -> RecommendResponse:
        t0 = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        variant = variant or assign_variant(session_id, self.s.ab_ranker_share)
        k = k or self.s.top_k

        cand = self.retriever.retrieve(q, self.s.candidate_pool_size)
        recs: list[Recommendation] = []
        if not cand.empty:
            X = build_features(q, cand)
            if q.max_commute_min and resolve_hub(q.work_location):
                keep = (X["commute_min"] <= q.max_commute_min * 1.15).to_numpy()
                if keep.sum() >= k:              # soft constraint: only apply if the page stays full
                    cand, X = cand[keep], X[keep]
            scores = self.ranker.score(X) if variant == "ranker" else X["retrieval_score"].to_numpy()
            order = np.argsort(-scores)
            # Diversity re-rank: at most MAX_PER_COMMUNITY homes from one community on a page
            per_comm: dict[str, int] = {}
            diverse, overflow = [], []
            for i in order:
                c = cand.iloc[i]["community"]
                (diverse if per_comm.get(c, 0) < MAX_PER_COMMUNITY else overflow).append(i)
                per_comm[c] = per_comm.get(c, 0) + 1
            order = np.array(diverse + overflow)
            top_idx = list(order[:k])
            # Exploration slot: swap one position for a bandit pick from the rest of the pool
            explore_pos = None
            if len(order) > k and self.s.exploration_slot < k:
                rest = cand.iloc[order[k:k + 60]]
                pick = self.bandit.pick(rest)
                if pick is not None:
                    pick_idx = int(np.flatnonzero(cand.index == pick.name)[0])
                    top_idx[self.s.exploration_slot] = pick_idx
                    explore_pos = self.s.exploration_slot
            hub = resolve_hub(q.work_location)
            for rank, i in enumerate(top_idx):
                row, feats = cand.iloc[i], X.iloc[i]
                c = COMMUNITY_BY_NAME[row["community"].lower()]
                recs.append(Recommendation(
                    listing=to_listing(row), score=float(scores[i]), rank=rank + 1,
                    reasons=explain(q, row, feats),
                    commute_min=commute_minutes(c, hub) if hub else None,
                    golden_visa_eligible=bool(row["price_aed"] >= self.s.golden_visa_threshold_aed) if q.purpose == "sale" else None,
                    source="explore" if rank == explore_pos else variant,
                ))
        return RecommendResponse(session_id=session_id, variant=variant, query=q, recommendations=recs,
                                 candidates_considered=len(cand),
                                 latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def ab_summary(events: list[dict], samples: int = 20_000, seed: int = 0) -> dict:
    """Bayesian A/B read-out: CTR per variant with Beta posteriors and P(ranker > baseline)."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for v in ("ranker", "baseline"):
        imp = sum(1 for e in events if e.get("variant") == v and e["event"] == "impression")
        pos = sum(1 for e in events if e.get("variant") == v and e["event"] in {"click", "save", "contact"})
        out[v] = {"impressions": imp, "positives": pos, "ctr": pos / imp if imp else None,
                  "_draws": rng.beta(pos + 1, max(imp - pos, 0) + 1, samples)}
    p = float((out["ranker"]["_draws"] > out["baseline"]["_draws"]).mean())
    lift = out["ranker"]["_draws"] / out["baseline"]["_draws"] - 1
    for v in out.values():
        v.pop("_draws")
    out["p_ranker_better"] = round(p, 4)
    out["expected_lift"] = round(float(np.mean(lift)), 4)
    out["lift_95ci"] = [round(float(x), 4) for x in np.percentile(lift, [2.5, 97.5])]
    out["decision"] = ("ship ranker" if p > 0.95 else "keep baseline" if p < 0.05 else "keep collecting data")
    return out
