"""Stage 1 - candidate generation.

Hard filters (purpose, budget band, bedrooms, type, community) + semantic
similarity between the user's free text and listing descriptions.
LocalRetriever works in-memory; AzureSearchRetriever pushes the same logic
into Azure AI Search (OData filter + vector query)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.config import Settings
from app.data.reference import COMMUNITY_BY_NAME, resolve_community
from app.schemas import UserQuery

log = logging.getLogger(__name__)


def query_text(q: UserQuery) -> str:
    parts = [q.free_text, " ".join(q.lifestyle_tags), " ".join(q.property_types)]
    if q.family_with_kids:
        parts.append("family schools kids play area green")
    if q.min_bedrooms:
        parts.append(f"{q.min_bedrooms}-bedroom")
    return " ".join(p for p in parts if p).strip() or "home in Dubai"


def price_col(q: UserQuery) -> str:
    return "price_aed" if q.purpose == "sale" else "annual_rent_aed"


def tag_overlap(q: UserQuery, community: str) -> int:
    c = COMMUNITY_BY_NAME.get(community.lower())
    return len(set(q.lifestyle_tags) & set(c.tags)) if c else 0


class LocalRetriever:
    def __init__(self, listings: pd.DataFrame, vectors: np.ndarray, embedder):
        self.df = listings.reset_index(drop=True)
        self.vectors = vectors
        self.embedder = embedder

    def _mask(self, q: UserQuery, budget_hi: float, use_communities: bool) -> np.ndarray:
        df = self.df
        m = (df["purpose"] == q.purpose).to_numpy().copy()
        if q.budget_aed:
            p = df[price_col(q)].fillna(np.inf).to_numpy()
            m = m & (p <= q.budget_aed * budget_hi) & (p >= q.budget_aed * 0.35)
        m = m & (df["bedrooms"] >= q.min_bedrooms).to_numpy()
        if q.min_bedrooms > 0:
            m = m & (df["bedrooms"] <= q.min_bedrooms + 2).to_numpy()
        if q.property_types:
            m = m & df["property_type"].isin(q.property_types).to_numpy()
        if use_communities and q.preferred_communities:
            names = [c.name for c in map(resolve_community, q.preferred_communities) if c]
            if names:
                m = m & df["community"].isin(names).to_numpy()
        if not q.off_plan_ok:
            m = m & ~df["off_plan"].to_numpy(dtype=bool)
        return m

    def retrieve(self, q: UserQuery, k: int = 200) -> pd.DataFrame:
        # Progressive relaxation so we never return an empty page
        for budget_hi, use_comm in [(1.10, True), (1.10, False), (1.30, False)]:
            mask = self._mask(q, budget_hi, use_comm)
            if mask.sum() >= 20:
                break
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return self.df.iloc[0:0].assign(retrieval_score=[])
        qv = self.embedder.embed([query_text(q)])[0]
        sim = self.vectors[idx] @ qv
        cand = self.df.iloc[idx].copy()
        cand["semantic_sim"] = sim
        cand["tag_overlap"] = [tag_overlap(q, c) for c in cand["community"]]
        cand["retrieval_score"] = cand["semantic_sim"] + 0.15 * cand["tag_overlap"]
        return cand.nlargest(k, "retrieval_score")


class AzureSearchRetriever:
    """Same contract as LocalRetriever, backed by Azure AI Search: OData filters + hybrid query
    (BM25 on the description + HNSW vector search, fused with RRF), optionally the semantic ranker.
    Queries only its own catalogue `version` (see search_index.py). Any error or empty result falls
    back to the in-memory retriever, so search never goes down with the service."""

    def __init__(self, settings: Settings, listings: pd.DataFrame, embedder, version: str,
                 fallback: LocalRetriever | None = None, client=None):
        if client is None:
            from app.recsys.search_index import clients
            client = clients(settings)[1]
        self.client, self.s, self.version = client, settings, version
        self.df = listings.set_index("listing_id", drop=False)
        self.embedder, self.fallback = embedder, fallback
        self.last_backend = "azure-ai-search"

    def odata_filter(self, q: UserQuery, budget_hi: float = 1.10, use_communities: bool = True) -> str:
        f = [f"version eq '{self.version}'", f"purpose eq '{q.purpose}'", f"bedrooms ge {q.min_bedrooms}"]
        if q.min_bedrooms > 0:
            f.append(f"bedrooms le {q.min_bedrooms + 2}")
        if q.budget_aed:
            f.append(f"{price_col(q)} le {q.budget_aed * budget_hi:.0f} and {price_col(q)} ge {q.budget_aed * 0.35:.0f}")
        if q.property_types:
            f.append("search.in(property_type, '" + ",".join(q.property_types) + "', ',')")
        if use_communities and q.preferred_communities:
            names = [c.name.replace("'", "''") for c in map(resolve_community, q.preferred_communities) if c]
            if names:
                f.append("search.in(community, '" + "|".join(names) + "', '|')")
        if not q.off_plan_ok:
            f.append("off_plan eq false")
        return " and ".join(f)

    def _search(self, q: UserQuery, k: int, flt: str, qv: list[float]):
        from azure.search.documents.models import VectorizedQuery
        kw = dict(search_text=query_text(q), filter=flt, top=k, select=["listing_id"],
                  vector_queries=[VectorizedQuery(vector=qv, k_nearest_neighbors=k, fields="embedding")])
        if self.s.azure_search_semantic:
            kw.update(query_type="semantic", semantic_configuration_name="homes")
        return [(r["listing_id"], r["@search.score"]) for r in self.client.search(**kw)]

    def retrieve(self, q: UserQuery, k: int = 200) -> pd.DataFrame:
        try:
            qv = self.embedder.embed([query_text(q)])[0].tolist()
            hits = []
            for budget_hi, use_comm in [(1.10, True), (1.10, False), (1.30, False)]:   # same relaxation as local
                hits = self._search(q, k, self.odata_filter(q, budget_hi, use_comm), qv)
                if len(hits) >= 20:
                    break
            hits = [h for h in hits if h[0] in self.df.index]
            if not hits:
                raise LookupError("no hits")
        except Exception as e:
            if self.fallback is None:
                raise
            log.warning("Azure AI Search retrieval failed (%s); using in-memory retrieval", e)
            self.last_backend = "local-fallback"
            return self.fallback.retrieve(q, k)
        self.last_backend = "azure-ai-search"
        cand = self.df.loc[[h[0] for h in hits]].copy()
        scores = np.array([h[1] for h in hits], dtype=float)
        cand["semantic_sim"] = scores / (scores.max() + 1e-9)   # RRF scores -> [0,1]
        cand["tag_overlap"] = [tag_overlap(q, c) for c in cand["community"]]
        cand["retrieval_score"] = cand["semantic_sim"] + 0.15 * cand["tag_overlap"]
        return cand.sort_values("retrieval_score", ascending=False)
