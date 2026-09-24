"""Azure AI Search index for candidate retrieval: create it, publish homes into it, version them.

Each published catalogue gets a version (hash of the homes + the embedder), stored on every document.
The retriever filters on its own version, so a new catalogue is uploaded next to the old one and each
web worker switches when it is ready - no half-updated index is ever queried. Old versions are removed
an hour later. Publishing the same version twice is a no-op, so any number of workers can call it.

Index fields: filterable facts (purpose, price, rent, bedrooms, type, community, off-plan) for OData
filters, a searchable description for BM25, and the description vector for HNSW vector search.
Queries are hybrid (BM25 + vector, fused with RRF); the semantic ranker is optional (limited on Free)."""
from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
import pandas as pd

from app.config import Settings

log = logging.getLogger(__name__)
BATCH = 500
KEEP_OLD_SECONDS = 3600


def _cred(settings: Settings):
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential
    return AzureKeyCredential(settings.azure_search_api_key) if settings.azure_search_api_key else DefaultAzureCredential()


def clients(settings: Settings):
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    cred = _cred(settings)
    return (SearchIndexClient(settings.azure_search_endpoint, cred),
            SearchClient(settings.azure_search_endpoint, settings.azure_search_index, cred))


def catalogue_version(listings: pd.DataFrame, embedder_id: str) -> str:
    cols = [c for c in ("listing_id", "price_aed", "annual_rent_aed", "bedrooms", "description") if c in listings]
    h = hashlib.sha1(pd.util.hash_pandas_object(listings[cols], index=False).values.tobytes())
    h.update(embedder_id.encode())
    return h.hexdigest()[:12]


def index_definition(name: str, dim: int):
    from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration, SearchableField, SearchField,
                                                       SearchFieldDataType as F, SearchIndex, SemanticConfiguration,
                                                       SemanticField, SemanticPrioritizedFields, SemanticSearch,
                                                       SimpleField, VectorSearch, VectorSearchProfile)
    fields = [
        SimpleField(name="doc_id", type=F.String, key=True),
        SimpleField(name="version", type=F.String, filterable=True),
        SimpleField(name="published_at", type=F.Int64, filterable=True),
        SimpleField(name="listing_id", type=F.String, filterable=True),
        SimpleField(name="community", type=F.String, filterable=True, facetable=True),
        SimpleField(name="purpose", type=F.String, filterable=True),
        SimpleField(name="property_type", type=F.String, filterable=True, facetable=True),
        SimpleField(name="bedrooms", type=F.Int32, filterable=True, sortable=True),
        SimpleField(name="size_sqft", type=F.Int32, filterable=True),
        SimpleField(name="price_aed", type=F.Double, filterable=True, sortable=True),
        SimpleField(name="annual_rent_aed", type=F.Double, filterable=True, sortable=True),
        SimpleField(name="off_plan", type=F.Boolean, filterable=True),
        SearchableField(name="description", type=F.String, analyzer_name="en.microsoft"),
        SearchField(name="embedding", type=F.Collection(F.Single), searchable=True,
                    vector_search_dimensions=dim, vector_search_profile_name="hnsw-profile"),
    ]
    return SearchIndex(
        name=name, fields=fields,
        vector_search=VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                                   profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")]),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name="homes", prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="description")]))]))


def _docs(listings: pd.DataFrame, vectors: np.ndarray, version: str, now: int):
    for (_, r), v in zip(listings.iterrows(), vectors):
        rent = r.get("annual_rent_aed")
        yield {"doc_id": f"{version}-{r.listing_id}", "version": version, "published_at": now,
               "listing_id": r.listing_id, "community": r.community, "purpose": r.purpose,
               "property_type": r.property_type, "bedrooms": int(r.bedrooms), "size_sqft": int(r.size_sqft),
               "price_aed": float(r.price_aed), "annual_rent_aed": None if rent is None or pd.isna(rent) else float(rent),
               "off_plan": bool(r.off_plan), "description": str(r.description), "embedding": [float(x) for x in v]}


def _create_index(index_client, definition, attempts: int = 6) -> None:
    """Several web workers start at once and all try to create the index; the service rejects the
    concurrent ones ("created ... concurrently"). Back off with jitter and try again."""
    import random
    for i in range(attempts):
        try:
            index_client.create_or_update_index(definition)
            return
        except Exception as e:
            if "concurrent" not in str(e).lower() or i == attempts - 1:
                raise
            time.sleep(1.5 * (i + 1) + random.random() * 2)


def publish(settings: Settings, listings: pd.DataFrame, vectors: np.ndarray, version: str, index_client=None,
            search_client=None) -> dict:
    """Upload a catalogue version (idempotent) and clean up versions older than an hour."""
    if index_client is None or search_client is None:
        index_client, search_client = clients(settings)
    t0 = time.time()
    _create_index(index_client, index_definition(settings.azure_search_index, int(vectors.shape[1])))
    existing = search_client.search(search_text="*", filter=f"version eq '{version}'", include_total_count=True, top=0)
    have = existing.get_count() or 0
    if have >= len(listings):
        return {"version": version, "uploaded": 0, "status": "already published"}
    now = int(time.time())
    docs, uploaded = list(_docs(listings, vectors, version, now)), 0
    for i in range(0, len(docs), BATCH):
        res = search_client.merge_or_upload_documents(docs[i:i + BATCH])
        uploaded += sum(1 for x in res if getattr(x, "succeeded", True))
    stale = [d["doc_id"] for d in search_client.search(
        search_text="*", filter=f"version ne '{version}' and published_at lt {now - KEEP_OLD_SECONDS}",
        select=["doc_id"], top=100_000)]
    for i in range(0, len(stale), BATCH):
        search_client.delete_documents([{"doc_id": d} for d in stale[i:i + BATCH]])
    log.info("Azure AI Search: published %s (%d docs, removed %d stale) in %.1fs", version, uploaded, len(stale), time.time() - t0)
    return {"version": version, "uploaded": uploaded, "removed": len(stale), "seconds": round(time.time() - t0, 1)}
