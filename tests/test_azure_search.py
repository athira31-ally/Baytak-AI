"""Azure AI Search retrieval with an in-memory fake of the SDK: versioned publish, hybrid-style
query with OData filters, idempotency, stale-version cleanup and fallback to local retrieval."""
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.recsys.retrieval import AzureSearchRetriever
from app.schemas import UserQuery


class FakeResults(list):
    def __init__(self, items, count):
        super().__init__(items)
        self.count = count

    def get_count(self):
        return self.count


class FakeSearch:
    """Understands the filters we generate: eq/ne/ge/le/lt on fields, search.in, 'and'."""
    def __init__(self):
        self.docs, self.index, self.fail = {}, None, False

    def create_or_update_index(self, index):
        self.index = index

    def merge_or_upload_documents(self, docs):
        for d in docs:
            self.docs[d["doc_id"]] = d
        return [NS(succeeded=True) for _ in docs]

    def delete_documents(self, docs):
        for d in docs:
            self.docs.pop(d["doc_id"], None)

    def _match(self, d, flt):
        for cond in flt.split(" and "):
            cond = cond.strip()
            if m := re.match(r"search\.in\((\w+), '([^']*)', '(.)'\)", cond):
                if str(d[m.group(1)]) not in m.group(2).split(m.group(3)):
                    return False
                continue
            f, op, v = re.match(r"(\w+) (eq|ne|ge|le|lt) (.+)", cond).groups()
            v = v.strip("'")
            x = d.get(f)
            v = (v == "true") if v in ("true", "false") else (float(v) if re.fullmatch(r"[\d.]+", v) else v)
            if x is None:
                return False
            if not {"eq": x == v, "ne": x != v, "ge": x >= v, "le": x <= v, "lt": x < v}[op]:
                return False
        return True

    def search(self, search_text="*", filter=None, top=50, select=None, vector_queries=None, include_total_count=False, **kw):
        if self.fail:
            raise RuntimeError("service unavailable")
        hits = [d for d in self.docs.values() if not filter or self._match(d, filter)]
        if vector_queries:
            q = np.array(vector_queries[0].vector)
            hits.sort(key=lambda d: -float(np.dot(q, d["embedding"])))
        return FakeResults([{**d, "@search.score": 1.0 / (i + 1)} for i, d in enumerate(hits[:top])], len(hits))


@pytest.fixture
def rec_and_fake(client):
    from app.main import state
    rec = state["rec"]
    fake = FakeSearch()
    emb = np.load(rec.s.data_dir / "embeddings.npz", allow_pickle=True)["vectors"]
    rec_s = rec.s
    rec.s = rec.s.model_copy(update={"azure_search_endpoint": "https://fake.search.windows.net"})
    local = rec.retriever
    rec.attach_search(rec.listings, emb, local, index_client=fake, search_client=fake)
    yield rec, fake
    rec.retriever, rec.s, rec.search_status = local, rec_s, {"backend": "local"}


def test_publish_is_versioned_and_idempotent(rec_and_fake):
    rec, fake = rec_and_fake
    assert rec.search_status["backend"] == "azure-ai-search"
    assert len(fake.docs) == len(rec.listings) and fake.index.name == "baytak-homes"
    from app.recsys.search_index import publish
    again = publish(rec.s, rec.listings, np.zeros((len(rec.listings), 96)), rec.retriever.version, fake, fake)
    assert again["status"] == "already published"


def test_hybrid_query_respects_filters(rec_and_fake):
    rec, _ = rec_and_fake
    q = UserQuery(purpose="sale", budget_aed=2_000_000, min_bedrooms=2, property_types=["apartment"])
    cand = rec.retriever.retrieve(q, 50)
    assert rec.retriever.last_backend == "azure-ai-search" and len(cand)
    assert (cand["purpose"] == "sale").all() and (cand["bedrooms"] >= 2).all()
    assert (cand["price_aed"] <= 2_000_000 * 1.3).all() and (cand["property_type"] == "apartment").all()
    r = rec.recommend(q)                       # the full pipeline (ranker etc.) on top of it
    assert r.recommendations


def test_stale_versions_are_removed_after_grace_period(rec_and_fake):
    rec, fake = rec_and_fake
    fake.merge_or_upload_documents([{"doc_id": "old-1", "version": "old", "published_at": 0, "purpose": "sale"}])
    from app.recsys.search_index import publish
    publish(rec.s, rec.listings.head(10), np.zeros((10, 96)), "newversion", fake, fake)
    assert "old-1" not in fake.docs and any(d.startswith("newversion-") for d in fake.docs)


def test_falls_back_to_local_when_search_fails(rec_and_fake):
    rec, fake = rec_and_fake
    fake.fail = True
    cand = rec.retriever.retrieve(UserQuery(purpose="sale", budget_aed=2_000_000), 50)
    assert len(cand) and rec.retriever.last_backend == "local-fallback"


def test_odata_filter_escapes_and_versions():
    r = AzureSearchRetriever.__new__(AzureSearchRetriever)
    r.version = "abc"
    f = r.odata_filter(UserQuery(purpose="rent", budget_aed=100_000, preferred_communities=["Dubai Marina"]))
    assert f.startswith("version eq 'abc' and purpose eq 'rent'") and "search.in(community, 'Dubai Marina', '|')" in f
