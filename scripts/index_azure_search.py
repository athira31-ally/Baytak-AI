"""Create the Azure AI Search index and upload listings + vectors.

    python -m scripts.index_azure_search

Requires AZURE_SEARCH_ENDPOINT (+ key or Entra ID) and artifacts from
scripts.bootstrap. Build embeddings with Azure OpenAI first (set
AZURE_OPENAI_ENDPOINT before bootstrap) so query and document vectors match.
"""
import numpy as np
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration, SearchableField, SearchField,
                                                   SearchFieldDataType, SearchIndex, SimpleField, VectorSearch,
                                                   VectorSearchProfile)

from app.config import get_settings


def main() -> None:
    s = get_settings()
    assert s.azure_search_endpoint, "Set AZURE_SEARCH_ENDPOINT in .env"
    cred = AzureKeyCredential(s.azure_search_api_key) if s.azure_search_api_key else DefaultAzureCredential()
    listings = pd.read_pickle(s.data_dir / "listings.pkl")
    emb = np.load(s.data_dir / "embeddings.npz", allow_pickle=True)
    vectors = emb["vectors"]
    print(f"Indexing {len(listings)} listings, dim={vectors.shape[1]}, embedder={emb['embedder']}")

    F = SearchFieldDataType
    fields = [
        SimpleField(name="listing_id", type=F.String, key=True, filterable=True),
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
                    vector_search_dimensions=int(vectors.shape[1]), vector_search_profile_name="hnsw-profile"),
    ]
    index = SearchIndex(name=s.azure_search_index, fields=fields, vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")]))
    SearchIndexClient(s.azure_search_endpoint, cred).create_or_update_index(index)

    client = SearchClient(s.azure_search_endpoint, s.azure_search_index, cred)
    docs = []
    for (_, r), v in zip(listings.iterrows(), vectors):
        docs.append({
            "listing_id": r.listing_id, "community": r.community, "purpose": r.purpose,
            "property_type": r.property_type, "bedrooms": int(r.bedrooms), "size_sqft": int(r.size_sqft),
            "price_aed": float(r.price_aed),
            "annual_rent_aed": None if pd.isna(r.annual_rent_aed) else float(r.annual_rent_aed),
            "off_plan": bool(r.off_plan), "description": r.description, "embedding": v.tolist(),
        })
    for i in range(0, len(docs), 500):
        res = client.upload_documents(docs[i:i + 500])
        print(f"  uploaded {i + len(res)} / {len(docs)}")
    print("Done. Set AZURE_SEARCH_ENDPOINT in the app to switch retrieval to Azure AI Search.")


if __name__ == "__main__":
    main()
