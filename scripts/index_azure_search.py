"""Publish the current homes to Azure AI Search by hand (the web app also does this automatically
at startup and after every daily refresh when AZURE_SEARCH_ENDPOINT is set).

    python -m scripts.index_azure_search
"""
import numpy as np
import pandas as pd

from app.config import get_settings
from app.recsys.embeddings import get_embedder
from app.recsys.search_index import catalogue_version, publish


def main() -> None:
    s = get_settings()
    assert s.azure_search_endpoint, "Set AZURE_SEARCH_ENDPOINT (and AZURE_SEARCH_API_KEY) in .env"
    listings = pd.read_pickle(s.data_dir / "listings.pkl").reset_index(drop=True)
    vectors = np.load(s.data_dir / "embeddings.npz", allow_pickle=True)["vectors"]
    version = catalogue_version(listings, get_embedder(s).signature)
    print(f"Publishing {len(listings)} homes (dim={vectors.shape[1]}) as version {version} to {s.azure_search_index}")
    print(publish(s, listings, vectors, version))


if __name__ == "__main__":
    main()
