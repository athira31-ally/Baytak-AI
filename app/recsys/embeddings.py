"""Text embeddings: Azure OpenAI when configured, local TF-IDF+SVD otherwise.
Both return L2-normalised float32 vectors so cosine similarity = dot product."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from app.config import Settings

log = logging.getLogger(__name__)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class LocalEmbedder:
    name = "local-tfidf-svd"

    def __init__(self, path: Path):
        self.path = path
        self.model = joblib.load(path) if path.exists() else None

    def fit(self, texts: list[str], dim: int = 96) -> None:
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                   TruncatedSVD(n_components=dim, random_state=0))
        self.model.fit(texts)
        joblib.dump(self.model, self.path)

    @property
    def signature(self) -> str:
        """Identifies the exact fitted model, so index vectors and query vectors always match."""
        import hashlib
        return self.name + ":" + (hashlib.md5(self.path.read_bytes()).hexdigest()[:10] if self.path.exists() else "unfitted")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Local embedder not fitted. Run `python -m scripts.bootstrap`.")
        return _normalise(self.model.transform(texts))


class AzureOpenAIEmbedder:
    name = "azure-openai"

    def __init__(self, settings: Settings):
        from app.agents.llm import azure_openai_client
        self.client = azure_openai_client(settings)
        self.deployment = settings.azure_openai_embedding_deployment

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.deployment}"

    def fit(self, texts: list[str], dim: int = 0) -> None:  # nothing to fit
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            resp = self.client.embeddings.create(model=self.deployment, input=texts[i:i + 100])
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.array(out))


def get_embedder(settings: Settings):
    if settings.embedding_provider == "azure":
        if not settings.use_azure_openai:
            raise RuntimeError("EMBEDDING_PROVIDER=azure needs AZURE_OPENAI_ENDPOINT")
        log.info("Using Azure OpenAI embeddings (%s)", settings.azure_openai_embedding_deployment)
        return AzureOpenAIEmbedder(settings)
    return LocalEmbedder(settings.data_dir / "local_embedder.joblib")
