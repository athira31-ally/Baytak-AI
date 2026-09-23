"""Central configuration. Every Azure service is optional: if its env vars are
missing, the app falls back to a local implementation so you can develop
offline and switch to Azure by filling in .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- Paths ---
    data_dir: Path = ROOT / "artifacts"

    # --- Azure OpenAI (agents + embeddings) ---
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None          # leave empty to use Managed Identity
    azure_openai_api_version: str = "2025-04-01-preview"   # needed for newer (incl. reasoning) models
    azure_openai_chat_deployment: str = "chat"         # deployment NAME, not model name
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    # "local" (TF-IDF+SVD, free, baked into the image) or "azure" (needed for Azure AI Search).
    # Listing vectors are built at bootstrap time, so changing this means re-running bootstrap.
    embedding_provider: str = "local"

    # --- Azure AI Search (candidate retrieval) ---
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "dubai-listings"

    # --- Azure Cosmos DB (feedback / bandit state) ---
    cosmos_endpoint: str | None = None
    cosmos_key: str | None = None
    cosmos_database: str = "homematch"
    cosmos_container: str = "feedback"

    # --- Observability ---
    applicationinsights_connection_string: str | None = None

    # --- Recommender knobs ---
    candidate_pool_size: int = 200
    top_k: int = 10
    exploration_slot: int = 4            # 0-based position reserved for a bandit pick
    ab_ranker_share: float = 0.8         # share of sessions that get the LightGBM ranker (80/20 rollout)

    # --- UAE finance assumptions (verify against current Central Bank rules) ---
    mortgage_rate: float = 0.045
    mortgage_years: int = 25
    max_dbr: float = 0.50                # debt-burden ratio cap
    golden_visa_threshold_aed: int = 2_000_000
    dld_transfer_fee: float = 0.04

    @property
    def use_azure_openai(self) -> bool:
        return bool(self.azure_openai_endpoint)

    @property
    def use_azure_search(self) -> bool:
        return bool(self.azure_search_endpoint)

    @property
    def use_cosmos(self) -> bool:
        return bool(self.cosmos_endpoint)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
