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
    azure_openai_chat_model: str = "gpt-5-mini"        # the model behind that deployment (reasoning or not)
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    llm_reasoning_effort: str = "low"   # for reasoning models: minimal | low | medium | high
    llm_fast_reasoning_effort: str = "minimal"   # LangGraph supervisor/specialists: short structured steps
    # "local" (TF-IDF+SVD, free, baked into the image) or "azure" (needed for Azure AI Search).
    # Listing vectors are built at bootstrap time, so changing this means re-running bootstrap.
    embedding_provider: str = "local"

    # --- Which LLM powers the agents ---
    # "azure"             Azure OpenAI / Foundry (default when AZURE_OPENAI_ENDPOINT is set)
    # "openai_compatible" any OpenAI-compatible server: Ollama, vLLM, NVIDIA NIM, LM Studio... (on-prem / open-source)
    llm_provider: str = "azure"
    llm_base_url: str | None = None     # e.g. http://localhost:11434/v1 (Ollama) or http://vllm:8000/v1
    llm_model: str = "qwen2.5:7b-instruct"   # model name on that server
    llm_api_key: str = "not-needed"     # Ollama/vLLM ignore it unless you configure one
    # Home-search agent engine: "langgraph" (supervisor + specialist agents) or "classic" (single tool loop)
    agent_engine: str = "langgraph"

    # --- Safety: Azure AI Content Safety Prompt Shields (the Foundry/AI Services resource includes it) ---
    content_safety_endpoint: str | None = None   # https://<resource>.cognitiveservices.azure.com
    content_safety_key: str | None = None        # empty = Entra ID

    # --- Azure AI Search (candidate retrieval) ---
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "baytak-homes"
    azure_search_semantic: bool = False   # semantic ranker on top of hybrid (limited queries on the Free tier)

    # --- Azure Cosmos DB (feedback / bandit state) ---
    cosmos_endpoint: str | None = None
    cosmos_key: str | None = None
    cosmos_database: str = "homematch"
    cosmos_container: str = "feedback"

    # --- Live Dubai Land Department data (Dubai Pulse API) ---
    dubai_pulse_api_key: str | None = None
    dubai_pulse_api_secret: str | None = None
    dubai_pulse_sales_url: str = "https://api.dubaipulse.gov.ae/open/dld/dld_transactions-open-api"
    dubai_pulse_rents_url: str | None = "https://api.dubaipulse.gov.ae/open/dld/dld_rent_contracts-open-api"
    live_refresh_hours: float = 24      # DLD publishes daily
    live_months: int = 12
    live_max_rows: int = 300_000
    # --- Market Data Agent (daily refresh from data.dubai files) ---
    dld_file_urls: str | None = None       # comma-separated; leave empty to auto-discover on data.dubai
    max_data_age_days: int = 10            # validation: latest deal must be this recent
    min_homes: int = 500                   # validation: minimum homes built
    min_communities: int = 12              # validation: of the 21 supported communities
    max_price_drift: float = 0.25          # validation: max day-on-day change in community AED/sq ft
    max_new_deals_per_run: int = 30_000    # validation: a normal day is a few hundred to a few thousand
    dld_rent_file_urls: str | None = None  # comma-separated Ejari rent CSVs; empty = data.dubai download API
    rent_refresh_days: int = 7             # rents are a rolling 12-month aggregate over ~5 GB: refresh weekly
    min_rent_rows: int = 200               # validation: minimum rent homes (building x type x bedrooms)
    rent_max_minutes: float = 40           # stop a slow rent scan cleanly before the job's 60-min timeout
    refetch_overlap_days: int = 7          # re-read the last week: DLD publishes ~4 days behind and registers some deals late
    # Where the rolling deal store + published homes live (Azure Blob). Empty = local folder.

    # --- Microsoft Foundry Agent Service (Market Data Agent lives in Foundry when set) ---
    foundry_project_endpoint: str | None = None   # https://<res>.services.ai.azure.com/api/projects/<project>
    foundry_model_deployment: str = "gpt-5-mini"
    foundry_agent_name: str = "baytak-market-data-agent"

    # --- Hot cache (Azure Cache for Redis). Empty = in-process cache ---
    redis_url: str | None = None            # e.g. rediss://:<key>@<name>.redis.cache.windows.net:6380/0
    homes_sync_seconds: int = 300           # how often the app checks for a new homes version

    # --- MCP server (/mcp). Empty = open (read-only tools, like /chat) ---
    mcp_api_key: str | None = None

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
        return self.llm_provider == "azure" and bool(self.azure_openai_endpoint)

    @property
    def use_llm(self) -> bool:
        """True when any chat model is configured (Azure or an OpenAI-compatible server)."""
        return self.use_azure_openai or (self.llm_provider == "openai_compatible" and bool(self.llm_base_url))

    @property
    def chat_model_name(self) -> str:
        return self.azure_openai_chat_deployment if self.llm_provider == "azure" else self.llm_model

    @property
    def llm_label(self) -> str:
        return "azure-openai" if self.llm_provider == "azure" else f"open-source:{self.llm_model}"

    @property
    def use_azure_search(self) -> bool:
        return bool(self.azure_search_endpoint)

    @property
    def use_live_data(self) -> bool:
        return bool(self.dubai_pulse_api_key and self.dubai_pulse_api_secret)

    @property
    def use_cosmos(self) -> bool:
        return bool(self.cosmos_endpoint)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
