from typing import Literal

from pydantic import BaseModel, Field

PropertyType = Literal["apartment", "villa", "townhouse"]
Purpose = Literal["sale", "rent"]


class UserQuery(BaseModel):
    """Structured search intent. The agent fills this from free text."""
    purpose: Purpose = "sale"
    budget_aed: float | None = Field(None, description="Total price for sale, or annual rent for rent")
    min_bedrooms: int = 0
    property_types: list[PropertyType] = []
    work_location: str | None = Field(None, description="Work hub, e.g. DIFC, Internet City")
    max_commute_min: float | None = None
    family_with_kids: bool = False
    wants_golden_visa: bool = False
    lifestyle_tags: list[str] = Field([], description="e.g. waterfront, family, metro, quiet, golf")
    preferred_communities: list[str] = []
    off_plan_ok: bool = True
    free_text: str = ""


class Listing(BaseModel):
    listing_id: str
    community: str
    purpose: Purpose
    property_type: PropertyType
    bedrooms: int
    size_sqft: int
    price_aed: int
    annual_rent_aed: int | None = None
    off_plan: bool = False
    handover_year: int | None = None
    furnished: bool = False
    amenities: list[str] = []
    days_listed: int = 0          # synthetic: days on market; DLD: days since the latest real deal
    description: str = ""
    # Present when the home comes from real Dubai Land Department data
    data_source: str = "synthetic"
    building: str | None = None
    project: str | None = None
    n_transactions: int | None = None
    last_transaction_date: str | None = None


class Recommendation(BaseModel):
    listing: Listing
    score: float
    rank: int
    reasons: list[str]
    commute_min: float | None = None
    golden_visa_eligible: bool | None = None
    source: Literal["ranker", "baseline", "explore"] = "ranker"


class RecommendResponse(BaseModel):
    session_id: str
    variant: Literal["ranker", "baseline"]
    query: UserQuery
    recommendations: list[Recommendation]
    candidates_considered: int
    latency_ms: float


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    monthly_income_aed: float | None = None


class ToolCallTrace(BaseModel):
    tool: str
    arguments: dict
    result_summary: str
    latency_ms: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    query: UserQuery | None
    recommendations: list[Recommendation]
    tool_trace: list[ToolCallTrace]
    grounded: bool
    ungrounded_ids: list[str] = []
    mode: Literal["azure-openai", "offline"]
    latency_ms: float


class FeedbackEvent(BaseModel):
    session_id: str
    listing_id: str
    event: Literal["impression", "click", "save", "dismiss", "contact"]
    variant: Literal["ranker", "baseline"] | None = None
    position: int | None = None
    source: str | None = None
