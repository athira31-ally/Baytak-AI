"""Baytak AI as an MCP (Model Context Protocol) server.

Any MCP client - Claude Desktop, Claude Code, VS Code / GitHub Copilot agent mode, Cursor, or your own
agent - can use Baytak's tools directly:

  search_homes         ranked Dubai homes for a brief (retrieval + LightGBM LambdaRank)
  check_affordability  UAE mortgage rules: LTV cap, installment, debt-burden ratio, upfront costs
  check_golden_visa    10-year Golden Visa property threshold
  estimate_commute     peak / off-peak drive time from a community to a work hub
  community_profile    price per sq ft (real DLD median), yield, metro, schools
  market_brief         today's note from the Market Data Agent (real DLD deals)
  ask_baytak           the whole LangGraph agent team in one call (answer + cited homes)

Transports:
  HTTP   mounted on the web app at /mcp (streamable HTTP) - e.g. https://<app>/mcp/
  stdio  python -m app.mcp_server   (for local clients such as Claude Desktop)

Tools are read-only. Set MCP_API_KEY to require `Authorization: Bearer <key>` on /mcp."""
from __future__ import annotations

import json
from typing import Callable, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.data.reference import COMMUNITIES, WORK_HUBS

INSTRUCTIONS = ("Baytak AI: Dubai property search and advice grounded in real Dubai Land Department (DLD) deals. "
                "Call search_homes before recommending homes; cite listings by ID; take numbers from tool results. "
                "Rent budgets are annual (AED). Estimates only - not financial or legal advice.")

_state: dict = {}   # set by bind(): {"rec": Recommender, "settings": Settings, "agent": chat agent, "store": market store}


def bind(get_rec: Callable, settings, get_agent: Callable | None = None, get_store: Callable | None = None) -> None:
    _state.update(get_rec=get_rec, settings=settings, get_agent=get_agent, get_store=get_store)


def _toolbox():
    from app.agents.tools import Toolbox
    return Toolbox(_state["get_rec"](), _state["settings"])


mcp = FastMCP(
    "Baytak AI", instructions=INSTRUCTIONS, stateless_http=True, json_response=True, streamable_http_path="/",
    # behind Azure Container Apps ingress the Host header is the public FQDN, not localhost
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def search_homes(purpose: Literal["sale", "rent"] = "sale", budget_aed: float | None = None, min_bedrooms: int = 0,
                 property_types: list[Literal["apartment", "villa", "townhouse"]] | None = None,
                 work_location: str | None = None, max_commute_min: float | None = None,
                 family_with_kids: bool = False, wants_golden_visa: bool = False,
                 lifestyle_tags: list[str] | None = None, preferred_communities: list[str] | None = None,
                 free_text: str = "") -> dict:
    """Search and rank Dubai homes. budget_aed is the total price for sale, or the ANNUAL rent for rent.
    Returns up to 6 homes with price, size, commute, Golden Visa eligibility and the reasons they were ranked."""
    return _toolbox().search_homes(
        purpose=purpose, budget_aed=budget_aed, min_bedrooms=min_bedrooms, property_types=property_types or [],
        work_location=work_location, max_commute_min=max_commute_min, family_with_kids=family_with_kids,
        wants_golden_visa=wants_golden_visa, lifestyle_tags=lifestyle_tags or [],
        preferred_communities=preferred_communities or [], free_text=free_text)


@mcp.tool()
def check_affordability(price_aed: float, monthly_income_aed: float, existing_monthly_debt_aed: float = 0,
                        uae_national: bool = False, first_home: bool = True, off_plan: bool = False) -> dict:
    """Mortgage affordability under UAE Central Bank rules: max LTV, monthly installment, debt-burden ratio
    (50% cap) and cash needed upfront (down payment, 4% DLD fee, registration, agent, trustee)."""
    return _toolbox().check_affordability(price_aed=price_aed, monthly_income_aed=monthly_income_aed,
                                          existing_monthly_debt_aed=existing_monthly_debt_aed, uae_national=uae_national,
                                          first_home=first_home, off_plan=off_plan)


@mcp.tool()
def check_golden_visa(price_aed: float, mortgaged: bool = False, off_plan: bool = False) -> dict:
    """Does a property purchase meet the UAE 10-year Golden Visa (investor) property threshold?"""
    return _toolbox().check_golden_visa(price_aed=price_aed, mortgaged=mortgaged, off_plan=off_plan)


@mcp.tool()
def estimate_commute(community: str, work_location: str) -> dict:
    """Estimated peak and off-peak drive time from a Dubai community to a work location (e.g. DIFC)."""
    return _toolbox().estimate_commute(community=community, work_location=work_location)


@mcp.tool()
def community_profile(community: str) -> dict:
    """Price per sq ft (real DLD median), rental yield, metro distance, schools and lifestyle of a Dubai community."""
    return _toolbox().community_profile(community=community)


@mcp.tool()
def market_brief() -> dict:
    """The latest daily brief from the Market Data Agent: newest DLD deal date, deals in store, recent runs."""
    get_store = _state.get("get_store")
    store = get_store() if get_store else None
    if store is None:
        return {"note": "Market data store not configured on this server."}
    return {"brief": store.get_memory("market_brief"), "watermark": store.get_memory("watermark"),
            "store": store.stats(), "recent_runs": (store.get_memory("runs", []) or [])[-3:]}


@mcp.tool()
def ask_baytak(question: str, monthly_income_aed: float | None = None) -> dict:
    """Ask the Baytak AI agent team (supervisor + search, finance, visa and neighbourhood specialists).
    Returns the written answer, the cited homes and which agents/tools ran."""
    from app.schemas import ChatRequest
    agent = _state["get_agent"]() if _state.get("get_agent") else None
    if agent is None:
        from app.agents.graph import GraphAgent
        agent = GraphAgent(_state["get_rec"](), _state["settings"])
    r = agent.chat(ChatRequest(message=question, monthly_income_aed=monthly_income_aed))
    return {"answer": r.answer, "grounded": r.grounded, "agents": r.agents, "tools": [t.tool for t in r.tool_trace],
            "homes": [{"listing_id": x.listing.listing_id, "community": x.listing.community,
                       "bedrooms": x.listing.bedrooms, "price_aed": x.listing.price_aed,
                       "annual_rent_aed": x.listing.annual_rent_aed} for x in r.recommendations[:5]]}


@mcp.resource("baytak://communities")
def communities() -> str:
    """The Dubai communities and work hubs Baytak AI knows about."""
    return json.dumps({"communities": [c.name for c in COMMUNITIES], "work_hubs": [h.name for h in WORK_HUBS]})


@mcp.prompt()
def find_a_home(budget: str, bedrooms: str = "2", purpose: str = "buy", work_location: str = "") -> str:
    """Prompt template: find a home in Dubai."""
    near = f", commuting to {work_location}" if work_location else ""
    return (f"I want to {purpose} a {bedrooms}-bedroom home in Dubai for {budget}{near}. "
            "Use search_homes, then check affordability / Golden Visa / commute where relevant, and cite listing IDs.")


def http_app(api_key: str | None = None):
    """ASGI app for mounting at /mcp (optionally behind a bearer key)."""
    inner = mcp.streamable_http_app()
    if not api_key:
        return inner

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
            if auth != f"Bearer {api_key}":
                from starlette.responses import JSONResponse
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await inner(scope, receive, send)
    return guarded


def main() -> None:
    """stdio server for local MCP clients (loads the same artifacts as the web app)."""
    from app.config import get_settings
    from app.recsys.pipeline import Recommender
    from app.storage.feedback import get_store
    s = get_settings()
    rec = Recommender(s, get_store(s))
    bind(lambda: rec, s)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
