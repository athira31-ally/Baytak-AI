"""Multi-agent home search on LangGraph.

    guard -> supervisor -> search -> (finance | visa | neighbourhood, in parallel) -> writer -> grounding
                                                                                      ^           |
                                                                                      +-- retry --+

guard          input guardrail (Azure AI Content Safety Prompt Shields + local checks)
supervisor     reads the request, extracts the search brief and decides which specialists are needed
search         calls search_homes; if nothing matches it relaxes the brief itself and searches again
finance        mortgage affordability under UAE rules (LTV cap, installment, debt-burden ratio, upfront costs)
visa           10-year Golden Visa property threshold
neighbourhood  commute estimates and community profiles
writer         writes the answer from the shortlist + specialist findings only
grounding      every listing ID in the answer must have come from a tool; otherwise one rewrite, then flag

Every node works with or without an LLM: with one (Azure OpenAI, or an open-source model on
Ollama/vLLM) the supervisor, specialists and writer reason with it; without one - or if a call
fails - each node falls back to a deterministic version, so the app never goes down. Numbers
always come from tools, never from the model. Each node is an OpenTelemetry span."""
from __future__ import annotations

import json
import logging
import operator
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app import observability, safety
from app.agents.llm import chat_model
from app.agents.parser import is_arabic, parse_query
from app.agents.tools import TOOL_SPECS, Toolbox
from app.config import Settings
from app.recsys.pipeline import Recommender
from app.schemas import LISTING_ID_RE, ChatRequest, ChatResponse, ToolCallTrace, UserQuery

log = logging.getLogger(__name__)
ID_RE = LISTING_ID_RE
SPECIALISTS = ("finance", "visa", "neighbourhood")
SPECIALIST_TOOLS = {"finance": ["check_affordability"], "visa": ["check_golden_visa"],
                    "neighbourhood": ["estimate_commute", "community_profile"]}
SPEC_BY_NAME = {t["function"]["name"]: t for t in TOOL_SPECS}
MAX_SPECIALIST_STEPS = 3

FINANCE_WORDS = re.compile(r"mortgage|afford|installment|instalment|emi|loan|down ?payment|salary|income|dbr|"
                           r"تمويل|قرض|راتب", re.I)
VISA_WORDS = re.compile(r"golden visa|visa|residen|إقامة|الذهبية|تأشيرة", re.I)
AREA_WORDS = re.compile(r"commut|drive|near|close to|school|community|area|neighbou?rhood|metro|office|work|"
                        r"قريب|مدرسة|منطقة", re.I)


# --------------------------------------------------------------------------- state
class State(TypedDict, total=False):
    message: str
    income: float | None
    query: dict
    plan: list[str]
    shortlist: list[dict]
    findings: Annotated[list[dict], operator.add]
    agents: Annotated[list[str], operator.add]
    answer: str
    blocked: bool
    retries: int
    ungrounded: list[str]


@dataclass
class RunCtx:
    """Per-request context shared by the nodes (tool box, trace, LLM)."""
    tools: Toolbox
    llm: object | None
    settings: Settings
    fast: object | None = None          # same model at minimal reasoning effort, for short routing/tool steps
    trace: list[ToolCallTrace] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    used_llm: bool = False
    degraded: list[str] = field(default_factory=list)

    @property
    def quick(self):
        return self.fast or self.llm

    def run_tool(self, name: str, args: dict) -> dict:
        t = time.perf_counter()
        with observability.span(f"tool.{name}", tool=name, args=json.dumps(args, default=str)[:500]):
            out = self.tools.call(name, args)
        self.seen_ids.update(ID_RE.findall(json.dumps(out, default=str)))
        s = json.dumps(out, default=str)
        self.trace.append(ToolCallTrace(tool=name, arguments=args, result_summary=s[:300] + ("..." if len(s) > 300 else ""),
                                        latency_ms=round((time.perf_counter() - t) * 1000, 1)))
        return out


def _ctx(config) -> RunCtx:
    return config["configurable"]["ctx"]


def _text(msg) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c or [])


def _llm_step(ctx: RunCtx, node: str, fn, fallback):
    """Run the LLM version of a node; on any error fall back to the deterministic version."""
    if ctx.llm is None:
        return fallback()
    try:
        out = fn()
        ctx.used_llm = True
        return out
    except Exception as e:
        log.warning("%s: LLM step failed, using deterministic fallback: %s", node, e)
        ctx.degraded.append(node)
        return fallback()


# --------------------------------------------------------------------------- guard
def guard(state: State, config) -> dict:
    ctx = _ctx(config)
    with observability.span("agent.guard"):
        res = safety.check_input(ctx.settings, state["message"])
    if not res.allowed:
        log.info("guard blocked request: %s", res.reason)
        return {"blocked": True, "answer": f"{safety.REFUSAL} ({res.reason})", "agents": ["guard"]}
    return {"blocked": False, "agents": ["guard"]}


# ---------------------------------------------------------------------- supervisor
class Brief(BaseModel):
    """What the user wants, and which specialists should work on it."""
    purpose: Literal["sale", "rent"] = "sale"
    budget_aed: float | None = Field(None, description="Total price to buy, or ANNUAL rent (monthly x 12)")
    min_bedrooms: int = Field(0, description="0 = studio")
    property_types: list[Literal["apartment", "villa", "townhouse"]] = []
    work_location: str | None = Field(None, description="Where they work, e.g. DIFC, Dubai Internet City")
    family_with_kids: bool = False
    wants_golden_visa: bool = False
    lifestyle_tags: list[str] = []
    preferred_communities: list[str] = []
    specialists: list[Literal["finance", "visa", "neighbourhood"]] = Field(
        [], description="finance: affordability/mortgage/income; visa: Golden Visa or residency; "
                        "neighbourhood: commute, schools, community questions")


SUPERVISOR_PROMPT = """You are the supervisor of a team of Dubai property agents. Read the user's request and fill
in the brief: what they want to buy or rent, and which specialists should help.
Rent budgets are ANNUAL in Dubai (convert monthly x12). Choose specialists only when the request needs them:
finance (mortgage, affordability, income given), visa (Golden Visa / residency), neighbourhood (commute,
schools, what an area is like). A search specialist always runs; you don't list it."""


def _rule_plan(msg: str, q: UserQuery, income) -> list[str]:
    plan = []
    if q.purpose == "sale" and (income or FINANCE_WORDS.search(msg)):
        plan.append("finance")
    if q.purpose == "sale" and (q.wants_golden_visa or VISA_WORDS.search(msg)):
        plan.append("visa")
    if q.work_location or q.family_with_kids or AREA_WORDS.search(msg):
        plan.append("neighbourhood")
    return plan


def supervisor(state: State, config) -> dict:
    ctx = _ctx(config)
    msg, income = state["message"], state.get("income")
    base = parse_query(msg)

    def with_llm():
        text = msg + (f"\n(Monthly income: AED {income:,.0f})" if income else "")
        brief: Brief = ctx.quick.with_structured_output(Brief, method="function_calling").invoke(
            [("system", SUPERVISOR_PROMPT), ("user", text)])
        q = base.model_copy(update={k: v for k, v in brief.model_dump(exclude={"specialists"}).items()
                                    if v not in (None, [], 0, False) or k == "purpose"})
        plan = list(dict.fromkeys(brief.specialists))
        if income and q.purpose == "sale" and "finance" not in plan:
            plan.append("finance")                       # income given -> always check affordability
        return q, plan

    with observability.span("agent.supervisor") as sp:
        q, plan = _llm_step(ctx, "supervisor", with_llm, lambda: (base, _rule_plan(msg, base, income)))
        if q.purpose == "rent":
            plan = [p for p in plan if p == "neighbourhood"]     # no mortgage / property visa for renters
        q.free_text = msg
        sp.set_attribute("baytak.plan", ",".join(plan))
    return {"query": q.model_dump(), "plan": plan, "agents": ["supervisor"]}


# -------------------------------------------------------------------------- search
def search(state: State, config) -> dict:
    ctx = _ctx(config)
    q = UserQuery(**state["query"])
    with observability.span("agent.search"):
        res = ctx.run_tool("search_homes", q.model_dump())
        if not res.get("results"):
            # self-correction: relax the brief (LLM decides how, or a fixed relaxation) and search again
            def relax_llm():
                llm = ctx.quick.bind_tools([SPEC_BY_NAME["search_homes"]], tool_choice="search_homes")
                r = llm.invoke([("system", "The search returned no homes. Call search_homes again with a relaxed brief "
                                           "(raise the budget ~20%, drop community/lifestyle filters) but keep the purpose."),
                                ("user", json.dumps(q.model_dump(), default=str))])
                return r.tool_calls[0]["args"]

            def relax_rules():
                return q.model_copy(update={"budget_aed": q.budget_aed * 1.25 if q.budget_aed else None,
                                            "preferred_communities": [], "lifestyle_tags": [],
                                            "min_bedrooms": max(0, q.min_bedrooms - 1)}).model_dump()
            args = _llm_step(ctx, "search", relax_llm, relax_rules)
            args.setdefault("purpose", q.purpose)
            res = ctx.run_tool("search_homes", args)
        if not res.get("results"):
            # last resort: only what the user certainly said (purpose + their words). The writer is told
            # these are the closest options, not exact matches.
            res = ctx.run_tool("search_homes", {"purpose": q.purpose, "free_text": state["message"]})
            if res.get("results"):
                res["closest_matches_only"] = True
    shortlist = res.get("results", [])[:5]
    notes = [{"agent": "search", "text": "No homes matched every requirement; these are the closest options "
                                         "- say so and suggest what to relax."}] if res.get("closest_matches_only") else []
    return {"shortlist": shortlist, "findings": notes, "agents": ["search"]}


def route_after_search(state: State):
    if not state.get("shortlist"):
        return "writer"
    sends = [Send(p, state) for p in state.get("plan", []) if p in SPECIALISTS]
    return sends or "writer"


# --------------------------------------------------------------------- specialists
SPECIALIST_PROMPTS = {
    "finance": "You are the mortgage specialist. For the top 1-2 homes on the shortlist, call check_affordability "
               "with the listing price and the user's monthly income (and off_plan flag). Report installment, "
               "debt-burden ratio vs the 50% cap and cash needed upfront. If no income is known, say what the user should share.",
    "visa": "You are the Golden Visa specialist. Call check_golden_visa for the top homes on the shortlist and say which "
            "meet the AED 2M threshold. Mention mortgage/off-plan caveats from the tool.",
    "neighbourhood": "You are the neighbourhood specialist. If the user works somewhere, call estimate_commute for the "
                     "communities on the shortlist; otherwise call community_profile for the top 1-2 communities. "
                     "Summarise commute, metro access, schools and lifestyle in 2-3 lines.",
}


def _compact(shortlist: list[dict]) -> list[dict]:
    keep = ("listing_id", "community", "type", "bedrooms", "price_aed", "annual_rent_aed", "off_plan", "commute_min")
    return [{k: r.get(k) for k in keep} for r in shortlist[:3]]


def _specialist_llm(ctx: RunCtx, name: str, state: State) -> str:
    llm = ctx.quick.bind_tools([SPEC_BY_NAME[t] for t in SPECIALIST_TOOLS[name]])
    from langchain_core.messages import ToolMessage
    q = state["query"]
    msgs = [("system", SPECIALIST_PROMPTS[name] + " Use only numbers returned by your tools. Cite listings by ID in square brackets, e.g. [DLD-XXXXXXXX]."),
            ("user", json.dumps({"request": state["message"], "monthly_income_aed": state.get("income"),
                                 "work_location": q.get("work_location"), "shortlist": _compact(state["shortlist"])},
                                default=str))]
    for _ in range(MAX_SPECIALIST_STEPS):
        r = llm.invoke(msgs)
        if not r.tool_calls:
            return _text(r)
        msgs.append(r)
        for tc in r.tool_calls:
            if tc["name"] not in SPECIALIST_TOOLS[name]:
                out = {"error": f"{tc['name']} is not one of your tools"}
            else:
                out = ctx.run_tool(tc["name"], tc["args"])
            msgs.append(ToolMessage(content=json.dumps(out, default=str)[:4000], tool_call_id=tc["id"]))
    return _text(ctx.quick.invoke(msgs + [("user", "Summarise your findings now.")]))


def _specialist_rules(ctx: RunCtx, name: str, state: State) -> str:
    top, q, income = state["shortlist"][0], state["query"], state.get("income")
    if name == "finance":
        if not income:
            return "To check mortgage affordability, share your monthly income."
        a = ctx.run_tool("check_affordability", {"price_aed": top["price_aed"], "monthly_income_aed": income,
                                                 "off_plan": top["off_plan"]})
        return (f"Affordability for [{top['listing_id']}]: ~AED {a['monthly_installment_aed']:,}/month, "
                f"DBR {a['debt_burden_ratio']:.0%} ({'within' if a['within_dbr_cap'] else 'above'} the 50% cap), "
                f"cash needed upfront ~AED {a['total_cash_needed_aed']:,}.")
    if name == "visa":
        gv = ctx.run_tool("check_golden_visa", {"price_aed": top["price_aed"], "off_plan": top["off_plan"]})
        return (f"Golden Visa: [{top['listing_id']}] {'meets' if gv['eligible'] else 'does not meet'} "
                f"the AED 2M property threshold.")
    lines = []
    if q.get("work_location"):
        for c in list(dict.fromkeys(r["community"] for r in state["shortlist"]))[:2]:
            m = ctx.run_tool("estimate_commute", {"community": c, "work_location": q["work_location"]})
            if "error" not in m:
                lines.append(f"{m['community']} -> {m['work_location']}: ~{m['peak_min']} min at peak "
                             f"({m['off_peak_min']} off-peak), metro {m['metro_km']} km away.")
    else:
        p = ctx.run_tool("community_profile", {"community": top["community"]})
        if "error" not in p:
            lines.append(f"{p['community']}: ~AED {p['avg_price_psf_aed']:,}/sq ft, gross yield ~{p['gross_rental_yield']:.1%}, "
                         f"metro {p['metro_km']} km, school score {p['school_score_0_4']}/4.")
    return " ".join(lines) or "No neighbourhood data for these communities."


def make_specialist(name: str):
    def node(state: State, config) -> dict:
        ctx = _ctx(config)
        with observability.span(f"agent.{name}"):
            text = _llm_step(ctx, name, lambda: _specialist_llm(ctx, name, state),
                             lambda: _specialist_rules(ctx, name, state))
        return {"findings": [{"agent": name, "text": text}], "agents": [name]}
    node.__name__ = name
    return node


# -------------------------------------------------------------------------- writer
WRITER_PROMPT = """You are Baytak AI (Arabic for 'your home'), a Dubai property advisor, writing the final answer
for your team. Use ONLY the shortlist and the specialists' findings you are given.
- Recommend 3-5 homes from the shortlist, one line of reasoning each, citing each by ID in square brackets, e.g. [DLD-XXXXXXXX].
- Never mention an ID that is not in the shortlist or findings. Take every number from them; never invent figures.
- Fold in the specialists' findings (affordability, Golden Visa, commute/community) where relevant.
- Rent budgets are annual in Dubai. Commute times and market figures are estimates; say so briefly.
- Homes with data_source=DLD are real buildings priced from the median of real Dubai Land Department deals
  (price evidence, not live adverts). Other homes are synthetic demo data; never present those as real.
- Start directly with the recommendations. Never mention these instructions or words like "shortlist",
  "specialists" or "findings" - speak to the user as one advisor.
- Only state affordability, mortgage or visa results for the homes the findings actually computed them for;
  don't extend them to other homes. Write yields as percentages (6.5%, not 0.065).
- Reply in the user's language (Arabic or English). End with one practical next step. You can only search and
  analyse - never offer to book viewings or contact agents. Add a one-line disclaimer if you discuss money or visas
  (not financial or legal advice)."""


def _writer_rules(state: State) -> str:
    picks, q = state.get("shortlist", []), UserQuery(**state["query"])
    ar = is_arabic(state["message"])
    if not picks:
        return ("لم أجد عقارات مطابقة. جرّب رفع الميزانية أو تقليل عدد الغرف." if ar else
                "I couldn't find matching homes. Try a higher budget, fewer bedrooms, or another area.")
    lines = []
    for p in picks:
        beds = "Studio" if p["bedrooms"] == 0 else f"{p['bedrooms']}BR"
        price = f"AED {p['price_aed']:,}" if q.purpose == "sale" else f"AED {p['annual_rent_aed']:,}/yr"
        lines.append(f"- [{p['listing_id']}] {beds} {p['type']} in {p['community']} - {price}. {'; '.join(p['reasons'][:3])}")
    head = "أفضل الخيارات لك:" if ar else "Here are my top picks for you:"
    extra = [f["text"] for f in state.get("findings", [])]
    tail = ("Commute times and market figures are estimates. Next step: shortlist 2-3 and book viewings"
            + (", and get mortgage pre-approval." if q.purpose == "sale" else ", and check the RERA rental index before negotiating."))
    return "\n".join([head, *lines, *extra, "", tail])


def writer(state: State, config) -> dict:
    ctx = _ctx(config)

    def with_llm():
        payload = {"request": state["message"], "monthly_income_aed": state.get("income"),
                   "shortlist": state.get("shortlist", []), "findings": state.get("findings", [])}
        msgs = [("system", WRITER_PROMPT), ("user", json.dumps(payload, default=str)[:12000])]
        if state.get("ungrounded"):
            msgs.append(("user", f"Your previous draft cited IDs that are not in the shortlist: {state['ungrounded']}. "
                                 "Rewrite it citing only shortlist IDs."))
        return _text(ctx.llm.invoke(msgs))

    with observability.span("agent.writer", retry=state.get("retries", 0)):
        answer = _llm_step(ctx, "writer", with_llm, lambda: _writer_rules(state)) if state.get("shortlist") \
            else _writer_rules(state)
    return {"answer": answer, "agents": ["writer"]}


def grounding(state: State, config) -> dict:
    ctx = _ctx(config)
    ungrounded = sorted(set(ID_RE.findall(state.get("answer", ""))) - ctx.seen_ids)
    with observability.span("agent.grounding", ungrounded=len(ungrounded)):
        pass
    return {"ungrounded": ungrounded, "retries": state.get("retries", 0) + (1 if ungrounded else 0),
            "agents": ["grounding"]}


def route_after_grounding(state: State, config):
    ctx = _ctx(config)
    if state.get("ungrounded") and state.get("retries", 0) <= 1 and ctx.llm is not None:
        return "writer"
    return END


# --------------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("guard", guard)
    g.add_node("supervisor", supervisor)
    g.add_node("search", search)
    for name in SPECIALISTS:
        g.add_node(name, make_specialist(name))
    g.add_node("writer", writer)
    g.add_node("grounding", grounding)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: END if s.get("blocked") else "supervisor", ["supervisor", END])
    g.add_edge("supervisor", "search")
    g.add_conditional_edges("search", route_after_search, [*SPECIALISTS, "writer"])
    for name in SPECIALISTS:
        g.add_edge(name, "writer")
    g.add_edge("writer", "grounding")
    g.add_conditional_edges("grounding", route_after_grounding, ["writer", END])
    return g.compile()


GRAPH = build_graph()


class GraphAgent:
    """Drop-in replacement for the classic Agent: same chat() -> ChatResponse contract."""

    def __init__(self, recommender: Recommender, settings: Settings, llm=None):
        self.rec, self.s = recommender, settings
        self.llm = llm if llm is not None else chat_model(settings)
        # routing and specialist tool steps are short and structured: run them at minimal reasoning effort
        self.fast = llm if llm is not None else (chat_model(settings, effort=settings.llm_fast_reasoning_effort)
                                                 if self.llm is not None else None)

    def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        tools = Toolbox(self.rec, self.s)
        tools.session_id = req.session_id or str(uuid.uuid4())
        ctx = RunCtx(tools=tools, llm=self.llm, fast=self.fast, settings=self.s)
        with observability.span("agent.chat", engine="langgraph", session=tools.session_id):
            out = GRAPH.invoke({"message": req.message, "income": req.monthly_income_aed, "findings": [], "agents": []},
                               config={"configurable": {"ctx": ctx}, "recursion_limit": 25})
        last = tools.last_search
        mode = self.s.llm_label if ctx.used_llm else "offline"
        return ChatResponse(
            session_id=tools.session_id, answer=out.get("answer", ""), query=last.query if last else None,
            recommendations=last.recommendations if last else [], tool_trace=ctx.trace,
            grounded=not out.get("ungrounded"), ungrounded_ids=out.get("ungrounded", []), mode=mode,
            engine="langgraph", agents=out.get("agents", []), blocked=bool(out.get("blocked")),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def mermaid() -> str:
    """The graph as a Mermaid diagram (for the README / docs)."""
    return GRAPH.get_graph().draw_mermaid()
