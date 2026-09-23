"""The agent loop.

azure-openai mode: GPT (Azure OpenAI) plans and calls tools via function
calling, up to MAX_STEPS rounds, then writes the answer.
offline mode:      rule-based parser + deterministic plan + templated answer.

Both modes end with a grounding check: every listing ID mentioned in the
answer must have come from a tool result, otherwise the answer is flagged."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

from app.agents.llm import azure_openai_client
from app.agents.parser import is_arabic, parse_query
from app.agents.tools import TOOL_SPECS, Toolbox
from app.config import Settings
from app.recsys.pipeline import Recommender
from app.schemas import ChatRequest, ChatResponse, ToolCallTrace

log = logging.getLogger(__name__)
MAX_STEPS = 6
ID_RE = re.compile(r"DHM-\d{5}")

SYSTEM_PROMPT = """You are Dubai Home Match, a property advisor for people buying or renting in Dubai.

Rules:
- ALWAYS call search_homes before recommending any property. Recommend only listings returned by tools.
- Cite every listing by its ID in square brackets, e.g. [DHM-01234].
- Take all numbers (prices, commute times, fees, installments) from tool results. Never invent figures.
- If the user gives a monthly income and is buying, call check_affordability for your top pick.
- If the user mentions the Golden Visa or residency, call check_golden_visa.
- Rent budgets are ANNUAL in Dubai; convert monthly figures x12 before searching.
- Commute times and market figures are estimates; say so briefly.
- Reply in the user's language (Arabic or English). Keep it concise: 3-5 picks, one line of reasoning each,
  then one practical next step (e.g. viewing, mortgage pre-approval, checking the Ejari/RERA rental index).
- You are not a licensed financial or legal advisor; add a one-line disclaimer when discussing money or visas."""


def _summ(result: dict) -> str:
    s = json.dumps(result, default=str)
    return s[:300] + ("..." if len(s) > 300 else "")


class Agent:
    def __init__(self, recommender: Recommender, settings: Settings):
        self.rec = recommender
        self.s = settings
        self.client = azure_openai_client(settings) if settings.use_azure_openai else None
        self.temperature_ok = True  # reasoning models (gpt-5*, o-series) reject a custom temperature

    def _complete(self, **kw):
        kw.setdefault("model", self.s.azure_openai_chat_deployment)
        if self.temperature_ok:
            try:
                return self.client.chat.completions.create(temperature=0.2, **kw)
            except Exception as e:
                if "temperature" not in str(e).lower():
                    raise
                log.info("Model rejected temperature; retrying without it")
                self.temperature_ok = False
        return self.client.chat.completions.create(**kw)

    def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        tools = Toolbox(self.rec, self.s)
        tools.session_id = req.session_id or str(uuid.uuid4())
        trace: list[ToolCallTrace] = []
        seen_ids: set[str] = set()

        def run_tool(name: str, args: dict) -> dict:
            t = time.perf_counter()
            out = tools.call(name, args)
            seen_ids.update(ID_RE.findall(json.dumps(out, default=str)))
            trace.append(ToolCallTrace(tool=name, arguments=args, result_summary=_summ(out),
                                       latency_ms=round((time.perf_counter() - t) * 1000, 1)))
            return out

        mode = "offline"
        answer = None
        if self.client is not None:
            try:
                answer = self._llm_loop(req, run_tool)
                mode = "azure-openai"
            except Exception as e:  # degrade gracefully, never 500 the user
                log.exception("LLM loop failed, falling back to offline planner: %s", e)
        if answer is None:
            answer = self._offline(req, run_tool)

        mentioned = set(ID_RE.findall(answer))
        ungrounded = sorted(mentioned - seen_ids)
        last = tools.last_search
        return ChatResponse(
            session_id=tools.session_id, answer=answer, query=last.query if last else None,
            recommendations=last.recommendations if last else [], tool_trace=trace,
            grounded=not ungrounded, ungrounded_ids=ungrounded, mode=mode,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ---------------------------------------------------------------- LLM mode
    def _llm_loop(self, req: ChatRequest, run_tool) -> str:
        user_msg = req.message
        if req.monthly_income_aed:
            user_msg += f"\n(My monthly income: AED {req.monthly_income_aed:,.0f})"
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        for _ in range(MAX_STEPS):
            resp = self._complete(messages=messages, tools=TOOL_SPECS, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = run_tool(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)[:6000]})
        # Out of steps: force a final answer without tools
        messages.append({"role": "user", "content": "Summarise your recommendation now using only the tool results above."})
        resp = self._complete(messages=messages)
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------ offline mode
    def _offline(self, req: ChatRequest, run_tool) -> str:
        q = parse_query(req.message)
        res = run_tool("search_homes", q.model_dump())
        picks = res.get("results", [])[:5]
        ar = is_arabic(req.message)
        if not picks:
            return ("لم أجد عقارات مطابقة. جرّب رفع الميزانية أو تقليل عدد الغرف." if ar else
                    "I couldn't find matching homes. Try a higher budget, fewer bedrooms, or another area.")
        lines = []
        for p in picks:
            beds = "Studio" if p["bedrooms"] == 0 else f"{p['bedrooms']}BR"
            price = (f"AED {p['price_aed']:,}" if q.purpose == "sale" else f"AED {p['annual_rent_aed']:,}/yr")
            why = "; ".join(p["reasons"][:3])
            lines.append(f"- [{p['listing_id']}] {beds} {p['type']} in {p['community']} - {price}. {why}")
        head = "أفضل الخيارات لك:" if ar else "Here are my top picks for you:"
        extra = []
        top = picks[0]
        if q.wants_golden_visa and q.purpose == "sale":
            gv = run_tool("check_golden_visa", {"price_aed": top["price_aed"], "off_plan": top["off_plan"]})
            extra.append(f"Golden Visa: [{top['listing_id']}] {'meets' if gv['eligible'] else 'does not meet'} "
                         f"the AED 2M property threshold.")
        if req.monthly_income_aed and q.purpose == "sale":
            a = run_tool("check_affordability", {"price_aed": top["price_aed"], "monthly_income_aed": req.monthly_income_aed,
                                                 "off_plan": top["off_plan"]})
            extra.append(f"Affordability for [{top['listing_id']}]: ~AED {a['monthly_installment_aed']:,}/month, "
                         f"DBR {a['debt_burden_ratio']:.0%} ({'within' if a['within_dbr_cap'] else 'above'} the 50% cap), "
                         f"cash needed upfront ~AED {a['total_cash_needed_aed']:,}.")
        tail = ("Commute times and market figures are estimates. Next step: shortlist 2-3 and book viewings"
                + (", and get mortgage pre-approval." if q.purpose == "sale" else ", and check the RERA rental index before negotiating."))
        return "\n".join([head, *lines, *extra, "", tail])
