"""Tools the agent can call. Each tool is a plain function with a JSON schema,
so the same code serves the Azure OpenAI function-calling loop and the
offline planner. Tools return dicts; numbers come from here, never the LLM."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from app.config import Settings
from app.data.reference import COMMUNITIES, commute_minutes, resolve_community, resolve_hub
from app.recsys.pipeline import Recommender
from app.schemas import UserQuery

DISCLAIMER = "Estimates only - confirm with your bank / DLD / GDRFA. Not financial advice."


class Toolbox:
    def __init__(self, recommender: Recommender, settings: Settings):
        self.rec = recommender
        self.s = settings
        self.last_search = None  # RecommendResponse from the most recent search_homes call
        self.session_id: str | None = None
        self.registry: dict[str, Callable[..., dict]] = {
            "search_homes": self.search_homes,
            "check_affordability": self.check_affordability,
            "check_golden_visa": self.check_golden_visa,
            "estimate_commute": self.estimate_commute,
            "community_profile": self.community_profile,
        }

    # ------------------------------------------------------------------ tools
    def search_homes(self, **kwargs) -> dict:
        q = UserQuery(**kwargs)
        resp = self.rec.recommend(q, session_id=self.session_id)
        self.last_search = resp
        return {
            "variant": resp.variant,
            "candidates_considered": resp.candidates_considered,
            "results": [{
                "listing_id": r.listing.listing_id, "community": r.listing.community,
                "type": r.listing.property_type, "bedrooms": r.listing.bedrooms, "size_sqft": r.listing.size_sqft,
                "price_aed": r.listing.price_aed, "annual_rent_aed": r.listing.annual_rent_aed,
                "off_plan": r.listing.off_plan, "commute_min": r.commute_min,
                "golden_visa_eligible": r.golden_visa_eligible, "reasons": r.reasons,
            } for r in resp.recommendations[:6]],
        }

    def check_affordability(self, price_aed: float, monthly_income_aed: float, existing_monthly_debt_aed: float = 0,
                            uae_national: bool = False, first_home: bool = True, off_plan: bool = False) -> dict:
        # LTV caps per UAE Central Bank mortgage regulations (verify current rules)
        if off_plan:
            ltv = 0.50
        elif not first_home:
            ltv = 0.65 if uae_national else 0.60
        elif price_aed <= 5_000_000:
            ltv = 0.85 if uae_national else 0.80
        else:
            ltv = 0.75 if uae_national else 0.70
        loan = price_aed * ltv
        r, n = self.s.mortgage_rate / 12, self.s.mortgage_years * 12
        emi = loan * r / (1 - (1 + r) ** -n)
        dbr = (emi + existing_monthly_debt_aed) / monthly_income_aed if monthly_income_aed else None
        upfront = {
            "down_payment": round(price_aed - loan),
            "dld_transfer_fee_4pct": round(price_aed * self.s.dld_transfer_fee),
            "mortgage_registration_0_25pct": round(loan * 0.0025),
            "agent_commission_2pct_plus_vat": round(price_aed * 0.02 * 1.05),
            "trustee_and_admin_fees_approx": 4_500,
        }
        return {
            "price_aed": price_aed, "max_ltv": ltv, "loan_aed": round(loan),
            "monthly_installment_aed": round(emi), "assumed_rate": self.s.mortgage_rate,
            "tenure_years": self.s.mortgage_years, "debt_burden_ratio": round(dbr, 3) if dbr is not None else None,
            "within_dbr_cap": dbr is not None and dbr <= self.s.max_dbr,
            "upfront_costs": upfront, "total_cash_needed_aed": sum(upfront.values()),
            "note": DISCLAIMER,
        }

    def check_golden_visa(self, price_aed: float, mortgaged: bool = False, off_plan: bool = False) -> dict:
        eligible = price_aed >= self.s.golden_visa_threshold_aed
        notes = [f"Threshold used: property value of at least AED {self.s.golden_visa_threshold_aed:,}."]
        if mortgaged:
            notes.append("Mortgaged property: a bank letter/NOC is typically required.")
        if off_plan:
            notes.append("Off-plan: generally accepted only from approved developers - check with DLD.")
        return {"eligible": eligible, "visa": "10-year Golden Visa (investor)", "notes": notes, "note": DISCLAIMER}

    def estimate_commute(self, community: str, work_location: str) -> dict:
        c, h = resolve_community(community), resolve_hub(work_location)
        if not c or not h:
            return {"error": f"Unknown community or work location: {community!r}, {work_location!r}",
                    "known_communities": [x.name for x in COMMUNITIES]}
        return {"community": c.name, "work_location": h.name, "peak_min": commute_minutes(c, h, True),
                "off_peak_min": commute_minutes(c, h, False), "metro_km": c.metro_km,
                "method": "straight-line x 1.35 road factor at 45 km/h; replace with Azure Maps Route API"}

    def community_profile(self, community: str) -> dict:
        c = resolve_community(community)
        if not c:
            return {"error": f"Unknown community {community!r}", "known_communities": [x.name for x in COMMUNITIES]}
        return {"community": c.name, "name_ar": c.name_ar, "avg_price_psf_aed": c.price_psf,
                "gross_rental_yield": c.gross_yield, "metro_km": c.metro_km, "school_score_0_4": c.school_score,
                "property_types": list(c.types), "tags": list(c.tags),
                "data_note": "Illustrative seed data - replace with DLD/RERA figures"}

    # --------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict[str, Any]) -> dict:
        fn = self.registry.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**args)
        except (TypeError, ValidationError, ValueError) as e:
            return {"error": f"Bad arguments for {name}: {e}"}


_QUERY_PROPS = {
    "purpose": {"type": "string", "enum": ["sale", "rent"]},
    "budget_aed": {"type": "number", "description": "Total price for sale, or ANNUAL rent for rent (convert monthly x12)"},
    "min_bedrooms": {"type": "integer", "description": "0 = studio"},
    "property_types": {"type": "array", "items": {"type": "string", "enum": ["apartment", "villa", "townhouse"]}},
    "work_location": {"type": "string", "description": "e.g. DIFC, Dubai Internet City, Business Bay, Expo City, Meydan Free Zone"},
    "max_commute_min": {"type": "number"},
    "family_with_kids": {"type": "boolean"},
    "wants_golden_visa": {"type": "boolean"},
    "lifestyle_tags": {"type": "array", "items": {"type": "string", "enum": [
        "waterfront", "beach", "nightlife", "walkable", "metro", "luxury", "city-views", "investment", "affordable",
        "quiet", "family", "green", "golf", "schools", "new-build", "tech-hub"]}},
    "preferred_communities": {"type": "array", "items": {"type": "string"}},
    "off_plan_ok": {"type": "boolean"},
    "free_text": {"type": "string", "description": "The user's own words, for semantic matching"},
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_homes",
        "description": "Search and rank Dubai property listings for the user's needs. Always call before recommending homes.",
        "parameters": {"type": "object", "properties": _QUERY_PROPS, "required": ["purpose"]}}},
    {"type": "function", "function": {
        "name": "check_affordability",
        "description": "Mortgage affordability under UAE rules: LTV cap, monthly installment, debt-burden ratio, upfront costs.",
        "parameters": {"type": "object", "properties": {
            "price_aed": {"type": "number"}, "monthly_income_aed": {"type": "number"},
            "existing_monthly_debt_aed": {"type": "number"}, "uae_national": {"type": "boolean"},
            "first_home": {"type": "boolean"}, "off_plan": {"type": "boolean"}},
            "required": ["price_aed", "monthly_income_aed"]}}},
    {"type": "function", "function": {
        "name": "check_golden_visa",
        "description": "Check whether a property purchase meets the UAE 10-year Golden Visa property threshold.",
        "parameters": {"type": "object", "properties": {
            "price_aed": {"type": "number"}, "mortgaged": {"type": "boolean"}, "off_plan": {"type": "boolean"}},
            "required": ["price_aed"]}}},
    {"type": "function", "function": {
        "name": "estimate_commute",
        "description": "Estimate peak and off-peak drive time from a Dubai community to a work location.",
        "parameters": {"type": "object", "properties": {
            "community": {"type": "string"}, "work_location": {"type": "string"}},
            "required": ["community", "work_location"]}}},
    {"type": "function", "function": {
        "name": "community_profile",
        "description": "Price per sq ft, rental yield, metro access, schools and lifestyle tags for a Dubai community.",
        "parameters": {"type": "object", "properties": {"community": {"type": "string"}}, "required": ["community"]}}},
]
