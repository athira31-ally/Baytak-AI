"""Rule-based intent parser (English + basic Arabic). Used when Azure OpenAI is
not configured, and as a fallback if the LLM call fails. It is also a handy
baseline to evaluate the LLM's extraction against."""
from __future__ import annotations

import re

from app.data.reference import COMMUNITIES, COMMUNITY_ALIASES, WORK_HUBS
from app.schemas import UserQuery

TAG_KEYWORDS = {
    "beach": ["beach", "شاطئ"], "waterfront": ["waterfront", "sea view", "marina view", "water view", "بحر"],
    "metro": ["metro", "مترو"], "quiet": ["quiet", "peaceful", "هادئ"], "golf": ["golf"],
    "green": ["green", "park", "garden"], "nightlife": ["nightlife", "restaurants", "bars", "lively"],
    "luxury": ["luxury", "premium", "high-end", "فاخر"], "affordable": ["affordable", "cheap", "budget-friendly", "value"],
    "investment": ["investment", "invest", "roi", "yield", "rental income", "استثمار"],
    "new-build": ["new build", "brand new", "off-plan", "off plan"], "walkable": ["walkable", "walk to"],
    "city-views": ["burj view", "city view", "skyline"], "family": ["family", "kids", "children", "عائلة", "أطفال"],
}

_NUM = r"(\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?)"


def _parse_money(text: str) -> float | None:
    t = text.lower()
    best = None
    for m in re.finditer(_NUM + r"\s*(m\b|mn\b|million|k\b|thousand|مليون|ألف)?", t):
        raw, unit = m.group(1), (m.group(2) or "").strip()
        val = float(raw.replace(",", "").replace(" ", ""))
        val *= {"m": 1e6, "mn": 1e6, "million": 1e6, "مليون": 1e6, "k": 1e3, "thousand": 1e3, "ألف": 1e3}.get(unit, 1)
        if val >= 2_000:  # ignore bedrooms, minutes, family size
            best = max(best or 0, val)
    if best and re.search(r"per month|/month|monthly|a month|pm\b|شهري", t):
        return best * 12
    return best if best and best >= 20_000 else None


def parse_query(text: str) -> UserQuery:
    t = text.lower()
    purpose = "rent" if re.search(r"\b(?:rent|renting|to let|lease|leasing|tenant)\b|إيجار|استئجار|للإيجار", t) else "sale"

    beds = 0
    if m := re.search(r"(\d)\s*-?\s*(?:bed|br\b|bhk|bedroom|غرف)", t):
        beds = int(m.group(1))
    elif m := re.search(r"family of (\d)", t):
        beds = max(2, int(m.group(1)) - 1)
    elif "studio" in t:
        beds = 0
    elif re.search(r"kid|child|family|عائلة|أطفال", t):
        beds = 2

    types = []
    if re.search(r"villa|فيلا", t):
        types.append("villa")
    if "townhouse" in t or "town house" in t:
        types.append("townhouse")
    if re.search(r"apartment|flat|شقة", t):
        types.append("apartment")

    # Work location: prefer an explicit "work in/at X" phrase
    work = None
    m = re.search(r"(?:work|office|job|working)\s+(?:is\s+)?(?:in|at|near|from)\s+([a-z /&]+?)(?:[,.;]|\band\b|$)", t)
    scope = m.group(1) if m else (t if re.search(r"work|office|commute|أعمل|عمل", t) else "")
    for h in WORK_HUBS:
        if any(re.search(rf"\b{re.escape(a)}\b", scope) for a in (h.name.lower(), *h.aliases)):
            work = h.name
            break

    communities = []
    for alias, name in COMMUNITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t) and name not in communities:
            communities.append(name)
    for c in COMMUNITIES:
        if c.name_ar and c.name_ar in text and c.name not in communities:
            communities.append(c.name)
    if work:  # "work in Business Bay" is not a place to live
        communities = [c for c in communities if c.lower() != work.lower() and not (m and c.lower() in m.group(1))]

    tags = [tag for tag, kws in TAG_KEYWORDS.items() if any(k in t for k in kws)]
    family = bool(re.search(r"kid|child|school|family|son|daughter|أطفال|مدرسة|مدارس|عائلة|للعائلة", t))
    if family and "family" not in tags:
        tags.append("family")
    gv = bool(re.search(r"golden visa|residency|الإقامة الذهبية|الفيزا الذهبية", t))
    commute = float(m.group(1)) if (m := re.search(r"(\d+)\s*(?:min|minutes|mins)", t)) else None

    return UserQuery(purpose=purpose, budget_aed=_parse_money(text), min_bedrooms=beds, property_types=types,
                     work_location=work, max_commute_min=commute, family_with_kids=family,
                     wants_golden_visa=gv and purpose == "sale", lifestyle_tags=tags,
                     preferred_communities=communities, off_plan_ok=not re.search(r"\bready\b|جاهز", t),
                     free_text=text)


def is_arabic(text: str) -> bool:
    return len(re.findall(r"[؀-ۿ]", text)) > len(text) * 0.3
