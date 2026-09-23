"""Stage 2 features: everything the ranker can observe about (query, listing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.reference import COMMUNITY_BY_NAME, TYPE_PSF_FACTOR, commute_minutes, resolve_hub
from app.schemas import UserQuery

GOLDEN_VISA_AED = 2_000_000

FEATURES = [
    "price_ratio", "over_budget", "commute_min", "school_score", "family_x_school",
    "tag_overlap", "bed_surplus", "size_per_room", "deal_score", "gross_yield", "metro_km",
    "new_build", "days_listed", "gv_eligible", "gv_match", "semantic_sim", "retrieval_score",
    "is_villa", "is_townhouse",
]


def build_features(q: UserQuery, cand: pd.DataFrame) -> pd.DataFrame:
    hub = resolve_hub(q.work_location)
    comms = [COMMUNITY_BY_NAME[c.lower()] for c in cand["community"]]
    price = cand["price_aed"] if q.purpose == "sale" else cand["annual_rent_aed"]
    budget = q.budget_aed or float(price.median())
    f = pd.DataFrame(index=cand.index)
    f["price_ratio"] = price / budget
    f["over_budget"] = (f["price_ratio"] - 1).clip(lower=0)
    f["commute_min"] = [commute_minutes(c, hub) if hub else 30.0 for c in comms]
    f["school_score"] = [c.school_score for c in comms]
    f["family_x_school"] = f["school_score"] * float(q.family_with_kids)
    f["tag_overlap"] = cand["tag_overlap"] if "tag_overlap" in cand else 0
    f["bed_surplus"] = cand["bedrooms"] - q.min_bedrooms
    f["size_per_room"] = cand["size_sqft"] / (cand["bedrooms"] + 1)
    expected_psf = np.array([c.price_psf * TYPE_PSF_FACTOR[t] for c, t in zip(comms, cand["property_type"])])
    f["deal_score"] = expected_psf / (cand["price_aed"] / cand["size_sqft"]).to_numpy()
    f["gross_yield"] = [c.gross_yield for c in comms]
    f["metro_km"] = [c.metro_km for c in comms]
    f["new_build"] = [float(o or "new-build" in c.tags) for o, c in zip(cand["off_plan"], comms)]
    f["days_listed"] = cand["days_listed"]
    f["gv_eligible"] = (cand["price_aed"] >= GOLDEN_VISA_AED).astype(float) * float(q.purpose == "sale")
    f["gv_match"] = f["gv_eligible"] * (1.0 if q.wants_golden_visa else 0.0) \
        - (1 - f["gv_eligible"]) * (1.0 if q.wants_golden_visa else 0.0)
    f["semantic_sim"] = cand.get("semantic_sim", 0.0)
    f["retrieval_score"] = cand.get("retrieval_score", 0.0)
    f["is_villa"] = (cand["property_type"] == "villa").astype(float)
    f["is_townhouse"] = (cand["property_type"] == "townhouse").astype(float)
    return f[FEATURES].astype(float).replace([np.inf, -np.inf], 0).fillna(0)


def explain(q: UserQuery, row: pd.Series, feats: pd.Series) -> list[str]:
    """Human-readable reasons, derived from features (never invented)."""
    reasons = []
    hub = resolve_hub(q.work_location)
    if hub:
        reasons.append(f"~{feats['commute_min']:.0f} min peak-hour drive to {hub.name} (estimate)")
    pr = feats["price_ratio"]
    if q.budget_aed and pr <= 0.95:
        reasons.append(f"{(1 - pr) * 100:.0f}% under your budget")
    elif q.budget_aed and pr > 1.0:
        reasons.append(f"{(pr - 1) * 100:.0f}% over budget - room to negotiate?")
    if feats["deal_score"] >= 1.08:
        reasons.append(f"Price/sq ft ~{(feats['deal_score'] - 1) * 100:.0f}% below similar {row['property_type']}s in {row['community']}")
    if q.family_with_kids and feats["school_score"] >= 3:
        reasons.append("Highly rated schools nearby")
    if q.wants_golden_visa and feats["gv_eligible"]:
        reasons.append("Meets the AED 2M property threshold for the 10-year Golden Visa")
    c = COMMUNITY_BY_NAME[row["community"].lower()]
    matched = sorted(set(q.lifestyle_tags) & set(c.tags))
    if matched:
        reasons.append("Matches your lifestyle: " + ", ".join(matched))
    if c.metro_km <= 1.0:
        reasons.append("Walking distance to the Metro")
    if row.get("off_plan"):
        reasons.append(f"Off-plan, handover {int(row['handover_year'])}")
    return reasons[:5]
