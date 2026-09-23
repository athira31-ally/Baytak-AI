"""Synthetic Dubai listings + simulated user sessions for training the ranker.

Why synthetic: public DLD data has transactions, not live listings or user
clicks. We simulate realistic users (budgets, work hubs, families) whose hidden
preferences generate clicks/saves. The ranker must learn those preferences
from observable features, which is exactly the real-world problem.
Swap in real listings via scripts/load_dld_transactions.py when ready.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.reference import COMMUNITIES, TYPE_PSF_FACTOR, WORK_HUBS, commute_minutes

AMENITIES = ["pool", "gym", "covered parking", "balcony", "maid's room", "study", "private garden",
             "kids play area", "concierge", "sea view", "burj view", "golf view", "smart home",
             "chiller-free", "pet friendly", "shared bbq area", "private pool", "storage room"]
ALL_TAGS = sorted({t for c in COMMUNITIES for t in c.tags})


def generate_listings(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        c = COMMUNITIES[rng.integers(len(COMMUNITIES))]
        ptype = c.types[rng.integers(len(c.types))]
        purpose = "sale" if rng.random() < 0.65 else "rent"
        if ptype == "apartment":
            beds = int(rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.15, 0.05]))
            size = int(max(350, rng.normal(480 + beds * 430, 120)))
        elif ptype == "townhouse":
            beds = int(rng.choice([2, 3, 4], p=[0.25, 0.5, 0.25]))
            size = int(rng.normal(1500 + beds * 380, 200))
        else:
            beds = int(rng.choice([3, 4, 5, 6], p=[0.3, 0.35, 0.25, 0.1]))
            size = int(rng.normal(2400 + beds * 650, 400))
        price = int(size * c.price_psf * TYPE_PSF_FACTOR[ptype] * rng.lognormal(0, 0.12) / 1000) * 1000
        rent = int(price * c.gross_yield * rng.lognormal(0, 0.08) / 1000) * 1000
        off_plan = bool(rng.random() < (0.3 if "new-build" in c.tags else 0.05)) and purpose == "sale"
        amen = list(rng.choice(AMENITIES, size=rng.integers(3, 8), replace=False))
        if ptype == "villa" and rng.random() < 0.4:
            amen.append("private pool")
        furnished = bool(rng.random() < 0.4)
        handover = int(rng.integers(2027, 2030)) if off_plan else None
        bed_txt = "Studio" if beds == 0 else f"{beds}-bedroom"
        desc = (f"{bed_txt} {ptype} in {c.name}, {size:,} sq ft. "
                f"{f'Off-plan, handover {handover}. ' if off_plan else 'Ready to move. '}"
                f"{'Furnished. ' if furnished else ''}Features: {', '.join(sorted(set(amen)))}. "
                f"Area: {', '.join(c.tags)}.")
        rows.append(dict(
            listing_id=f"DHM-{i:05d}", community=c.name, purpose=purpose, property_type=ptype,
            bedrooms=beds, size_sqft=size, price_aed=price,
            annual_rent_aed=rent if purpose == "rent" else None,
            off_plan=off_plan, handover_year=handover,
            furnished=furnished, amenities=sorted(set(amen)),
            days_listed=int(rng.exponential(25)), description=desc,
        ))
    return pd.DataFrame(rows)


def sample_user(rng: np.random.Generator) -> dict:
    """A simulated user: an observable query plus hidden preference weights."""
    purpose = "sale" if rng.random() < 0.6 else "rent"
    family = bool(rng.random() < 0.45)
    min_beds = int(rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])) if family else int(rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
    if purpose == "sale":
        budget = float(rng.choice([800_000, 1_200_000, 1_600_000, 2_200_000, 3_000_000, 4_500_000, 7_000_000]))
    else:
        budget = float(rng.choice([55_000, 75_000, 95_000, 130_000, 180_000, 260_000]))
    hub = WORK_HUBS[rng.integers(len(WORK_HUBS))]
    tags = list(rng.choice(ALL_TAGS, size=rng.integers(1, 3), replace=False))
    if family and "family" not in tags:
        tags.append("family")
    return dict(
        query=dict(purpose=purpose, budget_aed=budget, min_bedrooms=min_beds, work_location=hub.name,
                   family_with_kids=family, wants_golden_visa=bool(purpose == "sale" and budget >= 2_000_000 and rng.random() < 0.5),
                   lifestyle_tags=tags, free_text=" ".join(tags)),
        hidden=dict(  # never shown to the model
            w_commute=rng.uniform(0.5, 2.0), w_price=rng.uniform(0.5, 2.0),
            w_tags=rng.uniform(0.5, 1.5), w_school=rng.uniform(1.0, 2.5) if family else 0.0,
            w_space=rng.uniform(0.2, 1.0), w_new=rng.uniform(-0.5, 0.8),
        ),
    )


def true_utility(user: dict, row: pd.Series, community, hub) -> float:
    """Ground-truth utility used only to simulate clicks."""
    q, h = user["query"], user["hidden"]
    price = row["price_aed"] if q["purpose"] == "sale" else row["annual_rent_aed"]
    ratio = price / q["budget_aed"]
    u = 0.0
    u -= h["w_price"] * max(0.0, ratio - 1.0) * 6        # over budget hurts a lot
    u += h["w_price"] * 0.4 * (1 - abs(ratio - 0.85))     # sweet spot slightly under budget
    u -= h["w_commute"] * commute_minutes(community, hub) / 30
    u += h["w_tags"] * len(set(q["lifestyle_tags"]) & set(community.tags)) * 0.6
    u += h["w_school"] * community.school_score / 4
    u += h["w_space"] * (row["size_sqft"] / max(1, row["bedrooms"] + 1)) / 700
    u += h["w_new"] * (1.0 if row["off_plan"] or "new-build" in community.tags else 0.0)
    if q["wants_golden_visa"]:
        u += 1.5 if row["price_aed"] >= 2_000_000 else -1.5
    u -= 0.3 * max(0, q["min_bedrooms"] - row["bedrooms"])
    return u
