"""Replace the illustrative community prices with REAL Dubai Land Department data.

1. Download the DLD "Transactions" CSV from Dubai Pulse / the DLD open-data portal.
2. python -m scripts.load_dld_transactions path/to/transactions.csv
3. Re-run python -m scripts.bootstrap

Writes artifacts/community_overrides.json, which app/data/reference.py loads
automatically, so features, explanations and the community_profile tool all
use real median AED/sq ft.

Column names below match the DLD open-data export at the time of writing;
the script prints what it finds so you can adjust COLS if they change.
"""
import json
import sys

import pandas as pd

from app.config import get_settings
from app.data.reference import COMMUNITIES

COLS = {"area": "area_name_en", "group": "trans_group_en", "worth": "actual_worth",
        "size_sqm": "procedure_area", "date": "instance_date", "reg": "reg_type_en"}

# DLD uses official area names, which differ from marketing names. Verify each
# mapping against a few transactions before relying on it and extend as needed.
DLD_AREA_MAP = {
    "Marsa Dubai": "Dubai Marina",
    "Burj Khalifa": "Downtown Dubai",
    "Business Bay": "Business Bay",
    "Palm Jumeirah": "Palm Jumeirah",
    "Al Thanyah Fifth": "Jumeirah Lake Towers",
    "Al Barsha South Fourth": "Jumeirah Village Circle",
    "Hadaeq Sheikh Mohammed Bin Rashid": "Dubai Hills Estate",
    "Dubai Silicon Oasis": "Dubai Silicon Oasis",
    "International City": "International City",
}


def main(path: str) -> None:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in COLS.values() if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. Found: {list(df.columns)[:40]}")
    df = df[df[COLS["group"]].astype(str).str.contains("Sale", case=False)]
    df = df[(df[COLS["worth"]] > 100_000) & (df[COLS["size_sqm"]] > 15)]
    df["date"] = pd.to_datetime(df[COLS["date"]], errors="coerce", dayfirst=True)
    df = df[df["date"] >= df["date"].max() - pd.DateOffset(months=12)]   # last 12 months
    df["psf"] = df[COLS["worth"]] / (df[COLS["size_sqm"]] * 10.7639)
    df["community"] = df[COLS["area"]].map(DLD_AREA_MAP)

    unmapped = df[df["community"].isna()][COLS["area"]].value_counts().head(25)
    print("Top unmapped DLD areas (add to DLD_AREA_MAP if they match a community):\n", unmapped.to_string())

    stats = (df.dropna(subset=["community"]).groupby("community")
             .agg(price_psf=("psf", "median"), transactions=("psf", "size"),
                  off_plan_share=(COLS["reg"], lambda s: float((s.astype(str).str.contains("Off", case=False)).mean())))
             .round(3))
    print("\nReal 12-month medians:\n", stats.to_string())
    known = {c.name for c in COMMUNITIES}
    overrides = {k: {"price_psf": int(v["price_psf"]), "transactions_12m": int(v["transactions"]),
                     "off_plan_share": float(v["off_plan_share"])}
                 for k, v in stats.to_dict("index").items() if k in known}
    out = get_settings().data_dir / "community_overrides.json"
    out.write_text(json.dumps({"source": f"DLD transactions ({path})", "communities": overrides}, indent=2))
    print(f"\nWrote {out}. Re-run `python -m scripts.bootstrap` to rebuild listings and the ranker.")


if __name__ == "__main__":
    main(sys.argv[1])
