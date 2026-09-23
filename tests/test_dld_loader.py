"""The real-data loader understands DLD column names and maps areas to communities."""
import pandas as pd

from scripts.load_dld_real import aggregate, load, to_homes


def _sales(tmp_path):
    rows = []
    for i in range(12):
        rows.append({"instance_date": f"{1 + i:02d}-06-2026", "trans_group_en": "Sales", "property_sub_type_en": "Flat",
                     "property_usage_en": "Residential", "reg_type_en": "Existing Properties",
                     "area_name_en": "Marsa Dubai", "master_project_en": "Dubai Marina", "project_name_en": "Marina Gate",
                     "building_name_en": "Marina Gate 1", "rooms_en": "2 B/R", "procedure_area": 110 + i,
                     "actual_worth": 2_400_000 + i * 10_000})
    rows.append({**rows[0], "master_project_en": "DAMAC Hills 2 (Akoya by DAMAC)", "area_name_en": "Al Hebiah Sixth"})
    rows.append({**rows[0], "trans_group_en": "Mortgages"})
    p = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_loader_builds_real_homes(tmp_path):
    s = load(_sales(tmp_path), "sale", months=12)
    assert set(s["community"]) == {"Dubai Marina"}          # Akoya excluded, mortgage row dropped
    homes = to_homes(aggregate(s, "sale", min_deals=2), None, s["date"].max())
    assert len(homes) == 1
    h = homes.iloc[0]
    assert h.data_source == "DLD" and h.building == "Marina Gate 1" and h.bedrooms == 2
    assert h.n_transactions == 12 and 2_400_000 <= h.price_aed <= 2_520_000
    assert h.listing_id.startswith("DLD-")
