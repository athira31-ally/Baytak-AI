"""Turn REAL Dubai Land Department data into Baytak AI's recommendable homes.

Used by scripts/load_dld_real.py (CSV files) and app/data/live.py (Dubai Pulse API, live).

    python -m scripts.load_dld_real --sales data/raw/transactions.csv [--rents data/raw/rents.csv]
    python -m scripts.bootstrap          # picks up data/real/dld_homes.csv automatically

Where to get the CSVs (free, public):
  * DLD portal  https://dubailand.gov.ae/en/open-data/real-estate-data/  -> Transactions / Rents
    -> filter the last 12 months -> "Download as CSV" (needs a CAPTCHA, so it's a manual step)
  * Dubai Pulse https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open
    (full history; API key needed for automated access)

What a "home" is here
  DLD publishes registered transactions, not live adverts. So each recommendable item is a
  real building + unit type, e.g. "2-bed flat in Marina Gate 1", priced at the MEDIAN of its
  real registered sales (or Ejari rents) over the period, with the number of deals behind it.
  That is honest evidence of what such a home actually costs. Live adverts come from portals
  (Bayut, Property Finder), which don't offer a public API.

Both the Dubai Pulse column names (area_name_en, actual_worth, ...) and the DLD portal export
names (Area, Amount, ...) are recognised automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import ROOT
from app.data.reference import COMMUNITIES, COMMUNITY_ALIASES

OUT = ROOT / "data" / "real" / "dld_homes.csv"
SQM_TO_SQFT = 10.7639

# First matching column wins. Dubai Pulse names first, DLD portal export names second.
CANDIDATES = {
    "id": ["transaction_id", "transaction_number", "trans_id", "contract_id", "transaction_no"],
    "date": ["instance_date", "transaction_date", "contract_start_date", "start_date", "registration_date"],
    "group": ["trans_group_en", "transaction_type", "trans_group"],
    "subtype": ["property_sub_type_en", "property_sub_type", "ejari_property_type_en", "property_type_en", "property_type"],
    "usage": ["property_usage_en", "usage", "property_usage"],
    "reg": ["reg_type_en", "registration_type", "is_offplan", "is_offplan?"],
    "area": ["area_name_en", "area"],
    "master": ["master_project_en", "master_project"],
    "project": ["project_name_en", "project"],
    "building": ["building_name_en", "building_name", "building"],
    "rooms": ["rooms_en", "room(s)", "rooms", "number_of_rooms", "ejari_property_sub_type_en", "property_sub_type"],
    "size_sqm": ["procedure_area", "property_size_(sq.m)", "property_size_sq_m", "actual_area", "area_sqm"],
    "price": ["actual_worth", "amount", "transaction_value", "trans_value"],
    "rent": ["annual_amount", "annual_rent", "contract_amount"],
    "metro": ["nearest_metro_en", "nearest_metro"],
    "parking": ["has_parking", "parking"],
    "units": ["no_of_prop", "number_of_properties", "no_of_properties"],   # Ejari: units in one contract
}

# DLD's official area names -> marketing community names. Master-project names usually match
# directly; these cover rows where only the official area is filled in. Verify with the report.
DLD_AREA_HINTS = {
    # official area names (used when the master project is blank)
    "marsa dubai": "Dubai Marina", "burj khalifa": "Downtown Dubai", "al thanyah fifth": "Jumeirah Lake Towers",
    "al barsha south fourth": "Jumeirah Village Circle", "al barsha south fifth": "Jumeirah Village Triangle",
    "al barshaa south fourth": "Jumeirah Village Circle", "al barshaa south fifth": "Jumeirah Village Triangle",
    "al barsha south third": "Arjan", "al barshaa south third": "Arjan",
    "hadaeq sheikh mohammed bin rashid": "Dubai Hills Estate", "nadd hessa": "Dubai Silicon Oasis",
    "madinat al mataar": "Dubai South", "al khairan first": "Dubai Creek Harbour",
    # master-project names seen in the real DLD file (Sept 2026)
    "dubai world central": "Dubai South", "dubai south residential district": "Dubai South",
    "dmcc master community": "Jumeirah Lake Towers", "dmcc ez": "Jumeirah Lake Towers",
    "mohammed bin rashid al maktoum district": "Mohammed Bin Rashid City", "sobha hartland": "Mohammed Bin Rashid City",
    "meydan": "Mohammed Bin Rashid City", "silicon oasis": "Dubai Silicon Oasis",
}# Names that contain a community name but are a different place
EXCLUDE = ["damachills2", "akoya", "jumeirahvillagecirclemall", "palmdeira", "palmjebelali",
           "downtownjabalali", "dubailandresidencecomplex", "marinagatejebelali"]


def to_dt(series: pd.Series) -> pd.Series:
    """DLD dates come as ISO (2026-09-21) from Dubai Pulse / data.dubai and as 21-09-2026 from the portal."""
    sample = series.dropna().astype(str).head(20)
    iso = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5 if len(sample) else True
    return pd.to_datetime(series, errors="coerce", dayfirst=not iso, format="ISO8601" if iso else None)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""


def _snake(c: str) -> str:
    return re.sub(r"\s+", "_", c.strip().lower())


def pick(df: pd.DataFrame, key: str) -> str | None:
    cols = {_snake(c): c for c in df.columns}
    for cand in CANDIDATES[key]:
        if cand in cols:
            return cols[cand]
    return None


def build_matcher():
    table = [(norm(c.name), c.name) for c in COMMUNITIES]
    table += [(norm(a), n) for a, n in COMMUNITY_ALIASES.items() if len(norm(a)) >= 6]
    table += [(norm(a), n) for a, n in DLD_AREA_HINTS.items()]
    table.sort(key=lambda t: -len(t[0]))            # longest (most specific) first

    def match(*fields) -> str | None:
        if any(x in norm(f) for f in fields for x in EXCLUDE):
            return None                              # a known look-alike (e.g. Damac Hills 2)
        for f in fields:                             # master project > project > area
            nf = norm(f)
            if not nf:
                continue
            for key, name in table:
                if key == nf or key in nf:
                    return name
        return None
    return match


def parse_rooms(v) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).lower()
    if "studio" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m and int(m.group(1)) <= 8 else None


def map_type(v) -> str | None:
    s = str(v).lower()
    if "villa" in s:
        return "villa"
    if "town" in s:
        return "townhouse"
    if any(k in s for k in ("flat", "apartment", "unit", "hotel apartment")):
        return "apartment"
    return None


def type_series(df: pd.DataFrame) -> pd.Series:
    """Property type from the first type-like column that maps (sub type, then type)."""
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    cols = {_snake(c): c for c in df.columns}
    for cand in CANDIDATES["subtype"]:
        if cand in cols:
            out = out.fillna(df[cols[cand]].map(map_type))
    return out


class ColumnError(ValueError):
    pass


def load(path: Path, kind: str, months: int) -> pd.DataFrame:
    try:
        return prepare(pd.read_csv(path, low_memory=False), kind, months, path.name)
    except ColumnError as e:
        sys.exit(str(e))


def prepare(df: pd.DataFrame, kind: str, months: int, name: str = "data", verbose: bool = True) -> pd.DataFrame:
    """Clean raw DLD rows (CSV export or Dubai Pulse API) and tag each with a supported community."""
    say = print if verbose else (lambda *a, **k: None)
    cols = {k: pick(df, k) for k in CANDIDATES}
    need = ["date", "area", "size_sqm", "subtype", "rooms", "price" if kind == "sale" else "rent"]
    missing = [k for k in need if not cols[k]]
    if missing:
        raise ColumnError(f"{name}: can't find columns for {missing}.\nColumns present: {list(df.columns)}\n"
                          f"Add the right name to CANDIDATES in {__file__}.")
    say(f"{name}: {len(df):,} rows. Using columns: " + ", ".join(f"{k}={v}" for k, v in cols.items() if v))

    get = lambda k: df[cols[k]] if cols[k] else pd.Series([None] * len(df), index=df.index)
    out = pd.DataFrame({
        "date": to_dt(get("date")),
        "group": get("group").astype(str), "usage": get("usage").astype(str),
        "type": type_series(df), "rooms": get("rooms").map(parse_rooms),
        "size_sqft": pd.to_numeric(get("size_sqm"), errors="coerce") * SQM_TO_SQFT,
        "value": pd.to_numeric(get("price" if kind == "sale" else "rent"), errors="coerce"),
        "off_plan": get("reg").astype(str).str.contains("off", case=False, na=False),
        "area": get("area"), "master": get("master"), "project": get("project"),
        "building": get("building"), "metro": get("metro"), "parking": get("parking"),
    })
    n0 = len(out)
    if kind == "sale" and cols["group"]:
        out = out[out["group"].str.contains("sale|sell", case=False)]
    if cols["usage"]:
        out = out[out["usage"].str.contains("resid", case=False) | out["usage"].isin(["nan", "None"])]
    if kind == "rent" and cols["units"]:
        # bulk leases (one contract for many units) would distort per-home rents
        out = out[pd.to_numeric(get("units").reindex(out.index), errors="coerce").fillna(1) <= 1]
    out = out.dropna(subset=["date", "type", "rooms", "size_sqft", "value"])
    lo = 200_000 if kind == "sale" else 15_000
    out = out[(out["value"] >= lo) & (out["size_sqft"].between(250, 20_000))]
    out = out[out["date"] >= out["date"].max() - pd.DateOffset(months=months)]
    match = build_matcher()
    out["community"] = [match(m, p, a) for m, p, a in zip(out["master"], out["project"], out["area"])]
    unmatched = out[out["community"].isna()]
    say(f"  kept {len(out):,}/{n0:,} residential {kind} rows from the last {months} months; "
          f"{len(out) - len(unmatched):,} in the 21 supported communities")
    top = unmatched["area"].value_counts().head(12)
    if len(top):
        say("  biggest unmatched areas (add to DLD_AREA_HINTS if they belong to a community):\n    "
              + "\n    ".join(f"{a}: {n}" for a, n in top.items()))
    return out.dropna(subset=["community"])


def aggregate(df: pd.DataFrame, kind: str, min_deals: int) -> pd.DataFrame:
    df = df.assign(place=df["building"].fillna(df["project"]).fillna(df["master"]).fillna(df["community"]))
    g = df.groupby(["community", "place", "type", "rooms"], dropna=False)
    agg = g.agg(value=("value", "median"), size_sqft=("size_sqft", "median"), deals=("value", "size"),
                last=("date", "max"), off_plan=("off_plan", "mean"), metro=("metro", "first"),
                parking=("parking", "first"), project=("project", "first"), master=("master", "first")).reset_index()
    agg = agg[agg["deals"] >= min_deals]
    agg["purpose"] = kind
    return agg


def to_homes(sales: pd.DataFrame, rents: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    comm = {c.name: c for c in COMMUNITIES}
    parts = [sales] + ([rents] if rents is not None else [])
    items = pd.concat(parts, ignore_index=True)
    rows = []
    for r in items.itertuples(index=False):
        c = comm[r.community]
        beds = int(r.rooms)
        size = int(round(r.size_sqft))
        if r.purpose == "sale":
            price, rent = int(round(r.value, -3)), None
        else:  # implied value from the community yield, only used for Golden Visa / deal features
            rent, price = int(round(r.value, -3)), int(round(r.value / c.gross_yield, -3))
        bed_txt = "Studio" if beds == 0 else f"{beds}-bedroom"
        what = "registered sales" if r.purpose == "sale" else "Ejari rent contracts"
        place = str(r.place)
        desc = (f"{bed_txt} {r.type} in {place}, {r.community}. Median of {int(r.deals)} real DLD {what}; "
                f"typical size {size:,} sq ft. {'Mostly off-plan. ' if r.off_plan >= 0.5 else ''}"
                f"{'Nearest metro: ' + str(r.metro) + '. ' if pd.notna(r.metro) and str(r.metro) != 'nan' else ''}"
                f"Area: {', '.join(c.tags)}.")
        key = f"{r.community}|{place}|{r.type}|{beds}|{r.purpose}"
        rows.append(dict(
            listing_id="DLD-" + hashlib.sha1(key.encode()).hexdigest()[:8].upper(),
            community=r.community, purpose=r.purpose, property_type=r.type, bedrooms=beds, size_sqft=size,
            price_aed=price, annual_rent_aed=rent, off_plan=bool(r.off_plan >= 0.5), handover_year=None,
            furnished=False, amenities="covered parking" if str(r.parking).lower() in ("1", "true", "yes") else "",
            days_listed=int((as_of - r.last).days), description=desc,
            building=place, project=None if pd.isna(r.project) else str(r.project),
            n_transactions=int(r.deals), last_transaction_date=r.last.date().isoformat(), data_source="DLD",
        ))
    return pd.DataFrame(rows).drop_duplicates("listing_id")


WANTED = {c for cands in CANDIDATES.values() for c in cands}
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class _IterStream(io.RawIOBase):
    """Wrap an iterator of bytes (an HTTP download) as a file, so pandas can read it in chunks
    while it downloads - the 1 GB file never has to fit in memory or on disk."""

    def __init__(self, it):
        self.it, self.buf = it, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def _chunks(source, chunksize: int, counter: list | None = None):
    usecols = lambda c: _snake(c) in WANTED          # only the ~15 columns we use
    if isinstance(source, str) and source.startswith("http"):
        import gzip
        import httpx

        def counted(it):
            for b in it:
                if counter is not None:
                    counter[0] += len(b)
                yield b
        with httpx.stream("GET", source, timeout=httpx.Timeout(60, read=300), follow_redirects=True,
                          headers={"User-Agent": BROWSER_UA}) as r:
            r.raise_for_status()
            stream = io.BufferedReader(_IterStream(counted(r.iter_bytes(1 << 20))), buffer_size=1 << 20)
            if stream.peek(2)[:2] == b"\x1f\x8b":   # data.dubai serves *.csv.gz; unzip on the fly if needed
                stream = io.BufferedReader(gzip.GzipFile(fileobj=stream), buffer_size=1 << 20)
            yield from pd.read_csv(stream, usecols=usecols, chunksize=chunksize, low_memory=False)
    else:
        yield from pd.read_csv(source, usecols=usecols, chunksize=chunksize, low_memory=False)


def read_recent(sources: list, months: int, chunksize: int = 250_000, verbose: bool = True,
                since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Stream one or more (possibly huge, multi-part) DLD CSVs - local paths or URLs - keeping only
    the columns we use and rows from the last `months` (or, with `since`, only rows on/after that
    date: the incremental path). The full DLD history is >1 GB, so it is never loaded whole."""
    if since is not None:
        return _read_since(sources, pd.Timestamp(since), chunksize, verbose)
    keep_parts, newest, date_col = [], None, None
    for src in sources:
        n = 0
        for chunk in _chunks(src, chunksize):
            date_col = date_col or pick(chunk, "date")
            if not date_col:
                raise ColumnError(f"{src}: no date column found. Columns: {list(chunk.columns)}")
            n += len(chunk)
            d = to_dt(chunk[date_col])
            newest = d.max() if newest is None or pd.isna(newest) else max(newest, d.max())
            keep_parts.append(chunk[d >= newest - pd.DateOffset(months=months)])
        if verbose:
            print(f"{str(src).rsplit('/', 1)[-1]}: scanned {n:,} rows")
    df = pd.concat(keep_parts, ignore_index=True)
    # the cutoff moves as newer rows appear, so filter once more against the final newest date
    d = to_dt(df[date_col])
    return df[d >= newest - pd.DateOffset(months=months)]


def _read_since(sources, since, chunksize, verbose) -> pd.DataFrame:
    parts, n = [], 0
    for src in sources:
        for chunk in _chunks(src, chunksize):
            col = pick(chunk, "date")
            if not col:
                raise ColumnError(f"{src}: no date column found. Columns: {list(chunk.columns)}")
            n += len(chunk)
            parts.append(chunk[to_dt(chunk[col]) >= since])
    if verbose:
        print(f"scanned {n:,} rows, {sum(len(p) for p in parts):,} on/after {since.date()}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


RENT_KEEP = ["community", "building", "project", "master", "type", "rooms", "size_sqft", "value", "date",
             "off_plan", "metro", "parking"]


def rent_table(sources: list, months: int = 12, min_deals: int = 2, chunksize: int = 250_000,
               resolve=None, today: pd.Timestamp | None = None, max_seconds: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Stream the Ejari rent-contract files (~5 GB, 10M+ contracts) once and return the aggregated
    rent table: median annual rent per (community, building/project, type, bedrooms) over the last
    `months`, with the number of contracts behind each. Only residential single-unit contracts in the
    supported communities are kept, so memory stays small while the whole history streams past."""
    import time as _time
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    cutoff = today - pd.DateOffset(months=months)
    # Ejari contracts can be registered to START in the future: window on today, not on the newest date
    horizon = today + pd.Timedelta(days=62)
    kept, scanned, bytes_read, t0 = [], 0, [0], _time.time()
    for src in sources:
        url = resolve(src) if resolve else src
        for chunk in _chunks(url, chunksize, bytes_read):
            if max_seconds and _time.time() - t0 > max_seconds:
                raise TimeoutError(f"rent scan exceeded {max_seconds / 60:.0f} min after {scanned:,} contracts")
            scanned += len(chunk)
            col = pick(chunk, "date")
            if not col:
                raise ColumnError(f"{src}: no date column. Columns: {list(chunk.columns)}")
            d = to_dt(chunk[col])
            chunk = chunk[(d >= cutoff) & (d <= horizon)]
            if len(chunk):
                p = prepare(chunk, "rent", months=1200, name=str(src), verbose=False)
                kept.append(p[[c for c in RENT_KEEP if c in p]])
    if not kept:
        return pd.DataFrame(), {"contracts_scanned": scanned, "contracts_used": 0, "mb_downloaded": round(bytes_read[0] / 1e6, 1)}
    r = pd.concat(kept, ignore_index=True)
    r = r.assign(date=r["date"].clip(upper=today))             # future starts count as "current"
    agg = aggregate(r, "rent", min_deals)
    return agg, {"contracts_scanned": scanned, "contracts_used": len(r), "as_of": str(r["date"].max().date()),
                 "mb_streamed": round(bytes_read[0] / 1e6, 1), "seconds": round(_time.time() - t0, 1)}


def build(sales_raw: pd.DataFrame, rents_raw: pd.DataFrame | None = None, months: int = 12, min_deals: int = 2,
          verbose: bool = True, rents_agg: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw DLD rows -> (homes table, community price overrides). Used by the CLI and the live refresher.
    `rents_agg` is an already-aggregated rent table (from rent_table), used by the Market Data Agent."""
    s = prepare(sales_raw, "sale", months, "sales", verbose)
    rents = None
    as_of = s["date"].max()
    if rents_agg is not None and len(rents_agg):
        rents = rents_agg.assign(last=pd.to_datetime(rents_agg["last"]), purpose="rent")
        rents = rents[rents["community"].isin({c.name for c in COMMUNITIES})]
    elif rents_raw is not None and len(rents_raw):
        r = prepare(rents_raw, "rent", months, "rents", verbose)
        rents = aggregate(r, "rent", min_deals)
        as_of = max(as_of, r["date"].max())
    homes = to_homes(aggregate(s, "sale", min_deals), rents, as_of)
    psf = (s["value"] / s["size_sqft"]).groupby(s["community"]).median().round().astype(int)
    overrides = {"source": f"DLD sales, last {months} months to {as_of.date()}", "as_of": str(as_of.date()),
                 "communities": {k: {"price_psf": int(v)} for k, v in psf.items()}}
    return homes, overrides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sales", required=True, type=Path, nargs="+", help="one or more CSV parts")
    ap.add_argument("--rents", type=Path, nargs="+")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--min-deals", type=int, default=2, help="drop building/unit types with fewer deals")
    a = ap.parse_args()
    try:
        homes, overrides = build(read_recent(a.sales, a.months),
                                 read_recent(a.rents, a.months) if a.rents else None, a.months, a.min_deals)
    except ColumnError as e:
        sys.exit(str(e))
    if len(homes) < 200:
        print(f"!! Only {len(homes)} homes - download a longer period or lower --min-deals.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    homes.to_csv(OUT, index=False)
    print(f"\nWrote {len(homes):,} real homes to {OUT.relative_to(ROOT)}  "
          f"(sale: {(homes.purpose == 'sale').sum():,}, rent: {(homes.purpose == 'rent').sum():,})")
    print(homes.groupby("community").size().sort_values(ascending=False).to_string())
    import json
    (ROOT / "data" / "real" / "community_overrides.json").write_text(json.dumps(overrides, indent=2))
    print("Wrote data/real/community_overrides.json (real median AED/sq ft per community).")
    print("Next: python -m scripts.bootstrap")


if __name__ == "__main__":
    main()
