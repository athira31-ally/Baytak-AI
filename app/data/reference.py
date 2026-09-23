"""Dubai reference data: freehold communities and major work hubs.

IMPORTANT: coordinates are approximate and the price/yield/school figures are
ILLUSTRATIVE seed values for development. Replace them with real figures from
Dubai Land Department open data (Dubai Pulse), the RERA rental index and KHDA
school inspection results before presenting numbers as facts.
See scripts/load_dld_transactions.py for the real-data path.
"""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Community:
    name: str
    lat: float
    lon: float
    price_psf: int            # AED per sq ft (sale), illustrative
    gross_yield: float        # annual rent / price, illustrative
    metro_km: float           # distance to nearest metro station
    school_score: int         # 0-4 (4 = "Outstanding"-rated school nearby), illustrative
    types: tuple[str, ...]    # property types commonly available
    tags: tuple[str, ...] = field(default_factory=tuple)
    name_ar: str = ""


COMMUNITIES: list[Community] = [
    Community("Dubai Marina", 25.080, 55.140, 2100, 0.065, 0.4, 3, ("apartment",), ("waterfront", "nightlife", "walkable", "metro", "luxury"), "دبي مارينا"),
    Community("Downtown Dubai", 25.197, 55.274, 3000, 0.055, 0.5, 3, ("apartment",), ("luxury", "walkable", "metro", "nightlife", "city-views"), "وسط مدينة دبي"),
    Community("Business Bay", 25.186, 55.262, 2100, 0.065, 0.8, 2, ("apartment",), ("walkable", "metro", "city-views", "investment"), "الخليج التجاري"),
    Community("Jumeirah Lake Towers", 25.069, 55.142, 1600, 0.072, 0.5, 2, ("apartment",), ("metro", "affordable", "walkable", "investment"), "أبراج بحيرات جميرا"),
    Community("Palm Jumeirah", 25.112, 55.139, 3500, 0.050, 3.0, 3, ("apartment", "villa"), ("waterfront", "beach", "luxury", "quiet"), "نخلة جميرا"),
    Community("Emaar Beachfront", 25.095, 55.140, 3200, 0.055, 1.5, 3, ("apartment",), ("beach", "waterfront", "luxury", "new-build"), "إعمار بيتشفرونت"),
    Community("Dubai Hills Estate", 25.110, 55.245, 2400, 0.058, 4.0, 4, ("apartment", "villa", "townhouse"), ("family", "green", "golf", "schools", "new-build"), "دبي هيلز استيت"),
    Community("Arabian Ranches", 25.055, 55.270, 1700, 0.052, 9.0, 4, ("villa", "townhouse"), ("family", "quiet", "green", "schools", "golf"), "المرابع العربية"),
    Community("Jumeirah Village Circle", 25.058, 55.209, 1300, 0.080, 5.0, 2, ("apartment", "townhouse"), ("affordable", "investment", "family"), "قرية جميرا الدائرية"),
    Community("Jumeirah Village Triangle", 25.045, 55.190, 1400, 0.068, 5.0, 3, ("villa", "townhouse", "apartment"), ("family", "quiet", "green"), "قرية جميرا المثلث"),
    Community("Arjan", 25.062, 55.237, 1350, 0.078, 6.0, 2, ("apartment",), ("affordable", "investment", "new-build"), "أرجان"),
    Community("Al Furjan", 25.028, 55.150, 1400, 0.070, 1.2, 3, ("apartment", "villa", "townhouse"), ("family", "metro", "quiet"), "الفرجان"),
    Community("Damac Hills", 25.025, 55.250, 1400, 0.065, 12.0, 3, ("villa", "townhouse", "apartment"), ("golf", "family", "green"), "داماك هيلز"),
    Community("Motor City", 25.048, 55.237, 1300, 0.068, 7.0, 3, ("apartment", "townhouse"), ("family", "quiet", "affordable"), "موتور سيتي"),
    Community("Dubai Sports City", 25.040, 55.220, 1100, 0.078, 7.0, 2, ("apartment", "villa"), ("affordable", "investment", "golf"), "مدينة دبي الرياضية"),
    Community("Town Square", 24.995, 55.290, 1100, 0.072, 15.0, 2, ("apartment", "townhouse"), ("affordable", "family", "new-build"), "تاون سكوير"),
    Community("Dubai Silicon Oasis", 25.118, 55.383, 1000, 0.080, 8.0, 3, ("apartment", "villa"), ("affordable", "family", "tech-hub"), "واحة دبي للسيليكون"),
    Community("Dubai Creek Harbour", 25.200, 55.345, 2300, 0.060, 3.0, 2, ("apartment",), ("waterfront", "new-build", "city-views", "green"), "مرسى خور دبي"),
    Community("Mohammed Bin Rashid City", 25.160, 55.310, 2000, 0.058, 5.0, 3, ("apartment", "villa", "townhouse"), ("luxury", "green", "new-build", "family"), "مدينة محمد بن راشد"),
    Community("International City", 25.165, 55.408, 750, 0.090, 9.0, 1, ("apartment",), ("affordable", "investment"), "المدينة العالمية"),
    Community("Dubai South", 24.890, 55.160, 1000, 0.070, 6.0, 2, ("apartment", "villa", "townhouse"), ("affordable", "new-build", "family", "quiet"), "دبي الجنوب"),
]


def _apply_real_data_overrides() -> None:
    """If scripts/load_dld_transactions.py has produced real medians, use them."""
    import dataclasses, json, os
    from pathlib import Path
    path = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "artifacts")) / "community_overrides.json"
    if not path.exists():
        return
    over = json.loads(path.read_text()).get("communities", {})
    for i, c in enumerate(COMMUNITIES):
        if c.name in over:
            COMMUNITIES[i] = dataclasses.replace(c, price_psf=int(over[c.name]["price_psf"]))


_apply_real_data_overrides()
COMMUNITY_BY_NAME = {c.name.lower(): c for c in COMMUNITIES}

# Common short names people actually type
COMMUNITY_ALIASES = {
    "marina": "Dubai Marina", "downtown": "Downtown Dubai", "jlt": "Jumeirah Lake Towers",
    "palm": "Palm Jumeirah", "the palm": "Palm Jumeirah", "dubai hills": "Dubai Hills Estate",
    "ranches": "Arabian Ranches", "jvc": "Jumeirah Village Circle", "jvt": "Jumeirah Village Triangle",
    "dso": "Dubai Silicon Oasis", "silicon oasis": "Dubai Silicon Oasis", "creek harbour": "Dubai Creek Harbour",
    "mbr city": "Mohammed Bin Rashid City", "meydan": "Mohammed Bin Rashid City", "business bay": "Business Bay",
    "arjan": "Arjan", "furjan": "Al Furjan", "al furjan": "Al Furjan", "damac hills": "Damac Hills",
    "motor city": "Motor City", "sports city": "Dubai Sports City", "town square": "Town Square",
    "international city": "International City", "dubai south": "Dubai South", "beachfront": "Emaar Beachfront",
}


@dataclass(frozen=True)
class WorkHub:
    name: str
    lat: float
    lon: float
    aliases: tuple[str, ...] = ()


WORK_HUBS: list[WorkHub] = [
    WorkHub("DIFC", 25.212, 55.281, ("difc", "financial centre", "financial center")),
    WorkHub("Downtown Dubai", 25.197, 55.274, ("downtown",)),
    WorkHub("Business Bay", 25.186, 55.262, ("business bay",)),
    WorkHub("Dubai Internet City", 25.095, 55.160, ("internet city", "media city", "dic", "tecom", "knowledge park")),
    WorkHub("DMCC / JLT", 25.069, 55.142, ("dmcc", "jlt")),
    WorkHub("Dubai Silicon Oasis", 25.118, 55.383, ("silicon oasis", "dso")),
    WorkHub("Expo City / Dubai South", 24.963, 55.150, ("expo", "expo city", "dubai south", "dwc", "al maktoum airport")),
    WorkHub("Deira", 25.270, 55.320, ("deira", "port saeed")),
    WorkHub("Dubai Healthcare City", 25.231, 55.323, ("healthcare city", "dhcc")),
    WorkHub("Meydan Free Zone", 25.160, 55.305, ("meydan free zone", "meydan fz")),
    WorkHub("Jebel Ali Free Zone", 25.010, 55.070, ("jafza", "jebel ali")),
    WorkHub("Al Quoz", 25.140, 55.230, ("al quoz", "alserkal")),
    WorkHub("Dubai Airport (DXB)", 25.253, 55.365, ("airport", "dxb", "airport free zone", "dafz")),
]


def resolve_hub(text: str | None) -> WorkHub | None:
    if not text:
        return None
    t = text.lower().strip()
    for h in WORK_HUBS:
        if t == h.name.lower() or t in h.aliases:
            return h
    for h in WORK_HUBS:  # substring fallback
        if h.name.lower() in t or any(a in t for a in h.aliases):
            return h
    return None


def resolve_community(text: str | None) -> Community | None:
    if not text:
        return None
    t = text.lower().strip()
    if t in COMMUNITY_BY_NAME:
        return COMMUNITY_BY_NAME[t]
    if t in COMMUNITY_ALIASES:
        return COMMUNITY_BY_NAME[COMMUNITY_ALIASES[t].lower()]
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Typical price/sq ft of each property type relative to the community's apartment average
TYPE_PSF_FACTOR = {"apartment": 1.0, "townhouse": 0.85, "villa": 0.8}


def commute_minutes(community: Community, hub: WorkHub, peak: bool = True) -> float:
    """Rough driving estimate. Road distance ~ 1.35 x straight line; average speed
    rises with trip length (local roads -> Sheikh Zayed Rd / E311); +8 min access.
    Swap for the Azure Maps Route API for real numbers (see README roadmap)."""
    km = haversine_km(community.lat, community.lon, hub.lat, hub.lon) * 1.35
    speed = 30 + min(km, 40) * 0.9        # km/h
    minutes = km / speed * 60 + 8
    return round(minutes * (1.25 if peak else 1.0), 1)
