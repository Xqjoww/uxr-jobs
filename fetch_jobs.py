#!/usr/bin/env python3
"""
Build jobs.json for the UXR Jobs Board (US-only) from jobhive's hosted dataset.

Two filters run over the hosted snapshot:
  1. is_uxr  -> keep only genuine UX/user/design research roles (and close
                adjacents: research ops, human factors, usability, ethnography,
                mixed methods). Explicitly rejects the look-alikes that flooded
                the board: quantitative finance "research", academic postdocs,
                R&D / lab / clinical research, and market research.
  2. is_us   -> keep only US positions (by geocoordinates when present, else by
                parsing the location text; anything clearly abroad is dropped).

Install (the GitHub Action does this for you):
    pip install "jobhive-py[scrapers]" pyarrow
"""

import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import httpx
import pandas as pd
import pyarrow.parquet as pq
import jobhive as jh
from jobhive.manifest import Manifest, DEFAULT_MANIFEST_URL
from jobhive.models import ATSType

OUTPUT_PATH = "jobs.json"

# Drop anything older than this. Freshness is the whole pitch.
MAX_AGE_DAYS = 60

WANT_COLS = ["title", "company", "location", "lat", "lon", "is_remote",
             "url", "posted_at", "ats_type", "ats_id",
             "salary_min", "salary_max", "salary_currency"]

# Lossless cheap gate (vectorised) before any per-row work. Every title that
# is_uxr() can accept contains one of these stems.
PRE_STEMS = r"research|\buxr\b|human factor|usability|ethnograph"

# =====================================================================
#  Tolerant manifest fetch (ignore ATS platforms the package doesn't know).
# =====================================================================
_VALID_ATS = {a.value for a in ATSType}


def _strip_unknown_ats(data: dict) -> dict:
    ba = data.get("by_ats")
    if isinstance(ba, dict):
        unknown = sorted(k for k in ba if k not in _VALID_ATS)
        if unknown:
            print(f"  note: ignoring unknown ATS platform(s): {', '.join(unknown)}")
            data = {**data, "by_ats": {k: v for k, v in ba.items() if k in _VALID_ATS}}
    return data


def _tolerant_manifest_fetch(cls, url=DEFAULT_MANIFEST_URL, *, client=None, timeout=30.0):
    owns = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return cls.model_validate(_strip_unknown_ats(resp.json()))
    finally:
        if owns:
            client.close()


Manifest.fetch = classmethod(_tolerant_manifest_fetch)

# =====================================================================
#  UXR filter
# =====================================================================
# Academic / lab / pharma / pure-science roles. These ALWAYS lose, even if a
# "research" word is present. Finance/quant terms are deliberately NOT here:
# a bare "Quantitative Researcher" already fails is_uxr for lacking a UX signal,
# while a legit "Quantitative UX Researcher" (Google has a whole family) should
# pass. Excluding "quantitative" outright was killing the good ones.
HARD_EXCLUDE = re.compile(
    r"\b(postdoc|postdoctoral|research fellow|research assistant|research associate|"
    r"phd (student|intern|candidate|researcher|graduate)|professor|faculty|"
    r"dissertation|tenure|research scientist|research chemist|clinical research|"
    r"laborator|drug product|chemist|chemistry|metallurg|genetic|vaccine|biomedic|"
    r"immunogen|epigenomic|turbomachinery|hydraulic|propulsion)\b", re.I)

# (ux | user experience | user | design) immediately before "research"
UX_RESEARCH = re.compile(r"\b(ux|user experience|user|design)[\s/-]*(ui[\s/-]*)?research", re.I)


def is_uxr(title: str) -> bool:
    t = title or ""
    if HARD_EXCLUDE.search(t):
        return False
    if UX_RESEARCH.search(t):
        return True
    if re.search(r"\bux\b", t, re.I) and re.search(r"research", t, re.I):
        return True          # "UX & Research", "UX/Product Researcher", etc.
    if re.search(r"\buxr\b", t, re.I):
        return True
    if re.search(r"\bhuman factors\b", t, re.I):
        return True
    if re.search(r"\busability\b", t, re.I):
        return True
    if re.search(r"\bethnograph", t, re.I):
        return True
    if re.search(r"\bmixed[\s-]?methods?\b", t, re.I) and re.search(r"research", t, re.I):
        return True
    if re.search(r"research\s?op(s|erations)\b", t, re.I) and \
       re.search(r"\b(ux|user|customer|cx|design|product|insight)", t, re.I):
        return True
    return False


# =====================================================================
#  US filter
# =====================================================================
_STATE_CODES = ("AL AK AZ AR CA CO CT FL GA HI ID IL IN IA KS KY LA ME MD MA MI "
                "MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT "
                "VT VA WA WV WI WY DC").split()
_STATE_NAMES = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "hawaii", "idaho", "illinois", "indiana",
    "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts",
    "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming"]

US_BARE = re.compile(r"\bUS\b")                              # case-sensitive
US_WORDS = re.compile(r"\b(u\.?s\.?a\.?|usa|united states)\b", re.I)
US_STATECODE = re.compile(r"(?:^|[,\s(/])(" + "|".join(_STATE_CODES) + r")(?:$|[,\s)/])")
US_STATENAME = re.compile(r"\b(" + "|".join(_STATE_NAMES) + r")\b", re.I)
US_MISC = re.compile(r"\ball\s+states\b|anywhere in the u\.?s|north america", re.I)

_FOREIGN = [
    "united kingdom", "england", "scotland", "wales", "northern ireland", "ireland",
    "germany", "deutschland", "france", "spain", "espana", "italy", "italia",
    "netherlands", "nederland", "belgium", "belgie", "luxembourg", "switzerland",
    "schweiz", "suisse", "austria", "osterreich", "sweden", "sverige", "norway",
    "denmark", "danmark", "finland", "iceland", "poland", "polska", "portugal",
    "greece", "romania", "hungary", "czech", "czechia", "slovakia", "ukraine",
    "bulgaria", "serbia", "croatia", "slovenia", "cyprus", "armenia", "turkey", "russia",
    "canada", "mexico", "brazil", "brasil", "argentina", "chile", "colombia", "peru",
    "uruguay", "panama", "dominican republic", "india", "china", "taiwan", "hong kong",
    "japan", "singapore", "malaysia", "indonesia", "philippines", "vietnam", "thailand",
    "south korea", "korea", "pakistan", "bangladesh", "australia", "new zealand",
    "israel", "united arab emirates", "saudi", "qatar", "kuwait", "bahrain", "oman",
    "egypt", "jordan", "lebanon", "nigeria", "kenya", "south africa", "ghana", "morocco",
    # cities
    "london", "manchester", "birmingham", "edinburgh", "glasgow", "belfast", "newcastle",
    "bristol", "leicester", "sheffield", "cardiff", "swansea", "cheltenham", "dublin",
    "berlin", "munich", "munchen", "hamburg", "frankfurt", "cologne", "koln", "stuttgart",
    "dusseldorf", "dortmund", "hannover", "regensburg", "goppingen", "neuss", "wolfsburg",
    "erlangen", "nuremberg", "wuppertal", "homburg", "huerth", "weissbach", "castellvi",
    "paris", "lyon", "bordeaux", "toulouse", "clichy", "issy", "marseille", "lille",
    "madrid", "barcelona", "valencia", "bilbao", "bellville", "sant cugat", "lisbon",
    "lisboa", "porto", "amsterdam", "rotterdam", "utrecht", "zwolle", "eindhoven",
    "wageningen", "majadahonda", "brussels", "antwerp", "zurich", "zuerich", "geneva",
    "geneve", "basel", "baar", "zug", "lausanne", "vienna", "wien", "graz", "stockholm",
    "gothenburg", "goteborg", "malmo", "copenhagen", "kobenhavn", "aarhus", "lyngby",
    "oslo", "bergen", "grimstad", "helsinki", "espoo", "otaniemi", "warsaw", "warszawa",
    "krakow", "wroclaw", "prague", "praha", "brno", "budapest", "bucharest", "sofia",
    "athens", "milan", "milano", "rome", "roma", "turin", "naples", "marbella",
    "toronto", "vancouver", "montreal", "ottawa", "calgary", "edmonton", "winnipeg",
    "waterloo", "markham", "mississauga", "kitchener", "etobicoke",
    "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "gurugram", "noida",
    "hyderabad", "pune", "chennai", "kolkata", "ahmedabad", "karnataka", "maharashtra",
    "haryana", "telangana", "gujarat", "shanghai", "beijing", "shenzhen", "guangzhou",
    "hangzhou", "chengdu", "taipei", "tokyo", "osaka", "minato", "yokohama", "seoul",
    "gangnam", "busan", "kuala lumpur", "jakarta", "manila", "mandaluyong", "makati",
    "hanoi", "ho chi minh", "bangkok", "sydney", "melbourne", "brisbane", "perth",
    "adelaide", "canberra", "auckland", "wellington", "parnell", "tel aviv", "jerusalem",
    "haifa", "dubai", "abu dhabi", "riyadh", "jeddah", "doha", "cairo", "amman", "beirut",
    "lagos", "abuja", "nairobi", "johannesburg", "cape town", "casablanca", "kyiv", "kiev",
    "yerevan", "tbilisi", "limassol", "bratislava", "ljubljana", "zagreb", "belgrade",
    "istanbul", "ankara", "sao paulo", "rio de janeiro", "mexico city", "cdmx",
    "buenos aires", "santiago", "bogota", "lima", "montevideo",
    # provinces / regions / scopes
    "ontario", "quebec", "british columbia", "alberta", "manitoba", "saskatchewan",
    "ile-de-france", "bavaria", "hessen", "nordrhein", "catalonia", "lombardy", "attica",
    "midlothian", "emea", "apac", "latam", "worldwide", "europe", "mena",
]
FOREIGN = re.compile(r"\b(" + "|".join(_FOREIGN) + r")\b", re.I)
FOREIGN_CODE = re.compile(
    r"(?:^|[,·\s(])(gb|uk|de|fr|nl|es|it|se|no|dk|fi|pl|cz|hu|ro|gr|pt|ie|be|at|ch|lu|"
    r"hr|si|rs|tr|ru|br|ar|cl|pe|cn|hk|tw|jp|kr|sg|my|id|ph|vn|th|pk|bd|au|nz|il|ae|sa|"
    r"qa|eg|ng|ke|za|ma|ua|am|ge|cy|bg|ca)(?:$|[,\s)])")          # lowercase only
CA_PROVINCE = re.compile(r"(?:^|[,\s])(ON|QC|BC|AB|MB|SK|NS|NB)(?:$|[,\s])")
EURES = re.compile(r"\b[A-Z]{2}\s+\([A-Z]{2}")                    # "DE (DE212)" NUTS format


# Any of these scripts means the location is not a US one (CJK, Japanese kana,
# Hangul, Cyrillic, Hebrew/Arabic, Thai, Devanagari).
NON_LATIN = re.compile(
    r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff\u0590-\u06ff\u0e00-\u0e7f\u0900-\u097f]")


def _ascii_fold(s: str) -> str:
    """Strip European diacritics so 'São'/'Québec'/'Göppingen' match the
    plain-text foreign list. Case is preserved (NFKD only touches accents)."""
    folded = unicodedata.normalize("NFKD", s.replace("\u00df", "ss"))
    return "".join(c for c in folded if not unicodedata.combining(c))


def _in_us_bbox(lat, lon):
    try:
        lat = float(lat); lon = float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat) or math.isnan(lon):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if abs(lat) < 0.5 and abs(lon) < 0.5:               # null island
        return None
    if 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.5:  # continental US
        return True
    if 51.0 <= lat <= 72.0 and -170.0 <= lon <= -129.0:  # Alaska
        return True
    if 18.0 <= lat <= 23.0 and -161.0 <= lon <= -154.0:  # Hawaii
        return True
    return False


def is_us(location, lat=None, lon=None) -> bool:
    geo = _in_us_bbox(lat, lon)
    if geo is True:
        return True
    if geo is False:
        return False
    loc = (location or "").strip()
    if not loc:
        return True                                    # unspecified -> treat as US
    if NON_LATIN.search(loc):
        return False                                   # CJK / Hangul / Cyrillic etc.
    s = _ascii_fold(loc)                               # "São Paulo" -> "Sao Paulo"
    if US_BARE.search(s) or US_WORDS.search(s):
        return True                                    # explicit US wins
    if FOREIGN.search(s) or FOREIGN_CODE.search(s) or CA_PROVINCE.search(s) or EURES.search(s):
        return False                                   # clearly abroad
    if US_STATECODE.search(s) or US_STATENAME.search(s) or US_MISC.search(s):
        return True
    return True                                        # bare city / remote / unknown


# =====================================================================
#  Region classification (replaces the old US-only drop)
# =====================================================================
# Coarse buckets for the board's region buttons. Anything is_us() calls US
# (including unknown / bare city / global-remote) stays "US"; everything else
# is sorted into one of the six foreign buckets. Tune the token lists freely.
_R_INDIA = [
    "india", "bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "noida", "hyderabad", "pune", "chennai", "kolkata", "ahmedabad",
    "karnataka", "maharashtra", "haryana", "telangana", "gujarat",
]
_R_CANADA = [
    "canada", "canadian", "toronto", "vancouver", "montreal", "ottawa", "calgary",
    "edmonton", "winnipeg", "waterloo", "markham", "mississauga", "kitchener",
    "etobicoke", "ontario", "quebec", "british columbia", "alberta", "manitoba",
    "saskatchewan", "nova scotia",
]
_R_UKIE = [
    "united kingdom", "england", "scotland", "wales", "northern ireland", "ireland",
    "london", "manchester", "birmingham", "edinburgh", "glasgow", "belfast",
    "newcastle", "bristol", "leicester", "sheffield", "cardiff", "swansea",
    "cheltenham", "dublin", "midlothian", "leeds", "liverpool", "cambridge", "oxford",
]
_R_APAC = [
    "apac", "china", "taiwan", "hong kong", "japan", "singapore", "malaysia",
    "indonesia", "philippines", "vietnam", "thailand", "south korea", "korea",
    "pakistan", "bangladesh", "australia", "new zealand", "shanghai", "beijing",
    "shenzhen", "guangzhou", "hangzhou", "chengdu", "taipei", "tokyo", "osaka",
    "minato", "yokohama", "seoul", "gangnam", "busan", "kuala lumpur", "jakarta",
    "manila", "mandaluyong", "makati", "hanoi", "ho chi minh", "bangkok", "sydney",
    "melbourne", "brisbane", "perth", "adelaide", "canberra", "auckland",
    "wellington", "parnell",
]
_R_EUROPE = [
    "germany", "deutschland", "france", "spain", "espana", "italy", "italia",
    "netherlands", "nederland", "belgium", "belgie", "luxembourg", "switzerland",
    "schweiz", "suisse", "austria", "osterreich", "sweden", "sverige", "norway",
    "denmark", "danmark", "finland", "iceland", "poland", "polska", "portugal",
    "greece", "romania", "hungary", "czech", "czechia", "slovakia", "ukraine",
    "bulgaria", "serbia", "croatia", "slovenia", "cyprus", "armenia", "turkey",
    "russia", "berlin", "munich", "munchen", "hamburg", "frankfurt", "cologne",
    "koln", "stuttgart", "dusseldorf", "dortmund", "hannover", "regensburg",
    "goppingen", "neuss", "wolfsburg", "erlangen", "nuremberg", "wuppertal",
    "homburg", "huerth", "weissbach", "castellvi", "paris", "lyon", "bordeaux",
    "toulouse", "clichy", "issy", "marseille", "lille", "madrid", "barcelona",
    "valencia", "bilbao", "bellville", "sant cugat", "lisbon", "lisboa", "porto",
    "amsterdam", "rotterdam", "utrecht", "zwolle", "eindhoven", "wageningen",
    "majadahonda", "brussels", "antwerp", "zurich", "zuerich", "geneva", "geneve",
    "basel", "baar", "zug", "lausanne", "vienna", "wien", "graz", "stockholm",
    "gothenburg", "goteborg", "malmo", "copenhagen", "kobenhavn", "aarhus", "lyngby",
    "oslo", "bergen", "grimstad", "helsinki", "espoo", "otaniemi", "warsaw",
    "warszawa", "krakow", "wroclaw", "prague", "praha", "brno", "budapest",
    "bucharest", "sofia", "athens", "milan", "milano", "rome", "roma", "turin",
    "naples", "marbella", "istanbul", "ankara", "kyiv", "kiev", "yerevan", "tbilisi",
    "limassol", "bratislava", "ljubljana", "zagreb", "belgrade", "ile-de-france",
    "bavaria", "hessen", "nordrhein", "catalonia", "lombardy", "attica", "emea",
    "europe",
]


def _region_re(tokens):
    return re.compile(r"\b(" + "|".join(sorted(set(tokens), key=len, reverse=True)) + r")\b", re.I)


RE_INDIA = _region_re(_R_INDIA)
RE_CANADA = _region_re(_R_CANADA)
RE_UKIE = _region_re(_R_UKIE)
RE_APAC = _region_re(_R_APAC)
RE_EUROPE = _region_re(_R_EUROPE)

# Bare 2-letter country codes -> region (lowercase, matched on folded text).
_CODE_REGION = [
    (re.compile(r"(?:^|[,·\s(])(gb|uk|ie)(?:$|[,\s)])"), "UK & Ireland"),
    (re.compile(r"(?:^|[,·\s(])(ca)(?:$|[,\s)])"), "Canada"),
    (re.compile(r"(?:^|[,·\s(])(cn|hk|tw|jp|kr|sg|my|id|ph|vn|th|pk|bd|au|nz)(?:$|[,\s)])"), "APAC"),
    (re.compile(r"(?:^|[,·\s(])(de|fr|nl|es|it|se|no|dk|fi|pl|cz|hu|ro|gr|pt|be|at|ch|"
                r"lu|hr|si|rs|tr|ru|ua|am|ge|cy|bg)(?:$|[,\s)])"), "Europe"),
]


def region_of(location, lat=None, lon=None) -> str:
    """US (incl. unknown / bare city / remote) -> 'US'; otherwise one of
    'UK & Ireland', 'Europe', 'India', 'Canada', 'APAC', 'Rest of World'.
    Order matters: a Canadian 'London, ON' must beat the UK city 'London'."""
    if is_us(location, lat, lon):
        return "US"
    s = _ascii_fold((location or "").strip())
    if RE_INDIA.search(s):
        return "India"
    if RE_CANADA.search(s) or CA_PROVINCE.search(s):
        return "Canada"
    if RE_UKIE.search(s):
        return "UK & Ireland"
    if RE_APAC.search(s):
        return "APAC"
    if RE_EUROPE.search(s) or EURES.search(s):
        return "Europe"
    for rx, reg in _CODE_REGION:
        if rx.search(s):
            return reg
    return "Rest of World"


# =====================================================================
#  State, company name, salary, freshness
# =====================================================================
NAME2CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC",
}
STATE_NAME_RE = re.compile(r"\b(" + "|".join(sorted(NAME2CODE, key=len, reverse=True)) + r")\b", re.I)


def derive_state(location, is_remote=None) -> str:
    """Return a 2-letter state code, 'Remote', or 'US' (nationwide/unknown)."""
    s = _ascii_fold(location or "")
    m = US_STATECODE.search(s)
    if m:
        return m.group(1)
    m = STATE_NAME_RE.search(s)
    if m:
        return NAME2CODE[m.group(1).lower()]
    if is_remote is True or re.search(r"\bremote\b", s, re.I):
        return "Remote"
    return "US"


# Known ugly slugs / ATS hosts -> clean display names. Extend freely.
COMPANY_MAP = {
    "jpmc": "J.P. Morgan", "usbank": "U.S. Bank", "statestreet": "State Street",
    "thermofisher": "Thermo Fisher", "morningstar": "Morningstar",
    "wbd": "Warner Bros. Discovery", "gleanwork": "Glean", "glean": "Glean",
    "springhealth": "Spring Health", "hinge-health": "Hinge Health",
    "hinge health": "Hinge Health", "monzo": "Monzo", "monzoreferrals": "Monzo",
    "financialtimes": "Financial Times", "financialtimes33": "Financial Times",
    "bancopan": "Banco Pan", "a-place-for-mom": "A Place for Mom",
    "buyersedgeplatformrecruiting": "Buyers Edge Platform", "whoop": "WHOOP",
    "khanacademy": "Khan Academy", "honeybook": "HoneyBook",
    "datadog": "Datadog", "smartsheet": "Smartsheet", "stackadapt": "StackAdapt",
    "getyourguide": "GetYourGuide", "alphasense": "AlphaSense",
    "woven-by-toyota": "Woven by Toyota", "thomson reuters": "Thomson Reuters",
    "rocketcommunications": "Rocket Communications", "openai": "OpenAI",
    "vesync": "VeSync", "imanagecom": "iManage", "onetrust": "OneTrust",
    "beyondtrust": "BeyondTrust", "sentinellabs": "SentinelOne",
    "day1academies": "Day One Academies", "creditgenie": "Credit Genie",
    "slingshotaerospace": "Slingshot Aerospace", "evolutioniq": "EvolutionIQ",
    "cityandcountyofsanfrancisco1": "City of San Francisco",
    "allegisgroup": "Allegis Group", "interactivestrategies": "Interactive Strategies",
    "navapbc": "Nava PBC", "skylighthq": "Skylight", "understood": "Understood",
    "fetch": "Fetch", "abridge": "Abridge", "cursor": "Cursor", "figma": "Figma",
}
_ATS_INFRA = {"com", "net", "org", "io", "co", "edu", "gov", "ai", "us", "uk",
              "fa", "www", "careers", "jobs", "apply", "job", "recruiting", "hire",
              "talent", "work", "my", "external", "oraclecloud", "myworkdayjobs",
              "workday", "icims", "us2", "us6", "em2", "em3", "ocs", "saasfaprod1"}


def clean_company(raw) -> str:
    if not raw:
        return "Unknown"
    key = str(raw).strip().lower()
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    # Opaque Oracle / Workday tenant hosts: only recoverable via the map.
    if "oraclecloud" in key or "myworkdayjobs" in key:
        return COMPANY_MAP.get(key.split(".")[0], "Unknown")
    # Domain-style host (careers.adobe.com -> adobe).
    if "." in key and " " not in key:
        labels = [p for p in key.split(".") if p not in _ATS_INFRA and not p.isdigit()]
        cand = labels[0] if labels else ""
        if cand in COMPANY_MAP:
            return COMPANY_MAP[cand]
        key = cand or key
    key = re.sub(r"\d+$", "", key)                       # springhealth66 -> springhealth
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    name = re.sub(r"[-_]+", " ", key).strip()
    if not name:
        return "Unknown"
    # Title-case but keep short all-caps-ish tokens reasonable.
    return " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in name.split())


def format_salary(row) -> str:
    lo, hi = cell(row.get("salary_min")), cell(row.get("salary_max"))
    cur = cell(row.get("salary_currency"))
    sym = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CAD": "$"}.get(
        str(cur).upper() if cur else "", "$" if cur is None else "")

    def k(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v <= 0:
            return None
        return f"{sym}{round(v/1000)}k" if v >= 1000 else f"{sym}{round(v)}"

    klo, khi = k(lo), k(hi)
    if klo and khi and klo != khi:
        return f"{klo} to {khi}"
    if klo or khi:
        return (klo or khi) + "+"
    return ""


def age_days(iso) -> float:
    """Age of an ISO timestamp in days, or None if unparseable."""
    if not iso:
        return None
    try:
        s = str(iso).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


# =====================================================================
#  Tagging + row mapping
# =====================================================================
def classify_seniority(title: str) -> str:
    t = (title or "").lower()
    if re.search(r"\b(intern|internship|co-?op|apprentice)\b", t):
        return "junior"
    if re.search(r"\b(manager|mgr|head of|head,|director|vp|lead)\b", t):
        return "lead"
    if re.search(r"\b(principal|distinguished|staff)\b", t):
        return "staff"
    if re.search(r"\b(senior|sr\.?|snr)\b", t):
        return "senior"
    if re.search(r"\b(junior|jr\.?|associate|entry|graduate|\bi\b)\b", t):
        return "junior"
    return "mid"


def classify_location(location: str, is_remote=None) -> str:
    if is_remote is True:
        return "remote"
    loc = (location or "").lower()
    if "remote" in loc:
        return "remote"
    if "hybrid" in loc:
        return "hybrid"
    return "onsite"


TAG_RULES = [
    ("ai", r"\b(ai|ml|machine learning|genai|llm)\b"),
    ("quant", r"\bquantitative\b"),
    ("qual", r"\bqualitative\b"),
    ("mixed methods", r"\bmixed[\s-]?methods?\b"),
    ("research ops", r"research\s?op"),
    ("design systems", r"\bdesign systems?\b"),
    ("human factors", r"\bhuman factors\b"),
    ("accessibility", r"\b(accessibility|a11y)\b"),
]


def derive_tags(title: str):
    t = (title or "").lower()
    return [label for label, pat in TAG_RULES if re.search(pat, t)][:3]


def source_label(ats_type) -> str:
    if ats_type is None or (isinstance(ats_type, float) and math.isnan(ats_type)):
        return ""
    val = getattr(ats_type, "value", None) or str(ats_type)
    return str(val).replace("_", " ").title()


def cell(v):
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def parse_bool(v):
    v = cell(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return None


def parse_posted(v):
    v = cell(v)
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


def map_row(row: dict):
    title = str(cell(row.get("title")) or "").strip()
    if not title or not is_uxr(title):
        return None
    loc = str(cell(row.get("location")) or "").strip()
    region = region_of(loc, cell(row.get("lat")), cell(row.get("lon")))
    url = str(cell(row.get("url")) or "").strip()
    if not url:
        return None
    posted = parse_posted(row.get("posted_at"))
    age = age_days(posted)
    if age is not None and age > MAX_AGE_DAYS:          # too old, drop
        return None
    is_remote = parse_bool(row.get("is_remote"))
    ats = cell(row.get("ats_type"))
    return {
        "id": f"{source_label(ats)}-{cell(row.get('ats_id')) or url}",
        "title": title,
        "company": clean_company(cell(row.get("company"))),
        "location": loc,
        "region": region,
        "state": derive_state(loc, is_remote=is_remote) if region == "US" else "",
        "locationType": classify_location(loc, is_remote=is_remote),
        "seniority": classify_seniority(title),
        "tags": derive_tags(title),
        "salary": format_salary(row),
        "url": url,
        "source": source_label(ats),
        "postedAt": posted,
    }


def load_snapshot() -> pd.DataFrame:
    client = jh.Client()
    url = client.manifest.url_for_all(prefer_parquet=True)
    print(f"Downloading snapshot: {url}")
    resp = httpx.get(url, timeout=300.0, follow_redirects=True)
    resp.raise_for_status()
    content = resp.content
    del resp
    avail = set(pq.read_schema(io.BytesIO(content)).names)
    use = [c for c in WANT_COLS if c in avail] or None
    df = pd.read_parquet(io.BytesIO(content), columns=use)
    del content
    return df


def _posted_key(r):
    return r.get("postedAt") or ""


def records_from_df(df: pd.DataFrame):
    mask = df["title"].fillna("").str.contains(PRE_STEMS, case=False, regex=True)
    sub = df[mask]
    records = [r for row in sub.to_dict("records") if (r := map_row(row))]

    # Collapse the same role posted via multiple sources / cities. Key on
    # company + punctuation-stripped title; keep the freshest.
    best = {}
    for r in records:
        norm = re.sub(r"[^a-z0-9]+", " ", r["title"].lower()).strip()
        key = (r["company"].lower(), norm, r["region"])
        if key not in best or _posted_key(r) > _posted_key(best[key]):
            best[key] = r
    deduped = list(best.values())
    deduped.sort(key=_posted_key, reverse=True)
    return deduped


def main():
    df = load_snapshot()
    print(f"  snapshot rows: {len(df)}")
    records = records_from_df(df)
    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(records),
        "roles": records,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    by_src = Counter(r["source"] for r in records)
    by_region = Counter(r["region"] for r in records)
    by_state = Counter(r["state"] for r in records if r["region"] == "US")
    print(f"  UXR roles kept: {len(records)}")
    print("  by region: " + ", ".join(f"{k}={v}" for k, v in by_region.most_common()))
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in by_src.most_common(8)))
    print("  by US state: " + ", ".join(f"{k}={v}" for k, v in by_state.most_common(8)))
    print(f"Wrote {len(records)} roles to {OUTPUT_PATH} (updated {payload['updated']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
