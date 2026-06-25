#!/usr/bin/env python3
"""
Build jobs.json for the UXR Jobs Board from jobhive's hosted dataset.

Loads jobhive's hosted snapshot (which aggregates jobs across thousands of
companies and every supported ATS), keeps only genuine UXR roles, and writes
them out. No company list to maintain.

Memory notes (important on small CI runners):
  * Reads only the few columns we need from the parquet snapshot, never the
    full-text description column, so the load stays small.
  * Filters to research-titled rows inside pandas BEFORE expanding rows into
    Python objects, since every UXR title contains the word "research".

Resilience: jobhive's hosted manifest occasionally lists a new ATS platform
before the installed package knows about it, which otherwise crashes the load.
We strip unknown platforms from the manifest before validating.

Install (the GitHub Action does this for you):
    pip install "jobhive-py[scrapers]" pyarrow
"""

import io
import json
import re
import sys
from collections import Counter

import httpx
import pandas as pd
import pyarrow.parquet as pq
import jobhive as jh
from jobhive.manifest import Manifest, DEFAULT_MANIFEST_URL
from jobhive.models import ATSType

OUTPUT_PATH = "jobs.json"

# Optional: restrict by location text (e.g. "United States", "Remote").
# Leave as None for the full global dataset (the front-end has its own
# region filter, so you can keep everything here).
LOCATION_FILTER = None

# Only the columns the board needs. Skipping description/salary keeps the
# in-memory DataFrame small enough for a free CI runner.
WANT_COLS = ["title", "company", "location", "is_remote", "url",
             "posted_at", "ats_type", "ats_id"]

# =====================================================================
#  Make manifest parsing tolerant of platforms the installed package
#  doesn't recognize yet (e.g. a newly added ATS upstream).
# =====================================================================
_VALID_ATS = {a.value for a in ATSType}


def _strip_unknown_ats(data: dict) -> dict:
    ba = data.get("by_ats")
    if isinstance(ba, dict):
        unknown = sorted(k for k in ba if k not in _VALID_ATS)
        if unknown:
            print(f"  note: ignoring unknown ATS platform(s) in manifest: {', '.join(unknown)}")
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

# --- UXR title matching ----------------------------------------------
TITLE_STRONG      = re.compile(r"(ux|user|design)\s*-?\s*research", re.I)
TITLE_RESEARCHOPS = re.compile(r"research\s?ops|research operations", re.I)
TITLE_ROLE        = re.compile(r"\bresearch(er|ers)?\b", re.I)
TITLE_CONTEXT     = re.compile(
    r"\b(ux|user experience|user|design|product|qualitative|quantitative|"
    r"mixed[\s-]?methods?|usability|ethnograph|human factors|hci)\b", re.I)
TITLE_EXCLUDE     = re.compile(
    r"\b(market research|research scientist|clinical|research engineer|"
    r"equity research|investment|chemistry|biology|in vivo|"
    r"product manager|program manager|project manager|engineering manager|product management|"
    r"data scientist|data science|data analyst|data engineer|"
    r"model behavio|interpretability|pretraining|frontier model|"
    r"machine learning|reinforcement learning)\b", re.I)


def is_uxr(title: str) -> bool:
    t = title or ""
    if TITLE_EXCLUDE.search(t):
        return False
    if TITLE_STRONG.search(t) or TITLE_RESEARCHOPS.search(t):
        return True
    return bool(TITLE_ROLE.search(t) and TITLE_CONTEXT.search(t))


def classify_seniority(title: str) -> str:
    t = (title or "").lower()
    if re.search(r"\b(intern|internship|co-?op)\b", t):
        return "junior"
    if re.search(r"\b(manager|mgr|head of|head,|director|vp)\b", t):
        return "lead"
    if re.search(r"\b(principal|distinguished|staff)\b", t):
        return "staff"
    if re.search(r"\blead\b", t):
        return "lead"
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
    ("ai",             r"\b(ai|ml|machine learning|genai|llm)\b"),
    ("quant",          r"\b(quant|quantitative)\b"),
    ("qual",           r"\b(qual|qualitative)\b"),
    ("mixed methods",  r"\bmixed[\s-]?methods?\b"),
    ("growth",         r"\bgrowth\b"),
    ("design systems", r"\bdesign systems?\b"),
    ("platform",       r"\bplatform\b"),
    ("accessibility",  r"\b(accessibility|a11y)\b"),
    ("research ops",   r"\bresearch\s?ops|research operations\b"),
]


def derive_tags(title: str):
    t = (title or "").lower()
    return [label for label, pat in TAG_RULES if re.search(pat, t)][:3]


# --- DataFrame cell helpers (snapshot has NaNs for missing values) ----
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


def source_label(ats_type) -> str:
    ats_type = cell(ats_type)
    if not ats_type:
        return ""
    val = getattr(ats_type, "value", None) or str(ats_type)
    return str(val).replace("_", " ").title()


def map_row(row: dict):
    """Map one snapshot row to the board's JSON shape, or None if not UXR."""
    title = str(cell(row.get("title")) or "").strip()
    if not title or not is_uxr(title):
        return None
    url = str(cell(row.get("url")) or "").strip()
    if not url:
        return None
    loc = str(cell(row.get("location")) or "").strip()
    ats = row.get("ats_type")
    return {
        "id": f"{source_label(ats)}-{cell(row.get('ats_id')) or url}",
        "title": title,
        "company": str(cell(row.get("company")) or "").strip() or "Unknown",
        "location": loc,
        "locationType": classify_location(loc, is_remote=parse_bool(row.get("is_remote"))),
        "seniority": classify_seniority(title),
        "tags": derive_tags(title),
        "url": url,
        "source": source_label(ats),
        "postedAt": parse_posted(row.get("posted_at")),
    }


def load_snapshot() -> pd.DataFrame:
    """Download the snapshot parquet and read only the columns we need."""
    client = jh.Client()  # .manifest uses the tolerant fetch above
    url = client.manifest.url_for_all(prefer_parquet=True)
    print(f"Downloading snapshot: {url}")
    resp = httpx.get(url, timeout=300.0, follow_redirects=True)
    resp.raise_for_status()
    content = resp.content
    del resp
    avail = set(pq.read_schema(io.BytesIO(content)).names)  # footer only, cheap
    use = [c for c in WANT_COLS if c in avail] or None
    df = pd.read_parquet(io.BytesIO(content), columns=use)
    del content
    return df


def records_from_df(df: pd.DataFrame):
    # Lossless pre-filter: every UXR title contains "research", so this drops
    # the ~99% of unrelated rows before we expand anything into Python objects.
    mask = df["title"].fillna("").str.contains("research", case=False, regex=False)
    if LOCATION_FILTER:
        mask = mask & df["location"].fillna("").str.contains(LOCATION_FILTER, case=False, regex=False)
    sub = df[mask]

    records = []
    for row in sub.to_dict("records"):
        rec = map_row(row)
        if rec:
            records.append(rec)

    seen, deduped = set(), []
    for r in records:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)

    deduped.sort(key=lambda r: r.get("postedAt") or "", reverse=True)
    return deduped


def main():
    df = load_snapshot()
    print(f"  snapshot rows: {len(df)}")
    records = records_from_df(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    by_src = Counter(r["source"] for r in records)
    print(f"  UXR roles kept: {len(records)}")
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in by_src.most_common()))
    print(f"Wrote {len(records)} roles to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
