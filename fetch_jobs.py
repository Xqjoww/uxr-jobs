#!/usr/bin/env python3
"""
Build jobs.json for the UXR Jobs Board from jobhive's hosted dataset.

Instead of watching a hand-picked list of companies, this loads jobhive's
hosted snapshot, which already aggregates jobs across thousands of companies
and every supported ATS, then keeps only the genuine UXR roles. No company
list to maintain.

Install (the GitHub Action does this for you):
    pip install "jobhive-py[scrapers]"
"""

import json
import re
import sys
from collections import Counter

import pandas as pd
import jobhive as jh

OUTPUT_PATH = "jobs.json"

# Optional: restrict by location text (e.g. "United States", "Remote").
# Leave as None for the full global dataset (maximum roles).
LOCATION_FILTER = None

# --- UXR title matching ----------------------------------------------
# UXR if the title names user/design research outright or is a research(er)
# role with a UX-ish qualifier (or research ops). Then knock out the common
# look-alikes: product/program managers, data science, and AI-lab ML research.
TITLE_STRONG    = re.compile(r"(ux|user|design)\s*-?\s*research", re.I)
TITLE_RESEARCHOPS = re.compile(r"research\s?ops|research operations", re.I)
TITLE_ROLE      = re.compile(r"\bresearch(er|ers)?\b", re.I)
TITLE_CONTEXT   = re.compile(
    r"\b(ux|user experience|user|design|product|qualitative|quantitative|"
    r"mixed[\s-]?methods?|usability|ethnograph|human factors|hci)\b", re.I)
TITLE_EXCLUDE   = re.compile(
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
        return None  # no apply link is useless on a board
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


def main():
    # prefer_parquet=False loads the CSV snapshot, so no pyarrow dependency.
    client = jh.Client(prefer_parquet=False)
    print("Loading jobhive hosted dataset (all companies, all ATS)...")
    df = client.search(location=LOCATION_FILTER) if LOCATION_FILTER else client.search()
    print(f"  dataset rows: {len(df)}")

    records = []
    for row in df.to_dict("records"):
        rec = map_row(row)
        if rec:
            records.append(rec)

    # de-duplicate by URL
    seen, deduped = set(), []
    for r in records:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)

    deduped.sort(key=lambda r: r.get("postedAt") or "", reverse=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    by_src = Counter(r["source"] for r in deduped)
    print(f"  UXR roles kept: {len(deduped)}")
    print("  by source: " + ", ".join(f"{k}={v}" for k, v in by_src.most_common()))
    print(f"Wrote {len(deduped)} roles to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
