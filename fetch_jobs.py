#!/usr/bin/env python3
"""
Fetch UXR roles via jobhive and write jobs.json for the Ghost job board.

jobhive (PyPI: jobhive-py) owns the fragile part: hitting each company's public
ATS endpoint and parsing it into a typed Job. This script keeps only the parts
that are yours: the UXR title filter, seniority/location/tag tagging, and the
exact JSON shape your board reads.

Install (the GitHub Action does this for you):
    pip install "jobhive-py[scrapers]"
"""

import json
import re
import sys
from jobhive.scrapers import GreenhouseScraper, LeverScraper, AshbyScraper

# Map a platform name to its jobhive scraper. jobhive ships ~50 of these
# (Workday, SmartRecruiters, Recruitee, Personio, Workable...), so adding a
# platform later is one import + one line here.
SCRAPERS = {
    "greenhouse": GreenhouseScraper,
    "lever": LeverScraper,
    "ashby": AshbyScraper,
}

# =====================================================================
#  EDIT THIS: companies on the board. token = the slug in the careers URL
#  (boards.greenhouse.io/<token>, jobs.lever.co/<token>, jobs.ashbyhq.com/<token>).
#  The summary printed after each run shows which resolved and how many
#  UXR roles each returned, so prune from there.
# =====================================================================
COMPANIES = [
    {"platform": "greenhouse", "token": "anthropic",  "name": "Anthropic"},
    {"platform": "greenhouse", "token": "stripe",     "name": "Stripe"},
    {"platform": "greenhouse", "token": "figma",      "name": "Figma"},
    {"platform": "greenhouse", "token": "databricks", "name": "Databricks"},
    {"platform": "greenhouse", "token": "discord",    "name": "Discord"},
    {"platform": "ashby",      "token": "ramp",       "name": "Ramp"},
    {"platform": "ashby",      "token": "notion",     "name": "Notion"},
    {"platform": "ashby",      "token": "linear",     "name": "Linear"},
    {"platform": "ashby",      "token": "vanta",      "name": "Vanta"},
    {"platform": "lever",      "token": "spotify",    "name": "Spotify"},
    {"platform": "lever",      "token": "netflix",    "name": "Netflix"},
]

OUTPUT_PATH = "jobs.json"

# --- UXR title matching (yours) --------------------------------------
TITLE_STRONG  = re.compile(r"(ux|user|design)\s*-?\s*research", re.I)
TITLE_ROLE    = re.compile(r"\bresearch(er|ers)?\b", re.I)
TITLE_CONTEXT = re.compile(
    r"\b(ux|user experience|user|design|product|qualitative|quantitative|"
    r"mixed[\s-]?methods?|insights|human factors|hci)\b", re.I)
TITLE_EXCLUDE = re.compile(
    r"\b(market research|research scientist|clinical|research engineer|"
    r"equity research|investment|chemistry|biology|in vivo)\b", re.I)


def is_uxr(title: str) -> bool:
    t = title or ""
    if TITLE_EXCLUDE.search(t):
        return False
    if TITLE_STRONG.search(t):
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


def source_label(ats_type) -> str:
    val = getattr(ats_type, "value", None) or str(ats_type)
    return str(val).replace("_", " ").title()


# --- the thin mapper: jobhive Job -> your board's JSON shape ----------
def map_job(job, display_name: str):
    """Return a board record, or None if it is not a UXR role."""
    if not is_uxr(job.title):
        return None
    return {
        "id": f"{source_label(job.ats_type)}-{job.ats_id}",
        "title": job.title.strip(),
        "company": job.company or display_name,
        "location": job.location or "",
        "locationType": classify_location(job.location, is_remote=job.is_remote),
        "seniority": classify_seniority(job.title),
        "tags": derive_tags(job.title),
        "url": str(job.url),
        "source": source_label(job.ats_type),
        "postedAt": job.posted_at.isoformat() if job.posted_at else "",
    }


def main():
    records = []
    print("Fetching UXR roles via jobhive\n" + "-" * 52)
    for c in COMPANIES:
        platform, token, name = c["platform"], c["token"], c["name"]
        scraper_cls = SCRAPERS.get(platform)
        if scraper_cls is None:
            print(f"  SKIP {name:<14} unknown platform '{platform}'")
            continue
        try:
            jobs = scraper_cls(token).fetch()              # hits the public ATS API
            uxr = [r for j in jobs if (r := map_job(j, name))]
            records.extend(uxr)
            print(f"  ok   {name:<14} {platform:<11} {len(jobs):>4} open, {len(uxr):>3} UXR")
        except Exception as e:  # one bad slug shouldn't kill the run
            print(f"  FAIL {name:<14} {platform:<11} {type(e).__name__}: {e}")

    # de-duplicate across companies by URL
    seen, deduped = set(), []
    for r in records:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)

    deduped.sort(key=lambda r: r.get("postedAt") or "", reverse=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print("-" * 52)
    print(f"Wrote {len(deduped)} roles to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
