#!/usr/bin/env python3
"""
update_sitemap.py — Regenerates sitemap.xml with accurate lastmod dates.

For each page, picks the most recent git commit date across the page file
itself and any data dependency files (e.g. AI narrative JSON for index.html).
Called by .github/workflows/update-sitemap.yml on every push to main.

Output: sitemap.xml in the repo root.
"""

import subprocess
import os
from datetime import datetime, timezone

BASE_URL = "https://globalinvesting.github.io"

# Pages with their priority, changefreq, and optional data dependencies.
# Dependencies: if any listed file has a newer commit than the page itself,
# that date is used as lastmod instead (keeps the sitemap honest for pages
# with daily-updated data but infrequent HTML edits).
PAGES = [
    {
        "loc": "/",
        "file": "index.html",
        "deps": ["ai-analysis/narrative.json", "cot-data/cot_data.json",
                 "rates/USD.json", "ois-rates/rates.json"],
        "priority": "1.0",
        "changefreq": "daily",
    },
    {
        "loc": "/guide-gbpjpy-cot.html",
        "file": "guide-gbpjpy-cot.html",
        "deps": ["cot-data/cot_data.json"],
        "priority": "0.9",
        "changefreq": "weekly",
    },
    {
        "loc": "/guide-cot.html",
        "file": "guide-cot.html",
        "deps": ["cot-data/cot_data.json"],
        "priority": "0.9",
        "changefreq": "weekly",
    },
    {
        "loc": "/guide-dashboard.html",
        "file": "guide-dashboard.html",
        "deps": [],
        "priority": "0.8",
        "changefreq": "monthly",
    },
    {
        "loc": "/guide-economic-surprises.html",
        "file": "guide-economic-surprises.html",
        "deps": ["economic-data/surprises.json"],
        "priority": "0.8",
        "changefreq": "weekly",
    },
    {
        "loc": "/guide-cross-asset-risk.html",
        "file": "guide-cross-asset-risk.html",
        "deps": [],
        "priority": "0.8",
        "changefreq": "monthly",
    },
    {
        "loc": "/guide-rates-yield-curve.html",
        "file": "guide-rates-yield-curve.html",
        "deps": ["rates/USD.json", "ois-rates/rates.json"],
        "priority": "0.8",
        "changefreq": "weekly",
    },
    {
        "loc": "/guide-fx-liquidity.html",
        "file": "guide-fx-liquidity.html",
        "deps": [],
        "priority": "0.7",
        "changefreq": "monthly",
    },
    {
        "loc": "/guide-market-sentiment.html",
        "file": "guide-market-sentiment.html",
        "deps": ["sentiment-data/retail.json"],
        "priority": "0.7",
        "changefreq": "weekly",
    },
    {
        "loc": "/about.html",
        "file": "about.html",
        "deps": [],
        "priority": "0.5",
        "changefreq": "monthly",
    },
    {
        "loc": "/contact.html",
        "file": "contact.html",
        "deps": [],
        "priority": "0.4",
        "changefreq": "yearly",
    },
    {
        "loc": "/privacy.html",
        "file": "privacy.html",
        "deps": [],
        "priority": "0.3",
        "changefreq": "yearly",
    },
    {
        "loc": "/terms.html",
        "file": "terms.html",
        "deps": [],
        "priority": "0.3",
        "changefreq": "yearly",
    },
]


def git_last_commit_date(filepath: str) -> datetime | None:
    """Return the last commit date for a file as a UTC-aware datetime, or None."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", filepath],
            capture_output=True, text=True, check=True
        )
        iso = result.stdout.strip()
        if not iso:
            return None
        # Parse ISO 8601 with timezone offset
        dt = datetime.fromisoformat(iso)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_lastmod(page: dict) -> str:
    """Return YYYY-MM-DD string for the most recently modified file in the page + its deps."""
    files = [page["file"]] + page.get("deps", [])
    dates = []
    for f in files:
        if os.path.exists(f):
            d = git_last_commit_date(f)
            if d:
                dates.append(d)
    if dates:
        return max(dates).strftime("%Y-%m-%d")
    # Fallback: today
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_sitemap(pages: list[dict]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for page in pages:
        lastmod = get_lastmod(page)
        lines.append("  <url>")
        lines.append(f"    <loc>{BASE_URL}{page['loc']}</loc>")
        lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{page['changefreq']}</changefreq>")
        lines.append(f"    <priority>{page['priority']}</priority>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sitemap = build_sitemap(PAGES)
    with open("sitemap.xml", "w") as f:
        f.write(sitemap)
    print("✅ sitemap.xml written")
    # Print a summary
    for page in PAGES:
        lastmod = get_lastmod(page)
        print(f"  {lastmod}  {page['loc']}")
