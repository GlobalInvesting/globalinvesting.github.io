"""
update_sitemap.py — Regenerates sitemap.xml with accurate lastmod dates.

For each page, picks the most recent git commit date across the page file
itself and any data dependency files (e.g. AI narrative JSON for index.html).
Called by .github/workflows/update-sitemap.yml on every push to main.

Output: sitemap.xml in the repo root.
"""

import subprocess
import os
import re
import html
from datetime import datetime, timezone
from xml.sax.saxutils import escape as xml_escape

BASE_URL = "https://globalinvesting.github.io"

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
        "loc": "/guide-mt5-ea.html",
        "file": "guide-mt5-ea.html",
        "deps": [],
        "priority": "0.9",
        "changefreq": "monthly",
    },
    {
        "loc": "/access.html",
        "file": "access.html",
        "deps": [],
        "priority": "0.9",
        "changefreq": "monthly",
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
        "loc": "/guide-fundamental-analysis.html",
        "file": "guide-fundamental-analysis.html",
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
        "loc": "/guide-csi-indicator.html",
        "file": "guide-csi-indicator.html",
        "deps": [],
        "priority": "0.7",
        "changefreq": "monthly",
    },
    {
        "loc": "/guide-monte-carlo-simulator.html",
        "file": "guide-monte-carlo-simulator.html",
        "deps": [],
        "priority": "0.7",
        "changefreq": "monthly",
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

IMG_TAG_RE = re.compile(r"<img\b[^>]*>")
SRC_RE = re.compile(r'src="([^"]*)"')
ALT_RE = re.compile(r'alt="([^"]*)"')


def scan_page_images(filepath: str) -> list[dict]:
    """Scan a page's HTML for real content screenshots (assets/screenshot-*).

    Skips favicons/icons/logos automatically since those never match the
    'screenshot' filename convention used for every real UI capture in this
    repo. Returns a de-duplicated list of {"loc", "caption"} dicts in
    document order.
    """
    if not os.path.exists(filepath):
        return []
    txt = open(filepath, encoding="utf-8").read()
    seen_srcs = set()
    images = []
    for tag in IMG_TAG_RE.findall(txt):
        src_m = SRC_RE.search(tag)
        if not src_m or "screenshot" not in src_m.group(1):
            continue
        src = src_m.group(1)
        if src in seen_srcs:
            continue
        seen_srcs.add(src)
        alt_m = ALT_RE.search(tag)
        caption = html.unescape(alt_m.group(1)) if alt_m else ""
        images.append({"loc": f"{BASE_URL}/{src}", "caption": caption})
    return images


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_sitemap(pages: list[dict]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
        ' xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">',
    ]
    for page in pages:
        lastmod = get_lastmod(page)
        lines.append("  <url>")
        lines.append(f"    <loc>{BASE_URL}{page['loc']}</loc>")
        lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{page['changefreq']}</changefreq>")
        lines.append(f"    <priority>{page['priority']}</priority>")
        for img in scan_page_images(page["file"]):
            lines.append("    <image:image>")
            lines.append(f"      <image:loc>{xml_escape(img['loc'])}</image:loc>")
            if img["caption"]:
                lines.append(
                    f"      <image:caption>{xml_escape(img['caption'])}</image:caption>"
                )
            lines.append("    </image:image>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sitemap = build_sitemap(PAGES)
    with open("sitemap.xml", "w") as f:
        f.write(sitemap)
    print("✅ sitemap.xml written")
    for page in PAGES:
        lastmod = get_lastmod(page)
        print(f"  {lastmod}  {page['loc']}")
