#!/usr/bin/env python3
"""
fetch_bank_research.py — v1.4
Fetches institutional FX research notes from public RSS feeds and writes
research-data/bank-research.json.

Design principles:
  · Zero LLM calls — metadata only (title, bank, URL, currency tags, date).
  · No content reproduction — complies with copyright. Body text is never
    stored or displayed; the "Read full note" link opens the source directly.
  · Industry standard: mirrors Bloomberg Research Monitor row layout.
    Each note shows: bank badge · series label · headline · currency tags ·
    category chip · relative age. Drawer shows: bank full name · author ·
    pairs · external link.

Changes v1.4:
  · EXCERPT LENGTH: summary buffer raised from 600 → 1200 chars; excerpt
    limit raised from 400 → 800 chars. Prior limit was truncating RSS
    descriptions prematurely — sources like Saxo and CME provide 400-700 char
    summaries in their RSS feeds. Publishers include <description> in RSS
    specifically for aggregators to display as a teaser (Bloomberg/Reuters
    standard). 800 chars provides meaningful context without reproducing
    full article body. Trimmed at word boundary with "…" if not a complete
    sentence.

Changes v1.2:
  · SAXO BUG FIX: bank_counts[bank] = 0 was resetting the counter every time a future
    completed — so when the second Saxo feed (0 items) finished after the first (7 items),
    the counter reset to 0 and all 7 items were lost. Fixed with setdefault(bank, 0) which
    only initialises the counter if the bank hasn't been seen yet.
  · MUFG DEBUG: Added html[:600] debug print to GA log so we can see the live page
    structure. Added 2 new extraction patterns (1b: data-title/aria-label attributes;
    2b: any /[section]/fx/ path, catches /insights/fx/, /research/fx/, etc.).
  · UBS: Added second URL fallback (UBS CIO Daily Updates) in case the investment bank
    RSS returns empty. Both try in parallel; results are deduplicated by item_id.

Changes v1.1:
  · ING: URL changed from /market/fx/feed/ (timeout in GA — likely geo-restricted) to
    /rss/ (general ING Think RSS). Added FX relevance filter to discard non-FX articles
    from the broader feed (economics, banking, sustainability).
  · Saxo: URL changed from /content-hub/rss (malformed feed) to two specific feeds:
    /rss/trade-views (trade recommendations) and /rss/articles (macro commentary).
    Both are confirmed active and return valid XML.
  · MUFG: Improved HTML scraping with more robust multi-pattern link extraction.
    Added fallback patterns for absolute URLs and alternative tag structures.
  · CME Group FX: added cmegroup.com/rss/fx-video-commentary.rss — institutional
    FX market commentary from CME Group, the world's largest FX futures exchange.
    Trade ideas and market views with direct futures market context.
  · UBS Global Research: added ubs.com research RSS — sell-side research from one
    of the largest FX desks globally. Filtered for FX relevance.

Sources (all free, no registration required):
  ING Think         — think.ing.com/rss/ (general RSS, FX-filtered)
  Saxo Trade Views  — home.saxo/insights/content-hub/rss/trade-views
  Saxo Articles     — home.saxo/insights/content-hub/rss/articles
  CME FX Commentary — cmegroup.com/rss/fx-video-commentary.rss
  UBS Research      — ubs.com global research RSS
  BIS Speeches      — bis.org/doclist/cbspeeches.rss
  MUFG Research     — mufgresearch.com/fx/ (HTML index scraping)

Currency detection: regex on title + description, no NLP, no external APIs.
Series detection: pattern matching on title prefix ("FX Daily:", "FX Weekly:").
Category: inferred from series label + keywords (trade_idea / macro / technical / flow).

Output schema (research-data/bank-research.json):
  {
    "updated_utc":   "ISO-8601",
    "updated_label": "HH:MM UTC",
    "total":         N,
    "sources":       ["ING", "Saxo", ...],
    "items": [
      {
        "bank":       "ING",
        "bank_full":  "ING Think",
        "series":     "FX Daily",
        "title":      "FX Daily: Dollar holds as Fed holds firm",
        "author":     "Francesco Pesole",
        "url":        "https://think.ing.com/...",
        "currencies": ["USD", "EUR", "GBP"],
        "pairs":      ["EUR/USD", "GBP/USD"],
        "category":   "macro",
        "ts":         1748995200000,
        "published":  "2026-06-03T06:45:00Z"
      }
    ]
  }

Context injection: also writes ai-analysis/context_snapshot.json with a
research_context block (top 20 items, ≤48h) so the existing narrative
pipeline picks up bank views without an additional LLM call.
"""

import json
import re
import hashlib
import os
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
from dateutil import parser as dateparser

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE          = "research-data/bank-research.json"
CONTEXT_SNAPSHOT     = "ai-analysis/context_snapshot.json"
MAX_AGE_DAYS         = 7          # keep notes up to 7 days (research has longer shelf life)
MAX_PER_BANK         = 10         # cap per source to keep JSON lean
MAX_ITEMS_TOTAL      = 50
FETCH_TIMEOUT        = 15
FETCH_WORKERS        = 6

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
RSS_SOURCES = [
    # ── ING Think — general RSS, FX-filtered (v1.1: /rss/ replaces /market/fx/feed/ which
    # timed out in GitHub Actions — likely geo-restricted. /rss/ is the root feed and is
    # reachable. FX relevance filter applied post-fetch to discard non-FX articles.)
    {
        "bank":      "ING",
        "bank_full": "ING Think",
        "url":       "https://think.ing.com/rss/",
        "type":      "rss",
        "fx_filter": True,   # apply is_fx_relevant() — broad feed includes non-FX content
    },
    # ── Saxo Bank — trade views feed (v1.1: replaces /content-hub/rss which was malformed)
    # SaxoStrats trade recommendations with directional bias and entry/exit context.
    {
        "bank":      "Saxo",
        "bank_full": "Saxo Bank (SaxoStrats) — Trade Views",
        "url":       "https://www.home.saxo/insights/content-hub/rss/trade-views",
        "type":      "rss",
        "fx_filter": True,
    },
    # ── Saxo Bank — macro articles (v1.1: second Saxo feed for broader macro commentary)
    {
        "bank":      "Saxo",
        "bank_full": "Saxo Bank (SaxoStrats) — Articles",
        "url":       "https://www.home.saxo/insights/content-hub/rss/articles",
        "type":      "rss",
        "fx_filter": True,
    },
    # ── CME Group FX Video Commentary — institutional FX market views from the operator
    # of the world's largest FX futures exchange. Includes trade ideas with futures context.
    {
        "bank":      "CME",
        "bank_full": "CME Group FX",
        "url":       "https://www.cmegroup.com/rss/fx-video-commentary.rss",
        "type":      "rss",
        "fx_filter": False,  # all content is FX-focused
    },
    # ── UBS Global Research — sell-side research from one of the largest FX desks globally.
    # v1.2: Two UBS feeds tried in parallel; deduplicated by item_id.
    {
        "bank":      "UBS",
        "bank_full": "UBS Global Research",
        "url":       "https://www.ubs.com/global/en/investment-bank/insights-and-data/global-research/_jcr_content/root/contentarea/mainpar/toplevelgrid_copy/col_1/responsivepodcast.rss20.xml?campID=RSS",
        "type":      "rss",
        "fx_filter": True,
    },
    # UBS CIO Daily Updates — Chief Investment Office macro views. More broadly accessible
    # than the IB research RSS. FX relevance filter applied.
    {
        "bank":      "UBS",
        "bank_full": "UBS CIO — Chief Investment Office",
        "url":       "https://www.ubs.com/global/en/wealth-management/chief-investment-office/daily-updates.rss",
        "type":      "rss",
        "fx_filter": True,
    },
    # ── BIS Central Banker Speeches — pre-release policy signals from global CBs.
    # No fx_filter — all CB speeches are relevant to the currencies they govern.
    {
        "bank":      "BIS",
        "bank_full": "Bank for International Settlements",
        "url":       "https://www.bis.org/doclist/cbspeeches.rss",
        "type":      "rss",
        "fx_filter": False,
    },
    # ── MUFG Research — HTML index scraping (no RSS available)
    {
        "bank":      "MUFG",
        "bank_full": "MUFG Research (Mitsubishi UFJ Financial Group)",
        "url":       "https://www.mufgresearch.com/fx/",
        "type":      "html_index",
        "fx_filter": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CURRENCY / PAIR DETECTION (regex, no LLM)
# ─────────────────────────────────────────────────────────────────────────────
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "NZD/USD", "USD/CAD", "USD/CHF",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD", "GBP/AUD",
    "EUR/CAD", "GBP/CHF", "NZD/JPY", "CAD/JPY", "EUR/NZD", "GBP/NZD",
    "EUR/CHF", "AUD/NZD", "GBP/CAD", "CHF/JPY", "NZD/CHF", "AUD/CAD", "AUD/CHF",
]
PAIR_RE = re.compile(
    r"\b(" + "|".join(p.replace("/", r"[/]?") for p in PAIRS) + r")\b",
    re.IGNORECASE,
)

CURRENCY_KEYWORDS = {
    "USD": ["usd", "dollar", "fed", "federal reserve", "fomc", "powell", "dxy", "greenback"],
    "EUR": ["eur", "euro", "ecb", "eurozone", "lagarde", "bund", "euro area"],
    "GBP": ["gbp", "sterling", "pound", "bank of england", "boe", "bailey", "gilts"],
    "JPY": ["jpy", "yen", "boj", "bank of japan", "ueda", "nikkei"],
    "AUD": ["aud", "aussie", "rba", "reserve bank of australia", "bullock"],
    "CAD": ["cad", "loonie", "bank of canada", "boc", "macklem"],
    "CHF": ["chf", "franc", "snb", "swiss national bank", "schlegel"],
    "NZD": ["nzd", "kiwi", "rbnz", "reserve bank of new zealand"],
}

def detect_pairs(text: str) -> list:
    found = []
    seen = set()
    for m in PAIR_RE.finditer(text):
        p = m.group(0).upper().replace("", "/") if "/" not in m.group(0) else m.group(0).upper()
        # normalise: ensure slash
        normed = p[:3] + "/" + p[-3:] if "/" not in p else p
        if normed not in seen:
            seen.add(normed)
            found.append(normed)
    return found[:6]

def detect_currencies(title: str, body: str) -> list:
    text = (title + " " + body).lower()
    found = []
    for cur, kws in CURRENCY_KEYWORDS.items():
        if any(kw in text for kw in kws):
            found.append(cur)
    # Also add currencies from detected pairs
    for pair in detect_pairs(title + " " + body):
        for cur in [pair[:3], pair[4:]]:
            if cur not in found:
                found.append(cur)
    return found[:6]

# ─────────────────────────────────────────────────────────────────────────────
# SERIES / CATEGORY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
SERIES_PATTERNS = [
    (re.compile(r"^FX\s+Daily[\s:–—]", re.I),   "FX Daily",   "macro"),
    (re.compile(r"^FX\s+Weekly[\s:–—]", re.I),  "FX Weekly",  "macro"),
    (re.compile(r"^FX\s+Talking[\s:–—]", re.I), "FX Talking", "macro"),
    (re.compile(r"^FX\s+Snapshot[\s:–—]", re.I),"FX Snapshot","macro"),
    (re.compile(r"^FX\s+Monthly[\s:–—]", re.I), "FX Monthly", "macro"),
    (re.compile(r"^FX\s+Outlook[\s:–—]", re.I), "FX Outlook", "macro"),
    (re.compile(r"^FX\s+Strategy[\s:–—]", re.I),"FX Strategy","macro"),
    (re.compile(r"^Market\s+Call[\s:–—]", re.I),"Market Call","macro"),
    (re.compile(r"^Trade\s+(Idea|Alert)[\s:–—]", re.I), "Trade Idea", "trade_idea"),
    (re.compile(r"\btrade\s+idea\b", re.I),      "Trade Idea", "trade_idea"),
    (re.compile(r"\bflow\s+(watch|monitor|report)\b", re.I), "Flow", "flow"),
]

TRADE_IDEA_KW = re.compile(
    r"\b(long|short|buy|sell|entry|target|stop.loss|tp\b|sl\b|trade\s+idea|position)\b",
    re.I,
)
TECHNICAL_KW = re.compile(
    r"\b(support|resistance|moving\s+average|rsi|macd|fibonacci|trend\s+line|breakout|"
    r"technical\s+analysis|chart\s+pattern|elliott\s+wave)\b",
    re.I,
)

def detect_series_category(title: str, body: str) -> tuple:
    for pattern, series, category in SERIES_PATTERNS:
        if pattern.search(title):
            return series, category
    # Fallback category detection
    text = title + " " + body
    if TRADE_IDEA_KW.search(text):
        return "", "trade_idea"
    if TECHNICAL_KW.search(text):
        return "", "technical"
    return "", "macro"

# ─────────────────────────────────────────────────────────────────────────────
# AUTHOR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_author(entry) -> str:
    """Extracts author from feedparser entry fields, gracefully."""
    for attr in ("author", "author_detail"):
        val = getattr(entry, attr, None)
        if not val:
            continue
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            return val.get("name", "").strip()
    tags = getattr(entry, "tags", [])
    for tag in tags:
        if isinstance(tag, dict) and tag.get("scheme") == "author":
            return tag.get("term", "").strip()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# FX RELEVANCE FILTER
# Applied to broad feeds (ING Think general, Saxo all-content, UBS) to discard
# non-FX articles (banking regulation, sustainability, credit, equity research).
# ─────────────────────────────────────────────────────────────────────────────
FX_RELEVANCE_KW = [
    "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
    "dollar", "euro", "pound", "sterling", "yen", "franc",
    "forex", " fx ", "currency", "currencies", "exchange rate",
    "fed", "ecb", "boe", "boj", "rba", "boc", "snb", "rbnz",
    "federal reserve", "central bank", "bank of england", "bank of japan",
    "interest rate", "monetary policy", "rate hike", "rate cut",
    "inflation", "cpi", "gdp", "nonfarm", "payroll", "fomc",
    "treasury yield", "yield curve", "carry trade",
    "eur/usd", "gbp/usd", "usd/jpy", "aud/usd", "usd/cad", "usd/chf",
    "g10", "g7", "fx strategy", "fx outlook", "fx daily", "fx weekly",
    "hawkish", "dovish", "tightening", "easing",
]

def is_fx_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FX_RELEVANCE_KW)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_entry_date(entry):
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)

def item_id(bank: str, title: str) -> str:
    return hashlib.md5(f"{bank}:{title}".encode()).hexdigest()[:12]

def fetch_rss(source: dict, cutoff: datetime) -> list:
    """Fetches and parses an RSS feed. Returns list of structured items."""
    bank      = source["bank"]
    bank_full = source["bank_full"]
    url       = source["url"]
    fx_filter = source.get("fx_filter", False)
    results   = []

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GlobalInvesting-ResearchBot/1.0)"},
            timeout=FETCH_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            print(f"  [{bank}] HTTP {resp.status_code} — skipping")
            return []

        feed = feedparser.parse(resp.content)
        if feed.bozo and not feed.entries:
            print(f"  [{bank}] Malformed feed — skipping")
            return []

        for entry in feed.entries:
            title = clean_html(getattr(entry, "title", ""))
            if not title or len(title) < 10:
                continue

            pub_date = parse_entry_date(entry)
            if pub_date < cutoff:
                continue

            # Summary — used for currency/pair extraction, fx_filter, and excerpt
            summary = clean_html(
                getattr(entry, "summary", "") or
                getattr(entry, "description", "") or ""
            )[:1200]

            # Apply FX relevance filter for broad feeds
            if fx_filter and not is_fx_relevant(title, summary):
                continue

            link   = getattr(entry, "link", "") or getattr(entry, "id", "")
            author = extract_author(entry)

            series, category = detect_series_category(title, summary)
            currencies       = detect_currencies(title, summary)
            pairs            = detect_pairs(title + " " + summary)

            ts = int(pub_date.timestamp() * 1000)

            # Excerpt — up to 800 chars of RSS description, trimmed at word boundary.
            # Publishers include <description> in RSS specifically for aggregators to
            # display as a teaser — this is the intended use (Bloomberg/Reuters standard).
            excerpt = summary[:800].strip()
            if excerpt and not excerpt.endswith(('.', '…', '!', '?')):
                excerpt = excerpt.rsplit(' ', 1)[0] + '…'

            results.append({
                "id":        item_id(bank, title),
                "bank":      bank,
                "bank_full": bank_full,
                "series":    series,
                "title":     title,
                "excerpt":   excerpt,
                "author":    author,
                "url":       link,
                "currencies": currencies,
                "pairs":     pairs,
                "category":  category,
                "ts":        ts,
                "published": pub_date.isoformat(),
            })

        print(f"  [{bank}] {len(results)} items from RSS ({url.split('/')[2]})")

    except Exception as e:
        print(f"  [{bank}] Exception: {e}")

    return results

def fetch_mufg_html(source: dict, cutoff: datetime) -> list:
    """
    Scrapes MUFG Research FX index page.
    No RSS — discovers article links from the HTML index using multiple patterns.
    Title and URL only — no body content stored (copyright compliant).

    v1.1: Multiple extraction patterns added:
      1. Relative /fx/[slug]/ href with adjacent link text
      2. Absolute https://www.mufgresearch.com/fx/[slug]/ href
      3. <h2>/<h3>/<h4> headline tags with sibling/parent <a> link
    """
    bank      = source["bank"]
    bank_full = source["bank_full"]
    url       = source["url"]
    results   = []
    base_url  = "https://www.mufgresearch.com"

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=FETCH_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            print(f"  [{bank}] HTTP {resp.status_code} — skipping")
            return []

        html = resp.text

        # v1.2: debug — print first 600 chars of HTML so we can diagnose structure changes in GA
        print(f"  [{bank}] DEBUG html[:600]: {html[:600]!r}")

        seen_slugs = set()
        candidates = []  # list of (slug_or_url, title)

        # ── Pattern 1: relative href="/fx/[slug]/" with link text (original)
        for m in re.finditer(
            r'href=["\'](/fx/[a-z0-9][a-z0-9\-]+/)["\'][^>]*>([^<]{8,150})',
            html, re.I,
        ):
            slug, title = m.group(1), clean_html(m.group(2)).strip()
            if slug not in seen_slugs and len(title) >= 8:
                seen_slugs.add(slug)
                candidates.append((base_url + slug, title))

        # ── Pattern 1b: href="/fx/[slug]/" anywhere — title in data-title or aria-label attr
        for m in re.finditer(
            r'href=["\'](/fx/[a-z0-9][a-z0-9\-]+/)["\'][^>]*(?:data-title|aria-label|title)=["\']([^"\']{8,150})["\']',
            html, re.I,
        ):
            slug, title = m.group(1), clean_html(m.group(2)).strip()
            if slug not in seen_slugs and len(title) >= 8:
                seen_slugs.add(slug)
                candidates.append((base_url + slug, title))

        # ── Pattern 2: absolute href with mufgresearch.com/fx/
        for m in re.finditer(
            r'href=["\']https?://(?:www\.)?mufgresearch\.com(/fx/[a-z0-9][a-z0-9\-]+/?)["\'][^>]*>([^<]{8,150})',
            html, re.I,
        ):
            slug, title = m.group(1), clean_html(m.group(2)).strip()
            if slug not in seen_slugs and len(title) >= 8:
                seen_slugs.add(slug)
                candidates.append((base_url + slug, title))

        # ── Pattern 2b: any mufgresearch.com/[any-path]/ href (catches /insights/fx/, /research/fx/, etc.)
        for m in re.finditer(
            r'href=["\'](?:https?://(?:www\.)?mufgresearch\.com)?(/[a-z0-9\-]+/[a-z0-9][a-z0-9\-]+/)["\'][^>]*>\s*([^<]{8,150})',
            html, re.I,
        ):
            slug, title = m.group(1), clean_html(m.group(2)).strip()
            # Must contain "fx" or research keyword in slug
            if any(kw in slug.lower() for kw in ["fx", "foreign", "currency", "research"]):
                if slug not in seen_slugs and len(title) >= 8:
                    seen_slugs.add(slug)
                    candidates.append((base_url + slug, title))

        # ── Pattern 3: article/h-tag headline with nearby /fx/ link
        # Captures <a href="/fx/..."><h2>Title</h2></a> or <h2><a href="/fx/...">Title</a></h2>
        for m in re.finditer(
            r'<(?:h[234]|a\s[^>]*href=["\'](?:https?://www\.mufgresearch\.com)?(/fx/[a-z0-9][a-z0-9\-]+/)["\'])[^>]*>\s*'
            r'(?:<a[^>]*>)?\s*([A-Z][^<]{10,150}?)\s*(?:</a>)?\s*</(?:h[234]|a)>',
            html, re.I | re.DOTALL,
        ):
            slug_or_title = m.group(1) or ""
            title = clean_html(m.group(2)).strip()
            if slug_or_title and slug_or_title not in seen_slugs and len(title) >= 8:
                seen_slugs.add(slug_or_title)
                candidates.append((base_url + slug_or_title, title))

        if not candidates:
            print(f"  [{bank}] 0 candidates found — page structure may have changed")
            return []

        for full_url, title in candidates:
            # Extract slug from URL for date detection
            slug = full_url.replace(base_url, "")

            # Date extraction from slug: fx-weekly-26-may-2026 or fx-daily-2026-05-26
            pub_date = datetime.now(timezone.utc)
            date_m = re.search(
                r"(\d{1,2})[\-_]([a-z]+)[\-_](\d{4})|(\d{4})[\-_](\d{2})[\-_](\d{2})",
                slug, re.I,
            )
            if date_m:
                try:
                    if date_m.group(1):
                        parsed = dateparser.parse(
                            f"{date_m.group(1)} {date_m.group(2)} {date_m.group(3)}"
                        )
                        if parsed:
                            pub_date = parsed.replace(tzinfo=timezone.utc)
                    else:
                        pub_date = datetime(
                            int(date_m.group(4)),
                            int(date_m.group(5)),
                            int(date_m.group(6)),
                            tzinfo=timezone.utc,
                        )
                except Exception:
                    pass

            if pub_date < cutoff:
                continue

            series, category = detect_series_category(title, "")
            currencies       = detect_currencies(title, "")
            pairs            = detect_pairs(title)
            ts               = int(pub_date.timestamp() * 1000)

            results.append({
                "id":        item_id(bank, title),
                "bank":      bank,
                "bank_full": bank_full,
                "series":    series,
                "title":     title,
                "excerpt":   "",
                "author":    "",
                "url":       full_url,
                "currencies": currencies,
                "pairs":     pairs,
                "category":  category,
                "ts":        ts,
                "published": pub_date.isoformat(),
            })

        print(f"  [{bank}] {len(results)} items from HTML index ({len(candidates)} candidates found)")

    except Exception as e:
        print(f"  [{bank}] Exception: {e}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(days=MAX_AGE_DAYS)

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_bank_research.py v1.4")
    print(f"  Fetching {len(RSS_SOURCES)} sources in parallel (workers={FETCH_WORKERS})...")

    all_items   = []
    seen_ids    = set()
    bank_counts = {}

    def fetch_source(src):
        if src["type"] == "html_index":
            return fetch_mufg_html(src, cutoff)
        return fetch_rss(src, cutoff)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_source, s): s for s in RSS_SOURCES}
        for future in as_completed(futures):
            try:
                items = future.result()
            except Exception as e:
                src = futures[future]
                print(f"  [{src['bank']}] Unhandled exception: {e}")
                items = []

            src  = futures[future]
            bank = src["bank"]
            bank_counts.setdefault(bank, 0)  # v1.2: do NOT reset — multiple feeds per bank

            for item in items:
                if item["id"] in seen_ids:
                    continue
                if bank_counts[bank] >= MAX_PER_BANK:
                    continue
                seen_ids.add(item["id"])
                all_items.append(item)
                bank_counts[bank] += 1

    # Sort newest first
    all_items.sort(key=lambda x: x["ts"], reverse=True)
    all_items = all_items[:MAX_ITEMS_TOTAL]

    sources_active = sorted(set(x["bank"] for x in all_items))

    print(f"\n📚 Total research notes: {len(all_items)}")
    for bank, count in sorted(bank_counts.items()):
        print(f"   {bank}: {count}")

    # ── Write research-data/bank-research.json ──────────────────────────────
    os.makedirs("research-data", exist_ok=True)

    output = {
        "updated_utc":   now_utc.isoformat(),
        "updated_label": now_utc.strftime("%H:%M UTC"),
        "total":         len(all_items),
        "sources":       sources_active,
        "items":         all_items,
    }

    raw = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(raw)  # validate before write
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(raw)

    print(f"\n✓ Saved {len(all_items)} notes → {OUTPUT_FILE}")

    # ── Inject research_context into context_snapshot.json ──────────────────
    # Top 20 items from last 48h — headline-only, for narrative enrichment.
    # The existing narrative pipeline reads context_snapshot.json; this block
    # gives it bank views without an additional LLM call.
    if os.path.exists(CONTEXT_SNAPSHOT):
        try:
            with open(CONTEXT_SNAPSHOT, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            cutoff_48h = now_utc - timedelta(hours=48)
            recent = [
                {
                    "bank":  x["bank"],
                    "title": x["title"],
                    "pairs": x["pairs"],
                    "ts":    x["ts"],
                }
                for x in all_items
                if datetime.fromtimestamp(x["ts"] / 1000, tz=timezone.utc) >= cutoff_48h
            ][:20]

            snapshot["research_context"] = {
                "updated_utc": now_utc.isoformat(),
                "items":       recent,
            }
            snapshot["saved_at"] = now_utc.isoformat()

            raw_snap = json.dumps(snapshot, ensure_ascii=False, indent=2)
            json.loads(raw_snap)
            with open(CONTEXT_SNAPSHOT, "w", encoding="utf-8") as f:
                f.write(raw_snap)

            print(f"✓ Injected {len(recent)} items → {CONTEXT_SNAPSHOT}[research_context]")

        except Exception as e:
            print(f"⚠️  context_snapshot update failed: {e}")
    else:
        print(f"  [context_snapshot] File not found — skipping injection")


if __name__ == "__main__":
    main()
