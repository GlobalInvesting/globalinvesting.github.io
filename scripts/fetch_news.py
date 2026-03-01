#!/usr/bin/env python3
"""
fetch_news.py
Obtiene noticias forex desde múltiples fuentes RSS y genera news.json.
Corre via GitHub Actions cada hora.
"""

import json
import re
import hashlib
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
MAX_NEWS     = 50   # noticias máximas en el JSON final
MAX_AGE_DAYS = 3    # descartar noticias más antiguas que esto
OUTPUT_FILE  = "news.json"

# Palabras clave para detectar qué divisa menciona cada noticia.
# Orden importa: se asigna la primera coincidencia.
CURRENCY_KEYWORDS = {
    "USD": ["fed ", "federal reserve", "fomc", "powell", "dollar", "usd", "us economy",
            "us gdp", "nonfarm", "non-farm", "jobless claims", "cpi usa", "us inflation",
            "treasury", "debt ceiling", "ism ", "us jobs", "american economy"],
    "EUR": ["ecb", "european central bank", "lagarde", "euro ", "eur", "eurozone",
            "euro zone", "germany", "france", "italy", "spain", "draghi", "ifo",
            "zew", "pmi europe", "eu economy", "european economy", "bund"],
    "GBP": ["boe", "bank of england", "bailey", "pound", "gbp", "sterling",
            "uk economy", "united kingdom", "britain", "brexit", "gilts",
            "uk gdp", "uk inflation", "uk jobs"],
    "JPY": ["boj", "bank of japan", "ueda", "yen", "jpy", "japan economy",
            "japanese", "nikkei", "shunto", "boj meeting", "japan gdp"],
    "AUD": ["rba", "reserve bank of australia", "aussie", "aud", "australia",
            "australian economy", "australian jobs", "caixin"],
    "CAD": ["boc", "bank of canada", "macklem", "canadian dollar", "cad",
            "canada economy", "loonie", "oil prices", "crude oil", "wti"],
    "CHF": ["snb", "swiss national bank", "jordan", "swiss franc", "chf",
            "switzerland", "swiss economy", "swiss inflation"],
    "NZD": ["rbnz", "reserve bank of new zealand", "orr", "kiwi", "nzd",
            "new zealand", "nz economy", "nz jobs"],
}

# Feeds RSS — en orden de prioridad / confiabilidad
FEEDS = [
    # ── FXStreet ──────────────────────────────────────────────────
    {
        "source": "FXStreet",
        "url": "https://www.fxstreet.com/rss/news",
        "method": "feedparser",
    },
    {
        "source": "FXStreet",
        "url": "https://www.fxstreet.com/rss/analysis",
        "method": "feedparser",
    },
    # ── ForexLive ─────────────────────────────────────────────────
    {
        "source": "ForexLive",
        "url": "https://www.forexlive.com/feed/news",
        "method": "feedparser",
    },
    {
        "source": "ForexLive",
        "url": "https://www.forexlive.com/feed/centralbank",
        "method": "feedparser",
    },
    # ── Reuters (vía rss.app proxy público, sin auth) ─────────────
    {
        "source": "Reuters",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "method": "feedparser",
    },
    {
        "source": "Reuters",
        "url": "https://feeds.reuters.com/reuters/UKBusinessNews",
        "method": "feedparser",
    },
    # ── Investing.com (vía allorigins proxy CORS) ──────────────────
    {
        "source": "Investing.com",
        "url": "https://api.allorigins.win/raw?url=https://www.investing.com/rss/news_25.rss",
        "method": "proxy_xml",
    },
    {
        "source": "Investing.com",
        "url": "https://api.allorigins.win/raw?url=https://www.investing.com/rss/news_14.rss",
        "method": "proxy_xml",
    },
    # ── MQL5 Economic Calendar ─────────────────────────────────────
    {
        "source": "MQL5",
        "url": "https://www.mql5.com/en/economic-calendar/rss",
        "method": "feedparser",
    },
    # ── Bancos Centrales (fuentes oficiales) ───────────────────────
    {
        "source": "Federal Reserve",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "method": "feedparser",
    },
    {
        "source": "ECB",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "method": "feedparser",
    },
    {
        "source": "Bank of England",
        "url": "https://www.bankofengland.co.uk/rss/news",
        "method": "feedparser",
    },
    {
        "source": "Bank of Japan",
        "url": "https://www.boj.or.jp/en/about/press/index.htm",
        "method": "feedparser",
    },
    {
        "source": "RBA",
        "url": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
        "method": "feedparser",
    },
    {
        "source": "Bank of Canada",
        "url": "https://www.bankofcanada.ca/feed/",
        "method": "feedparser",
    },
    {
        "source": "SNB",
        "url": "https://www.snb.ch/en/rss/medmit",
        "method": "feedparser",
    },
    {
        "source": "RBNZ",
        "url": "https://www.rbnz.govt.nz/feed/news",
        "method": "feedparser",
    },
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def detect_currency(title: str, summary: str) -> str:
    """Detecta la divisa principal mencionada en la noticia."""
    text = (title + " " + summary).lower()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return currency
    return "GENERAL"


def detect_impact(title: str, summary: str) -> str:
    """Estima el nivel de impacto por palabras clave."""
    text = (title + " " + summary).lower()
    high_kw = [
        "rate decision", "interest rate", "hike", "cut", "fomc", "ecb meeting",
        "boe meeting", "boj meeting", "nonfarm", "non-farm", "cpi", "inflation",
        "gdp", "recession", "emergency", "crisis", "default", "shock",
        "surprise", "unexpected", "powell", "lagarde", "ueda", "bailey",
        "central bank", "rate hike", "rate cut", "quantitative",
    ]
    med_kw = [
        "pmi", "employment", "jobless", "trade balance", "retail sales",
        "industrial production", "consumer confidence", "business confidence",
        "housing", "wages", "earnings", "exports", "imports", "deficit",
        "surplus", "forecast", "outlook", "guidance",
    ]
    for kw in high_kw:
        if kw in text:
            return "HIGH"
    for kw in med_kw:
        if kw in text:
            return "MED"
    return "LOW"


def parse_date(entry) -> datetime:
    """Intenta parsear la fecha de un entry RSS."""
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return dateparser.parse(val).astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def clean_html(text: str) -> str:
    """Elimina tags HTML de un string."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def entry_id(title: str, source: str) -> str:
    """Genera un ID único por noticia."""
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg: dict) -> list:
    """Fetcha un feed con feedparser estándar."""
    try:
        d = feedparser.parse(
            feed_cfg["url"],
            request_headers={
                "User-Agent": "Mozilla/5.0 (compatible; ForexNewsBot/1.0)",
                "Accept": "application/rss+xml, application/xml, text/xml",
            }
        )
        return d.entries
    except Exception as e:
        print(f"  [ERROR] feedparser {feed_cfg['url'][:60]}: {e}")
        return []


def fetch_via_proxy(feed_cfg: dict) -> list:
    """Fetcha un feed a través de un proxy CORS y lo parsea con feedparser."""
    try:
        resp = requests.get(feed_cfg["url"], timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ForexNewsBot/1.0)"
        })
        resp.raise_for_status()
        d = feedparser.parse(resp.text)
        return d.entries
    except Exception as e:
        print(f"  [ERROR] proxy {feed_cfg['url'][:60]}: {e}")
        return []


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    now_utc    = datetime.now(timezone.utc)
    cutoff     = now_utc - timedelta(days=MAX_AGE_DAYS)
    seen_ids   = set()
    all_items  = []

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] Iniciando fetch de {len(FEEDS)} feeds...")

    for feed_cfg in FEEDS:
        source = feed_cfg["source"]
        print(f"  Fetching {source} — {feed_cfg['url'][:65]}...")

        if feed_cfg["method"] == "proxy_xml":
            entries = fetch_via_proxy(feed_cfg)
        else:
            entries = fetch_via_feedparser(feed_cfg)

        count = 0
        for entry in entries:
            title   = clean_html(getattr(entry, "title",   ""))
            summary = clean_html(getattr(entry, "summary", "") or
                                 getattr(entry, "description", ""))
            link    = getattr(entry, "link", "")

            if not title or len(title) < 15:
                continue

            pub_date = parse_date(entry)
            if pub_date < cutoff:
                continue

            nid = entry_id(title, source)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            currency = detect_currency(title, summary)
            impact   = detect_impact(title, summary)

            # Recortar summary a 280 chars
            if len(summary) > 280:
                summary = summary[:277] + "..."

            all_items.append({
                "id":       nid,
                "title":    title,
                "summary":  summary,
                "currency": currency,
                "impact":   impact,
                "source":   source,
                "link":     link,
                "time":     pub_date.strftime("%H:%M"),
                "datetime": pub_date.isoformat(),
                "date":     pub_date.strftime("%d %b"),
            })
            count += 1

        print(f"    → {count} noticias válidas de {source}")

    # Ordenar por fecha descendente y tomar top MAX_NEWS
    all_items.sort(key=lambda x: x["datetime"], reverse=True)
    all_items = all_items[:MAX_NEWS]

    # Estadísticas para el status bar
    total_high = sum(1 for n in all_items if n["impact"] == "HIGH")
    total_med  = sum(1 for n in all_items if n["impact"] == "MED")
    sources_ok = list({n["source"] for n in all_items})

    output = {
        "updated_utc": now_utc.isoformat(),
        "updated_label": now_utc.strftime("%H:%M UTC"),
        "total": len(all_items),
        "total_high": total_high,
        "total_med": total_med,
        "sources_active": sources_ok,
        "news": all_items,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(all_items)} noticias guardadas en {OUTPUT_FILE}")
    print(f"  Alto impacto: {total_high} | Medio: {total_med}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
