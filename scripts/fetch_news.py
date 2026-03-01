#!/usr/bin/env python3
"""
fetch_news.py
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.
Corre via GitHub Actions 3 veces al día junto con enrich_news.py.

Formato de salida compatible con news.html:
  {
    "articles": [ { "cur", "impact", "title", "expand", "source", "link", "time", "ts", "featured" }, ... ],
    "updated_utc": "...",
    "updated_label": "...",
    "total": N,
    "total_high": N,
    "total_med": N,
    "sources_active": [...]
  }
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
MAX_NEWS     = 60   # noticias máximas en el JSON final
MAX_AGE_DAYS = 3    # descartar noticias más antiguas que esto
OUTPUT_FILE  = "news-data/news.json"

# ─────────────────────────────────────────────
# DETECCIÓN DE DIVISA
# ─────────────────────────────────────────────
CURRENCY_KEYWORDS = {
    "USD": [
        "fed ", "federal reserve", "fomc", "powell", "dollar", "usd", "us economy",
        "us gdp", "nonfarm", "non-farm", "jobless claims", "us inflation",
        "treasury", "debt ceiling", "ism ", "us jobs", "american economy",
        "wall street", "nasdaq", "dow jones", "s&p 500",
        "reserva federal", "dólar", "dolar americano", "economía de eeuu",
        "economía estadounidense", "pib eeuu", "inflación eeuu", "inflación de estados unidos",
        "mercado laboral eeuu", "bonos del tesoro", "deuda eeuu",
    ],
    "EUR": [
        "ecb", "european central bank", "lagarde", "euro ", "eur", "eurozone",
        "euro zone", "germany", "france", "italy", "spain", "draghi", "ifo",
        "zew", "pmi europe", "eu economy", "european economy", "bund",
        "banco central europeo", "bce", "zona euro", "eurozona",
        "alemania", "economía europea", "pib zona euro", "inflación zona euro",
        "inflación de la zona euro", "ipc zona euro",
        "balanza comercial alemania", "confianza zew", "confianza ifo",
    ],
    "GBP": [
        "boe", "bank of england", "bailey", "pound", "gbp", "sterling",
        "uk economy", "united kingdom", "britain", "brexit", "gilts",
        "uk gdp", "uk inflation", "uk jobs",
        "banco de inglaterra", "libra esterlina", "libra ", "reino unido",
        "economía uk", "economía del reino unido", "pib uk", "inflación uk",
        "inflación del reino unido", "ipc uk", "pmi uk",
    ],
    "JPY": [
        "boj", "bank of japan", "ueda", "yen", "jpy", "japan economy",
        "japanese", "nikkei", "shunto", "boj meeting", "japan gdp",
        "banco de japón", "banco de japan", "yen japonés", "yen japones",
        "economía japonesa", "economía de japón", "pib japón", "inflación japón",
        "ipc japón", "producción industrial japón",
    ],
    "AUD": [
        "rba", "reserve bank of australia", "aussie", "aud", "australia",
        "australian economy", "australian jobs", "caixin",
        "banco de la reserva de australia", "dólar australiano", "dolar australiano",
        "economía australia", "economía de australia", "pib australia",
        "empleo australia", "inflación australia",
    ],
    "CAD": [
        "boc", "bank of canada", "macklem", "canadian dollar", "cad",
        "canada economy", "loonie", "oil prices", "crude oil", "wti",
        "banco de canadá", "banco de canada", "dólar canadiense", "dolar canadiense",
        "economía canadá", "economía de canadá", "pib canadá", "inflación canadá",
        "ipc canadá", "petróleo", "precio del petróleo", "crudo ",
    ],
    "CHF": [
        "snb", "swiss national bank", "jordan", "swiss franc", "chf",
        "switzerland", "swiss economy", "swiss inflation",
        "banco nacional suizo", "franco suizo", "suiza", "economía suiza",
        "pib suiza", "inflación suiza", "ipc suiza",
    ],
    "NZD": [
        "rbnz", "reserve bank of new zealand", "orr", "kiwi", "nzd",
        "new zealand", "nz economy", "nz jobs",
        "banco de la reserva de nueva zelanda", "dólar neozelandés", "dolar neozelandes",
        "nueva zelanda", "economía de nueva zelanda", "pib nueva zelanda",
        "inflación nueva zelanda",
    ],
}

# ─────────────────────────────────────────────
# FEEDS RSS
# ─────────────────────────────────────────────
FEEDS = [

    # ══════════════════════════════════════════
    # FUENTES EN ESPAÑOL
    # ══════════════════════════════════════════

    {
        "source": "FXStreet ES",
        "url": "https://www.fxstreet.es/rss/news",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/forexnews.xml",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/TechnicalAnalysis.xml",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/FundamentalAnalysis.xml",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/forexarticles.xml",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "Investing.com ES",
        "url": "https://es.investing.com/rss/news_1.rss",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "Investing.com ES",
        "url": "https://es.investing.com/rss/news_25.rss",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "Investing.com ES",
        "url": "https://es.investing.com/rss/news_14.rss",
        "method": "feedparser",
        "lang": "es",
    },
    # ✗ ElEconomista  — bloquea el User-Agent de Actions
    # ✗ InfoMercados  — noticias no especializadas en forex
    # ✗ Forexduet ES  — 0 noticias consistentemente
    # ✗ Cinco Días    — noticias no especializadas en forex

    # ══════════════════════════════════════════
    # FUENTES EN INGLÉS
    # ══════════════════════════════════════════

    {
        "source": "FXStreet",
        "url": "https://www.fxstreet.com/rss/news",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "FXStreet",
        "url": "https://www.fxstreet.com/rss/analysis",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "ForexLive",
        "url": "https://www.forexlive.com/feed/news",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "ForexLive",
        "url": "https://www.forexlive.com/feed/centralbank",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "ECB",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "Bank of England",
        "url": "https://www.bankofengland.co.uk/rss/news",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "Bank of Canada",
        "url": "https://www.bankofcanada.ca/feed/",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "DailyForex",
        "url": "https://www.dailyforex.com/rss/forexnews.xml",
        "method": "feedparser",
        "lang": "en",
    },

    # ── NUEVAS FUENTES EN INGLÉS ───────────────────────────────────────────────

    # Action Forex — análisis técnico y comentarios en vivo de alta calidad
    {
        "source": "ActionForex",
        "url": "https://www.actionforex.com/category/live-comments/feed/",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "ActionForex",
        "url": "https://www.actionforex.com/category/action-insight/feed/",
        "method": "feedparser",
        "lang": "en",
    },

    # InvestingLive — banco central y análisis técnico
    {
        "source": "InvestingLive",
        "url": "https://investinglive.com/feed/centralbank/",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "InvestingLive",
        "url": "https://investinglive.com/feed/technicalanalysis/",
        "method": "feedparser",
        "lang": "en",
    },

    # MyFXBook — noticias forex en tiempo real
    {
        "source": "MyFXBook",
        "url": "https://www.myfxbook.com/rss/latest-forex-news",
        "method": "feedparser",
        "lang": "en",
    },

    # Investing.com EN — secciones específicas de forex
    {
        "source": "Investing.com",
        "url": "https://www.investing.com/rss/forex_Technical.rss",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "Investing.com",
        "url": "https://www.investing.com/rss/forex_Fundamental.rss",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "Investing.com",
        "url": "https://www.investing.com/rss/forex_Opinion.rss",
        "method": "feedparser",
        "lang": "en",
    },
    {
        "source": "Investing.com",
        "url": "https://www.investing.com/rss/forex_Signals.rss",
        "method": "feedparser",
        "lang": "en",
    },
]

# ─────────────────────────────────────────────
# PALABRAS CLAVE DE IMPACTO — ES + EN
# ─────────────────────────────────────────────
HIGH_IMPACT_KW = [
    "rate decision", "interest rate", "hike", "cut rates", "fomc", "ecb meeting",
    "boe meeting", "boj meeting", "nonfarm", "non-farm", "cpi", "inflation report",
    "gdp", "recession", "emergency", "crisis", "default", "shock",
    "surprise", "unexpected", "powell", "lagarde", "ueda", "bailey",
    "central bank", "rate hike", "rate cut", "quantitative easing",
    "quantitative tightening", "monetary policy",
    "decisión de tasas", "tasa de interés", "subida de tipos", "bajada de tipos",
    "alza de tasas", "recorte de tasas", "sube tasas", "baja tasas",
    "inflación", "ipc ", "pib ", "recesión", "crisis ", "sorprende",
    "inesperado", "inesperada", "política monetaria", "banco central",
    "hawkish", "dovish", "restrictivo", "acomodaticio",
]

MED_IMPACT_KW = [
    "pmi", "employment", "jobless", "trade balance", "retail sales",
    "industrial production", "consumer confidence", "business confidence",
    "housing", "wages", "earnings", "exports", "imports", "deficit",
    "surplus", "forecast", "outlook", "guidance", "payroll",
    "manufacturing", "services sector",
    "pmi manufacturero", "pmi de servicios", "desempleo", "tasa de paro",
    "balanza comercial", "ventas minoristas", "producción industrial",
    "confianza del consumidor", "confianza empresarial",
    "salarios", "exportaciones", "importaciones", "déficit",
    "superávit", "previsión", "perspectivas", "nóminas",
    "manufactura", "sector servicios", "cuenta corriente",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

SOURCE_CURRENCY = {
    "Bank of Canada":  "CAD",
    "ECB":             "EUR",
    "Bank of England": "GBP",
    "Bank of Japan":   "JPY",
    "RBA":             "AUD",
    "RBNZ":            "NZD",
    "SNB":             "CHF",
    "Federal Reserve": "USD",
}

# Todas las fuentes del feed son especializadas en forex — artículos siempre relevantes
FOREX_SOURCES = {
    "FXStreet ES", "FXStreet", "ForexLive", "DailyForex ES", "DailyForex",
    "Bank of Canada", "ECB", "Bank of England", "Bank of Japan",
    "RBA", "RBNZ", "SNB", "Federal Reserve",
    "ActionForex", "InvestingLive", "MyFXBook", "Investing.com",
}

FOREX_RELEVANCE_KW = [
    "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
    "dollar", "dólar", "euro", "pound", "yen", "franc", "franco",
    "forex", " fx ", "currency", "currencies", "divisa", "divisas", "tipo de cambio",
    "fed", "bce", "ecb", "boe", "boj", "rba", "boc", "snb", "rbnz",
    "banco central", "central bank",
    "interest rate", "tasa de interés", "tipos de interés",
    "inflation", "inflación", "gdp", "pib", "cpi", "ipc",
    "unemployment", "desempleo", "payroll", "nóminas",
    "trade balance", "balanza", "pmi", "retail sales", "ventas minoristas",
    "recession", "recesión", "monetary policy", "política monetaria",
    "stock market", "bolsa", "bonos", "bond", "yield", "treasury",
    "oil", "petróleo", "gold", "oro", "commodit",
    "market", "mercado", "trading", "investor", "inversor",
    "sanctions", "sanciones", "tariff", "arancel", "trade war", "guerra comercial",
]


def is_forex_relevant(title: str, summary: str, source: str) -> bool:
    if source in FOREX_SOURCES:
        return True
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_RELEVANCE_KW)


def detect_currency(title: str, summary: str, source: str = "") -> str:
    text = (title + " " + summary).lower()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return currency
    return SOURCE_CURRENCY.get(source, "USD")


def detect_impact(title: str, summary: str) -> str:
    text = (title + " " + summary).lower()
    for kw in HIGH_IMPACT_KW:
        if kw in text:
            return "high"
    for kw in MED_IMPACT_KW:
        if kw in text:
            return "med"
    return "low"


def parse_date(entry) -> datetime:
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return dateparser.parse(val).astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def entry_id(title: str, source: str) -> str:
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg: dict) -> list:
    try:
        d = feedparser.parse(
            feed_cfg["url"],
            request_headers={
                "User-Agent": "Mozilla/5.0 (compatible; ForexNewsBot/1.0)",
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            }
        )
        return d.entries
    except Exception as e:
        print(f"  [ERROR] feedparser {feed_cfg['url'][:65]}: {e}")
        return []


def fetch_via_proxy(feed_cfg: dict) -> list:
    try:
        resp = requests.get(feed_cfg["url"], timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ForexNewsBot/1.0)"
        })
        resp.raise_for_status()
        d = feedparser.parse(resp.text)
        return d.entries
    except Exception as e:
        print(f"  [ERROR] proxy {feed_cfg['url'][:65]}: {e}")
        return []


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    now_utc  = datetime.now(timezone.utc)
    cutoff   = now_utc - timedelta(days=MAX_AGE_DAYS)
    seen_ids = set()
    articles = []

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] Iniciando fetch de {len(FEEDS)} feeds...")

    es_count = 0
    en_count = 0

    for feed_cfg in FEEDS:
        source = feed_cfg["source"]
        lang   = feed_cfg.get("lang", "en")
        print(f"  [{lang.upper()}] Fetching {source} — {feed_cfg['url'][:65]}...")

        if feed_cfg["method"] == "proxy_xml":
            entries = fetch_via_proxy(feed_cfg)
        else:
            entries = fetch_via_feedparser(feed_cfg)

        count = 0
        for entry in entries:
            title   = clean_html(getattr(entry, "title",   ""))
            summary = clean_html(
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or (getattr(entry, "content", [{}])[0].get("value", "") if hasattr(entry, "content") else "")
            )
            link = getattr(entry, "link", "") or getattr(entry, "id", "")

            if not title or len(title) < 15:
                continue

            pub_date = parse_date(entry)
            if pub_date < cutoff:
                continue

            nid = entry_id(title, source)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            if not is_forex_relevant(title, summary, source):
                continue

            cur    = detect_currency(title, summary, source)
            impact = detect_impact(title, summary)

            expand = summary[:300] + ("..." if len(summary) > 300 else "")

            articles.append({
                "id":       nid,
                "cur":      cur,
                "impact":   impact,
                "title":    title,
                "expand":   expand,
                "source":   source,
                "link":     link,
                "time":     pub_date.strftime("%H:%M"),
                "ts":       int(pub_date.timestamp() * 1000),
                "featured": impact == "high",
                "lang":     lang,
                "date":     pub_date.strftime("%d %b"),
                "datetime": pub_date.isoformat(),
            })
            count += 1
            if lang == "es":
                es_count += 1
            else:
                en_count += 1

        print(f"    → {count} noticias válidas")

    articles.sort(key=lambda x: x["ts"], reverse=True)
    articles = articles[:MAX_NEWS]

    total_high = sum(1 for a in articles if a["impact"] == "high")
    total_med  = sum(1 for a in articles if a["impact"] == "med")
    sources_ok = sorted(set(a["source"] for a in articles))

    output = {
        "updated_utc":    now_utc.isoformat(),
        "updated_label":  now_utc.strftime("%H:%M UTC"),
        "total":          len(articles),
        "total_high":     total_high,
        "total_med":      total_med,
        "sources_active": sources_ok,
        "lang_counts": {
            "es": es_count,
            "en": en_count,
        },
        "articles": articles,
    }

    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(articles)} artículos guardados en {OUTPUT_FILE}")
    print(f"  ES: {es_count} | EN: {en_count}")
    print(f"  Alto impacto: {total_high} | Medio: {total_med}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
