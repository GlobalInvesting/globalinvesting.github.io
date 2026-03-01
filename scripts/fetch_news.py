#!/usr/bin/env python3
"""
fetch_news.py
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.
Corre via GitHub Actions 3 veces al día junto con enrich_news.py.

SELECCIÓN INTELIGENTE (MAX_NEWS = 30):
  - Recopila todos los artículos válidos de los últimos MAX_AGE_DAYS días
  - Agrupa por divisa (8 divisas: USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD)
  - Dentro de cada divisa, ordena por impacto (high > med > low) y luego por fecha
  - Fase 1: toma hasta GUARANTEED_PER_CUR artículos de cada divisa que tenga contenido
  - Fase 2: reparte los slots restantes hasta MAX_NEWS entre todas las divisas
            priorizando siempre high > med > low
  - Garantiza cobertura de todas las divisas disponibles sin desperdiciar slots
"""

import json
import re
import hashlib
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
MAX_NEWS          = 30   # noticias máximas en el JSON final
MAX_AGE_DAYS      = 2    # descartar noticias más antiguas
GUARANTEED_PER_CUR = 2   # mínimo garantizado por divisa (si hay contenido)
MAX_PER_CUR       = 6    # máximo por divisa (para evitar monopolio)
OUTPUT_FILE       = "news-data/news.json"
IMPACT_ORDER      = {"high": 0, "med": 1, "low": 2}
CURRENCIES        = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

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
        "economía estadounidense", "pib eeuu", "inflación eeuu",
        "mercado laboral eeuu", "bonos del tesoro", "deuda eeuu",
    ],
    "EUR": [
        "ecb", "european central bank", "lagarde", "euro ", "eur", "eurozone",
        "euro zone", "germany", "france", "italy", "spain", "draghi", "ifo",
        "zew", "pmi europe", "eu economy", "european economy", "bund",
        "banco central europeo", "bce", "zona euro", "eurozona",
        "alemania", "economía europea", "pib zona euro", "inflación zona euro",
        "ipc zona euro", "balanza comercial alemania",
    ],
    "GBP": [
        "boe", "bank of england", "bailey", "pound", "gbp", "sterling",
        "uk economy", "united kingdom", "britain", "brexit", "gilts",
        "uk gdp", "uk inflation", "uk jobs",
        "banco de inglaterra", "libra esterlina", "libra ", "reino unido",
        "economía uk", "pib uk", "inflación uk", "ipc uk", "pmi uk",
    ],
    "JPY": [
        "boj", "bank of japan", "ueda", "yen", "jpy", "japan economy",
        "japanese", "nikkei", "shunto", "boj meeting", "japan gdp",
        "banco de japón", "yen japonés", "economía japonesa", "pib japón",
        "inflación japón", "ipc japón",
    ],
    "AUD": [
        "rba", "reserve bank of australia", "aussie", "aud", "australia",
        "australian economy", "australian jobs", "caixin",
        "banco de la reserva de australia", "dólar australiano",
        "economía australia", "pib australia", "empleo australia",
    ],
    "CAD": [
        "boc", "bank of canada", "macklem", "canadian dollar", "cad",
        "canada economy", "loonie", "oil prices", "crude oil", "wti",
        "banco de canadá", "dólar canadiense",
        "economía canadá", "pib canadá", "inflación canadá", "ipc canadá",
        "petróleo", "precio del petróleo", "crudo ",
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
        "banco de la reserva de nueva zelanda", "dólar neozelandés",
        "nueva zelanda", "economía de nueva zelanda", "pib nueva zelanda",
    ],
}

# ─────────────────────────────────────────────
# FEEDS RSS
# ─────────────────────────────────────────────
FEEDS = [
    # ── ESPAÑOL ──────────────────────────────────────────────────────────────
    { "source": "FXStreet ES",    "url": "https://www.fxstreet.es/rss/news",                              "method": "feedparser", "lang": "es" },
    { "source": "DailyForex ES",  "url": "https://es.dailyforex.com/rss/es/forexnews.xml",                "method": "feedparser", "lang": "es" },
    { "source": "DailyForex ES",  "url": "https://es.dailyforex.com/rss/es/TechnicalAnalysis.xml",        "method": "feedparser", "lang": "es" },
    { "source": "DailyForex ES",  "url": "https://es.dailyforex.com/rss/es/FundamentalAnalysis.xml",      "method": "feedparser", "lang": "es" },
    { "source": "DailyForex ES",  "url": "https://es.dailyforex.com/rss/es/forexarticles.xml",            "method": "feedparser", "lang": "es" },
    { "source": "Investing.com ES","url": "https://es.investing.com/rss/news_1.rss",                       "method": "feedparser", "lang": "es" },
    { "source": "Investing.com ES","url": "https://es.investing.com/rss/news_25.rss",                      "method": "feedparser", "lang": "es" },
    { "source": "Investing.com ES","url": "https://es.investing.com/rss/news_14.rss",                      "method": "feedparser", "lang": "es" },
    # ── INGLÉS ───────────────────────────────────────────────────────────────
    { "source": "FXStreet",       "url": "https://www.fxstreet.com/rss/news",                              "method": "feedparser", "lang": "en" },
    { "source": "FXStreet",       "url": "https://www.fxstreet.com/rss/analysis",                          "method": "feedparser", "lang": "en" },
    { "source": "ForexLive",      "url": "https://www.forexlive.com/feed/news",                            "method": "feedparser", "lang": "en" },
    { "source": "ForexLive",      "url": "https://www.forexlive.com/feed/centralbank",                     "method": "feedparser", "lang": "en" },
    { "source": "ECB",            "url": "https://www.ecb.europa.eu/rss/press.html",                       "method": "feedparser", "lang": "en" },
    { "source": "Bank of England","url": "https://www.bankofengland.co.uk/rss/news",                       "method": "feedparser", "lang": "en" },
    { "source": "Bank of Canada", "url": "https://www.bankofcanada.ca/feed/",                              "method": "feedparser", "lang": "en" },
    { "source": "DailyForex",     "url": "https://www.dailyforex.com/rss/forexnews.xml",                   "method": "feedparser", "lang": "en" },
    { "source": "ActionForex",    "url": "https://www.actionforex.com/category/live-comments/feed/",       "method": "feedparser", "lang": "en" },
    { "source": "ActionForex",    "url": "https://www.actionforex.com/category/action-insight/feed/",      "method": "feedparser", "lang": "en" },
    { "source": "InvestingLive",  "url": "https://investinglive.com/feed/centralbank/",                    "method": "feedparser", "lang": "en" },
    { "source": "InvestingLive",  "url": "https://investinglive.com/feed/technicalanalysis/",              "method": "feedparser", "lang": "en" },
    { "source": "MyFXBook",       "url": "https://www.myfxbook.com/rss/latest-forex-news",                 "method": "feedparser", "lang": "en" },
    { "source": "Investing.com",  "url": "https://www.investing.com/rss/forex_Technical.rss",              "method": "feedparser", "lang": "en" },
    { "source": "Investing.com",  "url": "https://www.investing.com/rss/forex_Fundamental.rss",            "method": "feedparser", "lang": "en" },
    { "source": "Investing.com",  "url": "https://www.investing.com/rss/forex_Opinion.rss",                "method": "feedparser", "lang": "en" },
    { "source": "Investing.com",  "url": "https://www.investing.com/rss/forex_Signals.rss",                "method": "feedparser", "lang": "en" },
]

# ─────────────────────────────────────────────
# PALABRAS CLAVE DE IMPACTO
# ─────────────────────────────────────────────
HIGH_IMPACT_KW = [
    "rate decision", "interest rate", "hike", "cut rates", "fomc", "ecb meeting",
    "boe meeting", "boj meeting", "nonfarm", "non-farm", "cpi", "inflation report",
    "gdp", "recession", "emergency", "crisis", "default", "shock",
    "surprise", "unexpected", "powell", "lagarde", "ueda", "bailey",
    "central bank", "rate hike", "rate cut", "monetary policy",
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
    "surplus", "forecast", "outlook", "guidance", "payroll", "manufacturing",
    "pmi manufacturero", "pmi de servicios", "desempleo", "tasa de paro",
    "balanza comercial", "ventas minoristas", "producción industrial",
    "confianza del consumidor", "salarios", "exportaciones", "importaciones",
    "manufactura", "sector servicios", "cuenta corriente",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
SOURCE_CURRENCY = {
    "Bank of Canada":  "CAD", "ECB": "EUR", "Bank of England": "GBP",
    "Bank of Japan":   "JPY", "RBA": "AUD", "RBNZ":            "NZD",
    "SNB":             "CHF", "Federal Reserve": "USD",
}

FOREX_SOURCES = {
    "FXStreet ES", "FXStreet", "ForexLive", "DailyForex ES", "DailyForex",
    "Bank of Canada", "ECB", "Bank of England", "Bank of Japan",
    "RBA", "RBNZ", "SNB", "Federal Reserve",
    "ActionForex", "InvestingLive", "MyFXBook", "Investing.com",
}

FOREX_RELEVANCE_KW = [
    "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
    "dollar", "dólar", "euro", "pound", "yen", "franc", "franco",
    "forex", " fx ", "currency", "currencies", "divisa", "divisas",
    "fed", "bce", "ecb", "boe", "boj", "rba", "boc", "snb", "rbnz",
    "banco central", "central bank", "interest rate", "tasa de interés",
    "inflation", "inflación", "gdp", "pib", "cpi", "ipc",
    "unemployment", "desempleo", "payroll", "pmi", "retail sales",
    "recession", "recesión", "monetary policy", "política monetaria",
    "bond", "yield", "treasury", "oil", "petróleo", "gold", "oro",
    "market", "mercado", "trading", "tariff", "arancel",
]


def is_forex_relevant(title, summary, source):
    if source in FOREX_SOURCES:
        return True
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_RELEVANCE_KW)


def detect_currency(title, summary, source=""):
    text = (title + " " + summary).lower()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return currency
    return SOURCE_CURRENCY.get(source, "USD")


def detect_impact(title, summary):
    text = (title + " " + summary).lower()
    if any(kw in text for kw in HIGH_IMPACT_KW):
        return "high"
    if any(kw in text for kw in MED_IMPACT_KW):
        return "med"
    return "low"


def parse_date(entry):
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return dateparser.parse(val).astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def clean_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entry_id(title, source):
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg):
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
        print(f"  [ERROR] {feed_cfg['url'][:65]}: {e}")
        return []


# ─────────────────────────────────────────────
# SELECCIÓN INTELIGENTE
# ─────────────────────────────────────────────

def smart_select(articles, max_total, guaranteed_per_cur, max_per_cur):
    """
    Selección en dos fases:
      Fase 1 — Garantía: toma hasta `guaranteed_per_cur` artículos de CADA divisa
               que tenga contenido, priorizando high > med > low.
      Fase 2 — Relleno: con los slots restantes, toma más artículos del pool
               completo, priorizando high > med > low y luego por ts desc,
               respetando el límite max_per_cur por divisa.

    Esto garantiza cobertura de todas las divisas disponibles y rellena
    el resto con los artículos de mayor impacto disponibles.
    """
    # Agrupar y ordenar por impacto + fecha dentro de cada grupo
    groups = {cur: [] for cur in CURRENCIES}
    for a in articles:
        cur = a.get("cur", "USD")
        if cur in groups:
            groups[cur].append(a)

    for cur in CURRENCIES:
        groups[cur].sort(key=lambda x: (IMPACT_ORDER.get(x["impact"], 2), -x["ts"]))

    selected_ids = set()
    selected = []
    taken = {cur: 0 for cur in CURRENCIES}

    # ── Fase 1: garantía mínima por divisa ───────────────────────────────────
    for cur in CURRENCIES:
        for a in groups[cur]:
            if taken[cur] >= guaranteed_per_cur:
                break
            if len(selected) >= max_total:
                break
            selected.append(a)
            selected_ids.add(a["id"])
            taken[cur] += 1

    # ── Fase 2: relleno con los mejores del pool completo ────────────────────
    remaining_pool = [
        a for a in articles
        if a["id"] not in selected_ids and taken.get(a["cur"], 0) < max_per_cur
    ]
    # Ordenar el pool por impacto y luego por ts desc
    remaining_pool.sort(key=lambda x: (IMPACT_ORDER.get(x["impact"], 2), -x["ts"]))

    for a in remaining_pool:
        if len(selected) >= max_total:
            break
        cur = a.get("cur", "USD")
        if taken.get(cur, 0) >= max_per_cur:
            continue
        selected.append(a)
        selected_ids.add(a["id"])
        taken[cur] = taken.get(cur, 0) + 1

    # Ordenar resultado final cronológicamente
    selected.sort(key=lambda x: x["ts"], reverse=True)
    return selected


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(days=MAX_AGE_DAYS)
    seen_ids = set()
    raw_articles = []
    es_raw = en_raw = 0

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] Iniciando fetch de {len(FEEDS)} feeds...")

    for feed_cfg in FEEDS:
        source = feed_cfg["source"]
        lang   = feed_cfg.get("lang", "en")
        print(f"  [{lang.upper()}] Fetching {source} — {feed_cfg['url'][:65]}...")

        entries = fetch_via_feedparser(feed_cfg)
        count = 0

        for entry in entries:
            title   = clean_html(getattr(entry, "title", ""))
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

            raw_articles.append({
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
                es_raw += 1
            else:
                en_raw += 1

        print(f"    → {count} noticias válidas")

    print(f"\n📦 Total artículos recopilados: {len(raw_articles)}")
    print(f"   ES: {es_raw} | EN: {en_raw}")

    dist_before   = Counter(a["cur"] for a in raw_articles)
    impact_before = Counter(a["impact"] for a in raw_articles)
    print(f"   Distribución: {dict(sorted(dist_before.items()))}")
    print(f"   Impacto: high={impact_before['high']} | med={impact_before['med']} | low={impact_before['low']}")

    # Divisas sin cobertura
    missing = [c for c in CURRENCIES if dist_before.get(c, 0) == 0]
    if missing:
        print(f"   ⚠️  Sin artículos hoy: {', '.join(missing)}")

    # ── Selección inteligente ─────────────────────────────────────────────────
    articles = smart_select(
        raw_articles,
        max_total=MAX_NEWS,
        guaranteed_per_cur=GUARANTEED_PER_CUR,
        max_per_cur=MAX_PER_CUR,
    )

    dist_after   = Counter(a["cur"] for a in articles)
    impact_after = Counter(a["impact"] for a in articles)
    print(f"\n✂️  Selección final ({len(articles)} artículos | garantía: {GUARANTEED_PER_CUR}/divisa | máx: {MAX_PER_CUR}/divisa):")
    print(f"   Distribución: {dict(sorted(dist_after.items()))}")
    print(f"   Impacto: high={impact_after['high']} | med={impact_after['med']} | low={impact_after['low']}")

    sources_ok = sorted(set(a["source"] for a in articles))

    output = {
        "updated_utc":    now_utc.isoformat(),
        "updated_label":  now_utc.strftime("%H:%M UTC"),
        "total":          len(articles),
        "total_high":     impact_after["high"],
        "total_med":      impact_after["med"],
        "sources_active": sources_ok,
        "lang_counts": {
            "es": sum(1 for a in articles if a.get("lang") == "es"),
            "en": sum(1 for a in articles if a.get("lang") == "en"),
        },
        "articles": articles,
    }

    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(articles)} artículos guardados en {OUTPUT_FILE}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
