#!/usr/bin/env python3
"""
fetch_news.py — v3
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.

CAMBIOS v3:
  - Nuevos feeds: FXStreet /rss, InvestingLive /feed/, BabyPips, InvestMacro, ForexCrunch
  - Filtro anti-calendario: descarta entradas que son eventos futuros del calendario
    económico (sin contenido editorial real)
  - Validación de contenido mínimo: título + descripción deben tener sustancia real
  - Filtro anti-duplicado mejorado por similitud de título (no solo hash exacto)
  - is_real_news(): rechaza anuncios de agenda, recordatorios, comunicados vacíos
  - MIN_DESCRIPTION_WORDS: descarta artículos sin descripción sustancial
  - MAX_AGE_DAYS = 3 para divisas menos cubiertas (JPY, AUD, CHF, NZD)
"""

import json
import re
import hashlib
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from collections import Counter
import os

# ─────────────────────────────────────────────
MAX_NEWS              = 35   # Aumentado por más fuentes
MAX_AGE_DAYS          = 3
GUARANTEED_PER_CUR    = 2
MAX_PER_CUR           = 6
OUTPUT_FILE           = "news-data/news.json"
IMPACT_ORDER          = {"high": 0, "med": 1, "low": 2}
CURRENCIES            = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# Contenido mínimo aceptable: descripción debe tener al menos estas palabras
# para no ser un simple anuncio de calendario o encabezado vacío
MIN_DESCRIPTION_WORDS = 12

# ─────────────────────────────────────────────
# PATRONES QUE INDICAN CONTENIDO DE CALENDARIO / EVENTO FUTURO
# (no son noticias editoriales reales)
# ─────────────────────────────────────────────
CALENDAR_PATTERNS = [
    # Frases explícitas de agenda/calendario
    r"upcoming event",
    r"content type.*upcoming",
    r"scheduled date",
    r"eight scheduled",
    r"on eight",
    r"share this page by email",
    r"governing council presents",
    r"\bpress release explaining\b",
    r"four times a year.*governing",
    r"announces the setting for the overnight rate",
    r"bank of canada announces",
    # Patrones de agenda económica genérica
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$",
    r"^\d{1,2}:\d{2}\s*\(et\)",   # Sólo hora en ET sin contenido
    r"content type\(s\):",
    # Resúmenes vacíos habituales de bancos centrales
    r"^on (eight|four|six) scheduled dates",
    r"^the bank of (canada|england|japan|reserve)",
    r"monetary policy report$",
    r"rate announcement$",
    # BabyPips/educativo puro sin evento de mercado
    r"^(what is|how to|learn|guide to|introduction to|basics of|beginner)",
]
CALENDAR_RE = re.compile("|".join(CALENDAR_PATTERNS), re.IGNORECASE)

# Títulos que son eventos de calendario, no noticias
CALENDAR_TITLE_PATTERNS = [
    r"^interest rate announcement",
    r"^monetary policy (report|decision|statement|meeting)$",
    r"^(fomc|ecb|boe|boj|rba|rbnz|boc|snb) (meeting|statement|decision|minutes)$",
    r"^rate (announcement|decision|statement)$",
    r"^upcoming (event|release|data)",
    r"^economic (calendar|data release)",
    r"^(january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2}",
]
CALENDAR_TITLE_RE = re.compile("|".join(CALENDAR_TITLE_PATTERNS), re.IGNORECASE)

# Palabras clave educativas puras (BabyPips tipo "School of Pipsology")
EDUCATIONAL_ONLY_PATTERNS = [
    r"school of pipsology",
    r"babypips quiz",
    r"^quiz:",
    r"^lesson \d+",
    r"pips? glossary",
]
EDUCATIONAL_RE = re.compile("|".join(EDUCATIONAL_ONLY_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────
CURRENCY_KEYWORDS = {
    "USD": [
        "fed ", "federal reserve", "fomc", "powell", "dollar", "usd", "us economy",
        "us gdp", "nonfarm", "non-farm", "jobless claims", "us inflation",
        "treasury", "debt ceiling", "ism ", "us jobs", "american economy",
        "wall street", "nasdaq", "dow jones", "s&p 500",
        "reserva federal", "dólar", "dolar americano", "economía de eeuu",
        "economía estadounidense", "pib eeuu", "inflación eeuu",
        "mercado laboral eeuu", "bonos del tesoro", "tariff", "arancel",
    ],
    "EUR": [
        "ecb", "european central bank", "lagarde", "euro ", "eur", "eurozone",
        "euro zone", "germany", "france", "italy", "spain", "ifo", "zew",
        "pmi europe", "eu economy", "european economy", "bund",
        "banco central europeo", "bce", "zona euro", "eurozona",
        "alemania", "economía europea", "pib zona euro", "inflación zona euro",
    ],
    "GBP": [
        "boe", "bank of england", "bailey", "pound", "gbp", "sterling",
        "uk economy", "united kingdom", "britain", "brexit", "gilts",
        "uk gdp", "uk inflation", "uk jobs",
        "banco de inglaterra", "libra esterlina", "libra ", "reino unido",
    ],
    "JPY": [
        "boj", "bank of japan", "ueda", "yen", "jpy", "japan economy",
        "japanese", "nikkei", "shunto", "japan gdp", "japan inflation",
        "japan cpi", "japan pmi", "japan trade", "japan unemployment",
        "banco de japón", "yen japonés", "economía japonesa", "pib japón",
    ],
    "AUD": [
        "rba", "reserve bank of australia", "aussie", "aud", "australia",
        "australian economy", "australian jobs", "australia inflation",
        "australia cpi", "australia gdp", "australia trade", "australia retail",
        "banco de la reserva de australia", "dólar australiano",
    ],
    "CAD": [
        "boc", "bank of canada", "macklem", "canadian dollar", "cad",
        "canada economy", "loonie", "oil prices", "crude oil", "wti",
        "banco de canadá", "dólar canadiense",
        "economía canadá", "pib canadá", "inflación canadá", "petróleo",
    ],
    "CHF": [
        "snb", "swiss national bank", "jordan", "swiss franc", "chf",
        "switzerland", "swiss economy", "swiss inflation",
        "switzerland gdp", "switzerland cpi", "switzerland trade",
        "banco nacional suizo", "franco suizo", "suiza",
    ],
    "NZD": [
        "rbnz", "reserve bank of new zealand", "orr", "kiwi", "nzd",
        "new zealand", "nz economy", "nz jobs",
        "new zealand inflation", "new zealand cpi", "new zealand gdp",
        "new zealand trade", "new zealand employment",
        "banco de la reserva de nueva zelanda", "dólar neozelandés",
    ],
}

# ─────────────────────────────────────────────
FEEDS = [
    # ── ESPAÑOL ──────────────────────────────────────────────────────────────
    { "source": "FXStreet ES",      "url": "https://www.fxstreet.es/rss/news",                              "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/forexnews.xml",                "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/TechnicalAnalysis.xml",        "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/FundamentalAnalysis.xml",      "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/forexarticles.xml",            "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_1.rss",                        "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_25.rss",                       "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_14.rss",                       "lang": "es" },
    # ── INGLÉS — EXISTENTES ──────────────────────────────────────────────────
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/news",                              "lang": "en" },
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/analysis",                          "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/news",                             "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/centralbank",                      "lang": "en" },
    { "source": "ECB",              "url": "https://www.ecb.europa.eu/rss/press.html",                        "lang": "en" },
    { "source": "Bank of England",  "url": "https://www.bankofengland.co.uk/rss/news",                        "lang": "en" },
    # Bank of Canada ELIMINADO: su feed trae eventos de calendario sin contenido editorial
    { "source": "DailyForex",       "url": "https://www.dailyforex.com/rss/forexnews.xml",                   "lang": "en" },
    { "source": "ActionForex",      "url": "https://www.actionforex.com/category/live-comments/feed/",        "lang": "en" },
    { "source": "ActionForex",      "url": "https://www.actionforex.com/category/action-insight/feed/",       "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/centralbank/",                     "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/technicalanalysis/",               "lang": "en" },
    { "source": "MyFXBook",         "url": "https://www.myfxbook.com/rss/latest-forex-news",                  "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Technical.rss",               "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Fundamental.rss",             "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Opinion.rss",                 "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Signals.rss",                 "lang": "en" },
    { "source": "InstaForex",       "url": "https://news.instaforex.com/news",                                "lang": "en" },
    { "source": "InstaForex",       "url": "https://news.instaforex.com/analytics",                           "lang": "en" },
    # ── INGLÉS — NUEVOS ──────────────────────────────────────────────────────
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss",                                    "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/",                                  "lang": "en" },
    { "source": "BabyPips",         "url": "https://www.babypips.com/feed.rss",                               "lang": "en" },
    { "source": "InvestMacro",      "url": "https://investmacro.com/feed/",                                    "lang": "en" },
    { "source": "ForexCrunch",      "url": "https://forexcrunch.com/feed/",                                    "lang": "en" },
]

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
    "hawkish", "dovish",
]

MED_IMPACT_KW = [
    "pmi", "employment", "jobless", "trade balance", "retail sales",
    "industrial production", "consumer confidence", "business confidence",
    "housing", "wages", "earnings", "exports", "imports", "deficit",
    "surplus", "forecast", "outlook", "guidance", "payroll", "manufacturing",
    "pmi manufacturero", "desempleo", "balanza comercial", "ventas minoristas",
    "producción industrial", "confianza del consumidor", "salarios",
]

SOURCE_CURRENCY = {
    "ECB": "EUR",
    "Bank of England": "GBP",
    "Bank of Japan":   "JPY",
    "RBA":             "AUD",
    "RBNZ":            "NZD",
    "SNB":             "CHF",
    "Federal Reserve": "USD",
    # Bank of Canada eliminado del mapping para no asignar CAD por defecto
}

FOREX_SOURCES = {
    "FXStreet ES", "FXStreet", "ForexLive", "DailyForex ES", "DailyForex",
    "ECB", "Bank of England", "Bank of Japan", "RBA", "RBNZ", "SNB",
    "Federal Reserve", "ActionForex", "InvestingLive", "MyFXBook",
    "Investing.com", "InstaForex", "BabyPips", "InvestMacro", "ForexCrunch",
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


# ─────────────────────────────────────────────
# FILTROS DE CALIDAD
# ─────────────────────────────────────────────

def is_calendar_entry(title: str, description: str) -> bool:
    """
    Detecta si una entrada es un evento de calendario económico futuro
    o un anuncio institucional vacío de contenido editorial.
    Retorna True si debe ser DESCARTADA.
    """
    combined = (title + " " + description).strip()

    # 1. Título que coincide con patrones de calendario
    if CALENDAR_TITLE_RE.search(title.strip()):
        return True

    # 2. Descripción que contiene frases típicas de calendario/agenda
    if CALENDAR_RE.search(combined):
        return True

    # 3. Descripción demasiado corta para ser una noticia real
    #    (evento de calendario suele tener descripción de <10 palabras útiles)
    desc_words = len(description.split())
    if desc_words < MIN_DESCRIPTION_WORDS:
        return True

    # 4. Contenido educativo puro (BabyPips school, etc.)
    if EDUCATIONAL_RE.search(combined):
        return True

    return False


def has_real_content(title: str, description: str) -> bool:
    """
    Verifica que el artículo tenga contenido editorial real:
    - Título con al menos 5 palabras
    - Descripción con al menos MIN_DESCRIPTION_WORDS palabras
    - No es solo metadatos (fechas, horarios, tipos de contenido)
    """
    title_words = len(title.split())
    desc_words  = len(description.split())

    if title_words < 5:
        return False
    if desc_words < MIN_DESCRIPTION_WORDS:
        return False

    # Detectar si la descripción es casi idéntica al título (sin valor añadido)
    title_lower = title.lower().strip()
    desc_lower  = description.lower().strip()
    if desc_lower.startswith(title_lower[:30]) and desc_words < 20:
        return False

    return True


def is_forex_relevant(title: str, summary: str, source: str) -> bool:
    if source in FOREX_SOURCES:
        text = (title + " " + summary).lower()
        return any(kw in text for kw in FOREX_RELEVANCE_KW)
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_RELEVANCE_KW)


def detect_currency(title: str, summary: str, source: str = "") -> str:
    text = (title + " " + summary).lower()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return currency
    return SOURCE_CURRENCY.get(source, "USD")


def detect_impact(title: str, summary: str) -> str:
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


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """Normaliza un título para detección de duplicados por similitud."""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def entry_id(title: str, source: str) -> str:
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg: dict):
    try:
        d = feedparser.parse(
            feed_cfg["url"],
            request_headers={
                "User-Agent": "Mozilla/5.0 (compatible; ForexNewsBot/2.0)",
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            }
        )
        if d.bozo and not d.entries:
            print(f"    [WARN] Feed malformado o vacío: {feed_cfg['url'][:65]}")
        return d.entries
    except Exception as e:
        print(f"  [ERROR] {feed_cfg['url'][:65]}: {e}")
        return []


def smart_select(articles, max_total, guaranteed_per_cur, max_per_cur):
    """
    Fase 1: garantía mínima por divisa (priorizando high > med > low).
    Fase 2: rellena slots restantes con los mejores del pool completo.
    """
    groups = {cur: [] for cur in CURRENCIES}
    for a in articles:
        cur = a.get("cur", "USD")
        if cur in groups:
            groups[cur].append(a)

    for cur in CURRENCIES:
        groups[cur].sort(key=lambda x: (IMPACT_ORDER.get(x["impact"], 2), -x["ts"]))

    selected_ids = set()
    selected     = []
    taken        = {cur: 0 for cur in CURRENCIES}

    # Fase 1 — garantía mínima
    for cur in CURRENCIES:
        for a in groups[cur]:
            if taken[cur] >= guaranteed_per_cur or len(selected) >= max_total:
                break
            selected.append(a)
            selected_ids.add(a["id"])
            taken[cur] += 1

    # Fase 2 — relleno por calidad
    remaining = [
        a for a in articles
        if a["id"] not in selected_ids and taken.get(a["cur"], 0) < max_per_cur
    ]
    remaining.sort(key=lambda x: (IMPACT_ORDER.get(x["impact"], 2), -x["ts"]))

    for a in remaining:
        if len(selected) >= max_total:
            break
        cur = a.get("cur", "USD")
        if taken.get(cur, 0) >= max_per_cur:
            continue
        selected.append(a)
        selected_ids.add(a["id"])
        taken[cur] = taken.get(cur, 0) + 1

    selected.sort(key=lambda x: x["ts"], reverse=True)
    return selected


def main():
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(days=MAX_AGE_DAYS)
    seen_ids     = set()
    seen_titles  = set()   # Para deduplicación por similitud de título
    raw_articles = []
    es_raw = en_raw = 0
    filtered_calendar = 0
    filtered_quality  = 0
    filtered_relevance = 0

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

            # ── Filtro de antigüedad ─────────────────────────────────────────
            pub_date = parse_date(entry)
            if pub_date < cutoff:
                continue

            # ── Filtro de eventos de calendario / contenido vacío ────────────
            # Esta es la corrección principal para el problema de Bank of Canada
            # y fuentes similares que publican avisos de agenda sin contenido
            if is_calendar_entry(title, summary):
                filtered_calendar += 1
                continue

            # ── Filtro de contenido mínimo real ──────────────────────────────
            if not has_real_content(title, summary):
                filtered_quality += 1
                continue

            # ── Filtro de relevancia forex ────────────────────────────────────
            if not is_forex_relevant(title, summary, source):
                filtered_relevance += 1
                continue

            # ── Deduplicación por ID exacto ───────────────────────────────────
            nid = entry_id(title, source)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            # ── Deduplicación por similitud de título (cross-source) ──────────
            norm_title = normalize_title(title)
            title_key  = norm_title[:60]  # primeros 60 chars normalizados
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            cur    = detect_currency(title, summary, source)
            impact = detect_impact(title, summary)
            expand = summary[:350] + ("..." if len(summary) > 350 else "")
            age_hours = (now_utc - pub_date).total_seconds() / 3600

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
                "recent":   age_hours <= 24,
            })
            count += 1
            if lang == "es":
                es_raw += 1
            else:
                en_raw += 1

        print(f"    → {count} noticias válidas")

    print(f"\n📦 Total artículos recopilados: {len(raw_articles)}")
    print(f"   ES: {es_raw} | EN: {en_raw}")
    print(f"   🚫 Descartados — Calendario/vacíos: {filtered_calendar} | Sin contenido: {filtered_quality} | Sin relevancia: {filtered_relevance}")

    dist_before   = Counter(a["cur"] for a in raw_articles)
    impact_before = Counter(a["impact"] for a in raw_articles)
    print(f"   Distribución: {dict(sorted(dist_before.items()))}")
    print(f"   Impacto: high={impact_before['high']} | med={impact_before['med']} | low={impact_before['low']}")

    missing = [c for c in CURRENCIES if dist_before.get(c, 0) == 0]
    if missing:
        print(f"   ⚠️  Sin artículos en {MAX_AGE_DAYS} días: {', '.join(missing)}")

    articles = smart_select(
        raw_articles,
        max_total=MAX_NEWS,
        guaranteed_per_cur=GUARANTEED_PER_CUR,
        max_per_cur=MAX_PER_CUR,
    )

    dist_after   = Counter(a["cur"] for a in articles)
    impact_after = Counter(a["impact"] for a in articles)
    recent_count = sum(1 for a in articles if a.get("recent", True))
    print(f"\n✂️  Selección final ({len(articles)} artículos | garantía: {GUARANTEED_PER_CUR}/divisa | máx: {MAX_PER_CUR}/divisa):")
    print(f"   Distribución: {dict(sorted(dist_after.items()))}")
    print(f"   Impacto: high={impact_after['high']} | med={impact_after['med']} | low={impact_after['low']}")
    print(f"   Recientes (<24h): {recent_count} | Históricos (1-3 días): {len(articles)-recent_count}")

    sources_ok = sorted(set(a["source"] for a in articles))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(articles)} artículos guardados en {OUTPUT_FILE}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
