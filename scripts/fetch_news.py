#!/usr/bin/env python3
"""
fetch_news.py — v4
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.

CAMBIOS v4:
  - detect_currency() reescrito con sistema de scoring por peso:
      • Keywords primarios (nombre del banco central, par de divisas explícito) = 3 pts
      • Keywords secundarios (nombre país/economía, indicador macroeconómico) = 1 pt
      • Mínimo 3 puntos para asignar divisa; si ninguna alcanza → descartar artículo
      • Resuelve falsos positivos clásicos:
          "per pound" → GBP  (sugar futures)
          "australia" en el expand de un artículo sobre Turquía → AUD
          "oil prices" → CAD  (cuando el artículo es sobre otra región)
  - detect_currency() ya no tiene fallback a "USD":
      si ninguna divisa alcanza el umbral → retorna None → artículo descartado
  - Filtro de "relevancia de divisa confirmada": reemplaza is_forex_relevant()
    con una validación más estricta basada en el propio score de detect_currency()
  - TSX ahora detecta correctamente CAD (keyword "tsx" añadido)
  - FALSE POSITIVE GUARDS: "per pound", "sugar", "silver", "gold" como artículos
    de commodities puros no triggerean GBP/NZD/CAD salvo que haya contexto forex real
"""

import json
import re
import hashlib
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ─────────────────────────────────────────────
MAX_NEWS              = 35
MAX_AGE_DAYS          = 4
GUARANTEED_PER_CUR    = 2
MAX_PER_CUR           = 6
OUTPUT_FILE           = "news-data/news.json"
IMPACT_ORDER          = {"high": 0, "med": 1, "low": 2}
CURRENCIES            = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
FETCH_TIMEOUT         = 12
FETCH_WORKERS         = 8
MIN_DESCRIPTION_WORDS = 12

# Umbral mínimo de score para asignar una divisa.
# 3 puntos = al menos un keyword primario (banco central, par FX explícito)
# o tres keywords secundarios (economía + indicador + activo).
CURRENCY_MIN_SCORE = 3

# ─────────────────────────────────────────────
# PATRONES DE CALENDARIO / EVENTOS FUTUROS
# ─────────────────────────────────────────────
CALENDAR_PATTERNS = [
    r"upcoming event", r"content type.*upcoming", r"scheduled date",
    r"eight scheduled", r"on eight", r"share this page by email",
    r"governing council presents", r"\bpress release explaining\b",
    r"four times a year.*governing", r"announces the setting for the overnight rate",
    r"bank of canada announces",
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$",
    r"^\d{1,2}:\d{2}\s*\(et\)", r"content type\(s\):",
    r"^on (eight|four|six) scheduled dates",
    r"^the bank of (canada|england|japan|reserve)",
    r"monetary policy report$", r"rate announcement$",
    r"^(what is|how to|learn|guide to|introduction to|basics of|beginner)",
]
CALENDAR_RE = re.compile("|".join(CALENDAR_PATTERNS), re.IGNORECASE)

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

EDUCATIONAL_ONLY_PATTERNS = [
    r"school of pipsology", r"babypips quiz", r"^quiz:", r"^lesson \d+", r"pips? glossary",
]
EDUCATIONAL_RE = re.compile("|".join(EDUCATIONAL_ONLY_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────
# CURRENCY KEYWORDS CON PESOS
# Cada entrada: (keyword, peso)
#   peso 3 = señal fuerte (banco central, ticker FX, par de divisas explícito)
#   peso 1 = señal débil  (nombre de país, indicador macroeconómico genérico)
#
# NOTA IMPORTANTE sobre falsos positivos:
#   "pound" (peso=1 para GBP) puede matchear "per pound" (unidad de medida)
#   → Se mitiga con FALSE_POSITIVE_GUARDS en detect_currency()
# ─────────────────────────────────────────────
CURRENCY_KEYWORDS_WEIGHTED = {
    "USD": [
        # Fuerte (3)
        ("fed ", 3), ("federal reserve", 3), ("fomc", 3), ("powell", 3),
        ("usd/", 3), ("/usd", 3), ("usd/jpy", 3), ("eur/usd", 3), ("gbp/usd", 3),
        ("aud/usd", 3), ("nzd/usd", 3), ("usd/cad", 3), ("usd/chf", 3),
        ("dxy", 3), ("dollar index", 3), ("us dollar", 3),
        ("treasury yield", 3), ("us 10-year", 3), ("10-year treasury", 3),
        ("nonfarm", 3), ("non-farm payroll", 3),
        # Moderado (2)
        ("dollar", 2), ("usd ", 2),
        # Débil (1)
        ("us economy", 1), ("us gdp", 1), ("us inflation", 1),
        ("wall street", 1), ("nasdaq", 1), ("dow jones", 1), ("s&p 500", 1),
        ("us jobs", 1), ("american economy", 1), ("jobless claims", 1),
        ("tariff", 1), ("arancel", 1), ("reserva federal", 1),
        ("dólar", 1), ("economía de eeuu", 1), ("pib eeuu", 1),
        ("inflación eeuu", 1), ("bonos del tesoro", 1),
        # ISM — contexto EEUU específico
        ("ism ", 1),
    ],
    "EUR": [
        # Fuerte (3)
        ("ecb", 3), ("european central bank", 3), ("lagarde", 3),
        ("eur/usd", 3), ("eur/gbp", 3), ("eur/jpy", 3), ("eur/chf", 3),
        ("euro area", 3), ("eurozone", 3), ("euro zone", 3),
        ("bce", 3), ("banco central europeo", 3),
        # Moderado (2)
        ("euro ", 2), ("eur ", 2),
        # Débil (1)
        ("eurozona", 1), ("germany", 1), ("france", 1), ("italy", 1), ("spain", 1),
        ("ifo", 1), ("zew", 1), ("eu economy", 1), ("european economy", 1),
        ("bund", 1), ("eu gdp", 1), ("alemania", 1), ("zona euro", 1),
        ("inflación zona euro", 1), ("pib zona euro", 1),
    ],
    "GBP": [
        # Fuerte (3)
        ("boe", 3), ("bank of england", 3), ("bailey", 3),
        ("gbp/usd", 3), ("eur/gbp", 3), ("gbp/jpy", 3),
        ("uk gilt", 3), ("gilt yield", 3), ("gilts", 3),
        ("sterling", 3), ("libra esterlina", 3),
        ("banco de inglaterra", 3),
        # Moderado (2)
        ("gbp ", 2), ("pound sterling", 2),
        # Débil (1) — NOTA: "pound" solo sin contexto = peso 1 (puede ser unidad de medida)
        ("uk economy", 1), ("united kingdom", 1), ("britain", 1),
        ("uk gdp", 1), ("uk inflation", 1), ("uk jobs", 1),
        ("reino unido", 1), ("brexit", 1),
        # "pound" aislado = 1 (falso positivo mitigado por guard)
        ("pound", 1),
    ],
    "JPY": [
        # Fuerte (3)
        ("boj", 3), ("bank of japan", 3), ("ueda", 3), ("himino", 3),
        ("usd/jpy", 3), ("eur/jpy", 3), ("gbp/jpy", 3), ("aud/jpy", 3),
        ("japanese yen", 3), ("yen japonés", 3),
        ("banco de japón", 3),
        # Moderado (2)
        ("jpy ", 2), ("jpy/", 2), ("/jpy", 2),
        ("yen ", 2),
        # Débil (1)
        ("japan economy", 1), ("japanese", 1), ("nikkei", 1),
        ("japan gdp", 1), ("japan inflation", 1), ("japan cpi", 1),
        ("japan pmi", 1), ("japan trade", 1), ("japan unemployment", 1),
        ("economía japonesa", 1), ("pib japón", 1),
    ],
    "AUD": [
        # Fuerte (3)
        ("rba", 3), ("reserve bank of australia", 3), ("bullock", 3),
        ("aud/usd", 3), ("aud/jpy", 3), ("aud/nzd", 3),
        ("australian dollar", 3), ("dólar australiano", 3),
        ("aussie dollar", 3),
        # Moderado (2)
        ("aud ", 2), ("aud/", 2), ("/aud", 2),
        ("aussie ", 2),
        # Débil (1)
        ("australia gdp", 1), ("australian gdp", 1),
        ("australia inflation", 1), ("australia cpi", 1),
        ("australia trade", 1), ("australia retail", 1),
        ("australia jobs", 1), ("australian jobs", 1),
        ("australian economy", 1), ("australia economy", 1),
        ("banco de la reserva de australia", 1),
        # "australia" solo vale 1 — si aparece de pasada (ej. "mercados esperan GDP australiano")
        # no es suficiente para asignar AUD sin otro keyword
        ("australia", 1),
    ],
    "CAD": [
        # Fuerte (3)
        ("boc", 3), ("bank of canada", 3), ("macklem", 3),
        ("usd/cad", 3), ("cad/jpy", 3),
        ("canadian dollar", 3), ("dólar canadiense", 3),
        ("loonie", 3), ("banco de canadá", 3),
        # Mercado canadiense específico (3)
        ("tsx", 3), ("s&p/tsx", 3), ("s p/tsx", 3),
        # Moderado (2)
        ("cad ", 2), ("cad/", 2), ("/cad", 2),
        # Débil (1)
        ("canada economy", 1), ("economía canadá", 1),
        ("canada gdp", 1), ("pib canadá", 1),
        ("canada inflation", 1), ("inflación canadá", 1),
        ("canada trade", 1), ("canada jobs", 1),
        ("canadian economy", 1),
        # Petróleo = correlación CAD, pero solo vale 1 (contexto necesario)
        # Un artículo sobre petróleo en general no es automáticamente CAD
        ("crude oil", 1), ("wti ", 1), ("brent", 1), ("petróleo", 1),
        ("oil prices", 1), ("opec", 1),
    ],
    "CHF": [
        # Fuerte (3)
        ("snb", 3), ("swiss national bank", 3), ("jordan", 3), ("schlegel", 3),
        ("usd/chf", 3), ("eur/chf", 3), ("chf/jpy", 3),
        ("swiss franc", 3), ("franco suizo", 3),
        ("banco nacional suizo", 3),
        # Moderado (2)
        ("chf ", 2), ("chf/", 2), ("/chf", 2),
        # Débil (1)
        ("switzerland", 1), ("swiss economy", 1), ("swiss inflation", 1),
        ("swiss cpi", 1), ("switzerland gdp", 1), ("swiss pmi", 1),
        ("swiss kof", 1), ("suiza", 1),
    ],
    "NZD": [
        # Fuerte (3)
        ("rbnz", 3), ("reserve bank of new zealand", 3), ("orr", 3),
        ("nzd/usd", 3), ("aud/nzd", 3), ("nzd/jpy", 3),
        ("new zealand dollar", 3), ("dólar neozelandés", 3),
        ("kiwi dollar", 3),
        ("banco de la reserva de nueva zelanda", 3),
        # Moderado (2)
        ("nzd ", 2), ("nzd/", 2), ("/nzd", 2),
        ("kiwi ", 2),
        # Débil (1)
        ("new zealand", 1), ("nueva zelanda", 1),
        ("nz economy", 1), ("nz gdp", 1), ("nz cpi", 1),
        ("nz pmi", 1), ("nz trade", 1), ("nz retail", 1),
        ("nz jobs", 1), ("new zealand inflation", 1),
        ("new zealand gdp", 1), ("new zealand trade", 1),
    ],
}

# ─────────────────────────────────────────────
# FALSE POSITIVE GUARDS
# Si alguno de estos patterns está presente en el texto,
# se penaliza el score de la divisa indicada.
# Evita que "per pound" → GBP, "silver" → NZD, etc.
# ─────────────────────────────────────────────
FALSE_POSITIVE_GUARDS = [
    # "per pound" como unidad de peso/precio → penalizar GBP
    {
        "pattern": re.compile(r"\bper pound\b|\bcents? per pound\b|\bper lb\b", re.IGNORECASE),
        "penalize": {"GBP": 2},
    },
    # "pound" en contexto de peso/alimentos → penalizar GBP si no hay sterling/UK
    {
        "pattern": re.compile(r"\b(sugar|coffee|cotton|cocoa|wheat|corn|grain|commodity|commodit)\b", re.IGNORECASE),
        "penalize": {"GBP": 2},
    },
    # Artículo sobre bolsa turca/emergente sin nexo AUD → penalizar AUD, NZD
    {
        "pattern": re.compile(r"\b(turkey|turkish|bist|istanbul|ankara|lira)\b", re.IGNORECASE),
        "penalize": {"AUD": 3, "NZD": 3},
    },
    # Artículo sobre bolsa brasileña sin nexo CAD → penalizar CAD
    {
        "pattern": re.compile(r"\b(ibovespa|bovespa|brazil|brasil|real brasileiro|brl)\b", re.IGNORECASE),
        "penalize": {"CAD": 3, "AUD": 2},
    },
    # Plata/metales preciosos puros sin FX → penalizar NZD, AUD si no hay keywords propios
    {
        "pattern": re.compile(r"\b(silver|platinum|palladium|precious metal)\b", re.IGNORECASE),
        "penalize": {"NZD": 2, "AUD": 1},
    },
    # Rand sudafricano → no debería asignarse a USD
    {
        "pattern": re.compile(r"\b(rand|south african|zar|johannesburg|pretoria)\b", re.IGNORECASE),
        "penalize": {"USD": 2, "AUD": 2, "NZD": 2},
    },
    # Bolsa china → no CAD
    {
        "pattern": re.compile(r"\b(shanghai|shenzhen|hang seng|csi 300|yuan|renminbi|cny)\b", re.IGNORECASE),
        "penalize": {"CAD": 2, "AUD": 1},
    },
]

# Artículos con estos patrones en el título son de mercados emergentes/commodities
# y deben tener un score forex muy alto para ser incluidos
EMERGING_MARKET_TITLE_RE = re.compile(
    r"\b(ibovespa|bovespa|bist 100|istanbul|turkish stocks?|south african rand|"
    r"rand weakens?|sugar futures?|silver (price|slammed|falls?)|"
    r"copper (price|falls?|rises?)|gold (price|slammed)|platinum|palladium|"
    r"crude oil (price|rises?|falls?)|brent (price|rises?|falls?)|"
    r"turkish (lira|assets?)|emerging market)\b",
    re.IGNORECASE,
)

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
    # ── INGLÉS ───────────────────────────────────────────────────────────────
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/news",                              "lang": "en" },
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/analysis",                          "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/news",                             "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/centralbank",                      "lang": "en" },
    { "source": "ECB",              "url": "https://www.ecb.europa.eu/rss/press.html",                        "lang": "en" },
    { "source": "Bank of England",  "url": "https://www.bankofengland.co.uk/rss/news",                        "lang": "en" },
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
    "surprise", "unexpected", "powell", "lagarde", "ueda", "bailey", "bullock",
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
}

FOREX_SOURCES = {
    "FXStreet ES", "FXStreet", "ForexLive", "DailyForex ES", "DailyForex",
    "ECB", "Bank of England", "Bank of Japan", "RBA", "RBNZ", "SNB",
    "Federal Reserve", "ActionForex", "InvestingLive", "MyFXBook",
    "Investing.com", "InstaForex", "BabyPips", "InvestMacro", "ForexCrunch",
    "Investing.com ES",
}

# ─────────────────────────────────────────────
# FILTROS DE CALIDAD
# ─────────────────────────────────────────────

def is_calendar_entry(title: str, description: str) -> bool:
    combined = (title + " " + description).strip()
    if CALENDAR_TITLE_RE.search(title.strip()):
        return True
    if CALENDAR_RE.search(combined):
        return True
    if len(description.split()) < MIN_DESCRIPTION_WORDS:
        return True
    if EDUCATIONAL_RE.search(combined):
        return True
    return False


def has_real_content(title: str, description: str) -> bool:
    if len(title.split()) < 5:
        return False
    if len(description.split()) < MIN_DESCRIPTION_WORDS:
        return False
    title_lower = title.lower().strip()
    desc_lower  = description.lower().strip()
    if desc_lower.startswith(title_lower[:30]) and len(description.split()) < 20:
        return False
    return True


def is_forex_relevant(title: str, summary: str) -> bool:
    """
    Verificación liviana de relevancia forex.
    detect_currency() hace el filtrado pesado; este es el pre-filtro.
    """
    FOREX_RELEVANCE_KW = [
        "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
        "dollar", "dólar", "euro", "pound sterling", "yen", "franc", "franco",
        "forex", " fx ", "currency", "currencies", "divisa", "divisas",
        "fed", "bce", "ecb", "boe", "boj", "rba", "boc", "snb", "rbnz",
        "banco central", "central bank", "interest rate", "tasa de interés",
        "inflation", "inflación", "gdp", "pib", "cpi", "ipc",
        "unemployment", "desempleo", "payroll", "pmi", "retail sales",
        "recession", "recesión", "monetary policy", "política monetaria",
        "yield", "treasury", "gilt", "bond",
        "tariff", "arancel", "oil", "petróleo",
    ]
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_RELEVANCE_KW)


def detect_currency(title: str, summary: str, source: str = "") -> str | None:
    """
    Asigna la divisa más relevante usando un sistema de scoring ponderado.

    Retorna:
      str  → código de divisa (USD, EUR, GBP, ...)
      None → ninguna divisa alcanzó el umbral mínimo → artículo debe descartarse

    Lógica:
      1. Calcular score bruto por divisa sumando pesos de keywords encontrados
      2. Aplicar penalizaciones de FALSE_POSITIVE_GUARDS
      3. Si la divisa ganadora tiene score >= CURRENCY_MIN_SCORE → asignarla
      4. Si no → intentar SOURCE_CURRENCY como fallback conservador
      5. Si source tampoco está → retornar None
    """
    text = (title + " " + summary).lower()

    # Paso 1 — scoring bruto
    scores: dict[str, int] = {cur: 0 for cur in CURRENCIES}
    for cur, kws in CURRENCY_KEYWORDS_WEIGHTED.items():
        for kw, weight in kws:
            if kw in text:
                scores[cur] += weight

    # Paso 2 — penalizaciones de falsos positivos
    for guard in FALSE_POSITIVE_GUARDS:
        if guard["pattern"].search(text):
            for cur, penalty in guard["penalize"].items():
                scores[cur] = max(0, scores[cur] - penalty)

    # Paso 3 — divisa con mayor score
    best_cur   = max(scores, key=lambda c: scores[c])
    best_score = scores[best_cur]

    if best_score >= CURRENCY_MIN_SCORE:
        return best_cur

    # Paso 4 — fallback conservador: solo para fuentes institucionales conocidas
    if source in SOURCE_CURRENCY:
        return SOURCE_CURRENCY[source]

    # Paso 5 — no hay divisa confiable
    return None


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
                dt = dateparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def entry_id(title: str, source: str) -> str:
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg: dict):
    ua_map = {
        "ForexCrunch": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    ua = ua_map.get(feed_cfg.get("source", ""), "Mozilla/5.0 (compatible; ForexNewsBot/2.0)")
    try:
        resp = requests.get(
            feed_cfg["url"],
            headers={
                "User-Agent": ua,
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            },
            timeout=FETCH_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        d = feedparser.parse(resp.content)
        if d.bozo and not d.entries:
            return []
        return d.entries
    except Exception:
        return []


def fetch_all_feeds(feeds: list) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        future_to_feed = {executor.submit(fetch_via_feedparser, f): f for f in feeds}
        for future in as_completed(future_to_feed):
            feed_cfg = future_to_feed[future]
            try:
                entries = future.result()
                results[feed_cfg["url"]] = entries
            except Exception:
                results[feed_cfg["url"]] = []
    return results


def load_previous_headlines() -> dict:
    if not os.path.exists(OUTPUT_FILE):
        return {}
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        prev = {}
        for a in data.get("articles", []):
            if a.get("ai_headline") and a.get("id"):
                prev[a["id"]] = a["ai_headline"]
        return prev
    except Exception:
        return {}


def smart_select(articles, max_total, guaranteed_per_cur, max_per_cur):
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

    for cur in CURRENCIES:
        for a in groups[cur]:
            if taken[cur] >= guaranteed_per_cur or len(selected) >= max_total:
                break
            selected.append(a)
            selected_ids.add(a["id"])
            taken[cur] += 1

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
    seen_ids            = set()
    seen_titles         = set()
    raw_articles        = []
    es_raw = en_raw     = 0
    filtered_calendar   = 0
    filtered_quality    = 0
    filtered_relevance  = 0
    filtered_no_currency = 0

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] Iniciando fetch de {len(FEEDS)} feeds...")

    print(f"  Descargando {len(FEEDS)} feeds en paralelo (workers={FETCH_WORKERS})...")
    all_entries = fetch_all_feeds(FEEDS)
    print(f"  Descarga completada.")

    for feed_cfg in FEEDS:
        source = feed_cfg["source"]
        lang   = feed_cfg.get("lang", "en")
        entries = all_entries.get(feed_cfg["url"], [])
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

            if is_calendar_entry(title, summary):
                filtered_calendar += 1
                continue

            if not has_real_content(title, summary):
                filtered_quality += 1
                continue

            if not is_forex_relevant(title, summary):
                filtered_relevance += 1
                continue

            nid = entry_id(title, source)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            norm_title = normalize_title(title)
            title_key  = norm_title[:60]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # ── Detección de divisa con scoring ponderado ──────────────────
            cur = detect_currency(title, summary, source)
            if cur is None:
                filtered_no_currency += 1
                continue  # Artículo descartado: no tiene divisa confiable

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

        if count > 0:
            print(f"    [{lang.upper()}] {source}: {count} noticias")

    print(f"\n📦 Total artículos recopilados: {len(raw_articles)}")
    print(f"   ES: {es_raw} | EN: {en_raw}")
    print(f"   🚫 Descartados:")
    print(f"      Calendario/vacíos:   {filtered_calendar}")
    print(f"      Sin contenido:       {filtered_quality}")
    print(f"      Sin relevancia FX:   {filtered_relevance}")
    print(f"      Sin divisa confiable: {filtered_no_currency}  ← nuevo filtro v4")

    dist_before   = Counter(a["cur"] for a in raw_articles)
    impact_before = Counter(a["impact"] for a in raw_articles)
    print(f"   Distribución: {dict(sorted(dist_before.items()))}")
    print(f"   Impacto: high={impact_before['high']} | med={impact_before['med']} | low={impact_before['low']}")

    missing = [c for c in CURRENCIES if dist_before.get(c, 0) == 0]
    if missing:
        print(f"   ⚠️  Sin artículos en {MAX_AGE_DAYS} días: {', '.join(missing)}")

    prev_headlines = load_previous_headlines()
    reused = 0
    for a in raw_articles:
        if a["id"] in prev_headlines:
            a["ai_headline"] = prev_headlines[a["id"]]
            reused += 1
    if reused:
        print(f"   ♻️  Titulares reutilizados del JSON anterior: {reused}")

    articles = smart_select(
        raw_articles,
        max_total=MAX_NEWS,
        guaranteed_per_cur=GUARANTEED_PER_CUR,
        max_per_cur=MAX_PER_CUR,
    )

    dist_after   = Counter(a["cur"] for a in articles)
    impact_after = Counter(a["impact"] for a in articles)
    recent_count = sum(1 for a in articles if a.get("recent", True))
    print(f"\n✂️  Selección final ({len(articles)} artículos):")
    print(f"   Distribución: {dict(sorted(dist_after.items()))}")
    print(f"   Impacto: high={impact_after['high']} | med={impact_after['med']} | low={impact_after['low']}")
    print(f"   Recientes (<24h): {recent_count} | Históricos: {len(articles) - recent_count}")

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

    new_count    = sum(1 for a in articles if not a.get("ai_headline"))
    reused_final = sum(1 for a in articles if a.get("ai_headline"))
    print(f"\n✓ {len(articles)} artículos guardados en {OUTPUT_FILE}")
    print(f"  ♻️  Con titular previo: {reused_final}")
    print(f"  🆕 Pendientes de Groq: {new_count}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
