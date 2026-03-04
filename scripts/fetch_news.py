#!/usr/bin/env python3
"""
fetch_news.py — v5
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.

CAMBIOS v5 (sobre v4):
  - detect_currency() integra la misma lógica de scoring de news.html v5.3:
      • PASO 1: Par de divisas explícito en el TÍTULO (EUR/USD, NZD/USD, AUD/JPY...)
                → asigna divisa protagonista según PAIR_PROTAGONIST_MAP
                → resuelve el bug principal: "NZD/USD" asignado a USD, "EUR/GBP" a GBP
      • PASO 2: Scoring acumulativo por pesos (igual que v4, sin cambios)
      • Eliminados duplicados: la lógica de pares ya cubre EUR/GBP, AUD/JPY, etc.
  - Añadido keyword 'strait of hormuz' → USD (peso 9) en CURRENCY_KEYWORDS_WEIGHTED
    → resuelve "US could provide military protection to Strait of Hormuz" → NZD→USD
  - 'trump ' (con espacio) → USD peso 9 (geopolítica liderada por EEUU)
  - PAIR_PROTAGONIST_MAP sincronizado con news.html para coherencia
  - detect_currency() sigue retornando None si ninguna divisa alcanza el umbral
  - Sin cambios en feeds, smart_select, impacto, ni estructura del JSON
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

# Umbral mínimo de score para asignar divisa por keywords
# (el paso de pares explícitos ignora este umbral)
CURRENCY_MIN_SCORE = 3

# ─────────────────────────────────────────────
# PAR EXPLÍCITO → DIVISA PROTAGONISTA
# Sincronizado con PAIR_PROTAGONIST_MAP de news.html v5.3
# El protagonista es la divisa BASE del par (la que se está analizando/moviendo).
# ─────────────────────────────────────────────
PAIR_PROTAGONIST_MAP = {
    'EUR/USD': 'EUR', 'EURUSD': 'EUR',
    'GBP/USD': 'GBP', 'GBPUSD': 'GBP',
    'USD/JPY': 'JPY', 'USDJPY': 'JPY',
    'AUD/USD': 'AUD', 'AUDUSD': 'AUD',
    'NZD/USD': 'NZD', 'NZDUSD': 'NZD',
    'USD/CAD': 'CAD', 'USDCAD': 'CAD',
    'USD/CHF': 'CHF', 'USDCHF': 'CHF',
    'EUR/GBP': 'EUR', 'EURGBP': 'EUR',
    'EUR/JPY': 'EUR', 'EURJPY': 'EUR',
    'GBP/JPY': 'GBP', 'GBPJPY': 'GBP',
    'AUD/JPY': 'AUD', 'AUDJPY': 'AUD',
    'EUR/AUD': 'EUR', 'EURAUD': 'EUR',
    'GBP/AUD': 'GBP', 'GBPAUD': 'GBP',
    'AUD/CHF': 'AUD', 'AUDCHF': 'AUD',
    'EUR/CAD': 'EUR', 'EURCAD': 'EUR',
    'GBP/CHF': 'GBP', 'GBPCHF': 'GBP',
    'NZD/JPY': 'NZD', 'NZDJPY': 'NZD',
    'CAD/JPY': 'CAD', 'CADJPY': 'CAD',
    'NZD/CAD': 'NZD', 'NZDCAD': 'NZD',
    'EUR/NZD': 'EUR', 'EURNZD': 'EUR',
    'GBP/NZD': 'GBP', 'GBPNZD': 'GBP',
    'EUR/CHF': 'EUR', 'EURCHF': 'EUR',
    'AUD/NZD': 'AUD', 'AUDNZD': 'AUD',
    'GBP/CAD': 'GBP', 'GBPCAD': 'GBP',
    'CHF/JPY': 'CHF', 'CHFJPY': 'CHF',
    'NZD/CHF': 'NZD', 'NZDCHF': 'NZD',
    'AUD/CAD': 'AUD', 'AUDCAD': 'AUD',
}

# ─────────────────────────────────────────────
# PATRONES DE CALENDARIO
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
# KEYWORDS CON PESOS — v5
# Cambios vs v4:
#   - USD: añadidos 'strait of hormuz' (peso 9), 'trump ' (peso 9),
#          'us military', 'us navy', 'white house', 'pentagon' (peso 9)
#   - Sin otros cambios de keywords (el paso de pares explícitos
#     ya resuelve la mayoría de los falsos positivos anteriores)
# ─────────────────────────────────────────────
CURRENCY_KEYWORDS_WEIGHTED = {
    "USD": [
        # Banco central / institucional (10)
        ("fed ", 10), ("federal reserve", 10), ("fomc", 10), ("powell", 10),
        ("us treasury", 10), ("reserva federal", 10),
        # Geopolítica liderada por EEUU (9) — NUEVO v5
        ("strait of hormuz", 9), ("estrecho de ormuz", 9),
        ("us military", 9), ("us navy", 9), ("us sanctions", 9),
        ("us-iran", 9), ("us iran conflict", 9), ("us strikes", 9),
        ("trump ", 9), ("white house", 9), ("pentagon", 9),
        # Tickers y pares (5) — sin pares cruzados (manejados por PAIR_PROTAGONIST_MAP)
        ("usd/", 5), ("dollar index", 5), ("dxy", 5), ("us dollar", 5),
        ("dólar estadounidense", 5),
        # Macro EEUU (8)
        ("us economy", 8), ("us gdp", 8), ("us cpi", 8), ("us inflation", 8),
        ("us jobs", 8), ("nonfarm", 8), ("non-farm payroll", 8),
        ("jobless claims", 8), ("american economy", 8),
        # Mercados EEUU (3)
        ("treasury yield", 3), ("wall street", 3), ("nasdaq", 3),
        ("dow jones", 3), ("s&p 500", 3), ("us 10-year", 3),
        # Débil (1)
        ("dollar", 1), ("dólar", 1), ("tariff", 1), ("arancel", 1),
        ("ism ", 1), ("estados unidos", 1), ("united states", 1),
    ],
    "EUR": [
        # Banco central / institucional (10)
        ("ecb", 10), ("european central bank", 10), ("lagarde", 10),
        ("kazaks", 10), ("schnabel", 10), ("de guindos", 10),
        ("bce", 10), ("banco central europeo", 10),
        # Zona euro (8)
        ("euro area", 8), ("eurozone", 8), ("euro zone", 8), ("eurozona", 8),
        ("zona euro", 8),
        # Macro Eurozona (5)
        ("germany gdp", 5), ("german cpi", 5), ("ifo", 5), ("zew", 5),
        ("bund", 5), ("eu gdp", 5), ("eurozone inflation", 5), ("eurozone gdp", 5),
        # Tickers (5)
        ("eur/", 5), ("/eur", 5), ("euro ", 2),
        # Débil (1)
        ("eu economy", 1), ("european economy", 1),
        ("economía europea", 1), ("alemania", 1),
        ("france", 1), ("italy", 1), ("spain", 1),
    ],
    "GBP": [
        # Banco central / institucional (10)
        ("boe", 10), ("bank of england", 10), ("bailey", 10),
        ("mpc meeting", 10), ("mpc decision", 10),
        ("banco de inglaterra", 10),
        # Sterling (8)
        ("sterling", 8), ("pound sterling", 8), ("libra esterlina", 8),
        ("uk gilt", 8), ("gilt yield", 8), ("gilts", 8),
        # Macro UK (5)
        ("uk economy", 5), ("united kingdom", 5), ("britain", 5),
        ("uk gdp", 5), ("uk inflation", 5), ("uk jobs", 5),
        ("uk cpi", 5), ("reino unido", 5),
        # Tickers (5)
        ("gbp/", 5), ("/gbp", 5),
        # Débil (1) — "pound" solo puede ser unidad de medida
        ("pound", 1), ("british", 1), ("ftse", 1), ("brexit", 1),
    ],
    "JPY": [
        # Banco central / institucional (10)
        ("boj", 10), ("bank of japan", 10), ("ueda", 10), ("himino", 10),
        ("kuroda", 10), ("banco de japón", 10),
        # Yen (8)
        ("japanese yen", 8), ("yen japonés", 8),
        # Macro Japón (5)
        ("japan economy", 5), ("japanese economy", 5), ("economía japonesa", 5),
        ("japan gdp", 5), ("japan inflation", 5), ("japan cpi", 5),
        ("japan pmi", 5), ("japan trade", 5), ("japan unemployment", 5),
        # Tickers (5)
        ("jpy ", 5), ("jpy/", 5), ("/jpy", 5), ("usd/jpy", 5),
        # Débil (1)
        ("yen ", 1), ("nikkei", 1), ("japanese", 1), ("japan ", 1),
        ("japón ", 1),
    ],
    "AUD": [
        # Banco central / institucional (10)
        ("rba", 10), ("reserve bank of australia", 10), ("bullock", 10),
        ("banco de la reserva de australia", 10),
        # Dólar australiano (8)
        ("australian dollar", 8), ("dólar australiano", 8), ("aussie dollar", 8),
        # Macro Australia (5)
        ("australia gdp", 5), ("australian gdp", 5),
        ("australia inflation", 5), ("australia cpi", 5),
        ("australia trade", 5), ("australia retail", 5),
        ("australia jobs", 5), ("australian jobs", 5),
        ("australian economy", 5), ("australia economy", 5),
        # Tickers (5)
        ("aud/", 5), ("/aud", 5), ("aud ", 3), ("aussie ", 3),
        # Débil (1)
        ("australia", 1),
    ],
    "CAD": [
        # Banco central / institucional (10)
        ("boc", 10), ("bank of canada", 10), ("macklem", 10),
        ("banco de canadá", 10),
        # Dólar canadiense (8)
        ("canadian dollar", 8), ("dólar canadiense", 8), ("loonie", 8),
        # Mercado canadiense (8)
        ("tsx", 8), ("s&p/tsx", 8),
        # Macro Canadá (5)
        ("canada economy", 5), ("economía canadá", 5),
        ("canada gdp", 5), ("pib canadá", 5),
        ("canada inflation", 5), ("canada trade", 5), ("canada jobs", 5),
        ("canadian economy", 5),
        # Tickers (5)
        ("cad/", 5), ("/cad", 5), ("usd/cad", 5), ("cad ", 3),
        # Petróleo (correlación CAD, pero solo peso 1 para evitar falsos positivos
        # en artículos puramente geopolíticos — el scoring de USD con peso 9 gana)
        ("crude oil", 1), ("wti ", 1), ("brent", 1), ("petróleo", 1),
        ("oil prices", 1), ("opec", 1),
        # Débil (1)
        ("canada ", 1), ("canadá ", 1),
    ],
    "CHF": [
        # Banco central / institucional (10)
        ("snb", 10), ("swiss national bank", 10), ("jordan", 10),
        ("schlegel", 10), ("banco nacional suizo", 10),
        # Franco suizo (8)
        ("swiss franc", 8), ("franco suizo", 8),
        # Macro Suiza (5)
        ("switzerland", 5), ("swiss economy", 5), ("swiss inflation", 5),
        ("swiss cpi", 5), ("switzerland gdp", 5), ("swiss pmi", 5),
        ("swiss kof", 5), ("suiza", 5),
        # Tickers (5)
        ("chf/", 5), ("/chf", 5), ("usd/chf", 5), ("chf ", 3),
        # Débil (1)
        ("swiss ", 1),
    ],
    "NZD": [
        # Banco central / institucional (10)
        ("rbnz", 10), ("reserve bank of new zealand", 10), ("orr", 10),
        ("banco de la reserva de nueva zelanda", 10),
        # Dólar neozelandés (8)
        ("new zealand dollar", 8), ("dólar neozelandés", 8), ("kiwi dollar", 8),
        # Macro NZ (5)
        ("new zealand gdp", 5), ("nz gdp", 5), ("nz cpi", 5),
        ("nz economy", 5), ("nz pmi", 5), ("nz trade", 5), ("nz retail", 5),
        ("nz jobs", 5), ("new zealand inflation", 5), ("new zealand trade", 5),
        ("nueva zelanda", 5),
        # Tickers (5)
        ("nzd/", 5), ("/nzd", 5), ("nzd ", 3), ("kiwi ", 3),
        # Débil (1)
        ("new zealand", 1),
    ],
}

# ─────────────────────────────────────────────
# FALSE POSITIVE GUARDS — sin cambios vs v4
# ─────────────────────────────────────────────
FALSE_POSITIVE_GUARDS = [
    {
        "pattern": re.compile(r"\bper pound\b|\bcents? per pound\b|\bper lb\b", re.IGNORECASE),
        "penalize": {"GBP": 2},
    },
    {
        "pattern": re.compile(r"\b(sugar|coffee|cotton|cocoa|wheat|corn|grain|commodity|commodit)\b", re.IGNORECASE),
        "penalize": {"GBP": 2},
    },
    {
        "pattern": re.compile(r"\b(turkey|turkish|bist|istanbul|ankara|lira)\b", re.IGNORECASE),
        "penalize": {"AUD": 3, "NZD": 3},
    },
    {
        "pattern": re.compile(r"\b(ibovespa|bovespa|brazil|brasil|real brasileiro|brl)\b", re.IGNORECASE),
        "penalize": {"CAD": 3, "AUD": 2},
    },
    {
        "pattern": re.compile(r"\b(silver|platinum|palladium|precious metal)\b", re.IGNORECASE),
        "penalize": {"NZD": 2, "AUD": 1},
    },
    {
        "pattern": re.compile(r"\b(rand|south african|zar|johannesburg|pretoria)\b", re.IGNORECASE),
        "penalize": {"USD": 2, "AUD": 2, "NZD": 2},
    },
    {
        "pattern": re.compile(r"\b(shanghai|shenzhen|hang seng|csi 300|yuan|renminbi|cny)\b", re.IGNORECASE),
        "penalize": {"CAD": 2, "AUD": 1},
    },
]

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
    # Geopolítica sistémica — v5
    "strait of hormuz", "estrecho de ormuz", "war ", "guerra ",
    "military strike", "ataque militar", "sanctions", "sanciones",
]

MED_IMPACT_KW = [
    "pmi", "employment", "jobless", "trade balance", "retail sales",
    "industrial production", "consumer confidence", "business confidence",
    "housing", "wages", "earnings", "exports", "imports", "deficit",
    "surplus", "forecast", "outlook", "guidance", "payroll", "manufacturing",
    "pmi manufacturero", "desempleo", "balanza comercial", "ventas minoristas",
    "producción industrial", "confianza del consumidor", "salarios",
    # Petróleo → MED (era LOW, incorrecto para sesiones de energía)
    "crude oil", "oil prices", "petróleo", "brent", "wti",
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
    v5: añade PASO 1 (par explícito en título) antes del scoring.

    PASO 1: busca par de divisas explícito en el TÍTULO.
            Si el título empieza con "EUR/USD", "NZD/USD:", "AUD/JPY breaks..."
            → asigna la divisa protagonista (generalmente la base del par).
            Esto resuelve: "NZD/USD: bajistas acechan" → NZD (no USD)
                           "EUR/GBP se suaviza" → EUR (no GBP)
                           "AUD/JPY breaks high" → AUD (no JPY)

    PASO 2: scoring ponderado (igual que v4).
            Resuelve: "US could provide military protection to Strait of Hormuz"
                      → USD (peso 9 por 'strait of hormuz') > CAD (peso 1 por 'oil')

    PASO 3: fallback a SOURCE_CURRENCY si la fuente es institucional conocida.

    PASO 4: retorna None → artículo descartado.
    """
    title_clean = (title or '').strip()
    title_upper = title_clean.upper().replace(' ', '')

    # ── PASO 1: par explícito en el título ──────────────────────────────────
    # Busca primero al inicio del título (primeros 35 chars), luego en todo el título
    title_start_upper = title_clean.upper()[:35].replace(' ', '')
    for pair, protagonist in PAIR_PROTAGONIST_MAP.items():
        pair_noslash = pair.replace('/', '').upper()
        pair_slash   = pair.upper()
        # Inicio del título (prioridad máxima)
        if pair_noslash in title_start_upper or pair_slash in title_clean.upper()[:35]:
            return protagonist
    # En cualquier parte del título
    for pair, protagonist in PAIR_PROTAGONIST_MAP.items():
        pair_noslash = pair.replace('/', '').upper()
        if pair_noslash in title_upper or pair.upper() in title_clean.upper():
            return protagonist

    # ── PASO 2: scoring acumulativo ──────────────────────────────────────────
    text   = (title_clean + " " + (summary or '')).lower()
    title_lower = title_clean.lower()
    scores: dict[str, float] = {cur: 0.0 for cur in CURRENCIES}

    for cur, kws in CURRENCY_KEYWORDS_WEIGHTED.items():
        for kw, weight in kws:
            if kw in text:
                scores[cur] += weight
                # Bonus 50% si la keyword está en el título
                if kw in title_lower:
                    scores[cur] += weight * 0.5

    # Penalizaciones de falsos positivos
    for guard in FALSE_POSITIVE_GUARDS:
        if guard["pattern"].search(text):
            for cur, penalty in guard["penalize"].items():
                scores[cur] = max(0, scores[cur] - penalty)

    best_cur   = max(scores, key=lambda c: scores[c])
    best_score = scores[best_cur]

    if best_score >= CURRENCY_MIN_SCORE:
        return best_cur

    # ── PASO 3: fallback institucional ───────────────────────────────────────
    if source in SOURCE_CURRENCY:
        return SOURCE_CURRENCY[source]

    # ── PASO 4: sin divisa confiable ─────────────────────────────────────────
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
                prev[a["id"]] = {
                    "ai_headline": a["ai_headline"],
                    "sentiment":   a.get("sentiment", "neut"),
                }
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
    seen_ids             = set()
    seen_titles          = set()
    raw_articles         = []
    es_raw = en_raw      = 0
    filtered_calendar    = 0
    filtered_quality     = 0
    filtered_relevance   = 0
    filtered_no_currency = 0

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_news.py v5 — {len(FEEDS)} feeds")

    print(f"  Descargando en paralelo (workers={FETCH_WORKERS})...")
    all_entries = fetch_all_feeds(FEEDS)
    print(f"  Descarga completada.")

    for feed_cfg in FEEDS:
        source  = feed_cfg["source"]
        lang    = feed_cfg.get("lang", "en")
        entries = all_entries.get(feed_cfg["url"], [])
        count   = 0

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

            cur = detect_currency(title, summary, source)
            if cur is None:
                filtered_no_currency += 1
                continue

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
    print(f"      Calendario/vacíos:    {filtered_calendar}")
    print(f"      Sin contenido:        {filtered_quality}")
    print(f"      Sin relevancia FX:    {filtered_relevance}")
    print(f"      Sin divisa confiable: {filtered_no_currency}")

    dist_before   = Counter(a["cur"] for a in raw_articles)
    impact_before = Counter(a["impact"] for a in raw_articles)
    print(f"   Distribución: {dict(sorted(dist_before.items()))}")
    print(f"   Impacto: high={impact_before['high']} | med={impact_before['med']} | low={impact_before['low']}")

    missing = [c for c in CURRENCIES if dist_before.get(c, 0) == 0]
    if missing:
        print(f"   ⚠️  Sin artículos en {MAX_AGE_DAYS} días: {', '.join(missing)}")

    # Reutilizar titulares y sentimientos del JSON anterior
    prev_data = load_previous_headlines()
    reused = 0
    for a in raw_articles:
        if a["id"] in prev_data:
            prev = prev_data[a["id"]]
            a["ai_headline"] = prev["ai_headline"]
            a["sentiment"]   = prev.get("sentiment", "neut")
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
