#!/usr/bin/env python3
"""
fetch_news.py — v5.6
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.

CAMBIOS v5.7 (sobre v5.6):
  GNEWS → NEWSDATA.IO:
    · GNews plan gratuito tiene 12h de delay — inutilizable para noticias FX.
    · Reemplazado por NewsData.io (https://newsdata.io) — plan gratuito:
      200 créditos/día, sin delay, artículos en tiempo real.
    · 3 ejecuciones/día × 8 divisas = 24 créditos — margen del 88% sobre límite.
    · Requiere secret NEWSDATA_API_KEY en GitHub Actions (Settings → Secrets).
    · Si la key no está configurada, el script continúa sin NewsData (no falla).
    · fetch_newsdata() corre secuencialmente con 2s entre queries.

CAMBIOS v5.4 (sobre v5.3):
  CALIDAD DE FUENTES:
    · FXStreet ES eliminado — duplicaba contenido de FXStreet EN traducido.
    · InstaForex penalizado: impacto forzado a "low" y cap de 2 artículos
      por ejecución (INSTAFOREX_MAX). Pasa a fallback de último recurso.
"""

import json
import re
import hashlib
import sys
import time
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# FIX C-01: Importar CURRENCIES desde el módulo compartido fx_config.py
sys.path.insert(0, os.path.dirname(__file__))
from fx_config import CURRENCIES

# ─────────────────────────────────────────────
MAX_NEWS              = 48          # v5.3: subido de 40 para absorber nuevos feeds
MAX_AGE_DAYS          = 4
GUARANTEED_PER_CUR    = 3
MAX_PER_CUR           = 8
OUTPUT_FILE           = "news-data/news.json"
IMPACT_ORDER          = {"high": 0, "med": 1, "low": 2}
INSTAFOREX_MAX        = 2           # v5.4: InstaForex limitado a 2 artículos por ejecución
# FIX C-01: CURRENCIES importado desde fx_config.py (ver imports al inicio del archivo)
FETCH_TIMEOUT         = 12
FETCH_WORKERS         = 14          # v5.3: subido para más feeds en paralelo
MIN_DESCRIPTION_WORDS = 12

# v5.3: subido de 3 a 5 para reducir falsos positivos
CURRENCY_MIN_SCORE = 5

# ─────────────────────────────────────────────
# PAR EXPLÍCITO → DIVISA PROTAGONISTA
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
# PATRONES DE FILTRO
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

EMERGING_MARKET_TITLE_RE = re.compile(
    r"\b(ibovespa|bovespa|bist 100|istanbul|turkish stocks?|south african rand|"
    r"rand weakens?|sugar futures?|silver (price|slammed|falls?)|"
    r"copper (price|falls?|rises?)|gold (price|slammed)|platinum|palladium|"
    r"crude oil (price|rises?|falls?)|brent (price|rises?|falls?)|"
    r"turkish (lira|assets?)|emerging market|"
    r"kospi|nikkei 225|hang seng|shanghai composite|sensex|"
    r"peso mexicano|real brasileiro|lira turca)\b",
    re.IGNORECASE,
)

# v5.3: nuevo filtro para índices bursátiles europeos que no son relevantes para divisas
EQUITY_INDEX_TITLE_RE = re.compile(
    r"\b(dax|cac 40|eurostoxx|euro stoxx|ibex 35|ftse mib|stoxx 600|aex|"
    r"bel 20|omx|wig20)\b.{0,40}\b(rises?|falls?|gains?|drops?|points?|higher|lower|climbs?|slides?)\b",
    re.IGNORECASE,
)

CRYPTO_NOISE_RE = re.compile(
    r"\b(bitcoin|ethereum|crypto|cryptocurrency|altcoin|blockchain|defi|nft|"
    r"solana|ripple|binance|dogecoin|litecoin|cardano|polkadot|avalanche|"
    r"trading recommendations? for (bitcoin|ethereum|crypto)|"
    r"crypto market|digital asset)\b",
    re.IGNORECASE,
)

CRYPTO_FOREX_BRIDGE_KW = [
    "central bank", "federal reserve", "ecb", "boe", "cbdc",
    "digital dollar", "digital euro", "regulation", "sec ruling",
    "us dollar", "usd stablecoin",
]

# ─────────────────────────────────────────────
# KEYWORDS CON PESOS
# ─────────────────────────────────────────────
CURRENCY_KEYWORDS_WEIGHTED = {
    "USD": [
        ("fed ", 10), ("federal reserve", 10), ("fomc", 10), ("powell", 10),
        ("us treasury", 10), ("reserva federal", 10),
        ("strait of hormuz", 9), ("estrecho de ormuz", 9),
        ("us military", 9), ("us navy", 9), ("us sanctions", 9),
        ("us-iran", 9), ("us iran conflict", 9), ("us strikes", 9),
        ("trump ", 9), ("white house", 9), ("pentagon", 9),
        ("usd/", 5), ("dollar index", 5), ("dxy", 5), ("us dollar", 5),
        ("dólar estadounidense", 5),
        ("us economy", 8), ("us gdp", 8), ("us cpi", 8), ("us inflation", 8),
        ("us jobs", 8), ("nonfarm", 8), ("non-farm payroll", 8),
        ("jobless claims", 8), ("american economy", 8),
        ("treasury yield", 3), ("wall street", 3), ("nasdaq", 3),
        ("dow jones", 3), ("s&p 500", 3), ("us 10-year", 3),
        ("dollar", 1), ("dólar", 1), ("tariff", 1), ("arancel", 1),
        ("ism ", 1), ("estados unidos", 1), ("united states", 1),
    ],
    "EUR": [
        ("ecb", 10), ("european central bank", 10), ("lagarde", 10),
        ("kazaks", 10), ("schnabel", 10), ("de guindos", 10),
        ("bce", 10), ("banco central europeo", 10),
        ("euro area", 8), ("eurozone", 8), ("euro zone", 8), ("eurozona", 8),
        ("zona euro", 8),
        ("germany gdp", 5), ("german cpi", 5), ("ifo", 5), ("zew", 5),
        ("bund", 5), ("eu gdp", 5), ("eurozone inflation", 5), ("eurozone gdp", 5),
        ("eur/", 5), ("/eur", 5), ("euro ", 2),
        ("eu economy", 1), ("european economy", 1),
        ("economía europea", 1), ("alemania", 1),
        ("france", 1), ("italy", 1), ("spain", 1),
    ],
    "GBP": [
        ("boe", 10), ("bank of england", 10), ("bailey", 10),
        ("mpc meeting", 10), ("mpc decision", 10),
        ("banco de inglaterra", 10),
        ("sterling", 8), ("pound sterling", 8), ("libra esterlina", 8),
        ("uk gilt", 8), ("gilt yield", 8), ("gilts", 8),
        ("uk economy", 5), ("united kingdom", 5), ("britain", 5),
        ("uk gdp", 5), ("uk inflation", 5), ("uk jobs", 5),
        ("uk cpi", 5), ("reino unido", 5),
        ("gbp/", 5), ("/gbp", 5),
        ("pound", 1), ("british", 1), ("ftse", 1), ("brexit", 1),
    ],
    "JPY": [
        ("boj", 10), ("bank of japan", 10), ("ueda", 10), ("himino", 10),
        ("kuroda", 10), ("banco de japón", 10),
        ("japanese yen", 8), ("yen japonés", 8),
        ("japan economy", 5), ("japanese economy", 5), ("economía japonesa", 5),
        ("japan gdp", 5), ("japan inflation", 5), ("japan cpi", 5),
        ("japan pmi", 5), ("japan trade", 5), ("japan unemployment", 5),
        ("jpy ", 5), ("jpy/", 5), ("/jpy", 5), ("usd/jpy", 5),
        ("yen ", 1), ("nikkei", 1), ("japanese", 1), ("japan ", 1),
        ("japón ", 1),
    ],
    "AUD": [
        ("rba", 10), ("reserve bank of australia", 10), ("bullock", 10),
        ("banco de la reserva de australia", 10),
        ("australian dollar", 8), ("dólar australiano", 8), ("aussie dollar", 8),
        ("australia gdp", 5), ("australian gdp", 5),
        ("australia inflation", 5), ("australia cpi", 5),
        ("australia trade", 5), ("australia retail", 5),
        ("australia jobs", 5), ("australian jobs", 5),
        ("australian economy", 5), ("australia economy", 5),
        ("aud/", 5), ("/aud", 5), ("aud ", 3), ("aussie ", 3),
        ("australia", 1),
    ],
    "CAD": [
        ("boc", 10), ("bank of canada", 10), ("macklem", 10),
        ("banco de canadá", 10),
        ("canadian dollar", 8), ("dólar canadiense", 8), ("loonie", 8),
        ("tsx", 8), ("s&p/tsx", 8),
        ("canada economy", 5), ("economía canadá", 5),
        ("canada gdp", 5), ("pib canadá", 5),
        ("canada inflation", 5), ("canada trade", 5), ("canada jobs", 5),
        ("canadian economy", 5),
        ("cad/", 5), ("/cad", 5), ("usd/cad", 5), ("cad ", 3),
        ("crude oil", 1), ("wti ", 1), ("brent", 1), ("petróleo", 1),
        ("oil prices", 1), ("opec", 1),
        ("canada ", 1), ("canadá ", 1),
    ],
    "CHF": [
        ("snb", 10), ("swiss national bank", 10), ("jordan", 10),
        ("schlegel", 10), ("banco nacional suizo", 10),
        ("swiss franc", 8), ("franco suizo", 8),
        ("switzerland", 5), ("swiss economy", 5), ("swiss inflation", 5),
        ("swiss cpi", 5), ("switzerland gdp", 5), ("swiss pmi", 5),
        ("swiss kof", 5), ("suiza", 5),
        ("chf/", 5), ("/chf", 5), ("usd/chf", 5), ("chf ", 3),
        ("swiss ", 1),
    ],
    "NZD": [
        ("rbnz", 10), ("reserve bank of new zealand", 10), ("orr", 10),
        ("banco de la reserva de nueva zelanda", 10),
        ("new zealand dollar", 8), ("dólar neozelandés", 8), ("kiwi dollar", 8),
        ("new zealand gdp", 5), ("nz gdp", 5), ("nz cpi", 5),
        ("nz economy", 5), ("nz pmi", 5), ("nz trade", 5), ("nz retail", 5),
        ("nz jobs", 5), ("new zealand inflation", 5), ("new zealand trade", 5),
        ("nueva zelanda", 5),
        ("nzd/", 5), ("/nzd", 5), ("nzd ", 3), ("kiwi ", 3),
        ("new zealand", 1),
    ],
}

# ─────────────────────────────────────────────
# FALSE POSITIVE GUARDS
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
    # v5.3: nuevo guard — UK housing penaliza divisas sin relación directa
    {
        "pattern": re.compile(r"\b(halifax|house prices?|housing market uk|uk property|uk housing|dwelling consents)\b", re.IGNORECASE),
        "penalize": {"NZD": 5, "AUD": 3, "CHF": 3, "CAD": 2},
    },
]

# ─────────────────────────────────────────────
# FEEDS
# ─────────────────────────────────────────────
FEEDS = [
    # ── ESPAÑOL ──────────────────────────────────────────────────────────────
    # v5.4: FXStreet ES eliminado — duplicaba contenido de FXStreet EN traducido
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/forexnews.xml",                "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/TechnicalAnalysis.xml",        "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/FundamentalAnalysis.xml",      "lang": "es" },
    { "source": "DailyForex ES",    "url": "https://es.dailyforex.com/rss/es/forexarticles.xml",            "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_1.rss",                        "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_25.rss",                       "lang": "es" },
    { "source": "Investing.com ES", "url": "https://es.investing.com/rss/news_14.rss",                       "lang": "es" },

    # ── INGLÉS — fuentes existentes ──────────────────────────────────────────
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/news",                              "lang": "en" },
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss/analysis",                          "lang": "en" },
    { "source": "FXStreet",         "url": "https://www.fxstreet.com/rss",                                    "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/news",                             "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/centralbank",                      "lang": "en" },
    { "source": "ForexLive",        "url": "https://www.forexlive.com/feed/analysis",                         "lang": "en" },
    { "source": "ECB",              "url": "https://www.ecb.europa.eu/rss/press.html",                        "lang": "en" },
    { "source": "Bank of England",  "url": "https://www.bankofengland.co.uk/rss/news",                        "lang": "en" },
    { "source": "DailyForex",       "url": "https://www.dailyforex.com/rss/forexnews.xml",                   "lang": "en" },
    { "source": "ActionForex",      "url": "https://www.actionforex.com/category/live-comments/feed/",        "lang": "en" },
    { "source": "ActionForex",      "url": "https://www.actionforex.com/category/action-insight/feed/",       "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/centralbank/",                     "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/technicalanalysis/",               "lang": "en" },
    { "source": "InvestingLive",    "url": "https://investinglive.com/feed/",                                  "lang": "en" },
    { "source": "MyFXBook",         "url": "https://www.myfxbook.com/rss/latest-forex-news",                  "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Technical.rss",               "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Fundamental.rss",             "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Opinion.rss",                 "lang": "en" },
    { "source": "Investing.com",    "url": "https://www.investing.com/rss/forex_Signals.rss",                 "lang": "en" },
    { "source": "InstaForex",       "url": "https://news.instaforex.com/news",                                "lang": "en" },
    { "source": "InstaForex",       "url": "https://news.instaforex.com/analytics",                           "lang": "en" },
    { "source": "BabyPips",         "url": "https://www.babypips.com/feed.rss",                               "lang": "en" },
    { "source": "InvestMacro",      "url": "https://investmacro.com/feed/",                                    "lang": "en" },
    { "source": "ForexCrunch",      "url": "https://forexcrunch.com/feed/",                                    "lang": "en" },

    # ── INGLÉS — bancos centrales oficiales (v5.2) ───────────────────────────
    { "source": "RBA",              "url": "https://www.rba.gov.au/rss/rss-cb-speeches.xml",                  "lang": "en" },
    { "source": "RBA",              "url": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",            "lang": "en" },
    { "source": "RBNZ",             "url": "https://www.rbnz.govt.nz/hub/news/feed",                          "lang": "en" },
    { "source": "SNB",              "url": "https://www.snb.ch/en/snb/medmit/medienmitteilungen/id/rss",       "lang": "en" },
    { "source": "Bank of Japan",    "url": "https://www.boj.or.jp/en/about/press/index.htm/rss.xml",           "lang": "en" },

    # ── INGLÉS — fuentes de análisis (v5.2) ─────────────────────────────────
    { "source": "MarketPulse",      "url": "https://www.marketpulse.com/feed/",                               "lang": "en" },
    { "source": "MarketPulse",      "url": "https://www.marketpulse.com/forex/feed/",                         "lang": "en" },
    { "source": "Reuters FX",       "url": "https://feeds.reuters.com/reuters/currenciesNews",                 "lang": "en" },
    { "source": "Nasdaq FX",        "url": "https://www.nasdaq.com/feed/rssoutbound?category=currencies",      "lang": "en" },
    { "source": "FX Empire",        "url": "https://www.fxempire.com/api/v1/en/articles/rss?category=news",   "lang": "en" },
    { "source": "FX Empire",        "url": "https://www.fxempire.com/api/v1/en/articles/rss?category=forecast", "lang": "en" },

    # ── GOOGLE NEWS vía NewsData.io API — 1 query por divisa (v5.7) ──────────
    # NewsData.io: plan gratuito: 200 créditos/día, sin delay.
    # NO van en FEEDS (usa JSON, no RSS). Se procesan por fetch_newsdata() en main().
    # Las queries están definidas en NEWSDATA_QUERIES más abajo.
]

# ─────────────────────────────────────────────
# NEWSDATA.IO API — configuración (v5.7)
# Reemplaza GNews — plan gratuito: 200 créditos/día, sin delay.
# 3 ejecuciones × 8 divisas × 1 crédito = 24 créditos/día (12% del límite).
# Registro: https://newsdata.io
# ─────────────────────────────────────────────
NEWSDATA_API_KEY_ENV  = "NEWSDATA_API_KEY"
NEWSDATA_MAX_RESULTS  = 5                    # por query (plan free: max 10)
NEWSDATA_BASE_URL     = "https://newsdata.io/api/1/news"

NEWSDATA_QUERIES = {
    "USD": "US dollar OR federal reserve OR FOMC",
    "EUR": "euro OR ECB OR european central bank",
    "GBP": "british pound OR sterling OR bank of england",
    "JPY": "japanese yen OR bank of japan OR BOJ",
    "AUD": "australian dollar OR RBA OR reserve bank australia",
    "CAD": "canadian dollar OR bank of canada",
    "CHF": "swiss franc OR SNB OR swiss national bank",
    "NZD": "new zealand dollar OR RBNZ",
}

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
    "strait of hormuz", "estrecho de ormuz", "war ", "guerra ",
    "military strike", "ataque militar", "sanctions", "sanciones",
    "statement on monetary policy", "minutes of", "board decision",
    "official cash rate", "cash rate target", "policy rate",
    "orr", "bullock", "schlegel", "ueda",
]

MED_IMPACT_KW = [
    "pmi", "employment", "jobless", "trade balance", "retail sales",
    "industrial production", "consumer confidence", "business confidence",
    "housing", "wages", "earnings", "exports", "imports", "deficit",
    "surplus", "forecast", "outlook", "guidance", "payroll", "manufacturing",
    "pmi manufacturero", "desempleo", "balanza comercial", "ventas minoristas",
    "producción industrial", "confianza del consumidor", "salarios",
    "crude oil", "oil prices", "petróleo", "brent", "wti",
    "business nz", "nz biz", "westpac nz", "anz nz",
    "building permits", "dwelling consents", "trade deficit nz",
    "australia business", "australia consumer", "anz australia",
    "westpac australia", "nab business", "swiss kof",
    "swiss manufacturing", "swiss retail",
]

SOURCE_CURRENCY = {
    "ECB":             "EUR",
    "Bank of England": "GBP",
    "Bank of Japan":   "JPY",
    "RBA":             "AUD",
    "RBNZ":            "NZD",
    "SNB":             "CHF",
    "Federal Reserve": "USD",
    # v5.7: NewsData — divisa ya asignada en fetch_newsdata(), esto es fallback defensivo
    "NewsData USD": "USD",
    "NewsData EUR": "EUR",
    "NewsData GBP": "GBP",
    "NewsData JPY": "JPY",
    "NewsData AUD": "AUD",
    "NewsData CAD": "CAD",
    "NewsData CHF": "CHF",
    "NewsData NZD": "NZD",
}

FOREX_SOURCES = {
    "FXStreet", "ForexLive", "DailyForex ES", "DailyForex",
    "ECB", "Bank of England", "Bank of Japan", "RBA", "RBNZ", "SNB",
    "Federal Reserve", "ActionForex", "InvestingLive", "MyFXBook",
    "Investing.com", "InstaForex", "BabyPips", "InvestMacro", "ForexCrunch",
    "Investing.com ES", "MarketPulse", "Reuters FX", "Nasdaq FX", "FX Empire",
    # v5.7: NewsData API
    "NewsData USD", "NewsData EUR", "NewsData GBP", "NewsData JPY",
    "NewsData AUD", "NewsData CAD", "NewsData CHF", "NewsData NZD",
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
    # v5.3: filtrar índices bursátiles europeos que no son relevantes para divisas
    if EQUITY_INDEX_TITLE_RE.search(title):
        return False

    if EMERGING_MARKET_TITLE_RE.search(title):
        return False

    if CRYPTO_NOISE_RE.search(title):
        combined_lower = (title + " " + summary).lower()
        if not any(kw in combined_lower for kw in CRYPTO_FOREX_BRIDGE_KW):
            return False

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
        "official cash rate", "cash rate", "policy rate", "rbnz", "rba",
        "reserve bank", "swiss national", "snb",
    ]
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_RELEVANCE_KW)


def detect_currency(title: str, summary: str, source: str = "") -> str | None:
    title_clean = (title or '').strip()
    title_upper = title_clean.upper().replace(' ', '')

    # PASO 1: par explícito en el título
    title_start_upper = title_clean.upper()[:35].replace(' ', '')
    for pair, protagonist in PAIR_PROTAGONIST_MAP.items():
        pair_noslash = pair.replace('/', '').upper()
        pair_slash   = pair.upper()
        if pair_noslash in title_start_upper or pair_slash in title_clean.upper()[:35]:
            return protagonist
    for pair, protagonist in PAIR_PROTAGONIST_MAP.items():
        pair_noslash = pair.replace('/', '').upper()
        if pair_noslash in title_upper or pair.upper() in title_clean.upper():
            return protagonist

    # PASO 2: scoring acumulativo
    text        = (title_clean + " " + (summary or '')).lower()
    title_lower = title_clean.lower()
    scores: dict[str, float] = {cur: 0.0 for cur in CURRENCIES}

    for cur, kws in CURRENCY_KEYWORDS_WEIGHTED.items():
        for kw, weight in kws:
            if kw in text:
                scores[cur] += weight
                if kw in title_lower:
                    scores[cur] += weight * 0.5

    for guard in FALSE_POSITIVE_GUARDS:
        if guard["pattern"].search(text):
            for cur, penalty in guard["penalize"].items():
                scores[cur] = max(0, scores[cur] - penalty)

    best_cur   = max(scores, key=lambda c: scores[c])
    best_score = scores[best_cur]

    if best_score >= CURRENCY_MIN_SCORE:
        return best_cur

    # PASO 3: fallback institucional
    if source in SOURCE_CURRENCY:
        return SOURCE_CURRENCY[source]

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
        "ForexCrunch":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Reuters FX":     "Mozilla/5.0 (compatible; RSSReader/1.0)",
        "Nasdaq FX":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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


def fetch_newsdata(api_key: str, now_utc: datetime) -> list:
    """
    v5.7: Consulta NewsData.io por cada divisa en NEWSDATA_QUERIES.
    Plan gratuito: 200 créditos/día, sin delay — artículos en tiempo real.
    Consume 8 créditos por ejecución (1 por divisa).
    """
    if not api_key:
        print("  [NewsData] NEWSDATA_API_KEY no configurada — omitiendo")
        return []

    cutoff = now_utc - timedelta(days=MAX_AGE_DAYS)
    articles = []
    seen_newsdata_titles = set()

    for cur, query in NEWSDATA_QUERIES.items():
        try:
            params = {
                "apikey":   api_key,
                "q":        query,
                "language": "en",
                "size":     NEWSDATA_MAX_RESULTS,
                "category": "business",
            }
            resp = requests.get(
                NEWSDATA_BASE_URL,
                params=params,
                timeout=FETCH_TIMEOUT,
            )

            if resp.status_code == 401:
                print(f"  [NewsData] API key inválida (401)")
                break
            if resp.status_code == 429:
                print(f"  [NewsData] Rate limit alcanzado (429) — parando consultas")
                break
            if resp.status_code != 200:
                print(f"  [NewsData] {cur}: error {resp.status_code} — {resp.text[:150]}")
                continue

            data = resp.json()
            if data.get("status") != "success":
                print(f"  [NewsData] {cur}: status={data.get('status')} — {str(data)[:150]}")
                continue

            results = data.get("results", [])
            count = 0

            for item in results:
                title   = clean_html(item.get("title", ""))
                summary = clean_html(
                    item.get("description", "") or
                    item.get("content", "") or ""
                )
                link = item.get("link", "")

                if not title or len(title) < 15:
                    continue

                # NewsData usa pubDate: "2026-03-11 12:30:00"
                pub_date = now_utc
                raw_date = item.get("pubDate", "")
                if raw_date:
                    try:
                        pub_date = dateparser.parse(raw_date)
                        if pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=timezone.utc)
                        pub_date = pub_date.astimezone(timezone.utc)
                    except Exception:
                        pass

                if pub_date < cutoff:
                    continue

                if is_calendar_entry(title, summary):
                    continue
                if not has_real_content(title, summary):
                    continue

                norm_title = normalize_title(title)
                title_key  = norm_title[:60]
                if title_key in seen_newsdata_titles:
                    continue
                seen_newsdata_titles.add(title_key)

                nid       = entry_id(title, f"NewsData {cur}")
                impact    = detect_impact(title, summary)
                expand    = summary[:350] + ("..." if len(summary) > 350 else "")
                age_hours = (now_utc - pub_date).total_seconds() / 3600

                articles.append({
                    "id":       nid,
                    "cur":      cur,
                    "impact":   impact,
                    "title":    title,
                    "expand":   expand,
                    "source":   f"NewsData {cur}",
                    "link":     link,
                    "time":     pub_date.strftime("%H:%M"),
                    "ts":       int(pub_date.timestamp() * 1000),
                    "featured": impact == "high" and age_hours <= 6,
                    "lang":     "en",
                    "date":     pub_date.strftime("%d %b"),
                    "datetime": pub_date.isoformat(),
                    "recent":   age_hours <= 24,
                })
                count += 1

            if count > 0:
                print(f"    [NewsData] {cur}: {count} artículos")
            else:
                print(f"  [NewsData] {cur}: 0 artículos (total API: {data.get('totalResults','?')})")

            time.sleep(2)  # evitar burst rate limit

        except Exception as e:
            print(f"  [NewsData] {cur}: excepción — {e}")

    return articles
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
    instaforex_count     = 0          # v5.4: contador para cap de InstaForex

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_news.py v5.7 — {len(FEEDS)} feeds")

    print(f"  Descargando en paralelo (workers={FETCH_WORKERS})...")
    all_entries = fetch_all_feeds(FEEDS)
    print(f"  Descarga completada.")

    # v5.7: NewsData API — fetch separado, resultados se añaden a raw_articles al final
    newsdata_api_key = os.environ.get(NEWSDATA_API_KEY_ENV, "").strip()
    newsdata_articles = []
    if newsdata_api_key:
        print(f"  Consultando NewsData.io API ({len(NEWSDATA_QUERIES)} divisas)...")
        newsdata_articles = fetch_newsdata(newsdata_api_key, now_utc)
        print(f"  NewsData: {len(newsdata_articles)} artículos obtenidos")
    else:
        print(f"  [NewsData] NEWSDATA_API_KEY no configurada — omitiendo")

    for feed_cfg in FEEDS:
        source           = feed_cfg["source"]
        lang             = feed_cfg.get("lang", "en")
        forced_currency  = feed_cfg.get("currency")   # v5.3: cortocircuito por divisa
        entries          = all_entries.get(feed_cfg["url"], [])
        count            = 0

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

            # Para feeds con divisa forzada, relajamos is_forex_relevant()
            # ya que el feed es por definición relevante para esa divisa.
            if not forced_currency and not is_forex_relevant(title, summary):
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

            # v5.3: cortocircuito — si el feed declara su divisa, úsala directamente
            if forced_currency:
                cur = forced_currency
            else:
                cur = detect_currency(title, summary, source)

            if cur is None:
                filtered_no_currency += 1
                continue

            impact    = detect_impact(title, summary)

            # v5.4: InstaForex — forzar impacto low y limitar a INSTAFOREX_MAX artículos
            if source == "InstaForex":
                impact = "low"
                if instaforex_count >= INSTAFOREX_MAX:
                    continue
                instaforex_count += 1

            expand    = summary[:350] + ("..." if len(summary) > 350 else "")
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
                "featured": impact == "high" and age_hours <= 6,
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
            tag = f" [{forced_currency}]" if forced_currency else ""
            print(f"    [{lang.upper()}] {source}{tag}: {count} noticias")

    # v5.7: añadir artículos de NewsData con deduplicación por título
    newsdata_added = 0
    for a in newsdata_articles:
        if a["id"] in seen_ids:
            continue
        title_key = normalize_title(a["title"])[:60]
        if title_key in seen_titles:
            continue
        seen_ids.add(a["id"])
        seen_titles.add(title_key)
        raw_articles.append(a)
        en_raw += 1
        newsdata_added += 1
    if newsdata_added:
        print(f"    [NewsData] {newsdata_added} artículos añadidos (tras deduplicación)")

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

    dist_after     = Counter(a["cur"] for a in articles)
    impact_after   = Counter(a["impact"] for a in articles)
    recent_count   = sum(1 for a in articles if a.get("recent", True))
    featured_count = sum(1 for a in articles if a.get("featured", False))
    print(f"\n✂️  Selección final ({len(articles)} artículos):")
    print(f"   Distribución: {dict(sorted(dist_after.items()))}")
    print(f"   Impacto: high={impact_after['high']} | med={impact_after['med']} | low={impact_after['low']}")
    print(f"   Recientes (<24h): {recent_count} | Históricos: {len(articles) - recent_count}")
    print(f"   Destacados (<6h + high): {featured_count}")

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
        # FIX R-02: Serializar en memoria, validar, y solo entonces escribir a disco.
        # Previene que un error de encoding o interrupción corrompa news.json en el repo.
        output_json = json.dumps(output, ensure_ascii=False, indent=2)
        json.loads(output_json)  # Valida que es parseable (lanza ValueError si no)
        f.write(output_json)

    new_count    = sum(1 for a in articles if not a.get("ai_headline"))
    reused_final = sum(1 for a in articles if a.get("ai_headline"))
    print(f"\n✓ {len(articles)} artículos guardados en {OUTPUT_FILE}")
    print(f"  ♻️  Con titular previo: {reused_final}")
    print(f"  🆕 Pendientes de Groq: {new_count}")
    print(f"  Fuentes activas: {', '.join(sources_ok)}")


if __name__ == "__main__":
    main()
