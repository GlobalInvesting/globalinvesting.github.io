#!/usr/bin/env python3
"""
fetch_news.py
Obtiene noticias forex desde múltiples fuentes RSS (ES + EN) y genera news.json.
Corre via GitHub Actions cada hora.

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
# Palabras clave ES + EN para cada divisa.
# Orden importa: se asigna la primera coincidencia.
# ─────────────────────────────────────────────
CURRENCY_KEYWORDS = {
    "USD": [
        # EN
        "fed ", "federal reserve", "fomc", "powell", "dollar", "usd", "us economy",
        "us gdp", "nonfarm", "non-farm", "jobless claims", "us inflation",
        "treasury", "debt ceiling", "ism ", "us jobs", "american economy",
        "wall street", "nasdaq", "dow jones", "s&p 500",
        # ES
        "reserva federal", "dólar", "dolar americano", "economía de eeuu",
        "economía estadounidense", "pib eeuu", "inflación eeuu", "inflación de estados unidos",
        "mercado laboral eeuu", "bonos del tesoro", "deuda eeuu",
    ],
    "EUR": [
        # EN
        "ecb", "european central bank", "lagarde", "euro ", "eur", "eurozone",
        "euro zone", "germany", "france", "italy", "spain", "draghi", "ifo",
        "zew", "pmi europe", "eu economy", "european economy", "bund",
        # ES
        "banco central europeo", "bce", "zona euro", "eurozona", "lagarde",
        "alemania", "economía europea", "pib zona euro", "inflación zona euro",
        "inflación de la zona euro", "ipc zona euro", "alemania ",
        "balanza comercial alemania", "confianza zew", "confianza ifo",
    ],
    "GBP": [
        # EN
        "boe", "bank of england", "bailey", "pound", "gbp", "sterling",
        "uk economy", "united kingdom", "britain", "brexit", "gilts",
        "uk gdp", "uk inflation", "uk jobs",
        # ES
        "banco de inglaterra", "libra esterlina", "libra ", "reino unido",
        "economía uk", "economía del reino unido", "pib uk", "inflación uk",
        "inflación del reino unido", "ipc uk", "pmi uk",
    ],
    "JPY": [
        # EN
        "boj", "bank of japan", "ueda", "yen", "jpy", "japan economy",
        "japanese", "nikkei", "shunto", "boj meeting", "japan gdp",
        # ES
        "banco de japón", "banco de japan", "yen japonés", "yen japones",
        "economía japonesa", "economía de japón", "pib japón", "inflación japón",
        "ipc japón", "producción industrial japón", "shunto",
    ],
    "AUD": [
        # EN
        "rba", "reserve bank of australia", "aussie", "aud", "australia",
        "australian economy", "australian jobs", "caixin",
        # ES
        "banco de la reserva de australia", "dólar australiano", "dolar australiano",
        "economía australia", "economía de australia", "pib australia",
        "empleo australia", "inflación australia",
    ],
    "CAD": [
        # EN
        "boc", "bank of canada", "macklem", "canadian dollar", "cad",
        "canada economy", "loonie", "oil prices", "crude oil", "wti",
        # ES
        "banco de canadá", "banco de canada", "dólar canadiense", "dolar canadiense",
        "economía canadá", "economía de canadá", "pib canadá", "inflación canadá",
        "ipc canadá", "petróleo", "precio del petróleo", "crudo ",
    ],
    "CHF": [
        # EN
        "snb", "swiss national bank", "jordan", "swiss franc", "chf",
        "switzerland", "swiss economy", "swiss inflation",
        # ES
        "banco nacional suizo", "franco suizo", "suiza", "economía suiza",
        "pib suiza", "inflación suiza", "ipc suiza",
    ],
    "NZD": [
        # EN
        "rbnz", "reserve bank of new zealand", "orr", "kiwi", "nzd",
        "new zealand", "nz economy", "nz jobs",
        # ES
        "banco de la reserva de nueva zelanda", "dólar neozelandés", "dolar neozelandes",
        "nueva zelanda", "economía de nueva zelanda", "pib nueva zelanda",
        "inflación nueva zelanda",
    ],
}

# ─────────────────────────────────────────────
# FEEDS RSS — ES primero, luego EN
# Fuentes marcadas según resultado del log 2026-03-01:
#   ✓ = devuelve artículos   ✗ = 0 resultados / bloqueado (eliminado)
# ─────────────────────────────────────────────
FEEDS = [

    # ══════════════════════════════════════════
    # FUENTES EN ESPAÑOL
    # ══════════════════════════════════════════

    # ✓ 30 noticias
    {
        "source": "FXStreet ES",
        "url": "https://www.fxstreet.es/rss/news",
        "method": "feedparser",
        "lang": "es",
    },
    # ✓ 21 + 26 noticias
    {
        "source": "Expansión",
        "url": "https://www.expansion.com/rss/mercados.xml",
        "method": "feedparser",
        "lang": "es",
    },
    {
        "source": "Expansión",
        "url": "https://www.expansion.com/rss/economia.xml",
        "method": "feedparser",
        "lang": "es",
    },
    # ✓ 1 noticia (feed real confirmado)
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/forexnews.xml",
        "method": "feedparser",
        "lang": "es",
    },
    # ✓ 11 noticias
    {
        "source": "DailyForex ES",
        "url": "https://es.dailyforex.com/rss/es/TechnicalAnalysis.xml",
        "method": "feedparser",
        "lang": "es",
    },
    # ✓ feed confirmado (0 hoy por falta de artículos recientes, no por bloqueo)
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
    # Investing.com ES — fetchar directamente sin proxy (allorigins da 500)
    # GitHub Actions tiene acceso directo a internet
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
    # ✗ ElEconomista — bloquea el User-Agent de Actions (0 noticias)
    # ✗ Forexduet ES  — 0 noticias
    # ✗ InfoMercados  — 0 noticias
    # ✗ Cinco Días    — 0 noticias

    # ══════════════════════════════════════════
    # FUENTES EN INGLÉS
    # ══════════════════════════════════════════

    # ✓ 30 + 13 noticias
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
    # ✓ 25 + 14 noticias
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
    # ✓ 5 noticias (comunicados oficiales BCE)
    {
        "source": "ECB",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "method": "feedparser",
        "lang": "en",
    },
    # ✓ 3 noticias
    {
        "source": "Bank of England",
        "url": "https://www.bankofengland.co.uk/rss/news",
        "method": "feedparser",
        "lang": "en",
    },
    # ✓ 7 noticias
    {
        "source": "Bank of Canada",
        "url": "https://www.bankofcanada.ca/feed/",
        "method": "feedparser",
        "lang": "en",
    },
    # ✓ 1 noticia
    {
        "source": "DailyForex",
        "url": "https://www.dailyforex.com/rss/forexnews.xml",
        "method": "feedparser",
        "lang": "en",
    },
    # ✗ Reuters       — bloquea Actions (0 noticias)
    # ✗ MQL5          — 0 noticias
    # ✗ Federal Reserve — 0 noticias
    # ✗ Bank of Japan — 0 noticias
    # ✗ RBA           — 0 noticias
    # ✗ SNB           — 0 noticias
    # ✗ RBNZ          — 0 noticias
    # ✗ DailyForex fundamentalanalysis.xml — 0 noticias
]

# ─────────────────────────────────────────────
# PALABRAS CLAVE DE IMPACTO — ES + EN
# ─────────────────────────────────────────────
HIGH_IMPACT_KW = [
    # EN
    "rate decision", "interest rate", "hike", "cut rates", "fomc", "ecb meeting",
    "boe meeting", "boj meeting", "nonfarm", "non-farm", "cpi", "inflation report",
    "gdp", "recession", "emergency", "crisis", "default", "shock",
    "surprise", "unexpected", "powell", "lagarde", "ueda", "bailey",
    "central bank", "rate hike", "rate cut", "quantitative easing",
    "quantitative tightening", "monetary policy",
    # ES
    "decisión de tasas", "tasa de interés", "subida de tipos", "bajada de tipos",
    "alza de tasas", "recorte de tasas", "sube tasas", "baja tasas",
    "inflación", "ipc ", "pib ", "recesión", "crisis ", "sorprende",
    "inesperado", "inesperada", "política monetaria", "banco central",
    "reunión del bce", "reunión de la fed", "reunión del boj",
    "reunión del boe", "hawkish", "dovish", "restrictivo", "acomodaticio",
    "reunión de política", "decisión de política",
]

MED_IMPACT_KW = [
    # EN
    "pmi", "employment", "jobless", "trade balance", "retail sales",
    "industrial production", "consumer confidence", "business confidence",
    "housing", "wages", "earnings", "exports", "imports", "deficit",
    "surplus", "forecast", "outlook", "guidance", "payroll",
    "manufacturing", "services sector",
    # ES
    "pmi manufacturero", "pmi de servicios", "desempleo", "tasa de paro",
    "balanza comercial", "ventas minoristas", "producción industrial",
    "confianza del consumidor", "confianza empresarial", "confianza del inversor",
    "salarios", "beneficios", "exportaciones", "importaciones", "déficit",
    "superávit", "previsión", "perspectivas", "orientación", "nóminas",
    "manufactura", "sector servicios", "producción", "cuenta corriente",
    "posicionamiento", "cot ", "balanza de pagos",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def detect_currency(title: str, summary: str) -> str:
    """Detecta la divisa principal (devuelve código 3 letras o 'USD' por defecto)."""
    text = (title + " " + summary).lower()
    for currency, keywords in CURRENCY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return currency
    return "USD"  # Fallback: la mayoría de noticias genéricas de mercados son USD


def detect_impact(title: str, summary: str) -> str:
    """
    Estima nivel de impacto. Devuelve 'high', 'med' o 'low' (minúsculas)
    para coincidir con lo que espera news.html.
    """
    text = (title + " " + summary).lower()
    for kw in HIGH_IMPACT_KW:
        if kw in text:
            return "high"
    for kw in MED_IMPACT_KW:
        if kw in text:
            return "med"
    return "low"


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
    """Elimina tags HTML y recorta espacios."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def entry_id(title: str, source: str) -> str:
    """Genera un ID único por noticia para deduplicar."""
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def fetch_via_feedparser(feed_cfg: dict) -> list:
    """Fetcha un feed con feedparser estándar."""
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
    """Fetcha un feed a través de un proxy CORS y lo parsea con feedparser."""
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
    articles = []  # Lista final — nombre alineado con news.html ("articles")

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
                or getattr(entry, "content", [{}])[0].get("value", "") if hasattr(entry, "content") else ""
            )
            link = getattr(entry, "link", "") or getattr(entry, "id", "")

            # Filtrar títulos vacíos o muy cortos (evita entradas de calendario sin texto)
            if not title or len(title) < 15:
                continue

            pub_date = parse_date(entry)
            if pub_date < cutoff:
                continue

            nid = entry_id(title, source)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            cur    = detect_currency(title, summary)
            impact = detect_impact(title, summary)   # 'high' | 'med' | 'low'

            # Recortar summary a 300 chars
            expand = summary[:300] + ("..." if len(summary) > 300 else "")

            # ── Formato alineado con news.html ───────────────────────────────
            # news.html espera:
            #   cur, impact (lower), title, expand, source, link, time, ts, featured
            articles.append({
                "id":       nid,
                "cur":      cur,                                # 'USD', 'EUR', etc.
                "impact":   impact,                             # 'high' | 'med' | 'low'
                "title":    title,
                "expand":   expand,                             # descripción corta
                "source":   source,
                "link":     link,
                "time":     pub_date.strftime("%H:%M"),         # "14:30" (UTC)
                "ts":       int(pub_date.timestamp() * 1000),   # epoch ms para ordenar
                "featured": impact == "high",                   # azul destacado en UI
                "lang":     lang,                               # 'es' | 'en'
                "date":     pub_date.strftime("%d %b"),         # "12 Mar"
                "datetime": pub_date.isoformat(),               # ISO completo
            })
            count += 1
            if lang == "es":
                es_count += 1
            else:
                en_count += 1

        print(f"    → {count} noticias válidas")

    # Ordenar por timestamp descendente (más reciente primero)
    articles.sort(key=lambda x: x["ts"], reverse=True)

    # Limitar al máximo configurado
    articles = articles[:MAX_NEWS]

    # ── Estadísticas ─────────────────────────────────────────────────────────
    total_high = sum(1 for a in articles if a["impact"] == "high")
    total_med  = sum(1 for a in articles if a["impact"] == "med")
    sources_ok = sorted(set(a["source"] for a in articles))

    output = {
        # ── Metadata ─────────────────────────────────────────────────────────
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

        # ── Artículos (clave que espera news.html) ────────────────────────────
        "articles": articles,
    }

    # Crear directorio si no existe
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
