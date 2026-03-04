#!/usr/bin/env python3
"""
enrich_news.py — v8
Soporte para múltiples API keys de Groq con fallback automático.

CAMBIOS v8 (sobre v7):
  - SYSTEM_PROMPT: añadidas reglas críticas para JPY y tasas:
      • "Posponer/retrasar/cancelar una subida de tasas = BAJISTA para la divisa"
      • "Política acomodaticia / tasas reales negativas = BAJISTA (dovish)"
      • Excepción JPY documentada: risk-off geopolítico puede ser ALCISTA para JPY
        aunque el BOJ sea dovish (activo refugio)
      • Regla SNB intervención: si SNB amenaza con intervenir para frenar apreciación
        → CHF neutral (no alcista), porque la propia intervención limita el alza
  - score_from_headline(): corregida inversión dovish/hawkish para JPY en contexto
    de tasas (retraso de alza = bajista)
  - Sin otros cambios funcionales respecto a v7
"""

import os
import sys
import json
import time
import re
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
NEWS_FILE             = Path("news-data/news.json")
GROQ_URL              = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL            = "llama-3.3-70b-versatile"
MAX_TOKENS            = 200
TEMPERATURE           = 0.15

SLEEP_BETWEEN         = 4
MAX_RETRIES           = 2
DEFAULT_RETRY_WAIT    = 65
SKIP_IF_WAIT_EXCEEDS  = 90
GLOBAL_TIMEOUT_MIN    = 14
MIN_CONTENT_WORDS     = 15

_START_TIME = time.time()


def elapsed_min():
    return (time.time() - _START_TIME) / 60.0


def timeout_reached():
    return elapsed_min() >= GLOBAL_TIMEOUT_MIN


# ─────────────────────────────────────────────
CURRENCY_NAMES = {
    "USD": "dólar estadounidense", "EUR": "euro",
    "GBP": "libra esterlina",      "JPY": "yen japonés",
    "AUD": "dólar australiano",    "CAD": "dólar canadiense",
    "CHF": "franco suizo",         "NZD": "dólar neozelandés",
}

CALENDAR_RE = re.compile(
    r"upcoming event|content type|scheduled date|share this page|"
    r"governing council presents|press release explaining|"
    r"announces the setting for the overnight|on eight scheduled|"
    r"monetary policy report$|rate announcement$",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT — ESTÁNDAR INSTITUCIONAL v8
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista senior de mercados de divisas (FX) de una mesa institucional de trading.
Tu tarea es sintetizar noticias económicas y financieras en titulares estructurados para una plataforma profesional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO OBLIGATORIO DEL TITULAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Evento/dato concreto]: [Mecanismo de transmisión] → [DIVISA] [SENTIMIENTO]

Donde SENTIMIENTO es exactamente una de estas cuatro palabras:
  alcista | bajista | neutral | mixto

Longitud: entre 90 y 160 caracteres.
Idioma: español profesional.
Sin punto final. Sin comillas. Sin markdown. Sin emojis.

EJEMPLOS CORRECTOS:
  IPC EEUU sube 3.2% en enero superando estimaciones: presión inflacionaria mantiene sesgo hawkish → USD alcista
  BoE recorta tasas 25pb a 4.5%: política más acomodaticia de lo esperado → GBP bajista
  PMI manufacturero zona euro cae a 44.2 puntos: contracción industrial profundiza presión sobre el BCE → EUR bajista
  Petróleo WTI sube 8% por bloqueo del Estrecho de Ormuz: ingresos por exportación de crudo en riesgo → CAD alcista
  BOJ pospone alza de tasas de marzo por volatilidad de mercados: decisión dovish retrasa normalización → JPY bajista
  SNB podría intervenir para frenar apreciación excesiva del franco: techo implícito sobre valorización → CHF neutral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLAS ABSOLUTAS — NUNCA VIOLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. USA SOLO información presente EXPLÍCITAMENTE en el título o descripción.
   Nunca inventes cifras, fechas, porcentajes ni instituciones que no aparezcan en el texto.

2. Si el artículo habla de un activo que NO es la divisa asignada (ej. bolsa turca, plata,
   ibovespa, rand sudafricano), responde únicamente: IRRELEVANTE

3. Nunca confundas el activo cubierto con la divisa afectada:
   "TSX cae" → el artículo es sobre CAD, no USD
   "Petróleo sube" + CAD asignado → petróleo es exportación clave de Canadá → CAD alcista
   "Petróleo sube" + EUR asignado → Europa importa energía → EUR bajista

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLAS CRÍTICAS DE TASAS DE INTERÉS — APLICA SIEMPRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HAWKISH (alcista para la divisa):
  - Subida de tasas confirmada o anticipada
  - Banco central mantiene sesgo restrictivo
  - Inflación por encima del objetivo que fuerza inacción o subida

DOVISH (bajista para la divisa):
  - Recorte de tasas confirmado o anticipado
  - POSPONER, RETRASAR o CANCELAR una subida de tasas previamente esperada
    → Ejemplo: "BOJ podría posponer alza de marzo" → JPY BAJISTA (no alcista)
    → El retraso de una subida reduce las expectativas de rendimiento → bajista
  - Banco central mantiene tasas sin cambios cuando el mercado esperaba subida
  - Política acomodaticia / tasas reales negativas / condiciones expansivas

EXCEPCIÓN JPY — ACTIVO REFUGIO:
  En contextos de RIESGO GEOPOLÍTICO EXTREMO (guerra, crisis sistémica):
  - Si el artículo trata PRINCIPALMENTE de la demanda de JPY como refugio seguro → JPY alcista
  - Si el artículo trata PRINCIPALMENTE de la política del BOJ (dovish/hawkish) → usa la regla normal
  - Si ambos factores están presentes y se contrarrestan → JPY mixto

REGLA SNB:
  - Si el SNB AMENAZA con intervenir para FRENAR la apreciación del franco → CHF neutral
    (la intervención es un techo implícito que limita la subida)
  - Si el franco se aprecia sin mención de intervención → CHF alcista

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRELACIONES MACROECONÓMICAS CLAVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PETRÓLEO / ENERGÍA:
  ↑ precio petróleo → CAD alcista (Canadá = mayor exportador de crudo a EEUU)
  ↑ precio petróleo → EUR bajista, JPY bajista (ambos son importadores netos)
  ↑ precio energía → NOK alcista, RUB alcista (si son mencionados)

ACTIVOS REFUGIO (risk-off):
  ↑ tensión geopolítica / ↑ volatilidad → JPY alcista, CHF alcista (si no hay intervención SNB)
  ↑ tensión geopolítica → AUD bajista, NZD bajista (divisas de riesgo)

CRECIMIENTO / DATOS MACRO:
  PIB / empleo / PMI por encima de expectativas → alcista
  PIB / empleo / PMI por debajo de expectativas → bajista
  Inflación por encima de meta + banco central reactivo → alcista (hawkish)
  Inflación por encima de meta + banco central permisivo → bajista (erosión)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CASOS ESPECIALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Análisis técnico con niveles: incluye el nivel clave, soporte/resistencia y sesgo
- Pares cruzados (EUR/GBP, GBP/JPY): describe el impacto en la divisa base
- Si el impacto es genuinamente ambiguo según el propio artículo: usa "mixto"
- Si no hay suficiente información analítica: responde solo: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur  = article.get("cur", "USD")
    lang = article.get("lang", "en")
    return (
        f"DIVISA ASIGNADA: {cur} — {CURRENCY_NAMES.get(cur, cur)}\n"
        f"FUENTE: {article.get('source', '')} ({'español' if lang == 'es' else 'inglés'})\n"
        f"TÍTULO ORIGINAL: {article.get('title', '')}\n"
        f"DESCRIPCIÓN: {(article.get('expand', '') or '')[:500] or 'Sin descripción'}\n\n"
        f"INSTRUCCIÓN: Sintetiza en un titular SOLO con la información del texto anterior.\n"
        f"El titular debe describir el impacto sobre {cur} ({CURRENCY_NAMES.get(cur, cur)}).\n"
        f"Si el artículo NO trata sobre {cur} ni tiene impacto demostrable en {cur}, "
        f"responde solo: IRRELEVANTE\n\n"
        f"Titular:"
    )


# ─────────────────────────────────────────────
# SCORING DE SENTIMIENTO — v8
# ─────────────────────────────────────────────
SENTIMENT_FINAL_RE = re.compile(
    r"→\s*(?:\w+\s+)?(alcista|bajista|neutral|mixto)\s*$",
    re.IGNORECASE,
)

BULL_KW = [
    "alcista", "hawkish", "sube", "subida", "alza", "fortalece", "soporte",
    "positivo", "supera expectativas", "por encima", "sorpresa positiva",
    "refugio", "apreciación",
]
BEAR_KW = [
    "bajista", "dovish", "cae", "caída", "baja", "debilita", "presión",
    "negativo", "por debajo", "sorpresa negativa", "deprecia", "deterioro",
    "recorte de tasas", "recesión",
    # Patrones que indican postergación de subida (DOVISH) — v8
    "posponer", "pospone", "retrasar", "retrasa", "postergar", "postpone",
    "put off", "hold off", "delay", "pause", "pausar",
    "acomodaticia", "acomodativo", "expansiva", "tasas reales negativas",
    "sin cambios", "mantiene tasas",
]
MIXED_KW   = ["mixto", "contradictorio", "ambiguo", "señal contradictoria"]
NEUTRAL_KW = ["neutral", "neutro", "sin impacto"]

# Patrones de postergación de subida (dovish para cualquier divisa) — v8
POSTPONE_HIKE_RE = re.compile(
    r"\b(posponer?|retrasar?|postergar?|put off|hold off|delay|pause)\b.{0,60}"
    r"\b(alza|hike|subida|rate rise|suba de tasas|raise rates)\b",
    re.IGNORECASE,
)

# Patrones SNB intervención → neutral (no alcista)
SNB_INTERVENTION_RE = re.compile(
    r"\b(snb|swiss national bank|banco nacional suizo)\b.{0,80}"
    r"\b(intervenir|intervene|interven|frenar|dampen|curb|limit)\b",
    re.IGNORECASE,
)


def score_from_headline(headline: str | None) -> str:
    """
    Extrae el sentimiento del ai_headline.

    Prioridad:
      1. Patrón final estructurado "→ [divisa] alcista/bajista/neutral/mixto"
      2. Detección de postergación de alza (DOVISH) — v8
      3. Detección de intervención SNB → neutral — v8
      4. Conteo de keywords
    """
    if not headline:
        return "neut"

    hl = headline.lower()

    # Prioridad 1 — patrón estructurado al final
    m = SENTIMENT_FINAL_RE.search(hl)
    if m:
        word = m.group(1).lower()
        if word == "alcista":  return "bull"
        if word == "bajista":  return "bear"
        if word == "mixto":    return "mixed"
        if word in ("neutral", "neutro"): return "neut"

    # Prioridad 2 — postergación de subida = dovish = bajista (v8)
    if POSTPONE_HIKE_RE.search(hl):
        return "bear"

    # Prioridad 3 — SNB intervención = neutral (v8)
    if SNB_INTERVENTION_RE.search(hl):
        return "neut"

    # Prioridad 4 — fallback keywords
    bull_score = sum(1 for kw in BULL_KW  if kw in hl)
    bear_score = sum(1 for kw in BEAR_KW  if kw in hl)
    mixed_flag = any(kw in hl for kw in MIXED_KW)
    neut_flag  = any(kw in hl for kw in NEUTRAL_KW)

    if mixed_flag:
        return "mixed"
    if neut_flag and bull_score == 0 and bear_score == 0:
        return "neut"
    if bull_score > bear_score:
        return "bull"
    if bear_score > bull_score:
        return "bear"
    if bull_score == bear_score and bull_score > 0:
        return "mixed"
    return "neut"


# ─────────────────────────────────────────────
def is_enrichable(article: dict) -> tuple[bool, str]:
    title  = (article.get("title", "") or "").strip()
    expand = (article.get("expand", "") or "").strip()

    word_count = len(expand.split())
    if word_count < MIN_CONTENT_WORDS:
        return False, f"descripción muy corta ({word_count} palabras)"

    combined = (title + " " + expand).lower()
    if CALENDAR_RE.search(combined):
        return False, "contenido de calendario/agenda"

    title_lower  = title.lower()[:50]
    expand_lower = expand.lower()
    if expand_lower.startswith(title_lower) and word_count < 25:
        return False, "descripción idéntica al título"

    meta_patterns = [
        r"^\s*(read more|continue reading|click here|leer más|ver más)\b",
        r"^\s*[\[\(].*[\]\)]\s*$",
        r"^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*$",
    ]
    for pat in meta_patterns:
        if re.match(pat, expand, re.IGNORECASE):
            return False, "descripción es solo metadatos"

    EM_NOISE_RE = re.compile(
        r"\b(ibovespa|bovespa|bist 100|turkish stocks?|istanbul stock|"
        r"south african rand weakens|rand (falls?|drops?|weakens?)|"
        r"sugar futures?|cocoa futures?|coffee futures?|"
        r"silver (price|slammed|falls?)|copper (price|falls?))\b",
        re.IGNORECASE,
    )
    cur = article.get("cur", "")
    if EM_NOISE_RE.search(title):
        return False, f"artículo de mercado emergente/commodity sin nexo con {cur}"

    return True, ""


# ─────────────────────────────────────────────
def load_api_keys() -> list:
    keys = []
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = os.environ.get(var, "").strip()
        if k:
            keys.append(k)
    return keys


def mask_key(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"


def is_daily_limit_message(response) -> bool:
    try:
        body = response.json()
        msg  = str(body.get("error", {}).get("message", "")).lower()
        is_daily      = any(x in msg for x in ("per day", "tpd", "tokens per day"))
        is_per_minute = "per minute" in msg or "rpm" in msg or "tpm" in msg
        return is_daily and not is_per_minute
    except Exception:
        return False


def check_key(api_key: str) -> str:
    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model":    GROQ_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        if r.status_code == 401:
            return "invalid"
        if r.status_code == 429:
            return "daily_limit" if is_daily_limit_message(r) else "ok"
        return "ok"
    except Exception:
        return "ok"


def get_retry_after(response) -> int:
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        val = response.headers.get(header)
        if val:
            try:
                return min(int(float(str(val).lower().replace("s", "").strip())) + 2, 180)
            except (ValueError, TypeError):
                pass
    return DEFAULT_RETRY_WAIT


def call_groq(api_key: str, article: dict) -> str | None:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(article)},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)

            if r.status_code == 429:
                if is_daily_limit_message(r):
                    return "DAILY_LIMIT"
                wait = get_retry_after(r)
                if wait > SKIP_IF_WAIT_EXCEEDS:
                    print(f"  ⏭️  Retry-After={wait}s muy largo — omitiendo artículo")
                    return None
                print(f"  ⏳ Rate limit (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                return "DAILY_LIMIT"

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            if content.upper() == "IRRELEVANTE":
                return None

            content = content.strip('"\'').strip()
            for prefix in ("titular:", "headline:", "síntesis:", "análisis:", "titolo:"):
                if content.lower().startswith(prefix):
                    content = content[len(prefix):].strip()

            return content if len(content) >= 25 else None

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout (intento {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(5)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            return None

    return None


def save_progress(data, articles, now_utc):
    data["articles"]       = articles
    data["ai_enriched_at"] = now_utc.isoformat()
    data["ai_model"]       = GROQ_MODEL
    with open(NEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 65)
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}  |  v8 institucional")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}  |  Timeout: {GLOBAL_TIMEOUT_MIN} min")
    print("=" * 65)

    all_keys = load_api_keys()
    if not all_keys:
        raise EnvironmentError("❌ No se encontró ninguna GROQ_API_KEY.")

    print(f"\n🔑 Keys configuradas: {len(all_keys)}")
    available_keys = []
    for i, key in enumerate(all_keys, 1):
        status = check_key(key)
        label  = f"Key {i} ({mask_key(key)})"
        if status == "daily_limit":
            print(f"  ⛔ {label} — límite diario confirmado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            print(f"  ✅ {label} — disponible")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys agotadas. Saliendo.")
        sys.exit(0)

    print(f"\n✅ {len(available_keys)} key(s) disponible(s)\n")
    current_key_idx = 0

    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró {NEWS_FILE}.")

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        print("⚠️  Sin artículos.")
        return

    print(f"📰 Total artículos: {len(articles)}")
    dist   = Counter(a.get("cur", "?") for a in articles)
    impact = Counter(a.get("impact", "?") for a in articles)
    print(f"   Divisas: {dict(sorted(dist.items()))}")
    print(f"   Impacto: high={impact['high']} | med={impact['med']} | low={impact['low']}")

    # Recalcular sentimientos previos con la nueva lógica v8
    sentiments_recalculated = 0
    for article in articles:
        if article.get("ai_headline"):
            new_sentiment = score_from_headline(article["ai_headline"])
            if article.get("sentiment") != new_sentiment:
                article["sentiment"] = new_sentiment
                sentiments_recalculated += 1
    if sentiments_recalculated:
        print(f"   🔄 Sentimientos recalculados (lógica v8): {sentiments_recalculated}")

    to_process = [(i, a) for i, a in enumerate(articles) if not a.get("ai_headline")]
    print(f"\n🔍 A enriquecer: {len(to_process)}")

    if not to_process:
        print("✅ Todos tienen ai_headline.")
        save_progress(data, articles, now_utc)
        return

    est_time = len(to_process) * SLEEP_BETWEEN
    print(f"⏱️  Estimado: ~{est_time // 60}min {est_time % 60}s  |  Timeout: {GLOBAL_TIMEOUT_MIN}min\n")

    enriched = skipped = irrelevant = not_enrichable = 0
    stopped_early = False

    for idx, (article_idx, article) in enumerate(to_process):

        if timeout_reached():
            print(f"\n⏰ Timeout {GLOBAL_TIMEOUT_MIN} min — guardando parcial")
            stopped_early = True
            break

        cur = article.get("cur", "?")
        imp = article.get("impact", "?")
        print(f"[{idx+1}/{len(to_process)}] {cur} [{imp}] {article.get('title','')[:65]}...")

        ok, reason = is_enrichable(article)
        if not ok:
            not_enrichable += 1
            article["sentiment"] = "neut"
            article["ai_headline"] = None
            print(f"  ⛔ No enriquecible: {reason}")
            continue

        result = call_groq(available_keys[current_key_idx], article)

        if result == "DAILY_LIMIT":
            print(f"  ⛔ Key {current_key_idx + 1} agotada — buscando siguiente...")
            current_key_idx += 1
            if current_key_idx >= len(available_keys):
                print("  ⛔ Todas las keys agotadas — guardando progreso")
                stopped_early = True
                break
            print(f"  🔄 Cambiando a Key {current_key_idx + 1} ({mask_key(available_keys[current_key_idx])})")
            time.sleep(3)
            result = call_groq(available_keys[current_key_idx], article)
            if result == "DAILY_LIMIT":
                print("  ⛔ Key siguiente también agotada — deteniendo")
                stopped_early = True
                break

        if result and result != "DAILY_LIMIT":
            articles[article_idx]["ai_headline"] = result
            articles[article_idx]["sentiment"]   = score_from_headline(result)
            enriched += 1
            sentiment_label = articles[article_idx]["sentiment"]
            print(f"  ✅ [{sentiment_label}] {result[:90]}{'...' if len(result) > 90 else ''}")
        elif result is None:
            irrelevant += 1
            articles[article_idx]["sentiment"] = "neut"
            print("  ⏭️  Sin síntesis (irrelevante/sin info suficiente)")
        else:
            skipped += 1
            articles[article_idx]["sentiment"] = "neut"
            print("  ⚠️  Sin síntesis (error de API)")

        if idx < len(to_process) - 1 and not timeout_reached():
            time.sleep(SLEEP_BETWEEN)

    # Asegurar campo sentiment en todos los artículos
    for article in articles:
        if "sentiment" not in article:
            article["sentiment"] = score_from_headline(article.get("ai_headline"))

    save_progress(data, articles, now_utc)

    # Estadísticas finales
    sent_by_cur: dict[str, Counter] = {}
    for a in articles:
        cur = a.get("cur", "?")
        s   = a.get("sentiment", "neut")
        if cur not in sent_by_cur:
            sent_by_cur[cur] = Counter()
        sent_by_cur[cur][s] += 1

    print()
    print("=" * 65)
    print("📊 SENTIMIENTO POR DIVISA (scoring ponderado aplicado en frontend)")
    print("-" * 65)
    for cur in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        c = sent_by_cur.get(cur, Counter())
        total = sum(c.values())
        if total == 0:
            continue
        bull  = c.get("bull",  0)
        bear  = c.get("bear",  0)
        neut  = c.get("neut",  0)
        mixed = c.get("mixed", 0)
        dominant = max(c, key=c.get) if c else "neut"
        dominant_label = {"bull": "ALCISTA", "bear": "BAJISTA", "neut": "NEUTRAL", "mixed": "MIXTO"}.get(dominant, "?")
        print(f"  {cur:4s}  bull={bull} bear={bear} neut={neut} mixed={mixed}  → {dominant_label}")

    print()
    print("📋 RESUMEN ENRIQUECIMIENTO")
    print(f"   ✅ Enriquecidos:        {enriched}")
    print(f"   ⛔ No enriquecibles:    {not_enrichable}")
    print(f"   ⚠️  Error API:          {skipped}")
    print(f"   ⏭️  Irrelevantes:       {irrelevant}")
    if stopped_early:
        print(f"   ⏰ Detenido antes de completar")
    print(f"   🔑 Keys usadas:        hasta Key {current_key_idx + 1} de {len(available_keys)}")
    print(f"   ⏱️  Tiempo total:       {elapsed_min():.1f} min")
    print(f"   💾 Guardado en:        {NEWS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
