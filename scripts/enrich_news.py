#!/usr/bin/env python3
"""
enrich_news.py — v9.2
Soporte para múltiples API keys de Groq con fallback automático.

CAMBIOS v9.2 (sobre v9.1):
  MEJORA 1 — CURRENCY_MACRO_CONTEXT inyectado en build_user_prompt():
    El perfil macro de cada divisa (mecanismo de transmisión específico)
    se incluye en el user prompt para que el modelo escriba el mecanismo
    correcto en el cuerpo del titular, no solo en el sentimiento final.
    Reduce titulares genéricos como "genera incertidumbre → EUR bajista"
    a favor de "eleva costes energéticos zona euro → EUR bajista".

  MEJORA 2 — Correlación NZD/petróleo más precisa en SYSTEM_PROMPT:
    Antes: "NZD bajista" de forma simple cuando sube el petróleo.
    Ahora: NZD bajista por ser divisa de riesgo en entornos risk-off
    (el mecanismo es el risk-off, no la correlación directa con el crudo).
    Diferencia importante: petróleo sube por demanda global → NZD puede
    ser neutral o alcista; petróleo sube por crisis/guerra → NZD bajista.

  Sin otros cambios funcionales respecto a v9.1.
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

# v9.2: Perfil macro de cada divisa — inyectado en el user prompt
# para anclar el mecanismo de transmisión en el cuerpo del titular
CURRENCY_MACRO_CONTEXT = {
    "USD": "Activo refugio global. Se beneficia de risk-off y tensiones geopolíticas. "
           "Sensible a postura Fed y diferencial de tasas con G10.",
    "EUR": "Importador neto de energía. Conflicto geopolítico = mayores costes energéticos "
           "= presión sobre crecimiento eurozona = dilema BCE entre inflación y recesión.",
    "GBP": "No es activo refugio. Sensible a inflación UK, política del BoE y datos laborales. "
           "En risk-off cae frente a USD, JPY y CHF.",
    "JPY": "Activo refugio tradicional pero debilitado por importación de petróleo. "
           "Driver dominante: diferencial tasas US-JP. Fed hawkish o BoJ dovish = JPY bajista.",
    "AUD": "Divisa de riesgo correlacionada con commodities (hierro, cobre) y ciclo chino. "
           "En crisis/guerra cae por risk-off aunque Australia exporta algunos commodities.",
    "CAD": "Correlacionado con petróleo WTI: Canadá es exportador neto de crudo. "
           "Petróleo alto = soporte estructural CAD independientemente de postura BoC.",
    "CHF": "Activo refugio por excelencia. Se aprecia en crisis/guerra/risk-off global. "
           "SNB puede intervenir para limitar apreciación excesiva.",
    "NZD": "Divisa de riesgo de alta beta. En crisis/guerra cae por risk-off global, "
           "no por correlación directa con petróleo. RBNZ y datos domésticos NZ son drivers propios.",
}

CALENDAR_RE = re.compile(
    r"upcoming event|content type|scheduled date|share this page|"
    r"governing council presents|press release explaining|"
    r"announces the setting for the overnight|on eight scheduled|"
    r"monetary policy report$|rate announcement$",
    re.IGNORECASE,
)

MAJORS_VS_USD = {"EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"}

# ─────────────────────────────────────────────
# SYSTEM PROMPT — v9.2
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista senior de mercados de divisas (FX) de una mesa institucional de trading.
Tu tarea es sintetizar noticias económicas y financieras en titulares estructurados para una plataforma profesional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO OBLIGATORIO DEL TITULAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Evento/dato concreto]: [Mecanismo de transmisión ESPECÍFICO para esta divisa] → [DIVISA] [SENTIMIENTO]

Donde:
  - DIVISA ASIGNADA es SIEMPRE la divisa indicada en "DIVISA ASIGNADA:"
  - SENTIMIENTO es exactamente una de estas cuatro palabras: alcista | bajista | neutral | mixto

Longitud: entre 90 y 160 caracteres.
Idioma: español profesional.
Sin punto final. Sin comillas. Sin markdown. Sin emojis.

EJEMPLOS CORRECTOS:
  IPC EEUU sube 3.2% en enero superando estimaciones: presión inflacionaria mantiene sesgo hawkish → USD alcista
  BoE recorta tasas 25pb a 4.5%: política más acomodaticia de lo esperado → GBP bajista
  PMI manufacturero zona euro cae a 44.2 puntos: contracción industrial profundiza presión sobre el BCE → EUR bajista
  Petróleo WTI sube 8% por bloqueo del Estrecho de Ormuz: ingresos por exportación de crudo en riesgo → CAD alcista
  BOJ pospone alza de tasas de marzo por volatilidad de mercados: decisión dovish retrasa normalización → JPY bajista
  SNB podría intervenir para frenar apreciación excesiva del franco: techo implícito sobre valorización → CHF mixto
  EUR/USD cae 0.63% pese a inflación en zona euro: tensiones en Oriente Medio impulsan al USD → EUR bajista
  PIB de Canadá se contrae 0.6% en Q4 superando pronóstico de caída: debilidad económica presiona al BoC → CAD bajista
  Conflicto en Irán eleva petróleo a $80: costes energéticos zona euro suben y complican dilema del BCE → EUR bajista
  Guerra en Oriente Medio provoca risk-off global: divisas de alto riesgo bajo presión vendedora → NZD bajista
  Petróleo sube 5% por tensiones en Ormuz: soporte estructural para exportaciones canadienses de crudo → CAD alcista

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLA CRÍTICA DE PERSPECTIVA — NUNCA VIOLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El titular SIEMPRE describe el impacto desde la perspectiva de la DIVISA ASIGNADA.
NUNCA uses otra divisa como sujeto del sentimiento final.

INCORRECTO (artículo EUR, USD como sujeto):
  "Tensiones en Oriente Medio: aumento de la aversión al riesgo → USD alcista"

CORRECTO (artículo EUR, EUR como sujeto):
  "Tensiones en Oriente Medio elevan costes energéticos europeos: dilema para el BCE → EUR bajista"

INCORRECTO (artículo NZD, USD como sujeto):
  "Conflicto en Oriente Medio aviva temores de inflación → USD alcista"

CORRECTO (artículo NZD, NZD como sujeto):
  "Conflicto en Oriente Medio desencadena risk-off global: divisa de riesgo bajo presión → NZD bajista"

La regla: si el USD sube, el EUR/GBP/AUD/NZD/CAD/CHF/JPY bajan. Escribe SIEMPRE
el impacto SOBRE LA DIVISA ASIGNADA con su mecanismo específico.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLA DE MECANISMO ESPECÍFICO — CRÍTICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El mecanismo en el cuerpo del titular debe ser ESPECÍFICO para la divisa asignada,
no una descripción genérica del mercado global.

El user prompt incluye el "PERFIL MACRO" de la divisa asignada.
USA ese perfil para describir el mecanismo correcto.

INCORRECTO (genérico, aplica a cualquier divisa):
  "Conflicto en Irán genera incertidumbre en mercados → EUR bajista"
  "Tensión geopolítica pesa sobre los mercados → NZD bajista"

CORRECTO (específico para EUR):
  "Conflicto en Irán eleva precios energéticos: costes de importación suben en zona euro → EUR bajista"

CORRECTO (específico para NZD):
  "Conflicto en Irán desencadena aversión al riesgo global: kiwi bajo presión como divisa de alto riesgo → NZD bajista"

CORRECTO (específico para CAD):
  "Conflicto en Irán impulsa petróleo WTI a $80: soporte para exportaciones canadienses de crudo → CAD alcista"

CORRECTO (específico para JPY):
  "Conflicto en Irán eleva petróleo y riesgo global: yen enfrenta presión contrapuesta entre refugio y costes de importación → JPY mixto"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLA DE ESPECIFICIDAD — OBLIGATORIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Si el artículo menciona un porcentaje, precio, nivel o nombre de institución
concreto, el titular DEBE incluirlo.

INCORRECTO (dato disponible, titular genérico):
  "PIB canadiense se contrae en Q4: debilidad económica presiona al CAD → CAD bajista"

CORRECTO:
  "PIB de Canadá se contrae 0.6% en Q4: debilidad económica por encima del pronóstico presiona al BoC → CAD bajista"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLAS ABSOLUTAS — NUNCA VIOLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. USA SOLO información presente EXPLÍCITAMENTE en el título o descripción.
   Nunca inventes cifras, fechas, porcentajes ni instituciones que no aparezcan en el texto.

2. Si el artículo habla de un activo que NO es la divisa asignada (ej. bolsa turca, plata,
   ibovespa, rand sudafricano), responde únicamente: IRRELEVANTE

3. Nunca confundas el activo cubierto con la divisa afectada:
   "TSX cae" → artículo sobre CAD, no USD
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
  - Banco central mantiene tasas sin cambios cuando el mercado esperaba subida
  - Política acomodaticia / tasas reales negativas / condiciones expansivas

EXCEPCIÓN JPY — ACTIVO REFUGIO:
  En contextos de RIESGO GEOPOLÍTICO EXTREMO (guerra, crisis sistémica):
  - Artículo trata PRINCIPALMENTE de demanda de JPY como refugio → JPY alcista
  - Artículo trata PRINCIPALMENTE de política BOJ (dovish) → JPY bajista
  - Ambos factores presentes → JPY mixto

REGLA SNB — TRES ESTADOS:
  - SNB interviene ACTIVAMENTE para DEPRECIAR el franco (compra divisas) → CHF bajista
  - SNB AMENAZA con intervenir como techo implícito, CHF sigue estructuralmente fuerte → CHF mixto
  - Franco se aprecia sin mención de intervención SNB → CHF alcista

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRELACIONES MACROECONÓMICAS CLAVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PETRÓLEO / ENERGÍA:
  ↑ precio petróleo por crisis/guerra → CAD alcista (exportador neto)
  ↑ precio petróleo por crisis/guerra → EUR bajista (importador, costes energéticos)
  ↑ precio petróleo por crisis/guerra → JPY mixto (refugio vs costes importación)
  ↑ precio petróleo por crisis/guerra → NZD bajista (por risk-off, no por petróleo directamente)
  ↑ precio petróleo por crisis/guerra → AUD bajista (por risk-off; Australia importa petróleo refinado)

  DISTINCIÓN IMPORTANTE para NZD y AUD:
  - Petróleo sube por DEMANDA GLOBAL FUERTE (ciclo expansivo) → NZD/AUD pueden ser neutrales o alcistas
  - Petróleo sube por CRISIS/GUERRA (risk-off) → NZD bajista, AUD bajista

ACTIVOS REFUGIO (risk-off):
  ↑ tensión geopolítica / ↑ volatilidad → JPY alcista, CHF alcista (si no hay intervención SNB)
  ↑ tensión geopolítica → AUD bajista, NZD bajista (divisas de riesgo)
  ↑ tensión geopolítica → EUR bajista (importador de energía, más vulnerable)
  ↑ tensión geopolítica → GBP bajista (no es refugio)
  ↑ tensión geopolítica → USD alcista (refugio global)

CRECIMIENTO / DATOS MACRO:
  PIB / empleo / PMI por encima de expectativas → alcista
  PIB / empleo / PMI por debajo de expectativas → bajista
  Inflación por encima de meta + banco central reactivo → alcista (hawkish)
  Inflación por encima de meta + banco central permisivo → bajista (erosión)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CASOS ESPECIALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Análisis técnico con niveles: incluye el nivel clave, soporte/resistencia y sesgo
- Pares cruzados (EUR/GBP, GBP/JPY): describe el impacto en la DIVISA ASIGNADA
- Si el impacto es genuinamente ambiguo según el propio artículo: usa "mixto"
- Si no hay suficiente información analítica: responde solo: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur       = article.get("cur", "USD")
    lang      = article.get("lang", "en")
    # v9.2: inyectar perfil macro de la divisa
    macro_ctx = CURRENCY_MACRO_CONTEXT.get(cur, "")

    return (
        f"DIVISA ASIGNADA: {cur} — {CURRENCY_NAMES.get(cur, cur)}\n"
        f"PERFIL MACRO DE {cur}: {macro_ctx}\n"
        f"FUENTE: {article.get('source', '')} ({'español' if lang == 'es' else 'inglés'})\n"
        f"TÍTULO ORIGINAL: {article.get('title', '')}\n"
        f"DESCRIPCIÓN: {(article.get('expand', '') or '')[:500] or 'Sin descripción'}\n\n"
        f"INSTRUCCIÓN: Sintetiza en un titular donde:\n"
        f"  1. El sujeto del sentimiento final es SIEMPRE {cur}, nunca otra divisa.\n"
        f"  2. El mecanismo usa el PERFIL MACRO de {cur} (no una descripción genérica del mercado).\n"
        f"  3. Si el texto incluye cifras o niveles específicos, inclúyelos en el titular.\n"
        f"  4. Si el artículo NO trata sobre {cur} ni tiene impacto demostrable en {cur}, "
        f"responde solo: IRRELEVANTE\n\n"
        f"Titular (debe terminar en '→ {cur} [alcista|bajista|neutral|mixto]'):"
    )


# ─────────────────────────────────────────────
# SCORING DE SENTIMIENTO — v9.2 (sin cambios sobre v9.1)
# ─────────────────────────────────────────────
SENTIMENT_FINAL_RE = re.compile(
    r"→\s*([A-Z]{3})?\s*(alcista|bajista|neutral|neutro|mixto)\s*$",
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
    "posponer", "pospone", "retrasar", "retrasa", "postergar", "postpone",
    "put off", "hold off", "delay", "pause", "pausar",
    "acomodaticia", "acomodativo", "expansiva", "tasas reales negativas",
    "sin cambios", "mantiene tasas",
]
MIXED_KW   = ["mixto", "contradictorio", "ambiguo", "señal contradictoria"]
NEUTRAL_KW = ["neutral", "neutro", "sin impacto"]

POSTPONE_HIKE_RE = re.compile(
    r"\b(posponer?|retrasar?|postergar?|put off|hold off|delay|pause)\b.{0,60}"
    r"\b(alza|hike|subida|rate rise|suba de tasas|raise rates)\b",
    re.IGNORECASE,
)

SNB_ACTIVE_INTERVENTION_RE = re.compile(
    r"\b(snb|swiss national bank|banco nacional suizo)\b.{0,80}"
    r"\b(interviene|intervino|intervened|selling francs?|compra(ndo)? divisas|"
    r"fx (purchase|buying)|actively intervening)\b",
    re.IGNORECASE,
)
SNB_THREAT_RE = re.compile(
    r"\b(snb|swiss national bank|banco nacional suizo)\b.{0,120}"
    r"\b(podría intervenir|could intervene|may intervene|might intervene|"
    r"intervenir si|intervention (focus|risk|threat)|techo|cap|ceiling|"
    r"frenar (la )?apreciación|dampen|curb|limit (appreciation|strength))\b",
    re.IGNORECASE,
)


def score_from_headline(headline: str | None, cur: str = "USD") -> str:
    if not headline:
        return "neut"

    hl = headline.lower()

    m = SENTIMENT_FINAL_RE.search(headline)
    if m:
        pattern_cur  = m.group(1).upper() if m.group(1) else None
        pattern_word = m.group(2).lower()

        base_sent = {
            "alcista": "bull", "bajista": "bear",
            "neutral": "neut", "neutro":  "neut",
            "mixto":   "mixed",
        }.get(pattern_word, "neut")

        if pattern_cur and pattern_cur != cur.upper():
            if pattern_cur == "USD" and cur.upper() in MAJORS_VS_USD:
                if base_sent == "bull":
                    return "bear"
                elif base_sent == "bear":
                    return "bull"

        return base_sent

    if POSTPONE_HIKE_RE.search(hl):
        return "bear"

    if SNB_ACTIVE_INTERVENTION_RE.search(hl):
        return "bear"
    if SNB_THREAT_RE.search(hl):
        return "mixed"

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
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 401: return "invalid"
        if r.status_code == 429: return "daily_limit" if is_daily_limit_message(r) else "ok"
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)

            if r.status_code == 429:
                if is_daily_limit_message(r): return "DAILY_LIMIT"
                wait = get_retry_after(r)
                if wait > SKIP_IF_WAIT_EXCEEDS:
                    print(f"  ⏭️  Retry-After={wait}s muy largo — omitiendo artículo")
                    return None
                print(f"  ⏳ Rate limit (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401: return "DAILY_LIMIT"
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            if content.upper() == "IRRELEVANTE": return None

            content = content.strip('"\'').strip()
            for prefix in ("titular:", "headline:", "síntesis:", "análisis:", "titolo:"):
                if content.lower().startswith(prefix):
                    content = content[len(prefix):].strip()

            return content if len(content) >= 25 else None

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout (intento {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES: time.sleep(5)
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
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}  |  v9.2 institucional")
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

    # Recalcular sentimientos con lógica v9.2
    sentiments_recalculated = 0
    corrections_log = []
    for article in articles:
        if article.get("ai_headline"):
            cur      = article.get("cur", "USD")
            old_sent = article.get("sentiment", "neut")
            new_sent = score_from_headline(article["ai_headline"], cur)
            if old_sent != new_sent:
                article["sentiment"] = new_sent
                sentiments_recalculated += 1
                corrections_log.append(
                    f"    {cur} | {old_sent}→{new_sent} | {article['ai_headline'][:70]}..."
                )
    if sentiments_recalculated:
        print(f"\n   🔄 Sentimientos corregidos (v9.2): {sentiments_recalculated}")
        for log in corrections_log[:10]:
            print(log)
        if len(corrections_log) > 10:
            print(f"    ... y {len(corrections_log)-10} más")

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
            article["sentiment"]   = "neut"
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
            articles[article_idx]["sentiment"]   = score_from_headline(result, cur)
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

    for article in articles:
        if "sentiment" not in article:
            article["sentiment"] = score_from_headline(
                article.get("ai_headline"), article.get("cur", "USD")
            )

    save_progress(data, articles, now_utc)

    sent_by_cur: dict[str, Counter] = {}
    for a in articles:
        cur = a.get("cur", "?")
        s   = a.get("sentiment", "neut")
        if cur not in sent_by_cur:
            sent_by_cur[cur] = Counter()
        sent_by_cur[cur][s] += 1

    print()
    print("=" * 65)
    print("📊 SENTIMIENTO POR DIVISA (post-fix v9.2)")
    print("-" * 65)
    for cur in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        c = sent_by_cur.get(cur, Counter())
        total = sum(c.values())
        if total == 0: continue
        bull  = c.get("bull",  0)
        bear  = c.get("bear",  0)
        neut  = c.get("neut",  0)
        mixed = c.get("mixed", 0)
        dominant = max(c, key=c.get) if c else "neut"
        dominant_label = {"bull":"ALCISTA","bear":"BAJISTA","neut":"NEUTRAL","mixed":"MIXTO"}.get(dominant,"?")
        print(f"  {cur:4s}  bull={bull} bear={bear} neut={neut} mixed={mixed}  → {dominant_label}")

    print()
    print("📋 RESUMEN ENRIQUECIMIENTO")
    print(f"   ✅ Enriquecidos:            {enriched}")
    print(f"   🔄 Sentimientos corregidos: {sentiments_recalculated}")
    print(f"   ⛔ No enriquecibles:        {not_enrichable}")
    print(f"   ⚠️  Error API:              {skipped}")
    print(f"   ⏭️  Irrelevantes:           {irrelevant}")
    if stopped_early:
        print(f"   ⏰ Detenido antes de completar")
    print(f"   🔑 Keys usadas:            hasta Key {current_key_idx + 1} de {len(available_keys)}")
    print(f"   ⏱️  Tiempo total:           {elapsed_min():.1f} min")
    print(f"   💾 Guardado en:            {NEWS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
