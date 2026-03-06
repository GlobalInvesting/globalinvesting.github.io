#!/usr/bin/env python3
"""
generate_summaries.py — v3.2
Genera un bloque de análisis consolidado por divisa a partir de news.json.
Llama a Groq una vez por divisa (8 llamadas total) y escribe summaries.json.

CAMBIOS v3.2 (sobre v3.1):
  CALENDARIO ECONÓMICO — FIX PRINCIPAL:
    · Lee news-data/calendar.json si existe y extrae los próximos eventos por divisa
      (solo aquellos cuya fecha+hora UTC es estrictamente futura respecto a now_utc).
    · Inyecta esos eventos en el prompt de usuario bajo la sección
      "PRÓXIMOS EVENTOS DEL CALENDARIO (ya verificados como futuros)".
    · El modelo ya NO necesita inferir si un evento ocurrió o no: recibe
      explícitamente la lista de eventos pendientes o la indicación de que no hay ninguno.
    · Post-proceso defensivo: si `upcoming_event` devuelto por el modelo contiene
      palabras clave de eventos que ya aparecen como ocurridos en el calendario
      (actual != ""), se fuerza a null.

  OBSERVABILIDAD:
    · Log al inicio indicando cuántos eventos futuros se encontraron por divisa.
    · Log por divisa si se suprime un upcoming_event obsoleto.

  Sin cambios en lógica de prompts base, modelo, keys, rate limiting,
  ni estructura JSON de output.
"""

import os
import sys
import json
import time
import re
import requests
from datetime import datetime, timezone
from pathlib import Path

# FIX C-01: Configuración de divisas centralizada en fx_config.py.
# Ya no se duplican CURRENCIES, CURRENCY_NAMES ni CURRENCY_MACRO_CONTEXT.
sys.path.insert(0, os.path.dirname(__file__))
from fx_config import CURRENCIES, CURRENCY_NAMES, CURRENCY_MACRO_CONTEXT

# ─────────────────────────────────────────────
NEWS_FILE      = Path("news-data/news.json")
CALENDAR_FILE  = Path("news-data/calendar.json")   # v3.2: leído si existe
SUMMARIES_FILE = Path("news-data/summaries.json")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 800
TEMPERATURE    = 0.3
SLEEP_BETWEEN  = 8
MAX_RETRIES    = 2

# Máximo de eventos del calendario a incluir en el prompt por divisa
MAX_CALENDAR_EVENTS_IN_PROMPT = 4

# ─────────────────────────────────────────────
# v3.2: CALENDAR UTILITIES
# ─────────────────────────────────────────────

def load_upcoming_calendar_events(now_utc: datetime) -> dict:
    """
    Lee calendar.json y devuelve un dict {currency: [event_str, ...]}
    con SOLO los eventos cuya fecha+hora UTC es estrictamente futura.

    Formato esperado de calendar.json (estructura real del proyecto):
      {
        "events": [
          {
            "dateISO": "2026-03-11",
            "timeUTC": "12:30",
            "currency": "USD",
            "event": "Core CPI (MoM) (Feb)",
            "impact": "high",
            "actual": "",        <-- vacío = no ha ocurrido aún
            "forecast": "0.2%",
            "previous": "0.3%"
          }, ...
        ]
      }

    Un evento se considera "futuro" cuando:
      · actual == "" (no hay dato publicado), Y
      · su datetime UTC (dateISO + timeUTC) > now_utc

    Eventos con actual != "" ya ocurrieron → se excluyen.
    """
    upcoming: dict = {cur: [] for cur in CURRENCIES}

    if not CALENDAR_FILE.exists():
        print("  [calendar] calendar.json no encontrado — se omite sección de próximos eventos")
        return upcoming

    try:
        with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
    except Exception as e:
        print(f"  [calendar] Error leyendo calendar.json: {e}")
        return upcoming

    events = cal_data.get("events", [])
    total_future = 0

    for ev in events:
        cur       = ev.get("currency", "").upper()
        actual    = (ev.get("actual") or "").strip()
        date_iso  = ev.get("dateISO", "")
        time_utc  = ev.get("timeUTC", "00:00")
        event_name = ev.get("event", "")
        impact    = ev.get("impact", "low")
        forecast  = (ev.get("forecast") or "").strip()

        # Skip already-published events
        if actual:
            continue

        # Skip non-tracked currencies
        if cur not in CURRENCIES:
            continue

        # Parse event datetime in UTC
        try:
            ev_dt = datetime.strptime(f"{date_iso} {time_utc}", "%Y-%m-%d %H:%M")
            ev_dt = ev_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        # Must be strictly in the future
        if ev_dt <= now_utc:
            continue

        # Format a concise human-readable string for the prompt
        # e.g. "Core CPI (MoM) — Mié 11 Mar 12:30 UTC [high] prev: 0.3%"
        day_name = ev_dt.strftime("%a %d %b")
        fc_str   = f" prev: {ev.get('previous','')}" if ev.get("previous") else ""
        if forecast:
            fc_str = f" prev: {ev.get('previous','')} fcst: {forecast}"
        entry = f"{event_name} — {day_name} {time_utc} UTC [{impact}]{fc_str}"
        upcoming[cur].append((ev_dt, entry))
        total_future += 1

    # Sort by datetime and keep only the nearest MAX_CALENDAR_EVENTS_IN_PROMPT
    for cur in CURRENCIES:
        upcoming[cur].sort(key=lambda x: x[0])
        upcoming[cur] = [e for _, e in upcoming[cur][:MAX_CALENDAR_EVENTS_IN_PROMPT]]

    print(f"  [calendar] {total_future} eventos futuros cargados desde calendar.json")
    for cur in CURRENCIES:
        n = len(upcoming[cur])
        if n:
            print(f"    {cur}: {n} evento{'s' if n>1 else ''} próximo{'s' if n>1 else ''}")

    return upcoming


def build_already_occurred_set(now_utc: datetime) -> set:
    """
    Devuelve un set de palabras clave en minúsculas de eventos que ya
    tienen 'actual' publicado en calendar.json.
    Se usa como post-proceso defensivo para suprimir upcoming_event obsoletos.
    """
    occurred: set = set()

    if not CALENDAR_FILE.exists():
        return occurred

    try:
        with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
    except Exception:
        return occurred

    for ev in cal_data.get("events", []):
        actual = (ev.get("actual") or "").strip()
        if actual:
            # Store normalised event name tokens for fuzzy matching
            name = ev.get("event", "").lower()
            # Also mark events whose time has passed even if actual is missing
            try:
                date_iso = ev.get("dateISO", "")
                time_utc = ev.get("timeUTC", "00:00")
                ev_dt = datetime.strptime(f"{date_iso} {time_utc}", "%Y-%m-%d %H:%M")
                ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                if ev_dt <= now_utc:
                    occurred.add(name)
            except ValueError:
                if actual:
                    occurred.add(name)

    return occurred


def is_upcoming_event_stale(event_text: str, occurred_set: set) -> bool:
    """
    Returns True if the upcoming_event string from the model looks like
    an event that has already occurred according to the calendar.
    Uses a simple token overlap heuristic.
    """
    if not event_text or not occurred_set:
        return False

    text_lower = event_text.lower()
    # Common stale-event signals
    STALE_PHRASES = [
        "esta noche", "esta mañana", "esta tarde", "today", "tonight",
        "this morning", "earlier today",
    ]
    for phrase in STALE_PHRASES:
        if phrase in text_lower:
            return True

    # Check token overlap with known occurred events
    text_tokens = set(re.findall(r"[a-z]{3,}", text_lower))
    for occurred_name in occurred_set:
        occurred_tokens = set(re.findall(r"[a-z]{3,}", occurred_name))
        if len(occurred_tokens) >= 2:
            overlap = text_tokens & occurred_tokens
            if len(overlap) >= 2:
                return True

    return False


# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista senior de mercados de divisas (FX) de una mesa institucional de trading con más de 15 años de experiencia. Tu especialidad es sintetizar múltiples fuentes de información en análisis accionables, precisos y con contexto macroeconómico profundo.

Tu tarea es leer una serie de titulares y descripciones de noticias sobre una divisa específica y producir un análisis consolidado de alta calidad.

═══════════════════════════════════════════════
INSTRUCCIONES DE REDACCIÓN
═══════════════════════════════════════════════

EXTENSIÓN Y ESTRUCTURA:
- El campo "analysis" debe tener entre 120 y 200 palabras.
- Redacta en párrafos fluidos, como un briefing institucional real.
- Usa un tono profesional pero directo, evitando frases genéricas o vagas.

CONTENIDO OBLIGATORIO — incluye SIEMPRE que los datos estén disponibles:
1. Contexto macro principal: ¿qué evento o dato está moviendo esta divisa hoy?
2. Niveles técnicos clave: menciona niveles de precio, soportes o resistencias cuando aparezcan en los titulares (ej: "EUR/USD en 1.1590, soporte en 1.1540").
3. Postura del banco central: hawkish / dovish / en espera, con implicación concreta para la divisa.
4. Catalizadores secundarios específicos de esta divisa (no del mercado global).
5. Perspectiva a corto plazo: próximo evento o dato que puede cambiar el sesgo.

═══════════════════════════════════════════════
REGLA CRÍTICA — CONTEXTO GLOBAL VS. DIVISA ESPECÍFICA
═══════════════════════════════════════════════

Cuando un evento macro global (guerra, petróleo, datos de empleo EEUU) afecte a esta divisa, DEBES explicar el mecanismo específico de transmisión para ESTA divisa en particular usando su PERFIL MACRO:

- NO escribas: "La tensión en Medio Oriente genera incertidumbre."
- SÍ escribe (para EUR): "La escalada en Medio Oriente eleva los costes energéticos europeos, creando un dilema para el BCE entre contener inflación y evitar recesión."
- SÍ escribe (para CAD): "La escalada en Medio Oriente impulsa el petróleo WTI, ofreciendo soporte estructural al CAD como economía exportadora de crudo."
- SÍ escribe (para NZD): "La escalada en Medio Oriente desencadena risk-off global, presionando al NZD como divisa de alto riesgo."

Si el mismo evento aparece en el análisis de varias divisas, el mecanismo de transmisión debe ser DIFERENTE y ESPECÍFICO para cada una.

═══════════════════════════════════════════════
REGLA CRÍTICA — SEÑALES CONTRADICTORIAS
═══════════════════════════════════════════════

Si hay señales alcistas Y bajistas en los titulares, NO las ignores. Debes:
1. Reconocer explícitamente la contradicción.
2. Explicar cuál señal domina actualmente y por qué.
3. Reflejar esto en el sentimiento: usa "mixed" si ninguna domina claramente.

PROHIBIDO: asignar "bull" en sentimiento y mencionar en el análisis que el banco central es dovish sin explicar la contradicción.

═══════════════════════════════════════════════
REGLA DE DRIVERS
═══════════════════════════════════════════════

- Máximo 4 drivers, ordenados por importancia.
- Al menos 2 deben ser ESPECÍFICOS de esta divisa o su banco central.
- Un driver de contexto global solo es válido si incluye el mecanismo: "Petróleo >$80 → soporte CAD", "Risk-off → CHF refugio".
- Formato: frases de 2-6 palabras con cifras cuando sea posible.
- PROHIBIDO: drivers de una sola palabra ("Tensión", "Datos", "Mercado", "Guerra").

═══════════════════════════════════════════════
REGLA CRÍTICA — UPCOMING_EVENT
═══════════════════════════════════════════════

El campo "upcoming_event" SOLO debe referenciar eventos de la sección
"PRÓXIMOS EVENTOS DEL CALENDARIO" que se te proporciona más abajo.
Esos eventos han sido pre-verificados como FUTUROS (aún no publicados).

- Si la sección de próximos eventos está vacía o marcada como "Sin eventos",
  devuelve null en upcoming_event.
- NUNCA inventes un evento que no esté en esa lista.
- NUNCA menciones eventos que ya hayan ocurrido (NFP de hoy, datos de esta mañana, etc.).
- Formato: "Nombre del evento — fecha y hora local o UTC si se proporciona"

═══════════════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════════════

JSON puro, sin markdown, sin texto extra:

{
  "sentiment": "bull|bear|neut|mixed",
  "confidence": 0-100,
  "analysis": "análisis completo entre 120 y 200 palabras",
  "drivers": ["Driver específico 1", "Driver específico 2", "Driver 3", "Driver 4"],
  "upcoming_event": "Descripción breve del evento próximo más relevante, o null"
}

REGLAS DE SENTIMIENTO:
- bull:  señales predominantemente positivas para la divisa (>60% alcistas)
- bear:  señales predominantemente negativas para la divisa (>60% bajistas)
- mixed: señales contradictorias o equilibradas
- neut:  sin señal clara o cobertura muy limitada (<2 noticias)

REGLA DE CONFIDENCE:
- 90-100: consenso absoluto, todas las fuentes apuntan al mismo lado
- 70-89:  mayoría clara con algún matiz menor
- 50-69:  tendencia moderada con contradicciones relevantes
- 30-49:  señal débil, mixta, o pocas fuentes disponibles
- 0-29:   sin señal (usar neut)"""


def build_user_prompt(cur: str, articles: list, upcoming_events: list) -> str:
    name      = CURRENCY_NAMES.get(cur, cur)
    macro_ctx = CURRENCY_MACRO_CONTEXT.get(cur, "")

    high_impact = [a for a in articles if a.get("impact") == "high"]
    other       = [a for a in articles if a.get("impact") != "high"]

    lines = []

    if high_impact:
        lines.append("── ALTO IMPACTO ──")
        for a in high_impact:
            title   = a.get("title", "")
            source  = a.get("source", "")
            expand  = (a.get("expand", "") or "").strip()
            snippet = f"\n    Contexto: {expand[:300]}..." if expand and len(expand.split()) > 8 else ""
            lines.append(f"  • {title}  ({source}){snippet}")

    if other:
        lines.append("── MEDIO/BAJO IMPACTO ──")
        for a in other:
            title  = a.get("title", "")
            source = a.get("source", "")
            expand  = (a.get("expand", "") or "").strip()
            snippet = f"\n    Contexto: {expand[:200]}..." if expand and len(expand.split()) > 8 else ""
            lines.append(f"  • {title}  ({source}){snippet}")

    headlines_block = "\n".join(lines) if lines else "Sin noticias disponibles."

    # ── v3.2: CALENDAR SECTION ────────────────────────────────────────────────
    if upcoming_events:
        calendar_block = "\n".join(f"  · {ev}" for ev in upcoming_events)
        calendar_section = (
            f"\nPRÓXIMOS EVENTOS DEL CALENDARIO (ya verificados como futuros, no han ocurrido aún):\n"
            f"{calendar_block}\n"
        )
    else:
        calendar_section = (
            "\nPRÓXIMOS EVENTOS DEL CALENDARIO: Sin eventos relevantes pendientes para esta divisa "
            "en el horizonte inmediato. Devuelve null en upcoming_event.\n"
        )

    return (
        f"DIVISA: {cur} — {name}\n"
        f"PERFIL MACRO DE {cur}: {macro_ctx}\n"
        f"TOTAL ARTÍCULOS: {len(articles)}\n\n"
        f"TITULARES Y DESCRIPCIONES:\n"
        f"{headlines_block}\n"
        f"{calendar_section}\n"
        f"INSTRUCCIONES ESPECÍFICAS PARA {cur}:\n"
        f"1. Usa el PERFIL MACRO para explicar cómo los eventos globales afectan ESPECÍFICAMENTE al {cur}.\n"
        f"2. Si hay señales contradictorias, identifícalas y explica cuál domina.\n"
        f"3. Al menos 2 drivers deben ser específicos del {cur} o su banco central.\n"
        f"4. Incluye niveles de precio concretos si aparecen en los titulares o contexto.\n"
        f"5. Para upcoming_event: usa SOLO la lista de próximos eventos del calendario de arriba.\n\n"
        f"Genera el análisis consolidado institucional en JSON para {cur}. "
        f"Recuerda: entre 120 y 200 palabras en 'analysis'."
    )


# ─────────────────────────────────────────────
def load_api_keys() -> list:
    keys = []
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = os.environ.get(var, "").strip()
        if k:
            keys.append(k)
    return keys


def mask_key(key: str) -> str:
    # FIX S-02: Reducido de key[:8] a key[:4] para no exponer demasiado
    # de keys tipo 'gsk_XXXXXXXXXXXX' en logs públicos de GitHub Actions.
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "***"


def get_retry_after(response) -> int:
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        val = response.headers.get(header)
        if val:
            try:
                return min(int(float(str(val).lower().replace("s", "").strip())) + 2, 120)
            except (ValueError, TypeError):
                pass
    return 65


def is_daily_limit(response) -> bool:
    try:
        body = response.json()
        msg  = str(body.get("error", {}).get("message", "")).lower()
        return ("per day" in msg or "tpd" in msg) and "per minute" not in msg
    except Exception:
        return False


def call_groq(api_key: str, cur: str, articles: list, upcoming_events: list) -> dict | None:
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(cur, articles, upcoming_events)},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

            if r.status_code == 429:
                if is_daily_limit(r):
                    return "DAILY_LIMIT"
                wait = get_retry_after(r)
                print(f"    ⏳ Rate limit (intento {attempt}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                return "DAILY_LIMIT"

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()

            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"```\s*$",    "", raw).strip()

            parsed = json.loads(raw)

            required = {"sentiment", "confidence", "analysis", "drivers"}
            if not required.issubset(parsed.keys()):
                print(f"    ⚠️  JSON incompleto para {cur}: {list(parsed.keys())}")
                return None

            word_count = len(parsed.get("analysis", "").split())
            if word_count < 80:
                print(f"    ⚠️  Análisis demasiado corto para {cur}: {word_count} palabras — reintentando")
                return None

            parsed["sentiment"]      = parsed["sentiment"].lower().strip()
            parsed["confidence"]     = max(0, min(100, int(parsed["confidence"])))
            parsed["drivers"]        = [str(d)[:60] for d in parsed.get("drivers", [])[:4]]
            parsed["upcoming_event"] = parsed.get("upcoming_event") or None

            print(f"    📝 {word_count} palabras generadas")
            return parsed

        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parse error para {cur}: {e}")
            return None
        except requests.exceptions.Timeout:
            print(f"    ⚠️  Timeout (intento {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(8)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            return None

    return None


def fallback_summary(cur: str, articles: list) -> dict:
    name  = CURRENCY_NAMES.get(cur, cur)
    total = len(articles)
    high  = sum(1 for a in articles if a.get("impact") == "high")

    analysis = (
        f"Análisis basado en {total} {'noticia' if total == 1 else 'noticias'} "
        f"para {name} ({high} de alto impacto). "
        f"Resumen AI no disponible en este ciclo."
    )
    sources = list(set(a.get("source", "") for a in articles if a.get("source")))[:3]

    return {
        "sentiment":      "neut",
        "confidence":     0,
        "analysis":       analysis,
        "drivers":        [],
        "upcoming_event": None,
        "ai_generated":   False,
        "sources":        sources,
        "articles_count": total,
        "latest_ts":      max((a.get("ts", 0) for a in articles), default=0),
    }


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 65)
    print(f"📊 generate_summaries.py v3.2 — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Sleep entre llamadas: {SLEEP_BETWEEN}s")
    print("=" * 65)

    all_keys = load_api_keys()
    if not all_keys:
        print("⚠️  Sin GROQ_API_KEY — generando resúmenes con fallback estadístico")
    else:
        print(f"🔑 Keys disponibles: {len(all_keys)}")
        for i, k in enumerate(all_keys, 1):
            print(f"   Key {i}: {mask_key(k)}")

    current_key_idx = 0
    use_ai = bool(all_keys)

    if not NEWS_FILE.exists():
        print(f"❌ No se encontró {NEWS_FILE}")
        sys.exit(1)

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    articles = news_data.get("articles", [])
    print(f"\n📰 Artículos cargados: {len(articles)}")

    # ── v3.2: load calendar ───────────────────────────────────────────────────
    print(f"\n📅 Cargando calendario económico...")
    upcoming_by_currency = load_upcoming_calendar_events(now_utc)
    occurred_set         = build_already_occurred_set(now_utc)
    print(f"   Eventos ya ocurridos en caché: {len(occurred_set)}")

    groups = {cur: [] for cur in CURRENCIES}
    for a in articles:
        cur = a.get("cur", "")
        if cur in groups:
            groups[cur].append(a)

    for cur in CURRENCIES:
        groups[cur].sort(key=lambda x: x.get("ts", 0), reverse=True)

    print(f"\n   Distribución de artículos:")
    for cur in CURRENCIES:
        n = len(groups[cur])
        high = sum(1 for a in groups[cur] if a.get("impact") == "high")
        ev_count = len(upcoming_by_currency.get(cur, []))
        print(f"   {cur}: {n} artículo{'s' if n != 1 else ''} ({high} high) | {ev_count} eventos próximos")

    summaries    = {}
    generated    = 0
    fallbacks    = 0
    ai_curs      = []
    fallback_curs = []
    suppressed_events = 0

    print(f"\n{'─'*65}")
    for cur in CURRENCIES:
        cur_articles     = groups[cur]
        cur_upcoming     = upcoming_by_currency.get(cur, [])
        n = len(cur_articles)
        print(f"\n[{cur}] {CURRENCY_NAMES[cur]} — {n} artículo{'s' if n!=1 else ''} | {len(cur_upcoming)} próximos eventos")

        if n == 0:
            summaries[cur] = {
                "sentiment":      "neut",
                "confidence":     0,
                "analysis":       f"Sin cobertura para {CURRENCY_NAMES[cur]} en las últimas 72h.",
                "drivers":        [],
                "upcoming_event": None,
                "ai_generated":   False,
                "sources":        [],
                "articles_count": 0,
                "latest_ts":      0,
            }
            fallbacks += 1
            fallback_curs.append(cur)
            print(f"  ⚪ Sin artículos — resumen vacío")
            continue

        result = None
        if use_ai and current_key_idx < len(all_keys):
            # v3.2: pass upcoming_events to call_groq
            result = call_groq(all_keys[current_key_idx], cur, cur_articles, cur_upcoming)

            if result == "DAILY_LIMIT":
                print(f"  ⛔ Key {current_key_idx+1} agotada")
                current_key_idx += 1
                while current_key_idx < len(all_keys):
                    print(f"  🔄 Usando Key {current_key_idx+1}")
                    result = call_groq(all_keys[current_key_idx], cur, cur_articles, cur_upcoming)
                    if result != "DAILY_LIMIT":
                        break
                    print(f"  ⛔ Key {current_key_idx+1} también agotada")
                    current_key_idx += 1
                if result == "DAILY_LIMIT":
                    print(f"  ⛔ Todas las keys agotadas — fallback")
                    use_ai = False
                    result = None

        if result and isinstance(result, dict):
            # ── v3.2: post-process — suppress stale upcoming_event ────────────
            if result.get("upcoming_event"):
                if is_upcoming_event_stale(result["upcoming_event"], occurred_set):
                    print(f"  🗑️  upcoming_event suprimido (evento ya ocurrido): "
                          f"\"{result['upcoming_event'][:60]}\"")
                    result["upcoming_event"] = None
                    suppressed_events += 1
                elif not cur_upcoming:
                    # Model hallucinated an event when there were none in the calendar
                    print(f"  🗑️  upcoming_event suprimido (no había eventos en calendario): "
                          f"\"{result['upcoming_event'][:60]}\"")
                    result["upcoming_event"] = None
                    suppressed_events += 1

            sources = list(set(a.get("source", "") for a in cur_articles if a.get("source")))[:4]
            latest  = max(a.get("ts", 0) for a in cur_articles)
            summaries[cur] = {
                **result,
                "ai_generated":   True,
                "sources":        sources,
                "articles_count": n,
                "latest_ts":      latest,
            }
            generated += 1
            ai_curs.append(cur)
            sent_label = {"bull":"ALCISTA","bear":"BAJISTA","neut":"NEUTRAL","mixed":"MIXTO"}.get(
                result["sentiment"], result["sentiment"].upper()
            )
            ev_str = f" | próximo: {result['upcoming_event'][:40]}" if result.get("upcoming_event") else " | sin próximo"
            print(f"  ✅ AI → {sent_label} ({result['confidence']}%) | {len(result['drivers'])} drivers{ev_str}")
            print(f"     {result['analysis'][:120]}...")
        else:
            fb = fallback_summary(cur, cur_articles)
            summaries[cur] = fb
            fallbacks += 1
            fallback_curs.append(cur)
            print(f"  📊 Fallback → sin análisis AI disponible")

        if use_ai and cur != CURRENCIES[-1]:
            time.sleep(SLEEP_BETWEEN)

    SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "updated_utc":    now_utc.isoformat(),
        "updated_label":  now_utc.strftime("%H:%M UTC"),
        "model":          GROQ_MODEL if generated > 0 else "fallback",
        "total_ai":       generated,
        "total_fallback": fallbacks,
        "summaries":      summaries,
    }

    # FIX R-02: Serializar en memoria y validar antes de escribir a disco.
    # Previene archivos corruptos en el repo ante errores de encoding o interrupciones.
    try:
        output_json = json.dumps(output, ensure_ascii=False, indent=2)
        json.loads(output_json)  # Validar que es parseable
    except (TypeError, ValueError) as e:
        print(f"\n❌ Error crítico: el output generado no es JSON válido: {e}")
        print("   No se escribió el archivo para evitar corromper el repositorio.")
        sys.exit(1)

    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
        f.write(output_json)

    print(f"\n{'='*65}")
    print(f"📋 RESUMEN FINAL")
    print(f"   ✅ Generados con AI:        {generated}/8  {' · '.join(ai_curs) if ai_curs else '—'}")
    print(f"   📊 Fallback:                {fallbacks}/8  {' · '.join(fallback_curs) if fallback_curs else '—'}")
    print(f"   🗑️  Upcoming events suprim.: {suppressed_events}")
    print(f"   💾 Guardado en:             {SUMMARIES_FILE}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
