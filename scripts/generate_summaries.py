#!/usr/bin/env python3
"""
generate_summaries.py — v3.0
Genera un bloque de análisis consolidado por divisa a partir de news.json.
Llama a Groq una vez por divisa (8 llamadas total) y escribe summaries.json.

CAMBIOS v3.0 (sobre v2.1):
  ARQUITECTURA — eliminado enrich_news.py del pipeline:
    El script ya no depende de ai_headline ni sentiment pre-calculados.
    Trabaja directamente con title + expand (descripción cruda del RSS).
    Esto libera todas las keys de Groq para el análisis consolidado,
    que es el único output visible en el frontend.

  INPUT — build_user_prompt() adaptado:
    Usa title + expand directamente en lugar de ai_headline + sentiment.
    Incluye el expand completo (hasta 300 chars) como contexto para el modelo.
    Mantiene separación high-impact vs medio/bajo por campo "impact".

  KEYS — uso secuencial normal Key 1 → Key 2 → Key 3:
    Sin reserva de keys. Las 8 llamadas consumen ~800 tokens cada una,
    muy por debajo del límite diario por key.
"""

import os
import sys
import json
import time
import re
import requests
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
NEWS_FILE      = Path("news-data/news.json")
SUMMARIES_FILE = Path("news-data/summaries.json")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 800
TEMPERATURE    = 0.3
SLEEP_BETWEEN  = 6
MAX_RETRIES    = 2

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

CURRENCY_NAMES = {
    "USD": "Dólar Estadounidense",
    "EUR": "Euro",
    "GBP": "Libra Esterlina",
    "JPY": "Yen Japonés",
    "AUD": "Dólar Australiano",
    "CAD": "Dólar Canadiense",
    "CHF": "Franco Suizo",
    "NZD": "Dólar Neozelandés",
}

# Perfil macro de cada divisa — ancla el mecanismo de transmisión correcto
CURRENCY_MACRO_CONTEXT = {
    "USD": "Activo refugio global y divisa de reserva. Se beneficia de risk-off, tensiones geopolíticas y datos macro sólidos en EEUU. Sensible a postura Fed (hawkish/dovish) y al diferencial de tasas con otras economías G10.",
    "EUR": "Importador neto de energía. Conflicto geopolítico = mayores costes energéticos = presión sobre crecimiento eurozona = dilema BCE entre inflación y recesión. Sensible a spreads de bonos periféricos y postura BCE.",
    "GBP": "No es activo refugio. Sensible a inflación UK, política del BoE y datos laborales británicos. En risk-off cae frente a USD, JPY y CHF.",
    "JPY": "Activo refugio tradicional pero debilitado cuando sube el petróleo (Japón importa casi todo su crudo). Driver dominante: diferencial tasas US-JP. Fed hawkish o BoJ dovish = JPY bajista.",
    "AUD": "Divisa de riesgo correlacionada con commodities (hierro, cobre) y ciclo económico chino. En entornos risk-off cae. RBA hawkish ofrece soporte doméstico.",
    "CAD": "Correlacionada con petróleo WTI: Canadá es exportador neto de crudo. Petróleo alto = soporte estructural CAD incluso con BoC dovish. USMCA y comercio con EEUU son el mayor riesgo de cola.",
    "CHF": "Activo refugio por excelencia. Se aprecia en crisis/guerra/risk-off. SNB puede intervenir para limitar apreciación excesiva — distinguir intervención activa (bajista) de amenaza como techo (mixto).",
    "NZD": "Divisa de riesgo de alta beta. En crisis/guerra cae por risk-off global (no por correlación directa con petróleo). RBNZ y datos domésticos NZ son drivers fundamentales propios.",
}

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
- 0-29:   sin señal (usar neut)

REGLA DE UPCOMING_EVENT:
- Solo incluir si aparece explícitamente en los titulares o es consecuencia directa de la noticia.
- Formato: "Nombre del evento — fecha/plazo si se menciona"
- Si no hay ninguno relevante: null"""


def build_user_prompt(cur: str, articles: list) -> str:
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

    return (
        f"DIVISA: {cur} — {name}\n"
        f"PERFIL MACRO DE {cur}: {macro_ctx}\n"
        f"TOTAL ARTÍCULOS: {len(articles)}\n\n"
        f"TITULARES Y DESCRIPCIONES:\n"
        f"{headlines_block}\n\n"
        f"INSTRUCCIONES ESPECÍFICAS PARA {cur}:\n"
        f"1. Usa el PERFIL MACRO para explicar cómo los eventos globales afectan ESPECÍFICAMENTE al {cur}.\n"
        f"2. Si hay señales contradictorias, identifícalas y explica cuál domina.\n"
        f"3. Al menos 2 drivers deben ser específicos del {cur} o su banco central.\n"
        f"4. Incluye niveles de precio concretos si aparecen en los titulares o contexto.\n\n"
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
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"


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


def call_groq(api_key: str, cur: str, articles: list) -> dict | None:
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(cur, articles)},
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
    print(f"📊 generate_summaries.py v3.0 — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print("   Pipeline: fetch_news → generate_summaries (sin enrich)")
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

    groups = {cur: [] for cur in CURRENCIES}
    for a in articles:
        cur = a.get("cur", "")
        if cur in groups:
            groups[cur].append(a)

    for cur in CURRENCIES:
        groups[cur].sort(key=lambda x: x.get("ts", 0), reverse=True)

    print(f"\n   Distribución:")
    for cur in CURRENCIES:
        n = len(groups[cur])
        high = sum(1 for a in groups[cur] if a.get("impact") == "high")
        print(f"   {cur}: {n} artículo{'s' if n != 1 else ''} ({high} high)")

    summaries = {}
    generated = 0
    fallbacks = 0

    print(f"\n{'─'*65}")
    for cur in CURRENCIES:
        cur_articles = groups[cur]
        n = len(cur_articles)
        print(f"\n[{cur}] {CURRENCY_NAMES[cur]} — {n} artículo{'s' if n!=1 else ''}")

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
            print(f"  ⚪ Sin artículos — resumen vacío")
            continue

        result = None
        if use_ai and current_key_idx < len(all_keys):
            result = call_groq(all_keys[current_key_idx], cur, cur_articles)

            if result == "DAILY_LIMIT":
                print(f"  ⛔ Key {current_key_idx+1} agotada")
                current_key_idx += 1
                while current_key_idx < len(all_keys):
                    print(f"  🔄 Usando Key {current_key_idx+1}")
                    result = call_groq(all_keys[current_key_idx], cur, cur_articles)
                    if result != "DAILY_LIMIT":
                        break
                    print(f"  ⛔ Key {current_key_idx+1} también agotada")
                    current_key_idx += 1
                if result == "DAILY_LIMIT":
                    print(f"  ⛔ Todas las keys agotadas — fallback")
                    use_ai = False
                    result = None

        if result and isinstance(result, dict):
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
            sent_label = {"bull":"ALCISTA","bear":"BAJISTA","neut":"NEUTRAL","mixed":"MIXTO"}.get(
                result["sentiment"], result["sentiment"].upper()
            )
            print(f"  ✅ AI → {sent_label} ({result['confidence']}%) | {len(result['drivers'])} drivers")
            print(f"     {result['analysis'][:120]}...")
        else:
            fb = fallback_summary(cur, cur_articles)
            summaries[cur] = fb
            fallbacks += 1
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

    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"📋 RESUMEN FINAL")
    print(f"   ✅ Generados con AI:   {generated}/8")
    print(f"   📊 Fallback:           {fallbacks}/8")
    print(f"   💾 Guardado en:        {SUMMARIES_FILE}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
