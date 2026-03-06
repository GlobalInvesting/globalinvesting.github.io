#!/usr/bin/env python3
"""
generate_summaries.py — v2.1
Genera un bloque de análisis consolidado por divisa a partir de news.json.
Llama a Groq una vez por divisa (8 llamadas total) y escribe summaries.json.

CAMBIOS v2.1 (sobre v2.0):
  PROMPT — instrucción explícita para evitar repetir el contexto geopolítico
    global en todas las divisas sin explicar el mecanismo específico de
    transmisión a cada una. Drivers ahora deben ser específicos por divisa.
  PROMPT — instrucción para resolver explícitamente señales contradictorias
    en lugar de omitirlas (ej: CAD bull con BoC dovish).
  PROMPT — CURRENCY_MACRO_CONTEXT: perfil macro de cada divisa inyectado
    en el user prompt para anclar el análisis al mecanismo correcto.
  VALIDACIÓN — rechazo de análisis con < 80 palabras (subido desde 50).
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

# Perfil macro de cada divisa — inyectado en el user prompt
# para anclar el análisis al mecanismo de transmisión correcto
CURRENCY_MACRO_CONTEXT = {
    "USD": "Activo refugio global y divisa de reserva. Se beneficia de risk-off, tensiones geopolíticas y datos macro sólidos en EEUU. Sensible a postura Fed (hawkish/dovish) y al diferencial de tasas con otras economías G10.",
    "EUR": "Divisa sensible al crecimiento de la Eurozona, spreads de bonos periféricos y postura BCE. El conflicto geopolítico eleva costes energéticos europeos, presionando el crecimiento y creando dilema para el BCE entre inflación y recesión.",
    "GBP": "Divisa sensible a inflación UK, expectativas de política del BoE y datos del mercado laboral británico. No es activo refugio: en entornos risk-off cae frente a USD, JPY y CHF.",
    "JPY": "Activo refugio tradicional, aunque su rol se debilita cuando sube el petróleo (Japón importa casi todo su crudo). El diferencial de tasas US-JP es el driver dominante: Fed hawkish o BoJ dovish presionan al JPY a la baja.",
    "AUD": "Divisa de riesgo correlacionada con commodities (mineral de hierro, cobre, carbón) y con el ciclo económico chino. En entornos risk-off cae. El RBA hawkish y la inflación doméstica ofrecen soporte, pero la tensión global puede anularlo.",
    "CAD": "Correlacionada con el precio del petróleo WTI (Canadá es exportador neto). Un petróleo alto es estructuralmente positivo para el CAD, incluso si el BoC es dovish. El USMCA y el comercio con EEUU son el mayor riesgo de cola.",
    "CHF": "Activo refugio por excelencia: se aprecia en entornos de crisis, guerra o risk-off global. El SNB interviene para evitar apreciación excesiva. La inflación suiza muy baja históricamente limita el margen de subida de tasas.",
    "NZD": "Divisa de riesgo con alta sensibilidad a datos domésticos de NZ, ciclo chino y sentimiento global. En risk-off cae junto al AUD. El RBNZ y los datos de inflación/empleo NZ son los drivers fundamentales propios.",
}

# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista senior de mercados de divisas (FX) de una mesa institucional de trading con más de 15 años de experiencia. Tu especialidad es sintetizar múltiples fuentes de información en análisis accionables, precisos y con contexto macroeconómico profundo.

Tu tarea es leer una serie de titulares y resúmenes de noticias sobre una divisa específica y producir un análisis consolidado de alta calidad.

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

Cuando un evento macro global (guerra, petróleo, datos de empleo EEUU) afecte a esta divisa, DEBES explicar el mecanismo específico de transmisión para ESTA divisa en particular:

- NO escribas: "La tensión en Medio Oriente genera incertidumbre."
- SÍ escribe: "La escalada en Medio Oriente eleva los costes energéticos europeos, creando un dilema para el BCE entre contener inflación y evitar recesión, lo que limita el margen de maniobra hawkish del banco central y pesa sobre el EUR."

Si el mismo evento aparece en el análisis de varias divisas, el mecanismo de transmisión debe ser DIFERENTE y ESPECÍFICO para cada una.

═══════════════════════════════════════════════
REGLA CRÍTICA — SEÑALES CONTRADICTORIAS
═══════════════════════════════════════════════

Si hay señales alcistas Y bajistas en los titulares, NO las ignores ni las suavices. Debes:
1. Reconocer explícitamente la contradicción.
2. Explicar cuál señal domina actualmente y por qué.
3. Reflejar esto en el sentimiento: usa "mixed" si ninguna domina claramente.

PROHIBIDO: concluir "bull" en el sentimiento y mencionar en el análisis que el banco central es dovish sin explicar la contradicción.

═══════════════════════════════════════════════
REGLA DE DRIVERS
═══════════════════════════════════════════════

- Máximo 4 drivers, ordenados por importancia.
- Al menos 2 deben ser ESPECÍFICOS de esta divisa o su banco central.
- Un driver de contexto global solo es válido si incluye el mecanismo: "Petróleo >$80 → soporte CAD", "Risk-off → CHF refugio", "Guerra → costes energía zona euro".
- Formato: frases de 2-5 palabras con cifras cuando sea posible.
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
- mixed: señales contradictorias o equilibradas entre alcistas y bajistas
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
            headline  = a.get("ai_headline") or a.get("title", "")
            sentiment = a.get("sentiment", "neut").upper()
            source    = a.get("source", "")
            expand    = a.get("expand", "")
            snippet   = f" | {expand[:200].strip()}..." if expand and len(expand) > 30 else ""
            lines.append(f"  [{sentiment}] {headline}  ({source}){snippet}")

    if other:
        lines.append("── MEDIO/BAJO IMPACTO ──")
        for a in other:
            headline  = a.get("ai_headline") or a.get("title", "")
            sentiment = a.get("sentiment", "neut").upper()
            source    = a.get("source", "")
            lines.append(f"  [{sentiment}] {headline}  ({source})")

    headlines_block = "\n".join(lines) if lines else "Sin noticias disponibles."

    bull_count  = sum(1 for a in articles if a.get("sentiment") in ("bull", "alcista"))
    bear_count  = sum(1 for a in articles if a.get("sentiment") in ("bear", "bajista"))
    mixed_count = sum(1 for a in articles if a.get("sentiment") in ("mixed", "mixto"))
    neut_count  = len(articles) - bull_count - bear_count - mixed_count

    return (
        f"DIVISA: {cur} — {name}\n"
        f"PERFIL MACRO DE ESTA DIVISA: {macro_ctx}\n"
        f"DISTRIBUCIÓN: {len(articles)} artículos — "
        f"{bull_count} alcistas · {bear_count} bajistas · {mixed_count} mixtos · {neut_count} neutrales\n\n"
        f"TITULARES Y CONTEXTO:\n"
        f"{headlines_block}\n\n"
        f"INSTRUCCIONES ESPECÍFICAS PARA {cur}:\n"
        f"1. Usa el perfil macro para explicar cómo los eventos globales afectan ESPECÍFICAMENTE al {cur}.\n"
        f"2. Si hay señales contradictorias entre los titulares, identifícalas y explica cuál domina.\n"
        f"3. Los drivers deben ser específicos del {cur} — al menos 2 deben mencionar su banco central "
        f"o un indicador propio de su economía.\n\n"
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

            # v2.1: mínimo de palabras subido a 80
            word_count = len(parsed.get("analysis", "").split())
            if word_count < 80:
                print(f"    ⚠️  Análisis demasiado corto para {cur}: {word_count} palabras — reintentando")
                return None

            parsed["sentiment"]  = parsed["sentiment"].lower().strip()
            parsed["confidence"] = max(0, min(100, int(parsed["confidence"])))
            parsed["drivers"]    = [str(d)[:60] for d in parsed.get("drivers", [])[:4]]
            parsed["upcoming_event"] = parsed.get("upcoming_event") or None

            print(f"    📝 Análisis generado: {word_count} palabras")
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
    from collections import Counter

    sentiments = [a.get("sentiment", "neut") for a in articles]
    sent_count = Counter(sentiments)

    bull  = sent_count.get("bull",  0)
    bear  = sent_count.get("bear",  0)
    mixed = sent_count.get("mixed", 0)
    total = len(articles)

    if total == 0:
        dominant = "neut"; confidence = 0
    elif bull > bear and bull > mixed:
        dominant = "bull"; confidence = min(95, int((bull / total) * 100))
    elif bear > bull and bear > mixed:
        dominant = "bear"; confidence = min(95, int((bear / total) * 100))
    elif mixed > 0 or (bull > 0 and bear > 0):
        dominant = "mixed"; confidence = 50
    else:
        dominant = "neut"; confidence = 30

    name     = CURRENCY_NAMES.get(cur, cur)
    analysis = (
        f"Análisis basado en {total} {'noticia' if total == 1 else 'noticias'} "
        f"para {name}. "
        f"Señales: {bull} alcistas, {bear} bajistas, {mixed} mixtas. "
        f"Resumen AI no disponible en este ciclo."
    )
    sources = list(set(a.get("source", "") for a in articles if a.get("source")))[:3]

    return {
        "sentiment":      dominant,
        "confidence":     confidence,
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
    print(f"📊 generate_summaries.py v2.1 — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
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
        print(f"   {cur}: {n} artículo{'s' if n != 1 else ''}")

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
                if current_key_idx < len(all_keys):
                    print(f"  🔄 Usando Key {current_key_idx+1}")
                    result = call_groq(all_keys[current_key_idx], cur, cur_articles)
                    if result == "DAILY_LIMIT":
                        print(f"  ⛔ Key {current_key_idx+1} también agotada — fallback")
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
            sent_label = {"bull":"ALCISTA","bear":"BAJISTA","neut":"NEUTRAL","mixed":"MIXTO"}.get(
                fb["sentiment"], fb["sentiment"].upper()
            )
            print(f"  📊 Fallback estadístico → {sent_label} ({fb['confidence']}%)")

        if use_ai and cur != CURRENCIES[-1]:
            time.sleep(SLEEP_BETWEEN)

    SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "updated_utc":    now_utc.isoformat(),
        "updated_label":  now_utc.strftime("%H:%M UTC"),
        "model":          GROQ_MODEL if use_ai else "fallback",
        "total_ai":       generated,
        "total_fallback": fallbacks,
        "summaries":      summaries,
    }

    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"📋 RESUMEN FINAL")
    print(f"   ✅ Generados con AI:   {generated}")
    print(f"   📊 Fallback estadíst.: {fallbacks}")
    print(f"   💾 Guardado en:        {SUMMARIES_FILE}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
