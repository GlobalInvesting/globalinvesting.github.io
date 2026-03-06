#!/usr/bin/env python3
"""
generate_summaries.py — v2.0
Genera un bloque de análisis consolidado por divisa a partir de news.json.
Llama a Groq una vez por divisa (8 llamadas total) y escribe summaries.json.

Ejecutar DESPUÉS de enrich_news.py (requiere ai_headline ya generados).
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
MAX_TOKENS     = 800   # aumentado para análisis más ricos
TEMPERATURE    = 0.3   # ligeramente más creativo
SLEEP_BETWEEN  = 6     # segundos entre llamadas
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

FLAG_CODES = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "CHF": "ch", "NZD": "nz",
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
1. Contexto macro principal: ¿qué evento o dato está moviendo la divisa hoy?
2. Niveles técnicos clave: menciona niveles de precio, soportes, resistencias o rangos cuando aparezcan en los titulares (ej: "EUR/USD cotiza en 1.1590, con soporte en 1.1540").
3. Postura del banco central relevante: hawkish / dovish / en espera, y qué implica para la divisa.
4. Catalizadores secundarios: flujos de riesgo (risk-on/off), correlaciones con materias primas, datos de empleo, etc.
5. Perspectiva a corto plazo: ¿qué evento o dato próximo puede cambiar el sesgo?

ESTILO:
- Cita cifras concretas cuando aparezcan: porcentajes de caída, niveles de precio, variaciones en puntos básicos.
- Conecta causas con efectos (ej: "el alza del petróleo amplifica las expectativas de inflación, lo que reduce el margen del BCE para recortar tasas").
- Si hay señales contradictorias, reconócelas explícitamente y explica cuál domina y por qué.
- No repitas información: si el EUR cayó a 1.1590, mencionarlo una vez con contexto es suficiente.

LO QUE DEBES EVITAR:
- Frases vacías como "la divisa muestra señales" o "los mercados reaccionaron".
- Análisis de menos de 3 oraciones sustanciales.
- Repetir el nombre de la divisa al inicio de cada oración.
- Listar hechos sin conectarlos causalmente.

═══════════════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════════════

JSON puro, sin markdown, sin texto extra:

{
  "sentiment": "bull|bear|neut|mixed",
  "confidence": 0-100,
  "analysis": "análisis completo entre 120 y 200 palabras",
  "drivers": ["Driver 1", "Driver 2", "Driver 3", "Driver 4"],
  "upcoming_event": "Descripción breve del evento próximo más relevante, o null"
}

REGLAS DE SENTIMIENTO:
- bull:  señales predominantemente positivas para la divisa (>60% de titulares alcistas)
- bear:  señales predominantemente negativas para la divisa (>60% de titulares bajistas)
- mixed: señales contradictorias o equilibradas entre alcistas y bajistas
- neut:  sin señal clara o cobertura muy limitada (<2 noticias)

REGLA DE CONFIDENCE (refleja consenso entre las fuentes, no la intensidad del movimiento):
- 90-100: consenso absoluto, todas las fuentes apuntan al mismo lado
- 70-89:  mayoría clara con algún matiz menor
- 50-69:  tendencia moderada con contradicciones relevantes
- 30-49:  señal débil o muy mixta
- 0-29:   sin señal (usar neut)

REGLA DE DRIVERS (máx. 4, ordenados por importancia):
- Frases de 2-4 palabras, específicas y concretas
- Ejemplos buenos: "Petróleo >$80", "BoE hawkish", "NFP positivo", "Risk-off global"
- Ejemplos malos: "Tensión", "Datos", "Mercado"

REGLA DE UPCOMING_EVENT:
- Solo incluir si aparece explícitamente en los titulares o es la consecuencia directa de la noticia principal.
- Formato: "Nombre del evento — fecha/plazo si se menciona"
- Si no hay ninguno relevante: null"""


def build_user_prompt(cur: str, articles: list) -> str:
    name = CURRENCY_NAMES.get(cur, cur)

    # Separar por impacto para dar más peso a los de alto impacto
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
            # Incluir el resumen expandido si existe, truncado a 200 chars
            snippet = f" | {expand[:200].strip()}..." if expand and len(expand) > 30 else ""
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

    sentiment_summary = (
        f"Distribución de sentimiento: {bull_count} alcistas, "
        f"{bear_count} bajistas, {mixed_count} mixtos, {neut_count} neutrales."
    )

    return (
        f"DIVISA: {cur} — {name}\n"
        f"TOTAL DE ARTÍCULOS: {len(articles)}\n"
        f"{sentiment_summary}\n\n"
        f"TITULARES Y CONTEXTO:\n"
        f"{headlines_block}\n\n"
        f"Genera el análisis consolidado institucional en JSON para {cur}. "
        f"Recuerda: el campo 'analysis' debe tener entre 120 y 200 palabras, "
        f"con niveles de precio concretos, postura del banco central y perspectiva a corto plazo."
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

            # Limpiar posibles backticks de markdown
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"```\s*$",    "", raw).strip()

            parsed = json.loads(raw)

            # Validar campos obligatorios
            required = {"sentiment", "confidence", "analysis", "drivers"}
            if not required.issubset(parsed.keys()):
                print(f"    ⚠️  JSON incompleto para {cur}: {list(parsed.keys())}")
                return None

            # Validar extensión del análisis
            word_count = len(parsed.get("analysis", "").split())
            if word_count < 50:
                print(f"    ⚠️  Análisis demasiado corto para {cur}: {word_count} palabras")
                return None

            # Normalizar
            parsed["sentiment"]  = parsed["sentiment"].lower().strip()
            parsed["confidence"] = max(0, min(100, int(parsed["confidence"])))
            parsed["drivers"]    = [str(d)[:50] for d in parsed.get("drivers", [])[:4]]
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
    """Genera un resumen básico sin AI cuando Groq no está disponible."""
    from collections import Counter

    sentiments = [a.get("sentiment", "neut") for a in articles]
    sent_count = Counter(sentiments)

    bull  = sent_count.get("bull",  0)
    bear  = sent_count.get("bear",  0)
    mixed = sent_count.get("mixed", 0)
    total = len(articles)

    if total == 0:
        dominant = "neut"
        confidence = 0
    elif bull > bear and bull > mixed:
        dominant   = "bull"
        confidence = min(95, int((bull / total) * 100))
    elif bear > bull and bear > mixed:
        dominant   = "bear"
        confidence = min(95, int((bear / total) * 100))
    elif mixed > 0 or (bull > 0 and bear > 0):
        dominant   = "mixed"
        confidence = 50
    else:
        dominant   = "neut"
        confidence = 30

    name = CURRENCY_NAMES.get(cur, cur)
    analysis = (
        f"Análisis basado en {total} {'noticia' if total == 1 else 'noticias'} "
        f"para {name}. "
        f"Señales: {bull} alcistas, {bear} bajistas, {mixed} mixtas. "
        f"Resumen AI no disponible en este ciclo."
    )

    sources = list(set(a.get("source", "") for a in articles if a.get("source")))[:3]

    return {
        "sentiment":       dominant,
        "confidence":      confidence,
        "analysis":        analysis,
        "drivers":         [],
        "upcoming_event":  None,
        "ai_generated":    False,
        "sources":         sources,
        "articles_count":  total,
        "latest_ts":       max((a.get("ts", 0) for a in articles), default=0),
    }


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 65)
    print(f"📊 generate_summaries.py v2.0 — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    # Cargar API keys
    all_keys = load_api_keys()
    if not all_keys:
        print("⚠️  Sin GROQ_API_KEY — generando resúmenes con fallback estadístico")
    else:
        print(f"🔑 Keys disponibles: {len(all_keys)}")
        for i, k in enumerate(all_keys, 1):
            print(f"   Key {i}: {mask_key(k)}")

    current_key_idx = 0
    use_ai = bool(all_keys)

    # Cargar news.json
    if not NEWS_FILE.exists():
        print(f"❌ No se encontró {NEWS_FILE}")
        sys.exit(1)

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    articles = news_data.get("articles", [])
    print(f"\n📰 Artículos cargados: {len(articles)}")

    # Agrupar por divisa
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

    # Cargar summaries previos para no perder datos si falla
    existing_summaries = {}
    if SUMMARIES_FILE.exists():
        try:
            with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_summaries = existing_data.get("summaries", {})
        except Exception:
            pass

    summaries = {}
    generated  = 0
    fallbacks  = 0

    print(f"\n{'─'*65}")
    for cur in CURRENCIES:
        cur_articles = groups[cur]
        n = len(cur_articles)
        print(f"\n[{cur}] {CURRENCY_NAMES[cur]} — {n} artículo{'s' if n!=1 else ''}")

        # Sin artículos → fallback vacío
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

        # Intentar con AI
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
                        use_ai  = False
                        result  = None

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
            # Fallback estadístico
            fb = fallback_summary(cur, cur_articles)
            summaries[cur] = fb
            fallbacks += 1
            sent_label = {"bull":"ALCISTA","bear":"BAJISTA","neut":"NEUTRAL","mixed":"MIXTO"}.get(
                fb["sentiment"], fb["sentiment"].upper()
            )
            print(f"  📊 Fallback estadístico → {sent_label} ({fb['confidence']}%)")

        # Pausa entre llamadas AI
        if use_ai and cur != CURRENCIES[-1]:
            time.sleep(SLEEP_BETWEEN)

    # Guardar summaries.json
    SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "updated_utc":   now_utc.isoformat(),
        "updated_label": now_utc.strftime("%H:%M UTC"),
        "model":         GROQ_MODEL if use_ai else "fallback",
        "total_ai":      generated,
        "total_fallback":fallbacks,
        "summaries":     summaries,
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
