#!/usr/bin/env python3
"""
enrich_news.py — v6
Soporte para múltiples API keys de Groq con fallback automático.

CAMBIOS v6:
  - Validación previa al envío a Groq: verifica que el artículo tenga
    contenido editorial real antes de pedirle un titular sintetizado
  - is_enrichable(): rechaza artículos sin descripción sustancial,
    eventos de calendario, o contenido claramente no apto para síntesis
  - MIN_CONTENT_WORDS: descripción debe tener al menos N palabras útiles
  - Descripción muy similar al título → se usa el título directamente sin Groq
  - Prompt mejorado con instrucción explícita de no inventar datos
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
# CONFIGURACIÓN
# ─────────────────────────────────────────────
NEWS_FILE             = Path("news-data/news.json")
GROQ_URL              = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL            = "llama-3.3-70b-versatile"
MAX_TOKENS            = 180
TEMPERATURE           = 0.2   # Más conservador: menos "creatividad", más fidelidad

# Groq free tier: ~30 req/min en llama-3.3-70b → 2s entre requests es seguro
# Con 35 artículos: 35 * 4s = ~2.3 min en lugar de 35 * 12s = ~7 min
SLEEP_BETWEEN         = 4    # Reducido de 12s → 4s (Groq soporta ~15 req/min con holgura)
MAX_RETRIES           = 2
DEFAULT_RETRY_WAIT    = 65
SKIP_IF_WAIT_EXCEEDS  = 90
GLOBAL_TIMEOUT_MIN    = 14   # Aumentado: fetch ~1.5min + enrich ~2.5min + margen

# Contenido mínimo que debe tener la descripción para justificar enriquecimiento
MIN_CONTENT_WORDS = 15

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

# Patrones que indican contenido de calendario / sin valor analítico
CALENDAR_RE = re.compile(
    r"upcoming event|content type|scheduled date|share this page|"
    r"governing council presents|press release explaining|"
    r"announces the setting for the overnight|on eight scheduled|"
    r"monetary policy report$|rate announcement$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Eres un analista senior de mercados forex que redacta titulares sintéticos para una plataforma profesional de trading.

TAREA: Dado el título y descripción de una noticia económica o financiera REAL, genera un titular de UNA SOLA LÍNEA.

REGLAS ABSOLUTAS — NUNCA VIOLAR:
1. SOLO usa información que aparezca EXPLÍCITAMENTE en el título o descripción proporcionados
2. NUNCA inventes cifras, porcentajes, fechas o datos que no estén en el texto original
3. NUNCA supongas el resultado de una reunión futura o evento que aún no ha ocurrido
4. Si el texto no tiene información analítica suficiente, responde solo: IRRELEVANTE

FORMATO DEL TITULAR:
- Una sola línea, sin punto final
- Entre 90 y 150 caracteres
- En español, tono profesional y analítico
- Sin markdown, sin comillas, sin emojis, sin introducción
- Empieza por el dato/evento concreto (no por "El" o "La")
- Incluye el impacto en la divisa: alcista / bajista / neutro / mixto

EJEMPLOS CORRECTOS (basados solo en lo que dice el artículo):
- IPC EEUU sube 3.2% en enero superando estimaciones: presión hawkish sobre Fed, soporte en USD
- BoC mantiene tasas en 3% con sesgo dovish: señal de recorte próximo pesa sobre CAD
- PMI manufacturero zona euro cae a 44.2: contracción del sector industrial presiona EUR

EJEMPLOS INCORRECTOS (inventan datos no presentes):
- "BoC mantiene tasas en 4.5%" ← si el artículo no menciona el 4.5%
- "Fed subirá tasas en marzo" ← si el artículo no lo afirma
- Cualquier cifra que no aparezca textualmente en título o descripción

CASOS ESPECIALES:
- Análisis técnico con niveles: incluir el nivel clave y sesgo
- Pares de divisas (GBP/USD, EUR/USD): siempre relevantes
- Impacto ambiguo: usar "mixto" o "señal contradictoria"
- Sin suficiente info analítica → responde solo: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur  = article.get("cur", "USD")
    lang = article.get("lang", "en")
    return (
        f"DIVISA PRINCIPAL AFECTADA: {cur} ({CURRENCY_NAMES.get(cur, cur)})\n"
        f"FUENTE: {article.get('source', '')} ({'español' if lang == 'es' else 'inglés'})\n"
        f"TÍTULO ORIGINAL: {article.get('title', '')}\n"
        f"DESCRIPCIÓN: {(article.get('expand', '') or '')[:400] or 'Sin descripción'}\n\n"
        f"IMPORTANTE: Genera el titular SOLO con información presente en el texto anterior. "
        f"Si no hay información analítica suficiente, responde únicamente: IRRELEVANTE\n\n"
        f"Titular sintético:"
    )


def is_enrichable(article: dict) -> tuple[bool, str]:
    """
    Verifica si un artículo tiene suficiente contenido real para ser enriquecido por IA.
    Retorna (True, "") si es apto, o (False, razón) si debe saltarse.
    """
    title   = (article.get("title", "") or "").strip()
    expand  = (article.get("expand", "") or "").strip()

    # 1. Descripción demasiado corta para síntesis real
    word_count = len(expand.split())
    if word_count < MIN_CONTENT_WORDS:
        return False, f"descripción muy corta ({word_count} palabras)"

    # 2. Detectar contenido de calendario que pasó el primer filtro
    combined = (title + " " + expand).lower()
    if CALENDAR_RE.search(combined):
        return False, "contenido de calendario/agenda"

    # 3. La descripción es casi idéntica al título (sin valor añadido para síntesis)
    title_lower  = title.lower()[:50]
    expand_lower = expand.lower()
    if expand_lower.startswith(title_lower) and word_count < 25:
        return False, "descripción idéntica al título"

    # 4. Descripción es solo metadatos o encabezados de sección
    meta_patterns = [
        r"^\s*(read more|continue reading|click here|leer más|ver más)\b",
        r"^\s*[\[\(].*[\]\)]\s*$",
        r"^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*$",
    ]
    for pat in meta_patterns:
        if re.match(pat, expand, re.IGNORECASE):
            return False, "descripción es solo metadatos"

    return True, ""


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
        is_daily     = any(x in msg for x in ("per day", "tpd", "tokens per day"))
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
    """
    Retorna:
      str           → titular generado
      None          → irrelevante o error no recuperable
      "DAILY_LIMIT" → límite diario confirmado
    """
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
                print(f"  ⏳ Rate limit temporal (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                return "DAILY_LIMIT"

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            if content.upper() == "IRRELEVANTE":
                return None

            # Limpiar artefactos de formato
            content = content.strip('"\'').strip()
            # Eliminar prefijos que el modelo a veces añade
            for prefix in ("titular:", "headline:", "titolo:", "síntesis:", "análisis:"):
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


FOREX_TERMS = [
    "usd","eur","gbp","jpy","aud","cad","chf","nzd","dollar","euro",
    "pound","yen","franc","/usd","/eur","forex","fx ","rate","fed",
    "ecb","boe","gdp","cpi","inflation","inflación",
]


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}  |  Timeout: {GLOBAL_TIMEOUT_MIN} min")
    print("=" * 60)

    # ── Cargar keys ───────────────────────────────────────────────────────────
    all_keys = load_api_keys()
    if not all_keys:
        raise EnvironmentError("❌ No se encontró ninguna GROQ_API_KEY.")

    print(f"\n🔑 Keys configuradas: {len(all_keys)}")
    available_keys = []
    for i, key in enumerate(all_keys, 1):
        status = check_key(key)
        label  = f"Key {i} ({mask_key(key)})"
        if status == "daily_limit":
            print(f"  ⛔ {label} — límite diario de tokens confirmado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            print(f"  ✅ {label} — disponible")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys confirmadas agotadas. Saliendo.")
        sys.exit(0)

    print(f"\n✅ {len(available_keys)} key(s) disponible(s)\n")
    current_key_idx = 0

    # ── Cargar news.json ──────────────────────────────────────────────────────
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

        # ── Validación previa: ¿vale la pena enviar a Groq? ──────────────────
        ok, reason = is_enrichable(article)
        if not ok:
            not_enrichable += 1
            print(f"  ⛔ No enriquecible: {reason} — omitiendo")
            continue

        result = call_groq(available_keys[current_key_idx], article)

        # ── Fallback a siguiente key ───────────────────────────────────────────
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
            enriched += 1
            print(f"  ✅ {result[:95]}{'...' if len(result) > 95 else ''}")
        elif result is None:
            txt = article.get("title","").lower() + " " + (article.get("expand","") or "").lower()
            if any(t in txt for t in FOREX_TERMS):
                skipped += 1
                print("  ⚠️  Sin síntesis (posible error de API)")
            else:
                irrelevant += 1
                print("  ⏭️  Sin síntesis (irrelevante)")

        # Pausa adaptativa: SLEEP_BETWEEN base, pero si la llamada fue rápida
        # podemos ser más agresivos; si hubo rate limit ya esperamos dentro de call_groq
        if idx < len(to_process) - 1 and not timeout_reached():
            time.sleep(SLEEP_BETWEEN)

    save_progress(data, articles, now_utc)

    print()
    print("=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Enriquecidos:        {enriched}")
    print(f"   ⛔ No enriquecibles:    {not_enrichable}")
    print(f"   ⚠️  Error API:          {skipped}")
    print(f"   ⏭️  Irrelevantes:       {irrelevant}")
    if stopped_early:
        print(f"   ⏰ Detenido antes de completar")
    print(f"   🔑 Keys usadas:        hasta Key {current_key_idx + 1} de {len(available_keys)}")
    print(f"   ⏱️  Tiempo total:       {elapsed_min():.1f} min")
    print(f"   💾 Guardado en:        {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
