#!/usr/bin/env python3
"""
enrich_news.py — v4
Soporte para múltiples API keys de Groq con fallback automático.
Si la key principal alcanza el límite diario, cambia automáticamente a la siguiente.

Configura en GitHub Actions Secrets:
  GROQ_API_KEY   → key principal
  GROQ_API_KEY_2 → key de respaldo (opcional)
"""

import os
import sys
import json
import time
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
TEMPERATURE           = 0.25

SLEEP_BETWEEN         = 12
MAX_RETRIES           = 2
DEFAULT_RETRY_WAIT    = 65
SKIP_IF_WAIT_EXCEEDS  = 90
GLOBAL_TIMEOUT_MIN    = 10  # un poco más holgado con 2 keys

_START_TIME = time.time()

def elapsed_min():
    return (time.time() - _START_TIME) / 60.0

def timeout_reached():
    return elapsed_min() >= GLOBAL_TIMEOUT_MIN

# ─────────────────────────────────────────────
# MÚLTIPLES API KEYS
# ─────────────────────────────────────────────
def load_api_keys() -> list[str]:
    """Carga todas las keys disponibles en orden de prioridad."""
    keys = []
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = os.environ.get(var, "").strip()
        if k:
            keys.append(k)
    return keys


def mask_key(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"

# ─────────────────────────────────────────────
CURRENCY_NAMES = {
    "USD": "dólar estadounidense", "EUR": "euro",
    "GBP": "libra esterlina",      "JPY": "yen japonés",
    "AUD": "dólar australiano",    "CAD": "dólar canadiense",
    "CHF": "franco suizo",         "NZD": "dólar neozelandés",
}

SYSTEM_PROMPT = """Eres un analista senior de mercados forex que redacta titulares sintéticos para una plataforma profesional de trading.

TAREA: Dado el título y descripción de una noticia económica o financiera, genera un titular de UNA SOLA LÍNEA que:
1. Describe el evento o dato concreto con precisión (incluye cifras si están disponibles)
2. Explica brevemente el mecanismo o contexto (ej: subida de PPI → temores estanflación)
3. Indica el impacto esperado en la divisa indicada (alcista / bajista / neutro / mixto)

FORMATO OBLIGATORIO:
- Una sola línea, sin punto final
- Entre 100 y 150 caracteres
- En español, tono profesional y analítico
- Sin markdown, sin comillas, sin emojis, sin introducción
- Nunca empieces con "El" o "La" — empieza por el dato/evento

EJEMPLOS:
- IPC EEUU sube 3.2% en enero superando estimaciones: presión hawkish sobre Fed mantiene soporte en USD
- BoC mantiene tasas en 3% pero señala recorte en abril: orientación dovish pesa sobre CAD a medio plazo
- PMI manufacturero zona euro cae a 44.2: contracción profunda del sector industrial → presión bajista EUR
- GBP/USD cae por escalada en Oriente Próximo: aversión al riesgo favorece activos refugio, presión bajista GBP
- RBA sube tasas 25pb a 4.35%: ciclo restrictivo continúa respaldando demanda de AUD en el corto plazo

REGLAS:
- Pares de divisas (GBP/USD, AUD/USD, etc.) SIEMPRE son relevantes
- Para análisis técnicos incluye nivel clave o rango y el sesgo resultante
- Si impacto ambiguo: "mixto" o "señal contradictoria"
- Si completamente irrelevante para forex (moda, deportes, cultura): responde solo IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur  = article.get("cur", "USD")
    lang = article.get("lang", "en")
    return (
        f"DIVISA PRINCIPAL AFECTADA: {cur} ({CURRENCY_NAMES.get(cur, cur)})\n"
        f"FUENTE: {article.get('source', '')} ({'español' if lang == 'es' else 'inglés'})\n"
        f"TÍTULO ORIGINAL: {article.get('title', '')}\n"
        f"DESCRIPCIÓN: {(article.get('expand', '') or '')[:400] or 'Sin descripción'}\n\n"
        f"Genera el titular sintético de impacto:"
    )


def get_retry_after(response) -> int:
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        val = response.headers.get(header)
        if val:
            try:
                return min(int(float(str(val).lower().replace("s", "").strip())) + 2, 180)
            except (ValueError, TypeError):
                pass
    return DEFAULT_RETRY_WAIT


def is_daily_limit(response) -> bool:
    """Detecta si el 429 es por límite diario (no por minuto)."""
    try:
        body = response.json()
        msg  = str(body.get("error", {}).get("message", "")).lower()
        return any(x in msg for x in ("per day", "daily", "tokens per day", "tpd"))
    except Exception:
        return False


def check_key(api_key: str) -> str:
    """
    Verifica una key. Devuelve:
      'ok'          → key funciona
      'daily_limit' → límite diario agotado
      'invalid'     → key inválida (401)
      'error'       → otro error (asumir OK)
    """
    try:
        r = requests.post(
            GROQ_URL,
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 401:
            return "invalid"
        if r.status_code == 429:
            if is_daily_limit(r):
                return "daily_limit"
            return "ok"  # rate limit por minuto, no diario
        return "ok"
    except Exception:
        return "error"


def call_groq(api_key: str, article: dict) -> str | None:
    """
    Devuelve:
      str           → titular generado
      None          → irrelevante o error no recuperable
      "DAILY_LIMIT" → límite diario → cambiar a siguiente key
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)

            if r.status_code == 429:
                if is_daily_limit(r):
                    return "DAILY_LIMIT"
                wait = get_retry_after(r)
                if wait > SKIP_IF_WAIT_EXCEEDS:
                    print(f"  ⏭️  Retry-After={wait}s > {SKIP_IF_WAIT_EXCEEDS}s — omitiendo")
                    return None
                print(f"  ⏳ Rate limit (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                return "DAILY_LIMIT"  # tratar como no disponible, pasar a siguiente key

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            if content.upper() == "IRRELEVANTE":
                return None

            content = content.strip('"\'')
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

    # ── Cargar y verificar todas las keys disponibles ─────────────────────────
    all_keys = load_api_keys()
    if not all_keys:
        raise EnvironmentError("❌ No se encontró ninguna GROQ_API_KEY.")

    print(f"\n🔑 Keys disponibles: {len(all_keys)}")
    available_keys = []
    for i, key in enumerate(all_keys, 1):
        status = check_key(key)
        label  = f"Key {i} ({mask_key(key)})"
        if status == "ok":
            print(f"  ✅ {label} — operativa")
            available_keys.append(key)
        elif status == "daily_limit":
            print(f"  ⛔ {label} — límite diario agotado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            print(f"  ⚠️  {label} — error de red, asumiendo OK")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys han alcanzado el límite diario. Saliendo.")
        sys.exit(0)

    print(f"\n✅ Keys disponibles para usar: {len(available_keys)}")
    current_key_idx = 0
    current_key     = available_keys[current_key_idx]
    print(f"   Usando: Key con índice {current_key_idx + 1}\n")

    # ── Cargar news.json ──────────────────────────────────────────────────────
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró {NEWS_FILE}.")

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        print("⚠️  Sin artículos. Nada que procesar.")
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
    print(f"⏱️  Estimado: ~{est_time // 60}min {est_time % 60}s  |  tokens: ~{len(to_process) * 420:,}\n")

    enriched = skipped = irrelevant = 0
    stopped_early = False

    for idx, (article_idx, article) in enumerate(to_process):

        if timeout_reached():
            print(f"\n⏰ Timeout {GLOBAL_TIMEOUT_MIN} min — guardando parcial")
            stopped_early = True
            break

        cur = article.get("cur", "?")
        imp = article.get("impact", "?")
        print(f"[{idx+1}/{len(to_process)}] {cur} [{imp}] {article.get('title','')[:60]}...")

        result = call_groq(current_key, article)

        # ── Si la key actual se agotó, intentar con la siguiente ──────────────
        if result == "DAILY_LIMIT":
            print(f"  ⛔ Key {current_key_idx + 1} agotada — buscando siguiente key...")
            current_key_idx += 1
            if current_key_idx >= len(available_keys):
                print("  ⛔ Todas las keys agotadas — guardando progreso")
                stopped_early = True
                break
            current_key = available_keys[current_key_idx]
            print(f"  🔄 Cambiando a Key {current_key_idx + 1} ({mask_key(current_key)})")
            # Reintentar el mismo artículo con la nueva key
            result = call_groq(current_key, article)
            if result == "DAILY_LIMIT":
                print("  ⛔ Nueva key también agotada — deteniendo")
                stopped_early = True
                break

        if result and result != "DAILY_LIMIT":
            articles[article_idx]["ai_headline"] = result
            enriched += 1
            print(f"  ✅ {result[:90]}{'...' if len(result) > 90 else ''}")
        elif result is None:
            txt = article.get("title","").lower() + " " + (article.get("expand","") or "").lower()
            if any(t in txt for t in FOREX_TERMS):
                skipped += 1
                print("  ⚠️  Sin síntesis (posible error de API)")
            else:
                irrelevant += 1
                print("  ⏭️  Sin síntesis (irrelevante)")

        if idx < len(to_process) - 1:
            time.sleep(SLEEP_BETWEEN)

    save_progress(data, articles, now_utc)

    print()
    print("=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Enriquecidos:  {enriched}")
    print(f"   ⚠️  Error API:     {skipped}")
    print(f"   ⏭️  Irrelevantes:  {irrelevant}")
    if stopped_early:
        print(f"   ⏰ Detenido antes de completar")
    print(f"   🔑 Keys usadas:   {current_key_idx + 1} de {len(available_keys)}")
    print(f"   ⏱️  Tiempo total:  {elapsed_min():.1f} min")
    print(f"   💾 Guardado en:   {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
