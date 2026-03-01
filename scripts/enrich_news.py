#!/usr/bin/env python3
"""
enrich_news.py — v3
Lee news-data/news.json, llama a Groq API para generar titulares AI.

PROTECCIONES ANTI-CUELGUE:
  - check_groq_quota(): prueba antes de empezar, detecta límite diario vs por minuto
  - GLOBAL_TIMEOUT_MIN: para y guarda tras N minutos pase lo que pase
  - SKIP_IF_WAIT_EXCEEDS: si Retry-After > N seg, omite artículo en vez de esperar
  - Sentinel "DAILY_LIMIT": para inmediatamente si Groq confirma límite diario
  - save_progress(): guarda siempre, aunque sea parcial
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

SLEEP_BETWEEN         = 12   # segundos entre llamadas (5 req/min << límite 14 req/min)
MAX_RETRIES           = 2    # reintentos por artículo tras 429
DEFAULT_RETRY_WAIT    = 65   # segundos de espera si Groq no da Retry-After
SKIP_IF_WAIT_EXCEEDS  = 90   # si Retry-After > esto, omitir artículo
GLOBAL_TIMEOUT_MIN    = 8    # minutos: el script para y guarda aunque no termine

_START_TIME = time.time()

def elapsed_min() -> float:
    return (time.time() - _START_TIME) / 60.0

def timeout_reached() -> bool:
    return elapsed_min() >= GLOBAL_TIMEOUT_MIN

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
2. Explica brevemente el mecanismo o contexto (ej: subida de PPI → temores estanflación, tensión geopolítica → aversión al riesgo)
3. Indica el impacto esperado en la divisa indicada (alcista / bajista / neutro / mixto)

FORMATO OBLIGATORIO:
- Una sola línea, sin punto final
- Entre 100 y 150 caracteres
- En español, tono profesional y analítico
- Sin markdown, sin comillas, sin emojis, sin introducción
- Nunca empieces con "El" o "La" si puedes evitarlo — empieza por el dato/evento

EJEMPLOS DE BUENOS TITULARES:
- IPC EEUU sube 3.2% en enero superando estimaciones: presión hawkish sobre Fed mantiene soporte en USD
- BoC mantiene tasas en 3% pero señala recorte en abril: orientación dovish pesa sobre CAD a medio plazo
- PMI manufacturero zona euro cae a 44.2: contracción profunda del sector industrial → presión bajista EUR
- Nóminas EEUU: 256K empleos vs 160K esperado, sorpresa alcista refuerza narrativa de excepción americana → USD
- GBP/USD cae por escalada en Oriente Próximo: aversión al riesgo favorece activos refugio, presión bajista GBP
- AUD/USD mantiene alzas pese a retroceso USD: resistencia técnica sugiere sesgo alcista AUD a corto plazo
- RBA sube tasas 25pb a 4.35%: ciclo restrictivo continúa respaldando demanda de AUD en el corto plazo

REGLAS:
- Pares de divisas (GBP/USD, AUD/USD, etc.) SIEMPRE son relevantes
- Para análisis técnicos incluye nivel clave o rango y el sesgo resultante
- Si impacto ambiguo: "mixto" o "señal contradictoria"
- Si la noticia es completamente irrelevante para forex (moda, deportes, cultura): responde solo IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur  = article.get("cur", "USD")
    lang = article.get("lang", "en")
    return (
        f"DIVISA PRINCIPAL AFECTADA: {cur} ({CURRENCY_NAMES.get(cur, cur)})\n"
        f"FUENTE: {article.get('source', '')} (artículo en {'español' if lang == 'es' else 'inglés'})\n"
        f"TÍTULO ORIGINAL: {article.get('title', '')}\n"
        f"DESCRIPCIÓN: {(article.get('expand', '') or '')[:400] or 'Sin descripción'}\n\n"
        f"Nota: Si el artículo menciona un par de divisas como GBP/USD o AUD/USD, "
        f"es SIEMPRE relevante. Analiza el movimiento del par y genera el titular.\n\n"
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


def check_groq_quota(api_key: str) -> bool:
    """Llamada de prueba (1 token) para detectar límite diario antes de empezar."""
    try:
        r = requests.post(
            GROQ_URL,
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 401:
            print("  ❌ GROQ_API_KEY inválida (401)")
            return False
        if r.status_code == 429:
            wait = get_retry_after(r)
            body = {}
            try:
                body = r.json()
            except Exception:
                pass
            error_msg = str(body.get("error", {}).get("message", "")).lower()
            print(f"  ⚠️  Groq 429 en check inicial — wait sugerido: {wait}s")
            print(f"      Mensaje: {body.get('error', {}).get('message', 'N/A')}")
            if any(x in error_msg for x in ("day", "daily", "exceeded", "limit")):
                print("  ❌ LÍMITE DIARIO DE TOKENS — no se puede enriquecer hoy")
                return False
            # Es rate limit por minuto, no diario → se puede continuar con espera
            print("  ℹ️  Rate limit por minuto (no diario) — continuando...")
            time.sleep(min(wait, 30))
            return True
        return True  # 200 o cualquier otro código → OK
    except Exception as e:
        print(f"  ⚠️  No se pudo verificar Groq ({e}) — asumiendo OK")
        return True


def call_groq(api_key: str, article: dict) -> str | None:
    """
    Devuelve:
      - str: titular generado
      - None: artículo irrelevante o error no recuperable
      - "DAILY_LIMIT": límite diario alcanzado → parar todo
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
                wait = get_retry_after(r)
                body = {}
                try:
                    body = r.json()
                except Exception:
                    pass
                error_msg = str(body.get("error", {}).get("message", "")).lower()

                # Detectar límite diario
                if any(x in error_msg for x in ("day", "daily", "exceeded")):
                    print(f"\n  🛑 LÍMITE DIARIO: {body.get('error', {}).get('message', 'N/A')}")
                    return "DAILY_LIMIT"

                # Retry-After demasiado largo → omitir artículo
                if wait > SKIP_IF_WAIT_EXCEEDS:
                    print(f"  ⏭️  Retry-After={wait}s > {SKIP_IF_WAIT_EXCEEDS}s — omitiendo")
                    return None

                print(f"  ⏳ Rate limit (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                raise RuntimeError("GROQ_API_KEY inválida.")

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
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            return None

    print("  ❌ Reintentos agotados — omitiendo")
    return None


def save_progress(data: dict, articles: list, now_utc: datetime) -> None:
    data["articles"]       = articles
    data["ai_enriched_at"] = now_utc.isoformat()
    data["ai_model"]       = GROQ_MODEL
    with open(NEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


FOREX_TERMS = ["usd","eur","gbp","jpy","aud","cad","chf","nzd","dollar","euro",
               "pound","yen","franc","/usd","/eur","forex","fx ","rate","fed",
               "ecb","boe","gdp","cpi","inflation","inflación"]


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}  |  Timeout: {GLOBAL_TIMEOUT_MIN} min")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ GROQ_API_KEY no configurada.")
    print(f"✅ GROQ_API_KEY configurada ({len(api_key)} chars)")

    print("\n🔎 Verificando estado de Groq API...")
    if not check_groq_quota(api_key):
        print("⛔ Groq no disponible. Saliendo sin modificar news.json.")
        sys.exit(0)
    print("✅ Groq API operativa\n")

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
            print(f"\n⏰ Timeout {GLOBAL_TIMEOUT_MIN} min tras {idx} artículos — guardando parcial")
            stopped_early = True
            break

        cur = article.get("cur", "?")
        imp = article.get("impact", "?")
        print(f"[{idx+1}/{len(to_process)}] {cur} [{imp}] {article.get('title','')[:60]}...")

        result = call_groq(api_key, article)

        if result == "DAILY_LIMIT":
            print(f"⛔ Límite diario — guardando {enriched} enriquecidos")
            stopped_early = True
            break
        elif result:
            articles[article_idx]["ai_headline"] = result
            enriched += 1
            print(f"  ✅ {result[:90]}{'...' if len(result) > 90 else ''}")
        else:
            txt = article.get("title","").lower() + " " + (article.get("expand","") or "").lower()
            if any(t in txt for t in FOREX_TERMS):
                skipped += 1
                print("  ⚠️  Sin síntesis (posible error de API)")
            else:
                irrelevant += 1
                print("  ⏭️  Sin síntesis (irrelevante para forex)")

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
    print(f"   ⏱️  Tiempo total:  {elapsed_min():.1f} min")
    print(f"   💾 Guardado en:   {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
