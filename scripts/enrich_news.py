#!/usr/bin/env python3
"""
enrich_news.py
Lee news-data/news.json (generado por fetch_news.py), llama a Groq API para
generar un titular sintético de impacto para cada artículo sin `ai_headline`,
y sobreescribe el JSON preservando todos los campos existentes.

Corre 2 veces al día via GitHub Actions (8:00 y 20:00 UTC).
Procesa TODOS los artículos (hasta 60) sin filtrar por impacto.

Modelo: llama-3.3-70b-versatile (Groq free tier)
Estimación de tokens: ~25.200 por ejecución × 2 = ~50.400/día (10% del límite de 500K)
Tiempo de ejecución: ~10 min para 60 artículos (10s entre llamadas, sin rate limits)
  → SLEEP_BETWEEN = 10s evita los 429 de raíz (6K tokens/min ÷ 420 tok/req = 14 req/min máx)
  → Si igual ocurre un 429, se lee el header Retry-After de Groq y se espera exactamente eso

Campos que lee de news.json (generados por fetch_news.py):
  - title    → título original
  - expand   → descripción corta (hasta 300 chars)
  - cur      → divisa detectada ('USD', 'EUR', etc.)
  - impact   → nivel de impacto ('high' | 'med' | 'low')
  - source   → nombre de la fuente
  - lang     → idioma ('es' | 'en')

Campo que agrega este script:
  - ai_headline → titular sintético en español generado por Groq
"""

import os
import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
NEWS_FILE      = Path("news-data/news.json")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 180       # Suficiente para titulares elaborados (~150 chars)
TEMPERATURE    = 0.25      # Bajo: respuestas consistentes y factuales

# ── Rate limiting ──────────────────────────────────────────────────────────────
# Groq free tier: 6,000 tokens/minuto para llama-3.3-70b-versatile
# Cada request usa ~420 tokens (prompt ~240 + respuesta ~180)
# 6000 / 420 = ~14 requests/min máximo → 60s / 14 = ~4.3s entre llamadas
# Usamos 10s para tener margen cómodo y evitar 429 completamente
SLEEP_BETWEEN  = 10        # Segundos entre llamadas — evita rate limit de raíz
MAX_RETRIES    = 3         # Reintentos por artículo tras rate limit
DEFAULT_RETRY_WAIT = 62    # Segundos de espera si Groq no devuelve Retry-After

# ─────────────────────────────────────────────
# NOMBRES DE DIVISA PARA EL PROMPT
# ─────────────────────────────────────────────
CURRENCY_NAMES = {
    "USD": "dólar estadounidense",
    "EUR": "euro",
    "GBP": "libra esterlina",
    "JPY": "yen japonés",
    "AUD": "dólar australiano",
    "CAD": "dólar canadiense",
    "CHF": "franco suizo",
    "NZD": "dólar neozelandés",
}

# ─────────────────────────────────────────────
# PROMPT DEL SISTEMA
# ─────────────────────────────────────────────
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

EJEMPLOS DE BUENOS TITULARES (observa la estructura: dato → mecanismo → impacto divisa):
- IPC EEUU sube 3.2% en enero superando estimaciones: presión hawkish sobre Fed mantiene soporte en USD
- BoC mantiene tasas en 3% pero señala recorte en abril: orientación dovish pesa sobre CAD a medio plazo
- PMI manufacturero zona euro cae a 44.2: contracción profunda del sector industrial → presión bajista EUR
- Nóminas EEUU: 256K empleos vs 160K esperado, sorpresa alcista refuerza narrativa de excepción americana → USD
- GBP/USD cae por escalada en Oriente Próximo: aversión al riesgo favorece activos refugio, presión bajista GBP
- AUD/USD mantiene alzas pese a retroceso USD: resistencia técnica sugiere sesgo alcista AUD a corto plazo
- PPI EEUU supera previsiones reavivando temores de estanflación: lectura bajista para USD por dudas sobre Fed
- RBA sube tasas 25pb a 4.35%: ciclo restrictivo continúa respaldando demanda de AUD en el corto plazo
- Tensión EEUU-Irán escala: petróleo sube y aversión al riesgo crece → alcista CAD por exportaciones, bajista AUD
- Atlanta Fed GDPNow Q1 baja a 3.0% desde 3.1%: leve revisión a la baja del crecimiento, impacto neutro USD

REGLAS IMPORTANTES:
- Artículos sobre pares de divisas (GBP/USD, AUD/USD, etc.) SIEMPRE son relevantes — describe el movimiento y el motivo
- Artículos sobre economías de terceros países son relevantes si afectan flujos de capital, commodities o risk sentiment
- Para análisis técnicos incluye el nivel clave o rango y el sesgo resultante
- Si el impacto es ambiguo, usar "mixto" o "señal contradictoria"
- Si la noticia es completamente irrelevante para forex y macroeconomía global (ej: moda, deportes, cultura sin impacto económico), responde únicamente la palabra: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur        = article.get("cur", "USD")
    cur_name   = CURRENCY_NAMES.get(cur, cur)
    title      = article.get("title", "")
    expand     = article.get("expand", "") or ""
    source     = article.get("source", "")
    lang       = article.get("lang", "en")
    lang_label = "español" if lang == "es" else "inglés"

    return (
        f"DIVISA PRINCIPAL AFECTADA: {cur} ({cur_name})\n"
        f"FUENTE: {source} (artículo en {lang_label})\n"
        f"TÍTULO ORIGINAL: {title}\n"
        f"DESCRIPCIÓN: {expand[:400] if expand else 'Sin descripción'}\n\n"
        f"Nota: Si el artículo menciona un par de divisas como GBP/USD o AUD/USD, "
        f"es SIEMPRE relevante para forex. Analiza el movimiento del par y genera el titular.\n\n"
        f"Genera el titular sintético de impacto:"
    )


def get_retry_after(response) -> int:
    """
    Extrae el tiempo de espera recomendado del header Retry-After de Groq.
    Groq devuelve 'x-ratelimit-reset-tokens' o 'retry-after' en segundos.
    Si no está disponible, devuelve DEFAULT_RETRY_WAIT.
    """
    # Groq puede devolver cualquiera de estos headers
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        val = response.headers.get(header)
        if val:
            try:
                # Puede ser un número de segundos o una cadena como "60s"
                val_clean = str(val).lower().replace("s", "").strip()
                wait = int(float(val_clean))
                return min(wait + 2, 120)  # +2s de margen, cap en 120s
            except (ValueError, TypeError):
                pass
    return DEFAULT_RETRY_WAIT


def call_groq(api_key: str, article: dict) -> str | None:
    """
    Llama a Groq con reintentos automáticos tras rate limit (429).
    Usa el header Retry-After de Groq para esperar exactamente lo necesario.
    Devuelve el titular generado o None si el artículo es irrelevante/error permanente.
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
                wait = get_retry_after(r)
                print(f"  ⏳ Rate limit (intento {attempt}/{MAX_RETRIES}) — esperando {wait}s...")
                time.sleep(wait)
                continue  # Reintentar tras la espera exacta

            if r.status_code == 401:
                raise RuntimeError("GROQ_API_KEY inválida o no configurada.")

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            # Solo marcar como irrelevante si la respuesta es EXCLUSIVAMENTE esa palabra
            if content.strip().upper() == "IRRELEVANTE":
                return None

            # Limpiar comillas residuales del modelo
            content = content.strip('"').strip("'").strip()

            # Validar longitud mínima
            if len(content) < 25:
                print(f"  ⚠️  Respuesta demasiado corta ({len(content)} chars): '{content}'")
                return None

            return content

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout (intento {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue
            return None
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  ⚠️  Error Groq: {e}")
            return None

    print("  ❌ Rate limit persistente tras todos los reintentos — artículo omitido")
    return None


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "❌ GROQ_API_KEY no configurada. "
            "Agrégala en Settings → Secrets → Actions."
        )
    print(f"✅ GROQ_API_KEY configurada ({len(api_key)} chars)")

    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró {NEWS_FILE}.")

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        print("⚠️  news.json sin artículos. Nada que procesar.")
        return

    print(f"\n📰 Total artículos en news.json: {len(articles)}")

    # Todos los artículos sin ai_headline, sin filtro de impacto
    to_process = [
        (i, a) for i, a in enumerate(articles)
        if not a.get("ai_headline")
    ]

    print(f"🔍 Artículos a enriquecer (sin ai_headline): {len(to_process)}")

    if not to_process:
        print("✅ Todos los artículos ya tienen ai_headline.")
        data["ai_enriched_at"] = now_utc.isoformat()
        with open(NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return

    estimated_tokens = len(to_process) * 420
    estimated_time   = len(to_process) * SLEEP_BETWEEN
    print(f"📊 Tokens estimados: ~{estimated_tokens:,} (límite diario: 500,000)")
    print(f"⏱️  Tiempo estimado: ~{estimated_time // 60}min {estimated_time % 60}s "
          f"({SLEEP_BETWEEN}s entre llamadas, sin rate limits)")
    print()

    enriched = 0
    skipped  = 0
    irrelevant = 0

    for idx, (article_idx, article) in enumerate(to_process):
        title_preview = article.get("title", "")[:60]
        cur    = article.get("cur", "?")
        impact = article.get("impact", "?")

        print(f"[{idx+1}/{len(to_process)}] {cur} [{impact}] {title_preview}...")

        headline = call_groq(api_key, article)

        if headline:
            articles[article_idx]["ai_headline"] = headline
            enriched += 1
            preview = headline[:90] + ('...' if len(headline) > 90 else '')
            print(f"  ✅ {preview}")
        else:
            # Distinguir entre irrelevante real y error/rate-limit
            # Si el título menciona pares de divisas o términos forex, es probable un error
            title_lower = article.get("title", "").lower()
            expand_lower = (article.get("expand", "") or "").lower()
            forex_terms = ["usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
                          "dollar", "euro", "pound", "yen", "franc",
                          "/usd", "/eur", "forex", "fx ", "rate", "fed", "ecb", "boe",
                          "gdp", "cpi", "inflation", "inflación"]
            looks_forex = any(t in title_lower or t in expand_lower for t in forex_terms)

            if looks_forex:
                skipped += 1
                print(f"  ⚠️  Sin síntesis (posible error de API — artículo parece relevante)")
            else:
                irrelevant += 1
                print(f"  ⏭️  Sin síntesis (irrelevante para forex)")

        if idx < len(to_process) - 1:
            time.sleep(SLEEP_BETWEEN)

    data["articles"] = articles
    data["ai_enriched_at"] = now_utc.isoformat()
    data["ai_model"] = GROQ_MODEL

    with open(NEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Enriquecidos:         {enriched}")
    print(f"   ⚠️  Omitidos (API error): {skipped}")
    print(f"   ⏭️  Irrelevantes:         {irrelevant}")
    print(f"   💾 Guardado en:          {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
