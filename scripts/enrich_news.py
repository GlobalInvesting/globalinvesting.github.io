#!/usr/bin/env python3
"""
enrich_news.py
Lee news-data/news.json, llama a Groq API para generar un titular sintético
de impacto para cada artículo sin `ai_headline`, y sobreescribe el JSON.

Corre 2 veces al día via GitHub Actions (8:00 y 20:00 UTC).
Solo procesa artículos con impacto 'high' o 'med' para optimizar tokens.

Modelo: llama-3.3-70b-versatile (Groq free tier)
Estimación de tokens por ejecución: ~24.000 (< 5% del límite diario de 500K)
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
NEWS_FILE    = Path("news-data/news.json")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 120          # La síntesis es corta, no necesitamos más
TEMPERATURE  = 0.3          # Bajo para respuestas consistentes y factuales
SLEEP_BETWEEN = 1.2         # Segundos entre llamadas (evita rate limit de 6K tok/min)
PROCESS_IMPACTS = {"high", "med"}  # Solo procesar artículos relevantes

# ─────────────────────────────────────────────
# DIVISAS PRINCIPALES (para el contexto del prompt)
# ─────────────────────────────────────────────
CURRENCY_NAMES = {
    "USD": "el dólar estadounidense",
    "EUR": "el euro",
    "GBP": "la libra esterlina",
    "JPY": "el yen japonés",
    "AUD": "el dólar australiano",
    "CAD": "el dólar canadiense",
    "CHF": "el franco suizo",
    "NZD": "el dólar neozelandés",
}


# ─────────────────────────────────────────────
# PROMPT SYSTEM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de mercados forex que redacta titulares sintéticos para una plataforma profesional de trading.

TAREA: Dado el título y descripción de una noticia económica, genera un titular de una sola línea que:
1. Explica el dato o evento en términos concretos (con cifras si las hay)
2. Indica el impacto esperado en la divisa indicada (alcista/bajista/neutral)
3. Menciona el mecanismo de transmisión si es relevante (carry, inflación, crecimiento, etc.)

FORMATO OBLIGATORIO:
- Una sola línea, sin punto final
- Entre 80 y 140 caracteres
- En español, tono profesional y directo
- Sin markdown, sin comillas, sin emojis

EJEMPLOS DE BUENOS TITULARES:
- IPC EEUU sube 3.2% en enero, supera estimaciones: hawkish para Fed → alcista USD
- BoC mantiene tasas en 3%: mercado descuenta recorte en abril → bajista CAD medio plazo
- PMI manufacturero zona euro cae a 44.2: contracción profunda → presión bajista EUR
- Nóminas EEUU: 256K empleos vs 160K esperados, sorpresa alcista → rally USD probable
- RBA sube tasas 25pb a 4.35%: ciclo restrictivo continúa → alcista AUD corto plazo

REGLAS:
- Si no hay dato numérico concreto, sintetiza el mensaje clave del banco central o institución
- Si el impacto es ambiguo, indicar "neutro" o "mixto"
- Nunca inventes datos que no estén en el texto
- Si la noticia no tiene relación clara con forex/macro, responde solo: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    """Construye el prompt de usuario para un artículo."""
    currency = article.get("cur", "USD")
    currency_name = CURRENCY_NAMES.get(currency, currency)
    title = article.get("title", "")
    expand = article.get("expand", "")
    source = article.get("source", "")

    return (
        f"DIVISA AFECTADA: {currency} ({currency_name})\n"
        f"FUENTE: {source}\n"
        f"TÍTULO ORIGINAL: {title}\n"
        f"DESCRIPCIÓN: {expand[:400] if expand else 'Sin descripción'}\n\n"
        f"Genera el titular sintético de impacto para {currency}:"
    )


def call_groq(api_key: str, article: dict) -> str | None:
    """
    Llama a Groq API para un artículo.
    Devuelve el titular generado o None si falla.
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

    try:
        r = requests.post(
            GROQ_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=20,
        )

        if r.status_code == 429:
            print("  ⏳ Rate limit — esperando 60s...")
            time.sleep(60)
            return None

        if r.status_code == 401:
            raise RuntimeError("GROQ_API_KEY inválida o no configurada.")

        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()

        # Descartar respuestas que indican irrelevancia
        if content.upper().startswith("IRRELEVANTE"):
            return None

        # Limpiar posibles artefactos de formato
        content = content.strip('"').strip("'").strip()

        return content if len(content) > 20 else None

    except requests.exceptions.Timeout:
        print("  ⚠️  Timeout en llamada a Groq")
        return None
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  ⚠️  Error Groq: {e}")
        return None


def main():
    now_utc = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"🤖 Enriquecedor AI — {GROQ_MODEL}")
    print(f"   {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # ── Leer GROQ_API_KEY ────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "❌ GROQ_API_KEY no configurada. "
            "Agrégala en Settings → Secrets → Actions."
        )
    print(f"✅ GROQ_API_KEY configurada ({len(api_key)} chars)")

    # ── Leer news.json ────────────────────────────────────────────
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró {NEWS_FILE}. Ejecuta fetch_news.py primero.")

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    if not articles:
        print("⚠️  news.json sin artículos. Nada que procesar.")
        return

    print(f"\n📰 Total artículos en news.json: {len(articles)}")

    # ── Filtrar los que necesitan enriquecimiento ─────────────────
    # Solo artículos de impacto relevante que aún no tienen ai_headline
    to_process = [
        (i, a) for i, a in enumerate(articles)
        if a.get("impact") in PROCESS_IMPACTS
        and not a.get("ai_headline")
    ]

    print(f"🔍 Artículos a enriquecer (high/med sin ai_headline): {len(to_process)}")

    if not to_process:
        print("✅ Todos los artículos relevantes ya tienen ai_headline. Sin trabajo adicional.")
        # Actualizar timestamp de enriquecimiento igualmente
        data["ai_enriched_at"] = now_utc.isoformat()
        with open(NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return

    # Estimación de tokens
    estimated_tokens = len(to_process) * 420  # ~300 entrada + 120 salida
    print(f"📊 Tokens estimados: ~{estimated_tokens:,} (límite diario: 500,000)")
    print()

    # ── Procesar artículos ────────────────────────────────────────
    enriched = 0
    skipped  = 0
    errors   = 0

    for idx, (article_idx, article) in enumerate(to_process):
        title_preview = article.get("title", "")[:60]
        cur = article.get("cur", "?")
        impact = article.get("impact", "?")

        print(f"[{idx+1}/{len(to_process)}] {cur} [{impact}] {title_preview}...")

        headline = call_groq(api_key, article)

        if headline:
            articles[article_idx]["ai_headline"] = headline
            enriched += 1
            print(f"  ✅ {headline[:80]}{'...' if len(headline) > 80 else ''}")
        else:
            skipped += 1
            print(f"  ⏭️  Sin síntesis (irrelevante o error)")

        # Pausa entre llamadas para respetar rate limit
        if idx < len(to_process) - 1:
            time.sleep(SLEEP_BETWEEN)

    # ── Guardar JSON actualizado ──────────────────────────────────
    data["articles"] = articles
    data["ai_enriched_at"] = now_utc.isoformat()
    data["ai_model"] = GROQ_MODEL

    with open(NEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Resumen ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Enriquecidos:  {enriched}")
    print(f"   ⏭️  Omitidos:      {skipped}")
    print(f"   ❌ Errores:       {errors}")
    print(f"   💾 Guardado en:   {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
