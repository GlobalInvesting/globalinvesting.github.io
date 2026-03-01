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
NEWS_FILE     = Path("news-data/news.json")
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"
MAX_TOKENS    = 140       # Suficiente para un titular de ~120 chars
TEMPERATURE   = 0.25      # Bajo: respuestas consistentes y factuales
SLEEP_BETWEEN = 1.2       # Segundos entre llamadas (evita rate limit 6K tok/min)
# Sin filtro de impacto — se procesan TODOS los artículos sin ai_headline

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
SYSTEM_PROMPT = """Eres un analista de mercados forex que redacta titulares sintéticos para una plataforma profesional de trading.

TAREA: Dado el título y descripción de una noticia económica, genera un titular de UNA SOLA LÍNEA que:
1. Expresa el dato o evento concreto (con cifras si las hay en el texto)
2. Indica el impacto esperado en la divisa indicada (alcista / bajista / neutro)
3. Menciona el mecanismo si es relevante (inflación, empleo, política monetaria, etc.)

FORMATO OBLIGATORIO:
- Una sola línea, sin punto final
- Entre 80 y 135 caracteres
- En español, tono profesional y directo
- Sin markdown, sin comillas, sin emojis, sin introducción

EJEMPLOS DE BUENOS TITULARES:
- IPC EEUU sube 3.2% en enero, supera estimaciones: hawkish para Fed → alcista USD
- BoC mantiene tasas en 3%: mercado descuenta recorte en abril → bajista CAD medio plazo
- PMI manufacturero zona euro cae a 44.2: contracción profunda → presión bajista EUR
- Nóminas EEUU: 256K empleos vs 160K esperados, sorpresa alcista → rally USD probable
- RBA sube tasas 25pb a 4.35%: ciclo restrictivo continúa → alcista AUD corto plazo
- EUR/USD consolida en 1.08 tras datos mixtos: sesgo neutro sin catalizador claro

REGLAS:
- Si el texto no trae cifra concreta, sintetiza la decisión o mensaje clave
- Si el impacto es ambiguo, usar "mixto" o "neutro"
- Para análisis técnicos (soporte, resistencia, niveles clave), indicar el rango y sesgo
- Nunca inventes datos que no estén en el texto recibido
- Si la noticia es irrelevante para forex/macro, responde solo la palabra: IRRELEVANTE"""


def build_user_prompt(article: dict) -> str:
    cur        = article.get("cur", "USD")
    cur_name   = CURRENCY_NAMES.get(cur, cur)
    title      = article.get("title", "")
    expand     = article.get("expand", "") or ""
    source     = article.get("source", "")
    lang       = article.get("lang", "en")
    lang_label = "español" if lang == "es" else "inglés"

    return (
        f"DIVISA AFECTADA: {cur} ({cur_name})\n"
        f"FUENTE: {source} (artículo en {lang_label})\n"
        f"TÍTULO ORIGINAL: {title}\n"
        f"DESCRIPCIÓN: {expand[:400] if expand else 'Sin descripción'}\n\n"
        f"Genera el titular sintético de impacto para {cur}:"
    )


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

        if content.upper().startswith("IRRELEVANTE"):
            return None

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
    print(f"📊 Tokens estimados: ~{estimated_tokens:,} (límite diario: 500,000)")
    print()

    enriched = 0
    skipped  = 0

    for idx, (article_idx, article) in enumerate(to_process):
        title_preview = article.get("title", "")[:60]
        cur    = article.get("cur", "?")
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
    print(f"   ✅ Enriquecidos:  {enriched}")
    print(f"   ⏭️  Omitidos:      {skipped}")
    print(f"   💾 Guardado en:   {NEWS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
