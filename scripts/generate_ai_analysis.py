#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Google Gemini API (gratuita).
Usa el nuevo SDK google-genai (reemplaza al deprecado google-generativeai).

API gratuita: gemini-2.0-flash — 15 req/min, 1500 req/día
Obtener key gratis: https://aistudio.google.com
"""

import os
import json
import time
import socket
import requests
from datetime import datetime, timezone
from pathlib import Path

# Timeout global de red — evita que el script se cuelgue indefinidamente
socket.setdefaulttimeout(10)

from google import genai
from google.genai import types

# ── Configuración ──────────────────────────────────────────────────────────────

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']

COUNTRY_META = {
    'USD': {'name': 'Estados Unidos',  'bank': 'Reserva Federal (Fed)'},
    'EUR': {'name': 'Eurozona',         'bank': 'Banco Central Europeo (BCE)'},
    'GBP': {'name': 'Reino Unido',     'bank': 'Banco de Inglaterra (BoE)'},
    'JPY': {'name': 'Japón',           'bank': 'Banco de Japón (BoJ)'},
    'AUD': {'name': 'Australia',       'bank': 'Banco de la Reserva de Australia (RBA)'},
    'CAD': {'name': 'Canadá',          'bank': 'Banco de Canadá (BoC)'},
    'CHF': {'name': 'Suiza',           'bank': 'Banco Nacional Suizo (SNB)'},
    'NZD': {'name': 'Nueva Zelanda',   'bank': 'Banco de la Reserva de Nueva Zelanda (RBNZ)'},
}

GITHUB_BASE = 'https://globalinvesting.github.io'
OUTPUT_DIR  = Path('ai-analysis')

# ── Carga de datos desde GitHub Pages ─────────────────────────────────────────

def fetch_json(url: str, timeout: int = 8) -> dict | None:
    """Descarga un JSON desde GitHub Pages. Devuelve None si falla."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None


def load_economic_data(currency: str) -> dict:
    """Carga todos los datos disponibles para una divisa."""
    data = {}

    # 1. Datos económicos principales
    main = fetch_json(f'{GITHUB_BASE}/economic-data/{currency}.json')
    if main and 'data' in main:
        d = main['data']
        data.update({
            'gdp':              d.get('gdp'),
            'gdpGrowth':        d.get('gdpGrowth'),
            'inflation':        d.get('inflation'),
            'unemployment':     d.get('unemployment'),
            'currentAccount':   d.get('currentAccount'),
            'debt':             d.get('debt'),
            'tradeBalance':     d.get('tradeBalance'),
            'production':       d.get('production'),
            'retailSales':      d.get('retailSales'),
            'wageGrowth':       d.get('wageGrowth'),
            'manufacturingPMI': d.get('manufacturingPMI'),
            'termsOfTrade':     d.get('termsOfTrade'),
            'lastUpdate':       main.get('lastUpdate'),
        })

    # 2. Tasa de interés
    rates = fetch_json(f'{GITHUB_BASE}/rates/{currency}.json', timeout=6)
    if rates and rates.get('observations'):
        obs = rates['observations'][0]
        val = obs.get('value')
        if val and val != '.':
            try:
                data['interestRate'] = float(val)
            except ValueError:
                pass

    # 3. Datos extendidos
    ext = fetch_json(f'{GITHUB_BASE}/extended-data/{currency}.json', timeout=6)
    if ext and 'data' in ext:
        d = ext['data']
        data.update({
            'bond10y':               d.get('bond10y'),
            'consumerConfidence':    d.get('consumerConfidence'),
            'businessConfidence':    d.get('businessConfidence'),
            'capitalFlows':          d.get('capitalFlows'),
            'inflationExpectations': d.get('inflationExpectations'),
            'rateMomentum':          d.get('rateMomentum'),
        })

    # 4. COT positioning
    cot = fetch_json(f'{GITHUB_BASE}/cot-data/{currency}.json', timeout=5)
    if cot and cot.get('netPosition') is not None:
        data['cotPositioning'] = cot['netPosition']

    # 5. FX performance
    fxp = fetch_json(f'{GITHUB_BASE}/fx-performance/{currency}.json', timeout=5)
    if fxp and fxp.get('fxPerformance1M') is not None:
        data['fxPerformance1M'] = fxp['fxPerformance1M']

    return data


# ── Formateo para el prompt ────────────────────────────────────────────────────

def fmt(value, decimals: int = 1, suffix: str = ''):
    if value is None:
        return None
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return None


def build_data_summary(currency: str, data: dict) -> str:
    meta = COUNTRY_META[currency]
    lines = [
        f"DIVISA: {currency} — {meta['name']}",
        f"BANCO CENTRAL: {meta['bank']}",
        "",
        "INDICADORES ECONÓMICOS ACTUALES:",
    ]

    indicators = [
        ('gdp',                  'PIB Total',                 lambda v: fmt(v, 2, ' T USD')),
        ('gdpGrowth',            'Crecimiento PIB',           lambda v: fmt(v, 1, '% anual')),
        ('interestRate',         'Tasa de Interés',           lambda v: fmt(v, 2, '%')),
        ('inflation',            'Inflación (IPC)',           lambda v: fmt(v, 1, '% anual')),
        ('unemployment',         'Desempleo',                 lambda v: fmt(v, 1, '%')),
        ('currentAccount',       'Cuenta Corriente',          lambda v: fmt(v, 1, '% PIB')),
        ('debt',                 'Deuda Pública',             lambda v: fmt(v, 1, '% PIB')),
        ('tradeBalance',         'Balanza Comercial',         lambda v: fmt(v / 1000, 1, 'B USD/mes') if v else None),
        ('production',           'Producción Industrial',     lambda v: fmt(v, 1, '% MoM')),
        ('retailSales',          'Ventas Minoristas',         lambda v: fmt(v, 1, '% MoM')),
        ('wageGrowth',           'Crecimiento Salarial',      lambda v: fmt(v, 1, '% anual')),
        ('manufacturingPMI',     'PMI Manufacturero',         lambda v: fmt(v, 1, ' (>50=expansión)')),
        ('cotPositioning',       'COT Positioning (CFTC)',    lambda v: fmt(v / 1000, 1, 'K contratos netos') if v else None),
        ('bond10y',              'Yield Bono 10Y',            lambda v: fmt(v, 2, '%')),
        ('consumerConfidence',   'Confianza Consumidor',      lambda v: fmt(v, 1, ' (base 100)')),
        ('businessConfidence',   'Confianza Empresarial',     lambda v: fmt(v, 1, ' (base 100)')),
        ('capitalFlows',         'Flujos de Capital',         lambda v: fmt(v / 1000, 1, 'B USD') if v else None),
        ('inflationExpectations','Expect. de Inflación',      lambda v: fmt(v, 1, '%')),
        ('termsOfTrade',         'Términos de Intercambio',   lambda v: fmt(v, 1, ' (base 100)')),
        ('fxPerformance1M',      'Rendimiento FX 1M',         lambda v: fmt(v, 2, '% vs USD')),
        ('rateMomentum',         'Momentum de Tasas',         lambda v: fmt(v, 2, '% (cambio 12M)')),
    ]

    available = 0
    for key, label, formatter in indicators:
        value = data.get(key)
        if value is not None:
            formatted = formatter(value)
            if formatted:
                lines.append(f"  • {label}: {formatted}")
                available += 1

    lines.append(f"\n[{available} indicadores disponibles de 21]")
    last_update = data.get('lastUpdate')
    if last_update:
        lines.append(f"[Datos actualizados: {str(last_update)[:10]}]")

    return "\n".join(lines)


# ── Prompt del sistema ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el motor de análisis fundamental de un dashboard profesional de forex usado por traders.

TAREA: Generar un análisis económico riguroso y conciso sobre la divisa indicada.

FORMATO OBLIGATORIO:
- Texto corrido en español, sin bullets, sin títulos, sin markdown
- Exactamente 3 párrafos separados por línea en blanco
- Cada párrafo: 2-4 oraciones densas en información
- Total: entre 180 y 250 palabras

ESTRUCTURA DE LOS PÁRRAFOS:
1. Política monetaria: banco central, tasa actual, postura (hawkish/dovish/neutral), inflación
2. Actividad económica: crecimiento, empleo, consumo, sector exterior (balanza, cuenta corriente)
3. Sentimiento de mercado y perspectivas: COT, rendimiento FX reciente, qué presiona al alza/baja

REGLAS:
- Cita siempre los valores numéricos exactos del input
- Menciona el banco central por su nombre completo en el primer párrafo
- No uses frases vacías como "es importante destacar" o "cabe señalar"
- Si un indicador no tiene dato, no lo menciones ni escribas "sin dato"
- El tono es profesional y directo, como un informe de research de banco de inversión
- No incluyas saludos, despedidas ni meta-comentarios sobre el análisis"""


# ── Generación con Gemini (nuevo SDK google-genai) ─────────────────────────────

def generate_analysis(client: genai.Client, currency: str, data: dict) -> str:
    """Llama a Gemini para generar el análisis de una divisa."""
    data_summary = build_data_summary(currency, data)

    full_prompt = f"""{SYSTEM_PROMPT}

---

{data_summary}

---

Genera ahora el análisis fundamental para {currency}:"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.4,
                    top_p=0.85,
                )
            )

            text = response.text.strip()
            word_count = len(text.split())

            if word_count < 80:
                raise ValueError(f"Respuesta demasiado corta: {word_count} palabras")

            print(f"  ✅ {word_count} palabras generadas")
            return text

        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                wait = 60 if attempt == 0 else 120
                print(f"  ⏳ Rate limit, esperando {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  ⚠️  Error (intento {attempt + 1}/{max_retries}): {e}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"No se pudo generar análisis para {currency}: {e}")

    raise RuntimeError(f"Agotados los reintentos para {currency}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🤖 Generador de Análisis AI — Gemini 2.0 Flash (gratuito)")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "❌ GEMINI_API_KEY no configurada.\n"
            "   Obtenla gratis en: https://aistudio.google.com\n"
            "   Luego: GitHub repo → Settings → Secrets → GEMINI_API_KEY"
        )

    client = genai.Client(api_key=api_key)
    print("✅ Gemini 2.0 Flash configurado (SDK google-genai)\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = {}
    errors  = []

    for i, currency in enumerate(CURRENCIES):
        print(f"[{i+1}/{len(CURRENCIES)}] {currency}...")

        try:
            print(f"  📥 Cargando datos económicos...")
            data = load_economic_data(currency)

            available = sum(1 for v in data.values() if v is not None)
            print(f"  📊 {available} indicadores disponibles")

            if available < 4:
                msg = f"Datos insuficientes ({available} indicadores)"
                print(f"  ⚠️  {msg}, saltando...")
                errors.append(f"{currency}: {msg}")
                results[currency] = {"success": False, "error": msg}
                continue

            print(f"  🧠 Generando con Gemini 2.0 Flash...")
            analysis_text = generate_analysis(client, currency, data)

            output = {
                "currency":    currency,
                "country":     COUNTRY_META[currency]['name'],
                "bank":        COUNTRY_META[currency]['bank'],
                "analysis":    analysis_text,
                "model":       "gemini-2.0-flash",
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "dataSnapshot": {
                    "interestRate":    data.get('interestRate'),
                    "gdpGrowth":       data.get('gdpGrowth'),
                    "inflation":       data.get('inflation'),
                    "unemployment":    data.get('unemployment'),
                    "currentAccount":  data.get('currentAccount'),
                    "cotPositioning":  data.get('cotPositioning'),
                    "fxPerformance1M": data.get('fxPerformance1M'),
                    "lastUpdate":      data.get('lastUpdate'),
                }
            }

            output_path = OUTPUT_DIR / f"{currency}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            results[currency] = {
                "success":     True,
                "wordCount":   len(analysis_text.split()),
                "generatedAt": output["generatedAt"],
            }

            print(f"  💾 Guardado en {output_path}")

            if i < len(CURRENCIES) - 1:
                print(f"  ⏸  Pausa 5s...")
                time.sleep(5)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append(f"{currency}: {str(e)}")
            results[currency] = {"success": False, "error": str(e)}

    # index.json
    successful = [c for c, r in results.items() if r.get('success')]
    index = {
        "generatedAt":    datetime.now(timezone.utc).isoformat(),
        "model":          "gemini-2.0-flash",
        "currencies":     successful,
        "totalGenerated": len(successful),
        "errors":         errors,
        "results":        results,
    }

    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos: {len(successful)}/{len(CURRENCIES)} — {', '.join(successful) or 'ninguno'}")
    if errors:
        print(f"   ❌ Errores ({len(errors)}):")
        for err in errors:
            print(f"      • {err}")
    print(f"   📁 Archivos en: {OUTPUT_DIR}/")
    print("=" * 60)

    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos vs {len(successful)} éxitos")


if __name__ == '__main__':
    main()
