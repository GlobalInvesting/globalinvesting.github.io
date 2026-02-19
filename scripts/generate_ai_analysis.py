#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.
"""

import os
import json
import time
import socket
import requests
from datetime import datetime, timezone
from pathlib import Path

socket.setdefaulttimeout(15)

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']

COUNTRY_META = {
    'USD': {'name': 'Estados Unidos',  'bank': 'Reserva Federal (Fed)'},
    'EUR': {'name': 'Eurozona',        'bank': 'Banco Central Europeo (BCE)'},
    'GBP': {'name': 'Reino Unido',     'bank': 'Banco de Inglaterra (BoE)'},
    'JPY': {'name': 'Japón',           'bank': 'Banco de Japón (BoJ)'},
    'AUD': {'name': 'Australia',       'bank': 'Banco de la Reserva de Australia (RBA)'},
    'CAD': {'name': 'Canadá',          'bank': 'Banco de Canadá (BoC)'},
    'CHF': {'name': 'Suiza',           'bank': 'Banco Nacional Suizo (SNB)'},
    'NZD': {'name': 'Nueva Zelanda',   'bank': 'Banco de la Reserva de Nueva Zelanda (RBNZ)'},
}

GITHUB_BASE = 'https://globalinvesting.github.io'
OUTPUT_DIR  = Path('ai-analysis')
GROQ_MODEL  = 'llama-3.3-70b-versatile'
GROQ_URL    = 'https://api.groq.com/openai/v1/chat/completions'

def fetch_json(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None

def load_economic_data(currency):
    data = {}
    main = fetch_json(f'{GITHUB_BASE}/economic-data/{currency}.json')
    if main and 'data' in main:
        d = main['data']
        data.update({
            'gdp': d.get('gdp'), 'gdpGrowth': d.get('gdpGrowth'),
            'inflation': d.get('inflation'), 'unemployment': d.get('unemployment'),
            'currentAccount': d.get('currentAccount'), 'debt': d.get('debt'),
            'tradeBalance': d.get('tradeBalance'), 'production': d.get('production'),
            'retailSales': d.get('retailSales'), 'wageGrowth': d.get('wageGrowth'),
            'manufacturingPMI': d.get('manufacturingPMI'), 'termsOfTrade': d.get('termsOfTrade'),
            'lastUpdate': main.get('lastUpdate'),
        })
    rates = fetch_json(f'{GITHUB_BASE}/rates/{currency}.json', timeout=6)
    if rates and rates.get('observations'):
        val = rates['observations'][0].get('value')
        if val and val != '.':
            try: data['interestRate'] = float(val)
            except ValueError: pass
    ext = fetch_json(f'{GITHUB_BASE}/extended-data/{currency}.json', timeout=6)
    if ext and 'data' in ext:
        d = ext['data']
        data.update({
            'bond10y': d.get('bond10y'), 'consumerConfidence': d.get('consumerConfidence'),
            'businessConfidence': d.get('businessConfidence'), 'capitalFlows': d.get('capitalFlows'),
            'inflationExpectations': d.get('inflationExpectations'), 'rateMomentum': d.get('rateMomentum'),
        })
    cot = fetch_json(f'{GITHUB_BASE}/cot-data/{currency}.json', timeout=5)
    if cot and cot.get('netPosition') is not None:
        data['cotPositioning'] = cot['netPosition']
    fxp = fetch_json(f'{GITHUB_BASE}/fx-performance/{currency}.json', timeout=5)
    if fxp and fxp.get('fxPerformance1M') is not None:
        data['fxPerformance1M'] = fxp['fxPerformance1M']
    return data

def fmt(value, decimals=1, suffix=''):
    if value is None: return None
    try: return f"{float(value):.{decimals}f}{suffix}"
    except: return None

def build_data_summary(currency, data):
    meta = COUNTRY_META[currency]
    lines = [f"DIVISA: {currency} — {meta['name']}", f"BANCO CENTRAL: {meta['bank']}", "", "INDICADORES ECONÓMICOS ACTUALES:"]
    indicators = [
        ('gdp','PIB Total',lambda v: fmt(v,2,' T USD')),
        ('gdpGrowth','Crecimiento PIB',lambda v: fmt(v,1,'% anual')),
        ('interestRate','Tasa de Interés',lambda v: fmt(v,2,'%')),
        ('inflation','Inflación (IPC)',lambda v: fmt(v,1,'% anual')),
        ('unemployment','Desempleo',lambda v: fmt(v,1,'%')),
        ('currentAccount','Cuenta Corriente',lambda v: fmt(v,1,'% PIB')),
        ('debt','Deuda Pública',lambda v: fmt(v,1,'% PIB')),
        ('tradeBalance','Balanza Comercial',lambda v: fmt(v/1000,1,'B USD/mes') if v else None),
        ('production','Producción Industrial',lambda v: fmt(v,1,'% MoM')),
        ('retailSales','Ventas Minoristas',lambda v: fmt(v,1,'% MoM')),
        ('wageGrowth','Crecimiento Salarial',lambda v: fmt(v,1,'% anual')),
        ('manufacturingPMI','PMI Manufacturero',lambda v: fmt(v,1,' (>50=expansión)')),
        ('cotPositioning','COT Positioning',lambda v: fmt(v/1000,1,'K contratos netos') if v else None),
        ('bond10y','Yield Bono 10Y',lambda v: fmt(v,2,'%')),
        ('consumerConfidence','Confianza Consumidor',lambda v: fmt(v,1,' (base 100)')),
        ('businessConfidence','Confianza Empresarial',lambda v: fmt(v,1,' (base 100)')),
        ('capitalFlows','Flujos de Capital',lambda v: fmt(v/1000,1,'B USD') if v else None),
        ('inflationExpectations','Expect. Inflación',lambda v: fmt(v,1,'%')),
        ('termsOfTrade','Términos de Intercambio',lambda v: fmt(v,1,' (base 100)')),
        ('fxPerformance1M','Rendimiento FX 1M',lambda v: fmt(v,2,'% vs USD')),
        ('rateMomentum','Momentum de Tasas',lambda v: fmt(v,2,'% (cambio 12M)')),
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
    if data.get('lastUpdate'):
        lines.append(f"[Datos actualizados: {str(data['lastUpdate'])[:10]}]")
    return "\n".join(lines)

SYSTEM_PROMPT = """Eres el motor de análisis fundamental de un dashboard profesional de forex.

TAREA: Generar un análisis económico riguroso y conciso sobre la divisa indicada.

FORMATO:
- Texto corrido en español, sin bullets, sin títulos, sin markdown
- Exactamente 3 párrafos separados por línea en blanco
- Total: entre 180 y 250 palabras

ESTRUCTURA:
1. Política monetaria: banco central (nombre completo), tasa actual, postura hawkish/dovish/neutral, inflación
2. Actividad económica: crecimiento, empleo, consumo, sector exterior
3. Sentimiento de mercado: COT, rendimiento FX reciente, perspectivas

REGLAS:
- Cita los valores numéricos exactos del input
- Si un indicador no tiene dato, no lo menciones
- Tono profesional y directo, como research de banco de inversión
- Sin saludos ni meta-comentarios"""

def call_groq_api(api_key, data_summary, currency):
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{data_summary}\n\n---\n\nGenera el análisis fundamental para {currency}:"}
        ],
        "max_tokens": 600,
        "temperature": 0.4,
        "top_p": 0.85,
    }
    response = requests.post(GROQ_URL, json=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }, timeout=40)
    if response.status_code == 429:
        raise RuntimeError("RATE_LIMIT")
    if response.status_code == 401:
        raise RuntimeError("INVALID_KEY: verifica que GROQ_API_KEY esté correctamente configurada")
    response.raise_for_status()
    data = response.json()
    try:
        return data['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Respuesta inesperada: {data}") from e

def generate_analysis(api_key, currency, data):
    data_summary = build_data_summary(currency, data)
    for attempt in range(3):
        try:
            text = call_groq_api(api_key, data_summary, currency)
              # Limpiar saltos de línea irregulares entre párrafos
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            text = '\n\n'.join(paragraphs)
            word_count = len(text.split())
            if word_count < 80:
                raise ValueError(f"Respuesta corta: {word_count} palabras")
            print(f"  ✅ {word_count} palabras generadas")
            return text
        except RuntimeError as e:
            err_str = str(e)
            if "RATE_LIMIT" in err_str:
                wait = 60 if attempt == 0 else 120
                print(f"  ⏳ Rate limit, esperando {wait}s...")
                time.sleep(wait)
            elif "INVALID_KEY" in err_str:
                raise
            elif attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"  ⚠️  Error intento {attempt+1}: {e}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"  ⚠️  Error intento {attempt+1}: {e}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Falló para {currency}: {e}")
    raise RuntimeError(f"Agotados reintentos para {currency}")

def main():
    print("=" * 60)
    print(f"🤖 Generador AI — {GROQ_MODEL} via Groq API")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError("❌ GROQ_API_KEY no configurada. Agrégala en Settings → Secrets → Actions del repo.")

    print(f"✅ API key configurada ({len(api_key)} caracteres)")
    print(f"🔧 Modelo: {GROQ_MODEL}\n")

    print("🔍 Testeando conectividad...")
    try:
        r = requests.get('https://httpbin.org/get', timeout=5)
        print(f"  ✅ Internet OK ({r.status_code})")
    except Exception as e:
        print(f"  ❌ Sin internet: {e}")
    try:
        r = requests.get('https://globalinvesting.github.io/economic-data/USD.json', timeout=5)
        print(f"  ✅ GitHub Pages OK ({r.status_code})")
    except Exception as e:
        print(f"  ❌ GitHub Pages bloqueado: {e}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {}
    errors = []

    for i, currency in enumerate(CURRENCIES):
        print(f"[{i+1}/{len(CURRENCIES)}] {currency}...")
        try:
            print(f"  📥 Cargando datos...")
            data = load_economic_data(currency)
            available = sum(1 for v in data.values() if v is not None)
            print(f"  📊 {available} indicadores disponibles")
            if available < 4:
                msg = f"Datos insuficientes ({available})"
                print(f"  ⚠️  {msg}, saltando...")
                errors.append(f"{currency}: {msg}")
                results[currency] = {"success": False, "error": msg}
                continue
            print(f"  🧠 Llamando a Groq API ({GROQ_MODEL})...")
            analysis_text = generate_analysis(api_key, currency, data)
            output = {
                "currency": currency, "country": COUNTRY_META[currency]['name'],
                "bank": COUNTRY_META[currency]['bank'], "analysis": analysis_text,
                "model": GROQ_MODEL, "generatedAt": datetime.now(timezone.utc).isoformat(),
                "dataSnapshot": {
                    "interestRate": data.get('interestRate'), "gdpGrowth": data.get('gdpGrowth'),
                    "inflation": data.get('inflation'), "unemployment": data.get('unemployment'),
                    "currentAccount": data.get('currentAccount'), "cotPositioning": data.get('cotPositioning'),
                    "fxPerformance1M": data.get('fxPerformance1M'), "lastUpdate": data.get('lastUpdate'),
                }
            }
            output_path = OUTPUT_DIR / f"{currency}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            results[currency] = {"success": True, "wordCount": len(analysis_text.split()), "generatedAt": output["generatedAt"]}
            print(f"  💾 Guardado → {output_path}")
            if i < len(CURRENCIES) - 1:
                print(f"  ⏸  Pausa 3s...")
                time.sleep(3)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append(f"{currency}: {str(e)}")
            results[currency] = {"success": False, "error": str(e)}

    successful = [c for c, r in results.items() if r.get('success')]
    index = {
        "generatedAt": datetime.now(timezone.utc).isoformat(), "model": GROQ_MODEL,
        "currencies": successful, "totalGenerated": len(successful),
        "errors": errors, "results": results,
    }
    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos: {len(successful)}/{len(CURRENCIES)} — {', '.join(successful) or 'ninguno'}")
    if errors:
        print(f"   ❌ Errores:")
        for err in errors:
            print(f"      • {err}")
    print("=" * 60)
    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos")

if __name__ == '__main__':
    main()
