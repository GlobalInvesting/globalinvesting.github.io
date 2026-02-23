#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.

v2.1 — Contexto estructural inferido dinámicamente desde los datos.
       Sin hardcoding de características por divisa.
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


# ─── SYSTEM PROMPT v2.1 ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de forex institucional senior. Escribes research notes para traders profesionales.

TAREA: Generar un análisis fundamental interpretativo de la divisa indicada, en español.

FORMATO:
- Texto corrido, sin bullets, sin títulos, sin markdown
- 3 párrafos separados por línea en blanco
- Entre 130 y 170 palabras totales

ESTRUCTURA:
1. Política monetaria e inflación: ¿Qué está haciendo el banco central y POR QUÉ? ¿Está subiendo, recortando o en pausa? Usa el "Momentum de Tasas" para precisar la dirección del ciclo. Compara la tasa actual vs el promedio global del contexto.
2. Fundamentos macroeconómicos: interpreta el PIB, empleo y balanza exterior. Usa las "Señales estructurales inferidas" al final del input para enriquecer el análisis — por ejemplo, si la divisa financia carry trade, si depende de commodities, si actúa como safe haven, etc. Explica por qué esos factores estructurales amplifican o atenúan lo que dicen los datos actuales.
3. Sentimiento e impulso de mercado: interpreta el COT (¿qué dice el posicionamiento institucional?) y el rendimiento FX del último mes (¿por qué se movió así considerando el contexto estructural?). Cierra con perspectivas a corto plazo.

REGLAS CRÍTICAS:
- INTERPRETA, no listes. El objetivo es explicar POR QUÉ los datos importan, no solo QUÉ dicen.
- Usa el "Contexto global" para hacer comparaciones relativas concretas.
- Si el COT supera ±30K contratos, señala el riesgo de saturación/reversión o la convicción institucional.
- Si el FX 1M supera ±3%, explica el catalizador probable (carry, commodities, risk-on/off, diferencial de tasas).
- El campo "PIB Total" es el tamaño de la economía en trillones USD — NO es la tasa de crecimiento.
- La tasa de crecimiento del PIB es el campo "Crecimiento PIB (% anual)".
- Para USD: el "Rendimiento FX 1M" refleja el índice dólar (DXY) — redáctalo como "el índice dólar subió/cayó X%", nunca como "vs USD".
- Si un indicador no tiene dato, no lo menciones.
- Tono: directo, analítico, como un research note de Goldman Sachs o JP Morgan. Sin saludos ni meta-comentarios."""


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

    rates = fetch_json(f'{GITHUB_BASE}/rates/{currency}.json', timeout=6)
    if rates and rates.get('observations'):
        val = rates['observations'][0].get('value')
        if val and val != '.':
            try:
                data['interestRate'] = float(val)
            except ValueError:
                pass

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

    cot = fetch_json(f'{GITHUB_BASE}/cot-data/{currency}.json', timeout=5)
    if cot and cot.get('netPosition') is not None:
        data['cotPositioning'] = cot['netPosition']

    fxp = fetch_json(f'{GITHUB_BASE}/fx-performance/{currency}.json', timeout=5)
    if fxp and fxp.get('fxPerformance1M') is not None:
        data['fxPerformance1M'] = fxp['fxPerformance1M']

    return data


def compute_global_context(all_data):
    """Calcula promedios globales entre las 8 divisas para comparaciones relativas."""
    def avg(key):
        values = [d[key] for d in all_data.values() if d.get(key) is not None]
        return round(sum(values) / len(values), 2) if values else None

    return {
        'avg_interest_rate': avg('interestRate'),
        'avg_gdp_growth':    avg('gdpGrowth'),
        'avg_inflation':     avg('inflation'),
        'avg_unemployment':  avg('unemployment'),
        'avg_bond10y':       avg('bond10y'),
        'avg_fx_perf_1m':    avg('fxPerformance1M'),
        'avg_cot':           avg('cotPositioning'),
    }


def infer_structural_signals(currency, data, global_context):
    """
    Genera señales estructurales inferidas 100% desde los datos actuales.
    Sin hardcoding: si los datos cambian, las señales cambian.
    Devuelve una lista de strings descriptivos para incluir en el prompt.
    """
    signals = []
    avg_rate = global_context.get('avg_interest_rate') or 3.0

    rate = data.get('interestRate')
    rate_momentum = data.get('rateMomentum')
    current_account = data.get('currentAccount')
    trade_balance = data.get('tradeBalance')
    debt = data.get('debt')
    cot = data.get('cotPositioning')
    fx_1m = data.get('fxPerformance1M')
    terms_of_trade = data.get('termsOfTrade')
    gdp_growth = data.get('gdpGrowth')
    avg_cot = global_context.get('avg_cot') or 0

    # ── Carry trade (tasa significativamente baja vs promedio) ─────────────
    if rate is not None and avg_rate is not None:
        rate_gap = avg_rate - rate
        if rate_gap >= 1.5:
            signals.append(
                f"Tasa muy por debajo del promedio global ({rate:.2f}% vs {avg_rate:.2f}% promedio): "
                f"esta divisa es probable financiadora de carry trade. "
                f"Su apreciación en risk-off puede ser brusca por desmantelamiento de posiciones."
            )
        elif rate - avg_rate >= 1.5:
            signals.append(
                f"Tasa significativamente superior al promedio global ({rate:.2f}% vs {avg_rate:.2f}% promedio): "
                f"la divisa ofrece carry atractivo y puede captar flujos de inversión de renta fija internacional."
            )

    # ── Ciclo del banco central (rateMomentum) ─────────────────────────────
    if rate_momentum is not None:
        if rate_momentum > 0.5:
            signals.append(
                f"Momentum de tasas positivo ({rate_momentum:+.2f}% en 12 meses): "
                f"el banco central está en ciclo de normalización/subida, lo cual es structuralmente alcista para la divisa."
            )
        elif rate_momentum < -0.5:
            signals.append(
                f"Momentum de tasas negativo ({rate_momentum:+.2f}% en 12 meses): "
                f"el banco central está recortando activamente, lo cual presiona a la baja la divisa vía reducción del carry."
            )

    # ── Safe haven (superávit CA + deuda baja + tasa baja) ────────────────
    ca_surplus = current_account is not None and current_account > 2.0
    low_debt   = debt is not None and debt < 60
    low_rate   = rate is not None and avg_rate is not None and (avg_rate - rate) >= 1.5
    if ca_surplus and low_debt and low_rate:
        signals.append(
            f"Combinación de superávit de cuenta corriente ({current_account:.1f}% PIB), "
            f"deuda baja ({debt:.0f}% PIB) y tasa por debajo del promedio global: "
            f"perfil clásico de divisa safe haven. Aprecia en episodios de aversión al riesgo global."
        )
    elif ca_surplus and low_debt:
        signals.append(
            f"Superávit de cuenta corriente ({current_account:.1f}% PIB) y deuda controlada ({debt:.0f}% PIB): "
            f"balance exterior sólido que genera demanda estructural de la divisa."
        )

    # ── Dependencia de exportaciones/commodities ──────────────────────────
    tb = trade_balance
    tot = terms_of_trade
    if tb is not None and tb > 2000 and tot is not None and tot > 100:
        signals.append(
            f"Superávit comercial de {tb/1000:.1f}B USD/mes con términos de intercambio favorables ({tot:.1f}): "
            f"la divisa se beneficia del precio de sus exportaciones (probablemente commodities o manufactura de alto valor). "
            f"Monitorear demanda global e industrial como driver clave."
        )
    elif tb is not None and tb > 2000:
        signals.append(
            f"Superávit comercial robusto ({tb/1000:.1f}B USD/mes): "
            f"genera demanda estructural de la divisa por conversión de ingresos de exportación."
        )
    elif tb is not None and tb < -15000:
        signals.append(
            f"Déficit comercial pronunciado ({tb/1000:.1f}B USD/mes): "
            f"presión vendedora estructural sobre la divisa compensada por otros flujos de capital."
        )

    # ── Posicionamiento institucional extremo (COT) ────────────────────────
    if cot is not None:
        if cot > 30000:
            signals.append(
                f"Posicionamiento especulativo alcista extremo ({cot/1000:.0f}K contratos netos): "
                f"mercado saturado, riesgo elevado de toma de ganancias o reversión si los datos decepcionan."
            )
        elif cot < -30000:
            signals.append(
                f"Posicionamiento especulativo bajista extremo ({cot/1000:.0f}K contratos netos): "
                f"mercado muy corto, potencial rebote contrarian si los fundamentales mejoran o el BC sorprende."
            )

    # ── Divergencia COT vs FX momentum ────────────────────────────────────
    if cot is not None and fx_1m is not None:
        if cot < -20000 and fx_1m > 2.5:
            signals.append(
                f"Divergencia relevante: posicionamiento bajista ({cot/1000:.0f}K) pero apreciación del {fx_1m:.1f}% en el último mes. "
                f"Posible squeeze de cortos en curso — la tendencia puede amplificarse si los cortos cierran posiciones."
            )
        elif cot > 20000 and fx_1m < -2.0:
            signals.append(
                f"Divergencia relevante: posicionamiento alcista ({cot/1000:.0f}K) pero depreciación del {fx_1m:.1f}% en el último mes. "
                f"Posible inicio de liquidación de largos — señal de alerta para posiciones alcistas."
            )

    # ── Régimen de mercado global (risk-on vs risk-off) ───────────────────
    if avg_cot is not None:
        if avg_cot > 8000:
            signals.append(
                "Régimen global actual: risk-on (COT promedio positivo entre las 8 divisas principales). "
                "Divisas de alto carry y commodities tienden a outperform; safe havens de bajo rendimiento, a underperform."
            )
        elif avg_cot < -8000:
            signals.append(
                "Régimen global actual: risk-off (COT promedio negativo entre las 8 divisas principales). "
                "Divisas safe haven y de bajo carry tienden a apreciarse; divisas pro-cíclicas, a depreciarse."
            )

    return signals


def fmt(value, decimals=1, suffix=''):
    if value is None:
        return None
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return None


def build_data_summary(currency, data, global_context=None):
    meta = COUNTRY_META[currency]

    lines = [
        f"DIVISA: {currency} — {meta['name']}",
        f"BANCO CENTRAL: {meta['bank']}",
        "",
        "INDICADORES ECONÓMICOS ACTUALES:",
    ]

    indicators = [
        ('gdp',                  'PIB Total',
         lambda v: fmt(v, 2, ' T USD')),
        ('gdpGrowth',            'Crecimiento PIB',
         lambda v: fmt(v, 1, '% anual')),
        ('interestRate',         'Tasa de Interés',
         lambda v: fmt(v, 2, '%')),
        ('rateMomentum',         'Momentum de Tasas (cambio 12M)',
         lambda v: fmt(v, 2, '% — positivo=subiendo, negativo=recortando')),
        ('inflation',            'Inflación (IPC)',
         lambda v: fmt(v, 1, '% anual')),
        ('inflationExpectations','Expect. Inflación',
         lambda v: fmt(v, 1, '%')),
        ('unemployment',         'Desempleo',
         lambda v: fmt(v, 1, '%')),
        ('currentAccount',       'Cuenta Corriente',
         lambda v: fmt(v, 1, '% PIB')),
        ('debt',                 'Deuda Pública',
         lambda v: fmt(v, 1, '% PIB')),
        ('tradeBalance',         'Balanza Comercial',
         lambda v: fmt(v / 1000, 1, 'B USD/mes') if v else None),
        ('production',           'Producción Industrial',
         lambda v: fmt(v, 1, '% MoM')),
        ('retailSales',          'Ventas Minoristas',
         lambda v: fmt(v, 1, '% MoM')),
        ('wageGrowth',           'Crecimiento Salarial',
         lambda v: fmt(v, 1, '% anual')),
        ('manufacturingPMI',     'PMI Manufacturero',
         lambda v: fmt(v, 1, ' (>50=expansión, <50=contracción)')),
        ('termsOfTrade',         'Términos de Intercambio',
         lambda v: fmt(v, 1, ' (base 100, >100=favorable)')),
        ('cotPositioning',       'COT Positioning',
         lambda v: fmt(v / 1000, 1, 'K contratos netos (positivo=alcista, negativo=bajista)') if v else None),
        ('bond10y',              'Yield Bono 10Y',
         lambda v: fmt(v, 2, '%')),
        ('consumerConfidence',   'Confianza Consumidor',
         lambda v: fmt(v, 1, ' (base 100)')),
        ('businessConfidence',   'Confianza Empresarial',
         lambda v: fmt(v, 1, ' (base 100)')),
        ('capitalFlows',         'Flujos de Capital',
         lambda v: fmt(v / 1000, 1, 'B USD (positivo=entrada, negativo=salida)') if v else None),
        ('fxPerformance1M',      'Rendimiento FX 1M',
         lambda v: fmt(v, 2, '% vs USD (positivo=apreciación)')),
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

    # ── Contexto global para comparaciones relativas ───────────────────────
    if global_context:
        lines.append("")
        lines.append("CONTEXTO GLOBAL (promedio de las 8 divisas principales):")
        mappings = [
            ('avg_interest_rate', 'Tasa de interés promedio', 2, '%'),
            ('avg_gdp_growth',    'Crecimiento PIB promedio', 1, '% anual'),
            ('avg_inflation',     'Inflación promedio',       1, '%'),
            ('avg_unemployment',  'Desempleo promedio',       1, '%'),
            ('avg_bond10y',       'Yield bono 10Y promedio',  2, '%'),
            ('avg_fx_perf_1m',    'Rendimiento FX 1M promedio', 2, '%'),
        ]
        for key, label, decimals, suffix in mappings:
            val = global_context.get(key)
            if val is not None:
                lines.append(f"  • {label}: {val:.{decimals}f}{suffix}")

    # ── Señales estructurales inferidas desde los datos ────────────────────
    if global_context:
        signals = infer_structural_signals(currency, data, global_context)
        if signals:
            lines.append("")
            lines.append("SEÑALES ESTRUCTURALES INFERIDAS DESDE LOS DATOS:")
            lines.append("(Usa estas señales para contextualizar el análisis — son datos calculados, no suposiciones)")
            for signal in signals:
                lines.append(f"  → {signal}")

    return "\n".join(lines)


def call_groq_api(api_key, data_summary, currency):
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{data_summary}\n\n"
                    f"---\n\n"
                    f"Genera el análisis fundamental interpretativo para {currency}. "
                    f"Interpreta el contexto y los comparativos globales. "
                    f"No te limites a listar los datos — explica qué implican para la divisa:"
                ),
            },
        ],
        "max_tokens": 600,
        "temperature": 0.4,
        "top_p": 0.85,
    }
    response = requests.post(
        GROQ_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=40,
    )
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


def generate_analysis(api_key, currency, data, global_context=None):
    data_summary = build_data_summary(currency, data, global_context)

    for attempt in range(3):
        try:
            text = call_groq_api(api_key, data_summary, currency)

            # Normalizar párrafos
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            text = '\n\n'.join(paragraphs)

            word_count = len(text.split())
            if word_count < 60:
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
    print(f"🤖 Generador AI v2.1 — {GROQ_MODEL} via Groq API")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "❌ GROQ_API_KEY no configurada. "
            "Agrégala en Settings → Secrets → Actions del repo."
        )

    print(f"✅ API key configurada ({len(api_key)} caracteres)")
    print(f"🔧 Modelo: {GROQ_MODEL}\n")

    # Test de conectividad
    print("🔍 Testeando conectividad...")
    try:
        r = requests.get('https://httpbin.org/get', timeout=5)
        print(f"  ✅ Internet OK ({r.status_code})")
    except Exception as e:
        print(f"  ❌ Sin internet: {e}")
    try:
        r = requests.get(
            'https://globalinvesting.github.io/economic-data/USD.json', timeout=5
        )
        print(f"  ✅ GitHub Pages OK ({r.status_code})")
    except Exception as e:
        print(f"  ❌ GitHub Pages bloqueado: {e}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Paso 1: cargar todos los datos para calcular contexto global ───────
    print("📥 Cargando datos de las 8 divisas para calcular contexto global...")
    all_data = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        all_data[currency] = load_economic_data(currency)
        available = sum(1 for v in all_data[currency].values() if v is not None)
        print(f"{available} indicadores")

    global_context = compute_global_context(all_data)
    print(f"\n🌐 Contexto global calculado:")
    for k, v in global_context.items():
        if v is not None:
            print(f"   {k}: {v}")
    print()

    # ── Paso 2: generar análisis con contexto global ───────────────────────
    results = {}
    errors  = []

    for i, currency in enumerate(CURRENCIES):
        print(f"[{i+1}/{len(CURRENCIES)}] {currency}...")
        data = all_data[currency]

        available = sum(1 for v in data.values() if v is not None)
        print(f"  📊 {available} indicadores disponibles")

        if available < 4:
            msg = f"Datos insuficientes ({available})"
            print(f"  ⚠️  {msg}, saltando...")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            continue

        try:
            print(f"  🧠 Llamando a Groq API ({GROQ_MODEL})...")
            analysis_text = generate_analysis(api_key, currency, data, global_context)

            output = {
                "currency":    currency,
                "country":     COUNTRY_META[currency]['name'],
                "bank":        COUNTRY_META[currency]['bank'],
                "analysis":    analysis_text,
                "model":       GROQ_MODEL,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "dataSnapshot": {
                    "interestRate":    data.get('interestRate'),
                    "gdpGrowth":       data.get('gdpGrowth'),
                    "inflation":       data.get('inflation'),
                    "unemployment":    data.get('unemployment'),
                    "currentAccount":  data.get('currentAccount'),
                    "rateMomentum":    data.get('rateMomentum'),
                    "cotPositioning":  data.get('cotPositioning'),
                    "fxPerformance1M": data.get('fxPerformance1M'),
                    "lastUpdate":      data.get('lastUpdate'),
                },
                "globalContext": global_context,
            }

            output_path = OUTPUT_DIR / f"{currency}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            results[currency] = {
                "success":     True,
                "wordCount":   len(analysis_text.split()),
                "generatedAt": output["generatedAt"],
            }
            print(f"  💾 Guardado → {output_path}")

            if i < len(CURRENCIES) - 1:
                print(f"  ⏸  Pausa 3s...")
                time.sleep(3)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append(f"{currency}: {str(e)}")
            results[currency] = {"success": False, "error": str(e)}

    # ── Índice general ─────────────────────────────────────────────────────
    successful = [c for c, r in results.items() if r.get('success')]
    index = {
        "generatedAt":    datetime.now(timezone.utc).isoformat(),
        "model":          GROQ_MODEL,
        "currencies":     successful,
        "totalGenerated": len(successful),
        "errors":         errors,
        "results":        results,
        "globalContext":  global_context,
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
