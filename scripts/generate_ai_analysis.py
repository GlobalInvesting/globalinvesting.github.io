#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.

v2.3 — Prompt reescrito para redacción más natural y concisa en español.
       Composición exportadora obtenida en vivo desde UN Comtrade API.
       Fallback a HS2 codes estáticos si la API no responde.
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

COMTRADE_REPORTER_CODES = {
    'AUD': '36',
    'CAD': '124',
    'NZD': '554',
    'CHF': '756',
    'USD': '842',
    'EUR': '276',
    'GBP': '826',
    'JPY': '392',
}

HS2_NAMES_ES = {
    '01': 'animales vivos',
    '02': 'carne y despojos comestibles',
    '03': 'pescado y crustáceos',
    '04': 'leche y productos lácteos',
    '05': 'otros productos de origen animal',
    '06': 'plantas vivas y floricultura',
    '07': 'hortalizas y tubérculos',
    '08': 'frutas y frutos comestibles',
    '09': 'café, té y especias',
    '10': 'cereales',
    '11': 'productos de la molinería',
    '12': 'semillas oleaginosas',
    '13': 'gomas y resinas vegetales',
    '14': 'materias vegetales trenzables',
    '15': 'grasas y aceites animales o vegetales',
    '16': 'preparaciones de carne o pescado',
    '17': 'azúcares y confitería',
    '18': 'cacao y sus preparaciones',
    '19': 'preparaciones a base de cereales',
    '20': 'preparaciones de hortalizas o frutas',
    '21': 'preparaciones alimenticias diversas',
    '22': 'bebidas y líquidos alcohólicos',
    '23': 'residuos de industrias alimentarias',
    '24': 'tabaco',
    '25': 'sal, azufre y cementos',
    '26': 'minerales metalíferos',
    '27': 'combustibles y petróleo',
    '28': 'productos químicos inorgánicos',
    '29': 'productos químicos orgánicos',
    '30': 'productos farmacéuticos',
    '31': 'abonos',
    '32': 'colorantes y pigmentos',
    '33': 'aceites esenciales y perfumería',
    '34': 'jabones y detergentes',
    '35': 'materias albuminoideas',
    '36': 'explosivos',
    '37': 'productos fotográficos',
    '38': 'productos químicos diversos',
    '39': 'plásticos',
    '40': 'caucho',
    '41': 'pieles y cueros en bruto',
    '42': 'manufacturas de cuero',
    '43': 'peletería',
    '44': 'madera y manufacturas',
    '45': 'corcho',
    '46': 'cestería',
    '47': 'pasta de madera y papel reciclado',
    '48': 'papel y cartón',
    '49': 'libros e impresos',
    '50': 'seda',
    '51': 'lana y pelo animal',
    '52': 'algodón',
    '53': 'otras fibras textiles',
    '54': 'filamentos sintéticos',
    '55': 'fibras sintéticas discontinuas',
    '56': 'guata y fieltro',
    '57': 'alfombras',
    '58': 'tejidos especiales',
    '59': 'tejidos técnicos',
    '60': 'tejidos de punto',
    '61': 'prendas de punto',
    '62': 'prendas excepto de punto',
    '63': 'artículos textiles confeccionados',
    '64': 'calzado',
    '65': 'sombreros',
    '66': 'paraguas',
    '67': 'plumas y artículos',
    '68': 'manufacturas de piedra',
    '69': 'cerámica',
    '70': 'vidrio',
    '71': 'piedras preciosas y metales',
    '72': 'hierro y acero',
    '73': 'manufacturas de acero',
    '74': 'cobre',
    '75': 'níquel',
    '76': 'aluminio',
    '77': 'reservado HS',
    '78': 'plomo',
    '79': 'cinc',
    '80': 'estaño',
    '81': 'metales comunes diversos',
    '82': 'herramientas metálicas',
    '83': 'manufacturas de metal',
    '84': 'maquinaria industrial',
    '85': 'equipos eléctricos y electrónicos',
    '86': 'material ferroviario',
    '87': 'vehículos y autopartes',
    '88': 'aeronaves',
    '89': 'embarcaciones',
    '90': 'instrumentos de precisión',
    '91': 'relojería',
    '92': 'instrumentos musicales',
    '93': 'armas y municiones',
    '94': 'muebles',
    '95': 'juguetes y artículos deportivos',
    '96': 'manufacturas diversas',
    '97': 'obras de arte y antigüedades',
    '99': 'mercancías no clasificadas',
}

EXPORT_FALLBACK_HS2 = {
    'AUD': ['26', '27', '10', '12'],
    'CAD': ['27', '87', '84', '26'],
    'NZD': ['04', '02', '10', '03'],
    'CHF': ['30', '90', '71', '84'],
    'USD': ['84', '85', '88', '30'],
    'EUR': ['84', '87', '85', '30'],
    'GBP': ['84', '30', '85', '88'],
    'JPY': ['87', '84', '85', '90'],
}

GITHUB_BASE = 'https://globalinvesting.github.io'
OUTPUT_DIR  = Path('ai-analysis')
GROQ_MODEL  = 'llama-3.3-70b-versatile'
GROQ_URL    = 'https://api.groq.com/openai/v1/chat/completions'

_export_cache = {}


# ─── SYSTEM PROMPT v2.3 ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de mercados cambiarios senior que escribe comentarios diarios para una mesa de trading. Tu estilo es el de un profesional hispanohablante nativo: directo, sin rodeos, con frases cortas y vocabulario preciso del mundo financiero.

TAREA: Redactar un análisis fundamental conciso sobre la divisa indicada.

FORMATO OBLIGATORIO:
- Texto corrido, sin títulos, sin viñetas, sin markdown
- Exactamente 3 párrafos separados por línea en blanco
- Entre 100 y 130 palabras en total
- Cada párrafo: 2-3 oraciones como máximo

ESTRUCTURA DE LOS PÁRRAFOS:
1. Política monetaria: ¿Qué está haciendo el banco central y por qué? Menciona si está subiendo, bajando o pausando tasas, y qué dato lo justifica (inflación, crecimiento).
2. Macroeconomía: interpreta el PIB, el empleo y la balanza exterior. Si tienes la composición exportadora verificada, úsala para explicar el origen del superávit o déficit.
3. Mercado: ¿Qué dicen el COT y el rendimiento del último mes? ¿Hay convergencia o divergencia entre posicionamiento y precio? Cierra con perspectivas breves.

REGLAS DE REDACCIÓN:
- Escribe como lo haría un español o latinoamericano nativo, no como una traducción del inglés
- Evita construcciones calcadas del inglés: "esto se debe a que", "lo que sugiere que", "en un contexto donde"
- Usa conectores naturales: "así", "de ahí que", "por eso", "con todo", "aunque", "si bien"
- Varía la estructura de las oraciones; no empieces tres seguidas con el mismo sujeto
- Nada de frases genéricas del tipo "la perspectiva a corto plazo dependerá de..."

REGLAS SOBRE LOS DATOS:
- Momentum de tasas NEGATIVO = el banco central está RECORTANDO → ciclo expansivo → presión bajista sobre la divisa
- Momentum de tasas POSITIVO = el banco central está SUBIENDO → ciclo restrictivo → presión alcista
- COT > +30K contratos: posicionamiento muy estirado al alza; riesgo de corrección
- COT < -30K contratos: posicionamiento muy estirado a la baja; potencial rebote contrario
- Si un indicador no tiene dato, ignóralo completamente; no lo menciones ni lo infieras
- El campo "Rendimiento FX 1M" ya tiene el signo correcto; no lo inviertas
- Para USD: el rendimiento FX refleja el dólar índice (DXY)"""


def fetch_json(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None


def fetch_export_composition(currency):
    if currency in _export_cache:
        return _export_cache[currency]

    reporter_code = COMTRADE_REPORTER_CODES.get(currency)
    if not reporter_code:
        return None

    def _parse_comtrade_response(data, year_label):
        commodities = data.get('data', [])
        if not commodities:
            if isinstance(data, list):
                commodities = data
            else:
                return None
        if not commodities:
            return None

        sample = commodities[0]
        code_field = next(
            (k for k in ['cmdCode', 'CmdCode', 'cmd_code', 'classificationCode'] if k in sample),
            None
        )
        value_field = next(
            (k for k in ['primaryValue', 'PrimaryValue', 'TradeValue', 'tradeValue', 'fobvalue'] if k in sample),
            None
        )
        if not code_field or not value_field:
            return None

        hs2_totals = {}
        for c in commodities:
            code_raw = str(c.get(code_field, '')).strip()
            if code_raw.isdigit() and len(code_raw) >= 1:
                hs2_code = code_raw[:2].zfill(2) if len(code_raw) >= 2 else code_raw.zfill(2)
                if hs2_code == '99':
                    continue
                val = float(c.get(value_field, 0) or 0)
                hs2_totals[hs2_code] = hs2_totals.get(hs2_code, 0) + val

        if not hs2_totals:
            return None

        top3 = sorted(hs2_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        total_exports = sum(hs2_totals.values())

        sectors = []
        for code, value in top3:
            name = HS2_NAMES_ES.get(code, f'HS{code}')
            pct = (value / total_exports * 100) if total_exports > 0 else 0
            value_b = value / 1e9
            sectors.append(f"{name} ({pct:.0f}%, ${value_b:.1f}B)")

        if sectors:
            return f"Comtrade {year_label}: {', '.join(sectors)}"
        return None

    for year in ['2023', '2022']:
        try:
            url = (
                f"https://comtradeapi.un.org/public/v1/preview/C/A/HS"
                f"?reporterCode={reporter_code}"
                f"&period={year}"
                f"&partnerCode=0"
                f"&flowCode=X"
                f"&customsCode=C00"
                f"&motCode=0"
            )
            r = requests.get(url, timeout=12, headers={'Accept': 'application/json'})
            if r.status_code == 429:
                time.sleep(10)
                r = requests.get(url, timeout=12, headers={'Accept': 'application/json'})
            if r.ok and len(r.content) > 500:
                result_str = _parse_comtrade_response(r.json(), year)
                if result_str:
                    print(f"  🌐 Comtrade {year} OK para {currency}: {result_str[:80]}")
                    _export_cache[currency] = result_str
                    return result_str
        except Exception as e:
            print(f"  ⚠️  Comtrade {year} error para {currency}: {e}")
        time.sleep(2)

    fallback_codes = EXPORT_FALLBACK_HS2.get(currency, [])
    if fallback_codes:
        sectors = [HS2_NAMES_ES.get(code, f'HS{code}') for code in fallback_codes[:3]]
        result = f"Principales exportaciones: {', '.join(sectors)}"
        print(f"  📦 Fallback para {currency}: {result}")
        _export_cache[currency] = result
        return result

    _export_cache[currency] = None
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
    signals = []
    avg_rate  = global_context.get('avg_interest_rate') or 3.0
    avg_cot   = global_context.get('avg_cot') or 0

    rate          = data.get('interestRate')
    rate_momentum = data.get('rateMomentum')
    current_account = data.get('currentAccount')
    trade_balance = data.get('tradeBalance')
    debt          = data.get('debt')
    cot           = data.get('cotPositioning')
    fx_1m         = data.get('fxPerformance1M')
    terms_of_trade = data.get('termsOfTrade')

    if rate is not None and avg_rate is not None:
        rate_gap = avg_rate - rate
        if rate_gap >= 1.5:
            signals.append(
                f"Tasa muy por debajo del promedio global ({rate:.2f}% vs {avg_rate:.2f}%): "
                f"divisa probable financiadora de carry trade. "
                f"En episodios de aversión al riesgo puede apreciarse bruscamente por el cierre de posiciones cortas."
            )
        elif rate - avg_rate >= 1.5:
            signals.append(
                f"Tasa significativamente superior al promedio global ({rate:.2f}% vs {avg_rate:.2f}%): "
                f"carry atractivo para inversores de renta fija internacional."
            )

    if rate_momentum is not None:
        if rate_momentum > 0.5:
            signals.append(
                f"El banco central lleva subiendo tasas {rate_momentum:+.2f}pp en 12 meses: ciclo restrictivo activo, structuralmente alcista para la divisa."
            )
        elif rate_momentum < -0.5:
            signals.append(
                f"El banco central lleva recortando {rate_momentum:+.2f}pp en 12 meses: ciclo expansivo, el menor carry presiona a la baja la divisa."
            )

    ca_surplus = current_account is not None and current_account > 2.0
    low_debt   = debt is not None and debt < 60
    low_rate   = rate is not None and avg_rate is not None and (avg_rate - rate) >= 1.5

    if ca_surplus and low_debt and low_rate:
        signals.append(
            f"Perfil de activo refugio: superávit de cuenta corriente ({current_account:.1f}% del PIB), "
            f"deuda contenida ({debt:.0f}% del PIB) y tasa baja. "
            f"En momentos de tensión global esta combinación genera flujos defensivos hacia la divisa."
        )
    elif ca_surplus and low_debt:
        signals.append(
            f"Balance exterior sólido: superávit de cuenta corriente ({current_account:.1f}% del PIB) y deuda controlada ({debt:.0f}% del PIB)."
        )

    if trade_balance is not None and terms_of_trade is not None:
        if trade_balance > 2000 and terms_of_trade > 100:
            signals.append(
                f"Superávit comercial de {trade_balance/1000:.1f}B USD/mes con términos de intercambio favorables (índice {terms_of_trade:.1f}): "
                f"los precios de exportación superan a los de importación, lo que amplía el excedente."
            )
        elif trade_balance < -15000:
            signals.append(
                f"Déficit comercial pronunciado ({trade_balance/1000:.1f}B USD/mes): presión vendedora estructural sobre la divisa."
            )

    if cot is not None:
        if cot > 30000:
            signals.append(
                f"COT muy estirado al alza ({cot/1000:.0f}K contratos): el mercado está saturado de largos; cualquier decepción en los datos puede provocar una corrección rápida."
            )
        elif cot < -30000:
            signals.append(
                f"COT muy estirado a la baja ({cot/1000:.0f}K contratos): mercado con posición corta extrema; potencial rebote contrario si los datos mejoran."
            )

    if cot is not None and fx_1m is not None:
        if cot < -20000 and fx_1m > 2.5:
            signals.append(
                f"Divergencia relevante: posicionamiento bajista ({cot/1000:.0f}K) pero la divisa se apreció {fx_1m:.1f}% el último mes — posible squeeze de cortos."
            )
        elif cot > 20000 and fx_1m < -2.0:
            signals.append(
                f"Divergencia relevante: posicionamiento alcista ({cot/1000:.0f}K) pero la divisa cayó {abs(fx_1m):.1f}% el último mes — posible inicio de cierre de largos."
            )

    if avg_cot is not None:
        if avg_cot > 8000:
            signals.append("Régimen global risk-on: el apetito por riesgo favorece divisas de alto carry y exportadoras de materias primas.")
        elif avg_cot < -8000:
            signals.append("Régimen global risk-off: los flujos se dirigen hacia activos refugio.")

    return signals


def fmt(value, decimals=1, suffix=''):
    if value is None:
        return None
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return None


def build_data_summary(currency, data, global_context=None, export_composition=None):
    meta = COUNTRY_META[currency]

    lines = [
        f"DIVISA: {currency} — {meta['name']}",
        f"BANCO CENTRAL: {meta['bank']}",
        "",
        "INDICADORES:",
    ]

    indicators = [
        ('gdpGrowth',            'Crecimiento PIB',
         lambda v: fmt(v, 1, '% anual')),
        ('interestRate',         'Tasa de interés',
         lambda v: fmt(v, 2, '%')),
        ('rateMomentum',         'Momentum de tasas (12M)',
         lambda v: fmt(v, 2, '% — NEGATIVO=recortando, POSITIVO=subiendo')),
        ('inflation',            'Inflación',
         lambda v: fmt(v, 1, '% anual')),
        ('unemployment',         'Desempleo',
         lambda v: fmt(v, 1, '%')),
        ('currentAccount',       'Cuenta corriente',
         lambda v: (
             f"{v:.1f}% del PIB ({'superávit' if v > 0 else 'déficit'})"
         ) if v is not None else None),
        ('tradeBalance',         'Balanza comercial (bienes)',
         lambda v: (
             f"{v/1000:.1f}B USD/mes ({'superávit' if v > 0 else 'déficit'})"
         ) if v is not None else None),
        ('debt',                 'Deuda pública',
         lambda v: fmt(v, 1, '% del PIB')),
        ('manufacturingPMI',     'PMI manufacturero',
         lambda v: fmt(v, 1, ' (>50 = expansión)')),
        ('termsOfTrade',         'Términos de intercambio',
         lambda v: fmt(v, 1, ' (base 100)')),
        ('cotPositioning',       'COT (posicionamiento especulativo)',
         lambda v: (
             f"{v/1000:+.1f}K contratos netos ({'alcista' if v > 0 else 'bajista'})"
         ) if v is not None else None),
        ('bond10y',              'Yield bono 10Y',
         lambda v: fmt(v, 2, '%')),
        ('fxPerformance1M',      'Rendimiento FX último mes',
         lambda v: (
             f"{v:+.2f}% vs USD ({'apreciación' if v > 0 else 'depreciación'})"
         ) if v is not None else None),
        ('inflationExpectations', 'Expectativas de inflación',
         lambda v: fmt(v, 1, '%')),
        ('wageGrowth',           'Crecimiento salarial',
         lambda v: fmt(v, 1, '% anual')),
    ]

    available = 0
    for key, label, formatter in indicators:
        value = data.get(key)
        if value is not None:
            formatted = formatter(value)
            if formatted:
                lines.append(f"  • {label}: {formatted}")
                available += 1

    lines.append(f"\n[{available} indicadores disponibles]")
    if data.get('lastUpdate'):
        lines.append(f"[Datos a: {str(data['lastUpdate'])[:10]}]")

    if export_composition:
        lines.append("")
        lines.append(f"COMPOSICIÓN EXPORTADORA: {export_composition}")

    if global_context:
        lines.append("")
        lines.append("PROMEDIOS GLOBALES (8 divisas principales):")
        mappings = [
            ('avg_interest_rate', 'Tasa', 2, '%'),
            ('avg_gdp_growth',    'PIB', 1, '% anual'),
            ('avg_inflation',     'Inflación', 1, '%'),
            ('avg_fx_perf_1m',    'FX 1M', 2, '%'),
        ]
        for key, label, decimals, suffix in mappings:
            val = global_context.get(key)
            if val is not None:
                lines.append(f"  • {label}: {val:.{decimals}f}{suffix}")

    if global_context:
        signals = infer_structural_signals(currency, data, global_context)
        if signals:
            lines.append("")
            lines.append("SEÑALES ESTRUCTURALES:")
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
                    f"Redacta el análisis para {currency}. "
                    f"Recuerda: 3 párrafos, 100-130 palabras en total, español natural y directo. "
                    f"Interpreta causas y consecuencias; no te limites a listar cifras."
                ),
            },
        ],
        "max_tokens": 500,
        "temperature": 0.45,
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


def generate_analysis(api_key, currency, data, global_context=None, export_composition=None):
    data_summary = build_data_summary(currency, data, global_context, export_composition)

    for attempt in range(3):
        try:
            text = call_groq_api(api_key, data_summary, currency)

            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            text = '\n\n'.join(paragraphs)

            word_count = len(text.split())
            if word_count < 60:
                raise ValueError(f"Respuesta demasiado corta: {word_count} palabras")

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
    print(f"🤖 Generador AI v2.3 — {GROQ_MODEL} via Groq API")
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

    print("🔍 Testeando conectividad...")
    for label, url in [
        ("Internet",      "https://httpbin.org/get"),
        ("GitHub Pages",  "https://globalinvesting.github.io/economic-data/USD.json"),
        ("UN Comtrade",   "https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=36&period=2022&partnerCode=0&cmdCode=TOTAL&flowCode=X&customsCode=C00&motCode=0"),
    ]:
        try:
            r = requests.get(url, timeout=8)
            print(f"  ✅ {label} OK ({r.status_code})")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("📥 Cargando datos económicos...")
    all_data = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        all_data[currency] = load_economic_data(currency)
        available = sum(1 for v in all_data[currency].values() if v is not None)
        print(f"{available} indicadores")

    global_context = compute_global_context(all_data)
    print(f"\n🌐 Contexto global:")
    for k, v in global_context.items():
        if v is not None:
            print(f"   {k}: {v}")
    print()

    print("📦 Obteniendo composición exportadora (UN Comtrade)...")
    export_compositions = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        composition = fetch_export_composition(currency)
        export_compositions[currency] = composition
        if not composition:
            print("sin dato")
    print()

    results = {}
    errors  = []

    for i, currency in enumerate(CURRENCIES):
        print(f"[{i+1}/{len(CURRENCIES)}] {currency}...")
        data = all_data[currency]
        export_comp = export_compositions.get(currency)

        available = sum(1 for v in data.values() if v is not None)
        print(f"  📊 {available} indicadores económicos")
        if export_comp:
            print(f"  🏭 Exportaciones: {export_comp[:80]}...")

        if available < 4:
            msg = f"Datos insuficientes ({available})"
            print(f"  ⚠️  {msg}, saltando...")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            continue

        try:
            print(f"  🧠 Llamando a Groq API...")
            analysis_text = generate_analysis(
                api_key, currency, data, global_context, export_comp
            )

            output = {
                "currency":    currency,
                "country":     COUNTRY_META[currency]['name'],
                "bank":        COUNTRY_META[currency]['bank'],
                "analysis":    analysis_text,
                "model":       GROQ_MODEL,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "exportComposition": export_comp,
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
                "success":           True,
                "wordCount":         len(analysis_text.split()),
                "generatedAt":       output["generatedAt"],
                "exportDataSource":  "comtrade" if export_comp and "Comtrade" in export_comp else "fallback",
            }
            print(f"  💾 Guardado → {output_path}")

            if i < len(CURRENCIES) - 1:
                print(f"  ⏸  Pausa 3s...")
                time.sleep(3)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append(f"{currency}: {str(e)}")
            results[currency] = {"success": False, "error": str(e)}

    successful = [c for c, r in results.items() if r.get('success')]
    comtrade_count = sum(
        1 for r in results.values()
        if r.get('exportDataSource') == 'comtrade'
    )

    index = {
        "generatedAt":      datetime.now(timezone.utc).isoformat(),
        "model":            GROQ_MODEL,
        "version":          "2.3",
        "currencies":       successful,
        "totalGenerated":   len(successful),
        "comtradeHits":     comtrade_count,
        "errors":           errors,
        "results":          results,
        "globalContext":    global_context,
    }
    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos: {len(successful)}/{len(CURRENCIES)} — {', '.join(successful) or 'ninguno'}")
    print(f"   🌐 Comtrade en vivo: {comtrade_count}/{len(CURRENCIES)} divisas")
    if errors:
        print(f"   ❌ Errores:")
        for err in errors:
            print(f"      • {err}")
    print("=" * 60)

    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos")


if __name__ == '__main__':
    main()
