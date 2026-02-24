#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.

v2.5 — Nuevo módulo de datos en tiempo real:
       • fetch_live_rate_decision(): obtiene la decisión REAL del último meeting
         del banco central desde FRED API (gratis, requiere FRED_API_KEY en GitHub
         Secrets) con fallback a World Bank API (sin key) y luego a rateMomentum.
       • fetch_frankfurter_fx(): calcula fxPerformance1M real desde Frankfurter
         API (completamente gratuita, sin key) en lugar del JSON estático.
       • Nuevo campo 'lastRateDecision' en el summary con dirección inequívoca.
       • Composición exportadora en vivo desde UN Comtrade API.
       • Fallback en cascada para cada fuente — nunca se rompe sin datos.
"""

import os
import json
import time
import socket
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

socket.setdefaulttimeout(15)

# ─── Frankfurter API — FX en tiempo real (ECB rates, sin key) ────────────────
FRANKFURTER_BASE = 'https://api.frankfurter.dev/v1'

# Cache para evitar llamadas repetidas en el mismo run
_fx_performance_cache = {}

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


# ─── SYSTEM PROMPT v2.4 ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de mercados de divisas senior que redacta comentarios de mercado para una plataforma de trading profesional. Escribes en español de forma nativa, fluida y analítica — como un profesional de habla hispana que lleva años cubriendo los mercados cambiarios, no como alguien que traduce del inglés.

TAREA: Redactar un análisis fundamental de la divisa indicada a partir de los datos económicos proporcionados.

FORMATO OBLIGATORIO:
- Texto corrido, sin títulos, sin viñetas, sin markdown, sin negrita
- Exactamente 3 párrafos separados por línea en blanco
- Entre 150 y 200 palabras en total
- Los párrafos deben tener longitud similar; ninguno puede ser de una sola oración

ESTRUCTURA NARRATIVA:
Párrafo 1 — Política monetaria y contexto macro: explica qué está haciendo el banco central, por qué, y cómo encaja con los datos de inflación y crecimiento. Compara la tasa con el promedio global si es relevante. No te limites a describir el dato; interpreta qué significa para la divisa.

Párrafo 2 — Fundamentos externos y sectoriales: analiza la balanza comercial y la cuenta corriente. Si tienes la composición exportadora, úsala para explicar de dónde viene el superávit o déficit y qué factores externos (precios de commodities, demanda global, ciclo de inversión) influyen. Conecta los datos con las perspectivas del sector.

Párrafo 3 — Posicionamiento de mercado y perspectivas: interpreta el COT y el rendimiento FX del último mes. ¿El precio confirma o contradice el posicionamiento especulativo? ¿Hay riesgo de squeeze, corrección o continuación de tendencia? Cierra con una valoración equilibrada del panorama a corto plazo, mencionando los principales catalizadores o riesgos.

ESTILO Y REDACCIÓN:
- Escribe con fluidez narrativa: conecta ideas con "aunque", "de ahí que", "sin embargo", "así", "con todo", "por eso", "si bien", "en cambio"
- Varía la longitud y estructura de las oraciones para evitar monotonía
- Evita estructuras calcadas del inglés: "esto se debe a que", "lo que sugiere que", "en un contexto de", "en un entorno donde"
- Prohibido usar frases vacías: "la perspectiva dependerá de los datos", "el mercado estará atento a", "sigue siendo clave"
- No repitas el nombre de la divisa más de dos veces por párrafo; usa pronombres o referencias indirectas

INTERPRETACIÓN DE DATOS — REGLAS ESTRICTAS:
- Momentum de tasas NEGATIVO = banco central está RECORTANDO → presión bajista sobre la divisa por menor carry
- Momentum de tasas POSITIVO = banco central está SUBIENDO → presión alcista por mayor carry diferencial
- COT > +30K contratos netos: posicionamiento muy cargado al alza; menciona el riesgo de corrección ante cualquier decepción
- COT < -30K contratos netos: posicionamiento muy cargado a la baja; señala el potencial rebote contrario
- Si un indicador no tiene dato, ignóralo por completo; no lo menciones ni lo infieras desde otros datos
- El signo del "Rendimiento FX 1M" ya está calculado correctamente; no lo inviertas nunca
- Para USD: el rendimiento FX 1M refleja el comportamiento del dólar índice (DXY)
- El campo "PIB Total" es el tamaño absoluto de la economía en trillones USD, no la tasa de crecimiento"""


def fetch_json(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None


def derive_rate_decision_from_history(currency, observations):
    """
    Deriva la decisión real del banco central analizando el historial de tasas
    almacenado en rates/{currency}.json.

    Lógica:
      1. Filtra observaciones con valor numérico válido.
      2. Busca el PRIMER cambio real respecto al valor más reciente
         (ignorando días consecutivos con el mismo valor — son confirmaciones,
         no nuevas decisiones).
      3. Calcula delta, dirección y fecha de la última decisión.
      4. También analiza la tendencia de los últimos 12 meses para distinguir
         entre "subida puntual dentro de un ciclo bajista" vs. "giro de política".

    Devuelve dict con: direction, delta, current_rate, prev_rate, date,
                       source, cycle_12m, cycle_label.
    """
    if not observations:
        return None

    # Parsear y ordenar: más reciente primero
    parsed = []
    for obs in observations:
        val = obs.get('value')
        date = obs.get('date', '')
        if val is None or val == '.':
            continue
        try:
            parsed.append({'value': float(val), 'date': date})
        except (ValueError, TypeError):
            continue

    if not parsed:
        return None

    # Ordenar descendente por fecha
    parsed.sort(key=lambda x: x['date'], reverse=True)

    current_rate = parsed[0]['value']
    current_date = parsed[0]['date']

    # ── Encontrar la última decisión: primer valor distinto al actual ─────────
    prev_rate = None
    decision_date = current_date
    for obs in parsed[1:]:
        if abs(obs['value'] - current_rate) > 0.01:
            prev_rate = obs['value']
            decision_date = parsed[parsed.index(obs) - 1]['date'] if parsed.index(obs) > 0 else current_date
            # La fecha de la decisión es cuando APARECIÓ el nuevo valor
            # (el primer registro con el valor actual)
            for i, o in enumerate(parsed):
                if abs(o['value'] - current_rate) > 0.01:
                    decision_date = parsed[i - 1]['date'] if i > 0 else current_date
                    break
            break

    if prev_rate is None:
        # Tasa estable en todo el historial disponible
        return {
            'direction':    'MANTUVO',
            'delta':        0.0,
            'current_rate': current_rate,
            'prev_rate':    current_rate,
            'date':         current_date,
            'source':       'rates_history',
            'cycle_12m':    0.0,
            'cycle_label':  'sin cambios en el período analizado',
        }

    delta = round(current_rate - prev_rate, 4)
    direction = 'SUBIÓ' if delta > 0.01 else ('BAJÓ' if delta < -0.01 else 'MANTUVO')

    # ── Calcular ciclo de los últimos 12 meses ────────────────────────────────
    from datetime import datetime as dt
    try:
        today = dt.strptime(current_date[:10], '%Y-%m-%d')
        cutoff = today.replace(year=today.year - 1)
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        within_12m = [o for o in parsed if o['date'] >= cutoff_str]
        if within_12m:
            oldest_in_12m = within_12m[-1]['value']
            cycle_12m = round(current_rate - oldest_in_12m, 4)
        else:
            cycle_12m = 0.0
    except Exception:
        cycle_12m = 0.0

    # Etiqueta del ciclo para el contexto del modelo
    if cycle_12m > 0.1:
        cycle_label = f"ciclo ALCISTA en 12M (+{cycle_12m:.2f}pp acumulados)"
    elif cycle_12m < -0.1:
        cycle_label = f"ciclo BAJISTA en 12M ({cycle_12m:.2f}pp acumulados)"
    else:
        cycle_label = "estable en los últimos 12 meses"

    return {
        'direction':    direction,
        'delta':        delta,
        'current_rate': current_rate,
        'prev_rate':    prev_rate,
        'date':         decision_date,
        'source':       'rates_history',
        'cycle_12m':    cycle_12m,
        'cycle_label':  cycle_label,
    }


def fetch_live_rate_decision(currency, current_rate, fred_api_key=None):
    """
    DEPRECATED en v2.6 — reemplazado por derive_rate_decision_from_history().
    Se mantiene como stub por compatibilidad; retorna None para forzar
    el uso del historial local.
    """
    return None


def fetch_frankfurter_fx(currency, days=30):
    """
    Calcula el rendimiento FX real del último mes usando Frankfurter API.
    - API gratuita del BCE, sin key, sin límite de llamadas.
    - Devuelve el % de cambio de la divisa vs USD en los últimos `days` días.
    - Para USD calcula contra una cesta ponderada (EUR, GBP, JPY, CAD, CHF, AUD).
    - Fallback: retorna None si la API no responde.
    """
    if currency in _fx_performance_cache:
        return _fx_performance_cache[currency]

    try:
        today = datetime.now(timezone.utc).date()
        past  = today - timedelta(days=days + 5)  # +5 por días no hábiles

        if currency == 'USD':
            # Para USD: variación del DXY aproximado (USD vs cesta)
            # Tomamos EUR/USD: si EUR sube, USD baja → invertimos
            url_now  = f"{FRANKFURTER_BASE}/latest?base=EUR&symbols=USD"
            url_past = f"{FRANKFURTER_BASE}/{past}?base=EUR&symbols=USD"
            now_data  = fetch_json(url_now,  timeout=8)
            past_data = fetch_json(url_past, timeout=8)
            if now_data and past_data:
                rate_now  = now_data['rates']['USD']
                rate_past = past_data['rates']['USD']
                # Si EUR/USD sube, el USD se debilitó → fx_perf negativo para USD
                pct = round((rate_past / rate_now - 1) * 100, 4)
                _fx_performance_cache[currency] = pct
                print(f"  💱 Frankfurter USD: {pct:+.2f}% (1M vs EUR proxy)")
                return pct
        else:
            # Para todas las demás: cotizamos contra USD
            # Frankfurter usa EUR como base obligatoria, así que calculamos
            # currency/USD = (currency/EUR) / (USD/EUR)
            symbols = f"{currency},USD"
            url_now  = f"{FRANKFURTER_BASE}/latest?base=EUR&symbols={symbols}"
            url_past = f"{FRANKFURTER_BASE}/{past}?base=EUR&symbols={symbols}"
            now_data  = fetch_json(url_now,  timeout=8)
            past_data = fetch_json(url_past, timeout=8)

            if now_data and past_data and currency in now_data.get('rates', {}):
                # rate(currency/EUR) hoy y hace 30 días
                cur_now  = now_data['rates'][currency]
                usd_now  = now_data['rates']['USD']
                cur_past = past_data['rates'][currency]
                usd_past = past_data['rates']['USD']

                # currency/USD hoy y hace 30 días
                cross_now  = cur_now  / usd_now
                cross_past = cur_past / usd_past

                pct = round((cross_now / cross_past - 1) * 100, 4)
                _fx_performance_cache[currency] = pct
                print(f"  💱 Frankfurter {currency}: {pct:+.2f}% (1M vs USD real)")
                return pct

    except Exception as e:
        print(f"  ⚠️  Frankfurter FX error {currency}: {e}")

    _fx_performance_cache[currency] = None
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


def load_economic_data(currency, fred_api_key=None):
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

    # ── Tasa de interés + historial completo para derivar decisión ────────────
    rates_raw = fetch_json(f'{GITHUB_BASE}/rates/{currency}.json', timeout=6)
    if rates_raw and rates_raw.get('observations'):
        observations = rates_raw['observations']
        # Tasa actual: primer valor válido
        for obs in observations:
            val = obs.get('value')
            if val and val != '.':
                try:
                    data['interestRate'] = float(val)
                    break
                except ValueError:
                    pass

        # Decisión del banco central: derivada del historial propio
        rate_decision = derive_rate_decision_from_history(currency, observations)
        if rate_decision:
            data['lastRateDecision'] = rate_decision
            dir_str = rate_decision['direction']
            delta   = rate_decision.get('delta', 0) or 0
            cycle   = rate_decision.get('cycle_label', '')
            print(f"  📊 {currency}: {dir_str} {delta:+.2f}pp | {cycle}")

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

    # ── FX performance: Frankfurter API (real) con fallback al JSON estático ──
    fx_live = fetch_frankfurter_fx(currency, days=30)
    if fx_live is not None:
        data['fxPerformance1M'] = fx_live
        data['fxSource'] = 'frankfurter'
    else:
        fxp = fetch_json(f'{GITHUB_BASE}/fx-performance/{currency}.json', timeout=5)
        if fxp and fxp.get('fxPerformance1M') is not None:
            data['fxPerformance1M'] = fxp['fxPerformance1M']
            data['fxSource'] = 'static'

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

    rate_momentum = data.get('rateMomentum')
    last_decision = data.get('lastRateDecision')

    # Preferimos la decisión real (FRED/WorldBank) sobre el momentum acumulado
    if last_decision and last_decision.get('direction'):
        rd    = last_decision['direction']
        delta = last_decision.get('delta') or 0
        src   = last_decision.get('source', '')
        src_note = ''  # datos propios siempre son confiables
        if rd == 'SUBIÓ':
            signals.append(
                f"El banco central acaba de SUBIR tasas ({delta:+.2f}pp): ciclo restrictivo activo, "
                f"estructuralmente alcista para la divisa por mayor carry diferencial.{src_note}"
            )
        elif rd == 'BAJÓ':
            signals.append(
                f"El banco central acaba de BAJAR tasas ({abs(delta):.2f}pp): ciclo expansivo, "
                f"el menor carry presiona a la baja la divisa.{src_note}"
            )
        elif rd == 'MANTUVO' and rate_momentum is not None:
            if rate_momentum < -0.5:
                signals.append(
                    f"Pausa en el ciclo: el banco central mantuvo tasas pero lleva {rate_momentum:+.2f}pp acumulados en 12M — "
                    f"política aún laxa respecto al pico, carry reducido.{src_note}"
                )
            elif rate_momentum > 0.5:
                signals.append(
                    f"Pausa en el ciclo: el banco central mantuvo tasas pero lleva {rate_momentum:+.2f}pp de subidas en 12M — "
                    f"política restrictiva sostenida, carry por encima del ciclo anterior.{src_note}"
                )
    elif rate_momentum is not None:
        if rate_momentum > 0.5:
            signals.append(
                f"El banco central lleva subiendo tasas {rate_momentum:+.2f}pp en 12 meses: "
                f"ciclo restrictivo activo, estructuralmente alcista para la divisa. "
                f"[⚠️ dato acumulado 12M — agrega FRED_API_KEY para mayor precisión]"
            )
        elif rate_momentum < -0.5:
            signals.append(
                f"El banco central lleva recortando {rate_momentum:+.2f}pp en 12 meses: "
                f"ciclo expansivo, el menor carry presiona a la baja la divisa. "
                f"[⚠️ dato acumulado 12M — agrega FRED_API_KEY para mayor precisión]"
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
    avg_rate = (global_context or {}).get('avg_interest_rate') or 3.0
    avg_fx   = (global_context or {}).get('avg_fx_perf_1m') or 0.0

    lines = [
        f"DIVISA: {currency} — {meta['name']}",
        f"BANCO CENTRAL: {meta['bank']}",
        "",
        "INDICADORES ECONÓMICOS:",
    ]

    # ── Política monetaria ─────────────────────────────────────────────────
    rate = data.get('interestRate')
    rate_mom = data.get('rateMomentum')
    inflation = data.get('inflation')
    gdp_growth = data.get('gdpGrowth')
    last_decision = data.get('lastRateDecision')

    if rate is not None:
        rate_vs_avg = rate - avg_rate
        direction = "por encima" if rate_vs_avg > 0 else "por debajo"
        lines.append(f"  • Tasa de interés: {rate:.2f}% ({abs(rate_vs_avg):.2f}pp {direction} del promedio global de {avg_rate:.2f}%)")

    # Ciclo monetario: usamos la decisión real si está disponible, si no el momentum
    if last_decision and last_decision.get('direction'):
        direction = last_decision['direction']
        delta     = last_decision.get('delta') or 0
        prev      = last_decision.get('prev_rate')
        date      = last_decision.get('date', 'N/D')
        cycle     = last_decision.get('cycle_label', '')

        if direction == 'SUBIÓ':
            lines.append(
                f"  • Última decisión monetaria: SUBIÓ tasas {delta:+.2f}pp "
                f"(de {prev:.2f}% a {rate:.2f}%, ef. {date}) "
                f"→ HAWKISH — presión ALCISTA sobre la divisa"
            )
        elif direction == 'BAJÓ':
            lines.append(
                f"  • Última decisión monetaria: BAJÓ tasas {abs(delta):.2f}pp "
                f"(de {prev:.2f}% a {rate:.2f}%, ef. {date}) "
                f"→ DOVISH — presión BAJISTA sobre la divisa"
            )
        else:
            lines.append(
                f"  • Última decisión monetaria: MANTUVO en {rate:.2f}% ({date}) "
                f"→ neutral a corto plazo"
            )
        if cycle:
            lines.append(f"  • Contexto de ciclo (12M): {cycle}")
    elif rate_mom is not None:
        # Fallback: momentum acumulado 12M — menos preciso pero útil
        if rate_mom > 0:
            lines.append(
                f"  • Ciclo monetario (12M acumulado): SUBIENDO tasas ({rate_mom:+.2f}pp) → hawkish "
                f"[⚠️ dato acumulado, puede no reflejar la decisión más reciente]"
            )
        elif rate_mom < 0:
            lines.append(
                f"  • Ciclo monetario (12M acumulado): RECORTANDO tasas ({rate_mom:+.2f}pp) → dovish "
                f"[⚠️ dato acumulado, puede no reflejar la decisión más reciente]"
            )
        else:
            lines.append(f"  • Ciclo monetario (12M acumulado): en pausa (0.00pp)")
    if inflation is not None:
        target_diff = inflation - 2.0
        status = "por encima del objetivo" if target_diff > 0.3 else ("por debajo del objetivo" if target_diff < -0.3 else "cerca del objetivo del 2%")
        lines.append(f"  • Inflación: {inflation:.1f}% anual ({status})")
    if gdp_growth is not None:
        avg_gdp = (global_context or {}).get('avg_gdp_growth') or 0.5
        vs_global = "superior" if gdp_growth > avg_gdp else "inferior"
        lines.append(f"  • Crecimiento PIB: {gdp_growth:.1f}% anual ({vs_global} al promedio global de {avg_gdp:.1f}%)")

    # ── Empleo y consumo ───────────────────────────────────────────────────
    unemployment = data.get('unemployment')
    wage_growth  = data.get('wageGrowth')
    retail_sales = data.get('retailSales')
    if unemployment is not None:
        avg_unemp = (global_context or {}).get('avg_unemployment') or 4.5
        label = "bajo" if unemployment < avg_unemp else "elevado"
        lines.append(f"  • Desempleo: {unemployment:.1f}% ({label} respecto al promedio de {avg_unemp:.1f}%)")
    if wage_growth is not None:
        lines.append(f"  • Crecimiento salarial: {wage_growth:.1f}% anual")
    if retail_sales is not None:
        lines.append(f"  • Ventas minoristas: {retail_sales:+.1f}% mensual")

    # ── Balance exterior ───────────────────────────────────────────────────
    ca = data.get('currentAccount')
    tb = data.get('tradeBalance')
    tot = data.get('termsOfTrade')
    if ca is not None:
        lines.append(f"  • Cuenta corriente: {ca:+.1f}% del PIB ({'superávit — demanda estructural de la divisa' if ca > 0 else 'déficit — presión vendedora estructural'})")
    if tb is not None:
        lines.append(f"  • Balanza comercial: {tb/1000:+.1f}B USD/mes ({'superávit' if tb > 0 else 'déficit'} en bienes)")
    if tot is not None:
        label = "favorable (exportaciones ganan valor relativo)" if tot > 100 else "desfavorable (importaciones encarecidas)"
        lines.append(f"  • Términos de intercambio: {tot:.1f} (base 100 — {label})")

    # ── Actividad ──────────────────────────────────────────────────────────
    pmi = data.get('manufacturingPMI')
    prod = data.get('production')
    debt = data.get('debt')
    bond = data.get('bond10y')
    if pmi is not None:
        lines.append(f"  • PMI manufacturero: {pmi:.1f} ({'expansión' if pmi >= 50 else 'contracción'})")
    if prod is not None:
        lines.append(f"  • Producción industrial: {prod:+.1f}% mensual")
    if debt is not None:
        lines.append(f"  • Deuda pública: {debt:.0f}% del PIB")
    if bond is not None:
        avg_bond = (global_context or {}).get('avg_bond10y') or 3.0
        lines.append(f"  • Yield bono 10Y: {bond:.2f}% (promedio global: {avg_bond:.2f}%)")

    # ── Sentimiento e inflación esperada ───────────────────────────────────
    cc  = data.get('consumerConfidence')
    bc  = data.get('businessConfidence')
    ie  = data.get('inflationExpectations')
    cf  = data.get('capitalFlows')
    if cc is not None:
        lines.append(f"  • Confianza del consumidor: {cc:.1f}")
    if bc is not None:
        lines.append(f"  • Confianza empresarial: {bc:.1f}")
    if ie is not None:
        lines.append(f"  • Expectativas de inflación: {ie:.1f}%")
    if cf is not None:
        lines.append(f"  • Flujos de capital: {cf/1000:+.1f}B USD ({'entrada neta' if cf > 0 else 'salida neta'})")

    # ── Mercado ────────────────────────────────────────────────────────────
    cot  = data.get('cotPositioning')
    fx1m = data.get('fxPerformance1M')
    if cot is not None:
        if cot > 30000:
            cot_interp = f"POSICIÓN ALCISTA EXTREMA — mercado muy cargado al alza, riesgo de corrección ante cualquier decepción"
        elif cot < -30000:
            cot_interp = f"POSICIÓN BAJISTA EXTREMA — mercado muy cargado a la baja, potencial rebote contrario si mejoran los datos"
        elif cot > 0:
            cot_interp = "sesgo especulativo neto alcista"
        else:
            cot_interp = "sesgo especulativo neto bajista"
        lines.append(f"  • COT (posicionamiento especulativo): {cot/1000:+.1f}K contratos netos — {cot_interp}")
    if fx1m is not None:
        fx_vs_avg = fx1m - avg_fx
        move = "apreciación" if fx1m > 0 else "depreciación"
        rel = "por encima" if fx_vs_avg > 0 else "por debajo"
        lines.append(f"  • Rendimiento FX último mes: {fx1m:+.2f}% vs USD ({move}) — {abs(fx_vs_avg):.2f}pp {rel} del promedio de las 8 divisas principales ({avg_fx:+.2f}%)")

    available = sum(1 for v in [rate, rate_mom, inflation, gdp_growth, unemployment, wage_growth,
                                 retail_sales, ca, tb, tot, pmi, prod, debt, bond, cc, bc, ie, cf, cot, fx1m]
                    if v is not None)
    lines.append(f"\n[{available} indicadores disponibles | Datos a: {str(data.get('lastUpdate', 'N/D'))[:10]}]")

    if export_composition:
        lines.append("")
        lines.append(f"COMPOSICIÓN EXPORTADORA VERIFICADA:")
        lines.append(f"  {export_composition}")
        lines.append("  (Dato objetivo: úsalo para explicar el origen del superávit/déficit comercial sin especular)")

    if global_context:
        lines.append("")
        lines.append("CONTEXTO GLOBAL — PROMEDIOS DE LAS 8 DIVISAS PRINCIPALES:")
        mappings = [
            ('avg_interest_rate', 'Tasa de interés promedio', 2, '%'),
            ('avg_gdp_growth',    'Crecimiento PIB promedio',  1, '% anual'),
            ('avg_inflation',     'Inflación promedio',        1, '%'),
            ('avg_unemployment',  'Desempleo promedio',        1, '%'),
            ('avg_bond10y',       'Yield bono 10Y promedio',   2, '%'),
            ('avg_fx_perf_1m',    'Rendimiento FX 1M promedio', 2, '%'),
            ('avg_cot',           'COT promedio',               0, ' contratos netos'),
        ]
        for key, label, decimals, suffix in mappings:
            val = global_context.get(key)
            if val is not None:
                lines.append(f"  • {label}: {val:.{decimals}f}{suffix}")

    if global_context:
        signals = infer_structural_signals(currency, data, global_context)
        if signals:
            lines.append("")
            lines.append("SEÑALES ESTRUCTURALES INFERIDAS (úsalas para enriquecer el análisis):")
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
                    f"Recuerda: 3 párrafos con línea en blanco entre ellos, 150-200 palabras en total. "
                    f"El objetivo es interpretar las causas y consecuencias de los datos, "
                    f"no simplemente listarlos. Redacción en español natural y fluido, "
                    f"como un análisis de mercado profesional hispanohablante."
                ),
            },
        ],
        "max_tokens": 700,
        "temperature": 0.5,
        "top_p": 0.9,
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
            if word_count < 80:
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
    print(f"🤖 Generador AI v2.6 — {GROQ_MODEL} via Groq API")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "❌ GROQ_API_KEY no configurada. "
            "Agrégala en Settings → Secrets → Actions del repo."
        )

    print(f"✅ GROQ_API_KEY configurada ({len(api_key)} caracteres)")
    print(f"🔧 Modelo: {GROQ_MODEL}")
    print(f"📊 Decisiones monetarias: historial propio rates/{{currency}}.json (sin APIs externas)")
    print(f"💱 FX Performance: Frankfurter API (ECB, sin key) con fallback estático\n")

    print("🔍 Testeando conectividad...")
    for label, url in [
        ("Internet",        "https://httpbin.org/get"),
        ("GitHub Pages",    "https://globalinvesting.github.io/economic-data/USD.json"),
        ("Frankfurter API", "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD"),
        ("UN Comtrade",     "https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=36&period=2022&partnerCode=0&cmdCode=TOTAL&flowCode=X&customsCode=C00&motCode=0"),
    ]:
        try:
            r = requests.get(url, timeout=8)
            print(f"  ✅ {label} OK ({r.status_code})")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("📥 Cargando datos económicos + decisiones monetarias del historial...")
    all_data = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        all_data[currency] = load_economic_data(currency)
        available = sum(1 for k, v in all_data[currency].items()
                        if v is not None and k not in ('lastRateDecision', 'fxSource'))
        fx_src = all_data[currency].get('fxSource', '?')
        print(f"{available} indicadores | FX:{fx_src}")

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
                    "interestRate":      data.get('interestRate'),
                    "gdpGrowth":         data.get('gdpGrowth'),
                    "inflation":         data.get('inflation'),
                    "unemployment":      data.get('unemployment'),
                    "currentAccount":    data.get('currentAccount'),
                    "rateMomentum":      data.get('rateMomentum'),
                    "lastRateDecision":  data.get('lastRateDecision'),
                    "cotPositioning":    data.get('cotPositioning'),
                    "fxPerformance1M":   data.get('fxPerformance1M'),
                    "fxSource":          data.get('fxSource'),
                    "lastUpdate":        data.get('lastUpdate'),
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
        "version":          "2.6",
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
