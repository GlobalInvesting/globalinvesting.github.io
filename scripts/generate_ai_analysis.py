#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.

v2.2 — Composición exportadora obtenida en vivo desde UN Comtrade API.
       Fallback a HS2 codes estáticos si la API no responde.
       Sin hardcoding de narrativas: los datos determinan el análisis.
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

# ─── UN Comtrade reporter codes ───────────────────────────────────────────────
# Estándar ISO 3166-1 numeric — no cambian, son códigos oficiales de la ONU
COMTRADE_REPORTER_CODES = {
    'AUD': '36',    # Australia
    'CAD': '124',   # Canada
    'NZD': '554',   # New Zealand
    'CHF': '756',   # Switzerland
    'USD': '842',   # United States
    'EUR': '276',   # Germany como proxy de Eurozona (mayor exportador del bloque)
    'GBP': '826',   # United Kingdom
    'JPY': '392',   # Japan
}

# ─── Traducción de códigos HS2 a español ──────────────────────────────────────
# El Harmonized System (HS) es un estándar internacional — los códigos no cambian.
# Esto traduce códigos objetivos a términos legibles. NO es una descripción
# de qué exporta cada país — eso lo determina la API en tiempo real.
HS2_NAMES_ES = {
    '01': 'animales vivos',
    '02': 'carne y despojos comestibles',
    '03': 'pescado y crustáceos',
    '04': 'leche, productos lácteos y huevos',
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
    '17': 'azúcares y artículos de confitería',
    '18': 'cacao y sus preparaciones',
    '19': 'preparaciones a base de cereales',
    '20': 'preparaciones de hortalizas o frutas',
    '21': 'preparaciones alimenticias diversas',
    '22': 'bebidas y líquidos alcohólicos',
    '23': 'residuos de industrias alimentarias',
    '24': 'tabaco y sucedáneos',
    '25': 'sal, azufre, piedras y cementos',
    '26': 'minerales metalíferos y concentrados',
    '27': 'combustibles minerales y petróleo',
    '28': 'productos químicos inorgánicos',
    '29': 'productos químicos orgánicos',
    '30': 'productos farmacéuticos',
    '31': 'abonos',
    '32': 'extractos curtientes y colorantes',
    '33': 'aceites esenciales y perfumería',
    '34': 'jabones y agentes de superficie',
    '35': 'materias albuminoideas y almidones',
    '36': 'pólvoras y explosivos',
    '37': 'productos fotográficos y cinematográficos',
    '38': 'productos diversos de industrias químicas',
    '39': 'plásticos y manufacturas',
    '40': 'caucho y manufacturas',
    '41': 'pieles y cueros en bruto',
    '42': 'manufacturas de cuero',
    '43': 'peletería y confecciones',
    '44': 'madera y manufacturas de madera',
    '45': 'corcho y manufacturas',
    '46': 'manufacturas de cestería',
    '47': 'pasta de madera y papel reciclado',
    '48': 'papel, cartón y manufacturas',
    '49': 'productos editoriales e impresos',
    '50': 'seda',
    '51': 'lana y pelo animal',
    '52': 'algodón',
    '53': 'otras fibras textiles vegetales',
    '54': 'filamentos sintéticos o artificiales',
    '55': 'fibras sintéticas o artificiales discontinuas',
    '56': 'guata, fieltro y artículos no tejidos',
    '57': 'alfombras y revestimientos textiles',
    '58': 'tejidos especiales',
    '59': 'tejidos impregnados o recubiertos',
    '60': 'tejidos de punto',
    '61': 'prendas y complementos de punto',
    '62': 'prendas y complementos excepto de punto',
    '63': 'demás artículos textiles confeccionados',
    '64': 'calzado y partes',
    '65': 'sombreros y tocados',
    '66': 'paraguas y bastones',
    '67': 'plumas preparadas y artículos',
    '68': 'manufacturas de piedra y yeso',
    '69': 'productos cerámicos',
    '70': 'vidrio y manufacturas',
    '71': 'piedras preciosas y metales',
    '72': 'fundición, hierro y acero',
    '73': 'manufacturas de fundición y acero',
    '74': 'cobre y manufacturas',
    '75': 'níquel y manufacturas',
    '76': 'aluminio y manufacturas',
    '77': 'reservado (futuro uso HS)',
    '78': 'plomo y manufacturas',
    '79': 'cinc y manufacturas',
    '80': 'estaño y manufacturas',
    '81': 'demás metales comunes',
    '82': 'herramientas y cuchillería metálica',
    '83': 'manufacturas diversas de metal común',
    '84': 'maquinaria y equipos mecánicos',
    '85': 'maquinaria eléctrica y electrónica',
    '86': 'vehículos y material ferroviario',
    '87': 'vehículos automóviles y partes',
    '88': 'aeronaves y partes',
    '89': 'barcos y embarcaciones',
    '90': 'instrumentos de óptica y precisión',
    '91': 'relojes y aparatos de relojería',
    '92': 'instrumentos musicales',
    '93': 'armas y municiones',
    '94': 'muebles y artículos de cama',
    '95': 'juguetes y artículos de deporte',
    '96': 'manufacturas diversas',
    '97': 'objetos de arte y antigüedades',
    '99': 'mercancías no clasificadas',
}

# ─── Fallback estático (HS2 codes, no narrativas) ─────────────────────────────
# Solo se usa si la API de Comtrade falla completamente.
# Son códigos objetivos del HS — si la composición exportadora cambia
# significativamente, actualizar estos códigos (no el texto).
# Fuente de referencia: OEC / UN Comtrade 2022-2023.
EXPORT_FALLBACK_HS2 = {
    'AUD': ['26', '27', '10', '12'],   # Minerales, combustibles, cereales, semillas
    'CAD': ['27', '87', '84', '26'],   # Combustibles, vehículos, maquinaria, minerales
    'NZD': ['04', '02', '10', '03'],   # Lácteos, carne, cereales, pescado
    'CHF': ['30', '90', '71', '84'],   # Farmacéuticos, instrumentos, metales preciosos, maquinaria
    'USD': ['84', '85', '88', '30'],   # Maquinaria, electrónica, aeronaves, farmacéuticos
    'EUR': ['84', '87', '85', '30'],   # Maquinaria, vehículos, electrónica, farmacéuticos
    'GBP': ['84', '30', '85', '88'],   # Maquinaria, farmacéuticos, electrónica, aeronaves
    'JPY': ['87', '84', '85', '90'],   # Vehículos, maquinaria, electrónica, instrumentos
}

GITHUB_BASE = 'https://globalinvesting.github.io'
OUTPUT_DIR  = Path('ai-analysis')
GROQ_MODEL  = 'llama-3.3-70b-versatile'
GROQ_URL    = 'https://api.groq.com/openai/v1/chat/completions'

# Cache para no hacer múltiples requests a Comtrade en el mismo run
_export_cache = {}


# ─── SYSTEM PROMPT v2.2 ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de forex institucional senior. Escribes research notes para traders profesionales.

TAREA: Generar un análisis fundamental interpretativo de la divisa indicada, en español.

FORMATO:
- Texto corrido, sin bullets, sin títulos, sin markdown
- 3 párrafos separados por línea en blanco
- Entre 130 y 170 palabras totales

ESTRUCTURA:
1. Política monetaria e inflación: ¿Qué está haciendo el banco central y POR QUÉ?
   REGLA ESTRICTA para el campo "Momentum de Tasas":
   - Valor NEGATIVO = el BC está RECORTANDO tasas → ciclo dovish → presión bajista sobre la divisa
   - Valor POSITIVO = el BC está SUBIENDO tasas → ciclo hawkish → presión alcista
   - Valor cero o cercano a cero = pausa. No inviertas esta lógica bajo ninguna circunstancia.
   Compara la tasa actual vs el promedio global del contexto.

2. Fundamentos macroeconómicos: interpreta el PIB, empleo y balanza exterior.
   Si el input incluye "Composición exportadora (Comtrade)", úsala para afirmar con precisión
   qué sectores impulsan el superávit o déficit comercial. No especules sobre el sector exportador
   si no tienes el dato — solo afirma lo que el dato confirma.
   Usa las "Señales estructurales inferidas" para enriquecer el contexto.

3. Sentimiento e impulso de mercado: interpreta el COT (¿qué dice el posicionamiento institucional?)
   y el rendimiento FX del último mes (¿por qué se movió así?). Cierra con perspectivas a corto plazo.

REGLAS CRÍTICAS:
- INTERPRETA, no listes. El objetivo es explicar POR QUÉ los datos importan, no solo QUÉ dicen.
- Usa el "Contexto global" para hacer comparaciones relativas concretas.
- Si el COT supera ±30K contratos, señala el riesgo de saturación/reversión o la convicción institucional.
- Si el FX 1M supera ±3%, explica el catalizador probable (carry, commodities, risk-on/off, diferencial de tasas).
- El campo "PIB Total" es el tamaño de la economía en trillones USD — NO es la tasa de crecimiento.
- La tasa de crecimiento del PIB es el campo "Crecimiento PIB (% anual)".
- Para USD: el "Rendimiento FX 1M" refleja el índice dólar (DXY) — redáctalo como "el índice dólar subió/cayó X%".
- Si un indicador no tiene dato, NO lo menciones ni lo infieras desde otros indicadores.
  EJEMPLOS DE ERRORES A EVITAR:
  · Si no hay dato de Balanza Comercial, NO inferir déficit/superávit desde la Cuenta Corriente.
  · Si no hay dato de exposición a commodities, NO inferirla desde el contexto de otras divisas.
  · El signo del Rendimiento FX ya viene interpretado en el dato — NO lo inviertas.
- Tono: directo, analítico, como un research note de Goldman Sachs o JP Morgan. Sin saludos ni meta-comentarios."""


def fetch_json(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None


def fetch_export_composition(currency):
    """
    Obtiene la composición exportadora del país desde UN Comtrade API pública.
    Devuelve una string descriptiva con los top sectores exportadores.
    Si la API falla, usa el fallback de HS2 codes estáticos.
    
    Fuente: UN Comtrade Public Preview API (sin autenticación, gratuita).
    Datos: último año disponible (generalmente 2022-2023).
    """
    if currency in _export_cache:
        return _export_cache[currency]

    reporter_code = COMTRADE_REPORTER_CODES.get(currency)
    if not reporter_code:
        return None

    def _parse_comtrade_response(data, year_label):
        """
        Parsea la respuesta de Comtrade.
        La API pública devuelve HS6 (6 dígitos) — truncamos a HS2 y agregamos por capítulo.
        """
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

        # Agregar valores por capítulo HS2 (primeros 2 dígitos del código HS4/HS6)
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
            return f"Top exportaciones — Comtrade {year_label}: {', '.join(sectors)}"
        return None

    # ── Intento: Comtrade sin cmdCode, año 2023 luego 2022 ───────────────────
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

    # ── Fallback: HS2 codes estáticos ─────────────────────────────────────
    fallback_codes = EXPORT_FALLBACK_HS2.get(currency, [])
    if fallback_codes:
        sectors = [HS2_NAMES_ES.get(code, f'HS{code}') for code in fallback_codes[:3]]
        result = f"Top exportaciones {currency}: {', '.join(sectors)}"
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
    Sin hardcoding de narrativas: los datos determinan las señales.
    """
    signals = []
    avg_rate = global_context.get('avg_interest_rate') or 3.0

    rate         = data.get('interestRate')
    rate_momentum = data.get('rateMomentum')
    current_account = data.get('currentAccount')
    trade_balance = data.get('tradeBalance')
    debt         = data.get('debt')
    cot          = data.get('cotPositioning')
    fx_1m        = data.get('fxPerformance1M')
    terms_of_trade = data.get('termsOfTrade')
    avg_cot      = global_context.get('avg_cot') or 0

    # ── Carry trade / safe haven por diferencial de tasas ─────────────────
    if rate is not None and avg_rate is not None:
        rate_gap = avg_rate - rate
        if rate_gap >= 1.5:
            signals.append(
                f"Tasa muy por debajo del promedio global ({rate:.2f}% vs {avg_rate:.2f}% promedio): "
                f"divisa probable financiadora de carry trade. "
                f"Su apreciación en episodios de risk-off puede ser brusca por desmantelamiento de posiciones cortas."
            )
        elif rate - avg_rate >= 1.5:
            signals.append(
                f"Tasa significativamente superior al promedio global ({rate:.2f}% vs {avg_rate:.2f}% promedio): "
                f"la divisa ofrece carry atractivo y puede captar flujos de inversión de renta fija internacional."
            )

    # ── Ciclo del banco central ────────────────────────────────────────────
    if rate_momentum is not None:
        if rate_momentum > 0.5:
            signals.append(
                f"Momentum de tasas positivo ({rate_momentum:+.2f}% en 12 meses): "
                f"banco central en ciclo de normalización — structuralmente alcista para la divisa."
            )
        elif rate_momentum < -0.5:
            signals.append(
                f"Momentum de tasas negativo ({rate_momentum:+.2f}% en 12 meses): "
                f"banco central recortando activamente — presiona a la baja la divisa vía reducción del carry."
            )

    # ── Safe haven por balance exterior + deuda baja ──────────────────────
    ca_surplus = current_account is not None and current_account > 2.0
    low_debt   = debt is not None and debt < 60
    low_rate   = rate is not None and avg_rate is not None and (avg_rate - rate) >= 1.5

    if ca_surplus and low_debt and low_rate:
        signals.append(
            f"Perfil safe haven detectado: superávit de cuenta corriente ({current_account:.1f}% PIB), "
            f"deuda baja ({debt:.0f}% PIB) y tasa por debajo del promedio global. "
            f"Esta combinación genera demanda defensiva de la divisa en episodios de aversión al riesgo."
        )
    elif ca_surplus and low_debt:
        signals.append(
            f"Superávit de cuenta corriente ({current_account:.1f}% PIB) y deuda controlada ({debt:.0f}% PIB): "
            f"balance exterior sólido que genera demanda estructural de la divisa."
        )

    # ── Balance comercial + términos de intercambio ───────────────────────
    # NOTA: Si fetch_export_composition() tuvo éxito, el modelo ya tiene el dato
    # concreto de qué sectores exporta el país — estas señales son complementarias.
    if trade_balance is not None and terms_of_trade is not None:
        if trade_balance > 2000 and terms_of_trade > 100:
            signals.append(
                f"Superávit comercial de {trade_balance/1000:.1f}B USD/mes con términos de intercambio "
                f"favorables (índice {terms_of_trade:.1f} sobre base 100): "
                f"los precios de exportación superan a los de importación, lo que amplifica "
                f"el beneficio del superávit y genera demanda adicional de la divisa."
            )
        elif trade_balance < -15000:
            signals.append(
                f"Déficit comercial pronunciado ({trade_balance/1000:.1f}B USD/mes): "
                f"presión vendedora estructural sobre la divisa compensada por flujos financieros."
            )
    elif trade_balance is not None and trade_balance > 2000:
        signals.append(
            f"Superávit comercial de {trade_balance/1000:.1f}B USD/mes: "
            f"genera demanda estructural de la divisa por conversión de ingresos de exportación."
        )

    # ── COT extremo ────────────────────────────────────────────────────────
    if cot is not None:
        if cot > 30000:
            signals.append(
                f"Posicionamiento especulativo alcista extremo ({cot/1000:.0f}K contratos netos): "
                f"mercado saturado — riesgo elevado de toma de ganancias si los datos decepcionan."
            )
        elif cot < -30000:
            signals.append(
                f"Posicionamiento especulativo bajista extremo ({cot/1000:.0f}K contratos netos): "
                f"mercado muy corto — potencial rebote contrarian si los fundamentales mejoran."
            )

    # ── Divergencia COT vs FX ──────────────────────────────────────────────
    if cot is not None and fx_1m is not None:
        if cot < -20000 and fx_1m > 2.5:
            signals.append(
                f"Divergencia: posicionamiento bajista ({cot/1000:.0f}K) pero apreciación del {fx_1m:.1f}% "
                f"en el último mes — posible squeeze de cortos en curso."
            )
        elif cot > 20000 and fx_1m < -2.0:
            signals.append(
                f"Divergencia: posicionamiento alcista ({cot/1000:.0f}K) pero depreciación del {fx_1m:.1f}% "
                f"— posible inicio de liquidación de largos."
            )

    # ── Régimen global ─────────────────────────────────────────────────────
    if avg_cot is not None:
        if avg_cot > 8000:
            signals.append(
                "Régimen global: risk-on (COT promedio positivo entre las 8 divisas). "
                "Divisas de alto carry y commodities tienden a outperform."
            )
        elif avg_cot < -8000:
            signals.append(
                "Régimen global: risk-off (COT promedio negativo entre las 8 divisas). "
                "Divisas safe haven y de bajo carry tienden a apreciarse."
            )

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
         lambda v: fmt(v, 2, '% — NEGATIVO=recortando, POSITIVO=subiendo')),
        ('inflation',            'Inflación (IPC)',
         lambda v: fmt(v, 1, '% anual')),
        ('inflationExpectations', 'Expect. Inflación',
         lambda v: fmt(v, 1, '%')),
        ('unemployment',         'Desempleo',
         lambda v: fmt(v, 1, '%')),
        ('currentAccount',       'Cuenta Corriente (% PIB)',
         lambda v: (
             f"{v:.1f}% PIB — {'SUPERÁVIT' if v > 0 else 'DÉFICIT'} "
             f"(≠ balanza comercial; incluye servicios, rentas y transferencias)"
         ) if v is not None else None),
        ('debt',                 'Deuda Pública',
         lambda v: fmt(v, 1, '% PIB')),
        ('tradeBalance',         'Balanza Comercial (bienes)',
         lambda v: (
             f"{v/1000:.1f}B USD/mes — {'SUPERÁVIT' if v > 0 else 'DÉFICIT'} comercial en bienes"
         ) if v is not None else None),
        ('production',           'Producción Industrial',
         lambda v: fmt(v, 1, '% MoM')),
        ('retailSales',          'Ventas Minoristas',
         lambda v: fmt(v, 1, '% MoM')),
        ('wageGrowth',           'Crecimiento Salarial',
         lambda v: fmt(v, 1, '% anual')),
        ('manufacturingPMI',     'PMI Manufacturero',
         lambda v: fmt(v, 1, ' (>50=expansión, <50=contracción)')),
        ('termsOfTrade',         'Términos de Intercambio',
         lambda v: fmt(v, 1, ' (base 100, >100=precios exportación > importación)')),
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
         lambda v: (
             f"{v:+.2f}% vs USD — {'APRECIACIÓN' if v > 0 else 'DEPRECIACIÓN'} "
             f"({'divisa se fortaleció' if v > 0 else 'divisa se debilitó'} frente al USD)"
         ) if v is not None else None),
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

    # ── Composición exportadora (dato en vivo de Comtrade) ─────────────────
    if export_composition:
        lines.append("")
        lines.append("COMPOSICIÓN EXPORTADORA (dato verificado):")
        lines.append(f"  • {export_composition}")
        lines.append("  (Usa este dato para afirmar con precisión qué sectores impulsan el balance comercial)")

    # ── Contexto global ────────────────────────────────────────────────────
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

    # ── Señales estructurales inferidas ────────────────────────────────────
    if global_context:
        signals = infer_structural_signals(currency, data, global_context)
        if signals:
            lines.append("")
            lines.append("SEÑALES ESTRUCTURALES INFERIDAS DESDE LOS DATOS:")
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
                    f"Genera el análisis fundamental para {currency}. "
                    f"PROHIBIDO listar datos. OBLIGATORIO explicar causas y consecuencias. "
                    f"Si tienes composición exportadora verificada, úsala para afirmar, no especular. "
                    f"Usa las señales estructurales y el contexto global para contextualizar cada punto:"
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


def generate_analysis(api_key, currency, data, global_context=None, export_composition=None):
    data_summary = build_data_summary(currency, data, global_context, export_composition)

    for attempt in range(3):
        try:
            text = call_groq_api(api_key, data_summary, currency)

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
    print(f"🤖 Generador AI v2.2 — {GROQ_MODEL} via Groq API")
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

    # ── Paso 1: cargar todos los datos económicos ──────────────────────────
    print("📥 Cargando datos económicos de las 8 divisas...")
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

    # ── Paso 2: obtener composición exportadora (Comtrade) ─────────────────
    print("📦 Obteniendo composición exportadora (UN Comtrade)...")
    export_compositions = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        composition = fetch_export_composition(currency)
        export_compositions[currency] = composition
        if not composition:
            print("sin dato")
    print()

    # ── Paso 3: generar análisis ───────────────────────────────────────────
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

    # ── Índice general ─────────────────────────────────────────────────────
    successful = [c for c, r in results.items() if r.get('success')]
    comtrade_count = sum(
        1 for r in results.values()
        if r.get('exportDataSource') == 'comtrade'
    )

    index = {
        "generatedAt":      datetime.now(timezone.utc).isoformat(),
        "model":            GROQ_MODEL,
        "version":          "2.2",
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
