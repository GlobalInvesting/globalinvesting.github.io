#!/usr/bin/env python3
"""
generate_ai_analysis.py
Genera análisis fundamentales de divisas forex usando Groq API (gratuita).
Modelo: llama-3.3-70b-versatile — sin SDK, solo requests.

v5.0 — Integración de noticias recientes en la narrativa:
       • get_relevant_news(): filtra artículos de alto/medio impacto
         para cada divisa desde news-data/news.json (generado por fetch_news.py)
       • format_news_for_prompt(): serializa titulares en formato compacto
         orientado a argumento, sin nombres de fuentes
       • System prompt actualizado: las noticias se tejen DENTRO de los
         párrafos del análisis, no como sección separada
       • news_has_changed(): fuerza regeneración cuando llegan noticias
         nuevas de alto impacto no contempladas en el snapshot previo
       • dataSnapshot ampliado con latestNewsTs y newsHeadlines para
         detectar cambios noticiosos en el pre-check de caché
       • TTL reducido a 12h por defecto (era 24h) dado que las noticias
         de alto impacto pueden cambiar el contexto en pocas horas

v4.0 — Patch de caché inteligente sobre v2.6:
       • load_previous_analysis(): carga análisis guardado para una divisa
       • data_has_changed(): compara dataSnapshot anterior vs datos actuales
       • TTL de 24h: fuerza regeneración diaria aunque los datos no cambien
       • main() con soporte de múltiples keys (GROQ_API_KEY + GROQ_API_KEY_2)
         y fallback automático al agotar el límite diario de una key
       • Reporte detallado: divisas regeneradas vs reutilizadas
"""

import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# FIX R-03: Eliminado socket.setdefaulttimeout(15) global.
# Cada llamada a requests usa su propio parámetro timeout, lo cual es suficiente
# y evita interferir con conexiones que necesitan más tiempo (ej: Groq con 40s).

# FIX C-01: Configuración de divisas centralizada en fx_config.py.
# Ya no se duplican CURRENCIES ni COUNTRY_META en cada script.
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(__file__))
from fx_config import CURRENCIES, COUNTRY_META

# ─── Frankfurter API — FX en tiempo real (ECB rates, sin key) ────────────────
FRANKFURTER_BASE = 'https://api.frankfurter.dev/v1'

# Cache para evitar llamadas repetidas en el mismo run
_fx_performance_cache = {}

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

GITHUB_BASE  = 'https://globalinvesting.github.io'
OUTPUT_DIR   = Path('ai-analysis')
NEWS_PATH    = Path('news-data/news.json')   # generado por fetch_news.py
GROQ_MODEL   = 'llama-3.3-70b-versatile'
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'
KEY_SWITCH_PAUSE = 5  # segundos entre cambio de key

# TTL máximo de un análisis aunque los datos macro no cambien.
# Reducido a 12h en v5.0: las noticias de alto impacto pueden cambiar
# el contexto de mercado en pocas horas, especialmente en días de eventos.
MAX_ANALYSIS_AGE_HOURS = 12

_export_cache = {}


# ─── SYSTEM PROMPT v5.0 ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de mercados de divisas senior que redacta comentarios de mercado para una plataforma de trading profesional. Escribes en español de forma nativa, fluida y analítica — como un profesional de habla hispana que lleva años cubriendo los mercados cambiarios, no como alguien que traduce del inglés.

TAREA: Redactar un análisis fundamental de la divisa indicada a partir de los datos económicos y el contexto noticioso proporcionados.

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
- El campo "PIB Total" es el tamaño absoluto de la economía en trillones USD, no la tasa de crecimiento

INTEGRACIÓN DE NOTICIAS RECIENTES — REGLAS DE REDACCIÓN:
Las noticias se proporcionan como CONTEXTO_NOTICIOSO_DIVISA.
No son un apartado separado. Son evidencia de mercado que debes tejer
dentro de los tres párrafos del análisis cuando sea relevante.

Cómo integrarlas por párrafo:
- Párrafo 1 (política monetaria): si hay una noticia sobre el banco central
  o sobre expectativas de tasas, incorpórala como evidencia que confirma o
  cuestiona la postura del BC. Ejemplo: "Los mercados de futuros ya
  descuentan un recorte adicional en marzo, coherente con las declaraciones
  del gobernador que anticipan aproximadamente 100pb de bajadas este año."
- Párrafo 2 (balanza/sector): si hay noticias sobre commodities, energía,
  comercio exterior o datos sectoriales, úsalas para enriquecer el
  argumento sobre la balanza comercial o la cuenta corriente.
- Párrafo 3 (posicionamiento/perspectivas): si hay una noticia que
  contradice o refuerza el COT o el rendimiento FX reciente, intégrala
  como catalizador de corto plazo dentro del argumento.

Reglas estrictas para la integración de noticias:
1. No cites la fuente ni el nombre del medio en el texto final.
2. No uses la fórmula "según X" ni "de acuerdo con Y".
3. Integra preferentemente noticias de impacto ALTO. Las de impacto MEDIO
   solo si son directamente relevantes y no hay noticias de alto impacto.
4. Si una noticia contradice el análisis macro, señala explícitamente
   la divergencia: "aunque los datos estructurales apuntan a X, el mercado
   reaccionó esta semana a Y, lo que introduce incertidumbre de corto plazo."
5. Si no hay noticias relevantes, no lo menciones en absoluto. Escribe
   el análisis macro puro sin referencias a la ausencia de noticias.
6. Máximo dos noticias integradas en todo el análisis. Elige las de
   mayor impacto y mayor coherencia con la narrativa macro.
7. El tono sigue siendo institucional. El resultado debe parecer que el
   analista conoce el contexto de mercado, no un resumen de titulares."""


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def fetch_json(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠️  No se pudo cargar {url}: {e}")
        return None


def load_groq_keys():
    """Carga todas las keys de Groq disponibles en variables de entorno."""
    keys = []
    for var in ['GROQ_API_KEY', 'GROQ_API_KEY_2', 'GROQ_API_KEY_3']:
        val = os.environ.get(var, '').strip()
        if val:
            keys.append(val)
    return keys


def mask_key(key):
    """FIX S-02: Muestra solo los primeros 4 y últimos 4 caracteres de la key.
    Reducido desde key[:4] para no exponer demasiado de keys tipo 'gsk_XXXX...'."""
    if len(key) <= 8:
        return '****'
    return f"{key[:4]}...{key[-4:]}"


def check_groq_key(key):
    """
    Verifica rápidamente si una key es válida y no ha alcanzado el límite diario.
    Retorna: 'ok', 'daily_limit', 'invalid', 'rate_limit'

    FIX S-01: Los errores de autenticación (401) se manejan silenciosamente
    sin loguear detalles que podrían exponer información en logs públicos de CI.
    """
    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=10,
        )
        if r.status_code == 401:
            # FIX S-01: No logueamos el status code ni el body para evitar
            # exponer información de autenticación en logs públicos de GitHub Actions.
            return 'invalid'
        if r.status_code == 429:
            # Inspeccionamos el body solo para distinguir daily_limit vs rate_limit,
            # sin imprimirlo en ningún caso.
            try:
                body = r.json().get('error', {}).get('message', '').lower()
            except Exception:
                body = r.text.lower()
            if 'daily' in body or 'quota' in body or ('per day' in body):
                return 'daily_limit'
            return 'rate_limit'
        return 'ok'
    except Exception:
        return 'ok'


# ─── NOTICIAS RECIENTES v5.0 ─────────────────────────────────────────────────

def load_news_file() -> dict:
    """
    Carga news-data/news.json desde disco (generado por fetch_news.py).
    Retorna el dict completo o {} si no existe o está corrupto.
    """
    if not NEWS_PATH.exists():
        print(f"  ⚠️  {NEWS_PATH} no encontrado — análisis sin contexto noticioso")
        return {}
    try:
        with open(NEWS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        age_h = ""
        updated = data.get("updated_utc")
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                h  = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                age_h = f" (actualizado hace {h:.1f}h)"
            except Exception:
                pass
        total = len(data.get("articles", []))
        print(f"  📰 news.json cargado: {total} artículos{age_h}")
        return data
    except Exception as e:
        print(f"  ⚠️  Error cargando {NEWS_PATH}: {e}")
        return {}


def get_relevant_news(currency: str, news_data: dict, max_items: int = 3) -> list:
    """
    v5.1 — Sin enrich_news: usa title directamente, solo impact == "high".
    Eliminados los filtros de ai_headline y sentiment (ya no existen).
    """
    articles = news_data.get("articles", [])
    if not articles:
        return []

    relevant = [
        a for a in articles
        if a.get("cur") == currency
        and a.get("impact") == "high"       # solo high (antes "high" o "med")
        and a.get("title", "").strip()      # requiere título no vacío
    ]

    relevant.sort(key=lambda x: -(x.get("ts") or 0))
    return relevant[:max_items]


def format_news_for_prompt(news_items: list, currency: str) -> str:
    """
    v5.1 — Usa title directo + expand como contexto adicional.
    Eliminado SENT_MAP (ya no hay sentiment disponible).
    """
    if not news_items:
        return f"Sin noticias de alto impacto para {currency} en las últimas 24h."

    lines = []
    for a in news_items:
        title  = (a.get("title") or "").strip()
        expand = (a.get("expand") or "").strip()

        age_label = ""
        ts = a.get("ts")
        if ts:
            h = (time.time() - ts / 1000) / 3600
            age_label = "<1h" if h < 1 else (f"{h:.0f}h" if h < 24 else f"{h/24:.0f}d")

        line = f"  • [HIGH·{age_label}] {title}"

        # Añadir descripción si aporta contexto más allá del título
        if expand and len(expand.split()) > 10 and not expand.lower().startswith(title.lower()[:40]):
            snippet = expand[:150].strip()
            if not snippet.endswith((".", "…")):
                snippet += "…"
            line += f"\n    → {snippet}"

        lines.append(line)

    return "\n".join(lines)


def news_has_changed(currency: str, prev_analysis: dict, current_news: list) -> bool:
    """
    Retorna True si hay noticias de alto impacto nuevas que no estaban
    en el snapshot anterior — fuerza regeneración sin esperar al TTL.

    Compara timestamps: si el ts máximo de las noticias actuales de alto
    impacto es posterior al latestNewsTs guardado, hay novedades.
    """
    if prev_analysis is None:
        return True

    prev_ts = prev_analysis.get("dataSnapshot", {}).get("latestNewsTs", 0)

    high_news = [a for a in current_news if a.get("impact") == "high"]
    if not high_news:
        return False  # sin noticias de alto impacto → no forzar regeneración

    max_ts = max((a.get("ts", 0) or 0) for a in high_news) / 1000  # ms → s
    return max_ts > prev_ts


# ─── CACHÉ INTELIGENTE v5.0 ──────────────────────────────────────────────────

def load_previous_analysis(currency: str) -> dict | None:
    """
    Carga el análisis JSON guardado previamente para una divisa.
    Retorna el dict completo o None si no existe o está corrupto.
    """
    path = OUTPUT_DIR / f"{currency}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Campos del dataSnapshot que se comparan para detectar cambios en datos macro.
SNAPSHOT_COMPARE_KEYS = [
    "interestRate",
    "gdpGrowth",
    "inflation",
    "unemployment",
    "currentAccount",
    "rateMomentum",
    "lastRateDecision",
    "cotPositioning",
    "fxPerformance1M",
]


def data_has_changed(currency: str, current_data: dict, prev_analysis: dict | None) -> bool:
    """
    Retorna True si:
    1. No hay análisis previo
    2. El análisis tiene más de MAX_ANALYSIS_AGE_HOURS horas (TTL)
    3. Algún indicador macro clave difiere del snapshot guardado
    4. [v5.0] Hay noticias de alto impacto nuevas (se evalúa por separado
       en news_has_changed() y se combina en main())
    """
    if prev_analysis is None:
        return True

    # TTL
    generated_at = prev_analysis.get("generatedAt")
    if generated_at:
        try:
            gen_dt    = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
            if age_hours > MAX_ANALYSIS_AGE_HOURS:
                print(f"    ⏰ TTL expirado ({age_hours:.0f}h > {MAX_ANALYSIS_AGE_HOURS}h) → regenerar")
                return True
        except Exception:
            return True
    else:
        return True

    # Snapshot de datos macro
    prev_snapshot = prev_analysis.get("dataSnapshot", {})
    if not prev_snapshot:
        return True

    for key in SNAPSHOT_COMPARE_KEYS:
        curr_val = current_data.get(key)
        prev_val = prev_snapshot.get(key)

        if curr_val is None and prev_val is None:
            continue
        if (curr_val is None) != (prev_val is None):
            return True

        if key == "lastRateDecision":
            if json.dumps(curr_val, sort_keys=True) != json.dumps(prev_val, sort_keys=True):
                return True
            continue

        if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
            if abs(float(curr_val) - float(prev_val)) > 0.001:
                return True
            continue

        if str(curr_val).strip() != str(prev_val).strip():
            return True

    return False


# ─── DATOS DE TASAS ──────────────────────────────────────────────────────────

def derive_rate_decision_from_history(currency: str, observations: list) -> dict | None:
    """
    Deriva la decisión real del banco central analizando el historial de tasas
    almacenado en rates/{currency}.json.
    """
    if not observations:
        return None

    parsed = []
    for obs in observations:
        val  = obs.get('value')
        date = obs.get('date', '')
        if val is None or val == '.':
            continue
        try:
            parsed.append({'value': float(val), 'date': date})
        except (ValueError, TypeError):
            continue

    if not parsed:
        return None

    parsed.sort(key=lambda x: x['date'], reverse=True)
    current_rate = parsed[0]['value']
    current_date = parsed[0]['date']

    prev_rate     = None
    decision_date = current_date
    for obs in parsed[1:]:
        if abs(obs['value'] - current_rate) > 0.01:
            prev_rate = obs['value']
            for i, o in enumerate(parsed):
                if abs(o['value'] - current_rate) > 0.01:
                    decision_date = parsed[i - 1]['date'] if i > 0 else current_date
                    break
            break

    if prev_rate is None:
        return {
            'direction':   'MANTUVO',
            'delta':       0.0,
            'current_rate': current_rate,
            'prev_rate':   current_rate,
            'date':        current_date,
            'source':      'rates_history',
            'cycle_12m':   0.0,
            'cycle_label': 'sin cambios en el período analizado',
        }

    delta     = round(current_rate - prev_rate, 4)
    direction = 'SUBIÓ' if delta > 0.01 else ('BAJÓ' if delta < -0.01 else 'MANTUVO')

    try:
        today    = datetime.strptime(current_date[:10], '%Y-%m-%d')
        cutoff   = today.replace(year=today.year - 1)
        cutoff_s = cutoff.strftime('%Y-%m-%d')
        within   = [o for o in parsed if o['date'] >= cutoff_s]
        cycle_12m = round(current_rate - within[-1]['value'], 4) if within else 0.0
    except Exception:
        cycle_12m = 0.0

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
    """Deprecado en v2.6 — mantenido por compatibilidad."""
    return None


# ─── FX PERFORMANCE ──────────────────────────────────────────────────────────

def fetch_frankfurter_fx(currency: str, days: int = 30) -> float | None:
    if currency in _fx_performance_cache:
        return _fx_performance_cache[currency]

    try:
        today = datetime.now(timezone.utc).date()
        past  = today - timedelta(days=days + 5)

        if currency == 'USD':
            url_now  = f"{FRANKFURTER_BASE}/latest?base=EUR&symbols=USD"
            url_past = f"{FRANKFURTER_BASE}/{past}?base=EUR&symbols=USD"
            now_data  = fetch_json(url_now,  timeout=8)
            past_data = fetch_json(url_past, timeout=8)
            if now_data and past_data:
                rate_now  = now_data['rates']['USD']
                rate_past = past_data['rates']['USD']
                pct = round((rate_past / rate_now - 1) * 100, 4)
                _fx_performance_cache[currency] = pct
                print(f"  💱 Frankfurter USD: {pct:+.2f}% (1M vs EUR proxy)")
                return pct
        else:
            symbols   = f"{currency},USD"
            url_now   = f"{FRANKFURTER_BASE}/latest?base=EUR&symbols={symbols}"
            url_past  = f"{FRANKFURTER_BASE}/{past}?base=EUR&symbols={symbols}"
            now_data  = fetch_json(url_now,  timeout=8)
            past_data = fetch_json(url_past, timeout=8)

            if now_data and past_data and currency in now_data.get('rates', {}):
                cur_now  = now_data['rates'][currency]
                usd_now  = now_data['rates']['USD']
                cur_past = past_data['rates'][currency]
                usd_past = past_data['rates']['USD']
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


# ─── COMPOSICIÓN EXPORTADORA ─────────────────────────────────────────────────

def fetch_export_composition(currency: str) -> str | None:
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

        sample      = commodities[0]
        code_field  = next(
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

        top3           = sorted(hs2_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        total_exports  = sum(hs2_totals.values())
        sectors        = []
        for code, value in top3:
            name    = HS2_NAMES_ES.get(code, f'HS{code}')
            pct     = (value / total_exports * 100) if total_exports > 0 else 0
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
        result  = f"Principales exportaciones: {', '.join(sectors)}"
        print(f"  📦 Fallback para {currency}: {result}")
        _export_cache[currency] = result
        return result

    _export_cache[currency] = None
    return None


# ─── CARGA DE DATOS ECONÓMICOS ───────────────────────────────────────────────

def load_economic_data(currency: str, fred_api_key=None) -> dict:
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

    rates_raw = fetch_json(f'{GITHUB_BASE}/rates/{currency}.json', timeout=6)
    if rates_raw and rates_raw.get('observations'):
        observations = rates_raw['observations']
        for obs in observations:
            val = obs.get('value')
            if val and val != '.':
                try:
                    data['interestRate'] = float(val)
                    break
                except ValueError:
                    pass

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


def compute_global_context(all_data: dict) -> dict:
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


# ─── SEÑALES ESTRUCTURALES ───────────────────────────────────────────────────

def infer_structural_signals(currency: str, data: dict, global_context: dict) -> list:
    signals  = []
    avg_rate = global_context.get('avg_interest_rate') or 3.0
    avg_cot  = global_context.get('avg_cot') or 0

    rate           = data.get('interestRate')
    rate_momentum  = data.get('rateMomentum')
    current_account= data.get('currentAccount')
    trade_balance  = data.get('tradeBalance')
    debt           = data.get('debt')
    cot            = data.get('cotPositioning')
    fx_1m          = data.get('fxPerformance1M')
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

    last_decision = data.get('lastRateDecision')
    if last_decision and last_decision.get('direction'):
        rd    = last_decision['direction']
        delta = last_decision.get('delta') or 0
        if rd == 'SUBIÓ':
            signals.append(
                f"El banco central acaba de SUBIR tasas ({delta:+.2f}pp): ciclo restrictivo activo, "
                f"estructuralmente alcista para la divisa por mayor carry diferencial."
            )
        elif rd == 'BAJÓ':
            signals.append(
                f"El banco central acaba de BAJAR tasas ({abs(delta):.2f}pp): ciclo expansivo, "
                f"el menor carry presiona a la baja la divisa."
            )
        elif rd == 'MANTUVO' and rate_momentum is not None:
            if rate_momentum < -0.5:
                signals.append(
                    f"Pausa en el ciclo: el banco central mantuvo tasas pero lleva {rate_momentum:+.2f}pp acumulados en 12M — "
                    f"política aún laxa respecto al pico, carry reducido."
                )
            elif rate_momentum > 0.5:
                signals.append(
                    f"Pausa en el ciclo: el banco central mantuvo tasas pero lleva {rate_momentum:+.2f}pp de subidas en 12M — "
                    f"política restrictiva sostenida, carry por encima del ciclo anterior."
                )
    elif rate_momentum is not None:
        if rate_momentum > 0.5:
            signals.append(
                f"El banco central lleva subiendo tasas {rate_momentum:+.2f}pp en 12 meses: "
                f"ciclo restrictivo activo, estructuralmente alcista para la divisa."
            )
        elif rate_momentum < -0.5:
            signals.append(
                f"El banco central lleva recortando {rate_momentum:+.2f}pp en 12 meses: "
                f"ciclo expansivo, el menor carry presiona a la baja la divisa."
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
            f"Balance exterior sólido: superávit de cuenta corriente ({current_account:.1f}% del PIB) "
            f"y deuda controlada ({debt:.0f}% del PIB)."
        )

    if trade_balance is not None and terms_of_trade is not None:
        if trade_balance > 2000 and terms_of_trade > 100:
            signals.append(
                f"Superávit comercial de {trade_balance/1000:.1f}B USD/mes con términos de intercambio "
                f"favorables (índice {terms_of_trade:.1f}): los precios de exportación superan a los de "
                f"importación, lo que amplía el excedente."
            )
        elif trade_balance < -15000:
            signals.append(
                f"Déficit comercial pronunciado ({trade_balance/1000:.1f}B USD/mes): "
                f"presión vendedora estructural sobre la divisa."
            )

    if cot is not None:
        if cot > 30000:
            signals.append(
                f"COT muy estirado al alza ({cot/1000:.0f}K contratos): el mercado está saturado de largos; "
                f"cualquier decepción en los datos puede provocar una corrección rápida."
            )
        elif cot < -30000:
            signals.append(
                f"COT muy estirado a la baja ({cot/1000:.0f}K contratos): mercado con posición corta extrema; "
                f"potencial rebote contrario si los datos mejoran."
            )

    if cot is not None and fx_1m is not None:
        if cot < -20000 and fx_1m > 2.5:
            signals.append(
                f"Divergencia relevante: posicionamiento bajista ({cot/1000:.0f}K) pero la divisa "
                f"se apreció {fx_1m:.1f}% el último mes — posible squeeze de cortos."
            )
        elif cot > 20000 and fx_1m < -2.0:
            signals.append(
                f"Divergencia relevante: posicionamiento alcista ({cot/1000:.0f}K) pero la divisa "
                f"cayó {abs(fx_1m):.1f}% el último mes — posible inicio de cierre de largos."
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


# ─── CONSTRUCCIÓN DEL PROMPT ─────────────────────────────────────────────────

def build_data_summary(currency: str, data: dict,
                       global_context: dict = None,
                       export_composition: str = None) -> str:
    meta     = COUNTRY_META[currency]
    avg_rate = (global_context or {}).get('avg_interest_rate') or 3.0
    avg_fx   = (global_context or {}).get('avg_fx_perf_1m') or 0.0

    lines = [
        f"DIVISA: {currency} — {meta['name']}",
        f"BANCO CENTRAL: {meta['bank']}",
        "",
        "INDICADORES ECONÓMICOS:",
    ]

    rate          = data.get('interestRate')
    rate_mom      = data.get('rateMomentum')
    inflation     = data.get('inflation')
    gdp_growth    = data.get('gdpGrowth')
    last_decision = data.get('lastRateDecision')

    if rate is not None:
        rate_vs_avg = rate - avg_rate
        direction   = "por encima" if rate_vs_avg > 0 else "por debajo"
        lines.append(
            f"  • Tasa de interés: {rate:.2f}% "
            f"({abs(rate_vs_avg):.2f}pp {direction} del promedio global de {avg_rate:.2f}%)"
        )

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
        if rate_mom > 0:
            lines.append(f"  • Ciclo monetario (12M acumulado): SUBIENDO tasas ({rate_mom:+.2f}pp) → hawkish")
        elif rate_mom < 0:
            lines.append(f"  • Ciclo monetario (12M acumulado): RECORTANDO tasas ({rate_mom:+.2f}pp) → dovish")
        else:
            lines.append(f"  • Ciclo monetario (12M acumulado): en pausa (0.00pp)")

    if inflation is not None:
        target_diff = inflation - 2.0
        status = (
            "por encima del objetivo" if target_diff > 0.3
            else "por debajo del objetivo" if target_diff < -0.3
            else "cerca del objetivo del 2%"
        )
        lines.append(f"  • Inflación: {inflation:.1f}% anual ({status})")

    if gdp_growth is not None:
        avg_gdp   = (global_context or {}).get('avg_gdp_growth') or 0.5
        vs_global = "superior" if gdp_growth > avg_gdp else "inferior"
        lines.append(
            f"  • Crecimiento PIB: {gdp_growth:.1f}% anual "
            f"({vs_global} al promedio global de {avg_gdp:.1f}%)"
        )

    unemployment = data.get('unemployment')
    wage_growth  = data.get('wageGrowth')
    retail_sales = data.get('retailSales')
    if unemployment is not None:
        avg_unemp = (global_context or {}).get('avg_unemployment') or 4.5
        label     = "bajo" if unemployment < avg_unemp else "elevado"
        lines.append(f"  • Desempleo: {unemployment:.1f}% ({label} respecto al promedio de {avg_unemp:.1f}%)")
    if wage_growth is not None:
        lines.append(f"  • Crecimiento salarial: {wage_growth:.1f}% anual")
    if retail_sales is not None:
        lines.append(f"  • Ventas minoristas: {retail_sales:+.1f}% mensual")

    ca  = data.get('currentAccount')
    tb  = data.get('tradeBalance')
    tot = data.get('termsOfTrade')
    if ca is not None:
        lines.append(
            f"  • Cuenta corriente: {ca:+.1f}% del PIB "
            f"({'superávit — demanda estructural de la divisa' if ca > 0 else 'déficit — presión vendedora estructural'})"
        )
    if tb is not None:
        lines.append(
            f"  • Balanza comercial: {tb/1000:+.1f}B USD/mes "
            f"({'superávit' if tb > 0 else 'déficit'} en bienes)"
        )
    if tot is not None:
        label = "favorable (exportaciones ganan valor relativo)" if tot > 100 else "desfavorable (importaciones encarecidas)"
        lines.append(f"  • Términos de intercambio: {tot:.1f} (base 100 — {label})")

    pmi  = data.get('manufacturingPMI')
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

    cc = data.get('consumerConfidence')
    bc = data.get('businessConfidence')
    ie = data.get('inflationExpectations')
    cf = data.get('capitalFlows')
    if cc is not None:
        lines.append(f"  • Confianza del consumidor: {cc:.1f}")
    if bc is not None:
        lines.append(f"  • Confianza empresarial: {bc:.1f}")
    if ie is not None:
        lines.append(f"  • Expectativas de inflación: {ie:.1f}%")
    if cf is not None:
        lines.append(
            f"  • Flujos de capital: {cf/1000:+.1f}B USD "
            f"({'entrada neta' if cf > 0 else 'salida neta'})"
        )

    cot  = data.get('cotPositioning')
    fx1m = data.get('fxPerformance1M')
    if cot is not None:
        if cot > 30000:
            cot_interp = "POSICIÓN ALCISTA EXTREMA — mercado muy cargado al alza, riesgo de corrección ante cualquier decepción"
        elif cot < -30000:
            cot_interp = "POSICIÓN BAJISTA EXTREMA — mercado muy cargado a la baja, potencial rebote contrario si mejoran los datos"
        elif cot > 0:
            cot_interp = "sesgo especulativo neto alcista"
        else:
            cot_interp = "sesgo especulativo neto bajista"
        lines.append(f"  • COT (posicionamiento especulativo): {cot/1000:+.1f}K contratos netos — {cot_interp}")
    if fx1m is not None:
        fx_vs_avg = fx1m - avg_fx
        move = "apreciación" if fx1m > 0 else "depreciación"
        rel  = "por encima" if fx_vs_avg > 0 else "por debajo"
        lines.append(
            f"  • Rendimiento FX último mes: {fx1m:+.2f}% vs USD ({move}) — "
            f"{abs(fx_vs_avg):.2f}pp {rel} del promedio de las 8 divisas principales ({avg_fx:+.2f}%)"
        )

    available = sum(
        1 for v in [rate, rate_mom, inflation, gdp_growth, unemployment, wage_growth,
                    retail_sales, ca, tb, tot, pmi, prod, debt, bond, cc, bc, ie, cf, cot, fx1m]
        if v is not None
    )
    lines.append(f"\n[{available} indicadores disponibles | Datos a: {str(data.get('lastUpdate', 'N/D'))[:10]}]")

    if export_composition:
        lines.append("")
        lines.append("COMPOSICIÓN EXPORTADORA VERIFICADA:")
        lines.append(f"  {export_composition}")
        lines.append("  (Dato objetivo: úsalo para explicar el origen del superávit/déficit comercial sin especular)")

    if global_context:
        lines.append("")
        lines.append("CONTEXTO GLOBAL — PROMEDIOS DE LAS 8 DIVISAS PRINCIPALES:")
        mappings = [
            ('avg_interest_rate', 'Tasa de interés promedio',    2, '%'),
            ('avg_gdp_growth',    'Crecimiento PIB promedio',     1, '% anual'),
            ('avg_inflation',     'Inflación promedio',           1, '%'),
            ('avg_unemployment',  'Desempleo promedio',           1, '%'),
            ('avg_bond10y',       'Yield bono 10Y promedio',      2, '%'),
            ('avg_fx_perf_1m',    'Rendimiento FX 1M promedio',   2, '%'),
            ('avg_cot',           'COT promedio',                  0, ' contratos netos'),
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


def build_user_prompt(currency: str, data_summary: str, news_items: list) -> str:
    """
    Construye el user prompt incluyendo el bloque noticioso.
    Las noticias van en CONTEXTO_NOTICIOSO_DIVISA para que Groq las
    teja dentro de los párrafos pertinentes.
    """
    news_block = format_news_for_prompt(news_items, currency)

    return (
        f"{data_summary}\n\n"
        f"CONTEXTO_NOTICIOSO_DIVISA:\n{news_block}\n\n"
        f"---\n\n"
        f"Redacta el análisis para {currency}. "
        f"Recuerda: exactamente 3 párrafos separados por línea en blanco, 150-200 palabras en total. "
        f"Las noticias del CONTEXTO_NOTICIOSO_DIVISA deben quedar integradas "
        f"dentro de los párrafos pertinentes, no como sección separada. "
        f"El objetivo es interpretar causas y consecuencias, no listar datos. "
        f"Redacción en español natural y fluido."
    )


# ─── LLAMADA A GROQ ──────────────────────────────────────────────────────────

def call_groq_api(api_key: str, user_prompt: str) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":  700,
        "temperature": 0.5,
        "top_p":       0.9,
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
        body = response.text.lower()
        if 'daily' in body or 'quota' in body:
            raise RuntimeError("DAILY_LIMIT")
        raise RuntimeError("RATE_LIMIT")
    if response.status_code == 401:
        raise RuntimeError("INVALID_KEY: verifica que GROQ_API_KEY esté correctamente configurada")
    response.raise_for_status()
    data = response.json()
    try:
        return data['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Respuesta inesperada: {data}") from e


def generate_analysis(api_key: str, currency: str, data: dict,
                      global_context: dict = None,
                      export_composition: str = None,
                      news_items: list = None) -> str:
    """
    Genera el análisis llamando a Groq.
    news_items: lista de artículos ya filtrados para esta divisa.
    """
    news_items    = news_items or []
    data_summary  = build_data_summary(currency, data, global_context, export_composition)
    user_prompt   = build_user_prompt(currency, data_summary, news_items)

    for attempt in range(3):
        try:
            text       = call_groq_api(api_key, user_prompt)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            text       = '\n\n'.join(paragraphs)
            word_count = len(text.split())
            if word_count < 80:
                raise ValueError(f"Respuesta demasiado corta: {word_count} palabras")
            print(f"  ✅ {word_count} palabras generadas")
            return text

        except RuntimeError as e:
            err_str = str(e)
            if "DAILY_LIMIT" in err_str or "RATE_LIMIT" in err_str:
                raise
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


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"🤖 Generador AI v5.0 — {GROQ_MODEL} via Groq API")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   TTL análisis: {MAX_ANALYSIS_AGE_HOURS}h | Noticias integradas en narrativa")
    print("=" * 60)

    # ── Keys ─────────────────────────────────────────────────────────────────
    all_keys = load_groq_keys()
    if not all_keys:
        raise EnvironmentError("❌ No se encontró ninguna GROQ_API_KEY.")

    print(f"\n🔑 Keys configuradas: {len(all_keys)}")
    available_keys = []
    for i, key in enumerate(all_keys, 1):
        status = check_groq_key(key)
        label  = f"Key {i} ({mask_key(key)})"
        if status == "daily_limit":
            print(f"  ⛔ {label} — límite diario confirmado")
        elif status == "invalid":
            print(f"  ❌ {label} — inválida (401)")
        else:
            print(f"  ✅ {label} — disponible")
            available_keys.append(key)

    if not available_keys:
        print("\n⛔ Todas las keys confirmadas agotadas. Saliendo sin error.")
        import sys
        sys.exit(0)

    current_key_idx = 0
    print(f"\n✅ {len(available_keys)} key(s) disponible(s) | Usando Key 1")
    print(f"🔧 Modelo: {GROQ_MODEL}")
    print(f"📊 Decisiones monetarias: historial propio rates/{{currency}}.json")
    print(f"💱 FX Performance: Frankfurter API (ECB, sin key) con fallback estático")
    print(f"📰 Noticias: {NEWS_PATH}\n")

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

    # ── Cargar noticias una sola vez para todo el run ─────────────────────────
    print("📰 Cargando noticias recientes...")
    news_data = load_news_file()
    # Pre-calcular noticias por divisa para el pre-check de caché
    all_news: dict[str, list] = {
        c: get_relevant_news(c, news_data) for c in CURRENCIES
    }
    for c, items in all_news.items():
        if items:
            print(f"  • {c}: {len(items)} noticia(s) relevante(s) "
                  f"[{', '.join(a.get('impact','?') for a in items)}]")
    print()

    print("📥 Cargando datos económicos + decisiones monetarias del historial...")
    all_data = {}
    for currency in CURRENCIES:
        print(f"  • {currency}...", end=' ', flush=True)
        all_data[currency] = load_economic_data(currency)
        available = sum(
            1 for k, v in all_data[currency].items()
            if v is not None and k not in ('lastRateDecision', 'fxSource')
        )
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

    # ── Pre-check: TTL + snapshot de datos + noticias nuevas ─────────────────
    print("🔎 Verificando TTL, cambios en datos y noticias recientes...")
    needs_regen   = []
    can_reuse     = []
    prev_analyses = {}

    for currency in CURRENCIES:
        prev    = load_previous_analysis(currency)
        prev_analyses[currency] = prev

        macro_changed = data_has_changed(currency, all_data[currency], prev)
        news_changed  = news_has_changed(currency, prev or {}, all_news[currency])
        changed       = macro_changed or news_changed

        if changed:
            needs_regen.append(currency)
            if prev is None:
                reason = "sin análisis previo"
            else:
                generated_at = prev.get("generatedAt", "")
                try:
                    gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                    age_h  = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
                    if age_h > MAX_ANALYSIS_AGE_HOURS:
                        reason = f"TTL expirado ({age_h:.0f}h)"
                    elif news_changed:
                        n_new = len([
                            a for a in all_news[currency]
                            if a.get("impact") == "high"
                            and (a.get("ts") or 0) / 1000
                               > (prev.get("dataSnapshot", {}).get("latestNewsTs") or 0)
                        ])
                        reason = f"noticias nuevas de alto impacto ({n_new})"
                    else:
                        reason = "datos macroeconómicos actualizados"
                except Exception:
                    reason = "datos actualizados"
            print(f"  🆕 {currency} — {reason}")
        else:
            can_reuse.append(currency)
            prev_age = ""
            if prev and prev.get("generatedAt"):
                try:
                    gen_at = datetime.fromisoformat(prev["generatedAt"].replace("Z", "+00:00"))
                    hours  = (datetime.now(timezone.utc) - gen_at).total_seconds() / 3600
                    prev_age = f" (generado hace {hours:.0f}h)"
                except Exception:
                    pass
            print(f"  ♻️  {currency} — sin cambios{prev_age}, reutilizando")

    print(f"\n   🆕 A regenerar:  {len(needs_regen)} — {', '.join(needs_regen) or 'ninguna'}")
    print(f"   ♻️  A reutilizar: {len(can_reuse)} — {', '.join(can_reuse) or 'ninguna'}")

    if not needs_regen:
        print("\n✅ Todos los análisis están actualizados. Sin llamadas a Groq necesarias.")
        successful = [c for c in CURRENCIES if prev_analyses.get(c) is not None]
        index = {
            "generatedAt":    datetime.now(timezone.utc).isoformat(),
            "model":          GROQ_MODEL,
            "version":        "5.0",
            "currencies":     successful,
            "totalGenerated": 0,
            "totalReused":    len(can_reuse),
            "keysUsed":       0,
            "errors":         [],
            "results":        {c: {"success": True, "reused": True} for c in can_reuse},
            "globalContext":  global_context,
        }
        with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        return
    print()

    # ── Generar divisas que necesitan actualización ───────────────────────────
    results = {c: {"success": True, "reused": True} for c in can_reuse}
    errors  = []

    for i, currency in enumerate(needs_regen):
        print(f"[{i+1}/{len(needs_regen)}] {currency} — generando análisis...")
        data        = all_data[currency]
        export_comp = export_compositions.get(currency)
        news_items  = all_news[currency]

        available = sum(1 for v in data.values() if v is not None)
        print(f"  📊 {available} indicadores económicos")
        if export_comp:
            print(f"  🏭 Exportaciones: {export_comp[:80]}...")
        if news_items:
            print(f"  📰 Noticias: {len(news_items)} artículo(s) integrado(s) en el prompt")
        else:
            print(f"  📰 Noticias: sin contexto noticioso disponible")

        if available < 4:
            msg = f"Datos insuficientes ({available})"
            print(f"  ⚠️  {msg}, saltando...")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            continue

        # Intentar con key actual, rotar si DAILY_LIMIT
        analysis_text = None
        while current_key_idx < len(available_keys):
            try:
                print(f"  🧠 Groq API (Key {current_key_idx + 1})...")
                analysis_text = generate_analysis(
                    available_keys[current_key_idx],
                    currency,
                    data,
                    global_context,
                    export_comp,
                    news_items,
                )
                break

            except RuntimeError as e:
                err_str = str(e)
                if "DAILY_LIMIT" in err_str or "RATE_LIMIT" in err_str:
                    label = (
                        "agotada (límite diario)" if "DAILY_LIMIT" in err_str
                        else "con rate limit persistente — rotando"
                    )
                    print(f"  ⛔ Key {current_key_idx + 1} {label} — buscando siguiente...")
                    current_key_idx += 1
                    if current_key_idx >= len(available_keys):
                        print("  ⛔ Todas las keys agotadas o con rate limit — deteniendo")
                        break
                    print(
                        f"  🔄 Cambiando a Key {current_key_idx + 1} "
                        f"({mask_key(available_keys[current_key_idx])}) — pausa {KEY_SWITCH_PAUSE}s..."
                    )
                    time.sleep(KEY_SWITCH_PAUSE)
                else:
                    raise

        if analysis_text is None:
            prev = prev_analyses.get(currency)
            if prev and prev.get("analysis"):
                print(f"  ⚠️  Groq falló — conservando análisis previo")
                results[currency] = {"success": True, "reused": True, "fallback": True}
                continue
            msg = (
                "Todas las keys agotadas"
                if current_key_idx >= len(available_keys)
                else "Error en generación"
            )
            print(f"  ❌ {msg}")
            errors.append(f"{currency}: {msg}")
            results[currency] = {"success": False, "error": msg}
            if current_key_idx >= len(available_keys):
                print("\n⛔ Sin keys disponibles — deteniendo generación")
                break
            continue

 # Calcular latestNewsTs para el snapshot — permite detectar noticias nuevas
        latest_news_ts = 0
        news_headlines = []
        if news_items:
            latest_news_ts = max(
                (a.get("ts", 0) or 0) for a in news_items
            ) / 1000  # ms → s
            news_headlines = [
                a.get("title", "")[:80]
                for a in news_items[:3]
            ]

        output = {
            "currency":    currency,
            "country":     COUNTRY_META[currency]['name'],
            "bank":        COUNTRY_META[currency]['bank'],
            "analysis":    analysis_text,
            "model":       GROQ_MODEL,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "exportComposition": export_comp,
            "newsCount":   len(news_items),
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
                # ── v5.0: campos para detectar cambios noticiosos ──
                "latestNewsTs":      latest_news_ts,
                "newsHeadlines":     news_headlines,
            },
            "globalContext": global_context,
        }

        output_path = OUTPUT_DIR / f"{currency}.json"
        # FIX R-02: Serializar primero en memoria y validar antes de escribir a disco.
        # Esto previene archivos corruptos en el repo si hay un error de encoding
        # o una interrupción durante la escritura.
        try:
            output_json = json.dumps(output, ensure_ascii=False, indent=2)
            json.loads(output_json)  # Validar que el JSON es parseable
        except (TypeError, ValueError) as e:
            print(f"  ❌ Error de serialización JSON para {currency}: {e} — saltando escritura")
            errors.append(f"{currency}: JSON inválido ({e})")
            results[currency] = {"success": False, "error": f"JSON inválido: {e}"}
            continue
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json)

        results[currency] = {
            "success":          True,
            "reused":           False,
            "wordCount":        len(analysis_text.split()),
            "generatedAt":      output["generatedAt"],
            "newsIntegrated":   len(news_items),
            "exportDataSource": "comtrade" if export_comp and "Comtrade" in export_comp else "fallback",
            "keyUsed":          current_key_idx + 1,
        }
        print(f"  💾 Guardado → {output_path}")

        if i < len(needs_regen) - 1 and current_key_idx < len(available_keys):
            print(f"  ⏸  Pausa 3s...")
            time.sleep(3)

    # ── Índice final ──────────────────────────────────────────────────────────
    successful     = [c for c, r in results.items() if r.get('success')]
    regenerated    = [c for c, r in results.items() if r.get('success') and not r.get('reused')]
    reused_final   = [c for c, r in results.items() if r.get('reused')]
    comtrade_count = sum(1 for r in results.values() if r.get('exportDataSource') == 'comtrade')
    news_total     = sum(r.get('newsIntegrated', 0) for r in results.values() if not r.get('reused'))

    index = {
        "generatedAt":    datetime.now(timezone.utc).isoformat(),
        "model":          GROQ_MODEL,
        "version":        "5.0",
        "currencies":     successful,
        "totalGenerated": len(regenerated),
        "totalReused":    len(reused_final),
        "comtradeHits":   comtrade_count,
        "newsIntegrated": news_total,
        "keysUsed":       current_key_idx + 1 if needs_regen else 0,
        "errors":         errors,
        "results":        results,
        "globalContext":  global_context,
    }
    with open(OUTPUT_DIR / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print(f"   ✅ Exitosos:          {len(successful)}/{len(CURRENCIES)}")
    print(f"   🆕 Regenerados:      {len(regenerated)} — {', '.join(regenerated) or 'ninguno'}")
    print(f"   ♻️  Reutilizados:     {len(reused_final)} — {', '.join(reused_final) or 'ninguno'}")
    print(f"   🌐 Comtrade:         {comtrade_count}/{len(CURRENCIES)} divisas")
    print(f"   📰 Noticias en prompt: {news_total} artículos totales")
    if needs_regen:
        print(f"   🔑 Keys usadas:      hasta Key {current_key_idx + 1} de {len(available_keys)}")
    else:
        print(f"   🔑 Groq:             0 llamadas (todos reutilizados)")
    if errors:
        print(f"   ❌ Errores:")
        for err in errors:
            print(f"      • {err}")
    print("=" * 60)

    if len(errors) > len(successful):
        raise RuntimeError(f"Demasiados errores: {len(errors)} fallos")


if __name__ == '__main__':
    main()
