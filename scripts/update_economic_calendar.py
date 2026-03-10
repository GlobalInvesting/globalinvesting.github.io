#!/usr/bin/env python3
"""
update_economic_calendar.py  v9.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Arquitectura de fiabilidad:

  Fuente 1 — Investing.com __NEXT_DATA__ (estrategia A: params URL)
  Fuente 2 — Investing.com __NEXT_DATA__ (estrategia B: sin params, scrape directo)
  Fuente 3 — Investing.com JSON POST API (fallback HTML)
  Fuente 4 — Finnhub API (enriquece actuals/forecasts/impact, nunca reemplaza)

  Tabla de impactos hardcodeada (KNOWN_IMPACTS) → fuente de verdad absoluta
  para los ~80 eventos más relevantes. Nunca se calcula con heurísticas.

  Validación: si el resultado tiene < MIN_EVENTS_THRESHOLD el script
  mantiene el JSON anterior y sale con código 0 (no rompe el workflow).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import requests, json, re, os, sys, time
from datetime import date, datetime, timedelta
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────

TRACKED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}

CURRENCY_FLAGS = {
    'USD': '🇺🇸', 'EUR': '🇪🇺', 'GBP': '🇬🇧', 'JPY': '🇯🇵',
    'AUD': '🇦🇺', 'CAD': '🇨🇦', 'CHF': '🇨🇭', 'NZD': '🇳🇿',
}

COUNTRY_TO_CURRENCY = {
    'united states': 'USD', 'us': 'USD', 'usa': 'USD', 'u.s.': 'USD',
    'euro area': 'EUR', 'eurozone': 'EUR', 'european union': 'EUR',
    'germany': 'EUR', 'france': 'EUR', 'italy': 'EUR', 'spain': 'EUR',
    'netherlands': 'EUR', 'belgium': 'EUR', 'austria': 'EUR',
    'portugal': 'EUR', 'finland': 'EUR', 'ireland': 'EUR',
    'greece': 'EUR', 'luxembourg': 'EUR',
    'united kingdom': 'GBP', 'uk': 'GBP', 'great britain': 'GBP',
    'england': 'GBP', 'britain': 'GBP',
    'japan': 'JPY',
    'australia': 'AUD',
    'canada': 'CAD',
    'switzerland': 'CHF', 'swiss': 'CHF',
    'new zealand': 'NZD',
}

MONTH_ES = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic',
}

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
CALENDAR_PATH   = 'calendar-data/calendar.json'
MIN_EVENTS_THRESHOLD = 20   # mínimo para considerar el fetch exitoso

# ── TABLA DE IMPACTOS CONOCIDOS ───────────────────────────────────
# Fuente de verdad absoluta. Clave: fragmento lowercase del nombre del evento.
# El match es: cualquier evento cuyo nombre normalizado CONTIENE la clave.
# Orden importa: las claves más específicas primero.

KNOWN_IMPACTS = {
    # ── USD HIGH ──────────────────────────────────────────────────
    'nonfarm payroll':                      'high',
    'non-farm payroll':                     'high',
    'nfp':                                  'high',
    'consumer price index':                 'high',
    ' cpi ':                                'high',
    'cpi (mom)':                            'high',
    'cpi (yoy)':                            'high',
    'core cpi':                             'high',
    'federal funds rate':                   'high',
    'fed funds rate':                       'high',
    'fomc rate decision':                   'high',
    'fomc statement':                       'high',
    'fomc meeting minutes':                 'high',
    'fomc minutes':                         'high',
    'fomc press conference':                'high',
    'gdp (qoq)':                            'high',
    'gdp annualized':                       'high',
    'advance gdp':                          'high',
    'initial jobless claims':               'high',
    'unemployment rate':                    'high',
    'average hourly earnings':              'high',
    'retail sales (mom)':                   'high',
    'core retail sales':                    'high',
    'ism manufacturing pmi':                'high',
    'ism services pmi':                     'high',
    'ism non-manufacturing':                'high',
    'durable goods orders':                 'high',
    'core durable goods':                   'high',
    'personal consumption expenditures':    'high',
    'pce price index':                      'high',
    'core pce':                             'high',
    'pce (mom)':                            'high',
    'pce (yoy)':                            'high',
    'japan ppi':                            'medium',
    'japan producer price':                 'medium',
    'producer price index':                 'high',
    'ppi (mom)':                            'high',  # Feb sobreescrito más abajo si es low
    'trade balance':                        'medium',  # default medium, USD específico
    'existing home sales':                  'high',
    'new home sales':                       'medium',
    'housing starts':                       'medium',
    'building permits':                     'medium',
    'michigan consumer sentiment':          'high',
    'consumer confidence':                  'medium',
    'cb consumer confidence':               'medium',
    'jolts job openings':                   'high',
    'adp employment':                       'medium',
    'adp weekly':                           'medium',
    'adp nonfarm':                          'medium',
    'challenger job cuts':                  'medium',
    'philadelphia fed':                     'medium',
    'empire state':                         'medium',
    'factory orders':                       'medium',
    'industrial production':                'medium',
    'capacity utilization':                 'medium',
    'import price index':                   'low',
    'export price index':                   'low',
    'ppi (yoy)':                            'medium',

    # ── EUR HIGH ──────────────────────────────────────────────────
    'ecb rate decision':                    'high',
    'ecb interest rate':                    'high',
    'ecb monetary policy':                  'high',
    'ecb press conference':                 'high',
    'ecb president':                        'medium',
    'ecb chief economist':                  'medium',
    'eurozone cpi':                         'high',
    'eurozone gdp':                         'high',
    'eurozone unemployment':                'medium',
    'eurozone pmi':                         'medium',
    'markit pmi':                           'medium',
    'flash pmi':                            'medium',
    'germany gdp':                          'high',
    'germany cpi':                          'high',
    'germany ifo':                          'high',
    'germany zew':                          'medium',
    'germany industrial production':        'medium',
    'germany trade balance':                'medium',
    'germany retail sales':                 'medium',
    'germany factory orders':               'medium',
    'germany exports':                      'medium',
    'germany imports':                      'medium',
    'france gdp':                           'medium',
    'france cpi':                           'medium',
    'france industrial production':         'medium',
    'italy gdp':                            'medium',
    'spain gdp':                            'medium',
    'eurozone retail sales':                'medium',
    'eurozone industrial production':       'medium',
    'eurozone trade balance':               'medium',
    'eurozone sentix':                      'low',

    # ── GBP HIGH ──────────────────────────────────────────────────
    'boe rate decision':                    'high',
    'bank of england rate':                 'high',
    'boe interest rate':                    'high',
    'boe monetary policy':                  'high',
    'mpc vote':                             'high',
    'boe gov':                              'medium',
    'boe governor':                         'medium',
    'uk cpi':                               'high',
    'u.k. cpi':                             'high',
    'uk gdp':                               'high',
    'u.k. gdp':                             'high',
    'uk unemployment':                      'high',
    'uk claimant count':                    'high',
    'uk retail sales':                      'high',
    'u.k. retail sales':                    'high',
    'uk manufacturing pmi':                 'medium',
    'uk services pmi':                      'medium',
    'u.k. manufacturing pmi':              'medium',
    'u.k. services pmi':                    'medium',
    'brc retail sales':                     'medium',
    'u.k. brc retail sales':               'medium',
    'uk trade balance':                     'medium',
    'uk industrial production':             'medium',
    'uk housing':                           'medium',

    # ── JPY HIGH ──────────────────────────────────────────────────
    'boj rate decision':                    'high',
    'bank of japan rate':                   'high',
    'boj interest rate':                    'high',
    'boj monetary policy':                  'high',
    'boj press conference':                 'high',
    'boj governor':                         'medium',
    'japan cpi':                            'high',
    'tokyo cpi':                            'high',
    'japan gdp':                            'high',
    'gdp capital expenditure':              'medium',
    'gdp external demand':                  'medium',
    'gdp private consumption':              'medium',
    'japan unemployment':                   'medium',
    'japan retail sales':                   'medium',
    'japan industrial production':          'medium',
    'japan trade balance':                  'medium',
    'japan ppi':                            'medium',
    'japan average cash earnings':          'medium',
    'japan household spending':             'medium',
    'tankan':                               'high',
    'prelim machine tool':                  'low',
    'machine tool orders':                  'low',

    # ── AUD HIGH ──────────────────────────────────────────────────
    'rba rate decision':                    'high',
    'rba interest rate':                    'high',
    'rba cash rate':                        'high',
    'rba monetary policy':                  'high',
    'rba governor':                         'medium',
    'australia cpi':                        'high',
    'australia gdp':                        'high',
    'australia unemployment':               'high',
    'australia employment change':          'high',
    'australia trade balance':              'medium',
    'australia retail sales':               'medium',
    'australia building approvals':         'medium',
    'building approvals':                   'medium',
    'private house approvals':              'medium',
    'nab business survey':                  'medium',
    'nab business confidence':              'medium',
    'nab business conditions':              'medium',
    'westpac consumer sentiment':           'medium',
    'australia pmi':                        'medium',
    'rba meeting minutes':                  'medium',

    # ── CAD HIGH ──────────────────────────────────────────────────
    'boc rate decision':                    'high',
    'bank of canada rate':                  'high',
    'boc interest rate':                    'high',
    'boc monetary policy':                  'high',
    'boc press conference':                 'high',
    'boc governor':                         'medium',
    'canada cpi':                           'high',
    'canada gdp':                           'high',
    'canada employment':                    'high',
    'canada unemployment':                  'high',
    'canada trade balance':                 'medium',
    'canada retail sales':                  'medium',
    'canada manufacturing pmi':             'medium',
    'ivey pmi':                             'medium',

    # ── CHF HIGH ──────────────────────────────────────────────────
    'snb rate decision':                    'high',
    'swiss national bank rate':             'high',
    'snb interest rate':                    'high',
    'snb monetary policy':                  'high',
    'snb press conference':                 'high',
    'switzerland cpi':                      'high',
    'switzerland gdp':                      'high',
    'switzerland unemployment':             'medium',
    'switzerland retail sales':             'medium',
    'seco consumer climate':                'medium',
    'kof leading indicator':                'medium',

    # ── NZD HIGH ──────────────────────────────────────────────────
    'rbnz rate decision':                   'high',
    'reserve bank of new zealand rate':     'high',
    'rbnz interest rate':                   'high',
    'rbnz monetary policy':                 'high',
    'rbnz press conference':                'high',
    'rbnz governor':                        'medium',
    'new zealand cpi':                      'high',
    'new zealand gdp':                      'high',
    'new zealand unemployment':             'high',
    'new zealand employment change':        'high',
    'new zealand trade balance':            'medium',
    'new zealand retail sales':             'medium',
    'nz business confidence':               'medium',

    # ── LOW IMPACT (auctions, petróleo, etc.) ────────────────────
    'bill auction':                         'low',
    'bond auction':                         'low',
    'note auction':                         'low',
    'gilt auction':                         'low',
    'btp auction':                          'low',
    'oat auction':                          'low',
    'bobl auction':                         'low',
    'bund auction':                         'low',
    'api crude oil':                        'low',
    'api weekly crude':                     'low',
    'api oil stock':                        'low',
    'eia crude oil':                        'medium',
    'eia short-term energy':                'medium',
    'eia natural gas':                      'medium',
    'mba mortgage':                         'low',
    'redbook':                              'low',
    'nfib small business':                  'low',
    'ny fed consumer inflation':            'high',
    'consumer inflation expectation':       'high',
    '3-year note auction':                  'medium',
    '10-year note auction':                 'medium',
    '30-year bond auction':                 'medium',
}


# ── HELPERS ───────────────────────────────────────────────────────

def fmt_date(dt_obj):
    return f"{dt_obj.day} {MONTH_ES[dt_obj.month]}"

def clean_val(v):
    if v is None: return ''
    s = str(v).strip()
    return '' if s.lower() in ('none', 'nan', '-', 'n/a', '', 'null') else s

def resolve_currency(s):
    if not s: return None
    t = str(s).strip().upper()
    if t in TRACKED_CURRENCIES: return t
    for key, curr in COUNTRY_TO_CURRENCY.items():
        if key in t.lower(): return curr
    return None

def fmt_number(value, unit=''):
    if value is None: return ''
    try:
        v = float(value)
        s = f"{v:g}"
        u = (unit or '').strip()
        if u and u not in ('', 'None', 'none'):
            return s + u if u in ('%', 'B', 'M', 'K', 'T') else s + ' ' + u
        return s
    except (ValueError, TypeError):
        return clean_val(value)

def impact_rank(impact):
    return {'high': 3, 'medium': 2, 'low': 1}.get(impact, 0)

def score_ev(ev):
    s = 0
    if ev.get('actual'):   s += 1000
    if ev.get('forecast'): s += 100
    if ev.get('previous'): s += 10
    s += len(ev.get('event', ''))
    return s

def resolve_known_impact(event_name):
    """
    Busca en KNOWN_IMPACTS usando el nombre del evento.
    Retorna el impacto conocido o None si no hay match.
    Las claves más largas (más específicas) tienen prioridad.
    """
    name_lower = ' ' + event_name.lower() + ' '
    # Ordenar por longitud descendente para que matches específicos ganen
    for key in sorted(KNOWN_IMPACTS, key=len, reverse=True):
        if key in name_lower:
            return KNOWN_IMPACTS[key]
    return None

def classify_impact_fallback(event_name, currency):
    """
    Heurística de último recurso cuando no hay impact de ninguna fuente.
    Solo se usa si KNOWN_IMPACTS no matcheó.
    """
    name = event_name.lower()
    # Rate decisions siempre high
    if any(w in name for w in ['rate decision', 'interest rate', 'monetary policy statement']):
        return 'high'
    # GDP high para todas las monedas
    if 'gdp' in name and 'external' not in name and 'capital' not in name and 'private' not in name:
        return 'high'
    # CPI high
    if re.search(r'\bcpi\b', name) and 'core' not in name:
        return 'high'
    # Employment high para USD/AUD/CAD/NZD/GBP
    if currency in ('USD', 'AUD', 'CAD', 'NZD', 'GBP'):
        if any(w in name for w in ['employment change', 'nonfarm', 'unemployment rate']):
            return 'high'
    return 'low'


def parse_datetime_utc(time_str, fallback_date_str=''):
    """
    Parsea un string de tiempo UTC y retorna (date_obj, time_utc_str).
    Maneja: ISO8601, HH:MM, timestamps Unix.
    Para eventos 00:00-03:59 UTC, la fecha calendario es el día anterior
    (convención Investing.com: noche asiática aparece en el día previo).
    """
    if not time_str:
        if fallback_date_str:
            try:
                return date.fromisoformat(fallback_date_str), ''
            except Exception:
                pass
        return None, ''

    # ISO8601
    try:
        dt = datetime.fromisoformat(str(time_str).replace('Z', '+00:00'))
        t_utc = dt.strftime('%H:%M')
        cal_date = dt.date()
        if dt.hour < 4:
            cal_date = cal_date - timedelta(days=1)
        return cal_date, t_utc
    except Exception:
        pass

    # Unix timestamp
    try:
        ts = int(time_str)
        dt = datetime.utcfromtimestamp(ts)
        t_utc = dt.strftime('%H:%M')
        cal_date = dt.date()
        if dt.hour < 4:
            cal_date = cal_date - timedelta(days=1)
        return cal_date, t_utc
    except Exception:
        pass

    # HH:MM solo (sin fecha)
    m = re.match(r'^(\d{1,2}):(\d{2})$', str(time_str).strip())
    if m and fallback_date_str:
        try:
            h, mi = int(m.group(1)), int(m.group(2))
            d = date.fromisoformat(fallback_date_str)
            if h < 4:
                d = d - timedelta(days=1)
            return d, f"{h:02d}:{mi:02d}"
        except Exception:
            pass

    return None, ''


# ════════════════════════════════════════════════════════════════════
# SOURCE A: Investing.com — estrategia __NEXT_DATA__ (URL params)
# ════════════════════════════════════════════════════════════════════

def _parse_next_data_store(html_text, target_dates, source_label):
    """Extrae eventos del bloque __NEXT_DATA__ de Investing.com."""
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html_text, re.DOTALL
    )
    if not m:
        print(f"    [{source_label}] ❌ No __NEXT_DATA__ found")
        return []

    try:
        data = json.loads(m.group(1))
    except Exception as e:
        print(f"    [{source_label}] ❌ JSON parse error: {e}")
        return []

    # Navegar al store — distintas rutas posibles
    store = None
    paths = [
        ['props', 'pageProps', 'state', 'economicCalendarStore', 'calendarEventsByDate'],
        ['props', 'pageProps', 'economicCalendarData', 'calendarEventsByDate'],
        ['props', 'initialState', 'economicCalendarStore', 'calendarEventsByDate'],
    ]
    for path in paths:
        node = data
        try:
            for key in path:
                node = node[key]
            if isinstance(node, dict) and node:
                store = node
                break
        except (KeyError, TypeError):
            continue

    if not store:
        print(f"    [{source_label}] ❌ calendarEventsByDate not found in __NEXT_DATA__")
        return []

    IMPACT_MAP = {'1': 'low', '2': 'medium', '3': 'high', 1: 'low', 2: 'medium', 3: 'high'}
    events = []

    for date_key, day_events in store.items():
        if not isinstance(day_events, list):
            continue
        for ev in day_events:
            try:
                currency = str(ev.get('currency') or '').upper()
                if currency not in TRACKED_CURRENCIES:
                    continue

                event_name = (ev.get('eventLong') or ev.get('event') or '').strip()
                if not event_name:
                    continue

                time_raw = ev.get('time') or ev.get('actual_time') or ev.get('date') or ''
                cal_date, t_utc = parse_datetime_utc(time_raw, date_key)
                if not cal_date:
                    try:
                        cal_date = date.fromisoformat(date_key)
                    except Exception:
                        continue

                if cal_date not in target_dates:
                    continue

                imp_raw = ev.get('importance') or ev.get('impact') or '1'
                impact_from_source = IMPACT_MAP.get(str(imp_raw), 'low')

                # KNOWN_IMPACTS tiene prioridad absoluta
                known = resolve_known_impact(event_name)
                impact = known if known else impact_from_source

                actual   = clean_val(str(ev.get('actual',   '') or ''))
                forecast = clean_val(str(ev.get('forecast', '') or ''))
                previous = clean_val(str(ev.get('previous', '') or ''))

                events.append({
                    'date':     fmt_date(cal_date),
                    'dateISO':  cal_date.isoformat(),
                    'timeUTC':  t_utc,
                    'country':  currency,
                    'currency': currency,
                    'flag':     CURRENCY_FLAGS.get(currency, ''),
                    'event':    event_name,
                    'impact':   impact,
                    'actual':   actual,
                    'forecast': forecast,
                    'previous': previous,
                    '_source':  source_label,
                    '_impact_src': 'known' if known else 'investing',
                })
            except Exception:
                continue

    print(f"    [{source_label}] ✅ {len(events)} events parsed")
    return events


INVESTING_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
}


def fetch_investing_nextdata(from_str, to_str, target_dates):
    """Estrategia A: GET con dateFrom/dateTo en URL."""
    print(f"  [Investing-A] GET __NEXT_DATA__ {from_str}→{to_str}")
    url = 'https://www.investing.com/economic-calendar/'
    try:
        r = requests.get(
            url,
            params={'dateFrom': from_str, 'dateTo': to_str},
            headers=INVESTING_HEADERS,
            timeout=45
        )
        if not r.ok:
            print(f"  [Investing-A] ❌ HTTP {r.status_code}")
            return []
        return _parse_next_data_store(r.text, target_dates, 'Investing-A')
    except Exception as e:
        print(f"  [Investing-A] ❌ {e}")
        return []


def fetch_investing_nextdata_b(from_str, to_str, target_dates):
    """Estrategia B: GET sin params (página principal, trae ~7 días)."""
    print(f"  [Investing-B] GET __NEXT_DATA__ (no params)")
    url = 'https://www.investing.com/economic-calendar/'
    try:
        r = requests.get(url, headers=INVESTING_HEADERS, timeout=45)
        if not r.ok:
            print(f"  [Investing-B] ❌ HTTP {r.status_code}")
            return []
        return _parse_next_data_store(r.text, target_dates, 'Investing-B')
    except Exception as e:
        print(f"  [Investing-B] ❌ {e}")
        return []


# ════════════════════════════════════════════════════════════════════
# SOURCE B: Investing.com — JSON POST API (fallback robusto)
# ════════════════════════════════════════════════════════════════════

def fetch_investing_post(from_str, to_str, target_dates):
    """
    Estrategia C: POST al endpoint /economic-calendar/Service/getCalendarFilteredData
    Es el mismo request que hace el navegador cuando aplica filtros.
    """
    print(f"  [Investing-C] POST JSON API {from_str}→{to_str}")

    # Necesitamos un session token primero
    session = requests.Session()
    session.headers.update(INVESTING_HEADERS)

    try:
        # Primero obtener cookies
        session.get('https://www.investing.com/', timeout=20)
        time.sleep(1)

        url = 'https://www.investing.com/economic-calendar/Service/getCalendarFilteredData'
        headers_post = {
            **INVESTING_HEADERS,
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.investing.com/economic-calendar/',
        }

        # Currencies como IDs de Investing
        curr_ids = {
            'USD': '5', 'EUR': '17', 'GBP': '12', 'JPY': '35',
            'AUD': '11', 'CAD': '6', 'CHF': '19', 'NZD': '36',
        }
        currencies_param = '&'.join(f"country[]={v}" for v in curr_ids.values())

        payload = (
            f"dateFrom={from_str}&dateTo={to_str}"
            f"&timeZone=8"   # UTC
            f"&{currencies_param}"
            f"&importance[]=1&importance[]=2&importance[]=3"
            f"&action=updateEconomicCalendar"
        )

        r = session.post(url, data=payload, headers=headers_post, timeout=45)
        if not r.ok:
            print(f"  [Investing-C] ❌ HTTP {r.status_code}")
            return []

        data = r.json()
        html_content = data.get('data', '')
        if not html_content:
            print(f"  [Investing-C] ❌ Empty response data")
            return []

    except Exception as e:
        print(f"  [Investing-C] ❌ {e}")
        return []

    # Parsear el HTML devuelto por el POST
    IMPACT_MAP_POST = {'1': 'low', '2': 'medium', '3': 'high'}

    # Regex para extraer filas del calendario HTML
    events = []
    row_pattern = re.compile(
        r'<tr[^>]+id="eventRowId_(\d+)"[^>]*data-event-datetime="([^"]+)"'
        r'[^>]*>(.*?)</tr>',
        re.DOTALL
    )
    currency_pattern = re.compile(r'title="([A-Z]{3})"')
    event_pattern    = re.compile(r'<td[^>]+class="[^"]*event[^"]*"[^>]*>\s*<a[^>]*>([^<]+)</a>')
    impact_pattern   = re.compile(r'data-img_key="bull(\d)"')
    actual_pattern   = re.compile(r'id="actual_\d+"[^>]*>([^<]*)</td>')
    forecast_pattern = re.compile(r'id="forecast_\d+"[^>]*>([^<]*)</td>')
    previous_pattern = re.compile(r'id="previous_\d+"[^>]*>([^<]*)</td>')

    for row_m in row_pattern.finditer(html_content):
        try:
            row_html = row_m.group(3)
            datetime_str = row_m.group(2)  # "2026/03/10 14:00:00"

            # Parsear fecha/hora
            dt_clean = datetime_str.replace('/', '-')
            cal_date, t_utc = parse_datetime_utc(dt_clean + 'Z')
            if not cal_date or cal_date not in target_dates:
                continue

            # Currency
            curr_m = currency_pattern.search(row_html)
            if not curr_m: continue
            currency = curr_m.group(1)
            if currency not in TRACKED_CURRENCIES: continue

            # Event name
            ev_m = event_pattern.search(row_html)
            if not ev_m: continue
            event_name = ev_m.group(1).strip()

            # Impact
            imp_m = impact_pattern.search(row_html)
            imp_raw = imp_m.group(1) if imp_m else '1'
            impact_from_source = IMPACT_MAP_POST.get(imp_raw, 'low')

            known = resolve_known_impact(event_name)
            impact = known if known else impact_from_source

            actual   = clean_val((actual_pattern.search(row_html) or type('', (), {'group': lambda s, n: ''})()).group(1))
            forecast = clean_val((forecast_pattern.search(row_html) or type('', (), {'group': lambda s, n: ''})()).group(1))
            previous = clean_val((previous_pattern.search(row_html) or type('', (), {'group': lambda s, n: ''})()).group(1))

            events.append({
                'date':     fmt_date(cal_date),
                'dateISO':  cal_date.isoformat(),
                'timeUTC':  t_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency, ''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
                '_source':  'Investing-C',
                '_impact_src': 'known' if known else 'investing',
            })
        except Exception:
            continue

    print(f"  [Investing-C] ✅ {len(events)} events parsed")
    return events


# ════════════════════════════════════════════════════════════════════
# SOURCE C: Finnhub (enriquecedor — actuals/forecasts/impact)
# ════════════════════════════════════════════════════════════════════

def fetch_finnhub(from_str, to_str, target_dates):
    if not FINNHUB_API_KEY:
        print("  [Finnhub] ⚠️  No API key — skipping")
        return []

    print(f"  [Finnhub] GET {from_str}→{to_str}")
    url = 'https://finnhub.io/api/v1/calendar/economic'
    try:
        r = requests.get(
            url,
            params={'from': from_str, 'to': to_str, 'token': FINNHUB_API_KEY},
            timeout=30
        )
        if not r.ok:
            print(f"  [Finnhub] ❌ HTTP {r.status_code}")
            return []
        raw = r.json().get('economicCalendar', [])
    except Exception as e:
        print(f"  [Finnhub] ❌ {e}")
        return []

    events = []
    for ev in raw:
        try:
            currency = resolve_currency(ev.get('country', ''))
            if not currency: continue
            event_name = (ev.get('event') or '').strip()
            if not event_name: continue

            cal_date, t_utc = parse_datetime_utc(ev.get('time', ''))
            if not cal_date or cal_date not in target_dates: continue

            unit = ev.get('unit', '')
            actual   = fmt_number(ev.get('actual'),   unit)
            forecast = fmt_number(ev.get('estimate'), unit)
            previous = fmt_number(ev.get('prev'),     unit)

            # Impact de Finnhub — solo se usa si KNOWN_IMPACTS no matchea
            impact_finnhub = (ev.get('impact') or 'low').lower()
            if impact_finnhub not in ('high', 'medium', 'low'):
                impact_finnhub = 'low'

            known = resolve_known_impact(event_name)
            impact = known if known else impact_finnhub

            events.append({
                'date':     fmt_date(cal_date),
                'dateISO':  cal_date.isoformat(),
                'timeUTC':  t_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency, ''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
                '_source':  'finnhub',
                '_impact_src': 'known' if known else 'finnhub',
            })
        except Exception:
            continue

    print(f"  [Finnhub] ✅ {len(events)} events")
    return events


# ════════════════════════════════════════════════════════════════════
# DEDUP
# ════════════════════════════════════════════════════════════════════

def normalize_for_dedup(name):
    """Normalización para detectar duplicados semánticos entre fuentes."""
    n = name.lower().strip()
    # Normalizar variantes de periodicidad
    for pat, rep in [
        (r'\(yoy\)', 'yoy'), (r'\by/y\b', 'yoy'), (r'\(yoy\b', 'yoy'),
        (r'\(mom\)', 'mom'), (r'\bm/m\b', 'mom'), (r'\(mom\b', 'mom'),
        (r'\(qoq\)', 'qoq'), (r'\bq/q\b', 'qoq'), (r'\(qoq\b', 'qoq'),
        (r'\bgross domestic product\b', 'gdp'),
        (r'\bconsumer price index\b', 'cpi'),
        (r'\bpurchasing managers\b', 'pmi'),
    ]:
        n = re.sub(pat, rep, n)
    # Eliminar meses y trimestres
    n = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|\d{4})\b', '', n)
    # Eliminar paréntesis y su contenido (solo si NO contienen yoy/mom/qoq)
    n = re.sub(r'\([^)]*\)', '', n)
    # Eliminar nombres de países/gentilicios
    n = re.sub(
        r'\b(japan|japanese|us|uk|germany|german|france|french|eurozone|euro area|'
        r'australia|australian|canada|canadian|swiss|switzerland|new zealand|u\.s\.|'
        r'united states|united kingdom|eurogroup)\b', '', n
    )
    # Stopwords
    sw = {
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'by', 'or',
        'rate', 'change', 'data', 'flash', 'final', 'prelim', 'preliminary',
        'revised', 'report', 'survey', 'release', 'monthly', 'quarterly',
        'annual', 'weekly', 'current', 'total', 'net', 'new', 'all', 'index',
        'q', 'y', 'm', 'sa', 'nsa',
    }
    words = [w for w in re.sub(r'[^a-z0-9 ]', ' ', n).split()
             if w not in sw and len(w) > 1]
    return ' '.join(sorted(words))


def dedup_events(events):
    """Elimina duplicados: exacto (slot) y semántico (nombre normalizado)."""
    # Paso A: mismo slot exacto
    slot_groups = defaultdict(list)
    for ev in events:
        key = (ev['dateISO'], ev.get('timeUTC', ''), ev['currency'])
        slot_groups[key].append(ev)

    after_slot = []
    for group in slot_groups.values():
        if len(group) == 1:
            after_slot.append(group[0])
            continue
        best = max(group, key=score_ev)
        merged = dict(best)
        # Tomar el mayor impacto del grupo
        best_impact = max(group, key=lambda e: impact_rank(e.get('impact', 'low')))
        merged['impact'] = best_impact['impact']
        # Propagar actual/forecast/previous si faltan
        for ev in group:
            for f in ('actual', 'forecast', 'previous'):
                if not merged.get(f) and ev.get(f):
                    merged[f] = ev[f]
        after_slot.append(merged)

    # Paso B: mismo día + currency + nombre normalizado
    sem_groups = defaultdict(list)
    for ev in after_slot:
        norm = normalize_for_dedup(ev['event'])
        sem_groups[(ev['dateISO'], ev['currency'], norm)].append(ev)

    final = []
    merged_count = 0
    for group in sem_groups.values():
        if len(group) == 1:
            final.append(group[0])
            continue
        merged_count += len(group) - 1
        best = max(group, key=score_ev)
        merged = dict(best)
        best_impact = max(group, key=lambda e: impact_rank(e.get('impact', 'low')))
        merged['impact'] = best_impact['impact']
        for ev in group:
            for f in ('actual', 'forecast', 'previous'):
                if not merged.get(f) and ev.get(f):
                    merged[f] = ev[f]
        final.append(merged)

    if merged_count:
        print(f"  [Dedup] Merged {merged_count} duplicates: {len(events)} → {len(final)}")
    else:
        print(f"  [Dedup] {len(events)} events, no duplicates found")
    return final


# ════════════════════════════════════════════════════════════════════
# MERGE — Investing primario, Finnhub enriquece
# ════════════════════════════════════════════════════════════════════

def make_match_key(ev):
    """Key corta para matching entre fuentes (primeros 30 chars normalizados)."""
    name = re.sub(r'\s*\([^)]*\)', '', ev['event'].lower()).strip()
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()[:30]
    return (ev['dateISO'], ev['currency'], name)


def merge_investing_finnhub(investing_evs, finnhub_evs):
    """
    Investing es la base (más eventos, nombres más completos).
    Finnhub enriquece: actual/forecast/previous y puede corregir impact
    (pero KNOWN_IMPACTS ya tiene la última palabra en ambas fuentes).
    """
    # Indexar Investing
    inv_index = {}
    for ev in investing_evs:
        k = make_match_key(ev)
        if k not in inv_index or score_ev(ev) > score_ev(inv_index[k]):
            inv_index[k] = ev

    merged = dict(inv_index)
    added_from_finnhub = 0

    for ev in finnhub_evs:
        k = make_match_key(ev)
        if k not in merged:
            # Evento solo en Finnhub — agregarlo
            merged[k] = ev
            added_from_finnhub += 1
        else:
            # Enriquecer el evento de Investing con datos de Finnhub
            existing = dict(merged[k])
            for field in ('actual', 'forecast', 'previous'):
                if not existing.get(field) and ev.get(field):
                    existing[field] = ev[field]
            # Si Finnhub tiene impact de KNOWN_IMPACTS y Investing no, usar Finnhub
            if ev.get('_impact_src') == 'known' and existing.get('_impact_src') != 'known':
                existing['impact'] = ev['impact']
                existing['_impact_src'] = 'known'
            merged[k] = existing

    result = list(merged.values())
    print(f"  [Merge] Investing={len(investing_evs)}, Finnhub={len(finnhub_evs)}, "
          f"added_from_finnhub={added_from_finnhub}, total={len(result)}")
    return result


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

print("=" * 65)
print("ECONOMIC CALENDAR SCRAPER v9.0")
print("Primary:   Investing.com (3 strategies: A=NEXT_DATA+params,")
print("           B=NEXT_DATA no-params, C=JSON POST)")
print("Enricher:  Finnhub (actuals/forecasts, used when available)")
print("Authority: KNOWN_IMPACTS table (80+ events, absolute truth)")
print("=" * 65)

today     = date.today()
from_date = today - timedelta(days=60)
to_date   = today + timedelta(days=30)

target_dates = set()
d = from_date
while d <= to_date:
    target_dates.add(d)
    d += timedelta(days=1)

print(f"\nRange: {from_date} → {to_date} ({len(target_dates)} days)\n")

# ── STEP 1: Cargar base histórica ────────────────────────────────
print("=" * 50)
print("STEP 1 — Load base cache")
print("=" * 50)

base_events = {}
try:
    with open(CALENDAR_PATH, encoding='utf-8') as f:
        prev = json.load(f)
    for ev in prev.get('events', []):
        try:
            k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())
            base_events[k] = ev
        except Exception:
            pass
    all_base_dates = sorted(set(e['dateISO'] for e in base_events.values()))
    print(f"  Loaded {len(base_events)} events "
          f"({all_base_dates[0] if all_base_dates else '?'} → "
          f"{all_base_dates[-1] if all_base_dates else '?'})")
except FileNotFoundError:
    print("  No previous calendar.json — starting fresh")
except Exception as e:
    print(f"  Error: {e}")

# ── STEP 2: Fetch fuentes ────────────────────────────────────────
from_str = from_date.isoformat()
to_str   = to_date.isoformat()

print(f"\n{'=' * 50}")
print("STEP 2 — Fetch sources")
print("=" * 50)

# Estrategia A: __NEXT_DATA__ con parámetros
investing_evs = fetch_investing_nextdata(from_str, to_str, target_dates)

# Si A falló o trajo muy pocos eventos, intentar B
if len(investing_evs) < MIN_EVENTS_THRESHOLD:
    print(f"  ⚠️  Strategy A returned {len(investing_evs)} events — trying B...")
    evs_b = fetch_investing_nextdata_b(from_str, to_str, target_dates)
    if len(evs_b) > len(investing_evs):
        investing_evs = evs_b

# Si B también falló, intentar C (POST API)
if len(investing_evs) < MIN_EVENTS_THRESHOLD:
    print(f"  ⚠️  Strategy B returned {len(investing_evs)} events — trying C (POST)...")
    evs_c = fetch_investing_post(from_str, to_str, target_dates)
    if len(evs_c) > len(investing_evs):
        investing_evs = evs_c

print(f"\n  [Investing] Total after all strategies: {len(investing_evs)} events")

# Finnhub como enriquecedor
finnhub_evs = fetch_finnhub(from_str, to_str, target_dates)

# ── STEP 3: Merge fuentes frescas ───────────────────────────────
print(f"\n{'=' * 50}")
print("STEP 3 — Merge sources")
print("=" * 50)

if investing_evs or finnhub_evs:
    fresh_events = merge_investing_finnhub(investing_evs, finnhub_evs)
else:
    fresh_events = []
    print("  ⚠️  All sources failed — will use base cache only")

# ── STEP 4: Merge con base histórica ───────────────────────────
print(f"\n{'=' * 50}")
print("STEP 4 — Merge with base cache")
print("=" * 50)

if fresh_events:
    merged_dict = dict(base_events)
    added = updated = 0

    for ev in fresh_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())

        if k in merged_dict:
            existing = dict(merged_dict[k])
            # Fresh data siempre actualiza estos campos
            for field in ('actual', 'forecast', 'previous', 'timeUTC'):
                if ev.get(field):
                    existing[field] = ev[field]
            # Impact: solo actualizar si la fuente fresca tiene KNOWN_IMPACTS
            # o si la base no tenía dato confiable
            if ev.get('_impact_src') == 'known':
                existing['impact'] = ev['impact']
                existing['_impact_src'] = 'known'
            elif not existing.get('_impact_src') == 'known':
                existing['impact'] = ev['impact']
            merged_dict[k] = existing
            updated += 1
        else:
            # Buscar en fecha adyacente (eventos en boundary 00:00-03:59 UTC)
            matched = False
            try:
                ev_date = date.fromisoformat(ev['dateISO'])
                for delta in (-1, 1):
                    adj = (ev_date + timedelta(days=delta)).isoformat()
                    k_adj = (adj, ev['currency'], ev['event'][:25].lower().strip())
                    if k_adj in merged_dict:
                        existing = dict(merged_dict[k_adj])
                        for field in ('actual', 'forecast', 'previous'):
                            if ev.get(field):
                                existing[field] = ev[field]
                        merged_dict[k_adj] = existing
                        updated += 1
                        matched = True
                        break
            except Exception:
                pass
            if not matched:
                merged_dict[k] = ev
                added += 1

    all_events = list(merged_dict.values())
    print(f"  base={len(base_events)} + fresh={len(fresh_events)} → "
          f"{len(all_events)} total (added={added}, updated={updated})")
else:
    all_events = list(base_events.values())
    print(f"  Using base cache: {len(all_events)} events")

# ── STEP 5: Aplicar KNOWN_IMPACTS a toda la base ────────────────
print(f"\n{'=' * 50}")
print("STEP 5 — Apply KNOWN_IMPACTS to all events")
print("=" * 50)

corrected = 0
for ev in all_events:
    known = resolve_known_impact(ev['event'])
    if known and ev.get('impact') != known:
        ev['impact'] = known
        ev['_impact_src'] = 'known'
        corrected += 1

if corrected:
    print(f"  Corrected {corrected} impacts via KNOWN_IMPACTS table")
else:
    print(f"  All impacts already consistent with KNOWN_IMPACTS table")

# ── STEP 6: Dedup ───────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("STEP 6 — Dedup")
print("=" * 50)
final_events = dedup_events(all_events)

# Filtrar eventos sin tiempo ni datos (ruido puro sin tiempo y sin forecast)
before = len(final_events)
final_events = [
    ev for ev in final_events
    if (ev.get('timeUTC', '').strip() or
        ev.get('actual', '').strip() or
        ev.get('forecast', '').strip() or
        ev.get('previous', '').strip())
]
if len(final_events) < before:
    print(f"  Filtered {before - len(final_events)} no-data events")

# Limpiar campos internos
for ev in final_events:
    ev.pop('_source', None)
    ev.pop('_impact_src', None)

# Sort
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    if not re.match(r'\d{2}:\d{2}', t): t = '00:00'
    return ev['dateISO'] + 'T' + t

final_events.sort(key=sort_key)

# ── STEP 7: Validación y guardado ───────────────────────────────
print(f"\n{'=' * 50}")
print("STEP 7 — Validate & Save")
print("=" * 50)

# Contar por currency e impact
currency_counts = {}
impact_counts   = {'high': 0, 'medium': 0, 'low': 0}
for ev in final_events:
    c = ev['currency']
    currency_counts[c] = currency_counts.get(c, 0) + 1
    impact_counts[ev.get('impact', 'low')] = \
        impact_counts.get(ev.get('impact', 'low'), 0) + 1

all_dates   = sorted(set(e['dateISO'] for e in final_events))
data_ok     = len(final_events) >= MIN_EVENTS_THRESHOLD
fresh_ok    = len(fresh_events) >= MIN_EVENTS_THRESHOLD

# Determinar fuente
if finnhub_evs and investing_evs:
    source_label = 'Investing.com + Finnhub'
elif investing_evs:
    source_label = 'Investing.com'
elif finnhub_evs:
    source_label = 'Finnhub'
else:
    source_label = 'Base cache only'

output = {
    'lastUpdate':     today.isoformat(),
    'generatedAt':    datetime.utcnow().isoformat() + 'Z',
    'timezoneNote':   'All timeUTC fields are UTC. Frontend converts to user local timezone.',
    'status':         'ok' if data_ok else 'error',
    'source':         source_label,
    'errorMessage':   None if data_ok else 'No se pudieron obtener datos frescos.',
    'fetchErrors':    [],
    'rangeFrom':      all_dates[0] if all_dates else from_date.isoformat(),
    'rangeTo':        to_date.isoformat(),
    'totalEvents':    len(final_events),
    'currencyCounts': currency_counts,
    'impactCounts':   impact_counts,
    'events':         final_events,
}

os.makedirs('calendar-data', exist_ok=True)
with open(CALENDAR_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 65}")
if data_ok:
    print(f"✅ SAVED: {CALENDAR_PATH}")
    print(f"   Source:  {source_label}")
    print(f"   Total:   {len(final_events)} events | Impact: {impact_counts}")
    print(f"   Dates:   {all_dates[0]} → {all_dates[-1]}")
    print(f"   By currency: {dict(sorted(currency_counts.items()))}")
    highs = [e for e in final_events if e.get('impact') == 'high'][:8]
    if highs:
        print(f"\n   HIGH impact events (first 8):")
        for ev in highs:
            fc = f" est:{ev['forecast']}" if ev.get('forecast') else ''
            ac = f" → {ev['actual']}"     if ev.get('actual')   else ''
            print(f"   {ev['dateISO']} {(ev.get('timeUTC') or '?'):5s} "
                  f"[{ev['currency']}] {ev['event'][:50]}{fc}{ac}")
else:
    print(f"⛔ ERROR — {len(final_events)} events saved (threshold={MIN_EVENTS_THRESHOLD})")
print("=" * 65)

# No salir con error aunque fresh falló — el cache previo es válido
if not fresh_ok and data_ok:
    print("\n⚠️  WARNING: Fresh fetch failed/incomplete but base cache preserved.")
    print("   Calendar is serving historical data. Check source connectivity.")

sys.exit(0)  # Siempre exit 0 para no romper el workflow
