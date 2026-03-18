#!/usr/bin/env python3
"""
update_economic_calendar.py  v13.1 — FXStreet suplementario para actuals de hoy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fuentes (en orden de prioridad):
  A) investing.com API interna — HTML con actuals, sin browser
  B) FXStreet Calendar API   — JSON fallback con actuals
  C) Cache previo            — último recurso

v13.1 — FXStreet siempre corre para hoy como fuente suplementaria:
  - MODO SMART: si NO hay eventos de alto impacto en la próxima
    hora, solo refresca hoy+mañana (ventana corta, rápido).
  - MODO FULL: si HAY eventos próximos (o es la primera hora del
    día), refresca ventana completa -60/+30 días.
  - FXStreet SIEMPRE corre para hoy como fuente suplementaria:
    investing.com usa hora ET para sus displays, lo que introduce
    un lag de hasta 1h respecto a la hora real de publicación.
    FXStreet usa UTC nativo y publica el actual antes. Al combinar
    ambas fuentes se captura el dato en el run inmediato posterior.
  - Sin checkout de repo privado: el script vive en el repo público.

Cero dependencias externas de pago. Sin Playwright.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, re, os, sys, time
from datetime import date, datetime, timedelta
from collections import defaultdict, Counter

# ── CONFIG ──────────────────────────────────────────────────────────────────

TRACKED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}
CURRENCY_FLAGS = {
    'USD':'🇺🇸','EUR':'🇪🇺','GBP':'🇬🇧','JPY':'🇯🇵',
    'AUD':'🇦🇺','CAD':'🇨🇦','CHF':'🇨🇭','NZD':'🇳🇿',
}

# MQL5 country_id → currency
# Obtenidos de la documentación pública de la API MQL5
MQL5_COUNTRY_CURRENCY = {
    5:   'USD',  # United States
    6:   'USD',  # United States (Fed)
    11:  'EUR',  # Euro Area / ECB
    12:  'EUR',  # Germany
    14:  'EUR',  # France
    17:  'EUR',  # Italy
    26:  'EUR',  # Spain
    29:  'GBP',  # United Kingdom
    35:  'JPY',  # Japan
    37:  'AUD',  # Australia
    39:  'NZD',  # New Zealand
    40:  'CAD',  # Canada
    48:  'CHF',  # Switzerland
    # Países adicionales del área euro
    15:  'EUR',  # Netherlands
    16:  'EUR',  # Belgium
    18:  'EUR',  # Portugal
    19:  'EUR',  # Austria
    20:  'EUR',  # Finland
    21:  'EUR',  # Greece
    22:  'EUR',  # Ireland
    23:  'EUR',  # Luxembourg
}

# Mapeo texto de país → currency (para parsers alternativos)
COUNTRY_TEXT_TO_CURRENCY = {
    'united states':   'USD', 'euro area':        'EUR', 'european union': 'EUR',
    'germany':         'EUR', 'france':            'EUR', 'italy':          'EUR',
    'spain':           'EUR', 'netherlands':       'EUR', 'portugal':       'EUR',
    'finland':         'EUR', 'austria':           'EUR', 'ireland':        'EUR',
    'belgium':         'EUR', 'greece':            'EUR', 'luxembourg':     'EUR',
    'united kingdom':  'GBP', 'japan':             'JPY', 'australia':      'AUD',
    'canada':          'CAD', 'switzerland':       'CHF', 'new zealand':    'NZD',
}

# MQL5 importance: 1=low, 2=medium, 3=high
MQL5_IMPORTANCE = {1: 'low', 2: 'medium', 3: 'high', '1':'low', '2':'medium', '3':'high'}

MONTH_ES = {
    1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
    7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic',
}

CALENDAR_PATH = 'calendar-data/calendar.json'
MIN_EVENTS    = 50

# ── KNOWN IMPACTS: overrides para eventos donde la fuente puede errar ────────
KNOWN_IMPACTS = sorted([
    ('fed interest rate decision',  'high'),
    ('fomc press conference',       'high'),
    ('fomc meeting minutes',        'high'),
    ('fomc minutes',                'high'),
    ('boe interest rate decision',  'high'),
    ('boe monetary policy report',  'high'),
    ('ecb interest rate decision',  'high'),
    ('ecb press conference',        'high'),
    ('rba interest rate decision',  'high'),
    ('boc interest rate decision',  'high'),
    ('boc press conference',        'high'),
    ('snb interest rate decision',  'high'),
    ('boj interest rate decision',  'high'),
    ('boj quarterly outlook',       'high'),
    ('rbnz interest rate decision', 'high'),
    ('non farm payrolls',           'high'),
    ('nonfarm payrolls',            'high'),
    ('nfib business optimism',      'low'),
    ('adp employment change weekly','low'),
], key=lambda x: len(x[0]), reverse=True)

def resolve_known_impact(name):
    nl = ' ' + name.lower() + ' '
    for key, impact in KNOWN_IMPACTS:
        if key in nl: return impact
    return None


# ── HELPERS ──────────────────────────────────────────────────────────────────

def fmt_date(d): return f"{d.day} {MONTH_ES[d.month]}"

def clean_val(v):
    if not v: return ''
    s = str(v).strip().replace('®','').strip()
    return '' if s.lower() in ('none','nan','-','n/a','','null','--','na') else s

def impact_rank(i): return {'high':3,'medium':2,'low':1}.get(str(i).lower(), 0)

def score_ev(ev):
    return (1000 if ev.get('actual') else 0) + \
           (100  if ev.get('forecast') else 0) + \
           (10   if ev.get('previous') else 0) + \
           len(ev.get('event',''))

def clean_event_name(raw):
    """Normaliza nombre: elimina periodo pegado al final si lo hay."""
    r = re.sub(
        r'([a-zA-Z0-9%\)])((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:/\d+)?)$',
        r'\1 (\2)', raw.strip()
    )
    r = re.sub(r'([a-zA-Z0-9%\)])(Q[1-4])$', r'\1 (\2)', r)
    return re.sub(r'\s+', ' ', r).strip()


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY A: investing.com Economic Calendar (API interna, sin browser)
# POST https://www.investing.com/economic-calendar/Service/getCalendarFilteredData
# Devuelve HTML con tabla completa incl. actual/forecast/previous
# País IDs: USD=5, EUR=72, GBP=4, JPY=35, AUD=25, CAD=6, CHF=12, NZD=43
# ════════════════════════════════════════════════════════════════════════════

# investing.com country_id → currency
INV_COUNTRY_CURRENCY = {
    '5':  'USD', '72': 'EUR', '4':  'GBP', '35': 'JPY',
    '25': 'AUD', '6':  'CAD', '12': 'CHF', '43': 'NZD',
}
INV_COUNTRY_IDS = list(INV_COUNTRY_CURRENCY.keys())

# investing.com devuelve los horarios en Eastern Time.
# ET = EDT (UTC-4) durante DST o EST (UTC-5) fuera de DST.
# Calculamos el offset real por fecha para manejar el cambio estacional.
# Reglas US DST: segundo domingo de marzo → primer domingo de noviembre.

def _et_utc_offset(d):
    """Devuelve timedelta para convertir ET→UTC según si d está en EDT o EST."""
    year = d.year
    # Segundo domingo de marzo
    march1 = date(year, 3, 1)
    dst_start = march1 + timedelta(days=(6 - march1.weekday()) % 7 + 7)
    # Primer domingo de noviembre
    nov1 = date(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    in_edt = dst_start <= d < dst_end
    return timedelta(hours=4) if in_edt else timedelta(hours=5)


def fetch_investing(from_date, to_date):
    """
    Descarga el calendario de investing.com via su API interna.

    Endpoint: POST /economic-calendar/Service/getCalendarFilteredData
    Devuelve JSON con campo 'data' = HTML de la tabla del calendario.
    Sin Playwright, sin autenticación, sin API key.

    Nota: la API tiene rate-limit (~10 req/min). Usamos chunks de 14 días
    y delay de 1.5s para mantenernos dentro del límite.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"  [INV] ❌ Missing: {e}")
        return []

    url = 'https://www.investing.com/economic-calendar/Service/getCalendarFilteredData'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept':           'application/json, text/javascript, */*; q=0.01',
        'Accept-Language':  'en-US,en;q=0.9',
        'Content-Type':     'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer':          'https://www.investing.com/economic-calendar/',
        'Origin':           'https://www.investing.com',
    }

    # Limitar el rango: últimos 14 días + próximos 21 días
    # El cache cubre el historial; solo necesitamos ventana reciente.
    #
    # IMPORTANTE: se usa chunk de 1 día (dateFrom == dateTo) en lugar de 14 días.
    # Razón: cuando se pide un rango multi-día, investing.com no siempre emite
    # un <tr class="theDay"> para cada día intermedio. El parser queda con
    # cur_date = chunk_start para todos los eventos sin header de día propio,
    # lo que colapsa múltiples días en una sola fecha (bug de misassignment).
    # Con chunk = 1 día, dateFrom == dateTo == la fecha real → todos los eventos
    # quedan en su día correcto independientemente del HTML que devuelva investing.
    today      = date.today()
    eff_from   = max(from_date, today - timedelta(days=14))
    eff_to     = min(to_date,   today + timedelta(days=21))

    # Omitir fines de semana (investing no tiene eventos Sáb/Dom y devuelve HTML vacío)
    def is_weekday(d): return d.weekday() < 5

    all_events = []
    cur_day = eff_from
    while cur_day <= eff_to:
        if not is_weekday(cur_day):
            cur_day += timedelta(days=1)
            continue

        body_parts = [f'country[]={cid}' for cid in INV_COUNTRY_IDS]
        body_parts += ['importance[]=1', 'importance[]=2', 'importance[]=3']
        body_parts += [
            f'dateFrom={cur_day.isoformat()}',
            f'dateTo={cur_day.isoformat()}',   # mismo día → 1 día por request
            'currentTab=custom', 'submitFilters=1', 'limit_from=0',
        ]
        body = '&'.join(body_parts)

        print(f"  [INV] POST {cur_day}")
        try:
            r = requests.post(url, data=body, headers=headers, timeout=30)
            print(f"  [INV] Status: {r.status_code} | Size: {len(r.content):,} bytes")
            if r.status_code == 200:
                resp = r.json()
                html = resp.get('data', '')
                if html:
                    evs = parse_investing_html(html, cur_day, cur_day)
                    all_events.extend(evs)
                    print(f"  [INV] ✅ +{len(evs)} events")
                else:
                    print(f"  [INV] ⚠️ data vacío. Keys: {list(resp.keys())[:6]}")
            elif r.status_code == 429:
                print(f"  [INV] ⚠️ 429 rate-limit — deteniendo")
                break
            else:
                print(f"  [INV] ❌ HTTP {r.status_code}")
        except Exception as e:
            print(f"  [INV] ❌ {cur_day}: {e}")

        cur_day += timedelta(days=1)
        time.sleep(1.5)  # cortesía: ~40 req/min → 1.5s da margen cómodo

    print(f"  [INV] Total: {len(all_events)} events")
    return all_events


def parse_investing_html(html_fragment, from_date, to_date):
    """
    Parsea el HTML del calendario de investing.com.

    Cada fila de fecha:
      <tr id="theDay..." class="theDay"><td colspan="7">Friday, March 13, 2026</td></tr>

    Cada fila de evento:
      <tr event_attr_id="..." data-country-id="5">
        <td class="... time">13:30</td>
        <td class="left flagCur noWrap"><span class="ceFlags USD"></span> USD</td>
        <td class="... sentiment imp">  ← bulls count = importance
          <i class="grayFullBullishIcon"></i>  ← activo
          <i class="grayFullBullishIcon"></i>
          <i class="grayFullBullishIcon"></i>
        </td>
        <td class="left event"><a href="...">Non-Farm Payrolls</a></td>
        <td class="bold act blackFont">143K</td>   ← actual
        <td class="fore">170K</td>                 ← forecast
        <td class="prev">307K</td>                 ← previous
      </tr>
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html_fragment, 'lxml')

    events    = []
    cur_date  = from_date

    for row in soup.find_all('tr'):
        row_id    = row.get('id', '')
        row_class = ' '.join(row.get('class', []))

        # Fila de fecha
        if 'theDay' in row_id or 'theDay' in row_class:
            td = row.find('td')
            if td:
                txt = td.get_text(strip=True)
                for fmt in ('%A, %B %d, %Y', '%B %d, %Y', '%b %d, %Y'):
                    try:
                        cur_date = datetime.strptime(txt, fmt).date()
                        break
                    except Exception:
                        pass
            continue

        # Fila de evento — investing.com usa data-event-datetime (no data-country-id)
        # La moneda está en columna 1 como texto "USD", "EUR", etc.
        if not (row.get('data-event-datetime') or row.get('data-country-id') or row.get('event_country_id')):
            continue

        tds = row.find_all('td')
        if len(tds) < 4:
            continue

        try:
            # Hora (columna 0)
            time_raw = tds[0].get_text(strip=True)

            # Moneda (columna 1): extraer código de 3 letras del texto
            col1_text = tds[1].get_text(strip=True) if len(tds) > 1 else ''
            # El texto puede ser "USD" o incluir el flag span — buscar 3 letras mayúsculas
            currency_match = re.search(r'\b([A-Z]{3})\b', col1_text)
            if not currency_match:
                # Fallback: data-country-id si existe
                country_id = row.get('data-country-id') or row.get('event_country_id')
                currency = INV_COUNTRY_CURRENCY.get(str(country_id)) if country_id else None
                if not currency:
                    continue
            else:
                currency = currency_match.group(1)
                if currency not in TRACKED_CURRENCIES:
                    continue
            # investing.com devuelve horas en ET (EDT=UTC-4 en verano, EST=UTC-5 en invierno).
            # Convertir a UTC usando el offset real para la fecha del evento.
            # Si el resultado cruza medianoche, event_date avanza un día.
            time_utc   = ''
            event_date = cur_date
            if time_raw and re.match(r'\d{1,2}:\d{2}', time_raw):
                et_str = time_raw.zfill(5)[:5]
                try:
                    et_dt  = datetime.strptime(f"{cur_date.isoformat()} {et_str}", "%Y-%m-%d %H:%M")
                    utc_dt = et_dt + _et_utc_offset(cur_date)
                    time_utc   = utc_dt.strftime('%H:%M')
                    event_date = utc_dt.date()
                except Exception:
                    time_utc   = et_str   # fallback: guardar sin convertir

            # Importancia (columna 2: contar grayFullBullishIcon)
            imp_td   = tds[2] if len(tds) > 2 else None
            imp_src  = 'low'
            if imp_td:
                n_full = len(imp_td.find_all('i', class_=lambda c: c and 'Full' in c))
                imp_src = {3: 'high', 2: 'medium', 1: 'low'}.get(n_full, 'low')

            # Nombre del evento (columna 3 o td con class "event")
            ev_td = next(
                (td for td in tds if 'event' in ' '.join(td.get('class', []))),
                tds[3] if len(tds) > 3 else None
            )
            if not ev_td:
                continue
            event_name = clean_event_name(ev_td.get_text(strip=True))
            if not event_name:
                continue

            # Actual / Forecast / Previous (columnas 4, 5, 6)
            def _td(i): return clean_val(tds[i].get_text(strip=True)) if len(tds) > i else ''

            actual   = _td(4)
            forecast = _td(5)
            previous = _td(6)

            known  = resolve_known_impact(event_name)
            impact = known if known else imp_src

            events.append({
                'date':     fmt_date(event_date),
                'dateISO':  event_date.isoformat(),
                'timeUTC':  time_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency, ''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
            })
        except Exception:
            continue

    return events


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY B: FXStreet Calendar API (fallback con actuals reales)
# Endpoint: https://calendar.fxstreet.com/eventdate/
# JSON puro, sin autenticación, incluye actual/previous/consensus
# Nota: Forexfactory CDN NO incluye "actual" en su feed JSON — por eso no se usa.
# ════════════════════════════════════════════════════════════════════════════

# FXStreet currency → nuestro código
FXSTREET_CURRENCY_MAP = {
    'USD':'USD', 'EUR':'EUR', 'GBP':'GBP', 'JPY':'JPY',
    'AUD':'AUD', 'CAD':'CAD', 'CHF':'CHF', 'NZD':'NZD',
}

# FXStreet importance: 0=low, 1=medium, 2=high  (o "Low"/"Medium"/"High")
FXSTREET_IMPORTANCE = {
    0:'low', 1:'medium', 2:'high',
    '0':'low', '1':'medium', '2':'high',
    'Low':'low', 'Medium':'medium', 'High':'high',
    'low':'low', 'medium':'medium', 'high':'high',
    'NonRated':'low', 'Nonrated':'low',
}


def fetch_fxstreet(from_date, to_date):
    """
    Descarga el calendario de FXStreet via JSON API.
    Incluye actual/previous/consensus para todos los eventos.
    
    Endpoint: https://calendar.fxstreet.com/eventdate/
    Parámetros: timezone=UTC, fromdate, todate, volatility (0=low,1=med,2=high)
    Sin autenticación. JSON directo.
    """
    try:
        import requests
    except ImportError:
        print("  [FXS] ❌ requests not installed")
        return []

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, */*',
        'Referer': 'https://www.fxstreet.com/economic-calendar',
        'Origin':  'https://www.fxstreet.com',
    }

    # FXStreet usa un rango de fechas en formato ISO
    from_str = from_date.strftime('%Y-%m-%dT00:00:00')
    to_str   = to_date.strftime('%Y-%m-%dT23:59:59')

    # Solicitar los tres niveles de importancia por separado o todos a la vez
    url = 'https://calendar.fxstreet.com/eventdate/'
    params = {
        'timezone': 'UTC',
        'fromdate': from_str,
        'todate':   to_str,
        'volatility': '0,1,2',  # todos los niveles
    }

    print(f"  [FXS] GET {url}?timezone=UTC&fromdate={from_date}&todate={to_date}")
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"  [FXS] Status: {r.status_code} | Size: {len(r.content):,} bytes")

        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"  [FXS] ✅ {len(data)} raw records")
                return parse_fxstreet_json(data, from_date, to_date)
            elif isinstance(data, dict):
                # Algunos endpoints devuelven wrapper
                for key in ('data', 'events', 'calendar', 'items'):
                    if key in data and isinstance(data[key], list):
                        print(f"  [FXS] ✅ {len(data[key])} records (key='{key}')")
                        return parse_fxstreet_json(data[key], from_date, to_date)
                print(f"  [FXS] ⚠️ Dict keys: {list(data.keys())[:6]}")
                return []
            else:
                print(f"  [FXS] ⚠️ Respuesta vacía o inesperada")
                return []
        else:
            print(f"  [FXS] ❌ HTTP {r.status_code}: {r.text[:200]}")
            return []
    except Exception as e:
        print(f"  [FXS] ❌ {e}")
        return []


def parse_fxstreet_json(records, from_date, to_date):
    """
    Parsea JSON de FXStreet Calendar API.

    Estructura real confirmada (2026-03):
    {
      "InternationalCode": "US",        ← ISO 2 country code → currency
      "Country":           "United States",
      "Currency":          "USD",       ← a veces es currency ISO, a veces país
      "Name":              "Non-Farm Payrolls",
      "DateTime": {
        "Date": "2026-02-07T13:30:00Z", ← UTC directo
        ...
      },
      "Volatility":  "High",            ← string "High"/"Medium"/"Low"/"NonRated"
      "Actual":      "143K",            ← string ya formateado (o null)
      "Consensus":   "170K",            ← forecast
      "Previous":    "307K",
      ...
    }
    """
    # ISO 2-letter country code → currency (para InternationalCode)
    ISO2_CURRENCY = {
        'US': 'USD',
        'EU': 'EUR', 'DE': 'EUR', 'FR': 'EUR', 'IT': 'EUR', 'ES': 'EUR',
        'NL': 'EUR', 'PT': 'EUR', 'FI': 'EUR', 'AT': 'EUR', 'IE': 'EUR',
        'BE': 'EUR', 'GR': 'EUR', 'LU': 'EUR', 'SK': 'EUR', 'EE': 'EUR',
        'LV': 'EUR', 'LT': 'EUR', 'SI': 'EUR', 'MT': 'EUR', 'CY': 'EUR',
        'GB': 'GBP',
        'JP': 'JPY',
        'AU': 'AUD',
        'CA': 'CAD',
        'CH': 'CHF',
        'NZ': 'NZD',
    }

    events = []
    skipped = 0
    target_set = {from_date + timedelta(days=i)
                  for i in range((to_date - from_date).days + 1)}

    for rec in records:
        try:
            # ── Currency ─────────────────────────────────────────────────────
            # Prioridad: InternationalCode (ISO2 country) > Country name > Currency field
            currency = ISO2_CURRENCY.get(rec.get('InternationalCode', '').upper().strip())
            if not currency:
                country_name = (rec.get('Country') or '').lower().strip()
                currency = COUNTRY_TEXT_TO_CURRENCY.get(country_name)
            if not currency:
                # Currency field puede ser "USD" directamente en algunos registros
                cur_raw = (rec.get('Currency') or '').upper().strip()
                if cur_raw in TRACKED_CURRENCIES:
                    currency = cur_raw
            if not currency:
                skipped += 1
                continue

            # ── Fecha/hora ───────────────────────────────────────────────────
            # DateTime es un objeto: {"Date": "2026-02-07T13:30:00Z", ...}
            dt_obj   = rec.get('DateTime') or {}
            date_str = ''
            if isinstance(dt_obj, dict):
                date_str = (dt_obj.get('Date') or dt_obj.get('date') or '').replace('Z', '')
            if not date_str:
                # Fallback: campo Date directo
                date_str = (rec.get('Date') or rec.get('date') or '').replace('Z', '')
            if not date_str:
                skipped += 1
                continue
            try:
                dt_utc   = datetime.fromisoformat(date_str)
                cal_date = dt_utc.date()
                time_utc = dt_utc.strftime('%H:%M')
            except Exception:
                skipped += 1
                continue

            if cal_date not in target_set:
                skipped += 1
                continue

            # ── Nombre ───────────────────────────────────────────────────────
            event_name = clean_event_name((rec.get('Name') or '').strip())
            if not event_name:
                skipped += 1
                continue

            # ── Impacto ──────────────────────────────────────────────────────
            imp_raw    = rec.get('Volatility') or rec.get('volatility') or 0
            impact_src = FXSTREET_IMPORTANCE.get(imp_raw,
                         FXSTREET_IMPORTANCE.get(str(imp_raw), 'low'))
            known  = resolve_known_impact(event_name)
            impact = known if known else impact_src

            # ── Valores ──────────────────────────────────────────────────────
            # Puede ser string formateado ("143K") o null
            def _fxs_val(key):
                v = rec.get(key)
                if v is None:
                    return ''
                return clean_val(str(v))

            actual   = _fxs_val('Actual')
            forecast = _fxs_val('Consensus') or _fxs_val('Forecast')
            previous = _fxs_val('Previous')

            events.append({
                'date':     fmt_date(cal_date),
                'dateISO':  cal_date.isoformat(),
                'timeUTC':  time_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency, ''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
            })
        except Exception:
            continue

    print(f"  [FXS Parser] ✅ {len(events)} events (skipped {skipped})")
    return events


# ════════════════════════════════════════════════════════════════════════════
# DEDUP (idéntico al v11, funciona bien)
# ════════════════════════════════════════════════════════════════════════════

def normalize_for_dedup(name):
    """
    Normaliza nombres de eventos para deduplicación cross-fuente.
    Estrategia:
      1) Expandir abreviaturas de frecuencia entre paréntesis → palabras libres
      2) Stripear todos los demás paréntesis
      3) Sustituir variantes de nombres entre fuentes
      4) Eliminar stopwords y ordenar tokens
    """
    n = name.lower().strip()

    # 1) Frecuencias entre paréntesis → tokens libres antes de stripear
    n = re.sub(r'\(m/m\)', ' mom ', n)
    n = re.sub(r'\(y/y\)', ' yoy ', n)
    n = re.sub(r'\(q/q\)', ' qoq ', n)
    n = re.sub(r'\(mom\)', ' mom ', n)
    n = re.sub(r'\(yoy\)', ' yoy ', n)
    n = re.sub(r'\(qoq\)', ' qoq ', n)

    # 2) Stripear todos los paréntesis restantes (periodos, Q1-Q4, "prel", etc.)
    n = re.sub(r'\([^)]*\)', ' ', n)

    # 3) Sustituciones cross-fuente
    for pat, rep in [
        # Frecuencias sueltas residuales
        (r'\by/y\b', 'yoy'), (r'\bm/m\b', 'mom'), (r'\bq/q\b', 'qoq'),
        # Nombres largos → cortos
        (r'\bgross domestic product\b', 'gdp'),
        (r'\bconsumer price index\b', 'cpi'),
        (r'\bpersonal consumption expenditures\b', 'pce'),
        (r'\bcore personal consumption expenditures\b', 'core pce'),
        # CFTC: "speculative net positions" == "NC Net Positions"
        (r'\bnc net positions\b', 'cftc net'),
        (r'\bspeculative net positions\b', 'cftc net'),
        # GDP variants entre fuentes
        (r'\bgdp growth\b', 'gdp'),
        (r'\bgdp annualized\b', 'gdp'),
        (r'\bgdp price\b', 'gdp deflator'),
        (r'\bgdp deflator\b', 'gdp deflator'),
        # Calificadores de revisión
        (r'\b2nd\b', ''), (r'\b3rd\b', ''), (r'\badv\b', ''),
    ]:
        n = re.sub(pat, rep, n)

    sw = {
        'a','an','the','of','in','on','at','to','by','or','and',
        'prel','prelim','preliminary','final','flash','revised','adv','est',
        'sa','nsa','s.a','s.a.',
        'change','rate','index','price','prices',
        'qoq',   # implícito en releases trimestrales: no diferencia GDP QoQ vs "Annualized"
    }
    words = [w for w in re.sub(r'[^a-z0-9 ]', ' ', n).split() if w not in sw and len(w) > 1]
    return ' '.join(sorted(words))
def dedup_events(events):
    # Paso A: slot exacto (misma fecha/hora/divisa/nombre)
    eg = defaultdict(list)
    for ev in events:
        eg[(ev['dateISO'], ev.get('timeUTC',''), ev['currency'],
            ev['event'][:30].lower())].append(ev)
    after = []
    for g in eg.values():
        if len(g) == 1:
            after.append(g[0])
            continue
        best = dict(max(g, key=score_ev))
        best['impact'] = max(g, key=lambda e: impact_rank(e.get('impact','low')))['impact']
        for ev in g:
            for f in ('actual', 'forecast', 'previous'):
                if not best.get(f) and ev.get(f): best[f] = ev[f]
        after.append(best)
    # Paso B: semántico (mismo evento, misma fecha, mismo currency)
    sg = defaultdict(list)
    for ev in after:
        sg[(ev['dateISO'], ev['currency'], normalize_for_dedup(ev['event']))].append(ev)
    final = []
    merged_n = 0
    for g in sg.values():
        if len(g) == 1:
            final.append(g[0])
            continue
        merged_n += len(g) - 1
        best = dict(max(g, key=score_ev))
        best['impact'] = max(g, key=lambda e: impact_rank(e.get('impact','low')))['impact']
        for ev in g:
            for f in ('actual', 'forecast', 'previous'):
                if not best.get(f) and ev.get(f): best[f] = ev[f]
        final.append(best)
    if merged_n:
        print(f"  [Dedup] Merged {merged_n} dupes: {len(events)} → {len(final)}")
    else:
        print(f"  [Dedup] No duplicates. Total: {len(final)}")
    return final


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("ECONOMIC CALENDAR v13.1 — smart-fetch + FXStreet suplementario")
print("Strategy A: investing.com (API interna, actuals reales)")
print("Strategy B: FXStreet Calendar API (siempre para hoy + fallback)")
print("Strategy C: Cache previo (último recurso)")
print("=" * 65)

now_utc   = datetime.utcnow()
today     = now_utc.date()
now_hour  = now_utc.hour

# ── SMART-FETCH: decidir modo según hora y eventos próximos ──────────────────
# MODO FULL  : primer run del día (hora 0) o forzado → ventana -60/+30 días
# MODO SMART : resto de horas → ventana hoy ± 2 días (más rápido)
# Tras cargar el cache, se sube a FULL si hay eventos de alto impacto
# en la próxima hora (sin actual todavía) → captura el dato en seguida.

FORCE_FULL = (now_hour == 0)          # primer run del día siempre full
smart_from = today - timedelta(days=2)
smart_to   = today + timedelta(days=2)
full_from  = today - timedelta(days=60)
full_to    = today + timedelta(days=30)

# ── STEP 1: Cargar cache previo ──────────────────────────────────────────────
print("=" * 50)
print("STEP 1 — Load base cache")
print("=" * 50)
base_events = {}
try:
    with open(CALENDAR_PATH, encoding='utf-8') as f:
        prev = json.load(f)
    for ev in prev.get('events', []):
        try:
            base_events[(ev['dateISO'], ev['currency'],
                         ev['event'][:30].lower().strip())] = ev
        except Exception:
            pass
    bd = sorted(set(e['dateISO'] for e in base_events.values()))
    print(f"  Loaded {len(base_events)} events "
          f"({bd[0] if bd else '?'} → {bd[-1] if bd else '?'})")
    # Limpiar duplicados del cache: conservar solo fecha más futura sin actual
    by_name = defaultdict(list)
    for k, ev in list(base_events.items()):
        norm_key = (ev['currency'], normalize_for_dedup(ev['event']))
        by_name[norm_key].append(k)
    removed = 0
    for norm_key, keys in by_name.items():
        if len(keys) <= 1:
            continue
        keys_no_actual = [k for k in keys if not base_events[k].get('actual','').strip()]
        if len(keys_no_actual) <= 1:
            continue
        keys_no_actual.sort(key=lambda k: k[0])
        for k in keys_no_actual[:-1]:
            del base_events[k]
            removed += 1
    if removed:
        print(f"  Cache dedup: removed {removed} stale cross-date duplicates")
except FileNotFoundError:
    print("  No previous cache — starting fresh")
    FORCE_FULL = True   # sin cache siempre full
except Exception as e:
    print(f"  Warning loading cache: {e}")

# ── Detectar eventos de alto impacto en la próxima hora (sin actual) ─────────
# Si existen → subir a MODO FULL para capturar el dato en cuanto salga
def has_upcoming_high_impact(cache, now_dt, window_minutes=75):
    """
    Retorna True si hay eventos high-impact sin actual en los próximos
    window_minutes minutos respecto a now_dt (UTC).
    """
    cutoff = now_dt + timedelta(minutes=window_minutes)
    for ev in cache.values():
        if ev.get('impact') != 'high':
            continue
        if ev.get('actual','').strip():
            continue   # ya tiene dato — no urgente
        t = ev.get('timeUTC','').strip()
        d = ev.get('dateISO','').strip()
        if not t or not d or not re.match(r'\d{2}:\d{2}', t):
            continue
        try:
            ev_dt = datetime.strptime(f"{d}T{t}", "%Y-%m-%dT%H:%M")
            if now_dt <= ev_dt <= cutoff:
                return True, ev['event'], ev['currency']
        except Exception:
            pass
    return False, '', ''

if not FORCE_FULL and base_events:
    upcoming, ev_name, ev_ccy = has_upcoming_high_impact(base_events, now_utc)
    if upcoming:
        FORCE_FULL = True
        print(f"  🔔 Evento próximo detectado: [{ev_ccy}] {ev_name} → MODO FULL")
    else:
        print(f"  ✅ Sin eventos próximos → MODO SMART (ventana ±2 días)")

fetch_mode = 'FULL' if FORCE_FULL else 'SMART'
from_date  = full_from  if FORCE_FULL else smart_from
to_date    = full_to    if FORCE_FULL else smart_to
print(f"\nMODE: {fetch_mode}  |  Range: {from_date} → {to_date}  |  UTC now: {now_utc.strftime('%H:%M')}\n")

# ── STEP 2: Fetch ────────────────────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 2 — Fetch ({fetch_mode})\n{'='*50}")

fresh_events  = []
fetch_method  = ''
fetch_errors  = []

# Estrategia A: investing.com — ventana completa según modo
print("\n  [Strategy A] investing.com API...")
try:
    inv_events = fetch_investing(from_date, to_date)
    if inv_events:
        fresh_events = inv_events
        fetch_method = 'investing.com'
        print(f"  ✅ Strategy A: {len(fresh_events)} events")
    else:
        fetch_errors.append('investing.com returned 0 events')
        print("  ⚠️  Strategy A: 0 events")
except Exception as e:
    fetch_errors.append(f'investing.com error: {e}')
    print(f"  ❌ Strategy A failed: {e}")

# Estrategia B: FXStreet
#
# Se ejecuta SIEMPRE para hoy (fxs_today_from / fxs_today_to).
# Razón: investing.com muestra las horas en ET (Eastern Time) y las
# convierte a UTC internamente. Esto introduce un lag de hasta 60 min
# respecto a la hora real de publicación del dato (ej: NZD GDP sale a
# las 21:45 UTC pero investing.com lo lista como 22:45 UTC por la
# conversión ET→UTC). FXStreet usa UTC nativo y publica el actual en
# el momento correcto. Al combinar ambas fuentes siempre se captura
# el dato en el run inmediato posterior a que salga, sin esperar a
# que investing.com "alcance" la hora ET convertida.
#
# Adicionalmente, si Strategy A falló o devolvió menos de lo esperado,
# FXStreet también cubre la ventana completa como fallback.
fxs_today_from = today
fxs_today_to   = today
min_expected   = 5 if fetch_mode == 'SMART' else MIN_EVENTS
run_fxs_full   = len(fresh_events) < min_expected

print(f"\n  [Strategy B] FXStreet — hoy suplementario + {'ventana completa' if run_fxs_full else 'solo hoy'}...")
try:
    # Siempre fetch de hoy para capturar actuals con UTC nativo
    fxs_today = fetch_fxstreet(fxs_today_from, fxs_today_to)
    # Si A falló, también fetch de ventana completa
    fxs_full  = fetch_fxstreet(from_date, to_date) if run_fxs_full else []

    # Combinar: full primero, today encima (today sobreescribe en merge)
    fxs_events = fxs_full + fxs_today
    # Dedup rápido por clave exacta para no inflar el conteo
    fxs_seen = {}
    for ev in fxs_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
        # Preferir el que tenga actual
        if k not in fxs_seen or (ev.get('actual') and not fxs_seen[k].get('actual')):
            fxs_seen[k] = ev
    fxs_events = list(fxs_seen.values())

    if fxs_events:
        if fresh_events:
            combined = list(fresh_events)
            combined.extend(fxs_events)
            fresh_events = combined
            fetch_method = 'investing.com + FXStreet'
        else:
            fresh_events = fxs_events
            fetch_method = 'FXStreet'
        today_cnt = len([e for e in fxs_events if e.get('dateISO') == today.isoformat()])
        print(f"  ✅ Strategy B: {len(fxs_events)} events ({today_cnt} de hoy con UTC nativo)")
    else:
        fetch_errors.append('FXStreet returned 0 events')
        print("  ⚠️  Strategy B: 0 events")
except Exception as e:
    fetch_errors.append(f'FXStreet error: {e}')
    print(f"  ❌ Strategy B failed: {e}")

print(f"\n  Fresh total: {len(fresh_events)}")
if not fresh_events:
    print("  ⚠️  All strategies failed — using base cache only")
    fetch_method = 'cache'

# ── STEP 3: Merge con cache ──────────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 3 — Merge\n{'='*50}")
if fresh_events:
    merged = dict(base_events)
    added = updated = actuals_added = 0
    for ev in fresh_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
        if k in merged:
            ex = dict(merged[k])
            for f in ('actual', 'forecast', 'previous', 'timeUTC', 'impact'):
                if ev.get(f):
                    if f == 'actual' and not ex.get('actual'):
                        actuals_added += 1
                    ex[f] = ev[f]
            merged[k] = ex
            updated += 1
        else:
            merged[k] = ev
            added += 1
    all_events = list(merged.values())
    print(f"  base={len(base_events)} + fresh={len(fresh_events)} → "
          f"{len(all_events)} (added={added}, updated={updated}, new_actuals={actuals_added})")
else:
    all_events = list(base_events.values())
    print(f"  Using base cache: {len(all_events)}")

# ── STEP 4: Dedup ────────────────────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 4 — Dedup\n{'='*50}")
final_events = dedup_events(all_events)

# Filtrar eventos sin ningún dato útil
before = len(final_events)
final_events = [
    e for e in final_events
    if (e.get('timeUTC','').strip() or e.get('actual','').strip()
        or e.get('forecast','').strip() or e.get('previous','').strip())
]
if len(final_events) < before:
    print(f"  Filtered {before - len(final_events)} no-data events")

# Sort por fecha+hora
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    return ev['dateISO'] + 'T' + (t if re.match(r'\d{2}:\d{2}', t) else '00:00')

final_events.sort(key=sort_key)

# ── STEP 5: Guardar ──────────────────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 5 — Save\n{'='*50}")

cc = dict(sorted(Counter(e['currency'] for e in final_events).items()))
ic = dict(Counter(e['impact'] for e in final_events))
ad = sorted(set(e['dateISO'] for e in final_events))
data_ok = len(final_events) >= MIN_EVENTS

output = {
    'lastUpdate':    today.isoformat(),
    'generatedAt':   datetime.utcnow().isoformat() + 'Z',
    'fetchMode':     fetch_mode,
    'timezoneNote':  'All timeUTC are UTC.',
    'status':        'ok' if data_ok else 'error',
    'source':        fetch_method,
    'errorMessage':  None if data_ok else 'No fresh data available.',
    'fetchErrors':   fetch_errors,
    'rangeFrom':     ad[0] if ad else from_date.isoformat(),
    'rangeTo':       full_to.isoformat(),   # siempre mostrar rango completo en metadata
    'totalEvents':   len(final_events),
    'currencyCounts': cc,
    'impactCounts':  ic,
    'events':        final_events,
}

os.makedirs('calendar-data', exist_ok=True)
with open(CALENDAR_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*65}")
if data_ok:
    print(f"✅ SAVED: {CALENDAR_PATH}  [{fetch_mode}]")
    print(f"   Source:  {fetch_method}")
    print(f"   Total:   {len(final_events)} | Impact: {ic}")
    print(f"   Dates:   {ad[0] if ad else '?'} → {ad[-1] if ad else '?'}")
    print(f"   By currency: {cc}")
    highs = [e for e in final_events if e.get('impact') == 'high'][-8:]
    if highs:
        print(f"\n   HIGH events (recientes):")
        for ev in highs:
            ac = f" → {ev['actual']}" if ev.get('actual') else ' (pendiente)'
            print(f"   {ev['dateISO']} {(ev.get('timeUTC') or '?'):5} "
                  f"[{ev['currency']}] {ev['event'][:52]}{ac}")
else:
    print(f"⛔ Only {len(final_events)} events (min={MIN_EVENTS}) — check connectivity")
    print(f"   Errors: {fetch_errors}")
print("=" * 65)
sys.exit(0 if data_ok else 1)
