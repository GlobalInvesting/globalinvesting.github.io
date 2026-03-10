#!/usr/bin/env python3
"""
update_economic_calendar.py  v11.0 — Trading Economics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fuente:  tradingeconomics.com/calendar
Método:  A) API JSON interna  api.tradingeconomics.com/calendar/country/...
         B) Playwright Chromium (fallback si API falla)
Parser:  Tabla #calendar del HTML renderizado (para Playwright)

Por qué TE es mejor que Investing.com:
  - Importancia codificada en CSS (calendar-date-1/2/3), sin heurísticas
  - Columnas separadas: Actual / Previous / Consensus / Forecast
  - HTML estable y predecible, tabla simple

Timezone: TE muestra en UTC-3. Se suma 3h para obtener UTC real.
          Si la hora cruza medianoche la fecha en UTC cambia (+1 día).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, re, os, sys, time
from datetime import date, datetime, timedelta
from collections import defaultdict, Counter

# ── CONFIG ──────────────────────────────────────────────────────

TRACKED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}
CURRENCY_FLAGS = {
    'USD':'🇺🇸','EUR':'🇪🇺','GBP':'🇬🇧','JPY':'🇯🇵',
    'AUD':'🇦🇺','CAD':'🇨🇦','CHF':'🇨🇭','NZD':'🇳🇿',
}
# TE usa códigos de países ISO 2 letras → moneda
TE_TO_CURRENCY = {
    'US':'USD', 'EA':'EUR', 'EU':'EUR',
    'GB':'GBP', 'JP':'JPY', 'AU':'AUD',
    'CA':'CAD', 'CH':'CHF', 'NZ':'NZD',
}
# TE API JSON usa nombres de país en inglés → moneda
TE_COUNTRY_TO_CURRENCY = {
    'united states':   'USD', 'euro area':        'EUR',
    'germany':         'EUR', 'france':            'EUR',
    'italy':           'EUR', 'spain':             'EUR',
    'netherlands':     'EUR', 'portugal':          'EUR',
    'finland':         'EUR', 'austria':           'EUR',
    'ireland':         'EUR', 'belgium':           'EUR',
    'greece':          'EUR', 'luxembourg':        'EUR',
    'slovakia':        'EUR', 'estonia':           'EUR',
    'latvia':          'EUR', 'lithuania':         'EUR',
    'slovenia':        'EUR', 'malta':             'EUR',
    'cyprus':          'EUR', 'european union':    'EUR',
    'united kingdom':  'GBP', 'japan':             'JPY',
    'australia':       'AUD', 'canada':            'CAD',
    'switzerland':     'CHF', 'new zealand':       'NZD',
}
# Importancia en HTML: calendar-date-1/2/3
TE_IMPACT_MAP = {
    'calendar-date-1': 'low',
    'calendar-date-2': 'medium',
    'calendar-date-3': 'high',
}
# TE API JSON: importance campo numérico (0/1/2/3) o string ('low'/'medium'/'high')
TE_API_IMPORTANCE = {
    0: 'low', 1: 'low', 2: 'medium', 3: 'high',
    '0':'low', '1':'low', '2':'medium', '3':'high',
    'low':'low', 'medium':'medium', 'high':'high',
}

MONTH_ES = {
    1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
    7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic',
}

CALENDAR_PATH = 'calendar-data/calendar.json'
MIN_EVENTS    = 50

# ── KNOWN_IMPACTS: overrides puntuales donde TE puede fallar ──────
# TE ya es bastante preciso. Estos son solo casos edge conocidos.
KNOWN_IMPACTS = sorted([
    # Rate decisions siempre high (TE a veces los pone medium)
    ('fed interest rate decision',  'high'),
    ('fomc press conference',       'high'),
    ('fomc meeting minutes',        'high'),
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
    # TE a veces clasifica estos como medium
    ('non farm payrolls',           'high'),
    ('nonfarm payrolls',            'high'),
    # Versión semanal de ADP es low (la mensual la deja TE como medium)
    ('adp employment change weekly','low'),
    ('nfib business optimism',      'low'),
], key=lambda x: len(x[0]), reverse=True)

def resolve_known_impact(name):
    nl = ' ' + name.lower() + ' '
    for key, impact in KNOWN_IMPACTS:
        if key in nl: return impact
    return None


# ── HELPERS ──────────────────────────────────────────────────────

def fmt_date(d): return f"{d.day} {MONTH_ES[d.month]}"

def clean_val(v):
    if not v: return ''
    s = str(v).strip().replace('®','').strip()
    return '' if s.lower() in ('none','nan','-','n/a','','null','--','na') else s

def impact_rank(i): return {'high':3,'medium':2,'low':1}.get(str(i).lower(),0)

def score_ev(ev):
    return (1000 if ev.get('actual') else 0) + \
           (100  if ev.get('forecast') else 0) + \
           (10   if ev.get('previous') else 0) + \
           len(ev.get('event',''))

def clean_event_name(raw):
    """TE pega el periodo al final: 'ISM Manufacturing PMIDEC' → 'ISM Manufacturing PMI (DEC)'"""
    r = re.sub(
        r'([a-zA-Z0-9%\)])((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?:/\d+)?)$',
        r'\1 (\2)', raw.strip()
    )
    r = re.sub(r'([a-zA-Z0-9%\)])(Q[1-4])$', r'\1 (\2)', r)
    return re.sub(r'\s+', ' ', r).strip()


# ════════════════════════════════════════════════════════════════
# STRATEGY A: TE Internal JSON API
# URL: https://api.tradingeconomics.com/calendar/country/all/{from}/{to}
# Esta URL se usa como botón "Download CSV" del sitio —
# está disponible sin API key cuando se accede desde el browser del sitio.
# En GitHub Actions la intentamos con Referer + cookies de sesión del HTML.
# ════════════════════════════════════════════════════════════════

def fetch_te_json_api(from_str, to_str):
    """
    Intenta la API JSON interna de TE.
    Retorna lista de dicts o [] si falla.
    """
    try:
        import requests
    except ImportError:
        print("  [TE-API] ❌ requests not installed")
        return []

    # La URL que aparece en el botón Download del sitio
    url = f"https://api.tradingeconomics.com/calendar/country/all/{from_str}/{to_str}"

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept':          'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer':         'https://tradingeconomics.com/calendar',
        'Origin':          'https://tradingeconomics.com',
    }

    print(f"  [TE-API] GET {url}")
    try:
        r = requests.get(url, headers=headers, timeout=30)
        print(f"  [TE-API] Status: {r.status_code} | Size: {len(r.content)} bytes")

        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"  [TE-API] ✅ {len(data)} raw records")
                return data
            else:
                print(f"  [TE-API] ⚠️  Unexpected response: {str(data)[:200]}")
                return []
        else:
            print(f"  [TE-API] ❌ HTTP {r.status_code}: {r.text[:200]}")
            return []
    except Exception as e:
        print(f"  [TE-API] ❌ {e}")
        return []


def parse_te_json(records, target_dates):
    """
    Parsea la respuesta JSON de la API de TE.

    Formato de cada registro:
    {
      "CalendarId": "...",
      "Date": "2026-01-05T15:00:00",   ← UTC directamente en la API JSON
      "Country": "United States",
      "Category": "ISM Manufacturing PMI",
      "Event": "ISM Manufacturing PMI",
      "Reference": "Dec",
      "ReferenceDate": "2025-12-31T00:00:00",
      "Source": "Institute for Supply Management",
      "SourceURL": "...",
      "Actual": "47.9",
      "Previous": "48.2",
      "Forecast": "48.3",
      "TEForecast": "48",
      "URL": "/united-states/ism-manufacturing-pmi",
      "Importance": 3,
      "LastUpdate": "2026-01-05T15:00:00"
    }

    IMPORTANTE: La API JSON usa timestamps UTC directamente (no UTC-3).
    """
    events = []
    skipped = 0

    for rec in records:
        try:
            country = (rec.get('Country') or '').strip().lower()
            currency = TE_COUNTRY_TO_CURRENCY.get(country)
            if not currency:
                skipped += 1
                continue

            # El campo Date en la API JSON ya es UTC
            date_str = rec.get('Date') or rec.get('date') or ''
            if not date_str: continue

            try:
                dt_utc = datetime.fromisoformat(date_str.replace('Z',''))
                time_utc = dt_utc.strftime('%H:%M')
                cal_date = dt_utc.date()
            except Exception:
                continue

            if target_dates and cal_date not in target_dates:
                continue

            event_name = (
                rec.get('Event') or rec.get('event') or
                rec.get('Category') or rec.get('category') or ''
            ).strip()
            if not event_name: continue

            # Limpiar nombre (la API a veces tiene el periodo pegado)
            event_name = clean_event_name(event_name)

            imp_raw = rec.get('Importance') or rec.get('importance') or 1
            impact_api = TE_API_IMPORTANCE.get(imp_raw, TE_API_IMPORTANCE.get(str(imp_raw), 'low'))
            known  = resolve_known_impact(event_name)
            impact = known if known else impact_api

            actual   = clean_val(str(rec.get('Actual')   or ''))
            forecast = clean_val(str(rec.get('Forecast') or rec.get('TEForecast') or ''))
            previous = clean_val(str(rec.get('Previous') or ''))

            events.append({
                'date':     fmt_date(cal_date),
                'dateISO':  cal_date.isoformat(),
                'timeUTC':  time_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency,''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast,
                'previous': previous,
            })
        except Exception:
            continue

    print(f"  [TE-API Parser] ✅ {len(events)} events (skipped {skipped} non-tracked)")
    return events


# ════════════════════════════════════════════════════════════════
# STRATEGY B: Playwright → HTML Parser (fallback)
# ════════════════════════════════════════════════════════════════

def parse_te_html(html_content, target_dates=None):
    """
    Parsea tabla#calendar del HTML de TE.
    TE muestra en UTC-3. Convierte sumando 3h.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [HTML] ❌ beautifulsoup4 not installed")
        return []

    soup = BeautifulSoup(html_content, 'lxml')
    table = soup.find('table', id='calendar')
    if not table:
        print("  [HTML] ❌ table#calendar not found")
        return []

    rows = table.find_all('tr')
    print(f"  [HTML] {len(rows)} rows in table#calendar")

    events = []
    skipped = 0

    for row in rows:
        tds = row.find_all('td')
        if len(tds) < 10: continue
        try:
            time_td  = tds[0]
            iso      = tds[3].get_text(strip=True)
            currency = TE_TO_CURRENCY.get(iso)
            if not currency: skipped += 1; continue

            date_cls = [c for c in time_td.get('class',[]) if re.match(r'\d{4}-\d{2}-\d{2}$',c)]
            if not date_cls: skipped += 1; continue
            date_str = date_cls[0]

            time_raw = time_td.get_text(strip=True)
            if time_raw:
                dt_local = datetime.strptime(date_str + ' ' + time_raw, '%Y-%m-%d %I:%M %p')
                dt_utc   = dt_local + timedelta(hours=3)
                time_utc = dt_utc.strftime('%H:%M')
                cal_date = dt_utc.date()
            else:
                time_utc = ''
                cal_date = date.fromisoformat(date_str)

            if target_dates and cal_date not in target_dates: continue

            event_name = clean_event_name(tds[4].get_text(strip=True))
            if not event_name: continue

            span    = time_td.find('span')
            imp_cls = next((c for c in (span.get('class',[]) if span else []) if 'calendar-date-' in c), 'calendar-date-1')
            impact_te = TE_IMPACT_MAP.get(imp_cls, 'low')
            known     = resolve_known_impact(event_name)
            impact    = known if known else impact_te

            actual    = clean_val(tds[5].get_text(strip=True))
            previous  = clean_val(tds[6].get_text(strip=True))
            consensus = clean_val(tds[7].get_text(strip=True))
            forecast  = clean_val(tds[8].get_text(strip=True))
            forecast_final = forecast if forecast else consensus

            events.append({
                'date':     fmt_date(cal_date),
                'dateISO':  cal_date.isoformat(),
                'timeUTC':  time_utc,
                'country':  currency,
                'currency': currency,
                'flag':     CURRENCY_FLAGS.get(currency,''),
                'event':    event_name,
                'impact':   impact,
                'actual':   actual,
                'forecast': forecast_final,
                'previous': previous,
            })
        except Exception:
            continue

    print(f"  [HTML] ✅ {len(events)} events (skipped {skipped})")
    return events


def fetch_te_playwright(from_date, to_date):
    """Playwright → tradingeconomics.com/calendar (fallback)"""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("  [Playwright] ❌ playwright not installed")
        return ''

    print("  [Playwright] Launching Chromium...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=['--no-sandbox','--disable-setuid-sandbox',
                  '--disable-dev-shm-usage',
                  '--disable-blink-features=AutomationControlled'],
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            locale='en-US',
            timezone_id='America/Montevideo',  # UTC-3
            extra_http_headers={'Accept-Language':'en-US,en;q=0.9'},
        )
        context.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )
        page = context.new_page()
        html = ''

        try:
            # TE acepta rango de fechas via startDate/endDate en la URL
            # Navegar y luego usar el selector de fechas
            url = 'https://tradingeconomics.com/calendar'
            print(f"  [Playwright] → {url}")
            page.goto(url, wait_until='networkidle', timeout=60000)
            page.wait_for_selector('table#calendar', timeout=20000)
            page.wait_for_timeout(3000)
            html = page.content()
            print(f"  [Playwright] ✅ {len(html):,} chars")
        except Exception as e:
            print(f"  [Playwright] ❌ {e}")

        browser.close()
    return html


# ════════════════════════════════════════════════════════════════
# DEDUP
# ════════════════════════════════════════════════════════════════

def normalize_for_dedup(name):
    n = name.lower().strip()
    # Eliminar periodos entre paréntesis
    n = re.sub(r'\((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:/\d+)?\)', '', n)
    n = re.sub(r'\(q[1-4]\)', '', n)
    for pat, rep in [
        (r'\by/y\b','yoy'),(r'\bm/m\b','mom'),(r'\bq/q\b','qoq'),
        (r'\bgross domestic product\b','gdp'),(r'\bconsumer price index\b','cpi'),
    ]:
        n = re.sub(pat, rep, n)
    sw = {'a','an','the','of','in','on','at','to','by','or','and',
          'prel','prelim','preliminary','final','flash','revised',
          'sa','nsa','change','s.a','s.a.'}
    words = [w for w in re.sub(r'[^a-z0-9 ]',' ',n).split() if w not in sw and len(w)>1]
    return ' '.join(sorted(words))


def dedup_events(events):
    # Paso A: slot exacto
    eg = defaultdict(list)
    for ev in events:
        eg[(ev['dateISO'], ev.get('timeUTC',''), ev['currency'],
            ev['event'][:30].lower())].append(ev)
    after = []
    for g in eg.values():
        if len(g)==1: after.append(g[0]); continue
        best = dict(max(g, key=score_ev))
        best['impact'] = max(g, key=lambda e: impact_rank(e.get('impact','low')))['impact']
        for ev in g:
            for f in ('actual','forecast','previous'):
                if not best.get(f) and ev.get(f): best[f]=ev[f]
        after.append(best)
    # Paso B: semántico
    sg = defaultdict(list)
    for ev in after:
        sg[(ev['dateISO'], ev['currency'], normalize_for_dedup(ev['event']))].append(ev)
    final = []; merged_n = 0
    for g in sg.values():
        if len(g)==1: final.append(g[0]); continue
        merged_n += len(g)-1
        best = dict(max(g, key=score_ev))
        best['impact'] = max(g, key=lambda e: impact_rank(e.get('impact','low')))['impact']
        for ev in g:
            for f in ('actual','forecast','previous'):
                if not best.get(f) and ev.get(f): best[f]=ev[f]
        final.append(best)
    if merged_n:
        print(f"  [Dedup] Merged {merged_n} dupes: {len(events)} → {len(final)}")
    else:
        print(f"  [Dedup] No duplicates. Total: {len(final)}")
    return final


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

print("="*65)
print("ECONOMIC CALENDAR SCRAPER v11.0 — Trading Economics")
print("Strategy A: api.tradingeconomics.com JSON (no browser needed)")
print("Strategy B: Playwright Chromium (fallback)")
print("="*65)

today     = date.today()
from_date = today - timedelta(days=60)
to_date   = today + timedelta(days=30)
target_dates = {from_date + timedelta(days=i) for i in range((to_date-from_date).days+1)}
print(f"\nRange: {from_date} → {to_date} ({len(target_dates)} days)\n")

# ── STEP 1: Base cache ──────────────────────────────────────────
print("="*50); print("STEP 1 — Load base cache"); print("="*50)
base_events = {}
try:
    with open(CALENDAR_PATH, encoding='utf-8') as f:
        prev = json.load(f)
    for ev in prev.get('events',[]):
        try:
            base_events[(ev['dateISO'],ev['currency'],ev['event'][:30].lower().strip())] = ev
        except: pass
    bd = sorted(set(e['dateISO'] for e in base_events.values()))
    print(f"  Loaded {len(base_events)} events ({bd[0] if bd else '?'} → {bd[-1] if bd else '?'})")
except FileNotFoundError:
    print("  No previous cache — starting fresh")
except Exception as e:
    print(f"  Warning: {e}")

# ── STEP 2: Fetch ───────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 2 — Fetch Trading Economics\n{'='*50}")
from_str = from_date.isoformat()
to_str   = to_date.isoformat()

fresh_events = []
fetch_method = ''

# Estrategia A: API JSON
print("\n  [Strategy A] TE JSON API...")
te_records = fetch_te_json_api(from_str, to_str)
if te_records:
    fresh_events = parse_te_json(te_records, target_dates)
    if fresh_events:
        fetch_method = 'Trading Economics API (JSON)'

# Estrategia B: Playwright (si A falla)
if not fresh_events:
    print("\n  [Strategy B] Playwright fallback...")
    html = fetch_te_playwright(from_date, to_date)
    if html:
        fresh_events = parse_te_html(html, target_dates)
        if fresh_events:
            fetch_method = 'Trading Economics (Playwright)'

print(f"\n  Fresh events: {len(fresh_events)}")
if not fresh_events:
    print("  ⚠️  All strategies failed — using base cache only")
    fetch_method = 'Base cache only'

# ── STEP 3: Merge con base cache ────────────────────────────────
print(f"\n{'='*50}\nSTEP 3 — Merge with base cache\n{'='*50}")
if fresh_events:
    merged = dict(base_events)
    added = updated = 0
    for ev in fresh_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
        if k in merged:
            ex = dict(merged[k])
            for f in ('actual','forecast','previous','timeUTC','impact'):
                if ev.get(f): ex[f] = ev[f]
            merged[k] = ex; updated += 1
        else:
            merged[k] = ev; added += 1
    all_events = list(merged.values())
    print(f"  base={len(base_events)} + fresh={len(fresh_events)} → {len(all_events)} (added={added}, updated={updated})")
else:
    all_events = list(base_events.values())
    print(f"  Using base cache: {len(all_events)}")

# ── STEP 4: Dedup ───────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 4 — Dedup\n{'='*50}")
final_events = dedup_events(all_events)

# Filtrar eventos sin datos
before = len(final_events)
final_events = [e for e in final_events
                if e.get('timeUTC','').strip() or e.get('actual','').strip()
                or e.get('forecast','').strip() or e.get('previous','').strip()]
if len(final_events) < before:
    print(f"  Filtered {before-len(final_events)} no-data events")

# Sort
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    return ev['dateISO'] + 'T' + (t if re.match(r'\d{2}:\d{2}',t) else '00:00')
final_events.sort(key=sort_key)

# ── STEP 5: Save ────────────────────────────────────────────────
print(f"\n{'='*50}\nSTEP 5 — Save\n{'='*50}")
cc = dict(sorted(Counter(e['currency'] for e in final_events).items()))
ic = dict(Counter(e['impact'] for e in final_events))
ad = sorted(set(e['dateISO'] for e in final_events))
data_ok = len(final_events) >= MIN_EVENTS

output = {
    'lastUpdate':    today.isoformat(),
    'generatedAt':   datetime.now().isoformat() + 'Z',
    'timezoneNote':  'All timeUTC are UTC. TE local time (UTC-3) converted by +3h.',
    'status':        'ok' if data_ok else 'error',
    'source':        fetch_method,
    'errorMessage':  None if data_ok else 'No fresh data available.',
    'fetchErrors':   [],
    'rangeFrom':     ad[0] if ad else from_date.isoformat(),
    'rangeTo':       to_date.isoformat(),
    'totalEvents':   len(final_events),
    'currencyCounts':cc,
    'impactCounts':  ic,
    'events':        final_events,
}
os.makedirs('calendar-data', exist_ok=True)
with open(CALENDAR_PATH,'w',encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*65}")
if data_ok:
    print(f"✅ SAVED: {CALENDAR_PATH}")
    print(f"   Source:  {fetch_method}")
    print(f"   Total:   {len(final_events)} | Impact: {ic}")
    print(f"   Dates:   {ad[0] if ad else '?'} → {ad[-1] if ad else '?'}")
    print(f"   By currency: {cc}")
    highs = [e for e in final_events if e.get('impact')=='high'][:6]
    if highs:
        print(f"\n   HIGH events (sample):")
        for ev in highs:
            ac = f" → {ev['actual']}" if ev.get('actual') else ''
            print(f"   {ev['dateISO']} {(ev.get('timeUTC') or '?'):5} [{ev['currency']}] {ev['event'][:52]}{ac}")
else:
    print(f"⛔ Only {len(final_events)} events (min={MIN_EVENTS}) — check connectivity")
print("="*65)
sys.exit(0)
