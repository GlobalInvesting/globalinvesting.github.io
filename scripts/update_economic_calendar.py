#!/usr/bin/env python3
"""
update_economic_calendar.py
Descarga y parsea el calendario económico de eventos macro para las 8 divisas.

Extraído desde .github/workflows/update-economic-calendar.yml como parte de la
refactorización P-01 (auditoría de código).
"""

import requests
from bs4 import BeautifulSoup
import json, re, time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

# ── CONSTANTS ────────────────────────────────────────────────────────

TRACKED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}

CURRENCY_FLAGS = {
    'USD': '🇺🇸', 'EUR': '🇪🇺', 'GBP': '🇬🇧', 'JPY': '🇯🇵',
    'AUD': '🇦🇺', 'CAD': '🇨🇦', 'CHF': '🇨🇭', 'NZD': '🇳🇿',
}

COUNTRY_TO_CURRENCY = {
    'united states': 'USD', 'us': 'USD',
    'euro area': 'EUR', 'european': 'EUR', 'eurozone': 'EUR',
    'germany': 'EUR', 'france': 'EUR', 'italy': 'EUR', 'spain': 'EUR',
    'united kingdom': 'GBP', 'uk': 'GBP', 'britain': 'GBP',
    'japan': 'JPY', 'australia': 'AUD', 'canada': 'CAD',
    'switzerland': 'CHF', 'new zealand': 'NZD',
}

HIGH_KW = [
    'interest rate', 'rate decision', 'fed funds', 'bank rate', 'policy rate',
    'gdp', 'gross domestic product', 'cpi', 'consumer price index',
    'nonfarm payrolls', 'nonfarm payroll',
    'unemployment rate', 'jobless rate',
    'fomc', 'ecb', 'boe', 'boj', 'rba', 'boc', 'snb', 'rbnz',
    'monetary policy statement', 'retail sales',
    'trade balance', 'current account',
]
MED_KW = [
    'employment change', 'claimant count', 'unemployment claims',
    'inflation', 'consumer price', 'pmi', 'purchasing managers',
    'manufacturing', 'industrial production', 'factory orders',
    'housing', 'building permits', 'consumer confidence', 'business confidence',
    'ppi', 'producer price', 'wage', 'earnings', 'ism', 'durable goods',
    'job openings', 'jolts', 'import price', 'export price', 'services pmi',
    'nfib', 'small business', 'existing home', 'new home', 'pending home',
    'adp', 'redbook',
]

MONTH_ES = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
            7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

def classify_impact_kw(name):
    n = name.lower()
    for k in HIGH_KW:
        if k in n: return 'high'
    for k in MED_KW:
        if k in n: return 'medium'
    return 'low'

def resolve_currency(text):
    if not text: return None
    t = text.strip()
    if t.upper() in TRACKED_CURRENCIES: return t.upper()
    for key, curr in COUNTRY_TO_CURRENCY.items():
        if key in t.lower(): return curr
    return None

def fmt_date(dt_obj):
    return f"{dt_obj.day} {MONTH_ES[dt_obj.month]}"

def clean_val(v):
    if not v: return ''
    s = str(v).strip()
    return '' if s.lower() in ('none','nan','-','n/a','') else s

def parse_desc_html(raw_desc):
    """
    Parse actual/forecast/previous/impact from description field.
    Handles HTML format and plain text format.
    """
    actual = forecast = previous = impact = ''
    if not raw_desc: return actual, forecast, previous, impact

    m = re.search(r'<b>\s*Actual\s*:?\s*</b>\s*([^<\n]*)', raw_desc, re.I)
    if m: actual = clean_val(m.group(1))

    m = re.search(r'<b>\s*(?:Forecast|Consensus)\s*:?\s*</b>\s*([^<\n]*)', raw_desc, re.I)
    if m: forecast = clean_val(m.group(1))

    m = re.search(r'<b>\s*Previous\s*:?\s*</b>\s*([^<\n]*)', raw_desc, re.I)
    if m: previous = clean_val(m.group(1))

    m = re.search(r'<b>\s*Impact\s*:?\s*</b>\s*([^<\n]*)', raw_desc, re.I)
    if m:
        vl = m.group(1).strip().lower()
        impact = 'high' if 'high' in vl else 'medium' if 'medium' in vl else 'low'

    if not actual and not forecast:
        m = re.search(r'Actual[:\s]+([^\.\n,<;]+)', raw_desc, re.I)
        if m: actual = clean_val(m.group(1))
        m = re.search(r'(?:Forecast|Consensus)[:\s]+([^\.\n,<;]+)', raw_desc, re.I)
        if m: forecast = clean_val(m.group(1))
        m = re.search(r'Previous[:\s]+([^\.\n,<;]+)', raw_desc, re.I)
        if m: previous = clean_val(m.group(1))

    return actual, forecast, previous, impact


# ════════════════════════════════════════════════════════════════════
# SOURCE: Investing.com
# IMPORTANT: Investing.com HTML POST returns times in EST (UTC-5).
# JSON API returns UTC ISO strings directly.
# We convert EST → UTC by adding 5 hours before storing.
# ════════════════════════════════════════════════════════════════════

def est_to_utc(time_str, event_date):
    """
    Convert EST (UTC-5) time string 'HH:MM' to UTC.
    Returns (utc_time_str, possibly_adjusted_date).
    Investing.com always returns EST regardless of timeZone parameter.
    """
    if not time_str:
        return time_str, event_date
    try:
        h, m = map(int, time_str.split(':'))
        total_minutes = h * 60 + m + 300  # +5 hours = EST → UTC
        day_overflow = total_minutes >= 1440
        total_minutes = total_minutes % 1440
        utc_str = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
        if day_overflow and event_date:
            event_date = event_date + timedelta(days=1)
        return utc_str, event_date
    except:
        return time_str, event_date
def fetch_investing_calendar(from_str, to_str, target_dates):
    """
    Fetches calendar via Investing.com JSON API (Next.js __NEXT_DATA__).
    Times are returned as UTC ISO strings. Impact is 1/2/3 → low/medium/high.
    Falls back to legacy HTML scraping if the JSON API fails.
    """
    print(f"  [Investing] Fetching {from_str} → {to_str}")

    IMPACT_MAP = {'1': 'low', '2': 'medium', '3': 'high',
                   1: 'low',   2: 'medium',   3: 'high'}

    # ── Strategy A: JSON API (/api/economic-calendar) ──────────────────
    def fetch_via_json_api():
        url = 'https://www.investing.com/economic-calendar/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          'Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        params = {'dateFrom': from_str, 'dateTo': to_str}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if not r.ok:
            raise ValueError(f"HTTP {r.status_code}")

        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                      r.text, re.DOTALL)
        if not m:
            raise ValueError("No __NEXT_DATA__ found")

        data = json.loads(m.group(1))
        store = (data.get('props', {})
                     .get('pageProps', {})
                     .get('state', {})
                     .get('economicCalendarStore', {})
                     .get('calendarEventsByDate', {}))
        if not store:
            raise ValueError("Empty economicCalendarStore")

        events = []
        for date_key, day_events in store.items():
            for ev in day_events:
                try:
                    currency = str(ev.get('currency', '')).upper()
                    if currency not in TRACKED_CURRENCIES:
                        continue

                    # time is UTC ISO: "2026-03-09T01:30:00Z"
                    time_iso = ev.get('time', '') or ev.get('actual_time', '')
                    event_date = None
                    time_utc = ''
                    if time_iso:
                        dt = datetime.fromisoformat(time_iso.replace('Z', '+00:00'))
                        event_date = dt.date()
                        time_utc = dt.strftime('%H:%M')
                        # Events at 00:00–03:59 UTC belong to the PREVIOUS calendar day
                        # (they are late-night Asia/Pacific events: 20:00-23:59 local Investing.com display)
                        if dt.hour < 4:
                            event_date = event_date - timedelta(days=1)
                    else:
                        try:
                            event_date = date.fromisoformat(date_key)
                        except Exception:
                            continue

                    if event_date not in target_dates:
                        continue

                    imp_raw = ev.get('importance', '1')
                    impact = IMPACT_MAP.get(imp_raw, 'low')

                    # Prefer eventLong name, fall back to event
                    event_name = (ev.get('eventLong') or ev.get('event') or '').strip()
                    if not event_name:
                        continue

                    actual   = clean_val(str(ev.get('actual',   '') or ''))
                    forecast = clean_val(str(ev.get('forecast', '') or ''))
                    previous = clean_val(str(ev.get('previous', '') or ''))

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

        print(f"  [Investing JSON] ✅ Parsed {len(events)} events")
        return events

    # ── Strategy B: legacy HTML POST fallback ──────────────────────────
    def fetch_via_html_post():
        print("  [Investing] Falling back to HTML POST scraping")
        url = 'https://www.investing.com/economic-calendar/Service/getCalendarFilteredData'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          'Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.investing.com/economic-calendar/',
            'Origin': 'https://www.investing.com',
        }
        post_data = [
            ('dateFrom', from_str), ('dateTo', to_str),
            ('timeZone', '0'), ('timeFilter', 'timeRemain'),
            ('currentTab', 'custom'), ('limit_from', '0'),
        ]
        for cid in [5, 72, 4, 35, 25, 6, 12, 43]:
            post_data.append(('country[]', str(cid)))

        r = requests.post(url, headers=headers, data=post_data, timeout=25)
        if not r.ok:
            print(f"  [Investing HTML] HTTP {r.status_code}")
            return []
        resp = r.json()
        html_data = resp.get('data', '') if isinstance(resp, dict) else ''
        if not html_data:
            return []

        soup = BeautifulSoup(html_data, 'lxml')
        rows = soup.find_all('tr', {'id': re.compile('eventRowId_')})
        print(f"  [Investing HTML] Found {len(rows)} rows")

        # Debug: print first row structure to diagnose parsing issues
        if rows:
            r0 = rows[0]
            print(f"  [Investing HTML] Sample row attrs: data-currency={repr(r0.attrs.get('data-currency'))} "
                  f"data-event-datetime={repr(r0.attrs.get('data-event-datetime'))}")
            tds = r0.find_all('td')
            if tds:
                print(f"  [Investing HTML] TD classes sample: {[' '.join(td.get('class',[])) for td in tds[:6]]}")

        events = []
        _dbg_no_date = _dbg_no_cur = _dbg_no_name = _dbg_out_range = 0
        for row in rows:
            try:
                dt_str = row.get('data-event-datetime', '')
                if not dt_str:
                    _dbg_no_date += 1
                    continue
                dt_norm = dt_str.strip().replace('T', ' ')
                event_date = None
                for fmt in ['%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                            '%Y/%m/%d %H:%M',    '%Y-%m-%d %H:%M']:
                    try:
                        event_date = datetime.strptime(dt_norm[:19], fmt).date()
                        break
                    except Exception:
                        pass
                if event_date is None:
                    _dbg_no_date += 1
                    continue

                m_t = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}[ T](\d{2}):(\d{2})', dt_norm)
                raw_time_est = f"{m_t.group(1)}:{m_t.group(2)}" if m_t else ''
                time_str_utc, event_date = est_to_utc(raw_time_est, event_date)

                extended_dates = target_dates | {max(target_dates) + timedelta(days=1)}
                if event_date not in extended_dates:
                    _dbg_out_range += 1
                    continue

                currency = row.attrs.get('data-currency', '') or ''
                # Fallback: read from the currency <td> if attr missing
                if not currency or currency not in TRACKED_CURRENCIES:
                    cur_td = row.find('td', class_=re.compile(r'flagCur|currency'))
                    if cur_td:
                        currency = cur_td.get_text(strip=True).upper()
                if currency not in TRACKED_CURRENCIES:
                    _dbg_no_cur += 1
                    continue

                # Impact: check data-img on row for bull count
                impact = 'low'
                row_img = (row.get('data-img', '') + ' ' +
                           row.get('data-img_pair', '')).lower()
                sent_td = row.find('td', class_=re.compile('sentiment'))
                sentiment_text = ''
                if sent_td:
                    sentiment_text = ' '.join(
                        ' '.join(el.get('class', [])) + ' ' + el.get('title', '') + ' ' +
                        el.get('data-tooltip', '')
                        for el in sent_td.find_all(True)
                    ).lower()
                combined = row_img + ' ' + sentiment_text
                if any(x in combined for x in ['bull3', '3bull', 'highimpact', 'high impact', 'redicon']):
                    impact = 'high'
                elif any(x in combined for x in ['bull2', '2bull', 'mediumimpact', 'medium impact', 'orangeicon']):
                    impact = 'medium'

                # Find event name td — avoid actual/forecast/prev cells that also have 'event' in class
                ev_td = None
                for _td in row.find_all('td'):
                    _cls = ' '.join(_td.get('class', []))
                    if 'event' in _cls and not any(x in _cls for x in ('actual', 'forecast', 'prev', 'sentiment')):
                        ev_td = _td
                        break
                event_name = ''
                if ev_td:
                    a = ev_td.find('a')
                    event_name = (a or ev_td).get_text(strip=True)
                    event_name = re.sub(r'\s+[A-Z]{3}/\d+$', '', event_name).strip()
                if not event_name:
                    continue

                def gcell(pat):
                    td = row.find('td', id=re.compile(pat))
                    return clean_val(td.get_text(strip=True)) if td else ''

                actual   = gcell(r'eventActual_')
                forecast = gcell(r'eventForecast_')
                previous = gcell(r'eventPrevious_')

                if impact == 'low':
                    impact = classify_impact_kw(event_name)

                events.append({
                    'date':     fmt_date(event_date),
                    'dateISO':  event_date.isoformat(),
                    'timeUTC':  time_str_utc,
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

        print(f"  [Investing HTML] ✅ Parsed {len(events)} events "
              f"(skipped: no_date={_dbg_no_date}, out_range={_dbg_out_range}, "
              f"no_currency={_dbg_no_cur}, no_name={_dbg_no_name})")
        return events

    # ── Run Strategy A (JSON API), enrich/fallback with Strategy B (HTML POST) ──
    # JSON API only returns ~30 events from page state — not enough for ESI Component A.
    # HTML POST returns 200+ with full actuals. Always run both and keep the richer one.
    json_events = []
    html_events = []

    try:
        json_events = fetch_via_json_api()
        print(f"  [Investing] JSON API: {len(json_events)} events")
    except Exception as e:
        import traceback
        print(f"  [Investing] JSON API error: {e}")
        print(f"  [Investing] {traceback.format_exc()[:400]}")

    # Run HTML POST if JSON returned < 50 events (likely incomplete) OR as enrichment
    if len(json_events) < 50:
        try:
            html_events = fetch_via_html_post()
            print(f"  [Investing] HTML POST: {len(html_events)} events")
        except Exception as e:
            print(f"  [Investing] HTML fallback error: {e}")

    # Use whichever source has more events; merge actuals+impact from JSON into HTML result
    if len(html_events) >= len(json_events):
        # HTML POST has more coverage — use as base, enrich with JSON API data
        # JSON API is the authoritative source for impact (importance field is exact)
        json_by_key = {}
        for ev in json_events:
            k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
            json_by_key[k] = ev
        merged = []
        for ev in html_events:
            k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
            if k in json_by_key:
                updated = dict(ev)
                # Always inherit impact from JSON API — it maps directly from Investing's importance field
                updated['impact'] = json_by_key[k]['impact']
                # Inherit actual if JSON has it and HTML doesn't
                if json_by_key[k].get('actual') and not ev.get('actual'):
                    updated['actual'] = json_by_key[k]['actual']
                merged.append(updated)
            else:
                merged.append(ev)
        print(f"  [Investing] Using HTML POST ({len(html_events)} events) as primary source")
        return merged
    elif json_events:
        print(f"  [Investing] Using JSON API ({len(json_events)} events) as primary source")
        return json_events
    else:
        print("  [Investing] Both sources failed — returning empty")
        return []
# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ECONOMIC CALENDAR SCRAPER v6.4 (Investing.com only)")
print("Source: Investing.com (JSON API + HTML POST fallback)")
print("Timezone policy:")
print("  - Investing.com JSON API: returns UTC ISO strings directly")
print("  - Investing.com HTML POST: returns EST (UTC-5), converted +5h to UTC")
print("  - All timeUTC fields are UTC — frontend converts to user local timezone")
print("=" * 60)

import sys

today     = date.today()
from_date = today - timedelta(days=90)  # v6.4: extendido de 28→90 días para ESI más robusto (3 meses de sorpresas)
to_date   = today + timedelta(days=30)

target_dates = set()
d = from_date
while d <= to_date:
    target_dates.add(d); d += timedelta(days=1)

print(f"\nTarget: {from_date} → {to_date} ({len(target_dates)} days)\n")

CALENDAR_PATH = 'calendar-data/calendar.json'

# ── STEP 1: Load the existing JSON as the base (full historical accumulation) ──
# All past events are preserved indefinitely in the JSON.
# Fresh data from live sources will OVERWRITE matching events (updating actuals/forecasts).
base_events = {}   # key → event dict
try:
    with open(CALENDAR_PATH, encoding='utf-8') as _f:
        _prev = json.load(_f)
    kept = 0
    for _ev in _prev.get('events', []):
        try:
            _k = (_ev['dateISO'], _ev['currency'], _ev['event'][:25].lower().strip())
            base_events[_k] = _ev
            kept += 1
        except Exception:
            pass
    all_dates = sorted(set(e['dateISO'] for e in base_events.values()))
    print(f"  [Base] Loaded {kept} events from previous JSON "
          f"(dates: {all_dates[:3]}...{all_dates[-3:] if len(all_dates) > 3 else ''})")
except FileNotFoundError:
    print("  [Base] No previous calendar.json — starting fresh")
except Exception as _e:
    print(f"  [Base] Could not load previous calendar: {_e}")

# ── STEP 2: Fetch from Investing.com — única fuente (v6.3) ───────────────────
# Investing.com devuelve actual + forecast + previous en una sola llamada.
# Se hacen DOS pasadas:
#   Pasada A: últimos 28 días (para obtener actuals de eventos publicados)
#   Pasada B: próximos 30 días (para obtener forecast de eventos futuros)
# No hay matching ni enriquecimiento separado — todo viene de la misma fuente.
# ─────────────────────────────────────────────────────────────────────────────
all_events  = []
source_used = None
fetch_errors = []

past_from = (today - timedelta(days=90)).isoformat()
past_to   = today.isoformat()
future_to = to_date.isoformat()

print(f"\n{'='*50}")
print(f"SOURCE: Investing.com (past: {past_from}→{past_to} + future: {past_to}→{future_to})")
print(f"{'='*50}")

# Pasada A: pasado con actuals
inv_past = []
try:
    inv_past = fetch_investing_calendar(past_from, past_to, target_dates)
    has_actuals = sum(1 for e in inv_past if e.get('actual'))
    print(f"✅ Investing.com past: {len(inv_past)} events ({has_actuals} with actuals)")
except Exception as e:
    print(f"❌ Investing.com past failed: {e}")
    fetch_errors.append(f"Investing.com past: {e}")

# Pasada B: futuro con forecasts
inv_future = []
try:
    inv_future = fetch_investing_calendar(past_to, future_to, target_dates)
    print(f"✅ Investing.com future: {len(inv_future)} events")
except Exception as e:
    print(f"❌ Investing.com future failed: {e}")
    fetch_errors.append(f"Investing.com future: {e}")

# Combinar ambas pasadas — past wins on actuals, future adds upcoming events
combined_by_key = {}
for ev in inv_future:
    k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
    combined_by_key[k] = ev
for ev in inv_past:
    k = (ev['dateISO'], ev['currency'], ev['event'][:30].lower().strip())
    if k in combined_by_key:
        # Past wins on actuals/previous, keep future forecast if not overridden
        existing = dict(combined_by_key[k])
        if ev.get('actual'):   existing['actual']   = ev['actual']
        if ev.get('previous'): existing['previous'] = ev['previous']
        if ev.get('forecast'): existing['forecast'] = ev['forecast']
        combined_by_key[k] = existing
    else:
        combined_by_key[k] = ev

all_events = list(combined_by_key.values())
if all_events:
    source_used = 'Investing.com'
    print(f"\n  [Combined] {len(inv_past)} past + {len(inv_future)} future → {len(all_events)} unique events")
else:
    print("⛔ Investing.com returned no events")

if not all_events and not base_events:
    print(f"\n{'='*50}")
    print("⛔ ALL SOURCES FAILED AND NO BASE DATA — saving empty calendar")
    print("=" * 50)

# ── STEP 3: Build final event list ────────────────────────────────────────────
if all_events:
    # Merge with base — fresh Investing.com data always wins
    merged_values = dict(base_events)
    fresh_added = fresh_updated = 0
    for ev in all_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())
        if k in merged_values:
            updated = dict(merged_values[k])
            # Fresh source wins on all fields
            if ev.get('forecast'): updated['forecast'] = ev['forecast']
            if ev.get('previous'): updated['previous'] = ev['previous']
            if ev.get('actual'):   updated['actual']   = ev['actual']
            # Keep highest impact (fresh data sometimes downgrades it after publication)
            impact_rank = {'high': 3, 'medium': 2, 'low': 1}
            if impact_rank.get(ev['impact'], 1) > impact_rank.get(updated.get('impact', 'low'), 1):
                updated['impact'] = ev['impact']
            updated['timeUTC']  = ev.get('timeUTC', updated.get('timeUTC', ''))
            merged_values[k] = updated
            fresh_updated += 1
        else:
            # Try adjacent date (day-boundary events: 23:50 Mar9 ↔ 00:50 Mar10)
            matched_boundary = False
            try:
                ev_date = date.fromisoformat(ev['dateISO'])
                for delta in (-1, 1):
                    adj_date = (ev_date + timedelta(days=delta)).isoformat()
                    k_adj = (adj_date, ev['currency'], ev['event'][:25].lower().strip())
                    if k_adj in merged_values:
                        updated = dict(merged_values[k_adj])
                        if ev.get('forecast'): updated['forecast'] = ev['forecast']
                        if ev.get('previous'): updated['previous'] = ev['previous']
                        if ev.get('actual'):   updated['actual']   = ev['actual']
                        # Keep highest impact
                        impact_rank = {'high': 3, 'medium': 2, 'low': 1}
                        if impact_rank.get(ev.get('impact','low'), 1) > impact_rank.get(updated.get('impact','low'), 1):
                            updated['impact'] = ev['impact']
                        # Do NOT overwrite timeUTC — keep the existing (correct) time
                        merged_values[k_adj] = updated
                        fresh_updated += 1
                        matched_boundary = True
                        break
            except Exception:
                pass
            if not matched_boundary:
                merged_values[k] = ev
                fresh_added += 1
    print(f"  [Merge] base={len(base_events)} + fresh={len(all_events)} → {len(merged_values)} total "
          f"(added={fresh_added}, updated={fresh_updated})")
    actuals_count = sum(1 for ev in merged_values.values() if ev.get('actual'))
    print(f"  [Actuals] {actuals_count} events have actual values")
else:
    merged_values = base_events
    print(f"  [Merge] Fresh failed — using base only: {len(base_events)} events")

# ── STEP 3b-pre: Remove timezone-shift duplicates ────────────────────────────
def normalize_name(name):
    n = name.lower()
    # Preserve YoY/MoM/QoQ distinctions BEFORE stripping (e.g. "PPI (YoY)" vs "PPI (MoM)")
    n = re.sub(r'\(\s*(y/?o?y|m/?o?m|q/?o?q)\s*\)', lambda m: ' ' + m.group(1).replace('/',''), n)
    # Also normalize inline variants without parentheses: y/y → yoy, m/m → mom, q/q → qoq
    n = re.sub(r'\by/y\b', 'yoy', n)
    n = re.sub(r'\bm/m\b', 'mom', n)
    n = re.sub(r'\bq/q\b', 'qoq', n)
    n = re.sub(r'\s*\([^)]*\)', '', n)           # remove remaining parenthetical content
    n = re.sub(r'[^a-z0-9 ]', ' ', n)            # non-alphanumeric → space
    n = re.sub(r'\b[ymq]\b', '', n)              # remove single-letter artifacts
    # Remove country/noise words
    n = re.sub(r'\b(japan|us|uk|germany|german|france|french|eurozone|euro area|australia|canada|swiss|switzerland|new zealand|final|prelim|preliminary|gross domestic product)\b', '', n)
    n = re.sub(r'\b(in jpy|in usd|in eur|in gbp)\b', '', n)
    # For names with >=3 meaningful tokens, strip YoY/MoM/QoQ (they're just variants of same indicator)
    tokens = [t for t in n.split() if t and t not in ('yoy','mom','qoq','yy','mm','qq')]
    if len(tokens) >= 3:
        n = re.sub(r'\b(yoy|mom|qoq|yy|mm|qq)\b', '', n)
    n = n.replace('gdp', '').replace('cpi', '')
    return re.sub(r'\s+', ' ', n).strip()
# FF JSON uses local time → event appears on Mar 9 with no time (timeUTC="")
# FF XML uses UTC → same event appears on Mar 10 at 00:01 UTC
# Rule: if two events have same (currency, normalized_name) and dates differ by
# exactly 1 day, and ONE has no time (timeUTC==""), keep the one WITH a time.
from collections import defaultdict
name_cur_groups = defaultdict(list)
for k, ev in list(merged_values.items()):
    key = (ev['currency'], normalize_name(ev['event']))
    name_cur_groups[key].append((k, ev))

tz_dupes_removed = 0
for key, items in name_cur_groups.items():
    if len(items) < 2:
        continue
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            k_a, ev_a = items[i]
            k_b, ev_b = items[j]
            if k_a not in merged_values or k_b not in merged_values:
                continue
            try:
                d_a = date.fromisoformat(ev_a['dateISO'])
                d_b = date.fromisoformat(ev_b['dateISO'])
            except Exception:
                continue
            diff = abs((d_a - d_b).days)
            if diff != 1:
                continue
            has_time_a = bool(ev_a.get('timeUTC', '').strip())
            has_time_b = bool(ev_b.get('timeUTC', '').strip())
            if has_time_a == has_time_b:
                # Both have times — check if it's a day-boundary shift (e.g. 23:50 Mar9 vs 00:50 Mar10)
                # This happens when past-pass and future-pass store same event with slightly different UTC hours
                if has_time_a and has_time_b:
                    try:
                        h_a = int(ev_a['timeUTC'].split(':')[0])
                        h_b = int(ev_b['timeUTC'].split(':')[0])
                        # One is late night (>= 22h) and other is early morning (<= 2h), dates differ by 1
                        is_boundary = (h_a >= 22 and h_b <= 2) or (h_b >= 22 and h_a <= 2)
                        if not is_boundary:
                            continue
                        # Merge: keep the one with actual data, or the earlier date version
                        has_actual_a = bool(ev_a.get('actual', '').strip())
                        has_actual_b = bool(ev_b.get('actual', '').strip())
                        keep_k  = k_a if (has_actual_a or d_a < d_b) else k_b
                        drop_k  = k_b if keep_k == k_a else k_a
                        keep_ev = dict(merged_values[keep_k])
                        drop_ev = merged_values[drop_k]
                        for field in ('actual', 'forecast', 'previous'):
                            if not keep_ev.get(field) and drop_ev.get(field):
                                keep_ev[field] = drop_ev[field]
                        merged_values[keep_k] = keep_ev
                        del merged_values[drop_k]
                        tz_dupes_removed += 1
                    except Exception:
                        pass
                continue  # not a TZ dupe in the no-time case
            # One has no time, one has a time → TZ duplicate
            # Keep the one with a time (it has UTC time from FF XML)
            # Also merge actual/forecast/previous from whichever has them
            keep_k   = k_a if has_time_a else k_b
            drop_k   = k_b if has_time_a else k_a
            keep_ev  = dict(merged_values[keep_k])
            drop_ev  = merged_values[drop_k]
            for field in ('actual', 'forecast', 'previous'):
                if not keep_ev.get(field) and drop_ev.get(field):
                    keep_ev[field] = drop_ev[field]
            merged_values[keep_k] = keep_ev
            del merged_values[drop_k]
            tz_dupes_removed += 1

if tz_dupes_removed:
    print(f"  [TZ-dedup] Removed {tz_dupes_removed} timezone-shift duplicates")

# ── STEP 3b: Conservative dedup — only remove true aliases (same event, different source name) ──
# "Japan Leading Index MoM" == "Leading Index (MoM) (Jan)" → deduplicate (short alias ≤4 tokens contained in longer)
# "GDP QoQ" vs "GDP Capital Expenditure QoQ" → keep both (short has ≤4 tokens but NOT contained since extra words differ)
# "3-Month Bill Auction" vs "6-Month Bill Auction" → keep both (different events in same slot)

def score_event(ev):
    s = 0
    if ev.get('actual'):   s += 1000
    if ev.get('forecast'): s += 100
    if ev.get('previous'): s += 10
    s += len(ev.get('event', ''))
    return s

def are_true_duplicates(name_a, name_b):
    """Deduplica si los nombres comparten suficiente contenido semántico.
    Estrategia dual:
    1. Alias corto: un nombre (≤4 tokens) está contenido en el otro,
       siempre que no haya palabras diferenciadores (core, 3-month, etc.)
    2. Solapamiento semántico: comparten ≥2 palabras clave de contenido (no stopwords)
    """
    def _norm(name):
        n = re.sub(r'\s*\([^)]*\)', '', name)
        n = re.sub(r'[^a-z0-9]+', ' ', n.lower())   # reemplazar con espacio, no eliminar
        n = re.sub(r'\bgross domestic product\b', 'gdp', n)
        n = re.sub(r'\bconsumer price index\b', 'cpi', n)
        n = re.sub(r'\bpurchasing managers\b', 'pmi', n)
        n = re.sub(r'\b(japan|us|uk|germany|german|france|french|eurozone|euro area|australia|canada|swiss|switzerland|new zealand)\b', '', n)
        return re.sub(r'\s+', ' ', n).strip()

    STOPWORDS = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'by', 'or',
                 'mm', 'qoq', 'yoy', 'mom', 'q', 'y', 'm', 'sa', 'nsa',
                 'rate', 'change', 'data', 'flash', 'final', 'prelim',
                 'preliminary', 'revised', 'report', 'survey', 'release',
                 'monthly', 'quarterly', 'annual', 'weekly', 'current',
                 'total', 'net', 'new', 'all', 'index'}
    # Palabras que diferencian eventos aunque el resto del nombre se solape
    DIFFERENTIATORS = {'core', 'headline', 'underlying', 'trimmed', 'ex', 'excluding',
                       'external', 'domestic', 'capital', 'expenditure', 'capacity',
                       '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '30'}

    na, nb = _norm(name_a), _norm(name_b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    # Helper: palabras de contenido (sin stopwords)
    def content_words(s):
        return {w for w in s.split() if w not in STOPWORDS and len(w) > 2}

    def all_words(s):
        """Todas las palabras incluyendo números cortos (para diferenciadores)."""
        return {w for w in s.split() if w not in STOPWORDS}

    words_a = content_words(na)
    words_b = content_words(nb)
    # Diferenciadores: incluir números cortos (1,2,3...) usando all_words
    all_a = all_words(na)
    all_b = all_words(nb)
    diff_a = all_a & DIFFERENTIATORS
    diff_b = all_b & DIFFERENTIATORS
    if diff_a - diff_b or diff_b - diff_a:
        return False

    # Estrategia 1: alias corto contenido en el nombre largo
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    tokens_short = short.split()
    if len(tokens_short) <= 4 and short in long_:
        return True

    # Estrategia 2: solapamiento semántico
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    min_len = min(len(words_a), len(words_b))
    # Nombre muy corto (1-2 palabras útiles): basta 1 palabra si ≥80% match
    if min_len <= 2 and len(overlap) >= 1 and len(overlap) / max(min_len, 1) >= 0.8:
        return True
    # Caso general: ≥2 palabras y ≥60% del nombre más corto
    if len(overlap) >= 2 and len(overlap) / max(min_len, 1) >= 0.6:
        return True

    return False

from collections import defaultdict
slot_groups = defaultdict(list)
for ev in merged_values.values():
    slot = (ev['dateISO'], ev.get('timeUTC', ''), ev['currency'])
    slot_groups[slot].append(ev)

deduped = []
for slot, group in slot_groups.items():
    if len(group) == 1:
        deduped.append(group[0])
        continue

    dominated = set()
    for i in range(len(group)):
        for j in range(len(group)):
            if i == j or i in dominated:
                continue
            if are_true_duplicates(group[i]['event'], group[j]['event']):
                if score_event(group[i]) <= score_event(group[j]):
                    dominated.add(i)

    survivors = [ev for idx, ev in enumerate(group) if idx not in dominated]
    deduped.extend(survivors if survivors else group)

print(f"  [Dedup] {len(merged_values)} → {len(deduped)} after dedup")

# ── STEP 3b-post: Cross-day boundary dedup ────────────────────────────────────
# Events at 22:00-23:59 on day N and 00:00-02:00 on day N+1 are the same event
# stored with different UTC times from past vs future scrape passes.
# Group by (currency, normalized_name) across consecutive days in the boundary window.
boundary_groups = defaultdict(list)
for ev in deduped:
    h_str = ev.get('timeUTC', '')
    try:
        h = int(h_str.split(':')[0]) if h_str else -1
    except Exception:
        h = -1
    # Normalize: late-night events (>=22h) treated as "day N boundary",
    # early-morning (<=2h) treated as "day N-1 boundary" → use previous date as group key
    if h >= 22:
        anchor_date = ev['dateISO']
    elif h <= 2:
        try:
            anchor_date = (date.fromisoformat(ev['dateISO']) - timedelta(days=1)).isoformat()
        except Exception:
            anchor_date = ev['dateISO']
    else:
        anchor_date = None  # Not a boundary event
    if anchor_date:
        norm = normalize_name(ev['event'])
        boundary_groups[(ev['currency'], anchor_date, norm)].append(ev)

boundary_merged = 0
boundary_dominated = set()
for group_key, group in boundary_groups.items():
    if len(group) < 2:
        continue
    # Always keep the EARLIER-date event (the one with h>=22, i.e. still on day N)
    # The later-date event (h<=2, day N+1) has the correct impact from the future pass,
    # but wrong date. Merge data from all into the earliest-date event.
    group_sorted = sorted(group, key=lambda e: (e['dateISO'], e.get('timeUTC', '')))
    canonical = group_sorted[0]  # earliest date = correct date to show
    # Inherit best impact (high > medium > low) from any entry in group
    impact_rank = {'high': 3, 'medium': 2, 'low': 1}
    best_impact = max(group, key=lambda e: impact_rank.get(e.get('impact', 'low'), 1))
    canonical['impact'] = best_impact['impact']
    # Merge actual/forecast/previous from all entries
    for ev in group:
        for field in ('actual', 'forecast', 'previous'):
            if not canonical.get(field) and ev.get(field):
                canonical[field] = ev[field]
        if ev is not canonical:
            boundary_dominated.add(id(ev))
    boundary_merged += len(group) - 1

if boundary_merged:
    deduped = [ev for ev in deduped if id(ev) not in boundary_dominated]
    print(f"  [Boundary-dedup] Merged {boundary_merged} cross-day duplicate events")


# ── STEP 3c: Same-day semantic dedup (different timeUTC, same event name) ────────
# Catches cases like "ADP Weekly Employment Change" (08:00) vs "ADP Employment Change Weekly" (10:15)
# Uses aggressive normalization: removes stopwords (incl. "weekly"), expands GDP/CPI, sorts words.
def normalize_for_dedup(name):
    n = name.lower().strip()
    # Normalize yoy/mom/qoq variants BEFORE removing parens so they survive as differentiators
    n = re.sub(r'\(yoy\)', 'yoy', n); n = re.sub(r'\(mom\)', 'mom', n); n = re.sub(r'\(qoq\)', 'qoq', n)
    n = re.sub(r'\by/y\b', 'yoy', n); n = re.sub(r'\bm/m\b', 'mom', n); n = re.sub(r'\bq/q\b', 'qoq', n)
    n = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|\d{4})\b', '', n)
    n = re.sub(r'\([^)]*\)', '', n)
    n = re.sub(r'\bgross domestic product\b', 'gdp', n)
    n = re.sub(r'\bconsumer price index\b', 'cpi', n)
    n = re.sub(r'\bpurchasing managers\b', 'pmi', n)
    n = re.sub(r'\b(japan|japanese|us|uk|germany|german|france|french|eurozone|euro area|australia|australian|canada|canadian|swiss|switzerland|new zealand|u\.s\.)\b', '', n)
    sw = {'a','an','the','of','in','on','at','to','by','or','mm','rate','change',
          'data','flash','final','prelim','preliminary','revised','report','survey',
          'release','monthly','quarterly','annual','weekly','current','total','net',
          'new','all','index','q','y','m','sa','nsa'}
    words = [w for w in re.sub(r'[^a-z0-9 ]', ' ', n).split() if w not in sw and len(w) > 1]
    return ' '.join(sorted(words))

semantic_groups = defaultdict(list)
for ev in deduped:
    norm = normalize_for_dedup(ev['event'])
    semantic_groups[(ev['dateISO'], ev['currency'], norm)].append(ev)

semantic_dominated = set()
semantic_merged = 0
for group_key, group in semantic_groups.items():
    if len(group) < 2:
        continue
    best = max(group, key=score_event)
    impact_rank = {'high': 3, 'medium': 2, 'low': 1}
    best_impact = max(group, key=lambda e: impact_rank.get(e.get('impact', 'low'), 1))
    best['impact'] = best_impact['impact']
    for ev in group:
        for field in ('actual', 'forecast', 'previous'):
            if not best.get(field) and ev.get(field):
                best[field] = ev[field]
        if ev is not best:
            semantic_dominated.add(id(ev))
    semantic_merged += len(group) - 1

if semantic_merged:
    deduped = [ev for ev in deduped if id(ev) not in semantic_dominated]
    print(f"  [Semantic-dedup] Merged {semantic_merged} same-day duplicate events")


# ── Filter noise: remove events with no time AND no actual/forecast/previous data ──
# These are "X Speaks" / ceremonial events with no economic signal value
def has_economic_data(ev):
    return (ev.get('timeUTC', '').strip() or
            ev.get('actual', '').strip() or
            ev.get('forecast', '').strip() or
            ev.get('previous', '').strip())

before_filter = len(deduped)
unique_events = [ev for ev in deduped if has_economic_data(ev)]
removed_noise = before_filter - len(unique_events)
if removed_noise:
    print(f"  [Filter] Removed {removed_noise} no-time/no-data noise events")


# Sort by dateISO + timeUTC
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    return ev['dateISO'] + 'T' + (t if re.match(r'\d{2}:\d{2}', t) else '00:00')

unique_events.sort(key=sort_key)

# Keep ALL events (full history + future) — no date cutoff
# Display window is controlled by the frontend, not the scraper
final_events = unique_events

print(f"\n  [Result] fresh={len(all_events)} → merged={len(merged_values)} → deduped={len(deduped)} → final={len(final_events)}")
all_dates_final = sorted(set(e['dateISO'] for e in final_events))
print(f"  [Dates] {all_dates_final[:4]}...{all_dates_final[-4:] if len(all_dates_final) > 4 else ''}")

# ── STEP 4: Stats and save ─────────────────────────────────────────────────
currency_counts = {}
impact_counts   = {'high': 0, 'medium': 0, 'low': 0}
for ev in final_events:
    c = ev['currency']
    currency_counts[c] = currency_counts.get(c, 0) + 1
    impact_counts[ev.get('impact','low')] = impact_counts.get(ev.get('impact','low'), 0) + 1

data_ok = len(final_events) >= 5

output = {
    'lastUpdate':     today.isoformat(),
    'generatedAt':    datetime.utcnow().isoformat() + 'Z',
    'timezoneNote':   'All timeUTC fields are UTC. Frontend converts to user local timezone.',
    'status':         'ok'    if data_ok else 'error',
    'source':         source_used if data_ok else None,
    'errorMessage':   None    if data_ok else (
        'No se pudieron obtener datos de ninguna fuente. '
        'Por favor consulte directamente: '
        'investing.com/economic-calendar o tradingeconomics.com/calendar'
    ),
    'fetchErrors':    fetch_errors if not data_ok else [],
    'rangeFrom':      (all_dates_final[0] if all_dates_final else from_date.isoformat()),
    'rangeTo':        to_date.isoformat(),
    'totalEvents':    len(final_events),
    'currencyCounts': currency_counts,
    'impactCounts':   impact_counts,
    'events':         final_events,
}

with open(CALENDAR_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
if data_ok:
    print(f"✅ SAVED: {CALENDAR_PATH}")
    print(f"   Source:  {source_used}")
    print(f"   Total:   {len(final_events)} events")
    print(f"   Impact:  {impact_counts}")
    print(f"   By currency: {dict(sorted(currency_counts.items()))}")
    high = [e for e in final_events if e.get('impact') == 'high'][:8]
    if high:
        print(f"\n   Next high-impact events (UTC):")
        for ev in high:
            fc = f" | est:{ev['forecast']}" if ev.get('forecast') else ''
            ac = f" → {ev['actual']}"        if ev.get('actual')   else ''
            print(f"   {ev['dateISO']} {(ev['timeUTC'] or '?'):5s} UTC [{ev['currency']}] {ev['event']}{fc}{ac}")
else:
    print(f"⛔ SAVED EMPTY CALENDAR — all sources failed")
    for e in fetch_errors:
        print(f"   • {e}")
print("=" * 60)

if not data_ok:
    sys.exit(1)
