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
    'gdp', 'gross domestic product', 'cpi', 'consumer price', 'inflation',
    'nonfarm payrolls', 'nonfarm', 'employment change', 'claimant count',
    'unemployment rate', 'jobless rate', 'unemployment claims',
    'fomc', 'ecb', 'boe', 'boj', 'rba', 'boc', 'snb', 'rbnz',
    'monetary policy', 'retail sales', 'pmi', 'purchasing managers',
    'trade balance', 'current account',
]
MED_KW = [
    'manufacturing', 'industrial production', 'factory orders',
    'housing', 'building permits', 'consumer confidence', 'business confidence',
    'ppi', 'producer price', 'wage', 'earnings', 'ism', 'durable goods',
    'job openings', 'jolts', 'import price', 'export price', 'services pmi',
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
    Handles both MQL5 HTML format and plain text format.
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
# SOURCE 0: ForexFactory (nfs.faireconomy.media)
# - JSON endpoint: has actual, forecast, previous, full week
# - XML endpoint: fallback, has forecast + previous only
# - Times are in GMT (UTC)
# - Impact: 'Holiday', 'Low', 'Medium', 'High'
# ════════════════════════════════════════════════════════════════════

def _parse_ff_time(time_str):
    """Convert FF time string '2:00am', '10:30pm' → 'HH:MM' UTC string."""
    if not time_str or time_str.lower() in ('all day', 'tentative', ''):
        return ''
    try:
        ts = time_str.strip().upper().replace('AM', ' AM').replace('PM', ' PM')
        for tfmt in ['%I:%M %p', '%I %p']:
            try:
                t = datetime.strptime(ts.strip(), tfmt)
                return t.strftime('%H:%M')
            except Exception:
                pass
    except Exception:
        pass
    return ''

def _parse_ff_date(date_str):
    """Parse FF date string into a date object.
    Handles: MM-DD-YYYY, YYYY-MM-DD, ISO with T, and FF JSON full format
    e.g. 'Sun Mar 08 2026 00:00:00 GMT+0000' or '2026-03-08T00:00:00+00:00'
    """
    if not date_str:
        return None
    # Try simple formats first
    for fmt in ['%m-%d-%Y', '%Y-%m-%d', '%d-%m-%Y']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except Exception:
            pass
    # ISO with time component: "2026-03-08T00:00:00+00:00" or "2026-03-08T00:00:00Z"
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
    except Exception:
        pass
    # FF JSON full format: "Sun Mar 08 2026 00:00:00 GMT+0000"
    try:
        # Strip timezone part "GMT+0000" or "GMT-0500"
        clean = re.sub(r'\s+GMT[+-]\d{4}$', '', date_str.strip())
        return datetime.strptime(clean, '%a %b %d %Y %H:%M:%S').date()
    except Exception:
        pass
    # Try just extracting YYYY-MM-DD from the string
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except Exception:
            pass
    return None

def fetch_forexfactory_xml(target_dates):
    """
    Fetches ForexFactory weekly calendar.
    Strategy: tries JSON first (has actual values), then XML as fallback.
    JSON URL: https://nfs.faireconomy.media/ff_calendar_thisweek.json
    XML URL:  https://nfs.faireconomy.media/ff_calendar_thisweek.xml
    Times are GMT (UTC). Impact mapped: Low/Medium/High.
    """
    IMPACT_MAP = {'Holiday': None, 'Low': 'low', 'Medium': 'medium', 'High': 'high'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; economic-calendar-bot/1.0)',
        'Accept': 'application/json, application/xml, text/xml, */*',
    }

    # ── Try JSON first (has actual values) ──────────────────────────
    json_urls = [
        'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
        'https://www.forexfactory.com/ff_calendar_thisweek.json',
    ]
    for url in json_urls:
        print(f"  [FF] Fetching JSON: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                print(f"  [FF] HTTP {r.status_code}")
                continue
            print(f"  [FF] Got {len(r.content)} bytes")
            items = r.json()
            if not isinstance(items, list) or not items:
                print(f"  [FF] Empty or invalid JSON")
                continue

            print(f"  [FF] {len(items)} events in JSON")
            if items:
                sample = items[0]
                print(f"  [FF] Sample keys: {list(sample.keys())[:8]}")
                print(f"  [FF] Sample date={sample.get('date','?')!r} country={sample.get('country','?')!r} impact={sample.get('impact','?')!r}")
            events = []
            skip_date = skip_cur = skip_imp = 0
            for item in items:
                try:
                    # FF JSON uses 'country' with country name (e.g. "Japan"), not 'currency'
                    country_raw = str(item.get('country', item.get('currency', ''))).strip()
                    currency = resolve_currency(country_raw)
                    if not currency:
                        # Try direct currency code match
                        c = country_raw.upper()
                        currency = c if c in TRACKED_CURRENCIES else None
                    if not currency:
                        skip_cur += 1
                        continue

                    impact_raw = str(item.get('impact', 'Low')).capitalize()
                    impact = IMPACT_MAP.get(impact_raw)
                    if impact is None:
                        skip_imp += 1
                        continue

                    event_name = str(item.get('title', item.get('name', ''))).strip()
                    if not event_name:
                        continue

                    date_str = str(item.get('date', ''))
                    event_date = _parse_ff_date(date_str)
                    if not event_date:
                        skip_date += 1
                        continue
                    # FF covers the current week — expand the window by ±7 days to be safe
                    extended = target_dates | {today - timedelta(days=i) for i in range(1, 8)}
                    if event_date not in extended:
                        skip_date += 1
                        continue

                    time_utc = _parse_ff_time(str(item.get('time', '')))

                    actual   = clean_val(str(item.get('actual',   '') or ''))
                    forecast = clean_val(str(item.get('forecast', '') or ''))
                    previous = clean_val(str(item.get('previous', '') or ''))

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
                except Exception as _ie:
                    print(f"  [FF] Item error: {_ie}")
                    continue

            print(f"  [FF] Skipped: date={skip_date} cur={skip_cur} impact={skip_imp}")
            if events:
                print(f"  [FF] ✅ Parsed {len(events)} events from JSON (with actuals)")
                return events
            print(f"  [FF] 0 events parsed from JSON")

        except Exception as e:
            print(f"  [FF] JSON error: {e}")
            continue

    # ── Fallback: XML (no actual values) ────────────────────────────
    xml_urls = [
        'https://nfs.faireconomy.media/ff_calendar_thisweek.xml',
        'https://www.forexfactory.com/ff_calendar_thisweek.xml',
    ]
    for url in xml_urls:
        print(f"  [FF] Fetching XML: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                print(f"  [FF] HTTP {r.status_code}")
                continue
            print(f"  [FF] Got {len(r.content)} bytes")

            items = []
            for parser in ['lxml-xml', 'lxml', 'html.parser']:
                soup = BeautifulSoup(r.text, parser)
                items = soup.find_all('event')
                if items:
                    print(f"  [FF] {len(items)} <event> tags (parser: {parser})")
                    break

            if not items:
                print(f"  [FF] No <event> tags found")
                continue

            events = []
            for item in items:
                try:
                    cur_tag = item.find('currency') or item.find('country')
                    if not cur_tag:
                        continue
                    currency = cur_tag.get_text(strip=True).upper()
                    if currency not in TRACKED_CURRENCIES:
                        continue

                    imp_tag = item.find('impact')
                    impact_raw = imp_tag.get_text(strip=True) if imp_tag else 'Low'
                    impact = IMPACT_MAP.get(impact_raw)
                    if impact is None:
                        continue

                    title_tag = item.find('title') or item.find('name')
                    event_name = title_tag.get_text(strip=True) if title_tag else ''
                    if not event_name:
                        continue

                    date_tag = item.find('date')
                    time_tag = item.find('time')
                    event_date = _parse_ff_date(date_tag.get_text(strip=True) if date_tag else '')
                    if not event_date or event_date not in target_dates:
                        continue

                    time_utc = _parse_ff_time(time_tag.get_text(strip=True) if time_tag else '')

                    fc_tag   = item.find('forecast')
                    prev_tag = item.find('previous')
                    forecast = clean_val(fc_tag.get_text(strip=True))  if fc_tag  else ''
                    previous = clean_val(prev_tag.get_text(strip=True)) if prev_tag else ''

                    events.append({
                        'date':     fmt_date(event_date),
                        'dateISO':  event_date.isoformat(),
                        'timeUTC':  time_utc,
                        'country':  currency,
                        'currency': currency,
                        'flag':     CURRENCY_FLAGS.get(currency, ''),
                        'event':    event_name,
                        'impact':   impact,
                        'actual':   '',
                        'forecast': forecast,
                        'previous': previous,
                    })
                except Exception:
                    continue

            if events:
                print(f"  [FF] ✅ Parsed {len(events)} events from XML (no actuals)")
                return events
            print(f"  [FF] 0 events parsed from {len(items)} tags")

        except Exception as e:
            print(f"  [FF] XML error: {e}")
            continue

    return []

# ════════════════════════════════════════════════════════════════════
# SOURCE 1: MQL5 Economic Calendar RSS
# Times from pubDate are already in UTC (RFC 2822 with timezone info)
# ════════════════════════════════════════════════════════════════════

def fetch_mql5_rss(target_dates):
    events = []
    from_str = min(target_dates).isoformat()
    to_str   = max(target_dates).isoformat()

    urls = [
        f"https://calendar.mql5.com/en/economic_calendar/rss?from={from_str}&to={to_str}",
        f"https://calendar.mql5.com/en/economic_calendar/rss?currencies=USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD",
        "https://calendar.mql5.com/en/economic_calendar/rss",
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; RSS reader/2.0)',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    for url in urls:
        print(f"  [MQL5] Fetching: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 429:
                print("  [MQL5] Rate limited, waiting 20s...")
                time.sleep(20)
                r = requests.get(url, headers=headers, timeout=25)
            if not r.ok:
                print(f"  [MQL5] HTTP {r.status_code}")
                continue

            print(f"  [MQL5] Got {len(r.content)} bytes")
            for parser in ['lxml-xml', 'lxml', 'html.parser']:
                soup = BeautifulSoup(r.text, parser)
                items = soup.find_all('item')
                if items:
                    print(f"  [MQL5] {len(items)} items (parser: {parser})")
                    break

            if not items:
                continue

            batch = []
            for item in items:
                try:
                    title_tag = item.find('title')
                    if not title_tag: continue
                    title_text = title_tag.get_text(strip=True)

                    currency = ''
                    event_name = title_text
                    if ':' in title_text:
                        prefix, rest = title_text.split(':', 1)
                        prefix = prefix.strip().upper()
                        rest = rest.strip()
                        if prefix in TRACKED_CURRENCIES:
                            currency = prefix
                            event_name = rest
                        else:
                            c = resolve_currency(prefix)
                            if c: currency, event_name = c, rest

                    if not currency:
                        cat = item.find('category')
                        if cat: currency = clean_val(cat.get_text(strip=True)).upper()

                    if currency not in TRACKED_CURRENCIES: continue

                    # pubDate includes timezone info → parsedate_to_datetime
                    # normalizes to UTC automatically
                    pub = item.find('pubDate')
                    event_date = None
                    time_str = ''
                    if pub:
                        try:
                            dt = parsedate_to_datetime(pub.get_text(strip=True))
                            # Convert to UTC if it has timezone info
                            if dt.tzinfo is not None:
                                import datetime as dt_module
                                dt = dt.astimezone(dt_module.timezone.utc)
                            event_date = dt.date()
                            time_str = dt.strftime('%H:%M')
                        except: pass

                    if not event_date or event_date not in target_dates: continue

                    desc_tag = item.find('description')
                    desc_raw = ''
                    if desc_tag:
                        desc_raw = str(desc_tag.string) if desc_tag.string else desc_tag.get_text()

                    actual, forecast, previous, desc_impact = parse_desc_html(desc_raw)

                    impact = ''
                    imp_tag = item.find(re.compile(r'importance', re.I))
                    if imp_tag:
                        try:
                            iv = int(imp_tag.get_text(strip=True))
                            impact = 'high' if iv >= 3 else 'medium' if iv == 2 else 'low'
                        except: pass
                    if not impact:
                        impact = desc_impact if desc_impact else classify_impact_kw(event_name)

                    batch.append({
                        'date': fmt_date(event_date),
                        'dateISO': event_date.isoformat(),
                        'timeUTC': time_str,
                        'country': currency, 'currency': currency,
                        'flag': CURRENCY_FLAGS.get(currency, ''),
                        'event': event_name,
                        'impact': impact,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                    })
                except: continue

            if batch:
                print(f"  [MQL5] ✅ Parsed {len(batch)} events")
                return batch

        except Exception as e:
            print(f"  [MQL5] Error: {e}")
            continue

    return events

# ════════════════════════════════════════════════════════════════════
# SOURCE 2: Trading Economics RSS
# Times from pubDate include timezone → normalized to UTC
# ════════════════════════════════════════════════════════════════════

def fetch_te_rss(target_dates):
    events = []
    urls = [
        "https://tradingeconomics.com/calendar/rss",
        "https://tradingeconomics.com/rss/calendar.aspx",
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; RSS/2.0)',
        'Accept': 'application/rss+xml, application/xml, text/xml',
    }
    for url in urls:
        print(f"  [TE RSS] Fetching: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                print(f"  [TE RSS] HTTP {r.status_code}")
                continue
            print(f"  [TE RSS] Got {len(r.content)} bytes")

            for parser in ['lxml-xml', 'lxml']:
                soup = BeautifulSoup(r.text, parser)
                items = soup.find_all('item')
                if items: break

            print(f"  [TE RSS] {len(items)} items")
            batch = []
            for item in items:
                try:
                    title_tag = item.find('title')
                    if not title_tag: continue
                    title = title_tag.get_text(strip=True)

                    currency = ''
                    event_name = title
                    for sep in [' - ', ': ', ' | ']:
                        if sep in title:
                            parts = title.split(sep, 1)
                            c = resolve_currency(parts[0].strip())
                            if c:
                                currency, event_name = c, parts[1].strip()
                                break
                    if not currency:
                        cat = item.find('category')
                        if cat: currency = resolve_currency(cat.get_text(strip=True)) or ''
                    if currency not in TRACKED_CURRENCIES: continue

                    pub = item.find('pubDate')
                    event_date = None
                    time_str = ''
                    if pub:
                        try:
                            dt = parsedate_to_datetime(pub.get_text(strip=True))
                            if dt.tzinfo is not None:
                                import datetime as dt_module
                                dt = dt.astimezone(dt_module.timezone.utc)
                            event_date = dt.date()
                            time_str = dt.strftime('%H:%M')
                        except: pass
                    if not event_date or event_date not in target_dates: continue

                    desc_tag = item.find('description')
                    desc_raw = str(desc_tag.string) if (desc_tag and desc_tag.string) else (desc_tag.get_text() if desc_tag else '')
                    actual, forecast, previous, imp = parse_desc_html(desc_raw)
                    impact = imp if imp else classify_impact_kw(event_name)

                    batch.append({
                        'date': fmt_date(event_date),
                        'dateISO': event_date.isoformat(),
                        'timeUTC': time_str,
                        'country': currency, 'currency': currency,
                        'flag': CURRENCY_FLAGS.get(currency, ''),
                        'event': event_name,
                        'impact': impact,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                    })
                except: continue

            if batch:
                print(f"  [TE RSS] ✅ Parsed {len(batch)} events")
                return batch
        except Exception as e:
            print(f"  [TE RSS] Error: {e}")
    return events

# ════════════════════════════════════════════════════════════════════
# SOURCE 3: Investing.com
# IMPORTANT: Investing.com returns times in EST (UTC-5) regardless of
# the timeZone parameter sent in the POST request.
# We convert EST → UTC by adding 5 hours before storing.
# Verification: ECB Lagarde stored as 03:30 EST → 08:30 UTC
#               Frontend (UTC-3) shows 05:30 ✅
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

        events = []
        for row in rows:
            try:
                dt_str = row.get('data-event-datetime', '')
                if not dt_str:
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
                    continue

                m_t = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}[ T](\d{2}):(\d{2})', dt_norm)
                raw_time_est = f"{m_t.group(1)}:{m_t.group(2)}" if m_t else ''
                time_str_utc, event_date = est_to_utc(raw_time_est, event_date)

                extended_dates = target_dates | {max(target_dates) + timedelta(days=1)}
                if event_date not in extended_dates:
                    continue

                currency = row.get('data-currency', '').strip().upper()
                if currency not in TRACKED_CURRENCIES:
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

                ev_td = row.find('td', class_=re.compile('event'))
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

        print(f"  [Investing HTML] ✅ Parsed {len(events)} events")
        return events

    # ── Run Strategy A, fall back to B ────────────────────────────────
    try:
        events = fetch_via_json_api()
        if events:
            return events
        print("  [Investing] JSON API returned 0 events, trying HTML fallback")
    except Exception as e:
        import traceback
        print(f"  [Investing] JSON API error: {e}")
        print(f"  [Investing] {traceback.format_exc()[:400]}")

    try:
        return fetch_via_html_post()
    except Exception as e:
        print(f"  [Investing] HTML fallback error: {e}")
        return []

# ════════════════════════════════════════════════════════════════════
# SOURCE 4: Official Government / Central Bank RSS
# pubDate includes timezone → normalized to UTC
# ════════════════════════════════════════════════════════════════════

def fetch_official_rss(target_dates):
    feeds = [
        ('USD', 'https://www.bls.gov/feed/bls_latest.rss'),
        ('USD', 'https://www.federalreserve.gov/feeds/press_all.xml'),
        ('EUR', 'https://www.ecb.europa.eu/rss/press.html'),
        ('GBP', 'https://www.bankofengland.co.uk/rss/publications'),
        ('CAD', 'https://www150.statcan.gc.ca/n1/rss/new-nouveau.xml'),
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; RSS/2.0)',
               'Accept': 'application/rss+xml, application/xml, text/xml'}
    events = []
    for currency, url in feeds:
        try:
            print(f"  [Official] {currency}: {url}")
            r = requests.get(url, headers=headers, timeout=15)
            if not r.ok: continue
            print(f"  [Official] {currency}: {len(r.content)} bytes")
            soup = BeautifulSoup(r.text, 'lxml-xml')
            items = soup.find_all('item')
            print(f"  [Official] {currency}: {len(items)} items")
            for item in items:
                title_tag = item.find('title')
                if not title_tag: continue
                event_name = title_tag.get_text(strip=True)
                pub = item.find('pubDate')
                event_date = time_str = None
                if pub:
                    try:
                        dt = parsedate_to_datetime(pub.get_text(strip=True))
                        if dt.tzinfo is not None:
                            import datetime as dt_module
                            dt = dt.astimezone(dt_module.timezone.utc)
                        event_date = dt.date()
                        time_str = dt.strftime('%H:%M')
                    except: pass
                if not event_date or event_date not in target_dates: continue
                events.append({
                    'date': fmt_date(event_date),
                    'dateISO': event_date.isoformat(),
                    'timeUTC': time_str or '',
                    'country': currency, 'currency': currency,
                    'flag': CURRENCY_FLAGS.get(currency, ''),
                    'event': event_name,
                    'impact': classify_impact_kw(event_name),
                    'actual': '', 'forecast': '', 'previous': '',
                })
        except Exception as e:
            print(f"  [Official] {currency} error: {e}")
    return events

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ECONOMIC CALENDAR SCRAPER v5.2 (accumulative merge)")
print("Sources: ForexFactory XML → MQL5 RSS → TE RSS → Investing.com → Official RSS")
print("Timezone policy:")
print("  - MQL5/TE/Official RSS: pubDate normalized to UTC via parsedate_to_datetime")
print("  - Investing.com: returns EST (UTC-5), converted +5h to UTC before storing")
print("  - All timeUTC fields are UTC — frontend converts to user local timezone")
print("=" * 60)

import sys

today     = date.today()
from_date = today - timedelta(days=2)
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

# ── STEP 2: Fetch fresh data from live sources ───────────────────────────────
all_events  = []
source_used = None
fetch_errors = []

for label, fetcher, args in [
    ("0: ForexFactory XML", fetch_forexfactory_xml,   (target_dates,)),
    ("1: MQL5 RSS",         fetch_mql5_rss,           (target_dates,)),
    ("2: TE RSS",           fetch_te_rss,             (target_dates,)),
    ("3: Investing.com",    fetch_investing_calendar, (from_date.isoformat(), to_date.isoformat(), target_dates)),
    ("4: Official RSS",     fetch_official_rss,       (target_dates,)),
]:
    if len(all_events) >= 3:
        break
    print(f"\n{'='*50}\nSTRATEGY {label}\n{'='*50}")
    try:
        result = fetcher(*args)
        if len(result) >= 3:
            all_events  = result
            source_used = label.split(': ', 1)[1]
            print(f"✅ {label}: {len(all_events)} events — using this source")
        else:
            msg = f"{label}: only {len(result)} events returned"
            print(f"⚠️  {msg}")
            fetch_errors.append(msg)
            if result:
                for ev in result:
                    if ev not in all_events:
                        all_events.append(ev)
                if not source_used:
                    source_used = label.split(': ', 1)[1] + ' (partial)'
    except Exception as e:
        msg = f"{label} failed: {e}"
        print(f"❌ {msg}")
        fetch_errors.append(msg)
    time.sleep(2)

if not all_events and not base_events:
    print(f"\n{'='*50}")
    print("⛔ ALL SOURCES FAILED AND NO BASE DATA — saving empty calendar")
    print("=" * 50)

# ── STEP 3: Build final event list ────────────────────────────────────────────
# Strategy: if we got fresh events from a live source, use ONLY those (avoids
# mixing base duplicates). Only fall back to base if fresh fetch failed entirely.
if all_events:
    # Start from full historical base, then overwrite with fresh data
    # This preserves all past events AND updates forecasts/actuals with latest data
    merged_values = dict(base_events)  # copy full history
    fresh_added = fresh_updated = 0
    for ev in all_events:
        k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())
        if k in merged_values:
            existing = merged_values[k]
            # Update: fresh data wins on forecast/previous; actual only updates if fresh has one
            updated = dict(existing)
            if ev.get('forecast'): updated['forecast'] = ev['forecast']
            if ev.get('previous'): updated['previous'] = ev['previous']
            if ev.get('actual'):   updated['actual']   = ev['actual']
            # Always update impact and event name from fresh source
            updated['impact'] = ev['impact']
            merged_values[k] = updated
            fresh_updated += 1
        else:
            merged_values[k] = ev
            fresh_added += 1
    print(f"  [Merge] base={len(base_events)} + fresh={len(all_events)} → {len(merged_values)} total "
          f"(added={fresh_added}, updated={fresh_updated})")
else:
    # Fresh failed — use base as fallback
    merged_values = base_events
    print(f"  [Merge] Fresh failed — using base only: {len(base_events)} events")

# ── STEP 3a: Enrich actuals from Investing.com ───────────────────────────────
# FF XML has no actuals. After merging, fetch today's events from Investing.com
# (which DOES return actuals for events that already published today/yesterday)
# and patch them into our merged set.
events_needing_actual = [ev for ev in merged_values.values()
                         if not ev.get('actual') and
                         date.fromisoformat(ev['dateISO']) >= today - timedelta(days=2) and
                         date.fromisoformat(ev['dateISO']) <= today]
if events_needing_actual:
    print(f"\n  [Actuals] Fetching Investing.com to fill actuals for "
          f"{len(events_needing_actual)} events (today/yesterday without actual)...")
    try:
        investing_recent = fetch_investing_calendar(
            (today - timedelta(days=2)).isoformat(),
            today.isoformat(),
            {today - timedelta(days=i) for i in range(3)}
        )
        enriched = 0

        # Build a lookup: (dateISO, timeUTC, currency) → [inv_ev, ...]
        # Investing.com times are already UTC. Match slot exactly.
        from collections import defaultdict as _dd
        inv_by_slot = _dd(list)
        for inv_ev in investing_recent:
            if inv_ev.get('actual'):
                slot = (inv_ev['dateISO'], inv_ev.get('timeUTC',''), inv_ev['currency'])
                inv_by_slot[slot].append(inv_ev)

        for k, ev in list(merged_values.items()):
            if ev.get('actual'):
                continue
            slot = (ev['dateISO'], ev.get('timeUTC',''), ev['currency'])
            candidates = inv_by_slot.get(slot, [])

            if len(candidates) == 1:
                # Exact unique slot match — assign actual directly
                inv_ev = candidates[0]
                updated = dict(ev)
                updated['actual'] = inv_ev['actual']
                if inv_ev.get('forecast') and not ev.get('forecast'):
                    updated['forecast'] = inv_ev['forecast']
                if inv_ev.get('previous') and not ev.get('previous'):
                    updated['previous'] = inv_ev['previous']
                merged_values[k] = updated
                enriched += 1
            elif len(candidates) > 1:
                # Multiple events in same slot — use name similarity as tiebreaker
                def _norm(s):
                    s = re.sub(r'\s*\([^)]*\)', '', s).lower()
                    return set(re.sub(r'[^a-z0-9 ]', ' ', s).split())
                words_ev = _norm(ev['event'])
                best_match = None
                best_score = 0.0
                for cand in candidates:
                    words_c = _norm(cand['event'])
                    if not words_ev or not words_c:
                        continue
                    score = len(words_ev & words_c) / max(len(words_ev), len(words_c))
                    if score > best_score:
                        best_score = score
                        best_match = cand
                if best_match and best_score >= 0.3:
                    updated = dict(ev)
                    updated['actual'] = best_match['actual']
                    if best_match.get('forecast') and not ev.get('forecast'):
                        updated['forecast'] = best_match['forecast']
                    if best_match.get('previous') and not ev.get('previous'):
                        updated['previous'] = best_match['previous']
                    merged_values[k] = updated
                    enriched += 1

        print(f"  [Actuals] Enriched {enriched} events with actual values from Investing.com")
        print(f"  [Actuals] Investing slots available: {len(inv_by_slot)}, FF events needing actual: {len(events_needing_actual)}")
    except Exception as _ae:
        import traceback
        print(f"  [Actuals] Investing.com enrichment failed: {_ae}")
        print(f"  [Actuals] {traceback.format_exc()[:300]}")

# ── STEP 3b: Conservative dedup — only remove true aliases (same event, different source name) ──
# "Japan Leading Index MoM" == "Leading Index (MoM) (Jan)" → deduplicate (short alias ≤4 tokens contained in longer)
# "GDP QoQ" vs "GDP Capital Expenditure QoQ" → keep both (short has ≤4 tokens but NOT contained since extra words differ)
# "3-Month Bill Auction" vs "6-Month Bill Auction" → keep both (different events in same slot)
def normalize_name(name):
    n = re.sub(r'\s*\([^)]*\)', '', name)
    n = re.sub(r'[^a-z0-9 ]', '', n.lower())
    n = re.sub(r'\b(japan|us|uk|germany|german|france|french|eurozone|euro area|australia|canada|swiss|switzerland|new zealand)\b', '', n)
    return re.sub(r'\s+', ' ', n).strip()

def score_event(ev):
    s = 0
    if ev.get('actual'):   s += 1000
    if ev.get('forecast'): s += 100
    if ev.get('previous'): s += 10
    s += len(ev.get('event', ''))
    return s

def are_true_duplicates(name_a, name_b):
    """Only deduplicate if one name is clearly a short alias contained in the other (≤4 tokens)."""
    na, nb = normalize_name(name_a), normalize_name(name_b)
    if not na or not nb or na == nb:
        return na == nb
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    tokens = short.split()
    if len(tokens) > 4:
        return False  # too specific — likely a distinct event, not just an alias
    return short in long_

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

unique_events = deduped

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
        'Por favor consulte directamente: forexfactory.com, '
        'investing.com/economic-calendar, o tradingeconomics.com/calendar'
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
