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
# SOURCE 0: Forex Factory XML (primary — official impact/forecast/previous)
# https://nfs.faireconomy.media/ff_calendar_thisweek.xml
# Times in the XML are in US/Eastern (ET). We convert to UTC.
# Impact values: High → high, Medium → medium, Low → low, Holiday → skip
# ════════════════════════════════════════════════════════════════════

def fetch_ff_xml(target_dates):
    """
    Fetch the Forex Factory weekly XML calendar.
    Returns a list of event dicts in the same format as all other sources.
    Only includes the 8 tracked currencies; skips Holiday entries.
    """
    FF_URL      = 'https://nfs.faireconomy.media/ff_calendar_thisweek.xml'
    FF_NEXT_URL = 'https://nfs.faireconomy.media/ff_calendar_nextweek.xml'

    # FF times are US Eastern. We need to convert to UTC.
    # ET = UTC-5 (EST) or UTC-4 (EDT). We use a simple approach:
    # parse the 12h time string, combine with the date, then apply ET offset.
    import datetime as dt_module

    def ff_time_to_utc(time_str, date_obj):
        """
        Convert FF time string like '8:30pm' + date to UTC HH:MM and possibly
        advance the date by 1 day (when ET→UTC crosses midnight).
        Returns (utc_time_str, utc_date).
        """
        if not time_str or time_str.strip().lower() in ('all day', 'tentative', ''):
            return '', date_obj
        try:
            # Parse 12h format
            t = time_str.strip().lower().replace(' ', '')
            fmt = '%I:%M%p' if ':' in t else '%I%p'
            naive = datetime.strptime(t, fmt)
            # Combine with date
            naive_dt = dt_module.datetime(
                date_obj.year, date_obj.month, date_obj.day,
                naive.hour, naive.minute
            )
            # Determine ET offset: EDT (UTC-4) Mar 2nd Sun → Nov 1st Sun, else EST (UTC-5)
            # Simple rule: DST in effect between 2nd Sun of March and 1st Sun of November
            def is_edt(d):
                import calendar
                # 2nd Sunday of March
                mar = d.replace(month=3, day=1)
                first_sun_mar = mar + dt_module.timedelta(days=(6 - mar.weekday()) % 7)
                dst_start = first_sun_mar + dt_module.timedelta(weeks=1)  # 2nd Sunday
                # 1st Sunday of November
                nov = d.replace(month=11, day=1)
                first_sun_nov = nov + dt_module.timedelta(days=(6 - nov.weekday()) % 7)
                return dst_start <= d < first_sun_nov

            offset_hours = 4 if is_edt(date_obj) else 5
            utc_dt = naive_dt + dt_module.timedelta(hours=offset_hours)
            return utc_dt.strftime('%H:%M'), utc_dt.date()
        except Exception as e:
            return '', date_obj

    IMPACT_MAP = {
        'high':    'high',
        'medium':  'medium',
        'low':     'low',
        'holiday': None,   # skip holidays
        'non-economic': None,
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; EconCalendar/1.0)',
        'Accept': 'application/xml, text/xml, */*',
    }

    events = []
    # Fetch this week; also try next week if target_dates extend beyond
    urls_to_try = [FF_URL]
    max_date = max(target_dates)
    # If target range extends more than 7 days from today, also fetch next week
    if max_date > date.today() + timedelta(days=7):
        urls_to_try.append(FF_NEXT_URL)

    for url in urls_to_try:
        print(f"  [FF XML] Fetching: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                if r.status_code == 404 and 'nextweek' in url:
                    print(f"  [FF XML] Next week XML not yet available (HTTP 404) — skipping")
                else:
                    print(f"  [FF XML] HTTP {r.status_code}")
                continue
            print(f"  [FF XML] {len(r.content)} bytes")

            # Parse XML — FF uses windows-1252 encoding declared in header
            # BeautifulSoup handles this gracefully with lxml-xml
            content = r.content.decode('windows-1252', errors='replace')
            soup = BeautifulSoup(content, 'lxml-xml')
            items = soup.find_all('event')
            print(f"  [FF XML] {len(items)} events in XML")

            for item in items:
                try:
                    # Currency / country
                    currency_tag = item.find('country')
                    currency = currency_tag.get_text(strip=True).upper() if currency_tag else ''
                    if currency not in TRACKED_CURRENCIES:
                        continue

                    # Impact
                    impact_tag = item.find('impact')
                    impact_raw = impact_tag.get_text(strip=True).lower() if impact_tag else 'low'
                    impact = IMPACT_MAP.get(impact_raw)
                    if impact is None:
                        continue  # skip Holiday / Non-Economic

                    # Date  — format: MM-DD-YYYY
                    date_tag = item.find('date')
                    date_str = date_tag.get_text(strip=True) if date_tag else ''
                    try:
                        event_date = datetime.strptime(date_str, '%m-%d-%Y').date()
                    except:
                        continue

                    # Time — format: 12h ET  e.g. "8:30am"
                    time_tag = item.find('time')
                    time_raw = time_tag.get_text(strip=True) if time_tag else ''
                    time_utc, event_date_utc = ff_time_to_utc(time_raw, event_date)

                    if event_date_utc not in target_dates and event_date not in target_dates:
                        continue

                    # Title
                    title_tag = item.find('title')
                    event_name = title_tag.get_text(strip=True) if title_tag else ''
                    if not event_name:
                        continue

                    # Forecast / Previous (actual not in FF XML — added later by other sources)
                    fc_tag   = item.find('forecast')
                    prev_tag = item.find('previous')
                    forecast = clean_val(fc_tag.get_text(strip=True))   if fc_tag   else ''
                    previous = clean_val(prev_tag.get_text(strip=True)) if prev_tag else ''

                    use_date = event_date_utc if event_date_utc in target_dates else event_date
                    events.append({
                        'date':     fmt_date(use_date),
                        'dateISO':  use_date.isoformat(),
                        'timeUTC':  time_utc,
                        'country':  currency,
                        'currency': currency,
                        'flag':     CURRENCY_FLAGS.get(currency, ''),
                        'event':    event_name,
                        'impact':   impact,
                        'actual':   '',        # FF XML doesn't include actuals
                        'forecast': forecast,
                        'previous': previous,
                        'ff_url':   item.find('url').get_text(strip=True) if item.find('url') else '',
                    })
                except Exception as e:
                    continue

            print(f"  [FF XML] ✅ Parsed {len(events)} tracked-currency events")
        except Exception as e:
            import traceback
            print(f"  [FF XML] Error: {e}")
            print(traceback.format_exc()[:400])

    return events


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
    print(f"  [Investing] Fetching {from_str} → {to_str}")
    print(f"  [Investing] NOTE: Times will be converted EST → UTC (+5h)")
    url = 'https://www.investing.com/economic-calendar/Service/getCalendarFilteredData'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://www.investing.com/economic-calendar/',
        'Origin': 'https://www.investing.com',
    }
    data = [
        ('dateFrom', from_str), ('dateTo', to_str),
        ('timeZone', '0'), ('timeFilter', 'timeRemain'),
        ('currentTab', 'custom'), ('limit_from', '0'),
    ]
    for cid in [5, 72, 4, 35, 25, 6, 12, 43]:
        data.append(('country[]', str(cid)))

    try:
        r = requests.post(url, headers=headers, data=data, timeout=25)
        if not r.ok:
            print(f"  [Investing] HTTP {r.status_code}")
            return []
        print(f"  [Investing] Got {len(r.content)} bytes")
        resp = r.json()
        html_data = resp.get('data', '') if isinstance(resp, dict) else ''
        if not html_data: return []

        soup = BeautifulSoup(html_data, 'lxml')
        rows = soup.find_all('tr', {'id': re.compile('eventRowId_')})
        print(f"  [Investing] Found {len(rows)} rows")

        cc_map = {
            'US':'USD','EU':'EUR','GB':'GBP','JP':'JPY',
            'AU':'AUD','CA':'CAD','CH':'CHF','NZ':'NZD',
            'DE':'EUR','FR':'EUR','IT':'EUR','ES':'EUR','PT':'EUR','NL':'EUR',
            'usd':'USD','eur':'EUR','gbp':'GBP','jpy':'JPY',
            'aud':'AUD','cad':'CAD','chf':'CHF','nzd':'NZD',
            'us':'USD','eu':'EUR','gb':'GBP','jp':'JPY',
            'au':'AUD','ca':'CAD','ch':'CHF','nz':'NZD',
            'de':'EUR','fr':'EUR','it':'EUR','es':'EUR',
        }
        events = []
        for row in rows:
            try:
                dt_str = row.get('data-event-datetime', '')
                event_date = None
                if dt_str:
                    dt_norm = dt_str.strip().replace('T', ' ')
                    for fmt in ['%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                                '%Y/%m/%d %H:%M',    '%Y-%m-%d %H:%M']:
                        try:
                            event_date = datetime.strptime(dt_norm[:19], fmt).date()
                            break
                        except: pass
                    if event_date is None:
                        m2 = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', dt_str)
                        if m2:
                            try: event_date = date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
                            except: pass
                if not event_date: continue

                # Extract raw EST time from datetime string
                raw_time_est = ''
                if dt_str:
                    dt_norm = dt_str.strip().replace('T', ' ')
                    m_t = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}[ T](\d{2}):(\d{2})', dt_norm)
                    if m_t:
                        raw_time_est = f"{m_t.group(1)}:{m_t.group(2)}"

                # Convert EST → UTC (Investing always returns EST)
                time_str_utc, event_date = est_to_utc(raw_time_est, event_date)

                # After date adjustment, check if still in target range
                # (extend target_dates by 1 day to catch overflow events)
                extended_dates = target_dates | {max(target_dates) + timedelta(days=1)}
                if event_date not in extended_dates: continue

                currency = ''
                dc = row.get('data-currency', '').strip().upper()
                if dc and dc in TRACKED_CURRENCIES:
                    currency = dc
                if not currency:
                    pair = row.get('data-img_pair', '').strip().lower()
                    if pair:
                        currency = cc_map.get(pair.upper(), cc_map.get(pair, ''))
                if not currency:
                    for sp in row.find_all('span', class_=True):
                        cls = ' '.join(sp.get('class', []))
                        m = re.search(r'flag_(\w+)', cls)
                        if m:
                            currency = cc_map.get(m.group(1).upper(), '')
                            if currency: break
                if not currency:
                    flag_td = row.find('td', class_=re.compile('flagCur'))
                    if flag_td:
                        txt = flag_td.get_text(strip=True).upper()
                        for c in TRACKED_CURRENCIES:
                            if c in txt:
                                currency = c; break
                if currency not in TRACKED_CURRENCIES: continue

                impact = 'low'
                sent_td = row.find('td', class_=re.compile('sentiment'))
                if sent_td:
                    icon = sent_td.find(['i', 'span'], class_=True)
                    if icon:
                        ic = ' '.join(icon.get('class', []))
                        title_attr = icon.get('title', '').lower()
                        if 'red' in ic or 'high' in ic.lower() or 'high' in title_attr:
                            impact = 'high'
                        elif 'orange' in ic or 'medium' in ic.lower() or 'medium' in title_attr:
                            impact = 'medium'

                ev_td = row.find('td', class_=re.compile('event'))
                event_name = ''
                if ev_td:
                    a = ev_td.find('a')
                    event_name = (a or ev_td).get_text(strip=True)
                    event_name = re.sub(r'\s+[A-Z]{3}/\d+$', '', event_name).strip()
                if not event_name: continue

                def gcell(pat):
                    td = row.find('td', id=re.compile(pat))
                    return clean_val(td.get_text(strip=True)) if td else ''

                actual   = gcell(r'eventActual_')
                forecast = gcell(r'eventForecast_')
                previous = gcell(r'eventPrevious_')

                if impact == 'low':
                    impact = classify_impact_kw(event_name)

                events.append({
                    'date': fmt_date(event_date),
                    'dateISO': event_date.isoformat(),
                    'timeUTC': time_str_utc,
                    'country': currency, 'currency': currency,
                    'flag': CURRENCY_FLAGS.get(currency, ''),
                    'event': event_name,
                    'impact': impact,
                    'actual': actual,
                    'forecast': forecast,
                    'previous': previous,
                })
            except: continue

        print(f"  [Investing] ✅ Parsed {len(events)} events")
        if len(events) == 0 and len(rows) > 0:
            first = rows[0]
            print(f"  [Investing] DEBUG first row attrs: {dict(first.attrs)}")
            print(f"  [Investing] DEBUG first row HTML[:300]: {str(first)[:300]}")
        return events
    except Exception as e:
        import traceback
        print(f"  [Investing] Error: {e}")
        print(f"  [Investing] Traceback: {traceback.format_exc()[:500]}")
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
print("ECONOMIC CALENDAR SCRAPER v6.0 (FF XML primary source)")
print("Sources: FF XML → [enrich actuals: Investing.com] → MQL5 RSS → TE RSS → Official RSS")
print("Timezone policy:")
print("  - MQL5/TE/Official RSS: pubDate normalized to UTC via parsedate_to_datetime")
print("  - Investing.com: returns EST (UTC-5), converted +5h to UTC before storing")
print("  - All timeUTC fields are UTC — frontend converts to user local timezone")
print("=" * 60)

today      = date.today()
from_date  = today - timedelta(days=1)
to_date    = today + timedelta(days=30)

target_dates = set()
d = from_date
while d <= to_date:
    target_dates.add(d); d += timedelta(days=1)

print(f"\nTarget: {from_date} → {to_date} ({len(target_dates)} days)\n")

all_events  = []
source_used = None
fetch_errors = []

# ── Strategy 0: Forex Factory XML (primary source for impact/forecast/previous) ──
print(f"\n{'='*50}\nSTRATEGY 0: Forex Factory XML\n{'='*50}")
ff_events = []
try:
    ff_events = fetch_ff_xml(target_dates)
    if len(ff_events) >= 3:
        all_events  = ff_events
        source_used = 'Forex Factory'
        print(f"✅ FF XML: {len(ff_events)} events — using as primary source")
    else:
        msg = f"FF XML: only {len(ff_events)} events returned"
        print(f"⚠️  {msg}")
        fetch_errors.append(msg)
except Exception as e:
    msg = f"FF XML failed: {e}"
    print(f"❌ {msg}")
    fetch_errors.append(msg)
time.sleep(1)

for label, fetcher, args in [
    ("1: MQL5 RSS",      fetch_mql5_rss,           (target_dates,)),
    ("2: TE RSS",        fetch_te_rss,             (target_dates,)),
    ("3: Investing.com", fetch_investing_calendar, (from_date.isoformat(), to_date.isoformat(), target_dates)),
    ("4: Official RSS",  fetch_official_rss,       (target_dates,)),
]:
    # If FF worked, only run Investing.com to enrich actuals; skip others
    if len(all_events) >= 3 and label != "3: Investing.com":
        continue
    print(f"\n{'='*50}\nSTRATEGY {label}\n{'='*50}")
    try:
        result = fetcher(*args)
        if len(all_events) >= 3 and label == "3: Investing.com":
            # Enrich FF events with actuals from Investing.com
            inv_index = {}
            for ev in result:
                if ev.get('actual'):
                    k = (ev['dateISO'], ev['currency'], ev['event'][:20].lower().strip())
                    inv_index[k] = ev
            enriched = 0
            for ev in all_events:
                if not ev.get('actual'):
                    k = (ev['dateISO'], ev['currency'], ev['event'][:20].lower().strip())
                    if k in inv_index:
                        ev['actual'] = inv_index[k].get('actual', '')
                        if not ev.get('forecast'):
                            ev['forecast'] = inv_index[k].get('forecast', '')
                        if not ev.get('previous'):
                            ev['previous'] = inv_index[k].get('previous', '')
                        enriched += 1
            print(f"  [Enrich] ✅ Filled actuals for {enriched} events from Investing.com")
            source_used = 'Forex Factory + Investing.com'
        elif len(result) >= 3 and len(all_events) < 3:
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

if not all_events:
    print(f"\n{'='*50}")
    print("⛔ ALL SOURCES FAILED — saving empty calendar with error status")
    for e in fetch_errors:
        print(f"   • {e}")
    print("=" * 50)

# Deduplicate
seen = set()
unique_events = []
for ev in all_events:
    k = (ev['dateISO'], ev['currency'], ev['event'][:25].lower().strip())
    if k not in seen:
        seen.add(k); unique_events.append(ev)

# Sort by dateISO + timeUTC
def sort_key(ev):
    t = ev.get('timeUTC') or '00:00'
    return ev['dateISO'] + 'T' + (t if re.match(r'\d{2}:\d{2}', t) else '00:00')

unique_events.sort(key=sort_key)

# Filter: keep future + yesterday-with-actual
final_events = []
for ev in unique_events:
    try:
        ev_date = date.fromisoformat(ev['dateISO'])
        if ev_date >= today:
            final_events.append(ev)
        elif ev_date >= today - timedelta(days=1) and ev.get('actual'):
            final_events.append(ev)
    except: pass

# Stats
currency_counts = {}
impact_counts   = {'high': 0, 'medium': 0, 'low': 0}
for ev in final_events:
    c = ev['currency']
    currency_counts[c] = currency_counts.get(c, 0) + 1
    impact_counts[ev.get('impact','low')] = impact_counts.get(ev.get('impact','low'), 0) + 1

import sys
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
    'rangeFrom':      from_date.isoformat(),
    'rangeTo':        to_date.isoformat(),
    'totalEvents':    len(final_events),
    'currencyCounts': currency_counts,
    'impactCounts':   impact_counts,
    'events':         final_events,
}

with open('calendar-data/calendar.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
if data_ok:
    print(f"✅ SAVED: calendar-data/calendar.json")
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
