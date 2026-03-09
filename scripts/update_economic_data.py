#!/usr/bin/env python3
"""
update_economic_data.py
Extrae datos económicos fundamentales para 8 divisas forex desde Trading Economics,
ONS (UK), OECD (EUR) y World Bank.

Extraído desde .github/workflows/update-economic-data.yml como parte de la
refactorización P-01 (auditoría de código). El workflow ahora llama a este script
en lugar de embeber el código Python inline.
"""

import requests, sys
sys.stdout.reconfigure(line_buffering=True)
from bs4 import BeautifulSoup
import json, re, time
from datetime import date, datetime

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}
JSON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/csv, */*',
}

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']

TRADING_ECONOMICS_URLS = {
    'gdp':              'https://tradingeconomics.com/country-list/gdp?continent=world',
    'gdpGrowth':        'https://tradingeconomics.com/country-list/gdp-growth-rate?continent=world',
    'inflation':        'https://tradingeconomics.com/country-list/inflation-rate?continent=world',
    'unemployment':     'https://tradingeconomics.com/country-list/unemployment-rate?continent=world',
    'debt':             'https://tradingeconomics.com/country-list/government-debt-to-gdp',
    'currentAccount':   'https://tradingeconomics.com/country-list/current-account-to-gdp?continent=world',
    'production':       'https://tradingeconomics.com/country-list/industrial-production-mom?continent=world',
    'tradeBalance':     'https://tradingeconomics.com/country-list/balance-of-trade',
    'retailSales':      'https://tradingeconomics.com/country-list/retail-sales-mom',
    'wageGrowth':       'https://tradingeconomics.com/country-list/wage-growth',
    'manufacturingPMI': 'https://tradingeconomics.com/country-list/manufacturing-pmi',
    'servicesPMI':      'https://tradingeconomics.com/country-list/services-pmi',
    'termsOfTrade':     'https://tradingeconomics.com/country-list/terms-of-trade',
}

COUNTRY_URLS = {
    'USD': 'united-states', 'EUR': 'euro-area', 'GBP': 'united-kingdom',
    'JPY': 'japan',         'CAD': 'canada',    'AUD': 'australia',
    'CHF': 'switzerland',   'NZD': 'new-zealand',
}

INDICATOR_PAGES = {
    'retailSales':      'retail-sales',
    'wageGrowth':       'wage-growth',
    'manufacturingPMI': 'manufacturing-pmi',
    'servicesPMI':      'services-pmi',
    'production':       'industrial-production',
}

COUNTRY_NAMES = {
    'USD': ['United States'], 'EUR': ['Euro Area'], 'GBP': ['United Kingdom'],
    'JPY': ['Japan'],         'CAD': ['Canada'],    'AUD': ['Australia'],
    'CHF': ['Switzerland'],   'NZD': ['New Zealand'],
}

TOT_MIN, TOT_MAX = 50.0, 200.0

exchange_rates = {}

def fetch_fx():
    global exchange_rates
    try:
        r = requests.get('https://api.frankfurter.app/latest?from=USD', timeout=10)
        if r.ok:
            exchange_rates = r.json().get('rates', {})
            exchange_rates['USD'] = 1.0
            return
    except:
        pass
    exchange_rates = {'USD':1.0,'EUR':0.84,'GBP':0.73,'JPY':153.71,
                      'CAD':1.36,'AUD':1.40,'CHF':0.77,'NZD':1.65}

def clean_num(text):
    if not text:
        return None
    text = str(text).strip().replace(',','').replace('%','')
    m = re.search(r'(-?\d+\.?\d*)', text)
    return float(m.group(1)) if m else None

def parse_te_date(date_text):
    if not date_text:
        return str(date.today())
    date_text = date_text.strip()
    try:
        m = re.search(r'([A-Za-z]{3})\s*[/\s]\s*(\d{2,4})', date_text)
        if m:
            yr = m.group(2)
            if len(yr) == 2: yr = '20' + yr
            dt = datetime.strptime(f"{m.group(1)} {yr}", '%b %Y')
            return dt.strftime('%Y-%m-15')
        m = re.search(r'Q(\d)[/\s](\d{4})', date_text)
        if m:
            q, yr = int(m.group(1)), int(m.group(2))
            return f"{yr}-{(q-1)*3+2:02d}-15"
        if re.match(r'^\d{4}$', date_text):
            return f"{date_text}-06-15"
    except:
        pass
    return str(date.today())

def te_get(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 429:
        print(f"  ⚠ Rate limited, waiting 30s...")
        time.sleep(30)
        r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r

def fetch_tot_eur_oecd():
    legacy_urls = [
        "https://stats.oecd.org/SDMX-JSON/data/MEI_PRICES_PPI/EA20.PPPI.A/all"
        "?dimensionAtObservation=allDimensions&lastNObservations=3",
        "https://stats.oecd.org/SDMX-JSON/data/MEI/EA20.PRICEULNA.A/all"
        "?dimensionAtObservation=allDimensions&lastNObservations=3",
    ]
    for url in legacy_urls:
        print(f"    → OECD legacy: {url[:80]}...")
        try:
            r = requests.get(url, headers=JSON_HEADERS, timeout=8)
            if r.status_code in (400, 404, 501):
                print(f"    ✗ HTTP {r.status_code}")
                continue
            r.raise_for_status()
            data = r.json()
            obs = data.get('dataSets', [{}])[0].get('observations', {})
            if not obs:
                print(f"    ✗ OECD legacy: empty observations")
                continue
            vals = []
            for key, v in obs.items():
                if v and v[0] is not None:
                    vals.append((key, float(v[0])))
            vals.sort(key=lambda x: x[0], reverse=True)
            for key, val in vals:
                if TOT_MIN <= val <= TOT_MAX:
                    try:
                        time_idx = int(key.split(':')[-1])
                        time_dim = data['structure']['dimensions']['observation']
                        time_entry = next(d for d in time_dim if d['id'] == 'TIME_PERIOD')
                        yr = time_entry['values'][time_idx]['id'][:4]
                    except Exception:
                        yr = str(date.today().year - 1)
                    print(f"    ✓ EUR [OECD legacy]: {val} ({yr})")
                    return round(val, 2), f"{yr}-06-15"
            print(f"    ✗ OECD legacy: no value in range (got {[v for _,v in vals[:3]]})")
        except Exception as e:
            print(f"    ✗ OECD legacy: {e}")

    new_urls = [
        "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_TOT@DF_TOT"
        "/EA20.A.NBTT...?lastNObservations=3&format=jsondata",
        "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAAG@DF_NAAG_I"
        "/EA20..TOT..A?lastNObservations=3&format=jsondata",
    ]
    for url in new_urls:
        print(f"    → OECD new: {url[:80]}...")
        try:
            r = requests.get(url, headers=JSON_HEADERS, timeout=8)
            if r.status_code in (400, 404, 501):
                print(f"    ✗ HTTP {r.status_code}")
                continue
            r.raise_for_status()
            data = r.json()
            obs = data.get('dataSets', [{}])[0].get('observations', {})
            if not obs:
                continue
            vals = sorted(
                [(k, float(v[0])) for k, v in obs.items() if v and v[0] is not None],
                key=lambda x: x[0], reverse=True
            )
            for key, val in vals:
                if TOT_MIN <= val <= TOT_MAX:
                    try:
                        time_idx = int(key.split(':')[-1])
                        time_dim = data['structure']['dimensions']['observation']
                        time_entry = next(d for d in time_dim if d['id'] == 'TIME_PERIOD')
                        yr = time_entry['values'][time_idx]['id'][:4]
                    except Exception:
                        yr = str(date.today().year - 1)
                    print(f"    ✓ EUR [OECD new]: {val} ({yr})")
                    return round(val, 2), f"{yr}-06-15"
        except Exception as e:
            print(f"    ✗ OECD new: {e}")
    return None, None

def scrape_tot_ons_gbp():
    urls = [
        "https://api.ons.gov.uk/v1/datasets/MRET/timeseries/CTSL/data",
        "https://api.ons.gov.uk/v1/timeseries/CTSL/dataset/MRET/data",
        "https://api.ons.gov.uk/dataset/MRET/timeseries/CTSL/data",
    ]
    for url in urls:
        print(f"    → ONS: {url}")
        try:
            r = requests.get(url, headers=JSON_HEADERS, timeout=20)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            data = r.json()
            quarters = data.get('quarters', [])
            if not quarters:
                print(f"    ✗ ONS: no quarters. Keys: {list(data.keys())[:5]}")
                continue
            def qsort(q):
                m = re.match(r'(\d{4})\s+Q(\d)', q.get('date',''))
                return int(m.group(1))*10+int(m.group(2)) if m else 0
            for q in sorted(quarters, key=qsort, reverse=True):
                val = clean_num(q.get('value',''))
                date_str = q.get('date','')
                if val is not None and TOT_MIN <= val <= TOT_MAX:
                    m = re.match(r'(\d{4})\s+Q(\d)', date_str)
                    if m:
                        yr, qn = int(m.group(1)), int(m.group(2))
                        d = f"{yr}-{(qn-1)*3+2:02d}-15"
                    else:
                        d = str(date.today())
                    print(f"    ✓ GBP [ONS CTSL]: {val} ({date_str})")
                    return round(val, 2), d
        except Exception as e:
            print(f"    ✗ ONS {url}: {e}")
    return None, None

def scrape_tot_wb(currency_code, timeout=30):
    wb_map = {
        'USD':'US','GBP':'GB','JPY':'JP','CAD':'CA',
        'AUD':'AU','CHF':'CH','NZD':'NZ','EUR':'XC',
    }
    wb = wb_map.get(currency_code)
    if not wb:
        return None, None
    url = (f"https://api.worldbank.org/v2/country/{wb}"
           f"/indicator/TT.PRI.MRCH.XD.WD?format=json&mrv=5&per_page=5")
    try:
        r = requests.get(url, headers=JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2 or not data[1]:
            return None, None
        for obs in data[1]:
            val = obs.get('value')
            if val is None:
                continue
            val = float(val)
            if TOT_MIN <= val <= TOT_MAX:
                yr = obs.get('date', str(date.today().year))
                print(f"    ✓ {currency_code} [World Bank ToT {yr}]: {val}")
                return round(val, 2), f"{yr}-06-15"
    except Exception as e:
        print(f"    ✗ World Bank ToT {currency_code}: {e}")
    return None, None

def fetch_all_tot(te_list_data, te_list_dates):
    """
    ToT resolution chain:
      1. TE list values (all currencies, reject out-of-range).
         NZD: TE uses base 2002Q2=1000 → raw value saved as NZD_RAW,
         normalized by ÷10 to base-~100. Then validated against
         institutional bounds (C6: TOT_INSTITUTIONAL_MAX=130).
         If normalized > 130, World Bank cross-validation attempted.
      2. GBP  → ONS API (quarterly, most current)
      3. EUR  → OECD legacy → OECD new → Germany proxy (from TE list)
      4. Rest → World Bank (annual, ~2yr lag)
    """
    data  = {}
    dates = {}

    de_proxy_val  = None
    de_proxy_date = str(date.today())
    for curr, val in te_list_data.items():
        if curr == 'DE_PROXY':
            if TOT_MIN <= val <= TOT_MAX:
                de_proxy_val  = val
                de_proxy_date = te_list_dates.get('DE_PROXY', str(date.today()))
            continue
        if curr == 'NZD_RAW':
            continue  # handled below
        if TOT_MIN <= val <= TOT_MAX:
            data[curr]  = val
            dates[curr] = te_list_dates.get(curr, str(date.today()))
        else:
            print(f"  ✗ {curr} [TE list]: {val} out of range → specialist scraper")

    # ── C6: NZD ToT bounds check data-driven ──────────────────────────────
    # NZD: normalize raw TE value (base 2002Q2=1000) by dividing by 10
    # C6: bounds check post-normalización.
    # Ningún G8 debería tener ToT > 130 (base ~100).
    # Si el normalizado sigue fuera de rango → World Bank como validación cruzada.
    TOT_INSTITUTIONAL_MAX = 130.0  # bound data-driven G8
    if 'NZD' not in data:
        raw = te_list_data.get('NZD_RAW')
        if raw is not None and 500.0 <= raw <= 2500.0:
            normalized = round(raw / 10.0, 2)
            d = te_list_dates.get('NZD_RAW', str(date.today()))
            if normalized <= TOT_INSTITUTIONAL_MAX:
                print(f"  ✓ NZD: TE raw={raw} → normalized {normalized} (÷10, base 2002Q2=1000→~100)")
                data['NZD']  = normalized
                dates['NZD'] = d
            else:
                print(f"  ⚠ NZD: normalized={normalized} > {TOT_INSTITUTIONAL_MAX} (bounds G8)")
                print(f"    → Intentando World Bank para validación cruzada...")
                wb_val, wb_d = scrape_tot_wb('NZD', timeout=20)
                if wb_val is not None:
                    print(f"    ✓ NZD [World Bank]: {wb_val} — usando WB como fuente primaria")
                    data['NZD']  = wb_val
                    dates['NZD'] = wb_d
                else:
                    # Sin validación disponible: usar normalizado con advertencia
                    print(f"    ⚠ NZD: World Bank no disponible, usando {normalized} con advertencia")
                    data['NZD']  = normalized
                    dates['NZD'] = d
    # ── fin C6 ────────────────────────────────────────────────────────────

    missing = [c for c in CURRENCIES if c not in data]
    if not missing:
        return data, dates

    print(f"\n  → Specialist ToT for: {', '.join(missing)}")

    # GBP via ONS
    if 'GBP' in missing:
        val, d = scrape_tot_ons_gbp()
        if val is not None:
            data['GBP']  = val
            dates['GBP'] = d
            missing.remove('GBP')

    # EUR — Germany proxy first (zero requests), OECD only if unavailable
    if 'EUR' in missing:
        if de_proxy_val is not None:
            data['EUR']  = de_proxy_val
            dates['EUR'] = de_proxy_date
            missing.remove('EUR')
            print(f"  ✓ EUR [proxy:DE={de_proxy_val}] (skipping slow OECD)")
        else:
            print("  → EUR ToT: no DE proxy, trying OECD API...")
            val, d = fetch_tot_eur_oecd()
            if val is not None:
                data['EUR']  = val
                dates['EUR'] = d
                missing.remove('EUR')
            else:
                print("  ✗ EUR: OECD also failed, no data")

    # World Bank for anything still missing
    still_missing = [c for c in CURRENCIES if c not in data]
    if still_missing:
        print(f"\n  → World Bank ToT fallback for: {', '.join(still_missing)}")
        for curr in still_missing:
            val, d = scrape_tot_wb(curr, timeout=30)
            if val is not None:
                data[curr]  = val
                dates[curr] = d
            time.sleep(0.5)

    final_missing = [c for c in CURRENCIES if c not in data]
    if final_missing:
        print(f"  ⚠ No ToT available for: {', '.join(final_missing)} → null")

    return data, dates

def scrape_country_page(country_slug, indicator_page, currency_code):
    url = f'https://tradingeconomics.com/{country_slug}/{indicator_page}'
    try:
        r = te_get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        for table in soup.find_all('table'):
            headers = [h.get_text(strip=True).lower() for h in table.find_all('th')]
            if 'last' in headers and 'reference' in headers:
                li, ri = headers.index('last'), headers.index('reference')
                kw = indicator_page.replace('-',' ').split()[0].lower()
                for row in table.find_all('tr')[1:]:
                    cells = row.find_all('td')
                    if len(cells) <= max(li, ri): continue
                    if kw in cells[0].get_text(strip=True).lower():
                        val = clean_num(cells[li].get_text(strip=True))
                        if val is not None:
                            d = parse_te_date(cells[ri].get_text(strip=True))
                            print(f"    ✓ {currency_code} [related-table]: {val} ({d})")
                            return val, d
        for table in soup.find_all('table'):
            headers = [h.get_text(strip=True).lower() for h in table.find_all('th')]
            if 'actual' in headers:
                ai = headers.index('actual')
                di = headers.index('reference') if 'reference' in headers else None
                rows = table.find_all('tr')[1:]
                if rows:
                    cells = rows[0].find_all('td')
                    if len(cells) > ai:
                        val = clean_num(cells[ai].get_text(strip=True))
                        d = str(date.today())
                        if di and len(cells) > di:
                            d = parse_te_date(cells[di].get_text(strip=True))
                        if val is not None:
                            print(f"    ✓ {currency_code} [actual-table]: {val} ({d})")
                            return val, d
    except Exception as e:
        print(f"    ✗ {country_slug}: {e}")
    return None, None

def parse_tb(cols):
    if len(cols) < 5: return None
    val  = clean_num(cols[1].get_text(strip=True))
    unit = cols[4].get_text(strip=True)
    if val is None: return None
    scale = 1000 if 'billion' in unit.lower() else 1
    curr  = next((c for c in CURRENCIES if c in unit.upper()), 'USD')
    val_m = val * scale
    if curr != 'USD':
        fx = exchange_rates.get(curr, 1)
        val_m = val_m / fx if fx else val_m
    return round(val_m, 2)

def scrape_te_list(url, indicator):
    print(f"\n{'='*50}\nSCRAPING: {indicator.upper()}\n{'='*50}")
    DE_NAMES = ['Germany']
    try:
        r = te_get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        data, dates = {}, {}
        table = soup.find('table', {'class': 'table'})
        if not table: return data, dates
        headers = [h.get_text(strip=True).lower() for h in table.find_all('th')]
        ai = 1
        if 'actual' in headers: ai = headers.index('actual')
        elif 'last' in headers: ai = headers.index('last')
        di = headers.index('reference') if 'reference' in headers else None
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) < 2: continue
            ctry = cols[0].get_text(strip=True)

            if indicator == 'termsOfTrade' and 'DE_PROXY' not in data:
                if any(n.lower() in ctry.lower() for n in DE_NAMES):
                    val = clean_num(cols[ai].get_text(strip=True))
                    d = str(date.today())
                    if di and len(cols) > di:
                        d = parse_te_date(cols[di].get_text(strip=True))
                    if val is not None and TOT_MIN <= val <= TOT_MAX:
                        data['DE_PROXY']  = round(val, 2)
                        dates['DE_PROXY'] = d
                        print(f"  ✓ DE (EUR proxy): {val} ({d})")

            for code, names in COUNTRY_NAMES.items():
                if any(n.lower() in ctry.lower() for n in names):
                    val = parse_tb(cols) if indicator == 'tradeBalance' \
                          else clean_num(cols[ai].get_text(strip=True))
                    d = str(date.today())
                    if di and len(cols) > di:
                        d = parse_te_date(cols[di].get_text(strip=True))
                    if val is not None:
                        if indicator == 'termsOfTrade' and not (TOT_MIN <= val <= TOT_MAX):
                            print(f"  ✗ {code}: {val} (out of range)")
                            if code == 'NZD':
                                # Save raw for normalization (base 2002Q2=1000 → ÷10)
                                data['NZD_RAW']  = round(val, 2)
                                dates['NZD_RAW'] = d
                            break
                        data[code]  = round(val, 2)
                        dates[code] = d
                        print(f"  ✓ {code}: {val} ({d})")
                    break
        return data, dates
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {}, {}

def scrape_fallback(indicator, url):
    if indicator == 'termsOfTrade':
        print(f"\n{'='*50}\nSCRAPING: TERMSOFTRADE\n{'='*50}")
        te_data, te_dates = scrape_te_list(url, indicator)
        return fetch_all_tot(te_data, te_dates)

    data, dates = scrape_te_list(url, indicator)
    missing = [c for c in CURRENCIES if c not in data]
    if missing and indicator in INDICATOR_PAGES:
        print(f"\n  → Country page fallback for: {', '.join(missing)}")
        for code in missing:
            slug = COUNTRY_URLS.get(code)
            if slug:
                val, d = scrape_country_page(slug, INDICATOR_PAGES[indicator], code)
                if val is not None:
                    data[code]  = val
                    dates[code] = d or str(date.today())
            time.sleep(1)
    return data, dates

def load_cot():
    import os
    cot = {}
    if not os.path.exists('cot-data'): return {}
    for c in ['EUR','GBP','JPY','CAD','AUD','CHF','NZD']:
        fp = f'cot-data/{c}.json'
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    cot[c] = json.load(f).get('netPosition')
            except: pass
    return cot

print("="*60)
print("ECONOMIC DATA SCRAPER v11.4 + C6 (institutional ToT bounds)")
print("ToT NZD: TE raw ÷10 → bounds check (max 130) → World Bank fallback")
print("ToT EUR: OECD legacy → OECD new → Germany proxy")
print("ToT GBP: ONS CTSL → World Bank")
print("ToT REST: World Bank fallback")
print("="*60)

fetch_fx()
all_data  = {c: {} for c in CURRENCIES}
all_dates = {c: {} for c in CURRENCIES}

for ind, url in TRADING_ECONOMICS_URLS.items():
    ind_data, ind_dates = scrape_fallback(ind, url)
    for code, val in ind_data.items():
        all_data[code][ind]  = val
        all_dates[code][ind] = ind_dates.get(code, str(date.today()))
    time.sleep(2)

for code, val in load_cot().items():
    all_data[code]['cotPositioning'] = val

today = str(date.today())
print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
for curr in CURRENCIES:
    pkg = {
        'lastUpdate': today,
        'source': 'TradingEconomics + ONS (GBP) + OECD (EUR) + WorldBank + COT — C6 ToT bounds',
        'data':   all_data[curr],
        'dates':  all_dates[curr],
    }
    with open(f'economic-data/{curr}.json', 'w') as f:
        json.dump(pkg, f, indent=2)
    missing = [k for k in ['retailSales','wageGrowth','manufacturingPMI','servicesPMI','termsOfTrade']
               if all_data[curr].get(k) is None]
    tot    = all_data[curr].get('termsOfTrade')
    status = '✓' if not missing else '⚠'
    print(f"  {status} {curr}: ToT={tot}{' | Missing: '+', '.join(missing) if missing else ''}")

print("\nCOMPLETED")
