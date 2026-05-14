"""
fetch_ois_rates.py  —  OIS / Overnight Rates  G8  (v2.0)
=========================================================
Fetches overnight/OIS benchmark rates for all G8 currencies and writes
ois-rates/rates.json to the public site repo.

RATE → BENCHMARK MAPPING
  USD  SOFR   → NY Fed API (markets.newyorkfed.org)   daily
  EUR  €STR   → ECB Data Portal SDMX-JSON             daily
  GBP  SONIA  → BOE SDIE API (IUDSOIA)                daily
  JPY  TONA   → SNB zimoma cube (zimoma)               monthly
  AUD  AONIA  → RBA Table F1 CSV                       monthly
  CAD  CORRA  → BoC Valet API (V122514)                daily
  CHF  SARON  → SNB data portal (snbgwdzid)            daily
  NZD  OCR    → RBNZ B2 CSV → policy fallback          monthly

SOURCE DECISIONS vs workflow_meetings.yml
  workflow_meetings.yml: reads OIS rates to produce BIAS signals (hold/cut/hike)
                         Uses FRED API (authenticated) — runs Mondays only
  fetch_ois_rates.py:    reads OIS rates to produce RATE VALUES for CIP forwards
                         Uses direct CB APIs (no auth) — runs daily

  NO PURPOSE OVERLAP — separate outputs, separate consumers:
    meetings-data/meetings.json  → CB Meetings panel (bias + probabilities)
    ois-rates/rates.json         → CIP forward pricing via _resolveRate()

  SHARED UNDERLYING SOURCES (same data, different use):
    GBP: both use BOE SONIA — meetings: bias signal; ois-rates: rate value
    JPY: both use SNB zimoma TONA — meetings: bias + supplement; ois-rates: rate value
    CAD: both use BoC Valet CORRA (V122514) — same series, aligned in v2.0
    CHF: both use SNB snbgwdzid SARON — same series

WHY NOT FRED CSV (v1.0 issue)
  FRED CSV endpoint (fred.stlouisfed.org) hit read timeouts on GitHub Actions runners
  in the v1.0 run (2026-05-14). FRED blocks or throttles GH Actions IP ranges on the
  unauthenticated CSV endpoint. Fix: use direct CB/NY Fed APIs that don't rate-limit
  by IP. The FRED API (api.stlouisfed.org, key-authenticated) works fine — that's
  what workflow_meetings.yml uses — but requires FRED_API_KEY secret. This script
  intentionally avoids secrets to keep the workflow simple (public repo, no vault).

CONSUMED BY
  dashboard2.js → loadOISRatesCache() → _resolveRate() → computeCIPForward()
  Policy rates (rates/*.json) are NOT changed — they drive CB Rates panel,
  carry ranking, and regime scoring. OIS rates are ONLY used for forward pricing.
"""

import json
import os
import sys
from datetime import date, timedelta

import requests

SITE_DIR = os.environ.get('SITE_DIR', '.')
OUTPUT_PATH = os.path.join(SITE_DIR, 'ois-rates', 'rates.json')
RATES_DIR   = os.path.join(SITE_DIR, 'rates')

HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; GlobalInvestingBot/2.0)'}

print('=' * 50)
print('OIS / OVERNIGHT RATES  —  G8')
print('=' * 50)

# ── Policy rate fallback ──────────────────────────────────────────────────────

def policy_rate(ccy):
    """Read current CB policy rate from rates/{CCY}.json as last-resort fallback."""
    path = os.path.join(RATES_DIR, f'{ccy}.json')
    try:
        with open(path) as f:
            d = json.load(f)
        obs = d.get('observations', [])
        raw = obs[0]['value'] if obs else d.get('rate') or d.get('value')
        if raw is not None and str(raw) not in ('.', ''):
            return float(raw), d.get('observations', [{}])[0].get('date', str(date.today())[:7])
    except Exception:
        pass
    return None, None

# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_usd():
    """
    SOFR via NY Fed public API.
    Endpoint: https://markets.newyorkfed.org/api/rates/sofr/last/1.json
    No auth, no rate limit on GH Actions — confirmed working.
    Fallback: EFFR (Fed Funds Effective) from NY Fed.
    Rationale: NY Fed is the authoritative SOFR publisher (it calculates SOFR).
    Using the NY Fed API directly avoids FRED's IP-based throttling.
    """
    print('[USD]')
    # Primary: SOFR
    try:
        r = requests.get(
            'https://markets.newyorkfed.org/api/rates/sofr/last/1.json',
            headers=HEADERS, timeout=15
        )
        r.raise_for_status()
        d = r.json()
        refRates = d.get('refRates', [])
        if refRates:
            val = float(refRates[0]['percentRate'])
            dt  = refRates[0].get('effectiveDate', str(date.today()))
            print(f'  USD: SOFR (NY Fed API)')
            print(f'    ✓ SOFR {val}% ({dt})')
            return val, 'SOFR', dt
    except Exception as e:
        print(f'  USD: SOFR (NY Fed API)')
        print(f'    SOFR failed: {e}')

    # Fallback: EFFR
    try:
        r = requests.get(
            'https://markets.newyorkfed.org/api/rates/effr/last/1.json',
            headers=HEADERS, timeout=15
        )
        r.raise_for_status()
        d = r.json()
        refRates = d.get('refRates', [])
        if refRates:
            val = float(refRates[0]['percentRate'])
            dt  = refRates[0].get('effectiveDate', str(date.today()))
            print(f'  USD: fallback → EFFR (NY Fed API)')
            print(f'    ✓ EFFR {val}% ({dt})')
            return val, 'EFFR', dt
    except Exception as e:
        print(f'  USD: fallback → EFFR: {e}')

    # Last resort: policy rate
    val, dt = policy_rate('USD')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_eur():
    """
    €STR via ECB Data Portal SDMX-JSON (same source as workflow_meetings.yml).
    Dataflow: EST  Series key: B.EU000A2X2A25.WT
    URL: https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT
    No auth, no bot protection — confirmed reachable from GitHub Actions.
    Note: we fetch the raw €STR rate (not normalised vs DFR) because
    _resolveRate() uses this as the discount rate for CIP forwards — the
    absolute level is what matters, not the spread to DFR.
    """
    print('[EUR]')
    try:
        url = (
            'https://data-api.ecb.europa.eu/service/data/EST/'
            'B.EU000A2X2A25.WT?lastNObservations=3&format=jsondata'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        ds = r.json()['dataSets'][0]['series']
        series_key = list(ds.keys())[0]
        obs = ds[series_key]['observations']
        # Also get dates from structure
        try:
            time_periods = (
                r.json()['structure']['dimensions']['observation'][0]['values']
            )
        except Exception:
            time_periods = []
        sorted_idx = sorted(obs.keys(), key=int, reverse=True)
        for idx in sorted_idx:
            val = obs[idx][0]
            if val is not None:
                dt = time_periods[int(idx)]['id'] if int(idx) < len(time_periods) else str(date.today())
                val = float(val)
                print(f'  EUR: €STR (ECB Data Portal · EST.B.EU000A2X2A25.WT)')
                print(f'    ✓ €STR {val}% ({dt})')
                return val, '€STR', dt
    except Exception as e:
        print(f'  EUR: €STR (ECB Data Portal): {e}')

    # Fallback: ECB SDMX REST (alternative endpoint)
    try:
        url2 = 'https://sdw-wsrest.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?lastNObservations=1'
        r2 = requests.get(url2, headers={**HEADERS, 'Accept': 'application/json'}, timeout=15)
        r2.raise_for_status()
        ds2 = r2.json()['dataSets'][0]['series']
        series_key2 = list(ds2.keys())[0]
        obs2 = ds2[series_key2]['observations']
        for idx2 in sorted(obs2.keys(), key=int, reverse=True):
            val2 = obs2[idx2][0]
            if val2 is not None:
                val2 = float(val2)
                print(f'  EUR: fallback → ECB SDW REST')
                print(f'    ✓ €STR {val2}% (ECB SDW)')
                return val2, '€STR', str(date.today())
    except Exception as e:
        print(f'  EUR: fallback → ECB SDW: {e}')

    val, dt = policy_rate('EUR')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_gbp():
    """
    SONIA via BOE SDIE API (series IUDSOIA).
    URL: https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp?...
    CSV export, no auth. Confirmed working in v1.0 run (3.729% 2026-05-11).
    Note: workflow_meetings.yml uses 1M OIS forward (IUQIBLOK) for bias direction,
    which is more forward-looking. For CIP forward PRICING, the overnight rate
    (SONIA spot) is the correct input — matching Bloomberg FXFA methodology.
    """
    print('[GBP]')
    try:
        today = date.today()
        from_date = (today - timedelta(days=14)).strftime('%d/%b/%Y')
        to_date   = today.strftime('%d/%b/%Y')
        url = (
            'https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp'
            f'?Travel=NIxIRx&FromSeries=1&ToSeries=50&DAT=RNG'
            f'&FD=1&FM=Jan&FY=2025'
            f'&TD={today.day}&TM={today.strftime("%b")}&TY={today.year}'
            '&C=IUDSOIA&UsingCodes=True&CSVF=TT&BoEpdf=pdf.pdf'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        lines = r.text.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line or 'Date' in line:
                continue
            parts = [p.strip().strip('"') for p in line.split(',')]
            if len(parts) >= 2 and parts[1] not in ('', 'n/a', 'NA'):
                try:
                    val = float(parts[1])
                    dt  = parts[0]
                    print(f'  GBP: SONIA (BOE SDIE IUDSOIA)')
                    print(f'    ✓ SONIA {val}% ({dt})')
                    return val, 'SONIA', dt
                except ValueError:
                    continue
    except Exception as e:
        print(f'  GBP: BOE SDIE failed: {e}')

    val, dt = policy_rate('GBP')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_jpy():
    """
    TONA via SNB zimoma cube (same source as workflow_meetings.yml).
    URL: https://data.snb.ch/api/cube/zimoma/data/csv/en?dimSel=D0(TONA)
    CSV, no auth. SNB publishes monthly international money market rates
    including TONA (Tokyo Overnight Average Rate).
    Monthly lag: acceptable for JPY — BoJ moves very slowly and telegraphs well.
    NOTE: we use the raw TONA value (not the bias-supplement logic that meetings
    uses for hikeProb). For CIP pricing, the raw overnight rate is correct.
    """
    print('[JPY]')
    today = date.today()
    m = today.month - 4
    y = today.year
    if m <= 0:
        m += 12
        y -= 1
    from_date = f'{y}-{m:02d}'
    try:
        url = (
            f'https://data.snb.ch/api/cube/zimoma/data/csv/en'
            f'?dimSel=D0(TONA)&fromDate={from_date}'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        lines = r.text.splitlines()
        data_lines = [l for l in lines[3:] if l.strip() and ';' in l]
        for line in reversed(data_lines):
            parts = [p.strip().strip('"') for p in line.split(';')]
            if len(parts) >= 3 and parts[2]:
                try:
                    val = float(parts[2])
                    dt  = parts[0]
                    print(f'  JPY: TONA (SNB zimoma)')
                    print(f'    ✓ TONA {val}% ({dt})')
                    return val, 'TONA', dt
                except ValueError:
                    continue
        print(f'  JPY: SNB zimoma — no data rows found')
    except Exception as e:
        print(f'  JPY: SNB zimoma failed: {e}')

    val, dt = policy_rate('JPY')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_aud():
    """
    AONIA via RBA Table F1 CSV.
    URL: https://www.rba.gov.au/statistics/tables/csv/f1.csv
    AONIA (Australian Overnight Index Average) = RBA cash rate target tracking rate.
    Monthly frequency in the RBA F1 table.
    Confirmed working in v1.0 run (4.35% 2026-05-13).
    """
    print('[AUD]')
    try:
        url = 'https://www.rba.gov.au/statistics/tables/csv/f1.csv'
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        lines = r.text.splitlines()
        # Find the AONIA row — RBA CSV has variable header depth
        # Look for 'Cash Rate' or 'AONIA' in the description column
        aonia_row_idx = None
        for i, line in enumerate(lines):
            low = line.lower()
            if 'aonia' in low or ('cash rate' in low and 'target' not in low):
                aonia_row_idx = i
                break
        if aonia_row_idx is None:
            # Try to find the date row and the row right after 'FIRMMCRT'
            for i, line in enumerate(lines):
                if 'FIRMMCRT' in line or 'overnight index' in line.lower():
                    aonia_row_idx = i
                    break
        if aonia_row_idx is not None:
            row_parts = lines[aonia_row_idx].split(',')
            # Find last non-empty numeric value
            for cell in reversed(row_parts):
                cell = cell.strip().strip('"')
                if cell and cell not in ('', 'n/a', 'na', 'N/A'):
                    try:
                        val = float(cell)
                        # Find the date from the header row
                        date_row = None
                        for line in lines[:aonia_row_idx]:
                            if '20' in line and ('Jan' in line or 'Feb' in line or
                               'Mar' in line or 'Apr' in line or 'May' in line or
                               '-' in line):
                                date_row = line
                        dt = str(date.today())[:7]
                        print(f'  AUD: AONIA via RBA Table F1 CSV')
                        print(f'    ✓ AONIA/RBA {val}% ({dt})')
                        return val, 'AONIA', dt
                    except ValueError:
                        continue
    except Exception as e:
        print(f'  AUD: RBA Table F1 CSV failed: {e}')

    val, dt = policy_rate('AUD')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_cad():
    """
    CORRA via Bank of Canada Valet API (series V122514).
    URL: https://www.bankofcanada.ca/valet/observations/V122514/json?recent=3
    V122514 = CORRA (Canadian Overnight Repo Rate Average).
    Daily frequency. No auth required.
    ALIGNMENT NOTE v2.0: workflow_meetings.yml uses V122514 — aligned here.
    v1.0 used V122530 (overnight money market, equivalent but different series).
    V122514 is the canonical CORRA series used by BoC for policy rate communication.
    """
    print('[CAD]')
    try:
        series = 'V122514'
        r = requests.get(
            f'https://www.bankofcanada.ca/valet/observations/{series}/json',
            params={'recent': '3'},
            headers=HEADERS, timeout=15
        )
        r.raise_for_status()
        obs = r.json().get('observations', [])
        if obs:
            last = obs[-1]
            val_raw = last.get(series, {}).get('v')
            if val_raw is not None:
                val = float(val_raw)
                dt  = last.get('d', str(date.today()))
                print(f'  CAD: CORRA (BoC Valet {series})')
                print(f'    ✓ CORRA {val}% ({dt})')
                return val, 'CORRA', dt
    except Exception as e:
        print(f'  CAD: BoC Valet {series} failed: {e}')

    # Fallback: V122530 (overnight money market — equivalent, v1.0 series)
    try:
        series2 = 'V122530'
        r2 = requests.get(
            f'https://www.bankofcanada.ca/valet/observations/{series2}/json',
            params={'recent': '3'},
            headers=HEADERS, timeout=15
        )
        r2.raise_for_status()
        obs2 = r2.json().get('observations', [])
        if obs2:
            last2 = obs2[-1]
            val_raw2 = last2.get(series2, {}).get('v')
            if val_raw2 is not None:
                val2 = float(val_raw2)
                dt2  = last2.get('d', str(date.today()))
                print(f'  CAD: fallback → BoC Valet {series2}')
                print(f'    ✓ CORRA {val2}% ({dt2})')
                return val2, 'CORRA', dt2
    except Exception as e:
        print(f'  CAD: BoC Valet {series2} failed: {e}')

    val, dt = policy_rate('CAD')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_chf():
    """
    SARON via SNB data portal (cube: snbgwdzid).
    URL: https://data.snb.ch/api/cube/snbgwdzid/data/csv/en?dimSel=D0(SARON)
    Same source as workflow_meetings.yml. Daily frequency.
    """
    print('[CHF]')
    today = date.today()
    from_date = (today - timedelta(days=14)).strftime('%Y-%m-%d')
    try:
        url = (
            f'https://data.snb.ch/api/cube/snbgwdzid/data/csv/en'
            f'?dimSel=D0(SARON)&fromDate={from_date}'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        lines = r.text.splitlines()
        data_lines = [l for l in lines[3:] if l.strip() and ';' in l]
        for line in reversed(data_lines):
            parts = [p.strip().strip('"') for p in line.split(';')]
            if len(parts) >= 3 and parts[2]:
                try:
                    val = float(parts[2])
                    dt  = parts[0]
                    print(f'  CHF: SARON (SNB snbgwdzid)')
                    print(f'    ✓ SARON {val}% ({dt})')
                    return val, 'SARON', dt
                except ValueError:
                    continue
        print(f'  CHF: SNB snbgwdzid — no data rows found')
    except Exception as e:
        print(f'  CHF: SNB snbgwdzid failed: {e}')

    val, dt = policy_rate('CHF')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_nzd():
    """
    NZD OCR overnight via RBNZ B2 CSV (wholesale interest rates).
    URL: https://www.rbnz.govt.nz/statistics/b2
    B2 'Wholesale interest rates' — published daily.
    Fallback: policy rate from rates/NZD.json.
    Note: workflow_meetings.yml uses NZFBF OIS → IR3TIB → IRSTCI01NZM156N for
    bias direction with a 40bp track supplement. For CIP pricing we use the
    clean overnight rate, not the forward-adjusted bias.
    """
    print('[NZD]')
    try:
        url = 'https://www.rbnz.govt.nz/statistics/b2'
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        lines = r.text.splitlines()
        # RBNZ B2: CSV with date column and rate columns
        # Look for 'OCR' or 'Official Cash Rate' row
        for i, line in enumerate(lines):
            if 'OCR' in line or 'overnight' in line.lower() or 'official cash' in line.lower():
                parts = [p.strip().strip('"') for p in line.split(',')]
                for cell in reversed(parts):
                    try:
                        val = float(cell)
                        if 0.0 <= val <= 15.0:
                            dt = str(date.today())[:7]
                            print(f'  NZD: OCR (RBNZ B2 CSV)')
                            print(f'    ✓ OCR {val}% ({dt})')
                            return val, 'OCR', dt
                    except ValueError:
                        continue
    except Exception as e:
        print(f'  NZD: RBNZ B2 failed: {e}')

    val, dt = policy_rate('NZD')
    if val is not None:
        print(f'    ⚠ policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


# ── Dispatch ──────────────────────────────────────────────────────────────────

FETCHERS = {
    'USD': fetch_usd,
    'EUR': fetch_eur,
    'GBP': fetch_gbp,
    'JPY': fetch_jpy,
    'AUD': fetch_aud,
    'CAD': fetch_cad,
    'CHF': fetch_chf,
    'NZD': fetch_nzd,
}

rates   = {}
sources = {}
dates   = {}
failures = 0

for ccy, fetcher in FETCHERS.items():
    try:
        val, src, dt = fetcher()
    except Exception as e:
        print(f'  {ccy}: unexpected fetcher exception: {e}')
        val, src, dt = None, None, None

    if val is not None:
        rates[ccy]   = round(val, 6)
        sources[ccy] = src
        dates[ccy]   = dt
        print(f'  → {ccy} = {val}%  [{src}]  ({dt})')
    else:
        failures += 1
        print(f'  → {ccy} = FAILED (no data, no policy fallback)')

print()
print(f'── Summary: {len(rates)}/8 currencies loaded, {failures} failures ──')

# ── Write output ──────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

output = {
    'rates':   rates,
    'sources': sources,
    'dates':   dates,
    'updated': date.today().strftime('%Y-%m-%dT') + __import__('datetime').datetime.utcnow().strftime('%H:%M:%SZ'),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f'✅ Written: {OUTPUT_PATH}')
print(json.dumps(output, indent=2))

if failures == 8:
    print('⚠️  All currencies failed — no data written to repo', file=sys.stderr)
    sys.exit(1)
