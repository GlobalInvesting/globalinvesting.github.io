"""
fetch_ois_rates.py  —  OIS / Overnight Rates  G8  (v3.0)
=========================================================
Fetches overnight/OIS benchmark rates for all G8 currencies and writes
ois-rates/rates.json to the public site repo.

RATE → BENCHMARK MAPPING
  USD  SOFR   → FRED API (SOFR)                          daily
  EUR  €STR   → ECB Data Portal SDMX-JSON                daily
  GBP  SONIA  → FRED API (IUDSOIA)                       daily
  JPY  TONA   → SNB zimoma cube                          monthly
  AUD  AONIA  → FRED API (IRSTCI01AUM156N, staleness-gd) monthly
  CAD  CORRA  → BoC Valet API (V122514)                  daily
  CHF  SARON  → SNB data portal (snbgwdzid)              daily
  NZD  OCR    → FRED API (IRSTCI01NZM156N, staleness-gd) monthly

SOURCE DESIGN vs workflow_meetings.yml
  workflow_meetings.yml uses FRED API for: SOFR, DFF, IUDSOIA, ECBDFR,
    IRSTCI01AUM156N, IR3TIB01NZM156N, IRSTCI01NZM156N.
  This script uses FRED API for: SOFR (USD), IUDSOIA (GBP), IRSTCI01AUM156N (AUD),
    IRSTCI01NZM156N (NZD). EUR/JPY/CAD/CHF use direct CB APIs that work without keys.

  WHY FRED API (not direct CB APIs) for USD/GBP/AUD/NZD:
    - NY Fed /api/rates/* returned 403 on GH Actions (confirmed v2.0 run)
    - BOE boeapps CSV returned 403 on GH Actions (confirmed v2.0 run)
    - RBA CSV/API returned 403 on GH Actions (confirmed v2.0 run)
    - RBNZ CSV/page returned 403 on GH Actions (confirmed v2.0 run)
    - FRED API (api.stlouisfed.org, key-auth) confirmed working on GH Actions
      (workflow_meetings.yml uses it every Monday without failures)

  FRED_API_KEY: already present in engine repo (used by workflow_meetings.yml).
  The update-ois-rates.yml workflow must pass it as env: FRED_API_KEY.

  NO PURPOSE OVERLAP with workflow_meetings.yml:
    meetings-data/meetings.json  CB Meetings panel (bias hold/cut/hike + probs)
    ois-rates/rates.json         CIP forward pricing via _resolveRate()

  STALENESS GUARDS (AUD/NZD):
    FRED OECD monthly series lag 4-6 weeks. Guard: if abs(ois - policy) > 50bp,
    the series predates the last CB move -> fall back to policy rate.
    Same logic as workflow_meetings.yml v7.45.0+.

CONSUMED BY
  dashboard2.js -> loadOISRatesCache() -> _resolveRate() -> computeCIPForward()
  Policy rates (rates/*.json) unchanged — CB Rates panel, carry, regime scoring.
"""

import json
import os
import sys
import datetime as _dt
from datetime import date, timedelta

import requests

SITE_DIR     = os.environ.get('SITE_DIR', '.')
OUTPUT_PATH  = os.path.join(SITE_DIR, 'ois-rates', 'rates.json')
RATES_DIR    = os.path.join(SITE_DIR, 'rates')
HEADERS      = {'User-Agent': 'Mozilla/5.0 (compatible; GlobalInvestingBot/2.0)'}
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
FRED_BASE    = 'https://api.stlouisfed.org/fred/series/observations'

print('=' * 50)
print('OIS / OVERNIGHT RATES  --  G8')
print('=' * 50)

# ── Helpers ───────────────────────────────────────────────────────────────────

def policy_rate(ccy):
    """Read current CB policy rate from rates/{CCY}.json — last-resort fallback."""
    path = os.path.join(RATES_DIR, f'{ccy}.json')
    try:
        with open(path) as f:
            d = json.load(f)
        obs = d.get('observations', [])
        raw = obs[0]['value'] if obs else d.get('rate') or d.get('value')
        dt  = obs[0].get('date', str(date.today())[:7]) if obs else str(date.today())[:7]
        if raw is not None and str(raw) not in ('.', ''):
            return float(raw), dt
    except Exception:
        pass
    return None, None


def fred_latest(series_id, label=None):
    """
    Fetch most recent observation from FRED API (api.stlouisfed.org, key-auth).
    Confirmed working on GH Actions (workflow_meetings.yml uses same key every Monday).
    Returns (value: float, date: str) or (None, None).
    """
    lbl = label or series_id
    if not FRED_API_KEY:
        print(f'    [FRED] No API key -- skipping {lbl}')
        return None, None
    try:
        r = requests.get(FRED_BASE, params={
            'series_id':  series_id,
            'api_key':    FRED_API_KEY,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      '5',
        }, headers=HEADERS, timeout=15)
        r.raise_for_status()
        for o in r.json().get('observations', []):
            if o.get('value') not in ('.', '', None):
                return float(o['value']), o.get('date', str(date.today()))
    except Exception as e:
        print(f'    FRED {lbl}: {e}')
    return None, None

# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_usd():
    """
    SOFR via FRED API (series SOFR).
    Same key and series as workflow_meetings.yml fetch_bias_usd().
    Fallback: EFFR (DFF).
    """
    print('[USD]')
    val, dt = fred_latest('SOFR', 'SOFR')
    if val is not None:
        print(f'  USD: SOFR (FRED API)')
        print(f'    OK SOFR {val}% ({dt})')
        return val, 'SOFR', dt

    val, dt = fred_latest('DFF', 'EFFR/DFF')
    if val is not None:
        print(f'  USD: fallback EFFR (FRED API DFF)')
        print(f'    OK EFFR {val}% ({dt})')
        return val, 'EFFR', dt

    val, dt = policy_rate('USD')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_eur():
    """
    EUR/STR via ECB Data Portal SDMX-JSON (no auth).
    Confirmed working on GH Actions (v2.0 run: 1.929% 2026-05-12).
    Fallback: ECB SDW REST alternative endpoint.
    """
    print('[EUR]')
    try:
        url = (
            'https://data-api.ecb.europa.eu/service/data/EST/'
            'B.EU000A2X2A25.WT?lastNObservations=3&format=jsondata'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        body = r.json()
        ds = body['dataSets'][0]['series']
        series_key = list(ds.keys())[0]
        obs = ds[series_key]['observations']
        try:
            time_periods = body['structure']['dimensions']['observation'][0]['values']
        except Exception:
            time_periods = []
        for idx in sorted(obs.keys(), key=int, reverse=True):
            v = obs[idx][0]
            if v is not None:
                dt  = time_periods[int(idx)]['id'] if int(idx) < len(time_periods) else str(date.today())
                val = float(v)
                print(f'  EUR: ESTR (ECB Data Portal EST.B.EU000A2X2A25.WT)')
                print(f'    OK ESTR {val}% ({dt})')
                return val, 'ESTR', dt
    except Exception as e:
        print(f'  EUR: ECB Data Portal failed: {e}')

    try:
        url2 = 'https://sdw-wsrest.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?lastNObservations=1'
        r2 = requests.get(url2, headers={**HEADERS, 'Accept': 'application/json'}, timeout=15)
        r2.raise_for_status()
        ds2 = r2.json()['dataSets'][0]['series']
        sk2 = list(ds2.keys())[0]
        for idx2 in sorted(ds2[sk2]['observations'].keys(), key=int, reverse=True):
            v2 = ds2[sk2]['observations'][idx2][0]
            if v2 is not None:
                val2 = float(v2)
                print(f'  EUR: fallback ECB SDW REST OK {val2}%')
                return val2, 'ESTR', str(date.today())
    except Exception as e:
        print(f'  EUR: ECB SDW fallback failed: {e}')

    val, dt = policy_rate('EUR')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_gbp():
    """
    SONIA via FRED API (series IUDSOIA).
    workflow_meetings.yml falls back to FRED IUDSOIA for SONIA spot — same key/series.
    For CIP pricing, SONIA spot is the correct rate (not 1M OIS forward which meetings uses).
    """
    print('[GBP]')
    val, dt = fred_latest('IUDSOIA', 'SONIA')
    if val is not None:
        print(f'  GBP: SONIA (FRED API IUDSOIA)')
        print(f'    OK SONIA {val}% ({dt})')
        return val, 'SONIA', dt

    val, dt = policy_rate('GBP')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_jpy():
    """
    TONA via SNB zimoma (data.snb.ch/api/cube/zimoma).
    Confirmed working on GH Actions (v2.0 run: 0.727% 2026-04).
    Same source as workflow_meetings.yml. Monthly lag acceptable for JPY.
    Raw TONA value only -- bias-supplement logic belongs in meetings workflow only.
    Fallback: FRED OECD overnight Japan (IRSTCI01JPM156N) with staleness guard.
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
        url = f'https://data.snb.ch/api/cube/zimoma/data/csv/en?dimSel=D0(TONA)&fromDate={from_date}'
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
                    print(f'    OK TONA {val}% ({dt})')
                    return val, 'TONA', dt
                except ValueError:
                    continue
        print(f'  JPY: SNB zimoma -- no data rows found')
    except Exception as e:
        print(f'  JPY: SNB zimoma failed: {e}')

    val, dt = fred_latest('IRSTCI01JPM156N', 'OECD-JPY')
    if val is not None:
        pol, _ = policy_rate('JPY')
        if pol is not None and abs(val - pol) > 0.50:
            print(f'  JPY: FRED OECD={val}% vs policy={pol}% -- stale (>50bp), skipping')
        else:
            print(f'  JPY: fallback FRED OECD overnight {val}% ({dt})')
            return val, 'OECD-overnight', dt

    val, dt = policy_rate('JPY')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_aud():
    """
    AONIA via FRED API (series IRSTCI01AUM156N -- OECD AUD overnight rate).
    Staleness guard: abs(ois - policy) > 50bp OR ois < policy - 10bp -> policy fallback.
    Same series and guard as workflow_meetings.yml v7.45.0+.
    ASX IB futures and RBA API both return 403 on GH Actions (confirmed v2.0 run).
    """
    print('[AUD]')
    rba_target, rba_dt = policy_rate('AUD')

    val, dt = fred_latest('IRSTCI01AUM156N', 'AONIA/OECD')
    if val is not None:
        if rba_target is not None:
            diff = val - rba_target
            stale = abs(diff) > 0.50 or diff < -0.10
            if stale:
                print(f'  AUD: FRED OECD={val}% vs RBA={rba_target}% -- stale (diff={diff:.2f}%)')
                print(f'    policy fallback {rba_target:.4f}%')
                return rba_target, 'policy-fallback', rba_dt
        print(f'  AUD: AONIA (FRED API IRSTCI01AUM156N)')
        print(f'    OK AONIA {val}% ({dt})')
        return val, 'AONIA', dt

    if rba_target is not None:
        print(f'    policy fallback {rba_target:.4f}%')
        return rba_target, 'policy-fallback', rba_dt
    return None, None, None


def fetch_cad():
    """
    CORRA via Bank of Canada Valet API (series V122514).
    Confirmed working on GH Actions (v2.0 run: 2.2492% 2026-02-01).
    V122514 = canonical CORRA series (aligned with workflow_meetings.yml).
    V122530 retained as fallback.
    """
    print('[CAD]')
    for series in ('V122514', 'V122530'):
        try:
            r = requests.get(
                f'https://www.bankofcanada.ca/valet/observations/{series}/json',
                params={'recent': '3'}, headers=HEADERS, timeout=15
            )
            r.raise_for_status()
            obs = r.json().get('observations', [])
            if obs:
                last = obs[-1]
                raw  = last.get(series, {}).get('v')
                if raw is not None:
                    val = float(raw)
                    dt  = last.get('d', str(date.today()))
                    print(f'  CAD: CORRA (BoC Valet {series})')
                    print(f'    OK CORRA {val}% ({dt})')
                    return val, 'CORRA', dt
        except Exception as e:
            print(f'  CAD: BoC Valet {series} failed: {e}')

    val, dt = policy_rate('CAD')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_chf():
    """
    SARON via SNB data portal (cube snbgwdzid).
    Confirmed working on GH Actions (v2.0 run: -0.04% 2026-05-08).
    Same source as workflow_meetings.yml.
    Fallback: FRED OECD CHF overnight (IRSTCI01CHM156N) with staleness guard.
    """
    print('[CHF]')
    today = date.today()
    from_date = (today - timedelta(days=14)).strftime('%Y-%m-%d')
    try:
        url = f'https://data.snb.ch/api/cube/snbgwdzid/data/csv/en?dimSel=D0(SARON)&fromDate={from_date}'
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
                    print(f'    OK SARON {val}% ({dt})')
                    return val, 'SARON', dt
                except ValueError:
                    continue
        print(f'  CHF: SNB snbgwdzid -- no data rows found')
    except Exception as e:
        print(f'  CHF: SNB snbgwdzid failed: {e}')

    val, dt = fred_latest('IRSTCI01CHM156N', 'OECD-CHF')
    if val is not None:
        pol, _ = policy_rate('CHF')
        if pol is not None and abs(val - pol) > 0.50:
            print(f'  CHF: FRED OECD={val}% vs policy={pol}% -- stale (>50bp), skipping')
        else:
            print(f'  CHF: fallback FRED OECD overnight {val}% ({dt})')
            return val, 'OECD-overnight', dt

    val, dt = policy_rate('CHF')
    if val is not None:
        print(f'    policy fallback {val:.4f}%')
        return val, 'policy-fallback', dt
    return None, None, None


def fetch_nzd():
    """
    NZD overnight via FRED API (series IRSTCI01NZM156N -- OECD NZD overnight rate).
    Staleness guard: abs(ois - policy) > 50bp -> fall back to policy rate.
    RBNZ CSV and NZFBF both return 403 on GH Actions (confirmed v2.0 run).
    workflow_meetings.yml uses IRSTCI01NZM156N as tertiary source -- same series here.
    Note: meetings applies credit-adj + track supplement for bias direction; this
    script uses the raw overnight rate as the CIP discount rate (no adjustment needed).
    """
    print('[NZD]')
    rbnz_ocr, rbnz_dt = policy_rate('NZD')

    val, dt = fred_latest('IRSTCI01NZM156N', 'NZD-OECD')
    if val is not None:
        if rbnz_ocr is not None and abs(val - rbnz_ocr) > 0.50:
            print(f'  NZD: FRED OECD={val}% vs RBNZ={rbnz_ocr}% -- stale (>{abs(val-rbnz_ocr):.2f}%)')
            print(f'    policy fallback {rbnz_ocr:.4f}%')
            return rbnz_ocr, 'policy-fallback', rbnz_dt
        print(f'  NZD: OCR overnight (FRED API IRSTCI01NZM156N)')
        print(f'    OK NZD overnight {val}% ({dt})')
        return val, 'OCR-overnight', dt

    if rbnz_ocr is not None:
        print(f'    policy fallback {rbnz_ocr:.4f}%')
        return rbnz_ocr, 'policy-fallback', rbnz_dt
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

rates    = {}
sources  = {}
dates    = {}
failures = 0

for ccy, fetcher in FETCHERS.items():
    try:
        val, src, dt = fetcher()
    except Exception as e:
        print(f'  {ccy}: unexpected exception: {e}')
        val, src, dt = None, None, None

    if val is not None:
        rates[ccy]   = round(val, 6)
        sources[ccy] = src
        dates[ccy]   = dt
        print(f'  -> {ccy} = {val}%  [{src}]  ({dt})')
    else:
        failures += 1
        print(f'  -> {ccy} = FAILED (no data, no policy fallback)')

print()
print(f'-- Summary: {len(rates)}/8 currencies loaded, {failures} failures --')

# ── Write output ──────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

output = {
    'rates':   rates,
    'sources': sources,
    'dates':   dates,
    'updated': _dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f'OK Written: {OUTPUT_PATH}')
print(json.dumps(output, indent=2))

if failures == 8:
    print('WARNING: All currencies failed -- check FRED_API_KEY secret', file=sys.stderr)
    sys.exit(1)
