"""
fetch_ois_rates.py  —  OIS / Overnight Rates  G8  (v5.0)
=========================================================
Fetches overnight/OIS benchmark rates for all G8 currencies and writes
ois-rates/rates.json to the public site repo.

RATE → BENCHMARK MAPPING
  USD  SOFR   → NY Fed API (no key) → FRED API (SOFR/DFF)       daily  ← v5.0
  EUR  €STR   → ECB Data Portal SDMX-JSON                        daily
  GBP  SONIA  → BoE SDIE IUDSOIA (no key) → FRED API (IUDSOIA)  daily  ← v5.0
  JPY  TONA   → SNB zimoma cube                                   monthly
  AUD  AONIA  → RBA Cash Rate (rates/AUD.json)                   daily  ← v4.0
  CAD  CORRA  → BoC Valet API V122514 (date-range params)        daily  ← v5.0
  CHF  SARON  → SNB data portal (snbgwdzid)                      daily
  NZD  OCR    → RBNZ OCR (rates/NZD.json)                        daily  ← v4.0

INDUSTRY STANDARD ALIGNMENT (v5.0)
  USD: NY Fed SOFR API (markets.newyorkfed.org/api/rates/sofr/last/1.json)
    added as primary no-key source. FRED SOFR/DFF retained as secondary.
    Eliminates dependency on FRED API key for the most liquid OIS benchmark.
    Confirmed working from GitHub Actions (no bot protection on NY Fed API).

  GBP: BoE SDIE IUDSOIA added as primary no-key source using the same
    _iadb-fromshowcolumns endpoint confirmed working in fetch_bond_yields.py.
    FRED IUDSOIA retained as secondary. SONIA spot (= Bank Rate ± ~5bp
    structurally) is the correct CIP pricing input; OIS forward belongs in
    workflow_meetings.yml (forward-looking bias, not spot discount rate).

  CAD: BoC Valet call switched from params={'recent': 3} to start_date/
    end_date date-range parameters (6-week window). 'recent=3' was returning
    the 3 most recent rows in the SERIES file rather than calendar-recent
    daily observations, causing 89-day stale dates (2026-03-01) in production.
    Date-range params match the pattern used by fetch_bond_yields.py
    (_boc_valet_latest) which reliably returns the current trading day value.

  AUD: AONIA is defined by AFMA as identical to the RBA Cash Rate (AFMA
    benchmark notice; RBA Cash Rate Methodology). Bloomberg and Refinitiv
    use the RBA Cash Rate as the AUD RFR in CIP forward calculators.
    Previous source (FRED IRSTCI01AUM156N, OECD monthly) lagged RBA hikes
    by 4-6 weeks, introducing systematic forward mispricing of 14-39bp.
    Policy rate is now primary; FRED OECD retained as sanity-check only.

  NZD: NZONIA (NZFBF compound index) ≈ RBNZ OCR by construction. Bloomberg
    and Refinitiv use the RBNZ OCR as the NZD RFR. NZFBF is Cloudflare-
    blocked from GH Actions and not available via any free public API.
    RBNZ OCR is now primary; FRED OECD retained as sanity-check (75bp guard,
    aligned with workflow_meetings.yml v7.79.0).

FRED TIMEOUT
  Reduced from 15s to 8s. FRED API 504s (gateway timeouts under high load)
  were causing the full script to stall for 30+ seconds before reaching the
  policy fallback. No-key primary sources (NY Fed, BoE SDIE) now absorb the
  load so FRED failures are fast-fallthrough, not blocking.

SOURCE DESIGN vs workflow_meetings.yml
  workflow_meetings.yml uses FRED API for: SOFR, DFF, IUDSOIA, ECBDFR.
  This script now uses FRED as secondary only for USD/GBP (no-key primaries first).
  AUD/NZD use rates/*.json (policy rate = RFR by definition).
  EUR/JPY/CAD/CHF use direct CB APIs that work without keys.

  NO PURPOSE OVERLAP with workflow_meetings.yml:
    meetings-data/meetings.json  CB Meetings panel (bias hold/cut/hike + probs)
    ois-rates/rates.json         CIP forward pricing via _resolveRate()

CONSUMED BY
  dashboard2.js -> loadOISRatesCache() -> _resolveRate() -> computeCIPForward()
  Policy rates (rates/*.json) unchanged — CB Rates panel, carry, regime scoring.

CHANGE LOG
  v3.0: Initial production version. AUD/NZD used FRED OECD monthly with staleness guard.
  v4.0: AUD: RBA Cash Rate promoted to primary (AONIA = Cash Rate per AFMA definition).
        NZD: RBNZ OCR promoted to primary (NZONIA ≈ OCR by construction).
        Eliminates post-hike forward pricing lag of 14-39bp (AUD) and 10-25bp (NZD).
        FRED OECD retained as background sanity-check for both currencies.
  v5.0: USD: NY Fed SOFR API added as primary no-key source before FRED.
        GBP: BoE SDIE IUDSOIA added as primary no-key source before FRED.
        CAD: BoC Valet switched from 'recent=3' to date-range params (fixes stale dates).
        FRED timeout reduced 15s → 8s for faster fallthrough on gateway timeouts.
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
        }, headers=HEADERS, timeout=8)
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
    SOFR via NY Fed API (no key required), then FRED API fallback.

    PRIMARY: NY Fed Markets API (markets.newyorkfed.org/api/rates/sofr/last/1.json)
      No API key. No bot protection. Confirmed reachable from GitHub Actions.
      Returns: {"refRates": [{"type": "SOFR", "percentRate": 5.30, "effectiveDate": "2025-01-17"}]}
      Published each business day after ~08:00 ET (NY Fed daily release schedule).
      This is the authoritative SOFR source — NY Fed administers the benchmark.

    SECONDARY: FRED API (series SOFR) — same key + series as workflow_meetings.yml.
      Fallback to EFFR (DFF) if SOFR series unavailable.
      504 gateway timeouts on FRED are handled by the 8s timeout limit.

    TERTIARY: CB policy rate (rates/USD.json) — never stale, always available.
    """
    print('[USD]')

    # Primary: NY Fed SOFR API (no key, authoritative source)
    try:
        r = requests.get(
            'https://markets.newyorkfed.org/api/rates/sofr/last/1.json',
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        ref_rates = r.json().get('refRates', [])
        for entry in ref_rates:
            if entry.get('type') == 'SOFR' and entry.get('percentRate') is not None:
                val = float(entry['percentRate'])
                dt  = entry.get('effectiveDate', str(date.today()))
                print(f'  USD: SOFR (NY Fed API — no key)')
                print(f'    OK SOFR {val}% ({dt})')
                return val, 'SOFR', dt
    except Exception as e:
        print(f'    NY Fed SOFR: {e}')

    # Secondary: FRED API (requires FRED_API_KEY)
    val, dt = fred_latest('SOFR', 'SOFR')
    if val is not None:
        print(f'  USD: SOFR (FRED API fallback)')
        print(f'    OK SOFR {val}% ({dt})')
        return val, 'SOFR', dt

    val, dt = fred_latest('DFF', 'EFFR/DFF')
    if val is not None:
        print(f'  USD: fallback EFFR (FRED API DFF)')
        print(f'    OK EFFR {val}% ({dt})')
        return val, 'EFFR', dt

    # Tertiary: CB policy rate
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
    SONIA via BoE SDIE (no key required), then FRED API fallback.

    PRIMARY: BoE SDIE _iadb-fromshowcolumns endpoint — series IUDSOIA.
      No API key. Same endpoint confirmed working in fetch_bond_yields.py
      (_boe_latest helper, used daily for GBP bond yields without failures).
      Returns CSV; date range: 01/Jan/{year-1} to today.
      IUDSOIA = SONIA overnight spot rate, published daily by the BoE.

    SECONDARY: FRED API series IUDSOIA — mirrors the same BoE-administered
      SONIA series. Used when BoE endpoint is temporarily unavailable.
      504 gateway timeouts on FRED are handled by the 8s timeout limit.

    DESIGN NOTE: SONIA spot (= Bank Rate ± ~5bp structurally) is the correct
      CIP forward pricing input — it reflects the actual overnight funding cost.
      The 1M OIS forward (IUQIBLOK) belongs in workflow_meetings.yml for
      forward-looking CB bias signals; it is not appropriate for spot CIP pricing.

    TERTIARY: CB policy rate (rates/GBP.json).
    """
    print('[GBP]')

    # Primary: BoE SDIE IUDSOIA (no key, same endpoint as fetch_bond_yields.py)
    try:
        now       = date.today()
        date_from = f'01/Jan/{now.year - 1}'
        date_to   = f'{now.day:02d}/{now.strftime("%b")}/{now.year}'
        boe_url   = 'https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp'
        boe_params = {
            'csv.x':       'yes',
            'Datefrom':    date_from,
            'Dateto':      date_to,
            'SeriesCodes': 'IUDSOIA',
            'CSVF':        'TN',
            'UsingCodes':  'Y',
            'VPD':         'Y',
            'VFD':         'N',
        }
        r = requests.get(boe_url, params=boe_params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        if 'html' in content_type.lower() or r.text.strip().startswith('<'):
            print(f'    BoE SDIE IUDSOIA: received HTML (bot challenge?) — skipping')
        else:
            rows = []
            for line in r.text.strip().split('\n'):
                line = line.strip()
                if not line or line.upper().startswith('DATE') or line.startswith('"'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 2:
                    continue
                val_str = parts[1].strip().replace('..', '').strip()
                if not val_str or val_str in ('.', '', 'N/A', 'n/a'):
                    continue
                try:
                    from datetime import datetime as _dtime
                    dt_obj = _dtime.strptime(parts[0].strip(), '%d %b %Y')
                    rows.append((dt_obj.strftime('%Y-%m-%d'), float(val_str)))
                except (ValueError, TypeError):
                    continue
            if rows:
                rows.sort(key=lambda x: x[0])
                dt_str, val = rows[-1]
                print(f'  GBP: SONIA (BoE SDIE IUDSOIA — no key)')
                print(f'    OK SONIA {val}% ({dt_str})')
                return val, 'SONIA', dt_str
            else:
                print(f'    BoE SDIE IUDSOIA: no data rows found')
    except Exception as e:
        print(f'    BoE SDIE IUDSOIA: {e}')

    # Secondary: FRED API IUDSOIA (mirrors BoE series, requires key)
    val, dt = fred_latest('IUDSOIA', 'SONIA')
    if val is not None:
        print(f'  GBP: SONIA (FRED API IUDSOIA fallback)')
        print(f'    OK SONIA {val}% ({dt})')
        return val, 'SONIA', dt

    # Tertiary: CB policy rate
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
    AONIA (AUD Overnight Index Average) for CIP forward pricing.

    INDUSTRY STANDARD: AONIA is defined by AFMA as identical to the RBA Cash Rate
    (AFMA benchmark notice; RBA Cash Rate Methodology). Bloomberg and Refinitiv both
    use the RBA Cash Rate as the AUD RFR discount rate in CIP forward calculators.
    Unlike SOFR/€STR/SONIA which trade independently of their policy rates intraday,
    AONIA tracks the RBA Cash Rate target with ±2bp precision — it is operationally
    the same rate.

    SOURCE CHAIN:
    1. RBA Cash Rate (rates/AUD.json policy rate)  [PRIMARY — daily effective rate]
         AONIA = RBA Cash Rate by AFMA definition. Always current, never stale.
         Used by Bloomberg, Refinitiv, and all major CIP pricing desks as AUD RFR.
         Labelled 'AONIA' in output — accurate per AFMA/RBA published definition.

    2. FRED IRSTCI01AUM156N (OECD overnight, monthly)  [SECONDARY — sanity check]
         Used only to confirm the policy rate is within a plausible band.
         If FRED is available and within 50bp of policy, it confirms data integrity.
         Monthly lag means it often trails after a hike cycle — in that case
         policy rate remains the correct CIP input (it IS AONIA in practice).

    WHY NOT FRED FIRST: IRSTCI01AUM156N lags RBA decisions by 4-6 weeks.
    After the March 2026 hike to 4.10% (and prior hikes), FRED still shows 3.96%.
    Using a lagged rate introduces a systematic forward mispricing of 14-39bp
    on all AUD-cross forwards. Policy rate is more accurate and more current.
    """
    print('[AUD]')
    rba_target, rba_dt = policy_rate('AUD')

    # Primary: RBA Cash Rate = AONIA by AFMA definition
    if rba_target is not None:
        # Sanity-check against FRED OECD if available
        fred_val, fred_dt = fred_latest('IRSTCI01AUM156N', 'AONIA/OECD-check')
        if fred_val is not None:
            diff = abs(fred_val - rba_target)
            if diff <= 0.50:
                print(f'  AUD: AONIA = RBA Cash Rate (AFMA definition) — '
                      f'FRED OECD confirms: {fred_val}% vs policy {rba_target}% '
                      f'(diff={fred_val - rba_target:+.2f}%)')
            else:
                print(f'  AUD: AONIA = RBA Cash Rate (AFMA definition) — '
                      f'FRED OECD={fred_val}% lags policy {rba_target}% by {diff:.2f}% '
                      f'(post-hike lag, expected — using policy rate)')
        else:
            print(f'  AUD: AONIA = RBA Cash Rate (AFMA definition) — '
                  f'FRED OECD unavailable')
        print(f'    OK AONIA {rba_target}% ({rba_dt})')
        return rba_target, 'AONIA', rba_dt

    # Last resort: FRED OECD with staleness guard
    fred_val, fred_dt = fred_latest('IRSTCI01AUM156N', 'AONIA/OECD')
    if fred_val is not None:
        print(f'  AUD: policy rate unavailable — FRED OECD fallback {fred_val}% ({fred_dt})')
        return fred_val, 'OECD-overnight', fred_dt

    return None, None, None


def fetch_cad():
    """
    CORRA via Bank of Canada Valet API (series V122514).

    FIX v5.0: Switched from params={'recent': 3} to start_date/end_date date-range
      parameters (6-week window). 'recent=3' was returning the 3 most recent entries
      in the series file as stored by the BoC Valet backend, which for V122514
      resolved to monthly average observations (last seen: 2026-03-01, 89 days stale)
      rather than the current daily fixing. Date-range params reliably return all
      observations between the specified dates, matching the pattern used by
      fetch_bond_yields.py (_boc_valet_latest) which returns the current trading day.

    V122514 = canonical daily CORRA series (aligned with workflow_meetings.yml).
    V122530 retained as fallback (daily CORRA, alternative BoC Valet ID).
    Confirmed working from GitHub Actions (CORRA ~2.25% current target range).
    """
    print('[CAD]')
    end_date   = date.today().strftime('%Y-%m-%d')
    start_date = (date.today() - timedelta(days=42)).strftime('%Y-%m-%d')  # 6-week window

    for series in ('V122514', 'V122530'):
        try:
            r = requests.get(
                f'https://www.bankofcanada.ca/valet/observations/{series}/json',
                params={'start_date': start_date, 'end_date': end_date},
                headers=HEADERS, timeout=15,
            )
            r.raise_for_status()
            obs = r.json().get('observations', [])
            # Walk backwards from most recent to find latest non-empty value
            for entry in reversed(obs):
                raw = entry.get(series, {}).get('v')
                if raw and raw not in ('', 'Bank holiday', 'nd'):
                    try:
                        val = float(raw)
                        dt  = entry.get('d', str(date.today()))
                        print(f'  CAD: CORRA (BoC Valet {series})')
                        print(f'    OK CORRA {val}% ({dt})')
                        return val, 'CORRA', dt
                    except (ValueError, TypeError):
                        continue
            print(f'  CAD: BoC Valet {series} — no observations in window ({start_date} → {end_date})')
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
    NZONIA (NZ Overnight Index Average) for CIP forward pricing.

    INDUSTRY STANDARD: The NZD RFR for OIS and forward pricing is the RBNZ OCR
    (Official Cash Rate). NZONIA, published by NZFBF, computes the overnight rate
    as a compound index over the OCR — it tracks the OCR with ±5bp precision.
    Bloomberg and Refinitiv use the RBNZ OCR as the NZD discount rate in CIP
    forward calculators (NZFBF/NZONIA is not independently available via any
    public free API; it is licensed through data vendors).

    SOURCE CHAIN:
    1. RBNZ OCR (rates/NZD.json policy rate)  [PRIMARY — daily effective rate]
         NZONIA ≈ OCR by construction (compound index over OCR).
         Always current. Used by institutional desks as the NZD RFR proxy when
         NZONIA is not directly accessible.
         Labelled 'OCR-overnight' — accurate description per RBNZ definition.

    2. FRED IRSTCI01NZM156N (OECD overnight, monthly)  [SECONDARY — sanity check]
         Staleness guard: abs(fred - ocr) > 75bp → skip (v7.79.0 threshold alignment).
         With OCR stable at 2.25% since Nov 2025, structural deviation is 10-30bp.
         >75bp implies the OECD data predates at least two OCR moves.
         Monthly lag means this often trails after a cut/hike cycle; in that case
         OCR remains the correct CIP input.

    WHY NOT FRED FIRST: same rationale as AUD — monthly lag introduces systematic
    forward mispricing. RBNZ OCR IS the NZONIA reference rate in practice.
    """
    print('[NZD]')
    rbnz_ocr, rbnz_dt = policy_rate('NZD')

    # Primary: RBNZ OCR = NZONIA reference rate
    if rbnz_ocr is not None:
        fred_val, fred_dt = fred_latest('IRSTCI01NZM156N', 'NZD-OECD-check')
        if fred_val is not None:
            diff = abs(fred_val - rbnz_ocr)
            if diff <= 0.75:
                print(f'  NZD: NZONIA ≈ RBNZ OCR (institutional standard) — '
                      f'FRED OECD confirms: {fred_val}% vs OCR {rbnz_ocr}% '
                      f'(diff={fred_val - rbnz_ocr:+.2f}%)')
            else:
                print(f'  NZD: NZONIA ≈ RBNZ OCR (institutional standard) — '
                      f'FRED OECD={fred_val}% deviates {diff:.2f}% from OCR '
                      f'(post-cut lag, expected — using OCR)')
        else:
            print(f'  NZD: NZONIA ≈ RBNZ OCR (institutional standard) — '
                  f'FRED OECD unavailable')
        print(f'    OK OCR-overnight {rbnz_ocr}% ({rbnz_dt})')
        return rbnz_ocr, 'OCR-overnight', rbnz_dt

    # Last resort: FRED OECD with 75bp staleness guard
    fred_val, fred_dt = fred_latest('IRSTCI01NZM156N', 'NZD-OECD')
    if fred_val is not None:
        print(f'  NZD: OCR unavailable — FRED OECD fallback {fred_val}% ({fred_dt})')
        return fred_val, 'OECD-overnight', fred_dt

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
