#!/usr/bin/env python3
"""
fetch_rates.py  v13.0 — Fuentes oficiales de bancos centrales
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Obtiene tasas de política monetaria para las 8 divisas G8,
priorizando fuentes directas de los bancos centrales para
eliminar el lag que tenía global-rates.com como fuente primaria.

Orden de fuentes por divisa:
  AUD  1. RBA CSV oficial  (rba.gov.au/statistics/tables)  — mismo día
  CAD  1. BoC Valet API    (bankofcanada.ca/valet)          — mismo día
  EUR  1. ECB Data Portal  (data-api.ecb.europa.eu)         — mismo día
  GBP  1. BoE Database CSV (bankofengland.co.uk/boeapps)    — mismo día
  USD  1. NY Fed EFFR API  (markets.newyorkfed.org)         — mismo día
  JPY  1. BoJ scraping     (boj.or.jp)                      — mismo día
  CHF  1. SNB scraping     (snb.ch)                         — mismo día
  NZD  1. RBNZ tabla B2    (rbnz.govt.nz/statistics)        — mismo día

  Todos → FRED API (si FRED_API_KEY disponible)             — fallback
  Todos → FRED CSV público                                  — fallback
  Todos → global-rates.com                                  — fallback
  Todos → archivo guardado                                  — último recurso

  Validación: Frankfurter ECB API (cross-check FX)

Salidas:
  rates/XX.json      historial acumulado de observaciones
  rates/health.json  estado de fiabilidad + timestamp del run
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import os
import csv
from io import StringIO
from datetime import date, datetime, timedelta
import time

# ── Configuración ──────────────────────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; globalinvesting-engine/13.0)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

# Series FRED — usadas como fallback
FRED_RATE_SERIES = {
    'USD': 'FEDFUNDS',
    'EUR': 'ECBDFR',
    'GBP': 'BOERUKM',
    'JPY': 'IRSTCB01JPM156N',
    'CAD': 'IRSTCB01CAM156N',
    'AUD': 'IRSTCB01AUM156N',
    'CHF': 'IRSTCB01CHM156N',
    'NZD': 'IRSTCB01NZM156N',
}

# Último fallback externo (lag potencial de horas/días)
GLOBAL_RATES_URLS = {
    'USD': 'central-bank-america/fed-interest-rate.aspx',
    'EUR': 'central-bank-europe/ecb-interest-rate.aspx',
    'GBP': 'central-bank-england/boe-interest-rate.aspx',
    'JPY': 'central-bank-japan/boj-interest-rate.aspx',
    'CHF': 'central-bank-switzerland/snb-interest-rate.aspx',
    'CAD': 'central-bank-canada/boc-interest-rate.aspx',
    'AUD': 'central-bank-australia/rba-interest-rate.aspx',
    'NZD': 'central-bank-new-zealand/rbnz-interest-rate.aspx',
}

MAX_POLICY_RATE_PP    = 15.0
MIN_POLICY_RATE_PP    = -2.0
MAX_CHANGE_VS_PREV_PP = 0.75  # G8: máximo movimiento en un run (~75bp emergencia)

# Fuentes con lag conocido — no bloquean la validación al corregir
STALE_SOURCES = (
    'global-rates.com',
    'FRED-CSV:ECBDFR',
    'FRED-CSV:BOERUKM',
    'FRED-CSV:IRSTCB01JPM156N',
    'FRED-CSV:IRSTCB01CAM156N',
    'FRED-CSV:IRSTCB01AUM156N',
    'FRED-CSV:IRSTCB01CHM156N',
    'FRED-CSV:IRSTCB01NZM156N',
)

# Fuentes consideradas "ok" (no se marcan como fallback en health.json)
OFFICIAL_SOURCES = {
    'RBA-HTML', 'RBA-CSV', 'BoC-Valet', 'ECB-SDMX', 'BoE-CSV',
    'NYFed-EFFR', 'FRED:FEDFUNDS', 'BoJ-API', 'BoJ-scraping',
    'SNB-scraping', 'RBNZ-CSV', 'BIS-CBPOL',
    'manual-override',
}

# ── Overrides manuales ─────────────────────────────────────────────────────────
# Usar solo si TODAS las fuentes primarias tienen lag y se conoce el dato oficial.
# Formato: 'XXX': ('tasa', 'YYYY-MM-DD')
# Eliminar la entrada en cuanto la fuente primaria se actualice.
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    # Use ONLY when ALL live sources have a known lag AND the official value is publicly confirmed.
    # Never add EUR here — unverified estimates cause more harm than a stale-but-labeled value.
    # Example (remove once primary source updates):
    # 'XXX': ('rate', 'YYYY-MM-DD'),  # Bank decision date — remove when primary source updates
}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def clean_rate(text):
    if not text:
        return None
    text = str(text).strip().replace('%', '').replace(',', '.')
    match = re.search(r'(-?\d+\.?\d*)', text)
    if match:
        try:
            val = float(match.group(1))
            if MIN_POLICY_RATE_PP <= val <= MAX_POLICY_RATE_PP:
                return match.group(1)
        except:
            pass
    return None


def load_existing_observations(currency):
    path = f'rates/{currency}.json'
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('observations', [])
    except Exception as e:
        print(f'    Warning: could not read history for {currency}: {e}')
        return []


def merge_observations(existing, new_obs):
    """Merge a new observation into the existing history.

    Core logic: builds a month-keyed map (YYYY-MM → observation) so duplicate
    months are deduplicated and the newest value wins.

    Gap-fill logic: when a central bank holds its rate, the primary source
    (e.g. ECB SDMX, BIS-CBPOL) returns the last *decision* date rather than
    today's date. This means merge_observations would keep overwriting the same
    month slot and leave all intermediate months unpopulated — even though the
    rate was valid and unchanged throughout. The fix: after inserting new_obs,
    fill every missing month between the latest existing observation and today
    with the rate from the most recent observation before that gap. This is
    correct by definition: a "hold" decision means the rate was unchanged.

    Uses stdlib only (no dateutil) — compatible with the workflow pip environment.
    The :36 cap and descending sort are preserved.
    """
    obs_map = {}
    for o in existing:
        mk = o['date'][:7]
        obs_map[mk] = {'date': mk + '-01', 'value': o['value']}
    mk_new = new_obs['date'][:7]
    obs_map[mk_new] = {'date': mk_new + '-01', 'value': new_obs['value']}

    # ── Gap-fill: propagate rate forward into missing months up to today ──
    # Only applies when there was pre-existing history — if existing was empty
    # we just inserted one observation and there is nothing to fill forward.
    today_ym = date.today().strftime('%Y-%m')
    if obs_map and existing:
        latest_month = max(obs_map.keys())
        if latest_month < today_ym:
            latest_rate = obs_map[latest_month]['value']
            # Advance one month at a time using stdlib date arithmetic
            y, m = int(latest_month[:4]), int(latest_month[5:7])
            while True:
                m += 1
                if m > 12:
                    m = 1
                    y += 1
                ym = f'{y:04d}-{m:02d}'
                if ym > today_ym:
                    break
                if ym not in obs_map:
                    obs_map[ym] = {'date': ym + '-01', 'value': latest_rate}

    return sorted(obs_map.values(), key=lambda x: x['date'], reverse=True)[:36]


def get_prev_rate(currency):
    existing = load_existing_observations(currency)
    if existing:
        try:
            return float(existing[0]['value'])
        except:
            pass
    return None


def get_prev_source(currency):
    path = f'rates/{currency}.json'
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        obs = data.get('observations', [])
        return obs[0].get('source', '') if obs else None
    except:
        return None


def make_result(rate, ref_date, source):
    """Construye el dict de resultado de forma consistente."""
    return {'rate': rate, 'ref_date': ref_date, 'ref_raw': ref_date, 'source': source}


# ── Fuentes primarias oficiales ────────────────────────────────────────────────

def fetch_aud_rba():
    """
    AUD — Reserve Bank of Australia
    Fuente 1: rba.gov.au/cash-rate-target-overview.html — página HTML estática
      (sin JS requerido para el dato principal), actualización el mismo día.
    Fuente 2: CSV f1.1-data.csv — multi-columna, columna FIRMMCRT.
      El CSV usa formato de fecha variado (YYYY-MM-DD o DD-Mon-YYYY).
    """
    today_str = str(date.today())

    # ── Fuente 1: página cash-rate-target-overview (HTML estático) ──────────
    try:
        url = 'https://www.rba.gov.au/cash-rate-target-overview.html'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'lxml')
            text = soup.get_text(' ', strip=True)

            # El patrón en la página es: "Cash rate target X.XX % Effective date DD Month YYYY"
            # Buscar número seguido de % en el contexto de la sección del cash rate
            m = re.search(
                r'[Cc]ash rate target\s+(\d+\.?\d*)\s*%.*?[Ee]ffective date\s+(\d+\s+\w+\s+\d{4})',
                text, re.DOTALL
            )
            if m:
                rate = clean_rate(m.group(1))
                raw_date = m.group(2).strip()
                try:
                    dt = datetime.strptime(raw_date, '%d %B %Y').strftime('%Y-%m-%d')
                except ValueError:
                    dt = today_str
                if rate:
                    print(f'    ✓ AUD: {rate}%  date={dt}  [RBA cash-rate-overview]')
                    return make_result(rate, dt, 'RBA-HTML')

            # Fallback dentro de la página: buscar cualquier número % prominente
            # La sección tiene el rate como texto grande antes de "Effective date"
            m2 = re.search(r'(\d+\.?\d*)\s*%\s*Effective date\s+(\d+\s+\w+\s+\d{4})', text)
            if m2:
                rate = clean_rate(m2.group(1))
                raw_date = m2.group(2).strip()
                try:
                    dt = datetime.strptime(raw_date, '%d %B %Y').strftime('%Y-%m-%d')
                except ValueError:
                    dt = today_str
                if rate:
                    print(f'    ✓ AUD: {rate}%  date={dt}  [RBA cash-rate-overview fallback]')
                    return make_result(rate, dt, 'RBA-HTML')
    except Exception as e:
        print(f'    ✗ AUD: RBA HTML — {e}')

    # ── Fuente 2: CSV f1.1-data.csv (columna FIRMMCRT) ──────────────────────
    try:
        url = 'https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'    ✗ AUD: RBA CSV HTTP {r.status_code}')
            return None

        reader = csv.reader(r.text.splitlines())
        all_rows = list(reader)
        if not all_rows:
            return None

        # Encontrar columna FIRMMCRT en fila 0
        target_col = None
        for j, cell in enumerate(all_rows[0]):
            if cell.strip().upper() == 'FIRMMCRT':
                target_col = j
                break
        if target_col is None and len(all_rows) > 1:
            for j, cell in enumerate(all_rows[1]):
                if 'cash rate target' in cell.strip().lower():
                    target_col = j
                    break
        if target_col is None:
            print('    ✗ AUD: RBA CSV — no se encontró columna FIRMMCRT')
            return None

        print(f'    → AUD: RBA CSV columna {target_col} = FIRMMCRT')

        # Recorrer filas — el RBA puede usar YYYY-MM-DD o DD-Mon-YYYY
        rows_with_data = []
        for row in all_rows:
            if len(row) > target_col:
                raw_date = row[0].strip()
                val_str  = row[target_col].strip()
                if val_str in ('', '.'):
                    continue
                # Intentar parsear la fecha en múltiples formatos
                dt = None
                for fmt in ('%Y-%m-%d', '%d-%b-%Y', '%d/%m/%Y', '%d %b %Y'):
                    try:
                        dt = datetime.strptime(raw_date, fmt).strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
                if dt:
                    try:
                        float(val_str)
                        rows_with_data.append((dt, val_str))
                    except ValueError:
                        continue

        if not rows_with_data:
            print('    ✗ AUD: RBA CSV — sin filas de datos válidas')
            # Imprimir primeras filas para diagnóstico
            for i, row in enumerate(all_rows[:12]):
                print(f'      row {i}: {row[:3]}')
            return None

        last_date, last_val = rows_with_data[-1]
        rate = clean_rate(last_val)
        if rate:
            print(f'    ✓ AUD: {rate}%  date={last_date}  [RBA CSV f1.1 col={target_col}]')
            return make_result(rate, last_date, 'RBA-CSV')
    except Exception as e:
        print(f'    ✗ AUD: RBA CSV — {e}')
    return None


def fetch_cad_boc():
    """
    CAD — Bank of Canada Valet API
    Serie V39079: Target for the Overnight Rate.
    API pública JSON, sin key. Actualización: mismo día.
    """
    try:
        url = 'https://www.bankofcanada.ca/valet/observations/V39079/json?recent=5'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'    ✗ CAD: BoC Valet HTTP {r.status_code}')
            return None

        data = r.json()
        observations = data.get('observations', [])
        for obs in reversed(observations):
            val = obs.get('V39079', {}).get('v', '')
            dt  = obs.get('d', '')
            if val and val not in ('.', ''):
                rate = clean_rate(val)
                if rate:
                    print(f'    ✓ CAD: {rate}%  date={dt}  [BoC Valet V39079]')
                    return make_result(rate, dt, 'BoC-Valet')
        print('    ✗ CAD: BoC Valet — sin observaciones válidas')
    except Exception as e:
        print(f'    ✗ CAD: BoC Valet — {e}')
    return None


def fetch_eur_ecb():
    """
    EUR — ECB Deposit Facility Rate (DFR).
    The DFR is the FX-market standard for EUR policy rate:
    it sets the floor of the rate corridor and anchors the money market.
    The MRO (Main Refinancing Operations) is the ceiling — used as fallback.

    Source priority:
      1. ECB SDMX REST API (data-api.ecb.europa.eu) — official, same-day
         Series: FM/B.U2.EUR.4F.KR.DFR.LEV (B=business) / FM/M... (M=monthly)
         Tries both DFR and MRO, both B and M frequencies (4 attempts total)
      2. ECB website key-dates HTML page — scraped when API unavailable
         URL: ecb.europa.eu/mopo/implement/sf/html/index.en.html
      3. BIS CBPOL data portal — official CB rates, reliable fallback (eurozone = XM)
      4. global-rates.com — last-resort scraping (may lag 1-2 days)

    Note: Never use MANUAL_OVERRIDES for EUR — unverified estimates cause
    more harm than showing a clearly labelled stale value.
    """
    today_str = str(date.today())

    # ── Source 1: ECB SDMX API ────────────────────────────────────────────────
    series_list = [
        ('FM/B.U2.EUR.4F.KR.DFR.LEV',    'DFR'),
        ('FM/B.U2.EUR.4F.KR.MRR_FR.LEV', 'MRO'),
    ]
    for series_key_ecb, label in series_list:
        for freq in ['B', 'M']:
            parts     = series_key_ecb.split('/')
            flow      = parts[0]
            key       = '/'.join(parts[1:])
            key_parts = key.split('.')
            key_parts[0] = freq
            key_with_freq = '.'.join(key_parts)
            try:
                url = (
                    f'https://data-api.ecb.europa.eu/service/data/'
                    f'{flow}/{key_with_freq}'
                    f'?lastNObservations=40&format=jsondata'
                )
                r = requests.get(url, headers={**HEADERS, 'Accept': 'application/json'}, timeout=15)
                if r.status_code != 200:
                    print(f'    ✗ EUR: ECB SDMX {label}/{freq} HTTP {r.status_code}')
                    continue

                data        = r.json()
                series      = data['dataSets'][0]['series']
                s_key       = list(series.keys())[0]
                obs_dict    = series[s_key]['observations']
                dates       = data['structure']['dimensions']['observation'][0]['values']

                last_idx = last_val = None
                for idx_str, obs_vals in obs_dict.items():
                    idx = int(idx_str)
                    val = obs_vals[0]
                    if val is not None:
                        if last_idx is None or idx > last_idx:
                            last_idx = idx
                            last_val = val

                if last_idx is not None and last_val is not None:
                    dt = dates[last_idx]['id']
                    if re.match(r'^\d{4}-\d{2}$', dt):
                        dt = dt + '-01'
                    rate = clean_rate(str(last_val))
                    if rate:
                        print(f'    ✓ EUR: {rate}%  date={dt}  [ECB SDMX {label}/{freq}]')
                        return make_result(rate, dt, 'ECB-SDMX')

                print(f'    ✗ EUR: ECB SDMX {label}/{freq} — no valid observations')
            except Exception as e:
                print(f'    ✗ EUR: ECB SDMX {label}/{freq} — {e}')

    # ── Source 2: ECB website — key interest rates HTML page ─────────────────
    # The ECB publishes current rates on a human-readable page that changes rarely.
    # Scraping this is resilient to SDMX API changes.
    ecb_html_urls = [
        'https://www.ecb.europa.eu/mopo/implement/sf/html/index.en.html',
        'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html',
    ]
    for url in ecb_html_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f'    ✗ EUR: ECB HTML {r.status_code} — {url[-60:]}')
                continue
            soup = BeautifulSoup(r.content, 'lxml')
            text = soup.get_text(' ', strip=True)
            # Look for "Deposit facility" followed by a rate number
            # Page format: "Deposit facility  X.XX"
            patterns = [
                r'[Dd]eposit facility[\s\S]{0,80}?(\d+\.\d{2})\s*%',
                r'[Dd]eposit facility[\s\S]{0,80}?([+-]?\d+\.\d{2})(?:\s|%)',
                r'(\d+\.\d{2})\s*%[\s\S]{0,80}?[Dd]eposit facility',
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    rate = clean_rate(m.group(1))
                    if rate is not None and MIN_POLICY_RATE_PP <= float(rate) <= MAX_POLICY_RATE_PP:
                        print(f'    ✓ EUR: {rate}%  date={today_str}  [ECB HTML key-rates]')
                        return make_result(rate, today_str, 'ECB-HTML')
            print(f'    ✗ EUR: ECB HTML — deposit facility rate not found in page')
        except Exception as e:
            print(f'    ✗ EUR: ECB HTML — {e}')

    # ── Source 3: BIS CBPOL (eurozone = XM) ──────────────────────────────────
    result = fetch_bis_policy_rate('EUR')
    if result:
        return result

    # ── Source 4: global-rates.com (last resort, may lag 1-2 days) ───────────
    result = fetch_global_rates_fallback('EUR')
    if result:
        return result

    print('    ✗ EUR: all sources failed')
    return None


def fetch_gbp_boe():
    """
    GBP — Bank of England Interactive Database
    Serie IUDBEDR: Official Bank Rate, CSV parametrizado.
    Actualización: mismo día de la decisión del MPC.

    CRÍTICO: usar csv.x=yes (NO xml.x=yes — ese devuelve XML con metadatos
    que contamina el parsing: clean_rate() puede extraer números del texto).
    """
    try:
        url = (
            'https://www.bankofengland.co.uk/boeapps/database/'
            '_iadb-fromshowcolumns.asp'
            '?csv.x=yes'
            '&Datefrom=01/Jan/2020&Dateto=now'
            '&SeriesCodes=IUDBEDR'
            '&CSVF=TN&UsingCodes=Y'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'    ✗ GBP: BoE Database HTTP {r.status_code}')
            return None

        # Con CSVF=TN no hay fila de encabezado: cada fila es (fecha, valor)
        reader = csv.reader(StringIO(r.text))
        rows   = []
        for row in reader:
            if len(row) >= 2:
                raw_date = row[0].strip().strip('"')
                raw_val  = row[1].strip().strip('"')
                # Excluir encabezados textuales si los hubiera
                if raw_val not in ('', '.', 'IUDBEDR') and not raw_date.startswith('D'):
                    rows.append((raw_date, raw_val))

        if not rows:
            print('    ✗ GBP: BoE Database — sin filas válidas')
            return None

        raw_date, raw_val = rows[-1]
        rate = clean_rate(raw_val)
        try:
            dt = datetime.strptime(raw_date, '%d %b %Y').strftime('%Y-%m-%d')
        except ValueError:
            dt = raw_date

        if rate:
            print(f'    ✓ GBP: {rate}%  date={dt}  [BoE IUDBEDR]')
            return make_result(rate, dt, 'BoE-CSV')
        print(f'    ✗ GBP: BoE Database — valor no válido: {raw_val!r}')
    except Exception as e:
        print(f'    ✗ GBP: BoE Database — {e}')
    return None


def fetch_usd_nyfed():
    """
    USD — Federal Reserve Bank of New York (API oficial, sin key)
    Endpoint: markets.newyorkfed.org — EFFR con target range.
    El campo <targetRateTo> es el techo del rango objetivo de la Fed Funds Rate.
    Actualización: mismo día hábil. Sin API key. XML limpio.
    URL: https://markets.newyorkfed.org/read?productCode=50&eventCodes=500&limit=25&startPosition=0&sort=postDt:-1&format=xml
    """
    try:
        url = (
            'https://markets.newyorkfed.org/read'
            '?productCode=50&eventCodes=500&limit=25'
            '&startPosition=0&sort=postDt:-1&format=xml'
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'    ✗ USD: NY Fed HTTP {r.status_code}')
            return None

        # Parsear XML: tomar el primer <rate> (más reciente) y extraer <targetRateTo>
        soup = BeautifulSoup(r.content, 'xml')
        first_rate = soup.find('rate')
        if not first_rate:
            print('    ✗ USD: NY Fed — sin elementos <rate> en respuesta')
            return None

        target_to   = first_rate.find('targetRateTo')
        eff_date    = first_rate.find('effectiveDate')

        if not target_to or not target_to.text.strip():
            print('    ✗ USD: NY Fed — <targetRateTo> ausente o vacío')
            return None

        rate = clean_rate(target_to.text.strip())
        dt   = eff_date.text.strip() if eff_date else str(date.today())

        if rate:
            print(f'    ✓ USD: {rate}%  date={dt}  [NY Fed EFFR targetRateTo]')
            return make_result(rate, dt, 'NYFed-EFFR')

        print(f'    ✗ USD: NY Fed — valor no válido: {target_to.text!r}')
    except Exception as e:
        print(f'    ✗ USD: NY Fed — {e}')
    return None


def fetch_usd_fred():
    """
    USD — FRED API (fallback si NY Fed falla). Requiere FRED_API_KEY.
    Serie FEDFUNDS: Federal Funds Effective Rate. Lag de ~1 día hábil.
    """
    if not FRED_API_KEY:
        return None
    try:
        params = {
            'series_id':  'FEDFUNDS',
            'api_key':    FRED_API_KEY,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      '3',
        }
        r = requests.get(
            'https://api.stlouisfed.org/fred/series/observations',
            params=params, timeout=15,
        )
        if r.status_code != 200:
            print(f'    ✗ USD: FRED API HTTP {r.status_code}')
            return None
        for obs in r.json().get('observations', []):
            if obs.get('value') not in ('.', '', None):
                rate = clean_rate(obs['value'])
                if rate:
                    print(f'    ✓ USD: {rate}%  date={obs["date"]}  [FRED FEDFUNDS]')
                    return make_result(rate, obs['date'], 'FRED:FEDFUNDS')
        print('    ✗ USD: FRED API — sin observaciones válidas')
    except Exception as e:
        print(f'    ✗ USD: FRED API — {e}')
    return None


def fetch_jpy_boj():
    """
    JPY — orden de fuentes:
      1. BIS Data Portal WS_CBPOL/M.JP  — API oficial, sin key, más estable
      2. BoJ Time-Series API FM01'MADR1Z@D — API oficial BoJ, lanzada Feb 2026
      3. Scraping boj.or.jp              — último recurso HTML
    """
    today_str = str(date.today())

    # Intento 1: BIS Data Portal (más estable y sin problemas de series code)
    result = fetch_bis_policy_rate('JPY')
    if result:
        return result

    # Intento 2: BoJ Time-Series API
    try:
        start = (date.today() - timedelta(days=14)).strftime('%Y%m%d')
        end   = date.today().strftime('%Y%m%d')
        params = {
            'format':    'json',
            'lang':      'en',
            'db':        'FM',
            'code':      "FM01'MADR1Z@D",
            'startDate': start,
            'endDate':   end,
        }
        r = requests.get(
            'https://www.stat-search.boj.or.jp/api/v1/getDataCode',
            params=params, headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            for series in (data.get('data') or []):
                for obs in reversed(series.get('seriesData') or []):
                    val = str(obs.get('value', '') or '').strip()
                    dt  = str(obs.get('date',  '') or '').strip()
                    if val not in ('', '.', 'ND') and dt:
                        rate = clean_rate(val)
                        if rate is not None:
                            if re.match(r'^\d{8}$', dt):
                                dt = f'{dt[:4]}-{dt[4:6]}-{dt[6:]}'
                            print(f'    ✓ JPY: {rate}%  date={dt}  [BoJ API FM01]')
                            return make_result(rate, dt, 'BoJ-API')
        else:
            print(f'    ✗ JPY: BoJ API HTTP {r.status_code}')
    except Exception as e:
        print(f'    ✗ JPY: BoJ API — {e}')

    # Intento 3: scraping boj.or.jp
    try:
        url = 'https://www.boj.or.jp/en/statistics/boj/other/discount/index.htm'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'lxml')
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    text = row.get_text(' ', strip=True).lower()
                    if 'policy interest rate' in text or 'complementary deposit' in text:
                        for col in row.find_all('td'):
                            rate = clean_rate(col.get_text(strip=True))
                            if rate is not None:
                                print(f'    ✓ JPY: {rate}%  date={today_str}  [BoJ scraping]')
                                return make_result(rate, today_str, 'BoJ-scraping')
            for m in re.finditer(r'(-?\d+\.\d{2})\s*%', soup.get_text()):
                rate = clean_rate(m.group(1))
                if rate is not None and abs(float(rate)) <= 2.0:
                    print(f'    ✓ JPY: {rate}%  date={today_str}  [BoJ scraping text]')
                    return make_result(rate, today_str, 'BoJ-scraping')
    except Exception as e:
        print(f'    ✗ JPY: BoJ scraping — {e}')

    print('    ✗ JPY: todas las fuentes fallaron')
    return None





def fetch_chf_snb():
    """
    CHF -- Swiss National Bank (scraping directo)
    URLs de tasas e intereses actuales del SNB.
    Actualizacion: mismo dia de la decision del SNB (trimestral).
    Nota: el SNB cambia sus URLs periodicamente; intenta varias en orden.
    URL confirmada activa: /en/the-snb/mandates-goals/statistics/statistics-pub/...
    """
    urls = [
        # URL confirmada activa (encontrada en búsqueda 2026)
        'https://www.snb.ch/en/the-snb/mandates-goals/statistics/statistics-pub/current_interest_exchange_rates',
        # URL alternativa — página de instrumento de política monetaria
        'https://www.snb.ch/en/monetary-policy/monetary-policy-instruments/snb-policy-rate',
        # URL alternativa — página de tasas actuales
        'https://www.snb.ch/en/the-snb/mandates-goals/monetary-policy/current-interest-rates-and-exchange-rates',
    ]
    today_str = str(date.today())
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, 'lxml')
            for tag in soup.find_all(['tr', 'p', 'div', 'td', 'li']):
                text = tag.get_text(' ', strip=True).lower()
                if 'snb policy rate' in text or ('policy rate' in text and 'snb' in text):
                    m = re.search(r'(-?\d+\.?\d*)\s*%', tag.get_text(strip=True))
                    if m:
                        rate = clean_rate(m.group(1))
                        if rate is not None:
                            print(f'    ok CHF: {rate}%  date={today_str}  [SNB scraping]')
                            return make_result(rate, today_str, 'SNB-scraping')
        except Exception:
            continue
    print('    x CHF: SNB scraping -- no se encontro la policy rate')
    return None


def fetch_bis_policy_rate(currency):
    """
    Fuente BIS Data Portal — Central Bank Policy Rates (WS_CBPOL).
    API pública SDMX v2, sin key. Cubre NZD, JPY, y todos los G8.
    Endpoint: https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/{freq}.{country}
    Frecuencia mensual (M) — la serie diaria tiene lag mayor.
    CSV como formato, columnas TIME_PERIOD y OBS_VALUE.
    """
    BIS_COUNTRY = {
        'NZD': 'NZ', 'JPY': 'JP', 'AUD': 'AU', 'CHF': 'CH',
        'CAD': 'CA', 'GBP': 'GB', 'EUR': 'XM', 'USD': 'US',
    }
    country = BIS_COUNTRY.get(currency)
    if not country:
        return None

    # Intentar mensual primero (más completa), luego diaria
    for freq in ['M', 'D']:
        try:
            url = (
                f'https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/'
                f'{freq}.{country}'
                f'?startPeriod=2025-01&format=csv&detail=dataonly'
            )
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f'    ✗ {currency}: BIS API {freq} HTTP {r.status_code}')
                continue

            lines = r.text.splitlines()
            if not lines:
                continue

            reader = csv.reader(lines)
            headers_row = next(reader, None)
            if not headers_row:
                continue

            headers_row = [h.strip().strip('"') for h in headers_row]
            try:
                date_col = headers_row.index('TIME_PERIOD')
                val_col  = headers_row.index('OBS_VALUE')
            except ValueError:
                date_col = 0
                val_col  = len(headers_row) - 1

            rows_with_data = []
            for row in reader:
                if len(row) > max(date_col, val_col):
                    dt_str  = row[date_col].strip().strip('"')
                    val_str = row[val_col].strip().strip('"')
                    if dt_str and val_str not in ('', '.'):
                        try:
                            float(val_str)
                            # Normalizar YYYY-MM → YYYY-MM-01
                            if re.match(r'^\d{4}-\d{2}$', dt_str):
                                dt_str = dt_str + '-01'
                            rows_with_data.append((dt_str, val_str))
                        except ValueError:
                            continue

            if not rows_with_data:
                continue

            last_date, last_val = rows_with_data[-1]
            rate = clean_rate(last_val)
            if rate:
                print(f'    ✓ {currency}: {rate}%  date={last_date}  [BIS CBPOL/{freq}.{country}]')
                return make_result(rate, last_date, 'BIS-CBPOL')

        except Exception as e:
            print(f'    ✗ {currency}: BIS API {freq} — {e}')

    print(f'    ✗ {currency}: BIS API — todas las frecuencias fallaron')
    return None


def fetch_nzd_rbnz():
    """
    NZD — BIS Data Portal como fuente primaria (RBNZ bloquea todo con 403).
    El BIS publica tasas de política monetaria oficiales para NZD con frecuencia diaria.
    Fallback: RBNZ tabla B2 con browser headers (intento aunque probablemente 403).
    """
    # ── Fuente 1: BIS Data Portal (más confiable para NZD) ──────────────────
    result = fetch_bis_policy_rate('NZD')
    if result:
        return result

    # ── Fuente 2: RBNZ tabla B2 con browser headers ──────────────────────────
    browser_headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.rbnz.govt.nz/',
    }
    urls = [
        'https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates/hb2-download',
        'https://www.rbnz.govt.nz/assets/Uploads/Statistics/hb2-monthly.csv',
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=browser_headers, timeout=20)
            if r.status_code != 200:
                print(f'    ✗ NZD: RBNZ {r.status_code} — {url[-50:]}')
                continue
            lines = r.text.splitlines()
            ocr_col = header_idx = None
            for i, line in enumerate(lines):
                if 'official cash rate' in line.lower() or 'ocr' in line.lower():
                    cols = [c.strip().strip('"') for c in line.split(',')]
                    for j, col in enumerate(cols):
                        if 'official cash rate' in col.lower() or col.lower() == 'ocr':
                            ocr_col = j; header_idx = i; break
                    if ocr_col is not None:
                        break
            if header_idx is None or ocr_col is None:
                continue
            rows_with_data = []
            reader = csv.reader(lines[header_idx + 1:])
            for row in reader:
                if len(row) > ocr_col:
                    date_str = row[0].strip().strip('"')
                    val_str  = row[ocr_col].strip().strip('"')
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str) and val_str not in ('', '.'):
                        try:
                            float(val_str)
                            rows_with_data.append((date_str, val_str))
                        except ValueError:
                            continue
            if rows_with_data:
                last_date, last_val = rows_with_data[-1]
                rate = clean_rate(last_val)
                if rate:
                    print(f'    ✓ NZD: {rate}%  date={last_date}  [RBNZ B2 CSV]')
                    return make_result(rate, last_date, 'RBNZ-CSV')
        except Exception as e:
            print(f'    ✗ NZD: RBNZ — {e}')

    print('    ✗ NZD: todas las fuentes fallaron')
    return None


# ── Fuentes de fallback ────────────────────────────────────────────────────────

def fetch_fred_api_single(currency):
    """FRED API para una divisa (fallback). Requiere FRED_API_KEY."""
    if not FRED_API_KEY:
        return None
    series_id = FRED_RATE_SERIES.get(currency)
    if not series_id:
        return None
    try:
        params = {
            'series_id':  series_id,
            'api_key':    FRED_API_KEY,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      '3',
        }
        r = requests.get(
            'https://api.stlouisfed.org/fred/series/observations',
            params=params, timeout=15,
        )
        if r.status_code != 200:
            return None
        for obs in r.json().get('observations', []):
            if obs.get('value') not in ('.', '', None):
                rate = clean_rate(obs['value'])
                if rate:
                    print(f'    ✓ {currency}: {rate}%  date={obs["date"]}  [FRED:{series_id}]')
                    return make_result(rate, obs['date'], f'FRED:{series_id}')
        time.sleep(0.3)
    except Exception as e:
        print(f'    ✗ {currency}: FRED API fallback — {e}')
    return None


def fetch_fred_csv_single(currency):
    """FRED CSV público (fallback sin API key). Lag conocido en algunas series."""
    series_id = FRED_RATE_SERIES.get(currency)
    if not series_id:
        return None
    try:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        reader = csv.reader(StringIO(r.text))
        rows = []
        for row in reader:
            if len(row) == 2 and row[0] != 'DATE' and row[1] not in ('.', ''):
                try:
                    rows.append((row[0], float(row[1])))
                except:
                    continue
        if rows:
            dt, val = rows[-1]
            rate = clean_rate(str(val))
            if rate:
                print(f'    ✓ {currency}: {rate}%  date={dt}  [FRED-CSV:{series_id}]')
                return make_result(rate, dt, f'FRED-CSV:{series_id}')
        time.sleep(0.5)
    except Exception as e:
        print(f'    ✗ {currency}: FRED CSV fallback — {e}')
    return None


def fetch_global_rates_fallback(currency):
    """global-rates.com — último fallback externo antes del archivo guardado."""
    if currency not in GLOBAL_RATES_URLS:
        return None
    try:
        url = (f'https://www.global-rates.com/en/interest-rates/central-banks/'
               f'{GLOBAL_RATES_URLS[currency]}')
        r = requests.get(url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(r.content, 'lxml')
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) >= 2:
                cols = rows[1].find_all('td')
                if len(cols) >= 2:
                    rate = clean_rate(cols[1].get_text(strip=True))
                    if rate:
                        print(f'    ✓ {currency}: {rate}%  [global-rates.com fallback]')
                        return make_result(rate, None, 'global-rates.com')
    except Exception as e:
        print(f'    ✗ {currency}: global-rates.com — {e}')
    return None


# ── Validación cruzada: Frankfurter ────────────────────────────────────────────

def fetch_frankfurter_rates():
    print('\n' + '=' * 70)
    print('VALIDATION: Frankfurter ECB (FX rates for cross-check)')
    print('=' * 70)
    try:
        r = requests.get('https://api.frankfurter.app/latest?from=USD', timeout=10)
        if not r.ok:
            print(f'  WARNING: Frankfurter returned {r.status_code}')
            return {}
        data  = r.json()
        rates = data.get('rates', {})
        rates['USD'] = 1.0
        print(f'  OK: {len(rates)} currencies via Frankfurter')
        return rates
    except Exception as e:
        print(f'  WARNING: Frankfurter unavailable: {e}')
        return {}


def validate_rates(final_rates, frankfurter_fx):
    issues, warnings = [], []
    for currency, entry in final_rates.items():
        rate = float(entry['rate'])
        if not (MIN_POLICY_RATE_PP <= rate <= MAX_POLICY_RATE_PP):
            issues.append(
                f'{currency}: rate {rate}% outside plausible range '
                f'[{MIN_POLICY_RATE_PP}, {MAX_POLICY_RATE_PP}]'
            )
        prev     = get_prev_rate(currency)
        prev_src = get_prev_source(currency)
        if prev is not None:
            # Si el valor previo en el historial está fuera del rango plausible,
            # es una observación corrupta (ej. "12%" por bug de parseo XML).
            # En ese caso ignorar la comparación delta y registrar solo un warning.
            if not (MIN_POLICY_RATE_PP <= prev <= MAX_POLICY_RATE_PP):
                warnings.append(
                    f'{currency}: historial previo corrupto ({prev}% fuera de rango) — '
                    f'ignorando comparación delta. Nuevo valor: {rate}%  [{entry["source"]}]'
                )
            else:
                delta = abs(rate - prev)
                if delta > MAX_CHANGE_VS_PREV_PP:
                    if prev_src in STALE_SOURCES:
                        warnings.append(
                            f'{currency}: rate corrected {prev}%→{rate}% (Δ={delta:.2f}pp) '
                            f'— prev source was stale ({prev_src}), now using {entry["source"]}'
                        )
                    else:
                        issues.append(
                            f'{currency}: rate changed {rate}% vs previous {prev}% '
                            f'(Δ={delta:.2f}pp > {MAX_CHANGE_VS_PREV_PP}pp). '
                            f'Central bank meeting or scraping error?'
                        )

    if frankfurter_fx and len(final_rates) >= 4:
        policy_rates = {c: float(e['rate']) for c, e in final_rates.items()}
        lowest  = min(policy_rates, key=lambda c: policy_rates[c])
        highest = max(policy_rates, key=lambda c: policy_rates[c])
        print(f'\n  FX cross-check: lowest={lowest} ({policy_rates[lowest]}%), '
              f'highest={highest} ({policy_rates[highest]}%)')

    return issues, warnings


# ── Tabla de fuentes primarias por divisa ──────────────────────────────────────

PRIMARY_FETCHERS = {
    'AUD': fetch_aud_rba,
    'CAD': fetch_cad_boc,
    'EUR': fetch_eur_ecb,
    'GBP': fetch_gbp_boe,
    'USD': lambda: fetch_usd_nyfed() or fetch_usd_fred(),
    'JPY': fetch_jpy_boj,   # BIS primero, luego BoJ scraping
    'CHF': lambda: fetch_bis_policy_rate('CHF') or fetch_chf_snb(),
    'NZD': fetch_nzd_rbnz,  # BIS primero, luego RBNZ
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today             = str(date.today())
    today_month_start = date.today().strftime('%Y-%m-01')
    run_ts            = datetime.utcnow().isoformat() + 'Z'

    print(f'\nRun date:  {today}')
    print(f'Version:   fetch_rates.py v13.0 — fuentes oficiales primarias')
    os.makedirs('rates', exist_ok=True)

    final_rates = {}

    # ── Paso 0: overrides manuales (máxima prioridad) ─────────────────────
    if MANUAL_OVERRIDES:
        print('\n' + '=' * 70)
        print('SOURCE 0: MANUAL OVERRIDES (highest priority)')
        print('=' * 70)
        for currency, (rate_val, ref_date) in MANUAL_OVERRIDES.items():
            if currency in CURRENCIES:
                final_rates[currency] = make_result(rate_val, ref_date, 'manual-override')
                print(f'  ✓ {currency}: {rate_val}%  date={ref_date}  [MANUAL OVERRIDE]')

    # ── Paso 1: fuentes primarias oficiales ───────────────────────────────
    print('\n' + '=' * 70)
    print('SOURCE 1: Official Central Bank Sources (primary)')
    print('=' * 70)
    for currency in CURRENCIES:
        if currency in final_rates:
            print(f'  ↷ {currency}: skipped (manual override active)')
            continue
        fetcher = PRIMARY_FETCHERS.get(currency)
        if fetcher:
            result = fetcher()
            if result:
                final_rates[currency] = result
        time.sleep(0.3)

    # ── Paso 2: FRED API fallback (requiere FRED_API_KEY) ─────────────────
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing and FRED_API_KEY:
        print('\n' + '=' * 70)
        print(f'SOURCE 2: FRED API fallback — {", ".join(missing)}')
        print('=' * 70)
        for currency in missing:
            if currency == 'USD':
                continue  # USD ya intentó FRED en paso 1
            result = fetch_fred_api_single(currency)
            if result:
                final_rates[currency] = result
            time.sleep(0.3)

    # ── Paso 3: BIS CBPOL fallback universal ──────────────────────────────
    # El BIS cubre todas las divisas G8 con datos oficiales de bancos centrales.
    # Útil como red de seguridad cuando las fuentes primarias fallan.
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing:
        print('\n' + '=' * 70)
        print(f'SOURCE 3: BIS CBPOL universal fallback — {", ".join(missing)}')
        print('=' * 70)
        for currency in missing:
            result = fetch_bis_policy_rate(currency)
            if result:
                final_rates[currency] = result
            time.sleep(0.3)

    # ── Paso 4: FRED CSV público (sin key, lag conocido) ──────────────────
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing:
        print('\n' + '=' * 70)
        print(f'SOURCE 4: FRED CSV public fallback — {", ".join(missing)}')
        print('=' * 70)
        for currency in missing:
            result = fetch_fred_csv_single(currency)
            if result:
                final_rates[currency] = result

    # ── Paso 5: global-rates.com (último fallback externo) ────────────────
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing:
        print('\n' + '=' * 70)
        print(f'SOURCE 5: global-rates.com fallback — {", ".join(missing)}')
        print('=' * 70)
        for currency in missing:
            result = fetch_global_rates_fallback(currency)
            if result:
                final_rates[currency] = result
            time.sleep(0.4)

    # ── Paso 6: archivo guardado (último recurso) ─────────────────────────
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing:
        print('\n' + '=' * 70)
        print(f'SOURCE 6: Cached file (last resort) — {", ".join(missing)}')
        print('=' * 70)
        for currency in missing:
            cached_obs = load_existing_observations(currency)
            if cached_obs:
                try:
                    latest = cached_obs[0]
                    cached_date = latest['date'][:10]
                    cached_age  = (date.today() - date.fromisoformat(cached_date)).days
                    # Skip stale caches (>60d): prefer FRED CSV over perpetually-recycled
                    # old values. This fixes EUR stuck at Jun-2025 after ECB API blocked in CI.
                    if cached_age > 60:
                        print(f'  ↷ {currency}: cached {latest["value"]}% from {cached_date} '
                              f'({cached_age}d) — too stale, trying FRED CSV first')
                        fred_result = fetch_fred_csv_single(currency)
                        if fred_result:
                            final_rates[currency] = fred_result
                            continue
                        # FRED CSV also failed — fall through to cached as last resort
                        print(f'  ↷ {currency}: FRED CSV failed, using stale cache anyway')
                    final_rates[currency] = make_result(
                        latest['value'], latest['date'], 'cached-file'
                    )
                    print(f'  ✓ {currency}: cached {latest["value"]}% from {cached_date}')
                except Exception as e:
                    print(f'  ✗ {currency}: cached file unreadable — {e}')
            else:
                print(f'  ✗ {currency}: no cached data available')

    # ── Paso 7: validación cruzada con Frankfurter ────────────────────────
    frankfurter_fx = fetch_frankfurter_rates()
    issues, warnings = validate_rates(final_rates, frankfurter_fx)

    if warnings:
        print('\nWARNINGS:')
        for w in warnings:
            print(f'  ⚠  {w}')
    if issues:
        print('\nVALIDATION ISSUES:')
        for issue in issues:
            print(f'  ✗  {issue}')

    # ── Paso 8: guardar rates/XX.json ─────────────────────────────────────
    print('\n' + '=' * 70)
    print('SAVING WITH HISTORICAL ACCUMULATION')
    print('=' * 70)

    health_data = {
        'runTimestamp':  run_ts,
        'runDate':       today,
        'frankfurterOk': bool(frankfurter_fx),
        'currencies':    {},
        'issues':        issues,
        'warnings':      warnings,
        'overallStatus': 'ok',
    }

    saved = []
    for currency in CURRENCIES:
        if currency not in final_rates:
            print(f'  SKIP {currency}: no data')
            health_data['currencies'][currency] = {
                'status': 'missing', 'rate': None, 'source': None, 'date': None,
            }
            continue

        entry            = final_rates[currency]
        rate_value       = entry['rate']
        observation_date = entry['ref_date'] if entry['ref_date'] else today_month_start

        existing_obs = load_existing_observations(currency)
        # NOTE: Do NOT filter existing_obs by observation_date here.
        # The original filter (`o['date'][:7] <= observation_date[:7]`) was intended
        # to prevent future-dated cache corruption, but it had a destructive side effect:
        # since official CB sources return the decision date (which is recent), the filter
        # kept discarding all historical observations, leaving files with only 1-2 months
        # of history (e.g. USD: 2 obs). The merge_observations() function already handles
        # deduplication correctly via the month-keyed map — no pre-filtering needed.

        new_observation = {
            'value':  rate_value,
            'date':   observation_date,
            'source': entry['source'],
        }
        merged_obs = merge_observations(existing_obs, new_observation)

        currency_issues = [i for i in issues if i.startswith(currency + ':')]
        if entry['source'] == 'cached-file':
            ccy_status = 'cached'
        elif currency_issues:
            ccy_status = 'warning'
        elif entry['source'] in OFFICIAL_SOURCES or 'FRED' in entry['source']:
            ccy_status = 'ok'
        else:
            ccy_status = 'fallback'

        rate_data = {
            'observations':      merged_obs,
            # lastUpdate refleja la fecha del run (hoy), no la fecha de publicación
            # del banco central. Esto evita que el campo aparezca stale cuando la
            # tasa no ha cambiado pero el script sí se ejecutó correctamente.
            # La fecha real de la publicación oficial queda en observations[0].date
            'lastUpdate':        today,
            'rateDate':          observation_date,   # fecha real del dato del CB
            'totalObservations': len(merged_obs),
        }
        with open(f'rates/{currency}.json', 'w') as f:
            json.dump(rate_data, f, indent=2)

        health_data['currencies'][currency] = {
            'status':  ccy_status,
            'rate':    rate_value,
            'source':  entry['source'],
            'date':    observation_date,
            'issues':  currency_issues,
        }
        print(f'  SAVED {currency}: {rate_value}% | date={observation_date} | '
              f'history={len(merged_obs)} obs | source={entry["source"]} | status={ccy_status}')
        saved.append(currency)

    # ── Paso 9: guardar rates/health.json ─────────────────────────────────
    missing_final = [c for c in CURRENCIES if c not in final_rates]
    cached_final  = [c for c in CURRENCIES
                     if c in final_rates and final_rates[c]['source'] == 'cached-file']

    if missing_final or issues:
        health_data['overallStatus'] = 'degraded'
    if cached_final and not missing_final and not issues:
        health_data['overallStatus'] = 'degraded_cached'
    if len(missing_final) > 2:
        health_data['overallStatus'] = 'critical'

    with open('rates/health.json', 'w') as f:
        json.dump(health_data, f, indent=2)
    print(f'\n  SAVED rates/health.json — overall status: {health_data["overallStatus"]}')

    # ── Paso 10: exit code ────────────────────────────────────────────────
    print('\n' + '=' * 70)
    if issues:
        print(f'RESULT: FAILED — {len(issues)} validation issue(s)')
        for issue in issues:
            print(f'  ✗  {issue}')
        sys.exit(1)
    elif missing_final:
        print(f'RESULT: PARTIAL — missing: {", ".join(missing_final)}')
        sys.exit(1)
    elif cached_final:
        print(f'RESULT: OK (cached fallback used for: {", ".join(cached_final)})')
        sys.exit(0)
    else:
        print(f'ALL {len(saved)} CURRENCIES UPDATED OK — no validation issues')
        sys.exit(0)


if __name__ == '__main__':
    main()
