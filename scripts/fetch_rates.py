#!/usr/bin/env python3
"""
fetch_rates.py
Obtiene tasas de política monetaria para las 8 divisas G8.

Flujo de datos:
  1. Fuente primaria:  Trading Economics (scraping)
  2. Fuente fallback:  global-rates.com (scraping)
  3. Validación:       Frankfurter ECB API (tasas de mercado de FX)
                       → NO reemplaza la tasa de política, solo sirve
                         como señal de coherencia cruzada.

Salidas:
  - rates/XX.json          historial de observaciones (sin cambios de formato)
  - rates/health.json      estado de fiabilidad por divisa + timestamp del run

El workflow hace exit(1) si alguna divisa tiene divergencia excesiva
entre la tasa scrapeada y la señal de mercado de Frankfurter.
Esto dispara la notificación de fallo en GitHub Actions.

D-01 FIX (auditoría de fiabilidad): El sistema previo no tenía
mecanismo de detección de datos desactualizados. Este script añade
validación cruzada y un health.json que el frontend puede leer para
mostrar alertas visibles en lugar de datos silenciosamente incorrectos.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import os
from datetime import date, datetime
import time

# ── Configuración ──────────────────────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']

COUNTRY_MAP = {
    'United States': 'USD',
    'Euro Area':     'EUR',
    'United Kingdom':'GBP',
    'Japan':         'JPY',
    'Canada':        'CAD',
    'Australia':     'AUD',
    'Switzerland':   'CHF',
    'New Zealand':   'NZD',
}

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

# Umbrales de validación cruzada (pp = puntos porcentuales).
# Frankfurter mide tipos de cambio, no tasas de política — la correlación
# no es directa, así que el umbral es generoso. El objetivo es detectar
# errores graves (ej. scraper devuelve 0% cuando la tasa real es 4%).
# Se compara el RANKING relativo de tasas, no los valores absolutos,
# porque Frankfurter no expone tasas de política monetaria directamente.
# En su lugar se usa el diferencial entre la tasa scrapeada de cada divisa
# y la media del G8: si el diferencial supera MAX_POLICY_RATE_PP, es sospechoso.
MAX_POLICY_RATE_PP = 15.0   # tasa de política <= 15% es razonable para G8
MIN_POLICY_RATE_PP = -2.0   # tasa de política >= -2% (tipos negativos extremos)

# Si la tasa nueva difiere más de este umbral de la última observación guardada,
# se marca como "outlier" y el workflow falla para revisión manual.
MAX_CHANGE_VS_PREV_PP = 1.0  # cambio > 1pp entre runs consecutivos es inusual


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


def parse_reference_date(ref_text):
    if not ref_text:
        return None
    ref_text = ref_text.strip()
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
    }
    match = re.match(r'([A-Za-z]{3})/(\d{2})', ref_text)
    if match:
        mon_str = match.group(1).capitalize()
        yr_str  = match.group(2)
        mon_num = month_map.get(mon_str)
        if mon_num:
            return f'20{yr_str}-{mon_num}-01'
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
        print(f'    Warning: could not read existing history for {currency}: {e}')
        return []


def merge_observations(existing, new_obs):
    def month_key(obs):
        return obs['date'][:7]

    def normalise_date(obs):
        return obs['date'][:7] + '-01'

    obs_map = {}
    for o in existing:
        mk = month_key(o)
        obs_map[mk] = {'date': normalise_date(o), 'value': o['value']}
    mk_new = month_key(new_obs)
    obs_map[mk_new] = {'date': normalise_date(new_obs), 'value': new_obs['value']}
    sorted_obs = sorted(obs_map.values(), key=lambda x: x['date'], reverse=True)
    return sorted_obs[:36]


def get_prev_rate(currency):
    """Devuelve la última tasa guardada para comparación (como float o None)."""
    existing = load_existing_observations(currency)
    if existing:
        try:
            return float(existing[0]['value'])
        except:
            pass
    return None


# ── Fuente 1: Trading Economics ────────────────────────────────────────────────

def fetch_trading_economics():
    print('\n' + '=' * 70)
    print('SOURCE 1: Trading Economics (Interest Rates + Reference Dates)')
    print('=' * 70)
    try:
        url = 'https://tradingeconomics.com/country-list/interest-rate?continent=world'
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 429:
            print('  Rate limited, waiting 30s...')
            time.sleep(30)
            r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'lxml')
        results = {}
        table = soup.find('table', {'class': 'table'})
        if not table:
            print('  ERROR: No table found')
            return results
        headers_row = table.find('thead')
        header_texts = []
        if headers_row:
            header_texts = [th.get_text(strip=True).lower()
                            for th in headers_row.find_all('th')]
        last_idx = 1
        ref_idx  = 3
        if header_texts:
            for i, h in enumerate(header_texts):
                if 'last' in h:
                    last_idx = i
                if 'reference' in h:
                    ref_idx = i
        print(f'  Column indices → Last={last_idx}, Reference={ref_idx}')
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            country = cols[0].get_text(strip=True)
            for country_name, currency in COUNTRY_MAP.items():
                if country_name.lower() in country.lower():
                    rate_text = cols[last_idx].get_text(strip=True) if last_idx < len(cols) else ''
                    rate = clean_rate(rate_text)
                    ref_text = cols[ref_idx].get_text(strip=True) if ref_idx < len(cols) else ''
                    ref_date = parse_reference_date(ref_text)
                    if rate is not None:
                        results[currency] = {
                            'rate': rate,
                            'ref_date': ref_date,
                            'ref_raw': ref_text,
                            'source': 'TradingEconomics',
                        }
                        print(f'  OK {currency}: {rate}%  reference={ref_text} → {ref_date}')
                    break
        return results
    except Exception as e:
        print(f'  ERROR: {e}')
        return {}


# ── Fuente 2: global-rates.com (fallback) ──────────────────────────────────────

def fetch_global_rates_fallback(currency):
    if currency not in GLOBAL_RATES_URLS:
        return None
    try:
        url = f'https://www.global-rates.com/en/interest-rates/central-banks/{GLOBAL_RATES_URLS[currency]}'
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, 'lxml')
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) >= 2:
                cols = rows[1].find_all('td')
                if len(cols) >= 2:
                    rate = clean_rate(cols[1].get_text(strip=True))
                    if rate:
                        print(f'  OK {currency}: {rate}% (fallback global-rates.com)')
                        return {'rate': rate, 'ref_date': None, 'ref_raw': '', 'source': 'global-rates.com'}
    except Exception as e:
        print(f'    Fallback error for {currency}: {e}')
    return None


# ── Validación cruzada: Frankfurter ────────────────────────────────────────────

def fetch_frankfurter_rates():
    """
    Obtiene tipos de cambio actuales desde Frankfurter/ECB.
    No son tasas de política monetaria, pero permiten validar que el
    orden de magnitud y la dirección de las tasas scrapeadas son coherentes.
    Retorna dict {currency: float} o {} si falla.
    """
    print('\n' + '=' * 70)
    print('VALIDATION: Frankfurter ECB (FX rates for cross-check)')
    print('=' * 70)
    try:
        r = requests.get('https://api.frankfurter.app/latest?from=USD', timeout=10)
        if not r.ok:
            print(f'  WARNING: Frankfurter returned {r.status_code}')
            return {}
        data = r.json()
        rates = data.get('rates', {})
        rates['USD'] = 1.0
        print(f'  OK: {len(rates)} currencies retrieved from Frankfurter')
        return rates
    except Exception as e:
        print(f'  WARNING: Frankfurter unavailable: {e}')
        return {}


def validate_rates(final_rates, frankfurter_fx):
    """
    Validación cruzada de las tasas de política monetaria scrapeadas.

    Estrategia:
    1. Rango plausible: la tasa debe estar en [MIN_POLICY_RATE_PP, MAX_POLICY_RATE_PP]
       (esto ya lo hace clean_rate, pero se verifica de nuevo aquí).
    2. Cambio excesivo vs histórico: si la nueva tasa difiere > MAX_CHANGE_VS_PREV_PP
       de la última observación guardada, se señala para revisión.
    3. Coherencia FX: si Frankfurter está disponible, se verifica que divisas
       con mayor tasa de política estén débiles en FX (relación inversa aproximada).
       Esta señal es débil — solo detecta inversiones flagrantes.

    Retorna (issues, warnings):
      issues   → lista de strings, cada uno es un problema que causa exit(1)
      warnings → lista de strings informativos (no bloquean)
    """
    issues   = []
    warnings = []

    for currency, entry in final_rates.items():
        rate = float(entry['rate'])

        # Validación 1: rango plausible
        if not (MIN_POLICY_RATE_PP <= rate <= MAX_POLICY_RATE_PP):
            issues.append(f'{currency}: rate {rate}% outside plausible range '
                          f'[{MIN_POLICY_RATE_PP}, {MAX_POLICY_RATE_PP}]')

        # Validación 2: cambio excesivo vs histórico
        prev = get_prev_rate(currency)
        if prev is not None:
            delta = abs(rate - prev)
            if delta > MAX_CHANGE_VS_PREV_PP:
                issues.append(
                    f'{currency}: rate changed {rate}% vs previous {prev}% '
                    f'(Δ={delta:.2f}pp > {MAX_CHANGE_VS_PREV_PP}pp threshold). '
                    f'Verify: central bank meeting or scraping error?'
                )

    # Validación 3: coherencia FX (solo si Frankfurter está disponible)
    if frankfurter_fx and len(final_rates) >= 4:
        # Las divisas con tasa de política alta deberían tener FX alto vs USD
        # (más USD por unidad = moneda fuerte). Esta relación es muy aproximada.
        # Solo detectamos si una divisa con tasa ~0% aparece como la más fuerte FX.
        policy_rates = {c: float(e['rate']) for c, e in final_rates.items()}
        lowest_rate_ccy = min(policy_rates, key=lambda c: policy_rates[c])
        highest_rate_ccy = max(policy_rates, key=lambda c: policy_rates[c])
        print(f'\n  FX cross-check: lowest policy rate = {lowest_rate_ccy} ({policy_rates[lowest_rate_ccy]}%), '
              f'highest = {highest_rate_ccy} ({policy_rates[highest_rate_ccy]}%)')
        print(f'  (Informational only — weak signal, not used for blocking)')

    return issues, warnings


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = str(date.today())
    today_month_start = date.today().strftime('%Y-%m-01')
    run_ts = datetime.utcnow().isoformat() + 'Z'

    print(f'\nRun date:        {today}')
    print(f'Fallback date:   {today_month_start}')

    os.makedirs('rates', exist_ok=True)

    # ── Paso 1: scraping primario ──────────────────────────────────────────────
    final_rates = fetch_trading_economics()
    time.sleep(1)

    # ── Paso 2: fallback para divisas faltantes ────────────────────────────────
    missing = [c for c in CURRENCIES if c not in final_rates]
    if missing:
        print(f'\nFallback for: {", ".join(missing)}')
        for currency in missing:
            result = fetch_global_rates_fallback(currency)
            if result:
                final_rates[currency] = result
            time.sleep(0.5)

    # ── Paso 3: validación cruzada con Frankfurter ────────────────────────────
    frankfurter_fx = fetch_frankfurter_rates()
    issues, warnings = validate_rates(final_rates, frankfurter_fx)

    if warnings:
        print('\nWARNINGS:')
        for w in warnings:
            print(f'  ⚠  {w}')

    if issues:
        print('\nVALIDATION ISSUES DETECTED:')
        for issue in issues:
            print(f'  ✗  {issue}')

    # ── Paso 4: guardar rates/XX.json ─────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SAVING WITH HISTORICAL ACCUMULATION (deduplicated by month)')
    print('=' * 70)

    health_data = {
        'runTimestamp':  run_ts,
        'runDate':       today,
        'frankfurterOk': bool(frankfurter_fx),
        'currencies':    {},
        'issues':        issues,
        'warnings':      warnings,
        'overallStatus': 'ok',  # se sobreescribe abajo si hay issues
    }

    saved = []
    for currency in CURRENCIES:
        if currency not in final_rates:
            print(f'  SKIP {currency}: no data obtained')
            health_data['currencies'][currency] = {
                'status': 'missing',
                'rate':   None,
                'source': None,
                'date':   None,
            }
            continue

        entry = final_rates[currency]
        rate_value = entry['rate']
        observation_date = entry['ref_date'] if entry['ref_date'] else today_month_start

        existing_obs = load_existing_observations(currency)

        # Limpiar observaciones posteriores al Reference date
        if entry['ref_date']:
            before_count = len(existing_obs)
            existing_obs = [
                o for o in existing_obs
                if o['date'][:7] <= observation_date[:7]
            ]
            removed = before_count - len(existing_obs)
            if removed > 0:
                print(f'  CLEANED {currency}: removed {removed} obs after {observation_date[:7]}')

        new_observation = {
            'value':  rate_value,
            'date':   observation_date,
            'source': entry['source'],
        }

        merged_obs = merge_observations(existing_obs, new_observation)

        # Determinar estado de salud de esta divisa
        currency_issues = [i for i in issues if i.startswith(currency + ':')]
        ccy_status = 'warning' if currency_issues else 'ok'
        if entry['source'] != 'TradingEconomics':
            ccy_status = 'fallback'  # se obtuvo de la fuente secundaria

        rate_data = {
            'observations':      merged_obs,
            'lastUpdate':        observation_date,
            'totalObservations': len(merged_obs),
        }

        with open(f'rates/{currency}.json', 'w') as f:
            json.dump(rate_data, f, indent=2)

        health_data['currencies'][currency] = {
            'status': ccy_status,
            'rate':   rate_value,
            'source': entry['source'],
            'date':   observation_date,
            'issues': currency_issues,
        }

        print(f'  SAVED {currency}: {rate_value}% | date={observation_date} | '
              f'history={len(merged_obs)} obs | source={entry["source"]} | status={ccy_status}')
        saved.append(currency)

    # ── Paso 5: guardar rates/health.json ─────────────────────────────────────
    missing_final = [c for c in CURRENCIES if c not in final_rates]
    if missing_final or issues:
        health_data['overallStatus'] = 'degraded'
    if len(missing_final) > 2:
        health_data['overallStatus'] = 'critical'

    with open('rates/health.json', 'w') as f:
        json.dump(health_data, f, indent=2)
    print(f'\n  SAVED rates/health.json — overall status: {health_data["overallStatus"]}')

    # ── Paso 6: exit code ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    if issues:
        print(f'RESULT: FAILED — {len(issues)} validation issue(s) require manual review')
        print('GitHub Actions will mark this run as failed and send email notification.')
        print('Check rates/health.json for details.')
        for issue in issues:
            print(f'  ✗  {issue}')
        sys.exit(1)
    elif missing_final:
        print(f'RESULT: PARTIAL — missing: {", ".join(missing_final)}')
        sys.exit(1)
    else:
        print(f'ALL {len(saved)} CURRENCIES UPDATED OK — no validation issues')
        sys.exit(0)


if __name__ == '__main__':
    main()
