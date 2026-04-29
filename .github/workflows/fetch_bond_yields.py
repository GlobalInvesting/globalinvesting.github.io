#!/usr/bin/env python3
"""
fetch_bond_yields.py  v1.0  —  Lightweight bond yield pipeline (public repo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Writes extended-data/{USD,EUR,GBP,JPY}.json with the 5 yield values consumed
by the live terminal and the AI narrative pipeline:

  USD.json → bond10y (US 10Y), bond2y (US 2Y), bond5y (US 5Y), vix
  EUR.json → bond10y (DE 10Y / Bund)
  GBP.json → bond10y (UK 10Y / Gilt)
  JPY.json → bond10y (JP 10Y / JGB)

Data sources (all public, no API keys required):
  US 10Y / VIX  → ohlc-data/us10y.json / ohlc-data/vix.json
                   (already fetched daily by fetch_ohlc.py via yfinance ^TNX / ^VIX)
  US 2Y         → FRED public CSV: fredgraph.csv?id=DGS2  (no key)
  US 5Y         → FRED public CSV: fredgraph.csv?id=DGS5  (no key)
  DE 10Y        → ECB SDMX daily yield curve (no key)
                   Fallback: DBnomics/ECB
  UK 10Y        → BOE SDIE daily CSV (no key)
                   Fallback: DBnomics/BOE
  JP 10Y        → ECB Financial Markets SDMX (monthly, no key)
                   Fallback: OECD SDMX (monthly, no key)

Design notes:
  - No FRED_API_KEY required — uses the public fredgraph.csv endpoint.
  - Reads us10y/vix from ohlc-data/ (already on disk from fetch_ohlc.py)
    rather than hitting yfinance again. This avoids redundant network calls
    and ensures the two files are always in sync.
  - Only writes the fields that are actually consumed downstream:
      generate_narrative_signals.py  → bond10y (USD/EUR/GBP/JPY), bond2y, bond5y
      fetch_intraday_quotes.py       → bond2y (PASO 4, replaces ^IRX proxy), bond10y/vix fallback
      dashboard.js                   → bond10y (USD/EUR/JPY), bond2y, bond5y, vix (all fallback)
  - Fields NOT written: rateMomentum, consumerConfidence, capitalFlows, gdpGrowth,
    wageGrowth, tradeBalance, businessConfidence, inflationExpectations, fdi —
    none of these are consumed by any live consumer as of v7.49.4.

Replaces: engine/.github/workflows/update-extended-data.yml (private repo, 1×/day)
Runs in:  site/.github/workflows/update-bond-yields.yml    (public repo,  1×/day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import csv
import json
import os
import sys
from datetime import date, datetime, timedelta
from io import StringIO

import requests

SITE_DIR  = os.environ.get("SITE_DIR", ".")
OUT_DIR   = os.path.join(SITE_DIR, "extended-data")
OHLC_DIR  = os.path.join(SITE_DIR, "ohlc-data")
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; GlobalInvestingBot/2.0)"}
TODAY_STR = str(date.today())
VERSION   = "1.0"
SOURCE    = "FRED-CSV / ECB-SDMX / BOE-SDIE / ohlc-data — v1.0"


# ── OHLC READER ──────────────────────────────────────────────────────────────

def read_ohlc_latest(symbol: str):
    """
    Read the latest daily close from ohlc-data/{symbol}.json.
    Returns (date_str, float) or (None, None).
    ohlc-data files are a JSON array of {time, open, high, low, close, volume}.
    """
    path = os.path.join(OHLC_DIR, f"{symbol}.json")
    try:
        with open(path) as f:
            bars = json.load(f)
        if not bars or not isinstance(bars, list):
            return None, None
        last = bars[-1]
        return last.get("time"), float(last["close"])
    except Exception as e:
        print(f"  [ohlc:{symbol}] {e}")
        return None, None


# ── FRED PUBLIC CSV ───────────────────────────────────────────────────────────

def fred_csv_latest(series_id: str):
    """
    Fetch the latest non-missing observation from FRED's public CSV endpoint.
    No API key required.
    Returns (date_str, float) or (None, None).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [FRED:{series_id}] HTTP {r.status_code}")
            return None, None
        reader = csv.reader(StringIO(r.text))
        last_date, last_val = None, None
        for row in reader:
            if len(row) != 2 or row[0] == "DATE" or row[1] in (".", ""):
                continue
            try:
                last_date = row[0]
                last_val  = float(row[1])
            except ValueError:
                continue
        if last_val is not None:
            return last_date, last_val
        print(f"  [FRED:{series_id}] no valid observations")
        return None, None
    except Exception as e:
        print(f"  [FRED:{series_id}] {e}")
        return None, None


# ── ECB SDMX ─────────────────────────────────────────────────────────────────

def ecb_sdmx_latest(flow: str, key: str, last_n: int = 5):
    """
    Fetch latest from ECB SDMX REST API.
    Returns (date_str, float) or (None, None).
    """
    url    = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
    params = {"lastNObservations": last_n, "format": "jsondata", "detail": "dataonly"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            print(f"  [ECB:{flow}/{key}] HTTP {r.status_code}")
            return None, None
        data        = r.json()
        series_dict = data.get("dataSets", [{}])[0].get("series", {})
        if not series_dict:
            return None, None
        series_data = next(iter(series_dict.values()))
        obs         = series_data.get("observations", {})
        if not obs:
            return None, None
        dims     = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next((d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), None)
        if not time_dim:
            return None, None
        for idx_str, obs_vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = obs_vals[0] if obs_vals else None
            if val is not None and idx < len(time_dim):
                period   = time_dim[idx].get("id", "")
                date_str = period if len(period) == 10 else period + "-01"
                try:
                    return date_str, float(val)
                except (ValueError, TypeError):
                    pass
        return None, None
    except Exception as e:
        print(f"  [ECB:{flow}/{key}] {e}")
        return None, None


# ── BOE SDIE ──────────────────────────────────────────────────────────────────

def boe_sdie_latest(series_code: str):
    """
    Fetch latest from Bank of England Statistical Interactive Database.
    Returns (date_str, float) or (None, None).
    """
    now       = datetime.now()
    date_from = f"01/Jan/{now.year - 1}"
    date_to   = f"{now.day:02d}/{now.strftime('%b')}/{now.year}"
    url       = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    params    = {
        "csv.x":       "yes",
        "Datefrom":    date_from,
        "Dateto":      date_to,
        "SeriesCodes": series_code,
        "CSVF":        "TN",
        "UsingCodes":  "Y",
        "VPD":         "Y",
        "VFD":         "N",
    }
    MONTHS = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,  "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            print(f"  [BOE:{series_code}] HTTP {r.status_code}")
            return None, None
        last_date, last_val = None, None
        for line in r.text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("DATE") or line.startswith('"'):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[1] or parts[1] in (".", ""):
                continue
            # Date format: "21 Mar 2026" → "2026-03-21"
            try:
                day_str, mon_str, yr_str = parts[0].split()
                mo = MONTHS.get(mon_str)
                if mo:
                    last_date = f"{yr_str}-{mo:02d}-{int(day_str):02d}"
                    last_val  = float(parts[1])
            except Exception:
                continue
        if last_val is not None:
            return last_date, last_val
        print(f"  [BOE:{series_code}] no valid observations in response")
        return None, None
    except Exception as e:
        print(f"  [BOE:{series_code}] {e}")
        return None, None


# ── DBnomics FALLBACK ─────────────────────────────────────────────────────────

def dbnomics_latest(provider: str, dataset: str, series_code: str):
    """
    Fetch latest from DBnomics public API. Returns (date_str, float) or (None, None).
    Used as fallback for ECB/BOE when primary sources are unavailable.
    """
    url    = f"https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series_code}"
    params = {"observations": 1, "last_n_periods": 24, "format": "json"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            return None, None
        docs = r.json().get("series", {}).get("docs", [])
        if not docs:
            return None, None
        s       = docs[0]
        periods = s.get("period", [])
        values  = s.get("value", [])
        for period, val in reversed(list(zip(periods, values))):
            if val is not None:
                date_str = period if len(period) == 10 else period + "-01"
                try:
                    return date_str, float(val)
                except (ValueError, TypeError):
                    pass
        return None, None
    except Exception as e:
        print(f"  [DBnomics:{provider}/{dataset}/{series_code}] {e}")
        return None, None


# ── PER-CURRENCY FETCHERS ─────────────────────────────────────────────────────

def fetch_usd():
    """
    USD: US10Y and VIX from ohlc-data (already written by fetch_ohlc.py).
         US2Y and US5Y from FRED public CSV (DGS2, DGS5 — no key required).
    """
    dt10, us10y = read_ohlc_latest("us10y")
    dt_vix, vix = read_ohlc_latest("vix")
    dt2,  us2y  = fred_csv_latest("DGS2")
    dt5,  us5y  = fred_csv_latest("DGS5")

    print(f"  [USD] US10Y={us10y} ({dt10})  US2Y={us2y} ({dt2})  US5Y={us5y} ({dt5})  VIX={vix} ({dt_vix})")

    data  = {}
    dates = {}
    if us10y is not None: data["bond10y"] = round(us10y, 4); dates["bond10y"] = dt10 or TODAY_STR
    if us2y  is not None: data["bond2y"]  = round(us2y,  4); dates["bond2y"]  = dt2  or TODAY_STR
    if us5y  is not None: data["bond5y"]  = round(us5y,  4); dates["bond5y"]  = dt5  or TODAY_STR
    if vix   is not None: data["vix"]     = round(vix,   4); dates["vix"]     = dt_vix or TODAY_STR
    return data, dates


def fetch_eur():
    """
    EUR: DE 10Y (Bund) via ECB SDMX daily yield curve.
    """
    dt, val = ecb_sdmx_latest("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
    if val is None:
        print("  [EUR] ECB SDMX unavailable — trying DBnomics/ECB fallback")
        dt, val = dbnomics_latest("ECB", "YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
    print(f"  [EUR] DE10Y={val} ({dt})")
    data, dates = {}, {}
    if val is not None: data["bond10y"] = round(val, 4); dates["bond10y"] = dt or TODAY_STR
    return data, dates


def fetch_gbp():
    """
    GBP: UK 10Y (Gilt) via BOE SDIE daily.
    Series IUDMNZC: 10-year nominal zero coupon gilt yield.
    """
    dt, val = boe_sdie_latest("IUDMNZC")
    if val is None:
        print("  [GBP] BOE SDIE unavailable — trying DBnomics/BOE fallback")
        dt, val = dbnomics_latest("BOE", "rates", "IUDMNZC")
    print(f"  [GBP] UK10Y={val} ({dt})")
    data, dates = {}, {}
    if val is not None: data["bond10y"] = round(val, 4); dates["bond10y"] = dt or TODAY_STR
    return data, dates


def fetch_jpy():
    """
    JPY: JP 10Y (JGB) via ECB Financial Markets SDMX (monthly nominal yield).
    Correct series: FM.M.JP.JPY.4F.BB.JP10YT_RR.YLDA (nominal — no R_ prefix).
    Fallback: OECD SDMX monthly.
    """
    dt, val = None, None
    for key in ("M.JP.JPY.4F.BB.JP10YT_RR.YLDA", "M.JP.JPY.4F.BB.JP10YT_RR.YLD"):
        dt, val = ecb_sdmx_latest("FM", key)
        if val is not None and -5 < val < 20:
            break
        val = None  # reject implausible values (real yield series returns negatives)

    if val is None:
        print("  [JPY] ECB FM unavailable — trying OECD SDMX fallback")
        try:
            url    = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/M.JPN.IRLTLT01.ST"
            params = {"lastNObservations": 3, "format": "jsondata", "detail": "dataonly"}
            r      = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.ok:
                d_data      = r.json()
                series_dict = d_data.get("dataSets", [{}])[0].get("series", {})
                dims        = d_data.get("structure", {}).get("dimensions", {}).get("observation", [])
                time_dim    = next((d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), None)
                if series_dict and time_dim:
                    obs = next(iter(series_dict.values())).get("observations", {})
                    for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
                        idx  = int(idx_str)
                        oval = vals[0] if vals else None
                        if oval is not None and idx < len(time_dim):
                            period = time_dim[idx].get("id", "")
                            dt     = period if len(period) == 10 else period + "-01"
                            try:
                                val = float(oval)
                                break
                            except (ValueError, TypeError):
                                pass
        except Exception as e:
            print(f"  [JPY] OECD SDMX: {e}")

    print(f"  [JPY] JP10Y={val} ({dt})")
    data, dates = {}, {}
    if val is not None: data["bond10y"] = round(val, 4); dates["bond10y"] = dt or TODAY_STR
    return data, dates


# ── WRITE OUTPUT ──────────────────────────────────────────────────────────────

def write_json(ccy: str, data: dict, dates: dict):
    """Write extended-data/{CCY}.json, preserving existing fields not produced here."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{ccy}.json")

    # Preserve any existing fields we don't produce (forward compat)
    existing = {}
    try:
        with open(path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    out = {
        "lastUpdate":           TODAY_STR,
        "source":               SOURCE,
        "data":  {**existing.get("data",  {}), **data},
        "dates": {**existing.get("dates", {}), **dates},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Written: extended-data/{ccy}.json  ({len(data)} fields)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"fetch_bond_yields.py  v{VERSION}  —  {ts}")
    print(f"{'='*60}\n")

    results = {}

    print("[USD]")
    usd_data, usd_dates = fetch_usd()
    write_json("USD", usd_data, usd_dates)
    results["USD"] = usd_data

    print("\n[EUR]")
    eur_data, eur_dates = fetch_eur()
    write_json("EUR", eur_data, eur_dates)
    results["EUR"] = eur_data

    print("\n[GBP]")
    gbp_data, gbp_dates = fetch_gbp()
    write_json("GBP", gbp_data, gbp_dates)
    results["GBP"] = gbp_data

    print("\n[JPY]")
    jpy_data, jpy_dates = fetch_jpy()
    write_json("JPY", jpy_data, jpy_dates)
    results["JPY"] = jpy_data

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ccy, data in results.items():
        fields = ", ".join(f"{k}={v}" for k, v in data.items())
        print(f"  {ccy}: {fields if fields else '⚠ no data written'}")

    # Exit non-zero only if USD bond10y is missing — it's the most critical field
    if "bond10y" not in results.get("USD", {}):
        print("\n⚠ WARNING: USD bond10y unavailable — US10Y is the primary yield reference.")
        sys.exit(1)

    print(f"\n[Exit] Success.")


if __name__ == "__main__":
    main()
