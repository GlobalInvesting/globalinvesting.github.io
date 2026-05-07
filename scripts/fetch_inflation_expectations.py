#!/usr/bin/env python3
"""
fetch_inflation_expectations.py  —  v4.3
=========================================
Fetch CPI inflation (proxy for inflation expectations) for G8 FX currencies
and patch extended-data/{CCY}.json with the result.

SOURCE CASCADE (v4.3):
  USD  →  FRED T5YIE          (5-year breakeven, market-implied, daily)
  EUR  →  FRED T5YIFR         (EUR 5Y5Y breakeven, market-implied, daily)
  G6   →  IMF SDMX 3.0 API   (api.imf.org, monthly, lag ~4-6 weeks) [PRIMARY]
       →  OECD Data Explorer  (sdmx.oecd.org, monthly/quarterly) [FALLBACK 1]
       →  FRED MEI index→YoY  (~12-14 month structural lag) [FALLBACK 2]
       →  World Bank annual   (>12 months lag) [LAST RESORT]

WHY v4.3:
  OECD DF_PRICES_ALL (COICOP 1999) does NOT include Japan — JPN returns HTTP 404.
  Japan transitioned to COICOP 2018 and is only available in DF_PRICES_N_CP01_*.
  AUS/NZD quarterly data from OECD uses "2024-Q4" time format which the v4.2
  parser failed to handle, producing "no parseable rows".
  CHF stale: OECD last obs Dec 2025, today May 2026 → 157d > 120d threshold.

  IMF SDMX 3.0 API (api.imf.org) provides monthly CPI index (PCPI_IX) for all
  G6 countries including Japan, with the same ~4-6 week lag as OECD Explorer.
  YoY is computed via 12-month pct change on the index series, matching the
  approach used by Bloomberg and professional data vendors.

IMF API key format:
  {ISO3}.CPI._T.IX.M
  Time periods are returned as "2024-M01" — converted to "2024-01" before parsing.

OECD quarterly date fix:
  "2024-Q4" is now parsed as Dec 1 of that quarter (month = quarter * 3).

Output: patches extended-data/{CCY}.json in the site repo, preserving all
  other fields. Only updates inflationExpectations + dates.inflationExpectations.
"""

import os
import sys
import csv
import json
import datetime
import requests
from io import StringIO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

BASE_HEADERS = {
    "User-Agent": "GlobalInvesting-FX-Terminal/4.3 (+https://globalinvesting.github.io)"
}

STALE_DAYS = {
    "monthly":   120,
    "quarterly": 210,
    "market":     10,
}

# IMF ISO-3 country codes
IMF_COUNTRY = {
    "GBP": "GBR",
    "JPY": "JPN",
    "AUD": "AUS",
    "CAD": "CAN",
    "CHF": "CHE",
    "NZD": "NZL",
}

# OECD country codes + quarterly flag (AUD/NZD are quarterly in DF_PRICES_ALL)
OECD_COUNTRY = {
    "GBP": ("GBR", "M"),
    "JPY": ("JPN", "M"),   # Will 404 — COICOP 2018 only; kept for documentation
    "AUD": ("AUS", "Q"),
    "CAD": ("CAN", "M"),
    "CHF": ("CHE", "M"),
    "NZD": ("NZL", "Q"),
}

FRED_INDEX_SERIES = {
    "GBP": ("GBRCPIALLMINMEI",  "monthly"),
    "JPY": ("JPNCPIALLMINMEI",  "monthly"),
    "AUD": ("AUSCPIALLQINMEI",  "quarterly"),
    "CAD": ("CANCPIALLMINMEI",  "monthly"),
    "CHF": ("CHECPIALLMINMEI",  "monthly"),
    "NZD": ("NZLCPIALLQINMEI",  "quarterly"),
}

WB_COUNTRY = {
    "GBP": "GB", "JPY": "JP", "AUD": "AU",
    "CAD": "CA", "CHF": "CH", "NZD": "NZ",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url, custom_headers=None, **kwargs):
    h = {**BASE_HEADERS, **(custom_headers or {})}
    return requests.get(url, headers=h, timeout=25, **kwargs)


def _parse_date(s):
    """Parse YYYY-MM, YYYY-MM-DD, YYYY-Mxx (IMF), or YYYY-Qx (OECD quarterly)."""
    s = str(s).strip()
    # IMF format: "2024-M01" (len=8) → "2024-01"
    if len(s) == 8 and s[5] == "M":
        s = s[:5] + s[6:]      # "2024-M01" → "2024-01"
    # Quarterly format: "2024-Q4" → last month of that quarter
    if len(s) == 7 and "Q" in s:
        try:
            year = int(s[:4])
            quarter = int(s[-1])
            month = quarter * 3          # Q1→3, Q2→6, Q3→9, Q4→12
            return datetime.date(year, month, 1)
        except ValueError:
            raise ValueError(f"Cannot parse quarterly date: {s!r}")
    # Standard: "YYYY-MM" → "YYYY-MM-01"
    if len(s) == 7:
        s += "-01"
    return datetime.date.fromisoformat(s)


def _is_stale(obs_date, freq):
    threshold = STALE_DAYS.get(freq, 120)
    return (datetime.date.today() - obs_date).days > threshold


def _yoy_from_index(series):
    """Compute YoY % from an index series (list of (date, value) tuples)."""
    if len(series) < 13:
        return None
    series_sorted = sorted(series, key=lambda x: x[0])
    last_date, last_val = series_sorted[-1]
    # Find observation ~12 months prior (within ±92 days)
    target_prior = datetime.date(last_date.year - 1, last_date.month, 1)
    best = None
    for d, v in series_sorted:
        diff = abs((d - target_prior).days)
        if diff <= 92:
            if best is None or diff < best[0]:
                best = (diff, d, v)
    if best is None:
        return None
    _, _, prior_val = best
    if prior_val == 0:
        return None
    return round((last_val / prior_val - 1) * 100, 4), last_date


# ---------------------------------------------------------------------------
# USD / EUR  —  FRED breakevens
# ---------------------------------------------------------------------------

def fetch_fred_breakeven(series_id, currency, label):
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&sort_order=desc&limit=5&file_type=json"
        f"&api_key={FRED_API_KEY}"
    )
    try:
        r = _get(url)
        r.raise_for_status()
        for o in r.json().get("observations", []):
            if o.get("value") not in (".", ""):
                val = float(o["value"])
                date = _parse_date(o["date"])
                if not _is_stale(date, "market"):
                    print(f"    OK {currency}: {val:.4f}% ({date}) [{label}]")
                    return val, date, label
        print(f"    MISS {currency}: {label} — no recent observations")
        return None
    except Exception as e:
        print(f"    ERR {currency}: {label} — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 PRIMARY  —  IMF SDMX 3.0 API (monthly CPI index → YoY)
# Endpoint: api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/~/{key}
# Key:  {ISO3}.CPI._T.IX.M
# Time: ?c[TIME_PERIOD]=ge:YYYY-M01
# Response: CSV with TIME_PERIOD like "2024-M01", OBS_VALUE = index level
# YoY is computed from the index series (12-month pct change).
# Lag: ~4-6 weeks after national release. No API key required.
# ---------------------------------------------------------------------------

def fetch_imf_cpi(currency):
    country = IMF_COUNTRY[currency]
    key = f"{country}.CPI._T.IX.M"
    # Request 26 months to ensure we have a full 12-month YoY window
    start = (datetime.date.today() - datetime.timedelta(days=800)).strftime("%Y-M%m")
    url = (
        f"https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/~/"
        f"{key}?c[TIME_PERIOD]=ge:{start}"
    )
    label = f"IMF CPI {country} (monthly index→YoY)"
    print(f"    Trying IMF SDMX 3.0 ({country})...")
    try:
        r = _get(url, custom_headers={"Accept": "text/csv"})
        if r.status_code != 200:
            print(f"    MISS {currency}: IMF HTTP {r.status_code}")
            return None

        reader = csv.DictReader(StringIO(r.text))
        rows = list(reader)
        if not rows:
            print(f"    MISS {currency}: IMF — empty CSV")
            return None

        headers_lower = {k.strip().lower(): k for k in rows[0].keys()}
        val_col  = headers_lower.get("obs_value") or headers_lower.get("obsvalue")
        time_col = headers_lower.get("time_period") or headers_lower.get("timeperiod")
        if not val_col or not time_col:
            print(f"    MISS {currency}: IMF — missing OBS_VALUE/TIME_PERIOD "
                  f"(cols: {list(rows[0].keys())[:6]})")
            return None

        entries = []
        for row in rows:
            try:
                raw_val = row[val_col].strip()
                raw_time = row[time_col].strip()
                if not raw_val or raw_val in ("..", ".", "NaN", ""):
                    continue
                entries.append((_parse_date(raw_time), float(raw_val)))
            except (ValueError, KeyError):
                continue

        if len(entries) < 13:
            print(f"    MISS {currency}: IMF — only {len(entries)} obs (need ≥13 for YoY)")
            return None

        result = _yoy_from_index(entries)
        if result is None:
            print(f"    MISS {currency}: IMF — YoY calculation failed")
            return None

        yoy, obs_date = result
        if _is_stale(obs_date, "monthly"):
            print(f"    STALE {currency}: IMF obs {obs_date} is >{STALE_DAYS['monthly']}d — skipping")
            return None

        print(f"    OK {currency}: {yoy:.4f}% CPI YoY ({obs_date}) [{label}]")
        return yoy, obs_date, label

    except Exception as e:
        print(f"    ERR {currency}: IMF — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 FALLBACK 1  —  OECD Data Explorer SDMX CSV (direct YoY series)
# Key: {country}.{M|Q}.N.CPI.PA._T.N.GY  (TRANSFORMATION=GY = annual % change)
# AUS/NZD use quarterly (Q); time format "2024-Q4" is now parsed correctly.
# JPN will return 404 (COICOP 2018 only — not in DF_PRICES_ALL).
# CHF: last obs Dec 2025 (~157 days) exceeds 120d threshold. Still tried.
# ---------------------------------------------------------------------------

def fetch_oecd_explorer(currency):
    country, freq_code = OECD_COUNTRY[currency]
    key = f"{country}.{freq_code}.N.CPI.PA._T.N.GY"
    url = (
        f"https://sdmx.oecd.org/public/rest/data/"
        f"OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/{key}"
        f"?startPeriod=2024-01&dimensionAtObservation=AllDimensions"
        f"&format=csvfilewithlabels"
    )
    label = f"OECD CPI YoY ({country})"
    print(f"    Trying OECD Explorer ({country} {freq_code} GY)...")
    try:
        r = _get(url, custom_headers={"Accept": "text/csv"})
        if r.status_code != 200:
            print(f"    MISS {currency}: OECD HTTP {r.status_code}")
            return None

        reader = csv.DictReader(StringIO(r.text))
        rows = list(reader)
        if not rows:
            print(f"    MISS {currency}: OECD — empty CSV")
            return None

        headers_lower = {k.strip().lower(): k for k in rows[0].keys()}
        val_col  = headers_lower.get("obs_value") or headers_lower.get("obsvalue")
        time_col = headers_lower.get("time_period") or headers_lower.get("timeperiod")
        if not val_col or not time_col:
            print(f"    MISS {currency}: OECD — missing columns")
            return None

        entries = []
        for row in rows:
            try:
                raw_val = row[val_col].strip()
                raw_time = row[time_col].strip()
                if not raw_val or raw_val in ("..", ".", "NaN", ""):
                    continue
                entries.append((_parse_date(raw_time), float(raw_val)))
            except (ValueError, KeyError):
                continue

        if not entries:
            print(f"    MISS {currency}: OECD — no parseable rows")
            return None

        entries.sort(key=lambda x: x[0])
        obs_date, obs_val = entries[-1]
        freq = "quarterly" if freq_code == "Q" else "monthly"

        if _is_stale(obs_date, freq):
            print(f"    STALE {currency}: OECD obs {obs_date} is >{STALE_DAYS[freq]}d — skipping")
            return None

        print(f"    OK {currency}: {obs_val:.4f}% CPI YoY ({obs_date}) [{label}]")
        return obs_val, obs_date, label

    except Exception as e:
        print(f"    ERR {currency}: OECD — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 FALLBACK 2  —  FRED index series → YoY
# Structural ~12-14 month lag as of 2026. Will almost always be stale.
# ---------------------------------------------------------------------------

def fetch_fred_index_yoy(currency):
    if not FRED_API_KEY:
        print(f"    SKIP {currency}: FRED_API_KEY not set")
        return None
    series_id, freq = FRED_INDEX_SERIES[currency]
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&sort_order=desc&limit=36&file_type=json"
        f"&api_key={FRED_API_KEY}"
    )
    label = f"FRED {series_id}→YoY"
    print(f"    Trying FRED {series_id}...")
    try:
        r = _get(url)
        r.raise_for_status()
        series = [
            (_parse_date(o["date"]), float(o["value"]))
            for o in r.json().get("observations", [])
            if o.get("value") not in (".", "")
        ]
        result = _yoy_from_index(series)
        if result is None:
            print(f"    MISS {currency}: FRED {series_id} — insufficient data")
            return None
        yoy, obs_date = result
        print(f"    ~ {currency}: {yoy:.4f}% ({obs_date}) [{label}]")
        if _is_stale(obs_date, freq):
            print(f"    STALE {currency}: obs {obs_date} is >{STALE_DAYS[freq]}d — skipping")
            return None
        return yoy, obs_date, label
    except Exception as e:
        print(f"    ERR {currency}: FRED {series_id} — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 LAST RESORT  —  World Bank (annual, lag >12 months)
# ---------------------------------------------------------------------------

def fetch_world_bank(currency):
    country = WB_COUNTRY[currency]
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/FP.CPI.TOTL.ZG"
        f"?format=json&mrv=5&per_page=5"
    )
    label = "World Bank"
    print(f"    Trying World Bank...")
    try:
        r = _get(url)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            print(f"    MISS {currency}: World Bank — no data")
            return None
        for entry in data[1]:
            if entry.get("value") is not None:
                val = float(entry["value"])
                obs_date = datetime.date(int(entry["date"]), 6, 15)
                print(f"    ~ {currency}: {val:.4f}% ({obs_date}) [{label}]")
                if _is_stale(obs_date, "monthly"):
                    print(f"    STALE {currency}: obs {obs_date} is >{STALE_DAYS['monthly']}d — skipping")
                    return None
                return val, obs_date, label
        print(f"    MISS {currency}: World Bank — all values null")
        return None
    except Exception as e:
        print(f"    ERR {currency}: World Bank — {e}")
        return None


# ---------------------------------------------------------------------------
# Patch extended-data/{CCY}.json
# ---------------------------------------------------------------------------

def patch_extended_data(site_root, currency, val, date):
    path = os.path.join(site_root, "extended-data", f"{currency}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
        else:
            d = {"data": {}, "dates": {}}
            os.makedirs(os.path.dirname(path), exist_ok=True)
        d.setdefault("data", {})
        d.setdefault("dates", {})
        d["data"]["inflationExpectations"] = round(val, 4)
        d["dates"]["inflationExpectations"] = str(date)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"    OK patched {path}")
    except Exception as e:
        print(f"    WARN: could not patch {path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        site_root = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "..", "site"),
            os.path.join(script_dir, ".."),
        ]
        site_root = next(
            (p for p in candidates if os.path.isdir(os.path.join(p, "extended-data"))),
            candidates[0]
        )

    print("=" * 60)
    print("INFLATION EXPECTATIONS — G8 currencies  (v4.3)")
    print(f"Run: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Site root: {os.path.abspath(site_root)}")
    print("=" * 60)

    results = {}
    failures = []

    print("[USD]")
    r = fetch_fred_breakeven("T5YIE", "USD", "FRED T5YIE")
    if r:
        results["USD"] = r
    else:
        failures.append("USD")

    print("[EUR]")
    r = fetch_fred_breakeven("T5YIFR", "EUR", "FRED T5YIFR")
    if r:
        results["EUR"] = r
    else:
        failures.append("EUR")

    for currency in ["GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        print(f"[{currency}]")
        # 1. IMF primary
        r = fetch_imf_cpi(currency)
        # 2. OECD fallback
        if r is None:
            r = fetch_oecd_explorer(currency)
        # 3. FRED MEI
        if r is None:
            r = fetch_fred_index_yoy(currency)
        # 4. World Bank
        if r is None:
            r = fetch_world_bank(currency)
        if r:
            results[currency] = r
        else:
            failures.append(currency)
            print(f"Warning: fetch_inflation_expectations: {currency} stale or unavailable",
                  file=sys.stderr)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ccy, (val, date, src) in sorted(results.items()):
        sign = "+" if val >= 0 else ""
        print(f"  {ccy:5s} {sign}{val:.4f}%  ({date})  [{src}]")
    if failures:
        print(f"  Skipped/failed: {', '.join(failures)} ({len(failures)} issue(s))")

    print()
    print("Patching extended-data/...")
    for ccy, (val, date, _src) in results.items():
        patch_extended_data(site_root, ccy, val, date)

    g6_failures = [c for c in failures if c not in ("USD", "EUR")]
    if len(g6_failures) >= 3:
        print("Error: fetch_inflation_expectations: >=3 G6 currencies failed — likely upstream outage",
              file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
