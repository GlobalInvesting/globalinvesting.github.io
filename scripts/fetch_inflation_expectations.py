#!/usr/bin/env python3
"""
fetch_inflation_expectations.py  —  v4.2
=========================================
Fetch CPI inflation (proxy for inflation expectations) for G8 FX currencies
and patch extended-data/{CCY}.json with the result.

PRIMARY source cascade (per currency):
  USD  →  FRED T5YIE          (5-year breakeven, market-implied, daily)
  EUR  →  FRED T5YIFR         (EUR 5Y5Y breakeven, market-implied, daily)
  G6   →  OECD Data Explorer SDMX CSV
             Endpoint: sdmx.oecd.org/public/rest/data/
                       OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/
             Key: {country}.{M|Q}.N.CPI.PA._T.N.GY
             Format: csvfilewithlabels (TRANSFORMATION=GY = annual % change)
             Lag: ~4–6 weeks from national release
       →  FRED API index series (OECD MEI mirror, ~12–14 month structural lag)
       →  World Bank API (annual, always >12 months old)

Architecture note (v4.2):
  FRED OECD MEI series (GBRCPIALLMINMEI etc.) have a ~12–14 month structural lag
  as of 2026 — they are updated ~annually by OECD MEI and FRED mirrors them
  with no faster update cycle. They will always fail the 120-day stale guard.
  OECD Data Explorer (sdmx.oecd.org) uses the same underlying data but updates
  within ~4–6 weeks of national release — the correct primary source.

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
    "User-Agent": "GlobalInvesting-FX-Terminal/4.2 (+https://globalinvesting.github.io)"
}

STALE_DAYS = {
    "monthly":   120,
    "quarterly": 210,   # AUD, NZD quarterly CPI
    "market":     10,   # USD, EUR breakevens
}

OECD_COUNTRY = {
    "GBP": "GBR",
    "JPY": "JPN",
    "AUD": "AUS",
    "CAD": "CAN",
    "CHF": "CHE",
    "NZD": "NZL",
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
    return requests.get(url, headers=h, timeout=20, **kwargs)


def _parse_date(s):
    s = str(s).strip()
    if len(s) == 7:
        s += "-01"
    return datetime.date.fromisoformat(s)


def _is_stale(obs_date, freq):
    threshold = STALE_DAYS.get(freq, 120)
    return (datetime.date.today() - obs_date).days > threshold


def _yoy_from_index(series):
    if len(series) < 2:
        return None
    last_date, last_val = series[-1]
    target_prior = datetime.date(last_date.year - 1, last_date.month, 1)
    best = None
    for d, v in series:
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
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&sort_order=desc&limit=5&file_type=json"
        f"&api_key={FRED_API_KEY}"
    )
    try:
        r = _get(url)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        for o in obs:
            if o.get("value") not in (".", ""):
                val = float(o["value"])
                date = _parse_date(o["date"])
                if not _is_stale(date, "market"):
                    print(f"    ✓ {currency}: {val:.4f}% ({date}) [{label}]")
                    return val, date, label
        print(f"    ✗ {currency}: {label} — no recent observations")
        return None
    except Exception as e:
        print(f"    ✗ {currency}: {label} error — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 PRIMARY  —  OECD Data Explorer SDMX CSV
# Key structure: {country}.{M|Q}.N.CPI.PA._T.N.GY
# TRANSFORMATION=GY = annual % change (YoY)
# AUS and NZD: quarterly (Q); all others: monthly (M)
# ---------------------------------------------------------------------------

def fetch_oecd_explorer(currency):
    country = OECD_COUNTRY[currency]
    freq_code = "Q" if currency in ("AUD", "NZD") else "M"
    key = f"{country}.{freq_code}.N.CPI.PA._T.N.GY"
    url = (
        f"https://sdmx.oecd.org/public/rest/data/"
        f"OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/{key}"
        f"?startPeriod=2024-01&dimensionAtObservation=AllDimensions"
        f"&format=csvfilewithlabels"
    )
    label = f"OECD Data Explorer CPI YoY ({country})"
    print(f"    Trying OECD Data Explorer ({country} {freq_code} GY)…")
    try:
        r = _get(url, custom_headers={"Accept": "text/csv"})
        if r.status_code != 200:
            print(f"    ✗ {currency}: OECD Explorer HTTP {r.status_code}")
            return None

        reader = csv.DictReader(StringIO(r.text))
        rows = list(reader)
        if not rows:
            print(f"    ✗ {currency}: OECD Explorer — empty CSV")
            return None

        # Column lookup (case-insensitive)
        headers_lower = {k.strip().lower(): k for k in rows[0].keys()}
        val_col  = headers_lower.get("obs_value") or headers_lower.get("obsvalue")
        time_col = headers_lower.get("time_period") or headers_lower.get("timeperiod")
        if not val_col or not time_col:
            print(f"    ✗ {currency}: OECD Explorer — missing OBS_VALUE/TIME_PERIOD columns "
                  f"(got: {list(rows[0].keys())[:8]})")
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
            print(f"    ✗ {currency}: OECD Explorer — no parseable rows in CSV")
            return None

        entries.sort(key=lambda x: x[0])
        obs_date, obs_val = entries[-1]
        freq = "quarterly" if currency in ("AUD", "NZD") else "monthly"

        if _is_stale(obs_date, freq):
            print(f"    ⚠ {currency}: OECD Explorer obs {obs_date} is "
                  f">{STALE_DAYS[freq]}d old — skipping")
            return None

        print(f"    ✓ {currency}: {obs_val:.4f}% CPI YoY ({obs_date}) [{label}]")
        return obs_val, obs_date, label

    except Exception as e:
        print(f"    ✗ {currency}: OECD Explorer error — {e}")
        return None


# ---------------------------------------------------------------------------
# G6 FALLBACK  —  FRED index series → YoY
# NOTE: FRED OECD MEI has ~12–14 month structural lag (as of 2026).
# Will almost always be rejected by stale guard. Kept as cold spare.
# ---------------------------------------------------------------------------

def fetch_fred_index_yoy(currency):
    if not FRED_API_KEY:
        print(f"    ✗ {currency}: FRED_API_KEY not set — skipping FRED fallback")
        return None
    series_id, freq = FRED_INDEX_SERIES[currency]
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&sort_order=desc&limit=36&file_type=json"
        f"&api_key={FRED_API_KEY}"
    )
    label = f"FRED {series_id}→YoY"
    print(f"    Trying FRED {series_id}…")
    try:
        r = _get(url)
        r.raise_for_status()
        series = []
        for o in r.json().get("observations", []):
            if o.get("value") not in (".", ""):
                series.append((_parse_date(o["date"]), float(o["value"])))
        series.sort(key=lambda x: x[0])
        result = _yoy_from_index(series)
        if result is None:
            print(f"    ✗ {currency}: FRED {series_id} — insufficient data for YoY")
            return None
        yoy, obs_date = result
        print(f"    ~ {currency}: {yoy:.4f}% CPI YoY ({obs_date}) [{label}]")
        if _is_stale(obs_date, freq):
            print(f"    ⚠ {currency}: obs {obs_date} is >{STALE_DAYS[freq]}d old — skipping")
            return None
        return yoy, obs_date, label
    except Exception as e:
        print(f"    ✗ {currency}: FRED {series_id} error — {e}")
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
    print(f"    Trying World Bank…")
    try:
        r = _get(url)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            print(f"    ✗ {currency}: World Bank — no data")
            return None
        for entry in data[1]:
            if entry.get("value") is not None:
                val = float(entry["value"])
                yr = int(entry["date"])
                obs_date = datetime.date(yr, 6, 15)
                print(f"    ~ {currency}: {val:.4f}% ({obs_date}) [{label}]")
                if _is_stale(obs_date, "monthly"):
                    print(f"    ⚠ {currency}: obs {obs_date} is >{STALE_DAYS['monthly']}d old — skipping")
                    return None
                return val, obs_date, label
        print(f"    ✗ {currency}: World Bank — all values null")
        return None
    except Exception as e:
        print(f"    ✗ {currency}: World Bank error — {e}")
        return None


# ---------------------------------------------------------------------------
# Patch extended-data/{CCY}.json
# ---------------------------------------------------------------------------

def patch_extended_data(site_root, currency, val, date):
    """Update only inflationExpectations in extended-data/{CCY}.json, preserving all other fields."""
    path = os.path.join(site_root, "extended-data", f"{currency}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
        else:
            d = {"data": {}, "dates": {}}
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if "data" not in d:
            d["data"] = {}
        if "dates" not in d:
            d["dates"] = {}

        d["data"]["inflationExpectations"] = round(val, 4)
        d["dates"]["inflationExpectations"] = str(date)

        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"    ✓ Patched {path}")
    except Exception as e:
        print(f"    ⚠ Could not patch {path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Site root is either specified as argv[1] or inferred relative to this script
    if len(sys.argv) > 1:
        site_root = sys.argv[1]
    else:
        # Default: script lives in engine/scripts/, site is ../site or sibling 'site'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try ../site first (GHA checkout layout), then ../.. (direct layout)
        candidates = [
            os.path.join(script_dir, "..", "..", "site"),  # engine/scripts/ → site/
            os.path.join(script_dir, ".."),                # fallback
        ]
        site_root = next(
            (p for p in candidates if os.path.isdir(os.path.join(p, "extended-data"))),
            candidates[0]
        )

    print("=" * 60)
    print("INFLATION EXPECTATIONS — G8 currencies")
    print(f"Run: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Site root: {os.path.abspath(site_root)}")
    print("=" * 60)

    results = {}
    failures = []

    # USD
    print("[USD]")
    r = fetch_fred_breakeven("T5YIE", "USD", "FRED T5YIE")
    if r:
        results["USD"] = r
    else:
        failures.append("USD")

    # EUR
    print("[EUR]")
    r = fetch_fred_breakeven("T5YIFR", "EUR", "FRED T5YIFR")
    if r:
        results["EUR"] = r
    else:
        failures.append("EUR")

    # G6
    for currency in ["GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
        print(f"[{currency}]")
        r = fetch_oecd_explorer(currency)
        if r is None:
            r = fetch_fred_index_yoy(currency)
        if r is None:
            r = fetch_world_bank(currency)
        if r:
            results[currency] = r
        else:
            failures.append(currency)
            print(f"Warning: fetch_inflation_expectations: {currency} stale or unavailable",
                  file=sys.stderr)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ccy, (val, date, src) in sorted(results.items()):
        sign = "+" if val >= 0 else ""
        print(f"  {ccy:5s} {sign}{val:.4f}%  ({date})  [{src}]")
    if failures:
        print(f"  Skipped/failed: {', '.join(failures)} ({len(failures)} issue(s))")

    # Patch extended-data files
    print()
    print("Patching extended-data/…")
    for ccy, (val, date, _src) in results.items():
        patch_extended_data(site_root, ccy, val, date)

    # Exit code
    g6_failures = [c for c in failures if c not in ("USD", "EUR")]
    if len(g6_failures) >= 3:
        print(
            "Error: fetch_inflation_expectations: >=3 G6 currencies failed — likely upstream outage",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
