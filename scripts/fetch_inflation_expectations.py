"""
fetch_inflation_expectations.py  v2.0
──────────────────────────────────────
Fetches inflation expectations for all G8 currencies and writes
inflationExpectations into extended-data/{CCY}.json (non-destructive patch).

Runs weekly (Mondays 06:00 UTC) from the PUBLIC site repo — no API keys required.

CHANGELOG v2.0
──────────────
Root cause fixed: The v1.1 cascade relied on economic-data/{CCY}.json as its
primary source for G6. That file was maintained by update_pmi_from_calendar.py,
which in turn depended on calendar.json — a file produced by update-extended-data.yml
(disabled in v7.50.0). calendar.json froze at 2026-03-28, causing economic-data/ to
freeze too. Result: 0.3–2.3pp stale values in the modal for GBP/JPY/AUD/CAD/CHF/NZD.

Fix: replaced the broken economic-data/ dependency with DIRECT API calls to
authoritative stat offices for each G6 currency via OECD SDMX REST API:
  GBP  → OECD SDMX MEI CPI YoY monthly (GBR) — same API domain as fetch_bond_yields.py
  JPY  → OECD SDMX MEI CPI YoY monthly (JPN)
  AUD  → OECD SDMX MEI CPI YoY quarterly (AUS) → monthly fallback
  CAD  → OECD SDMX MEI CPI YoY monthly (CAN)
  CHF  → OECD SDMX MEI CPI YoY monthly (CHE)
  NZD  → OECD SDMX MEI CPI YoY quarterly (NZL) → monthly fallback

The OECD SDMX endpoint (sdmx.oecd.org/public/rest/) is confirmed working in
GitHub Actions — same domain used successfully by fetch_bond_yields.py for JPY 10Y.

WHY ALL 8 (INCLUDING USD AND EUR)
──────────────────────────────────
The Real Rate Carry modal fetches T5YIE (USD) and T5YIFR (EUR) live from FRED on
every open. This workflow serves two additional purposes:
  1. Fallback: if the live FRED fetch fails, the modal falls back to extended-data.
  2. Audit trail: weekly batch values provide a second independent data point.

SOURCE CASCADE PER CURRENCY
────────────────────────────
  USD  → FRED T5YIE    (5Y breakeven, daily, market-implied)
  EUR  → FRED T5YIFR   (EUR 5Y5Y inflation swap, daily, market-implied)
           ECB HICP YoY (Eurozone flash, monthly) [fallback]
  GBP  → OECD SDMX MEI CPALTT01 monthly (GBR)   [primary — <30d lag]
           ECB HICP flash YoY (GB area)           [fallback]
           FRED CPGRLE01GBM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]
  JPY  → OECD SDMX MEI CPALTT01 monthly (JPN)   [primary]
           FRED CPALTT01JPM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]
  AUD  → OECD SDMX MEI CPALTT01 quarterly (AUS) [primary]
           OECD SDMX MEI CPALTT01 monthly (AUS)  [fallback]
           FRED CPALTT01AUM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]
  CAD  → OECD SDMX MEI CPALTT01 monthly (CAN)   [primary]
           FRED CPALTT01CAM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]
  CHF  → OECD SDMX MEI CPALTT01 monthly (CHE)   [primary]
           FRED CPALTT01CHM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]
  NZD  → OECD SDMX MEI CPALTT01 quarterly (NZL) [primary]
           OECD SDMX MEI CPALTT01 monthly (NZL)  [fallback]
           FRED CPALTT01NZM659N YoY direct        [fallback]
           World Bank FP.CPI.TOTL.ZG              [final fallback]

STALE GUARD
───────────
If the freshest observation for a currency is older than 120 days, the script
emits ::warning and skips that currency (preserves existing data rather than
overwriting with clearly outdated values).

EXIT CODES
──────────
  0 — all currencies updated or skipped with warnings
  1 — ≥3 failures (likely upstream outage — alert the operator)
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO

try:
    import requests
except ImportError:
    print("::error::requests not installed — run: pip install requests --break-system-packages")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────

SITE_DIR = os.environ.get("SITE_DIR", ".")
OUT_DIR  = os.path.join(SITE_DIR, "extended-data")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "globalinvesting-bot/2.0 (https://globalinvesting.github.io)",
    "Accept":     "application/json, text/csv;q=0.9, */*;q=0.8",
}

# ── GitHub Actions annotation helpers ────────────────────────────────────────

def _gha_warning(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)

def _gha_error(msg: str) -> None:
    print(f"::error::{msg}", flush=True)

MAX_AGE_DAYS = 120   # skip writing if observation is older than this

# ── FRED CSV helper (no API key) ─────────────────────────────────────────────

def fred_csv(series_id: str) -> list:
    """
    Fetch FRED series via public CSV endpoint (no API key required).
    Returns list of (datetime, float) sorted descending (newest first).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"    FRED {series_id}: HTTP {r.status_code}")
            return []
        rows = []
        for row in csv.reader(StringIO(r.text)):
            if len(row) != 2 or row[0] == "DATE" or row[1].strip() in (".", "", "NA"):
                continue
            try:
                rows.append((datetime.strptime(row[0].strip(), "%Y-%m-%d"), float(row[1])))
            except (ValueError, TypeError):
                continue
        rows.sort(key=lambda x: x[0], reverse=True)
        return rows
    except Exception as e:
        print(f"    FRED {series_id}: {e}")
        return []


# ── OECD SDMX CPI YoY helper ─────────────────────────────────────────────────

def oecd_cpi_yoy(country_code: str, freq: str = "M"):
    """
    Fetch CPI YoY % from OECD SDMX REST API.
    Uses MEI (Main Economic Indicators) CPALTT01 series = CPI all items YoY.
    Dataflow: OECD.SDD.STES,DSD_KEI@DF_KEI,4.0

    country_code: GBR, JPN, AUS, CAN, CHE, NZL
    freq: 'M' (monthly) or 'Q' (quarterly)

    Returns (date_str YYYY-MM-DD, value) or (None, None).
    """
    key = f"OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/{freq}.{country_code}.CPALTT01.GY"
    try:
        r = requests.get(
            f"https://sdmx.oecd.org/public/rest/data/{key}",
            params={"lastNObservations": 3, "format": "jsondata", "detail": "dataonly"},
            headers=HEADERS,
            timeout=20,
        )
        if not r.ok:
            print(f"    OECD SDMX {country_code} ({freq}): HTTP {r.status_code}")
            return None, None
        data = r.json()
        series_dict = data.get("dataSets", [{}])[0].get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next(
            (d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), []
        )
        if not series_dict or not time_dim:
            print(f"    OECD SDMX {country_code} ({freq}): empty response")
            return None, None
        series = next(iter(series_dict.values()))
        obs = series.get("observations", {})
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                # Handle quarterly period strings like "2026-Q1"
                if "Q" in period:
                    qmap = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
                    parts = period.split("-")
                    if len(parts) == 2:
                        month = qmap.get(parts[1], "03")
                        date_str = f"{parts[0]}-{month}-15"
                    else:
                        date_str = period
                elif len(period) == 7:  # YYYY-MM
                    date_str = period + "-01"
                else:
                    date_str = period
                try:
                    return date_str, round(float(val), 4)
                except (ValueError, TypeError):
                    continue
        return None, None
    except Exception as e:
        print(f"    OECD SDMX {country_code} ({freq}): {e}")
        return None, None


# ── ECB SDMX HICP helper ─────────────────────────────────────────────────────

def ecb_hicp_yoy(area_code: str = "U2"):
    """
    Fetch HICP YoY % from ECB SDMX (ICP dataset).
    area_code: U2 = Eurozone, GB = UK (Eurostat HICP)
    Returns (date_str YYYY-MM-DD, value) or (None, None).
    """
    key = f"M.{area_code}.N.000000.4.ANR"
    try:
        r = requests.get(
            f"https://data-api.ecb.europa.eu/service/data/ICP/{key}",
            params={"lastNObservations": 3, "format": "jsondata", "detail": "dataonly"},
            headers={**HEADERS, "Accept": "application/json"},
            timeout=20,
        )
        if not r.ok:
            print(f"    ECB HICP {area_code}: HTTP {r.status_code}")
            return None, None
        data = r.json()
        series_dict = data.get("dataSets", [{}])[0].get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next(
            (d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), []
        )
        if not series_dict or not time_dim:
            return None, None
        series = next(iter(series_dict.values()))
        obs = series.get("observations", {})
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                date_str = (period + "-01") if len(period) == 7 else period
                try:
                    return date_str, round(float(val), 4)
                except (ValueError, TypeError):
                    continue
        return None, None
    except Exception as e:
        print(f"    ECB HICP {area_code}: {e}")
        return None, None


# ── World Bank fallback ───────────────────────────────────────────────────────

def wb_cpi_yoy(iso2: str):
    """
    CPI YoY % from World Bank (FP.CPI.TOTL.ZG). Annual — last resort only.
    Returns (date_str YYYY-MM-DD, value) or (None, None).
    """
    url = f"https://api.worldbank.org/v2/country/{iso2}/indicator/FP.CPI.TOTL.ZG"
    try:
        r = requests.get(
            url,
            params={"format": "json", "mrv": 3, "per_page": 5},
            headers=HEADERS,
            timeout=20,
        )
        if not r.ok:
            return None, None
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return None, None
        for entry in payload[1]:
            if entry.get("value") is not None:
                yr = entry.get("date", "")
                return f"{yr}-06-15", round(float(entry["value"]), 4)
    except Exception as e:
        print(f"    WB {iso2}: {e}")
    return None, None


# ── Stale guard ───────────────────────────────────────────────────────────────

def is_stale(date_str):
    if not date_str:
        return True
    try:
        age = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            - datetime.strptime(date_str[:10], "%Y-%m-%d")
        ).days
        return age > MAX_AGE_DAYS
    except (ValueError, TypeError):
        return True


# ── JSON patch ────────────────────────────────────────────────────────────────

def patch_json(ccy: str, ie_val: float, ie_date: str) -> bool:
    """
    Non-destructive update: read existing {CCY}.json, update only the
    inflationExpectations field in data{} and dates{}, write back.
    """
    path = os.path.join(OUT_DIR, f"{ccy}.json")
    try:
        d = json.load(open(path)) if os.path.exists(path) else {}
        d.setdefault("data",  {})
        d.setdefault("dates", {})
        d["data"]["inflationExpectations"]  = ie_val
        d["dates"]["inflationExpectations"] = ie_date
        with open(path, "w") as f:
            json.dump(d, f, separators=(",", ":"))
        return True
    except Exception as e:
        print(f"    patch_json {ccy}: {e}")
        return False


# ── Per-currency fetch functions ──────────────────────────────────────────────

def fetch_usd():
    """USD: FRED T5YIE (5Y breakeven inflation rate, daily, market-implied)."""
    rows = fred_csv("T5YIE")
    if rows:
        dt, val = rows[0]
        if -2.0 < val < 15.0:
            print(f"    ✓ USD: {val:.4f}% ({dt.date()}) [T5YIE]")
            return dt.strftime("%Y-%m-%d"), round(val, 4)
        print(f"    ✗ USD: T5YIE value {val:.4f}% out of range")
    print("    ✗ USD: T5YIE returned no data")
    return None, None


def fetch_eur():
    """EUR: FRED T5YIFR (5Y5Y EUR inflation swap, daily, market-implied).
    Fallback: ECB HICP flash YoY (Eurozone).
    """
    rows = fred_csv("T5YIFR")
    if rows:
        dt, val = rows[0]
        if -2.0 < val < 15.0:
            print(f"    ✓ EUR: {val:.4f}% ({dt.date()}) [T5YIFR]")
            return dt.strftime("%Y-%m-%d"), round(val, 4)
        print(f"    ✗ EUR: T5YIFR value {val:.4f}% out of range")
    print("    ✗ EUR: T5YIFR miss — trying ECB HICP Eurozone")
    dt_ecb, val_ecb = ecb_hicp_yoy("U2")
    if val_ecb is not None and -5.0 < val_ecb < 25.0:
        print(f"    ~ EUR: {val_ecb:.4f}% ({dt_ecb}) [ECB HICP fallback]")
        return dt_ecb, val_ecb
    return None, None


def fetch_gbp():
    """
    GBP CPI YoY cascade:
      1. OECD SDMX MEI CPALTT01 monthly (GBR)
      2. ECB HICP flash (GB area)
      3. FRED CPGRLE01GBM659N YoY direct
      4. World Bank
    """
    dt, val = oecd_cpi_yoy("GBR", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ GBP: {val:.4f}% CPI YoY ({dt}) [OECD SDMX]")
        return dt, val
    print("    ✗ GBP: OECD miss — trying ECB HICP GB")
    dt, val = ecb_hicp_yoy("GB")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ~ GBP: {val:.4f}% CPI YoY ({dt}) [ECB HICP]")
        return dt, val
    print("    ✗ GBP: ECB miss — trying FRED CPGRLE01GBM659N")
    rows = fred_csv("CPGRLE01GBM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ GBP: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ GBP: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("GB")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ GBP: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_jpy():
    """
    JPY CPI YoY cascade:
      1. OECD SDMX MEI CPALTT01 monthly (JPN)
      2. FRED CPALTT01JPM659N YoY direct
      3. World Bank
    """
    dt, val = oecd_cpi_yoy("JPN", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ JPY: {val:.4f}% CPI YoY ({dt}) [OECD SDMX]")
        return dt, val
    print("    ✗ JPY: OECD miss — trying FRED CPALTT01JPM659N")
    rows = fred_csv("CPALTT01JPM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ JPY: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ JPY: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("JP")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ JPY: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_aud():
    """
    AUD CPI YoY cascade (ABS is quarterly):
      1. OECD SDMX MEI CPALTT01 quarterly (AUS)
      2. OECD SDMX MEI CPALTT01 monthly (AUS)   — monthly ABS gauge
      3. FRED CPALTT01AUM659N YoY direct
      4. World Bank
    """
    dt, val = oecd_cpi_yoy("AUS", "Q")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ AUD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX quarterly]")
        return dt, val
    print("    ✗ AUD: OECD quarterly miss — trying OECD monthly")
    dt, val = oecd_cpi_yoy("AUS", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ~ AUD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX monthly]")
        return dt, val
    print("    ✗ AUD: OECD monthly miss — trying FRED CPALTT01AUM659N")
    rows = fred_csv("CPALTT01AUM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ AUD: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ AUD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("AU")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ AUD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_cad():
    """
    CAD CPI YoY cascade:
      1. OECD SDMX MEI CPALTT01 monthly (CAN)
      2. FRED CPALTT01CAM659N YoY direct
      3. World Bank
    """
    dt, val = oecd_cpi_yoy("CAN", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ CAD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX]")
        return dt, val
    print("    ✗ CAD: OECD miss — trying FRED CPALTT01CAM659N")
    rows = fred_csv("CPALTT01CAM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ CAD: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ CAD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CA")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ CAD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_chf():
    """
    CHF CPI YoY cascade:
      1. OECD SDMX MEI CPALTT01 monthly (CHE)
      2. FRED CPALTT01CHM659N YoY direct
      3. World Bank
    """
    dt, val = oecd_cpi_yoy("CHE", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ CHF: {val:.4f}% CPI YoY ({dt}) [OECD SDMX]")
        return dt, val
    print("    ✗ CHF: OECD miss — trying FRED CPALTT01CHM659N")
    rows = fred_csv("CPALTT01CHM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ CHF: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ CHF: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CH")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ CHF: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_nzd():
    """
    NZD CPI YoY cascade (Stats NZ is quarterly):
      1. OECD SDMX MEI CPALTT01 quarterly (NZL)
      2. OECD SDMX MEI CPALTT01 monthly (NZL)
      3. FRED CPALTT01NZM659N YoY direct
      4. World Bank
    """
    dt, val = oecd_cpi_yoy("NZL", "Q")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ✓ NZD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX quarterly]")
        return dt, val
    print("    ✗ NZD: OECD quarterly miss — trying OECD monthly")
    dt, val = oecd_cpi_yoy("NZL", "M")
    if val is not None and -5.0 < val < 25.0:
        print(f"    ~ NZD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX monthly]")
        return dt, val
    print("    ✗ NZD: OECD monthly miss — trying FRED CPALTT01NZM659N")
    rows = fred_csv("CPALTT01NZM659N")
    if rows:
        dt_f, val_f = rows[0]
        if -5.0 < val_f < 25.0:
            print(f"    ~ NZD: {val_f:.4f}% ({dt_f.date()}) [FRED YoY direct]")
            return dt_f.strftime("%Y-%m-%d"), round(val_f, 4)
    print("    ✗ NZD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("NZ")
    if val_wb is not None and -5.0 < val_wb < 25.0:
        print(f"    ~ NZD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


# ── Dispatch table ────────────────────────────────────────────────────────────

FETCH_FN = {
    "USD": fetch_usd,
    "EUR": fetch_eur,
    "GBP": fetch_gbp,
    "JPY": fetch_jpy,
    "AUD": fetch_aud,
    "CAD": fetch_cad,
    "CHF": fetch_chf,
    "NZD": fetch_nzd,
}

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("INFLATION EXPECTATIONS — G8 currencies")
    print(f"Run: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    failures = 0
    results  = {}

    for ccy, fn in FETCH_FN.items():
        print(f"\n[{ccy}]")
        date_str, val = fn()

        if val is None:
            print(f"    ✗ {ccy}: all sources failed — skipping (preserving existing data)")
            _gha_warning(f"fetch_inflation_expectations: {ccy} all sources failed")
            failures += 1
            continue

        if is_stale(date_str):
            print(f"    ⚠ {ccy}: observation {date_str} is >{MAX_AGE_DAYS}d old — skipping")
            _gha_warning(f"fetch_inflation_expectations: {ccy} stale ({date_str})")
            failures += 1
            continue

        if patch_json(ccy, val, date_str):
            results[ccy] = (val, date_str)
        else:
            failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ccy, (val, dt) in sorted(results.items()):
        src = "direct (market-implied)" if ccy in ("USD", "EUR") else "CPI YoY"
        print(f"  {ccy:<4}  {val:+.4f}%  ({dt})  [{src}]")

    if failures:
        skipped = [c for c in FETCH_FN if c not in results]
        print(f"\n  Skipped/failed: {', '.join(skipped)} ({failures} issue(s))")

    if failures >= 3:
        _gha_error("fetch_inflation_expectations: ≥3 currencies failed — likely upstream outage")
        return 1

    print(f"\n  {len(results)}/8 currencies written successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
