"""
fetch_inflation_expectations.py  v3.0
──────────────────────────────────────
Fetches inflation expectations for all G8 currencies and writes
inflationExpectations into extended-data/{CCY}.json (non-destructive patch).

Runs weekly (Mondays 06:00 UTC) from the PUBLIC site repo — no API keys required.

CHANGELOG v3.0
──────────────
Root cause fixed (v2.0 regression):

v2.0 used OECD SDMX key CPALTT01.GY — but GY is a TRANSFORMATION code, not a
MEASURE code. The DSD_KEI@DF_KEI,4.0 dataflow has a MEASURE dimension, not a
TRANSFORMATION dimension. The correct measure code for the standardised series
is ST (same as IRLTLT01.ST for bond yields — confirmed working in GHA).

Fix: changed OECD SDMX key from CPALTT01.GY → CPALTT01.ST and added
index_to_yoy() to compute YoY % from the raw CPI index observations.
This is identical to what update_extended_data.py v13.0 does with FRED MINMEI/QINMEI.

Additional fixes vs v2.0:
  • Added FRED MINMEI/QINMEI public CSV as cascade step 2 (before World Bank).
    These series are the active OECD MEI equivalents — CPALTT01*M659N ended 2021.
  • Fixed AUD/NZD fallback: CPALTT01AUM659N and CPALTT01NZM659N return 404 because
    they never existed. Replaced with AUSCPIALLQINMEI / NZLCPIALLQINMEI (index-to-YoY).
  • Per-currency stale thresholds: quarterly series (AUD, NZD) get 200d; all others 120d.

SOURCE CASCADE PER CURRENCY
────────────────────────────
  USD  → FRED T5YIE          (5Y breakeven, daily, market-implied)
  EUR  → FRED T5YIFR         (EUR 5Y5Y inflation swap, daily, market-implied)
           ECB HICP YoY      (Eurozone flash, monthly) [fallback]
  GBP  → OECD SDMX CPALTT01.ST monthly (GBR) → index-to-YoY  [primary, <4w lag]
           FRED GBRCPIALLMINMEI → index-to-YoY                 [fallback]
           ECB HICP flash YoY (GB area)                        [fallback]
           World Bank FP.CPI.TOTL.ZG                           [final fallback]
  JPY  → OECD SDMX CPALTT01.ST monthly (JPN) → index-to-YoY  [primary]
           World Bank FP.CPI.TOTL.ZG                           [final fallback]
           (JPNCPIALLMINMEI ended Jun-2021 — not used)
  AUD  → OECD SDMX CPALTT01.ST quarterly (AUS) → index-to-YoY [primary, ~6w lag]
           OECD SDMX CPALTT01.ST monthly (AUS) → index-to-YoY  [fallback]
           FRED AUSCPIALLQINMEI → index-to-YoY                  [fallback]
           World Bank FP.CPI.TOTL.ZG                            [final fallback]
  CAD  → OECD SDMX CPALTT01.ST monthly (CAN) → index-to-YoY  [primary]
           FRED CANCPIALLMINMEI → index-to-YoY                  [fallback]
           World Bank FP.CPI.TOTL.ZG                            [final fallback]
  CHF  → OECD SDMX CPALTT01.ST monthly (CHE) → index-to-YoY  [primary]
           FRED CHECPIALLMINMEI → index-to-YoY                  [fallback]
           World Bank FP.CPI.TOTL.ZG                            [final fallback]
  NZD  → OECD SDMX CPALTT01.ST quarterly (NZL) → index-to-YoY [primary]
           OECD SDMX CPALTT01.ST monthly (NZL) → index-to-YoY  [fallback]
           FRED NZLCPIALLQINMEI → index-to-YoY                  [fallback]
           World Bank FP.CPI.TOTL.ZG                            [final fallback]

STALE GUARD
───────────
Monthly series:  observation must be ≤120 days old.
Quarterly series (AUD, NZD): observation must be ≤200 days old (Q4 release ≈ Jan,
  meaning May run sees Jan data → ~120d; some years slip to Feb → 150d; buffer added).

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
    "User-Agent": "globalinvesting-bot/3.0 (https://globalinvesting.github.io)",
    "Accept":     "application/json, text/csv;q=0.9, */*;q=0.8",
}

# ── GitHub Actions annotation helpers ────────────────────────────────────────

def _gha_warning(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)

def _gha_error(msg: str) -> None:
    print(f"::error::{msg}", flush=True)

# Per-currency stale thresholds (days)
MAX_AGE = {
    "USD": 10,    # market-implied daily — warn if >10 days stale
    "EUR": 10,
    "GBP": 120,
    "JPY": 120,
    "AUD": 200,   # quarterly CPI: Q4→Jan release, May run = ~120d; buffer to 200d
    "CAD": 120,
    "CHF": 120,
    "NZD": 200,   # quarterly CPI: same as AUD
}

# ── Index-to-YoY helper ───────────────────────────────────────────────────────

def index_to_yoy(obs: list) -> tuple:
    """
    Compute CPI YoY % from a list of (date_str, index_value) sorted descending.
    Requires enough observations spanning ~12 months.

    Returns (date_str, yoy_pct) or (None, None).
    """
    if len(obs) < 2:
        return None, None

    parsed = []
    for dt_s, v in obs:
        try:
            parsed.append((datetime.strptime(dt_s[:10], "%Y-%m-%d"), float(v)))
        except (ValueError, TypeError):
            continue

    if not parsed:
        return None, None

    parsed.sort(key=lambda x: x[0], reverse=True)
    d_curr, v_curr = parsed[0]

    # Find observation closest to 12 months prior (75-day tolerance for quarterly)
    target = d_curr - timedelta(days=365)
    best_val, best_diff = None, float("inf")
    for d, v in parsed[1:]:
        diff = abs((d - target).days)
        if diff < best_diff:
            best_diff, best_val = diff, v

    if best_val is None or best_diff > 75 or best_val == 0:
        return None, None

    yoy = round((v_curr / best_val - 1) * 100, 4)
    return d_curr.strftime("%Y-%m-%d"), yoy


# ── FRED CSV helper (no API key) ─────────────────────────────────────────────

def fred_csv_series(series_id: str, start_year: int = 2019) -> list:
    """
    Fetch FRED series via public CSV (no API key required).
    Returns list of (date_str, float) sorted descending (newest first).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code != 200:
            print(f"    FRED {series_id}: HTTP {r.status_code}")
            return []
        rows = []
        for row in csv.reader(StringIO(r.text)):
            if len(row) != 2 or row[0] == "DATE" or row[1].strip() in (".", "", "NA"):
                continue
            try:
                dt = datetime.strptime(row[0].strip(), "%Y-%m-%d")
                if dt.year < start_year:
                    continue
                rows.append((dt.strftime("%Y-%m-%d"), float(row[1])))
            except (ValueError, TypeError):
                continue
        rows.sort(key=lambda x: x[0], reverse=True)
        return rows
    except Exception as e:
        print(f"    FRED {series_id}: {e}")
        return []


def fred_index_to_yoy(series_id: str) -> tuple:
    """Fetch FRED index series, compute YoY %."""
    rows = fred_csv_series(series_id, start_year=2019)
    if not rows:
        return None, None
    return index_to_yoy(rows)


# ── OECD SDMX CPI Index helper ───────────────────────────────────────────────

def oecd_cpi_index(country_code: str, freq: str = "M") -> list:
    """
    Fetch CPI standardised index (CPALTT01.ST) from OECD SDMX REST API.
    Dataflow: OECD.SDD.STES,DSD_KEI@DF_KEI,4.0
    MEASURE=ST: standardised series value (index, base 2015=100).

    The v2.0 key CPALTT01.GY was wrong: GY is a transformation code, not a
    MEASURE code. ST is the correct measure for the index series.

    country_code: GBR, JPN, AUS, CAN, CHE, NZL
    freq: 'M' (monthly) or 'Q' (quarterly)

    Returns list of (date_str, float) sorted descending (newest first).
    """
    key = f"OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/{freq}.{country_code}.CPALTT01.ST"
    try:
        r = requests.get(
            f"https://sdmx.oecd.org/public/rest/data/{key}",
            params={"lastNObservations": 16, "format": "jsondata", "detail": "dataonly"},
            headers=HEADERS,
            timeout=25,
        )
        if not r.ok:
            print(f"    OECD SDMX {country_code} ({freq}): HTTP {r.status_code}")
            return []
        data = r.json()
        series_dict = data.get("dataSets", [{}])[0].get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next(
            (d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), []
        )
        if not series_dict or not time_dim:
            print(f"    OECD SDMX {country_code} ({freq}): empty response")
            return []
        series = next(iter(series_dict.values()))
        obs = series.get("observations", {})
        rows = []
        for idx_str, vals in obs.items():
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                if "Q" in period:
                    qmap = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
                    parts = period.split("-")
                    month = qmap.get(parts[1], "03") if len(parts) == 2 else "03"
                    date_str = f"{parts[0]}-{month}-15"
                elif len(period) == 7:
                    date_str = period + "-01"
                else:
                    date_str = period
                try:
                    rows.append((date_str, float(val)))
                except (ValueError, TypeError):
                    continue
        rows.sort(key=lambda x: x[0], reverse=True)
        return rows
    except Exception as e:
        print(f"    OECD SDMX {country_code} ({freq}): {e}")
        return []


def oecd_cpi_yoy(country_code: str, freq: str = "M") -> tuple:
    """Fetch OECD SDMX CPI index and compute YoY %."""
    rows = oecd_cpi_index(country_code, freq)
    if not rows:
        return None, None
    return index_to_yoy(rows)


# ── ECB SDMX HICP helper ─────────────────────────────────────────────────────

def ecb_hicp_yoy(area_code: str = "U2") -> tuple:
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
            timeout=25,
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

def wb_cpi_yoy(iso2: str) -> tuple:
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
            timeout=25,
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

def is_stale(date_str: str, ccy: str) -> bool:
    if not date_str:
        return True
    try:
        age = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            - datetime.strptime(date_str[:10], "%Y-%m-%d")
        ).days
        return age > MAX_AGE.get(ccy, 120)
    except (ValueError, TypeError):
        return True


# ── JSON patch ────────────────────────────────────────────────────────────────

def patch_json(ccy: str, ie_val: float, ie_date: str) -> bool:
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


# ── Sanity check ─────────────────────────────────────────────────────────────

def _valid(val, ccy: str) -> bool:
    """Reject obviously wrong values."""
    if val is None:
        return False
    return -5.0 < val < 25.0


# ── Per-currency fetch functions ──────────────────────────────────────────────

def fetch_usd() -> tuple:
    rows = fred_csv_series("T5YIE")
    if rows:
        dt_s, val = rows[0]
        if _valid(val, "USD"):
            print(f"    ✓ USD: {val:.4f}% ({dt_s}) [T5YIE]")
            return dt_s, round(val, 4)
    print("    ✗ USD: T5YIE no data")
    return None, None


def fetch_eur() -> tuple:
    rows = fred_csv_series("T5YIFR")
    if rows:
        dt_s, val = rows[0]
        if _valid(val, "EUR"):
            print(f"    ✓ EUR: {val:.4f}% ({dt_s}) [T5YIFR]")
            return dt_s, round(val, 4)
    print("    ✗ EUR: T5YIFR miss — trying ECB HICP Eurozone")
    dt_ecb, val_ecb = ecb_hicp_yoy("U2")
    if val_ecb is not None and _valid(val_ecb, "EUR"):
        print(f"    ~ EUR: {val_ecb:.4f}% ({dt_ecb}) [ECB HICP fallback]")
        return dt_ecb, val_ecb
    return None, None


def fetch_gbp() -> tuple:
    """
    GBP: OECD SDMX CPALTT01.ST monthly → FRED GBRCPIALLMINMEI → ECB HICP GB → WB
    """
    dt, val = oecd_cpi_yoy("GBR", "M")
    if val is not None and _valid(val, "GBP"):
        print(f"    ✓ GBP: {val:.4f}% CPI YoY ({dt}) [OECD SDMX ST→YoY]")
        return dt, val

    print("    ✗ GBP: OECD miss — trying FRED GBRCPIALLMINMEI")
    dt, val = fred_index_to_yoy("GBRCPIALLMINMEI")
    if val is not None and _valid(val, "GBP"):
        print(f"    ~ GBP: {val:.4f}% CPI YoY ({dt}) [FRED GBRCPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ GBP: FRED miss — trying ECB HICP GB")
    dt, val = ecb_hicp_yoy("GB")
    if val is not None and _valid(val, "GBP"):
        print(f"    ~ GBP: {val:.4f}% CPI YoY ({dt}) [ECB HICP]")
        return dt, val

    print("    ✗ GBP: ECB miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("GB")
    if val_wb is not None and _valid(val_wb, "GBP"):
        print(f"    ~ GBP: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_jpy() -> tuple:
    """
    JPY: OECD SDMX CPALTT01.ST monthly → WB
    (JPNCPIALLMINMEI ended Jun-2021 — not in cascade)
    """
    dt, val = oecd_cpi_yoy("JPN", "M")
    if val is not None and _valid(val, "JPY"):
        print(f"    ✓ JPY: {val:.4f}% CPI YoY ({dt}) [OECD SDMX ST→YoY]")
        return dt, val

    print("    ✗ JPY: OECD miss — trying World Bank (FRED JPY series ended 2021)")
    dt_wb, val_wb = wb_cpi_yoy("JP")
    if val_wb is not None and _valid(val_wb, "JPY"):
        print(f"    ~ JPY: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_aud() -> tuple:
    """
    AUD: OECD SDMX quarterly → OECD monthly → FRED AUSCPIALLQINMEI → WB
    (CPALTT01AUM659N series never existed — not in cascade)
    """
    dt, val = oecd_cpi_yoy("AUS", "Q")
    if val is not None and _valid(val, "AUD"):
        print(f"    ✓ AUD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX Q ST→YoY]")
        return dt, val

    print("    ✗ AUD: OECD quarterly miss — trying OECD monthly")
    dt, val = oecd_cpi_yoy("AUS", "M")
    if val is not None and _valid(val, "AUD"):
        print(f"    ~ AUD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX M ST→YoY]")
        return dt, val

    print("    ✗ AUD: OECD monthly miss — trying FRED AUSCPIALLQINMEI")
    dt, val = fred_index_to_yoy("AUSCPIALLQINMEI")
    if val is not None and _valid(val, "AUD"):
        print(f"    ~ AUD: {val:.4f}% CPI YoY ({dt}) [FRED AUSCPIALLQINMEI→YoY]")
        return dt, val

    print("    ✗ AUD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("AU")
    if val_wb is not None and _valid(val_wb, "AUD"):
        print(f"    ~ AUD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_cad() -> tuple:
    """CAD: OECD SDMX monthly → FRED CANCPIALLMINMEI → WB"""
    dt, val = oecd_cpi_yoy("CAN", "M")
    if val is not None and _valid(val, "CAD"):
        print(f"    ✓ CAD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX ST→YoY]")
        return dt, val

    print("    ✗ CAD: OECD miss — trying FRED CANCPIALLMINMEI")
    dt, val = fred_index_to_yoy("CANCPIALLMINMEI")
    if val is not None and _valid(val, "CAD"):
        print(f"    ~ CAD: {val:.4f}% CPI YoY ({dt}) [FRED CANCPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ CAD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CA")
    if val_wb is not None and _valid(val_wb, "CAD"):
        print(f"    ~ CAD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_chf() -> tuple:
    """CHF: OECD SDMX monthly → FRED CHECPIALLMINMEI → WB"""
    dt, val = oecd_cpi_yoy("CHE", "M")
    if val is not None and _valid(val, "CHF"):
        print(f"    ✓ CHF: {val:.4f}% CPI YoY ({dt}) [OECD SDMX ST→YoY]")
        return dt, val

    print("    ✗ CHF: OECD miss — trying FRED CHECPIALLMINMEI")
    dt, val = fred_index_to_yoy("CHECPIALLMINMEI")
    if val is not None and _valid(val, "CHF"):
        print(f"    ~ CHF: {val:.4f}% CPI YoY ({dt}) [FRED CHECPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ CHF: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CH")
    if val_wb is not None and _valid(val_wb, "CHF"):
        print(f"    ~ CHF: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_nzd() -> tuple:
    """NZD: OECD SDMX quarterly → OECD monthly → FRED NZLCPIALLQINMEI → WB"""
    dt, val = oecd_cpi_yoy("NZL", "Q")
    if val is not None and _valid(val, "NZD"):
        print(f"    ✓ NZD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX Q ST→YoY]")
        return dt, val

    print("    ✗ NZD: OECD quarterly miss — trying OECD monthly")
    dt, val = oecd_cpi_yoy("NZL", "M")
    if val is not None and _valid(val, "NZD"):
        print(f"    ~ NZD: {val:.4f}% CPI YoY ({dt}) [OECD SDMX M ST→YoY]")
        return dt, val

    print("    ✗ NZD: OECD monthly miss — trying FRED NZLCPIALLQINMEI")
    dt, val = fred_index_to_yoy("NZLCPIALLQINMEI")
    if val is not None and _valid(val, "NZD"):
        print(f"    ~ NZD: {val:.4f}% CPI YoY ({dt}) [FRED NZLCPIALLQINMEI→YoY]")
        return dt, val

    print("    ✗ NZD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("NZ")
    if val_wb is not None and _valid(val_wb, "NZD"):
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

        if is_stale(date_str, ccy):
            threshold = MAX_AGE.get(ccy, 120)
            print(f"    ⚠ {ccy}: observation {date_str} is >{threshold}d old — skipping")
            _gha_warning(f"fetch_inflation_expectations: {ccy} stale ({date_str})")
            failures += 1
            continue

        if patch_json(ccy, val, date_str):
            results[ccy] = (val, date_str)
        else:
            failures += 1

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
        _gha_error("fetch_inflation_expectations: >=3 currencies failed — likely upstream outage")
        return 1

    print(f"\n  {len(results)}/8 currencies written successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
