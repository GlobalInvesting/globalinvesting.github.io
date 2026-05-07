"""
fetch_inflation_expectations.py  v1.1
──────────────────────────────────────
Fetches inflation expectations for all G8 currencies and writes
inflationExpectations into extended-data/{CCY}.json (non-destructive patch).

Runs weekly (Mondays 06:00 UTC) from the PUBLIC site repo — no API keys required.

WHY ALL 8 (INCLUDING USD AND EUR)
──────────────────────────────────
The Real Rate Carry modal fetches T5YIE (USD) and T5YIFR (EUR) live from FRED on
every open. But this workflow serves two additional purposes:
  1. Fallback: if the live FRED fetch fails (outage, CSP regression, timeout), the
     modal falls back to extended-data. Without this workflow, those fallback values
     stagnate indefinitely. With it, they stay ≤7 days stale.
  2. Audit trail: weekly batch values written to extended-data provide a second,
     independent data point that can be diffed against the live fetch to detect
     data quality issues (e.g. FRED series revision, anomalous spike, stale value
     returned without a proper date).

SOURCE CASCADE PER CURRENCY
────────────────────────────
For non-USD/EUR currencies the cascade is tried in this order:
  1. economic-data/{CCY}.json  — calendar actuals written daily by
     update_pmi_from_calendar.py (ForexFactory CPI YoY releases). Freshest source.
  2. FRED CPI index series → YoY calculation. Often 1–18 months lag for
     OECD international series (MINMEI/QINMEI).
  3. World Bank FP.CPI.TOTL.ZG — annual, last resort.

  USD  → FRED T5YIE    (5Y breakeven inflation, daily, market-implied)
  EUR  → FRED T5YIFR   (EUR 5Y5Y inflation swap, daily, market-implied)
  GBP  → economic-data/GBP.json (calendar CPI YoY, daily) [primary]
           FRED GBRCPIALLMINMEI  (ONS CPIH monthly index → YoY calc)
           FRED CPALTT01GBM661N (CPI monthly → YoY) [fallback]
           World Bank FP.CPI.TOTL.ZG [final fallback]
  JPY  → economic-data/JPY.json (calendar CPI YoY, daily) [primary]
           World Bank FP.CPI.TOTL.ZG [fallback — both FRED JPY series ended 2021]
  AUD  → economic-data/AUD.json (calendar CPI YoY, daily) [primary]
           FRED AUSCPIALLQINMEI  (ABS quarterly index → YoY)
           World Bank FP.CPI.TOTL.ZG [fallback]
  CAD  → economic-data/CAD.json (calendar CPI YoY, daily) [primary]
           FRED CANCPIALLMINMEI  (StatsCan monthly index → YoY)
           World Bank FP.CPI.TOTL.ZG [fallback]
  CHF  → economic-data/CHF.json (calendar CPI YoY, daily) [primary]
           FRED CHECPIALLMINMEI  (FSO monthly index → YoY)
           World Bank FP.CPI.TOTL.ZG [fallback]
  NZD  → economic-data/NZD.json (calendar CPI YoY, daily) [primary]
           FRED NZLCPIALLQINMEI  (Stats NZ quarterly index → YoY)
           World Bank FP.CPI.TOTL.ZG [fallback]

USD and EUR use live daily market-implied series (breakeven / swap rate) — not
index-derived YoY. These are the same series the modal fetches live; the batch
copy is purely a staleness-bounded fallback.

INDEX → YOY METHODOLOGY
────────────────────────
FRED CPI series return index values. YoY is computed as:
    YoY% = (index_latest / index_~12_months_prior − 1) × 100
Prior-year observation tolerance: ±60 days. Minimum 13 obs (monthly) or 5 (quarterly).

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

SITE_DIR      = os.environ.get("SITE_DIR", ".")
OUT_DIR       = os.path.join(SITE_DIR, "extended-data")
ECON_DATA_DIR = os.environ.get("ECON_DATA_DIR",
                                os.path.join(SITE_DIR, "economic-data"))
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "globalinvesting-bot/1.0 (https://globalinvesting.github.io)",
    "Accept":     "text/csv,application/json;q=0.9",
}

# ── GitHub Actions annotation helpers ────────────────────────────────────────

def _gha_warning(msg: str) -> None:
    """Emit a GitHub Actions warning annotation (shows in run summary)."""
    print(f"::warning::{msg}", flush=True)


def _gha_error(msg: str) -> None:
    """Emit a GitHub Actions error annotation."""
    print(f"::error::{msg}", flush=True)

MAX_AGE_DAYS = 120   # skip writing if observation is older than this

# ── economic-data/{CCY}.json reader ─────────────────────────────────────────

def read_econ_data(ccy: str) -> tuple[str | None, float | None]:
    """
    Read CPI YoY from economic-data/{CCY}.json — maintained daily by
    update_pmi_from_calendar.py (from ForexFactory calendar actuals).
    This is the freshest source: calendar actuals within days of release.

    Returns (date_str YYYY-MM-DD, value) or (None, None).
    """
    path = os.path.join(ECON_DATA_DIR, f"{ccy}.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            d = json.load(f)
        # Structure: {"data": {"inflation": 3.0}, "dates": {"inflation": "2026-03-25"}}
        val = d.get("data", {}).get("inflation")
        dt  = d.get("dates", {}).get("inflation")
        if val is not None and dt:
            return dt[:10], round(float(val), 4)
    except Exception as e:
        print(f"    read_econ_data {ccy}: {e}")
    return None, None


# ── Source catalogue ────────────────────────────────────────────────────────
# format: (ccy, mode, series_or_sources, wb_iso2)
#   mode 'direct'  → FRED series returns the value directly (no YoY calc needed)
#   mode 'index_m' → FRED returns monthly CPI index; compute YoY
#   mode 'index_q' → FRED returns quarterly CPI index; compute YoY
#
# Each series entry is a list so primaries and fallbacks can cascade.

SOURCES = [
    # USD — T5YIE: 5Y Treasury breakeven inflation rate (daily, market-implied)
    ("USD", "direct", ["T5YIE"],                                           None),
    # EUR — T5YIFR: EUR 5Y5Y inflation swap forward rate (daily, market-implied)
    ("EUR", "direct", ["T5YIFR"],                                          None),
    # GBP — ONS CPIH monthly index (primary); CPI monthly (fallback)
    ("GBP", "index_m", ["GBRCPIALLMINMEI", "CPALTT01GBM661N"],             "GB"),
    # JPY — MIC monthly CPI index. World Bank used as fallback if FRED lags.
    ("JPY", "index_m", ["JPNCPIALLMINMEI"],                                "JP"),
    # AUD — ABS quarterly CPI index
    ("AUD", "index_q", ["AUSCPIALLQINMEI"],                                "AU"),
    # CAD — StatsCan monthly CPI index
    ("CAD", "index_m", ["CANCPIALLMINMEI"],                                "CA"),
    # CHF — FSO monthly CPI index
    ("CHF", "index_m", ["CHECPIALLMINMEI"],                                "CH"),
    # NZD — Stats NZ quarterly CPI index
    ("NZD", "index_q", ["NZLCPIALLQINMEI"],                                "NZ"),
]

# ── FRED CSV helper (no API key) ────────────────────────────────────────────

def fred_csv(series_id: str) -> list[tuple[datetime, float]]:
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

# ── World Bank fallback ─────────────────────────────────────────────────────

def wb_cpi_yoy(iso2: str) -> tuple[str | None, float | None]:
    """
    CPI YoY % from World Bank (FP.CPI.TOTL.ZG). Already YoY — no index calc.
    Returns (date_str YYYY-MM-DD, value) or (None, None).
    """
    url = f"https://api.worldbank.org/v2/country/{iso2}/indicator/FP.CPI.TOTL.ZG"
    try:
        r = requests.get(url, params={"format": "json", "mrv": 3, "per_page": 5},
                         headers=HEADERS, timeout=20)
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

# ── YoY from index ──────────────────────────────────────────────────────────

def index_to_yoy(obs: list[tuple[datetime, float]],
                 freq: str) -> tuple[str | None, float | None]:
    """
    Compute CPI YoY % from a descending list of (date, index_value) observations.
    freq: 'm' (monthly, need ≥13 obs) or 'q' (quarterly, need ≥5 obs).
    Prior-year obs must be within ±60 days of exactly 12 months ago.
    Returns (date_str, yoy_pct) or (None, None).
    """
    min_obs = 13 if freq == "m" else 5
    if len(obs) < min_obs:
        print(f"    YoY calc: only {len(obs)} obs (need {min_obs})")
        return None, None

    dt_latest, v_latest = obs[0]
    best_val, best_diff = None, float("inf")

    for dt_prior, v_prior in obs[1:]:
        diff = abs((dt_latest - dt_prior).days - 365)
        if diff < best_diff:
            best_diff, best_val = diff, v_prior
        if (dt_latest - dt_prior).days > 430:
            break

    if best_val is None or best_diff > 60:
        print(f"    YoY calc: no prior-year obs within ±60d (closest={best_diff}d)")
        return None, None

    yoy = round((v_latest / best_val - 1) * 100, 4)
    return dt_latest.strftime("%Y-%m-%d"), yoy

# ── Stale guard ─────────────────────────────────────────────────────────────

def is_stale(date_str: str | None) -> bool:
    if not date_str:
        return True
    try:
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - datetime.strptime(date_str[:10], "%Y-%m-%d")).days
        return age > MAX_AGE_DAYS
    except (ValueError, TypeError):
        return True

# ── JSON patch ──────────────────────────────────────────────────────────────

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

# ── Per-currency fetch ──────────────────────────────────────────────────────

def fetch_one(ccy: str, mode: str, series_list: list[str],
              wb_iso2: str | None) -> tuple[str | None, float | None]:
    """
    Try each FRED series in order. Fall back to World Bank if all fail.
    Returns (date_str, value) or (None, None).
    """
    # ── direct mode (USD T5YIE, EUR T5YIFR) — value IS the expectation
    if mode == "direct":
        for sid in series_list:
            rows = fred_csv(sid)
            if rows:
                dt, val = rows[0]
                if not (-2.0 < val < 15.0):
                    print(f"    ✗ {ccy}: {sid} value {val:.4f}% out of plausible range [-2%, 15%]")
                    continue
                print(f"    ✓ {ccy}: {val:.4f}% ({dt.date()}) [{sid}]")
                return dt.strftime("%Y-%m-%d"), round(val, 4)
            print(f"    ✗ {ccy}: {sid} returned no data")
        if wb_iso2:
            dt_wb, val_wb = wb_cpi_yoy(wb_iso2)
            if val_wb is not None and -5.0 < val_wb < 30.0:
                print(f"    ~ {ccy}: {val_wb:.4f}% ({dt_wb}) [WB fallback]")
                return dt_wb, val_wb
        return None, None

    # ── index modes — compute YoY from index observations
    # Primary: economic-data/{CCY}.json (daily calendar actuals — freshest source)
    freq = "m" if mode == "index_m" else "q"
    econ_dt, econ_val = read_econ_data(ccy)
    if econ_val is not None:
        if not (-5.0 < econ_val < 30.0):
            print(f"    ✗ {ccy}: econ-data value {econ_val:.4f}% out of plausible range [-5%, 30%]")
        else:
            print(f"    ✓ {ccy}: {econ_val:.4f}% CPI YoY ({econ_dt}) [economic-data/calendar]")
            return econ_dt, econ_val

    # Secondary: FRED CPI index series → YoY calculation
    for sid in series_list:
        rows = fred_csv(sid)
        if rows:
            dt_str, yoy = index_to_yoy(rows, freq)
            if yoy is not None:
                if not (-5.0 < yoy < 30.0):
                    print(f"    ✗ {ccy}: {sid} YoY {yoy:.4f}% out of plausible range [-5%, 30%]")
                else:
                    print(f"    ✓ {ccy}: {yoy:.4f}% CPI YoY ({dt_str}) [{sid}→YoY]")
                    return dt_str, yoy
            print(f"    ~ {ccy}: {sid} fetched but YoY calc failed")
        else:
            print(f"    ✗ {ccy}: {sid} returned no data")

    # World Bank fallback
    if wb_iso2:
        dt_wb, val_wb = wb_cpi_yoy(wb_iso2)
        if val_wb is not None and -5.0 < val_wb < 30.0:
            print(f"    ~ {ccy}: {val_wb:.4f}% ({dt_wb}) [WB fallback]")
            return dt_wb, val_wb

    return None, None

# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("INFLATION EXPECTATIONS — G8 currencies")
    print(f"Run: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    failures = 0
    results  = {}

    for ccy, mode, series_list, wb_iso2 in SOURCES:
        print(f"\n[{ccy}]")
        date_str, val = fetch_one(ccy, mode, series_list, wb_iso2)

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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ccy, (val, dt) in sorted(results.items()):
        mode_label = next(m for c, m, *_ in SOURCES if c == ccy)
        src = "direct (market-implied)" if mode_label == "direct" else "CPI YoY"
        print(f"  {ccy:<4}  {val:+.4f}%  ({dt})  [{src}]")

    if failures:
        skipped = [ccy for ccy, *_ in SOURCES if ccy not in results]
        print(f"\n  Skipped/failed: {', '.join(skipped)} ({failures} issue(s))")

    if failures >= 3:
        _gha_error("fetch_inflation_expectations: ≥3 currencies failed — likely upstream outage")
        return 1

    print(f"\n  {len(results)}/8 currencies written successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
