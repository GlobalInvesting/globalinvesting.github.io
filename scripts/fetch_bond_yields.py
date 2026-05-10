"""
fetch_bond_yields.py  v1.3  —  Bond yields for USD / EUR / GBP / JPY + G8 bond2y

CHANGELOG v1.3 (fixes vs v1.2)
────────────────────────────────
BUG-5  JPY bond2y was never fetched. fetch_jpy() only produced bond10y, leaving
       JPY.bond2y = None in all extended-data/JPY.json files and causing the sovereign
       spreads table (dashboard2.js renderSovereignSpreads) to show — in the 2Y column
       for JP. Fixed: added FRED IRLTST01JPM156N (Japan 2Y government bond, monthly)
       as the JPY bond2y source — consistent with the AUD/CAD/NZD treatment.

BUG-6  AUD/CAD/NZD bond2y was computed correctly by fetch_aud_2y/cad_2y/nzd_2y but
       the workflow update-bond-yields.yml only git-added USD/EUR/GBP/JPY.json, so the
       AUD/CAD/NZD values were discarded on every run without ever reaching the repo.
       Fixed in update-bond-yields.yml: expanded git add to include AUD/CAD/NZD.json.

CHANGELOG v1.2 (fixes vs v1.0)
────────────────────────────────
BUG-1  EUR / GBP / JPY had no fallbacks despite being specified in CHANGELOG v7.50.0.
       Now every currency has a full cascade with at least 2 independent sources.

BUG-2  GBP BOE endpoint was wrong: `fromshowcolumns.asp` (HTML endpoint, returns 403
       or bot-challenge HTML) instead of `_iadb-fromshowcolumns.asp` (CSV endpoint with
       `csv.x=yes`). Corrected to match the working implementation in
       update_extended_data.py v13.0.

BUG-3  Silent failure: WARN paths logged a message but did NOT emit GitHub Actions
       annotations (::warning / ::error) and did not set a non-zero exit code, so
       the workflow appeared green while writing stale data. Fixed: each failed fetch
       now emits a `::warning::` annotation and increments a soft-failure counter.
       USD bond10y failure → exit(1) (hard failure, most critical field).
       All other fields → warning annotation + counter; exit(1) if ≥2 fields failed.

BUG-4  EUR ECB SDMX fails on EU public holidays (e.g. May 1 Labour Day, Dec 25 …).
       Added FRED IRLTLT01EZM156N as monthly fallback (public CSV, no key).

Source cascade per currency:
    USD bond10y  → ohlc-data/us10y.json  (update-ohlc, daily)
                   FRED DGS10 public CSV (daily, no key) [fallback]
    USD vix      → ohlc-data/vix.json    (update-ohlc, daily)
    USD bond2y   → FRED DGS2  public CSV (daily, no key)
    USD bond5y   → FRED DGS5  public CSV (daily, no key)
    EUR bond10y  → ECB SDMX YC daily     (no key)
                   FRED IRLTLT01EZM156N  (monthly, no key) [fallback]
    EUR bond2y   → ECB SDMX YC daily SR_2Y (no key)
    GBP bond10y  → BOE SDIE _iadb CSV    (daily, no key)   ← BUG-2 fixed endpoint
                   FRED IRLTLT01GBM156N  (monthly, no key) [fallback]
    GBP bond2y   → BOE SDIE _iadb CSV IUDMNPY (daily, no key)
                   FRED IRLTST01GBM156N  (monthly, no key) [fallback]
    JPY bond10y  → ECB FM SDMX monthly   (no key)
                   OECD SDMX monthly     (no key)          [fallback]
                   FRED IRLTLT01JPM156N  (monthly, no key) [final fallback]
    JPY bond2y   → FRED IRLTST01JPM156N  (monthly, no key)
    CAD bond2y   → FRED IRLTST01CAM156N  (monthly, no key)
    AUD bond2y   → FRED IRLTLT01AUM156N  (monthly, no key)

Exit policy (BUG-3):
    USD bond10y unavailable  → exit(1)  (hard failure)
    ≥2 other fields failed   → exit(1)  (degraded run)
    1 other field failed     → exit(0) + ::warning annotation
"""

import json
import os
import sys
import csv
import requests
from io import StringIO
from datetime import datetime, date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

SITE_DIR = os.environ.get("SITE_DIR", ".")
OHLC_DIR = os.path.join(SITE_DIR, "ohlc-data")
OUT_DIR  = os.path.join(SITE_DIR, "extended-data")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "globalinvesting-bot/1.1 (https://globalinvesting.github.io)",
    "Accept":     "application/json, text/plain, */*",
}

# ── GitHub Actions annotation helpers ────────────────────────────────────────

def _gha_warning(msg: str) -> None:
    """Emit a GitHub Actions warning annotation (shows in run summary)."""
    print(f"::warning::{msg}", flush=True)


def _gha_error(msg: str) -> None:
    """Emit a GitHub Actions error annotation."""
    print(f"::error::{msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_existing(ccy: str) -> tuple[dict, dict]:
    """Load existing extended-data/XX.json; return (data_dict, dates_dict)."""
    path = os.path.join(OUT_DIR, f"{ccy}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                obj = json.load(f)
            return obj.get("data", {}), obj.get("dates", {})
        except Exception:
            pass
    return {}, {}


def _save(ccy: str, data: dict, dates: dict) -> None:
    path = os.path.join(OUT_DIR, f"{ccy}.json")
    with open(path, "w") as f:
        json.dump({"data": data, "dates": dates}, f, separators=(",", ":"))
    print(f"  ✓ Wrote {path}")


# ── Source: FRED public CSV (no key) ─────────────────────────────────────────

def _fred_csv_latest(series_id: str) -> tuple[str | None, float | None]:
    """Fetch the most recent non-missing observation from FRED public CSV endpoint."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"    FRED CSV {series_id}: HTTP {r.status_code}")
            return None, None
        rows = []
        reader = csv.reader(StringIO(r.text))
        for row in reader:
            if len(row) == 2 and row[0] != "DATE" and row[1] not in (".", ""):
                try:
                    rows.append((row[0], float(row[1])))
                except ValueError:
                    continue
        if rows:
            return rows[-1]
        return None, None
    except Exception as e:
        print(f"    FRED CSV {series_id}: {e}")
        return None, None


# ── Source: ohlc-data/<symbol>.json ──────────────────────────────────────────

def _ohlc_latest(symbol: str) -> tuple[str | None, float | None]:
    """Read the last bar's close from ohlc-data/<symbol>.json."""
    path = os.path.join(OHLC_DIR, f"{symbol}.json")
    if not os.path.exists(path):
        print(f"    ohlc-data/{symbol}.json not found — skipping")
        return None, None
    try:
        with open(path) as f:
            bars = json.load(f)
        if bars:
            last = bars[-1]
            return last["time"], round(float(last["close"]), 4)
        return None, None
    except Exception as e:
        print(f"    ohlc-data/{symbol}.json read error: {e}")
        return None, None


# ── Source: ECB SDMX REST API ────────────────────────────────────────────────

def _ecb_sdmx_latest(flow: str, key: str) -> tuple[str | None, float | None]:
    """
    Fetch latest observation from ECB SDMX REST API.
    Returns None on EU public holidays (ECB does not publish data those days).
    """
    try:
        url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
        params = {"lastNObservations": 5, "format": "jsondata", "detail": "dataonly"}
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            print(f"    ECB {flow}/{key}: HTTP {r.status_code}")
            return None, None
        data = r.json()
        dataset = data.get("dataSets", [{}])[0]
        series_dict = dataset.get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next((d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), [])
        if not series_dict or not time_dim:
            return None, None
        series = next(iter(series_dict.values()))
        obs = series.get("observations", {})
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                date_str = period if len(period) == 10 else period + "-01"
                try:
                    return date_str, float(val)
                except (ValueError, TypeError):
                    continue
        return None, None
    except Exception as e:
        print(f"    ECB {flow}/{key}: {e}")
        return None, None


# ── Source: BOE SDIE CSV ──────────────────────────────────────────────────────

def _boe_latest(series_code: str) -> tuple[str | None, float | None]:
    """
    Fetch latest daily observation from BOE SDIE CSV endpoint.

    BUG-2 FIX: uses _iadb-fromshowcolumns.asp with csv.x=yes, SeriesCodes, CSVF=TN
    (the correct CSV endpoint). The previous implementation used fromshowcolumns.asp
    with Travel=NIxRSx / html.x params — the HTML endpoint — which returns 403 or
    a Cloudflare bot-challenge HTML page from GitHub Actions IPs.

    Date format in BOE CSV response: 'DD Mon YYYY' — e.g. ' 2 May 2026' or '28 Apr 2026'.
    strptime('%d %b %Y') handles both single and double digit days correctly.
    """
    try:
        now        = datetime.utcnow()
        date_from  = f"01/Jan/{now.year - 1}"
        date_to    = f"{now.day:02d}/{now.strftime('%b')}/{now.year}"
        url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
        params = {
            "csv.x":       "yes",
            "Datefrom":    date_from,
            "Dateto":      date_to,
            "SeriesCodes": series_code,
            "CSVF":        "TN",   # Tabular No titles — bare DATE,VALUE rows
            "UsingCodes":  "Y",
            "VPD":         "Y",
            "VFD":         "N",
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            print(f"    BOE {series_code}: HTTP {r.status_code}")
            return None, None
        # Guard against HTML bot-challenge response (Cloudflare returns HTML with 200)
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type.lower() or r.text.strip().startswith("<"):
            print(f"    BOE {series_code}: received HTML instead of CSV (bot challenge?)")
            return None, None
        rows = []
        for line in r.text.strip().split("\n"):
            line = line.strip()
            if not line or line.upper().startswith("DATE") or line.startswith('"'):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            date_str_raw = parts[0].strip()
            val_str = parts[1].strip().replace("..", "").strip()
            if not val_str or val_str in (".", "", "N/A"):
                continue
            try:
                # BOE format: 'DD Mon YYYY' e.g. ' 2 May 2026' or '28 Apr 2026'
                dt = datetime.strptime(date_str_raw.strip(), "%d %b %Y")
                rows.append((dt.strftime("%Y-%m-%d"), float(val_str)))
            except (ValueError, TypeError):
                continue
        if rows:
            rows.sort(key=lambda x: x[0])
            return rows[-1]
        return None, None
    except Exception as e:
        print(f"    BOE {series_code}: {e}")
        return None, None


# ── Source: OECD SDMX ────────────────────────────────────────────────────────

def _oecd_sdmx_latest(key: str) -> tuple[str | None, float | None]:
    """Fetch latest observation from OECD SDMX public REST API."""
    try:
        url = f"https://sdmx.oecd.org/public/rest/data/{key}"
        params = {"lastNObservations": 3, "format": "jsondata", "detail": "dataonly"}
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            print(f"    OECD SDMX {key}: HTTP {r.status_code}")
            return None, None
        data = r.json()
        series_dict = data.get("dataSets", [{}])[0].get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next((d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), [])
        if not series_dict or not time_dim:
            return None, None
        series = next(iter(series_dict.values()))
        obs = series.get("observations", {})
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                date_str = period if len(period) == 10 else period + "-01"
                try:
                    return date_str, float(val)
                except (ValueError, TypeError):
                    continue
        return None, None
    except Exception as e:
        print(f"    OECD SDMX {key}: {e}")
        return None, None


# ── USD ───────────────────────────────────────────────────────────────────────

def fetch_usd(soft_failures: list) -> None:
    print("\nUSD")
    data, dates = _load_existing("USD")

    # bond10y — primary: ohlc-data/us10y.json, fallback: FRED DGS10
    print("  bond10y  (ohlc-data/us10y.json → FRED:DGS10)")
    dt, val = _ohlc_latest("us10y")
    if val is None:
        print("    ohlc miss — trying FRED DGS10")
        dt, val = _fred_csv_latest("DGS10")
    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        msg = "USD bond10y unavailable from all sources — LLM yield context will be stale"
        _gha_error(msg)
        print(f"    ERROR: {msg}")
        _save("USD", data, dates)
        sys.exit(1)  # hard failure — most critical field

    # vix — ohlc-data/vix.json only (no API fallback needed, from same OHLC run)
    print("  vix  (ohlc-data/vix.json)")
    dt, val = _ohlc_latest("vix")
    if val is not None and 5 < val < 100:
        data["vix"]  = round(val, 2)
        dates["vix"] = dt
        print(f"    {val:.2f}  ({dt})")
    else:
        _gha_warning(f"USD vix: value {val} out of range or unavailable — keeping existing")
        soft_failures.append("USD.vix")

    # bond2y — FRED DGS2 (public CSV, no key)
    print("  bond2y  (FRED:DGS2)")
    dt, val = _fred_csv_latest("DGS2")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        _gha_warning(f"USD bond2y: FRED DGS2 unavailable (val={val}) — keeping existing")
        soft_failures.append("USD.bond2y")

    # bond5y — FRED DGS5 (public CSV, no key)
    print("  bond5y  (FRED:DGS5)")
    dt, val = _fred_csv_latest("DGS5")
    if val is not None and 0 < val < 20:
        data["bond5y"]  = round(val, 4)
        dates["bond5y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        _gha_warning(f"USD bond5y: FRED DGS5 unavailable (val={val}) — keeping existing")
        soft_failures.append("USD.bond5y")

    _save("USD", data, dates)


# ── EUR ───────────────────────────────────────────────────────────────────────

def fetch_eur(soft_failures: list) -> None:
    print("\nEUR")
    data, dates = _load_existing("EUR")

    # bond10y cascade:
    #   1. ECB SDMX daily yield curve (best — daily, T+0)
    #      Fails on EU public holidays (ECB closed: Jan 1, Good Friday, Easter Mon,
    #      May 1, Dec 25, Dec 26) — returns empty observations
    #   2. FRED IRLTLT01EZM156N (monthly, ~1 month lag — but never fails on holidays)
    print("  bond10y  (ECB SDMX YC daily → FRED:IRLTLT01EZM156N monthly)")
    dt, val = _ecb_sdmx_latest("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
    source = "ECB-SDMX-daily"
    if val is None or not (0 < val < 20):
        print(f"    ECB SDMX miss (val={val}) — trying FRED IRLTLT01EZM156N")
        dt, val = _fred_csv_latest("IRLTLT01EZM156N")
        source = "FRED-monthly"

    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("EUR bond10y: all sources unavailable — keeping existing value")
        soft_failures.append("EUR.bond10y")

    # bond2y — ECB SDMX YC SR_2Y (same cascade as 10Y)
    print("  bond2y  (ECB SDMX YC SR_2Y)")
    dt, val = _ecb_sdmx_latest("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [ECB-SDMX-daily]")
    else:
        _gha_warning("EUR bond2y: ECB SDMX SR_2Y unavailable — keeping existing")
        soft_failures.append("EUR.bond2y")

    _save("EUR", data, dates)


# ── GBP ───────────────────────────────────────────────────────────────────────

def fetch_gbp(soft_failures: list) -> None:
    print("\nGBP")
    data, dates = _load_existing("GBP")

    # bond10y cascade:
    #   1. BOE SDIE _iadb CSV (daily, IUDMNZC — 10Y nominal zero coupon gilt)
    #      BUG-2 FIX: correct endpoint is _iadb-fromshowcolumns.asp with csv.x=yes
    #   2. FRED IRLTLT01GBM156N (monthly — reliable fallback, no bot issues)
    print("  bond10y  (BOE SDIE _iadb:IUDMNZC daily → FRED:IRLTLT01GBM156N monthly)")
    dt, val = _boe_latest("IUDMNZC")
    source = "BOE-SDIE-daily"
    if val is None or not (0 < val < 20):
        print(f"    BOE SDIE miss (val={val}) — trying FRED IRLTLT01GBM156N")
        dt, val = _fred_csv_latest("IRLTLT01GBM156N")
        source = "FRED-monthly"

    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("GBP bond10y: all sources unavailable — keeping existing value")
        soft_failures.append("GBP.bond10y")

    # bond2y — BOE SDIE IUDMNPY (2Y nominal par yield, same endpoint as 10Y)
    #   Fallback: FRED IRLTST01GBM156N (monthly)
    print("  bond2y  (BOE SDIE IUDMNPY → FRED:IRLTST01GBM156N)")
    dt, val = _boe_latest("IUDMNPY")
    source = "BOE-SDIE-daily"
    if val is None or not (0 < val < 20):
        print(f"    BOE SDIE IUDMNPY miss (val={val}) — trying FRED IRLTST01GBM156N")
        dt, val = _fred_csv_latest("IRLTST01GBM156N")
        source = "FRED-monthly"
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("GBP bond2y: all sources unavailable — keeping existing")
        soft_failures.append("GBP.bond2y")

    _save("GBP", data, dates)


# ── JPY ───────────────────────────────────────────────────────────────────────

def fetch_jpy(soft_failures: list) -> None:
    print("\nJPY")
    data, dates = _load_existing("JPY")

    # bond10y cascade (all monthly — no daily source available without API key):
    #   1. ECB FM SDMX nominal JGB 10Y (JP10YT_RR without R_ prefix)
    #      Note: R_ prefix = REAL yield (inflation-adjusted, often negative) — wrong series
    #   2. OECD SDMX monthly (IRLTLT01JPM156N equivalent)
    #   3. FRED IRLTLT01JPM156N (monthly — last resort)
    print("  bond10y  (ECB FM SDMX monthly → OECD SDMX → FRED:IRLTLT01JPM156N)")
    dt, val, source = None, None, None

    # 1. ECB FM — try both period-average and end-of-period variants
    for ecb_key in (
        "M.JP.JPY.4F.BB.JP10YT_RR.YLDA",   # nominal period average
        "M.JP.JPY.4F.BB.JP10YT_RR.YLD",    # nominal end-of-period
    ):
        dt, val = _ecb_sdmx_latest("FM", ecb_key)
        if val is not None and -5 < val < 20:
            source = "ECB-FM-monthly"
            break
        dt, val = None, None

    # 2. OECD SDMX
    if val is None:
        print("    ECB FM miss — trying OECD SDMX")
        dt, val = _oecd_sdmx_latest(
            "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/M.JPN.IRLTLT01.ST"
        )
        if val is not None and -5 < val < 20:
            source = "OECD-SDMX-monthly"
        else:
            dt, val = None, None

    # 3. FRED public CSV
    if val is None:
        print("    OECD miss — trying FRED IRLTLT01JPM156N")
        dt, val = _fred_csv_latest("IRLTLT01JPM156N")
        if val is not None and -5 < val < 20:
            source = "FRED-monthly"
        else:
            dt, val = None, None

    if val is not None:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("JPY bond10y: all sources unavailable — keeping existing value")
        soft_failures.append("JPY.bond10y")

    # bond2y — FRED IRLTST01JPM156N (Japan 2Y government bond yield, monthly)
    #   No daily public source available without API key; monthly FRED is consistent
    #   with the AUD/CAD/NZD treatment and good enough for the sovereign spread table.
    print("  bond2y  (FRED:IRLTST01JPM156N monthly)")
    dt2, val2 = _fred_csv_latest("IRLTST01JPM156N")
    if val2 is not None and -5 < val2 < 20:
        data["bond2y"]  = round(val2, 4)
        dates["bond2y"] = dt2
        print(f"    {val2:.4f}%  ({dt2})  [FRED-monthly]")
    else:
        _gha_warning("JPY bond2y: FRED IRLTST01JPM156N unavailable — keeping existing")
        soft_failures.append("JPY.bond2y")

    _save("JPY", data, dates)


# ── AUD ───────────────────────────────────────────────────────────────────────

def fetch_aud_2y(soft_failures: list) -> None:
    """AUD bond2y only — bond10y already written by update-bond-yields (not in scope here)."""
    print("\nAUD (bond2y only)")
    data, dates = _load_existing("AUD")
    print("  bond2y  (FRED:IRLTLT01AUM156N monthly)")
    dt, val = _fred_csv_latest("IRLTLT01AUM156N")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [FRED-monthly]")
    else:
        _gha_warning("AUD bond2y: FRED IRLTLT01AUM156N unavailable — keeping existing")
        soft_failures.append("AUD.bond2y")
    _save("AUD", data, dates)


# ── CAD ───────────────────────────────────────────────────────────────────────

def fetch_cad_2y(soft_failures: list) -> None:
    """CAD bond2y only — bond10y already handled elsewhere."""
    print("\nCAD (bond2y only)")
    data, dates = _load_existing("CAD")
    print("  bond2y  (FRED:IRLTST01CAM156N monthly)")
    dt, val = _fred_csv_latest("IRLTST01CAM156N")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [FRED-monthly]")
    else:
        _gha_warning("CAD bond2y: FRED IRLTST01CAM156N unavailable — keeping existing")
        soft_failures.append("CAD.bond2y")
    _save("CAD", data, dates)


# ── NZD ───────────────────────────────────────────────────────────────────────

def fetch_nzd_2y(soft_failures: list) -> None:
    """NZD bond2y only — FRED IRLTLT01NZM156N (monthly)."""
    print("\nNZD (bond2y only)")
    data, dates = _load_existing("NZD")
    print("  bond2y  (FRED:IRLTLT01NZM156N monthly)")
    dt, val = _fred_csv_latest("IRLTLT01NZM156N")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [FRED-monthly]")
    else:
        _gha_warning("NZD bond2y: FRED unavailable — keeping existing")
        soft_failures.append("NZD.bond2y")
    _save("NZD", data, dates)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_utc = datetime.utcnow()
    print(f"fetch_bond_yields.py v1.3 — {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"SITE_DIR : {os.path.abspath(SITE_DIR)}")
    print(f"OHLC_DIR : {os.path.abspath(OHLC_DIR)}")
    print(f"OUT_DIR  : {os.path.abspath(OUT_DIR)}")

    soft_failures: list[str] = []
    hard_errors:   list[str] = []

    for ccy, fn in [("USD", fetch_usd), ("EUR", fetch_eur), ("GBP", fetch_gbp),
                    ("JPY", fetch_jpy), ("AUD", fetch_aud_2y),
                    ("CAD", fetch_cad_2y), ("NZD", fetch_nzd_2y)]:
        try:
            fn(soft_failures)
        except SystemExit:
            raise   # propagate hard failure from fetch_usd
        except Exception as exc:
            _gha_error(f"{ccy} fetch crashed: {exc}")
            print(f"  ERROR [{ccy}]: {exc}")
            hard_errors.append(ccy)

    print(f"\nDone — {7 - len(hard_errors)}/7 currencies processed.")

    if soft_failures:
        print(f"Soft failures (kept existing value): {', '.join(soft_failures)}")

    if hard_errors:
        print(f"Hard errors (script crashed): {', '.join(hard_errors)}")
        sys.exit(1)

    # BUG-3 FIX: exit non-zero if too many soft failures (degraded run)
    # threshold: ≥2 fields failed means something systemic is wrong
    if len(soft_failures) >= 2:
        _gha_error(
            f"Degraded run: {len(soft_failures)} fields failed "
            f"({', '.join(soft_failures)}) — investigate source availability"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
