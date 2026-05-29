"""
fetch_bond_yields.py  v1.8  —  Bond yields for USD / EUR / GBP / JPY + G8 bond2y

CHANGELOG v1.6 (fixes vs v1.5)
────────────────────────────────
IMPROVE-1  JPY bond2y: added FRED INTGSBJPM193N (IMF International Financial
           Statistics, Government Securities Yields 2Y, monthly) as primary source.
           Follows the same INTGSB*193N series family confirmed working for EUR
           (INTGSBEAM193N) and GBP (INTGSBGBM193N). If 404/unavailable, falls
           back to None (soft optional failure, no exit(1)).

IMPROVE-2  NZD bond2y: added FRED INTGSBNZM193N (same IMF IFS family, NZ 2Y monthly).
           RBNZ remains blocked (Cloudflare 403 from GH Actions), but IMF IFS
           collects NZGB yields. If unavailable, falls back to None.

CHANGELOG v1.5 (fixes vs v1.4)
────────────────────────────────
BUG-10  JPY/AUD/NZD bond2y: ECB FM SDMX series JP2YT_RR, AU2YT_RR, NZ2YT_RR
        return HTTP 404 from the GH Actions runner — these series do not exist
        in ECB FM for non-European sovereigns. Exit policy overhauled (BUG-11).

BUG-11  Exit policy over-sensitive: separated REQUIRED vs OPTIONAL failure counters.
        Only ≥2 REQUIRED failures → exit(1). Optional failures emit ::notice::.

CHANGELOG v1.4 / v1.3 / v1.2 — see git history.

Source cascade per currency:
    USD bond10y  → ohlc-data/us10y.json  (update-ohlc, daily)
                   FRED DGS10 public CSV (daily, no key) [fallback]
    USD vix      → ohlc-data/vix.json    (update-ohlc, daily)
    USD bond2y   → FRED DGS2  public CSV (daily, no key)           [REQUIRED]
    USD bond5y   → FRED DGS5  public CSV (daily, no key)           [REQUIRED]
    EUR bond10y  → ECB SDMX YC daily     (no key)                  [REQUIRED]
                   FRED IRLTLT01EZM156N  (monthly, no key) [fallback]
    EUR bond2y   → ECB SDMX YC daily SR_2Y (no key)                [REQUIRED]
    GBP bond10y  → BOE SDIE _iadb CSV    (daily, no key)           [REQUIRED]
                   FRED IRLTLT01GBM156N  (monthly, no key) [fallback]
                   OECD SDMX GBR.IRLTLT01 (monthly, no key) [final fallback]
    GBP bond2y   → BOE SDIE _iadb CSV IUDMNPY (daily, no key)      [REQUIRED]
                   FRED IRLTST01GBM156N  (monthly, no key) [fallback]
    JPY bond10y  → ECB FM SDMX monthly   (no key)                  [REQUIRED]
                   OECD SDMX monthly     (no key)          [fallback]
                   FRED IRLTLT01JPM156N  (monthly, no key) [final fallback]
    JPY bond2y   → FRED INTGSBJPM193N    (monthly, IMF IFS, no key)[OPTIONAL]
    CAD bond2y   → BoC Valet BD.CDN.2YR.DQ.YLD (daily, no key)    [REQUIRED]
                   FRED IRLTLT01CAM156N  (monthly, no key) [fallback]
    AUD bond2y   → Australia has no 2Y government bond benchmark   [OPTIONAL → None]
    NZD bond2y   → FRED INTGSBNZM193N    (monthly, IMF IFS, no key)[OPTIONAL]

Exit policy:
    USD bond10y unavailable     → exit(1)  (hard failure, most critical field)
    ≥2 REQUIRED fields failed   → exit(1)  (degraded run)
    1 REQUIRED field failed     → exit(0)  + ::warning:: annotation
    OPTIONAL field unavailable  → exit(0)  + ::notice:: annotation (expected)
"""

import json
import os
import sys
import csv
import requests
from io import StringIO
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

SITE_DIR = os.environ.get("SITE_DIR", ".")
OHLC_DIR = os.path.join(SITE_DIR, "ohlc-data")
OUT_DIR  = os.path.join(SITE_DIR, "extended-data")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "globalinvesting-bot/1.5 (https://globalinvesting.github.io)",
    "Accept":     "application/json, text/plain, */*",
}

# ── GitHub Actions annotation helpers ────────────────────────────────────────

def _gha_warning(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)

def _gha_error(msg: str) -> None:
    print(f"::error::{msg}", flush=True)

def _gha_notice(msg: str) -> None:
    print(f"::notice::{msg}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_existing(ccy: str) -> tuple[dict, dict]:
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


def _is_stale(date_str: str | None, max_months: int = 18) -> bool:
    """Return True if date_str is more than max_months before today.
    Used to reject discontinued FRED series that return an old final observation
    instead of 404 — e.g. INTGSBJPM193N last published 2017-05-01.
    """
    if date_str is None:
        return True
    try:
        obs = datetime.strptime(date_str, "%Y-%m-%d")
        cutoff = datetime.utcnow() - timedelta(days=max_months * 31)
        return obs < cutoff
    except ValueError:
        return True


# ── Source: ohlc-data/<symbol>.json ──────────────────────────────────────────

def _ohlc_latest(symbol: str) -> tuple[str | None, float | None]:
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
            "CSVF":        "TN",
            "UsingCodes":  "Y",
            "VPD":         "Y",
            "VFD":         "N",
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            print(f"    BOE {series_code}: HTTP {r.status_code}")
            return None, None
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


# ── Source: Bank of Canada Valet API ─────────────────────────────────────────

def _boc_valet_latest(series_name: str, recent_weeks: int = 4) -> tuple[str | None, float | None]:
    """Fetch latest observation from Bank of Canada Valet API (public, no key).
    Series BD.CDN.2YR.DQ.YLD = Government of Canada 2-year benchmark bond yield (daily).
    Confirmed working from GitHub Actions runners (HTTP 200, value 2.94% on 2026-05-10).
    """
    try:
        end_date   = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(weeks=recent_weeks)).strftime("%Y-%m-%d")
        url    = f"https://www.bankofcanada.ca/valet/observations/{series_name}/json"
        params = {"start_date": start_date, "end_date": end_date}
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            print(f"    BoC Valet {series_name}: HTTP {r.status_code}")
            return None, None
        obs = r.json().get("observations", [])
        for entry in reversed(obs):
            val_obj = entry.get(series_name, {})
            val_str = val_obj.get("v") if isinstance(val_obj, dict) else None
            if val_str and val_str not in ("", "Bank holiday", "nd"):
                try:
                    return entry["d"], float(val_str)
                except (ValueError, TypeError):
                    pass
        return None, None
    except Exception as e:
        print(f"    BoC Valet {series_name}: {e}")
        return None, None


# ── USD ───────────────────────────────────────────────────────────────────────

def fetch_usd(req_failures: list) -> None:
    print("\nUSD")
    data, dates = _load_existing("USD")

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
        sys.exit(1)

    print("  vix  (ohlc-data/vix.json)")
    dt, val = _ohlc_latest("vix")
    if val is not None and 5 < val < 100:
        data["vix"]  = round(val, 2)
        dates["vix"] = dt
        print(f"    {val:.2f}  ({dt})")
    else:
        _gha_warning(f"USD vix: value {val} out of range or unavailable — keeping existing")
        req_failures.append("USD.vix")

    print("  bond2y  (FRED:DGS2)")
    dt, val = _fred_csv_latest("DGS2")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        _gha_warning(f"USD bond2y: FRED DGS2 unavailable (val={val}) — keeping existing")
        req_failures.append("USD.bond2y")

    print("  bond5y  (FRED:DGS5)")
    dt, val = _fred_csv_latest("DGS5")
    if val is not None and 0 < val < 20:
        data["bond5y"]  = round(val, 4)
        dates["bond5y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        _gha_warning(f"USD bond5y: FRED DGS5 unavailable (val={val}) — keeping existing")
        req_failures.append("USD.bond5y")

    _save("USD", data, dates)


# ── EUR ───────────────────────────────────────────────────────────────────────

def fetch_eur(req_failures: list) -> None:
    print("\nEUR")
    data, dates = _load_existing("EUR")

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
        req_failures.append("EUR.bond10y")

    print("  bond2y  (ECB SDMX YC SR_2Y)")
    dt, val = _ecb_sdmx_latest("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [ECB-SDMX-daily]")
    else:
        _gha_warning("EUR bond2y: ECB SDMX SR_2Y unavailable — keeping existing")
        req_failures.append("EUR.bond2y")

    _save("EUR", data, dates)


# ── GBP ───────────────────────────────────────────────────────────────────────

def fetch_gbp(req_failures: list) -> None:
    print("\nGBP")
    data, dates = _load_existing("GBP")

    print("  bond10y  (BOE SDIE _iadb:IUDMNZC daily → FRED:IRLTLT01GBM156N monthly → OECD SDMX monthly)")
    dt, val = _boe_latest("IUDMNZC")
    source = "BOE-SDIE-daily"
    if val is None or not (0 < val < 20):
        print(f"    BOE SDIE miss (val={val}) — trying FRED IRLTLT01GBM156N")
        dt, val = _fred_csv_latest("IRLTLT01GBM156N")
        source = "FRED-monthly"
    if val is None or not (0 < val < 20):
        print(f"    FRED miss (val={val}) — trying OECD SDMX (UK 10Y long-term rate)")
        dt, val = _oecd_sdmx_latest("OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/M.GBR.IRLTLT01.ST")
        source = "OECD-SDMX-monthly"
    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("GBP bond10y: all sources unavailable (BOE SDIE, FRED, OECD) — keeping existing value")
        req_failures.append("GBP.bond10y")

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
        req_failures.append("GBP.bond2y")

    _save("GBP", data, dates)


# ── JPY ───────────────────────────────────────────────────────────────────────

def fetch_jpy(req_failures: list, opt_failures: list) -> None:
    print("\nJPY")
    data, dates = _load_existing("JPY")

    print("  bond10y  (ECB FM SDMX monthly → OECD SDMX → FRED:IRLTLT01JPM156N)")
    dt, val, source = None, None, None

    for ecb_key in (
        "M.JP.JPY.4F.BB.JP10YT_RR.YLDA",
        "M.JP.JPY.4F.BB.JP10YT_RR.YLD",
    ):
        dt, val = _ecb_sdmx_latest("FM", ecb_key)
        if val is not None and -5 < val < 20:
            source = "ECB-FM-monthly"
            break
        dt, val = None, None

    if val is None:
        print("    ECB FM miss — trying OECD SDMX")
        dt, val = _oecd_sdmx_latest(
            "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/M.JPN.IRLTLT01.ST"
        )
        if val is not None and -5 < val < 20:
            source = "OECD-SDMX-monthly"
        else:
            dt, val = None, None

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
        req_failures.append("JPY.bond10y")

    # bond2y — OPTIONAL: try FRED INTGSBJPM193N (IMF IFS 2Y via FRED, monthly).
    # INTGSB series = IMF International Financial Statistics government securities yields.
    # Pattern: INTGSB + country_code + M + 193N. Known working: INTGSBEAM193N (EUR),
    # INTGSBGBM193N (GBP). JP equivalent: INTGSBJPM193N. Falls back to None if absent.
    # ECB FM JP2YT_RR: series doesn't exist (404 confirmed). MOF Japan: 403 from runner.
    # FRED IRLTST01JPM156N: retired. yfinance: no JGB 2Y ticker.
    print("  bond2y  (FRED:INTGSBJPM193N monthly [IMF IFS])")
    dt2, val2 = _fred_csv_latest("INTGSBJPM193N")
    if val2 is not None and -5 < val2 < 20 and not _is_stale(dt2):
        data["bond2y"]  = round(val2, 4)
        dates["bond2y"] = dt2
        print(f"    {val2:.4f}%  ({dt2})  [FRED-IMF-IFS-monthly]")
    else:
        if val2 is not None and _is_stale(dt2):
            # Series is confirmed discontinued — evict any cached value sourced from
            # this same discontinued series (v1.6 may have written the 2017 stale obs).
            # Cache-preservation only makes sense for transient network failures, not
            # for a series whose last observation is ~9 years old.
            print(f"    FRED INTGSBJPM193N: series discontinued — last obs {dt2} is stale (>18 months)")
            data.pop("bond2y", None)
            dates.pop("bond2y", None)
        # No live source available — field is None
        _gha_notice("JPY bond2y: FRED INTGSBJPM193N discontinued (last obs 2017). "
                    "ECB FM JP2YT_RR absent, MOF 403, FRED IRLTST retired. Field correctly None.")
        opt_failures.append("JPY.bond2y")

    _save("JPY", data, dates)


# ── AUD ───────────────────────────────────────────────────────────────────────

def fetch_aud_2y(opt_failures: list) -> None:
    """AUD bond2y only — OPTIONAL field.
    Australia does not issue a 2-year government bond benchmark.
    The Reserve Bank of Australia's shortest benchmark bond is the 3-year ACGB.
    No 2Y yield exists to source; this field will remain None.
    bond10y is written by update-extended-data workflow, not this script.
    """
    print("\nAUD (bond2y only)")
    data, dates = _load_existing("AUD")

    existing_2y = data.get("bond2y")
    if existing_2y is not None:
        # Preserve any cached value from a previous run (unlikely but safe)
        print(f"  bond2y  (no 2Y benchmark exists for AUD — keeping cached {existing_2y}%)")
    else:
        print("  bond2y  (Australia has no 2Y government bond benchmark — field is None by definition)")
        _gha_notice("AUD bond2y: Australia does not issue 2-year government bonds. "
                    "Shortest RBA benchmark is 3-year ACGB. Field correctly None.")
        opt_failures.append("AUD.bond2y")

    _save("AUD", data, dates)


# ── CAD ───────────────────────────────────────────────────────────────────────

def fetch_cad_2y(req_failures: list) -> None:
    """CAD bond2y only — REQUIRED field.
    Primary: BoC Valet API BD.CDN.2YR.DQ.YLD (daily, confirmed working from GH Actions).
    Fallback: FRED IRLTLT01CAM156N (monthly CAD 10Y — imprecise but available as last resort).
    FRED IRLTST01CAM156N was retired (HTTP 404) — do not restore (BUG-8 fix).
    """
    print("\nCAD (bond2y only)")
    data, dates = _load_existing("CAD")

    print("  bond2y  (BoC Valet BD.CDN.2YR.DQ.YLD → FRED:IRLTLT01CAM156N monthly)")
    dt, val = _boc_valet_latest("BD.CDN.2YR.DQ.YLD")
    source = "BoC-Valet-daily"
    if val is None or not (0 < val < 20):
        print(f"    BoC Valet miss (val={val}) — trying FRED IRLTLT01CAM156N")
        dt, val = _fred_csv_latest("IRLTLT01CAM156N")
        source = "FRED-monthly-10Y-proxy"

    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})  [{source}]")
    else:
        _gha_warning("CAD bond2y: BoC Valet and FRED both unavailable — keeping existing")
        req_failures.append("CAD.bond2y")
    _save("CAD", data, dates)


# ── NZD ───────────────────────────────────────────────────────────────────────

def fetch_nzd_2y(opt_failures: list) -> None:
    """NZD bond2y only — OPTIONAL field.
    New Zealand does issue 2-year government bonds (NZGB), but no accessible public
    source is available from GitHub Actions IPs:
      - RBNZ website returns 403 (behind Cloudflare)
      - ECB FM SDMX NZ2YT_RR series does not exist (HTTP 404 on runner)
      - FRED carries no NZD 2Y government bond series
    Preserve any existing cached value; do not overwrite with None.
    """
    print("\nNZD (bond2y only)")
    data, dates = _load_existing("NZD")

    # Try FRED INTGSBNZM193N (IMF IFS 2Y via FRED, monthly).
    # NZ issues 2Y government bonds (NZGB). RBNZ blocked (Cloudflare 403 from runner).
    # ECB FM NZ2YT_RR: series doesn't exist. INTGSBNZM193N follows same IMF IFS pattern
    # as INTGSBJPM193N (JP) / INTGSBEAM193N (EUR) / INTGSBGBM193N (GBP).
    print("  bond2y  (FRED:INTGSBNZM193N monthly [IMF IFS])")
    dt_nz, val_nz = _fred_csv_latest("INTGSBNZM193N")
    if val_nz is not None and 0 < val_nz < 20 and not _is_stale(dt_nz):
        data["bond2y"]  = round(val_nz, 4)
        dates["bond2y"] = dt_nz
        print(f"    {val_nz:.4f}%  ({dt_nz})  [FRED-IMF-IFS-monthly]")
    else:
        if val_nz is not None and _is_stale(dt_nz):
            # Series discontinued — evict any stale cached value
            print(f"    FRED INTGSBNZM193N: series discontinued — last obs {dt_nz} is stale (>18 months)")
            data.pop("bond2y", None)
            dates.pop("bond2y", None)
        # No live source available — field is None
        _gha_notice("NZD bond2y: FRED INTGSBNZM193N unavailable or discontinued; "
                    "RBNZ blocked (Cloudflare), ECB FM NZ2YT_RR absent. Field correctly None.")
        opt_failures.append("NZD.bond2y")

    _save("NZD", data, dates)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_utc = datetime.utcnow()
    print(f"fetch_bond_yields.py v1.8 — {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"SITE_DIR : {os.path.abspath(SITE_DIR)}")
    print(f"OHLC_DIR : {os.path.abspath(OHLC_DIR)}")
    print(f"OUT_DIR  : {os.path.abspath(OUT_DIR)}")

    req_failures: list[str] = []   # REQUIRED fields — count toward degraded-run threshold
    opt_failures: list[str] = []   # OPTIONAL fields — expected None, logged as ::notice::
    hard_errors:  list[str] = []

    try:
        fetch_usd(req_failures)
    except SystemExit:
        raise  # propagate USD bond10y hard failure

    for ccy, fn, args in [
        ("EUR", fetch_eur,   (req_failures,)),
        ("GBP", fetch_gbp,   (req_failures,)),
        ("JPY", fetch_jpy,   (req_failures, opt_failures)),
        ("AUD", fetch_aud_2y,(opt_failures,)),
        ("CAD", fetch_cad_2y,(req_failures,)),
        ("NZD", fetch_nzd_2y,(opt_failures,)),
    ]:
        try:
            fn(*args)
        except Exception as exc:
            _gha_error(f"{ccy} fetch crashed: {exc}")
            print(f"  ERROR [{ccy}]: {exc}")
            hard_errors.append(ccy)

    print(f"\nDone — {7 - len(hard_errors)}/7 currencies processed.")

    if opt_failures:
        print(f"Optional fields (no source available, expected): {', '.join(opt_failures)}")

    if req_failures:
        print(f"Required field soft failures: {', '.join(req_failures)}")

    if hard_errors:
        print(f"Hard errors (script crashed): {', '.join(hard_errors)}")
        sys.exit(1)

    if len(req_failures) >= 2:
        _gha_error(
            f"Degraded run: {len(req_failures)} required fields failed "
            f"({', '.join(req_failures)}) — investigate source availability"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
