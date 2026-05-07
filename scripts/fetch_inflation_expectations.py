"""
fetch_inflation_expectations.py  v4.1
──────────────────────────────────────
Fetches inflation expectations for all G8 currencies and writes
inflationExpectations into extended-data/{CCY}.json (non-destructive patch).

Runs weekly (Mondays 06:00 UTC) from the PUBLIC site repo — no API keys required.

CHANGELOG v4.1 (bug-fix release)
──────────────────────────────────
Six bugs fixed from the first v4.0 GHA run:

  1. headers conflict (ABS, Stats NZ): _get() was called with both headers=HEADERS (base)
     and headers=custom_headers in kwargs → TypeError. Fixed by removing the hardcoded
     headers= from requests.get(); callers now pass custom_headers kwarg explicitly.

  2. ONS URL wrong path order: /v1/datasets/mm23/timeseries/D7G7/data is 404.
     Correct ONS Timeseries API v1 path: /v1/timeseries/D7G7/dataset/mm23/data.
     Added second URL attempt with the original path as fallback.

  3. SNB cube ID wrong: pkcpival was discontinued.
     Correct cube: plkopr ("Consumer prices – total"), CSV format.
     URL: https://data.snb.ch/api/cube/plkopr/data/csv/en

  4. JPY -1.5% wrong value: e-Stat CSV parser matched the wrong 総合 row.
     The cpim.csv file has multiple sections; the first 総合 row in the context of
     前年同月比 was a sub-category. Fixed by using the e-Stat REST API directly with
     the explicit statsDataId for 消費者物価指数 All Items YoY, with URL-based CSV
     as fallback. Also added a validity guard: JPY CPI YoY must be > -3.0% (deflation
     floor — Japan has never had worse than -2.5% in modern history).

  5. StatsCan connection timeout: WDS API (getDataFromCubePidCoordAndLatestNPeriods)
     returns 404 or times out in GHA. Added FRED CANCPIALLMINMEI as immediate step 2
     before the vector resolution fallback.

  6. Stats NZ OData: same headers kwarg conflict as ABS (fixed by item 1).
"""

import csv
import json
import os
import re
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

UA = "globalinvesting-bot/4.1 (https://globalinvesting.github.io)"
BASE_HEADERS = {
    "User-Agent": UA,
    "Accept":     "application/json, text/csv;q=0.9, */*;q=0.8",
}

def _gha_warning(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)

def _gha_error(msg: str) -> None:
    print(f"::error::{msg}", flush=True)

MAX_AGE = {
    "USD": 10,
    "EUR": 10,
    "GBP": 120,
    "JPY": 120,
    "AUD": 200,
    "CAD": 120,
    "CHF": 120,
    "NZD": 200,
}

# ── Shared helpers ────────────────────────────────────────────────────────────

def _get(url: str, custom_headers: dict | None = None, **kwargs) -> "requests.Response | None":
    """
    GET with timeout. Uses BASE_HEADERS merged with custom_headers.
    Callers that need custom Accept headers pass custom_headers={"Accept": "..."}.
    This avoids the TypeError from passing headers= twice via **kwargs.
    """
    headers = {**BASE_HEADERS, **(custom_headers or {})}
    try:
        r = requests.get(url, headers=headers, timeout=25, **kwargs)
        return r
    except Exception as e:
        print(f"    GET {url[:70]}: {e}")
        return None


def index_to_yoy(obs: list) -> tuple:
    """
    Compute YoY % from list of (date_str, index_float) sorted descending.
    Finds observation closest to 12 months prior (75-day tolerance).
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
    target = d_curr - timedelta(days=365)
    best_val, best_diff = None, float("inf")
    for d, v in parsed[1:]:
        diff = abs((d - target).days)
        if diff < best_diff:
            best_diff, best_val = diff, v
    if best_val is None or best_diff > 75 or best_val == 0:
        return None, None
    return d_curr.strftime("%Y-%m-%d"), round((v_curr / best_val - 1) * 100, 4)


def is_stale(date_str: str, ccy: str) -> bool:
    if not date_str:
        return True
    try:
        age = (datetime.now(timezone.utc).replace(tzinfo=None)
               - datetime.strptime(date_str[:10], "%Y-%m-%d")).days
        return age > MAX_AGE.get(ccy, 120)
    except (ValueError, TypeError):
        return True


def _valid(val, lo: float = -5.0, hi: float = 25.0) -> bool:
    return val is not None and lo < val < hi


# ── FRED CSV (no API key) ─────────────────────────────────────────────────────

def fred_csv(series_id: str, start_year: int = 2019) -> list:
    """Returns [(date_str, float), …] sorted descending."""
    r = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    if not r or r.status_code != 200:
        if r:
            print(f"    FRED {series_id}: HTTP {r.status_code}")
        return []
    rows = []
    for row in csv.reader(StringIO(r.text)):
        if len(row) != 2 or row[0] == "DATE" or row[1].strip() in (".", "", "NA"):
            continue
        try:
            dt = datetime.strptime(row[0].strip(), "%Y-%m-%d")
            if dt.year >= start_year:
                rows.append((dt.strftime("%Y-%m-%d"), float(row[1])))
        except (ValueError, TypeError):
            continue
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def fred_index_yoy(series_id: str) -> tuple:
    """Fetch FRED index series and compute YoY %."""
    return index_to_yoy(fred_csv(series_id))


# ── ECB SDMX ─────────────────────────────────────────────────────────────────

def ecb_hicp_yoy(area_code: str = "U2") -> tuple:
    """ECB ICP SDMX — HICP YoY for Eurozone (U2) or GB (Eurostat HICP)."""
    key = f"M.{area_code}.N.000000.4.ANR"
    r = _get(f"https://data-api.ecb.europa.eu/service/data/ICP/{key}",
             params={"lastNObservations": 3, "format": "jsondata", "detail": "dataonly"})
    if not r or not r.ok:
        if r:
            print(f"    ECB HICP {area_code}: HTTP {r.status_code}")
        return None, None
    try:
        data = r.json()
        sd = data.get("dataSets", [{}])[0].get("series", {})
        dims = data.get("structure", {}).get("dimensions", {}).get("observation", [])
        time_dim = next((d.get("values", []) for d in dims if d.get("id") == "TIME_PERIOD"), [])
        if not sd or not time_dim:
            return None, None
        obs = next(iter(sd.values())).get("observations", {})
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            v = vals[0] if vals else None
            if v is not None and idx < len(time_dim):
                p = time_dim[idx].get("id", "")
                date_str = (p + "-01") if len(p) == 7 else p
                return date_str, round(float(v), 4)
    except Exception as e:
        print(f"    ECB HICP {area_code}: parse error — {e}")
    return None, None


# ── World Bank ────────────────────────────────────────────────────────────────

def wb_cpi_yoy(iso2: str) -> tuple:
    """Annual CPI YoY from World Bank — last resort."""
    r = _get(f"https://api.worldbank.org/v2/country/{iso2}/indicator/FP.CPI.TOTL.ZG",
             params={"format": "json", "mrv": 3, "per_page": 5})
    if not r or not r.ok:
        return None, None
    try:
        payload = r.json()
        if len(payload) >= 2:
            for entry in payload[1]:
                if entry.get("value") is not None:
                    yr = entry.get("date", "")
                    return f"{yr}-06-15", round(float(entry["value"]), 4)
    except Exception as e:
        print(f"    WB {iso2}: {e}")
    return None, None


# ── JSON patch ────────────────────────────────────────────────────────────────

def patch_json(ccy: str, val: float, date_str: str) -> bool:
    path = os.path.join(OUT_DIR, f"{ccy}.json")
    try:
        d = json.load(open(path)) if os.path.exists(path) else {}
        d.setdefault("data", {})
        d.setdefault("dates", {})
        d["data"]["inflationExpectations"]  = val
        d["dates"]["inflationExpectations"] = date_str
        with open(path, "w") as f:
            json.dump(d, f, separators=(",", ":"))
        return True
    except Exception as e:
        print(f"    patch_json {ccy}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# NATIONAL STATISTICS OFFICE APIs
# ═══════════════════════════════════════════════════════════════════════════════

# ── GBP: ONS (Office for National Statistics) ────────────────────────────────

def ons_cpi_yoy() -> tuple:
    """
    ONS Timeseries API — CPI All Items, 12-month rate (series D7G7, dataset mm23).
    Correct v1 path: /v1/timeseries/{series}/dataset/{dataset}/data
    Returns (date_str YYYY-MM-DD, yoy_pct) or (None, None).
    """
    # Try correct v1 path first, then the reversed path as fallback
    urls = [
        "https://api.ons.gov.uk/v1/timeseries/D7G7/dataset/mm23/data",
        "https://api.ons.gov.uk/v1/datasets/mm23/timeseries/D7G7/data",
    ]
    for url in urls:
        r = _get(url)
        if not r or r.status_code != 200:
            if r:
                print(f"    ONS API ({url[-30:]}): HTTP {r.status_code}")
            continue
        try:
            data = r.json()
            months = data.get("months", [])
            if not months:
                print(f"    ONS API: empty months — trying next URL")
                continue
            latest = months[0]
            date_str_raw = latest.get("date", "")   # e.g. "2026 MAR"
            val_str = latest.get("value", "")
            try:
                dt = datetime.strptime(date_str_raw.strip(), "%Y %b")
            except ValueError:
                try:
                    dt = datetime.strptime(date_str_raw.strip(), "%Y %B")
                except ValueError:
                    print(f"    ONS API: unparseable date '{date_str_raw}'")
                    continue
            return dt.strftime("%Y-%m-01"), round(float(val_str), 4)
        except Exception as e:
            print(f"    ONS API: parse error — {e}")
    return None, None


# ── JPY: Statistics Bureau Japan ──────────────────────────────────────────────

def estatjp_cpi_yoy() -> tuple:
    """
    Statistics Bureau Japan — CPI All Items YoY % via e-Stat REST API.
    statsDataId 0003427113 = Monthly CPI, All Japan, All Items, YoY change (前年同月比).

    Falls back to the cpim.csv approach if API fails.
    JPY validity: -3.0% < val < 8.0% (Japan's modern CPI range)
    """
    # e-Stat REST API — no key required for basic queries with limited results
    url = ("https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
           "?statsDataId=0003427113&metaGetFlg=N&cntGetFlg=N"
           "&explanationGetFlg=N&annotationGetFlg=N&sectionHeaderFlg=1"
           "&replaceSpChars=0&limit=3&startPosition=1&lang=J")
    r = _get(url)
    if r and r.status_code == 200:
        try:
            data = r.json()
            values = (data.get("GET_STATS_DATA", {})
                         .get("STATISTICAL_DATA", {})
                         .get("DATA_INF", {})
                         .get("VALUE", []))
            # Find All Items (品目コード = 0000) entries
            for v in values:
                cat = v.get("@cat01", "") or v.get("@itemCode", "")
                if cat == "0000":  # All Items
                    period = v.get("@time", "")  # e.g. "2026000003" = 2026/March
                    val_s = v.get("$", "")
                    if period and val_s and val_s not in ("…", "-", ""):
                        # Period format: YYYYMM00NN where NN = month
                        yr = period[:4]
                        mn = period[6:8] if len(period) >= 8 else "01"
                        try:
                            dt_str = f"{yr}-{mn}-01"
                            datetime.strptime(dt_str, "%Y-%m-%d")  # validate
                            val = float(val_s)
                            if _valid(val, lo=-3.0, hi=8.0):
                                return dt_str, round(val, 4)
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            print(f"    e-Stat API: {e}")

    # Fallback: Stats Japan cpim.csv — YoY table (not index table)
    return _estatjp_csv_yoy()


def _estatjp_csv_yoy() -> tuple:
    """
    Stats Japan cpim.csv — parse the 前年同月比 (YoY) section for 総合 (All Items).
    The CSV has multiple sections separated by blank lines.
    The 前年同月比 section header identifies the YoY block.
    """
    url = "https://www.stat.go.jp/data/cpi/sokuhou/tsuki/zuhyou/cpim.csv"
    r = _get(url)
    if not r or r.status_code != 200:
        if r:
            print(f"    e-Stat CSV: HTTP {r.status_code}")
        return None, None
    try:
        text = r.content.decode("shift_jis", errors="replace")
        lines = text.splitlines()

        # Find the 前年同月比 (YoY) section
        yoy_section_start = None
        for i, line in enumerate(lines):
            if "前年同月比" in line and "指数" not in line:
                yoy_section_start = i
                break

        if yoy_section_start is None:
            print("    e-Stat CSV: 前年同月比 section not found")
            return None, None

        # Within the YoY section, find the date header row and the 総合 data row
        section = lines[yoy_section_start:]
        header_cols = None
        for i, line in enumerate(section[:10]):  # Header should be within first 10 rows
            if "年" in line and "月" in line:
                cols = [c.strip() for c in line.split(",")]
                if any("年" in c for c in cols):
                    header_cols = cols
                    # Data rows follow immediately
                    for j in range(i + 1, min(i + 5, len(section))):
                        data_line = section[j]
                        if "総合" in data_line:
                            data_cols = [c.strip() for c in data_line.split(",")]
                            # Walk right-to-left to find latest non-empty value
                            for k in range(len(header_cols) - 1, 0, -1):
                                if k >= len(data_cols):
                                    continue
                                hdr = header_cols[k]
                                val_s = data_cols[k].replace(" ", "")
                                if not hdr or not val_s or "年" not in hdr:
                                    continue
                                try:
                                    hdr_clean = hdr.replace("年", "-").replace("月", "")
                                    dt = datetime.strptime(hdr_clean.strip(), "%Y-%m")
                                    val = float(val_s)
                                    if _valid(val, lo=-3.0, hi=8.0):
                                        return dt.strftime("%Y-%m-01"), round(val, 4)
                                except (ValueError, TypeError):
                                    continue
                    break
    except Exception as e:
        print(f"    e-Stat CSV: {e}")
    return None, None


# ── AUD: ABS (Australian Bureau of Statistics) SDMX API ──────────────────────

def abs_cpi_yoy() -> tuple:
    """
    ABS SDMX-JSON API — CPI All Groups Australia, quarterly.
    Uses custom_headers kwarg to avoid headers= conflict with _get().
    """
    url = "https://api.data.abs.gov.au/data/ABS,CPI,1.0.0/1.10001.10.50.Q"
    r = _get(url,
             custom_headers={"Accept": "application/vnd.sdmx.data+json;version=1.0"},
             params={"startPeriod": "2023-Q1", "detail": "dataonly"})
    if not r or r.status_code != 200:
        if r:
            print(f"    ABS SDMX: HTTP {r.status_code}")
        return None, None
    try:
        data = r.json()
        sd = data.get("dataSets", [{}])[0].get("series", {})
        dims = (data.get("structure", {})
                    .get("dimensions", {})
                    .get("observation", []))
        time_dim = next((d.get("values", []) for d in dims
                         if d.get("id") == "TIME_PERIOD"), [])
        if not sd or not time_dim:
            print("    ABS SDMX: empty response")
            return None, None
        series = next(iter(sd.values()))
        obs = series.get("observations", {})
        rows = []
        for idx_str, vals in obs.items():
            idx = int(idx_str)
            v = vals[0] if vals else None
            if v is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                if "Q" in period:
                    qmap = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
                    parts = period.split("-")
                    m = qmap.get(parts[1], "03") if len(parts) == 2 else "03"
                    rows.append((f"{parts[0]}-{m}-15", float(v)))
        rows.sort(key=lambda x: x[0], reverse=True)
        return index_to_yoy(rows)
    except Exception as e:
        print(f"    ABS SDMX: {e}")
    return None, None


# ── CAD: Statistics Canada WDS API ───────────────────────────────────────────

def statcan_cpi_yoy() -> tuple:
    """
    Statistics Canada WDS API — CPI All-Items Canada (Table 18-10-0004-01).
    Fetches 15 months of index and computes YoY.
    """
    url = ("https://www150.statcan.gc.ca/t1/tbl1/en/dtbl!"
           "getDataFromCubePidCoordAndLatestNPeriods/18100004/1.1/15")
    r = _get(url)
    if not r or r.status_code != 200:
        if r:
            print(f"    StatsCan WDS: HTTP {r.status_code}")
        return None, None
    try:
        data = r.json()
        obj = data.get("object", {})
        pts = obj.get("vectorDataPoints", [])
        rows = []
        for pt in pts:
            ref = pt.get("refPer", "")
            val = pt.get("value")
            if ref and val is not None:
                try:
                    dt = datetime.strptime(ref[:7], "%Y-%m")
                    rows.append((dt.strftime("%Y-%m-01"), float(val)))
                except (ValueError, TypeError):
                    continue
        rows.sort(key=lambda x: x[0], reverse=True)
        return index_to_yoy(rows)
    except Exception as e:
        print(f"    StatsCan WDS: {e}")
    return None, None


# ── CHF: SNB Data Portal ──────────────────────────────────────────────────────

def snb_cpi_yoy() -> tuple:
    """
    SNB Data Portal — Swiss CPI All items, monthly.
    Cube: plkopr (Consumer prices – total). CSV format (JSON endpoint discontinued).
    URL: https://data.snb.ch/api/cube/plkopr/data/csv/en
    YoY computed from 12-month index delta.
    """
    url = "https://data.snb.ch/api/cube/plkopr/data/csv/en"
    r = _get(url, custom_headers={"Accept": "text/csv, */*"})
    if not r or r.status_code != 200:
        if r:
            print(f"    SNB CSV: HTTP {r.status_code}")
        return None, None
    try:
        # SNB CSV format (after 3 header rows):
        # Date;D0;Value
        # 2026-03;T;108.42
        # D0=T = Total (all items)
        rows = []
        lines = r.text.splitlines()
        for line in lines:
            if not line or line.startswith("#") or "Date" in line or "date" in line:
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            date_s = parts[0].strip()
            dim = parts[1].strip() if len(parts) > 1 else ""
            val_s = parts[-1].strip()
            # Filter for Total (T) or single-dimension rows
            if dim not in ("T", "TOT", "", "0"):
                continue
            if not val_s or val_s in (".", "NA", "-"):
                continue
            try:
                dt = datetime.strptime(date_s[:7], "%Y-%m")
                rows.append((dt.strftime("%Y-%m-01"), float(val_s)))
            except (ValueError, TypeError):
                continue
        rows.sort(key=lambda x: x[0], reverse=True)
        if not rows:
            print("    SNB CSV: no usable rows parsed")
            return None, None
        return index_to_yoy(rows)
    except Exception as e:
        print(f"    SNB CSV: {e}")
    return None, None


# ── NZD: Stats NZ OData ───────────────────────────────────────────────────────

def statsnz_cpi_yoy() -> tuple:
    """
    Stats NZ OData API — CPI All Groups, quarterly.
    Uses custom_headers to avoid headers= conflict.
    """
    url = ("https://api.stats.govt.nz/opendata/v1/CPI?"
           "$filter=Subject eq 'Consumer Price Index - CPI' and "
           "Group eq 'All groups' and Series_title_1 eq 'All groups'"
           "&$orderby=Period desc&$top=8")
    r = _get(url, custom_headers={"Accept": "application/json"})
    if r and r.status_code == 200:
        try:
            data = r.json()
            items = data.get("value", [])
            rows = []
            for item in items:
                period = item.get("Period", "")
                val = item.get("Data_value")
                if period and val is not None:
                    period = period.replace("-", "")
                    if "Q" in period:
                        yr = period[:4]
                        q = period[5] if len(period) > 5 else period[4]
                        qmap = {"1": "03", "2": "06", "3": "09", "4": "12"}
                        m = qmap.get(q, "03")
                        rows.append((f"{yr}-{m}-15", float(val)))
            rows.sort(key=lambda x: x[0], reverse=True)
            if rows:
                return index_to_yoy(rows)
        except Exception as e:
            print(f"    Stats NZ OData: {e}")
    elif r:
        print(f"    Stats NZ OData: HTTP {r.status_code}")
    return _rbnz_survey()


def _rbnz_survey() -> tuple:
    """RBNZ Survey of Expectations — 1-year ahead inflation (CSV fallback)."""
    url = ("https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/"
           "statistics/tables/s3/hsvs.csv")
    r = _get(url)
    if not r or r.status_code != 200:
        if r:
            print(f"    RBNZ Survey CSV: HTTP {r.status_code}")
        return None, None
    try:
        lines = r.text.splitlines()
        rows_parsed = list(csv.reader(lines))
        data_rows = []
        for row in rows_parsed:
            if not row:
                continue
            if re.match(r'\d{4}\s?[Qq]\d', row[0].strip()):
                data_rows.append(row)
        if data_rows:
            latest = data_rows[-1]
            date_raw = latest[0].strip()
            q_match = re.search(r'(\d{4})\s?[Qq](\d)', date_raw)
            if q_match:
                yr, q = q_match.group(1), q_match.group(2)
                qmap = {"1": "03", "2": "06", "3": "09", "4": "12"}
                m = qmap.get(q, "03")
                date_str = f"{yr}-{m}-15"
                for col in latest[1:4]:
                    col_clean = col.strip()
                    if col_clean:
                        try:
                            val = float(col_clean)
                            if _valid(val):
                                return date_str, round(val, 4)
                        except (ValueError, TypeError):
                            continue
    except Exception as e:
        print(f"    RBNZ Survey CSV: {e}")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CURRENCY FETCH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_usd() -> tuple:
    rows = fred_csv("T5YIE")
    if rows:
        dt_s, val = rows[0]
        if _valid(val):
            print(f"    ✓ USD: {val:.4f}% ({dt_s}) [FRED T5YIE]")
            return dt_s, round(val, 4)
    print("    ✗ USD: T5YIE no data")
    return None, None


def fetch_eur() -> tuple:
    rows = fred_csv("T5YIFR")
    if rows:
        dt_s, val = rows[0]
        if _valid(val):
            print(f"    ✓ EUR: {val:.4f}% ({dt_s}) [FRED T5YIFR]")
            return dt_s, round(val, 4)
    print("    ✗ EUR: T5YIFR miss — trying ECB HICP Eurozone")
    dt, val = ecb_hicp_yoy("U2")
    if val is not None and _valid(val):
        print(f"    ~ EUR: {val:.4f}% ({dt}) [ECB HICP fallback]")
        return dt, val
    return None, None


def fetch_gbp() -> tuple:
    """GBP: ONS API → FRED GBRCPIALLMINMEI → ECB HICP GB → World Bank"""
    print("    Trying ONS API (D7G7 — CPI All Items YoY)…")
    dt, val = ons_cpi_yoy()
    if val is not None and _valid(val):
        print(f"    ✓ GBP: {val:.4f}% CPI YoY ({dt}) [ONS API]")
        return dt, val

    print("    ✗ GBP: ONS miss — trying FRED GBRCPIALLMINMEI")
    dt, val = fred_index_yoy("GBRCPIALLMINMEI")
    if val is not None and _valid(val):
        print(f"    ~ GBP: {val:.4f}% CPI YoY ({dt}) [FRED GBRCPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ GBP: FRED miss — trying ECB HICP GB")
    dt, val = ecb_hicp_yoy("GB")
    if val is not None and _valid(val):
        print(f"    ~ GBP: {val:.4f}% CPI YoY ({dt}) [ECB HICP]")
        return dt, val

    print("    ✗ GBP: ECB miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("GB")
    if val_wb is not None and _valid(val_wb):
        print(f"    ~ GBP: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_jpy() -> tuple:
    """JPY: e-Stat API → e-Stat CSV (前年同月比 section) → World Bank"""
    print("    Trying e-Stat Japan API (statsDataId 0003427113)…")
    dt, val = estatjp_cpi_yoy()
    if val is not None and _valid(val, lo=-3.0, hi=8.0):
        print(f"    ✓ JPY: {val:.4f}% CPI YoY ({dt}) [Stats Japan]")
        return dt, val

    print("    ✗ JPY: e-Stat miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("JP")
    if val_wb is not None and _valid(val_wb):
        print(f"    ~ JPY: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_aud() -> tuple:
    """AUD: ABS SDMX API → FRED AUSCPIALLQINMEI → World Bank"""
    print("    Trying ABS SDMX API…")
    dt, val = abs_cpi_yoy()
    if val is not None and _valid(val):
        print(f"    ✓ AUD: {val:.4f}% CPI YoY ({dt}) [ABS SDMX]")
        return dt, val

    print("    ✗ AUD: ABS miss — trying FRED AUSCPIALLQINMEI")
    dt, val = fred_index_yoy("AUSCPIALLQINMEI")
    if val is not None and _valid(val):
        print(f"    ~ AUD: {val:.4f}% CPI YoY ({dt}) [FRED AUSCPIALLQINMEI→YoY]")
        return dt, val

    print("    ✗ AUD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("AU")
    if val_wb is not None and _valid(val_wb):
        print(f"    ~ AUD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_cad() -> tuple:
    """CAD: StatsCan WDS API → FRED CANCPIALLMINMEI → World Bank"""
    print("    Trying Statistics Canada WDS API…")
    dt, val = statcan_cpi_yoy()
    if val is not None and _valid(val):
        print(f"    ✓ CAD: {val:.4f}% CPI YoY ({dt}) [StatsCan WDS]")
        return dt, val

    print("    ✗ CAD: StatsCan miss — trying FRED CANCPIALLMINMEI")
    dt, val = fred_index_yoy("CANCPIALLMINMEI")
    if val is not None and _valid(val):
        print(f"    ~ CAD: {val:.4f}% CPI YoY ({dt}) [FRED CANCPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ CAD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CA")
    if val_wb is not None and _valid(val_wb):
        print(f"    ~ CAD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_chf() -> tuple:
    """CHF: SNB Data Portal (plkopr CSV) → FRED CHECPIALLMINMEI → World Bank"""
    print("    Trying SNB Data Portal (plkopr)…")
    dt, val = snb_cpi_yoy()
    if val is not None and _valid(val, lo=-5.0, hi=15.0):
        print(f"    ✓ CHF: {val:.4f}% CPI YoY ({dt}) [SNB]")
        return dt, val

    print("    ✗ CHF: SNB miss — trying FRED CHECPIALLMINMEI")
    dt, val = fred_index_yoy("CHECPIALLMINMEI")
    if val is not None and _valid(val, lo=-5.0, hi=15.0):
        print(f"    ~ CHF: {val:.4f}% CPI YoY ({dt}) [FRED CHECPIALLMINMEI→YoY]")
        return dt, val

    print("    ✗ CHF: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("CH")
    if val_wb is not None and _valid(val_wb, lo=-5.0, hi=15.0):
        print(f"    ~ CHF: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


def fetch_nzd() -> tuple:
    """NZD: Stats NZ OData → RBNZ Survey → FRED NZLCPIALLQINMEI → World Bank"""
    print("    Trying Stats NZ OData API…")
    dt, val = statsnz_cpi_yoy()
    if val is not None and _valid(val):
        print(f"    ✓ NZD: {val:.4f}% CPI YoY ({dt}) [Stats NZ]")
        return dt, val

    print("    ✗ NZD: Stats NZ miss — trying FRED NZLCPIALLQINMEI")
    dt, val = fred_index_yoy("NZLCPIALLQINMEI")
    if val is not None and _valid(val):
        print(f"    ~ NZD: {val:.4f}% CPI YoY ({dt}) [FRED NZLCPIALLQINMEI→YoY]")
        return dt, val

    print("    ✗ NZD: FRED miss — trying World Bank")
    dt_wb, val_wb = wb_cpi_yoy("NZ")
    if val_wb is not None and _valid(val_wb):
        print(f"    ~ NZD: {val_wb:.4f}% ({dt_wb}) [World Bank]")
        return dt_wb, val_wb
    return None, None


# ── Dispatch ──────────────────────────────────────────────────────────────────

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
            print(f"    ✗ {ccy}: all sources failed — skipping")
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
        src = "market-implied" if ccy in ("USD", "EUR") else "CPI YoY"
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
