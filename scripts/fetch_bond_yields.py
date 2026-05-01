"""
fetch_bond_yields.py  v1.0  —  Bond yields for USD / EUR / GBP / JPY

Produces:
    extended-data/USD.json  →  bond10y, bond2y, bond5y, vix
    extended-data/EUR.json  →  bond10y
    extended-data/GBP.json  →  bond10y
    extended-data/JPY.json  →  bond10y

Sources (all public, no API key required):
    USD bond10y  →  ohlc-data/us10y.json on disk  (written by update-ohlc at 22:30 UTC)
    USD vix      →  ohlc-data/vix.json on disk     (written by update-ohlc at 22:30 UTC)
    USD bond2y   →  FRED public CSV  (DGS2)
    USD bond5y   →  FRED public CSV  (DGS5)
    EUR bond10y  →  ECB SDMX daily yield curve (YC flow, B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y)
    GBP bond10y  →  BOE SDIE daily CSV (IUDMNZC — 10Y nominal zero coupon gilt)
    JPY bond10y  →  ECB FM SDMX monthly (FM.M.JP.JPY.4F.BB.JP10YT_RR.YLDA — nominal)

Reads SITE_DIR env var (set by workflow to '.') to locate ohlc-data/ and extended-data/.
Output format matches update_extended_data.py:  {"data": {...}, "dates": {...}}
Existing keys in extended-data/XX.json that are NOT managed here (e.g. consumerConfidence)
are preserved — only bond10y / bond2y / bond5y / vix are overwritten.
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

HEADERS = {"User-Agent": "globalinvesting-bot/1.0 (https://globalinvesting.github.io)"}

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
            dt, val = rows[-1]
            return dt, val
        return None, None
    except Exception as e:
        print(f"    FRED CSV {series_id}: {e}")
        return None, None


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


def _ecb_sdmx_latest(flow: str, key: str) -> tuple[str | None, float | None]:
    """
    Fetch latest observation from ECB SDMX REST API.

    flow examples: 'YC' (yield curve), 'FM' (financial markets)
    key examples:
        YC: 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y'
        FM: 'M.JP.JPY.4F.BB.JP10YT_RR.YLDA'
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
        # obs keys are string indices into time_dim
        for idx_str, vals in sorted(obs.items(), key=lambda x: int(x[0]), reverse=True):
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(time_dim):
                period = time_dim[idx].get("id", "")
                # Monthly periods come as "2026-03" — append "-01" for consistency
                date_str = period if len(period) == 10 else period + "-01"
                try:
                    return date_str, float(val)
                except (ValueError, TypeError):
                    continue
        return None, None
    except Exception as e:
        print(f"    ECB {flow}/{key}: {e}")
        return None, None


def _boe_latest(series_code: str) -> tuple[str | None, float | None]:
    """Fetch latest daily observation from BOE SDIE CSV endpoint."""
    try:
        # BOE SDIE provides a rolling 5-year window in CSV format
        end_date   = date.today().strftime("%d/%b/%Y")
        start_date = (date.today() - timedelta(days=30)).strftime("%d/%b/%Y")
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        params = {
            "Travel": "NIxRSx",
            "FromSeries": "1",
            "ToSeries": "50",
            "DAT": "RNG",
            "FD": start_date.split("/")[0],
            "FM": start_date.split("/")[1],
            "FY": start_date.split("/")[2],
            "TD": end_date.split("/")[0],
            "TM": end_date.split("/")[1],
            "TY": end_date.split("/")[2],
            "VFD": "Y",
            "html.x": "66",
            "html.y": "26",
            "C": series_code,
            "Filter": "N",
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            print(f"    BOE {series_code}: HTTP {r.status_code}")
            return None, None
        rows = []
        reader = csv.reader(StringIO(r.text))
        header_passed = False
        for row in reader:
            if not row:
                continue
            if not header_passed:
                if row[0].strip().upper() == "DATE":
                    header_passed = True
                continue
            if len(row) < 2 or not row[1].strip():
                continue
            date_str_raw = row[0].strip()
            val_str = row[1].strip()
            if val_str in (".", "", "N/A"):
                continue
            # BOE date format: 'DD Mmm YYYY'
            for fmt in ("%d %b %Y", "%d/%b/%Y", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_str_raw, fmt)
                    rows.append((dt.strftime("%Y-%m-%d"), float(val_str)))
                    break
                except (ValueError, TypeError):
                    continue
        if rows:
            rows.sort(key=lambda x: x[0])
            return rows[-1]
        return None, None
    except Exception as e:
        print(f"    BOE {series_code}: {e}")
        return None, None


# ── USD ───────────────────────────────────────────────────────────────────────

def fetch_usd() -> None:
    print("\nUSD")
    data, dates = _load_existing("USD")

    # bond10y — from ohlc-data/us10y.json (^TNX, already updated by update-ohlc)
    print("  bond10y  (ohlc-data/us10y.json)")
    dt, val = _ohlc_latest("us10y")
    if val is not None:
        # ^TNX is quoted as percentage points (4.35 = 4.35%)
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print("    WARN: us10y.json unavailable — bond10y unchanged")

    # vix — from ohlc-data/vix.json (^VIX)
    print("  vix  (ohlc-data/vix.json)")
    dt, val = _ohlc_latest("vix")
    if val is not None and 5 < val < 100:
        data["vix"]  = round(val, 2)
        dates["vix"] = dt
        print(f"    {val:.2f}  ({dt})")
    else:
        print(f"    WARN: vix value {val} out of range or unavailable — vix unchanged")

    # bond2y — FRED DGS2 daily (public CSV, no key)
    print("  bond2y  (FRED:DGS2 public CSV)")
    dt, val = _fred_csv_latest("DGS2")
    if val is not None and 0 < val < 20:
        data["bond2y"]  = round(val, 4)
        dates["bond2y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print(f"    WARN: DGS2 val={val} — bond2y unchanged")

    # bond5y — FRED DGS5 daily (public CSV, no key)
    print("  bond5y  (FRED:DGS5 public CSV)")
    dt, val = _fred_csv_latest("DGS5")
    if val is not None and 0 < val < 20:
        data["bond5y"]  = round(val, 4)
        dates["bond5y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print(f"    WARN: DGS5 val={val} — bond5y unchanged")

    _save("USD", data, dates)


# ── EUR ───────────────────────────────────────────────────────────────────────

def fetch_eur() -> None:
    print("\nEUR")
    data, dates = _load_existing("EUR")

    # bond10y — ECB SDMX daily yield curve
    print("  bond10y  (ECB SDMX YC daily)")
    dt, val = _ecb_sdmx_latest("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print(f"    WARN: ECB YC val={val} — bond10y unchanged")

    _save("EUR", data, dates)


# ── GBP ───────────────────────────────────────────────────────────────────────

def fetch_gbp() -> None:
    print("\nGBP")
    data, dates = _load_existing("GBP")

    # bond10y — BOE SDIE IUDMNZC (10-year nominal zero coupon gilt, daily)
    print("  bond10y  (BOE SDIE:IUDMNZC daily)")
    dt, val = _boe_latest("IUDMNZC")
    if val is not None and 0 < val < 20:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print(f"    WARN: BOE IUDMNZC val={val} — bond10y unchanged")

    _save("GBP", data, dates)


# ── JPY ───────────────────────────────────────────────────────────────────────

def fetch_jpy() -> None:
    print("\nJPY")
    data, dates = _load_existing("JPY")

    # bond10y — ECB FM SDMX monthly (nominal JGB 10Y)
    # Series: FM.M.JP.JPY.4F.BB.JP10YT_RR.YLDA (period average, nominal — without R_ prefix)
    print("  bond10y  (ECB FM SDMX monthly — JP10YT_RR nominal)")
    dt, val = None, None
    for key in (
        "M.JP.JPY.4F.BB.JP10YT_RR.YLDA",  # nominal period average (confirmed working)
        "M.JP.JPY.4F.BB.JP10YT_RR.YLD",   # nominal end-of-period variant
    ):
        dt, val = _ecb_sdmx_latest("FM", key)
        if val is not None and -5 < val < 20:
            break
        dt, val = None, None

    if val is not None:
        data["bond10y"]  = round(val, 4)
        dates["bond10y"] = dt
        print(f"    {val:.4f}%  ({dt})")
    else:
        print("    WARN: ECB FM JPY val unavailable — bond10y unchanged")

    _save("JPY", data, dates)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_utc = datetime.utcnow()
    print(f"fetch_bond_yields.py v1.0 — {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"SITE_DIR : {os.path.abspath(SITE_DIR)}")
    print(f"OHLC_DIR : {os.path.abspath(OHLC_DIR)}")
    print(f"OUT_DIR  : {os.path.abspath(OUT_DIR)}")

    errors = []
    for ccy, fn in [("USD", fetch_usd), ("EUR", fetch_eur), ("GBP", fetch_gbp), ("JPY", fetch_jpy)]:
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR [{ccy}]: {exc}")
            errors.append(ccy)

    print(f"\nDone — {4 - len(errors)}/4 currencies processed.")
    if errors:
        print(f"Errors: {', '.join(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
