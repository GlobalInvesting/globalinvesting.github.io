#!/usr/bin/env python3
"""
fetch_ois_rates.py  v1.0  —  OIS / overnight rates for G8 currencies

Writes:  ois-rates/rates.json
Schema:  { "rates": { "USD": 4.33, "EUR": 2.17, ... },
           "sources": { "USD": "SOFR", ... },
           "dates":   { "USD": "2026-05-13", ... },
           "updated": "2026-05-13T23:00:00Z" }

Used by: dashboard2.js → computeCIPForward()
         Replaces CB policy rates for forward pricing.

Rate → benchmark mapping (industry standard):
  USD  SOFR      (Secured Overnight Financing Rate, NY Fed / FRED)
  EUR  €STR      (Euro Short-Term Rate, ECB / FRED)
  GBP  SONIA     (Sterling Overnight Index Average, BOE SDIE: IUDSOIA)
  JPY  TONA      (Tokyo Overnight Average Rate, BOJ Statistics)
  AUD  AONIA     (Australian Overnight Index Average, RBA Table F1)
  CAD  CORRA     (Canadian Overnight Repo Rate Average, BoC Valet)
  CHF  SARON     (Swiss Average Rate Overnight, SNB / SIX — approx 0%)
  NZD  OCR       (RBNZ Official Cash Rate — no intraday OIS market)

Source cascade per currency:
  USD   FRED CSV SOFR (daily, no key)  → FRED CSV EFFR → rates/USD.json fallback
  EUR   FRED CSV ECBESTRVOLWGTD (daily) → ECB SDMX → rates/EUR.json fallback
  GBP   BOE SDIE IUDSOIA (daily)        → FRED CSV IUDSOIA → rates/GBP.json fallback
  JPY   BOJ Statistics JSON (daily)     → rates/JPY.json fallback
  AUD   RBA Table F1 CSV (monthly-ish)  → rates/AUD.json fallback
  CAD   BoC Valet V122530 (daily)       → rates/CAD.json fallback
  CHF   SNB data API (weekly)           → rates/CHF.json fallback (policy = OIS)
  NZD   RBNZ API B2 (monthly)          → rates/NZD.json fallback

Exit policy:
  ≥5 currencies unavailable  → exit(1)
  Otherwise                  → exit(0), missing currencies fall back to policy rate

All sources are public (no API keys required).
Public repo → unlimited GitHub Actions minutes.
"""

import csv
import json
import os
import sys
from datetime import datetime, date, timedelta
from io import StringIO

import requests

# ── Config ───────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; globalinvesting-ois/1.0)",
    "Accept": "application/json, text/csv, */*",
}
TIMEOUT  = 20
SITE_DIR = os.environ.get("SITE_DIR", ".")
OUT_DIR  = os.path.join(SITE_DIR, "ois-rates")
OUT_FILE = os.path.join(OUT_DIR, "rates.json")

# Plausibility bounds (annualised %)
OIS_MIN = -2.0
OIS_MAX = 15.0

# ── FRED CSV helper ──────────────────────────────────────────────────────────

def fred_csv_latest(series_id: str) -> tuple[str | None, float | None]:
    """Fetch latest observation from FRED public CSV endpoint (no key needed)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if not r.ok:
            print(f"    FRED CSV {series_id}: HTTP {r.status_code}")
            return None, None
        rows = []
        reader = csv.reader(StringIO(r.text))
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            dt_str, val_str = row[0].strip(), row[1].strip()
            if val_str in (".", "", "N/A"):
                continue
            try:
                rows.append((dt_str, float(val_str)))
            except ValueError:
                continue
        if rows:
            return rows[-1]
        return None, None
    except Exception as e:
        print(f"    FRED CSV {series_id}: {e}")
        return None, None


# ── Policy rate fallback ─────────────────────────────────────────────────────

def policy_rate_fallback(ccy: str) -> tuple[str | None, float | None]:
    """Read most recent observation from rates/{CCY}.json as last resort."""
    path = os.path.join(SITE_DIR, "rates", f"{ccy}.json")
    try:
        with open(path) as f:
            d = json.load(f)
        obs = d.get("observations", [])
        if obs:
            return obs[0]["date"], float(obs[0]["value"])
        v = d.get("rate") or d.get("value")
        if v is not None:
            return str(date.today()), float(v)
    except Exception as e:
        print(f"    fallback {ccy}.json: {e}")
    return None, None


# ── Per-currency fetchers ────────────────────────────────────────────────────

def fetch_usd() -> tuple[str | None, float | None, str]:
    """SOFR daily → EFFR → policy fallback."""
    print("  USD: SOFR (FRED CSV)")
    dt, val = fred_csv_latest("SOFR")
    if val is not None:
        print(f"    ✓ SOFR {val:.4f}% ({dt})")
        return dt, val, "SOFR"
    print("  USD: fallback → EFFR (FRED CSV DFF)")
    dt, val = fred_csv_latest("DFF")
    if val is not None:
        print(f"    ✓ EFFR {val:.4f}% ({dt})")
        return dt, val, "EFFR"
    dt, val = policy_rate_fallback("USD")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_eur() -> tuple[str | None, float | None, str]:
    """€STR daily (FRED) → ECB SDMX → policy fallback."""
    print("  EUR: €STR (FRED CSV ECBESTRVOLWGTD)")
    dt, val = fred_csv_latest("ECBESTRVOLWGTD")
    if val is not None:
        print(f"    ✓ €STR {val:.4f}% ({dt})")
        return dt, val, "ESTR"
    # ECB SDMX FM: €STR
    print("  EUR: fallback → ECB SDMX FM")
    try:
        url = ("https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.1D._Z.C.SP00.A?"
               "lastNObservations=5&format=jsondata")
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok:
            j = r.json()
            obs_list = j["dataSets"][0]["series"]["0:0:0:0:0:0:0:0"]["observations"]
            times = j["structure"]["dimensions"]["observation"][0]["values"]
            latest_idx = max(int(k) for k in obs_list)
            val2 = obs_list[str(latest_idx)][0]
            dt2 = times[latest_idx]["id"]
            if val2 is not None:
                print(f"    ✓ ECB SDMX €STR {val2:.4f}% ({dt2})")
                return dt2, float(val2), "ESTR-ECB"
    except Exception as e:
        print(f"    ECB SDMX: {e}")
    dt, val = policy_rate_fallback("EUR")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_gbp() -> tuple[str | None, float | None, str]:
    """SONIA daily (BOE SDIE IUDSOIA) → FRED CSV → policy fallback."""
    print("  GBP: SONIA (BOE SDIE IUDSOIA)")
    try:
        now = datetime.utcnow()
        date_from = f"01/Jan/{now.year - 1}"
        date_to   = f"{now.day:02d}/{now.strftime('%b')}/{now.year}"
        url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
        params = {
            "csv.x": "yes", "Datefrom": date_from, "Dateto": date_to,
            "SeriesCodes": "IUDSOIA", "CSVF": "TN", "UsingCodes": "Y",
            "VPD": "Y", "VFD": "N",
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.ok and "html" not in r.headers.get("Content-Type", "").lower():
            rows = []
            for line in r.text.strip().split("\n"):
                line = line.strip()
                if not line or line.upper().startswith("DATE") or line.startswith('"'):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                val_str = parts[1].strip().replace("..", "").strip()
                if not val_str or val_str in (".", "", "N/A"):
                    continue
                try:
                    dt_obj = datetime.strptime(parts[0].strip(), "%d %b %Y")
                    rows.append((dt_obj.strftime("%Y-%m-%d"), float(val_str)))
                except (ValueError, TypeError):
                    continue
            if rows:
                rows.sort(key=lambda x: x[0])
                dt2, val2 = rows[-1]
                print(f"    ✓ SONIA {val2:.4f}% ({dt2})")
                return dt2, val2, "SONIA"
    except Exception as e:
        print(f"    BOE SDIE: {e}")
    print("  GBP: fallback → FRED CSV IUDSOIA")
    dt, val = fred_csv_latest("IUDSOIA")
    if val is not None:
        print(f"    ✓ FRED SONIA {val:.4f}% ({dt})")
        return dt, val, "SONIA-FRED"
    dt, val = policy_rate_fallback("GBP")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_jpy() -> tuple[str | None, float | None, str]:
    """TONA (BOJ Statistics) → policy fallback."""
    print("  JPY: TONA (BOJ Statistics)")
    try:
        # BOJ provides call rate data via their statistics search API
        # Series: FF01 = Uncollateralized overnight call rate (TONA precursor / same benchmark)
        url = "https://www.stat-search.boj.or.jp/ssi/mtsLastDataJson.do?stype=html&stat=FM&disp=FF&wc=FF01&lang=en"
        r = requests.get(url, headers={**HEADERS, "Referer": "https://www.stat-search.boj.or.jp/"}, timeout=TIMEOUT)
        if r.ok:
            text = r.text.strip()
            # Response format: [[date, value], ...]
            if text.startswith("["):
                data = json.loads(text)
                # Filter valid entries
                rows = []
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        dt_raw, val_raw = item[0], item[1]
                        try:
                            val2 = float(str(val_raw).replace(",", ""))
                            rows.append((str(dt_raw), val2))
                        except (ValueError, TypeError):
                            continue
                if rows:
                    rows.sort(key=lambda x: x[0])
                    dt2, val2 = rows[-1]
                    print(f"    ✓ TONA {val2:.4f}% ({dt2})")
                    return dt2, val2, "TONA"
    except Exception as e:
        print(f"    BOJ TONA: {e}")
    # Fallback: FRED IRSTCI01JPM156N (monthly call rate)
    print("  JPY: fallback → FRED CSV IRSTCI01JPM156N")
    dt, val = fred_csv_latest("IRSTCI01JPM156N")
    if val is not None:
        print(f"    ✓ FRED call rate {val:.4f}% ({dt})")
        return dt, val, "TONA-FRED"
    dt, val = policy_rate_fallback("JPY")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_aud() -> tuple[str | None, float | None, str]:
    """AONIA (RBA cash rate target = overnight rate) → policy fallback.
    RBA Table F1: https://www.rba.gov.au/statistics/tables/csv/f1-data.csv
    The RBA cash rate target IS the overnight rate — no separate AONIA published.
    """
    print("  AUD: AONIA via RBA Table F1 CSV")
    try:
        url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok:
            rows = []
            lines = r.text.strip().split("\n")
            # RBA CSV: first col = date (DD-Mon-YYYY), second col = cash rate target
            for line in lines[1:]:  # skip header
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) < 2:
                    continue
                dt_raw, val_raw = parts[0], parts[1]
                if not val_raw or val_raw in (".", ""):
                    continue
                try:
                    dt_obj = datetime.strptime(dt_raw, "%d-%b-%Y")
                    rows.append((dt_obj.strftime("%Y-%m-%d"), float(val_raw)))
                except (ValueError, TypeError):
                    continue
            if rows:
                rows.sort(key=lambda x: x[0])
                dt2, val2 = rows[-1]
                print(f"    ✓ AONIA/RBA {val2:.4f}% ({dt2})")
                return dt2, val2, "AONIA"
    except Exception as e:
        print(f"    RBA F1: {e}")
    dt, val = policy_rate_fallback("AUD")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_cad() -> tuple[str | None, float | None, str]:
    """CORRA (BoC Valet V122530) → FRED → policy fallback."""
    print("  CAD: CORRA (BoC Valet)")
    try:
        # BoC Valet API — CORRA series
        url = "https://www.bankofcanada.ca/valet/observations/V122530/json?recent=10"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok:
            j = r.json()
            obs = j.get("observations", [])
            # Filter non-null values
            valid = [(o["d"], float(o["V122530"]["v"])) for o in obs
                     if o.get("V122530", {}).get("v") not in (None, "")]
            if valid:
                valid.sort(key=lambda x: x[0])
                dt2, val2 = valid[-1]
                print(f"    ✓ CORRA {val2:.4f}% ({dt2})")
                return dt2, val2, "CORRA"
    except Exception as e:
        print(f"    BoC Valet CORRA: {e}")
    # Fallback: FRED CORRA series
    print("  CAD: fallback → FRED CSV CORRA")
    dt, val = fred_csv_latest("CORRA")
    if val is not None:
        print(f"    ✓ FRED CORRA {val:.4f}% ({dt})")
        return dt, val, "CORRA-FRED"
    dt, val = policy_rate_fallback("CAD")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_chf() -> tuple[str | None, float | None, str]:
    """SARON (SNB API) → policy fallback.
    SARON is published by SIX / SNB. SNB data API provides it.
    With SNB policy at 0.00%, SARON ≈ 0.00% — policy fallback is accurate.
    """
    print("  CHF: SARON (SNB data API)")
    try:
        # SNB data portal — SARON (series: SARON_OND)
        # https://data.snb.ch/api/cube/snb_saron_ond/data/json?lang=en&limit=5
        url = "https://data.snb.ch/api/cube/snb_saron_ond/data/json?lang=en&limit=5"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok:
            j = r.json()
            series_data = j.get("data", {}).get("dataSets", [{}])[0].get("series", {})
            # Parse first available series
            for key, series in series_data.items():
                obs = series.get("observations", {})
                time_values = j["data"]["structure"]["dimensions"]["observation"][0]["values"]
                valid = []
                for idx_str, vals in obs.items():
                    idx = int(idx_str)
                    if vals and vals[0] is not None:
                        valid.append((time_values[idx]["id"], float(vals[0])))
                if valid:
                    valid.sort(key=lambda x: x[0])
                    dt2, val2 = valid[-1]
                    print(f"    ✓ SARON {val2:.4f}% ({dt2})")
                    return dt2, val2, "SARON"
                break
    except Exception as e:
        print(f"    SNB SARON: {e}")
    dt, val = policy_rate_fallback("CHF")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


def fetch_nzd() -> tuple[str | None, float | None, str]:
    """OCR (RBNZ B2 API) → policy fallback.
    RBNZ is blocked by Cloudflare from GH Actions.
    OCR overnight ≈ OCR policy rate in NZ (no separate deep OIS market).
    Best available: RBNZ B2 or policy fallback.
    """
    print("  NZD: OCR (RBNZ B2 → policy fallback)")
    try:
        # RBNZ Statistics API — B2 series (OCR)
        url = "https://www.rbnz.govt.nz/api/v1/series/b2/data.json"
        r = requests.get(url, headers={**HEADERS, "Referer": "https://www.rbnz.govt.nz/"}, timeout=TIMEOUT)
        if r.ok:
            j = r.json()
            # Schema varies — try common shapes
            obs = j.get("data", j.get("observations", []))
            if isinstance(obs, list) and obs:
                rows = []
                for item in obs:
                    try:
                        dt_raw = item.get("date") or item.get("period") or item[0]
                        val_raw = item.get("value") or item.get("ocr") or item[1]
                        rows.append((str(dt_raw)[:10], float(val_raw)))
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                if rows:
                    rows.sort(key=lambda x: x[0])
                    dt2, val2 = rows[-1]
                    print(f"    ✓ RBNZ OCR {val2:.4f}% ({dt2})")
                    return dt2, val2, "RBNZ-OCR"
    except Exception as e:
        print(f"    RBNZ B2: {e}")
    dt, val = policy_rate_fallback("NZD")
    if val is not None:
        print(f"    ⚠ policy fallback {val:.4f}%")
        return dt, val, "policy-fallback"
    return None, None, "unavailable"


# ── Validation ───────────────────────────────────────────────────────────────

def is_plausible(val: float, ccy: str) -> bool:
    return OIS_MIN <= val <= OIS_MAX


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 50}\nOIS / OVERNIGHT RATES  —  G8\n{'=' * 50}")

    fetchers = {
        "USD": fetch_usd,
        "EUR": fetch_eur,
        "GBP": fetch_gbp,
        "JPY": fetch_jpy,
        "AUD": fetch_aud,
        "CAD": fetch_cad,
        "CHF": fetch_chf,
        "NZD": fetch_nzd,
    }

    rates, sources, dates = {}, {}, {}
    failures = []

    for ccy, fn in fetchers.items():
        print(f"\n[{ccy}]")
        try:
            dt, val, src = fn()
            if val is None:
                print(f"  ✗ {ccy}: no data obtained")
                failures.append(ccy)
                continue
            if not is_plausible(val, ccy):
                print(f"  ✗ {ccy}: implausible value {val:.4f}% — skipping")
                failures.append(ccy)
                continue
            rates[ccy]   = round(val, 4)
            sources[ccy] = src
            dates[ccy]   = dt or str(date.today())
            print(f"  → {ccy} = {val:.4f}%  [{src}]  ({dates[ccy]})")
        except Exception as e:
            print(f"  ✗ {ccy}: unexpected error: {e}")
            failures.append(ccy)

    print(f"\n── Summary: {len(rates)}/8 currencies loaded, {len(failures)} failures ──")
    if failures:
        print(f"   Failures: {', '.join(failures)}")

    if len(failures) >= 5:
        print("ERROR: ≥5 currencies unavailable — aborting write.")
        sys.exit(1)

    # Write output
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "rates":   rates,
        "sources": sources,
        "dates":   dates,
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✅ Written: {OUT_FILE}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
