#!/usr/bin/env python3
"""
backfill_economic_calendar.py  v2.0
──────────────────────────────────────────────────────────────────────────────
One-shot historical backfill for calendar-data/calendar.json.

Injects up to 12 months of past economic event data so the Economic Surprises
modal has a full-year chart series from day one, rather than waiting for the
rolling accumulation window to fill naturally.

SOURCES (in priority order)
  1. FRED API (St Louis Fed) — api.stlouisfed.org
     Free public API. Requires FRED_API_KEY env var.
     Covers USD comprehensively. Also covers EUR/GBP/JPY/CAD monthly indicators
     via OECD MEI series hosted on FRED.

  2. OECD Data API — sdmx.oecd.org
     Free, no key required. Used for series that FRED does not expose publicly:
     AUD CPI (quarterly), NZD CPI (quarterly), NZD Unemployment, CHF Unemployment.

CHANGES vs v1.1
  - EUR CPI: replaced FPCPITOTLZGEMU (annual, World Bank) → CPHPTT01EZM659N (monthly HICP)
  - EUR GDP: added correct QoQ % computation from NAEXKP01EZQ657S index
  - JPY GDP: replaced JPNRGDPEXP (nominal level, billions yen) → JPNRGDPRQPSMEI
             with fallback to QoQ computed from GDP index JPNRGDPNQDSMEI
  - Added --force flag: re-injects FRED events even if they already have actuals
    (used to overwrite corrupted v1.0 data)
  - OECD fallback for AUD CPI, NZD CPI, NZD Unemployment, CHF Unemployment
  - Added EUR Unemployment via LRHUTTTTEZM156S with correct window handling

MERGE STRATEGY
  - Default: existing events with actuals are protected (never overwritten)
  - --force: FRED-sourced events are overwritten regardless of actuals
  - ForexFactory events are NEVER overwritten (even with --force)
  - Deduplication key: (currency, dateISO, event_name_normalised)

HOW TO RUN
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py --force
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from collections import Counter

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

FRED_API_KEY     = os.environ.get("FRED_API_KEY", "")
FRED_BASE        = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_PATH      = "calendar-data/calendar.json"
MAX_HISTORY_DAYS = 365
FETCH_TIMEOUT    = 25
RATE_LIMIT_SLEEP = 0.30   # seconds between FRED API calls

HEADERS = {
    "User-Agent": "globalinvesting-bot/2.0 (https://globalinvesting.github.io)",
}

FLAG_MAP = {
    "USD": "\U0001f1fa\U0001f1f8",
    "EUR": "\U0001f1ea\U0001f1fa",
    "GBP": "\U0001f1ec\U0001f1e7",
    "JPY": "\U0001f1ef\U0001f1f5",
    "AUD": "\U0001f1e6\U0001f1fa",
    "CAD": "\U0001f1e8\U0001f1e6",
    "CHF": "\U0001f1e8\U0001f1ed",
    "NZD": "\U0001f1f3\U0001f1ff",
}

# ── FRED series catalogue ─────────────────────────────────────────────────────
#
# as_change=True  → compute MoM or QoQ change (level series → difference)
# pct_change=True → compute percentage change instead of absolute
# quarterly=True  → use quarterly release lag (60 days vs 45)
# index_qoq=True  → compute QoQ % from an index series (not as_change math)

FRED_SERIES = {
    # ── USD ───────────────────────────────────────────────────────────────────
    "PAYEMS": {
        "event": "Non-Farm Payrolls", "currency": "USD", "impact": "high",
        "unit": "K", "as_change": True, "pct_change": False, "is_inverse": False,
    },
    "UNRATE": {
        "event": "Unemployment Rate", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CPIAUCSL": {
        "event": "CPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "CPILFESL": {
        "event": "Core CPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "PCEPI": {
        "event": "PCE Price Index (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "PCEPILFE": {
        "event": "Core PCE Price Index (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "RSXFS": {
        "event": "Retail Sales (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },
    "A191RL1Q225SBEA": {
        "event": "GDP (QoQ)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False, "quarterly": True,
    },
    "BOPGSTB": {
        "event": "Trade Balance", "currency": "USD", "impact": "medium",
        "unit": "B", "as_change": False, "is_inverse": True,
    },
    "HOUST": {
        "event": "Housing Starts", "currency": "USD", "impact": "medium",
        "unit": "K", "as_change": False, "is_inverse": False,
    },
    "PERMIT": {
        "event": "Building Permits", "currency": "USD", "impact": "medium",
        "unit": "K", "as_change": False, "is_inverse": False,
    },
    "UMCSENT": {
        "event": "Michigan Consumer Sentiment", "currency": "USD", "impact": "medium",
        "unit": "", "as_change": False, "is_inverse": False,
    },
    "PPIACO": {
        "event": "PPI (MoM)", "currency": "USD", "impact": "high",
        "unit": "%", "as_change": True, "pct_change": True, "is_inverse": False,
    },

    # ── EUR ───────────────────────────────────────────────────────────────────
    # CPHPTT01EZM659N: HICP All Items for Euro Area — monthly, YoY %
    # This is the ECB's primary inflation target measure (replaces broken FPCPITOTLZGEMU)
    "CPHPTT01EZM659N": {
        "event": "CPI (YoY)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRHUTTTTEZM156S: OECD MEI Unemployment Rate, Euro Area, Monthly
    "LRHUTTTTEZM156S": {
        "event": "Unemployment Rate", "currency": "EUR", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # NAEXKP01EZQ657S: GDP by expenditure, index 2015=100, quarterly
    # We compute QoQ % from consecutive index values
    "NAEXKP01EZQ657S": {
        "event": "GDP (QoQ)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── GBP ───────────────────────────────────────────────────────────────────
    # CPALTT01GBM659N: OECD CPI All Items, UK, Monthly, YoY %
    "CPALTT01GBM659N": {
        "event": "CPI (YoY)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRHUTTTTGBM156S: OECD Unemployment Rate, UK, Monthly
    "LRHUTTTTGBM156S": {
        "event": "Unemployment Rate", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # CLVMNACSCAB1GQUK: UK GDP chained volume, QoQ % (already % change)
    "CLVMNACSCAB1GQUK": {
        "event": "GDP (QoQ)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False, "quarterly": True,
    },

    # ── JPY ───────────────────────────────────────────────────────────────────
    # CPALTT01JPM659N: OECD CPI All Items, Japan, Monthly, YoY %
    "CPALTT01JPM659N": {
        "event": "CPI (YoY)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRHUTTTTJPM156S: OECD Unemployment Rate, Japan, Monthly
    "LRHUTTTTJPM156S": {
        "event": "Unemployment Rate", "currency": "JPY", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # JPNRGDPNQDSMEI: Japan GDP, index 2015=100, quarterly — compute QoQ %
    # Replacing broken JPNRGDPEXP (nominal level) and JPNRGDPRQPSMEI (400 error)
    "JPNRGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── AUD ───────────────────────────────────────────────────────────────────
    # AUD CPI monthly not available on FRED public API
    # AUSCPIALLQINMEI: Australia CPI, index 2015=100, quarterly — compute QoQ %
    # Note: Australia CPI is quarterly (unlike most G8 which report monthly)
    "AUSCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # LRHUTTTTAUM156S: OECD Unemployment Rate, Australia, Monthly
    "LRHUTTTTAUM156S": {
        "event": "Unemployment Rate", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # AUSGDPNQDSMEI: Australia GDP, index 2015=100, quarterly — compute QoQ %
    "AUSGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── CAD ───────────────────────────────────────────────────────────────────
    # CPALTT01CAM659N: OECD CPI All Items, Canada, Monthly, YoY %
    "CPALTT01CAM659N": {
        "event": "CPI (YoY)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRHUTTTTCAM156S: OECD Unemployment Rate, Canada, Monthly
    "LRHUTTTTCAM156S": {
        "event": "Unemployment Rate", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # CANGDPNQDSMEI: Canada GDP, index 2015=100, quarterly — compute QoQ %
    "CANGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── CHF ───────────────────────────────────────────────────────────────────
    # CPALTT01CHM659N: OECD CPI All Items, Switzerland, Monthly, YoY %
    "CPALTT01CHM659N": {
        "event": "CPI (YoY)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # CHF Unemployment: LRHUTTTTCHM156S returns HTTP 400 on public FRED API
    # Will be handled by OECD fallback below
    # CHEGDPNQDSMEI: Switzerland GDP, index 2015=100, quarterly — compute QoQ %
    "CHEGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── NZD ───────────────────────────────────────────────────────────────────
    # NZD CPI quarterly: NZLCPIALLQINMEI (index 2015=100) — compute QoQ %
    "NZLCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # NZD Unemployment: LRHUTTTTDZM156S returns 400 on FRED public API
    # Will be handled by OECD fallback below
    # NZLGDPNQDSMEI: New Zealand GDP, index 2015=100, quarterly — compute QoQ %
    "NZLGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
}

# ── OECD fallback series ──────────────────────────────────────────────────────
# Used for series that FRED does not expose publicly.
# API: https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{key}
# No API key required. Free public access.

OECD_SERIES = [
    # CHF Unemployment Rate (monthly)
    # Dataflow: OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0
    # Key: CHE.M.Y._T._T.Y._T.LR.IDX (unemployment rate)
    {
        "agency":    "OECD.SDD.TPS",
        "dataflow":  "DSD_LFS@DF_IALFS_UNE_M",
        "version":   "1.0",
        "key":       "CHE.M.Y._T._T.Y._T.LR.IDX",
        "event":     "Unemployment Rate",
        "currency":  "CHF",
        "impact":    "medium",
        "unit":      "%",
        "is_inverse": True,
        "quarterly": False,
        "index_qoq": False,
    },
    # NZD Unemployment Rate (quarterly — NZ releases unemployment quarterly)
    {
        "agency":    "OECD.SDD.TPS",
        "dataflow":  "DSD_LFS@DF_IALFS_UNE_M",
        "version":   "1.0",
        "key":       "NZL.Q.Y._T._T.Y._T.LR.IDX",
        "event":     "Unemployment Rate",
        "currency":  "NZD",
        "impact":    "high",
        "unit":      "%",
        "is_inverse": True,
        "quarterly": True,
        "index_qoq": False,
    },
]

# ── Release lag in days after reference period end ────────────────────────────
RELEASE_LAG = {
    "monthly":   45,   # ~6 weeks after month end
    "quarterly": 60,   # ~2 months after quarter end
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(value: float, unit: str, meta: dict) -> str:
    """Format a numeric value to match how ForexFactory displays it."""
    if unit == "K":
        return f"{value:.0f}K"
    elif unit == "B":
        return f"{value:.1f}B"
    elif unit == "%":
        return f"{value:.1f}%"
    else:
        return f"{value:.1f}"


def _release_date(obs_date_str: str, meta: dict) -> str:
    """
    Given the FRED observation date (period start), estimate the realistic
    release date by adding the release lag after period end.
    """
    dt = datetime.strptime(obs_date_str, "%Y-%m-%d")
    freq = "quarterly" if meta.get("quarterly") else "monthly"

    if freq == "quarterly":
        # obs_date is quarter start (e.g., 2025-01-01 = Q1 2025)
        # quarter end = last day of Q1 = 2025-03-31
        month = dt.month + 3
        year  = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        quarter_end = datetime(year, month, 1) - timedelta(days=1)
        release_dt  = quarter_end + timedelta(days=RELEASE_LAG["quarterly"])
    else:
        # obs_date is month start (e.g., 2025-01-01 = January 2025)
        # month end = 2025-01-31
        if dt.month == 12:
            month_end = datetime(dt.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(dt.year, dt.month + 1, 1) - timedelta(days=1)
        release_dt = month_end + timedelta(days=RELEASE_LAG["monthly"])

    # Never assign a future date
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    release_dt = min(release_dt, today)
    return release_dt.strftime("%Y-%m-%d")


def _norm_event(s: str) -> str:
    """Normalise event name for dedup matching against ForexFactory names."""
    # Remove month/quarter suffixes FF appends: (Feb), (Q4), (Jan 2025)
    s = re.sub(r'\s*\([A-Z][a-z]{2}\)$', '', s)
    s = re.sub(r'\s*\(Q[1-4]\)$', '', s)
    s = re.sub(r'\s*\([A-Z][a-z]{2}\s+[0-9]+\)$', '', s)
    return s.strip().lower()


# ── FRED API ──────────────────────────────────────────────────────────────────

def fred_observations(series_id: str, start_date: str) -> list[dict]:
    """Fetch all FRED observations from start_date. Returns [] on any error."""
    if not FRED_API_KEY:
        print(f"  WARNING: FRED_API_KEY not set — skipping {series_id}")
        return []

    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start_date,
        "sort_order":        "asc",
        "limit":             "1000",
    }
    try:
        r = requests.get(FRED_BASE, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code == 400:
            print(f"  WARNING: FRED {series_id} HTTP 400 — series not available on public API, skipping.")
            return []
        if r.status_code == 404:
            print(f"  WARNING: FRED {series_id} not found (404), skipping.")
            return []
        if r.status_code != 200:
            print(f"  WARNING: FRED {series_id} HTTP {r.status_code}, skipping.")
            return []
        obs = r.json().get("observations", [])
        return [
            {"date": o["date"], "value": float(o["value"])}
            for o in obs
            if o.get("value") not in (".", "", None)
        ]
    except Exception as e:
        print(f"  WARNING: FRED {series_id}: {e}")
        return []
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


# ── OECD Data API ─────────────────────────────────────────────────────────────

def oecd_observations(agency: str, dataflow: str, version: str, key: str, start_year: int) -> list[dict]:
    """
    Fetch OECD SDMX-JSON observations. Returns list of {date, value} sorted oldest-first.
    No API key required.
    """
    url = f"https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{key}"
    params = {
        "startPeriod": str(start_year),
        "format": "jsondata",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            print(f"  WARNING: OECD {dataflow}/{key} HTTP {r.status_code}, skipping.")
            return []
        data = r.json()
        # OECD SDMX-JSON structure:
        # data.dataSets[0].series → {key_str: {observations: {idx: [value, ...]}}}
        # data.structure.dimensions.observation → [{values: [{id: "2025-Q1", ...}]}]
        datasets = data.get("dataSets", [])
        structure = data.get("structure", {})
        if not datasets:
            return []

        obs_dim = structure.get("dimensions", {}).get("observation", [])
        if not obs_dim:
            return []
        periods = [v.get("id", "") for v in obs_dim[0].get("values", [])]

        series_data = datasets[0].get("series", {})
        if not series_data:
            return []

        # Take the first series key (should be only one for our specific queries)
        series_key = next(iter(series_data))
        observations = series_data[series_key].get("observations", {})

        result = []
        for idx_str, obs_val in observations.items():
            idx = int(idx_str)
            if idx >= len(periods):
                continue
            period = periods[idx]
            value = obs_val[0] if obs_val else None
            if value is None:
                continue
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                continue
            # Convert period to date string
            # Monthly: "2025-01" → "2025-01-01"
            # Quarterly: "2025-Q1" → "2025-01-01"
            if re.match(r'^\d{4}-Q[1-4]$', period):
                q = int(period[-1])
                month = (q - 1) * 3 + 1
                date_str = f"{period[:4]}-{month:02d}-01"
            elif re.match(r'^\d{4}-\d{2}$', period):
                date_str = f"{period}-01"
            else:
                continue
            result.append({"date": date_str, "value": float_val})

        result.sort(key=lambda x: x["date"])
        return result

    except Exception as e:
        print(f"  WARNING: OECD {dataflow}/{key}: {e}")
        return []
    finally:
        time.sleep(0.5)


# ── Build events from FRED ────────────────────────────────────────────────────

def build_fred_events(start_date: str) -> list[dict]:
    """Fetch all FRED series and convert to calendar.json event format."""
    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    events    = []
    skipped   = 0
    fetched   = 0

    for series_id, meta in FRED_SERIES.items():
        ccy    = meta["currency"]
        ename  = meta["event"]
        impact = meta["impact"]
        unit   = meta["unit"]

        print(f"  Fetching FRED {series_id:32} [{ccy}] {ename}")

        obs = fred_observations(series_id, start_date)
        if not obs:
            skipped += 1
            continue

        fetched += 1
        as_change  = meta.get("as_change", False)
        pct_change = meta.get("pct_change", False)
        index_qoq  = meta.get("index_qoq", False)

        for i, o in enumerate(obs):
            obs_date = o["date"]
            value    = o["value"]

            if obs_date > today_str:
                continue

            # ── Compute derived value ──────────────────────────────────────
            if index_qoq:
                # QoQ % change from index series
                if i == 0:
                    continue  # need previous period
                prev_val = obs[i - 1]["value"]
                if prev_val == 0:
                    continue
                value = round((value - prev_val) / abs(prev_val) * 100, 2)
                forecast_str = None
                if i >= 2:
                    pp = obs[i - 2]["value"]
                    if pp != 0:
                        prev_change = round((obs[i - 1]["value"] - pp) / abs(pp) * 100, 2)
                        forecast_str = _fmt(prev_change, unit, meta)
            elif as_change:
                if i == 0:
                    continue
                prev_val = obs[i - 1]["value"]
                if pct_change:
                    if prev_val == 0:
                        continue
                    value = round((value - prev_val) / abs(prev_val) * 100, 2)
                else:
                    value = round(value - prev_val, 3)
                # NFP: FRED PAYEMS is already in thousands
                if series_id == "PAYEMS":
                    value = round(value, 1)
                forecast_str = None
                if i >= 2:
                    pp2 = obs[i - 2]["value"]
                    pp1 = obs[i - 1]["value"]
                    if pct_change:
                        if pp2 != 0:
                            prev_change = round((pp1 - pp2) / abs(pp2) * 100, 2)
                            forecast_str = _fmt(prev_change, unit, meta)
                    else:
                        forecast_str = _fmt(round(pp1 - pp2, 3), unit, meta)
            else:
                # Level series — value as-is
                forecast_str = _fmt(obs[i - 1]["value"], unit, meta) if i > 0 else None

            date_iso = _release_date(obs_date, meta)

            if date_iso > today_str:
                continue

            actual_str = _fmt(value, unit, meta)

            try:
                display_date = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%-d %b")
            except (ValueError, AttributeError):
                display_date = date_iso

            events.append({
                "date":     display_date,
                "dateISO":  date_iso,
                "timeUTC":  "12:30",
                "country":  ccy,
                "currency": ccy,
                "flag":     FLAG_MAP.get(ccy, ""),
                "event":    ename,
                "impact":   impact,
                "actual":   actual_str,
                "forecast": forecast_str,
                "previous": forecast_str,
                "source":   "FRED",
            })

    print(f"\n  FRED fetch complete: {fetched} series fetched, {skipped} skipped")
    events.sort(key=lambda e: (e["dateISO"], e["currency"], e["event"]))
    return events


# ── Build events from OECD ────────────────────────────────────────────────────

def build_oecd_events(start_year: int) -> list[dict]:
    """Fetch OECD fallback series and convert to calendar.json event format."""
    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    events    = []
    fetched   = 0
    skipped   = 0

    for series in OECD_SERIES:
        ccy    = series["currency"]
        ename  = series["event"]
        impact = series["impact"]
        unit   = series["unit"]

        print(f"  Fetching OECD {series['dataflow']}/{series['key'][:20]:20} [{ccy}] {ename}")

        obs = oecd_observations(
            series["agency"], series["dataflow"], series["version"],
            series["key"], start_year
        )
        if not obs:
            skipped += 1
            continue

        fetched += 1

        for i, o in enumerate(obs):
            obs_date = o["date"]
            value    = o["value"]

            if obs_date > today_str:
                continue

            date_iso = _release_date(obs_date, series)
            if date_iso > today_str:
                continue

            actual_str = _fmt(value, unit, series)
            forecast_str = _fmt(obs[i - 1]["value"], unit, series) if i > 0 else None

            try:
                display_date = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%-d %b")
            except (ValueError, AttributeError):
                display_date = date_iso

            events.append({
                "date":     display_date,
                "dateISO":  date_iso,
                "timeUTC":  "12:30",
                "country":  ccy,
                "currency": ccy,
                "flag":     FLAG_MAP.get(ccy, ""),
                "event":    ename,
                "impact":   impact,
                "actual":   actual_str,
                "forecast": forecast_str,
                "previous": forecast_str,
                "source":   "OECD",
            })

    print(f"\n  OECD fetch complete: {fetched} series fetched, {skipped} skipped")
    events.sort(key=lambda e: (e["dateISO"], e["currency"], e["event"]))
    return events


# ── Load / save calendar.json ─────────────────────────────────────────────────

def load_calendar() -> list[dict]:
    if not os.path.exists(OUTPUT_PATH):
        print(f"  INFO: {OUTPUT_PATH} not found — will create from scratch.")
        return []
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("events", [])
    except Exception as e:
        print(f"  WARNING: Could not read {OUTPUT_PATH} — {e}")
        return []


def save_calendar(events: list[dict], now_utc: datetime) -> None:
    events_sorted = sorted(events, key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", ""), e.get("currency", "")))

    all_dates   = [e["dateISO"] for e in events_sorted if e.get("dateISO")]
    range_from  = min(all_dates) if all_dates else ""
    range_to    = max(all_dates) if all_dates else ""
    ccy_dist    = dict(Counter(e.get("currency", "") for e in events_sorted))
    impact_dist = dict(Counter(e.get("impact", "") for e in events_sorted))

    output = {
        "lastUpdate":     now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generatedAt":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetchMode":      "FRED + OECD backfill + ForexFactory rolling",
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         "FRED / OECD / ForexFactory",
        "errorMessage":   None,
        "fetchErrors":    [],
        "rangeFrom":      range_from,
        "rangeTo":        range_to,
        "totalEvents":    len(events_sorted),
        "currencyCounts": ccy_dist,
        "impactCounts":   impact_dist,
        "events":         events_sorted,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate before writing

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    force_overwrite = "--force" in sys.argv

    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] backfill_economic_calendar.py v2.0")
    print(f"  MAX_HISTORY_DAYS: {MAX_HISTORY_DAYS}")
    print(f"  Force-overwrite FRED events: {force_overwrite}")

    if not FRED_API_KEY:
        print("\n  ERROR: FRED_API_KEY environment variable not set.")
        print("  Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then run: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py")
        sys.exit(1)

    # Backfill window
    cutoff_old = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    cutoff_new = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
    fred_start = (now_utc - timedelta(days=MAX_HISTORY_DAYS + 90)).strftime("%Y-%m-%d")
    oecd_start_year = (now_utc - timedelta(days=MAX_HISTORY_DAYS + 90)).year

    print(f"  Backfill window: {cutoff_old} → {cutoff_new}")
    print(f"  FRED fetch start: {fred_start}")
    print(f"  OECD fetch start year: {oecd_start_year}\n")

    # ── Step 1: Load existing calendar ────────────────────────────────────────
    existing_events = load_calendar()
    print(f"  Existing calendar.json: {len(existing_events)} events")

    # Build dedup key set.
    # With --force: protect only ForexFactory events (never overwrite real data).
    # Without --force: protect all events with actuals (safe default).
    protected_keys: set[tuple] = set()
    for ev in existing_events:
        if ev.get("actual"):
            source = ev.get("source", "")
            # With --force: only protect non-FRED/non-OECD events
            if force_overwrite and source in ("FRED", "OECD"):
                continue  # allow overwrite of bad backfill data
            key = (
                ev.get("currency", ""),
                ev.get("dateISO", ""),
                _norm_event(ev.get("event", "")),
            )
            protected_keys.add(key)

    print(f"  Protected events (will not be overwritten): {len(protected_keys)}")

    # With --force: remove existing FRED/OECD events so they can be re-injected
    if force_overwrite:
        before = len(existing_events)
        existing_events = [
            e for e in existing_events
            if e.get("source", "") not in ("FRED", "OECD")
        ]
        removed = before - len(existing_events)
        print(f"  --force: removed {removed} existing FRED/OECD events for re-injection\n")

    # ── Step 2: Fetch from FRED ────────────────────────────────────────────────
    print(f"  Fetching FRED series ({len(FRED_SERIES)} total)...")
    fred_events = build_fred_events(fred_start)
    print(f"  FRED raw events generated: {len(fred_events)}")

    # ── Step 3: Fetch from OECD fallback ──────────────────────────────────────
    print(f"\n  Fetching OECD fallback series ({len(OECD_SERIES)} total)...")
    oecd_events = build_oecd_events(oecd_start_year)
    print(f"  OECD raw events generated: {len(oecd_events)}")

    # ── Step 4: Filter to backfill window and deduplicate ─────────────────────
    all_new_events = fred_events + oecd_events
    injected   = 0
    duplicates = 0
    out_window = 0

    for ev in all_new_events:
        date_iso = ev.get("dateISO", "")

        if date_iso < cutoff_old or date_iso > cutoff_new:
            out_window += 1
            continue

        key = (
            ev.get("currency", ""),
            date_iso,
            _norm_event(ev.get("event", "")),
        )

        if key in protected_keys:
            duplicates += 1
            continue

        existing_events.append(ev)
        protected_keys.add(key)
        injected += 1

    print(f"\n  Backfill results:")
    print(f"    Injected:      {injected} new events")
    print(f"    Duplicates:    {duplicates} (already in calendar.json — preserved)")
    print(f"    Out of window: {out_window} (outside {cutoff_old}–{cutoff_new})")

    if injected == 0 and not force_overwrite:
        print("\n  INFO: Nothing to inject — calendar.json already has full coverage.")
        print("  Run with --force to re-inject FRED/OECD events (e.g. to fix corrupted data).")
        print("  No changes written.")
        sys.exit(0)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    save_calendar(existing_events, now_utc)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    all_dates    = [e["dateISO"] for e in existing_events if e.get("dateISO")]
    range_from   = min(all_dates) if all_dates else "—"
    range_to     = max(all_dates) if all_dates else "—"
    with_actuals = sum(1 for e in existing_events if e.get("actual") not in (None, ""))
    in_90d       = sum(1 for e in existing_events
                       if e.get("actual") and e.get("dateISO", "") >= (now_utc - timedelta(days=90)).strftime("%Y-%m-%d"))

    by_ccy = Counter(e.get("currency", "") for e in existing_events if e.get("actual") not in (None, ""))

    print(f"\n  {'=' * 41}")
    print(f"    ECONOMIC CALENDAR BACKFILL SUMMARY")
    print(f"  {'=' * 41}")
    print(f"  Total events:         {len(existing_events)}")
    print(f"  With actuals:         {with_actuals}")
    print(f"  In 90d window:        {in_90d}")
    print(f"  Injected this run:    {injected}")
    print(f"  Date range:           {range_from} → {range_to}")
    print(f"  Coverage by currency: {dict(sorted(by_ccy.items()))}")
    print(f"  {'=' * 41}")
    print(f"\n✓ Backfill complete. Run update-economic-calendar.yml to continue rolling accumulation.")


if __name__ == "__main__":
    main()
