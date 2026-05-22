#!/usr/bin/env python3
"""
backfill_economic_calendar.py  v3.0
──────────────────────────────────────────────────────────────────────────────
One-shot historical backfill for calendar-data/calendar.json.

Injects up to 12 months of past economic event data so the Economic Surprises
modal has a full-year chart series from day one, rather than waiting for the
rolling accumulation window to fill naturally.

SOURCES
  1. FRED API (St Louis Fed) — api.stlouisfed.org
     Free public API. Requires FRED_API_KEY env var.
     Covers USD comprehensively. Also covers EUR/GBP/JPY/AUD/CAD/CHF/NZD
     via OECD MEI and national statistics series hosted on FRED.

CHANGES vs v2.0
  - BUG FIX: EUR CPI replaced CPHPTT01EZM659N (ECB HICP — returned 0 observations
    in FRED's public catalog window) with CPALTT01EZM659N (OECD CPI All Items,
    same family as JPY/AUD/CAD/GBP/CHF — confirmed working, 12+ months of data).
  - BUG FIX: JPY GDP replaced JPNRGDPNQDSMEI (HTTP 400 on public FRED API) with
    JPNRGDPEXP (Japan nominal GDP, billions yen, seasonally adjusted) using
    index_qoq=True to compute QoQ % from consecutive level values.
  - BUG FIX: cutoff_new changed from today-14 days to today-2 days. The 14-day
    buffer was excluding FRED data for the most recent month (e.g. April 2026 data
    released in mid-May has release_date ~May 14 which was > cutoff_new May 8).
    This manifested as "no data past April" in the Economic Surprises chart.
  - CHF Unemployment: LRUN64TTCHM156S added as primary FRED series (harmonised
    unemployment, CH), replacing the broken OECD SDMX endpoint that returns 404
    from GitHub Actions IPs.
  - NZD Unemployment: LRHUTTTTDZM156S added as primary FRED series attempt;
    falls back gracefully if series returns HTTP 400.
  - Removed OECD SDMX fallback calls entirely: sdmx.oecd.org returns HTTP 403
    from GitHub Actions runner IPs — these calls were wasting 2× timeout slots
    on every run and producing 0 data.

MERGE STRATEGY
  - Default: existing events with actuals are protected (never overwritten)
  - --force: FRED events are overwritten regardless of actuals
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
    "User-Agent": "globalinvesting-bot/3.0 (https://globalinvesting.github.io)",
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
# index_qoq=True  → compute QoQ % from consecutive level/index values

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
    # CPALTT01EZM659N: OECD CPI All Items, Euro Area, Monthly, YoY %
    # SAME series family as JPY/AUD/CAD/GBP/CHF — confirmed 12+ months on FRED public API.
    # v2.0 used CPHPTT01EZM659N (ECB HICP) which returned 0 observations in FRED's
    # public catalog window and injected nothing for EUR.
    "CPALTT01EZM659N": {
        "event": "CPI (YoY)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRHUTTTTEZM156S: OECD MEI Unemployment Rate, Euro Area, Monthly
    "LRHUTTTTEZM156S": {
        "event": "Unemployment Rate", "currency": "EUR", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # NAEXKP01EZQ657S: GDP by expenditure, index 2015=100, quarterly
    "NAEXKP01EZQ657S": {
        "event": "GDP (QoQ)", "currency": "EUR", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── GBP ───────────────────────────────────────────────────────────────────
    "CPALTT01GBM659N": {
        "event": "CPI (YoY)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    "LRHUTTTTGBM156S": {
        "event": "Unemployment Rate", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # CLVMNACSCAB1GQUK: UK GDP chained volume, QoQ % (direct % change series)
    "CLVMNACSCAB1GQUK": {
        "event": "GDP (QoQ)", "currency": "GBP", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False, "quarterly": True,
    },

    # ── JPY ───────────────────────────────────────────────────────────────────
    "CPALTT01JPM659N": {
        "event": "CPI (YoY)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    "LRHUTTTTJPM156S": {
        "event": "Unemployment Rate", "currency": "JPY", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    # JPNRGDPEXP: Japan nominal GDP, billions yen, seasonally adjusted, quarterly.
    # We compute QoQ % from consecutive level values (index_qoq=True).
    # Replaces JPNRGDPNQDSMEI (HTTP 400 on FRED public API) and
    # JPNRGDPRQPSMEI (also HTTP 400). QoQ values ~±0.3–0.6% match official Cabinet
    # Office releases. The billions-yen level values are never exposed directly.
    "JPNRGDPEXP": {
        "event": "GDP (QoQ)", "currency": "JPY", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── AUD ───────────────────────────────────────────────────────────────────
    # AUD CPI is quarterly (ABS reports quarterly, not monthly)
    "AUSCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    "LRHUTTTTAUM156S": {
        "event": "Unemployment Rate", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "AUSGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "AUD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── CAD ───────────────────────────────────────────────────────────────────
    "CPALTT01CAM659N": {
        "event": "CPI (YoY)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    "LRHUTTTTCAM156S": {
        "event": "Unemployment Rate", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CANGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CAD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── CHF ───────────────────────────────────────────────────────────────────
    "CPALTT01CHM659N": {
        "event": "CPI (YoY)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
    },
    # LRUN64TTCHM156S: OECD harmonised unemployment rate, Switzerland, 15–64 age group.
    # This series IS accessible on the FRED public API (unlike LRHUTTTTCHM156S which 400s).
    # Falls back gracefully if it also returns 400.
    "LRUN64TTCHM156S": {
        "event": "Unemployment Rate", "currency": "CHF", "impact": "medium",
        "unit": "%", "as_change": False, "is_inverse": True,
    },
    "CHEGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "CHF", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },

    # ── NZD ───────────────────────────────────────────────────────────────────
    # NZD CPI is quarterly (Stats NZ reports quarterly)
    "NZLCPIALLQINMEI": {
        "event": "CPI (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
    # LRHUTTTTDZM156S: OECD MEI unemployment for NZ (DZ = NZ in OECD FRED naming).
    # Quarterly because Stats NZ HLFS releases quarterly. Falls back gracefully if 400.
    "LRHUTTTTDZM156S": {
        "event": "Unemployment Rate", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": True,
        "quarterly": True,
    },
    "NZLGDPNQDSMEI": {
        "event": "GDP (QoQ)", "currency": "NZD", "impact": "high",
        "unit": "%", "as_change": False, "is_inverse": False,
        "quarterly": True, "index_qoq": True,
    },
}

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
        # quarter end = last day of that quarter
        month = dt.month + 3
        year  = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        quarter_end = datetime(year, month, 1) - timedelta(days=1)
        release_dt  = quarter_end + timedelta(days=RELEASE_LAG["quarterly"])
    else:
        # obs_date is month start (e.g., 2025-01-01 = January 2025)
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
            print(f"  WARNING: FRED {series_id} HTTP 400 — series not on public API, skipping.")
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
                # QoQ % change from consecutive level/index values
                if i == 0:
                    continue
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
    events_sorted = sorted(
        events,
        key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", ""), e.get("currency", ""))
    )

    all_dates   = [e["dateISO"] for e in events_sorted if e.get("dateISO")]
    range_from  = min(all_dates) if all_dates else ""
    range_to    = max(all_dates) if all_dates else ""
    ccy_dist    = dict(Counter(e.get("currency", "") for e in events_sorted))
    impact_dist = dict(Counter(e.get("impact", "") for e in events_sorted))

    output = {
        "lastUpdate":     now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generatedAt":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetchMode":      "FRED backfill + ForexFactory rolling",
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         "FRED / ForexFactory",
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

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] backfill_economic_calendar.py v3.0")
    print(f"  MAX_HISTORY_DAYS: {MAX_HISTORY_DAYS}")
    print(f"  Force-overwrite FRED events: {force_overwrite}")

    if not FRED_API_KEY:
        print("\n  ERROR: FRED_API_KEY environment variable not set.")
        print("  Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then run: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py")
        sys.exit(1)

    # ── Backfill window ────────────────────────────────────────────────────────
    cutoff_old = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    # Use today-2 (not today-14) so FRED data released in the last 2 weeks is included.
    # Example: April 2026 CPI released ~May 14 would have been excluded with today-14
    # cutoff (May 8). With today-2 (May 20) it is correctly included.
    cutoff_new = (now_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    fred_start = (now_utc - timedelta(days=MAX_HISTORY_DAYS + 90)).strftime("%Y-%m-%d")

    print(f"  Backfill window: {cutoff_old} → {cutoff_new}")
    print(f"  FRED fetch start: {fred_start}\n")

    # ── Step 1: Load existing calendar ────────────────────────────────────────
    existing_events = load_calendar()
    print(f"  Existing calendar.json: {len(existing_events)} events")

    protected_keys: set[tuple] = set()
    for ev in existing_events:
        if ev.get("actual"):
            source = ev.get("source", "")
            if force_overwrite and source in ("FRED", "OECD"):
                continue
            key = (
                ev.get("currency", ""),
                ev.get("dateISO", ""),
                _norm_event(ev.get("event", "")),
            )
            protected_keys.add(key)

    print(f"  Protected events (will not be overwritten): {len(protected_keys)}")

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

    # ── Step 3: Filter to backfill window and deduplicate ─────────────────────
    injected   = 0
    duplicates = 0
    out_window = 0

    for ev in fred_events:
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
        print("  Run with --force to re-inject FRED events (e.g. to fix corrupted data).")
        print("  No changes written.")
        sys.exit(0)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    save_calendar(existing_events, now_utc)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    all_dates    = [e["dateISO"] for e in existing_events if e.get("dateISO")]
    range_from   = min(all_dates) if all_dates else "—"
    range_to     = max(all_dates) if all_dates else "—"
    with_actuals = sum(1 for e in existing_events if e.get("actual") not in (None, ""))
    in_90d       = sum(
        1 for e in existing_events
        if e.get("actual") and
        e.get("dateISO", "") >= (now_utc - timedelta(days=90)).strftime("%Y-%m-%d")
    )

    by_ccy = Counter(
        e.get("currency", "") for e in existing_events
        if e.get("actual") not in (None, "")
    )

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
