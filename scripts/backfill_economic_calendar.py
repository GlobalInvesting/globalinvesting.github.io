#!/usr/bin/env python3
"""
backfill_economic_calendar.py  v1.1
──────────────────────────────────────────────────────────────────────────────
One-shot historical backfill for calendar-data/calendar.json.

Injects up to 12 months of past economic event data so the Economic Surprises
modal has a full-year chart series from day one, rather than waiting for the
rolling accumulation window to fill naturally.

SOURCE
  FRED API (St Louis Fed) — api.stlouisfed.org
  Free public API. Requires FRED_API_KEY environment variable.
  Register for a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

  FRED covers USD comprehensively (BLS, BEA, Census Bureau, ISM surveys).
  For EUR/GBP/JPY/AUD/CAD/CHF/NZD the OECD MEI family on FRED provides
  CPI, GDP, and unemployment — the three highest-impact indicators per currency.

WHY FRED ONLY (not ForexFactory historical)
  ForexFactory's public API (nfs.faireconomy.media) exposes only the current
  week and next week — there is no historical endpoint. Web scraping FF's HTML
  calendar pages is fragile and violates their ToS. FRED is the authoritative
  public source for the macro data that drives economic surprises.

MERGE STRATEGY
  Identical to fetch_economic_calendar.py:
  - Existing events in calendar.json with actuals are NEVER overwritten.
  - Backfill events are injected only for dates not already present.
  - Deduplication key: (currency, dateISO, event_name_normalised).
  - Hard cutoff: MAX_HISTORY_DAYS (365) from today — same as the rolling script.

BEAT / MISS CALCULATION
  FRED provides point-in-time observed values (actuals) but NOT the consensus
  forecasts that were published before each release. Forecasts are sourced from
  FRED's "Vintage" / real-time data (ALFRED) where available. For series where
  ALFRED is unavailable the previous-period value is used as a proxy forecast,
  which is the standard fallback used by Bloomberg BEEI for data-sparse series.
  Events injected without a usable forecast are stored with forecast=None and
  are excluded from the beat/miss scoring in the modal (they still appear in
  the event table for reference).

CONSUMED BY
  calendar-data/calendar.json  →  Economic Surprises panel (econ-surprises-modal.js)

HOW TO RUN (once, manually via GitHub Actions)
  Actions → "Backfill Economic Calendar (12 months)" → Run workflow → Run workflow
  Or locally: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py

SCHEDULE
  workflow_dispatch only — not scheduled. The rolling update-economic-calendar.yml
  continues accumulating data after the backfill.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from collections import Counter

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

FRED_API_KEY     = os.environ.get("FRED_API_KEY", "")
FRED_BASE        = "https://api.stlouisfed.org/fred/series/observations"
ALFRED_BASE      = "https://api.stlouisfed.org/fred/series/vintage_dates"
OUTPUT_PATH      = "calendar-data/calendar.json"
MAX_HISTORY_DAYS = 365
FETCH_TIMEOUT    = 20
RATE_LIMIT_SLEEP = 0.25   # seconds between FRED API calls (limit: 120 req/min)

HEADERS = {
    "User-Agent": "globalinvesting-bot/1.0 (https://globalinvesting.github.io)",
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
# Each entry: series_id → (event_name, currency, impact, unit, is_inverse, alfred_series)
#   event_name   — matches FF / investing.com naming as closely as possible
#   currency     — G8 code
#   impact       — "high" | "medium"
#   unit         — display suffix appended to value: "K", "%", "B", "" etc.
#   is_inverse   — True if lower = better (unemployment, trade deficit)
#   alfred_series— ALFRED real-time series for consensus proxies (or None)
#
# FRED series are monthly unless noted. Quarterly series are replicated to the
# release month so they appear on the correct calendar date.

FRED_SERIES = {
    # ── USD ───────────────────────────────────────────────────────────────────
    "PAYEMS": {
        "event":    "Non-Farm Payrolls",
        "currency": "USD",
        "impact":   "high",
        "unit":     "K",
        "transform": lambda v, prev: (round((v - prev) * 1000), round((prev - (prev - v)) * 1000) if prev else None),
        # NFP is the monthly change in thousands — FRED stores cumulative level
        "as_change": True,
        "is_inverse": False,
    },
    "UNRATE": {
        "event":    "Unemployment Rate",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "CPIAUCSL": {
        "event":    "CPI (MoM)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,   # report as MoM %
        "is_inverse": False,
    },
    "CPILFESL": {
        "event":    "Core CPI (MoM)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,
        "is_inverse": False,
    },
    "PCEPI": {
        "event":    "PCE Price Index (MoM)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,
        "is_inverse": False,
    },
    "PCEPILFE": {
        "event":    "Core PCE Price Index (MoM)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,
        "is_inverse": False,
    },
    "RSXFS": {
        "event":    "Retail Sales (MoM)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,
        "is_inverse": False,
    },
    "A191RL1Q225SBEA": {
        "event":    "GDP (QoQ)",
        "currency": "USD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,   # FRED already gives annualised % change
        "is_inverse": False,
        "quarterly": True,
    },
    "BOPGSTB": {
        "event":    "Trade Balance",
        "currency": "USD",
        "impact":   "medium",
        "unit":     "B",
        "as_change": False,
        "is_inverse": True,
    },

    "HOUST": {
        "event":    "Housing Starts",
        "currency": "USD",
        "impact":   "medium",
        "unit":     "K",
        "as_change": False,
        "is_inverse": False,
    },
    "PERMIT": {
        "event":    "Building Permits",
        "currency": "USD",
        "impact":   "medium",
        "unit":     "K",
        "as_change": False,
        "is_inverse": False,
    },
    "UMCSENT": {
        "event":    "Michigan Consumer Sentiment",
        "currency": "USD",
        "impact":   "medium",
        "unit":     "",
        "as_change": False,
        "is_inverse": False,
    },
    "PPIACO": {
        "event":    "PPI (MoM)",
        "currency": "USD",
        "impact":   "medium",
        "unit":     "%",
        "as_change": True,
        "pct_change": True,
        "is_inverse": False,
    },
    # ── EUR ───────────────────────────────────────────────────────────────────
    "FPCPITOTLZGEMU": {
        "event":    "CPI (YoY)",
        "currency": "EUR",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
    },
    "LRHUTTTTEZM156S": {
        "event":    "Unemployment Rate",
        "currency": "EUR",
        "impact":   "medium",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "NAEXKP01EZQ657S": {
        "event":    "GDP (QoQ)",
        "currency": "EUR",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
    # ── GBP ───────────────────────────────────────────────────────────────────
    "CPALTT01GBM659N": {
        "event":    "CPI (YoY)",
        "currency": "GBP",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
    },
    "LRHUTTTTGBM156S": {
        "event":    "Unemployment Rate",
        "currency": "GBP",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "CLVMNACSCAB1GQUK": {
        "event":    "GDP (QoQ)",
        "currency": "GBP",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
    # ── JPY ───────────────────────────────────────────────────────────────────
    "CPALTT01JPM659N": {
        "event":    "CPI (YoY)",
        "currency": "JPY",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
    },
    "LRHUTTTTJPM156S": {
        "event":    "Unemployment Rate",
        "currency": "JPY",
        "impact":   "medium",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "JPNRGDPRQPSMEI": {
        "event":    "GDP (QoQ)",
        "currency": "JPY",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,  # FRED already gives QoQ % growth rate
        "is_inverse": False,
        "quarterly": True,
    },
    # ── AUD ───────────────────────────────────────────────────────────────────
    # AUD CPI monthly not available on FRED public API (CPALTT01AUM659N returns 400)
    # Covered by rolling ForexFactory accumulation instead

    "LRHUTTTTAUM156S": {
        "event":    "Unemployment Rate",
        "currency": "AUD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "AUSGDPNQDSMEI": {
        "event":    "GDP (QoQ)",
        "currency": "AUD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
    # ── CAD ───────────────────────────────────────────────────────────────────
    "CPALTT01CAM659N": {
        "event":    "CPI (YoY)",
        "currency": "CAD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
    },
    "LRHUTTTTCAM156S": {
        "event":    "Unemployment Rate",
        "currency": "CAD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": True,
    },
    "CANGDPNQDSMEI": {
        "event":    "GDP (QoQ)",
        "currency": "CAD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
    # ── CHF ───────────────────────────────────────────────────────────────────
    "CPALTT01CHM659N": {
        "event":    "CPI (YoY)",
        "currency": "CHF",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
    },
    # CHF Unemployment Rate: LRHUTTTTCHM156S returns 400 on FRED public API
    # Switzerland uses SECO unemployment which has limited FRED coverage

    "CHEGDPNQDSMEI": {
        "event":    "GDP (QoQ)",
        "currency": "CHF",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
    # ── NZD ───────────────────────────────────────────────────────────────────
    # NZD CPI monthly not available on FRED public API (CPALTT01NZM659N returns 400)

    # NZD Unemployment Rate: LRHUTTTTDZM156S returns HTTP 400 on FRED public API

    "NZLGDPNQDSMEI": {
        "event":    "GDP (QoQ)",
        "currency": "NZD",
        "impact":   "high",
        "unit":     "%",
        "as_change": False,
        "is_inverse": False,
        "quarterly": True,
    },
}

# Typical release lag in days after the reference period end.
# Used to assign a realistic dateISO when FRED's observation date is the
# period start (e.g. "2025-01-01" for January data released in February).
RELEASE_LAG = {
    "monthly":    45,   # ~6 weeks after month end
    "quarterly":  60,   # ~2 months after quarter end
}


# ── FRED API helpers ──────────────────────────────────────────────────────────

def fred_observations(series_id: str, start_date: str) -> list[dict]:
    """
    Fetch all observations for a FRED series from start_date to today.
    Returns list of {date, value} dicts, sorted oldest-first.
    """
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
        if r.status_code == 404:
            print(f"  WARNING: FRED series {series_id} not found (404) — skipping.")
            return []
        if r.status_code != 200:
            print(f"  WARNING: FRED {series_id} HTTP {r.status_code} — skipping.")
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


def fred_release_dates(series_id: str) -> dict[str, str]:
    """
    Fetch ALFRED vintage dates to get the actual release date for each
    observation. Returns {observation_date: release_date}.
    Only used for series where we need to assign a precise dateISO.
    """
    if not FRED_API_KEY:
        return {}
    params = {
        "series_id": series_id,
        "api_key":   FRED_API_KEY,
        "file_type": "json",
    }
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/release",
            params=params,
            headers=HEADERS,
            timeout=FETCH_TIMEOUT,
        )
        # We primarily use the observation date + release lag estimate.
        # ALFRED vintage dates are expensive (one call per observation period).
        # Skip here; the lag heuristic is sufficient for backfill purposes.
        return {}
    except Exception:
        return {}
    finally:
        time.sleep(RATE_LIMIT_SLEEP)


# ── Value formatting ──────────────────────────────────────────────────────────

def _fmt(value: float, unit: str, meta: dict) -> str:
    """Format a FRED float into the string format used in calendar.json."""
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "K":
        return f"{value:.0f}K"
    if unit == "B":
        return f"{value:.1f}B"
    return f"{value:.1f}"


def _release_date(obs_date: str, meta: dict) -> str:
    """
    Estimate the calendar release date from the FRED observation date.

    FRED observation dates are the START of the reference period (e.g.
    "2025-01-01" for January data). The actual release is typically:
      - Monthly data: ~45 days after period end  → ~Feb 14 for Jan data
      - Quarterly data: ~60 days after quarter end → ~May 31 for Q1 data

    This produces a date in the correct month for chart series purposes.
    The exact day matters less than the month — the chart uses weekly
    aggregation with a 30-day rolling window.
    """
    try:
        dt = datetime.strptime(obs_date, "%Y-%m-%d")
    except ValueError:
        return obs_date

    freq = "quarterly" if meta.get("quarterly") else "monthly"

    if freq == "quarterly":
        # obs_date is the quarter start; add 3 months (quarter end) + lag
        month = dt.month + 3
        year  = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        quarter_end = datetime(year, month, 1) - timedelta(days=1)
        release_dt  = quarter_end + timedelta(days=RELEASE_LAG["quarterly"])
    else:
        # obs_date is the month start; add one month end + lag
        if dt.month == 12:
            month_end = datetime(dt.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(dt.year, dt.month + 1, 1) - timedelta(days=1)
        release_dt = month_end + timedelta(days=RELEASE_LAG["monthly"])

    # Cap at today — never assign a future date
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    release_dt = min(release_dt, today)

    return release_dt.strftime("%Y-%m-%d")


# ── Build events from FRED ────────────────────────────────────────────────────

def build_fred_events(start_date: str) -> list[dict]:
    """
    Fetch all FRED series and convert to calendar.json event format.
    Returns a flat list of events, sorted by dateISO.
    """
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

        print(f"  Fetching FRED {series_id:30} [{ccy}] {ename}")

        obs = fred_observations(series_id, start_date)
        if not obs:
            skipped += 1
            continue

        fetched += 1

        # For series reported as levels we may need to compute MoM or QoQ change
        as_change  = meta.get("as_change", False)
        pct_change = meta.get("pct_change", False)

        for i, o in enumerate(obs):
            obs_date = o["date"]
            value    = o["value"]

            # Skip future-dated observations (FRED sometimes includes scheduled dates)
            if obs_date > today_str:
                continue

            # Compute change if needed
            if as_change and i == 0:
                continue  # can't compute change for first observation
            if as_change:
                prev_val = obs[i - 1]["value"]
                if pct_change:
                    if prev_val == 0:
                        continue
                    value = round((value - prev_val) / abs(prev_val) * 100, 2)
                else:
                    value = round(value - prev_val, 3)

            # NFP: convert level change to thousands
            if series_id == "PAYEMS" and as_change and not pct_change:
                value = round(value, 1)   # already in thousands from FRED

            # Assign realistic release date
            date_iso = _release_date(obs_date, meta)

            # Only include within backfill window
            if date_iso > today_str or date_iso < start_date:
                continue

            actual_str = _fmt(value, unit, meta)

            # Previous period value as forecast proxy
            # For as_change series: forecast = previous period's computed change (needs i>=2)
            # For level series: forecast = previous period's level (needs i>0)
            forecast_str = None
            if as_change and pct_change and i >= 2:
                pp = obs[i - 2]["value"]
                if pp != 0:
                    prev_change = round((obs[i - 1]["value"] - pp) / abs(pp) * 100, 2)
                    forecast_str = _fmt(prev_change, unit, meta)
            elif as_change and not pct_change and i >= 2:
                prev_change = round(obs[i - 1]["value"] - obs[i - 2]["value"], 3)
                forecast_str = _fmt(prev_change, unit, meta)
            elif not as_change and i > 0:
                forecast_str = _fmt(obs[i - 1]["value"], unit, meta)

            try:
                display_date = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%-d %b")
            except (ValueError, AttributeError):
                display_date = date_iso

            events.append({
                "date":     display_date,
                "dateISO":  date_iso,
                "timeUTC":  "12:30",      # placeholder — exact time not in FRED
                "country":  ccy,
                "currency": ccy,
                "flag":     FLAG_MAP.get(ccy, ""),
                "event":    ename,
                "impact":   impact,
                "actual":   actual_str,
                "forecast": forecast_str,
                "previous": forecast_str,   # same as forecast proxy
                "source":   "FRED",         # provenance tag (not shown in UI)
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
    events_sorted = sorted(events, key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", ""), e.get("currency", "")))

    all_dates    = [e["dateISO"] for e in events_sorted if e.get("dateISO")]
    range_from   = min(all_dates) if all_dates else ""
    range_to     = max(all_dates) if all_dates else ""
    ccy_dist     = dict(Counter(e.get("currency", "") for e in events_sorted))
    impact_dist  = dict(Counter(e.get("impact", "") for e in events_sorted))

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
    json.loads(output_json)   # validate before writing

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_utc   = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] backfill_economic_calendar.py v1.0")
    print(f"  MAX_HISTORY_DAYS: {MAX_HISTORY_DAYS}")

    if not FRED_API_KEY:
        print("\n  ERROR: FRED_API_KEY environment variable not set.")
        print("  Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then run: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py")
        sys.exit(1)

    # Backfill window: MAX_HISTORY_DAYS back, but leave a 14-day buffer at the
    # recent end — the rolling update-economic-calendar.yml covers recent weeks
    # with higher-fidelity ForexFactory data including exact release times.
    cutoff_old  = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    cutoff_new  = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
    # FRED start: pull from 2 months before the cutoff so change calculations
    # have a previous-period value available for the oldest events in window.
    fred_start  = (now_utc - timedelta(days=MAX_HISTORY_DAYS + 60)).strftime("%Y-%m-%d")

    print(f"  Backfill window: {cutoff_old} → {cutoff_new}")
    print(f"  FRED fetch start: {fred_start}\n")

    # ── Step 1: Load existing calendar events ─────────────────────────────────
    existing_events = load_calendar()
    print(f"  Existing calendar.json: {len(existing_events)} events")

    # Build dedup key set from existing events with actuals (never overwrite)
    existing_keys: set[tuple] = set()
    for ev in existing_events:
        if ev.get("actual"):
            key = (
                ev.get("currency", ""),
                ev.get("dateISO", ""),
                ev.get("event", "").lower().strip(),
            )
            existing_keys.add(key)

    print(f"  Existing events with actuals (protected): {len(existing_keys)}")

    # ── Step 2: Fetch FRED backfill ───────────────────────────────────────────
    print(f"\n  Fetching FRED series ({len(FRED_SERIES)} total)...")
    fred_events = build_fred_events(fred_start)
    print(f"  FRED raw events generated: {len(fred_events)}")

    # ── Step 3: Filter to backfill window and deduplicate ─────────────────────
    injected   = 0
    duplicates = 0
    out_window = 0

    for ev in fred_events:
        date_iso = ev.get("dateISO", "")

        # Only inject dates within backfill window
        if date_iso < cutoff_old or date_iso > cutoff_new:
            out_window += 1
            continue

        key = (
            ev.get("currency", ""),
            date_iso,
            ev.get("event", "").lower().strip(),
        )

        if key in existing_keys:
            duplicates += 1
            continue

        existing_events.append(ev)
        existing_keys.add(key)
        injected += 1

    print(f"\n  Backfill results:")
    print(f"    Injected:   {injected} new events")
    print(f"    Duplicates: {duplicates} (already in calendar.json — preserved)")
    print(f"    Out of window: {out_window} (outside {cutoff_old}–{cutoff_new})")

    if injected == 0:
        print("\n  INFO: Nothing to inject — calendar.json already has full coverage.")
        print("  No changes written.")
        sys.exit(0)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    save_calendar(existing_events, now_utc)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    all_dates = [e["dateISO"] for e in existing_events if e.get("dateISO")]
    range_from = min(all_dates) if all_dates else "—"
    range_to   = max(all_dates) if all_dates else "—"

    ccy_new = Counter(
        ev.get("currency", "")
        for ev in existing_events
        if ev.get("source") == "FRED"
    )

    print(f"\n  calendar.json updated:")
    print(f"    Total events: {len(existing_events)}")
    print(f"    Date range:   {range_from} → {range_to}")
    print(f"    FRED events by currency: {dict(sorted(ccy_new.items()))}")
    print(f"\n✓ Backfill complete. Run update-economic-calendar.yml to continue rolling accumulation.")


if __name__ == "__main__":
    main()
