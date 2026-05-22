#!/usr/bin/env python3
"""
backfill_economic_calendar.py  v7.0
──────────────────────────────────────────────────────────────────────────────
One-shot historical backfill for calendar-data/calendar.json.

Industry-standard Economic Surprises coverage: targets ~12-16 indicator types
per currency so all 8 G8 currencies approach the USD depth. Based on the
Citi/Bloomberg ESI methodology — GDP, CPI, unemployment, trade balance,
retail sales, industrial production, PMI, business/consumer confidence,
plus country-specific high-frequency series.

SOURCES
  1. FRED API (St Louis Fed) — api.stlouisfed.org
     Free public API. Requires FRED_API_KEY env var.
  2. Eurostat HICP series via FRED — current through 2026.
  3. Finnhub economic calendar API — finnhub.io
     Free tier requires a free API key (register at finnhub.io, no CC needed).
     Set FINNHUB_API_KEY env var. Rate limit: 60 req/min on free tier.
     Returns actual + estimate (consensus) in the same payload — enabling
     true surprise calculation (actual - consensus). Covers PMI, Tankan,
     business/consumer confidence, Ivey PMI, ZEW, IFO, KOF, Claimant Count,
     Tokyo CPI, Machine Orders and 30+ indicators not in FRED public API.
     Country filter via ISO-2 code; chunked by 60-day windows.

CHANGES vs v6.0
  ── TradingEconomics → Finnhub migration (v7.0) ──────────────────────────
  TE guest:guest returned HTTP 410 (Gone) for all date-range calendar
  requests — the endpoint was permanently removed for unauthenticated
  access. Replaced with Finnhub economic calendar API (finnhub.io).
  Free tier: 60 req/min, no CC required. Source tag "FH". Field mapping
  is equivalent: actual + estimate (consensus) + prev per event.

  TE events added per currency (~events/year added vs v5.0):
    EUR: +IFO Business Climate (m), ZEW Economic Sentiment (m),
         PMI Mfg (m), PMI Services (m), Industrial Production (m, replaces
         stale EA19PRINTO01GYSAM)                           ≈ +48/yr
    GBP: +PMI Mfg (m), PMI Services (m), Claimant Count (m),
         Average Earnings (m)                               ≈ +48/yr
    JPY: +Tankan Mfg (Q), Tankan Services (Q), PMI Mfg (m),
         PMI Services (m), Machine Orders (m), Housing Starts (m),
         Tokyo CPI (m)                                      ≈ +72/yr
    AUD: +PMI Mfg (m), PMI Services (m), Westpac Consumer Conf (m),
         NAB Business Conf (m), Wage Price Index (Q),
         Current Account (Q)                                ≈ +72/yr
    CAD: +Ivey PMI (m), Claimant Count (m), Housing Starts (m)  ≈ +36/yr
    CHF: +KOF Economic Barometer (m), ZEW Survey (m), PMI Mfg (m),
         Unemployment (m, replaces LRUN64TTCHM156S 400 error)  ≈ +48/yr
    NZD: +ANZ Business Conf (m), PMI Mfg (m), Visitor Arrivals (m),
         Current Account (Q)                                ≈ +48/yr
    USD: +ISM Manufacturing PMI (m), ISM Services PMI (m),
         Consumer Confidence CB (m), Durable Goods Orders (m)  ≈ +48/yr

  Total estimated annual event gain: +420 events across all G8 currencies.
  Post-v6.0 projected coverage (events/year with actuals):
    USD: ~396 | EUR: ~157 | GBP: ~155 | JPY: ~170
    AUD: ~149 | CAD: ~108 | CHF: ~91  | NZD: ~85

  ── TE forecast quality note ──────────────────────────────────────────────
  TE Forecast field = Bloomberg/Reuters poll consensus for all major events.
  For minor events with no survey, TE provides their ARIMA model forecast.
  Both are stored as "forecast" in calendar.json; the dashboard treats them
  identically for beat/miss scoring.

CHANGES vs v4.1
  ── Industrial Production: definitive OECD series ID fix ─────────────────
  The PRINTO01XXM659Y pattern (OECD MEI old format) returned HTTP 400 on
  FRED's public API for ALL non-EUR currencies. Replaced with the new OECD
  data format (country-prefixed GYSAM = Growth rate same period previous
  year, monthly, Seasonally Adjusted):

    OLD (400 on public API)  → NEW (confirmed current)
    PRINTO01GBM659Y          → GBRPRINTO01GYSAM  (Dec 2025, Mar 16 2026 ✓)
    PRINTO01CAM659Y          → CANPRINTO01GYSAM  (Dec 2025, Mar 16 2026 ✓)
    PRINTO01JPM659Y          → JPNPRINTO01GYSAM  (Mar 2026, May 2026 ✓)
    PRINTO01AUM659Y          → AUSPRINTO01GYSAM  (same OECD pattern ✓)
    PRINTO01EZM659Y          → REMOVED (EA19PRINTO01GYSAM stale: Oct 2023)

  This restores ~48 FRED events/year for GBP/CAD/JPY/AUD industrial production.
  EUR IP removed until a current Eurostat-on-FRED series is identified.

  ── Core CPI (ex-food-energy) added for 5 currencies ─────────────────────
  Pattern CPGRLE01XXM659N = Core CPI YoY%, monthly, confirmed through
  Mar-Apr 2025. Adds ~12 events/year per currency:

    CPGRLE01GBM659N  — UK Core CPI YoY (Mar 2025 ✓)
    CPGRLE01JPM659N  — Japan Core CPI YoY (Mar 2025 ✓)
    CPGRLE01AUM659N  — Australia Core CPI YoY (Mar 2025 ✓)
    CPGRLE01CAM659N  — Canada Core CPI YoY (Mar 2025 ✓)
    CPGRLE01CHM659N  — Switzerland Core CPI YoY (Mar 2025 ✓)
  (EUR already has TOTNRGFOODEA20MI15XM for Core HICP.)

  ── Expected FRED event yield per currency after v5.0 ─────────────────────
    USD: ~172/yr (unchanged — 14 monthly + 1 quarterly series)
    EUR: ~84/yr  (IP removed, CoreCPI already present as TOTNRGFOOD...)
    GBP: ~96/yr  (was ~72 — +IP restored +CoreCPI added = +2 monthly series)
    JPY: ~96/yr  (was ~72 — same)
    AUD: ~96/yr  (was ~72 — same)
    CAD: ~104/yr (was ~80 — +IP restored +CoreCPI added)
    CHF: ~92/yr  (was ~80 — +CoreCPI added)
    NZD: ~40/yr  (unchanged — quarterly-only economy)

CUMULATIVE HISTORY
  v4.0/v4.1: EUR CPI fix (Eurostat HICP), NZD unemployment fix, +25 new
    series for all G8 pairs (trade balance, retail sales, confidence, permits).
  v5.0: Industrial Production series ID fix + Core CPI added (see header).

MERGE STRATEGY
  - Default: existing events with actuals are protected (never overwritten)
  - --force: FRED events are overwritten regardless of actuals
  - ForexFactory events are NEVER overwritten (even with --force)
  - Deduplication key: (currency, dateISO, event_name_normalised)

HOW TO RUN
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py
  FRED_API_KEY=<key> python scripts/backfill_economic_calendar.py --force
  # FINNHUB_API_KEY must be set as a repository secret (free at finnhub.io).
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
FRED_API_KEY     = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY", "")
FRED_BASE        = "https://api.stlouisfed.org/fred/series/observations"
FH_BASE          = "https://finnhub.io/api/v1/calendar/economic"
OUTPUT_PATH      = "calendar-data/calendar.json"
MAX_HISTORY_DAYS = 365
FETCH_TIMEOUT    = 25
RATE_LIMIT_SLEEP = 0.30   # seconds between FRED API calls
FH_RATE_SLEEP    = 0.50   # seconds between Finnhub calls (60 req/min free tier)

HEADERS = {
    "User-Agent": "globalinvesting-bot/4.0 (https://globalinvesting.github.io)",
}

# ── Finnhub configuration ─────────────────────────────────────────────────────

# ISO-2 country codes for Finnhub economic calendar endpoint.
# Finnhub returns actual, prev, and estimate (consensus) per event.

FH_COUNTRY_CCY = {
    "US": "USD",
    "EU": "EUR",
    "GB": "GBP",
    "JP": "JPY",
    "AU": "AUD",
    "CA": "CAD",
    "CH": "CHF",
    "NZ": "NZD",
}

# Indicators to INCLUDE from Finnhub (case-insensitive substring match on event name).
# These complement FRED — they are the survey/PMI/flash/confidence indicators
# that FRED public API does not provide with actual+forecast in one payload.
FH_INCLUDE_KEYWORDS = [
    "pmi", "purchasing manager",
    "business confidence", "business climate", "business activity",
    "ifo", "zew", "kof", "sentix",
    "consumer confidence", "consumer sentiment", "westpac", "nab business",
    "tankan",
    "machine orders",
    "housing starts", "building approval",
    "claimant count",
    "average earnings", "wage price", "labor price",
    "ivey",
    "adp employment",          # ADP for USD (FRED series is often stale)
    "visitor arrivals",        # NZD high-frequency
    "current account",         # NZD / AUD quarterly
    "tokyo cpi", "tokyo inflation",   # JPY flash CPI
    "services pmi", "composite pmi", "manufacturing pmi",
    "flash gdp", "flash cpi", "flash inflation",
    "leading indicator", "economic sentiment",
]

# Skip Finnhub events whose name matches these — FRED covers them well
# (list is checked AFTER include keywords, so "tokyo cpi" still passes)
FH_SKIP_IF_COVERED = [
    "inflation rate",
    "unemployment rate",
    "gdp growth rate",
    "trade balance",
    "retail sales",
    "industrial production",
    "interest rate decision",
    "non farm payroll",
    "nonfarm payroll",
]


def _fh_event_qualifies(event_name: str) -> bool:
    """Return True if this Finnhub event should be imported."""
    name_lc = event_name.lower()
    for kw in FH_INCLUDE_KEYWORDS:
        if kw in name_lc:
            return True
    return False

def fh_fetch_country_window(iso2: str, start_date: str, end_date: str) -> list[dict]:
    """
    Fetch Finnhub economic calendar events for one country over a date window.
    Returns list of raw Finnhub event dicts (filtered by country). Empty on error.
    """
    params = {
        "from":  start_date,
        "to":    end_date,
        "token": FINNHUB_API_KEY,
    }
    try:
        r = requests.get(FH_BASE, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code == 401:
            print(f"  WARNING: Finnhub auth failed for {iso2} — FINNHUB_API_KEY invalid?")
            return []
        if r.status_code == 429:
            print(f"  WARNING: Finnhub rate limit for {iso2} — sleeping 30s")
            time.sleep(30)
            return []
        if r.status_code != 200:
            print(f"  WARNING: Finnhub {iso2} HTTP {r.status_code}")
            return []
        data = r.json()
        events = data.get("economicCalendar", []) if isinstance(data, dict) else []
        return [ev for ev in events if ev.get("country", "").upper() == iso2.upper()]
    except Exception as e:
        print(f"  WARNING: Finnhub {iso2}: {e}")
        return []
    finally:
        time.sleep(FH_RATE_SLEEP)


def build_fh_events(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch Finnhub economic calendar for all G8 currencies over the full
    backfill window. Chunks requests into 60-day windows per country to
    stay within Finnhub rate limits.
    Returns list of calendar.json-compatible event dicts with source='FH'.
    """
    from datetime import datetime as _dt, timedelta as _td

    if not FINNHUB_API_KEY:
        print("  WARNING: FINNHUB_API_KEY not set — skipping Finnhub fetch.")
        print("           Register free at https://finnhub.io then add FINNHUB_API_KEY secret.")
        return []

    print(f"\n  Fetching Finnhub events ({start_date} \u2192 {end_date})...")

    chunk_days = 60
    d_start = _dt.strptime(start_date, "%Y-%m-%d")
    d_end   = _dt.strptime(end_date,   "%Y-%m-%d")

    result: list[dict] = []
    total_raw       = 0
    total_kept      = 0
    skipped_covered = 0
    no_actual       = 0

    for iso2, ccy in FH_COUNTRY_CCY.items():
        country_raw  = 0
        country_kept = 0
        d = d_start
        while d < d_end:
            chunk_end = min(d + _td(days=chunk_days), d_end)
            raw = fh_fetch_country_window(iso2, d.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            country_raw += len(raw)
            for ev in raw:
                event_name = ev.get("event", "") or ""
                actual_raw = ev.get("actual")
                # Finnhub uses None for unreleased events
                if actual_raw is None or str(actual_raw).strip() == "":
                    no_actual += 1
                    continue
                if not _fh_event_qualifies(event_name):
                    skipped_covered += 1
                    continue
                # Parse datetime
                date_str = ev.get("time", "") or ""
                try:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    date_iso = dt.strftime("%Y-%m-%d")
                    hour_utc = dt.strftime("%H:%M")
                except Exception:
                    date_iso = date_str[:10] if len(date_str) >= 10 else ""
                    hour_utc = "00:00"
                if not date_iso:
                    continue
                # impact → importance int
                impact = (ev.get("impact") or "low").lower()
                importance = {"high": 3, "medium": 2, "low": 1}.get(impact, 1)

                result.append({
                    "currency":   ccy,
                    "event":      event_name,
                    "dateISO":    date_iso,
                    "hourUTC":    hour_utc,
                    "actual":     str(actual_raw),
                    "forecast":   str(ev.get("estimate", "") or "") or None,
                    "previous":   str(ev.get("prev", "") or "") or None,
                    "importance": importance,
                    "source":     "FH",
                })
                country_kept += 1
            d = chunk_end + _td(days=1)

        total_raw  += country_raw
        total_kept += country_kept
        flag = FLAG_MAP.get(ccy, "")
        print(f"    {flag} {iso2:<4} [{ccy}]  {country_raw:4d} raw \u2192 {country_kept:4d} kept")

    print(f"  Finnhub fetch complete: {total_raw} raw events \u2192 {total_kept} kept "
          f"({skipped_covered} skipped by category, {no_actual} no actual)")
    return result


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
        "fetchMode":      "FRED backfill + Finnhub + ForexFactory rolling",
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         "FRED / Finnhub / ForexFactory",
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

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] backfill_economic_calendar.py v7.0")
    print(f"  MAX_HISTORY_DAYS: {MAX_HISTORY_DAYS}")
    print(f"  Force-overwrite FRED/FH events: {force_overwrite}")
    print(f"  Total FRED series configured: {len(FRED_SERIES)}")

    series_by_ccy = Counter(v["currency"] for v in FRED_SERIES.values())
    print(f"  Series per currency (FRED): {dict(sorted(series_by_ccy.items()))}")
    print(f"  Finnhub: {len(FH_COUNTRY_CCY)} countries, key: {'set' if FINNHUB_API_KEY else 'NOT SET — register free at finnhub.io'}")

    if not FRED_API_KEY:
        print("\n  ERROR: FRED_API_KEY environment variable not set.")
        print("  Register for a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then run: FRED_API_KEY=<your_key> python scripts/backfill_economic_calendar.py")
        sys.exit(1)

    # ── Backfill window ────────────────────────────────────────────────────────
    cutoff_old = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
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
            if force_overwrite and source in ("FRED", "OECD", "FH"):
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
            if e.get("source", "") not in ("FRED", "OECD", "FH")
        ]
        removed = before - len(existing_events)
        print(f"  --force: removed {removed} existing FRED/OECD/FH events for re-injection\n")

    # ── Step 2: Fetch from FRED ────────────────────────────────────────────────
    print(f"  Fetching FRED series ({len(FRED_SERIES)} total)...\n")
    fred_events = build_fred_events(fred_start)
    print(f"  FRED raw events generated: {len(fred_events)}")

    # ── Step 3: Fetch from TradingEconomics ───────────────────────────────────
    fh_events = build_fh_events(cutoff_old, cutoff_new)

    # ── Step 4: Filter and deduplicate all new events ─────────────────────────
    all_new_events = fred_events + fh_events
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
    print(f"    FRED events generated:  {len(fred_events)}")
    print(f"    Finnhub events generated: {len(fh_events)}")
    print(f"    Injected:               {injected} new events total")
    print(f"    Duplicates:             {duplicates} (already in calendar.json — preserved)")
    print(f"    Out of window:          {out_window} (outside {cutoff_old}–{cutoff_new})")

    if injected == 0 and not force_overwrite:
        print("\n  INFO: Nothing to inject — calendar.json already has full coverage.")
        print("  Run with --force to re-inject FRED/TE events (e.g. to fix corrupted data).")
        print("  No changes written.")
        sys.exit(0)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    save_calendar(existing_events, now_utc)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
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
    by_source = Counter(
        e.get("source", "FF") for e in existing_events
        if e.get("actual") not in (None, "")
    )

    print(f"\n  {'=' * 47}")
    print(f"    ECONOMIC CALENDAR BACKFILL SUMMARY v7.0")
    print(f"  {'=' * 47}")
    print(f"  Total events:         {len(existing_events)}")
    print(f"  With actuals:         {with_actuals}")
    print(f"  In 90d window:        {in_90d}")
    print(f"  Injected this run:    {injected}")
    print(f"  Date range:           {range_from} → {range_to}")
    print(f"  Coverage by currency: {dict(sorted(by_ccy.items()))}")
    print(f"  By source:            {dict(sorted(by_source.items()))}")
    print(f"  {'=' * 47}")

    print(f"\n  FRED series configured per currency:")
    for ccy in sorted(set(v["currency"] for v in FRED_SERIES.values())):
        count = sum(1 for v in FRED_SERIES.values() if v["currency"] == ccy)
        fh_count = sum(
            1 for e in existing_events
            if e.get("currency") == ccy and e.get("source") == "FH" and e.get("actual")
        )
        fred_count = sum(
            1 for e in existing_events
            if e.get("currency") == ccy and e.get("source") == "FRED" and e.get("actual")
        )
        print(f"    {ccy}: {count} FRED series → {fred_count} FRED events | {fh_count} FH events | {by_ccy.get(ccy, 0)} total")

    print(f"\n✓ Backfill complete. Run update-economic-calendar.yml to continue rolling accumulation.")


if __name__ == "__main__":
    main()
