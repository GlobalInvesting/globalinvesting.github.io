#!/usr/bin/env python3
"""
fetch_economic_calendar.py  v3.0
──────────────────────────────────────────────────────────────────────────────
Builds calendar-data/calendar.json (Economic Surprises panel source) by
converting and merging data from calendar-data/ff_calendar.json, which is
maintained by fetch_ff_calendar.py / update-ff-calendar.yml and updates 4×/day.

WHY THIS APPROACH (v3.0)
  v2.0 scraped investing.com via a hidden POST endpoint, which worked until
  May 2026 when investing.com began returning HTTP 503 to GitHub Actions
  runner IP ranges (datacenter traffic detection).

  ForexFactory (nfs.faireconomy.media) is a REST JSON API that remains
  accessible from GitHub Actions runners. fetch_ff_calendar.py already
  fetches and maintains ff_calendar.json with a 21-day rolling window.

  This script reads ff_calendar.json, converts the schema to the format
  expected by the Economic Surprises panel, and merges with the existing
  calendar.json to maintain up to MAX_HISTORY_DAYS of beat/miss history.

DEPENDENCY
  Requires calendar-data/ff_calendar.json to exist and be recent.
  This file is written by fetch_ff_calendar.py, which runs on the same
  4x/daily schedule via update-ff-calendar.yml.
  The update-economic-calendar.yml workflow must run AFTER update-ff-calendar.yml.
  Use `needs: [update-ff-calendar]` or schedule it 30min later.

MERGE STRATEGY
  Fresh data from ff_calendar.json takes priority over stale cached events.
  Historical events (up to MAX_HISTORY_DAYS old) from the previous
  calendar.json are preserved if they have actuals and are not in the
  fresh FF window. This maintains the 90-day rolling beat/miss dataset
  even though FF only provides 2 weeks of live data per run.

CONSUMED BY
  dashboard2.js -> renderEconSurprises() -- Economic Surprises panel

SCHEDULE
  update-economic-calendar.yml -- 4x daily (05:30, 09:30, 13:30, 20:30 UTC)
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta

FF_CALENDAR_PATH = "calendar-data/ff_calendar.json"
OUTPUT_PATH      = "calendar-data/calendar.json"

LOOKBACK_DAYS    = 90   # Economic Surprises rolling window
MAX_HISTORY_DAYS = 365  # Hard cutoff for old events — 12 months for chart history depth

G8_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}

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


def load_ff_calendar():
    """Load and return events from ff_calendar.json."""
    if not os.path.exists(FF_CALENDAR_PATH):
        print(f"  ERROR: {FF_CALENDAR_PATH} not found -- run update-ff-calendar workflow first.")
        return []
    try:
        with open(FF_CALENDAR_PATH, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        generated_at = data.get("generated_at", "unknown")
        print(f"  Loaded ff_calendar.json: {len(events)} events (generated {generated_at})")
        return events
    except Exception as e:
        print(f"  ERROR: Could not read ff_calendar.json -- {e}")
        return []


def ff_to_cal_event(ff):
    """
    Convert a ff_calendar.json event to calendar.json schema.

    FF schema:  title, currency, dateISO, timeUTC, impact, forecast, previous, actual, released
    Cal schema: date, dateISO, timeUTC, country, currency, flag, event, impact, actual, forecast, previous
    """
    currency = (ff.get("currency") or "").strip().upper()
    if currency not in G8_CURRENCIES:
        return None

    impact = (ff.get("impact") or "low").lower()
    if impact not in ("high", "medium"):
        return None  # Economic Surprises only shows medium/high

    date_iso = ff.get("dateISO", "")
    if not date_iso:
        return None

    title = (ff.get("title") or "").strip()
    if not title:
        return None

    try:
        display_date = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%-d %b")
    except (ValueError, AttributeError):
        display_date = date_iso

    def _clean(v):
        if v is None:
            return None
        s = str(v).strip()
        return None if s in ("", "\u2014", "-", "N/A", "--") else s

    return {
        "date":     display_date,
        "dateISO":  date_iso,
        "timeUTC":  ff.get("timeUTC", "00:00"),
        "country":  currency,
        "currency": currency,
        "flag":     FLAG_MAP.get(currency, ""),
        "event":    title,
        "impact":   impact,
        "actual":   _clean(ff.get("actual")),
        "forecast": _clean(ff.get("forecast")),
        "previous": _clean(ff.get("previous")),
    }


def load_previous_calendar():
    if not os.path.exists(OUTPUT_PATH):
        return []
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("events", [])
    except Exception as e:
        print(f"  WARNING: Could not read previous calendar.json -- {e}")
        return []


def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_economic_calendar.py v3.0")

    # Step 1: Load FF calendar
    ff_events = load_ff_calendar()
    if not ff_events:
        print("  WARNING: No FF events available -- preserving previous calendar.json.")
        sys.exit(0)

    # Step 2: Convert FF events to calendar.json schema
    fresh = []
    skipped_low = 0
    skipped_bad = 0
    for ff in ff_events:
        cal = ff_to_cal_event(ff)
        if cal is None:
            if (ff.get("impact") or "low").lower() == "low":
                skipped_low += 1
            else:
                skipped_bad += 1
            continue
        fresh.append(cal)

    print(f"  Converted: {len(fresh)} medium/high-impact G8 events "
          f"(skipped {skipped_low} low-impact, {skipped_bad} non-G8/invalid)")

    # Step 3: Merge historical events from previous calendar.json
    # FF provides only ~2 weeks of live data. Preserve older events with actuals
    # so the 90-day beat/miss window stays populated.
    fresh_keys  = {(e["currency"], e["dateISO"], e["timeUTC"], e["event"]) for e in fresh}
    hard_cutoff = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    prev_events = load_previous_calendar()
    merged = 0
    for ev in prev_events:
        d = ev.get("dateISO", "")
        if d < hard_cutoff:
            continue
        if not (ev.get("actual") or ev.get("forecast")):
            continue
        # Normalise legacy field name (older calendar.json used "title" instead of "event")
        if "title" in ev and "event" not in ev:
            ev["event"] = ev.pop("title")
        k = (ev.get("currency", ""), d, ev.get("timeUTC", ""), ev.get("event", ""))
        if k not in fresh_keys:
            fresh.append(ev)
            fresh_keys.add(k)
            merged += 1
    print(f"  Merged {merged} historical events from previous calendar.json")

    if not fresh:
        print("  WARNING: 0 events after merge -- preserving previous calendar.json.")
        sys.exit(0)

    # Step 4: Sort and stats
    fresh.sort(key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", ""), e.get("currency", "")))

    today           = now_utc.strftime("%Y-%m-%d")
    lookback_cutoff = (now_utc - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    all_dates       = [e["dateISO"] for e in fresh if e.get("dateISO")]
    range_from      = min(all_dates) if all_dates else today
    range_to        = max(all_dates) if all_dates else today

    released_total  = sum(1 for e in fresh if e.get("actual"))
    released_in_win = sum(1 for e in fresh if e.get("actual") and e.get("dateISO", "") >= lookback_cutoff)
    beat_miss_ok    = sum(1 for e in fresh if e.get("actual") and e.get("forecast") and e.get("dateISO", "") >= lookback_cutoff)
    impact_dist     = Counter(e.get("impact", "low") for e in fresh)
    ccy_dist        = Counter(e.get("currency", "") for e in fresh)

    print(f"\n  Total events: {len(fresh)} | range: {range_from} --> {range_to}")
    print(f"  With actuals: {released_total} total | {released_in_win} in {LOOKBACK_DAYS}d | {beat_miss_ok} with beat/miss")
    print(f"  Impact: high={impact_dist['high']} medium={impact_dist['medium']}")
    print(f"  Currencies: {dict(sorted(ccy_dist.items()))}")

    high_today = [e for e in fresh if e["dateISO"] == today and e["impact"] == "high"]
    print(f"\n  Today ({today}) high-impact: {len(high_today)}")
    for e in high_today:
        print(f"    {e['timeUTC']} [{e['currency']}] {e['event'][:45]} | actual={e.get('actual') or 'pending'} forecast={e.get('forecast', '--')}")

    # Step 5: Write output
    output = {
        "lastUpdate":     now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generatedAt":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetchMode":      "ForexFactory via ff_calendar.json",
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         "ForexFactory",
        "errorMessage":   None,
        "fetchErrors":    [],
        "rangeFrom":      range_from,
        "rangeTo":        range_to,
        "totalEvents":    len(fresh),
        "currencyCounts": dict(ccy_dist),
        "impactCounts":   dict(impact_dist),
        "events":         fresh,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate before writing

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)

    print(f"\n  {len(fresh)} events written to {OUTPUT_PATH} "
          f"({beat_miss_ok} with beat/miss in {LOOKBACK_DAYS}d window)")


if __name__ == "__main__":
    main()
