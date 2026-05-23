#!/usr/bin/env python3
"""
fetch_economic_calendar.py  v3.2
──────────────────────────────────────────────────────────────────────────────
Builds calendar-data/calendar.json (Economic Surprises panel source) by
converting and merging data from calendar-data/ff_calendar.json, which is
maintained by fetch_ff_calendar.py / update-ff-calendar.yml.

WHY THIS APPROACH
  v2.0 scraped investing.com via a hidden POST endpoint, which worked until
  May 2026 when investing.com began returning HTTP 503 to GitHub Actions
  runner IP ranges (datacenter traffic detection).

  ForexFactory/Finnhub (via ff_calendar.json) is the replacement. This script
  reads ff_calendar.json, converts the schema to the format expected by the
  Economic Surprises panel, and merges with the existing calendar.json to
  maintain up to MAX_HISTORY_DAYS of beat/miss history.

DEPENDENCY
  Requires calendar-data/ff_calendar.json to exist and be recent.
  Written by fetch_ff_calendar.py via update-ff-calendar.yml.
  This script runs automatically via workflow_run trigger after
  update-ff-calendar.yml completes (inherits CF Worker latency ~2-3 min).

MERGE STRATEGY
  Fresh data from ff_calendar.json always takes priority.
  Merge key: (currency, dateISO, event) — excludes timeUTC intentionally.
  Finnhub occasionally adjusts scheduled times between runs; including timeUTC
  caused the same event to appear twice (fresh with forecast + stale without).
  A purge pass also removes any lingering stale duplicates from the historical
  section that were created before the v3.1 merge key fix.
  Historical events (up to MAX_HISTORY_DAYS) without a fresh counterpart are
  preserved only if they have actuals — this maintains the 90-day beat/miss dataset.

SURPRISE STATS
  Computes per-event-series statistics (n, mean, std of actual-vs-forecast surprise)
  over the LOOKBACK_DAYS window. Written to calendar.json as `surpriseStats` field.
  Consumed by dashboard.js for z-score normalisation (graduated from beat/miss
  to z-score when n >= 5 for a given series).

CONSUMED BY
  dashboard.js        -> renderEconSurprises() -- Economic Surprises panel (inline)
  econ-surprises-modal.js -> full modal with chart + table

SCHEDULE
  update-economic-calendar.yml -- workflow_run after update-ff-calendar.yml
  (inherits CF Worker latency: ~2-3 min end-to-end). Falls back to 4x daily cron.

CHANGELOG
  v3.2 (2026-05-23): Propagate source from ff_calendar.json (was hardcoded
    'ForexFactory'); add purge pass for stale pre-v3.1 duplicate entries;
    generate surpriseStats for z-score scoring.
  v3.1 (2026-05-23): Fix merge key — exclude timeUTC.
  v3.0 (2026-05-23): Initial version replacing investing.com scraper.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
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

# Indicators where lower-than-forecast = positive surprise
INVERSE_EVENTS = frozenset({
    "unemployment", "jobless", "claims", "deficit", "trade balance",
})


def load_ff_calendar():
    """Load events and source from ff_calendar.json."""
    if not os.path.exists(FF_CALENDAR_PATH):
        print(f"  ERROR: {FF_CALENDAR_PATH} not found -- run update-ff-calendar workflow first.")
        return [], "unknown"
    try:
        with open(FF_CALENDAR_PATH, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        generated_at = data.get("generated_at", "unknown")
        source = data.get("source", "Finnhub")
        print(f"  Loaded ff_calendar.json: {len(events)} events (generated {generated_at}, source: {source})")
        return events, source
    except Exception as e:
        print(f"  ERROR: Could not read ff_calendar.json -- {e}")
        return [], "unknown"


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


def compute_surprise_stats(events, lookback_cutoff):
    """
    Compute per-event-series surprise statistics for z-score normalisation.

    For each (currency, canonical_event_name) series, computes:
      n    — number of observations in the lookback window
      mean — mean surprise (actual - forecast, sign-corrected for inverse events)
      std  — sample standard deviation of surprises

    Only includes events with both actual and forecast (numeric) in the window.
    Returns dict keyed by "CCY/EventName" matching the format expected by dashboard.js.

    Industry standard: Bloomberg uses 1-year rolling z-score per series.
    We use LOOKBACK_DAYS (90d) as our window — sufficient for most monthly series
    (gives ~6 data points for monthly, ~12 for weekly like Claims).
    """
    series = defaultdict(list)

    for ev in events:
        if not ev.get("actual") or not ev.get("forecast"):
            continue
        if ev.get("dateISO", "") < lookback_cutoff:
            continue

        ccy = ev.get("currency", "")
        title = ev.get("event", "")
        if not ccy or not title:
            continue

        # Canonical name: strip trailing parentheticals like "(MoM)" for grouping
        # Keep the full name for display; canon is for stats key only
        canon = title.strip()

        try:
            actual_f   = float(str(ev["actual"]).replace("%", "").replace(",", "").strip())
            forecast_f = float(str(ev["forecast"]).replace("%", "").replace(",", "").strip())
        except (ValueError, TypeError):
            continue

        raw_surprise = actual_f - forecast_f
        # Sign-correct: for inverse indicators, lower actual = positive surprise
        title_lower = title.lower()
        is_inverse  = any(kw in title_lower for kw in INVERSE_EVENTS)
        surprise    = -raw_surprise if is_inverse else raw_surprise

        series[f"{ccy}/{canon}"].append(surprise)

    stats = {}
    for key, surprises in series.items():
        n = len(surprises)
        if n < 2:
            continue  # need at least 2 for a meaningful std
        mean = sum(surprises) / n
        variance = sum((x - mean) ** 2 for x in surprises) / (n - 1)  # sample variance
        std = math.sqrt(variance)
        stats[key] = {"n": n, "mean": round(mean, 6), "std": round(std, 6)}

    return stats


def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_economic_calendar.py v3.2")

    # Step 1: Load FF calendar (source propagated, not hardcoded)
    ff_events, ff_source = load_ff_calendar()
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
    # Merge key: (currency, dateISO, event) — intentionally excludes timeUTC.
    # Fresh ff_calendar data always wins; stale historical versions are skipped.
    fresh_keys  = {(e["currency"], e["dateISO"], e["event"]) for e in fresh}
    hard_cutoff = (now_utc - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y-%m-%d")
    prev_events = load_previous_calendar()
    merged = 0
    purged = 0
    for ev in prev_events:
        d = ev.get("dateISO", "")
        if d < hard_cutoff:
            continue
        if not ev.get("actual"):
            # Events without actuals add no value to the beat/miss history
            continue
        # Normalise legacy field name (older calendar.json used "title" instead of "event")
        if "title" in ev and "event" not in ev:
            ev["event"] = ev.pop("title")
        k = (ev.get("currency", ""), d, ev.get("event", ""))
        if k in fresh_keys:
            # Fresh version already covers this event — skip the stale historical copy.
            # This also purges pre-v3.1 duplicates that slipped in before the merge
            # key fix (where the same event had two entries with different timeUTC).
            purged += 1
            continue
        fresh.append(ev)
        fresh_keys.add(k)
        merged += 1

    if purged:
        print(f"  Purged {purged} stale historical duplicates (superseded by fresh data)")
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

    # Step 5: Compute surpriseStats for z-score scoring in dashboard.js
    surprise_stats = compute_surprise_stats(fresh, lookback_cutoff)
    series_with_stats = sum(1 for v in surprise_stats.values() if v["n"] >= 5)
    print(f"\n  surpriseStats: {len(surprise_stats)} series computed ({series_with_stats} with n>=5 for z-score)")

    # Step 6: Write output
    # Source propagated from ff_calendar.json ("Finnhub" or "ForexFactory")
    fetch_mode = f"{ff_source} via ff_calendar.json"
    output = {
        "lastUpdate":     now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generatedAt":    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetchMode":      fetch_mode,
        "timezoneNote":   "All times UTC",
        "status":         "ok",
        "source":         ff_source,
        "errorMessage":   None,
        "fetchErrors":    [],
        "rangeFrom":      range_from,
        "rangeTo":        range_to,
        "totalEvents":    len(fresh),
        "currencyCounts": dict(ccy_dist),
        "impactCounts":   dict(impact_dist),
        "surpriseStats":  surprise_stats,
        "events":         fresh,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate before writing

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)

    print(f"\n  {len(fresh)} events written to {OUTPUT_PATH} "
          f"({beat_miss_ok} with beat/miss in {LOOKBACK_DAYS}d window) — source: {ff_source}")


if __name__ == "__main__":
    main()
