#!/usr/bin/env python3
"""
fetch_ff_calendar.py — v1.2
Fetches the ForexFactory economic calendar (last week + this week + next week) and writes
a normalised snapshot to calendar-data/ff_calendar.json.

SOURCE
  ForexFactory public JSON API (no auth required):
    https://nfs.faireconomy.media/ff_calendar_lastweek.json  (actuals for last 7 days)
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
    https://nfs.faireconomy.media/ff_calendar_nextweek.json

FF SCHEMA (per event)
  title    — event name  e.g. "Non-Farm Payrolls"
  country  — currency    e.g. "USD"  (already G8-aligned)
  date     — ISO 8601 in US Eastern Time  e.g. "2026-04-03T08:30:00-0400"
  impact   — "High" | "Medium" | "Low" | "Holiday"
  forecast — string or null  e.g. "48K", "0.4%"
  previous — string or null
  actual   — string or null  (null before release, populated after)

OUTPUT SCHEMA (ff_calendar.json)
  generated_at  — ISO UTC timestamp of this run
  source        — "ForexFactory"
  events[]      — array of normalised events:
    title       — event name (stripped)
    currency    — G8 currency code (USD/EUR/GBP/JPY/AUD/CAD/CHF/NZD) or null
    dateISO     — YYYY-MM-DD (UTC date)
    timeUTC     — HH:MM (UTC)
    impact      — "high" | "medium" | "low"  (lower-cased, "Holiday" excluded)
    forecast    — string or null
    previous    — string or null
    actual      — string or null
    released    — bool  (True when actual is present)

DESIGN DECISIONS
  · "Holiday" events are excluded — no market-moving content.
  · Only G8 currencies are retained (USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD).
    FF also covers CNY, MXN, etc. — not consumed by the FX narrative pipeline.
  · Dates are normalised to UTC. FF publishes times in US Eastern
    (America/New_York) — the converter accounts for both EST (-0500) and
    EDT (-0400) via the offset embedded in the ISO string.
  · "All Day" events (no specific time) are assigned timeUTC = "00:00" and
    treated as low-impact for session bucketing purposes.
  · The script is idempotent: re-running produces an identical file when FF
    data has not changed. Change detection in generate_narrative_signals.py
    uses the event list (not generated_at) to decide whether to regenerate.

CONSUMED BY
  generate_narrative_signals.py → build_calendar_block()
  Injected into SESSION CONTEXT user_prompt so Groq can cite specific
  catalysts (CB speakers, data releases) per session instead of hallucinating.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

import requests

FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_lastweek.json",   # includes actuals for last week
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

G8_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
OUTPUT_PATH   = "calendar-data/ff_calendar.json"
FETCH_TIMEOUT = 15

IMPACT_MAP = {
    "High":   "high",
    "Medium": "medium",
    "Low":    "low",
}


def fetch_ff(url: str) -> list:
    """Fetch a ForexFactory calendar JSON endpoint. Returns list of raw events."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FXTerminal/1.0)"},
            timeout=FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"  WARNING: {url} returned HTTP {resp.status_code} — skipping.")
            return []
        data = resp.json()
        if not isinstance(data, list):
            print(f"  WARNING: {url} returned unexpected type {type(data).__name__} — skipping.")
            return []
        return data
    except requests.RequestException as e:
        print(f"  WARNING: fetch failed for {url} — {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON decode error for {url} — {e}")
        return []


def normalise_event(raw: dict) -> dict | None:
    """Normalise a single FF event to the internal schema. Returns None to exclude."""
    impact_raw = raw.get("impact", "")
    if impact_raw == "Holiday":
        return None  # exclude public holidays — no market content

    currency = (raw.get("country") or "").strip().upper()
    if currency not in G8_CURRENCIES:
        return None  # only G8 — exclude CNY, MXN, etc.

    impact = IMPACT_MAP.get(impact_raw, "low")
    title  = (raw.get("title") or "").strip()
    if not title:
        return None

    # Parse date — FF uses ISO 8601 with Eastern Time offset embedded
    date_str = raw.get("date") or ""
    if date_str:
        try:
            dt_local = dateparser.parse(date_str)
            if dt_local.tzinfo is None:
                # Assume Eastern — FF convention when no offset
                from datetime import timezone as tz
                et_offset = timedelta(hours=-5)  # conservative EST
                dt_local = dt_local.replace(tzinfo=timezone(et_offset))
            dt_utc = dt_local.astimezone(timezone.utc)
            date_iso = dt_utc.strftime("%Y-%m-%d")
            time_utc = dt_utc.strftime("%H:%M")
        except Exception:
            return None  # unparseable date — skip
    else:
        return None  # no date — skip

    actual   = (raw.get("actual")   or "").strip() or None
    forecast = (raw.get("forecast") or "").strip() or None
    previous = (raw.get("previous") or "").strip() or None

    return {
        "title":    title,
        "currency": currency,
        "dateISO":  date_iso,
        "timeUTC":  time_utc,
        "impact":   impact,
        "forecast": forecast,
        "previous": previous,
        "actual":   actual,
        "released": actual is not None,
    }


def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_ff_calendar.py v1.1")

    raw_events: list[dict] = []
    for url in FF_URLS:
        batch = fetch_ff(url)
        print(f"  {url.split('/')[-1]}: {len(batch)} raw events")
        raw_events.extend(batch)

    if not raw_events:
        print("  ERROR: No events fetched from ForexFactory — aborting.")
        # Preserve previous file if it exists
        if os.path.exists(OUTPUT_PATH):
            print("  Preserving previous ff_calendar.json.")
        sys.exit(1)

    # ── Merge historical events from previous file ────────────────────────
    # FF only provides this-week + next-week data. Past events are lost each run.
    # We merge ALL recent events (released or not) from the previous file so:
    #   1. Released events with actuals are preserved for the 21-day lookback.
    #   2. Unreleased events that were in the previous file can be updated if FF
    #      delivers their actual in a later run before they exit the weekly window.
    # NOTE: calendar-data/calendar.json (TradingEconomics) is the primary source
    # for Economic Surprises. ff_calendar.json is used by the AI narrative pipeline.
    LOOKBACK_DAYS = 21
    cutoff = (now_utc - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    hist_events: list[dict] = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as _f:
                _prev = json.load(_f)
            for ev in _prev.get("events", []):
                if ev.get("dateISO", "") >= cutoff:
                    hist_events.append(ev)
            released_hist = sum(1 for e in hist_events if e.get("released"))
            print(f"  Historical merge: {len(hist_events)} events from previous file ({released_hist} released, cutoff {cutoff})")
        except Exception as _e:
            print(f"  Historical merge: skipped ({_e})")

    # Normalise and filter
    events: list[dict] = []
    excluded_holiday = 0
    excluded_non_g8  = 0
    excluded_other   = 0

    seen: set[tuple] = set()
    for raw in raw_events:
        norm = normalise_event(raw)
        if norm is None:
            impact_raw = raw.get("impact", "")
            if impact_raw == "Holiday":
                excluded_holiday += 1
            elif (raw.get("country") or "").upper() not in G8_CURRENCIES:
                excluded_non_g8 += 1
            else:
                excluded_other += 1
            continue

        # Deduplicate by (currency, dateISO, timeUTC, title)
        key = (norm["currency"], norm["dateISO"], norm["timeUTC"], norm["title"])
        if key in seen:
            continue
        seen.add(key)
        events.append(norm)

    # Merge historical released events (not in FF current window)
    if hist_events:
        # Build key set from freshly fetched events
        fresh_keys: set[tuple] = {(e["currency"], e["dateISO"], e["timeUTC"], e["title"]) for e in events}
        added = 0
        for ev in hist_events:
            k = (ev["currency"], ev["dateISO"], ev["timeUTC"], ev["title"])
            if k not in fresh_keys:
                events.append(ev)
                fresh_keys.add(k)
                added += 1
        print(f"  Historical merge: {added} past events appended (total now {len(events)})")

    # Sort chronologically
    events.sort(key=lambda e: (e["dateISO"], e["timeUTC"], e["currency"]))

    # Stats
    today = now_utc.strftime("%Y-%m-%d")
    today_events  = [e for e in events if e["dateISO"] == today]
    high_today    = [e for e in today_events if e["impact"] == "high"]
    released_today = [e for e in high_today if e["released"]]
    pending_today  = [e for e in high_today if not e["released"]]

    from collections import Counter
    impact_dist = Counter(e["impact"] for e in events)
    ccy_dist    = Counter(e["currency"] for e in events)

    print(f"\n  Normalised: {len(events)} events retained")
    print(f"  Excluded: {excluded_holiday} holidays, {excluded_non_g8} non-G8, {excluded_other} other")
    print(f"  Impact: high={impact_dist['high']} | medium={impact_dist['medium']} | low={impact_dist['low']}")
    print(f"  Currencies: {dict(sorted(ccy_dist.items()))}")
    print(f"\n  Today ({today}) high-impact: {len(high_today)} ({len(released_today)} released, {len(pending_today)} pending)")
    for e in high_today:
        status = e["actual"] or "—"
        print(f"    {e['timeUTC']} [{e['currency']}] {e['title']} | actual={status} forecast={e.get('forecast','—')}")

    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source":       "ForexFactory",
        "events":       events,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Validate JSON before writing (prevents partial-write corruption)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # raises ValueError if invalid
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)

    released_total = sum(1 for e in events if e.get("released"))
    print(f"\n✓ {len(events)} events written to {OUTPUT_PATH} ({released_total} released with actuals)")


if __name__ == "__main__":
    main()
