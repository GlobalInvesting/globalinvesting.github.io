#!/usr/bin/env python3
"""
backfill_ff_calendar.py — v1.0
──────────────────────────────────────────────────────────────────────────────
One-shot backfill for calendar-data/ff_calendar.json using the Financial
Modeling Prep (FMP) economic calendar API (free tier, no CC required).

PURPOSE
  ForexFactory's public JSON feed only covers the current week. After the
  migration from Finnhub → ForexFactory (fetch_ff_calendar.py v3.17), the
  rolling ff_calendar.json accumulates weekly but starts with only the current
  week's events. This script does a single 90-day fetch via FMP and merges the
  result into ff_calendar.json, giving the Economic Surprises panel (calendar.json)
  a full 90-day history immediately — without waiting 13 weeks for organic accumulation.

  After this one-shot run, ff_calendar.py takes over as the sole ongoing source.
  FMP is not used anywhere else in the pipeline.

SOURCE
  Financial Modeling Prep stable API — https://financialmodelingprep.com/stable/economic-calendar
  Free tier: 250 req/day, no CC. Register at financialmodelingprep.com → get API key.
  Set FMP_API_KEY env var before running (or pass via workflow secret).
  Query window: up to 90 days per request (no chunking needed).

OUTPUT SCHEMA (ff_calendar.json event fields — identical to fetch_ff_calendar.py output)
  title      str   — event name (from FMP 'event' field)
  currency   str   — ISO-4217 currency code (USD/EUR/GBP/JPY/AUD/CAD/CHF/NZD)
  dateISO    str   — YYYY-MM-DD (UTC date extracted from FMP 'date' datetime)
  timeUTC    str   — HH:MM (UTC time extracted from FMP 'date' datetime)
  impact     str   — 'high' | 'medium' (FMP 'impact' titlecase → lowercase; low filtered)
  forecast   str|None — analyst consensus estimate (FMP 'estimate')
  previous   str|None — prior period value (FMP 'previous')
  actual     str|None — released value (FMP 'actual')
  released   bool  — True when actual is not None/empty

MERGE STRATEGY
  Existing ff_calendar.json events always win for any (title, currency, dateISO, timeUTC)
  key that already exists — this preserves the current week's live FF data.
  FMP backfill only fills gaps (events not yet in ff_calendar.json).
  Dedup pass after merge ensures no duplicates.

IMPACT UPGRADES
  Same _IMPACT_UPGRADES table as fetch_ff_calendar.py — events FMP may under-tag
  as 'Low' are upgraded to 'medium' before the low-impact filter.

FMP FIELD MAPPING
  FMP 'date'      → dateISO + timeUTC  (split on space; 'YYYY-MM-DD HH:MM:SS' format)
  FMP 'event'     → title
  FMP 'currency'  → currency           (already ISO currency code)
  FMP 'actual'    → actual
  FMP 'estimate'  → forecast
  FMP 'previous'  → previous
  FMP 'impact'    → impact             (titlecase: 'High'/'Medium'/'Low')
  (FMP 'country', 'change', 'changePercentage', 'unit' — not used)

USAGE
  python scripts/backfill_ff_calendar.py

REQUIREMENTS
  pip install requests
  FMP_API_KEY env var set

v1.0 (2026-06-10): Initial release — FMP 90-day backfill into ff_calendar.json.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

# ── Config ────────────────────────────────────────────────────────────────────
FMP_API_KEY   = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL  = "https://financialmodelingprep.com/stable/economic-calendar"
OUTPUT_PATH   = "calendar-data/ff_calendar.json"
BACKFILL_DAYS = 90
FETCH_TIMEOUT = 30

G8_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}

# Same impact upgrade table as fetch_ff_calendar.py — keeps impact classification consistent
_IMPACT_UPGRADES = [
    ("JPY", "producer price",          "medium"),  # JPY PPI MoM & YoY
    ("JPY", "machine tool orders",     "medium"),  # JPY Prelim Machine Tool Orders
    ("AUD", "building approvals",      "medium"),  # AUD Building Approvals
    ("AUD", "private house approvals", "medium"),  # AUD Private House Approvals
]


# ── FMP fetch ─────────────────────────────────────────────────────────────────

def fetch_fmp(date_from: str, date_to: str) -> list[dict]:
    """
    Fetch economic calendar from FMP for the given date range.
    Returns the raw JSON array (list of event dicts).
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY env var not set — register at financialmodelingprep.com")

    params = {
        "from":   date_from,
        "to":     date_to,
        "apikey": FMP_API_KEY,
    }
    r = requests.get(FMP_BASE_URL, params=params, timeout=FETCH_TIMEOUT,
                     headers={"User-Agent": "globalinvesting-backfill/1.0"})
    if r.status_code != 200:
        raise RuntimeError(f"FMP HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"FMP returned unexpected type: {type(data).__name__}")
    return data


# ── Normalise FMP event → ff_calendar schema ──────────────────────────────────

def _clean(v) -> str | None:
    """Strip and return None for blank/dash sentinel values."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "—", "-", "N/A", "--", "null") else s


def _parse_fmp_datetime(date_str: str) -> tuple[str, str]:
    """
    Parse FMP date string ('YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS') into
    (dateISO, timeUTC) tuple.  Falls back to ('', '00:00') on parse error.
    """
    date_str = date_str.strip().replace("T", " ")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except ValueError:
        # Try date-only
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d"), "00:00"
        except ValueError:
            return "", "00:00"


def normalise_fmp_event(raw: dict) -> dict | None:
    """
    Convert a raw FMP event dict to ff_calendar.json event schema.
    Returns None if the event should be filtered (non-G8, low-impact).
    """
    currency = _clean(raw.get("currency") or raw.get("country"))
    if not currency:
        return None
    currency = currency.upper()
    if currency not in G8_CURRENCIES:
        return None

    # Impact — FMP uses titlecase: 'High', 'Medium', 'Low'
    impact_raw = (_clean(raw.get("impact")) or "low").lower()
    if impact_raw not in ("high", "medium", "low"):
        impact_raw = "low"

    title = _clean(raw.get("event") or "")
    if not title:
        return None

    # Apply impact upgrades (same logic as fetch_ff_calendar.py)
    if impact_raw == "low":
        title_lower = title.lower()
        for upg_ccy, upg_frag, upg_impact in _IMPACT_UPGRADES:
            if currency == upg_ccy and upg_frag in title_lower:
                impact_raw = upg_impact
                break

    if impact_raw == "low":
        return None  # filter out low-impact events

    date_str = _clean(raw.get("date") or "")
    if not date_str:
        return None
    date_iso, time_utc = _parse_fmp_datetime(date_str)
    if not date_iso:
        return None

    actual   = _clean(raw.get("actual"))
    forecast = _clean(raw.get("estimate"))
    previous = _clean(raw.get("previous"))
    released = actual is not None

    return {
        "title":    title,
        "currency": currency,
        "dateISO":  date_iso,
        "timeUTC":  time_utc,
        "impact":   impact_raw,
        "forecast": forecast,
        "previous": previous,
        "actual":   actual,
        "released": released,
    }


# ── Load / save ff_calendar.json ──────────────────────────────────────────────

def load_existing() -> tuple[list[dict], list[dict]]:
    """
    Load existing ff_calendar.json.
    Returns (events, holidays).  Both are empty lists if file doesn't exist.
    """
    if not os.path.exists(OUTPUT_PATH):
        return [], []
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("events", []), data.get("holidays", [])
    except Exception as e:
        print(f"  WARNING: could not read {OUTPUT_PATH} — {e}")
        return [], []


def save(events: list[dict], holidays: list[dict]) -> None:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    out = {
        "generated_at": now_utc,
        "source":       "ForexFactory",
        "holidays":     holidays,
        "events":       events,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  ✓ Wrote {len(events)} events + {len(holidays)} holidays → {OUTPUT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_utc   = datetime.now(timezone.utc)
    date_to   = now_utc.strftime("%Y-%m-%d")
    date_from = (now_utc - timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%d")

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M UTC')}] backfill_ff_calendar.py v1.0")
    print(f"  FMP: fetching {date_from} → {date_to} ({BACKFILL_DAYS} days) ...")

    # ── Step 1: Fetch from FMP ────────────────────────────────────────────────
    try:
        raw = fetch_fmp(date_from, date_to)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  FMP: {len(raw)} raw events received")

    # ── Step 2: Normalise ─────────────────────────────────────────────────────
    fmp_events: list[dict] = []
    skipped_ccy    = 0
    skipped_impact = 0

    for ev in raw:
        norm = normalise_fmp_event(ev)
        if norm is None:
            currency = (ev.get("currency") or ev.get("country") or "").upper()
            if currency not in G8_CURRENCIES:
                skipped_ccy += 1
            else:
                skipped_impact += 1
        else:
            fmp_events.append(norm)

    with_actuals = sum(1 for e in fmp_events if e.get("released"))
    print(f"  FMP: {len(fmp_events)} G8 medium/high events "
          f"(skipped {skipped_ccy} non-G8, {skipped_impact} low-impact)")
    print(f"  FMP: {with_actuals} events with actuals")

    # ── Step 3: Load existing ff_calendar.json ────────────────────────────────
    existing_events, existing_holidays = load_existing()
    print(f"  Existing ff_calendar.json: {len(existing_events)} events, "
          f"{len(existing_holidays)} holidays")

    # ── Step 4: Merge — existing events always win ────────────────────────────
    # Key: (title, currency, dateISO, timeUTC)
    existing_keys: set[tuple] = {
        (e["title"], e["currency"], e["dateISO"], e["timeUTC"])
        for e in existing_events
    }

    injected = 0
    for ev in fmp_events:
        k = (ev["title"], ev["currency"], ev["dateISO"], ev["timeUTC"])
        if k not in existing_keys:
            existing_events.append(ev)
            existing_keys.add(k)
            injected += 1

    print(f"  Injected: {injected} new events from FMP backfill")
    print(f"  Skipped:  {len(fmp_events) - injected} FMP events already in ff_calendar.json")

    # ── Step 5: Sort by dateISO + timeUTC ────────────────────────────────────
    existing_events.sort(key=lambda e: (e.get("dateISO", ""), e.get("timeUTC", "")))

    # ── Step 6: Trim to BACKFILL_DAYS lookback ────────────────────────────────
    cutoff = (now_utc - timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%d")
    before_trim = len(existing_events)
    existing_events = [e for e in existing_events if e.get("dateISO", "") >= cutoff]
    trimmed = before_trim - len(existing_events)
    if trimmed:
        print(f"  Trimmed: {trimmed} events older than {BACKFILL_DAYS} days")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    total_actuals   = sum(1 for e in existing_events if e.get("released"))
    by_ccy: dict[str, int] = {}
    for e in existing_events:
        c = e.get("currency", "?")
        by_ccy[c] = by_ccy.get(c, 0) + 1

    print()
    print("  =========================================")
    print("    FF CALENDAR BACKFILL SUMMARY v1.0")
    print("  =========================================")
    print(f"  Total events:    {len(existing_events)}")
    print(f"  With actuals:    {total_actuals}")
    print(f"  Injected today:  {injected}")
    print(f"  Date range:      {date_from} → {date_to}")
    print(f"  By currency:     {dict(sorted(by_ccy.items()))}")
    print("  =========================================")

    # ── Step 8: Write ─────────────────────────────────────────────────────────
    save(existing_events, existing_holidays)
    print()
    print("  Run update-ff-calendar.yml (workflow_dispatch) after this to refresh "
          "the current week's actuals from ForexFactory.")


if __name__ == "__main__":
    main()
