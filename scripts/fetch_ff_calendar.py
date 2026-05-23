#!/usr/bin/env python3
"""
fetch_ff_calendar.py — v2.0
Scrapes the ForexFactory economic calendar HTML page using Playwright (headless Chromium)
and writes a normalised snapshot to calendar-data/ff_calendar.json.

WHY PLAYWRIGHT (v2.0 change from v1.x)
  v1.x fetched nfs.faireconomy.media JSON API (thisweek + nextweek).
  That API intentionally omits actual values — it is documented as a planning feed,
  not a real-time data source. Actuals never appeared in the JSON even hours after
  events released, rendering the "released" and "actual" fields permanently null.

  The ForexFactory HTML page (forexfactory.com/calendar) shows actuals in real-time
  and is the canonical FF data source used by traders. Playwright renders the full
  JS-driven page and extracts the same data visible to a human visitor.

SOURCE
  https://www.forexfactory.com/calendar

FF HTML SCHEMA (extracted from DOM)
  table#calentable > tbody > tr.calendar__row
  Columns: date, time, currency, impact (dot color), event, actual, forecast, previous

OUTPUT SCHEMA (ff_calendar.json)  — unchanged from v1.x
  generated_at  — ISO UTC timestamp of this run
  source        — "ForexFactory"
  events[]      — array of normalised events:
    title       — event name (stripped)
    currency    — G8 currency code
    dateISO     — YYYY-MM-DD (UTC)
    timeUTC     — HH:MM (UTC)
    impact      — "high" | "medium" | "low"
    forecast    — string or null
    previous    — string or null
    actual      — string or null  ← NOW POPULATED IN REAL TIME
    released    — bool  (True when actual is present)

HISTORICAL MERGE
  FF HTML shows this-week + next-week events. Past events from the previous
  ff_calendar.json are preserved (21-day rolling window) so actuals are not
  lost between runs. Released events (actual != null) are always kept.

CONSUMED BY
  calendar-panel.js  → Economic Calendar panel (direct read)
  fetch_economic_calendar.py → calendar.json → Economic Surprises panel

SCHEDULE (update-ff-calendar.yml)
  4× daily: 00:30, 06:30, 12:30, 20:30 UTC
"""

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

OUTPUT_PATH   = "calendar-data/ff_calendar.json"
FF_URL        = "https://www.forexfactory.com/calendar"
LOOKBACK_DAYS = 21
FETCH_TIMEOUT = 45_000   # ms — FF page is JS-heavy

G8_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}

# FF renders times in ET (Eastern Time). We convert to UTC.
# The offset is embedded in the page meta or can be derived from the HTML.
# FF uses ET. We detect DST via known offset ranges.
# Simpler approach: parse the time + date, assume ET, convert.
IMPACT_MAP = {
    "red":    "high",
    "orange": "medium",
    "yellow": "low",
    "gray":   "low",
    "grey":   "low",
}

CUR_TO_FLAG_TZ = {
    "USD": "America/New_York",
    "EUR": "Europe/Frankfurt",
    "GBP": "Europe/London",
    "JPY": "Asia/Tokyo",
    "AUD": "Australia/Sydney",
    "CAD": "America/Toronto",
    "CHF": "Europe/Zurich",
    "NZD": "Pacific/Auckland",
}


def fetch_ff_html() -> str | None:
    """Render forexfactory.com/calendar with headless Chromium. Returns page HTML."""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",   # FF displays times in ET
            )
            page = context.new_page()
            print(f"  Loading {FF_URL} ...")
            page.goto(FF_URL, wait_until="domcontentloaded", timeout=FETCH_TIMEOUT)
            # Wait for the calendar table to appear
            try:
                page.wait_for_selector("table.calendar__table", timeout=20_000)
                print("  Calendar table found.")
            except PlaywrightTimeout:
                print("  WARNING: Timed out waiting for calendar__table — using available HTML.")
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"  ERROR: Playwright error — {e}")
        return None


def parse_et_to_utc(date_iso: str, time_str: str) -> tuple[str, str]:
    """
    Convert a date (YYYY-MM-DD) + time string (e.g. '8:30am') from ET to UTC.
    Returns (date_iso_utc, time_utc_hhmm).
    FF times are in US Eastern Time (ET). We use a fixed UTC offset based on
    standard DST rules: EDT (UTC-4) Mar 2nd Sun–Nov 1st Sun, EST (UTC-5) otherwise.
    """
    time_str = time_str.strip().lower()
    if not time_str or time_str in ("all day", "tentative", ""):
        return date_iso, "00:00"

    try:
        # Parse "8:30am", "12:00pm", "4:15pm"
        match = re.match(r"(\d{1,2}):(\d{2})(am|pm)", time_str)
        if not match:
            return date_iso, "00:00"
        h, m, ampm = int(match.group(1)), int(match.group(2)), match.group(3)
        if ampm == "pm" and h != 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0

        year, mon, day = int(date_iso[:4]), int(date_iso[5:7]), int(date_iso[8:10])
        # Determine ET UTC offset: EDT=-4 or EST=-5
        # DST starts: 2nd Sunday of March at 2am ET
        # DST ends:   1st Sunday of November at 2am ET
        # Simple approximation: UTC-4 from March 8 to Nov 7 (covers all edge cases safely)
        def is_edt(y, mo, da):
            if mo < 3 or mo > 11: return False
            if mo > 3 and mo < 11: return True
            if mo == 3: return da >= 8
            if mo == 11: return da < 7
            return False

        utc_offset = -4 if is_edt(year, mon, day) else -5
        dt_et = datetime(year, mon, day, h, m, tzinfo=timezone(timedelta(hours=utc_offset)))
        dt_utc = dt_et.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%d"), dt_utc.strftime("%H:%M")
    except Exception:
        return date_iso, "00:00"


def clean_value(v: str | None) -> str | None:
    """Strip and normalise a scraped value cell. Returns None if empty/dash."""
    if v is None:
        return None
    s = v.strip()
    if s in ("", "—", "-", "\u2014", "N/A", "n/a"):
        return None
    return s


def parse_ff_html(html: str) -> list[dict]:
    """Parse FF calendar HTML and return list of normalised event dicts."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"calendar__table"))
    if not table:
        print("  ERROR: calendar__table not found in HTML.")
        return []

    rows = table.find_all("tr", class_=re.compile(r"calendar__row"))
    print(f"  Found {len(rows)} calendar rows.")

    events = []
    current_date_iso = None
    now_utc = datetime.now(timezone.utc)

    for row in rows:
        # Skip header/spacer rows
        if "calendar__row--day-breaker" in row.get("class", []):
            # Date separator — extract date from this row
            date_cell = row.find("td", class_=re.compile(r"calendar__date"))
            if date_cell:
                date_text = date_cell.get_text(strip=True)
                # FF format: "Thu May 22" — parse relative to current year
                try:
                    # Add current year — FF only shows month/day in the separator
                    parsed = datetime.strptime(f"{date_text} {now_utc.year}", "%a %b %d %Y")
                    # Handle year boundary (Dec→Jan wrap)
                    if abs((parsed - now_utc.replace(tzinfo=None)).days) > 180:
                        parsed = parsed.replace(year=now_utc.year + 1)
                    current_date_iso = parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            continue

        # Skip non-event rows
        if not row.find("td", class_=re.compile(r"calendar__event")):
            continue

        # Currency
        ccy_cell = row.find("td", class_=re.compile(r"calendar__currency"))
        currency = ccy_cell.get_text(strip=True).upper() if ccy_cell else ""
        if currency not in G8_CURRENCIES:
            continue

        # Time
        time_cell = row.find("td", class_=re.compile(r"calendar__time"))
        time_et = time_cell.get_text(strip=True) if time_cell else ""

        # Impact
        impact_cell = row.find("td", class_=re.compile(r"calendar__impact"))
        impact = "low"
        if impact_cell:
            span = impact_cell.find("span")
            if span:
                cls_str = " ".join(span.get("class", []))
                for color, level in IMPACT_MAP.items():
                    if color in cls_str:
                        impact = level
                        break

        # Event title
        event_cell = row.find("td", class_=re.compile(r"calendar__event"))
        title = ""
        if event_cell:
            # FF wraps title in a span inside an anchor
            span = event_cell.find("span", class_=re.compile(r"calendar__event-title"))
            if span:
                title = span.get_text(strip=True)
            else:
                title = event_cell.get_text(strip=True)
        title = title.strip()
        if not title:
            continue

        # Actual, Forecast, Previous
        actual_cell   = row.find("td", class_=re.compile(r"calendar__actual"))
        forecast_cell = row.find("td", class_=re.compile(r"calendar__forecast"))
        previous_cell = row.find("td", class_=re.compile(r"calendar__previous"))

        actual   = clean_value(actual_cell.get_text()   if actual_cell   else None)
        forecast = clean_value(forecast_cell.get_text() if forecast_cell else None)
        previous = clean_value(previous_cell.get_text() if previous_cell else None)

        # Convert ET time to UTC
        if current_date_iso:
            date_iso_utc, time_utc = parse_et_to_utc(current_date_iso, time_et)
        else:
            date_iso_utc = now_utc.strftime("%Y-%m-%d")
            time_utc = "00:00"

        events.append({
            "title":    title,
            "currency": currency,
            "dateISO":  date_iso_utc,
            "timeUTC":  time_utc,
            "impact":   impact,
            "forecast": forecast,
            "previous": previous,
            "actual":   actual,
            "released": actual is not None,
        })

    return events


def load_previous() -> list[dict]:
    """Load events from existing ff_calendar.json for historical merge."""
    if not os.path.exists(OUTPUT_PATH):
        return []
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        released = sum(1 for e in events if e.get("released"))
        print(f"  Previous file: {len(events)} events ({released} released)")
        return events
    except Exception as e:
        print(f"  WARNING: Could not read previous ff_calendar.json — {e}")
        return []


def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_ff_calendar.py v2.0")

    # Step 1: Fetch FF HTML
    html = fetch_ff_html()
    if not html:
        print("  ERROR: Could not fetch FF page — preserving previous file.")
        sys.exit(0)

    # Step 2: Parse events
    fresh = parse_ff_html(html)
    if not fresh:
        print("  ERROR: No events parsed — preserving previous file.")
        sys.exit(0)

    released_fresh = sum(1 for e in fresh if e.get("released"))
    print(f"  Parsed: {len(fresh)} events ({released_fresh} with actuals)")

    # Step 3: Historical merge — preserve past events with actuals
    cutoff = (now_utc - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    fresh_keys = {(e["currency"], e["dateISO"], e["timeUTC"], e["title"]) for e in fresh}
    prev_events = load_previous()
    merged = 0
    for ev in prev_events:
        d = ev.get("dateISO", "")
        if d < cutoff:
            continue
        k = (ev.get("currency", ""), d, ev.get("timeUTC", ""), ev.get("title", ""))
        if k not in fresh_keys:
            fresh.append(ev)
            fresh_keys.add(k)
            merged += 1
    print(f"  Merged: {merged} historical events from previous file (cutoff {cutoff})")

    # Step 4: Sort and dedupe
    fresh.sort(key=lambda e: (e["dateISO"], e["timeUTC"], e["currency"]))

    # Step 5: Stats
    from collections import Counter
    today = now_utc.strftime("%Y-%m-%d")
    today_evs = [e for e in fresh if e["dateISO"] == today]
    high_today = [e for e in today_evs if e["impact"] == "high"]
    released_today = [e for e in today_evs if e.get("released")]
    impact_dist = Counter(e["impact"] for e in fresh)
    ccy_dist    = Counter(e["currency"] for e in fresh)

    print(f"\n  Total: {len(fresh)} events | {sum(1 for e in fresh if e.get('released'))} with actuals")
    print(f"  Impact: high={impact_dist['high']} medium={impact_dist['medium']} low={impact_dist['low']}")
    print(f"  Currencies: {dict(sorted(ccy_dist.items()))}")
    print(f"\n  Today ({today}): {len(today_evs)} events | {len(released_today)} released")
    for e in high_today:
        print(f"    {e['timeUTC']} [{e['currency']}] {e['title'][:45]} | actual={e.get('actual') or 'pending'} forecast={e.get('forecast','—')}")

    # Step 6: Write output
    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source":       "ForexFactory",
        "events":       fresh,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)

    released_total = sum(1 for e in fresh if e.get("released"))
    print(f"\n✓ {len(fresh)} events written to {OUTPUT_PATH} ({released_total} with actuals)")


if __name__ == "__main__":
    main()
