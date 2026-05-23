#!/usr/bin/env python3
"""
fetch_ff_calendar.py — v3.0
Fetches the G8 economic calendar with real-time actuals from Finnhub
and writes calendar-data/ff_calendar.json to the public site repo.

WHY FINNHUB (v3.0 change from v2.0)
  v2.0 used Playwright to scrape ForexFactory HTML. FF HTML parsing proved
  fragile: date separators were mis-assigned when FF regrouped past events
  under the current week, producing incorrect dateISO/timeUTC for released
  events (e.g. Claimant Count Change appeared as 2026-05-23 00:00 instead
  of 2026-05-22 06:00). Actuals from today's events were also missed because
  FF's JS-rendered page uses a structure that changes layout once events
  move into the "last week" bucket.

  Finnhub economic calendar API (finnhub.io):
  - Returns `actual`, `estimate` (consensus), `prev`, `impact`, `country`
    per event in a clean JSON payload — no HTML parsing needed.
  - `time` field is ISO UTC — no ET→UTC conversion risk.
  - Free tier: 60 req/min, no CC required. Already used in backfill_economic_calendar.py.
  - Covers all G8 currencies, medium & high impact events.
  - Actuals populate the same run the event releases — no multi-day lag.

  ForexFactory HTML (Playwright) remains as fallback when FINNHUB_API_KEY
  is unset, preserving backward compatibility.

SOURCE
  Primary:  https://finnhub.io/api/v1/calendar/economic (requires FINNHUB_API_KEY secret)
  Fallback: https://www.forexfactory.com/calendar (Playwright, no key needed)

OUTPUT SCHEMA (ff_calendar.json) — unchanged from v1.x/v2.0
  generated_at  — ISO UTC timestamp
  source        — "Finnhub" or "ForexFactory"
  events[]
    title       — event name
    currency    — G8 currency code
    dateISO     — YYYY-MM-DD (UTC)
    timeUTC     — HH:MM (UTC)
    impact      — "high" | "medium" | "low"
    forecast    — string or null (Finnhub: estimate/consensus)
    previous    — string or null
    actual      — string or null  ← populated in real-time by Finnhub
    released    — bool

FETCH WINDOW
  today - 7 days → today + 14 days (captures last week's actuals + 2 weeks ahead)

HISTORICAL MERGE
  Past events (21-day rolling window) from the previous ff_calendar.json are
  preserved so actuals are not lost between runs.

CONSUMED BY
  calendar-panel.js  → Economic Calendar panel
  fetch_economic_calendar.py → calendar.json → Economic Surprises panel

SCHEDULE (update-ff-calendar.yml)
  4× daily: 00:30, 06:30, 12:30, 20:30 UTC
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_PATH      = "calendar-data/ff_calendar.json"
FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY", "")
FH_BASE          = "https://finnhub.io/api/v1/calendar/economic"
FETCH_TIMEOUT    = 25
FH_RATE_SLEEP    = 0.6   # seconds between calls (60 req/min free tier)
LOOKBACK_DAYS    = 21
FETCH_PAST_DAYS  = 7     # fetch actuals from last 7 days
FETCH_FUTURE_DAYS = 14   # fetch upcoming events 2 weeks out

G8 = {
    "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY",
    "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD",
}
HEADERS = {"User-Agent": "globalinvesting-bot/3.0 (https://globalinvesting.github.io)"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(v) -> str | None:
    """Normalise a value from Finnhub. Returns None for empty/zero-string."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None", "null", "N/A", "—", "-") else s


def _impact(raw: str | None) -> str:
    return {"high": "high", "medium": "medium", "low": "low"}.get(
        (raw or "low").lower(), "low"
    )


# ── Finnhub fetch ─────────────────────────────────────────────────────────────

def fetch_finnhub(date_from: str, date_to: str) -> list[dict]:
    """
    Fetch ALL G8 economic events from Finnhub for the given UTC date range.
    Returns list of normalised ff_calendar.json event dicts.
    Finnhub endpoint returns all countries in one call when no country filter
    is applied — we filter to G8 on our side.
    """
    print(f"  Finnhub: fetching {date_from} → {date_to} ...")
    params = {"from": date_from, "to": date_to, "token": FINNHUB_API_KEY}
    try:
        r = requests.get(FH_BASE, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code == 401:
            print("  ERROR: Finnhub auth failed — FINNHUB_API_KEY invalid or not set.")
            return []
        if r.status_code == 429:
            print("  WARNING: Finnhub rate limit hit — sleeping 30s and retrying.")
            time.sleep(30)
            r = requests.get(FH_BASE, params=params, headers=HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            print(f"  ERROR: Finnhub HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  ERROR: Finnhub request failed — {e}")
        return []

    raw_events = data.get("economicCalendar", []) if isinstance(data, dict) else []
    print(f"  Finnhub: {len(raw_events)} raw events received")

    events = []
    skipped_country = 0
    skipped_impact  = 0

    for ev in raw_events:
        # Country → currency
        iso2 = (ev.get("country") or "").upper()
        currency = G8.get(iso2)
        if not currency:
            skipped_country += 1
            continue

        # Impact filter — keep medium + high only (same as FF panel filter)
        impact = _impact(ev.get("impact"))
        if impact == "low":
            skipped_impact += 1
            continue

        # Title
        title = (ev.get("event") or "").strip()
        if not title:
            continue

        # Time: Finnhub `time` is ISO UTC e.g. "2026-05-22T06:00:00+00:00"
        time_raw = ev.get("time") or ""
        try:
            dt = datetime.fromisoformat(time_raw.replace("Z", "+00:00"))
            dt_utc  = dt.astimezone(timezone.utc)
            date_iso = dt_utc.strftime("%Y-%m-%d")
            time_utc = dt_utc.strftime("%H:%M")
        except Exception:
            date_iso = time_raw[:10] if len(time_raw) >= 10 else ""
            time_utc = "00:00"
        if not date_iso:
            continue

        actual   = _clean(ev.get("actual"))
        forecast = _clean(ev.get("estimate"))   # Finnhub uses "estimate" for consensus
        previous = _clean(ev.get("prev"))

        events.append({
            "title":    title,
            "currency": currency,
            "dateISO":  date_iso,
            "timeUTC":  time_utc,
            "impact":   impact,
            "forecast": forecast,
            "previous": previous,
            "actual":   actual,
            "released": actual is not None,
        })

    print(f"  Finnhub: {len(events)} G8 medium/high events "
          f"(skipped {skipped_country} non-G8, {skipped_impact} low-impact)")
    return events


# ── ForexFactory HTML fallback (Playwright) ───────────────────────────────────

def fetch_forexfactory_fallback() -> list[dict]:
    """
    Fallback: scrape FF HTML with Playwright when FINNHUB_API_KEY is not set.
    Returns list of normalised events (medium+high only).
    NOTE: FF HTML does not reliably include actuals for events that have
    moved out of the current week. Use Finnhub for production.
    """
    print("  Fallback: loading ForexFactory HTML via Playwright ...")
    try:
        from bs4 import BeautifulSoup
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError as e:
        print(f"  ERROR: Playwright/bs4 not available — {e}")
        return []

    FF_URL = "https://www.forexfactory.com/calendar"
    IMPACT_MAP = {"red": "high", "orange": "medium", "yellow": "low", "gray": "low"}
    G8_CCY = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}

    def _et_to_utc(date_iso: str, time_str: str):
        time_str = (time_str or "").strip().lower()
        if not time_str or time_str in ("all day", "tentative", ""):
            return date_iso, "00:00"
        try:
            m = re.match(r"(\d{1,2}):(\d{2})(am|pm)", time_str)
            if not m:
                return date_iso, "00:00"
            h, mn, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
            if ampm == "pm" and h != 12: h += 12
            if ampm == "am" and h == 12: h = 0
            y, mo, da = int(date_iso[:4]), int(date_iso[5:7]), int(date_iso[8:10])
            is_edt = (mo > 3 and mo < 11) or (mo == 3 and da >= 8) or (mo == 11 and da < 7)
            off = -4 if is_edt else -5
            dt = datetime(y, mo, da, h, mn, tzinfo=timezone(timedelta(hours=off)))
            u = dt.astimezone(timezone.utc)
            return u.strftime("%Y-%m-%d"), u.strftime("%H:%M")
        except Exception:
            return date_iso, "00:00"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                locale="en-US", timezone_id="America/New_York"
            )
            page = ctx.new_page()
            page.goto(FF_URL, wait_until="domcontentloaded", timeout=45_000)
            try:
                page.wait_for_selector("table.calendar__table", timeout=20_000)
            except PlaywrightTimeout:
                pass
            html = page.content()
            browser.close()
    except Exception as e:
        print(f"  ERROR: Playwright failed — {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"calendar__table"))
    if not table:
        print("  ERROR: calendar__table not found.")
        return []

    rows = table.find_all("tr", class_=re.compile(r"calendar__row"))
    now_utc = datetime.now(timezone.utc)
    events, current_date_iso = [], None

    for row in rows:
        if "calendar__row--day-breaker" in row.get("class", []):
            dc = row.find("td", class_=re.compile(r"calendar__date"))
            if dc:
                txt = dc.get_text(strip=True)
                try:
                    parsed = datetime.strptime(f"{txt} {now_utc.year}", "%a %b %d %Y")
                    if abs((parsed - now_utc.replace(tzinfo=None)).days) > 180:
                        parsed = parsed.replace(year=now_utc.year + 1)
                    current_date_iso = parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            continue
        if not row.find("td", class_=re.compile(r"calendar__event")):
            continue
        cc = row.find("td", class_=re.compile(r"calendar__currency"))
        ccy = cc.get_text(strip=True).upper() if cc else ""
        if ccy not in G8_CCY:
            continue
        imp_cell = row.find("td", class_=re.compile(r"calendar__impact"))
        impact = "low"
        if imp_cell:
            sp = imp_cell.find("span")
            if sp:
                cs = " ".join(sp.get("class", []))
                for col, lvl in IMPACT_MAP.items():
                    if col in cs:
                        impact = lvl; break
        if impact == "low":
            continue
        ev_cell = row.find("td", class_=re.compile(r"calendar__event"))
        title = ""
        if ev_cell:
            sp = ev_cell.find("span", class_=re.compile(r"calendar__event-title"))
            title = sp.get_text(strip=True) if sp else ev_cell.get_text(strip=True)
        if not title:
            continue
        tc = row.find("td", class_=re.compile(r"calendar__time"))
        time_et = tc.get_text(strip=True) if tc else ""
        ac = row.find("td", class_=re.compile(r"calendar__actual"))
        fc = row.find("td", class_=re.compile(r"calendar__forecast"))
        pc = row.find("td", class_=re.compile(r"calendar__previous"))
        def cv(x): s = x.get_text().strip() if x else ""; return None if s in ("","—","-") else s
        actual = cv(ac); forecast = cv(fc); previous = cv(pc)
        if current_date_iso:
            date_iso_u, time_utc = _et_to_utc(current_date_iso, time_et)
        else:
            date_iso_u, time_utc = now_utc.strftime("%Y-%m-%d"), "00:00"
        events.append({
            "title": title, "currency": ccy, "dateISO": date_iso_u,
            "timeUTC": time_utc, "impact": impact, "forecast": forecast,
            "previous": previous, "actual": actual, "released": actual is not None,
        })
    print(f"  FF fallback: {len(events)} medium/high events parsed")
    return events


# ── Historical merge ──────────────────────────────────────────────────────────

def load_previous() -> list[dict]:
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
        print(f"  WARNING: Could not read previous file — {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_ff_calendar.py v3.0")

    date_from = (now_utc - timedelta(days=FETCH_PAST_DAYS)).strftime("%Y-%m-%d")
    date_to   = (now_utc + timedelta(days=FETCH_FUTURE_DAYS)).strftime("%Y-%m-%d")

    # Step 1: Fetch fresh events
    if FINNHUB_API_KEY:
        fresh = fetch_finnhub(date_from, date_to)
        source = "Finnhub"
    else:
        print("  WARNING: FINNHUB_API_KEY not set — using ForexFactory fallback.")
        print("           Register free at https://finnhub.io and add FINNHUB_API_KEY secret.")
        fresh = fetch_forexfactory_fallback()
        source = "ForexFactory"

    if not fresh:
        print("  ERROR: No events fetched — preserving previous file.")
        sys.exit(0)

    released_fresh = sum(1 for e in fresh if e.get("released"))
    print(f"  Fetched: {len(fresh)} events ({released_fresh} with actuals)")

    # Step 2: Historical merge — keep past events outside the fetch window
    cutoff = (now_utc - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    fresh_keys = {(e["currency"], e["dateISO"], e["timeUTC"], e["title"]) for e in fresh}
    merged = 0
    for ev in load_previous():
        d = ev.get("dateISO", "")
        if d < cutoff or d >= date_from:   # skip if outside lookback OR inside fetch window (already refreshed)
            continue
        k = (ev.get("currency",""), d, ev.get("timeUTC",""), ev.get("title",""))
        if k not in fresh_keys:
            fresh.append(ev)
            fresh_keys.add(k)
            merged += 1
    print(f"  Merged: {merged} historical events outside fetch window")

    # Step 3: Sort
    fresh.sort(key=lambda e: (e["dateISO"], e["timeUTC"], e["currency"]))

    # Step 4: Stats
    from collections import Counter
    today = now_utc.strftime("%Y-%m-%d")
    today_evs = [e for e in fresh if e["dateISO"] == today]
    released_today = [e for e in today_evs if e.get("released")]
    high_today = [e for e in today_evs if e["impact"] == "high"]
    impact_dist = Counter(e["impact"] for e in fresh)
    ccy_dist    = Counter(e["currency"] for e in fresh)
    released_total = sum(1 for e in fresh if e.get("released"))

    print(f"\n  Total: {len(fresh)} events | {released_total} with actuals")
    print(f"  Impact: high={impact_dist['high']} medium={impact_dist['medium']}")
    print(f"  Currencies: {dict(sorted(ccy_dist.items()))}")
    print(f"\n  Today ({today}): {len(today_evs)} events | {len(released_today)} released")
    for e in high_today:
        print(f"    {e['timeUTC']} [{e['currency']}] {e['title'][:45]} | actual={e.get('actual') or 'pending'} est={e.get('forecast','—')}")

    # Step 5: Write
    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "events": fresh,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\n✓ {len(fresh)} events written to {OUTPUT_PATH} ({released_total} with actuals) — source: {source}")


if __name__ == "__main__":
    main()
