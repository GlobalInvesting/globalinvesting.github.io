#!/usr/bin/env python3
"""
fetch_ff_calendar.py — v3.6
Fetches the G8 economic calendar with real-time actuals from Finnhub
and writes calendar-data/ff_calendar.json to the public site repo.

v3.6 changes (2026-05-23):
- Fix critical bug: FETCH_TIMEOUT was referenced in fetch_finnhub() but never
  defined in the config block. This caused a NameError caught by the broad
  except clause, silently returning [] and preserving the previous ff_calendar.json
  unchanged on every run. Added FETCH_TIMEOUT = 30 to the config block.

v3.5 changes:
- Smart-change detection: compares new actuals/forecasts against the previous
  ff_calendar.json before writing. If nothing changed, the file is NOT rewritten
  and a CHANGED_FLAG file is NOT created. The workflow reads this flag to decide
  whether to commit and push, keeping git history clean (no "no changes" noise
  commits on every 5-min poll).
- CHANGED_FLAG (/tmp/cal_changed): written with content "1" when at least one
  actual or forecast value differs from the previous file. Absent or "0" = skip commit.

v3.4 changes:
- Dedup step (Step 2d): Finnhub occasionally emits the same event twice with
  slightly different times (e.g. API Crude Oil Stock Change at 20:30 and 21:30
  with identical actual values). When all copies of a (title, currency, date)
  group share the same actual, the earliest-time entry is kept and duplicates
  are dropped. Groups with distinct actuals (prelim vs revised) are preserved.

v3.3 changes:
- derive_previous_from_history(): new step that fills missing `previous` fields
  by finding the most recent prior actual of the same event series in the combined
  ff_calendar + calendar.json history. Fixes Flash PMIs (EUR/GBP/AUD/JPY), energy
  inventories (EIA/API), jobless claims, and other events where Finnhub returns
  previous=null. The prior occurrence's `actual` becomes the current `previous`.
- Fixed _TITLE_IGNORE: removed 'pmi' so it acts as a keyword discriminator,
  preventing false matches between "Manufacturing PMI" and "Manufacturing Production".
- _STRONG_WORDS updated accordingly.

v3.2 change: Added calendar.json enrichment step. Finnhub does not have
licensed actuals for Flash PMIs (EUR/GBP/AUD/JPY) — these are S&P Global
proprietary data. After fetching from Finnhub, the script cross-references
calendar-data/calendar.json (the FRED+FH backfill) to fill in `actual` and
`previous` for events that Finnhub left empty, using fuzzy title matching
on (currency, date, keyword overlap).

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
  today - 21 days → today + 14 days (covers 3 weeks of history to backfill actuals + 2 weeks ahead)

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
OUTPUT_PATH   = "calendar-data/ff_calendar.json"  # output — this script writes here
CALENDAR_PATH = "calendar-data/calendar.json"    # backfill — FRED + Finnhub historical (read-only)
FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY", "")
FH_BASE          = "https://finnhub.io/api/v1/calendar/economic"
CHANGED_FLAG     = "/tmp/cal_changed"    # written "1" if actuals/forecasts changed vs prev file
FH_RATE_SLEEP    = 0.6   # seconds between calls (60 req/min free tier)
FETCH_TIMEOUT    = 30    # seconds — requests.get timeout for Finnhub API calls
LOOKBACK_DAYS    = 21
FETCH_PAST_DAYS  = 21    # fetch actuals from last 21 days (covers 3 weeks of history)
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


# ── calendar.json enrichment ──────────────────────────────────────────────────
# Finnhub does not carry licensed actuals for Flash PMIs (EUR/GBP/AUD/JPY) or
# certain other events. This step reads calendar-data/calendar.json (the FRED+FH
# backfill) and fills in `actual` and `previous` for events that Finnhub left null.
#
# Matching strategy: same currency + same date + fuzzy title keyword overlap (>=2 words).
# The backfill uses different provider names (e.g. "HCOB Manufacturing PMI Flash" vs
# "S&P Global Manufacturing PMI Flash") so exact matching fails — keyword overlap works.

_TITLE_IGNORE = frozenset({
    'flash','prelim','prel','final','s&p','global','rate','index','change',
    'm/m','y/y','q/q','mom','yoy','qoq','of','the','a','and','or','for',
    'hcob','ism','cb','fed','ecb','boe','boj','rba','rbnz','snb','boc',
    'pct',
    # NOTE: 'pmi' intentionally NOT in ignore — it discriminates PMI events from
    # similar events like "Manufacturing Production", "Services Sentiment", etc.
    'jibun','unicredit','markit','nab','westpac','bank',
})

# Words that are distinctive enough to match alone (score=1 threshold OK)
_STRONG_WORDS = frozenset({
    'unemployment','payrolls','cpi','ppi','pce','gdp','pmi',
    'retail','housing','inflation','employment',
    'sentiment','confidence','permits','balance',
    'trade','sales','michigan','chicago','adp','nfp','inventory','jobless','claims',
    'construction','consumer',
    # NOTE: 'manufacturing', 'services', 'composite', 'production', 'industrial'
    # are NOT listed here — they appear across many different event types.
    # They do match correctly when 'pmi' is also present (score >= 2).
})

def _title_keywords(title: str) -> frozenset:
    words = title.lower().replace('/', ' ').replace('-', ' ').split()
    return frozenset(w for w in words if w not in _TITLE_IGNORE and len(w) > 2)


def enrich_from_calendar_json(events: list[dict]) -> int:
    """
    Fill in missing `actual` and `previous` fields by cross-referencing
    calendar-data/calendar.json (FRED + Finnhub backfill).
    Returns count of events enriched.
    """
    if not os.path.exists(CALENDAR_PATH):
        print("  Enrichment: calendar.json not found — skipping.")
        return 0

    try:
        with open(CALENDAR_PATH, encoding="utf-8") as f:
            cal = json.load(f)
        cal_events = cal.get("events", [])
    except Exception as e:
        print(f"  Enrichment: could not read calendar.json — {e}")
        return 0

    # Build lookup: (currency, dateISO) → list of calendar events with actuals
    from collections import defaultdict
    cal_by_cd = defaultdict(list)
    for ce in cal_events:
        if ce.get("actual") is not None:
            cal_by_cd[(ce.get("currency", ""), ce.get("dateISO", ""))].append(ce)

    enriched = 0
    for ev in events:
        needs_actual   = ev.get("actual")   is None
        needs_previous = ev.get("previous") is None
        if not needs_actual and not needs_previous:
            continue

        candidates = cal_by_cd.get((ev["currency"], ev["dateISO"]), [])
        if not candidates:
            continue

        ff_kw = _title_keywords(ev["title"])
        if not ff_kw:
            continue

        best, best_score = None, 0
        for ce in candidates:
            cal_kw = _title_keywords(ce.get("event") or ce.get("title") or "")
            overlap = ff_kw & cal_kw
            score   = len(overlap)
            # Strong single-word match is sufficient; otherwise need 2+
            if score == 1 and not (overlap & _STRONG_WORDS):
                score = 0
            if score > best_score:
                best_score, best = score, ce

        if best and best_score >= 1:
            changed = False
            if needs_actual and best.get("actual") is not None:
                ev["actual"]   = str(best["actual"])
                ev["released"] = True
                changed = True
            if needs_previous and best.get("previous") is not None:
                ev["previous"] = str(best["previous"])
                changed = True
            if changed:
                enriched += 1

    print(f"  Enrichment from calendar.json: {enriched} events updated")
    return enriched


# ── Previous derivation from historical series ────────────────────────────────
# For events where Finnhub has no `previous` (e.g. Flash PMIs for EUR/GBP/AUD/JPY,
# energy inventories, jobless claims), we derive it by finding the most recent prior
# occurrence of the same event series in the combined ff_calendar + calendar.json
# history. The prior occurrence's `actual` becomes the current event's `previous`.
#
# Matching: same currency + keyword overlap >= 1 strong word OR >= 2 words total.
# Using `actual` of the prior event (not `previous`) because that is the value
# that was the "latest reading" when the current event was due.

def derive_previous_from_history(events: list[dict]) -> int:
    """
    Derive missing `previous` values from historical actuals of the same event
    series. Reads both calendar-data/calendar.json and the current ff_calendar
    events list. Returns count of events updated.
    """
    from collections import defaultdict

    # Load calendar.json for additional history
    cal_history: list[tuple] = []   # (dateISO, currency, event_name, value, kw_frozenset)
    if os.path.exists(CALENDAR_PATH):
        try:
            with open(CALENDAR_PATH, encoding="utf-8") as f:
                cal = json.load(f)
            for ce in cal.get("events", []):
                # Prefer actual; fall back to previous when actual is empty/null
                # (calendar.json sometimes stores actual="" for events that have been
                # released but where the backfill script didn't capture the value)
                raw = ce.get("actual")
                val = str(raw).strip() if raw is not None else None
                if not val:
                    raw_prev = ce.get("previous")
                    val = str(raw_prev).strip() if raw_prev is not None else None
                if not val:
                    continue
                kw = _title_keywords(ce.get("event") or "")
                if kw:
                    cal_history.append((
                        ce.get("dateISO", ""),
                        ce.get("currency", ""),
                        ce.get("event", ""),
                        val,
                        kw,
                    ))
        except Exception:
            pass

    # Also include actuals already in the current ff_calendar batch
    ff_history: list[tuple] = []
    for ev in events:
        if ev.get("actual") is not None:
            kw = _title_keywords(ev.get("title") or "")
            if kw:
                ff_history.append((
                    ev.get("dateISO", ""),
                    ev.get("currency", ""),
                    ev.get("title", ""),
                    str(ev["actual"]),
                    kw,
                ))

    combined = cal_history + ff_history

    # Group by currency for fast lookup
    by_ccy: dict[str, list] = defaultdict(list)
    for row in combined:
        by_ccy[row[1]].append(row)

    # Sort each currency list by date descending (most recent first)
    for ccy in by_ccy:
        by_ccy[ccy].sort(key=lambda x: x[0], reverse=True)

    # Commodity words that must match exactly if present in target title
    _COMMODITY_WORDS = frozenset({'gasoline','crude','natural','gas','distillate','heating'})

    derived = 0
    for ev in events:
        if ev.get("previous") is not None:
            continue  # already has previous — skip

        ff_kw = _title_keywords(ev.get("title") or "")
        if not ff_kw:
            continue

        ev_date = ev.get("dateISO", "")
        ev_ccy  = ev.get("currency", "")
        # If this event involves a specific commodity, require that commodity to match
        ev_commodities = ff_kw & _COMMODITY_WORDS

        best_actual: str | None = None
        best_score  = 0

        for h_date, h_ccy, h_event, h_actual, h_kw in by_ccy.get(ev_ccy, []):
            if h_date >= ev_date:
                continue  # must be strictly before this event
            # Commodity guard: if the target has a commodity word (e.g. "gasoline"),
            # the candidate must also contain that word — prevents crude→gasoline matches
            if ev_commodities and not (ev_commodities & h_kw):
                continue
            overlap = ff_kw & h_kw
            score   = len(overlap)
            if score == 1 and not (overlap & _STRONG_WORDS):
                score = 0
            if score > best_score:
                best_score  = score
                best_actual = h_actual
            if best_score >= 2:
                break  # good enough match found, take it

        if best_actual is not None and best_score >= 1:
            ev["previous"] = best_actual
            derived += 1

    print(f"  Previous derived from history: {derived} events updated")
    return derived


def load_previous() -> tuple[list, set]:
    """
    Returns (events_list, actuals_fingerprint).
    actuals_fingerprint: set of (title, currency, dateISO, actual, forecast)
    tuples for all released events — used for smart change detection so the
    workflow only commits when real data values changed.
    """
    if not os.path.exists(OUTPUT_PATH):
        return [], set()
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        released = sum(1 for e in events if e.get("released"))
        print(f"  Previous file: {len(events)} events ({released} released)")
        fingerprint = {
            (e["title"], e["currency"], e["dateISO"],
             str(e.get("actual") or ""), str(e.get("forecast") or ""))
            for e in events
            if e.get("actual") is not None
        }
        return events, fingerprint
    except Exception as e:
        print(f"  WARNING: Could not read previous file — {e}")
        return [], set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_utc = datetime.now(timezone.utc)
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_ff_calendar.py v3.6")

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
    prev_events, prev_fingerprint = load_previous()
    fresh_keys = {(e["currency"], e["dateISO"], e["timeUTC"], e["title"]) for e in fresh}
    merged = 0
    for ev in prev_events:
        d = ev.get("dateISO", "")
        if d < cutoff or d >= date_from:   # skip if outside lookback OR inside fetch window (already refreshed)
            continue
        k = (ev.get("currency",""), d, ev.get("timeUTC",""), ev.get("title",""))
        if k not in fresh_keys:
            fresh.append(ev)
            fresh_keys.add(k)
            merged += 1
    print(f"  Merged: {merged} historical events outside fetch window")

    # Step 2b: Enrich from calendar.json backfill (fills actuals/previous Finnhub can't provide)
    enrich_from_calendar_json(fresh)

    # Step 2c: Derive missing `previous` from historical series
    # Covers Flash PMIs (EUR/GBP/AUD/JPY), energy inventories, jobless claims, etc.
    # where Finnhub returns previous=null but prior actuals exist in the combined history.
    derive_previous_from_history(fresh)

    # Step 2d: Dedup — Finnhub occasionally emits the same release twice with slightly
    # different times (e.g. API Crude Oil at 20:30 and 21:30 with identical actuals).
    # Strategy: for each (title, currency, date) group where all entries share the same
    # actual value (or all are unreleased), keep the entry with the earliest timeUTC.
    # Entries with distinct actuals are kept as separate rows (prelim vs revised, etc).
    dedup_map: dict = {}
    for ev in fresh:
        key = (ev["title"], ev["currency"], ev["dateISO"])
        if key not in dedup_map:
            dedup_map[key] = []
        dedup_map[key].append(ev)
    deduped: list = []
    dedup_removed = 0
    for key, group in dedup_map.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        actuals = [ev.get("actual") for ev in group]
        all_same_actual = len(set(str(a) for a in actuals)) == 1
        if all_same_actual:
            group.sort(key=lambda e: e.get("timeUTC", ""))
            deduped.append(group[0])
            dedup_removed += len(group) - 1
        else:
            deduped.extend(group)
    if dedup_removed:
        print(f"  Deduped: removed {dedup_removed} duplicate entries (same title+currency+date+actual)")
    fresh = deduped

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

    # Step 5: Smart change detection — only write and signal commit if data changed.
    # Compares (title, currency, dateISO, actual, forecast) tuples for all released events.
    # New actuals, changed actuals, and new forecasts all trigger a write.
    # If nothing changed (e.g. a no-op poll between releases), skip the write entirely
    # so the workflow doesn't create a meaningless git commit.
    new_fingerprint = {
        (e["title"], e["currency"], e["dateISO"],
         str(e.get("actual") or ""), str(e.get("forecast") or ""))
        for e in fresh
        if e.get("actual") is not None
    }
    # Also detect new forecasts on future events (not yet released)
    new_forecasts = {
        (e["title"], e["currency"], e["dateISO"], str(e.get("forecast") or ""))
        for e in fresh
        if not e.get("released") and e.get("forecast") is not None
    }
    prev_forecasts = {
        (e["title"], e["currency"], e["dateISO"], str(e.get("forecast") or ""))
        for e in prev_events
        if not e.get("released") and e.get("forecast") is not None
    }

    new_actuals_added   = new_fingerprint - prev_fingerprint
    new_forecasts_added = new_forecasts - prev_forecasts
    data_changed = bool(new_actuals_added or new_forecasts_added)

    if data_changed:
        if new_actuals_added:
            print(f"\n  NEW actuals: {len(new_actuals_added)} event(s) updated")
            for t, ccy, d, act, fc in sorted(new_actuals_added, key=lambda x: x[2]):
                print(f"    {d} [{ccy}] {t[:50]} → actual={act}" + (f" (fc was {fc})" if fc else ""))
        if new_forecasts_added:
            print(f"  NEW forecasts: {len(new_forecasts_added)} event(s) updated")
    else:
        print("\n  No new actuals or forecasts — skipping write and commit.")
        # Clear the flag file so the workflow knows to skip the commit step
        try:
            with open(CHANGED_FLAG, "w") as f:
                f.write("0")
        except Exception:
            pass
        print(f"✓ No changes — {OUTPUT_PATH} unchanged.")
        return

    # Step 5b: Write
    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "events": fresh,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    json.loads(output_json)  # validate before writing
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"✓ {len(fresh)} events written to {OUTPUT_PATH} ({released_total} with actuals) — source: {source}")

    # Step 5c: Signal the workflow that a commit is needed
    try:
        with open(CHANGED_FLAG, "w") as f:
            f.write("1")
    except Exception as e:
        print(f"  WARNING: Could not write changed flag — {e}")


if __name__ == "__main__":
    main()
