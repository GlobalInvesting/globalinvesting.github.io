#!/usr/bin/env python3
"""
fetch_ff_calendar.py — v3.26
v3.26 changes (2026-06-11):
- PRIMARY SOURCE SWITCHED: Myfxbook RSS → Myfxbook HTML page.
  RSS feed only covers a rolling ~24h window and does NOT expose actuals for
  events that have already passed — by the time a cron runs, the actual has
  left the feed. The HTML page (myfxbook.com/forex-economic-calendar) shows
  the full current-week calendar with actuals permanently retained in the
  HTML attributes once released (data-actual OID + span.actualCell). This is
  the same source that was confirmed accessible from GH Actions in v3.23.
  The parser fetch_myfxbook_calendar() already exists (written in v3.23) and
  handles all extraction including holidays. No new parsing code needed.
- REMOVED: fetch_myfxbook_rss() call in main(). MFB_RSS_URL constant kept for
  reference but no longer used. MFB_HTML_URL replaces it as the active constant.
- main() Step 1: GET https://www.myfxbook.com/forex-economic-calendar with
  browser User-Agent + Accept text/html. On failure → exit 0 (preserve previous
  file), same failsafe as v3.25. On success → fetch_myfxbook_calendar(html).
- CF Worker (v5.0) still polls RSS every 30min for near-real-time actual
  detection and dispatch — RSS is fine for that use-case (event is in the feed
  while it is happening). Python script uses HTML for full historical actuals.
- Version bump in main() print statement: v3.25 → v3.26.

v3.25 changes (2026-06-11):
- PRIMARY SOURCE: Myfxbook RSS feed (myfxbook.com/rss/forex-economic-calendar-events).
  RSS is accessible from GitHub Actions (no WAF — confirmed ~127KB response, instant).
  Covers a rolling ~24h window. Impact from sprite class. Actuals appear within
  minutes of release. Accumulation strategy: each run captures newly-released
  actuals; prev_events carry-forward preserves them indefinitely across 21-day window.
- NEW: fetch_myfxbook_rss() — parses RSS XML, extracts G8 med/high events,
  previous/consensus/actual from HTML-encoded <description> <td> fields.
  Country slug from <link> URL maps to G8 currency (same SLUG_TO_CCY as HTML parser).
  Released detection: actual present OR time_left starts with "-" (negative seconds).
- SINGLE SOURCE: FF JSON removed entirely — mixing FF and Myfxbook titles causes
  key mismatches in the (currency, dateISO, timeUTC, title) dedup, silently creating
  duplicates or erasing carry-forward actuals. Myfxbook RSS is the sole source.
  If RSS fetch fails → exit 0 (preserve previous file, same behavior as any failure).
- REMOVED: fetch_myfxbook_calendar() — direct HTML scraping, blocked from both
  GH Actions and CF Worker IPs (Cloudflare WAF / error 1102).
- REMOVED: _normalise_worker_events() — CF Worker /myfxbook endpoint also blocked
  (error 1102 from CF edge to myfxbook.com). CF_WORKER_MFB_URL constant removed.
- REMOVED: FF_BASE_URL, FF_NEXTWEEK_URL constants, FF JSON fetch path in main().
  fetch_forexfactory_json(), fetch_ff_holidays() kept as dead code for reference only.
- Version bump in main() print statement: v3.24 → v3.25.

v3.24 changes (2026-06-10):
- FIX: Myfxbook HTML fetch routed through CF Worker proxy endpoint /myfxbook.
  CF Worker also blocked by Myfxbook WAF (error 1102). Approach abandoned.
- Version bump in main() print statement: v3.23 → v3.24.

v3.23 changes (2026-06-10):
- PRIMARY SOURCE REPLACED: ForexFactory JSON → Myfxbook HTML calendar.
  FF JSON has a 10h+ actual lag and limited forecast coverage. Myfxbook HTML
  (myfxbook.com/forex-economic-calendar) is accessible from GitHub Actions
  (confirmed: 3300 DOM nodes returned with User-Agent header, no WAF block).
  Data is embedded in HTML attributes — no JS execution required:
    previous-value="..."  on <td data-previous="OID">
    concensus="..."       on <td data-concensus="OID" concensus="VALUE">
    <span class="actualCell"><span>VALUE</span></span>
  Myfxbook covers ~8 days ahead (current + next week visible on one page) and
  retains today's actuals post-release. Actual injection lag: unknown but fast
  (HTML calendar updates within minutes on the myfxbook.com site).
- NEW: fetch_myfxbook_calendar() replaces fetch_forexfactory_json(). Parses
  the HTML table using only stdlib `re`. Country slug in URL maps to G8 currency.
  Impact extracted from impact_high/impact_medium/impact_low CSS classes.
  `previous-value` attribute used directly (avoids unit/symbol in display text).
  `concensus` attribute (note: myfxbook typo, preserved) for forecast value.
  Actual: innermost <span> inside class="actualCell", stripped of time-left
  strings ("Xh Ymin") — those indicate event not yet released.
- REMOVED: FF JSON fetch (Steps 1 and nextweek), CF Worker /trigger call,
  FCS API path. KV_PAYLOAD_URL and CF_WORKER_TRIGGER_URL constants removed.
  FF_BASE_URL, FF_NEXTWEEK_URL constants removed.
- RENAMED: source field in ff_calendar.json now "Myfxbook" instead of
  "ForexFactory". Print statements updated accordingly.
- HOLIDAYS: fetch_ff_holidays() now parses Myfxbook HTML for holiday rows
  (impact_no class). Falls back gracefully if no holidays found.
- Step 1b (holidays): sourced from Myfxbook HTML same request, no second fetch.
- Step 1c (CF Worker KV inject): REMOVED — FCS API free tier exhausted and
  has same actual lag. Myfxbook HTML actuals are the direct replacement.
- Version bump in main() print statement: v3.22 → v3.23.
v3.22 changes (2026-06-10):
- UPDATE: CF Worker v3.1 source switch: FCS API replaces FF HTML scraping.
  FF HTML returns HTTP 403 to CF Worker edge IPs (blocked by FF's Cloudflare WAF).
  FCS API (api-v4.fcsapi.com/forex/economy_cal) is a REST API that accepts programmatic
  access from CF edge IPs. Provides actual/forecast/previous for G8 med/high events.
- FIX: Step 1c updated for FCS API date format: "YYYY-MM-DD HH:MM:SS" UTC — was "Jun 10"
  ET format from FF HTML parser. Date extraction now takes raw_date[:10] (YYYY-MM-DD).
- NEW: Step 1c now calls CF Worker /trigger before reading /payload. This fires an
  on-demand FCS poll (1 API credit) so actuals are fresh rather than waiting for the
  30-min cron. 6-second sleep gives the Worker time to complete the async FCS fetch.
- Added CF_WORKER_TRIGGER_URL constant alongside KV_PAYLOAD_URL.
- KV_PAYLOAD_URL comment updated: "HTML-scraped" → "FCS-API actuals (v3.1)".
v3.21 changes (2026-06-10):
- FIX: Actuals carried forward from prev_events when FF JSON returns actual=None for
  an already-released event. New Step 2a: for each event in `fresh` that matches a
  prev_event by (currency, dateISO, timeUTC, title), if fresh.actual is None and
  prev.actual is not None, carry prev.actual and prev.released into fresh. This prevents
  FF JSON lag (or transient misses) from erasing actuals that were successfully fetched in
  a prior run. Fixes: CPI, BOC, etc. showing "—" hours after release.
- FIX: Expand _IMPACT_UPGRADES with medium-impact events that FF JSON consistently
  mislabels as "Low": EIA Crude Oil Inventories (USD), Federal Budget Balance (USD),
  RICS House Price Balance (GBP), BSI Manufacturing Index (JPY), MI Inflation
  Expectations (AUD), ANZ Business Confidence (NZD). These events were silently
  dropped despite appearing as orange (medium) on FF's HTML calendar.
- FIX: Also fetch ff_calendar_nextweek.json (second FF request) to include next-week
  events in the panel. Previously only thisweek.json was fetched, leaving Mon-Sun+1
  through Mon-Sun+2 events absent from the calendar panel.

v3.19 changes (2026-06-10):
- FF-only pipeline: removed all calendar.json (FRED + Finnhub) dependencies.
  CALENDAR_PATH constant removed. enrich_from_calendar_json() converted to no-op
  stub. derive_forecast_from_history() and derive_previous_from_history() now
  source history exclusively from the ff_calendar.json rolling accumulation
  (fresh events + merged prev_events). The FMP 90-day backfill
  (backfill_ff_calendar.py) seeds this history so all three derivation functions
  have sufficient data from day one. Eliminates the cross-source naming-mismatch
  bugs (e.g. FRED CPI index level 335.12 contaminating FF CPI m/m percentage
  actuals via fuzzy title matching).

v3.18 changes (2026-06-10):
- Historical merge fixed: removed `d >= date_from` guard that excluded all events
  within the 21-day lookback when FF JSON only covers the current week (v3.17+).
  Now merges all prev_events not in fresh_keys; dedup handles duplicates downstream.
- `_title_keywords()` normalises FF slash-notation before split: m/m→mom, y/y→yoy,
  q/q→qoq. Fixes CPI index level (335.12) contaminating CPI m/m/y/y actuals via the
  enrich_from_calendar_json step — variant guard now fires correctly for FF titles.

Fetches the G8 economic calendar with real-time actuals from the ForexFactory
public JSON feed and writes calendar-data/ff_calendar.json to the public site repo.

v3.17 changes (2026-06-10):
- PRIMARY SOURCE MIGRATED: Finnhub → ForexFactory public JSON
  (https://nfs.faireconomy.media/ff_calendar_thisweek.json).
  Finnhub free tier blocks the /api/v1/calendar/economic endpoint with HTTP 403.
  The same ForexFactory public JSON already used by fetch_ff_holidays() and by
  the Cloudflare Worker (calendar-watcher.js v2.0) is now the primary fetch source.
  New function: fetch_forexfactory_json() — fetches the current-week events JSON,
  parses title/country/date/impact/actual/forecast/previous fields, converts ET→UTC,
  filters to G8 medium+high impact, and returns the normalised event list.
  The Playwright HTML fallback (fetch_forexfactory_fallback()) is removed — it was
  fragile, slow, and required headless Chromium. The historical merge (Step 2) in
  main() now carries the full weight of multi-week history, using prev_events from
  the previous ff_calendar.json to preserve actuals outside the current-week window.
  FINNHUB_API_KEY is no longer read or required. The workflow step that passed
  FINNHUB_API_KEY as an env var has been updated accordingly.
  _IMPACT_UPGRADES table retained — FF may still need minor override corrections.
- FETCH WINDOW: current-week JSON covers Mon–Sun of the current week (FF convention).
  Events outside this window are merged from prev_events as before (21-day lookback).
- NO CHANGE to enrichment, dedup, scoring, forecast derivation, or output schema.

v3.16 changes (2026-06-10):
- Structural change detection: the smart change-detection block (Step 5) now also
  triggers a write when dedup passes (same-day Step 2d or cross-day Step 2e) remove
  phantom upcoming entries that existed in the previous ff_calendar.json. Previously,
  a run that only removed phantoms (no new actuals/forecasts/holidays) computed
  data_changed=False and skipped the write, leaving stale phantom rows in the file
  until the next run with a genuine actual change. Fix: compares the set of unreleased
  upcoming events in prev_events vs fresh; any phantom removed by either dedup pass
  sets structural_changed=True and forces a write+commit. On subsequent runs, both
  sets are identical (phantoms gone from both) so structural_changed=False and the
  no-op path is restored. This is self-healing: exactly one extra commit is produced
  (the one that actually removes the phantoms), then it goes silent.

v3.15 changes (2026-06-09):
- Cross-day dedup (Step 2d extension): Finnhub occasionally emits the same weekly event
  twice — once with an actual (the release that already occurred) and again 1-7 days later
  as an unreleased "upcoming" entry with the same timeUTC (e.g. Westpac Consumer Confidence
  at 00:30 on Jun 9 with actual=-2.9, AND again on Jun 10 with no actual). This caused the
  panel to show both the old released row AND a phantom duplicate for the "next" occurrence.
  Fix: after the same-day dedup pass, a cross-day pass removes any unreleased event
  (actual=None, released=False) when an entry for the same (title, currency, timeUTC)
  already has an actual within the prior 7 days. The released entry is kept; the duplicate
  phantom upcoming entry is dropped.

v3.14 changes (2026-06-09):
- Impact override table (_IMPACT_UPGRADES): events that Finnhub classifies as "low" impact
  but ForexFactory and FXStreet list as "medium" (orange) are now upgraded before the
  low-impact filter runs. This corrects systematic under-classification for:
    • JPY Producer Price Index MoM and YoY (forexfactory.com: medium/orange)
    • JPY Prelim Machine Tool Orders YoY (forexfactory.com: medium/orange)
    • AUD Building Approvals MoM / YoY (forexfactory.com: medium/orange)
    • AUD Private House Approvals MoM (forexfactory.com: medium/orange)
  These events were previously silently dropped from ff_calendar.json, causing the
  economic calendar panel to show gaps for JPY data releases (e.g. JPY PPI on 2026-06-09
  was absent from the panel despite being a market-moving release at 6.3% YoY actual).
  The override table is defined once before the event loop (not per-iteration) and uses
  case-insensitive substring matching on the event title — robust to minor Finnhub naming
  variations. Upgrades are logged implicitly via the existing skipped_impact counter
  (upgraded events do not increment skipped_impact).

v3.13 changes (2026-05-27):
- Industry-standard audit of last-known-consensus (derive_forecast_from_history):
  1. Released-event guard: derive_forecast_from_history() now skips events that
     are already released (actual != None or released == True). Filling a stale
     consensus forecast AFTER an event has released is not standard practice —
     Bloomberg/Reuters only show estimate vs actual when the estimate was published
     before the release. Without this guard, a released event with no Finnhub
     estimate would get a derived "*" forecast that could suggest a spurious miss.
  2. Recency window on eligibility count (ELIGIBILITY_WINDOW_DAYS=180): the
     consensus eligibility gate (MIN_FORECAST_HISTORY=2) now only counts prior
     forecasts within the last 180 days. Previously any historical forecast —
     including ones from years ago — counted toward eligibility, meaning a series
     that had forecasts years ago but whose API coverage lapsed would still pass
     the gate and receive a stale derived forecast. 180 days accommodates quarterly
     series (GDP, Corporate Profits, current account) while excluding genuinely
     discontinued series. Mirrors Refinitiv's recency window for consensus eligibility.
  3. Z-score magnitude guard (Z_SCORE_THRESHOLD=3.0): after a best-match forecast
     is found, it is validated against the recent actuals of the same matched series
     from ff_history (Finnhub-sourced data only). If the candidate forecast value is
     a numeric outlier more than 3 standard deviations from the series mean, the
     derived forecast is suppressed. This catches unit-mismatch scenarios where
     FRED historical data stored values in a different unit scale than Finnhub
     (e.g. ADP in thousands vs. FRED raw units). Requires at least
     Z_SCORE_MIN_ACTUALS=3 Finnhub actuals to fire; otherwise the guard is skipped.
     Non-numeric forecast values (e.g. "0.3%", "Unchanged") pass through unchanged.
  4. BUG FIX: main() print statement corrected from "v3.8" to "v3.13" — the version
     string was hardcoded at v3.8 since the initial implementation and never updated.

v3.12 changes (2026-05-27):
- Scoring model fully rewritten (three-tier non-variant scoring):
  1. Variant guard: (target & _VARIANT_WORDS) == (history & _VARIANT_WORDS) — unchanged from v3.10.
  2. Non-variant score: score = len(overlap - _VARIANT_WORDS). Variant words excluded from count,
     preventing {spending, yoy} from scoring 2 and matching "Capital Spending YoY" ←→ "Household Spending YoY".
  3. Minimum-score thresholds (new in v3.12):
       nv_score == 1  → word must be in _STRONG_SINGLE_OK (high-specificity event words only)
       nv_score == 2  → at least 1 word must be in _ANCHOR (_STRONG_SINGLE_OK | _SECTOR_ANCHOR)
       nv_score >= 3  → always OK
     This blocks {home, sales}, {crude, oil}, {autos, retail}, {change, eia}, {pmi} alone, etc.
     while keeping all legitimate same-series matches.
- _STRONG_SINGLE_OK: refined set of words sufficient alone for a match. Removed:
    pmi        (Manufacturing PMI ≠ Services PMI ≠ Chicago PMI — needs companion)
    sales      (Existing Home Sales ≠ Retail Sales — needs companion)
    retail     (Retail Sales ≠ Retail Inventories — needs companion)
    consumer   (Consumer Confidence ≠ Consumer Credit — needs companion)
    confidence (Consumer Confidence ≠ Business Confidence — needs companion)
    change     (generic suffix, appears in hundreds of unrelated events)
    prel       (release-stage qualifier, not an event concept)
    chicago    (moved to _SECTOR_ANCHOR — Chicago PMI passes via chicago+pmi=2, not chicago alone)
- _SECTOR_ANCHOR: new set of domain-specific words that anchor a nv_score=2 match:
    manufacturing, services, composite, industrial, production,
    consumer, confidence, household, factory, orders, spending,
    business, current, account, chicago
- Commodity guard added to derive_forecast_from_history (was already in derive_previous):
    EIA Gasoline Stocks Change ← EIA Crude Oil Stocks Change now blocked (gasoline ≠ crude).
    API Crude Oil Stock Change ← EIA Crude Oil Stocks Change still passes (same commodity).
- False matches eliminated vs v3.8 baseline: Building Permits MoM ← Building Permits (absolute),
  ADP Weekly ← ADP Monthly, Housing Starts MoM ← Housing Starts, CPI YoY ← CPI MoM,
  Existing Home Sales MoM ← Retail Sales MoM, Corporate Profits QoQ ← Nonfarm Productivity QoQ,
  ISM Services ← Chicago PMI, EIA Gasoline ← EIA Natural Gas, Chicago Fed ← Chicago PMI, and more.

v3.11 changes (2026-05-27):
- Fix secondary cross-series contamination in score=1 guard: variant words (mom/yoy/qoq/weekly)
  should NOT be sufficient by themselves to authorize a single-word match. Previously, the guard
  was: score=1 only if overlap contains a STRONG_WORD. Since mom/yoy/qoq/weekly are in STRONG_WORDS,
  "Housing Starts MoM" still matched "Factory Orders MoM" (overlap={mom}, score=1 → allowed).
  Fix: add _STRONG_NON_VARIANT = _STRONG_WORDS - _VARIANT_WORDS. All four score=1 guards in
  enrich_from_calendar_json(), derive_forecast_from_history() (both loops), and
  derive_previous_from_history() now check overlap against _STRONG_NON_VARIANT instead of
  _STRONG_WORDS. A variant-only overlap (e.g. {mom}, {weekly}) now scores 0 and is rejected
  unless another non-variant strong word is also in the overlap.
  Examples now correctly BLOCKED:
    • "Housing Starts MoM" vs "Factory Orders MoM" (overlap={mom} → score=0)
    • "Building Permits MoM Prel" vs "Retail Sales MoM" (overlap={mom} → score=0)
    • "API Weekly Crude Oil Stock" vs "ADP Employment Change Weekly" (overlap={weekly} → score=0)
  Examples still correctly PASSING:
    • "Housing Starts MoM" vs "Housing Starts MoM" (overlap={starts,housing,mom} → score=3)
    • "CPI MoM" vs "Core CPI MoM" (overlap={cpi,mom} → score=2, cpi is non-variant strong)
    • "Retail Sales MoM" vs "Retail Sales MoM Prel" (overlap={retail,sales,mom} → score=3)

v3.10 changes (2026-05-27):
- Fix variant guard in derive_forecast_from_history() and derive_previous_from_history():
  _VARIANT_WORDS was defined in v3.9 but never wired into the actual matching loops.
  The variant guard is now applied in BOTH the eligibility-count loop and the
  best-match loop of derive_forecast_from_history(), and in the best-match loop of
  derive_previous_from_history(). Guard logic: (h_kw & _VARIANT_WORDS) must equal
  (ff_kw & _VARIANT_WORDS) — if the rate-of-change type differs, the candidate is
  skipped. This prevents cross-series contamination where:
    • "Building Permits MoM Prel" (target, variants={mom}) matches "Building Permits"
      (history, variants={}) scoring 2 on {building, permits} → now BLOCKED.
    • "ADP Employment Change Weekly" (variants={weekly}) matches "ADP Employment Change"
      (variants={}) → now BLOCKED.
    • "Housing Starts MoM" vs "Housing Starts" → now BLOCKED.
    • "CPI YoY" vs "CPI MoM" → now BLOCKED (yoy ≠ mom).
  Correct intra-series matches are preserved:
    • "Building Permits MoM Prel" vs "Building Permits MoM" → PASS (both have mom).
    • "Retail Sales MoM" vs "Retail Sales MoM Prel" → PASS (prel excluded from guard).
    • "Manufacturing PMI Flash" vs "Manufacturing PMI" → PASS (flash excluded from guard).
- _VARIANT_WORDS refined: removed 'prel' and 'flash' from the set (prel/flash vs
  final of the same series are the same release — forecasts are interchangeable and
  should be cross-used). Guard set is now {'mom','yoy','qoq','weekly'} only.

v3.9 changes (2026-05-27):
- Fix cross-series contamination in derive_forecast_from_history() (and derive_previous_from_history()):
  'mom', 'yoy', 'qoq', 'prel', 'change', and 'weekly' were in _TITLE_IGNORE, causing them to be
  stripped before keyword matching. This made "Building Permits MoM Prel" and "Building Permits"
  (absolute level, thousands) produce identical keyword sets, so the last known forecast for the
  absolute-level series (e.g. "2K") was being injected as the derived forecast for the MoM % series.
  Same contamination affected "ADP Employment Change" (monthly, thousands-unit, "6000K") vs
  "ADP Employment Change Weekly" (weekly, raw number).
  Fix: removed 'mom', 'yoy', 'qoq', 'prel', 'change', 'weekly' from _TITLE_IGNORE so they are
  preserved as discriminating keywords. Also added them to _STRONG_WORDS so a single-word match
  on these suffixes is sufficient for the score threshold. Result: series variants that differ only
  by rate-of-change suffix now resolve to distinct keyword sets and will not cross-contaminate.
  Affected events in the May 27 run: Building Permits MoM Prel (was "2K*" → now no derived forecast
  until a proper MoM history entry exists), ADP Employment Change Weekly (was "6000K*" → now none),
  Housing Starts MoM (was "1K*" → now none), Retail Sales MoM Prel (was "1.2%*" — this one was
  actually correct since it matched prior MoM forecasts, so it will continue working).

v3.8 changes (2026-05-27):
- derive_forecast_from_history(): new Step 2e that fills missing `forecast` fields
  using last-known-consensus fallback — the industry standard approach used by
  Bloomberg Terminal and Reuters Eikon. For event series that structurally carry
  analyst consensus (ADP, MBA Mortgage Rate, CBI Distributive Trades, etc.), when
  Finnhub returns estimate=null (timing gap, API tier, or provider lag), the most
  recent prior forecast for that series is used as a fallback.
  Two safeguards mirror industry practice:
    1. Consensus eligibility gate (MIN_FORECAST_HISTORY=2): only fills series that
       had a forecast in at least 2 prior occurrences — prevents filling speech,
       testimony, and other no-consensus event types.
    2. Derived values are suffixed with "*" (e.g. "0.3%*") so the frontend can
       render them in a muted style, distinct from live estimates. Users can see
       the value is the last known consensus, not a fresh provider estimate.
  History is sourced from calendar.json (FRED+Finnhub backfill) and the current
  ff_calendar batch. Matching uses the same keyword overlap logic as
  derive_previous_from_history().

v3.7 changes (2026-05-25):
- fetch_ff_holidays(): new step that fetches bank/market holidays from the
  ForexFactory public JSON feed (nfs.faireconomy.media/ff_calendar_thisweek.json).
  FF uses impact="Holiday" to mark days when a G8 market is closed. These events
  explain why certain symbols show stale quotes on days that are not weekends —
  the underlying exchange is closed. The holidays are written as a separate top-level
  field `holidays` in ff_calendar.json (list of {title, currency, dateISO}) so the
  frontend can surface a visual indicator ("Market closed — public holiday") instead
  of showing a stale/frozen price with no context.
  The holiday fetch is always attempted regardless of whether Finnhub event data
  changed — holidays can appear or disappear mid-week as FF updates their feed.
  Change-detection fingerprint extended to include holidays so a new/removed holiday
  triggers a commit even when economic event actuals are unchanged.
- Output schema extended: `holidays` top-level field (see schema below).

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

SOURCE
  Primary:  https://nfs.faireconomy.media/ff_calendar_thisweek.json (public, no key required)
  Coverage: current calendar week (Mon–Sun, ET). Prior-week actuals preserved via
            historical merge from prev_events (21-day rolling lookback in main()).

OUTPUT SCHEMA (ff_calendar.json) — v3.7
  generated_at  — ISO UTC timestamp
  source        — "ForexFactory"
  holidays[]    — bank/market holidays this week from ForexFactory public JSON
    title       — holiday name (e.g. "Memorial Day", "Bank Holiday")
    currency    — G8 currency code of the affected market
    dateISO     — YYYY-MM-DD
  events[]
    title       — event name
    currency    — G8 currency code
    dateISO     — YYYY-MM-DD (UTC)
    timeUTC     — HH:MM (UTC)
    impact      — "high" | "medium" | "low"
    forecast    — string or null (ForexFactory consensus estimate)
    previous    — string or null
    actual      — string or null  ← populated in real-time by ForexFactory
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
import sys
from datetime import datetime, timezone, timedelta

import requests

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_PATH   = "calendar-data/ff_calendar.json"  # output — this script writes here
# Myfxbook RSS — primary and only source (no WAF, accessible from GH Actions, ~24h rolling window)
MFB_RSS_URL  = "https://www.myfxbook.com/rss/forex-economic-calendar-events"  # kept for CF Worker reference
MFB_HTML_URL = "https://www.myfxbook.com/forex-economic-calendar"
CHANGED_FLAG = "/tmp/cal_changed"    # written "1" if actuals/forecasts changed vs prev file
FETCH_TIMEOUT    = 30    # seconds — requests.get timeout
LOOKBACK_DAYS    = 21
FETCH_PAST_DAYS  = 21    # historical merge lookback (prev_events preserved up to 21 days)
FETCH_FUTURE_DAYS = 14   # upcoming events window

G8 = {
    "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY",
    "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD",
}
# Reverse map: currency → ISO2 country code (for holiday matching)
G8_CCY_TO_COUNTRY = {v: k for k, v in G8.items()}
# FF uses country names in the holiday title — map known country strings to G8 currencies
FF_COUNTRY_NAME_TO_CCY = {
    "united states": "USD", "us": "USD",
    "eurozone": "EUR", "euro zone": "EUR", "european": "EUR",
    "united kingdom": "GBP", "uk": "GBP", "britain": "GBP",
    "japan": "JPY", "japanese": "JPY",
    "australia": "AUD", "australian": "AUD",
    "canada": "CAD", "canadian": "CAD",
    "switzerland": "CHF", "swiss": "CHF",
    "new zealand": "NZD",
}
HEADERS = {"User-Agent": "globalinvesting-bot/3.0 (https://globalinvesting.github.io)"}

# ── Myfxbook RSS calendar fetch ──────────────────────────────────────────────

def fetch_myfxbook_rss(xml: str) -> tuple[list[dict], list[dict]]:
    """
    Parse G8 economic events from the Myfxbook RSS feed XML.

    Feed URL: https://www.myfxbook.com/rss/forex-economic-calendar-events
    No WAF — accessible from GitHub Actions runner IPs (confirmed).
    Covers a rolling ~24h window. Medium+high events only.

    Each <item> has:
      <title>   — event name
      <link>    — https://www.myfxbook.com/forex-economic-calendar/{country_slug}/{event_slug}
      <pubDate> — RFC 2822 datetime in UTC (e.g. "Thu, 11 Jun 2026 12:15 GMT")
      <description> — HTML-encoded <table> with one data row:
          <td> time_left </td>       e.g. "10h 49min" or "-1534 seconds"
          <td> <span class="sprite sprite-common sprite-{impact}-impact"> </td>
          <td> previous_value </td>
          <td> consensus_value </td>
          <td> actual_value </td>

    Returns (events, holidays):
      events  — list of normalised ff_calendar.json event dicts (medium + high, G8)
      holidays — empty list (RSS does not include bank holiday entries)
    """
    import re as _re
    from email.utils import parsedate_to_datetime as _parse_dt

    SLUG_TO_CCY: dict[str, str] = {
        "united-states": "USD", "euro-area": "EUR", "germany": "EUR",
        "france": "EUR", "italy": "EUR", "spain": "EUR", "netherlands": "EUR",
        "belgium": "EUR", "ireland": "EUR", "portugal": "EUR", "finland": "EUR",
        "austria": "EUR", "greece": "EUR", "european-union": "EUR",
        "united-kingdom": "GBP", "japan": "JPY", "canada": "CAD",
        "australia": "AUD", "new-zealand": "NZD", "switzerland": "CHF",
    }
    G8_CCY = set(SLUG_TO_CCY.values())

    events: list[dict] = []
    skipped_ccy = skipped_impact = 0

    for item in _re.finditer(r'<item>(.*?)</item>', xml, _re.DOTALL):
        block = item.group(1)

        # Currency from country slug in <link>
        link_m = _re.search(
            r'<link>https?://[^/]+/forex-economic-calendar/([^/]+)/[^<]+</link>', block
        )
        if not link_m:
            continue
        currency = SLUG_TO_CCY.get(link_m.group(1), "")
        if not currency:
            skipped_ccy += 1
            continue

        # Event title
        title_m = _re.search(r'<title>([^<]+)</title>', block)
        title = title_m.group(1).strip() if title_m else ""
        if not title:
            continue

        # Date/time from <pubDate> (RFC 2822, UTC)
        date_iso, time_utc = "", "00:00"
        pub_m = _re.search(r'<pubDate>([^<]+)</pubDate>', block)
        if pub_m:
            try:
                dt = _parse_dt(pub_m.group(1).strip())
                date_iso = dt.strftime("%Y-%m-%d")
                time_utc = dt.strftime("%H:%M")
            except Exception:
                pass
        if not date_iso:
            continue

        # Decode HTML entities in <description> (&#60; = <, &#62; = >, &#39; = ')
        desc_m = _re.search(r'<description>(.*?)</description>', block, _re.DOTALL)
        if not desc_m:
            continue
        desc_raw = desc_m.group(1)
        desc = (desc_raw
                .replace('&#60;', '<').replace('&#62;', '>')
                .replace('&#39;', "'").replace('&lt;', '<')
                .replace('&gt;', '>').replace('&amp;', '&'))

        # Impact from sprite class in decoded description
        imp_m = _re.search(r'sprite-(high|medium|low)-impact', desc)
        impact = imp_m.group(1) if imp_m else "low"
        if impact == "low":
            skipped_impact += 1
            continue

        # Values: extract all <td> text nodes from data row (skip header <th> row)
        tds = _re.findall(r'<td>(.*?)</td>', desc, _re.DOTALL)

        def _td(s: str) -> str | None:
            v = _re.sub(r'<[^>]+>', '', s).strip()
            return v if v and v not in ('-', '') else None

        time_left_raw = _td(tds[0]) if tds else None
        previous = _td(tds[2]) if len(tds) > 2 else None
        forecast = _td(tds[3]) if len(tds) > 3 else None
        actual   = _td(tds[4]) if len(tds) > 4 else None

        # Released: actual present OR time_left is negative seconds (already passed)
        is_passed = bool(time_left_raw and time_left_raw.startswith("-") and "second" in time_left_raw)
        released  = actual is not None or is_passed

        events.append({
            "title":    title,
            "currency": currency,
            "dateISO":  date_iso,
            "timeUTC":  time_utc,
            "impact":   impact,
            "forecast": _clean(forecast),
            "previous": _clean(previous),
            "actual":   _clean(actual),
            "released": released,
        })

    events.sort(key=lambda e: (e["dateISO"], e["timeUTC"], e["currency"]))
    released_count = sum(1 for e in events if e.get("released"))
    print(f"  Myfxbook RSS: {len(events)} G8 medium/high events "
          f"({released_count} with actuals; skipped {skipped_ccy} non-G8, "
          f"{skipped_impact} low-impact)")
    return events, []


def fetch_myfxbook_calendar(html: str) -> tuple[list[dict], list[dict]]:
    """
    Parse G8 economic events AND bank holidays from the Myfxbook HTML calendar page.

    Data is embedded in HTML attributes — no JS execution required:
      data-calendardatetd="YYYY-MM-DD HH:MM:SS.0"  (UTC — myfxbook stores in UTC)
      importance="1|2|3"                             (1=Low, 2=Medium, 3=High)
      currency 3-letter code in a plain <td>
      <a href="https://www.myfxbook.com/forex-economic-calendar/{country}/{slug}" class="calendar-event-link" ...>
      previous-value="..."
      concensus="VALUE" on <td data-concensus="OID" concensus="VALUE">
      <span class="actualCell"><span>VALUE</span> or countdown text</span>

    Returns (events, holidays) where:
      events  — list of normalised ff_calendar.json event dicts (medium + high, G8)
      holidays — list of holiday dicts {title, currency, dateISO}
    """
    import re as _re

    G8_CCY = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}

    # Country slug → currency code for G8 countries + EUR-area members
    SLUG_TO_CCY: dict[str, str] = {
        "united-states": "USD", "euro-area": "EUR", "germany": "EUR",
        "france": "EUR", "italy": "EUR", "spain": "EUR", "netherlands": "EUR",
        "belgium": "EUR", "ireland": "EUR", "portugal": "EUR", "finland": "EUR",
        "austria": "EUR", "greece": "EUR", "european-union": "EUR",
        "united-kingdom": "GBP", "japan": "JPY", "canada": "CAD",
        "australia": "AUD", "new-zealand": "NZD", "switzerland": "CHF",
    }

    # Build lookup dicts by OID from the raw HTML (full page)
    # previous-value
    prev_map: dict[str, str] = dict(
        _re.findall(r'data-previous="(\d+)"[^>]*previous-value="([^"]*)"', html)
    )
    # forecast/consensus — pattern: data-concensus="OID" concensus="VALUE" consistconcensus
    cons_map: dict[str, str] = {}
    for oid, val in _re.findall(
        r'<td[^>]*data-concensus="(\d+)" concensus="([^"]*)" consistconcensus', html
    ):
        cons_map[oid] = val

    # actual — innermost <span> inside actualCell; exclude countdown strings (contain "min" / "h ")
    actual_map: dict[str, str] = {}
    for m in _re.finditer(
        r'data-actual="(\d+)".*?class="actualCell">(.*?)</span>\s*</span>',
        html, _re.DOTALL
    ):
        oid = m.group(1)
        inner = m.group(2)
        spans = _re.findall(r'<span[^>]*>\s*([^<]+?)\s*</span>', inner)
        raw_val = (spans[-1].strip() if spans else _re.sub(r'<[^>]+>', '', inner).strip())
        # Skip countdown strings — event not yet released
        if raw_val and "min" not in raw_val and "h " not in raw_val and raw_val not in ("", "-"):
            actual_map[oid] = raw_val

    # Split into <tr> blocks
    tbody_m = _re.search(r'<tbody>(.*?)</tbody>', html, _re.DOTALL)
    if not tbody_m:
        print("  Myfxbook: <tbody> not found in HTML — source may have changed.")
        return [], []
    tbody = tbody_m.group(1)
    all_trs = _re.findall(r'<tr[^>]*>(.*?)</tr>', tbody, _re.DOTALL)

    events: list[dict] = []
    holidays: list[dict] = []
    skipped_ccy = skipped_impact = 0

    for tr in all_trs:
        oid_m = _re.search(r'id="itemOid" value="(\d+)"', tr)
        if not oid_m:
            continue
        oid = oid_m.group(1)

        dt_m = _re.search(r'data-calendardatetd="([^"]+)"', tr)
        if not dt_m:
            continue
        dt_raw = dt_m.group(1)          # "2026-06-10 12:30:00.0" UTC
        date_iso = dt_raw[:10]
        time_utc = dt_raw[11:16]        # "12:30"

        # Currency: standalone <td> with exactly 3 uppercase letters
        cur_m = _re.search(r'<td[^>]*calendarToggleCell[^>]*>\s*([A-Z]{3})\s*</td>', tr)
        currency = cur_m.group(1).strip() if cur_m else ""

        # Impact from importance attribute (1=Low, 2=Medium, 3=High)
        imp_m = _re.search(r'importance="(\d)"', tr)
        imp_num = int(imp_m.group(1)) if imp_m else 0
        if imp_num == 3:
            impact = "high"
        elif imp_num == 2:
            impact = "medium"
        else:
            impact = "low"

        # Event name from calendar-event-link anchor
        # href attribute comes before class in myfxbook HTML.
        # href may be absolute (https://www.myfxbook.com/...) or relative.
        ev_m = _re.search(r'<a href="(?:https?://[^/]+)?(/forex-economic-calendar/[^"]+)"[^>]*calendar-event-link[^>]*>([^<]+)</a>', tr)
        if not ev_m:
            # Holiday rows have no calendar-event-link
            # Check for holiday: impact_no class or "Holiday" in title
            if _re.search(r'impact_no', tr):
                # extract title text from td
                title_m = _re.search(r'class="[^"]*calendarToggleCell[^"]*text-left[^"]*"[^>]*>(.*?)</td>', tr, _re.DOTALL)
                title = _re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else "Bank Holiday"
                if currency in G8_CCY and date_iso:
                    holidays.append({"title": title or "Bank Holiday", "currency": currency, "dateISO": date_iso})
            continue

        event_slug_url = ev_m.group(1)   # "/forex-economic-calendar/united-states/cpi-s-a"
        event_name = ev_m.group(2).strip()

        # Period suffix e.g. "(May)"
        period_m = _re.search(r'<span>\(([^)]+)\)</span>', tr)
        if period_m:
            event_name = f"{event_name} ({period_m.group(1)})"

        # Derive currency from URL slug if td extraction failed
        if not currency or currency not in G8_CCY:
            parts = event_slug_url.strip("/").split("/")
            # parts[0]="forex-economic-calendar", parts[1]=country_slug, parts[2]=event_slug
            country_slug = parts[1] if len(parts) >= 2 else ""
            currency = SLUG_TO_CCY.get(country_slug, "")

        if not currency or currency not in G8_CCY:
            skipped_ccy += 1
            continue

        if impact == "low":
            skipped_impact += 1
            continue

        # Values from attribute lookup
        previous = _clean(prev_map.get(oid, ""))
        forecast = _clean(cons_map.get(oid, ""))
        actual   = _clean(actual_map.get(oid, ""))

        # Mark released if actual present OR if ispassed="1" on the calendarLeft span
        is_passed = bool(_re.search(r'ispassed="1"', tr))
        released  = actual is not None or is_passed

        events.append({
            "title":    event_name,
            "currency": currency,
            "dateISO":  date_iso,
            "timeUTC":  time_utc,
            "impact":   impact,
            "forecast": forecast,
            "previous": previous,
            "actual":   actual,
            "released": released,
        })

    events.sort(key=lambda e: (e["dateISO"], e["timeUTC"], e["currency"]))
    released_count = sum(1 for e in events if e.get("released"))
    print(f"  Myfxbook: {len(events)} G8 medium/high events "
          f"({released_count} with actuals; skipped {skipped_ccy} non-G8, "
          f"{skipped_impact} low-impact, {len(holidays)} holidays)")
    return events, holidays



def _normalise_worker_events(raw_events: list) -> list[dict]:
    """
    Normalise CF Worker /myfxbook response events to ff_calendar.json event schema.

    Worker events have:
      date     — "YYYY-MM-DD HH:MM:SS" UTC
      dateISO  — "YYYY-MM-DD"
      timeUTC  — "HH:MM"
      currency — 3-letter G8 code
      impact   — "high" | "medium"
      title    — event name (may include period suffix)
      actual   — string or null
      forecast — string or null
      previous — string or null
      released — bool

    Returns list of normalised dicts matching the ff_calendar.json schema.
    """
    events = []
    for ev in raw_events:
        date_iso = ev.get("dateISO") or (ev.get("date") or "")[:10]
        time_utc = ev.get("timeUTC") or (ev.get("date") or " 00:00")[11:16]
        if not date_iso:
            continue
        currency = (ev.get("currency") or "").strip().upper()
        G8_CCY = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
        if currency not in G8_CCY:
            continue
        impact_raw = (ev.get("impact") or "").lower()
        if impact_raw not in ("high", "medium"):
            continue
        actual   = _clean(ev.get("actual"))
        forecast = _clean(ev.get("forecast"))
        previous = _clean(ev.get("previous"))
        released = bool(ev.get("released")) or actual is not None
        events.append({
            "title":    (ev.get("title") or "").strip(),
            "currency": currency,
            "dateISO":  date_iso,
            "timeUTC":  time_utc,
            "impact":   impact_raw,
            "forecast": forecast,
            "previous": previous,
            "actual":   actual,
            "released": released,
        })
    return events

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


# ── calendar.json enrichment ──────────────────────────────────────────────────
# ForexFactory JSON does not carry licensed actuals for Flash PMIs (EUR/GBP/AUD/JPY)
# or certain other events. This step reads calendar-data/calendar.json (the FRED+FH
# backfill) and fills in `actual` and `previous` for events that FF left null.
#
# Matching strategy: same currency + same date + fuzzy title keyword overlap (>=2 words).
# The backfill uses different provider names (e.g. "HCOB Manufacturing PMI Flash" vs
# "S&P Global Manufacturing PMI Flash") so exact matching fails — keyword overlap works.

_TITLE_IGNORE = frozenset({
    'flash','prelim','final','s&p','global','rate','index',
    'm/m','y/y','q/q',
    'of','the','a','and','or','for',
    'hcob','ism','cb','fed','ecb','boe','boj','rba','rbnz','snb','boc',
    'pct',
    # NOTE: 'pmi' intentionally NOT in ignore — it discriminates PMI events from
    # similar events like "Manufacturing Production", "Services Sentiment", etc.
    # NOTE: 'mom','yoy','qoq','prel','change','weekly' intentionally NOT in ignore —
    # they discriminate series variants (e.g. "Building Permits" vs "Building Permits MoM Prel",
    # "ADP Employment Change" vs "ADP Employment Change Weekly"). Removing them prevents
    # cross-series contamination in derive_forecast_from_history and derive_previous_from_history.
    'jibun','unicredit','markit','nab','westpac','bank',
})

# ── Keyword scoring constants (v3.12) ─────────────────────────────────────────
#
# SCORING MODEL:
#   1. Variant guard: (target_kw & _VARIANT_WORDS) must equal (history_kw & _VARIANT_WORDS).
#      Prevents cross-series contamination between rate-of-change types and frequencies:
#      "CPI MoM" vs "CPI YoY", "Building Permits MoM" vs "Building Permits" (absolute, 1000s),
#      "ADP Employment Change Weekly" vs "ADP Employment Change" (monthly).
#      NOTE: 'prel'/'flash' intentionally excluded — Prel vs Final of the same series are
#      interchangeable for forecast purposes.
#
#   2. Non-variant score: score = len(overlap - _VARIANT_WORDS).
#      Variant words do NOT count toward the match score. This prevents "Capital Spending YoY"
#      matching "Household Spending YoY" via {spending, yoy} — the non-variant overlap {spending}
#      is not in _STRONG_SINGLE_OK so the match is rejected.
#
#   3. Single non-variant word: only sufficient if that word is in _STRONG_SINGLE_OK.
#      Generic words (pmi, sales, retail, consumer, confidence, change, prel) appear in too many
#      unrelated events to justify a single-word match; they require score >= 2.
#        pmi        — Manufacturing PMI ≠ Services PMI ≠ Chicago PMI
#        sales      — Existing Home Sales ≠ Retail Sales ≠ New Home Sales
#        retail     — Retail Sales ≠ Retail Inventories
#        consumer   — Consumer Confidence ≠ Consumer Credit ≠ Consumer Spending
#        confidence — Consumer Confidence ≠ Business Confidence
#        change     — appears in hundreds of unrelated events (suffix, not a concept)
#        prel       — qualifier (preliminary), not an event-type word
#
# ── Three-tier scoring model (v3.12) ─────────────────────────────────────────
#
# TIER 1 — Variant guard (applied first, before scoring):
#   (target_kw & _VARIANT_WORDS) must equal (history_kw & _VARIANT_WORDS).
#   Prevents cross-series contamination between rate-of-change types/frequencies.
#   'prel'/'flash' intentionally excluded: Prel vs Final are the same underlying
#   release — their forecasts are interchangeable.
#
# TIER 2 — Non-variant score:
#   score = len(overlap - _VARIANT_WORDS)  (variant words excluded from count)
#   Prevents {spending, yoy} from scoring 2 — the yoy is a variant marker, not
#   a meaningful disambiguator between "Capital Spending" and "Household Spending".
#
# TIER 3 — Minimum-score thresholds:
#   nv_score == 0  → no match
#   nv_score == 1  → ONLY if that word is in _STRONG_SINGLE_OK
#                    (unemployment, payrolls, cpi, adp, permits, etc.)
#   nv_score == 2  → at least 1 word must be in _ANCHOR
#                    (_STRONG_SINGLE_OK | _SECTOR_ANCHOR)
#                    Blocks: {home, sales}, {crude, oil}, {autos, retail}, {change, eia}
#                    Passes: {manufacturing, pmi}, {household, spending}, {factory, orders}
#   nv_score >= 3  → always OK

# Words sufficient alone (nv_score=1) to justify a same-series match.
_STRONG_SINGLE_OK = frozenset({
    'unemployment', 'payrolls', 'cpi', 'ppi', 'pce', 'gdp',
    'housing', 'inflation', 'employment',
    'sentiment', 'permits', 'balance',
    'trade', 'michigan', 'adp', 'nfp', 'inventory', 'jobless', 'claims',
    'construction',
    # NOTE: 'chicago' moved to _SECTOR_ANCHOR — "Chicago PMI" passes via nv=2 {chicago,pmi},
    # but "Chicago Fed National Activity Index" ← "Chicago PMI" is now blocked (nv=1 via chicago only).
})

# Domain-specific words that anchor a score=2 match to a particular economic concept.
# Not strong enough alone (nv_score=1 would be too loose), but sufficient when paired
# with at least one other word from the same economic domain.
_SECTOR_ANCHOR = frozenset({
    'manufacturing', 'services', 'composite', 'industrial', 'production',
    'consumer', 'confidence', 'household', 'factory', 'orders', 'spending',
    'business', 'current', 'account',
    'chicago',   # "Chicago PMI" needs chicago+pmi (score=2 via anchor), not chicago alone
})

# For nv_score==2, at least 1 non-variant word must be in _ANCHOR.
_ANCHOR = _STRONG_SINGLE_OK | _SECTOR_ANCHOR

# _STRONG_WORDS: superset for backward compatibility (e.g. _TITLE_IGNORE removal decisions).
_STRONG_WORDS = _STRONG_SINGLE_OK | _SECTOR_ANCHOR | frozenset({
    'mom', 'yoy', 'qoq', 'prel', 'weekly', 'change',
    'pmi', 'sales', 'retail', 'consumer', 'confidence',
})

# Series-variant words: rate-of-change type and frequency discriminators.
_VARIANT_WORDS = frozenset({'mom', 'yoy', 'qoq', 'weekly'})

# Alias for the score=1 guard (= _STRONG_SINGLE_OK).
_STRONG_NON_VARIANT = _STRONG_SINGLE_OK

def _title_keywords(title: str) -> frozenset:
    # Normalise ForexFactory slash-notation variants to the canonical _VARIANT_WORDS
    # used by the scoring model. FF uses "m/m", "y/y", "q/q" — these become single
    # characters after the '/' replace and get dropped by the len>2 filter, causing
    # "CPI m/m" to reduce to {cpi} (same as "CPI" the index level) and defeating
    # the variant guard that blocks CPI level (335.12) from filling CPI MoM (%).
    t = title.lower()
    t = t.replace('m/m', 'mom').replace('y/y', 'yoy').replace('q/q', 'qoq')
    words = t.replace('/', ' ').replace('-', ' ').split()
    return frozenset(w for w in words if w not in _TITLE_IGNORE and len(w) > 2)


def enrich_from_calendar_json(events: list[dict]) -> int:
    """
    Retained as a no-op stub — calendar.json (FRED + Finnhub backfill) is no
    longer used as an enrichment source.  ForexFactory is the sole data provider;
    historical actuals/previous values are carried by the ff_calendar.json rolling
    accumulation (seeded by backfill_ff_calendar.py / FMP one-shot).

    Removing the cross-source enrichment eliminates the naming-mismatch bugs
    where FRED absolute index values (e.g. CPI level = 335.12) contaminated
    ForexFactory percentage actuals (CPI m/m = 0.2%) via fuzzy title matching.

    v3.19 (2026-06-10): Converted to stub — no calendar.json read.
    """
    print("  Enrichment from calendar.json: skipped (FF-only pipeline)")
    return 0


# ── Forecast derivation from historical series ───────────────────────────────
# Industry standard (Bloomberg/Reuters Eikon): for events that structurally carry
# analyst consensus (e.g. ADP, MBA Mortgage Rate, CBI Distributive Trades), when
# the current run has no estimate from the provider, fill with the *most recent
# prior forecast* for that event series — this is called "last known consensus"
# or "stale forecast fallback".
#
# Four critical safeguards (matching industry practice, v3.13):
#   1. Released-event guard: only upcoming events are filled. Bloomberg/Reuters
#      only show estimate vs actual when the estimate was published before the
#      release — filling post-release implies a false miss.
#   2. Consensus eligibility gate (MIN_FORECAST_HISTORY=2, ELIGIBILITY_WINDOW_DAYS=180):
#      only fills series that had a forecast in >= 2 of the last 180 days. The recency
#      window prevents filling events whose API coverage has lapsed (stale FRED-era
#      coverage only). 180 days accommodates quarterly series (GDP, corporate profits).
#   3. The derived value is suffixed with "*" so downstream consumers can render it
#      differently (e.g. "0.3%*" vs "0.3%" from the live feed). The calendar panel
#      uses this to show a muted style distinguishing derived from live forecasts.
#   4. Z-score magnitude guard (Z_SCORE_THRESHOLD=3.0, Z_SCORE_MIN_ACTUALS=3):
#      derived forecast must be within 3σ of recent Finnhub actuals for the same
#      series. Catches unit-mismatch scenarios where FRED stored values in a
#      different scale than Finnhub (e.g. "6000" vs "220" for ADP Employment Change).
#      Only fires when >= Z_SCORE_MIN_ACTUALS Finnhub actuals exist for the series.
#
# NOT applied to:
#   - Events already having a forecast (live value takes priority)
#   - Events that are already released (actual != None)
#   - Events not yet confirmed to have a consensus history (eligibility gate)
#   - Events matching known no-consensus patterns (speeches, testimonies)

MIN_FORECAST_HISTORY     = 2    # need at least 2 prior forecasts in recency window to be eligible
ELIGIBILITY_WINDOW_DAYS  = 180  # recency window for eligibility count — accommodates quarterly series
Z_SCORE_THRESHOLD        = 3.0  # suppress derived forecast if |z| > this vs Finnhub actuals
Z_SCORE_MIN_ACTUALS      = 3    # minimum Finnhub actuals needed to activate z-score guard
_NO_CONSENSUS_WORDS  = frozenset({
    "speech", "speaks", "testimony", "testifies", "statement", "press",
    "conference", "minutes", "vote", "meeting", "forum", "symposium",
    "appearance", "hearing", "panel",
})

def derive_forecast_from_history(events: list[dict]) -> int:
    """
    Fill missing `forecast` fields using the most recent prior forecast for the
    same event series (last-known-consensus fallback). Derived values are suffixed
    with "*" to indicate they are estimated, not live from the provider.

    Guards (v3.13 industry-standard audit):
      1. Released-event guard: skips events that are already released
         (actual != None or released == True). Only upcoming events get derived
         forecasts — filling post-release is misleading (implied miss without real estimate).
      2. Eligibility recency window (ELIGIBILITY_WINDOW_DAYS=180): only counts
         prior forecasts within 180 days toward the eligibility threshold. Prevents
         series with stale historical coverage from passing the eligibility gate.
      3. Z-score magnitude guard (Z_SCORE_THRESHOLD=3.0): derived forecast must be
         within 3σ of recent Finnhub actuals for the same series, or it is suppressed.
         Catches unit-mismatch scenarios (e.g. FRED-unit values vs Finnhub-unit values).

    Returns count of events updated.
    """
    import math
    from collections import defaultdict

    _COMMODITY_WORDS = frozenset({'gasoline', 'crude', 'natural', 'gas', 'distillate', 'heating'})

    # Compute cutoff date for eligibility recency window
    now_utc = datetime.now(timezone.utc)
    eligibility_cutoff = (now_utc - timedelta(days=ELIGIBILITY_WINDOW_DAYS)).strftime("%Y-%m-%d")

    # ── Build forecast history from ff_calendar batch (FF-only pipeline) ───────
    # v3.19: calendar.json (FRED + Finnhub) removed as history source.
    # Forecast history now comes entirely from the current ff_calendar.json events
    # list (fresh + merged prev_events).  The FMP 90-day backfill (backfill_ff_calendar.py)
    # seeds this history so the eligibility gate fires correctly from day one.
    ff_fc_history: list[tuple] = []
    for ev in events:
        raw_fc = ev.get("forecast")
        if raw_fc is None:
            continue
        fc_str = str(raw_fc).strip()
        if not fc_str or fc_str in ("—", "-", "N/A", "--"):
            continue
        # Strip existing "*" suffix to avoid double-marking re-used derived values
        fc_str = fc_str.rstrip("*")
        kw = _title_keywords(ev.get("title") or "")
        if kw:
            ff_fc_history.append((
                ev.get("dateISO", ""),
                ev.get("currency", ""),
                ev.get("title", ""),
                fc_str,
                kw,
            ))

    combined = ff_fc_history

    # ── Build actual history from ff_calendar batch ───────────────────────────
    # Used by z-score guard to validate candidate forecast magnitudes.
    ff_actual_history: list[tuple] = []  # (dateISO, currency, actual_float, kw_frozenset)
    for ev in events:
        raw_act = ev.get("actual")
        if raw_act is None:
            continue
        act_str = str(raw_act).strip().rstrip("*")
        try:
            act_float = float(act_str)
            kw = _title_keywords(ev.get("title") or "")
            if kw:
                ff_actual_history.append((
                    ev.get("dateISO", ""),
                    ev.get("currency", ""),
                    act_float,
                    kw,
                ))
        except (ValueError, AttributeError):
            pass  # non-numeric actual — skip for z-score purposes

    # Group by currency for fast lookup
    by_ccy: dict[str, list] = defaultdict(list)
    for row in combined:
        by_ccy[row[1]].append(row)
    for ccy in by_ccy:
        by_ccy[ccy].sort(key=lambda x: x[0], reverse=True)

    by_ccy_actuals: dict[str, list] = defaultdict(list)
    for row in ff_actual_history:
        by_ccy_actuals[row[1]].append(row)
    for ccy in by_ccy_actuals:
        by_ccy_actuals[ccy].sort(key=lambda x: x[0], reverse=True)

    derived = 0
    suppressed_released = 0
    suppressed_zscore = 0

    for ev in events:
        # Skip if already has a forecast (live value takes priority)
        if ev.get("forecast") is not None:
            continue

        # ── Guard 1: Released-event guard (v3.13) ────────────────────────────
        # Only derive forecasts for upcoming/unreleased events.
        # Bloomberg/Reuters only show estimate vs actual when the estimate was
        # published before release — filling post-release implies a false miss.
        if ev.get("released") or ev.get("actual") is not None:
            suppressed_released += 1
            continue

        title = ev.get("title") or ""
        ff_kw = _title_keywords(title)
        if not ff_kw:
            continue

        # No-consensus guard: skip known non-forecastable event types
        title_lower = title.lower()
        if any(w in title_lower for w in _NO_CONSENSUS_WORDS):
            continue

        ev_date = ev.get("dateISO", "")
        ev_ccy  = ev.get("currency", "")
        ev_commodities = ff_kw & _COMMODITY_WORDS
        ev_variants = ff_kw & _VARIANT_WORDS

        # ── Guard 2: Consensus eligibility gate (with recency window, v3.13) ──
        # Count how many prior forecasts exist within ELIGIBILITY_WINDOW_DAYS.
        # Only proceed if >= MIN_FORECAST_HISTORY — this prevents filling events
        # that structurally never carry consensus (speeches, testimonies) AND
        # events whose API coverage has lapsed (stale FRED-era coverage only).
        eligible_count = 0
        for h_date, h_ccy, _, _, h_kw in by_ccy.get(ev_ccy, []):
            if h_date >= ev_date:
                continue
            # Recency window: only count forecasts within ELIGIBILITY_WINDOW_DAYS
            if h_date < eligibility_cutoff:
                continue
            # Variant guard
            if (h_kw & _VARIANT_WORDS) != ev_variants:
                continue
            # Commodity guard
            if ev_commodities and not (ev_commodities & h_kw):
                continue
            overlap = ff_kw & h_kw
            score = len(overlap - _VARIANT_WORDS)
            if score == 1 and not (overlap & _STRONG_NON_VARIANT):
                score = 0
            elif score == 2 and not (overlap & _ANCHOR):
                score = 0
            if score >= 1:
                eligible_count += 1
            if eligible_count >= MIN_FORECAST_HISTORY:
                break

        if eligible_count < MIN_FORECAST_HISTORY:
            continue  # series not eligible

        # ── Find most recent prior forecast ───────────────────────────────────
        best_forecast: str | None = None
        best_score = 0
        best_match_kw: frozenset | None = None

        for h_date, h_ccy, _, h_fc, h_kw in by_ccy.get(ev_ccy, []):
            if h_date >= ev_date:
                continue
            if (h_kw & _VARIANT_WORDS) != ev_variants:
                continue
            if ev_commodities and not (ev_commodities & h_kw):
                continue
            overlap = ff_kw & h_kw
            score = len(overlap - _VARIANT_WORDS)
            if score == 1 and not (overlap & _STRONG_NON_VARIANT):
                score = 0
            elif score == 2 and not (overlap & _ANCHOR):
                score = 0
            if score > best_score:
                best_score = score
                best_forecast = h_fc
                best_match_kw = h_kw
            if best_score >= 2:
                break  # most recent high-quality match found

        if best_forecast is None or best_score < 1:
            continue

        # ── Guard 3: Z-score magnitude guard (v3.13) ─────────────────────────
        # Validate the candidate forecast against recent Finnhub actuals for the
        # same matched series. Suppresses unit-mismatch outliers (e.g. FRED "6000"
        # vs Finnhub "220" for ADP Employment Change).
        # Only activates when >= Z_SCORE_MIN_ACTUALS Finnhub actuals exist.
        # Non-numeric candidates pass through (no numeric check possible).
        fc_numeric: float | None = None
        try:
            fc_numeric = float(best_forecast.rstrip("*%KkMmBb").replace(",", ""))
        except (ValueError, AttributeError):
            pass  # non-numeric forecast — skip z-score check

        if fc_numeric is not None:
            # Collect Finnhub actuals for the matched series
            matched_actuals: list[float] = []
            for a_date, a_ccy, a_val, a_kw in by_ccy_actuals.get(ev_ccy, []):
                if a_date >= ev_date:
                    continue
                if best_match_kw is not None:
                    overlap_a = ff_kw & a_kw
                    score_a = len(overlap_a - _VARIANT_WORDS)
                    if score_a == 1 and not (overlap_a & _STRONG_NON_VARIANT):
                        score_a = 0
                    elif score_a == 2 and not (overlap_a & _ANCHOR):
                        score_a = 0
                    if score_a < 1:
                        continue
                matched_actuals.append(a_val)
                if len(matched_actuals) >= 8:  # use up to 8 most recent Finnhub actuals
                    break

            if len(matched_actuals) >= Z_SCORE_MIN_ACTUALS:
                n = len(matched_actuals)
                mean_a = sum(matched_actuals) / n
                variance = sum((x - mean_a) ** 2 for x in matched_actuals) / (n - 1)
                std_a = math.sqrt(variance) if variance > 0 else 0
                if std_a > 1e-9:
                    z = abs(fc_numeric - mean_a) / std_a
                    if z > Z_SCORE_THRESHOLD:
                        suppressed_zscore += 1
                        continue  # suppress — outlier vs Finnhub actuals

        # Suffix with "*" — industry convention for "last known consensus / estimated"
        ev["forecast"] = best_forecast.rstrip("*") + "*"
        derived += 1

    if suppressed_released:
        pass  # intentionally not printed — released events are expected to be skipped silently
    if suppressed_zscore:
        print(f"  Forecast z-score guard: {suppressed_zscore} outlier candidate(s) suppressed")
    print(f"  Forecast derived from history: {derived} events updated (suffixed '*' = last known consensus)")
    return derived


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
    series. Uses the current ff_calendar events list (fresh + merged prev_events).
    v3.19: calendar.json (FRED + Finnhub) removed as history source — FF-only pipeline.
    Returns count of events updated.
    """
    from collections import defaultdict

    # Build history from ff_calendar batch only (v3.19: calendar.json removed)
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

    combined = ff_history

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
        ev_variants = ff_kw & _VARIANT_WORDS   # rate-of-change type for this event

        best_actual: str | None = None
        best_score  = 0

        for h_date, h_ccy, h_event, h_actual, h_kw in by_ccy.get(ev_ccy, []):
            if h_date >= ev_date:
                continue  # must be strictly before this event
            # Commodity guard: if the target has a commodity word (e.g. "gasoline"),
            # the candidate must also contain that word — prevents crude→gasoline matches
            if ev_commodities and not (ev_commodities & h_kw):
                continue
            # Variant guard: reject cross-series matches (MoM vs absolute, weekly vs monthly)
            if (h_kw & _VARIANT_WORDS) != ev_variants:
                continue
            overlap = ff_kw & h_kw
            score   = len(overlap - _VARIANT_WORDS)   # only non-variant words count
            if score == 1 and not (overlap & _STRONG_NON_VARIANT):
                score = 0
            elif score == 2 and not (overlap & _ANCHOR):
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
    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] fetch_ff_calendar.py v3.26")

    date_from = (now_utc - timedelta(days=FETCH_PAST_DAYS)).strftime("%Y-%m-%d")
    date_to   = (now_utc + timedelta(days=FETCH_FUTURE_DAYS)).strftime("%Y-%m-%d")

    # Step 1: Fetch Myfxbook HTML calendar page.
    #
    # URL: https://www.myfxbook.com/forex-economic-calendar
    # Accessible from GitHub Actions (confirmed in v3.23: no WAF, full HTML returned).
    # Covers current week + ~1 week ahead. Actuals are permanently retained in HTML
    # attributes (data-actual OID, span.actualCell) once an event releases — no rolling
    # expiry unlike the RSS feed which only covers ~24h and drops actuals after the fact.
    # fetch_myfxbook_calendar() parses the full page including holidays.
    fresh    = []
    holidays = []
    source   = "Myfxbook"

    MFB_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    print(f"  Myfxbook HTML: fetching {MFB_HTML_URL} ...")
    try:
        r = requests.get(MFB_HTML_URL, headers=MFB_HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            print(f"  ERROR: Myfxbook HTML HTTP {r.status_code} — preserving previous file.")
            sys.exit(0)
        if "economicCalendarRow" not in r.text:
            print("  ERROR: Myfxbook HTML missing calendar rows — unexpected response or WAF block.")
            sys.exit(0)
        print(f"  Myfxbook HTML: {len(r.text):,} bytes received")
        fresh, holidays = fetch_myfxbook_calendar(r.text)
    except Exception as e:
        print(f"  ERROR: Myfxbook HTML request failed — {e}")
        sys.exit(0)

    # Step 1a: Validate
    if not fresh:
        print("  ERROR: No G8 medium/high events parsed from any source — preserving previous file.")
        sys.exit(0)

    released_fresh = sum(1 for e in fresh if e.get("released"))
    print(f"  Fetched: {len(fresh)} events ({released_fresh} with actuals) [source: {source}]")

    # Step 2: Historical merge — preserve events from prev_events not covered by the fresh fetch.
    # With Finnhub (old), fresh covered a 21-day window so the guard `d >= date_from` correctly
    # excluded events already refreshed. With ForexFactory public JSON (v3.17+), fresh covers
    # only the current calendar week — so we must merge ALL prev_events within the lookback
    # window that are not already in fresh_keys (keyed on currency+dateISO+timeUTC+title).
    # The dedup step (Step 2d) downstream handles any true duplicates.
    cutoff = (now_utc - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    prev_events, prev_fingerprint = load_previous()
    fresh_keys = {(e["currency"], e["dateISO"], e["timeUTC"], e["title"]) for e in fresh}
    merged = 0
    for ev in prev_events:
        d = ev.get("dateISO", "")
        if d < cutoff:
            continue   # outside 21-day lookback window — drop
        k = (ev.get("currency",""), d, ev.get("timeUTC",""), ev.get("title",""))
        if k not in fresh_keys:
            fresh.append(ev)
            fresh_keys.add(k)
            merged += 1
    print(f"  Merged: {merged} historical events from previous file (within {LOOKBACK_DAYS}-day window)")

    # Step 2a: Carry-forward actuals from prev_events for events that are IN fresh but
    # where fresh.actual is None. FF JSON occasionally returns actual="" for an event that
    # was successfully fetched in a prior run (API lag, transient miss, or week rollover).
    # Without this guard, a stale FF response silently erases actuals that are already known.
    # Rule: if fresh.actual is None AND prev had actual=X for the same event key, carry X forward.
    prev_actual_map = {
        (ev.get("currency",""), ev.get("dateISO",""), ev.get("timeUTC",""), ev.get("title","")): ev
        for ev in prev_events
        if ev.get("actual") is not None
    }
    carried = 0
    for ev in fresh:
        if ev.get("actual") is not None:
            continue   # fresh already has an actual — no carry needed
        k = (ev.get("currency",""), ev.get("dateISO",""), ev.get("timeUTC",""), ev.get("title",""))
        if k in prev_actual_map:
            ev["actual"]   = prev_actual_map[k]["actual"]
            ev["released"] = True
            carried += 1
    if carried:
        print(f"  Carried forward: {carried} actual(s) from previous file (FF JSON lag guard)")

    # Step 2b: Enrich from calendar.json backfill (fills actuals/previous Finnhub can't provide)
    enrich_from_calendar_json(fresh)

    # Step 2c: Derive missing `previous` from historical series
    # Covers Flash PMIs (EUR/GBP/AUD/JPY), energy inventories, jobless claims, etc.
    # where Finnhub returns previous=null but prior actuals exist in the combined history.
    derive_previous_from_history(fresh)

    # Step 2e: Derive missing `forecast` from historical series (last-known-consensus fallback)
    # Industry standard (Bloomberg/Reuters Eikon): for events that structurally carry analyst
    # consensus but whose current-run estimate is null (timing gap, API tier, provider lag),
    # fill with the most recent prior forecast for that series. Derived values are suffixed
    # with "*" so the frontend renders them in a muted style distinct from live estimates.
    # The MIN_FORECAST_HISTORY gate ensures speech/testimony events (no consensus) are skipped.
    derive_forecast_from_history(fresh)

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

    # Step 2e: Cross-day dedup — Finnhub sometimes re-emits a weekly event as an "upcoming"
    # entry 1-7 days after the actual release, with the same timeUTC. This creates phantom
    # duplicates like "Westpac Jun 9 (actual=-2.9)" + "Westpac Jun 10 (no actual, same time)".
    # Strategy: for each unreleased event, check if a released copy exists within the prior
    # 7 days with the same (title, currency, timeUTC). If so, drop the unreleased phantom.
    # Build lookup: (title, currency, timeUTC) -> set of dateISO that have actuals
    released_index: dict = {}
    for ev in fresh:
        if ev.get("actual") is not None:
            key = (ev["title"], ev["currency"], ev.get("timeUTC", ""))
            if key not in released_index:
                released_index[key] = []
            released_index[key].append(ev["dateISO"])
    cross_day_removed = 0
    cross_day_deduped: list = []
    for ev in fresh:
        if ev.get("actual") is not None or ev.get("released"):
            cross_day_deduped.append(ev)
            continue
        key = (ev["title"], ev["currency"], ev.get("timeUTC", ""))
        prior_dates = released_index.get(key, [])
        ev_date = datetime.strptime(ev["dateISO"], "%Y-%m-%d").date()
        is_phantom = any(
            0 < (ev_date - datetime.strptime(d, "%Y-%m-%d").date()).days <= 7
            for d in prior_dates
        )
        if is_phantom:
            cross_day_removed += 1
        else:
            cross_day_deduped.append(ev)
    if cross_day_removed:
        print(f"  Cross-day dedup: removed {cross_day_removed} phantom upcoming entries (released copy exists within 7d)")
    fresh = cross_day_deduped

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

    # Also detect holiday changes (new or removed holidays vs previous file)
    prev_holidays_fp: set[tuple] = set()
    try:
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                prev_output = json.load(f)
            for h in prev_output.get("holidays", []):
                prev_holidays_fp.add((h.get("currency",""), h.get("dateISO",""), h.get("title","")))
    except Exception:
        pass
    new_holidays_fp = {(h["currency"], h["dateISO"], h["title"]) for h in holidays}
    holidays_changed = new_holidays_fp != prev_holidays_fp

    if holidays_changed:
        added_h   = new_holidays_fp - prev_holidays_fp
        removed_h = prev_holidays_fp - new_holidays_fp
        if added_h:
            print(f"  NEW holidays: {', '.join(f'{c} {d}' for c,d,_ in sorted(added_h))}")
        if removed_h:
            print(f"  REMOVED holidays: {', '.join(f'{c} {d}' for c,d,_ in sorted(removed_h))}")

    # Detect structural changes: phantom upcoming entries removed by cross-day dedup
    # (or same-day dedup) that existed in the previous file but are absent in the new one.
    # These are unreleased events dropped from the panel — the file should be rewritten
    # so the client no longer serves the phantom rows. Without this check, a run that
    # only removes phantoms (no new actuals/forecasts) would compute data_changed=False
    # and skip the write, leaving the stale phantoms in ff_calendar.json until the next
    # run that happens to have a real actual/forecast change.
    prev_upcoming_fp = {
        (e["title"], e["currency"], e["dateISO"], e.get("timeUTC",""))
        for e in prev_events
        if not e.get("released") and e.get("actual") is None
    }
    new_upcoming_fp = {
        (e["title"], e["currency"], e["dateISO"], e.get("timeUTC",""))
        for e in fresh
        if not e.get("released") and e.get("actual") is None
    }
    phantoms_removed = prev_upcoming_fp - new_upcoming_fp
    structural_changed = bool(phantoms_removed)
    if structural_changed:
        print(f"  STRUCTURAL: {len(phantoms_removed)} phantom upcoming entries removed by dedup")

    data_changed = bool(new_actuals_added or new_forecasts_added or holidays_changed or structural_changed)

    if data_changed:
        if new_actuals_added:
            print(f"\n  NEW actuals: {len(new_actuals_added)} event(s) updated")
            for t, ccy, d, act, fc in sorted(new_actuals_added, key=lambda x: x[2]):
                print(f"    {d} [{ccy}] {t[:50]} → actual={act}" + (f" (fc was {fc})" if fc else ""))
        if new_forecasts_added:
            print(f"  NEW forecasts: {len(new_forecasts_added)} event(s) updated")
    else:
        print("\n  No new actuals, forecasts, or holiday changes — skipping write and commit.")
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
        "holidays": holidays,
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
