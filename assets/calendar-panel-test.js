/**
 * calendar-panel.js v1.12 — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (ForexFactory, G10 currencies, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 *
 * v1.1 (2026-06-09): Display window filter — show only yesterday through +14 days.
 * v1.2 (2026-06-09): Client-side cross-day dedup — mirrors Step 2e of fetch_ff_calendar.py
 *   so phantom upcoming entries are removed immediately, even from stale cached JSON.
 *   ff_calendar.json carries 21 days of actuals history for backfill purposes; without
 *   a display cutoff the panel rendered 3 weeks of past events above today. Now clamped
 *   to yesterday–today+14 so the panel stays focused on current and upcoming events.
 * v1.3 (2026-06-10): Reduced poll interval from 5 min to 2 min. The CF Worker + GitHub
 *   Actions pipeline delivers updated ff_calendar.json within ~2 min of a ForexFactory actual
 *   publishing. The previous 5-min client poll added up to 3 min of unnecessary lag on top
 *   of the pipeline latency. At 2 min the worst-case end-to-end delay is ~4 min; best-case
 *   (visibilitychange fires on tab focus) is near-instant. Cache-bust in index.html bumped
 *   to v=1.3.0 so all browsers discard the previously cached v1.0.0 file immediately.
 * v1.4 (2026-06-10): Source label corrected from 'Finnhub' to 'ForexFactory'. The calendar
 *   data has always been sourced from ForexFactory (ff_calendar.json via fetch_ff_calendar.py);
 *   the Finnhub label was a stale reference from the original CF Worker implementation.
 * v1.5 (2026-06-15): Display window extended from yesterday to 3 days back. Industry standard
 *   (Bloomberg, Refinitiv Eikon) shows 2–3 prior sessions alongside current day. Also ensures
 *   Friday sessions remain visible on Monday morning and covers overnight JPY/AUD releases.
 * v1.6 (2026-07-15): Cache-bust the data fetch itself. `fetchEconomicCalendar()` used
 *   `cache: 'no-store'` (browser-cache-only) with a static URL on every poll — GitHub Pages'
 *   CDN (Fastly) can hold an edge copy of that exact URL for several minutes regardless of the
 *   browser's own cache directive, so a workflow run that updated ff_calendar.json wasn't
 *   reflected in the panel until the CDN's TTL expired, well past the 2-min poll interval.
 *   Found live 2026-07-15: workflow run committed 4 new actuals, panel still showed "—" 23min
 *   later. Now appends a minute-bucketed cache-buster (mirrors the existing pattern on
 *   ./intraday-data/quotes.json) so each poll hits a URL the CDN hasn't served before.
 * v1.7 (2026-08-01): Added a fullscreen toggle button (#cal-fs-btn) matching the Price
 *   Chart's existing fullscreen pattern (#lw-fs-btn in dashboard.js). Same DOM-lift
 *   approach — #section-tvcalendar is moved into #cal-fullscreen-overlay on open and
 *   restored to its original position on close — but without any chart-resize logic,
 *   since this panel is a plain scrollable list. The 330px inline max-height on
 *   #cal-events-body is overridden via the .cal-fs-active CSS rule in index.html so the
 *   full viewport height is used while fullscreen.
 * v1.8 (2026-08-04): BUG FIX — Actual-column coloring never accounted for inverse
 *   indicators (Unemployment Rate, Jobless Claims, deficit-type levels), where a higher
 *   actual than forecast is a negative surprise. The naive `actualN > forecastN ? up :
 *   down` comparison painted any numerically larger actual green, even when it meant
 *   worse economic news. Found live: NZD Unemployment Rate (Q2) printed 5.6% vs. a 5.4%
 *   forecast/previous — a negative surprise (unemployment rising) — and rendered green.
 *   Added CAL_INVERSE_KW (mirrors INVERSE_KW in dashboard.js, _ESM_INVERSE_KW in
 *   econ-surprises-modal.js, INVERSE_EVENTS in fetch_economic_calendar.py, which this
 *   file had never implemented) and sign-correct the beat/miss check for matching
 *   titles before assigning the up/down class.
 * v1.9 (2026-08-04): Audited all 4 inverse-keyword lists against the full year of G10
 *   events already in calendar-data/calendar.json (690 unique titles) instead of a
 *   fresh manual export. Found one substring gap: "unemployment" doesn't match
 *   "Unemployed Persons" (EUR/Germany monthly, NOK) — 15 real occurrences over the
 *   past year, all mis-colored the same way as the NZD case above. Added "unemployed"
 *   to CAL_INVERSE_KW (and the three sibling lists). No other gaps found in the
 *   dataset — checked for bankruptcies/redundancies/layoffs/defaults/delinquencies
 *   (none appear in the G10 title set) and confirmed diffusion-style indices (Ai
 *   Group Industry/Manufacturing/Construction Index) are correctly non-inverse.
 * v1.10 (2026-08-07): BUG FIX — the panel subtitle rendered the raw `source` field
 *   from ff_calendar.json verbatim, which can legitimately carry backend/pipeline
 *   detail for troubleshooting (e.g. calendar-watcher.js's direct-commit fallback
 *   label "Myfxbook · ForexFactory (CF Worker direct-commit fallback — GitHub
 *   Actions unavailable)"). That detail is useful in the raw JSON — it's how the
 *   2026-08-06/07 history-truncation incident was diagnosed — but it has no
 *   business appearing in the terminal UI; Bloomberg/Refinitiv don't expose their
 *   data-delivery mechanics to the user, only the data provider itself. New
 *   `cleanSourceLabel()` strips any trailing parenthetical before display,
 *   handling this case and any future one following the same "Label (pipeline
 *   detail)" convention used elsewhere in the Worker (e.g. quotes.json's
 *   DIRECT_COMMIT_SOURCE_LABEL). Found live from Santiago's screenshot.
 * v1.11 (2026-08-07): TWO BUG FIXES, both surfaced by the same incident.
 *   (1) Duplicate timezone label: the panel subtitle already ends in
 *   `tzLabel()` (e.g. "· GMT-3") AND the column-header row's time column
 *   (#cal-th-time) shows the same `tzLabel()` directly below it — Santiago
 *   flagged this as redundant on screen. Removed the trailing tzLabel() from
 *   the subtitle; the column header is the correct single place for it since
 *   it labels what the time column itself means.
 *   (2) Missing historical events: `fetchEconomicCalendar()`'s source-fallback
 *   loop picked ff_calendar.json whenever it had ANY events and never checked
 *   whether that data actually carried history — so the 2026-08-06/07
 *   truncation incident (ff_calendar.json collapsed to a single day) silently
 *   won the fallback race forever, even though calendar.json still had a full
 *   year of history sitting right there. ff_calendar.json can never self-heal
 *   this on its own (its own Step 2 merge reads its own prior content — see
 *   calendar-watcher.js v5.27 CHANGELOG entry), so a client-side guard is the
 *   only thing that stops a repeat of this from going unnoticed again.
 *   fetchEconomicCalendar() now fetches both files, and if ff_calendar.json
 *   covers fewer than 2 distinct past dates, fills in calendar.json's older
 *   events (deduped by currency+date+time+title) instead of dropping them.
 *   Events are also normalized to always have `.title` (calendar.json's
 *   native schema uses `.event`, not `.title` — previously only the dedup
 *   filters guarded against this with `ev.title || ev.event`, but the actual
 *   row renderer (buildPanel) read `ev.title` unguarded, so a calendar.json
 *   fallback would have rendered blank event names even after fix (1) above).
 * v1.16-TEST (2026-08-08): "Implement everything, industry-standard" round —
 *   Santiago asked for all viable items from the v1.15-TEST idea list, with
 *   anything cramped for row space moved into the new click-through history
 *   modal rather than another inline badge. Implemented 6 of 7:
 *   (1) Historical reaction per pair — reference-pair (per CAL_REF_PAIR)
 *       avg daily OHLC range on this series' past release days vs. its
 *       typical day, surfaced in the history modal with an explicit
 *       daily-bar-proxy caveat (no intraday post-release timestamp data
 *       exists in this project's fetched sources — stated as a real gap,
 *       not silently approximated as more precise than it is).
 *   (2) Surprise history drill-down — click any event title to open a
 *       modal (openHistModal()) with the methodology blurb, cadence tag,
 *       FOMC voter tag when relevant, and the last up to 8 actual/forecast
 *       prints from a new full-year series index (buildSeriesIndex(),
 *       sourced from calendar.json's ~3720-event/year history — NOT the
 *       ~21-day ff_calendar.json window used for the main row list).
 *   (3) FOMC voting-member tag — small "V"/"nv" superscript next to Fed
 *       speaker names only (_fomcVoterTag()); scoped to the Fed because
 *       it's the only G10 central bank in this calendar with a structural
 *       voting/non-voting split. Dated 2026-rotation snapshot, documented
 *       inline with the source and a re-verify-in-January note.
 *   (4) High-impact-only filter — second, independent toggle alongside
 *       the currency isolate (passesImpactFilter(), #cal-impact-filter),
 *       persisted the same way via localStorage.
 *   (5) Cadence tag ("Weekly"/"Monthly"/etc.) — data-driven from the
 *       actual gap variance between a series' own past release dates
 *       (inferCadence()), not a maintained keyword list, per the
 *       already-documented drift risk with keyword lists in this codebase.
 *       Needs ≥3 prior releases and low gap variance or shows nothing.
 *   (6) Week navigation — Prev/Next shift the whole -3d/+14d window by
 *       ±7 days (_calWeekOffsetDays, #cal-week-nav); not persisted, same
 *       "always resets to now" convention as a real terminal's paging.
 *       Live-countdown highlight and the empty-window ForexFactory-outage
 *       fallback are scoped to stop applying once paged away from the
 *       real current window (offset 0) — neither means anything otherwise.
 *   SKIPPED: consensus range (Surv(H)/Surv(L) + contributor count) — this
 *       project's calendar schema (ff_calendar.json / calendar.json) only
 *       ever carries a single point forecast, never a survey distribution;
 *       no available data source provides one, so implementing it would
 *       mean fabricating a range, which the project's data-integrity rules
 *       (GUIDELINES.md — no invented/estimated data without labeling as
 *       such, and no source exists here to label it against) rule out.
 *   BUGFIX during this pass: the prior edit session ended before
 *       setupImpactFilterUI()/setupWeekNavUI() were actually written (only
 *       their call sites landed) and before fetchEconomicCalendar() was
 *       wired to populate _lastFullHistory/_seriesIndex from calendar.json
 *       — both would have thrown/rendered empty on first load. Added here.
 * v1.15-TEST (2026-08-08): Follow-up per Santiago's review of v1.14-TEST:
 *   (1) REMOVED the ESI contribution badge entirely — Santiago judged it
 *       added more visual noise than value on the row. Deleted
 *       esiContribBadge(), _calCanonEsi(), _CAL_CCY_PFXS, CAL_ESI_NOISE_KW,
 *       CAL_ESI_DECAY_LAMBDA, _lastSurpriseStats, and the surpriseStats
 *       fetch/store step in fetchEconomicCalendar() — nothing else in this
 *       file read that field. The live-countdown highlight and methodology
 *       tooltip from v1.14-TEST are unaffected and unchanged.
 *   (2) Synthetic live-countdown fixture: real data rarely has a qualifying
 *       high-impact event sitting inside the countdown window at the exact
 *       moment someone opens the sandbox to look at it, so testing the
 *       feature meant waiting for a real release or scripting a one-off
 *       fixture in a throwaway test harness. Added an opt-in, in-page
 *       fixture instead — append `?calDebugLive=1` to index-test.html's URL
 *       and a clearly-labeled "[TEST FIXTURE] Non-Farm Payrolls" event is
 *       injected 20 minutes out, seeded once per page load so it counts
 *       down in real time and crosses from the "soon" tier into the
 *       pulsing "imminent" tier ~5 minutes after load — same behavior a
 *       real event would show. No-op with the flag absent; never touches
 *       any fetched JSON. See getSyntheticLiveEvent() / calDebugLiveEnabled().
 * v1.14-TEST (2026-08-08): SANDBOX — Two "medium effort" enhancements from
 *   Santiago's original Bloomberg/Refinitiv gap-analysis, built on top of the
 *   v1.13.x currency-filter work (still unshipped to production). [A third,
 *   an ESI contribution badge, shipped in this version too but was removed
 *   in v1.15-TEST — see above; left out of this list accordingly.]
 *   (1) Live/next-release highlight: the single soonest unreleased
 *       high-impact event due within the next 3h gets a highlighted row and
 *       its clock time is swapped for a live countdown (ticks every 20s,
 *       independent of the 2-min data poll); inside 15m the row switches to
 *       a stronger pulsing tier. Tooltip on the countdown still shows the
 *       actual local time. Scoped to `filtered`, so it respects whatever
 *       currency is isolated.
 *   (2) Event methodology tooltip: hovering a matched event title (dashed
 *       underline cue, same visual convention as the ATM IV tooltips
 *       Santiago referenced) shows what it measures and why FX desks watch
 *       it. ~25 G10 headline-release patterns; unmatched titles keep the
 *       plain native tooltip that was already there. Self-contained tooltip
 *       widget (own #cal-tt id) rather than reusing dashboard.js's
 *       attachRiskTip, since this sandbox harness doesn't load dashboard.js
 *       — delegated listeners bound once on #cal-events-body, not per-row,
 *       so re-renders never re-attach or leak handlers.
 *   Both verified via the same jsdom + Chromium smoke-test harness used
 *   for the v1.13.x rounds (docked / narrow-fullscreen / wide-fullscreen-
 *   split), plus a synthetic near-term high-impact fixture event to exercise
 *   the live-countdown path (production data rarely has one sitting exactly
 *   inside the 3h/15m windows at any given moment the harness happens to run;
 *   this fixture only lived in the ad-hoc test harness at the time — v1.15-TEST
 *   above makes an equivalent fixture a permanent, opt-in part of the sandbox).
 * v1.13.3-TEST (2026-08-08): Santiago caught a real misalignment in the
 *   v1.13.2 screenshot — Actual/Forecast/Previous no longer sat directly
 *   above their own data columns. Cause: v1.13.2 appended the button-group
 *   "auto" grid track AFTER the three trailing 58px columns. Grid tracks are
 *   per-row, and the data rows below (.cal-event-row, in inline-index-styles.css)
 *   still use the original 7-column grid with no button track — so the
 *   header's own 1fr (Event) track ate the button group's width out of ITS
 *   available space while the data rows' Event track didn't, leaving the
 *   header's trailing 58px columns start ~button-width px to the left of
 *   where the data's Actual/Forecast/Previous actually are. Fix: moved the
 *   "auto" track (and the #cal-ccy-filter span) between Event and Actual —
 *   fixed-width tracks stay pixel-locked to the right edge regardless of
 *   where 1fr sits, so as long as nothing new sits to the right of Previous,
 *   alignment holds. buildPanel()'s relocation logic updated to insert
 *   (not append) at that same position when moving the node back from
 *   #cal-panel-head-actions.
 * v1.13.2-TEST (2026-08-08): Follow-up per Santiago's review of v1.13.1-TEST —
 *   two problems, both in the harness/markup, not the filter logic itself:
 *   (a) The header bar did NOT look identical to production. v1.13.1 rebuilt
 *       #cal-static-col-header as a flex wrapper (grid div + button group)
 *       instead of keeping production's own single `display:grid;
 *       grid-template-columns:52px 52px 18px 1fr 58px 58px 58px` rule — an
 *       extra nesting level that changed how "Event"'s 1fr track and the
 *       trailing number columns actually rendered. Reverted to the exact
 *       production grid in index-test.html, with only one appended `auto`
 *       track at the end for the button group — this file's rendering logic
 *       is unaffected, the fix is markup-only.
 *   (b) In wide-fullscreen 2-column mode (shouldSplitCalColumns()),
 *       #cal-static-col-header — the only place the filter buttons lived —
 *       is hidden entirely (production behavior, untouched). buildPanel()
 *       now relocates the existing #cal-ccy-filter node into
 *       #cal-panel-head-actions (next to the panel title) whenever splitCols
 *       is true, and moves it back when not, so it's never simply gone.
 * v1.13.1-TEST (2026-08-08): Follow-up per Santiago's review of v1.13-TEST:
 *   (a) Currency filter changed from multi-select-with-removal to ISOLATE
 *       semantics (click a currency → show ONLY it; click again/All → show
 *       all), and moved from its own pill row to the right edge of the
 *       column-header bar, restyled to match #corr-window-btns (Cross-Asset
 *       Correlations' 30d/60d/90d buttons) instead of rounded flag pills.
 *   (b) Font mismatch in the harness was NOT a bug in this file — index.html
 *       loads Inter/JetBrains Mono via a Google Fonts <link> that
 *       index-test.html was missing; fixed there, not here.
 * v1.13-TEST (2026-08-08): SANDBOX — Three "quick win" enhancements requested by
 *   Santiago to move the panel closer to Bloomberg/Refinitiv conventions, built on
 *   an isolated test copy (calendar-panel-test.js / index-test.html) so production
 *   dashboard.js/calendar-panel.js/index.html are untouched pending review:
 *   (1) Currency filter pills (G8) above the event list, persisted in localStorage
 *       under 'gi_cal_ccy_filter'. Client-side only — the impact filter and G8 set
 *       already applied server-side stay exactly as they were; this just narrows
 *       what's rendered from the same fetched dataset.
 *   (2) Revision marker: when a released event's `previous` value doesn't match the
 *       `actual` that was recorded for the same title+currency the last time it was
 *       released, a small superscript "R" appears next to Previous with a tooltip
 *       showing old → new. Built entirely from data already in ff_calendar.json /
 *       calendar.json (21-day and full-year history respectively) — no backend or
 *       pipeline change needed.
 *   (3) Surprise-magnitude styling: the existing binary up/down coloring on Actual
 *       is now tiered (mild/moderate/strong) by relative deviation from forecast,
 *       so a small beat and a large beat no longer look identical. Heuristic tiers
 *       (2% / 8% / 20% relative deviation) — a placeholder pending calibration
 *       against real dispersion per indicator; documented inline at _surpriseTier().
 * v1.12 (2026-08-07): BUG FIX — Actual-column beat/miss coloring silently
 *   never applied to any currency-amount event (Balance of Trade, Imports,
 *   Exports, Current Account, etc.). The local `stripNum` only removed %,
 *   commas, K/M/B/T and whitespace — it left leading currency symbols
 *   ($, C$, A$, €, ¥...) in place, so `parseFloat("C$3.86B")` (after strip:
 *   "C$3.86") returned NaN, the `!isNaN` guard failed, and `cls` stayed ''.
 *   Found live from Santiago's screenshots: Canada/US/Australia Balance of
 *   Trade, US Imports/Exports all rendering with no green/red despite a
 *   clear actual-vs-forecast beat or miss. Same bug class dashboard.js's
 *   `_parseNum()` and fetch_economic_calendar.py's `_parse_num()` already
 *   fixed for ESI scoring — confirmed those two (and econ-surprises-modal.js)
 *   were unaffected, since they already strip-to-digits-and-restore-sign
 *   rather than pattern-excluding known suffixes. New module-scope
 *   `_calParseNum()` ports that same correct strategy here; this was purely
 *   a display bug isolated to this panel's own separate implementation.
 */
(function () {
  'use strict';

  const G8_CURRENCIES      = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD']);
  const G8_LIST             = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
  const IMPACTS = new Set(['medium','high']);

  // ── [TEST v1.13] Currency filter state ──────────────────────────────────
  // Persisted client-side only (localStorage) — narrows what's rendered from
  // the same already-fetched, already-server-filtered (G8 + medium/high
  // impact) dataset.
  // v1.13.1: changed from multi-select-with-removal (clicking a currency
  // hid it) to ISOLATE semantics (clicking a currency shows ONLY that
  // currency; clicking it again — or "All" — restores all). Matches how
  // Santiago actually wanted to use it and how #corr-window-btns' 30d/60d/90d
  // group behaves (single active selection, not a multi-toggle).
  // null = "show all" (default / initial state).
  const CAL_CCY_FILTER_KEY = 'gi_cal_ccy_filter';
  function loadCcyFilter() {
    try {
      const raw = localStorage.getItem(CAL_CCY_FILTER_KEY);
      if (!raw) return null;
      const v = JSON.parse(raw);
      return (typeof v === 'string' && G8_CURRENCIES.has(v)) ? v : null;
    } catch { return null; }
  }
  function saveCcyFilter(v) {
    try {
      if (v == null) localStorage.removeItem(CAL_CCY_FILTER_KEY);
      else localStorage.setItem(CAL_CCY_FILTER_KEY, JSON.stringify(v));
    } catch {}
  }
  let _ccyFilter = loadCcyFilter(); // string (single ccy) or null (all)

  // ── [TEST v1.16] Impact filter (High only) ───────────────────────────────
  // Second, independent filter alongside the currency isolate — narrows the
  // already-fetched, already G8+medium/high-filtered dataset down to just
  // high-impact events. Persisted the same way as the currency filter.
  const CAL_IMPACT_FILTER_KEY = 'gi_cal_impact_filter';
  function loadImpactFilter() {
    try { return localStorage.getItem(CAL_IMPACT_FILTER_KEY) === '1'; } catch { return false; }
  }
  function saveImpactFilter(v) {
    try {
      if (v) localStorage.setItem(CAL_IMPACT_FILTER_KEY, '1');
      else localStorage.removeItem(CAL_IMPACT_FILTER_KEY);
    } catch {}
  }
  let _impactHighOnly = loadImpactFilter();
  function passesImpactFilter(ev) {
    return IMPACTS.has(ev.impact) && (!_impactHighOnly || ev.impact === 'high');
  }

  // ── [TEST v1.16] Week navigation ──────────────────────────────────────────
  // Shifts the whole -3d/+14d display window by ±7 days per click. Not
  // persisted (resets to the current window on reload) — same convention as
  // a Bloomberg calendar paging forward/back without "remembering" where you
  // left off. offset 0 is always the real current window.
  let _calWeekOffsetDays = 0;

  // Cache of the last successful fetch — lets relayoutCalendar() re-render
  // (e.g. switching between 1 and 2 columns on fullscreen open/close/resize)
  // without a network round-trip.
  let _lastEvents   = null;
  let _lastSource   = null;
  let _lastHolidays = null;

  // ── [TEST v1.16] Full-year history index (for cadence + drill-down modal) ──
  // ff_calendar.json's own window is only ~21 days — nowhere near enough to
  // detect a monthly/quarterly cadence or show "last 8 prints" for anything
  // but a weekly series. calendar.json separately carries a full rolling
  // year (confirmed: 3720 events / ~690 unique titles as of this session) —
  // that's the dataset these two features need, independent of whichever
  // file `events`/`filtered` ends up using for the main render list. Kept as
  // its own module var, refreshed every fetch, never merged into `events`.
  let _lastFullHistory = [];
  let _seriesIndex     = {};


  const IMPACT_DOT = {
    high:   { color: 'var(--down)',   label: 'High'   },
    medium: { color: 'var(--orange)', label: 'Medium' },
  };

  const FLAG = { USD:'us', EUR:'eu', GBP:'gb', JPY:'jp', AUD:'au', CAD:'ca', CHF:'ch', NZD:'nz' };

  // Indicators where a higher actual than forecast is BAD news (rising unemployment,
  // rising jobless claims, a wider deficit) and must render as "down" (red), not "up"
  // (green). Without this, the naive actualN > forecastN comparison below paints a
  // worse-than-expected print green just because the number itself is numerically
  // larger — e.g. NZD Unemployment Rate printing 5.6% vs. 5.4% forecast/previous is a
  // negative surprise but was rendering green before this fix.
  // Must stay in sync with INVERSE_KW in dashboard.js, _ESM_INVERSE_KW in
  // econ-surprises-modal.js, and INVERSE_EVENTS in fetch_economic_calendar.py.
  // v8.100.7: added "unemployed" — "Unemployed Persons" (EUR/Germany, NOK) is not a
  // substring match of "unemployment". See dashboard.js INVERSE_KW comment.
  const CAL_INVERSE_KW = ['unemployment', 'unemployed', 'jobless', 'claims', 'deficit'];

  // ── Numeric parser for macro actual/forecast values ─────────────────────
  // parseFloat() alone fails on currency-symbol-prefixed strings such as
  // "$-226.8B", "A$1.791B", "C$3.86B", "¥3907B", "-€5.2B" — the leading
  // symbol makes parseFloat return NaN before it ever reaches the digits,
  // so every Balance of Trade / Imports / Exports / Current Account row
  // silently lost its Actual-column beat/miss coloring (no exception, no
  // console warning — cls just stayed '' and the span rendered uncolored).
  // Same bug class already fixed in dashboard.js's _parseNum() and
  // fetch_economic_calendar.py's _parse_num() for ESI scoring — this panel
  // had its own separate, cruder `stripNum` (%, comma, K/M/B/T only, no
  // currency symbols) that never got the same fix. Ports the same
  // strip-to-digits-and-restore-sign strategy so behavior matches exactly.
  // Display-only: does not touch ESI scoring, which was already correct.
  const _calParseNum = s => {
    if (s == null || s === '') return NaN;
    const str = String(s).replace(/,/g, '');
    const neg = str.includes('-');
    const digits = str.replace(/[^\d.]/g, '');
    const n = parseFloat(digits);
    return isNaN(n) ? NaN : (neg ? -n : n);
  };

  // ── [TEST v1.13] Surprise-magnitude tiering ─────────────────────────────
  // Existing logic only ever applied a binary up/down class regardless of how
  // large the beat/miss was. This buckets the *relative* deviation from
  // forecast into three tiers so a 0.1pp beat and a huge miss (e.g. NFP
  // -23K vs 80K forecast) read differently at a glance — closer to how
  // Bloomberg/Refinitiv shade surprise magnitude.
  // NOTE: relative-deviation-from-forecast is a simple, defensible proxy —
  // not a true z-score against the indicator's own historical dispersion
  // (that would need a volatility/std-dev table per title, which doesn't
  // exist yet). Thresholds (2% / 8% / 20%) are placeholder defaults; revisit
  // once we can calibrate per-indicator from the ESI history already on file.
  function _surpriseTier(actualN, forecastN) {
    if (forecastN === 0) return Math.abs(actualN) > 0 ? 'strong' : 'mild';
    const rel = Math.abs((actualN - forecastN) / forecastN);
    if (rel >= 0.20) return 'strong';
    if (rel >= 0.08) return 'moderate';
    return 'mild';
  }

  function _escAttr(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── [TEST v1.16] Series canonicalization + history index ────────────────
  // Shared by the cadence tag and the historical drill-down modal (and
  // previously by the now-removed ESI badge — this is a leaner version with
  // no ESI-specific noise list, just the country-prefix strip needed so
  // "United States Non Farm Payrolls" and a hypothetical bare "Non Farm
  // Payrolls" key the same series).
  const _CAL_CCY_PFXS = ['united states ', 'euro area ', 'united kingdom ', 'japan ',
    'australia ', 'canada ', 'switzerland ', 'new zealand ', 'norway ', 'sweden '];
  function _calCanonTitle(t) {
    let s = (t || '').toLowerCase().replace(/\s*\([^)]*\)/g, '').trim();
    for (const p of _CAL_CCY_PFXS) { if (s.startsWith(p)) { s = s.slice(p.length); break; } }
    return s;
  }
  function _calSeriesKey(ev) { return `${ev.currency}/${_calCanonTitle(ev.title)}`; }

  // Builds { "USD/non farm payrolls": [{dateISO,timeUTC,actual,forecast,previous}, ...] }
  // sorted oldest→newest, from the full-year history — only entries that
  // actually printed (actual present) count as a "release" for cadence/
  // history purposes.
  function buildSeriesIndex(fullHistory) {
    const idx = {};
    fullHistory.forEach(ev => {
      if (ev.actual == null || ev.actual === '' || ev.actual === '-') return;
      const key = _calSeriesKey(ev);
      (idx[key] = idx[key] || []).push({
        dateISO: ev.dateISO, timeUTC: ev.timeUTC,
        actual: ev.actual, forecast: ev.forecast, previous: ev.previous,
      });
    });
    Object.values(idx).forEach(arr => arr.sort((a, b) => a.dateISO < b.dateISO ? -1 : (a.dateISO > b.dateISO ? 1 : 0)));
    return idx;
  }

  // Data-driven cadence label — deliberately NOT a maintained keyword list
  // (this codebase already has a documented failure mode where keyword
  // lists drift out of sync across files/updates). Instead, look at the
  // actual gaps between this series' own past release dates: low variance
  // → real fixed cadence, bucketed by the mean gap. High variance (e.g. ad
  // hoc central-bank speeches, one-off reports) → no tag, since a "cadence"
  // label would be misleading. Needs ≥3 prior releases to say anything.
  function inferCadence(seriesArr) {
    if (!seriesArr || seriesArr.length < 3) return null;
    const dates = seriesArr.map(e => Date.parse(e.dateISO + 'T00:00:00Z'));
    const gaps = [];
    for (let i = 1; i < dates.length; i++) gaps.push((dates[i] - dates[i - 1]) / 86400000);
    const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    if (mean <= 0) return null;
    const variance = gaps.reduce((a, b) => a + (b - mean) * (b - mean), 0) / gaps.length;
    const cv = Math.sqrt(variance) / mean; // coefficient of variation
    if (cv > 0.35) return null; // irregular — don't mislabel
    if (mean <= 10)  return 'Weekly';
    if (mean <= 40)  return 'Monthly';
    if (mean <= 100) return 'Quarterly';
    if (mean <= 200) return 'Semi-Annual';
    if (mean <= 400) return 'Annual';
    return null;
  }

  // ── [TEST v1.16] FOMC voting-member tag ──────────────────────────────────
  // Bloomberg/Refinitiv flag whether a Fed speaker is a current FOMC voter —
  // a non-voter's remarks still matter but carry less direct near-term
  // policy weight. Scoped to the Fed ONLY: of the G10 central banks this
  // calendar covers, the FOMC is the one with a structural voting/non-voting
  // split (7 Governors + NY Fed always vote; 4 of the other 11 regional
  // presidents rotate onto a vote each year). The ECB Governing Council, BoE
  // MPC, BoC Governing Council, RBA Board, SNB Governing Board, RBNZ MPC,
  // Norges Bank committee and Riksbank Executive Board all have their full
  // monetary-policy body vote at every meeting — there's no non-voting
  // subset to flag, so no tag is shown for those speakers (not an omission).
  //
  // Snapshot sourced from federalreserve.gov/monetarypolicy/fomc.htm
  // (page's own "Last Update: July 29, 2026") — 2026 rotating seats: Cleveland
  // (Hammack), Dallas (Logan), Philadelphia (Paulson), Minneapolis (Kashkari)
  // vote; Boston, Chicago, Kansas City, St. Louis, Richmond, San Francisco,
  // Atlanta do not. Board of Governors + NY Fed always vote. This is a
  // dated snapshot, not a live feed — the rotation changes every Jan 1 and
  // Board seats can change with a confirmation at any time; re-verify
  // against the same source (same method as the NOK/threshold and inverse-
  // keyword audits already logged in GUIDELINES.md) before relying on this
  // for a full year, and definitely re-check every January.
  const FOMC_VOTERS_2026 = {
    // Board of Governors — always voting
    warsh:    { voting: true, role: 'Fed Chair' },
    jefferson:{ voting: true, role: 'Fed Vice Chair' },
    bowman:   { voting: true, role: 'Fed Vice Chair for Supervision' },
    barr:     { voting: true, role: 'Fed Governor' },
    cook:     { voting: true, role: 'Fed Governor' },
    waller:   { voting: true, role: 'Fed Governor' },
    powell:   { voting: true, role: 'Fed Governor' },
    // NY Fed — permanent voter
    williams: { voting: true, role: 'NY Fed President (permanent voter)' },
    // Rotating regional presidents — VOTING in 2026
    hammack:  { voting: true, role: 'Cleveland Fed President' },
    logan:    { voting: true, role: 'Dallas Fed President' },
    paulson:  { voting: true, role: 'Philadelphia Fed President' },
    kashkari: { voting: true, role: 'Minneapolis Fed President' },
    // Rotating regional presidents — NON-voting in 2026
    collins:  { voting: false, role: 'Boston Fed President' },
    goolsbee: { voting: false, role: 'Chicago Fed President' },
    schmid:   { voting: false, role: 'Kansas City Fed President' },
    musalem:  { voting: false, role: 'St. Louis Fed President' },
    barkin:   { voting: false, role: 'Richmond Fed President' },
    daly:     { voting: false, role: 'San Francisco Fed President' },
    venable:  { voting: false, role: 'Atlanta Fed Interim President' },
  };
  function _fomcVoterTag(currency, title) {
    if (currency !== 'USD') return '';
    const canon = _calCanonTitle(title); // e.g. "fed barkin speech"
    const m = canon.match(/^fed\s+([a-z]+)\s+speech$/);
    if (!m) return '';
    const info = FOMC_VOTERS_2026[m[1]];
    if (!info) return ''; // unrecognized name — say nothing rather than guess
    const cls  = info.voting ? 'up' : '';
    const style = info.voting
      ? 'color:var(--up);font-size:8px;cursor:help;font-weight:700;'
      : 'color:var(--text3);font-size:8px;cursor:help;';
    const label = info.voting ? 'V' : 'nv';
    const tip = `${info.role} — ${info.voting ? 'current FOMC voter' : 'non-voter this year'} ` +
      `(2026 rotation, federalreserve.gov as of Jul 2026 — verify if relying on this after a rotation change).`;
    return ` <sup style="${style}" title="${_escAttr(tip)}">${label}</sup>`;
  }

  // ── [TEST v1.14] Live / next-release highlight ───────────────────────────
  // Bloomberg-style: the single next high-impact event due within a short
  // forward window gets a highlighted row + a live countdown in place of its
  // clock time, so the user doesn't have to scan the whole list to see
  // what's about to print. Only ever one target at a time (the soonest),
  // scoped to whatever's currently visible (respects the currency filter and
  // display window) — matches "resalta la fila que está por publicarse en
  // los próximos minutos" rather than highlighting everything due today.
  const CAL_LIVE_WINDOW_MS     = 3  * 60 * 60 * 1000; // highlight if due within 3h
  const CAL_LIVE_IMMINENT_MS   = 15 * 60 * 1000;       // pulsing tier if due within 15m

  function findNextHighImpactEvent(filtered, nowMs) {
    let best = null;
    filtered.forEach(ev => {
      if (ev.impact !== 'high') return;
      const isReleased = !!(ev.actual && ev.actual !== '' && ev.actual !== '-');
      if (isReleased) return;
      const [h, m] = (ev.timeUTC || '23:59').split(':').map(Number);
      const evMs = Date.UTC(+ev.dateISO.slice(0,4), +ev.dateISO.slice(5,7)-1, +ev.dateISO.slice(8,10), h, m);
      const delta = evMs - nowMs;
      if (delta <= 0 || delta > CAL_LIVE_WINDOW_MS) return;
      if (!best || evMs < best.evMs) best = { ev, evMs };
    });
    return best;
  }

  function fmtCountdown(ms) {
    if (ms <= 0) return 'now';
    const totalMin = Math.round(ms / 60000);
    if (totalMin < 1) return '<1m';
    if (totalMin < 60) return totalMin + 'm';
    const h = Math.floor(totalMin / 60), m = totalMin % 60;
    return h + 'h' + (m ? ' ' + m + 'm' : '');
  }

  // One-time <style> injection (mirrors dashboard.js's attachRiskTip /
  // ticker-exact pattern) — pulsing dot + soft row tint, no new stylesheet
  // file needed for a sandbox-only feature.
  function ensureLiveStyles() {
    if (document.getElementById('cal-live-style')) return;
    const s = document.createElement('style');
    s.id = 'cal-live-style';
    s.textContent = `
      @keyframes calLivePulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }
      .cal-event-row.cal-live-soon     { background: rgba(255,167,38,.06); }
      .cal-event-row.cal-live-soon:hover { background: rgba(255,167,38,.12); }
      .cal-event-row.cal-live-imminent { background: rgba(239,83,80,.10); }
      .cal-event-row.cal-live-imminent:hover { background: rgba(239,83,80,.16); }
      .cal-live-countdown { animation: calLivePulse 1.1s ease-in-out infinite; color: var(--down); }
    `;
    document.head.appendChild(s);
  }

  // Ticks every 20s, independent of the 2-min data poll — just updates the
  // countdown text of whatever's currently tagged data-live-ms, so the timer
  // counts down smoothly instead of jumping in 2-min steps.
  function tickLiveCountdown() {
    document.querySelectorAll('[data-live-ms]').forEach(el => {
      const target = Number(el.dataset.liveMs);
      if (!target) return;
      el.textContent = fmtCountdown(target - Date.now());
    });
  }

  // ── [TEST v1.15] Synthetic live-countdown fixture ────────────────────────
  // Real data rarely has a qualifying high-impact event sitting inside the
  // 3h/15m live-countdown window at the exact moment someone opens this
  // sandbox to check the feature. Opt-in only — append ?calDebugLive=1 to
  // index-test.html's URL — injects one clearly-labeled fake event so the
  // countdown/highlight can be exercised on demand, independent of the
  // real-world clock. Never runs without the query flag, never touches any
  // fetched JSON, and the title is prefixed "[TEST FIXTURE]" so it can't be
  // mistaken for a real release in a screenshot. Target time is seeded once
  // per page load (20m out) rather than recomputed every 2-min poll, so it
  // actually counts down in real time and crosses from the "soon" tier into
  // the "imminent" pulsing tier ~5 minutes after load, same as a real event
  // would — reload the page to re-seed another 20m window.
  let _syntheticTargetMs = null;
  function getSyntheticLiveEvent(nowMs) {
    if (_syntheticTargetMs == null) _syntheticTargetMs = nowMs + 20 * 60 * 1000;
    const d = new Date(_syntheticTargetMs);
    return {
      dateISO: d.toISOString().slice(0, 10),
      timeUTC: d.toISOString().slice(11, 16),
      currency: 'USD', impact: 'high',
      title: '[TEST FIXTURE] Non-Farm Payrolls',
      forecast: '180K', previous: '175K', actual: null,
    };
  }
  function calDebugLiveEnabled() {
    try { return new URLSearchParams(location.search).get('calDebugLive') === '1'; }
    catch { return false; }
  }

  // ── [TEST v1.14] Event methodology tooltips ──────────────────────────────
  // Same pattern Santiago asked to reuse from the ATM IV tooltips: a clean,
  // named, plain-language explanation on hover — what the release measures
  // and why FX desks watch it — with no backend/pipeline attribution (this
  // is product copy, not sourced from any fetched document, so it carries
  // no citation obligation). Matched by keyword against the canonical title
  // (case-insensitive substring, first match wins — same convention as
  // CAL_INVERSE_KW above). Not exhaustive — G10 headline
  // releases only; anything unmatched falls back to the plain event-name
  // tooltip that was already there.
  const CAL_METHODOLOGY = [
    { kw: ['non-farm payrolls', 'nonfarm payrolls', 'employment change'],
      text: 'Net change in jobs outside farming, private households, and nonprofits. The single most-watched US labor print — a big beat/miss can move every USD pair within seconds of release.' },
    { kw: ['unemployment rate'],
      text: 'Share of the labor force that is jobless and actively looking for work. A rising rate is a negative surprise for the currency even though the headline number is numerically larger.' },
    { kw: ['average hourly earnings', 'wage price index', 'labour cost index', 'labor cost index'],
      text: 'Wage growth over the period. Central banks watch this as a leading indicator of sticky, demand-driven inflation — hot wage growth tends to firm up rate-hike expectations.' },
    { kw: ['initial jobless claims', 'continuing jobless claims', 'jobless claims'],
      text: 'Weekly count of new (or ongoing) unemployment benefit filings. A high-frequency, low-noise read on labor-market health between the monthly jobs reports.' },
    { kw: ['adp employment'],
      text: "Private-sector payrolls processor ADP's own employment estimate, released two days ahead of official Non-Farm Payrolls. Treated as an imperfect early read, not a reliable predictor of the NFP print." },
    { kw: ['cpi', 'consumer price index', 'inflation rate'],
      text: 'Headline consumer price inflation. Directly feeds central bank rate decisions — a hot print usually firms up hawkish rate expectations and supports the currency, and vice versa.' },
    { kw: ['core inflation', 'core cpi', 'core pce', 'pce price index'],
      text: 'Inflation excluding volatile food and energy prices. Central banks weight this more heavily than headline CPI when setting policy, since it better reflects underlying price pressure.' },
    { kw: ['ppi', 'producer price index'],
      text: "Prices received by producers at the factory gate. A leading indicator for consumer inflation a month or two out, since producer cost pressure tends to pass through to retail prices." },
    { kw: ['gdp'],
      text: 'Gross Domestic Product — the broadest measure of economic output. Quarterly growth (or contraction) versus consensus shapes the market\u2019s view of the whole economic cycle, not just one sector.' },
    { kw: ['retail sales'],
      text: 'Change in consumer spending at the retail level. Consumption drives the majority of GDP in most G10 economies, so this is a fast, monthly proxy for overall demand.' },
    { kw: ['ism manufacturing', 'ism services', 'ism non-manufacturing'],
      text: 'Institute for Supply Management survey of purchasing managers. Above 50 = sector expanding, below 50 = contracting. One of the earliest-available reads on the current month\u2019s activity.' },
    { kw: ['manufacturing pmi', 'services pmi', 'composite pmi', 'flash pmi'],
      text: 'Purchasing Managers\u2019 Index survey. Above 50 = sector expanding, below 50 = contracting — a timely, forward-looking gauge of business activity ahead of harder monthly data.' },
    { kw: ['interest rate decision', 'rate decision', 'cash rate', 'official cash rate', 'refinancing rate', 'ocr'],
      text: "Central bank policy rate announcement. Directly sets the currency's carry/funding cost — the decision itself usually matters less than the accompanying guidance on the path ahead." },
    { kw: ['fomc statement', 'fomc minutes', 'fomc press conference', 'monetary policy statement', 'monetary policy report', 'rate statement'],
      text: 'Central bank\u2019s own account of its policy discussion and forward guidance. Markets parse the language itself for hints on the future rate path, independent of the rate decision.' },
    { kw: ['balance of trade', 'trade balance'],
      text: 'Exports minus imports of goods and services. A signed net level, not a rate — a widening deficit or narrowing surplus can pressure the currency via the current-account channel.' },
    { kw: ['current account'],
      text: 'Broadest measure of a country\u2019s transactions with the rest of the world (trade plus income and transfers). Persistent deficits can weigh on a currency\u2019s longer-term valuation.' },
    { kw: ['industrial production', 'manufacturing production'],
      text: 'Output of factories, mines, and utilities. A real-activity read that complements survey-based PMI data with actual production volumes.' },
    { kw: ['durable goods', 'factory orders', 'core durable goods'],
      text: 'New orders for goods meant to last three years or more (autos, machinery, aircraft). A forward-looking proxy for business investment appetite.' },
    { kw: ['building permits', 'housing starts'],
      text: 'New residential construction authorized (permits) or begun (starts). An early-cycle housing indicator that feeds into broader growth and employment expectations.' },
    { kw: ['existing home sales', 'new home sales', 'pending home sales', 'home sales'],
      text: 'Volume of homes sold. Tracks the health of the housing market and, by extension, consumer wealth and willingness to spend.' },
    { kw: ['consumer confidence', 'consumer sentiment', 'michigan'],
      text: 'Survey of household attitudes toward current and expected economic conditions. A sentiment leading-indicator for future consumer spending.' },
    { kw: ['zew'],
      text: 'ZEW Institute survey of financial analysts\u2019 economic expectations for the next six months. A closely watched early-cycle sentiment gauge for the Eurozone/Germany.' },
    { kw: ['ifo'],
      text: 'Ifo Institute survey of German businesses on current conditions and expectations. One of the most-watched single-country business-climate indicators in the Eurozone.' },
    { kw: ['gdt price index', 'global dairy trade'],
      text: "Global Dairy Trade auction price index. Dairy is one of New Zealand's largest export categories, so this auction result is a direct NZD terms-of-trade signal." },
    { kw: ['housing price index', 'house price index', 'home price index'],
      text: 'Change in residential property prices. A wealth-effect and financial-stability indicator that central banks monitor alongside credit growth.' },
    { kw: ['claimant count'],
      text: 'UK measure of people claiming unemployment-related benefits. The UK\u2019s closest equivalent to the US jobless-claims series for tracking labor-market momentum between official unemployment reports.' },
  ];
  function _calMethodologyFor(title) {
    const t = (title || '').toLowerCase();
    for (const entry of CAL_METHODOLOGY) {
      if (entry.kw.some(k => t.includes(k))) return entry.text;
    }
    return '';
  }

  // Self-contained tooltip (this file has no dependency on dashboard.js
  // being loaded — the sandbox harness doesn't load it — so a scoped copy
  // of attachRiskTip's visual pattern lives here under its own #cal-tt id
  // rather than reusing window.attachRiskTip). Delegated listeners, bound
  // once on the scroll container, so re-renders never need to re-attach
  // per-row handlers or leak listeners.
  function ensureMethodologyTooltip() {
    if (document.getElementById('cal-tt-style')) return;
    const s = document.createElement('style');
    s.id = 'cal-tt-style';
    s.textContent = `
      #cal-tt {
        position:fixed;z-index:99999;width:min(240px, calc(100vw - 24px));
        background:var(--bg3);border:1px solid var(--border2);border-radius:4px;
        padding:9px 11px;font-size:11px;color:var(--text);line-height:1.55;
        pointer-events:none;display:none;font-family:var(--font-ui);box-sizing:border-box;
      }
      #cal-tt .tt-title { font-weight:700;font-size:11px;color:#fff;margin-bottom:3px; }
      .cal-col.cal-title[data-cal-tip] { border-bottom:1px dashed rgba(255,255,255,0.2); cursor:help; }
    `;
    document.head.appendChild(s);
    const ttEl = document.createElement('div');
    ttEl.id = 'cal-tt';
    ttEl.innerHTML = '<div class="tt-title" id="cal-tt-title"></div><div id="cal-tt-body"></div>';
    document.body.appendChild(ttEl);
    document.addEventListener('mousemove', ev => {
      const tt = document.getElementById('cal-tt');
      if (tt && tt.style.display === 'block') _calTTPos(ev.clientX, ev.clientY);
    });
  }
  function _calTTPos(cx, cy) {
    const tt = document.getElementById('cal-tt');
    if (!tt) return;
    const vw = window.innerWidth, vh = window.innerHeight;
    const ttW = Math.min(240, vw - 24);
    const ttH = tt.offsetHeight || 90;
    const PAD = 8;
    let x = cx + 14, y = cy + 14;
    if (x + ttW > vw - PAD) x = cx - ttW - 8;
    if (x < PAD) x = PAD;
    if (y + ttH > vh - PAD) y = cy - ttH - 8;
    if (y < PAD) y = PAD;
    tt.style.left = x + 'px'; tt.style.top = y + 'px';
  }
  function setupMethodologyTooltipDelegation(container) {
    if (!container || container.dataset.calTipInit === '1') return;
    container.dataset.calTipInit = '1';
    const show = (el, cx, cy) => {
      const tt = document.getElementById('cal-tt');
      if (!tt) return;
      document.getElementById('cal-tt-title').textContent = el.dataset.calTipTitle || '';
      document.getElementById('cal-tt-body').textContent  = el.dataset.calTipBody  || '';
      tt.style.display = 'block';
      requestAnimationFrame(() => _calTTPos(cx, cy));
    };
    const hide = () => { const tt = document.getElementById('cal-tt'); if (tt) tt.style.display = 'none'; };
    container.addEventListener('mouseover', e => {
      const el = e.target.closest('.cal-col.cal-title[data-cal-tip]');
      if (el) show(el, e.clientX, e.clientY);
    });
    container.addEventListener('mouseout', e => {
      if (e.target.closest('.cal-col.cal-title[data-cal-tip]')) hide();
    });
    container.addEventListener('touchstart', e => {
      const el = e.target.closest('.cal-col.cal-title[data-cal-tip]');
      if (el) { e.stopPropagation(); const t = e.touches[0]; show(el, t.clientX, t.clientY); }
    }, { passive: true });
  }

  // ── [TEST v1.16] Historical drill-down modal ─────────────────────────────
  // Click an event title (any event, not just methodology-matched ones) to
  // open a modal with: methodology blurb, cadence tag, FOMC voter info when
  // relevant, and the last up to 8 releases of that exact series (from the
  // full-year history — see buildSeriesIndex()) with the same beat/miss
  // coloring as the main row. Deliberately a click-through, not another
  // inline badge — Santiago flagged that per-row space is tight (this is
  // also why the earlier ESI contribution badge was dropped), so anything
  // beyond a 1-2 character marker belongs behind a click, not in the row.
  //
  // Also surfaces a coarse "reference pair" daily-move context: average
  // daily range on this series' past release days vs. this pair's typical
  // daily range, using the OHLC files already on disk (ohlc-data/*.json).
  // IMPORTANT CAVEAT stated in the UI itself, not just here: these are DAILY
  // bars, not intraday — this cannot isolate the specific minutes right
  // after the release from the rest of that day's news. It's a same-day
  // volatility-context proxy ("does this release tend to coincide with a
  // bigger-than-usual day for this pair"), not a measured post-release
  // reaction. A true post-release-window reaction metric would need
  // intraday bars timestamped against the release time, which isn't in any
  // data source this project currently fetches — flagged as a gap, not
  // silently approximated as more precise than it is.
  const CAL_REF_PAIR = { USD:'dxy', EUR:'eurusd', GBP:'gbpusd', JPY:'usdjpy',
    AUD:'audusd', CAD:'usdcad', CHF:'usdchf', NZD:'nzdusd' };
  function _pairMoveUnit(pairKey) {
    if (pairKey === 'dxy')    return { div: 1,      unit: 'pts',  dp: 2 };
    if (pairKey === 'usdjpy') return { div: 0.01,    unit: 'pips', dp: 0 };
    return                         { div: 0.0001,  unit: 'pips', dp: 0 };
  }
  const _ohlcCache = {};
  async function fetchRefPairOHLC(ccy) {
    const pairKey = CAL_REF_PAIR[ccy];
    if (!pairKey) return null;
    if (_ohlcCache[pairKey]) return _ohlcCache[pairKey];
    try {
      const res = await fetch(`./ohlc-data/${pairKey}.json`, { cache: 'no-store' });
      if (!res.ok) return null;
      const data = await res.json();
      _ohlcCache[pairKey] = data;
      return data;
    } catch { return null; }
  }
  function computeReleaseDayMove(bars, releaseDatesISO, unit) {
    if (!bars || !bars.length) return null;
    const byDate = {};
    bars.forEach(b => { byDate[b.time] = b; });
    const allRanges = bars
      .map(b => (b.high - b.low) / unit.div)
      .filter(n => isFinite(n) && n >= 0);
    if (!allRanges.length) return null;
    const overallAvg = allRanges.reduce((a, b) => a + b, 0) / allRanges.length;
    const relBars = releaseDatesISO.map(d => byDate[d]).filter(Boolean);
    if (relBars.length < 2) return null; // not enough overlap with available OHLC history to mean anything
    const relRanges = relBars.map(b => (b.high - b.low) / unit.div);
    const relAvg = relRanges.reduce((a, b) => a + b, 0) / relRanges.length;
    return { relAvg, overallAvg, n: relRanges.length, unit: unit.unit, dp: unit.dp };
  }

  function _calBeatClass(actualN, forecastN, isInverse) {
    if (isNaN(actualN) || isNaN(forecastN) || actualN === forecastN) return '';
    const beat = isInverse ? actualN < forecastN : actualN > forecastN;
    return beat ? 'up' : 'down';
  }

  function ensureHistModal() {
    if (document.getElementById('cal-hist-style')) return;
    const s = document.createElement('style');
    s.id = 'cal-hist-style';
    s.textContent = `
      #cal-hist-overlay {
        position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:100000;
        display:none;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;
      }
      #cal-hist-modal {
        background:var(--bg2, var(--bg3));border:1px solid var(--border2);border-radius:6px;
        width:min(420px, 100%);max-height:min(560px, 90vh);overflow-y:auto;
        font-family:var(--font-ui);color:var(--text);box-sizing:border-box;
      }
      #cal-hist-modal .ch-head {
        display:flex;align-items:center;justify-content:space-between;gap:8px;
        padding:10px 12px;border-bottom:1px solid var(--border2);position:sticky;top:0;
        background:var(--bg2, var(--bg3));
      }
      #cal-hist-modal .ch-title { font-size:12px;font-weight:700;color:#fff; }
      #cal-hist-modal .ch-close {
        background:none;border:none;color:var(--text3);cursor:pointer;font-size:14px;line-height:1;padding:2px 4px;
      }
      #cal-hist-modal .ch-body { padding:10px 12px;font-size:11px;line-height:1.55;color:var(--text2); }
      #cal-hist-modal .ch-tag {
        display:inline-block;font-size:8px;padding:1px 5px;border-radius:2px;margin-right:4px;
        background:var(--bg3);border:1px solid var(--border2);color:var(--text3);
      }
      #cal-hist-modal table { width:100%;border-collapse:collapse;margin-top:8px;font-size:10px; }
      #cal-hist-modal th { text-align:right;color:var(--text3);font-weight:400;font-size:9px;padding:3px 4px;text-transform:uppercase;letter-spacing:.03em; }
      #cal-hist-modal th:first-child, #cal-hist-modal td:first-child { text-align:left; }
      #cal-hist-modal td { text-align:right;padding:3px 4px;border-top:1px solid var(--border2); }
      #cal-hist-modal td.up { color:var(--up); }
      #cal-hist-modal td.down { color:var(--down); }
      #cal-hist-modal .ch-move { margin-top:10px;padding-top:8px;border-top:1px solid var(--border2);font-size:10px;color:var(--text3); }
    `;
    document.head.appendChild(s);
    const overlay = document.createElement('div');
    overlay.id = 'cal-hist-overlay';
    overlay.innerHTML = `<div id="cal-hist-modal" role="dialog" aria-modal="true" aria-labelledby="cal-hist-title">
      <div class="ch-head">
        <span class="ch-title" id="cal-hist-title"></span>
        <button type="button" class="ch-close" id="cal-hist-close" aria-label="Close">&#x2715;</button>
      </div>
      <div class="ch-body" id="cal-hist-body"></div>
    </div>`;
    document.body.appendChild(overlay);
    const close = () => { overlay.style.display = 'none'; };
    document.getElementById('cal-hist-close').addEventListener('click', close);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && overlay.style.display === 'flex') close();
    });
  }

  function openHistModal(ev) {
    ensureHistModal();
    const overlay = document.getElementById('cal-hist-overlay');
    const titleEl = document.getElementById('cal-hist-title');
    const bodyEl  = document.getElementById('cal-hist-body');
    if (!overlay || !titleEl || !bodyEl) return;

    const flag = FLAG[ev.currency] || '';
    const flagHtml = flag ? `<span class="fi fi-${flag}" style="margin-right:4px;font-size:10px;"></span>` : '';
    titleEl.innerHTML = `${flagHtml}${_escAttr(ev.currency)} \u00b7 ${_escAttr(ev.title)}`;

    const methodText = _calMethodologyFor(ev.title);
    const key         = _calSeriesKey(ev);
    const seriesArr   = _seriesIndex[key] || [];
    const cadence     = inferCadence(seriesArr);
    const evTitleLower = (ev.title || '').toLowerCase();
    const isInverse   = CAL_INVERSE_KW.some(kw => evTitleLower.includes(kw));

    let html = '';
    if (methodText) html += `<div>${_escAttr(methodText)}</div>`;
    if (cadence) html += `<div style="margin-top:6px;"><span class="ch-tag">${cadence}</span></div>`;
    if (isInverse) html += `<div style="margin-top:6px;color:var(--text3);font-style:italic;">Inverse indicator — a higher actual than forecast is colored as a miss, not a beat.</div>`;

    const last8 = seriesArr.slice(-8).reverse();
    if (last8.length) {
      html += `<table><thead><tr>
        <th>Date</th><th>Actual</th><th>Forecast</th><th>Previous</th>
      </tr></thead><tbody>`;
      last8.forEach(h => {
        const actualN   = _calParseNum(h.actual);
        const forecastN = _calParseNum(h.forecast ? String(h.forecast).replace(/\*$/, '') : (h.previous || ''));
        const cls = _calBeatClass(actualN, forecastN, isInverse);
        html += `<tr>
          <td>${_escAttr(h.dateISO)}</td>
          <td class="${cls}">${_escAttr(h.actual)}</td>
          <td>${_escAttr(h.forecast || '\u2014')}</td>
          <td>${_escAttr(h.previous || '\u2014')}</td>
        </tr>`;
      });
      html += `</tbody></table>`;
    } else {
      html += `<div style="margin-top:8px;color:var(--text3);">No prior actual/forecast history for this event in the last year.</div>`;
    }

    html += `<div class="ch-move" id="cal-hist-move">Loading reference-pair context\u2026</div>`;

    bodyEl.innerHTML = html;
    overlay.style.display = 'flex';

    // Reference-pair daily-move context — fetched lazily, only on modal
    // open, and cached per pair for the rest of the session.
    const pairKey = CAL_REF_PAIR[ev.currency];
    const moveEl  = document.getElementById('cal-hist-move');
    if (!pairKey || last8.length < 2) {
      if (moveEl) moveEl.textContent = '';
    } else {
      fetchRefPairOHLC(ev.currency).then(bars => {
        const el = document.getElementById('cal-hist-move');
        if (!el) return;
        const unit = _pairMoveUnit(pairKey);
        const releaseDates = seriesArr.map(h => h.dateISO);
        const move = computeReleaseDayMove(bars, releaseDates, unit);
        if (!move) { el.textContent = ''; return; }
        el.innerHTML = `${pairKey.toUpperCase()} avg daily range on this series\u2019 release days ` +
          `(${move.n} obs.): <b style="color:var(--text2);">${move.relAvg.toFixed(move.dp)} ${move.unit}</b> ` +
          `vs. <b style="color:var(--text2);">${move.overallAvg.toFixed(move.dp)} ${move.unit}</b> typical day. ` +
          `Daily-bar proxy, not an intraday post-release reaction measurement.`;
      });
    }
  }

  function setupHistModalDelegation(container) {
    if (!container || container.dataset.calHistInit === '1') return;
    container.dataset.calHistInit = '1';
    container.addEventListener('click', e => {
      const el = e.target.closest('.cal-col.cal-title[data-cal-hist-idx]');
      if (!el) return;
      const idx = Number(el.dataset.calHistIdx);
      const ev = _calRenderIndex[idx];
      if (ev) openHistModal(ev);
    });
  }
  let _calRenderIndex = []; // reset each buildPanel() call — see there

  // ── [TEST v1.13] Revision index ─────────────────────────────────────────
  // Bloomberg/Refinitiv mark a "previous" value with a small revision flag
  // when it doesn't match what was actually printed last time that same
  // series was released. Built purely from history already present in the
  // fetched dataset (ff_calendar.json's 21-day window / calendar.json's
  // full-year history) — no pipeline change required.
  // Returns: Map key `${currency}|${title}` -> sorted [{dateISO, actual}]
  function buildRevisionIndex(events) {
    const idx = {};
    events.forEach(ev => {
      if (ev.actual == null || ev.actual === '' || ev.actual === '-') return;
      const k = `${ev.currency}|${ev.title}`;
      (idx[k] = idx[k] || []).push({ dateISO: ev.dateISO, actual: ev.actual });
    });
    Object.values(idx).forEach(arr => arr.sort((a, b) => a.dateISO < b.dateISO ? -1 : 1));
    return idx;
  }
  // For a given event, find the actual that was recorded the last time this
  // series released BEFORE this event's own date, and compare it to this
  // event's `previous` field. Returns {old, new} if they differ, else null.
  function detectRevision(ev, revIdx) {
    if (!ev.previous) return null;
    const k = `${ev.currency}|${ev.title}`;
    const hist = revIdx[k];
    if (!hist || hist.length < 2) return null;
    // last release strictly before this one
    let priorActual = null;
    for (let i = hist.length - 1; i >= 0; i--) {
      if (hist[i].dateISO < ev.dateISO) { priorActual = hist[i].actual; break; }
    }
    if (priorActual == null) return null;
    const prevN  = _calParseNum(ev.previous);
    const priorN = _calParseNum(priorActual);
    if (isNaN(prevN) || isNaN(priorN)) return priorActual !== ev.previous ? { old: priorActual, new: ev.previous } : null;
    return prevN !== priorN ? { old: priorActual, new: ev.previous } : null;
  }

  // Browser timezone offset label e.g. "GMT-3"
  function tzLabel() {
    const off = -new Date().getTimezoneOffset();
    const sign = off >= 0 ? '+' : '-';
    const h = Math.floor(Math.abs(off) / 60);
    const m = Math.abs(off) % 60;
    return 'GMT' + sign + h + (m ? ':' + String(m).padStart(2,'0') : '');
  }

  // Convert "HH:MM" UTC on dateISO to browser local time "HH:MM"
  function toLocalTime(dateISO, timeUTC) {
    if (!timeUTC) return 'All Day';
    const [h, m] = timeUTC.split(':').map(Number);
    const d = new Date(Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), h, m
    ));
    return d.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit', hour12:false });
  }

  // "2026-05-28" → "Thursday, May 28"  (in the browser's local timezone)
  // dateISO here is already the LOCAL date (output of toLocalDateISO).
  // We parse it with the local Date constructor (no time/zone suffix) so the
  // browser never applies a UTC offset — it reads year/month/day as-is.
  function formatDate(dateISO) {
    const [y, mo, d] = dateISO.split('-').map(Number);
    const dt = new Date(y, mo - 1, d);   // local constructor — no UTC shift
    return dt.toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric' });
  }

  // Return the local-timezone YYYY-MM-DD for an event's UTC datetime.
  // Used to group events under the correct local date header.
  function toLocalDateISO(dateISO, timeUTC) {
    if (!timeUTC) return dateISO;
    const [h, m] = timeUTC.split(':').map(Number);
    const d = new Date(Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), h, m
    ));
    const ly = d.getFullYear();
    const lm = String(d.getMonth() + 1).padStart(2, '0');
    const ld = String(d.getDate()).padStart(2, '0');
    return `${ly}-${lm}-${ld}`;
  }

  // Has this event's datetime already passed?
  function isPastEvent(dateISO, timeUTC) {
    const [h, m] = (timeUTC || '23:59').split(':').map(Number);
    const evMs = Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), h, m
    );
    return evMs < Date.now();
  }

  // Today's date in the browser's local timezone (YYYY-MM-DD)
  function todayISO() {
    const now = new Date();
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, '0');
    const d = String(now.getDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
  }

  // Scroll cal-events-body to a child element (correct inner scroll, not outer panel)
  function scrollCalTo(container, target) {
    if (!target) { container.scrollTop = 0; return; }
    const offset = target.offsetTop - container.offsetTop;
    container.scrollTop = Math.max(0, offset - 2);
  }

  // ── Next-event jump button ────────────────────────────────────────────────
  // Industry standard: floating pill at bottom of calendar that shows
  // the next upcoming high/medium event and jumps to it on click.
  // Hides automatically when the next event is already in view.
  function setupNextEventButton(container, firstUpcomingEl) {
    // Remove any previous instance
    const prev = document.getElementById('cal-next-btn');
    if (prev) prev.remove();

    if (!firstUpcomingEl) return;

    // Read the event label for the button
    const timeEl  = firstUpcomingEl.querySelector('.cal-time');
    const ccyEl   = firstUpcomingEl.querySelector('.cal-ccy');
    const titleEl = firstUpcomingEl.querySelector('.cal-title');
    const dotEl   = firstUpcomingEl.querySelector('.cal-dot');

    const timeStr  = timeEl  ? timeEl.textContent.trim()  : '';
    const ccyStr   = ccyEl  ? ccyEl.textContent.trim()   : '';
    const titleStr = titleEl ? titleEl.textContent.trim() : 'Next event';
    // Truncate title to keep pill compact
    const shortTitle = titleStr.length > 28 ? titleStr.slice(0, 26) + '…' : titleStr;
    const dotColor = dotEl ? dotEl.style.background : 'var(--text3)';

    const btn = document.createElement('button');
    btn.id = 'cal-next-btn';
    btn.title = `Jump to next event: ${titleStr}`;
    btn.setAttribute('aria-label', `Jump to next event: ${titleStr}`);
    btn.innerHTML = `
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${dotColor};margin-right:5px;flex-shrink:0;"></span>
      <span style="color:var(--text2);margin-right:4px;font-family:var(--font-mono);font-size:10px;">${timeStr}</span>
      <span style="color:var(--text2);margin-right:4px;font-size:9px;">${ccyStr}</span>
      <span style="color:var(--text2);font-size:10px;">${shortTitle}</span>
      <span id="cal-next-btn-arrow" style="color:var(--text2);margin-left:5px;font-size:10px;">↓</span>`;
    btn.style.cssText = [
      'position:absolute',
      'bottom:6px',
      'left:50%',
      'transform:translateX(-50%)',
      'display:flex',
      'align-items:center',
      'padding:4px 10px',
      'background:var(--bg3)',
      'border:1px solid var(--border2)',
      'border-radius:12px',
      'cursor:pointer',
      'white-space:nowrap',
      'z-index:10',
      'transition:opacity .15s',
      'opacity:0',
      'pointer-events:none',
    ].join(';');

    // Parent needs position:relative for absolute positioning to work
    const wrapper = container.parentElement;
    if (wrapper) {
      wrapper.style.position = 'relative';
      wrapper.appendChild(btn);
    } else {
      return;
    }

    btn.addEventListener('click', () => {
      // Scroll to the date row just before the first upcoming event
      const prev = firstUpcomingEl.previousElementSibling;
      const target = (prev && prev.classList.contains('cal-date-row')) ? prev : firstUpcomingEl;
      scrollCalTo(container, target);
    });

    // Show/hide based on whether the first upcoming event is visible in the scroll box
    function updateBtnVisibility() {
      const cTop    = container.scrollTop;
      const cBottom = cTop + container.clientHeight;
      const eTop    = firstUpcomingEl.offsetTop - container.offsetTop;
      const eBottom = eTop + firstUpcomingEl.offsetHeight;
      const visible = eTop >= cTop && eBottom <= cBottom + 4;
      btn.style.opacity        = visible ? '0' : '0.92';
      btn.style.pointerEvents  = visible ? 'none' : 'auto';
      // Arrow points toward the next event:
      // If we've scrolled past it (next event is above) → arrow up ↑
      // If we're above it (next event is below, i.e. past events) → arrow down ↓
      const arrowEl = document.getElementById('cal-next-btn-arrow');
      if (arrowEl) arrowEl.textContent = eTop < cTop ? '↑' : '↓';
    }

    container.addEventListener('scroll', updateBtnVisibility, { passive: true });
    // Initial check after layout settles
    requestAnimationFrame(() => requestAnimationFrame(updateBtnVisibility));
  }

  // Institutional-facing source labels never mention backend/pipeline internals
  // (Worker, direct-commit, GitHub Actions, etc.) — Bloomberg/Refinitiv don't
  // expose their data-delivery mechanics in the terminal UI, only the data
  // provider itself. The raw `source` field in ff_calendar.json legitimately
  // carries that extra detail (useful for troubleshooting — it's how the
  // 2026-08-06/07 truncation incident was diagnosed), so it isn't stripped at
  // the source; display just always routes through this sanitizer first.
  // v1.10: strips any trailing parenthetical annotation. Handles today's one
  // offender (calendar-watcher.js's direct-commit fallback label) and any
  // future one following the same "Label (pipeline detail)" convention used
  // elsewhere in this Worker (e.g. DIRECT_COMMIT_SOURCE_LABEL for quotes.json).
  function cleanSourceLabel(raw) {
    if (!raw) return 'Myfxbook · ForexFactory';
    const stripped = String(raw).replace(/\s*\([^)]*\)\s*$/, '').trim();
    return stripped || 'Myfxbook · ForexFactory';
  }

  function buildPanel(events, source, holidays) {
    source   = cleanSourceLabel(source);
    holidays = holidays || [];
    const container = document.getElementById('cal-events-body');
    const sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;
    ensureLiveStyles();          // [TEST v1.14]
    ensureMethodologyTooltip();  // [TEST v1.14]
    ensureHistModal();           // [TEST v1.16]
    _calRenderIndex = [];        // [TEST v1.16] reset per render — see setupHistModalDelegation()

    // Display window: 3 days back through 14 days ahead, shifted by
    // [TEST v1.16] _calWeekOffsetDays (±7 per Prev/Next week click; 0 = the
    // real current window, same default as before week navigation existed).
    // ff_calendar.json carries 21 days of history for actuals backfill.
    // Industry standard (Bloomberg, Refinitiv Eikon): economic calendar panels
    // show 2–3 prior sessions alongside the current day and forward events.
    // 3-day lookback ensures Friday's COT-adjacent releases remain visible on
    // Monday morning and covers overnight JPY/AUD releases that display under
    // the prior local date for users in UTC-ahead timezones.
    const _now       = new Date();
    const nowMs      = _now.getTime(); // [TEST v1.14] shared by live-highlight weighting
    const _lookback  = new Date(_now); _lookback.setDate(_now.getDate() - 3 + _calWeekOffsetDays);
    const _maxAhead  = new Date(_now); _maxAhead.setDate(_now.getDate() + 14 + _calWeekOffsetDays);
    const _yISO = _lookback.toISOString().slice(0, 10);
    const _mISO = _maxAhead.toISOString().slice(0, 10);

    let filtered = events.filter(ev =>
      G8_CURRENCIES.has(ev.currency) && passesImpactFilter(ev) &&      // [TEST v1.16] impact filter
      (_ccyFilter == null || ev.currency === _ccyFilter) &&         // [TEST v1.13] currency filter
      ev.dateISO >= _yISO && ev.dateISO <= _mISO
    );

    // [TEST v1.13] Revision index — built from the FULL unfiltered dataset
    // (not `filtered`) so history outside the display window / currency
    // filter still counts as "the last known actual" for detection.
    const revIdx = buildRevisionIndex(events);

    // [TEST v1.14] Next high-impact release due soon — scoped to `filtered`,
    // so it respects whatever currency is isolated, and [TEST v1.16] only
    // computed for the real current window (offset 0) — "next release"
    // doesn't mean anything while paged into a past/future week.
    const liveTarget = _calWeekOffsetDays === 0 ? findNextHighImpactEvent(filtered, nowMs) : null;

    // Fallback (v3.30): if the pipeline hasn't run for a day or more (e.g. a quiet
    // weekend with no qualifying RSS events), the strict [yesterday, +14d] window can
    // be entirely empty even though the file has recent, valid data. Rather than show
    // "No events available", fall back to the most recent events on file within the
    // G8/impact filter, anchored to the latest available date.
    // [TEST v1.16] Only applies at offset 0 — this fallback exists for pipeline
    // staleness on the real current window, not to backfill a legitimately
    // quiet week the user paged into with Prev/Next.
    if (!filtered.length && _calWeekOffsetDays === 0) {
      const g8 = events.filter(ev => G8_CURRENCIES.has(ev.currency) && passesImpactFilter(ev));
      if (g8.length) {
        const latestISO = g8.reduce((max, ev) => ev.dateISO > max ? ev.dateISO : max, g8[0].dateISO);
        const fallbackFrom = new Date(latestISO + 'T00:00:00Z');
        fallbackFrom.setUTCDate(fallbackFrom.getUTCDate() - 3);
        const fallbackFromISO = fallbackFrom.toISOString().slice(0, 10);
        filtered = g8.filter(ev => ev.dateISO >= fallbackFromISO && ev.dateISO <= latestISO);
      }
    }

    // Build holiday lookup: dateISO → [{title, currency}]
    const holidayByDate = {};
    holidays.forEach(h => {
      if (!h.dateISO) return;
      if (!holidayByDate[h.dateISO]) holidayByDate[h.dateISO] = [];
      holidayByDate[h.dateISO].push(h);
    });

    // Collect all dates that need rendering — use LOCAL date (not UTC dateISO)
    // so that e.g. an event at 01:00 UTC on May 28 shows under May 27 for GMT-3 users.
    const allDates = new Set([
      ...filtered.map(ev => toLocalDateISO(ev.dateISO, ev.timeUTC)),
      ...Object.keys(holidayByDate),
    ]);

    if (!allDates.size) {
      container.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">No events available.</div>';
      return;
    }

    // Group events by LOCAL date
    const byDate = {};
    filtered.forEach(ev => {
      const localDate = toLocalDateISO(ev.dateISO, ev.timeUTC);
      if (!byDate[localDate]) byDate[localDate] = [];
      byDate[localDate].push(ev);
    });

    const today = todayISO();
    const groups = [];

    Array.from(allDates).sort().forEach(dateISO => {
      const dayEvs  = byDate[dateISO] || [];
      const dayHols = holidayByDate[dateISO] || [];
      const isToday = dateISO === today;
      let gHtml = `<div class="cal-date-row" data-date="${dateISO}"${isToday ? ' data-today="1"' : ''}>${formatDate(dateISO)}</div>`;

      // ── Holiday rows ────────────────────────────────────────────────────────
      // One row per holiday entry, shown at the top of the day above economic
      // events. Each row identifies its specific currency and holiday name so
      // users can see exactly which markets are closed.
      dayHols.forEach(hol => {
        const ccy = hol.currency || '';
        const f   = FLAG[ccy] || '';
        const flagHtml = f
          ? `<span class="fi fi-${f}" style="font-size:10px;margin-right:3px;flex-shrink:0;" title="${ccy}"></span>`
          : '';
        const holTitle  = hol.title || 'Bank Holiday';
        const tooltipTx = `${holTitle} — ${ccy} market closed`;
        gHtml += `<div class="cal-event-row cal-holiday-row" title="${tooltipTx}">` +
          `<div class="cal-col cal-time">All Day</div>` +
          `<div class="cal-col cal-ccy">${flagHtml}<span style="font-size:10px;">${ccy}</span></div>` +
          `<div class="cal-col cal-impact"><span class="cal-dot" style="background:var(--text3);" title="Market holiday"></span></div>` +
          `<div class="cal-col cal-title">${holTitle}</div>` +
          `<div class="cal-col cal-num"><span style="color:var(--text3)">—</span></div>` +
          `<div class="cal-col cal-num"><span style="color:var(--text3)">—</span></div>` +
          `<div class="cal-col cal-num"><span style="color:var(--text3)">—</span></div>` +
          `</div>`;
      });

      dayEvs.forEach(ev => {
        const dot        = IMPACT_DOT[ev.impact];
        const flag       = FLAG[ev.currency] || '';
        const flagHtml   = flag ? `<span class="fi fi-${flag}" style="margin-right:4px;font-size:10px;flex-shrink:0;"></span>` : '';
        const isReleased = !!(ev.actual && ev.actual !== '' && ev.actual !== '-');
        const isPast     = isPastEvent(ev.dateISO, ev.timeUTC);
        const dimmed     = isPast && isReleased;

        // Actual coloring — strip "*" suffix before numeric comparison (derived forecast)
        let actualHtml = '<span style="color:var(--text3)">—</span>';
        if (isReleased && ev.actual != null) {
          const forecastRaw = ev.forecast ? String(ev.forecast).replace(/\*$/, '') : null;
          const actualN   = _calParseNum(ev.actual);
          const forecastN = _calParseNum(forecastRaw || ev.previous || '');
          const evTitle   = (ev.title || '').toLowerCase();
          const isInverse = CAL_INVERSE_KW.some(kw => evTitle.includes(kw));
          let cls = '';
          let styleAttr = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            const beat = isInverse ? actualN < forecastN : actualN > forecastN;
            cls = beat ? ' class="up"' : ' class="down"';
            // [TEST v1.13] Surprise-magnitude tiering — mild stays as before,
            // moderate gets bold, strong gets bold + a faint background pill
            // so a large beat/miss (e.g. NFP -23K vs 80K) reads immediately.
            const tier = _surpriseTier(actualN, forecastN);
            if (tier === 'moderate') styleAttr = ' style="font-weight:600;"';
            if (tier === 'strong')   styleAttr = ` style="font-weight:700;background:${beat ? 'rgba(38,166,154,.14)' : 'rgba(239,83,80,.14)'};border-radius:2px;padding:0 3px;"`;
          }
          actualHtml = `<span${cls}${styleAttr}>${ev.actual}</span>`;
        }

        // Derived forecast (suffixed "*"): render in muted color with tooltip
        let forecastHtml;
        if (!ev.forecast) {
          forecastHtml = '<span style="color:var(--text3)">—</span>';
        } else if (String(ev.forecast).endsWith('*')) {
          const displayVal = String(ev.forecast).slice(0, -1); // strip "*" for display
          forecastHtml = `<span style="color:var(--text3)" title="Last known consensus (provider estimate unavailable)">${displayVal}*</span>`;
        } else {
          forecastHtml = `<span style="color:var(--text2)">${ev.forecast}</span>`;
        }
        // [TEST v1.13] Revision marker — small superscript "R" when this
        // event's `previous` doesn't match what was actually printed last
        // time the same series released (detected from history already in
        // the fetched dataset, see buildRevisionIndex()/detectRevision()).
        const revision = ev.previous ? detectRevision(ev, revIdx) : null;
        const revMarkHtml = revision
          ? ` <sup title="Revised from ${revision.old} to ${revision.new}" style="color:var(--orange);font-size:8px;cursor:help;">R</sup>`
          : '';
        const previousHtml = ev.previous
          ? `<span style="color:var(--text3)">${ev.previous}</span>${revMarkHtml}`
          : '<span style="color:var(--text3)">—</span>';

        const localTime = toLocalTime(ev.dateISO, ev.timeUTC);
        const upcomingAttr = (!isPast) ? ' data-upcoming="1"' : '';

        // [TEST v1.14] Live/next-release highlight — at most one row per render.
        const isLiveTarget = !!(liveTarget && liveTarget.ev === ev);
        let liveClass = '';
        let timeCellHtml = localTime;
        if (isLiveTarget) {
          const delta = liveTarget.evMs - nowMs;
          liveClass = delta <= CAL_LIVE_IMMINENT_MS ? ' cal-live-imminent' : ' cal-live-soon';
          timeCellHtml = `<span class="cal-live-countdown" data-live-ms="${liveTarget.evMs}" ` +
            `title="${localTime} local \u2014 next high-impact release">${fmtCountdown(delta)}</span>`;
        }

        // [TEST v1.14] Methodology tooltip — only attached when a known
        // pattern matches; unmatched titles keep the plain native tooltip
        // that was already there.
        // [TEST v1.16] Every title cell also gets a data-cal-hist-idx hook
        // (click → historical drill-down modal), regardless of whether a
        // methodology pattern matched — the modal is useful even without a
        // methodology blurb (still shows history/cadence/reference-pair
        // context). _calRenderIndex is reset at the top of buildPanel() and
        // grown in render order so the click handler can look the exact `ev`
        // object back up by index without re-serializing it into the DOM.
        const methodText  = _calMethodologyFor(ev.title);
        const histIdx     = _calRenderIndex.push(ev) - 1;
        const fomcTag      = _fomcVoterTag(ev.currency, ev.title); // [TEST v1.16]
        const titleInner  = `${ev.title}${fomcTag}`;
        const titleCellHtml = methodText
          ? `<div class="cal-col cal-title" data-cal-tip="1" data-cal-tip-title="${_escAttr(ev.title)}" data-cal-tip-body="${_escAttr(methodText)}" data-cal-hist-idx="${histIdx}" style="cursor:pointer;">${titleInner}</div>`
          : `<div class="cal-col cal-title" title="${_escAttr(ev.title)}" data-cal-hist-idx="${histIdx}" style="cursor:pointer;">${titleInner}</div>`;

        gHtml += `<div class="cal-event-row${dimmed ? ' cal-released' : ''}${liveClass}"${upcomingAttr}>
  <div class="cal-col cal-time">${timeCellHtml}</div>
  <div class="cal-col cal-ccy">${flagHtml}${ev.currency}</div>
  <div class="cal-col cal-impact"><span class="cal-dot" style="background:${dot.color}" title="${dot.label} impact"></span></div>
  ${titleCellHtml}
  <div class="cal-col cal-num">${actualHtml}</div>
  <div class="cal-col cal-num">${forecastHtml}</div>
  <div class="cal-col cal-num">${previousHtml}</div>
</div>`;
      });

      groups.push({ dateISO, html: gHtml, rowCount: 1 + dayHols.length + dayEvs.length });
    });

    // ── Column layout: 1 (docked / narrow fullscreen) or 2 (wide fullscreen) ──
    // Wide monitors in fullscreen have room to show two chronological columns
    // side by side instead of one list stretched edge-to-edge with a big gap
    // between the event text and the actual/forecast/previous numbers — same
    // idea as a newspaper stock table flowing top-to-bottom, left column then
    // right column, rather than one over-wide row.
    const splitCols = shouldSplitCalColumns() && groups.length > 1;
    container.classList.toggle('cal-cols-active', splitCols);
    // [TEST v1.13.2] `? 'none' : ''` (the exact production line) clears the
    // `display` LONGHAND from the inline style rather than restoring it —
    // since #cal-static-col-header has no stylesheet rule of its own (only
    // this inline `display:grid`), the empty string falls through to the
    // div UA default (`block`), silently degrading the header from a grid
    // to plain inline text flow on every render once this function has run
    // once. Confirmed with a standalone DOM check, not a jsdom quirk — this
    // is a live latent bug in production calendar-panel.js too (same line),
    // just easy to miss on a narrow docked panel where block-flow and a
    // narrow grid look similar at a glance; it's obvious once the 8th
    // "auto" filter-button column is added, which is what surfaced it here.
    // Restoring the explicit value instead of clearing it keeps the exact
    // same production layout intent without changing the toggle behavior.
    const staticHdr = document.getElementById('cal-static-col-header');
    if (staticHdr) staticHdr.style.display = splitCols ? 'none' : 'grid';
    document.getElementById('section-tvcalendar')?.classList.toggle('cal-fs-split', splitCols);

    // [TEST v1.13.2] Keep the currency filter visible even when splitCols
    // hides #cal-static-col-header. The two-column layout reuses ONE
    // buildCalColHeaderHtml() string for BOTH .cal-col-wrap headers (see
    // below), so a unique-id control can't live inside it without producing
    // duplicate #cal-ccy-filter nodes. Instead, relocate the SAME DOM node
    // (never cloned, so the delegated click listener + button states from
    // setupCcyFilterUI() keep working untouched) into the panel-head action
    // row while split, and back into the column-header bar once docked or
    // narrow-fullscreen again.
    const ccyBox      = document.getElementById('cal-ccy-filter');
    const headActions = document.getElementById('cal-panel-head-actions');
    if (ccyBox) {
      if (splitCols && headActions) {
        if (ccyBox.parentNode !== headActions) headActions.insertBefore(ccyBox, headActions.firstChild);
        ccyBox.style.borderLeft  = 'none';
        ccyBox.style.borderRight = 'none';
        ccyBox.style.padding     = '0';
        ccyBox.style.marginRight = '0';
      } else if (staticHdr) {
        if (ccyBox.parentNode !== staticHdr) {
          // [TEST v1.13.3] Insert BEFORE the Actual header span (i.e. right
          // after Event), matching the grid-column order in index-test.html —
          // appendChild would put it back after Previous and reintroduce the
          // 1fr/fixed-column misalignment this version fixed.
          const actualSpan = staticHdr.children[4] || null; // 0:Local 1:Ccy 2:· 3:Event 4:Actual
          staticHdr.insertBefore(ccyBox, actualSpan);
        }
        ccyBox.style.borderLeft  = '1px solid var(--border2)';
        ccyBox.style.borderRight = '1px solid var(--border2)';
        ccyBox.style.padding     = '0 8px';
        ccyBox.style.marginRight = '4px';
      }
    }

    let html;
    if (splitCols) {
      const totalRows = groups.reduce((s, g) => s + g.rowCount, 0);
      let acc = 0, splitAt = groups.length;
      for (let i = 0; i < groups.length; i++) {
        const prevAcc = acc;
        acc += groups[i].rowCount;
        if (acc >= totalRows / 2) {
          // Whichever side of this group is closer to an even 50/50 split wins —
          // taking the first group that merely crosses the midpoint (instead of
          // comparing before/after) can produce badly lopsided columns when one
          // date has far more events than its neighbors (e.g. an FOMC day).
          const diffAfter  = Math.abs(acc - totalRows / 2);
          const diffBefore = Math.abs(prevAcc - totalRows / 2);
          splitAt = (diffBefore <= diffAfter) ? i : i + 1;
          break;
        }
      }
      if (splitAt <= 0) splitAt = 1;                           // never leave column 1 empty
      if (splitAt >= groups.length) splitAt = Math.ceil(groups.length / 2); // never leave column 2 empty
      const colHdr  = buildCalColHeaderHtml();
      const col1Html = groups.slice(0, splitAt).map(g => g.html).join('');
      const col2Html = groups.slice(splitAt).map(g => g.html).join('');
      html = `<div class="cal-events-cols">` +
        `<div class="cal-col-wrap">${colHdr}${col1Html}</div>` +
        `<div class="cal-col-wrap">${colHdr}${col2Html}</div>` +
        `</div>`;
    } else {
      html = groups.map(g => g.html).join('');
    }

    // ── Scroll position preservation ──────────────────────────────────────
    // Capture scroll state BEFORE innerHTML wipe so re-renders can restore it.
    // isFirstRender: container has never been populated (data attribute absent).
    // On first render  → smart-scroll to today / first upcoming (see below).
    // On re-renders    → restore the user's exact scrollTop so manual navigation
    //   is never interrupted by the 5-min interval or visibilitychange refresh.
    // In 2-column mode there are two independent scroll containers (one per
    // .cal-col-wrap) instead of one, so scroll state is captured/restored as
    // an array; a layout-mode change (e.g. resizing across the breakpoint)
    // can leave a saved index unmatched, which just falls back to scrollTop 0
    // for that container rather than breaking anything.
    const isFirstRender     = container.dataset.calInitialized !== '1';
    const scrollRootsBefore = container.querySelectorAll('.cal-col-wrap');
    const savedScrollTops   = isFirstRender
      ? []
      : (scrollRootsBefore.length ? Array.from(scrollRootsBefore).map(r => r.scrollTop) : [container.scrollTop]);

    container.innerHTML = html;

    // ── Scroll logic ──────────────────────────────────────────────────────
    // Uses direct scrollTop on the relevant scroll container (cal-events-body,
    // or whichever .cal-col-wrap holds the target row in 2-column mode) —
    // NOT scrollIntoView which would scroll the outer #rightpanel instead.
    //
    // First-render priority order:
    // 1. Today's date row — always anchor on today if the day has events
    // 2. No today section — jump to first upcoming event's date row
    // 3. First future date (next trading day after today)
    // 4. Top (fallback — all events past, no future dates yet loaded)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const todayRow      = container.querySelector('[data-today="1"]');
      const firstUpcoming = container.querySelector('[data-upcoming="1"]');
      const scrollRootFor = el => (el && el.closest('.cal-col-wrap')) || container;

      if (!isFirstRender) {
        // Re-render (5-min refresh or tab focus regain) — restore user's position.
        // Row layout is stable between refreshes (actuals fill in but no rows are
        // inserted above existing ones), so pixel-level scrollTop is reliable.
        const roots = container.querySelectorAll('.cal-col-wrap');
        if (roots.length) {
          roots.forEach((r, i) => { r.scrollTop = savedScrollTops[i] || 0; });
        } else {
          container.scrollTop = savedScrollTops[0] || 0;
        }
      } else {
        // First render — smart-scroll to the most relevant date.
        if (todayRow) {
          scrollCalTo(scrollRootFor(todayRow), todayRow);
        } else if (firstUpcoming) {
          const prev = firstUpcoming.previousElementSibling;
          const target = (prev && prev.classList.contains('cal-date-row')) ? prev : firstUpcoming;
          scrollCalTo(scrollRootFor(firstUpcoming), target);
        } else {
          // Find first future date row
          const allDateRows = container.querySelectorAll('.cal-date-row[data-date]');
          let scrolled = false;
          for (const row of allDateRows) {
            if (row.dataset.date > today) {
              scrollCalTo(scrollRootFor(row), row);
              scrolled = true;
              break;
            }
          }
          if (!scrolled) {
            const roots = container.querySelectorAll('.cal-col-wrap');
            if (roots.length) roots.forEach(r => { r.scrollTop = 0; }); else container.scrollTop = 0;
          }
        }
        // Mark initialized so future re-renders take the restore path.
        container.dataset.calInitialized = '1';
      }

      // Setup "Next event" jump button
      setupNextEventButton(scrollRootFor(firstUpcoming), firstUpcoming);
    }));

    if (sourceEl) {
      // No trailing tzLabel() here — the column-header time cell just below
      // (#cal-th-time) already shows it, right above the time values it labels.
      sourceEl.textContent = `${source} · G10 currencies · medium & high impact`;
    }
    const thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();

    setupCcyFilterUI(); // [TEST v1.13]
    setupImpactFilterUI(); // [TEST v1.16]
    setupWeekNavUI(); // [TEST v1.16]
    setupMethodologyTooltipDelegation(container); // [TEST v1.14] — delegated, no-op after first call
    setupHistModalDelegation(container); // [TEST v1.16] — delegated, no-op after first call
    tickLiveCountdown(); // [TEST v1.14] — paint the correct value immediately, don't wait for the 20s tick
  }

  // ── [TEST v1.13] Currency filter buttons ────────────────────────────────
  // Renders once into #cal-ccy-filter (present in index-test.html, flush
  // right on the column-header row — same visual slot as #corr-window-btns
  // in the Cross-Asset Correlations panel). No-op harmlessly if the
  // container doesn't exist, so this file stays safe to diff against
  // production calendar-panel.js.
  // Style copied verbatim from index.html's #corr-btn-30/60/90 (dark bg3
  // pill, border2 border, text3/white text toggle) rather than the flag-icon
  // pills from v1.13.0 — isolate semantics: clicking a currency shows ONLY
  // that currency; clicking the active one again (or "All") restores all.
  function setupCcyFilterUI() {
    const box = document.getElementById('cal-ccy-filter');
    if (!box) return;

    const btnStyle = active =>
      `font-size:8px;padding:1px 5px;background:var(--bg3);border:1px solid var(--border2);` +
      `color:${active ? '#fff' : 'var(--text3)'};border-radius:2px;cursor:pointer;line-height:1.4;`;

    if (box.dataset.calCcyInit !== '1') {
      box.dataset.calCcyInit = '1';
      box.innerHTML =
        G8_LIST.map(ccy =>
          `<button type="button" class="cal-ccy-btn" data-ccy="${ccy}" style="${btnStyle(_ccyFilter === ccy)}">${ccy}</button>`
        ).join('') +
        `<button type="button" id="cal-ccy-all" style="${btnStyle(_ccyFilter == null)}">All</button>`;

      box.addEventListener('click', (e) => {
        const btn = e.target.closest('button');
        if (!btn) return;
        const ccy = btn.dataset.ccy;
        if (btn.id === 'cal-ccy-all') {
          _ccyFilter = null;
        } else if (_ccyFilter === ccy) {
          _ccyFilter = null; // clicking the already-active currency clears the filter
        } else {
          _ccyFilter = ccy;  // isolate: show ONLY this currency
        }
        saveCcyFilter(_ccyFilter);
        updateCcyFilterButtonStates();
        relayoutCalendar();
      });
    } else {
      updateCcyFilterButtonStates();
    }
  }

  function updateCcyFilterButtonStates() {
    const box = document.getElementById('cal-ccy-filter');
    if (!box) return;
    box.querySelectorAll('.cal-ccy-btn').forEach(b => {
      b.style.color = (_ccyFilter === b.dataset.ccy) ? '#fff' : 'var(--text3)';
    });
    const allBtn = document.getElementById('cal-ccy-all');
    if (allBtn) allBtn.style.color = (_ccyFilter == null) ? '#fff' : 'var(--text3)';
  }

  // ── [TEST v1.16] Impact filter (High only) buttons ──────────────────────
  // Renders into #cal-impact-filter, same button styling as the currency
  // filter above (isolate-style single toggle rather than a group, since
  // there's only one meaningful extra state: "High only" on/off — the
  // baseline is already medium+high, matching the currency filter's
  // convention of styling the ACTIVE state white and inactive text3).
  function setupImpactFilterUI() {
    const box = document.getElementById('cal-impact-filter');
    if (!box) return;

    const btnStyle = active =>
      `font-size:8px;padding:1px 5px;background:var(--bg3);border:1px solid var(--border2);` +
      `color:${active ? '#fff' : 'var(--text3)'};border-radius:2px;cursor:pointer;line-height:1.4;`;

    if (box.dataset.calImpactInit !== '1') {
      box.dataset.calImpactInit = '1';
      box.innerHTML =
        `<button type="button" id="cal-impact-high" style="${btnStyle(_impactHighOnly)}" ` +
        `title="Show only high-impact events">High only</button>`;

      box.addEventListener('click', (e) => {
        const btn = e.target.closest('button');
        if (!btn || btn.id !== 'cal-impact-high') return;
        _impactHighOnly = !_impactHighOnly;
        saveImpactFilter(_impactHighOnly);
        updateImpactFilterButtonStates();
        relayoutCalendar();
      });
    } else {
      updateImpactFilterButtonStates();
    }
  }

  function updateImpactFilterButtonStates() {
    const btn = document.getElementById('cal-impact-high');
    if (btn) btn.style.color = _impactHighOnly ? '#fff' : 'var(--text3)';
  }

  // ── [TEST v1.16] Week navigation (Prev / This week / Next) ──────────────
  // Renders into #cal-week-nav. Not persisted (see _calWeekOffsetDays note
  // above) — always resets to the real current window on reload, same as a
  // Bloomberg/Refinitiv calendar paging forward/back without "remembering"
  // where you left off. Middle button shows the current offset and, when
  // not on the real current window, doubles as a one-click reset to it.
  function setupWeekNavUI() {
    const box = document.getElementById('cal-week-nav');
    if (!box) return;

    const btnStyle = () =>
      `font-size:8px;padding:1px 5px;background:var(--bg3);border:1px solid var(--border2);` +
      `color:var(--text3);border-radius:2px;cursor:pointer;line-height:1.4;`;

    if (box.dataset.calWeekInit !== '1') {
      box.dataset.calWeekInit = '1';
      box.innerHTML =
        `<button type="button" id="cal-week-prev" style="${btnStyle()}" title="Previous week" aria-label="Previous week">&#8249;</button>` +
        `<button type="button" id="cal-week-label" style="${btnStyle()}"></button>` +
        `<button type="button" id="cal-week-next" style="${btnStyle()}" title="Next week" aria-label="Next week">&#8250;</button>`;

      box.addEventListener('click', (e) => {
        const btn = e.target.closest('button');
        if (!btn) return;
        if (btn.id === 'cal-week-prev') _calWeekOffsetDays -= 7;
        else if (btn.id === 'cal-week-next') _calWeekOffsetDays += 7;
        else if (btn.id === 'cal-week-label') _calWeekOffsetDays = 0; // reset shortcut
        else return;
        updateWeekNavUI();
        relayoutCalendar();
      });
    }
    updateWeekNavUI(); // paint the label on first build too, not just on later re-renders
  }

  function weekNavLabel() {
    if (_calWeekOffsetDays === 0) return 'This week';
    const wk = _calWeekOffsetDays / 7;
    return wk > 0 ? `Week +${wk}` : `Week ${wk}`;
  }

  function updateWeekNavUI() {
    const label = document.getElementById('cal-week-label');
    if (!label) return;
    label.textContent = weekNavLabel();
    const atCurrent = _calWeekOffsetDays === 0;
    label.style.color  = atCurrent ? 'var(--text3)' : '#fff';
    label.title         = atCurrent ? '' : 'Back to current window';
  }

  async function fetchEconomicCalendar() {
    try {
      // Cache-bust: GitHub Pages serves via a CDN (Fastly) that can hold an edge
      // copy of the same URL for several minutes independent of the browser's own
      // cache. `cache: 'no-store'` only controls the browser's local cache — it
      // does not force the CDN to revalidate. Bucketing the query string to this
      // panel's own 2-min refresh cadence (mirrors the pattern already used for
      // ./intraday-data/quotes.json) guarantees each poll hits a URL the CDN
      // hasn't served before, so a fresh commit is picked up within one cycle
      // instead of waiting out the CDN's TTL.
      const _cb = '?_=' + Math.floor(Date.now() / 120000);
      const [ffRes, calRes] = await Promise.all([
        fetch('./calendar-data/ff_calendar.json' + _cb, { cache: 'no-store' }).catch(() => null),
        fetch('./calendar-data/calendar.json' + _cb, { cache: 'no-store' }).catch(() => null)
      ]);
      const ffJson  = ffRes?.ok  ? await ffRes.json().catch(() => null)  : null;
      const calJson = calRes?.ok ? await calRes.json().catch(() => null) : null;

      // calendar.json's native schema (fetch_economic_calendar.py) uses `.event`,
      // not `.title` — normalize once here so every downstream consumer (dedup
      // filters and buildPanel's row renderer alike) can rely on `.title` always
      // being present, whichever file an event came from.
      const normalize = ev => { if (ev.title == null && ev.event != null) ev.title = ev.event; return ev; };
      const ffEvents  = (ffJson?.events  || []).map(normalize);
      const calEvents = (calJson?.events || []).map(normalize);

      // [TEST v1.16] calendar.json already carries a full rolling year (see
      // buildSeriesIndex() note above) — this is the ONLY place that data is
      // fetched, so wire it into the module-scope history/series-index vars
      // here, independent of whichever file `events` below ends up using for
      // the main render list. Missed in the original v1.16 edit pass (caught
      // when the cadence tag and history modal were rendering empty for
      // every event, even ones with plenty of real prior releases).
      _lastFullHistory = calEvents;
      _seriesIndex     = buildSeriesIndex(calEvents);

      let events   = ffEvents;
      let source   = ffJson?.source || calJson?.source || 'ForexFactory';
      // holidays only exist in ff_calendar.json (top-level field)
      let holidays = Array.isArray(ffJson?.holidays) ? ffJson.holidays : [];

      // Coverage guard against a repeat of the 2026-08-06/07 truncation incident:
      // ff_calendar.json is meant to carry a ~21-day rolling history, but a
      // direct-commit fallback write once collapsed it to a single day — and
      // because its own history comes from merging against its own prior content,
      // it can never recover that lost history on its own. If ff_calendar.json's
      // events don't reach back at least 2 distinct days before today, treat it
      // as truncated and backfill older days from calendar.json (deduped by
      // currency+date+time+title) instead of silently showing only today.
      const todayISO = new Date().toISOString().slice(0, 10);
      const ffPastDates = new Set(ffEvents.filter(e => e.dateISO < todayISO).map(e => e.dateISO));
      if (ffPastDates.size < 2 && calEvents.length) {
        const seen = new Set(ffEvents.map(e => `${e.currency}|${e.dateISO}|${e.timeUTC || e.hourUTC || ''}|${e.title}`));
        const fill = calEvents.filter(e => !seen.has(`${e.currency}|${e.dateISO}|${e.timeUTC || e.hourUTC || ''}|${e.title}`));
        events = ffEvents.concat(fill);
        if (!ffEvents.length) source = calJson?.source || source;
      }

      // Myfxbook "Sentiment" pseudo-events (e.g. "European Union Myfxbook EURUSD
      // Sentiment") are Myfxbook's own retail-positioning product, not an official
      // macro release — no real consensus forecast, tagged impact="medium" so they
      // pass the impact filter cleanly. fetch_ff_calendar.py v3.35 stops fetching
      // new ones, but ff_calendar.json's 21-day history window can still carry
      // already-fetched entries from before that fix, and calendar.json's history
      // is longer still — filter client-side too so the panel is clean immediately,
      // not just once the data files fully roll off. Mirrors NOISE_KW's 'myfxbook'
      // keyword in dashboard.js / econ-surprises-modal.js (ESI scoring exclusion).
      events = events.filter(ev => !((ev.title || ev.event || '').toLowerCase().includes('myfxbook')));

      // Client-side cross-day dedup: remove phantom "upcoming" entries that duplicate
      // an already-released event within the prior 7 days (same title+currency+timeUTC).
      // Mirrors Step 2e in fetch_ff_calendar.py; handles stale JSON cached before
      // the server-side fix was deployed.
      const _relIdx = {};
      for (const ev of events) {
        if (ev.actual != null || ev.released) {
          const k = (ev.title || ev.event || '') + '|' + ev.currency + '|' + (ev.timeUTC || ev.hourUTC || '');
          (_relIdx[k] = _relIdx[k] || []).push(ev.dateISO);
        }
      }
      events = events.filter(ev => {
        if (ev.actual != null || ev.released) return true;
        const k = (ev.title || ev.event || '') + '|' + ev.currency + '|' + (ev.timeUTC || ev.hourUTC || '');
        const prior = _relIdx[k] || [];
        const evMs = new Date(ev.dateISO).getTime();
        return !prior.some(d => { const diff = (evMs - new Date(d).getTime()) / 86400000; return diff > 0 && diff <= 7; });
      });

      // [TEST v1.15] Opt-in synthetic fixture for the live-countdown feature —
      // see getSyntheticLiveEvent() above. No-op unless ?calDebugLive=1.
      if (calDebugLiveEnabled()) events = [getSyntheticLiveEvent(Date.now())].concat(events);

      _lastEvents = events; _lastSource = source; _lastHolidays = holidays;
      buildPanel(events, source, holidays);
    } catch {
      const c = document.getElementById('cal-events-body');
      if (c) c.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">Calendar unavailable.</div>';
    }
  }

  // ── Fullscreen toggle — DOM-lift, mirrors dashboard.js's chart fullscreen
  // (_lwOpenFullscreen/_lwCloseFullscreen) but with no chart-resize step needed. ──
  let _calFsOriginalParent = null;
  let _calFsOriginalNext   = null;

  // Wide-monitor two-column layout — see the grouping/assembly logic in
  // buildPanel(). Only active in fullscreen; docked panel (180px tall,
  // narrow right-column width) stays single-column regardless of viewport.
  function shouldSplitCalColumns() {
    const overlay = document.getElementById('cal-fullscreen-overlay');
    return !!(overlay && overlay.classList.contains('cal-fs-active') && window.innerWidth >= 1400);
  }

  function buildCalColHeaderHtml() {
    return `<div class="cal-col-header">` +
      `<span>${tzLabel()}</span>` +
      `<span>Ccy</span>` +
      `<span>·</span>` +
      `<span>Event</span>` +
      `<span class="cal-th-num">Actual</span>` +
      `<span class="cal-th-num">Forecast</span>` +
      `<span class="cal-th-num">Previous</span>` +
      `</div>`;
  }

  // Re-render from cache — used when the column layout needs to change
  // (fullscreen open/close, or a resize crossing the 1400px breakpoint while
  // fullscreen is open) without waiting for the next 2-min data poll.
  function relayoutCalendar() {
    if (_lastEvents) buildPanel(_lastEvents, _lastSource, _lastHolidays);
  }

  function openCalFullscreen() {
    const overlay = document.getElementById('cal-fullscreen-overlay');
    const inner   = document.getElementById('cal-fullscreen-inner');
    const panel   = document.getElementById('section-tvcalendar');
    if (!overlay || !inner || !panel) return;
    if (overlay.classList.contains('cal-fs-active')) return;

    _calFsOriginalParent = panel.parentNode;
    _calFsOriginalNext   = panel.nextSibling;

    inner.appendChild(panel);
    overlay.classList.add('cal-fs-active');
    document.body.style.overflow = 'hidden';
    relayoutCalendar();
  }

  function closeCalFullscreen() {
    const overlay = document.getElementById('cal-fullscreen-overlay');
    const panel   = document.getElementById('section-tvcalendar');
    if (!overlay || !overlay.classList.contains('cal-fs-active')) return;

    overlay.classList.remove('cal-fs-active');
    document.body.style.overflow = '';

    if (_calFsOriginalParent && panel) {
      _calFsOriginalParent.insertBefore(panel, _calFsOriginalNext);
    }
    _calFsOriginalParent = null;
    _calFsOriginalNext   = null;
    relayoutCalendar();
  }

  // Debounced resize — only matters while fullscreen is open (relayoutCalendar
  // is a no-op cost otherwise beyond the early-return checks it performs).
  let _calResizeTimer = null;
  window.addEventListener('resize', function () {
    const overlay = document.getElementById('cal-fullscreen-overlay');
    if (!overlay || !overlay.classList.contains('cal-fs-active')) return;
    clearTimeout(_calResizeTimer);
    _calResizeTimer = setTimeout(relayoutCalendar, 150);
  });

  document.getElementById('cal-fs-btn')?.addEventListener('click', openCalFullscreen);
  document.getElementById('cal-fs-close')?.addEventListener('click', closeCalFullscreen);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && document.getElementById('cal-fullscreen-overlay')?.classList.contains('cal-fs-active')) {
      closeCalFullscreen();
    }
  });

  // [TEST v1.14] Smooth countdown — independent of the 2-min data poll, so
  // the live-target timer counts down every 20s instead of jumping in 2-min
  // steps. Cheap no-op when nothing is tagged data-live-ms.
  setInterval(tickLiveCountdown, 20 * 1000);

  // Refresh every 5 minutes so actuals appear shortly after each release
  setInterval(fetchEconomicCalendar, 2 * 60 * 1000);

  // Also refresh immediately when the tab regains focus (user returns to terminal)
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') fetchEconomicCalendar();
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fetchEconomicCalendar);
  } else {
    fetchEconomicCalendar();
  }
})();
