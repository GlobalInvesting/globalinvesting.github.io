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
 * v1.14-TEST (2026-08-08): SANDBOX — Three "medium effort" enhancements from
 *   Santiago's original Bloomberg/Refinitiv gap-analysis, built on top of the
 *   v1.13.x currency-filter work (still unshipped to production):
 *   (1) Live/next-release highlight: the single soonest unreleased
 *       high-impact event due within the next 3h gets a highlighted row and
 *       its clock time is swapped for a live countdown (ticks every 20s,
 *       independent of the 2-min data poll); inside 15m the row switches to
 *       a stronger pulsing tier. Tooltip on the countdown still shows the
 *       actual local time. Scoped to `filtered`, so it respects whatever
 *       currency is isolated.
 *   (2) ESI contribution badge: a small superscript next to Actual (e.g.
 *       "+1.2" / "-0.8", up/down colored) showing this event's approximate
 *       decay+impact-weighted pull on that currency's 90d Economic Surprise
 *       Index — z-score-normalized when the series has ≥5 history points in
 *       calendar.json's `surpriseStats`, direction-only otherwise. Explicitly
 *       NOT a replica of econ-surprises-modal.js's exact idx100 blend (that
 *       needs the full window's aggregate weights) — a self-contained,
 *       same-sign, same-shape proxy sized for a single-row badge, with the
 *       approximation stated plainly in its own tooltip. No new fetch:
 *       reuses `surpriseStats`, already present in calendar.json, which this
 *       file already fetches for the history-truncation guard (v1.11).
 *   (3) Event methodology tooltip: hovering a matched event title (dashed
 *       underline cue, same visual convention as the ATM IV tooltips
 *       Santiago referenced) shows what it measures and why FX desks watch
 *       it. ~25 G10 headline-release patterns; unmatched titles keep the
 *       plain native tooltip that was already there. Self-contained tooltip
 *       widget (own #cal-tt id) rather than reusing dashboard.js's
 *       attachRiskTip, since this sandbox harness doesn't load dashboard.js
 *       — delegated listeners bound once on #cal-events-body, not per-row,
 *       so re-renders never re-attach or leak handlers.
 *   All three verified via the same jsdom + Chromium smoke-test harness used
 *   for the v1.13.x rounds (docked / narrow-fullscreen / wide-fullscreen-
 *   split), plus a synthetic near-term high-impact fixture event to exercise
 *   the live-countdown path (production data rarely has one sitting exactly
 *   inside the 3h/15m windows at any given moment the harness happens to run).
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

  // Cache of the last successful fetch — lets relayoutCalendar() re-render
  // (e.g. switching between 1 and 2 columns on fullscreen open/close/resize)
  // without a network round-trip.
  let _lastEvents   = null;
  let _lastSource   = null;
  let _lastHolidays = null;

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

  // ── [TEST v1.14] Per-event ESI contribution badge ───────────────────────
  // Santiago already computes a 90d decay-weighted Economic Surprise Index
  // per currency (econ-surprises-modal.js / dashboard.js renderEconSurprises()).
  // Neither ForexFactory nor Myfxbook show how much any single release moved
  // that index — this surfaces a lightweight, badge-sized proxy for that,
  // right on the row, without exposing the internal aggregation pipeline
  // (only the resulting number + a plain-language tooltip).
  //
  // NOT a replica of the exact idx100 blend formula in econ-surprises-modal.js
  // (that requires the full 90d window's aggregate weights to normalize
  // correctly). This is a per-event, self-contained proxy: decay+impact
  // weight × (z-score if the series has ≥5 history points with std>0, else
  // a plain ±1 beat/miss direction) — same inputs, same shape, same sign
  // convention, but not divided by the aggregate window total. Good enough
  // to answer "did this print help or hurt, and roughly how much" at a
  // glance; not good enough to sum badges across a day and reproduce the
  // panel's own ESI number exactly. Documented the same way _surpriseTier's
  // placeholder thresholds are above.
  //
  // Canonical series key + noise filter + decay constant are copied (not
  // imported — this file has no module system) from _canonEsi/NOISE_KW/
  // DECAY_LAMBDA in dashboard.js and _ESM_* in econ-surprises-modal.js.
  // Must stay in sync with those three if the ESI methodology changes.
  const _CAL_CCY_PFXS = ['united states ', 'euro area ', 'united kingdom ', 'japan ',
    'australia ', 'canada ', 'switzerland ', 'new zealand ', 'norway ', 'sweden '];
  function _calCanonEsi(t) {
    let s = t.replace(/\s*\([^)]*\)/g, '').trim();
    for (const p of _CAL_CCY_PFXS) { if (s.startsWith(p)) { s = s.slice(p.length); break; } }
    return s;
  }
  const CAL_ESI_NOISE_KW = [
    'cftc', 'baker hughes', 'rig count', 'auction', 'api weekly',
    'milk auction', "fed's balance sheet", 'reserve balances',
    'redbook', 'ibd/tipp', 'tips auction', 'note auction', 'bond auction',
    'gilt auction', 'jgb auction', 'obligaciones', 'speculative net',
    'nc net position', 'crude oil inventories', 'crude oil imports',
    'distillate', 'gasoline inventorie', 'gasoline production',
    'refinery', 'heating oil', 'natural gas storage',
    'foreign bonds buying', 'foreign investments in japanese',
    'foreign bond investment', 'foreign investment in japan',
    'm2 money', 'm3 money', 'm4 money', 'reserve assets total',
    'cb leading index', 'atlanta fed gdpnow', 'ny fed', 'cleveland cpi',
    'ibd', '3-month bill', '4-week bill', '52-week bill',
    '4-week average', '4-week avg',
    'tic net', 'net long-term tic', 'total net tic',
    'interest rate projection', 'eia crude oil', 'eia crude', 'myfxbook',
  ];
  const CAL_ESI_DECAY_LAMBDA = Math.LN2 / 45; // half-life 45d, mirrors DECAY_LAMBDA

  // Populated from calendar.json's `surpriseStats` field each fetch (see
  // fetchEconomicCalendar()) — {n, mean, std} per "CCY/canonical title" key,
  // computed server-side by fetch_economic_calendar.py. Same field
  // dashboard.js reads into window._ECON_SURPRISE_STATS. Kept as its own
  // module var here (not the shared window global) so this file never
  // depends on dashboard.js having loaded first.
  let _lastSurpriseStats = {};

  function _escAttr(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // actualN/forecastN/isInverse/beat are passed in already computed by the
  // caller (buildPanel's Actual-column block computes the exact same values
  // one line above where this is called — no point recomputing).
  function esiContribBadge(ev, actualN, forecastN, isInverse, beat, nowMs) {
    if (isNaN(actualN) || isNaN(forecastN) || actualN === forecastN) return '';
    const evTitleLower = (ev.title || '').toLowerCase();
    if (CAL_ESI_NOISE_KW.some(kw => evTitleLower.includes(kw))) return '';
    if (!['medium', 'high'].includes(ev.impact)) return '';

    const canon    = _calCanonEsi(evTitleLower);
    const statsKey = `${ev.currency}/${canon}`;
    const stats    = _lastSurpriseStats[statsKey];
    const useZ     = stats && stats.n >= 5 && stats.std > 0;

    const rawSurprise = actualN - forecastN;
    const surprise     = isInverse ? -rawSurprise : rawSurprise;

    const [eh, em] = (ev.timeUTC || '00:00').split(':').map(Number);
    const evMs = Date.UTC(+ev.dateISO.slice(0,4), +ev.dateISO.slice(5,7)-1, +ev.dateISO.slice(8,10), eh, em);
    const ageDays    = Math.max(0, (nowMs - evMs) / 86400000);
    const impactMult = ev.impact === 'high' ? 1.0 : 0.5;
    const w          = Math.exp(-CAL_ESI_DECAY_LAMBDA * ageDays) * impactMult;

    const contrib = useZ ? ((surprise - stats.mean) / stats.std) * w : (beat ? 1 : -1) * w;
    const rounded = Math.round(contrib * 10) / 10;
    if (Math.abs(rounded) < 0.1) return '';

    const sign = rounded > 0 ? '+' : '';
    const cls  = rounded > 0 ? 'up' : 'down';
    const methodNote = useZ
      ? 'normalized against this series\u2019 own historical surprise distribution'
      : 'direction-only — not enough history yet for this series to normalize';
    const tip = `Approx. contribution to ${ev.currency}\u2019s 90d Economic Surprise Index ` +
      `(decay + impact weighted, ${methodNote}). Reference proxy, not the exact panel calculation.`;
    return ` <sup class="${cls}" style="font-size:7px;cursor:help;" title="${_escAttr(tip)}">${sign}${rounded.toFixed(1)}</sup>`;
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

  // ── [TEST v1.14] Event methodology tooltips ──────────────────────────────
  // Same pattern Santiago asked to reuse from the ATM IV tooltips: a clean,
  // named, plain-language explanation on hover — what the release measures
  // and why FX desks watch it — with no backend/pipeline attribution (this
  // is product copy, not sourced from any fetched document, so it carries
  // no citation obligation). Matched by keyword against the canonical title
  // (case-insensitive substring, first match wins — same convention as
  // CAL_INVERSE_KW/CAL_ESI_NOISE_KW above). Not exhaustive — G10 headline
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

    // Display window: 3 days back through 14 days ahead.
    // ff_calendar.json carries 21 days of history for actuals backfill.
    // Industry standard (Bloomberg, Refinitiv Eikon): economic calendar panels
    // show 2–3 prior sessions alongside the current day and forward events.
    // 3-day lookback ensures Friday's COT-adjacent releases remain visible on
    // Monday morning and covers overnight JPY/AUD releases that display under
    // the prior local date for users in UTC-ahead timezones.
    const _now       = new Date();
    const nowMs      = _now.getTime(); // [TEST v1.14] shared by live-highlight + ESI badge weighting
    const _lookback  = new Date(_now); _lookback.setDate(_now.getDate() - 3);
    const _maxAhead  = new Date(_now); _maxAhead.setDate(_now.getDate() + 14);
    const _yISO = _lookback.toISOString().slice(0, 10);
    const _mISO = _maxAhead.toISOString().slice(0, 10);

    let filtered = events.filter(ev =>
      G8_CURRENCIES.has(ev.currency) && IMPACTS.has(ev.impact) &&
      (_ccyFilter == null || ev.currency === _ccyFilter) &&         // [TEST v1.13] currency filter
      ev.dateISO >= _yISO && ev.dateISO <= _mISO
    );

    // [TEST v1.13] Revision index — built from the FULL unfiltered dataset
    // (not `filtered`) so history outside the display window / currency
    // filter still counts as "the last known actual" for detection.
    const revIdx = buildRevisionIndex(events);

    // [TEST v1.14] Next high-impact release due soon — scoped to `filtered`
    // so it respects whatever currency the user has isolated.
    const liveTarget = findNextHighImpactEvent(filtered, nowMs);

    // Fallback (v3.30): if the pipeline hasn't run for a day or more (e.g. a quiet
    // weekend with no qualifying RSS events), the strict [yesterday, +14d] window can
    // be entirely empty even though the file has recent, valid data. Rather than show
    // "No events available", fall back to the most recent events on file within the
    // G8/impact filter, anchored to the latest available date.
    if (!filtered.length) {
      const g8 = events.filter(ev => G8_CURRENCIES.has(ev.currency) && IMPACTS.has(ev.impact));
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
          let esiHtml = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            const beat = isInverse ? actualN < forecastN : actualN > forecastN;
            cls = beat ? ' class="up"' : ' class="down"';
            // [TEST v1.13] Surprise-magnitude tiering — mild stays as before,
            // moderate gets bold, strong gets bold + a faint background pill
            // so a large beat/miss (e.g. NFP -23K vs 80K) reads immediately.
            const tier = _surpriseTier(actualN, forecastN);
            if (tier === 'moderate') styleAttr = ' style="font-weight:600;"';
            if (tier === 'strong')   styleAttr = ` style="font-weight:700;background:${beat ? 'rgba(38,166,154,.14)' : 'rgba(239,83,80,.14)'};border-radius:2px;padding:0 3px;"`;
            // [TEST v1.14] ESI contribution badge — reuses actualN/forecastN/
            // isInverse/beat this block already computed, no recompute.
            esiHtml = esiContribBadge(ev, actualN, forecastN, isInverse, beat, nowMs);
          }
          actualHtml = `<span${cls}${styleAttr}>${ev.actual}</span>${esiHtml}`;
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
        const methodText = _calMethodologyFor(ev.title);
        const titleCellHtml = methodText
          ? `<div class="cal-col cal-title" data-cal-tip="1" data-cal-tip-title="${_escAttr(ev.title)}" data-cal-tip-body="${_escAttr(methodText)}">${ev.title}</div>`
          : `<div class="cal-col cal-title" title="${_escAttr(ev.title)}">${ev.title}</div>`;

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
    setupMethodologyTooltipDelegation(container); // [TEST v1.14] — delegated, no-op after first call
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

      // [TEST v1.14] surpriseStats only exists on calendar.json — keep the
      // last known copy if this poll's calendar.json fetch fails, rather
      // than blanking out every ESI badge on a transient network hiccup.
      if (calJson?.surpriseStats) _lastSurpriseStats = calJson.surpriseStats;

      // calendar.json's native schema (fetch_economic_calendar.py) uses `.event`,
      // not `.title` — normalize once here so every downstream consumer (dedup
      // filters and buildPanel's row renderer alike) can rely on `.title` always
      // being present, whichever file an event came from.
      const normalize = ev => { if (ev.title == null && ev.event != null) ev.title = ev.event; return ev; };
      const ffEvents  = (ffJson?.events  || []).map(normalize);
      const calEvents = (calJson?.events || []).map(normalize);

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
