/**
 * calendar-panel.js v1.19.20 — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (ForexFactory, G10 currencies, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 *
 * v1.19.20 (2026-08-19): fetchEconomicCalendar poll interval 2min → 90s, as
 *   part of the same-session backend latency audit that also lowered
 *   econ-matrix.js's ECONMX_POLL_MS (was drifted to 3min there vs 2min here
 *   since this file's own v1.3 2026-06-10 reduction — never mirrored). This
 *   file's own calendar.json read carries no third-party rate-limit exposure
 *   (GitHub Pages/CDN, same-origin static file), unlike the upstream
 *   calendar-watcher.js CF Worker poll against Myfxbook, which stays as-is.
 *   Both panels now poll calendar.json on the same 90s cadence again.
 *
 * v1.19.19 (2026-08-13): Reported by the client — NOK and SEK never appeared
 *   in the Economic Calendar panel or its currency filter buttons, despite
 *   the panel's own header already saying "G10 currencies · medium & high
 *   impact." Root cause: this file's own `G8_CURRENCIES`/`G8_LIST` (the
 *   event filter and the filter-button source list) and its local `FLAG`
 *   map were still hardcoded to the original 8 currencies — never extended
 *   when the rest of the pipeline (fetch_ff_calendar.py, calendar-watcher.js)
 *   moved to G10. Fix: renamed to `G10_CURRENCIES`/`G10_LIST`, added
 *   NOK/SEK to both and to `FLAG` (fi-no/fi-se, already used elsewhere on
 *   the site e.g. CB Rate Expectations). See CHANGELOG.md v8.135.3 for the
 *   companion fetch_economic_calendar.py fix (scripts repo) that was
 *   silently dropping NOK/SEK events from calendar.json on every run.
 *
 * v1.19.18 (2026-08-11): FIX — drill-down modal showed "No prior
 *   actual/forecast history for this event in the last year" for USD Core
 *   PPI m/m, Retail Sales m/m, and Core Retail Sales m/m despite each having
 *   a full year of Myfxbook-sourced history on file. Root cause: the v3.38
 *   hybrid architecture's forward-looking days come from ForexFactory's own
 *   JSON, titled with slash notation ("Core PPI m/m"); Myfxbook (today + all
 *   history) titles the identical indicator with concatenated notation
 *   ("Core PPI MoM"). `_calCanonTitle()`/`_calSeriesKey()` treated these as
 *   two unrelated series. Fix: normalise m/m|y/y|q/q → mom|yoy|qoq before
 *   country-prefix stripping. Separately, two indicators use genuinely
 *   different names across vendors (not just notation) — added a small,
 *   manually-verified `_CAL_VENDOR_ALIASES` map for those (Core Retail Sales
 *   ↔ Retail Sales Ex Autos; UK Prelim GDP ↔ GDP Growth Rate QoQ). Left
 *   NZD "Inflation Expectations q/q" (FF) unmerged with Myfxbook's "Business
 *   Inflation Expectations" — these may be two distinct RBNZ/ANZ surveys,
 *   not confirmed as the same release, so not aliased per Guard 8. Must stay
 *   in sync with _canonEsi in dashboard.js and compute_surprise_stats() in
 *   fetch_economic_calendar.py (engine repo).
 *
 * v1.19.17 (2026-08-09): Panel subtitle no longer names the underlying data
 *   vendor. Was `${source} · G10 currencies · medium & high impact` (e.g.
 *   "Myfxbook · ForexFactory · G10 currencies..."); now just "G10 currencies
 *   · medium & high impact" — matches about.html's Data Sources table, which
 *   never named a vendor for the Economic Calendar row either. The client's
 *   call: institutional terminals (Bloomberg, Refinitiv) don't disclose their
 *   calendar data provider in the live UI, only the coverage. The `source`
 *   field itself is untouched in ff_calendar.json/calendar.json and still
 *   flows through cleanSourceLabel() and _lastSource for internal use — this
 *   is a display-only change, not a data-pipeline change. index.html's static
 *   default subtitle updated to match (was "ForexFactory · G10 major
 *   currencies..."); guide-dashboard.html's Economic Calendar section updated
 *   to describe the panel the same way.
 *
 * v1.19.16 (2026-08-08): FIX — v1.19.15's fontFamily fix didn't resolve it
 *   either. Rather than propose a thirteenth resize/DPR/font theory, pulled
 *   the actual axis-canvas bitmaps directly (canvas.toBlob(), not a
 *   screenshot) from both this chart and corr-modal.js's (a chart that's
 *   never shown this issue) and compared them side by side at identical
 *   zoom. Both are structurally identical: 1x backing store, same font,
 *   same 9px size — that pattern was never the bug. The real difference is
 *   label length: this chart's custom tick formatter shows a full "Jan 9
 *   2026" (day+month+year, ~10 characters); corr-modal.js shows "Jun 26"
 *   (~6 characters). At a 9px line height, more characters packed into
 *   comparable label width leaves less room per glyph, reading as denser/
 *   blockier — not a rendering defect, a legibility limit of cramming that
 *   much text that small. The client asked to keep the day visible rather
 *   than shorten the format, so fixed the other side of the trade-off
 *   instead: bumped fontSize from 9 to 10, matching econ-surprises-modal.js
 *   and cot-modal-chart.js's precedent (neither of which has ever shown
 *   this issue either) — giving each glyph more vertical resolution without
 *   dropping any information from the label.
 *
 * v1.19.15 (2026-08-08): FIX (did not resolve it — see v1.19.16 above) —
 *   ResizeObserver cascade didn't change the visual result either. Stepped
 *   back from resize/DPR theories entirely and rechecked the four LWC chart
 *   configs side by side for anything unrelated to sizing. Found it: this
 *   file is the only one of the four that never sets `fontFamily` in
 *   `layout`. econ-surprises-modal.js, cot-modal-chart.js, and
 *   corr-modal.js — the three unaffected charts — all explicitly set it to
 *   `'JetBrains Mono','Courier New',monospace` (a webfont actually loaded
 *   on the page, confirmed via document.fonts in an earlier diagnostic
 *   dump). Without it, LWC falls back to its own built-in default font
 *   stack, which may not be installed on the client's Android device,
 *   forcing a further OS-level substitution — plausibly one with worse
 *   small-size hinting than the explicitly-loaded monospace font the other
 *   three force. Added the same explicit fontFamily here.
 *
 * v1.19.14 (2026-08-08): FIX (did not resolve it — see v1.19.15 above) —
 *   live, without a redeploy: calling `chart.resize(origWidth + 50, 190,
 *   true)` from the console didn't grow the canvas, it collapsed it to
 *   36px. That's the ResizeObserver antipattern — resizing the very element
 *   you're observing, synchronously, inside its own callback, when that
 *   element has no fixed CSS width (`#cal-hist-chart` doesn't), can
 *   re-trigger the observer mid-reflow and cascade to a garbage value
 *   before layout settles. Compared against econ-surprises-modal.js and
 *   corr-modal.js again, more carefully this time: both wrap their actual
 *   resize call in `requestAnimationFrame()`. v1.19.11 copied their
 *   ResizeObserver + staggered-timeout structure but dropped that rAF
 *   wrapper, calling `applyHistResize()` synchronously instead — the one
 *   piece of the reference pattern that specifically exists to prevent this
 *   exact cascade. Restored it. This is also the most likely explanation
 *   for why the axis stayed persistently blocky/pixelated through v1.19.9–
 *   v1.19.13: every ResizeObserver firing (including from our own code)
 *   could have been re-triggering a synchronous resize loop that never let
 *   the canvas settle at a stable, correctly-scaled size. See
 *   applyHistResize.
 *
 * v1.19.13 (2026-08-08): DEBUG (superseded, see v1.19.14 above) — added
 *   fix the axis clipping either (confirmed via a deployment-verification
 *   diagnostic that this time proved v1.19.12 genuinely was running, ruling
 *   out caching as the reason it "didn't work"). Checked LWC's own docs for
 *   what `forceRepaint` actually does: it only controls *when* the resize
 *   happens (synchronous vs. deferred to next frame) — it says nothing
 *   about forcing a backing-store reallocation when width/height are
 *   numerically unchanged from the chart's current size, which is the
 *   actual situation on every one of `applyHistResize`'s calls (the modal's
 *   container never changes size on its own). So v1.19.11 and v1.19.12 were
 *   almost certainly doing the same no-op, just synchronously vs. deferred.
 *   Rather than ship an eleventh blind guess and wait another full
 *   deploy+cache cycle to find out, added `window.__calHistDebug` (chart
 *   instance + axis-canvas pixel-dimension helper) so the DPR/backing-store
 *   hypothesis can be tested interactively from devtools — e.g. resizing to
 *   a genuinely different width to see whether *that* corrects the backing
 *   store, isolating whether the no-op-on-unchanged-size theory is right
 *   before writing the real fix. Remove this hook once closed.
 *
 * v1.19.12 (2026-08-08): FIX (did not resolve it — see v1.19.13 above) —
 *   structure was correct, but it called `chart.applyOptions({width,
 *   height})`, which didn't fix anything: the modal container's width never
 *   actually changes between chart creation and these later calls, so LWC's
 *   internal diffing almost certainly treated it as a no-op. Nearest-
 *   neighbor (unsmoothed) zoom into the client's screenshot settled the
 *   question this whole thread kept circling: the axis labels were never
 *   hard-clipped — no rectangular edge, nothing overlapping (checked and
 *   ruled out a border/element sitting over the text too). They're blocky
 *   and pixelated — the unmistakable signature of a low-res canvas bitmap
 *   being upscaled 2x by the compositor to fill a devicePixelRatio=2 box,
 *   matching the 1x-backing-store-equals-CSS-size pattern confirmed twice by
 *   console dumps. Switched `applyHistResize()` to
 *   `chart.resize(w, 190, true)` — the third argument (forceRepaint) exists
 *   specifically to force real canvas reallocation even when width/height
 *   are numerically unchanged, unlike applyOptions()'s diffed update. This
 *   is what v1.19.6 originally used before v1.19.9 removed it on a mistaken
 *   read of a since-superseded diagnostic. See applyHistResize.
 *
 * v1.19.11 (2026-08-08): FIX (superseded by v1.19.12, see above) — reverted
 *   The client asked why calendar-panel.js was the only LWC-based chart in the
 *   frontend showing this clipping and suggested comparing against the
 *   others instead of guessing further. econ-surprises-modal.js,
 *   cot-modal-chart.js, and corr-modal.js — three other modal charts with
 *   date axes — are all unaffected, and none of them use `autoSize`
 *   (dashboard.js has an explicit comment against it: "can mis-size before
 *   first paint"). All three instead use a `ResizeObserver` on the chart
 *   container plus several staggered `setTimeout` calls (60/250/600ms) that
 *   re-apply real `width`/`height` via `chart.applyOptions()` shortly after
 *   creation — giving LWC several automatic chances to reallocate the canvas
 *   backing store against whatever `devicePixelRatio` has actually settled
 *   to by then. calendar-panel.js only had a bare `window.resize` listener,
 *   which does nothing unless the user manually resizes the browser —
 *   never automatically, right after modal-open, when a not-yet-settled DPR
 *   would actually need correcting. Adopted the same ResizeObserver +
 *   staggered-reapply pattern used by the other three. See
 *   _calRenderHistChart / applyHistResize / _calDestroyHistChart.
 *
 * v1.19.10 (2026-08-08): FIX (superseded by v1.19.11, see above) — added
 *   after v1.19.9 (confirmed by the client on a fresh screenshot taken well
 *   after that deploy, on a different event's chart, ruling out the
 *   transitional-frame theory that justified removing the v1.19.6 resize).
 *   A repeat diagnostic dump found the axis canvas clean at rest again and
 *   ruled out every CSS ancestor overflow — but every canvas in the chart
 *   had a 1x backing store on a devicePixelRatio=2 screen. `createChart()`
 *   only reads `window.devicePixelRatio` once, synchronously, right after
 *   the container's `display:none → ''` toggle — a moment where DPR isn't
 *   guaranteed settled. Added `autoSize: true` so LWC's own ResizeObserver-
 *   driven sizing (continuous, not one-shot) owns canvas scaling instead;
 *   `width`/`height` kept only as the documented ResizeObserver-failure
 *   fallback. See _calRenderHistChart.
 *
 * v1.19.9 (2026-08-08): REMOVED a no-longer-justified RAF resize — v1.19.6's
 *   requestAnimationFrame'd `chart.resize(w, 190, true)` was added on a live
 *   dump showing the axis canvas's backing store (342x24) numerically equal
 *   to its CSS size despite devicePixelRatio=2. v1.19.7/8 chased (and ruled
 *   out) glyph-specific clipping instead, and a follow-up diagnostic dump —
 *   full row-by-row pixel brightness of the actual axis canvas — proved the
 *   canvas itself renders with zero clipping at rest: text occupied rows
 *   8-16 of a 24-row canvas, clean margin on both sides, nothing touching
 *   row 0 or row 23. So the DPR/backing-store pattern was confirmed (again)
 *   to be normal LWC v5 behavior, not a bug. With clipping ruled out at
 *   rest, the resize call had no remaining justification and became a
 *   liability: it forces a second layout/redraw pass one frame after the
 *   chart's already-correct initial paint, which on a slower device
 *   (the client's dump was captured via Edge Android remote debugging) can
 *   produce a visible transitional frame — a plausible source for a
 *   screenshot catching mis-rendered text not present in steady state.
 *   Removed outright rather than patched again. See _calRenderHistChart.
 *
 * v1.19.8 (2026-08-08): FOLLOW-UP FIX — v1.19.7 removed the comma after
 *   confirming (via pixel crop) it was clipped at the bottom by descender.
 *   The client's next screenshot showed "Jan 7 '26" still clipped — this time
 *   the apostrophe cut off at the TOP, same row-height-too-tight cause from
 *   the other direction (apostrophes commonly sit near/above cap-height).
 *   `_calFmtDateISO()` now drops the 2-digit-year-with-apostrophe shorthand
 *   entirely for the full 4-digit year ("Jan 9 2026") — built only from
 *   digits + capitalized month abbreviations, the one glyph set confirmed
 *   clean (top and bottom) across both screenshots. See _calFmtDateISO.
 *
 * v1.19.7 (2026-08-08): REAL FIX — chart X-axis clipping was never a
 *   canvas/DPR/height issue (all of v1.19.4-v1.19.6 were chasing the wrong
 *   cause). A pixel-level crop of the client's screenshot showed only the
 *   comma glyph's descender being clipped ("Jan 9, '26" losing its comma),
 *   not the whole label. Built a byte-identical repro of the chart (real
 *   lightweight-charts 5.0.7 + real theme CSS, headless Chromium
 *   screenshotted at deviceScaleFactor 2) to confirm: the comma rendered
 *   fine there, ruling out the v1.19.6 DPR/backing-store theory (that
 *   causes blur on HiDPI, not hard clipping). Root cause is LWC's
 *   time-axis row height leaving no headroom for a descender, which is
 *   font-metric-dependent per OS/browser. Fix: `_calFmtDateISO()` no
 *   longer emits a comma ("Jan 9, '26" -> "Jan 9 '26") — no digit, capitalized
 *   month abbreviation, or apostrophe has a descender, so there's nothing
 *   left to clip regardless of font/DPR. See full note at `_calFmtDateISO`.
 *
 * v1.19.6 (2026-08-08): FIX, diagnostic-confirmed this time — chart X-axis
 *   clipping. The client ran a devtools dump at my request instead of another
 *   screenshot, which ruled out the v1.19.5 hypothesis outright
 *   (`modal.scrollHeight === modal.clientHeight`, 584 === 584 — nothing was
 *   being cut by the modal's `max-height`) and revealed the real cause:
 *   every canvas LWC created inside `#cal-hist-chart` had a backing store
 *   equal to its CSS pixel size instead of scaled by `devicePixelRatio`
 *   (the browser reported DPR=2; the time-axis canvas was 342x24 physical
 *   pixels for a 342x24 CSS-px display box, when it needed 684x48). LWC's
 *   draw calls use DPR-scaled coordinates internally, so content sized for
 *   a 2x canvas was being drawn onto a 1x backing store and hard-clipped at
 *   its edge — exactly the "bottom half of every axis label sliced off"
 *   symptom across all three prior screenshots, and unrelated to any of the
 *   CSS height/max-height theories those attempts were built on. Likely
 *   mechanism: `createChart()` allocates each canvas's backing store
 *   synchronously, before the container's first real paint after the
 *   `display:none -> ''` toggle in `openHistModal()` — layout reads
 *   (`getBoundingClientRect`) are accurate same-tick, but canvas DPR
 *   allocation apparently isn't settled yet at that point. Fix: an explicit
 *   `chart.resize(width, 190, true)` call on the next animation frame
 *   (after a real paint has happened), guarded against the modal having
 *   been closed/reopened in the meantime via the existing `_calHistChart`
 *   identity check. Not independently confirmed visually this session
 *   (still no Chromium egress here), but for the first time this fix is
 *   built directly on a live measurement from the client's own browser rather
 *   than another guess from a screenshot.
 *
 * v1.19.5 (2026-08-08): FOURTH attempt at the chart X-axis clipping —
 *   different diagnosis this time, on desktop where the v1.19.4 mobile
 *   fixes don't apply. The three previous attempts (110→130→156→190px)
 *   all grew `#cal-hist-chart` itself, on the assumption LWC's internal
 *   time-axis pane was competing with the price pane for room inside that
 *   height. That assumption was likely wrong: LWC reserves the time-axis
 *   label row automatically, outside the price pane's `scaleMargins` — it
 *   isn't something those margins trade off against. Adding up the modal's
 *   actual content for a typical event (sticky head ~34px + methodology
 *   text ~30px + cadence tag ~26px + 8-row table ~170px + chart-wrap
 *   ~212px + reference-pair move line ~32px + body padding 20px) lands
 *   right around 520-560px — i.e. almost exactly at the modal's own hard
 *   `max-height:min(560px, 90vh)` cap. On a normal desktop window (90vh
 *   comfortably above 560), that 560px ceiling is what was actually
 *   clipping content, at whatever row/element happened to fall on that
 *   boundary for a given event's text length — which after three rounds of
 *   growing the chart, kept being the chart's own axis row, since the axis
 *   is the last thing rendered before `.ch-move`. That's why more chart
 *   height alone didn't help: it doesn't move the ceiling, only what's
 *   sitting at it. `max-height` raised 560px→680px (still capped by 90vh
 *   on genuinely short/laptop-sized viewports) so a typical event's full
 *   content fits without hitting the scroll boundary at all. Mirrored in
 *   the `max-width:480px` mobile tier (was 560px there too, now matches at
 *   680px capped by `92dvh`/`92vh`). Modal remains scrollable regardless
 *   (`overflow-y:auto` unchanged) as a safety net for unusually long
 *   methodology text. Not independently confirmed visually this session —
 *   same environment limitation as prior attempts — but this is a
 *   different root-cause hypothesis from the three that didn't work, not a
 *   repeat of the same fix.
 *
 * v1.19.4 (2026-08-08): BUG FIX — history modal (#cal-hist-modal) broken on
 *   mobile. Root cause found in dashboard.css, not this file: the global
 *   mobile rule `table { min-width: 480px; }` (added for the FX pairs
 *   table, meant to force horizontal scroll on a table with many columns)
 *   has no selector scoping, so it also applied to `#cal-hist-modal table`
 *   on every viewport ≤900px — i.e. effectively every phone. The modal
 *   itself is capped at `width:min(420px, 100%)` and never grew to match,
 *   so the 480px-wide table overflowed the dialog's border sideways. That
 *   read as clipped/truncated text (e.g. "Previous" header reduced to a
 *   sliver, title cut) in the client's screenshot — it wasn't text clipping,
 *   it was the table physically wider than the box it sat in. Fixed with a
 *   scoped override in dashboard.css (`#cal-hist-modal table { min-width:
 *   unset !important; width:100% !important; }`), mirroring the existing
 *   `#rightpanel table` exclusion already in that same media block. Also
 *   hardened locally in this file (independent of dashboard.css, in case
 *   the two are ever deployed out of sync): `.ch-body { overflow-x:auto }`
 *   as a safety net, plus a `max-width:480px` tier that trims overlay
 *   padding (16px→8px, more usable width on small phones), tightens
 *   table cell padding/font a notch, and switches `max-height` to prefer
 *   `92dvh` (falls back to `92vh` on browsers without `dvh` support) so
 *   the modal sizes against the real visible viewport rather than the
 *   layout viewport some mobile browsers report before the address bar
 *   collapses.
 *
 * v1.19.3 (2026-08-08): Three issues from the client's review — two real bugs,
 *   one more attempt at the still-unresolved chart X-axis clipping:
 *   (1) BUG FIX — some events with data showed no chart at all. Root cause:
 *       LWC's setData() requires strictly ascending, UNIQUE time values;
 *       when the source logs the same release twice under two title
 *       spellings for the same date (found live: CHF "Consumer Confidence"
 *       and "Switzerland Consumer Confidence" both dated 2026-06-15 —
 *       apparently a mid-year title-format change upstream that never got
 *       deduped at the source), `_calCanonTitle()` correctly merges both
 *       rows into one series for the table, but the chart then had two
 *       points on the same date and setData() threw, silently aborting the
 *       whole chart render (axes from createChart() still showed — hence
 *       "grid but no line", not a blank box). `_calRenderHistChart()` now
 *       dedupes by `dateISO` (keeping the last-ingested entry per date)
 *       before building chart points. The table is untouched — it still
 *       shows both raw rows as-is; this is a display-layer resilience fix,
 *       not a fix to the underlying duplicate — see "Flagged, not fixed"
 *       below.
 *   (2) BUG FIX — some events with an obvious real cadence (e.g. USD
 *       Nonfarm Payrolls Private) showed no frequency tag. `inferCadence()`
 *       used mean/stdev (coefficient of variation); a single one-off gap —
 *       either a genuine reporting delay from a year back, or the same
 *       duplicate-date issue as (1) producing a near-zero gap — was enough
 *       to blow the 0.35 CV cutoff on its own, since mean and variance are
 *       both outlier-sensitive. Switched to median gap + median absolute
 *       deviation (MAD), which barely moves for one outlier either
 *       direction. Verified against the real NFP Private series (11
 *       releases, one 76-day gap from a delayed report): old algorithm →
 *       null, new algorithm → "Monthly" (median 28d, relative MAD 0.16).
 *   (3) FOLLOW-UP (third attempt) — chart X-axis dates still clipped after
 *       110→130 (v1.18.0) and 130→156 (v1.19.1). Height increased again,
 *       156→190, `scaleMargins` tightened 0.12/0.12→0.10/0.10, and
 *       `.ch-chart-wrap` given 4px `padding-bottom` as extra headroom
 *       against the modal's own scroll boundary. No CSS `overflow:hidden`
 *       was found anywhere in the chain from `#cal-hist-chart` up to the
 *       scrollable `#cal-hist-modal`, so this still isn't a confirmed root
 *       cause, just a more generous version of the same fix that hasn't
 *       fully worked twice already — flagged below for the client to inspect
 *       directly (computed height / devtools) rather than iterate blind
 *       again.
 *
 * FLAGGED, NOT FIXED — data pipeline: same-date duplicate under two title
 *   spellings (see (1) above) is a real dedup gap in the source data,
 *   outside this file's scope. `fetch_ff_calendar.py` (or wherever the
 *   underlying calendar.json is written) should dedupe by
 *   currency+canonical-title+date, not by raw title string, so a mid-series
 *   title-format change can't produce two rows for one release. Worth
 *   grepping calendar.json for other same-date duplicates across titles —
 *   CHF Consumer Confidence was found by inspection, not an exhaustive
 *   check.
 *
 * v1.19.2 (2026-08-08): BUG FIX — history-modal chart hover tooltip colored
 *   actual-vs-forecast beats/misses the same way for every event, ignoring
 *   `isInverse` (the same flag the print table above it already uses via
 *   `_calBeatClass()`). For an inverse indicator — e.g. the U-6 Unemployment
 *   Rate screenshot the client sent, 7.9% actual vs. 7.7% forecast — a higher
 *   actual is worse, but the tooltip still showed "+0.20 vs. forecast" in
 *   green (beat color) instead of red (miss color), directly contradicting
 *   the "Inverse indicator" note and the correctly-red table row for the
 *   same date sitting right above the chart. `_calRenderHistChart()` now
 *   takes an `isInverse` parameter (threaded through from the same
 *   `openHistModal()` computation the table already uses) and applies the
 *   identical beat/miss rule the table uses: `isInverse ? diff < 0 : diff >
 *   0`. Tooltip also appends "(inverse)" after the delta line so the
 *   direction flip is visible without having to scroll up to the note.
 *   Applies to every inverse-keyword-matched event (`CAL_INVERSE_KW`:
 *   unemployment, unemployed, jobless, claims, deficit), not just this one.
 *
 * v1.19.1 (2026-08-08): Two follow-ups from the client's screenshots after the
 *   v1.19.0 production promotion:
 *   (1) STRUCTURAL — filter-row divider replaced with space-between layout.
 *       `#cal-toolbar` (week nav + impact filter) and `#cal-ccy-filter` no
 *       longer sit side-by-side separated by a border. DOM order flipped —
 *       currency filter first (left), toolbar second (right) — and
 *       `#cal-filter-row` now uses `justify-content:space-between`, so the
 *       gap lands in the middle of the bar instead of being marked by a
 *       divider line. Border-right removed from `#cal-toolbar` in both the
 *       docked and wide-fullscreen split-column layouts (was already `none`
 *       in split mode; now also `none` docked). The relocation logic that
 *       moves both groups into `#cal-panel-head-actions` in split mode keeps
 *       the same left-to-right order (currency, then toolbar).
 *   (2) FIX — history-modal chart X-axis dates still clipped after the
 *       v1.18.0 attempt (110→130px + tickMarkFormatter). The client's
 *       follow-up screenshot showed the bottom tick-label row still cut off.
 *       Chart height increased again, 130→156px (container CSS and the LWC
 *       `createChart` option kept in sync), and `rightPriceScale`'s
 *       `scaleMargins` tightened from 0.15/0.15 to 0.12/0.12 so the price
 *       series claims a little less of the taller total, leaving the time
 *       axis strip more room to render a full, unclipped line of text
 *       regardless of how LWC internally apportions the two panes.
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
 *   DIRECT_COMMIT_SOURCE_LABEL). Found live from the client's screenshot.
 * v1.11 (2026-08-07): TWO BUG FIXES, both surfaced by the same incident.
 *   (1) Duplicate timezone label: the panel subtitle already ends in
 *   `tzLabel()` (e.g. "· GMT-3") AND the column-header row's time column
 *   (#cal-th-time) shows the same `tzLabel()` directly below it — the client
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
 * v1.19.0 (2026-08-08): Five fixes from the client's screenshots of the
 *   v1.18.0 chart + toolbar:
 *   (1) Chart background now matches the MODAL's background token
 *       (var(--bg2, var(--bg3)), same as #cal-hist-modal itself), not the
 *       page background (--bg) — those two differ in this theme, which is
 *       why the chart previously rendered as a visibly different-colored
 *       box floating inside the modal.
 *   (2) Chart height 110→130 and explicit `tickMarkFormatter` added so the
 *       bottom axis always renders "Mon D, 'YY" — addresses both the
 *       cut-off axis labels and the missing year in one change.
 *   (3) NEW hover tooltip (`.ch-chart-tooltip`, same positioning/flip
 *       pattern as econ-surprises-modal.js's `.esm-lw-tooltip`) — shows
 *       date-with-year, actual, forecast, and the beat/miss delta for the
 *       point under the cursor. Previously hovering only surfaced LWC's
 *       default price-axis crosshair label, which carries neither the date
 *       nor both series.
 *   (4) FIXED double divider between "High only" and the currency filter —
 *       `#cal-toolbar` had `border-right` AND `#cal-ccy-filter` had
 *       `border-left` on the touching edge, drawing two 1px lines a few
 *       pixels apart instead of one. `#cal-ccy-filter`'s left border
 *       removed; the single divider is `#cal-toolbar`'s right border.
 *   (5) STRUCTURAL — `#cal-toolbar` + `#cal-ccy-filter` moved OFF the
 *       `#cal-static-col-header` grid (where v1.17.0 had put them as two
 *       extra `auto` tracks) into a new `#cal-filter-row`, a plain flex row
 *       (wrap:wrap) sitting above the column-header grid. The `auto` tracks
 *       couldn't shrink below their content's natural width, so as the
 *       panel got narrower they started eating into the `1fr` Event column
 *       and eventually pushed the fixed 58px Actual/Forecast/Previous
 *       columns out of alignment with the data rows beneath — the client
 *       flagged this as an approaching-narrow-width failure mode, not yet
 *       an active bug at the panel's normal docked width. A flex row just
 *       wraps onto a second line instead; it can't corrupt a grid it's no
 *       longer part of. `#cal-static-col-header`'s `grid-template-columns`
 *       reverted to the original 7-track production layout. Relocation
 *       into `#cal-panel-head-actions` for wide-fullscreen split-column
 *       mode unchanged in spirit — just targets `#cal-filter-row` instead
 *       of the grid header as the "docked" parent.
 * v1.18.0 (2026-08-08): Two follow-ups from the client's review of the
 *   history modal's DXY reference-pair line (screenshot showed "0.58 pts
 *   vs. 0.58 pts"):
 *   (1) NEW — actual-vs-forecast history chart. Added to the history modal
 *       below the print table: solid line = actual, dashed line = forecast,
 *       last up to 8 releases, ascending left-to-right. Trading
 *       Economics/Investing.com both carry this as a standard element of
 *       their event-history views, which is what the client referenced.
 *       Reuses the exact loader/theming/destroy pattern already established
 *       in econ-surprises-modal.js (own guarded `_calEnsureLWC()` — this
 *       file has no other script tag on the page to piggyback on, since
 *       index.html loads only calendar-panel.js). Chart is
 *       destroyed on modal close and re-guarded with a monotonic open-token
 *       so a slow CDN load can't paint into a modal the user already
 *       navigated away from (the overlay/table DOM nodes are a reused
 *       singleton, not recreated per open, so a naive "did the title
 *       change" check doesn't work here).
 *   (2) INVESTIGATED, not a bug — the "0.58 pts vs. 0.58 pts" match.
 *       Verified independently against DXY's on-disk OHLC (773 daily bars):
 *       NFP release-day avg range is 0.581, all-days avg is 0.585 — a real
 *       ~0.6% difference that both happened to round to the same 2dp value.
 *       `_pairMoveUnit()`'s dxy case bumped from dp:2 to dp:3 so the two
 *       numbers stop looking identical when they aren't. Separately: "pts"
 *       for USD vs "pips" for the other seven currencies is intentional,
 *       not an inconsistency to standardize away — DXY is a weighted basket
 *       index, not a currency pair, and is quoted in index points on every
 *       real venue (ICE, Bloomberg), never pips. Applying pips uniformly
 *       would itself be the non-standard choice.
 * v1.17.0 (2026-08-08): Two follow-ups from the client's review of the
 *   v1.16.0 screenshot:
 *   (1) REMOVED the FOMC voting-member tag entirely — deleted
 *       FOMC_VOTERS_2026 and _fomcVoterTag(). The client flagged that a
 *       hardcoded voter roster requiring manual updates (the annual Jan 1
 *       rotation, plus any Board confirmation changes) isn't worth
 *       maintaining. Nothing else in this file read that tag.
 *   (2) Week nav + impact filter moved out of their own separate toolbar
 *       row (added in v1.16.0) into the existing column-header row,
 *       positioned directly to the left of the currency filter — same
 *       #cal-static-col-header bar the currency filter already lives in,
 *       instead of a whole extra row. #cal-toolbar now travels as a pair
 *       with #cal-ccy-filter through the wide-fullscreen split-column
 *       relocation (always inserted immediately before it, whichever
 *       parent it currently lives in) rather than duplicating that
 *       branch's logic.
 * v1.16.0 (2026-08-08): "Implement everything, industry-standard" round —
 *   The client asked for all viable items from the v1.15.0 idea list, with
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
 * v1.15.0 (2026-08-08): Follow-up per the client's review of v1.14.0:
 *   (1) REMOVED the ESI contribution badge entirely — the client judged it
 *       added more visual noise than value on the row. Deleted
 *       esiContribBadge(), _calCanonEsi(), _CAL_CCY_PFXS, CAL_ESI_NOISE_KW,
 *       CAL_ESI_DECAY_LAMBDA, _lastSurpriseStats, and the surpriseStats
 *       fetch/store step in fetchEconomicCalendar() — nothing else in this
 *       file read that field. The live-countdown highlight and methodology
 *       tooltip from v1.14.0 are unaffected and unchanged.
 *   (2) Synthetic live-countdown fixture: real data rarely has a qualifying
 *       high-impact event sitting inside the countdown window at the exact
 *       moment someone opens the sandbox to look at it, so testing the
 *       feature meant waiting for a real release or scripting a one-off
 *       fixture in a throwaway test harness. Added an opt-in, in-page
 *       fixture instead — append `?calDebugLive=1` to index.html's URL
 *       and a clearly-labeled "[TEST FIXTURE] Non-Farm Payrolls" event is
 *       injected 20 minutes out, seeded once per page load so it counts
 *       down in real time and crosses from the "soon" tier into the
 *       pulsing "imminent" tier ~5 minutes after load — same behavior a
 *       real event would show. No-op with the flag absent; never touches
 *       any fetched JSON. See getSyntheticLiveEvent() / calDebugLiveEnabled().
 * v1.14.0 (2026-08-08): Two "medium effort" enhancements from
 *   The client's original Bloomberg/Refinitiv gap-analysis, built on top of the
 *   v1.13.x currency-filter work (still unshipped to production). [A third,
 *   an ESI contribution badge, shipped in this version too but was removed
 *   in v1.15.0 — see above; left out of this list accordingly.]
 *   (1) Live/next-release highlight: the single soonest unreleased
 *       high-impact event due within the next 3h gets a highlighted row and
 *       its clock time is swapped for a live countdown (ticks every 20s,
 *       independent of the 2-min data poll); inside 15m the row switches to
 *       a stronger pulsing tier. Tooltip on the countdown still shows the
 *       actual local time. Scoped to `filtered`, so it respects whatever
 *       currency is isolated.
 *   (2) Event methodology tooltip: hovering a matched event title (dashed
 *       underline cue, same visual convention as the ATM IV tooltips
 *       The client referenced) shows what it measures and why FX desks watch
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
 *   this fixture only lived in the ad-hoc test harness at the time — v1.15.0
 *   above makes an equivalent fixture a permanent, opt-in part of the sandbox).
 * v1.13.3 (2026-08-08): The client caught a real misalignment in the
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
 * v1.13.2 (2026-08-08): Follow-up per the client's review of v1.13.1 —
 *   two problems, both in the harness/markup, not the filter logic itself:
 *   (a) The header bar did NOT look identical to production. v1.13.1 rebuilt
 *       #cal-static-col-header as a flex wrapper (grid div + button group)
 *       instead of keeping production's own single `display:grid;
 *       grid-template-columns:52px 52px 18px 1fr 58px 58px 58px` rule — an
 *       extra nesting level that changed how "Event"'s 1fr track and the
 *       trailing number columns actually rendered. Reverted to the exact
 *       production grid in index.html, with only one appended `auto`
 *       track at the end for the button group — this file's rendering logic
 *       is unaffected, the fix is markup-only.
 *   (b) In wide-fullscreen 2-column mode (shouldSplitCalColumns()),
 *       #cal-static-col-header — the only place the filter buttons lived —
 *       is hidden entirely (production behavior, untouched). buildPanel()
 *       now relocates the existing #cal-ccy-filter node into
 *       #cal-panel-head-actions (next to the panel title) whenever splitCols
 *       is true, and moves it back when not, so it's never simply gone.
 * v1.13.1 (2026-08-08): Follow-up per the client's review of v1.13.0:
 *   (a) Currency filter changed from multi-select-with-removal to ISOLATE
 *       semantics (click a currency → show ONLY it; click again/All → show
 *       all), and moved from its own pill row to the right edge of the
 *       column-header bar, restyled to match #corr-window-btns (Cross-Asset
 *       Correlations' 30d/60d/90d buttons) instead of rounded flag pills.
 *   (b) Font mismatch in the harness was NOT a bug in this file — index.html
 *       loads Inter/JetBrains Mono via a Google Fonts <link> that
 *       index.html was missing; fixed there, not here.
 * v1.13.0 (2026-08-08): Three "quick win" enhancements requested by
 *   The client to move the panel closer to Bloomberg/Refinitiv conventions, built on
 *   an isolated test copy (calendar-panel.js / index.html) so production
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
 * v1.12.1 (2026-08-08): BUG FIX — #cal-static-col-header's `display:grid` was
 *   getting clobbered to the div UA default (`block`) after the first
 *   buildPanel() render, on every load. `staticHdr.style.display = splitCols
 *   ? 'none' : ''` clears the inline `display` longhand instead of restoring
 *   it, and this element has no stylesheet rule of its own to fall back to —
 *   only the inline `display:grid` set once in index.html's raw markup,
 *   which JS then immediately overwrote. Now explicitly restores `'grid'`
 *   instead of clearing to `''`. Found while building a sandboxed
 *   currency-filter enhancement to this same header (unreleased at the
 *   time, later shipped as v1.13.0 onward below) — this fix is isolated to
 *   the display-toggle line only, no other logic touched.
 *
 * v1.12 (2026-08-07): BUG FIX — Actual-column beat/miss coloring silently
 *   never applied to any currency-amount event (Balance of Trade, Imports,
 *   Exports, Current Account, etc.). The local `stripNum` only removed %,
 *   commas, K/M/B/T and whitespace — it left leading currency symbols
 *   ($, C$, A$, €, ¥...) in place, so `parseFloat("C$3.86B")` (after strip:
 *   "C$3.86") returned NaN, the `!isNaN` guard failed, and `cls` stayed ''.
 *   Found live from the client's screenshots: Canada/US/Australia Balance of
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

  const G10_CURRENCIES     = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK']);
  const G10_LIST            = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];
  const IMPACTS = new Set(['medium','high']);

  // ── [v1.13.0] Currency filter state ──────────────────────────────────
  // Persisted client-side only (localStorage) — narrows what's rendered from
  // the same already-fetched, already-server-filtered (G8 + medium/high
  // impact) dataset.
  // v1.13.1: changed from multi-select-with-removal (clicking a currency
  // hid it) to ISOLATE semantics (clicking a currency shows ONLY that
  // currency; clicking it again — or "All" — restores all). Matches how
  // the client actually wanted to use it and how #corr-window-btns' 30d/60d/90d
  // group behaves (single active selection, not a multi-toggle).
  // null = "show all" (default / initial state).
  const CAL_CCY_FILTER_KEY = 'gi_cal_ccy_filter';
  function loadCcyFilter() {
    try {
      const raw = localStorage.getItem(CAL_CCY_FILTER_KEY);
      if (!raw) return null;
      const v = JSON.parse(raw);
      return (typeof v === 'string' && G10_CURRENCIES.has(v)) ? v : null;
    } catch { return null; }
  }
  function saveCcyFilter(v) {
    try {
      if (v == null) localStorage.removeItem(CAL_CCY_FILTER_KEY);
      else localStorage.setItem(CAL_CCY_FILTER_KEY, JSON.stringify(v));
    } catch {}
  }
  let _ccyFilter = loadCcyFilter(); // string (single ccy) or null (all)

  // ── [v1.16.0] Impact filter (High only) ───────────────────────────────
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

  // ── [v1.16.0] Week navigation ──────────────────────────────────────────
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

  // ── [v1.16.0] Full-year history index (for cadence + drill-down modal) ──
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

  const FLAG = { USD:'us', EUR:'eu', GBP:'gb', JPY:'jp', AUD:'au', CAD:'ca', CHF:'ch', NZD:'nz', NOK:'no', SEK:'se' };

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

  // ── [v1.19.15] Rate-decision keyword list ────────────────────────────
  // Single source of truth for "is this a central-bank policy-rate event" —
  // was previously only inlined once, inside CAL_METHODOLOGY's own kw array
  // (see below), with no other caller able to reuse it. Hoisted out so
  // _calRenderHistChart can key off the same list to render the
  // actual/forecast history chart as a step (stairstep) line for these
  // events instead of a straight-line interpolation — a policy rate is
  // constant between meetings then jumps discretely on the decision date, so
  // a straight diagonal line between two prints (as every other numeric
  // series correctly uses) implies a gradual drift that never happened. This
  // is the industry-standard convention (Bloomberg/Refinitiv rate-path
  // charts are always stepped, never interpolated). CAL_METHODOLOGY's entry
  // below now references this array instead of its own inline copy.
  const CAL_RATE_KW = ['interest rate decision', 'rate decision', 'cash rate', 'official cash rate', 'refinancing rate', 'ocr'];

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

  // ── [v1.13.0] Surprise-magnitude tiering ─────────────────────────────
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

  // ── [v1.16.0] Series canonicalization + history index ────────────────
  // Shared by the cadence tag and the historical drill-down modal (and
  // previously by the now-removed ESI badge — this is a leaner version with
  // no ESI-specific noise list, just the country-prefix strip needed so
  // "United States Non Farm Payrolls" and a hypothetical bare "Non Farm
  // Payrolls" key the same series).
  const _CAL_CCY_PFXS = ['united states ', 'euro area ', 'united kingdom ', 'japan ',
    'australia ', 'canada ', 'switzerland ', 'new zealand ', 'norway ', 'sweden '];
  // [v1.19.18] Two vendors, two vocabularies for the same indicator. FF sometimes
  // uses a different name entirely for an indicator Myfxbook already has a year
  // of history for — not just a notation difference (that's the mom/yoy/qoq
  // normalisation below). Verified manually against both vendors' own definitions
  // before adding — an incorrect pairing here would silently blend two different
  // indicators' history into one series (Guard 8: never merge without a real
  // source check). Only pairs confirmed to be the same underlying release are
  // listed; anything uncertain is left unmerged on purpose. Applied AFTER
  // mom/yoy/qoq normalisation and country-prefix stripping, so keys are bare
  // canonical form. Must stay in sync with _canonEsi in dashboard.js and
  // compute_surprise_stats() in fetch_economic_calendar.py (engine repo).
  const _CAL_VENDOR_ALIASES = {
    // FF "Core Retail Sales m/m" == Myfxbook "Retail Sales Ex Autos MoM" —
    // "core" retail sales is the standard industry term for ex-autos.
    'core retail sales mom': 'retail sales ex autos mom',
    // FF "Prelim GDP q/q" (UK) == Myfxbook "GDP Growth Rate QoQ" — both are
    // the UK's preliminary quarterly GDP print, just named differently.
    'prelim gdp qoq': 'gdp growth rate qoq',
  };
  function _calCanonTitle(t) {
    let s = (t || '').toLowerCase().replace(/\s*\([^)]*\)/g, '').trim();
    // [v1.19.18] Normalise ForexFactory's slash-notation unit suffixes to
    // Myfxbook's concatenated form BEFORE country-prefix stripping. Root
    // cause: the v3.38 hybrid architecture (fetch_ff_calendar.py) sources
    // forward-looking days from ForexFactory's own JSON, which titles events
    // "Core PPI m/m" / "Retail Sales m/m" — Myfxbook (today + all history)
    // titles the identical indicator "Core PPI MoM" / "United States Retail
    // Sales MoM". Without this, every ForexFactory-sourced forward event keys
    // to a series _seriesIndex has never heard of, so its drill-down modal
    // always shows "No prior actual/forecast history" even for indicators
    // with a full year on file — confirmed live for USD Core PPI m/m, Retail
    // Sales m/m, Core Retail Sales m/m (the client, 2026-08-11). Same
    // normalisation _title_keywords() already applies in fetch_ff_calendar.py
    // for its own (unrelated) fuzzy-dedup pass — reused here for the series
    // key instead. Must stay in sync with _canonEsi in dashboard.js and
    // compute_surprise_stats() in fetch_economic_calendar.py (engine repo).
    s = s.replace(/\bm\/m\b/g, 'mom').replace(/\by\/y\b/g, 'yoy').replace(/\bq\/q\b/g, 'qoq');
    for (const p of _CAL_CCY_PFXS) { if (s.startsWith(p)) { s = s.slice(p.length); break; } }
    if (_CAL_VENDOR_ALIASES[s]) s = _CAL_VENDOR_ALIASES[s];
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
  // → real fixed cadence, bucketed by the median gap. High variance (e.g. ad
  // hoc central-bank speeches, one-off reports) → no tag, since a "cadence"
  // label would be misleading. Needs ≥3 prior releases to say anything.
  //
  // v1.19.3: switched from mean/stdev (coefficient of variation) to
  // median/MAD (median absolute deviation). Found live: USD Nonfarm
  // Payrolls Private — genuinely monthly (7 of 8 recent gaps land in
  // 23–36 days) — showed no "Monthly" tag at all. Cause: one 76-day gap a
  // year back (a real reporting delay, not a data bug) was enough on its
  // own to blow the mean/stdev-based coefficient of variation past the 0.35
  // cutoff, since a single outlier disproportionately drags both the mean
  // and the variance. A second, unrelated failure mode shares the same
  // root: a duplicate same-date entry in the source data (e.g. CHF Consumer
  // Confidence logged twice for 2026-06-15 under two title spellings — see
  // the chart-dedup note in _calRenderHistChart()) produces a near-zero gap
  // that has the same distorting effect. Median/MAD is robust to either: a
  // single outlier gap (large or ~zero) barely moves the median, and MAD
  // (median of |gap − median gap|) doesn't square the deviation the way
  // variance does, so it isn't dominated by that one gap either.
  function inferCadence(seriesArr) {
    if (!seriesArr || seriesArr.length < 3) return null;
    const dates = seriesArr.map(e => Date.parse(e.dateISO + 'T00:00:00Z'));
    const gaps = [];
    for (let i = 1; i < dates.length; i++) gaps.push((dates[i] - dates[i - 1]) / 86400000);
    const sorted = gaps.slice().sort((a, b) => a - b);
    const median = n => {
      const mid = Math.floor(n.length / 2);
      return n.length % 2 ? n[mid] : (n[mid - 1] + n[mid]) / 2;
    };
    const gapMedian = median(sorted);
    if (gapMedian <= 0) return null;
    const absDevs = gaps.map(g => Math.abs(g - gapMedian)).sort((a, b) => a - b);
    const mad = median(absDevs);
    // Normalized MAD relative to the median gap — same role as the old CV,
    // just outlier-robust. Threshold kept generous (0.5) since MAD is
    // already a smaller number than stdev for the same spread.
    const relMad = mad / gapMedian;
    if (relMad > 0.5) return null; // still genuinely irregular — don't mislabel
    if (gapMedian <= 10)  return 'Weekly';
    if (gapMedian <= 40)  return 'Monthly';
    if (gapMedian <= 100) return 'Quarterly';
    if (gapMedian <= 200) return 'Semi-Annual';
    if (gapMedian <= 400) return 'Annual';
    return null;
  }

  // ── [v1.14.0] Live / next-release highlight ───────────────────────────
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
  // file needed for this.
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

  // ── [v1.15.0] Synthetic live-countdown fixture ────────────────────────
  // Real data rarely has a qualifying high-impact event sitting inside the
  // 3h/15m live-countdown window at the exact moment someone wants to check
  // the feature. Opt-in only — append ?calDebugLive=1 to
  // index.html's URL — injects one clearly-labeled fake event so the
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

  // ── [v1.14.0] Event methodology tooltips ──────────────────────────────
  // Same pattern the client asked to reuse from the ATM IV tooltips: a clean,
  // named, plain-language explanation on hover — what the release measures
  // and why FX desks watch it — with no backend/pipeline attribution (this
  // is product copy, not sourced from any fetched document, so it carries
  // no citation obligation). Matched by keyword against the canonical title
  // (case-insensitive substring, first match wins — same convention as
  // CAL_INVERSE_KW above). Not exhaustive — G10 headline
  // releases only; anything unmatched falls back to the plain event-name
  // tooltip that was already there.
  const CAL_METHODOLOGY = [
    { kw: ['nonfarm payrolls private', 'private nonfarm payrolls', 'nonfarm employment private', 'private payrolls'],
      text: 'Private-sector change in nonfarm jobs — the same net-jobs concept as headline payrolls, but with government employment stripped out. Watched as a cleaner read on private hiring momentum, since public-sector swings (elections, furloughs, census hiring) can distort the headline number without reflecting the private economy.' },
    { kw: ['non-farm payrolls', 'nonfarm payrolls', 'non farm payrolls', 'employment change'],
      text: 'Net change in jobs outside farming, private households, and nonprofits — includes both private and government employment. The single most-watched US labor print — a big beat/miss can move every USD pair within seconds of release.' },
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
    { kw: CAL_RATE_KW,
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

  // Self-contained tooltip (this file makes no assumption about dashboard.js
  // load order, so a scoped copy of attachRiskTip's visual pattern lives here
  // under its own #cal-tt id rather than reusing window.attachRiskTip).
  // Delegated listeners, bound
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

  // ── [v1.16.0] Historical drill-down modal ─────────────────────────────
  // Click an event title (any event, not just methodology-matched ones) to
  // open a modal with: methodology blurb, cadence tag, and the last up to
  // 8 releases of that exact series (from the
  // full-year history — see buildSeriesIndex()) with the same beat/miss
  // coloring as the main row. Deliberately a click-through, not another
  // inline badge — the client flagged that per-row space is tight (this is
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
    // dp:3 for DXY (was 2) — at 2dp, release-day and typical-day averages
    // frequently round to the same display value (e.g. 0.581 vs 0.585 both
    // showed "0.58 pts"), which reads as a bug even when the underlying
    // numbers genuinely differ. USD intentionally stays in index "pts", not
    // "pips" — DXY is a weighted basket index, not a currency pair, and
    // industry venues (Bloomberg/ICE) quote it in points, never pips.
    if (pairKey === 'dxy')    return { div: 1,      unit: 'pts',  dp: 3 };
    if (pairKey === 'usdjpy') return { div: 0.01,    unit: 'pips', dp: 0 };
    return                         { div: 0.0001,  unit: 'pips', dp: 0 };
  }

  // ── [v1.18.0] Actual-vs-forecast history chart (LWC) ─────────────────
  // the client asked whether an actual-vs-forecast chart in the history modal
  // is industry standard — it is (Trading Economics / Investing.com both
  // show one). Reuses the same loader/theming pattern already established
  // in econ-surprises-modal.js / cot-modal-chart.js: guarded loader (no-op
  // if LWC is already on the page), CSS-var theming, destroy-before-rebuild.
  let _calHistLwcPromise = null;
  function _calEnsureLWC() {
    if (window.LightweightCharts) return Promise.resolve();
    if (_calHistLwcPromise) return _calHistLwcPromise;
    _calHistLwcPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
      s.onload  = resolve;
      s.onerror = () => { _calHistLwcPromise = null; reject(new Error('LWC load failed')); };
      document.head.appendChild(s);
    });
    return _calHistLwcPromise;
  }

  let _calHistChart = null;
  let _calHistOpenToken = 0;
  let _calHistResizeApply = null;
  let _calHistRo = null;
  let _calHistTimers = [];
  function _calDestroyHistChart() {
    _calHistTimers.forEach(id => clearTimeout(id));
    _calHistTimers = [];
    if (_calHistRo) { try { _calHistRo.disconnect(); } catch (_) {} _calHistRo = null; }
    if (_calHistResizeApply) { window.removeEventListener('resize', _calHistResizeApply); _calHistResizeApply = null; }
    if (_calHistChart) { try { _calHistChart.remove(); } catch (_) {} _calHistChart = null; }
  }

  const _CAL_MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function _calFmtDateISO(iso) {
    // "2026-08-07" -> "Aug 7 '26" — always carries the year, since a
    // release history commonly spans a year boundary (this modal shows up
    // to 8 releases, which for a monthly series is 8 months back) and the
    // bare "Aug"/"mar" month-only labels LWC defaults to for sub-year
    // ranges don't disambiguate Aug 2025 from Aug 2026.
    //
    // NO COMMA, NO APOSTROPHE (v1.19.8): v1.19.7 removed the comma after
    // pixel-inspecting a clipped "Jan 9, '26" and confirming the comma's
    // descender was the cause. The client's next screenshot showed the label
    // ("Jan 7 '26") STILL clipped — this time the apostrophe cut off at the
    // TOP. Same root cause from the other direction: an apostrophe glyph
    // commonly sits high (near/above cap-height, sometimes into the
    // ascender zone depending on the font), and LWC's time-axis row height
    // has no headroom above cap-height either, not just below baseline.
    // Digits and capitalized month abbreviations ("Jan", "Aug") are the
    // only glyphs confirmed (via both screenshots) to render with zero
    // clipping, so this drops the 2-digit-year shorthand entirely in favor
    // of the full 4-digit year — same disambiguating information, built
    // only from the already-proven-safe glyph set (digits + caps), so
    // there is nothing left, top or bottom, for LWC's row to clip.
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
    if (!m) return iso;
    const mon = _CAL_MONTH_ABBR[parseInt(m[2], 10) - 1] || m[2];
    return `${mon} ${parseInt(m[3], 10)} ${m[1]}`;
  }

  function _calRenderHistChart(seriesArr, isInverse, isRateEvent) {
    const LWC = window.LightweightCharts;
    const container = document.getElementById('cal-hist-chart');
    if (!LWC || !container) return;
    _calDestroyHistChart();

    // Defensive dedup by dateISO before charting. LWC requires strictly
    // ascending, UNIQUE time values in setData() — two points sharing the
    // same date throw synchronously and abort the whole render (chart shell
    // + axes still show, from createChart() above, but no lines at all).
    // Real cause found live: the source occasionally logs the same release
    // twice under two title spellings for the same date (e.g. "Consumer
    // Confidence" and "Switzerland Consumer Confidence" both dated
    // 2026-06-15 — a title-format change mid-year that _calCanonTitle()
    // correctly merges into one series, but the underlying duplicate row
    // was never deduped upstream). This is a data-pipeline gap, not
    // something to silently paper over in the table (still shows both rows
    // as-is, since that's the raw truth), but the chart can't plot two
    // points on one x-value regardless of cause — keep the LAST occurrence
    // per date (array is ascending, so "last" = the most recently
    // ingested/reformatted version of that date's release).
    const byDateDedup = new Map();
    seriesArr.forEach(h => byDateDedup.set(h.dateISO, h));
    const dedupedArr = Array.from(byDateDedup.values());

    // Ascending (oldest→newest, left-to-right) — seriesArr is already sorted
    // ascending by buildSeriesIndex(); last8 in the caller was reversed for
    // the table's newest-first display, so this re-slices independently.
    const pts = dedupedArr.slice(-8)
      .map(h => ({
        time: h.dateISO,
        actual: _calParseNum(h.actual),
        forecast: _calParseNum(h.forecast ? String(h.forecast).replace(/\*$/, '') : (h.previous || '')),
      }))
      .filter(p => isFinite(p.actual) && isFinite(p.forecast));
    if (pts.length < 2) { container.style.display = 'none'; return; }
    container.style.display = '';

    // Match the MODAL's own background (var(--bg2, var(--bg3)) — see
    // #cal-hist-modal above), not the page background (--bg). Those two
    // tokens differ in this theme, which is why the chart previously
    // rendered as a visibly different-colored box floating inside the modal.
    const _cs    = getComputedStyle(document.documentElement);
    const _bg2   = _cs.getPropertyValue('--bg2').trim();
    const _bg3   = _cs.getPropertyValue('--bg3').trim();
    const _bg    = _bg2 || _bg3 || '#1e222d';
    const _text2 = _cs.getPropertyValue('--text2').trim() || '#9096a0';
    const _brd2  = _cs.getPropertyValue('--bg3').trim() || '#2a2e39';

    const rect = container.getBoundingClientRect();
    const chart = LWC.createChart(container, {
      width: Math.round(rect.width) || container.offsetWidth || 380,
      height: 190,
      layout: { background: { type: 'solid', color: _bg }, textColor: _text2, fontFamily: "'JetBrains Mono','Courier New',monospace", fontSize: 10, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255,255,255,0.04)' } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.10, bottom: 0.10 } },
      timeScale: {
        borderVisible: false, fixRightEdge: true, fixLeftEdge: true, rightOffset: 2,
        tickMarkFormatter: (time) => _calFmtDateISO(typeof time === 'string' ? time : ''),
      },
      crosshair: {
        mode: LWC.CrosshairMode?.Normal ?? 1,
        vertLine: { color: 'rgba(255,255,255,0.2)', style: 2, labelVisible: false },
        horzLine: { color: 'rgba(255,255,255,0.15)', style: 2, labelVisible: true, labelBackgroundColor: _brd2 },
      },
      handleScroll: false, handleScale: false,
    });
    _calHistChart = chart;

    // TEMP DEBUG HOOK (v1.19.13) — lets the client test resize/DPR hypotheses
    // live from devtools without a redeploy cycle per attempt. Remove once
    // the axis-clipping bug is confirmed fixed and closed.
    window.__calHistDebug = {
      chart, container,
      axisCanvas: () => [...container.querySelectorAll('canvas')].filter(c => c.height < 40).sort((a,b) => b.width - a.width)[0],
      dump: () => {
        const c = window.__calHistDebug.axisCanvas();
        return c ? { attrW: c.width, attrH: c.height, cssW: c.style.width, cssH: c.style.height, dpr: window.devicePixelRatio } : null;
      },
    };

    // [v1.19.15] Central-bank rate decisions: a policy rate is constant
    // between meetings then jumps discretely on the decision date — a
    // straight diagonal line between two prints (LWC's default) implies a
    // gradual drift that never happened. Bloomberg/Refinitiv rate-path charts
    // are always stepped, never interpolated; match that convention here.
    // LWC.LineType.WithSteps === 1 — fall back to the literal in case the
    // enum isn't exposed on this LWC build (mirrors the `?? 1` pattern
    // already used for CrosshairMode above).
    const _rateLineType = isRateEvent ? (LWC.LineType?.WithSteps ?? 1) : undefined;

    const actualSeries = chart.addSeries(LWC.LineSeries, {
      color: '#2596ff', lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: true, crosshairMarkerRadius: 3,
      ...(isRateEvent ? { lineType: _rateLineType } : {}),
    });
    actualSeries.setData(pts.map(p => ({ time: p.time, value: p.actual })));

    const forecastSeries = chart.addSeries(LWC.LineSeries, {
      color: 'rgba(144,150,160,0.85)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: true, crosshairMarkerRadius: 3,
      ...(isRateEvent ? { lineType: _rateLineType } : {}),
    });
    forecastSeries.setData(pts.map(p => ({ time: p.time, value: p.forecast })));

    chart.timeScale().fitContent();

    // v1.19.6 added a requestAnimationFrame'd chart.resize(w, 190, true) here
    // based on a live dump showing the axis canvas's backing store (342x24)
    // numerically equal to its CSS size (342px/24px) despite
    // devicePixelRatio=2. v1.19.7/8 chased (and ruled out) glyph-specific
    // clipping instead, and a follow-up diagnostic dump — full row-by-row
    // pixel brightness of the actual axis canvas — proved the canvas itself
    // was rendering with zero clipping at rest for that event. v1.19.9
    // removed the resize call as an unjustified liability.
    //
    // the client then confirmed, on a fresh screenshot taken well after v1.19.9
    // was live (so not a transitional-frame artifact), that a DIFFERENT
    // event's chart (DXY avg daily range) still showed a hard-edged clip —
    // and a repeat diagnostic dump against that specific chart again showed
    // the axis canvas clean at rest AND ruled out every CSS ancestor. v1.19.10
    // tried `autoSize: true`, reasoning LWC read `window.devicePixelRatio`
    // once at creation time in a not-yet-settled moment. The client pointed out
    // this codebase already has an answer for exactly this class of bug:
    // econ-surprises-modal.js, cot-modal-chart.js, and corr-modal.js — three
    // other LWC-based modal charts with date axes, all unaffected by this
    // clipping — none of them use `autoSize` (dashboard.js's own comment
    // warns it "can mis-size before first paint"). Instead all three use a
    // `ResizeObserver` on the container PLUS several staggered `setTimeout`
    // calls (60ms/200-250ms/500-600ms) that re-apply real `width`/`height` via
    // `chart.applyOptions()` shortly after creation. This file only had a
    // bare `window.addEventListener('resize', ...)` (added below, further
    // down) — which does nothing unless the user manually resizes the
    // browser window, so it could never correct a canvas mis-sized against a
    // not-yet-settled DPR at modal-open time. Adopting the same
    // ResizeObserver + staggered-reapply pattern used by the other three
    // modals (v1.19.11) gives LWC several automatic chances, in the seconds
    // right after creation, to reallocate the canvas backing store against
    // whatever `devicePixelRatio` has actually settled to by then.
    // Resize — mirrors econ-surprises-modal.js / cot-modal-chart.js /
    // corr-modal.js's ResizeObserver + staggered-timeout structure, but uses
    // chart.resize(w, h, true) rather than applyOptions({width, height}).
    // v1.19.11 used applyOptions(), v1.19.12 switched to resize(...,true) —
    // neither fixed it. The live console test (v1.19.13's __calHistDebug
    // hook) exposed why: manually calling chart.resize(origWidth+50, ...)
    // didn't grow the canvas — it collapsed to 36px. That's the classic
    // ResizeObserver antipattern: this callback resizes the very element
    // it's observing (#cal-hist-chart has no fixed CSS width, so changing
    // the canvases inside it can change its own measured size), which can
    // re-trigger the observer mid-reflow and cascade to a garbage value
    // before anything settles. econ-surprises-modal.js and corr-modal.js —
    // the two working references this fix was modeled on — both wrap their
    // actual resize call in requestAnimationFrame() specifically to avoid
    // this: it defers the measurement+resize to the next paint, after the
    // browser has already settled the current layout, instead of reacting
    // synchronously inside the observer's own callback. This file copied
    // their ResizeObserver+staggered-timeout structure but dropped that
    // rAF wrapper — restoring it here.
    const applyHistResize = () => {
      requestAnimationFrame(() => {
        if (!_calHistChart) return;
        const r = container.getBoundingClientRect();
        const w = Math.round(r.width) || container.offsetWidth || 380;
        if (w > 0) { try { _calHistChart.resize(w, 190, true); } catch (_) {} }
      });
    };
    if (window.ResizeObserver) {
      _calHistRo = new ResizeObserver(() => applyHistResize());
      _calHistRo.observe(container);
    }
    _calHistTimers = [
      setTimeout(applyHistResize, 60),
      setTimeout(applyHistResize, 250),
      setTimeout(applyHistResize, 600),
    ];

    // Hover tooltip — date (with year) + actual + forecast for the point
    // under the cursor. Same positioning/styling pattern as
    // econ-surprises-modal.js's .esm-lw-tooltip (flips side/above-below to
    // stay inside the container). Previously there was no tooltip at all —
    // the only thing that showed on hover was LWC's default price-axis
    // crosshair label, which carries neither the date nor both series.
    container.style.position = 'relative';
    const tip = document.createElement('div');
    tip.className = 'ch-chart-tooltip';
    container.appendChild(tip);
    const byTime = {};
    pts.forEach(p => { byTime[p.time] = p; });
    const TW = 150, TM = 10;
    chart.subscribeCrosshairMove(param => {
      if (!param?.point || !param.time) { tip.style.display = 'none'; return; }
      const p = byTime[param.time];
      if (!p) { tip.style.display = 'none'; return; }
      const diff = p.actual - p.forecast;
      // Same beat/miss rule as _calBeatClass() (used for the table rows
      // above): for an inverse indicator (unemployment, jobless claims,
      // etc.) a HIGHER actual than forecast is the miss, not the beat, so
      // the color has to flip with `isInverse` too — previously this
      // always colored diff>0 green regardless of the indicator's
      // direction, which showed a 7.9%-vs-7.7% unemployment miss (worse)
      // in the same green used for a genuine beat.
      const beat = diff === 0 ? null : (isInverse ? diff < 0 : diff > 0);
      const col  = beat === null ? _text2 : (beat ? '#26a69a' : '#ef5350');
      tip.innerHTML = `
        <div style="color:var(--text2,#9096a0);margin-bottom:3px;">${_calFmtDateISO(param.time)}</div>
        <div><span style="color:#2596ff;">Actual</span> ${p.actual}</div>
        <div><span style="color:rgba(144,150,160,0.9);">Forecast</span> ${p.forecast}</div>
        <div style="color:${col};margin-top:2px;">${diff >= 0 ? '+' : ''}${diff.toFixed(2)} vs. forecast${isInverse ? ' (inverse)' : ''}</div>
      `;
      tip.style.display = 'block';
      const cW = container.clientWidth || 380;
      const cx = param.point.x, cy = param.point.y;
      const th = tip.offsetHeight || 60;
      const tx = (cx + TM + TW <= cW - 4) ? cx + TM : cx - TM - TW;
      const ty = (cy - th - TM >= 4) ? cy - th - TM : cy + TM;
      tip.style.left = Math.max(0, tx) + 'px';
      tip.style.top  = Math.max(0, ty) + 'px';
    });

    // Window resize listener kept alongside the ResizeObserver above — belt-
    // and-suspenders, matching econ-surprises-modal.js/cot-modal-chart.js/
    // corr-modal.js, which all keep both rather than relying on just one.
    _calHistResizeApply = applyHistResize;
    window.addEventListener('resize', applyHistResize);
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
        width:min(420px, 100%);max-height:min(680px, 90vh);overflow-y:auto;
        font-family:var(--font-ui);color:var(--text);box-sizing:border-box;
      }
      /* Defensive fallback, independent of dashboard.css: this table must
         never be allowed to force itself wider than the modal (see the
         min-width:480px leak fixed in dashboard.css's mobile block — this
         mirrors that fix locally so the file is self-contained if the two
         are ever deployed out of sync). If content ever does need more
         room than a phone screen offers, scroll horizontally inside the
         body rather than silently overflowing the dialog's border. */
      #cal-hist-modal .ch-body { overflow-x:auto; }
      #cal-hist-modal table { min-width:0; }
      @media (max-width: 480px) {
        #cal-hist-overlay { padding:8px; }
        #cal-hist-modal { max-height:min(680px, 92dvh, 92vh); }
        #cal-hist-modal th, #cal-hist-modal td { padding:3px 3px;font-size:9px; }
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
      #cal-hist-modal .ch-chart-wrap { margin-top:10px;padding-top:8px;padding-bottom:4px;border-top:1px solid var(--border2); }
      #cal-hist-modal .ch-chart-title {
        font-size:9px;text-transform:uppercase;letter-spacing:.03em;color:var(--text3);margin-bottom:4px;
        display:flex;align-items:center;gap:10px;
      }
      #cal-hist-modal .ch-chart-legend { display:flex;align-items:center;gap:4px;font-size:9px;text-transform:none;letter-spacing:0; }
      #cal-hist-modal .ch-chart-swatch { display:inline-block;width:8px;height:2px; }
      #cal-hist-chart { height:190px;position:relative; }
      .ch-chart-tooltip {
        position:absolute;display:none;pointer-events:none;
        background:var(--bg2,#1e222d);border:1px solid var(--border2);
        border-radius:4px;padding:6px 8px;font-size:9px;line-height:1.6;
        font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
        color:var(--text,#d1d4dc);z-index:50;
        box-shadow:0 4px 16px rgba(0,0,0,.45);white-space:nowrap;
      }
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
    const close = () => { overlay.style.display = 'none'; _calHistOpenToken++; _calDestroyHistChart(); };
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
    // [v1.19.15] Central-bank rate decisions render the actual/forecast
    // history chart as a step line (see CAL_RATE_KW definition + chart below).
    const isRateEvent = CAL_RATE_KW.some(kw => evTitleLower.includes(kw));

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

    if (last8.length >= 2) {
      html += `<div class="ch-chart-wrap" id="cal-hist-chart-wrap" style="display:none;">
        <div class="ch-chart-title">Actual vs. forecast
          <span class="ch-chart-legend"><span class="ch-chart-swatch" style="background:#2596ff;"></span>Actual</span>
          <span class="ch-chart-legend"><span class="ch-chart-swatch" style="background:rgba(144,150,160,0.85);border-top:1px dashed rgba(144,150,160,0.85);height:0;"></span>Forecast</span>
        </div>
        <div id="cal-hist-chart"></div>
      </div>`;
    }

    html += `<div class="ch-move" id="cal-hist-move">Loading reference-pair context\u2026</div>`;

    bodyEl.innerHTML = html;
    overlay.style.display = 'flex';

    if (last8.length >= 2) {
      // Monotonic token guards against a slow LWC load resolving after the
      // user has already closed the modal or opened a different event —
      // titleEl/textContent comparisons don't work here since the modal's
      // DOM nodes are reused (singleton overlay), not recreated per open.
      const openToken = ++_calHistOpenToken;
      const chartWrap = document.getElementById('cal-hist-chart-wrap');
      _calEnsureLWC().then(() => {
        if (openToken !== _calHistOpenToken || overlay.style.display !== 'flex') return;
        if (chartWrap) chartWrap.style.display = '';
        _calRenderHistChart(seriesArr, isInverse, isRateEvent);
      }).catch(() => { if (chartWrap) chartWrap.style.display = 'none'; });
    } else {
      _calDestroyHistChart();
    }

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

  // ── [v1.13.0] Revision index ─────────────────────────────────────────
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
    ensureLiveStyles();          // [v1.14.0]
    ensureMethodologyTooltip();  // [v1.14.0]
    ensureHistModal();           // [v1.16.0]
    _calRenderIndex = [];        // [v1.16.0] reset per render — see setupHistModalDelegation()

    // Display window: 3 days back through 14 days ahead, shifted by
    // [v1.16.0] _calWeekOffsetDays (±7 per Prev/Next week click; 0 = the
    // real current window, same default as before week navigation existed).
    // ff_calendar.json carries 21 days of history for actuals backfill.
    // Industry standard (Bloomberg, Refinitiv Eikon): economic calendar panels
    // show 2–3 prior sessions alongside the current day and forward events.
    // 3-day lookback ensures Friday's COT-adjacent releases remain visible on
    // Monday morning and covers overnight JPY/AUD releases that display under
    // the prior local date for users in UTC-ahead timezones.
    const _now       = new Date();
    const nowMs      = _now.getTime(); // [v1.14.0] shared by live-highlight weighting
    const _lookback  = new Date(_now); _lookback.setDate(_now.getDate() - 3 + _calWeekOffsetDays);
    const _maxAhead  = new Date(_now); _maxAhead.setDate(_now.getDate() + 14 + _calWeekOffsetDays);
    const _yISO = _lookback.toISOString().slice(0, 10);
    const _mISO = _maxAhead.toISOString().slice(0, 10);

    let filtered = events.filter(ev =>
      G10_CURRENCIES.has(ev.currency) && passesImpactFilter(ev) &&      // [v1.16.0] impact filter
      (_ccyFilter == null || ev.currency === _ccyFilter) &&         // [v1.13.0] currency filter
      ev.dateISO >= _yISO && ev.dateISO <= _mISO
    );

    // [v1.13.0] Revision index — built from the FULL unfiltered dataset
    // (not `filtered`) so history outside the display window / currency
    // filter still counts as "the last known actual" for detection.
    const revIdx = buildRevisionIndex(events);

    // [v1.14.0] Next high-impact release due soon — scoped to `filtered`,
    // so it respects whatever currency is isolated, and [v1.16.0] only
    // computed for the real current window (offset 0) — "next release"
    // doesn't mean anything while paged into a past/future week.
    const liveTarget = _calWeekOffsetDays === 0 ? findNextHighImpactEvent(filtered, nowMs) : null;

    // Fallback (v3.30): if the pipeline hasn't run for a day or more (e.g. a quiet
    // weekend with no qualifying RSS events), the strict [yesterday, +14d] window can
    // be entirely empty even though the file has recent, valid data. Rather than show
    // "No events available", fall back to the most recent events on file within the
    // G8/impact filter, anchored to the latest available date.
    // [v1.16.0] Only applies at offset 0 — this fallback exists for pipeline
    // staleness on the real current window, not to backfill a legitimately
    // quiet week the user paged into with Prev/Next.
    if (!filtered.length && _calWeekOffsetDays === 0) {
      const g10 = events.filter(ev => G10_CURRENCIES.has(ev.currency) && passesImpactFilter(ev));
      if (g10.length) {
        const latestISO = g10.reduce((max, ev) => ev.dateISO > max ? ev.dateISO : max, g10[0].dateISO);
        const fallbackFrom = new Date(latestISO + 'T00:00:00Z');
        fallbackFrom.setUTCDate(fallbackFrom.getUTCDate() - 3);
        const fallbackFromISO = fallbackFrom.toISOString().slice(0, 10);
        filtered = g10.filter(ev => ev.dateISO >= fallbackFromISO && ev.dateISO <= latestISO);
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
            // [v1.13.0] Surprise-magnitude tiering — mild stays as before,
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
        // [v1.13.0] Revision marker — small superscript "R" when this
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

        // [v1.14.0] Live/next-release highlight — at most one row per render.
        const isLiveTarget = !!(liveTarget && liveTarget.ev === ev);
        let liveClass = '';
        let timeCellHtml = localTime;
        if (isLiveTarget) {
          const delta = liveTarget.evMs - nowMs;
          liveClass = delta <= CAL_LIVE_IMMINENT_MS ? ' cal-live-imminent' : ' cal-live-soon';
          timeCellHtml = `<span class="cal-live-countdown" data-live-ms="${liveTarget.evMs}" ` +
            `title="${localTime} local \u2014 next high-impact release">${fmtCountdown(delta)}</span>`;
        }

        // [v1.14.0] Methodology tooltip — only attached when a known
        // pattern matches; unmatched titles keep the plain native tooltip
        // that was already there.
        // [v1.16.0] Every title cell also gets a data-cal-hist-idx hook
        // (click → historical drill-down modal), regardless of whether a
        // methodology pattern matched — the modal is useful even without a
        // methodology blurb (still shows history/cadence/reference-pair
        // context). _calRenderIndex is reset at the top of buildPanel() and
        // grown in render order so the click handler can look the exact `ev`
        // object back up by index without re-serializing it into the DOM.
        const methodText  = _calMethodologyFor(ev.title);
        const histIdx     = _calRenderIndex.push(ev) - 1;
        const titleInner  = ev.title;
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
    // [v1.13.2] `? 'none' : ''` (the exact production line) clears the
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

    // [v1.19.0] Keep the currency filter visible even when splitCols
    // hides #cal-filter-row (see below). The two-column layout reuses ONE
    // buildCalColHeaderHtml() string for BOTH .cal-col-wrap headers, so a
    // unique-id control can't live inside it without producing duplicate
    // #cal-ccy-filter nodes. Instead, relocate the SAME DOM node (never
    // cloned, so the delegated click listener + button states from
    // setupCcyFilterUI() keep working untouched) into the panel-head action
    // row while split, and back into #cal-filter-row once docked or
    // narrow-fullscreen again.
    const ccyBox      = document.getElementById('cal-ccy-filter');
    const headActions = document.getElementById('cal-panel-head-actions');
    const filterRow   = document.getElementById('cal-filter-row');
    if (ccyBox) {
      if (splitCols && headActions) {
        if (ccyBox.parentNode !== headActions) headActions.insertBefore(ccyBox, headActions.firstChild);
        ccyBox.style.borderLeft  = 'none';
        ccyBox.style.padding     = '0';
        ccyBox.style.marginRight = '0';
      } else if (filterRow) {
        if (ccyBox.parentNode !== filterRow) filterRow.appendChild(ccyBox);
        ccyBox.style.borderLeft  = 'none';
        ccyBox.style.padding     = '0';
        ccyBox.style.marginRight = '0';
      }
    }

    // [v1.19.1] Week nav + impact filter toolbar — travels as a pair with
    // #cal-ccy-filter, same rationale as before: always inserted immediately
    // AFTER #cal-ccy-filter's CURRENT parent (already resolved above),
    // rather than duplicating the splitCols/filterRow branching. Order
    // flipped from v1.19.0 (toolbar-then-currency) to currency-then-toolbar
    // per the client's review: no visible divider between the two groups —
    // #cal-filter-row uses justify-content:space-between instead, so the
    // currency filter sits flush left, the toolbar sits flush right, and the
    // gap lands in the middle rather than being marked by a border.
    const toolBox = document.getElementById('cal-toolbar');
    if (toolBox && ccyBox && ccyBox.parentNode) {
      const targetParent = ccyBox.parentNode;
      if (toolBox.parentNode !== targetParent || toolBox.previousSibling !== ccyBox) {
        targetParent.insertBefore(toolBox, ccyBox.nextSibling);
      }
      toolBox.style.borderRight = 'none';
      toolBox.style.padding     = '0';
      toolBox.style.marginRight = '0';
      toolBox.style.marginLeft  = splitCols ? '4px' : '0';
    }

    // #cal-filter-row has nothing left in it once both groups relocate to
    // #cal-panel-head-actions in split mode — hide it so its border doesn't
    // draw as an empty strip.
    if (filterRow) filterRow.style.display = splitCols ? 'none' : 'flex';

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
      // [this session] Vendor name dropped from the subtitle entirely — Bloomberg/
      // Refinitiv don't disclose their calendar data provider in the terminal UI,
      // only the coverage (currencies, impact tiers). Matches about.html's existing
      // "Economic Calendar | G10 currencies · medium & high impact events" row,
      // which never named a vendor either.
      sourceEl.textContent = `G10 currencies · medium & high impact`;
    }
    const thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();

    setupCcyFilterUI(); // [v1.13.0]
    setupImpactFilterUI(); // [v1.16.0]
    setupWeekNavUI(); // [v1.16.0]
    setupMethodologyTooltipDelegation(container); // [v1.14.0] — delegated, no-op after first call
    setupHistModalDelegation(container); // [v1.16.0] — delegated, no-op after first call
    tickLiveCountdown(); // [v1.14.0] — paint the correct value immediately, don't wait for the 20s tick
  }

  // ── [v1.13.0] Currency filter buttons ────────────────────────────────
  // Renders once into #cal-ccy-filter (present in index.html, flush
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
        G10_LIST.map(ccy =>
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

  // ── [v1.16.0] Impact filter (High only) buttons ──────────────────────
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

  // ── [v1.16.0] Week navigation (Prev / This week / Next) ──────────────
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

      // [v1.16.0] calendar.json already carries a full rolling year (see
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

      // [v1.15.0] Opt-in synthetic fixture for the live-countdown feature —
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

  // [v1.14.0] Smooth countdown — independent of the 2-min data poll, so
  // the live-target timer counts down every 20s instead of jumping in 2-min
  // steps. Cheap no-op when nothing is tagged data-live-ms.
  setInterval(tickLiveCountdown, 20 * 1000);

  // Refresh every 5 minutes so actuals appear shortly after each release
  setInterval(fetchEconomicCalendar, 90 * 1000); // v1.19.20: 2min → 90s, matches econ-matrix.js's ECONMX_POLL_MS

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
