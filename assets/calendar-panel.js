/**
 * calendar-panel.js v1.7 — Native economic calendar renderer
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
 */
(function () {
  'use strict';

  const G8_CURRENCIES      = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD']);
  const IMPACTS = new Set(['medium','high']);

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

  function buildPanel(events, source, holidays) {
    source   = source   || 'ForexFactory';
    holidays = holidays || [];
    const container = document.getElementById('cal-events-body');
    const sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;

    // Display window: 3 days back through 14 days ahead.
    // ff_calendar.json carries 21 days of history for actuals backfill.
    // Industry standard (Bloomberg, Refinitiv Eikon): economic calendar panels
    // show 2–3 prior sessions alongside the current day and forward events.
    // 3-day lookback ensures Friday's COT-adjacent releases remain visible on
    // Monday morning and covers overnight JPY/AUD releases that display under
    // the prior local date for users in UTC-ahead timezones.
    const _now       = new Date();
    const _lookback  = new Date(_now); _lookback.setDate(_now.getDate() - 3);
    const _maxAhead  = new Date(_now); _maxAhead.setDate(_now.getDate() + 14);
    const _yISO = _lookback.toISOString().slice(0, 10);
    const _mISO = _maxAhead.toISOString().slice(0, 10);

    let filtered = events.filter(ev =>
      G8_CURRENCIES.has(ev.currency) && IMPACTS.has(ev.impact) &&
      ev.dateISO >= _yISO && ev.dateISO <= _mISO
    );

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
          const stripNum  = s => s.replace(/&#\d+;/g, '').replace(/[%,KMBT\s]/gi, '');
          const actualN   = parseFloat(stripNum(String(ev.actual)));
          const forecastN = parseFloat(stripNum(String(forecastRaw || ev.previous || '')));
          const evTitle   = (ev.title || '').toLowerCase();
          const isInverse = CAL_INVERSE_KW.some(kw => evTitle.includes(kw));
          let cls = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            const beat = isInverse ? actualN < forecastN : actualN > forecastN;
            cls = beat ? ' class="up"' : ' class="down"';
          }
          actualHtml = `<span${cls}>${ev.actual}</span>`;
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
        const previousHtml = ev.previous
          ? `<span style="color:var(--text3)">${ev.previous}</span>`
          : '<span style="color:var(--text3)">—</span>';

        const localTime = toLocalTime(ev.dateISO, ev.timeUTC);
        const upcomingAttr = (!isPast) ? ' data-upcoming="1"' : '';

        gHtml += `<div class="cal-event-row${dimmed ? ' cal-released' : ''}"${upcomingAttr}>
  <div class="cal-col cal-time">${localTime}</div>
  <div class="cal-col cal-ccy">${flagHtml}${ev.currency}</div>
  <div class="cal-col cal-impact"><span class="cal-dot" style="background:${dot.color}" title="${dot.label} impact"></span></div>
  <div class="cal-col cal-title" title="${ev.title}">${ev.title}</div>
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
    const staticHdr = document.getElementById('cal-static-col-header');
    if (staticHdr) staticHdr.style.display = splitCols ? 'none' : '';
    document.getElementById('section-tvcalendar')?.classList.toggle('cal-fs-split', splitCols);

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
      sourceEl.textContent = `${source} · G10 currencies · medium & high impact · ${tzLabel()}`;
    }
    const thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();
  }

  async function fetchEconomicCalendar() {
    try {
      let events = [];
      let holidays = [];
      let source = 'ForexFactory';
      // Cache-bust: GitHub Pages serves via a CDN (Fastly) that can hold an edge
      // copy of the same URL for several minutes independent of the browser's own
      // cache. `cache: 'no-store'` only controls the browser's local cache — it
      // does not force the CDN to revalidate. Bucketing the query string to this
      // panel's own 2-min refresh cadence (mirrors the pattern already used for
      // ./intraday-data/quotes.json) guarantees each poll hits a URL the CDN
      // hasn't served before, so a fresh commit is picked up within one cycle
      // instead of waiting out the CDN's TTL.
      const _cb = '?_=' + Math.floor(Date.now() / 120000);
      for (const path of ['./calendar-data/ff_calendar.json' + _cb, './calendar-data/calendar.json' + _cb]) {
        const res = await fetch(path, { cache: 'no-store' }).catch(() => null);
        if (!res?.ok) continue;
        const j = await res.json();
        if (j?.events?.length) {
          events = j.events;
          source = j.source || source;
          // holidays only exist in ff_calendar.json (top-level field)
          if (Array.isArray(j.holidays)) holidays = j.holidays;
          break;
        }
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
