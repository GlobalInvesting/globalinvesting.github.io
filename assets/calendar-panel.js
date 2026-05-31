/**
 * calendar-panel.js — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (Finnhub, 8 major currencies, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 * TEST FILE — not yet merged into dashboard.js.
 */
(function () {
  'use strict';

  const G8_CURRENCIES      = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD']);
  const IMPACTS = new Set(['medium','high']);

  const IMPACT_DOT = {
    high:   { color: 'var(--down)',   label: 'High'   },
    medium: { color: 'var(--orange)', label: 'Medium' },
  };

  const FLAG = { USD:'us', EUR:'eu', GBP:'gb', JPY:'jp', AUD:'au', CAD:'ca', CHF:'ch', NZD:'nz' };

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
    source   = source   || 'Finnhub';
    holidays = holidays || [];
    const container = document.getElementById('cal-events-body');
    const sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;

    const filtered = events.filter(ev =>
      G8_CURRENCIES.has(ev.currency) && IMPACTS.has(ev.impact)
    );

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
    let html = '';

    Array.from(allDates).sort().forEach(dateISO => {
      const dayEvs  = byDate[dateISO] || [];
      const dayHols = holidayByDate[dateISO] || [];
      const isToday = dateISO === today;
      html += `<div class="cal-date-row" data-date="${dateISO}"${isToday ? ' data-today="1"' : ''}>${formatDate(dateISO)}</div>`;

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
        html += `<div class="cal-event-row cal-holiday-row" title="${tooltipTx}">` +
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
          const actualN   = parseFloat(String(ev.actual).replace(/[%,KMB\s]/gi,''));
          const forecastN = parseFloat(String(forecastRaw || ev.previous || '').replace(/[%,KMB\s]/gi,''));
          let cls = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            cls = actualN > forecastN ? ' class="up"' : ' class="down"';
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

        html += `<div class="cal-event-row${dimmed ? ' cal-released' : ''}"${upcomingAttr}>
  <div class="cal-col cal-time">${localTime}</div>
  <div class="cal-col cal-ccy">${flagHtml}${ev.currency}</div>
  <div class="cal-col cal-impact"><span class="cal-dot" style="background:${dot.color}" title="${dot.label} impact"></span></div>
  <div class="cal-col cal-title" title="${ev.title}">${ev.title}</div>
  <div class="cal-col cal-num">${actualHtml}</div>
  <div class="cal-col cal-num">${forecastHtml}</div>
  <div class="cal-col cal-num">${previousHtml}</div>
</div>`;
      });
    });

    // ── Scroll position preservation ──────────────────────────────────────
    // Capture scroll state BEFORE innerHTML wipe so re-renders can restore it.
    // isFirstRender: container has never been populated (data attribute absent).
    // On first render  → smart-scroll to today / first upcoming (see below).
    // On re-renders    → restore the user's exact scrollTop so manual navigation
    //   is never interrupted by the 5-min interval or visibilitychange refresh.
    const isFirstRender  = container.dataset.calInitialized !== '1';
    const savedScrollTop = isFirstRender ? 0 : container.scrollTop;

    container.innerHTML = html;

    // ── Scroll logic ──────────────────────────────────────────────────────
    // Uses direct scrollTop on cal-events-body (the overflow:auto container),
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

      if (!isFirstRender) {
        // Re-render (5-min refresh or tab focus regain) — restore user's position.
        // Row layout is stable between refreshes (actuals fill in but no rows are
        // inserted above existing ones), so pixel-level scrollTop is reliable.
        container.scrollTop = savedScrollTop;
      } else {
        // First render — smart-scroll to the most relevant date.
        if (todayRow) {
          scrollCalTo(container, todayRow);
        } else if (firstUpcoming) {
          const prev = firstUpcoming.previousElementSibling;
          const target = (prev && prev.classList.contains('cal-date-row')) ? prev : firstUpcoming;
          scrollCalTo(container, target);
        } else {
          // Find first future date row
          const allDateRows = container.querySelectorAll('.cal-date-row[data-date]');
          let scrolled = false;
          for (const row of allDateRows) {
            if (row.dataset.date > today) {
              scrollCalTo(container, row);
              scrolled = true;
              break;
            }
          }
          if (!scrolled) container.scrollTop = 0;
        }
        // Mark initialized so future re-renders take the restore path.
        container.dataset.calInitialized = '1';
      }

      // Setup "Next event" jump button
      setupNextEventButton(container, firstUpcoming);
    }));

    if (sourceEl) {
      sourceEl.textContent = `${source} · 8 major currencies · medium & high impact · ${tzLabel()}`;
    }
    const thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();
  }

  async function fetchEconomicCalendar() {
    try {
      let events = [];
      let holidays = [];
      let source = 'Finnhub';
      for (const path of ['./calendar-data/ff_calendar.json', './calendar-data/calendar.json']) {
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
      buildPanel(events, source, holidays);
    } catch {
      const c = document.getElementById('cal-events-body');
      if (c) c.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">Calendar unavailable.</div>';
    }
  }

  // Refresh every 5 minutes so actuals appear shortly after each release
  setInterval(fetchEconomicCalendar, 5 * 60 * 1000);

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
