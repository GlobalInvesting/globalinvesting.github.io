/**
 * calendar-panel.js — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (Finnhub, G8, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 * TEST FILE — not yet merged into dashboard.js.
 */
(function () {
  'use strict';

  const G8      = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD']);
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

  // "2026-05-22" → "Friday, May 22"
  function formatDate(dateISO) {
    const d = new Date(dateISO + 'T12:00:00Z');
    return d.toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric', timeZone:'UTC' });
  }

  // Has this event's datetime already passed?
  function isPastEvent(dateISO, timeUTC) {
    const [h, m] = (timeUTC || '23:59').split(':').map(Number);
    const evMs = Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), h, m
    );
    return evMs < Date.now();
  }

  // Today's ISO date in UTC
  function todayISO() {
    return new Date().toISOString().slice(0, 10);
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
      <span style="color:var(--text3);margin-right:4px;font-size:9px;">${ccyStr}</span>
      <span style="color:var(--text1);font-size:10px;">${shortTitle}</span>
      <span style="color:var(--text3);margin-left:5px;font-size:10px;">↑</span>`;
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
    }

    container.addEventListener('scroll', updateBtnVisibility, { passive: true });
    // Initial check after layout settles
    requestAnimationFrame(() => requestAnimationFrame(updateBtnVisibility));
  }

  function buildPanel(events, source) {
    source = source || 'Finnhub';
    const container = document.getElementById('cal-events-body');
    const sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;

    const filtered = events.filter(ev =>
      G8.has(ev.currency) && IMPACTS.has(ev.impact)
    );

    if (!filtered.length) {
      container.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">No events available.</div>';
      return;
    }

    // Group by date
    const byDate = {};
    filtered.forEach(ev => {
      if (!byDate[ev.dateISO]) byDate[ev.dateISO] = [];
      byDate[ev.dateISO].push(ev);
    });

    const today = todayISO();
    let html = '';

    Object.keys(byDate).sort().forEach(dateISO => {
      const dayEvs = byDate[dateISO];
      const isToday = dateISO === today;
      html += `<div class="cal-date-row" data-date="${dateISO}"${isToday ? ' data-today="1"' : ''}>${formatDate(dateISO)}</div>`;

      dayEvs.forEach(ev => {
        const dot        = IMPACT_DOT[ev.impact];
        const flag       = FLAG[ev.currency] || '';
        const flagHtml   = flag ? `<span class="fi fi-${flag}" style="margin-right:4px;font-size:10px;flex-shrink:0;"></span>` : '';
        const isReleased = !!(ev.actual && ev.actual !== '' && ev.actual !== '-');
        const isPast     = isPastEvent(ev.dateISO, ev.timeUTC);
        const dimmed     = isPast && isReleased;

        // Actual coloring
        let actualHtml = '<span style="color:var(--text3)">—</span>';
        if (isReleased && ev.actual != null) {
          const actualN   = parseFloat(String(ev.actual).replace(/[%,KMB\s]/gi,''));
          const forecastN = parseFloat(String(ev.forecast || ev.previous || '').replace(/[%,KMB\s]/gi,''));
          let cls = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            cls = actualN > forecastN ? ' class="up"' : ' class="down"';
          }
          actualHtml = `<span${cls}>${ev.actual}</span>`;
        }

        const forecastHtml = ev.forecast
          ? `<span style="color:var(--text2)">${ev.forecast}</span>`
          : '<span style="color:var(--text3)">—</span>';
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

    container.innerHTML = html;

    // ── Scroll logic ──────────────────────────────────────────────────────
    // Uses direct scrollTop on cal-events-body (the overflow:auto container),
    // NOT scrollIntoView which would scroll the outer #rightpanel instead.
    //
    // Priority order:
    // 1. Today's date row — always anchor on today if the day has events
    // 2. No today section — jump to first upcoming event's date row
    // 3. First future date (next trading day after today)
    // 4. Top (fallback — all events past, no future dates yet loaded)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const todayRow      = container.querySelector('[data-today="1"]');
      const firstUpcoming = container.querySelector('[data-upcoming="1"]');

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

      // Setup "Next event" jump button
      setupNextEventButton(container, firstUpcoming);
    }));

    if (sourceEl) {
      sourceEl.textContent = `${source} · G8 · medium & high impact · ${tzLabel()}`;
    }
    const thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();
  }

  async function fetchEconomicCalendar() {
    try {
      let events = [];
      let source = 'Finnhub';
      for (const path of ['./calendar-data/ff_calendar.json', './calendar-data/calendar.json']) {
        const res = await fetch(path).catch(() => null);
        if (!res?.ok) continue;
        const j = await res.json();
        if (j?.events?.length) { events = j.events; source = j.source || source; break; }
      }
      buildPanel(events, source);
    } catch {
      const c = document.getElementById('cal-events-body');
      if (c) c.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">Calendar unavailable.</div>';
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fetchEconomicCalendar);
  } else {
    fetchEconomicCalendar();
  }
})();
