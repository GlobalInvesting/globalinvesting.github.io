/**
 * calendar-panel.js — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (ForexFactory, G8, medium+high impact)
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

    // Scroll logic — priority order:
    // 1. Today's date separator if today has any events (show today even if all past)
    // 2. First upcoming event within today (scroll to next pending event)
    // 3. First future date separator (next trading day, when no events today)
    // 4. Top (fallback)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      // Priority 1: today's date row — always anchor on today if events exist today
      const todayRow = container.querySelector('[data-today="1"]');
      if (todayRow) {
        todayRow.scrollIntoView({ block: 'start' });
        return;
      }

      // Priority 2: no today section — find first upcoming event date row
      const firstUpcoming = container.querySelector('[data-upcoming="1"]');
      if (firstUpcoming) {
        const prev = firstUpcoming.previousElementSibling;
        const target = (prev && prev.classList.contains('cal-date-row')) ? prev : firstUpcoming;
        target.scrollIntoView({ block: 'start' });
        return;
      }

      // Priority 3: first future date (next trading day after today)
      const allDateRows = container.querySelectorAll('.cal-date-row[data-date]');
      for (const row of allDateRows) {
        if (row.dataset.date > today) {
          row.scrollIntoView({ block: 'start' });
          return;
        }
      }

      // Fallback: top
      container.scrollTop = 0;
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
