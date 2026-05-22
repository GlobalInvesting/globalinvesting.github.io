/**
 * calendar-panel.js — Native economic calendar renderer
 * Reads calendar-data/ff_calendar.json (ForexFactory, G8, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 *
 * TEST FILE — not part of production dashboard.js yet.
 */

(function () {
  'use strict';

  const G8 = new Set(['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD']);
  const IMPACTS = new Set(['medium','high']);

  // Impact dot colors — map to terminal palette
  const IMPACT_DOT = {
    high:   { color: 'var(--down)',   label: 'High'   },
    medium: { color: 'var(--orange)', label: 'Medium' },
    low:    { color: 'var(--text3)',  label: 'Low'    },
  };

  // Currency flag — use existing fi flag-icons already loaded on page
  const FLAG = {
    USD: 'us', EUR: 'eu', GBP: 'gb', JPY: 'jp',
    AUD: 'au', CAD: 'ca', CHF: 'ch', NZD: 'nz',
  };

  function formatDate(dateISO) {
    // dateISO: "2026-05-22" → "Friday, May 22"
    const d = new Date(dateISO + 'T12:00:00Z');
    return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', timeZone: 'UTC' });
  }

  function formatTime(timeUTC) {
    // timeUTC: "13:30" — return as-is (UTC label added in header)
    return timeUTC || 'All Day';
  }

  function buildPanel(events) {
    const container = document.getElementById('cal-events-body');
    const sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;

    // Filter: G8, medium+high only, upcoming (not released) + today already released
    const nowMs  = Date.now();
    const todayISO = new Date().toISOString().slice(0, 10);

    // Show: all unreleased events + released events from today
    const filtered = events.filter(ev => {
      if (!G8.has(ev.currency)) return false;
      if (!IMPACTS.has(ev.impact)) return false;
      const isPast = ev.dateISO < todayISO;
      if (isPast) return false; // skip past days entirely
      return true;
    });

    if (!filtered.length) {
      container.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">No medium or high impact events found for this week.</div>';
      return;
    }

    // Group by date
    const byDate = {};
    filtered.forEach(ev => {
      if (!byDate[ev.dateISO]) byDate[ev.dateISO] = [];
      byDate[ev.dateISO].push(ev);
    });

    let html = '';
    Object.keys(byDate).sort().forEach(dateISO => {
      const dayEvs = byDate[dateISO];
      // Date separator row
      html += `<div class="cal-date-row">${formatDate(dateISO)}</div>`;

      dayEvs.forEach(ev => {
        const dot = IMPACT_DOT[ev.impact] || IMPACT_DOT.low;
        const flag = FLAG[ev.currency] || '';
        const flagHtml = flag ? `<span class="fi fi-${flag}" style="margin-right:4px;font-size:11px;"></span>` : '';
        const isReleased = ev.released || (ev.actual && ev.actual !== '' && ev.actual !== '-');

        // Actual vs forecast coloring
        let actualHtml = '<span style="color:var(--text3)">—</span>';
        if (isReleased && ev.actual != null) {
          // Simple beat/miss — if actual > forecast typically good (except inverse indicators)
          const actualN   = parseFloat(String(ev.actual).replace(/[%,K M B]/gi,''));
          const forecastN = parseFloat(String(ev.forecast || ev.previous || '').replace(/[%,K M B]/gi,''));
          let cls = '';
          if (!isNaN(actualN) && !isNaN(forecastN)) {
            cls = actualN > forecastN ? 'class="up"' : actualN < forecastN ? 'class="down"' : '';
          }
          actualHtml = `<span ${cls}>${ev.actual}</span>`;
        }

        const forecastHtml = ev.forecast ? `<span style="color:var(--text2)">${ev.forecast}</span>` : '<span style="color:var(--text3)">—</span>';
        const previousHtml = ev.previous ? `<span style="color:var(--text3)">${ev.previous}</span>` : '<span style="color:var(--text3)">—</span>';

        html += `
          <div class="cal-event-row${isReleased ? ' cal-released' : ''}">
            <div class="cal-col cal-time">${formatTime(ev.timeUTC)}</div>
            <div class="cal-col cal-ccy">${flagHtml}${ev.currency}</div>
            <div class="cal-col cal-impact"><span class="cal-dot" style="background:${dot.color};" title="${dot.label} impact"></span></div>
            <div class="cal-col cal-title">${ev.title}</div>
            <div class="cal-col cal-num">${actualHtml}</div>
            <div class="cal-col cal-num">${forecastHtml}</div>
            <div class="cal-col cal-num">${previousHtml}</div>
          </div>`;
      });
    });

    container.innerHTML = html;
    if (sourceEl) sourceEl.textContent = 'ForexFactory · G8 · medium & high impact · UTC';
  }

  async function fetchEconomicCalendar() {
    try {
      // Try enriched calendar.json first, then ff_calendar.json
      let events = [];
      for (const path of ['./calendar-data/ff_calendar.json', './calendar-data/calendar.json']) {
        const res = await fetch(path).catch(() => null);
        if (!res?.ok) continue;
        const j = await res.json();
        const evts = j?.events || [];
        if (evts.length) { events = evts; break; }
      }
      buildPanel(events);
    } catch (e) {
      const container = document.getElementById('cal-events-body');
      if (container) container.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">Calendar unavailable.</div>';
    }
  }

  // Run after DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fetchEconomicCalendar);
  } else {
    fetchEconomicCalendar();
  }
})();
