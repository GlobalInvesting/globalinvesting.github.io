/**
 * calendar-panel.js — Native economic calendar renderer  v1.2.0
 * Reads calendar-data/ff_calendar.json (ForexFactory / Finnhub, G8, medium+high impact)
 * Renders inline with terminal colors — no third-party iframes.
 *
 * v1.2.0 (2026-05-25) — Market holiday awareness
 * - Hardcoded G8 holiday table (2026). On any holiday date, a Bloomberg-style
 *   banner row appears under the date header: holiday name + affected currency
 *   chips + "Prior close" note. Covers all dates in the visible window.
 * - cal-panel-sub gets a holiday count suffix when today is a holiday.
 * - Exposes window._MARKET_HOLIDAYS for dashboard.js cross-asset tooltip logic.
 */
(function () {
  'use strict';

  // ─── G8 Market Holidays 2026 ─────────────────────────────────────────────
  // Sources: NYSE/CME (USD), Eurex/ECB (EUR), LSE (GBP), TSE (JPY),
  //          ASX (AUD), TSX (CAD), SIX (CHF), NZX (NZD)
  // Full-day closures only. Partial/early-close days are excluded.
  var G8_HOLIDAYS_RAW = [
    // USD
    { date:'2026-01-01', label:"New Year's Day",             currencies:['USD','EUR','GBP','AUD','CAD','CHF','NZD'] },
    { date:'2026-01-19', label:'Martin Luther King Jr. Day', currencies:['USD'] },
    { date:'2026-02-16', label:"Presidents' Day",            currencies:['USD'] },
    { date:'2026-04-03', label:'Good Friday',                currencies:['USD','EUR','GBP','AUD','CAD','CHF','NZD'] },
    { date:'2026-04-06', label:'Easter Monday',              currencies:['EUR','GBP','AUD','CAD','CHF','NZD'] },
    { date:'2026-05-25', label:'Memorial Day',               currencies:['USD'] },
    { date:'2026-07-03', label:'Independence Day (observed)',currencies:['USD'] },
    { date:'2026-09-07', label:'Labor Day',                  currencies:['USD','CAD'] },
    { date:'2026-10-12', label:'Columbus Day',               currencies:['USD'] },
    { date:'2026-11-11', label:'Veterans Day / Remembrance Day', currencies:['USD','CAD'] },
    { date:'2026-11-26', label:'Thanksgiving Day',           currencies:['USD'] },
    { date:'2026-12-25', label:'Christmas Day',              currencies:['USD','EUR','GBP','AUD','CAD','CHF','NZD'] },
    { date:'2026-12-26', label:'Boxing Day',                 currencies:['GBP','AUD','CAD','CHF','NZD'] },
    // GBP
    { date:'2026-05-04', label:'Early May Bank Holiday',     currencies:['GBP'] },
    { date:'2026-05-26', label:'Spring Bank Holiday',        currencies:['GBP'] },
    { date:'2026-08-03', label:'Summer Bank Holiday',        currencies:['GBP'] },
    // EUR / CHF
    { date:'2026-05-19', label:'Whit Monday',                currencies:['CHF'] },
    { date:'2026-05-26', label:'Whit Monday',                currencies:['EUR'] },
    // AUD
    { date:'2026-01-26', label:'Australia Day',              currencies:['AUD'] },
    { date:'2026-04-25', label:'ANZAC Day',                  currencies:['AUD','NZD'] },
    { date:'2026-06-08', label:"King's Birthday (AU)",       currencies:['AUD'] },
    // CAD
    { date:'2026-05-18', label:'Victoria Day',               currencies:['CAD'] },
    { date:'2026-10-12', label:'Thanksgiving Day (CA)',      currencies:['CAD'] },
    // NZD
    { date:'2026-01-02', label:"New Year's Day (observed)",  currencies:['NZD'] },
    { date:'2026-06-01', label:"King's Birthday (NZ)",       currencies:['NZD'] },
    { date:'2026-10-26', label:'Labour Day (NZ)',            currencies:['NZD'] },
    { date:'2026-12-28', label:'Boxing Day (observed)',      currencies:['NZD'] },
    // JPY — TSE
    { date:'2026-01-01', label:"New Year's Day",             currencies:['JPY'] },
    { date:'2026-01-12', label:'Coming of Age Day',          currencies:['JPY'] },
    { date:'2026-02-11', label:'National Foundation Day',    currencies:['JPY'] },
    { date:'2026-02-23', label:"Emperor's Birthday",         currencies:['JPY'] },
    { date:'2026-03-20', label:'Vernal Equinox Day',         currencies:['JPY'] },
    { date:'2026-04-29', label:'Showa Day',                  currencies:['JPY'] },
    { date:'2026-05-03', label:'Constitution Memorial Day',  currencies:['JPY'] },
    { date:'2026-05-04', label:'Greenery Day',               currencies:['JPY'] },
    { date:'2026-05-05', label:"Children's Day",             currencies:['JPY'] },
    { date:'2026-07-20', label:'Marine Day',                 currencies:['JPY'] },
    { date:'2026-08-11', label:'Mountain Day',               currencies:['JPY'] },
    { date:'2026-09-21', label:'Respect for the Aged Day',   currencies:['JPY'] },
    { date:'2026-09-23', label:'Autumnal Equinox',           currencies:['JPY'] },
    { date:'2026-10-12', label:'Sports Day',                 currencies:['JPY'] },
    { date:'2026-11-03', label:'Culture Day',                currencies:['JPY'] },
    { date:'2026-11-23', label:'Labour Thanksgiving Day',    currencies:['JPY'] },
  ];

  // Merge into one entry per date
  var HOLIDAYS_BY_DATE = (function () {
    var map = {};
    G8_HOLIDAYS_RAW.forEach(function (h) {
      if (!map[h.date]) map[h.date] = { labels: [], currencies: [] };
      if (map[h.date].labels.indexOf(h.label) === -1) map[h.date].labels.push(h.label);
      h.currencies.forEach(function (c) {
        if (map[h.date].currencies.indexOf(c) === -1) map[h.date].currencies.push(c);
      });
    });
    return map;
  })();

  // Expose for dashboard.js cross-asset holiday tooltip logic
  window._MARKET_HOLIDAYS = HOLIDAYS_BY_DATE;

  // ─── Constants ────────────────────────────────────────────────────────────
  var G8_SET    = { USD:1, EUR:1, GBP:1, JPY:1, AUD:1, CAD:1, CHF:1, NZD:1 };
  var IMPACTS   = { medium:1, high:1 };

  var IMPACT_DOT = {
    high:   { color: 'var(--down)',   label: 'High'   },
    medium: { color: 'var(--orange)', label: 'Medium' },
  };

  var FLAG = { USD:'us', EUR:'eu', GBP:'gb', JPY:'jp', AUD:'au', CAD:'ca', CHF:'ch', NZD:'nz' };

  var EXCHANGE = {
    USD:'NYSE/CME', EUR:'Eurex', GBP:'LSE', JPY:'TSE',
    AUD:'ASX', CAD:'TSX', CHF:'SIX', NZD:'NZX',
  };

  // ─── Helpers ──────────────────────────────────────────────────────────────
  function tzLabel() {
    var off = -new Date().getTimezoneOffset();
    var sign = off >= 0 ? '+' : '-';
    var h = Math.floor(Math.abs(off) / 60);
    var m = Math.abs(off) % 60;
    return 'GMT' + sign + h + (m ? ':' + String(m).padStart(2,'0') : '');
  }

  function toLocalTime(dateISO, timeUTC) {
    if (!timeUTC) return 'All Day';
    var parts = timeUTC.split(':').map(Number);
    var d = new Date(Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), parts[0], parts[1]
    ));
    return d.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit', hour12:false });
  }

  function formatDate(dateISO) {
    var d = new Date(dateISO + 'T12:00:00Z');
    return d.toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric', timeZone:'UTC' });
  }

  function isPastEvent(dateISO, timeUTC) {
    var t = (timeUTC || '23:59').split(':').map(Number);
    var evMs = Date.UTC(
      +dateISO.slice(0,4), +dateISO.slice(5,7)-1, +dateISO.slice(8,10), t[0], t[1]
    );
    return evMs < Date.now();
  }

  function todayISO() {
    return new Date().toISOString().slice(0, 10);
  }

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ─── Holiday row ─────────────────────────────────────────────────────────
  // Sits in the same 7-col grid as .cal-event-row — perfectly aligned.
  // Cols: [time=empty] [ccy=flag icons] [dot=⊘] [title=holiday name italic]
  //       [cols 5-7 span = "Prior close" right-aligned]
  function buildHolidayRowHtml(dateISO, holiday) {
    var ccys = holiday.currencies.filter(function (c) { return G8_SET[c]; });
    if (!ccys.length) return '';

    // Flag icons only — no text labels, no chips. Fits in the 52px ccy column.
    var flags = ccys.slice(0, 4).map(function (c) {
      var f  = FLAG[c] || '';
      var ex = EXCHANGE[c] || c;
      return f ? '<span class="fi fi-' + f + '" title="' + esc(ex) + ' closed" style="font-size:10px;line-height:1;"></span>' : '';
    }).filter(Boolean).join('');

    var label   = holiday.labels.join(' \u00b7 ');
    var exList  = ccys.map(function (c) { return c + ' (' + (EXCHANGE[c] || c) + ')'; }).join(', ');
    var tooltip = esc(label) + ' \u2014 ' + esc(exList) + ' closed \u2014 prices reflect prior close';

    return '<div class="cal-holiday-row" data-date="' + dateISO + '" title="' + tooltip + '">' +
      '<span></span>' +
      '<span class="cal-holiday-flags">' + flags + '</span>' +
      '<span class="cal-holiday-icon" aria-hidden="true">\u2298</span>' +
      '<span class="cal-holiday-name">' + esc(label) + '</span>' +
      '<span class="cal-holiday-note">Prior close</span>' +
      '</div>';
  }

  // ─── Scroll helpers ────────────────────────────────────────────────────────
  function scrollCalTo(container, target) {
    if (!target) { container.scrollTop = 0; return; }
    container.scrollTop = Math.max(0, target.offsetTop - container.offsetTop - 2);
  }

  // ─── Next-event jump button ───────────────────────────────────────────────
  function setupNextEventButton(container, firstUpcomingEl) {
    var prev = document.getElementById('cal-next-btn');
    if (prev) prev.remove();
    if (!firstUpcomingEl) return;

    var timeEl  = firstUpcomingEl.querySelector('.cal-time');
    var ccyEl   = firstUpcomingEl.querySelector('.cal-ccy');
    var titleEl = firstUpcomingEl.querySelector('.cal-title');
    var dotEl   = firstUpcomingEl.querySelector('.cal-dot');

    var timeStr    = timeEl  ? timeEl.textContent.trim()  : '';
    var ccyStr     = ccyEl   ? ccyEl.textContent.trim()   : '';
    var titleStr   = titleEl ? titleEl.textContent.trim() : 'Next event';
    var shortTitle = titleStr.length > 28 ? titleStr.slice(0, 26) + '\u2026' : titleStr;
    var dotColor   = dotEl ? dotEl.style.background : 'var(--text3)';

    var btn = document.createElement('button');
    btn.id = 'cal-next-btn';
    btn.title = 'Jump to next event: ' + titleStr;
    btn.setAttribute('aria-label', 'Jump to next event: ' + titleStr);
    btn.innerHTML =
      '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:' + dotColor + ';margin-right:5px;flex-shrink:0;"></span>' +
      '<span style="color:var(--text2);margin-right:4px;font-family:var(--font-mono);font-size:10px;">' + timeStr + '</span>' +
      '<span style="color:var(--text2);margin-right:4px;font-size:9px;">' + ccyStr + '</span>' +
      '<span style="color:var(--text2);font-size:10px;">' + shortTitle + '</span>' +
      '<span id="cal-next-btn-arrow" style="color:var(--text2);margin-left:5px;font-size:10px;">\u2193</span>';
    btn.style.cssText = [
      'position:absolute','bottom:6px','left:50%','transform:translateX(-50%)',
      'display:flex','align-items:center','padding:4px 10px',
      'background:var(--bg3)','border:1px solid var(--border2)','border-radius:12px',
      'cursor:pointer','white-space:nowrap','z-index:10',
      'transition:opacity .15s','opacity:0','pointer-events:none',
    ].join(';');

    var wrapper = container.parentElement;
    if (!wrapper) return;
    wrapper.style.position = 'relative';
    wrapper.appendChild(btn);

    btn.addEventListener('click', function () {
      var prevEl = firstUpcomingEl.previousElementSibling;
      var target = (prevEl && prevEl.classList.contains('cal-date-row')) ? prevEl : firstUpcomingEl;
      scrollCalTo(container, target);
    });

    function updateVisibility() {
      var cTop    = container.scrollTop;
      var cBottom = cTop + container.clientHeight;
      var eTop    = firstUpcomingEl.offsetTop - container.offsetTop;
      var eBottom = eTop + firstUpcomingEl.offsetHeight;
      var visible = eTop >= cTop && eBottom <= cBottom + 4;
      btn.style.opacity       = visible ? '0' : '0.92';
      btn.style.pointerEvents = visible ? 'none' : 'auto';
      var arrowEl = document.getElementById('cal-next-btn-arrow');
      if (arrowEl) arrowEl.textContent = eTop < cTop ? '\u2191' : '\u2193';
    }
    container.addEventListener('scroll', updateVisibility, { passive: true });
    requestAnimationFrame(function () { requestAnimationFrame(updateVisibility); });
  }

  // ─── Build panel ──────────────────────────────────────────────────────────
  function buildPanel(events, source) {
    source = source || 'Finnhub';
    var container = document.getElementById('cal-events-body');
    var sourceEl  = document.getElementById('cal-panel-sub');
    if (!container) return;

    var today = todayISO();

    // Filter to G8 medium+high impact
    var filtered = events.filter(function (ev) {
      return G8_SET[ev.currency] && IMPACTS[ev.impact];
    });

    // Collect all dates: event dates + holiday dates in the visible window
    // Window: 5 days back to 30 days ahead (covers this week's releases + upcoming)
    var windowStartMs = new Date(today).getTime() - 5 * 86400000;
    var windowEndMs   = new Date(today).getTime() + 30 * 86400000;

    var allDates = {};
    filtered.forEach(function (ev) { allDates[ev.dateISO] = 1; });
    Object.keys(HOLIDAYS_BY_DATE).forEach(function (d) {
      var dMs = new Date(d).getTime();
      if (dMs >= windowStartMs && dMs <= windowEndMs) allDates[d] = 1;
    });

    var sortedDates = Object.keys(allDates).sort();

    if (!sortedDates.length) {
      container.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">No events available.</div>';
      return;
    }

    // Group events by date
    var byDate = {};
    filtered.forEach(function (ev) {
      if (!byDate[ev.dateISO]) byDate[ev.dateISO] = [];
      byDate[ev.dateISO].push(ev);
    });

    var html = '';

    sortedDates.forEach(function (dateISO) {
      var dayEvs  = byDate[dateISO] || [];
      var holiday = HOLIDAYS_BY_DATE[dateISO];
      var isToday = dateISO === today;

      // Skip dates with no events AND no holiday
      if (!dayEvs.length && !holiday) return;
      // Skip past dates >5d ago that have no holiday marker
      var dMs = new Date(dateISO).getTime();
      if (!holiday && dMs < windowStartMs) return;

      html += '<div class="cal-date-row" data-date="' + dateISO + '"' +
        (isToday ? ' data-today="1"' : '') + '>' + formatDate(dateISO) + '</div>';

      // Bloomberg convention: holiday banner BEFORE event rows
      if (holiday) html += buildHolidayRowHtml(dateISO, holiday);

      dayEvs.forEach(function (ev) {
        var dot        = IMPACT_DOT[ev.impact];
        var flag       = FLAG[ev.currency] || '';
        var flagHtml   = flag ? '<span class="fi fi-' + flag + '" style="margin-right:4px;font-size:10px;flex-shrink:0;"></span>' : '';
        var isReleased = !!(ev.actual && ev.actual !== '' && ev.actual !== '-');
        var isPast     = isPastEvent(ev.dateISO, ev.timeUTC);
        var dimmed     = isPast && isReleased;

        var actualHtml = '<span style="color:var(--text3)">—</span>';
        if (isReleased && ev.actual != null) {
          var actualN   = parseFloat(String(ev.actual).replace(/[%,KMB\s]/gi,''));
          var forecastN = parseFloat(String(ev.forecast || ev.previous || '').replace(/[%,KMB\s]/gi,''));
          var cls = '';
          if (!isNaN(actualN) && !isNaN(forecastN) && actualN !== forecastN) {
            cls = actualN > forecastN ? ' class="up"' : ' class="down"';
          }
          actualHtml = '<span' + cls + '>' + esc(String(ev.actual)) + '</span>';
        }

        var forecastHtml = ev.forecast
          ? '<span style="color:var(--text2)">' + esc(String(ev.forecast)) + '</span>'
          : '<span style="color:var(--text3)">—</span>';
        var previousHtml = ev.previous
          ? '<span style="color:var(--text3)">' + esc(String(ev.previous)) + '</span>'
          : '<span style="color:var(--text3)">—</span>';

        var localTime    = toLocalTime(ev.dateISO, ev.timeUTC);
        var upcomingAttr = (!isPast) ? ' data-upcoming="1"' : '';

        html += '<div class="cal-event-row' + (dimmed ? ' cal-released' : '') + '"' + upcomingAttr + '>' +
          '<div class="cal-col cal-time">' + localTime + '</div>' +
          '<div class="cal-col cal-ccy">' + flagHtml + esc(ev.currency) + '</div>' +
          '<div class="cal-col cal-impact"><span class="cal-dot" style="background:' + dot.color + '" title="' + dot.label + ' impact"></span></div>' +
          '<div class="cal-col cal-title" title="' + esc(ev.title) + '">' + esc(ev.title) + '</div>' +
          '<div class="cal-col cal-num">' + actualHtml + '</div>' +
          '<div class="cal-col cal-num">' + forecastHtml + '</div>' +
          '<div class="cal-col cal-num">' + previousHtml + '</div>' +
          '</div>';
      });
    });

    container.innerHTML = html || '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">No events available.</div>';

    // ── Scroll: today > first upcoming > next future date > top ──────────────
    requestAnimationFrame(function () { requestAnimationFrame(function () {
      var todayRow      = container.querySelector('[data-today="1"]');
      var firstUpcoming = container.querySelector('[data-upcoming="1"]');

      if (todayRow) {
        scrollCalTo(container, todayRow);
      } else if (firstUpcoming) {
        var prevEl = firstUpcoming.previousElementSibling;
        var target = (prevEl && prevEl.classList.contains('cal-date-row')) ? prevEl : firstUpcoming;
        scrollCalTo(container, target);
      } else {
        var dateRows = container.querySelectorAll('.cal-date-row[data-date]');
        var scrolled = false;
        for (var i = 0; i < dateRows.length; i++) {
          if (dateRows[i].dataset.date > today) {
            scrollCalTo(container, dateRows[i]);
            scrolled = true;
            break;
          }
        }
        if (!scrolled) container.scrollTop = 0;
      }

      setupNextEventButton(container, firstUpcoming);
    }); });

    // ── Panel subtitle: source + timezone + holiday notice ────────────────────
    if (sourceEl) {
      var todayH = HOLIDAYS_BY_DATE[today];
      var sub = source + ' \u00b7 G8 \u00b7 medium & high impact \u00b7 ' + tzLabel();
      if (todayH) {
        var n = todayH.currencies.filter(function (c) { return G8_SET[c]; }).length;
        sub += ' \u00b7 ' + n + ' market holiday' + (n > 1 ? 's' : '') + ' today \u00b7 prices reflect prior close';
      }
      sourceEl.textContent = sub;
    }

    var thTime = document.getElementById('cal-th-time');
    if (thTime) thTime.textContent = tzLabel();
  }

  // ─── Fetch ────────────────────────────────────────────────────────────────
  async function fetchEconomicCalendar() {
    try {
      var events = [], source = 'Finnhub';
      var paths = ['./calendar-data/ff_calendar.json', './calendar-data/calendar.json'];
      for (var i = 0; i < paths.length; i++) {
        var res = await fetch(paths[i]).catch(function () { return null; });
        if (!res || !res.ok) continue;
        var j = await res.json();
        if (j && j.events && j.events.length) {
          events = j.events;
          source = j.source || source;
          // Merge live holidays from JSON into the lookup table (overrides hardcoded table
          // for dates in the JSON window; hardcoded table still covers dates outside it)
          if (j.holidays && Array.isArray(j.holidays)) {
            j.holidays.forEach(function (h) {
              if (!h.dateISO || !h.currencies) return;
              HOLIDAYS_BY_DATE[h.dateISO] = {
                labels: [h.label || 'Market Holiday'],
                currencies: h.currencies,
              };
            });
          }
          break;
        }
      }
      // Re-expose updated table for dashboard.js cross-asset annotation
      window._MARKET_HOLIDAYS = HOLIDAYS_BY_DATE;
      buildPanel(events, source);
    } catch (e) {
      var c = document.getElementById('cal-events-body');
      if (c) c.innerHTML = '<div style="padding:12px 10px;color:var(--text3);font-size:11px;">Calendar unavailable.</div>';
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fetchEconomicCalendar);
  } else {
    fetchEconomicCalendar();
  }
})();
