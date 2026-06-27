/**
 * econ-matrix.js v1.0.0 — Native Economic Matrix panel
 *
 * Replaces the third-party TradingView Economic Map widget (tv-economic-map.js)
 * with a native table in the style of an institutional regional economic matrix
 * (e.g. Bloomberg ECMX), built entirely from data the terminal already fetches
 * elsewhere — no new backend script or workflow required.
 *
 * Column sourcing:
 *   GDP, CPI, Unemp, Ind Prod, Bus Cond, Rtl Sales, Cur Acct, Trade Bal
 *     → calendar-data/calendar.json — latest "actual" print per category per
 *       currency. This file carries ~1yr of history with real released actuals
 *       (unlike economic-data/{CCY}.json, disabled in v7.24.1 for staleness —
 *       see GUIDELINES.md "Data directories").
 *   10Y Yld
 *     → extended-data/{CCY}.json `bond10y` — same field already used by
 *       yc-modal.js for the Yield Curve detail modal. No color/trend shown,
 *       matching the established precedent there (extended-data carries no
 *       intraday delta for this field).
 *   CB Rate
 *     → window._STATE_cbRates (populated by fetchCBRates() in dashboard.js) +
 *       computeCBTrend() for the trend arrow color — reused as-is so this
 *       panel never disagrees with the CB Rates table elsewhere on the page.
 *
 * Documented deviations from the Bloomberg ECMX column set (see CHANGELOG
 * v8.23.0 for full rationale):
 *   - "Bud %GDP" (fiscal budget balance) has no recurring calendar release
 *     outside the US in the current source, so the column is replaced with
 *     Trade Balance, which the calendar carries for all 10 currencies and is
 *     directly FX-relevant.
 *   - "CA %GDP" is shown as "Cur Acct" — the calendar's raw latest actual, in
 *     each currency's own native reporting units, rather than a %GDP ratio.
 *     The values are not uniformly unit-tagged at the source (some carry a
 *     currency prefix, some don't), so dividing by a GDP denominator would
 *     manufacture false precision. Trend coloring still works (see below).
 *
 * Two cells are intentionally left blank ("—") because the underlying release
 * does not exist in the current source for that currency:
 *   - Ind Prod: AUD, NZD — neither economy publishes a standalone industrial
 *     production series in this feed.
 *   - CPI: NZD — not currently tracked in the source feed for NZD (a feed
 *     gap, not a "doesn't exist" fact — NZ does publish quarterly CPI).
 *
 * Color convention: every calendar-derived cell is colored by the direction
 * of change vs. the previous reading (delta = actual − previous), not by the
 * raw sign of the level. This is purely descriptive (mirrors how price/D%/W%
 * deltas are colored elsewhere in the terminal) and avoids any "good/bad"
 * value judgement on a given print, consistent with GUIDELINES' ban on
 * investment-advice-flavored signal language.
 */
(function () {
  'use strict';

  const CCY_ORDER = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CHF', 'CAD', 'NZD', 'NOK', 'SEK'];
  const FLAG = { USD: 'us', EUR: 'eu', GBP: 'gb', JPY: 'jp', AUD: 'au', CHF: 'ch', CAD: 'ca', NZD: 'nz', NOK: 'no', SEK: 'se' };

  const COLUMNS = [
    { key: 'gdp',   label: 'GDP',       title: 'Latest GDP growth rate — native release cadence per economy' },
    { key: 'cpi',   label: 'CPI',       title: 'Latest CPI / inflation rate, year-on-year' },
    { key: 'unemp', label: 'Unemp',     title: 'Latest unemployment rate' },
    { key: 'prod',  label: 'Ind Prod',  title: 'Latest industrial / manufacturing production change' },
    { key: 'conf',  label: 'Bus Cond',  title: 'Latest manufacturing PMI, or the economy\u2019s standard business/industrial confidence survey where no PMI is published' },
    { key: 'rtl',   label: 'Rtl Sales', title: 'Latest retail sales change' },
    { key: 'ca',    label: 'Cur Acct',  title: 'Latest current account, native reporting units (see GUIDELINES — not normalized to %GDP)' },
    { key: 'trade', label: 'Trade Bal', title: 'Latest trade balance, native reporting units' },
  ];

  // Union of accepted ForexFactory/Myfxbook event-name prefixes per category,
  // per currency. Prefix match (startsWith) tolerates month-suffix variants
  // such as "(Mar)" or "(Q1)" appended by the upstream feed. Empty array =
  // confirmed gap for that currency (see header comment) — renders "—".
  const CATS = {
    USD: {
      gdp:   ['GDP (QoQ)', 'GDP Growth Rate QoQ', 'United States GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate YoY', 'CPI (YoY)', 'CPI y/y'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production MoM', 'United States Industrial Production MoM', 'Industrial Production (MoM)'],
      conf:  ['ISM Manufacturing PMI'],
      rtl:   ['Retail Sales MoM', 'United States Retail Sales MoM', 'Retail Sales (MoM)'],
      ca:    ['Current Account', 'United States Current Account'],
      trade: ['Trade Balance', 'Balance of Trade', 'Goods Trade Balance', 'United States Goods Trade Balance'],
    },
    EUR: {
      gdp:   ['Gross Domestic Product s.a (QoQ)', 'GDP (QoQ)'],
      cpi:   ['Inflation Rate YoY Flash', 'CPI (YoY)', 'Consumer Price Index (YoY)', 'Euro Area CPI'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production MoM', 'Euro Area Industrial Production MoM', 'Industrial Production (MoM)'],
      conf:  ['HCOB Eurozone Manufacturing PMI', 'HCOB Manufacturing PMI Flash'],
      rtl:   ['Retail Sales (MoM)', 'Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade', 'Trade Balance EU', 'Euro Area Balance of Trade'],
    },
    GBP: {
      gdp:   ['GDP MoM', 'United Kingdom GDP MoM'],
      cpi:   ['Inflation Rate YoY', 'CPI (YoY)', 'United Kingdom Inflation Rate YoY'],
      unemp: ['Unemployment Rate', 'United Kingdom Unemployment Rate'],
      prod:  ['Industrial Production MoM', 'United Kingdom Industrial Production MoM', 'Industrial Production (MoM)'],
      conf:  ['S&P Global Manufacturing PMI'],
      rtl:   ['Retail Sales MoM', 'United Kingdom Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Trade Balance', 'Goods Trade Balance', 'United Kingdom Goods Trade Balance'],
    },
    JPY: {
      gdp:   ['GDP Growth Rate QoQ Final', 'GDP Growth Rate QoQ Prel', 'GDP (QoQ)'],
      cpi:   ['National Core CPI (YoY)', 'Inflation Rate YoY', 'Core Inflation Rate YoY', 'Japan Inflation Rate YoY'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production (MoM)', 'Industrial Production MoM Prel'],
      conf:  ['Jibun Bank Manufacturing PMI'],
      rtl:   ['Retail Sales YoY', 'Retail Sales (QoQ)'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade', 'Trade Balance', 'Japan Balance of Trade'],
    },
    AUD: {
      gdp:   ['GDP Growth Rate QoQ', 'GDP Growth Rate YoY'],
      cpi:   ['Australia CPI', 'Inflation Rate YoY', 'Australia Inflation Rate YoY'],
      unemp: ['Australia Unemployment Rate', 'Unemployment Rate'],
      prod:  [], // not published as a standalone release in the current source
      conf:  ['NAB Business Confidence'],
      rtl:   ['Retail Sales (QoQ)'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade', 'Trade Balance'],
    },
    CAD: {
      gdp:   ['GDP Growth Rate Annualized', 'GDP MoM'],
      cpi:   ['Canada Inflation Rate YoY', 'Inflation Rate YoY'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production (YoY)'],
      conf:  ['Ivey PMI s.a', 'S&P Global Manufacturing PMI'],
      rtl:   ['Retail Sales MoM', 'Canada Retail Sales MoM', 'Retail Sales MoM Final'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade', 'Trade Balance'],
    },
    CHF: {
      gdp:   ['GDP Growth Rate YoY', 'GDP Growth Rate QoQ Flash'],
      cpi:   ['Inflation Rate YoY'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production YoY'],
      conf:  ['procure.ch Manufacturing PMI'],
      rtl:   ['Retail Sales YoY'],
      ca:    ['Switzerland Current Account', 'Current Account'],
      trade: ['Balance of Trade', 'Switzerland Balance of Trade', 'Trade Balance'],
    },
    NZD: {
      gdp:   ['New Zealand GDP Growth Rate QoQ', 'Gross Domestic Product (QoQ)', 'New Zealand GDP Growth Rate YoY'],
      cpi:   [], // not currently tracked in the source feed for NZD
      unemp: ['Unemployment Rate'],
      prod:  [], // not published as a standalone release in the current source
      conf:  ['New Zealand Business NZ PMI', 'Business NZ PMI'],
      rtl:   ['Retail Sales (QoQ)'],
      ca:    ['New Zealand Current Account', 'Current Account'],
      trade: ['Balance of Trade', 'New Zealand Balance of Trade', 'Trade Balance NZD (MoM)'],
    },
    SEK: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['CPIF YoY'],
      unemp: ['Unemployment Rate'],
      prod:  ['Industrial Production YoY'],
      conf:  ['Swedbank Manufacturing PMI'],
      rtl:   ['Retail Sales YoY'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
    },
    NOK: {
      gdp:   ['GDP Growth Rate QoQ'],
      cpi:   ['Inflation Rate YoY'],
      unemp: ['Unemployment Rate'],
      prod:  ['Manufacturing Production MoM'],
      conf:  ['Industrial Confidence'],
      rtl:   ['Retail Sales MoM'],
      ca:    ['Current Account'],
      trade: ['Balance of Trade'],
    },
  };

  const GAP_TITLE = {
    prod: 'Not published as a standalone release in the current source for this currency',
    cpi:  'Not currently tracked in the source feed for this currency',
  };

  // ── Value parsing — sign-aware, unit-agnostic ──────────────────────────────
  // Calendar "actual"/"previous" strings carry inconsistent prefixes (none,
  // "$", "CHF", "NZ$", "-SEK", ...) and suffixes ("%", "B"). We only need a
  // signed numeric value to compute trend direction for coloring — the cell
  // itself always displays the original string verbatim, so no precision is
  // invented and no unit conversion is attempted.
  function parseNum(s) {
    if (s == null) return null;
    const str = String(s).trim().replace(/,/g, '');
    const digitIdx = str.search(/\d/);
    if (digitIdx === -1) return null;
    const prefix = str.slice(0, digitIdx);
    const neg = prefix.indexOf('-') !== -1 || prefix.indexOf('(') !== -1;
    const m = str.slice(digitIdx).match(/\d+\.?\d*/);
    if (!m) return null;
    const v = parseFloat(m[0]);
    return neg ? -v : v;
  }

  function trendClass(actual, previous) {
    const a = parseNum(actual), p = parseNum(previous);
    if (a == null || p == null) return '';
    if (a > p) return 'up';
    if (a < p) return 'down';
    return 'flat';
  }

  // ── Build latest-actual index from calendar-data/calendar.json ────────────
  function findLatest(eventsByCcy, ccy, prefixes) {
    if (!prefixes || !prefixes.length) return null;
    const list = eventsByCcy[ccy];
    if (!list) return null;
    for (let i = list.length - 1; i >= 0; i--) {
      const ev = list[i];
      for (let j = 0; j < prefixes.length; j++) {
        if (ev.event.indexOf(prefixes[j]) === 0) return ev;
      }
    }
    return null;
  }

  async function loadCalendarData() {
    const res = await fetch('./calendar-data/calendar.json', { cache: 'no-store' }).catch(() => null);
    if (!res || !res.ok) return null;
    const data = await res.json().catch(() => null);
    if (!data || !Array.isArray(data.events)) return null;

    const byCcy = {};
    data.events.forEach(ev => {
      if (!ev || !ev.currency || !ev.dateISO || !ev.event) return;
      if (!byCcy[ev.currency]) byCcy[ev.currency] = [];
      byCcy[ev.currency].push(ev);
    });
    Object.keys(byCcy).forEach(c => byCcy[c].sort((a, b) => a.dateISO < b.dateISO ? -1 : 1));

    const out = {};
    CCY_ORDER.forEach(ccy => {
      out[ccy] = {};
      const cats = CATS[ccy] || {};
      COLUMNS.forEach(col => {
        if (col.key === 'ca' || col.key === 'trade') return; // handled below too, same path
      });
      Object.keys(cats).forEach(catKey => {
        out[ccy][catKey] = findLatest(byCcy, ccy, cats[catKey]);
      });
    });
    return { byCategory: out, lastUpdate: data.lastUpdate || null };
  }

  // ── 10Y yield — extended-data/{CCY}.json, same field as yc-modal.js ────────
  async function load10y(ccy) {
    const ext = await fetch('./extended-data/' + ccy + '.json').then(r => r.ok ? r.json() : null).catch(() => null);
    const v = ext && ext.data && ext.data.bond10y;
    if (v == null || isNaN(v)) return null;
    const date = (ext.dates && ext.dates.bond10y) || '';
    return { value: v, date };
  }

  // ── CB policy rate — reuse window._STATE_cbRates + computeCBTrend ─────────
  function waitForCBRates(timeoutMs) {
    return new Promise(resolve => {
      const start = Date.now();
      (function poll() {
        if (window._STATE_cbRates && Object.keys(window._STATE_cbRates).length) {
          resolve(window._STATE_cbRates);
        } else if (Date.now() - start > timeoutMs) {
          resolve(window._STATE_cbRates || null);
        } else {
          setTimeout(poll, 200);
        }
      }());
    });
  }

  function simpleTrend(obs) {
    if (!obs || obs.length < 2) return 'flat';
    const a = parseFloat(obs[0].value), b = parseFloat(obs[1].value);
    if (isNaN(a) || isNaN(b)) return 'flat';
    if (a > b) return 'up';
    if (a < b) return 'down';
    return 'flat';
  }

  async function getCBRate(ccy) {
    const store = await waitForCBRates(3000);
    const rec = store && store[ccy.toLowerCase()];
    if (rec && rec.rate != null) {
      const trend = (typeof window.computeCBTrend === 'function') ? window.computeCBTrend(rec.obs) : simpleTrend(rec.obs);
      return { rate: rec.rate, date: rec.date, trend };
    }
    // Fallback: independent fetch if STATE never populated (e.g. CB Rates
    // panel failed to load before this one came into view).
    const data = await fetch('./rates/' + ccy + '.json').then(r => r.ok ? r.json() : null).catch(() => null);
    const obs = data && data.observations;
    if (!obs || !obs.length) return null;
    const rate = parseFloat(obs[0].value);
    if (isNaN(rate)) return null;
    return { rate, date: obs[0].date, trend: simpleTrend(obs) };
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  function cellHTML(ev, gapKey) {
    if (!ev) {
      const title = (gapKey && GAP_TITLE[gapKey]) || 'No data available';
      return '<td class="flat" title="' + title + '">\u2014</td>';
    }
    const cls = trendClass(ev.actual, ev.previous);
    const title = ev.event + ' \u00b7 ' + ev.dateISO + (ev.previous != null ? ' \u00b7 prev ' + ev.previous : '');
    return '<td' + (cls ? ' class="' + cls + '"' : '') + ' title="' + title.replace(/"/g, '&quot;') + '">' + (ev.actual != null ? ev.actual : '\u2014') + '</td>';
  }

  function rowHTML(ccy, calRow, y10, cb) {
    const flag = FLAG[ccy] ? '<span class="fi fi-' + FLAG[ccy] + '" style="margin-right:5px;border-radius:2px;"></span>' : '';
    let html = '<tr><td style="white-space:nowrap;">' + flag + '<span style="font-size:10px;">' + ccy + '</span></td>';
    COLUMNS.forEach(col => {
      if (col.key === 'ca' || col.key === 'trade' || col.key === 'gdp' || col.key === 'cpi' ||
          col.key === 'unemp' || col.key === 'prod' || col.key === 'conf' || col.key === 'rtl') {
        html += cellHTML(calRow[col.key], col.key);
      }
    });
    if (y10) {
      html += '<td class="flat" title="10Y \u00b7 as of ' + (y10.date || '\u2014') + '">' + y10.value.toFixed(2) + '%</td>';
    } else {
      html += '<td class="flat" title="No data available">\u2014</td>';
    }
    if (cb) {
      const cls = cb.trend === 'up' ? 'up' : cb.trend === 'down' ? 'down' : 'flat';
      html += '<td class="' + cls + '" title="CB policy rate \u00b7 as of ' + (cb.date || '\u2014') + '">' + cb.rate.toFixed(2) + '%</td>';
    } else {
      html += '<td class="flat" title="No data available">\u2014</td>';
    }
    html += '</tr>';
    return html;
  }

  let _loaded = false;

  async function loadEconMatrix() {
    if (_loaded) return;
    _loaded = true;

    const tbody = document.getElementById('econmx-tbody');
    const sub   = document.getElementById('econmx-sub');

    try {
      const [cal, y10All, cbAll] = await Promise.all([
        loadCalendarData(),
        Promise.all(CCY_ORDER.map(load10y)),
        Promise.all(CCY_ORDER.map(getCBRate)),
      ]);

      if (!cal) {
        if (sub) sub.textContent = 'Economic Calendar \u00b7 data unavailable';
        return;
      }

      const rows = CCY_ORDER.map((ccy, i) => rowHTML(ccy, cal.byCategory[ccy] || {}, y10All[i], cbAll[i]));
      if (tbody) tbody.innerHTML = rows.join('');

      if (sub) {
        let label = 'Economic Calendar \u00b7 latest actuals \u00b7 G10';
        if (cal.lastUpdate) {
          const d = new Date(cal.lastUpdate);
          if (!isNaN(d)) label += ' \u00b7 updated ' + d.toLocaleDateString('en', { day: '2-digit', month: 'short' });
        }
        sub.textContent = label;
      }
    } catch (e) {
      if (sub) sub.textContent = 'Economic Calendar \u00b7 data unavailable';
    }
  }

  function attach() {
    const section = document.getElementById('section-econmap');
    if (!section) return;
    if (typeof IntersectionObserver === 'undefined') {
      loadEconMatrix();
      return;
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          loadEconMatrix();
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '150px' });
    io.observe(section);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
}());
