/*
 * pair-detail-beta.js v0.2.0
 * Unified Pair Detail panel (beta) — rendered below the main chart for the active pair.
 * Tabs: Overview · Macro Drivers · Session Context · Strength Drivers · Fair Value.
 *
 * Data sources (all already published by the engine, no new endpoints):
 *   ./ai-analysis/currency-catalysts.json  -> Macro Drivers   (per currency)
 *   ./ai-analysis/session-context.json     -> Session Context (per currency x session)
 *   ./ai-analysis/currency-drivers.json    -> Strength Drivers (per currency x pair note)
 *   ./fair-value-data/summary.json         -> Fair Value / Overview (per pair)
 *
 * Overview embeds the legacy per-row detail (dashboard.js buildInlineDetail: price, Price & Spreads,
 * Volatility, COT, Retail) so nothing from the old accordion is lost, plus the new snapshots.
 * Row clicks in FX Pairs / Crosses reach this panel through the 'gi:pairDetailToggle' event.
 *
 * Rules: no inline handlers, no innerHTML with data (text is set via textContent),
 * external URLs limited to http(s), ARIA tabs pattern (roving tabindex, arrow keys).
 */
(function () {
  'use strict';

  var G10 = ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'USD', 'NOK', 'SEK'];
  var TABS = ['overview', 'macro', 'session', 'strength', 'fairvalue'];
  var SESSIONS = [
    { name: 'Sydney',   start: 21, end: 6  },
    { name: 'Tokyo',    start: 0,  end: 9  },
    { name: 'London',   start: 7,  end: 16 },
    { name: 'New York', start: 12, end: 21 }
  ];
  var REFRESH_MS = 5 * 60 * 1000;
  var LEGACY_REFRESH_MS = 30 * 1000;
  var OPEN_KEY = 'gi.pairDetail.open';
  var STALE_HOURS = 26;

  var root, tabsEl, subEl, toggleBtn, contentEl;
  var state = { pair: null, sym: null, tab: 'overview', open: true, data: null, loadedAt: 0, loading: null, legacyEl: null, legacyAt: 0 };

  /* ---------- helpers ---------- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function append(parent) {
    for (var i = 1; i < arguments.length; i++) if (arguments[i]) parent.appendChild(arguments[i]);
    return parent;
  }
  function safeUrl(u) {
    try {
      var p = new URL(u, location.href);
      return (p.protocol === 'https:' || p.protocol === 'http:') ? p.href : null;
    } catch (e) { return null; }
  }
  function fmtUtc(iso) {
    if (!iso) return null;
    var d = new Date(iso);
    if (isNaN(d)) return null;
    return String(d.getUTCHours()).padStart(2, '0') + ':' + String(d.getUTCMinutes()).padStart(2, '0') + ' UTC';
  }
  function ageHours(iso) {
    var d = new Date(iso);
    return isNaN(d) ? null : (Date.now() - d.getTime()) / 36e5;
  }
  function parseSym(sym) {
    var m = /:([A-Z]{6})$/.exec(sym || '');
    if (!m) return null;
    var base = m[1].slice(0, 3), quote = m[1].slice(3);
    if (G10.indexOf(base) < 0 || G10.indexOf(quote) < 0) return null;
    return { base: base, quote: quote, key: (base + quote).toLowerCase(), label: base + '/' + quote };
  }
  function decimals(pair) { return pair.quote === 'JPY' ? 3 : 5; }
  function signed(v, d) { return (v >= 0 ? '+' : '') + v.toFixed(d); }

  function note(msg) { return el('p', 'pdt-empty', msg); }
  function footer(source, iso) {
    var t = fmtUtc(iso);
    var txt = source + (t ? ' \u00b7 Updated ' + t : '');
    var f = el('div', 'pdt-foot', txt);
    var h = iso ? ageHours(iso) : null;
    if (h != null && h > STALE_HOURS) f.appendChild(el('span', 'pdt-stale', ' \u00b7 data older than ' + STALE_HOURS + 'h'));
    return f;
  }
  function sourcesLine(list) {
    if (!Array.isArray(list) || !list.length) return null;
    var wrap = el('div', 'pdt-sources', 'Sources: ');
    var n = 0;
    list.slice(0, 4).forEach(function (s) {
      var href = s && safeUrl(s.url);
      if (!href) return;
      if (n++) wrap.appendChild(document.createTextNode(' \u00b7 '));
      var a = el('a', null, String(s.title || s.url).slice(0, 40));
      a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer';
      wrap.appendChild(a);
    });
    return n ? wrap : null;
  }
  function block(title, bodyNodes) {
    var s = el('section', 'pdt-block');
    s.appendChild(el('h3', 'pdt-h', title));
    bodyNodes.forEach(function (n) { if (n) s.appendChild(n); });
    return s;
  }
  function twoCol(a, b) { return append(el('div', 'pdt-cols'), a, b); }

  /* ---------- data ---------- */
  function fetchJson(url) {
    return fetch(url, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.json();
    });
  }
  function load(force) {
    if (state.loading) return state.loading;
    if (!force && state.data && Date.now() - state.loadedAt < REFRESH_MS) return Promise.resolve(state.data);
    var urls = {
      catalysts: './ai-analysis/currency-catalysts.json',
      session:   './ai-analysis/session-context.json',
      drivers:   './ai-analysis/currency-drivers.json',
      fair:      './fair-value-data/summary.json'
    };
    var keys = Object.keys(urls);
    state.loading = Promise.all(keys.map(function (k) {
      return fetchJson(urls[k]).catch(function () { return null; });
    })).then(function (res) {
      var d = {};
      keys.forEach(function (k, i) { d[k] = res[i]; });
      state.data = d; state.loadedAt = Date.now(); state.loading = null;
      return d;
    });
    return state.loading;
  }

  /* ---------- tab renderers ---------- */
  function renderMacro(p, d) {
    var cat = d.catalysts;
    var nodes = [p.base, p.quote].map(function (c) {
      var e = cat && cat.currencies && cat.currencies[c];
      if (!e || !e.catalyst) return block(c + ' MACRO DRIVERS', [note('No macro driver data available yet.')]);
      return block(c + ' MACRO DRIVERS', [
        el('p', 'pdt-text', e.catalyst), sourcesLine(e.sources), footer('AI Analytics', e.updated)
      ]);
    });
    return twoCol(nodes[0], nodes[1]);
  }

  function sessionState(s, h) {
    var open = s.start < s.end ? (h >= s.start && h < s.end) : (h >= s.start || h < s.end);
    if (open) return 'live';
    return 'closed';
  }
  function renderSession(p, d) {
    var sc = d.session;
    if (!sc || !sc.sessions) return block('SESSION CONTEXT', [note('Session context not available yet.')]);
    if (sc.market_closed) {
      return block('SESSION CONTEXT', [note('Market closed. Session context resumes Sunday 21:00 UTC.')]);
    }
    var h = new Date().getUTCHours();
    var cols = [p.base, p.quote].map(function (c) {
      var notes = sc.sessions[c];
      var s = block(c + ' \u00b7 SESSION CONTEXT', []);
      if (!notes) { s.appendChild(note('No session notes for ' + c + '.')); return s; }
      SESSIONS.forEach(function (sess) {
        var st = sessionState(sess, h);
        var row = el('div', 'pdt-sess' + (st === 'live' ? ' pdt-sess-live' : ''));
        var head = el('div', 'pdt-sess-head');
        head.appendChild(el('span', 'pdt-sess-name', sess.name.toUpperCase()));
        head.appendChild(el('span', 'pdt-chip pdt-chip-' + st, st === 'live' ? 'LIVE' : 'CLOSED'));
        row.appendChild(head);
        row.appendChild(el('p', 'pdt-text', notes[sess.name] || '\u2014'));
        s.appendChild(row);
      });
      return s;
    });
    var wrap = twoCol(cols[0], cols[1]);
    wrap.appendChild(footer('AI Analytics \u00b7 session windows in UTC', sc.generated_at));
    return wrap;
  }

  function renderStrength(p, d) {
    var dr = d.drivers;
    var cols = [p.base, p.quote].map(function (c) {
      var s = block(c + ' STRENGTH DRIVERS \u00b7 ' + p.label, []);
      var notes = dr && dr.drivers && dr.drivers[c];
      var txt = notes && (notes[p.base + '/' + p.quote] || notes[p.quote + '/' + p.base]);
      if (!txt) { s.appendChild(note('No strength-driver note for this pair from the ' + c + ' side today.')); return s; }
      s.appendChild(el('p', 'pdt-text', txt));
      s.appendChild(sourcesLine(dr.driver_sources && dr.driver_sources[c]));
      return s;
    });
    var wrap = twoCol(cols[0], cols[1]);
    if (dr) wrap.appendChild(footer('AI Analytics', dr.generated_at));
    return wrap;
  }

  function statCell(label, value, cls) {
    var c = el('div', 'pdt-stat');
    c.appendChild(el('div', 'pdt-stat-l', label));
    c.appendChild(el('div', 'pdt-stat-v' + (cls ? ' ' + cls : ''), value));
    return c;
  }
  function fvNumbers(p, d) {
    var fv = d.fair && d.fair.pairs && d.fair.pairs[p.key];
    if (!fv || fv.accumulating || fv.fairValue == null || fv.z == null) return { fv: fv, ok: false };
    return { fv: fv, ok: true };
  }
  function gauge(z) {
    var clamped = Math.max(-3, Math.min(3, z));
    var g = el('div', 'pdt-gauge');
    g.setAttribute('role', 'img');
    g.setAttribute('aria-label', 'Deviation from model ' + signed(z, 2) + ' standard deviations, scale minus 3 to plus 3');
    g.appendChild(el('div', 'pdt-gauge-mid'));
    var m = el('div', 'pdt-gauge-mark');
    m.style.left = ((clamped + 3) / 6 * 100).toFixed(1) + '%';
    g.appendChild(m);
    return g;
  }
  function zClass(z) { return Math.abs(z) < 1 ? '' : (z > 0 ? 'pdt-down' : 'pdt-up'); }

  function renderFairValue(p, d) {
    var r = fvNumbers(p, d), fv = r.fv, dec = decimals(p);
    if (!fv) return block('FX FAIR VALUE \u00b7 ' + p.label, [note('No fair-value model output for this pair.')]);
    if (!r.ok) {
      return block('FX FAIR VALUE \u00b7 ' + p.label, [
        note('Model still accumulating history for this pair (' + fv.usableRows + ' usable rows).'),
        footer('Fair Value model', d.fair.generated_at)
      ]);
    }
    var grid = el('div', 'pdt-stats');
    grid.appendChild(statCell('Spot', fv.spot.toFixed(dec)));
    grid.appendChild(statCell('Model fair value', fv.fairValue.toFixed(dec)));
    grid.appendChild(statCell('Deviation', signed(fv.z, 2) + '\u03c3', zClass(fv.z)));
    grid.appendChild(statCell('Rate diff', fv.rate_diff != null ? signed(fv.rate_diff, 2) : '\u2014', fv.rate_diff != null ? (fv.rate_diff >= 0 ? 'pdt-up' : 'pdt-down') : ''));
    var meta = el('p', 'pdt-text pdt-muted',
      'Fit: ' + (fv.identifiable ? 'Solid' : 'Regularized') +
      ' \u00b7 ' + fv.usableRows + ' of ' + fv.totalRows + ' rows usable' +
      ' \u00b7 rolling ' + d.fair.rolling_window + '-day window. ' +
      'Model estimate of where spot sits relative to its fundamentals; descriptive, not a price forecast.');
    return block('FX FAIR VALUE \u00b7 ' + p.label, [grid, gauge(fv.z), meta, footer('Fair Value model', d.fair.generated_at)]);
  }

  function legacyDetail() {
    var host = el('div', 'pdt-legacy');
    state.legacyEl = host;
    if (typeof window.buildInlineDetail === 'function' && state.sym) {
      host.appendChild(el('div', 'pdt-empty', 'Loading\u2026'));
      state.legacyAt = Date.now();
      try { Promise.resolve(window.buildInlineDetail(state.sym, host)).catch(function () {}); } catch (e) { /* legacy renderer unavailable */ }
    }
    return host;
  }
  function refreshLegacy() {
    if (!state.open || state.tab !== 'overview' || !state.legacyEl || !state.legacyEl.isConnected) return;
    if (Date.now() - state.legacyAt < LEGACY_REFRESH_MS) return;
    state.legacyAt = Date.now();
    try { Promise.resolve(window.buildInlineDetail(state.sym, state.legacyEl)).catch(function () {}); } catch (e) { /* keep last render */ }
  }

  function renderOverview(p, d) {
    var r = fvNumbers(p, d), dec = decimals(p);
    var nodes = [];
    if (r.ok) {
      var grid = el('div', 'pdt-stats');
      grid.appendChild(statCell('Spot', r.fv.spot.toFixed(dec)));
      grid.appendChild(statCell('Model fair value', r.fv.fairValue.toFixed(dec)));
      grid.appendChild(statCell('Deviation', signed(r.fv.z, 2) + '\u03c3', zClass(r.fv.z)));
      grid.appendChild(statCell('Fit', r.fv.identifiable ? 'Solid' : 'Regularized'));
      nodes.push(block('FX FAIR VALUE \u00b7 SNAPSHOT', [grid, gauge(r.fv.z), footer('Fair Value model', d.fair.generated_at)]));
    } else {
      nodes.push(block('FX FAIR VALUE \u00b7 SNAPSHOT', [note('Model output not available for this pair yet.')]));
    }
    var h = new Date().getUTCHours();
    var live = SESSIONS.filter(function (s) { return sessionState(s, h) === 'live'; }).pop();
    var sc = d.session && d.session.sessions && !d.session.market_closed;
    var sessNodes = [p.base, p.quote].map(function (c) {
      var t = sc && live && d.session.sessions[c] && d.session.sessions[c][live.name];
      return t ? el('p', 'pdt-text', c + ' \u00b7 ' + live.name + ': ' + t) : null;
    }).filter(Boolean);
    nodes.push(block('LIVE SESSION', sessNodes.length ? sessNodes.concat([footer('AI Analytics', d.session.generated_at)]) : [note('No live-session note available.')]));
    var wrap = el('div', 'pdt-overview');
    wrap.appendChild(legacyDetail());
    wrap.appendChild(twoCol(nodes[0], nodes[1]));
    return wrap;
  }

  var RENDER = { overview: renderOverview, macro: renderMacro, session: renderSession, strength: renderStrength, fairvalue: renderFairValue };

  /* ---------- panel plumbing ---------- */
  function panelEl(tab) { return document.getElementById('pdt-p-' + tab); }

  function renderTab(tab) {
    var host = panelEl(tab);
    if (!host) return;
    if (!state.open) return;
    host.textContent = '';
    if (!state.pair) { host.appendChild(note('Detail is available for G10 currency pairs.')); return; }
    if (!state.data) { host.appendChild(note('Loading\u2026')); return; }
    host.appendChild(RENDER[tab](state.pair, state.data));
  }

  function selectTab(tab, focus) {
    if (TABS.indexOf(tab) < 0) return;
    state.tab = tab;
    TABS.forEach(function (t) {
      var btn = document.getElementById('pdt-t-' + t), pnl = panelEl(t), on = t === tab;
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
      btn.tabIndex = on ? 0 : -1;
      btn.classList.toggle('on', on);
      pnl.hidden = !on;
    });
    if (focus) document.getElementById('pdt-t-' + tab).focus();
    renderTab(tab);
  }

  function setOpen(open, persist) {
    var was = state.open;
    state.open = !!open;
    toggleBtn.setAttribute('aria-expanded', state.open ? 'true' : 'false');
    contentEl.setAttribute('data-open', state.open ? 'true' : 'false');
    if (state.open) contentEl.removeAttribute('inert'); else contentEl.setAttribute('inert', '');
    if (persist !== false) {
      try { localStorage.setItem(OPEN_KEY, state.open ? 'true' : 'false'); } catch (e) { /* storage unavailable */ }
    }
    if (state.open && !was) renderTab(state.tab);
  }

  function onRowToggle(e) {
    var sym = e.detail && e.detail.sym, p = parseSym(sym);
    if (!p) return;
    var same = state.pair && state.pair.key === p.key;
    if (same && state.open) { setOpen(false); return; }
    state.sym = sym;
    if (!same) setPair(sym, true);
    setOpen(true);
    if (root.scrollIntoView) {
      var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      root.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' });
    }
  }

  function setPair(sym, force) {
    var p = parseSym(sym);
    var key = p ? p.key : null;
    if (!force && (state.pair && state.pair.key) === key && state.data) return;
    state.pair = p;
    state.sym = p ? sym : null;
    subEl.textContent = p ? p.label : '\u2014';
    var before = state.loadedAt;
    renderTab(state.tab);
    load(false).then(function () { if (state.loadedAt !== before) renderTab(state.tab); });
  }

  function currentSym() {
    var a = document.querySelector('#tv-pair-tabs .tv-tab.active');
    return a ? a.getAttribute('data-sym') : null;
  }

  function init() {
    root = document.getElementById('pdt-panel');
    toggleBtn = document.getElementById('pdt-toggle');
    contentEl = document.getElementById('pdt-content');
    tabsEl = document.getElementById('pdt-tabs');
    subEl = document.getElementById('pdt-sub');
    if (!root || !tabsEl || !subEl || !toggleBtn || !contentEl) return;

    toggleBtn.addEventListener('click', function () { setOpen(!state.open); });
    document.addEventListener('gi:pairDetailToggle', onRowToggle);

    tabsEl.addEventListener('click', function (e) {
      var b = e.target.closest('[data-pdt-tab]');
      if (b) selectTab(b.getAttribute('data-pdt-tab'), false);
    });
    tabsEl.addEventListener('keydown', function (e) {
      var i = TABS.indexOf(state.tab), n = null;
      if (e.key === 'ArrowRight') n = (i + 1) % TABS.length;
      else if (e.key === 'ArrowLeft') n = (i - 1 + TABS.length) % TABS.length;
      else if (e.key === 'Home') n = 0;
      else if (e.key === 'End') n = TABS.length - 1;
      if (n == null) return;
      e.preventDefault();
      selectTab(TABS[n], true);
    });

    var chartTabs = document.getElementById('tv-pair-tabs');
    if (chartTabs && 'MutationObserver' in window) {
      new MutationObserver(function () { setPair(currentSym()); })
        .observe(chartTabs, { attributes: true, subtree: true, attributeFilter: ['class'] });
    }
    document.addEventListener('gi:quotesLoaded', function () { if (!state.pair) setPair(currentSym()); else refreshLegacy(); });

    var stored = null;
    try { stored = localStorage.getItem(OPEN_KEY); } catch (e) { /* storage unavailable */ }
    setOpen(stored !== 'false', false);
    selectTab('overview', false);
    setPair(currentSym());
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
