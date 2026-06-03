// Disable browser scroll-position restoration so our explicit scrollTop = 0 calls
// in boot() are never overridden by the browser restoring a previous scroll position.
// Must be set before any scroll resets run. Standard pattern for dashboard/SPA pages.
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

// ═══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════
const STATE = {
  rates: {},      // Frankfurter rates (USD base)
  prevRates: {},  // Yesterday's rates for % change
  cbRates: {},    // Central bank rates from rates/*.json
  cotData: {},    // COT data from cot-data/*.json
};

// Currency config: which pairs to compute from Frankfurter USD-base
const PAIRS = [
  { id:'eurusd', base:'EUR', quote:'USD', invert:true,  dec:5, label:'EUR/USD' },
  { id:'gbpusd', base:'GBP', quote:'USD', invert:true,  dec:5, label:'GBP/USD' },
  { id:'usdjpy', base:'JPY', quote:'USD', invert:false, dec:3, label:'USD/JPY' },
  { id:'audusd', base:'AUD', quote:'USD', invert:true,  dec:5, label:'AUD/USD' },
  { id:'usdchf', base:'CHF', quote:'USD', invert:false, dec:5, label:'USD/CHF' },
  { id:'usdcad', base:'CAD', quote:'USD', invert:false, dec:5, label:'USD/CAD' },
  { id:'nzdusd', base:'NZD', quote:'USD', invert:true,  dec:5, label:'NZD/USD' },
  { id:'eurgbp', base:'EUR', quote:'GBP', cross:['EUR','GBP'], dec:5 },
  { id:'eurjpy', base:'EUR', quote:'JPY', cross:['EUR','JPY'], dec:3 },
  { id:'eurchf', base:'EUR', quote:'CHF', cross:['EUR','CHF'], dec:5 },
  { id:'eurcad', base:'EUR', quote:'CAD', cross:['EUR','CAD'], dec:5 },
  { id:'euraud', base:'EUR', quote:'AUD', cross:['EUR','AUD'], dec:5 },
  { id:'gbpjpy', base:'GBP', quote:'JPY', cross:['GBP','JPY'], dec:3 },
  { id:'gbpchf', base:'GBP', quote:'CHF', cross:['GBP','CHF'], dec:5 },
  { id:'gbpcad', base:'GBP', quote:'CAD', cross:['GBP','CAD'], dec:5 },
  { id:'audjpy', base:'AUD', quote:'JPY', cross:['AUD','JPY'], dec:3 },
  { id:'audnzd', base:'AUD', quote:'NZD', cross:['AUD','NZD'], dec:5 },
  { id:'audchf', base:'AUD', quote:'CHF', cross:['AUD','CHF'], dec:5 },
  { id:'cadjpy', base:'CAD', quote:'JPY', cross:['CAD','JPY'], dec:3 },
  { id:'chfjpy', base:'CHF', quote:'JPY', cross:['CHF','JPY'], dec:3 },
  { id:'nzdjpy', base:'NZD', quote:'JPY', cross:['NZD','JPY'], dec:3 },
  { id:'eurnzd', base:'EUR', quote:'NZD', cross:['EUR','NZD'], dec:5 },
  { id:'gbpaud', base:'GBP', quote:'AUD', cross:['GBP','AUD'], dec:5 },
  { id:'gbpnzd', base:'GBP', quote:'NZD', cross:['GBP','NZD'], dec:5 },
  { id:'audcad', base:'AUD', quote:'CAD', cross:['AUD','CAD'], dec:5 },
  { id:'cadchf', base:'CAD', quote:'CHF', cross:['CAD','CHF'], dec:5 },
  { id:'nzdcad', base:'NZD', quote:'CAD', cross:['NZD','CAD'], dec:5 },
  { id:'nzdchf', base:'NZD', quote:'CHF', cross:['NZD','CHF'], dec:5 },
];

// CB rate config
const CB_CONFIG = [
  { id:'usd', file:'USD', label:'Fed (US)' },
  { id:'eur', file:'EUR', label:'ECB (EU)' },
  { id:'gbp', file:'GBP', label:'BoE (UK)' },
  { id:'jpy', file:'JPY', label:'BoJ (JP)' },
  { id:'aud', file:'AUD', label:'RBA (AU)' },
  { id:'chf', file:'CHF', label:'SNB (CH)' },
  { id:'cad', file:'CAD', label:'BoC (CA)' },
  { id:'nzd', file:'NZD', label:'RBNZ (NZ)' },
];

// COT currencies available
const COT_CURRENCIES = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];

// ═══════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════
function fmt(val, dec) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(dec);
}

function clsDir(val) {
  if (val > 0.0001) return 'up';
  if (val < -0.0001) return 'down';
  return 'flat';
}

function pctStr(val) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(2) + '%';
}

function setEl(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (cls) el.className = cls;
}

// ═══════════════════════════════════════════════════════════════════
// CLOCK & SESSION
// ═══════════════════════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  // Local time for display
  const lh = now.getHours(), lm = now.getMinutes(), ls = now.getSeconds();
  const localStr = [lh,lm,ls].map(n=>String(n).padStart(2,'0')).join(':');
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('clock', localStr + ' ' + tzAbbr);
  // sb-clock removed (redundant with header clock)
  setEl('footer-clock', localStr);
  // Sessions use UTC internally
  updateSessions(now.getUTCHours(), now.getUTCMinutes());
}

function isOpen(openH, closeH, h) {
  return openH < closeH ? (h >= openH && h < closeH) : (h >= openH || h < closeH);
}

function updateSessions(h) {
  const sessions = [
    { id:'sydney',  open:22, close:7  },
    { id:'tokyo',   open:0,  close:9  },
    { id:'london',  open:8,  close:17 },
    { id:'newyork', open:13, close:22 },
  ];

  const now = new Date();
  const utcDay = now.getUTCDay();   // 0=Sun, 6=Sat
  const utcHour = now.getUTCHours();
  // FX market: opens Sun 21:00 UTC, closes Fri 21:00 UTC
  const isWeekend = utcDay === 6
    || (utcDay === 0 && utcHour < 21)
    || (utcDay === 5 && utcHour >= 21);

  let activeLabel = isWeekend ? 'MARKET CLOSED' : 'INTER-SESSION';

  // Convert UTC hour to local HH:MM string
  function utcHourToLocal(utcHour) {
    const d = new Date();
    d.setUTCHours(utcHour, 0, 0, 0);
    return d.toLocaleTimeString('en', {hour:'2-digit', minute:'2-digit', hour12:false});
  }

  // Update session column header to show local timezone
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'Local';
  const colOpen = document.getElementById('sess-col-open');
  const colClose = document.getElementById('sess-col-close');
  if (colOpen) colOpen.textContent = 'Open (' + tzAbbr + ')';
  if (colClose) colClose.textContent = 'Close (' + tzAbbr + ')';

  sessions.forEach(s => {
    const open = !isWeekend && isOpen(s.open, s.close, h);
    const badge = document.getElementById('sess-' + s.id);
    const status = document.getElementById('status-' + s.id);
    const openEl = document.getElementById('sess-open-' + s.id);
    const closeEl = document.getElementById('sess-close-' + s.id);
    if (badge) badge.classList.toggle('active', open);
    if (status) {
      status.textContent = isWeekend ? 'Weekend' : (open ? 'Open' : 'Closed');
      status.className = open ? 'up' : 'flat';
    }
    if (openEl) openEl.textContent = utcHourToLocal(s.open);
    if (closeEl) closeEl.textContent = utcHourToLocal(s.close);
    if (open) activeLabel = s.id.toUpperCase().replace('NEWYORK','NEW YORK');
  });

  setEl('session-label', activeLabel + (isWeekend ? '' : ' SESSION'));
  setEl('session-status', activeLabel + (isWeekend ? ' · CLOSED' : ' · ACTIVE'));
}

setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════════════════
// FRANKFURTER — ECB daily rates (read from server-side cache to avoid CORS)
// Cache is updated every 4h by the engine workflow update-frankfurter-cache.yml
// and deposited at /fx-data/frankfurter.json in the public repo.
// ═══════════════════════════════════════════════════════════════════
async function fetchFrankfurter() {
  try {
    const res = await fetch('/fx-data/frankfurter.json');
    if (!res.ok) return;
    const data = await res.json();

    STATE.rates = (data.today && data.today.rates) ? data.today.rates : {};
    STATE.prevRates = (data.prev && data.prev.rates) ? data.prev.rates : {};

    // Only use Frankfurter data to populate UI if intraday RT cache is not yet loaded
    // (avoids overwriting live yfinance prices with stale ECB daily rates)
    if (Object.keys(STOOQ_RT_CACHE).length === 0) {
      populateQuoteBar();
      populateFxPairsTable();
      populateHeatmap();
      populateCrossRows();
      const updEl = document.getElementById('fx-table-updated');
      if (updEl) updEl.textContent = 'ECB · updated ' + (data.today.date || '') + ' · daily rate';
    }
  } catch(e) {
    console.warn('Frankfurter cache fetch failed:', e);
  }
}

function getLatestBizDate() {
  const d = new Date();
  // If weekend, go to last Friday
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0,10);
}

function getPrevBizDate() {
  const d = new Date();
  // First skip to last business day (handles weekend today)
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  // Then go one more business day back
  d.setUTCDate(d.getUTCDate() - 1);
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0,10);
}

// Convert USD-base rates to any pair rate
function computeRate(pair) {
  const r = STATE.rates;
  if (!r) return null;
  if (pair.cross) {
    // Cross: e.g. EUR/GBP = (1/EUR_from_USD) / (1/GBP_from_USD)
    const [base, quote] = pair.cross;
    const baseUSD = r[base]; // how many base per USD
    const quoteUSD = r[quote];
    if (!baseUSD || !quoteUSD) return null;
    // EUR/USD = 1/baseUSD; GBP/USD = 1/quoteUSD; EUR/GBP = EUR/USD / GBP/USD
    return (1/baseUSD) / (1/quoteUSD);
  }
  if (pair.invert) {
    // USD/X → 1/X; e.g. EUR/USD = 1 / (EUR_from_USD)
    return r[pair.base] ? 1 / r[pair.base] : null;
  } else {
    // USD/X: e.g. USD/JPY = JPY_from_USD
    return r[pair.base] || null;
  }
}

function computePrevRate(pair) {
  const r = STATE.prevRates;
  if (!r || !Object.keys(r).length) return null;
  const orig = STATE.rates;
  STATE.rates = r;
  const v = computeRate(pair);
  STATE.rates = orig;
  return v;
}

function populateQuoteBar() {
  PAIRS.slice(0,8).forEach(pair => {
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);
    if (rate == null) return;
    const priceEl = document.getElementById('q-' + pair.id);
    const chgEl   = document.getElementById('qc-' + pair.id);
    if (!priceEl || !chgEl) return;
    priceEl.textContent = fmt(rate, pair.dec);
    if (prev && prev > 0) {
      const pct = (rate - prev) / prev * 100;
      chgEl.textContent  = pctStr(pct);
      const cls = clsDir(pct);
      priceEl.className  = 'q-price ' + cls;
      chgEl.className    = 'q-chg '  + cls;
    } else {
      chgEl.textContent = '+0.00%';
      priceEl.className = 'q-price flat';
      chgEl.className   = 'q-chg flat';
    }
  });

  // EUR/GBP cross in quote bar
  const egPair = PAIRS.find(p=>p.id==='eurgbp');
  const eg     = computeRate(egPair);
  const egPrev = computePrevRate(egPair);
  const egEl   = document.getElementById('q-eurgbp');
  const egcEl  = document.getElementById('qc-eurgbp');
  if (eg && egEl) {
    egEl.textContent = fmt(eg, 5);
    if (egPrev && egPrev > 0) {
      const pct = (eg - egPrev) / egPrev * 100;
      const cls = clsDir(pct);
      egEl.className  = 'q-price ' + cls;
      if (egcEl) { egcEl.textContent = pctStr(pct); egcEl.className = 'q-chg ' + cls; }
    }
  }
}


function populateCrossRows() {
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf'];
  crossIds.forEach(id => {
    const pair = PAIRS.find(p=>p.id===id);
    if (!pair) return;
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);
    const priceEl = document.getElementById('sb-' + id);
    const chgEl   = document.getElementById('sbc-' + id);
    if (priceEl && rate != null) {
      priceEl.textContent = fmt(rate, pair.dec);
      if (prev && prev > 0) {
        const pct = (rate - prev) / prev * 100;
        const cls = clsDir(pct);
        priceEl.className = 'sb-price ' + cls;
        if (chgEl) { chgEl.textContent = pctStr(pct); chgEl.className = 'sb-chg ' + cls; }
      } else {
        priceEl.className = 'sb-price flat';
        if (chgEl) { chgEl.textContent = '+0.00%'; chgEl.className = 'sb-chg flat'; }
      }
    }
  });
}

// Typical interbank spreads in pips per pair
// LIVE_SPREADS is updated by fetchReferenceSpreads() whenever the intraday JSON loads.
// Falls back to ECN_FLOOR_SPREADS (static institutional minimums) until first update.
// ECB_FLOOR values calibrated against IC Markets Razor, Pepperstone Razor, LMAX avg.
const ECN_FLOOR_SPREADS = {
  eurusd:0.1, gbpusd:0.2, usdjpy:0.1, audusd:0.2,
  usdchf:0.2, usdcad:0.2, nzdusd:0.3, eurgbp:0.5,
  eurjpy:0.5, eurchf:1.0, eurcad:0.8, euraud:1.0,
  gbpjpy:1.2, gbpchf:1.2, gbpcad:1.5,
  audjpy:0.8, audnzd:1.5, audchf:1.5,
  cadjpy:1.0, chfjpy:1.5, nzdjpy:1.8,
};
// Live spread cache — populated by fetchReferenceSpreads() from HV30+VIX+MOVE model.
// Using a Proxy so TYPICAL_SPREADS reads from LIVE_SPREADS when a key has been set,
// and from ECN_FLOOR_SPREADS as fallback. All existing code uses TYPICAL_SPREADS unchanged.
const LIVE_SPREADS = {};
const TYPICAL_SPREADS = new Proxy({}, {
  get(_, pair) {
    return LIVE_SPREADS[pair] ?? ECN_FLOOR_SPREADS[pair] ?? 0.5;
  }
});
// Repo performance data cache
const FX_PERF_CACHE = {};

// ── Key Correlations — populated from intraday-data/quotes.json (computed by Python script) ──
// Supports three selectable windows: 30d, 60d (default), 90d.
// The Python script emits corr30/corr90 alongside corr (60d) in every correlation entry.

let _corrWindow = 60;  // active window; toggled by setCorrWindow()
let _corrDataCache = []; // correlation objects cached for modal access
window._corrDataCache = _corrDataCache; // expose globally for onclick handlers

function setCorrWindow(w) {
  if (w === _corrWindow) return;
  _corrWindow = w;
  // Update button styles — active: white text on bg3 (matches .tv-tab.active); inactive: text3
  [30, 60, 90].forEach(n => {
    const btn = document.getElementById('corr-btn-' + n);
    if (!btn) return;
    btn.style.color = n === w ? '#fff' : 'var(--text3)';
  });
  // Update column header
  const th = document.getElementById('corr-th-window');
  if (th) th.textContent = w + 'd';
  // Re-render with cached data
  populateCorrelations();
}

async function populateCorrelations() {
  try {
    _corrDataCache.length = 0; // reset on each render (keeps window reference intact)
    const data = await loadIntradayQuotes();
    const tbody = document.getElementById('correlations-tbody');
    if (!tbody) return;
    const corrs = data?.correlations;
    if (!Array.isArray(corrs) || corrs.length === 0) return;

    tbody.innerHTML = corrs.map(c => {
      // Pick the value for the active window
      let v;
      if (_corrWindow === 30)      v = c.corr30 ?? c.corr ?? null;
      else if (_corrWindow === 90) v = c.corr90 ?? c.corr ?? null;
      else                         v = c.corr ?? null;

      const corrCell = v == null
        ? `<td style="color:var(--text3)">—</td>`
        : (() => {
            const sign = v >= 0 ? '+' : '';
            const cls = v >= 0.3 ? 'up' : v <= -0.3 ? 'down' : '';
            return `<td class="${cls}">${sign}${v.toFixed(2)}</td>`;
          })();

      // vs norm cell: badge based on z_score (30d Pearson vs rolling 30d-window norm — apples-to-apples)
      const z = c.z_score;
      let normCell;
      if (z == null || c.norm == null) {
        normCell = `<td style="color:var(--text3)">—</td>`;
      } else {
        const absZ = Math.abs(z);
        const normSign = c.norm >= 0 ? '+' : '';
        let badgeCls, badgeLabel;
        if (absZ >= 2.5)      { badgeCls = 'down'; badgeLabel = '⚠ break'; }
        else if (absZ >= 1.5) { badgeCls = 'warn'; badgeLabel = '~ stretched'; }
        else                  { badgeCls = 'flat'; badgeLabel = '● normal'; }
        const title = `Norm (252d): ${normSign}${c.norm.toFixed(2)} · Z-score: ${z >= 0 ? '+' : ''}${z.toFixed(2)}σ`;
        normCell = `<td class="${badgeCls}" title="${title}" style="font-size:9px;white-space:nowrap;">${badgeLabel}</td>`;
      }

      // Store corr object on window so onclick can retrieve it without embedding JSON in HTML
      const corrIdx = _corrDataCache.length;
      _corrDataCache.push(c);
      return `<tr
        style="cursor:pointer;"
        title="Click to view correlation detail · ${c.a} vs ${c.b}"
        onclick="(function(el){ var idx=+el.dataset.corrIdx; var d=window._corrDataCache&&window._corrDataCache[idx]; if(d&&typeof window.openCorrModal==='function') window.openCorrModal(d); })(this)"
        data-corr-idx="${corrIdx}"
      ><td>${c.a}</td><td>${c.b}</td>${corrCell}${normCell}</tr>`;
    }).join('');
  } catch (e) {
    console.warn('[Correlations] Failed to load:', e);
  }
}

async function loadFxPerfData() {
  // 1W CHG is now sourced directly from quotes.json (pct1w field per FX pair),
  // calculated by fetch_intraday_quotes.py using the prior-Friday-close convention.
  // This function is kept as a no-op for backward compatibility.
  // fx-performance/*.json is no longer used for the 1W column.
}

function populateFxPairsTable() {
  const tbody = document.getElementById('fx-pairs-tbody');
  if (!tbody) return;
  const _d = new Date().getUTCDay(), _h = new Date().getUTCHours();
  const isWeekend = _d === 6 || (_d === 0 && _h < 21) || (_d === 5 && _h >= 21);

  const rows = PAIRS.filter(p=>!p.cross).map(pair => {
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);

    // 1D change — primary source: RT cache (quotes.json yfinance, real prev_close)
    // Fallback: ECB Frankfurter (only if RT cache is not yet available)
    let chg1d = '—', cls1d = 'flat';
    const rtD1 = STOOQ_RT_CACHE[pair.id];
    if (rtD1?.pct != null) {
      chg1d = pctStr(rtD1.pct);
      cls1d = clsDir(rtD1.pct);
    } else if (rate != null && prev && prev > 0) {
      const pct = (rate - prev) / prev * 100;
      chg1d = pctStr(pct);
      cls1d = clsDir(pct);
    }

    // 1W change — from quotes.json pct1w field (prior-Friday-close convention)
    // Calculated by fetch_intraday_quotes.py every 5 min via yfinance daily history.
    // pct1w is already expressed as % change of the pair (EUR/USD positive = pair up,
    // USD/JPY positive = pair up — yfinance USDJPY=X goes up when USD strengthens).
    // No inversion needed: yfinance returns the pair's own price, so pct1w directly
    // reflects the pair's move.
    let chg1w = '—', cls1w = 'flat';
    const rtD1w = STOOQ_RT_CACHE[pair.id];
    if (rtD1w?.pct1w != null) {
      chg1w = pctStr(rtD1w.pct1w);
      cls1w = clsDir(rtD1w.pct1w);
    }

    // Bid / Ask — rate ± half-spread
    const pipVal = pair.dec === 3 ? 0.01 : 0.0001;
    const spreadPips = TYPICAL_SPREADS[pair.id] || 0.5;
    const halfSpread = spreadPips * pipVal / 2;
    const bid = rate != null ? fmt(rate - halfSpread, pair.dec) : '—';
    const ask = rate != null ? fmt(rate + halfSpread, pair.dec) : '—';
    const spreadStr = rate != null ? spreadPips.toFixed(1) : '—';

    // HV30 — 30-day historical volatility computed by fetch_intraday_quotes.py
    // Fuente: quotes.json campo hv30 por par, inyectado en STOOQ_RT_CACHE
    // Replaces hardcoded EST_IV. Shows '—' if not yet available.
    const rtDhv = STOOQ_RT_CACHE[pair.id];
    const hv30val = rtDhv?.hv30 ?? null;
    const ivStr = hv30val != null ? hv30val.toFixed(1) + '%' : '—';

    // Session High/Low — from intraday RT cache (STOOQ_RT_CACHE populated by yfinance JSON).
    // Prefer session_high/session_low (21:00 UTC FX session boundary, same as fetch_ohlc.py
    // historical bars) over high/low (Yahoo UTC-midnight cutoff, which excludes Tokyo/Sydney
    // open hours 21:00–23:59 UTC). Falls back to high/low if session values are null.
    const rtD = STOOQ_RT_CACHE[pair.id];
    const sessH = (rtD?.session_high != null) ? fmt(rtD.session_high, pair.dec) : (rtD?.high != null) ? fmt(rtD.high, pair.dec) : '—';
    const sessL = (rtD?.session_low  != null) ? fmt(rtD.session_low,  pair.dec) : (rtD?.low  != null) ? fmt(rtD.low,  pair.dec) : '—';
    const sessStyle = isWeekend ? 'color:var(--text3);font-size:10px' : 'color:var(--text1);font-size:10px';

    const rateFmt = rate != null ? fmt(rate, pair.dec) : '—';

    const tvSym = pair.invert
      ? `FX_IDC:${pair.base}${pair.quote}`
      : `FX_IDC:${pair.quote}${pair.base}`;
    return `<tr data-sym="${tvSym}" style="cursor:pointer;" title="Click: chart + expand detail · Click again: collapse">
      <td class="sym" style="font-weight:600">${pair.label || (pair.base+'/'+pair.quote)}</td>
      <td style="color:var(--text1)">${bid}</td>
      <td style="color:var(--text1)">${ask}</td>
      <td style="color:var(--text3);font-size:10px">${spreadStr}</td>
      <td class="${cls1d}">${chg1d}</td>
      <td class="${cls1w}">${chg1w}</td>
      <td style="color:var(--text2);font-size:10px">${ivStr}</td>
      <td style="color:var(--text3);font-size:10px">—</td>
      <td style="color:var(--text3);font-size:10px">—</td>
      <td style="color:var(--text3);font-size:10px">—</td>
      <td style="${sessStyle}">${sessH}</td>
      <td style="${sessStyle}">${sessL}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join('');
  const upd = document.getElementById('fx-table-updated');
  if (upd) {
    const now = new Date();
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    upd.textContent = 'ECB · ' + now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',hour12:false}) + ' ' + tzAbbr + (isWeekend ? ' · Last close: Fri' : '');
  }
}

// Throttle guard for populateHeatmap — Finnhub sends 2-5 ticks/second across 28 pairs.
// Rebuilding the full heatmap grid on every tick causes visible jank.
// Bloomberg convention: strength panels refresh at ~1s cadence, not per-tick.
// The throttle limits DOM rebuilds to at most once per 800ms — fast enough to feel live,
// cheap enough to never block the main thread.
let _hmThrottleTimer = null;
const _HM_THROTTLE_MS = 800;

function populateHeatmap() {
  const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD'];

  // Prefer STOOQ_RT_CACHE (intraday ~5min delay) over ECB daily rates
  // because ECB daily rates have zero intraday movement (same open/close on weekends)
  const rtAvailable = Object.keys(STOOQ_RT_CACHE).length >= 21; // need ≥75% of 28 pairs for reliable composite

  let strengths;
  if (rtAvailable) {
    // Map each currency to its avg % change across all 28 G8 pairs.
    // Each currency appears in exactly 7 pairs — equal statistical weight.
    const pctMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0 };
    const countMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0 };

    // All 28 G8 pairs (8×7÷2) — industry-standard currency strength calculation.
    // Each of the 8 currencies appears in exactly 7 pairs, giving equal statistical weight.
    // sign: +1 means base strengthens when price rises; -1 means quote strengthens.
    const pairDefs = [
      // 7 USD majors
      { id: 'eurusd', base: 'EUR', quote: 'USD', sign: 1 },
      { id: 'gbpusd', base: 'GBP', quote: 'USD', sign: 1 },
      { id: 'audusd', base: 'AUD', quote: 'USD', sign: 1 },
      { id: 'nzdusd', base: 'NZD', quote: 'USD', sign: 1 },
      { id: 'usdjpy', base: 'USD', quote: 'JPY', sign: -1 },
      { id: 'usdchf', base: 'USD', quote: 'CHF', sign: -1 },
      { id: 'usdcad', base: 'USD', quote: 'CAD', sign: -1 },
      // 6 EUR crosses
      { id: 'eurgbp', base: 'EUR', quote: 'GBP', sign: 1 },
      { id: 'eurjpy', base: 'EUR', quote: 'JPY', sign: 1 },
      { id: 'eurchf', base: 'EUR', quote: 'CHF', sign: 1 },
      { id: 'eurcad', base: 'EUR', quote: 'CAD', sign: 1 },
      { id: 'euraud', base: 'EUR', quote: 'AUD', sign: 1 },
      { id: 'eurnzd', base: 'EUR', quote: 'NZD', sign: 1 },
      // 5 GBP crosses
      { id: 'gbpjpy', base: 'GBP', quote: 'JPY', sign: 1 },
      { id: 'gbpchf', base: 'GBP', quote: 'CHF', sign: 1 },
      { id: 'gbpcad', base: 'GBP', quote: 'CAD', sign: 1 },
      { id: 'gbpaud', base: 'GBP', quote: 'AUD', sign: 1 },
      { id: 'gbpnzd', base: 'GBP', quote: 'NZD', sign: 1 },
      // 4 AUD crosses
      { id: 'audjpy', base: 'AUD', quote: 'JPY', sign: 1 },
      { id: 'audchf', base: 'AUD', quote: 'CHF', sign: 1 },
      { id: 'audcad', base: 'AUD', quote: 'CAD', sign: 1 },
      { id: 'audnzd', base: 'AUD', quote: 'NZD', sign: 1 },
      // 3 NZD crosses
      { id: 'nzdjpy', base: 'NZD', quote: 'JPY', sign: 1 },
      { id: 'nzdchf', base: 'NZD', quote: 'CHF', sign: 1 },
      { id: 'nzdcad', base: 'NZD', quote: 'CAD', sign: 1 },
      // 2 CAD crosses
      { id: 'cadjpy', base: 'CAD', quote: 'JPY', sign: 1 },
      { id: 'cadchf', base: 'CAD', quote: 'CHF', sign: 1 },
      // 1 CHF cross
      { id: 'chfjpy', base: 'CHF', quote: 'JPY', sign: 1 },
    ];

    pairDefs.forEach(({ id, base, quote, sign }) => {
      const d = STOOQ_RT_CACHE[id];
      if (!d || !d.pct) return;
      const p = d.pct * sign;
      if (base in pctMap)  { pctMap[base]  += p;  countMap[base]++;  }
      if (quote in pctMap) { pctMap[quote] -= p;  countMap[quote]++; }
    });

    // Average out each currency
    strengths = ccys.map(ccy => ({
      ccy,
      pct: countMap[ccy] > 0 ? pctMap[ccy] / countMap[ccy] : 0
    }));
  } else {
    // Fallback: ECB daily rates
    const r = STATE.rates;
    const p = STATE.prevRates;
    strengths = ccys.map(ccy => {
      if (ccy === 'USD') return { ccy, pct: 0 };
      const cur  = r[ccy];
      const prev = p[ccy];
      if (!cur) return { ccy, pct: 0 };
      const rateCur  = 1 / cur;
      const ratePrev = prev ? 1 / prev : null;
      const pct = ratePrev ? (rateCur - ratePrev) / ratePrev * 100 : 0;
      return { ccy, pct };
    });
    strengths.find(s=>s.ccy==='USD').pct = -strengths.filter(s=>s.ccy!=='USD').reduce((a,b)=>a+b.pct,0)/7;
  }

  strengths.sort((a,b)=>b.pct-a.pct);

  const grid = document.getElementById('heatmap-grid');
  if (!grid) return;
  // Store strengths in a module-level variable so the modal can read them
  // without embedding JSON in an HTML attribute (which breaks on double-quotes).
  window._hmStrengths = strengths;
  grid.innerHTML = strengths.map(s => {
    let bg = 'h-flat';
    if (s.pct > 0.15) bg = 'h-s-up';
    else if (s.pct > 0.05) bg = 'h-up';
    else if (s.pct < -0.15) bg = 'h-s-down';
    else if (s.pct < -0.05) bg = 'h-down';
    const cls = s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat';
    const sign = s.pct >= 0 ? '+' : '';
    return `<div class="hm-cell ${bg}" role="button" tabindex="0" aria-label="${s.ccy} currency strength ${sign}${s.pct.toFixed(2)}%" style="cursor:pointer" title="Click to open ${s.ccy} breakdown · 7 direct pairs · COT · vol · correlations" onclick="if(window.openHeatmapModal)openHeatmapModal('${s.ccy}',window._hmStrengths,STOOQ_RT_CACHE)">
      <span class="hm-sym">${s.ccy}</span>
      <span class="hm-val ${cls}">${sign}${s.pct.toFixed(2)}</span>
    </div>`;
  }).join('');

  // ── Heatmap source label — reflects active data source (Finnhub live vs yfinance) ──
  // Located in the panel subtitle below the heatmap title.
  const _hasFhHm = Object.values(STOOQ_RT_CACHE).some(e => e?.fromFinnhub);
  const _hmSubEl = document.getElementById('hm-panel-sub');
  if (_hmSubEl) {
    _hmSubEl.textContent = _hasFhHm
      ? 'Finnhub \u00b7 live \u00b7 28-pair equal-weighted \u00b7 8 G8 currencies'
      : 'yfinance \u00b7 ~5min delay \u00b7 28-pair equal-weighted \u00b7 8 G8 currencies';
  }

  // ── Live-refresh open modal — if the heatmap modal is currently open, push ──
  // the latest strengths and RT cache so all tabs reflect Finnhub live prices.
  // Only refreshes the active tab to avoid jank on tabs the user isn't viewing.
  if (typeof window._hmRefreshIfOpen === 'function') {
    window._hmRefreshIfOpen(strengths, STOOQ_RT_CACHE);
  }
}

// Throttled entry point — called by updateFxPairsTableRT() on every Finnhub tick.
// Direct calls (boot, full refresh) bypass the throttle by calling populateHeatmap() directly.
function populateHeatmapThrottled() {
  if (_hmThrottleTimer) return; // already scheduled — skip
  _hmThrottleTimer = setTimeout(() => {
    _hmThrottleTimer = null;
    populateHeatmap();
  }, _HM_THROTTLE_MS);
}

// ═══════════════════════════════════════════════════════════════════
// CENTRAL BANK RATES — from rates/*.json
// ═══════════════════════════════════════════════════════════════════

/**
 * Compute CB trend direction dynamically from rates/*.json observations.
 * Uses two-layer logic matching the workflow bias detection standard:
 *
 * Layer 1 — Recent momentum: did the rate move in the last ~90 days?
 *   If obs[0] is older than PAUSE_DAYS, skip — stale data should not imply trend.
 *   If rate rose vs obs[1] or obs[2] → 'up'. If fell → 'down'.
 *
 * Layer 2 — Pause detection: if the rate has been flat for PAUSE_DAYS or more,
 *   return 'flat' regardless of the longer-run direction.
 *   This prevents the ECB (last cut Jun 2025, ~10 months ago) from showing ↓.
 *
 * Returns 'up' | 'down' | 'flat'.
 */
function computeCBTrend(obs) {
  if (!obs || obs.length < 2) return 'flat';
  const PAUSE_DAYS = 90;  // 3 months — consistent with workflow PAUSE_MONTHS = 3
  const today = new Date();

  const latest = parseFloat(obs[0].value);
  if (isNaN(latest)) return 'flat';

  // Age of the most recent data point in days
  const d0 = new Date(obs[0].date);
  const dataAgeDays = (today - d0) / 86400000;

  const r1 = obs.length > 1 ? parseFloat(obs[1].value) : latest;
  const r2 = obs.length > 2 ? parseFloat(obs[2].value) : r1;

  // Layer 1: only apply momentum if the data is recent enough
  if (dataAgeDays <= PAUSE_DAYS) {
    const recentUp   = latest > r1 || latest > r2;
    const recentDown = latest < r1 || latest < r2;
    if (recentUp  && !recentDown) return 'up';
    if (recentDown && !recentUp)  return 'down';
  }

  // Layer 2: count consecutive flat months from obs[0]
  let flatMonths = 0;
  for (let i = 1; i < obs.length; i++) {
    if (parseFloat(obs[i].value) === latest) flatMonths++;
    else break;
  }
  // effective flat = max(consecutive flat periods, data age in months − 1)
  const dataAgeMonths = Math.floor(dataAgeDays / 30);
  const effectiveFlat = Math.max(flatMonths, dataAgeMonths - 1);
  if (effectiveFlat >= 3) return 'flat';

  // Short pause: use 6-obs trend direction as tiebreaker
  const oldest = parseFloat(obs[Math.min(5, obs.length - 1)].value);
  if (!isNaN(oldest)) {
    if (latest - oldest >=  0.05) return 'up';
    if (latest - oldest <= -0.05) return 'down';
  }
  return 'flat';
}

async function fetchCBRates() {
  const promises = CB_CONFIG.map(async cfg => {
    try {
      const r = await fetch('./rates/' + cfg.file + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      const obs = data.observations;
      if (!obs || !obs.length) return null;
      return { id: cfg.id, label: cfg.label, rate: parseFloat(obs[0].value), date: obs[0].date, obs };
    } catch { return null; }
  });

  const results = await Promise.all(promises);

  // Populate sidebar CB rates
  results.forEach(res => {
    if (!res) return;
    STATE.cbRates[res.id] = res;
    setEl('cbr-' + res.id, res.rate.toFixed(2) + '%');
  });

  // Populate right-panel CB rates table
  // Expose cbRates state globally so the modal can access obs arrays on click
  window._STATE_cbRates = STATE.cbRates;

  const tbody = document.getElementById('cbrates-tbody');
  if (tbody) {
    const bankInfo = {
      usd: { flag: 'us', name: 'Federal Reserve',          short: 'Fed'  },
      eur: { flag: 'eu', name: 'European Central Bank',    short: 'ECB'  },
      gbp: { flag: 'gb', name: 'Bank of England',          short: 'BoE'  },
      jpy: { flag: 'jp', name: 'Bank of Japan',            short: 'BoJ'  },
      aud: { flag: 'au', name: 'Reserve Bank of Australia',short: 'RBA'  },
      chf: { flag: 'ch', name: 'Swiss National Bank',      short: 'SNB'  },
      cad: { flag: 'ca', name: 'Bank of Canada',           short: 'BoC'  },
      nzd: { flag: 'nz', name: 'Reserve Bank of NZ',       short: 'RBNZ' },
    };
    // Expose bankInfo globally so onclick handlers can look it up without embedding JSON in HTML
    window._STATE_bankInfo = bankInfo;
    const trendMap = { up:'<span class="up">↑</span>', down:'<span class="down">↓</span>', flat:'<span class="flat">—</span>' };
    tbody.innerHTML = results.filter(Boolean).map(res => {
      const info      = bankInfo[res.id] || { flag: '', name: res.label, short: res.label };
      const trend     = computeCBTrend(res.obs);
      const flag      = info.flag ? `<span class="fi fi-${info.flag}" style="margin-right:5px;border-radius:2px;"></span>` : '';
      const rateClass = trend === 'up' ? 'up' : trend === 'down' ? 'down' : '';
      return `<tr
        title="Click to view rate history · ${info.name}"
        style="cursor:pointer;"
        data-cbr-id="${res.id}"
        onclick="(function(el){
          var id  = el.dataset.cbrId;
          var st  = window._STATE_cbRates;
          var r   = st && st[id];
          if (!r || typeof window.openCBRatesModal !== 'function') return;
          var bi  = (window._STATE_bankInfo && window._STATE_bankInfo[id]) || {};
          var mtg = window._STATE_meetings && window._STATE_meetings.meetings && window._STATE_meetings.meetings[id.toUpperCase()];
          window.openCBRatesModal(id.toUpperCase(), r.obs, bi, mtg);
        })(this)"
      >
        <td style="white-space:nowrap;">${flag}<span style="font-size:10px;">${info.short}</span></td>
        <td${rateClass ? ` class="${rateClass}"` : ''}>${res.rate.toFixed(2)}%</td>
        <td>${trendMap[trend]||'—'}</td>
      </tr>`;
    }).join('');
  }
}
// ═══════════════════════════════════════════════════════════════════
// COT DATA — from cot-data/*.json
// ═══════════════════════════════════════════════════════════════════
// TradingView COT chart symbols — CFTC Traders in Financial Futures (TFF) report
// COT3 prefix = Financial/TFF report · suffix _FO_LMP_L = Futures+Options Combined · Leveraged Funds · Long
// This matches the panel data source: CFTC Disaggregated TFF · Leveraged Funds · Options+Futures Combined
// Codes: EUR=099741, GBP=096742, JPY=097741, AUD=232741,
//        CAD=090741, CHF=092741, NZD=112741, USD=098662 (US Dollar Index futures)
const COT_TV_SYMBOLS = {
  EUR: 'COT3:099741_FO_LMP_L',
  GBP: 'COT3:096742_FO_LMP_L',
  JPY: 'COT3:097741_FO_LMP_L',
  AUD: 'COT3:232741_FO_LMP_L',
  CAD: 'COT3:090741_FO_LMP_L',
  CHF: 'COT3:092741_FO_LMP_L',
  NZD: 'COT3:112741_FO_LMP_L',
  USD: 'COT3:098662_FO_LMP_L',
};
// Short counterparts (same contract codes, suffix _FO_LMP_S)
const COT_TV_SYMBOLS_SHORT = {
  EUR: 'COT3:099741_FO_LMP_S',
  GBP: 'COT3:096742_FO_LMP_S',
  JPY: 'COT3:097741_FO_LMP_S',
  AUD: 'COT3:232741_FO_LMP_S',
  CAD: 'COT3:090741_FO_LMP_S',
  CHF: 'COT3:092741_FO_LMP_S',
  NZD: 'COT3:112741_FO_LMP_S',
  USD: 'COT3:098662_FO_LMP_S',
};

// Formats Open Interest as abbreviated number: 193390 → "193k", 1200000 → "1.2M"
function fmtOI(n) {
  if (!n || n <= 0) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1000) return Math.round(n / 1000) + 'k';
  return n.toString();
}

async function fetchCOTData() {
  const promises = COT_CURRENCIES.map(async ccy => {
    try {
      const r = await fetch('./cot-data/' + ccy + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      return { ccy, ...data };
    } catch { return null; }
  });

  const results = (await Promise.all(promises)).filter(Boolean);
  if (!results.length) return;

  // Timestamp label — "CFTC · week ending 2026-03-28 · updated Sat 04 Apr · loaded HH:MM TZ · N days ago"
  const latest = results[0];
  const weekEnd = latest.weekEnding || latest.reportDate || '';
  let updLabel = 'CFTC · week ending ' + weekEnd;
  if (latest.lastUpdate) {
    try {
      const d = new Date(latest.lastUpdate);
      if (!isNaN(d)) {
        updLabel += ' · updated ' + d.toLocaleDateString('en', { weekday: 'short', day: '2-digit', month: 'short' });
      }
    } catch {}
  }
  // Add local load timestamp so traders know data freshness in their timezone
  const _cotNow = new Date();
  const _cotHHMM = _cotNow.getHours().toString().padStart(2,'0') + ':' + _cotNow.getMinutes().toString().padStart(2,'0');
  const _cotTZ = _cotNow.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  updLabel += ' · loaded ' + _cotHHMM + ' ' + _cotTZ;

  // ── Lag indicator: days since week-ending date ──────────────────────────────
  // COT data has a structural lag: CFTC reports Tuesday positions on Friday,
  // terminal updates Saturday. Show the lag visually so traders can judge staleness.
  let lagHtml = '';
  if (weekEnd) {
    try {
      const msPerDay = 86400000;
      const lagDays  = Math.floor((_cotNow.getTime() - new Date(weekEnd + 'T00:00:00Z').getTime()) / msPerDay);
      if (lagDays >= 0) {
        // Freshness: ≤7d = green (just published), ≤14d = amber (one cycle old), >14d = red (stale)
        const lagColor = lagDays <= 7 ? 'var(--up)' : lagDays <= 14 ? '#c8952a' : 'var(--down)';
        const lagDot   = lagDays <= 7 ? '●' : lagDays <= 14 ? '◐' : '○';
        lagHtml = ` · <span title="Days since week-ending date · CFTC publishes Fri, terminal updates Sat" style="color:${lagColor};font-variant-numeric:tabular-nums;">${lagDot} ${lagDays}d lag</span>`;
      }
    } catch {}
  }

  const subEl = document.getElementById('cot-date-sub');
  if (subEl) subEl.innerHTML = updLabel + lagHtml;

  const container = document.getElementById('cot-rows');
  if (!container) return;

  // Sort rows by Long% descending — industry standard for COT panels
  // Most bullishly positioned currencies appear at the top for quick scanning
  results.sort((a, b) => {
    const totalA = (a.longPositions || 0) + (a.shortPositions || 0);
    const totalB = (b.longPositions || 0) + (b.shortPositions || 0);
    const pctA = totalA > 0 ? (a.longPositions || 0) / totalA : 0.5;
    const pctB = totalB > 0 ? (b.longPositions || 0) / totalB : 0.5;
    return pctB - pctA;
  });

  // Expose full COT data for the modal chart
  window.COT_DATA_STORE = window.COT_DATA_STORE || {};
  results.forEach(d => { window.COT_DATA_STORE[d.ccy] = d; });

  container.innerHTML = results.map(d => {
    const net   = d.netPosition || 0;
    const long  = d.longPositions || 0;
    const short = d.shortPositions || 0;
    const total = long + short;
    const longPct = total > 0 ? Math.round(long / total * 100) : 50;
    const cls   = net > 0 ? 'up' : net < 0 ? 'down' : 'flat';
    const netStr = (net >= 0 ? '+' : '') + net.toLocaleString();

    // LF vs AM divergence dot — filled = aligned, hollow = diverge
    const amNet = d.assetManagerNet;
    let divHtml = '';
    if (amNet != null) {
      const lfDir = net > 0 ? 1 : net < 0 ? -1 : 0;
      const amDir = amNet > 0 ? 1 : amNet < 0 ? -1 : 0;
      if (lfDir !== 0 && amDir !== 0) {
        if (lfDir === amDir) {
          divHtml = '<span class="cot-div aligned" title="LF + AM aligned — ' + (lfDir > 0 ? 'both net long' : 'both net short') + '">●</span>';
        } else {
          divHtml = '<span class="cot-div diverge" title="LF/AM diverge — LF ' + (net > 0 ? 'long' : 'short') + ' · AM ' + (amNet > 0 ? 'long' : 'short') + '">○</span>';
        }
      }
    }

    // Open Interest — LF long + short
    const oi    = long + short;
    const oiStr = fmtOI(oi);

    // OI direction vs prior week.
    // History is sorted chronologically oldest→newest; prior week = second-to-last entry.
    let oiArrow = '';
    if (d.history && d.history.length >= 2) {
      const prev = d.history[d.history.length - 2]; // ← fixed: was history[1]
      const prevOI = (prev.levLong || 0) + (prev.levShort || 0);
      if (prevOI > 0) {
        const delta = oi - prevOI;
        if (delta > 0)       oiArrow = '<span class="oi-up">▲</span>';
        else if (delta < 0)  oiArrow = '<span class="oi-dn">▼</span>';
      }
    }

    // Week-over-week net change — read from root if present, else derive from history.
    // History is sorted oldest→newest; prior week = second-to-last entry.
    let wow = d.wowNetChange ?? null;
    if (wow == null && d.history && d.history.length >= 2) {
      const prevSnap = d.history[d.history.length - 2];
      const prevNet  = prevSnap.levNet ?? ((prevSnap.levLong || 0) - (prevSnap.levShort || 0));
      wow = net - prevNet;
    }
    let wowHtml  = '<span class="cot-wow">—</span>';
    if (wow != null) {
      const wowCls = wow > 0 ? 'up' : wow < 0 ? 'down' : 'flat';
      const wowStr = (wow > 0 ? '+' : '') + (Math.abs(wow) >= 1000
        ? Math.round(wow / 1000) + 'k'
        : wow.toLocaleString());
      wowHtml = '<span class="cot-wow ' + wowCls + '" title="Week-over-week change in LF net contracts. Positive = specs adding longs/covering shorts. Negative = specs adding shorts/reducing longs.">' + wowStr + '</span>';
    }

    // Net as % of LF OI — read from root if present, else derive from current long+short.
    let pctOI = d.levNetPctOI ?? null;
    if (pctOI == null && oi > 0) {
      pctOI = Math.round(net / oi * 1000) / 10; // one decimal
    }
    let pctOIHtml  = '<span class="cot-pcoi">—</span>';
    if (pctOI != null) {
      const pctCls = pctOI > 0 ? 'up' : pctOI < 0 ? 'down' : 'flat';
      const pctStr = (pctOI > 0 ? '+' : '') + pctOI.toFixed(1) + '%';
      pctOIHtml = '<span class="cot-pcoi ' + pctCls + '" title="LF net as % of LF Open Interest. Normalised across currencies — comparable regardless of contract size differences.">' + pctStr + '</span>';
    }

    // TradingView COT chart symbol for row click
    const tvSym = COT_TV_SYMBOLS[d.ccy] || '';

    return '<div class="cot-row" style="cursor:pointer;" data-sym="' + tvSym + '" data-ccy="' + d.ccy + '" title="Click to open ' + d.ccy + ' COT positioning detail">'
      + '<span class="cot-sym">' + d.ccy + '</span>'
      + '<div class="cot-bar-outer">'
      + '<div class="cot-long-fill" style="width:' + longPct + '%"></div>'
      + '<div class="cot-short-fill" style="width:' + (100 - longPct) + '%"></div>'
      + '</div>'
      + '<span class="cot-pct ' + cls + '">' + longPct + '%</span>'
      + '<span class="cot-net ' + cls + '" title="LF net contracts (longs minus shorts). Positive = net long speculative positioning; negative = net short. Primary directional signal from CFTC Disaggregated TFF report.">' + netStr + '</span>'
      + wowHtml
      + divHtml
      + pctOIHtml
      + '<span class="cot-oi" title="LF Open Interest: ' + oi.toLocaleString() + ' contracts (long + short). Rising OI signals new money; falling OI signals liquidation.">' + oiArrow + oiStr + '</span>'
      + '</div>';
  }).join('');

  // Click any COT row → open institutional modal chart (fallback: TradingView widget)
  container.querySelectorAll('.cot-row[data-sym]').forEach(row => {
    row.addEventListener('click', () => {
      const ccy  = row.dataset.ccy;
      const data = window.COT_DATA_STORE && window.COT_DATA_STORE[ccy];
      if (ccy && data && typeof window.openCOTModal === 'function') {
        window.openCOTModal(ccy, data);
      } else {
        const sym = row.dataset.sym;
        if (sym) loadCOTChart(sym);
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// NEWS FEED — from news-data/news.json (RSS engine output)
// ═══════════════════════════════════════════════════════════════════
let _newsEtag = null;

async function fetchNewsData() {
  try {
    const headers = {};
    if (_newsEtag) headers['If-None-Match'] = _newsEtag;
    const r = await fetch('./news-data/news.json', { headers });
    // 304 Not Modified — no change, skip re-render
    if (r.status === 304) return;
    if (!r.ok) return;
    // Store ETag for next request
    const etag = r.headers.get('ETag');
    if (etag) _newsEtag = etag;
    const data = await r.json();
    const items = Array.isArray(data) ? data : (data.articles || data.items || []);
    if (!items.length) return;

    // Only EN articles
    const enItems = items.filter(i => !i.lang || i.lang === 'en');

    // ── NEWS TICKER
    buildNewsTicker(enItems);

    // ── NEWS SECTION (dedicated panel below narrative — always hydrates so it is ready when opened)
    renderNewsSection(enItems, data);

    // ── NEWS FEED (fill the full panel, up to 24 items)
    const feedEl = document.getElementById('news-feed-items');
    if (feedEl) {
    feedEl.innerHTML = '';
    enItems.slice(0, 24).forEach(item => {
        // Convert UTC timestamp to user's local time
        let time = item.time || '--:--';
        if (item.ts) {
          const d = new Date(item.ts);
          time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
        } else if (item.datetime) {
          const d = new Date(item.datetime);
          if (!isNaN(d)) time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
        }
        const headline = item.title || '';
        const cur      = item.cur || item.currency || '';
        const source   = item.source || '';
        // Only allow https:// links — blocks javascript: and data: URIs
        const rawLink  = item.link || '';
        const safeLink = rawLink.startsWith('https://') ? rawLink : '';
        const date     = item.date || '';
        // Build item via DOM (never innerHTML for user-controlled strings)
        const wrap = document.createElement('div');
        wrap.className = 'news-item';
        if (safeLink) {
          wrap.style.cursor = 'pointer';
          wrap.addEventListener('click', () => window.open(safeLink, '_blank', 'noopener,noreferrer'));
        }
        const timeEl = document.createElement('span');
        timeEl.className = 'news-time';
        timeEl.textContent = time;
        const bodyEl = document.createElement('div');
        bodyEl.className = 'news-body';
        const headEl = document.createElement('div');
        headEl.className = 'news-headline';
        headEl.textContent = headline;
        const metaEl = document.createElement('div');
        metaEl.className = 'news-meta';
        if (cur) { const s = document.createElement('span'); s.className = 'news-cur-tag'; s.textContent = cur; metaEl.appendChild(s); }
        if (source) { const s = document.createElement('span'); s.className = 'news-source'; s.textContent = source; metaEl.appendChild(s); }
        if (date) { const s = document.createElement('span'); s.style.color = 'var(--text3)'; s.textContent = date; metaEl.appendChild(s); }
        bodyEl.appendChild(headEl);
        bodyEl.appendChild(metaEl);
        wrap.appendChild(timeEl);
        wrap.appendChild(bodyEl);
        feedEl.appendChild(wrap);
      });

      const sub = document.getElementById('news-sub');
      if (sub && data.total) sub.textContent = `FX-relevant · ${data.total} stories · sorted by impact`;
    }
  } catch(e) {
    console.warn('News fetch failed:', e);
  }
}

function buildNewsTicker(items) {
  const track = document.getElementById('ticker-track');
  if (!track || !items.length) return;

  // Use up to 15 items; duplicate for seamless infinite loop
  const src = items.slice(0, 15);
  const makeItem = item => {
    const cur   = item.cur || item.currency || '';
    const title = item.title || '';
    const short = title.length > 90 ? title.slice(0, 87) + '\u2026' : title;
    return '<span class="ticker-item">' + (cur ? '<span class="t-tag">' + cur + '</span> \u00b7 ' : '') + short + '</span>';
  };

  // Render set A + identical set B side by side.
  // Animation scrolls exactly one full set-A width, then resets invisibly.
  track.innerHTML = src.map(makeItem).join('') + src.map(makeItem).join('');

  // Reset any running animation first
  track.style.animation = 'none';
  track.style.transform = 'translateX(0)';

  // Double rAF ensures the browser has laid out the new innerHTML before we measure
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const halfW = track.scrollWidth / 2;
      if (!halfW) return;

      const speed    = 35;  // px/s — lower = slower/more readable
      const duration = Math.max(60, halfW / speed);

      // Inject a pixel-exact keyframe so the loop jump is invisible
      const styleId = 'ticker-kf-style';
      let styleEl = document.getElementById(styleId);
      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);
      }
      styleEl.textContent =
        '@keyframes ticker-exact {' +
        '  0%   { transform: translateX(0); }' +
        '  100% { transform: translateX(-' + halfW + 'px); }' +
        '}';

      track.style.animation = 'ticker-exact ' + duration + 's linear infinite';

      // Re-measure on container resize (e.g. sidebar toggle)
      if (window._tickerRO) window._tickerRO.disconnect();
      window._tickerRO = new ResizeObserver(() => {
        const newHalf = track.scrollWidth / 2;
        if (!newHalf || Math.abs(newHalf - halfW) < 2) return;
        buildNewsTicker(items);
      });
      window._tickerRO.observe(track.parentElement);
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// AI DATA — narrative from ai-analysis/index.json,
//           signals from ai-analysis/signals.json
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// QUOTE BAR + FX TABLE — REAL-TIME FX via yfinance (intraday JSON, ~5 min delay)
// Runs every 60s. Updates quote bar, FX pairs table and heatmap.
// Falls back to Frankfurter data if yfinance JSON unavailable.
// ═══════════════════════════════════════════════════════════════════
const QB_STOOQ_PAIRS = [
  { sym: 'eurusd',  id: 'eurusd',  dec: 5 },
  { sym: 'usdjpy',  id: 'usdjpy',  dec: 3 },
  { sym: 'gbpusd',  id: 'gbpusd',  dec: 5 },
  { sym: 'audusd',  id: 'audusd',  dec: 5 },
  { sym: 'usdcad',  id: 'usdcad',  dec: 5 },
  { sym: 'usdchf',  id: 'usdchf',  dec: 5 },
  { sym: 'nzdusd',  id: 'nzdusd',  dec: 5 },
  { sym: 'eurgbp',  id: 'eurgbp',  dec: 5 },
  { sym: 'eurjpy',  id: 'eurjpy',  dec: 3 },
  { sym: 'eurchf',  id: 'eurchf',  dec: 5 },
  { sym: 'eurcad',  id: 'eurcad',  dec: 5 },
  { sym: 'euraud',  id: 'euraud',  dec: 5 },
  { sym: 'gbpjpy',  id: 'gbpjpy',  dec: 3 },
  { sym: 'gbpchf',  id: 'gbpchf',  dec: 5 },
  { sym: 'gbpcad',  id: 'gbpcad',  dec: 5 },
  { sym: 'audjpy',  id: 'audjpy',  dec: 3 },
  { sym: 'audnzd',  id: 'audnzd',  dec: 5 },
  { sym: 'audchf',  id: 'audchf',  dec: 5 },
  { sym: 'cadjpy',  id: 'cadjpy',  dec: 3 },
  { sym: 'chfjpy',  id: 'chfjpy',  dec: 3 },
  { sym: 'nzdjpy',  id: 'nzdjpy',  dec: 3 },
  { sym: 'eurnzd',  id: 'eurnzd',  dec: 5 },
  { sym: 'gbpaud',  id: 'gbpaud',  dec: 5 },
  { sym: 'gbpnzd',  id: 'gbpnzd',  dec: 5 },
  { sym: 'audcad',  id: 'audcad',  dec: 5 },
  { sym: 'cadchf',  id: 'cadchf',  dec: 5 },
  { sym: 'nzdcad',  id: 'nzdcad',  dec: 5 },
  { sym: 'nzdchf',  id: 'nzdchf',  dec: 5 },
];

// ── Intraday quotes cache (from GitHub Action — Twelve Data + Alpha Vantage) ──
// Loaded once per refresh cycle and shared between fetchRiskData and fetchCrossAssetData.
// Avoids double-fetching the same JSON in the same 2-min cycle.
let _intradayCacheTime  = 0;
let _intradayCache      = null;
let _intradayInFlight   = null;  // promise dedup: prevents concurrent callers from each firing a separate fetch

async function loadIntradayQuotes() {
  const now = Date.now();
  // Re-use cache for up to 90 seconds within the same refresh cycle
  if (_intradayCache && (now - _intradayCacheTime) < 90_000) return _intradayCache;
  // If a fetch is already in flight, wait for it instead of firing a duplicate request
  if (_intradayInFlight) return _intradayInFlight;

  _intradayInFlight = (async () => {
    try {
      const r = await fetch('./intraday-data/quotes.json?_=' + Math.floor(now / 60000), {
        signal: AbortSignal.timeout(5000)
      });
      if (!r.ok) return null;
      const data = await r.json();
      if (!data?.quotes) return null;

      // Validate freshness — warn if file is older than 35 minutes
      if (data.updated) {
        const age = (now - new Date(data.updated).getTime()) / 60000;
        if (age > 35) {
          console.warn(`[Intraday] File is ${age.toFixed(0)}min old — treating as stale`);
          Object.values(data.quotes).forEach(q => q.stale = true);
        }
      }

      _intradayCache     = data;
      _intradayCacheTime = now;
      window._intradayQuotes = data;  // expose for watchlist module
      document.dispatchEvent(new CustomEvent('gi:quotesLoaded'));
      console.log(`[Intraday] Loaded ${Object.keys(data.quotes).length} quotes — source: ${data.source}`);
      return data;
    } catch (e) {
      console.warn('[Intraday] Could not load quotes.json:', e.message);
      return null;
    } finally {
      _intradayInFlight = null;  // release lock so next cycle can fetch fresh data
    }
  })();

  return _intradayInFlight;
}

// Helper: extract a standardised quote object from intraday cache
function intradayQuote(cache, id) {
  if (!cache?.quotes?.[id]) return null;
  const q = cache.quotes[id];
  if (!q.close || isNaN(q.close) || q.close <= 0) return null;
  // chg/pct are only valid when prev_close exists — otherwise null (avoids spurious +0.00% display)
  const hasPrev = q.prev_close != null && q.prev_close > 0;
  return {
    close:        q.close,
    prev_close:   q.prev_close ?? null,
    // open: real intraday open (regularMarketOpen) when available — used for candle body color.
    // Falls back to prev_close so the candle open is at yesterday's close (correct fallback).
    open:         (q.open != null && q.open > 0) ? q.open : (q.prev_close ?? q.close),
    // high/low: Yahoo dayHigh/dayLow — used by _lwBuildTodayBar for non-FX candle wicks.
    // Without these, _lwBuildTodayBar falls back to max(o,c)/min(o,c) producing H==O and L==C
    // (no wicks at all), which was the root cause of flat WTI and DXY today-bars.
    high:         (q.high != null && q.high > 0) ? q.high : null,
    low:          (q.low  != null && q.low  > 0) ? q.low  : null,
    chg:          hasPrev ? (q.chg  ?? null) : null,
    pct:          hasPrev ? (q.pct  ?? null) : null,
    fromIntraday: true,
    stale:        q.stale ?? false,
    market_state: q.market_state ?? null,  // "REGULAR"|"PRE"|"POST"|"CLOSED" — for today-bar guard
    market_time:  q.market_time  ?? null,  // Unix timestamp of last trade — for today-bar guard
  };
}
// ──────────────────────────────────────────────────────────────────────────────

// Cache for intraday RT rates — fed by yfinance JSON, used to update FX table + heatmap
const STOOQ_RT_CACHE = {};  // id → { close, open, chg, pct }
window.STOOQ_RT_CACHE = STOOQ_RT_CACHE;  // expose for fx-websocket.js (const doesn't auto-bind to window)

// proxyUrls / proxyUrlsYahoo removed — all data now comes from
// intraday-data/quotes.json (yfinance via GitHub Action, same-origin).
// No CORS proxies needed.

// fetchStooqQuoteSingle removed — yfinance JSON is sole source

async function fetchQuoteBarRT() {
  // ── STEP 1: intraday quotes.json (yfinance via GitHub Action — primary source) ──
  // Covers all 35 symbols including every FX pair with a real prev_close (real chg/pct).
  // No CORS proxies required — same-origin, always available.
  const intradayData = await loadIntradayQuotes();
  let updatedFromIntraday = 0;

  if (intradayData?.quotes) {
    for (const pair of QB_STOOQ_PAIRS) {
      const q = intradayData.quotes[pair.id];
      if (!q?.close || isNaN(q.close) || q.close <= 0) continue;
      // chg/pct only valid when prev_close exists — null prevents a spurious +0.00% display
      const hasPrev = q.prev_close != null && q.prev_close > 0;
      const data = {
        close: q.close,
        open:  (q.open != null && q.open > 0) ? q.open : (q.prev_close ?? q.close),
        prev_close: q.prev_close ?? null,
        chg:   hasPrev ? (q.chg  ?? null) : null,
        pct:   hasPrev ? (q.pct  ?? null) : null,
        high:  (q.high  != null && q.high  > 0) ? q.high  : null,
        low:   (q.low   != null && q.low   > 0) ? q.low   : null,
        session_high: (q.session_high != null && q.session_high > 0) ? q.session_high : null,
        session_low:  (q.session_low  != null && q.session_low  > 0) ? q.session_low  : null,
        hv30:  (q.hv30  != null) ? q.hv30 : (intradayData.hv30?.[pair.id] ?? null),
        pct1w: (q.pct1w != null) ? q.pct1w : null,
        pct1w_date: q.pct1w_date ?? null,
        fromIntraday: true,
        stale: q.stale ?? false,
      };
      STOOQ_RT_CACHE[pair.id] = data;

      const priceEl = document.getElementById('q-' + pair.id);
      const chgEl   = document.getElementById('qc-' + pair.id);
      if (priceEl) {
        priceEl.textContent = data.close.toFixed(pair.dec);
        priceEl.className   = 'q-price ' + clsDir(data.chg);
      }
      if (chgEl) { chgEl.textContent = pctStr(data.pct); chgEl.className = 'q-chg ' + clsDir(data.chg); }
      updatedFromIntraday++;
    }
  }

  // Stooq fallback removed — yfinance JSON covers all FX pairs

  const totalUpdated = Object.keys(STOOQ_RT_CACHE).length;
  if (totalUpdated > 0) {
    updateFxPairsTableRT();
    _lwUpdateTodayBar();   // push live price to the active LW chart (if open)
    const now = new Date();
    const hh = now.getHours().toString().padStart(2,'0');
    const mm = now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    const srcLabel = 'yfinance';  // sole source
    const qbLabel = document.getElementById('qb-source-label');
    if (qbLabel) qbLabel.textContent = `${srcLabel} · ${hh}:${mm} ${tzAbbr}`;
  }
}

// Update FX pairs table with real-time yfinance prices (from intraday JSON)
function updateFxPairsTableRT() {
  // ── Update FX Pairs — Majors table ──
  const _rtDay2 = new Date().getUTCDay(), _rtH2 = new Date().getUTCHours();
  const _isWeekendRT = _rtDay2 === 6 || (_rtDay2 === 0 && _rtH2 < 21) || (_rtDay2 === 5 && _rtH2 >= 21);

  // Show/hide MARKET CLOSED badge — removed; weekend state communicated via timestamp only

  const tbody = document.getElementById('fx-pairs-tbody');
  if (tbody) {
    const rows = tbody.querySelectorAll('tr');
    rows.forEach(row => {
      const symCell = row.querySelector('td.sym');
      if (!symCell) return;
      const symText = symCell.textContent.trim();
      const pairId = symText.replace('/', '').toLowerCase();
      const data = STOOQ_RT_CACHE[pairId];
      if (!data) return;
      const pairCfg = PAIRS.find(p => p.id === pairId);
      if (!pairCfg) return;
      const tds = row.querySelectorAll('td');
      if (tds.length < 6) return;
      const pipVal = pairCfg.dec === 3 ? 0.01 : 0.0001;
      const spreadPips = TYPICAL_SPREADS[pairId] || 0.5;
      const halfSpread = spreadPips * pipVal / 2;
      tds[1].textContent = fmt(data.close - halfSpread, pairCfg.dec);
      tds[2].textContent = fmt(data.close + halfSpread, pairCfg.dec);
      // Spread: keep in sync with TYPICAL_SPREADS (may have been updated by fetchReferenceSpreads)
      if (tds[3]) tds[3].textContent = spreadPips.toFixed(1);
      // 1D Chg: respetar null — mostrar '—' en vez de '+0.00%' cuando prev_close no existe
      if (data.pct != null) {
        tds[4].textContent = pctStr(data.pct);
        tds[4].className   = clsDir(data.pct);
      } else {
        tds[4].textContent = '—';
        tds[4].className   = 'flat';
      }
      // 1W Chg (tds[5]) — from pct1w in cache (prior-Friday-close convention).
      // This column was previously only set in populateFxPairsTable() (initial render).
      // Without updating it here, Finnhub ticks that call updateFxPairsTableRT()
      // never refresh tds[5], so the 1W column stays stale until the next full
      // page render. Fix: mirror the same pct1w logic as populateFxPairsTable().
      if (tds[5]) {
        if (data.pct1w != null) {
          tds[5].textContent = pctStr(data.pct1w);
          tds[5].className   = clsDir(data.pct1w);
        } else {
          tds[5].textContent = '—';
          tds[5].className   = 'flat';
        }
      }
      // HV30: update if data is available in cache (column index 6)
      if (tds[6] && data.hv30 != null) {
        tds[6].textContent = data.hv30.toFixed(1) + '%';
      }
      // Fwd 1M (tds[7]) and Fwd 3M (tds[8]) — populated by renderCIPForwards()
      // RR 1M (tds[9]) — populated by renderRRSurface() from rr-data/rr.json
      // SESS H / SESS L — now at tds[10]/tds[11] due to 3 new columns
      // Use session_high/session_low (21:00 UTC FX session boundary, same as fetch_ohlc.py)
      // instead of high/low (UTC midnight cutoff, which misses the Tokyo/Sydney open hours
      // 21:00–23:59 UTC of the prior calendar day). Falls back to high/low if session
      // values are null (e.g. on weekend or if fetch_fx_session_hl() failed).
      const sessColor = _isWeekendRT ? 'var(--text3)' : 'var(--text1)';
      const _sessH = data.session_high ?? data.high;
      const _sessL = data.session_low  ?? data.low;
      if (tds[10]) { tds[10].textContent = (_sessH != null) ? fmt(_sessH, pairCfg.dec) : '—'; tds[10].style.color = sessColor; }
      if (tds[11]) { tds[11].textContent = (_sessL != null) ? fmt(_sessL, pairCfg.dec) : '—'; tds[11].style.color = sessColor; }
    });
  }

  // ── Update Crosses sidebar from the same RT cache ──
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf'];
  crossIds.forEach(id => {
    const data = STOOQ_RT_CACHE[id];
    if (!data) return;
    const pairCfg = PAIRS.find(p => p.id === id);
    if (!pairCfg) return;
    const priceEl = document.getElementById('sb-' + id);
    const chgEl   = document.getElementById('sbc-' + id);
    if (priceEl) {
      priceEl.textContent = fmt(data.close, pairCfg.dec);
      priceEl.className = 'sb-price ' + clsDir(data.pct);
    }
    if (chgEl) {
      chgEl.textContent = pctStr(data.pct);
      chgEl.className = 'sb-chg ' + clsDir(data.pct);
    }
  });

  // ── Update Cross-Asset gold/wti cells if commodity cache is available ──
  function setCA_rt(caId, data) {
    if (!data) return;
    const vEl = document.getElementById('ca-' + caId);
    const cEl = document.getElementById('cac-' + caId);
    if (!vEl || !cEl) return;
    const cls   = data.pct > 0.05 ? 'up' : data.pct < -0.05 ? 'down' : '';
    const arrow = data.pct > 0.05 ? '▲' : data.pct < -0.05 ? '▼' : '→';
    const sign  = data.pct >= 0 ? '+' : '';
    vEl.textContent = data.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    vEl.className = 'ca-val';
    if (data.chg != null) {
      const absSign = data.chg >= 0 ? '+' : '';
      const absFmt  = Math.abs(data.chg) >= 10 ? (absSign + data.chg.toFixed(1)) : (absSign + data.chg.toFixed(2));
      cEl.textContent = arrow + ' ' + absFmt + ' (' + sign + data.pct.toFixed(2) + '%)';
    } else {
      cEl.textContent = arrow + ' ' + sign + data.pct.toFixed(2) + '%';
    }
    cEl.className = 'ca-chg ' + cls;
  }
  setCA_rt('gold', STOOQ_RT_CACHE['xauusd']);
  setCA_rt('wti',  STOOQ_RT_CACHE['wti']);

  // ── Refresh heatmap with latest RT data (throttled — Finnhub ~2-5 ticks/s) ──
  populateHeatmapThrottled();

  // ── Timestamp ──
  const upd = document.getElementById('fx-table-updated');
  if (upd) {
    const now = new Date();
    const hh = now.getHours().toString().padStart(2,'0');
    const mm = now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    const _rtDay = now.getUTCDay(), _rtH = now.getUTCHours();
    const _rtWeekend = _rtDay === 6 || (_rtDay === 0 && _rtH < 21) || (_rtDay === 5 && _rtH >= 21);
    const _hasFinnhub = Object.values(STOOQ_RT_CACHE).some(e => e?.fromFinnhub);
    upd.textContent = _rtWeekend
      ? `yfinance · Last close: Fri · ~5min delay`
      : _hasFinnhub
        ? `Finnhub · live`
        : `yfinance · ${hh}:${mm} ${tzAbbr} · ~5min delay`;
  }

  // Update Price Chart panel-sub label to match active source
  const _chartSub = document.querySelector('#section-fxpairs .panel-sub');
  if (_chartSub && _chartSub.textContent !== 'TradingView \u00b7 live data') {
    const _hasFh = Object.values(STOOQ_RT_CACHE).some(e => e?.fromFinnhub);
    _chartSub.textContent = _hasFh ? 'Finnhub \u00b7 live' : `yfinance \u00b7 ~5min delay`;
  }
}

// COMMODITY QUOTES — Gold (XAU) + WTI via free APIs
// ═══════════════════════════════════════════════════════════════════
async function fetchCommodityQuotes() {
  // Gold and WTI come from intraday quotes.json (yfinance GC=F / CL=F).
  // Stooq/Yahoo removed — CORS blocked. Data already loaded in loadIntradayQuotes().
  const intraday = await loadIntradayQuotes();
  if (!intraday) return;

  const gold = intradayQuote(intraday, 'gold');
  const wti  = intradayQuote(intraday, 'wti');

  if (gold) {
    STOOQ_RT_CACHE['xauusd'] = gold;
    const el = document.getElementById('q-xauusd'), ce = document.getElementById('qc-xauusd');
    if (el) { el.textContent = gold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); el.className = 'q-price ' + clsDir(gold.chg); }
    if (ce) { ce.textContent = pctStr(gold.pct); ce.className = 'q-chg ' + clsDir(gold.chg); }
  }
  if (wti) {
    STOOQ_RT_CACHE['wti'] = wti;
    const el = document.getElementById('q-wti'), ce = document.getElementById('qc-wti');
    if (el) { el.textContent = wti.close.toFixed(2); el.className = 'q-price ' + clsDir(wti.chg); }
    if (ce) { ce.textContent = pctStr(wti.pct); ce.className = 'q-chg ' + clsDir(wti.chg); }
  }
  updateFxPairsTableRT();
}

// ═══════════════════════════════════════════════════════════════════
// MARKET SENTIMENT — Dukascopy (free, CORS-allowed)
// ═══════════════════════════════════════════════════════════════════
// COT-derived sentiment cache
const COT_SENTIMENT_CACHE = {};
// Retail sentiment cache — populated by fetchSentiment() from myfxbook.json
// keyed by normalised sym e.g. "EUR/USD" → { longPct, shortPct, longPos, shortPos, avgL, avgS }
const RETAIL_SENTIMENT_CACHE = {};
// Static sentiment fallback (last resort only)
const SENTIMENT_FALLBACK = [
  { sym:'EUR/USD', buy:56, sell:44 }, { sym:'GBP/USD', buy:51, sell:49 },
  { sym:'USD/JPY', buy:35, sell:65 }, { sym:'AUD/USD', buy:46, sell:54 },
  { sym:'USD/CHF', buy:60, sell:40 }, { sym:'USD/CAD', buy:48, sell:52 },
  { sym:'NZD/USD', buy:54, sell:46 }, { sym:'EUR/GBP', buy:44, sell:56 },
  { sym:'EUR/JPY', buy:61, sell:39 }, { sym:'GBP/JPY', buy:57, sell:43 },
];

// Build sentiment from COT positions (net position → bullish/bearish bias)
async function buildCOTSentiment() {
  const COT_CCYS = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
  const results = {};
  await Promise.all(COT_CCYS.map(async ccy => {
    try {
      const r = await fetch('./cot-data/' + ccy + '.json');
      if (!r.ok) return;
      const d = await r.json();
      if (d.longPositions != null && d.shortPositions != null) {
        const total = d.longPositions + d.shortPositions;
        const buyPct = total > 0 ? Math.round(d.longPositions / total * 100) : 50;
        results[ccy] = { buy: buyPct, sell: 100 - buyPct, net: d.netPosition, date: d.reportDate };
      }
    } catch {}
  }));
  return results;
}

function renderSentiment(pairs, sourceLabel, general) {
  const container = document.getElementById('sent-rows');
  if (!container) return;

  // ── Inject tooltip engine once ──
  if (!document.getElementById('fx-tt-style')) {
    const s = document.createElement('style');
    s.id = 'fx-tt-style';
    s.textContent = `
      #fx-tt {
        position:fixed;z-index:99999;
        width:min(240px, calc(100vw - 24px));
        background:var(--bg3);border:1px solid var(--border2);
        border-radius:4px;padding:9px 11px;
        font-size:11px;color:var(--text);line-height:1.55;
        pointer-events:none;display:none;font-family:var(--font-ui);
        box-sizing:border-box;
      }
      #fx-tt .tt-title { font-weight:700;font-size:11px;color:#fff;margin-bottom:3px; }
      #fx-tt .tt-ex { margin-top:5px;padding-top:5px;border-top:1px solid var(--border2);font-size:10px;color:var(--text2);font-style:italic; }
      .fx-tip { border-bottom:1px dashed rgba(255,255,255,0.2);cursor:help; }
    `;
    document.head.appendChild(s);

    const ttEl = document.createElement('div');
    ttEl.id = 'fx-tt';
    ttEl.innerHTML = '<div class="tt-title" id="fx-tt-title"></div><div id="fx-tt-body"></div><div class="tt-ex" id="fx-tt-ex"></div>';
    document.body.appendChild(ttEl);

    window._fxTTPos = function(cx, cy) {
      const tt = document.getElementById('fx-tt');
      if (!tt) return;
      const vw = window.innerWidth, vh = window.innerHeight;
      const ttW = Math.min(240, vw - 24);
      const ttH = tt.offsetHeight || 130;
      const PAD = 8;
      let x = cx + 14, y = cy + 14;
      if (x + ttW > vw - PAD) x = cx - ttW - 8;
      if (x < PAD) x = PAD;
      if (y + ttH > vh - PAD) y = cy - ttH - 8;
      if (y < PAD) y = PAD;
      tt.style.left = x + 'px';
      tt.style.top  = y + 'px';
    };

    document.addEventListener('mousemove', ev => {
      const tt = document.getElementById('fx-tt');
      if (tt && tt.style.display === 'block') window._fxTTPos(ev.clientX, ev.clientY);
    });

    document.addEventListener('touchstart', ev => {
      if (!ev.target.closest('.fx-tip')) {
        const tt = document.getElementById('fx-tt');
        if (tt) tt.style.display = 'none';
      }
    }, { passive: true });
  }

  function attachTip(el, title, body, ex) {
    if (!el) return;
    el.classList.add('fx-tip');

    function _showTip(cx, cy) {
      const tt = document.getElementById('fx-tt');
      document.getElementById('fx-tt-title').textContent = title;
      document.getElementById('fx-tt-body').textContent  = body;
      const exEl = document.getElementById('fx-tt-ex');
      exEl.textContent = ex || ''; exEl.style.display = ex ? 'block' : 'none';
      tt.style.display = 'block';
      requestAnimationFrame(() => window._fxTTPos(cx, cy));
    }

    el.addEventListener('mouseenter', ev => _showTip(ev.clientX, ev.clientY));
    el.addEventListener('mouseleave', () => { document.getElementById('fx-tt').style.display = 'none'; });

    el.addEventListener('touchstart', ev => {
      ev.stopPropagation();
      const t = ev.touches[0];
      _showTip(t.clientX, t.clientY);
    }, { passive: true });
  }

  function fmtK(n) { return n >= 1000 ? (n/1000).toFixed(1) + 'K' : String(n); }

  // Sort by totalPos descending, fallback to conviction
  const sorted = [...pairs].sort((a, b) =>
    (b.totalPos || 0) !== (a.totalPos || 0)
      ? (b.totalPos || 0) - (a.totalPos || 0)
      : Math.max(b.buy, b.sell) - Math.max(a.buy, a.sell)
  );

  // ── Compact table header ──
  container.innerHTML = `
    <div style="display:grid;grid-template-columns:58px 1fr 38px 38px 12px 52px;align-items:center;gap:0;padding:3px 8px 3px;background:var(--head-bg);border-bottom:1px solid var(--border2);position:sticky;top:0;z-index:1;">
      <span style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.05em;font-family:var(--font-ui);">Pair</span>
      <span style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.05em;font-family:var(--font-ui);">Long / Short</span>
      <span style="font-size:9px;color:var(--up);text-transform:uppercase;letter-spacing:.05em;font-family:var(--font-ui);text-align:right;">L%</span>
      <span style="font-size:9px;color:var(--down);text-transform:uppercase;letter-spacing:.05em;font-family:var(--font-ui);text-align:right;">S%</span>
      <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);text-align:center;"> </span>
      <span style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.05em;font-family:var(--font-ui);text-align:right;">Pos</span>
    </div>`;

  sorted.forEach(p => {
    const hasRich = p.totalPos > 0 && p.avgL > 0 && p.avgS > 0;
    const domLong = p.buy >= p.sell;
    const biasCol = domLong ? 'var(--up)' : 'var(--down)';
    const biasLbl = domLong ? 'L' : 'S';

    // ── Price distance + tick ──
    let distPct = null, distPips = null, trapped = false, currentPrice = 0, domAvg = 0, decimals = 4;
    let tickPct = null;

    if (hasRich) {
      domAvg   = domLong ? p.avgL : p.avgS;
      decimals = domAvg > 20 ? 2 : 4;
      const qCache = (typeof intradayQuote === 'function' && _intradayCache)
        ? intradayQuote(_intradayCache, p.sym.replace('/', '').toLowerCase())
        : null;
      currentPrice = qCache ? qCache.close : 0;

      if (currentPrice > 0) {
        trapped   = domLong ? currentPrice < domAvg : currentPrice > domAvg;
        distPct   = (currentPrice - domAvg) / domAvg * 100;
        distPips  = Math.abs(Math.round((currentPrice - domAvg) * (domAvg > 20 ? 100 : 10000)));
        const lo    = Math.min(p.avgL, p.avgS);
        const hi    = Math.max(p.avgL, p.avgS);
        const range = (hi - lo) || domAvg * 0.01;
        tickPct = Math.min(98, Math.max(2,
          (currentPrice - (lo - range * 1.5)) / (range * 4) * 100
        ));
      }
    }

    const distSign = distPct !== null && distPct >= 0 ? '+' : '';
    const tvSym = 'FX_IDC:' + p.sym.replace('/', '');

    // ── Compact single-row layout ──
    const row = document.createElement('div');
    row.style.cssText = 'display:grid;grid-template-columns:58px 1fr 38px 38px 12px 52px;align-items:center;gap:0;padding:3px 8px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s;';
    // No row.title — the native browser tooltip overlaps the custom #fx-tt tooltips on child cells.
    // Screen-reader label via aria-label instead.
    row.setAttribute('aria-label', 'Click to open ' + p.sym + ' chart');
    row.addEventListener('mouseenter', () => row.style.background = 'var(--bg3)');
    row.addEventListener('mouseleave', () => row.style.background = '');
    row.addEventListener('click', () => loadTVChart(tvSym));

    // Col 1: Symbol
    const symDiv = document.createElement('div');
    symDiv.style.cssText = 'display:flex;flex-direction:column;gap:0;';
    const symSpan = document.createElement('span');
    symSpan.style.cssText = 'font-size:10px;font-weight:700;color:#fff;font-family:var(--font-ui);white-space:nowrap;line-height:1.2;';
    symSpan.textContent = p.sym;
    symDiv.appendChild(symSpan);
    // Sub-line: avg entry price + trapped/profit arrow
    if (hasRich) {
      const statusSpan = document.createElement('span');
      const distCol2 = distPct !== null ? (trapped ? 'var(--down)' : 'var(--up)') : 'var(--text3)';
      statusSpan.style.cssText = `font-size:8px;font-family:var(--font-mono);color:${distCol2};white-space:nowrap;line-height:1.2;`;
      const avgStr = domAvg.toFixed(decimals);
      const arrow = distPct !== null ? (trapped ? ' ▼' : ' ▲') : '';
      statusSpan.textContent = avgStr + arrow;
      symDiv.appendChild(statusSpan);
    }

    // Col 2: Bar (compact 6px height)
    const barDiv = document.createElement('div');
    barDiv.style.cssText = 'position:relative;height:6px;background:var(--bg3);border-radius:1px;overflow:visible;margin:0 4px;cursor:help;';
    barDiv.innerHTML = `
      <div style="position:absolute;left:0;top:0;height:100%;width:${p.buy}%;background:var(--up);opacity:.85;border-radius:1px 0 0 1px;"></div>
      <div style="position:absolute;right:0;top:0;height:100%;width:${p.sell}%;background:var(--down);opacity:.85;border-radius:0 1px 1px 0;"></div>`;

    let tickEl = null;
    if (tickPct !== null) {
      tickEl = document.createElement('div');
      tickEl.style.cssText = `position:absolute;top:-3px;width:2px;height:12px;background:#fff;opacity:.9;border-radius:1px;left:${tickPct}%;transform:translateX(-1px);z-index:2;cursor:help;`;
      barDiv.appendChild(tickEl);
    }

    // Col 3: % Long
    const buySpan = document.createElement('span');
    buySpan.style.cssText = 'font-size:10px;color:var(--up);font-family:var(--font-mono);text-align:right;cursor:help;';
    buySpan.textContent = p.buy + '%';

    // Col 4: % Short
    const sellSpan = document.createElement('span');
    sellSpan.style.cssText = 'font-size:10px;color:var(--down);font-family:var(--font-mono);text-align:right;cursor:help;';
    sellSpan.textContent = p.sell + '%';

    // Col 5: Bias dot
    const biasSpan = document.createElement('span');
    biasSpan.style.cssText = `font-size:9px;font-weight:700;color:${biasCol};font-family:var(--font-ui);text-align:center;`;
    biasSpan.textContent = biasLbl;

    // Col 6: Positions
    const posSpan = document.createElement('span');
    posSpan.style.cssText = 'font-size:9px;color:var(--text3);font-family:var(--font-ui);text-align:right;white-space:nowrap;';
    posSpan.textContent = hasRich ? fmtK(p.totalPos) : '—';

    row.append(symDiv, barDiv, buySpan, sellSpan, biasSpan, posSpan);
    container.appendChild(row);

    // ── Tooltips ──
    const domSideTxt = domLong ? 'longs' : 'shorts';
    attachTip(symDiv,
      'Click to open ' + p.sym + ' chart',
      `Opens the ${p.sym} price chart. Retail positioning is most useful when cross-referenced with price action.`,
      null
    );
    attachTip(barDiv,
      'Long / Short bar',
      `Shows the split between retail buyers (green, left) and sellers (red, right). Extreme readings are often contrarian signals.`,
      `A nearly all-red bar means retail is heavily short — historically a bullish contrarian signal.`
    );
    if (tickEl) {
      attachTip(tickEl,
        'Current price (white line)',
        `The white bar shows where price is now relative to the retail average entry for the dominant side.`,
        `Line to the left of center = price fell below where retail longs entered — they are underwater.`
      );
    }
    attachTip(buySpan, '% Long',
      `Percentage of retail traders currently holding long (buy) positions in ${p.sym}.`,
      `Readings above 70% long are unusual and often precede a drop as crowded longs get squeezed.`
    );
    attachTip(sellSpan, '% Short',
      `Percentage of retail traders currently holding short (sell) positions in ${p.sym}.`,
      `Readings above 70% short are unusual and often precede a rally as crowded shorts get squeezed.`
    );
    if (hasRich) {
      attachTip(posSpan,
        'Open positions',
        `Number of Myfxbook traders with ${p.sym} open right now. Higher count = more statistically representative.`,
        `EUR/USD with 54K positions is the most-followed pair — mass stop-outs here move the market.`
      );
    }
    if (distPct !== null) {
      const distEl = symDiv.lastChild;
      const distTitle2 = trapped ? 'Retail trapped' : 'Retail in profit';
      const distBody2  = `The dominant side (${domSideTxt}) entered at avg ${domAvg.toFixed(decimals)}. Current price: ${currentPrice.toFixed(decimals)}. They are ${trapped ? 'underwater (losing)' : 'in profit'}.`;
      const distEx2    = trapped
        ? 'If price continues against them, mass stop-outs can trigger a sharp move.'
        : 'They may take profits soon, creating pressure in the opposite direction.';
      attachTip(distEl, distTitle2, distBody2, distEx2);
    }
  });

  // ── General stats footer ──
  const genEl = document.getElementById('sent-general');
  if (genEl && general) {
    const profPct   = general.profitablePercentage   || 0;
    const realPct   = general.realAccountsPercentage || 0;
    const funds     = general.totalFunds             || '';
    const avgDep    = general.averageDeposit         || '';
    const avgProfit = general.averageAccountProfit   || '';
    const avgLoss   = general.averageAccountLoss     || '';

    // No extra background — inherits var(--bg2) from myfxbook-wrap
    genEl.style.cssText = 'padding:5px 0 2px;border-top:1px solid var(--border);flex-shrink:0;';
    genEl.innerHTML = '';

    // Profitable row with mini bar
    const profRow = document.createElement('div');
    profRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:help;';
    profRow.innerHTML = `
      <span style="font-size:9px;color:var(--text2);font-family:var(--font-ui);white-space:nowrap;">Profitable</span>
      <div style="flex:1;height:3px;background:var(--bg3);border-radius:1px;">
        <div style="height:3px;width:${profPct}%;background:var(--up);border-radius:1px;"></div>
      </div>
      <span style="font-size:9px;color:var(--up);font-family:var(--font-mono);">${profPct}%</span>
      <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);white-space:nowrap;">Real ${realPct}%</span>
    `;
    genEl.appendChild(profRow);

    // Stats row
    const statsRow = document.createElement('div');
    statsRow.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;cursor:help;';
    statsRow.innerHTML = `
      <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">Funds <span style="color:var(--text2);">$${funds}</span></span>
      <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">Avg dep <span style="color:var(--text2);">$${avgDep}</span></span>
      <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">P&amp;L <span style="color:var(--up);">+$${avgProfit}</span> <span style="color:var(--down);">${avgLoss}</span></span>
    `;
    genEl.appendChild(statsRow);

    // Tooltips on footer items
    attachTip(profRow,
      'Profitable accounts',
      'Percentage of Myfxbook accounts currently showing a positive balance. Above 60% is common in trending markets.',
      'Falls sharply during high-volatility periods — a rising profitable % can signal market stabilization.'
    );
    // Individual stat tooltips
    const statSpans = statsRow.querySelectorAll('span');
    attachTip(statSpans[0],
      'Total funds',
      'Sum of capital in all sampled Myfxbook accounts. Larger sample = more statistical weight.',
      'More total funds means the sentiment data better reflects real institutional-retail behavior.'
    );
    attachTip(statSpans[1],
      'Average deposit',
      'Average account size in the sample. Higher values indicate more experienced or semi-professional traders.',
      '$96K average suggests the sample skews toward serious traders, not micro accounts — data carries more weight.'
    );
    attachTip(statSpans[2],
      'Community P&L',
      'Average profit of winning accounts vs average loss of losing accounts.',
      'If avg loss exceeds avg profit, retail is in capitulation mode — often a contrarian signal for reversals.'
    );
    // Real accounts tooltip on profRow's last span
    const realSpan = profRow.querySelector('span:last-child');
    attachTip(realSpan,
      'Real accounts %',
      'Share of accounts using real money (vs demo). Higher % = more meaningful signal.',
      'Above 50% real accounts means the data reflects actual capital at risk, not practice accounts.'
    );
  }

  // ── Timestamp & source label ──
  const now = new Date();
  const lh = now.getHours().toString().padStart(2,'0');
  const lm = now.getMinutes().toString().padStart(2,'0');
  const tzAbbr2 = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('sent-updated', (sourceLabel || '') + ' · ' + lh + ':' + lm + ' ' + tzAbbr2);

  const isCOT = sourceLabel && sourceLabel.includes('COT');
  const isHistorical = sourceLabel && sourceLabel.includes('Historical');
  const isDukascopy = sourceLabel && sourceLabel.includes('Dukascopy');
  const subEl = document.getElementById('sent-source-sub');
  if (subEl) {
    const _sentTime = lh + ':' + lm + ' ' + tzAbbr2;
    if (isCOT) subEl.textContent = `CFTC COT · speculative positioning · loaded ${_sentTime}`;
    else if (isHistorical) subEl.textContent = `Static fallback · live feeds unavailable · loaded ${_sentTime}`;
    else if (isDukascopy) subEl.textContent = `Dukascopy · retail positioning · updated ${_sentTime}`;
    else subEl.textContent = `Myfxbook · retail positioning · updated ${_sentTime}`;
  }
}

async function fetchSentiment() {
  // Pre-load intraday quotes so renderSentiment can access _intradayCache for price distances
  await loadIntradayQuotes().catch(() => null);

  // ── SOURCE 1: Myfxbook community outlook (primary — updated every hour via GitHub Action) ──
  // Skipped automatically when apiBlocked=true (GitHub Actions IPs blocked by provider).
  // In that case the dashboard promotes Dukascopy to SOURCE 2 for real-time retail sentiment.
  try {
    const r = await fetch('./sentiment-data/myfxbook.json');
    if (r.ok) {
      const d = await r.json();
      // If Myfxbook API is blocking GitHub Actions IPs, skip to Dukascopy immediately.
      if (d.apiBlocked) throw new Error('apiBlocked');
      // Freshness check: reject if data is older than 15 hours (covers overnight/weekend gaps between workflow runs)
      const updatedMs = d.updated ? new Date(d.updated).getTime() : 0;
      const ageMin = (Date.now() - updatedMs) / 60000;
      if (d.pairs && d.pairs.length >= 5 && ageMin < 900) {
        const pairs = d.pairs.map(p => ({
          sym:      p.sym,
          buy:      p.long,
          sell:     p.short,
          totalPos: p.totalPos  || 0,
          longPos:  p.longPos   || 0,
          shortPos: p.shortPos  || 0,
          avgL:     p.avgLongPx || 0,
          avgS:     p.avgShortPx|| 0,
        }));
        const ageLabel = ageMin < 60
          ? Math.round(ageMin) + 'min ago'
          : Math.round(ageMin / 60) + 'h ago';
        const general = d.general || null;
        // Populate RETAIL_SENTIMENT_CACHE for use in pair detail popover
        pairs.forEach(p => {
          const key = (p.sym || '').toUpperCase().replace(/\./g, '/');
          RETAIL_SENTIMENT_CACHE[key] = {
            longPct:  p.buy  ?? null,
            shortPct: p.sell ?? null,
            longPos:  p.longPos  || 0,
            shortPos: p.shortPos || 0,
            avgL: p.avgL || 0,
            avgS: p.avgS || 0,
          };
        });
        renderSentiment(pairs, 'Myfxbook · ' + ageLabel, general);
        return;
      }
    }
  } catch {}

  // ── SOURCE 2: Dukascopy live sentiment (CORS-allowed, real-time) ──
  // Promoted above COT: Dukascopy provides real-time retail positioning,
  // which is semantically equivalent to Myfxbook. COT (weekly, speculative)
  // is a weaker substitute for retail sentiment and is kept as last resort.
  try {
    const r = await fetch('https://freeserv.dukascopy.com/2.0/api?path=sentiment/list&prettyprint=true&jsonp=false', {mode:'cors'});
    if (r.ok) {
      const data = await r.json();
      if (data && data.data && data.data.length) {
        const mapped = data.data.slice(0,10).map(d => ({
          sym:  (d.instrument||d.sym||'').replace('_','/'),
          buy:  Math.round(d.longVolume || d.buy || 50),
          sell: Math.round(d.shortVolume || d.sell || 50),
        })).filter(d=>d.sym);
        if (mapped.length) { renderSentiment(mapped, 'Dukascopy live'); return; }
      }
    }
  } catch {}

  // ── SOURCE 3: Static reference fallback ──
  // COT data is intentionally excluded from this fallback pipeline:
  // it belongs to its own dedicated section in the terminal and has
  // different semantics (speculative positioning, weekly) vs retail sentiment.
  renderSentiment(SENTIMENT_FALLBACK, 'Static fallback · live feeds unavailable');
}

// ═══════════════════════════════════════════════════════════════════
// RISK MONITOR + YIELD DATA — multiple free sources with fallback
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// RISK MONITOR TOOLTIPS
// Uses the same #fx-tt engine as renderSentiment.
// attachRiskTip() is a standalone wrapper that works even if
// renderSentiment hasn't run yet (it bootstraps the engine itself).
// ═══════════════════════════════════════════════════════════════════
function attachRiskTip(el, title, body, ex) {
  if (!el) return;

  // Bootstrap tooltip DOM once (shared with fx sentiment engine)
  if (!document.getElementById('fx-tt-style')) {
    const s = document.createElement('style');
    s.id = 'fx-tt-style';
    s.textContent = `
      #fx-tt {
        position:fixed;z-index:99999;
        width:min(240px, calc(100vw - 24px));
        background:var(--bg3);border:1px solid var(--border2);
        border-radius:4px;padding:9px 11px;
        font-size:11px;color:var(--text);line-height:1.55;
        pointer-events:none;display:none;font-family:var(--font-ui);
        box-sizing:border-box;
      }
      #fx-tt .tt-title { font-weight:700;font-size:11px;color:#fff;margin-bottom:3px; }
      #fx-tt .tt-ex { margin-top:5px;padding-top:5px;border-top:1px solid var(--border2);font-size:10px;color:var(--text2);font-style:italic; }
      .fx-tip { border-bottom:1px dashed rgba(255,255,255,0.2);cursor:help; }
    `;
    document.head.appendChild(s);
    const ttEl = document.createElement('div');
    ttEl.id = 'fx-tt';
    ttEl.innerHTML = '<div class="tt-title" id="fx-tt-title"></div><div id="fx-tt-body"></div><div class="tt-ex" id="fx-tt-ex"></div>';
    document.body.appendChild(ttEl);
    window._fxTTPos = function(cx, cy) {
      const tt = document.getElementById('fx-tt');
      if (!tt) return;
      const vw = window.innerWidth, vh = window.innerHeight;
      const ttW = Math.min(240, vw - 24);
      const ttH = tt.offsetHeight || 130;
      const PAD = 8;
      let x = cx + 14, y = cy + 14;
      if (x + ttW > vw - PAD) x = cx - ttW - 8;
      if (x < PAD) x = PAD;
      if (y + ttH > vh - PAD) y = cy - ttH - 8;
      if (y < PAD) y = PAD;
      tt.style.left = x + 'px'; tt.style.top = y + 'px';
    };
    document.addEventListener('mousemove', ev => {
      const tt = document.getElementById('fx-tt');
      if (tt && tt.style.display === 'block') window._fxTTPos(ev.clientX, ev.clientY);
    });
    document.addEventListener('touchstart', ev => {
      if (!ev.target.closest('.fx-tip')) {
        const tt = document.getElementById('fx-tt');
        if (tt) tt.style.display = 'none';
      }
    }, { passive: true });
  }

  function _showTip(cx, cy) {
    const tt = document.getElementById('fx-tt');
    document.getElementById('fx-tt-title').textContent = title;
    document.getElementById('fx-tt-body').textContent  = body;
    const exEl = document.getElementById('fx-tt-ex');
    exEl.textContent = ex || ''; exEl.style.display = ex ? 'block' : 'none';
    tt.style.display = 'block';
    requestAnimationFrame(() => window._fxTTPos(cx, cy));
  }

  el.classList.add('fx-tip');
  el.addEventListener('mouseenter', ev => _showTip(ev.clientX, ev.clientY));
  el.addEventListener('mouseleave', () => { document.getElementById('fx-tt').style.display = 'none'; });
  el.addEventListener('touchstart', ev => {
    ev.stopPropagation();
    const t = ev.touches[0];
    _showTip(t.clientX, t.clientY);
  }, { passive: true });
}

function attachRiskMonitorTooltips() {
  // ── VIX ──────────────────────────────────────────────────────────
  const vixCell = document.querySelector('#section-risk .risk-cell:nth-child(1)');
  if (vixCell) attachRiskTip(vixCell,
    'VIX — CBOE Volatility Index',
    'Measures expected 30-day volatility of the S&P 500 derived from options prices. Known as the "fear gauge." Above 30 = high stress. Below 18 = complacency.',
    'A VIX spike above 25 mid-session signals institutional hedging activity — often precedes sharp moves in risk assets and FX.'
  );

  // ── MOVE Index ───────────────────────────────────────────────────
  const moveCell = document.querySelector('#section-risk .risk-cell:nth-child(2)');
  if (moveCell) attachRiskTip(moveCell,
    'MOVE Index — ICE BofA',
    'Bond market equivalent of VIX. Measures expected 30-day volatility across US Treasuries (1M, 3M, 6M, 1Y options). Elevated MOVE = bond market uncertainty.',
    'MOVE > 120 signals bond stress that typically spills into FX. USD pairs become erratic when MOVE is elevated because rate expectations are unstable.'
  );

  // ── EUR/USD HV 30d ───────────────────────────────────────────────
  const hvCell = document.querySelector('#section-risk .risk-cell:nth-child(3)');
  if (hvCell) attachRiskTip(hvCell,
    'EUR/USD Historical Volatility (30d)',
    'Realized volatility of EUR/USD over the past 30 days, calculated from daily log returns. Not a forecast — it measures what actually happened. Useful for sizing positions.',
    'If HV 30d is 8% and your stop is 100 pips on EUR/USD (≈0.86%), that stop is ~1σ for the current regime. Below 7% = quiet market, above 12% = trending/stressed.'
  );

  // ── Regime ───────────────────────────────────────────────────────
  const regCell = document.querySelector('#section-risk .risk-cell:nth-child(4)');
  if (regCell) attachRiskTip(regCell,
    'Market Regime',
    'Composite live assessment: VIX level (primary driver), yield curve shape, gold intraday demand (>2% = stress signal), S&P 500 daily move (< -1.5% = stress), MOVE index (>100 = elevated per BofA/ICE), AUD/JPY intraday move (the canonical cross-asset risk barometer — sharp selloff >-1.5% = risk-off signal), and USD/JPY (yen safe-haven bid). Updates in real time.',
    'RISK-ON: VIX <18, no stress signals active. MIXED: 1 stress factor (e.g. VIX 18–25). CAUTION: 2–3 factors. RISK-OFF: 4+ factors — high stress, USD/JPY/CHF bid, equities sold. Note: AUD/USD and NZD/USD falling modestly in isolation is normal when CBs diverge (RBA/RBNZ cuts) — AUD/JPY captures risk sentiment more cleanly.'
  );

  // ── Risk Indicators table rows ───────────────────────────────────
  const riRows = document.querySelectorAll('#risk-indicators-tbody tr');
  const riTips = [
    {
      title: 'US–EU Spread 10Y',
      body:  'Difference between US 10-year Treasury yield and German 10-year Bund yield (in basis points). Measures relative monetary policy divergence between the Fed and ECB.',
      ex:    'Spread > +100bp historically supports USD. Narrowing spread (ECB hiking or Fed cutting) tends to push EUR/USD higher.'
    },
    {
      title: 'Gold / SPX Ratio',
      body:  'Price of gold divided by the S&P 500 level. Rising ratio = investors moving from risk assets to safe havens. Falling ratio = risk appetite dominant.',
      ex:    'Ratio > 0.8 and rising historically aligns with USD strength (safe-haven flows), JPY appreciation, and commodity currency weakness.'
    },
    {
      title: 'USD/JPY vs VIX — 60d Correlation',
      body:  'Rolling 60-day Pearson correlation between USD/JPY and VIX, computed from real price data. Normally negative (−0.3 to −0.7): when VIX spikes (risk-off), JPY is bid and USD/JPY falls. A positive reading is unusual.',
      ex:    'Positive correlation means USD and volatility are rising together — typically a USD funding stress episode (2020, 2008). Neutral (near 0) means the relationship has broken down temporarily.'
    },
    {
      title: 'DXY vs SPX — 60d Correlation',
      body:  'Rolling 60-day Pearson correlation between the Dollar Index and S&P 500. The normal relationship is negative: risk-on rallies tend to weaken USD, risk-off bids USD as safe haven.',
      ex:    'Positive reading (both rising together) = USD funding stress or stagflation regime. Sustained positive correlation above +0.3 has preceded episodes of EM FX stress and sharp USD squeezes.'
    },
    {
      title: 'Gold vs DXY — 60d Correlation',
      body:  'Rolling 60-day Pearson correlation between Gold and the Dollar Index. The normal relationship is negative: Gold is priced in USD, so a stronger dollar typically suppresses gold prices.',
      ex:    'Persistent positive correlation means gold is rallying despite USD strength — a signal of real inflation demand, central bank buying, or deep safe-haven flows that override the USD pricing mechanism.'
    },
  ];
  riRows.forEach((row, i) => {
    if (riTips[i]) attachRiskTip(row, riTips[i].title, riTips[i].body, riTips[i].ex);
  });

  // ── Yield Spreads table rows ─────────────────────────────────────
  const ysRows = document.querySelectorAll('#yield-spreads-tbody tr');
  const ysTips = [
    {
      title: '2Y–10Y Spread (US)',
      body:  'Difference between US 10-year and 2-year Treasury yields. Positive = normal curve (growth expected). Negative = inverted curve (recession signal).',
      ex:    'Inversion sustained > 3 months has preceded every US recession since 1980. When disinversion begins (curve steepening), USD typically weakens as Fed cut bets increase.'
    },
    {
      title: 'US–DE 10Y Spread',
      body:  'Difference between US and German 10-year yields. Reflects Fed vs ECB policy divergence. Wide positive spread = USD yield advantage, typically bearish for EUR/USD.',
      ex:    'Spread above +150bp historically coincides with EUR/USD below 1.05. Compression below +100bp tends to support EUR/USD recovery.'
    },
    {
      title: 'US–JP 10Y Spread',
      body:  'Difference between US and Japanese 10-year yields. Wide spread = USD yield advantage over JPY. Drives carry trade flows into USD/JPY.',
      ex:    'Spread > +350bp = strong carry incentive to be long USD/JPY. BoJ YCC adjustments that lift JGB yields compress this spread rapidly, causing sharp JPY strength.'
    },
  ];
  ysRows.forEach((row, i) => {
    if (ysTips[i]) attachRiskTip(row, ysTips[i].title, ysTips[i].body, ysTips[i].ex);
  });

  // ── Option Skew table ─────────────────────────────────────────────
  // Header row
  const skewHead = document.querySelector('table[aria-label="COT-derived directional positioning bias per pair"] thead tr');
  if (skewHead) attachRiskTip(skewHead,
    'Positioning Bias — ETF IV + COT + 25d RR',
    'ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) — nearest expiry ≥4 days. ETF IV is the closest free proxy for OTC interbank implied vol (not publicly available). COT bias from CFTC Disaggregated TFF · Leveraged Funds · Options+Futures Combined. 25-delta Risk Reversal from Saxo Bank public options page (1M tenor, indicative mid) — positive = calls bid over puts (upside skew on base currency); negative = puts bid (downside protection dominant).',
    'ETF options are less liquid than OTC interbank FX options — ATM IV may diverge 1–5 vol points from true OTC levels. RR from Saxo is indicative mid-market, updated during European hours; treat as directional context, not a tradeable quote. Direction signal always comes from Leveraged Funds net positioning (most reactive speculative category in CFTC data).'
  );
  // skew-tbody may be absent (Positioning Bias panel removed) — safe to skip
  const skewRows = document.querySelectorAll('#skew-tbody tr');
  skewRows.forEach(row => {
    // Attach tooltip to each <td> individually — tooltip changes per cell hovered
    row.querySelectorAll('td').forEach(td => {
      const title = td.dataset.tipTitle || '';
      const body  = td.dataset.tipBody  || '';
      const ex    = td.dataset.tipEx    || '';
      if (!title && !body) return;
      attachRiskTip(td, title, body, ex);
    });

    // Attach tooltip to the RR chip <div> inside the bias cell — uses its own tip data
    const rrChip = row.querySelector('[data-rr-tip-title]');
    if (rrChip) {
      const rrTitle = rrChip.dataset.rrTipTitle || '';
      const rrBody  = rrChip.dataset.rrTipBody  || '';
      if (rrTitle || rrBody) attachRiskTip(rrChip, rrTitle, rrBody, '');
    }
  });
}

async function fetchRiskData() {
  // ── STEP 1: Load repo extended-data first (same-origin, instant, no CORS) ──
  // These files are updated daily by the engine. Populating byId here avoids
  // triggering any external API call for data we already have fresh.
  const byId = {};
  try {
    const [usdExt, eurExt, jpyExt] = await Promise.all([
      fetch('./extended-data/USD.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('./extended-data/EUR.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('./extended-data/JPY.json').then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
    if (usdExt?.data) {
      const d = usdExt.data;
      const repo = (v) => ({ close: v, open: v, chg: 0, pct: 0, fromRepo: true });
      if (d.vix    != null && d.vix > 5 && d.vix < 100)            byId.vix   = repo(d.vix);
      if (d.bond10y != null && !isNaN(d.bond10y))                   byId.us10y = repo(d.bond10y);
      if (d.bond2y  != null && !isNaN(d.bond2y)  && d.bond2y > 0)  byId.us2y  = repo(d.bond2y);
      if (d.bond5y  != null && !isNaN(d.bond5y)  && d.bond5y > 0)  byId.us5y  = repo(d.bond5y);
    }
    if (eurExt?.data?.bond10y != null) byId.de10y = { close: eurExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
    if (jpyExt?.data?.bond10y != null) byId.jp10y = { close: jpyExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
  } catch {}

  // ── STEP 1.5: Load intraday quotes JSON (GitHub Action — yfinance) ──
  // Same-origin fetch — instant if boot() already pre-loaded it (90s cache).
  // Enriches byId with fresh intraday data BEFORE the first render.
  const _intradayData = await loadIntradayQuotes();
  if (_intradayData) {
    const _iq = (id) => intradayQuote(_intradayData, id);
    const _set = (id, guard) => { const q = _iq(id); if (q && guard(q.close)) byId[id] = q; };
    _set('vix',   v => v > 5 && v < 100);
    _set('us10y', v => v > 0 && v < 20);
    _set('us3m',  v => v > 0 && v < 20);
    _set('us2y',  v => v > 0 && v < 20);
    _set('us5y',  v => v > 0 && v < 20);
    _set('us30y', v => v > 0 && v < 20);
    _set('dxy',   v => v > 50 && v < 130);
    // MOVE — guardado en byId para usarlo en renderRiskData
    _set('move',  v => v > 10 && v < 400);
  }

  // Render inmediato con repo + intraday JSON — el usuario ve valores en <100ms.
  renderRiskData(byId);

  // ── STEP 2: Enrich byId with intraday quotes.json (yfinance — all symbols) ──
  // Stooq and Yahoo removed: both fail with CORS errors in production.
  // quotes.json (same-origin, GitHub Action) covers all needed symbols.
  if (_intradayData) {
    const _enrich2 = (id, guard) => { const q = intradayQuote(_intradayData, id); if (q && guard(q.close)) byId[id] = q; };
    _enrich2('vix',    v => v > 5 && v < 100);
    _enrich2('us10y',  v => v > 0 && v < 20);
    _enrich2('us2y',   v => v > 0 && v < 20);
    _enrich2('us3m',   v => v > 0 && v < 20);
    _enrich2('us5y',   v => v > 0 && v < 20);
    _enrich2('us30y',  v => v > 0 && v < 20);
    _enrich2('dxy',    v => v > 50 && v < 130);
    _enrich2('move',   v => v > 10 && v < 400);
    // FX risk proxies — used by regime scoring (AUD/JPY is the canonical cross-asset risk barometer)
    _enrich2('audjpy', v => v > 50 && v < 150);
    _enrich2('usdjpy', v => v > 80 && v < 200);
  }

  // ── STEP 3: Final render ──
  await renderRiskData(byId);
}

// renderRiskData — called twice: once with repo data (fast), once after intraday JSON enrichment.
async function renderRiskData(byId) {
  // Check if it's a weekend — on weekends Stooq returns last close, so chg will be 0
  const _rd = new Date().getUTCDay(), _rh = new Date().getUTCHours();
  const isWeekend = _rd === 6 || (_rd === 0 && _rh < 21) || (_rd === 5 && _rh >= 21);
  const weekendNote = isWeekend ? ' (last close)' : '';

  // VIX
  if (byId.vix) {
    const vix = byId.vix.close;
    const cls = vix > 30 ? 'risk-val down' : vix > 25 ? 'risk-val down' : vix > 18 ? 'risk-val warning' : 'risk-val up';  // v7.88.0: aligned with stress score >18 threshold
    setEl('risk-vix', vix.toFixed(1), cls);
    // Bloomberg 4-level VIX classification: <18=Low, 18-25=Moderate, 25-30=Elevated, >30=High
    // Aligns with stress scoring thresholds: >18=+1pt, >25=+2pts, >30=+3pts
    const signal = vix > 30 ? 'High' : vix > 25 ? 'Elevated' : vix > 18 ? 'Moderate' : 'Low';
    const chg = byId.vix.chg || 0;
    const arrow = chg > 0 ? '▲' : chg < 0 ? '▼' : '→';
    const chgStr = (chg >= 0 ? ' +' : ' ') + chg.toFixed(1);
    const srcNote = byId.vix.fromRepo ? ' · FRED' : ' · CBOE';
    setEl('risk-vix-sub', arrow + chgStr + ' · ' + signal + srcNote);
    // Seed STOOQ_RT_CACHE so LW chart today-bar works for VIX tab
    STOOQ_RT_CACHE['vix'] = {
      close:        byId.vix.close,
      open:         byId.vix.open  ?? (byId.vix.prev_close ?? byId.vix.close),
      high:         byId.vix.high  ?? byId.vix.close,
      low:          byId.vix.low   ?? byId.vix.close,
      prev_close:   byId.vix.prev_close ?? null,
      chg:          byId.vix.chg  ?? null,
      pct:          byId.vix.pct  ?? null,
      market_state: byId.vix.market_state ?? null,
      market_time:  byId.vix.market_time  ?? null,
    };
    _lwUpdateTodayBar();
  } else {
    setEl('risk-vix', '—', 'risk-val');
    setEl('risk-vix-sub', 'CBOE · unavailable');
  }

  // MOVE — from intraday quotes.json (yfinance ^MOVE). No external fallback.
  const move = (byId.move && byId.move.close > 10) ? byId.move : null;

  // MOVE Index — ^MOVE via yfinance (ICE BofA bond volatility index)
  {
    if (move && move.close > 10) {
      // MOVE thresholds: >100=elevated (BofA/ICE standard), >120=late-stage crisis (per GUIDELINES)
      const cls = move.close > 120 ? 'risk-val down' : move.close > 100 ? 'risk-val warning' : 'risk-val up';
      setEl('risk-move', move.close.toFixed(1), cls);
      const signal = move.close > 120 ? 'High' : move.close > 100 ? 'Elevated' : 'Low';
      const arrow = move.chg > 0 ? '▲' : move.chg < 0 ? '▼' : '→';
      const chgStr = (move.chg >= 0 ? ' +' : ' ') + move.chg.toFixed(1);
      setEl('risk-move-sub', arrow + chgStr + ' · ' + signal + ' · ICE BofA');
      // Seed STOOQ_RT_CACHE so LW chart today-bar works for MOVE tab
      STOOQ_RT_CACHE['move'] = {
        close:        move.close,
        open:         move.open  ?? (move.prev_close ?? move.close),
        high:         move.high  ?? move.close,
        low:          move.low   ?? move.close,
        prev_close:   move.prev_close ?? null,
        chg:          move.chg  ?? null,
        pct:          move.pct  ?? null,
        market_state: move.market_state ?? null,
        market_time:  move.market_time  ?? null,
      };
      _lwUpdateTodayBar();
    } else if (byId.us10y) {
      // Proxy: MOVE ≈ VIX-like measure from 10Y move
      const vixLevel = byId.vix ? byId.vix.close : 20;
      const approx = Math.round(vixLevel * 4.5);  // v7.88.0: raised from 3.8, empirical 2020-2025 avg MOVE/VIX ratio
      const cls = approx > 150 ? 'risk-val down' : approx > 100 ? 'risk-val warning' : 'risk-val up';
      setEl('risk-move', approx.toString(), cls);
      setEl('risk-move-sub', 'Bond vol · estimated');
    } else {
      setEl('risk-move', '—', 'risk-val');
      setEl('risk-move-sub', 'ICE BofA · unavailable');
    }
  }

  // EUR/USD HV 30d — primary source: HV30 computed by fetch_intraday_quotes.py
  // Fallback: proxy VIX × 0.22 (documented empirical relationship)
  {
    const eurusdHV = STOOQ_RT_CACHE['eurusd']?.hv30 ?? null;
    if (eurusdHV != null && eurusdHV > 1 && eurusdHV < 40) {
      const cls = eurusdHV > 10 ? 'risk-val down' : eurusdHV > 7 ? 'risk-val' : 'risk-val up';
      setEl('risk-eurusd-iv', eurusdHV.toFixed(1) + '%', cls);
      const signal = eurusdHV > 10 ? 'Stress elevated' : eurusdHV > 7 ? 'Moderate' : 'Low vol';
      setEl('risk-eurusd-iv-sub', signal + ' · HV 30d');
    } else if (byId.vix) {
      // Empirical proxy: EUR/USD HV ≈ VIX × 0.22
      const estIV = (byId.vix.close * 0.22).toFixed(1);
      const fNum = parseFloat(estIV);
      const cls = fNum > 10 ? 'risk-val down' : fNum > 7 ? 'risk-val' : 'risk-val up';
      setEl('risk-eurusd-iv', estIV + '%', cls);
      const ivSig = fNum > 10 ? 'Stress elevated' : fNum > 7 ? 'Moderate' : 'Low vol';
      const vixSrc = byId.vix.fromRepo ? ' · est via FRED VIX' : ' · est via VIX';
      setEl('risk-eurusd-iv-sub', ivSig + vixSrc);
    } else {
      setEl('risk-eurusd-iv-sub', 'HV 30d · unavailable');
    }
  }

  // Update topbar US10Y + DXY (live quotes) — no longer shown in indicator table
  if (byId.us10y) {
    const y10 = byId.us10y.close, chg = byId.us10y.chg;
    const _usEl = document.getElementById('q-us10y');
    const _uscEl = document.getElementById('qc-us10y');
    if (_usEl) { _usEl.textContent = y10.toFixed(2) + '%'; _usEl.className = 'q-price ' + (chg > 0 ? 'up' : chg < 0 ? 'down' : 'flat'); }
    if (_uscEl) { _uscEl.textContent = byId.us10y.fromRepo ? '—' : pctStr(byId.us10y.pct); _uscEl.className = 'q-chg flat'; }
  }
  if (byId.dxy) {
    const dxy = byId.dxy.close, chg = byId.dxy.chg;
    const clsD = chg > 0.05 ? 'up' : chg < -0.05 ? 'down' : 'flat';
    const dEl = document.getElementById('q-dxy');
    const dcEl = document.getElementById('qc-dxy');
    if (dEl) { dEl.textContent = dxy.toFixed(1); dEl.className = 'q-price ' + clsDir(chg); }
    if (dcEl) { dcEl.textContent = pctStr(byId.dxy.pct); dcEl.className = 'q-chg ' + clsDir(chg); }
  }

  // Yield spreads — 2Y-10Y (prefer us2y, fallback us3m)
  const short2 = byId.us2y || byId.us3m;
  if (byId.us10y && short2) {
    const y10  = byId.us10y.close;
    const y2   = short2.close;
    const spr  = y10 - y2;
    const bp   = (spr * 100).toFixed(0);
    const cls  = spr < 0 ? 'down' : 'up';
    const sign = spr >= 0 ? '+' : '';
    setEl('ys-2-10', sign + bp + 'bp', cls);
    setEl('ys-2-10-sig', spr < 0 ? 'Inverted' : 'Normal', cls);
  }

  // US–DE 10Y spread
  if (byId.de10y && byId.us10y) {
    const spread = byId.us10y.close - byId.de10y.close;
    const bp2 = (spread * 100).toFixed(0);
    const sign2 = spread >= 0 ? '+' : '';
    setEl('ys-usde', sign2 + bp2 + 'bp');
    setEl('ys-usde-sig', spread > 0 ? 'US Premium' : 'DE Premium');
    // Also update Risk Monitor indicator table
    setEl('ri-us-eu', sign2 + bp2 + 'bp');
    setEl('ri-us-eu-sig', spread > 100/100 ? 'USD+' : spread < -50/100 ? 'EUR+' : 'Neutral', spread > 1 ? 'up' : spread < -0.5 ? 'down' : 'flat');
  }

  // US–JP 10Y spread
  if (byId.jp10y && byId.us10y) {
    const spreadJP = byId.us10y.close - byId.jp10y.close;
    const bpJP = (spreadJP * 100).toFixed(0);
    const signJP = spreadJP >= 0 ? '+' : '';
    setEl('ys-usjp', signJP + bpJP + 'bp');
    setEl('ys-usjp-sig', spreadJP > 0 ? 'US Premium' : 'JP Premium');
  }

  // Rate cells — only show data from real (non-approximated) sources
  if (byId.us3m) {
    const v = byId.us3m.close, chg = byId.us3m.chg;
    setEl('rate-3m', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-3m-chg', (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us2y) {
    const v = byId.us2y.close, chg = byId.us2y.chg;
    setEl('rate-2y', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-2y-chg', byId.us2y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us5y) {
    const v = byId.us5y.close, chg = byId.us5y.chg;
    setEl('rate-5y', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-5y-chg', byId.us5y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us10y) {
    const v = byId.us10y.close, chg = byId.us10y.chg;
    setEl('rate-10y', v.toFixed(2) + '%', 'rate-val');
    // fromRepo means we have the value but no intraday change
    setEl('rate-10y-chg', byId.us10y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }

  // Draw yield curve — only real data points, no interpolation
  // Tenors with real data: 3M (us3m), 2Y (us2y), 5Y (us5y), 10Y (us10y), 30Y (us30y)
  const REAL_TENORS = [
    { label:'3M',  key:'us3m'  },
    { label:'2Y',  key:'us2y'  },
    { label:'5Y',  key:'us5y'  },
    { label:'10Y', key:'us10y' },
    { label:'30Y', key:'us30y' },
  ];
  const realPoints = REAL_TENORS
    .map(t => ({ label: t.label, val: byId[t.key]?.close ?? null }))
    .filter(p => p.val !== null);

  // Need at least 2 points to draw the curve
  // Build prior curve from prev_close in byId (comes from quotes.json via intraday JSON)
  const priorPoints = REAL_TENORS
    .map(t => {
      const q = byId[t.key];
      const prev = q?.prev_close ?? null;
      return prev != null ? { label: t.label, val: prev } : null;
    })
    .filter(Boolean);

  // Populate STATIC_YIELDS from prev_close — eliminates stale hardcoded constants.
  // Used only when realPoints < 2 (rare: live fetch failed). STATIC_LABELS order: 3M,2Y,5Y,10Y,30Y
  if (priorPoints.length >= 3 && STATIC_YIELDS === null) {
    const pLookup = {};
    priorPoints.forEach(p => { pLookup[p.label] = p.val; });
    STATIC_YIELDS = STATIC_LABELS.map(l => pLookup[l] ?? null);
  }

  // Expose tenor data globally for the yield curve modal
  window._STATE_ycTenors = REAL_TENORS.map(t => ({
    label:      t.label,
    close:      byId[t.key]?.close ?? null,
    prev_close: byId[t.key]?.prev_close ?? null,
    chg:        byId[t.key]?.chg ?? null,
  })).filter(t => t.close !== null);

  if (realPoints.length >= 2) {
    drawYieldCurveAndCache(realPoints, priorPoints.length >= 2 ? priorPoints : null);
  } else {
    // Not enough live data — draw with runtime-derived static fallback
    drawYieldCurveAndCache(null, null);
  }

  // Regime assessment based on VIX + yield curve + cross-asset context
  if (byId.vix) {
    const vix = byId.vix.close;
    const isInverted = byId.us10y && byId.us3m && (byId.us10y.close < byId.us3m.close);

    // Multi-factor scoring — each bearish signal adds weight
    // RISK-ON requires VIX < 18 AND no other stress signals (more conservative threshold)
    let stressScore = 0;
    if (vix > 30) stressScore += 3;
    else if (vix > 25) stressScore += 2;
    else if (vix > 18) stressScore += 1;
    if (isInverted) stressScore += 1;
    // Gold up strongly (>2%) as safe-haven = stress signal (intraday; >1% too noisy on normal days)
    if (byId.gold && byId.gold.pct > 2.0) stressScore += 1;
    // SPX down (>1.5%) on the day = meaningful risk pressure (>0.5% too sensitive to routine dips)
    if (byId.spx && byId.spx.pct < -1.5) stressScore += 1;
    // MOVE index elevated = bond market stress (>100 = elevated per BofA/ICE; >120 is late-stage crisis)
    if (byId.move && byId.move.close > 100) stressScore += 1;
    // AUD/JPY is the canonical cross-asset risk barometer (used by JPM, Deutsche Bank, Bloomberg).
    // A move >-1.5% intraday signals genuine risk-off rotation (yen demand + AUD selling).
    // Threshold calibrated to avoid false signals from CB divergence (RBA cuts, etc.)
    // which typically produce moves of -0.3% to -0.8% in isolation.
    if (byId.audjpy && byId.audjpy.pct < -1.5) stressScore += 1;
    // USD/JPY falling sharply (>-1%) = yen safe-haven bid = confirms risk-off.
    // Only add if AUD/JPY also weak to avoid double-counting pure USD moves.
    if (byId.usdjpy && byId.usdjpy.pct < -1.0 && byId.audjpy && byId.audjpy.pct < -0.5) stressScore += 1;

    let regime, regimeSub;
    if (stressScore >= 4)      { regime = 'RISK-OFF'; regimeSub = `High stress · VIX ${vix.toFixed(1)}`; }
    else if (stressScore >= 2) { regime = 'CAUTION';  regimeSub = `Elevated volatility · VIX ${vix.toFixed(1)}`; }
    else if (stressScore === 1){ regime = 'MIXED';    regimeSub = `Mixed signals · VIX ${vix.toFixed(1)}`; }
    else                       { regime = 'RISK-ON';  regimeSub = `Risk appetite active · VIX ${vix.toFixed(1)}`; }
    if (isInverted && regime !== 'RISK-OFF') regimeSub += ' · inverted curve';

    // ── Risk Monitor badge ──
    const regEl = document.getElementById('risk-regime');
    if (regEl) {
      regEl.textContent = regime;
      regEl.className = 'risk-val ' + (regime === 'RISK-ON' ? 'up' : regime === 'RISK-OFF' ? 'down' : '');
      regEl.style.opacity = '';
    }
    setEl('risk-regime-sub', regimeSub);

    // ── Narrative badge (above narrative text) ──
    // Rule: always mirrors the live stress score so both badges are consistent.
    // The AI narrative text below the badge retains its qualitative context;
    // the badge itself is a semaphore that must be unambiguous at a glance.
    const narrRegEl = document.getElementById('narrative-regime');
    if (narrRegEl) {
      const isOn  = regime === 'RISK-ON';
      const isOff = regime === 'RISK-OFF';
      narrRegEl.textContent = regime;
      narrRegEl.className = 'narr-regime';
      narrRegEl.style.borderColor = isOn ? 'var(--up)' : isOff ? 'var(--down)' : 'var(--orange)';
      narrRegEl.style.color       = isOn ? 'var(--up)' : isOff ? 'var(--down)' : 'var(--orange)';
      const narrTsLabel = _narrativeGeneratedAt
        ? ` · AI narrative: ${new Date(_narrativeGeneratedAt).toUTCString().slice(17, 22)} UTC`
        : '';
      // Show AI regime mismatch in tooltip when live score differs from the regime
      // the narrative was written under — explains why narrative tone may not match the badge.
      const aiMismatchNote = (_narrativeAiRegime && _narrativeAiRegime !== regime)
        ? ` · Narrative written under ${_narrativeAiRegime} (conditions changed since generation)`
        : '';
      narrRegEl.title = `Live assessment · VIX ${vix.toFixed(1)}${isInverted ? ' · inverted curve' : ''}${narrTsLabel}${aiMismatchNote}`;
    }
  }

  // ── Yield Curve panel timestamp ─────────────────────────────────────
  const yieldSub = document.getElementById('yield-panel-sub');
  if (yieldSub) {
    const now = new Date();
    const hhmm = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    const yieldSrc = (byId.us2y && !byId.us2y.fromRepo) ? 'yfinance ~5min delay' : 'FRED DGS2 · daily batch';
    yieldSub.textContent = 'Nominal yields · ' + yieldSrc + ' · updated ' + hhmm + ' ' + tzAbbr;
  }

  // Gold/SPX ratio — computed in fetchCrossAssetData() after gold & SPX are fetched
  // Note: ri-us-eu and ri-us-eu-sig are written by the yield spreads block above (canonical path).
  // USD/JPY vs VIX — real 60-day rolling Pearson from quotes.json (computed by engine).
  // Replaces the previous hardcoded proxy coefficients (-0.72, -0.41, etc.) which were
  // invented values. Now shows the actual computed correlation or '—' if unavailable.
  // Label updated in index.html from 'USD/JPY vs Nikkei' → 'USD/JPY vs VIX (60d)'.
  // USD/JPY vs VIX correlation — always force a fresh cache read to avoid boot-order race.
  // loadIntradayQuotes() returns the 90s cache if already loaded, so this costs nothing
  // on second render but guarantees the data is available on first paint.
  loadIntradayQuotes().then(_freshData => {
    try {
      const corrs = _freshData?.correlations;
      if (!Array.isArray(corrs)) return;
      const entry = corrs.find(c =>
        (c.a === 'USD/JPY' && c.b === 'VIX') || (c.a === 'VIX' && c.b === 'USD/JPY')
      );
      if (entry?.corr != null) {
        const v = entry.corr;
        const sign = v >= 0 ? '+' : '';
        const corrLabel = sign + v.toFixed(2) + 'r';
        const corrSig = v < -0.3 ? 'Normal (risk-off)' : v > 0.3 ? 'Unusual' : 'Neutral';
        const corrCls = v < -0.3 ? 'up' : v > 0.3 ? 'down' : 'flat';
        setEl('ri-usdjpy-nk', corrLabel);
        setEl('ri-usdjpy-nk-sig', corrSig, corrCls);
      } else {
        setEl('ri-usdjpy-nk', '—');
        setEl('ri-usdjpy-nk-sig', 'No data', 'flat');
      }

      // DXY vs SPX — positive = funding stress (breaks normal negative relationship)
      const dxySpxEntry = corrs.find(c =>
        (c.a === 'DXY' && c.b === 'SPX') || (c.a === 'SPX' && c.b === 'DXY')
      );
      if (dxySpxEntry?.corr != null) {
        const v = dxySpxEntry.corr;
        const sign = v >= 0 ? '+' : '';
        const corrLabel = sign + v.toFixed(2) + 'r';
        // Normal relationship is negative (USD safe haven, equities risk)
        // Positive = stress break. Tooltip via title attr on the row is handled by JS tooltips.
        const corrSig = v > 0.3 ? 'Stress break' : v < -0.3 ? 'Normal' : 'Neutral';
        const corrCls = v > 0.3 ? 'down' : v < -0.3 ? 'up' : 'flat';
        setEl('ri-dxy-spx', corrLabel);
        setEl('ri-dxy-spx-sig', corrSig, corrCls);
      } else {
        setEl('ri-dxy-spx', '—');
        setEl('ri-dxy-spx-sig', 'No data', 'flat');
      }

      // Gold vs DXY — positive = safe-haven model broken or real inflation bid
      const goldDxyEntry = corrs.find(c =>
        (c.a === 'Gold' && c.b === 'DXY') || (c.a === 'DXY' && c.b === 'Gold')
      );
      if (goldDxyEntry?.corr != null) {
        const v = goldDxyEntry.corr;
        const sign = v >= 0 ? '+' : '';
        const corrLabel = sign + v.toFixed(2) + 'r';
        // Normal relationship is negative (gold priced in USD, inverse)
        const corrSig = v > 0.3 ? 'Inflation bid' : v < -0.3 ? 'Normal' : 'Neutral';  // v7.88.0: raised from 0.2 for Bloomberg +-0.3 symmetry
        const corrCls = v > 0.3 ? 'down' : v < -0.3 ? 'up' : 'flat';
        setEl('ri-gold-dxy', corrLabel);
        setEl('ri-gold-dxy-sig', corrSig, corrCls);
      } else {
        setEl('ri-gold-dxy', '—');
        setEl('ri-gold-dxy-sig', 'No data', 'flat');
      }
    } catch {}
  }).catch(() => {});

  // ── Risk Monitor panel timestamp ─────────────────────────────────────
  const riskSub = document.getElementById('risk-panel-sub');
  if (riskSub) {
    const now = new Date();
    const hhmm = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    riskSub.textContent = 'VIX · MOVE · HV30 · yfinance ~5min delay · updated ' + hhmm + ' ' + tzAbbr;
  }

  // ── VaR/CVaR panel ───────────────────────────────────────────────────
  renderVarCvarPanel();
}

// ── VaR/CVaR Panel renderer ───────────────────────────────────────────────────
// Reads var_cvar key from quotes.json (populated by fetch_intraday_quotes.py PASO 8).
// Displays 1d Historical VaR 95% and CVaR 95% per instrument with regime-shift flag
// when rolling 60d VaR is >25% above the 252d baseline.
async function renderVarCvarPanel() {
  const container = document.getElementById('var-cvar-tbody');
  if (!container) return;

  const intra = await loadIntradayQuotes().catch(() => null);
  const vc = intra?.var_cvar;

  if (!vc || !Object.keys(vc).length) {
    container.innerHTML = '<tr style="display:table;width:100%;table-layout:fixed;"><td colspan="5" style="color:var(--text3);padding:6px 8px;font-size:10px;">VaR data not yet available — runs with daily engine update</td></tr>';
    return;
  }

  const ROWS = [
    { id:'eurusd', label:'EUR/USD',  pip: 0.0001, tip:'Most liquid FX pair. VaR reflects daily move risk in EUR terms. Benchmark for G10 vol regime.' },
    { id:'gbpusd', label:'GBP/USD',  pip: 0.0001, tip:'Cable. Higher vol than EUR/USD; sensitive to UK macro and BoE policy divergence vs Fed.' },
    { id:'usdjpy', label:'USD/JPY',  pip: 0.01,   tip:'Key risk-sentiment pair. Yen acts as safe-haven; elevated VaR signals risk-off pressure or BoJ intervention risk.' },
    { id:'audusd', label:'AUD/USD',  pip: 0.0001, tip:'Commodity and China-proxy currency. VaR spikes with commodity sell-offs or CNY stress.' },
    { id:'usdchf', label:'USD/CHF',  pip: 0.0001, tip:'Swiss franc is a safe-haven. Low VaR in calm markets; can gap sharply on SNB intervention or crisis flows.' },
    { id:'usdcad', label:'USD/CAD',  pip: 0.0001, tip:'Petro-currency pair. VaR driven by WTI moves and BoC vs Fed policy divergence.' },
    { id:'nzdusd', label:'NZD/USD',  pip: 0.0001, tip:'Highest-beta G10 pair. Sensitive to risk appetite, dairy prices, and RBNZ rate path.' },
    { id:'gold',   label:'XAU/USD',  pip: 0.1,    tip:'Safe-haven and inflation hedge. Tail events can move 2-3% in a session; ES/VaR ratio often elevated.' },
    { id:'spx',    label:'SPX',      pip: 1,       tip:'S&P 500 index. Cross-asset risk anchor - SPX vol regime drives correlated moves across G10 pairs.' },
    { id:'dxy',    label:'DXY',      pip: 0.01,   tip:'USD index vs G6 basket. VaR here captures broad dollar vol, useful as normalisation benchmark for FX pairs.' },
    { id:'vix',    label:'VIX',      pip: 0.01,   tip:'CBOE Volatility Index. VaR on VIX itself measures how much implied vol can move in a day - a vol-of-vol indicator.' },
  ];

  container.innerHTML = ROWS.map(row => {
    const d = vc[row.id];
    if (!d) return '';

    const var95  = d.var_pct;
    const cvar95 = d.cvar_pct;
    const v60    = d.var60_pct;

    // Regime flag: 60d VaR > 125% of 252d baseline = stress
    const stressed = v60 != null && var95 > 0 && (v60 / var95) > 1.25;
    // CVaR / VaR ratio: tail risk multiplier (healthy ~1.2–1.5; above 2 = fat tails)
    const ratio = (var95 > 0) ? (cvar95 / var95) : null;
    const ratioCls = ratio == null ? '' : ratio > 2 ? 'down' : ratio > 1.5 ? '' : 'up';

    // VaR colour: green < 0.5%, amber 0.5–1%, red > 1%
    const varCls = var95 > 1.0 ? 'down' : var95 > 0.5 ? '' : 'up';
    const stressFlag = stressed
      ? `<span title="60d VaR (${v60?.toFixed(3)}%) elevated vs 252d baseline — regime stress" style="color:var(--amber,#EF9F27);margin-left:3px;font-size:9px;">⚠</span>`
      : '';

    return `<tr style="display:table;width:100%;table-layout:fixed;">
      <td title="${row.tip}" style="font-family:var(--font-mono);font-size:10px;white-space:nowrap;cursor:default;">${row.label}</td>
      <td class="${varCls}" style="font-family:var(--font-mono);font-size:10px;text-align:right;">${var95.toFixed(3)}%${stressFlag}</td>
      <td style="font-family:var(--font-mono);font-size:10px;text-align:right;color:var(--text2);">${cvar95.toFixed(3)}%</td>
      <td class="${ratioCls}" style="font-family:var(--font-mono);font-size:10px;text-align:right;">${ratio != null ? ratio.toFixed(2) + 'x' : '—'}</td>
      <td style="font-family:var(--font-mono);font-size:10px;text-align:right;color:var(--text3);">${d.n}</td>
    </tr>`;
  }).filter(Boolean).join('');
}

// Yield curve labels — fixed set of tenors we display
const STATIC_LABELS = ['3M','2Y','5Y','10Y','30Y'];
// STATIC_YIELDS: populated at runtime from the first successful quotes.json fetch
// (prev_close of each tenor). Falls back to null → drawYieldCurve shows dashes.
// This eliminates the stale hardcoded [4.35, 4.28, 4.32, 4.42, 4.58] constants.
let STATIC_YIELDS = null;
let _lastDrawnYields = null; // {label, val}[] or null
let _lastDrawnPrior  = null; // {label, val}[] from prev_close, or null

function drawYieldCurveAndCache(points, priorPoints) {
  // points can be: {label,val}[] (real data) or null (use static)
  // priorPoints: {label,val}[] from prev_close in quotes.json (optional, overrides PRIOR_MAP)
  _lastDrawnYields = points;
  _lastDrawnPrior  = priorPoints || null;
  drawYieldCurve(points, priorPoints);
}

// ═══════════════════════════════════════════════════════════════════
// YIELD CURVE — canvas drawing, accepts real sparse data points
// ═══════════════════════════════════════════════════════════════════
function drawYieldCurve(points, priorPoints) {
  const canvas = document.getElementById('yield-canvas');
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth - 20, H = 100;
  // Guard: if the panel is hidden (display:none), clientWidth is 0.
  // Setting canvas.width=0 clears it permanently. Abort and let the next
  // rAF pass (triggered by hideDerivatives double-rAF) redraw correctly.
  if (W <= 0) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  // Build display data — real points or runtime-derived fallback
  let labels, vals, isLive;
  if (points && points.length >= 2) {
    labels = points.map(p => p.label);
    vals   = points.map(p => p.val);
    isLive = true;
  } else if (STATIC_YIELDS) {
    // STATIC_YIELDS populated at runtime from prev_close — not a hardcoded constant
    labels = STATIC_LABELS;
    vals   = STATIC_YIELDS;
    isLive = false;
  } else {
    // No data at all — draw nothing meaningful
    labels = STATIC_LABELS;
    vals   = [null, null, null, null, null];
    isLive = false;
  }

  // Prior curve — exclusively from prev_close in quotes.json (priorPoints).
  // No hardcoded PRIOR_MAP: if prev_close is absent the prior line simply isn't drawn.
  let prevVals;
  if (priorPoints && priorPoints.length >= 2) {
    const priorLookup = {};
    priorPoints.forEach(p => { priorLookup[p.label] = p.val; });
    prevVals = labels.map(l => priorLookup[l] ?? null);
  } else {
    prevVals = labels.map(() => null);   // prior line hidden — no stale fallback
  }

  const n = labels.length;
  const validV = vals.filter(v => v != null && !isNaN(v));
  const rawMin = validV.length ? Math.min(...validV, ...prevVals.filter(Boolean)) : 4.10;
  const rawMax = validV.length ? Math.max(...validV, ...prevVals.filter(Boolean)) : 5.40;
  const minY = Math.floor(rawMin * 4) / 4 - 0.15;
  const maxY = Math.ceil(rawMax  * 4) / 4 + 0.15;
  const yRange = maxY - minY || 1;

  const PAD_L=32, PAD_R=8, PAD_T=12, PAD_B=20;
  const cW=W-PAD_L-PAD_R, cH=H-PAD_T-PAD_B;
  const px = i => PAD_L+(i/(n-1))*cW;
  const py = v => PAD_T+(1-(v-minY)/yRange)*cH;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#131722'; ctx.fillRect(0,0,W,H);

  // Grid lines
  const step = yRange <= 0.5 ? 0.1 : yRange <= 1 ? 0.25 : 0.5;
  const gridStart = Math.ceil(minY / step) * step;
  for (let v = gridStart; v <= maxY + 0.001; v = Math.round((v + step) * 1000) / 1000) {
    const y = py(v);
    ctx.strokeStyle='#2a2e39'; ctx.lineWidth=0.5; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(PAD_L,y); ctx.lineTo(W-PAD_R,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#9ba1ae'; ctx.font='bold 8px Courier New'; ctx.textAlign='right';
    ctx.fillText(v.toFixed(2)+'%', PAD_L-3, y+3);
  }

  // Inverted zone — shade between shortest and longest tenor if inverted
  const firstV = vals[0], lastV = vals[n-1];
  if (firstV != null && lastV != null && firstV > lastV) {
    ctx.fillStyle='#ef535012';
    ctx.fillRect(PAD_L, PAD_T, cW, cH);
  }

  // Prior curve
  const priorPts = prevVals.map((v,i) => v != null ? [px(i), py(v)] : null).filter(Boolean);
  if (priorPts.length >= 2) {
    ctx.beginPath(); ctx.strokeStyle='#363c4e'; ctx.lineWidth=1;
    priorPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();
  }

  // Fill under current curve
  const curPts = vals.map((v,i) => v != null ? [px(i), py(v)] : null).filter(Boolean);
  if (curPts.length >= 2) {
    ctx.beginPath();
    curPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.lineTo(curPts[curPts.length-1][0], PAD_T+cH);
    ctx.lineTo(PAD_L, PAD_T+cH);
    ctx.closePath();
    ctx.fillStyle='#4f7fff12'; ctx.fill();

    // Current curve line
    ctx.beginPath(); ctx.strokeStyle='#4f7fff'; ctx.lineWidth=1.8;
    curPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();

    // Dots + value labels at each real point
    vals.forEach((v, i) => {
      if (v == null) return;
      const x = px(i), y = py(v);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2);
      ctx.fillStyle='#4f7fff'; ctx.fill();
    });
  }

  // X-axis labels
  ctx.fillStyle='#9ba1ae'; ctx.font='bold 8.5px Courier New'; ctx.textAlign='center';
  labels.forEach((t,i) => ctx.fillText(t, px(i), H-5));

  // Legend
  ctx.textAlign='left';
  ctx.fillStyle='#4d8ffa'; ctx.fillText('● Current', PAD_L, PAD_T-2);
  ctx.fillStyle='#6b7280'; ctx.fillText('● Prior',   PAD_L+52, PAD_T-2);
  if (!isLive) {
    ctx.fillStyle='#6b7280'; ctx.fillText('(static)', PAD_L+92, PAD_T-2);
  } else {
    // Check inversion
    const spr = (vals[n-1] ?? 0) - (vals[0] ?? 0); // long - short
    if (spr < 0) { ctx.fillStyle='#ef535099'; ctx.fillText('■ Inverted', PAD_L+92, PAD_T-2); }
  }

  // Update yield spread table using real 2Y and 10Y
  const real2y  = vals[labels.indexOf('2Y')];
  const real3m  = vals[labels.indexOf('3M')];
  const real10y = vals[labels.indexOf('10Y')];
  const shortKey = real2y ?? real3m;
  if (shortKey != null && real10y != null) {
    const spread = real10y - shortKey;
    const bp = Math.round(spread * 100);
    const sign = bp >= 0 ? '+' : '';
    setEl('ys-2-10', sign + bp + 'bp', spread < 0 ? 'down' : 'up');
    setEl('ys-2-10-sig', spread < 0 ? 'Inverted' : 'Normal', spread < 0 ? 'down' : 'up');
  }
}

setTimeout(() => drawYieldCurveAndCache(null), 60);
window.addEventListener('resize', () => drawYieldCurve(_lastDrawnYields, _lastDrawnPrior));

// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// loadCOTChart — COT Long+Short overlaid on same scale in TV widget
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// LIGHTWEIGHT CHARTS — replaces TradingView embed widget for all
// symbols that have ohlc-data/{id}.json (yfinance daily OHLC, 2y).
// Symbols without OHLC data fall back to the TradingView widget.
// ═══════════════════════════════════════════════════════════════════

// Map TradingView data-sym values → ohlc-data file IDs
// Full display names for LW chart header (mirrors TradingView legend)
const _OHLC_FULL_NAMES = {
  eurusd:'Euro / U.S. Dollar',   gbpusd:'British Pound / U.S. Dollar',
  usdjpy:'U.S. Dollar / Japanese Yen', audusd:'Australian Dollar / U.S. Dollar',
  usdcad:'U.S. Dollar / Canadian Dollar', usdchf:'U.S. Dollar / Swiss Franc',
  nzdusd:'New Zealand Dollar / U.S. Dollar', eurgbp:'Euro / British Pound',
  eurjpy:'Euro / Japanese Yen', eurchf:'Euro / Swiss Franc',
  eurcad:'Euro / Canadian Dollar', euraud:'Euro / Australian Dollar',
  eurnzd:'Euro / New Zealand Dollar', gbpjpy:'British Pound / Japanese Yen',
  gbpchf:'British Pound / Swiss Franc', gbpcad:'British Pound / Canadian Dollar',
  gbpaud:'British Pound / Australian Dollar', gbpnzd:'British Pound / New Zealand Dollar',
  audjpy:'Australian Dollar / Japanese Yen', audnzd:'Australian Dollar / New Zealand Dollar',
  audchf:'Australian Dollar / Swiss Franc', audcad:'Australian Dollar / Canadian Dollar',
  cadjpy:'Canadian Dollar / Japanese Yen', cadchf:'Canadian Dollar / Swiss Franc',
  nzdjpy:'New Zealand Dollar / Japanese Yen', nzdcad:'New Zealand Dollar / Canadian Dollar',
  nzdchf:'New Zealand Dollar / Swiss Franc', chfjpy:'Swiss Franc / Japanese Yen',
  gold:'Gold Futures', wti:'Crude Oil WTI Futures', btc:'Bitcoin / U.S. Dollar',
  us10y:'US 10Y Treasury Yield', spx:'S&P 500 Index', nasdaq:'Nasdaq Composite',
  nikkei:'Nikkei 225', stoxx:'Euro Stoxx 50', eth:'Ethereum / U.S. Dollar',
  dxy:'U.S. Dollar Index',
  vix:'CBOE Volatility Index',
  silver:'Silver Futures', brent:'Crude Oil Brent Futures',
  dax:'DAX Performance Index', ftse:'FTSE 100 Index',
  hsi:'Hang Seng Index', dji:'Dow Jones Industrial Average',
};

const _TV_TO_OHLC = {
  'FX_IDC:EURUSD': 'eurusd',  'FX_IDC:USDJPY': 'usdjpy',
  'FX_IDC:GBPUSD': 'gbpusd',  'FX_IDC:AUDUSD': 'audusd',
  'FX_IDC:USDCAD': 'usdcad',  'FX_IDC:USDCHF': 'usdchf',
  'FX_IDC:NZDUSD': 'nzdusd',  'FX_IDC:EURGBP': 'eurgbp',
  'FX_IDC:EURJPY': 'eurjpy',  'FX_IDC:EURCHF': 'eurchf',
  'FX_IDC:EURCAD': 'eurcad',  'FX_IDC:EURAUD': 'euraud',
  'FX_IDC:EURNZD': 'eurnzd',  'FX_IDC:GBPJPY': 'gbpjpy',
  'FX_IDC:GBPCHF': 'gbpchf',  'FX_IDC:GBPCAD': 'gbpcad',
  'FX_IDC:GBPAUD': 'gbpaud',  'FX_IDC:GBPNZD': 'gbpnzd',
  'FX_IDC:AUDJPY': 'audjpy',  'FX_IDC:AUDNZD': 'audnzd',
  'FX_IDC:AUDCHF': 'audchf',  'FX_IDC:AUDCAD': 'audcad',
  'FX_IDC:CADJPY': 'cadjpy',  'FX_IDC:CADCHF': 'cadchf',
  'FX_IDC:NZDJPY': 'nzdjpy',  'FX_IDC:NZDCAD': 'nzdcad',
  'FX_IDC:NZDCHF': 'nzdchf',  'FX_IDC:CHFJPY': 'chfjpy',
  // Metals
  'OANDA:XAUUSD':         'gold',
  'CMCMARKETS:GOLDM2026': 'gold',   // legacy alias
  'OANDA:XAGUSD':         'silver',
  // Energy
  'OANDA:WTICOUSD':       'wti',
  'FPMARKETS:WTI':        'wti',    // legacy alias
  'OANDA:BCOUSD':         'brent',
  // Crypto
  'BITSTAMP:BTCUSD':      'btc',
  'COINBASE:BTCUSD':      'btc',
  // Yields
  'FRED:DGS10':           'us10y',
  // Equity indices
  'FOREXCOM:SPXUSD':      'spx',
  'CMCMARKETS:SPX500':    'spx',    // legacy alias
  'FOREXCOM:NSXUSD':      'nasdaq',
  'CFI:US100':            'nasdaq', // legacy alias
  'INDEX:NI225':          'nikkei',
  'OSE:NK2251!':          'nikkei', // legacy alias
  'FOREXCOM:EU50':        'stoxx',
  'GOMARKETS:STOXX50':    'stoxx',  // legacy alias
  'FOREXCOM:DJI':         'dji',
  'FOREXCOM:DEU40':       'dax',
  'FOREXCOM:UK100':       'ftse',
  'FOREXCOM:HKG33':       'hsi',
  // Crypto
  'BITSTAMP:ETHUSD':      'eth',
  'COINBASE:ETHUSD':      'eth',
  // FX Index
  'PEPPERSTONE:USDX':     'dxy',
  // Volatility
  'CBOE:VIX':             'vix',
  'FRED:VIXCLS':          'vix',
  'TVC:MOVE':             'move',
};

// Human-readable labels for the chart source footer
const _OHLC_LABELS = {
  gold: 'GC=F', wti: 'CL=F', btc: 'BTC-USD', us10y: '^TNX',
  spx: '^GSPC', nasdaq: '^NDX', nikkei: '^N225', stoxx: '^STOXX50E',
  eth: 'ETH-USD', dxy: 'DX-Y.NYB', vix: '^VIX', move: '^MOVE',
  silver: 'SI=F', brent: 'BZ=F', dax: '^GDAXI', ftse: '^FTSE', hsi: '^HSI', dji: '^DJI',
};

// Active LW chart instance — destroyed before each new render
let _lwChart = null;
let _lwResizeObs = null;
let _lwCandleSeries = null;   // reference for live today-bar updates

// Chart mode flag — set synchronously at the START of each chart load, before any async work.
// 'lw'  = LW chart is active or being loaded (do NOT reload TV widget on visibility change)
// 'tv'  = TradingView widget is active
// Using a dedicated flag avoids the race where _lwChart===null during the async fetch/render
// window even though the user's intent is clearly to show the LW chart.
let _chartMode = 'lw'; // default: LW chart (FX pairs load first)
let _lwActiveOhlcId = null;   // ohlcId currently displayed
let _lwActiveUpdateHeader = null; // ref to _updateLWHeader of the active chart (for RT header refresh)
let _lwActivePrevCloseMap = null; // ref to _prevCloseMap of the active chart (for today-bar % calc)
let _lwLastJsonBarDate   = null; // ISO date string of the last bar in the loaded OHLC JSON (before strip)

// Ensure the Lightweight Charts library is loaded (lazy, once)
let _lwLibPromise = null;
function _ensureLWLib() {
  if (window.LightweightCharts) return Promise.resolve();
  if (_lwLibPromise) return _lwLibPromise;
  _lwLibPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
    s.onload  = resolve;
    s.onerror = () => { _lwLibPromise = null; reject(new Error('LW lib load failed')); };
    document.head.appendChild(s);
  });
  return _lwLibPromise;
}

// Destroy any active LW chart instance cleanly
function _destroyLWChart() {
  if (_lwResizeObs)  { _lwResizeObs.disconnect(); _lwResizeObs = null; }
  if (_lwChart)      { try { _lwChart.remove(); } catch(_) {} _lwChart = null; }
  _lwCandleSeries = null;
  _lwActiveOhlcId = null;
  _lwActiveUpdateHeader = null;
  _lwActivePrevCloseMap = null;
  _lwLastJsonBarDate   = null;
  _lwPeriodOpen = null;
  _lwPeriodHigh = null;
  _lwPeriodLow  = null;
}

// Compute MA(n) over close prices
function _calcMA(bars, n) {
  return bars.map((b, i) => {
    if (i < n - 1) return null;
    const sum = bars.slice(i - n + 1, i + 1).reduce((a, x) => a + x.close, 0);
    return { time: b.time, value: parseFloat((sum / n).toFixed(6)) };
  }).filter(Boolean);
}

// FX spot IDs — weekend today-bar injection is skipped for these because
// FX is closed Saturday/Sunday and injecting a flat open=close bar creates
// a phantom doji candle after the last real Friday bar.
const _LW_FX_IDS = new Set([
  'eurusd','gbpusd','usdjpy','audusd','usdcad','usdchf','nzdusd',
  'eurgbp','eurjpy','eurchf','eurcad','euraud','eurnzd','gbpjpy',
  'gbpchf','gbpcad','gbpaud','gbpnzd','audjpy','audnzd','audchf',
  'audcad','cadjpy','cadchf','nzdjpy','nzdcad','nzdchf','chfjpy',
  // DXY (DX-Y.NYB) excluded: ICE futures contract, not OTC FX.
  // Its JSON uses native yfinance 1D (UTC midnight boundary, same as SPX/WTI/Gold).
  // Must use the non-FX today-bar path; market_state guard handles phantom bars.
]);

// Build a today-bar object from STOOQ_RT_CACHE for a given ohlcId.
// ohlcId (e.g. 'eurusd') maps directly to STOOQ_RT_CACHE keys, with two
// special aliases: gold → xauusd, wti → wti (already correct).
// Returns null when the market is closed and no live session bar should be shown.
function _lwBuildTodayBar(ohlcId) {
  const nowUTC = new Date();
  const dowUTC = nowUTC.getUTCDay(); // 0=Sun, 6=Sat

  // FX markets are closed Saturday and most of Sunday — skip today-bar to avoid
  // injecting a flat open=close phantom doji after the last real bar.
  // Exception: Sunday >= 21:00 UTC — the FX week opens (Sydney/Tokyo session).
  const hourUTC = nowUTC.getUTCHours();
  if (_LW_FX_IDS.has(ohlcId) && dowUTC === 6) return null;  // all Saturday
  if (_LW_FX_IDS.has(ohlcId) && dowUTC === 0 && hourUTC < 21) return null;  // Sunday before open

  // FX Friday-after-close guard: after 21:00 UTC on Friday the session boundary
  // logic (hourUTC >= 21 → use tomorrow's date) produces dateStr = Saturday.
  // No FX session opens on Saturday — returning that bar creates a phantom May 9-type
  // candle that should not exist. The weekend guard above only catches Sat/Sun UTC days;
  // this closes the Friday-night gap window (21:00 UTC Fri → 00:00 UTC Sat).
  if (_LW_FX_IDS.has(ohlcId) && dowUTC === 5 && hourUTC >= 21) return null;

  // STOOQ_RT_CACHE key for this ohlcId
  const cacheKey = ohlcId === 'gold' ? 'xauusd' : ohlcId;
  const q = STOOQ_RT_CACHE[cacheKey];
  if (!q || !q.close || isNaN(q.close) || q.close <= 0) return null;

  const isFxBar = _LW_FX_IDS.has(ohlcId);

  // ── Date for the today-bar ──────────────────────────────────────────────────
  // The "correct" date for the today-bar is the session date that the live price
  // belongs to — NOT necessarily the current UTC calendar date.
  //
  // FX (OTC, 21:00 UTC session boundary):
  //   fetch_fx_ohlc_from_1h assigns each day's bar to the UTC date of the session
  //   OPEN (21:00 UTC). Between 21:00–00:00 UTC the new session has started but the
  //   calendar hasn't flipped. Fix: if hourUTC >= 21, use tomorrow's date.
  //
  // Non-FX with session_boundary instruments (CME Gold/WTI open 23:00 UTC,
  //   ICE DXY opens 22:00 UTC):
  //   Between the session open and midnight UTC, Yahoo already reflects the NEW
  //   session's OHLC (open/high/low/close) while the UTC calendar date is still
  //   yesterday. Using nowUTC.toISOString() would assign these new-session prices
  //   to the PREVIOUS day's bar date, overwriting the completed bar with wrong data.
  //
  //   Fix: use market_time (regularMarketTime Unix timestamp) to derive the date.
  //   market_time is the timestamp of the LAST TRADE, which is in the current session.
  //   Its UTC date is the correct bar date — it already accounts for any boundary.
  //   This is superior to hardcoding per-instrument boundaries.
  //
  // Non-FX standard exchanges (SPX, Nikkei, Stoxx — close well before 22:00 UTC):
  //   market_time from the closed session will have the same UTC date as the clock,
  //   so using market_time date == using UTC date: no change in behavior.
  // Session-boundary UTC hour for instruments that reopen before calendar midnight.
  // FX OTC: 21:00 UTC (17:00 EDT) / 22:00 UTC (17:00 EST)
  // DXY (ICE): 22:00 UTC (17:00 EDT) / 23:00 UTC (17:00 EST) — same as FX but 1h later
  let dateStr;
  if (isFxBar) {
    if (hourUTC >= 21) {
      // The FX session boundary is 21:00 UTC. A bar at or after 21:00 UTC belongs to
      // the session that will be dated TOMORROW in fetch_ohlc.py.
      //
      // Gap-window fix (21:00–22:30 UTC):
      // The OHLC workflow runs at 22:30 UTC. Between 21:00 and 22:30 UTC the OHLC JSON
      // was written by YESTERDAY's run, so it ends at the session dated (yesterday) — the
      // session that closed just now at 21:00 UTC today is NOT yet in the JSON.
      // Injecting a today-bar dated tomorrow creates a visual gap (missing today bar).
      //
      // Detection: if the last JSON bar date < today UTC, the JSON is stale and the
      // just-closed session is missing. In that case, date the today-bar TODAY so it
      // fills the gap, representing the closed session via session_high/session_low
      // (which fetch_intraday_quotes.py computes over the full 21:00→21:00 window).
      //
      // After 22:30 UTC, the OHLC workflow writes the completed today bar into the JSON.
      // _lwLastJsonBarDate then equals today, the condition is false, and the tomorrow
      // date is used correctly for the new live session.
      const todayUtcStr = nowUTC.toISOString().slice(0, 10);
      const jsonIsStale = _lwLastJsonBarDate != null && _lwLastJsonBarDate < todayUtcStr;
      if (jsonIsStale) {
        // JSON doesn't have today's completed session yet — use today's date to fill the gap.
        dateStr = todayUtcStr;
      } else {
        const tomorrow = new Date(nowUTC);
        tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
        dateStr = tomorrow.toISOString().slice(0, 10);
      }
    } else {
      dateStr = nowUTC.toISOString().slice(0, 10);
    }
  } else if (q.market_time != null) {
    // Non-FX: use the raw UTC calendar date of the last trade as the bar date.
    //
    // PREVIOUS APPROACH (removed): advanced dateStr by +1 day when market_time's UTC
    // hour >= the session reopen boundary (22 UTC for DXY/Gold/WTI in EDT). The intent
    // was to match fetch_ohlc.py's historical-bar convention, where session_date = the
    // calendar date of the session CLOSE (i.e. the next calendar day after the open).
    //
    // WHY THAT CAUSED THE DUPLICATE:
    // At 22:57 UTC May 7 (DXY reopened at 22:00 UTC):
    //   market_time UTC date = 2026-05-07, hour 22 >= boundary 22 → advance → '2026-05-08'
    //   strip also advances → _stripFrom = '2026-05-08' → JSON May 7 bar NOT stripped
    //   update({time:'2026-05-08'}) injected a new bar
    //   Result: JSON May 7 (complete) + live May 8 (57-min doji) → visual "duplicate"
    //   TradingView shows only ONE bar because it dates the live session bar by its OPEN date.
    //
    // CORRECT APPROACH (session-open date):
    // The live today-bar represents the session that IS OPEN RIGHT NOW. Its natural date
    // is the UTC calendar date when the session started — the market_time UTC date without
    // any advance. This always matches TradingView's behavior for ICE/CME instruments.
    //
    // Consistency with strip: the strip block below uses the same raw market_time date,
    // so _stripFrom = market_time UTC date, which strips the JSON bar for the same date
    // and lets update() replace it with the live data. No phantom second candle.
    //
    // Edge case (post-midnight, same session): at 00:30 UTC May 8 the session that
    // opened at 22:00 May 7 is still running. market_time UTC date = '2026-05-08'.
    // _stripFrom = '2026-05-08'. JSON ends at May 7. Nothing stripped. update() adds
    // May 8 bar. Correct — this is a new calendar date, a naturally separate candle.
    const _mtDate = new Date(q.market_time * 1000);
    dateStr = _mtDate.toISOString().slice(0, 10);
  } else {
    // Fallback: no market_time available — use UTC clock date.
    dateStr = nowUTC.toISOString().slice(0, 10);
  }

  // ── Non-FX: guard against phantom bars on closed exchanges ─────────────────
  // When a non-FX exchange is CLOSED and its last trade was on a prior calendar
  // date, injecting a bar dated today creates a phantom candle built from
  // yesterday's closing price (e.g. an SPX bar dated 2026-05-01 at 01:00 UTC).
  // Use market_state + market_time from quotes.json (populated by
  // fetch_intraday_quotes.py via ticker.info) to detect this precisely.
  if (!isFxBar && q.market_state != null && q.market_time != null) {
    const isClosed = (q.market_state === 'CLOSED' || q.market_state === 'POSTPOST'
                   || q.market_state === 'PREPRE');
    if (isClosed) {
      // market_time is a Unix timestamp in seconds
      const lastTradeDate = new Date(q.market_time * 1000).toISOString().slice(0, 10);
      if (lastTradeDate < dateStr) {
        // Last trade was on a previous date — exchange hasn't opened yet today.
        // Don't inject a phantom bar; the chart ends correctly at the last closed bar.
        return null;
      }
    }
  }

  const dec = { eurusd:5,gbpusd:5,usdjpy:3,audusd:5,usdcad:5,usdchf:5,nzdusd:5,
                eurgbp:5,eurjpy:3,eurchf:5,eurcad:5,euraud:5,eurnzd:5,gbpjpy:3,
                gbpchf:5,gbpcad:5,gbpaud:5,gbpnzd:5,audjpy:3,audnzd:5,audchf:5,
                audcad:5,cadjpy:3,cadchf:5,nzdjpy:3,nzdcad:5,nzdchf:5,chfjpy:3,
                gold:2,wti:2,btc:2,us10y:4,spx:2,nasdaq:2,nikkei:2,stoxx:2,eth:2,dxy:3,
                silver:2,brent:2,dax:2,ftse:2,hsi:2,dji:2 }[ohlcId] ?? 5;
  const c = parseFloat(q.close.toFixed(dec));
  // Candle open convention:
  //   FX pairs  → prev_close (open = last bar's close, consistent with Yahoo daily FX data
  //               convention; ensures candle color always matches the pct sign)
  //   Non-FX    → regularMarketOpen (exchanges have a real session open; use it so the
  //               candle body reflects intraday movement, as TradingView does for BTC/SPX)
  let o;
  if (isFxBar) {
    // FX: anchor candle body to prev_close so green/red == pct direction
    o = q.prev_close != null && q.prev_close > 0
      ? parseFloat(q.prev_close.toFixed(dec))
      : (q.open != null && q.open > 0 ? parseFloat(q.open.toFixed(dec)) : c);
  } else {
    // Non-FX: use the real session open (regularMarketOpen)
    o = q.open != null && q.open > 0
      ? parseFloat(q.open.toFixed(dec))
      : (q.prev_close != null && q.prev_close > 0 ? parseFloat(q.prev_close.toFixed(dec)) : c);
  }
  let h, l;
  if (isFxBar) {
    // For FX, prefer the session H/L (computed from 1H bars over the 21:00 UTC boundary)
    // over Yahoo's dayHigh/dayLow (which uses a UTC-midnight cutoff and, critically,
    // is NOT cleared at the FX session open — Yahoo keeps serving Friday's H/L range
    // through the early hours of Monday UTC until real intraday ticks accumulate).
    // If session H/L are unavailable (e.g. at session open when 0 bars have been
    // aggregated yet), fall back to the o/c range only — never to stale dayH/dayL.
    if (q.session_high != null && q.session_high > 0 &&
        q.session_low  != null && q.session_low  > 0) {
      h = parseFloat(q.session_high.toFixed(dec));
      l = parseFloat(q.session_low.toFixed(dec));
    } else {
      h = Math.max(o, c);
      l = Math.min(o, c);
    }
  } else {
    h = q.high != null && q.high > 0 ? parseFloat(q.high.toFixed(dec)) : Math.max(o, c);
    l = q.low  != null && q.low  > 0 ? parseFloat(q.low.toFixed(dec))  : Math.min(o, c);
  }
  // ── W1/MN: override O/H/L with period-wide accumulated values ─────────────
  // For W1/MN, o/h/l computed above from prev_close/session_high/session_low are
  // wrong for these longer TFs:
  //   open   → prev_close (yesterday close) instead of first D1 open of the period
  //   high   → session_high (last 24h only)  instead of cumulative period high
  //   low    → session_low  (last 24h only)  instead of cumulative period low
  // _lwPeriodOpen/High/Low are snapshotted in _renderLWChart after W1/MN aggregation
  // and hold exactly the values from the aggregated current-period bar (which covers
  // all completed D1 bars in the period). Override here, then let the integrity clamp
  // below extend the wick to include today's live close if it sets a new period extreme.
  if ((_lwActiveTf === 'W1' || _lwActiveTf === 'MN') &&
      _lwPeriodOpen != null && _lwPeriodHigh != null && _lwPeriodLow != null) {
    o = parseFloat(_lwPeriodOpen.toFixed(dec));
    h = parseFloat(_lwPeriodHigh.toFixed(dec));
    l = parseFloat(_lwPeriodLow.toFixed(dec));
  }

  // ── OHLC structural integrity clamp ──────────────────────────────────────
  // Guarantee H >= max(O,C) and L <= min(O,C) for every bar, regardless of source.
  // Root cause: the live today-bar uses prev_close as Open (correct for coloring the
  // pct-direction body), but session_high/session_low from quotes.json reflect only
  // real intraday ticks. On gap-down sessions (e.g. USD/JPY May 7 2026), prev_close
  // can exceed session_high by >1 pip, producing H < O — a structurally impossible
  // candle that LightweightCharts renders as a malformed/inverted chart. Same for
  // L > min(O,C) on gap-up sessions. Clamping extends the wick to include the open/
  // close body without discarding the real intraday range.
  h = Math.max(h, o, c);
  l = Math.min(l, o, c);

  // ── FX stale-quote guard ─────────────────────────────────────────────────
  // At the very start of the FX week (Sunday 21:00 UTC – Monday ~02:00 UTC),
  // yfinance sometimes returns Friday's closing price as the "live" quote
  // because no real trades have been reported yet in the new session.
  // When that happens, open == high == low == close == prev_close, producing a
  // flat phantom doji that visually duplicates the last completed Friday bar.
  // Guard: if the bar is a pure doji (o == h == l == c) AND it falls on the
  // same calendar date as the latest completed OHLC bar, skip it entirely.
  // (LightweightCharts silently overwrites any bar with the same date, so even
  // if dateStr differs the doji is harmless — but we still skip it to keep the
  // chart clean and avoid confusing "no change" labels.)
  if (isFxBar && o === h && h === l && l === c) return null;

  // ── W1/MN period-key alignment ────────────────────────────────────────────
  // W1 and MN bars are aggregated from D1 bars and keyed by ISO Monday
  // (YYYY-MM-DD of Monday) and month start (YYYY-MM-01) respectively.
  // dateStr above is a daily date (YYYY-MM-DD). If we pass it as-is to
  // LWC update(), it won't match any existing aggregated bar and LWC will
  // append a new orphan candle instead of updating the current period.
  // Fix: remap dateStr to the period key that the aggregation uses.
  let barTime = dateStr;
  if (_lwActiveTf === 'W1') {
    // ISO Monday of dateStr's week
    const _d = new Date(dateStr + 'T00:00:00Z');
    const _dow = _d.getUTCDay() || 7; // Mon=1 … Sun=7
    const _mon = new Date(_d);
    _mon.setUTCDate(_d.getUTCDate() - (_dow - 1));
    barTime = _mon.toISOString().slice(0, 10);
  } else if (_lwActiveTf === 'MN') {
    // Month start key: YYYY-MM-01
    barTime = dateStr.slice(0, 7) + '-01';
  }

  return { time: barTime, open: o, high: h, low: l, close: c };
}

// Push/update the live today-bar on the active LW chart (called every 5 min).
// Safe to call when no chart is open — exits silently.
function _lwUpdateTodayBar() {
  if (!_lwCandleSeries || !_lwActiveOhlcId) return;

  // H1/H4 live partial-bar update
  // H1/H4 bars come from static JSON files updated every hour Mon–Fri (:30 UTC).
  // The JSON gap is at most 1 H1 period. The partial bar is the current incomplete block.
  // We build a live partial bar from STOOQ_RT_CACHE:
  //   time  = unix timestamp of the start of the current H1 or H4 UTC block
  //   open  = close of the last completed H1/H4 bar in the JSON (Bloomberg standard)
  //   high  = running block high since block start (resets at block boundary)
  //   low   = running block low since block start (resets at block boundary)
  //   close = live close from cache (Finnhub tick or yfinance 5-min poll)
  // LightweightCharts.update() appends or replaces only the current block's bar --
  // it never touches earlier completed bars. Completely safe.
  if (_lwActiveTf === 'H1' || _lwActiveTf === 'H4') {
    const _isFxId = _LW_FX_IDS?.has(_lwActiveOhlcId) ?? false;
    const _ck = _lwActiveOhlcId === 'gold' ? 'xauusd' : _lwActiveOhlcId;
    const _rt = STOOQ_RT_CACHE[_ck];
    if (!_rt?.close || !(_rt.close > 0)) return;

    const _now = new Date();

    // Compute the start of the current H1 or H4 block (aligned to UTC clock)
    let _blockTs;
    if (_lwActiveTf === 'H1') {
      const _d = new Date(_now);
      _d.setUTCMinutes(0, 0, 0);
      _blockTs = Math.floor(_d.getTime() / 1000);
    } else {
      const _blockH = Math.floor(_now.getUTCHours() / 4) * 4;
      const _d = new Date(_now);
      _d.setUTCHours(_blockH, 0, 0, 0);
      _blockTs = Math.floor(_d.getTime() / 1000);
    }

    // Skip weekend for FX (Sat all-day, Sun before 21:00 UTC, Fri after 21:00 UTC)
    const _utcDay = _now.getUTCDay();
    const _utcH   = _now.getUTCHours();
    const _isFxWeekend = _isFxId && (
      _utcDay === 6 ||
      (_utcDay === 0 && _utcH < 21) ||
      (_utcDay === 5 && _utcH >= 21)
    );
    if (_isFxWeekend) return;

    const _c = _rt.close;
    // Bloomberg institutional standard: H1/H4 open = close of the last completed bar
    // in the JSON (the most recent finished H1/H4 candle), NOT the daily prev_close.
    // Using prev_close (D-1 daily close) made the live bar's body span the entire
    // trading session instead of just the current H1/H4 period — structurally wrong.
    // _lwLastIntradayBarClose is set by _renderLWChart after setData() for H1/H4.
    // Falls back to close (open=close, doji candle) if the bar hasn't been set yet.
    const _o = (_lwLastIntradayBarClose != null && _lwLastIntradayBarClose > 0)
      ? _lwLastIntradayBarClose
      : _c;

    // ── Per-block H/L tracking (Bloomberg standard for live partial bars) ──────
    // session_high/session_low span the full 21:00 UTC trading session — using them
    // for the current H1/H4 block would show the day's full range on the partial bar,
    // which is structurally incorrect (a 14:00–15:00 bar showing the 05:00 session high).
    // Instead, maintain running block H/L that resets at every block boundary.
    if (_lwBlockTs !== _blockTs) {
      // Block has rolled over — the previous block is now complete.
      // Update _lwLastIntradayBarClose to the close of the completed block so the
      // new block's open = last completed H1/H4 bar close (Bloomberg standard).
      // Without this, _lwLastIntradayBarClose stays at the stale value from page-load
      // for the entire session, making every subsequent hour's open wrong.
      if (_lwBlockTs !== null && _c > 0) {
        _lwLastIntradayBarClose = _c;
      }
      // Reset block H/L tracking to the current price at the rollover point.
      _lwBlockHigh = _c;
      _lwBlockLow  = _c;
      _lwBlockTs   = _blockTs;
    }
    // Always update running H/L with the latest tick
    _lwBlockHigh = Math.max(_lwBlockHigh ?? _c, _o, _c);
    _lwBlockLow  = Math.min(_lwBlockLow  ?? _c, _o, _c);
    const _h2 = _lwBlockHigh;
    const _l2 = _lwBlockLow;
    if (!(_h2 > 0 && _l2 > 0 && _h2 >= _l2)) return;

    const _liveBar = { time: _blockTs, open: _o, high: _h2, low: _l2, close: _c };
    try {
      const _isLA = (window._lwChartType === 'line' || window._lwChartType === 'area');
      _lwCandleSeries.update(_isLA ? { time: _blockTs, value: _c } : _liveBar);
    } catch(_) {}

    // Sync chart header % with RT data
    if (_lwActiveUpdateHeader && _rt.pct != null && _lwActivePrevCloseMap) {
      _lwActiveUpdateHeader(_liveBar, null, { pct: _rt.pct, chg: _rt.chg });
    }
    return;
  }

  // D1 / W1 / MN live today-bar (unchanged path)
  const bar = _lwBuildTodayBar(_lwActiveOhlcId);
  if (!bar) return;
  try {
    // Line/Area series use {time, value} -- not OHLC format
    const isLineArea = (window._lwChartType === 'line' || window._lwChartType === 'area');
    _lwCandleSeries.update(isLineArea ? { time: bar.time, value: bar.close } : bar);
  } catch(_) {}

  // Sync the chart header % with yfinance RT data -- DIRECT from rt.pct/rt.chg,
  // never recalculated from bar OHLC differences.
  if (_lwActiveUpdateHeader) {
    const cacheKey = _lwActiveOhlcId === 'gold' ? 'xauusd' : _lwActiveOhlcId;
    const rt = STOOQ_RT_CACHE[cacheKey];
    if (rt?.pct != null && rt.pct !== undefined && _lwActivePrevCloseMap) {
      if (rt.open != null && rt.open > 0) {
        _lwActivePrevCloseMap.set(bar.time, rt.open);
      }
      _lwActiveUpdateHeader(bar, null, { pct: rt.pct, chg: rt.chg });
    } else {
      _lwActiveUpdateHeader(bar, null, null);
    }
  }
}

// Apply a date-range window to the active LW chart.
// days=0 → fit all data. Otherwise show the last N calendar days.
let _lwTotalBars = 0;  // set after each chart load; used by range buttons

function _lwSetRange(days, totalBars) {
  if (!_lwChart) return;
  // If totalBars provided, update the stored value
  if (totalBars != null) _lwTotalBars = totalBars;
  const n = _lwTotalBars;
  const ts = _lwChart.timeScale();

  if (days === 0) {
    ts.fitContent();
    document.querySelectorAll('.lw-range-btn').forEach(b => b.classList.toggle('active', b.dataset.days === '0'));
    _lwActiveDays = 0;
    return;
  }

  // LW Charts v4.2: index 0 = FIRST bar, index (n-1) = LAST bar.
  if (n < 1) { ts.fitContent(); return; }

  // Convert calendar days → logical bar count based on active timeframe.
  let barsPerDay;
  switch (_lwActiveTf) {
    case 'H1': barsPerDay = 17;      break; // FX ~17 1H bars/calendar-day
    case 'H4': barsPerDay = 4.25;    break; // FX ~4.25 H4 bars/calendar-day
    case 'W1': barsPerDay = 1 / 7;   break; // 1 weekly bar per 7 days
    case 'MN': barsPerDay = 1 / 30;  break; // 1 monthly bar per 30 days
    default:   barsPerDay = 5 / 7;   break; // D1: 5 trading days per week
  }
  const tradingBars = Math.round(days * barsPerDay);
  const rightPad    = 8;
  const from = n - tradingBars - 1;
  const to   = n + rightPad - 1;

  // If computed range would exceed total bars, just fitContent
  if (tradingBars >= n) { ts.fitContent(); _lwActiveDays = days; return; }

  setTimeout(() => {
    try { ts.setVisibleLogicalRange({ from: Math.max(0, from), to }); } catch (_) { ts.fitContent(); }
  }, 30);

  document.querySelectorAll('.lw-range-btn').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.days) === days);
  });
  _lwActiveDays = days;
}

let _lwActiveDays = 91; // default: 3M (calendar days)
let _lwActiveTf   = 'D1'; // active timeframe: H1 | H4 | D1 | W1 | MN
let _lwCompareSeries = null;  // LineSeries for compare overlay
let _lwCompareId     = null;  // ohlcId of the compared symbol
// Fullscreen: DOM-lift vars are declared in the FS block below

// Institutional standard: the open of a live H1/H4 partial bar = close of the last
// completed bar in the JSON, not the daily prev_close.  Bloomberg H1: open = first
// real tick of that hour = last bar's close.  Stored here after each setData() call
// so _lwUpdateTodayBar() can use it without the bars array being in scope.
let _lwLastIntradayBarClose = null; // set by _renderLWChart for H1/H4, null for D1+


// Per-block H/L tracking for H1/H4 live partial bar (Bloomberg standard).
// H1/H4 live bar H/L must reflect only the CURRENT incomplete block's tick range,
// not the full session high/low (which spans the entire 21:00 UTC trading session).
// These globals are reset whenever the block boundary changes and updated on every
// Finnhub tick or yfinance poll — producing the correct intrabar range at all times.
let _lwBlockHigh      = null; // running high within the current H1/H4 block
let _lwBlockLow       = null; // running low within the current H1/H4 block
let _lwBlockTs        = null; // unix ts of the current block start (detects rollovers)
let _lwPeriodOpen     = null; // W1/MN: open of the current period (first D1 open) — set after W1/MN aggregation
let _lwPeriodHigh     = null; // W1/MN: cumulative high of all D1 bars in the current period — set after W1/MN aggregation
let _lwPeriodLow      = null; // W1/MN: cumulative low  of all D1 bars in the current period — set after W1/MN aggregation

// Render a Lightweight Charts candlestick chart inside #tv-chart-wrap
async function _renderLWChart(ohlcId, label) {
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;

  _chartMode = 'lw'; // set synchronously — visibility handler checks this, not _lwChart
  _destroyLWChart();
  wrap.innerHTML = '';

  // Loading state
  const loader = document.createElement('div');
  loader.style.cssText = 'height:100%;display:flex;align-items:center;justify-content:center;color:var(--text2);font-size:12px;font-family:var(--font-ui,sans-serif);';
  loader.textContent = 'Loading chart\u2026';
  wrap.appendChild(loader);

  await _ensureLWLib();

  // ── Resolve JSON path based on active timeframe ──────────────────────────────
  // H1/H4: ohlc-data/h1/{id}.json or ohlc-data/h4/{id}.json (unix timestamp bars)
  // D1/W1/MN: ohlc-data/{id}.json (YYYY-MM-DD date bars); W1/MN aggregated below
  const _activeTf = _lwActiveTf;
  const _isIntradayTf = (_activeTf === 'H1' || _activeTf === 'H4');
  let _jsonPath;
  if (_activeTf === 'H1')      _jsonPath = './ohlc-data/h1/' + ohlcId + '.json';
  else if (_activeTf === 'H4') _jsonPath = './ohlc-data/h4/' + ohlcId + '.json';
  else                         _jsonPath = './ohlc-data/' + ohlcId + '.json';

  const r = await fetch(_jsonPath, { signal: AbortSignal.timeout(6000) });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  let bars = await r.json();
  if (!Array.isArray(bars) || bars.length < 10) throw new Error('insufficient data');

  // ── H1/H4 FX gap-fill via Cloudflare Worker /candles ─────────────────────────
  // The JSON is updated every :30 UTC Mon–Fri. At worst, 1 completed H1 bar or
  // 3 completed H4 bars are missing (bars that closed after the last workflow run
  // but before the user opened the chart). This block fetches those missing completed
  // bars from Finnhub via the CF Worker and splices them in before setData().
  //
  // Scope: FX pairs only (Finnhub OANDA covers exactly the 28 pairs in _LW_FX_IDS).
  //        H1/H4 only (unix timestamp bars). Non-FX (gold, BTC, etc.) has no
  //        Finnhub FX equivalent — their gap is handled by _lwUpdateTodayBar alone.
  //
  // Failure mode: silent — if the Worker is unreachable, returns empty, or times out
  //               (1.5s budget), the chart renders normally with the JSON bars and
  //               the live partial bar from _lwUpdateTodayBar. No user-visible error.
  if (_isIntradayTf && _LW_FX_IDS.has(ohlcId)) {
    try {
      const _resolutionSec  = (_activeTf === 'H1') ? 3600 : 14400;
      const _lastJsonTs     = bars[bars.length - 1].time;
      const _nowUTC2        = new Date();
      const _utcDow         = _nowUTC2.getUTCDay();
      const _utcHr          = _nowUTC2.getUTCHours();

      // Skip outside FX market hours
      const _fxClosed = (
        _utcDow === 6 ||
        (_utcDow === 0 && _utcHr < 21) ||
        (_utcDow === 5 && _utcHr >= 21)
      );

      // Current live block start (in-progress bar — must be excluded)
      let _currentBlockTs;
      if (_activeTf === 'H1') {
        const _d = new Date(_nowUTC2);
        _d.setUTCMinutes(0, 0, 0);
        _currentBlockTs = Math.floor(_d.getTime() / 1000);
      } else {
        const _blockH = Math.floor(_nowUTC2.getUTCHours() / 4) * 4;
        const _d = new Date(_nowUTC2);
        _d.setUTCHours(_blockH, 0, 0, 0);
        _currentBlockTs = Math.floor(_d.getTime() / 1000);
      }

      if (!_fxClosed) {
        // ── Session start: most recent Sunday 21:00 UTC ───────────────────────
        // We fetch Finnhub bars from session open to current block. This lets us:
        // (a) replace yfinance artifact bars in the JSON (O≈L / C≈L artifacts that
        //     occur in the first hours of the FX week), AND
        // (b) fill any gap between the last JSON bar and the current live block.
        // Finnhub OANDA data for the current session is consistently cleaner than
        // the yfinance stub bars produced at session open.
        const _daysSinceSun = _utcDow;                  // Sun=0, Mon=1 … Sat=6 — days since last Sunday
        const _lastSun      = new Date(_nowUTC2);
        _lastSun.setUTCDate(_nowUTC2.getUTCDate() - _daysSinceSun);
        _lastSun.setUTCHours(21, 0, 0, 0);
        // If the computed Sunday 21:00 is in the future (e.g. it's Sunday but before 21:00),
        // step back 7 days — but _fxClosed already guards that case above.
        const _sessionStartTs = Math.floor(_lastSun.getTime() / 1000);

        // Only fire the fetch if there are bars in the current session window
        // (avoids a request when session just opened and JSON already has today's bars)
        const _sessionBarsInJson = bars.filter(b => b.time >= _sessionStartTs && b.time < _currentBlockTs);
        const _expectedNextTs    = _lastJsonTs + _resolutionSec;
        const _hasGap            = _expectedNextTs < _currentBlockTs;
        const _sessionHasData    = _sessionBarsInJson.length > 0;

        if (_sessionHasData || _hasGap) {
          const _wsUrl      = (typeof FX_PROXY_WS_URL !== 'undefined') ? FX_PROXY_WS_URL : '';
          const _candleBase = _wsUrl.replace(/^wss:\/\//, 'https://').replace(/\/ws$/, '');

          if (_candleBase) {
            const _resParam   = (_activeTf === 'H1') ? '60' : '240';
            // Request from session start (to capture artifact bars) up to the current block
            const _candleUrl  = `${_candleBase}/candles?id=${encodeURIComponent(ohlcId)}&resolution=${_resParam}&from=${_sessionStartTs}&to=${_currentBlockTs}`;

            const _gapResp = await fetch(_candleUrl, { signal: AbortSignal.timeout(2000) });
            if (_gapResp.ok) {
              const _gapData = await _gapResp.json();
              if (Array.isArray(_gapData.bars) && _gapData.bars.length > 0) {
                // Validate bars: completed, within session window, sensible OHLC values
                const _finnhubBars = _gapData.bars.filter(b =>
                  b.time >= _sessionStartTs && b.time < _currentBlockTs &&
                  b.open > 0 && b.high > 0 && b.low > 0 && b.close > 0 &&
                  b.high >= b.open && b.high >= b.close &&
                  b.low  <= b.open && b.low  <= b.close
                );
                if (_finnhubBars.length > 0) {
                  _finnhubBars.sort((a, b) => a.time - b.time);
                  // Build a timestamp Set for O(1) lookup
                  const _finnhubTs = new Set(_finnhubBars.map(b => b.time));
                  // Keep JSON bars that predate the session (historical) or are not
                  // covered by Finnhub (non-FX session bars). Replace everything
                  // within the session window that Finnhub returned.
                  // Keep pre-session bars (historical, unaffected by artifacts).
                  // Discard session bars covered by Finnhub (cleaner OANDA data).
                  // Keep any in-session bars Finnhub didn't return (defensive).
                  const _keptJsonBars = bars.filter(b =>
                    b.time < _sessionStartTs ||
                    (b.time >= _sessionStartTs && !_finnhubTs.has(b.time) && b.time < _currentBlockTs)
                  );
                  bars = [..._keptJsonBars, ..._finnhubBars].sort((a, b) => a.time - b.time);
                }
              }
            }
          }
        }
      }
    } catch (_gapErr) {
      // Silent fallback — if Worker unreachable/timeout, chart renders with JSON bars.
      // _lwUpdateTodayBar() always covers the live block regardless.
    }
  }

  // ── W1/MN aggregation from D1 bars ───────────────────────────────────────────
  // For W1: group D1 bars by ISO week Monday. For MN: group by YYYY-MM-01.
  // H1/H4 bars already have unix timestamps and need no aggregation.
  if (_activeTf === 'W1' || _activeTf === 'MN') {
    const agg = {};
    for (const b of bars) {
      let key;
      if (_activeTf === 'W1') {
        // ISO week Monday date
        const d   = new Date(b.time + 'T00:00:00Z');
        const dow = d.getUTCDay() || 7; // Mon=1 … Sun=7
        const mon = new Date(d);
        mon.setUTCDate(d.getUTCDate() - (dow - 1));
        key = mon.toISOString().slice(0, 10);
      } else {
        key = b.time.slice(0, 7) + '-01'; // YYYY-MM-01
      }
      if (!agg[key]) {
        agg[key] = { time: key, open: b.open, high: b.high, low: b.low, close: b.close };
      } else {
        const a   = agg[key];
        a.high    = Math.max(a.high, b.high);
        a.low     = Math.min(a.low,  b.low);
        a.close   = b.close;
      }
    }
    bars = Object.values(agg).sort((a, b) => a.time < b.time ? -1 : 1);
    if (bars.length < 4) throw new Error('insufficient aggregated data');
    // ── Snapshot current-period O/H/L for _lwBuildTodayBar ─────────────────
    // The last aggregated bar IS the current incomplete period (it will be stripped
    // below and replaced by the live today-bar). Capture its accumulated O/H/L now
    // so _lwBuildTodayBar can produce a bar with the REAL period open and cumulative
    // period H/L rather than prev_close + session H/L (today only).
    const _curPeriodBar = bars[bars.length - 1];
    _lwPeriodOpen = _curPeriodBar ? _curPeriodBar.open : null;
    _lwPeriodHigh = _curPeriodBar ? _curPeriodBar.high : null;
    _lwPeriodLow  = _curPeriodBar ? _curPeriodBar.low  : null;
  }

  // ── Today-bar strip and gap-window injection (D1/W1/MN only) ─────────────────
  // For H1/H4 intraday TFs: bars have unix timestamps, no live today-bar to inject.
  if (!_isIntradayTf) {
  // _lwLastJsonBarDate was already set from raw D1 bars before W1/MN aggregation.
  // For plain D1 TF (no aggregation), bars[] was never mutated — update it here too
  // so D1 stays consistent. Skip for W1/MN: bars[] now holds aggregated period keys
  // (e.g. '2026-05-01') which would make the gap-window stale check always fire.
  if (_activeTf === 'D1') {
    _lwLastJsonBarDate = bars[bars.length - 1]?.time ?? null;
  }

  // ── Strip today-bar from JSON before setData ────────────────────────────────
  // fetch_ohlc.py keeps today's in-progress bar in the JSON. dashboard.js replaces
  // it with the live price via candleSeries.update(todayBar). Without stripping,
  // two bars appear for the same session (stale JSON + live update).
  //
  // _stripFrom must match exactly what _lwBuildTodayBar assigns as dateStr.
  // For non-FX instruments (DXY, Gold, WTI): both use the raw market_time UTC date
  // with no session-boundary advance. See _lwBuildTodayBar for the full rationale.
  {
    const _isFxStrip = _LW_FX_IDS.has(ohlcId);
    const _nowUTC    = new Date();
    const _hourUTC   = _nowUTC.getUTCHours();
    let   _stripFrom;
    if (_isFxStrip && _hourUTC >= 21) {
      // FX: new session started at 21:00 UTC.
      // _stripFrom must match _lwBuildTodayBar's dateStr exactly.
      // Gap-window: if the JSON is stale (last bar < today), today-bar is dated TODAY.
      //   → strip bars >= today (i.e. _stripFrom = today). In practice the JSON ends at
      //   yesterday, so nothing is stripped — the today-bar fills the gap cleanly.
      // Normal: JSON has today's bar, today-bar is dated tomorrow.
      //   → strip bars >= tomorrow (i.e. _stripFrom = tomorrow).
      const _todayStr = _nowUTC.toISOString().slice(0, 10);
      const _jsonStale = _lwLastJsonBarDate != null && _lwLastJsonBarDate < _todayStr;
      if (_jsonStale) {
        _stripFrom = _todayStr;
      } else {
        const _tom = new Date(_nowUTC);
        _tom.setUTCDate(_tom.getUTCDate() + 1);
        _stripFrom = _tom.toISOString().slice(0, 10);
      }
    } else if (!_isFxStrip) {
      // Non-FX: use raw market_time UTC date as _stripFrom — no boundary advance.
      // This mirrors the fix applied to _lwBuildTodayBar: both use the session-open
      // date (raw market_time UTC date) so they always agree. stripFrom = todayBar.time,
      // which strips exactly the JSON bar that the live bar will replace via update().
      const _ck = ohlcId === 'gold' ? 'xauusd' : ohlcId;
      const _qt = STOOQ_RT_CACHE[_ck];
      if (_qt?.market_time != null) {
        const _mtDate = new Date(_qt.market_time * 1000);
        _stripFrom = _mtDate.toISOString().slice(0, 10);
      } else {
        // Cache not ready yet — fall back to UTC clock date
        _stripFrom = _nowUTC.toISOString().slice(0, 10);
      }
    } else {
      // FX before 21:00 UTC: strip today UTC
      _stripFrom = _nowUTC.toISOString().slice(0, 10);
    }
    bars = bars.filter(b => b.time < _stripFrom);
    if (bars.length < 10) throw new Error('insufficient data after strip');

    // ── Gap-window prev-bar injection ───────────────────────────────────────
    // The OHLC gap window spans 21:00 UTC (session close) → 01:30 UTC next day
    // (when the OHLC workflow writes the completed bar).  This crosses midnight UTC,
    // so two separate hour ranges must be handled:
    //
    //   A) 21:00–23:59 UTC (same calendar day as session close):
    //      _hourUTC >= 21.  The strip block already used _stripFrom = today because
    //      _jsonStale was true.  The gap is active.
    //
    //   B) 00:00–01:29 UTC (calendar day has flipped to the next day):
    //      _hourUTC < 21.  The strip block used _stripFrom = today (UTC date has
    //      advanced by 1 relative to the gap start).  The JSON is still stale
    //      (_lwLastJsonBarDate = two calendar days ago) but the hour check in the
    //      original guard (_hourUTC >= 21) excluded this window.  Fix: also check
    //      _lwLastJsonBarDate < (today − 1 day) to detect the cross-midnight stale.
    //
    // Guard conditions (all must be true to inject):
    //   1. The pair is an FX pair (only FX uses the 21:00 UTC boundary)
    //   2. The OHLC JSON is stale — two sub-cases:
    //      A) hourUTC >= 21 AND lastJsonBar < today  (same-night window)
    //      B) hourUTC <  21 AND lastJsonBar < yesterday  (cross-midnight window, 00:00–01:30)
    //   3. The STOOQ_RT_CACHE entry has a valid prev_bar from quotes.json
    //   4. The prev_bar.time is strictly later than the last bar in the stripped
    //      array and strictly earlier than _stripFrom (no collision, no duplicate)
    if (_isFxStrip) {
      const _todayStr2     = _nowUTC.toISOString().slice(0, 10);
      const _yesterdayDate = new Date(_nowUTC);
      _yesterdayDate.setUTCDate(_yesterdayDate.getUTCDate() - 1);
      const _yesterdayStr2 = _yesterdayDate.toISOString().slice(0, 10);

      // Case A: 21:00–23:59 UTC — same night as session close
      const _gapA = _hourUTC >= 21 && _lwLastJsonBarDate != null && _lwLastJsonBarDate < _todayStr2;
      // Case B: 00:00–01:29 UTC — cross-midnight (JSON still stale from yesterday's gap)
      const _gapB = _hourUTC < 21 && _lwLastJsonBarDate != null && _lwLastJsonBarDate < _yesterdayStr2;
      const _isGapWindow = _gapA || _gapB;

      if (_isGapWindow) {
        const _cacheKey = ohlcId === 'gold' ? 'xauusd' : ohlcId;
        const _q = STOOQ_RT_CACHE[_cacheKey];
        const _pb = _q?.prev_bar;
        if (_pb && _pb.time && _pb.open > 0 && _pb.high > 0 && _pb.low > 0 && _pb.close > 0) {
          const _lastBarTime = bars.length > 0 ? bars[bars.length - 1].time : null;
          const _pbInRange   = (!_lastBarTime || _pb.time > _lastBarTime) && _pb.time < _stripFrom;
          if (_pbInRange) {
            const _dec2 = { eurusd:5,gbpusd:5,usdjpy:3,audusd:5,usdcad:5,usdchf:5,nzdusd:5,
                            eurgbp:5,eurjpy:3,eurchf:5,eurcad:5,euraud:5,eurnzd:5,gbpjpy:3,
                            gbpchf:5,gbpcad:5,gbpaud:5,gbpnzd:5,audjpy:3,audnzd:5,audchf:5,
                            audcad:5,cadjpy:3,cadchf:5,nzdjpy:3,nzdcad:5,nzdchf:5,chfjpy:3 }[ohlcId] ?? 5;
            const _pbBar = {
              time:  _pb.time,
              open:  parseFloat(_pb.open.toFixed(_dec2)),
              high:  parseFloat(_pb.high.toFixed(_dec2)),
              low:   parseFloat(_pb.low.toFixed(_dec2)),
              close: parseFloat(_pb.close.toFixed(_dec2)),
            };
            bars.push(_pbBar);
          }
        }
      }
    }
    // ── End gap-window prev-bar injection ───────────────────────────────────
  }
  // ── End today-bar strip ─────────────────────────────────────────────────────
  } // end if (!_isIntradayTf)

  wrap.innerHTML = '';

  // Remove the negative margin used to hide TradingView widget footer — not needed for LW
  wrap.style.marginBottom = '0';

  const chartDiv = document.createElement('div');
  chartDiv.style.cssText = 'width:100%;height:100%;';
  wrap.appendChild(chartDiv);

  // Enable pointer events for LW chart interactivity (zoom, pan, crosshair)
  wrap.style.pointerEvents = 'auto';

  // Decimal precision map — drives minMove and formatting
  const dec = { eurusd:5,gbpusd:5,usdjpy:3,audusd:5,usdcad:5,usdchf:5,nzdusd:5,
                eurgbp:5,eurjpy:3,eurchf:5,eurcad:5,euraud:5,eurnzd:5,gbpjpy:3,
                gbpchf:5,gbpcad:5,gbpaud:5,gbpnzd:5,audjpy:3,audnzd:5,audchf:5,
                audcad:5,cadjpy:3,cadchf:5,nzdjpy:3,nzdcad:5,nzdchf:5,chfjpy:3,
                gold:2,wti:2,btc:2,us10y:4,spx:2,nasdaq:2,nikkei:2,stoxx:2,eth:2,dxy:3,
                silver:2,brent:2,dax:2,ftse:2,hsi:2,dji:2 }[ohlcId] ?? 5;
  // minMove must match the precision: 5dp → 0.00001, 4dp → 0.0001, 3dp → 0.001, 2dp → 0.01
  const minMove = parseFloat((1 / Math.pow(10, dec)).toFixed(dec));

  const LWC = window.LightweightCharts;
  // Use explicit dimensions — autoSize requires ResizeObserver and can mis-size before first paint
  const chartW = wrap.offsetWidth  || wrap.clientWidth  || 600;
  const chartH = wrap.offsetHeight || wrap.clientHeight || 290;

  // Detect if bars have volume data (new fetch_ohlc.py output includes volume field)
  const hasVolume = bars.length > 0 && typeof bars[0].volume === 'number' && bars[0].volume > 0;

  // scaleMargins: reserve bottom 22% for volume pane when data is available
  const mainScaleMargins = hasVolume
    ? { top: 0.08, bottom: 0.22 }
    : { top: 0.10, bottom: 0.08 };

  _lwChart = LWC.createChart(chartDiv, {
    layout:      { background: { color: '#131722' }, textColor: '#d1d4dc', attributionLogo: false },
    grid:        { vertLines: { color: 'rgba(42,46,57,0.5)' }, horzLines: { color: 'rgba(42,46,57,0.5)' } },
    crosshair:   { mode: LWC.CrosshairMode.Normal,
                   vertLine: { color: 'rgba(144,150,160,0.5)', labelBackgroundColor: '#2a2e39' },
                   horzLine: { color: 'rgba(144,150,160,0.5)', labelBackgroundColor: '#2a2e39' } },
    rightPriceScale: { borderColor: '#2a2e39', minimumWidth: 65,
                       scaleMargins: mainScaleMargins },
    timeScale:   { borderColor: '#2a2e39', timeVisible: false, secondsVisible: false,
                   rightOffset: 8, minBarSpacing: 1,
                   fixLeftEdge: false, fixRightEdge: false },
    handleScroll:  { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale:   { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true } },
    localization: { priceFormatter: v => v.toFixed(dec) },
    width:  chartW,
    height: chartH,
  });

  // ── Symbol watermark — institutional standard (Bloomberg shows pair name in chart background) ──
  // Uses LWC v5 createTextWatermark() API — gracefully skipped on older versions
  const _wmLabel = (ohlcId === 'gold' ? 'XAUUSD' : ohlcId === 'wti' ? 'USOIL' : ohlcId.toUpperCase());
  if (typeof window._lwShowWm === 'undefined') window._lwShowWm = false;
  let _wmHandle = null;
  function _applyWatermark() {
    // Remove existing watermark if any
    if (_wmHandle && typeof _wmHandle.detach === 'function') { try { _wmHandle.detach(); } catch(_) {} _wmHandle = null; }
    if (!window._lwShowWm) {
      const _domWm = document.getElementById('_lw-dom-watermark');
      if (_domWm) _domWm.remove();
      return;
    }
    try {
      // Remove DOM-based fallback watermark
      const _domWm = document.getElementById('_lw-dom-watermark');
      if (_domWm) _domWm.remove();
      if (typeof LWC.createTextWatermark === 'function') {
        // Proportional font size: ~15% of chart width, clamped 24–96px
        const _cw2 = chartW || 300;
        const _wmFs = Math.min(Math.max(Math.round(_cw2 * 0.15), 24), 96);
        _wmHandle = LWC.createTextWatermark(_lwChart.panes()[0], {
          horzAlign: 'center',
          vertAlign: 'center',
          lines: [
            { text: _wmLabel, color: 'rgba(209,212,220,0.08)', fontSize: _wmFs, fontWeight: 'bold', fontFamily: 'Inter,sans-serif' },
          ],
        });
      } else {
        // DOM-based fallback — absolutely positioned over chart container
        // Font size proportional to chart width (~15% — Bloomberg standard for pair watermarks),
        // clamped 24–96px so it never overflows on mobile viewports.
        const _chartWrap = document.getElementById('tv-chart-wrap');
        if (_chartWrap) {
          const _cw = _chartWrap.offsetWidth || chartW || 300;
          const _rawFs = Math.round(_cw * 0.15);
          const _fs = Math.min(Math.max(_rawFs, 24), 96);
          const _wm = document.createElement('div');
          _wm.id = '_lw-dom-watermark';
          _wm.textContent = _wmLabel;
          _wm.style.cssText = `position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:${_fs}px;font-weight:700;font-family:var(--font-ui,sans-serif);color:rgba(255,255,255,0.10);pointer-events:none;user-select:none;z-index:1;white-space:nowrap;letter-spacing:${Math.round(_fs*0.05)}px;`;
          _chartWrap.style.position = 'relative';
          _chartWrap.appendChild(_wm);
        }
      }
    } catch(_wmErr) {}
  }
  _applyWatermark();
  // Sync WM button state
  const _wmBtn = document.getElementById('lw-wm-btn');
  if (_wmBtn) {
    _wmBtn.classList.toggle('on', window._lwShowWm);
    _wmBtn.setAttribute('aria-pressed', window._lwShowWm ? 'true' : 'false');
  }

  // ── Chart type selector — Candlestick / Bar / Line / Area (LWC v5 API) ──
  // Bloomberg: Candlestick default; Bar, Line, Area available. Baseline excluded (FX has no natural
  // zero reference). State persisted in window._lwChartType across symbol switches.
  if (typeof window._lwChartType === 'undefined') window._lwChartType = 'candle';

  // Sync TYPE button state on render
  document.querySelectorAll('[data-chart-type]').forEach(btn => {
    const isActive = btn.dataset.chartType === window._lwChartType;
    btn.classList.toggle('sel', isActive);
    btn.classList.remove('on');
  });

  // Helper: convert OHLC bars to close-only for line/area
  const closeBars = bars.filter(b => b.close != null).map(b => ({ time: b.time, value: b.close }));

  let candleSeries;
  const _priceFormat = { type: 'price', precision: dec, minMove };

  if (window._lwChartType === 'bar') {
    // Bar (OHLC) series — same data as candlestick, different visual
    if (typeof LWC.BarSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.BarSeries, {
        upColor: '#26a69a', downColor: '#ef5350',
        openVisible: true, thinBars: false,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addBarSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        priceFormat: _priceFormat,
      });
    }
    candleSeries.setData(bars);
  } else if (window._lwChartType === 'line') {
    // Line series — close prices only
    if (typeof LWC.LineSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.LineSeries, {
        color: '#4f7fff', lineWidth: 2,
        priceLineVisible: false, lastValueVisible: true,
        crosshairMarkerVisible: true, crosshairMarkerRadius: 4,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addLineSeries({ color: '#4f7fff', lineWidth: 2, priceFormat: _priceFormat });
    }
    candleSeries.setData(closeBars);
  } else if (window._lwChartType === 'area') {
    // Area series — close prices with gradient fill
    if (typeof LWC.AreaSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.AreaSeries, {
        lineColor: '#4f7fff', lineWidth: 2,
        topColor: 'rgba(79,127,255,0.28)', bottomColor: 'rgba(79,127,255,0.02)',
        priceLineVisible: false, lastValueVisible: true,
        crosshairMarkerVisible: true, crosshairMarkerRadius: 4,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addAreaSeries({ lineColor: '#4f7fff', lineWidth: 2, priceFormat: _priceFormat });
    }
    candleSeries.setData(closeBars);
  } else {
    // Default: Candlestick — LWC v5 API with v4 fallback
    if (typeof LWC.CandlestickSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.CandlestickSeries, {
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        priceFormat: _priceFormat,
      });
    }
    candleSeries.setData(bars);
  }

  // ── Store last completed bar close for H1/H4 live-bar open (Bloomberg standard) ──
  // Bloomberg H1 open = first real tick of that hour = close of the last completed H1 bar.
  // This is NOT the same as prev_close (daily close from D-1) which the previous version
  // used incorrectly, causing the live bar's body to span the entire session instead of
  // just the current hour. Reset to null for D1/W1/MN (those TFs use _lwBuildTodayBar).
  if (_isIntradayTf && bars.length > 0) {
    _lwLastIntradayBarClose = bars[bars.length - 1].close;
  } else {
    _lwLastIntradayBarClose = null;
  }
  // Reset per-block H/L on every chart load — the block tracking starts fresh
  // from the first tick received, ensuring clean state after TF or symbol changes.
  _lwBlockHigh  = null;
  _lwBlockLow   = null;
  _lwBlockTs    = null;
  // _lwPeriodOpen/High/Low are NOT reset here — they were snapshotted earlier in this
  // same _renderLWChart call (after W1/MN aggregation) and must survive for
  // _lwBuildTodayBar to use. For non-W1/MN TFs they remain null from _destroyLWChart.


  // Uses separate priceScaleId 'volume' pinned to bottom 20% — clean Bloomberg-style presentation
  if (typeof window._lwShowVol === 'undefined') window._lwShowVol = false;
  let volumeSeries = null;
  function _applyVolume() {
    if (volumeSeries) { try { _lwChart.removeSeries(volumeSeries); } catch(_) {} volumeSeries = null; }
    if (!hasVolume || !window._lwShowVol) {
      _lwChart.applyOptions({ layout: {}, timeScale: {} });
      _lwChart.priceScale('right').applyOptions({ scaleMargins: { top: 0.10, bottom: 0.08 } });
      return;
    }
    try {
      const volOpts = {
        priceScaleId: 'volume',
        priceFormat: { type: 'volume' },
        lastValueVisible: false,
        priceLineVisible: false,
      };
      if (typeof LWC.HistogramSeries !== 'undefined') {
        volumeSeries = _lwChart.addSeries(LWC.HistogramSeries, volOpts);
      } else if (typeof _lwChart.addHistogramSeries === 'function') {
        volumeSeries = _lwChart.addHistogramSeries(volOpts);
      }
      if (volumeSeries) {
        _lwChart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.82, bottom: 0 },
          borderVisible: false,
          visible: false,
        });
        _lwChart.priceScale('right').applyOptions({ scaleMargins: { top: 0.08, bottom: 0.22 } });
        const volData = bars.map(b => ({
          time:  b.time,
          value: b.volume,
          color: (b.close >= b.open) ? 'rgba(38,166,154,0.30)' : 'rgba(239,83,80,0.30)',
        }));
        volumeSeries.setData(volData);
      }
    } catch(_volErr) { volumeSeries = null; }
  }
  _applyVolume();
  // Sync VOL button state
  const _volBtn = document.getElementById('lw-vol-btn');
  if (_volBtn) {
    const _volActive = hasVolume && window._lwShowVol;
    _volBtn.classList.toggle('on', _volActive);
    _volBtn.style.opacity = hasVolume ? '1' : '0.4';
    _volBtn.title = hasVolume ? 'Volume histogram' : 'Volume (unavailable — no data)';
    _volBtn.setAttribute('aria-pressed', _volActive ? 'true' : 'false');
  }

  // ── Prev close price line — Bloomberg standard: dashed horizontal reference ──
  // Always visible by default, toggle via PC button
  if (typeof window._lwShowPc === 'undefined') window._lwShowPc = true;
  let _prevCloseLine = null;
  // For D1: bars[-1] is the last completed day before strip — use its close.
  // For W1/MN: bars[-1] is the current INCOMPLETE period (e.g. the May MN bar whose
  // close = last D1 close in the JSON, not the true month close). The "Prev C" line
  // should reflect the PREVIOUS completed period (e.g. April for MN), which is bars[-2].
  // For H1/H4: _prevCloseLine is not shown (PC button is hidden for intraday TFs).
  const _lastHistClose = (() => {
    if (_activeTf === 'W1' || _activeTf === 'MN') {
      return bars.length > 2 ? bars[bars.length - 2].close : null;
    }
    return bars.length > 1 ? bars[bars.length - 1].close : null;
  })();
  function _applyPrevClose() {
    if (_prevCloseLine) { try { candleSeries.removePriceLine(_prevCloseLine); } catch(_) {} _prevCloseLine = null; }
    if (!window._lwShowPc || _lastHistClose == null) return;
    try {
      _prevCloseLine = candleSeries.createPriceLine({
        price: _lastHistClose,
        color: 'rgba(144,150,160,0.55)',
        lineWidth: 1,
        lineStyle: 2, // LineStyle.Dashed
        axisLabelVisible: true,
        axisLabelColor: '#2a2e39',
        axisLabelTextColor: '#848ea0',
        title: 'Prev C',
      });
    } catch(_plErr) {}
  }
  _applyPrevClose();
  // Sync PC button state
  const _pcBtn = document.getElementById('lw-pc-btn');
  if (_pcBtn) {
    _pcBtn.classList.toggle('on', window._lwShowPc);
    _pcBtn.setAttribute('aria-pressed', window._lwShowPc ? 'true' : 'false');
  }

  // ── Log scale toggle state — persists across symbol switches ──
  if (typeof window._lwLogScale === 'undefined') window._lwLogScale = false;
  // Apply persisted log scale mode on each new chart render
  if (window._lwLogScale) {
    try { _lwChart.priceScale('right').applyOptions({ mode: 1 }); } catch(_) {}
  }
  // Sync button visual state
  const _logBtn = document.getElementById('lw-log-btn');
  if (_logBtn) {
    _logBtn.classList.toggle('on', window._lwLogScale);
    _logBtn.setAttribute('aria-pressed', window._lwLogScale ? 'true' : 'false');
  }

  // Store global refs so _lwUpdateTodayBar() can push live prices
  _lwCandleSeries = candleSeries;
  _lwActiveOhlcId = ohlcId;

  // Inject today's live bar immediately (STOOQ_RT_CACHE may already be populated).
  // For D1/W1/MN: _lwBuildTodayBar() constructs the bar.
  // For H1/H4: _lwUpdateTodayBar() handles the live partial-bar injection directly
  //            (block-aligned unix timestamp + per-block running H/L from ticks).
  // todayBar hoisted to function scope — referenced further below for lastBar calculation
  // regardless of TF. For H1/H4 it stays null (live bar pushed via _lwUpdateTodayBar).
  let todayBar = null;
  if (_lwActiveTf === 'H1' || _lwActiveTf === 'H4') {
    _lwUpdateTodayBar();
  } else {
    todayBar = _lwBuildTodayBar(ohlcId);
    if (todayBar) {
      try {
        const _isLA = (window._lwChartType === 'line' || window._lwChartType === 'area');
        candleSeries.update(_isLA ? { time: todayBar.time, value: todayBar.close } : todayBar);
      } catch(_) {}
    }
  }

  // ── Multi-MA legacy state cleanup — MA overlays now handled by Full Indicator Library ──
  // Clear any stale series refs from previous chart renders
  if (window._lwMaState) window._lwMaState.forEach(m => { m.series = null; });

  // ── CB Meeting markers — Bloomberg/Reuters standard: vertical dashed lines with label ──
  // Industry standard: thin vertical line at CB decision date, labeled with the bank acronym
  // (FOMC, ECB, BoE etc.) pinned at the top of the chart area, with a hover tooltip.
  // Implementation: DOM SVG overlay updated via LWC timeScale subscribeVisibleTimeRangeChange
  // and scrolled/zoomed in sync with the chart — same pattern used by institutional terminals.
  if (typeof window._lwShowCb === 'undefined') window._lwShowCb = false;
  let _cbRafId = null;
  let _cbOverlay = null;   // SVG element overlay
  let _cbMeetingData = []; // [{date, cbs:[{cb,color}]}] — built once, reused on each draw

  function _drawCbLines() {
    if (_cbRafId) cancelAnimationFrame(_cbRafId);
    _cbRafId = requestAnimationFrame(() => {
      _cbRafId = null;
      if (!_cbOverlay || !_lwChart || !window._lwShowCb || _cbMeetingData.length === 0) {
        if (_cbOverlay) _cbOverlay.innerHTML = '';
        return;
      }
      const ts = _lwChart.timeScale();
      const chartH = chartDiv.offsetHeight;
      const labelZone = 18; // px from top reserved for labels
      let svgContent = '';
      _cbMeetingData.forEach(ev => {
        try {
          const x = ts.timeToCoordinate(ev.date);
          if (x == null || x < 0 || x > chartDiv.offsetWidth) return;
          // One vertical line per unique date — stack labels if multiple CBs same day
          ev.cbs.forEach((cbItem, i) => {
            const col = cbItem.color;
            const solidCol = col.replace(/rgba\(([^,]+,[^,]+,[^,]+),[^)]+\)/, 'rgba($1,0.55)');
            const labelCol = col.replace(/rgba\(([^,]+,[^,]+,[^,]+),[^)]+\)/, 'rgba($1,0.9)');
            // Dashed vertical line
            svgContent += `<line x1="${x.toFixed(1)}" y1="${labelZone}" x2="${x.toFixed(1)}" y2="${chartH - 28}" `
              + `stroke="${solidCol}" stroke-width="1" stroke-dasharray="3,3"/>`;
            // Label at top
            const labelX = x + 3;
            const labelY = labelZone + i * 12;
            svgContent += `<text x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" `
              + `font-size="9" font-family="var(--font-ui,sans-serif)" fill="${labelCol}" `
              + `font-weight="600">${cbItem.cb}</text>`;
          });
        } catch(_) {}
      });
      _cbOverlay.innerHTML = svgContent;
    });
  }

  async function _applyMarkers() {
    // Clear overlay
    if (_cbOverlay) { _cbOverlay.innerHTML = ''; }
    _cbMeetingData = [];
    window._lwCbMarkerMap = {};
    if (!window._lwShowCb) return;
    try {
      const _CB_MAP = {
        eurusd:['EUR','USD'], gbpusd:['GBP','USD'], usdjpy:['USD','JPY'],
        audusd:['AUD','USD'], usdcad:['USD','CAD'], usdchf:['USD','CHF'],
        nzdusd:['NZD','USD'], eurgbp:['EUR','GBP'], eurjpy:['EUR','JPY'],
        eurchf:['EUR','CHF'], eurcad:['EUR','CAD'], euraud:['EUR','AUD'],
        eurnzd:['EUR','NZD'], gbpjpy:['GBP','JPY'], gbpchf:['GBP','CHF'],
        gbpcad:['GBP','CAD'], gbpaud:['GBP','AUD'], gbpnzd:['GBP','NZD'],
        audjpy:['AUD','JPY'], audnzd:['AUD','NZD'], audchf:['AUD','CHF'],
        audcad:['AUD','CAD'], cadjpy:['CAD','JPY'], cadchf:['CAD','CHF'],
        nzdjpy:['NZD','JPY'], nzdcad:['NZD','CAD'], nzdchf:['NZD','CHF'],
        chfjpy:['CHF','JPY'], gold:['USD'], wti:['USD'], btc:[], us10y:['USD'],
        spx:['USD'], nasdaq:['USD'], dxy:['USD'], nikkei:['JPY'], stoxx:['EUR'],
      };
      const relevantCBs = _CB_MAP[ohlcId] || [];
      if (relevantCBs.length === 0) return;
      const mtgData = window._STATE_meetings || await fetch('./meetings-data/meetings.json')
        .then(r => r.ok ? r.json() : null).catch(() => null);
      if (!mtgData?.meetings) return;
      const barDates = new Set(bars.map(b => b.time));
      const firstDate = bars[0]?.time;
      const lastDate  = bars[bars.length - 1]?.time;
      const _CB_COLORS = { USD:'rgba(79,127,255,0.85)',  EUR:'rgba(246,148,28,0.85)',
                           GBP:'rgba(156,77,255,0.85)',  JPY:'rgba(255,213,0,0.85)',
                           AUD:'rgba(0,188,212,0.85)',   CAD:'rgba(255,87,34,0.85)',
                           CHF:'rgba(156,204,101,0.85)', NZD:'rgba(0,230,118,0.85)' };
      // dateMap: date → [{cb, color}]
      const dateMap = {};
      relevantCBs.forEach(cb => {
        const cbMtg = mtgData.meetings[cb];
        if (!cbMtg?.allMeetings) return;
        const color = _CB_COLORS[cb] || 'rgba(144,150,160,0.8)';
        cbMtg.allMeetings.forEach(dateStr => {
          if (dateStr < firstDate || dateStr > lastDate) return;
          let targetDate = dateStr;
          if (!barDates.has(dateStr)) {
            const d = new Date(dateStr + 'T12:00:00Z');
            d.setDate(d.getDate() + 1);
            const next = d.toISOString().slice(0, 10);
            if (barDates.has(next)) targetDate = next; else return;
          }
          if (!dateMap[targetDate]) dateMap[targetDate] = [];
          // Avoid dupe CBs on same date
          if (!dateMap[targetDate].find(e => e.cb === cb)) {
            dateMap[targetDate].push({ cb, color });
          }
        });
      });
      // Build _cbMeetingData array and tooltip map
      Object.entries(dateMap).sort((a,b) => a[0] < b[0] ? -1 : 1).forEach(([date, cbs]) => {
        _cbMeetingData.push({ date, cbs });
        window._lwCbMarkerMap[date] = cbs.map(e => ({ cb: e.cb, color: e.color }));
      });
      // Create SVG overlay if not already present
      if (!_cbOverlay) {
        _cbOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        _cbOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;overflow:visible;';
        chartDiv.style.position = 'relative';
        chartDiv.appendChild(_cbOverlay);
      }
      // Draw immediately and subscribe to time-range changes for scroll/zoom sync
      _drawCbLines();
      _lwChart.timeScale().subscribeVisibleTimeRangeChange(_drawCbLines);
    } catch(_cbErr) { console.warn('CB markers error:', _cbErr); }
  }
  const _cbBtn = document.getElementById('lw-cb-btn');
  if (_cbBtn) {
    _cbBtn.classList.toggle('on', window._lwShowCb);
    _cbBtn.setAttribute('aria-pressed', window._lwShowCb ? 'true' : 'false');
  }
  _applyMarkers();

  // ── Full Indicator Library — Bloomberg/Eikon/TradingView standard set ───────
  // Indicators are rendered in separate sub-panes (oscillators) or overlaid on
  // the main price pane (overlays). All calculations are deterministic — no
  // Math.random(). State persists across symbol switches via window._lwIndState.

  // ── Shared math helpers ─────────────────────────────────────────────────────

  function _iSMA(src, n) {
    const out = [];
    for (let i = n - 1; i < src.length; i++) {
      let s = 0; for (let j = 0; j < n; j++) s += src[i - j];
      out.push(s / n);
    }
    return out;
  }
  function _iEMA(src, n) {
    const k = 2 / (n + 1); const out = [src[0]];
    for (let i = 1; i < src.length; i++) out.push(src[i] * k + out[i - 1] * (1 - k));
    return out;
  }
  function _iWMA(src, n) {
    const out = []; const denom = n * (n + 1) / 2;
    for (let i = n - 1; i < src.length; i++) {
      let s = 0; for (let j = 0; j < n; j++) s += src[i - j] * (n - j);
      out.push(s / denom);
    }
    return out;
  }
  function _iDEMA(src, n) { // Double EMA
    const e1 = _iEMA(src, n);
    const e2 = _iEMA(e1, n);
    return e1.slice(e1.length - e2.length).map((v, i) => 2 * v - e2[i]);
  }
  function _iTEMA(src, n) { // Triple EMA
    const e1 = _iEMA(src, n);
    const e2 = _iEMA(e1, n);
    const e3 = _iEMA(e2, n);
    const off1 = e1.length - e3.length;
    const off2 = e2.length - e3.length;
    return e3.map((v3, i) => 3 * e1[off1 + i] - 3 * e2[off2 + i] + v3);
  }
  function _iVWMA(bars, n) { // Volume-Weighted MA
    const out = [];
    for (let i = n - 1; i < bars.length; i++) {
      let sumPV = 0, sumV = 0;
      for (let j = 0; j < n; j++) { sumPV += bars[i-j].close * (bars[i-j].volume||1); sumV += (bars[i-j].volume||1); }
      out.push(sumPV / sumV);
    }
    return out;
  }
  // Compute any MA type from closes (and bars for VWMA) — returns raw array
  function _iMA(type, closes, bars, n) {
    switch (type) {
      case 'SMA':  return _iSMA(closes, n);
      case 'EMA':  return _iEMA(closes, n);
      case 'WMA':  return _iWMA(closes, n);
      case 'HMA': { const half=Math.round(n/2),sqrtp=Math.round(Math.sqrt(n));
                    const wH=_iWMA(closes,half),wP=_iWMA(closes,n);
                    const off=n-half;
                    const raw=wH.slice(off).map((v,i)=>2*v-wP[i+off]);
                    return _iWMA(raw,sqrtp); }
      case 'DEMA': return _iDEMA(closes, n);
      case 'TEMA': return _iTEMA(closes, n);
      case 'VWMA': return _iVWMA(bars, n);
      default:     return _iEMA(closes, n);
    }
  }
  function _iStdev(src, n) {
    const out = [];
    for (let i = n - 1; i < src.length; i++) {
      const slice = src.slice(i - n + 1, i + 1);
      const mean = slice.reduce((a, b) => a + b, 0) / n;
      const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
      out.push(Math.sqrt(variance));
    }
    return out;
  }
  function _iRMA(src, n) { // Wilder smoothing (RMA)
    const k = 1 / n; const out = [src[0]];
    for (let i = 1; i < src.length; i++) out.push(src[i] * k + out[i - 1] * (1 - k));
    return out;
  }
  function _iTR(bars) { // True Range
    return bars.map((b, i) => {
      if (i === 0) return b.high - b.low;
      const pc = bars[i - 1].close;
      return Math.max(b.high - b.low, Math.abs(b.high - pc), Math.abs(b.low - pc));
    });
  }
  // Align a calculated array (shorter) to bars — pad = bars.length - arr.length
  // No offset param: the array's own length determines the correct alignment automatically.
  function _iAlign(arr, bars) {
    const pad = bars.length - arr.length;
    return bars.map((b, i) => {
      const v = arr[i - pad];
      return { time: b.time, value: (v != null && !isNaN(v)) ? v : NaN };
    }).filter(d => !isNaN(d.value));
  }
  // Merge two aligned arrays into { time, value } pairs starting at the later offset
  function _iZip(timesA, valA, valB) {
    return timesA.map((t, i) => ({ time: t, value: valB[i] })).filter(d => !isNaN(d.value));
  }

  // ── Indicator definitions catalogue ────────────────────────────────────────
  // Each entry: { id, label, group, desc, defaultParams, type }
  // type: 'overlay' = drawn on main price pane; 'oscillator' = sub-pane below
  // paramDefs: array of { key, label, type:'int'|'float', min, max, step }
  // colors: array of hex colors — one per series returned by _calcIndData
  const _IND_CATALOGUE = [
    // ── Overlays ──────────────────────────────────────────────────────────────
    { id:'ma',       group:'Moving Averages', label:'Moving Average',    desc:'Add configurable MAs (SMA/EMA/WMA/HMA/DEMA/TEMA/VWMA)', type:'overlay',    defaultParams:{},                              paramDefs:[], colors:[] },
    { id:'vwap',     group:'Overlays',        label:'VWAP',              desc:'Volume-Weighted Avg Price (daily sessions)',             type:'overlay',    defaultParams:{},                              paramDefs:[], colors:['#ff5722'] },
    { id:'bb',       group:'Overlays',        label:'Bollinger Bands',   desc:'Bollinger Bands',                                        type:'overlay',    defaultParams:{ period:20, mult:2 },           paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1},{key:'mult',label:'Mult',type:'float',min:0.1,max:10,step:0.1}], colors:['rgba(33,150,243,0.5)','rgba(33,150,243,0.9)','rgba(33,150,243,0.9)'] },
    { id:'keltner',  group:'Overlays',        label:'Keltner Channel',   desc:'Keltner Channel',                                        type:'overlay',    defaultParams:{ period:20, mult:1.5 },         paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1},{key:'mult',label:'Mult',type:'float',min:0.1,max:10,step:0.1}], colors:['rgba(255,152,0,0.5)','rgba(255,152,0,0.9)','rgba(255,152,0,0.9)'] },
    { id:'donchian', group:'Overlays',        label:'Donchian Channel',  desc:'Donchian Channel',                                       type:'overlay',    defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1}], colors:['rgba(156,39,176,0.8)','rgba(156,39,176,0.8)','rgba(156,39,176,0.4)'] },
    { id:'psar',     group:'Overlays',        label:'Parabolic SAR',     desc:'Parabolic SAR',                                          type:'overlay',    defaultParams:{ step:0.02, max:0.2 },          paramDefs:[{key:'step',label:'Step',type:'float',min:0.001,max:0.1,step:0.001},{key:'max',label:'Max AF',type:'float',min:0.01,max:0.5,step:0.01}], colors:['#f44336'] },
    { id:'ichimoku', group:'Overlays',        label:'Ichimoku Cloud',    desc:'Ichimoku Kinko Hyo · 9/26/52',                          type:'overlay',    defaultParams:{},                              paramDefs:[], colors:['#26a69a','#ef5350','rgba(38,166,154,0.3)','rgba(239,83,80,0.3)','rgba(120,123,134,0.4)'] },
    // ── Oscillators ───────────────────────────────────────────────────────────
    { id:'rsi',      group:'Oscillators',     label:'RSI',               desc:'Relative Strength Index',                                type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#9c27b0'] },
    { id:'stoch',    group:'Oscillators',     label:'Stochastic',        desc:'Stochastic Oscillator',                                  type:'oscillator', defaultParams:{ k:14, d:3, smooth:3 },         paramDefs:[{key:'k',label:'%K',type:'int',min:1,max:100,step:1},{key:'smooth',label:'Smooth',type:'int',min:1,max:20,step:1},{key:'d',label:'%D',type:'int',min:1,max:20,step:1}], colors:['#2196f3','#ff9800'] },
    { id:'macd',     group:'Oscillators',     label:'MACD',              desc:'MACD',                                                   type:'oscillator', defaultParams:{ fast:12, slow:26, signal:9 },  paramDefs:[{key:'fast',label:'Fast',type:'int',min:2,max:100,step:1},{key:'slow',label:'Slow',type:'int',min:2,max:200,step:1},{key:'signal',label:'Signal',type:'int',min:1,max:50,step:1}], colors:['#26a69a','#2196f3','#ff9800'], histoIdx:[0] },
    { id:'cci',      group:'Oscillators',     label:'CCI',               desc:'Commodity Channel Index',                                type:'oscillator', defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#00bcd4'] },
    { id:'willr',    group:'Oscillators',     label:'Williams %R',       desc:'Williams %R',                                            type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#ff5722'] },
    { id:'roc',      group:'Oscillators',     label:'ROC',               desc:'Rate of Change',                                         type:'oscillator', defaultParams:{ period:12 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#4caf50'] },
    { id:'mom',      group:'Oscillators',     label:'Momentum',          desc:'Momentum',                                               type:'oscillator', defaultParams:{ period:10 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#9c27b0'] },
    { id:'mfi',      group:'Oscillators',     label:'MFI',               desc:'Money Flow Index (uses volume)',                         type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#03a9f4'] },
    { id:'ao',       group:'Oscillators',     label:'Awesome Oscillator',desc:'Awesome Oscillator · 5/34',                              type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#26a69a'], histoIdx:[0] },
    { id:'trix',     group:'Oscillators',     label:'TRIX',              desc:'Triple Smoothed EMA',                                    type:'oscillator', defaultParams:{ period:18 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#673ab7'] },
    { id:'dpo',      group:'Oscillators',     label:'DPO',               desc:'Detrended Price Oscillator',                             type:'oscillator', defaultParams:{ period:21 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#ff9800'] },
    { id:'uo',       group:'Oscillators',     label:'Ultimate Osc.',     desc:'Ultimate Oscillator · 7/14/28',                          type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#8bc34a'] },
    // ── Volatility ────────────────────────────────────────────────────────────
    { id:'atr',      group:'Volatility',      label:'ATR',               desc:'Average True Range',                                     type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#ff9800'] },
    { id:'adx',      group:'Volatility',      label:'ADX / DMI',         desc:'Average Directional Index + DI±',                        type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#f44336','#26a69a','#ef5350'] },
    { id:'aroon',    group:'Volatility',      label:'Aroon',             desc:'Aroon Up/Down',                                          type:'oscillator', defaultParams:{ period:25 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#26a69a','#ef5350'] },
    { id:'chop',     group:'Volatility',      label:'Choppiness',        desc:'Choppiness Index',                                       type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#607d8b'] },
    // ── Volume ────────────────────────────────────────────────────────────────
    { id:'obv',      group:'Volume',          label:'OBV',               desc:'On-Balance Volume',                                      type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#3f51b5'] },
    { id:'cmf',      group:'Volume',          label:'CMF',               desc:'Chaikin Money Flow',                                     type:'oscillator', defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#00acc1'] },
  ];

  // ── Active indicator state (persists across symbol switches) ─────────────────
  // ── Persistent state — survives page reloads via localStorage ────────────────
  const _LS_IND   = 'gi_ind_state';   // { id: bool }
  const _LS_PARAMS = 'gi_ind_params'; // { id: { param: val } }
  const _LS_MA    = 'gi_ma_list';     // [ { uid, type, period, color, lineWidth, lineStyle } ]

  const _DEFAULT_MA_LIST = [
    { uid:'ma_sma20',  type:'SMA', period:20,  color:'#2196f3', lineWidth:1, lineStyle:0 },
  ];

  function _lsGet(key, fallback) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch(_) { return fallback; }
  }
  function _lsSet(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch(_) {}
  }

  // Load persisted state (first run uses defaults)
  if (typeof window._lwIndState  === 'undefined') window._lwIndState  = _lsGet(_LS_IND,    {});
  if (typeof window._lwIndParams === 'undefined') window._lwIndParams = _lsGet(_LS_PARAMS,  {});
  if (typeof window._lwMaList    === 'undefined') window._lwMaList    = _lsGet(_LS_MA,      _DEFAULT_MA_LIST);

  // Save helpers — call after any mutation
  function _saveIndState()  { _lsSet(_LS_IND,    window._lwIndState);  }
  function _saveIndParams() { _lsSet(_LS_PARAMS,  window._lwIndParams); }
  function _saveMaList()    { _lsSet(_LS_MA,      window._lwMaList);    }

  const _maSeries = {}; // uid → series object
  // Active pane indices — keyed by indicator id, reset each render (chart destroyed)
  const _indPaneIndex = {}; // id → pane index number (oscillators only)
  window._indSeries = {}; const _indSeries = window._indSeries;
  const _indRefSeries = {}; // paneIndex → array of ref-line series

  // Get effective params for an indicator (custom overrides defaultParams)
  function _iP(id) {
    const cfg = _IND_CATALOGUE.find(c => c.id === id);
    return Object.assign({}, cfg?.defaultParams || {}, window._lwIndParams[id] || {});
  }
  // Get effective color for indicator id, series index i
  function _iC(id, i) {
    const cfg = _IND_CATALOGUE.find(c => c.id === id);
    const defaults = cfg?.colors || [];
    const custom   = (window._lwIndParams[id] || {}).colors || [];
    return custom[i] || defaults[i] || '#787b86';
  }

  // ── Calculation functions — one per indicator id ───────────────────────────

  function _calcIndData(id, bars) {
    const closes = bars.map(b => b.close);
    const highs  = bars.map(b => b.high);
    const lows   = bars.map(b => b.low);
    const vols   = bars.map(b => b.volume || 0);
    const p      = _iP(id); // effective params (defaults + user overrides)

    switch (id) {
      case 'ma': {
        // MA indicator now renders via _buildMaSeries, not _calcIndData
        // Return an empty stub so _buildIndicatorPane doesn't fail
        return [];
      }
      case 'vwap': {
        const typicals = bars.map((b, i) => ({ t: b.time, tp: (b.high+b.low+b.close)/3, v: vols[i] }));
        let cumTPV = 0, cumV = 0;
        const data = typicals.map(({ t, tp, v }) => { cumTPV += tp*v; cumV += v; return { time:t, value: cumV>0 ? cumTPV/cumV : tp }; });
        return [{ data, color:_iC(id,0), lineWidth:1, label:'VWAP', dashed:true }];
      }
      case 'bb': {
        const { period:n, mult } = p;
        const sma   = _iSMA(closes, n);
        const stdev = _iStdev(closes, n);
        const mid   = _iAlign(sma, bars);
        const upper = _iAlign(sma.map((v,i) => v + mult * stdev[i]), bars);
        const lower = _iAlign(sma.map((v,i) => v - mult * stdev[i]), bars);
        return [
          { data: mid,   color:_iC(id,0), lineWidth:1, label:`BB(${n}) Mid` },
          { data: upper, color:_iC(id,1), lineWidth:1, label:`+${mult}σ` },
          { data: lower, color:_iC(id,2), lineWidth:1, label:`-${mult}σ` },
        ];
      }
      case 'keltner': {
        const { period:n, mult } = p;
        const ema   = _iEMA(closes, n);
        const tr    = _iTR(bars);
        const atr   = _iRMA(tr, n);
        const upper = ema.map((v,i) => v + mult * atr[i]);
        const lower = ema.map((v,i) => v - mult * atr[i]);
        return [
          { data: _iAlign(ema,   bars), color:_iC(id,0), lineWidth:1, label:`KC(${n}) Mid` },
          { data: _iAlign(upper, bars), color:_iC(id,1), lineWidth:1, label:`+${mult}×ATR` },
          { data: _iAlign(lower, bars), color:_iC(id,2), lineWidth:1, label:`-${mult}×ATR` },
        ];
      }
      case 'donchian': {
        const n = p.period;
        const upper = [], lower = [], mid = [];
        for (let i = n-1; i < bars.length; i++) {
          const sl = bars.slice(i-n+1, i+1);
          const h = Math.max(...sl.map(b=>b.high)), l = Math.min(...sl.map(b=>b.low));
          upper.push({ time:bars[i].time, value:h });
          lower.push({ time:bars[i].time, value:l });
          mid.push(  { time:bars[i].time, value:(h+l)/2 });
        }
        return [
          { data:upper, color:_iC(id,0), lineWidth:1, label:`DC(${n}) Upper` },
          { data:lower, color:_iC(id,1), lineWidth:1, label:`DC Lower` },
          { data:mid,   color:_iC(id,2), lineWidth:1, label:`DC Mid`, dashed:true },
        ];
      }
      case 'psar': {
        const { step, max:maxAF } = p;
        let bull=true, ep=bars[0].high, af=step, sar=bars[0].low;
        const data=[];
        for(let i=1;i<bars.length;i++){
          const prev=bars[i-1];
          sar = sar + af*(ep-sar);
          if(bull){
            if(bars[i].low<sar){bull=false;sar=ep;ep=bars[i].low;af=step;}
            else{if(bars[i].high>ep){ep=bars[i].high;af=Math.min(af+step,maxAF);}}
            sar=Math.min(sar,prev.low,bars[Math.max(0,i-2)].low);
          } else {
            if(bars[i].high>sar){bull=true;sar=ep;ep=bars[i].high;af=step;}
            else{if(bars[i].low<ep){ep=bars[i].low;af=Math.min(af+step,maxAF);}}
            sar=Math.max(sar,prev.high,bars[Math.max(0,i-2)].high);
          }
          data.push({time:bars[i].time,value:parseFloat(sar.toFixed(dec))});
        }
        return [{ data, color:_iC(id,0), lineWidth:0, label:'PSAR', markers:true }];
      }
      case 'ichimoku': {
        function tenkan(i,n){const s=bars.slice(Math.max(0,i-n+1),i+1);return(Math.max(...s.map(b=>b.high))+Math.min(...s.map(b=>b.low)))/2;}
        const TK=9,KJ=26,SB2=52,DISP=26;
        const tLine=[],kLine=[],sa=[],sb=[],cl=[];
        for(let i=0;i<bars.length;i++){
          const tk=tenkan(i,TK),kj=tenkan(i,KJ);
          if(i>=TK-1) tLine.push({time:bars[i].time,value:tk});
          if(i>=KJ-1){
            kLine.push({time:bars[i].time,value:kj});
            if(i+DISP<bars.length) sa.push({time:bars[i+DISP].time,value:(tk+kj)/2});
          }
          if(i>=SB2-1&&i+DISP<bars.length) sb.push({time:bars[i+DISP].time,value:tenkan(i,SB2)});
          if(i>=KJ-1&&i>DISP) cl.push({time:bars[i-DISP].time,value:bars[i].close});
        }
        return [
          { data:tLine, color:_iC(id,0), lineWidth:1, label:'Tenkan' },
          { data:kLine, color:_iC(id,1), lineWidth:1, label:'Kijun' },
          { data:sa,    color:_iC(id,2), lineWidth:1, label:'Span A' },
          { data:sb,    color:_iC(id,3), lineWidth:1, label:'Span B' },
          { data:cl,    color:_iC(id,4), lineWidth:1, label:'Chikou', dashed:true },
        ];
      }
      // ── Oscillators ─────────────────────────────────────────────────────────
      case 'rsi': {
        const n = p.period;
        const gains=[], losses=[];
        for(let i=1;i<closes.length;i++){const d=closes[i]-closes[i-1];gains.push(d>0?d:0);losses.push(d<0?-d:0);}
        const avgG=_iRMA(gains,n), avgL=_iRMA(losses,n);
        const data=avgG.map((g,i)=>{const l=avgL[i];const rs=l===0?Infinity:g/l;return{time:bars[i+1].time,value:parseFloat((l===0?100:100-100/(1+rs)).toFixed(2))};});
        return [{data,color:_iC(id,0),lineWidth:1,label:`RSI(${n})`,
          refs:[{v:30,color:'rgba(239,83,80,0.3)'},{v:50,color:'rgba(120,123,134,0.2)'},{v:70,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'stoch': {
        const { k:kPer, d:dPer, smooth } = p;
        const rawK=[];
        for(let i=kPer-1;i<bars.length;i++){
          const s=bars.slice(i-kPer+1,i+1);
          const h=Math.max(...s.map(b=>b.high)),l=Math.min(...s.map(b=>b.low));
          rawK.push(h===l?50:((bars[i].close-l)/(h-l))*100);
        }
        const sK=_iSMA(rawK,smooth), sD=_iSMA(sK,dPer);
        const off=bars.length-rawK.length;
        const kData=sK.map((v,i)=>({time:bars[off+i+smooth-1].time,value:parseFloat(v.toFixed(2))}));
        const dData=sD.map((v,i)=>({time:bars[off+i+smooth-1+dPer-1].time,value:parseFloat(v.toFixed(2))}));
        return [
          {data:kData,color:_iC(id,0),lineWidth:1,label:`%K(${kPer},${smooth})`,refs:[{v:20,color:'rgba(239,83,80,0.3)'},{v:50,color:'rgba(120,123,134,0.2)'},{v:80,color:'rgba(239,83,80,0.3)'}]},
          {data:dData,color:_iC(id,1),lineWidth:1,label:`%D(${dPer})`},
        ];
      }
      case 'macd': {
        const { fast, slow, signal:sig } = p;
        const ef=_iEMA(closes,fast), es=_iEMA(closes,slow);
        const ml=ef.map((v,i)=>v-es[i]);
        // sl2 = EMA of MACD line starting from bar (slow-1).
        // sl2[j] corresponds to bars index (slow-1+j), so for bar i use sl2[si] where si=i-(slow-1).
        const sl2=_iEMA(ml.slice(slow-1),sig);
        const offset=slow-1+sig-1;
        const macdD=[],sigD=[],histD=[];
        for(let i=offset;i<bars.length;i++){
          const si=i-(slow-1); // sl2 index aligned to bar i
          const m=ml[i],s=sl2[si],h=m-s;
          macdD.push({time:bars[i].time,value:parseFloat(m.toFixed(6))});
          sigD.push( {time:bars[i].time,value:parseFloat(s.toFixed(6))});
          const hBase=_iC(id,0);histD.push({time:bars[i].time,value:parseFloat(h.toFixed(6)),color:h>=0?hBase:'rgba(239,83,80,0.7)'});
        }
        return [
          {data:histD,color:_iC(id,0),lineWidth:0,label:'Hist',histogram:true,refs:[{v:0,color:'rgba(120,123,134,0.2)'}]},
          {data:macdD,color:_iC(id,1),lineWidth:1,label:`MACD(${fast},${slow})`},
          {data:sigD, color:_iC(id,2),lineWidth:1,label:`Sig(${sig})`},
        ];
      }
      case 'cci': {
        const n = p.period;
        const tp=bars.map(b=>(b.high+b.low+b.close)/3);
        const sma=_iSMA(tp,n);
        const data=sma.map((avg,i)=>{
          const slice=tp.slice(i,i+n);
          const meanDev=slice.reduce((s,v)=>s+Math.abs(v-avg),0)/n;
          return{time:bars[i+n-1].time,value:parseFloat((meanDev===0?0:(tp[i+n-1]-avg)/(0.015*meanDev)).toFixed(2))};
        });
        return [{data,color:_iC(id,0),lineWidth:1,label:`CCI(${n})`,
          refs:[{v:-100,color:'rgba(239,83,80,0.3)'},{v:0,color:'rgba(120,123,134,0.2)'},{v:100,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'willr': {
        const n = p.period;
        const data=[];
        for(let i=n-1;i<bars.length;i++){
          const sl=bars.slice(i-n+1,i+1);
          const h=Math.max(...sl.map(b=>b.high)),l=Math.min(...sl.map(b=>b.low));
          data.push({time:bars[i].time,value:parseFloat((h===l?-50:((h-bars[i].close)/(h-l))*-100).toFixed(2))});
        }
        return [{data,color:_iC(id,0),lineWidth:1,label:`%R(${n})`,
          refs:[{v:-80,color:'rgba(239,83,80,0.3)'},{v:-50,color:'rgba(120,123,134,0.2)'},{v:-20,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'roc': {
        const n = p.period;
        const data=bars.slice(n).map((b,i)=>({time:b.time,value:parseFloat(((b.close-bars[i].close)/bars[i].close*100).toFixed(4))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`ROC(${n})`,refs:[{v:0,color:'rgba(120,123,134,0.3)'}]}];
      }
      case 'mom': {
        const n = p.period;
        const data=bars.slice(n).map((b,i)=>({time:b.time,value:parseFloat((b.close-bars[i].close).toFixed(dec))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`Mom(${n})`,refs:[{v:0,color:'rgba(120,123,134,0.3)'}]}];
      }
      case 'mfi': {
        const n = p.period;
        const data=[];
        for(let i=n;i<bars.length;i++){
          let pmf=0,nmf=0;
          for(let j=i-n+1;j<=i;j++){
            const tp=(bars[j].high+bars[j].low+bars[j].close)/3;
            const prevTp=(bars[j-1].high+bars[j-1].low+bars[j-1].close)/3;
            const mf=tp*(bars[j].volume||1);
            if(tp>prevTp) pmf+=mf; else nmf+=mf;
          }
          data.push({time:bars[i].time,value:parseFloat((nmf===0?100:100-100/(1+pmf/nmf)).toFixed(2))});
        }
        return [{data,color:_iC(id,0),lineWidth:1,label:`MFI(${n})`,
          refs:[{v:20,color:'rgba(239,83,80,0.3)'},{v:50,color:'rgba(120,123,134,0.2)'},{v:80,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'ao': {
        const midAO=bars.map(b=>(b.high+b.low)/2);
        const s5=_iSMA(midAO,5),s34=_iSMA(midAO,34);
        const off=34-1;
        const data=s34.map((v,i)=>{
          const ao=s5[i+(34-5)]-v;
          const prev=i>0?s5[i+(34-5)-1]-s34[i-1]:ao;
          const aoBase=_iC(id,0);return{time:bars[off+i].time,value:parseFloat(ao.toFixed(6)),color:ao>=prev?aoBase:'rgba(239,83,80,0.7)'};
        });
        return [{data,color:_iC(id,0),lineWidth:0,label:'AO',histogram:true,refs:[{v:0,color:'rgba(120,123,134,0.2)'}]}];
      }
      case 'trix': {
        const n = p.period;
        const e1=_iEMA(closes,n),e2=_iEMA(e1,n),e3=_iEMA(e2,n);
        const data=e3.slice(1).map((v,i)=>({time:bars[bars.length-e3.length+i+1].time,value:parseFloat(((v-e3[i])/e3[i]*100).toFixed(6))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`TRIX(${n})`,refs:[{v:0,color:'rgba(120,123,134,0.3)'}]}];
      }
      case 'dpo': {
        const n = p.period; const disp=Math.floor(n/2)+1;
        const sma=_iSMA(closes,n);
        const data=sma.map((v,i)=>{
          const barIdx=i+n-1-disp;
          if(barIdx<0) return null;
          return{time:bars[i+n-1].time,value:parseFloat((closes[barIdx]-v).toFixed(dec))};
        }).filter(Boolean);
        return [{data,color:_iC(id,0),lineWidth:1,label:`DPO(${n})`,refs:[{v:0,color:'rgba(120,123,134,0.3)'}]}];
      }
      case 'uo': {
        const data=[];
        for(let i=28;i<bars.length;i++){
          function _uoBP(j){return bars[j].close-Math.min(bars[j].low,bars[j-1].close);}
          function _uoTR(j){return Math.max(bars[j].high,bars[j-1].close)-Math.min(bars[j].low,bars[j-1].close);}
          let [bp7,tr7,bp14,tr14,bp28,tr28]=[0,0,0,0,0,0];
          for(let j=i-6;j<=i;j++){bp7+=_uoBP(j);tr7+=_uoTR(j);}
          for(let j=i-13;j<=i;j++){bp14+=_uoBP(j);tr14+=_uoTR(j);}
          for(let j=i-27;j<=i;j++){bp28+=_uoBP(j);tr28+=_uoTR(j);}
          data.push({time:bars[i].time,value:parseFloat((100*(4*(bp7/tr7)+2*(bp14/tr14)+(bp28/tr28))/7).toFixed(2))});
        }
        return [{data,color:_iC(id,0),lineWidth:1,label:'UO(7,14,28)',
          refs:[{v:30,color:'rgba(239,83,80,0.3)'},{v:50,color:'rgba(120,123,134,0.2)'},{v:70,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'atr': {
        const n = p.period;
        const tr=_iTR(bars);
        const atr=_iRMA(tr,n);
        return [{data:bars.map((b,i)=>({time:b.time,value:parseFloat(atr[i].toFixed(dec))})),color:_iC(id,0),lineWidth:1,label:`ATR(${n})`}];
      }
      case 'adx': {
        const n = p.period;
        const plusDM=[],minusDM=[],tr=_iTR(bars);
        for(let i=1;i<bars.length;i++){
          const upMove=bars[i].high-bars[i-1].high,downMove=bars[i-1].low-bars[i].low;
          plusDM.push(upMove>downMove&&upMove>0?upMove:0);
          minusDM.push(downMove>upMove&&downMove>0?downMove:0);
        }
        const atr=_iRMA(tr.slice(1),n);
        const pDI=_iRMA(plusDM,n).map((v,i)=>100*v/atr[i]);
        const mDI=_iRMA(minusDM,n).map((v,i)=>100*v/atr[i]);
        const dx=pDI.map((p,i)=>{const s=p+mDI[i];return s===0?0:100*Math.abs(p-mDI[i])/s;});
        const adx=_iRMA(dx,n);
        const off=bars.length-adx.length;
        return [
          {data:adx.map((v,i)=>({time:bars[off+i].time,value:parseFloat(v.toFixed(2))})),color:_iC(id,0),lineWidth:1,label:`ADX(${n})`,refs:[{v:25,color:'rgba(239,83,80,0.3)'}]},
          {data:pDI.map((v,i)=>({time:bars[off+i].time,value:parseFloat(v.toFixed(2))})),color:_iC(id,1),lineWidth:1,label:'+DI'},
          {data:mDI.map((v,i)=>({time:bars[off+i].time,value:parseFloat(v.toFixed(2))})),color:_iC(id,2),lineWidth:1,label:'-DI'},
        ];
      }
      case 'aroon': {
        const n = p.period;
        const up=[],dn=[];
        for(let i=n;i<bars.length;i++){
          const sl=bars.slice(i-n,i+1);
          const hiIdx=sl.reduce((mi,b,j)=>b.high>sl[mi].high?j:mi,0);
          const loIdx=sl.reduce((mi,b,j)=>b.low<sl[mi].low?j:mi,0);
          up.push({time:bars[i].time,value:parseFloat(((hiIdx/n)*100).toFixed(2))});
          dn.push({time:bars[i].time,value:parseFloat(((loIdx/n)*100).toFixed(2))});
        }
        return [
          {data:up,color:_iC(id,0),lineWidth:1,label:`Aroon Up(${n})`,refs:[{v:50,color:'rgba(120,123,134,0.2)'}]},
          {data:dn,color:_iC(id,1),lineWidth:1,label:`Aroon Down`},
        ];
      }
      case 'chop': {
        const n = p.period;
        const tr=_iTR(bars);
        const data=[];
        for(let i=n-1;i<bars.length;i++){
          const atrSum=tr.slice(i-n+1,i+1).reduce((s,v)=>s+v,0);
          const sl=bars.slice(i-n+1,i+1);
          const hl=Math.max(...sl.map(b=>b.high))-Math.min(...sl.map(b=>b.low));
          data.push({time:bars[i].time,value:parseFloat((hl===0?100:(100*Math.log10(atrSum/hl)/Math.log10(n))).toFixed(2))});
        }
        return [{data,color:_iC(id,0),lineWidth:1,label:`Chop(${n})`,
          refs:[{v:38.2,color:'rgba(38,166,154,0.3)'},{v:61.8,color:'rgba(239,83,80,0.3)'}]}];
      }
      case 'obv': {
        let obv=0;
        const data=bars.map((b,i)=>{
          if(i>0){if(b.close>bars[i-1].close)obv+=b.volume||0;else if(b.close<bars[i-1].close)obv-=b.volume||0;}
          return{time:b.time,value:obv};
        });
        return [{data,color:_iC(id,0),lineWidth:1,label:'OBV'}];
      }
      case 'cmf': {
        const n = p.period;
        const mfv=bars.map(b=>{const hl=b.high-b.low;return hl===0?0:((b.close-b.low)-(b.high-b.close))/hl*(b.volume||0);});
        const data=[];
        for(let i=n-1;i<bars.length;i++){
          const volSum=bars.slice(i-n+1,i+1).reduce((s,b)=>s+(b.volume||0),0);
          const mfvSum=mfv.slice(i-n+1,i+1).reduce((s,v)=>s+v,0);
          data.push({time:bars[i].time,value:parseFloat((volSum===0?0:mfvSum/volSum).toFixed(4))});
        }
        return [{data,color:_iC(id,0),lineWidth:1,label:`CMF(${n})`,refs:[{v:0,color:'rgba(120,123,134,0.3)'}]}];
      }
      default: return [];
    }
  }

  // ── Pane / series rendering helpers ─────────────────────────────────────────

  function _addPaneLegend(paneEl, id, html) {
    if (!paneEl) return;
    paneEl.style.position = 'relative';
    const el = document.createElement('div');
    el.id = id;
    el.style.cssText = 'position:absolute;top:4px;left:8px;z-index:3;pointer-events:none;'
      + 'font-size:10px;font-family:var(--font-mono,monospace);line-height:1.3;user-select:none;color:#d1d4dc;';
    el.innerHTML = html;
    paneEl.appendChild(el);
  }

  function _addRefLines(paneIndex, refs, barData, n) {
    if (!refs || paneIndex === null) return;
    refs.forEach(ref => {
      try {
        const s = _lwChart.addSeries(LWC.LineSeries, {
          color: ref.color, lineWidth: 1, lineStyle: 2,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        }, paneIndex);
        s.setData(barData.map(b => ({ time: b.time, value: ref.v })).slice(-n));
        // Track ref-line series so they are removed when the indicator is destroyed
        if (!_indRefSeries[paneIndex]) _indRefSeries[paneIndex] = [];
        _indRefSeries[paneIndex].push(s);
      } catch(_) {}
    });
  }

  // ── MA series management ─────────────────────────────────────────────────────
  function _calcMaData(cfg) {
    if (!bars || bars.length < 2) return [];
    const closes = bars.map(b => b.close);
    const raw = _iMA(cfg.type, closes, bars, cfg.period);
    return _iAlign(raw, bars);
  }
  function _buildMaSeries(cfg) {
    _destroyMaSeries(cfg.uid);
    if (!_lwChart) return;
    try {
      const s = _lwChart.addSeries(LWC.LineSeries, {
        color: cfg.color, lineWidth: cfg.lineWidth || 1, lineStyle: cfg.lineStyle || 0,
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        priceFormat: { type: 'price', precision: dec, minMove },
      }, 0);
      s.setData(_calcMaData(cfg));
      _maSeries[cfg.uid] = s;
    } catch(e) { console.warn('MA series error', e); }
  }
  function _destroyMaSeries(uid) {
    if (_maSeries[uid]) { try { _lwChart.removeSeries(_maSeries[uid]); } catch(_) {} _maSeries[uid] = null; }
  }
  function _buildAllMaSeries() {
    Object.keys(_maSeries).forEach(uid => {
      if (!window._lwMaList.find(m => m.uid === uid)) _destroyMaSeries(uid);
    });
    window._lwMaList.forEach(cfg => _buildMaSeries(cfg));
  }
  function _genMaUid() { return 'ma_' + Date.now() + '_' + Math.floor(Math.random()*1000); }

  function _buildIndicatorPane(id) {
    const cfg = _IND_CATALOGUE.find(c => c.id === id);
    if (!cfg || !window._lwIndState[id]) return;

    // Destroy old series for this indicator first
    _destroyIndicatorPane(id);

    try {
      const seriesList = _calcIndData(id, bars);
      if (!seriesList || seriesList.length === 0) return;

      const isOverlay = cfg.type === 'overlay';
      let paneIndex;

      if (isOverlay) {
        paneIndex = 0; // main price pane
      } else {
        // LWC v5: addSeries with paneIndex >= current pane count auto-creates a new pane
        paneIndex = _lwChart.panes().length;
        _indPaneIndex[id] = paneIndex;
      }

      _indSeries[id] = [];

      seriesList.forEach((s, si) => {
        try {
          let series;
          if (s.histogram) {
            series = _lwChart.addSeries(LWC.HistogramSeries, {
              priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
              priceFormat: { type: 'price', precision: 5, minMove: 0.00001 },
            }, paneIndex);
          } else if (s.markers) {
            // Point series (e.g. PSAR) — LineSeries with lineWidth:0
            series = _lwChart.addSeries(LWC.LineSeries, {
              color: s.color, lineWidth: 0,
              priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: true,
              priceFormat: { type: 'price', precision: dec, minMove },
            }, paneIndex);
          } else {
            series = _lwChart.addSeries(LWC.LineSeries, {
              color: s.color, lineWidth: s.lineWidth || 1,
              lineStyle: s.dashed ? 2 : 0,
              priceLineVisible: false, lastValueVisible: si === 0, crosshairMarkerVisible: false,
              priceFormat: { type: 'price', precision: (isOverlay ? dec : 2), minMove: (isOverlay ? minMove : 0.01) },
            }, paneIndex);
          }
          series.setData(s.data);
          _indSeries[id].push(series);

          // Set oscillator pane height after first series is added (triggers pane creation)
          if (!isOverlay && si === 0) {
            try {
              const paneH = (id === 'macd' || id === 'adx') ? 90 : 80;
              _lwChart.panes()[paneIndex]?.setHeight(paneH);
            } catch(_) {}
          }

          // Reference lines — only for first series in a sub-pane
          if (!isOverlay && si === 0 && s.refs) {
            _addRefLines(paneIndex, s.refs, bars, s.data.length + 10);
          }
        } catch(serErr) { console.warn('[LW] series error for', id, serErr); }
      });

      // Pane legend for oscillators
      if (!isOverlay && _indPaneIndex[id] != null) {
        try {
          const paneEl = _lwChart.panes()[_indPaneIndex[id]]?.getHTMLElement();
          if (paneEl) {
            const labelHtml = seriesList.map(s => `<span style="color:${s.color}">${s.label}</span>`).join(' ');
            _addPaneLegend(paneEl, '_lw-ind-legend-' + id, labelHtml);
          }
        } catch(_) {}
      }
    } catch(e) { console.warn('[LW] indicator build error for', id, e); }
  }

  function _destroyIndicatorPane(id) {
    const cfg = _IND_CATALOGUE.find(c => c.id === id);
    const isOverlay = cfg && cfg.type === 'overlay';

    // Remove all series for this indicator (works for both overlays and oscillators)
    if (_indSeries[id]) {
      _indSeries[id].forEach(s => {
        try { _lwChart.removeSeries(s); } catch(_) {}
      });
      _indSeries[id] = null;
    }

    // Remove ref lines for oscillator panes
    if (!isOverlay && _indPaneIndex[id] != null) {
      const refs = _indRefSeries[_indPaneIndex[id]];
      if (refs) {
        refs.forEach(s => { try { _lwChart.removeSeries(s); } catch(_) {} });
        _indRefSeries[_indPaneIndex[id]] = null;
      }
      // Remove the pane itself if it still exists and is empty
      try {
        const panes = _lwChart.panes();
        const pane = panes[_indPaneIndex[id]];
        if (pane && pane.getSeries().length === 0) {
          _lwChart.removePane(_indPaneIndex[id]);
        }
      } catch(_) {}
      _indPaneIndex[id] = null;
    }
  }

  // Build all currently-active indicators on this chart render
  _IND_CATALOGUE.forEach(cfg => {
    if (window._lwIndState[cfg.id]) _buildIndicatorPane(cfg.id);
  });
  // Build all active MA series
  _buildAllMaSeries();

  // ── Active pills bar — shows which indicators are on, with × to remove ──────
  function _renderIndPills() {
    const pillBar = document.getElementById('lw-ind-pills');
    if (!pillBar) return;
    pillBar.innerHTML = '';
    // MA pills
    window._lwMaList.forEach(ma => {
      const pill = document.createElement('span');
      pill.style.cssText = 'display:inline-flex;align-items:center;gap:3px;background:#1e222d;border:1px solid #2a2e39;border-radius:3px;padding:1px 5px;font-size:9px;font-family:var(--font-ui,sans-serif);white-space:nowrap;';
      const dot = `<span style="width:6px;height:6px;border-radius:50%;background:${ma.color};display:inline-block;flex-shrink:0"></span>`;
      pill.innerHTML = `${dot}<span style="color:#787b86">${ma.type} ${ma.period}</span>`;
      const rm = document.createElement('span');
      rm.textContent = '\u00d7';
      rm.style.cssText = 'color:#4a5060;cursor:pointer;font-size:10px;margin-left:1px;';
      rm.title = `Remove ${ma.type} ${ma.period}`;
      rm.addEventListener('click', e => {
        e.stopPropagation();
        window._lwMaList = window._lwMaList.filter(m => m.uid !== ma.uid);
        _saveMaList();
        _destroyMaSeries(ma.uid);
        _renderIndPills();
        _updateIndBtn();
      });
      pill.appendChild(rm);
      pillBar.appendChild(pill);
    });
    // Other indicator pills
    _IND_CATALOGUE.filter(c => c.id !== 'ma' && window._lwIndState[c.id]).forEach(cfg => {
      const pill = document.createElement('span');
      pill.style.cssText = 'display:inline-flex;align-items:center;gap:3px;background:#1e222d;border:1px solid #2a2e39;border-radius:3px;padding:1px 5px;font-size:9px;font-family:var(--font-ui,sans-serif);white-space:nowrap;';
      pill.innerHTML = `<span style="color:#787b86">${cfg.label}</span>`;
      const rm = document.createElement('span');
      rm.textContent = '\u00d7';
      rm.style.cssText = 'color:#4a5060;cursor:pointer;font-size:10px;margin-left:1px;';
      rm.title = 'Remove ' + cfg.label;
      rm.addEventListener('click', e => {
        e.stopPropagation();
        window._lwIndState[cfg.id] = false;
        _saveIndState();
        _destroyIndicatorPane(cfg.id);
        _renderIndPills();
        _updateIndBtn();
      });
      pill.appendChild(rm);
      pillBar.appendChild(pill);
    });
  }

  function _updateIndBtn() {
    const btn = document.getElementById('lw-ind-btn');
    if (!btn) return;
    const anyOn = window._lwMaList.length > 0 || _IND_CATALOGUE.some(c => c.id !== 'ma' && window._lwIndState[c.id]);
    btn.classList.toggle('on', anyOn);
  }

  // ── Indicators dropdown menu ─────────────────────────────────────────────────
  let _indDropdownOpen = false;

  function _closeIndDropdown() {
    const pop = document.getElementById('_lw-ind-dropdown');
    if (pop) pop.remove();
    _indDropdownOpen = false;
    const btn = document.getElementById('lw-ind-btn');
    if (btn) btn.setAttribute('aria-expanded', 'false');
  }

  function _openIndDropdown() {
    if (_indDropdownOpen) { _closeIndDropdown(); return; }
    _closeIndDropdown();
    _indDropdownOpen = true;

    const btn = document.getElementById('lw-ind-btn');
    if (btn) btn.setAttribute('aria-expanded', 'true');

    const pop = document.createElement('div');
    pop.id = '_lw-ind-dropdown';
    pop.style.cssText = [
      'position:fixed;z-index:9999;background:#1a1d29;border:1px solid #2a2e39;',
      'border-radius:6px;box-shadow:0 8px 32px rgba(0,0,0,.7);',
      'font-size:11px;font-family:var(--font-ui,sans-serif);',
      'min-width:300px;max-height:520px;overflow-y:auto;',
      'scrollbar-width:thin;scrollbar-color:#2a2e39 transparent;',
    ].join('');

    // ── MA SECTION ────────────────────────────────────────────────────────────
    const MA_TYPES  = ['SMA','EMA','WMA','HMA','DEMA','TEMA','VWMA'];
    const MA_COLORS = ['#2196f3','#ff9800','#e91e63','#4caf50','#9c27b0','#00bcd4','#ff5722','#607d8b','#795548'];
    const LINE_STYLES = [ {v:0,l:'Solid'}, {v:1,l:'Dotted'}, {v:2,l:'Dashed'} ];

    function _nextColor() {
      const used = new Set(window._lwMaList.map(m => m.color));
      return MA_COLORS.find(c => !used.has(c)) || MA_COLORS[window._lwMaList.length % MA_COLORS.length];
    }

    // MA group header
    const maHeader = document.createElement('div');
    maHeader.style.cssText = 'padding:8px 12px 4px;color:#4a5060;font-size:9px;letter-spacing:.08em;font-weight:700;border-bottom:1px solid #2a2e39;display:flex;align-items:center;justify-content:space-between;';
    maHeader.innerHTML = '<span>MOVING AVERAGES</span>';

    const addMaBtn = document.createElement('button');
    addMaBtn.textContent = '+ Add MA';
    addMaBtn.style.cssText = 'background:#4f7fff;color:#fff;border:none;border-radius:3px;padding:2px 7px;font-size:9px;font-weight:600;cursor:pointer;letter-spacing:.04em;';
    addMaBtn.addEventListener('mouseenter', () => addMaBtn.style.background = '#5f8fff');
    addMaBtn.addEventListener('mouseleave', () => addMaBtn.style.background = '#4f7fff');
    addMaBtn.addEventListener('click', e => {
      e.stopPropagation();
      const newMa = { uid: _genMaUid(), type:'EMA', period:20, color:_nextColor(), lineWidth:1, lineStyle:0 };
      window._lwMaList.push(newMa);
      _saveMaList();
      _buildMaSeries(newMa);
      _renderIndPills();
      _updateIndBtn();
      pop.remove(); _indDropdownOpen = false; _openIndDropdown();
    });
    maHeader.appendChild(addMaBtn);
    pop.appendChild(maHeader);

    if (window._lwMaList.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:10px 12px;color:#4a5060;font-size:10px;font-style:italic;';
      empty.textContent = 'No moving averages — click "+ Add MA" to add one.';
      pop.appendChild(empty);
    }

    window._lwMaList.forEach((ma, idx) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:5px 10px 5px 12px;border-bottom:1px solid rgba(42,46,57,0.5);';

      // Color swatch + picker
      const colorWrap = document.createElement('label');
      colorWrap.style.cssText = 'position:relative;cursor:pointer;flex-shrink:0;';
      const colorSwatch = document.createElement('span');
      colorSwatch.style.cssText = `display:inline-block;width:12px;height:12px;border-radius:50%;background:${ma.color};border:1px solid rgba(255,255,255,0.15);cursor:pointer;`;
      const colorInput = document.createElement('input');
      colorInput.type = 'color'; colorInput.value = ma.color;
      colorInput.style.cssText = 'position:absolute;opacity:0;width:0;height:0;';
      colorInput.addEventListener('input', e => {
        e.stopPropagation();
        ma.color = e.target.value;
        colorSwatch.style.background = ma.color;
        if (_maSeries[ma.uid]) { try { _maSeries[ma.uid].applyOptions({ color: ma.color }); } catch(_) {} }
        _saveMaList();
        _renderIndPills();
      });
      colorWrap.appendChild(colorSwatch);
      colorWrap.appendChild(colorInput);
      row.appendChild(colorWrap);

      // MA type selector
      const typeSelect = document.createElement('select');
      typeSelect.style.cssText = 'background:#131722;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;';
      MA_TYPES.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t; opt.textContent = t;
        if (t === ma.type) opt.selected = true;
        typeSelect.appendChild(opt);
      });
      typeSelect.addEventListener('change', e => {
        e.stopPropagation();
        ma.type = e.target.value;
        _saveMaList();
        _buildMaSeries(ma);
        _renderIndPills();
      });
      row.appendChild(typeSelect);

      // Period input
      const periodInput = document.createElement('input');
      periodInput.type = 'number'; periodInput.value = ma.period; periodInput.min = 1; periodInput.max = 500;
      periodInput.style.cssText = 'width:44px;background:#131722;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;font-size:10px;text-align:center;';
      periodInput.addEventListener('click', e => e.stopPropagation());
      periodInput.addEventListener('change', e => {
        e.stopPropagation();
        const v = parseInt(e.target.value);
        if (v > 0 && v <= 500) { ma.period = v; _saveMaList(); _buildMaSeries(ma); _renderIndPills(); }
      });
      row.appendChild(periodInput);

      // Line style selector
      const styleSelect = document.createElement('select');
      styleSelect.style.cssText = 'background:#131722;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;';
      LINE_STYLES.forEach(ls => {
        const opt = document.createElement('option');
        opt.value = ls.v; opt.textContent = ls.l;
        if (ls.v === ma.lineStyle) opt.selected = true;
        styleSelect.appendChild(opt);
      });
      styleSelect.addEventListener('change', e => {
        e.stopPropagation();
        ma.lineStyle = parseInt(e.target.value);
        _saveMaList();
        if (_maSeries[ma.uid]) { try { _maSeries[ma.uid].applyOptions({ lineStyle: ma.lineStyle }); } catch(_) {} }
      });
      row.appendChild(styleSelect);

      // Line width selector
      const widthSelect = document.createElement('select');
      widthSelect.style.cssText = 'background:#131722;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;width:36px;';
      [1,2,3].forEach(w => {
        const opt = document.createElement('option');
        opt.value = w; opt.textContent = w + 'px';
        if (w === (ma.lineWidth||1)) opt.selected = true;
        widthSelect.appendChild(opt);
      });
      widthSelect.addEventListener('change', e => {
        e.stopPropagation();
        ma.lineWidth = parseInt(e.target.value);
        _saveMaList();
        if (_maSeries[ma.uid]) { try { _maSeries[ma.uid].applyOptions({ lineWidth: ma.lineWidth }); } catch(_) {} }
      });
      row.appendChild(widthSelect);

      // Remove button
      const rmBtn = document.createElement('button');
      rmBtn.innerHTML = '&times;';
      rmBtn.style.cssText = 'background:none;border:none;color:#4a5060;cursor:pointer;font-size:14px;margin-left:auto;padding:0 2px;line-height:1;flex-shrink:0;';
      rmBtn.title = `Remove ${ma.type} ${ma.period}`;
      rmBtn.addEventListener('mouseenter', () => rmBtn.style.color = '#ef5350');
      rmBtn.addEventListener('mouseleave', () => rmBtn.style.color = '#4a5060');
      rmBtn.addEventListener('click', e => {
        e.stopPropagation();
        window._lwMaList = window._lwMaList.filter(m => m.uid !== ma.uid);
        _saveMaList();
        _destroyMaSeries(ma.uid);
        _renderIndPills(); _updateIndBtn();
        pop.remove(); _indDropdownOpen = false; _openIndDropdown();
      });
      row.appendChild(rmBtn);

      pop.appendChild(row);
    });

    // ── OTHER INDICATOR GROUPS ────────────────────────────────────────────────
    const groups = {};
    _IND_CATALOGUE.filter(c => c.id !== 'ma').forEach(cfg => {
      if (!groups[cfg.group]) groups[cfg.group] = [];
      groups[cfg.group].push(cfg);
    });

    // Helper: build a color swatch + hidden input that updates _lwIndParams[id].colors[i]
    function _makeColorSwatch(id, i, label) {
      const wrap = document.createElement('label');
      wrap.title = label;
      wrap.style.cssText = 'position:relative;cursor:pointer;flex-shrink:0;display:flex;align-items:center;gap:3px;';
      const swatch = document.createElement('span');
      const curColor = _iC(id, i);
      swatch.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:2px;background:${curColor};border:1px solid rgba(255,255,255,0.15);cursor:pointer;`;
      const inp = document.createElement('input');
      inp.type = 'color';
      // Normalise to 6-digit hex (strip alpha if needed)
      const hexOnly = curColor.replace(/^rgba?\([^)]+\)$/, '#888888').replace(/^(#[0-9a-fA-F]{6}).*/, '$1');
      inp.value = hexOnly.startsWith('#') ? hexOnly : '#888888';
      inp.style.cssText = 'position:absolute;opacity:0;width:0;height:0;';
      inp.addEventListener('input', e => {
        e.stopPropagation();
        if (!window._lwIndParams[id]) window._lwIndParams[id] = {};
        if (!window._lwIndParams[id].colors) window._lwIndParams[id].colors = [...((_IND_CATALOGUE.find(c=>c.id===id)||{}).colors||[])];
        window._lwIndParams[id].colors[i] = e.target.value;
        swatch.style.background = e.target.value;
        _saveIndParams();
        // Live-update series color if active
        if (window._lwIndState[id] && window._indSeries && window._indSeries[id] && window._indSeries[id][i]) {
          const cfg2 = _IND_CATALOGUE.find(c => c.id === id);
          if ((cfg2?.histoIdx || []).includes(i)) {
            // Histogram uses per-bar colors — rebuild the whole pane to pick up new color
            try { _buildIndicatorPane(id); } catch(_) {}
          } else {
            try { window._indSeries[id][i].applyOptions({ color: e.target.value }); } catch(_) {}
          }
        }
      });
      wrap.appendChild(swatch);
      wrap.appendChild(inp);
      return wrap;
    }

    Object.entries(groups).forEach(([groupName, items]) => {
      const header = document.createElement('div');
      header.textContent = groupName.toUpperCase();
      header.style.cssText = 'padding:8px 12px 4px;color:#4a5060;font-size:9px;letter-spacing:.08em;font-weight:700;border-top:1px solid #2a2e39;';
      pop.appendChild(header);

      items.forEach(cfg => {
        const isOn = !!window._lwIndState[cfg.id];
        const hasParams = cfg.paramDefs.length > 0;
        const hasColors = cfg.colors.length > 0;
        const expandable = isOn && (hasParams || hasColors);

        // ── Main toggle row ────────────────────────────────────
        const row = document.createElement('div');
        row.style.cssText = `display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;background:${isOn?'rgba(79,127,255,0.08)':'transparent'};border-bottom:${expandable?'none':'1px solid rgba(42,46,57,0.3)'};`;
        row.addEventListener('mouseenter', () => { if (!isOn) row.style.background='rgba(255,255,255,0.04)'; });
        row.addEventListener('mouseleave', () => { row.style.background=isOn?'rgba(79,127,255,0.08)':'transparent'; });

        // Checkbox
        const check = document.createElement('div');
        check.style.cssText = `width:14px;height:14px;border-radius:3px;border:1px solid ${isOn?'#4f7fff':'#3a3f52'};background:${isOn?'#4f7fff':'transparent'};flex-shrink:0;display:flex;align-items:center;justify-content:center;`;
        if (isOn) check.innerHTML = '<svg width="8" height="6" viewBox="0 0 8 6" fill="none"><polyline points="1,3 3,5 7,1" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

        // Label + desc
        const left = document.createElement('div');
        left.style.cssText = 'flex:1;min-width:0;';
        left.innerHTML = `<div style="color:${isOn?'#d1d4dc':'#9da5b4'};font-weight:${isOn?'600':'400'};font-size:11px">${cfg.label}</div>`
          + `<div style="color:#4a5060;font-size:9px;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${cfg.desc}</div>`;

        row.appendChild(check);
        row.appendChild(left);

        // Toggle click
        row.addEventListener('click', e => {
          e.stopPropagation();
          window._lwIndState[cfg.id] = !window._lwIndState[cfg.id];
          _saveIndState();
          if (window._lwIndState[cfg.id]) { _buildIndicatorPane(cfg.id); } else { _destroyIndicatorPane(cfg.id); }
          _renderIndPills(); _updateIndBtn();
          pop.remove(); _indDropdownOpen = false; _openIndDropdown();
        });

        pop.appendChild(row);

        // ── Inline param/color row (only when indicator is ON) ────
        if (expandable) {
          const paramRow = document.createElement('div');
          paramRow.style.cssText = 'display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:4px 12px 7px 34px;background:rgba(79,127,255,0.05);border-bottom:1px solid rgba(42,46,57,0.5);';

          // Numeric params
          cfg.paramDefs.forEach(pd => {
            const lbl = document.createElement('label');
            lbl.style.cssText = 'display:flex;align-items:center;gap:3px;color:#6b7280;font-size:9px;font-weight:600;letter-spacing:.03em;';
            lbl.textContent = pd.label;

            const inp = document.createElement('input');
            inp.type = 'number';
            inp.value = _iP(cfg.id)[pd.key];
            inp.min = pd.min; inp.max = pd.max; inp.step = pd.step || 1;
            inp.style.cssText = 'width:46px;background:#131722;color:#d1d4dc;border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;font-size:10px;text-align:center;';
            inp.addEventListener('click', e => e.stopPropagation());
            inp.addEventListener('change', e => {
              e.stopPropagation();
              const raw = pd.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value);
              if (isNaN(raw) || raw < pd.min || raw > pd.max) return;
              if (!window._lwIndParams[cfg.id]) window._lwIndParams[cfg.id] = {};
              window._lwIndParams[cfg.id][pd.key] = raw;
              _saveIndParams();
              // Rebuild indicator with new params
              if (window._lwIndState[cfg.id]) {
                _destroyIndicatorPane(cfg.id);
                _buildIndicatorPane(cfg.id);
              }
            });
            lbl.appendChild(inp);
            paramRow.appendChild(lbl);
          });

          // Color swatches (with series labels from catalogue)
          const seriesLabels = {
            vwap:    ['Line'],
            bb:      ['Mid','Upper','Lower'],
            keltner: ['Mid','Upper','Lower'],
            donchian:['Upper','Lower','Mid'],
            psar:    ['Dots'],
            ichimoku:['Tenkan','Kijun','Span A','Span B','Chikou'],
            rsi:     ['Line'],
            stoch:   ['%K','%D'],
            macd:    ['Hist','MACD','Signal'],
            cci:     ['Line'],
            willr:   ['Line'],
            roc:     ['Line'],
            mom:     ['Line'],
            mfi:     ['Line'],
            ao:      ['Hist'],
            trix:    ['Line'],
            dpo:     ['Line'],
            uo:      ['Line'],
            atr:     ['Line'],
            adx:     ['ADX','+DI','-DI'],
            aroon:   ['Up','Down'],
            chop:    ['Line'],
            obv:     ['Line'],
            cmf:     ['Line'],
          };
          const labels = seriesLabels[cfg.id] || cfg.colors.map((_,i) => `S${i+1}`);
          cfg.colors.forEach((_, ci) => {
            const sw = _makeColorSwatch(cfg.id, ci, labels[ci] || `Series ${ci+1}`);
            const lbl = document.createElement('span');
            lbl.style.cssText = 'color:#6b7280;font-size:9px;';
            lbl.textContent = labels[ci] || `S${ci+1}`;
            const wrap2 = document.createElement('div');
            wrap2.style.cssText = 'display:flex;align-items:center;gap:2px;';
            wrap2.appendChild(sw);
            wrap2.appendChild(lbl);
            paramRow.appendChild(wrap2);
          });

          pop.appendChild(paramRow);
        }
      });
    });

    document.body.appendChild(pop);

    // Position below the button
    if (btn) {
      const rect = btn.getBoundingClientRect();
      const popH = Math.min(520, pop.scrollHeight || 450);
      const spaceBelow = window.innerHeight - rect.bottom;
      const top = spaceBelow >= 80 ? rect.bottom + 4 : rect.top - popH - 4;
      pop.style.top  = Math.max(8, top) + 'px';
      pop.style.left = Math.max(8, rect.left) + 'px';
    }

    // Stop ALL clicks inside the popup from bubbling to document
    pop.addEventListener('click',     e => e.stopPropagation());
    pop.addEventListener('mousedown', e => e.stopPropagation());

    // Close when user clicks/mousedowns outside the popup
    setTimeout(() => {
      document.addEventListener('mousedown', _closeIndDropdown, { once: true });
    }, 0);
  }

  // Attach dropdown handler — clone to clear prior listeners
  (function _attachIndBtn() {
    const btn = document.getElementById('lw-ind-btn');
    if (!btn) return;
    const fresh = btn.cloneNode(true);
    btn.parentNode.replaceChild(fresh, btn);
    fresh.addEventListener('click', e => { e.stopPropagation(); _openIndDropdown(); });
  })();

  _renderIndPills();
  _updateIndBtn();

  // ── Symbol legend header (mirrors TradingView legend) ──────────────────────
  function _fmtHdrVal(v) { return v != null && !isNaN(v) ? v.toFixed(dec) : '\u2014'; }

  // MA legend removed — MAs are now shown via the indicator pills bar
  function _updateAllMALegend() {}   // no-op shim — referenced by crosshair handler
  function _updateMALegend() {}      // no-op shim

  // prevClose map: date → prev bar's close, for day-over-day % change in header
  const _prevCloseMap = new Map();
  for (let i = 1; i < bars.length; i++) {
    _prevCloseMap.set(bars[i].time, bars[i - 1].close);
  }
  // Expose to _lwUpdateTodayBar so it can inject today's prevClose from yfinance RT cache
  _lwActivePrevCloseMap = _prevCloseMap;

  function _updateLWHeader(bar, maVal, rtOverride) {
    const symEl  = document.getElementById('lw-hdr-sym');
    const oEl    = document.getElementById('lw-hdr-o-val');
    const hEl    = document.getElementById('lw-hdr-h-val');
    const lEl    = document.getElementById('lw-hdr-l-val');
    const cEl    = document.getElementById('lw-hdr-c-val');
    const chgEl  = document.getElementById('lw-hdr-chg-val');
    if (symEl) symEl.textContent = (_OHLC_FULL_NAMES[ohlcId] || label) + ' \u00b7 ' + _lwActiveTf;
    if (bar) {
      // Determine direction first so O/H/L/C all share the same color (industry standard)
      let isUp;
      {
        let _pctForDir;
        if (rtOverride?.pct != null) {
          _pctForDir = rtOverride.pct;
        } else {
          const _pc = _prevCloseMap.get(bar.time) ?? bar.open;
          _pctForDir = (_pc != null && _pc > 0 && bar.close != null) ? ((bar.close - _pc) / _pc) * 100 : null;
        }
        isUp = _pctForDir != null ? _pctForDir >= 0 : (bar.close != null && bar.close >= (bar.open ?? bar.close));
      }
      const ohlcColor = isUp ? '#26a69a' : '#ef5350';
      const _isOHLCType = (window._lwChartType === 'candle' || window._lwChartType === 'bar');
      // Hide O/H/L labels for Line/Area — only Close is meaningful
      const _ohlcWrap = document.getElementById('lw-hdr-ohlc-wrap');
      if (_ohlcWrap) _ohlcWrap.style.display = _isOHLCType ? '' : 'none';
      if (oEl) { oEl.textContent = _fmtHdrVal(bar.open); oEl.style.color = ohlcColor; }
      if (hEl) { hEl.textContent = _fmtHdrVal(bar.high); hEl.style.color = ohlcColor; }
      if (lEl) { lEl.textContent = _fmtHdrVal(bar.low);  lEl.style.color = ohlcColor; }
      if (cEl) { cEl.textContent = _fmtHdrVal(bar.close); cEl.style.color = ohlcColor; }
      if (chgEl) {
        // rtOverride: use yfinance pct/chg directly (avoids JSON-vs-yfinance prevClose divergence)
        // Fallback: recalculate from _prevCloseMap (used for crosshair hover on historical bars)
        let chg, pct;
        if (rtOverride?.pct != null) {
          pct = rtOverride.pct;
          chg = rtOverride.chg ?? (bar.close != null && bar.open != null ? bar.close - bar.open : null);
        } else {
          const prevClose = _prevCloseMap.get(bar.time) ?? bar.open;
          chg = (prevClose != null && bar.close != null) ? bar.close - prevClose : null;
          pct = (prevClose != null && prevClose > 0 && bar.close != null) ? (chg / prevClose) * 100 : null;
        }
        if (pct != null && chg != null) {
          const sign = chg >= 0 ? '+' : '';
          chgEl.textContent = ' ' + sign + chg.toFixed(dec) + ' (' + sign + pct.toFixed(2) + '%)';
          chgEl.className = 'lw-hdr-chg ' + (chg >= 0 ? 'up' : 'dn');
        } else {
          chgEl.textContent = '';
        }
      }
    }
    _updateMALegend(maVal);
  }

  // Expose _updateLWHeader to _lwUpdateTodayBar so live RT data syncs the header % with the ticker
  _lwActiveUpdateHeader = _updateLWHeader;

  // Helper: get yfinance RT override for the active symbol (used on initial render + crosshair restore)
  function _getRtOverride() {
    const ck = ohlcId === 'gold' ? 'xauusd' : ohlcId;
    const rt = STOOQ_RT_CACHE[ck];
    return (rt?.pct != null) ? { pct: rt.pct, chg: rt.chg } : null;
  }

  // Show the header and populate with last bar
  const hdrEl = document.getElementById('lw-chart-header');
  if (hdrEl) hdrEl.style.display = 'flex';

  // Populate with last available bar — use yfinance RT pct if available (avoids JSON prevClose drift)
  const lastBar = todayBar || (bars.length > 0 ? bars[bars.length - 1] : null);
  _updateLWHeader(lastBar, null, _getRtOverride());

  // Update panel-sub to reflect active data source
  const panelSub = document.querySelector('#section-fxpairs .panel-sub');
  if (panelSub) {
    const _hasFinnhubLive = Object.values(STOOQ_RT_CACHE).some(e => e?.fromFinnhub);
    panelSub.textContent = _hasFinnhubLive ? 'Finnhub \u00b7 live' : 'yfinance \u00b7 ~5min delay';
  }

  // Crosshair subscription — update OHLC legend on hover, clear MA label on leave
  // ── CB Meeting floating tooltip — TradingView floating-tooltip pattern ────
  // Follows https://tradingview.github.io/lightweight-charts/tutorials/how_to/tooltips#floating-tooltip
  // A single positioned div is created once per chart render and repositioned on
  // every crosshairMove tick. It flips left when near the right edge and below
  // when near the top, matching Bloomberg's CB annotation UX exactly.
  const _CB_NAMES = {
    USD:'Federal Reserve (FOMC)', EUR:'ECB Governing Council',
    GBP:'Bank of England',        JPY:'Bank of Japan',
    AUD:'Reserve Bank of Australia', CAD:'Bank of Canada',
    CHF:'Swiss National Bank',    NZD:'Reserve Bank of New Zealand',
  };
  const TOOLTIP_W  = 200; // px — fixed width so we can flip without measuring
  const TOOLTIP_H  = 48;  // px — estimated max height (2 CB rows); actual may be less
  const TOOLTIP_MARGIN = 12; // gap between crosshair point and tooltip corner

  const _cbTooltip = document.createElement('div');
  _cbTooltip.id = '_lw-cb-tooltip';
  // Base styles — matches LWC floating tooltip reference implementation
  Object.assign(_cbTooltip.style, {
    position:       'absolute',
    display:        'none',
    pointerEvents:  'none',
    boxSizing:      'border-box',
    width:          TOOLTIP_W + 'px',
    background:     '#1e222d',
    border:         '1px solid #363c4e',
    borderRadius:   '4px',
    padding:        '6px 10px',
    fontSize:       '11px',
    lineHeight:     '1.5',
    fontFamily:     'var(--font-ui,sans-serif)',
    color:          '#d1d4dc',
    zIndex:         '50',
    boxShadow:      '0 4px 12px rgba(0,0,0,.6)',
  });
  chartDiv.style.position = 'relative';
  chartDiv.appendChild(_cbTooltip);

  _lwChart.subscribeCrosshairMove(param => {
    // ── Header update & MA legend (runs regardless of CB tooltip state) ──
    if (!param || !param.time || !param.seriesData) {
      _updateLWHeader(lastBar, null, _getRtOverride());
      _updateAllMALegend(null);
      _cbTooltip.style.display = 'none';
      return;
    }
    const _rawSeriesData = param.seriesData.get(candleSeries);
    // Normalize Line/Area {time,value} → OHLC-like for _updateLWHeader
    const candleData = _rawSeriesData
      ? (_rawSeriesData.close != null ? _rawSeriesData
         : { ..._rawSeriesData, open: _rawSeriesData.value, high: _rawSeriesData.value,
             low: _rawSeriesData.value, close: _rawSeriesData.value })
      : null;
    if (candleData) _updateAllMALegend(param.seriesData);
    const isCurrentBar = lastBar && candleData && candleData.time === lastBar.time;
    if (candleData) _updateLWHeader(candleData, null, isCurrentBar ? _getRtOverride() : null);

    // ── CB floating tooltip ──
    const dateStr = typeof param.time === 'string' ? param.time
      : new Date(param.time * 1000).toISOString().slice(0, 10);
    const cbEvents = window._lwShowCb && window._lwCbMarkerMap && window._lwCbMarkerMap[dateStr];
    if (!cbEvents || cbEvents.length === 0) {
      _cbTooltip.style.display = 'none';
      return;
    }

    // Build tooltip content
    const lines = cbEvents.map(ev => {
      const name = _CB_NAMES[ev.cb] || ev.cb;
      return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:1px;">`
        + `<span style="display:inline-block;width:3px;height:12px;background:${ev.color};border-radius:1px;flex-shrink:0;"></span>`
        + `<span><span style="color:${ev.color};font-weight:700;">${ev.cb}</span>`
        + ` <span style="color:#848ea0;font-size:10px;">${name}</span></span>`
        + `</div>`;
    }).join('');
    _cbTooltip.innerHTML =
      `<div style="font-size:9px;color:#6b7280;letter-spacing:.05em;margin-bottom:3px;">CB MEETING</div>`
      + lines;

    // Position tooltip — floating-tooltip flip logic
    // Flip horizontally when crosshair is past the midpoint of the chart,
    // flip vertically when crosshair is in the top 25% of the chart.
    _cbTooltip.style.display = 'block';
    const cW = chartDiv.offsetWidth;
    const cH = chartDiv.offsetHeight;
    const cx = param.point?.x ?? 0;
    const cy = param.point?.y ?? 0;
    // Horizontal: default = right of crosshair; flip left if not enough room
    const tx = (cx + TOOLTIP_MARGIN + TOOLTIP_W <= cW - 4)
      ? cx + TOOLTIP_MARGIN
      : cx - TOOLTIP_MARGIN - TOOLTIP_W;
    // Vertical: default = above crosshair; flip below if near top
    const actualH = _cbTooltip.offsetHeight || TOOLTIP_H;
    const ty = (cy - actualH - TOOLTIP_MARGIN >= 4)
      ? cy - actualH - TOOLTIP_MARGIN
      : cy + TOOLTIP_MARGIN;
    _cbTooltip.style.left = Math.max(0, tx) + 'px';
    _cbTooltip.style.top  = Math.max(0, ty) + 'px';
  });

  // Apply the active range window (default 3M, persists across symbol switches)
  _lwSetRange(_lwActiveDays, bars.length);

  // Show range toolbar and sync active button
  const rangeBar = document.getElementById('lw-range-bar');
  if (rangeBar) {
    rangeBar.style.display = 'flex';
    // Sync TF selector
    rangeBar.querySelectorAll('.lw-tf-btn').forEach(b => {
      b.classList.toggle('sel', b.dataset.tf === _lwActiveTf);
    });
    // Rebuild range buttons for the current TF
    _lwUpdateRangeBtns();
    // Sync active range button
    rangeBar.querySelectorAll('.lw-range-btn').forEach(b => {
      b.classList.toggle('active', parseInt(b.dataset.days) === _lwActiveDays);
    });
  }

  // Responsive resize
  if (typeof ResizeObserver !== 'undefined') {
    _lwResizeObs = new ResizeObserver(entries => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (_lwChart && width > 0 && height > 0) _lwChart.resize(width, height);
      }
    });
    _lwResizeObs.observe(chartDiv);
  }
}

// ── COT Chart: always uses TradingView widget (comparative overlay) ──
function loadCOTChart(longSym) {
  const shortSym = longSym.replace(/_L$/, '_S');
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  _chartMode = 'tv'; // set synchronously before destroying LW chart
  _destroyLWChart();
  wrap.innerHTML = '';
  wrap.style.pointerEvents = 'none';
  wrap.style.marginBottom = '-32px';
  const rangeBar = document.getElementById('lw-range-bar');
  if (rangeBar) rangeBar.style.display = 'none';
  const cotHdr = document.getElementById('lw-chart-header');
  if (cotHdr) cotHdr.style.display = 'none';
  const container = document.createElement('div');
  container.className = 'tradingview-widget-container';
  container.style.cssText = 'height:100%;width:100%;';
  const widget = document.createElement('div');
  widget.className = 'tradingview-widget-container__widget';
  widget.style.cssText = 'height:100%;width:100%;';
  container.appendChild(widget);
  const copyright = document.createElement('div');
  copyright.className = 'tradingview-widget-copyright';
  copyright.style.display = 'none';
  container.appendChild(copyright);
  const script = document.createElement('script');
  script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
  script.async = true;
  script.text = JSON.stringify({
    allow_symbol_change: false, calendar: false, details: false,
    hide_side_toolbar: true, hide_top_toolbar: true, hide_legend: false,
    hide_volume: true, interval: 'W', locale: 'en', save_image: true,
    style: '2', symbol: longSym, theme: 'dark', timezone: 'Etc/UTC',
    backgroundColor: '#131722', gridColor: 'rgba(42,46,57,0.8)',
    withdateranges: false, compareSymbols: [{ symbol: shortSym, position: 'SameScale' }],
    scaleMode: 2, studies: [], autosize: true,
  });
  container.appendChild(script);
  wrap.appendChild(container);
  const chartSection = document.getElementById('section-fxpairs') || wrap.closest('.panel') || wrap;
  chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Internal: TV widget fallback for symbols without OHLC data ──
function _loadTVWidgetFallback(sym) {
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  _chartMode = 'tv'; // set synchronously before destroying LW chart
  _destroyLWChart();
  wrap.innerHTML = '';
  // Restore pointer-events:none — TV widget manages its own interaction via iframe
  wrap.style.pointerEvents = 'none';
  // Restore negative margin to hide TradingView widget's internal iframe footer bar
  wrap.style.marginBottom = '-32px';
  // Hide range toolbar and symbol header — not applicable to TV widget
  const rangeBar = document.getElementById('lw-range-bar');
  if (rangeBar) rangeBar.style.display = 'none';
  const hdrEl = document.getElementById('lw-chart-header');
  if (hdrEl) hdrEl.style.display = 'none';
  const panelSub = document.querySelector('#section-fxpairs .panel-sub');
  if (panelSub) panelSub.textContent = 'TradingView \u00b7 live data';
  const container = document.createElement('div');
  container.className = 'tradingview-widget-container';
  container.style.cssText = 'height:100%;width:100%;';
  const widget = document.createElement('div');
  widget.id = 'tv-chart-widget';
  widget.className = 'tradingview-widget-container__widget';
  widget.style.cssText = 'height:100%;width:100%;';
  container.appendChild(widget);
  const copyright = document.createElement('div');
  copyright.className = 'tradingview-widget-copyright';
  copyright.style.display = 'none';
  container.appendChild(copyright);
  const script = document.createElement('script');
  script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
  script.async = true;
  const chartStyle = sym === 'FRED:DGS10' ? '3' : '1';
  script.text = JSON.stringify({
    allow_symbol_change:false, calendar:false, details:true,
    hide_side_toolbar:true, hide_top_toolbar:true, hide_legend:false,
    hide_volume:true, interval:'D', locale:'en', save_image:false,
    style:chartStyle, symbol:sym, theme:'dark', timezone:'Etc/UTC',
    backgroundColor:'#131722', gridColor:'rgba(42,46,57,0.8)',
    withdateranges:false, studies:[{id:'MASimple@tv-basicstudies',inputs:{length:20}}], autosize:true
  });
  container.appendChild(script);
  wrap.appendChild(container);
}

// SHARED: load any symbol into the chart + scroll to it
// Prefers Lightweight Charts (yfinance OHLC); falls back to TradingView widget.
// ═══════════════════════════════════════════════════════════════════
function loadTVChart(sym) {
  document.querySelectorAll('.tv-tab').forEach(t => {
    t.classList.remove('active');
    if (t.dataset.sym === sym) t.classList.add('active');
  });
  updatePairDetail(sym);
  const chartSection = document.getElementById('section-fxpairs') ||
    document.getElementById('tv-chart-wrap')?.closest('.panel') ||
    document.getElementById('tv-chart-wrap');
  const ohlcId = _TV_TO_OHLC[sym];
  if (ohlcId) {
    const label = sym.split(':').pop().replace(/[^A-Z0-9/]/gi, '');
    _renderLWChart(ohlcId, label)
      .then(() => { if (chartSection) chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); })
      .catch(err => {
        // Log the real exception — primary diagnostic for the TV-fallback regression.
        // Without this log the error was silently swallowed and the TV widget loaded
        // with no console trace of the root cause.
        console.error('[LWChart] _renderLWChart failed for', ohlcId, '—', err);
        _loadTVWidgetFallback(sym);
        if (chartSection) chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
  } else {
    _loadTVWidgetFallback(sym);
    if (chartSection) chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── Quote bar: click any item to open chart ──
document.getElementById('quotebar-inner')?.addEventListener('click', e => {
  const item = e.target.closest('.q-item');
  if (!item) return;
  const sym = item.dataset.sym;
  if (sym) loadTVChart(sym);
});

// Range toolbar buttons — update visible window on active LW chart
document.getElementById('lw-range-bar')?.addEventListener('click', e => {
  const btn = e.target.closest('.lw-range-btn');
  if (!btn) return;
  _lwSetRange(parseInt(btn.dataset.days));
});

// ── Log Scale toggle ──
document.getElementById('lw-log-btn')?.addEventListener('click', function() {
  window._lwLogScale = !window._lwLogScale;
  this.classList.toggle('on', window._lwLogScale);
  this.setAttribute('aria-pressed', window._lwLogScale ? 'true' : 'false');
  if (_lwChart) {
    try { _lwChart.priceScale('right').applyOptions({ mode: window._lwLogScale ? 1 : 0 }); } catch(_) {}
  }
});

// ── Overlay toggle handlers — all share the same pattern ──
// Toggle class 'on' for visual state (defined in index.html <style>) + re-render
document.getElementById('lw-wm-btn')?.addEventListener('click', function() {
  window._lwShowWm = !window._lwShowWm;
  this.classList.toggle('on', window._lwShowWm);
  this.setAttribute('aria-pressed', window._lwShowWm ? 'true' : 'false');
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

document.getElementById('lw-pc-btn')?.addEventListener('click', function() {
  window._lwShowPc = !window._lwShowPc;
  this.classList.toggle('on', window._lwShowPc);
  this.setAttribute('aria-pressed', window._lwShowPc ? 'true' : 'false');
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

document.getElementById('lw-vol-btn')?.addEventListener('click', function() {
  window._lwShowVol = !window._lwShowVol;
  this.classList.toggle('on', window._lwShowVol);
  this.setAttribute('aria-pressed', window._lwShowVol ? 'true' : 'false');
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

document.getElementById('lw-cb-btn')?.addEventListener('click', function() {
  window._lwShowCb = !window._lwShowCb;
  this.classList.toggle('on', window._lwShowCb);
  this.setAttribute('aria-pressed', window._lwShowCb ? 'true' : 'false');
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});



// ── Chart type selector ──
// Bloomberg standard: Candlestick default; Bar, Line, Area as alternatives.
// Chart type persists across symbol switches via window._lwChartType.
document.getElementById('lw-range-bar')?.addEventListener('click', function(e) {
  const typeBtn = e.target.closest('[data-chart-type]');
  if (!typeBtn) return;
  window._lwChartType = typeBtn.dataset.chartType;
  document.querySelectorAll('[data-chart-type]').forEach(b => {
    b.classList.toggle('sel', b === typeBtn);
    b.classList.remove('on');
  });
  // Immediately show/hide OHLC header — no need to wait for chart re-render
  const _ohlcWrap = document.getElementById('lw-hdr-ohlc-wrap');
  if (_ohlcWrap) _ohlcWrap.style.display = (window._lwChartType === 'candle' || window._lwChartType === 'bar') ? '' : 'none';
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

// ── Pair Detail Popover ─────────────────────────────────────────────────────
// ── INLINE EXPAND-IN-ROW DETAIL (FX Pairs table) ─────────────────────────
// Clicking a pair row in the FX Pairs table expands an inline detail strip
// immediately below the row — no overlay, no focus loss, chart + table coexist.
// Pattern: Bloomberg/Refinitiv inline expansion for compact terminal tables.

function toggleInlineDetail(row) {
  const tvSym = row.dataset.sym;
  const tbody = row.closest('tbody');
  if (!tbody) return;

  // If this row is already open, collapse it
  const existingExpand = tbody.querySelector('tr.pd-expand-row');
  const wasThisRow = existingExpand?.dataset.forSym === tvSym;

  // Always remove any existing expand row first
  if (existingExpand) {
    const inner = existingExpand.querySelector('td > div');
    if (inner) {
      // Snap to scrollHeight first so CSS transition can animate from a numeric value → 0
      inner.style.maxHeight = inner.scrollHeight + 'px';
      inner.style.overflow = 'hidden';
      requestAnimationFrame(() => { inner.style.maxHeight = '0'; });
    }
    setTimeout(() => existingExpand.remove(), 200);
    tbody.querySelector('tr.pd-selected')?.classList.remove('pd-selected');
  }

  if (wasThisRow) return; // toggle off

  // Mark selected row
  row.classList.add('pd-selected');

  // Insert expansion row after selected row
  const expandRow = document.createElement('tr');
  expandRow.className = 'pd-expand-row';
  expandRow.dataset.forSym = tvSym;
  const td = document.createElement('td');
  td.colSpan = 12; // FX table has 12 columns
  const inner = document.createElement('div');
  inner.innerHTML = '<div style="padding:6px 10px;font-size:10px;color:var(--text3);">Loading…</div>';
  td.appendChild(inner);
  expandRow.appendChild(td);
  row.after(expandRow);

  // Animate open, then remove the cap so content is never clipped
  requestAnimationFrame(() => {
    expandRow.classList.add('pd-open');
    inner.style.maxHeight = '185px';
    setTimeout(() => {
      if (expandRow.classList.contains('pd-open')) {
        inner.style.maxHeight = 'none';
        inner.style.overflow  = 'visible';
      }
    }, 200); // slightly after the 180ms transition
  });

  // Populate with real data
  buildInlineDetail(tvSym, inner);
}

async function buildInlineDetail(tvSym, container) {
  const meta   = pairMetaFromSym(tvSym);
  const label  = meta?.label || tvSym.replace(/^.*:/,'').replace(/(.{3})(.{3})/,'$1/$2').toUpperCase();
  const pairId = meta?.id || null;
  const base   = meta?.base  || null;
  const quote  = meta?.quote || null;
  const invert = meta?.invert ?? false;
  const dec    = meta?.dec   ?? 5;

  const rt    = pairId ? STOOQ_RT_CACHE[pairId] : null;
  const price = rt?.close ?? null;
  const pct1d = rt?.pct   ?? null;
  const pct1w = rt?.pct1w ?? null;
  const hv30  = rt?.hv30  ?? null;
  const sessH = rt?.high  ?? null;
  const sessL = rt?.low   ?? null;

  const pipVal     = dec === 3 ? 0.01 : 0.0001;
  const spreadPips = pairId ? (TYPICAL_SPREADS[pairId] || null) : null;
  let adr = null;
  if (hv30 != null && price != null) {
    adr = Math.round(price * (hv30 / 100) / Math.sqrt(252) / pipVal);
  }

  // ATM IV (reuse same logic as updatePairDetail)
  const CROSS_IV_RHO = {
    'eurgbp':0.65,'eurjpy':0.55,'eurchf':0.60,'eurcad':0.40,'euraud':0.35,'eurnzd':0.30,
    'gbpjpy':0.45,'gbpchf':0.55,'gbpcad':0.30,'gbpaud':0.25,'gbpnzd':0.20,
    'audjpy':0.40,'audnzd':0.55,'audchf':0.30,'audcad':0.50,
    'cadjpy':0.35,'cadchf':0.25,'chfjpy':0.40,'nzdjpy':0.35,'nzdcad':0.45,'nzdchf':0.20,
  };
  const USD_IV = {};
  let atmIv = null;
  try {
    const intra = await loadIntradayQuotes();
    const etfIv = intra?.fx_etf_iv || {};
    for (const [pid, entry] of Object.entries(etfIv)) {
      if (entry?.iv == null) continue;
      const p = PAIRS.find(x => x.id === pid);
      if (!p) continue;
      const nonUsd = p.base !== 'USD' ? p.base : p.quote;
      USD_IV[nonUsd] = entry.iv;
    }
    if (USD_IV['AUD'] != null && USD_IV['NZD'] == null) USD_IV['NZD'] = Math.round(USD_IV['AUD'] * 1.08 * 10) / 10;
    const ivEntry = etfIv[pairId];
    if (ivEntry?.iv != null) {
      atmIv = ivEntry.iv;
    } else if (pairId && meta?.cross) {
      const ivA = USD_IV[base] ?? null, ivB = USD_IV[quote] ?? null;
      if (ivA != null && ivB != null) {
        const rho = CROSS_IV_RHO[pairId] ?? 0.40;
        atmIv = Math.round(Math.sqrt(ivA*ivA + ivB*ivB - 2*rho*ivA*ivB) * 10) / 10;
      }
    }
  } catch {}

  // COT — for crosses, load BOTH component currencies
  const isCrossPair = !!meta?.cross;
  const cotCcy = base && base !== 'USD' ? base : (quote && quote !== 'USD' ? quote : base);
  const cotRaw = cotCcy ? (COT_DATA_CACHE[cotCcy] || null) : null;
  const cotCcy2 = isCrossPair && quote && quote !== cotCcy ? quote : null;
  const cotRaw2 = cotCcy2 ? (COT_DATA_CACHE[cotCcy2] || null) : null;
  let cotNet = null, cotAmNet = null, cotWow = null, cotPctOI = null, cotWeek = '';
  if (cotRaw) {
    const flip = (invert && cotCcy === quote) ? -1 : 1;
    cotNet   = cotRaw.net   != null ? cotRaw.net   * flip : null;
    cotAmNet = cotRaw.amNet != null ? cotRaw.amNet * flip : null;
    cotWow   = cotRaw.wowNetChange != null ? cotRaw.wowNetChange * flip : null;
    cotPctOI = cotRaw.levNetPctOI  != null ? cotRaw.levNetPctOI  * flip : null;
    cotWeek  = cotRaw.weekEnding || '';
  }
  let cot2Net = null, cot2AmNet = null, cot2Wow = null, cot2PctOI = null;
  if (cotRaw2) {
    cot2Net    = cotRaw2.net          ?? null;
    cot2AmNet  = cotRaw2.amNet        ?? null;
    cot2Wow    = cotRaw2.wowNetChange ?? null;
    cot2PctOI  = cotRaw2.levNetPctOI  ?? null;
    if (!cotWeek && cotRaw2.weekEnding) cotWeek = cotRaw2.weekEnding;
  }

  // Carry — OIS rate preferred over CB policy rate (Bloomberg standard)
  // OIS reflects the market's current funding cost; policy rate lags by one meeting.
  // _resolveRate() returns [rate, source] — OIS if available, policy fallback.
  const [oisBase,  oisSrcBase]  = (typeof _resolveRate === 'function' && base)  ? _resolveRate(base)  : [null, null];
  const [oisQuote, oisSrcQuote] = (typeof _resolveRate === 'function' && quote) ? _resolveRate(quote) : [null, null];
  const cbBase  = oisBase  ?? (base  ? (STATE.cbRates?.[base.toLowerCase()]?.rate  ?? null) : null);
  const cbQuote = oisQuote ?? (quote ? (STATE.cbRates?.[quote.toLowerCase()]?.rate ?? null) : null);
  const carrySource = (oisBase != null || oisQuote != null) ? 'OIS' : 'policy rate';
  let carryDiff = null;
  if (cbBase != null && cbQuote != null) {
    carryDiff = meta?.cross ? cbBase - cbQuote : (invert ? cbBase - cbQuote : cbQuote - cbBase);
  }

  // RR
  const rrKey = base && quote ? (base + quote).toUpperCase() : null;
  const rrVal = rrKey ? (RR_DATA_CACHE[rrKey]?.rr25d ?? null) : null;

  // Retail
  const retKey = label.toUpperCase();
  const ret     = RETAIL_SENTIMENT_CACHE[retKey] || null;
  const retL    = ret?.longPct  ?? null;
  const retS    = ret?.shortPct ?? null;
  const retAvgL = ret?.avgL     ?? null;   // avg entry price of retail longs
  const retAvgS = ret?.avgS     ?? null;   // avg entry price of retail shorts
  const retLPos = ret?.longPos  ?? null;   // number of long positions
  const retSPos = ret?.shortPos ?? null;   // number of short positions
  // Contrarian skew label (IG / Bloomberg convention: >65% = extreme)
  const retSkew = retL == null ? null
    : retL >= 75 ? 'Heavily Long'
    : retL >= 65 ? 'Majority Long'
    : retL <= 25 ? 'Heavily Short'
    : retL <= 35 ? 'Majority Short'
    : 'Mixed';
  const retSkewCls = retL == null ? ''
    : retL >= 65 ? 'pd-dn'   // contrarian = bearish signal when heavily long
    : retL <= 35 ? 'pd-up'   // contrarian = bullish signal when heavily short
    : 'pd-dim';
  // Avg entry vs current price: are retail longs underwater?
  const retLUnder = (retAvgL != null && price != null && retL != null && retL >= 50)
    ? price < retAvgL   // longs are underwater if price below avg entry
    : null;
  const retSUnder = (retAvgS != null && price != null && retS != null && retS > 50)
    ? price > retAvgS   // shorts are underwater if price above avg entry
    : null;

  // Formatting helpers
  const fmtP  = v => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
  const fmtN  = v => v == null ? '—' : (v >= 0 ? '+' : '') + Math.round(v).toLocaleString();
  // Use pd-up/pd-dn throughout — these have explicit .pd-val.pd-up rules that
  // override the base color:var(--text) on .pd-inline-val without specificity fights.
  const cls   = v => v == null ? '' : v > 0 ? 'pd-up' : v < 0 ? 'pd-dn' : '';
  const clsI  = v => v == null ? '' : v > 0 ? 'pd-up' : v < 0 ? 'pd-dn' : '';
  const fmtV  = (v, suffix='') => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + suffix;
  const ivCls = v => v == null ? '' : v > 12 ? 'pd-dn' : v < 7 ? 'pd-up' : '';

  // COT summary tag — for crosses show both component currencies
  let cotTag = '—';
  if (isCrossPair && cotCcy2) {
    const parts = [];
    for (const [ccy, net, amNet] of [[cotCcy, cotNet, cotAmNet], [cotCcy2, cot2Net, cot2AmNet]]) {
      if (net == null) continue;
      const lfD = net > 0 ? 'Long' : net < 0 ? 'Short' : null;
      const amD = amNet != null ? (amNet > 0 ? 'Long' : amNet < 0 ? 'Short' : null) : null;
      if (!lfD) continue;
      const lfC = net > 0 ? 'pd-up' : 'pd-dn';
      const amPart = amD ? ` · <span class="${amNet > 0 ? 'pd-up' : 'pd-dn'}">${amD}</span> <span style="color:var(--text3);font-size:8px;">AM</span>` : '';
      const alignedStr = lfD && amD ? (lfD === amD ? ' · <span style="color:var(--text3);font-size:8px;">aligned</span>' : ' · <span style="color:var(--text3);font-size:8px;">diverging</span>') : '';
      parts.push(`<span style="color:var(--text3);font-size:8px;text-transform:uppercase;">${ccy}</span> <span class="${lfC}">${lfD}</span> <span style="color:var(--text3);font-size:8px;">LF</span>${amPart}${alignedStr}`);
    }
    cotTag = parts.join('<span style="color:var(--text3);"> · </span>') || '—';
  } else {
    const lfDir = cotNet == null ? null : cotNet > 0 ? 'Long' : cotNet < 0 ? 'Short' : null;
    const amDir = cotAmNet == null ? null : cotAmNet > 0 ? 'Long' : cotAmNet < 0 ? 'Short' : null;
    const aligned = lfDir && amDir && lfDir === amDir;
    cotTag = lfDir && amDir
      ? `<span class="${clsI(cotNet)}">${lfDir}</span> <span style="color:var(--text3);font-size:8px;">LF · </span><span class="${clsI(cotAmNet)}">${amDir}</span> <span style="color:var(--text3);font-size:8px;">AM · ${aligned ? 'aligned' : 'diverging'}</span>`
      : '—';
  }

  const footerSources = [cotWeek ? 'COT ' + cotWeek : null, 'Myfxbook', rrVal != null ? 'Saxo RR' : null].filter(Boolean).join(' · ');

  container.innerHTML = `
    <div class="pd-inline-scroll">
    <div class="pd-inline">
      <div class="pd-inline-price">
        <div class="pd-inline-sym">${label}</div>
        <div class="pd-inline-rate">${price != null ? price.toFixed(dec) : '—'}</div>
        <div class="pd-inline-chg ${cls(pct1d)}">${fmtP(pct1d)}</div>
        ${sessH != null && sessL != null ? `<div class="pd-inline-meta">H ${sessH.toFixed(dec)} · L ${sessL.toFixed(dec)}</div>` : ''}
      </div>

      <div class="pd-inline-group">
        <div class="pd-inline-group-lbl">Price</div>
        <div class="pd-inline-metrics">
          <div class="pd-inline-metric fx-tip" data-tip-title="1-Week Change" data-tip-body="Weekly % change vs prior Friday close.">
            <div class="pd-inline-lbl">1W Chg</div><div class="pd-inline-val ${cls(pct1w)}">${fmtP(pct1w)}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="Carry Differential" data-tip-body="OIS/overnight rate differential (OIS preferred; falls back to CB policy rate). Positive = carry favours long base currency." data-tip-ex="Long the higher-yielding currency, short the lower-yielding currency. OIS reflects actual overnight funding cost; CB policy rate is the ceiling.">
            <div class="pd-inline-lbl">Carry</div><div class="pd-inline-val ${clsI(carryDiff)}">${carryDiff != null ? (carryDiff >= 0 ? '+' : '') + carryDiff.toFixed(2) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="Average Daily Range" data-tip-body="Estimated avg daily range in pips from HV 30d. Useful for stop/target sizing.">
            <div class="pd-inline-lbl">ADR</div><div class="pd-inline-val">${adr != null ? adr + ' pip' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="${base || 'Base'} Policy Rate" data-tip-body="${base || 'Base'} central bank policy rate (annualised).">
            <div class="pd-inline-lbl">${base || 'Base'} Rate</div><div class="pd-inline-val">${cbBase != null ? cbBase.toFixed(2) + '%' : '—'}</div>
          </div>
        </div>
      </div>

      <div class="pd-inline-group">
        <div class="pd-inline-group-lbl">Volatility</div>
        <div class="pd-inline-metrics">
          <div class="pd-inline-metric fx-tip" data-tip-title="Historical Volatility 30d" data-tip-body="30-day realised volatility, annualised. Measures recent actual movement.">
            <div class="pd-inline-lbl">HV 30d</div><div class="pd-inline-val">${hv30 != null ? hv30.toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="ATM Implied Volatility" data-tip-body="ATM IV from CBOE/CME FX Volatility Indexes (^EUVIX, ^BPVIX, ^JYVIX, ^AUDVIX) — same variance-swap methodology as VIX, published jointly by CBOE and CME. Proxy for OTC interbank IV. Green ≤7%; red >12%.">
            <div class="pd-inline-lbl">ATM IV</div><div class="pd-inline-val ${ivCls(atmIv)}">${atmIv != null ? atmIv.toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="IV minus HV" data-tip-body="Implied minus realised vol. Positive = options expensive vs recent moves.">
            <div class="pd-inline-lbl">IV − HV</div><div class="pd-inline-val ${atmIv != null && hv30 != null ? clsI(atmIv - hv30) : ''}">${atmIv != null && hv30 != null ? (atmIv > hv30 ? '+' : '') + (atmIv - hv30).toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="25d Risk Reversal · Saxo Bank (1M)" data-tip-body="25d call IV minus 25d put IV. Positive = calls bid, upside skew on ${base || label.split('/')[0]}. Negative = puts bid, downside protection dominant.">
            <div class="pd-inline-lbl">25d RR</div><div class="pd-inline-val ${clsI(rrVal)}">${rrVal != null ? (rrVal >= 0 ? '+' : '') + rrVal.toFixed(2) : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="Bid-Ask Spread" data-tip-body="Estimated interbank ECN spread in pips.">
            <div class="pd-inline-lbl">Spread</div><div class="pd-inline-val">${spreadPips != null ? spreadPips.toFixed(1) + ' pip' : '—'}</div>
          </div>
        </div>
      </div>

      <div class="pd-inline-group">
        ${isCrossPair ? '' : '<div class="pd-inline-group-lbl">COT Positioning</div>'}
        ${(() => {
          // Helper: render one 4-metric COT block for a given currency
          const cotBlock = (ccy, net, wow, amNet, pctOI, isCross, addTopBorder) => {
            const crossNote = isCross ? ` CFTC tracks ${ccy} vs USD — use as ${ccy} sentiment proxy for this cross.` : '';
            const lfD = net == null ? null : net > 0 ? 'Long' : net < 0 ? 'Short' : null;
            const amD = amNet == null ? null : amNet > 0 ? 'Long' : amNet < 0 ? 'Short' : null;
            const alignedStr = lfD && amD ? (lfD === amD
              ? `<span style="color:var(--text3);font-size:8px;"> · aligned</span>`
              : `<span style="color:var(--text3);font-size:8px;"> · diverging</span>`) : '';
            const summaryLine = lfD
              ? `<div style="margin-top:4px;font-size:9px;font-family:var(--font-mono);">` +
                `<span class="${clsI(net)}">${lfD}</span><span style="color:var(--text3);font-size:8px;"> LF</span>` +
                (amD ? ` · <span class="${clsI(amNet)}">${amD}</span><span style="color:var(--text3);font-size:8px;"> AM</span>${alignedStr}` : '') +
                `</div>` : '';
            return `
            ${isCross ? `<div style="font-size:8px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);padding:4px 0 3px;${addTopBorder ? 'border-top:1px solid var(--border);margin-top:4px;' : ''}">COT ${ccy}</div>` : (addTopBorder ? `<div style="border-top:1px solid var(--border);margin-top:4px;"></div>` : '')}
            <div class="pd-inline-metrics">
              <div class="pd-inline-metric fx-tip" data-tip-title="CFTC Leveraged Funds Net${isCross ? ` · ${ccy}` : ''}" data-tip-body="Net contracts (longs minus shorts) held by Leveraged Funds — hedge funds and CTAs.${crossNote}" data-tip-ex="Extreme net long historically precedes reversals as the speculative crowd becomes crowded.">
                <div class="pd-inline-lbl">LF Net</div><div class="pd-inline-val ${clsI(net)}">${fmtN(net)}</div>
              </div>
              <div class="pd-inline-metric fx-tip" data-tip-title="LF Week-over-Week Change${isCross ? ` · ${ccy}` : ''}" data-tip-body="Change in LF net contracts vs prior week. Primary momentum signal in institutional COT analysis." data-tip-ex="Reversal in WoW change is often the earliest signal of a positioning shift.">
                <div class="pd-inline-lbl">LF WoW Δ</div><div class="pd-inline-val ${clsI(wow)}">${fmtN(wow)}</div>
              </div>
              <div class="pd-inline-metric fx-tip" data-tip-title="Asset Managers Net${isCross ? ` · ${ccy}` : ''}" data-tip-body="Net contracts held by Asset Managers — pension funds, mutual funds. Structural positioning.${crossNote}" data-tip-ex="Divergence between LF and AM often signals a positioning squeeze.">
                <div class="pd-inline-lbl">AM Net</div><div class="pd-inline-val ${clsI(amNet)}">${fmtN(amNet)}</div>
              </div>
              <div class="pd-inline-metric fx-tip" data-tip-title="LF Net as % of OI${isCross ? ` · ${ccy}` : ''}" data-tip-body="LF net divided by LF Open Interest. Normalises positioning across currencies for direct comparison.${crossNote}" data-tip-ex="+15% = LF hold net long equivalent to 15% of total OI — historically a crowded position.">
                <div class="pd-inline-lbl">Net % OI</div><div class="pd-inline-val ${clsI(pctOI)}">${pctOI != null ? (pctOI > 0 ? '+' : '') + pctOI.toFixed(1) + '%' : '—'}</div>
              </div>
            </div>
            ${summaryLine}`;
          };

          if (isCrossPair && cotCcy2 && cotRaw2) {
            return `
            ${cotBlock(cotCcy, cotNet, cotWow, cotAmNet, cotPctOI, true, false)}
            ${cotBlock(cotCcy2, cot2Net, cot2Wow, cot2AmNet, cot2PctOI, true, true)}`;
          } else {
            return cotBlock(cotCcy, cotNet, cotWow, cotAmNet, cotPctOI, false, false);
          }
        })()}
      </div>

      <div class="pd-inline-group pd-inline-group--retail fx-tip"
        data-tip-title="Retail Client Positioning · Myfxbook"
        data-tip-body="Long/short % from Myfxbook community (retail traders only). Contrarian indicator — extreme retail long bias historically aligns with institutional short positioning. Avg entry shows where the dominant side opened; if price has moved against them, a stop-hunt or squeeze becomes more likely."
        data-tip-ex="Heavily Long (>65%) = contrarian bearish signal. Heavily Short (<35% long) = contrarian bullish signal. Avg entry underwater = retail under pressure, reversal risk elevated."
        style="border-right:none; justify-content:flex-start;">
        <div class="pd-inline-group-lbl">Retail <span class="pd-inline-retail-skew ${retSkewCls}">${retSkew || ''}</span></div>
        <div class="pd-inline-retail-bar" style="margin-bottom:3px;"><div class="pd-inline-retail-fill" style="width:${retL != null ? retL : 50}%"></div></div>
        <div class="pd-inline-retail-row">
          <span class="pd-inline-val ${retL != null && retL >= 65 ? 'pd-dn' : retL != null && retL <= 35 ? 'pd-up' : ''}">${retL != null ? retL + '% L' : '—'}</span>
          <span class="pd-dim"> / </span>
          <span>${retS != null ? retS + '% S' : '—'}</span>
        </div>
        ${retAvgL != null || retAvgS != null ? `<div class="pd-inline-retail-avg">
          ${retAvgL != null && retL != null && retL >= 50 ? `<span class="pd-inline-lbl">Avg L </span><span class="pd-inline-val ${retLUnder ? 'pd-dn' : 'pd-up'}">${retAvgL.toFixed(dec)}${retLUnder ? ' ▼' : ' ▲'}</span>` : ''}
          ${retAvgS != null && retS != null && retS >= 50 ? `<span class="pd-inline-lbl">Avg S </span><span class="pd-inline-val ${retSUnder ? 'pd-dn' : 'pd-up'}">${retAvgS.toFixed(dec)}${retSUnder ? ' ▼' : ' ▲'}</span>` : ''}
        </div>` : ''}
      </div>
    </div>
    </div>
    <div class="pd-inline-footer">${footerSources}</div>`;

  // Attach #fx-tt tooltips to each metric cell
  container.querySelectorAll('.fx-tip').forEach(cell => {
    const title = cell.dataset.tipTitle || '';
    const body  = cell.dataset.tipBody  || '';
    const ex    = cell.dataset.tipEx    || '';
    if (!title && !body) return;
    attachRiskTip(cell, title, body, ex);
  });
}

// Floating panel triggered by double-click on any pair row (FX table or crosses).
// Anchors near the row, closes on Escape or outside-click.
function openPairPopover(rowEl, tvSym) {
  const pop = document.getElementById('pd-popover');
  if (!pop) return;

  // If same pair is already open, close it (toggle)
  if (pop.dataset.sym === tvSym && pop.style.display !== 'none') {
    closePairPopover();
    return;
  }

  pop.dataset.sym = tvSym;

  // Render off-screen first to measure real dimensions
  pop.style.visibility = 'hidden';
  pop.style.display = 'block';
  pop.style.left = '0px';
  pop.style.top  = '0px';

  updatePairDetail(tvSym);

  // After paint: read real size and clamp within viewport
  requestAnimationFrame(() => {
    // On mobile the CSS converts the popover into a bottom sheet — no JS positioning needed
    if (window.innerWidth <= 900) {
      pop.style.left = '';
      pop.style.top  = '';
      pop.style.visibility = 'visible';
      return;
    }

    const rect = rowEl.getBoundingClientRect();
    const popRect = pop.getBoundingClientRect();
    const pw = popRect.width  || 270;
    const ph = popRect.height || 400;
    const vw = window.innerWidth, vh = window.innerHeight;
    const GAP = 6, MARGIN = 8;

    // Prefer right of row; fall back to left if it would overflow
    let x = rect.right + GAP;
    if (x + pw > vw - MARGIN) x = rect.left - pw - GAP;
    if (x < MARGIN) x = MARGIN;

    // Align top of popup with row; shift up if it overflows bottom
    let y = rect.top;
    if (y + ph > vh - MARGIN) y = vh - ph - MARGIN;
    if (y < MARGIN) y = MARGIN;

    pop.style.left = x + 'px';
    pop.style.top  = y + 'px';
    pop.style.visibility = 'visible';
  });
}

function closePairPopover() {
  const pop = document.getElementById('pd-popover');
  if (pop) { pop.style.display = 'none'; pop.dataset.sym = ''; }
}

// Close on outside click
document.addEventListener('click', e => {
  const pop = document.getElementById('pd-popover');
  if (!pop || pop.style.display === 'none') return;
  if (!pop.contains(e.target)) closePairPopover();
}, true);

// Close on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closePairPopover();
});

// ── Sidebar crosses: single click → chart + inline detail (same pattern as majors table) ──
document.getElementById('sidebar')?.addEventListener('click', e => {
  const row = e.target.closest('.sb-row[data-sym]');
  if (!row) return;
  loadTVChart(row.dataset.sym);
  toggleSidebarDetail(row);
});

function toggleSidebarDetail(row) {
  const tvSym  = row.dataset.sym;
  const sidebar = row.closest('#sidebar');
  if (!sidebar) return;

  // If this row is already open, collapse it
  const existing   = sidebar.querySelector('.sb-expand-row');
  const wasThisRow = existing?.dataset.forSym === tvSym;

  if (existing) {
    const inner = existing.querySelector('.sb-expand-inner');
    if (inner) inner.style.maxHeight = '0';
    setTimeout(() => existing.remove(), 220);
    sidebar.querySelector('.sb-row.sb-selected')?.classList.remove('sb-selected');
  }
  if (wasThisRow) return;

  row.classList.add('sb-selected');

  const expandDiv = document.createElement('div');
  expandDiv.className = 'sb-expand-row';
  expandDiv.dataset.forSym = tvSym;
  const inner = document.createElement('div');
  inner.className = 'sb-expand-inner';
  inner.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Loading…</div>';
  expandDiv.appendChild(inner);
  row.after(expandDiv);

  // Animate open after next paint
  requestAnimationFrame(() => {
    inner.style.maxHeight = '600px'; // generous — content drives real height
  });

  buildInlineDetail(tvSym, inner);
}

// ── FX Pairs table: click = chart + expand detail inline ──────────────────
document.getElementById('fx-pairs-tbody')?.addEventListener('click', e => {
  const row = e.target.closest('tr[data-sym]');
  if (!row) return;
  loadTVChart(row.dataset.sym);
  toggleInlineDetail(row);
});

// ── Cross-Asset cells: click to open chart (US 10Y excluded — no TV symbol) ──
document.querySelectorAll('#cross-asset-grid .ca-cell[data-sym]').forEach(cell => {
  cell.addEventListener('click', function() {
    loadTVChart(this.dataset.sym);
  });
});

// ── Risk Monitor VIX cell: click to open chart ──
document.getElementById('risk-vix')?.closest('.risk-cell')?.addEventListener('click', () => {
  loadTVChart('CBOE:VIX');
});

// ── Risk Monitor MOVE cell: click to open chart ──
document.getElementById('risk-move')?.closest('.risk-cell')?.addEventListener('click', () => {
  loadTVChart('TVC:MOVE');
});

// ═══════════════════════════════════════════════════════════════════
// PAIR DETAIL PANEL — Eikon-style linked panel, updates #pair-detail on every pair click
// All data read from in-memory caches — zero additional fetches on click.
// ═══════════════════════════════════════════════════════════════════
const COT_DATA_CACHE = {};   // ccy → { net, long, short, amNet, weekEnding, prevOI, wowNetChange, totalOI, levNetPctOI }
const RR_DATA_CACHE  = {};   // rrKey (e.g. 'EURUSD') → { rr25d: number } — populated by fetchOptionSkew()

(async function prefetchCOT() {
  const CCYS = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];
  await Promise.all(CCYS.map(async ccy => {
    try {
      const r = await fetch('./cot-data/' + ccy + '.json');
      if (!r.ok) return;
      const d = await r.json();
      // prevOI and wowNetChange from history (history sorted oldest→newest)
      let prevOI = null;
      let wowNetChange = d.wowNetChange ?? null;
      if (Array.isArray(d.history) && d.history.length >= 2) {
        const prev = d.history[d.history.length - 2]; // prior week
        if (prev.levLong != null && prev.levShort != null)
          prevOI = prev.levLong + prev.levShort;
        // Derive WoW if not in root
        if (wowNetChange == null && d.netPosition != null) {
          const prevNet = prev.levNet ?? ((prev.levLong || 0) - (prev.levShort || 0));
          wowNetChange = d.netPosition - prevNet;
        }
      }
      // Derive levNetPctOI if not in root
      const levOI = (d.longPositions || 0) + (d.shortPositions || 0);
      const levNetPctOI = d.levNetPctOI ?? (levOI > 0 ? Math.round(d.netPosition / levOI * 1000) / 10 : null);
      COT_DATA_CACHE[ccy] = {
        net:          d.netPosition    ?? null,
        long:         d.longPositions  ?? null,
        short:        d.shortPositions ?? null,
        amNet:        d.assetManagerNet ?? null,
        weekEnding:   d.weekEnding || d.reportDate || '',
        prevOI,
        wowNetChange,
        totalOI:       d.totalOpenInterest ?? null,
        levNetPctOI,
      };
    } catch {}
  }));
})();

function pairMetaFromSym(tvSym) {
  const raw = tvSym.replace(/^(FX_IDC:|FX:|CAPITALCOM:)/i, '').toLowerCase();
  return PAIRS.find(x => x.id === raw
    || (x.base + x.quote).toLowerCase() === raw
    || (x.quote + x.base).toLowerCase() === raw) || null;
}

async function updatePairDetail(tvSym) {
  const panel = document.getElementById('pd-popover');
  if (!panel) return;

  // Ensure #fx-tt tooltip engine is initialised (may not exist if sentiment hasn't loaded yet)
  if (!document.getElementById('fx-tt')) {
    const s = document.createElement('style');
    s.id = 'fx-tt-style';
    s.textContent = `#fx-tt{position:fixed;z-index:99999;width:min(240px,calc(100vw - 24px));background:var(--bg3);border:1px solid var(--border2);border-radius:4px;padding:9px 11px;font-size:11px;color:var(--text);line-height:1.55;pointer-events:none;display:none;font-family:var(--font-ui);box-sizing:border-box;}#fx-tt .tt-title{font-weight:700;font-size:11px;color:#fff;margin-bottom:3px;}#fx-tt .tt-ex{margin-top:5px;padding-top:5px;border-top:1px solid var(--border2);font-size:10px;color:var(--text2);font-style:italic;}.fx-tip{cursor:help;}`;
    document.head.appendChild(s);
    const ttEl = document.createElement('div');
    ttEl.id = 'fx-tt';
    ttEl.innerHTML = '<div class="tt-title" id="fx-tt-title"></div><div id="fx-tt-body"></div><div class="tt-ex" id="fx-tt-ex"></div>';
    document.body.appendChild(ttEl);
    window._fxTTPos = function(cx, cy) {
      const tt = document.getElementById('fx-tt');
      if (!tt) return;
      const vw = window.innerWidth, vh = window.innerHeight;
      const ttW = Math.min(240, vw - 24), ttH = tt.offsetHeight || 80, PAD = 8;
      let x = cx + 14, y = cy + 14;
      if (x + ttW > vw - PAD) x = cx - ttW - 8;
      if (x < PAD) x = PAD;
      if (y + ttH > vh - PAD) y = cy - ttH - 8;
      if (y < PAD) y = PAD;
      tt.style.left = x + 'px'; tt.style.top = y + 'px';
    };
    document.addEventListener('mousemove', ev => {
      const tt = document.getElementById('fx-tt');
      if (tt && tt.style.display === 'block') window._fxTTPos(ev.clientX, ev.clientY);
    });
  }

  const meta   = pairMetaFromSym(tvSym);
  const label  = meta?.label || tvSym.replace(/^.*:/,'').replace(/(.{3})(.{3})/,'$1/$2').toUpperCase();
  const pairId = meta?.id || null;
  const base   = meta?.base  || null;
  const quote  = meta?.quote || null;
  const invert = meta?.invert ?? false;
  const dec    = meta?.dec   ?? 5;

  const rt    = pairId ? STOOQ_RT_CACHE[pairId] : null;
  const price = rt?.close ?? null;
  const pct1d = rt?.pct   ?? null;
  const hv30  = rt?.hv30  ?? null;
  const sessH = rt?.high  ?? null;
  const sessL = rt?.low   ?? null;

  // 1W from quotes.json pct1w field (prior-Friday-close convention, same source as FX table)
  let pct1w = null;
  if (rt?.pct1w != null) {
    pct1w = rt.pct1w;
  }

  // ATM IV — direct ETF option chain for 6 USD majors; synthesised via triangulation for 21 crosses.
  // Cross formula: IV_AB ≈ √(IV_A² + IV_B² − 2·ρ·IV_A·IV_B)
  // ρ values are long-run empirical FX vol correlations (conservative, rounded to nearest 0.05).
  const CROSS_IV_RHO = {
    'eurgbp':0.65,'eurjpy':0.55,'eurchf':0.60,'eurcad':0.40,'euraud':0.35,'eurnzd':0.30,
    'gbpjpy':0.45,'gbpchf':0.55,'gbpcad':0.30,'gbpaud':0.25,'gbpnzd':0.20,
    'audjpy':0.40,'audnzd':0.55,'audchf':0.30,'audcad':0.50,
    'cadjpy':0.35,'cadchf':0.25,'chfjpy':0.40,
    'nzdjpy':0.35,'nzdcad':0.45,'nzdchf':0.20,
  };
  const USD_IV = {}; // non-USD ccy → IV%
  let atmIv = null;
  let nzdProxy = false;
  try {
    const intra = await loadIntradayQuotes();
    const etfIv = intra?.fx_etf_iv || {};
    // Build USD_IV map from available ETF option data
    for (const [pid, entry] of Object.entries(etfIv)) {
      if (entry?.iv == null) continue;
      const p = PAIRS.find(x => x.id === pid);
      if (!p) continue;
      const nonUsd = p.base !== 'USD' ? p.base : p.quote;
      USD_IV[nonUsd] = entry.iv;
    }
    // NZD proxy: no CBOE-listed NZD ETF options. Derive from AUD IV × 1.08 (long-run NZD/AUD vol ratio).
    if (USD_IV['AUD'] != null && USD_IV['NZD'] == null) {
      USD_IV['NZD'] = Math.round(USD_IV['AUD'] * 1.08 * 10) / 10;
      nzdProxy = true;
    }

    // Direct ETF IV for USD majors
    const ivEntry = etfIv[pairId];
    if (ivEntry?.iv != null) {
      atmIv = ivEntry.iv;
    } else if (pairId && meta?.cross) {
      // Synthesise cross IV from component USD-pair IVs
      const ivA = USD_IV[base]  ?? null;
      const ivB = USD_IV[quote] ?? null;
      if (ivA != null && ivB != null) {
        const rho = CROSS_IV_RHO[pairId] ?? 0.40;
        atmIv = Math.round(Math.sqrt(ivA * ivA + ivB * ivB - 2 * rho * ivA * ivB) * 10) / 10;
      }
    }
  } catch {}

  // COT — for crosses, load BOTH component currencies
  const isCrossPair = !!meta?.cross;
  const cotCcy = base && base !== 'USD' ? base : (quote && quote !== 'USD' ? quote : base);
  const cotRaw = cotCcy ? (COT_DATA_CACHE[cotCcy] || null) : null;
  // Second COT ccy for crosses (quote when base ≠ USD, else null for majors)
  const cotCcy2 = isCrossPair && quote && quote !== cotCcy ? quote : null;
  const cotRaw2 = cotCcy2 ? (COT_DATA_CACHE[cotCcy2] || null) : null;
  let cotNet = null, cotAmNet = null, cotOI = null, cotPrevOI = null, cotWeek = '';
  let cotWow = null, cotPctOI = null, cotTotalOI = null;
  if (cotRaw) {
    const flip = (invert && cotCcy === quote) ? -1 : 1;
    cotNet      = cotRaw.net   != null ? cotRaw.net   * flip : null;
    cotAmNet    = cotRaw.amNet != null ? cotRaw.amNet * flip : null;
    cotWow      = cotRaw.wowNetChange != null ? cotRaw.wowNetChange * flip : null;
    cotPctOI    = cotRaw.levNetPctOI  != null ? cotRaw.levNetPctOI  * flip : null;
    cotTotalOI  = cotRaw.totalOI      ?? null;
    // OI = LF longs + LF shorts (futures+options combined, LF category)
    if (cotRaw.long != null && cotRaw.short != null)
      cotOI = cotRaw.long + cotRaw.short;
    cotPrevOI = cotRaw.prevOI ?? null;
    cotWeek   = cotRaw.weekEnding;
  }
  // Second COT block — quote currency of cross pair (e.g. JPY in GBP/JPY)
  let cot2Net = null, cot2AmNet = null, cot2Wow = null, cot2PctOI = null, cot2OI = null;
  if (cotRaw2) {
    cot2Net    = cotRaw2.net          ?? null;
    cot2AmNet  = cotRaw2.amNet        ?? null;
    cot2Wow    = cotRaw2.wowNetChange ?? null;
    cot2PctOI  = cotRaw2.levNetPctOI  ?? null;
    if (cotRaw2.long != null && cotRaw2.short != null)
      cot2OI = cotRaw2.long + cotRaw2.short;
    if (!cotWeek && cotRaw2.weekEnding) cotWeek = cotRaw2.weekEnding;
  }

  // Carry differential (CB rates)
  // For USD major pairs:
  //   invert:true  = CCY/USD pair (EUR/USD) → numerator = base (EUR) → carry = cbBase − cbQuote
  //   invert:false = USD/CCY pair (USD/JPY) → numerator = USD (quote) → carry = cbQuote − cbBase
  // For cross pairs (no invert field):
  //   The pair label is always BASE/QUOTE (e.g. AUD/CHF), so numerator = base
  //   carry = cbBase − cbQuote  (AUD rate − CHF rate = 4.10% − 0% = +4.10%)
  //   Using meta.cross to detect cross pairs and always apply cbBase − cbQuote.
  // OIS rate preferred over CB policy rate (Bloomberg standard for carry display).
  // _resolveRate() returns [rate, source] — OIS if loaded, policy rate as fallback.
  const [_oisBase,  ]  = (typeof _resolveRate === 'function' && base)  ? _resolveRate(base)  : [null];
  const [_oisQuote, ]  = (typeof _resolveRate === 'function' && quote) ? _resolveRate(quote) : [null];
  const cbBase  = _oisBase  ?? (base  ? (STATE.cbRates?.[base.toLowerCase()]?.rate  ?? null) : null);
  const cbQuote = _oisQuote ?? (quote ? (STATE.cbRates?.[quote.toLowerCase()]?.rate ?? null) : null);
  let carryDiff = null;
  if (cbBase != null && cbQuote != null) {
    if (meta?.cross) {
      // Cross pair: base is always the numerator currency in the pair label
      carryDiff = cbBase - cbQuote;
    } else {
      carryDiff = invert ? (cbBase - cbQuote) : (cbQuote - cbBase);
    }
  }

  const fmtPct = v => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
  const fmtNet = v => v == null ? '—' : (v >= 0 ? '+' : '') + Math.round(v).toLocaleString();
  const cls    = v => v == null ? '' : v > 0 ? 'pd-up' : v < 0 ? 'pd-dn' : '';

  // Spread
  const spreadPips = pairId ? TYPICAL_SPREADS[pairId] : null;

  // ADR — derived from HV30: daily range ≈ close × (HV30/100) / √252, converted to pips
  let adr = null;
  if (hv30 != null && price != null) {
    const pipSize = dec === 3 ? 0.01 : 0.0001; // JPY pairs have 3 decimals, pip = 0.01
    adr = Math.round(price * (hv30 / 100) / Math.sqrt(252) / pipSize);
  }

  // Retail sentiment from myfxbook cache
  const retKey = label.replace('/', '/').toUpperCase();
  const ret = RETAIL_SENTIMENT_CACHE[retKey] || null;
  const retL = ret?.longPct ?? null;
  const retS = ret?.shortPct ?? null;
  const retBarL = retL != null ? retL : 50;

  // 25d Risk Reversal — from RR_DATA_CACHE (populated by fetchOptionSkew)
  // rrKey = base+quote uppercase, no slash (e.g. EURUSD, USDJPY)
  // Saxo data covers the 7 majors in the Positioning Bias table; crosses show '—'
  const rrKey  = base && quote ? (base + quote).toUpperCase() : null;
  const rrVal  = rrKey ? (RR_DATA_CACHE[rrKey]?.rr25d ?? null) : null;
  // Direction label from base-currency perspective (same convention as RR chip in positioning table)
  const rrBase = base || (label.split('/')[0] || '');

  // COT positioning summary text (replaces badge)
  let cotSummaryHtml = '';
  if (isCrossPair && cotCcy2) {
    // Cross: show one line per component currency
    const parts = [];
    for (const [ccy, net, amNet] of [[cotCcy, cotNet, cotAmNet], [cotCcy2, cot2Net, cot2AmNet]]) {
      if (net == null) continue;
      const lfDir = net > 0 ? 'Long' : net < 0 ? 'Short' : null;
      const amDir = amNet != null ? (amNet > 0 ? 'Long' : amNet < 0 ? 'Short' : null) : null;
      const lfCls = net > 0 ? 'pd-up' : 'pd-dn';
      const amCls = amNet != null ? (amNet > 0 ? 'pd-up' : 'pd-dn') : '';
      const amPart = amDir ? ` · AM <span class="${amCls}">${amDir}</span>` : '';
      const aligned = lfDir && amDir ? (lfDir === amDir ? ' · <span class="pd-dim">aligned</span>' : ' · <span class="pd-dim">diverging</span>') : '';
      if (lfDir) parts.push(`<span class="pd-dim" style="font-size:9px;text-transform:uppercase;letter-spacing:.04em;">${ccy}</span> LF <span class="${lfCls}">${lfDir}</span>${amPart}${aligned}`);
    }
    if (parts.length) cotSummaryHtml = `<div class="pd-cot-summary">${parts.join('<span class="pd-dim"> · </span>')}</div>`;
  } else if (cotNet != null && cotAmNet != null) {
    const lfDir = cotNet > 0 ? 'Long' : cotNet < 0 ? 'Short' : null;
    const amDir = cotAmNet > 0 ? 'Long' : cotAmNet < 0 ? 'Short' : null;
    if (lfDir && amDir) {
      const aligned = lfDir === amDir;
      const lfCls   = cotNet   > 0 ? 'pd-up' : 'pd-dn';
      const amCls   = cotAmNet > 0 ? 'pd-up' : 'pd-dn';
      const alignStr = aligned ? 'aligned' : 'diverging';
      cotSummaryHtml = `<div class="pd-cot-summary">LF <span class="${lfCls}">${lfDir}</span> · AM <span class="${amCls}">${amDir}</span> · <span class="pd-dim">${alignStr}</span></div>`;
    }
  }

  panel.innerHTML = `
    <div class="pd-header">
      <span class="pd-sym">${label}</span>
      <button class="pd-close" onclick="closePairPopover()" aria-label="Close pair detail">&#x2715;</button>
    </div>

    <div class="pd-price-block">
      <div class="pd-price-row">
        <div class="pd-price ${price == null ? 'pd-dim' : ''}">${price != null ? price.toFixed(dec) : '—'}</div>
        <span class="${cls(pct1d)} pd-chg">${fmtPct(pct1d)}</span>
      </div>
      ${sessH != null && sessL != null ? `<div class="pd-range">H ${sessH.toFixed(dec)} · L ${sessL.toFixed(dec)}</div>` : ''}
      <div class="pd-spread-row">${spreadPips != null ? 'Spread ' + spreadPips.toFixed(1) + ' pip' : ''}${spreadPips != null && adr != null ? ' · ' : ''}${adr != null ? 'ADR ' + adr + ' pip' : ''}</div>
    </div>

    <div class="pd-section">
      <div class="pd-section-lbl">Price</div>
      <div class="pd-grid">
        <div class="pd-cell fx-tip" data-tip-title="1-Week Change" data-tip-body="Weekly % change vs prior Friday close. Source: FX performance cache."><div class="pd-lbl">1W Chg</div><div class="pd-val ${cls(pct1w)}">${fmtPct(pct1w)}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Carry Differential" data-tip-body="OIS overnight rate differential (SOFR/€STR/SONIA/TONA/CORRA/SARON — institutional overnight benchmarks). Falls back to CB policy rate when OIS data unavailable. Positive = base currency yields more, carry favours long." data-tip-ex="Positive carry = the long leg earns more than it costs to fund the short. OIS reflects actual overnight funding cost — more accurate than CB policy rate for carry calculations. Carry is most reliable as a persistent trend signal; it can reverse quickly on policy surprises."><div class="pd-lbl">Carry</div><div class="pd-val ${cls(carryDiff)}">${carryDiff != null ? (carryDiff >= 0 ? '+' : '') + carryDiff.toFixed(2)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Average Daily Range" data-tip-body="Estimated average daily range in pips, derived from HV 30d: close × (HV / √252). Indicates typical intraday movement — useful for stop and target sizing." data-tip-ex="ADR of 85 pip on EUR/USD means the pair moves ~85 pip on an average day."><div class="pd-lbl">ADR</div><div class="pd-val">${adr != null ? adr + ' pip' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="${base || 'Base'} Policy Rate" data-tip-body="${base || 'Base'} central bank policy rate (annualised). Source: CB rates cache."><div class="pd-lbl">${base || 'Base'} Rate</div><div class="pd-val">${cbBase != null ? cbBase.toFixed(2)+'%' : '—'}</div></div>
      </div>
    </div>

    <div class="pd-section">
      <div class="pd-section-lbl">Volatility</div>
      <div class="pd-grid">
        <div class="pd-cell fx-tip" data-tip-title="Historical Volatility 30d" data-tip-body="30-day realised (historical) volatility, annualised. Measures how much the pair has actually moved recently. Low HV = quiet market; high HV = volatile market."><div class="pd-lbl">HV 30d</div><div class="pd-val">${hv30 != null ? hv30.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="ATM Implied Volatility${(meta?.cross || nzdProxy) && atmIv != null ? ' (estimated)' : ''}" data-tip-body="${meta?.cross && atmIv != null ? 'Synthesised from component USD-pair CBOE/CME vol index values via triangulation: √(IVa²+IVb²−2ρ·IVa·IVb). Proxy for OTC interbank IV — indicative only.' : nzdProxy && atmIv != null ? 'Estimated from AUD/USD CBOE/CME vol index (^AUDVIX) × 1.08 (long-run NZD/AUD realised vol ratio). No dedicated CBOE/CME NZD vol index exists — treat as directional context only.' : 'ATM implied vol from CBOE/CME FX Volatility Index (^EUVIX/^BPVIX/^JYVIX/^AUDVIX) — same variance-swap methodology as VIX, published jointly by CBOE and CME. Institutional benchmark for FX options pricing. CHF/CAD: CME futures options or CBOE ETF fallback.'} Color = cost of hedging: green ≤7% (cheap), red >12% (expensive). Not a directional signal."><div class="pd-lbl">ATM IV${(meta?.cross || nzdProxy) && atmIv != null ? '<span style="font-size:8px;color:var(--text3);margin-left:2px;">~</span>' : ''}</div><div class="pd-val ${atmIv != null ? (atmIv > 12 ? 'pd-dn' : atmIv > 7 ? '' : 'pd-up') : ''}">${atmIv != null ? atmIv.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="IV minus HV" data-tip-body="Implied vol minus realised vol. Positive = options are expensive relative to recent moves (market pricing in risk premium). Negative = options are cheap vs realised. Not a directional signal." data-tip-ex="IV−HV > +3% historically indicates options are pricing in a premium above recent realised moves — hedging costs are elevated relative to actual market movement."><div class="pd-lbl">IV − HV</div><div class="pd-val ${atmIv != null && hv30 != null ? cls(atmIv - hv30) : ''}">${atmIv != null && hv30 != null ? (atmIv > hv30 ? '+' : '') + (atmIv - hv30).toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="25-delta Risk Reversal (1M) · Saxo Bank" data-tip-body="25d RR = 25d call IV minus 25d put IV. Positive = calls bid over puts — market skewed for upside on ${rrBase}. Negative = puts bid — downside protection dominant. Source: Saxo Bank public options page, 1M tenor, indicative mid-market. Updated during European hours." data-tip-ex="RR is a directional skew signal, not a vol-level signal. A strongly negative RR alongside high ATM IV = market pricing in both expensive hedging AND downside risk — historically a high-conviction bearish setup."><div class="pd-lbl">25d RR</div><div class="pd-val ${rrVal != null ? cls(rrVal) : ''}">${rrVal != null ? (rrVal >= 0 ? '+' : '') + rrVal.toFixed(2) : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Bid-Ask Spread" data-tip-body="Estimated interbank ECN spread in pips. Dynamically adjusted for current volatility conditions — wider during high-vol sessions and around news events. Lower spread = more liquid." data-tip-ex="EUR/USD typically trades 0.1–0.3 pip during London/NY overlap. Spreads widen significantly in the Asian session and around data releases."><div class="pd-lbl">Spread</div><div class="pd-val">${spreadPips != null ? spreadPips.toFixed(1) + ' pip' : '—'}</div></div>
      </div>
    </div>

    <div class="pd-section">
      ${isCrossPair ? '' : '<div class="pd-section-lbl">COT Positioning</div>'}
      ${(() => {
        // Helper: render one COT block for a given currency (for the popover grid layout)
        const cotBlockGrid = (ccy, net, wow, amNet, pctOI, oi, prevOI, isCross, addTopBorder) => {
          const crossNote = isCross ? ` CFTC tracks ${ccy} futures vs USD — not this cross specifically. Use as ${ccy} sentiment proxy.` : '';
          const oiDelta    = (prevOI != null && oi != null) ? oi - prevOI : null;
          const oiArrow    = oiDelta == null ? '' : oiDelta > 0 ? '<span class="pd-oi-up">▲</span> ' : oiDelta < 0 ? '<span class="pd-oi-dn">▼</span> ' : '';
          const oiDeltaStr = oiDelta == null ? '' : ` <span class="pd-dim" style="font-size:9px;">(${oiDelta > 0 ? '+' : ''}${Math.round(oiDelta).toLocaleString()})</span>`;
          return `
            ${isCross ? `<div class="pd-cell pd-cell--wide pd-section-lbl" style="${addTopBorder ? 'border-top:1px solid var(--border);margin-top:2px;' : ''}">COT ${ccy}</div>` : ''}
            <div class="pd-cell fx-tip"
              data-tip-title="CFTC Leveraged Funds Net${isCross ? ` · ${ccy}` : ''}"
              data-tip-body="Net contracts (longs minus shorts) held by Leveraged Funds — hedge funds and CTAs. Speculative / trend-following positioning. Source: CFTC Disaggregated TFF report.${crossNote}"
              data-tip-ex="Extreme LF net long positioning has historically preceded reversals as the speculative crowd becomes crowded.">
              <div class="pd-lbl">LF Net</div>
              <div class="pd-val ${cls(net)}">${fmtNet(net)}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="LF WoW Change${isCross ? ` · ${ccy}` : ''}"
              data-tip-body="Week-over-week change in Leveraged Funds net contracts. Positive = specs adding longs or covering shorts. Negative = specs adding shorts or reducing longs. The primary momentum signal in institutional COT analysis.${crossNote}"
              data-tip-ex="A large positive WoW change alongside rising net = conviction build-up. A reversal in WoW change is often the earliest signal of a positioning shift.">
              <div class="pd-lbl">LF WoW Δ</div>
              <div class="pd-val ${cls(wow)}">${wow != null ? (wow > 0 ? '+' : '') + Math.round(wow).toLocaleString() : '—'}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="CFTC Asset Managers Net${isCross ? ` · ${ccy}` : ''}"
              data-tip-body="Net contracts held by Asset Managers — pension funds, mutual funds, and institutional investors. Structural / longer-term positioning. Source: CFTC Disaggregated TFF report.${crossNote}"
              data-tip-ex="AM positioning tends to be more persistent than LF. Divergence between LF and AM can signal a positioning squeeze.">
              <div class="pd-lbl">AM Net</div>
              <div class="pd-val ${cls(amNet)}">${fmtNet(amNet)}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="LF Net as % of Total OI${isCross ? ` · ${ccy}` : ''}"
              data-tip-body="LF net contracts divided by LF Open Interest (long + short). Normalises positioning across currencies — EUR and JPY have very different raw contract counts; this makes them directly comparable.${crossNote}"
              data-tip-ex="+15% means Leveraged Funds hold a net long equivalent to 15% of the entire market's open interest — a heavily crowded position historically associated with reversal risk.">
              <div class="pd-lbl">Net % OI</div>
              <div class="pd-val ${cls(pctOI)}">${pctOI != null ? (pctOI > 0 ? '+' : '') + pctOI.toFixed(1) + '%' : '—'}</div>
            </div>
            ${oi != null ? `<div class="pd-cell pd-cell--wide fx-tip" style="${isCross ? '' : 'border-bottom:none;'}"
              data-tip-title="LF Open Interest${isCross ? ` · ${ccy}` : ''}"
              data-tip-body="Total open interest in the Leveraged Funds category: long + short contracts. Rising OI = new money entering; falling OI = positions closing. Source: CFTC TFF report.${crossNote}"
              data-tip-ex="${oiDelta != null ? `This week: ${oiDelta > 0 ? '▲' : oiDelta < 0 ? '▼' : '='} ${Math.abs(Math.round(oiDelta)).toLocaleString()} vs prior week. ${oiDelta > 0 ? 'New money entering — expanding participation.' : 'Positions being closed — shrinking participation.'}` : 'Expanding OI alongside rising net long = conviction build-up. Falling OI alongside persistent net = position unwinding.'}">
              <div class="pd-lbl">LF Open Interest</div>
              <div class="pd-val">${oiArrow}${Math.round(oi).toLocaleString()}${oiDeltaStr}</div>
            </div>` : ''}`;
        };

        const gridHtml = isCrossPair && cotCcy2 && cotRaw2
          ? cotBlockGrid(cotCcy, cotNet, cotWow, cotAmNet, cotPctOI, cotOI, cotPrevOI, true, false) +
            cotBlockGrid(cotCcy2, cot2Net, cot2Wow, cot2AmNet, cot2PctOI, null, null, true, true)
          : cotBlockGrid(cotCcy, cotNet, cotWow, cotAmNet, cotPctOI, cotOI, cotPrevOI, false, false);

        return `<div class="pd-grid">${gridHtml}</div>`;
      })()}
      ${cotSummaryHtml}
    </div>

    <div class="pd-section pd-section--last">
      <div class="pd-section-lbl">Retail Sentiment</div>
      <div class="pd-cell pd-cell--wide fx-tip" data-tip-title="Retail Client Positioning" data-tip-body="Long/short ratio from Myfxbook community outlook — retail traders only, not institutional. Contrarian indicator: extreme retail long bias historically aligns with institutional short positioning. Source: Myfxbook, updated every 30min." data-tip-ex="Extreme readings — above 70% long or below 30% long — have historically coincided with elevated positioning risk in the dominant direction. Retail extremes are one input among many; always cross-reference with COT and CB differential data.">
        <div class="pd-retail-bar"><div class="pd-retail-fill" style="width:${retBarL}%"></div></div>
        <div class="pd-retail-nums">${retL != null ? retL+'% L' : '—'}<span class="pd-retail-sep">/</span>${retS != null ? retS+'% S' : '—'}</div>
      </div>
    </div>

    <div class="pd-footer">
      ${cotWeek ? '<span class="pd-dim">COT ' + cotWeek + ' · Myfxbook' + (rrVal != null ? ' · Saxo RR' : '') + '</span>' : '<span class="pd-dim">Myfxbook' + (rrVal != null ? ' · Saxo RR' : '') + '</span>'}
    </div>`;


  // ── Attach #fx-tt tooltips to each .fx-tip cell ──────────────────────────
  if (window._fxTTPos) {
    panel.querySelectorAll('.fx-tip').forEach(cell => {
      const title = cell.dataset.tipTitle || '';
      const body  = cell.dataset.tipBody  || '';
      const ex    = cell.dataset.tipEx    || '';
      if (!title && !body) return;
      cell.addEventListener('mouseenter', ev => {
        const tt = document.getElementById('fx-tt');
        if (!tt) return;
        document.getElementById('fx-tt-title').textContent = title;
        document.getElementById('fx-tt-body').textContent  = body;
        const exEl = document.getElementById('fx-tt-ex');
        if (ex) { exEl.textContent = ex; exEl.style.display = 'block'; }
        else    { exEl.textContent = ''; exEl.style.display = 'none';  }
        tt.style.display = 'block';
        requestAnimationFrame(() => window._fxTTPos(ev.clientX, ev.clientY));
      });
      cell.addEventListener('mouseleave', () => {
        const tt = document.getElementById('fx-tt');
        if (tt) tt.style.display = 'none';
      });
    });
  }
}

// TV CHART TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════
document.querySelectorAll('.tv-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    loadTVChart(this.dataset.sym);
  });
});

// ── TradingView legend auto-minimize (MA 20, Close, Vol labels) ──────────
// The widget renders inside an iframe so we poll for the minimize buttons
// and click them. Runs on initial load and on each symbol change.
function minimizeTVLegend() {
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  const iframe = wrap.querySelector('iframe');
  if (!iframe) return;
  // Try accessing the iframe document (same-origin if TV embeds it same-domain, else blocked)
  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    if (!doc) return;
    // Click all legend item minimize/collapse buttons (aria-label or title contains "Minimize")
    const btns = doc.querySelectorAll(
      '[data-name="legend-source-item"] button[aria-label], ' +
      '.legendItemControls button, ' +
      '[class*="minimizeButton"], ' +
      '[class*="collapseButton"]'
    );
    btns.forEach(btn => { try { btn.click(); } catch(_){} });
  } catch(_) {
    // Cross-origin — can't access iframe internals, nothing we can do
  }
}
// Run once after initial widget loads (give it ~4s to render)
setTimeout(minimizeTVLegend, 4000);
// Pair detail popover opens only on user action (ⓘ button) — no auto-populate.
// ─────────────────────────────────────────────────────────────────────────

// ── HORIZONTAL SCROLL WITH MOUSE WHEEL (desktop) ─────────────────────────
// Converts vertical wheel events into horizontal scroll on designated bars
(function() {
  function addWheelScroll(el) {
    if (!el) return;
    el.addEventListener('wheel', function(e) {
      if (e.deltaY === 0) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    }, { passive: false });
  }
  addWheelScroll(document.getElementById('tv-pair-tabs'));
  addWheelScroll(document.getElementById('tv-ticker'));
  addWheelScroll(document.getElementById('quotebar-inner'));

  // Arrow visibility for tv-pair-tabs
  const tabs   = document.getElementById('tv-pair-tabs');
  const btnPrev = document.getElementById('tv-tabs-prev');
  const btnNext = document.getElementById('tv-tabs-next');
  function updateTabArrows() {
    if (!tabs || !btnPrev || !btnNext) return;
    const atStart = tabs.scrollLeft <= 2;
    const atEnd   = tabs.scrollLeft + tabs.clientWidth >= tabs.scrollWidth - 2;
    btnPrev.style.display = atStart ? 'none' : 'flex';
    btnNext.style.display = atEnd   ? 'none' : 'flex';
  }
  if (tabs) {
    tabs.addEventListener('scroll', updateTabArrows, { passive: true });
    setTimeout(updateTabArrows, 200);
  }

  // Arrow visibility for quote bar
  const ticker   = document.getElementById('tv-ticker');
  const qbPrev   = document.getElementById('qb-prev');
  const qbNext   = document.getElementById('qb-next');
  function updateQbArrows() {
    if (!ticker || !qbPrev || !qbNext) return;
    const atStart = ticker.scrollLeft <= 2;
    const atEnd   = ticker.scrollLeft + ticker.clientWidth >= ticker.scrollWidth - 2;
    qbPrev.style.display = atStart ? 'none' : 'flex';
    qbNext.style.display = atEnd   ? 'none' : 'flex';
  }
  if (ticker) {
    ticker.addEventListener('scroll', updateQbArrows, { passive: true });
    setTimeout(updateQbArrows, 400);
  }

  // Scroll click handlers — migrated from inline onclick= in index.html (CSP fix)
  if (btnPrev) btnPrev.addEventListener('click', () => tabs  && tabs.scrollBy({left: -200, behavior: 'smooth'}));
  if (btnNext) btnNext.addEventListener('click', () => tabs  && tabs.scrollBy({left:  200, behavior: 'smooth'}));
  if (qbPrev)  qbPrev.addEventListener('click',  () => ticker && ticker.scrollBy({left: -200, behavior: 'smooth'}));
  if (qbNext)  qbNext.addEventListener('click',  () => ticker && ticker.scrollBy({left:  200, behavior: 'smooth'}));
})();
// ─────────────────────────────────────────────────────────────────────────

// TOP NAV
document.querySelectorAll('.top-nav a').forEach(a => {
  a.addEventListener('click', function() {
    document.querySelectorAll('.top-nav a').forEach(x => x.classList.remove('active'));
    this.classList.add('active');
  });
});

// ═══════════════════════════════════════════════════════════════════
// CARRY TRADE RANKING — full G8 28-pair differential, left sidebar
// ═══════════════════════════════════════════════════════════════════
// Institutional-grade carry ranking
// and JP Morgan GBI conventions:
//
//   Primary sort:  carry-to-vol ratio = rate differential / HV30
//                  (vol-adjusted carry — the industry standard metric)
//   Secondary col: raw rate differential (basis for the bar width)
//   Regime badge:  ↑ hiking  ↓ cutting  → hold  for each leg,
//                  derived from computeCBTrend() — same logic as CB Rates panel
//   Tooltip:       long rate / short rate / HV30 / carry-to-vol
//
// HV30 source: intraday-data/quotes.json → hv30 field per pair (same
// source used by the main FX table and the pair detail popover).
// Falls back to gross differential ranking when HV30 unavailable.
// ═══════════════════════════════════════════════════════════════════
// CARRY TRADE RANKING — G8 · real carry · annualised
// ═══════════════════════════════════════════════════════════════════
// Institutional-grade carry ranking per Bloomberg FXFR / Refinitiv conventions:
//
//   Primary sort:  real carry = nominal OIS differential − ΔInflation expectations
//                  = realRate(long) − realRate(short)
//                  (carry adjusted for inflation — the standard institutional metric)
//                  NOTE: this is real carry, NOT Covered Interest Parity (CIP).
//                  True CIP uses FX forward points; this uses inflation differentials.
//   Tiebreak:      carry-to-vol ratio = real carry / HV30
//                  (vol-adjusted carry; Bloomberg carry screens use this for pair selection)
//   Last fallback: gross nominal differential (when extended-data unavailable)
//
//   Display:       rank · pair · nominal spread label · proportional bar · real carry value
//                  Bar width = proportional to top pair's real carry (or nominal fallback)
//                  Value coloring: ≥+0.5% green (carry positive after infl.) / ≤−0.5% red
//
//   Tooltip:       long rate / short rate / real carry / HV30 / click for real rate analysis
// ═══════════════════════════════════════════════════════════════════
async function fetchCarryRanking() {
  const G8 = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD'];

  // TradingView symbol for a given long/short ccy pair

  // Bloomberg/Refinitiv market convention: normalises a long/short pair to the
  // canonical display label regardless of which leg has the higher rate.
  // E.g. long=USD, short=NZD → "NZD/USD"; long=GBP, short=EUR → "EUR/GBP".
  // ISO 4217 priority: EUR > GBP > AUD > NZD > USD > CAD > CHF > JPY (for crosses).
  // Reference: Bloomberg FX convention table; Refinitiv Eikon pair naming.
  const BBGFX_BASE = {
    // Pairs where USD is quote (commodity / European currencies are base)
    'USD-EUR':'EUR', 'USD-GBP':'GBP', 'USD-AUD':'AUD', 'USD-NZD':'NZD',
    // USD is base vs funding currencies
    'USD-JPY':'USD', 'USD-CAD':'USD', 'USD-CHF':'USD',
    // EUR crosses (EUR always base)
    'EUR-GBP':'EUR', 'EUR-JPY':'EUR', 'EUR-CHF':'EUR', 'EUR-CAD':'EUR',
    'EUR-AUD':'EUR', 'EUR-NZD':'EUR',
    // GBP crosses
    'GBP-JPY':'GBP', 'GBP-CHF':'GBP', 'GBP-CAD':'GBP',
    'GBP-AUD':'GBP', 'GBP-NZD':'GBP',
    // AUD/NZD crosses
    'AUD-JPY':'AUD', 'AUD-CAD':'AUD', 'AUD-CHF':'AUD', 'AUD-NZD':'AUD',
    'NZD-JPY':'NZD', 'NZD-CHF':'NZD', 'NZD-CAD':'NZD',
    // Remaining
    'CAD-JPY':'CAD', 'CHF-JPY':'CHF', 'CAD-CHF':'CAD',
  };
  function carryDisplayPair(long, short) {
    const key1 = long + '-' + short, key2 = short + '-' + long;
    const base = BBGFX_BASE[key1] ?? BBGFX_BASE[key2] ?? long;
    const quote = base === long ? short : long;
    return base + '/' + quote;
  }
  function carryTV(long, short) {
    // v8.4.5: delegate to carryDisplayPair() for Bloomberg market convention.
    // Ensures FX_IDC:NZDUSD (not FX_IDC:USDNZD), FX_IDC:EURGBP (not FX_IDC:GBPEUR), etc.
    const label = carryDisplayPair(long, short);  // e.g. "NZD/USD"
    return 'FX_IDC:' + label.replace('/', '');    // → "FX_IDC:NZDUSD"
  }

  // Canonical pair ID used in quotes.json / hv30 map — FX market convention,
  // not alphabetical for crosses (e.g. EUR/AUD = 'euraud', GBP/CHF = 'gbpchf').
  function pairId(a, b) {
    const HV30_PAIRS = new Set([
      'eurusd','gbpusd','usdjpy','audusd','usdchf','usdcad','nzdusd',
      'eurgbp','eurjpy','eurchf','eurcad','euraud',
      'gbpjpy','gbpchf','gbpcad',
      'audjpy','audnzd','audchf',
      'cadjpy','chfjpy','nzdjpy',
      'eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf',
    ]);
    const c1 = (a + b).toLowerCase();
    const c2 = (b + a).toLowerCase();
    if (HV30_PAIRS.has(c1)) return c1;
    if (HV30_PAIRS.has(c2)) return c2;
    return a < b ? c1 : c2;
  }

  const container = document.getElementById('carry-rank-rows');
  if (!container) return;

  try {
    // ── 1. CB policy rates (use STATE cache from fetchCBRates if available) ──
    const cbRates = {};
    await Promise.all(G8.map(async ccy => {
      const cached = STATE.cbRates?.[ccy.toLowerCase()];
      if (cached?.rate != null) { cbRates[ccy] = cached.rate; return; }
      try {
        const r = await fetch('./rates/' + ccy + '.json');
        if (!r.ok) return;
        const d = await r.json();
        if (d.observations?.[0]?.value) cbRates[ccy] = parseFloat(d.observations[0].value);
      } catch {}
    }));

    if (Object.keys(cbRates).length < 4) {
      container.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Rate data unavailable</div>';
      return;
    }

    // ── 1.5. OIS rates — preferred over CB policy rate (Bloomberg standard) ──
    // ois-rates/rates.json: SOFR(USD) €STR(EUR) SONIA(GBP) TONA(JPY)
    //                       CORRA(CAD) SARON(CHF) AONIA(AUD) OCR(NZD)
    // Falls back to CB policy rate when OIS unavailable (AUD/NZD staleness guard).
    // rateSource[ccy] tracks which benchmark is active for tooltip display.
    const oisCache = window._OIS_RATES_CACHE || {};
    const oisSrcs  = window._OIS_RATE_SOURCES || {};
    // If _OIS_RATES_CACHE is unpopulated (loadOISRatesCache not yet called), fetch inline
    let oisData = null;
    if (Object.keys(oisCache).length === 0) {
      try {
        const or = await fetch('./ois-rates/rates.json');
        if (or.ok) oisData = await or.json();
      } catch {}
    }
    const rates       = {};
    const rateSource  = {}; // e.g. { USD: 'SOFR', EUR: '€STR', AUD: 'policy' }
    for (const ccy of G8) {
      const ois = oisCache[ccy] ?? oisData?.rates?.[ccy] ?? null;
      const src = oisSrcs[ccy]  ?? oisData?.sources?.[ccy] ?? null;
      if (ois != null) {
        rates[ccy]      = ois;
        rateSource[ccy] = src || 'OIS';
      } else if (cbRates[ccy] != null) {
        rates[ccy]      = cbRates[ccy];
        rateSource[ccy] = 'policy';
      }
    }

    // ── 2. HV30 per pair from intraday cache ─────────────────────────────────
    const intra = await loadIntradayQuotes().catch(() => null);
    const hv30Map = {};
    if (intra?.hv30) Object.assign(hv30Map, intra.hv30);
    for (const [id, entry] of Object.entries(STOOQ_RT_CACHE)) {
      if (entry?.hv30 != null && hv30Map[id] == null) hv30Map[id] = entry.hv30;
    }

    // ── 3. Inflation expectations (same source as real-carry-modal2.js) ──────
    // extended-data/{CCY}.json written weekly by update-inflation-expectations.yml
    // Real rate = nominal CB rate − inflationExpectations
    // If modal was opened earlier, reuse _rcmData to avoid duplicate fetches.
    const inflExp = {};
    await Promise.all(G8.map(async ccy => {
      if (typeof _rcmData !== 'undefined' && _rcmData?.inflExp?.[ccy]?.val != null) {
        inflExp[ccy] = _rcmData.inflExp[ccy].val;
        return;
      }
      try {
        const r = await fetch('./extended-data/' + ccy + '.json');
        if (!r.ok) return;
        const d = await r.json();
        const ie = d?.data?.inflationExpectations;
        if (ie != null && ie > -5 && ie < 20) inflExp[ccy] = ie; // -5 floor accepts deflation (CHF/JPY history)
      } catch {}
    }));

    // ── 4. Build all 28 G8 pairs ─────────────────────────────────────────────
    // Rates now use OIS benchmarks (SOFR/€STR/SONIA/TONA/CORRA/SARON/AONIA/OCR)
    // with per-currency policy-rate fallback — matching Bloomberg FXFR convention.
    const allPairs = [];
    for (let i = 0; i < G8.length; i++) {
      for (let j = i + 1; j < G8.length; j++) {
        const a = G8[i], b = G8[j];
        const rA = rates[a] ?? null, rB = rates[b] ?? null;
        if (rA == null || rB == null) continue;

        const diff   = rA - rB;
        const long   = diff >= 0 ? a : b;
        const short  = diff >= 0 ? b : a;
        const rLong  = diff >= 0 ? rA : rB;
        const rShort = diff >= 0 ? rB : rA;
        const srcLong  = rateSource[long]  || 'OIS';
        const srcShort = rateSource[short] || 'OIS';
        const absDiff = Math.abs(diff);

        const pid  = pairId(long, short);
        const hv30 = hv30Map[pid] ?? null;

        // Real carry: nominal OIS differential minus inflation expectations differential
        // = realRate(long) − realRate(short). Inflation-adjusted carry, NOT CIP.
        // True CIP (Covered Interest Parity) uses FX forward points, not inflation data.
        const ieLong  = inflExp[long]  ?? null;
        const ieShort = inflExp[short] ?? null;
        const realCarry = (ieLong != null && ieShort != null)
          ? parseFloat((absDiff - (ieLong - ieShort)).toFixed(3))
          : null;

        // Carry-to-vol: real carry / HV30 — used as tiebreak
        const carryVol = (hv30 != null && hv30 > 0)
          ? (realCarry != null ? Math.abs(realCarry) : absDiff) / hv30
          : null;

        allPairs.push({ long, short, diff: absDiff, rLong, rShort, hv30, carryVol, realCarry, pid });
      }
    }

    // ── 5. Sort: real carry (primary) → carry-to-vol (tiebreak) → gross diff ─
    const hasRealCarryData = allPairs.some(p => p.realCarry != null);
    const hasVolData = allPairs.some(p => p.carryVol != null);
    allPairs.sort((a, b) => {
      if (hasRealCarryData) {
        const cipA = a.realCarry ?? -Infinity;
        const cipB = b.realCarry ?? -Infinity;
        if (Math.abs(cipB - cipA) > 0.001) return cipB - cipA;
      }
      if (hasVolData) {
        const cvA = a.carryVol ?? -Infinity;
        const cvB = b.carryVol ?? -Infinity;
        return cvB - cvA;
      }
      return b.diff - a.diff;
    });

    const top = allPairs.slice(0, 10);

    // Bar scale: proportional to the top pair's display value
    // Use real carry when available; fall back to nominal diff
    const topDisplay = top.map(p => Math.max(p.realCarry ?? p.diff, 0));
    const maxDisplay = Math.max(...topDisplay, 0.01);

    // ── 6. Update panel subtitle ──────────────────────────────────────────────
    const headSpan = container.closest('.sb-section')?.querySelector('.sb-head span');
    if (headSpan) {
      headSpan.textContent = hasRealCarryData
        ? 'Major FX · real carry · annualised'
        : 'Major FX · CB rate differential';
    }

    // ── 7. Attach header tooltip (once) ──────────────────────────────────────
    const sbHead = container.closest('.sb-section')?.querySelector('.sb-head');
    if (sbHead && !sbHead._carryTipAttached) {
      sbHead._carryTipAttached = true;
      sbHead.style.cursor = 'help';
      const tipTitle = hasRealCarryData ? 'Real Carry Ranking' : 'CB Rate Differential';
      const tipBody  = hasRealCarryData
        ? 'Ranked by real carry: nominal OIS rate differential minus the inflation expectations differential between the two legs (= real rate long − real rate short). Tiebreak: carry-to-vol (carry per unit of HV30 risk). Industry standard per Bloomberg FXFR. Click any row for full real rate breakdown.'
        : 'CB policy rate differential (%) between the long and short leg. Real carry ranking requires inflation expectations data (unavailable). Click any row for real rate analysis.';
      const tipEx = hasRealCarryData
        ? 'Example: GBP/CHF nominal +3.77% − (BoE infl.exp 3.45% − SNB infl.exp 0.31%) = real carry +0.63%. Positive = long leg earns positive real carry after purchasing power adjustment.'
        : 'Example: AUD 4.35% − CHF 0.00% = +4.35% gross nominal differential.';

      sbHead.addEventListener('mouseenter', ev => {
        const tt = document.getElementById('fx-tt');
        if (!tt) return;
        document.getElementById('fx-tt-title').textContent = tipTitle;
        document.getElementById('fx-tt-body').textContent  = tipBody;
        const exEl = document.getElementById('fx-tt-ex');
        exEl.textContent = tipEx; exEl.style.display = 'block';
        tt.style.display = 'block';
        requestAnimationFrame(() => window._fxTTPos && window._fxTTPos(ev.clientX, ev.clientY));
      });
      sbHead.addEventListener('mouseleave', () => {
        const tt = document.getElementById('fx-tt');
        if (tt) tt.style.display = 'none';
      });
    }

    // ── 8. Render rows ────────────────────────────────────────────────────────
    // Design: rank · pair · nominal spread label · proportional bar · real carry value
    // This matches Bloomberg/Refinitiv carry screen conventions:
    //   - Nominal spread shown as reference (what the market quotes)
    //   - Bar width proportional to real carry (true ranking metric)
    //   - Real carry value shown on right with color coding (green ≥+0.5%, red ≤−0.5%)
    container.innerHTML = top.map((p, idx) => {
      const sym = carryTV(p.long, p.short);

      // Nominal spread — the raw OIS rate differential, shown as context
      const spreadLabel = '+' + p.diff.toFixed(2) + '%';

      // Real carry — primary ranking value shown on the right
      const realCarryVal = p.realCarry;
      const displayVal = realCarryVal != null
        ? (realCarryVal >= 0 ? '+' : '') + realCarryVal.toFixed(2)
        : '+' + p.diff.toFixed(2);

      // Bar width: proportional to real carry of the top pair
      // Clamped to [4%, 100%] — never invisible, never overflows
      const barRaw = realCarryVal != null ? Math.max(realCarryVal, 0) : p.diff;
      const barPct = Math.max(Math.round((barRaw / maxDisplay) * 100), 4);

      // v8.4.8: Bloomberg canonical pair + carry direction awareness
      // carryDisplayPair() normalises to Bloomberg convention (e.g. NZD/USD, EUR/GBP).
      // Bar color: green = long the base (base has higher carry), red = short the base (carry on quote leg).
      const displayPair  = carryDisplayPair(p.long, p.short);
      const displayBase  = displayPair.split('/')[0];
      const isShortBase  = displayBase !== p.long;

      const barColor = isShortBase ? 'var(--down)' : 'var(--up)';

      const cls = realCarryVal != null
        ? (realCarryVal >= 0.5 ? 'pd-up' : realCarryVal <= -0.1 ? 'pd-dim' : '')
        : (p.diff > 2 ? 'pd-up' : p.diff > 0.5 ? '' : 'pd-dim');

      const realStr = realCarryVal != null ? (realCarryVal >= 0 ? '+' : '') + realCarryVal.toFixed(2) + '%' : '—';
      const hvStr   = p.hv30 != null ? p.hv30.toFixed(1) + '%' : 'n/a';
      const dirTip  = isShortBase ? `Short ${displayBase} / Long ${p.long}` : `Long ${displayBase}`;
      const tip = `${displayPair} · ${dirTip} · Nominal ${spreadLabel} · Real carry ${realStr} · HV30 ${hvStr} — Click for real rate analysis`;

      return `<div class="carry-rank-row" data-long="${p.long}" data-short="${p.short}" data-sym="${sym}" title="${tip}">
        <span class="cr-rank">${idx + 1}</span>
        <span class="cr-pair">${displayPair}</span>
        <span class="cr-spread">${spreadLabel}</span>
        <div class="cr-bar-wrap"><div class="cr-bar" style="width:${barPct}%;background:${barColor}"></div></div>
        <span class="cr-diff ${cls}">${displayVal}</span>
      </div>`;
    }).join('');

    // ── 9. Row click → open Real Rate Carry Modal ────────────────────────────
    // v8.4.6: pass Bloomberg canonical pair (base, quote) to the modal instead of
    // (highRateCcy, lowRateCcy). The modal derives carry direction from data,
    // so it never shows "LONG USD / SHORT NZD" when the display pair is NZD/USD.
    container.querySelectorAll('.carry-rank-row[data-long]').forEach(row => {
      row.addEventListener('click', () => {
        const displayPair = row.querySelector('.cr-pair')?.textContent || '';
        const [baseCcy, quoteCcy] = displayPair.split('/');
        if (typeof window.openRealCarryModal === 'function' && baseCcy && quoteCcy) {
          window.openRealCarryModal(baseCcy, quoteCcy);
        } else {
          loadTVChart(row.dataset.sym);
        }
      });
    });

  } catch(e) {
    console.warn('[CarryRanking]', e);
    if (container) container.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Unavailable</div>';
  }
}

// ═══════════════════════════════════════════════════════════════════
// CARRY TRADE SIDEBAR — from rates/*.json + extended-data/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchCarryData() {
  const CURRENCIES = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD'];
  const LABELS = { USD:'USD Fed', EUR:'EUR ECB', GBP:'GBP BoE', JPY:'JPY BoJ',
                   AUD:'AUD RBA', CHF:'CHF SNB', CAD:'CAD BoC', NZD:'NZD RBNZ' };

  try {
    // Fetch rates from repo
    const rateData = {};
    await Promise.all(CURRENCIES.map(async ccy => {
      try {
        const r = await fetch('./rates/' + ccy + '.json');
        if (!r.ok) return;
        const d = await r.json();
        if (d.observations && d.observations.length) {
          rateData[ccy] = parseFloat(d.observations[0].value);
        }
      } catch {}
    }));

    // Build carry pairs: long high-yield, short low-yield
    const carryPairs = [
      { long: 'AUD', short: 'JPY' },
      { long: 'NZD', short: 'JPY' },
      { long: 'GBP', short: 'JPY' },
      { long: 'AUD', short: 'CHF' },
      { long: 'NZD', short: 'CHF' },
      { long: 'USD', short: 'JPY' },
    ].map(p => {
      const diff = (rateData[p.long] ?? 0) - (rateData[p.short] ?? 0);
      return { ...p, diff };
    }).sort((a,b) => b.diff - a.diff);

    const container = document.getElementById('sb-carry-rows');
    if (!container) return;
    // Map carry pair to TradingView FX_IDC symbol
    // Convention: if USD is the quote (e.g. AUD/USD), symbol = FX_IDC:AUDUSD
    // Otherwise standard cross: FX_IDC:AUDJPY etc.
    function carrySymbol(long, short) {
      // USD-based pairs: the non-USD currency is either base or quote
      if (short === 'USD') return 'FX_IDC:' + long + 'USD';
      if (long  === 'USD') return 'FX_IDC:USD' + short;
      return 'FX_IDC:' + long + short;
    }

    container.innerHTML = carryPairs.map(p => {
      const sign = p.diff >= 0 ? '+' : '';
      const cls = p.diff > 1 ? 'up' : p.diff < 0 ? 'down' : 'flat';
      const longRate = (rateData[p.long]??0).toFixed(2);
      const shortRate = (rateData[p.short]??0).toFixed(2);
      const sym = carrySymbol(p.long, p.short);
      return `<div class="sb-row" data-sym="${sym}" style="cursor:pointer;" title="Open ${p.long}/${p.short} chart">
        <span class="sb-sym">${p.long}/${p.short}</span>
        <span class="sb-price" style="font-size:8.5px;color:var(--text3);letter-spacing:-0.01em;">${longRate}% · ${shortRate}%</span>
        <span class="sb-chg ${cls}">${sign}${p.diff.toFixed(2)}%</span>
      </div>`;
    }).join('');
  } catch(e) { console.warn('Carry fetch failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// CROSS-ASSET — custom grid from Stooq/yfinance + repo extended-data
// ═══════════════════════════════════════════════════════════════════
async function fetchCrossAssetData() {
  // stooq() helper removed — yfinance JSON used exclusively

  function setCA(id, val, chgPct, isYield, chgAbs) {
    const vEl = document.getElementById('ca-' + id);
    const cEl = document.getElementById('cac-' + id);
    if (!vEl || !cEl) return;
    if (val == null) return;
    if (chgPct == null) {
      vEl.textContent = isYield ? val.toFixed(2) + '%' : val.toLocaleString(undefined, { maximumFractionDigits: val > 100 ? 2 : 4 });
      vEl.className = 'ca-val flat';
      cEl.textContent = '—'; cEl.className = 'ca-chg flat';
      return;
    }
    const cls   = chgPct > 0.05 ? 'up' : chgPct < -0.05 ? 'down' : '';
    const arrow = chgPct > 0.05 ? '▲' : chgPct < -0.05 ? '▼' : '→';
    const sign  = chgPct >= 0 ? '+' : '';
    vEl.textContent = isYield ? val.toFixed(2) + '%' : val.toLocaleString(undefined, { maximumFractionDigits: val > 100 ? 2 : 4 });
    vEl.className = 'ca-val';
    // Format: "▲ +18.4 (+0.35%)" when absolute available, "▲ +0.35%" when not
    if (chgAbs != null && !isYield) {
      const absSign = chgAbs >= 0 ? '+' : '';
      const absFmt  = Math.abs(chgAbs) >= 1000
        ? chgAbs.toLocaleString(undefined, { maximumFractionDigits: 0 })
        : Math.abs(chgAbs) >= 10
          ? (absSign + chgAbs.toFixed(1))
          : (absSign + chgAbs.toFixed(2));
      cEl.textContent = arrow + ' ' + absFmt + ' (' + sign + chgPct.toFixed(2) + '%)';
    } else {
      cEl.textContent = arrow + ' ' + sign + chgPct.toFixed(2) + '%';
    }
    cEl.className = 'ca-chg ' + cls;
  }

  // ── STEP 1: Pre-load repo data (same-origin, instant) so US10Y is available immediately ──
  let _repoUs10y = null;
  try {
    const usdExt = await fetch('./extended-data/USD.json').then(r => r.ok ? r.json() : null).catch(() => null);
    if (usdExt?.data?.bond10y != null && !isNaN(usdExt.data.bond10y)) {
      _repoUs10y = { close: usdExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
      // Render US10Y immediately so cross-asset table isn't blank while data loads
      setCA('us10y', _repoUs10y.close, null, true);
    }
  } catch {}

  // ── STEP 1.5: Intraday quotes from GitHub Action (yfinance) ──
  // Pre-populate all cross-asset cells. yfinance JSON is the sole real-time source.
  const _caIntraday = await loadIntradayQuotes();  // uses cache — no extra network call if already loaded
  let _caGold   = _caIntraday ? intradayQuote(_caIntraday, 'gold')   : null;
  let _caWti    = _caIntraday ? intradayQuote(_caIntraday, 'wti')    : null;
  let _caSpx    = _caIntraday ? intradayQuote(_caIntraday, 'spx')    : null;
  let _caNikkei = _caIntraday ? intradayQuote(_caIntraday, 'nikkei') : null;
  let _caStoxx  = _caIntraday ? intradayQuote(_caIntraday, 'stoxx')  : null;
  let _caDxy    = _caIntraday ? intradayQuote(_caIntraday, 'dxy')    : null;

  // Render inmediato con JSON intraday — el usuario ve valores en <100ms.
  if (_caSpx)    setCA('spx',    _caSpx.close,    _caSpx.pct,    false, _caSpx.chg);
  if (_caGold) {
    setCA('gold', _caGold.close, _caGold.pct, false, _caGold.chg);
    const gEl = document.getElementById('q-xauusd'), gcEl = document.getElementById('qc-xauusd');
    if (gEl)  { gEl.textContent  = _caGold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); gEl.className  = 'q-price ' + clsDir(_caGold.chg); }
    if (gcEl) { gcEl.textContent = pctStr(_caGold.pct); gcEl.className = 'q-chg '   + clsDir(_caGold.chg); }
  }
  if (_caWti)    setCA('wti',    _caWti.close,    _caWti.pct,    false, _caWti.chg);
  if (_caNikkei) setCA('nikkei', _caNikkei.close, _caNikkei.pct, false, _caNikkei.chg);
  if (_caStoxx)  setCA('stoxx',  _caStoxx.close,  _caStoxx.pct,  false, _caStoxx.chg);
  // US10Y desde intraday JSON — sobreescribe el valor de repo (que puede tener 1 día de delay)
  const _caUs10yEarly = _caIntraday ? intradayQuote(_caIntraday, 'us10y') : null;
  if (_caUs10yEarly && _caUs10yEarly.close > 0) setCA('us10y', _caUs10yEarly.close, _caUs10yEarly.pct, true);
  // Gold/SPX ratio — calculado apenas tenemos ambos valores del JSON
  if (_caGold && _caSpx && _caSpx.close > 0) {
    const ratio = (_caGold.close / _caSpx.close).toFixed(3);
    const rNum  = parseFloat(ratio);
    const sig   = rNum > 0.75 ? 'Risk-Off signal' : rNum > 0.55 ? 'Neutral' : 'Risk-On signal';
    const cls   = rNum > 0.75 ? 'down' : rNum < 0.55 ? 'up' : 'flat';
    setEl('ri-gold-spx', ratio);
    setEl('ri-gold-spx-sig', sig, cls);
  }
  if (_caDxy) {
    setCA('dxy', _caDxy.close, _caDxy.pct, false, _caDxy.chg);
    const dEl = document.getElementById('q-dxy'), dcEl = document.getElementById('qc-dxy');
    if (dEl)  { dEl.textContent  = _caDxy.close.toFixed(1); dEl.className  = 'q-price ' + clsDir(_caDxy.chg); }
    if (dcEl) { dcEl.textContent = pctStr(_caDxy.pct);      dcEl.className = 'q-chg '   + clsDir(_caDxy.chg); }
  }
  // BTC inmediato desde JSON
  const _caBtcEarly = _caIntraday ? intradayQuote(_caIntraday, 'btc') : null;
  if (_caBtcEarly) {
    const btcFmtE = _caBtcEarly.close.toLocaleString(undefined, {minimumFractionDigits:0, maximumFractionDigits:0});
    const bEl = document.getElementById('ca-btc'), bcEl = document.getElementById('cac-btc');
    const qbEl = document.getElementById('q-btcusd'), qbcEl = document.getElementById('qc-btcusd');
    if (bEl)  { bEl.textContent  = btcFmtE; bEl.className  = 'ca-val'; }
    if (bcEl) {
      const _btcArrow = (_caBtcEarly.chg??0) > 0 ? '▲' : (_caBtcEarly.chg??0) < 0 ? '▼' : '→';
      const _btcSign  = (_caBtcEarly.pct??0) >= 0 ? '+' : '';
      if (_caBtcEarly.chg != null) {
        const _btcAbs = _caBtcEarly.chg.toLocaleString(undefined,{maximumFractionDigits:0});
        bcEl.textContent = _btcArrow + ' ' + (_caBtcEarly.chg>=0?'+':'') + _btcAbs + ' (' + _btcSign + (_caBtcEarly.pct??0).toFixed(2) + '%)';
      } else {
        bcEl.textContent = _btcArrow + ' ' + _btcSign + (_caBtcEarly.pct??0).toFixed(2) + '%';
      }
      bcEl.className = 'ca-chg ' + clsDir(_caBtcEarly.chg);
    }
    // Always overwrite topbar BTC from yfinance (CoinGecko is only a pre-load placeholder)
    if (qbEl)  { qbEl.textContent  = btcFmtE; qbEl.className  = 'q-price ' + clsDir(_caBtcEarly.chg); }
    if (qbcEl) { qbcEl.textContent = pctStr(_caBtcEarly.pct); qbcEl.className = 'q-chg ' + clsDir(_caBtcEarly.chg); }
    // Seed STOOQ_RT_CACHE early so the chart has yfinance data immediately
    STOOQ_RT_CACHE['btc'] = _caBtcEarly;
  }
  // ETH inmediato desde JSON — same early-seed pattern as BTC so the LW chart
  // today-bar is available as soon as the modal opens (before STEP 2 completes).
  const _caEthEarly = _caIntraday ? intradayQuote(_caIntraday, 'eth') : null;
  if (_caEthEarly) STOOQ_RT_CACHE['eth'] = _caEthEarly;

  // ── STEP 2: All cross-asset data from intraday quotes.json (yfinance) ──
  // Stooq and Yahoo removed — both blocked by CORS in production.
  // quotes.json (same-origin, ~5min delay) covers all symbols.
  const finalSpx    = _caSpx;
  const finalGold   = _caGold;
  const finalWti    = _caWti;
  const finalNikkei = _caNikkei;
  const finalStoxx  = _caStoxx;
  const finalDxy    = _caDxy;
  const us10y       = (_caIntraday ? intradayQuote(_caIntraday, 'us10y') : null) || _repoUs10y;

  // Mirror cross-asset quotes into STOOQ_RT_CACHE so _lwUpdateTodayBar() can
  // push live prices to LW charts for non-FX instruments (BTC, gold, SPX, etc.)
  if (finalSpx)    { STOOQ_RT_CACHE['spx']    = finalSpx;    setCA('spx',    finalSpx.close,    finalSpx.pct,    false, finalSpx.chg); }
  if (finalGold) {
    STOOQ_RT_CACHE['xauusd'] = STOOQ_RT_CACHE['gold'] = finalGold;
    setCA('gold', finalGold.close, finalGold.pct, false, finalGold.chg);
    const gEl = document.getElementById('q-xauusd'), gcEl = document.getElementById('qc-xauusd');
    if (gEl)  { gEl.textContent  = finalGold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); gEl.className  = 'q-price ' + clsDir(finalGold.chg); }
    if (gcEl) { gcEl.textContent = pctStr(finalGold.pct); gcEl.className = 'q-chg ' + clsDir(finalGold.chg); }
  }
  if (finalWti)    { STOOQ_RT_CACHE['wti']    = finalWti;    setCA('wti',    finalWti.close,    finalWti.pct,    false, finalWti.chg); }
  if (finalNikkei) { STOOQ_RT_CACHE['nikkei'] = finalNikkei; setCA('nikkei', finalNikkei.close, finalNikkei.pct, false, finalNikkei.chg); }
  if (finalStoxx)  { STOOQ_RT_CACHE['stoxx']  = finalStoxx;  setCA('stoxx',  finalStoxx.close,  finalStoxx.pct,  false, finalStoxx.chg); }
  if (us10y)       { if (!us10y.fromRepo) STOOQ_RT_CACHE['us10y'] = us10y; setCA('us10y', us10y.close, us10y.fromRepo ? null : us10y.pct, true); }

  const dxyData = finalDxy;
  if (dxyData) {
    STOOQ_RT_CACHE['dxy'] = dxyData;
    setCA('dxy', dxyData.close, dxyData.pct, false, dxyData.chg);
    const dEl = document.getElementById('q-dxy');
    const dcEl = document.getElementById('qc-dxy');
    if (dEl) { dEl.textContent = dxyData.close.toFixed(1); dEl.className = 'q-price ' + clsDir(dxyData.chg); }
    if (dcEl) { dcEl.textContent = pctStr(dxyData.pct); dcEl.className = 'q-chg ' + clsDir(dxyData.chg); }
  }

  // BTC — intraday JSON (yfinance BTC-USD) primary, CoinGecko topbar cache fallback
  const btcEl = document.getElementById('ca-btc');
  const btcCEl = document.getElementById('cac-btc');
  const qBtc = document.getElementById('q-btcusd');
  const qBtcC = document.getElementById('qc-btcusd');
  const _btcIntraday = _caIntraday ? intradayQuote(_caIntraday, 'btc') : null;
  if (_btcIntraday) STOOQ_RT_CACHE['btc'] = _btcIntraday;  // feed LW chart live bar (yfinance)
  const _ethIntraday = _caIntraday ? intradayQuote(_caIntraday, 'eth') : null;
  if (_ethIntraday) STOOQ_RT_CACHE['eth'] = _ethIntraday;  // feed LW chart live bar (yfinance)
  if (_btcIntraday && btcEl) {
    const btcFmt = _btcIntraday.close.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
    btcEl.textContent  = btcFmt;
    btcEl.className    = 'ca-val';
    if (btcCEl) {
      const _biArrow = (_btcIntraday.chg??0) > 0 ? '▲' : (_btcIntraday.chg??0) < 0 ? '▼' : '→';
      const _biSign  = (_btcIntraday.pct??0) >= 0 ? '+' : '';
      if (_btcIntraday.chg != null) {
        const _biAbs = _btcIntraday.chg.toLocaleString(undefined,{maximumFractionDigits:0});
        btcCEl.textContent = _biArrow + ' ' + (_btcIntraday.chg>=0?'+':'') + _biAbs + ' (' + _biSign + (_btcIntraday.pct??0).toFixed(2) + '%)';
      } else {
        btcCEl.textContent = _biArrow + ' ' + _biSign + (_btcIntraday.pct??0).toFixed(2) + '%';
      }
      btcCEl.className = 'ca-chg ' + clsDir(_btcIntraday.chg);
    }
    // Always update topbar q-btcusd from yfinance — CoinGecko is only a pre-load fallback
    if (qBtc) {
      qBtc.textContent  = btcFmt;
      qBtc.className    = 'q-price ' + clsDir(_btcIntraday.chg);
      if (qBtcC) { qBtcC.textContent = pctStr(_btcIntraday.pct); qBtcC.className = 'q-chg ' + clsDir(_btcIntraday.chg); }
    }
  } else if (btcEl && qBtc && qBtc.textContent !== '—') {
    btcEl.textContent = qBtc.textContent;
    btcEl.className = qBtc.className.replace('q-price', 'ca-val');
    if (btcCEl && qBtcC) { btcCEl.textContent = qBtcC.textContent; btcCEl.className = qBtcC.className.replace('q-chg', 'ca-chg'); }
  }

  const upd = document.getElementById('ca-updated');
  const now = new Date();
  const localHHMM = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  // Show the actual source — intraday JSON (yfinance) when fresh, Stooq otherwise
  let sourceLabel = 'yfinance';
  if (_caIntraday?.source && _caIntraday.source !== 'repo') {
    const srcName = _caIntraday.source === 'yfinance'      ? 'yfinance'
                  : _caIntraday.source === 'twelve_data'   ? 'Twelve Data'
                  : _caIntraday.source === 'alpha_vantage' ? 'Alpha Vantage'
                  : 'mixed APIs';
    // Check if the file is fresh (under 8 min old — 5 min interval + 3 min margin)
    const fileAge = _caIntraday.updated
      ? (Date.now() - new Date(_caIntraday.updated).getTime()) / 60000
      : 999;
    sourceLabel = fileAge < 8 ? `${srcName} · ~5min delay` : 'yfinance';
  }
  if (upd) upd.textContent = sourceLabel + ' · ' + localHHMM + ' ' + tzAbbr;

  // Gold / SPX ratio — computed here where gold & spx are in scope
  if (finalGold && finalSpx && finalSpx.close > 0) {
    const ratio = (finalGold.close / finalSpx.close).toFixed(3);
    const rNum  = parseFloat(ratio);
    const sig   = rNum > 0.75 ? 'Risk-Off signal' : rNum > 0.55 ? 'Neutral' : 'Risk-On signal';
    const cls   = rNum > 0.75 ? 'down' : rNum < 0.55 ? 'up' : 'flat';
    setEl('ri-gold-spx', ratio);
    setEl('ri-gold-spx-sig', sig, cls);
  }

  // Push updated prices to the active LW chart (gold, SPX, WTI, etc.)
  // FX pairs are handled by fetchQuoteBarRT; cross-asset needs this extra call.
  _lwUpdateTodayBar();
}

// ═══════════════════════════════════════════════════════════════════
// BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// FED RATE EXPECTATIONS — computed from meetings-data if available,
// otherwise from CB rate trajectory in rates/USD.json
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// CB RATE EXPECTATIONS — todos los bancos centrales
// Usa meetings-data/meetings.json + rates/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchFedExpectations() {
  try {
    const tbody = document.getElementById('fed-exp-tbody');
    if (!tbody) return;

    // Load meetings and all rates in parallel
    const [meetingsRes, ...rateResponses] = await Promise.all([
      fetch('./meetings-data/meetings.json').then(r => r.ok ? r.json() : null).catch(() => null),
      ...['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'].map(c =>
        fetch(`./rates/${c}.json`).then(r => r.ok ? r.json() : null).catch(() => null)
      )
    ]);

    const currencies = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
    const bankMeta = {
      USD: { flag:'us', short:'Fed'  },
      EUR: { flag:'eu', short:'ECB'  },
      GBP: { flag:'gb', short:'BoE'  },
      JPY: { flag:'jp', short:'BoJ'  },
      AUD: { flag:'au', short:'RBA'  },
      CAD: { flag:'ca', short:'BoC'  },
      CHF: { flag:'ch', short:'SNB'  },
      NZD: { flag:'nz', short:'RBNZ' },
    };

    // CIP spot sources — quote convention (how many USD per 1 unit of ccy, or inverse)
    // EUR/GBP/AUD/NZD: spot is direct (EURUSD etc.) → base currency is the foreign one
    // JPY/CHF/CAD:     spot is inverse (USDJPY etc.) → USD is the base
    // USD rate — kept for potential future use; cipSpot/CIP removed in v7.25.4
    // (Column now shows uniform implied policy rate for all currencies)

    const rows = [];
    currencies.forEach((ccy, i) => {
      const rateData = rateResponses[i];
      if (!rateData) return;
      const obs = rateData.observations || [];
      if (obs.length < 2) return;

      const current = parseFloat(obs[0].value);

      const meetings = meetingsRes?.meetings?.[ccy];
      const nextMtg  = meetings?.nextMeeting || '—';

      // ── Bias: prefer explicit market-consensus field from meetings.json ──
      // meetings.bias       = 'cut' | 'hold' | 'hike' — OIS/overnight rate implied direction
      // meetings.biasMethod = 'ois' | 'ois-preserved' | 'heuristic'
      // meetings.biasSource = human-readable source label (e.g. "CME FedWatch (SOFR futures)")
      // meetings.biasUpdated = ISO date the bias was last computed by the engine
      // Always compute trendDir for use in FWD projection — bias field only overrides the label
      const trendDir     = computeCBTrend(obs);   // 'up' | 'down' | 'flat' — always needed for FWD
      const meetingsBias = meetings?.bias;
      const biasMethod   = meetings?.biasMethod ?? null;
      const biasSource   = meetings?.biasSource ?? null;
      const biasUpdated  = meetings?.biasUpdated ?? null;

      // Build tooltip: method + source + freshness
      function buildBiasTooltip() {
        const isOIS  = biasMethod === 'ois' || biasMethod === 'ois-preserved';
        const src    = biasSource || (isOIS ? 'OIS/overnight rate' : 'rate trajectory');
        const upd    = biasUpdated ? ` · updated ${biasUpdated}` : '';
        const pres   = biasMethod === 'ois-preserved' ? ' (OIS source temporarily unavailable — last known signal preserved)' : '';
        const heur   = biasMethod === 'heuristic' ? ' (OIS source unavailable — estimated from rate trajectory)' : '';
        return `Market forward direction · ${src}${upd}${pres}${heur}`;
      }
      const biasTip = buildBiasTooltip();

      // ── Market-implied move probability (CME/ASX where available; null otherwise) ──
      // Show cut probability chip for cut/hold bias; hike probability chip for hike bias.
      // Bloomberg WIRP standard: display the probability of the expected direction.
      const cutProb  = meetings?.cutProb  ?? null;  // number (0–100) or null
      const hikeProb = meetings?.hikeProb ?? null;  // number (0–100) or null
      const probSrc  = biasSource || 'OIS/futures';
      let probSuffix = '';
      if (meetingsBias === 'hike' && hikeProb !== null) {
        const probCls = hikeProb >= 60 ? 'up' : hikeProb >= 40 ? '' : 'flat';
        probSuffix = ` <span class="${probCls}" style="font-size:8px;font-family:var(--font-mono);opacity:0.85;white-space:nowrap;" title="Market-implied probability of a hike at next meeting · ${probSrc}">${hikeProb}%↑</span>`;
      } else if (cutProb !== null) {
        const probCls = cutProb >= 60 ? 'down' : cutProb >= 40 ? '' : 'flat';
        probSuffix = ` <span class="${probCls}" style="font-size:8px;font-family:var(--font-mono);opacity:0.85;white-space:nowrap;" title="Market-implied probability of a cut at next meeting · ${probSrc}">${cutProb}%↓</span>`;
      }

      let biasLabel;
      if (meetingsBias === 'cut') {
        biasLabel = `<span class="down" title="${biasTip}">↓ Cut</span>` + probSuffix;
      } else if (meetingsBias === 'hike') {
        biasLabel = `<span class="up" title="${biasTip}">↑ Hike</span>` + probSuffix;
      } else if (meetingsBias === 'hold') {
        biasLabel = `<span class="flat" title="${biasTip}">→ Hold</span>` + probSuffix;
      } else {
        // Fallback: derive from historical rate trajectory (no OIS/futures data available).
        // ~ prefix signals this is an estimate, not a market-consensus value —
        // per GUIDELINES.md: "prefixes the label with ~ to signal estimation".
        const fbTip = 'Estimated from rate trajectory · OIS source unavailable';
        biasLabel = trendDir === 'down' ? `<span class="down" title="${fbTip}">~ ↓ Cut</span>`
                  : trendDir === 'up'   ? `<span class="up" title="${fbTip}">~ ↑ Hike</span>`
                  :                       `<span class="flat" title="${fbTip}">~ → Hold</span>`;
        biasLabel += probSuffix;
      }

      // ── Implied policy rate — expected rate at next meeting ─────────
      // Industry standard (Bloomberg WIRP / CME FedWatch): probability-weighted
      // expected rate = Σ(scenario_prob × scenario_rate).
      //
      // Three-scenario model: cut / hold / hike.
      // Each scenario assumes one standard 25bp step.
      //   implied = current
      //             + (hikeProb/100 × +0.25)
      //             − (cutProb/100  × +0.25)
      //   holdProb = 100 − cutProb − hikeProb  (residual, not stored separately)
      //
      // Priority 1: explicit fwdRate from meetings.json (prob-weighted · computed by workflow)
      //             Workflow writes this field using the same formula as Priority 2.
      //             No ~ prefix — label shows 'prob. weighted · OIS' in the modal.
      // Priority 2: compute on-the-fly if cutProb or hikeProb available (≥1 field)
      //             No ~ prefix — probability data from OIS/futures is authoritative.
      // Priority 3: ±step naive estimate (no prob data at all) → ~ prefix signals estimation
      let fwdDisplay = '—';
      const meetingBias = (() => {
        if (!meetings) return null;
        const b = meetings.bias;
        if (!b) return null;
        if (/cut|dovish/i.test(b))  return 'down';
        if (/hike|hawkish/i.test(b)) return 'up';
        return 'flat';
      })();
      // Priority 1: meetings.json fwdRate (prob-weighted · workflow-computed)
      // Bloomberg standard: accept 0 and negative values — currencies like CHF and JPY
      // can have OIS-implied rates below zero. Rejecting fwdRate=0 as "missing" is wrong.
      if (meetings?.fwdRate != null && !isNaN(meetings.fwdRate)) {
        fwdDisplay = meetings.fwdRate.toFixed(2) + '%';
      } else {
        const pCut  = (meetings?.cutProb  != null && !isNaN(meetings.cutProb))  ? Math.min(100, Math.max(0, meetings.cutProb))  : null;
        const pHike = (meetings?.hikeProb != null && !isNaN(meetings.hikeProb)) ? Math.min(100, Math.max(0, meetings.hikeProb)) : null;

        // Per-bank standard move size (Bloomberg WIRP convention):
        // BoJ historically moves in 10bp increments; SNB uses 25bp standard (may use 50bp).
        // All others: 25bp standard.
        const CB_STEP = { JPY: 0.10, CHF: 0.25 };
        const cbStep  = CB_STEP[ccy] ?? 0.25;

        if (pCut !== null || pHike !== null) {
          // Priority 2: probability-weighted — Bloomberg WIRP standard.
          // No floor: OIS-implied rates below zero are valid (CHF 2015–2022, JPY ongoing).
          const cut  = pCut  ?? 0;
          const hike = pHike ?? 0;
          // Clamp residual so probabilities never exceed 100%
          const cutC  = Math.min(cut,  100);
          const hikeC = Math.min(hike, 100 - cutC);
          const implied = current + (hikeC / 100) * cbStep - (cutC / 100) * cbStep;
          fwdDisplay = implied.toFixed(2) + '%';
        } else {
          // Priority 3: no OIS probabilities — naive ±step heuristic, ~ signals estimate.
          // Floor retained here: without probability data the heuristic is directional only.
          const dir  = meetingBias ?? trendDir;
          const step = dir === 'down' ? -cbStep : dir === 'up' ? cbStep : 0;
          fwdDisplay = '~' + Math.max(0, current + step).toFixed(2) + '%';
        }
      }

      const meta = bankMeta[ccy];
      const flag = `<span class="fi fi-${meta.flag}" style="margin-right:4px;border-radius:2px;vertical-align:middle;"></span>`;

      rows.push(`<tr title="Next meeting: ${nextMtg} · CIP 30d fwd">
        <td style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${flag}<span style="font-size:10px;">${meta.short}</span> <span style="color:var(--text3);font-size:9px;">${nextMtg}</span></td>
        <td style="overflow:hidden;white-space:nowrap;">${biasLabel}</td>
        <td style="color:var(--text2);font-family:var(--font-mono);font-size:10px;white-space:nowrap;padding-left:3px;padding-right:3px;">${fwdDisplay}</td>
      </tr>`);
    });

    if (rows.length) tbody.innerHTML = rows.join('');

    // Expose meetings data globally so cb-rates-modal can read bias/fwdRate on click
    if (meetingsRes?.meetings) window._STATE_meetings = meetingsRes; // store full {meetings:{}} wrapper
  } catch(e) { console.warn('CB expectations failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// POSITIONING BIAS — three data sources, rendered in priority order:
//
// SOURCE 1 — CBOE/CME Vol Index (primary): ATM implied vol from CBOE/CME FX Volatility Indexes (quotes.json).
//   FXE → EUR/USD  FXB → GBP/USD  FXY → USD/JPY  FXA → AUD/USD
//   When available: shows ATM IV column + IV Rank (when ≥4w history) or COT bias fallback.
//
// SOURCE 2 — COT (always loaded): CFTC Disaggregated TFF · Leveraged Funds net positioning.
//   Used as directional bias proxy and fallback when ETF IV is unavailable.
//
// SOURCE 3 — 25d Risk Reversal (supplemental): Saxo Bank public options page · 1M tenor.
//   rr-data/rr.json — updated Mon–Fri 08:30 UTC.
//   25d RR = 25d call IV − 25d put IV. Positive → base currency calls bid (upside skew).
//   Shown as a small chip below the Direction cell when available. Does not add a column.
//
// Column layout (when ETF IV available):
//   Pair | ATM IV | IV Rnk or COT bias | Direction [+ 25d RR chip if available]
// Column layout (COT fallback only):
//   Pair | 1W | 1M | Bias [+ 25d RR chip if available]
// ═══════════════════════════════════════════════════════════════════
async function fetchOptionSkew() {
  try {
    // skew-tbody may be absent if Positioning Bias panel was removed;
    // RR fetch must still run so RR_DATA_CACHE is populated for other panels.
    const tbody = document.getElementById('skew-tbody');

    const pairs = [
      { pair:'EUR/USD', cot:'EUR', etfId:'eurusd', rrKey:'EURUSD' },
      { pair:'GBP/USD', cot:'GBP', etfId:'gbpusd', rrKey:'GBPUSD' },
      { pair:'USD/JPY', cot:'JPY', etfId:'usdjpy', rrKey:'USDJPY' },
      { pair:'AUD/USD', cot:'AUD', etfId:'audusd', rrKey:'AUDUSD' },
      { pair:'USD/CAD', cot:'CAD', etfId:'usdcad',  rrKey:'USDCAD' },
      { pair:'USD/CHF', cot:'CHF', etfId:'usdchf',  rrKey:'USDCHF' },
      { pair:'NZD/USD', cot:'NZD', etfId:null,     rrKey:'NZDUSD' },
    ];

    // ── SOURCE 1: ETF IV from intraday quotes.json (primary) ──
    const intradayData = await loadIntradayQuotes().catch(() => null);
    const etfIvMap = intradayData?.fx_etf_iv || {};

    // ── SOURCE 2: COT positioning (bias direction + fallback values) ──
    const cotFiles = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
    const cotResults = await Promise.all(cotFiles.map(async ccy => {
      try {
        const r = await fetch('./cot-data/' + ccy + '.json');
        if (!r.ok) return null;
        const d = await r.json();
        return { ccy, net: d.netPosition || 0, long: d.longPositions||0, short: d.shortPositions||0 };
      } catch { return null; }
    }));
    const cotMap = {};
    cotResults.filter(Boolean).forEach(c => { cotMap[c.ccy] = c; });

    // ── SOURCE 3: 25d Risk Reversals from Saxo Bank (supplemental) ──
    // rr-data/rr.json — updated Mon–Fri 08:30 UTC by update-saxo-rr.yml
    // Graceful: if file missing or fetch fails, rrMap stays empty and RR chips are hidden.
    let rrMap = {};
    try {
      const rrRes = await fetch('./rr-data/rr.json').catch(() => null);
      if (rrRes?.ok) {
        const rrJson = await rrRes.json();
        if (rrJson?.pairs) rrMap = rrJson.pairs;  // { EURUSD: { rr25d: -0.45 }, … }
        // Populate global cache so pair detail popover can read RR without extra fetches
        Object.assign(RR_DATA_CACHE, rrMap);
      }
    } catch { /* RR unavailable — continue without it */ }

    // COT → directional bias proxy (used for Bias column + fallback 1W/1M)
    function netToSkew(net, invert) {
      const scale = Math.abs(net) / 50000;
      const val = Math.min(1.5, scale * 1.2);
      const signed = net > 0 ? val : -val;
      return invert ? -signed : signed;
    }

    // Update thead to reflect what's actually showing
    const hasAnyEtfIv = pairs.some(p => etfIvMap[p.etfId]?.iv != null);
    const hasIvRank   = pairs.some(p => etfIvMap[p.etfId]?.iv_rank != null);
    const thead = tbody ? tbody.closest('table')?.querySelector('thead tr') : null;
    if (thead) {
      if (hasAnyEtfIv) {
        // IV Rank column shown when history is available (≥4 weeks)
        thead.innerHTML = hasIvRank
          ? '<th style="text-align:left" scope="col">Pair</th><th scope="col">ATM IV</th><th scope="col" title="IV Rank: position of current IV within 52-week range (0=historically low, 100=historically high)">IV Rnk</th><th scope="col">Direction</th>'
          : '<th style="text-align:left" scope="col">Pair</th><th scope="col">ATM IV</th><th scope="col">COT bias</th><th scope="col">Direction</th>';
      } else {
        thead.innerHTML = '<th style="text-align:left" scope="col">Pair</th><th scope="col" title="Current CFTC COT net (LF longs minus shorts)">Net</th><th scope="col" title="Net ~4 weeks ago — direction of change">4W</th><th scope="col">Bias</th>';
      }
    }

    // Per-cell tooltip content — indexed by pair label, used inside pairs.map() below
    const skewCellTips = {
      'EUR/USD': { body: 'EUR/USD skew derived from CFTC Leveraged Funds net EUR positioning (Options+Futures Combined). Positive = EUR calls bid (market positioned for EUR upside). Negative = EUR puts bid (downside protection).', ex: 'Most reliable when Leveraged Funds and Asset Manager positioning agree in direction. Divergence between the two signals uncertainty or a potential positioning squeeze.' },
      'GBP/USD': { body: 'GBP/USD skew from CFTC Leveraged Funds net GBP positioning. Reflects speculative appetite for sterling vs dollar.', ex: 'GBP skew is especially sensitive to UK macro surprises (CPI, PMI). Watch for regime shifts around BoE meetings.' },
      'USD/JPY': { body: 'USD/JPY skew from CFTC Leveraged Funds net JPY positioning (inverted). Positive = USD calls bid / JPY puts bid (USD upside expected). Negative = JPY safe-haven demand dominant.', ex: 'Risk-off events flip USD/JPY skew negative quickly as JPY is bought as safe haven. Monitor against VIX for confirmation.' },
      'AUD/USD': { body: 'AUD/USD skew from CFTC Leveraged Funds net AUD positioning. AUD is a risk/commodity proxy — positive skew aligns with global risk appetite and commodity strength.', ex: 'AUD skew often leads iron ore and copper price expectations. Negative skew on AUD/USD with rising VIX = classic risk-off setup.' },
      'USD/CAD': { body: 'USD/CAD bias from CFTC Leveraged Funds net CAD positioning (inverted). Positive = USD calls bid / CAD puts bid (USD upside, CAD weakness). Negative = CAD demand dominant, often driven by oil strength or risk-on.', ex: 'CAD is tightly linked to WTI crude. Watch for divergence between COT bias and oil price direction — that spread often resolves in oil\'s favour.' },
      'USD/CHF': { body: 'USD/CHF bias from CFTC Leveraged Funds net CHF positioning (inverted). Positive = USD calls bid / CHF puts bid. Negative = CHF safe-haven demand dominant.', ex: 'CHF safe-haven flows can override COT positioning quickly during risk-off episodes. Treat CHF bias as a risk sentiment barometer alongside JPY.' },
      'NZD/USD': { body: 'NZD/USD bias from CFTC Leveraged Funds net NZD positioning. NZD is a high-beta risk/commodity proxy — positive bias aligns with global risk appetite and dairy/agricultural strength.', ex: 'NZD often moves in tandem with AUD. Divergence between the two — e.g. NZD negative while AUD positive — can signal idiosyncratic NZ macro risk (RBNZ, trade data).' },
    };

    if (tbody) tbody.innerHTML = pairs.map(p => {
      const cotData = cotMap[p.cot];
      const etfIv   = etfIvMap[p.etfId];
      const invert  = p.pair.startsWith('USD/');

      if (!cotData && !etfIv) {
        return `<tr><td>${p.pair}</td><td colspan="3" style="color:var(--text3)">—</td></tr>`;
      }

      // Directional bias from COT (unchanged — positioning signal)
      const cotSkew = cotData ? netToSkew(cotData.net, invert) : 0;
      const bias    = Math.abs(cotSkew) < 0.1 ? 'Neutral'
                    : cotSkew > 0 ? p.pair.split('/')[0]+'+'
                    : p.pair.split('/')[1]+'+';
      const biasCls = Math.abs(cotSkew) < 0.1 ? 'flat' : cotSkew > 0 ? 'up' : 'down';
      const fmtRR   = v => (v >= 0 ? '+' : '') + v.toFixed(2);

      if (etfIv?.iv != null) {
        // ── ETF IV available: show real implied vol ──
        const ivStr  = etfIv.iv.toFixed(1) + '%';
        const ivCls  = etfIv.iv > 12 ? 'down' : etfIv.iv > 7 ? '' : 'up';

        // IV Rank column: show when ≥4 weeks of history available
        let col2Html, col2Title;
        if (etfIv.iv_rank != null) {
          const rnk    = etfIv.iv_rank;
          const pct    = etfIv.iv_pct_rank ?? rnk;
          const n      = etfIv.iv_hist_n   ?? '?';
          const rnkCls = rnk > 75 ? 'down' : rnk < 25 ? 'up' : '';
          const rnkStr = Math.round(rnk) + 'rnk';  // e.g. "82rnk"
          col2Html  = `<td class="${rnkCls}" style="font-family:var(--font-mono);font-size:10px">${rnkStr}</td>`;
          col2Title = `IV Rank ${rnk.toFixed(0)} (${n}w history) · IV Percentile ${pct.toFixed(0)} · High rank = historically expensive vol`;
        } else {
          const cotStr = cotData ? fmtRR(cotSkew) : '—';
          const cotCls = cotData ? (cotSkew >= 0 ? 'up' : 'down') : 'flat';
          col2Html  = `<td class="${cotCls}" style="font-size:10px">${cotStr}</td>`;
          col2Title = `ETF: ${etfIv.source} · exp ${etfIv.expiry} · ATM strike ${etfIv.atm} · IV Rank building (need ≥4 weekly snapshots)`;
        }

        // 25d RR chip — shown below bias label when Saxo data available
        // Note: no native browser title= here — tooltip handled per-cell via #fx-tt
        const rrEntry  = rrMap[p.rrKey];
        const rrVal    = rrEntry?.rr25d ?? null;
        const rrTipText = rrVal !== null
          ? `25-delta Risk Reversal (1M) · Saxo Bank · ${rrVal > 0 ? 'calls bid — upside skew on ' + p.pair.split('/')[0] : 'puts bid — downside skew on ' + p.pair.split('/')[0]}`
          : '';
        const rrChip   = rrVal !== null
          ? `<div style="font-size:8px;font-family:var(--font-mono);opacity:0.8;margin-top:1px;color:${rrVal > 0 ? 'var(--up)' : rrVal < 0 ? 'var(--down)' : 'var(--text3)'};"
              data-rr-tip-title="25d RR · Saxo Bank (1M)"
              data-rr-tip-body="${rrTipText}"
             >RR ${rrVal >= 0 ? '+' : ''}${rrVal.toFixed(2)}</div>`
          : '';

        // Per-cell tooltip data — td[0]=Pair, td[1]=ATM IV, td[2]=IV Rank or COT skew, td[3]=Bias
        const pairTip  = skewCellTips[p.pair];
        const td0Title = p.pair + ' — Positioning Bias';
        const td0Body  = pairTip?.body || '';
        const td0Ex    = pairTip?.ex   || '';
        const td1Title = 'ATM Implied Volatility · ' + p.pair;
        const td1Body  = `ATM IV ${ivStr} from CBOE/CME FX Volatility Index (${etfIv.source || 'CBOE/CME'}) — variance-swap methodology, same as VIX. Institutional benchmark for OTC interbank implied vol. Green ≤7% (cheap vol); red >12% (expensive).`;
        const td1Ex    = 'CBOE/CME FX Volatility Indexes use the same variance-swap replication as VIX. EUR/GBP/JPY/AUD have dedicated CBOE indexes (^EUVIX/^BPVIX/^JYVIX/^AUDVIX). CHF and CAD use CME futures options (6S/6C) with ETF fallback (FXF/FXC). All values have ~15min delay.';
        const td3Title = p.pair + ' — Directional Bias';
        const td3Body  = pairTip?.body || '';
        const td3Ex    = pairTip?.ex   || '';

        return `<tr>
          <td data-tip-title="${td0Title}" data-tip-body="${td0Body}" data-tip-ex="${td0Ex}">${p.pair}</td>
          <td class="${ivCls}" style="font-family:var(--font-mono)"
              data-tip-title="${td1Title}" data-tip-body="${td1Body}" data-tip-ex="${td1Ex}">${ivStr}</td>
          ${col2Html.replace('<td ', `<td data-tip-title="IV Rank · ${p.pair}" data-tip-body="${col2Title}" `)}
          <td class="${biasCls}" style="line-height:1.3;"
              data-tip-title="${td3Title}" data-tip-body="${td3Body}" data-tip-ex="${td3Ex}">${bias}${rrChip}</td>
        </tr>`;
      } else {
        // ── COT fallback: original behavior ──
        const skew1w = cotData ? netToSkew(cotData.net, invert) : 0;
        // v7.88.0: 4W-ago net from real CFTC history, replaces fabricated 0.85 multiplier
        const _hist4w = window.COT_DATA_STORE?.[cotData?.ccy]?.history;
        const _net4w = (_hist4w && _hist4w.length >= 5) ? (_hist4w[_hist4w.length - 5]?.levNet ?? null) : null;
        const skew1m = cotData ? netToSkew(_net4w ?? cotData.net * 0.85, invert) : 0;
        // 25d RR chip — shown below bias label when Saxo data available
        // Note: no native browser title= here — tooltip handled per-cell via #fx-tt
        const rrEntryCot = rrMap[p.rrKey];
        const rrValCot   = rrEntryCot?.rr25d ?? null;
        const rrTipTextCot = rrValCot !== null
          ? `25-delta Risk Reversal (1M) · Saxo Bank · ${rrValCot > 0 ? 'calls bid — upside skew on ' + p.pair.split('/')[0] : 'puts bid — downside skew on ' + p.pair.split('/')[0]}`
          : '';
        const rrChipCot  = rrValCot !== null
          ? `<div style="font-size:8px;font-family:var(--font-mono);opacity:0.8;margin-top:1px;color:${rrValCot > 0 ? 'var(--up)' : rrValCot < 0 ? 'var(--down)' : 'var(--text3)'};"
              data-rr-tip-title="25d RR · Saxo Bank (1M)"
              data-rr-tip-body="${rrTipTextCot}"
             >RR ${rrValCot >= 0 ? '+' : ''}${rrValCot.toFixed(2)}</div>`
          : '';

        // Per-cell tooltip data — COT fallback mode: td[1]=1W skew, td[2]=1M skew, td[3]=Bias
        const pairTipCot = skewCellTips[p.pair];
        const td0TitleCot = p.pair + ' — Positioning Bias';
        const td0BodyCot  = pairTipCot?.body || '';
        const td0ExCot    = pairTipCot?.ex   || '';
        const td12Title   = 'COT Directional Skew · ' + p.pair;
        const td12Body    = 'est. via COT — no CBOE/CME volatility index available for this pair. Derived from CFTC Leveraged Funds net positioning (Disaggregated TFF · Options+Futures Combined). Net = current week; 4W = net ~4 weeks ago (real CFTC history, v7.88.0).';
        const td3TitleCot = p.pair + ' — Directional Bias';
        const td3BodyCot  = pairTipCot?.body || '';
        const td3ExCot    = pairTipCot?.ex   || '';

        return `<tr>
          <td data-tip-title="${td0TitleCot}" data-tip-body="${td0BodyCot}" data-tip-ex="${td0ExCot}">${p.pair}</td>
          <td class="${skew1w >= 0 ? 'up':'down'}"
              data-tip-title="${td12Title}" data-tip-body="${td12Body}">${fmtRR(skew1w)}</td>
          <td class="${skew1m >= 0 ? 'up':'down'}"
              data-tip-title="${td12Title} (1M)" data-tip-body="${td12Body}">${fmtRR(skew1m)}</td>
          <td class="${biasCls}" style="line-height:1.3;"
              data-tip-title="${td3TitleCot}" data-tip-body="${td3BodyCot}" data-tip-ex="${td3ExCot}">${bias}${rrChipCot}</td>
        </tr>`;
      }
    }).join('');

    // Update panel subtitle to reflect actual source
    const panelHead = document.getElementById('skew-source-label');
    const hasRR = Object.keys(rrMap).length > 0;
    if (panelHead) {
      if (hasAnyEtfIv) {
        panelHead.textContent = hasRR ? 'CBOE/CME Vol · 25d RR · Saxo' : 'CBOE/CME Vol Index · IV';
      } else {
        panelHead.textContent = hasRR ? 'COT · 25d RR · Saxo' : 'COT-derived · IV unavailable';
      }
    }

  } catch(e) { console.warn('Option skew failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// LOAD AI REGIME — fast-path: prime narrative text from cached AI JSON.
// Primes the narrative text from cached AI JSON before buildRichNarrative() runs.
// Regime badges (#risk-regime, #narrative-regime) are exclusively owned by
// renderRiskData() — always reflecting the live VIX stress score.
// ═══════════════════════════════════════════════════════════════════
async function loadAIRegime() {
  try {
    const res = await fetch('./ai-analysis/index.json');
    if (!res.ok) return;
    const d = await res.json();
    // Store generated_at so buildRichNarrative can compute staleness
    if (d.generated_at) _narrativeGeneratedAt = d.generated_at;
  } catch { /* silently skip */ }
}

// RICH AI NARRATIVE — build from ai-analysis/index.json + live data
// ═══════════════════════════════════════════════════════════════════
async function buildRichNarrative() {
  try {
    // Fetch AI narrative base
    const [narRes, newsRes] = await Promise.all([
      fetch('./ai-analysis/index.json'),
      fetch('./news-data/news.json'),
    ]);

    let baseNarrative = '';
    let regime = 'RISK-OFF';
    // _narrativeGeneratedAt is module-level — do not re-declare here

    if (narRes.ok) {
      const d = await narRes.json();
      baseNarrative = d.narrative || '';
      regime = d.regime || 'RISK-OFF';
      _narrativeGeneratedAt = d.generated_at || null;
      _narrativeAiRegime   = regime.replace(/^__STALE__/, '') || null; // store raw AI regime for mismatch note

      // Staleness check — if the AI JSON is older than 4 hours, mark regime badge as stale
      // so users know it may not reflect current market conditions
      if (_narrativeGeneratedAt) {
        const ageMinutes = (Date.now() - new Date(_narrativeGeneratedAt).getTime()) / 60000;
        if (ageMinutes > 240) {
          regime = '__STALE__' + regime;
        }
      }
    }

    // Pull key headlines from news to enrich narrative
    let newsContext = [];
    if (newsRes.ok) {
      const nd = await newsRes.json();
      const articles = nd.articles || [];
      // Get top 6 featured/recent high-impact items
      newsContext = articles
        .filter(a => a.impact === 'high' && (!a.lang || a.lang === 'en'))
        .slice(0, 6);
    }

    // Build contextual currency mentions from news
    const curMentions = {};
    newsContext.forEach(a => {
      if (a.cur) curMentions[a.cur] = (curMentions[a.cur] || 0) + 1;
    });
    const topCur = Object.entries(curMentions).sort((a,b) => b[1]-a[1]).map(e=>e[0]).slice(0,3);

    // Build FX context from Frankfurter rates if available
    const fxLines = [];
    const r = STATE.rates;
    const p = STATE.prevRates;
    if (r && Object.keys(r).length) {
      // EUR/USD
      if (r.EUR && p.EUR) {
        const eurusd = 1/r.EUR, prevEurusd = 1/p.EUR;
        const pct = (eurusd - prevEurusd)/prevEurusd*100;
        if (Math.abs(pct) > 0.05)
          fxLines.push(`EUR/USD ${pct>0?'bid':'offered'} at ${eurusd.toFixed(4)} (${pct>=0?'+':''}${pct.toFixed(2)}%)`);
      }
      // USD/JPY
      if (r.JPY && p.JPY) {
        const usdjpy = r.JPY, prevJpy = p.JPY;
        const pct = (usdjpy - prevJpy)/prevJpy*100;
        if (Math.abs(pct) > 0.05)
          fxLines.push(`USD/JPY ${pct>0?'extends gains':'retreats'} to ${usdjpy.toFixed(2)}`);
      }
      // DXY proxy (USD strength via basket)
      const majors = ['EUR','GBP','AUD','NZD'];
      const avgPct = majors.filter(c=>r[c]&&p[c]).map(c=>(r[c]-p[c])/p[c]*100);
      if (avgPct.length) {
        const usdAvg = -(avgPct.reduce((a,b)=>a+b,0)/avgPct.length);
        if (Math.abs(usdAvg) > 0.03)
          fxLines.push(`DXY ${usdAvg>0?'firming':'weakening'} — USD ${usdAvg>0?'broadly bid':'broadly offered'}`);
      }
    }

    // Pick headline from top news item — title only, never expand (expand is article body)
    let headlineSnippet = '';
    if (newsContext.length) {
      const topItem = newsContext[0];
      const title = (topItem.title || '').replace(/\s+/g,' ').trim().slice(0,100);
      if (title.length > 20) headlineSnippet = title + (title.length === 100 ? '…' : '');
    }

    // Compose final narrative — Groq narrative is authoritative; fxLines is fallback only.
    // The engine now sends real price levels — appending Frankfurter-derived fxLines
    // produces contradictory language ("USD broadly offered" after "USD mixed") and
    // grows the narrative beyond the 2-line layout budget. Removed in v7.23.10.
    let finalNarrative = '';

    if (baseNarrative.length > 40) {
      // Use Groq narrative as-is — it already contains current price levels and catalysts.
      // No Frankfurter enrichment: legacy fxLines used stale/different rates and contradicted Groq.
      finalNarrative = baseNarrative;
    } else if (fxLines.length || headlineSnippet) {
      // No Groq narrative available — build from live data as fallback
      const parts = [];
      if (fxLines.length) parts.push(fxLines.join('. '));
      if (topCur.length && topCur.length <= 4) parts.push(`${topCur.join(', ')} in focus`);
      if (headlineSnippet) parts.push(headlineSnippet);
      finalNarrative = parts.join('. ') + '.';
    }

    // Update narrative text only.
    // Regime badges (#risk-regime, #narrative-regime) are exclusively owned by
    // renderRiskData() — always live VIX stress score. Never written here.
    const el = document.getElementById('narrative-text');
    if (el && finalNarrative) el.textContent = finalNarrative;

    // Also load signals (moved here from fetchAIData to keep AI logic together)
    try {
      const sigR = await fetch('./ai-analysis/signals.json');
      if (sigR.ok) {
        const _sigRaw = await sigR.json();
        // signals.json may be a bare array (written by fetch_intraday_quotes.py) or
        // a dict { "generated_at": "...", "signals": [...] } (written by generate_narrative_signals.py).
        // Normalise to array before rendering.
        const signals = Array.isArray(_sigRaw) ? _sigRaw : (Array.isArray(_sigRaw?.signals) ? _sigRaw.signals : []);
        if (Array.isArray(signals) && signals.length) {
          const container = document.getElementById('alerts-container');
          const sub = document.getElementById('alerts-sub');
          if (container) {
            // Convert engine UTC time string "HH:MM" to user's local timezone
            function localizeSignalTime(timeStr) {
              if (!timeStr || timeStr === '--:--') return timeStr || '--:--';
              try {
                const [h, m] = timeStr.split(':').map(Number);
                if (isNaN(h) || isNaN(m)) return timeStr;
                const now = new Date();
                const utcDate = new Date(Date.UTC(
                  now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), h, m
                ));
                return utcDate.toLocaleTimeString(navigator.language || 'en', {
                  hour: '2-digit', minute: '2-digit', hour12: false
                });
              } catch { return timeStr; }
            }
            container.innerHTML = signals.map(s => {
              const dotCls = s.priority === 'critical' ? 'a-crit' : s.priority === 'warning' ? 'a-warn' : 'a-info';
              const localTime = localizeSignalTime(s.time);
              // evidence[]: "LABEL: VALUE" strings set by the engine for data traceability.
              // Rendered as a collapsible row below the signal text — hidden by default,
              // toggled by clicking the signal row. Tooltip on the row shows all evidence inline.
              const ev = Array.isArray(s.evidence) && s.evidence.length ? s.evidence : [];
              const evTooltip = ev.length ? ev.join(' · ') : '';
              const evHtml = ev.length
                ? `<div class="a-evidence" aria-label="Signal data sources">${ev.map(e => `<span class="a-ev-chip">${e}</span>`).join('')}</div>`
                : '';
              return `<div class="alert-row${ev.length ? ' a-has-ev' : ''}" ${evTooltip ? `title="${evTooltip}"` : ''}>
                <span class="a-time">${localTime}</span>
                <span class="a-dot ${dotCls}"></span>
                <div class="a-text"><strong>${s.title || ''}</strong>${s.title ? ' — ' : ''}${s.text || ''}${evHtml}</div>
              </div>`;
            }).join('');

            // Toggle evidence chips on row click (expand/collapse)
            container.querySelectorAll('.a-has-ev').forEach(row => {
              row.style.cursor = 'pointer';
              row.addEventListener('click', () => {
                const evEl = row.querySelector('.a-evidence');
                if (evEl) evEl.classList.toggle('a-evidence-open');
              });
            });
          }
          if (sub) {
            const now = new Date();
            const hhmm = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
            const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
            sub.textContent = signals.length + ' active · AI-generated · loaded ' + hhmm + ' ' + tzAbbr + ' · Not investment advice';
          }

          // Notify user if signal set changed and notifications are enabled
          maybeNotifyNewSignals(signals);
        }
      }
    } catch {}

  } catch(e) { console.warn('Narrative build failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE SPREADS — computed from HV30 + VIX + MOVE
//
// Methodology (professional ECN spread model):
//   spread = ECN_FLOOR + HV30 × VOL_COEF × vixMultiplier [× moveMultiplier]
//
//   ECN_FLOOR  — institutional minimum at peak liquidity (London/NY overlap),
//                calibrated against IC Markets, Pepperstone Razor, LMAX avg.
//   VOL_COEF   — pip sensitivity per 1% of 30-day realised vol, per pair.
//                Higher for commodity currencies (AUD, NZD) that gap more.
//   vixMult    — linear stress scalar: 1.0× at VIX 15 → 1.5× at VIX 30,
//                capped at 2.0×. Captures widening during risk-off spikes.
//   moveMult   — MOVE overlay applied to rates-sensitive pairs (JPY, CHF):
//                +5% per 10 MOVE points above 80 (IG desk convention).
//
//   All inputs from intraday-data/quotes.json — no external API required.
//   Refreshes every time the intraday JSON updates (~5 min in production).
// ═══════════════════════════════════════════════════════════════════
async function fetchReferenceSpreads() {
  try {
    // ── Model parameters ─────────────────────────────────────────────
    const ECN_FLOOR = {
      eurusd: 0.1, gbpusd: 0.2, usdjpy: 0.1,
      audusd: 0.2, usdchf: 0.2, usdcad: 0.2, nzdusd: 0.3,
    };
    const VOL_COEF = {
      eurusd: 0.035, gbpusd: 0.045, usdjpy: 0.030,
      audusd: 0.060, usdchf: 0.055, usdcad: 0.050, nzdusd: 0.070,
    };

    // ── Fetch vol inputs from the already-loaded intraday cache ───────
    const intradayData = await loadIntradayQuotes();
    if (!intradayData) return;   // silently keep static HTML fallback

    const quotes = intradayData.quotes || {};
    const hv30   = intradayData.hv30  || {};

    const vix  = quotes.vix?.close  || 15;
    const move = quotes.move?.close  || 80;

    // Stress multipliers
    const vixMult  = Math.min(2.0, Math.max(1.0, 1.0 + (vix  - 15) / 30));
    const moveMult = Math.min(1.3, Math.max(1.0, 1.0 + (move - 80) / 200));

    // ── Compute spreads ───────────────────────────────────────────────
    const computed = {};
    for (const pair of Object.keys(ECN_FLOOR)) {
      const hv      = hv30[pair] ?? quotes[pair]?.hv30 ?? 8.0;
      const isRates = pair === 'usdjpy' || pair === 'usdchf';
      const volMult = isRates ? vixMult * moveMult : vixMult;
      const raw     = ECN_FLOOR[pair] + hv * VOL_COEF[pair] * volMult;
      computed[pair] = Math.max(ECN_FLOOR[pair], Math.round(raw * 10) / 10);
    }

    // ── Write into LIVE_SPREADS so TYPICAL_SPREADS Proxy feeds dynamic Bid/Ask ──
    // All existing bid/ask calculations in populateFxPairsTable and updateFxPairsTableRT
    // automatically pick up the new values via the Proxy — no extra code needed.
    let _spreadsChanged = false;
    for (const [pair, pips] of Object.entries(computed)) {
      if (LIVE_SPREADS[pair] !== pips) { LIVE_SPREADS[pair] = pips; _spreadsChanged = true; }
    }
    if (_spreadsChanged && Object.keys(STOOQ_RT_CACHE).length > 0) updateFxPairsTableRT();

    // ── Render Reference Spreads panel ────────────────────────────────
    const MAX_PIP = 5.0;
    const pairMap = {
      eurusd: 'spr-eurusd', gbpusd: 'spr-gbpusd', usdjpy: 'spr-usdjpy',
      audusd: 'spr-audusd', usdchf: 'spr-usdchf', usdcad: 'spr-usdcad',
      nzdusd: 'spr-nzdusd',
    };

    for (const [pair, elId] of Object.entries(pairMap)) {
      const pips = computed[pair];
      if (pips == null) continue;
      const el = document.getElementById(elId);
      if (!el) continue;
      const fillEl = el.closest('.spread-row')?.querySelector('.spr-fill');

      const color = pips <= 1.0 ? 'var(--up)' : pips <= 2.0 ? 'var(--orange)' : 'var(--down)';
      const cls   = pips <= 1.0 ? 'up'        : pips <= 2.0 ? ''              : 'down';

      el.textContent = pips.toFixed(1) + ' pip';
      el.className   = 'spr-val' + (cls ? ' ' + cls : '');
      el.style.color = cls ? '' : 'var(--orange)';

      if (fillEl) {
        fillEl.style.width      = Math.min(100, (pips / MAX_PIP) * 100) + '%';
        fillEl.style.background = color;
      }
    }

    // Subtitle — vol regime label + timestamp
    const sub = document.getElementById('spreads-sub');
    if (sub) {
      const _sprNow = new Date();
      const _sprHHMM = _sprNow.getHours().toString().padStart(2,'0') + ':' + _sprNow.getMinutes().toString().padStart(2,'0');
      const _sprTZ = _sprNow.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
      const regime = vix < 20 ? 'Low vol' : vix < 28 ? 'Elevated vol' : 'High vol';
      sub.textContent = `ECN est. · ${regime} · VIX ${vix.toFixed(1)} · ${_sprHHMM} ${_sprTZ}`;
    }

  } catch(e) { console.warn('[Spreads] Failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// SESSION VOLATILITY — HV30-derived pip ranges per trading session
//
// Methodology:
//   daily_range_pips = close × (HV30/100) / √252 × pip_factor
//   session_range    = daily_range × SESSION_RATIO[session]
//
//   SESSION_RATIO: empirical session/daily range ratios from Myfxbook
//   5-year session statistics (2019-2024). Each session’s ratio reflects
//   how much of the total daily range it typically contributes, accounting
//   for session overlap (sum > 1.0 is expected and correct).
//
//   EUR/USD: pip_factor = 10000 (4-decimal pair)
//   USD/JPY: pip_factor = 100   (2-decimal pair)
//
//   Refreshes with every intraday JSON update (~5 min in production).
//   Falls back silently to static HTML values if data unavailable.
// ═══════════════════════════════════════════════════════════════════
async function computeSessionVol() {
  try {
    const data = await loadIntradayQuotes();
    if (!data?.quotes) return;

    const eur = data.quotes.eurusd;
    const jpy = data.quotes.usdjpy;
    if (!eur?.hv30 || !jpy?.hv30) return;

    // Session/daily range ratios — Myfxbook 5yr empirical averages
    const SESSION_RATIO_EUR = { syd: 0.28, tok: 0.50, lon: 0.87, ny: 0.83 };
    const SESSION_RATIO_JPY = { syd: 0.25, tok: 0.65, lon: 0.72, ny: 0.80 };  // v7.88.0: Tokyo raised 0.60→0.65, London lowered 0.75→0.72 (BIS 2022: USD/JPY Asia ~44% vol share > London ~34%)

    // Daily range estimate from HV30 (annualised % → daily pips)
    const dailyEur = eur.close * (eur.hv30 / 100) / Math.sqrt(252) * 10000;
    const dailyJpy = jpy.close * (jpy.hv30 / 100) / Math.sqrt(252) * 100;

    const sessions = [
      { key: 'syd', eurId: 'svol-syd-eur', jpyId: 'svol-syd-jpy' },
      { key: 'tok', eurId: 'svol-tok-eur', jpyId: 'svol-tok-jpy' },
      { key: 'lon', eurId: 'svol-lon-eur', jpyId: 'svol-lon-jpy' },
      { key: 'ny',  eurId: 'svol-ny-eur',  jpyId: 'svol-ny-jpy'  },
    ];

    sessions.forEach(({ key, eurId, jpyId }) => {
      const eurPips = Math.round(dailyEur * SESSION_RATIO_EUR[key]);
      const jpyPips = Math.round(dailyJpy * SESSION_RATIO_JPY[key]);

      // Colour tiers: low = flat, mid = neutral, high = up (brightest)
      const eurCls = eurPips < 25 ? 'flat' : eurPips < 55 ? '' : 'up';
      const jpyCls = jpyPips < 30 ? 'flat' : jpyPips < 60 ? '' : 'up';

      const elEur = document.getElementById(eurId);
      const elJpy = document.getElementById(jpyId);
      if (elEur) { elEur.textContent = `±${eurPips}p`; elEur.className = eurCls; }
      if (elJpy) { elJpy.textContent = `±${jpyPips}p`; elJpy.className = jpyCls; }
    });

    const sub = document.getElementById('svol-sub');
    if (sub) sub.textContent = `HV30 ${eur.hv30.toFixed(1)}% · 5yr historical session ratios`;  // v7.88.0: BIS/Myfxbook removed — BIS publishes volume share, not range ratios

  } catch(e) { console.warn('[SessionVol] Failed:', e); }
}



// ═══════════════════════════════════════════════════════════════════
// BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════════
async function boot() {
  // PHASE 1: Load intraday quotes.json (same-origin, no CORS) — primary data source
  // Frankfurter (ECB) is non-blocking background fallback — CORS may block it in some browsers
  fetchFrankfurter();                // background: populates STATE.rates as fallback only

  // PHASE 2: Parallel — all remaining data loads simultaneously

  // Pre-load intraday JSON now (same-origin, ~0ms) so that fetchRiskData
  // and fetchCrossAssetData find it in cache when they need it.
  // await guarantees the JSON is ready BEFORE fetchRiskData/fetchCrossAssetData
  // request it — prevents each function from issuing its own parallel fetch and racing.
  await loadIntradayQuotes();

  // fetchQuoteBarRT populates STOOQ_RT_CACHE (RT prices + hv30).
  // Expose promise so bootNewFeatures() can await it before renderCIPForwards().
  // Awaited here so populateFxPairsTable finds the RT cache ready when it renders.
  window._quotesReadyPromise = fetchQuoteBarRT();
  await window._quotesReadyPromise;
  if (typeof initFxWebSocket === 'function') initFxWebSocket();
  await window._quotesReadyPromise;
  loadFxPerfData().then(() => populateFxPairsTable()); // 1W perf data, re-render when ready
  populateCorrelations(); // 60-day rolling correlations from quotes.json

  // Static repo data — all parallel, fast (same GitHub Pages origin)
  fetchCBRates().then(() => fetchCarryRanking());   // ranking needs rates populated first
  fetchCOTData();
  fetchFedExpectations();
  fetchOptionSkew().then(() => attachRiskMonitorTooltips());
  fetchCarryData();
  initAlerts();
  fetchNewsData();
  fetchReferenceSpreads();          // HV30+VIX+MOVE vol model — no external API, updates with intraday JSON
  computeSessionVol();              // HV30-derived session pip ranges — replaces static table

  // ── CRITICAL: Load AI regime badge FIRST, before fetchRiskData touches the narrative badge.
  // loadAIRegime() is a lightweight fetch of ai-analysis/index.json (~same-origin, <50ms).
  // Awaited so _narrativeGeneratedAt is populated before buildRichNarrative runs.
  // Regime badges are set exclusively by renderRiskData() via the live VIX stress score.
  await loadAIRegime();

  // External API data — all in parallel.
  // fetchCrossAssetData runs immediately (no longer waits for fetchRiskData) so the
  // Cross-Asset panel populates from the intraday JSON cache on first render (~100ms).
  // Gold/SPX ratio is computed inside fetchCrossAssetData once it has both values.
  fetchRiskData();
  fetchCrossAssetData();
  fetchCommodityQuotes();
  // AI narrative full build (non-blocking, fills narrative text).
  // Chain a post-resolve scroll reset: injecting the full narrative text expands
  // #narrative's height, which can cause the browser to scroll #main down to
  // maintain the visual position of content below it. Resetting scrollTop after
  // the text is injected ensures the narrative is always visible on load.
  buildRichNarrative().then(() => {
    const _m = document.getElementById('main');
    if (_m) _m.scrollTop = 0;
    // Belt-and-suspenders: signals and regime badge also render async after the
    // narrative resolves (fetchRiskData → renderRiskData). Give them 300ms to
    // settle, then do a final reset so any secondary reflow is also corrected.
    setTimeout(() => { if (_m) _m.scrollTop = 0; }, 300);
  });
  setTimeout(fetchSentiment, 800);   // Dukascopy sentiment (last, non-critical)

  // Reset scroll on every load — prevents browser from restoring mid-panel positions
  // that would hide the narrative section or the calendar header on first view.
  const _rp = document.getElementById('rightpanel');
  if (_rp) _rp.scrollTop = 0;
  const _main = document.getElementById('main');
  if (_main) _main.scrollTop = 0;
}

boot();

// Refresh quote bar FX every 60 seconds via intraday JSON / yfinance (~5 min delay)
setInterval(fetchQuoteBarRT, 60 * 1000);
// Refresh ECB rates every 30 minutes (FX table + heatmap + cross rows)
setInterval(fetchFrankfurter, 30 * 60 * 1000);
// Refresh news every 10 minutes
setInterval(fetchNewsData, 2 * 60 * 1000);   // every 2 min — ETag returns 304 when unchanged (zero cost); server updates hourly
// Refresh narrative every 5 minutes
setInterval(buildRichNarrative, 15 * 60 * 1000);
// Refresh risk/yield data every 5 minutes

// ═══════════════════════════════════════════════════════════════════
// TOP NAV — smooth scroll to sections + active state
// ═══════════════════════════════════════════════════════════════════
(function() {
  const main = document.getElementById('main');
  const rightPanel = document.getElementById('rightpanel');
  // Targets that live in the right panel sidebar, not main
  const RIGHT_PANEL_TARGETS = new Set(['section-cbrates']);

  document.querySelectorAll('.top-nav a[data-target]').forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const target = this.dataset.target;
      document.querySelectorAll('.top-nav a').forEach(a => a.classList.remove('active'));
      this.classList.add('active');

      if (target === 'top') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        if (main) main.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }

      const el = document.getElementById(target);
      if (!el) return;

      // Check if this target is in the right panel
      if (RIGHT_PANEL_TARGETS.has(target) && rightPanel) {
        const mainScrollable = main && main.scrollHeight > main.clientHeight && getComputedStyle(main).overflowY !== 'visible';
        if (mainScrollable) {
          // Desktop: rightpanel is fixed aside — scroll rightpanel to the element
          const offset = el.offsetTop - rightPanel.offsetTop;
          rightPanel.scrollTo({ top: offset - 4, behavior: 'smooth' });
        } else {
          // Mobile: rightpanel is stacked below main — use scrollIntoView
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
      }

      // Normal main-panel targets
      const mainScrollable = main && main.scrollHeight > main.clientHeight && getComputedStyle(main).overflowY !== 'visible';
      if (mainScrollable) {
        const offset = el.offsetTop - (main.offsetTop || 0);
        main.scrollTo({ top: offset - 4, behavior: 'smooth' });
      } else {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();

// ─── FX LIQUIDITY CANVAS — real intraday activity via Frankfurter cache ──────
// Strategy: reads ECB rate series from /fx-data/frankfurter.json (server-side cache,
// updated every 4h by engine workflow) and maps daily price-change magnitude → proxy
// for interbank volume. Falls back to BIS/LSEG session-overlap baseline if unavailable.

const LIQ_BASE = [18,14,11,10,12,20,30,42,58,68,72,70,72,82,95,100,95,80,68,55,42,30,22,20];
// Session definitions (UTC hours)
const LIQ_SESSIONS = [
  { name:'Sydney',   start:22, end:7,  color:'rgba(120,100,255,0.10)' },
  { name:'Tokyo',    start:0,  end:9,  color:'rgba(79,127,255,0.08)'  },
  { name:'London',   start:8,  end:17, color:'rgba(38,166,154,0.10)' },
  { name:'New York', start:13, end:22, color:'rgba(246,148,28,0.07)' },
];

// _liqData:     48 half-hour values for the current day (real H-L range proxy when available)
// _liqBaseline: 48 half-hour values for the 30-day rolling average (drawn as reference line)
// _liqSource:   string for the panel subtitle label
let _liqData     = null;
let _liqBaseline = null;
let _liqSource   = null;
let _narrativeGeneratedAt = null; // ISO timestamp of last AI narrative — written by loadAIRegime() and buildRichNarrative()
let _narrativeAiRegime   = null; // Regime label from AI JSON (may differ from live score when market conditions changed since generation)

// Interpolate a 24-hour array to 48 half-hour slots
function _liqTo48(arr24) {
  return Array.from({length:48}, (_,i) => {
    const h = i/2, idx=Math.floor(h)%24, next=(idx+1)%24, frac=h-Math.floor(h);
    return arr24[idx]*(1-frac) + arr24[next]*frac;
  });
}

async function fetchLiquidityData() {
  const utcDay = new Date().getUTCDay(), utcHour = new Date().getUTCHours();
  // Canvas OFFSET=44 means left edge = 22:00 UTC. Keep weekend mode until 22:00 UTC Sunday
  // so that nowCanvasSlot starts at 0 (far left) when the chart begins — not 47 (far right).
  const isWeekend = utcDay === 6 || (utcDay === 0 && utcHour < 22) || (utcDay === 5 && utcHour >= 21);

  // ── Primary: fx-liquidity.json (yfinance H-L range proxy, updated hourly) ──
  try {
    const r = await fetch('/fx-data/fx-liquidity.json');
    if (!r.ok) throw new Error('fx-liquidity.json not available');
    const d = await r.json();

    if (!d.baseline_30d || d.baseline_30d.length !== 24) throw new Error('malformed baseline');

    // Baseline: 30-day rolling average (always shown as reference)
    _liqBaseline = _liqTo48(isWeekend ? Array(24).fill(2) : d.baseline_30d);

    // Today: real H-L data for completed hours, baseline for future hours
    const todayRaw = (d.today && d.today.length === 24) ? d.today : d.baseline_30d;
    const hoursComplete = d.hours_complete || 0;
    const nowH = new Date().getUTCHours() + new Date().getUTCMinutes()/60;

    const today24 = Array.from({length:24}, (_,h) => {
      if (isWeekend) return 2;
      if (h < hoursComplete && todayRaw[h] > 0) return todayRaw[h];   // real data
      if (h >= Math.floor(nowH)) return d.baseline_30d[h];             // future: 30d real baseline
      return d.baseline_30d[h];                                          // past gap: use baseline
    });

    _liqData   = _liqTo48(today24);
    _liqSource = d.fallback ? 'Historical avg · fixed reference' : 'yfinance · H-L range proxy · 30d avg';
    return;
  } catch(e) {
    // fall through to legacy fallback
  }

  // ── Fallback: frankfurter.json vol-scalar (legacy, kept for resilience) ──
  try {
    const r = await fetch('/fx-data/frankfurter.json');
    if (!r.ok) throw new Error('frankfurter.json not available');
    const cacheData = await r.json();
    const rates = Object.values((cacheData.series && cacheData.series.rates) ? cacheData.series.rates : {});
    let volScalar = 1.0;
    if (rates.length >= 2) {
      const changes = [];
      for (let i = 1; i < rates.length; i++) {
        const prev = rates[i-1], cur = rates[i];
        if (prev.USD && cur.USD) changes.push(Math.abs(cur.USD - prev.USD) / prev.USD);
        if (prev.GBP && cur.GBP) changes.push(Math.abs(cur.GBP - prev.GBP) / prev.GBP);
        if (prev.JPY && cur.JPY) changes.push(Math.abs(cur.JPY - prev.JPY) / prev.JPY);
      }
      const avgChange = changes.reduce((a,b)=>a+b,0)/changes.length || 0.005;
      volScalar = Math.min(2.0, Math.max(0.5, avgChange / 0.005));
    }
    const nowUTC = new Date().getUTCHours() + new Date().getUTCMinutes()/60;
    _liqData = Array.from({length:48}, (_,i) => {
      if (isWeekend) return 2;
      const h=i/2, idx=Math.floor(h)%24, next=(idx+1)%24, frac=h-Math.floor(h);
      const v = LIQ_BASE[idx]*(1-frac)+LIQ_BASE[next]*frac;
      return Math.max(2, v * (h > nowUTC ? 0.75 : volScalar));
    });
    _liqBaseline = _liqTo48(isWeekend ? Array(24).fill(2) : LIQ_BASE);
    _liqSource   = 'Historical avg · fixed reference';
    return;
  } catch(e) { /* fall through */ }

  // ── Last resort: pure LIQ_BASE ────────────────────────────────────────────
  const base48 = _liqTo48(isWeekend ? Array(24).fill(2) : LIQ_BASE);
  _liqData     = base48;
  _liqBaseline = base48;
  _liqSource   = 'Historical avg · fixed reference';
}

function drawLiquidityChart() {
  const canvas = document.getElementById('liquidity-canvas');
  if (!canvas) return;
  // Batch layout read before any DOM write to avoid forced reflow
  const W = canvas.parentElement.clientWidth - 16, H = 110;
  // Assign dimensions in one batch — no DOM reads after this point until ctx ops
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  const utcDay = new Date().getUTCDay();
  const utcHour = new Date().getUTCHours();
  // Canvas left edge = 22:00 UTC (OFFSET=44 slots). Keep weekend mode until 22:00 UTC Sunday
  // so nowCanvasSlot starts at 0 (far left) on market open, not 47 (far right).
  const isWeekend = utcDay === 6 || (utcDay === 0 && utcHour < 22) || (utcDay === 5 && utcHour >= 21);

  const hours = _liqData || _liqTo48(isWeekend ? Array(24).fill(2) : LIQ_BASE);
  const baseline = _liqBaseline || hours;

  const PAD_L=4, PAD_R=4, PAD_T=8, PAD_B=18;
  const cW=W-PAD_L-PAD_R, cH=H-PAD_T-PAD_B;
  const maxV=Math.max(...hours, ...baseline, 10);

  // ── FX day starts at 22:00 UTC (Sydney open) ─────────────────────────────
  // OFFSET=44 slots (22h × 2). Canvas slot i → array slot (i+OFFSET)%48
  const OFFSET = 44; // 22:00 UTC in half-hour slots
  const sa = i => (i + OFFSET) % 48;            // slot in array from canvas position
  const sc = i => (i - OFFSET + 48) % 48;       // canvas position from array slot

  const px = i => PAD_L + (i / 47) * cW;        // canvas X from canvas slot i
  const py = v => PAD_T + (1 - v / maxV) * cH;

  // Current time in canvas-slot coordinates
  const nowH = new Date().getUTCHours() + new Date().getUTCMinutes() / 60;
  const nowArraySlot = Math.min(47, Math.floor(nowH * 2));
  const nowCanvasSlot = sc(nowArraySlot);
  const nowX = PAD_L + (nowCanvasSlot / 47) * cW;

  ctx.clearRect(0, 0, W, H);

  // Session bands — convert UTC slot boundaries to canvas coordinates
  // Sydney 22:00-07:00 UTC = array slots 44-14 (wraps)
  // Tokyo  00:00-09:00 UTC = array slots 0-18
  // London 08:00-17:00 UTC = array slots 16-34
  // NY     13:00-22:00 UTC = array slots 26-44
  if (!isWeekend) {
    const drawBand = (aStart, aEnd, color) => {
      // Convert array slots to canvas slots, handling wrap
      let cStart = sc(aStart), cEnd = sc(aEnd);
      if (cEnd <= cStart) cEnd = 47; // clamp wrap-arounds at right edge
      ctx.fillStyle = color;
      ctx.fillRect(PAD_L + (cStart/47)*cW, PAD_T, ((cEnd-cStart)/47)*cW, cH);
    };
    drawBand(44, 48+14, 'rgba(120,100,255,0.07)'); // Sydney (wraps — draw as 22→end)
    drawBand(0,  18,    'rgba(79,127,255,0.07)');    // Tokyo
    drawBand(16, 34,    'rgba(38,166,154,0.08)');   // London
    drawBand(26, 44,    'rgba(246,148,28,0.06)');   // NY
  }

  if (!isWeekend) {
    // ── PAST: filled area sólida ──────────────────────────────────────────
    const gradPast = ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    gradPast.addColorStop(0,'rgba(79,127,255,0.32)');
    gradPast.addColorStop(1,'rgba(79,127,255,0.03)');
    ctx.beginPath();
    for (let ci=0; ci<=nowCanvasSlot; ci++) {
      const v = hours[sa(ci)];
      ci===0 ? ctx.moveTo(px(ci),py(v)) : ctx.lineTo(px(ci),py(v));
    }
    ctx.lineTo(nowX,PAD_T+cH); ctx.lineTo(px(0),PAD_T+cH); ctx.closePath();
    ctx.fillStyle=gradPast; ctx.fill();

    // ── FUTURE: filled area tenue ─────────────────────────────────────────
    const gradFut = ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    gradFut.addColorStop(0,'rgba(79,127,255,0.10)');
    gradFut.addColorStop(1,'rgba(79,127,255,0.01)');
    ctx.beginPath();
    ctx.moveTo(nowX, py(hours[sa(nowCanvasSlot)]));
    for (let ci=nowCanvasSlot+1; ci<48; ci++) ctx.lineTo(px(ci),py(hours[sa(ci)]));
    ctx.lineTo(px(47),PAD_T+cH); ctx.lineTo(nowX,PAD_T+cH); ctx.closePath();
    ctx.fillStyle=gradFut; ctx.fill();

    // ── PAST: línea sólida azul ───────────────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle='#4f7fff'; ctx.lineWidth=1.5; ctx.setLineDash([]);
    for (let ci=0; ci<=nowCanvasSlot; ci++) {
      const v = hours[sa(ci)];
      ci===0 ? ctx.moveTo(px(ci),py(v)) : ctx.lineTo(px(ci),py(v));
    }
    ctx.stroke();

    // ── FUTURE: línea punteada azul tenue (datos: baseline 30d real) ─────
    ctx.beginPath(); ctx.strokeStyle='rgba(79,127,255,0.35)'; ctx.lineWidth=1.2; ctx.setLineDash([3,4]);
    ctx.moveTo(nowX, py(hours[sa(nowCanvasSlot)]));
    for (let ci=nowCanvasSlot+1; ci<48; ci++) ctx.lineTo(px(ci),py(hours[sa(ci)]));
    ctx.stroke(); ctx.setLineDash([]);

    // ── NOW-LINE ──────────────────────────────────────────────────────────
    ctx.strokeStyle='rgba(246,148,28,0.6)'; ctx.lineWidth=1; ctx.setLineDash([2,3]);
    ctx.beginPath(); ctx.moveTo(nowX,PAD_T); ctx.lineTo(nowX,PAD_T+cH); ctx.stroke();
    ctx.setLineDash([]);

  } else {
    // Weekend: curva plana, fill gris
    const grad=ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    grad.addColorStop(0,'rgba(120,123,134,0.15)'); grad.addColorStop(1,'rgba(79,127,255,0.03)');
    ctx.beginPath();
    for (let ci=0; ci<48; ci++) {
      const v = hours[sa(ci)];
      ci===0 ? ctx.moveTo(px(ci),py(v)) : ctx.lineTo(px(ci),py(v));
    }
    ctx.lineTo(px(47),PAD_T+cH); ctx.lineTo(px(0),PAD_T+cH); ctx.closePath();
    ctx.fillStyle=grad; ctx.fill();
    ctx.beginPath(); ctx.strokeStyle='#363c4e'; ctx.lineWidth=1.5;
    for (let ci=0; ci<48; ci++) {
      const v = hours[sa(ci)];
      ci===0 ? ctx.moveTo(px(ci),py(v)) : ctx.lineTo(px(ci),py(v));
    }
    ctx.stroke();
    ctx.fillStyle='rgba(120,123,134,0.5)'; ctx.font='9px Courier New'; ctx.textAlign='center';
    ctx.fillText('MARKET CLOSED — WEEKEND', W/2, PAD_T+cH/2);
  }

  // Hour labels — starting 22:00 UTC, every 4h: 22,02,06,10,14,18
  ctx.fillStyle='#4c525e'; ctx.font='8px Courier New'; ctx.textAlign='center';
  [{lbl:'22',ci:0},{lbl:'02',ci:8},{lbl:'06',ci:16},{lbl:'10',ci:24},{lbl:'14',ci:32},{lbl:'18',ci:40},{lbl:'22',ci:47}]
    .forEach(({lbl,ci}) => ctx.fillText(lbl, PAD_L+(ci/47)*cW, H-4));


  // Bottom labels
  const now = new Date();
  const localH = now.getHours().toString().padStart(2,'0');
  const localM = now.getMinutes().toString().padStart(2,'0');
  const tzShort = now.toLocaleTimeString('en',{timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('liq-time-label', localH + ':' + localM + ' ' + tzShort);
  if (isWeekend) {
    const reopenDate = new Date();
    const daysUntilSun = (7 - reopenDate.getUTCDay()) % 7;
    reopenDate.setUTCDate(reopenDate.getUTCDate() + (daysUntilSun === 0 ? 0 : daysUntilSun));
    reopenDate.setUTCHours(21, 0, 0, 0);
    const rH = reopenDate.getHours().toString().padStart(2,'0');
    const rM = reopenDate.getMinutes().toString().padStart(2,'0');
    setEl('liq-peak-label', 'Sun ' + rH + ':' + rM);
  } else {
    const peakCanvasSlot = hours.indexOf(Math.max(...hours));
    const peakArraySlot = (peakCanvasSlot + OFFSET) % 48;
    const peakUTC = new Date(); peakUTC.setUTCHours(Math.floor(peakArraySlot/2), peakArraySlot%2===0?0:30, 0, 0);
    const pH = peakUTC.getHours().toString().padStart(2,'0');
    const pM = peakUTC.getMinutes().toString().padStart(2,'0');
    setEl('liq-peak-label', 'Peak ' + pH + ':' + pM);
  }
}

// Initial load: fetch real data then draw
fetchLiquidityData().then(() => {
  if (_liqSource) setEl('liq-source-label', _liqSource);
  drawLiquidityChart();
});
// Refresh data every 30 min, redraw every 60 s
setInterval(() => fetchLiquidityData().then(() => {
  if (_liqSource) setEl('liq-source-label', _liqSource);
  drawLiquidityChart();
}), 30 * 60 * 1000);
setInterval(drawLiquidityChart, 60 * 1000);
window.addEventListener('resize', drawLiquidityChart);

// ── FX Liquidity tooltip ──────────────────────────────────────────────────────
(function() {
  const SESSION_NAMES = [
    { name:'Sydney',   start:22, end:7  },
    { name:'Tokyo',    start:0,  end:9  },
    { name:'London',   start:8,  end:17 },
    { name:'New York', start:13, end:22 },
  ];

  function getActiveSessions(utcH) {
    const active = SESSION_NAMES.filter(s => {
      if (s.end < s.start) return utcH >= s.start || utcH < s.end; // wraps midnight
      return utcH >= s.start && utcH < s.end;
    }).map(s => s.name);
    return active.length ? active.join(' + ') : 'Inter-session';
  }

  function volLabel(pct) {
    if (pct >= 85) return 'Very High (' + pct + '%)';
    if (pct >= 60) return 'High (' + pct + '%)';
    if (pct >= 35) return 'Moderate (' + pct + '%)';
    if (pct >= 15) return 'Low (' + pct + '%)';
    return 'Very Low (' + pct + '%)';
  }

  const canvas = document.getElementById('liquidity-canvas');
  const tooltip = document.getElementById('liq-tooltip');
  if (!canvas || !tooltip) return;

  canvas.addEventListener('mousemove', function(e) {
    const hours = _liqData;
    if (!hours) return;
    // Use baseline 30d as the reference max — gives a stable % across the day
    const baseline = _liqBaseline || hours;

    const rect = canvas.getBoundingClientRect();
    const PAD_L = 4, PAD_R = 4, PAD_T = 8, PAD_B = 18;
    const W = canvas.width, H = canvas.height;
    const cW = W - PAD_L - PAD_R;

    // Scale mouse X from CSS pixels to canvas pixels
    const scaleX = W / rect.width;
    const mouseX = (e.clientX - rect.left) * scaleX;
    if (mouseX < PAD_L || mouseX > W - PAD_R) { tooltip.style.display = 'none'; return; }

    // Map x → canvas slot (0–47). Canvas slot 0 = 22:00 UTC (OFFSET=44 array slots)
    const frac = (mouseX - PAD_L) / cW;
    const canvasSlot = Math.max(0, Math.min(47, Math.round(frac * 47)));
    const OFFSET = 44;
    const slot = (canvasSlot + OFFSET) % 48;  // array slot = UTC index
    const utcH = slot / 2;

    const hh = Math.floor(utcH).toString().padStart(2,'0');
    const mm = utcH % 1 === 0 ? '00' : '30';

    // Convert UTC slot to local time for display
    const d = new Date(); d.setUTCHours(Math.floor(utcH), utcH%1===0?0:30, 0, 0);
    const localHH = d.getHours().toString().padStart(2,'0');
    const localMM = d.getMinutes().toString().padStart(2,'0');
    const tzShort = d.toLocaleTimeString('en',{timeZoneName:'short'}).split(' ').pop() || 'LT';

    // % relative to baseline 30d peak (stable denominator across all hours)
    const maxBaseline = Math.max(...baseline, 10);
    const v    = hours[slot];
    const vRef = baseline[slot];
    const pct  = Math.round((v / maxBaseline) * 100);

    // Past vs future — compare in canvas-slot space
    const nowArraySlot = Math.floor(new Date().getUTCHours()*2 + new Date().getUTCMinutes()/30);
    const nowCanvasSlot = (nowArraySlot - OFFSET + 48) % 48;
    const isPast = canvasSlot <= nowCanvasSlot;

    // vs 30d avg comparison (only meaningful for past slots with real data)
    let vsAvg = '';
    if (isPast && vRef > 0 && _liqBaseline && _liqBaseline !== _liqData) {
      const diff = Math.round(((v - vRef) / vRef) * 100);
      if (diff > 8)       vsAvg = '  ↑ +' + diff + '% vs 30d avg';
      else if (diff < -8) vsAvg = '  ↓ ' + diff + '% vs 30d avg';
      else                vsAvg = '  ≈ in line with 30d avg';
    }

    // Read tooltip dimensions BEFORE writing textContent — avoids forced reflow
    const ttW = tooltip.style.display === 'block' ? (tooltip.offsetWidth || 170) : 170;
    const ttH = tooltip.style.display === 'block' ? (tooltip.offsetHeight || 56) : 56;

    document.getElementById('liq-tt-time').textContent = hh + ':' + mm + ' UTC  (' + localHH + ':' + localMM + ' ' + tzShort + ')';
    document.getElementById('liq-tt-session').textContent = '▸ ' + getActiveSessions(Math.floor(utcH));
    document.getElementById('liq-tt-vol').textContent = (isPast ? '⬤' : '○') + ' ' + (isPast ? '' : '(est.) ') + volLabel(pct) + vsAvg;

    // Position tooltip next to cursor using fixed coordinates
    let left = e.clientX + 14;
    let top  = e.clientY - ttH / 2;
    // Flip left if near right edge of viewport
    if (left + ttW > window.innerWidth - 8) left = e.clientX - ttW - 14;
    if (top < 4) top = 4;
    if (top + ttH > window.innerHeight - 4) top = window.innerHeight - ttH - 4;
    tooltip.style.left = left + 'px';
    tooltip.style.top  = top  + 'px';
    tooltip.style.display = 'block';
  });

  canvas.addEventListener('mouseleave', function() {
    tooltip.style.display = 'none';
  });
})();

// Risk + Cross-Asset run in parallel every 2 min — same as boot() — no chaining
setInterval(() => { fetchRiskData(); fetchCrossAssetData(); fetchCommodityQuotes(); fetchOptionSkew().then(() => attachRiskMonitorTooltips()); }, 2 * 60 * 1000);
setInterval(fetchCarryData,    30 * 60 * 1000);
setInterval(fetchCarryRanking, 30 * 60 * 1000);
// Refresh sentiment every 30 seconds
setInterval(fetchSentiment, 10 * 60 * 1000);   // every 10 min — sentiment source updates every 30min
// Refresh calendar & expectations every 30 minutes
setInterval(fetchFedExpectations, 30 * 60 * 1000);

// ═══════════════════════════════════════════════════════════════════
// MOBILE VISIBILITY FIX — TradingView widgets + FX Liquidity chart
// When the browser tab/app returns to foreground on mobile, iframes
// may go blank and canvas charts may render at wrong dimensions.
// We force a redraw whenever the page becomes visible again.
// ═══════════════════════════════════════════════════════════════════
(function() {
  // Helper: reload the active TradingView chart by fully re-creating its widget
  // (simulating a click doesn't work when the tab is already active)
  function reloadActiveTVChart() {
    const activeTab = document.querySelector('.tv-tab.active');
    if (!activeTab) return;
    const sym = activeTab.dataset.sym;
    if (!sym) {
      // Fallback: dispatch click if no sym data attribute
      activeTab.dispatchEvent(new MouseEvent('click', {bubbles: true}));
      return;
    }
    const wrap = document.getElementById('tv-chart-wrap');
    if (!wrap) return;
    wrap.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'tradingview-widget-container';
    container.style.cssText = 'height:100%;width:100%;';
    const widget = document.createElement('div');
    widget.className = 'tradingview-widget-container__widget';
    widget.style.cssText = 'height:100%;width:100%;';
    container.appendChild(widget);
    const copyright = document.createElement('div');
    copyright.className = 'tradingview-widget-copyright';
    copyright.style.display = 'none';
    container.appendChild(copyright);
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.text = JSON.stringify({
      allow_symbol_change:false, calendar:false, details:true,
      hide_side_toolbar:true, hide_top_toolbar:true, hide_legend:false,
      hide_volume:true, interval:'D', locale:'en', save_image:false,
      style:'1', symbol:sym, theme:'dark', timezone:'Etc/UTC',
      backgroundColor:'#131722', gridColor:'rgba(42,46,57,0.8)',
      withdateranges:false, studies:[{id:'MASimple@tv-basicstudies',inputs:{length:20}}], autosize:true
    });
    container.appendChild(script);
    wrap.appendChild(container);
  }

  // Helper: reload the Economic Calendar widget by re-injecting its script
  function reloadTVCalendar() {
    const scaleWrap = document.getElementById('tvcal-scale');
    if (!scaleWrap) return;
    // Remove existing iframe/content and re-create the widget container
    const container = scaleWrap.querySelector('.tradingview-widget-container');
    if (!container) return;
    const existingScript = container.querySelector('script');
    if (!existingScript) return;
    // Clone the widget container content to force re-init
    const clone = container.cloneNode(true);
    container.parentNode.replaceChild(clone, container);
  }

  // FX Liquidity chart: force redraw when visible
  function redrawLiquidityIfVisible() {
    const canvas = document.getElementById('liquidity-canvas');
    if (!canvas) return;
    // Only redraw if canvas has zero dimensions (collapsed/invisible at paint time)
    if (canvas.parentElement.clientWidth > 0) drawLiquidityChart();
  }

  // Detect mobile once (pointer: coarse covers phones + tablets)
  var isMobile = window.matchMedia('(pointer: coarse)').matches;

  // On page visibility change (tab switch, app background/foreground)
  document.addEventListener('visibilitychange', function() {
    if (document.visibilityState !== 'visible') return;
    // Small delay to let the browser re-paint before we measure dimensions
    setTimeout(function() {
      redrawLiquidityIfVisible();
      // Only reload TV widget on mobile when TV is actually active (_chartMode === 'tv').
      // When LW chart is active or loading (_chartMode === 'lw'), skip entirely —
      // LW Charts persists correctly across tab switches without needing recreation.
      if (isMobile && _chartMode === 'tv') {
        reloadActiveTVChart();
        setTimeout(reloadTVCalendar, 800);
      }
    }, 350);
  });

  // On pageshow (iOS Safari fires this when returning from bfcache)
  window.addEventListener('pageshow', function(e) {
    if (!e.persisted) return; // only for bfcache restores
    if (isMobile) {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }
    // Always reset right panel and main panel to top on bfcache restore
    const _rp = document.getElementById('rightpanel');
    if (_rp) _rp.scrollTop = 0;
    const _main = document.getElementById('main');
    if (_main) _main.scrollTop = 0;
    setTimeout(function() {
      redrawLiquidityIfVisible();
      // Same logic: only recreate TV widget if TV is currently active.
      // LW Charts survives bfcache restores without any reload.
      if (isMobile && _chartMode === 'tv') {
        reloadActiveTVChart();
        setTimeout(reloadTVCalendar, 800);
      }
    }, 350);
  });

  // IntersectionObserver: redraw liquidity chart the first time it enters viewport
  // (fixes the wrong-position bug when the chart is not visible on initial paint)
  const liqCanvas = document.getElementById('liquidity-canvas');
  if (liqCanvas && typeof IntersectionObserver !== 'undefined') {
    const obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          drawLiquidityChart();
          obs.unobserve(entry.target); // only needed once per session
        }
      });
    }, { threshold: 0.1 });
    obs.observe(liqCanvas);
  }
})();

// ═══════════════════════════════════════════════════════════════════
// ACCESSIBILITY — WCAG 2.1 AA enhancements
// ═══════════════════════════════════════════════════════════════════
(function initA11y() {
  // ── 1. Site menu: sync aria-expanded with :focus-within state ──
  const menuBtn = document.querySelector('.site-menu-btn');
  const siteMenu = document.querySelector('.site-menu');
  if (menuBtn && siteMenu) {
    // :focus-within shows the panel via CSS; mirror state in aria-expanded
    siteMenu.addEventListener('focusin',  () => menuBtn.setAttribute('aria-expanded', 'true'));
    siteMenu.addEventListener('focusout', (e) => {
      // Only collapse if focus left the entire .site-menu
      if (!siteMenu.contains(e.relatedTarget)) {
        menuBtn.setAttribute('aria-expanded', 'false');
      }
    });
    siteMenu.addEventListener('mouseenter', () => menuBtn.setAttribute('aria-expanded', 'true'));
    siteMenu.addEventListener('mouseleave', () => menuBtn.setAttribute('aria-expanded', 'false'));
  }

  // ── 2. Chart tabs: sync aria-selected on click ──
  const tablist = document.getElementById('tv-pair-tabs');
  if (tablist) {
    tablist.addEventListener('click', (e) => {
      const btn = e.target.closest('.tv-tab');
      if (!btn) return;
      tablist.querySelectorAll('.tv-tab').forEach(t => {
        t.setAttribute('aria-selected', t === btn ? 'true' : 'false');
      });
    });
  }

  // ── 3. Top-nav scroll links: add aria-current="page" to active ──
  const topNavLinks = document.querySelectorAll('.top-nav a');
  topNavLinks.forEach(link => {
    link.addEventListener('click', () => {
      topNavLinks.forEach(l => l.removeAttribute('aria-current'));
      link.setAttribute('aria-current', 'location');
    });
  });
  // Set initial aria-current on Overview
  const firstNavLink = document.querySelector('.top-nav a.active');
  if (firstNavLink) firstNavLink.setAttribute('aria-current', 'location');

  // ── 4. Live region: announce price updates to screen readers ──
  // A visually-hidden sr-only announcement div for dynamic price changes
  if (!document.getElementById('sr-announce')) {
    const announce = document.createElement('div');
    announce.id = 'sr-announce';
    announce.setAttribute('role', 'status');
    announce.setAttribute('aria-live', 'polite');
    announce.setAttribute('aria-atomic', 'true');
    announce.className = 'sr-only';
    document.body.appendChild(announce);
  }
})();

// ── CLS fix: hide skeleton placeholders once TradingView iframes load ──────
// Uses MutationObserver to detect when TV injects its iframe, then marks the
// skeleton as loaded (fades out via CSS transition).
(function () {
  function hideSkeleton(container) {
    const sk = container.querySelector('.tv-skeleton');
    if (!sk) return;
    sk.classList.add('loaded');
    // Remove from DOM after fade completes so it never blocks interaction
    setTimeout(() => sk.remove(), 350);
  }

  function watchForIframe(widgetEl) {
    if (!widgetEl) return;
    // If iframe already present (fast load), hide immediately
    if (widgetEl.querySelector('iframe')) {
      hideSkeleton(widgetEl);
      return;
    }
    const obs = new MutationObserver(() => {
      if (widgetEl.querySelector('iframe')) {
        obs.disconnect();
        hideSkeleton(widgetEl);
      }
    });
    obs.observe(widgetEl, { childList: true, subtree: true });
    // Safety fallback: hide after 8s regardless (slow connections / blocked TV)
    setTimeout(() => { obs.disconnect(); hideSkeleton(widgetEl); }, 8000);
  }

  // TV advanced chart
  watchForIframe(document.getElementById('tv-chart-widget'));
  // TV events calendar (skeleton is on tvcal-inner, iframe appears inside tvcal-scale)
  watchForIframe(document.getElementById('tvcal-inner'));
}());

// ═══════════════════════════════════════════════════════════════════
// TV WIDGET LAZY-LOADER
// IntersectionObserver boots each TradingView widget only when its
// container scrolls into view. Migrated from index.html inline script
// per GUIDELINES architecture rule (no inline JS in index.html).
// ═══════════════════════════════════════════════════════════════════
(function initTVWidgets() {
  var _chartLoaded   = false;
  var _eventsLoaded  = false;
  var _econmapLoaded = false;

  function loadTVEvents() {
    var scaleWrap = document.getElementById('tvcal-scale');
    if (!scaleWrap) return;
    var container = scaleWrap.querySelector('.tradingview-widget-container__widget');
    if (!container) return;
    var s = document.createElement('script');
    s.type = 'text/javascript';
    s.src  = 'https://s3.tradingview.com/external-embedding/embed-widget-events.js';
    s.async = true;
    s.textContent = JSON.stringify({
      colorTheme: 'dark', isTransparent: true, locale: 'en',
      countryFilter: 'us,nz,au,ch,eu,ca,jp,gb',
      importanceFilter: '-1,0,1', width: '100%', height: '100%'
    });
    var skel = document.querySelector('#tvcal-inner .tv-skeleton');
    if (skel) skel.style.display = 'none';
    container.appendChild(s);
    _eventsLoaded = true;
  }

  function loadTVEconMap() {
    var placeholder = document.getElementById('tv-econmap-placeholder');
    if (!placeholder) return;
    var s = document.createElement('script');
    s.type = 'module';
    s.src  = 'https://widgets.tradingview-widget.com/w/en/tv-economic-map.js';
    var widget = document.createElement('tv-economic-map');
    widget.setAttribute('theme', 'dark');
    widget.setAttribute('transparent', '');
    widget.style.cssText = 'width:100%;height:100%;min-height:380px;display:block;background:#131722;';
    placeholder.replaceWith(widget);
    document.head.appendChild(s);
    _econmapLoaded = true;
  }

  if (typeof IntersectionObserver === 'undefined') {
    // Fallback for very old browsers: load everything immediately
    if (typeof loadTVChart === 'function') loadTVChart(window._tvCurrentSym || 'FX_IDC:EURUSD');
    loadTVEvents();
    loadTVEconMap();
    return;
  }

  var io = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (!entry.isIntersecting) return;
      var id = entry.target.id;
      if (id === 'tv-chart-wrap' && !_chartLoaded) {
        if (typeof loadTVChart === 'function') {
          loadTVChart(window._tvCurrentSym || 'FX_IDC:EURUSD');
        }
        _chartLoaded = true;
        io.unobserve(entry.target);
      } else if (id === 'tvcal-inner' && !_eventsLoaded) {
        loadTVEvents();
        io.unobserve(entry.target);
      } else if (id === 'section-econmap' && !_econmapLoaded) {
        loadTVEconMap();
        io.unobserve(entry.target);
      }
    });
  }, { rootMargin: '150px' });

  // defer scripts run after DOM is parsed — DOMContentLoaded may have already fired.
  // Guard: if readyState is already 'interactive' or 'complete', attach observers immediately.
  function attachObservers() {
    var chartWrap = document.getElementById('tv-chart-wrap');
    var calInner  = document.getElementById('tvcal-inner');
    var econMap   = document.getElementById('section-econmap');
    if (chartWrap) io.observe(chartWrap);
    if (calInner)  io.observe(calInner);
    if (econMap)   io.observe(econMap);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachObservers);
  } else {
    attachObservers();
  }
}());

// ═══════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// G → FX table   C → COT   R → Risk   X → Cross-Asset
// M → Macro      Y → Rates  K → Calendar
// ↑ / ↓ → navigate FX table rows (loads chart)
// ? → toggle shortcut legend overlay
// ═══════════════════════════════════════════════════════════════════
(function initKeyboardShortcuts() {
  const NAV_KEYS = {
    g: 'section-fxpairs',
    c: 'section-positioning',
    r: 'section-risk',
    x: 'section-crossasset',
    m: 'section-econmap',
    y: 'section-cbrates',
    k: 'section-tvcalendar',
    d: 'section-derivatives',
    n: 'section-news',
  };

  function navTo(target) {
    if (target === 'section-derivatives') {
      // Derivatives uses a custom show/hide toggle, not scroll-into-view
      const derivSection = window._derivNavSection;
      if (!derivSection) return;
      if (derivSection.style.display === 'none' || derivSection.style.display === '') {
        // If currently hidden, show it
        if (typeof window._derivNavShow === 'function') window._derivNavShow();
      } else {
        // Already visible — treat D as a toggle back to Overview
        if (typeof window._derivNavHide === 'function') window._derivNavHide();
      }
      return;
    }
    if (target === 'section-news') {
      // News uses same show/hide toggle pattern as Derivatives
      const newsSection = window._newsNavSection;
      if (!newsSection) return;
      if (newsSection.style.display === 'none' || newsSection.style.display === '') {
        if (typeof window._newsNavShow === 'function') window._newsNavShow();
      } else {
        if (typeof window._newsNavHide === 'function') window._newsNavHide();
      }
      return;
    }
    const link = document.querySelector(`.top-nav a[data-target="${target}"]`);
    if (link) link.click();
  }

  // FX table row navigation
  let _focusedRow = -1;

  function fxRows() {
    return Array.from(document.querySelectorAll('#fx-pairs-tbody tr[data-sym]'));
  }

  function activateFxRow(idx) {
    const rows = fxRows();
    if (!rows.length) return;
    _focusedRow = Math.max(0, Math.min(idx, rows.length - 1));
    const row = rows[_focusedRow];
    rows.forEach(r => r.classList.remove('kb-focus'));
    row.classList.add('kb-focus');
    row.scrollIntoView({ block: 'nearest' });
    const sym = row.dataset.sym;
    if (sym) loadTVChart(sym);
  }

  // Shortcut legend overlay
  function toggleLegend() {
    let overlay = document.getElementById('kb-legend');
    if (overlay) { overlay.remove(); return; }
    overlay = document.createElement('div');
    overlay.id = 'kb-legend';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Keyboard shortcuts');
    overlay.innerHTML = `
      <div class="kbl-inner">
        <div class="kbl-title">Keyboard shortcuts</div>
        <div class="kbl-grid">
          <span class="kbl-key">G</span><span class="kbl-desc">FX Pairs table</span>
          <span class="kbl-key">C</span><span class="kbl-desc">COT Positioning</span>
          <span class="kbl-key">R</span><span class="kbl-desc">Risk Monitor</span>
          <span class="kbl-key">X</span><span class="kbl-desc">Cross-Asset</span>
          <span class="kbl-key">M</span><span class="kbl-desc">Macro map</span>
          <span class="kbl-key">Y</span><span class="kbl-desc">Rates &amp; Yield Curve</span>
          <span class="kbl-key">K</span><span class="kbl-desc">Economic Calendar</span>
          <span class="kbl-key">D</span><span class="kbl-desc">Derivatives (toggle)</span>
          <span class="kbl-key">N</span><span class="kbl-desc">News Feed (toggle)</span>
          <span class="kbl-key">&uarr;&darr;</span><span class="kbl-desc">Navigate FX rows</span>
          <span class="kbl-key">?</span><span class="kbl-desc">Close this panel</span>
        </div>
        <div class="kbl-footer">Press any key or click to close</div>
      </div>`;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', () => overlay.remove());
  }

  // Main keydown handler
  document.addEventListener('keydown', e => {
    // Never intercept browser/OS shortcuts (Ctrl, Meta, Alt combos)
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select'
        || document.activeElement?.isContentEditable) return;

    const key = e.key;

    if (key === '?') { e.preventDefault(); toggleLegend(); return; }

    // Close legend on any key if open
    const legend = document.getElementById('kb-legend');
    if (legend && key !== '?') { legend.remove(); }

    if (NAV_KEYS[key.toLowerCase()]) {
      e.preventDefault();
      navTo(NAV_KEYS[key.toLowerCase()]);
      return;
    }

    if (key === 'ArrowDown') {
      e.preventDefault();
      activateFxRow(_focusedRow < 0 ? 0 : _focusedRow + 1);
      return;
    }
    if (key === 'ArrowUp') {
      e.preventDefault();
      activateFxRow(_focusedRow <= 0 ? 0 : _focusedRow - 1);
      return;
    }
  });
})();

// ═══════════════════════════════════════════════════════════════════
// CSV / JSON EXPORT
// ═══════════════════════════════════════════════════════════════════
// EXPORT BUTTON WIRING
// Uses addEventListener instead of onclick="" attributes to avoid
// inline handler restrictions in Edge Enhanced Tracking Prevention.
// ═══════════════════════════════════════════════════════════════════
(function wireExportButtons() {
  function bind(id, type, format) {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      exportPanel(type, format);
    });
  }
  bind('export-fx-csv',   'fx',    'csv');
  bind('export-fx-json',  'fx',    'json');
  bind('export-cot-csv',  'cot',   'csv');
  bind('export-cot-json', 'cot',   'json');
}());

// exportPanel(type, format) — reads in-memory caches, triggers download
// Types: 'fx' | 'cot' | 'yield' | 'carry'   Format: 'csv' | 'json'
// ═══════════════════════════════════════════════════════════════════
function exportPanel(type, format = 'csv') {
  const ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
  let rows, headers, filename;

  if (type === 'fx') {
    headers = ['Pair', 'Price', '1D_Pct', '1W_Pct', 'HV30', 'Session_High', 'Session_Low'];
    rows = PAIRS.map(p => {
      const rt  = STOOQ_RT_CACHE[p.id];
      const p1w = rt?.pct1w ?? null;
      return [
        p.label || (p.base + '/' + p.quote),
        rt?.close  ?? '',
        rt?.pct    != null ? rt.pct.toFixed(4)  : '',
        p1w        != null ? p1w.toFixed(4)      : '',
        rt?.hv30   != null ? rt.hv30.toFixed(2) : '',
        rt?.high   ?? '',
        rt?.low    ?? '',
      ];
    }).filter(r => r[1] !== '');
    filename = 'gi_fx_pairs_' + ts;
  }

  else if (type === 'cot') {
    headers = ['Currency', 'LF_Net', 'Long_Pct', 'Short_Pct', 'AM_Net', 'Week_Ending'];
    rows = Object.entries(COT_DATA_CACHE).map(([ccy, d]) => {
      const total = (d.long || 0) + (d.short || 0);
      const lPct  = total > 0 ? (d.long  / total * 100).toFixed(1) : '';
      const sPct  = total > 0 ? (d.short / total * 100).toFixed(1) : '';
      return [ccy, d.net ?? '', lPct, sPct, d.amNet ?? '', d.weekEnding ?? ''];
    });
    filename = 'gi_cot_' + ts;
  }

  else if (type === 'yield') {
    headers = ['Tenor', 'Yield_Pct', 'Change'];
    rows = [];
    // Read from rendered DOM rows
    document.querySelectorAll('#yield-tbody tr, #yield-table-body tr').forEach(tr => {
      const cells = tr.querySelectorAll('td');
      if (cells.length >= 2) {
        const t = cells[0]?.textContent?.trim() || '';
        const y = cells[1]?.textContent?.trim() || '';
        const c = cells[2]?.textContent?.trim() || '';
        if (t && y) rows.push([t, y, c]);
      }
    });
    // Fallback: named yield cells
    if (!rows.length) {
      [['US 3M','yc-3m'],['US 2Y','yc-2y'],['US 5Y','yc-5y'],
       ['US 10Y','yc-10y'],['US 30Y','yc-30y'],['DE 10Y','yc-de10y'],['JP 10Y','yc-jp10y']
      ].forEach(([label, id]) => {
        const el = document.getElementById(id);
        const v = el?.textContent?.trim();
        if (v && v !== '—') rows.push([label, v, '']);
      });
    }
    filename = 'gi_yield_curve_' + ts;
  }

  else if (type === 'carry') {
    headers = ['Long', 'Short', 'Carry_Diff_Pct', 'Long_Rate_Pct', 'Short_Rate_Pct'];
    const G8 = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD'];
    const rates = {};
    G8.forEach(ccy => {
      const r = STATE.cbRates?.[ccy.toLowerCase()]?.rate;
      if (r != null) rates[ccy] = r;
    });
    const pairs = [];
    for (let i = 0; i < G8.length; i++) {
      for (let j = i + 1; j < G8.length; j++) {
        const a = G8[i], b = G8[j];
        if (rates[a] == null || rates[b] == null) continue;
        const diff = rates[a] - rates[b];
        pairs.push(diff >= 0
          ? [a, b, diff.toFixed(4), rates[a].toFixed(2), rates[b].toFixed(2)]
          : [b, a, (-diff).toFixed(4), rates[b].toFixed(2), rates[a].toFixed(2)]);
      }
    }
    pairs.sort((a, b) => parseFloat(b[2]) - parseFloat(a[2]));
    rows = pairs;
    filename = 'gi_carry_' + ts;
  }

  else { console.warn('[Export] Unknown panel type:', type); return; }

  if (!rows || !rows.length) {
    // Visual feedback — flash the button that triggered this export
    document.querySelectorAll('.export-btn').forEach(b => {
      if (b.textContent.trim() === format.toUpperCase()) {
        const orig = b.textContent;
        b.textContent = 'NO DATA'; b.style.color = 'var(--orange)';
        setTimeout(() => { b.textContent = orig; b.style.color = ''; }, 1800);
      }
    });
    console.warn('[Export] No data available for:', type);
    return;
  }

  let blob_content, mime, ext;
  if (format === 'json') {
    const data = rows.map(r => {
      const obj = {};
      headers.forEach((h, i) => { obj[h] = r[i] !== '' ? r[i] : null; });
      return obj;
    });
    blob_content = JSON.stringify({ exported: new Date().toISOString(), panel: type, data }, null, 2);
    mime = 'application/json';
    ext = '.json';
  } else {
    const esc = v => (v == null || v === '') ? '' : String(v).includes(',') ? '"' + String(v) + '"' : String(v);
    blob_content = [headers, ...rows].map(r => r.map(esc).join(',')).join('\r\n');
    mime = 'text/csv';
    ext = '.csv';
  }

  // Use data: URL instead of blob: URL — Edge Enhanced Tracking Prevention silently
  // blocks programmatic blob: URL navigation triggered by a.click(), whereas
  // data: URLs are not subject to the same restriction.
  const encoded = 'data:' + mime + ';charset=utf-8,' + encodeURIComponent(blob_content);
  const a    = document.createElement('a');
  a.href = encoded;
  a.download = filename + ext;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => document.body.removeChild(a), 500);

  // Visual feedback — flash ✓ on every matching button in this panel
  document.querySelectorAll('.export-btn').forEach(b => {
    if (b.textContent.trim() === ext.slice(1).toUpperCase()) {
      const orig = b.textContent;
      b.textContent = '✓'; b.style.color = 'var(--up)';
      setTimeout(() => { b.textContent = orig; b.style.color = ''; }, 1400);
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// CONFIGURABLE ALERTS — threshold monitoring with Notifications API
// ═══════════════════════════════════════════════════════════════════
// Storage: localStorage key 'gi_alerts' → JSON array of alert objects
//
// Alert types:
//   PRICE  { type:'price',  sym, dir:'above'|'below', threshold }
//   SPREAD { type:'spread', sym, dir:'above'|'below', threshold }
//          sym = 'hv_iv_eurusd' | 'hv_iv_gbpusd' | 'hv_iv_usdjpy' | 'hv_iv_audusd'
//          Fires when HV30 > ATM IV (vol is cheap) or HV30 < ATM IV (vol is expensive)
//   IVRANK { type:'ivrank', sym, dir:'above'|'below', threshold }
//          sym = 'ivrank_eurusd' | etc.  threshold 0–100
//   REGIME { type:'regime', target:'RISK-OFF'|'CAUTION'|'MIXED'|'RISK-ON' }
//          Fires when computed live regime matches target
//   CORR   { type:'corr',   pair, dir:'above'|'below', threshold }
//          pair = 'usdjpy_vix' | 'dxy_spx' | 'gold_dxy' etc. (z-score threshold)
//   VAR    { type:'var',    sym, dir:'above'|'below', threshold }
//          Fires when current 1d VaR95% crosses threshold
// ═══════════════════════════════════════════════════════════════════

const ALERTS_KEY = 'gi_alerts';

// Price-alert labels (legacy + extended)
const ALERTS_LABELS = {
  vix:'VIX', eurusd:'EUR/USD', usdjpy:'USD/JPY', gbpusd:'GBP/USD',
  audusd:'AUD/USD', usdchf:'USD/CHF', xauusd:'Gold', us10y:'US 10Y', move:'MOVE',
  nzdusd:'NZD/USD', usdcad:'USD/CAD', dxy:'DXY', spx:'SPX', wti:'WTI', btc:'BTC',
};

// ── Advanced alert type definitions ──────────────────────────────────────────

const ADV_ALERT_TYPES = {
  // ── HV30 vs ATM IV spread alerts ────────────────────────────────────
  'hv_iv_eurusd': {
    label: 'EUR/USD HV30 vs IV', category: 'spread',
    description: 'Fires when realised vol (HV30) diverges from implied vol (ATM IV). HV > IV = vol is cheap; HV < IV = vol is expensive.',
    getValue(intra) {
      const hv = STOOQ_RT_CACHE['eurusd']?.hv30 ?? null;
      const iv = intra?.fx_etf_iv?.eurusd?.iv ?? null;
      return (hv != null && iv != null) ? parseFloat((hv - iv).toFixed(2)) : null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)} vol pts`,
  },
  'hv_iv_gbpusd': {
    label: 'GBP/USD HV30 vs IV', category: 'spread',
    description: 'HV30 minus ATM IV for GBP/USD. Positive = realised vol above implied (vol cheap).',
    getValue(intra) {
      const hv = STOOQ_RT_CACHE['gbpusd']?.hv30 ?? null;
      const iv = intra?.fx_etf_iv?.gbpusd?.iv ?? null;
      return (hv != null && iv != null) ? parseFloat((hv - iv).toFixed(2)) : null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)} vol pts`,
  },
  'hv_iv_usdjpy': {
    label: 'USD/JPY HV30 vs IV', category: 'spread',
    description: 'HV30 minus ATM IV for USD/JPY. Positive = vol cheap relative to implied.',
    getValue(intra) {
      const hv = STOOQ_RT_CACHE['usdjpy']?.hv30 ?? null;
      const iv = intra?.fx_etf_iv?.usdjpy?.iv ?? null;
      return (hv != null && iv != null) ? parseFloat((hv - iv).toFixed(2)) : null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)} vol pts`,
  },
  'hv_iv_audusd': {
    label: 'AUD/USD HV30 vs IV', category: 'spread',
    description: 'HV30 minus ATM IV for AUD/USD.',
    getValue(intra) {
      const hv = STOOQ_RT_CACHE['audusd']?.hv30 ?? null;
      const iv = intra?.fx_etf_iv?.audusd?.iv ?? null;
      return (hv != null && iv != null) ? parseFloat((hv - iv).toFixed(2)) : null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)} vol pts`,
  },

  // ── IV Rank alerts ───────────────────────────────────────────────────
  'ivrank_eurusd': {
    label: 'EUR/USD IV Rank', category: 'ivrank',
    description: 'IV Rank 0–100. Above 70 = historically expensive vol. Below 30 = historically cheap vol.',
    getValue(intra) { return intra?.fx_etf_iv?.eurusd?.iv_rank ?? null; },
    formatValue: v => `${v.toFixed(0)} rnk`,
  },
  'ivrank_gbpusd': {
    label: 'GBP/USD IV Rank', category: 'ivrank',
    description: 'IV Rank for GBP/USD (0–100 scale).',
    getValue(intra) { return intra?.fx_etf_iv?.gbpusd?.iv_rank ?? null; },
    formatValue: v => `${v.toFixed(0)} rnk`,
  },
  'ivrank_usdjpy': {
    label: 'USD/JPY IV Rank', category: 'ivrank',
    description: 'IV Rank for USD/JPY.',
    getValue(intra) { return intra?.fx_etf_iv?.usdjpy?.iv_rank ?? null; },
    formatValue: v => `${v.toFixed(0)} rnk`,
  },
  'ivrank_audusd': {
    label: 'AUD/USD IV Rank', category: 'ivrank',
    description: 'IV Rank for AUD/USD.',
    getValue(intra) { return intra?.fx_etf_iv?.audusd?.iv_rank ?? null; },
    formatValue: v => `${v.toFixed(0)} rnk`,
  },

  // ── Correlation Z-score break alerts ────────────────────────────────
  'corr_usdjpy_vix': {
    label: 'USD/JPY vs VIX corr Z', category: 'corr',
    description: 'Z-score of rolling 60d correlation between USD/JPY and VIX vs its 252d historical norm. |Z| > 1.5 = regime break.',
    getValue(intra) {
      const c = (intra?.correlations || []).find(r => r.a === 'USD/JPY' && r.b === 'VIX');
      return c?.z_score ?? null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}σ`,
  },
  'corr_dxy_spx': {
    label: 'DXY vs SPX corr Z', category: 'corr',
    description: 'Z-score of DXY/SPX rolling correlation. Positive = both rising together (USD funding stress).',
    getValue(intra) {
      const c = (intra?.correlations || []).find(r => r.a === 'DXY' && r.b === 'SPX');
      return c?.z_score ?? null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}σ`,
  },
  'corr_gold_dxy': {
    label: 'Gold vs DXY corr Z', category: 'corr',
    description: 'Z-score of Gold/DXY rolling correlation. Positive break = Gold and USD rising together (inflation/safe-haven demand).',
    getValue(intra) {
      const c = (intra?.correlations || []).find(r => r.a === 'Gold' && r.b === 'DXY');
      return c?.z_score ?? null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}σ`,
  },
  'corr_audusd_gold': {
    label: 'AUD/USD vs Gold corr Z', category: 'corr',
    description: 'Z-score of AUD/USD vs Gold correlation. Break signals China/domestic risk overriding the commodity link.',
    getValue(intra) {
      const c = (intra?.correlations || []).find(r => r.a === 'AUD/USD' && r.b === 'Gold');
      return c?.z_score ?? null;
    },
    formatValue: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}σ`,
  },

  // ── Historical VaR 95% alerts ────────────────────────────────────────
  'var_eurusd': {
    label: 'EUR/USD VaR 95% (1d)', category: 'var',
    description: '1-day Historical VaR 95% for EUR/USD, expressed as % of price. Rises during stressed regimes.',
    getValue(intra) { return intra?.var_cvar?.eurusd?.var_pct ?? null; },
    formatValue: v => `${v.toFixed(3)}%`,
  },
  'var_usdjpy': {
    label: 'USD/JPY VaR 95% (1d)', category: 'var',
    description: '1-day Historical VaR 95% for USD/JPY.',
    getValue(intra) { return intra?.var_cvar?.usdjpy?.var_pct ?? null; },
    formatValue: v => `${v.toFixed(3)}%`,
  },
  'var_gbpusd': {
    label: 'GBP/USD VaR 95% (1d)', category: 'var',
    description: '1-day Historical VaR 95% for GBP/USD.',
    getValue(intra) { return intra?.var_cvar?.gbpusd?.var_pct ?? null; },
    formatValue: v => `${v.toFixed(3)}%`,
  },
  'var_xauusd': {
    label: 'Gold VaR 95% (1d)', category: 'var',
    description: '1-day Historical VaR 95% for Gold (XAU/USD).',
    getValue(intra) { return intra?.var_cvar?.gold?.var_pct ?? null; },
    formatValue: v => `${v.toFixed(3)}%`,
  },
  'var_spx': {
    label: 'SPX VaR 95% (1d)', category: 'var',
    description: '1-day Historical VaR 95% for S&P 500.',
    getValue(intra) { return intra?.var_cvar?.spx?.var_pct ?? null; },
    formatValue: v => `${v.toFixed(3)}%`,
  },
};

// ── Regime alert — special singleton type ────────────────────────────────────
// Stored as { type:'regime', id, target:'RISK-OFF'|'CAUTION'|'MIXED'|'RISK-ON', fired, firedAt }
// Evaluated against the live computed regime (DOM element #risk-regime)
function _liveRegime() {
  return document.getElementById('risk-regime')?.textContent?.trim() ?? null;
}

// Expose to window for inline onchange handlers in the popover HTML
window._ADV_OPTS      = ADV_ALERT_TYPES;
window.ALERTS_LABELS  = ALERTS_LABELS;

// ── Signal Notifications — browser push for new AI signals ────────────────────
// Storage: localStorage key 'gi_sig_notif' → 'on' | 'off'  (default: 'off')
// Tracks last-seen signal fingerprint to detect new signals on each 15-min refresh.
const SIG_NOTIF_KEY      = 'gi_sig_notif';
const SIG_NOTIF_SEEN_KEY = 'gi_sig_seen';   // fingerprint of last-rendered signal set

function sigNotifEnabled() {
  return localStorage.getItem(SIG_NOTIF_KEY) === 'on';
}

function updateSignalNotifBtn() {
  const btn   = document.getElementById('sig-notif-btn');
  if (!btn) return;
  const on      = sigNotifEnabled();
  const blocked = typeof Notification !== 'undefined' && Notification.permission === 'denied';
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.setAttribute('aria-label', on ? 'Signal notifications on' : 'Signal notifications off');
  btn.classList.toggle('sig-notif-on',      on && !blocked);
  btn.classList.toggle('sig-notif-blocked', blocked);
  btn.title = blocked
    ? 'Notifications blocked by browser — enable in site settings'
    : on ? 'Signal notifications ON — click to disable' : 'Signal notifications OFF — click to enable';
}

async function toggleSignalNotifications() {
  const wasOn = sigNotifEnabled();
  if (!wasOn) {
    if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
      const perm = await Notification.requestPermission();
      if (perm !== 'granted') { updateSignalNotifBtn(); return; }
    }
    if (typeof Notification !== 'undefined' && Notification.permission === 'denied') {
      updateSignalNotifBtn(); return;
    }
    localStorage.setItem(SIG_NOTIF_KEY, 'on');
  } else {
    localStorage.setItem(SIG_NOTIF_KEY, 'off');
  }
  updateSignalNotifBtn();
}

function sigFingerprint(signals) {
  if (!Array.isArray(signals) || !signals.length) return '';
  return signals.map(s => `${s.time}|${s.title}|${s.priority}`).join(';;');
}

function maybeNotifyNewSignals(signals) {
  if (!sigNotifEnabled()) return;
  if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;
  const fp     = sigFingerprint(signals);
  if (!fp) return;
  const lastFp = localStorage.getItem(SIG_NOTIF_SEEN_KEY) || '';
  if (!lastFp) {
    // First load — record baseline only, no notification
    localStorage.setItem(SIG_NOTIF_SEEN_KEY, fp);
    return;
  }
  if (fp === lastFp) return;
  localStorage.setItem(SIG_NOTIF_SEEN_KEY, fp);
  const critCount = signals.filter(s => s.priority === 'critical').length;
  const warnCount = signals.filter(s => s.priority === 'warning').length;
  const parts = [];
  if (critCount) parts.push(`${critCount} critical`);
  if (warnCount) parts.push(`${warnCount} warning`);
  const body = parts.length
    ? `${signals.length} signals — ${parts.join(', ')}`
    : `${signals.length} market signals updated`;
  try {
    new Notification('GI Terminal — New Signals', {
      body,
      icon: '/favicon-192x192.png',
      tag : 'gi-signals-update',
    });
  } catch {}
}

function alertsLoad() {
  try { return JSON.parse(localStorage.getItem(ALERTS_KEY) || '[]'); } catch { return []; }
}
function alertsSave(arr) {
  try { localStorage.setItem(ALERTS_KEY, JSON.stringify(arr)); } catch {}
}

// ── Value resolvers ───────────────────────────────────────────────────────────

function alertsCurrentValue(a, intra) {
  if (a.type === 'price' || !a.type) {
    // Legacy + new price alerts
    const sym = a.sym;
    if (sym === 'vix')   return STOOQ_RT_CACHE['vix']?.close  ?? null;
    if (sym === 'move')  return STOOQ_RT_CACHE['move']?.close ?? null;
    if (sym === 'us10y') {
      const el = document.getElementById('yc-10y');
      const v  = parseFloat(el?.textContent);
      return isNaN(v) ? null : v;
    }
    return STOOQ_RT_CACHE[sym]?.close ?? null;
  }
  if (a.type === 'regime') {
    return _liveRegime();
  }
  // All advanced types require intraday data
  const def = ADV_ALERT_TYPES[a.sym];
  if (!def) return null;
  return def.getValue(intra);
}

function alertFormatValue(a, v) {
  if (v == null) return null;
  if (a.type === 'regime') return v;
  const def = ADV_ALERT_TYPES[a.sym];
  if (def?.formatValue) return def.formatValue(v);
  // Price alert: standard numeric
  return v.toFixed(v > 10 ? 2 : 5);
}

function alertDescribeCondition(a) {
  if (a.type === 'regime') return `Regime = ${a.target}`;
  const label = ADV_ALERT_TYPES[a.sym]?.label ?? ALERTS_LABELS[a.sym] ?? a.sym;
  const dirSym = a.dir === 'above' ? '>' : '<';
  return `${label} ${dirSym} ${a.threshold}`;
}

// ── Render ────────────────────────────────────────────────────────────────────

function alertsRender(intra) {
  const container = document.getElementById('alerts-rows');
  if (!container) return;
  const arr = alertsLoad();

  const firedCount = arr.filter(a => a.fired).length;
  const badge = document.getElementById('alerts-fired-badge');
  if (badge) {
    badge.textContent = firedCount;
    badge.style.display = firedCount > 0 ? 'inline-block' : 'none';
  }

  if (!arr.length) {
    container.innerHTML = '<div style="padding:5px 8px;font-size:10px;color:var(--text3);">No alerts set. Add one below.</div>';
    return;
  }

  container.innerHTML = arr.map(a => {
    const cls      = a.fired ? 'alert-row alert-row-active' : 'alert-row';
    const firedTxt = a.fired ? ` <span class="alert-fired">⚡ FIRED ${a.firedAt || ''}</span>` : '';
    const cur      = alertsCurrentValue(a, intra);
    const curFmt   = alertFormatValue(a, cur);
    const curTxt   = curFmt != null ? ` · now ${curFmt}` : '';
    const condTxt  = alertDescribeCondition(a);
    // Category badge
    const cat = a.type === 'regime' ? 'regime' : (ADV_ALERT_TYPES[a.sym]?.category ?? 'price');
    const catColors = { price:'var(--text2)', spread:'#1D9E75', ivrank:'#185FA5', corr:'#854F0B', var:'#A32D2D', regime:'#533AB7' };
    const catStyle  = `color:${catColors[cat]||'var(--text2)'};font-size:9px;margin-right:4px;`;
    return `<div class="${cls}" data-id="${a.id}">
      <span class="alert-lbl"><span style="${catStyle}">[${cat.toUpperCase()}]</span>${condTxt}${curTxt}${firedTxt}</span>
      <span class="alert-del" title="Remove alert" onclick="alertsRemove('${a.id}')">✕</span>
    </div>`;
  }).join('');
}

function alertsRemove(id) {
  alertsSave(alertsLoad().filter(a => a.id !== id));
  alertsRender(null);
}

// ── Add from UI ───────────────────────────────────────────────────────────────

function alertsAddFromUI() {
  const typeEl  = document.getElementById('alert-type-sel');
  const symEl   = document.getElementById('alert-sym-sel');
  const dirEl   = document.getElementById('alert-dir-sel');
  const valEl   = document.getElementById('alert-val-inp');
  const regEl   = document.getElementById('alert-regime-sel');

  const alertType = typeEl?.value || 'price';

  const arr = alertsLoad();

  if (alertType === 'regime') {
    const target = regEl?.value;
    if (!target) return;
    // Only one regime alert per target
    if (arr.find(a => a.type === 'regime' && a.target === target)) return;
    arr.push({ id: Date.now().toString(36), type: 'regime', target, fired: false, firedAt: null });
  } else {
    const sym = symEl?.value;
    const dir = dirEl?.value;
    const val = parseFloat(valEl?.value);
    if (!sym || !dir || isNaN(val)) return;
    const label = ADV_ALERT_TYPES[sym]?.label ?? ALERTS_LABELS[sym] ?? sym;
    arr.push({ id: Date.now().toString(36), type: alertType, sym, dir, threshold: val, label, fired: false, firedAt: null });
    if (valEl) valEl.value = '';
  }

  alertsSave(arr);
  alertsRender(null);
  if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
    Notification.requestPermission();
  }
}

// ── Check cycle ───────────────────────────────────────────────────────────────

async function alertsCheck() {
  const arr = alertsLoad();
  if (!arr.length) return;

  // Load intraday data once for all advanced alerts (uses 90s cache — no extra fetch)
  let intra = null;
  const needsIntra = arr.some(a => a.type && a.type !== 'price' && a.type !== 'regime');
  if (needsIntra) {
    intra = await loadIntradayQuotes().catch(() => null);
  }

  let changed = false;

  arr.forEach(a => {
    if (a.fired) return;
    const cur = alertsCurrentValue(a, intra);
    if (cur == null) return;

    let triggered = false;

    if (a.type === 'regime') {
      triggered = (cur === a.target);
    } else {
      // All numeric types: price, spread, ivrank, corr, var
      triggered = (a.dir === 'above' && cur > a.threshold) ||
                  (a.dir === 'below' && cur < a.threshold);
    }

    if (!triggered) return;

    a.fired   = true;
    a.firedAt = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    changed   = true;

    // Browser notification
    if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      const curFmt  = alertFormatValue(a, cur);
      const condTxt = alertDescribeCondition(a);
      const body    = curFmt ? `${condTxt}  ·  Now: ${curFmt}` : condTxt;
      try {
        new Notification('GI Terminal Alert', {
          body,
          icon: '/favicon-192x192.png',
          tag : 'gi-alert-' + a.id,
        });
      } catch {}
    }
  });

  if (changed) { alertsSave(arr); alertsRender(intra); }
}

function initAlerts() {
  alertsRender(null);
  // Delay the initial check so fetchRiskData / fetchCrossAssetData have time to
  // populate STOOQ_RT_CACHE before alertsCurrentValue() reads from it.
  // Without this, the very first check always returns cur==null for every price
  // alert and silently skips them — the 5-min interval then works correctly, but
  // the first evaluation on page load is always a no-op.
  // 8 s is well within the observed p95 round-trip for fetchQuoteBarRT (~2–3 s)
  // and fetchRiskData (~3–5 s), so the cache is reliably warm by then.
  setTimeout(alertsCheck, 8000);
  setInterval(alertsCheck, 5 * 60 * 1000);

  // Init signal notification button state from localStorage
  updateSignalNotifBtn();

  // Close popover when clicking outside — bubble phase so button onclick fires first
  document.addEventListener('click', e => {
    const anchor = document.getElementById('alerts-anchor');
    if (anchor && !anchor.contains(e.target)) {
      const pop = document.getElementById('alerts-popover');
      if (pop) pop.style.display = 'none';
      const btn = document.getElementById('alerts-bell-btn');
      if (btn) btn.setAttribute('aria-expanded', 'false');
    }
  });
}

function toggleAlertsPopover() {
  const pop = document.getElementById('alerts-popover');
  const btn = document.getElementById('alerts-bell-btn');
  if (!pop) return;
  const isOpen = pop.style.display !== 'none';
  if (isOpen) {
    pop.style.display = 'none';
    if (btn) btn.setAttribute('aria-expanded', 'false');
    return;
  }
  // Position above the button using fixed coords (escapes overflow:hidden parents)
  alertsRender();
  pop.style.display = 'block';
  if (btn) btn.setAttribute('aria-expanded', 'true');
  const rect = btn.getBoundingClientRect();
  const popW = 280;
  const PAD = 8;
  let left = rect.right - popW;
  if (left < PAD) left = PAD;
  pop.style.left = left + 'px';
  pop.style.top  = (rect.top - pop.offsetHeight - 8) + 'px';
  // Re-adjust after render (offsetHeight may be 0 before display:block reflow)
  requestAnimationFrame(() => {
    const h = pop.offsetHeight;
    pop.style.top = (rect.top - h - 8) + 'px';
  });
}

// ═══════════════════════════════════════════════════════════════════
// SPLIT LAYOUT — vertical left/right toggle + drag handle resize
// Migrated from inline <script> in index.html (v7.26.0)
// ═══════════════════════════════════════════════════════════════════
(function initSplitLayout(){
  var LS_KEY = 'gi_split_layout';
  var main   = document.getElementById('main');
  var btn    = document.getElementById('split-layout-btn');
  var handle = document.getElementById('split-drag-handle');
  var upper  = document.getElementById('split-upper');
  var lower  = document.getElementById('split-lower');

  var alertsPanel      = document.getElementById('section-macro');
  var alertsOrigParent = alertsPanel ? alertsPanel.parentNode : null;
  var alertsOrigNext   = alertsPanel ? alertsPanel.nextSibling : null;

  function isMobile(){ return window.innerWidth <= 900; }

  function applyState(active, leftPct){
    if(!main||!btn||!handle||!upper||!lower) return;
    if(isMobile()) active = false;
    btn.style.display = isMobile() ? 'none' : '';
    if(active){
      main.classList.add('split-layout');
      btn.classList.remove('active');
      btn.setAttribute('aria-pressed','true');
      handle.style.display = '';
      var pct = leftPct || 55;
      upper.style.width = pct + '%';
      upper.style.flex  = 'none';
      if(alertsPanel && alertsPanel.parentNode !== upper){
        upper.appendChild(alertsPanel);
      }
    } else {
      main.classList.remove('split-layout');
      btn.classList.add('active');
      btn.setAttribute('aria-pressed','false');
      handle.style.display = 'none';
      upper.style.width = '';
      upper.style.flex  = '';
      if(alertsPanel && alertsOrigParent && alertsPanel.parentNode !== alertsOrigParent){
        alertsOrigParent.insertBefore(alertsPanel, alertsOrigNext);
      }
    }
  }

  try {
    var saved = JSON.parse(localStorage.getItem(LS_KEY)||'null');
    if(saved === null){
      applyState(true, 55);
      localStorage.setItem(LS_KEY, JSON.stringify({active:true, leftPct:55}));
    } else if(saved.active){
      applyState(true, saved.leftPct||55);
    } else {
      applyState(false);
    }
  } catch(e){ applyState(true, 55); }

  var TIP_KEY = 'gi_split_tip_seen';
  var tip = document.getElementById('split-tip');
  function hideTip(){
    if(!tip) return;
    tip.classList.remove('visible');
    try { localStorage.setItem(TIP_KEY,'1'); } catch(e){}
  }
  try {
    if(!localStorage.getItem(TIP_KEY) && tip){
      setTimeout(function(){ tip.classList.add('visible'); }, 800);
      setTimeout(function(){ hideTip(); }, 6000);
    }
  } catch(e){}

  if(btn){
    btn.addEventListener('click', function(){
      hideTip();
      var isActive = main.classList.contains('split-layout');
      applyState(!isActive, 55);
      try { localStorage.setItem(LS_KEY, JSON.stringify({active:!isActive, leftPct:55})); } catch(e){}
    });
  }

  window.addEventListener('resize', function(){
    btn.style.display = isMobile() ? 'none' : '';
    if(isMobile() && main.classList.contains('split-layout')){
      applyState(false);
    }
  });

  // ── ResizeObserver on #layout: fixes snap/restore layout collapse ─────────
  // When the user uses OS window snap (Win+Left/Right, macOS Stage Manager,
  // browser split-view) and then restores to full screen, the CSS grid can
  // enter a broken state that window.resize alone doesn't recover from.
  // A ResizeObserver on #layout detects the actual element width change and
  // forces a style reflow via a class toggle — the standard industry fix.
  (function _watchLayoutResize(){
    var layout = document.getElementById('layout');
    if(!layout || typeof ResizeObserver === 'undefined') return;
    var _lastW = layout.offsetWidth;
    var _rafPending = false;
    var ro = new ResizeObserver(function(entries){
      if(_rafPending) return;
      var newW = entries[0].contentRect.width;
      // Only act on meaningful width changes (>20px) to avoid micro-reflows
      if(Math.abs(newW - _lastW) < 20) return;
      _lastW = newW;
      _rafPending = true;
      requestAnimationFrame(function(){
        _rafPending = false;
        // Force grid reflow: toggle a class that adds/removes display:contents
        layout.classList.add('_reflow');
        requestAnimationFrame(function(){ layout.classList.remove('_reflow'); });
        // Re-apply split state so widths recalculate correctly
        var isActive = main.classList.contains('split-layout');
        if(isActive){
          var pct = upper.offsetWidth > 0
            ? parseFloat((upper.offsetWidth / main.offsetWidth * 100).toFixed(1))
            : 55;
          upper.style.width = pct + '%';
        }
      });
    });
    ro.observe(layout);
  })();
  // When the browser window moves to a monitor with a different resolution or
  // DPR, the CSS grid layout (#layout: 180px minmax(0,1fr) 220px) can enter an
  // irrecoverable broken state where #main collapses to ~220px. No JS reflow
  // can reliably fix a broken grid mid-paint. The correct solution is to reload
  // the page when a screen change is detected. The reload is fast (all assets
  // are cached) and the user returns to the same state via localStorage.
  (function _watchScreenChange(){
    var _lastW = window.screen.width;
    var _lastH = window.screen.height;
    var _lastDPR = window.devicePixelRatio;
    var _reloadPending = false;

    function _onScreenChange(){
      if(_reloadPending) return;
      var w = window.screen.width;
      var h = window.screen.height;
      var dpr = window.devicePixelRatio;
      // Only reload if screen dimensions changed (rules out normal browser resize)
      if(w !== _lastW || h !== _lastH || Math.abs(dpr - _lastDPR) > 0.05){
        _reloadPending = true;
        // Small delay so the browser finishes moving the window before reload
        setTimeout(function(){ window.location.reload(); }, 300);
      }
    }

    // Primary: matchMedia on DPR — fires reliably on monitor change
    try{
      var _mq = window.matchMedia('(resolution: ' + _lastDPR + 'dppx)');
      _mq.addEventListener('change', _onScreenChange);
    }catch(e){}

    // Secondary: poll screen dimensions every 2s as fallback
    setInterval(function(){
      if(!_reloadPending) _onScreenChange();
    }, 2000);
  })();

  if(handle){
    var dragging = false, startX = 0, startW = 0;
    handle.addEventListener('mousedown', function(e){
      dragging = true;
      startX = e.clientX;
      startW = upper.offsetWidth;
      handle.classList.add('dragging');
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });
    document.addEventListener('mousemove', function(e){
      if(!dragging) return;
      var dx    = e.clientX - startX;
      var mainW = main.offsetWidth;
      var newW  = Math.min(Math.max(startW + dx, 240), mainW - 200);
      var pct   = (newW / mainW * 100).toFixed(1);
      upper.style.width = pct + '%';
    });
    document.addEventListener('mouseup', function(){
      if(!dragging) return;
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.userSelect = '';
      var pct = parseFloat((upper.offsetWidth / main.offsetWidth * 100).toFixed(1));
      try { localStorage.setItem(LS_KEY, JSON.stringify({active:true, leftPct:pct})); } catch(e){}
    });
    handle.addEventListener('touchstart', function(e){
      dragging = true;
      startX = e.touches[0].clientX;
      startW = upper.offsetWidth;
      handle.classList.add('dragging');
    }, {passive:true});
    document.addEventListener('touchmove', function(e){
      if(!dragging) return;
      var dx    = e.touches[0].clientX - startX;
      var mainW = main.offsetWidth;
      var newW  = Math.min(Math.max(startW + dx, 240), mainW - 200);
      upper.style.width = (newW / mainW * 100).toFixed(1) + '%';
    }, {passive:true});
    document.addEventListener('touchend', function(){
      if(!dragging) return;
      dragging = false;
      handle.classList.remove('dragging');
    });
  }
})();

// ── Onboarding Tooltip — surfaces the alerts feature to first-time users ──────
// Shows once after a 4-second delay on first visit (no existing alerts configured
// and no prior dismissal). Dismissed permanently via localStorage key 'gi_ob_done'.
// "SET ALERT" button: requests notification permission, adds a REGIME→RISK-OFF
// alert, opens the alerts popover briefly so the user sees it was added, then
// dismisses the tooltip.

const GI_OB_KEY = 'gi_ob_done';

function giOnboardShouldShow() {
  // Already dismissed or acted upon
  if (localStorage.getItem(GI_OB_KEY)) return false;
  // Welcome tour must complete first — don't compete visually with the 3-step tour
  try { if (!localStorage.getItem('gi_welcome_done')) return false; } catch { /* ignore */ }
  // User already has alerts configured — they know the feature exists
  try {
    const existing = JSON.parse(localStorage.getItem('gi_alerts') || '[]');
    if (existing.length > 0) return false;
  } catch { /* ignore */ }
  return true;
}

function giOnboardDismiss() {
  localStorage.setItem(GI_OB_KEY, '1');
  const el = document.getElementById('gi-onboard');
  if (el) {
    el.style.opacity = '0';
    el.style.transition = 'opacity .2s';
    setTimeout(() => { el.style.display = 'none'; }, 220);
  }
}

async function giOnboardActivate() {
  const btn = document.getElementById('gi-onboard-cta');
  if (btn) { btn.textContent = '…'; btn.disabled = true; }

  // Request browser notification permission
  if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
    await Notification.requestPermission();
  }

  // Add REGIME → RISK-OFF alert directly
  try {
    const arr = alertsLoad();
    const alreadyHasRegime = arr.some(a => a.type === 'regime' && a.target === 'RISK-OFF');
    if (!alreadyHasRegime) {
      arr.push({
        id: Date.now().toString(36),
        type: 'regime',
        target: 'RISK-OFF',
        label: 'Regime: RISK-OFF',
        fired: false,
        firedAt: null
      });
      alertsSave(arr);
      alertsRender(null);
    }
  } catch (e) {
    console.warn('giOnboardActivate: could not add alert', e);
  }

  // Open alerts popover briefly so user sees the alert was added
  const pop = document.getElementById('alerts-popover');
  const bellBtn = document.getElementById('alerts-bell-btn');
  if (pop && bellBtn) {
    toggleAlertsPopover();
    // Scroll popover into view in case it's off-screen
    setTimeout(() => {
      pop.scrollIntoView && pop.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
  }

  // Mark onboarding done and hide tooltip
  giOnboardDismiss();
}

function giOnboardInit() {
  if (!giOnboardShouldShow()) return;
  // Delay 4s — let the terminal finish loading data so it doesn't compete visually
  setTimeout(() => {
    if (!giOnboardShouldShow()) return; // re-check in case state changed during load
    const el = document.getElementById('gi-onboard');
    if (!el) return;
    el.style.opacity = '0';
    el.style.display = 'block';
    // Fade in
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.style.transition = 'opacity .35s ease';
        el.style.opacity = '1';
      });
    });
    // Auto-dismiss after 18s if user ignores it (non-intrusive)
    setTimeout(() => {
      if (el.style.display !== 'none') giOnboardDismiss();
    }, 18000);
  }, 4000);
}

// Hook into DOMContentLoaded — dashboard.js is deferred so DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', giOnboardInit);
} else {
  giOnboardInit();
}

// ═══════════════════════════════════════════════════════════════════
// NEW FEATURES v7.71.0 — CIP Forwards, RR Surface, HV Term Structure,
//                         G8 Rates tabs, Sovereign Spreads, Econ Surprises
// ═══════════════════════════════════════════════════════════════════

// ── Global cache: CB rates by currency (populated by fetchRiskData/renderCBRates) ──
window._CB_RATES_CACHE = window._CB_RATES_CACHE || {};

// ── OIS / Overnight rate cache (used exclusively for CIP forward pricing) ──
// Populated by loadOISRatesCache() from ois-rates/rates.json (daily workflow).
// Falls back to _CB_RATES_CACHE (policy rate) if file unavailable.
// Rate → benchmark: USD=SOFR, EUR=€STR, GBP=SONIA, JPY=TONA,
//                   AUD=AONIA, CAD=CORRA, CHF=SARON, NZD=OCR overnight.
window._OIS_RATES_CACHE  = window._OIS_RATES_CACHE  || {};
window._OIS_RATE_SOURCES = window._OIS_RATE_SOURCES || {};  // e.g. { USD: 'SOFR', EUR: '€STR' }

// ── CIP Forward Calculator ──
// F = S × (1 + r_RIGHT × T) / (1 + r_LEFT × T)
// r_left  = OIS rate of left-hand (base) currency
// r_right = OIS rate of right-hand (quote) currency
// T in years (1M=1/12, 3M=1/4, 6M=1/2, 1Y=1)
// Industry standard: use overnight/OIS benchmarks, not CB policy rates.
// Benchmarks: USD=SOFR, EUR=€STR, GBP=SONIA, JPY=TONA, AUD=AONIA, CAD=CORRA, CHF=SARON, NZD=OCR.
// Source: BIS FX conventions; Bloomberg FX Forward methodology (FXFA).
function computeCIPForward(spot, rLeft, rRight, T) {
  if (spot == null || rLeft == null || rRight == null) return null;
  const rL = rLeft  / 100;
  const rR = rRight / 100;
  return spot * ((1 + rR * T) / (1 + rL * T));
}

// ── Helper: resolve rate for a currency (OIS preferred, policy fallback) ──
// Returns [rate, sourceName] — sourceName used in tooltips.
function _resolveRate(ccy) {
  const ois = window._OIS_RATES_CACHE[ccy];
  if (ois != null) return [ois, window._OIS_RATE_SOURCES[ccy] || 'OIS'];
  const policy = window._CB_RATES_CACHE[ccy];
  if (policy != null) return [policy, 'policy'];
  return [null, null];
}

// ── Rate map: which CB rate applies to which currency ──
// CIP-eligible pairs — both legs have CB policy rates in rates/*.json
// Formula: F = S × (1 + r_RIGHT × T) / (1 + r_LEFT × T)
// Left-hand currency at forward discount when its rate exceeds the right-hand rate.
const CIP_CCY_RATES = new Set([
  'EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','NZD/USD',
  'EUR/GBP','EUR/JPY','GBP/JPY','AUD/JPY','EUR/AUD','EUR/CHF',
]);

// ── Render CIP Forwards in main FX Pairs table (tds[7]=Fwd1M, tds[8]=Fwd3M) ──
async function renderCIPForwards() {
  const fxTbody = document.getElementById('fx-pairs-tbody');
  if (!fxTbody) return;

  // Use for...of instead of forEach so we can await inside the loop
  // (needed for the STOOQ_RT_CACHE fallback to loadIntradayQuotes)
  const rows = fxTbody.querySelectorAll('tr');
  for (const row of rows) {
    const symCell = row.querySelector('td.sym');
    if (!symCell) continue;
    const pair = symCell.textContent.trim();
    if (!CIP_CCY_RATES.has(pair)) continue;
    const tds = row.querySelectorAll('td');
    if (tds.length < 12) continue;

    const [leftCcy, rightCcy] = pair.split('/');
    const pairId = pair.replace('/', '').toLowerCase();

    // Primary: read from STOOQ_RT_CACHE (populated by fetchQuoteBarRT).
    // Fallback: call loadIntradayQuotes() which has a 90-second in-memory cache —
    // near-zero cost if already loaded, and avoids the race condition on first render.
    let spot = STOOQ_RT_CACHE[pairId]?.close ?? null;
    if (spot == null) {
      try {
        const freshIntra = await loadIntradayQuotes().catch(() => null);
        if (freshIntra) {
          const matched = Object.entries(freshIntra).find(
            ([k]) => k.toLowerCase().replace('=x', '').replace('-', '').replace('/', '') === pairId
          );
          if (matched) spot = matched[1]?.close ?? matched[1]?.price ?? null;
        }
      } catch { /* stay null — cells render as — */ }
    }

    const [rLeft,  srcLeft]  = _resolveRate(leftCcy);
    const [rRight, srcRight] = _resolveRate(rightCcy);

    const tenors  = [1/12, 3/12];
    const indices = [7, 8];
    const pairCfg = PAIRS.find(p => p.id === pairId);
    const dec = pairCfg?.dec ?? 4;

    tenors.forEach((T, i) => {
      const fwd = computeCIPForward(spot, rLeft, rRight, T);
      const el  = tds[indices[i]];
      if (!el) return;
      if (fwd != null && spot != null) {
        el.textContent = fwd.toFixed(dec);
        const atDiscount = fwd < spot;
        el.style.color = atDiscount ? 'var(--down)' : 'var(--up)';
        const tLabel = T === 1/12 ? '1M' : '3M';
        el.title = `CIP ${tLabel} fwd · ${leftCcy}=${rLeft?.toFixed(2)}% (${srcLeft}) vs ${rightCcy}=${rRight?.toFixed(2)}% (${srcRight}) · ${leftCcy} at forward ${atDiscount ? 'discount' : 'premium'}`;
      } else {
        el.textContent = '—';
        el.style.color = 'var(--text3)';
      }
    });
  }
}

// ── Render RR 1M in main FX Pairs table (tds[9]) ──
async function renderRRInFXTable() {
  // RR_DATA_CACHE is populated by fetchOptionSkew — but also fetch directly as fallback
  let rrMap = window.RR_DATA_CACHE || {};
  if (Object.keys(rrMap).length === 0) {
    try {
      const res = await fetch('./rr-data/rr.json').catch(() => null);
      if (res?.ok) {
        const j = await res.json();
        if (j?.pairs) { rrMap = j.pairs; Object.assign(window.RR_DATA_CACHE, rrMap); }
      }
    } catch { /* leave empty */ }
  }
  const fxTbody = document.getElementById('fx-pairs-tbody');
  if (!fxTbody) return;

  const rrKeys = {
    'EUR/USD': 'EURUSD', 'GBP/USD': 'GBPUSD', 'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD', 'USD/CHF': 'USDCHF', 'USD/CAD': 'USDCAD', 'NZD/USD': 'NZDUSD'
  };

  const rows = fxTbody.querySelectorAll('tr');
  rows.forEach(row => {
    const symCell = row.querySelector('td.sym');
    if (!symCell) return;
    const pair = symCell.textContent.trim();
    const rrKey = rrKeys[pair];
    if (!rrKey) return;
    const tds = row.querySelectorAll('td');
    const el = tds[9];
    if (!el) return;

    const rrVal = rrMap[rrKey]?.rr25d ?? null;
    if (rrVal != null) {
      el.textContent = (rrVal >= 0 ? '+' : '') + rrVal.toFixed(2);
      el.style.color = rrVal > 0.1 ? 'var(--up)' : rrVal < -0.1 ? 'var(--down)' : 'var(--text2)';
      el.title = `25d RR 1M · ${rrKey} · Saxo Bank indicative mid`;
    } else {
      el.textContent = '—';
      el.style.color = 'var(--text3)';
    }
  });
}

// ── Render Derivatives section ──
async function renderDerivativesSection() {
  const ratesCache = window._CB_RATES_CACHE;
  const intraday = await loadIntradayQuotes().catch(() => null);

  // Guarantee RR data is available — fetch directly if cache is still empty
  let rrMap = window.RR_DATA_CACHE || {};
  if (Object.keys(rrMap).length === 0) {
    try {
      const res = await fetch('./rr-data/rr.json').catch(() => null);
      if (res?.ok) {
        const j = await res.json();
        if (j?.pairs) {
          rrMap = j.pairs;
          if (!window.RR_DATA_CACHE) window.RR_DATA_CACHE = {};
          Object.assign(window.RR_DATA_CACHE, rrMap);
        }
      }
    } catch { /* leave empty, cells show — */ }
  } else {
    rrMap = window.RR_DATA_CACHE;
  }

  // Load rr2.json if available (multi-tenor from fetch_saxo_rr2.py)
  let rr2Map = {};
  try {
    const rr2Res = await fetch('./rr-data/rr2.json').catch(() => null);
    if (rr2Res?.ok) {
      const rr2Json = await rr2Res.json();
      if (rr2Json?.pairs) rr2Map = rr2Json.pairs;
    }
  } catch { /* rr2.json not yet deployed — graceful fallback */ }

  const pairs = ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','NZD/USD'];
  const rrKeys = {
    'EUR/USD':'EURUSD','GBP/USD':'GBPUSD','USD/JPY':'USDJPY',
    'AUD/USD':'AUDUSD','USD/CHF':'USDCHF','USD/CAD':'USDCAD','NZD/USD':'NZDUSD'
  };

  // ── Forwards table ──
  const fwdTbody = document.getElementById('fwd-tbody');
  if (fwdTbody) {
    const rows = fwdTbody.querySelectorAll('tr');
    pairs.forEach((pair, idx) => {
      const row = rows[idx];
      if (!row) return;
      if (!CIP_CCY_RATES.has(pair)) return;
      const [leftCcy, rightCcy] = pair.split('/');
      const pairId = pair.replace('/','').toLowerCase();
      const pairCfg = PAIRS.find(p => p.id === pairId);
      const dec = pairCfg?.dec ?? 4;
      const spot  = STOOQ_RT_CACHE[pairId]?.close ?? intraday?.quotes?.[pairId]?.close ?? null;

      // ── OIS rates (preferred) with policy fallback ──
      const [rLeft,  srcLeft]  = _resolveRate(leftCcy);
      const [rRight, srcRight] = _resolveRate(rightCcy);

      const tds = row.querySelectorAll('td');

      // Spot
      if (tds[1]) tds[1].textContent = spot != null ? spot.toFixed(dec) : '—';

      // Forwards: 1M, 3M, 6M, 1Y
      const tenors = [1/12, 3/12, 6/12, 1];
      tenors.forEach((T, ti) => {
        const fwd = computeCIPForward(spot, rLeft, rRight, T);
        const el = tds[2 + ti];
        if (!el) return;
        if (fwd != null && spot != null) {
          el.textContent = fwd.toFixed(dec);
          const atDiscount = fwd < spot; // left-hand ccy at discount
          el.style.color = atDiscount ? 'var(--down)' : 'var(--up)';
        } else {
          el.textContent = '—';
          el.style.color = 'var(--text3)';
        }
      });

      // Rate Diff — OIS diff (positive = left has more carry → forward discount)
      if (tds[6]) {
        const diff = (rLeft != null && rRight != null) ? (rLeft - rRight) : null;
        if (diff != null) {
          tds[6].textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2) + '%';
          tds[6].style.color = diff > 0.1 ? 'var(--down)' : diff < -0.1 ? 'var(--up)' : 'var(--text2)';
          tds[6].title = `OIS rate diff: ${leftCcy}=${rLeft?.toFixed(2)}% (${srcLeft}) − ${rightCcy}=${rRight?.toFixed(2)}% (${srcRight}) · positive = ${leftCcy} at forward discount · Used for CIP forward pricing`;
        } else {
          tds[6].textContent = '—';
        }
      }
    });

    // ── Cross pairs CIP forwards ──
    const crossFwdPairs = ['EUR/GBP','EUR/JPY','GBP/JPY','AUD/JPY','EUR/AUD','EUR/CHF'];
    crossFwdPairs.forEach(pair => {
      const row = fwdTbody.querySelector(`tr[data-pair="${pair}"]`);
      if (!row) return;
      const [leftCcy, rightCcy] = pair.split('/');
      const pairId = pair.replace('/','').toLowerCase();
      const pairCfg = PAIRS.find(p => p.id === pairId);
      const dec = pairCfg?.dec ?? 5;
      const spot  = STOOQ_RT_CACHE[pairId]?.close ?? intraday?.quotes?.[pairId]?.close ?? null;

      // ── OIS rates (preferred) with policy fallback ──
      const [rLeft,  srcLeft]  = _resolveRate(leftCcy);
      const [rRight, srcRight] = _resolveRate(rightCcy);

      const tds = row.querySelectorAll('td');

      if (tds[1]) tds[1].textContent = spot != null ? spot.toFixed(dec) : '—';

      const tenors = [1/12, 3/12, 6/12, 1];
      tenors.forEach((T, ti) => {
        const fwd = computeCIPForward(spot, rLeft, rRight, T);
        const el = tds[2 + ti];
        if (!el) return;
        if (fwd != null && spot != null) {
          el.textContent = fwd.toFixed(dec);
          const atDiscount = fwd < spot;
          el.style.color = atDiscount ? 'var(--down)' : 'var(--up)';
        } else {
          el.textContent = '—';
          el.style.color = 'var(--text3)';
        }
      });

      if (tds[6]) {
        const diff = (rLeft != null && rRight != null) ? (rLeft - rRight) : null;
        if (diff != null) {
          tds[6].textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2) + '%';
          tds[6].style.color = diff > 0.1 ? 'var(--down)' : diff < -0.1 ? 'var(--up)' : 'var(--text2)';
          tds[6].title = `OIS rate diff: ${leftCcy}=${rLeft?.toFixed(2)}% (${srcLeft}) − ${rightCcy}=${rRight?.toFixed(2)}% (${srcRight}) · positive = ${leftCcy} at forward discount`;
        } else {
          tds[6].textContent = '—';
        }
      }
    });
  }

  // ── RR Surface table ──
  const rrSurfaceTbody = document.getElementById('rr-surface-tbody');
  if (rrSurfaceTbody) {
    // EUR/JPY is the only cross pair Saxo consistently publishes — include it.
    // NZD/USD excluded: Saxo does not publish NZD/USD 25d RR on their public page.
    const rrPairs = [...pairs.filter(p => p !== 'NZD/USD'), 'EUR/JPY'];
    const rrPairKeys = { ...rrKeys, 'EUR/JPY': 'EURJPY' };
    const rows = rrSurfaceTbody.querySelectorAll('tr');
    rrPairs.forEach((pair, idx) => {
      const row = rows[idx];
      if (!row) return;
      const rrKey = rrPairKeys[pair];
      const tds = row.querySelectorAll('td');
      const rr2 = rr2Map[rrKey] || {};
      const rr1m = rr2['1M'] ?? rrMap[rrKey]?.rr25d ?? null;
      const tenorData = [
        rr2['1W'] ?? null,
        rr1m,
        rr2['3M'] ?? null,
        rr2['6M'] ?? null,
        rr2['1Y'] ?? null,
      ];
      tenorData.forEach((v, ti) => {
        const el = tds[1 + ti];
        if (!el) return;
        if (v != null) {
          el.textContent = (v >= 0 ? '+' : '') + v.toFixed(2);
          el.style.color = v > 0.1 ? 'var(--up)' : v < -0.1 ? 'var(--down)' : 'var(--text2)';
        } else {
          el.textContent = '—';
          el.style.color = 'var(--text3)';
        }
      });
      // Skew direction
      if (tds[6] && rr1m != null) {
        const skewLbl = rr1m < -0.3 ? 'Put skew' : rr1m > 0.3 ? 'Call skew' : 'Balanced';
        tds[6].textContent = skewLbl;
        tds[6].style.color = rr1m < -0.3 ? 'var(--down)' : rr1m > 0.3 ? 'var(--up)' : 'var(--text3)';
      }
    });
  }

  // ── HV Term Structure table — 4 columns: Pair | HV 30d | RR 1M | RR/HV ──
  const hvTermTbody = document.getElementById('hv-term-tbody');
  if (hvTermTbody) {
    const rows = hvTermTbody.querySelectorAll('tr');
    const termPairs = ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','NZD/USD'];
    termPairs.forEach((pair, idx) => {
      const row = rows[idx];
      if (!row) return;
      const pairId = pair.replace('/','').toLowerCase();
      const q = intraday?.quotes?.[pairId];
      const tds = row.querySelectorAll('td');

      const hv30 = q?.hv30 ?? STOOQ_RT_CACHE[pairId]?.hv30 ?? null;
      const hv10 = q?.hv10 ?? null;
      const rrKey = rrKeys[pair] ?? pair.replace('/','');
      const rr1m = rrMap[rrKey]?.rr25d ?? null;

      // td[1] = HV 30d
      if (tds[1]) {
        tds[1].textContent = hv30 != null ? hv30.toFixed(1) + '%' : '—';
        tds[1].style.textAlign = 'right';
        tds[1].style.color = hv30 != null ? (hv30 > 12 ? 'var(--down)' : hv30 < 5 ? 'var(--up)' : 'var(--text)') : 'var(--text3)';
        tds[1].style.fontFamily = 'var(--font-mono)';
        tds[1].style.fontSize = '10px';
      }
      // td[2] = RR 1M
      if (tds[2]) {
        tds[2].textContent = rr1m != null ? (rr1m >= 0 ? '+' : '') + rr1m.toFixed(2) : '—';
        tds[2].style.textAlign = 'right';
        tds[2].style.color = rr1m != null ? (rr1m > 0.1 ? 'var(--up)' : rr1m < -0.1 ? 'var(--down)' : 'var(--text2)') : 'var(--text3)';
        tds[2].style.fontFamily = 'var(--font-mono)';
        tds[2].style.fontSize = '10px';
      }
      // td[3] = RR/HV ratio — skew premium relative to realized vol
      if (tds[3]) {
        if (rr1m != null && hv30 != null && hv30 > 0) {
          const ratio = (rr1m / hv30) * 100;
          tds[3].textContent = (ratio >= 0 ? '+' : '') + ratio.toFixed(0) + '%';
          tds[3].style.color = ratio > 5 ? 'var(--up)' : ratio < -5 ? 'var(--down)' : 'var(--text3)';
          tds[3].title = `RR 1M (${rr1m.toFixed(2)}) ÷ HV30 (${hv30.toFixed(1)}%) — options skew premium vs realized vol`;
        } else {
          tds[3].textContent = '—';
          tds[3].style.color = 'var(--text3)';
        }
        tds[3].style.textAlign = 'right';
        tds[3].style.fontFamily = 'var(--font-mono)';
        tds[3].style.fontSize = '10px';
      }
      // td[4] = Vol Trend — Bloomberg convention: HV 10d vs HV 30d
      // ↑ expanding (HV10 > HV30 + 1pp), ↓ contracting (HV10 < HV30 − 1pp), → neutral
      if (tds[4]) {
        if (hv10 != null && hv30 != null) {
          const diff = hv10 - hv30;
          let arrow, color, tip;
          if (diff > 1) {
            arrow = '↑'; color = 'var(--down)';  // expanding vol = risk-off color (red)
            tip = `HV10 (${hv10.toFixed(1)}%) > HV30 (${hv30.toFixed(1)}%) — short-term vol expanding`;
          } else if (diff < -1) {
            arrow = '↓'; color = 'var(--up)';    // contracting vol = green
            tip = `HV10 (${hv10.toFixed(1)}%) < HV30 (${hv30.toFixed(1)}%) — short-term vol contracting`;
          } else {
            arrow = '→'; color = 'var(--text3)';
            tip = `HV10 (${hv10.toFixed(1)}%) ≈ HV30 (${hv30.toFixed(1)}%) — vol stable (within 1pp)`;
          }
          tds[4].textContent = arrow;
          tds[4].style.color = color;
          tds[4].title = tip;
        } else {
          tds[4].textContent = '—';
          tds[4].style.color = 'var(--text3)';
          tds[4].title = 'HV 10d not yet available — pipeline computes on next run';
        }
        tds[4].style.textAlign = 'right';
        tds[4].style.fontSize = '11px';
      }
    });
  }

  // ── Cross-Pair Vol Monitor ──
  const crossVolTbody = document.getElementById('cross-vol-tbody');
  if (crossVolTbody && intraday) {
    const crossPairs = [
      { label: 'EUR/GBP', id: 'eurgbp' },
      { label: 'EUR/JPY', id: 'eurjpy' },
      { label: 'GBP/JPY', id: 'gbpjpy' },
      { label: 'AUD/JPY', id: 'audjpy' },
      { label: 'EUR/AUD', id: 'euraud' },
      { label: 'EUR/NZD', id: 'eurnzd' },
    ];
    const rows = crossVolTbody.querySelectorAll('tr');
    crossPairs.forEach((cp, idx) => {
      const row = rows[idx];
      if (!row) return;
      const q = intraday?.quotes?.[cp.id];
      const tds = row.querySelectorAll('td');
      const hv30 = q?.hv30 ?? null;
      const hv10 = q?.hv10 ?? null;
      const pct  = q?.pct  ?? null;

      // HV 30d
      if (tds[1]) {
        tds[1].textContent = hv30 != null ? hv30.toFixed(1) + '%' : '—';
        tds[1].style.color = hv30 != null ? (hv30 > 10 ? 'var(--down)' : hv30 < 4 ? 'var(--up)' : 'var(--text)') : 'var(--text3)';
        tds[1].style.fontFamily = 'var(--font-mono)'; tds[1].style.fontSize = '10px';
      }
      // HV 10d
      if (tds[2]) {
        tds[2].textContent = hv10 != null ? hv10.toFixed(1) + '%' : '—';
        tds[2].style.color = 'var(--text2)';
        tds[2].style.fontFamily = 'var(--font-mono)'; tds[2].style.fontSize = '10px';
      }
      // Vol Trend
      if (tds[3]) {
        if (hv10 != null && hv30 != null) {
          const diff = hv10 - hv30;
          const arrow = diff > 1 ? '↑' : diff < -1 ? '↓' : '→';
          const color = diff > 1 ? 'var(--down)' : diff < -1 ? 'var(--up)' : 'var(--text3)';
          tds[3].textContent = arrow; tds[3].style.color = color;
          tds[3].title = `HV10 ${hv10.toFixed(1)}% vs HV30 ${hv30.toFixed(1)}%`;
        } else {
          tds[3].textContent = '—'; tds[3].style.color = 'var(--text3)';
        }
        tds[3].style.fontSize = '11px';
      }
      // 1D Δ%
      if (tds[4]) {
        tds[4].textContent = pct != null ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : '—';
        tds[4].style.color = pct != null ? (pct > 0 ? 'var(--up)' : pct < 0 ? 'var(--down)' : 'var(--text3)') : 'var(--text3)';
        tds[4].style.fontFamily = 'var(--font-mono)'; tds[4].style.fontSize = '10px';
      }
    });
  }

  // ── ECB Reference Exchange Rates ──
  // Source: fx-data/frankfurter.json (server-side cached from api.frankfurter.app)
  // Shows today's ECB fixing vs previous day, plus offset from current spot
  const ecbTbody = document.getElementById('ecb-fixings-tbody');
  if (ecbTbody) {
    try {
      const fxRes = await fetch('./fx-data/frankfurter.json').catch(() => null);
      if (fxRes?.ok) {
        const fxJson = await fxRes.json();
        // Use EUR-base section for ECB panel (today_eur/prev_eur keys: USD, GBP, JPY, AUD, CAD, CHF, NZD)
        // Fall back to today/prev (USD-base) for older cached files — USD won't appear in that case
        const todayRates = fxJson?.today_eur?.rates ?? fxJson?.today?.rates ?? {};
        const prevRates  = fxJson?.prev_eur?.rates  ?? fxJson?.prev?.rates  ?? {};
        const fxDate     = fxJson?.today?.date  ?? '';

        // Pairs to display — all EUR-quoted
        const ecbPairs = [
          { label: 'EUR/USD', ccy: 'USD' },
          { label: 'EUR/GBP', ccy: 'GBP' },
          { label: 'EUR/JPY', ccy: 'JPY' },
          { label: 'EUR/CHF', ccy: 'CHF' },
          { label: 'EUR/AUD', ccy: 'AUD' },
          { label: 'EUR/CAD', ccy: 'CAD' },
          { label: 'EUR/NZD', ccy: 'NZD' },
        ];

        const rows = ecbTbody.querySelectorAll('tr');
        const MN = { USD: 4, GBP: 4, JPY: 2, CHF: 4, AUD: 4, CAD: 4, NZD: 4 };

        ecbPairs.forEach(({ label, ccy }, i) => {
          const row = rows[i];
          if (!row) return;
          const tds = row.querySelectorAll('td');
          const dec = MN[ccy] ?? 4;
          const today = todayRates[ccy];
          const prev  = prevRates[ccy];
          const chg   = (today != null && prev != null) ? today - prev : null;
          const chgPct = (chg != null && prev != null && prev !== 0) ? (chg / prev) * 100 : null;

          // Spot for vs-fix comparison: try to get EUR/XXX spot from intraday/stooq cache
          const pairId = ('eur' + ccy).toLowerCase();
          const spot = STOOQ_RT_CACHE?.[pairId]?.close ?? intraday?.quotes?.[pairId]?.close ?? null;
          const vsSpot = (spot != null && today != null) ? spot - today : null;

          const monoStyle = 'font-family:var(--font-mono);font-size:10px;text-align:right;';

          if (tds[0]) tds[0].textContent = label;
          if (tds[1]) { tds[1].textContent = today != null ? today.toFixed(dec) : '—'; tds[1].setAttribute('style', monoStyle); }
          if (tds[2]) { tds[2].textContent = prev  != null ? prev.toFixed(dec)  : '—'; tds[2].setAttribute('style', monoStyle + 'color:var(--text2);'); }
          if (tds[3]) {
            tds[3].textContent = chg != null ? (chg >= 0 ? '+' : '') + chg.toFixed(dec) : '—';
            tds[3].setAttribute('style', monoStyle + `color:${chg == null ? 'var(--text3)' : chg > 0 ? 'var(--up)' : chg < 0 ? 'var(--down)' : 'var(--text3)'};`);
          }
          if (tds[4]) {
            tds[4].textContent = chgPct != null ? (chgPct >= 0 ? '+' : '') + chgPct.toFixed(3) + '%' : '—';
            tds[4].setAttribute('style', monoStyle + `color:${chgPct == null ? 'var(--text3)' : chgPct > 0 ? 'var(--up)' : chgPct < 0 ? 'var(--down)' : 'var(--text3)'};`);
          }
          if (tds[5]) {
            tds[5].textContent = vsSpot != null ? (vsSpot >= 0 ? '+' : '') + vsSpot.toFixed(dec) : '—';
            tds[5].title       = vsSpot != null ? `Spot (${spot.toFixed(dec)}) minus ECB fix (${today.toFixed(dec)})` : 'Spot not available';
            tds[5].setAttribute('style', monoStyle + `color:${vsSpot == null ? 'var(--text3)' : Math.abs(vsSpot) < 0.001 ? 'var(--text3)' : 'var(--text2)'};`);
          }
        });

        const footer = document.getElementById('ecb-fixings-footer');
        if (footer && fxDate) footer.textContent = `ECB · official reference fixing · ${fxDate} · published ~16:00 CET · source: frankfurter.json`;
      }
    } catch { /* graceful — table shows dashes */ }
  }

  // ── DTCC GTR FX OTC Notional Volume ──
  // Source: dtcc-data/dtcc_fx.json (fetched daily by update-dtcc-fx.yml — public repo)
  // CFTC Recast public dissemination under Dodd-Frank 2(a)(13); no API key required
  const dtccTbody = document.getElementById('dtcc-tbody');
  if (dtccTbody) {
    try {
      const dtccRes = await fetch('./dtcc-data/dtcc_fx.json').catch(() => null);
      if (dtccRes?.ok) {
        const dtcc = await dtccRes.json();
        const pairs = dtcc?.pairs ?? {};
        const totals = dtcc?.totals ?? {};
        const totalNotional = totals?.notional_usd_bn ?? 0;

        const pairKeys = Object.keys(pairs);
        if (dtcc.status === 'pending' || pairKeys.length === 0) {
          // First run — data not yet fetched
          dtccTbody.innerHTML = '<tr><td colspan="7" style="color:var(--text3);text-align:center;padding:12px 0;font-size:10px;">Data pending — workflow runs Mon-Fri 14:00 UTC · DTCC GTR T+1</td></tr>';
        } else {
          // Build rows — sorted by notional (already sorted in JSON)
          const maxNotional = pairs[pairKeys[0]]?.notional_usd_bn ?? 1; // largest pair for heat bar scale

          const rows = pairKeys.map(pair => {
            const d = pairs[pair];
            const byProduct = d.by_product ?? {};
            const swapBn  = byProduct['FxSwap']?.notional_usd_bn    ?? 0;
            const fwdBn   = (byProduct['FxForward']?.notional_usd_bn ?? 0)
                          + (byProduct['FxNDF']?.notional_usd_bn     ?? 0);  // NDFs are forward-type
            const spotBn  = byProduct['FxSpot']?.notional_usd_bn    ?? 0;
            const sharePct = totalNotional > 0 ? (d.notional_usd_bn / totalNotional) * 100 : 0;
            // Heat bar: width proportional to this pair vs the largest pair (not total)
            const barPct = maxNotional > 0 ? Math.min((d.notional_usd_bn / maxNotional) * 100, 100) : 0;

            const mono = 'font-family:var(--font-mono);font-size:10px;text-align:right;';
            // Share cell: number + heat bar background
            const shareCell = `<td style="${mono}color:var(--text3);position:relative;padding-right:6px;">
              <div style="position:absolute;left:0;top:0;bottom:0;width:${barPct.toFixed(1)}%;background:var(--blue);opacity:0.18;border-radius:0 2px 2px 0;"></div>
              <span style="position:relative;">${sharePct.toFixed(1)}%</span>
            </td>`;
            return `<tr>
              <td style="font-size:10px;">${pair}</td>
              <td style="${mono}color:var(--text);">${d.notional_usd_bn.toFixed(1)}</td>
              <td style="${mono}color:var(--text2);">${d.trade_count.toLocaleString()}</td>
              <td style="${mono}color:var(--text2);">${swapBn > 0 ? swapBn.toFixed(1) : '—'}</td>
              <td style="${mono}color:var(--text2);">${fwdBn  > 0 ? fwdBn.toFixed(1)  : '—'}</td>
              <td style="${mono}color:var(--text2);">${spotBn > 0 ? spotBn.toFixed(1) : '—'}</td>
              ${shareCell}
            </tr>`;
          }).join('');

          // Totals row
          const byProd = totals.by_product ?? {};
          const totalSwap = byProd['FxSwap']?.notional_usd_bn ?? 0;
          const totalFwd  = (byProd['FxForward']?.notional_usd_bn ?? 0)
                          + (byProd['FxNDF']?.notional_usd_bn     ?? 0);
          const totalSpot = byProd['FxSpot']?.notional_usd_bn ?? 0;
          const mono = 'font-family:var(--font-mono);font-size:10px;text-align:right;';
          const totRow = `<tr style="border-top:1px solid var(--border2);font-weight:600;">
            <td style="font-size:10px;color:var(--text2);">TOTAL (G8)</td>
            <td style="${mono}color:var(--text);">${totalNotional.toFixed(1)}</td>
            <td style="${mono}color:var(--text2);">${totals.trade_count.toLocaleString()}</td>
            <td style="${mono}color:var(--text2);">${totalSwap > 0 ? totalSwap.toFixed(1) : '—'}</td>
            <td style="${mono}color:var(--text2);">${totalFwd  > 0 ? totalFwd.toFixed(1)  : '—'}</td>
            <td style="${mono}color:var(--text2);">${totalSpot > 0 ? totalSpot.toFixed(1) : '—'}</td>
            <td style="${mono}color:var(--text3);">100%</td>
          </tr>`;

          dtccTbody.innerHTML = rows + totRow;
        }

        const footer = document.getElementById('dtcc-footer');
        if (footer && dtcc.trade_date) {
          footer.textContent = `DTCC GTR · CFTC Recast · trade date ${dtcc.trade_date} · fetched ${dtcc.fetched} · Notional capped at $250M/trade · subset of total OTC FX volume`;
        }
      } else {
        dtccTbody.innerHTML = '<tr><td colspan="7" style="color:var(--text3);text-align:center;padding:12px 0;font-size:10px;">DTCC data unavailable</td></tr>';
      }
    } catch { dtccTbody.innerHTML = '<tr><td colspan="7" style="color:var(--text3);text-align:center;padding:12px 0;font-size:10px;">DTCC data error</td></tr>'; }
  }
}


// ── G8 Rates Tabs ──
function initG8RatesTabs() {
  const tabBar = document.getElementById('rates-country-tabs');
  if (!tabBar) return;
  tabBar.addEventListener('click', e => {
    const btn = e.target.closest('.rates-ctab');
    if (!btn) return;
    const cty = btn.dataset.cty;

    // Update tab styles
    tabBar.querySelectorAll('.rates-ctab').forEach(b => {
      const isActive = b === btn;
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      b.style.background = isActive ? 'var(--blue)' : 'none';
      b.style.color = isActive ? '#fff' : (b.dataset.cty === 'spreads' ? 'var(--blue)' : 'var(--text2)');
      b.style.border = isActive ? 'none' : '1px solid var(--border2)';
    });

    // Show/hide panes
    document.querySelectorAll('.rates-country-pane').forEach(p => { p.style.display = 'none'; });
    const pane = document.getElementById('rates-pane-' + cty);
    if (pane) pane.style.display = '';

    // Lazy-load G8 data on first open
    if (cty !== 'us' && cty !== 'spreads') renderG8YieldPane(cty);
    if (cty === 'spreads') renderSovereignSpreads();
  });
}

// Map country code to extended-data file key and yield tickers
const G8_YIELD_MAP = {
  de: { file: 'EUR', label: 'Germany', subtitle: 'GERMANY · SOVEREIGN BOND YIELDS', tenors: [{ k: 'bond2y', label: '2Y Bund' }, { k: 'bond10y', label: '10Y Bund' }] },
  gb: { file: 'GBP', label: 'UK',      subtitle: 'UK · SOVEREIGN BOND YIELDS',      tenors: [{ k: 'bond2y', label: '2Y Gilt' }, { k: 'bond10y', label: '10Y Gilt' }] },
  jp: { file: 'JPY', label: 'Japan',   subtitle: 'JAPAN · SOVEREIGN BOND YIELDS',   tenors: [{ k: 'bond10y', label: '10Y JGB' }] },
  au: { file: 'AUD', label: 'Australia', subtitle: 'AUSTRALIA · SOVEREIGN BOND YIELDS', tenors: [{ k: 'bond10y', label: '10Y ACGB' }] },
  ca: { file: 'CAD', label: 'Canada',  subtitle: 'CANADA · SOVEREIGN BOND YIELDS',  tenors: [{ k: 'bond2y', label: '2Y CGB' }, { k: 'bond10y', label: '10Y CGB' }] },
  nz: { file: 'NZD', label: 'New Zealand', subtitle: 'NEW ZEALAND · SOVEREIGN BOND YIELDS', tenors: [{ k: 'bond10y', label: '10Y NZGB' }] },
};

async function renderG8YieldPane(cty) {
  const pane = document.getElementById('rates-pane-' + cty);
  const contentEl = document.getElementById('rates-g8-content-' + cty);
  if (!pane || !contentEl) return;
  if (contentEl.dataset.loaded) return; // already populated

  const cfg = G8_YIELD_MAP[cty];
  if (!cfg) return;

  try {
    const ext = await fetch('./extended-data/' + cfg.file + '.json').then(r => r.ok ? r.json() : null).catch(() => null);
    if (!ext) { contentEl.textContent = 'Data unavailable — extended-data/' + cfg.file + '.json'; return; }

    const d = ext.data ?? ext;
    // Subtitle row
    let html = `<div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">${cfg.subtitle}</div>`;
    // Tile grid — columns match tenor count so single-tenor countries (JP, AU, NZ) don't leave a grey gap
    const cols = cfg.tenors.length === 1 ? '1fr' : '1fr 1fr';
    html += `<div class="rates-grid" style="margin-bottom:6px;grid-template-columns:${cols};">`;
    cfg.tenors.forEach(t => {
      const val = d[t.k];
      // Values in extended-data are stored as percentages (e.g. 3.04 = 3.04%)
      // US tiles use same scale. No conversion needed.
      const valStr = val != null ? val.toFixed(2) + '%' : '—';
      // Change indicator: extended-data has no intraday delta — show "—" in flat style
      // consistent with how US tiles show "—" when fromRepo=true
      html += `<div class="rate-cell">` +
        `<div class="rate-cty">${t.label}</div>` +
        `<div class="rate-val">${valStr}</div>` +
        `<div class="rate-chg flat">—</div>` +
        `</div>`;
    });
    html += '</div>';
    // Source attribution
    const dateLbl = ext?.dates?.bond10y ? ext.dates.bond10y : '';
    html += `<div style="font-size:9px;color:var(--text3);">Daily sovereign yield pipeline${dateLbl ? ' · ' + dateLbl : ''}</div>`;
    contentEl.innerHTML = html;
    contentEl.dataset.loaded = '1';
  } catch {
    contentEl.textContent = 'Failed to load yield data.';
  }
}

// ── Sovereign Spreads vs US ──
async function renderSovereignSpreads() {
  const tbody = document.getElementById('sovereign-spreads-tbody');
  if (!tbody) return;
  if (tbody.dataset.loaded === '2') return; // already rendered with flag spans

  const countries = [
    { code: 'de', file: 'EUR', label: 'DE' },
    { code: 'gb', file: 'GBP', label: 'GB' },
    { code: 'jp', file: 'JPY', label: 'JP' },
    { code: 'au', file: 'AUD', label: 'AU' },
    { code: 'ca', file: 'CAD', label: 'CA' },
    { code: 'nz', file: 'NZD', label: 'NZ' },
  ];

  // Load US first
  const usExt = await fetch('./extended-data/USD.json').then(r => r.ok ? r.json() : null).catch(() => null);
  const _usData = usExt?.data ?? usExt;
  const us10y = _usData?.bond10y ?? null;
  const us2y  = _usData?.bond2y  ?? null;

  const rows = tbody.querySelectorAll('tr');
  await Promise.all(countries.map(async (c, idx) => {
    const row = rows[idx];
    if (!row) return;
    const tds = row.querySelectorAll('td');

    try {
      const ext = await fetch('./extended-data/' + c.file + '.json').then(r => r.ok ? r.json() : null).catch(() => null);
      const _extData = ext?.data ?? ext;
      const cty10y = _extData?.bond10y ?? null;
      const cty2y  = _extData?.bond2y  ?? null;

      // Normalise: extended-data stores some yields as 0.04 (4%) and some as 4.0
      const norm = v => v != null ? (v < 1 ? v * 100 : v) : null;
      const n10 = norm(cty10y);
      const n2  = norm(cty2y);
      const us10 = norm(us10y);
      const us2  = norm(us2y);

      // Country flag + label
      if (tds[0]) { tds[0].innerHTML = `<span class="fi fi-${c.code}" style="margin-right:4px;border-radius:1px;vertical-align:middle;"></span><span>${c.label}</span>`; }
      // 10Y value
      if (tds[1]) { tds[1].textContent = n10 != null ? n10.toFixed(2) + '%' : '—'; }

      // Spread vs US
      if (tds[2]) {
        const spread = (n10 != null && us10 != null) ? (n10 - us10) * 100 : null; // in bp
        if (spread != null) {
          tds[2].textContent = (spread >= 0 ? '+' : '') + Math.round(spread) + ' bp';
          tds[2].style.color = spread > 20 ? 'var(--up)' : spread < -20 ? 'var(--down)' : 'var(--text2)';
        } else { tds[2].textContent = '—'; }
      }

      // 2Y
      if (tds[3]) { tds[3].textContent = n2 != null ? n2.toFixed(2) + '%' : '—'; }

      // 2Y-10Y curve slope
      if (tds[4]) {
        const slope = (n2 != null && n10 != null) ? n10 - n2 : null;
        if (slope != null) {
          tds[4].textContent = (slope >= 0 ? '+' : '') + slope.toFixed(0) + ' bp';
          tds[4].style.color = slope < 0 ? 'var(--down)' : slope > 50 ? 'var(--up)' : 'var(--text2)';
          tds[4].title = slope < 0 ? 'Inverted curve' : slope < 25 ? 'Flat curve' : 'Normal curve';
        } else { tds[4].textContent = '—'; }
      }
    } catch {
      tds.forEach((td, i) => { if (i > 0) td.textContent = '—'; });
    }
  }));
  tbody.dataset.loaded = '2';
}

// ── Economic Surprises — CESI-style centred bar index (v7.76.0) ──────────────
// Methodology: for each G8 currency, computes a normalised surprise index over
// a 90-day rolling window from Finnhub economic calendar (actual vs consensus).
// Index = (beats − misses) / total scored, scaled to [−100, +100].
// Bar chart centred at 0: green bar extends right for positive, red bar extends
// left for negative — matching Citi CESI / Bloomberg BEEI visual convention.
// N column shows count of events with actuals (sample size transparency).
async function renderEconSurprises() {
  const tbody = document.getElementById('econ-surprise-tbody');
  if (!tbody) return;

  const nowMs = Date.now();
  const LOOKBACK_MS = 90 * 24 * 60 * 60 * 1000;
  window._ES_SEEN = new Set(); // reset dedup guard on each render

  // ── Load calendar.json (Finnhub via ff_calendar.json) ─────────────────
  let calEvents = [];
  let calSource = '';
  try {
    const res = await fetch('./calendar-data/calendar.json').catch(() => null);
    if (res?.ok) {
      const calj = await res.json();
      const evts = (calj?.events || []).map(ev => ({
        title:    ev.event || ev.title || '',
        currency: ev.currency || '',
        dateISO:  ev.dateISO || '',
        impact:   ev.impact || 'low',
        forecast: ev.forecast || null,
        previous: ev.previous || null,
        actual:   ev.actual || null,
        released: !!(ev.actual && ev.actual !== '' && ev.actual !== '-'),
      }));
      const hasReleased = evts.some(ev => {
        const t = new Date(ev.dateISO).getTime();
        return !isNaN(t) && nowMs - t <= LOOKBACK_MS && ev.released;
      });
      if (hasReleased) { calEvents = evts; calSource = calj.source || 'investing.com'; }
      // Store surprise stats for z-score scoring (populated by engine v3.1+)
      window._ECON_SURPRISE_STATS = calj.surpriseStats || {};
    }
  } catch { /* graceful */ }

  // ── Fallback: ff_calendar.json ────────────────────────────────────────────
  if (!calEvents.length) {
    try {
      const res2 = await fetch('./calendar-data/ff_calendar.json').catch(() => null);
      if (res2?.ok) {
        const ffj = await res2.json();
        const win21 = 21 * 24 * 60 * 60 * 1000;
        const evts = (ffj?.events || []).map(ev => ({
          title: ev.event || ev.title || '', currency: ev.currency || '',
          dateISO: ev.dateISO || '', impact: ev.impact || 'low',
          forecast: ev.forecast || null, previous: ev.previous || null,
          actual: ev.actual || null,
          released: !!(ev.actual && ev.actual !== '' && ev.actual !== '-'),
        }));
        const hasReleased = evts.some(ev => {
          const t = new Date(ev.dateISO).getTime();
          return !isNaN(t) && nowMs - t <= win21 && ev.released && ev.actual != null;
        });
        if (hasReleased) { calEvents = evts; calSource = 'ForexFactory'; }
      }
    } catch { /* no fallback */ }
  }

  // ── Source label ──────────────────────────────────────────────────────────
  const srcEl = document.getElementById('econ-surprise-source');
  if (srcEl) {
    if (calSource === 'Finnhub' || calSource.startsWith('Finnhub')) {
      srcEl.textContent = 'Finnhub · actual vs consensus · G8 · 90d rolling';
    } else if (calSource.startsWith('investing.com') || calSource.startsWith('TradingEconomics')) {
      srcEl.textContent = 'investing.com · actual vs consensus · 90d rolling';
    } else if (calSource === 'ForexFactory') {
      // Fallback path: Finnhub key not set, FF scraper used instead
      srcEl.textContent = 'ForexFactory · actual vs consensus · 90d rolling';
    } else if (calSource && calSource.includes('ForexFactory')) {
      // Legacy backfill sources: multi-source historical string
      srcEl.textContent = 'FRED + Finnhub + ForexFactory · actual vs consensus · G8 · 90d rolling';
    } else if (calSource) {
      srcEl.textContent = calSource + ' · actual vs consensus · 90d rolling';
    } else {
      srcEl.textContent = 'Calendar data unavailable';
    }
  }

  // ── Score per currency ────────────────────────────────────────────────────
  // Inverse indicators: a lower actual is a positive surprise (e.g. unemployment fell)
  const INVERSE_KW = ['unemployment', 'jobless', 'claims', 'deficit', 'trade balance'];

  // ── Exponential time-decay (CESI convention) ────────────────────────────────────────
  // CESI applies decay so recent surprises dominate and old data fades.
  // Half-life = 45 days: w(0d)=1.00, w(45d)=0.50, w(90d)=0.25.
  // λ = ln(2) / 45 ≈ 0.01540. N column still shows raw event count for transparency.
  const DECAY_LAMBDA = Math.LN2 / 45;

  const ccyScores = {};

  calEvents.forEach(ev => {
    const evTime = new Date(ev.dateISO).getTime();
    if (isNaN(evTime) || evTime > nowMs || nowMs - evTime > LOOKBACK_MS) return;
    if (!ev.released || ev.actual == null) return;
    if (!['medium','high'].includes(ev.impact)) return;
    const ccy = ev.currency;
    if (!['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'].includes(ccy)) return;

    // ── Noise filter: exclude non-macro events ─────────────────────────────
    // CESI-style indices (Citi, DB, MS) only score fundamental macro releases.
    // Bond auctions, CFTC positioning, commodity inventory/rig data, derived
    // averages, financial flow data, and SEP dot projections are excluded.
    const evTitle = (ev.event || ev.title || '').toLowerCase();
    const NOISE_KW = [
      'cftc','baker hughes','rig count','auction','api weekly',
      'milk auction','fed\'s balance sheet','reserve balances',
      'redbook','ibd/tipp','tips auction','note auction','bond auction',
      'gilt auction','jgb auction','obligaciones','speculative net',
      'nc net position','crude oil inventories','crude oil imports',
      'distillate','gasoline inventorie','gasoline production',
      'refinery','heating oil','natural gas storage',
      'foreign bonds buying','foreign investments in japanese',
      'foreign bond investment','foreign investment in japan',
      'm2 money','m3 money','m4 money','reserve assets total',
      'cb leading index','atlanta fed gdpnow','ny fed','cleveland cpi',
      'ibd','3-month bill','4-week bill','52-week bill',
      // Additional noise: derived averages, financial flows, SEP projections, EIA energy
      '4-week average','4-week avg',
      'tic net','net long-term tic','total net tic',
      'interest rate projection',
      'eia crude oil','eia crude',
    ];
    if (NOISE_KW.some(kw => evTitle.includes(kw))) return;
    // ── Dedup guard: same canonical event + same actual → score only once ──
    // ForexFactory publishes Flash then Final PMIs with identical data on
    // different dates. Without dedup, each revision counts as a separate event,
    // inflating N and double-counting the same macro signal.
    const canonEvent = evTitle.replace(/\s*\([^)]*\)/g, '').trim();
    // Use forecast||previous in the dedup key — mirrors fetch_economic_calendar.py
    // so events without an explicit forecast but with a previous baseline deduplicate
    // consistently between JS scoring and Python surpriseStats computation.
    const dedupKey   = `${ccy}/${canonEvent}/${String(ev.actual).replace(/[%,\s]/g,'')}/${String(ev.forecast||ev.previous||'').replace(/[%,\s]/g,'')}`;
    if (!window._ES_SEEN) window._ES_SEEN = new Set();
    if (window._ES_SEEN.has(dedupKey)) return;
    window._ES_SEEN.add(dedupKey);
    // ───────────────────────────────────────────────────────────────────────

    const actualStr   = String(ev.actual   || '').replace(/[%,]/g,'');
    const forecastStr = String(ev.forecast || ev.previous || '').replace(/[%,]/g,'');
    const actual   = parseFloat(actualStr);
    const forecast = parseFloat(forecastStr);
    if (isNaN(actual) || isNaN(forecast)) return;

    const isInverse = INVERSE_KW.some(kw => evTitle.includes(kw));
    const beat = isInverse ? actual < forecast : actual > forecast;
    const miss = isInverse ? actual > forecast : actual < forecast;
    // rawSurprise is unsigned (actual − forecast). For the z-score we apply the
    // same sign correction that fetch_economic_calendar.py applies when building
    // surpriseStats: negate for inverse indicators so positive z-score always means
    // a positive surprise. beat/miss already encodes direction correctly above.
    const rawSurprise = actual - forecast;
    const surprise    = isInverse ? -rawSurprise : rawSurprise;

    // ── Exponential decay weight ─────────────────────────────────────────────────────────────
    // w = e^(-λ · ageDays). Recent events weight 1.0; older events fade smoothly.
    const ageDays = (nowMs - evTime) / 86400000;
    const w = Math.exp(-DECAY_LAMBDA * ageDays);

    // ── Z-score scoring (hybrid: z-score when stats available, beat/miss otherwise) ──
    // As history accumulates in surpriseStats (engine v3.1+), more events
    // graduate to z-score. MIN 5 observations required for a valid std estimate.
    const CANONICAL_MIN_N = 5;
    const statsKey = (() => {
      const canon = (evTitle.replace(/\s*\([^)]*\)/g, '').trim());
      return `${ccy}/${canon}`;
    })();
    const stats = (window._ECON_SURPRISE_STATS || {})[statsKey];
    const useZScore = stats && stats.n >= CANONICAL_MIN_N && stats.std > 0;
    const zScore = useZScore ? (surprise - stats.mean) / stats.std : null;

    if (!ccyScores[ccy]) ccyScores[ccy] = {
      // Raw counts — for N display and low-confidence threshold
      total: 0, beats: 0, misses: 0,
      // Decay-weighted accumulators — used for index calculation
      wTotal: 0, wBeats: 0, wMisses: 0,
      zWSum: 0, zWTotal: 0, zWBeats: 0, zWMisses: 0,
    };
    ccyScores[ccy].total++;
    ccyScores[ccy].wTotal += w;
    if (beat) { ccyScores[ccy].beats++;  ccyScores[ccy].wBeats  += w; }
    if (miss) { ccyScores[ccy].misses++; ccyScores[ccy].wMisses += w; }
    // Decay-weighted z-score accumulators for the blend formula.
    if (zScore !== null) {
      ccyScores[ccy].zWSum   += zScore * w;
      ccyScores[ccy].zWTotal += w;
      if (beat) ccyScores[ccy].zWBeats += w;
      if (miss) ccyScores[ccy].zWMisses += w;
    }
  });

  // ── Normalise to [−100, +100] index (Citi CESI convention) ───────────────
  // index = (beats − misses) / total × 100
  // Bar fill: 50% of bar width per side (each side = 50% of container)
  const G8 = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
  const rows = tbody.querySelectorAll('tr');

  G8.forEach((ccy, idx) => {
    const row = rows[idx];
    if (!row) return;
    const tds = row.querySelectorAll('td');
    const barFill = row.querySelector('.es-bar-fill');
    const s = ccyScores[ccy];

    if (!s || s.total === 0) {
      // No data — neutral empty bar
      if (barFill) { barFill.style.width = '0%'; barFill.style.left = '50%'; barFill.style.background = 'var(--border2)'; }
      if (tds[2]) { tds[2].textContent = '—'; tds[2].style.color = 'var(--text3)'; }
      row.title = `${ccy}: no released events with actuals in 90d window`;
      return;
    }

    // ── Index: decay-weighted z-score blend when available, beat/miss otherwise ───────
    // All contributions scaled by w = e^(-λ·ageDays) — recent surprises dominate.
    // Events with ≥5 historical observations use z-score (normalised surprise magnitude).
    // Remaining events use beat/miss. Both halves are decay-weighted consistently.
    let idx100;
    const zFraction = s.zWTotal / s.wTotal;
    if (s.zWTotal >= 10 || (s.zWTotal > 0 && zFraction >= 0.30)) {
      // Blend: weighted z-score contrib = (zWSum/zWTotal)*50 (maps ±2σ to ±100),
      // weighted non-z contrib = weighted beat/miss ratio * 100.
      const nonZWTotal = s.wTotal  - s.zWTotal;
      const nonZWBeat  = s.wBeats  - s.zWBeats;
      const nonZWMiss  = s.wMisses - s.zWMisses;
      const zPart  = s.zWTotal   > 0 ? (s.zWSum / s.zWTotal) * 50 : 0;
      const bmPart = nonZWTotal  > 0 ? ((nonZWBeat - nonZWMiss) / nonZWTotal) * 100 : 0;
      idx100 = (zPart * s.zWTotal + bmPart * nonZWTotal) / s.wTotal;
    } else {
      // Pure decay-weighted beat/miss (CESI convention)
      idx100 = s.wTotal > 0 ? ((s.wBeats - s.wMisses) / s.wTotal) * 100 : 0;
    }
    // Bar: max half-width = 50% of container (the zero line is at 50%)
    const halfPct = Math.min(Math.abs(idx100), 100) / 2; // 0–50%
    const positive = idx100 >= 0;
    const color = positive ? 'var(--up)' : 'var(--down)';

    const lowConf = s.total < 15;

    if (barFill) {
      barFill.style.width      = halfPct.toFixed(1) + '%';
      barFill.style.left       = positive ? '50%' : (50 - halfPct).toFixed(1) + '%';
      barFill.style.background = color;
      barFill.style.opacity    = '1';
    }

    // N column — dim number for low-N currencies; the visible N is the signal
    if (tds[2]) {
      tds[2].textContent = s.total;
      tds[2].style.color = lowConf ? 'var(--text4, rgba(255,255,255,0.3))' : 'var(--text3)';
      tds[2].title = lowConf ? 'Low sample size — interpret with caution' : '';
    }

    // Row tooltip
    const pct = (s.beats / s.total * 100).toFixed(0);
    const inLine = s.total - s.beats - s.misses;
    row.title = `${ccy}: ${s.beats} beat · ${s.misses} miss · ${inLine} in-line · ${pct}% beat rate · index ${idx100 >= 0 ? '+' : ''}${idx100.toFixed(0)} · decay-weighted (45d half-life) · click for detail`;
  });

  // ── Keyboard activation for clickable rows (Enter / Space) ──────────────
  // onclick is already in the static HTML; this adds keyboard parity.
  if (tbody && !tbody._esmKeyBound) {
    tbody._esmKeyBound = true;
    tbody.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        const row = ev.target.closest('tr');
        const ccy = row?.querySelector('td')?.textContent?.trim();
        if (ccy && window.openEconSurprisesModal) {
          ev.preventDefault();
          window.openEconSurprisesModal(ccy);
        }
      }
    });
  }
}

// ── Derivatives section visibility toggle ──
function initDerivativesNav() {
  const allNavLinks = document.querySelectorAll('.top-nav a[data-target]');
  allNavLinks.forEach(link => {
    link.addEventListener('click', () => {
      const target = link.dataset.target;
      const derivSection = document.getElementById('section-derivatives');
      if (!derivSection) return;
      // Show Derivatives section only when that tab is active; hide otherwise
      if (target === 'section-derivatives') {
        derivSection.style.display = '';
        renderDerivativesSection();
      } else {
        derivSection.style.display = 'none';
      }
    });
  });
}

// ── Bootstrap all new features ──
// ── Load CB rates from rates/*.json directly (reliable, not DOM-dependent) ──
async function loadCBRatesCache() {
  // Loads CB policy rates from rates/*.json — used for CB Rates panel,
  // carry ranking, regime scoring. NOT used for CIP forward pricing.
  // rates/*.json files: observations array, most recent first.
  // Schema: { observations: [{ date: "YYYY-MM-DD", value: "3.75" }, ...], ... }
  const ccyFiles = {
    USD: 'rates/USD.json', EUR: 'rates/EUR.json', GBP: 'rates/GBP.json',
    JPY: 'rates/JPY.json', AUD: 'rates/AUD.json', CAD: 'rates/CAD.json',
    CHF: 'rates/CHF.json', NZD: 'rates/NZD.json',
  };
  await Promise.all(Object.entries(ccyFiles).map(async ([ccy, path]) => {
    try {
      const r = await fetch('./' + path);
      if (!r.ok) return;
      const d = await r.json();
      // Use most recent observation (observations[0].value is a string like "3.75")
      const obs = d.observations;
      const raw = Array.isArray(obs) && obs.length > 0
        ? obs[0].value           // observations array format
        : (d.rate ?? d.value ?? null); // fallback for other shapes
      if (raw != null && !isNaN(+raw)) window._CB_RATES_CACHE[ccy] = +raw;
    } catch { /* graceful — leave missing */ }
  }));
}

async function loadOISRatesCache() {
  // Loads OIS/overnight benchmark rates from ois-rates/rates.json.
  // Used exclusively by computeCIPForward() via _resolveRate().
  // Falls back silently — _resolveRate() uses policy rate when OIS unavailable.
  // Benchmarks: USD=SOFR, EUR=€STR, GBP=SONIA, JPY=TONA, AUD=AONIA, CAD=CORRA, CHF=SARON, NZD=OCR.
  try {
    const r = await fetch('./ois-rates/rates.json');
    if (!r.ok) return;
    const d = await r.json();
    const rates   = d.rates   || {};
    const sources = d.sources || {};
    for (const [ccy, val] of Object.entries(rates)) {
      if (val != null && !isNaN(+val)) {
        window._OIS_RATES_CACHE[ccy]  = +val;
        window._OIS_RATE_SOURCES[ccy] = sources[ccy] || 'OIS';
      }
    }
  } catch {
    // File not yet deployed or network failure — _resolveRate() falls back to policy.
  }
}

// ── Section visibility: Derivatives panel toggle ──

// ═══════════════════════════════════════════════════════════════════════════
// NEWS SECTION — dedicated full-width panel (shown when "News" nav tab clicked)
// Mirrors Derivatives show/hide pattern. Shortcut: N.
// ═══════════════════════════════════════════════════════════════════════════

// Module state
let _newsAllItems = [];
let _newsMeta     = {};
let _newsFilter   = { cur: 'ALL', impact: 'ALL' };

function renderNewsSection(items, meta) {
  if (Array.isArray(items)) _newsAllItems = items;
  if (meta) _newsMeta = meta;

  const feed = document.getElementById('news-section-feed');
  if (!feed) return;

  const filtered = _newsAllItems.filter(function(item) {
    const curOk    = _newsFilter.cur    === 'ALL' || item.cur    === _newsFilter.cur;
    const impactOk = _newsFilter.impact === 'ALL' || item.impact === _newsFilter.impact;
    return curOk && impactOk;
  });

  const tsEl = document.getElementById('news-section-ts');
  if (tsEl && _newsMeta.updated_label) tsEl.textContent = _newsMeta.updated_label;

  const countEl = document.getElementById('news-section-count');
  if (countEl) countEl.textContent = filtered.length + ' stories';

  feed.innerHTML = '';
  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.style.cssText = 'padding:20px 14px;color:var(--text3);font-size:11px;';
    empty.textContent = 'No stories match current filter.';
    feed.appendChild(empty);
    return;
  }

  filtered.forEach(function(item) {
    let time = item.time || '--:--';
    let ageMs = 0;
    let pubDate = null;
    if (item.ts) {
      pubDate = new Date(item.ts);
      time = pubDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
      ageMs = Date.now() - item.ts;
    } else if (item.datetime) {
      const d = new Date(item.datetime);
      if (!isNaN(d)) {
        pubDate = d;
        time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
        ageMs = Date.now() - d.getTime();
      }
    }

    // Age label — Bloomberg compact: "<1h shows minutes, <24h shows hours, ≥24h shows date
    let ageLabel = '';
    let showDate = false;
    if (ageMs > 0) {
      const ageMin  = Math.floor(ageMs / 60000);
      const ageHr   = Math.floor(ageMs / 3600000);
      const ageDays = Math.floor(ageMs / 86400000);
      if (ageDays >= 1) {
        showDate = true;  // show date badge instead of relative age for old articles
      } else if (ageHr >= 1) {
        ageLabel = ageHr + 'h';
      } else if (ageMin >= 1) {
        ageLabel = ageMin + 'm';
      } else {
        ageLabel = 'now';
      }
    }

    const headline = item.title  || '';
    const snippet  = item.expand || '';
    const cur      = item.cur    || '';
    const source   = item.source || '';
    const impact   = item.impact || 'low';
    const rawLink  = item.link   || '';
    const safeLink = rawLink.startsWith('https://') ? rawLink : '';
    const hasSnip  = snippet.length > 0;

    // ── Outer wrapper ────────────────────────────────────────────────────────
    const wrap = document.createElement('div');
    wrap.className = 'ns-item' + (item.featured ? ' ns-featured' : '');

    // ── Single flex row: [time][dot][headline...][cur-tag][source][chevron] ──
    const row = document.createElement('div');
    row.className = 'ns-row';

    // ── Time cell: stacked HH:MM / Xh — Bloomberg Anywhere compact pattern ──
    // Single fixed-width container; no separate badge element in the flex row.
    const timeEl = document.createElement('span');
    timeEl.className = 'ns-time';

    const timeTop = document.createElement('span');
    timeTop.className = 'ns-time-hm';
    const timeBot = document.createElement('span');
    timeBot.className = 'ns-time-age';

    if (showDate && pubDate) {
      const dateStr = pubDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
      timeTop.textContent = dateStr;
      timeEl.title = time + ' · ' + pubDate.toLocaleDateString();
    } else {
      timeTop.textContent = time;
      if (ageLabel && ageLabel !== 'now') {
        timeBot.textContent = ageLabel;
      }
      if (ageLabel) timeEl.title = ageLabel + ' ago';
    }

    timeEl.appendChild(timeTop);
    if (timeBot.textContent) timeEl.appendChild(timeBot);

    const dot = document.createElement('span');
    dot.className = 'ns-dot ns-dot-' + impact;

    const headEl = document.createElement('span');
    headEl.className = 'ns-headline';
    headEl.textContent = headline;
    headEl.title = headline;  // native tooltip — full headline on hover (Bloomberg compact pattern)

    const chevron = document.createElement('span');
    chevron.className = 'ns-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    chevron.innerHTML = '<svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><polyline points="2,3.5 5,6.5 8,3.5"/></svg>';

    row.appendChild(timeEl);
    row.appendChild(dot);
    row.appendChild(headEl);

    if (cur) {
      const curTag = document.createElement('span');
      curTag.className = 'ns-cur-tag';
      curTag.textContent = cur;
      row.appendChild(curTag);
    }
    // Source moved to drawer — keeps headline full width (Bloomberg compact pattern)
    row.appendChild(chevron);
    wrap.appendChild(row);

    // ── Accordion drawer (hidden, expands below the row on click) ───────────
    if (hasSnip || safeLink || source) {
      const drawer = document.createElement('div');
      drawer.className = 'ns-drawer';

      // Source label at top of drawer — subtle, secondary color
      if (source) {
        const srcDrawer = document.createElement('p');
        srcDrawer.className = 'ns-drawer-source';
        srcDrawer.textContent = source;
        drawer.appendChild(srcDrawer);
      }

      if (hasSnip) {
        const snipEl = document.createElement('p');
        snipEl.className = 'ns-snippet';
        snipEl.textContent = snippet;   // full text — no truncation
        drawer.appendChild(snipEl);
      }

      if (safeLink) {
        const readLink = document.createElement('a');
        readLink.className = 'ns-read-link';
        readLink.textContent = 'Read full article';
        readLink.href = safeLink;
        readLink.target = '_blank';
        readLink.rel = 'noopener noreferrer';
        readLink.addEventListener('click', function(e) { e.stopPropagation(); });
        drawer.appendChild(readLink);
      }
      wrap.appendChild(drawer);

      // Click the row → open/close accordion (only one open at a time)
      row.addEventListener('click', function() {
        const isOpen = wrap.classList.contains('ns-open');
        feed.querySelectorAll('.ns-open').forEach(function(el) { el.classList.remove('ns-open'); });
        if (!isOpen) wrap.classList.add('ns-open');
      });
    }

    feed.appendChild(wrap);
  });
}
function _newsSetFilter(type, value) {
  _newsFilter[type] = value;
  // Update active pill styling
  const selector = type === 'cur' ? '.ns-cur-pill' : '.ns-imp-pill';
  document.querySelectorAll(selector).forEach(btn => {
    btn.classList.toggle('ns-pill-active', btn.dataset.val === value);
  });
  renderNewsSection();
}

function initNewsNav() {
  const newsSection = document.getElementById('section-news');
  if (!newsSection) return;

  const splitLowerRight = document.getElementById('split-lower-right');
  if (!splitLowerRight) return;

  function showNews() {
    // Flush any stale deriv-hidden state first so we snapshot clean display values.
    // If Derivatives was active when News was clicked, hideDerivatives() runs in bubble
    // phase (after this capture-phase handler) — but with the guard on the capture
    // listener this path is now unreachable. Belt-and-suspenders: clean up anyway.
    if (typeof window._derivNavHide === 'function') {
      const derivSection = document.getElementById('section-derivatives');
      if (derivSection && derivSection.style.display !== 'none') {
        window._derivNavHide();
      }
    }
    Array.from(splitLowerRight.children).forEach(el => {
      if (el.id !== 'section-news') {
        const originalDisplay = el.style.display || window.getComputedStyle(el).display;
        el.dataset.newsHidden = originalDisplay === 'none' ? 'none' : (el.style.display || '');
        el.style.display = 'none';
      }
    });
    newsSection.style.display = '';
    const splitLower = document.getElementById('split-lower');
    if (splitLower) splitLower.scrollTo({ top: 0, behavior: 'smooth' });
    // Re-render with current data so pills and count are up to date
    renderNewsSection();
  }

  function hideNews() {
    newsSection.style.display = 'none';
    splitLowerRight.querySelectorAll('[data-news-hidden]').forEach(el => {
      const saved = el.dataset.newsHidden;
      el.style.display = saved === '' ? '' : saved;
      delete el.dataset.newsHidden;
    });
    // Repaint canvases after display restore (same double-rAF pattern as Derivatives)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (typeof drawYieldCurve === 'function' && typeof _lastDrawnYields !== 'undefined') {
          drawYieldCurve(_lastDrawnYields, typeof _lastDrawnPrior !== 'undefined' ? _lastDrawnPrior : null);
        }
        const activeRatesTab = document.querySelector('.rates-ctab[aria-selected="true"]');
        if (activeRatesTab) {
          const cty = activeRatesTab.dataset.cty;
          if (cty && cty !== 'us') {
            if (cty === 'spreads' && typeof renderSovereignSpreads === 'function') {
              renderSovereignSpreads();
            } else if (typeof renderG8YieldPane === 'function') {
              const contentEl = document.getElementById('rates-g8-content-' + cty);
              if (contentEl) delete contentEl.dataset.loaded;
              renderG8YieldPane(cty);
            }
          }
        }
      });
    });
  }

  // Capture phase — only intercepts clicks ON the News link itself (not clicks away from News).
  // stopImmediatePropagation is scoped to the newsLink only; other nav links must propagate
  // normally so that hideNews() (bubble phase below) and hideDerivatives() run in the
  // correct order without corrupting each other's data-*-hidden state.
  const newsLink = document.querySelector('.top-nav a[data-target="section-news"]');
  if (newsLink) {
    newsLink.addEventListener('click', (e) => {
      e.stopImmediatePropagation();
      // Guard: if News is already visible, clicking the tab again is a no-op
      if (newsSection.style.display !== 'none') return;
      showNews();
    }, true);
  }

  // Hide when any other nav tab is clicked (bubble phase — runs after hideDerivatives etc.)
  document.querySelectorAll('.top-nav a[data-target]').forEach(link => {
    if (link.dataset.target !== 'section-news') {
      link.addEventListener('click', () => {
        if (newsSection.style.display !== 'none') hideNews();
      });
    }
  });

  // Expose for keyboard shortcut
  window._newsNavShow    = showNews;
  window._newsNavHide    = hideNews;
  window._newsNavSection = newsSection;

  // Expose filter setter for inline onclick
  window._newsSetFilter  = _newsSetFilter;
}

function initDerivativesNavFixed() {
  const derivSection = document.getElementById('section-derivatives');
  if (!derivSection) return;

  const splitLowerRight = document.getElementById('split-lower-right');
  if (!splitLowerRight) return;

  function showDerivatives() {
    Array.from(splitLowerRight.children).forEach(el => {
      if (el.id !== 'section-derivatives') {
        // Store original computed display so we can restore it exactly
        const originalDisplay = el.style.display || window.getComputedStyle(el).display;
        el.dataset.derivHidden = originalDisplay === 'none' ? 'none' : (el.style.display || '');
        el.style.display = 'none';
      }
    });
    derivSection.style.display = '';
    // Scroll split-lower to top so derivatives is visible
    const splitLower = document.getElementById('split-lower');
    if (splitLower) splitLower.scrollTo({ top: 0, behavior: 'smooth' });
    renderDerivativesSection();
  }

  function hideDerivatives() {
    derivSection.style.display = 'none';
    splitLowerRight.querySelectorAll('[data-deriv-hidden]').forEach(el => {
      // Restore the exact inline display value that was set before hiding
      const saved = el.dataset.derivHidden;
      el.style.display = saved === '' ? '' : saved;
      delete el.dataset.derivHidden;
    });
    // Canvas and lazy-loaded G8 panes need a repaint after display is restored.
    // Double rAF: split-lower-right uses display:contents which requires two frames
    // for the browser to commit the layout change and expose correct clientWidth values.
    // (Same pattern as the ticker strip — 'Double rAF ensures layout before we measure'.)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        // Yield curve canvas: redraws using cached data (_lastDrawnYields / _lastDrawnPrior)
        if (typeof drawYieldCurve === 'function' && typeof _lastDrawnYields !== 'undefined') {
          drawYieldCurve(_lastDrawnYields, typeof _lastDrawnPrior !== 'undefined' ? _lastDrawnPrior : null);
        }
        // Re-trigger whichever Rates tab is currently active so G8/Spreads panes re-render
        const activeRatesTab = document.querySelector('.rates-ctab[aria-selected="true"]');
        if (activeRatesTab) {
          const cty = activeRatesTab.dataset.cty;
          if (cty && cty !== 'us') {
            if (cty === 'spreads' && typeof renderSovereignSpreads === 'function') {
              renderSovereignSpreads();
            } else if (typeof renderG8YieldPane === 'function') {
              // Reset loaded flag so the pane re-renders with correct dimensions
              const contentEl = document.getElementById('rates-g8-content-' + cty);
              if (contentEl) delete contentEl.dataset.loaded;
              renderG8YieldPane(cty);
            }
          }
        }
      });
    });
  }

  // Use capture phase on Derivatives link to run before the main nav scroll handler
  const derivLink = document.querySelector('.top-nav a[data-target="section-derivatives"]');
  if (derivLink) {
    derivLink.addEventListener('click', (e) => {
      e.stopImmediatePropagation();
      showDerivatives();
    }, true);
  }

  // Restore on any other nav click
  document.querySelectorAll('.top-nav a[data-target]').forEach(link => {
    if (link.dataset.target !== 'section-derivatives') {
      link.addEventListener('click', () => {
        if (derivSection.style.display !== 'none') hideDerivatives();
      });
    }
  });

  // Expose for keyboard shortcut
  window._derivNavShow = showDerivatives;
  window._derivNavHide = hideDerivatives;
  window._derivNavSection = derivSection;
}

(function bootNewFeatures() {
  const run = async () => {
    initG8RatesTabs();
    initDerivativesNavFixed();
    initNewsNav();

    // Load CB policy rates, OIS benchmark rates, and intraday quotes in parallel.
    // _waitForQuotesPromise() polls until boot() has set window._quotesReadyPromise
    // (typically within 0–50 ms) then awaits it, guaranteeing STOOQ_RT_CACHE is
    // fully populated before renderCIPForwards() runs.
    // Without polling, bootNewFeatures() can reach this await before boot() has
    // assigned the promise (both run concurrently), causing Promise.resolve(undefined)
    // to resolve immediately and forwards to render as —.
    function _waitForQuotesPromise(timeoutMs) {
      return new Promise(function (resolve) {
        var deadline = Date.now() + (timeoutMs || 8000);
        (function poll() {
          if (window._quotesReadyPromise) {
            Promise.resolve(window._quotesReadyPromise).then(resolve, resolve);
          } else if (Date.now() < deadline) {
            setTimeout(poll, 20);
          } else {
            resolve(); // timed out — renderCIPForwards fallback handles it
          }
        })();
      });
    }
    await Promise.all([
      loadCBRatesCache(),
      loadOISRatesCache(),
      _waitForQuotesPromise(8000),
    ]);

    // All three panels fetch their own data independently.
    // renderRRInFXTable has its own direct rr.json fallback — no need to poll.
    // Run everything in parallel immediately after rates are ready.
    await Promise.all([
      renderCIPForwards(),
      renderRRInFXTable(),
      renderEconSurprises(),
    ]);

    // Refresh every 5 min
    setInterval(async () => {
      await Promise.all([loadCBRatesCache(), loadOISRatesCache()]);
      await renderCIPForwards();
      await renderRRInFXTable();
    }, 5 * 60 * 1000);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    // DOMContentLoaded already fired (dashboard.js is deferred — this runs after)
    run();
  }
})();


// ═══════════════════════════════════════════════════════════════════
// Personal Watchlist — localStorage-backed sidebar widget
// Pairs are stored as a JSON array under 'gi_watchlist' key.
// Prices are sourced from the intraday quotes cache (loadIntradayQuotes).
// FIX-WL (v7.91.0): Three bugs corrected —
//   1. render() called at init() before window._intradayQuotes is populated;
//      now defers with a short poll so prices show immediately on load.
//   2. gi:quotesLoaded listener was the only re-render path; if boot() already
//      ran (90s cache hit), the event never fired after init(). Retained as
//      primary path; poll fallback covers the cached case.
//   3. Duplicate event-listener registration on every addSymbol/remove call
//      replaced by event delegation on the container.
// ═══════════════════════════════════════════════════════════════════
(function initWatchlist() {
  'use strict';

  var WL_KEY = 'gi_watchlist';

  // Map user-entered symbol to intraday quotes key
  var SYMBOL_MAP = {
    'EURUSD': 'EURUSD', 'GBPUSD': 'GBPUSD', 'USDJPY': 'USDJPY',
    'AUDUSD': 'AUDUSD', 'USDCAD': 'USDCAD', 'USDCHF': 'USDCHF',
    'NZDUSD': 'NZDUSD', 'GBPJPY': 'GBPJPY', 'EURJPY': 'EURJPY',
    'EURGBP': 'EURGBP', 'AUDJPY': 'AUDJPY', 'EURAUD': 'EURAUD',
    'EURCHF': 'EURCHF', 'EURCAD': 'EURCAD', 'GBPCHF': 'GBPCHF',
    'GBPCAD': 'GBPCAD', 'GBPAUD': 'GBPAUD', 'CADJPY': 'CADJPY',
    'CHFJPY': 'CHFJPY', 'NZDJPY': 'NZDJPY', 'AUDNZD': 'AUDNZD',
    'XAUUSD': 'XAUUSD', 'XAGUSD': 'XAGUSD',
  };

  // Map watchlist symbol to TradingView FX_IDC symbol used by loadTVChart / sidebar handler.
  // XAUUSD and XAGUSD use the same FX_IDC prefix — loadTVChart will fall back to TV widget
  // if no OHLC file exists, which is the correct behaviour for commodities.
  var TV_SYM_PREFIX = 'FX_IDC:';

  // FIX-WL-4: In-memory fallback for environments where localStorage is blocked
  // (Privacy Badger, Tracking Prevention, Safari ITP, etc.).
  // When setItem() throws OR a subsequent getItem() round-trip returns null (silent
  // failure under Tracking Prevention), we fall back to a module-scoped array so
  // the watchlist remains functional for the session even without persistence.
  var _memList = null; // null = not yet initialised; [] after first load attempt

  function _lsAvailable() {
    // Test once per session — result is cached on _lsOk.
    if (typeof _lsAvailable._ok !== 'undefined') return _lsAvailable._ok;
    try {
      var t = '__gi_wl_test__';
      localStorage.setItem(t, '1');
      var ok = localStorage.getItem(t) === '1';
      localStorage.removeItem(t);
      _lsAvailable._ok = ok;
    } catch (e) {
      _lsAvailable._ok = false;
    }
    return _lsAvailable._ok;
  }

  function load() {
    if (_lsAvailable()) {
      try { return JSON.parse(localStorage.getItem(WL_KEY) || '[]'); } catch (e) {}
    }
    // localStorage unavailable — use in-memory list
    if (_memList === null) _memList = [];
    return _memList.slice();
  }
  function save(list) {
    if (_lsAvailable()) {
      try { localStorage.setItem(WL_KEY, JSON.stringify(list)); return; } catch (e) {}
    }
    // localStorage unavailable — persist in memory for this session
    _memList = list.slice();
  }

  function render() {
    var tbody = document.getElementById('watchlist-rows');
    if (!tbody) return;
    var list = load();
    if (list.length === 0) {
      tbody.innerHTML = '<div style="padding:4px 8px;font-size:10px;color:var(--text3);">No pairs added</div>';
      return;
    }
    var quotes = (window._intradayQuotes && window._intradayQuotes.quotes) || {};
    // FIX-WL-1: If quotes are not yet loaded, show skeleton prices and schedule
    // a re-render after a short delay rather than showing — permanently.
    var quotesReady = Object.keys(quotes).length > 0;
    tbody.innerHTML = list.map(function (sym) {
      var q = quotes[sym.toLowerCase()] || quotes[sym] || {};
      var price = (q.close != null) ? String(q.close) : (quotesReady ? '—' : '···');
      var chg = (q.pct != null) ? q.pct : null;
      var chgStr = (chg != null) ? ((chg >= 0 ? '+' : '') + chg.toFixed(2) + '%') : (quotesReady ? '—' : '···');
      var chgColor = (chg == null) ? 'var(--text3)' : (chg >= 0 ? 'var(--up)' : 'var(--down)');
      var tvSym = TV_SYM_PREFIX + sym;
      // data-sym makes this row compatible with the sidebar's delegated click handler
      // (line ~5650) which calls loadTVChart() + toggleSidebarDetail() automatically.
      // cursor:pointer and title match Crosses row conventions.
      return '<div class="sb-row" data-sym="' + tvSym + '" style="display:flex;align-items:center;gap:0;cursor:pointer;" title="Click to open chart">' +
        '<span class="sb-sym" style="flex:1;">' + sym + '</span>' +
        '<span class="sb-price" style="min-width:52px;text-align:right;font-family:var(--font-mono);font-size:10px;">' + price + '</span>' +
        '<span style="min-width:42px;text-align:right;font-family:var(--font-mono);font-size:10px;color:' + chgColor + ';">' + chgStr + '</span>' +
        '<button data-wl-remove="' + sym + '" style="background:none;border:none;cursor:pointer;color:var(--text3);font-size:11px;padding:0 4px;line-height:1;" aria-label="Remove ' + sym + '" title="Remove">&times;</button>' +
        '</div>';
    }).join('');
    // FIX-WL-1: If quotes weren't ready yet, retry after boot() has had time to load them.
    if (!quotesReady) {
      setTimeout(render, 800);
    }
  }

  function addSymbol(rawInput) {
    var sym = rawInput.trim().toUpperCase().replace(/[^A-Z]/g, '');
    if (!sym) return;
    if (!(sym in SYMBOL_MAP)) return; // only supported symbols
    var list = load();
    if (list.indexOf(sym) !== -1) return; // no duplicates
    if (list.length >= 8) { list.shift(); } // max 8 pairs
    list.push(sym);
    save(list);
    render();
  }

  function init() {
    var addBtn = document.getElementById('wl-add-btn');
    var inputRow = document.getElementById('wl-input-row');
    var input = document.getElementById('wl-input');
    var tbody = document.getElementById('watchlist-rows');
    if (!addBtn || !inputRow || !input || !tbody) return;

    render();

    addBtn.addEventListener('click', function () {
      var visible = inputRow.style.display !== 'none';
      inputRow.style.display = visible ? 'none' : 'block';
      if (!visible) {
        input.value = '';
        input.focus();
        // Scroll the input into view in case the watchlist section is near the
        // bottom of the sidebar and partially outside the visible scroll area.
        setTimeout(function () { inputRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, 50);
      }
    });

    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        var sym = input.value.trim().toUpperCase().replace(/[^A-Z]/g, '');
        if (sym && !(sym in SYMBOL_MAP)) {
          // Unknown symbol — shake the input briefly as visual feedback, don't close
          input.style.outline = '1px solid var(--down)';
          setTimeout(function () { input.style.outline = ''; }, 800);
          return;
        }
        addSymbol(input.value);
        input.value = '';
        inputRow.style.display = 'none';
        // Scroll the new row into view so the user sees it was added
        setTimeout(function () {
          var rows = tbody.querySelectorAll('.sb-row');
          if (rows.length) rows[rows.length - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 50);
      } else if (e.key === 'Escape') {
        inputRow.style.display = 'none';
      }
    });

    // FIX-WL-3: Use event delegation on the container instead of attaching
    // individual click listeners on every remove button on each render() call.
    // The old approach accumulated O(n * renders) listeners on the same nodes.
    // stopPropagation prevents the remove click from also triggering the sidebar's
    // delegated click handler (which would open the chart for a removed pair).
    tbody.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-wl-remove]');
      if (!btn) return;
      e.stopPropagation();
      var sym = btn.getAttribute('data-wl-remove');
      save(load().filter(function (s) { return s !== sym; }));
      render();
    });

    // FIX-WL-2: gi:quotesLoaded fires when boot() finishes loadIntradayQuotes().
    // On a 90s cache hit boot() runs synchronously before init() — the event
    // won't fire again. The render() retry loop above covers this case, but we
    // also keep the event listener as the primary fast path.
    document.addEventListener('gi:quotesLoaded', render);
    // Periodic refresh every 30s keeps prices current as the intraday cache updates.
    setInterval(render, 30000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

// =============================================================================
// TIMEFRAME SELECTOR — H1 · H4 · D1 · W1 · MN
// =============================================================================

const _TF_RANGE_SETS = {
  H1: [{days:1,label:'1D'},{days:5,label:'1W'},{days:14,label:'2W'},{days:30,label:'1M'}],
  H4: [{days:14,label:'2W'},{days:30,label:'1M'},{days:91,label:'3M'},{days:182,label:'6M'}],
  D1: [{days:91,label:'3M'},{days:182,label:'6M'},{days:365,label:'1Y'},{days:1095,label:'3Y'},{days:0,label:'ALL'}],
  W1: [{days:182,label:'6M'},{days:365,label:'1Y'},{days:730,label:'2Y'},{days:1095,label:'3Y'},{days:0,label:'ALL'}],
  MN: [{days:365,label:'1Y'},{days:1095,label:'3Y'},{days:1825,label:'5Y'},{days:0,label:'ALL'}],
};
const _TF_DEFAULT_DAYS = {H1:5,H4:14,D1:91,W1:365,MN:1095};

function _lwUpdateRangeBtns() {
  const wrap = document.getElementById('lw-range-btns');
  if (!wrap) return;
  const set = _TF_RANGE_SETS[_lwActiveTf] || _TF_RANGE_SETS.D1;
  wrap.innerHTML = set.map(r =>
    `<button class="lw-range-btn${r.days === _lwActiveDays ? ' active' : ''}" data-days="${r.days}" aria-label="${r.label}" style="flex-shrink:0;">${r.label}</button>`
  ).join('');
}

// TF button click handler (delegated on the range-bar)
document.getElementById('lw-range-bar')?.addEventListener('click', e => {
  const tfBtn = e.target.closest('.lw-tf-btn');
  if (!tfBtn) return;
  const newTf = tfBtn.dataset.tf;
  if (!newTf || newTf === _lwActiveTf) return;
  _lwActiveTf   = newTf;
  _lwActiveDays = _TF_DEFAULT_DAYS[newTf] ?? 91;
  _lwClearCompare(); // compare series has different timestamps on different TFs
  document.querySelectorAll('.lw-tf-btn').forEach(b => b.classList.toggle('sel', b.dataset.tf === newTf));
  _lwUpdateRangeBtns();
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

// =============================================================================
// COMPARE OVERLAY — normalised % change LineSeries on secondary price scale
// =============================================================================

// Toggle compare dropdown open/close
document.getElementById('lw-cmp-btn')?.addEventListener('click', function(e) {
  e.stopPropagation();
  const dd = document.getElementById('lw-cmp-dropdown');
  if (!dd) return;
  const open = dd.style.display === 'none' || !dd.style.display;
  if (open) {
    // Position with fixed coords to escape any overflow:hidden ancestor
    const rect = this.getBoundingClientRect();
    dd.style.position  = 'fixed';
    dd.style.top       = (rect.bottom + 4) + 'px';
    dd.style.right     = (window.innerWidth - rect.right) + 'px';
    dd.style.left      = 'auto';
    dd.style.zIndex    = '9100';
    dd.style.display   = 'block';
  } else {
    dd.style.display = 'none';
  }
  this.setAttribute('aria-expanded', String(open));
});
// Close on outside click
document.addEventListener('click', () => {
  const dd = document.getElementById('lw-cmp-dropdown');
  if (dd) dd.style.display = 'none';
  document.getElementById('lw-cmp-btn')?.setAttribute('aria-expanded','false');
});

// Item selection in compare dropdown
document.getElementById('lw-cmp-dropdown')?.addEventListener('click', e => {
  e.stopPropagation();
  const item = e.target.closest('.lw-cmp-item');
  if (!item || !_lwActiveOhlcId) return;
  const cmpId   = item.dataset.cmpid;
  const cmpType = item.dataset.cmptype || 'ohlc';
  if (!cmpId) return;
  // For ohlc: prevent comparing a symbol with itself; for cot/rate: always allow
  if (cmpType === 'ohlc' && cmpId === _lwActiveOhlcId) return;
  const uid = cmpType + ':' + cmpId;
  if (uid === _lwCompareId) { _lwClearCompare(); return; }
  _lwLoadCompare(cmpId, item.textContent.trim(), cmpType);
  document.getElementById('lw-cmp-dropdown').style.display = 'none';
});

function _lwClearCompare() {
  if (_lwCompareSeries && _lwChart) {
    try { _lwChart.removeSeries(_lwCompareSeries); } catch(_e) {}
  }
  _lwCompareSeries = null;
  _lwCompareId     = null;
  document.querySelectorAll('.lw-cmp-item').forEach(i => i.classList.remove('active'));
  document.getElementById('lw-cmp-pill')?.remove();
}

async function _lwLoadCompare(cmpId, cmpLabel, cmpType = 'ohlc') {
  if (!_lwChart || !_lwCandleSeries) return;
  _lwClearCompare();
  const LWC = window.LightweightCharts;
  const uid = cmpType + ':' + cmpId;

  // ── Colour per type ────────────────────────────────────────────────────────
  const CMP_COLOR = cmpType === 'cot'  ? '#9c27b0'   // purple for COT
                  : cmpType === 'rate' ? '#26a69a'   // teal for CB rate
                  : cmpType === 'esi'  ? '#2196f3'   // blue for ESI
                  :                      '#f0a500';  // amber for price overlay

  try {
    let seriesData = [];
    let priceFormat;

    // ── OHLC price overlay (existing behaviour) ───────────────────────────
    if (cmpType === 'ohlc') {
      let cmpPath;
      if (_lwActiveTf === 'H1')      cmpPath = `./ohlc-data/h1/${cmpId}.json`;
      else if (_lwActiveTf === 'H4') cmpPath = `./ohlc-data/h4/${cmpId}.json`;
      else                           cmpPath = `./ohlc-data/${cmpId}.json`;

      const r = await fetch(cmpPath, { signal: AbortSignal.timeout(6000) });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      let cmpBars = await r.json();
      if (!Array.isArray(cmpBars) || cmpBars.length < 4) throw new Error('no data');

      // Aggregate W1/MN
      if (_lwActiveTf === 'W1' || _lwActiveTf === 'MN') {
        const agg = {};
        for (const b of cmpBars) {
          let key;
          if (_lwActiveTf === 'W1') {
            const d = new Date(b.time + 'T00:00:00Z');
            const dow = d.getUTCDay() || 7;
            const mon = new Date(d); mon.setUTCDate(d.getUTCDate() - (dow-1));
            key = mon.toISOString().slice(0, 10);
          } else { key = b.time.slice(0,7) + '-01'; }
          if (!agg[key]) agg[key] = {time:key, open:b.open, high:b.high, low:b.low, close:b.close};
          else { const a=agg[key]; a.high=Math.max(a.high,b.high); a.low=Math.min(a.low,b.low); a.close=b.close; }
        }
        cmpBars = Object.values(agg).sort((a,b) => a.time < b.time ? -1 : 1);
      }

      // Normalise to % change from first visible bar
      let baseIdx = 0;
      try {
        const range = _lwChart.timeScale().getVisibleLogicalRange();
        if (range && range.from > 0) baseIdx = Math.max(0, Math.floor(range.from));
      } catch(_e) {}
      const basePrice = cmpBars[Math.min(baseIdx, cmpBars.length-1)]?.close;
      if (!basePrice || basePrice <= 0) throw new Error('no base price');

      seriesData  = cmpBars.map(b => ({ time: b.time, value: ((b.close - basePrice) / basePrice) * 100 }));
      priceFormat = { type: 'custom', formatter: v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%' };

    // ── COT Net Position (Leveraged Funds) ────────────────────────────────
    } else if (cmpType === 'cot') {
      const r = await fetch(`./cot-data/${cmpId}.json`, { signal: AbortSignal.timeout(6000) });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      const history = Array.isArray(d.history) ? d.history : [];
      if (history.length < 2) throw new Error('no COT history');

      // Add current week as the last point
      const allPoints = [
        ...history,
        { weekEnding: d.weekEnding, levNet: d.netPosition }
      ];

      seriesData = allPoints
        .filter(h => h.weekEnding && h.weekEnding.length === 10)
        .map(h => ({
          time:  h.weekEnding,
          value: h.levNet ?? ((h.levLong || 0) - (h.levShort || 0)),
        }))
        .sort((a, b) => a.time < b.time ? -1 : 1);

      // Remove duplicates (same weekEnding)
      seriesData = seriesData.filter((p, i) => i === 0 || p.time !== seriesData[i-1].time);
      if (seriesData.length < 2) throw new Error('insufficient COT points');

      priceFormat = {
        type: 'custom',
        formatter: v => {
          const abs = Math.abs(v);
          const str = abs >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toFixed(0);
          return (v >= 0 ? '+' : '') + str;
        }
      };

    // ── CB Policy Rate (step-line) ─────────────────────────────────────────
    } else if (cmpType === 'rate') {
      const r = await fetch(`./rates/${cmpId}.json`, { signal: AbortSignal.timeout(6000) });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      const obs = Array.isArray(d.observations) ? d.observations : [];
      if (obs.length < 2) throw new Error('no rate observations');

      // observations are newest-first — reverse to oldest-first for LWC
      seriesData = obs
        .filter(o => o.date && o.value != null)
        .map(o => ({ time: o.date, value: parseFloat(o.value) }))
        .sort((a, b) => a.time < b.time ? -1 : 1);

      priceFormat = {
        type: 'custom',
        formatter: v => v.toFixed(2) + '%'
      };

    // ── ESI (Economic Surprise Index, CESI-style) ─────────────────────────
    } else if (cmpType === 'esi') {
      // Fetch calendar.json — same source used by the ESI panel and modal
      const r = await fetch('./calendar-data/calendar.json', { signal: AbortSignal.timeout(8000) });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const calj = await r.json();
      const allEvents = calj.events || [];
      if (calj.surpriseStats) window._ECON_SURPRISE_STATS = calj.surpriseStats;

      // Use existing modal functions if already loaded, otherwise compute inline
      if (typeof _esmBuildSeries === 'function') {
        seriesData = _esmBuildSeries(allEvents, cmpId);
      } else {
        // Inline ESI computation — mirrors _esmBuildSeries / _esmScoreWindow
        const DECAY_LAMBDA = Math.LN2 / 45;
        const WINDOW_MS    = 90 * 24 * 60 * 60 * 1000;
        const STEP_MS      =  7 * 24 * 60 * 60 * 1000;
        const NOISE_KW     = [
          'cftc','baker hughes','rig count','auction','api weekly',
          'milk auction','fed\'s balance sheet','reserve balances',
          'redbook','ibd/tipp','tips auction','note auction','bond auction',
          'gilt auction','jgb auction','obligaciones','speculative net',
          'nc net position','crude oil inventories','crude oil imports',
          'distillate','gasoline inventorie','gasoline production',
          'refinery','heating oil','natural gas storage',
          'foreign bonds buying','foreign investments in japanese',
          'foreign bond investment','foreign investment in japan',
          'm2 money','m3 money','m4 money','reserve assets total',
          'cb leading index','atlanta fed gdpnow','ny fed','cleveland cpi',
          'ibd','3-month bill','4-week bill','52-week bill',
          '4-week average','4-week avg',
          'tic net','net long-term tic','total net tic',
          'interest rate projection',
          'eia crude oil','eia crude',
        ];
        const INVERSE_KW   = ['unemployment','jobless','claims','deficit','trade balance'];
        const stats        = window._ECON_SURPRISE_STATS || {};

        function _scoreWin(startMs, endMs) {
          const seen = new Set();
          let total=0, wTotal=0, wBeats=0, wMisses=0;
          let zWSum=0, zWTotal=0, zWBeats=0, zWMisses=0;
          allEvents.forEach(ev => {
            if (ev.currency !== cmpId) return;
            const t = new Date(ev.dateISO).getTime();
            if (isNaN(t) || t > endMs || t < startMs) return;
            if (!ev.actual || ev.actual === '' || ev.actual === '-') return;
            if (!['medium','high'].includes(ev.impact)) return;
            const name = (ev.event || '').toLowerCase();
            if (NOISE_KW.some(k => name.includes(k))) return;
            const canon = name.replace(/\s*\([^)]*\)/g,'').trim();
            const aS = String(ev.actual||'').replace(/[%,\s]/g,'');
            const fS = String(ev.forecast||ev.previous||'').replace(/[%,\s]/g,'');
            const key = `${cmpId}/${canon}/${aS}/${fS}`;
            if (seen.has(key)) return;
            seen.add(key);
            const actual   = parseFloat(String(ev.actual||'').replace(/[%,]/g,''));
            const forecast = parseFloat(String(ev.forecast||ev.previous||'').replace(/[%,]/g,''));
            if (isNaN(actual) || isNaN(forecast)) return;
            const inv     = INVERSE_KW.some(k => name.includes(k));
            const beat    = inv ? actual < forecast : actual > forecast;
            const miss    = inv ? actual > forecast : actual < forecast;
            const surp    = inv ? -(actual-forecast) : (actual-forecast);
            const ageDays = (endMs - t) / 86400000;
            const w       = Math.exp(-DECAY_LAMBDA * ageDays);
            const st      = stats[`${cmpId}/${canon}`];
            const useZ    = st && st.n >= 5 && st.std > 0;
            const z       = useZ ? (surp - st.mean) / st.std : null;
            total++; wTotal += w;
            if (beat) { wBeats  += w; }
            if (miss) { wMisses += w; }
            if (z !== null) {
              zWSum += z*w; zWTotal += w;
              if (beat) zWBeats += w;
              if (miss) zWMisses += w;
            }
          });
          if (!total) return null;
          const zFrac = zWTotal / wTotal;
          let idx100;
          if (zWTotal >= 10 || (zWTotal > 0 && zFrac >= 0.30)) {
            const nZW=wTotal-zWTotal, nZWB=wBeats-zWBeats, nZWM=wMisses-zWMisses;
            const zP  = zWTotal>0 ? (zWSum/zWTotal)*50 : 0;
            const bmP = nZW>0 ? ((nZWB-nZWM)/nZW)*100 : 0;
            idx100 = (zP*zWTotal + bmP*nZW) / wTotal;
          } else {
            idx100 = wTotal>0 ? ((wBeats-wMisses)/wTotal)*100 : 0;
          }
          return idx100;
        }

        const ccyEvts = allEvents.filter(ev =>
          ev.currency === cmpId && ev.actual && ev.actual !== '' && ev.actual !== '-'
        );
        if (!ccyEvts.length) throw new Error('no ESI events for ' + cmpId);

        const minDate = Math.min(...ccyEvts.map(ev => new Date(ev.dateISO).getTime()).filter(t => !isNaN(t)));
        const nowMs = Date.now();
        let cursor = minDate + WINDOW_MS;
        while (cursor <= nowMs + STEP_MS) {
          const endMs   = Math.min(cursor, nowMs);
          const startMs = endMs - WINDOW_MS;
          const idx     = _scoreWin(startMs, endMs);
          if (idx !== null) {
            const dt = new Date(endMs);
            seriesData.push({
              time:  `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`,
              value: parseFloat(idx.toFixed(2)),
            });
          }
          cursor += STEP_MS;
        }
      }

      if (seriesData.length < 2) throw new Error('insufficient ESI data');

      priceFormat = {
        type: 'custom',
        formatter: v => (v >= 0 ? '+' : '') + v.toFixed(1)
      };
    }

    if (!seriesData.length) throw new Error('empty data');

    // ── Render series ──────────────────────────────────────────────────────
    // All types use LineSeries: ohlc → % change, cot → net contracts, rate → step-line, esi → index
    _lwCompareSeries = LWC.LineSeries
      ? _lwChart.addSeries(LWC.LineSeries, {
          color: CMP_COLOR, lineWidth: cmpType === 'rate' ? 2 : 1.5,
          priceScaleId: 'cmp', priceFormat,
          lastValueVisible: false, priceLineVisible: false,
          crosshairMarkerVisible: cmpType !== 'ohlc' })
      : _lwChart.addLineSeries({
          color: CMP_COLOR, lineWidth: cmpType === 'rate' ? 2 : 1.5,
          priceScaleId: 'cmp', priceFormat,
          lastValueVisible: false, priceLineVisible: false,
          crosshairMarkerVisible: cmpType !== 'ohlc' });

      // For rate: expand monthly observations to daily step-line so it aligns with the chart
      if (cmpType === 'rate') {
        const expanded = [];
        for (let i = 0; i < seriesData.length; i++) {
          const cur  = seriesData[i];
          const next = seriesData[i + 1];
          expanded.push(cur);
          if (next) {
            // Fill every month between cur and next with cur's value
            let d = new Date(cur.time + 'T00:00:00Z');
            d.setUTCMonth(d.getUTCMonth() + 1);
            while (d.toISOString().slice(0,10) < next.time) {
              expanded.push({ time: d.toISOString().slice(0,10), value: cur.value });
              d.setUTCMonth(d.getUTCMonth() + 1);
            }
          }
        }
        // Extend to today
        const today = new Date().toISOString().slice(0,10);
        const last  = seriesData[seriesData.length - 1];
        let d = new Date(last.time + 'T00:00:00Z');
        d.setUTCMonth(d.getUTCMonth() + 1);
        while (d.toISOString().slice(0,10) <= today) {
          expanded.push({ time: d.toISOString().slice(0,10), value: last.value });
          d.setUTCMonth(d.getUTCMonth() + 1);
        }
        seriesData = expanded;
      }

    try {
      _lwChart.priceScale('cmp').applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
        borderVisible: false, textColor: CMP_COLOR,
      });
    } catch(_e) {}

    _lwCompareSeries.setData(seriesData);
    _lwCompareId = uid;

    document.querySelectorAll('.lw-cmp-item').forEach(i =>
      i.classList.toggle('active',
        (i.dataset.cmptype || 'ohlc') === cmpType && i.dataset.cmpid === cmpId));

    // Add pill
    const indPills = document.getElementById('lw-ind-pills');
    if (indPills) {
      const pill = document.createElement('span');
      pill.id = 'lw-cmp-pill';
      pill.className = 'lw-cmp-pill';
      pill.title = 'Remove compare overlay';
      pill.innerHTML = `<span style="width:8px;height:2px;background:${CMP_COLOR};display:inline-block;border-radius:1px;"></span> ${cmpLabel} ×`;
      pill.addEventListener('click', _lwClearCompare);
      indPills.parentNode.insertBefore(pill, indPills);
    }
  } catch(err) {
    console.warn('[lw-compare] Failed to load compare data:', err.message);
  }
}

// =============================================================================
// FULLSCREEN CHART — DOM-lift: move the real chart panel into the overlay
// This preserves ALL indicators, compare series, CB markers, event handlers.
// =============================================================================

let _lwFsOriginalParent = null;
let _lwFsOriginalNext   = null;
let _lwFsOriginalHeight = null;

function _lwOpenFullscreen() {
  const overlay   = document.getElementById('lw-fullscreen-overlay');
  const inner     = document.getElementById('lw-fullscreen-inner');
  const rangeBar  = document.getElementById('lw-range-bar');
  const chartHdr  = document.getElementById('lw-chart-header');
  const chartWrap = document.getElementById('tv-chart-wrap');
  if (!overlay || !inner || !chartWrap || _chartMode !== 'lw') return;
  if (overlay.classList.contains('lw-fs-active')) return;

  // Store anchor: the element immediately BEFORE rangeBar so we can
  // restore the full block (rangeBar→chartHdr→chartWrap) in one shot.
  _lwFsOriginalParent = rangeBar ? rangeBar.parentNode : chartWrap.parentNode;
  _lwFsOriginalNext   = chartWrap.nextSibling;     // element AFTER chartWrap
  _lwFsOriginalHeight = chartWrap.style.height;

  // Lift all three elements into the fullscreen inner container
  if (rangeBar)  inner.appendChild(rangeBar);
  if (chartHdr)  inner.appendChild(chartHdr);
  inner.appendChild(chartWrap);

  chartWrap.style.height    = '100%';
  chartWrap.style.minHeight = '0';
  chartWrap.style.flex      = '1';

  // Populate the FS tab strip to mirror the real pair tabs
  _lwFsPopulateTabs();

  overlay.classList.add('lw-fs-active');
  document.body.style.overflow = 'hidden';

  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (_lwChart && chartWrap) {
      // Use chartWrap (not inner) — inner also contains rangeBar and chartHdr above the chart.
      // Sizing to inner.offsetHeight makes the chart taller than its actual container,
      // pushing the time axis off the bottom edge.
      const w = chartWrap.offsetWidth  || inner.offsetWidth;
      const h = chartWrap.offsetHeight || inner.offsetHeight;
      if (w > 0 && h > 0) _lwChart.resize(w, h);
    }
  }));
}

function _lwCloseFullscreen() {
  const overlay   = document.getElementById('lw-fullscreen-overlay');
  const inner     = document.getElementById('lw-fullscreen-inner');
  const rangeBar  = document.getElementById('lw-range-bar');
  const chartHdr  = document.getElementById('lw-chart-header');
  const chartWrap = document.getElementById('tv-chart-wrap');
  if (!overlay || !overlay.classList.contains('lw-fs-active')) return;

  overlay.classList.remove('lw-fs-active');
  document.body.style.overflow = '';

  // Restore all three elements before the stored next-sibling reference.
  // insertBefore with a null ref appends to end, which is also correct.
  if (_lwFsOriginalParent) {
    if (rangeBar)  _lwFsOriginalParent.insertBefore(rangeBar,  _lwFsOriginalNext);
    if (chartHdr)  _lwFsOriginalParent.insertBefore(chartHdr,  _lwFsOriginalNext);
    if (chartWrap) _lwFsOriginalParent.insertBefore(chartWrap, _lwFsOriginalNext);
  }

  if (chartWrap) {
    chartWrap.style.height    = _lwFsOriginalHeight || '';
    chartWrap.style.minHeight = '';
    chartWrap.style.flex      = '';
  }

  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (_lwChart && chartWrap) {
      const w = chartWrap.offsetWidth, h = chartWrap.offsetHeight;
      if (w > 0 && h > 0) _lwChart.resize(w, h);
    }
  }));

  _lwFsOriginalParent = null;
  _lwFsOriginalNext   = null;
}

// Populate FS toolbar tab strip to mirror the main pair tabs
function _lwFsPopulateTabs() {
  // lw-fs-tabs is the scrollable inner strip; its parent lw-fs-tab-outer has ‹ › scroll buttons
  const fsOuter = document.getElementById('lw-fs-tab-outer');
  const fsTabs  = document.getElementById('lw-fs-tabs');
  if (!fsTabs) return;
  const realTabs = document.querySelectorAll('#tv-pair-tabs .tv-tab');
  if (!realTabs.length) return;
  fsTabs.innerHTML = '';
  realTabs.forEach(realTab => {
    const btn = document.createElement('button');
    btn.className = realTab.className;  // copies 'tv-tab active' etc.
    btn.textContent = realTab.textContent;
    btn.dataset.sym = realTab.dataset.sym;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', realTab.getAttribute('aria-selected'));
    btn.addEventListener('click', () => {
      realTab.click();
      fsTabs.querySelectorAll('.tv-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.sym === realTab.dataset.sym);
        b.setAttribute('aria-selected', b.dataset.sym === realTab.dataset.sym ? 'true' : 'false');
      });
      // Scroll active tab into view
      btn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    });
    fsTabs.appendChild(btn);
  });

  // Wire ‹ › scroll buttons (same logic as tv-tabs-prev/next in main toolbar)
  if (fsOuter) {
    const prevBtn = fsOuter.querySelector('#lw-fs-tabs-prev');
    const nextBtn = fsOuter.querySelector('#lw-fs-tabs-next');
    const updateArrows = () => {
      if (!prevBtn || !nextBtn) return;
      prevBtn.style.display = fsTabs.scrollLeft > 1 ? 'flex' : 'none';
      nextBtn.style.display = fsTabs.scrollLeft < fsTabs.scrollWidth - fsTabs.clientWidth - 1 ? 'flex' : 'none';
    };
    if (prevBtn) prevBtn.onclick = () => { fsTabs.scrollBy({ left: -160, behavior: 'smooth' }); setTimeout(updateArrows, 320); };
    if (nextBtn) nextBtn.onclick = () => { fsTabs.scrollBy({ left:  160, behavior: 'smooth' }); setTimeout(updateArrows, 320); };
    fsTabs.addEventListener('scroll', updateArrows, { passive: true });
    setTimeout(updateArrows, 50);
  }
}

// Keep FS tabs in sync when a real tab is clicked while NOT in fullscreen
document.getElementById('tv-pair-tabs')?.addEventListener('click', e => {
  const clicked = e.target.closest('.tv-tab');
  if (!clicked) return;
  const fsTabs = document.getElementById('lw-fs-tabs');
  if (!fsTabs) return;
  fsTabs.querySelectorAll('.tv-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.sym === clicked.dataset.sym);
    b.setAttribute('aria-selected', b.dataset.sym === clicked.dataset.sym ? 'true' : 'false');
  });
});

document.getElementById('lw-fs-btn')?.addEventListener('click', _lwOpenFullscreen);
document.getElementById('lw-fs-close')?.addEventListener('click', _lwCloseFullscreen);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && document.getElementById('lw-fullscreen-overlay')?.classList.contains('lw-fs-active'))
    _lwCloseFullscreen();
});
