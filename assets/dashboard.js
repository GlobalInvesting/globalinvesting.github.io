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

      // vs norm cell: badge based on z_score (always uses 60d corr as baseline — unchanged)
      const z = c.z_score;
      let normCell;
      if (z == null || c.norm == null) {
        normCell = `<td style="color:var(--text3)">—</td>`;
      } else {
        const absZ = Math.abs(z);
        const normSign = c.norm >= 0 ? '+' : '';
        let badgeCls, badgeLabel;
        if (absZ >= 2.5)      { badgeCls = 'down'; badgeLabel = '⚠ broken'; }
        else if (absZ >= 1.5) { badgeCls = 'down'; badgeLabel = '↯ break'; }
        else if (absZ >= 1.0) { badgeCls = '';     badgeLabel = '~ stretched'; }
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
        onclick="(function(el){ var idx=+el.dataset.corrIdx; var d=window._corrDataCache&&window._corrDataCache[idx]; if(d&&typeof openCorrModal==='function') openCorrModal(d); })(this)"
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
    // On weekdays: intraday high/low for current session.
    // On weekends: last Friday close values — shown dimmed per Bloomberg/Eikon convention.
    const rtD = STOOQ_RT_CACHE[pair.id];
    const sessH = (rtD?.high  != null) ? fmt(rtD.high,  pair.dec) : '—';
    const sessL = (rtD?.low   != null) ? fmt(rtD.low,   pair.dec) : '—';
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

function populateHeatmap() {
  const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD'];

  // Prefer STOOQ_RT_CACHE (intraday ~5min delay) over ECB daily rates
  // because ECB daily rates have zero intraday movement (same open/close on weekends)
  const rtAvailable = Object.keys(STOOQ_RT_CACHE).length >= 14; // need majority of 28 pairs

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
  grid.innerHTML = strengths.map(s => {
    let bg = 'h-flat';
    if (s.pct > 0.15) bg = 'h-s-up';
    else if (s.pct > 0.05) bg = 'h-up';
    else if (s.pct < -0.15) bg = 'h-s-down';
    else if (s.pct < -0.05) bg = 'h-down';
    const cls = s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat';
    const sign = s.pct >= 0 ? '+' : '';
    return `<div class="hm-cell ${bg}">
      <span class="hm-sym">${s.ccy}</span>
      <span class="hm-val ${cls}">${sign}${s.pct.toFixed(2)}</span>
    </div>`;
  }).join('');
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
          if (!r || typeof openCBRatesModal !== 'function') return;
          var bi  = (window._STATE_bankInfo && window._STATE_bankInfo[id]) || {};
          var mtg = window._STATE_meetings && window._STATE_meetings[id.toUpperCase()];
          openCBRatesModal(id.toUpperCase(), r.obs, bi, mtg);
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
      if (ccy && data && typeof openCOTModal === 'function') {
        openCOTModal(ccy, data);
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
    close:      q.close,
    prev_close: q.prev_close ?? null,
    open:       q.prev_close ?? q.close,
    chg:        hasPrev ? (q.chg  ?? null) : null,
    pct:        hasPrev ? (q.pct  ?? null) : null,
    fromIntraday: true,
    stale:      q.stale ?? false,
  };
}
// ──────────────────────────────────────────────────────────────────────────────

// Cache for intraday RT rates — fed by yfinance JSON, used to update FX table + heatmap
const STOOQ_RT_CACHE = {};  // id → { close, open, chg, pct }

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
        open:  q.prev_close ?? q.close,
        chg:   hasPrev ? (q.chg  ?? null) : null,
        pct:   hasPrev ? (q.pct  ?? null) : null,
        high:  (q.high  != null && q.high  > 0) ? q.high  : null,
        low:   (q.low   != null && q.low   > 0) ? q.low   : null,
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
      // HV30: update if data is available in cache (column index 6)
      if (tds[6] && data.hv30 != null) {
        tds[6].textContent = data.hv30.toFixed(1) + '%';
      }
      // SESS H / SESS L — populate from high/low; dim on weekends (last close values)
      const sessColor = _isWeekendRT ? 'var(--text3)' : 'var(--text1)';
      if (tds[7]) { tds[7].textContent = (data.high != null) ? fmt(data.high, pairCfg.dec) : '—'; tds[7].style.color = sessColor; }
      if (tds[8]) { tds[8].textContent = (data.low  != null) ? fmt(data.low,  pairCfg.dec) : '—'; tds[8].style.color = sessColor; }
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
    vEl.className = 'ca-val ' + cls;
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

  // ── Refresh heatmap with latest RT data ──
  populateHeatmap();

  // ── Timestamp ──
  const upd = document.getElementById('fx-table-updated');
  if (upd) {
    const now = new Date();
    const hh = now.getHours().toString().padStart(2,'0');
    const mm = now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    const _rtDay = now.getUTCDay(), _rtH = now.getUTCHours();
    const _rtWeekend = _rtDay === 6 || (_rtDay === 0 && _rtH < 21) || (_rtDay === 5 && _rtH >= 21);
    upd.textContent = _rtWeekend
      ? `yfinance · Last close: Fri · ~5min delay`
      : `yfinance · ${hh}:${mm} ${tzAbbr} · ~5min delay`;
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
// ═══════════════════════════════════════════════════════════════════
async function fetchCryptoQuotes() {
  try {
    const r = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true');
    if (!r.ok) return;
    const data = await r.json();
    if (data.bitcoin) {
      const price = data.bitcoin.usd;
      const chg24 = data.bitcoin.usd_24h_change;
      const priceEl = document.getElementById('q-btcusd');
      const chgEl   = document.getElementById('qc-btcusd');
      if (priceEl) { priceEl.textContent = price.toLocaleString(); priceEl.className = 'q-price ' + clsDir(chg24); }
      if (chgEl)   { chgEl.textContent = pctStr(chg24); chgEl.className = 'q-chg ' + clsDir(chg24); }
    }
  } catch(e) {}
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

  container.innerHTML = '';

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
        // tick: map currentPrice into [min-1.5r … max+1.5r]
        const lo    = Math.min(p.avgL, p.avgS);
        const hi    = Math.max(p.avgL, p.avgS);
        const range = (hi - lo) || domAvg * 0.01;
        tickPct = Math.min(98, Math.max(2,
          (currentPrice - (lo - range * 1.5)) / (range * 4) * 100
        ));
      }
    }

    const distCol  = trapped ? 'var(--down)' : 'var(--up)';
    const distSign = distPct !== null && distPct >= 0 ? '+' : '';

    // ── TV symbol: "EUR/USD" → "FX_IDC:EURUSD" ──
    const tvSym = 'FX_IDC:' + p.sym.replace('/', '');

    // ── Outer wrapper: two-row layout ──
    const wrap = document.createElement('div');
    wrap.style.cssText = 'border-bottom:1px solid var(--border);padding:5px 0 4px;cursor:pointer;';
    wrap.title = 'Click to open ' + p.sym + ' chart';
    wrap.addEventListener('click', () => loadTVChart(tvSym));

    // Row 1: sym | bar | %L %S bias
    const row1 = document.createElement('div');
    row1.style.cssText = 'display:grid;grid-template-columns:58px 1fr auto;align-items:center;gap:6px;';

    // Sym + positions
    const symDiv = document.createElement('div');
    symDiv.style.cssText = 'display:flex;flex-direction:column;gap:1px;min-width:0;';
    const symSpan = document.createElement('span');
    symSpan.style.cssText = 'font-size:10px;font-weight:700;color:#fff;font-family:var(--font-ui);white-space:nowrap;';
    symSpan.textContent = p.sym;
    symDiv.appendChild(symSpan);

    let posSpan = null;
    if (hasRich) {
      posSpan = document.createElement('span');
      posSpan.style.cssText = 'font-size:9px;color:var(--text3);font-family:var(--font-ui);cursor:help;display:inline-block;';
      posSpan.textContent = fmtK(p.totalPos) + ' pos';
      symDiv.appendChild(posSpan);
    }

    // Bar
    const barDiv = document.createElement('div');
    barDiv.style.cssText = 'position:relative;height:8px;background:var(--bg3);border-radius:1px;overflow:visible;cursor:help;';
    barDiv.innerHTML = `
      <div style="position:absolute;left:0;top:0;height:100%;width:${p.buy}%;background:var(--up);opacity:.85;border-radius:1px 0 0 1px;"></div>
      <div style="position:absolute;right:0;top:0;height:100%;width:${p.sell}%;background:var(--down);opacity:.85;border-radius:0 1px 1px 0;"></div>
    `;

    // Tick element (separate so we can attach its own tooltip)
    let tickEl = null;
    if (tickPct !== null) {
      tickEl = document.createElement('div');
      tickEl.style.cssText = `position:absolute;top:-3px;width:2px;height:14px;background:#fff;opacity:.9;border-radius:1px;left:${tickPct}%;transform:translateX(-1px);z-index:2;cursor:help;`;
      barDiv.appendChild(tickEl);
    }

    // Right block: %L %S bias
    const rightDiv = document.createElement('div');
    rightDiv.style.cssText = 'display:flex;align-items:center;gap:4px;flex-shrink:0;';

    const buySpan  = document.createElement('span');
    buySpan.style.cssText = 'font-size:10px;color:var(--up);font-family:var(--font-mono);cursor:help;';
    buySpan.textContent = p.buy + '%';

    const sellSpan = document.createElement('span');
    sellSpan.style.cssText = 'font-size:10px;color:var(--down);font-family:var(--font-mono);cursor:help;';
    sellSpan.textContent = p.sell + '%';

    const biasSpan = document.createElement('span');
    biasSpan.style.cssText = `font-size:9px;font-weight:700;color:${biasCol};font-family:var(--font-ui);min-width:10px;text-align:right;`;
    biasSpan.textContent = biasLbl;

    rightDiv.append(buySpan, sellSpan, biasSpan);
    row1.append(symDiv, barDiv, rightDiv);
    wrap.appendChild(row1);

    // Row 2: avg entry / dist info — show whenever we have rich data
    if (hasRich) {
      const row2 = document.createElement('div');
      row2.style.cssText = 'display:grid;grid-template-columns:58px 1fr auto;gap:6px;margin-top:2px;';

      const distSpan = document.createElement('span');
      distSpan.style.cssText = `font-size:9px;font-family:var(--font-mono);cursor:help;grid-column:2/4;`;

      let distTitle, distBody, distEx;

      if (distPct !== null) {
        // Full dist: live price available
        const distCol = trapped ? 'var(--down)' : 'var(--up)';
        distSpan.style.color = distCol;
        distSpan.innerHTML = `${trapped ? '▼' : '▲'} ${distSign}${distPct.toFixed(2)}% · ${distPips}pip · <span style="opacity:.7">${trapped ? 'trapped' : 'in profit'}</span>`;
        const domSideTxt = domLong ? 'longs' : 'shorts';
        distTitle = trapped ? 'Retail trapped' : 'Retail in profit';
        distBody  = `The dominant side (${domSideTxt}) entered at avg ${domAvg.toFixed(decimals)}. Current price: ${currentPrice.toFixed(decimals)}. They are ${trapped ? 'underwater (losing)' : 'in profit'}.`;
        distEx    = trapped
          ? 'If price continues against them, mass stop-outs can trigger a sharp move.'
          : 'They may take profits soon, creating pressure in the opposite direction.';
      } else {
        // Partial: show avg entry without live price
        const longDec = p.avgL > 20 ? 2 : 4;
        const shortDec = p.avgS > 20 ? 2 : 4;
        distSpan.style.color = 'var(--text3)';
        distSpan.innerHTML = `avg L ${p.avgL.toFixed(longDec)} · S ${p.avgS.toFixed(shortDec)}`;
        distTitle = 'Average entry price';
        distBody  = `Retail longs entered at avg ${p.avgL.toFixed(longDec)}, shorts at avg ${p.avgS.toFixed(shortDec)}. Live price not available for this pair — trapped/profit status cannot be calculated.`;
        distEx    = 'Compare these levels to the current chart price to assess whether the dominant side is underwater or in profit.';
      }

      attachTip(distSpan, distTitle, distBody, distEx);
      row2.append(document.createElement('span'), distSpan);
      wrap.appendChild(row2);
    }

    container.appendChild(wrap);

    // ── Attach tooltips ──
    if (posSpan) {
      attachTip(posSpan,
        'Open positions',
        `Number of Myfxbook traders with ${p.sym} open right now. Higher count = more statistically representative.`,
        `EUR/USD with 54K positions is the most-followed pair — mass stop-outs here move the market.`
      );
    }
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
    attachTip(buySpan,
      '% Long',
      `Percentage of retail traders currently holding long (buy) positions in ${p.sym}.`,
      `Readings above 70% long are unusual and often precede a drop as crowded longs get squeezed.`
    );
    attachTip(sellSpan,
      '% Short',
      `Percentage of retail traders currently holding short (sell) positions in ${p.sym}.`,
      `Readings above 70% short are unusual and often precede a rally as crowded shorts get squeezed.`
    );
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
    'Composite live assessment: VIX level (primary driver), yield curve shape, gold intraday demand (>2% = stress signal), S&P 500 daily move (< -1.5% = stress), and MOVE index (>100 = elevated per BofA/ICE). Updates in real time.',
    'RISK-ON: VIX <18, no stress signals active. MIXED: 1 stress factor (e.g. VIX 18–25). CAUTION: 2–3 factors. RISK-OFF: 4+ factors — high stress, USD/JPY/CHF bid, equities sold.'
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
    _enrich2('vix',   v => v > 5 && v < 100);
    _enrich2('us10y', v => v > 0 && v < 20);
    _enrich2('us2y',  v => v > 0 && v < 20);
    _enrich2('us3m',  v => v > 0 && v < 20);
    _enrich2('us5y',  v => v > 0 && v < 20);
    _enrich2('us30y', v => v > 0 && v < 20);
    _enrich2('dxy',   v => v > 50 && v < 130);
    _enrich2('move',  v => v > 10 && v < 400);
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
    const cls = vix > 30 ? 'risk-val down' : vix > 20 ? 'risk-val warning' : 'risk-val up';
    setEl('risk-vix', vix.toFixed(1), cls);
    const signal = vix > 30 ? 'High' : vix > 20 ? 'Elevated' : 'Low';
    const chg = byId.vix.chg || 0;
    const arrow = chg > 0 ? '▲' : chg < 0 ? '▼' : '→';
    const chgStr = (chg >= 0 ? ' +' : ' ') + chg.toFixed(1);
    const srcNote = byId.vix.fromRepo ? ' · FRED' : ' · CBOE';
    setEl('risk-vix-sub', arrow + chgStr + ' · ' + signal + srcNote);
  } else {
    setEl('risk-vix', '—', 'risk-val');
    setEl('risk-vix-sub', 'CBOE · unavailable');
  }

  // MOVE — from intraday quotes.json (yfinance ^MOVE). No external fallback.
  const move = (byId.move && byId.move.close > 10) ? byId.move : null;

  // MOVE Index — ^MOVE via yfinance (ICE BofA bond volatility index)
  {
    if (move && move.close > 10) {
      const cls = move.close > 150 ? 'risk-val down' : move.close > 100 ? 'risk-val warning' : 'risk-val up';
      setEl('risk-move', move.close.toFixed(1), cls);
      const signal = move.close > 150 ? 'High' : move.close > 100 ? 'Elevated' : 'Low';
      const arrow = move.chg > 0 ? '▲' : move.chg < 0 ? '▼' : '→';
      const chgStr = (move.chg >= 0 ? ' +' : ' ') + move.chg.toFixed(1);
      setEl('risk-move-sub', arrow + chgStr + ' · ' + signal + ' · ICE BofA');
    } else if (byId.us10y) {
      // Proxy: MOVE ≈ VIX-like measure from 10Y move
      const vixLevel = byId.vix ? byId.vix.close : 20;
      const approx = Math.round(vixLevel * 3.8);
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
      narrRegEl.title = `Live assessment · VIX ${vix.toFixed(1)}${isInverted ? ' · inverted curve' : ''}${narrTsLabel}`;
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
  // US–EU Spread 10Y (uses byId from this scope)
  if (byId.us10y && byId.de10y) {
    const spr = byId.us10y.close - byId.de10y.close;
    const bp = Math.round(spr * 100);
    const sign = bp >= 0 ? '+' : '';
    setEl('ri-us-eu', sign + bp + 'bp');
    setEl('ri-us-eu-sig', bp > 0 ? 'USD+' : 'EUR+', bp > 50 ? 'up' : bp < -50 ? 'down' : 'flat');
  }
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
        const corrSig = v > 0.2 ? 'Inflation bid' : v < -0.3 ? 'Normal' : 'Neutral';
        const corrCls = v > 0.2 ? 'down' : v < -0.3 ? 'up' : 'flat';
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
function loadCOTChart(longSym) {
  // Derive Short symbol: replace trailing _L with _S
  const shortSym = longSym.replace(/_L$/, '_S');
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
    allow_symbol_change: false,
    calendar: false,
    details: false,
    hide_side_toolbar: true,
    hide_top_toolbar: true,
    hide_legend: false,
    hide_volume: true,
    interval: 'W',
    locale: 'en',
    save_image: true,
    style: '2',
    symbol: longSym,
    theme: 'dark',
    timezone: 'Etc/UTC',
    backgroundColor: '#131722',
    gridColor: 'rgba(42,46,57,0.8)',
    withdateranges: false,
    compareSymbols: [{ symbol: shortSym, position: 'SameScale' }],
    scaleMode: 2,
    studies: [],
    autosize: true,
  });
  container.appendChild(script);
  wrap.appendChild(container);
  // Scroll chart into view
  const chartSection = document.getElementById('section-fxpairs') || wrap.closest('.panel') || wrap;
  chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// SHARED: load any symbol into the TradingView chart + scroll to it
// ═══════════════════════════════════════════════════════════════════
function loadTVChart(sym) {
  // Deactivate all tabs; activate matching tab if exists
  document.querySelectorAll('.tv-tab').forEach(t => {
    t.classList.remove('active');
    if (t.dataset.sym === sym) t.classList.add('active');
  });
  // Update pair detail panel (Eikon-style linked panels)
  updatePairDetail(sym);
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  wrap.innerHTML = '';
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
  // Scroll chart into view
  const chartSection = document.getElementById('section-fxpairs') || wrap.closest('.panel') || wrap;
  chartSection.scrollIntoView({ behavior:'smooth', block:'start' });
  setTimeout(minimizeTVLegend, 3000);
}

// ── Quote bar: click any item to open chart ──
document.getElementById('quotebar-inner')?.addEventListener('click', e => {
  const item = e.target.closest('.q-item');
  if (!item) return;
  const sym = item.dataset.sym;
  if (sym) loadTVChart(sym);
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
    if (inner) inner.style.maxHeight = '0';
    setTimeout(() => existingExpand.remove(), 180);
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
  td.colSpan = 9;
  const inner = document.createElement('div');
  inner.innerHTML = '<div style="padding:6px 10px;font-size:10px;color:var(--text3);">Loading…</div>';
  td.appendChild(inner);
  expandRow.appendChild(td);
  row.after(expandRow);

  // Animate open
  requestAnimationFrame(() => {
    expandRow.classList.add('pd-open');
    inner.style.maxHeight = '160px';
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

  // COT
  const cotCcy = base && base !== 'USD' ? base : (quote && quote !== 'USD' ? quote : base);
  const cotRaw = cotCcy ? (COT_DATA_CACHE[cotCcy] || null) : null;
  let cotNet = null, cotAmNet = null, cotWow = null, cotPctOI = null, cotWeek = '';
  if (cotRaw) {
    const flip = (invert && cotCcy === quote) ? -1 : 1;
    cotNet   = cotRaw.net   != null ? cotRaw.net   * flip : null;
    cotAmNet = cotRaw.amNet != null ? cotRaw.amNet * flip : null;
    cotWow   = cotRaw.wowNetChange != null ? cotRaw.wowNetChange * flip : null;
    cotPctOI = cotRaw.levNetPctOI  != null ? cotRaw.levNetPctOI  * flip : null;
    cotWeek  = cotRaw.weekEnding || '';
  }

  // Carry
  const cbBase  = base  ? (STATE.cbRates?.[base.toLowerCase()]?.rate  ?? null) : null;
  const cbQuote = quote ? (STATE.cbRates?.[quote.toLowerCase()]?.rate ?? null) : null;
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

  // COT summary tag
  const lfDir = cotNet == null ? null : cotNet > 0 ? 'Long' : cotNet < 0 ? 'Short' : null;
  const amDir = cotAmNet == null ? null : cotAmNet > 0 ? 'Long' : cotAmNet < 0 ? 'Short' : null;
  const aligned = lfDir && amDir && lfDir === amDir;
  const cotTag = lfDir && amDir
    ? `<span class="${clsI(cotNet)}">${lfDir}</span> <span style="color:var(--text3);font-size:8px;">LF · </span><span class="${clsI(cotAmNet)}">${amDir}</span> <span style="color:var(--text3);font-size:8px;">AM · ${aligned ? 'aligned' : 'diverging'}</span>`
    : '—';

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
          <div class="pd-inline-metric fx-tip" data-tip-title="Carry Differential" data-tip-body="CB rate differential: base minus quote. Positive = carry favours long." data-tip-ex="USD/JPY carry = Fed 4.5% − BoJ 0.5% = +4.0%.">
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
          <div class="pd-inline-metric fx-tip" data-tip-title="ATM Implied Volatility" data-tip-body="ATM IV from CBOE FX ETF options. Proxy for OTC interbank IV. Green ≤7%; red >12%.">
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
        <div class="pd-inline-group-lbl">COT Positioning</div>
        <div class="pd-inline-metrics">
          <div class="pd-inline-metric fx-tip" data-tip-title="CFTC Leveraged Funds Net" data-tip-body="Net contracts (longs minus shorts) held by Leveraged Funds — hedge funds and CTAs." data-tip-ex="Extreme net long historically precedes reversals as the crowd becomes crowded.">
            <div class="pd-inline-lbl">LF Net</div><div class="pd-inline-val ${clsI(cotNet)}">${fmtN(cotNet)}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="LF Week-over-Week Change" data-tip-body="Change in LF net contracts vs prior week. Primary momentum signal in institutional COT analysis." data-tip-ex="Reversal in WoW change is often the earliest signal of a positioning shift.">
            <div class="pd-inline-lbl">LF WoW Δ</div><div class="pd-inline-val ${clsI(cotWow)}">${fmtN(cotWow)}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="Asset Managers Net" data-tip-body="Net contracts held by Asset Managers — pension funds, mutual funds. Structural / longer-term positioning." data-tip-ex="Divergence between LF and AM often signals a positioning squeeze.">
            <div class="pd-inline-lbl">AM Net</div><div class="pd-inline-val ${clsI(cotAmNet)}">${fmtN(cotAmNet)}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="LF Net as % of Open Interest" data-tip-body="LF net divided by LF OI. Normalises positioning across currencies for direct comparison." data-tip-ex="+15% = LF hold net long equivalent to 15% of total OI — historically crowded.">
            <div class="pd-inline-lbl">Net % OI</div><div class="pd-inline-val ${clsI(cotPctOI)}">${cotPctOI != null ? (cotPctOI > 0 ? '+' : '') + cotPctOI.toFixed(1) + '%' : '—'}</div>
          </div>
        </div>
        <div style="margin-top:5px;font-size:9px;font-family:var(--font-mono);color:var(--text3);">${cotTag}</div>
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

// ── Sidebar crosses: click to open chart, double-click to open popover ──
document.getElementById('sidebar')?.addEventListener('click', e => {
  const row = e.target.closest('.sb-row[data-sym]');
  if (!row) return;
  loadTVChart(row.dataset.sym);
});

document.getElementById('sidebar')?.addEventListener('dblclick', e => {
  e.preventDefault();
  const row = e.target.closest('.sb-row[data-sym]');
  if (!row) return;
  openPairPopover(row, row.dataset.sym);
});

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
  loadTVChart('FXOPEN:VIX');
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
    let nzdProxy = false;
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

  // COT
  const cotCcy = base && base !== 'USD' ? base : (quote && quote !== 'USD' ? quote : base);
  const cotRaw = cotCcy ? (COT_DATA_CACHE[cotCcy] || null) : null;
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

  // Carry differential (CB rates)
  // For USD major pairs:
  //   invert:true  = CCY/USD pair (EUR/USD) → numerator = base (EUR) → carry = cbBase − cbQuote
  //   invert:false = USD/CCY pair (USD/JPY) → numerator = USD (quote) → carry = cbQuote − cbBase
  // For cross pairs (no invert field):
  //   The pair label is always BASE/QUOTE (e.g. AUD/CHF), so numerator = base
  //   carry = cbBase − cbQuote  (AUD rate − CHF rate = 4.10% − 0% = +4.10%)
  //   Using meta.cross to detect cross pairs and always apply cbBase − cbQuote.
  const cbBase  = base  ? (STATE.cbRates?.[base.toLowerCase()]?.rate  ?? null) : null;
  const cbQuote = quote ? (STATE.cbRates?.[quote.toLowerCase()]?.rate ?? null) : null;
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
  if (cotNet != null && cotAmNet != null) {
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
        <div class="pd-cell fx-tip" data-tip-title="Carry Differential" data-tip-body="Central bank policy rate differential: numerator currency rate minus denominator currency rate. Positive = pair&apos;s leading currency yields more, carry favours holding long." data-tip-ex="USD/JPY carry = Fed Funds (4.5%) − BoJ rate (0.5%) = +4.0%. Positive carry = long USD/JPY earns the differential."><div class="pd-lbl">Carry</div><div class="pd-val ${cls(carryDiff)}">${carryDiff != null ? (carryDiff >= 0 ? '+' : '') + carryDiff.toFixed(2)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Average Daily Range" data-tip-body="Estimated average daily range in pips, derived from HV 30d: close × (HV / √252). Indicates typical intraday movement — useful for stop and target sizing." data-tip-ex="ADR of 85 pip on EUR/USD means the pair moves ~85 pip on an average day."><div class="pd-lbl">ADR</div><div class="pd-val">${adr != null ? adr + ' pip' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="${base || 'Base'} Policy Rate" data-tip-body="${base || 'Base'} central bank policy rate (annualised). Source: CB rates cache."><div class="pd-lbl">${base || 'Base'} Rate</div><div class="pd-val">${cbBase != null ? cbBase.toFixed(2)+'%' : '—'}</div></div>
      </div>
    </div>

    <div class="pd-section">
      <div class="pd-section-lbl">Volatility</div>
      <div class="pd-grid">
        <div class="pd-cell fx-tip" data-tip-title="Historical Volatility 30d" data-tip-body="30-day realised (historical) volatility, annualised. Measures how much the pair has actually moved recently. Low HV = quiet market; high HV = volatile market."><div class="pd-lbl">HV 30d</div><div class="pd-val">${hv30 != null ? hv30.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="ATM Implied Volatility${(meta?.cross || nzdProxy) && atmIv != null ? ' (estimated)' : ''}" data-tip-body="${meta?.cross && atmIv != null ? 'Synthesised from component USD-pair ETF option IVs via triangulation: √(IVa²+IVb²−2ρ·IVa·IVb). Proxy for OTC interbank IV — indicative only.' : nzdProxy && atmIv != null ? 'Estimated from AUD/USD ETF IV × 1.08 (long-run NZD/AUD vol ratio). No CBOE-listed NZD ETF options available — treat as directional context only.' : 'ATM implied vol from nearest CBOE FX ETF option expiry (FXE/FXB/FXY/FXA via yfinance). Proxy for OTC interbank IV — ETF options may diverge 1–5 vol points from true OTC levels.'} Color = cost of hedging: green ≤7% (cheap), red >12% (expensive). Not a directional signal."><div class="pd-lbl">ATM IV${(meta?.cross || nzdProxy) && atmIv != null ? '<span style="font-size:8px;color:var(--text3);margin-left:2px;">~</span>' : ''}</div><div class="pd-val ${atmIv != null ? (atmIv > 12 ? 'pd-dn' : atmIv > 7 ? '' : 'pd-up') : ''}">${atmIv != null ? atmIv.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="IV minus HV" data-tip-body="Implied vol minus realised vol. Positive = options are expensive relative to recent moves (market pricing in risk premium). Negative = options are cheap vs realised. Not a directional signal." data-tip-ex="IV−HV > +3% historically indicates options are pricing in a premium above recent realised moves — hedging costs are elevated relative to actual market movement."><div class="pd-lbl">IV − HV</div><div class="pd-val ${atmIv != null && hv30 != null ? cls(atmIv - hv30) : ''}">${atmIv != null && hv30 != null ? (atmIv > hv30 ? '+' : '') + (atmIv - hv30).toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="25-delta Risk Reversal (1M) · Saxo Bank" data-tip-body="25d RR = 25d call IV minus 25d put IV. Positive = calls bid over puts — market skewed for upside on ${rrBase}. Negative = puts bid — downside protection dominant. Source: Saxo Bank public options page, 1M tenor, indicative mid-market. Updated during European hours." data-tip-ex="RR is a directional skew signal, not a vol-level signal. A strongly negative RR alongside high ATM IV = market pricing in both expensive hedging AND downside risk — historically a high-conviction bearish setup."><div class="pd-lbl">25d RR</div><div class="pd-val ${rrVal != null ? cls(rrVal) : ''}">${rrVal != null ? (rrVal >= 0 ? '+' : '') + rrVal.toFixed(2) : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Bid-Ask Spread" data-tip-body="Estimated interbank ECN spread in pips. Derived from live HV30 + VIX + MOVE model; falls back to ECN floor (IC Markets / Pepperstone Razor averages) when intraday data is unavailable. Lower spread = more liquid." data-tip-ex="EUR/USD typically trades 0.1–0.3 pip during London/NY overlap. Spreads widen significantly in Asian session and around news events."><div class="pd-lbl">Spread</div><div class="pd-val">${spreadPips != null ? spreadPips.toFixed(1) + ' pip' : '—'}</div></div>
      </div>
    </div>

    <div class="pd-section">
      <div class="pd-section-lbl">COT Positioning</div>
      <div class="pd-grid">
        ${(() => {
          // For crosses (EUR/GBP, GBP/JPY…) COT data tracks cotCcy vs USD — clarify in label and tooltip
          const isCross = !!meta?.cross;
          const ccyTag  = cotCcy ? ` (${cotCcy})` : '';
          const crossNote = isCross
            ? ` CFTC tracks ${cotCcy} futures vs USD — not this cross specifically. Use as a proxy for ${cotCcy} sentiment broadly.`
            : '';
          return `
            <div class="pd-cell fx-tip"
              data-tip-title="CFTC Leveraged Funds Net${ccyTag}"
              data-tip-body="Net contracts (longs minus shorts) held by Leveraged Funds — hedge funds and CTAs. Speculative / trend-following positioning. Source: CFTC Disaggregated TFF report.${crossNote}"
              data-tip-ex="Extreme LF net long positioning has historically preceded reversals as the speculative crowd becomes crowded.">
              <div class="pd-lbl">LF Net${ccyTag}</div>
              <div class="pd-val ${cls(cotNet)}">${fmtNet(cotNet)}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="LF WoW Change${ccyTag}"
              data-tip-body="Week-over-week change in Leveraged Funds net contracts. Positive = specs adding longs or covering shorts. Negative = specs adding shorts or reducing longs. The primary momentum signal in institutional COT analysis.${crossNote}"
              data-tip-ex="A large positive WoW change alongside rising net = conviction build-up. A reversal in WoW change is often the earliest signal of a positioning shift.">
              <div class="pd-lbl">LF WoW Δ${ccyTag}</div>
              <div class="pd-val ${cls(cotWow)}">${cotWow != null ? (cotWow > 0 ? '+' : '') + Math.round(cotWow).toLocaleString() : '—'}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="CFTC Asset Managers Net${ccyTag}"
              data-tip-body="Net contracts held by Asset Managers — pension funds, mutual funds, and institutional investors. Structural / longer-term positioning. Source: CFTC Disaggregated TFF report.${crossNote}"
              data-tip-ex="AM positioning tends to be more persistent than LF. Divergence between LF and AM can signal a positioning squeeze.">
              <div class="pd-lbl">AM Net${ccyTag}</div>
              <div class="pd-val ${cls(cotAmNet)}">${fmtNet(cotAmNet)}</div>
            </div>
            <div class="pd-cell fx-tip"
              data-tip-title="LF Net as % of Total OI${ccyTag}"
              data-tip-body="LF net contracts divided by LF Open Interest (long + short). Normalises positioning across currencies — EUR and JPY have very different raw contract counts; this makes them directly comparable.${crossNote}"
              data-tip-ex="+15% means Leveraged Funds hold a net long equivalent to 15% of the entire market's open interest — a heavily crowded position historically associated with reversal risk.">
              <div class="pd-lbl">Net % OI${ccyTag}</div>
              <div class="pd-val ${cls(cotPctOI)}">${cotPctOI != null ? (cotPctOI > 0 ? '+' : '') + cotPctOI.toFixed(1) + '%' : '—'}</div>
            </div>`;
        })()}
        ${cotOI != null ? (() => {
          const isCross   = !!meta?.cross;
          const ccyTag    = cotCcy ? ` (${cotCcy})` : '';
          const crossNote = isCross
            ? ` CFTC tracks ${cotCcy} futures vs USD — use as a proxy for ${cotCcy} market participation broadly.`
            : '';
          const oiDelta    = (cotPrevOI != null) ? cotOI - cotPrevOI : null;
          const oiArrow    = oiDelta == null ? '' : oiDelta > 0 ? '<span class="pd-oi-up">▲</span> ' : oiDelta < 0 ? '<span class="pd-oi-dn">▼</span> ' : '';
          const oiDeltaStr = oiDelta == null ? '' : ` <span class="pd-dim" style="font-size:9px;">(${oiDelta > 0 ? '+' : ''}${Math.round(oiDelta).toLocaleString()})</span>`;
          const oiTipEx    = oiDelta != null
            ? `This week: ${oiDelta > 0 ? '▲' : oiDelta < 0 ? '▼' : '='} ${Math.abs(Math.round(oiDelta)).toLocaleString()} vs prior week. ${oiDelta > 0 ? 'New money entering — expanding participation.' : 'Positions being closed — shrinking participation.'}`
            : 'Expanding OI alongside rising net long = conviction build-up. Falling OI alongside persistent net = position unwinding.';
          return `<div class="pd-cell pd-cell--wide fx-tip" style="border-bottom:none;"
            data-tip-title="LF Open Interest${ccyTag}"
            data-tip-body="Total open interest in the Leveraged Funds category: long + short contracts. Rising OI = new money entering; falling OI = positions closing. Source: CFTC TFF report.${crossNote}"
            data-tip-ex="${oiTipEx}">
            <div class="pd-lbl">LF Open Interest${ccyTag}</div>
            <div class="pd-val">${oiArrow}${Math.round(cotOI).toLocaleString()}${oiDeltaStr}</div>
          </div>`;
        })() : ''}
      </div>
      ${cotSummaryHtml}
    </div>

    <div class="pd-section pd-section--last">
      <div class="pd-section-lbl">Retail Sentiment</div>
      <div class="pd-cell pd-cell--wide fx-tip" data-tip-title="Retail Client Positioning" data-tip-body="Long/short ratio from Myfxbook community outlook — retail traders only, not institutional. Contrarian indicator: extreme retail long bias historically aligns with institutional short positioning. Source: Myfxbook, updated every 30min." data-tip-ex="When >70% of retail traders are long, institutional desks are often positioned short — price tends to move against the retail crowd over time.">
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
async function fetchCarryRanking() {
  const G8 = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD'];

  // TradingView symbol for a given long/short ccy pair
  function carryTV(long, short) {
    if (short === 'USD') return 'FX_IDC:' + long + 'USD';
    if (long  === 'USD') return 'FX_IDC:USD' + short;
    return 'FX_IDC:' + long + short;
  }

  // Canonical pair ID used in quotes.json / hv30 map (always alphabetical
  // except USD-majors which follow market convention — same as HV30_FX_PAIRS)
  function pairId(a, b) {
    const USD_MAJORS = {
      'EURUSD':1,'GBPUSD':1,'AUDUSD':1,'NZDUSD':1,
      'USDJPY':1,'USDCHF':1,'USDCAD':1,
    };
    const c1 = a + b, c2 = b + a;
    if (USD_MAJORS[c1]) return c1.toLowerCase();
    if (USD_MAJORS[c2]) return c2.toLowerCase();
    // Cross pairs: alphabetical
    return (a < b ? a + b : b + a).toLowerCase();
  }

  // Rate-cycle regime arrow for a currency, derived from STATE.cbRates trend
  // (same source as the CB Rates panel trend arrow — no double-computation)
  function regimeArrow(ccy) {
    const entry = STATE.cbRates?.[ccy.toLowerCase()];
    if (!entry) return '';
    const trend = entry.trend; // 'up' | 'down' | 'flat' | undefined
    if (trend === 'up')   return '<span class="cr-regime cr-hike">↑</span>';
    if (trend === 'down') return '<span class="cr-regime cr-cut">↓</span>';
    return '<span class="cr-regime cr-hold">→</span>';
  }

  const container = document.getElementById('carry-rank-rows');
  if (!container) return;

  try {
    // ── 1. CB rates ──────────────────────────────────────────────
    const rates = {};
    await Promise.all(G8.map(async ccy => {
      const cached = STATE.cbRates?.[ccy.toLowerCase()];
      if (cached?.rate != null) { rates[ccy] = cached.rate; return; }
      try {
        const r = await fetch('./rates/' + ccy + '.json');
        if (!r.ok) return;
        const d = await r.json();
        if (d.observations?.[0]?.value) rates[ccy] = parseFloat(d.observations[0].value);
      } catch {}
    }));

    if (Object.keys(rates).length < 4) {
      container.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Rate data unavailable</div>';
      return;
    }

    // ── 2. HV30 per pair from intraday cache ─────────────────────
    // quotes.json hv30 field is authoritative — same source used in FX table
    const intra = await loadIntradayQuotes().catch(() => null);
    const hv30Map = {};
    if (intra?.hv30) {
      Object.assign(hv30Map, intra.hv30);
    }
    // Also pull from STOOQ_RT_CACHE (populated by fetchQuoteBarRT) as supplement
    for (const [id, entry] of Object.entries(STOOQ_RT_CACHE)) {
      if (entry?.hv30 != null && hv30Map[id] == null) hv30Map[id] = entry.hv30;
    }

    // ── 3. Build all 28 G8 pairs ─────────────────────────────────
    const allPairs = [];
    for (let i = 0; i < G8.length; i++) {
      for (let j = i + 1; j < G8.length; j++) {
        const a = G8[i], b = G8[j];
        const rA = rates[a] ?? null, rB = rates[b] ?? null;
        if (rA == null || rB == null) continue;

        const diff = rA - rB;
        const long  = diff >= 0 ? a : b;
        const short = diff >= 0 ? b : a;
        const rLong  = diff >= 0 ? rA : rB;
        const rShort = diff >= 0 ? rB : rA;
        const absDiff = Math.abs(diff);

        const pid  = pairId(long, short);
        const hv30 = hv30Map[pid] ?? null;

        // Carry-to-vol: annualised rate diff / annualised HV30
        // Interpretation: units of carry earned per unit of realised vol risk
        // Primary ranking metric for carry panels
        const carryVol = (hv30 != null && hv30 > 0) ? absDiff / hv30 : null;

        allPairs.push({ long, short, diff: absDiff, rLong, rShort, hv30, carryVol, pid });
      }
    }

    // ── 4. Sort by carry-to-vol (primary); fall back to gross diff ─
    const hasVolData = allPairs.some(p => p.carryVol != null);
    allPairs.sort((a, b) => {
      if (hasVolData) {
        const cvA = a.carryVol ?? -Infinity;
        const cvB = b.carryVol ?? -Infinity;
        return cvB - cvA;
      }
      return b.diff - a.diff;
    });

    const top     = allPairs.slice(0, 10);
    const maxDiff = Math.max(...top.map(p => p.diff)) || 1;
    const maxCV   = hasVolData ? (Math.max(...top.map(p => p.carryVol ?? 0)) || 1) : 1;

    // Update panel subtitle to reflect sort method
    const headSpan = container.closest('.sb-section')
      ?.querySelector('.sb-head span');
    if (headSpan) {
      headSpan.textContent = hasVolData
        ? 'G8 · carry-to-vol ratio'
        : 'G8 · CB rate differential';
    }

    container.innerHTML = top.map((p, idx) => {
      const sym  = carryTV(p.long, p.short);
      // Bar width represents the vol-adjusted carry (or gross diff as fallback)
      const barVal = hasVolData && p.carryVol != null ? p.carryVol / maxCV : p.diff / maxDiff;
      // Bar width: percentage of the bar-wrap container (proportional to top pair)
      const barPct = Math.round(barVal * 100);

      // Color: strong carry-to-vol (>0.3) = green; moderate = neutral; weak = dim
      const cls = hasVolData && p.carryVol != null
        ? (p.carryVol > 0.30 ? 'pd-up' : p.carryVol > 0.12 ? '' : 'pd-dim')
        : (p.diff > 2 ? 'pd-up' : p.diff > 0.5 ? '' : 'pd-dim');

      const rArrowL = regimeArrow(p.long);
      const rArrowS = regimeArrow(p.short);

      // Tooltip: full institutional detail (rates, vol, carry-to-vol, regime)
      const cvStr  = p.carryVol != null ? p.carryVol.toFixed(2) : 'n/a';
      const hvStr  = p.hv30     != null ? p.hv30.toFixed(1) + '%' : 'n/a';
      const tip = `${p.long}/${p.short} · Long ${p.rLong.toFixed(2)}% / Short ${p.rShort.toFixed(2)}% · Diff ${p.diff.toFixed(2)}bp · HV30 ${hvStr} · Carry/Vol ${cvStr}`;

      // Display: show carry-to-vol when available, gross diff otherwise
      const displayVal = hasVolData && p.carryVol != null
        ? p.carryVol.toFixed(2)
        : '+' + p.diff.toFixed(2) + '%';

      // Spread label: show gross rate differential (e.g. +4.00%) — industry standard
      // for carry screens (Bloomberg/Refinitiv show rate spread, not CB trend arrows)
      const spreadLabel = (p.diff >= 0 ? '+' : '') + p.diff.toFixed(2) + '%';

      return `<div class="carry-rank-row" data-sym="${sym}" title="${tip}">
        <span class="cr-rank">${idx + 1}</span>
        <span class="cr-pair">${p.long}/${p.short}</span>
        <span class="cr-spread">${spreadLabel}</span>
        <div class="cr-bar-wrap"><div class="cr-bar" style="width:${barPct}%"></div></div>
        <span class="cr-diff ${cls}">${displayVal}</span>
      </div>`;
    }).join('');

    container.querySelectorAll('.carry-rank-row[data-sym]').forEach(row => {
      row.addEventListener('click', () => loadTVChart(row.dataset.sym));
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
    vEl.className = 'ca-val ' + cls;
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
    if (bEl)  { bEl.textContent  = btcFmtE; bEl.className  = 'ca-val '  + clsDir(_caBtcEarly.chg); }
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
    if (qbEl && qbEl.textContent === '—')  { qbEl.textContent  = btcFmtE; qbEl.className  = 'q-price ' + clsDir(_caBtcEarly.chg); }
    if (qbcEl && qbcEl.textContent === '—') { qbcEl.textContent = pctStr(_caBtcEarly.pct); qbcEl.className = 'q-chg ' + clsDir(_caBtcEarly.chg); }
  }

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

  if (finalSpx)    setCA('spx',    finalSpx.close,    finalSpx.pct,    false, finalSpx.chg);
  if (finalGold) {
    setCA('gold', finalGold.close, finalGold.pct, false, finalGold.chg);
    const gEl = document.getElementById('q-xauusd'), gcEl = document.getElementById('qc-xauusd');
    if (gEl)  { gEl.textContent  = finalGold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); gEl.className  = 'q-price ' + clsDir(finalGold.chg); }
    if (gcEl) { gcEl.textContent = pctStr(finalGold.pct); gcEl.className = 'q-chg ' + clsDir(finalGold.chg); }
  }
  if (finalWti)    setCA('wti',    finalWti.close,    finalWti.pct,    false, finalWti.chg);
  if (finalNikkei) setCA('nikkei', finalNikkei.close, finalNikkei.pct, false, finalNikkei.chg);
  if (finalStoxx)  setCA('stoxx',  finalStoxx.close,  finalStoxx.pct,  false, finalStoxx.chg);
  if (us10y)       setCA('us10y',  us10y.close, us10y.fromRepo ? null : us10y.pct, true);

  const dxyData = finalDxy;
  if (dxyData) {
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
  if (_btcIntraday && btcEl) {
    const btcFmt = _btcIntraday.close.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
    btcEl.textContent  = btcFmt;
    btcEl.className    = 'ca-val ' + clsDir(_btcIntraday.chg);
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
    // Also update topbar q-btcusd if still showing —
    if (qBtc && qBtc.textContent === '—') {
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
      const nextMtg  = meetings?.allMeetingsFormatted?.[0] || '—';

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
      // Priority 1: explicit fwdRate from meetings.json (OIS-derived — future use)
      // Priority 2: probability-weighted if cutProb or hikeProb available (≥1 field)
      // Priority 3: ±25bp naive step with ~ prefix (only when no probabilities at all)
      let fwdDisplay = '—';
      const meetingBias = (() => {
        if (!meetings) return null;
        const b = meetings.bias;
        if (!b) return null;
        if (/cut|dovish/i.test(b))  return 'down';
        if (/hike|hawkish/i.test(b)) return 'up';
        return 'flat';
      })();
      // Priority 1: explicit fwdRate from meetings.json (OIS-derived)
      if (meetings?.fwdRate != null && !isNaN(meetings.fwdRate) && meetings.fwdRate > 0) {
        fwdDisplay = meetings.fwdRate.toFixed(2) + '%';
      } else {
        const pCut  = (meetings?.cutProb  != null && !isNaN(meetings.cutProb))  ? Math.min(100, Math.max(0, meetings.cutProb))  : null;
        const pHike = (meetings?.hikeProb != null && !isNaN(meetings.hikeProb)) ? Math.min(100, Math.max(0, meetings.hikeProb)) : null;

        if (pCut !== null || pHike !== null) {
          // Priority 2: probability-weighted — Bloomberg WIRP standard
          const cut  = pCut  ?? 0;
          const hike = pHike ?? 0;
          // Clamp residual so probabilities never exceed 100%
          const cutC  = Math.min(cut,  100);
          const hikeC = Math.min(hike, 100 - cutC);
          const implied = current + (hikeC / 100) * 0.25 - (cutC / 100) * 0.25;
          fwdDisplay = Math.max(0, implied).toFixed(2) + '%';
        } else {
          // Priority 3: no probabilities available — naive ±25bp step, ~ signals estimate
          const dir  = meetingBias ?? trendDir;
          const step = dir === 'down' ? -0.25 : dir === 'up' ? 0.25 : 0;
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
    if (meetingsRes?.meetings) window._STATE_meetings = meetingsRes.meetings;
  } catch(e) { console.warn('CB expectations failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// POSITIONING BIAS — three data sources, rendered in priority order:
//
// SOURCE 1 — ETF IV (primary): real ATM implied vol from CBOE FX ETF options (quotes.json).
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
    const tbody = document.getElementById('skew-tbody');
    if (!tbody) return;

    const pairs = [
      { pair:'EUR/USD', cot:'EUR', etfId:'eurusd', rrKey:'EURUSD' },
      { pair:'GBP/USD', cot:'GBP', etfId:'gbpusd', rrKey:'GBPUSD' },
      { pair:'USD/JPY', cot:'JPY', etfId:'usdjpy', rrKey:'USDJPY' },
      { pair:'AUD/USD', cot:'AUD', etfId:'audusd', rrKey:'AUDUSD' },
      { pair:'USD/CAD', cot:'CAD', etfId:null,     rrKey:'USDCAD' },
      { pair:'USD/CHF', cot:'CHF', etfId:null,     rrKey:'USDCHF' },
      { pair:'NZD/USD', cot:'NZD', etfId:null,     rrKey:null     },
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
    const thead = tbody.closest('table')?.querySelector('thead tr');
    if (thead) {
      if (hasAnyEtfIv) {
        // IV Rank column shown when history is available (≥4 weeks)
        thead.innerHTML = hasIvRank
          ? '<th style="text-align:left" scope="col">Pair</th><th scope="col">ATM IV</th><th scope="col" title="IV Rank: position of current IV within 52-week range (0=historically low, 100=historically high)">IV Rnk</th><th scope="col">Direction</th>'
          : '<th style="text-align:left" scope="col">Pair</th><th scope="col">ATM IV</th><th scope="col">COT bias</th><th scope="col">Direction</th>';
      } else {
        thead.innerHTML = '<th style="text-align:left" scope="col">Pair</th><th scope="col">1W</th><th scope="col">1M</th><th scope="col">Bias</th>';
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

    tbody.innerHTML = pairs.map(p => {
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
        const td1Body  = `ATM IV ${ivStr} from CBOE FX ETF options (${etfIv.source || 'ETF'}) — nearest expiry ≥4 days. Proxy for OTC interbank implied vol. Green ≤7% (cheap vol); red >12% (expensive).`;
        const td1Ex    = 'ETF options are less liquid than OTC interbank FX options — ATM IV may diverge 1–5 vol points from true OTC levels.';
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
        const skew1m = cotData ? netToSkew(cotData.net * 0.85, invert) : 0;
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
        const td12Body    = 'COT-derived skew proxy — ETF IV unavailable for this pair. Derived from CFTC Leveraged Funds net positioning. 1W = current week net skew; 1M = smoothed (×0.85 decay).';
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
        panelHead.textContent = hasRR ? 'ETF IV · CBOE · 25d RR · Saxo' : 'ETF options IV · CBOE (proxy)';
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
        const signals = await sigR.json();
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
            sub.textContent = signals.length + ' active · AI-generated · loaded ' + hhmm + ' ' + tzAbbr;
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
    const SESSION_RATIO_JPY = { syd: 0.25, tok: 0.60, lon: 0.75, ny: 0.80 };

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
    if (sub) sub.textContent = `HV30 ${eur.hv30.toFixed(1)}% · session ratios BIS/Myfxbook`;

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

  // fetchQuoteBarRT popula STOOQ_RT_CACHE (precios RT + hv30).
  // Awaited so populateFxPairsTable finds the RT cache ready when it renders.
  await fetchQuoteBarRT();
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
  fetchCryptoQuotes();               // BTC from CoinGecko
  buildRichNarrative();              // AI narrative full build (non-blocking, fills narrative text)
  setTimeout(fetchSentiment, 800);   // Dukascopy sentiment (last, non-critical)
}

boot();

// Refresh quote bar FX every 60 seconds via intraday JSON / yfinance (~5 min delay)
setInterval(fetchQuoteBarRT, 60 * 1000);
// Refresh ECB rates every 30 minutes (FX table + heatmap + cross rows)
setInterval(fetchFrankfurter, 30 * 60 * 1000);
// Refresh news every 10 minutes
setInterval(fetchNewsData, 10 * 60 * 1000);  // every 10 min; ETag avoids re-download if unchanged
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
// Crypto: every 90 seconds
setInterval(fetchCryptoQuotes, 90 * 1000);
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
      // Only reload TV chart on mobile — desktop widgets stay alive across tab switches
      if (isMobile) {
        reloadActiveTVChart();
        // Stagger calendar reload to avoid TV scripts racing for the wrong container
        setTimeout(reloadTVCalendar, 800);
      }
    }, 350);
  });

  // On pageshow (iOS Safari fires this when returning from bfcache)
  window.addEventListener('pageshow', function(e) {
    if (!e.persisted) return; // only for bfcache restores
    // Reset scroll to top so page doesn't restore mid-page on mobile
    if (isMobile) {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }
    setTimeout(function() {
      redrawLiquidityIfVisible();
      // Mobile only: widgets may have gone blank after bfcache restore
      if (isMobile) {
        reloadActiveTVChart();
        // Stagger calendar reload to avoid TV scripts racing for the wrong container
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
  };

  function navTo(target) {
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
window._ADV_OPTS = ADV_ALERT_TYPES;

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
// UI STATE PERSISTENCE
// Saves and restores: last chart symbol + last visited section.
// Split layout state is handled separately by initSplitLayout().
// Storage key: 'gi_ui_state' → { sym, section }
// ═══════════════════════════════════════════════════════════════════
(function initUIStatePersistence() {
  'use strict';
  var STATE_KEY = 'gi_ui_state';
  var DEFAULT_SYM = 'FX_IDC:EURUSD';

  function loadState() {
    try { return JSON.parse(localStorage.getItem(STATE_KEY) || 'null'); } catch { return null; }
  }

  function saveState(patch) {
    try {
      var current = loadState() || {};
      localStorage.setItem(STATE_KEY, JSON.stringify(Object.assign(current, patch)));
    } catch (_) {}
  }

  // ── Restore last chart symbol ──────────────────────────────────────
  // Wrap loadTVChart to intercept every symbol change and persist it.
  var _origLoadTV = typeof loadTVChart === 'function' ? loadTVChart : null;
  if (_origLoadTV) {
    window.loadTVChart = function(sym) {
      _origLoadTV(sym);
      saveState({ sym: sym });
      window._tvCurrentSym = sym;
    };
  }

  // ── Restore last nav section ───────────────────────────────────────
  // Intercept top-nav link clicks to persist the target section.
  document.querySelectorAll('.top-nav a[data-target]').forEach(function(link) {
    link.addEventListener('click', function() {
      var t = this.dataset.target;
      if (t && t !== 'top') saveState({ section: t });
    });
  });

  // ── Apply saved state on load ──────────────────────────────────────
  // Defer until after boot() and all panels have initialized.
  window.addEventListener('load', function() {
    var state = loadState();
    if (!state) return;

    // Restore chart symbol (only if different from default to avoid redundant reload)
    var sym = (state.sym && state.sym !== DEFAULT_SYM) ? state.sym : null;
    if (sym && typeof window.loadTVChart === 'function') {
      // Small delay: let the initial boot chart render first, then replace
      setTimeout(function() { window.loadTVChart(sym); }, 800);
    }

    // Restore last section (scroll into view via nav link)
    if (state.section) {
      setTimeout(function() {
        var link = document.querySelector('.top-nav a[data-target="' + state.section + '"]');
        if (link) link.click();
      }, 1000);
    }
  });
}());

// ═══════════════════════════════════════════════════════════════════
// COT DIGEST — WEEKLY PUSH NOTIFICATION
// Reads cot-data/digest.json after the COT workflow runs (Fri ~21:30 UTC).
// Fires a push notification if:
//   1. Browser notifications are permitted
//   2. The digest weekEnding is newer than the last one seen by this user
// Storage key: 'gi_cot_notif_seen' → last weekEnding string sent
// ═══════════════════════════════════════════════════════════════════
(function initCOTDigestNotification() {
  'use strict';
  var SEEN_KEY = 'gi_cot_notif_seen';

  // Only run if SW is registered (required for showNotification)
  // and notifications are supported
  if (!('Notification' in window) || !('serviceWorker' in navigator)) return;

  // Don't run if permission was explicitly denied
  if (Notification.permission === 'denied') return;

  async function checkDigest() {
    // Fetch the digest — this is a same-origin static file, no CORS
    var res;
    try {
      res = await fetch('/cot-data/digest.json', { cache: 'no-cache' });
      if (!res.ok) return;
    } catch (_) { return; }

    var digest;
    try { digest = await res.json(); } catch (_) { return; }

    var weekEnding = digest.weekEnding;
    if (!weekEnding) return;

    // Already notified for this week?
    var lastSeen = '';
    try { lastSeen = localStorage.getItem(SEEN_KEY) || ''; } catch (_) {}
    if (lastSeen === weekEnding) return;

    // Only notify if permission is already granted.
    // We do NOT request permission here — that only happens via the onboarding
    // tooltip ("SET ALERT") so the user has explicitly opted in.
    if (Notification.permission !== 'granted') return;

    // Get the SW registration to call showNotification
    var reg;
    try { reg = await navigator.serviceWorker.ready; } catch (_) { return; }
    if (!reg) return;

    var message = digest.message ||
      'COT data updated for week ending ' + weekEnding;

    try {
      await reg.showNotification('COT Update — ' + weekEnding, {
        body: message,
        icon: '/favicon-192x192.png',
        badge: '/favicon-32x32.png',
        tag: 'gi-cot-' + weekEnding,   // prevents duplicate if tab reloads
        renotify: false,
        data: { url: 'https://globalinvesting.github.io/#section-positioning' },
      });
    } catch (_) { return; }

    // Mark this week as seen so we never re-fire for the same weekEnding
    try { localStorage.setItem(SEEN_KEY, weekEnding); } catch (_) {}
  }

  // Run after page load — no urgency, don't block anything
  if (document.readyState === 'complete') {
    setTimeout(checkDigest, 3000);
  } else {
    window.addEventListener('load', function() { setTimeout(checkDigest, 3000); });
  }
}());
