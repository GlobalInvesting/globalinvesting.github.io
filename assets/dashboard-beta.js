// Disable browser scroll-position restoration so our explicit scrollTop = 0 calls
// in boot() are never overridden by the browser restoring a previous scroll position.
// Must be set before any scroll resets run. Standard pattern for dashboard/SPA pages.
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

// ═══════════════════════════════════════════════════════════════════
// GI THEME MANAGER — moved from inline <script> in index.html (v8.41.0)
// Runs at the same point in document order as before (index.html's script tag
// sat right before </body>; dashboard.js loads via `defer`, which executes in
// document order right after parsing completes — same effective timing).
// ═══════════════════════════════════════════════════════════════════
(function () {
  const STORAGE_KEY = 'gi_theme';
  const THEMES = ['dark', 'mt5'];

  function apply(theme) {
    if (!THEMES.includes(theme)) theme = 'dark';
    const prev = saved;
    saved = theme;
    if (theme === 'dark') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
    // Update toggle button states
    THEMES.forEach(t => {
      const btn = document.getElementById('gi-theme-' + t);
      if (btn) btn.classList.toggle('active', t === theme);
    });
    try { localStorage.setItem(STORAGE_KEY, theme); } catch {}
    // Notify dashboard to re-apply theme-dependent colors (LWC charts, canvases)
    if (prev !== theme) {
      window.dispatchEvent(new CustomEvent('gi-theme-change', { detail: { theme, prev } }));
    }
  }

  // Apply saved theme immediately
  let saved = 'dark';
  try { saved = localStorage.getItem(STORAGE_KEY) || 'dark'; } catch {}
  apply(saved);

  window.GI_THEME = { set: apply, current: () => saved };
})();

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
  { id:'usdnok', base:'NOK', quote:'USD', invert:false, dec:4, label:'USD/NOK' },
  { id:'usdsek', base:'SEK', quote:'USD', invert:false, dec:4, label:'USD/SEK' },
  { id:'eurnok', base:'EUR', quote:'NOK', cross:['EUR','NOK'], dec:4, label:'EUR/NOK' },
  { id:'eursek', base:'EUR', quote:'SEK', cross:['EUR','SEK'], dec:4, label:'EUR/SEK' },
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

// ── FX Fair Value (v8.191.0, regression added v8.197.0, generalized to a
// 5-variable BEER model v8.200.0) ──────────────────────────────────────────
// Reads fair-value-data/{pair}.json (written daily by log_fair_value_inputs.py
// in globalinvesting-scripts — see log-fair-value-inputs.yml). Each file is an
// array of real {date, spot, rate_diff, stress, ca_diff, tb_diff} rows,
// oldest → newest, one per day the workflow has run. No client-side
// estimation of missing days: below FV_MIN_ROWS the panel shows an
// accumulation progress bar instead of a z-score, per Santiago's explicit
// "don't fabricate regression history" decision (2026-08-20) — see
// GUIDELINES.md § Data integrity.
//
// This file duplicates dashboard.js's PAIRS/renderFairValue/_fvRegress
// (same "no shared-module pattern in this repo" reasoning documented
// elsewhere — dashboard-beta.js is a full fork, not an include) — keep
// both in sync whenever one changes, per the standing pattern for this file.
const FV_MIN_ROWS = 60;
const FV_ROLLING_WINDOW = 60;

// Feature columns beyond the intercept — see dashboard.js's FV_FEATURE_KEYS
// for the full explanation of ca_diff/tb_diff (GDP-normalized Current
// Account / Trade Balance differentials, added v8.200.0).
const FV_FEATURE_KEYS = ['rate_diff', 'stress', 'ca_diff', 'tb_diff'];

// OLS regression: spot ~ intercept + Σ βᵢ·featureᵢ — see dashboard.js's
// _fvRegress()/_solveLinearSystem() for the full explanatory comment.
function _fvRegress(rows) {
  const usable = rows.filter(r => r && r.spot != null && FV_FEATURE_KEYS.every(k => r[k] != null));
  const k = FV_FEATURE_KEYS.length + 1; // +1 for the intercept
  if (usable.length < k * 2) return null;

  const n = usable.length;
  let Sxx = Array.from({ length: k }, () => new Array(k).fill(0));
  let Sxy = new Array(k).fill(0);
  usable.forEach(r => {
    const x = [1, ...FV_FEATURE_KEYS.map(key => r[key])];
    for (let i = 0; i < k; i++) {
      Sxy[i] += x[i] * r.spot;
      for (let j = 0; j < k; j++) Sxx[i][j] += x[i] * x[j];
    }
  });

  const beta = _solveLinearSystem(Sxx, Sxy);
  if (!beta) return null;

  const fitted = usable.map(r => beta[0] + FV_FEATURE_KEYS.reduce((s, key, i) => s + beta[i + 1] * r[key], 0));
  const residuals = usable.map((r, i) => r.spot - fitted[i]);
  const residMean = residuals.reduce((a, b) => a + b, 0) / n;
  const residVar = residuals.reduce((a, b) => a + (b - residMean) * (b - residMean), 0) / (n - k);
  const residStd = residVar > 0 ? Math.sqrt(residVar) : 0;

  return { beta, n, residStd, usable };
}

function _solveLinearSystem(A, b) {
  const k = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  const EPS = 1e-9;
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < EPS) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c < k + 1; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[k] / row[i]);
}

async function renderFairValue() {
  const accWrap = document.getElementById('fv-accumulating');
  const tblWrap = document.getElementById('fv-wrap');
  const tbody   = document.getElementById('fv-tbody');
  if (!accWrap || !tblWrap || !tbody) return;

  const results = await Promise.all(PAIRS.map(async p => {
    try {
      const r = await _fetchWithRetry('./fair-value-data/' + p.id + '.json');
      if (!r || !r.ok) return { pair: p, rows: [] };
      const rows = await r.json();
      return { pair: p, rows: Array.isArray(rows) ? rows : [] };
    } catch {
      return { pair: p, rows: [] };
    }
  }));

  const maxRows = results.reduce((m, x) => Math.max(m, x.rows.length), 0);

  if (maxRows < FV_MIN_ROWS) {
    accWrap.style.display = '';
    tblWrap.style.display = 'none';
    const pct = Math.min(100, Math.round((maxRows / FV_MIN_ROWS) * 100));
    setEl('fv-progress-text', `${maxRows}/${FV_MIN_ROWS}d`);
    const bar = document.getElementById('fv-progress-bar');
    if (bar) bar.style.width = pct + '%';
    return;
  }

  // Enough history exists for a real rolling regression (v8.197.0) — see
  // _fvRegress() above. Each pair independently gates on its OWN row count
  // and its OWN regression succeeding (not singular).
  accWrap.style.display = 'none';
  tblWrap.style.display = '';

  let html = '';
  results.forEach(({ pair, rows }) => {
    if (!rows.length) return;
    const last = rows[rows.length - 1];
    const spotTxt = last.spot != null ? last.spot.toFixed(pair.dec) : '—';
    const rdTxt   = last.rate_diff != null ? (last.rate_diff >= 0 ? '+' : '') + last.rate_diff.toFixed(2) : '—';
    const rdColor = last.rate_diff == null ? 'var(--text3)' : (last.rate_diff >= 0 ? 'var(--up)' : 'var(--down)');
    const stTxt   = last.stress != null ? last.stress.toFixed(0) : '—';

    let fvTxt = '—', zTxt = '—', zColor = 'var(--text3)';
    if (rows.length >= FV_MIN_ROWS) {
      const windowRows = rows.slice(-FV_ROLLING_WINDOW);
      const reg = _fvRegress(windowRows);
      const lastHasAllFeatures = FV_FEATURE_KEYS.every(k => last[k] != null);
      if (reg && reg.residStd > 0 && lastHasAllFeatures) {
        const fairValue = reg.beta[0] + FV_FEATURE_KEYS.reduce((s, k, i) => s + reg.beta[i + 1] * last[k], 0);
        const z = (last.spot - fairValue) / reg.residStd;
        fvTxt = fairValue.toFixed(pair.dec);
        zTxt = (z >= 0 ? '+' : '') + z.toFixed(2) + '\u03c3';
        zColor = Math.abs(z) < 1 ? 'var(--text3)' : (z > 0 ? 'var(--down)' : 'var(--up)');
      }
    }

    html += `<tr>
      <td>${pair.label || (pair.base + '/' + pair.quote)}</td>
      <td>${spotTxt}</td>
      <td style="color:${rdColor};">${rdTxt}</td>
      <td>${stTxt}</td>
      <td>${fvTxt}</td>
      <td style="color:${zColor};">${zTxt}</td>
      <td style="color:var(--text3);">${rows.length}d</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

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
  { id:'nok', file:'NOK', label:'NB (NO)' },
  { id:'sek', file:'SEK', label:'Riksbank (SE)' },
];

// COT currencies available
const COT_CURRENCIES = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD']; // NOK/SEK: not in CFTC TFF report (ICE futures, not CME)

// ═══════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════

// ── Theme color helpers — read resolved CSS variable values ─────────
// Used by LWC chart init so colors update when theme switches.
function _themeColor(cssVar) {
  return getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
}
function _themeColorAlpha(cssVar, alpha) {
  const hex = _themeColor(cssVar).replace('#', '');
  if (!hex || hex.length < 6) return `rgba(0,0,0,${alpha})`;
  const r = parseInt(hex.slice(0,2), 16);
  const g = parseInt(hex.slice(2,4), 16);
  const b = parseInt(hex.slice(4,6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function fmt(val, dec) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(dec);
}
function fmtDec(val, dec = 2) { return fmt(val, dec); }

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

// ── DST-aware session boundaries (v8.41.0) ──────────────────────────────────
// Previously hardcoded fixed UTC hours (e.g. London 8-17 UTC, New York 13-22 UTC).
// That is only correct for roughly half the year — London shifts GMT(+0)/BST(+1)
// and New York shifts EST(-5)/EDT(-4) across DST changes, so a fixed UTC boundary
// drifts 1 hour off the real local trading day for the other half of the year.
// Fix: define each session by its NOMINAL LOCAL hours + IANA timezone, and convert
// to today's UTC boundary dynamically — Intl.DateTimeFormat resolves each zone's
// current DST state automatically (no manual DST date-range table to maintain).
const SESSION_DEFS = [
  { id:'sydney',  zone:'Australia/Sydney', openLocal:8, closeLocal:17 },
  { id:'tokyo',   zone:'Asia/Tokyo',       openLocal:9, closeLocal:18 },
  { id:'london',  zone:'Europe/London',    openLocal:8, closeLocal:17 },
  { id:'newyork', zone:'America/New_York', openLocal:8, closeLocal:17 },
];

// Current UTC offset (whole hours) for an IANA zone, as of `now` — reflects
// that zone's DST state for today's date, not a fixed year-round assumption.
function getUTCOffsetHours(timeZone, now) {
  try {
    const parts = new Intl.DateTimeFormat('en-US', { timeZone, timeZoneName: 'shortOffset' }).formatToParts(now);
    const tzPart = parts.find(p => p.type === 'timeZoneName');
    const m = tzPart && tzPart.value.match(/GMT([+-]\d+)/);
    return m ? parseInt(m[1], 10) : 0;
  } catch { return 0; } // Unknown zone / Intl unsupported — falls back to UTC (no shift)
}

// Nominal local hour (0-23) in `timeZone` → equivalent UTC hour (0-23) for `now`'s date.
function localHourToUTC(timeZone, localHour, now) {
  const offset = getUTCOffsetHours(timeZone, now);
  return ((localHour - offset) % 24 + 24) % 24;
}

function updateSessions(h) {
  const now = new Date();
  const sessions = SESSION_DEFS.map(s => ({
    id: s.id,
    open: localHourToUTC(s.zone, s.openLocal, now),
    close: localHourToUTC(s.zone, s.closeLocal, now),
  }));

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
  setEl('session-status', isWeekend ? activeLabel : (activeLabel + ' · ACTIVE'));
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
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf','eurnok','eursek'];
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
  usdnok:2.0, usdsek:2.0,
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
  // If the Matrix tab is active, recompute it for the new window too
  // (shared window selector — see initCorrAssetTabs()).
  if (window._corrActiveView === 'matrix') renderCorrMatrix();
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
        if (absZ >= 2.5)      { badgeCls = 'down'; badgeLabel = '● break'; }
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

// ── Cross-Asset Correlations panel: Cross Asset / Matrix tabs ──
// Matrix tab: G10 currency correlation grid, computed client-side from the
// same 32-pair ohlc-data/*.json set the heatmap composite (populateHeatmap())
// already documents as "G10 composite · 32 pairs" — reused here as the pair
// list for building a synthetic per-currency return series, since no
// pre-computed currency-index time series exists in any data/*.json output.
// Order follows the BIS 2022 Triennial Central Bank Survey's per-currency
// turnover ranking (net-net, % of total turnover, each side of a trade
// counted): USD 194 > EUR 183 > JPY 168 > GBP 142 > AUD 125 > CAD 122 >
// CHF 114 > SEK 66 > NOK 54 > NZD 31 (non-G10 currencies in the same
// ranking — CNY/HKD/SGD/KRW — excluded, this app's G10 set only). This is
// the same convention Bloomberg/Refinitiv desk screens use for G10 currency
// ordering — corrected from the prior placeholder order (which had GBP
// ahead of JPY and NZD ahead of SEK/NOK, both backwards vs turnover) after
// Santiago asked whether the panel matched industry convention.
const CORR_MTX_CCYS = ['USD','EUR','JPY','GBP','AUD','CAD','CHF','SEK','NOK','NZD'];
// [ohlcId, base, quote] — same 32 pairs as the G10 composite heatmap.
const CORR_MTX_PAIRS = [
  ['eurusd','EUR','USD'], ['gbpusd','GBP','USD'], ['usdjpy','USD','JPY'], ['audusd','AUD','USD'],
  ['usdcad','USD','CAD'], ['usdchf','USD','CHF'], ['nzdusd','NZD','USD'], ['usdnok','USD','NOK'],
  ['usdsek','USD','SEK'], ['eurgbp','EUR','GBP'], ['eurjpy','EUR','JPY'], ['euraud','EUR','AUD'],
  ['eurcad','EUR','CAD'], ['eurchf','EUR','CHF'], ['eurnzd','EUR','NZD'], ['eurnok','EUR','NOK'],
  ['eursek','EUR','SEK'], ['gbpjpy','GBP','JPY'], ['gbpaud','GBP','AUD'], ['gbpcad','GBP','CAD'],
  ['gbpchf','GBP','CHF'], ['gbpnzd','GBP','NZD'], ['audjpy','AUD','JPY'], ['audcad','AUD','CAD'],
  ['audchf','AUD','CHF'], ['audnzd','AUD','NZD'], ['cadchf','CAD','CHF'], ['cadjpy','CAD','JPY'],
  ['chfjpy','CHF','JPY'], ['nzdcad','NZD','CAD'], ['nzdchf','NZD','CHF'], ['nzdjpy','NZD','JPY']
];

let _corrMtxPairCloses = null; // { pairId: [close, ...] } — raw D1 closes, most-recent-last
let _corrMtxLoadPromise = null;

// Fetches all 32 pair files (or 32 h1/h4 files) in parallel via Promise.all.
// A single transient failure mid-batch (GitHub Pages/browser connection-limit
// hiccup on 32 simultaneous requests — the actual cause behind CHF/JPY
// showing blank in the Hourly tab on a live run, confirmed: the file itself
// was never missing or short server-side, live-refetched 12,081 clean bars
// the same minute) silently drops that one pair with no retry anywhere in
// the loop, unlike every other data path in this app (fetch_intraday_quotes.py
// / fetch_ohlc.py, etc.) which already retries transient fetch failures.
// This closes that gap client-side too.
async function _fetchWithRetry(url, attempts = 3, delayMs = 400) {
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return r;
      lastErr = new Error('HTTP ' + r.status);
    } catch (e) {
      lastErr = e;
    }
    if (i < attempts - 1) await new Promise(res => setTimeout(res, delayMs * (i + 1)));
  }
  throw lastErr;
}

async function _corrMtxLoadPairData() {
  if (_corrMtxPairCloses) return _corrMtxPairCloses;
  if (_corrMtxLoadPromise) return _corrMtxLoadPromise;
  _corrMtxLoadPromise = (async () => {
    const out = {};
    await Promise.all(CORR_MTX_PAIRS.map(async ([id]) => {
      try {
        const r = await _fetchWithRetry('./ohlc-data/' + id + '.json');
        const bars = await r.json();
        if (Array.isArray(bars) && bars.length > 1) {
          out[id] = bars.map(b => b.close).filter(c => typeof c === 'number');
        }
      } catch (e) { /* pair unavailable after retries — matrix cells using it stay blank */ }
    }));
    _corrMtxPairCloses = out;
    return out;
  })();
  return _corrMtxLoadPromise;
}

function _pearsonCorr(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 5) return null;
  a = a.slice(-n); b = b.slice(-n);
  const ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { const xa = a[i] - ma, xb = b[i] - mb; num += xa * xb; da += xa * xa; db += xb * xb; }
  if (da === 0 || db === 0) return null;
  return num / Math.sqrt(da * db);
}

// Builds a composite daily log-return series per G10 currency: every pair
// containing that currency contributes its log-return (sign-flipped when the
// currency is the quote leg), averaged across all contributing pairs per day.
function _corrMtxBuildCcyReturns(pairCloses, windowDays) {
  const ccyRets = {}; // ccy -> array of arrays (one per contributing pair)
  CORR_MTX_CCYS.forEach(c => ccyRets[c] = []);
  CORR_MTX_PAIRS.forEach(([id, base, quote]) => {
    const closes = pairCloses[id];
    if (!closes || closes.length < windowDays + 2) return;
    const slice = closes.slice(-(windowDays + 1));
    const rets = [];
    for (let i = 1; i < slice.length; i++) rets.push(Math.log(slice[i] / slice[i - 1]));
    ccyRets[base].push(rets);
    ccyRets[quote].push(rets.map(v => -v));
  });
  const composite = {};
  CORR_MTX_CCYS.forEach(c => {
    const series = ccyRets[c];
    if (!series.length) { composite[c] = null; return; }
    const len = Math.min(...series.map(s => s.length));
    const avg = [];
    for (let i = 0; i < len; i++) {
      let sum = 0;
      series.forEach(s => sum += s[s.length - len + i]);
      avg.push(sum / series.length);
    }
    composite[c] = avg;
  });
  return composite;
}

function _corrMtxCellStyle(v) {
  if (v == null) return { bg: 'var(--bg2)', color: 'var(--text3)', txt: '—' };
  const a = Math.min(Math.abs(v), 1);
  const txt = (v * 100).toFixed(0);
  if (v >= 0) return { bg: `rgba(38,166,154,${(a * 0.35).toFixed(2)})`, color: 'var(--up)', txt };
  return { bg: `rgba(239,83,80,${(a * 0.35).toFixed(2)})`, color: 'var(--down)', txt };
}

async function renderCorrMatrix() {
  const table = document.getElementById('corr-matrix-table');
  if (!table) return;
  table.innerHTML = '<tr><td style="color:var(--text3);font-size:9px;padding:6px 2px;">Loading…</td></tr>';
  const pairCloses = await _corrMtxLoadPairData();
  const composite = _corrMtxBuildCcyReturns(pairCloses, _corrWindow);

  // NOTE: header/row-label <th> cells must carry the same explicit
  // background+border as the value <td> cells below (var(--bg2)/var(--border)
  // instead of "unset"). Without it, the browser's UA default table-cell
  // border/background shows through on hover repaint — the gray square
  // Santiago flagged to the left of "USD" was exactly this: the corner <td>
  // and the row-label <th> were the only two cells in the table with no
  // background/border declared at all.
  let html = '<tr><td style="background:var(--bg2);border:1px solid var(--border);"></td>' + CORR_MTX_CCYS.map(c =>
    `<th scope="col" style="font-size:8.5px;font-family:var(--font-mono);color:var(--text3);font-weight:400;text-align:center;padding:0 0 3px;background:var(--bg2);border:1px solid var(--border);">${c}</th>`
  ).join('') + '</tr>';

  CORR_MTX_CCYS.forEach(rowCcy => {
    html += `<tr><th scope="row" style="font-size:8.5px;font-family:var(--font-mono);color:var(--text3);font-weight:600;text-align:left;padding:0 4px 0 0;background:var(--bg2);border:1px solid var(--border);">${rowCcy}</th>`;
    CORR_MTX_CCYS.forEach(colCcy => {
      if (rowCcy === colCcy) {
        html += `<td style="height:20px;text-align:center;border:1px solid var(--border);background:var(--bg2);color:var(--text3);font-size:8.5px;font-family:var(--font-mono);">—</td>`;
        return;
      }
      const a = composite[rowCcy], b = composite[colCcy];
      const v = (a && b) ? _pearsonCorr(a, b) : null;
      const s = _corrMtxCellStyle(v);
      html += `<td title="${rowCcy}/${colCcy} · ${_corrWindow}d Pearson${v == null ? ' (insufficient data)' : ': ' + (v >= 0 ? '+' : '') + v.toFixed(2)}" style="height:20px;text-align:center;border:1px solid var(--border);background:${s.bg};color:${s.color};font-size:8.5px;font-family:var(--font-mono);">${s.txt}</td>`;
    });
    html += '</tr>';
  });
  table.innerHTML = html;
}

// Row/column highlight on hover for the docked currency×currency Matrix —
// hovering a data cell highlights BOTH its row header and its column
// header (not just the single cell, which already had a title tooltip but
// no visual link back to which two currencies it represents). Delegated on
// the table element itself, wired once at init — safe across re-renders
// since renderCorrMatrix() only replaces the table's innerHTML, never the
// table node the listener is attached to. Header cells here carry their
// background/color as inline styles (not a CSS class), so the highlight is
// applied/reverted the same way — via inline style, cached per-cell on
// first hover — rather than a CSS class, which a same-specificity inline
// style would otherwise silently outrank.
function _corrMtxWireHover() {
  const table = document.getElementById('corr-matrix-table');
  if (!table || table.dataset.hoverWired) return;
  table.dataset.hoverWired = '1';
  const applyHl = (td, on) => {
    if (!td || td.tagName !== 'TD') return;
    const tr = td.parentElement;
    if (!tr || tr.rowIndex === 0) return; // header row has no data cells to react to
    const rowTh = tr.cells[0];
    const colTh = table.rows[0]?.cells[td.cellIndex];
    [rowTh, colTh].forEach(th => {
      if (!th || th.tagName !== 'TH') return;
      if (on) {
        if (th.dataset._hlBg === undefined) { th.dataset._hlBg = th.style.background; th.dataset._hlFg = th.style.color; }
        th.style.background = 'var(--bg3)';
        th.style.color = '#fff';
      } else if (th.dataset._hlBg !== undefined) {
        th.style.background = th.dataset._hlBg;
        th.style.color = th.dataset._hlFg;
      }
    });
  };
  table.addEventListener('mouseover', e => applyHl(e.target.closest('td'), true));
  table.addEventListener('mouseout', e => applyHl(e.target.closest('td'), false));
}

function initCorrAssetTabs() {
  const tabBar = document.getElementById('corr-asset-tabs');
  if (!tabBar) return;
  window._corrActiveView = 'cross';
  tabBar.addEventListener('click', e => {
    const btn = e.target.closest('.corr-view-tab');
    if (!btn) return;
    const view = btn.dataset.view;
    if (view === window._corrActiveView) return;
    window._corrActiveView = view;

    // Same active-state convention as the 30d/60d/90d window buttons right
    // below this bar: background/border stay fixed, only text color swaps
    // (text3 -> #fff on active) — not the larger rates-ctab pill treatment.
    tabBar.querySelectorAll('.corr-view-tab').forEach(b => {
      const isActive = b === btn;
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      b.style.color = isActive ? '#fff' : 'var(--text3)';
      b.classList.toggle('active', isActive);
    });

    const crossWrap = document.getElementById('corr-cross-wrap');
    const matrixWrap = document.getElementById('corr-matrix-wrap');
    if (view === 'matrix') {
      if (crossWrap) crossWrap.style.display = 'none';
      if (matrixWrap) matrixWrap.style.display = '';
      renderCorrMatrix();
    } else {
      if (crossWrap) crossWrap.style.display = '';
      if (matrixWrap) matrixWrap.style.display = 'none';
    }
  });
}

// ── Pair×Pair Correlation Matrix fullscreen — content-swap, not a DOM-lift.
// Santiago's spec: the docked "Matrix" tab (#corr-matrix-wrap, currency x
// currency, _corrMtxCcys) stays exactly as-is — the sidebar panel is too
// small to fit a 32x32 pairs grid. The expand button instead opens a
// fullscreen overlay that builds an independent Par×Par matrix (every
// tracked FX pair vs every other, raw Pearson on log-returns, not the
// per-currency composite the docked Matrix tab uses) with a Daily/4h/Hourly
// timeframe selector.
//
// Data: Daily reuses the existing ohlc-data/{pair}.json D1 closes
// (_corrMtxLoadPairData()'s cache). 4h/Hourly read ohlc-data/h4/{pair}.json
// and ohlc-data/h1/{pair}.json — already written every run by fetch_ohlc.py's
// build_intraday_ohlc() for all 32 pairs (verified this session — no new
// fetcher needed, correcting the prior session's assumption that intraday
// granularity was unavailable). 15min/5min are NOT available anywhere in
// the pipeline. v8.189.0: dropped the matching caveat from the on-screen
// footnote at Santiago's request (unnecessary
// detail for the end user) — the gap itself is unchanged, just no longer
// called out in the UI; this comment is the only place it's noted now.
//
// Lookback window: a fixed ~60-period-equivalent per timeframe (60 daily
// closes / 360 4h bars / 1440 hourly bars — roughly 60 trading days at each
// granularity's typical bar density), independent of the docked panel's
// 30d/60d/90d toggle (hidden behind the fullscreen overlay, not reachable
// while it's open). Not a Santiago-specified number — flagged to him as an
// assumption, adjustable later if he wants a different default or a
// selector of its own.
const CORR_PAIRS_TF_CONFIG = {
  daily: { dir: '',   bars: 60   },
  '4h':  { dir: 'h4/', bars: 360  },
  '1h':  { dir: 'h1/', bars: 1440 },
};

let _corrPairsActiveTf = 'daily';
const _corrPairsCloseCache = {}; // tf -> { pairId: [close, ...] } | Promise

async function _corrPairsLoadCloses(tf) {
  if (_corrPairsCloseCache[tf] && !(_corrPairsCloseCache[tf] instanceof Promise)) {
    return _corrPairsCloseCache[tf];
  }
  if (_corrPairsCloseCache[tf] instanceof Promise) return _corrPairsCloseCache[tf];

  const dir = CORR_PAIRS_TF_CONFIG[tf].dir;
  const promise = (async () => {
    const out = {};
    await Promise.all(CORR_MTX_PAIRS.map(async ([id]) => {
      try {
        const r = await _fetchWithRetry('./ohlc-data/' + dir + id + '.json');
        const bars = await r.json();
        if (Array.isArray(bars) && bars.length > 1) {
          out[id] = bars.map(b => b.close).filter(c => typeof c === 'number');
        }
      } catch (e) { /* pair unavailable after retries at this timeframe — matrix cells using it stay blank */ }
    }));
    _corrPairsCloseCache[tf] = out;
    return out;
  })();
  _corrPairsCloseCache[tf] = promise;
  return promise;
}

// Last-`nBars` log-return series from a closes array, oldest→newest.
// Returns null if there isn't even enough data for _pearsonCorr's own
// 5-point floor once trimmed to nBars.
function _corrPairsLogReturns(closes, nBars) {
  if (!closes || closes.length < 6) return null;
  const slice = closes.slice(-(nBars + 1));
  if (slice.length < 6) return null;
  const rets = [];
  for (let i = 1; i < slice.length; i++) rets.push(Math.log(slice[i] / slice[i - 1]));
  return rets;
}

// Builds a full pairwise Pearson map ({id: {otherId: corr|null}}) from a
// {id: returns[]} dict — the shared input both _pairsClusterOrder() and the
// table renderer below need, computed once per render instead of twice.
function _pairsCorrMap(ids, retsById) {
  const map = {};
  ids.forEach(id => { map[id] = {}; });
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = retsById[ids[i]], b = retsById[ids[j]];
      const v = (a && b) ? _pearsonCorr(a, b) : null;
      map[ids[i]][ids[j]] = v;
      map[ids[j]][ids[i]] = v;
    }
  }
  return map;
}

// 1-D spectral ordering via the leading eigenvector of the pairwise
// correlation matrix (power iteration), not a nearest-neighbor chain or a
// plain average-correlation sort. This is what actually reproduces the
// target visual pattern (screenshot comparison against a competitor tool,
// Santiago, this session): pairs that share the grid's dominant common
// factor cluster at the two extremes — strongly loading one way at one
// edge, strongly loading the opposite way at the other edge — while pairs
// with near-zero loading (genuinely uncorrelated with that dominant factor)
// settle in the middle.
//
// Two earlier approaches in this same session both fell short:
// 1. Greedy nearest-neighbor chaining (original v8.185.0 version) started
//    from the single lowest-average-correlation pair and forced it to an
//    *edge* as the chain's starting point — backwards from the target
//    layout, which puts weakly-correlated pairs in the middle.
// 2. A plain sort by each pair's average signed correlation (tried next)
//    looked right in isolation but breaks on two anti-correlated clusters
//    of similar size: by symmetry, a member of cluster A and a member of
//    cluster B end up with near-identical average scores (both dragged
//    negative by their cross-cluster correlations), so the two clusters
//    interleave instead of separating to opposite edges — verified with a
//    synthetic two-cluster-plus-neutrals case before shipping this version.
//
// The leading eigenvector doesn't have this blind spot: it's the direction
// that captures the matrix's single largest source of shared variance, so
// cluster A and cluster B naturally land with opposite-signed loadings
// (same magnitude, opposite sign) rather than similar magnitudes with the
// same sign. Power iteration (starting from a uniform vector, ~40
// iterations — this is a small ~32x32 matrix, no numerical stability
// concerns at that size) is the standard, cheap way to get this without a
// full eigendecomposition. A pair with no valid correlation at all (fetch
// failure at this timeframe) is treated as 0 (no relationship measured),
// which lands it near the middle — reasonable given there's no data to
// place it anywhere else. Verified against a synthetic two-cluster case
// (two 3-pair groups, strongly anti-correlated with each other, plus 2
// neutral pairs) — correctly separates both clusters to opposite edges
// with the neutrals in between.
function _pairsClusterOrder(ids, corrMap) {
  const n = ids.length;
  if (n <= 2) return ids.slice();
  const M = ids.map(r => ids.map(c => (r === c) ? 0 : (corrMap[r][c] ?? 0)));
  let v = new Array(n).fill(1 / Math.sqrt(n));
  for (let iter = 0; iter < 40; iter++) {
    const w = M.map(row => row.reduce((s, val, j) => s + val * v[j], 0));
    const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1;
    v = w.map(x => x / norm);
  }
  return ids.map((id, i) => [id, v[i]]).sort((a, b) => a[1] - b[1]).map(p => p[0]);
}

async function renderCorrPairsMatrix(tf) {
  const inner = document.getElementById('corr-mtx-fullscreen-inner');
  if (!inner) return;
  inner.innerHTML = '<div style="color:var(--text3);font-size:11px;padding:8px 2px;">Loading…</div>';

  const closes = await _corrPairsLoadCloses(tf);
  const cfg = CORR_PAIRS_TF_CONFIG[tf];
  const rets = {};
  CORR_MTX_PAIRS.forEach(([id]) => {
    rets[id] = _corrPairsLogReturns(closes[id], cfg.bars);
  });

  const ids = CORR_MTX_PAIRS.map(([id]) => id);
  const lblById = {};
  CORR_MTX_PAIRS.forEach(([id, base, quote]) => { lblById[id] = base + quote; });
  const corrMap = _pairsCorrMap(ids, rets);
  const orderedIds = _pairsClusterOrder(ids, corrMap);

  // Column headers stay horizontal — a vertical/rotated-text version was
  // tried and reverted (Santiago flagged a sticky-positioning regression it
  // introduced; see GUIDELINES.md v8.186.0). The table itself is full-width
  // (CSS table-layout:fixed) so all 32 columns fit without horizontal
  // scroll on a normal desktop viewport regardless of header orientation.
  // Header/row-label cells are plain <th> — position:sticky is applied
  // directly to them (see index.html CSS). The earlier "sticky on an inner
  // <div>" wrapper (v8.186.0) is gone; it wasn't the actual fix for the
  // "row labels float near the top on scroll" bug (root cause was the
  // scroll container's own padding — see the CSS comment above
  // #corr-pairs-fs-table), so the extra div was unnecessary complexity.
  // The whole table is wrapped in #corr-pairs-fs-wrap, a non-scrolling div
  // that carries the visual padding — see that same CSS comment for why it
  // has to live there and not on the scroll container itself.
  let html = '<div id="corr-pairs-fs-wrap"><table id="corr-pairs-fs-table" aria-label="Pair correlation matrix, clustered by correlation"><thead><tr><th></th>' +
    orderedIds.map(id => `<th scope="col">${lblById[id]}</th>`).join('') + '</tr></thead><tbody>';

  orderedIds.forEach(rowId => {
    html += `<tr><th scope="row">${lblById[rowId]}</th>`;
    orderedIds.forEach(colId => {
      if (rowId === colId) {
        html += `<td style="background:var(--bg2);color:var(--text3);">—</td>`;
        return;
      }
      const v = corrMap[rowId][colId];
      const s = _corrMtxCellStyle(v);
      html += `<td title="${lblById[rowId]}/${lblById[colId]}${v == null ? ' (insufficient data)' : ': ' + (v >= 0 ? '+' : '') + v.toFixed(2)}" style="background:${s.bg};color:${s.color};">${s.txt}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>' +
    `<div style="padding:8px 0 0;font-size:9px;color:var(--text3);">Pairwise Pearson · log-returns, last ${cfg.bars} ${tf === 'daily' ? 'daily closes' : tf + ' bars'} · rows/columns ordered by spectral seriation — pairs sorted by their loading on the correlation matrix's leading eigenvector, so the most strongly correlated pairs (positive or negative) cluster at the two edges and weakly-correlated pairs sit in the middle — not alphabetical</div></div>`;

  inner.innerHTML = html;
}

// Row/column highlight on hover for the fullscreen Pairs matrix — same
// affordance as _corrMtxWireHover() above for the docked Matrix tab.
// Delegated on #corr-mtx-fullscreen-inner (the stable container div, wired
// once at init) rather than on #corr-pairs-fs-table itself, because
// renderCorrPairsMatrix() rebuilds the whole <table> node — including its
// id — on every render/timeframe switch, which would silently detach a
// listener bound directly to the table. Uses classList (not inline style
// like the docked version) since these header cells' styling is entirely
// CSS-class/selector driven, not inline — see th.corr-hl>div in index.html.
function _corrPairsWireHover() {
  const inner = document.getElementById('corr-mtx-fullscreen-inner');
  if (!inner || inner.dataset.hoverWired) return;
  inner.dataset.hoverWired = '1';
  const applyHl = (td, on) => {
    if (!td || td.tagName !== 'TD') return;
    const table = td.closest('#corr-pairs-fs-table');
    if (!table) return;
    const tr = td.parentElement;
    if (!tr || tr.rowIndex === 0) return;
    const rowTh = tr.cells[0];
    const colTh = table.rows[0]?.cells[td.cellIndex];
    [rowTh, colTh].forEach(th => {
      if (th && th.tagName === 'TH') th.classList.toggle('corr-hl', on);
    });
  };
  inner.addEventListener('mouseover', e => applyHl(e.target.closest('td'), true));
  inner.addEventListener('mouseout', e => applyHl(e.target.closest('td'), false));
}

function _corrPairsSetTf(tf) {
  if (tf === _corrPairsActiveTf) return;
  _corrPairsActiveTf = tf;
  ['daily', '4h', '1h'].forEach(t => {
    const btn = document.getElementById('corr-pairs-tf-' + t);
    if (!btn) return;
    const active = t === tf;
    btn.classList.toggle('intel-fs-tab-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  renderCorrPairsMatrix(tf);
}

function openCorrMtxFullscreen() {
  const overlay = document.getElementById('corr-mtx-fullscreen-overlay');
  if (!overlay) return;
  if (overlay.classList.contains('corr-mtx-fs-active')) return;

  overlay.classList.add('corr-mtx-fs-active');
  document.body.style.overflow = 'hidden';
  renderCorrPairsMatrix(_corrPairsActiveTf);
}

function closeCorrMtxFullscreen() {
  const overlay = document.getElementById('corr-mtx-fullscreen-overlay');
  if (!overlay || !overlay.classList.contains('corr-mtx-fs-active')) return;

  overlay.classList.remove('corr-mtx-fs-active');
  document.body.style.overflow = '';
}

function _corrMtxFsWireUp() {
  _corrPairsWireHover();
  document.getElementById('corr-mtx-fs-btn')?.addEventListener('click', openCorrMtxFullscreen);
  document.getElementById('corr-mtx-fs-close')?.addEventListener('click', closeCorrMtxFullscreen);
  ['daily', '4h', '1h'].forEach(t => {
    document.getElementById('corr-pairs-tf-' + t)?.addEventListener('click', () => _corrPairsSetTf(t));
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && document.getElementById('corr-mtx-fullscreen-overlay')?.classList.contains('corr-mtx-fs-active')) {
      closeCorrMtxFullscreen();
    }
  });
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
  const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD','NOK','SEK'];

  // Prefer STOOQ_RT_CACHE (intraday ~5min delay) over ECB daily rates
  // because ECB daily rates have zero intraday movement (same open/close on weekends)
  const rtAvailable = Object.keys(STOOQ_RT_CACHE).length >= 21; // need ≥75% of 32 pairs (G8 + 4 Scandi) for reliable composite

  // pairDefs hoisted to function scope — used both inside the rtAvailable branch
  // (strength computation) and outside it (pairCountByCcy tooltip counts).
  // Declaring inside the if-block caused ReferenceError when rtAvailable=false.
  //
  // sign is always +1: log(close/prevClose) of any base/quote pair already
  // represents the base currency's return, regardless of which currency is
  // base. v8.28.4: removed sign:-1 from usdjpy/usdchf/usdcad/usdnok/usdsek —
  // that inversion was the root cause of the composite/CSI divergence between
  // the web terminal and the EA (EA's CSI_Score() has never special-cased
  // USD-base pairs: `sum += is_base ? ret : -ret`). Do not reintroduce it.
  const pairDefs = [
      // 7 USD majors
      { id: 'eurusd', base: 'EUR', quote: 'USD', sign: 1 },
      { id: 'gbpusd', base: 'GBP', quote: 'USD', sign: 1 },
      { id: 'audusd', base: 'AUD', quote: 'USD', sign: 1 },
      { id: 'nzdusd', base: 'NZD', quote: 'USD', sign: 1 },
      { id: 'usdjpy', base: 'USD', quote: 'JPY', sign: 1 },
      { id: 'usdchf', base: 'USD', quote: 'CHF', sign: 1 },
      { id: 'usdcad', base: 'USD', quote: 'CAD', sign: 1 },
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
      // G10 Scandinavian — 4 live pairs
      { id: 'usdnok', base: 'USD', quote: 'NOK', sign:  1 },
      { id: 'usdsek', base: 'USD', quote: 'SEK', sign:  1 },
      { id: 'eurnok', base: 'EUR', quote: 'NOK', sign:  1 },
      { id: 'eursek', base: 'EUR', quote: 'SEK', sign:  1 },
    ];

  let strengths;
  if (rtAvailable) {
    // Map each currency to its avg % change across all 28 G8 pairs.
    // Each currency appears in exactly 7 pairs — equal statistical weight.
    const pctMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0, NOK: 0, SEK: 0 };
    const countMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0, NOK: 0, SEK: 0 };

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
    const _nonUsd = strengths.filter(s=>s.ccy!=='USD');
    const _usdS = strengths.find(s=>s.ccy==='USD');
    if (_usdS) _usdS.pct = -_nonUsd.reduce((a,b)=>a+b.pct,0)/_nonUsd.length;
  }

  strengths.sort((a,b)=>b.pct-a.pct);

  const grid = document.getElementById('heatmap-grid');
  if (!grid) return;
  // Store strengths in a module-level variable so the modal can read them
  // without embedding JSON in an HTML attribute (which breaks on double-quotes).
  window._hmStrengths = strengths;
  // Whether this pass used the live 32-pair composite (rtAvailable) or the
  // cruder ECB-daily-rates fallback (v8.131.0) — exposed so consumers like
  // gi-overview.js can wait for the real composite instead of rendering
  // the fallback estimate and then visibly jumping to a different number a
  // few seconds later once enough Finnhub ticks arrive.
  window._hmStrengthsLive = rtAvailable;
  // Per-currency direct-pair count, structural (independent of live data
  // availability) — matches heatmap-modal.js's `PAIR_DEFS.filter(p => p.base
  // === ccy || p.quote === ccy).length` exactly, so the tooltip never drifts
  // out of sync with what the modal actually shows. Was hardcoded "7" before
  // — wrong for EUR/USD (9 pairs each) and NOK/SEK (2 pairs each, structurally
  // asymmetric vs the rest of G10).
  const pairCountByCcy = {};
  ccys.forEach(c => {
    pairCountByCcy[c] = pairDefs.filter(p => p.base === c || p.quote === c).length;
  });
  grid.innerHTML = strengths.map(s => {
    let bg = 'h-flat';
    if (s.pct > 0.15) bg = 'h-s-up';
    else if (s.pct > 0.05) bg = 'h-up';
    else if (s.pct < -0.15) bg = 'h-s-down';
    else if (s.pct < -0.05) bg = 'h-down';
    const cls = s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat';
    const sign = s.pct >= 0 ? '+' : '';
    const nPairs = pairCountByCcy[s.ccy] || 0;
    return `<div class="hm-cell ${bg}" role="button" tabindex="0" aria-label="${s.ccy} currency strength ${sign}${s.pct.toFixed(2)}%" style="cursor:pointer" title="Click to open ${s.ccy} breakdown · ${nPairs} direct pair${nPairs===1?'':'s'} · COT · vol · correlations" onclick="if(window.openHeatmapModal)openHeatmapModal('${s.ccy}',window._hmStrengths,STOOQ_RT_CACHE)">
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
      ? 'Live \u00b7 G10 composite \u00b7 32 pairs'
      : 'Delayed ~5min \u00b7 G10 composite \u00b7 32 pairs';
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
      nok: { flag: 'no', name: 'Norges Bank',              short: 'NB'   },
      sek: { flag: 'se', name: 'Sveriges Riksbank',        short: 'Riksbank' },
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
// This matches the panel data source: CFTC TFF (Traders in Financial Futures) · Leveraged Funds · Options+Futures Combined
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

// Report-family label + abbreviations, derived from the record's own
// assetClass/positionCategory fields rather than a literal string — v8.161.0.
// FX/Indices come from CFTC's TFF report (primary signal: Leveraged Funds,
// secondary: Asset Manager). Commodities come from the Disaggregated report,
// a different report family whose primary signal is Managed Money (the
// hedge-fund/CTA analog of Leveraged Funds) and secondary is Swap Dealers —
// labeling commodity rows "LF"/"AM"/"TFF" would misstate the actual source.
function _cotReportMeta(rec) {
  if (rec && rec.assetClass === 'commodity') {
    return {
      report: 'Disaggregated',
      primaryLabel: 'Managed Money', primaryAbbr: 'MM',
      secondaryLabel: 'Swap Dealers', secondaryAbbr: 'SD',
      // v8.161.1 — tertiary (dealer/hedger) slot, added so cot-modal-chart.js
      // can derive its "Dealers" row/legend labels from this same helper
      // instead of hardcoding "LF"/"AM"/"Leveraged Funds" (see that file's
      // v2.7 header note and CHANGELOG v8.161.1).
      tertiaryLabel: 'Producer/Merchant', tertiaryAbbr: 'PM',
    };
  }
  return {
    report: 'TFF',
    primaryLabel: 'Leveraged Funds', primaryAbbr: 'LF',
    secondaryLabel: 'Asset Manager', secondaryAbbr: 'AM',
    tertiaryLabel: 'Dealers', tertiaryAbbr: 'DD',
  };
}

// Builds the "CFTC · week ending … · updated … · loaded … · Nd lag" label
// used by the FX, Indices, and Commodities COT tabs.
function _buildCOTUpdateLabel(latest) {
  const meta = _cotReportMeta(latest);
  const weekEnd = latest.weekEnding || latest.reportDate || '';
  let updLabel = meta.report + ' · ' + meta.primaryLabel + ' · week ending ' + weekEnd;
  if (latest.lastUpdate) {
    try {
      const d = new Date(latest.lastUpdate);
      if (!isNaN(d)) {
        updLabel += ' · updated ' + d.toLocaleDateString('en', { weekday: 'short', day: '2-digit', month: 'short' });
      }
    } catch {}
  }
  const _cotNow = new Date();
  const _cotHHMM = _cotNow.getHours().toString().padStart(2,'0') + ':' + _cotNow.getMinutes().toString().padStart(2,'0');
  const _cotTZ = _cotNow.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  updLabel += ' · loaded ' + _cotHHMM + ' ' + _cotTZ;

  let lagHtml = '';
  if (weekEnd) {
    try {
      const msPerDay = 86400000;
      const lagDays  = Math.floor((_cotNow.getTime() - new Date(weekEnd + 'T00:00:00Z').getTime()) / msPerDay);
      if (lagDays >= 0) {
        const lagColor = lagDays <= 7 ? 'var(--up)' : lagDays <= 14 ? '#c8952a' : 'var(--down)';
        const lagDot   = lagDays <= 7 ? '●' : lagDays <= 14 ? '◐' : '○';
        lagHtml = ` · <span title="Days since week-ending date · CFTC publishes Fri, terminal updates Sat" style="color:${lagColor};font-variant-numeric:tabular-nums;">${lagDot} ${lagDays}d lag</span>`;
      }
    } catch {}
  }
  return updLabel + lagHtml;
}

// Renders COT rows (FX currencies or equity indices) into #cot-rows and wires
// row-click → modal. Shared by fetchCOTData() (FX tab) and fetchCOTIndicesData()
// (Indices tab) so both tabs use identical row markup/behavior.
function _renderCOTRows(results, symMap, dataStoreKey) {
  const container = document.getElementById('cot-rows');
  if (!container) return;

  // Sort rows by Long% descending — industry standard for COT panels
  results.sort((a, b) => {
    const totalA = (a.longPositions || 0) + (a.shortPositions || 0);
    const totalB = (b.longPositions || 0) + (b.shortPositions || 0);
    const pctA = totalA > 0 ? (a.longPositions || 0) / totalA : 0.5;
    const pctB = totalB > 0 ? (b.longPositions || 0) / totalB : 0.5;
    return pctB - pctA;
  });

  // Expose full COT data for the modal chart
  window[dataStoreKey] = window[dataStoreKey] || {};
  results.forEach(d => { window[dataStoreKey][d.ccy] = d; });

  container.innerHTML = results.map(d => {
    const meta = _cotReportMeta(d);
    const net   = d.netPosition || 0;
    const long  = d.longPositions || 0;
    const short = d.shortPositions || 0;
    const total = long + short;
    const longPct = total > 0 ? Math.round(long / total * 100) : 50;
    const cls   = net > 0 ? 'up' : net < 0 ? 'down' : 'flat';
    const netStr = (net >= 0 ? '+' : '') + net.toLocaleString();

    // Primary vs secondary category divergence dot — filled = aligned, hollow = diverge
    const amNet = d.assetManagerNet;
    let divHtml = '';
    if (amNet != null) {
      const lfDir = net > 0 ? 1 : net < 0 ? -1 : 0;
      const amDir = amNet > 0 ? 1 : amNet < 0 ? -1 : 0;
      if (lfDir !== 0 && amDir !== 0) {
        if (lfDir === amDir) {
          divHtml = '<span class="cot-div aligned" title="' + meta.primaryAbbr + ' + ' + meta.secondaryAbbr + ' aligned — ' + (lfDir > 0 ? 'both net long' : 'both net short') + '">●</span>';
        } else {
          divHtml = '<span class="cot-div diverge" title="' + meta.primaryAbbr + '/' + meta.secondaryAbbr + ' diverge — ' + meta.primaryAbbr + ' ' + (net > 0 ? 'long' : 'short') + ' · ' + meta.secondaryAbbr + ' ' + (amNet > 0 ? 'long' : 'short') + '">○</span>';
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
      wowHtml = '<span class="cot-wow ' + wowCls + '" title="Week-over-week change in ' + meta.primaryAbbr + ' net contracts. Positive = specs adding longs/covering shorts. Negative = specs adding shorts/reducing longs.">' + wowStr + '</span>';
    }

    // Net as % of primary-category OI — read from root if present, else derive from current long+short.
    let pctOI = d.levNetPctOI ?? null;
    if (pctOI == null && oi > 0) {
      pctOI = Math.round(net / oi * 1000) / 10; // one decimal
    }
    let pctOIHtml  = '<span class="cot-pcoi">—</span>';
    if (pctOI != null) {
      const pctCls = pctOI > 0 ? 'up' : pctOI < 0 ? 'down' : 'flat';
      const pctStr = (pctOI > 0 ? '+' : '') + pctOI.toFixed(1) + '%';
      pctOIHtml = '<span class="cot-pcoi ' + pctCls + '" title="' + meta.primaryAbbr + ' net as % of ' + meta.primaryAbbr + ' Open Interest. Normalised across ' + (d.assetClass === 'commodity' ? 'commodities' : 'currencies') + ' — comparable regardless of contract size differences.">' + pctStr + '</span>';
    }

    // TradingView COT chart symbol for row click
    const tvSym = (symMap && symMap[d.ccy]) || '';

    return '<div class="cot-row" style="cursor:pointer;" data-sym="' + tvSym + '" data-ccy="' + d.ccy + '" title="Click to open ' + d.ccy + ' COT positioning detail">'
      + '<span class="cot-sym">' + d.ccy + '</span>'
      + '<div class="cot-bar-outer">'
      + '<div class="cot-long-fill" style="width:' + longPct + '%"></div>'
      + '<div class="cot-short-fill" style="width:' + (100 - longPct) + '%"></div>'
      + '</div>'
      + '<span class="cot-pct ' + cls + '">' + longPct + '%</span>'
      + '<span class="cot-net ' + cls + '" title="' + meta.primaryAbbr + ' net contracts (longs minus shorts). Positive = net long speculative positioning; negative = net short. Primary directional signal from CFTC ' + meta.report + ' report.">' + netStr + '</span>'
      + wowHtml
      + divHtml
      + pctOIHtml
      + '<span class="cot-oi" title="' + meta.primaryAbbr + ' Open Interest: ' + oi.toLocaleString() + ' contracts (long + short). Rising OI signals new money; falling OI signals liquidation.">' + oiArrow + oiStr + '</span>'
      + '</div>';
  }).join('');

  // Click any COT row → open institutional modal chart (fallback: TradingView widget)
  container.querySelectorAll('.cot-row[data-sym]').forEach(row => {
    row.addEventListener('click', () => {
      const ccy  = row.dataset.ccy;
      const data = window[dataStoreKey] && window[dataStoreKey][ccy];
      if (ccy && data && typeof window.openCOTModal === 'function') {
        window.openCOTModal(ccy, data);
      } else {
        const sym = row.dataset.sym;
        if (sym) loadCOTChart(sym);
      }
    });
  });
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

  // Cache raw results so the asset-class tab switcher can restore this tab
  // instantly without re-fetching when the user flips back from Indices.
  window._cotFXResults = results;

  const subEl = document.getElementById('cot-date-sub');
  if (subEl && (!window._cotActiveAsset || window._cotActiveAsset === 'fx')) {
    subEl.innerHTML = _buildCOTUpdateLabel(results[0]);
  }

  if (!window._cotActiveAsset || window._cotActiveAsset === 'fx') {
    _renderCOTRows(results, COT_TV_SYMBOLS, 'COT_DATA_STORE');
  }
}

// Equity-index COT data — cot-data/indices/{SPX,NAS100,DJ30}.json, written by
// update-cot-cftc-all.yml's Indices loop (v8.160.0). Lazy-fetched on first
// click of the "Indices" tab; cached in window._cotIndicesResults afterwards.
const COT_INDICES = ['SPX', 'NAS100', 'DJ30'];

async function fetchCOTIndicesData() {
  const container = document.getElementById('cot-rows');
  const subEl = document.getElementById('cot-date-sub');
  if (container) container.innerHTML = '<div class="cot-row" style="color:var(--text3);"><span>Loading index positioning data…</span></div>';

  const promises = COT_INDICES.map(async sym => {
    try {
      const r = await fetch('./cot-data/indices/' + sym + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      return { ccy: sym, ...data };
    } catch { return null; }
  });

  const results = (await Promise.all(promises)).filter(Boolean);
  window._cotIndicesResults = results;

  if (!results.length) {
    if (container) container.innerHTML = '<div class="cot-row" style="color:var(--text3);"><span>Index positioning data not yet published — pending next CFTC TFF run.</span></div>';
    if (subEl) subEl.innerHTML = 'TFF · Leveraged Funds · S&amp;P 500 / Nasdaq-100 / DJIA futures';
    return;
  }

  if (subEl) subEl.innerHTML = _buildCOTUpdateLabel(results[0]);
  _renderCOTRows(results, {}, 'COT_DATA_STORE_INDICES');
}

// Commodity COT data — cot-data/commodities/{XAU,XAG,COPPER,WTI}.json, written
// by update-cot-cftc-all.yml's Disaggregated-report leg (v8.161.0). Deliberately
// scoped to FX-relevant commodities: Gold/Silver (safe-haven, USD/JPY proxy),
// Copper ("Dr. Copper", AUD/China-demand proxy), WTI Crude Oil (CAD/NOK
// petrocurrency correlation). Lazy-fetched on first click of the "Commodities"
// tab; cached in window._cotCommoditiesResults afterwards. Same non-fatal
// "not yet published" pattern as the Indices tab if the workflow hasn't run.
const COT_COMMODITIES = ['XAU', 'XAG', 'COPPER', 'WTI'];

async function fetchCOTCommoditiesData() {
  const container = document.getElementById('cot-rows');
  const subEl = document.getElementById('cot-date-sub');
  if (container) container.innerHTML = '<div class="cot-row" style="color:var(--text3);"><span>Loading commodity positioning data…</span></div>';

  const promises = COT_COMMODITIES.map(async sym => {
    try {
      const r = await fetch('./cot-data/commodities/' + sym + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      return { ccy: sym, ...data };
    } catch { return null; }
  });

  const results = (await Promise.all(promises)).filter(Boolean);
  window._cotCommoditiesResults = results;

  if (!results.length) {
    if (container) container.innerHTML = '<div class="cot-row" style="color:var(--text3);"><span>Commodity positioning data not yet published — pending next CFTC Disaggregated run.</span></div>';
    if (subEl) subEl.innerHTML = 'Disaggregated · Managed Money · Gold / Silver / Copper / WTI futures';
    return;
  }

  if (subEl) subEl.innerHTML = _buildCOTUpdateLabel(results[0]);
  _renderCOTRows(results, {}, 'COT_DATA_STORE_COMMODITIES');
}

// ── COT panel asset-class tabs (FX / Indices / Commodities) ──
function initCOTAssetTabs() {
  const tabBar = document.getElementById('cot-asset-tabs');
  if (!tabBar) return;
  window._cotActiveAsset = 'fx';
  tabBar.addEventListener('click', e => {
    const btn = e.target.closest('.rates-ctab');
    if (!btn) return;
    const asset = btn.dataset.asset;
    if (asset === window._cotActiveAsset) return;
    window._cotActiveAsset = asset;

    tabBar.querySelectorAll('.rates-ctab').forEach(b => {
      const isActive = b === btn;
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      b.style.background = isActive ? 'var(--accent)' : 'none';
      b.style.color = isActive ? '#fff' : 'var(--text2)';
      b.style.border = isActive ? 'none' : '1px solid var(--border2)';
      b.style.fontWeight = isActive ? '600' : '400';
    });

    if (asset === 'fx') {
      const subEl = document.getElementById('cot-date-sub');
      if (window._cotFXResults && window._cotFXResults.length) {
        if (subEl) subEl.innerHTML = _buildCOTUpdateLabel(window._cotFXResults[0]);
        _renderCOTRows(window._cotFXResults.slice(), COT_TV_SYMBOLS, 'COT_DATA_STORE');
      } else {
        fetchCOTData();
      }
    } else if (asset === 'indices') {
      if (window._cotIndicesResults) {
        const subEl = document.getElementById('cot-date-sub');
        if (window._cotIndicesResults.length) {
          if (subEl) subEl.innerHTML = _buildCOTUpdateLabel(window._cotIndicesResults[0]);
          _renderCOTRows(window._cotIndicesResults.slice(), {}, 'COT_DATA_STORE_INDICES');
        } else {
          fetchCOTIndicesData();
        }
      } else {
        fetchCOTIndicesData();
      }
    } else if (asset === 'commodities') {
      if (window._cotCommoditiesResults) {
        const subEl = document.getElementById('cot-date-sub');
        if (window._cotCommoditiesResults.length) {
          if (subEl) subEl.innerHTML = _buildCOTUpdateLabel(window._cotCommoditiesResults[0]);
          _renderCOTRows(window._cotCommoditiesResults.slice(), {}, 'COT_DATA_STORE_COMMODITIES');
        } else {
          fetchCOTCommoditiesData();
        }
      } else {
        fetchCOTCommoditiesData();
      }
    }
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
  { sym: 'usdnok',  id: 'usdnok',  dec: 4 },
  { sym: 'usdsek',  id: 'usdsek',  dec: 4 },
  { sym: 'eurnok',  id: 'eurnok',  dec: 4 },
  { sym: 'eursek',  id: 'eursek',  dec: 4 },
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
    const srcLabel = 'Delayed ~5min';  // sole source
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
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf','eurnok','eursek'];
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
      ? `Last close: Fri · delayed`
      : _hasFinnhub
        ? `Live`
        : `${hh}:${mm} ${tzAbbr} · delayed ~5min`;
  }

  // Update Price Chart panel-sub label to match active source
  const _chartSub = document.querySelector('#section-fxpairs .panel-sub');
  if (_chartSub && _chartSub.textContent !== 'TradingView \u00b7 live data') {
    const _hasFh = Object.values(STOOQ_RT_CACHE).some(e => e?.fromFinnhub);
    _chartSub.textContent = _hasFh ? 'Live' : `Delayed ~5min`;
  }

  // ── Pair detail live refresh ───────────────────────────────────────────────
  // Re-render the pair detail popover (if open) with the latest cache values.
  // Throttled to once per 3 s — Finnhub fires 5–10 ticks/s and updatePairDetail()
  // does a full innerHTML rebuild including an async IV fetch.
  _throttledPairDetailRefresh();
}

// Throttle state for pair detail live updates
let _pairDetailRefreshTimer = null;
function _throttledPairDetailRefresh() {
  if (_pairDetailRefreshTimer) return;
  _pairDetailRefreshTimer = setTimeout(() => {
    _pairDetailRefreshTimer = null;
    const pop = document.getElementById('pd-popover');
    if (!pop || pop.style.display === 'none') return;
    const sym = pop.dataset.sym;
    if (sym && typeof updatePairDetail === 'function') updatePairDetail(sym);
  }, 3000);
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
// Retail FX Positioning: metals rows (myfxbook sym 'XAU/USD'/'XAG/USD') must map
// to the OANDA: TradingView symbols that _TV_TO_OHLC recognises as 'gold'/'silver',
// not the generic FX_IDC: prefix used for currency pairs — see loadTVChart() call
// site below in renderSentiment() for the incident this fixes.
const RETAIL_SENT_METAL_TV_SYM = { 'XAU/USD': 'OANDA:XAUUSD', 'XAG/USD': 'OANDA:XAGUSD' };
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
    <div style="display:grid;grid-template-columns:58px 1fr 38px 38px 12px 52px;align-items:center;gap:0;padding:3px 8px 3px;background:var(--head-bg);border-bottom:1px solid var(--border2);position:sticky;top:0;z-index:5;">
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
    // Metals rows need the OANDA: prefix to match _TV_TO_OHLC's 'gold'/'silver'
    // entries and load the Gold/Silver Futures LW chart. The generic FX_IDC:
    // prefix used for currency pairs below has no _TV_TO_OHLC entry for XAUUSD/
    // XAGUSD, so it was silently falling through to the TradingView widget
    // fallback for every Retail FX Positioning click on a metals row.
    const tvSym = RETAIL_SENT_METAL_TV_SYM[p.sym] || ('FX_IDC:' + p.sym.replace('/', ''));

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
      // No z-index here (was z-index:2) — this element's 12px height with
      // top:-3px on a 6px bar (overflow:visible) deliberately pokes 3px
      // above its own row's bar as a "current price" wick. That's fine in
      // isolation, but z-index:2 put it above the sticky header's z-index:1
      // in the stacking order, so as a row scrolled to sit just beneath the
      // sticky header, its tick's overflow rendered ON TOP of the header's
      // solid background instead of being covered by it — a stray white
      // line floating above the table. DOM append order already places
      // this element after the two colored fill divs in the same barDiv,
      // which is enough to paint it above them without an explicit
      // z-index; removing it lets the sticky header's z-index (bumped
      // below) win as intended.
      tickEl.style.cssText = `position:absolute;top:-3px;width:2px;height:12px;background:#fff;opacity:.9;border-radius:1px;left:${tickPct}%;transform:translateX(-1px);cursor:help;`;
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
  const isAltFeed = sourceLabel && sourceLabel.includes('Retail \u00b7 live');
  const subEl = document.getElementById('sent-source-sub');
  if (subEl) {
    const _sentTime = lh + ':' + lm + ' ' + tzAbbr2;
    if (isCOT) subEl.textContent = `CFTC COT · speculative positioning · loaded ${_sentTime}`;
    else if (isHistorical) subEl.textContent = `Static fallback · live feeds unavailable · loaded ${_sentTime}`;
    else if (isAltFeed) subEl.textContent = `Retail positioning · live · updated ${_sentTime}`;
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
          assetClass: p.assetClass || 'fx',
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
        _setSentimentSource(pairs, 'Myfxbook · ' + ageLabel, general);
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
          assetClass: 'fx',
        })).filter(d=>d.sym);
        // Dukascopy doesn't publish metals sentiment — carry over any Metals
        // rows already cached from Myfxbook so switching tabs doesn't blank out.
        const cachedMetals = (window._sentAllPairs || []).filter(p => p.assetClass === 'metal');
        if (mapped.length) { _setSentimentSource(mapped.concat(cachedMetals), 'Retail \u00b7 live'); return; }
      }
    }
  } catch {}

  // ── SOURCE 3: Static reference fallback ──
  // COT data is intentionally excluded from this fallback pipeline:
  // it belongs to its own dedicated section in the terminal and has
  // different semantics (speculative positioning, weekly) vs retail sentiment.
  const fallbackPairs = SENTIMENT_FALLBACK.map(p => ({ ...p, assetClass: 'fx' }));
  _setSentimentSource(fallbackPairs, 'Static fallback · live feeds unavailable');
}

// Caches the full (FX+Metals) pair set from whichever source just resolved,
// then renders whichever asset class is currently active in #sent-asset-tabs.
function _setSentimentSource(pairs, sourceLabel, general) {
  window._sentAllPairs    = pairs;
  window._sentSourceLabel = sourceLabel;
  window._sentGeneral     = general || null;
  _renderSentimentForActiveTab();
}

// data-asset button values vs. the assetClass strings the fetchers actually
// write are not 1:1 (button says "metals" for UI-label pluralization,
// fetch_myfxbook_sentiment.py v8.160.0 writes the singular "metal" per pair)
// — v8.160.4 incident: comparing them directly with === silently matched
// zero rows every time, so the Metals tab always rendered as "unavailable"
// even though the data was present. Normalize through this map instead of
// relying on the button/data string matching the JSON field verbatim.
const SENT_ASSET_CLASS_MAP = { fx: 'fx', metals: 'metal' };

function _renderSentimentForActiveTab() {
  const pairs = window._sentAllPairs || [];
  const asset = window._sentActiveAsset || 'fx';
  const targetClass = SENT_ASSET_CLASS_MAP[asset] || asset;
  const filtered = pairs.filter(p => (p.assetClass || 'fx') === targetClass);
  if (asset !== 'fx' && !filtered.length) {
    const container = document.getElementById('sent-rows');
    if (container) container.innerHTML = '<div style="color:var(--text3);font-size:11px;padding:8px 0;">Metals sentiment unavailable from the current source.</div>';
    return;
  }
  renderSentiment(filtered, window._sentSourceLabel, window._sentGeneral);
}

// ── Retail sentiment panel asset-class tabs (FX / Metals) ──
function initSentimentAssetTabs() {
  const tabBar = document.getElementById('sent-asset-tabs');
  if (!tabBar) return;
  window._sentActiveAsset = 'fx';
  tabBar.addEventListener('click', e => {
    const btn = e.target.closest('.rates-ctab');
    if (!btn) return;
    const asset = btn.dataset.asset;
    if (asset === window._sentActiveAsset) return;
    window._sentActiveAsset = asset;

    tabBar.querySelectorAll('.rates-ctab').forEach(b => {
      const isActive = b === btn;
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      b.style.background = isActive ? 'var(--accent)' : 'none';
      b.style.color = isActive ? '#fff' : 'var(--text2)';
      b.style.border = isActive ? 'none' : '1px solid var(--border2)';
      b.style.fontWeight = isActive ? '600' : '400';
    });

    _renderSentimentForActiveTab();
  });
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

  // ── US HY OAS ────────────────────────────────────────────────────
  const hyOasCell = document.querySelector('#section-risk .risk-cell:nth-child(3)');
  if (hyOasCell) attachRiskTip(hyOasCell,
    'US HY OAS — ICE BofA High Yield Option-Adjusted Spread',
    'The extra yield below-investment-grade (BB+/Ba1 and lower) US corporate bonds pay over Treasuries, adjusted for embedded call options. The standard institutional gauge of corporate credit-market stress — a distinct risk-off channel from equity vol (VIX) or bond vol (MOVE), and one that has historically led both into genuine credit events.',
    'Below ~350bp is the historically tight zone seen in calm, risk-on markets. Sustained widening past ~500bp has coincided with credit-stress episodes (2015 energy HY, March 2020). Level alone can lag — pair with the HY OAS 20d Δ row below to catch the direction of travel.'
  );

  // ── US IG OAS ────────────────────────────────────────────────────
  const igOasCell = document.querySelector('#section-risk .risk-cell:nth-child(4)');
  if (igOasCell) attachRiskTip(igOasCell,
    'US IG OAS — ICE BofA Investment Grade Option-Adjusted Spread',
    'The extra yield investment-grade (BBB-/Baa3 and higher) US corporate bonds pay over Treasuries, adjusted for embedded call options. The companion series to HY OAS — moves in IG confirm whether credit stress is broad-based or confined to junk-rated issuers.',
    'Below ~100bp is a historically tight reading (bottom-decile territory in the post-2009 era). Widening past ~150bp signals credit conditions tightening even for higher-quality issuers, typically alongside a rising HY-IG differential.'
  );

  // ── EUR/USD HV 30d ───────────────────────────────────────────────
  const hvCell = document.querySelector('#section-risk .risk-cell:nth-child(5)');
  if (hvCell) attachRiskTip(hvCell,
    'EUR/USD Historical Volatility (30d)',
    'Realized volatility of EUR/USD over the past 30 days, calculated from daily log returns. Not a forecast — it measures what actually happened. Useful for sizing positions.',
    'If HV 30d is 8% and your stop is 100 pips on EUR/USD (≈0.86%), that stop is ~1σ for the current regime. Below 7% = quiet market, above 12% = trending/stressed.'
  );

  // ── Regime ───────────────────────────────────────────────────────
  const regCell = document.querySelector('#section-risk .risk-cell:nth-child(6)');
  if (regCell) attachRiskTip(regCell,
    'Market Regime',
    'Composite live assessment: VIX level (primary driver), yield curve shape, gold intraday demand (>2% = stress signal), S&P 500 daily move (< -1.5% = stress), MOVE index (>100 = elevated per BofA/ICE), AUD/JPY intraday move (the canonical cross-asset risk barometer — sharp selloff >-1.5% = risk-off signal), USD/JPY (yen safe-haven bid), and HY OAS 20-day change (>+15bp widening = credit-market stress, often leads equity vol). Updates in real time.',
    'RISK-ON: VIX <18, no stress signals active. MIXED: 1 stress factor (e.g. VIX 18–25, or credit spreads widening while VIX stays calm). CAUTION: 2–3 factors. RISK-OFF: 4+ factors — high stress, USD/JPY/CHF bid, equities sold. Note: AUD/USD and NZD/USD falling modestly in isolation is normal when CBs diverge (RBA/RBNZ cuts) — AUD/JPY captures risk sentiment more cleanly.'
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
      title: 'HY OAS 20-day Change',
      body:  'Change in the ICE BofA US High Yield Option-Adjusted Spread over the last ~20 trading days. Captures direction, not just level — spreads can be historically tight yet still be widening, which the level alone would miss. Widening >15bp feeds into the Regime score above as a stress point.',
      ex:    'Widening (positive Δ) while equities are calm is often an early credit-market warning that leads risk-off moves in FX by weeks. Narrowing confirms improving risk appetite alongside a low VIX.'
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
    'ATM implied volatility from CBOE-listed FX ETF options (FXE, FXB, FXY, FXA) — nearest expiry ≥4 days. ETF IV is the closest free proxy for OTC interbank implied vol (not publicly available). COT bias from CFTC TFF · Leveraged Funds · Options+Futures Combined. 25-delta Risk Reversal from Saxo Bank public options page (1M tenor, indicative mid) — positive = calls bid over puts (upside skew on base currency); negative = puts bid (downside protection dominant).',
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
      // Credit spreads — USD-only (global USD credit market), from update_extended_data.py v14.0
      // NOTE: script stores hyOas/igOas/hyOasDelta20d in percentage points (e.g. 2.81 = 2.81%,
      // confirmed against the 2026-07-28 workflow_dispatch run: "HY OAS: 2.81%"/"IG OAS: 0.81%").
      // Convert to basis points here (×100) since the panel and its thresholds are bp-denominated.
      if (d.hyOas != null && !isNaN(d.hyOas) && d.hyOas > 0)        byId.hyOas = repo(d.hyOas * 100);
      if (d.igOas != null && !isNaN(d.igOas) && d.igOas > 0)        byId.igOas = repo(d.igOas * 100);
      if (d.hyOasDelta20d != null && !isNaN(d.hyOasDelta20d))       byId.hyOasDelta20d = d.hyOasDelta20d * 100;
      if (d.igOasDelta20d != null && !isNaN(d.igOasDelta20d))       byId.igOasDelta20d = d.igOasDelta20d * 100;
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

  // US HY OAS — ICE BofA High Yield Option-Adjusted Spread (FRED BAMLH0A0HYM2)
  // Thresholds: <350bp = historically tight/low-stress zone; >500bp = widely-cited
  // credit-stress threshold (2015 energy HY, March 2020 both crossed it decisively).
  if (byId.hyOas) {
    const hy = byId.hyOas.close;
    const cls = hy > 500 ? 'risk-val down' : hy > 350 ? 'risk-val warning' : 'risk-val up';
    setEl('risk-hyoas', Math.round(hy) + ' bp', cls);
    const signal = hy > 500 ? 'Wide' : hy > 350 ? 'Moderate' : 'Tight';
    if (byId.hyOasDelta20d != null) {
      const d = byId.hyOasDelta20d;
      const arrow = d > 0 ? '▲' : d < 0 ? '▼' : '→';
      const dStr = (d >= 0 ? ' +' : ' ') + Math.round(d) + 'bp';
      setEl('risk-hyoas-sub', arrow + dStr + ' · ' + signal + ' · ICE BofA');
    } else {
      setEl('risk-hyoas-sub', signal + ' · ICE BofA');
    }
  } else {
    setEl('risk-hyoas', '—', 'risk-val');
    setEl('risk-hyoas-sub', 'ICE BofA · unavailable');
  }

  // US IG OAS — ICE BofA Investment Grade Option-Adjusted Spread (FRED BAMLC0A0CM)
  // Thresholds: <100bp = tight (bottom-decile post-2009 territory); >150bp = widening.
  if (byId.igOas) {
    const ig = byId.igOas.close;
    const cls = ig > 150 ? 'risk-val down' : ig > 100 ? 'risk-val warning' : 'risk-val up';
    setEl('risk-igoas', Math.round(ig) + ' bp', cls);
    const signal = ig > 150 ? 'Wide' : ig > 100 ? 'Moderate' : 'Tight';
    if (byId.igOasDelta20d != null) {
      const d = byId.igOasDelta20d;
      const arrow = d > 0 ? '▲' : d < 0 ? '▼' : '→';
      const dStr = (d >= 0 ? ' +' : ' ') + Math.round(d) + 'bp';
      setEl('risk-igoas-sub', arrow + dStr + ' · ' + signal + ' · ICE BofA');
    } else {
      setEl('risk-igoas-sub', signal + ' · ICE BofA');
    }
  } else {
    setEl('risk-igoas', '—', 'risk-val');
    setEl('risk-igoas-sub', 'ICE BofA · unavailable');
  }

  // HY OAS 20d Δ — Risk Indicators table row (direction of travel, not just level)
  if (byId.hyOasDelta20d != null) {
    const delta = byId.hyOasDelta20d;
    const bp = Math.round(delta);
    const sign = bp >= 0 ? '+' : '';
    setEl('ri-hyoas-delta', sign + bp + 'bp');
    const sig = bp > 15 ? 'Widening (risk-off)' : bp < -15 ? 'Narrowing (risk-on)' : 'Stable';
    const cls = bp > 15 ? 'down' : bp < -15 ? 'up' : 'flat';
    setEl('ri-hyoas-delta-sig', sig, cls);
  } else {
    setEl('ri-hyoas-delta', '—');
    setEl('ri-hyoas-delta-sig', 'No data', 'flat');
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
    // chg is in percentage-point units (0.001 = 0.1bp); null when prev_close unavailable
    chg:        (byId[t.key]?.fromRepo || byId[t.key]?.prev_close == null) ? null : (byId[t.key]?.chg ?? null),
    fromRepo:   byId[t.key]?.fromRepo ?? false,
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
    // HY OAS 20d Δ widening (>15bp) = credit-spread stress — the "silent killer" leading
    // signal (credit stress precedes equity vol, e.g. 2007). Direction, not level: OAS
    // updates daily (not intraday) and can sit historically tight while still widening.
    // Industry precedent: KC Fed RORO Index, Gilchrist-Zakrajšek (2012) — credit spreads
    // are a standard leg of cross-asset risk-regime composites, often as/more predictive
    // than VIX alone. Added v8.72.0.
    if (byId.hyOasDelta20d != null && byId.hyOasDelta20d > 15) stressScore += 1;

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
    const yieldSrc = (byId.us2y && !byId.us2y.fromRepo) ? 'Live ~5min delay' : 'FRED DGS2 · daily batch';
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
    riskSub.textContent = 'VIX · MOVE · HV30 · ~5min delay · updated ' + hhmm + ' ' + tzAbbr;
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

  const _tc = v => _themeColor(v);
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle=_tc('--bg'); ctx.fillRect(0,0,W,H);

  // Grid lines
  const step = yRange <= 0.5 ? 0.1 : yRange <= 1 ? 0.25 : 0.5;
  const gridStart = Math.ceil(minY / step) * step;
  for (let v = gridStart; v <= maxY + 0.001; v = Math.round((v + step) * 1000) / 1000) {
    const y = py(v);
    ctx.strokeStyle=_tc('--border'); ctx.lineWidth=0.5; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(PAD_L,y); ctx.lineTo(W-PAD_R,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=_tc('--text2'); ctx.font='bold 8px Courier New'; ctx.textAlign='right';
    ctx.fillText(v.toFixed(2)+'%', PAD_L-3, y+3);
  }

  // Inverted zone — shade between shortest and longest tenor if inverted
  const firstV = vals[0], lastV = vals[n-1];
  if (firstV != null && lastV != null && firstV > lastV) {
    ctx.fillStyle=_themeColorAlpha('--down', 0.07);
    ctx.fillRect(PAD_L, PAD_T, cW, cH);
  }

  // Prior curve
  const priorPts = prevVals.map((v,i) => v != null ? [px(i), py(v)] : null).filter(Boolean);
  if (priorPts.length >= 2) {
    ctx.beginPath(); ctx.strokeStyle=_tc('--border2'); ctx.lineWidth=1;
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
    ctx.fillStyle=_themeColorAlpha('--chart-line', 0.07); ctx.fill();

    // Current curve line
    ctx.beginPath(); ctx.strokeStyle=_tc('--chart-line'); ctx.lineWidth=1.8;
    curPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();

    // Dots + value labels at each real point
    vals.forEach((v, i) => {
      if (v == null) return;
      const x = px(i), y = py(v);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2);
      ctx.fillStyle=_tc('--chart-line'); ctx.fill();
    });
  }

  // X-axis labels
  ctx.fillStyle=_tc('--text2'); ctx.font='bold 8.5px Courier New'; ctx.textAlign='center';
  labels.forEach((t,i) => ctx.fillText(t, px(i), H-5));

  // Legend
  ctx.textAlign='left';
  ctx.fillStyle=_tc('--chart-line'); ctx.fillText('● Current', PAD_L, PAD_T-2);
  ctx.fillStyle=_tc('--text3'); ctx.fillText('● Prior',   PAD_L+52, PAD_T-2);
  if (!isLive) {
    ctx.fillStyle=_tc('--text3'); ctx.fillText('(static)', PAD_L+92, PAD_T-2);
  } else {
    // Check inversion
    const spr = (vals[n-1] ?? 0) - (vals[0] ?? 0); // long - short
    if (spr < 0) { ctx.fillStyle=_themeColorAlpha('--down', 0.6); ctx.fillText('■ Inverted', PAD_L+92, PAD_T-2); }
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
  usdnok:'U.S. Dollar / Norwegian Krone', usdsek:'U.S. Dollar / Swedish Krona',
  eurnok:'Euro / Norwegian Krone', eursek:'Euro / Swedish Krona',
  move:'ICE BofA MOVE Index',
  hyoas:'ICE BofA US High Yield Index OAS', igoas:'ICE BofA US Corporate Index OAS',
};

// Symbols with no genuine intraday range — a single daily print from the source
// (FRED for us10y/hyoas/igoas), so any "candle"/"bar" would be a synthetic flat-body
// construction (open=prior close, high/low=min/max(open,close)), not real price action.
// Matches the existing TradingView-fallback convention (_LINE_STYLE_SYMS below) — Area
// is forced regardless of the user's globally-persisted chart-type selection, and the
// Candle/Bar buttons are disabled while one of these is the active symbol.
const _AREA_ONLY_IDS = new Set(['us10y', 'hyoas', 'igoas']);
function _effectiveChartType(ohlcId) {
  return _AREA_ONLY_IDS.has(ohlcId) ? 'area' : (window._lwChartType || 'candle');
}

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
  // G10 Scandinavian
  'FX_IDC:USDNOK': 'usdnok',  'FX_IDC:USDSEK': 'usdsek',
  'FX_IDC:EURNOK': 'eurnok',  'FX_IDC:EURSEK': 'eursek',
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
  // Credit spreads
  'FRED:BAMLH0A0HYM2':    'hyoas',
  'FRED:BAMLC0A0CM':      'igoas',
};

// Human-readable labels for the chart source footer
const _OHLC_LABELS = {
  gold: 'GC=F', wti: 'CL=F', btc: 'BTC-USD', us10y: '^TNX',
  spx: '^GSPC', nasdaq: '^NDX', nikkei: '^N225', stoxx: '^STOXX50E',
  eth: 'ETH-USD', dxy: 'DX-Y.NYB', vix: '^VIX', move: '^MOVE',
  silver: 'SI=F', brent: 'BZ=F', dax: '^GDAXI', ftse: '^FTSE', hsi: '^HSI', dji: '^DJI',
  hyoas: 'BAMLH0A0HYM2', igoas: 'BAMLC0A0CM',
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
  window._lwRenderDrawings = null;
  // Compare series belong to the chart instance being destroyed — clear the
  // runtime map only. window._lwCompareList (the persisted "what to compare"
  // list) is untouched here, same as window._lwIndState for indicators, so
  // _renderLWChart's restore pass (see COMPARE OVERLAY section) can re-add
  // matching series to the new chart once it's built.
  _lwCompareSeriesMap = {};
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
  'usdnok','usdsek',                               // G10 Scandinavian majors
  'eurgbp','eurjpy','eurchf','eurcad','euraud','eurnzd','gbpjpy',
  'gbpchf','gbpcad','gbpaud','gbpnzd','audjpy','audnzd','audchf',
  'audcad','cadjpy','cadchf','nzdjpy','nzdcad','nzdchf','chfjpy',
  'eurnok','eursek',                               // G10 Scandinavian crosses
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
                silver:2,brent:2,dax:2,ftse:2,hsi:2,dji:2,hyoas:0,igoas:0 }[ohlcId] ?? 5;
  const c = parseFloat(q.close.toFixed(dec));
  // Plausibility guard (defense-in-depth, v8.101.8/v8.101.9): an FX anchor
  // candidate (open source or session H/L) more than 2% away from the live
  // close is treated as unusable rather than trusted at face value — this is
  // what actually deformed the Sunday-reopen candle every week (root cause:
  // fetch_fx_prev_session() bailing out over the weekend, see
  // fetch_intraday_quotes.py v3.18). The root cause is fixed there; this
  // guard is a second, independent line of defense so that any OTHER stale/
  // wrong single-field anchor (e.g. a future yfinance hiccup) can't
  // reproduce the same deformed-body symptom silently. Hoisted to function
  // scope (v8.101.9 — was a `const` declared inside the open-anchor
  // `if (isFxBar)` block, out of scope for the separate session-H/L
  // `if (isFxBar)` block below, throwing "ReferenceError: _isPlausibleAnchor
  // is not defined" and taking down _lwBuildTodayBar/_renderLWChart entirely
  // for every FX pair — silently falling back to the TradingView iframe
  // widget instead of the native LWC chart).
  const _isPlausibleAnchor = (v) => v != null && v > 0 && Math.abs(v - c) / c <= 0.02;
  // Candle open convention:
  //   FX pairs  → prev_bar.close (open = the previous session's REAL close, self-computed
  //               every 5 min from 1H bars over the exact 21:00 UTC boundary — see
  //               fetch_fx_prev_session in fetch_intraday_quotes.py). Falls back to Yahoo's
  //               regularMarketPreviousClose (q.prev_close) only if prev_bar is missing.
  //   Non-FX    → regularMarketOpen (exchanges have a real session open; use it so the
  //               candle body reflects intraday movement, as TradingView does for BTC/SPX)
  let o;
  if (isFxBar) {
    // FX: anchor candle body to the last COMPLETED session's close so green/red == pct
    // direction. v8.117.16 fix: previously anchored to q.prev_close (Yahoo's
    // regularMarketPreviousClose), which can lag for hours after each 21:00 UTC session
    // rollover before Yahoo's own pipeline catches up — producing an oversized/wrong
    // body on the freshly-opened bar that self-corrected only once Yahoo's field updated
    // ("goes away a few hours later"). q.prev_bar.close is already computed independently
    // every 5-min cycle via direct 1H-bar aggregation over our own 21:00 UTC boundary
    // (fetch_fx_prev_session — previously wired into quotes.json only for gap-window
    // historical-bar injection, never used to anchor the LIVE today-bar). It is correct
    // from the instant the new session opens, with no lag window.
    o = _isPlausibleAnchor(q.prev_bar && q.prev_bar.close)
      ? parseFloat(q.prev_bar.close.toFixed(dec))
      : (_isPlausibleAnchor(q.prev_close)
          ? parseFloat(q.prev_close.toFixed(dec))
          : (_isPlausibleAnchor(q.open) ? parseFloat(q.open.toFixed(dec)) : c));
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
    if (_isPlausibleAnchor(q.session_high) && _isPlausibleAnchor(q.session_low)) {
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
      const _isLA = (_effectiveChartType(_lwActiveOhlcId) === 'line' || _effectiveChartType(_lwActiveOhlcId) === 'area');
      _lwCandleSeries.update(_isLA ? { time: _blockTs, value: _c } : _liveBar);
    } catch(_) {}

    // Sync chart header % with RT data.
    // BUGFIX (2026-07-29): this was previously gated on `_rt.pct != null`, but
    // _lwCandleSeries.update(_liveBar) above runs unconditionally on every tick.
    // Any tick where the feed delivered a valid `close` without a `pct` (briefly
    // missing % field, not a stale connection) updated the plotted candle but
    // left the O/H/L/C header text frozen at its last value — the header could
    // show a High far below what the candle was visibly drawing. The header
    // always needs the fresh bar; only the %/change override is conditional.
    if (_lwActiveUpdateHeader) {
      _lwActiveUpdateHeader(_liveBar, null, (_rt.pct != null) ? { pct: _rt.pct, chg: _rt.chg } : null);
    }
    // Re-project the trend-line/swing overlay: a live tick can widen the
    // autoscaled price range without firing subscribeVisibleTimeRangeChange,
    // which would otherwise leave the SVG diagonal stale against the new axis.
    // (Fib levels don't need this -- createPriceLine already tracks the axis
    // itself, see _syncFibPriceLines.)
    if (window._lwRenderDrawings) window._lwRenderDrawings();
    return;
  }

  // D1 / W1 / MN live today-bar (unchanged path)
  const bar = _lwBuildTodayBar(_lwActiveOhlcId);
  if (!bar) return;
  try {
    // Line/Area series use {time, value} -- not OHLC format
    const isLineArea = (_effectiveChartType(_lwActiveOhlcId) === 'line' || _effectiveChartType(_lwActiveOhlcId) === 'area');
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
  if (window._lwRenderDrawings) window._lwRenderDrawings();
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
  const rightPad    = 14;
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

// D1 default zoom: 3M on mobile/tablet viewports (less clutter on small screens),
// 6M on desktop. Same 900px breakpoint used elsewhere in this file (e.g. split-layout isMobile()).
function _lwDefaultD1Days() { return (window.innerWidth <= 900) ? 91 : 182; }

let _lwActiveDays = _lwDefaultD1Days(); // default: 3M (mobile) / 6M (desktop), calendar days
let _lwActiveTf   = 'D1'; // active timeframe: H1 | H4 | D1 | W1 | MN
// Compare overlay state (v8.172.0: multi-slot, persisted — see the COMPARE
// OVERLAY section below for the full rationale). _lwCompareSeriesMap is
// runtime-only (uid -> LWC series object), reset to {} every time the chart
// is destroyed/rebuilt (_destroyLWChart below) since a destroyed chart's own
// series objects can't be reused. window._lwCompareList is the *persisted*
// "what should be compared" list — survives symbol switches, timeframe
// switches, and leaving/returning to the chart, exactly like the indicator
// engine's window._lwIndState further down this file — and is what actually
// gets re-applied on every fresh _renderLWChart() call.
let _lwCompareSeriesMap = {};
const _LS_COMPARE = 'gi_compare_list'; // [ { uid, cmpId, cmpLabel, cmpType } ]
function _lsGetCompare() {
  try { const v = localStorage.getItem(_LS_COMPARE); return v ? JSON.parse(v) : []; }
  catch(_e) { return []; }
}
function _lsSetCompare(list) {
  try { localStorage.setItem(_LS_COMPARE, JSON.stringify(list)); } catch(_e) {}
}
if (typeof window._lwCompareList === 'undefined') window._lwCompareList = _lsGetCompare();
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

  // Seasonality (beta) tracks the active chart symbol so #szn-btn knows what
  // to fetch — set here (single call site for every symbol switch) rather
  // than re-deriving it from _TV_TO_OHLC in multiple places. Panel refetches
  // only if it's actually open; see _sznOnSymbolChange() below.
  window._sznActiveOhlcId = ohlcId;
  if (typeof window._sznOnSymbolChange === 'function') window._sznOnSymbolChange(ohlcId);

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

    // ── Current-period key, computed from "now" — NOT assumed from bars[] ───
    // Bug history: the snapshot below used to assume bars[bars.length-1] was
    // always the current incomplete period. That's false right after a period
    // boundary with no D1 bar yet for the new period — most commonly every
    // Monday between the FX session open (Sun 21:00 UTC) and the daily OHLC
    // workflow run (~22:30 UTC) that writes Monday's D1 bar. In that window the
    // last aggregated bar is the just-COMPLETED prior week, so using its O/H/L
    // made the live today-bar mimic the prior week's exact range — a visual
    // "duplicate candle" of the previous period (reported by user; confirmed
    // against production ohlc-data/eurusd.json on 2026-06-22: last D1 bar was
    // Fri 06-19, no 06-22 bar yet, so the aggregated 06-15 week was wrongly
    // snapshotted as "current").
    const _nowKeyD = new Date();
    let _currentPeriodKey;
    if (_activeTf === 'W1') {
      const _kDow = _nowKeyD.getUTCDay() || 7;
      const _kMon = new Date(_nowKeyD);
      _kMon.setUTCDate(_nowKeyD.getUTCDate() - (_kDow - 1));
      _currentPeriodKey = _kMon.toISOString().slice(0, 10);
    } else {
      _currentPeriodKey = _nowKeyD.toISOString().slice(0, 7) + '-01';
    }

    // ── Snapshot current-period O/H/L for _lwBuildTodayBar ─────────────────
    // Only use the last aggregated bar's O/H/L if it actually IS the current
    // period. Otherwise leave the globals null (already null from
    // _destroyLWChart above) so _lwBuildTodayBar falls back to its normal
    // prev_close + session H/L computation — the correct behaviour for a
    // period that has no D1 data yet.
    const _curPeriodBar = bars[bars.length - 1];
    if (_curPeriodBar && _curPeriodBar.time === _currentPeriodKey) {
      _lwPeriodOpen = _curPeriodBar.open;
      _lwPeriodHigh = _curPeriodBar.high;
      _lwPeriodLow  = _curPeriodBar.low;
    }
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
  // touch-action:none — see fix note in CHANGELOG v8.76.0 ("mobile drawing
  // tools committing the shape only after leaving the chart area").
  chartDiv.style.cssText = 'width:100%;height:100%;touch-action:none;';
  wrap.appendChild(chartDiv);

  // Enable pointer events for LW chart interactivity (zoom, pan, crosshair)
  wrap.style.pointerEvents = 'auto';

  // Decimal precision map — drives minMove and formatting
  const dec = { eurusd:5,gbpusd:5,usdjpy:3,audusd:5,usdcad:5,usdchf:5,nzdusd:5,
                eurgbp:5,eurjpy:3,eurchf:5,eurcad:5,euraud:5,eurnzd:5,gbpjpy:3,
                gbpchf:5,gbpcad:5,gbpaud:5,gbpnzd:5,audjpy:3,audnzd:5,audchf:5,
                audcad:5,cadjpy:3,cadchf:5,nzdjpy:3,nzdcad:5,nzdchf:5,chfjpy:3,
                gold:2,wti:2,btc:2,us10y:4,spx:2,nasdaq:2,nikkei:2,stoxx:2,eth:2,dxy:3,
                silver:2,brent:2,dax:2,ftse:2,hsi:2,dji:2,hyoas:0,igoas:0 }[ohlcId] ?? 5;
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
    layout:      { background: { color: _themeColor('--bg') }, textColor: _themeColor('--text'), attributionLogo: false,
                   panes: { separatorColor: _themeColorAlpha('--border', 0.6), separatorHoverColor: _themeColorAlpha('--text2', 0.25), enableResize: true } },
    grid:        { vertLines: { color: _themeColorAlpha('--border', 0.5) }, horzLines: { color: _themeColorAlpha('--border', 0.5) } },
    crosshair:   { mode: LWC.CrosshairMode.Normal,
                   vertLine: { color: _themeColorAlpha('--text2', 0.5), labelBackgroundColor: _themeColor('--bg3') },
                   horzLine: { color: _themeColorAlpha('--text2', 0.5), labelBackgroundColor: _themeColor('--bg3') } },
    rightPriceScale: { borderColor: _themeColor('--border'), minimumWidth: 65,
                       scaleMargins: mainScaleMargins },
    timeScale:   { borderColor: _themeColor('--border'), timeVisible: false, secondsVisible: false,
                   rightOffset: 14, minBarSpacing: 1,
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

  // Symbols in _AREA_ONLY_IDS (us10y, hyoas, igoas) have no genuine intraday range —
  // force Area regardless of the globally-persisted selection, and disable Candle/Bar
  // so the buttons can't produce a synthetic flat-body "candle" from a single daily print.
  const _isAreaOnlyId = _AREA_ONLY_IDS.has(ohlcId);
  const _chartType = _effectiveChartType(ohlcId);

  // Sync TYPE button state on render
  document.querySelectorAll('[data-chart-type]').forEach(btn => {
    const _isCandleOrBar = (btn.dataset.chartType === 'candle' || btn.dataset.chartType === 'bar');
    const isActive = btn.dataset.chartType === _chartType;
    btn.classList.toggle('sel', isActive);
    btn.classList.remove('on');
    btn.disabled = _isAreaOnlyId && _isCandleOrBar;
    btn.title = (_isAreaOnlyId && _isCandleOrBar)
      ? 'Not available — this series has one print per day, no intraday range'
      : '';
  });

  // Helper: convert OHLC bars to close-only for line/area
  const closeBars = bars.filter(b => b.close != null).map(b => ({ time: b.time, value: b.close }));

  let candleSeries;
  const _priceFormat = { type: 'price', precision: dec, minMove };

  if (_chartType === 'bar') {
    // Bar (OHLC) series — same data as candlestick, different visual
    if (typeof LWC.BarSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.BarSeries, {
        upColor: _themeColor('--candle-up'), downColor: _themeColor('--candle-down'),
        openVisible: true, thinBars: false,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addBarSeries({
        upColor: _themeColor('--candle-up'), downColor: _themeColor('--candle-down'),
        priceFormat: _priceFormat,
      });
    }
    candleSeries.setData(bars);
  } else if (_chartType === 'line') {
    // Line series — close prices only
    if (typeof LWC.LineSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.LineSeries, {
        color: _themeColor('--chart-line'), lineWidth: 2,
        priceLineVisible: false, lastValueVisible: true,
        crosshairMarkerVisible: true, crosshairMarkerRadius: 4,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addLineSeries({ color: _themeColor('--chart-line'), lineWidth: 2, priceFormat: _priceFormat });
    }
    candleSeries.setData(closeBars);
  } else if (_chartType === 'area') {
    // Area series — close prices with gradient fill
    if (typeof LWC.AreaSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.AreaSeries, {
        lineColor: _themeColor('--chart-line'), lineWidth: 2,
        topColor: _themeColorAlpha('--chart-line', 0.28), bottomColor: _themeColorAlpha('--chart-line', 0.02),
        priceLineVisible: false, lastValueVisible: true,
        crosshairMarkerVisible: true, crosshairMarkerRadius: 4,
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addAreaSeries({ lineColor: _themeColor('--chart-line'), lineWidth: 2, priceFormat: _priceFormat });
    }
    candleSeries.setData(closeBars);
  } else {
    // Default: Candlestick — LWC v5 API with v4 fallback
    if (typeof LWC.CandlestickSeries !== 'undefined') {
      candleSeries = _lwChart.addSeries(LWC.CandlestickSeries, {
        upColor: _themeColor('--candle-up'), downColor: _themeColor('--candle-down'),
        borderUpColor: _themeColor('--candle-up'), borderDownColor: _themeColor('--candle-down'),
        wickUpColor: _themeColor('--candle-up'), wickDownColor: _themeColor('--candle-down'),
        priceFormat: _priceFormat,
      });
    } else {
      candleSeries = _lwChart.addCandlestickSeries({
        upColor: _themeColor('--candle-up'), downColor: _themeColor('--candle-down'),
        borderUpColor: _themeColor('--candle-up'), borderDownColor: _themeColor('--candle-down'),
        wickUpColor: _themeColor('--candle-up'), wickDownColor: _themeColor('--candle-down'),
        priceFormat: _priceFormat,
      });
    }
    candleSeries.setData(bars);
  }
  // Expose to module scope so gi-theme-change listener can recolor on theme switch
  window._candleSeries = candleSeries;
  window._candleSeriesType = _chartType;

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
        axisLabelColor: _themeColor('--border'),
        axisLabelTextColor: _themeColor('--text3'),
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
        const _isLA = (_chartType === 'line' || _chartType === 'area');
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
      // Sorted list of actual bar dates, used to snap a meeting date onto the
      // bar that covers it. bars[] is already chronological.
      const barTimesSorted = bars.map(b => b.time);
      relevantCBs.forEach(cb => {
        const cbMtg = mtgData.meetings[cb];
        if (!cbMtg?.allMeetings) return;
        const color = _CB_COLORS[cb] || 'rgba(144,150,160,0.8)';
        cbMtg.allMeetings.forEach(dateStr => {
          if (dateStr < firstDate || dateStr > lastDate) return;
          let targetDate = barDates.has(dateStr) ? dateStr : null;
          if (!targetDate) {
            // D1 weekend/holiday case: meeting fell on a non-trading day —
            // try the next calendar day (matches a Monday after a Fri/Sat/Sun date).
            const d = new Date(dateStr + 'T12:00:00Z');
            d.setDate(d.getDate() + 1);
            const next = d.toISOString().slice(0, 10);
            if (barDates.has(next)) targetDate = next;
          }
          if (!targetDate) {
            // W1/MN case (and any D1 gap the +1-day shift didn't catch): bars
            // are keyed by period start (ISO Monday / first-of-month — see
            // W1/MN aggregation above), so an exact-date match essentially
            // never exists. Snap to the last bar whose date is <= the meeting
            // date, i.e. the bar for the period that actually contains it.
            for (let i = barTimesSorted.length - 1; i >= 0; i--) {
              if (barTimesSorted[i] <= dateStr) { targetDate = barTimesSorted[i]; break; }
            }
          }
          if (!targetDate) return;
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

  // ── Drawing Tools — Trend Line, Fibonacci Retracement, Rectangle ────────────
  // Industry-standard UX (TradingView/Bloomberg/MT5): pick a tool from the
  // "Draw" menu, then press-and-drag on the chart — the shape follows the
  // cursor live and commits on release. Persisted per symbol only (a line
  // drawn on EUR/USD doesn't show up on GBP/USD) — and, as of the fix below,
  // shows at the same real-world time/price on EVERY timeframe (D1, W1, MN,
  // H1, H4), not just the one it was drawn on. Rendered as an SVG overlay
  // synced to the chart via timeToCoordinate/priceToCoordinate on every
  // pan/zoom and on every live tick, same pattern as the CB meeting-markers
  // overlay just above.
  //
  // BUGFIX (2026-07-29): Fibonacci levels previously rendered via the native
  // `series.createPriceLine()` API to solve a label-collision bug — but native
  // price lines always span the *entire* chart width with no way to bound
  // them, which is what produced the "Fibonacci spans the whole chart"
  // complaint. Reverted to hand-drawn SVG lines, but bounded strictly between
  // the two x-coordinates the user actually dragged (xLeft..xRight) instead of
  // stretching to the chart edge — the box's width is now literally whatever
  // width the user defines by dragging, and the level labels sit just past
  // the right edge of that box rather than colliding with the price axis.
  //
  // BUGFIX (2026-07-30): points are stored as universal unix-epoch time (see
  // _timeToEpoch/_epochToActiveTime above _resolveTimeAt/_xForPoint) instead of
  // whatever raw format the timeframe active at draw-time happened to use
  // ('YYYY-MM-DD' string for D1/W1/MN, unix seconds for H1/H4). That's what makes
  // a single object project onto every timeframe's series at the correct
  // real-world coordinate, so storage no longer needs to be split per
  // timeframe (or timeframe-group) at all — one array per symbol.
  const _DRAW_LS_KEY  = 'gi_drawings';
  const _drawSymKey   = ohlcId;
  if (typeof window._lwDrawings === 'undefined') {
    try { const v = localStorage.getItem(_DRAW_LS_KEY); window._lwDrawings = v ? JSON.parse(v) : {}; }
    catch(_) { window._lwDrawings = {}; }
  }
  function _saveDrawings() {
    try { localStorage.setItem(_DRAW_LS_KEY, JSON.stringify(window._lwDrawings)); } catch(_) {}
  }
  function _curDrawings() {
    if (!window._lwDrawings[_drawSymKey]) window._lwDrawings[_drawSymKey] = [];
    return window._lwDrawings[_drawSymKey];
  }

  let _drawMode      = null;  // null | 'trend' | 'fib' | 'fibext' | 'rect'  (creation tool armed)
  let _drawAnchor    = null;  // {time, price} — drag-start point (creation)
  let _drawLiveEnd   = null;  // {time, price} — live drag-end point, updated on every move (creation)
  let _isDragging    = false; // true while creating a new shape
  let _drawOverlay   = null;  // SVG overlay element
  let _drawRafId     = null;

  // Selection / move / resize state — industry-standard interaction (TradingView/
  // MT5/Bloomberg): a plain click SELECTS an object (never deletes it), dragging
  // its body MOVES it, dragging an endpoint handle RESIZES it. Deletion and color
  // live in a floating toolbar shown above the selection, never on click.
  let _selectedIdx   = -1;    // index into _curDrawings() of the selected object, or -1
  let _dragMode      = null;  // null | 'move' | 'resize-p1' | 'resize-p2'
  let _dragStartPx   = null;  // {x, y} pixel position where the move-drag started
  let _dragOrigP1    = null;  // baseline p1 at move-drag start (for delta translation)
  let _dragOrigP2    = null;  // baseline p2 at move-drag start
  let _drawToolbarEl = null;  // floating color/delete toolbar DOM node
  const _HANDLE_R    = 8;     // px hit-radius for endpoint resize handles

  // Remove any toolbar left over from a previous render pass (symbol/timeframe
  // switch) — selection state resets to -1 on every rebuild of this block, so a
  // stale toolbar node would otherwise be orphaned with no owner to hide it.
  (function _cleanupStaleToolbar() {
    const stale = document.getElementById('_lw-draw-toolbar');
    if (stale) stale.remove();
  })();

  const _DRAW_COLORS = { trend: 'rgba(79,127,255,0.9)', fib: 'rgba(255,193,7,0.9)', fibext: 'rgba(0,191,165,0.9)', rect: 'rgba(126,211,138,0.9)' };
  const _SWATCH_COLORS = [
    'rgba(79,127,255,0.9)',   // blue
    'rgba(255,193,7,0.9)',    // amber
    'rgba(126,211,138,0.9)',  // green
    'rgba(255,99,99,0.9)',    // red
    'rgba(186,133,255,0.9)',  // purple
    'rgba(235,235,235,0.9)',  // white/gray
  ];

  // Standard Fibonacci retracement ratios (TradingView/MT default set)
  const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];

  // Standard Fibonacci extension ratios (2-point variant — projects
  // continuation targets beyond the drawn swing, same convention as
  // TradingView's "Fib Extension" tool and the 127.2/161.8/200/261.8%
  // levels commonly quoted as extension targets in institutional research).
  // This is the 2-point tool: 0% and 100% sit on the dragged swing exactly
  // like Fibonacci Retracement, and the levels past 100% project outward in
  // the same direction as the swing. A 3-point "trend-based" extension
  // (swing start → swing end → retracement end) is a materially bigger
  // feature — separate creation flow, a third handle, extended selection/
  // resize/serialization — and is not implemented here; flagged as a
  // possible follow-up if Santiago wants full Bloomberg/MT5 parity.
  const FIB_EXT_LEVELS = [0, 0.382, 0.618, 1, 1.272, 1.618, 2, 2.618];

  function _updateDrawBtnState() {
    const b = document.getElementById('lw-draw-btn');
    if (b) b.classList.toggle('on', !!_drawMode);
    try { chartDiv.style.cursor = _drawMode ? 'crosshair' : ''; } catch(_) {}
  }

  // Toggle chart panning/zooming off while the user is actively dragging out a
  // shape (otherwise a press-drag on the canvas would pan the chart instead of
  // drawing). Restored the instant the drag ends.
  const _panScrollOpts  = { mouseWheel: true, pressedMouseMove: true,  horzTouchDrag: true,  vertTouchDrag: false };
  const _panScaleOpts   = { mouseWheel: true, pinch: true,  axisPressedMouseMove: { time: true,  price: true  } };
  const _noPanScrollOpts = { mouseWheel: true, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false };
  const _noPanScaleOpts  = { mouseWheel: true, pinch: false, axisPressedMouseMove: { time: false, price: false } };
  function _setChartPannable(enabled) {
    try {
      _lwChart.applyOptions({
        handleScroll: enabled ? _panScrollOpts : _noPanScrollOpts,
        handleScale:  enabled ? _panScaleOpts  : _noPanScaleOpts,
      });
    } catch(_) {}
  }

  // Builds the SVG markup for one drawing (or a live in-progress preview).
  // isSelected adds a soft outer glow so the active selection reads clearly
  // against the busy chart background — the actual grab handles are drawn
  // separately by _svgHandles so they stay on top of everything.
  function _svgForDrawing(d, isPreview, isSelected) {
    const ts = _lwChart.timeScale();
    const x1 = _xForPoint(ts, d.p1), x2 = _xForPoint(ts, d.p2);
    const y1 = candleSeries.priceToCoordinate(d.p1.price), y2 = candleSeries.priceToCoordinate(d.p2.price);
    // ── Temporary diagnostic (v8.86.10) ─────────────────────────────────────
    // v8.86.8/9 only logged the null-coordinate and thrown-exception cases —
    // neither fired on W1/MN, which means _svgForDrawing IS completing with
    // non-null numeric coordinates. So log the actual numbers now (not just
    // "is it null") to see whether they're just wildly wrong / off-screen,
    // or something else (clip-path, color) is hiding an otherwise-correct
    // shape. Remove once root cause is confirmed.
    if (window._lwDebugDraw) {
      const _key = _lwActiveTf + '|' + d.type + '|' + x1 + '|' + x2 + '|' + y1 + '|' + y2;
      if (window._lwDebugLastKey !== _key) {
        window._lwDebugLastKey = _key;
        console.warn('[lw-draw] coords on', _lwActiveTf, {
          type: d.type, x1, x2, y1, y2,
          clipPath: _drawOverlay ? _drawOverlay.style.clipPath : null,
          chartDivW: chartDiv.clientWidth, chartDivH: chartDiv.clientHeight,
        });
      }
    }
    if (window._lwDebugDraw && (x1 == null || x2 == null || y1 == null || y2 == null)) {
      console.warn('[lw-draw] vanished on', _lwActiveTf, {
        type: d.type,
        p1: d.p1, p2: d.p2,
        x1, x2, y1, y2,
        dataLen: (candleSeries.data() || []).length,
        firstBar: (candleSeries.data() || [])[0]?.time,
        lastBar: (candleSeries.data() || []).slice(-1)[0]?.time,
      });
    }
    if (x1 == null || x2 == null || y1 == null || y2 == null) return '';
    const col = d.color || _DRAW_COLORS[d.type] || _DRAW_COLORS.trend;
    const previewDash = isPreview ? ' stroke-dasharray="3,3"' : '';
    const selW = isSelected ? 1 : 0; // extra stroke-width added when selected
    let svg = '';
    if (d.type === 'trend') {
      if (isSelected) svg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#fff" stroke-width="5" opacity="0.18"/>`;
      svg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="${(1.5 + selW).toFixed(1)}"${previewDash}/>`;
      svg += `<circle cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="3" fill="${col}"/>`;
      svg += `<circle cx="${x2.toFixed(1)}" cy="${y2.toFixed(1)}" r="3" fill="${col}"/>`;
    } else if (d.type === 'rect') {
      // Minimum on-screen size: a rectangle drawn on a fine timeframe (e.g. a
      // 9-day box on D1) can collapse to a sub-pixel sliver when viewed on a
      // much coarser one (a single W1 bar ≈ 7 days, a single MN bar ≈ 30 —
      // 9 real days can be a fraction of one bar's pixel width there). An SVG
      // <rect> with width or height rounding to 0 doesn't render AT ALL per
      // spec — not even its stroke — so the shape would silently vanish
      // exactly the way Santiago reported on W1/MN. TradingView/MT5 never let
      // a drawn object disappear this way; they keep at least a visible
      // sliver. Fix: clamp both dimensions to a 2px floor, expanding outward
      // from the shape's own center so it stays anchored at the same
      // real-world midpoint rather than snapping to one edge.
      const MIN_DIM = 2;
      let rx = Math.min(x1, x2), ry = Math.min(y1, y2);
      let rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1);
      if (rw < MIN_DIM) { const cx = (x1 + x2) / 2; rx = cx - MIN_DIM / 2; rw = MIN_DIM; }
      if (rh < MIN_DIM) { const cy = (y1 + y2) / 2; ry = cy - MIN_DIM / 2; rh = MIN_DIM; }
      const fillCol = col.replace(/[\d.]+\)$/, '0.14)');
      if (isSelected) svg += `<rect x="${(rx-2).toFixed(1)}" y="${(ry-2).toFixed(1)}" width="${(rw+4).toFixed(1)}" height="${(rh+4).toFixed(1)}" fill="none" stroke="#fff" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>`;
      svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" fill="${fillCol}" stroke="${col}" stroke-width="${(1.25 + selW).toFixed(2)}"${previewDash}/>`;
    } else if (d.type === 'fib' || d.type === 'fibext') {
      // Diagonal swing guide (dashed, low-opacity — the levels below are the point).
      svg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="1" stroke-dasharray="4,3" opacity="0.55"/>`;
      const drawnHighFirst = d.p1.price >= d.p2.price;
      const priceHigh = Math.max(d.p1.price, d.p2.price);
      const priceLow  = Math.min(d.p1.price, d.p2.price);
      const range = priceHigh - priceLow || 1e-9;
      // Bounded to the width the user actually dragged — NOT the chart edge.
      const xLeft  = Math.min(x1, x2);
      const xRight = Math.max(x1, x2);
      const levels = d.type === 'fibext' ? FIB_EXT_LEVELS : FIB_LEVELS;
      // BUGFIX (2026-08-07): Fibonacci Retracement had 0%/100% swapped.
      // Verified against TradingView/MT5's actual behavior: when you drag
      // from the swing low to the swing high, 0% lands at the point you
      // dragged TO (the most recent extreme, i.e. the high) and 100% at the
      // point you dragged FROM (i.e. the low) — NOT the other way around,
      // which is what the previous formula produced. That inversion is what
      // put 61.8%/78.6% near the top of an up-swing instead of near the
      // bottom, where traders expect the "golden pocket" to sit.
      // Fibonacci Extension intentionally keeps the OLDER from=0%/to=100%
      // anchor, since it's a distinct convention: levels past 100% are
      // meant to keep projecting outward beyond the TO point, in the same
      // direction as the drawn swing (a continuation target, not a
      // retracement zone) — flipping it would send extensions backward
      // past the FROM point instead.
      const isRetracement = d.type === 'fib';
      levels.forEach(lv => {
        const lvPrice = isRetracement
          ? (d.p2.price + lv * (d.p1.price - d.p2.price))
          : (drawnHighFirst ? (priceHigh - lv * range) : (priceLow + lv * range));
        const y = candleSeries.priceToCoordinate(lvPrice);
        if (y == null) return;
        const isAnchor = (lv === 0 || lv === 1);
        svg += `<line x1="${xLeft.toFixed(1)}" y1="${y.toFixed(1)}" x2="${xRight.toFixed(1)}" y2="${y.toFixed(1)}" `
             + `stroke="${col}" stroke-width="1" ${isAnchor ? '' : 'stroke-dasharray="2,2"'} opacity="0.9"${previewDash}/>`;
        svg += `<text x="${(xRight + 4).toFixed(1)}" y="${(y - 2).toFixed(1)}" text-anchor="start" `
             + `font-size="9" font-family="var(--font-ui,sans-serif)" fill="${col}">`
             + `${(lv * 100).toFixed(1)}% \u2013 ${lvPrice.toFixed(dec)}</text>`;
      });
    }
    return svg;
  }

  // Two square grab handles at p1/p2 for the selected object — dragging either
  // one reshapes that endpoint (resize); dragging the shape body between them
  // moves the whole object. Drawn last so they stay visually on top.
  function _svgHandles(d) {
    if (!d) return '';
    const ts = _lwChart.timeScale();
    const x1 = _xForPoint(ts, d.p1), y1 = candleSeries.priceToCoordinate(d.p1.price);
    const x2 = _xForPoint(ts, d.p2), y2 = candleSeries.priceToCoordinate(d.p2.price);
    if (x1 == null || y1 == null || x2 == null || y2 == null) return '';
    const hs = 5;
    let svg = '';
    [[x1, y1], [x2, y2]].forEach(([hx, hy]) => {
      svg += `<rect x="${(hx - hs).toFixed(1)}" y="${(hy - hs).toFixed(1)}" width="${hs * 2}" height="${hs * 2}" `
           + `fill="var(--bg,#131722)" stroke="#ffffff" stroke-width="1.5" rx="1.5"/>`;
    });
    return svg;
  }

  function _renderDrawings() {
    if (_drawRafId) cancelAnimationFrame(_drawRafId);
    _drawRafId = requestAnimationFrame(() => {
      _drawRafId = null;
      if (!_drawOverlay || !_lwChart) return;
      if (window._lwDebugDraw) {
        const _rKey = 'run|' + _lwActiveTf + '|' + _curDrawings().length;
        if (window._lwDebugLastRunKey !== _rKey) {
          window._lwDebugLastRunKey = _rKey;
          console.warn('[lw-draw] _renderDrawings ran on', _lwActiveTf, 'count:', _curDrawings().length);
        }
      }
      // Clip the overlay so drawings tuck UNDER the price-scale ribbon (right)
      // and the time-axis strip (bottom) instead of painting on top of them —
      // the overlay is a plain absolutely-positioned SVG sibling spanning the
      // full chartDiv, so without this a trend line/rectangle dragged into
      // that space visually sat above the price labels. Queried fresh on
      // every render since price-scale width isn't fixed — it grows with the
      // digit count of the current symbol's price format (e.g. USDJPY vs
      // EURUSD) and time-axis height can vary slightly with font metrics.
      try {
        const rightW = _lwChart.priceScale('right').width() || 0;
        let botH   = _lwChart.timeScale().height() || 0;
        // Every drawing type (trend/fib/rect) is anchored exclusively to the
        // MAIN price series (candleSeries, pane 0) — none of them can ever be
        // attached to an oscillator sub-pane (RSI, MACD, Stochastic, etc.).
        // The old clip-path only excluded the right price-scale ribbon and
        // the bottom time-axis strip from the WHOLE chartDiv, which spans
        // every pane stacked together once an indicator is active. A shape
        // whose price fell outside pane 0's own autoscaled range (a real
        // possibility once a shape can be viewed on a timeframe/zoom far
        // from where it was drawn — see _epochToXInterpolated above) simply
        // kept extending straight through pane 0's bottom edge into
        // whatever sub-pane sat below it, which is what Santiago's H1/H4
        // screenshots showed: a rectangle spilling into the oscillator's
        // plot area. Fix: also clip at pane 0's own bottom edge whenever
        // more than one pane exists, using its actual HTMLElement height
        // (the ground-truth pixel height, not an assumption about layout).
        const panes = _lwChart.panes();
        if (panes && panes.length > 1) {
          const mainPaneEl = panes[0].getHTMLElement && panes[0].getHTMLElement();
          const overlayH   = chartDiv.clientHeight || 0;
          const mainPaneH  = mainPaneEl ? mainPaneEl.clientHeight : 0;
          if (overlayH > 0 && mainPaneH > 0) botH = Math.max(botH, overlayH - mainPaneH);
        }
        _drawOverlay.style.clipPath = `inset(0 ${rightW}px ${botH}px 0)`;
      } catch(_) {}
      let svg = '';
      const arr = _curDrawings();
      arr.forEach((d, i) => {
        try { svg += _svgForDrawing(d, false, i === _selectedIdx); }
        catch(err) {
          // v8.86.9: was a silent catch(_){} — swallowed any exception thrown
          // inside _svgForDrawing (as opposed to a clean null-coordinate
          // return), which meant the v8.86.8 null-coordinate diagnostic could
          // never fire for that case. Surface it the same way so the real
          // failure (if it's a thrown error, not a null) is visible.
          if (window._lwDebugDraw) console.error('[lw-draw] _svgForDrawing threw on', _lwActiveTf, d.type, err);
        }
      });
      if (_isDragging && _drawAnchor && _drawLiveEnd) {
        try {
          svg += _svgForDrawing({ type: _drawMode, p1: _drawAnchor, p2: _drawLiveEnd, color: _DRAW_COLORS[_drawMode] }, true, false);
        } catch(_) {}
      }
      if (_selectedIdx >= 0 && arr[_selectedIdx] && !_isDragging) {
        try { svg += _svgHandles(arr[_selectedIdx]); } catch(_) {}
      }
      _drawOverlay.innerHTML = svg;
      if (_selectedIdx >= 0) _positionDrawToolbar();
    });
  }

  if (!_drawOverlay) {
    _drawOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    _drawOverlay.id = 'lw-draw-overlay';
    _drawOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:3;overflow:visible;';
    chartDiv.style.position = 'relative';
    chartDiv.appendChild(_drawOverlay);
  }
  _renderDrawings();
  _lwChart.timeScale().subscribeVisibleTimeRangeChange(_renderDrawings);
  // Expose for real-time redraw from _lwUpdateTodayBar() — every shape needs
  // re-projecting whenever a live tick shifts the price scale's autoscaled
  // range, which doesn't fire a visible-time-range-change event on its own.
  window._lwRenderDrawings = _renderDrawings;

  // Distance from a point to a line segment (for click-to-delete hit testing)
  function _ptSegDist(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  function _hitTestDrawing(x, y) {
    const ts = _lwChart.timeScale();
    const arr = _curDrawings();
    for (let i = arr.length - 1; i >= 0; i--) {
      const d = arr[i];
      const x1 = _xForPoint(ts, d.p1), x2 = _xForPoint(ts, d.p2);
      const y1 = candleSeries.priceToCoordinate(d.p1.price), y2 = candleSeries.priceToCoordinate(d.p2.price);
      if (x1 == null || x2 == null || y1 == null || y2 == null) continue;
      if (d.type === 'trend') {
        if (_ptSegDist(x, y, x1, y1, x2, y2) < 6) return i;
      } else if (d.type === 'rect') {
        const rx1 = Math.min(x1, x2) - 4, rx2 = Math.max(x1, x2) + 4;
        const ry1 = Math.min(y1, y2) - 4, ry2 = Math.max(y1, y2) + 4;
        if (x >= rx1 && x <= rx2 && y >= ry1 && y <= ry2) return i;
      } else if (d.type === 'fib' || d.type === 'fibext') {
        if (_ptSegDist(x, y, x1, y1, x2, y2) < 6) return i; // the diagonal swing guide itself
        const drawnHighFirst = d.p1.price >= d.p2.price;
        const priceHigh = Math.max(d.p1.price, d.p2.price), priceLow = Math.min(d.p1.price, d.p2.price);
        const range = priceHigh - priceLow || 1e-9;
        const xLeft = Math.min(x1, x2) - 4, xRight = Math.max(x1, x2) + 4;
        const levels = d.type === 'fibext' ? FIB_EXT_LEVELS : FIB_LEVELS;
        // Mirrors the retracement/extension anchor split in _svgForDrawing —
        // hit-testing must use the same 0%/100% convention as the render,
        // or clicking a visible level line would miss it.
        const isRetracement = d.type === 'fib';
        for (const lv of levels) {
          const lvPrice = isRetracement
            ? (d.p2.price + lv * (d.p1.price - d.p2.price))
            : (drawnHighFirst ? (priceHigh - lv * range) : (priceLow + lv * range));
          const ly = candleSeries.priceToCoordinate(lvPrice);
          if (ly != null && x >= xLeft && x <= xRight && Math.abs(y - ly) < 5) return i;
        }
      }
    }
    return -1;
  }

  // Which endpoint handle (if any) of the given drawing sits under x,y —
  // checked before body hit-testing so a resize grab always wins over a move.
  function _hitTestHandle(idx, x, y) {
    const d = _curDrawings()[idx];
    if (!d) return null;
    const ts = _lwChart.timeScale();
    const x1 = _xForPoint(ts, d.p1), y1 = candleSeries.priceToCoordinate(d.p1.price);
    const x2 = _xForPoint(ts, d.p2), y2 = candleSeries.priceToCoordinate(d.p2.price);
    if (x1 != null && y1 != null && Math.hypot(x - x1, y - y1) <= _HANDLE_R) return 'p1';
    if (x2 != null && y2 != null && Math.hypot(x - x2, y - y2) <= _HANDLE_R) return 'p2';
    return null;
  }

  // ── Floating selection toolbar — color swatches + delete. Shown above the
  // selected object (below it if there's no room above), never as a click
  // action on the object itself, per industry convention. ────────────────────
  function _hideDrawToolbar() {
    if (_drawToolbarEl) { _drawToolbarEl.remove(); _drawToolbarEl = null; }
  }

  function _positionDrawToolbar() {
    if (!_drawToolbarEl || _selectedIdx < 0) return;
    const d = _curDrawings()[_selectedIdx];
    if (!d) return;
    const ts = _lwChart.timeScale();
    const x1 = _xForPoint(ts, d.p1), y1 = candleSeries.priceToCoordinate(d.p1.price);
    const x2 = _xForPoint(ts, d.p2), y2 = candleSeries.priceToCoordinate(d.p2.price);
    if (x1 == null || y1 == null || x2 == null || y2 == null) return;
    const rect = chartDiv.getBoundingClientRect();
    const midX = (x1 + x2) / 2;
    const topY = Math.min(y1, y2);
    const bottomY = Math.max(y1, y2);
    const barW = _drawToolbarEl.offsetWidth || 200;
    const barH = _drawToolbarEl.offsetHeight || 32;
    let top = rect.top + topY - barH - 10;
    if (top < 8) top = rect.top + bottomY + 10; // flip below if no room above
    let left = rect.left + midX - barW / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - barW - 8));
    _drawToolbarEl.style.left = left + 'px';
    _drawToolbarEl.style.top  = top + 'px';
  }

  function _showDrawToolbar() {
    _hideDrawToolbar();
    if (_selectedIdx < 0) return;
    const d = _curDrawings()[_selectedIdx];
    if (!d) return;

    const bar = document.createElement('div');
    bar.id = '_lw-draw-toolbar';
    bar.style.cssText = [
      'position:fixed;z-index:9999;display:flex;align-items:center;gap:6px;',
      'background:var(--head-bg);border:1px solid var(--border);border-radius:6px;',
      'box-shadow:0 8px 32px rgba(0,0,0,.7);padding:5px 7px;',
    ].join('');

    _SWATCH_COLORS.forEach(c => {
      const sw = document.createElement('button');
      sw.type = 'button';
      sw.title = 'Color';
      sw.style.cssText = 'width:14px;height:14px;border-radius:50%;cursor:pointer;padding:0;'
        + `background:${c};border:1px solid rgba(255,255,255,0.25);`
        + (d.color === c ? 'outline:2px solid var(--text,#e8e8e8);outline-offset:1px;' : '');
      sw.addEventListener('click', e => {
        e.stopPropagation();
        d.color = c;
        _saveDrawings();
        _renderDrawings();
        _showDrawToolbar();
      });
      bar.appendChild(sw);
    });

    // Custom color — native color picker so any exact shade is available,
    // not just the six presets.
    const customWrap = document.createElement('label');
    customWrap.title = 'Custom color';
    customWrap.style.cssText = 'width:14px;height:14px;border-radius:50%;border:1px dashed var(--text3);'
      + 'position:relative;display:inline-block;cursor:pointer;overflow:hidden;';
    const customInput = document.createElement('input');
    customInput.type = 'color';
    customInput.style.cssText = 'position:absolute;inset:-4px;width:22px;height:22px;border:none;padding:0;cursor:pointer;opacity:0;';
    customInput.addEventListener('input', e => {
      const hex = e.target.value;
      const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
      d.color = `rgba(${r},${g},${b},0.9)`;
      _saveDrawings();
      _renderDrawings();
    });
    customWrap.appendChild(customInput);
    bar.appendChild(customWrap);

    const divider = document.createElement('div');
    divider.style.cssText = 'width:1px;height:16px;background:var(--border);margin:0 2px;';
    bar.appendChild(divider);

    const del = document.createElement('button');
    del.type = 'button';
    del.title = 'Delete';
    del.style.cssText = 'width:20px;height:20px;border:none;background:transparent;color:var(--text3);'
      + 'cursor:pointer;display:flex;align-items:center;justify-content:center;border-radius:4px;';
    del.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-.87 13.14A2 2 0 0 1 16.14 21H7.86a2 2 0 0 1-1.99-1.86L5 6"/></svg>';
    del.addEventListener('mouseenter', () => { del.style.background = 'rgba(255,99,99,0.15)'; del.style.color = '#ff6363'; });
    del.addEventListener('mouseleave', () => { del.style.background = 'transparent'; del.style.color = 'var(--text3)'; });
    del.addEventListener('click', e => {
      e.stopPropagation();
      _curDrawings().splice(_selectedIdx, 1);
      _selectedIdx = -1;
      _saveDrawings();
      _renderDrawings();
      _hideDrawToolbar();
    });
    bar.appendChild(del);

    bar.addEventListener('mousedown', e => e.stopPropagation());
    bar.addEventListener('click', e => e.stopPropagation());

    document.body.appendChild(bar);
    _drawToolbarEl = bar;
    _positionDrawToolbar();
  }

  function _cancelDrawMode() {
    _drawMode = null; _drawAnchor = null; _drawLiveEnd = null; _isDragging = false;
    _setChartPannable(true);
    _updateDrawBtnState();
    _renderDrawings();
  }

  // Convert a raw pointer-event client position into chart {time, price}.
  // BUGFIX (2026-07-29): the previous version sourced drag coordinates from
  // `subscribeCrosshairMove`, but LWC's own pan-handling latches onto a
  // pressed-mouse-move gesture at the moment `pointerdown` reaches its
  // internal (target-phase) listener -- which fires *before* our own
  // bubble-phase listener on `chartDiv` could disable panning -- and while
  // that internal pan/drag state is active, `crosshairMove` stops firing.
  // The result: the anchor and the drag-end point were both taken from the
  // same single (pre-drag) crosshair reading, so every drawing committed
  // with p1 === p2 (zero-length) — visible as a single collapsed dot for a
  // Trend Line, a stack of overlapping labels for Fibonacci (all 7 levels
  // at the same price), and literally nothing for a Rectangle (0×0 box).
  // It only ever looked like it "worked" after leaving and re-entering the
  // chart because that generates a fresh crosshairMove call once panning
  // (mistakenly already active) released control.
  // Fixed two ways: (1) coordinates are now computed directly from the
  // pointer event's own clientX/clientY against chartDiv's bounding rect —
  // no dependency on crosshairMove firing at all; (2) `pointerdown` is
  // registered on the CAPTURE phase, so panning is disabled before LWC's
  // own target-phase handler ever sees the event.
  function _clientToPixel(clientX, clientY) {
    try {
      const rect = chartDiv.getBoundingClientRect();
      return { x: clientX - rect.left, y: clientY - rect.top };
    } catch(_) { return null; }
  }
  // Extrapolates a synthetic bar time for a logical index beyond the last
  // real (or live today-) bar — e.g. index 3 bars past the last candle.
  // H1/H4 bars carry a plain UTCTimestamp (seconds), so stepping forward is
  // just adding whole intervals. D1/W1/MN bars carry a 'YYYY-MM-DD' string
  // (see _lwLoadChart's W1/MN aggregation and the D1 JSON itself), so we
  // step with real date arithmetic instead — D1 skips Sat/Sun since FX
  // doesn't trade those days, matching how the next *real* D1 bar would
  // actually land.
  function _extrapolateTimeForIndex(data, lastIdx, targetIdx) {
    const stepsAhead = targetIdx - lastIdx;
    if (stepsAhead <= 0) return null;
    const lastTime = data[lastIdx].time;
    if (_lwActiveTf === 'H1' || _lwActiveTf === 'H4') {
      const stepSec = _lwActiveTf === 'H1' ? 3600 : 14400;
      return lastTime + stepsAhead * stepSec;
    }
    if (typeof lastTime !== 'string') return null; // guard: unexpected shape
    let d = new Date(lastTime + 'T00:00:00Z');
    for (let i = 0; i < stepsAhead; i++) {
      if (_lwActiveTf === 'MN') {
        d.setUTCMonth(d.getUTCMonth() + 1);
      } else if (_lwActiveTf === 'W1') {
        d.setUTCDate(d.getUTCDate() + 7);
      } else {
        do { d.setUTCDate(d.getUTCDate() + 1); } while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
      }
    }
    return d.toISOString().slice(0, 10);
  }

  // Resolves a pixel x-coordinate to a chart time, extending past the last
  // real bar. LWC's own coordinateToTime() only resolves coordinates that
  // land on an actual data point (plus a short internally-extrapolated
  // stretch inside the rightOffset margin) and returns null past that —
  // which is what silently blocked drawing/moving objects to the right of
  // "now". TradingView/MT5/Bloomberg all allow free drawing into that empty
  // future space, so when coordinateToTime comes back null we fall back to
  // coordinateToLogical() (which is defined continuously, with no data-range
  // limit) and derive a synthetic time from it. The absolute logical index is
  // kept on the point (futureIndex) so it stays pinned to that exact future
  // slot even as new real bars arrive and fill in behind it.
  // ── Universal drawing-point time ────────────────────────────────────────────
  // A drawing's {time, price} points are stored as unix epoch seconds (UTC),
  // independent of whatever timeframe was active when the point was created.
  // D1/W1/MN bars use 'YYYY-MM-DD' business-day strings; H1/H4 bars use unix
  // seconds directly — timeToCoordinate() only accepts whichever format the
  // ACTIVE chart's series was built with, so every read/write converts through
  // this pair of helpers. This is what lets one object (trend line, Fib,
  // rectangle) show at the same real-world time/price on every timeframe —
  // D1, W1, MN, H1, H4 — not just the group it was originally drawn on.
  function _timeToEpoch(t) {
    if (t == null) return null;
    if (typeof t === 'number') return t; // H1/H4: already unix seconds
    if (typeof t === 'string') { const ms = Date.parse(t + 'T00:00:00Z'); return Number.isNaN(ms) ? null : Math.floor(ms / 1000); }
    return null;
  }
  function _epochToActiveTime(epochSec) {
    if (epochSec == null) return null;
    if (_lwActiveTf === 'H1' || _lwActiveTf === 'H4') return epochSec;
    return new Date(epochSec * 1000).toISOString().slice(0, 10);
  }

  function _resolveTimeAt(ts, x) {
    const time = ts.coordinateToTime(x);
    if (time != null) return { time: _timeToEpoch(time) };
    try {
      const logical = ts.coordinateToLogical(x);
      if (logical == null) return null;
      const data = candleSeries.data();
      if (!data || !data.length) return null;
      const lastIdx = data.length - 1;
      const futureIndex = Math.round(logical);
      if (futureIndex <= lastIdx) return null;
      const t = _extrapolateTimeForIndex(data, lastIdx, futureIndex);
      if (t == null) return null;
      // futureIndex is a logical bar-count position, meaningful only on the
      // timeframe it was created on (bar density differs across TFs) — kept
      // as a same-session fallback only; see _xForPoint.
      return { time: _timeToEpoch(t), futureIndex };
    } catch(_) { return null; }
  }

  // ── Epoch → x-coordinate, interpolated across bar spacing ──────────────────
  // timeToCoordinate() only resolves a Time that exactly matches an existing
  // bar on the ACTIVE series. That's fine when a point is read back on the
  // same timeframe it was drawn on (D1 epoch → D1 has a bar at that exact
  // date), but breaks the moment the active timeframe has different bar
  // density/placement — a W1 bar is stamped on one specific weekday, an H4
  // bar every 4 hours, so a D1-drawn epoch almost never lands on an exact W1
  // or H4 bar time, and timeToCoordinate() silently returns null (no error —
  // the shape just doesn't render). This is the actual reason drawings still
  // vanished across timeframes after v8.86.5's storage fix.
  // Fix, matching how TradingView/Bloomberg-style terminals anchor a drawing
  // to real time regardless of the active series' bar spacing: binary-search
  // the active series' own bars (converted to epoch via _timeToEpoch, so it
  // works whether the active series uses 'YYYY-MM-DD' strings or unix
  // seconds) for the two bars bracketing the target epoch, then interpolate
  // BETWEEN THEIR PIXEL COORDINATES. A direct timeToCoordinate() call is
  // still tried first as a fast path for the common exact-match case.
  //
  // v8.86.11 CORRECTION: the original version of this function passed a
  // fractional logical index (e.g. 150.57) straight into logicalToCoordinate()
  // on the assumption that it's defined continuously between bars, matching
  // its own doc comment. It is NOT, in this library version — its internal
  // implementation guards on Number.isInteger() and returns a bare 0 (not
  // null, not an error) for any non-integer input. That is the actual,
  // confirmed (via browser console) reason W1/MN drawings rendered at x=0:
  // every fallback call here was silently coerced to zero. Root-caused by
  // reading lightweight-charts' own source (TimeScale.qt / logicalToCoordinate)
  // and confirmed against live browser output showing x1=x2=0 exactly on
  // both W1 and MN while y1/y2 varied normally.
  // Fix: only ever pass INTEGER logical indices to logicalToCoordinate()
  // (always valid), and do the fractional interpolation ourselves in pixel
  // space — which is equivalent given bar spacing is uniform in pixels
  // between any two adjacent bars, and correct even when it's the two
  // bars flanking a boundary bar (index 0 or n-1) used for extrapolation.
  function _logicalToX(ts, logicalInt) {
    return ts.logicalToCoordinate(logicalInt);
  }
  function _interpX(ts, loIdx, hiIdx, frac) {
    const xLo = _logicalToX(ts, loIdx), xHi = _logicalToX(ts, hiIdx);
    if (xLo == null || xHi == null) return null;
    return xLo + (xHi - xLo) * frac;
  }
  function _epochToXInterpolated(ts, epochSec) {
    if (epochSec == null) return null;
    const direct = ts.timeToCoordinate(_epochToActiveTime(epochSec));
    if (direct != null) return direct;
    try {
      const data = candleSeries.data();
      if (!data || data.length < 2) return null;
      const n = data.length;
      const e0 = _timeToEpoch(data[0].time);
      const eN = _timeToEpoch(data[n - 1].time);
      if (epochSec <= e0) {
        const e1 = _timeToEpoch(data[1].time);
        const step = e1 - e0;
        if (!step) return null;
        return _interpX(ts, 0, 1, -(e0 - epochSec) / step);
      }
      if (epochSec >= eN) {
        const eNm1 = _timeToEpoch(data[n - 2].time);
        const step = eN - eNm1;
        if (!step) return _logicalToX(ts, n - 1);
        return _interpX(ts, n - 2, n - 1, 1 + (epochSec - eN) / step);
      }
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (_timeToEpoch(data[mid].time) <= epochSec) lo = mid; else hi = mid;
      }
      const eLo = _timeToEpoch(data[lo].time), eHi = _timeToEpoch(data[hi].time);
      const frac = eHi > eLo ? (epochSec - eLo) / (eHi - eLo) : 0;
      return _interpX(ts, lo, hi, frac);
    } catch (_) { return null; }
  }

  // Mirror of _resolveTimeAt for rendering: converts a stored point back to
  // an x-coordinate via _epochToXInterpolated (covers the common case, the
  // cross-timeframe case, and the case where a future point's slot has since
  // been filled by a real bar); falls back to raw logicalToCoordinate() on
  // futureIndex for points still out in undrawn future space beyond the last
  // bar on either side. That fallback only makes sense on the same timeframe
  // the point was created on (futureIndex is a bar-count position, not a
  // real-world time), so a future-space point viewed on a different
  // timeframe simply won't render until a real bar catches up to it on that
  // timeframe too — a rare edge case, not the normal drawn-on-real-history case.
  function _xForPoint(ts, pt) {
    if (!pt) return null;
    const x = _epochToXInterpolated(ts, pt.time);
    if (x != null) return x;
    if (pt.futureIndex != null) { try { return ts.logicalToCoordinate(pt.futureIndex); } catch(_) { return null; } }
    return null;
  }

  function _drawPointToTP(clientX, clientY) {
    const px = _clientToPixel(clientX, clientY);
    if (!px) return null;
    try {
      const ts    = _lwChart.timeScale();
      const price = candleSeries.coordinateToPrice(px.y);
      if (price == null) return null;
      const r = _resolveTimeAt(ts, px.x);
      if (!r) return null;
      return { time: r.time, price, futureIndex: r.futureIndex };
    } catch(_) { return null; }
  }

  // Press-and-drag to draw: pointerdown arms the anchor, pointermove updates
  // the live preview continuously, pointerup (anywhere, even released outside
  // the chart) commits the shape — the same click-drag convention as
  // TradingView/MT5. When no tool is armed, this same pointerdown handler
  // instead drives selection/move/resize of an existing object (see below) —
  // a plain click never deletes anything.
  chartDiv.addEventListener('pointerdown', e => {
    if (_drawMode) {
      const pt = _drawPointToTP(e.clientX, e.clientY);
      if (!pt) return;
      _drawAnchor  = pt;
      _drawLiveEnd = pt;
      _isDragging  = true;
      _setChartPannable(false); // must run before this event reaches LWC's own handler — see capture:true below
      try { e.preventDefault(); } catch(_) {}
      return;
    }

    // No tool armed: this is a selection / move / resize gesture, not creation.
    const px = _clientToPixel(e.clientX, e.clientY);
    if (!px) return;
    const arr = _curDrawings();

    // A handle on the current selection always wins over a body/move grab.
    if (_selectedIdx >= 0 && arr[_selectedIdx]) {
      const handle = _hitTestHandle(_selectedIdx, px.x, px.y);
      if (handle) {
        _dragMode = 'resize-' + handle;
        _setChartPannable(false);
        try { e.preventDefault(); } catch(_) {}
        return;
      }
    }

    const hitIdx = _hitTestDrawing(px.x, px.y);
    if (hitIdx >= 0) {
      // Select (switching selection if a different object was hit) and arm a
      // move-drag in the same gesture — a plain click with no movement simply
      // leaves the object selected, matching TradingView/MT5 behavior.
      _selectedIdx = hitIdx;
      _dragMode    = 'move';
      _dragStartPx = px;
      _dragOrigP1  = { time: arr[hitIdx].p1.time, price: arr[hitIdx].p1.price, futureIndex: arr[hitIdx].p1.futureIndex };
      _dragOrigP2  = { time: arr[hitIdx].p2.time, price: arr[hitIdx].p2.price, futureIndex: arr[hitIdx].p2.futureIndex };
      _setChartPannable(false);
      _renderDrawings();
      _showDrawToolbar();
      try { e.preventDefault(); } catch(_) {}
    } else if (_selectedIdx >= 0) {
      // Clicked empty space — deselect and let the click behave normally
      // (chart panning is untouched, since we never disabled it here).
      _selectedIdx = -1;
      _renderDrawings();
      _hideDrawToolbar();
    }
  }, true); // capture phase — runs ahead of LWC's own mousedown/pan handling

  // Global pointermove: de-duplicated document-level listener (same reasoning
  // as pointerup below) so the preview keeps updating even if the pointer
  // briefly leaves the chart bounds mid-drag.
  if (window._lwDrawDocPointerMove) document.removeEventListener('pointermove', window._lwDrawDocPointerMove);
  window._lwDrawDocPointerMove = function(e) {
    if (_isDragging) {
      const pt = _drawPointToTP(e.clientX, e.clientY);
      if (!pt) return;
      _drawLiveEnd = pt;
      _renderDrawings();
      return;
    }
    if (!_dragMode || _selectedIdx < 0) return;
    const d = _curDrawings()[_selectedIdx];
    if (!d) { _dragMode = null; return; }
    const ts = _lwChart.timeScale();
    if (_dragMode === 'move') {
      const px = _clientToPixel(e.clientX, e.clientY);
      if (!px || !_dragStartPx || !_dragOrigP1 || !_dragOrigP2) return;
      // Translate in pixel space, then convert back — avoids arithmetic on
      // BusinessDay time objects (D1/W1/MN), which aren't numeric. _xForPoint/
      // _resolveTimeAt (see above _drawPointToTP) extend this past the last
      // real bar so a shape can be dragged freely into future/empty space,
      // same as TradingView/MT5/Bloomberg.
      const ox1 = _xForPoint(ts, _dragOrigP1), oy1 = candleSeries.priceToCoordinate(_dragOrigP1.price);
      const ox2 = _xForPoint(ts, _dragOrigP2), oy2 = candleSeries.priceToCoordinate(_dragOrigP2.price);
      if (ox1 == null || oy1 == null || ox2 == null || oy2 == null) return;
      const dx = px.x - _dragStartPx.x, dy = px.y - _dragStartPx.y;
      const r1 = _resolveTimeAt(ts, ox1 + dx), np1 = candleSeries.coordinateToPrice(oy1 + dy);
      const r2 = _resolveTimeAt(ts, ox2 + dx), np2 = candleSeries.coordinateToPrice(oy2 + dy);
      if (!r1 || !r2 || np1 == null || np2 == null) return;
      d.p1 = { time: r1.time, price: np1, futureIndex: r1.futureIndex };
      d.p2 = { time: r2.time, price: np2, futureIndex: r2.futureIndex };
      _renderDrawings();
    } else if (_dragMode === 'resize-p1' || _dragMode === 'resize-p2') {
      const pt = _drawPointToTP(e.clientX, e.clientY);
      if (!pt) return;
      if (_dragMode === 'resize-p1') d.p1 = pt; else d.p2 = pt;
      _renderDrawings();
    }
  };
  document.addEventListener('pointermove', window._lwDrawDocPointerMove);

  // Global pointerup: a single de-duplicated document-level listener so a drag
  // released outside the chart bounds still commits, and so re-rendering the
  // chart (symbol/timeframe switch) doesn't accumulate stale listeners.
  if (window._lwDrawDocPointerUp) document.removeEventListener('pointerup', window._lwDrawDocPointerUp);
  window._lwDrawDocPointerUp = function(e) {
    if (_dragMode) {
      _dragMode = null;
      _dragStartPx = null; _dragOrigP1 = null; _dragOrigP2 = null;
      _setChartPannable(true);
      _saveDrawings();
      _renderDrawings();
      if (_selectedIdx >= 0) _positionDrawToolbar();
      return;
    }
    if (!_isDragging) return;
    _isDragging = false;
    _setChartPannable(true);
    const anchor = _drawAnchor;
    const end = _drawPointToTP(e.clientX, e.clientY) || _drawLiveEnd;
    _drawAnchor = null; _drawLiveEnd = null;
    if (!anchor || !end) { _renderDrawings(); return; }
    // Require a minimum on-screen drag distance so an accidental single
    // click doesn't commit a zero-size shape — the tool just stays armed.
    try {
      const ts = _lwChart.timeScale();
      const ax = _xForPoint(ts, anchor), ay = candleSeries.priceToCoordinate(anchor.price);
      const bx = _xForPoint(ts, end),    by = candleSeries.priceToCoordinate(end.price);
      const dist = (ax != null && ay != null && bx != null && by != null) ? Math.hypot(bx - ax, by - ay) : 0;
      if (dist < 8) { _renderDrawings(); return; }
    } catch(_) {}
    _curDrawings().push({
      id: 'dr_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      type: _drawMode,
      p1: anchor,
      p2: end,
      color: _DRAW_COLORS[_drawMode],
    });
    _saveDrawings();
    _drawMode = null;
    _updateDrawBtnState();
    _renderDrawings();
  };
  document.addEventListener('pointerup', window._lwDrawDocPointerUp);

  // Esc cancels an armed (or mid-drag) tool without drawing anything —
  // same de-duplication approach as the pointerup listener above.
  if (window._lwDrawEscHandler) document.removeEventListener('keydown', window._lwDrawEscHandler);
  window._lwDrawEscHandler = function(e) {
    if (e.key === 'Escape') {
      if (_drawMode) { _cancelDrawMode(); return; }
      if (_selectedIdx >= 0) { _selectedIdx = -1; _renderDrawings(); _hideDrawToolbar(); }
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (_selectedIdx < 0) return;
      // Don't hijack the key while the user is typing somewhere else on the page.
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
      _curDrawings().splice(_selectedIdx, 1);
      _selectedIdx = -1;
      _saveDrawings();
      _renderDrawings();
      _hideDrawToolbar();
      return;
    }
    // Ctrl+C / Cmd+C — copy the selected object (TradingView/MT5 convention:
    // objects support the same clipboard shortcuts as any other on-screen
    // element). Stored on window so it survives symbol/timeframe switches,
    // matching MT5's object clipboard behaviour.
    if ((e.key === 'c' || e.key === 'C') && (e.ctrlKey || e.metaKey)) {
      if (_selectedIdx < 0) return;
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
      const d = _curDrawings()[_selectedIdx];
      if (!d) return;
      window._lwDrawClipboard = JSON.parse(JSON.stringify(d));
      e.preventDefault();
      return;
    }
    // Ctrl+V / Cmd+V — paste the last copied object. Nudged by a fixed pixel
    // offset (down-right) so the duplicate doesn't land exactly on top of the
    // original and look like nothing happened — same convention as pasting a
    // duplicate shape in PowerPoint/Illustrator, or an object in MT5. Works
    // into future/empty space too via the same _resolveTimeAt fallback used
    // for drawing and moving.
    if ((e.key === 'v' || e.key === 'V') && (e.ctrlKey || e.metaKey)) {
      if (!window._lwDrawClipboard) return;
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
      const src = window._lwDrawClipboard;
      const ts  = _lwChart.timeScale();
      const PASTE_OFFSET_PX = 24;
      let p1 = { time: src.p1.time, price: src.p1.price, futureIndex: src.p1.futureIndex };
      let p2 = { time: src.p2.time, price: src.p2.price, futureIndex: src.p2.futureIndex };
      const x1 = _xForPoint(ts, src.p1), y1 = candleSeries.priceToCoordinate(src.p1.price);
      const x2 = _xForPoint(ts, src.p2), y2 = candleSeries.priceToCoordinate(src.p2.price);
      if (x1 != null && y1 != null && x2 != null && y2 != null) {
        const r1 = _resolveTimeAt(ts, x1 + PASTE_OFFSET_PX);
        const r2 = _resolveTimeAt(ts, x2 + PASTE_OFFSET_PX);
        const np1 = candleSeries.coordinateToPrice(y1 + PASTE_OFFSET_PX);
        const np2 = candleSeries.coordinateToPrice(y2 + PASTE_OFFSET_PX);
        if (r1 && r2 && np1 != null && np2 != null) {
          p1 = { time: r1.time, price: np1, futureIndex: r1.futureIndex };
          p2 = { time: r2.time, price: np2, futureIndex: r2.futureIndex };
        }
      }
      const copy = {
        id: 'dr_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        type: src.type,
        p1, p2,
        color: src.color,
      };
      _curDrawings().push(copy);
      _selectedIdx = _curDrawings().length - 1;
      _saveDrawings();
      _renderDrawings();
      _showDrawToolbar();
      e.preventDefault();
    }
  };
  document.addEventListener('keydown', window._lwDrawEscHandler);

  // Selection now lives entirely in the pointerdown handler above (which also
  // covers move/resize); a plain chart click never deletes a drawing anymore —
  // deletion is only available via the floating toolbar's trash icon or the
  // Delete/Backspace key above.

  // Draw menu — mirrors the Indicators dropdown UX pattern for consistency
  let _drawDropdownOpen = false;
  function _closeDrawDropdown() {
    const p = document.getElementById('_lw-draw-dropdown');
    if (p) p.remove();
    _drawDropdownOpen = false;
    const b = document.getElementById('lw-draw-btn');
    if (b) b.setAttribute('aria-expanded', 'false');
  }
  function _openDrawDropdown() {
    if (_drawDropdownOpen) { _closeDrawDropdown(); return; }
    _closeDrawDropdown();
    _drawDropdownOpen = true;
    const btn = document.getElementById('lw-draw-btn');
    if (btn) btn.setAttribute('aria-expanded', 'true');

    const pop = document.createElement('div');
    pop.id = '_lw-draw-dropdown';
    pop.style.cssText = [
      'position:fixed;z-index:9999;background:var(--head-bg);border:1px solid var(--border);',
      'border-radius:6px;box-shadow:0 8px 32px rgba(0,0,0,.7);',
      'font-size:11px;font-family:var(--font-ui,sans-serif);min-width:210px;max-width:260px;',
    ].join('');

    function _addOption(label, desc, onClick) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;flex-direction:column;padding:7px 12px;cursor:pointer;border-bottom:1px solid rgba(42,46,57,0.3);';
      row.innerHTML = `<div style="color:var(--text);font-weight:600;font-size:11px">${label}</div>`
                     + `<div style="color:var(--text3);font-size:9px;margin-top:1px">${desc}</div>`;
      row.addEventListener('mouseenter', () => row.style.background = 'rgba(255,255,255,0.04)');
      row.addEventListener('mouseleave', () => row.style.background = 'transparent');
      row.addEventListener('click', e => { e.stopPropagation(); onClick(); _closeDrawDropdown(); });
      pop.appendChild(row);
    }
    _addOption('Trend Line', 'Click and drag on the chart', () => {
      _selectedIdx = -1; _hideDrawToolbar();
      _drawMode = 'trend'; _drawAnchor = null; _drawLiveEnd = null; _updateDrawBtnState(); _renderDrawings();
    });
    _addOption('Fibonacci Retracement', 'Drag from the swing start to the swing end', () => {
      _selectedIdx = -1; _hideDrawToolbar();
      _drawMode = 'fib'; _drawAnchor = null; _drawLiveEnd = null; _updateDrawBtnState(); _renderDrawings();
    });
    _addOption('Fibonacci Extension', 'Drag from the swing start to the swing end — projects 127.2/161.8/200/261.8% continuation targets', () => {
      _selectedIdx = -1; _hideDrawToolbar();
      _drawMode = 'fibext'; _drawAnchor = null; _drawLiveEnd = null; _updateDrawBtnState(); _renderDrawings();
    });
    _addOption('Rectangle', 'Drag to mark a price/time zone', () => {
      _selectedIdx = -1; _hideDrawToolbar();
      _drawMode = 'rect'; _drawAnchor = null; _drawLiveEnd = null; _updateDrawBtnState(); _renderDrawings();
    });
    if (_curDrawings().length > 0) {
      _addOption('Clear All Drawings', `Remove ${_curDrawings().length} drawing(s) on this chart`, () => {
        window._lwDrawings[_drawSymKey] = [];
        _selectedIdx = -1;
        _hideDrawToolbar();
        _saveDrawings();
        _renderDrawings();
      });
    }

    const footer = document.createElement('div');
    footer.style.cssText = 'padding:6px 12px;color:var(--text3);font-size:9px;line-height:1.4;border-top:1px solid rgba(42,46,57,0.3);';
    footer.textContent = 'Click a drawing to select it. Drag its body to move, drag an endpoint to resize. Color and delete appear in a toolbar above the selection. Ctrl+C / Ctrl+V copies and pastes it.';
    pop.appendChild(footer);

    document.body.appendChild(pop);
    if (btn) {
      const rect = btn.getBoundingClientRect();
      const popW = pop.offsetWidth || 210;
      let left = rect.left;
      if (left + popW > window.innerWidth - 8) left = window.innerWidth - popW - 8;
      const popH = pop.offsetHeight || 130;
      const spaceBelow = window.innerHeight - rect.bottom;
      const top = spaceBelow >= popH + 8 ? rect.bottom + 4 : rect.top - popH - 4;
      pop.style.top  = Math.max(8, top) + 'px';
      pop.style.left = Math.max(8, left) + 'px';
    }
    pop.addEventListener('click',     e => e.stopPropagation());
    pop.addEventListener('mousedown', e => e.stopPropagation());
    // Same guard as the Indicators dropdown (see note there) — don't let a
    // mousedown on the toggle button itself close-then-immediately-reopen.
    setTimeout(() => {
      document.addEventListener('mousedown', function _outsideClose(e) {
        const b = document.getElementById('lw-draw-btn');
        if (b && b.contains(e.target)) return;
        _closeDrawDropdown();
      }, { once: true });
    }, 0);
  }

  // Attach dropdown handler — clone to clear prior listeners (same pattern as Indicators button)
  (function _attachDrawBtn() {
    const btn = document.getElementById('lw-draw-btn');
    if (!btn) return;
    const fresh = btn.cloneNode(true);
    btn.parentNode.replaceChild(fresh, btn);
    fresh.addEventListener('click', e => { e.stopPropagation(); _openDrawDropdown(); });
  })();

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
  // ── Pivot Points helpers ────────────────────────────────────────────────────
  // Groups bars into calendar day/ISO-week/month buckets so each period's
  // classic pivot levels can be computed from the PRIOR period's H/L/C (the
  // standard convention — a day's pivots are derived from yesterday's range).
  // 'D' uses a simple UTC calendar-day boundary (matches the existing VWAP
  // session-reset convention above); 'W' uses ISO week numbering (Mon-Sun);
  // 'M' uses UTC calendar month.
  // A period pivot (Daily/Weekly/Monthly) can only be drawn as isolated
  // per-period flat segments when the chart's own bar granularity is
  // strictly finer than that period. When it isn't (e.g. Daily Pivot viewed
  // on a Daily chart), EVERY bar is its own period, so there is no bar left
  // to sacrifice for a whitespace break without losing the period's only
  // data point — the result is every bar's differing level connected
  // straight to the next, i.e. the continuous diagonal zigzag Santiago
  // flagged (screenshot on an AUD/USD D1 chart with Daily Pivot on).
  // Matches the standard MT5/TradingView convention of only exposing a
  // period pivot on timeframes below that period.
  const _PIVOT_TF_RANK = { H1: 0, H4: 1, D1: 2, W1: 3, MN: 4 };
  const _PIVOT_MAX_TF  = { D: 1, W: 2, M: 3 }; // max allowed rank = strictly finer than the period
  function _pivotTfOk(unit) {
    const rank = _PIVOT_TF_RANK[_lwActiveTf];
    return rank !== undefined && rank <= _PIVOT_MAX_TF[unit];
  }
  function _iPivotPeriodKey(rawT, unit) {
    // bar.time is a 'YYYY-MM-DD' business-day string on D1/W1/MN and a plain
    // unix-seconds number on H1/H4 (see the "Universal drawing-point time"
    // note above) — always normalize through _timeToEpoch first. Skipping
    // this made `t * 1000` silently produce NaN for every D1/W1/MN bar
    // (string * number = NaN → Invalid Date), so every bar fell into the
    // same "NaN" period key and pivots rendered as one flat line for the
    // entire chart instead of per-day/week/month segments.
    const t = _timeToEpoch(rawT);
    const d = new Date(t * 1000);
    if (unit === 'D') return Math.floor(t / 86400);
    if (unit === 'M') return d.getUTCFullYear() * 12 + d.getUTCMonth();
    // ISO week: shift to the Thursday of the same week, then count weeks from
    // that year's first Thursday — the standard ISO-8601 week algorithm.
    const dt = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
    const dayNum = (dt.getUTCDay() + 6) % 7; // Mon=0..Sun=6
    dt.setUTCDate(dt.getUTCDate() - dayNum + 3);
    const firstThursday = new Date(Date.UTC(dt.getUTCFullYear(), 0, 4));
    const ftDayNum = (firstThursday.getUTCDay() + 6) % 7;
    firstThursday.setUTCDate(firstThursday.getUTCDate() - ftDayNum + 3);
    const weekNum = 1 + Math.round((dt - firstThursday) / (7 * 86400000));
    return dt.getUTCFullYear() * 100 + weekNum;
  }
  // Aggregates bars into period buckets, one entry per distinct period in
  // chronological order: { key, high, low, close, count }.
  function _iAggregatePeriods(bars, unit) {
    const periods = [];
    let cur = null;
    bars.forEach(b => {
      const key = _iPivotPeriodKey(b.time, unit);
      if (!cur || cur.key !== key) {
        cur = { key, high: b.high, low: b.low, close: b.close, count: 1 };
        periods.push(cur);
      } else {
        cur.high = Math.max(cur.high, b.high);
        cur.low = Math.min(cur.low, b.low);
        cur.close = b.close; // most recent close seen so far in this period
        cur.count++;
      }
    });
    return periods;
  }
  // Classic (floor trader) pivot formula.
  function _iPivotLevels(H, L, C) {
    const PP = (H + L + C) / 3;
    return {
      R3: H + 2 * (PP - L), R2: PP + (H - L), R1: 2 * PP - L,
      PP,
      S1: 2 * PP - H, S2: PP - (H - L), S3: L - 2 * (H - PP),
    };
  }
  // Builds the 7-series overlay data for a pivot indicator: for every bar,
  // look up its period key and plot that period's levels (derived from the
  // PRIOR period's aggregate H/L/C) — bars in the first period on file are
  // skipped since there is no prior period to derive levels from.
  // Industry-standard pivot display (TradingView/MT5) draws each period's
  // levels as its OWN isolated horizontal segment — a new day gets a new
  // flat line, not a continuation of yesterday's. A whitespace (time-only,
  // no value) point breaks the line at a period boundary so Lightweight
  // Charts stops connecting one period's segment to the next.
  function _calcPivotSeries(bars, unit, id, dec) {
    // See _pivotTfOk above: a period pivot needs bars strictly finer than
    // its own period to render as isolated segments at all. Returning empty
    // here (instead of the previous single-bar-per-period zigzag) is the
    // same "no meaningful line to draw" outcome _buildIndicatorPane already
    // handles silently for empty series lists.
    if (!_pivotTfOk(unit)) return [];
    const periods = _iAggregatePeriods(bars, unit);
    if (periods.length < 2) return [];
    const levelsByKey = {}, countByKey = {};
    for (let i = 1; i < periods.length; i++) {
      levelsByKey[periods[i].key] = _iPivotLevels(periods[i-1].high, periods[i-1].low, periods[i-1].close);
      countByKey[periods[i].key] = periods[i].count;
    }
    const fields = ['R3','R2','R1','PP','S1','S2','S3'];
    const out = {}; fields.forEach(f => out[f] = []);
    let curKey = null;
    for (let i = 0; i < bars.length; i++) {
      const key = _iPivotPeriodKey(bars[i].time, unit);
      const lv = levelsByKey[key];
      if (!lv) continue; // first period on file: no prior H/L/C to derive levels from
      // The break is placed on THIS bar's own existing time slot — there is
      // no arithmetic on bar.time here (it's a 'YYYY-MM-DD' string on
      // D1/W1/MN and a plain number on H1/H4; adding a number to that
      // string silently concatenates into a garbage time value instead of
      // throwing, corrupting every later series' time ordering), so the
      // only type-safe place for a whitespace point is a bar we're willing
      // to give up entirely. That's only safe when the incoming period has
      // more than one bar left to still show its flat level afterward —
      // e.g. Daily pivots viewed ON a Daily chart have exactly one bar per
      // period, so gapping there would silently drop every other value.
      // Those stay directly connected: a single thin one-bar-wide diagonal
      // at the transition, not the original bug (a flat line spanning the
      // whole chart, which was actually _iPivotPeriodKey returning the same
      // NaN key for every bar — fixed separately above).
      if (curKey !== null && key !== curKey && countByKey[key] > 1) {
        fields.forEach(f => out[f].push({ time: bars[i].time }));
        curKey = key;
        continue;
      }
      fields.forEach(f => out[f].push({ time: bars[i].time, value: parseFloat(lv[f].toFixed(dec)) }));
      curKey = key;
    }
    const unitLabel = unit === 'D' ? 'D' : (unit === 'W' ? 'W' : 'M');
    // Segment start: the first real (non-whitespace) point of the LATEST
    // (current) period run only — i.e. only the most recent flat segment
    // on the right side of the chart gets a tag, one per level. v8.90.4
    // tagged every historical segment (matching TradingView's per-segment
    // placement literally), but with 7 levels × many periods on screen at
    // once that reads as noise rather than signal — Santiago asked for a
    // single current-value tag per line instead, so only the last start
    // point is kept here.
    const segStartsByField = {};
    fields.forEach(f => {
      const arr = out[f];
      let lastStart = null;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i].value === undefined) continue;
        const prev = arr[i - 1];
        if (i === 0 || !prev || prev.value === undefined) {
          lastStart = { time: arr[i].time, value: arr[i].value };
        }
      }
      segStartsByField[f] = lastStart ? [lastStart] : [];
    });
    return fields.map((f, i) => ({
      data: out[f], color: _iC(id, i), lineWidth: 1, dashed: f === 'PP',
      label: `${unitLabel} ${f}`,
      title: `${unitLabel}${f}`,
      segStarts: segStartsByField[f],
    }));
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
    { id:'vwap',     group:'Overlays',        label:'VWAP',              desc:'Volume-Weighted Avg Price (daily sessions)',             type:'overlay',    defaultParams:{},                              paramDefs:[], colors:['#ff5722'], volRequired:true },
    { id:'bb',       group:'Overlays',        label:'Bollinger Bands',   desc:'Bollinger Bands',                                        type:'overlay',    defaultParams:{ period:20, mult:2 },           paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1},{key:'mult',label:'Mult',type:'float',min:0.1,max:10,step:0.1}], colors:['rgba(33,150,243,0.5)','rgba(33,150,243,0.9)','rgba(33,150,243,0.9)'] },
    { id:'keltner',  group:'Overlays',        label:'Keltner Channel',   desc:'Keltner Channel',                                        type:'overlay',    defaultParams:{ period:20, mult:1.5 },         paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1},{key:'mult',label:'Mult',type:'float',min:0.1,max:10,step:0.1}], colors:['rgba(255,152,0,0.5)','rgba(255,152,0,0.9)','rgba(255,152,0,0.9)'] },
    { id:'donchian', group:'Overlays',        label:'Donchian Channel',  desc:'Donchian Channel',                                       type:'overlay',    defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:500,step:1}], colors:['rgba(156,39,176,0.8)','rgba(156,39,176,0.8)','rgba(156,39,176,0.4)'] },
    { id:'psar',     group:'Overlays',        label:'Parabolic SAR',     desc:'Parabolic SAR',                                          type:'overlay',    defaultParams:{ step:0.02, max:0.2 },          paramDefs:[{key:'step',label:'Step',type:'float',min:0.001,max:0.1,step:0.001},{key:'max',label:'Max AF',type:'float',min:0.01,max:0.5,step:0.01}], colors:['#f44336'] },
    { id:'ichimoku', group:'Overlays',        label:'Ichimoku Cloud',    desc:'Ichimoku Kinko Hyo · 9/26/52',                          type:'overlay',    defaultParams:{},                              paramDefs:[], colors:['#26a69a','#ef5350','rgba(38,166,154,0.3)','rgba(239,83,80,0.3)','rgba(120,123,134,0.4)'] },
    { id:'supertrend', group:'Overlays',      label:'Supertrend',       desc:'ATR-based trend-following overlay — flips support/resistance on trend change', type:'overlay', defaultParams:{ period:10, mult:3 }, paramDefs:[{key:'period',label:'ATR Period',type:'int',min:1,max:100,step:1},{key:'mult',label:'Multiplier',type:'float',min:0.5,max:10,step:0.1}], colors:['#26a69a','#ef5350'] },
    { id:'pivotd',   group:'Overlays',        label:'Pivot Points (Daily)',   desc:'Classic pivot, R1-R3 / S1-S3 — computed from the prior day\'s H/L/C', type:'overlay', defaultParams:{}, paramDefs:[], colors:['#ef5350','#ff7043','#ffab91','#9e9e9e','#a5d6a7','#66bb6a','#26a69a'] },
    { id:'pivotw',   group:'Overlays',        label:'Pivot Points (Weekly)',  desc:'Classic pivot, R1-R3 / S1-S3 — computed from the prior week\'s H/L/C', type:'overlay', defaultParams:{}, paramDefs:[], colors:['#ef5350','#ff7043','#ffab91','#9e9e9e','#a5d6a7','#66bb6a','#26a69a'] },
    { id:'pivotm',   group:'Overlays',        label:'Pivot Points (Monthly)', desc:'Classic pivot, R1-R3 / S1-S3 — computed from the prior month\'s H/L/C', type:'overlay', defaultParams:{}, paramDefs:[], colors:['#ef5350','#ff7043','#ffab91','#9e9e9e','#a5d6a7','#66bb6a','#26a69a'] },
    // ── Oscillators ───────────────────────────────────────────────────────────
    { id:'rsi',      group:'Oscillators',     label:'RSI',               desc:'Relative Strength Index',                                type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#64b5f6'] },
    { id:'stoch',    group:'Oscillators',     label:'Stochastic',        desc:'Stochastic Oscillator',                                  type:'oscillator', defaultParams:{ k:14, d:3, smooth:3 },         paramDefs:[{key:'k',label:'%K',type:'int',min:1,max:100,step:1},{key:'smooth',label:'Smooth',type:'int',min:1,max:20,step:1},{key:'d',label:'%D',type:'int',min:1,max:20,step:1}], colors:['#2196f3','#ff9800'] },
    { id:'macd',     group:'Oscillators',     label:'MACD',              desc:'MACD',                                                   type:'oscillator', defaultParams:{ fast:12, slow:26, signal:9 },  paramDefs:[{key:'fast',label:'Fast',type:'int',min:2,max:100,step:1},{key:'slow',label:'Slow',type:'int',min:2,max:200,step:1},{key:'signal',label:'Signal',type:'int',min:1,max:50,step:1}], colors:['#26a69a','#2196f3','#ff9800'], histoIdx:[0] },
    { id:'cci',      group:'Oscillators',     label:'CCI',               desc:'Commodity Channel Index',                                type:'oscillator', defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#64b5f6'] },
    { id:'willr',    group:'Oscillators',     label:'Williams %R',       desc:'Williams %R',                                            type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#64b5f6'] },
    { id:'roc',      group:'Oscillators',     label:'ROC',               desc:'Rate of Change',                                         type:'oscillator', defaultParams:{ period:12 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#4caf50'] },
    { id:'mom',      group:'Oscillators',     label:'Momentum',          desc:'Momentum',                                               type:'oscillator', defaultParams:{ period:10 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#9c27b0'] },
    { id:'mfi',      group:'Oscillators',     label:'MFI',               desc:'Money Flow Index (uses volume)',                         type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#03a9f4'], volRequired:true },
    { id:'ao',       group:'Oscillators',     label:'Awesome Oscillator',desc:'Awesome Oscillator · 5/34',                              type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#26a69a'], histoIdx:[0] },
    { id:'trix',     group:'Oscillators',     label:'TRIX',              desc:'Triple Smoothed EMA',                                    type:'oscillator', defaultParams:{ period:18 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#64b5f6'] },
    { id:'dpo',      group:'Oscillators',     label:'DPO',               desc:'Detrended Price Oscillator',                             type:'oscillator', defaultParams:{ period:21 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#64b5f6'] },
    { id:'uo',       group:'Oscillators',     label:'Ultimate Osc.',     desc:'Ultimate Oscillator · 7/14/28',                          type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#64b5f6'] },
    // ── Volatility ────────────────────────────────────────────────────────────
    { id:'atr',      group:'Volatility',      label:'ATR',               desc:'Average True Range',                                     type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:1,max:200,step:1}], colors:['#64b5f6'] },
    { id:'adx',      group:'Volatility',      label:'ADX / DMI',         desc:'Average Directional Index + DI±',                        type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#64b5f6','#26a69a','#ef5350'] },
    { id:'aroon',    group:'Volatility',      label:'Aroon',             desc:'Aroon Up/Down',                                          type:'oscillator', defaultParams:{ period:25 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:200,step:1}], colors:['#26a69a','#ef5350'] },
    { id:'chop',     group:'Volatility',      label:'Choppiness',        desc:'Choppiness Index',                                       type:'oscillator', defaultParams:{ period:14 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#64b5f6'] },
    // ── Volume ────────────────────────────────────────────────────────────────
    { id:'obv',      group:'Volume',          label:'OBV',               desc:'On-Balance Volume',                                      type:'oscillator', defaultParams:{},                              paramDefs:[], colors:['#64b5f6'], volRequired:true },
    { id:'cmf',      group:'Volume',          label:'CMF',               desc:'Chaikin Money Flow',                                     type:'oscillator', defaultParams:{ period:20 },                   paramDefs:[{key:'period',label:'Period',type:'int',min:2,max:100,step:1}], colors:['#00acc1'], volRequired:true },
  ];

  // ── Active indicator state (persists across symbol switches) ─────────────────
  // ── Persistent state — survives page reloads via localStorage ────────────────
  const _LS_IND   = 'gi_ind_state';   // { id: bool }
  const _LS_PARAMS = 'gi_ind_params'; // { id: { param: val } }
  const _LS_MA    = 'gi_ma_list';     // [ { uid, type, period, color, lineWidth, lineStyle } ]
  const _LS_LEVELS = 'gi_ind_levels'; // { id: [v0, v1, ...] } — user-edited oscillator reference levels

  // Default reference-level values per oscillator, in on-screen (ascending or
  // logical) order — the single source of truth both _calcIndData() (via
  // _mkRefs below) and the Levels edit UI read from. Editing a level in the
  // Indicators ▾ dropdown overrides the matching index in window._lwIndLevels;
  // an override array is only honored if its length matches the defaults
  // (protects against stale localStorage from an older indicator version).
  const _IND_LEVEL_DEFAULTS = {
    rsi:[30,50,70], stoch:[20,50,80], cci:[-100,0,100], willr:[-80,-50,-20],
    mfi:[20,50,80], uo:[30,50,70], chop:[38.2,61.8],
    macd:[0], roc:[0], mom:[0], ao:[0], trix:[0], dpo:[0], cmf:[0],
    adx:[25], aroon:[50],
  };

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
  if (typeof window._lwIndLevels === 'undefined') window._lwIndLevels = _lsGet(_LS_LEVELS,  {});

  // Save helpers — call after any mutation
  function _saveIndState()  { _lsSet(_LS_IND,    window._lwIndState);  }
  function _saveIndParams() { _lsSet(_LS_PARAMS,  window._lwIndParams); }
  function _saveMaList()    { _lsSet(_LS_MA,      window._lwMaList);    }
  function _saveIndLevels() { _lsSet(_LS_LEVELS,  window._lwIndLevels); }

  // Effective reference levels for an indicator: user override (if present
  // and the right length) else the built-in defaults.
  function _iLevels(id) {
    const defaults = _IND_LEVEL_DEFAULTS[id] || [];
    const ov = window._lwIndLevels[id];
    return (Array.isArray(ov) && ov.length === defaults.length) ? ov : defaults.slice();
  }
  // Builds a refs[] array (as consumed by _addRefLines) from the indicator's
  // current effective levels, paired positionally with the given colors.
  function _mkRefs(id, colors) {
    return _iLevels(id).map((v, i) => ({ v, color: colors[i] }));
  }

  const _maSeries = {}; // uid → series object
  // Active pane indices — keyed by indicator id, reset each render (chart destroyed).
  // Exposed on window (below) so the top-level fullscreen/resize handlers — which
  // live outside this closure — can re-apply pane heights after chart.resize().
  const _indPaneIndex = {}; // id → pane index number (oscillators only)
  window._indPaneIndex = _indPaneIndex;
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
    return custom[i] || defaults[i] || _themeColor('--text3');
  }

  // ── Calculation functions — one per indicator id ───────────────────────────

  function _calcIndData(id, bars) {
    const closes = bars.map(b => b.close);
    const highs  = bars.map(b => b.high);
    const lows   = bars.map(b => b.low);
    const vols   = bars.map(b => b.volume || 0);
    const p      = _iP(id); // effective params (defaults + user overrides)

    // Volume-based indicators (OBV, CMF, MFI, VWAP) produce a flat/degenerate
    // series when every bar's volume is 0 — matches the dropdown row being
    // disabled for symbols with no volume data (see volRequired in the
    // catalogue). Bail out here too so a symbol switch away from a
    // volume-having symbol can't leave a stale flat pane on-screen.
    const _cfgVolReq = (_IND_CATALOGUE.find(c => c.id === id) || {}).volRequired;
    if (_cfgVolReq && !hasVolume) return [];

    switch (id) {
      case 'ma': {
        // MA indicator now renders via _buildMaSeries, not _calcIndData
        // Return an empty stub so _buildIndicatorPane doesn't fail
        return [];
      }
      case 'vwap': {
        // BUG FIX (2026-07-29): this previously accumulated cumTPV/cumV across the
        // ENTIRE loaded bars array with no reset, so on any chart holding more than
        // one session (which is every timeframe except a single intraday day) the
        // running average degenerates into what looks like an extremely long-period
        // moving average — flat and unresponsive — instead of a daily VWAP. This is
        // also why it looked visibly wrong switching to H1: more bars accumulate
        // before the average can move, making the flattening more obvious.
        // Fix: reset the cumulative sums at every UTC calendar-day boundary, matching
        // the indicator's own catalogue description ("VWAP · daily sessions").
        // bar.time is a unix timestamp (seconds) — see loader comments above.
        const typicals = bars.map((b, i) => ({ t: b.time, tp: (b.high+b.low+b.close)/3, v: vols[i] }));
        let cumTPV = 0, cumV = 0, curDay = null;
        const data = typicals.map(({ t, tp, v }) => {
          const day = Math.floor(t / 86400); // UTC calendar day index
          if (day !== curDay) { curDay = day; cumTPV = 0; cumV = 0; }
          cumTPV += tp*v; cumV += v;
          return { time:t, value: cumV>0 ? cumTPV/cumV : tp };
        });
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
      case 'supertrend': {
        // Standard ATR-based flip line (matches the widely-used reference
        // implementation): two candidate bands (up = support candidate,
        // dn = resistance candidate) are each "ratcheted" — they can only
        // move in the trend's favor while the trend holds — and the trend
        // flips when price closes through the OPPOSITE band's prior value.
        const { period:n, mult } = p;
        const tr  = _iTR(bars);
        const atr = _iRMA(tr, n); // same length/index alignment as bars — see Keltner above
        let upBand = null, dnBand = null, trend = 1;
        // ROOT CAUSE OF THE "CHANNEL" BUG (found after ruling out both the
        // calculation and Service Worker caching — the trend-gated math was
        // always correct and byte-identical to what shipped): this used to
        // be TWO series (up/down) with a single time-only "whitespace"
        // point dropped in at each flip to try to create a gap. Lightweight
        // Charts' line series does NOT actually render a visual break for
        // whitespace data — per the library's own maintainer (GitHub issue
        // #700): "Whitespace doesn't mean gap actually right now. It means
        // there is no value for a series." The renderer still draws a
        // straight connecting stroke between the nearest two REAL points on
        // either side of any whitespace, no matter how many whitespace
        // entries sit between them. So every multi-week inactive stretch
        // was silently bridged by a straight line from the old segment's
        // last point to the new segment's first point — which, stacked
        // across dozens of flips, is exactly the solid "channel" reported.
        // Fix: build ONE independent LineSeries per contiguous trend run
        // (below, in _buildIndicatorPane) instead of relying on whitespace.
        // Separate series objects can never bridge each other, so this is
        // the only mechanism this library actually supports for true gaps.
        const segments = [];
        let current = null;
        for (let i = 0; i < bars.length; i++) {
          const hl2  = (bars[i].high + bars[i].low) / 2;
          const rawUp = hl2 - mult * atr[i];
          const rawDn = hl2 + mult * atr[i];
          if (i === 0) {
            upBand = rawUp; dnBand = rawDn; trend = 1;
          } else {
            const prevUp = upBand, prevDn = dnBand;
            const newUp = (bars[i-1].close > prevUp) ? Math.max(rawUp, prevUp) : rawUp;
            const newDn = (bars[i-1].close < prevDn) ? Math.min(rawDn, prevDn) : rawDn;
            if (trend === -1 && bars[i].close > prevDn) trend = 1;
            else if (trend === 1 && bars[i].close < prevUp) trend = -1;
            upBand = newUp; dnBand = newDn;
          }
          const val = trend === 1 ? upBand : dnBand;
          const point = { time: bars[i].time, value: parseFloat(val.toFixed(dec)) };
          if (!current || current.trend !== trend) {
            current = { trend, points: [] };
            segments.push(current);
          }
          current.points.push(point);
        }
        return segments.map((seg, si) => ({
          data: seg.points,
          color: _iC(id, seg.trend === 1 ? 0 : 1),
          lineWidth: 2,
          label: `Supertrend(${n},${mult})`,
          // Only the most recent (current) run gets the right-axis value
          // tag — matching Bloomberg/TradingView convention — so turning
          // this on doesn't spam one tag per historical flip.
          lastValueVisible: si === segments.length - 1,
        }));
      }
      case 'pivotd': return _calcPivotSeries(bars, 'D', id, dec);
      case 'pivotw': return _calcPivotSeries(bars, 'W', id, dec);
      case 'pivotm': return _calcPivotSeries(bars, 'M', id, dec);
      // ── Oscillators ─────────────────────────────────────────────────────────
      case 'rsi': {
        const n = p.period;
        const gains=[], losses=[];
        for(let i=1;i<closes.length;i++){const d=closes[i]-closes[i-1];gains.push(d>0?d:0);losses.push(d<0?-d:0);}
        const avgG=_iRMA(gains,n), avgL=_iRMA(losses,n);
        const data=avgG.map((g,i)=>{const l=avgL[i];const rs=l===0?Infinity:g/l;return{time:bars[i+1].time,value:parseFloat((l===0?100:100-100/(1+rs)).toFixed(2))};});
        return [{data,color:_iC(id,0),lineWidth:1,label:`RSI(${n})`,
          refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])}];
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
          {data:kData,color:_iC(id,0),lineWidth:1,label:`%K(${kPer},${smooth})`,refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])},
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
          {data:histD,color:_iC(id,0),lineWidth:0,label:'Hist',histogram:true,refs:_mkRefs(id,['rgba(120,123,134,0.2)'])},
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
          refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])}];
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
          refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])}];
      }
      case 'roc': {
        const n = p.period;
        const data=bars.slice(n).map((b,i)=>({time:b.time,value:parseFloat(((b.close-bars[i].close)/bars[i].close*100).toFixed(4))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`ROC(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.3)'])}];
      }
      case 'mom': {
        const n = p.period;
        const data=bars.slice(n).map((b,i)=>({time:b.time,value:parseFloat((b.close-bars[i].close).toFixed(dec))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`Mom(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.3)'])}];
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
          refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])}];
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
        return [{data,color:_iC(id,0),lineWidth:0,label:'AO',histogram:true,refs:_mkRefs(id,['rgba(120,123,134,0.2)'])}];
      }
      case 'trix': {
        const n = p.period;
        const e1=_iEMA(closes,n),e2=_iEMA(e1,n),e3=_iEMA(e2,n);
        const data=e3.slice(1).map((v,i)=>({time:bars[bars.length-e3.length+i+1].time,value:parseFloat(((v-e3[i])/e3[i]*100).toFixed(6))}));
        return [{data,color:_iC(id,0),lineWidth:1,label:`TRIX(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.3)'])}];
      }
      case 'dpo': {
        const n = p.period; const disp=Math.floor(n/2)+1;
        const sma=_iSMA(closes,n);
        const data=sma.map((v,i)=>{
          const barIdx=i+n-1-disp;
          if(barIdx<0) return null;
          return{time:bars[i+n-1].time,value:parseFloat((closes[barIdx]-v).toFixed(dec))};
        }).filter(Boolean);
        return [{data,color:_iC(id,0),lineWidth:1,label:`DPO(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.3)'])}];
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
          refs:_mkRefs(id,['rgba(158,161,170,0.35)','rgba(120,123,134,0.2)','rgba(158,161,170,0.35)'])}];
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
          {data:adx.map((v,i)=>({time:bars[off+i].time,value:parseFloat(v.toFixed(2))})),color:_iC(id,0),lineWidth:1,label:`ADX(${n})`,refs:_mkRefs(id,['rgba(158,161,170,0.35)'])},
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
          {data:up,color:_iC(id,0),lineWidth:1,label:`Aroon Up(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.2)'])},
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
          refs:_mkRefs(id,['rgba(38,166,154,0.3)','rgba(158,161,170,0.35)'])}];
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
        return [{data,color:_iC(id,0),lineWidth:1,label:`CMF(${n})`,refs:_mkRefs(id,['rgba(120,123,134,0.3)'])}];
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
      + 'font-size:10px;font-family:var(--font-mono,monospace);line-height:1.3;user-select:none;color:var(--text);';
    el.innerHTML = html;
    paneEl.appendChild(el);
  }

  // ── Pivot per-segment inline labels — TradingView-standard placement ──
  // Unlike the CB-meeting overlay (one date → one static label), a pivot
  // indicator needs a small text tag at the START of EVERY period segment
  // (R3..S3 × every day/week/month on file), not just the latest one.
  // Reuses the same SVG-overlay-synced-to-timeScale pattern as _drawCbLines
  // above: a transparent, pointer-events:none SVG sits over the chart div
  // and is redrawn from time/price coordinates on every pan/zoom.
  let _pivotLabelOverlay = null;
  const _pivotLabelData = {}; // indicator id → [{ time, value, color, text }]

  function _drawPivotLabels() {
    if (!_pivotLabelOverlay || !_lwChart) return;
    const ts = _lwChart.timeScale();
    let svgContent = '';
    Object.values(_pivotLabelData).forEach(entries => {
      entries.forEach(e => {
        try {
          const x = ts.timeToCoordinate(e.time);
          const y = candleSeries.priceToCoordinate(e.value);
          if (x == null || y == null || x < 0 || x > chartDiv.offsetWidth) return;
          svgContent += `<text x="${(x + 3).toFixed(1)}" y="${(y - 3).toFixed(1)}" `
            + `font-size="9" font-family="var(--font-mono,monospace)" fill="${e.color}" `
            + `font-weight="700">${e.text}</text>`;
        } catch(_) {}
      });
    });
    _pivotLabelOverlay.innerHTML = svgContent;
  }
  function _ensurePivotLabelOverlay() {
    if (_pivotLabelOverlay) return;
    _pivotLabelOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    _pivotLabelOverlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;overflow:visible;';
    chartDiv.style.position = 'relative';
    chartDiv.appendChild(_pivotLabelOverlay);
    _lwChart.timeScale().subscribeVisibleTimeRangeChange(_drawPivotLabels);
  }
  // Registers/clears one indicator id's label entries and redraws. Called
  // with an empty seriesList to clear (indicator destroyed or toggled off).
  function _setPivotLabels(id, seriesList) {
    const withSeg = (seriesList || []).filter(s => s.segStarts && s.segStarts.length);
    if (withSeg.length === 0) { delete _pivotLabelData[id]; }
    else {
      _pivotLabelData[id] = [];
      withSeg.forEach(s => {
        s.segStarts.forEach(pt => {
          _pivotLabelData[id].push({ time: pt.time, value: pt.value, color: s.color, text: s.title });
        });
      });
    }
    _ensurePivotLabelOverlay();
    _drawPivotLabels();
  }

  function _addRefLines(paneIndex, refs, ownerSeries) {
    if (!refs || paneIndex === null || !ownerSeries) return;
    const lines = [];
    refs.forEach(ref => {
      try {
        const pl = ownerSeries.createPriceLine({
          price: ref.v, color: ref.color, lineWidth: 1, lineStyle: 2,
          axisLabelVisible: false, title: '',
        });
        lines.push(pl);
      } catch(_) {}
    });
    // Track { ownerSeries, lines } so they can be removed via removePriceLine()
    // when the indicator is destroyed — native price lines belong to the
    // series that created them, not the pane, so the owner must be kept too.
    _indRefSeries[paneIndex] = { ownerSeries, lines };
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
            // Point series (e.g. PSAR) — dots only, no connecting line.
            // BUG FIX: `lineWidth: 0` does NOT hide the line — LWC's LineWidth
            // type is a union clamped to 1|2|3|4, so 0 silently falls back to
            // a visible 1px stroke, which is why PSAR rendered as a solid
            // curve instead of discrete dots. The actual API for a dots-only
            // series is `lineVisible: false` + `pointMarkersVisible: true`.
            series = _lwChart.addSeries(LWC.LineSeries, {
              color: s.color, lineVisible: false,
              pointMarkersVisible: true, pointMarkersRadius: 2,
              priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: true,
              priceFormat: { type: 'price', precision: dec, minMove },
            }, paneIndex);
          } else {
            series = _lwChart.addSeries(LWC.LineSeries, {
              color: s.color, lineWidth: s.lineWidth || 1,
              lineStyle: s.dashed ? 2 : 0,
              priceLineVisible: false,
              lastValueVisible: s.lastValueVisible !== undefined ? s.lastValueVisible : si === 0,
              crosshairMarkerVisible: false,
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

          // Reference lines — only for first series in a sub-pane. Native
          // price lines (see _addRefLines) always span the full pane width,
          // so they never lag behind the last bar the way a data-bound line
          // series could.
          if (!isOverlay && si === 0 && s.refs) {
            _addRefLines(paneIndex, s.refs, series);
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

      // Inline per-segment labels (Pivot Points only — see _setPivotLabels).
      // No-op for any indicator whose series don't carry `segStarts`.
      _setPivotLabels(id, seriesList);
    } catch(e) { console.warn('[LW] indicator build error for', id, e); }
  }

  function _destroyIndicatorPane(id) {
    const cfg = _IND_CATALOGUE.find(c => c.id === id);
    const isOverlay = cfg && cfg.type === 'overlay';

    // Clear any inline pivot segment labels for this id (no-op for
    // non-pivot ids). Done unconditionally here, not only inside
    // _buildIndicatorPane's success path, so a rebuild that ends up with an
    // empty seriesList (e.g. a period pivot blocked by _pivotTfOk on the
    // current timeframe) doesn't leave stale labels from a prior timeframe.
    _setPivotLabels(id, []);

    // Remove all series for this indicator (works for both overlays and oscillators)
    if (_indSeries[id]) {
      _indSeries[id].forEach(s => {
        try { _lwChart.removeSeries(s); } catch(_) {}
      });
      _indSeries[id] = null;
    }

    // Remove ref lines for oscillator panes
    if (!isOverlay && _indPaneIndex[id] != null) {
      const refEntry = _indRefSeries[_indPaneIndex[id]];
      if (refEntry) {
        refEntry.lines.forEach(pl => { try { refEntry.ownerSeries.removePriceLine(pl); } catch(_) {} });
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

  // Re-apply any persisted compare overlays for this freshly-built chart —
  // same "survives a full chart rebuild" pattern as the indicator engine
  // just above. Previously compare had zero persistence at all: leaving and
  // returning to the chart (or even just switching timeframe, which also
  // destroys+rebuilds the LW chart instance) silently dropped every compare
  // series while its pill stayed on-screen — reading as "still comparing"
  // when nothing was actually drawn. _lwLoadCompare is defined further down
  // this file as a top-level function (module-scope _lwChart/_lwCandleSeries/
  // _lwCompareSeriesMap, already set by this point in this function).
  (window._lwCompareList || []).slice().forEach(function (entry) {
    if (entry.cmpType === 'ohlc' && entry.cmpId === ohlcId) {
      // Can't compare a symbol with itself (same guard as the dropdown
      // click handler) — this entry stays in the persisted list (it's a
      // valid compare against any *other* symbol), just not drawn on this
      // particular chart. Clear its now-stale pill so it doesn't read as
      // active with nothing behind it.
      document.querySelectorAll('.lw-cmp-pill').forEach(function (p) {
        if (p.dataset.uid === entry.uid) p.remove();
      });
      return;
    }
    _lwLoadCompare(entry.cmpId, entry.cmpLabel, entry.cmpType, true);
  });

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
      pill.style.cssText = 'display:inline-flex;align-items:center;gap:3px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:9px;font-family:var(--font-ui,sans-serif);white-space:nowrap;';
      const dot = `<span style="width:6px;height:6px;border-radius:50%;background:${ma.color};display:inline-block;flex-shrink:0"></span>`;
      pill.innerHTML = `${dot}<span style="color:var(--text2)">${ma.type} ${ma.period}</span>`;
      const rm = document.createElement('span');
      rm.textContent = '\u00d7';
      rm.style.cssText = 'color:var(--text3);cursor:pointer;font-size:10px;margin-left:1px;';
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
      pill.style.cssText = 'display:inline-flex;align-items:center;gap:3px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:9px;font-family:var(--font-ui,sans-serif);white-space:nowrap;';
      pill.innerHTML = `<span style="color:var(--text2)">${cfg.label}</span>`;
      const rm = document.createElement('span');
      rm.textContent = '\u00d7';
      rm.style.cssText = 'color:var(--text3);cursor:pointer;font-size:10px;margin-left:1px;';
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

  function _openIndDropdown(preserveScrollTop) {
    if (_indDropdownOpen) { _closeIndDropdown(); return; }
    _closeIndDropdown();
    _indDropdownOpen = true;

    const btn = document.getElementById('lw-ind-btn');
    if (btn) btn.setAttribute('aria-expanded', 'true');

    const pop = document.createElement('div');
    pop.id = '_lw-ind-dropdown';
    pop.style.cssText = [
      'position:fixed;z-index:9999;background:var(--head-bg);border:1px solid var(--border);',
      'border-radius:6px;box-shadow:0 8px 32px rgba(0,0,0,.7);',
      'font-size:11px;font-family:var(--font-ui,sans-serif);',
      'min-width:300px;max-height:520px;overflow-y:auto;',
      'scrollbar-width:thin;scrollbar-color:var(--border) transparent;',
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
    maHeader.style.cssText = 'padding:8px 12px 4px;color:var(--text3);font-size:9px;letter-spacing:.08em;font-weight:700;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;';
    maHeader.innerHTML = '<span>MOVING AVERAGES</span>';

    const addMaBtn = document.createElement('button');
    addMaBtn.textContent = '+ Add MA';
    addMaBtn.style.cssText = 'background:var(--accent);color:#fff;border:none;border-radius:3px;padding:2px 7px;font-size:9px;font-weight:600;cursor:pointer;letter-spacing:.04em;';
    addMaBtn.addEventListener('mouseenter', () => addMaBtn.style.background = _themeColor('--chart-line') + 'cc');
    addMaBtn.addEventListener('mouseleave', () => addMaBtn.style.background = 'var(--accent)');
    addMaBtn.addEventListener('click', e => {
      e.stopPropagation();
      const newMa = { uid: _genMaUid(), type:'EMA', period:20, color:_nextColor(), lineWidth:1, lineStyle:0 };
      window._lwMaList.push(newMa);
      _saveMaList();
      _buildMaSeries(newMa);
      _renderIndPills();
      _updateIndBtn();
      { const _st = pop.scrollTop; pop.remove(); _indDropdownOpen = false; _openIndDropdown(_st); }
    });
    maHeader.appendChild(addMaBtn);
    pop.appendChild(maHeader);

    if (window._lwMaList.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:10px 12px;color:var(--text3);font-size:10px;font-style:italic;';
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
      typeSelect.style.cssText = 'background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;';
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
      periodInput.style.cssText = 'width:44px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;text-align:center;';
      periodInput.addEventListener('click', e => e.stopPropagation());
      periodInput.addEventListener('change', e => {
        e.stopPropagation();
        const v = parseInt(e.target.value);
        if (v > 0 && v <= 500) { ma.period = v; _saveMaList(); _buildMaSeries(ma); _renderIndPills(); }
      });
      row.appendChild(periodInput);

      // Line style selector
      const styleSelect = document.createElement('select');
      styleSelect.style.cssText = 'background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;';
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
      widthSelect.style.cssText = 'background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;cursor:pointer;flex-shrink:0;width:50px;';
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
      rmBtn.style.cssText = 'background:none;border:none;color:var(--text3);cursor:pointer;font-size:14px;margin-left:auto;padding:0 2px;line-height:1;flex-shrink:0;';
      rmBtn.title = `Remove ${ma.type} ${ma.period}`;
      rmBtn.addEventListener('mouseenter', () => rmBtn.style.color = 'var(--down)');
      rmBtn.addEventListener('mouseleave', () => rmBtn.style.color = 'var(--text3)');
      rmBtn.addEventListener('click', e => {
        e.stopPropagation();
        window._lwMaList = window._lwMaList.filter(m => m.uid !== ma.uid);
        _saveMaList();
        _destroyMaSeries(ma.uid);
        _renderIndPills(); _updateIndBtn();
        { const _st = pop.scrollTop; pop.remove(); _indDropdownOpen = false; _openIndDropdown(_st); }
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
      header.style.cssText = 'padding:8px 12px 4px;color:var(--text3);font-size:9px;letter-spacing:.08em;font-weight:700;border-top:1px solid var(--border);';
      pop.appendChild(header);

      items.forEach(cfg => {
        // Period pivots (Daily/Weekly/Monthly) can't render meaningfully
        // once the chart's own timeframe is at or above their period — see
        // _pivotTfOk. Disable the row instead of letting the user turn on
        // an indicator that will silently draw nothing (or, before this
        // fix, the corrupted zigzag).
        const _pivotUnit = { pivotd: 'D', pivotw: 'W', pivotm: 'M' }[cfg.id];
        const pivotBlocked = !!_pivotUnit && !_pivotTfOk(_pivotUnit);
        const _PIVOT_TF_HINT = {
          D: 'Not available — Daily Pivots require an intraday timeframe (H1 or H4)',
          W: 'Not available — Weekly Pivots require H1, H4, or D1',
          M: 'Not available — Monthly Pivots require H1, H4, D1, or W1',
        };
        // Volume-based indicators (OBV, CMF, MFI, VWAP) render degenerate/flat
        // output on symbols with no real volume data (all fall back to 0),
        // per _calcIndData's `b.volume||0` handling — disable the row instead
        // of letting the user turn on an indicator that draws a meaningless flat line.
        const volBlocked = !!cfg.volRequired && !hasVolume;
        const tfBlocked = pivotBlocked || volBlocked;
        const isOn = !!window._lwIndState[cfg.id] && !tfBlocked;
        const hasParams = cfg.paramDefs.length > 0;
        const hasColors = cfg.colors.length > 0;
        const hasLevels = !!_IND_LEVEL_DEFAULTS[cfg.id];
        const expandable = isOn && (hasParams || hasColors || hasLevels);

        // ── Main toggle row ────────────────────────────────────
        const row = document.createElement('div');
        row.style.cssText = `display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:${tfBlocked?'not-allowed':'pointer'};opacity:${tfBlocked?'0.45':'1'};background:${isOn?'rgba(79,127,255,0.08)':'transparent'};border-bottom:${expandable?'none':'1px solid rgba(42,46,57,0.3)'};`;
        if (volBlocked) row.title = 'Not available — this symbol has no volume data';
        else if (pivotBlocked) row.title = _PIVOT_TF_HINT[_pivotUnit];
        row.addEventListener('mouseenter', () => { if (!isOn && !tfBlocked) row.style.background='rgba(255,255,255,0.04)'; });
        row.addEventListener('mouseleave', () => { row.style.background=isOn?'rgba(79,127,255,0.08)':'transparent'; });

        // Checkbox
        const check = document.createElement('div');
        check.style.cssText = `width:14px;height:14px;border-radius:3px;border:1px solid ${isOn?_themeColor('--chart-line'):_themeColor('--border2')};background:${isOn?_themeColor('--chart-line'):'transparent'};flex-shrink:0;display:flex;align-items:center;justify-content:center;`;
        if (isOn) check.innerHTML = '<svg width="8" height="6" viewBox="0 0 8 6" fill="none"><polyline points="1,3 3,5 7,1" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

        // Label + desc
        const left = document.createElement('div');
        left.style.cssText = 'flex:1;min-width:0;';
        left.innerHTML = `<div style="color:${isOn?_themeColor('--text'):_themeColor('--text2')};font-weight:${isOn?'600':'400'};font-size:11px">${cfg.label}</div>`
          + `<div style="color:var(--text3);font-size:9px;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${cfg.desc}</div>`;

        row.appendChild(check);
        row.appendChild(left);

        // Toggle click
        row.addEventListener('click', e => {
          e.stopPropagation();
          if (tfBlocked) return;
          window._lwIndState[cfg.id] = !window._lwIndState[cfg.id];
          _saveIndState();
          if (window._lwIndState[cfg.id]) { _buildIndicatorPane(cfg.id); } else { _destroyIndicatorPane(cfg.id); }
          _renderIndPills(); _updateIndBtn();
          { const _st = pop.scrollTop; pop.remove(); _indDropdownOpen = false; _openIndDropdown(_st); }
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
            inp.style.cssText = 'width:46px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;text-align:center;';
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

          // Reference levels (e.g. RSI's 30/50/70 overbought/oversold/mid
          // lines) — editable per industry convention (TradingView/MT5 both
          // expose these under the indicator's Levels settings). Rendered as
          // one small numeric input per level, in the same left-to-right
          // order the lines are defined in _IND_LEVEL_DEFAULTS.
          if (hasLevels) {
            const levels = _iLevels(cfg.id);
            levels.forEach((lv, li) => {
              const llbl = document.createElement('label');
              llbl.style.cssText = 'display:flex;align-items:center;gap:3px;color:#6b7280;font-size:9px;font-weight:600;letter-spacing:.03em;';
              llbl.textContent = levels.length > 1 ? `Lv${li + 1}` : 'Level';

              const linp = document.createElement('input');
              linp.type = 'number';
              linp.value = lv;
              linp.step = 'any';
              linp.title = 'Reference level — drawn as a dashed line across the full pane';
              linp.style.cssText = 'width:46px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:10px;text-align:center;';
              linp.addEventListener('click', e => e.stopPropagation());
              linp.addEventListener('change', e => {
                e.stopPropagation();
                const raw = parseFloat(e.target.value);
                if (isNaN(raw)) return;
                const next = _iLevels(cfg.id);
                next[li] = raw;
                window._lwIndLevels[cfg.id] = next;
                _saveIndLevels();
                if (window._lwIndState[cfg.id]) {
                  _destroyIndicatorPane(cfg.id);
                  _buildIndicatorPane(cfg.id);
                }
              });
              llbl.appendChild(linp);
              paramRow.appendChild(llbl);
            });
          }

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

    // Restore prior scroll position when this open() call is a rebuild triggered
    // by a toggle/param click inside the panel (see call sites below) — without
    // this, every click on an indicator row re-created `pop` from scratch and it
    // always mounted at scrollTop 0, so the list visibly jumped to the top.
    if (preserveScrollTop != null) pop.scrollTop = preserveScrollTop;

    // Position below the button
    if (btn) {
      const rect = btn.getBoundingClientRect();
      const popH = Math.min(520, pop.scrollHeight || 450);
      const spaceBelow = window.innerHeight - rect.bottom;
      const top = spaceBelow >= 80 ? rect.bottom + 4 : rect.top - popH - 4;
      // Clamp horizontally to the viewport — previously this only enforced an
      // 8px left margin, so on mobile (where the "Indicators" button sits near
      // the right edge of the toolbar) the panel's min-width:300px pushed its
      // right side past the viewport with no way to reach it (nothing scrolls
      // the fixed-position panel horizontally). Pull it back in from the right
      // as well, same 8px margin.
      const popW = pop.offsetWidth || 300;
      let left = rect.left;
      if (left + popW > window.innerWidth - 8) left = window.innerWidth - popW - 8;
      pop.style.top  = Math.max(8, top) + 'px';
      pop.style.left = Math.max(8, left) + 'px';
    }

    // Stop ALL clicks inside the popup from bubbling to document
    pop.addEventListener('click',     e => e.stopPropagation());
    pop.addEventListener('mousedown', e => e.stopPropagation());

    // Close when user clicks/mousedowns outside the popup — but not when the
    // mousedown target is the toggle button itself. mousedown fires before
    // click, so without this guard a tap on the button while the dropdown is
    // open would close it here first and then the button's own click handler
    // (which calls _openIndDropdown, i.e. toggle) would immediately reopen
    // it — the dropdown could never be closed by tapping the button again.
    setTimeout(() => {
      document.addEventListener('mousedown', function _outsideClose(e) {
        const b = document.getElementById('lw-ind-btn');
        if (b && b.contains(e.target)) return; // let the button's own click toggle handle it
        _closeIndDropdown();
      }, { once: true });
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
      const ohlcColor = isUp ? _themeColor('--up') : _themeColor('--down');
      const _isOHLCType = (_effectiveChartType(ohlcId) === 'candle' || _effectiveChartType(ohlcId) === 'bar');
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
    panelSub.textContent = _hasFinnhubLive ? 'Live' : 'Delayed ~5min';
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
    background:     'var(--bg2)',
    border:         '1px solid var(--border2)',
    borderRadius:   '4px',
    padding:        '6px 10px',
    fontSize:       '11px',
    lineHeight:     '1.5',
    fontFamily:     'var(--font-ui,sans-serif)',
    color:          'var(--text)',
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
        + ` <span style="color:var(--text3);font-size:10px;">${name}</span></span>`
        + `</div>`;
    }).join('');
    _cbTooltip.innerHTML =
      `<div style="font-size:9px;color:var(--text3);letter-spacing:.05em;margin-bottom:3px;">CB MEETING</div>`
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
        if (_lwChart && width > 0 && height > 0) { _lwChart.resize(width, height); _lwReapplyPaneHeights(); _lwReprojectDrawings(); }
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
    backgroundColor: _themeColor('--bg'), gridColor: _themeColorAlpha('--border', 0.8),
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
  const _LINE_STYLE_SYMS = new Set(['FRED:DGS10', 'FRED:BAMLH0A0HYM2', 'FRED:BAMLC0A0CM']);
  const chartStyle = _LINE_STYLE_SYMS.has(sym) ? '3' : '1';
  script.text = JSON.stringify({
    allow_symbol_change:false, calendar:false, details:true,
    hide_side_toolbar:true, hide_top_toolbar:true, hide_legend:false,
    hide_volume:true, interval:'D', locale:'en', save_image:false,
    style:chartStyle, symbol:sym, theme:'dark', timezone:'Etc/UTC',
    backgroundColor:_themeColor('--bg'), gridColor:_themeColorAlpha('--border', 0.8),
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
  if (!typeBtn || typeBtn.disabled) return;
  window._lwChartType = typeBtn.dataset.chartType;
  document.querySelectorAll('[data-chart-type]').forEach(b => {
    b.classList.toggle('sel', b === typeBtn);
    b.classList.remove('on');
  });
  // Immediately show/hide OHLC header — no need to wait for chart re-render
  const _effType = _effectiveChartType(_lwActiveOhlcId);
  const _ohlcWrap = document.getElementById('lw-hdr-ohlc-wrap');
  if (_ohlcWrap) _ohlcWrap.style.display = (_effType === 'candle' || _effType === 'bar') ? '' : 'none';
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

// Clean, single-line attribution for an ATM-IV tooltip, built from the raw
// `source` string fetch_intraday_quotes.py writes into fx_etf_iv (e.g.
// "CBOE ^JYVIX", "Saxo Bank FX Options Analytics (1M ATM, indicative mid)",
// "Barchart N6*0 options (aggregate) [cached]"). That raw string is an
// internal audit label — it can carry cache state, a scraped vendor name,
// and futures-continuation ticker codes that have no place in a user-facing
// tooltip (no terminal, Bloomberg included, discloses its own fallback
// plumbing or a scraped vendor's name to the end user — only the market
// source of the print). This maps it down to one clean, named source; it
// never lists the fallback order that produced it.
function _ivSourceLabel(raw) {
  if (!raw) return 'institutional options market';
  if (raw.startsWith('CBOE')) return raw.split(' ').slice(0, 2).join(' '); // "CBOE ^JYVIX"
  if (raw.startsWith('Saxo Bank')) return 'Saxo Bank FX Options Analytics';
  if (raw.includes('Barchart') || raw.includes('CME')) return 'CME futures options market';
  if (raw.startsWith('PHLX')) return 'PHLX World Currency Options';
  if (raw.startsWith('est. AUD')) return 'AUD/USD-derived proxy';
  if (raw.startsWith('est.')) return 'estimated proxy';
  return 'exchange-listed options market'; // ETF fallback tickers (FXE/FXB/…)
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
  const sessH = rt?.session_high ?? rt?.high ?? null;
  const sessL = rt?.session_low  ?? rt?.low  ?? null;

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
  let atmIvRank = null;
  let atmIvSource = null;
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
      atmIvRank = ivEntry.iv_rank ?? null;
      atmIvSource = ivEntry.source ?? null;
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

  // Sovereign bond yield spread — ΔY = Yield(base) − Yield(quote), same base/quote sign
  // convention as carryDiff above. 2Y preferred (short-end, best proxy for near-term
  // rate-expectations divergence); falls back to 10Y with an explicit tenor label when
  // either leg lacks 2Y coverage (JPY/NZD/NOK/SEK — see GUIDELINES.md). Never silently
  // mixes tenors. Mirrors updatePairDetail() exactly — same BOND_YIELD_CACHE, same fallback.
  // FIX-36 (v8.98.0): pick('y2') now also excludes a leg whose y2Stale flag is
  // set (backend-confirmed >90d-old cached value, e.g. CHF/EUR when SNB/ECB
  // feeds stop publishing) — falls through to the 10Y tenor instead of
  // building a spread on a year-old yield. See CHANGELOG.
  const bondBase  = base  ? (BOND_YIELD_CACHE[base]  || null) : null;
  const bondQuote = quote ? (BOND_YIELD_CACHE[quote] || null) : null;
  let bondTenor = null, bondDiff = null;
  if (bondBase && bondQuote) {
    const pick = (tenor) => (bondBase[tenor] != null && bondQuote[tenor] != null
      && !(tenor === 'y2' && (bondBase.y2Stale || bondQuote.y2Stale)))
      ? bondBase[tenor] - bondQuote[tenor] : null;
    const y2diff = pick('y2');
    if (y2diff != null) {
      bondTenor = '2Y'; bondDiff = y2diff;
    } else {
      const y10diff = pick('y10');
      if (y10diff != null) { bondTenor = '10Y'; bondDiff = y10diff; }
    }
    if (bondDiff != null && !meta?.cross) {
      bondDiff = invert ? bondDiff : -bondDiff;
    }
  }

  // RR — use pairId (ISO convention) not base+quote (PAIRS internal field)
  // base/quote in PAIRS represent commodity/money ccy, not ISO order.
  // e.g. usdjpy → base='JPY',quote='USD' → base+quote='JPYUSD' ≠ rr.json key 'USDJPY'.
  // pairId is always the ISO-convention name (no slash) matching rr.json keys exactly.
  const rrKey = pairId ? pairId.toUpperCase() : null;
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
  const ivCls = v => {
    if (v == null) return '';
    if (atmIvRank != null) return atmIvRank > 70 ? 'pd-dn' : atmIvRank < 30 ? 'pd-up' : '';
    return v > 12 ? 'pd-dn' : v < 7 ? 'pd-up' : '';
  };
  const hvCls = v => v == null ? '' : v > 12 ? 'pd-dn' : v < 7 ? 'pd-up' : '';

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
        <div class="pd-inline-group-lbl">Price &amp; Spreads</div>
        <div class="pd-inline-metrics">
          <div class="pd-inline-metric fx-tip" data-tip-title="1-Week Change" data-tip-body="Weekly % change vs prior Friday close.">
            <div class="pd-inline-lbl">1W Chg</div><div class="pd-inline-val ${cls(pct1w)}">${fmtP(pct1w)}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="Carry Differential" data-tip-body="OIS/overnight rate differential (OIS preferred; falls back to CB policy rate). Positive = carry favours long base currency." data-tip-ex="Long the higher-yielding currency, short the lower-yielding currency. OIS reflects actual overnight funding cost; CB policy rate is the ceiling.">
            <div class="pd-inline-lbl">Carry</div><div class="pd-inline-val ${clsI(carryDiff)}">${carryDiff != null ? (carryDiff >= 0 ? '+' : '') + carryDiff.toFixed(2) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="${bondTenor || '2Y'} Sovereign Bond Spread" data-tip-body="ΔY = Yield(${base || 'base'}) − Yield(${quote || 'quote'}) at the ${bondTenor || '2Y'} tenor. Short-end (2Y) yield differentials are the primary driver of sustained FX direction — they track near-term rate-expectations divergence more closely than 10Y, which reflects longer-run growth/inflation premia and duration flows.${bondTenor === '10Y' ? ' 2Y unavailable for one or both legs — showing 10Y as fallback.' : ''} Source: extended-data sovereign yield pipeline (FRED/ECB/BOE/BOC/SNB/DBnomics)." data-tip-ex="A rising ${bondTenor || '2Y'} spread in the base currency's favour has historically preceded sustained appreciation — it signals the market pricing in a widening policy-rate gap before central banks act.">
            <div class="pd-inline-lbl">${bondTenor || '2Y'} Spread</div><div class="pd-inline-val ${clsI(bondDiff)}">${bondDiff != null ? (bondDiff >= 0 ? '+' : '') + (bondDiff * 100).toFixed(0) + ' bp' : '—'}</div>
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
          <div class="pd-inline-metric fx-tip" data-tip-title="Historical Volatility 30d" data-tip-body="30-day realised volatility, annualised — how much the pair has actually moved. Low HV = quiet market; high HV = volatile market. No 52-week percentile is computed for realised vol, so color uses a fixed band: green ≤7% (quiet), red >12% (volatile). Not a directional signal.">
            <div class="pd-inline-lbl">HV 30d</div><div class="pd-inline-val ${hvCls(hv30)}">${hv30 != null ? hv30.toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="ATM Implied Volatility" data-tip-body="ATM (at-the-money) implied volatility — annualised, same variance-swap methodology as VIX where an exchange-computed index is available. Source: ${_ivSourceLabel(atmIvSource)}. ${atmIvRank != null ? 'Color = IV Rank vs 52-week range: green rank&lt;30 (historically cheap), red rank&gt;70 (historically expensive).' : 'Color = cost of hedging: green ≤7% (cheap), red >12% (expensive) — fixed band; IV Rank not yet available (needs ≥4 weeks history).'} Not a directional signal.">
            <div class="pd-inline-lbl">ATM IV</div><div class="pd-inline-val ${ivCls(atmIv)}">${atmIv != null ? atmIv.toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="IV minus HV" data-tip-body="Implied vol minus realised vol — the volatility risk premium (VRP). Positive = options pricing in more than recent realised moves (expensive, red). Negative = options cheap vs realised (green). IV running above HV is the market's normal state, since option sellers demand a premium — a negative spread is comparatively rare. Not a directional signal.">
            <div class="pd-inline-lbl">IV − HV</div><div class="pd-inline-val ${atmIv != null && hv30 != null ? clsI(hv30 - atmIv) : ''}">${atmIv != null && hv30 != null ? (atmIv > hv30 ? '+' : '') + (atmIv - hv30).toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="pd-inline-metric fx-tip" data-tip-title="25d Risk Reversal · Saxo Bank (1M)" data-tip-body="25-delta Risk Reversal, 1M tenor: 25d call IV minus 25d put IV. Source: Saxo Bank public options page, indicative mid-market. Positive = calls bid over puts, upside skew on ${base || label.split('/')[0]}. Negative = puts bid, downside protection dominant. Directional skew signal, not a vol-level signal.">
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

// ── Risk Monitor US HY OAS cell: click to open chart ──
document.getElementById('risk-hyoas')?.closest('.risk-cell')?.addEventListener('click', () => {
  loadTVChart('FRED:BAMLH0A0HYM2');
});

// ── Risk Monitor US IG OAS cell: click to open chart ──
document.getElementById('risk-igoas')?.closest('.risk-cell')?.addEventListener('click', () => {
  loadTVChart('FRED:BAMLC0A0CM');
});

// ═══════════════════════════════════════════════════════════════════
// PAIR DETAIL PANEL — Eikon-style linked panel, updates #pair-detail on every pair click
// All data read from in-memory caches — zero additional fetches on click.
// ═══════════════════════════════════════════════════════════════════
const COT_DATA_CACHE = {};   // ccy → { net, long, short, amNet, weekEnding, prevOI, wowNetChange, totalOI, levNetPctOI }
const RR_DATA_CACHE  = {};   // rrKey (e.g. 'EURUSD') → { rr25d: number } — populated by fetchOptionSkew()
const BOND_YIELD_CACHE = {}; // ccy → { y10: number|null, y2: number|null } — extended-data/{CCY}.json, file name == ccy code

(async function prefetchBondYields() {
  const CCYS = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];
  await Promise.all(CCYS.map(async ccy => {
    try {
      const r = await fetch('./extended-data/' + ccy + '.json');
      if (!r.ok) return;
      const j = await r.json();
      const d = j?.data ?? j;
      // y2Stale: fetch_bond_yields.py (FIX-36, v2.9.9) labels a cached bond2y
      // 'stale-cached' once it's >90d old and no live source is available
      // (e.g. CHF/EUR when SNB/ECB feeds stop publishing). Carried through
      // here so every downstream consumer of y2 can exclude it from spreads
      // instead of silently building a signal on a year-old yield.
      BOND_YIELD_CACHE[ccy] = {
        y10: d?.bond10y ?? null,
        y2: d?.bond2y ?? null,
        y2Stale: j?.sources?.bond2y === 'stale-cached',
      };
    } catch {}
  }));
})();

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
  // Use session_high/session_low (21:00 UTC FX session boundary) — same source as the FX pairs
  // table and updateFxPairsTableRT(). Falls back to high/low if session values are null.
  const sessH = rt?.session_high ?? rt?.high ?? null;
  const sessL = rt?.session_low  ?? rt?.low  ?? null;

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
  let atmIvRank = null;
  let atmIvSource = null;
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
      atmIvRank = ivEntry.iv_rank ?? null;
      atmIvSource = ivEntry.source ?? null;
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

  // Sovereign bond yield spread — ΔY = Yield(base) − Yield(quote), same base/quote
  // sign convention as carryDiff above. 2Y preferred (short-end, best proxy for near-term
  // rate-expectations divergence — the primary FX driver per RBA/BIS research); falls back
  // to 10Y with an explicit tenor label when either leg lacks 2Y coverage (JPY/NZD/NOK/SEK
  // currently have no free live 2Y source — see GUIDELINES.md). Never silently mixes tenors.
  // FIX-36 (v8.98.0): pick('y2') excludes a leg whose y2Stale flag is set — see
  // the identical fix in the analogous block above (row-badge spread calc).
  const bondBase  = base  ? (BOND_YIELD_CACHE[base]  || null) : null;
  const bondQuote = quote ? (BOND_YIELD_CACHE[quote] || null) : null;
  let bondTenor = null, bondDiff = null;
  if (bondBase && bondQuote) {
    const pick = (tenor) => (bondBase[tenor] != null && bondQuote[tenor] != null
      && !(tenor === 'y2' && (bondBase.y2Stale || bondQuote.y2Stale)))
      ? bondBase[tenor] - bondQuote[tenor] : null;
    const y2diff = pick('y2');
    if (y2diff != null) {
      bondTenor = '2Y'; bondDiff = y2diff;
    } else {
      const y10diff = pick('y10');
      if (y10diff != null) { bondTenor = '10Y'; bondDiff = y10diff; }
    }
    if (bondDiff != null && meta && !meta.cross) {
      bondDiff = invert ? bondDiff : -bondDiff;
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
  // Use pairId (ISO convention, e.g. 'usdjpy') not base+quote — PAIRS.base/quote are
  // commodity/money fields, not ISO order: usdjpy has base='JPY',quote='USD', so
  // base+quote='JPYUSD' which does not match rr.json key 'USDJPY'. pairId always matches.
  const rrKey  = pairId ? pairId.toUpperCase() : null;
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
      <div class="pd-section-lbl">Price &amp; Spreads</div>
      <div class="pd-grid">
        <div class="pd-cell fx-tip" data-tip-title="1-Week Change" data-tip-body="Weekly % change vs prior Friday close. Source: FX performance cache."><div class="pd-lbl">1W Chg</div><div class="pd-val ${cls(pct1w)}">${fmtPct(pct1w)}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Carry Differential" data-tip-body="OIS overnight rate differential (SOFR/€STR/SONIA/TONA/CORRA/SARON — institutional overnight benchmarks). Falls back to CB policy rate when OIS data unavailable. Positive = base currency yields more, carry favours long." data-tip-ex="Positive carry = the long leg earns more than it costs to fund the short. OIS reflects actual overnight funding cost — more accurate than CB policy rate for carry calculations. Carry is most reliable as a persistent trend signal; it can reverse quickly on policy surprises."><div class="pd-lbl">Carry</div><div class="pd-val ${cls(carryDiff)}">${carryDiff != null ? (carryDiff >= 0 ? '+' : '') + carryDiff.toFixed(2)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="${bondTenor || '2Y'} Sovereign Bond Spread" data-tip-body="ΔY = Yield(${base || 'base'}) − Yield(${quote || 'quote'}) at the ${bondTenor || '2Y'} tenor. Short-end (2Y) yield differentials are the primary driver of sustained FX direction — they track near-term rate-expectations divergence more closely than 10Y, which reflects longer-run growth/inflation premia and duration flows.${bondTenor === '10Y' ? ' 2Y unavailable for one or both legs — showing 10Y as fallback.' : ''} Source: extended-data sovereign yield pipeline (FRED/ECB/BOE/BOC/SNB/DBnomics)." data-tip-ex="A rising ${bondTenor || '2Y'} spread in the base currency's favour has historically preceded sustained appreciation — it signals the market pricing in a widening policy-rate gap before central banks act."><div class="pd-lbl">${bondTenor || '2Y'} Spread</div><div class="pd-val ${cls(bondDiff)}">${bondDiff != null ? (bondDiff >= 0 ? '+' : '') + (bondDiff * 100).toFixed(0)+' bp' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="Average Daily Range" data-tip-body="Estimated average daily range in pips, derived from HV 30d: close × (HV / √252). Indicates typical intraday movement — useful for stop and target sizing." data-tip-ex="ADR of 85 pip on EUR/USD means the pair moves ~85 pip on an average day."><div class="pd-lbl">ADR</div><div class="pd-val">${adr != null ? adr + ' pip' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="${base || 'Base'} Policy Rate" data-tip-body="${base || 'Base'} central bank policy rate (annualised). Source: CB rates cache."><div class="pd-lbl">${base || 'Base'} Rate</div><div class="pd-val">${cbBase != null ? cbBase.toFixed(2)+'%' : '—'}</div></div>
      </div>
    </div>

    <div class="pd-section">
      <div class="pd-section-lbl">Volatility</div>
      <div class="pd-grid">
        <div class="pd-cell fx-tip" data-tip-title="Historical Volatility 30d" data-tip-body="30-day realised (historical) volatility, annualised. Measures how much the pair has actually moved recently. Low HV = quiet market; high HV = volatile market. No 52-week percentile is computed for realised vol, so color uses a fixed band: green ≤7% (quiet), red >12% (volatile). Not a directional signal."><div class="pd-lbl">HV 30d</div><div class="pd-val ${hv30 != null ? (hv30 > 12 ? 'pd-dn' : hv30 > 7 ? '' : 'pd-up') : ''}">${hv30 != null ? hv30.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="ATM Implied Volatility${(meta?.cross || nzdProxy) && atmIv != null ? ' (estimated)' : ''}" data-tip-body="${meta?.cross && atmIv != null ? 'Synthesised from component USD-pair implied vol values via triangulation: √(IVa²+IVb²−2ρ·IVa·IVb). Proxy for OTC interbank IV — indicative only.' : nzdProxy && atmIv != null ? 'Estimated from AUD/USD resolved implied vol × 1.08 (long-run NZD/AUD realised vol ratio), used only when the PHLX World Currency Options index (^XDZ) is unavailable. Treat as directional context only.' : `ATM (at-the-money) implied volatility — annualised, same variance-swap methodology as VIX where an exchange-computed index is available. Source: ${_ivSourceLabel(atmIvSource)}.` } ${atmIvRank != null ? 'Color = IV Rank vs 52-week range: green rank&lt;30 (historically cheap), red rank&gt;70 (historically expensive).' : 'Color = cost of hedging: green ≤7% (cheap), red >12% (expensive) — fixed band; IV Rank not available for this pair (synthesised/proxy or &lt;4 weeks history).'} Not a directional signal."><div class="pd-lbl">ATM IV${(meta?.cross || nzdProxy) && atmIv != null ? '<span style="font-size:8px;color:var(--text3);margin-left:2px;">~</span>' : ''}</div><div class="pd-val ${atmIv != null ? (atmIvRank != null ? (atmIvRank > 70 ? 'pd-dn' : atmIvRank < 30 ? 'pd-up' : '') : (atmIv > 12 ? 'pd-dn' : atmIv > 7 ? '' : 'pd-up')) : ''}">${atmIv != null ? atmIv.toFixed(1)+'%' : '—'}</div></div>
        <div class="pd-cell fx-tip" data-tip-title="IV minus HV" data-tip-body="Implied vol minus realised vol — the volatility risk premium (VRP). Positive = options are expensive relative to recent moves (market pricing in a premium). Negative = options are cheap vs realised. IV running above HV is the market's normal state, since option sellers demand a premium — a negative spread is comparatively rare. Not a directional signal." data-tip-ex="IV−HV > +3% historically indicates options are pricing in a premium above recent realised moves — hedging costs are elevated relative to actual market movement. A persistently negative VRP (rare) has historically preceded vol-expansion events, as it signals options are underpricing risk relative to what's actually happening."><div class="pd-lbl">IV − HV</div><div class="pd-val ${atmIv != null && hv30 != null ? cls(hv30 - atmIv) : ''}">${atmIv != null && hv30 != null ? (atmIv > hv30 ? '+' : '') + (atmIv - hv30).toFixed(1)+'%' : '—'}</div></div>
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
              data-tip-body="Net contracts (longs minus shorts) held by Leveraged Funds — hedge funds and CTAs. Speculative / trend-following positioning. Source: CFTC TFF report.${crossNote}"
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
              data-tip-body="Net contracts held by Asset Managers — pension funds, mutual funds, and institutional investors. Structural / longer-term positioning. Source: CFTC TFF report.${crossNote}"
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
      <div class="pd-cell pd-cell--wide fx-tip" data-tip-title="Retail Client Positioning" data-tip-body="Long/short ratio from Myfxbook community outlook — retail traders only, not institutional. Contrarian indicator: extreme retail long bias historically aligns with institutional short positioning. Source: Myfxbook, updated every hour." data-tip-ex="Extreme readings — above 70% long or below 30% long — have historically coincided with elevated positioning risk in the dominant direction. Retail extremes are one input among many; always cross-reference with COT and CB differential data.">
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
// CARRY TRADE RANKING — full G10 45-pair differential, left sidebar
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
// CARRY TRADE RANKING — G10 · real carry · annualised
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
  const G8 = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD','NOK','SEK'];

  // TradingView symbol for a given long/short ccy pair
  function carryTV(long, short) {
    if (short === 'USD') return 'FX_IDC:' + long + 'USD';
    if (long  === 'USD') return 'FX_IDC:USD' + short;
    return 'FX_IDC:' + long + short;
  }

  // Canonical pair ID used in quotes.json / hv30 map — FX market convention,
  // not alphabetical for crosses (e.g. EUR/AUD = 'euraud', GBP/CHF = 'gbpchf').
  function pairId(a, b) {
    const HV30_PAIRS = new Set([
      'eurusd','gbpusd','usdjpy','audusd','usdchf','usdcad','nzdusd',
      'usdnok','usdsek','eurnok','eursek',
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
        ? 'G10 · real carry · annualised'
        : 'G10 · CB rate differential';
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

      // Color: green when real carry ≥+0.5% (positive after inflation)
      //        neutral when 0%–0.5% (marginal carry)
      //        dim when real carry is negative (inflation erodes the nominal spread)
      const cls = realCarryVal != null
        ? (realCarryVal >= 0.5 ? 'pd-up' : realCarryVal <= -0.1 ? 'pd-dim' : '')
        : (p.diff > 2 ? 'pd-up' : p.diff > 0.5 ? '' : 'pd-dim');

      const realStr = realCarryVal != null ? (realCarryVal >= 0 ? '+' : '') + realCarryVal.toFixed(2) + '%' : '—';
      const hvStr   = p.hv30 != null ? p.hv30.toFixed(1) + '%' : 'n/a';
      const tip = `${p.long}/${p.short} · Nominal ${spreadLabel} · Real carry ${realStr} · HV30 ${hvStr} — Click for real rate analysis`;

      return `<div class="carry-rank-row" data-long="${p.long}" data-short="${p.short}" data-sym="${sym}" title="${tip}">
        <span class="cr-rank">${idx + 1}</span>
        <span class="cr-pair">${p.long}/${p.short}</span>
        <span class="cr-spread">${spreadLabel}</span>
        <div class="cr-bar-wrap"><div class="cr-bar" style="width:${barPct}%"></div></div>
        <span class="cr-diff ${cls}">${displayVal}</span>
      </div>`;
    }).join('');

    // ── 9. Row click → open Real Rate Carry Modal ────────────────────────────
    container.querySelectorAll('.carry-rank-row[data-long]').forEach(row => {
      row.addEventListener('click', () => {
        const longCcy  = row.dataset.long;
        const shortCcy = row.dataset.short;
        if (typeof window.openRealCarryModal === 'function') {
          window.openRealCarryModal(longCcy, shortCcy);
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
// VOLATILITY LEADERBOARD — "trade the volatility, not the pair"
// Ranks all 28 G10 pairs by current ATM implied volatility (direct CBOE/CME
// FX Volatility Index for the 6 USD majors; triangulated for the 21 crosses;
// NOK/SEK excluded — no CBOE/CME vol index and no free institutional-grade
// substitute, see GUIDELINES "Data integrity"). Shows the top 5 — the pairs
// where the options market is currently pricing the most movement, i.e.
// where a catalyst (data print, sentiment shift) is most likely playing out
// right now, independent of which pair the trader habitually watches.
// Rendered as a ranked list (v8.104.1) — rank · pair · magnitude bar ·
// value, the same row+bar structure as Carry Trade Ranking below it
// (Bloomberg/Refinitiv Top-N convention). Two earlier designs were tried
// and rejected at the sidebar's typical 180-300px width: a treemap
// (v8.103.0-4, tile area = rank) and a strip plot (v8.104.0, dots on one
// shared axis with decluttered labels) — both needed 5 seven-character
// pair labels to share a single line or axis, which doesn't fit that width
// without cramming. A vertically stacked list has no such constraint.
// ═══════════════════════════════════════════════════════════════════
async function fetchVolLeaderboard() {
  const container = document.getElementById('vol-leaderboard-rows');
  if (!container) return;

  const CROSS_IV_RHO = {
    'eurgbp':0.65,'eurjpy':0.55,'eurchf':0.60,'eurcad':0.40,'euraud':0.35,'eurnzd':0.30,
    'gbpjpy':0.45,'gbpchf':0.55,'gbpcad':0.30,'gbpaud':0.25,'gbpnzd':0.20,
    'audjpy':0.40,'audnzd':0.55,'audchf':0.30,'audcad':0.50,
    'cadjpy':0.35,'cadchf':0.25,'chfjpy':0.40,'nzdjpy':0.35,'nzdcad':0.45,'nzdchf':0.20,
  };

  try {
    const intra = await loadIntradayQuotes();
    const etfIv = intra?.fx_etf_iv || {};

    // Build USD_IV map from direct ETF option data (same pattern as pair-detail)
    const USD_IV = {};
    for (const [pid, entry] of Object.entries(etfIv)) {
      if (entry?.iv == null) continue;
      const p = PAIRS.find(x => x.id === pid);
      if (!p) continue;
      const nonUsd = p.base !== 'USD' ? p.base : p.quote;
      USD_IV[nonUsd] = entry.iv;
    }
    // NZD proxy — no dedicated CBOE/CME NZD vol index
    if (USD_IV['AUD'] != null && USD_IV['NZD'] == null) USD_IV['NZD'] = Math.round(USD_IV['AUD'] * 1.08 * 10) / 10;

    const rows = [];
    for (const p of PAIRS) {
      const ivEntry = etfIv[p.id];
      let atmIv = null, ivRank = null, estimated = false;
      if (ivEntry?.iv != null) {
        atmIv = ivEntry.iv;
        ivRank = ivEntry.iv_rank ?? null;
      } else if (p.cross) {
        const ivA = USD_IV[p.base] ?? null, ivB = USD_IV[p.quote] ?? null;
        if (ivA != null && ivB != null) {
          const rho = CROSS_IV_RHO[p.id] ?? 0.40;
          atmIv = Math.round(Math.sqrt(ivA * ivA + ivB * ivB - 2 * rho * ivA * ivB) * 10) / 10;
          estimated = true;
        }
      }
      // NOK/SEK: no direct or derivable IV — correctly excluded, not fabricated
      if (atmIv == null) continue;
      const label = p.label || (p.id.slice(0, 3).toUpperCase() + '/' + p.id.slice(3).toUpperCase());
      rows.push({ id: p.id, label, atmIv, ivRank, estimated });
    }

    if (!rows.length) {
      container.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Vol data unavailable</div>';
      return;
    }

    rows.sort((a, b) => b.atmIv - a.atmIv);
    const top = rows.slice(0, 5);
    _volLbTopCache = top; // re-laid-out on resize without a refetch

    // Header tooltip — explains the ranking methodology, attached once
    const sbHead = container.closest('.sb-section')?.querySelector('.sb-head');
    if (sbHead && !sbHead._volLbTipAttached) {
      sbHead._volLbTipAttached = true;
      sbHead.style.cursor = 'help';
      const tipTitle = 'Volatility Leaderboard';
      const tipBody  = 'Ranks all 28 G10 pairs by current ATM implied volatility — each of the 6 USD majors sourced from an institutional-grade options market (CBOE/CME FX Volatility Index, PHLX World Currency Options, Saxo Bank, or CME futures/ETF, whichever is live for that pair right now), then triangulated for crosses. Shows the top 5, ranked highest to lowest, with a bar scaled to this group\u2019s own spread (not a fixed 0-100 scale) so the real gap — or lack of one — between them is visible at a glance. Hover any row for IV Rank, the percentile vs the pair\u2019s own 52-week range, shown for context only, not a buy/sell signal (this panel already selected for "high", so it isn\u2019t colored cheap/expensive). ~ prefix = triangulated cross value, not a direct market quote. NOK/SEK excluded — no CBOE/CME vol index exists for either. Note: EUR/GBP/JPY use CBOE/CME\u2019s variance-swap-style vol index, a different construction from the plain indicative ATM mid used for AUD/CHF/CAD/NZD — a gap between e.g. GBP and the rest may partly reflect that methodology difference, not just relative market risk, so treat cross-currency-family comparisons in this ranking as directional rather than strictly like-for-like.';
      const tipEx    = 'Principle: the best opportunity today may not be your usual pair — it\u2019s wherever a data print or sentiment catalyst is driving implied vol higher. Best used as a starting-market filter, not a standalone entry signal.';
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

    renderVolRankList(container);

  } catch (e) {
    console.warn('[VolLeaderboard]', e);
    if (container) container.innerHTML = '<div style="padding:6px 8px;font-size:10px;color:var(--text3);">Unavailable</div>';
  }
}

// Cache of the last-fetched top-5 rows. No longer used for resize-driven
// re-layout (the ranked list below is CSS-fluid — bar widths are
// percentages, so a container resize needs no JS at all) — kept only so a
// future re-render trigger doesn't need to re-fetch loadIntradayQuotes().
let _volLbTopCache = [];

// Renders the current _volLbTopCache as a ranked list: rank · pair ·
// magnitude bar · value — the same row+bar structure as Carry Trade
// Ranking directly below this panel in the sidebar (see .carry-rank-row in
// dashboard.js), which is itself the Bloomberg/Refinitiv Top-N convention.
// Replaces both the treemap (v8.103.0-4) and the strip plot (v8.104.0):
// neither had room for 5 seven-character pair labels at the sidebar's
// typical 180-300px width without cramming or overlap — a vertically
// stacked list has no such constraint, since it only ever needs to fit one
// label per row, not five sharing one line or one shared axis.
function renderVolRankList(container) {
  const top = _volLbTopCache;
  if (!container || !top.length) return;

  container.className = 'vol-lb-rows';

  // Bar length is proportional to this top-5 group's OWN min-max span, not
  // a fixed 0-100 scale — the top 5 are usually within a couple of vol
  // points of each other (same reasoning as the treemap's rank-weight
  // decision, v8.103.1), so a literal 0-100% scale would render five
  // nearly-identical full-width bars. Floored at 15% so the lowest-ranked
  // bar is never invisible.
  const vals = top.map(r => r.atmIv);
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min;

  container.innerHTML = top.map((r, idx) => {
    const sym = 'FX_IDC:' + r.id.toUpperCase();
    const tipRank = r.ivRank != null ? ` · IV Rank ${r.ivRank.toFixed(0)}` : '';
    const tip = `${r.label} · ATM IV ${r.atmIv.toFixed(1)}%${tipRank}${r.estimated ? ' (triangulated)' : ''} — Click for chart · detail`;
    const pct = span > 0 ? 15 + ((r.atmIv - min) / span) * 85 : 100;
    // Estimated (triangulated cross) values carry a visible "~" label, not
    // just a hover tooltip — required for any derived/non-live value
    // (GUIDELINES "Data integrity"), same convention already used by the
    // CB trend fallback (`~ Cut/Hold/Hike`).
    const valStr = (r.estimated ? '~' : '') + r.atmIv.toFixed(1) + '%';

    return `<div class="vol-lb-row" data-sym="${sym}" title="${tip}">
      <span class="vlr-rank">${idx + 1}</span>
      <span class="vlr-pair">${r.label}</span>
      <div class="vlr-bar-wrap"><div class="vlr-bar" style="width:${pct}%"></div></div>
      <span class="vlr-val">${valStr}</span>
    </div>`;
  }).join('');

  container.querySelectorAll('.vol-lb-row[data-sym]').forEach(row => {
    row.addEventListener('click', () => loadTVChart(row.dataset.sym));
  });
}


// ═══════════════════════════════════════════════════════════════════
// CARRY TRADE SIDEBAR — from rates/*.json + extended-data/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchCarryData() {
  const CURRENCIES = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD','NOK','SEK'];
  const LABELS = { USD:'USD Fed', EUR:'EUR ECB', GBP:'GBP BoE', JPY:'JPY BoJ',
                   AUD:'AUD RBA', CHF:'CHF SNB', CAD:'CAD BoC', NZD:'NZD RBNZ',
                   NOK:'NOK NB', SEK:'SEK Riksbank' };

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
      { long: 'NOK', short: 'JPY' },
      { long: 'SEK', short: 'JPY' },
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
  // Show the actual freshness state — live intraday feed when fresh, repo snapshot otherwise
  let sourceLabel = 'Delayed';
  if (_caIntraday?.source && _caIntraday.source !== 'repo') {
    // Check if the file is fresh (under 8 min old — 5 min interval + 3 min margin)
    const fileAge = _caIntraday.updated
      ? (Date.now() - new Date(_caIntraday.updated).getTime()) / 60000
      : 999;
    sourceLabel = fileAge < 8 ? `Live · ~5min delay` : 'Delayed';
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
      ...['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'].map(c =>
        fetch(`./rates/${c}.json`).then(r => r.ok ? r.json() : null).catch(() => null)
      )
    ]);

    const currencies = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];
    const bankMeta = {
      USD: { flag:'us', short:'Fed'    },
      EUR: { flag:'eu', short:'ECB'    },
      GBP: { flag:'gb', short:'BoE'    },
      JPY: { flag:'jp', short:'BoJ'    },
      AUD: { flag:'au', short:'RBA'    },
      CAD: { flag:'ca', short:'BoC'    },
      CHF: { flag:'ch', short:'SNB'    },
      NZD: { flag:'nz', short:'RBNZ'  },
      NOK: { flag:'no', short:'NB'     },
      SEK: { flag:'se', short:'Riksbank' },
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
      // Auto-advance nextMtg display: if nextMeetingISO is today or past and allMeetings
      // has a future date, show the next upcoming one. Prevents "17 Jun" showing stale
      // on the evening of FOMC day until the weekly workflow runs Monday.
      // Bloomberg WIRP auto-advances the meeting date column the moment the meeting passes.
      const _todayISO = new Date().toISOString().slice(0, 10);
      let nextMtg = meetings?.nextMeeting || '—';
      if (meetings?.nextMeetingISO && meetings.nextMeetingISO <= _todayISO && Array.isArray(meetings.allMeetings)) {
        const _nextFuture = meetings.allMeetings.find(d => d > _todayISO);
        if (_nextFuture) {
          const _nf = new Date(_nextFuture + 'T12:00:00Z');
          nextMtg = _nf.getDate() + ' ' + _nf.toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' });
        }
      }

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
      // Bloomberg WIRP standard: display the dominant-direction probability matching the bias.
      //   Hike bias → P(hike)%↑   Cut bias → P(cut)%↓   Hold bias → P(hold)%→
      // Hold probability is the residual: holdProb = 100 − cutProb − hikeProb.
      // Previously, the "else if (cutProb !== null)" branch fired for hold currencies,
      // showing "0%↓" (cut=0%) — misleading. Fixed: hold bias explicitly shows holdProb.
      const cutProb  = meetings?.cutProb  ?? null;  // number (0–100) or null
      const hikeProb = meetings?.hikeProb ?? null;  // number (0–100) or null
      const probSrc  = biasSource || 'OIS/futures';
      const _haveProbData = cutProb !== null || hikeProb !== null;
      let probSuffix = '';
      if (_haveProbData) {
        if (meetingsBias === 'hike') {
          const hp = hikeProb ?? 0;
          const probCls = hp >= 60 ? 'up' : hp >= 40 ? '' : 'flat';
          probSuffix = ` <span class="${probCls}" style="font-size:8px;font-family:var(--font-mono);opacity:0.85;white-space:nowrap;" title="Market-implied probability of a hike at next meeting · ${probSrc}">${hp}%↑</span>`;
        } else if (meetingsBias === 'cut') {
          const cp = cutProb ?? 0;
          const probCls = cp >= 60 ? 'down' : cp >= 40 ? '' : 'flat';
          probSuffix = ` <span class="${probCls}" style="font-size:8px;font-family:var(--font-mono);opacity:0.85;white-space:nowrap;" title="Market-implied probability of a cut at next meeting · ${probSrc}">${cp}%↓</span>`;
        } else {
          // Hold (or unrecognised bias): show hold probability = residual
          const holdProb = Math.max(0, 100 - (cutProb ?? 0) - (hikeProb ?? 0));
          const probCls = holdProb >= 60 ? 'flat' : '';
          probSuffix = ` <span class="${probCls}" style="font-size:8px;font-family:var(--font-mono);opacity:0.85;white-space:nowrap;" title="Market-implied probability of no change at next meeting · ${probSrc}">${holdProb}%→</span>`;
        }
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
      // Guard: if the fwdRate implies a base rate that differs from the current
      // rates/*.json value by more than 2×cbStep, the meetings.json was computed
      // with a stale rate (e.g. BoJ hiked after the last workflow_meetings run).
      // In that case fall through to Priority 2 on-the-fly recalculation.
      const CB_STEP_P1 = { JPY: 0.10, CHF: 0.25 };
      const cbStepP1 = CB_STEP_P1[ccy] ?? 0.25;
      const fwdRateRaw = meetings?.fwdRate;
      // Staleness guard: fwdRate must be within [current − cbStep, current + cbStep].
      // If outside that range, it was computed with a different (pre-hike/cut) base rate
      // and would show a misleading implied rate. Fall through to on-the-fly Priority 2.
      // Example: BoJ hikes 0.75→1.0 but meetings.json still has fwdRate=0.80 (from 0.75 base).
      const fwdRateStale = fwdRateRaw != null && !isNaN(fwdRateRaw) &&
        (fwdRateRaw < current - cbStepP1 - 0.001 || fwdRateRaw > current + cbStepP1 + 0.001);
      if (fwdRateRaw != null && !isNaN(fwdRateRaw) && fwdRateRaw > 0 && !fwdRateStale) {
        fwdDisplay = fwdRateRaw.toFixed(2) + '%';
      } else {
        const pCut  = (meetings?.cutProb  != null && !isNaN(meetings.cutProb))  ? Math.min(100, Math.max(0, meetings.cutProb))  : null;
        const pHike = (meetings?.hikeProb != null && !isNaN(meetings.hikeProb)) ? Math.min(100, Math.max(0, meetings.hikeProb)) : null;

        // Per-bank standard move size (Bloomberg WIRP convention):
        // BoJ historically moves in 10bp increments; SNB uses 25bp standard (may use 50bp).
        // All others: 25bp standard.
        const CB_STEP = { JPY: 0.10, CHF: 0.25 };
        const cbStep  = CB_STEP[ccy] ?? 0.25;

        if (pCut !== null || pHike !== null) {
          // Priority 2: probability-weighted — Bloomberg WIRP standard
          const cut  = pCut  ?? 0;
          const hike = pHike ?? 0;
          // Clamp residual so probabilities never exceed 100%
          const cutC  = Math.min(cut,  100);
          const hikeC = Math.min(hike, 100 - cutC);
          const implied = current + (hikeC / 100) * cbStep - (cutC / 100) * cbStep;
          fwdDisplay = Math.max(0, implied).toFixed(2) + '%';
        } else {
          // Priority 3: no probabilities available — naive ±step, ~ signals estimate
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
// SOURCE 2 — COT (always loaded): CFTC TFF (Traders in Financial Futures) · Leveraged Funds net positioning.
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
        const td1Body  = `ATM IV ${ivStr} from ${_ivSourceLabel(etfIv.source)} — institutional-grade source for OTC interbank implied vol. Green ≤7% (cheap vol); red >12% (expensive).`;
        const td1Ex    = 'Sourced from an institutional-grade options market — an exchange-computed volatility index (same variance-swap methodology as VIX) where available, otherwise a named options market. All values have ~15min delay.';
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
        const td12Body    = 'est. via COT — no CBOE/CME volatility index available for this pair. Derived from CFTC Leveraged Funds net positioning (TFF · Options+Futures Combined). Net = current week; 4W = net ~4 weeks ago (real CFTC history, v7.88.0).';
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
    const res = await fetch('./ai-analysis/index.json', { cache: 'no-store' });
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
      fetch('./ai-analysis/index.json', { cache: 'no-store' }),
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
      const sigR = await fetch('./ai-analysis/signals.json', { cache: 'no-store' });
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
            // Rule 10 (engine SIGNALS_SYSTEM): title format is "PAIR — Setup Name"
            // (em dash, spaces either side). The Setup Name is NOT a fixed frontend
            // taxonomy — it's whatever the LLM named the setup that cycle, generated
            // in the same pass as the body text, so it can't drift out of context.
            function parseTitle(title) {
              if (!title) return null;
              const parts = title.split(' — ');
              if (parts.length !== 2 || !parts[0].trim() || !parts[1].trim()) return null;
              return { pair: parts[0].trim(), badge: parts[1].trim() };
            }
            // Rule 14: text must close with "Trade bias: ... Catalyst: ... Risk: ..."
            // in that order. Split the intro/body from the three labeled clauses.
            const FOOTER_RE = /Trade bias:\s*([\s\S]+?)\s*Catalyst:\s*([\s\S]+?)\s*Risk:\s*([\s\S]+)$/;
            function parseFooter(text) {
              if (!text) return null;
              const m = text.match(FOOTER_RE);
              if (!m) return null;
              const body = text.slice(0, m.index).trim();
              return { body, bias: m[1].trim(), catalyst: m[2].trim(), risk: m[3].trim() };
            }

            container.innerHTML = signals.map(s => {
              const sevCls = s.priority === 'critical' ? 'a-sev-crit' : s.priority === 'warning' ? 'a-sev-warn' : 'a-sev-info';
              const sevTitle = s.priority === 'critical' ? 'High priority' : s.priority === 'warning' ? 'Medium priority' : 'Low priority';
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

              const titleParts = parseTitle(s.title);
              const footerParts = parseFooter(s.text);

              if (titleParts && footerParts) {
                // Evidence chips are intentionally NOT rendered in this structured
                // card — the mockup keeps the card clean (body + three-clause
                // footer only). The underlying data isn't lost: it's still on the
                // native title="" tooltip, available on hover.
                return `<div class="alert-row" ${evTooltip ? `title="${evTooltip}"` : ''}>
                  <div class="a-text">
                    <div class="a-head">
                      <span class="a-name"><span class="a-sev ${sevCls}" role="img" aria-label="${sevTitle}" title="${sevTitle}"></span><span class="a-pair">${titleParts.pair}</span></span>
                      <span class="a-badge">Regime: ${titleParts.badge} · ${localTime}</span>
                    </div>
                    <span class="a-desc">${footerParts.body}</span>
                    <div class="a-foot">
                      <div class="a-foot-line"><span class="a-foot-lbl">Trade bias:</span> ${footerParts.bias}</div>
                      <div class="a-foot-line"><span class="a-foot-lbl">Catalyst:</span> ${footerParts.catalyst}</div>
                      <div class="a-foot-line"><span class="a-foot-lbl">Risk:</span> ${footerParts.risk}</div>
                    </div>
                  </div>
                </div>`;
              }

              // Fallback — for signals without the Rule 10/14 shape (e.g. legacy
              // fetch_intraday_quotes.py entries).
              return `<div class="alert-row${ev.length ? ' a-has-ev' : ''}" ${evTooltip ? `title="${evTooltip}"` : ''}>
                <span class="a-time">${localTime}</span>
                <span class="a-dot ${dotCls}"></span>
                <div class="a-text"><strong>${s.title || ''}</strong>${s.title ? ' — ' : ''}${s.text || ''}${evHtml}</div>
              </div>`;
            }).join('');
            // Evidence chips render inline and always visible (no collapse/expand —
            // a prior click-to-toggle affordance was removed since it gave the false
            // impression the chips were hidden by default when they were not).
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
  fetchVolLeaderboard();
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
  renderFairValue();
  renderDollarSmile(); // beta v3 — Jen's real growth-differential axis only; see growth-differential-data/history.json + fetch_growth_differential.py
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
// Refresh news every 2 minutes — ETag returns 304 when unchanged (zero cost); server updates hourly
setInterval(fetchNewsData, 2 * 60 * 1000);
// Refresh narrative every 15 minutes
setInterval(buildRichNarrative, 15 * 60 * 1000);

// ── CB RATES LIVE POLL — health.json sentinel ─────────────────────────────
// Polls rates/health.json every 5 min (tiny JSON ~300B). If the `run`
// timestamp changed since last check, a new update-rates workflow ran and
// we call fetchCBRates() + fetchCarryRanking() to refresh the table silently.
// Zero flicker — only re-renders if data actually changed.
(function initCBRatesPoll() {
  let _lastRatesRun = null;  // ISO timestamp of the last-seen health.json run
  async function _pollCBRates() {
    try {
      const res = await fetch('./rates/health.json', { cache: 'no-store' });
      if (!res.ok) return;
      const h = await res.json();
      const runTs = h.run || h.timestamp || null;
      if (!runTs) return;
      if (_lastRatesRun === null) {
        // First poll — record baseline, don't refresh (already loaded at boot)
        _lastRatesRun = runTs;
        return;
      }
      if (runTs !== _lastRatesRun) {
        // New run detected — refresh rates silently
        _lastRatesRun = runTs;
        console.log('[CB Rates poll] New rates run detected (' + runTs + ') — refreshing…');
        await fetchCBRates();
        fetchCarryRanking();
        // Also refresh expectations panel — fwdRate uses current rate as base
        if (typeof fetchFedExpectations === 'function') fetchFedExpectations();
      }
    } catch (_e) { /* network error — skip silently */ }
  }
  setInterval(_pollCBRates, 5 * 60 * 1000);  // 5-min interval (health.json is ~300B)
})();
// ─────────────────────────────────────────────────────────────────────────────
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
    _liqSource = d.fallback ? 'Historical avg · fixed reference' : 'H-L range proxy · 30d avg';
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
    ctx.beginPath(); ctx.strokeStyle=_themeColor('--chart-line'); ctx.lineWidth=1.5; ctx.setLineDash([]);
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
    ctx.beginPath(); ctx.strokeStyle=_themeColor('--border2'); ctx.lineWidth=1.5;
    for (let ci=0; ci<48; ci++) {
      const v = hours[sa(ci)];
      ci===0 ? ctx.moveTo(px(ci),py(v)) : ctx.lineTo(px(ci),py(v));
    }
    ctx.stroke();
    ctx.fillStyle='rgba(120,123,134,0.5)'; ctx.font='9px Courier New'; ctx.textAlign='center';
    ctx.fillText('MARKET CLOSED — WEEKEND', W/2, PAD_T+cH/2);
  }

  // Hour labels — starting 22:00 UTC, every 4h: 22,02,06,10,14,18
  ctx.fillStyle=_themeColor('--text3'); ctx.font='8px Courier New'; ctx.textAlign='center';
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
setInterval(() => { fetchRiskData(); fetchCrossAssetData(); fetchCommodityQuotes(); fetchOptionSkew().then(() => attachRiskMonitorTooltips()); fetchVolLeaderboard(); }, 2 * 60 * 1000);
setInterval(fetchCarryData,    30 * 60 * 1000);
setInterval(fetchCarryRanking, 30 * 60 * 1000);
// Refresh sentiment every 30 seconds
setInterval(fetchSentiment, 10 * 60 * 1000);   // every 10 min — Myfxbook source updates hourly (cron '20 * * * *')
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
      backgroundColor:_themeColor('--bg'), gridColor:_themeColorAlpha('--border', 0.8),
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

  if (typeof IntersectionObserver === 'undefined') {
    // Fallback for very old browsers: load everything immediately
    if (typeof loadTVChart === 'function') loadTVChart(window._tvCurrentSym || 'FX_IDC:EURUSD');
    loadTVEvents();
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
      }
    });
  }, { rootMargin: '150px' });

  // defer scripts run after DOM is parsed — DOMContentLoaded may have already fired.
  // Guard: if readyState is already 'interactive' or 'complete', attach observers immediately.
  function attachObservers() {
    var chartWrap = document.getElementById('tv-chart-wrap');
    var calInner  = document.getElementById('tvcal-inner');
    if (chartWrap) io.observe(chartWrap);
    if (calInner)  io.observe(calInner);
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
      // Derivatives uses a custom show/hide toggle, not scroll-into-view.
      // v8.21.5: was checking `display === 'none' || display === ''`, which
      // matches both possible states — the toggle-off branch below was
      // unreachable, so pressing D while Derivatives was already open just
      // called showDerivatives() again instead of closing it. Now checks
      // the shared _activeExclusivePanel flag set by _setExclusivePanel().
      if (window._activeExclusivePanel === 'section-derivatives') {
        if (typeof window._derivNavHide === 'function') window._derivNavHide();
      } else {
        if (typeof window._derivNavShow === 'function') window._derivNavShow();
      }
      return;
    }
    if (target === 'section-news') {
      // News uses the same show/hide toggle pattern as Derivatives — same fix.
      if (window._activeExclusivePanel === 'section-news') {
        if (typeof window._newsNavHide === 'function') window._newsNavHide();
      } else {
        if (typeof window._newsNavShow === 'function') window._newsNavShow();
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
          <span class="kbl-key">B</span><span class="kbl-desc">Research (toggle)</span>
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
    const G8 = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD','NOK','SEK'];
    const rates = {};
    G8.forEach(ccy => {
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
// Stored as { type:'regime', id, target:'RISK-OFF'|'CAUTION'|'MIXED'|'RISK-ON', fired, firedAt }\n// Evaluated against the live computed regime (DOM element #risk-regime)
function _liveRegime() {
  return document.getElementById('risk-regime')?.textContent?.trim() ?? null;
}

// ── Eco Actual alert — event-driven type ─────────────────────────────────────
// Stored as { type:'eco_actual', id, currencies:['USD','EUR',...] or [] for all G8, fired:false }
// Fires once when a NEW actual appears in calendar-data/ff_calendar.json for the
// selected currency set. Resets automatically at midnight UTC (new trading day).
// localStorage key 'gi_eco_fp' → fingerprint of last-seen actuals set.
const ECO_FP_KEY = 'gi_eco_fp';

async function _buildEcoActualFp(currencies) {
  // Returns a fingerprint string of all today's actuals for the given currency set.
  // Empty string if no actuals yet.
  try {
    const res = await fetch('./calendar-data/ff_calendar.json', { cache: 'no-store' }).catch(() => null);
    if (!res?.ok) return null;
    const ffj = await res.json();
    const todayISO = new Date().toISOString().slice(0, 10);
    const events = (ffj?.events || []).filter(ev => {
      const matchDate = ev.dateISO === todayISO;
      const hasActual = ev.actual && ev.actual !== '' && ev.actual !== '-';
      const matchCcy  = !currencies.length || currencies.includes(ev.currency);
      return matchDate && hasActual && matchCcy;
    });
    if (!events.length) return '';
    return events.map(ev => `${ev.currency}|${ev.dateISO}|${ev.timeUTC || ''}|${ev.event||ev.title||''}|${ev.actual}`).sort().join(';;');
  } catch { return null; }
}

function _ecoFpLoad() {
  try {
    const raw = localStorage.getItem(ECO_FP_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    // Auto-expire: reset at midnight UTC
    const todayISO = new Date().toISOString().slice(0, 10);
    if (parsed.date !== todayISO) return {};
    return parsed;
  } catch { return {}; }
}

function _ecoFpSave(fp, date) {
  try { localStorage.setItem(ECO_FP_KEY, JSON.stringify({ fp, date })); } catch {}
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
  if (a.type === 'eco_actual') {
    if (a.lastActuals?.length) return a.lastActuals.map(n => `${n.ccy}: ${n.actual}`).join(' · ');
    return 'new actual';
  }
  const def = ADV_ALERT_TYPES[a.sym];
  if (def?.formatValue) return def.formatValue(v);
  // Price alert: standard numeric
  return v.toFixed(v > 10 ? 2 : 5);
}

function alertDescribeCondition(a) {
  if (a.type === 'regime') return `Regime = ${a.target}`;
  if (a.type === 'eco_actual') {
    const ccyLabel = a.currencies?.length ? a.currencies.join('/') : 'All G8';
    return `Eco actual released — ${ccyLabel}`;
  }
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
    const cat = a.type === 'regime' ? 'regime' : a.type === 'eco_actual' ? 'eco' : (ADV_ALERT_TYPES[a.sym]?.category ?? 'price');
    const catColors = { price:'var(--text2)', spread:'#1D9E75', ivrank:'#185FA5', corr:'#854F0B', var:'#A32D2D', regime:'#533AB7', eco:'#B87A0A' };
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

// ── alertsCheckEco — dedicated eco_actual check (runs every 2 min) ───────────
// Extracted from alertsCheck so eco_actual alerts can run on a 2-min cycle
// independent of the 5-min price/regime alert cycle. Both functions share the
// same fingerprint store (_ecoFpLoad/_ecoFpSave) and alert array (alertsLoad).
async function alertsCheckEco() {
  const arr      = alertsLoad();
  const ecoAlerts = arr.filter(a => a.type === 'eco_actual' && !a.fired);
  if (!ecoAlerts.length) return;

  const todayISO = new Date().toISOString().slice(0, 10);
  const stored   = _ecoFpLoad();
  const prevFp   = stored.fp ?? null;
  const allCcys  = [...new Set(ecoAlerts.flatMap(a => a.currencies || []))];
  const newFp    = await _buildEcoActualFp(allCcys);

  if (newFp === null) return;   // fetch failed — skip silently

  if (prevFp === null) {
    _ecoFpSave(newFp, todayISO);   // First load — baseline only, no notification
    return;
  }

  if (newFp === prevFp || newFp === '') return;   // no change

  const prevSet = new Set(prevFp.split(';;').filter(Boolean));
  const newSet  = new Set(newFp.split(';;').filter(Boolean));
  const added   = [...newSet].filter(x => !prevSet.has(x));
  _ecoFpSave(newFp, todayISO);

  if (!added.length) return;

  const newActuals = added.map(s => {
    const [ccy, , time, title, actual] = s.split('|');
    return { ccy, time, title, actual };
  });

  let changed = false;
  ecoAlerts.forEach(a => {
    const matching = newActuals.filter(n => !a.currencies.length || a.currencies.includes(n.ccy));
    if (!matching.length) return;

    a.fired       = true;
    a.firedAt     = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    a.lastActuals = matching.slice(0, 3);
    changed       = true;

    if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      const ccyLabel = a.currencies.length ? a.currencies.join('/') : 'G8';
      const preview  = matching.slice(0, 2).map(n => `${n.ccy} ${n.title}: ${n.actual}`).join(' · ');
      try {
        new Notification(`GI Terminal — ${ccyLabel} Economic Release`, {
          body : preview || `${matching.length} new actual(s)`,
          icon : '/favicon-192x192.png',
          tag  : 'gi-eco-' + a.id,
        });
      } catch {}
    }
  });

  if (changed) { alertsSave(arr); alertsRender(null); }
}

async function alertsCheck() {
  const arr = alertsLoad();
  if (!arr.length) return;

  // Load intraday data once for all advanced threshold alerts
  let intra = null;
  const needsIntra = arr.some(a => a.type && a.type !== 'price' && a.type !== 'regime' && a.type !== 'eco_actual');
  if (needsIntra) {
    intra = await loadIntradayQuotes().catch(() => null);
  }

  let changed = false;

  // ── eco_actual alerts: handled by alertsCheckEco() (2-min dedicated loop) ───
  // eco_actual is event-driven and runs independently of price/regime alerts.
  // alertsCheckEco() shares the same fingerprint store and alert array; no
  // duplicate handling needed here.

  // ── Threshold alerts: price, spread, ivrank, corr, var, regime ───────────
  arr.forEach(a => {
    if (a.fired) return;
    if (a.type === 'eco_actual') return;
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

  // ── Two separate loops: eco_actual (2 min) vs price/regime (5 min) ────────
  // eco_actual alerts poll ff_calendar.json — the CF Worker + GitHub Actions
  // pipeline delivers new actuals within ~2 min. Running eco checks on the same
  // 5-min cycle as price alerts added up to 3 min of unnecessary lag on top of
  // the pipeline latency. Mirrors calendar-panel.js v1.3 which uses 2 min for
  // the same reason. Price/regime alerts depend on intraday quotes (STOOQ_RT_CACHE)
  // which update every 5 min — no benefit from a faster cycle there.
  setInterval(alertsCheckEco, 2 * 60 * 1000);
  setInterval(alertsCheck,    5 * 60 * 1000);

  // visibilitychange fast-path: if the user focuses the tab after the pipeline
  // has delivered a new actual, eco check fires immediately rather than waiting
  // up to 2 min for the next interval. Mirrors calendar-panel.js behaviour.
  let _lastVisCheck = 0;
  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState !== 'visible') return;
    const now = Date.now();
    if (now - _lastVisCheck < 30 * 1000) return;   // debounce: max once per 30s
    _lastVisCheck = now;
    alertsCheckEco();
  });

  // Init signal notification button state from localStorage
  updateSignalNotifBtn();

  // Init News panel display density (expanded/compact) from localStorage
  _newsLoadDensity();

  // Init News panel text size (A-/A+, 4 steps) from localStorage
  _newsLoadFontSize();

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
      // Toggling split-layout is a class change, not a real window resize, so
      // it never fires the 'resize' listener that normally redraws the yield
      // curve canvas (see drawYieldCurve, which reads clientWidth). Left
      // un-redrawn, the canvas stays at whatever width it had under the
      // previous layout — most visibly, turning split OFF still shows the
      // narrow/compact chart from when split was ON. Reuse the same
      // double-rAF repaint helper already used elsewhere for this exact
      // class of stale-canvas-after-layout-change bug.
      if (typeof _repaintAfterExclusivePanelClosed === 'function') {
        _repaintAfterExclusivePanelClosed();
      }
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
// v8.100.9 (2026-08-12): giOnboardInit() now gates on window.giOnTerminalShown()
// instead of firing straight off DOMContentLoaded — see the function itself
// for the full explanation. Was appearing while the visitor was still on the
// Market Overview snapshot (index.html v8.129.0), pointing at an alerts bell
// that lives inside the still-hidden #gi-terminal-view.

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
  function attemptShow() {
    if (!giOnboardShouldShow()) return;
    // Delay 4s after entering the terminal — let the terminal finish
    // loading data so it doesn't compete visually with the panels
    // rendering in.
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

  // v8.100.9: gated on window.giOnTerminalShown() (gi-overview.js v1.1.0) —
  // this tooltip points at the alerts bell inside #gi-terminal-view, which
  // since v8.129.0 stays hidden behind the Market Overview snapshot until
  // the visitor enters the terminal. The old raw-DOMContentLoaded trigger
  // fired regardless, so it could appear while still on the Overview page
  // (reported by Santiago with a screenshot). Resolves immediately for
  // returning active users (terminal visible from load); otherwise waits
  // for the actual Overview→terminal transition.
  if (window.giOnTerminalShown) {
    window.giOnTerminalShown(attemptShow);
  } else {
    // Fallback for any page that doesn't load gi-overview.js — behave
    // exactly as before.
    attemptShow();
  }
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
  'USD/NOK','USD/SEK','EUR/NOK','EUR/SEK',
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

  const pairs = ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','NZD/USD','USD/NOK','USD/SEK'];
  const rrKeys = {
    'EUR/USD':'EURUSD','GBP/USD':'GBPUSD','USD/JPY':'USDJPY',
    'AUD/USD':'AUDUSD','USD/CHF':'USDCHF','USD/CAD':'USDCAD','NZD/USD':'NZDUSD',
    'USD/NOK':'USDNOK','USD/SEK':'USDSEK'
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
    const crossFwdPairs = ['EUR/GBP','EUR/JPY','GBP/JPY','AUD/JPY','EUR/AUD','EUR/CHF','EUR/NOK','EUR/SEK'];
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
    // EUR/JPY, EUR/GBP, EUR/CHF are all in rr2.json from Saxo.
    // NZD/USD, USD/NOK, USD/SEK excluded (Saxo does not publish these RRs publicly).
    // Order must match HTML #rr-surface-tbody skeleton row order exactly (index-based write).
    const rrPairs = ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','EUR/JPY','EUR/GBP','EUR/CHF'];
    const rrPairKeys = {
      'EUR/USD':'EURUSD','GBP/USD':'GBPUSD','USD/JPY':'USDJPY',
      'AUD/USD':'AUDUSD','USD/CHF':'USDCHF','USD/CAD':'USDCAD',
      'EUR/JPY':'EURJPY','EUR/GBP':'EURGBP','EUR/CHF':'EURCHF'
    };
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
    const termPairs = ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CHF','USD/CAD','NZD/USD','USD/NOK','USD/SEK'];
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
          { label: 'EUR/NOK', ccy: 'NOK' },
          { label: 'EUR/SEK', ccy: 'SEK' },
        ];

        const rows = ecbTbody.querySelectorAll('tr');
        const MN = { USD: 4, GBP: 4, JPY: 2, CHF: 4, AUD: 4, CAD: 4, NZD: 4, NOK: 4, SEK: 4 };

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
        if (footer && fxDate) footer.textContent = `ECB · official reference fixing · ${fxDate} · published ~16:00 CET · source: ECB via Frankfurter`;
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
              <div style="position:absolute;left:0;top:0;bottom:0;width:${barPct.toFixed(1)}%;background:var(--accent);opacity:0.18;border-radius:0 2px 2px 0;"></div>
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
            <td style="font-size:10px;color:var(--text2);">TOTAL (G10)</td>
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
      b.style.background = isActive ? 'var(--accent)' : 'none';
      b.style.color = isActive ? '#fff' : (b.dataset.cty === 'spreads' ? 'var(--accent)' : 'var(--text2)');
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
  no: { file: 'NOK', label: 'Norway',   subtitle: 'NORWAY · SOVEREIGN BOND YIELDS',   tenors: [{ k: 'bond2y', label: '2Y NGB' }, { k: 'bond10y', label: '10Y NGB' }] },
  se: { file: 'SEK', label: 'Sweden',   subtitle: 'SWEDEN · SOVEREIGN BOND YIELDS',   tenors: [{ k: 'bond2y', label: '2Y SGB' }, { k: 'bond10y', label: '10Y SGB' }] },
  ch: { file: 'CHF', label: 'Switzerland', subtitle: 'SWITZERLAND · SOVEREIGN BOND YIELDS', tenors: [{ k: 'bond2y', label: '2Y Conf.' }, { k: 'bond10y', label: '10Y Conf.' }] },
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
    { code: 'no', file: 'NOK', label: 'NO' },
    { code: 'se', file: 'SEK', label: 'SE' },
    { code: 'ch', file: 'CHF', label: 'CH' },
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
      // FIX-36 (v8.98.0): fetch_bond_yields.py (v2.9.9) labels a cached bond2y
      // 'stale-cached' once it's >90d old with no live source available (e.g.
      // CHF/EUR when SNB/ECB feeds stop publishing). This is the table Santiago
      // flagged with the CHF +39bp Curve reading built on a 368-day-old 2Y —
      // stale2y below excludes that value from the slope calc and flags the
      // raw 2Y cell instead of presenting it as a normal live print.
      const stale2y = ext?.sources?.bond2y === 'stale-cached';

      // extended-data always stores yields already as percent (e.g. 4.745 = 4.745%).
      // No fraction->percent conversion here: CHF legitimately trades under 1%
      // (e.g. 0.31 = 0.31%), and a "<1 means fraction" heuristic misreads that
      // as 0.0031 -> *100 -> 31.00%. See CHANGELOG for the incident this fixed.
      const n10 = cty10y;
      const n2  = cty2y;
      const us10 = us10y;
      const us2  = us2y;

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

      // 2Y — stale2y values are still shown (they're real, just old) but styled
      // as such rather than presented as a normal live print.
      if (tds[3]) {
        tds[3].textContent = n2 != null ? n2.toFixed(2) + '%' : '—';
        tds[3].style.color = stale2y ? 'var(--text3)' : '';
        tds[3].title = stale2y
          ? `Stale — no fresh 2Y source available this run (cached ${ext?.dates?.bond2y || 'unknown date'})`
          : '';
      }

      // 2Y-10Y curve slope — excluded when the 2Y leg is stale-cached, since a
      // slope built on a year-old 2Y against a fresh 10Y is not a real reading
      // of today's curve shape.
      if (tds[4]) {
        const slope = (n2 != null && n10 != null && !stale2y) ? (n10 - n2) * 100 : null; // pct-pts -> bp
        if (slope != null) {
          tds[4].textContent = (slope >= 0 ? '+' : '') + slope.toFixed(0) + ' bp';
          tds[4].style.color = slope < 0 ? 'var(--down)' : slope > 50 ? 'var(--up)' : 'var(--text2)';
          tds[4].title = slope < 0 ? 'Inverted curve' : slope < 25 ? 'Flat curve' : 'Normal curve';
        } else {
          tds[4].textContent = '—';
          tds[4].style.color = '';
          tds[4].title = stale2y ? '2Y is stale-cached — slope excluded to avoid a false reading' : '';
        }
      }
    } catch {
      tds.forEach((td, i) => { if (i > 0) td.textContent = '—'; });
    }
  }));
  tbody.dataset.loaded = '2';
}

// ── Economic Surprises — CESI-style centred bar index (v7.76.1) ──────────────
// Methodology: for each G8 currency, computes a normalised surprise index over
// a 90-day rolling window from Finnhub economic calendar (actual vs consensus).
// Index = (beats − misses) / total scored, scaled to [−100, +100].
// Bar chart centred at 0: green bar extends right for positive, red bar extends
// left for negative — matching Citi CESI / Bloomberg BEEI visual convention.
// N column shows count of events with actuals (sample size transparency).
// ── Shared ESI scoring helpers ──────────────────────────────────────────────
// v8.28.0: hoisted out of renderEconSurprises() to module scope. Previously
// _canonEsi/_parseNum/NOISE_KW/INVERSE_KW were const-declared INSIDE
// renderEconSurprises() only, so _lwLoadCompare() (a separate top-level
// function, used by the chart's "Compare ESI" overlay) could not see them.
// _lwLoadCompare() kept its own second copy of NOISE_KW/INVERSE_KW (drift
// risk — exactly what caused the Trade Balance double-inversion bug to need
// fixing in four places) and called the undefined _canonEsi() directly,
// which threw a ReferenceError on every "Compare ESI" click, silently
// swallowed by _lwLoadCompare's try/catch (console.warn only — the overlay
// just never rendered). One definition now, used by both call sites.

// Canonical ESI series key: strip parentheticals then country-name prefix.
// Prevents fragmentation when Myfxbook RSS alternates between short-form
// ("Initial Jobless Claims") and country-prefixed ("United States Initial
// Jobless Claims") titles for the same recurring monthly/weekly event.
// Must stay in sync with Python compute_surprise_stats() in fetch_economic_calendar.py
// and _canonEsi in econ-surprises-modal.js.
const _CCY_PFXS = ['united states ','euro area ','united kingdom ','japan ',
  'australia ','canada ','switzerland ','new zealand ','norway ','sweden '];
// [v8.127.0] Manually-verified cross-vendor title aliases — see
// calendar-panel.js's _CAL_VENDOR_ALIASES (v1.19.18) for the full rationale
// and Guard 8 note. Must stay in sync with that map and with
// compute_surprise_stats() in fetch_economic_calendar.py (engine repo).
const _ESI_VENDOR_ALIASES = {
  'core retail sales mom': 'retail sales ex autos mom',
  'prelim gdp qoq': 'gdp growth rate qoq',
};
const _canonEsi = t => {
  let s = t.replace(/\s*\([^)]*\)/g,'').trim();
  // [v8.126.0] Normalise ForexFactory's slash-notation unit suffixes ("m/m",
  // "y/y", "q/q") to Myfxbook's concatenated form ("MoM"/"YoY"/"QoQ") before
  // country-prefix stripping — same fix and same root cause as
  // calendar-panel.js's _calCanonTitle() (v1.19.16): ForexFactory-sourced
  // forward events (v3.38 hybrid architecture) never matched Myfxbook-titled
  // history, fragmenting the ESI baseline the same way it fragmented the
  // drill-down modal. `t` here is already lowercased by every caller, but the
  // regex is written case-insensitively regardless so this function is safe
  // to call directly in the future without relying on that convention.
  s = s.replace(/\bm\/m\b/gi, 'mom').replace(/\by\/y\b/gi, 'yoy').replace(/\bq\/q\b/gi, 'qoq');
  for (const p of _CCY_PFXS) { if (s.startsWith(p)) { s = s.slice(p.length); break; } }
  if (_ESI_VENDOR_ALIASES[s]) s = _ESI_VENDOR_ALIASES[s];
  return s;
};

// ── Numeric parser for macro actual/forecast values ──────────────────────
// parseFloat() alone fails on currency-symbol-prefixed strings such as
// "$-226.8B", "A$1.791B", "¥3907B", "CHF15.5B", "NOK62.6B", "-€5.2B".
// All Trade Balance and Current Account events carry these prefixes, so they
// were silently excluded from ESI scoring (isNaN check returned false).
// Strategy: strip everything except digits and the decimal point, then restore
// the sign by checking whether the original string contained a minus anywhere.
// This is safe because macro data strings never contain two separate numbers.
const _parseNum = s => {
  if (s == null || s === '') return NaN;
  const str = String(s).replace(/,/g, '');
  const neg  = str.includes('-');
  const digits = str.replace(/[^\d.]/g, '');
  const n = parseFloat(digits);
  return isNaN(n) ? NaN : (neg ? -n : n);
};

// Shared noise-keyword list (defined once, reused by every scorer).
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
  '4-week average','4-week avg',
  'tic net','net long-term tic','total net tic',
  'interest rate projection',
  'eia crude oil','eia crude',
  // v8.51.15: Myfxbook retail-positioning "Sentiment" releases are not official
  // macro data — they carry no real consensus forecast. calendar.json backfills
  // their missing `forecast` from `previous` (last week's sentiment %), which
  // fabricates a beat/miss against last week's reading rather than an actual
  // survey/estimate. Scoring them inflates N and distorts beat rate / index
  // (confirmed: ~12-34% of scored G10 events in some 90d windows were Myfxbook
  // Sentiment noise). Keyword is 'myfxbook' specifically (not 'sentiment') so
  // legitimate sentiment surveys — Michigan Consumer Sentiment, ZEW Economic
  // Sentiment, IFO, GfK — are NOT excluded.
  'myfxbook',
];

// Inverse indicators: a lower actual is a positive surprise (e.g. unemployment fell).
// v8.27.0: "trade balance" removed — Trade Balance is a SIGNED net level (deficit
// negative, surplus positive), same as Current Account which this list already
// correctly excludes. For a signed balance, actual > forecast already means a
// smaller deficit / bigger surplus than expected — the good direction — with no
// inversion needed. Confirmed against calendar.json: 36/36 Trade Balance prints
// (GBP/USD) are negative-signed, matching the same convention as Current Account.
// v8.100.6: calendar-panel.js's Actual-column coloring (CAL_INVERSE_KW) had never
// implemented this concept at all — any change here must now also be evaluated for
// calendar-panel.js, not just econ-surprises-modal.js and fetch_economic_calendar.py.
// v8.100.7: added "unemployed" — "unemployment" is NOT a substring of "unemployed"
// (differ after "employ-": "-ment" vs "-ed"), so "Unemployed Persons" (EUR/Germany
// monthly, NOK) silently missed inversion. Confirmed against a full year of G10
// calendar.json: 15 occurrences, all mis-colored; no other inverse-indicator gaps
// found across the 690 unique event titles in that dataset.
const INVERSE_KW = ['unemployment', 'unemployed', 'jobless', 'claims', 'deficit'];

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
      if (hasReleased) { calEvents = evts; calSource = calj.source || ''; }
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
        if (hasReleased) { calEvents = evts; calSource = ffj.source || 'ForexFactory'; }
      }
    } catch { /* no fallback */ }
  }

  // ── Source label ──────────────────────────────────────────────────────────
  // ── Score per currency ────────────────────────────────────────────────────
  // Inverse indicators: a lower actual is a positive surprise (e.g. unemployment fell)
  // v8.27.0: "trade balance" removed — Trade Balance is a SIGNED net level (deficit
  // negative, surplus positive), same as Current Account which this list already
  // correctly excludes. For a signed balance, actual > forecast already means a
  // ── Exponential time-decay (CESI convention) ────────────────────────────────────────
  // CESI applies decay so recent surprises dominate and old data fades.
  // Half-life = 45 days: w(0d)=1.00, w(45d)=0.50, w(90d)=0.25.
  // λ = ln(2) / 45 ≈ 0.01540. N column still shows raw event count for transparency.
  const DECAY_LAMBDA = Math.LN2 / 45;

  // v8.21.7: Two-pass adaptive window — mirrors EA ComputeESI() v8.4.3+.
  // Pass 0: standard 90d window (all G10). Pass 1: 90–180d band, ONLY for
  // currencies that end pass 0 with zero weight (NOK/SEK in practice — the
  // upstream provider tags almost all their releases "low" impact, leaving
  // fewer medium/high events in any 90d slice than G7 currencies).
  // The impact filter and decay function are identical across both passes —
  // widening the window does NOT lower methodology standards. An event 150
  // days old carries only ~13% weight (half-life=45d), so the extension never
  // floods the index signal; it merely provides a non-zero baseline rather
  // than forcing a blank row for structurally-thin-coverage currencies.
  const WIDE_LOOKBACK_MS = 180 * 24 * 60 * 60 * 1000;
  const ccyScores  = {};
  const widenedCcys = new Set();

  function _scorePass(minAgeMs, maxAgeMs, limitCcys) {
    calEvents.forEach(ev => {
      const evTime = new Date(ev.dateISO).getTime();
      if (isNaN(evTime) || evTime > nowMs) return;
      const ageMs = nowMs - evTime;
      if (ageMs <= minAgeMs || ageMs > maxAgeMs) return;
      if (!ev.released || ev.actual == null) return;
      if (!['medium','high'].includes(ev.impact)) return;
      const ccy = ev.currency;
      if (!['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'].includes(ccy)) return;
      if (limitCcys && !limitCcys.has(ccy)) return;

      // ── Noise filter: exclude non-macro events ────────────────────────────
      // CESI-style indices (Citi, DB, MS) only score fundamental macro releases.
      // Bond auctions, CFTC positioning, commodity inventory/rig data, derived
      // averages, financial flow data, and SEP dot projections are excluded.
      const evTitle = (ev.event || ev.title || '').toLowerCase();
      if (NOISE_KW.some(kw => evTitle.includes(kw))) return;

      // ── Dedup guard: same canonical event + same actual → score only once ──
      // ForexFactory publishes Flash then Final PMIs with identical data on
      // different dates. Without dedup, each revision counts as a separate event,
      // inflating N and double-counting the same macro signal.
      const canonEvent = _canonEsi(evTitle);
      // Use forecast||previous in the dedup key — mirrors fetch_economic_calendar.py
      // so events without an explicit forecast but with a previous baseline deduplicate
      // consistently between JS scoring and Python surpriseStats computation.
      const dedupKey = `${ccy}/${canonEvent}/${String(ev.actual).replace(/[%,\s]/g,'')}/${String(ev.forecast||ev.previous||'').replace(/[%,\s]/g,'')}`;
      if (!window._ES_SEEN) window._ES_SEEN = new Set();
      if (window._ES_SEEN.has(dedupKey)) return;
      window._ES_SEEN.add(dedupKey);
      // ──────────────────────────────────────────────────────────────────────

      const actual   = _parseNum(ev.actual);
      const forecast = _parseNum(ev.forecast || ev.previous);
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

      // ── Exponential decay × impact weight ──────────────────────────────────
      // w = e^(-λ · ageDays) × impactMult. Recent events dominate; high-impact
      // releases score twice the weight of medium (HIGH=1.0, MEDIUM=0.5) —
      // consistent with EA ComputeESI() and Citi/DB institutional conventions.
      const ageDays    = (nowMs - evTime) / 86400000;
      const impactMult = ev.impact === 'high' ? 1.0 : 0.5;
      const w = Math.exp(-DECAY_LAMBDA * ageDays) * impactMult;

      // ── Z-score scoring (hybrid: z-score when stats available, beat/miss otherwise) ──
      // As history accumulates in surpriseStats (engine v3.1+), more events
      // graduate to z-score. MIN 5 observations required for a valid std estimate.
      const CANONICAL_MIN_N = 5;
      const statsKey = `${ccy}/${_canonEsi(evTitle)}`;
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
  }

  // Pass 0: standard 90d window — all G10 currencies.
  _scorePass(0, LOOKBACK_MS, null);

  // Pass 1: 90–180d extension band — only for currencies with zero weight after pass 0.
  // Dedup set (window._ES_SEEN) is shared, so no event can be double-counted
  // even if the same release appears in both the 90d and 90–180d calendar slices.
  const G10_CCYS = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];
  const sparseCcys = new Set(G10_CCYS.filter(c => !ccyScores[c] || ccyScores[c].wTotal === 0));
  if (sparseCcys.size > 0) {
    _scorePass(LOOKBACK_MS, WIDE_LOOKBACK_MS, sparseCcys);
    // Track which currencies actually gained data from the extension.
    G10_CCYS.forEach(c => { if (sparseCcys.has(c) && ccyScores[c]?.wTotal > 0) widenedCcys.add(c); });
  }

  // ── Normalise to [−100, +100] index (Citi CESI convention) ───────────────
  // index = (beats − misses) / total × 100
  // Bar fill: 50% of bar width per side (each side = 50% of container)
  const G8 = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];
  const rows = tbody.querySelectorAll('tr');

  G8.forEach((ccy, idx) => {
    const row = rows[idx];
    if (!row) return;
    const tds = row.querySelectorAll('td');
    const barFill = row.querySelector('.es-bar-fill');
    const s = ccyScores[ccy];
    const isWidened = widenedCcys.has(ccy);

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
    // ── Confidence shrinkage (v8.87.0) ────────────────────────────────────
    // Parallel fix to ComputeESI() v3.347 in the MT5 EA (Global_Investing_
    // FX_Terminal.mq5): the bar itself was already scaled against a FIXED
    // ±100 range (halfPct below), not against the locally-observed max on
    // screen, so the EA's bar-scaling bug never existed here. But idx100 had
    // no penalty for thin sample support — a currency with N=1 that happened
    // to beat consensus could show idx100=+100, the same full-conviction bar
    // as a currency backed by dozens of independent releases. Reused the
    // s.total counter (already computed above for the N column) rather than
    // introducing a second counter — CONFIDENCE_MIN_N mirrors the existing
    // lowConf threshold a few lines below, so a currency only reaches full
    // (1.0) confidence once its N clears the same bar this panel already
    // uses to decide whether N itself is trustworthy enough to display at
    // full brightness.
    const CONFIDENCE_MIN_N = 15;
    const confidence = Math.min(1, s.total / CONFIDENCE_MIN_N);
    idx100 *= confidence;

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
      tds[2].title = isWidened
        ? 'Window extended to 180d (sparse coverage in 90d — same impact filter and decay)'
        : (lowConf ? 'Low sample size — interpret with caution' : '');
    }

    // Row tooltip
    const pct = (s.beats / s.total * 100).toFixed(0);
    const inLine = s.total - s.beats - s.misses;
    const windowNote = isWidened ? ' · 90d/180d adaptive window' : ' · 90d window';
    row.title = `${ccy}: ${s.beats} beat · ${s.misses} miss · ${inLine} in-line · ${pct}% beat rate · index ${idx100 >= 0 ? '+' : ''}${idx100.toFixed(0)} · decay-weighted (45d half-life)${windowNote} · click for detail`;
  });

  // ── Source label (written here so widenedCcys is fully populated) ──────
  // [this session] The header subtitle (static HTML, "Actual vs consensus
  // forecast · G10 major currencies · 90d rolling") already states the
  // panel's standing methodology. Repeating that same text in the footer
  // is redundant per Bloomberg/Refinitiv convention (a panel states its
  // methodology once). This footer is now reserved for state that the
  // header can't express: an unavailable feed, or an adaptive-window widen
  // triggered by sparse 90d coverage. When neither condition applies, the
  // line is cleared and collapsed so no empty gap is left under the table.
  (function _writeSourceLabel() {
    const srcEl = document.getElementById('econ-surprise-source');
    if (!srcEl) return;
    if (!calSource) {
      srcEl.textContent = 'Calendar data unavailable';
      srcEl.style.display = '';
    } else if (widenedCcys.size > 0) {
      srcEl.textContent = 'Window extended to 180d for sparse-coverage currencies';
      srcEl.style.display = '';
    } else {
      srcEl.textContent = '';
      srcEl.style.display = 'none';
    }
  })();

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

// ── Display density — expanded (excerpt always visible, v8.8.2 default) vs.
// compact (one line per item, no excerpt — Bloomberg TOP<GO> scan density).
// Storage: localStorage key 'gi_news_density' → 'expanded' | 'compact' (default: 'expanded').
// Applied via a single CSS class toggle on #section-news — collapses .ns-art-body/
// .rs-art-body across all three sub-panels (News/Research/Analysis) at once,
// no changes needed to _buildNsItem() or the Research inline article builder.
const NEWS_DENSITY_KEY = 'gi_news_density';
let _newsDensity = 'expanded';

function _newsLoadDensity() {
  try { _newsDensity = localStorage.getItem(NEWS_DENSITY_KEY) || 'expanded'; } catch { _newsDensity = 'expanded'; }
  if (_newsDensity !== 'compact' && _newsDensity !== 'expanded') _newsDensity = 'expanded';
  _newsApplyDensity();
}

function _newsApplyDensity() {
  // v8.169.0 FIX: the compact/expanded CSS rules were scoped to
  // `#section-news.ns-compact …` only. That worked fine in the stacked
  // (compact/docked) view, where #intel-scroll is a real descendant of
  // #section-news — but openIntelFullscreen() (v8.167.x) DOM-lifts
  // #intel-scroll out into #intel-fullscreen-inner, a sibling overlay
  // OUTSIDE #section-news. Once in fullscreen, .ns-article/.rs-article are
  // no longer descendants of #section-news at all, so the class toggle on
  // #section-news had nothing left to select — the buttons flipped their
  // own active state correctly but the layout never changed. Fixed by
  // toggling the same class on #intel-scroll too (the element that actually
  // moves and stays a real ancestor of the articles in both contexts) and
  // matching it in CSS — see #intel-scroll.ns-compact rules below.
  const section = document.getElementById('section-news');
  const scroll  = document.getElementById('intel-scroll');
  if (section) section.classList.toggle('ns-compact', _newsDensity === 'compact');
  if (scroll)  scroll.classList.toggle('ns-compact', _newsDensity === 'compact');
  document.querySelectorAll('.ns-density-btn').forEach(function(btn) {
    btn.classList.toggle('ns-density-active', btn.dataset.density === _newsDensity);
  });
}

function _newsSetDensity(mode) {
  if (mode !== 'compact' && mode !== 'expanded') return;
  _newsDensity = mode;
  try { localStorage.setItem(NEWS_DENSITY_KEY, mode); } catch {}
  _newsApplyDensity();
}
window._newsSetDensity = _newsSetDensity;

// ── Text size (A-/A+) — 4 discrete steps, index 0..3 → 9px/10px/12px/14px.
// Storage: localStorage key 'gi_news_fontsize' → '0'..'3' (default: '1',
// i.e. the pre-existing 10px baseline, so nobody sees a change unless they
// press the control). Applied the same way as density: a single CSS class
// toggle (.ns-fontsize-0/2/3 — step 1 has no override, it's the base CSS
// value already) on #section-news AND #intel-scroll, so the control keeps
// working whether #intel-scroll is docked or DOM-lifted into fullscreen.
const NEWS_FONTSIZE_KEY = 'gi_news_fontsize';
const NEWS_FONTSIZE_STEPS = [9, 10, 12, 14];
let _newsFontSizeIdx = 1;

function _newsLoadFontSize() {
  try {
    const raw = parseInt(localStorage.getItem(NEWS_FONTSIZE_KEY), 10);
    _newsFontSizeIdx = Number.isInteger(raw) && raw >= 0 && raw < NEWS_FONTSIZE_STEPS.length ? raw : 1;
  } catch { _newsFontSizeIdx = 1; }
  _newsApplyFontSize();
}

function _newsApplyFontSize() {
  const section = document.getElementById('section-news');
  const scroll  = document.getElementById('intel-scroll');
  [section, scroll].forEach(function (el) {
    if (!el) return;
    for (let i = 0; i < NEWS_FONTSIZE_STEPS.length; i++) el.classList.remove('ns-fontsize-' + i);
    el.classList.add('ns-fontsize-' + _newsFontSizeIdx);
  });
  const dec = document.getElementById('ns-fontsize-dec');
  const inc = document.getElementById('ns-fontsize-inc');
  if (dec) dec.disabled = _newsFontSizeIdx === 0;
  if (inc) inc.disabled = _newsFontSizeIdx === NEWS_FONTSIZE_STEPS.length - 1;
}

function _newsFontSizeAdjust(delta) {
  const next = _newsFontSizeIdx + delta;
  if (next < 0 || next >= NEWS_FONTSIZE_STEPS.length) return;
  _newsFontSizeIdx = next;
  try { localStorage.setItem(NEWS_FONTSIZE_KEY, String(_newsFontSizeIdx)); } catch {}
  _newsApplyFontSize();
}
window._newsFontSizeAdjust = _newsFontSizeAdjust;

// Sources classified as TA/market analysis — rendered in Analysis sub-panel
// (renamed from "Trading" — aligns with Bloomberg/Reuters/Risk.net terminology)
// CB official feeds, macro wires, institutional press → News sub-panel
const _ANALYSIS_SOURCES = new Set([
  'Barchart', 'BabyPips', 'ForexCrunch',
  'DailyForex TA', 'InvestingLive', 'ActionForex',
  'MyFXBook', 'Investing.com',
  'MarketPulse', 'FX Empire',        // reclassified from News: analysis/forecast, not macro wire
]);

// Sources from news feeds that belong in the Research sub-panel
// (alongside bank-research.json institutional notes)
const _RESEARCH_NEWS_SOURCES = new Set([
  'InvestMacro',     // COT + positioning data — institutional
  'Marc to Market',  // macro sell-side analysis (ex-HSBC/BBH CMO)
  'FX Markets',      // Risk.net institutional FX press
]);

// ── Helper: build one NS item element (shared by News and Trading feeds) ──────
// Styled to match the CB Rates modal's Market Commentary block (cbr-ps-art-*):
// a always-expanded newswire article — meta row (source · time · currency),
// title (clickable headline when a safe link is available), and body excerpt below.
// No collapse/expand interaction — Bloomberg/Reuters wire panels read top-to-bottom.
function _buildNsItem(item, containerEl) {
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

  // Time label: "HH:MM · Ns" for recent items, "HH:MM · Mon D" once a day has passed
  let timeLabel = time;
  if (ageMs > 0) {
    const ageMin  = Math.floor(ageMs / 60000);
    const ageHr   = Math.floor(ageMs / 3600000);
    const ageDays = Math.floor(ageMs / 86400000);
    if (ageDays >= 1 && pubDate) {
      timeLabel = time + ' \u00b7 ' + pubDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } else if (ageHr >= 1) {
      timeLabel = time + ' \u00b7 ' + ageHr + 'h';
    } else if (ageMin >= 1) {
      timeLabel = time + ' \u00b7 ' + ageMin + 'm';
    } else {
      timeLabel = time + ' \u00b7 now';
    }
  }

  const headline = item.title  || '';
  const cur      = item.cur    || '';
  const source   = item.source || '';
  const rawLink  = item.link   || '';
  const safeLink = rawLink.startsWith('https://') ? rawLink : '';

  // Body excerpt — truncate at the last full sentence within ~500 chars
  // (same convention as the CB Rates Market Commentary block).
  let body = (item.expand || '').replace(/\s+/g, ' ').trim();
  if (body.length > 500) {
    const cut = body.slice(0, 500);
    const lastPeriod = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('? '), cut.lastIndexOf('! '));
    body = (lastPeriod > 200 ? cut.slice(0, lastPeriod + 1) : cut) + '\u2026';
  }

  const wrap = document.createElement('div');
  wrap.className = 'ns-article' + (item.featured ? ' ns-featured' : '');

  // ── Meta row: source · time · currency tag ──
  const meta = document.createElement('div');
  meta.className = 'ns-art-meta';

  if (source) {
    const srcEl = document.createElement('span');
    srcEl.className = 'ns-art-source';
    srcEl.textContent = source;
    meta.appendChild(srcEl);
  }

  const timeEl = document.createElement('span');
  timeEl.className = 'ns-art-time';
  timeEl.textContent = timeLabel;
  meta.appendChild(timeEl);

  if (cur) {
    const curTag = document.createElement('span');
    curTag.className = 'ns-cur-tag';
    curTag.textContent = cur;
    meta.appendChild(curTag);
  }
  wrap.appendChild(meta);

  // ── Title — clickable headline when a safe link is available ──
  const titleEl = document.createElement('div');
  titleEl.className = 'ns-art-title';
  titleEl.title = headline; // full text on hover — compact mode truncates with ellipsis
  if (safeLink) {
    const a = document.createElement('a');
    a.href = safeLink;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = headline;
    titleEl.appendChild(a);
  } else {
    titleEl.textContent = headline;
  }
  wrap.appendChild(titleEl);

  // ── Body — always visible excerpt ──
  if (body) {
    const bodyEl = document.createElement('div');
    bodyEl.className = 'ns-art-body';
    bodyEl.textContent = body;
    wrap.appendChild(bodyEl);
  }

  return wrap;
}

function renderNewsSection(items, meta) {
  if (Array.isArray(items)) _newsAllItems = items;
  if (meta) _newsMeta = meta;

  const newsFeed    = document.getElementById('news-section-feed');
  const tradingFeed = document.getElementById('trading-section-feed');
  if (!newsFeed) return;

  // Apply currency filter to all items
  const filtered = _newsAllItems.filter(function(item) {
    return _newsFilter.cur === 'ALL' || item.cur === _newsFilter.cur;
  });

  // Three-way split:
  // Research news sources → injected into research panel (via _researchNewsItems)
  // Analysis sources → analysisFeed
  // Everything else (CB, macro wires) → newsFeed
  const newsItems     = filtered.filter(function(i) { return !_ANALYSIS_SOURCES.has(i.source) && !_RESEARCH_NEWS_SOURCES.has(i.source); });
  const analysisItems = filtered.filter(function(i) { return  _ANALYSIS_SOURCES.has(i.source); });
  const researchNews  = filtered.filter(function(i) { return  _RESEARCH_NEWS_SOURCES.has(i.source); });

  // Push research-news items into the research panel
  if (researchNews.length) {
    window._researchNewsItems = researchNews;
    renderResearchSection();
  }

  // News count
  const newsCountEl = document.getElementById('news-section-count');
  if (newsCountEl) newsCountEl.textContent = newsItems.length + ' stories';

  // Analysis count
  const tradingCountEl = document.getElementById('trading-section-count');
  if (tradingCountEl) tradingCountEl.textContent = analysisItems.length + ' items';

  // Render News feed
  newsFeed.innerHTML = '';
  if (!newsItems.length) {
    const empty = document.createElement('div');
    empty.style.cssText = 'padding:14px;color:var(--text3);font-size:11px;';
    empty.textContent = 'No stories match current filter.';
    newsFeed.appendChild(empty);
  } else {
    newsItems.forEach(function(item) {
      newsFeed.appendChild(_buildNsItem(item, newsFeed));
    });
  }

  // Render Analysis feed
  if (tradingFeed) {
    tradingFeed.innerHTML = '';
    if (!analysisItems.length) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:14px;color:var(--text3);font-size:11px;';
      empty.textContent = 'No analysis items match current filter.';
      tradingFeed.appendChild(empty);
    } else {
      analysisItems.forEach(function(item) {
        tradingFeed.appendChild(_buildNsItem(item, tradingFeed));
      });
    }
  }

  // Wide-monitor two-column layout (Intel Fullscreen only, see
  // _intelApplyColumnSplit() near openIntelFullscreen() below) — applied
  // after every fresh render of both feeds this function owns.
  _intelApplyColumnSplit('news-section-feed');
  _intelApplyColumnSplit('trading-section-feed');
}

function _newsSetFilter(type, value) {
  _newsFilter[type] = value;
  // Update active pill styling
  const selector = type === 'cur' ? '.ns-cur-pill' : '.ns-imp-pill';
  document.querySelectorAll(selector).forEach(btn => {
    btn.classList.toggle('ns-pill-active', btn.dataset.val === value);
  });
  renderNewsSection();
  // Apply same currency filter to research sub-panel if it has data — but
  // only in the compact/stacked view, where all three sub-panels are visible
  // together and a single shared currency filter reads as one control over
  // the whole scan. Skipped while Intel Fullscreen (v8.167.x) is open: each
  // tab owns its own filter there by design, so switching News's currency
  // pill must not silently change what the Research tab shows when the user
  // switches to it.
  const intelFsOverlay = document.getElementById('intel-fullscreen-overlay');
  const inIntelFullscreen = !!(intelFsOverlay && intelFsOverlay.classList.contains('intel-fs-active'));
  if (_researchAllItems.length && type === 'cur' && !inIntelFullscreen) {
    _researchFilter.cur = value;
    renderResearchSection();
  }
}

// ═══════════════════════════════════════════════════════════════════
// INTEL FULLSCREEN — News/Research/Analysis as tabs (v8.167.x)
// ═══════════════════════════════════════════════════════════════════
// Compact/docked #section-news shows all three sub-panels stacked at once
// (good for scanning across feeds). Fullscreen switches to a dedicated
// reading mode instead: one sub-panel at a time, each getting the full
// available height and its own filter bar — matches the Bloomberg NI /
// Refinitiv Eikon News Monitor pattern of category tabs in a maximized news
// view, and mirrors this app's own existing fullscreen pattern for the
// Economic Calendar (calendar-panel.js's cal-fs-btn/cal-fullscreen-overlay)
// and the price chart (dashboard.js's lw-fs-btn/lw-fullscreen-overlay).
//
// DOM-lift approach, same as those two: #intel-scroll (holding the three
// .intel-group wrappers) is appended into #intel-fullscreen-inner on open
// and restored to its original position on close. #ns-filter-bar is lifted
// separately, into #intel-group-news specifically — in the compact view it
// lives outside #intel-scroll as a sticky bar shared visually across all
// three stacked sub-panels, but in fullscreen (per Santiago's "own filters
// per tab" choice) it belongs to the News tab alone, sitting directly under
// the News header the same way #rs-filter-bar already sits under Research's.
// Neither .intel-group wrapper exists as a layout box in the compact view —
// they're `display:contents` there — so #intel-scroll's existing CSS Grid
// (grid-template-rows listing the 7 real children in DOM order, see
// dashboard.css v8.117.18) keeps working completely unchanged when not
// fullscreen; the wrappers only become real flex boxes, and #ns-filter-bar
// only gets reparented, while #intel-fullscreen-overlay.intel-fs-active.
let _intelFsOriginalScrollParent = null;
let _intelFsOriginalScrollNext   = null;
let _intelFsOriginalFilterParent = null;
let _intelFsOriginalFilterNext   = null;
let _intelFsActiveTab = 'news'; // default tab on open — News is the most recent/time-sensitive feed

function _intelFsSetTab(tab) {
  if (!['news', 'research', 'analysis'].includes(tab)) return;
  _intelFsActiveTab = tab;
  // Scoped to #intel-fullscreen-overlay: .intel-fs-tab is a shared visual
  // class reused by the unrelated Correlations Pairs-matrix timeframe tabs
  // (#corr-mtx-fullscreen-overlay). An unscoped querySelectorAll here would
  // also match those Daily/4h/Hourly buttons — their dataset.tab is always
  // undefined so they'd never gain intel-fs-tab-active, but they WOULD get
  // it silently stripped off (via the toggle() below) any time this runs,
  // desyncing their visual state from _corrPairsActiveTf until the next
  // click. See v8.184.0 CHANGELOG entry.
  document.querySelectorAll('#intel-fullscreen-overlay .intel-fs-tab').forEach(function (btn) {
    btn.classList.toggle('intel-fs-tab-active', btn.dataset.tab === tab);
    btn.setAttribute('aria-selected', btn.dataset.tab === tab ? 'true' : 'false');
  });
  document.querySelectorAll('.intel-group').forEach(function (grp) {
    grp.classList.toggle('intel-group-active', grp.id === 'intel-group-' + tab);
  });
}
window._intelFsSetTab = _intelFsSetTab;

// ── Wide-monitor two-column layout (v8.171.0) ───────────────────────────────
// Mirrors calendar-panel.js's shouldSplitCalColumns()/.cal-col-wrap pattern
// exactly — same reasoning: capping fullscreen content to a readable
// max-width (v8.170.0, ~880px here) is correct for line length, but on a
// genuinely wide monitor it leaves large empty gutters on both sides that
// a single centered column can't use. The calendar's fix is two real,
// independently-scrolling DOM columns (newspaper flow: read column 1
// top-to-bottom, then column 2) rather than CSS `column-count` — multi-col
// CSS doesn't scroll correctly against a growing list inside a
// fixed-height, overflow:auto container, since the browser sizes columns
// to fit one "page" instead of flowing content down predictably. Same
// 1400px breakpoint as the calendar, for one consistent "is this a wide
// monitor" threshold across the app rather than a second independently-
// tuned number.
function shouldSplitIntelColumns() {
  const overlay = document.getElementById('intel-fullscreen-overlay');
  return !!(overlay && overlay.classList.contains('intel-fs-active') && window.innerWidth >= 1400);
}

// Called right after a render function (renderNewsSection/
// renderResearchSection) has freshly rebuilt containerId's children as one
// flat top-to-bottom flow — never called on an already-split container, so
// this always starts from flat content and doesn't need to detect/undo a
// prior split state itself.
function _intelApplyColumnSplit(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.classList.remove('intel-cols-active');
  if (!shouldSplitIntelColumns()) return;
  const items = Array.from(container.children);
  if (items.length < 2) return; // not worth splitting 0-1 items (incl. the empty-state div)
  const mid  = Math.ceil(items.length / 2);
  const colA = document.createElement('div');
  colA.className = 'intel-col-wrap';
  const colB = document.createElement('div');
  colB.className = 'intel-col-wrap';
  items.slice(0, mid).forEach(function (el) { colA.appendChild(el); }); // appendChild MOVES, not clones
  items.slice(mid).forEach(function (el) { colB.appendChild(el); });
  container.appendChild(colA);
  container.appendChild(colB);
  container.classList.add('intel-cols-active');
}
window._intelApplyColumnSplit = _intelApplyColumnSplit;

// Re-derives all three feeds' column layout without waiting for their next
// data-driven render — needed on fullscreen open/close (the breakpoint
// check's own input, .intel-fs-active, just changed) and on a resize that
// crosses the 1400px breakpoint while fullscreen is already open.
function _intelRelayoutColumns() {
  ['news-section-feed', 'research-section-feed', 'trading-section-feed'].forEach(_intelApplyColumnSplit);
}
window._intelRelayoutColumns = _intelRelayoutColumns;

let _intelResizeRaf = null;
function _intelOnResize() {
  if (!document.getElementById('intel-fullscreen-overlay')?.classList.contains('intel-fs-active')) return;
  if (_intelResizeRaf) return;
  _intelResizeRaf = requestAnimationFrame(function () {
    _intelResizeRaf = null;
    _intelRelayoutColumns();
  });
}

function openIntelFullscreen() {
  const overlay    = document.getElementById('intel-fullscreen-overlay');
  const inner      = document.getElementById('intel-fullscreen-inner');
  const scroll     = document.getElementById('intel-scroll');
  const filterBar  = document.getElementById('ns-filter-bar');
  const newsGroup  = document.getElementById('intel-group-news');
  if (!overlay || !inner || !scroll || !newsGroup) return;
  if (overlay.classList.contains('intel-fs-active')) return;

  _intelFsOriginalScrollParent = scroll.parentNode;
  _intelFsOriginalScrollNext   = scroll.nextSibling;
  inner.appendChild(scroll);

  if (filterBar) {
    _intelFsOriginalFilterParent = filterBar.parentNode;
    _intelFsOriginalFilterNext   = filterBar.nextSibling;
    const newsHead = newsGroup.querySelector('.intel-sub-head');
    newsGroup.insertBefore(filterBar, newsHead ? newsHead.nextSibling : newsGroup.firstChild);
  }

  overlay.classList.add('intel-fs-active');
  document.body.style.overflow = 'hidden';
  _intelFsSetTab('news'); // always opens on News, per Santiago's choice
  _intelRelayoutColumns(); // re-check the 1400px breakpoint now that .intel-fs-active is set
}

function closeIntelFullscreen() {
  const overlay = document.getElementById('intel-fullscreen-overlay');
  const scroll  = document.getElementById('intel-scroll');
  const filterBar = document.getElementById('ns-filter-bar');
  if (!overlay || !overlay.classList.contains('intel-fs-active')) return;

  overlay.classList.remove('intel-fs-active');
  document.body.style.overflow = '';

  if (_intelFsOriginalScrollParent && scroll) {
    _intelFsOriginalScrollParent.insertBefore(scroll, _intelFsOriginalScrollNext);
  }
  if (_intelFsOriginalFilterParent && filterBar) {
    _intelFsOriginalFilterParent.insertBefore(filterBar, _intelFsOriginalFilterNext);
  }
  _intelFsOriginalScrollParent = null;
  _intelFsOriginalScrollNext   = null;
  _intelFsOriginalFilterParent = null;
  _intelFsOriginalFilterNext   = null;
  _intelRelayoutColumns(); // flatten back to one column — docked view is never split, any width
}

function _intelFsWireUp() {
  document.getElementById('intel-fs-btn')?.addEventListener('click', openIntelFullscreen);
  document.getElementById('intel-fs-close')?.addEventListener('click', closeIntelFullscreen);
  // Scoped to #intel-fullscreen-overlay — see the matching comment in
  // _intelFsSetTab(). Without this scope, the Pairs-matrix timeframe
  // buttons (also .intel-fs-tab, different overlay) would each get a
  // second, harmless-but-wasteful click listener wired here too.
  document.querySelectorAll('#intel-fullscreen-overlay .intel-fs-tab').forEach(function (btn) {
    btn.addEventListener('click', function () { _intelFsSetTab(btn.dataset.tab); });
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && document.getElementById('intel-fullscreen-overlay')?.classList.contains('intel-fs-active')) {
      closeIntelFullscreen();
    }
  });
  window.addEventListener('resize', _intelOnResize);
}
// dashboard.js is deferred so the DOM is already parsed by the time this
// runs — DOMContentLoaded may have already fired (same pattern used
// elsewhere in this file, e.g. giOnboardInit above), so check readyState
// rather than blindly waiting on an event that may never come.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _intelFsWireUp);
} else {
  _intelFsWireUp();
}

// ═══════════════════════════════════════════════════════════════════
// EXCLUSIVE FULL-PANEL TABS — News & Derivatives
// ═══════════════════════════════════════════════════════════════════
// News and Derivatives are "exclusive" panels: when one is active it takes
// over the whole #split-lower-right area, replacing the regular sections
// that live there as siblings (Risk, FX Pairs, etc).
//
// v8.21.5: REWRITTEN. The previous implementation gave News and Derivatives
// each their own independent show/hide pair, each with a private dataset
// key (data-news-hidden / data-deriv-hidden) used to snapshot and later
// restore sibling display values, plus a capture-phase click listener on
// every OTHER nav link that called stopImmediatePropagation() to make sure
// it ran before the generic scroll handler.
//
// stopImmediatePropagation() halts ALL subsequent listeners on the same
// element — not just the bubble-phase one it was meant to block, but also
// any other capture-phase listeners registered after it on that same node.
// Derivatives' own "show myself" listener was registered (at boot) directly
// on the Derivatives link, before News' "hide me, a different tab was
// clicked" listener got attached to that same link. So clicking Derivatives
// while News was open ran Derivatives' listener first, which called
// stopImmediatePropagation() immediately — News' teardown never got a
// chance to run. showDerivatives() then re-snapshotted every sibling's
// CURRENT display (since News' hideNews() never restored them), capturing
// "display:none" as #section-risk's "original" value to restore to later
// instead of its true pre-News state. The next time the user navigated to
// a regular section (e.g. Risk), whichever exclusive panel was open
// restored Risk from that corrupted snapshot — display:none — leaving a
// black panel. This only reproduced after visiting News and Derivatives
// back-to-back, which is why it didn't show up on every single News→Risk
// or Derivatives→Risk transition in isolation.
//
// Fixed by removing the snapshot/restore bookkeeping entirely. The set of
// exclusive panels is fixed and known ahead of time, so there's nothing to
// snapshot: regular siblings are simply visible whenever no exclusive panel
// is active, hidden whenever one is — one deterministic function of the
// current target, not per-element saved state that two independent modules
// can race over. One shared click listener replaces the two independent
// ones, so each nav link only ever has a single capture-phase listener and
// stopImmediatePropagation isn't needed at all.
const _EXCLUSIVE_PANEL_IDS = ['section-news', 'section-derivatives'];
window._activeExclusivePanel = null;

// v8.21.6 FIX: `el.style.display = ''` does NOT restore an element's
// original inline display value — it clears the inline `display` property
// entirely, falling back to whatever a stylesheet rule says (UA default
// `block` for a <div> if nothing matches). Three non-exclusive siblings of
// #split-lower-right declare display on purpose in their inline style:
// section-tvcalendar (display:flex + fixed height:180px), the sessions-row
// wrapper (display:grid, no id), and section-econmap (display:flex + fixed
// height:440px). Once toggled to 'none' and back to '', they silently fell
// back to display:block — breaking their flex/grid-dependent internal
// sizing while the fixed inline height stayed put with no overflow:hidden
// to contain it. Content then overflowed the fixed-height box and visually
// spilled onto whatever sat below it in source order — e.g. Economic
// Calendar bleeding into Cross-Asset/Risk after a News/Derivatives round
// trip. Fix: cache each sibling's TRUE original display value once, read
// directly from the DOM before any toggle ever runs, and restore to that
// cached value instead of an empty string. Cached by element reference
// (not id) since the sessions-row wrapper has none.
const _origSiblingDisplay = new WeakMap();
(function _cacheOriginalSiblingDisplay() {
  const splitLowerRight = document.getElementById('split-lower-right');
  if (!splitLowerRight) return;
  Array.from(splitLowerRight.children).forEach(el => {
    if (!_EXCLUSIVE_PANEL_IDS.includes(el.id)) {
      _origSiblingDisplay.set(el, el.style.display || '');
    }
  });
})();

function _setExclusivePanel(targetId) {
  const splitLowerRight = document.getElementById('split-lower-right');
  if (!splitLowerRight) return;
  Array.from(splitLowerRight.children).forEach(el => {
    if (_EXCLUSIVE_PANEL_IDS.includes(el.id)) {
      if (el.id === targetId) {
        // v8.117.19 FIX: `el.style.display = ''` clears the inline `display`
        // property entirely and falls back to the UA default (`block` for a
        // <div>) — it does NOT restore the element's original shown-state
        // display, exactly the failure mode the v8.21.6 comment above
        // already diagnosed for this function's sibling-restore branch, but
        // left unfixed here for the exclusive panel itself. #section-news's
        // HTML declares `display:none;flex-direction:column;height:100%` —
        // it needs `display:flex` when shown, or `flex-direction`/`height:
        // 100%` do nothing and its #intel-scroll child's `flex:1` sizing (and
        // everything nested under it) has no real parent height to grow
        // into, collapsing to ~0px regardless of #intel-scroll's own CSS.
        // #section-derivatives has no `flex-direction` in its inline style,
        // so falling back to block via '' is correct for it — left as-is.
        el.style.display = (el.id === 'section-news') ? 'flex' : '';
      } else {
        el.style.display = 'none';
      }
    } else {
      el.style.display = targetId ? 'none' : (_origSiblingDisplay.get(el) || '');
    }
  });
  window._activeExclusivePanel = targetId || null;
}

function _repaintAfterExclusivePanelClosed() {
  // Canvas/chart containers can go stale while sized 0×0 behind an exclusive
  // panel. Double rAF: #split-lower-right uses display:contents, which needs
  // two frames for the browser to commit the layout change before
  // clientWidth/offsetWidth are trustworthy again (same pattern used by the
  // ticker strip elsewhere in this file).
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
      // Force-repaint the main LWC chart — canvas backing store can go stale
      // when the chart container's layout was recomputed while hidden.
      const _chartWrap = document.getElementById('tv-chart-wrap');
      if (typeof _lwChart !== 'undefined' && _lwChart && _chartWrap) {
        const w = _chartWrap.offsetWidth, h = _chartWrap.offsetHeight;
        if (w > 0 && h > 0) try { _lwChart.resize(w, h); _lwReapplyPaneHeights(); _lwReprojectDrawings(); } catch(_) {}
      }
      // Sidebar liquidity canvas — same repaint-after-restore pattern.
      if (typeof drawLiquidityChart === 'function') drawLiquidityChart();
    });
  });
}

function showNews() {
  _setExclusivePanel('section-news');
  const splitLower = document.getElementById('split-lower');
  if (splitLower) splitLower.scrollTo({ top: 0, behavior: 'smooth' });
  // Re-render news + trading sub-panels with current data
  renderNewsSection();
  // Load or re-render research sub-panel
  if (!_researchAllItems.length) {
    loadBankResearch();
  } else {
    renderResearchSection();
  }
}

function hideNews() {
  _setExclusivePanel(null);
  _repaintAfterExclusivePanelClosed();
}

function showDerivatives() {
  _setExclusivePanel('section-derivatives');
  const splitLower = document.getElementById('split-lower');
  if (splitLower) splitLower.scrollTo({ top: 0, behavior: 'smooth' });
  renderDerivativesSection();
}

function hideDerivatives() {
  _setExclusivePanel(null);
  _repaintAfterExclusivePanelClosed();
}

function initExclusivePanelNav() {
  const newsSection  = document.getElementById('section-news');
  const derivSection = document.getElementById('section-derivatives');
  if (!newsSection || !derivSection) return;
  if (!document.getElementById('split-lower-right')) return;

  // One capture-phase listener per nav link. Capture phase still runs before
  // the generic bubble-phase scroll handler (registered elsewhere in this
  // file), so panel visibility is always resolved before that handler tries
  // to scroll/scrollIntoView — but unlike before, there's nothing left to
  // block, so no stopImmediatePropagation.
  document.querySelectorAll('.top-nav a[data-target]').forEach(link => {
    link.addEventListener('click', () => {
      const target = link.dataset.target;
      if (target === 'section-news') {
        showNews();
      } else if (target === 'section-derivatives') {
        showDerivatives();
      } else if (window._activeExclusivePanel) {
        _setExclusivePanel(null);
        _repaintAfterExclusivePanelClosed();
      }
    }, true);
  });

  // Expose for keyboard shortcuts (N / D)
  window._newsNavShow     = showNews;
  window._newsNavHide     = hideNews;
  window._newsNavSection  = newsSection;
  window._derivNavShow    = showDerivatives;
  window._derivNavHide    = hideDerivatives;
  window._derivNavSection = derivSection;

  // Expose filter setter for inline onclick
  window._newsSetFilter   = _newsSetFilter;
}

(function bootNewFeatures() {
  const run = async () => {
    initG8RatesTabs();
    initCOTAssetTabs();
    initCorrAssetTabs();
    _corrMtxWireHover();
    _corrMtxFsWireUp();
    initSentimentAssetTabs();
    initExclusivePanelNav();

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

    // v8.163.0: renderEconSurprises() (inline ESI sidebar table) was only
    // ever called once, in the Promise.all above at boot — no interval
    // refreshed it afterward, so a new economic actual never showed up
    // without a full page reload, even though calendar.json itself now
    // updates near-real-time upstream (v8.162.0's repository_dispatch
    // bridge). renderEconSurprises() is idempotent (fetches fresh and
    // resets its own dedup guard each call) so it's safe to re-run on an
    // interval. 3-min cadence matches econ-matrix.js's polling and
    // calendar-panel.js's existing Economic Calendar refresh, since all
    // three now read from the same near-real-time calendar.json chain.
    setInterval(renderEconSurprises, 3 * 60 * 1000);
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
  // FIX-WL-5: this whitelist had drifted out of sync with the site's canonical
  // 32-pair G10 FX catalogue (heatmap-modal.js's PAIR_DEFS), silently rejecting
  // valid pairs that already have full price + chart data (e.g. AUD/CHF,
  // USD/SEK, USD/NOK, EUR/NZD) — the "symbol not found" bug. Kept in sync with
  // PAIR_DEFS' 32 pairs + XAU/XAG. Not extended to crypto (BTC/ETH): those are
  // cross-asset context quotes only, not a supported FX product on this platform.
  var SYMBOL_MAP = {
    'EURUSD': 'EURUSD', 'GBPUSD': 'GBPUSD', 'USDJPY': 'USDJPY',
    'AUDUSD': 'AUDUSD', 'USDCAD': 'USDCAD', 'USDCHF': 'USDCHF',
    'NZDUSD': 'NZDUSD', 'GBPJPY': 'GBPJPY', 'EURJPY': 'EURJPY',
    'EURGBP': 'EURGBP', 'AUDJPY': 'AUDJPY', 'EURAUD': 'EURAUD',
    'EURCHF': 'EURCHF', 'EURCAD': 'EURCAD', 'GBPCHF': 'GBPCHF',
    'GBPCAD': 'GBPCAD', 'GBPAUD': 'GBPAUD', 'CADJPY': 'CADJPY',
    'CHFJPY': 'CHFJPY', 'NZDJPY': 'NZDJPY', 'AUDNZD': 'AUDNZD',
    'EURNZD': 'EURNZD', 'GBPNZD': 'GBPNZD', 'AUDCHF': 'AUDCHF',
    'AUDCAD': 'AUDCAD', 'NZDCHF': 'NZDCHF', 'NZDCAD': 'NZDCAD',
    'CADCHF': 'CADCHF', 'USDNOK': 'USDNOK', 'USDSEK': 'USDSEK',
    'EURNOK': 'EURNOK', 'EURSEK': 'EURSEK',
    'XAUUSD': 'XAUUSD', 'XAGUSD': 'XAGUSD',
  };

  // FIX-WL-5: quotes.json stores metals under 'gold'/'silver', not 'xauusd'/
  // 'xagusd' — a straight sym.toLowerCase() lookup for those two always
  // missed, so XAU/XAG rows could be added but their price never loaded.
  var QUOTE_KEY_ALIAS = { 'XAUUSD': 'gold', 'XAGUSD': 'silver' };

  // Map watchlist symbol to TradingView FX_IDC symbol used by loadTVChart / sidebar handler.
  // FIX-WL-6 (v8.161.3): XAUUSD/XAGUSD were previously given the same generic
  // FX_IDC: prefix as FX pairs — but _TV_TO_OHLC (dashboard.js) only recognises
  // OANDA:XAUUSD/OANDA:XAGUSD as the keys that resolve to the Gold/Silver Futures
  // LW chart. FX_IDC:XAUUSD/FX_IDC:XAGUSD matched no _TV_TO_OHLC entry, so every
  // watchlist click on a metals row silently fell through to the TradingView
  // widget fallback — the same incident already fixed for Retail FX Positioning's
  // renderSentiment() (see RETAIL_SENT_METAL_TV_SYM). The comment previously here
  // documented that fallback as "correct behaviour for commodities" — it was not;
  // XAU/XAG have live OHLC data and should open the LW chart like every other
  // watchlist symbol, with TradingView only as a true last-resort fallback.
  var TV_SYM_PREFIX = 'FX_IDC:';
  var METAL_TV_SYM = { 'XAUUSD': 'OANDA:XAUUSD', 'XAGUSD': 'OANDA:XAGUSD' };

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
      var qKey = (QUOTE_KEY_ALIAS[sym] || sym).toLowerCase();
      var q = quotes[qKey] || {};
      var price = (q.close != null) ? String(q.close) : (quotesReady ? '—' : '···');
      var chg = (q.pct != null) ? q.pct : null;
      var chgStr = (chg != null) ? ((chg >= 0 ? '+' : '') + chg.toFixed(2) + '%') : (quotesReady ? '—' : '···');
      var chgColor = (chg == null) ? 'var(--text3)' : (chg >= 0 ? 'var(--up)' : 'var(--down)');
      var tvSym = METAL_TV_SYM[sym] || (TV_SYM_PREFIX + sym);
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
const _TF_DEFAULT_DAYS = {H1:5,H4:14,get D1(){ return _lwDefaultD1Days(); },W1:365,MN:1095};

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
  // v8.172.0: no longer clears compare here — _renderLWChart() below now
  // destroys the chart (wiping the old TF's compare series with it, via
  // _destroyLWChart's _lwCompareSeriesMap reset) and re-applies every
  // persisted compare entry fresh against the new _lwActiveTf, so the
  // reloaded overlay already has the right TF's data with no separate clear
  // step needed — same fix that makes compare survive leaving/returning to
  // the chart also covers a TF switch, since both go through the same
  // destroy+rebuild path.
  // Scoped to #lw-range-bar (2026-08-07 fix): this used to be an unscoped
  // document.querySelectorAll('.lw-tf-btn'), which also matched the CSI
  // panel's timeframe buttons (heatmap-modal.js reused the same class) once
  // that panel had been built — so switching the main chart's TF silently
  // re-highlighted the CSI modal's TF row underneath it, out of sync with
  // its own _csiTf state. The CSI panel now has its own .hm-csi-btn class
  // instead, but this stays scoped as a hard guarantee against any future
  // widget reusing .lw-tf-btn and hitting the same cross-contamination.
  document.querySelectorAll('#lw-range-bar .lw-tf-btn').forEach(b => b.classList.toggle('sel', b.dataset.tf === newTf));
  _lwUpdateRangeBtns();
  if (_lwActiveOhlcId) _renderLWChart(_lwActiveOhlcId);
});

// =============================================================================
// COMPARE OVERLAY — normalised % change LineSeries on secondary price scale
// =============================================================================

// MOBILE FIX (Santiago report, 2026-08-19): "+ Compare" did nothing on mobile.
// Root cause — #lw-cmp-dropdown was left as static HTML nested inside
// #lw-range-bar's row-1 div, which has `overflow-x:auto` + WebKit's
// `-webkit-overflow-scrolling:touch` (needed for iOS momentum-scroll on that
// horizontally-scrollable toolbar). Switching the dropdown to
// `position:fixed` via JS is normally enough to escape an ancestor's overflow
// clipping — but `-webkit-overflow-scrolling:touch` is a well-known WebKit/iOS
// exception: it forces the whole subtree (including `position:fixed`
// descendants) into the container's own scrolling compositor layer, so the
// dropdown stayed clipped to the toolbar's row bounds and effectively
// invisible/untappable on iOS Safari/Chrome-iOS, while working fine on
// desktop (no touch/overflow quirk there) and even Android in some cases —
// consistent with Santiago not flagging any other browser.
// The Indicators (_lw-ind-dropdown) and Draw (_lw-draw-dropdown) popups never
// hit this because they're `document.createElement`'d and
// `document.body.appendChild()`'d fresh on every open — never a descendant of
// the scrollable toolbar row to begin with. Fix: reparent #lw-cmp-dropdown to
// <body> once (appendChild on an already-attached node just moves it, so this
// is idempotent — safe to run on every open), same pattern already proven for
// the other two chart dropdowns. Also added viewport clamping (both dropdowns'
// existing "min-width pushes it off the right/bottom edge on mobile" fix,
// mirrored here) since a body-level fixed element can now legitimately be
// positioned by any button anywhere in the toolbar, not just one that happens
// to sit at the row's visible right edge.
(function _lwCmpDropdownToBody() {
  const dd = document.getElementById('lw-cmp-dropdown');
  if (dd && dd.parentElement !== document.body) document.body.appendChild(dd);
})();

// Toggle compare dropdown open/close
document.getElementById('lw-cmp-btn')?.addEventListener('click', function(e) {
  e.stopPropagation();
  const dd = document.getElementById('lw-cmp-dropdown');
  if (!dd) return;
  if (dd.parentElement !== document.body) document.body.appendChild(dd); // safety net
  const open = dd.style.display === 'none' || !dd.style.display;
  if (open) {
    // Position with fixed coords to escape any overflow:hidden/scrolling ancestor
    const rect = this.getBoundingClientRect();
    dd.style.position  = 'fixed';
    dd.style.zIndex    = '9100';
    dd.style.display   = 'block';
    // Measure after display:block so offsetWidth/offsetHeight are real.
    const ddW = dd.offsetWidth  || 175;
    const ddH = dd.offsetHeight || 260;
    // Horizontal: prefer right-aligned under the button, clamp into viewport
    // with an 8px margin on both edges (same margin the Indicators dropdown
    // fix already uses) so it's reachable regardless of where the button sits
    // in the horizontally-scrolled toolbar.
    let left = rect.right - ddW;
    if (left < 8) left = 8;
    if (left + ddW > window.innerWidth - 8) left = window.innerWidth - ddW - 8;
    // Vertical: flip above the button if there isn't room below.
    const spaceBelow = window.innerHeight - rect.bottom;
    const top = spaceBelow >= 80 ? rect.bottom + 4 : Math.max(8, rect.top - ddH - 4);
    dd.style.left  = left + 'px';
    dd.style.right = 'auto';
    dd.style.top   = top + 'px';
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
  // v8.172.0: toggle OFF only this exact symbol if it's already being
  // compared — every other active compare overlay is left untouched.
  // Previously _lwLoadCompare() unconditionally cleared the single active
  // slot before adding the new one, so picking a second symbol silently
  // erased the first — compare now supports any number of simultaneous
  // overlays, matching TradingView/Bloomberg's multi-compare convention,
  // removed only via its own pill (or re-clicking its own dropdown row).
  if (_lwCompareSeriesMap[uid]) { _lwClearCompareOne(uid); return; }
  _lwLoadCompare(cmpId, item.textContent.trim(), cmpType);
  document.getElementById('lw-cmp-dropdown').style.display = 'none';
});

// Removes ONE compare overlay by uid ("<type>:<id>") — its series (if the
// chart still has one), its pill, and its dropdown active-state. Unless
// `keepPersisted` is passed (used only by the self-compare skip in
// _renderLWChart's restore pass), the entry is also dropped from
// window._lwCompareList + localStorage, so a user-initiated removal via the
// pill's × actually stays removed on the next chart rebuild instead of
// silently coming back.
function _lwClearCompareOne(uid, keepPersisted) {
  const series = _lwCompareSeriesMap[uid];
  if (series && _lwChart) { try { _lwChart.removeSeries(series); } catch(_e) {} }
  delete _lwCompareSeriesMap[uid];
  document.querySelectorAll('.lw-cmp-pill').forEach(function (p) {
    if (p.dataset.uid === uid) p.remove();
  });
  const entry = (window._lwCompareList || []).find(function (e) { return e.uid === uid; });
  const uType = entry ? entry.cmpType : uid.slice(0, uid.indexOf(':'));
  const uId   = entry ? entry.cmpId   : uid.slice(uid.indexOf(':') + 1);
  document.querySelectorAll('.lw-cmp-item').forEach(function (i) {
    if ((i.dataset.cmptype || 'ohlc') === uType && i.dataset.cmpid === uId) i.classList.remove('active');
  });
  if (!keepPersisted) {
    window._lwCompareList = (window._lwCompareList || []).filter(function (e) { return e.uid !== uid; });
    _lsSetCompare(window._lwCompareList);
  }
}

async function _lwLoadCompare(cmpId, cmpLabel, cmpType = 'ohlc', fromRestore) {
  if (!_lwChart || !_lwCandleSeries) return;
  const LWC = window.LightweightCharts;
  const uid = cmpType + ':' + cmpId;
  if (_lwCompareSeriesMap[uid]) return; // already active — callers guard this, but stay idempotent

  // ── Colour per type, cycling within-type (v8.172.0) ──────────────────────
  // Compare could only ever hold one overlay before, so one fixed colour per
  // type was enough. Now that multiple overlays of the same type can be
  // active together (e.g. two OHLC price comparisons), reusing one identical
  // colour for both would make them indistinguishable on the chart — each
  // type gets a small palette instead, and the Nth active overlay of a given
  // type takes the Nth colour in that type's palette, so the ordering stays
  // predictable session to session.
  const CMP_PALETTES = {
    cot:  ['#9c27b0', '#ce93d8', '#6a1b9a'],
    rate: [_themeColor('--up'), '#66bb6a', '#00897b'],
    esi:  [_themeColor('--chart-line'), '#4fc3f7', '#1565c0'],
    ohlc: [_themeColor('--orange'), '#ffb74d', '#e65100'],
  };
  const palette = CMP_PALETTES[cmpType] || CMP_PALETTES.ohlc;
  const sameTypeCount = Object.keys(_lwCompareSeriesMap)
    .filter(function (k) { return k.indexOf(cmpType + ':') === 0; }).length;
  const CMP_COLOR = palette[sameTypeCount % palette.length];

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
        // NOISE_KW / INVERSE_KW — use module-level shared consts (hoisted v8.28.0).
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
            const canon = _canonEsi(name);
            const aS = String(ev.actual||'').replace(/[%,\s]/g,'');
            const fS = String(ev.forecast||ev.previous||'').replace(/[%,\s]/g,'');
            const key = `${cmpId}/${canon}/${aS}/${fS}`;
            if (seen.has(key)) return;
            seen.add(key);
            const actual   = _parseNum(ev.actual);
            const forecast = _parseNum(ev.forecast || ev.previous);
            if (isNaN(actual) || isNaN(forecast)) return;
            const inv     = INVERSE_KW.some(k => name.includes(k));
            const beat    = inv ? actual < forecast : actual > forecast;
            const miss    = inv ? actual > forecast : actual < forecast;
            const surp    = inv ? -(actual-forecast) : (actual-forecast);
            const ageDays    = (endMs - t) / 86400000;
            const impactMult = ev.impact === 'high' ? 1.0 : 0.5;
            const w          = Math.exp(-DECAY_LAMBDA * ageDays) * impactMult;
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
    const cmpSeries = LWC.LineSeries
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

    cmpSeries.setData(seriesData);
    _lwCompareSeriesMap[uid] = cmpSeries;

    document.querySelectorAll('.lw-cmp-item').forEach(i =>
      i.classList.toggle('active',
        (i.dataset.cmptype || 'ohlc') === cmpType && i.dataset.cmpid === cmpId));

    // Add pill — one per active compare (data-uid keyed, not a singleton id
    // — v8.172.0). Skipped if a pill for this exact uid already exists,
    // which happens on the restore pass after a chart rebuild: the pill
    // survived the rebuild untouched (it lives outside the chart's own DOM
    // subtree), only its underlying series needed re-attaching above.
    const indPills = document.getElementById('lw-ind-pills');
    if (indPills && !Array.from(document.querySelectorAll('.lw-cmp-pill')).some(function (p) { return p.dataset.uid === uid; })) {
      const pill = document.createElement('span');
      pill.className = 'lw-cmp-pill';
      pill.dataset.uid = uid;
      pill.title = 'Remove compare overlay';
      pill.innerHTML = `<span style="width:8px;height:2px;background:${CMP_COLOR};display:inline-block;border-radius:1px;"></span> ${cmpLabel} ×`;
      pill.addEventListener('click', function () { _lwClearCompareOne(uid); });
      indPills.parentNode.insertBefore(pill, indPills);
    }

    // Persist — unless this call is itself the "re-apply the persisted list
    // on chart rebuild" pass (fromRestore), in which case the entry is
    // already in window._lwCompareList and re-adding it would duplicate it.
    if (!fromRestore) {
      window._lwCompareList = (window._lwCompareList || []).filter(function (e) { return e.uid !== uid; });
      window._lwCompareList.push({ uid: uid, cmpId: cmpId, cmpLabel: cmpLabel, cmpType: cmpType });
      _lsSetCompare(window._lwCompareList);
    }
  } catch(err) {
    console.warn('[lw-compare] Failed to load compare data:', err.message);
    if (fromRestore) {
      // The persisted symbol's data is no longer loadable (e.g. the
      // instrument was removed/renamed) — drop the dead entry instead of
      // leaving a permanently broken pill + a failing re-fetch attempt on
      // every future chart render, and clear any stale pill left over from
      // before this render (it would otherwise read as "still comparing"
      // with nothing behind it, the exact bug this session set out to fix).
      window._lwCompareList = (window._lwCompareList || []).filter(function (e) { return e.uid !== uid; });
      _lsSetCompare(window._lwCompareList);
      document.querySelectorAll('.lw-cmp-pill').forEach(function (p) {
        if (p.dataset.uid === uid) p.remove();
      });
    }
  }
}

// =============================================================================
// FULLSCREEN CHART — DOM-lift: move the real chart panel into the overlay
// This preserves ALL indicators, compare series, CB markers, event handlers.
// =============================================================================

let _lwFsOriginalParent = null;
let _lwFsOriginalNext   = null;
let _lwFsOriginalHeight = null;

// Oscillator sub-panes (RSI, MACD, Stochastic, etc.) are given a fixed pixel
// height via pane.setHeight(80 or 90) when built — see _buildIndicatorPane.
// Lightweight Charts does NOT treat that as a hard floor across a chart.resize():
// a large total-height change (e.g. leaving fullscreen, ~900px tall → ~290px)
// proportionally rescales every pane, including ones with an explicit
// setHeight() — a known library limitation (tradingview/lightweight-charts#1847).
// The result was exactly what was reported: after exiting fullscreen the
// oscillator strip (and, by the same proportional math, the main price pane)
// came back squashed/misproportioned instead of respecting its intended 80/90px.
// Re-applying setHeight() right after every resize() restores the fixed
// heights the same way _buildIndicatorPane originally set them.
function _lwReapplyPaneHeights() {
  if (!_lwChart || !window._indPaneIndex) return;
  try {
    const panes = _lwChart.panes();
    Object.keys(window._indPaneIndex).forEach(id => {
      const idx = window._indPaneIndex[id];
      if (idx == null) return;
      const paneH = (id === 'macd' || id === 'adx') ? 90 : 80;
      panes[idx]?.setHeight(paneH);
    });
  } catch(_) {}
}

// Companion fix for the other half of the same symptom: the drawing overlay
// (trend lines, rectangles, Fib guides — see _renderDrawings) only
// re-projects on timeScale().subscribeVisibleTimeRangeChange, i.e. pan/zoom.
// A resize() can shrink or grow the main pane's price-scale mapping (its
// pixel height changed) with the visible *time* range completely unchanged,
// so that listener never fires and every drawing is left rendered at its
// pre-resize pixel position — reading as shapes "shifted" relative to the
// candles they were anchored to. Re-running the reproject right after
// resize keeps drawings pinned to their actual price/time coordinates.
function _lwReprojectDrawings() {
  if (typeof window._lwRenderDrawings === 'function') window._lwRenderDrawings();
}

// Shared re-measure step for any change that alters #tv-chart-wrap's
// available height/width WITHOUT firing a window 'resize' event — a
// DOM-lift (fullscreen open/close) is one case, already handled inline in
// _lwOpenFullscreen()/_lwCloseFullscreen(); toggling a sibling panel
// (Seasonality) that shares the same flex column is another, added here
// specifically so _sznToggle() can call one shared, tested code path
// instead of a third inline copy of the same three-call sequence.
window._lwResizeAfterLayoutChange = function () {
  const chartWrap = document.getElementById('tv-chart-wrap');
  if (!_lwChart || !chartWrap) return;
  const w = chartWrap.offsetWidth, h = chartWrap.offsetHeight;
  if (w > 0 && h > 0) { _lwChart.resize(w, h, true); _lwReapplyPaneHeights(); _lwReprojectDrawings(); }
};

function _lwOpenFullscreen() {
  const overlay   = document.getElementById('lw-fullscreen-overlay');
  const inner     = document.getElementById('lw-fullscreen-inner');
  const rangeBar  = document.getElementById('lw-range-bar');
  const sznPanel  = document.getElementById('szn-panel');
  const chartHdr  = document.getElementById('lw-chart-header');
  const chartWrap = document.getElementById('tv-chart-wrap');
  if (!overlay || !inner || !chartWrap || _chartMode !== 'lw') return;
  if (overlay.classList.contains('lw-fs-active')) return;

  // Store anchor: the element immediately BEFORE rangeBar so we can
  // restore the full block (rangeBar→sznPanel→chartHdr→chartWrap) in one shot.
  _lwFsOriginalParent = rangeBar ? rangeBar.parentNode : chartWrap.parentNode;
  _lwFsOriginalNext   = chartWrap.nextSibling;     // element AFTER chartWrap
  _lwFsOriginalHeight = chartWrap.style.height;

  // Lift all elements into the fullscreen inner container. szn-panel is a
  // DOM sibling of rangeBar (not nested inside it — see index-beta.html),
  // so it was previously left behind: the Seasonality button lives inside
  // rangeBar and got lifted, but toggling it just flipped display:block on
  // a panel still sitting in the page underneath this overlay (z-index
  // 9000, position:fixed, covers the viewport) — invisible. Lifting it
  // here alongside the rest fixes that.
  if (rangeBar)  inner.appendChild(rangeBar);
  if (sznPanel)  inner.appendChild(sznPanel);
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
      // forceRepaint:true — paint immediately instead of on the next tick, so
      // there's no single frame at the old (pre-fullscreen) canvas size.
      if (w > 0 && h > 0) { _lwChart.resize(w, h, true); _lwReapplyPaneHeights(); _lwReprojectDrawings(); }
    }
    // Seasonality's LWC chart also needs an explicit resize after the DOM
    // move — it's a separate chart instance from the price chart and
    // won't pick up the new (wider) fullscreen width on its own.
    if (typeof window._sznResizeChart === 'function') window._sznResizeChart();
  }));
}

function _lwCloseFullscreen() {
  const overlay   = document.getElementById('lw-fullscreen-overlay');
  const inner     = document.getElementById('lw-fullscreen-inner');
  const rangeBar  = document.getElementById('lw-range-bar');
  const sznPanel  = document.getElementById('szn-panel');
  const chartHdr  = document.getElementById('lw-chart-header');
  const chartWrap = document.getElementById('tv-chart-wrap');
  if (!overlay || !overlay.classList.contains('lw-fs-active')) return;

  overlay.classList.remove('lw-fs-active');
  document.body.style.overflow = '';

  // Restore all elements before the stored next-sibling reference.
  // insertBefore with a null ref appends to end, which is also correct.
  if (_lwFsOriginalParent) {
    if (rangeBar)  _lwFsOriginalParent.insertBefore(rangeBar,  _lwFsOriginalNext);
    if (sznPanel)  _lwFsOriginalParent.insertBefore(sznPanel,  _lwFsOriginalNext);
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
      // forceRepaint:true (immediate paint) + _lwReapplyPaneHeights() — without
      // this the chart came back from fullscreen with panes proportionally
      // rescaled from the ~900px fullscreen height down to ~290px, squashing
      // the oscillator strip; _lwReprojectDrawings() re-pins every drawing to
      // its real price/time coordinates, since resize() alone doesn't fire
      // the visible-time-range event the overlay normally redraws on. See
      // _lwReapplyPaneHeights() above for detail.
      if (w > 0 && h > 0) { _lwChart.resize(w, h, true); _lwReapplyPaneHeights(); _lwReprojectDrawings(); }
    }
    if (typeof window._sznResizeChart === 'function') window._sznResizeChart();
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

  // Wire ‹ › scroll buttons (same logic as tv-tabs-prev/next in main toolbar).
  // Hardened to match the lw-tb-row1/row2 pattern: a single setTimeout(50)
  // measures scrollWidth/clientWidth once, right after this function rebuilds
  // fsTabs' innerHTML from scratch every open — if that measurement lands
  // before the fullscreen overlay's display:none -> flex transition has
  // actually reflowed (36 buttons, real layout work, on a slower device/
  // frame), clientWidth can read as its old (or 0) value and the arrows
  // never show even though real overflow exists — same class of bug already
  // fixed for Row 2 (async width changes needing more than one measurement
  // point). Also now re-checks on window resize, which it never did before.
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
    window.addEventListener('resize', updateArrows);
    [0, 50, 200, 800].forEach(ms => setTimeout(updateArrows, ms));
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

// ═══════════════════════════════════════════════════════════════════════════
// RESEARCH SECTION — Bank & institutional FX research notes
// Mirrors News/Derivatives show/hide pattern. Shortcut: B.
// Data source: research-data/bank-research.json (fetch_bank_research.py)
// Industry standard: Bloomberg Research Monitor row layout.
// Copyright compliant: title + bank + url only — no content reproduction.
// ═══════════════════════════════════════════════════════════════════════════

let _researchAllItems  = [];
let _researchMeta      = {};
let _researchFilter    = { bank: 'ALL', cur: 'ALL' };
let _researchNewsItems = [];  // news-feed items reclassified as research (InvestMacro, Marc to Market, FX Markets)

// Bank badge CSS class helper
function _resBankClass(bank) {
  const map = { ING: 'ING', Saxo: 'Saxo', MUFG: 'MUFG', DailyFX: 'DailyFX', BIS: 'BIS', CME: 'CME', UBS: 'UBS' };
  return 'rs-bank-' + (map[bank] || 'other');
}

function renderResearchSection(items, meta) {
  if (Array.isArray(items)) _researchAllItems = items;
  if (meta) _researchMeta = meta;

  const feed = document.getElementById('research-section-feed');
  if (!feed) return;

  // Convert _researchNewsItems (from news pipeline) to research-format objects
  const newsAsResearch = (_researchNewsItems || []).map(function(ni) {
    return {
      bank:      ni.source || '',
      bank_full: ni.source || '',
      title:     ni.title  || '',
      series:    '',
      author:    '',
      url:       ni.link   || ni.url || '',
      currencies: ni.cur ? [ni.cur] : [],
      pairs:     [],
      category:  'macro',
      ts:        ni.ts || (ni.datetime ? new Date(ni.datetime).getTime() : 0),
    };
  });

  // Merge bank-research.json items + reclassified news items, sort newest first
  const merged = _researchAllItems.concat(newsAsResearch).sort(function(a, b) {
    return (b.ts || 0) - (a.ts || 0);
  });
  const filtered = merged.filter(function(item) {
    const bankOk = _researchFilter.bank === 'ALL' || item.bank === _researchFilter.bank;
    const curOk  = _researchFilter.cur  === 'ALL' ||
                   (Array.isArray(item.currencies) && item.currencies.includes(_researchFilter.cur));
    return bankOk && curOk;
  });

  const tsEl = document.getElementById('research-section-ts');
  if (tsEl && _researchMeta.updated_label) tsEl.textContent = _researchMeta.updated_label;

  const countEl = document.getElementById('research-section-count');
  if (countEl) countEl.textContent = filtered.length + ' notes';

  feed.innerHTML = '';

  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'rs-empty';
    empty.textContent = 'No research notes match current filter.';
    feed.appendChild(empty);
    _intelApplyColumnSplit('research-section-feed');
    return;
  }

  filtered.forEach(function(item) {
    // ── Time label: "HH:MM · Ns" for recent items, "HH:MM · Mon D" past a day ──
    let timeLabel = '--:--';
    if (item.ts) {
      const pubDate = new Date(item.ts);
      const hm      = pubDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
      const ageMs   = Date.now() - item.ts;
      const ageMin  = Math.floor(ageMs / 60000);
      const ageHr   = Math.floor(ageMs / 3600000);
      const ageDays = Math.floor(ageMs / 86400000);
      if (ageDays >= 1) {
        timeLabel = hm + ' \u00b7 ' + pubDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
      } else if (ageHr >= 1) {
        timeLabel = hm + ' \u00b7 ' + ageHr + 'h';
      } else if (ageMin >= 1) {
        timeLabel = hm + ' \u00b7 ' + ageMin + 'm';
      } else {
        timeLabel = hm + ' \u00b7 now';
      }
    }

    const bank       = item.bank       || '';
    const bankFull   = item.bank_full  || bank;
    const title      = item.title      || '';
    const series     = item.series     || '';
    const url        = item.url        || '';
    const excerpt    = item.excerpt    || '';
    const currencies = item.currencies || [];
    const pairs      = item.pairs      || [];
    const category   = item.category   || 'macro';
    const safeUrl    = url.startsWith('https://') ? url : '';
    const isTradeIdea = category === 'trade_idea';

    let displayTitle = title;
    if (series && title.toLowerCase().startsWith(series.toLowerCase())) {
      displayTitle = title.slice(series.length).replace(/^[\s:–—]+/, '');
    }
    displayTitle = displayTitle || title;

    // ── Article block — always-expanded, matching News/Analysis (.ns-art-*) ────
    const wrap = document.createElement('div');
    wrap.className = 'rs-article' + (isTradeIdea ? ' rs-art-trade-idea' : '');

    // Meta row: source · time · currency tags
    const meta = document.createElement('div');
    meta.className = 'rs-art-meta';

    if (bank) {
      const srcEl = document.createElement('span');
      srcEl.className = 'rs-art-source';
      srcEl.textContent = bank;
      srcEl.title = bankFull;
      meta.appendChild(srcEl);
    }

    const timeEl = document.createElement('span');
    timeEl.className = 'rs-art-time';
    timeEl.textContent = timeLabel;
    meta.appendChild(timeEl);

    currencies.slice(0, 2).forEach(function(cur) {
      const tag = document.createElement('span');
      tag.className = 'rs-cur-tag';
      tag.textContent = cur;
      meta.appendChild(tag);
    });
    wrap.appendChild(meta);

    // Title — clickable headline when a safe link is available
    const titleEl = document.createElement('div');
    titleEl.className = 'rs-art-title';
    titleEl.title = displayTitle; // full text on hover — compact mode truncates with ellipsis
    if (safeUrl) {
      const a = document.createElement('a');
      a.href = safeUrl;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = displayTitle;
      titleEl.appendChild(a);
    } else {
      titleEl.textContent = displayTitle;
    }
    wrap.appendChild(titleEl);

    // Body — excerpt always visible; falls back to series + pairs when the
    // source provided no description (e.g. RSS items without a summary)
    let body = excerpt;
    if (!body && (series || pairs.length)) {
      const parts = [];
      if (series) parts.push(series);
      if (pairs.length) parts.push(pairs.slice(0, 3).join(' \u00b7 '));
      body = parts.join('  \u00b7  ');
    }
    if (body) {
      const bodyEl = document.createElement('div');
      bodyEl.className = 'rs-art-body';
      bodyEl.textContent = body;
      wrap.appendChild(bodyEl);
    }

    feed.appendChild(wrap);
  });

  // Wide-monitor two-column layout (Intel Fullscreen only) — see
  // _intelApplyColumnSplit() near openIntelFullscreen() below.
  _intelApplyColumnSplit('research-section-feed');
}

function _researchSetFilter(type, value) {
  _researchFilter[type] = value;
  const selector = type === 'bank' ? '.rs-bank-pill' : '.rs-cur-pill';
  document.querySelectorAll(selector).forEach(function(btn) {
    btn.classList.toggle('rs-pill-active', btn.dataset.val === value);
  });
  renderResearchSection();
}
// Exposed for the bank-filter pills' onclick handlers in index.html
// (data-val buttons under the News→Research sub-panel).
window._researchSetFilter = _researchSetFilter;

async function loadBankResearch() {
  try {
    const resp = await fetch('research-data/bank-research.json?v=' + Date.now());
    if (!resp.ok) {
      console.warn('[Research] bank-research.json not found — pipeline may not have run yet');
      const feed = document.getElementById('research-section-feed');
      if (feed) {
        feed.innerHTML = '<div class="rs-empty">Research data not yet available. Pipeline runs every 4 hours.</div>';
      }
      return;
    }
    const data = await resp.json();
    renderResearchSection(data.items || [], {
      updated_label: data.updated_label || '',
      total:         data.total || 0,
    });
  } catch (e) {
    console.error('[Research] Load failed:', e);
  }
}

// NOTE: there used to be an initResearchNav() here, an independent
// show/hide pair for a standalone "Research" tab. It targeted
// #section-research, which no longer exists in index.html (Research is now
// a sub-panel rendered inside News via renderResearchSection(), driven from
// showNews() above) and the function was never called from anywhere.
// Removed as dead code during the v8.21.5 exclusive-panel-nav rewrite.

// ═══════════════════════════════════════════════════════════════════
// THEME-CHANGE HANDLER
// Re-applies color-sensitive components when the user switches theme.
// ═══════════════════════════════════════════════════════════════════
window.addEventListener('gi-theme-change', function() {
  // 1. LWC price chart — update layout colors live
  if (typeof _lwChart !== 'undefined' && _lwChart) {
    try {
      _lwChart.applyOptions({
        layout: {
          background: { color: _themeColor('--bg') },
          textColor:  _themeColor('--text'),
        },
        grid: {
          vertLines: { color: _themeColorAlpha('--border', 0.5) },
          horzLines: { color: _themeColorAlpha('--border', 0.5) },
        },
        rightPriceScale: { borderColor: _themeColor('--border') },
        timeScale:       { borderColor: _themeColor('--border') },
      });
    } catch(_) {}

    // 1b. Recolor the main price series (candle/bar up-down colors, line/area chart-line)
    if (window._candleSeries) {
      try {
        const t = window._candleSeriesType;
        if (t === 'line') {
          window._candleSeries.applyOptions({ color: _themeColor('--chart-line') });
        } else if (t === 'area') {
          window._candleSeries.applyOptions({
            lineColor: _themeColor('--chart-line'),
            topColor:  _themeColorAlpha('--chart-line', 0.28),
            bottomColor: _themeColorAlpha('--chart-line', 0.02),
          });
        } else {
          // candle or bar
          window._candleSeries.applyOptions({
            upColor:         _themeColor('--candle-up'),
            downColor:       _themeColor('--candle-down'),
            borderUpColor:   _themeColor('--candle-up'),
            borderDownColor: _themeColor('--candle-down'),
            wickUpColor:     _themeColor('--candle-up'),
            wickDownColor:   _themeColor('--candle-down'),
          });
        }
      } catch(_) {}
    }
  }

  // 2. Yield curve canvas — redraw with new theme colors
  if (typeof drawYieldCurve === 'function' &&
      typeof _lastDrawnYields !== 'undefined' && _lastDrawnYields) {
    drawYieldCurve(_lastDrawnYields, _lastDrawnPrior);
  }

  // 3. Liquidity chart — redraw
  if (typeof drawLiquidityChart === 'function') {
    try { drawLiquidityChart(); } catch(_) {}
  }
});

// ═══════════════════════════════════════════════════════════════════
// SEASONALITY PANEL (beta) — button in the chart's Row 2 toolbar opens a
// monthly seasonal-return panel for the currently active chart symbol.
//
// Data source: seasonality-data/{pair}.json, written by the private
// scripts repo's compute_seasonality.py (server-side, off ohlc-data D1
// bars — see that script's header for methodology and its explicit
// monthly-not-daily scope note). FX pairs only — 32 majors/crosses match
// compute_seasonality.py's PAIRS list; indices/metals/crypto/rates aren't
// computed and the button shows a plain "not available for this symbol"
// state rather than silently doing nothing.
//
// Chart: reuses the same Lightweight Charts library instance already
// loaded for the Price Chart (_ensureLWLib()) rather than pulling in a
// second charting dependency — an Area series plotted against synthetic
// sequential dates (LWC requires a real increasing time axis; the axis
// itself is hidden and real month labels are rendered in the #szn-months
// row below it instead, same "hide the axis, label separately" approach
// already used elsewhere in this file for non-time x-axes).
// ═══════════════════════════════════════════════════════════════════
(function () {
  // Extended (v8.196.0) to match compute_seasonality.py's PAIRS list —
  // that script was widened to every non-FX id fetch_ohlc.py populates
  // with real 10y OHLC (metals, energy, equity indices, crypto, DXY,
  // VIX/MOVE, us10y/us5y); this gate must stay in sync or a symbol with
  // a real seasonality-data/{id}.json file would still show the "FX
  // pairs only" message. hyoas/igoas (fetch_credit_spreads.py, different
  // script) and us2y (no OHLC proxy exists) are excluded on both sides.
  const SZN_PAIRS = new Set([
    'eurusd','gbpusd','usdjpy','audusd','usdchf','usdcad','nzdusd',
    'usdnok','usdsek','eurnok','eursek','eurgbp','eurjpy','eurchf',
    'eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd',
    'audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd',
    'audcad','cadchf','nzdcad','nzdchf',
    'gold','silver','wti','brent','btc','eth','dxy','vix','move',
    'us10y','us5y','spx','nasdaq','nikkei','stoxx','dax','ftse',
    'hsi','dji',
  ]);

  let _sznChart = null, _sznSeries = null, _sznOpen = false, _sznLoadedPair = null, _sznCrosshairHandler = null;
  // v8.211.0 — actual pixel width LWC reserves for the right price scale
  // (rightPriceScale: { minimumWidth: 50 } above, plus its border — comes
  // out to ~56px live). #szn-months must exclude this from the width it
  // distributes its 12 columns across; see _sznRenderMonthLabels() below
  // for why.
  let _sznRightScaleWidth = 0;

  function _sznPairLabel(pair) {
    return pair.length === 6 ? (pair.slice(0, 3) + '/' + pair.slice(3)).toUpperCase() : pair.toUpperCase();
  }

  function _sznDestroyChart() {
    if (_sznChart) { try { _sznChart.remove(); } catch (_) {} }
    _sznChart = null; _sznSeries = null; _sznCrosshairHandler = null;
  }

  async function _sznRenderChart(curve) {
    const el = document.getElementById('szn-chart');
    // LWC is not a global — every other chart in this file (Price Chart,
    // etc.) pulls it from window.LightweightCharts locally. This IIFE
    // never did, so `typeof LWC === 'undefined'` was always true and the
    // chart silently no-op'd on every open. Same fix as the other call
    // sites: resolve it from window, and ensure the library is actually
    // loaded (Price Chart lazy-loads it via _ensureLWLib(); if this panel
    // is opened before that resolves, load it here too).
    if (typeof window._ensureLWLib === 'function') {
      try { await window._ensureLWLib(); } catch (_) {}
    }
    const LWC = window.LightweightCharts;
    if (!el || typeof LWC === 'undefined') return;
    _sznDestroyChart();
    el.innerHTML = '';

    _sznChart = LWC.createChart(el, {
      // Was `background: { color: 'transparent' }` — LWC's canvas paint
      // doesn't reliably render a true transparent backdrop once the pane
      // actually has content to draw (grid lines, series); in practice it
      // fell back to a lighter internal default gray-blue, visible as a
      // "gray box" filling the whole plot area regardless of hover (this
      // was already there before any mouse interaction — confirmed against
      // a screenshot with the cursor away from the chart). The main Price
      // Chart never relies on 'transparent' for exactly this reason — it
      // sets its background explicitly to the real backdrop color
      // (`--bg`, the same color `body`/`#szn-panel` actually paint behind
      // it). Matched that here instead of trusting 'transparent'.
      layout: { background: { color: _themeColor('--bg') }, textColor: _themeColor('--text'), attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: _themeColorAlpha('--border', 0.5) } },
      rightPriceScale: { borderColor: _themeColor('--border'), minimumWidth: 50 },
      timeScale: { visible: false, borderVisible: false },
      handleScroll: false, handleScale: false,
      // Explicit crosshair theming — without vertLine/horzLine colors set,
      // LWC falls back to its own library-default gray (line + label
      // background), which reads as an out-of-place gray box against this
      // theme on hover. The main Price Chart already themes this (see
      // _lwChart's own `crosshair:` block above); this panel's chart is a
      // separate LWC instance and needs the same treatment explicitly.
      crosshair: {
        mode: LWC.CrosshairMode.Normal,
        // v8.210.0 fix — vertLine.labelVisible was never explicitly set,
        // so it kept LWC's default `true`. That label draws into the
        // time-scale pane, but this chart sets `timeScale: { visible:
        // false }` (the real dates are shown by our own custom
        // #szn-months row below instead, sized/positioned to match the
        // day-of-year curve — see _sznRenderMonthLabels() above). With
        // the pane hidden, LWC's own hover-date label had nowhere
        // correct to paint and was spilling out below the chart in its
        // un-themed library-default color, overlapping/miscolored
        // against our custom month row — most visible as a stray blue
        // "Dec" near the end of the strip on hover. This chart already
        // has its own themed tooltip for date+value (#szn-chart-
        // tooltip, driven by subscribeCrosshairMove() below), so the
        // native vertLine label is fully redundant once disabled — not
        // a workaround, an actual dupe. Every OTHER LWC chart in this
        // file (Price Chart, Rates & Yield Curve, etc.) keeps its real
        // timeScale visible and never hides it, so this leak is unique
        // to this one chart's custom-axis pattern.
        vertLine: { color: _themeColorAlpha('--text2', 0.5), labelVisible: false },
        horzLine: { color: _themeColorAlpha('--text2', 0.5), labelBackgroundColor: _themeColor('--bg3') },
      },
      localization: { priceFormatter: v => v.toFixed(2) + '%' },
      width: el.clientWidth || 580,
      height: 120,
    });

    // Line/fill color: was `--down` (red) — no basis for that on a chart
    // that isn't showing a directional loss/decline; every other single-
    // series chart in this file (Price Chart's Area mode, Rates & Yield
    // Curve's 10Y chart, etc.) uses the dedicated `--chart-line` blue for
    // a neutral historical series, which is also the industry-standard
    // convention for a non-directional seasonal/statistical curve like
    // this one (EquityClock, Seasonax). Switched to match.
    const seriesOpts = {
      lineColor: _themeColor('--chart-line'), lineWidth: 1.6,
      topColor: _themeColorAlpha('--chart-line', 0.10), bottomColor: _themeColorAlpha('--chart-line', 0.01),
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: true,
    };
    _sznSeries = (typeof LWC.AreaSeries !== 'undefined')
      ? _sznChart.addSeries(LWC.AreaSeries, seriesOpts)
      : _sznChart.addAreaSeries(seriesOpts);

    // v8.195.0 — curve is now day-of-year (367 points: index 0 = pre-
    // Jan-1 baseline, then a real {month,day} for every calendar day of
    // a reference leap year — see compute_seasonality.py's
    // _build_daily_curve()). Unlike the old 13-point synthetic-date
    // scheme (spaced 14 days apart purely to satisfy LWC's strictly-
    // ascending-time requirement), every point here already carries a
    // REAL calendar date, so no synthetic spacing is needed — a genuine
    // Date.UTC(refYear, month-1, day) walk is ascending by construction
    // for every day of the year, including Feb 29 in the leap reference
    // year used here (2024). The baseline point (month:0, day:0) is
    // placed one day before Jan 1 of that same reference year so it
    // still sorts strictly before the first real point.
    const REF_YEAR = 2024; // leap year — accommodates the Feb 29 point
    const pts = curve.map(pt => {
      const d = (pt.month === 0)
        ? new Date(Date.UTC(REF_YEAR - 1, 11, 31)) // baseline: Dec 31 of the prior year
        : new Date(Date.UTC(REF_YEAR, pt.month - 1, pt.day));
      return {
        time: `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`,
        value: pt.cum_pct,
      };
    });
    _sznSeries.setData(pts);
    _sznChart.timeScale().fitContent();

    // v8.211.0 — capture the right price scale's real live pixel width
    // (rightPriceScale: { minimumWidth: 50 } above is a floor LWC can
    // exceed once real % values are laid out, e.g. "-1.00%" — must read
    // the actual rendered width, not assume the minimum). This is the
    // gutter #szn-months has to exclude from its own width below it —
    // see _sznRenderMonthLabels() for the actual fix; this only captures
    // the number. Guarded because priceScale().width() can legitimately
    // return 0 before the chart's first paint.
    try {
      const w = _sznChart.priceScale('right').width();
      if (w > 0) _sznRightScaleWidth = w;
    } catch (_) { /* keep previous value rather than zeroing it out */ }

    // v8.208.0 — hover tooltip (date + value). crosshairMarkerVisible:true
    // (set in seriesOpts above) only draws the dot on the line itself; LWC
    // does not render any text next to it on its own — a text readout needs
    // a manually-positioned DOM element driven by subscribeCrosshairMove(),
    // the same pattern the main Price Chart already uses for its own
    // hover readout (see _lwChart's crosshair-move handler elsewhere in
    // this file). REF_YEAR is a synthetic placeholder year (see comment
    // above pts) so the tooltip formats {month, day} directly rather than
    // showing that fake year to the user.
    const pointByTime = {};
    pts.forEach((p, i) => { pointByTime[p.time] = curve[i]; });
    let tip = document.getElementById('szn-chart-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.id = 'szn-chart-tooltip';
      tip.style.cssText = 'position:absolute;display:none;pointer-events:none;z-index:5;background:var(--bg3);border:1px solid var(--border2);border-radius:3px;padding:3px 7px;font-size:10px;font-family:var(--font-mono,monospace);color:var(--text);white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.35);';
      el.style.position = el.style.position || 'relative';
      el.appendChild(tip);
    }
    // _sznChart is a fresh instance every call (see _sznDestroyChart() above),
    // so there's no prior subscription on THIS chart to remove — no unsubscribe
    // call needed here, unlike a chart instance that persists across renders.
    const MONTH_FULL = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    _sznCrosshairHandler = param => {
      if (!param.point || !param.time || param.point.x < 0 || param.point.y < 0) {
        tip.style.display = 'none';
        return;
      }
      const pt = pointByTime[param.time];
      if (!pt) { tip.style.display = 'none'; return; }
      const dateLabel = (pt.month === 0) ? 'Baseline' : `${MONTH_FULL[pt.month - 1]} ${pt.day}`;
      const valTxt = (pt.cum_pct >= 0 ? '+' : '') + pt.cum_pct.toFixed(2) + '%';
      tip.innerHTML = `${dateLabel} &middot; <span style="color:${pt.cum_pct >= 0 ? 'var(--up)' : 'var(--down)'};">${valTxt}</span>`;
      tip.style.display = 'block';
      // Clamp so the tooltip never spills past the container's right/top edge.
      const maxLeft = el.clientWidth - tip.offsetWidth - 4;
      const left = Math.max(4, Math.min(param.point.x + 10, maxLeft));
      const top = Math.max(2, param.point.y - 28);
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
    };
    _sznChart.subscribeCrosshairMove(_sznCrosshairHandler);
  }

  const _SZN_MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  // v8.200.0: below each month label, show that month's own average
  // cumulative move (end-of-month cum_pct minus start-of-month cum_pct,
  // from the SAME additive day-of-year curve already fetched — no new
  // backend field, no new request). This is deliberately descriptive-only
  // (no win_rate, no p-value, no MIN_YEARS gate) — it does NOT feed or
  // relax the windows table's t-test significance gate above; the two are
  // independent, same "descriptive month-average alongside a stricter
  // significance-gated table" pattern used by reference tools like
  // EquityClock.
  //
  // v8.208.0 REVERT: the inline second line (month name + its own avg %
  // stacked below it inside the same grid cell) reintroduced exactly the
  // failure mode v8.203.0 had just closed for the month-name-only version —
  // a grid cell's row height is fixed by `grid-template-rows:1fr` against
  // the row container's own height, but that height was only ever sized
  // (via #szn-months' `min-height:22px`) for a single line of text. Once a
  // cell's content needs two stacked lines (name + value) taller than that
  // fixed row box, the overflow doesn't reflow the grid — it just spills
  // past the row's bottom edge, visually detached below the rest of the
  // strip. December was the visible case Santiago hit, but the mechanism
  // has nothing to do with December specifically (every month's two-line
  // stack is equally taller than the box; December's simply happened to be
  // the one whose overflow amount crossed into visibly separate territory
  // at the widths tested — a real width/font-metrics coincidence, not a
  // December-specific bug). Fixed at the root by reverting this row to
  // month-name-only (one line, matches its `min-height:22px` box exactly)
  // per Santiago's explicit instruction, rather than patching around it
  // with a taller fixed row height or an invisible spacer — the average-
  // move figures move to their own tab below instead (see
  // `_sznRenderMonthlyTable()`), where a table row has no such fixed-height
  // constraint to fight.
  let _sznMonthlyCols = []; // shared with _sznRenderMonthlyTable() — same curve, computed once per load
  function _sznRenderMonthLabels(curve) {
    const row = document.getElementById('szn-months');
    if (!row) return;
    // v8.195.0 — curve is now 367 day-of-year points instead of 13
    // monthly ones, so a label can no longer be emitted 1:1 per curve
    // point (that would print 366 labels). Instead, derive one label
    // per calendar month from the point where day===1 (every month has
    // exactly one such point in the 367-point curve), and size each
    // label's column width proportionally to how many days that month
    // actually spans in the curve — so the label row still lines up
    // approximately under the chart's real (non-uniform, since months
    // have 28-31 days) time axis below, rather than 12 equal-width
    // slots implying every month is the same length.
    //
    // v8.202.0 fix (incomplete — see v8.203.0 below): switched this row
    // from display:flex (flex:{span} 0 0 per item, no guaranteed single-row
    // fit) to CSS Grid with an explicit `grid-template-columns` built from
    // the real per-month spans — a grid's column count/order are fixed by
    // that declaration, unlike flex's content-dependent line-breaking.
    //
    // v8.203.0 fix: December kept dropping to its own row even after the
    // v8.202.0 grid switch — root cause was a second bug introduced by that
    // same fix: `grid-auto-flow: column` was set alongside the explicit
    // 12-column `grid-template-columns`. `column` flow places items down
    // the ROW axis first, wrapping to a new column only once the grid's row
    // count is exhausted — and since no `grid-template-rows` is set here,
    // that row count isn't reliably pinned to 1, so the auto-placement
    // algorithm doesn't guarantee 12 items land one-per-column in a single
    // row the way `row` flow (the CSS default) does for 12 items against
    // 12 explicit columns. Fixed by removing the `column` override — plain
    // `row` flow (left at its default, not set explicitly) fills the 12
    // explicit column tracks left-to-right in exactly one row, which is
    // the only behavior actually wanted here; there is no case where this
    // row should ever wrap, so no grid-auto-flow value should try to.
    const monthStarts = [];
    curve.forEach((p, i) => { if (p.month >= 1 && p.day === 1) monthStarts.push({ month: p.month, idx: i }); });

    const cols = monthStarts.map((m, i) => {
      const nextIdx = (i + 1 < monthStarts.length) ? monthStarts[i + 1].idx : curve.length;
      const span = nextIdx - m.idx; // days in this month, from the actual curve
      const startPct = curve[m.idx].cum_pct;
      const endPct = (nextIdx < curve.length) ? curve[nextIdx].cum_pct : curve[curve.length - 1].cum_pct;
      const monthPct = endPct - startPct;
      return { month: m.month, span, monthPct };
    });

    _sznMonthlyCols = cols; // stash for the Monthly Avg tab table — same data, no re-fetch/re-derive

    row.style.display = 'grid';
    row.style.gridTemplateColumns = cols.map(c => `${c.span}fr`).join(' ');
    row.style.gridTemplateRows = '1fr'; // pin to exactly one row — belt-and-suspenders alongside removing the column auto-flow below
    row.style.gridAutoFlow = 'row'; // explicit, not left to inherit — this must never be 'column' (see v8.203.0 comment above)
    row.style.gap = '1px';
    row.style.flexWrap = ''; // clear the stale flex-wrap inline value, if any, left over from this row's pre-v8.202.0 flex layout

    // v8.211.0 fix — root cause of "December sits under the price-scale
    // column" (confirmed via a live console diagnostic, not assumed):
    // #szn-months previously defaulted to the full width of #szn-panel
    // (same as the chart container, 404px in the diagnostic capture),
    // but LWC's canvas only actually plots the series across
    // (containerWidth - rightScaleWidth) — 348px in that same capture,
    // a ~56px gutter reserved for the "1.00% / 0.00% / -1.00%" price
    // labels. Distributing 12 grid columns across the full 404px put
    // the later months (Nov/Dec) increasingly right of where the real
    // curve ends, landing Dec visibly under the empty price-scale
    // gutter instead of under the chart's actual right edge. Capping
    // this row's own width to exclude that same gutter (measured live
    // off the chart itself in _sznRenderChart(), not hardcoded) makes
    // its 12 columns span the identical pixel range LWC uses for the
    // curve, so the last column lines up with the real end of the data
    // instead of the empty space beside it.
    row.style.width = _sznRightScaleWidth > 0 ? `calc(100% - ${_sznRightScaleWidth}px)` : '100%';

    // Month-name-only, single line — see v8.208.0 comment above for why the
    // inline value line was reverted.
    row.innerHTML = cols.map(c =>
      `<span style="text-align:center;line-height:22px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${_SZN_MONTH_LABELS[c.month - 1]}</span>`
    ).join('');

    _sznRenderMonthlyTable(cols);
  }

  // v8.209.0 — rebuilt as a horizontal table (months as columns, one
  // Average row + a Yearly Return column), matching the Seasonax/
  // EquityClock "Total Percent Returns" layout convention Santiago
  // referenced — this is the industry-standard shape for a compact
  // monthly-seasonality table, and uses far less vertical space than the
  // v8.208.0 row-per-month version. Cell backgrounds use a light tint of
  // the same --up/--down variables the rest of this app already uses for
  // sign, rather than the reference image's light-mode green/red (which
  // wouldn't read correctly against this dark theme).
  function _sznRenderMonthlyTable(cols) {
    const head = document.getElementById('szn-monthly-h-head');
    const row = document.getElementById('szn-monthly-h-row');
    if (!head || !row) return;
    const cellStyle = 'padding:5px 4px;text-align:center;border-left:1px solid var(--border2);';
    // v8.212.0 — every <th>/<td> here now carries an explicit
    // background (transparent, same visual result as before) instead
    // of none. Without it, the global `tr:hover td { background:
    // var(--bg3); }` rule (dashboard.css) is the only background these
    // cells have, so hovering the row darkened ONLY the ones lacking an
    // inline background — "Average"/the header row — while the numeric
    // cells below (which already set an inline background, even when
    // it's 'transparent', and inline always wins over that CSS rule)
    // stayed visually unchanged, reading as "Average turns gray on
    // hover". Exact same root cause as the correlation matrix's row/
    // corner-header hover artifact (GUIDELINES.md, v8.185.0) — a cell
    // missing the same explicit background every sibling cell has,
    // exposed specifically by :hover repaint rather than a real :hover
    // rule targeting it.
    head.innerHTML = '<th style="padding:5px 4px;text-align:left;font-weight:400;background:transparent;">Month</th>'
      + cols.map(c => `<th style="${cellStyle}font-weight:400;background:transparent;">${_SZN_MONTH_LABELS[c.month - 1]}</th>`).join('')
      + `<th style="${cellStyle}font-weight:400;border-left:2px solid var(--border);background:transparent;">Yearly</th>`;

    const yearly = cols.reduce((sum, c) => sum + c.monthPct, 0);
    const fmtCell = (v, wideDivider) => {
      const flat = Math.abs(v) < 0.05;
      const color = flat ? 'var(--text3)' : (v > 0 ? 'var(--up)' : 'var(--down)');
      const bg = flat ? 'transparent' : (v > 0 ? 'rgba(38,166,154,0.14)' : 'rgba(239,83,80,0.14)');
      const txt = (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
      const border = wideDivider ? 'border-left:2px solid var(--border);' : 'border-left:1px solid var(--border2);';
      return `<td style="padding:5px 4px;text-align:center;${border}background:${bg};color:${color};">${txt}</td>`;
    };
    row.innerHTML = '<td style="padding:5px 4px;color:var(--text);text-align:left;background:transparent;">Average</td>'
      + cols.map(c => fmtCell(c.monthPct)).join('')
      + fmtCell(yearly, true);
  }

  // v3.0 (2026-08-21 industry-standard audit): win_rate demoted from primary
  // gate/sort key to context-only column — Seasonax's own published
  // methodology treats hit rate as the LEAST significant of its reported
  // stats. The real gate is now a one-sample t-test p-value (computed
  // server-side in compute_seasonality.py), and avg_return is now shown
  // alongside std_dev rather than alone, since dispersion is part of the
  // story. Windows are also now guaranteed non-overlapping (server-side
  // dedup) so this table can no longer show the same pattern 2-3 times.
  function _sznRenderWindows(windows) {
    const tbody = document.getElementById('szn-windows-tbody');
    if (!tbody) return;
    if (!windows || !windows.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text3);padding:4px 0;">No window reached statistical significance (p&lt;0.05) for this pair over the available history.</td></tr>';
      return;
    }
    const M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    tbody.innerHTML = windows.map(w => `
      <tr style="color:var(--text);border-top:1px solid var(--border2);">
        <td style="padding:3px 0;">${M[w.start_month - 1]} \u2192 ${M[w.end_month - 1]}</td>
        <td style="text-align:right;color:${w.dir === 'Short' ? 'var(--down)' : 'var(--up)'};">${w.dir}</td>
        <td style="text-align:right;color:${w.avg_return < 0 ? 'var(--down)' : 'var(--up)'};">${w.avg_return > 0 ? '+' : ''}${w.avg_return}% \u00b1 ${w.std_dev != null ? w.std_dev : '\u2014'}%</td>
        <td style="text-align:right;color:var(--text3);">${w.win_rate}%</td>
        <td style="text-align:right;">${w.p_value != null ? w.p_value : '\u2014'}</td>
      </tr>`).join('');
  }

  function _sznRenderInsight(data, pair) {
    const insight = document.getElementById('szn-insight');
    if (!insight) return;
    const top = data.windows && data.windows[0];
    const label = _sznPairLabel(pair);
    if (!top) {
      insight.textContent = `${label} has ${data.years} years of history but no window reached statistical significance (p<0.05, one-sample t-test) over that period \u2014 no strong recurring seasonal pattern found.`;
      return;
    }
    const M = ['January','February','March','April','May','June','July','August','September','October','November','December'];
    const dirWord = top.dir === 'Short' ? 'weakness' : 'strength';
    // Sample-size caveat is conditional on the pair's REAL data.years (not
    // assumed) \u2014 fetch_ohlc.py v1.15 widened PERIOD 10y->20y specifically
    // to close this gap, but real yfinance/CFD depth still varies per pair,
    // so this must keep reading the live value rather than asserting a fixed
    // "well short" claim now that some pairs can clear 15y+.
    let sampleNote;
    if (data.years < 15) {
      sampleNote = `${data.years}y of history is well short of the 15-25y sample size seasonality research typically recommends`;
    } else if (data.years <= 25) {
      sampleNote = `${data.years}y of history is within the 15-25y sample size seasonality research typically recommends`;
    } else {
      sampleNote = `${data.years}y of history exceeds the 15-25y sample size seasonality research typically recommends`;
    }
    insight.textContent = `${label} showed ${dirWord} between ${M[top.start_month - 1]} and ${M[top.end_month - 1]} across the last ${top.n_years} years (avg ${top.avg_return > 0 ? '+' : ''}${top.avg_return}% \u00b1 ${top.std_dev}%, p=${top.p_value}; held in ${top.win_rate}% of qualifying years). Not a predictive signal \u2014 a historical statistical tendency, and ${sampleNote}. Windows above use monthly granularity; the chart uses day-of-year granularity.`;
  }

  async function _sznLoad(pair) {
    const insight = document.getElementById('szn-insight');
    const title = document.getElementById('szn-title');
    const tbody = document.getElementById('szn-windows-tbody');
    const monthsRow = document.getElementById('szn-months');
    const monthlyHead = document.getElementById('szn-monthly-h-head');
    const monthlyRow = document.getElementById('szn-monthly-h-row');
    const chartSection = document.getElementById('szn-chart-section');
    const windowsSection = document.getElementById('szn-windows-section');

    if (!pair || !SZN_PAIRS.has(pair)) {
      if (title) title.textContent = 'Daily \u00b7 10y lookback';
      if (insight) insight.textContent = 'Seasonality isn\u2019t available for this symbol \u2014 select an FX pair, metal, index, or other supported instrument on the Price Chart above.';
      if (tbody) tbody.innerHTML = '';
      if (monthsRow) monthsRow.innerHTML = '';
      if (monthlyHead) monthlyHead.innerHTML = '';
      if (monthlyRow) monthlyRow.innerHTML = '';
      if (chartSection) chartSection.style.display = 'none';
      if (windowsSection) windowsSection.style.display = 'none';
      if (typeof window._sznSwitchTab === 'function') window._sznSwitchTab('chart');
      _sznDestroyChart();
      _sznLoadedPair = null;
      return;
    }
    if (pair === _sznLoadedPair) return; // already showing this pair

    if (title) title.textContent = `${_sznPairLabel(pair)} \u00b7 Daily \u00b7 10y lookback`;
    if (insight) insight.textContent = 'Loading seasonality data\u2026';
    // Reset to visible before the fetch — a prior pair may have hit the
    // catch branch below and hidden these; this pair might succeed.
    if (chartSection) chartSection.style.display = '';
    if (windowsSection) windowsSection.style.display = '';

    try {
      const res = await fetch(`./seasonality-data/${pair}.json`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      await _sznRenderChart(data.curve);
      _sznRenderMonthLabels(data.curve);
      _sznRenderWindows(data.windows);
      _sznRenderInsight(data, pair);
      if (title) title.textContent = `${_sznPairLabel(pair)} \u00b7 Daily \u00b7 ${data.years}y lookback`;
      _sznLoadedPair = pair;
    } catch (e) {
      // A missing file means the pair didn't clear MIN_YEARS in
      // compute_seasonality.py (or hasn't run yet for this pair) — same
      // "nothing to show" treatment as an unsupported symbol: hide the
      // empty chart/table shell rather than leaving a blank black box
      // and an empty table above the one line of real text.
      if (insight) insight.textContent = `No seasonality data yet for ${_sznPairLabel(pair)} \u2014 needs at least 5 years of stored daily history.`;
      if (tbody) tbody.innerHTML = '';
      if (monthsRow) monthsRow.innerHTML = '';
      if (monthlyHead) monthlyHead.innerHTML = '';
      if (monthlyRow) monthlyRow.innerHTML = '';
      if (chartSection) chartSection.style.display = 'none';
      if (windowsSection) windowsSection.style.display = 'none';
      if (typeof window._sznSwitchTab === 'function') window._sznSwitchTab('chart');
      _sznDestroyChart();
      _sznLoadedPair = null;
    }
  }

  function _sznToggle(open) {
    const panel = document.getElementById('szn-panel');
    const btn = document.getElementById('szn-btn');
    if (!panel) return;
    _sznOpen = open != null ? open : panel.style.display === 'none';
    panel.style.display = _sznOpen ? 'block' : 'none';
    if (btn) btn.setAttribute('aria-expanded', String(_sznOpen));
    if (btn) btn.classList.toggle('on', _sznOpen);
    if (_sznOpen) _sznLoad(window._sznActiveOhlcId);
    // Toggling this panel adds/removes a whole block of vertical space
    // above the price chart (#tv-chart-wrap has flex:1 inside the fullscreen
    // overlay's #lw-fullscreen-inner, so its CSS box does shrink/grow
    // correctly) — but the LWC canvas itself was drawn at the OLD pixel
    // height and never gets told to repaint at the new one, since neither
    // display:block/none nor a flex-basis change fires a 'resize' event.
    // Symptom Santiago saw: opening Seasonality inside the fullscreen chart
    // overlay left the price chart candles rendered at their pre-toggle
    // (taller) height, now overflowing/clipped by the shrunk container —
    // reading as "the chart got cut in half", including its time axis at
    // the bottom. Same forceRepaint:true + pane/drawing re-sync already
    // used by _lwOpenFullscreen()/_lwCloseFullscreen() after their own
    // DOM-lift resizes.
    if (typeof window._lwResizeAfterLayoutChange === 'function') {
      requestAnimationFrame(() => requestAnimationFrame(window._lwResizeAfterLayoutChange));
    }
  }

  // Exposed so _renderLWChart's single call site can notify us of symbol
  // changes without this IIFE needing to be defined before that function.
  window._sznOnSymbolChange = function (ohlcId) {
    if (_sznOpen) _sznLoad(ohlcId);
  };

  // Exposed so _lwOpenFullscreen()/_lwCloseFullscreen() can force this
  // chart to re-measure its container after moving it in/out of the
  // fullscreen overlay (a DOM move doesn't fire a window 'resize' event).
  //
  // Was `_sznChart.applyOptions({ width })` — this updates the chart's
  // config but, unlike the main Price Chart's own fullscreen-resize path
  // (_lwChart.resize(w, h, true)), doesn't reliably force the canvas
  // itself to repaint at the new size, nor does it touch the time scale.
  // Net effect matching what Santiago saw: entering fullscreen widened
  // #szn-chart's container, but the chart kept rendering at its old
  // (pre-fullscreen) pixel width — visible as the curve confined to a
  // narrow strip instead of spanning the new, much wider panel. Switched
  // to the same `.resize(width, height, forceRepaint)` call the Price
  // Chart uses, plus `timeScale().fitContent()` to re-spread the 13-point
  // curve across the full new width (resize() alone repaints the canvas
  // at the new size but keeps the same visible logical range).
  window._sznResizeChart = function () {
    if (!_sznChart) return;
    const el = document.getElementById('szn-chart');
    if (!el) return;
    const w = el.clientWidth || 580;
    _sznChart.resize(w, 120, true);
    _sznChart.timeScale().fitContent();
  };

  // v8.208.0 — Chart / Monthly Avg tab strip (same show/hide-by-id onclick
  // pattern the Dollar Smile panel's tab bar used before it was reduced to
  // a single Growth view in v8.213.0 — see the note near _growthdiffRenderTable()).
  window._sznSwitchTab = function (tab) {
    ['chart', 'monthly'].forEach(t => {
      const btn = document.querySelector(`.szn-tab[data-szn-tab="${t}"]`);
      const panel = document.getElementById(`szn-tab-${t}`);
      const active = t === tab;
      if (btn) {
        btn.style.color = active ? 'var(--text)' : 'var(--text3)';
        btn.style.borderBottomColor = active ? 'var(--accent)' : 'transparent';
      }
      if (panel) panel.style.display = active ? '' : 'none';
    });
    // Chart tab's LWC canvas was sized while display:none on this branch of
    // the very first load (its container has 0 width then) — re-measure on
    // every switch back into view, same reasoning as the fullscreen/toggle
    // resize calls above.
    if (tab === 'chart' && typeof window._sznResizeChart === 'function') {
      requestAnimationFrame(() => requestAnimationFrame(window._sznResizeChart));
    }
  };

  document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('szn-btn');
    const closeBtn = document.getElementById('szn-close');
    if (btn) btn.addEventListener('click', () => _sznToggle());
    if (closeBtn) closeBtn.addEventListener('click', () => _sznToggle(false));
    window.addEventListener('resize', () => window._sznResizeChart());
  });
})();

// ── Row 2 toolbar scroll arrows (beta) — same prev/next pattern as
//    #tv-ticker/#tv-pair-tabs (see the existing addWheelScroll IIFE above),
//    added because the Seasonality button pushed Row 2 closer to overflow
//    on narrower viewports. ─────────────────────────────────────────────
(function () {
  const row = document.getElementById('lw-tb-row2');
  const btnPrev = document.getElementById('lw-tb-row2-prev');
  const btnNext = document.getElementById('lw-tb-row2-next');
  if (!row || !btnPrev || !btnNext) return;

  row.addEventListener('wheel', function (e) {
    if (e.deltaY === 0) return;
    e.preventDefault();
    row.scrollLeft += e.deltaY;
  }, { passive: false });

  function updateArrows() {
    const atStart = row.scrollLeft <= 2;
    const atEnd = row.scrollLeft + row.clientWidth >= row.scrollWidth - 2;
    btnPrev.style.display = atStart ? 'none' : 'flex';
    btnNext.style.display = atEnd ? 'none' : 'flex';
  }
  row.addEventListener('scroll', updateArrows, { passive: true });
  window.addEventListener('resize', updateArrows);

  // The single setTimeout(200) below isn't enough on its own: #lw-ind-pills
  // (EMA 50/EMA 20 etc.) and #lw-ind-btn's own overlay pills get appended
  // to this same row asynchronously, after the chart's indicator data has
  // loaded — which can land well after 200ms and after this row's initial
  // scrollWidth was already measured as "no overflow". Nothing was
  // listening for that later width change, so the right arrow only ever
  // appeared once the user manually scrolled (which fires 'scroll' and
  // re-runs updateArrows for the first time). Fix: watch the row itself
  // for content/size changes and re-check on every one of them.
  const mo = new MutationObserver(() => updateArrows());
  mo.observe(row, { childList: true, subtree: true, characterData: true });
  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => updateArrows());
    ro.observe(row);
  }
  // Belt-and-suspenders for browsers/timing where neither observer fires
  // in time (e.g. font swap changing button widths after paint).
  [0, 200, 800, 2000].forEach(ms => setTimeout(updateArrows, ms));

  btnPrev.addEventListener('click', () => row.scrollBy({ left: -160, behavior: 'smooth' }));
  btnNext.addEventListener('click', () => row.scrollBy({ left: 160, behavior: 'smooth' }));
})();

// ── Row 1 toolbar scroll arrows (beta) — same gap as Row 2 had before its
//    own fix: overflow-x:auto with zero affordance that it can scroll.
//    Row 1 (TF/Range/+Compare/Fullscreen) is the one Santiago flagged —
//    it's also the row that gets lifted (via #lw-range-bar) into the
//    fullscreen chart overlay, where the narrower available width made
//    +Compare/Fullscreen silently scroll out of reach. Identical
//    prev/next pattern to the Row 2 IIFE directly above. ──────────────
(function () {
  const row = document.getElementById('lw-tb-row1');
  const btnPrev = document.getElementById('lw-tb-row1-prev');
  const btnNext = document.getElementById('lw-tb-row1-next');
  if (!row || !btnPrev || !btnNext) return;

  row.addEventListener('wheel', function (e) {
    if (e.deltaY === 0) return;
    e.preventDefault();
    row.scrollLeft += e.deltaY;
  }, { passive: false });

  function updateArrows() {
    const atStart = row.scrollLeft <= 2;
    const atEnd = row.scrollLeft + row.clientWidth >= row.scrollWidth - 2;
    btnPrev.style.display = atStart ? 'none' : 'flex';
    btnNext.style.display = atEnd ? 'none' : 'flex';
  }
  row.addEventListener('scroll', updateArrows, { passive: true });
  window.addEventListener('resize', updateArrows);

  // Same reasoning as Row 2's own comment: content here doesn't change
  // asynchronously the way indicator pills do, but the row's available
  // width DOES change the moment fullscreen opens/closes (DOM-lift into
  // #lw-fullscreen-inner) — a ResizeObserver catches that without needing
  // this IIFE to know anything about _lwOpenFullscreen().
  const mo = new MutationObserver(() => updateArrows());
  mo.observe(row, { childList: true, subtree: true, characterData: true });
  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => updateArrows());
    ro.observe(row);
  }
  [0, 200, 800, 2000].forEach(ms => setTimeout(updateArrows, ms));

  btnPrev.addEventListener('click', () => row.scrollBy({ left: -160, behavior: 'smooth' }));
  btnNext.addEventListener('click', () => row.scrollBy({ left: 160, behavior: 'smooth' }));
})();

// ═══════════════════════════════════════════════════════════════════
// DOLLAR SMILE BLOCK (beta) — v3, 2026-08-22.
//
// Now Stephen Jen's ORIGINAL growth-differential framework only (US real
// GDP YoY vs. the equal-weighted rest of the G10) — the market-stress-
// regime proxy version this panel used through v8.201.0 (dollar-smile-
// data/history.json, log_dollar_smile_inputs.py, 4 RISK-ON/CAUTION/MIXED/
// RISK-OFF buckets keyed off VIX/MOVE/gold/SPX/AUDJPY/USDJPY/HY-OAS) has
// been REMOVED from this panel per Santiago's explicit instruction: having
// both a proxy version and the real version stacked in one panel read as
// if one was needed to interpret the other, and it wasn't — each stood on
// its own, so showing both was confusing, not additive. This panel now
// shows only the real thing.
//
// (log_dollar_smile_inputs.py / dollar-smile-data/history.json / its daily
// workflow are UNTOUCHED by this change — still logging real data server-
// side — this is a frontend-only removal. Flag if you'd like that backend
// job decommissioned too; leaving it running is harmless and reversible
// either way, so it wasn't turned off as part of this fix.)
//
// Data: growth-differential-data/history.json, written by
// fetch_growth_differential.py as a full idempotent recompute every run
// (never a daily append — GDP data revises, so a cached differential
// computed off a since-revised print would silently go stale/wrong).
// Real GDP YoY, all 10 G10 currencies from FRED, 121 quarters back to
// 1996-Q1 as of first backfill. Quarterly cadence — every historical
// quarter already has a real value, so unlike the removed proxy axis
// there is no "still accumulating" state to handle here.
//
// v2.0.0 (2026-08-22) — regime scheme changed from a pure growth-
// differential 3-way split to a combined crisis+growth classification,
// after Santiago flagged the chart didn't show a U-shape and a check
// against Jen & Yilmaz's actual framework confirmed why: the smile's left
// tail is a genuine global risk-off/crisis regime, not "the US grows a
// bit slower than the G9 average" — those are different things, and a
// pure growth-differential axis can never isolate the former (see
// fetch_growth_differential.py's module docstring for the full
// reasoning). GLOBAL-RISK-OFF now overrides the growth differential
// whenever the quarter's max VIX close hit 40+, regardless of where the
// US ranked that quarter; the old USD-UNDERPERFORMING bucket is folded
// into CALM-MUDDLING-THROUGH, since — absent an actual crisis — modest
// US underperformance is the theory's weak-dollar middle, not its left
// tail.
//
// Note on shape: the insight text below states what regime_stats actually
// show, not an assumed "classic smile" shape — this file was NOT changed
// based on a live post-fix run (no network access to FRED in the
// dev sandbox), only the classification logic. Verify against the next
// live fetch_growth_differential.py run before claiming the shape itself
// changed.
// ═══════════════════════════════════════════════════════════════════
const _GROWTHDIFF_LABELS = {
  'GLOBAL-RISK-OFF': 'Global Risk-Off',
  'CALM-MUDDLING-THROUGH': 'Calm / Muddling Through',
  'USD-GROWTH-OUTPERFORMING': 'USD Growth Outperforming',
};
// Smile x-axis order: left = crisis/risk-off, middle = calm/muddling
// through, right = US growth outperformance — the two ends are the
// thesis's actual "smile" extremes (see fetch_growth_differential.py
// v2.0.0 for why growth-differential alone previously mislabeled the
// left tail).
const _GROWTHDIFF_SMILE_ORDER = ['GLOBAL-RISK-OFF', 'CALM-MUDDLING-THROUGH', 'USD-GROWTH-OUTPERFORMING'];
const GROWTHDIFF_MIN_SAMPLES = 5; // defensive floor, matches compute_seasonality.py's MIN_YEARS spirit

async function renderDollarSmile() {
  const chartEl = document.getElementById('dsmile-chart');
  const insightEl = document.getElementById('dsmile-insight');
  const currentEl = document.getElementById('dsmile-current');
  const tbody = document.getElementById('growthdiff-tbody');
  if (!chartEl) return; // beta-only element, not present outside index-beta.html

  let doc;
  try {
    const res = await fetch('./growth-differential-data/history.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    doc = await res.json();
    if (!doc || !Array.isArray(doc.quarters) || !doc.quarters.length) throw new Error('malformed history.json');
  } catch (e) {
    if (insightEl) insightEl.textContent = 'Dollar Smile data unavailable right now.';
    if (tbody) tbody.innerHTML = '<tr><td colspan="3" style="padding:4px;color:var(--text3);">No data yet.</td></tr>';
    if (currentEl) currentEl.textContent = '';
    return;
  }

  const cur = doc.current;
  const rawStats = doc.regime_stats || {};
  const stats = {};
  _GROWTHDIFF_SMILE_ORDER.forEach(r => {
    const s = rawStats[r] || { n: 0, n_dxy: 0, avg_dxy_qret: null };
    stats[r] = {
      n: s.n_dxy || 0,
      avg: s.avg_dxy_qret,
      ready: (s.n_dxy || 0) >= GROWTHDIFF_MIN_SAMPLES && s.avg_dxy_qret !== null,
    };
  });

  if (currentEl && cur) {
    const diffTxt = `${cur.diff >= 0 ? '+' : ''}${cur.diff.toFixed(2)}pp`;
    // "Latest available" rather than implying this is the current calendar
    // quarter — real GDP prints ~1 quarter after quarter-end. When the
    // regime is crisis-driven, show the VIX read too — the diff alone
    // isn't why this quarter landed in Global Risk-Off.
    const vixTxt = cur.regime === 'GLOBAL-RISK-OFF' && cur.vix_max != null ? `, VIX max ${cur.vix_max.toFixed(1)}` : '';
    currentEl.textContent = `Latest available: ${cur.quarter} \u2014 ${_GROWTHDIFF_LABELS[cur.regime] || cur.regime} (${diffTxt}${vixTxt})`;
  }

  // Earliest quarter carrying a dxy_qret, not hardcoded — so this label
  // stays correct if dxy.json's history is ever backfilled further back
  // than 2006 (see GUIDELINES: two different "n" values in one row must
  // each be labeled with what they actually cover, not left ambiguous).
  const dxyFirstQ = doc.quarters.find(q => q.dxy_qret !== undefined && q.dxy_qret !== null);
  const dxyStartYear = dxyFirstQ ? dxyFirstQ.quarter.slice(0, 4) : null;

  _dsmileRenderSVG(chartEl, _GROWTHDIFF_SMILE_ORDER, stats, cur ? cur.regime : null);
  _dsmileRenderInsight(insightEl, doc, cur, stats);
  _growthdiffRenderTable(tbody, rawStats, cur, dxyStartYear);
}

// fmtOpts lets a caller override how point-label values are displayed
// without touching the curve's actual plotting math (yFor()/maxAbs still
// operate on the raw stats[r].avg value in its native unit regardless).
// Left generic (was previously shared with the now-removed Stress tab,
// which needed mult:100/unit:'bps' for its much smaller daily-return
// values) — only renderDollarSmile() calls this today, always with
// defaults, but the signature is harmless to keep general.
function _dsmileRenderSVG(el, regimes, stats, currentRegime, fmtOpts) {
  const fmt = Object.assign({ decimals: 2, mult: 1, unit: '%' }, fmtOpts || {});
  // Layout: fixed bands so the value labels can never collide with the
  // regime-name row, regardless of how extreme an average is. curveTop/
  // curveBottom bound where a point may ever be plotted; rowLabelY
  // (regime names) sits safely below that band with a fixed gap, and the
  // value label is placed at a fixed offset above its own point — both
  // quantities are independent of amplitude.
  const W = 620, H = 130, padL = 60, padR = 60;
  const curveTop = 22, curveBottom = 78, midY = (curveTop + curveBottom) / 2, ampY = (curveBottom - curveTop) / 2;
  const rowLabelY = 108, valueLabelGap = 12;
  const xs = regimes.map((_, i) => padL + i * ((W - padL - padR) / (regimes.length - 1)));

  // Only buckets clearing GROWTHDIFF_MIN_SAMPLES drive the curve's shape
  // and its scale. Real production data today clears this for all 3
  // buckets (n_dxy 15/31/32); this stays as a defensive floor in case a
  // future recompute ever narrows the historical window.
  const readyVals = regimes.map(r => stats[r].ready ? stats[r].avg : null).filter(v => v !== null);
  const maxAbs = readyVals.length ? Math.max(0.05, ...readyVals.map(Math.abs)) : 0.3;
  function yFor(r) {
    if (!stats[r].ready || stats[r].avg === null) return midY;
    const raw = midY - (stats[r].avg / maxAbs) * ampY;
    return Math.min(curveBottom, Math.max(curveTop, raw)); // hard clamp — belt and suspenders
  }

  const pts = regimes.map((r, i) => ({ x: xs[i], y: yFor(r), r, isCurrent: r === currentRegime }));
  const pathD = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');

  const up = _themeColor('--up'), text3 = _themeColor('--text3'),
        accent = _themeColor('--accent'), bg = _themeColor('--bg');

  const circles = pts.map(p => {
    const ready = stats[p.r].ready;
    const color = !ready ? text3 : (p.isCurrent ? up : text3);
    const r = p.isCurrent ? 4.5 : 3;
    const hollow = !ready ? ` fill="${bg}" stroke="${text3}" stroke-width="1.3"` : ` fill="${color}"${p.isCurrent ? ` stroke="${bg}" stroke-width="2"` : ''}`;
    return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r}"${hollow}></circle>`;
  }).join('');

  const labels = pts.map(p => {
    const ready = stats[p.r].ready;
    const dispVal = ready ? stats[p.r].avg * fmt.mult : null;
    const valTxt = ready ? `${dispVal >= 0 ? '+' : ''}${dispVal.toFixed(fmt.decimals)}${fmt.unit}` : `n=${stats[p.r].n}`;
    const valColor = !ready ? text3 : (p.isCurrent ? up : text3);
    const currentTag = p.isCurrent ? ' \u25cf latest' : '';
    const rowLabel = _GROWTHDIFF_LABELS[p.r] || p.r;
    // Fixed offset above the point, not above wherever the point landed —
    // p.y is already clamped to [curveTop, curveBottom], so this label
    // never gets closer than (curveTop - valueLabelGap) to the top edge
    // or crosses into the rowLabelY row below.
    const valueY = Math.max(12, p.y - valueLabelGap);
    return `
      <text x="${p.x.toFixed(1)}" y="${rowLabelY}" font-size="9.5" fill="${text3}" text-anchor="middle">${rowLabel}${currentTag}</text>
      <text x="${p.x.toFixed(1)}" y="${valueY.toFixed(1)}" font-size="9.5" fill="${valColor}" text-anchor="middle">${valTxt}</text>`;
  }).join('');

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:120px;display:block;">
      <line x1="${padL}" y1="${midY}" x2="${W - padR}" y2="${midY}" stroke="${_themeColorAlpha('--border', 0.9)}" stroke-width="1"></line>
      <path d="${pathD}" fill="none" stroke="${accent}" stroke-width="1.8"></path>
      ${circles}
      ${labels}
    </svg>`;
}

// One-line status + short hover tooltip (methodology in brief, then what
// it means for a trader) — v8.216.0 shortened from a 4-sentence paragraph
// per Santiago's feedback that it read too long for a tooltip; the fuller
// methodology disclosure lives in the static line above the chart
// (index-beta.html) and the panel-title tooltip, so this one only needs
// to orient a reader who hasn't seen those.
function _dsmileRenderInsight(el, doc, cur, stats) {
  if (!el) return;
  const regimeLabel = cur ? (_GROWTHDIFF_LABELS[cur.regime] || cur.regime) : '\u2014';
  const curTxt = cur
    ? `USD ${cur.usd_yoy >= 0 ? '+' : ''}${cur.usd_yoy.toFixed(1)}% vs G9 ${cur.g9_avg_yoy >= 0 ? '+' : ''}${cur.g9_avg_yoy.toFixed(1)}% YoY (${cur.diff >= 0 ? '+' : ''}${cur.diff.toFixed(2)}pp)`
    : 'no current reading';

  el.innerHTML = `${cur ? cur.quarter : '\u2014'}: ${curTxt} \u00b7 <span style="color:var(--up);">${regimeLabel}</span>`;

  el.title = `Regime = real GDP YoY differential (USD vs G9, FRED) with a VIX\u226540 crisis override \u2014 a genuine panic quarter is Global Risk-Off regardless of growth. ` +
    `Table below shows DXY's historical q/q return by regime, same quarter as the GDP reading (which lags ~1 quarter on release): context on dollar behavior by macro backdrop, not a trade signal.`;
}

function _growthdiffRenderTable(tbody, rawStats, cur, dxyStartYear) {
  if (!tbody) return;
  // n (GDP, full 1996- history) and n_dxy (subset with a matched same-
  // quarter DXY return, limited by dxy.json's own history) are genuinely
  // different denominators. Santiago flagged that showing both as bare
  // "n" in the same row reads as an inconsistency rather than two
  // disclosed sample sizes — so each gets its own coverage tag, per the
  // same "n designates a subsample, N the full population" convention
  // used in academic/clinical table reporting (JMIR stats guidelines).
  const dxyTag = dxyStartYear ? `, DXY ${dxyStartYear}\u2013` : '';
  tbody.innerHTML = _GROWTHDIFF_SMILE_ORDER.map(r => {
    const s = rawStats[r] || { n: 0, avg_dxy_qret: null, n_dxy: 0 };
    const isCurrent = cur && cur.regime === r;
    const avgTxt = s.avg_dxy_qret === null
      ? `\u2014 (n=${s.n_dxy || 0}${dxyTag})`
      : `${s.avg_dxy_qret >= 0 ? '+' : ''}${s.avg_dxy_qret.toFixed(2)}% (n=${s.n_dxy}${dxyTag})`;
    const rowStyle = isCurrent ? ` style="color:var(--up);"` : '';
    return `<tr${rowStyle}><td style="padding:2px 4px;">${_GROWTHDIFF_LABELS[r] || r}${isCurrent ? ' \u25cf' : ''}</td>` +
           `<td style="padding:2px 4px;">${s.n}</td>` +
           `<td style="padding:2px 4px;">${avgTxt}</td></tr>`;
  }).join('');
}

// (v8.205.0 added a Stress(VIX) tab and a Rate Diff placeholder tab
// alongside Growth here; both removed in v8.213.0 per Santiago's explicit
// instruction — this panel shows Jen's original growth-differential lens
// only, no tab chrome. renderDollarSmileStress()/_dsmileSwitchTab()/
// _dsmileStressRegimeFor()/_DSMILE_STRESS_REGIMES all deleted with it.)
