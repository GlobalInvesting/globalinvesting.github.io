// ═══════════════════════════════════════════════════════════════════════════
// ECONOMIC SURPRISES MODAL  v1.3.8
// File: assets/econ-surprises-modal.js
//
// Triggered by clicking any row in the Economic Surprises sidebar table.
// Mounts into #split-lower (right inline panel) via inline-panel.js intercept —
// same behaviour as COT / CB Rates / Yield Curve modals.
//
// DOM shape mirrors the other modals:
//   #esm-bd    (outer wrapper — position:static when transplanted)
//     #esm-modal (inner flex column — full width/height)
//
// Tabs: 10 major currencies (G10) with flag-icons (.fi.fi-xx) matching CB Rates convention.
// Chart: LightweightCharts v5 AreaSeries — rolling 30d surprise index, weekly.
// Table: Individual releases 90d window — beat/miss/in-line badge, H/M impact.
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('esm-css')) return;
  const s = document.createElement('style');
  s.id = 'esm-css';
  s.textContent = `
/* ── Desktop: inline panel layout (no fixed height on #esm-bd) ── */
#esm-bd { display:block!important; }
#esm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg,#131722)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
  display:flex;flex-direction:column;
}
#esm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 9px;border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg2,#1e222d);
}
#esm-hd-left { display:flex;align-items:center;gap:10px; }
#esm-title {
  font-size:14px;font-weight:600;color:var(--text,#d1d4dc);
  letter-spacing:-.01em;line-height:1.2;
  display:flex;align-items:center;gap:7px;
}
#esm-title .fi { font-size:15px;line-height:1;vertical-align:middle;border-radius:2px; }
#esm-sub {
  font-size:10px;color:var(--text2,#787b86);letter-spacing:.02em;margin-top:2px;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
#esm-close {
  background:none;border:none;color:var(--text3,#4e5c70);font-size:16px;
  cursor:pointer;padding:3px 6px;border-radius:3px;line-height:1;
  transition:color .1s,background .1s;
}
#esm-close:hover { color:var(--text,#d1d4dc);background:var(--bg3,#2a2e39); }
#esm-metrics {
  display:grid;grid-template-columns:repeat(5,1fr);
  border-bottom:1px solid var(--border,#252d3d);flex-shrink:0;
}
.esm-mm {
  padding:8px 12px;border-right:1px solid var(--border,#252d3d);
  display:flex;flex-direction:column;gap:1px;
}
.esm-mm:last-child { border-right:none; }
.esm-mm-lbl {
  font-size:9px;font-weight:600;color:var(--text2,#787b86);
  text-transform:uppercase;letter-spacing:.09em;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-mm-val {
  font-size:13px;font-weight:600;line-height:1;margin-top:3px;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-mm-sub {
  font-size:9px;color:var(--text2,#787b86);margin-top:1px;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
#esm-tabs {
  display:flex;padding:0 14px;border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;overflow-x:auto;scrollbar-width:none;background:var(--bg2,#1e222d);
}
#esm-tabs::-webkit-scrollbar { display:none; }
.esm-tab {
  font-size:11px;font-weight:500;padding:8px 12px;cursor:pointer;
  color:var(--text2,#787b86);border-bottom:2px solid transparent;
  transition:color .12s;white-space:nowrap;user-select:none;
  display:flex;align-items:center;gap:5px;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.esm-tab:hover { color:var(--text,#d1d4dc); }
.esm-tab.on { color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff); }

/* ── Desktop #esm-body: flex column, chart fixed height, events scroll internally ── */
#esm-body {
  display:flex;flex-direction:column;background:var(--bg,#131722);
  /* Desktop: contained scroll within the panel */
  flex:1;min-height:0;overflow:hidden;
}
#esm-chart-wrap {
  flex:0 0 220px;min-height:180px;max-height:240px;position:relative;
  border-bottom:1px solid var(--border,#252d3d);
}
#esm-chart-inner { position:absolute;inset:0; }
.esm-lw-tooltip {
  position:absolute;display:none;pointer-events:none;
  background:var(--bg2,#1e222d);border:1px solid var(--border2,#363c4e);
  border-radius:4px;padding:8px 10px;font-size:10px;line-height:1.6;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  color:var(--text,#d1d4dc);z-index:50;
  box-shadow:0 4px 16px rgba(0,0,0,.45);white-space:nowrap;
}
#esm-events-wrap {
  flex:1;min-height:0;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:var(--border2,#363c4e) transparent;
}
#esm-events-wrap::-webkit-scrollbar { width:3px; }
#esm-events-wrap::-webkit-scrollbar-thumb { background:var(--border2,#363c4e);border-radius:2px; }
#esm-events-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:6px 14px 5px;border-bottom:1px solid rgba(54,60,78,.5);
  position:sticky;top:0;background:var(--bg,#131722);z-index:2;
}
#esm-events-hd-title {
  font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
  color:var(--text2,#787b86);
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-evt-tbl {
  width:100%;border-collapse:collapse;font-size:10px;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-evt-tbl th {
  text-align:left;color:var(--text2,#787b86);font-weight:500;
  font-size:8px;text-transform:uppercase;letter-spacing:.08em;
  padding:5px 8px 4px;border-bottom:1px solid var(--border2,#363c4e);white-space:nowrap;
}
.esm-evt-tbl th:nth-child(3),
.esm-evt-tbl th:nth-child(4),
.esm-evt-tbl th:nth-child(5) { text-align:right; }
.esm-evt-tbl td {
  padding:5px 8px;border-bottom:1px solid rgba(54,60,78,.3);
  color:var(--text,#d1d4dc);vertical-align:middle;
}
.esm-evt-tbl td:nth-child(3),
.esm-evt-tbl td:nth-child(4),
.esm-evt-tbl td:nth-child(5) { text-align:right; }
.esm-evt-tbl tr:last-child td { border-bottom:none; }
.esm-evt-tbl tr:hover td { background:rgba(255,255,255,.025); }
.esm-beat { color:var(--up,#26a69a);font-weight:600; }
.esm-miss { color:var(--down,#ef5350);font-weight:600; }
.esm-badge-beat {
  display:inline-block;font-size:7px;font-weight:700;letter-spacing:.06em;
  padding:1px 4px;border-radius:2px;text-transform:uppercase;
  background:rgba(38,166,154,.15);color:var(--up,#26a69a);
}
.esm-badge-miss {
  display:inline-block;font-size:7px;font-weight:700;letter-spacing:.06em;
  padding:1px 4px;border-radius:2px;text-transform:uppercase;
  background:rgba(239,83,80,.15);color:var(--down,#ef5350);
}
.esm-badge-inline {
  display:inline-block;font-size:7px;font-weight:600;letter-spacing:.06em;
  padding:1px 4px;border-radius:2px;text-transform:uppercase;
  background:rgba(110,118,129,.12);color:var(--text3,#6b7280);
}
.esm-impact-h { color:var(--down,#ef5350); }
.esm-impact-m { color:var(--orange,#f6941c); }
.esm-no-data {
  padding:24px 16px;text-align:center;font-size:11px;color:var(--text3,#6b7280);
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}

/* ── Mobile (≤900px): fixed bottom-sheet overlay ──────────────────────────────
   Fixes Bug 1 (no scroll needed) and Bug 2 (layout has a defined height so
   flex:1;min-height:0 works and #esm-events-wrap scrolls internally).
   Uses 900px to match the terminal's own mobile breakpoint.
─────────────────────────────────────────────────────────────────────────────── */
@media(max-width:900px){
  #esm-bd {
    position:fixed!important;inset:0!important;
    background:rgba(0,0,0,.55)!important;
    z-index:9800!important;
    display:flex!important;align-items:flex-end!important;
    /* Tap backdrop to close */
    cursor:pointer;
  }
  #esm-modal {
    position:relative!important;
    width:100%!important;
    /* 80dvh gives enough room; fallback to 80vh */
    height:80vh!important;height:80dvh!important;
    max-height:80dvh!important;
    border-radius:12px 12px 0 0!important;
    box-shadow:0 -8px 40px rgba(0,0,0,.6)!important;
    animation:_esmSlideUp .22s cubic-bezier(.16,1,.3,1) both!important;
    /* Prevent backdrop tap from closing when tapping modal content */
    cursor:default;
    overflow:hidden!important;
  }
  @keyframes _esmSlideUp {
    from { transform:translateY(100%); opacity:.6; }
    to   { transform:translateY(0);    opacity:1;  }
  }
  /* With a fixed height on #esm-modal, flex:1 + min-height:0 work correctly */
  #esm-body {
    flex:1!important;min-height:0!important;overflow:hidden!important;
  }
  #esm-events-wrap {
    /* overflow-y:auto now works because the parent chain has a defined height */
    flex:1!important;min-height:0!important;overflow-y:auto!important;
    -webkit-overflow-scrolling:touch;
  }
  /* The sticky header is inside a scrollable container — no longer floats over chart */
  #esm-events-hd { position:sticky!important;top:0!important; }
  #esm-metrics { grid-template-columns:repeat(3,1fr); }
  .esm-mm:nth-child(3) { border-right:none; }
  .esm-mm:nth-child(4) { border-top:1px solid var(--border,#252d3d); }
  #esm-chart-wrap { flex:0 0 180px!important;min-height:160px!important;max-height:200px!important; }
}
`;
  document.head.appendChild(s);
})();

// ── Constants ────────────────────────────────────────────────────────────────
const _esmMonoF = "var(--font-mono,'JetBrains Mono','Courier New',monospace)";

const _ESM_CCY_META = {
  USD: { flag: 'us', label: 'US Dollar' },
  EUR: { flag: 'eu', label: 'Euro'      },
  GBP: { flag: 'gb', label: 'Pound'     },
  JPY: { flag: 'jp', label: 'Yen'       },
  AUD: { flag: 'au', label: 'Aussie'    },
  CAD: { flag: 'ca', label: 'Loonie'    },
  CHF: { flag: 'ch', label: 'Swissie'   },
  NZD: { flag: 'nz', label: 'Kiwi'     },
  NOK: { flag: 'no', label: 'Krone'    },
  SEK: { flag: 'se', label: 'Krona'    },
};

const _ESM_G8 = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','NOK','SEK'];

// v8.27.0: "trade balance" removed — Trade Balance is a SIGNED net level (deficit
// negative, surplus positive), same as Current Account which this list already
// correctly excludes. No inversion needed for a signed balance. Must stay in sync
// with INVERSE_KW in dashboard.js and INVERSE_EVENTS in fetch_economic_calendar.py.
const _ESM_INVERSE_KW = ['unemployment','jobless','claims','deficit'];

// _canonEsi — shared global defined in dashboard.js (loads before this file).
// _ESM_CCY_PFXS removed: _CCY_PFXS in dashboard.js is the single source of truth.

const _ESM_NOISE_KW = [
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

// ── LWC loader ───────────────────────────────────────────────────────────────
let _esmLwcPromise = null;
function _esmEnsureLWC() {
  if (window.LightweightCharts) return Promise.resolve();
  if (_esmLwcPromise) return _esmLwcPromise;
  _esmLwcPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
    s.onload  = resolve;
    s.onerror = () => { _esmLwcPromise = null; reject(new Error('LWC load failed')); };
    document.head.appendChild(s);
  });
  return _esmLwcPromise;
}

// ── State ────────────────────────────────────────────────────────────────────
let _esmChart     = null;
let _esmCalData   = null;
let _esmActiveCcy = 'USD';
let _esmRo           = null;   // active ResizeObserver — disconnected on destroy
let _esmTimers       = [];     // pending setTimeout IDs — cleared on destroy
let _esmResizeApply  = null;   // active window 'resize' handler — removed on destroy

// ── Time-decay constant (CESI convention, half-life 45d) ───────────────────────
// w = e^(-λ·ageDays), λ = ln(2)/45. Mirrors DECAY_LAMBDA in dashboard.js exactly.
const _ESM_DECAY_LAMBDA = Math.LN2 / 45;

// ── Sign-aware numeric parser — mirrors dashboard.js _scorePass exactly ────────
// parseFloat("$-226.8B") / parseFloat("CHF15.5B") / parseFloat("NZ$-1.01B") all
// return NaN because parseFloat cannot parse a leading currency-symbol prefix.
// Every Trade Balance / Current Account event for every G10 currency carries
// one of these prefixes, so the old parseFloat-based scoring here silently
// dropped them — same bug as dashboard.js _scorePass (fixed v8.25.8), just not
// yet ported to this file. Strip everything except digits/decimal, then
// restore the sign from the presence of '-' anywhere in the original string.
const _esmParseNum = s => {
  if (s == null || s === '') return NaN;
  const str = String(s).replace(/,/g, '');
  const neg = str.includes('-');
  const digits = str.replace(/[^\d.]/g, '');
  const n = parseFloat(digits);
  return isNaN(n) ? NaN : (neg ? -n : n);
};

// ── Score helpers ─────────────────────────────────────────────────────
// Decay-weighted scorer — mirrors dashboard.js renderEconSurprises() exactly.
// w = e^(-_ESM_DECAY_LAMBDA · ageDays), anchored to endMs (window right edge).
// For the 90d summary: endMs = now, w(0d)=1.0, w(45d)=0.5, w(90d)=0.25.
// For chart history: endMs = point date, keeping each slice consistent.
function _esmScoreWindow(events, ccy, startMs, endMs) {
  const seen = new Set();
  let total = 0, beats = 0, misses = 0;
  let wTotal = 0, wBeats = 0, wMisses = 0;
  let zWSum = 0, zWTotal = 0, zWBeats = 0, zWMisses = 0;
  const stats = window._ECON_SURPRISE_STATS || {};

  events.forEach(ev => {
    if (ev.currency !== ccy) return;
    const t = new Date(ev.dateISO).getTime();
    if (isNaN(t) || t > endMs || t < startMs) return;
    if (!ev.actual || ev.actual === '' || ev.actual === '-') return;
    if (!['medium','high'].includes(ev.impact)) return;

    const evTitle = (ev.event || ev.title || '').toLowerCase();
    if (_ESM_NOISE_KW.some(kw => evTitle.includes(kw))) return;

    const canon     = _canonEsi(evTitle);
    const actualS   = String(ev.actual   || '').replace(/[%,\s]/g, '');
    const forecastS = String(ev.forecast || ev.previous || '').replace(/[%,\s]/g, '');
    const key = `${ccy}/${canon}/${actualS}/${forecastS}`;
    if (seen.has(key)) return;
    seen.add(key);

    const actual   = _esmParseNum(ev.actual);
    const forecast = _esmParseNum(ev.forecast || ev.previous);
    if (isNaN(actual) || isNaN(forecast)) return;

    const isInverse = _ESM_INVERSE_KW.some(kw => evTitle.includes(kw));
    const beat = isInverse ? actual < forecast : actual > forecast;
    const miss = isInverse ? actual > forecast : actual < forecast;
    const surprise = isInverse ? -(actual - forecast) : (actual - forecast);

    // Decay weight × impact weight anchored to window right edge
    const ageDays    = (endMs - t) / 86400000;
    const impactMult = ev.impact === 'high' ? 1.0 : 0.5;
    const w          = Math.exp(-_ESM_DECAY_LAMBDA * ageDays) * impactMult;

    const st = stats[`${ccy}/${canon}`];
    const useZ = st && st.n >= 5 && st.std > 0;
    const zScore = useZ ? (surprise - st.mean) / st.std : null;

    total++;
    wTotal += w;
    if (beat) { beats++;  wBeats  += w; }
    if (miss) { misses++; wMisses += w; }
    if (zScore !== null) {
      zWSum   += zScore * w;
      zWTotal += w;
      if (beat) zWBeats += w;
      if (miss) zWMisses += w;
    }
  });

  if (!total) return null;

  // Identical formula to dashboard.js
  let idx100;
  const zFrac = zWTotal / wTotal;
  if (zWTotal >= 10 || (zWTotal > 0 && zFrac >= 0.30)) {
    const nonZW     = wTotal  - zWTotal;
    const nonZWBeat = wBeats  - zWBeats;
    const nonZWMiss = wMisses - zWMisses;
    const zPart  = zWTotal > 0 ? (zWSum / zWTotal) * 50 : 0;
    const bmPart = nonZW   > 0 ? ((nonZWBeat - nonZWMiss) / nonZW) * 100 : 0;
    idx100 = (zPart * zWTotal + bmPart * nonZW) / wTotal;
  } else {
    idx100 = wTotal > 0 ? ((wBeats - wMisses) / wTotal) * 100 : 0;
  }

  return { idx: idx100, beats, misses, total };
}
// Builds the rolling time-series for the chart.
// CESI convention: 90d rolling window, weekly step.
// Each point is the decay-weighted index over [pointDate-90d, pointDate].
// Decay anchor = pointDate (endMs), so the curve is consistent with the
// current-period score which anchors to now.
function _esmBuildSeries(events, ccy) {
  const nowMs     = Date.now();
  const WINDOW_MS = 90 * 24 * 60 * 60 * 1000; // 90d — CESI standard
  const STEP_MS   =  7 * 24 * 60 * 60 * 1000; // weekly step

  const ccyEvts = events.filter(ev =>
    ev.currency === ccy && ev.actual && ev.actual !== '' && ev.actual !== '-'
  );
  if (!ccyEvts.length) return [];

  const minDate = Math.min(...ccyEvts.map(ev => new Date(ev.dateISO).getTime()).filter(t => !isNaN(t)));
  // Map, not array — keyed by calendar-day ISO string. The last weekly cursor
  // before the loop exits is clamped to nowMs (`Math.min(cursor, nowMs)`), and
  // whenever that clamp lands on the same calendar day as the preceding
  // regular weekly step (happens whenever minDate's 7-day grid aligns with
  // "today" — roughly 1-in-7 currencies on any given day), both iterations
  // format to the identical ISO date string. LightweightCharts' setData()
  // requires strictly ascending, unique times and throws on a duplicate —
  // an uncaught throw here (observed on GBP/CHF/NZD/SEK) silently aborted the
  // rest of _esmRenderChart, leaving an empty chart with axes but no series.
  // Map.set on an existing key updates the value in place without moving its
  // position, so chronological order is preserved and the later (more
  // current, nowMs-anchored) score simply wins for that day.
  const byDate  = new Map();
  let cursor    = minDate + WINDOW_MS;

  while (cursor <= nowMs + STEP_MS) {
    const endMs   = Math.min(cursor, nowMs);
    const startMs = endMs - WINDOW_MS;
    const r       = _esmScoreWindow(events, ccy, startMs, endMs);
    if (r !== null) {
      const dt  = new Date(endMs);
      const iso = `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`;
      byDate.set(iso, parseFloat(r.idx.toFixed(2)));
    }
    cursor += STEP_MS;
  }
  return Array.from(byDate, ([time, value]) => ({ time, value }));
}
function _esmCurrentScore(events, ccy) {
  const nowMs  = Date.now();
  const W90_MS  = 90  * 24 * 60 * 60 * 1000;
  const W180_MS = 180 * 24 * 60 * 60 * 1000;
  // Pass 0: standard 90d window — mirrors dashboard.js renderEconSurprises() pass 0.
  const r0 = _esmScoreWindow(events, ccy, nowMs - W90_MS, nowMs);
  if (r0 !== null) return { ...r0, widened: false };
  // Pass 1: 90–180d extension band — only reached when no 90d data exists.
  // Applies the same impact filter and decay function as pass 0; widening the
  // window does NOT lower methodology standards (identical to EA v8.4.3+).
  const r1 = _esmScoreWindow(events, ccy, nowMs - W180_MS, nowMs - W90_MS);
  return r1 ? { ...r1, widened: true } : null;
}

function _esmGetEvents(events, ccy) {
  const nowMs  = Date.now();
  const W90_MS  = 90  * 24 * 60 * 60 * 1000;
  const W180_MS = 180 * 24 * 60 * 60 * 1000;

  // Determine effective window: check if any medium/high events exist in 90d.
  // Matches the adaptive window logic in _esmCurrentScore / dashboard.js pass 0/1.
  const has90d = events.some(ev => {
    if (ev.currency !== ccy) return false;
    const t = new Date(ev.dateISO).getTime();
    if (isNaN(t) || t > nowMs || nowMs - t > W90_MS) return false;
    return !!(ev.actual && ev.actual !== '' && ev.actual !== '-') &&
           ['medium','high'].includes(ev.impact);
  });
  const windowMs = has90d ? W90_MS : W180_MS;

  const seen   = new Set();
  const result = [];

  events.forEach(ev => {
    if (ev.currency !== ccy) return;
    const t = new Date(ev.dateISO).getTime();
    if (isNaN(t) || t > nowMs || nowMs - t > windowMs) return;
    if (!ev.actual || ev.actual === '' || ev.actual === '-') return;
    if (!['medium','high'].includes(ev.impact)) return;

    const evTitle   = (ev.event || ev.title || '').toLowerCase();
    if (_ESM_NOISE_KW.some(kw => evTitle.includes(kw))) return;

    const canon     = _canonEsi(evTitle);
    const actualS   = String(ev.actual   || '').replace(/[%,\s]/g, '');
    const forecastS = String(ev.forecast || ev.previous || '').replace(/[%,\s]/g, '');
    const key = `${ccy}/${canon}/${actualS}/${forecastS}`;
    if (seen.has(key)) return;
    seen.add(key);

    const actual   = _esmParseNum(ev.actual);
    const forecast = _esmParseNum(ev.forecast || ev.previous);
    const hasFc    = !isNaN(actual) && !isNaN(forecast);

    const isInverse = _ESM_INVERSE_KW.some(kw => evTitle.includes(kw));
    let outcome = 'n/a';
    if (hasFc) {
      const beat = isInverse ? actual < forecast : actual > forecast;
      const miss = isInverse ? actual > forecast : actual < forecast;
      outcome = beat ? 'beat' : miss ? 'miss' : 'inline';
    }

    result.push({
      dateISO: ev.dateISO,
      event:   ev.event || ev.title || '',
      impact:  ev.impact,
      actual:  ev.actual   || '—',
      forecast:ev.forecast || ev.previous || '—',
      outcome,
    });
  });

  result.sort((a, b) => b.dateISO.localeCompare(a.dateISO));
  return { rows: result, widened: !has90d };
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function _esmDestroyChart() {
  // Cancel pending resize timers before removing the chart — prevents
  // "Object is disposed" errors from ResizeObserver / setTimeout callbacks
  // that fire applyOptions() on an already-removed LWC instance.
  _esmTimers.forEach(id => clearTimeout(id));
  _esmTimers = [];
  if (_esmRo) { try { _esmRo.disconnect(); } catch (_) {} _esmRo = null; }
  if (_esmResizeApply) { window.removeEventListener('resize', _esmResizeApply); _esmResizeApply = null; }
  if (_esmChart) { try { _esmChart.remove(); } catch (_) {} _esmChart = null; }
}

function _esmRenderChart(ccy) {
  const LWC = window.LightweightCharts;
  if (!LWC || !_esmCalData) return;

  // Read dimensions BEFORE destroy — _esmChart.remove() causes LWC to clear the
  // container innerHTML which zeros offsetHeight in the same frame.
  // getBoundingClientRect() is more reliable (mirrors the COT modal fix).
  const container = document.getElementById('esm-chart-inner');
  if (!container) return;
  const _rect = container.getBoundingClientRect();
  const W = Math.round(_rect.width)  || container.offsetWidth  || 600;
  const H = Math.round(_rect.height) || container.offsetHeight
          || container.parentElement?.getBoundingClientRect().height || 220;

  _esmDestroyChart();

  const series = _esmBuildSeries(_esmCalData.events || [], ccy);

  const _esmBg    = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()    || '#131722';
  const _esmText2 = getComputedStyle(document.documentElement).getPropertyValue('--text2').trim() || '#9096a0';
  const _esmBrd   = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#2a2e39';
  const _esmBrd2  = getComputedStyle(document.documentElement).getPropertyValue('--bg3').trim()    || '#2a2e39';
  const chart = LWC.createChart(container, {
    width: W, height: H,
    layout: {
      background: { type: 'solid', color: _esmBg },
      textColor: _esmText2, fontFamily: _esmMonoF, fontSize: 10, attributionLogo: false,
    },
    grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.04)' } },
    crosshair: {
      mode: LWC.CrosshairMode?.Normal ?? 1,
      vertLine: { color: 'rgba(255,255,255,0.2)', style: 2, labelVisible: false, labelBackgroundColor: _esmBrd2 },
      horzLine: { color: 'rgba(255,255,255,0.15)', style: 2, labelVisible: true, labelBackgroundColor: _esmBrd2 },
    },
    rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.12 } },
    timeScale: { borderVisible: false, fixRightEdge: true },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale:  { mouseWheel: true, pinch: true },
    localization: { priceFormatter: v => v != null ? (v >= 0 ? '+' : '') + v.toFixed(1) : '—' },
  });
  _esmChart = chart;

  // Zero line
  const zeroLine = chart.addSeries(LWC.LineSeries, {
    color: 'rgba(110,118,129,0.4)', lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });
  if (series.length >= 2) {
    zeroLine.setData([
      { time: series[0].time, value: 0 },
      { time: series[series.length - 1].time, value: 0 },
    ]);
  }

  const lastVal   = series.length > 0 ? series[series.length - 1].value : 0;
  const pos       = lastVal >= 0;
  const lineColor = pos ? '#26a69a' : '#ef5350';
  const topColor  = pos ? 'rgba(38,166,154,0.28)' : 'rgba(239,83,80,0.28)';
  const botColor  = pos ? 'rgba(38,166,154,0.02)' : 'rgba(239,83,80,0.02)';

  const areaSeries = chart.addSeries(LWC.AreaSeries, {
    lineColor, topColor, bottomColor: botColor, lineWidth: 2,
    priceLineVisible: false, lastValueVisible: true,
    crosshairMarkerVisible: true, crosshairMarkerRadius: 4,
  });

  if (series.length > 0) { areaSeries.setData(series); chart.timeScale().fitContent(); }

  // Tooltip
  const tip = document.createElement('div');
  tip.className = 'esm-lw-tooltip';
  container.style.position = 'relative';
  container.appendChild(tip);

  const TW = 180, TM = 14;
  chart.subscribeCrosshairMove(param => {
    if (!param?.point || !param.seriesData) { tip.style.display = 'none'; return; }
    const d = param.seriesData.get(areaSeries);
    if (!d) { tip.style.display = 'none'; return; }

    const val = d.value, col = val >= 0 ? '#26a69a' : '#ef5350';
    const dateStr = typeof param.time === 'string' ? param.time : '';
    const endMs   = new Date(dateStr).getTime();
    const winSc   = _esmScoreWindow(_esmCalData.events || [], ccy, endMs - 90*24*60*60*1000, endMs);
    const nTxt    = winSc ? `${winSc.total} events · ${winSc.beats}B / ${winSc.misses}M` : '';

    tip.innerHTML = `
      <div style="font-size:9px;color:var(--text2,#787b86);margin-bottom:4px;">${dateStr} · 90d window · decay-weighted</div>
      <div style="font-size:13px;font-weight:700;color:${col}">${val >= 0 ? '+' : ''}${val.toFixed(1)}</div>
      ${nTxt ? `<div style="font-size:9px;color:var(--text3,#6b7280);margin-top:3px;">${nTxt}</div>` : ''}
    `;
    tip.style.display = 'block';

    const cW = container.offsetWidth;
    const cx = param.point.x, cy = param.point.y;
    const th = tip.offsetHeight || 60;
    const tx = (cx + TM + TW <= cW - 4) ? cx + TM : cx - TM - TW;
    const ty = (cy - th - TM >= 4) ? cy - th - TM : cy + TM;
    tip.style.left = Math.max(0, tx) + 'px';
    tip.style.top  = Math.max(0, ty) + 'px';
  });

  // Resize — store observer and timer IDs so _esmDestroyChart() can cancel them.
  // IMPORTANT: store `apply` in _esmResizeApply so the window 'resize' listener
  // can be removed on destroy. Without this, each tab switch leaks a listener
  // that holds a stale closure over the old `container` ref — `container.offsetHeight`
  // returns 0 after LWC clears the DOM, causing chart.applyOptions({ height: 0 })
  // which collapses the time-axis row and hides the date labels permanently.
  const apply = () => {
    requestAnimationFrame(() => {
      if (!_esmChart) return;   // guard: chart may have been destroyed before rAF fires
      const r = container.getBoundingClientRect();
      const w = Math.round(r.width)  || container.offsetWidth  || 600;
      const h = Math.round(r.height) || container.offsetHeight
              || container.parentElement?.getBoundingClientRect().height || 220;
      if (w > 0 && h > 10) { try { chart.applyOptions({ width: w, height: h }); } catch (_) {} }
    });
  };
  _esmResizeApply = apply;
  if (window.ResizeObserver) {
    _esmRo = new ResizeObserver(apply);
    _esmRo.observe(container);
  }
  window.addEventListener('resize', apply);
  _esmTimers = [
    setTimeout(apply, 60),
    setTimeout(apply, 250),
    setTimeout(apply, 600),
  ];
}

// ── Metrics ───────────────────────────────────────────────────────────────────
function _esmRenderMetrics(ccy) {
  if (!_esmCalData) return;
  const s = _esmCurrentScore(_esmCalData.events || [], ccy);
  const el = id => document.getElementById(id);

  if (!s) {
    ['esm-m-index','esm-m-beats','esm-m-misses','esm-m-n','esm-m-beat-rt'].forEach(id => {
      const e = el(id); if (e) { e.textContent = '—'; e.style.color = 'var(--text2)'; }
    });
    return;
  }

  const idx    = s.idx;
  const idxCol = idx > 5 ? 'var(--up,#26a69a)' : idx < -5 ? 'var(--down,#ef5350)' : 'var(--text,#d1d4dc)';
  const beatRt = s.total > 0 ? (s.beats / s.total * 100).toFixed(0) + '%' : '—';

  // Update the "N (90d)" metric label — when widened, change to "N (90d/180d)" to
  // match EA's subtitle "Citi methodology * 90D/180D" convention (g_esi_window_days).
  const nLbl = el('esm-m-n')?.closest('.esm-mm')?.querySelector('.esm-mm-lbl');
  if (nLbl) nLbl.textContent = s.widened ? 'N (90d/180d)' : 'N (90d)';

  const mi = el('esm-m-index');
  if (mi) { mi.textContent = (idx >= 0 ? '+' : '') + idx.toFixed(1); mi.style.color = idxCol; }
  const mb = el('esm-m-beats');  if (mb) { mb.textContent = s.beats;  mb.style.color = 'var(--up,#26a69a)'; }
  const mm = el('esm-m-misses'); if (mm) { mm.textContent = s.misses; mm.style.color = 'var(--down,#ef5350)'; }
  const mn = el('esm-m-n');      if (mn) { mn.textContent = s.total;  mn.style.color = 'var(--text)'; }
  const mr = el('esm-m-beat-rt');
  if (mr) {
    mr.textContent = beatRt;
    mr.style.color = parseFloat(beatRt) >= 50 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
  }

  // Update the Index metric label and sub-header when widened.
  const indexLbl = el('esm-m-index')?.closest('.esm-mm')?.querySelector('.esm-mm-lbl');
  if (indexLbl) indexLbl.textContent = s.widened ? 'Index (90d/180d)' : 'Index (90d)';

  // Update the header sub-line to reflect the adaptive window when in use.
  const sub = el('esm-sub');
  if (sub) {
    sub.textContent = s.widened
      ? 'Actual vs consensus · G10 major currencies · 90d/180d adaptive rolling · decay-weighted (45d ½life) · [−100, +100]'
      : 'Actual vs consensus · G10 major currencies · 90d rolling · decay-weighted (45d ½life) · [−100, +100]';
  }
}

// ── Events table ──────────────────────────────────────────────────────────────
function _esmRenderTable(ccy) {
  const tbody = document.getElementById('esm-evt-tbody');
  if (!tbody) return;
  if (!_esmCalData) { tbody.innerHTML = '<tr><td colspan="5" class="esm-no-data">Loading…</td></tr>'; return; }

  const { rows, widened } = _esmGetEvents(_esmCalData.events || [], ccy);

  // Update the "Events · 90d rolling window" header label when widened.
  const evHdTitle = document.getElementById('esm-events-hd-title');
  if (evHdTitle) evHdTitle.textContent = widened ? `Events · 90d/180d adaptive window` : `Events · 90d rolling window`;

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="esm-no-data">No released events with actuals in the 90d window for ${ccy}.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const badgeCls = r.outcome === 'beat' ? 'esm-badge-beat' : r.outcome === 'miss' ? 'esm-badge-miss' : 'esm-badge-inline';
    const badgeTxt = r.outcome === 'beat' ? 'BEAT' : r.outcome === 'miss' ? 'MISS' : r.outcome === 'inline' ? 'IN LINE' : '—';
    const impCls   = r.impact === 'high' ? 'esm-impact-h' : 'esm-impact-m';
    const actCls   = r.outcome === 'beat' ? 'esm-beat' : r.outcome === 'miss' ? 'esm-miss' : '';
    return `<tr>
      <td style="color:var(--text2,#787b86)">${r.dateISO.slice(5).replace('-','/')}</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.event}">
        <span class="${impCls}" style="font-size:8px;margin-right:4px;">${r.impact === 'high' ? 'H' : 'M'}</span>${r.event}
      </td>
      <td class="${actCls}">${r.actual}</td>
      <td style="color:var(--text2,#787b86)">${r.forecast}</td>
      <td><span class="${badgeCls}">${badgeTxt}</span></td>
    </tr>`;
  }).join('');
}

// ── Tab switch ────────────────────────────────────────────────────────────────
function esmTab(el, ccy) {
  document.querySelectorAll('.esm-tab').forEach(t => {
    t.classList.remove('on');
    t.setAttribute('aria-selected','false');
    t.setAttribute('tabindex','-1');
  });
  el.classList.add('on');
  el.setAttribute('aria-selected','true');
  el.setAttribute('tabindex','0');
  _esmActiveCcy = ccy;

  // Update title flag
  const titleEl = document.getElementById('esm-title');
  if (titleEl) {
    const meta = _ESM_CCY_META[ccy] || {};
    const flagHtml = meta.flag ? `<span class="fi fi-${meta.flag}"></span>` : '';
    titleEl.innerHTML = `${flagHtml}Economic Surprises &middot; ${ccy}`;
  }

  _esmRenderMetrics(ccy);
  _esmRenderChart(ccy);
  _esmRenderTable(ccy);
}

// ── Close ─────────────────────────────────────────────────────────────────────
function closeESModal() {
  _esmDestroyChart();
  const bd = document.getElementById('esm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _esmKeydown);
}

function _esmKeydown(e) { if (e.key === 'Escape') closeESModal(); }

// ── Open ──────────────────────────────────────────────────────────────────────
async function openEconSurprisesModal(initialCcy) {
  closeESModal();
  const ccy  = _ESM_G8.includes(initialCcy) ? initialCcy : 'USD';
  _esmActiveCcy = ccy;

  const initMeta = _ESM_CCY_META[ccy] || {};
  const initFlag = initMeta.flag ? `<span class="fi fi-${initMeta.flag}"></span>` : '';

  // outer #esm-bd → inner #esm-modal  (matches _transplant pattern)
  const bd = document.createElement('div');
  bd.id = 'esm-bd';

  bd.innerHTML = `
<div id="esm-modal" role="dialog" aria-modal="true" aria-label="Economic Surprises — ${ccy}">
  <div id="esm-hd">
    <div id="esm-hd-left">
      <div>
        <div id="esm-title">${initFlag}Economic Surprises &middot; ${ccy}</div>
        <div id="esm-sub">Actual vs consensus &middot; G10 major currencies &middot; 90d rolling &middot; decay-weighted (45d ½life) &middot; [&minus;100, +100]</div>
      </div>
    </div>
    <button id="esm-close" onclick="closeESModal()" aria-label="Close">&times;</button>
  </div>

  <div id="esm-metrics">
    <div class="esm-mm">
      <div class="esm-mm-lbl">Index (90d)</div>
      <div class="esm-mm-val" id="esm-m-index" style="color:var(--text2)">—</div>
      <div class="esm-mm-sub">[−100, +100]</div>
    </div>
    <div class="esm-mm">
      <div class="esm-mm-lbl">Beats</div>
      <div class="esm-mm-val" id="esm-m-beats">—</div>
      <div class="esm-mm-sub">events</div>
    </div>
    <div class="esm-mm">
      <div class="esm-mm-lbl">Misses</div>
      <div class="esm-mm-val" id="esm-m-misses">—</div>
      <div class="esm-mm-sub">events</div>
    </div>
    <div class="esm-mm">
      <div class="esm-mm-lbl">N (90d)</div>
      <div class="esm-mm-val" id="esm-m-n">—</div>
      <div class="esm-mm-sub">scored</div>
    </div>
    <div class="esm-mm">
      <div class="esm-mm-lbl">Beat Rate</div>
      <div class="esm-mm-val" id="esm-m-beat-rt">—</div>
      <div class="esm-mm-sub">of total</div>
    </div>
  </div>

  <div id="esm-tabs" role="tablist" aria-label="major currency tabs">
    ${_ESM_G8.map(c => {
      return `<div class="esm-tab${c === ccy ? ' on' : ''}" data-ccy="${c}" role="tab"
        aria-selected="${c === ccy ? 'true' : 'false'}"
        onclick="esmTab(this,'${c}')"
        tabindex="${c === ccy ? '0' : '-1'}">${c}</div>`;
    }).join('')}
  </div>

  <div id="esm-body">
    <div id="esm-chart-wrap">
      <div id="esm-chart-inner"></div>
    </div>
    <div id="esm-events-wrap">
      <div id="esm-events-hd">
        <div id="esm-events-hd-title">Events · 90d rolling window</div>
        <div style="font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);">H&nbsp;=&nbsp;high&nbsp;·&nbsp;M&nbsp;=&nbsp;medium</div>
      </div>
      <table class="esm-evt-tbl" aria-label="Economic event releases">
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Event</th>
            <th scope="col">Actual</th>
            <th scope="col">Forecast</th>
            <th scope="col">Outcome</th>
          </tr>
        </thead>
        <tbody id="esm-evt-tbody">
          <tr><td colspan="5" class="esm-no-data">Loading data…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>`;

  document.body.appendChild(bd);
  document.addEventListener('keydown', _esmKeydown);

  // Mobile: tap the dark backdrop (not the modal itself) to close
  bd.addEventListener('click', function(e) {
    if (e.target === bd) closeESModal();
  });

  // Fetch calendar
  try {
    const res = await fetch('./calendar-data/calendar.json').catch(() => null);
    if (res?.ok) {
      const calj = await res.json();
      _esmCalData = calj;
      if (calj.surpriseStats) window._ECON_SURPRISE_STATS = calj.surpriseStats;
    }
  } catch (_) {}

  try { await _esmEnsureLWC(); } catch (_) {}

  _esmRenderMetrics(ccy);
  _esmRenderChart(ccy);
  _esmRenderTable(ccy);
}

window.openEconSurprisesModal = openEconSurprisesModal;
window.closeESModal           = closeESModal;
window.esmTab                 = esmTab;
