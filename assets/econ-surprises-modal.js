// ═══════════════════════════════════════════════════════════════════════════
// ECONOMIC SURPRISES MODAL  v1.0.0
// File: assets/econ-surprises-modal.js
//
// Triggered by clicking any row in the Economic Surprises sidebar table.
// Displays a fullscreen panel (same pattern as COT / CB Rates modals) with:
//   Tab per G8 currency (USD · EUR · GBP · JPY · AUD · CAD · CHF · NZD)
//   LightweightCharts v5 area line — rolling 30d surprise index (weekly step)
//   Zero reference line at 0 (centre, CESI convention)
//   Events table — individual releases with beat / miss / in-line markers
//
// Data: calendar-data/calendar.json (same source as renderEconSurprises()).
// Index methodology: mirrors renderEconSurprises() — beat/miss beat rate scaled
//   to [−100, +100], 30d rolling window stepped weekly.
// ═══════════════════════════════════════════════════════════════════════════

// ── CSS (injected once) ──────────────────────────────────────────────────────
(function () {
  if (document.getElementById('esm-css')) return;
  const s = document.createElement('style');
  s.id = 'esm-css';
  s.textContent = `
#esm-bd {
  position:fixed!important;inset:0!important;z-index:9400;
  display:flex!important;flex-direction:column;overflow:hidden;
  background:var(--bg,#131722);
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}
#esm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 9px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg2,#1e222d);
}
#esm-hd-left { display:flex;align-items:center;gap:10px; }
#esm-badge {
  font-size:8px;font-weight:700;letter-spacing:.10em;
  color:var(--blue,#4f7fff);text-transform:uppercase;
  display:flex;align-items:center;gap:4px;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
#esm-badge::before {
  content:'';width:5px;height:5px;border-radius:50%;
  background:var(--blue,#4f7fff);flex-shrink:0;
}
#esm-title {
  font-size:14px;font-weight:600;color:var(--text,#d1d4dc);
  letter-spacing:-.01em;line-height:1.2;
}
#esm-sub {
  font-size:10px;color:var(--text2,#787b86);
  letter-spacing:.02em;margin-top:2px;
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
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg,#131722);
}
.esm-mm {
  padding:8px 12px;border-right:1px solid var(--border,#252d3d);
  display:flex;flex-direction:column;gap:1px;
}
.esm-mm:last-child { border-right:none; }
.esm-mm-lbl {
  font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  font-weight:600;color:var(--text2,#787b86);text-transform:uppercase;letter-spacing:.09em;
}
.esm-mm-val {
  font-size:13px;font-weight:600;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  line-height:1;margin-top:3px;
}
.esm-mm-sub {
  font-size:9px;color:var(--text2,#787b86);
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  margin-top:1px;
}
#esm-tabs {
  display:flex;padding:0 14px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;overflow-x:auto;scrollbar-width:none;
  background:var(--bg2,#1e222d);
  role:tablist;
}
#esm-tabs::-webkit-scrollbar { display:none; }
.esm-tab {
  font-size:11px;font-weight:500;padding:9px 13px;cursor:pointer;
  color:var(--text2,#787b86);border-bottom:2px solid transparent;
  transition:color .12s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.esm-tab:hover { color:var(--text,#d1d4dc); }
.esm-tab.on { color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff); }
#esm-body {
  flex:1;min-height:0;overflow:hidden;
  display:flex;flex-direction:column;
  background:var(--bg,#131722);
}
#esm-chart-wrap {
  flex:0 0 45%;min-height:200px;max-height:340px;
  position:relative;
  border-bottom:1px solid var(--border,#252d3d);
}
#esm-chart-inner {
  position:absolute;inset:0;
}
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
  padding:7px 14px 5px;
  border-bottom:1px solid rgba(54,60,78,.5);
  position:sticky;top:0;background:var(--bg,#131722);z-index:2;
}
#esm-events-hd-title {
  font-size:9px;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;
  color:var(--text2,#787b86);
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-evt-tbl {
  width:100%;border-collapse:collapse;
  font-size:10px;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.esm-evt-tbl th {
  text-align:left;color:var(--text2,#787b86);font-weight:500;
  font-size:8px;text-transform:uppercase;letter-spacing:.08em;
  padding:5px 8px 4px;border-bottom:1px solid var(--border2,#363c4e);
  white-space:nowrap;
}
.esm-evt-tbl th:nth-child(3),
.esm-evt-tbl th:nth-child(4),
.esm-evt-tbl th:nth-child(5) { text-align:right; }
.esm-evt-tbl td {
  padding:5px 8px;
  border-bottom:1px solid rgba(54,60,78,.3);
  color:var(--text,#d1d4dc);vertical-align:middle;
}
.esm-evt-tbl td:nth-child(3),
.esm-evt-tbl td:nth-child(4),
.esm-evt-tbl td:nth-child(5) { text-align:right; }
.esm-evt-tbl tr:last-child td { border-bottom:none; }
.esm-evt-tbl tr:hover td { background:rgba(255,255,255,.025); }
.esm-beat { color:var(--up,#26a69a);font-weight:600; }
.esm-miss { color:var(--down,#ef5350);font-weight:600; }
.esm-inline { color:var(--text3,#6b7280); }
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
  padding:24px 16px;text-align:center;
  font-size:11px;color:var(--text3,#6b7280);
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
@media(max-width:600px){
  #esm-metrics { grid-template-columns:repeat(3,1fr); }
  .esm-mm:nth-child(3) { border-right:none; }
  .esm-mm:nth-child(4) { border-top:1px solid var(--border,#252d3d); }
  #esm-chart-wrap { flex:0 0 40%;max-height:260px; }
}
`;
  document.head.appendChild(s);
})();

// ── Constants ────────────────────────────────────────────────────────────────
const _esmMonoF = "var(--font-mono,'JetBrains Mono','Courier New',monospace)";

const _ESM_G8 = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];

const _ESM_INVERSE_KW = ['unemployment','jobless','claims','deficit','trade balance'];

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

const _ESM_FLAG = {
  USD:'US', EUR:'EU', GBP:'GB', JPY:'JP',
  AUD:'AU', CAD:'CA', CHF:'CH', NZD:'NZ',
};

// ── LWC library loader (idempotent) ──────────────────────────────────────────
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
let _esmChart      = null;
let _esmCalData    = null;  // cached calendar.json payload
let _esmActiveCcy  = 'USD';

// ── Score computation helpers ────────────────────────────────────────────────
// Mirrors renderEconSurprises() exactly.

function _esmScoreWindow(events, ccy, startMs, endMs) {
  const seen   = new Set();
  let beats = 0, misses = 0, total = 0, zSum = 0, zN = 0, zBeats = 0, zMisses = 0;
  const stats = (window._ECON_SURPRISE_STATS || {});

  events.forEach(ev => {
    if (ev.currency !== ccy) return;
    const t = new Date(ev.dateISO).getTime();
    if (isNaN(t) || t > endMs || t < startMs) return;
    if (!ev.actual || ev.actual === '' || ev.actual === '-') return;
    if (!['medium','high'].includes(ev.impact)) return;

    const evTitle = (ev.event || ev.title || '').toLowerCase();
    if (_ESM_NOISE_KW.some(kw => evTitle.includes(kw))) return;

    const canon    = evTitle.replace(/\s*\([^)]*\)/g, '').trim();
    const actualS  = String(ev.actual   || '').replace(/[%,\s]/g, '');
    const forecastS= String(ev.forecast || ev.previous || '').replace(/[%,\s]/g, '');
    const dedupKey = `${ccy}/${canon}/${actualS}/${forecastS}`;
    if (seen.has(dedupKey)) return;
    seen.add(dedupKey);

    const actual   = parseFloat(String(ev.actual   || '').replace(/[%,]/g, ''));
    const forecast = parseFloat(String(ev.forecast || ev.previous || '').replace(/[%,]/g, ''));
    if (isNaN(actual) || isNaN(forecast)) return;

    const isInverse = _ESM_INVERSE_KW.some(kw => evTitle.includes(kw));
    const beat = isInverse ? actual < forecast : actual > forecast;
    const miss = isInverse ? actual > forecast : actual < forecast;
    const rawSurprise = actual - forecast;
    const surprise    = isInverse ? -rawSurprise : rawSurprise;

    const statsKey = `${ccy}/${canon}`;
    const st = stats[statsKey];
    const useZ = st && st.n >= 5 && st.std > 0;
    const zScore = useZ ? (surprise - st.mean) / st.std : null;

    total++;
    if (beat) beats++;
    if (miss) misses++;
    if (zScore !== null) {
      zSum += zScore; zN++;
      if (beat) zBeats++;
      if (miss) zMisses++;
    }
  });

  if (total === 0) return null;

  let idx100;
  const zFrac = zN / total;
  if (zN >= 10 || (zN > 0 && zFrac >= 0.30)) {
    const nonZN    = total - zN;
    const nonZBeat = beats  - zBeats;
    const nonZMiss = misses - zMisses;
    const zPart    = zN  > 0 ? (zSum / zN) * 50 : 0;
    const bmPart   = nonZN > 0 ? ((nonZBeat - nonZMiss) / nonZN) * 100 : 0;
    idx100 = (zPart * zN + bmPart * nonZN) / total;
  } else {
    idx100 = ((beats - misses) / total) * 100;
  }

  return { idx: idx100, beats, misses, total };
}

// ── Rolling time-series: weekly steps, 30d window ────────────────────────────
// Returns [{date: "YYYY-MM-DD", value: number}] for use in LWC area series.
function _esmBuildSeries(events, ccy) {
  const nowMs      = Date.now();
  const WINDOW_MS  = 30 * 24 * 60 * 60 * 1000;
  const STEP_MS    = 7  * 24 * 60 * 60 * 1000;

  // Find earliest released event for this currency
  const ccyEvts = events.filter(ev =>
    ev.currency === ccy &&
    (ev.actual && ev.actual !== '' && ev.actual !== '-')
  );
  if (!ccyEvts.length) return [];

  const minDate = Math.min(...ccyEvts.map(ev => new Date(ev.dateISO).getTime()).filter(t => !isNaN(t)));

  const series = [];
  // Step from (minDate + WINDOW) to now, weekly
  let cursor = minDate + WINDOW_MS;
  while (cursor <= nowMs + STEP_MS) {
    const endMs   = Math.min(cursor, nowMs);
    const startMs = endMs - WINDOW_MS;
    const r = _esmScoreWindow(events, ccy, startMs, endMs);
    if (r !== null) {
      const dt = new Date(endMs);
      const iso = `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`;
      series.push({ time: iso, value: parseFloat(r.idx.toFixed(2)) });
    }
    cursor += STEP_MS;
  }
  return series;
}

// ── Current 90d score (for header metrics) ───────────────────────────────────
function _esmCurrentScore(events, ccy) {
  const nowMs = Date.now();
  const W90   = 90 * 24 * 60 * 60 * 1000;
  return _esmScoreWindow(events, ccy, nowMs - W90, nowMs);
}

// ── Collect individual events for the table ──────────────────────────────────
function _esmGetEvents(events, ccy) {
  const nowMs   = Date.now();
  const W90_MS  = 90 * 24 * 60 * 60 * 1000;
  const seen    = new Set();
  const result  = [];

  events.forEach(ev => {
    if (ev.currency !== ccy) return;
    const t = new Date(ev.dateISO).getTime();
    if (isNaN(t) || t > nowMs || nowMs - t > W90_MS) return;
    if (!ev.actual || ev.actual === '' || ev.actual === '-') return;
    if (!['medium','high'].includes(ev.impact)) return;

    const evTitle  = (ev.event || ev.title || '').toLowerCase();
    if (_ESM_NOISE_KW.some(kw => evTitle.includes(kw))) return;

    const actualS  = String(ev.actual   || '').replace(/[%,\s]/g, '');
    const forecastS= String(ev.forecast || ev.previous || '').replace(/[%,\s]/g, '');
    const canon    = evTitle.replace(/\s*\([^)]*\)/g, '').trim();
    const dedupKey = `${ccy}/${canon}/${actualS}/${forecastS}`;
    if (seen.has(dedupKey)) return;
    seen.add(dedupKey);

    const actual   = parseFloat(String(ev.actual   || '').replace(/[%,]/g, ''));
    const forecast = parseFloat(String(ev.forecast || ev.previous || '').replace(/[%,]/g, ''));
    const hasFc    = !isNaN(actual) && !isNaN(forecast);

    const isInverse = _ESM_INVERSE_KW.some(kw => evTitle.includes(kw));
    let outcome = 'n/a';
    if (hasFc) {
      const beat = isInverse ? actual < forecast : actual > forecast;
      const miss = isInverse ? actual > forecast : actual < forecast;
      outcome = beat ? 'beat' : miss ? 'miss' : 'inline';
    }

    result.push({
      dateISO:  ev.dateISO,
      event:    ev.event || ev.title || '',
      impact:   ev.impact,
      actual:   ev.actual   || '—',
      forecast: ev.forecast || ev.previous || '—',
      outcome,
    });
  });

  result.sort((a, b) => b.dateISO.localeCompare(a.dateISO));
  return result;
}

// ── Chart destroy helper ──────────────────────────────────────────────────────
function _esmDestroyChart() {
  if (_esmChart) {
    try { _esmChart.remove(); } catch (_) {}
    _esmChart = null;
  }
}

// ── Resize observer ──────────────────────────────────────────────────────────
function _esmBindResize(container, lwChart) {
  const apply = () => {
    requestAnimationFrame(() => {
      const rect = container.getBoundingClientRect();
      const w = Math.round(rect.width)  || container.offsetWidth  || 600;
      const h = Math.round(rect.height) || container.offsetHeight || 260;
      if (lwChart && w > 0 && h > 10) lwChart.applyOptions({ width: w, height: h });
    });
  };
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => apply());
    ro.observe(container);
    container._esmRo = ro;
  }
  window.addEventListener('resize', apply);
  container._esmResize = apply;
  setTimeout(apply, 60);
  setTimeout(apply, 250);
  setTimeout(apply, 600);
}

// ── Build (or rebuild) chart for the given currency ───────────────────────────
function _esmRenderChart(ccy) {
  const LWC = window.LightweightCharts;
  if (!LWC) return;
  if (!_esmCalData) return;

  _esmDestroyChart();

  const container = document.getElementById('esm-chart-inner');
  if (!container) return;

  const events = _esmCalData.events || [];
  const series = _esmBuildSeries(events, ccy);

  const W = container.offsetWidth  || 600;
  const H = container.offsetHeight || 260;

  const chart = LWC.createChart(container, {
    width:  W,
    height: H,
    layout: {
      background: { type: 'solid', color: '#131722' },
      textColor: '#6e7681',
      fontFamily: _esmMonoF,
      fontSize: 10,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: 'rgba(255,255,255,0.04)' },
      horzLines: { color: 'rgba(255,255,255,0.04)' },
    },
    crosshair: {
      mode: LWC.CrosshairMode?.Normal ?? 1,
      vertLine: { color: 'rgba(255,255,255,0.2)', style: 2, labelVisible: false },
      horzLine: { color: 'rgba(255,255,255,0.15)', style: 2, labelVisible: true },
    },
    rightPriceScale: {
      borderVisible: false,
      scaleMargins: { top: 0.12, bottom: 0.12 },
    },
    timeScale: {
      borderVisible: false,
      fixRightEdge: true,
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale:  { mouseWheel: true, pinch: true },
    localization: {
      priceFormatter: v => v != null ? (v >= 0 ? '+' : '') + v.toFixed(1) : '—',
    },
  });

  _esmChart = chart;

  // Zero reference line (Priceline at 0)
  const zeroLine = chart.addSeries(LWC.LineSeries, {
    color: 'rgba(110,118,129,0.4)',
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  // Extend zero line across the full date range
  if (series.length >= 2) {
    zeroLine.setData([
      { time: series[0].time, value: 0 },
      { time: series[series.length - 1].time, value: 0 },
    ]);
  }

  // Determine positive/negative for gradient color — use last value
  const lastVal = series.length > 0 ? series[series.length - 1].value : 0;
  const isPositive = lastVal >= 0;
  const lineColor   = isPositive ? '#26a69a' : '#ef5350';
  const topColor    = isPositive ? 'rgba(38,166,154,0.28)' : 'rgba(239,83,80,0.28)';
  const bottomColor = isPositive ? 'rgba(38,166,154,0.02)' : 'rgba(239,83,80,0.02)';

  const areaSeries = chart.addSeries(LWC.AreaSeries, {
    lineColor,
    topColor,
    bottomColor,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: true,
    crosshairMarkerRadius: 4,
  });

  if (series.length > 0) {
    areaSeries.setData(series);
    chart.timeScale().fitContent();
  }

  // ── Tooltip ──────────────────────────────────────────────────────────────
  const tip = document.createElement('div');
  tip.className = 'esm-lw-tooltip';
  container.style.position = 'relative';
  container.appendChild(tip);

  const TW = 180, TM = 14;
  chart.subscribeCrosshairMove(param => {
    if (!param?.point || !param.seriesData) { tip.style.display = 'none'; return; }
    const d = param.seriesData.get(areaSeries);
    if (!d) { tip.style.display = 'none'; return; }

    const val   = d.value;
    const col   = val >= 0 ? '#26a69a' : '#ef5350';
    const sign  = val >= 0 ? '+' : '';
    const dateStr = typeof param.time === 'string' ? param.time : '';
    // Find events scored in this 30d window
    const endMs   = new Date(dateStr).getTime();
    const startMs = endMs - 30 * 24 * 60 * 60 * 1000;
    const winScore = _esmScoreWindow(_esmCalData.events || [], ccy, startMs, endMs);
    const nTxt = winScore ? `${winScore.total} events` : '';

    tip.innerHTML = `
      <div style="font-size:9px;color:var(--text2,#787b86);margin-bottom:4px;">${dateStr} · 30d window</div>
      <div style="font-size:12px;font-weight:700;color:${col}">${sign}${val.toFixed(1)}</div>
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

  _esmBindResize(container, chart);
}

// ── Populate metrics bar ──────────────────────────────────────────────────────
function _esmRenderMetrics(ccy) {
  if (!_esmCalData) return;
  const events = _esmCalData.events || [];
  const s = _esmCurrentScore(events, ccy);

  const el = id => document.getElementById(id);

  if (!s) {
    el('esm-m-index').textContent   = '—';
    el('esm-m-beats').textContent   = '—';
    el('esm-m-misses').textContent  = '—';
    el('esm-m-n').textContent       = '—';
    el('esm-m-beat-rt').textContent = '—';
    el('esm-m-index').style.color = 'var(--text2)';
    return;
  }

  const idx    = s.idx;
  const idxCol = idx > 5 ? 'var(--up,#26a69a)' : idx < -5 ? 'var(--down,#ef5350)' : 'var(--text,#d1d4dc)';
  const inLine = s.total - s.beats - s.misses;
  const beatRt = s.total > 0 ? (s.beats / s.total * 100).toFixed(0) + '%' : '—';

  el('esm-m-index').textContent   = (idx >= 0 ? '+' : '') + idx.toFixed(1);
  el('esm-m-index').style.color   = idxCol;
  el('esm-m-beats').textContent   = s.beats;
  el('esm-m-beats').style.color   = 'var(--up,#26a69a)';
  el('esm-m-misses').textContent  = s.misses;
  el('esm-m-misses').style.color  = 'var(--down,#ef5350)';
  el('esm-m-n').textContent       = s.total;
  el('esm-m-beat-rt').textContent = beatRt;
  el('esm-m-beat-rt').style.color = parseFloat(beatRt) >= 50 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
}

// ── Populate events table ─────────────────────────────────────────────────────
function _esmRenderTable(ccy) {
  const tbody = document.getElementById('esm-evt-tbody');
  if (!tbody) return;
  if (!_esmCalData) { tbody.innerHTML = '<tr><td colspan="5" class="esm-no-data">Loading…</td></tr>'; return; }

  const rows = _esmGetEvents(_esmCalData.events || [], ccy);

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="esm-no-data" style="padding:20px 16px;">No released events with actuals in the 90d window for ${ccy}.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const badgeCls  = r.outcome === 'beat' ? 'esm-badge-beat' : r.outcome === 'miss' ? 'esm-badge-miss' : 'esm-badge-inline';
    const badgeTxt  = r.outcome === 'beat' ? 'BEAT' : r.outcome === 'miss' ? 'MISS' : r.outcome === 'inline' ? 'IN LINE' : '—';
    const impactCls = r.impact === 'high' ? 'esm-impact-h' : 'esm-impact-m';
    const actualCls = r.outcome === 'beat' ? 'esm-beat' : r.outcome === 'miss' ? 'esm-miss' : '';
    const dateDisp  = r.dateISO.slice(5).replace('-', '/');
    return `<tr>
      <td style="color:var(--text2,#787b86)">${dateDisp}</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.event}">
        <span class="${impactCls}" style="font-size:8px;margin-right:4px;">${r.impact === 'high' ? 'H' : 'M'}</span>${r.event}
      </td>
      <td class="${actualCls}">${r.actual}</td>
      <td style="color:var(--text2,#787b86)">${r.forecast}</td>
      <td><span class="${badgeCls}">${badgeTxt}</span></td>
    </tr>`;
  }).join('');
}

// ── Switch currency tab ───────────────────────────────────────────────────────
function esmTab(el, ccy) {
  document.querySelectorAll('.esm-tab').forEach(t => {
    t.classList.remove('on');
    t.setAttribute('aria-selected', 'false');
  });
  el.classList.add('on');
  el.setAttribute('aria-selected', 'true');
  _esmActiveCcy = ccy;

  // Update title
  const titleEl = document.getElementById('esm-title');
  if (titleEl) titleEl.textContent = `Economic Surprises · ${ccy}`;

  _esmRenderMetrics(ccy);
  _esmRenderChart(ccy);
  _esmRenderTable(ccy);
}

// ── Close modal ───────────────────────────────────────────────────────────────
function closeESModal() {
  _esmDestroyChart();
  const bd = document.getElementById('esm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _esmKeydown);
}

function _esmKeydown(e) {
  if (e.key === 'Escape') closeESModal();
}

// ── Open modal ────────────────────────────────────────────────────────────────
async function openEconSurprisesModal(initialCcy) {
  closeESModal(); // reset any stale instance

  const ccy = _ESM_G8.includes(initialCcy) ? initialCcy : 'USD';
  _esmActiveCcy = ccy;

  // ── Build DOM ──────────────────────────────────────────────────────────────
  const bd = document.createElement('div');
  bd.id = 'esm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', `Economic Surprises — ${ccy}`);

  bd.innerHTML = `
<div id="esm-hd">
  <div id="esm-hd-left">
    <div>
      <div id="esm-badge">CESI-STYLE INDEX</div>
      <div id="esm-title">Economic Surprises &middot; ${ccy}</div>
      <div id="esm-sub">ForexFactory &middot; actual vs consensus &middot; 30d rolling window &middot; [&minus;100, +100]</div>
    </div>
  </div>
  <button id="esm-close" onclick="closeESModal()" aria-label="Close Economic Surprises modal">&times;</button>
</div>

<div id="esm-metrics">
  <div class="esm-mm">
    <div class="esm-mm-lbl">Index (90d)</div>
    <div class="esm-mm-val" id="esm-m-index" style="color:var(--text2)">—</div>
    <div class="esm-mm-sub">[-100, +100]</div>
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

<div id="esm-tabs" role="tablist" aria-label="G8 currency tabs">
  ${_ESM_G8.map(c => `
  <div class="esm-tab${c === ccy ? ' on' : ''}"
       data-ccy="${c}"
       role="tab"
       aria-selected="${c === ccy ? 'true' : 'false'}"
       onclick="esmTab(this,'${c}')"
       tabindex="${c === ccy ? '0' : '-1'}">
    ${c}
  </div>`).join('')}
</div>

<div id="esm-body">
  <div id="esm-chart-wrap">
    <div id="esm-chart-inner"></div>
  </div>
  <div id="esm-events-wrap">
    <div id="esm-events-hd">
      <div id="esm-events-hd-title">Events (90d rolling window)</div>
      <div style="font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);">
        H&nbsp;=&nbsp;high&nbsp;impact &nbsp;·&nbsp; M&nbsp;=&nbsp;medium
      </div>
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
`;

  document.body.appendChild(bd);
  document.addEventListener('keydown', _esmKeydown);

  // ── Load calendar data (re-use window cache from renderEconSurprises if fresh) ──
  try {
    const res  = await fetch('./calendar-data/calendar.json').catch(() => null);
    if (res?.ok) {
      const calj = await res.json();
      _esmCalData = calj;
      // Sync surpriseStats for z-score scoring (same key used by renderEconSurprises)
      if (calj.surpriseStats) {
        window._ECON_SURPRISE_STATS = calj.surpriseStats;
      }
    }
  } catch (_) { /* graceful — table will show no-data */ }

  // ── Ensure LWC then render ────────────────────────────────────────────────
  try {
    await _esmEnsureLWC();
  } catch (_) { /* chart will not render but table still works */ }

  _esmRenderMetrics(ccy);
  _esmRenderChart(ccy);
  _esmRenderTable(ccy);
}

// ── Expose globals ────────────────────────────────────────────────────────────
window.openEconSurprisesModal = openEconSurprisesModal;
window.closeESModal           = closeESModal;
window.esmTab                 = esmTab;
