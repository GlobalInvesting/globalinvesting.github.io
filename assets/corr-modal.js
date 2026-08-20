// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v2.3  — inline-panel edition
// Fluid layout, terminal CSS variables throughout.
// v2.3: renamed to "Correlations" (title no longer per-pair); added Cross
// asset / Matrix tabs; 30d/60d/90d window buttons moved beneath the tab bar
// and now drive both tabs; added a G10 currency correlation matrix tab
// (client-side, mirrors the EA's DrawCorrelation()); added a hedge-ratio
// spread z-score pairs-trade signal to the Cross asset tab (spread_z/signal/
// beta from fetch_intraday_quotes.py fetch_correlations()).
// ═══════════════════════════════════════════════════════════════════════════
(function () {
  if (document.getElementById('cm2-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cm2-modal-css';
  s.textContent = `
#cm-bd { display:block!important; }
#cm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}
#cm-modal::before { display:none; }
#cm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 8px;border-bottom:1px solid var(--border,#252d3d);flex-shrink:0;
  background:var(--bg2);
}
#cm-title { font-size:12px;font-weight:600;color:var(--text);letter-spacing:-.01em; }
#cm-sub   { font-size:9px;color:var(--text2);margin-top:1px;font-family:var(--font-mono); }
#cm-close { background:none;border:none;color:var(--text2);font-size:16px;cursor:pointer;padding:3px 6px;border-radius:4px;line-height:1;transition:color .1s,background .1s; }
#cm-close:hover { color:var(--text);background:var(--bg3); }
#cm-strip {
  display:grid;grid-template-columns:repeat(5,1fr);
  background:var(--bg);border-bottom:1px solid var(--border,#252d3d);flex-shrink:0;
}
.cm-metric { padding:7px 10px;background:var(--bg);border-right:1px solid var(--border,#252d3d); }
.cm-metric:last-child { border-right:none; }
.cm-m-lbl { font-size:8px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;font-family:var(--font-mono); }
.cm-m-val { font-size:14px;font-weight:600;font-family:var(--font-mono);line-height:1;color:var(--text); }
.cm-m-val.up   { color:var(--up); }
.cm-m-val.down { color:var(--down); }
.cm-m-val.warn { color:var(--orange); }
.cm-m-sub { font-size:8px;color:var(--text2);margin-top:2px;font-family:var(--font-mono); }
#cm-body {
  flex:1;overflow-y:auto;padding:0;
  display:flex;flex-direction:column;
  background:var(--bg);scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent;
}
#cm-body::-webkit-scrollbar { width:3px!important; }
#cm-body::-webkit-scrollbar-track { background:transparent; }
#cm-body::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
#cm-body::-webkit-scrollbar-thumb:hover { background:var(--text2); }
.cm-section-title { display:none; }
#cm-chart-wrap { position:relative;flex-shrink:0;border-bottom:1px solid var(--border,#252d3d); }
#cm-lwc-container { width:100%;height:200px; }
#cm-tooltip { position:absolute;top:6px;left:10px;background:var(--bg2);border:1px solid var(--border,#252d3d);border-radius:4px;padding:4px 8px;font-size:9px;font-family:var(--font-mono);color:var(--text);pointer-events:none;display:none;z-index:10;white-space:nowrap; }
#cm-legend { display:flex;gap:12px;flex-wrap:wrap;padding:8px 14px;border-bottom:1px solid var(--border,#252d3d); }
.cm-leg-item { display:flex;align-items:center;gap:4px;font-size:8.5px;color:var(--text2);font-family:var(--font-mono); }
.cm-leg-swatch { width:14px;height:2px;border-radius:1px;flex-shrink:0; }
.cm-leg-swatch.solid-blue { background:var(--blue); }
.cm-leg-swatch.dash-white { background:repeating-linear-gradient(90deg,rgba(209,212,220,.5) 0,rgba(209,212,220,.5) 3px,transparent 3px,transparent 6px); }
.cm-leg-swatch.dash-amber { background:repeating-linear-gradient(90deg,rgba(246,148,28,.8) 0,rgba(246,148,28,.8) 3px,transparent 3px,transparent 6px); }
.cm-leg-swatch.dash-red   { background:repeating-linear-gradient(90deg,rgba(239,83,80,.8) 0,rgba(239,83,80,.8) 3px,transparent 3px,transparent 6px); }
.cm-regime-row { display:flex;justify-content:space-between;align-items:baseline;padding:8px 14px;border-bottom:1px solid var(--border,#252d3d); }
.cm-regime-row:last-child { border-bottom:none; }
.cm-regime-key { font-size:9.5px;color:var(--text2);font-family:var(--font-mono); }
.cm-regime-val { font-size:10px;font-weight:600;font-family:var(--font-mono);color:var(--text); }
.cm-regime-val.up   { color:var(--up); }
.cm-regime-val.down { color:var(--down); }
.cm-regime-val.warn { color:var(--orange); }
.cm-regime-val.flat { color:var(--text2); }
.cm-trend-rising  { color:var(--up); }
.cm-trend-falling { color:var(--down); }
.cm-trend-stable  { color:var(--text2); }
.cm-signal { display:flex;align-items:baseline;gap:8px;padding:8px 14px;border-top:1px solid var(--border,#252d3d);border-left:3px solid var(--border,#252d3d);font-size:9.5px;line-height:1.5;color:var(--text2);font-family:var(--font-mono);margin:0; }
.cm-signal.warn { border-left-color:var(--down); }
.cm-signal.ok   { border-left-color:var(--up); }
.cm-signal-tag { font-size:8.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;flex-shrink:0; }
.cm-signal.warn .cm-signal-tag { color:var(--down); }
.cm-signal.ok   .cm-signal-tag { color:var(--up); }
.cm-signal-body { color:var(--text2); }
.cm-related-title { font-size:8.5px;font-weight:600;color:var(--text3,#4e5c70);text-transform:uppercase;letter-spacing:.08em;padding:10px 14px 4px;border-top:1px solid var(--border,#252d3d); }
.cm-related-row { display:flex;justify-content:space-between;align-items:baseline;padding:6px 14px;border-bottom:1px solid var(--border,#252d3d); }
.cm-related-row:last-child { border-bottom:none; }
.cm-related-key { font-size:9.5px;color:var(--text2);font-family:var(--font-mono); }
.cm-related-val { font-size:10px;font-weight:600;font-family:var(--font-mono);color:var(--text); }
.cm-related-val.up   { color:var(--up); }
.cm-related-val.down { color:var(--down); }
#cm-tabs { display:flex;gap:2px;padding:8px 14px 0;background:var(--bg2);border-bottom:1px solid var(--border,#252d3d);flex-shrink:0; }
.cm-tab { flex:0 0 auto;font-size:10.5px;font-family:var(--font-ui,sans-serif);padding:6px 12px;background:transparent;border:1px solid transparent;border-bottom:none;color:var(--text3,#4e5c70);cursor:pointer;border-radius:3px 3px 0 0;transition:color .1s,background .1s; }
.cm-tab:hover { color:var(--text2); }
.cm-tab.active { background:var(--bg);border-color:var(--border,#252d3d);border-bottom:2px solid var(--blue);color:var(--text); }
#cm-window-btns { display:flex;align-items:center;justify-content:flex-end;gap:2px;padding:6px 14px;background:var(--bg2);border-bottom:1px solid var(--border,#252d3d);flex-shrink:0; }
.cm-win-btn { font-size:8px;padding:2px 6px;background:var(--bg3);border:1px solid var(--border2,#363c4e);color:var(--text3,#4e5c70);border-radius:2px;cursor:pointer;line-height:1.4;font-family:var(--font-mono); }
.cm-win-btn.active { color:#fff;border-color:var(--blue); }
.cm-pair-hd { padding:10px 14px 2px; }
#cm-pair-title { font-size:12px;font-weight:600;color:var(--text); }
#cm-pair-sub { font-size:9px;color:var(--text2);margin-top:1px;font-family:var(--font-mono); }
.cm-metric.win-active { background:var(--bg2);box-shadow:inset 0 2px 0 var(--blue); }
.cm-psig { display:flex;flex-direction:column;gap:3px;margin:8px 14px;padding:8px 10px;background:var(--bg2);border:1px solid var(--border,#252d3d);border-left:3px solid var(--blue);border-radius:0 4px 4px 0; }
.cm-psig-hd { display:flex;align-items:center;justify-content:space-between; }
.cm-psig-tag { font-size:8.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--blue); }
.cm-psig-meta { font-size:8px;color:var(--text3,#4e5c70);font-family:var(--font-mono); }
.cm-psig-dir { font-size:11px;font-weight:600;color:var(--text);font-family:var(--font-mono); }
.cm-psig-note { font-size:9px;color:var(--text2);line-height:1.5; }
.cm-psig.short { border-left-color:var(--down); }
.cm-psig.short .cm-psig-tag { color:var(--down); }
.cm-psig.long { border-left-color:var(--up); }
.cm-psig.long .cm-psig-tag { color:var(--up); }
.cm-psig.neutral, .cm-psig.pending { border-left-color:var(--text3,#4e5c70); }
.cm-psig.neutral .cm-psig-tag, .cm-psig.pending .cm-psig-tag { color:var(--text3,#4e5c70); }
#cm-matrix-wrap { padding:10px 14px 14px; }
#cm-matrix-sub { font-size:9px;color:var(--text2);margin-bottom:8px;font-family:var(--font-mono); }
.cm-mtx-table { border-collapse:collapse;width:100%;table-layout:fixed; }
.cm-mtx-table td,.cm-mtx-table th { text-align:center;font-size:9px;font-family:var(--font-mono);padding:0; }
.cm-mtx-hd { color:var(--text3,#4e5c70);padding-bottom:4px!important;font-weight:400; }
.cm-mtx-lbl { text-align:left!important;color:var(--blue);padding:0 4px 0 0!important;font-weight:600;width:26px; }
.cm-mtx-cell { height:26px;border:1px solid var(--border,#252d3d); }
`;
  document.head.appendChild(s);
})();

let _cmChart = null;
function _cmCls(v) { if (v == null) return ''; return v >= 0.3 ? 'up' : v <= -0.3 ? 'down' : ''; }
function _cmZcls(z) { if (z == null) return ''; const a = Math.abs(z); return a >= 2.5 ? 'down' : a >= 1.5 ? 'warn' : ''; }
function _cmFmt(v, d) { if (v == null) return '\u2014'; return (v >= 0 ? '+' : '') + v.toFixed(d ?? 2); }
function _cmParseDate(iso) { if (!iso || typeof iso !== 'string') return null; const p = iso.split('-'); if (p.length < 3) return null; return { year: +p[0], month: +p[1], day: +p[2] }; }
function _cmFmtDate(iso) { if (!iso) return ''; try { const d = new Date(iso + 'T12:00:00Z'); return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }); } catch (_) { return iso; } }

function _cmDrawChart(container, history, histDates, norm, std) {
  if (!container || !window.LightweightCharts || !history || !history.length) return;
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }
  const LWC = window.LightweightCharts;
  const n = history.length;
  const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(3);
  const hasRealDates = Array.isArray(histDates) && histDates.length === n;
  const mkPt = (i, v) => { if (hasRealDates) { const t = _cmParseDate(histDates[i]); if (t) return { time: t, value: v }; } return { time: i + 1, value: v }; };
  const tFirst = mkPt(0, 0).time, tLast = mkPt(n - 1, 0).time;
  const hLine = (v) => [{ time: tFirst, value: v }, { time: tLast, value: v }];
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#131722';
  const text2 = getComputedStyle(document.documentElement).getPropertyValue('--text2').trim() || '#9096a0';

  _cmChart = LWC.createChart(container, {
    layout: { background: { type: 'solid', color: bg }, textColor: text2, fontSize: 9, fontFamily: getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim()||"'Courier New',monospace", attributionLogo: false },
    grid: { vertLines: { color: 'rgba(255,255,255,.04)' }, horzLines: { color: 'rgba(255,255,255,.04)' } },
    crosshair: { mode: LWC.CrosshairMode.Magnet, vertLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--bg3').trim() || '#2a2e39' }, horzLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--bg3').trim() || '#2a2e39' } },
    rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.06, bottom: 0.06 } },
    timeScale: { borderVisible: false, tickMarkFormatter: hasRealDates ? (time) => { try { const d = new Date(Date.UTC(time.year, time.month - 1, time.day)); return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit', timeZone: 'UTC' }); } catch (_) { return ''; } } : undefined },
    handleScroll: false, handleScale: false,
  });

  const zeroSer = _cmChart.addSeries(LWC.LineSeries, { color: 'rgba(255,255,255,.12)', lineWidth: 1, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, priceFormat: { type: 'custom', formatter: fmt } });
  zeroSer.setData(hLine(0));
  if (norm != null) {
    const normSer = _cmChart.addSeries(LWC.LineSeries, { color: 'rgba(209,212,220,.4)', lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, priceFormat: { type: 'custom', formatter: fmt } });
    normSer.setData(hLine(norm));
    if (std != null) {
      [[norm + 1.5 * std, 'rgba(246,148,28,.7)'], [norm - 1.5 * std, 'rgba(246,148,28,.7)'], [norm + 2.5 * std, 'rgba(239,83,80,.7)'], [norm - 2.5 * std, 'rgba(239,83,80,.7)']].forEach(([val, color]) => {
        const ser = _cmChart.addSeries(LWC.LineSeries, { color, lineWidth: 1, lineStyle: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, priceFormat: { type: 'custom', formatter: fmt } });
        ser.setData(hLine(val));
      });
    }
  }
  const blue = getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim() || '#4f7fff';
  const mainSer = _cmChart.addSeries(LWC.LineSeries, { color: blue, lineWidth: 2, lastValueVisible: true, priceLineVisible: false, crosshairMarkerRadius: 4, priceFormat: { type: 'custom', formatter: fmt } });
  mainSer.setData(history.map((v, i) => mkPt(i, v)));
  [30, 60, 90].forEach((days) => {
    const idx = n - days; if (idx < 0) return;
    const tMark = mkPt(idx, 0).time;
    const markerSer = _cmChart.addSeries(LWC.LineSeries, { color: 'rgba(255,255,255,.18)', lineWidth: 1, lineStyle: 3, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, priceFormat: { type: 'custom', formatter: () => days + 'd' } });
    markerSer.setData([{ time: tMark, value: -1 }, { time: tMark, value: 1 }]);
  });
  const tooltip = document.getElementById('cm-tooltip');
  if (tooltip) {
    _cmChart.subscribeCrosshairMove(param => {
      if (!param || !param.point || !param.seriesData) { tooltip.style.display = 'none'; return; }
      const val = param.seriesData.get(mainSer);
      if (val == null) { tooltip.style.display = 'none'; return; }
      let dateLabel = '';
      if (hasRealDates && val.time && val.time.year) {
        const iso = val.time.year + '-' + String(val.time.month).padStart(2, '0') + '-' + String(val.time.day).padStart(2, '0');
        dateLabel = _cmFmtDate(iso);
      } else if (val.time) { const idx = (val.time || 1) - 1; const daysAgo = n - 1 - idx; dateLabel = daysAgo === 0 ? 'today' : daysAgo + 'd ago'; }
      tooltip.style.display = 'block';
      tooltip.textContent = fmt(val.value) + (dateLabel ? '  \u00b7  ' + dateLabel : '');
    });
  }

  const applySize = () => {
    requestAnimationFrame(() => {
      const w = container.offsetWidth || 600;
      const h = container.offsetHeight || 200;
      if (_cmChart && w > 0 && h > 10) _cmChart.applyOptions({ width: w, height: h });
    });
  };
  if (window.ResizeObserver) { const ro = new ResizeObserver(applySize); ro.observe(container); container._cmRo = ro; }
  window.addEventListener('resize', applySize); container._cmResize = applySize;
  setTimeout(applySize, 60); setTimeout(applySize, 250);
}

// ── Tab / window-selector state (shared by Cross asset + Matrix tabs) ───────
let _cmActiveTab = 'cross';
let _cmWindow = 60;
let _cmCurrentObj = null;

// ── G10 currency matrix — client-side, mirrors the EA's DrawCorrelation() ──
// Same symbol set/order/labels as CORR_SYMS_X / CORR_LABELS_X in the EA, and
// the same compact "value * 10" display + banded heat coloring. Computed from
// the same daily ohlc-data/*.json files the chart already uses — no new
// backend endpoint needed, matching the EA's own live-computed matrix.
const _CM_MTX_SYMS = [
  { id: 'eurusd', label: 'EUR' }, { id: 'gbpusd', label: 'GBP' }, { id: 'usdjpy', label: 'JPY' },
  { id: 'audusd', label: 'AUD' }, { id: 'usdcad', label: 'CAD' }, { id: 'usdchf', label: 'CHF' },
  { id: 'nzdusd', label: 'NZD' }, { id: 'usdnok', label: 'NOK' }, { id: 'usdsek', label: 'SEK' },
  { id: 'dxy',    label: 'DXY' },
];
let _cmMtxRaw = null;
let _cmMtxLoading = null;
function _cmLoadMatrixSeries() {
  if (_cmMtxRaw) return Promise.resolve(_cmMtxRaw);
  if (_cmMtxLoading) return _cmMtxLoading;
  _cmMtxLoading = Promise.all(_CM_MTX_SYMS.map(s =>
    fetch('./ohlc-data/' + s.id + '.json', { cache: 'no-store' })
      .then(r => r.ok ? r.json() : [])
      .catch(() => [])
      .then(bars => [s.id, Array.isArray(bars) ? bars : []])
  )).then(pairs => {
    const out = {};
    pairs.forEach(([id, bars]) => { out[id] = bars; });
    _cmMtxRaw = out;
    return out;
  });
  return _cmMtxLoading;
}
function _cmPearsonArr(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 10) return null;
  const xs = x.slice(-n), ys = y.slice(-n);
  const mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { const a = xs[i] - mx, b = ys[i] - my; num += a * b; dx += a * a; dy += b * b; }
  if (dx === 0 || dy === 0) return null;
  return num / Math.sqrt(dx * dy);
}
function _cmComputeMatrix(raw, windowDays) {
  const bySym = {};
  _CM_MTX_SYMS.forEach(s => {
    const m = new Map();
    (raw[s.id] || []).forEach(bar => { if (bar && bar.time != null && bar.close != null) m.set(bar.time, bar.close); });
    bySym[s.id] = m;
  });
  let commonDates = null;
  _CM_MTX_SYMS.forEach(s => {
    const ds = new Set(bySym[s.id].keys());
    commonDates = commonDates ? new Set([...commonDates].filter(d => ds.has(d))) : ds;
  });
  const sortedDates = Array.from(commonDates || []).sort();
  const useDates = sortedDates.slice(-(windowDays + 1));
  if (useDates.length < 15) return null;

  const returns = {};
  _CM_MTX_SYMS.forEach(s => {
    const closes = useDates.map(d => bySym[s.id].get(d));
    const r = [];
    for (let i = 1; i < closes.length; i++) if (closes[i - 1] > 0 && closes[i] > 0) r.push(Math.log(closes[i] / closes[i - 1]));
    returns[s.id] = r;
  });
  const n = _CM_MTX_SYMS.length;
  const matrix = [];
  for (let i = 0; i < n; i++) {
    matrix.push([]);
    for (let j = 0; j < n; j++) matrix[i].push(i === j ? 1 : _cmPearsonArr(returns[_CM_MTX_SYMS[i].id], returns[_CM_MTX_SYMS[j].id]));
  }
  return { matrix, n: useDates.length - 1 };
}
function _cmMtxBg(v) {
  if (v == null) return 'var(--bg2)';
  if (v >= 0.70) return 'rgba(38,166,91,.32)';
  if (v >= 0.30) return 'rgba(38,166,91,.15)';
  if (v >= -0.30) return 'var(--bg2)';
  if (v >= -0.70) return 'rgba(239,83,80,.15)';
  return 'rgba(239,83,80,.32)';
}
function _cmMtxFg(v) { if (v == null) return 'var(--text3)'; if (v >= 0.30) return 'var(--up)'; if (v <= -0.30) return 'var(--down)'; return 'var(--text2)'; }

async function _cmRenderMatrixBody() {
  const tabBody = document.getElementById('cm-tab-body');
  if (!tabBody) return;
  tabBody.innerHTML = '<div id="cm-matrix-wrap"><div id="cm-matrix-sub">Loading G10 correlation matrix\u2026</div></div>';
  const raw = await _cmLoadMatrixSeries();
  if (_cmActiveTab !== 'matrix' || !document.getElementById('cm-tab-body')) return; // tab/modal changed meanwhile
  const result = _cmComputeMatrix(raw, _cmWindow);
  const mw = document.getElementById('cm-matrix-wrap');
  if (!mw) return;
  if (!result) { mw.innerHTML = '<div id="cm-matrix-sub">Not enough overlapping daily history to compute the matrix.</div>'; return; }
  const { matrix, n } = result;
  let html = '<div id="cm-matrix-sub">G10 currency correlation matrix \u00b7 log-return Pearson \u00b7 ' + _cmWindow + 'd (' + n + ' sess.)</div>';
  html += '<table class="cm-mtx-table" aria-label="G10 currency correlation matrix"><tr><th class="cm-mtx-hd"></th>' +
    _CM_MTX_SYMS.map(s => '<th class="cm-mtx-hd">' + s.label + '</th>').join('') + '</tr>';
  _CM_MTX_SYMS.forEach((s, i) => {
    html += '<tr><td class="cm-mtx-lbl">' + s.label + '</td>' +
      _CM_MTX_SYMS.map((s2, j) => {
        const v = matrix[i][j], isDiag = i === j;
        const txt = isDiag ? '\u2014' : (v == null ? '\u2014' : (v * 10).toFixed(1));
        const title = isDiag ? s.label : (v == null ? 'Insufficient data' : (s.label + '/' + s2.label + ': ' + (v >= 0 ? '+' : '') + v.toFixed(2)));
        return '<td class="cm-mtx-cell" style="background:' + (isDiag ? 'var(--bg2)' : _cmMtxBg(v)) + ';color:' + (isDiag ? 'var(--text3)' : _cmMtxFg(v)) + ';" title="' + title + '">' + txt + '</td>';
      }).join('') + '</tr>';
  });
  html += '</table>';
  mw.innerHTML = html;
}

// ── Pairs-trade signal — hedge-ratio spread z-score, computed server-side ──
// (fetch_intraday_quotes.py fetch_correlations(), spread_z/signal/beta fields).
// `signal` is undefined on cached data from before this feature shipped (not
// yet re-fetched), null when the underlying correlation was too weak this
// window for a stable hedge ratio, or one of 'long_a_short_b'/'short_a_long_b'/
// 'neutral' once computed. Handle all three states rather than assuming data.
function _cmPairsSignalHtml(corrObj) {
  const { a, b, signal, spread_z, beta } = corrObj;
  if (typeof signal === 'undefined') {
    return '<div class="cm-psig pending"><div class="cm-psig-hd"><span class="cm-psig-tag">Pairs signal</span><span class="cm-psig-meta">pending</span></div>' +
      '<div class="cm-psig-note">Not available for this pair yet \u2014 lands with the next scheduled data refresh.</div></div>';
  }
  if (signal == null) {
    return '<div class="cm-psig neutral"><div class="cm-psig-hd"><span class="cm-psig-tag">Pairs signal</span><span class="cm-psig-meta">60d hedge-ratio spread</span></div>' +
      '<div class="cm-psig-note">Underlying correlation too weak this window for a stable hedge ratio \u2014 no directional read.</div></div>';
  }
  const dirCls = signal === 'neutral' ? 'neutral' : (signal === 'short_a_long_b' ? 'short' : 'long');
  const dirText = signal === 'neutral' ? 'No stretch \u2014 spread within \u00b11\u03c3'
    : signal === 'short_a_long_b' ? 'Short ' + a + ' \u00b7 Long ' + b
    : 'Long ' + a + ' \u00b7 Short ' + b;
  const zTxt = spread_z != null ? (spread_z >= 0 ? '+' : '') + spread_z.toFixed(2) + '\u03c3' : '\u2014';
  const note = signal === 'neutral'
    ? 'The hedge-ratio-adjusted spread between the two legs is tracking within its normal range \u2014 no mean-reversion edge currently.'
    : 'The hedge-ratio-adjusted spread (beta ' + (beta != null ? beta.toFixed(2) : '\u2014') + ') is stretched ' + zTxt + ' vs its own 60d norm \u2014 ' + (signal === 'short_a_long_b' ? a : b) + ' has run ahead of what the pair\u2019s usual co-movement implies. Mean-reversion read, not investment advice.';
  return '<div class="cm-psig ' + dirCls + '"><div class="cm-psig-hd"><span class="cm-psig-tag">Pairs signal</span><span class="cm-psig-meta">60d hedge-ratio spread \u00b7 ' + zTxt + '</span></div>' +
    '<div class="cm-psig-dir">' + dirText + '</div><div class="cm-psig-note">' + note + '</div></div>';
}

function _cmSetTab(tab) {
  if (tab === _cmActiveTab) return;
  _cmActiveTab = tab;
  const crossBtn = document.getElementById('cm-tab-cross'), matrixBtn = document.getElementById('cm-tab-matrix');
  crossBtn?.classList.toggle('active', tab === 'cross');
  matrixBtn?.classList.toggle('active', tab === 'matrix');
  crossBtn?.setAttribute('aria-selected', tab === 'cross' ? 'true' : 'false');
  matrixBtn?.setAttribute('aria-selected', tab === 'matrix' ? 'true' : 'false');
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }
  if (tab === 'cross') _cmRenderCrossAssetBody(_cmCurrentObj);
  else _cmRenderMatrixBody();
}
function _cmSetWindow(w) {
  if (w === _cmWindow) return;
  _cmWindow = w;
  document.querySelectorAll('.cm-win-btn').forEach(btn => btn.classList.toggle('active', +btn.dataset.w === w));
  if (_cmActiveTab === 'cross') _cmRenderCrossAssetBody(_cmCurrentObj);
  else _cmRenderMatrixBody();
}
window._cmSetTab = _cmSetTab;
window._cmSetWindow = _cmSetWindow;

function _cmRenderCrossAssetBody(corrObj) {
  const tabBody = document.getElementById('cm-tab-body');
  if (!tabBody) return;
  const { a, b, corr30, corr, corr90, norm, z_score, std, n30, n, n90, history, hist_dates } = corrObj;
  const absZ = z_score != null ? Math.abs(z_score) : null;
  let sigCls = '', sigTag = '', sigTxt = '';
  if (absZ != null) {
    const zStr = (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3';
    if (absZ >= 2.5) { sigCls = 'warn'; sigTag = 'Break'; sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002Sharp deviation from historical norm — potential regime change or structural dislocation.'; }
    else if (absZ >= 1.5) { sigCls = 'warn'; sigTag = 'Stretched'; sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002Relationship under stress vs norm. Monitor for mean reversion or confirmed break.'; }
    else { sigCls = 'ok'; sigTag = 'Normal'; sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002Tracking within historical norm.'; }
  }
  let trendHtml = '\u2014';
  if (corr30 != null && corr90 != null) {
    const drift = corr30 - corr90;
    const cls = Math.abs(drift) < 0.03 ? 'stable' : drift > 0 ? 'rising' : 'falling';
    const arrow = cls === 'rising' ? '\u2191' : cls === 'falling' ? '\u2193' : '\u2192';
    trendHtml = '<span class="cm-trend-' + cls + '">' + arrow + ' ' + (cls === 'rising' ? 'Rising' : cls === 'falling' ? 'Falling' : 'Stable') + '</span>&thinsp;(30d ' + _cmFmt(corr30) + ' vs 90d ' + _cmFmt(corr90) + ')';
  }
  const normDelta = corr30 != null && norm != null ? corr30 - norm : null;
  const hist = Array.isArray(history) ? history : [];
  const dates = Array.isArray(hist_dates) ? hist_dates : [];

  // Regime label — qualitative description of the SELECTED window's correlation
  // (30d/60d/90d, via the window buttons) rather than always 30d, so the
  // buttons actually change what "Regime" reports.
  const selCorr = _cmWindow === 30 ? corr30 : _cmWindow === 90 ? corr90 : corr;
  let regimeLabel = '\u2014', regimeCls = '';
  if (selCorr != null) {
    const v = selCorr;
    if      (v >=  0.70) { regimeLabel = 'Strong positive'; regimeCls = 'up'; }
    else if (v >=  0.40) { regimeLabel = 'Moderate positive'; regimeCls = 'up'; }
    else if (v >=  0.10) { regimeLabel = 'Weak positive'; regimeCls = ''; }
    else if (v >  -0.10) { regimeLabel = 'Decorrelated'; regimeCls = 'flat'; }
    else if (v >  -0.40) { regimeLabel = 'Weak inverse'; regimeCls = ''; }
    else if (v >  -0.70) { regimeLabel = 'Moderate inverse'; regimeCls = 'down'; }
    else                  { regimeLabel = 'Strong inverse'; regimeCls = 'down'; }
  }

  // 252d range from history array
  let rangeHtml = '\u2014';
  if (hist.length > 0) {
    const hi = Math.max(...hist), lo = Math.min(...hist);
    const hiCls = _cmCls(hi), loCls = _cmCls(lo);
    rangeHtml = '<span class="' + hiCls + '">' + _cmFmt(hi) + '</span>'
      + '<span style="color:var(--text2);margin:0 4px;">\u00b7</span>'
      + '<span class="' + loCls + '">' + _cmFmt(lo) + '</span>';
  }
  let dateRangeLabel = '';
  if (dates.length >= 2) dateRangeLabel = ' \u00b7 ' + _cmFmtDate(dates[0]) + ' \u2013 ' + _cmFmtDate(dates[dates.length - 1]);
  const psigHtml = _cmPairsSignalHtml(corrObj);

  // Related correlations — other cached pairs sharing an instrument with this one (a or b).
  // Cross-asset confluence check: a Bloomberg CORR matrix reduced to "what else moves with this pair right now".
  const _cache = Array.isArray(window._corrDataCache) ? window._corrDataCache : [];
  const related = _cache
    .filter(c => c && c !== corrObj && (c.a === a || c.b === a || c.a === b || c.b === b))
    .map(c => ({ label: c.a + ' vs ' + c.b, val: c.corr ?? c.corr30 ?? c.corr90 ?? null }))
    .filter(r => r.val != null)
    .sort((p, q) => Math.abs(q.val) - Math.abs(p.val))
    .slice(0, 4);
  const relatedHtml = related.length
    ? '<div class="cm-related-title">Related Correlations \u00b7 60d</div>' +
      related.map(r => '<div class="cm-related-row"><span class="cm-related-key">' + r.label + '</span><span class="cm-related-val ' + _cmCls(r.val) + '">' + _cmFmt(r.val) + '</span></div>').join('')
    : '';

  // Metric strip cell class picks up win-active when it matches the shared window selector,
  // so the 30/60/90 buttons visibly tie back to which cell is "live" for Regime/Trend below.
  const winCls = (w) => w === _cmWindow ? ' win-active' : '';

  tabBody.innerHTML =
    '<div class="cm-pair-hd">' +
      '<div id="cm-pair-title">' + a + ' <span style="color:var(--text2);font-weight:400">vs</span> ' + b + '</div>' +
      '<div id="cm-pair-sub">Rolling Pearson \u00b7 252-day history' + dateRangeLabel + '</div>' +
    '</div>' +
    '<div id="cm-strip">' +
      '<div class="cm-metric' + winCls(30) + '"><div class="cm-m-lbl">30d</div><div class="cm-m-val ' + _cmCls(corr30) + '">' + _cmFmt(corr30) + '</div><div class="cm-m-sub">' + (n30 != null ? n30 + ' sess.' : '\u2014') + '</div></div>' +
      '<div class="cm-metric' + winCls(60) + '"><div class="cm-m-lbl">60d</div><div class="cm-m-val ' + _cmCls(corr) + '">' + _cmFmt(corr) + '</div><div class="cm-m-sub">' + (n != null ? n + ' sess.' : '\u2014') + '</div></div>' +
      '<div class="cm-metric' + winCls(90) + '"><div class="cm-m-lbl">90d</div><div class="cm-m-val ' + _cmCls(corr90) + '">' + _cmFmt(corr90) + '</div><div class="cm-m-sub">' + (n90 != null ? n90 + ' sess.' : '\u2014') + '</div></div>' +
      '<div class="cm-metric"><div class="cm-m-lbl">Norm</div><div class="cm-m-val ' + _cmCls(norm) + '">' + _cmFmt(norm) + '</div><div class="cm-m-sub">252d avg</div></div>' +
      '<div class="cm-metric"><div class="cm-m-lbl">Z-Score</div><div class="cm-m-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</div><div class="cm-m-sub">30d vs norm</div></div>' +
    '</div>' +
    '<div id="cm-chart-wrap"><div id="cm-lwc-container"></div><div id="cm-tooltip"></div></div>' +
    '<div id="cm-legend">' +
      '<div class="cm-leg-item"><div class="cm-leg-swatch solid-blue"></div>30d</div>' +
      '<div class="cm-leg-item"><div class="cm-leg-swatch dash-white"></div>252d norm (' + _cmFmt(norm) + ')</div>' +
      (std != null ? '<div class="cm-leg-item"><div class="cm-leg-swatch dash-amber"></div>\u00b11.5\u03c3</div><div class="cm-leg-item"><div class="cm-leg-swatch dash-red"></div>\u00b12.5\u03c3</div>' : '') +
    '</div>' +
    (sigTxt ? '<div class="cm-signal ' + sigCls + '"><span class="cm-signal-tag">' + sigTag + '</span><span class="cm-signal-body">' + sigTxt + '</span></div>' : '') +
    psigHtml +
    '<div class="cm-regime-row"><span class="cm-regime-key">Regime</span><span class="cm-regime-val ' + regimeCls + '">' + regimeLabel + ' &thinsp;· ' + _cmFmt(selCorr) + '</span></div>' +
    '<div class="cm-regime-row"><span class="cm-regime-key">Trend</span><span class="cm-regime-val">' + trendHtml + '</span></div>' +
    '<div class="cm-regime-row"><span class="cm-regime-key">30d vs norm</span><span class="cm-regime-val ' + (normDelta != null ? _cmZcls(z_score) : '') + '">' + (normDelta != null ? _cmFmt(normDelta) : '\u2014') + '</span></div>' +
    '<div class="cm-regime-row"><span class="cm-regime-key">Z-score</span><span class="cm-regime-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</span></div>' +
    '<div class="cm-regime-row"><span class="cm-regime-key">252d range</span><span class="cm-regime-val">' + rangeHtml + '</span></div>' +
    relatedHtml;

  const container = document.getElementById('cm-lwc-container');
  if (window.LightweightCharts) {
    requestAnimationFrame(() => _cmDrawChart(container, hist, dates, norm, std));
  } else {
    const t0 = Date.now();
    const poll = setInterval(() => {
      if (window.LightweightCharts || Date.now() - t0 > 8000) { clearInterval(poll); if (window.LightweightCharts) _cmDrawChart(container, hist, dates, norm, std); }
    }, 120);
  }
}

// ── Modal shell — title is always "Correlations" (no longer per-pair), with
// Cross asset / Matrix tabs and a shared 30d/60d/90d window selector beneath
// the tab bar. Tab content renders into #cm-tab-body via _cmRenderCrossAssetBody()
// / _cmRenderMatrixBody(); this function only builds the static chrome once per open.
function openCorrModal(corrObj) {
  _cmCurrentObj = corrObj;
  _cmActiveTab = 'cross'; // always open on Cross asset, regardless of which tab was active last time

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'Correlations');

  bd.innerHTML =
    '<div id="cm-modal">' +
      '<div id="cm-hd">' +
        '<div id="cm-title">Correlations</div>' +
        '<button id="cm-close" onclick="closeCorrModal()" aria-label="Close">\u00d7</button>' +
      '</div>' +
      '<div id="cm-tabs" role="tablist" aria-label="Correlation view">' +
        '<button id="cm-tab-cross" class="cm-tab active" role="tab" aria-selected="true" onclick="_cmSetTab(\'cross\')">Cross asset</button>' +
        '<button id="cm-tab-matrix" class="cm-tab" role="tab" aria-selected="false" onclick="_cmSetTab(\'matrix\')">Matrix</button>' +
      '</div>' +
      '<div id="cm-window-btns" role="group" aria-label="Correlation window">' +
        [30, 60, 90].map(w => '<button class="cm-win-btn' + (w === _cmWindow ? ' active' : '') + '" data-w="' + w + '" onclick="_cmSetWindow(' + w + ')">' + w + 'd</button>').join('') +
      '</div>' +
      '<div id="cm-body"><div id="cm-tab-body"></div></div>' +
    '</div>';

  document.body.appendChild(bd);
  requestAnimationFrame(()=>requestAnimationFrame(()=>{ bd.scrollIntoView({behavior:'smooth',block:'start'}); }));
  bd.addEventListener('click', e => { if (e.target === bd) closeCorrModal(); });
  document.addEventListener('keydown', _cmKeydown);
  _cmRenderCrossAssetBody(corrObj);
}
function _cmKeydown(e) { if (e.key === 'Escape') closeCorrModal(); }
function closeCorrModal() {
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }
  const bd = document.getElementById('cm-bd');
  if (bd) bd.remove();
  const container = document.getElementById('cm-lwc-container');
  if (container?._cmResize) window.removeEventListener('resize', container._cmResize);
  if (container?._cmRo) container._cmRo.disconnect();
  document.removeEventListener('keydown', _cmKeydown);
}
window.openCorrModal  = openCorrModal;
window.closeCorrModal = closeCorrModal;
