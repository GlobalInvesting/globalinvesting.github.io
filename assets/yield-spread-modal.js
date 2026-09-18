(function () {
  if (document.getElementById('ysm-css')) return;
  const s = document.createElement('style');
  s.id = 'ysm-css';
  s.textContent = `
#ysm-bd {
  display:block!important;
  position:absolute!important;
  top:0!important; bottom:0!important;
  left:50%!important; right:0!important;
  overflow-y:auto!important;
  z-index:500!important;
  background:var(--bg)!important;
  border-left:1px solid var(--border2)!important;
  scrollbar-width:thin;
  scrollbar-color:var(--border2) transparent;
}
#ysm-bd::-webkit-scrollbar { width:3px; }
#ysm-bd::-webkit-scrollbar-thumb { background:var(--border2); border-radius:2px; }
#ysm-modal {
  width:100%!important;max-width:none!important;height:auto!important;min-height:100%!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}
#ysm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 14px 6px;border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg2);
}
#ysm-title { font-size:12px;font-weight:600;letter-spacing:-.01em;color:var(--text); }
#ysm-sub   { font-size:9px;color:var(--text2);margin-top:1px;font-family:var(--font-mono);letter-spacing:.02em; }
#ysm-close { background:none;border:none;color:var(--text2);font-size:16px;cursor:pointer;padding:3px 6px;border-radius:4px;line-height:1;transition:color .1s,background .1s; }
#ysm-close:hover { color:var(--text);background:var(--bg3); }
#ysm-strip { display:flex;border-bottom:1px solid var(--border2);flex-shrink:0;overflow-x:auto;background:var(--bg); }
.ysm-metric { flex:1;min-width:70px;padding:5px 10px;border-right:1px solid var(--border2);background:var(--bg);text-align:center; }
.ysm-metric:last-child { border-right:none; }
.ysm-m-lbl { font-size:8px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;font-family:var(--font-mono); }
.ysm-m-val { font-size:13px;font-weight:600;font-family:var(--font-mono);color:var(--text); }
.ysm-loading, .ysm-empty { padding:10px 14px;font-size:10px;color:var(--text2);font-family:var(--font-mono); }
#ysm-chart-wrap { height:280px;flex-shrink:0;position:relative;padding:8px 14px 4px;display:flex;flex-direction:column;background:var(--bg); }
#ysm-legend { display:flex;gap:14px;margin-bottom:8px;flex-shrink:0;flex-wrap:wrap;align-items:center; }
.ysm-leg-item { display:flex;align-items:center;gap:5px;font-size:9px;color:var(--text2);font-family:var(--font-mono); }
.ysm-leg-dot  { width:14px;height:2px;border-radius:1px;flex-shrink:0; }
.ysm-leg-val  { color:var(--text);font-weight:600; }
#ysm-canvas-wrap { flex:1;position:relative;min-height:190px; }
.ysm-tooltip {
  position:absolute;pointer-events:none;display:none;z-index:5;
  background:var(--bg2);border:1px solid var(--border2);border-radius:4px;
  padding:6px 8px;font-size:9.5px;font-family:var(--font-mono);color:var(--text);
  white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.3);
}
.ysm-tt-row { display:flex;align-items:center;gap:5px;padding:1px 0; }
.ysm-tt-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.ysm-tt-spread { margin-top:3px;padding-top:3px;border-top:1px solid var(--border2);color:var(--text2); }
`;
  document.head.appendChild(s);
})();

let _ysmChart = null;
let _ysmResizeFn = null;
let _ysmRo = null;

function _ysmEscHandler(e) { if (e.key === 'Escape') closeYieldSpreadModal(); }

function _ysmFlagCode(ccy) {
  const info = (typeof G10_RATE_CCYS !== 'undefined') ? G10_RATE_CCYS.find(c => c.ccy === ccy) : null;
  return info ? info.code : ccy.slice(0, 2).toLowerCase();
}

function _ysmFmtBp(v) {
  if (v == null || isNaN(v)) return '\u2014';
  return (v > 0 ? '+' : '') + Math.round(v) + ' bp';
}

function _ysmFmtPct(v) {
  return v == null || isNaN(v) ? '\u2014' : v.toFixed(2) + '%';
}

// Same date-aligned spread computation the Spread & Momentum table itself uses
// (renderMomentumScreener, dashboard.js) — kept as the single source of truth for
// what "the spread" means, so this modal's stat strip never drifts from the table
// that opens it.
function _ysmLegStats(hist) {
  if (!Array.isArray(hist) || !hist.length) return null;
  const sorted = hist
    .filter(r => r && r.date && r.value != null)
    .map(r => ({ date: r.date, value: parseFloat(r.value) }))
    .sort((a, b) => a.date < b.date ? -1 : 1);
  const n = sorted.length;
  if (n < 2) return null;
  const now = sorted[n - 1].value;
  const d5 = n >= 6 ? (now - sorted[n - 6].value) * 100 : null;
  const d20 = n >= 21 ? (now - sorted[n - 21].value) * 100 : null;
  return { sorted, now, d5, d20 };
}

async function openYieldSpreadModal(baseCcy, rowCcy) {
  closeYieldSpreadModal();
  if (!baseCcy || !rowCcy) return;

  const baseCode = _ysmFlagCode(baseCcy);
  const rowCode = _ysmFlagCode(rowCcy);

  const bd = document.createElement('div');
  bd.id = 'ysm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', rowCcy + ' vs ' + baseCcy + ' 2Y Yield Spread');
  bd.innerHTML = `
<div id="ysm-modal">
  <div id="ysm-hd">
    <div>
      <div id="ysm-title"><span class="fi fi-${rowCode}" style="margin-right:5px;border-radius:2px;font-size:13px;vertical-align:middle;"></span>${rowCcy} <span style="color:var(--text2);font-weight:500;">vs</span> <span class="fi fi-${baseCode}" style="margin:0 5px 0 7px;border-radius:2px;font-size:13px;vertical-align:middle;"></span>${baseCcy} <span style="color:var(--text2);font-weight:500;">\u00b7 2Y Yield</span></div>
      <div id="ysm-sub">bond2y-data/ \u00b7 daily close \u00b7 accumulated history</div>
    </div>
    <button id="ysm-close" onclick="closeYieldSpreadModal()" aria-label="Close">\u00d7</button>
  </div>
  <div id="ysm-strip"><div class="ysm-loading">Loading 2Y yield history\u2026</div></div>
  <div id="ysm-chart-wrap">
    <div id="ysm-legend"></div>
    <div id="ysm-canvas-wrap"></div>
  </div>
</div>`;

  (document.getElementById('main') || document.body).appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeYieldSpreadModal(); });
  document.addEventListener('keydown', _ysmEscHandler);
  requestAnimationFrame(() => requestAnimationFrame(() => { bd.scrollIntoView({ behavior: 'smooth', block: 'start' }); }));

  bd._baseCcy = baseCcy;
  bd._rowCcy = rowCcy;

  let baseHist = null, rowHist = null;
  try {
    [baseHist, rowHist] = await Promise.all([
      fetch(`./bond2y-data/${baseCcy}.json`, { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`./bond2y-data/${rowCcy}.json`, { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
  } catch (_) { /* handled below via null checks */ }

  if (!document.getElementById('ysm-bd') || document.getElementById('ysm-bd')._rowCcy !== rowCcy) return; // closed/replaced while loading

  const baseStats = _ysmLegStats(baseHist);
  const rowStats = _ysmLegStats(rowHist);
  const spreadStats = (typeof _bond2ySpreadStats === 'function')
    ? _bond2ySpreadStats(Array.isArray(baseHist) ? baseHist : [], Array.isArray(rowHist) ? rowHist : [])
    : null;

  if (!baseStats || !rowStats) {
    const strip = document.getElementById('ysm-strip');
    if (strip) strip.innerHTML = `<div class="ysm-empty">Not enough accumulated 2Y history yet for ${!baseStats ? baseCcy : rowCcy}.</div>`;
    return;
  }

  _ysmRenderStrip(baseCcy, rowCcy, baseStats, rowStats, spreadStats);
  _ysmRenderChart(baseCcy, rowCcy, baseStats.sorted, rowStats.sorted);
}

function _ysmRenderStrip(baseCcy, rowCcy, baseStats, rowStats, spreadStats) {
  const strip = document.getElementById('ysm-strip');
  if (!strip) return;
  const legMetric = (label, val, d5, d20) => `
    <div class="ysm-metric">
      <div class="ysm-m-lbl">${label}</div>
      <div class="ysm-m-val">${_ysmFmtPct(val)}</div>
    </div>
    <div class="ysm-metric">
      <div class="ysm-m-lbl">${label} 5D</div>
      <div class="ysm-m-val" style="color:${d5 == null ? '' : d5 > 0 ? 'var(--up)' : d5 < 0 ? 'var(--down)' : ''}">${_ysmFmtBp(d5)}</div>
    </div>`;
  const spreadNow = spreadStats ? spreadStats.now : null;
  const spread5d = spreadStats ? spreadStats.d5 : null;
  const spread20d = spreadStats ? spreadStats.d20 : null;
  strip.innerHTML =
    legMetric(baseCcy, baseStats.now, baseStats.d5) +
    legMetric(rowCcy, rowStats.now, rowStats.d5) +
    `<div class="ysm-metric">
      <div class="ysm-m-lbl" title="Current spread: ${rowCcy} 2Y minus ${baseCcy} 2Y">Spread</div>
      <div class="ysm-m-val" style="color:${spreadNow == null ? '' : spreadNow > 0 ? 'var(--up)' : spreadNow < 0 ? 'var(--down)' : ''}">${_ysmFmtBp(spreadNow)}</div>
    </div>
    <div class="ysm-metric">
      <div class="ysm-m-lbl" title="20D change in the spread">Spread 20D</div>
      <div class="ysm-m-val" style="color:${spread20d == null ? '' : spread20d > 0 ? 'var(--up)' : spread20d < 0 ? 'var(--down)' : ''}">${_ysmFmtBp(spread20d)}</div>
    </div>`;
}

function _ysmDefaultWindow(chart, times) {
  chart.timeScale().fitContent();
  if (!times || times.length < 2) return;
  const first = times[0], last = times[times.length - 1];
  if (!first || !last) return;
  const lastD = new Date(last + 'T00:00:00Z');
  if (isNaN(lastD.getTime())) return;
  const cutoff = new Date(lastD);
  cutoff.setUTCFullYear(cutoff.getUTCFullYear() - 1);
  const cutoffStr = cutoff.toISOString().slice(0, 10);
  if (cutoffStr <= first) return;
  try { chart.timeScale().setVisibleRange({ from: cutoffStr, to: last }); } catch (_) {}
}

function _ysmRenderChart(baseCcy, rowCcy, baseSeries, rowSeries) {
  const container = document.getElementById('ysm-canvas-wrap');
  const legendEl = document.getElementById('ysm-legend');
  if (!container || typeof window.LightweightCharts === 'undefined') return;
  const LWC = window.LightweightCharts;

  const cs = getComputedStyle(document.documentElement);
  const bg = cs.getPropertyValue('--bg').trim() || '#131722';
  const text2 = cs.getPropertyValue('--text2').trim() || '#9096a0';
  const border = cs.getPropertyValue('--border').trim() || '#2e2e2e';
  const monoF = cs.getPropertyValue('--font-mono').trim() || "'JetBrains Mono','Courier New',monospace";
  // Same 2-slot bond2y comparison palette used elsewhere in the codebase, so
  // this modal's colors read as the same "2Y yield" data family as any other
  // 2Y-yield visual on the site — never the same color for both legs.
  const baseColor = '#42a5f5';
  const rowColor = (typeof _themeColor === 'function') ? _themeColor('--up') : '#26a69a';

  const W = container.offsetWidth || 600, H = container.offsetHeight || 190;
  const chart = LWC.createChart(container, {
    width: W, height: H,
    layout: { background: { type: 'solid', color: bg }, textColor: text2, fontFamily: monoF, fontSize: 10, attributionLogo: false },
    grid: { vertLines: { color: border + '28' }, horzLines: { color: border + '28' } },
    crosshair: {
      mode: LWC.CrosshairMode ? LWC.CrosshairMode.Normal : 1,
      vertLine: { color: text2 + '55', style: 2, labelVisible: false },
      horzLine: { color: text2 + '33', style: 2, labelVisible: true },
    },
    rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderVisible: false, lockVisibleTimeRangeOnResize: true, fixRightEdge: true },
    handleScroll: { mouseWheel: true, pressedMouseMove: true },
    handleScale: { mouseWheel: true, pinch: true },
    localization: { priceFormatter: v => v != null ? v.toFixed(2) + '%' : '\u2014' },
  });

  const baseS = chart.addSeries(LWC.LineSeries, {
    color: baseColor, lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerRadius: 4,
    priceFormat: { type: 'custom', formatter: v => v.toFixed(2) + '%' },
  });
  const rowS = chart.addSeries(LWC.LineSeries, {
    color: rowColor, lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerRadius: 4,
    priceFormat: { type: 'custom', formatter: v => v.toFixed(2) + '%' },
  });

  const baseData = baseSeries.map(p => ({ time: p.date, value: p.value }));
  const rowData = rowSeries.map(p => ({ time: p.date, value: p.value }));
  baseS.setData(baseData);
  rowS.setData(rowData);

  const allDates = [...new Set([...baseData.map(d => d.time), ...rowData.map(d => d.time)])].sort();
  _ysmDefaultWindow(chart, allDates);

  const baseLast = baseData[baseData.length - 1]?.value;
  const rowLast = rowData[rowData.length - 1]?.value;
  legendEl.innerHTML = `
    <div class="ysm-leg-item"><div class="ysm-leg-dot" style="background:${baseColor};"></div>${baseCcy} 2Y <span class="ysm-leg-val">${_ysmFmtPct(baseLast)}</span></div>
    <div class="ysm-leg-item"><div class="ysm-leg-dot" style="background:${rowColor};"></div>${rowCcy} 2Y <span class="ysm-leg-val">${_ysmFmtPct(rowLast)}</span></div>
  `;

  const tip = document.createElement('div');
  tip.className = 'ysm-tooltip';
  container.style.position = 'relative';
  container.appendChild(tip);
  chart.subscribeCrosshairMove(param => {
    if (!param?.point || !param.time) { tip.style.display = 'none'; return; }
    const bV = param.seriesData.get(baseS)?.value;
    const rV = param.seriesData.get(rowS)?.value;
    if (bV == null && rV == null) { tip.style.display = 'none'; return; }
    const spread = (bV != null && rV != null) ? (rV - bV) * 100 : null;
    tip.innerHTML = `
      <div>${param.time}</div>
      <div class="ysm-tt-row"><span class="ysm-tt-dot" style="background:${baseColor};"></span>${baseCcy} ${_ysmFmtPct(bV)}</div>
      <div class="ysm-tt-row"><span class="ysm-tt-dot" style="background:${rowColor};"></span>${rowCcy} ${_ysmFmtPct(rV)}</div>
      ${spread != null ? `<div class="ysm-tt-spread">Spread ${_ysmFmtBp(spread)}</div>` : ''}
    `;
    tip.style.display = 'block';
    const cW = container.offsetWidth, cx = param.point.x, cy = param.point.y, tw = tip.offsetWidth || 140, th = tip.offsetHeight || 60;
    const tx = (cx + 12 + tw <= cW - 4) ? cx + 12 : cx - 12 - tw;
    const ty = (cy - th - 12 >= 4) ? cy - th - 12 : cy + 12;
    tip.style.left = Math.max(0, tx) + 'px';
    tip.style.top = Math.max(0, ty) + 'px';
  });

  _ysmChart = chart;
  const apply = () => {
    requestAnimationFrame(() => {
      const rect = container.getBoundingClientRect();
      const h = Math.round(rect.height) || 190;
      const w = Math.round(rect.width) || 600;
      if (chart && w > 0 && h > 10) chart.applyOptions({ width: w, height: h });
    });
  };
  if (window.ResizeObserver) { _ysmRo = new ResizeObserver(() => apply()); _ysmRo.observe(container); }
  window.addEventListener('resize', apply);
  _ysmResizeFn = apply;
  setTimeout(apply, 60); setTimeout(apply, 200);
}

function closeYieldSpreadModal() {
  if (_ysmChart) { try { _ysmChart.remove(); } catch (_) {} _ysmChart = null; }
  if (_ysmRo) { _ysmRo.disconnect(); _ysmRo = null; }
  if (_ysmResizeFn) { window.removeEventListener('resize', _ysmResizeFn); _ysmResizeFn = null; }
  const bd = document.getElementById('ysm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _ysmEscHandler);
}

window.openYieldSpreadModal = openYieldSpreadModal;
window.closeYieldSpreadModal = closeYieldSpreadModal;

window.addEventListener('gi-theme-change', function () {
  const bd = document.getElementById('ysm-bd');
  if (!bd || !bd._baseCcy || !bd._rowCcy) return;
  openYieldSpreadModal(bd._baseCcy, bd._rowCcy);
});
