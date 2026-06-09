// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v2.1  — inline-panel edition
// Fluid layout, terminal CSS variables throughout.
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

function openCorrModal(corrObj) {
  closeCorrModal();
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

  // Regime label — qualitative description of current 30d correlation
  let regimeLabel = '\u2014', regimeCls = '';
  if (corr30 != null) {
    const v = corr30;
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

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'Correlation: ' + a + ' vs ' + b);

  bd.innerHTML =
    '<div id="cm-modal">' +
      '<div id="cm-hd">' +
        '<div><div id="cm-title">' + a + ' <span style="color:var(--text2);font-weight:400">vs</span> ' + b + '</div>' +
        '<div id="cm-sub">Rolling Pearson \u00b7 252-day history</div></div>' +
        '<button id="cm-close" onclick="closeCorrModal()" aria-label="Close">\u00d7</button>' +
      '</div>' +
      '<div id="cm-strip">' +
        '<div class="cm-metric"><div class="cm-m-lbl">30d</div><div class="cm-m-val ' + _cmCls(corr30) + '">' + _cmFmt(corr30) + '</div><div class="cm-m-sub">' + (n30 != null ? n30 + ' sess.' : '\u2014') + '</div></div>' +
        '<div class="cm-metric"><div class="cm-m-lbl">60d</div><div class="cm-m-val ' + _cmCls(corr) + '">' + _cmFmt(corr) + '</div><div class="cm-m-sub">' + (n != null ? n + ' sess.' : '\u2014') + '</div></div>' +
        '<div class="cm-metric"><div class="cm-m-lbl">90d</div><div class="cm-m-val ' + _cmCls(corr90) + '">' + _cmFmt(corr90) + '</div><div class="cm-m-sub">' + (n90 != null ? n90 + ' sess.' : '\u2014') + '</div></div>' +
        '<div class="cm-metric"><div class="cm-m-lbl">Norm</div><div class="cm-m-val ' + _cmCls(norm) + '">' + _cmFmt(norm) + '</div><div class="cm-m-sub">252d avg</div></div>' +
        '<div class="cm-metric"><div class="cm-m-lbl">Z-Score</div><div class="cm-m-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</div><div class="cm-m-sub">30d vs norm</div></div>' +
      '</div>' +
      '<div id="cm-body">' +
        '<div id="cm-chart-wrap"><div id="cm-lwc-container"></div><div id="cm-tooltip"></div></div>' +
        '<div id="cm-legend">' +
          '<div class="cm-leg-item"><div class="cm-leg-swatch solid-blue"></div>30d</div>' +
          '<div class="cm-leg-item"><div class="cm-leg-swatch dash-white"></div>252d norm (' + _cmFmt(norm) + ')</div>' +
          (std != null ? '<div class="cm-leg-item"><div class="cm-leg-swatch dash-amber"></div>\u00b11.5\u03c3</div><div class="cm-leg-item"><div class="cm-leg-swatch dash-red"></div>\u00b12.5\u03c3</div>' : '') +
        '</div>' +
        '<div class="cm-regime-row"><span class="cm-regime-key">Regime</span><span class="cm-regime-val ' + regimeCls + '">' + regimeLabel + ' &thinsp;· ' + _cmFmt(corr30) + '</span></div>' +
        '<div class="cm-regime-row"><span class="cm-regime-key">Trend</span><span class="cm-regime-val">' + trendHtml + '</span></div>' +
        '<div class="cm-regime-row"><span class="cm-regime-key">30d vs norm</span><span class="cm-regime-val ' + (normDelta != null ? _cmZcls(z_score) : '') + '">' + (normDelta != null ? _cmFmt(normDelta) : '\u2014') + '</span></div>' +
        '<div class="cm-regime-row"><span class="cm-regime-key">Z-score</span><span class="cm-regime-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</span></div>' +
        '<div class="cm-regime-row"><span class="cm-regime-key">252d range</span><span class="cm-regime-val">' + rangeHtml + '</span></div>' +
        (sigTxt ? '<div class="cm-signal ' + sigCls + '"><span class="cm-signal-tag">' + sigTag + '</span><span class="cm-signal-body">' + sigTxt + '</span></div>' : '') +
      '</div>' +
    '</div>';

  document.body.appendChild(bd);
  requestAnimationFrame(()=>requestAnimationFrame(()=>{ bd.scrollIntoView({behavior:'smooth',block:'start'}); }));
  bd.addEventListener('click', e => { if (e.target === bd) closeCorrModal(); });
  document.addEventListener('keydown', _cmKeydown);
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
