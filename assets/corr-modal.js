// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v2.3
// File: assets/corr-modal.js
//
// LightweightCharts v5 — Bloomberg-style rolling 30d correlation chart.
// Shows the 252-day rolling-30d history with real calendar dates on X-axis.
//   • Line series with crosshair tooltip (value + calendar date)
//   • 252d norm dashed baseline + ±1.5σ (amber) / ±2.5σ (red) horizontal bands
//   • Vertical price-line markers at 30d / 60d / 90d lookback positions
//   • Design tokens matched to existing site modals (heatmap-modal, cb-rates-modal)
//
//   openCorrModal(corrObj)  ← called from correlations-tbody onclick
//   closeCorrModal()
//
// corrObj fields (from quotes.json correlations[]):
//   a, b            — instrument labels
//   corr30/corr/corr90  — rolling Pearson at 30d/60d/90d windows
//   norm            — 252d historical mean of rolling-30d windows
//   std             — std dev of those 30d-rolling windows
//   z_score         — (corr30 - norm) / std  [apples-to-apples]
//   history         — array of ~223 rolling-30d values, oldest→newest
//   hist_dates      — ISO date strings aligned 1:1 with history (end date of each window)
//   n30/n/n90       — actual session counts per window
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('cm-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cm-modal-css';
  s.textContent = `
#cm-bd {
  position:fixed;inset:0;z-index:9200;
  background:rgba(0,0,0,.85);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:cm-fi .15s ease;
}
@keyframes cm-fi { from{opacity:0} to{opacity:1} }
@keyframes cm-su { from{transform:translateY(12px);opacity:0} to{transform:none;opacity:1} }

#cm-modal {
  background:#161b22;
  border:1px solid #30363d;
  border-radius:8px;
  width:min(800px,100%);
  max-height:90vh;
  display:flex;flex-direction:column;
  overflow:hidden;
  box-shadow:0 24px 80px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.04);
  animation:cm-su .18s cubic-bezier(.16,1,.3,1);
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:#e6edf3;
  position:relative;
}
#cm-modal::before {
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#1f6feb 0%,#58a6ff 50%,#26a69a 100%);
  border-radius:8px 8px 0 0;z-index:1;
}

#cm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px 12px;
  border-bottom:1px solid #30363d;
  flex-shrink:0;background:#161b22;
}
#cm-title { font-size:14px;font-weight:600;color:#e6edf3;letter-spacing:-.01em; }
#cm-sub   { font-size:10px;color:#6e7681;margin-top:2px;font-family:'IBM Plex Mono',var(--font-mono,monospace);letter-spacing:.02em; }
#cm-close {
  background:none;border:none;color:#6e7681;font-size:18px;
  cursor:pointer;padding:5px 7px;border-radius:5px;line-height:1;
  transition:color .1s,background .1s;
}
#cm-close:hover { color:#e6edf3;background:#21262d; }

#cm-strip {
  display:flex;
  gap:0;background:#0d1117;
  border-bottom:1px solid #30363d;
  flex-shrink:0;overflow-x:auto;
}
.cm-metric {
  flex:1;min-width:90px;padding:9px 16px;
  background:#0d1117;border-right:1px solid #30363d;
}
.cm-metric:last-child { border-right:none; }
.cm-m-lbl { font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:'IBM Plex Mono',var(--font-mono,monospace); }
.cm-m-val {
  font-size:20px;font-weight:600;
  font-family:'IBM Plex Mono',var(--font-mono,monospace);
  line-height:1;color:#e6edf3;
}
.cm-m-val.up   { color:#26a69a; }
.cm-m-val.down { color:#ef5350; }
.cm-m-val.warn { color:#f6941c; }
.cm-m-sub { font-size:9px;color:#6e7681;margin-top:3px;font-family:'IBM Plex Mono',var(--font-mono,monospace); }

#cm-body {
  flex:1;overflow-y:auto;padding:16px 18px 20px;
  min-height:0;display:flex;flex-direction:column;gap:16px;
  background:#0d1117;
  scrollbar-width:thin;scrollbar-color:#444c56 transparent;
}
#cm-body::-webkit-scrollbar { width:4px; }
#cm-body::-webkit-scrollbar-track { background:transparent; }
#cm-body::-webkit-scrollbar-thumb { background:#444c56;border-radius:2px; }

.cm-section-title {
  font-size:9px;color:#6e7681;
  text-transform:uppercase;letter-spacing:.08em;
  margin-bottom:10px;font-family:'IBM Plex Mono',var(--font-mono,monospace);
}

#cm-chart-wrap {
  position:relative;height:220px;
  background:#161b22;
  border:1px solid #30363d;
  border-radius:6px;overflow:hidden;
}
#cm-lwc-container { width:100%;height:100%; }

#cm-tooltip {
  position:absolute;top:8px;left:12px;
  background:#161b22;
  border:1px solid #30363d;
  border-radius:4px;padding:5px 9px;
  font-size:10px;font-family:'IBM Plex Mono',var(--font-mono,monospace);
  color:#e6edf3;pointer-events:none;
  display:none;z-index:10;white-space:nowrap;
}

#cm-legend {
  display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;
}
.cm-leg-item {
  display:flex;align-items:center;gap:5px;
  font-size:9px;color:#6e7681;
  font-family:'IBM Plex Mono',var(--font-mono,monospace);
}
.cm-leg-swatch { width:18px;height:2px;border-radius:1px;flex-shrink:0; }
.cm-leg-swatch.solid-blue { background:#388bfd; }
.cm-leg-swatch.dash-white {
  background:repeating-linear-gradient(90deg,rgba(255,255,255,.5) 0,rgba(255,255,255,.5) 3px,transparent 3px,transparent 6px);
}
.cm-leg-swatch.dash-amber {
  background:repeating-linear-gradient(90deg,rgba(240,148,28,.8) 0,rgba(240,148,28,.8) 3px,transparent 3px,transparent 6px);
}
.cm-leg-swatch.dash-red {
  background:repeating-linear-gradient(90deg,rgba(239,83,80,.8) 0,rgba(239,83,80,.8) 3px,transparent 3px,transparent 6px);
}

.cm-regime-card {
  background:#161b22;
  border:1px solid #30363d;
  border-radius:6px;padding:12px 14px;
  display:flex;flex-direction:column;gap:7px;
}
.cm-regime-row { display:flex;justify-content:space-between;align-items:baseline; }
.cm-regime-key { font-size:10px;color:#8b949e;font-family:'IBM Plex Mono',var(--font-mono,monospace); }
.cm-regime-val {
  font-size:11px;font-weight:600;
  font-family:'IBM Plex Mono',var(--font-mono,monospace);
  color:#e6edf3;
}
.cm-regime-val.up   { color:#26a69a; }
.cm-regime-val.down { color:#ef5350; }
.cm-regime-val.warn { color:#f6941c; }
.cm-regime-val.flat { color:#8b949e; }

.cm-trend-rising  { color:#26a69a; }
.cm-trend-falling { color:#ef5350; }
.cm-trend-stable  { color:#8b949e; }

.cm-signal {
  margin-top:8px;
  display:flex;align-items:baseline;gap:10px;
  padding:6px 10px;
  border-left:2px solid #30363d;
  font-size:10px;line-height:1.5;
  color:#8b949e;
  font-family:'IBM Plex Mono',var(--font-mono,monospace);
}
.cm-signal.warn { border-left-color:#ef5350; }
.cm-signal.ok   { border-left-color:#26a69a; }
.cm-signal-tag {
  font-size:9px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  font-family:'IBM Plex Mono',var(--font-mono,monospace);white-space:nowrap;flex-shrink:0;
}
.cm-signal.warn .cm-signal-tag { color:#ef5350; }
.cm-signal.ok   .cm-signal-tag { color:#26a69a; }
.cm-signal-body { color:#8b949e; }

@media(max-width:600px){
  #cm-modal{border-radius:12px 12px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;max-height:88vh;}
  #cm-bd{align-items:flex-end;padding:0;}
  .cm-metric{min-width:74px;padding:8px 10px;}
  .cm-m-val{font-size:16px;}
  #cm-chart-wrap{height:180px;}
}
`;
  document.head.appendChild(s);
})();

// ── Helpers ──────────────────────────────────────────────────────────────────
let _cmChart = null;

function _cmCls(v) {
  if (v == null) return '';
  return v >= 0.3 ? 'up' : v <= -0.3 ? 'down' : '';
}
// z-score coloring matches dashboard.js badge: >=2.5 = red, >=1.5 = orange/warn, else neutral
function _cmZcls(z) {
  if (z == null) return '';
  const a = Math.abs(z);
  return a >= 2.5 ? 'down' : a >= 1.5 ? 'warn' : '';
}
function _cmFmt(v, d) {
  if (v == null) return '\u2014';
  return (v >= 0 ? '+' : '') + v.toFixed(d ?? 2);
}
// Parse ISO date string "YYYY-MM-DD" to LWC time {year,month,day}
function _cmParseDate(iso) {
  if (!iso || typeof iso !== 'string') return null;
  const p = iso.split('-');
  if (p.length < 3) return null;
  return { year: +p[0], month: +p[1], day: +p[2] };
}
// Format ISO date for tooltip: "Apr 25, 2026"
function _cmFmtDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso + 'T12:00:00Z');
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
  } catch (_) { return iso; }
}

// ── LWC chart ─────────────────────────────────────────────────────────────────
function _cmDrawChart(container, history, histDates, norm, std) {
  if (!container || !window.LightweightCharts || !history || !history.length) return;
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }

  const LWC = window.LightweightCharts;
  const n   = history.length;
  const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(3);

  // Build time-based data points if dates are available, otherwise fall back to integer index
  const hasRealDates = Array.isArray(histDates) && histDates.length === n;
  const mkPt = (i, v) => {
    if (hasRealDates) {
      const t = _cmParseDate(histDates[i]);
      if (t) return { time: t, value: v };
    }
    return { time: i + 1, value: v };
  };

  // For horizontal reference lines (norm/bands/zero), use full date range
  const tFirst = mkPt(0, 0).time;
  const tLast  = mkPt(n - 1, 0).time;
  const hLine  = (v) => [{ time: tFirst, value: v }, { time: tLast, value: v }];

  _cmChart = LWC.createChart(container, {
    autoSize: true,
    layout: {
      background: { type: 'solid', color: '#161b22' },
      textColor: '#6e7681',
      fontSize: 10,
      fontFamily: "'JetBrains Mono','Courier New',monospace",
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: 'rgba(255,255,255,.04)' },
      horzLines: { color: 'rgba(255,255,255,.04)' },
    },
    crosshair: {
      mode: LWC.CrosshairMode.Magnet,
      vertLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: '#2a2e39' },
      horzLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: '#2a2e39' },
    },
    rightPriceScale: {
      borderVisible: false,
      scaleMargins: { top: 0.06, bottom: 0.06 },
    },
    timeScale: {
      borderVisible: false,
      tickMarkFormatter: hasRealDates
        ? (time) => {
            try {
              const d = new Date(Date.UTC(time.year, time.month - 1, time.day));
              return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit', timeZone: 'UTC' });
            } catch (_) { return ''; }
          }
        : undefined,
    },
    handleScroll: false,
    handleScale: false,
  });

  // Zero line
  const zeroSer = _cmChart.addSeries(LWC.LineSeries, {
    color: 'rgba(255,255,255,.12)',
    lineWidth: 1,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
    priceFormat: { type: 'custom', formatter: fmt },
  });
  zeroSer.setData(hLine(0));

  // Norm + sigma bands (drawn before main so main renders on top)
  if (norm != null) {
    const normSer = _cmChart.addSeries(LWC.LineSeries, {
      color: 'rgba(255,255,255,.45)',
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: { type: 'custom', formatter: fmt },
    });
    normSer.setData(hLine(norm));

    if (std != null && std > 0) {
      [[1.5, 'rgba(240,148,28,.55)'], [2.5, 'rgba(239,83,80,.55)']].forEach(([mult, color]) => {
        [Math.min(1, norm + mult * std), Math.max(-1, norm - mult * std)].forEach(v => {
          const ser = _cmChart.addSeries(LWC.LineSeries, {
            color,
            lineWidth: 1,
            lineStyle: 3,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            priceFormat: { type: 'custom', formatter: fmt },
          });
          ser.setData(hLine(v));
        });
      });
    }
  }

  // Main rolling-30d line (added last = renders on top)
  const mainSer = _cmChart.addSeries(LWC.LineSeries, {
    color: '#4f7fff',
    lineWidth: 2,
    lastValueVisible: true,
    priceLineVisible: false,
    crosshairMarkerRadius: 4,
    crosshairMarkerBackgroundColor: '#4f7fff',
    crosshairMarkerBorderColor: '#161b22',
    priceFormat: { type: 'custom', formatter: fmt },
  });
  mainSer.setData(history.map((v, i) => mkPt(i, v)));

  _cmChart.timeScale().fitContent();

  // Vertical day markers at 30d / 60d / 90d from right — using createPriceLine on mainSer
  // approach: add very thin price lines at the corr value at each lookback point, labeled
  // Better approach: draw a separate invisible line series and use it as anchor is complex.
  // Simplest correct approach: overlay using the chart's priceScale range [-1,1]
  // Use a SEPARATE line series per marker with just 2 points on the SAME x-axis date
  // LWC supports single-point series but they need 2 identical points to show a vertical.
  // Actually LWC doesn't support vertical lines natively. Use the existing approach but
  // with real dates so they are vertical (both points share the same time value).
  if (hasRealDates) {
    [30, 60, 90].forEach((days, idx) => {
      const pos = n - days;
      if (pos < 1 || pos >= n - 1) return;
      const tMark = mkPt(pos, 0).time;
      if (!tMark) return;
      const colors = ['rgba(79,127,255,.45)', 'rgba(79,127,255,.28)', 'rgba(79,127,255,.18)'];
      // Draw a price line (horizontal by default) — not useful for vertical.
      // Instead: add a 2-point LineSeries at the marker date spanning -1 to +1
      // Both points MUST have different time values for LWC not to skip — use adjacent dates
      // So just draw a very narrow series: [date-1day, date] spanning full Y
      // This approximates a vertical without causing the diagonal-line bug
      // The diagonal bug was caused by using integer time (i+1) with different y values,
      // which LWC interpolates — with REAL calendar dates this works correctly.
      const markerSer = _cmChart.addSeries(LWC.LineSeries, {
        color: colors[idx],
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
        priceFormat: { type: 'custom', formatter: () => days + 'd' },
      });
      // Same-date points spanning full Y — LWC will draw them as a vertical segment
      markerSer.setData([
        { time: tMark, value: -1 },
        { time: tMark, value:  1 },
      ]);
    });
  }

  // Crosshair tooltip — shows corr value + calendar date
  const tooltip = document.getElementById('cm-tooltip');
  if (tooltip) {
    _cmChart.subscribeCrosshairMove(param => {
      if (!param || !param.point || !param.seriesData) {
        tooltip.style.display = 'none';
        return;
      }
      const val = param.seriesData.get(mainSer);
      if (val == null) { tooltip.style.display = 'none'; return; }

      let dateLabel = '';
      if (hasRealDates && val.time && val.time.year) {
        const iso = val.time.year + '-'
          + String(val.time.month).padStart(2, '0') + '-'
          + String(val.time.day).padStart(2, '0');
        dateLabel = _cmFmtDate(iso);
      } else if (val.time) {
        const idx = (val.time || 1) - 1;
        const daysAgo = n - 1 - idx;
        dateLabel = daysAgo === 0 ? 'today' : daysAgo + 'd ago';
      }

      tooltip.style.display = 'block';
      tooltip.textContent = fmt(val.value) + (dateLabel ? '  \u00b7  ' + dateLabel : '');
    });
  }
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openCorrModal(corrObj) {
  closeCorrModal();

  const { a, b, corr30, corr, corr90, norm, z_score, std, n30, n, n90, history, hist_dates } = corrObj;
  const absZ = z_score != null ? Math.abs(z_score) : null;

  // Signal row — two tiers (break / stretched / stable), matching badge thresholds
  let sigCls = '', sigTag = '', sigTxt = '';
  if (absZ != null) {
    const zStr = (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3';
    if (absZ >= 2.5) {
      sigCls = 'warn'; sigTag = 'Break';
      sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002The 30d correlation has deviated sharply from its historical norm, signaling a potential regime change or structural dislocation.';
    } else if (absZ >= 1.5) {
      sigCls = 'warn'; sigTag = 'Stretched';
      sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002Relationship under stress vs historical norm. Monitor for mean reversion or a confirmed break above 2.5\u03c3.';
    } else {
      sigCls = 'ok'; sigTag = 'Normal';
      sigTxt = 'Z\u2011score\u00a0' + zStr + '\u2002\u00b7\u2002Tracking within its historical norm.';
    }
  }

  // Trend: 30d vs 90d snapshot
  let trendHtml = '\u2014';
  if (corr30 != null && corr90 != null) {
    const drift = corr30 - corr90;
    const cls = Math.abs(drift) < 0.03 ? 'stable' : drift > 0 ? 'rising' : 'falling';
    const arrow = cls === 'rising' ? '\u2191' : cls === 'falling' ? '\u2193' : '\u2192';
    const label = cls === 'rising' ? 'Rising' : cls === 'falling' ? 'Falling' : 'Stable';
    trendHtml = '<span class="cm-trend-' + cls + '">' + arrow + ' ' + label + '</span>&thinsp;(30d ' + _cmFmt(corr30) + ' vs 90d ' + _cmFmt(corr90) + ')';
  }

  // 30d vs norm delta — apples-to-apples
  const normDelta = corr30 != null && norm != null ? corr30 - norm : null;

  const hist = Array.isArray(history) ? history : [];
  const dates = Array.isArray(hist_dates) ? hist_dates : [];

  // Date range label for section title
  let dateRangeLabel = '';
  if (dates.length >= 2) {
    dateRangeLabel = ' \u00b7 ' + _cmFmtDate(dates[0]) + ' \u2013 ' + _cmFmtDate(dates[dates.length - 1]);
  }

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'Correlation: ' + a + ' vs ' + b);

  bd.innerHTML =
    '<div id="cm-modal">' +
      '<div id="cm-hd">' +
        '<div>' +
          '<div id="cm-title">' + a + ' <span style="color:#6e7681;font-weight:400">vs</span> ' + b + '</div>' +
          '<div id="cm-sub">Rolling Pearson \u00b7 yfinance \u00b7 252-day history \u00b7 z-score = 30d vs 30d-rolling norm</div>' +
        '</div>' +
        '<button id="cm-close" onclick="closeCorrModal()" aria-label="Close">\u00d7</button>' +
      '</div>' +

      '<div id="cm-strip">' +
        '<div class="cm-metric">' +
          '<div class="cm-m-lbl">30d Corr</div>' +
          '<div class="cm-m-val ' + _cmCls(corr30) + '">' + _cmFmt(corr30) + '</div>' +
          '<div class="cm-m-sub">' + (n30 != null ? n30 + ' sessions' : '\u2014') + '</div>' +
        '</div>' +
        '<div class="cm-metric">' +
          '<div class="cm-m-lbl">60d Corr</div>' +
          '<div class="cm-m-val ' + _cmCls(corr) + '">' + _cmFmt(corr) + '</div>' +
          '<div class="cm-m-sub">' + (n != null ? n + ' sessions' : '\u2014') + '</div>' +
        '</div>' +
        '<div class="cm-metric">' +
          '<div class="cm-m-lbl">90d Corr</div>' +
          '<div class="cm-m-val ' + _cmCls(corr90) + '">' + _cmFmt(corr90) + '</div>' +
          '<div class="cm-m-sub">' + (n90 != null ? n90 + ' sessions' : '\u2014') + '</div>' +
        '</div>' +
        '<div class="cm-metric">' +
          '<div class="cm-m-lbl">252d Norm</div>' +
          '<div class="cm-m-val ' + _cmCls(norm) + '">' + _cmFmt(norm) + '</div>' +
          '<div class="cm-m-sub">30d-rolling avg</div>' +
        '</div>' +
        '<div class="cm-metric">' +
          '<div class="cm-m-lbl">Z-Score</div>' +
          '<div class="cm-m-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</div>' +
          '<div class="cm-m-sub">30d vs norm</div>' +
        '</div>' +
      '</div>' +

      '<div id="cm-body">' +

        '<div>' +
          '<div class="cm-section-title">Rolling 30d correlation \u2014 252-day history' + dateRangeLabel + '</div>' +
          '<div id="cm-chart-wrap">' +
            '<div id="cm-lwc-container"></div>' +
            '<div id="cm-tooltip"></div>' +
          '</div>' +
          '<div id="cm-legend">' +
            '<div class="cm-leg-item"><div class="cm-leg-swatch solid-blue"></div>Rolling 30d</div>' +
            '<div class="cm-leg-item"><div class="cm-leg-swatch dash-white"></div>252d norm (' + _cmFmt(norm) + ')</div>' +
            (std != null ? '<div class="cm-leg-item"><div class="cm-leg-swatch dash-amber"></div>\u00b11.5\u03c3</div><div class="cm-leg-item"><div class="cm-leg-swatch dash-red"></div>\u00b12.5\u03c3</div>' : '') +
          '</div>' +
        '</div>' +

        '<div>' +
          '<div class="cm-section-title">Regime assessment</div>' +
          '<div class="cm-regime-card">' +
            '<div class="cm-regime-row">' +
              '<span class="cm-regime-key">Short-term trend</span>' +
              '<span class="cm-regime-val">' + trendHtml + '</span>' +
            '</div>' +
            '<div class="cm-regime-row">' +
              '<span class="cm-regime-key">30d vs 252d norm</span>' +
              '<span class="cm-regime-val ' + (normDelta != null ? _cmCls(normDelta) : '') + '">' + (normDelta != null ? _cmFmt(normDelta) : '\u2014') + '</span>' +
            '</div>' +
            '<div class="cm-regime-row">' +
              '<span class="cm-regime-key">Z-score (std deviations)</span>' +
              '<span class="cm-regime-val ' + _cmZcls(z_score) + '">' + (z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3' : '\u2014') + '</span>' +
            '</div>' +
            '<div class="cm-regime-row">' +
              '<span class="cm-regime-key">Signal threshold</span>' +
              '<span class="cm-regime-val flat">|z| \u2265 1.5 = stretched \u00b7 \u2265 2.5 = break</span>' +
            '</div>' +
            (sigTxt ? '<div class="cm-signal ' + sigCls + '"><span class="cm-signal-tag">' + sigTag + '</span><span class="cm-signal-body">' + sigTxt + '</span></div>' : '') +
          '</div>' +
        '</div>' +

      '</div>' +
    '</div>';

  document.body.appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeCorrModal(); });
  document.addEventListener('keydown', _cmKeydown);
  document.getElementById('cm-close').focus();

  // Draw LWC chart — poll until CDN is ready
  const container = document.getElementById('cm-lwc-container');
  if (window.LightweightCharts) {
    requestAnimationFrame(() => _cmDrawChart(container, hist, dates, norm, std));
  } else {
    const t0 = Date.now();
    const poll = setInterval(() => {
      if (window.LightweightCharts || Date.now() - t0 > 8000) {
        clearInterval(poll);
        if (window.LightweightCharts) _cmDrawChart(container, hist, dates, norm, std);
      }
    }, 120);
  }
}

function _cmKeydown(e) { if (e.key === 'Escape') closeCorrModal(); }

function closeCorrModal() {
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }
  const bd = document.getElementById('cm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _cmKeydown);
}

window.openCorrModal  = openCorrModal;
window.closeCorrModal = closeCorrModal;
