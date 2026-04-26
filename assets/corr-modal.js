// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v2.2
// File: assets/corr-modal.js
//
// LightweightCharts v5 — Bloomberg-style rolling 30d correlation chart.
// Shows the 252-day rolling-30d history with:
//   • Line series with crosshair tooltip ("Nd ago" / "today")
//   • 252d norm dashed baseline + ±1.5σ (amber) / ±2.5σ (red) threshold bands
//   • Vertical dashed markers at 30d / 60d / 90d positions from right
//   • Fixed background #070a10 — eliminates LWC hover-lighten artifact
//
//   openCorrModal(corrObj)  ← called from correlations-tbody onclick
//   closeCorrModal()
//
// corrObj fields (from quotes.json correlations[]):
//   a, b          — instrument labels
//   corr30/corr/corr90  — rolling Pearson at 30d/60d/90d windows
//   norm          — 252d historical mean of rolling-30d windows
//   std           — std dev of those 30d-rolling windows
//   z_score       — (corr30 - norm) / std  [apples-to-apples: 30d vs 30d-rolling norm]
//   history       — array of ~223 rolling-30d values, oldest→newest
//   n30/n/n90     — actual session counts used per window
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('cm-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cm-modal-css';
  s.textContent = `
#cm-bd {
  position:fixed;inset:0;z-index:9200;
  background:rgba(0,0,0,.8);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:cm-fi .15s ease;
}
@keyframes cm-fi { from{opacity:0} to{opacity:1} }
@keyframes cm-su { from{transform:translateY(12px);opacity:0} to{transform:none;opacity:1} }

#cm-modal {
  background:#0e1118;
  border:1px solid rgba(255,255,255,.08);
  border-radius:4px;
  width:min(800px,100%);
  max-height:90vh;
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:cm-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:#d1d4dc;
}

#cm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.06);
  flex-shrink:0;
}
#cm-title { font-size:14px;font-weight:600;letter-spacing:.01em; }
#cm-sub   { font-size:10px;color:#6b7280;margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#cm-close {
  background:none;border:none;color:#6b7280;font-size:20px;
  cursor:pointer;padding:0 4px;line-height:1;flex-shrink:0;
}
#cm-close:hover { color:#d1d4dc; }

#cm-strip {
  display:flex;border-bottom:1px solid rgba(255,255,255,.06);
  flex-shrink:0;overflow-x:auto;
  background:#0b0e15;
}
.cm-metric {
  flex:1;min-width:90px;padding:10px 14px;
  border-right:1px solid rgba(255,255,255,.06);
}
.cm-metric:last-child { border-right:none; }
.cm-m-lbl { font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px; }
.cm-m-val {
  font-size:20px;font-weight:600;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
  line-height:1;color:#d1d4dc;
}
.cm-m-val.up   { color:#26a69a; }
.cm-m-val.down { color:#ef5350; }
.cm-m-val.warn { color:#e07f00; }
.cm-m-sub { font-size:9px;color:#6b7280;margin-top:3px; }

#cm-body {
  flex:1;overflow-y:auto;padding:16px 18px 20px;
  min-height:0;display:flex;flex-direction:column;gap:16px;
}
.cm-section-title {
  font-size:9px;color:#6b7280;
  text-transform:uppercase;letter-spacing:.08em;
  margin-bottom:10px;
}

#cm-chart-wrap {
  position:relative;height:210px;
  background:#070a10;
  border:1px solid rgba(255,255,255,.06);
  border-radius:4px;overflow:hidden;
}
#cm-lwc-container { width:100%;height:100%; }

#cm-tooltip {
  position:absolute;top:8px;left:12px;
  background:rgba(7,10,16,.95);
  border:1px solid rgba(255,255,255,.10);
  border-radius:3px;padding:5px 9px;
  font-size:10px;font-family:var(--font-mono,'JetBrains Mono',monospace);
  color:#d1d4dc;pointer-events:none;
  display:none;z-index:10;white-space:nowrap;
}

#cm-legend {
  display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;
}
.cm-leg-item {
  display:flex;align-items:center;gap:5px;
  font-size:9px;color:#6b7280;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
}
.cm-leg-swatch { width:16px;height:2px;border-radius:1px;flex-shrink:0; }
.cm-leg-swatch.solid-blue { background:#4f7fff; }
.cm-leg-swatch.dash-white {
  background:repeating-linear-gradient(90deg,rgba(255,255,255,.5) 0,rgba(255,255,255,.5) 3px,transparent 3px,transparent 6px);
}
.cm-leg-swatch.dash-amber {
  background:repeating-linear-gradient(90deg,rgba(224,127,0,.8) 0,rgba(224,127,0,.8) 3px,transparent 3px,transparent 6px);
}
.cm-leg-swatch.dash-red {
  background:repeating-linear-gradient(90deg,rgba(239,83,80,.8) 0,rgba(239,83,80,.8) 3px,transparent 3px,transparent 6px);
}

.cm-regime-card {
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.06);
  border-radius:4px;padding:12px 14px;
  display:flex;flex-direction:column;gap:7px;
}
.cm-regime-row { display:flex;justify-content:space-between;align-items:baseline; }
.cm-regime-key { font-size:10px;color:#6b7280; }
.cm-regime-val {
  font-size:11px;font-weight:600;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
  color:#d1d4dc;
}
.cm-regime-val.up   { color:#26a69a; }
.cm-regime-val.down { color:#ef5350; }
.cm-regime-val.warn { color:#e07f00; }

.cm-trend-rising  { color:#26a69a; }
.cm-trend-falling { color:#ef5350; }
.cm-trend-stable  { color:#9ca3af; }

.cm-signal-banner {
  margin-top:6px;padding:8px 12px;border-radius:3px;
  font-size:10px;line-height:1.55;
  background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
  color:#9ca3af;
}
.cm-signal-banner.warn { background:rgba(239,83,80,.07);border-color:rgba(239,83,80,.2);color:#ef9a9a; }
.cm-signal-banner.ok   { background:rgba(38,166,154,.05);border-color:rgba(38,166,154,.18);color:#80cbc4; }

@media(max-width:600px){
  #cm-modal{border-radius:4px 4px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;max-height:88vh;}
  #cm-bd{align-items:flex-end;padding:0;}
  .cm-metric{min-width:74px;padding:8px 10px;}
  .cm-m-val{font-size:16px;}
  #cm-chart-wrap{height:170px;}
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
// z-score coloring: >=2.5sigma = down (red), >=1.5sigma = warn (amber), else neutral
// Matches dashboard.js badge thresholds exactly (industry standard: Bloomberg/Reuters)
function _cmZcls(z) {
  if (z == null) return '';
  const a = Math.abs(z);
  return a >= 2.5 ? 'down' : a >= 1.5 ? 'warn' : '';
}
function _cmFmt(v, d) {
  if (v == null) return '\u2014';
  return (v >= 0 ? '+' : '') + v.toFixed(d ?? 2);
}

// ── LWC chart ─────────────────────────────────────────────────────────────────
function _cmDrawChart(container, history, norm, std) {
  if (!container || !window.LightweightCharts || !history || !history.length) return;
  if (_cmChart) { try { _cmChart.remove(); } catch (_) {} _cmChart = null; }

  const LWC = window.LightweightCharts;
  const n   = history.length;
  const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(3);
  const pt  = (i, v) => ({ time: i + 1, value: v });

  _cmChart = LWC.createChart(container, {
    autoSize: true,
    layout: {
      // Solid dark color — prevents the LWC crosshair hover-lighten artifact
      background: { type: 'solid', color: '#070a10' },
      textColor: '#6b7280',
      fontSize: 10,
      fontFamily: "'JetBrains Mono','Courier New',monospace",
      attributionLogo: false,
    },
    grid: {
      vertLines: { visible: false },
      horzLines: { color: 'rgba(255,255,255,.04)' },
    },
    crosshair: {
      mode: LWC.CrosshairMode.Magnet,
      vertLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: '#1e2330' },
      horzLine: { color: 'rgba(255,255,255,.25)', width: 1, style: 2, labelBackgroundColor: '#1e2330' },
    },
    rightPriceScale: {
      borderVisible: false,
      scaleMargins: { top: 0.05, bottom: 0.05 },
    },
    timeScale: { borderVisible: false, visible: false },
    handleScroll: false,
    handleScale: false,
  });

  // Zero line
  const zeroSer = _cmChart.addSeries(LWC.LineSeries, {
    color: 'rgba(255,255,255,.10)',
    lineWidth: 1,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
    priceFormat: { type: 'custom', formatter: fmt },
  });
  zeroSer.setData([pt(0, 0), pt(n - 1, 0)]);

  // Norm + sigma bands (drawn before main so main renders on top)
  if (norm != null) {
    const normSer = _cmChart.addSeries(LWC.LineSeries, {
      color: 'rgba(255,255,255,.40)',
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: { type: 'custom', formatter: fmt },
    });
    normSer.setData([pt(0, norm), pt(n - 1, norm)]);

    if (std != null && std > 0) {
      [[1.5, 'rgba(224,127,0,.5)'], [2.5, 'rgba(239,83,80,.5)']].forEach(([mult, color]) => {
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
          ser.setData([pt(0, v), pt(n - 1, v)]);
        });
      });
    }
  }

  // Vertical day markers at 30d / 60d / 90d positions from the right edge.
  // These visually connect the strip metrics to their corresponding position in the sparkline.
  // Drawn with decreasing opacity so 30d (most recent) is most prominent.
  const markerColors = ['rgba(79,127,255,.38)', 'rgba(79,127,255,.22)', 'rgba(79,127,255,.15)'];
  [30, 60, 90].forEach((days, idx) => {
    const pos = n - days;
    if (pos < 1 || pos >= n - 1) return;
    const vSer = _cmChart.addSeries(LWC.LineSeries, {
      color: markerColors[idx],
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: { type: 'custom', formatter: () => days + 'd' },
    });
    vSer.setData([pt(pos, -1), pt(pos, 1)]);
  });

  // Main rolling-30d line (added last = renders on top of all reference lines)
  const mainSer = _cmChart.addSeries(LWC.LineSeries, {
    color: '#4f7fff',
    lineWidth: 2,
    lastValueVisible: true,
    priceLineVisible: false,
    crosshairMarkerRadius: 4,
    crosshairMarkerBackgroundColor: '#4f7fff',
    crosshairMarkerBorderColor: '#070a10',
    priceFormat: { type: 'custom', formatter: fmt },
  });
  mainSer.setData(history.map((v, i) => pt(i, v)));

  _cmChart.timeScale().fitContent();

  // Crosshair tooltip — "Nd ago" format or "today" for the last point
  const tooltip = document.getElementById('cm-tooltip');
  if (tooltip) {
    _cmChart.subscribeCrosshairMove(param => {
      if (!param || !param.point || !param.seriesData) {
        tooltip.style.display = 'none';
        return;
      }
      const val = param.seriesData.get(mainSer);
      if (val == null) { tooltip.style.display = 'none'; return; }
      const idx = (val.time || 1) - 1;
      const daysAgo = n - 1 - idx;
      const dayLabel = daysAgo === 0 ? 'today' : daysAgo + 'd ago';
      tooltip.style.display = 'block';
      tooltip.textContent = fmt(val.value) + '  \u00b7  ' + dayLabel;
    });
  }
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openCorrModal(corrObj) {
  closeCorrModal();

  const { a, b, corr30, corr, corr90, norm, z_score, std, n30, n, n90, history } = corrObj;
  const absZ = z_score != null ? Math.abs(z_score) : null;

  // Signal banner — three tiers only, matching table badge thresholds (industry standard)
  let sigCls = '', sigTxt = '';
  if (absZ != null) {
    if (absZ >= 2.5) {
      sigCls = 'warn';
      sigTxt = 'Correlation break \u2014 Z-score ' + (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3. The 30d correlation has deviated sharply from its historical norm. This signals a regime change, a temporary dislocation, or an emerging structural shift between the two instruments.';
    } else if (absZ >= 1.5) {
      sigCls = 'warn';
      sigTxt = 'Correlation stretched \u2014 Z-score ' + (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3. The relationship is under stress relative to its historical norm. Monitor for mean reversion or a confirmed break above 2.5\u03c3.';
    } else {
      sigCls = 'ok';
      sigTxt = 'Stable \u2014 Z-score ' + (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + '\u03c3. The correlation is tracking within its historical norm.';
    }
  }

  // Trend: compare 30d snapshot vs 90d snapshot
  let trendHtml = '\u2014';
  if (corr30 != null && corr90 != null) {
    const drift = corr30 - corr90;
    const cls = Math.abs(drift) < 0.03 ? 'stable' : drift > 0 ? 'rising' : 'falling';
    const arrow = cls === 'rising' ? '\u2191' : cls === 'falling' ? '\u2193' : '\u2192';
    const label = cls === 'rising' ? 'Rising' : cls === 'falling' ? 'Falling' : 'Stable';
    trendHtml = '<span class="cm-trend-' + cls + '">' + arrow + ' ' + label + '</span>&nbsp;(30d ' + _cmFmt(corr30) + ' vs 90d ' + _cmFmt(corr90) + ')';
  }

  // 30d vs 252d norm delta — apples-to-apples (both 30d-window based)
  const normDelta = corr30 != null && norm != null ? corr30 - norm : null;

  const hist = Array.isArray(history) ? history : [];

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'Correlation: ' + a + ' vs ' + b);

  bd.innerHTML =
    '<div id="cm-modal">' +
      '<div id="cm-hd">' +
        '<div>' +
          '<div id="cm-title">' + a + ' <span style="color:#6b7280;font-weight:400">vs</span> ' + b + '</div>' +
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
          '<div class="cm-section-title">Rolling 30d correlation \u2014 252-day history \u00b7 dashed verticals = 30d / 60d / 90d lookback</div>' +
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
              '<span class="cm-regime-val" style="color:#6b7280">|z| \u2265 1.5 = stretched &nbsp;\u00b7&nbsp; \u2265 2.5 = break</span>' +
            '</div>' +
            (sigTxt ? '<div class="cm-signal-banner ' + sigCls + '">' + sigTxt + '</div>' : '') +
          '</div>' +
        '</div>' +

      '</div>' +
    '</div>';

  document.body.appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeCorrModal(); });
  document.addEventListener('keydown', _cmKeydown);
  document.getElementById('cm-close').focus();

  // Draw LWC chart — poll until CDN is ready (lazy-loaded by dashboard.js)
  const container = document.getElementById('cm-lwc-container');
  if (window.LightweightCharts) {
    requestAnimationFrame(() => _cmDrawChart(container, hist, norm, std));
  } else {
    const t0 = Date.now();
    const poll = setInterval(() => {
      if (window.LightweightCharts || Date.now() - t0 > 8000) {
        clearInterval(poll);
        if (window.LightweightCharts) _cmDrawChart(container, hist, norm, std);
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
