// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v2.1
// File: assets/corr-modal.js
//
// LightweightCharts v5 — Bloomberg-style rolling 30d correlation chart.
// Shows the 252-day rolling-30d history with:
//   • Line series with crosshair tooltip
//   • 252d norm baseline (dashed white)
//   • ±1.5σ / ±2.5σ threshold bands (amber / red dotted)
//   • Fixed Y-axis −1 → +1
//
//   openCorrModal(corrObj)  ← called from correlations-tbody onclick
//   closeCorrModal()
//
// corrObj fields (from quotes.json correlations[]):
//   a, b          — instrument labels
//   corr30/60/90  — rolling Pearson at 30d/60d/90d windows
//   norm          — 252d historical mean of rolling 30d windows
//   std           — std dev of those windows (for band lines)
//   z_score       — (corr60 - norm) / std
//   history       — array of ~223 rolling-30d values, oldest→newest
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
  background:#131722;
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(780px,100%);
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
  border-bottom:1px solid rgba(255,255,255,.07);
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
  display:flex;border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;
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
.cm-m-val.up   { color:#22c55e; }
.cm-m-val.down { color:#ef4444; }
.cm-m-val.warn { color:#f59e0b; }
.cm-m-sub { font-size:9px;color:#6b7280;margin-top:3px; }

#cm-body {
  flex:1;overflow-y:auto;padding:18px 18px 20px;
  min-height:0;display:flex;flex-direction:column;gap:18px;
}
.cm-section-title {
  font-size:9px;color:#6b7280;
  text-transform:uppercase;letter-spacing:.08em;
  margin-bottom:10px;
}

#cm-chart-wrap {
  position:relative;height:210px;
  background:rgba(255,255,255,.015);
  border:1px solid rgba(255,255,255,.07);
  border-radius:6px;overflow:hidden;
}
#cm-lwc-container { width:100%;height:100%; }

#cm-tooltip {
  position:absolute;top:8px;left:12px;
  background:rgba(19,23,34,.92);
  border:1px solid rgba(255,255,255,.12);
  border-radius:4px;padding:5px 9px;
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
  background:repeating-linear-gradient(90deg,rgba(245,158,11,.8) 0,rgba(245,158,11,.8) 3px,transparent 3px,transparent 6px);
}
.cm-leg-swatch.dash-red {
  background:repeating-linear-gradient(90deg,rgba(239,68,68,.8) 0,rgba(239,68,68,.8) 3px,transparent 3px,transparent 6px);
}

.cm-regime-card {
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.07);
  border-radius:6px;padding:12px 14px;
  display:flex;flex-direction:column;gap:7px;
}
.cm-regime-row { display:flex;justify-content:space-between;align-items:baseline; }
.cm-regime-key { font-size:10px;color:#6b7280; }
.cm-regime-val {
  font-size:11px;font-weight:600;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
  color:#d1d4dc;
}
.cm-regime-val.up   { color:#22c55e; }
.cm-regime-val.down { color:#ef4444; }
.cm-regime-val.warn { color:#f59e0b; }

.cm-trend-rising  { color:#22c55e; }
.cm-trend-falling { color:#ef4444; }
.cm-trend-stable  { color:#9ca3af; }

.cm-signal-banner {
  margin-top:6px;padding:8px 12px;border-radius:4px;
  font-size:10px;line-height:1.55;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  color:#9ca3af;
}
.cm-signal-banner.warn { background:rgba(220,38,38,.07);border-color:rgba(220,38,38,.2);color:#f87171; }
.cm-signal-banner.ok   { background:rgba(34,197,94,.05);border-color:rgba(34,197,94,.18);color:#86efac; }

@media(max-width:600px){
  #cm-modal{border-radius:12px 12px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;max-height:88vh;}
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
function _cmZcls(z) {
  if (z == null) return '';
  const a = Math.abs(z);
  return a >= 2.5 ? 'down' : a >= 1.5 ? 'warn' : '';
}
function _cmFmt(v, d) {
  if (v == null) return '—';
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
      background: { type: 'solid', color: 'transparent' },
      textColor: '#6b7280',
      fontSize: 10,
      fontFamily: "'JetBrains Mono','Courier New',monospace",
      attributionLogo: false,
    },
    grid: {
      vertLines: { visible: false },
      horzLines: { color: 'rgba(255,255,255,.05)' },
    },
    crosshair: {
      mode: LWC.CrosshairMode.Magnet,
      vertLine: { color: 'rgba(255,255,255,.3)', width: 1, style: 2, labelBackgroundColor: '#1e2330' },
      horzLine: { color: 'rgba(255,255,255,.3)', width: 1, style: 2, labelBackgroundColor: '#1e2330' },
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
    color: 'rgba(255,255,255,.12)',
    lineWidth: 1,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
    priceFormat: { type: 'custom', formatter: fmt },
  });
  zeroSer.setData([pt(0, 0), pt(n - 1, 0)]);

  // Norm + bands (drawn before main so main renders on top)
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
    normSer.setData([pt(0, norm), pt(n - 1, norm)]);

    if (std != null && std > 0) {
      [[1.5, 'rgba(245,158,11,.5)'], [2.5, 'rgba(239,68,68,.5)']].forEach(([mult, color]) => {
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

  // Main rolling-30d line (added last = renders on top)
  const mainSer = _cmChart.addSeries(LWC.LineSeries, {
    color: '#4f7fff',
    lineWidth: 2,
    lastValueVisible: true,
    priceLineVisible: false,
    crosshairMarkerRadius: 4,
    crosshairMarkerBackgroundColor: '#4f7fff',
    crosshairMarkerBorderColor: '#131722',
    priceFormat: { type: 'custom', formatter: fmt },
  });
  mainSer.setData(history.map((v, i) => pt(i, v)));

  _cmChart.timeScale().fitContent();

  // Crosshair tooltip
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
      tooltip.style.display = 'block';
      tooltip.textContent = fmt(val.value) + '  ·  day ' + (idx + 1) + ' / ' + n;
    });
  }
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openCorrModal(corrObj) {
  closeCorrModal();

  const { a, b, corr30, corr, corr90, norm, z_score, std, n30, n, n90, history } = corrObj;
  const absZ = z_score != null ? Math.abs(z_score) : null;

  // Signal
  let sigCls = '', sigTxt = '';
  if (absZ != null) {
    if (absZ >= 2.5) {
      sigCls = 'warn';
      sigTxt = `Correlation break — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The 60d correlation has deviated sharply from its 252-day norm. This can signal a regime change, a temporary dislocation, or an emerging structural shift between the two instruments.`;
    } else if (absZ >= 1.5) {
      sigCls = 'warn';
      sigTxt = `Correlation stretched — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The relationship is under stress. Monitor for mean reversion or a confirmed break.`;
    } else if (absZ >= 1.0) {
      sigTxt = `Mild deviation — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. Within one standard deviation of the historical norm but showing early signs of drift.`;
    } else {
      sigCls = 'ok';
      sigTxt = `Stable — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The correlation is tracking within its 252-day historical norm.`;
    }
  }

  // Trend
  let trendHtml = '—';
  if (corr30 != null && corr90 != null) {
    const drift = corr30 - corr90;
    const cls = Math.abs(drift) < 0.03 ? 'stable' : drift > 0 ? 'rising' : 'falling';
    const arrow = cls === 'rising' ? '↑' : cls === 'falling' ? '↓' : '→';
    const label = cls === 'rising' ? 'Rising' : cls === 'falling' ? 'Falling' : 'Stable';
    trendHtml = `<span class="cm-trend-${cls}">${arrow} ${label}</span>&nbsp; (30d ${_cmFmt(corr30)} vs 90d ${_cmFmt(corr90)})`;
  }

  const hist = Array.isArray(history) ? history : [];

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', `Correlation: ${a} vs ${b}`);

  bd.innerHTML = `
    <div id="cm-modal">
      <div id="cm-hd">
        <div>
          <div id="cm-title">${a} <span style="color:#6b7280;font-weight:400">vs</span> ${b}</div>
          <div id="cm-sub">Correlation · yfinance · rolling Pearson · 252-day history</div>
        </div>
        <button id="cm-close" onclick="closeCorrModal()" aria-label="Close">×</button>
      </div>

      <div id="cm-strip">
        <div class="cm-metric">
          <div class="cm-m-lbl">30d Corr</div>
          <div class="cm-m-val ${_cmCls(corr30)}">${_cmFmt(corr30)}</div>
          <div class="cm-m-sub">${n30 != null ? n30 + ' sessions' : '—'}</div>
        </div>
        <div class="cm-metric">
          <div class="cm-m-lbl">60d Corr</div>
          <div class="cm-m-val ${_cmCls(corr)}">${_cmFmt(corr)}</div>
          <div class="cm-m-sub">${n != null ? n + ' sessions' : '—'}</div>
        </div>
        <div class="cm-metric">
          <div class="cm-m-lbl">90d Corr</div>
          <div class="cm-m-val ${_cmCls(corr90)}">${_cmFmt(corr90)}</div>
          <div class="cm-m-sub">${n90 != null ? n90 + ' sessions' : '—'}</div>
        </div>
        <div class="cm-metric">
          <div class="cm-m-lbl">252d Norm</div>
          <div class="cm-m-val ${_cmCls(norm)}">${_cmFmt(norm)}</div>
          <div class="cm-m-sub">historical avg</div>
        </div>
        <div class="cm-metric">
          <div class="cm-m-lbl">Z-Score</div>
          <div class="cm-m-val ${_cmZcls(z_score)}">${z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + 'σ' : '—'}</div>
          <div class="cm-m-sub">vs 252d norm</div>
        </div>
      </div>

      <div id="cm-body">

        <div>
          <div class="cm-section-title">Rolling 30d correlation — 252-day history</div>
          <div id="cm-chart-wrap">
            <div id="cm-lwc-container"></div>
            <div id="cm-tooltip"></div>
          </div>
          <div id="cm-legend">
            <div class="cm-leg-item">
              <div class="cm-leg-swatch solid-blue"></div>Rolling 30d
            </div>
            <div class="cm-leg-item">
              <div class="cm-leg-swatch dash-white"></div>252d norm (${_cmFmt(norm)})
            </div>
            ${std != null ? `
            <div class="cm-leg-item">
              <div class="cm-leg-swatch dash-amber"></div>±1.5σ
            </div>
            <div class="cm-leg-item">
              <div class="cm-leg-swatch dash-red"></div>±2.5σ
            </div>` : ''}
          </div>
        </div>

        <div>
          <div class="cm-section-title">Regime assessment</div>
          <div class="cm-regime-card">
            <div class="cm-regime-row">
              <span class="cm-regime-key">Short-term trend</span>
              <span class="cm-regime-val">${trendHtml}</span>
            </div>
            <div class="cm-regime-row">
              <span class="cm-regime-key">60d vs 252d norm</span>
              <span class="cm-regime-val ${corr != null && norm != null ? _cmCls(corr - norm) : ''}">${corr != null && norm != null ? _cmFmt(corr - norm) : '—'}</span>
            </div>
            <div class="cm-regime-row">
              <span class="cm-regime-key">Z-score (std deviations)</span>
              <span class="cm-regime-val ${_cmZcls(z_score)}">${z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + 'σ' : '—'}</span>
            </div>
            <div class="cm-regime-row">
              <span class="cm-regime-key">Signal threshold</span>
              <span class="cm-regime-val" style="color:#6b7280">|z| ≥ 1.5 = stretched &nbsp;·&nbsp; ≥ 2.5 = break</span>
            </div>
            ${sigTxt ? `<div class="cm-signal-banner ${sigCls}">${sigTxt}</div>` : ''}
          </div>
        </div>

      </div>
    </div>`;

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
