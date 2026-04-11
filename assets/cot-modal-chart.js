// ═══════════════════════════════════════════════════════════════════════════
// COT MODAL CHART  v1.1
// File: assets/cot-modal-chart.js
// Loaded AFTER dashboard.js and Chart.js (see index.html)
// ═══════════════════════════════════════════════════════════════════════════

// ── CSS ─────────────────────────────────────────────────────────────────────
(function () {
  if (document.getElementById('cot-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cot-modal-css';
  s.textContent = `
#cot-bd {
  position:fixed;inset:0;z-index:9100;
  background:rgba(0,0,0,.78);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:cot-fi .15s ease;
}
@keyframes cot-fi { from{opacity:0} to{opacity:1} }
@keyframes cot-su { from{transform:translateY(14px);opacity:0} to{transform:none;opacity:1} }

#cot-modal {
  background:var(--bg,#131722);
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(920px,100%);
  height:min(680px,90vh);
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:cot-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}
#cot-m-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
#cot-m-title { font-size:14px;font-weight:600;color:var(--text,#d1d4dc);letter-spacing:.01em; }
#cot-m-sub   { font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#cot-m-close {
  background:none;border:none;color:var(--text3,#6b7280);font-size:20px;
  cursor:pointer;padding:4px 8px;border-radius:4px;line-height:1;
  transition:color .1s,background .1s;
}
#cot-m-close:hover { color:var(--text,#d1d4dc);background:rgba(255,255,255,.08); }

/* Metrics strip */
#cot-m-metrics {
  display:grid;grid-template-columns:repeat(6,1fr);
  gap:1px;background:rgba(255,255,255,.05);
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
.cot-mm {
  background:var(--bg,#131722);padding:9px 14px;
  display:flex;flex-direction:column;gap:2px;
}
.cot-mm-lbl { font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-mm-val { font-size:13px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-mm-sub { font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }

/* Tabs */
#cot-m-tabs {
  display:flex;padding:0 18px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;
  scrollbar-width:none;
}
#cot-m-tabs::-webkit-scrollbar { display:none; }
.cot-tab {
  font-size:11px;padding:9px 13px;cursor:pointer;
  color:var(--text3,#6b7280);border-bottom:2px solid transparent;
  transition:color .1s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.cot-tab:hover { color:var(--text2,#9096a0); }
.cot-tab.on { color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff); }

/* Body: always fills remaining height, never changes size between tabs */
#cot-m-body {
  flex:1;min-height:0;
  overflow-y:auto;
  padding:14px 16px;
  display:flex;flex-direction:column;
  scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,.12) transparent;
}
#cot-m-body::-webkit-scrollbar { width:5px; }
#cot-m-body::-webkit-scrollbar-track { background:transparent; }
#cot-m-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12);border-radius:3px; }
/* Disable scroll when a single-chart tab is active so flex-fill reaches the canvas */
#cot-m-body.cot-body--chart { overflow-y:hidden; }

/* All panels fill the body completely */
.cot-panel { display:none; }
.cot-panel.on { display:flex;flex:1;flex-direction:column;min-height:0; }

/* Multi-section panels scroll their own content */
#p-overview.on, #p-history.on { display:block; }

/* Net Position and Long/Short: single-chart panels fill entire body */
#p-net.on .cot-cw,
#p-split.on .cot-cw {
  flex:1;display:flex;flex-direction:column;margin-bottom:0;min-height:0;
}
#p-net.on .cot-cw > .cot-chart-area,
#p-split.on .cot-cw > .cot-chart-area { flex:1;min-height:0;position:relative; }

/* Participants: chart has fixed height, description scrolls below */
#p-participants.on { overflow-y:auto; }
#p-participants .cot-chart-area { height:260px;position:relative; }

/* Chart wrapper */
.cot-cw {
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.06);
  border-radius:6px;padding:12px 14px;margin-bottom:10px;
}
.cot-ct { font-size:10px;color:var(--text2,#9096a0);margin-bottom:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.03em; }

/* Gauge */
.cot-gauge-track {
  height:8px;border-radius:4px;
  background:linear-gradient(to right,
    #ef5350 0%,#ef5350 14%,
    #ff9800 14%,#ff9800 28%,
    #42a5f5 28%,#42a5f5 72%,
    #ff9800 72%,#ff9800 86%,
    #ef5350 86%,#ef5350 100%);
  position:relative;margin:8px 0;
}
.cot-gauge-pin {
  position:absolute;top:-5px;
  width:18px;height:18px;border-radius:50%;
  background:#fff;border:2px solid var(--bg,#131722);
  transform:translateX(-50%);
  box-shadow:0 0 0 2px rgba(255,255,255,.35);
  transition:left .6s cubic-bezier(.34,1.56,.64,1);
}
.cot-gauge-lbls {
  display:flex;justify-content:space-between;
  font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}

/* Long/Short bar */
.cot-posbar {
  height:14px;background:rgba(255,255,255,.06);
  border-radius:3px;overflow:hidden;position:relative;flex:1;
}
.cot-posbar-fill { height:100%;transition:width .4s ease; }
.cot-posbar-mid {
  position:absolute;left:50%;top:0;bottom:0;
  width:1px;background:rgba(255,255,255,.2);
}

/* Table */
.cot-tbl {
  width:100%;border-collapse:collapse;
  font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.cot-tbl th {
  text-align:right;color:var(--text3,#6b7280);font-weight:400;
  font-size:9px;text-transform:uppercase;letter-spacing:.05em;
  padding:5px 8px 4px;border-bottom:1px solid rgba(255,255,255,.07);
}
.cot-tbl th:first-child { text-align:left; }
.cot-tbl td {
  text-align:right;padding:4px 8px;
  border-bottom:1px solid rgba(255,255,255,.04);
}
.cot-tbl td:first-child { text-align:left;color:var(--text2,#9096a0); }
.cot-tbl tr:last-child td { border-bottom:none; }
.cot-tbl tr:hover td { background:rgba(255,255,255,.03); }

.cu { color:var(--up,#26a69a); } .cd { color:var(--down,#ef5350); } .cf { color:var(--text2,#9096a0); }
.badge-ext { font-size:9px;padding:2px 6px;border-radius:3px;margin-left:6px; }
.badge-warn { background:rgba(255,152,0,.15);color:#ff9800; }
.badge-ok   { background:rgba(38,166,154,.12);color:var(--up,#26a69a); }

@media(max-width:600px) {
  /* Modal: bottom sheet pattern */
  #cot-bd { padding:0; align-items:flex-end; }
  #cot-modal {
    width:100%;
    height:93vh;
    border-radius:12px 12px 0 0;
    border-bottom:none;
  }

  /* Header: tighter, title single line with ellipsis */
  #cot-m-hd { padding:10px 14px 9px; }
  #cot-m-title { font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:calc(100% - 36px); }
  #cot-m-sub { font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  /* Metrics: 3-col compact */
  #cot-m-metrics { grid-template-columns:repeat(3,1fr); }
  .cot-mm { padding:6px 10px; }
  .cot-mm-val { font-size:11px; }
  .cot-mm-lbl, .cot-mm-sub { font-size:8px; }

  /* Tabs: all 5 fit */
  #cot-m-tabs { padding:0 8px; }
  .cot-tab { font-size:10px; padding:8px 8px; }

  /* Body: tighter padding, must remain flex column with min-height:0 for chart fill */
  #cot-m-body { padding:8px; }
  .cot-cw { padding:9px 10px; margin-bottom:8px; }
  .cot-ct { font-size:9px; }

  /* Net / Long/Short: body must NOT have overflow-y:auto so flex-fill works on canvas.
     Class toggled by cotTab() JS — avoids :has() which has uneven mobile support */
  #cot-m-body.cot-body--chart { overflow-y:hidden; }

  /* Participants: taller chart on mobile so data is visible */
  #p-participants .cot-chart-area { height:220px; }
  /* Description text compact */
  #p-participants .cot-cw:last-child { font-size:9px; line-height:1.5; }

  /* Overview: gauge section compact */
  #p-overview .cot-cw:first-child .cot-gauge-lbls { font-size:8px; }

  /* Overview Participants table: horizontal scroll, no text wrap in cells */
  #p-overview .cot-cw:last-child { overflow-x:auto; -webkit-overflow-scrolling:touch; }
  #p-overview .cot-tbl { min-width:380px; }
  #p-overview .cot-tbl td,
  #p-overview .cot-tbl th { white-space:nowrap; font-size:9px; padding:4px 6px; }

  /* History: horizontal scroll */
  #p-history .cot-cw > div { overflow-x:auto; -webkit-overflow-scrolling:touch; }
  #p-history .cot-tbl { min-width:540px; font-size:9px; }
  #p-history .cot-tbl th,
  #p-history .cot-tbl td { padding:3px 5px; }
}`;
  document.head.appendChild(s);
})();

// ── Helpers ──────────────────────────────────────────────────────────────────

function _cotFmt(n) {
  if (n == null || isNaN(n)) return '—';
  const a = Math.abs(n), sg = n >= 0 ? '+' : '-';
  if (a >= 1e6) return sg + (a / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return sg + Math.round(a / 1e3) + 'k';
  return (n > 0 ? '+' : '') + n.toLocaleString();
}

function _cotCls(v) {
  return v == null ? 'cf' : v > 0 ? 'cu' : v < 0 ? 'cd' : 'cf';
}

function _calcZ(history, field = 'levNet') {
  const vals = history.map(h => h[field]).filter(v => v != null);
  if (vals.length < 6) return null;
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  const std  = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
  if (std === 0) return 0;
  return (vals[vals.length - 1] - mean) / std;
}

function _calcPct(history, field = 'levNet') {
  const vals = history.map(h => h[field]).filter(v => v != null);
  if (vals.length < 6) return null;
  const cur   = vals[vals.length - 1];
  const below = vals.slice(0, -1).filter(v => v < cur).length;
  return Math.round(below / (vals.length - 1) * 100);
}

function _posLabel(z) {
  if (z == null) return { text: 'N/A',             color: 'var(--text2,#9096a0)' };
  if (z >  2.0)  return { text: 'Extreme Long',    color: 'var(--down,#ef5350)' };
  if (z >  1.5)  return { text: 'Very Long',        color: '#ff9800' };
  if (z >  0.5)  return { text: 'Moderately Long', color: 'var(--up,#26a69a)' };
  if (z > -0.5)  return { text: 'Neutral',          color: 'var(--text2,#9096a0)' };
  if (z > -1.5)  return { text: 'Moderately Short',color: 'var(--up,#26a69a)' };
  if (z > -2.0)  return { text: 'Very Short',       color: '#ff9800' };
  return               { text: 'Extreme Short',    color: 'var(--down,#ef5350)' };
}

// ── Chart.js wrappers ─────────────────────────────────────────────────────────

const _cotCharts = [];

function _destroyCharts() {
  _cotCharts.forEach(c => { try { c.destroy(); } catch (_) {} });
  _cotCharts.length = 0;
}

const _monoFont = "'JetBrains Mono','Courier New',monospace";

const _chartDefaults = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 350 },
  interaction: { mode: 'index', intersect: false },
  layout: { padding: { top: 8, right: 4, bottom: 0, left: 0 } },
  plugins: {
    legend: {
      position: 'top', align: 'start',
      labels: { color: '#9096a0', font: { family: _monoFont, size: 10 }, boxWidth: 14, padding: 14 }
    },
    tooltip: {
      backgroundColor: '#1e222d', titleColor: '#d1d4dc', bodyColor: '#9096a0',
      borderColor: 'rgba(255,255,255,.1)', borderWidth: 1, padding: 10, cornerRadius: 6,
      callbacks: { label: ctx => ` ${ctx.dataset.label}: ${_cotFmt(ctx.raw)}` }
    }
  },
  scales: {
    x: {
      ticks: { color: '#6b7280', font: { family: _monoFont, size: 9 }, maxRotation: 45, maxTicksLimit: 14 },
      grid:  { color: 'rgba(255,255,255,.04)' }
    },
    y: {
      ticks: { color: '#6b7280', font: { family: _monoFont, size: 9 }, callback: v => _cotFmt(v) },
      grid:  { color: 'rgba(255,255,255,.06)' }
    }
  }
};

function _lineChart(canvas, labels, datasets) {
  if (typeof Chart === 'undefined') return null;
  const cfg = JSON.parse(JSON.stringify(_chartDefaults));
  cfg.type = 'line';
  cfg.data = { labels, datasets };
  const c = new Chart(canvas, cfg);
  _cotCharts.push(c);
  return c;
}

function _barChart(canvas, labels, datasets) {
  if (typeof Chart === 'undefined') return null;
  const cfg = JSON.parse(JSON.stringify(_chartDefaults));
  cfg.type = 'bar';
  cfg.data = { labels, datasets };
  cfg.plugins = cfg.plugins || {};
  cfg.plugins.zeroLine = {
    id: 'zeroLine',
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      const y0 = scales.y.getPixelForValue(0);
      if (y0 >= chartArea.top && y0 <= chartArea.bottom) {
        ctx.save(); ctx.beginPath();
        ctx.moveTo(chartArea.left, y0); ctx.lineTo(chartArea.right, y0);
        ctx.strokeStyle = 'rgba(255,255,255,.2)'; ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); ctx.stroke(); ctx.restore();
      }
    }
  };
  const c = new Chart(canvas, cfg);
  _cotCharts.push(c);
  return c;
}

// ── Build modal HTML ──────────────────────────────────────────────────────────

function openCOTModal(ccy, data) {
  closeCOTModal();

  const history = Array.isArray(data.history) ? [...data.history] : [];
  const net     = data.netPosition    || 0;
  const long_   = data.longPositions  || 0;
  const short_  = data.shortPositions || 0;
  const total   = long_ + short_;
  const lPct    = total > 0 ? Math.round(long_ / total * 100) : 50;
  const amNet   = data.assetManagerNet;
  const ddNet   = data.dealerNet;
  const weekEnd = data.weekEnding || '';
  const nWks    = history.length;

  const zScore  = _calcZ(history);
  const pctHist = _calcPct(history);
  const zInfo   = _posLabel(zScore);
  const isCrowded = Math.abs(zScore || 0) >= 1.5;

  // WoW
  let wow = null;
  if (history.length >= 2) {
    const prev = history[history.length - 2];
    const pn   = prev.levNet ?? ((prev.levLong || 0) - (prev.levShort || 0));
    wow = net - pn;
  }

  // Net%OI
  const netPctOI  = total > 0 ? (net / total * 100) : null;
  const netPctStr = netPctOI != null
    ? (netPctOI > 0 ? '+' : '') + netPctOI.toFixed(1) + '%'
    : '—';

  const zStr = zScore != null ? (zScore > 0 ? '+' : '') + zScore.toFixed(2) : '—';
  const pStr = pctHist != null ? pctHist + '%' : '—';
  const zCol = isCrowded ? '#ff9800' : 'var(--text,#d1d4dc)';

  // LF/AM alignment
  const lfDir  = Math.sign(net);
  const amDir  = amNet != null ? Math.sign(amNet) : 0;
  const aligned = lfDir !== 0 && amDir !== 0 && lfDir === amDir;

  // Gauge position: z in [-3,+3] → [5%,95%]
  const gaugeLeft = zScore != null
    ? ((Math.max(-3, Math.min(3, zScore)) + 3) / 6 * 90 + 5).toFixed(1) + '%'
    : '50%';

  // Build HTML
  const bd = document.createElement('div');
  bd.id = 'cot-bd';
  bd.innerHTML = `
<div id="cot-modal">

  <div id="cot-m-hd">
    <div>
      <div id="cot-m-title">CFTC Positioning · ${ccy} · Leveraged Funds</div>
      <div id="cot-m-sub">week ending ${weekEnd} · ${nWks}w history · CFTC TFF Disaggregated · Options+Futures Combined</div>
    </div>
    <button id="cot-m-close" onclick="closeCOTModal()" aria-label="Close">✕</button>
  </div>

  <div id="cot-m-metrics">
    <div class="cot-mm">
      <div class="cot-mm-lbl">Net LF</div>
      <div class="cot-mm-val ${_cotCls(net)}">${_cotFmt(net)}</div>
      <div class="cot-mm-sub">contracts</div>
    </div>
    <div class="cot-mm">
      <div class="cot-mm-lbl">Long %</div>
      <div class="cot-mm-val ${_cotCls(lPct - 50)}">${lPct}%</div>
      <div class="cot-mm-sub">of own OI</div>
    </div>
    <div class="cot-mm">
      <div class="cot-mm-lbl">WoW Δ</div>
      <div class="cot-mm-val ${_cotCls(wow)}">${_cotFmt(wow)}</div>
      <div class="cot-mm-sub">weekly change</div>
    </div>
    <div class="cot-mm">
      <div class="cot-mm-lbl">Net%OI</div>
      <div class="cot-mm-val ${_cotCls(netPctOI)}">${netPctStr}</div>
      <div class="cot-mm-sub">normalised</div>
    </div>
    <div class="cot-mm">
      <div class="cot-mm-lbl">Z-Score</div>
      <div class="cot-mm-val" style="color:${zCol}">${zStr}</div>
      <div class="cot-mm-sub">${nWks}w history</div>
    </div>
    <div class="cot-mm">
      <div class="cot-mm-lbl">Percentile</div>
      <div class="cot-mm-val" style="color:${zCol}">${pStr}</div>
      <div class="cot-mm-sub" style="color:${zInfo.color}">${zInfo.text}</div>
    </div>
  </div>

  <div id="cot-m-tabs" role="tablist" aria-label="COT chart views">
    <div class="cot-tab on"  data-tab="overview"      onclick="cotTab(this,'overview')"      role="tab" aria-selected="true">Overview</div>
    <div class="cot-tab"     data-tab="net"            onclick="cotTab(this,'net')"            role="tab" aria-selected="false">Net Position</div>
    <div class="cot-tab"     data-tab="split"          onclick="cotTab(this,'split')"          role="tab" aria-selected="false">Long / Short</div>
    <div class="cot-tab"     data-tab="participants"   onclick="cotTab(this,'participants')"   role="tab" aria-selected="false">Participants</div>
    <div class="cot-tab"     data-tab="history"        onclick="cotTab(this,'history')"        role="tab" aria-selected="false">History</div>
  </div>

  <div id="cot-m-body">

    <!-- OVERVIEW ──────────────────────────────────────────── -->
    <div id="p-overview" class="cot-panel on">

      <div class="cot-cw">
        <div class="cot-ct">POSITIONING GAUGE · Z-SCORE VS ${nWks}-WEEK HISTORY</div>
        <div class="cot-gauge-track">
          <div class="cot-gauge-pin" id="cot-pin" style="left:50%"></div>
        </div>
        <div class="cot-gauge-lbls">
          <span>Extreme Short<br>(&lt;−2σ)</span>
          <span style="text-align:center">Short<br>(−1.5σ)</span>
          <span style="text-align:center">Neutral</span>
          <span style="text-align:center">Long<br>(+1.5σ)</span>
          <span style="text-align:right">Extreme Long<br>(&gt;+2σ)</span>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--text2,#9096a0);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.6">
          Positioning: <strong style="color:${zInfo.color}">${zInfo.text}</strong>
          ${pctHist != null ? ` · Historical percentile <strong>${pctHist}%</strong>` : ''}
          ${isCrowded ? `<span class="badge-ext badge-warn">CROWDED TRADE</span>` : `<span class="badge-ext badge-ok">Within normal range</span>`}
        </div>
      </div>

      <div class="cot-cw">
        <div class="cot-ct">LONG / SHORT SPLIT · LEVERAGED FUNDS OPEN INTEREST</div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
          <span style="font-size:10px;color:var(--up,#26a69a);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);min-width:50px">${long_.toLocaleString()}</span>
          <div class="cot-posbar">
            <div class="cot-posbar-fill" style="width:${lPct}%;background:var(--up,#26a69a);opacity:.8"></div>
            <div class="cot-posbar-mid"></div>
          </div>
          <span style="font-size:10px;color:var(--down,#ef5350);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);min-width:50px;text-align:right">${short_.toLocaleString()}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace)">
          <span>LONGS ${lPct}%</span><span>SHORTS ${100 - lPct}%</span>
        </div>
      </div>

      <div class="cot-cw">
        <div class="cot-ct">PARTICIPANTS · NET BY CATEGORY</div>
        <table class="cot-tbl" aria-label="COT positioning by participant category">
          <thead><tr>
            <th scope="col">Category</th><th scope="col">Net</th><th scope="col">Direction</th><th scope="col">Note</th>
          </tr></thead>
          <tbody>
            <tr>
              <td>Leveraged Funds</td>
              <td class="${_cotCls(net)}">${_cotFmt(net)}</td>
              <td class="${_cotCls(net)}">${net > 0 ? '▲ Long' : net < 0 ? '▼ Short' : '— Neutral'}</td>
              <td style="font-size:9px;color:var(--text3,#6b7280)">Hedge funds / CTAs — primary signal</td>
            </tr>
            ${amNet != null ? `<tr>
              <td>Asset Managers</td>
              <td class="${_cotCls(amNet)}">${_cotFmt(amNet)}</td>
              <td class="${_cotCls(amNet)}">${amNet > 0 ? '▲ Long' : amNet < 0 ? '▼ Short' : '— Neutral'}</td>
              <td style="font-size:9px;color:var(--text3,#6b7280)">Mutual funds / pensions — slow trend follower</td>
            </tr>` : ''}
            ${ddNet != null ? `<tr>
              <td>Dealers</td>
              <td class="${_cotCls(ddNet)}">${_cotFmt(ddNet)}</td>
              <td class="${_cotCls(ddNet)}">${ddNet > 0 ? '▲ Long' : ddNet < 0 ? '▼ Short' : '— Neutral'}</td>
              <td style="font-size:9px;color:var(--text3,#6b7280)">Market-makers — contrarian signal</td>
            </tr>` : ''}
            ${amNet != null ? `<tr>
              <td style="color:var(--text2,#9096a0)">LF / AM</td>
              <td colspan="3" style="color:${aligned ? 'var(--up,#26a69a)' : '#ff9800'}">
                ${aligned
                  ? '● Aligned — both in the same direction (reinforced signal)'
                  : '○ Diverging — LF and AM are opposed (exercise caution)'}
              </td>
            </tr>` : ''}
          </tbody>
        </table>
      </div>
    </div>

    <!-- NET POSITION ──────────────────────────────────────── -->
    <div id="p-net" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">NET POSITION · LEVERAGED FUNDS · WEEKLY CONTRACTS</div>
        <div class="cot-chart-area"><canvas id="c-net"></canvas></div>
      </div>
    </div>

    <!-- LONG / SHORT ──────────────────────────────────────── -->
    <div id="p-split" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">LONGS VS SHORTS · LEVERAGED FUNDS · CONTRACTS</div>
        <div class="cot-chart-area"><canvas id="c-split"></canvas></div>
      </div>
    </div>

    <!-- PARTICIPANTS ─────────────────────────────────────── -->
    <div id="p-participants" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">LF vs AM vs DEALER · NET BY CATEGORY</div>
        <div class="cot-chart-area"><canvas id="c-part"></canvas></div>
      </div>
      <div class="cot-cw">
        <div style="font-size:10px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.7">
          <strong style="color:var(--text2,#9096a0)">LF (Leveraged Funds):</strong> Hedge funds and CTAs. Primary speculative momentum signal. Extreme readings historically precede reversals.<br>
          <strong style="color:var(--text2,#9096a0)">AM (Asset Managers):</strong> Mutual funds and pensions. Slow trend-followers. Confluence with LF = stronger signal.<br>
          <strong style="color:var(--text2,#9096a0)">DD (Dealers):</strong> Market-makers. Typically positioned against speculative momentum. Useful contrarian signal at historical extremes.
        </div>
      </div>
    </div>

    <!-- HISTORY ─────────────────────────────────────────── -->
    <div id="p-history" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">WEEKLY HISTORY · ${nWks} WEEKS</div>
        <div style="overflow-x:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.12) transparent">
          <table class="cot-tbl" aria-label="Weekly COT history">
            <thead><tr>
              <th scope="col">Week</th><th scope="col">Net LF</th><th scope="col">WoW Δ</th><th scope="col">Longs</th><th scope="col">Shorts</th><th scope="col">Long%</th><th scope="col">Net%OI</th><th scope="col">AM Net</th><th scope="col">Dealer</th>
            </tr></thead>
            <tbody id="cot-hist-body"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</div>`;

  document.body.appendChild(bd);

  // Close on backdrop click or Escape
  bd.addEventListener('click', e => { if (e.target === bd) closeCOTModal(); });
  const esc = e => { if (e.key === 'Escape') closeCOTModal(); };
  document.addEventListener('keydown', esc);
  bd._esc = esc;

  // Animate gauge pin after paint
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const pin = document.getElementById('cot-pin');
    if (pin) pin.style.left = gaugeLeft;
  }));

  // ── Prepare chart data ────────────────────────────────────────────────────
  const labels   = history.map(h => (h.weekEnding || '').replace(/(\d{4})-(\d{2})-(\d{2})/, '$3/$2'));
  const netData  = history.map(h => h.levNet ?? ((h.levLong || 0) - (h.levShort || 0)));
  const lngData  = history.map(h => h.levLong);
  const shrtData = history.map(h => h.levShort);
  const amData   = history.map(h => h.assetManagerNet);
  const ddData   = history.map(h => h.dealerNet);
  const barCols  = netData.map(v => v >= 0 ? 'rgba(38,166,154,.75)' : 'rgba(239,83,80,.75)');

  // ── Build history table ───────────────────────────────────────────────────
  const tbody = document.getElementById('cot-hist-body');
  if (tbody) {
    const rev = [...history].reverse();
    tbody.innerHTML = rev.map((h, i) => {
      const hNet   = h.levNet ?? ((h.levLong || 0) - (h.levShort || 0));
      const hL     = h.levLong, hS = h.levShort;
      const hTot   = (hL || 0) + (hS || 0);
      const hLP    = hTot > 0 ? Math.round(hL / hTot * 100) : null;
      const hPctOI = hTot > 0 ? (hNet / hTot * 100) : null;
      const prevH  = rev[i + 1];
      const hWow   = prevH
        ? hNet - (prevH.levNet ?? ((prevH.levLong || 0) - (prevH.levShort || 0)))
        : null;
      const isNow = i === 0;
      return `<tr style="${isNow ? 'background:rgba(255,255,255,.04)' : ''}">
        <td>${h.weekEnding}${isNow ? ' <span style="color:var(--up,#26a69a);font-size:9px">now</span>' : ''}</td>
        <td class="${_cotCls(hNet)}">${_cotFmt(hNet)}</td>
        <td class="${_cotCls(hWow)}">${hWow != null ? _cotFmt(hWow) : '—'}</td>
        <td style="color:var(--up,#26a69a)">${hL != null ? hL.toLocaleString() : '—'}</td>
        <td style="color:var(--down,#ef5350)">${hS != null ? hS.toLocaleString() : '—'}</td>
        <td class="${_cotCls(hLP != null ? hLP - 50 : null)}">${hLP != null ? hLP + '%' : '—'}</td>
        <td class="${_cotCls(hPctOI)}">${hPctOI != null ? (hPctOI > 0 ? '+' : '') + hPctOI.toFixed(1) + '%' : '—'}</td>
        <td class="${_cotCls(h.assetManagerNet)}">${h.assetManagerNet != null ? _cotFmt(h.assetManagerNet) : '—'}</td>
        <td class="${_cotCls(h.dealerNet ? -h.dealerNet : null)}">${h.dealerNet != null ? _cotFmt(h.dealerNet) : '—'}</td>
      </tr>`;
    }).join('');
  }

  // ── Lazy chart init ───────────────────────────────────────────────────────
  const built = {};

  function buildChart(tabId) {
    if (built[tabId]) return;
    built[tabId] = true;

    if (tabId === 'net') {
      const cv = document.getElementById('c-net');
      if (cv) _barChart(cv, labels, [{ label: `${ccy} LF Net`, data: netData, backgroundColor: barCols, borderWidth: 0, borderRadius: 2 }]);
    }
    if (tabId === 'split') {
      const cv = document.getElementById('c-split');
      if (cv) _lineChart(cv, labels, [
        { label: 'Longs',  data: lngData,  borderColor: '#26a69a', backgroundColor: 'rgba(38,166,154,.08)', fill: true, tension: .3, pointRadius: 2 },
        { label: 'Shorts', data: shrtData, borderColor: '#ef5350', backgroundColor: 'rgba(239,83,80,.08)',  fill: true, tension: .3, pointRadius: 2 },
      ]);
    }
    if (tabId === 'participants') {
      const cv = document.getElementById('c-part');
      if (cv) {
        const ds = [{ label: 'Leveraged Funds', data: netData, borderColor: '#4f7fff', backgroundColor: 'transparent', tension: .3, pointRadius: 2, borderWidth: 2 }];
        if (amData.some(v => v != null)) ds.push({ label: 'Asset Managers', data: amData, borderColor: '#ff9800', backgroundColor: 'transparent', tension: .3, pointRadius: 2, borderDash: [4, 4], borderWidth: 2 });
        if (ddData.some(v => v != null)) ds.push({ label: 'Dealers', data: ddData, borderColor: '#ef5350', backgroundColor: 'transparent', tension: .3, pointRadius: 2, borderDash: [2, 4], borderWidth: 2 });
        // Custom inline legend drawn inside the chart area — prevents Chart.js legend from stealing horizontal space
        const inlineLegend = {
          id: 'inlineLegend',
          afterDraw(chart) {
            const { ctx, chartArea } = chart;
            const lineH = 16, gap = 10, lineW = 20, textOff = 24;
            const maxX = chartArea.right - 4;
            let x = chartArea.left + 8;
            let y = chartArea.top + 10;
            ctx.save();
            ctx.font = `10px 'JetBrains Mono','Courier New',monospace`;
            chart.data.datasets.forEach(ds => {
              const labelW = ctx.measureText(ds.label).width;
              // Wrap to next line if this item would overflow
              if (x + lineW + textOff + labelW > maxX && x > chartArea.left + 8) {
                x = chartArea.left + 8;
                y += lineH;
              }
              ctx.setLineDash(ds.borderDash || []);
              ctx.strokeStyle = ds.borderColor;
              ctx.lineWidth = 2;
              ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + lineW, y); ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = '#9096a0';
              ctx.fillText(ds.label, x + textOff, y + 4);
              x += lineW + textOff + labelW + gap;
            });
            ctx.restore();
          }
        };
        // Build config explicitly — do NOT use spread `{ ...cfg, plugins: [inlineLegend] }`
        // because that overwrites cfg.plugins (an options-level object) with the plugins array,
        // making Chart.js restore the native legend. The correct Chart.js API separates
        // `options.plugins` (configuration) from top-level `plugins` (plugin instances).
        const partCfg = JSON.parse(JSON.stringify(_chartDefaults));
        const c = new Chart(cv, {
          type: 'line',
          data: { labels, datasets: ds },
          options: {
            responsive: partCfg.responsive,
            maintainAspectRatio: partCfg.maintainAspectRatio,
            animation: partCfg.animation,
            interaction: partCfg.interaction,
            layout: { padding: { top: 44, right: 4, bottom: 0, left: 0 } },
            plugins: { legend: { display: false }, tooltip: partCfg.plugins.tooltip },
            scales: partCfg.scales
          },
          plugins: [inlineLegend]
        });
        _cotCharts.push(c);
      }
    }
  }

  bd._build = buildChart;
  // Do NOT pre-build any chart here — charts must be built lazily when the user
  // clicks a tab, so that (a) the panel is visible, (b) cot-body--chart class is set,
  // and (c) Chart.js can measure the canvas at its actual rendered size.
}

// ── Tab switching ─────────────────────────────────────────────────────────────

function cotTab(el, tabId) {
  document.querySelectorAll('.cot-tab').forEach(t => {
    t.classList.remove('on');
    t.setAttribute('aria-selected', 'false');
  });
  document.querySelectorAll('.cot-panel').forEach(p => p.classList.remove('on'));
  el.classList.add('on');
  el.setAttribute('aria-selected', 'true');
  const panel = document.getElementById('p-' + tabId);
  if (panel) panel.classList.add('on');
  // Toggle class on body so CSS can target single-chart vs multi-section tabs
  // without relying on :has() which has uneven mobile browser support
  const body = document.getElementById('cot-m-body');
  if (body) {
    const isChart = tabId === 'net' || tabId === 'split';
    body.classList.toggle('cot-body--chart', isChart);
  }
  const bd = document.getElementById('cot-bd');
  if (bd && bd._build) setTimeout(() => bd._build(tabId), 80);
}

// ── Close ─────────────────────────────────────────────────────────────────────

function closeCOTModal() {
  const bd = document.getElementById('cot-bd');
  if (bd) {
    if (bd._esc) document.removeEventListener('keydown', bd._esc);
    bd.remove();
  }
  _destroyCharts();
}

// ── Expose globals ────────────────────────────────────────────────────────────

window.openCOTModal  = openCOTModal;
window.closeCOTModal = closeCOTModal;
window.cotTab        = cotTab;
