// ═══════════════════════════════════════════════════════════════════════════
// CB RATES MODAL  v1.0
// File: assets/cb-rates-modal.js
// Loaded AFTER dashboard.js and Chart.js (see index.html)
//
// Pattern mirrors cot-modal-chart.js:
//   openCBRatesModal(ccy, obs, bankInfo, meetingData)  ← called from dashboard.js
//   closeCBRatesModal()
//   cbRatesTab(el, tabId)
// ═══════════════════════════════════════════════════════════════════════════

// ── CSS ─────────────────────────────────────────────────────────────────────
(function () {
  if (document.getElementById('cbr-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cbr-modal-css';
  s.textContent = `
#cbr-bd {
  position:fixed;inset:0;z-index:9100;
  background:rgba(0,0,0,.78);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:cbr-fi .15s ease;
}
@keyframes cbr-fi { from{opacity:0} to{opacity:1} }
@keyframes cbr-su { from{transform:translateY(14px);opacity:0} to{transform:none;opacity:1} }

#cbr-modal {
  background:var(--bg,#131722);
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(860px,100%);
  height:min(600px,90vh);
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:cbr-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}

#cbr-m-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
#cbr-m-title { font-size:14px;font-weight:600;color:var(--text,#d1d4dc);letter-spacing:.01em; }
#cbr-m-sub   { font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#cbr-m-close {
  background:none;border:none;color:var(--text3,#6b7280);font-size:20px;
  cursor:pointer;padding:4px 8px;border-radius:4px;line-height:1;
  transition:color .1s,background .1s;
}
#cbr-m-close:hover { color:var(--text,#d1d4dc);background:rgba(255,255,255,.08); }

/* Metrics strip */
#cbr-m-metrics {
  display:grid;grid-template-columns:repeat(5,1fr);
  gap:1px;background:rgba(255,255,255,.05);
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
.cbr-mm {
  background:var(--bg,#131722);padding:9px 14px;
  display:flex;flex-direction:column;gap:2px;
}
.cbr-mm-lbl { font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cbr-mm-val { font-size:14px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cbr-mm-sub { font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }

/* Tabs */
#cbr-m-tabs {
  display:flex;padding:0 18px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;scrollbar-width:none;
}
#cbr-m-tabs::-webkit-scrollbar { display:none; }
.cbr-tab {
  font-size:11px;padding:9px 13px;cursor:pointer;
  color:var(--text3,#6b7280);border-bottom:2px solid transparent;
  transition:color .1s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.cbr-tab:hover { color:var(--text2,#9096a0); }
.cbr-tab.on { color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff); }

/* Body */
#cbr-m-body {
  flex:1;min-height:0;overflow-y:hidden;
  padding:14px 16px;
  display:flex;flex-direction:column;
  scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,.12) transparent;
}
#cbr-m-body.cbr-body--history { overflow-y:auto; }
#cbr-m-body::-webkit-scrollbar { width:5px; }
#cbr-m-body::-webkit-scrollbar-track { background:transparent; }
#cbr-m-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12);border-radius:3px; }

/* Panels */
.cbr-panel { display:none; }
.cbr-panel.on { display:flex;flex:1;flex-direction:column;min-height:0; }
#cbr-p-history.on { display:block;flex:none; }

/* Chart wrapper */
.cbr-cw {
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.06);
  border-radius:6px;padding:12px 14px;margin-bottom:10px;
  display:flex;flex-direction:column;
}
.cbr-cw.fill { flex:1;min-height:0; }
.cbr-ct { font-size:10px;color:var(--text2,#9096a0);margin-bottom:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.03em; }
.cbr-chart-area { flex:1;min-height:0;height:100%;position:relative; }

/* Decision table */
.cbr-tbl {
  width:100%;border-collapse:collapse;
  font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.cbr-tbl th {
  text-align:right;color:var(--text3,#6b7280);font-weight:400;
  font-size:9px;text-transform:uppercase;letter-spacing:.05em;
  padding:5px 8px 4px;border-bottom:1px solid rgba(255,255,255,.07);
}
.cbr-tbl th:first-child { text-align:left; }
.cbr-tbl td {
  text-align:right;padding:5px 8px;
  border-bottom:1px solid rgba(255,255,255,.04);
}
.cbr-tbl td:first-child { text-align:left;color:var(--text2,#9096a0); }
.cbr-tbl tr:last-child td { border-bottom:none; }
.cbr-tbl tr:hover td { background:rgba(255,255,255,.03); }
.cbr-tbl .now-row td { background:rgba(79,127,255,.06); }

/* Decision marker dot on chart */
.cbr-decision-badge {
  display:inline-flex;align-items:center;gap:4px;
  font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  padding:2px 7px;border-radius:3px;
}

/* Cycle strip */
.cbr-cycle-bar {
  height:4px;border-radius:2px;flex:1;
  transition:width .4s ease;
}

.cu { color:var(--up,#26a69a); }
.cd { color:var(--down,#ef5350); }
.cf { color:var(--text2,#9096a0); }
.cn { color:var(--text3,#6b7280); }

@media(max-width:600px) {
  #cbr-bd { padding:0;align-items:flex-end; }
  #cbr-modal { width:100%;height:93vh;border-radius:12px 12px 0 0;border-bottom:none; }
  #cbr-m-hd { padding:10px 14px 9px; }
  #cbr-m-title { font-size:13px; }
  #cbr-m-metrics { grid-template-columns:repeat(3,1fr); }
  .cbr-mm { padding:6px 10px; }
  .cbr-mm-val { font-size:12px; }
  #cbr-m-tabs { padding:0 8px; }
  .cbr-tab { font-size:10px;padding:8px 8px; }
  #cbr-m-body { padding:8px; }
  .cbr-cw { padding:9px 10px;margin-bottom:8px; }
}
`;
  document.head.appendChild(s);
})();

// ── Helpers ──────────────────────────────────────────────────────────────────

const _cbrCharts = [];

function _destroyCBRCharts() {
  _cbrCharts.forEach(c => { try { c.destroy(); } catch (_) {} });
  _cbrCharts.length = 0;
}

const _CBR_GRID    = 'rgba(242,242,242,0.06)';
const _CBR_TICK    = '#787b86';
const _CBR_BG_TIP  = '#1e222d';
const _CBR_BDR_TIP = 'rgba(255,255,255,0.12)';
const _monoF       = "'JetBrains Mono','Courier New',monospace";

// Reuse crosshair plugin from COT modal if available, else define inline
const _cbrCrosshair = (typeof _crosshairPlugin !== 'undefined') ? _crosshairPlugin : {
  id: 'cbrCrosshair',
  afterDraw(chart) {
    const { ctx, chartArea, tooltip } = chart;
    if (!tooltip?._active?.length) return;
    const x = tooltip._active[0].element.x;
    if (x < chartArea.left || x > chartArea.right) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.restore();
  }
};

// Decision markers plugin — draws vertical dotted lines + labels at rate-change points
function _makeDecisionPlugin(decisionIndices, decisions) {
  return {
    id: 'cbrDecisions',
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      if (!scales.x || !decisionIndices.length) return;
      decisionIndices.forEach((idx, i) => {
        if (idx < 0 || idx >= chart.data.labels.length) return;
        const x = scales.x.getPixelForValue(idx);
        if (x < chartArea.left || x > chartArea.right) return;
        const delta = decisions[i]?.delta || 0;
        const col = delta > 0 ? '#26a69a' : delta < 0 ? '#ef5350' : '#787b86';
        // Vertical line
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top);
        ctx.lineTo(x, chartArea.bottom);
        ctx.strokeStyle = col + '60';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Small dot on the line at the rate value
        const rate = decisions[i]?.rate;
        if (rate != null && scales.y) {
          const y = scales.y.getPixelForValue(rate);
          if (y >= chartArea.top && y <= chartArea.bottom) {
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = col;
            ctx.fill();
            ctx.strokeStyle = 'var(--bg,#131722)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
        // Delta label at top
        if (delta !== 0) {
          const sign = delta > 0 ? '+' : '';
          const label = sign + Math.round(delta * 100) + 'bp';
          ctx.font = `600 9px ${_monoF}`;
          const tw = ctx.measureText(label).width;
          const bX = x - tw / 2 - 3;
          const bY = chartArea.top + 3;
          ctx.fillStyle = col + '22';
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(bX, bY, tw + 6, 14, 2);
          else ctx.rect(bX, bY, tw + 6, 14);
          ctx.fill();
          ctx.fillStyle = col;
          ctx.textBaseline = 'top';
          ctx.textAlign = 'center';
          ctx.fillText(label, x, bY + 2);
        }
        ctx.restore();
      });
    }
  };
}

// Y-axis price badge (last value)
const _cbrPriceBadge = {
  id: 'cbrPriceBadge',
  afterDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!scales.y) return;
    const fSize = 10;
    chart.data.datasets.forEach((ds, i) => {
      if (ds._noBadge) return;
      const meta = chart.getDatasetMeta(i);
      if (!meta.visible) return;
      let lastIdx = -1;
      for (let k = ds.data.length - 1; k >= 0; k--) {
        if (ds.data[k] != null) { lastIdx = k; break; }
      }
      if (lastIdx < 0) return;
      const val = ds.data[lastIdx];
      const yPx = scales.y.getPixelForValue(val);
      if (yPx < chartArea.top || yPx > chartArea.bottom) return;
      const col = typeof ds.borderColor === 'string' ? ds.borderColor : '#787b86';
      const label = typeof val === 'number' ? val.toFixed(2) + '%' : val;
      ctx.save();
      ctx.font = `600 ${fSize}px ${_monoF}`;
      const tw = ctx.measureText(label).width;
      const bW = tw + 10, bH = 16;
      const bX = chartArea.right + 2;
      const bY = yPx - bH / 2;
      ctx.fillStyle = col;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bX, bY, bW, bH, 2);
      else ctx.rect(bX, bY, bW, bH);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.fillText(label, bX + 5, yPx);
      ctx.restore();
    });
  }
};

function _buildCBROptions(decisionPlugin, extraOpts) {
  const isMob = typeof window !== 'undefined' && window.innerWidth < 600;
  return Object.assign({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400, easing: 'easeOutQuart' },
    interaction: { mode: 'index', intersect: false },
    layout: { padding: { top: 6, right: 0, bottom: 0, left: 0 } },
    plugins: {
      legend: {
        position: 'top', align: 'start',
        labels: {
          color: _CBR_TICK,
          font: { family: _monoF, size: 10 },
          boxWidth: 18, boxHeight: 2, padding: 14,
        }
      },
      tooltip: {
        backgroundColor: _CBR_BG_TIP,
        titleColor: '#d1d4dc',
        bodyColor: '#9096a0',
        borderColor: _CBR_BDR_TIP,
        borderWidth: 1,
        padding: { x: 12, y: 8 },
        cornerRadius: 4,
        callbacks: {
          label(ctx) {
            if (ctx.raw == null) return null;
            return ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(2)}%`;
          }
        }
      },
      cbrCrosshair:    _cbrCrosshair,
      cbrDecisions:    decisionPlugin || { id: 'cbrDecisionsNoop', afterDraw() {} },
      cbrPriceBadge:   _cbrPriceBadge,
    },
    scales: {
      x: {
        ticks: {
          color: _CBR_TICK,
          font: { family: _monoF, size: isMob ? 9 : 9 },
          maxRotation: 0, autoSkip: true,
          maxTicksLimit: isMob ? 6 : 12, padding: 4
        },
        grid: { color: _CBR_GRID, drawBorder: false },
        border: { display: false }
      },
      y: {
        position: 'right',
        ticks: {
          color: _CBR_TICK,
          font: { family: _monoF, size: isMob ? 9 : 9 },
          callback: v => v.toFixed(2) + '%',
          padding: 6, maxTicksLimit: 8
        },
        grid: { color: _CBR_GRID, drawBorder: false },
        border: { display: false }
      }
    },
    events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
  }, extraOpts || {});
}

// ── Derive decision points from monthly observations ─────────────────────────
// obs: [{date, value}, ...] newest-first
// Returns { labels, rates, decisions: [{idx, date, rate, delta, rateAfter}] }
function _processCBRateData(obs) {
  if (!obs || !obs.length) return { labels: [], rates: [], decisions: [] };

  // Reverse to chronological order for charting
  const chron = [...obs].reverse();

  const labels = chron.map(o => {
    const [y, m] = o.date.split('-');
    return m + '/' + y.slice(2);  // "01/24", "02/24" etc
  });
  const rates = chron.map(o => parseFloat(o.value));

  // Decision points: where the rate changed vs previous month
  const decisions = [];
  for (let i = 1; i < chron.length; i++) {
    const prev = parseFloat(chron[i - 1].value);
    const curr = parseFloat(chron[i].value);
    const delta = curr - prev;
    if (Math.abs(delta) >= 0.01) {
      decisions.push({
        idx: i,
        date: chron[i].date,
        rate: curr,
        delta,
        rateAfter: curr,
      });
    }
  }

  return { labels, rates, decisions };
}

// ── Main modal builder ────────────────────────────────────────────────────────

function openCBRatesModal(ccy, obs, bankInfo, meetingData) {
  closeCBRatesModal();

  const { labels, rates, decisions } = _processCBRateData(obs);
  const currentRate  = rates[rates.length - 1] ?? 0;
  const rateStart    = rates[0] ?? 0;
  const totalChange  = currentRate - rateStart;
  const nDecisions   = decisions.length;
  const nMonths      = obs.length;

  // Cycle direction
  const trend = (function() {
    if (!decisions.length) return 'flat';
    const last = decisions[decisions.length - 1];
    if (last.delta > 0) return 'hiking';
    if (last.delta < 0) return 'cutting';
    return 'flat';
  })();
  const trendLabel = trend === 'hiking' ? '↑ Hiking cycle' : trend === 'cutting' ? '↓ Cutting cycle' : '— On hold';
  const trendCol   = trend === 'hiking' ? 'var(--up,#26a69a)' : trend === 'cutting' ? 'var(--down,#ef5350)' : 'var(--text2,#9096a0)';

  // Forward rate from meetingData
  const fwdRate = meetingData?.fwdRate ?? null;
  const bias    = meetingData?.bias ?? null;
  const nextMtg = meetingData?.nextMeeting ?? '—';
  const biasLabel = bias === 'cut' ? '↓ Cut' : bias === 'hike' ? '↑ Hike' : '→ Hold';
  const biasCol   = bias === 'cut' ? 'var(--down,#ef5350)' : bias === 'hike' ? 'var(--up,#26a69a)' : 'var(--text2,#9096a0)';

  // Cumulative cut/hike in this cycle (consecutive decisions from latest)
  const lastDir = decisions.length ? Math.sign(decisions[decisions.length - 1].delta) : 0;
  let cycleCumulative = 0;
  for (let i = decisions.length - 1; i >= 0; i--) {
    if (Math.sign(decisions[i].delta) === lastDir) cycleCumulative += decisions[i].delta;
    else break;
  }
  const cycleStr = lastDir !== 0
    ? (lastDir > 0 ? '+' : '') + Math.round(cycleCumulative * 100) + 'bp this cycle'
    : '—';

  const bankName = bankInfo?.name || ccy;
  const bankShort = bankInfo?.short || ccy;
  const flagHtml = bankInfo?.flag
    ? `<span class="fi fi-${bankInfo.flag}" style="margin-right:8px;border-radius:2px;font-size:16px;vertical-align:middle;"></span>`
    : '';

  // Build modal HTML
  const bd = document.createElement('div');
  bd.id = 'cbr-bd';
  bd.innerHTML = `
<div id="cbr-modal">

  <div id="cbr-m-hd">
    <div>
      <div id="cbr-m-title">${flagHtml}${bankName} — Policy Rate History</div>
      <div id="cbr-m-sub">${nMonths} months · ${nDecisions} decisions · CB rates cache · monthly observations</div>
    </div>
    <button id="cbr-m-close" onclick="closeCBRatesModal()" aria-label="Close">✕</button>
  </div>

  <div id="cbr-m-metrics">
    <div class="cbr-mm">
      <div class="cbr-mm-lbl">Current Rate</div>
      <div class="cbr-mm-val">${currentRate.toFixed(2)}%</div>
      <div class="cbr-mm-sub">${bankShort}</div>
    </div>
    <div class="cbr-mm">
      <div class="cbr-mm-lbl">Cycle</div>
      <div class="cbr-mm-val" style="font-size:12px;color:${trendCol}">${trendLabel}</div>
      <div class="cbr-mm-sub">${cycleStr}</div>
    </div>
    <div class="cbr-mm">
      <div class="cbr-mm-lbl">Total Change</div>
      <div class="cbr-mm-val ${totalChange > 0 ? 'cu' : totalChange < 0 ? 'cd' : 'cf'}">${totalChange > 0 ? '+' : ''}${Math.round(totalChange * 100)}bp</div>
      <div class="cbr-mm-sub">${nMonths}m period</div>
    </div>
    <div class="cbr-mm">
      <div class="cbr-mm-lbl">Next Meeting</div>
      <div class="cbr-mm-val cf" style="font-size:12px">${nextMtg}</div>
      <div class="cbr-mm-sub" style="color:${biasCol}">${biasLabel}</div>
    </div>
    <div class="cbr-mm">
      <div class="cbr-mm-lbl">Fwd Rate</div>
      <div class="cbr-mm-val ${bias === 'cut' ? 'cd' : bias === 'hike' ? 'cu' : 'cf'}">${fwdRate != null ? fwdRate : '—'}</div>
      <div class="cbr-mm-sub">OIS implied</div>
    </div>
  </div>

  <div id="cbr-m-tabs" role="tablist" aria-label="CB rate chart views">
    <div class="cbr-tab on"  data-tab="chart"    onclick="cbRatesTab(this,'chart')"   role="tab" aria-selected="true">Rate Chart</div>
    <div class="cbr-tab"     data-tab="decisions" onclick="cbRatesTab(this,'decisions')" role="tab" aria-selected="false">Decisions</div>
  </div>

  <div id="cbr-m-body">

    <!-- CHART TAB ──────────────────────────────────────────────── -->
    <div id="cbr-p-chart" class="cbr-panel on">
      <div class="cbr-cw fill">
        <div class="cbr-ct" id="cbr-chart-label">POLICY RATE · MONTHLY · DECISION MARKERS</div>
        <div class="cbr-chart-area"><canvas id="cbr-c-main"></canvas></div>
      </div>
    </div>

    <!-- DECISIONS TAB ──────────────────────────────────────────── -->
    <div id="cbr-p-decisions" class="cbr-panel">
      <div class="cbr-cw">
        <div class="cbr-ct">RATE DECISIONS · ${nDecisions} CHANGES IN ${nMonths} MONTHS</div>
        <div style="overflow-x:auto;">
          <table class="cbr-tbl" aria-label="CB rate decisions">
            <thead><tr>
              <th scope="col" style="text-align:left">Date</th>
              <th scope="col">Before</th>
              <th scope="col">After</th>
              <th scope="col">Change</th>
              <th scope="col">Cumulative</th>
            </tr></thead>
            <tbody id="cbr-decisions-body"></tbody>
          </table>
        </div>
      </div>
      <div class="cbr-cw" style="margin-top:4px;">
        <div class="cbr-ct">RATE PROFILE · ${rateStart.toFixed(2)}% → ${currentRate.toFixed(2)}%</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden;">
          <div style="background:var(--bg,#131722);padding:10px 14px;">
            <div style="font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_monoF}">Period High</div>
            <div style="font-size:18px;font-weight:600;font-family:${_monoF};color:var(--up,#26a69a)">${Math.max(...rates).toFixed(2)}%</div>
          </div>
          <div style="background:var(--bg,#131722);padding:10px 14px;">
            <div style="font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_monoF}">Period Low</div>
            <div style="font-size:18px;font-weight:600;font-family:${_monoF};color:var(--down,#ef5350)">${Math.min(...rates).toFixed(2)}%</div>
          </div>
          <div style="background:var(--bg,#131722);padding:10px 14px;">
            <div style="font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_monoF}">Decisions</div>
            <div style="font-size:18px;font-weight:600;font-family:${_monoF};color:var(--text,#d1d4dc)">${nDecisions}</div>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>`;

  document.body.appendChild(bd);

  // Populate decisions table
  const dBody = document.getElementById('cbr-decisions-body');
  if (dBody && decisions.length) {
    // Reverse for newest-first display
    const reversed = [...decisions].reverse();
    let cumulFromStart = 0;

    // Pre-compute cumulative from oldest
    const cumuls = [];
    let running = 0;
    for (const d of decisions) { running += d.delta; cumuls.push(running); }
    // Now match to reversed
    const dWithCumul = reversed.map((d, i) => {
      const origIdx = decisions.length - 1 - i;
      return { ...d, cumul: cumuls[origIdx] };
    });

    dBody.innerHTML = dWithCumul.map((d, i) => {
      const isLatest = i === 0;
      const before = (d.rate - d.delta).toFixed(2);
      const after  = d.rate.toFixed(2);
      const chgBp  = Math.round(d.delta * 100);
      const cumBp  = Math.round(d.cumul * 100);
      const chgCls = chgBp > 0 ? 'cu' : chgBp < 0 ? 'cd' : 'cf';
      const cumCls = cumBp > 0 ? 'cu' : cumBp < 0 ? 'cd' : 'cf';
      return `<tr${isLatest ? ' class="now-row"' : ''}>
        <td>${d.date}${isLatest ? ' <span style="color:var(--up,#26a69a);font-size:9px">latest</span>' : ''}</td>
        <td>${before}%</td>
        <td>${after}%</td>
        <td class="${chgCls}">${chgBp > 0 ? '+' : ''}${chgBp}bp</td>
        <td class="${cumCls}">${cumBp > 0 ? '+' : ''}${cumBp}bp</td>
      </tr>`;
    }).join('');
  } else if (dBody) {
    dBody.innerHTML = `<tr><td colspan="5" style="color:var(--text3,#6b7280);padding:12px 8px;">No rate changes in the available history.</td></tr>`;
  }

  // Close handlers
  bd.addEventListener('click', e => { if (e.target === bd) closeCBRatesModal(); });
  const esc = e => { if (e.key === 'Escape') closeCBRatesModal(); };
  document.addEventListener('keydown', esc);
  bd._esc = esc;

  // Store data for lazy chart build
  bd._chartData = { labels, rates, decisions, fwdRate, bias, currentRate };

  // Build chart immediately (it's the default tab)
  requestAnimationFrame(() => requestAnimationFrame(() => {
    _buildCBRChart(bd._chartData);
  }));
}

// ── Chart builder ─────────────────────────────────────────────────────────────

function _buildCBRChart(data) {
  const cv = document.getElementById('cbr-c-main');
  if (!cv || typeof Chart === 'undefined') return;

  // Destroy previous instance
  const existing = _cbrCharts.find(c => c.canvas === cv);
  if (existing) {
    existing.destroy();
    _cbrCharts.splice(_cbrCharts.indexOf(existing), 1);
  }

  const { labels, rates, decisions, fwdRate, bias, currentRate } = data;

  // Decision plugin indices and deltas
  const decisionPlugin = _makeDecisionPlugin(
    decisions.map(d => d.idx),
    decisions
  );

  // Datasets
  const datasets = [
    {
      label: 'Policy Rate',
      data: rates,
      borderColor: '#4f7fff',
      backgroundColor: 'transparent',
      stepped: 'before',       // ← step chart: rate is discrete, not interpolated
      tension: 0,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointBackgroundColor: '#4f7fff',
      fill: true,
      _gradHex: '#4f7fff',
    }
  ];

  // Forward rate line (dotted) if available
  if (fwdRate != null) {
    // Extend one data point past the end as a projected value
    const fwdVal = parseFloat(fwdRate);
    if (!isNaN(fwdVal)) {
      // Add one label + null to main dataset, then fwd dataset with last two points
      const extLabels = [...labels, 'Fwd'];
      const extRates  = [...rates, null];
      const fwdData   = rates.map(() => null);
      fwdData[fwdData.length - 1] = currentRate;
      fwdData.push(fwdVal);

      datasets[0].data = extRates;

      datasets.push({
        label: 'OIS Forward',
        data: fwdData.concat([]),  // will be set below
        borderColor: bias === 'cut' ? '#ef5350' : bias === 'hike' ? '#26a69a' : '#787b86',
        backgroundColor: 'transparent',
        borderDash: [4, 4],
        borderWidth: 1.5,
        tension: 0,
        pointRadius: [
          ...rates.map(() => 0),
          4   // dot at forward point
        ],
        pointBackgroundColor: bias === 'cut' ? '#ef5350' : bias === 'hike' ? '#26a69a' : '#787b86',
        stepped: false,
        fill: false,
        spanGaps: true,
        _noBadge: true,
      });
      datasets[1].data = fwdData;

      // Update label in chart header
      const lbl = document.getElementById('cbr-chart-label');
      if (lbl) lbl.textContent = 'POLICY RATE · MONTHLY · DECISION MARKERS · OIS FORWARD';

      const opts = _buildCBROptions(decisionPlugin, {
        scales: {
          x: { labels: extLabels },
        }
      });
      opts.scales.x.labels = undefined; // Chart.js uses data labels, not scale labels

      const c = new Chart(cv, {
        type: 'line',
        data: { labels: extLabels, datasets },
        options: _buildCBROptions(decisionPlugin)
      });
      _cbrCharts.push(c);
      return;
    }
  }

  const c = new Chart(cv, {
    type: 'line',
    data: { labels, datasets },
    options: _buildCBROptions(decisionPlugin)
  });
  _cbrCharts.push(c);
}

// ── Tab switching ─────────────────────────────────────────────────────────────

function cbRatesTab(el, tabId) {
  document.querySelectorAll('.cbr-tab').forEach(t => {
    t.classList.remove('on');
    t.setAttribute('aria-selected', 'false');
  });
  document.querySelectorAll('.cbr-panel').forEach(p => p.classList.remove('on'));
  el.classList.add('on');
  el.setAttribute('aria-selected', 'true');
  const panel = document.getElementById('cbr-p-' + tabId);
  if (panel) panel.classList.add('on');
  const body = document.getElementById('cbr-m-body');
  if (body) body.classList.toggle('cbr-body--history', tabId === 'decisions');

  if (tabId === 'chart') {
    const bd = document.getElementById('cbr-bd');
    if (bd?._chartData) {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        _buildCBRChart(bd._chartData);
      }));
    }
  }
}

// ── Close ─────────────────────────────────────────────────────────────────────

function closeCBRatesModal() {
  const bd = document.getElementById('cbr-bd');
  if (bd) {
    if (bd._esc) document.removeEventListener('keydown', bd._esc);
    bd.remove();
  }
  _destroyCBRCharts();
}

// ── Expose globals ────────────────────────────────────────────────────────────

window.openCBRatesModal  = openCBRatesModal;
window.closeCBRatesModal = closeCBRatesModal;
window.cbRatesTab        = cbRatesTab;
