// ═══════════════════════════════════════════════════════════════════════════
// YIELD CURVE MODAL  v1.2
// File: assets/yc-modal.js
// Loaded AFTER dashboard.js. Chart.js 4.4.1 loaded via index.html CDN tag.
// Dynamic loader below is a fallback in case CDN is blocked or delayed.
//
//   openYCModal(tenorData)  ← called from yield curve section onclick
//   closeYCModal()
//
// tenorData: array of { label, close, prev_close, chg } — one per tenor
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('ycm-css')) return;
  const s = document.createElement('style');
  s.id = 'ycm-css';
  s.textContent = `
#ycm-bd {
  display:block!important;
  position:absolute!important;
  top:0!important; bottom:0!important;
  left:50%!important; right:0!important;
  overflow-y:auto!important;
  z-index:500!important;
  background:var(--bg,#0d1117)!important;
  border-left:1px solid var(--border2,#30363d)!important;
  scrollbar-width:thin;
  scrollbar-color:var(--border2,#30363d) transparent;
}
#ycm-bd::-webkit-scrollbar { width:3px; }
#ycm-bd::-webkit-scrollbar-thumb { background:var(--border2,#30363d); border-radius:2px; }
/* #main needs position:relative to contain the absolute modal */
#main { position:relative; }

#ycm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg,#0d1117)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#e6edf3);
  display:flex;flex-direction:column;
}
#ycm-modal::before { display:none; }

#ycm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 8px;
  border-bottom:1px solid var(--border2,#30363d);
  flex-shrink:0;background:var(--bg2,#161b22);
}
#ycm-title { font-size:12px;font-weight:600;letter-spacing:-.01em;color:var(--text,#e6edf3); }
#ycm-sub   { font-size:9px;color:var(--text2,#6e7681);margin-top:1px;font-family:var(--font-mono,monospace);letter-spacing:.02em; }
#ycm-close {
  background:none;border:none;color:var(--text2,#6e7681);font-size:16px;
  cursor:pointer;padding:3px 6px;border-radius:4px;line-height:1;
  transition:color .1s,background .1s;
}
#ycm-close:hover { color:var(--text,#e6edf3);background:var(--bg3,#21262d); }

#ycm-strip {
  display:flex;border-bottom:1px solid var(--border2,#30363d);
  flex-shrink:0;overflow-x:auto;background:var(--bg,#0d1117);
  scrollbar-width:none;
}
#ycm-strip::-webkit-scrollbar { display:none; }
.ycm-metric {
  flex:1;min-width:60px;padding:7px 10px;
  border-right:1px solid var(--border2,#30363d);
  background:var(--bg,#0d1117);text-align:center;
}
.ycm-metric:last-child { border-right:none; }
.ycm-m-lbl { font-size:8px;color:var(--text2,#6e7681);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;font-family:var(--font-mono,monospace); }
.ycm-m-val { font-size:13px;font-weight:600;font-family:var(--font-mono,monospace);color:var(--text,#e6edf3); }
.ycm-m-val.up { color:var(--up,#26a69a); } .ycm-m-val.down { color:var(--down,#ef5350); }
.ycm-m-chg { font-size:8px;margin-top:1px;font-family:var(--font-mono,monospace);color:var(--text2,#6e7681); }
.ycm-m-chg.up { color:var(--up,#26a69a); } .ycm-m-chg.down { color:var(--down,#ef5350); }

#ycm-chart-wrap {
  flex:1;position:relative;padding:12px 14px 10px;
  display:flex;flex-direction:column;
  min-height:200px;background:var(--bg,#0d1117);
}
#ycm-legend {
  display:flex;gap:14px;margin-bottom:8px;flex-shrink:0;flex-wrap:wrap;
}
.ycm-leg-item { display:flex;align-items:center;gap:4px;font-size:8.5px;color:var(--text2,#6e7681);font-family:var(--font-mono,monospace); }
.ycm-leg-dot  { width:16px;height:2px;border-radius:1px;flex-shrink:0; }
.ycm-leg-dot.dashed { background:repeating-linear-gradient(90deg,currentColor 0,currentColor 4px,transparent 4px,transparent 8px); }

#ycm-canvas-wrap { flex:1;position:relative;min-height:160px; }
#ycm-canvas { width:100%!important;height:100%!important; }

#ycm-shape {
  position:absolute;top:4px;right:4px;
  background:rgba(255,255,255,.04);
  border:1px solid var(--border2,#30363d);
  border-radius:4px;padding:2px 7px;
  font-size:8.5px;color:var(--text2,#8b949e);
  font-family:var(--font-mono,monospace);
  pointer-events:none;
}

#ycm-table-wrap {
  flex-shrink:0;border-top:1px solid var(--border2,#30363d);
  overflow-x:auto;background:var(--bg2,#161b22);
}
#ycm-table {
  width:100%;border-collapse:collapse;font-size:10px;
  font-family:var(--font-mono,monospace);
}
#ycm-table th {
  padding:4px 8px;text-align:right;color:var(--text2,#6e7681);
  font-weight:500;font-size:8px;text-transform:uppercase;letter-spacing:.08em;
  border-bottom:1px solid var(--border2,#30363d);background:var(--bg2,#161b22);
}
#ycm-table th:first-child { text-align:left; }
#ycm-table td {
  padding:4px 8px;text-align:right;border-top:1px solid rgba(54,60,78,.4);color:var(--text,#e6edf3);
}
#ycm-table td:first-child { text-align:left;color:var(--text2,#8b949e); }
#ycm-table tr:hover td { background:rgba(255,255,255,.03); }
`;
  document.head.appendChild(s);
})();

// ── Helpers ──────────────────────────────────────────────────────────────────
let _ycChart = null;

function _ycChg(chg) {
  if (chg == null || isNaN(chg)) return { txt: '—', cls: '' };
  const sign = chg > 0 ? '+' : '';
  const cls = chg > 0.001 ? 'up' : chg < -0.001 ? 'down' : '';
  return { txt: sign + (chg * 100).toFixed(1) + 'bp', cls };
}

function _ycShape(tenors) {
  // Bloomberg standard: yield curve shape is primarily defined by the 10Y–2Y spread.
  // The 10Y–3M spread is a useful secondary indicator (used by the NY Fed recession model)
  // but the 10Y–2Y is the primary institutional benchmark. Inversión = 10Y–2Y < 0.
  const t2y  = tenors.find(t => t.label === '2Y')?.close;
  const t10y = tenors.find(t => t.label === '10Y')?.close;
  const t3m  = tenors.find(t => t.label === '3M')?.close;
  if (t10y == null) return null;
  // Primary: 10Y–2Y spread (Bloomberg standard)
  const spread_10_2y = t2y != null ? t10y - t2y : null;
  // Secondary: 10Y–3M spread (NY Fed recession indicator)
  const spread_10_3m = t3m != null ? t10y - t3m : null;
  // Inversion: both spreads negative (confirmed inversion across the curve)
  // Flat: within ±20bp of zero on the 10Y–2Y
  // Steep: 10Y–2Y > 50bp
  if (spread_10_2y != null) {
    if (spread_10_2y < 0) return 'Inverted';
    if (spread_10_2y > 0.5) return 'Steep';
    if (spread_10_2y >= -0.2 && spread_10_2y <= 0.2) return 'Flat';
    return 'Normal';
  }
  // Fallback to 10Y–3M if no 2Y data
  if (spread_10_3m != null) {
    if (spread_10_3m < 0) return 'Inverted';
    if (spread_10_3m > 0.5) return 'Steep';
    if (Math.abs(spread_10_3m) <= 0.2) return 'Flat';
    return 'Normal';
  }
  return null;
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openYCModal(tenorData) {
  closeYCModal();
  if (!Array.isArray(tenorData) || tenorData.length === 0) return;

  const shape    = _ycShape(tenorData);
  const shapeColors = { Inverted: '#ef4444', Flat: '#f59e0b', Normal: '#26a69a', Steep: '#4f7fff' };
  const shapeCol = shapeColors[shape] || 'var(--text3)';

  const labels    = tenorData.map(t => t.label);
  const todayVals = tenorData.map(t => t.close);
  const priorVals = tenorData.map(t => t.prev_close);

  // Compute Bloomberg-standard spread metrics for the strip
  const _t = lbl => tenorData.find(t => t.label === lbl);
  const _t2y  = _t('2Y'),  _t10y = _t('10Y'), _t30y = _t('30Y');
  const _tp2y  = _t('2Y'),  _tp10y = _t('10Y'), _tp30y = _t('30Y');
  function _spreadMetric(lbl, aT, bT) {
    if (!aT || !bT || aT.close == null || bT.close == null) return '';
    const val = (aT.close - bT.close) * 100; // in bp
    const priorVal = (aT.prev_close != null && bT.prev_close != null)
      ? (aT.prev_close - bT.prev_close) * 100 : null;
    const chgBp = priorVal != null ? val - priorVal : null;
    const valCol = val < 0 ? 'var(--down,#ef5350)' : val > 0 ? 'var(--up,#26a69a)' : '#8b949e';
    const chgTxt = chgBp != null ? (chgBp > 0 ? '+' : '') + chgBp.toFixed(1) + 'bp' : '—';
    const chgCls = chgBp != null ? (chgBp > 0.5 ? 'up' : chgBp < -0.5 ? 'down' : '') : '';
    return `<div class="ycm-metric">
      <div class="ycm-m-lbl">${lbl}</div>
      <div class="ycm-m-val" style="color:${valCol}">${val.toFixed(0)}bp</div>
      <div class="ycm-m-chg ${chgCls}">${chgTxt}</div>
    </div>`;
  }
  const spreadMetrics = _spreadMetric('10Y–2Y', _t10y, _t2y) + _spreadMetric('30Y–2Y', _t30y, _t2y);

  // Strip HTML for tenor metrics
  const metricsHtml = tenorData.map(t => {
    const { txt, cls } = _ycChg(t.chg);
    const valCls = t.chg > 0.001 ? 'up' : t.chg < -0.001 ? 'down' : '';
    return `<div class="ycm-metric">
      <div class="ycm-m-lbl">${t.label}</div>
      <div class="ycm-m-val ${valCls}">${t.close != null ? t.close.toFixed(2) + '%' : '—'}</div>
      <div class="ycm-m-chg ${cls}">${txt}</div>
    </div>`;
  }).join('');

  // Table rows
  const tableRows = tenorData.map(t => {
    const { txt, cls } = _ycChg(t.chg);
    return `<tr>
      <td>${t.label}</td>
      <td>${t.close != null ? t.close.toFixed(3) + '%' : '—'}</td>
      <td>${t.prev_close != null ? t.prev_close.toFixed(3) + '%' : '—'}</td>
      <td class="${cls}">${txt}</td>
    </tr>`;
  }).join('');

  const bd = document.createElement('div');
  bd.id = 'ycm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'US Treasury Yield Curve');
  bd.innerHTML = `
    <div id="ycm-modal">
      <div id="ycm-hd">
        <div>
          <div id="ycm-title">US Treasury Yield Curve</div>
          <div id="ycm-sub">FRED · today vs prior close · basis points change</div>
        </div>
        <button id="ycm-close" onclick="closeYCModal()" aria-label="Close">×</button>
      </div>

      <div id="ycm-strip">${spreadMetrics}${metricsHtml}</div>

      <div id="ycm-chart-wrap">
        <div id="ycm-legend">
          <div class="ycm-leg-item">
            <div class="ycm-leg-dot" style="background:#4f7fff;height:2px;"></div>
            Today
          </div>
          <div class="ycm-leg-item" style="color:#6e7681;">
            <div class="ycm-leg-dot dashed" style="color:#6e7681;"></div>
            Prior close
          </div>
          ${shape ? `<div class="ycm-leg-item" style="margin-left:auto;color:${shapeCol};font-weight:600;">${shape}</div>` : ''}
        </div>
        <div id="ycm-canvas-wrap">
          <canvas id="ycm-canvas"></canvas>
        </div>
      </div>

      <div id="ycm-table-wrap">
        <table id="ycm-table">
          <thead><tr>
            <th>Tenor</th>
            <th>Today</th>
            <th>Prior close</th>
            <th>Change (bp)</th>
          </tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>`;

  (document.getElementById('main') || document.body).appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeYCModal(); });
  document.addEventListener('keydown', _ycKeydown);

  const closeBtn = document.getElementById('ycm-close');
  if (closeBtn) closeBtn.focus();

  // Render chart after DOM paint
  requestAnimationFrame(() => _ycDrawChart(labels, todayVals, priorVals));
}

function _ycDrawChart(labels, todayVals, priorVals) {
  const canvas = document.getElementById('ycm-canvas');
  if (!canvas) return;

  // Chart.js should be loaded via CDN in index.html; dynamic loader is a fallback.
  if (typeof Chart === 'undefined') {
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
    s.onload = () => _ycDrawChart(labels, todayVals, priorVals);
    document.head.appendChild(s);
    return;
  }

  if (_ycChart) { _ycChart.destroy(); _ycChart = null; }

  const ctx = canvas.getContext('2d');
  _ycChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Today',
          data: todayVals,
          borderColor: '#4f7fff',
          backgroundColor: 'rgba(79,127,255,.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#4f7fff',
          pointBorderColor: '#161b22',
          pointBorderWidth: 2,
        },
        {
          label: 'Prior close',
          data: priorVals,
          borderColor: 'rgba(107,114,128,.55)',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.35,
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: 'rgba(107,114,128,.55)',
          pointBorderColor: '#161b22',
          pointBorderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(19,23,34,.95)',
          borderColor: 'rgba(255,255,255,.12)',
          borderWidth: 1,
          titleFont: { family: "'JetBrains Mono','Courier New',monospace", size: 11 },
          bodyFont:  { family: "'JetBrains Mono','Courier New',monospace", size: 11 },
          padding: 10,
          callbacks: {
            title: items => items[0]?.label + ' Treasury',
            label: item => {
              const v = item.raw;
              if (v == null) return '';
              const today = item.datasetIndex === 0;
              const other = today
                ? item.chart.data.datasets[1].data[item.dataIndex]
                : item.chart.data.datasets[0].data[item.dataIndex];
              const diff = today && other != null ? ((v - other) * 100).toFixed(1) : null;
              let line = (today ? 'Today  ' : 'Prior  ') + v.toFixed(3) + '%';
              if (diff != null) line += '  (' + (diff > 0 ? '+' : '') + diff + 'bp)';
              return line;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,.05)' },
          ticks: {
            color: '#6e7681',
            font: { family: "'JetBrains Mono','Courier New',monospace", size: 10 }
          }
        },
        y: {
          grid: { color: 'rgba(255,255,255,.05)' },
          ticks: {
            color: '#6e7681',
            font: { family: "'JetBrains Mono','Courier New',monospace", size: 10 },
            callback: v => v.toFixed(2) + '%'
          },
          grace: '5%'
        }
      }
    }
  });
}

function _ycKeydown(e) {
  if (e.key === 'Escape') closeYCModal();
}

function closeYCModal() {
  if (_ycChart) { _ycChart.destroy(); _ycChart = null; }
  const bd = document.getElementById('ycm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _ycKeydown);
}

window.openYCModal  = openYCModal;
window.closeYCModal = closeYCModal;
