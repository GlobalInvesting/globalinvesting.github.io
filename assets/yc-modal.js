// ═══════════════════════════════════════════════════════════════════════════
// YIELD CURVE MODAL  v1.0
// File: assets/yc-modal.js
// Loaded AFTER dashboard.js and Chart.js
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
  position:fixed;inset:0;z-index:9200;
  background:rgba(0,0,0,.78);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:ycm-fi .15s ease;
}
@keyframes ycm-fi { from{opacity:0} to{opacity:1} }
@keyframes ycm-su { from{transform:translateY(14px);opacity:0} to{transform:none;opacity:1} }

#ycm-modal {
  background:var(--bg,#131722);
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(800px,100%);
  height:min(560px,90vh);
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:ycm-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}

#ycm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
#ycm-title { font-size:14px;font-weight:600;letter-spacing:.01em; }
#ycm-sub   { font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#ycm-close {
  background:none;border:none;color:var(--text3,#6b7280);font-size:20px;
  cursor:pointer;padding:0 4px;line-height:1;
}
#ycm-close:hover { color:var(--text,#d1d4dc); }

#ycm-strip {
  display:flex;border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;
}
.ycm-metric {
  flex:1;min-width:70px;padding:9px 12px;
  border-right:1px solid rgba(255,255,255,.06);
  text-align:center;
}
.ycm-metric:last-child { border-right:none; }
.ycm-m-lbl { font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px; }
.ycm-m-val { font-size:15px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono',monospace); }
.ycm-m-chg { font-size:9px;margin-top:2px;font-family:var(--font-mono,'JetBrains Mono',monospace); }

#ycm-chart-wrap {
  flex:1;position:relative;padding:16px 18px 12px;
  display:flex;flex-direction:column;
  min-height:0;
}
#ycm-legend {
  display:flex;gap:16px;margin-bottom:10px;flex-shrink:0;
}
.ycm-leg-item { display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2,#9ca3af); }
.ycm-leg-dot  { width:20px;height:2px;border-radius:1px;flex-shrink:0; }
.ycm-leg-dot.dashed { background:repeating-linear-gradient(90deg,currentColor 0,currentColor 4px,transparent 4px,transparent 8px); }

#ycm-canvas-wrap { flex:1;position:relative;min-height:0; }
#ycm-canvas { width:100%!important;height:100%!important; }

/* Shape label badges */
#ycm-shape {
  position:absolute;top:6px;right:6px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.1);
  border-radius:4px;padding:3px 8px;
  font-size:9px;color:var(--text2,#9ca3af);
  font-family:var(--font-mono,'JetBrains Mono',monospace);
  pointer-events:none;
}

/* Tenor table at bottom */
#ycm-table-wrap {
  flex-shrink:0;border-top:1px solid rgba(255,255,255,.07);
  overflow-x:auto;
}
#ycm-table {
  width:100%;border-collapse:collapse;font-size:10px;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
}
#ycm-table th {
  padding:5px 10px;text-align:right;color:var(--text3,#6b7280);
  font-weight:400;font-size:9px;text-transform:uppercase;letter-spacing:.05em;
}
#ycm-table th:first-child { text-align:left; }
#ycm-table td {
  padding:5px 10px;text-align:right;border-top:1px solid rgba(255,255,255,.04);
}
#ycm-table td:first-child { text-align:left;color:var(--text2,#9ca3af); }

@media(max-width:600px){
  #ycm-modal{border-radius:12px 12px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;height:88vh;}
  #ycm-bd{align-items:flex-end;padding:0;}
  .ycm-metric{min-width:55px;padding:7px 8px;}
  .ycm-m-val{font-size:12px;}
}
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
  // Determine curve shape from 3M, 2Y, 10Y
  const t3m  = tenors.find(t => t.label === '3M')?.close;
  const t2y  = tenors.find(t => t.label === '2Y')?.close;
  const t10y = tenors.find(t => t.label === '10Y')?.close;
  if (t3m == null || t10y == null) return null;
  const spread_10_3m = t10y - t3m;
  const spread_10_2y = t2y != null ? t10y - t2y : null;
  if (spread_10_3m < -0.05 && (spread_10_2y == null || spread_10_2y < -0.05)) return 'Inverted';
  if (spread_10_3m > 0.5) return 'Steep';
  if (spread_10_3m > 0.05) return 'Normal';
  return 'Flat';
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openYCModal(tenorData) {
  closeYCModal();
  if (!Array.isArray(tenorData) || tenorData.length === 0) return;

  const shape    = _ycShape(tenorData);
  const shapeColors = { Inverted: '#ef4444', Flat: '#f59e0b', Normal: '#22c55e', Steep: '#4f7fff' };
  const shapeCol = shapeColors[shape] || 'var(--text3)';

  const labels    = tenorData.map(t => t.label);
  const todayVals = tenorData.map(t => t.close);
  const priorVals = tenorData.map(t => t.prev_close);

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

      <div id="ycm-strip">${metricsHtml}</div>

      <div id="ycm-chart-wrap">
        <div id="ycm-legend">
          <div class="ycm-leg-item">
            <div class="ycm-leg-dot" style="background:#4f7fff;height:2px;"></div>
            Today
          </div>
          <div class="ycm-leg-item" style="color:var(--text3,#6b7280);">
            <div class="ycm-leg-dot dashed" style="color:#6b7280;"></div>
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

  document.body.appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeYCModal(); });
  document.addEventListener('keydown', _ycKeydown);

  const closeBtn = document.getElementById('ycm-close');
  if (closeBtn) closeBtn.focus();

  // Render chart after DOM paint
  requestAnimationFrame(() => _ycDrawChart(labels, todayVals, priorVals));
}

function _ycDrawChart(labels, todayVals, priorVals) {
  const canvas = document.getElementById('ycm-canvas');
  if (!canvas || typeof Chart === 'undefined') return;

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
          pointBorderColor: '#131722',
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
          pointBorderColor: '#131722',
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
            color: '#6b7280',
            font: { family: "'JetBrains Mono','Courier New',monospace", size: 10 }
          }
        },
        y: {
          grid: { color: 'rgba(255,255,255,.05)' },
          ticks: {
            color: '#6b7280',
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
