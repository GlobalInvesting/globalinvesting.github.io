// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION MODAL  v1.0
// File: assets/corr-modal.js
// Loaded AFTER dashboard.js and Chart.js
//
//   openCorrModal(corrObj)  ← called from correlations-tbody onclick
//   closeCorrModal()
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('cm-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'cm-modal-css';
  s.textContent = `
#cm-bd {
  position:fixed;inset:0;z-index:9200;
  background:rgba(0,0,0,.78);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:cm-fi .15s ease;
}
@keyframes cm-fi { from{opacity:0} to{opacity:1} }
@keyframes cm-su { from{transform:translateY(14px);opacity:0} to{transform:none;opacity:1} }

#cm-modal {
  background:var(--bg,#131722);
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(700px,100%);
  max-height:90vh;
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:cm-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}

#cm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
#cm-title { font-size:14px;font-weight:600;color:var(--text,#d1d4dc);letter-spacing:.01em; }
#cm-sub   { font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#cm-close {
  background:none;border:none;color:var(--text3,#6b7280);font-size:20px;
  cursor:pointer;padding:0 4px;line-height:1;
}
#cm-close:hover { color:var(--text,#d1d4dc); }

#cm-strip {
  display:flex;gap:0;border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;
}
.cm-metric {
  flex:1;min-width:100px;padding:10px 14px;
  border-right:1px solid rgba(255,255,255,.06);
}
.cm-metric:last-child { border-right:none; }
.cm-m-lbl { font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px; }
.cm-m-val { font-size:18px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono',monospace); }
.cm-m-sub { font-size:9px;color:var(--text3,#6b7280);margin-top:2px; }

#cm-body {
  flex:1;overflow-y:auto;padding:20px 20px 16px;
  min-height:260px;
}

.cm-section-title {
  font-size:10px;color:var(--text3,#6b7280);
  text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:14px;
}

/* Horizontal bar rows */
.cm-bar-group { display:flex;flex-direction:column;gap:10px;margin-bottom:24px; }
.cm-bar-row { display:flex;align-items:center;gap:10px; }
.cm-bar-label {
  font-size:10px;color:var(--text2,#9ca3af);
  width:60px;flex-shrink:0;text-align:right;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
}
.cm-bar-track {
  flex:1;height:20px;background:rgba(255,255,255,.04);
  border-radius:3px;position:relative;overflow:visible;
}
/* Center line at 0 */
.cm-bar-track::after {
  content:'';position:absolute;left:50%;top:0;bottom:0;
  width:1px;background:rgba(255,255,255,.12);
}
.cm-bar-fill {
  position:absolute;top:3px;bottom:3px;
  border-radius:2px;transition:width .3s ease;
}
.cm-bar-val {
  font-size:10px;font-weight:600;
  font-family:var(--font-mono,'JetBrains Mono',monospace);
  width:44px;flex-shrink:0;
}

/* Z-score section */
.cm-zscore-card {
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.07);
  border-radius:6px;padding:12px 14px;
}
.cm-zscore-row { display:flex;justify-content:space-between;align-items:center;margin-bottom:6px; }
.cm-zscore-row:last-child { margin-bottom:0; }
.cm-zscore-key { font-size:10px;color:var(--text3,#6b7280); }
.cm-zscore-val { font-size:11px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono',monospace); }
.cm-signal-banner {
  margin-top:14px;padding:8px 12px;border-radius:4px;
  font-size:10px;line-height:1.5;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  color:var(--text2,#9ca3af);
}
.cm-signal-banner.warn { background:rgba(220,38,38,.08);border-color:rgba(220,38,38,.25);color:#f87171; }
.cm-signal-banner.ok   { background:rgba(34,197,94,.06);border-color:rgba(34,197,94,.2);color:#86efac; }

/* Mobile */
@media(max-width:600px){
  #cm-modal{border-radius:12px 12px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;max-height:88vh;}
  #cm-bd{align-items:flex-end;padding:0;}
  .cm-metric{min-width:80px;padding:8px 10px;}
  .cm-m-val{font-size:15px;}
}
`;
  document.head.appendChild(s);
})();

// ── Helpers ──────────────────────────────────────────────────────────────────
function _cmCls(v) {
  if (v == null) return '';
  return v >= 0.3 ? 'up' : v <= -0.3 ? 'down' : '';
}
function _cmFmt(v) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(2);
}
// Map corr value to a horizontal bar: returns {left, width, color}
// Track spans -1 to +1; center at 50%
function _corrBar(v) {
  if (v == null) return null;
  const clamped = Math.max(-1, Math.min(1, v));
  const isPos = clamped >= 0;
  const pct = Math.abs(clamped) * 50; // max 50% from center
  if (isPos) {
    return { left: '50%', width: pct + '%', color: '#22c55e' };
  } else {
    return { left: (50 - pct) + '%', width: pct + '%', color: '#ef4444' };
  }
}

// ── Open ─────────────────────────────────────────────────────────────────────
function openCorrModal(corrObj) {
  closeCorrModal();

  const { a, b, corr30, corr, corr90, norm, z_score, n30, n, n90 } = corrObj;

  // Z-score interpretation
  const absZ = z_score != null ? Math.abs(z_score) : null;
  let signalClass = '', signalText = '';
  if (absZ != null) {
    if (absZ >= 2.5) {
      signalClass = 'warn';
      signalText = `⚠ Correlation break — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The 60d correlation has deviated sharply from its 252-day norm. This can signal a regime change, a temporary dislocation, or an emerging structural shift between the two instruments.`;
    } else if (absZ >= 1.5) {
      signalClass = 'warn';
      signalText = `↯ Correlation stretched — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The relationship is under stress. Monitor for mean reversion or a confirmed break.`;
    } else if (absZ >= 1.0) {
      signalClass = '';
      signalText = `~ Mild deviation — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. Within one standard deviation of the historical norm but showing early signs of drift.`;
    } else {
      signalClass = 'ok';
      signalText = `● Stable — Z-score ${z_score >= 0 ? '+' : ''}${z_score.toFixed(2)}σ. The correlation is behaving in line with its 252-day historical norm.`;
    }
  }

  const windows = [
    { label: '30d', val: corr30, n: n30 },
    { label: '60d', val: corr, n: n },
    { label: '90d', val: corr90, n: n90 },
    { label: '252d norm', val: norm, n: null },
  ];

  // Build bar rows HTML
  const barsHtml = windows.map(w => {
    const bar = _corrBar(w.val);
    const cls = _cmCls(w.val);
    const fillStyle = bar
      ? `left:${bar.left};width:${bar.width};background:${bar.color};`
      : '';
    const isNorm = w.label === '252d norm';
    const labelStyle = isNorm ? 'color:var(--text3,#6b7280);' : '';
    const normDash = isNorm
      ? `position:absolute;left:${bar ? bar.left : '50%'};width:${bar ? bar.width : '0'};top:3px;bottom:3px;background:repeating-linear-gradient(90deg,rgba(255,255,255,.35) 0,rgba(255,255,255,.35) 3px,transparent 3px,transparent 6px);border-radius:2px;`
      : '';
    return `
      <div class="cm-bar-row">
        <div class="cm-bar-label" style="${labelStyle}">${w.label}${w.n ? `<span style="color:var(--text3);font-size:8px;"> (${w.n})</span>` : ''}</div>
        <div class="cm-bar-track">
          ${bar ? `<div class="cm-bar-fill" style="${isNorm ? normDash : fillStyle}"></div>` : ''}
        </div>
        <div class="cm-bar-val ${cls}">${_cmFmt(w.val)}</div>
      </div>`;
  }).join('');

  const bd = document.createElement('div');
  bd.id = 'cm-bd';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', `Correlation detail: ${a} vs ${b}`);
  bd.innerHTML = `
    <div id="cm-modal">
      <div id="cm-hd">
        <div>
          <div id="cm-title">${a} <span style="color:var(--text3,#6b7280);font-weight:400;">vs</span> ${b}</div>
          <div id="cm-sub">Correlation · yfinance · rolling windows</div>
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
          <div class="cm-m-val ${absZ != null && absZ >= 1.5 ? 'down' : ''}">${z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + 'σ' : '—'}</div>
          <div class="cm-m-sub">vs 252d norm</div>
        </div>
      </div>

      <div id="cm-body">
        <div class="cm-section-title">Correlation across windows</div>
        <div class="cm-bar-group">
          ${barsHtml}
        </div>

        <div class="cm-section-title">Regime assessment</div>
        <div class="cm-zscore-card">
          <div class="cm-zscore-row">
            <span class="cm-zscore-key">60d vs 252d norm</span>
            <span class="cm-zscore-val">${corr != null && norm != null ? _cmFmt(corr - norm) : '—'}</span>
          </div>
          <div class="cm-zscore-row">
            <span class="cm-zscore-key">Z-score (standard deviations)</span>
            <span class="cm-zscore-val ${absZ != null && absZ >= 1.5 ? 'down' : ''}">${z_score != null ? (z_score >= 0 ? '+' : '') + z_score.toFixed(2) + 'σ' : '—'}</span>
          </div>
          <div class="cm-zscore-row">
            <span class="cm-zscore-key">Signal threshold</span>
            <span class="cm-zscore-val" style="color:var(--text3,#6b7280);">|z| ≥ 1.5 = stretched · ≥ 2.5 = break</span>
          </div>
          ${signalText ? `<div class="cm-signal-banner ${signalClass}">${signalText}</div>` : ''}
        </div>
      </div>
    </div>`;

  document.body.appendChild(bd);
  bd.addEventListener('click', e => { if (e.target === bd) closeCorrModal(); });
  document.addEventListener('keydown', _cmKeydown);

  // Focus trap
  const closeBtn = document.getElementById('cm-close');
  if (closeBtn) closeBtn.focus();
}

function _cmKeydown(e) {
  if (e.key === 'Escape') closeCorrModal();
}

function closeCorrModal() {
  const bd = document.getElementById('cm-bd');
  if (bd) bd.remove();
  document.removeEventListener('keydown', _cmKeydown);
}

window.openCorrModal  = openCorrModal;
window.closeCorrModal = closeCorrModal;
