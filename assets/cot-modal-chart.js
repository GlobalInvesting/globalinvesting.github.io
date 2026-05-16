// COT MODAL CHART  v2.1 — inline-panel edition, terminal CSS variables
// COT MODAL CHART  v2.0 — LightweightCharts v5 (replaces Chart.js)
// File: assets/cot-modal-chart.js
// ═══════════════════════════════════════════════════════════════════════════

// ── CSS ─────────────────────────────────────────────────────────────────────
(function () {
  if (document.getElementById('cot-modal2-css')) return;
  const s = document.createElement('style');
  s.id = 'cot-modal2-css';
  s.textContent = `
#cot-bd {
  display:block!important;
}


#cot-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}
#cot-modal::before {
  display:none;
}
#cot-m-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 9px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg2);
}
#cot-m-title { font-size:14px;font-weight:600;color:var(--text);letter-spacing:-.01em;line-height:1.2;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
#cot-m-sub   { font-size:10px;color:var(--text2);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.02em; }
#cot-m-close {
  background:none;border:none;color:var(--text3,#4e5c70);font-size:16px;
  cursor:pointer;padding:3px 6px;border-radius:3px;line-height:1;
  transition:color .1s,background .1s;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
#cot-m-close:hover { color:var(--text);background:var(--bg3); }
#cot-m-metrics {
  display:grid;grid-template-columns:repeat(6,1fr);
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;
  background:var(--bg);
}
.cot-mm { padding:9px 14px;border-right:1px solid var(--border,#252d3d);display:flex;flex-direction:column;gap:1px; }
.cot-mm:last-child { border-right:none; }
.cot-mm-lbl { font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.09em; }
.cot-mm-val { font-size:13px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1;margin-top:2px; }
.cot-mm-sub { font-size:9px;color:var(--text2);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);margin-top:1px; }
#cot-m-tabs {
  display:flex;padding:0 14px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;overflow-x:auto;scrollbar-width:none;
  background:var(--bg2);
}
#cot-m-tabs::-webkit-scrollbar { display:none; }
.cot-tab {
  font-size:11px;font-weight:500;padding:9px 13px;cursor:pointer;
  color:var(--text2);border-bottom:2px solid transparent;
  transition:color .12s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.cot-tab:hover { color:var(--text2); }
.cot-tab.on { color:var(--text);border-bottom-color:var(--blue); }
#cot-m-body {
  flex:1;min-height:0;overflow-y:auto;
  padding:0;
  display:flex;flex-direction:column;
  background:var(--bg);
  scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent;
}
#cot-m-body::-webkit-scrollbar { width:3px!important; }
#cot-m-body::-webkit-scrollbar-track { background:transparent; }
#cot-m-body::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
#cot-m-body::-webkit-scrollbar-thumb:hover { background:var(--text2); }
#cot-m-body.cot-body--chart,
#cot-m-body.cot-body--overview { overflow-y:hidden; }
.cot-panel { display:none; }
.cot-panel.on { display:flex;flex:1;flex-direction:column;min-height:0; }
#p-history.on { display:block;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent; }
#p-history.on::-webkit-scrollbar { width:3px!important; }
#p-history.on::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
#p-overview.on { display:flex;flex:1;flex-direction:column;min-height:0;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent; }
#p-overview.on::-webkit-scrollbar { width:3px!important; }
#p-overview.on::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
/* Overview — KFV single-column layout */
#p-overview .cot-ov-sec {
  display:flex;align-items:center;justify-content:space-between;
  padding:6px 14px 5px;
  border-top:1px solid var(--border2,#1e2636);
  border-bottom:1px solid var(--border2,#1e2636);
  background:var(--bg,#0d1117);
  flex-shrink:0;
}
#p-overview .cot-ov-sec:first-child { border-top:none; }
#p-overview .cot-ov-sec-lbl { font-size:8.5px;font-weight:600;color:var(--text3,#4e5c70);text-transform:uppercase;letter-spacing:.1em;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
#p-overview .cot-ov-sec-note { font-size:8.5px;color:var(--text3,#4e5c70);opacity:.6;letter-spacing:.02em;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
/* Top row: Positioning + L/S side by side */
#p-overview .cot-ov-top-row { display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--border,#252d3d);flex-shrink:0; }
#p-overview .cot-ov-top-row > .cot-ov-half { padding:12px 14px; }
#p-overview .cot-ov-top-row > .cot-ov-half:first-child { border-right:1px solid var(--border,#252d3d); }
/* KFV rows */
#p-overview .cot-kfv { display:flex;align-items:center;padding:5px 14px;border-bottom:1px solid rgba(255,255,255,.04);min-height:28px;flex-shrink:0; }
#p-overview .cot-kfv:last-child { border-bottom:none; }
#p-overview .cot-kfv-key { font-size:10px;color:var(--text2,#8b949e);flex:1;min-width:0;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#p-overview .cot-kfv-bar { flex:0 0 80px;margin:0 10px;height:3px;background:rgba(255,255,255,.07);border-radius:2px;position:relative;flex-shrink:0; }
#p-overview .cot-kfv-bar-fill { position:absolute;left:0;top:0;height:100%;border-radius:2px; }
#p-overview .cot-kfv-val { font-size:11px;font-weight:600;text-align:right;flex-shrink:0;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#p-overview .cot-kfv-badge { font-size:8px;font-weight:700;letter-spacing:.06em;padding:2px 6px;border-radius:2px;margin-left:8px;flex-shrink:0;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
.cot-badge-l { background:rgba(38,166,154,.15);color:#26a69a; }
.cot-badge-s { background:rgba(239,83,80,.15);color:#ef5350; }
.cot-badge-n { background:rgba(88,166,255,.1);color:#58a6ff; }
.cot-badge-w { background:rgba(243,156,18,.12);color:#f39c12; }
/* Spark row */
#p-overview .cot-ov-spark-row { padding:10px 0 12px;flex-shrink:0; }
#p-overview .cot-ov-spark-top { display:flex;justify-content:space-between;margin-bottom:8px;padding:0 14px; }
#p-overview .cot-ov-spark-trend { font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#p-overview .cot-ov-spark-row .cot-spark { max-width:100%;width:100%;height:72px;display:block;box-sizing:border-box; }
#cot-ov-spark-lw { width:100%;max-width:100%!important;box-sizing:border-box;overflow:hidden; }

#p-net.on .cot-cw,
#p-split.on .cot-cw { flex:1;min-height:0;margin-bottom:0;border-bottom:none;display:flex;flex-direction:column; }
#p-net.on .cot-cw > .cot-chart-area,
#p-split.on .cot-cw > .cot-chart-area { flex:1;min-height:0;display:flex;flex-direction:column; }
#p-net.on .cot-cw > .cot-chart-area > .cot-lw-wrap,
#p-split.on .cot-cw > .cot-chart-area > .cot-lw-wrap { flex:1;min-height:0;height:100%; }
#p-participants .cot-chart-area { height:300px;position:relative; }
#p-participants.on { overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2,#2e3a50) transparent; }
#p-participants.on::-webkit-scrollbar { width:3px!important; }
#p-participants.on::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
.cot-chart-area { position:relative;flex:1;min-height:0;display:flex;flex-direction:column; }
.cot-lw-wrap { width:100%;flex:1;min-height:180px;position:relative;min-width:0; }
.cot-lw-tooltip {
  position:absolute;display:none;pointer-events:none;
  background:var(--bg2);border:1px solid var(--border2);border-radius:4px;
  padding:7px 11px;font-size:11px;line-height:1.55;
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
  color:var(--text);z-index:50;box-shadow:0 4px 16px rgba(0,0,0,.7);white-space:nowrap;
}
.cot-cw {
  background:var(--bg);
  border:none;
  border-radius:0;
  padding:14px;
  margin-bottom:0;
  border-bottom:1px solid var(--border,#252d3d);
  display:flex;flex-direction:column;
}
.cot-cw:last-child { border-bottom:none; }
.cot-ct {
  font-size:8.5px;color:var(--text3,#4e5c70);margin-bottom:10px;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
  letter-spacing:.07em;flex-shrink:0;text-transform:uppercase;font-weight:600;
}
.cot-gauge-track { height:6px;background:rgba(255,255,255,.06);border-radius:3px;position:relative;margin:10px 0 6px; }
.cot-gauge-fill { position:absolute;left:0;top:0;height:100%;border-radius:3px;background:linear-gradient(90deg,#ef5350 0%,#ef5350 20%,var(--orange) 20%,var(--orange) 27%,var(--blue) 27%,var(--blue) 73%,var(--orange) 73%,var(--orange) 80%,#26a69a 80%,#26a69a 100%);width:100%; }
.cot-gauge-pin {
  position:absolute;top:-4px;width:10px;height:10px;border-radius:50%;
  background:var(--text);border:2px solid var(--bg2);
  box-shadow:0 0 0 1px rgba(255,255,255,.2);
  transition:left .4s cubic-bezier(.25,.46,.45,.94);transform:translateX(-50%);
}
.cot-gauge-lbls { display:flex;justify-content:space-between;font-size:8px;color:var(--text2);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-ov-bignum { font-size:24px;font-weight:700;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);margin:4px 0 2px; }
.cot-ov-sub { font-size:9px;color:var(--text2);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-ls-row { display:flex;justify-content:space-between;align-items:center;gap:12px;margin:4px 0; }
.cot-ls-num { font-size:18px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-ls-vs { font-size:11px;color:var(--text2);flex-shrink:0; }
.cot-ls-bar { height:4px;background:rgba(255,255,255,.06);border-radius:2px;position:relative;overflow:hidden;margin-top:8px; }
.cot-ls-bar-fill { height:100%;border-radius:2px; }
.cot-sig-dot { display:inline-block;border-radius:50%;width:8px;height:8px;flex-shrink:0; }
.cot-spark { display:block;width:100%;max-width:200px;overflow:visible; }
.cot-tbl { width:100%;border-collapse:collapse;font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.cot-tbl thead th { text-align:right;color:var(--text2);font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:7px 10px;border-bottom:1px solid var(--border2);white-space:nowrap; }
.cot-tbl thead th:first-child { text-align:left; }
.cot-tbl tbody tr { transition:background .08s; }
.cot-tbl tbody tr:nth-child(even) td { background:rgba(255,255,255,.015); }
.cot-tbl tbody tr:hover td { background:rgba(88,166,255,.05); }
.cot-tbl td { text-align:right;padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text);vertical-align:middle;white-space:nowrap; }
.cot-tbl td:first-child { text-align:left;color:var(--text2); }
.cot-tbl tr:last-child td { border-bottom:none; }
.cu { color:#26a69a; }
.cd { color:#ef5350; }
.cn { color:var(--text2); }

@media (max-width:480px){
  #cot-modal{width:100%;height:93vh;border-radius:12px 12px 0 0;border-bottom:none;}
  #cot-m-metrics{grid-template-columns:repeat(3,1fr);}
  .cot-mm{padding:6px 10px;}.cot-mm-val{font-size:11px;}
  #cot-m-tabs{padding:0 8px;}.cot-tab{font-size:10px;padding:8px 8px;}
  #cot-m-body{padding:0;}.cot-cw{padding:10px 12px;}
  #p-overview .cot-ov-top-row{grid-template-columns:1fr;}
  #p-overview .cot-ov-top-row > .cot-ov-half:first-child{border-right:none;border-bottom:1px solid var(--border,#252d3d);}
  #p-overview .cot-kfv-bar{flex:0 0 50px;}
  .cot-ls-vs{display:block !important;text-align:center;}
  #p-history .cot-cw > div{overflow-x:auto;-webkit-overflow-scrolling:touch;}
  #p-history .cot-tbl{min-width:540px;font-size:9px;}
  #p-history .cot-tbl th,#p-history .cot-tbl td{padding:4px 5px;}
  #p-participants .cot-chart-area{height:260px;}
  #p-overview{flex:none;}
  #p-overview .cot-ov-bignum{font-size:20px !important;}
  .cot-gauge-lbls{font-size:7.5px;}.cot-ls-num{font-size:16px !important;}
  .cot-tbl{min-width:0;}.cot-tbl td,.cot-tbl th{font-size:9px;padding:4px 5px;white-space:nowrap;}
}
`;
  document.head.appendChild(s);
})();

// ── Utilities ─────────────────────────────────────────────────────────────────
const _monoF = "'JetBrains Mono','Courier New',monospace";

function _cotFmt(v) {
  if (v == null || isNaN(v)) return '—';
  return (v > 0 ? '+' : '') + Math.round(v).toLocaleString();
}
function _cotCls(v) {
  if (v == null || isNaN(v) || Math.abs(v) < 1) return '';
  return v > 0 ? 'cu' : 'cd';
}
function _calcZ(history) {
  const vals = history.map(h => h.levNet ?? ((h.levLong||0)-(h.levShort||0))).filter(v=>v!=null);
  if (vals.length < 4) return null;
  const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
  const std  = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/(vals.length-1));
  if (std < 1) return null;
  return (vals[vals.length-1]-mean)/std;
}
function _calcPct(history) {
  const vals = history.map(h => h.levNet ?? ((h.levLong||0)-(h.levShort||0))).filter(v=>v!=null);
  if (vals.length < 4) return null;
  const cur = vals[vals.length-1];
  return Math.round(vals.filter(v=>v<=cur).length/vals.length*100);
}
function _posLabel(z) {
  if (z == null) return {txt:'—',col:'var(--text2)'};
  if (z >  2)   return {txt:'Extreme Long',  col:'#ef5350'};
  if (z >  1.5) return {txt:'Crowded Long',  col:'#ff9800'};
  if (z >  0.5) return {txt:'Long',          col:'#26a69a'};
  if (z > -0.5) return {txt:'Neutral',        col:'#8b949e'};
  if (z > -1.5) return {txt:'Short',          col:'#26a69a'};
  if (z > -2)   return {txt:'Crowded Short',  col:'#ff9800'};
  return {txt:'Extreme Short',col:'#ef5350'};
}

// ── Overview helpers ──────────────────────────────────────────────────────────
function _cotSparkline(history, nWeeks) {
  const vals = history.slice(-nWeeks).map(h => h.levNet ?? ((h.levLong||0)-(h.levShort||0)));
  if (vals.length < 2) return '<div style="height:72px;display:flex;align-items:center;font-size:9px;color:#6e7681">Insufficient data</div>';

  const last = vals[vals.length - 1];
  const isPos = last >= 0;
  const lineCol = isPos ? '#26c6b0' : '#ef5350';
  const fillCol = isPos ? 'rgba(38,198,176,0.18)' : 'rgba(239,83,80,0.18)';

  const W = 1000, H = 72; // viewBox coords — scales to any container width
  const PAD = { t: 6, b: 6, l: 14, r: 14 }; // horizontal padding keeps line away from edges
  const minV = Math.min(...vals), maxV = Math.max(...vals);
  const range = maxV - minV || 1;

  const n = vals.length;
  const xOf = i => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const yOf = v => PAD.t + (1 - (v - minV) / range) * (H - PAD.t - PAD.b);

  const pts = vals.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(' ');
  const firstX = xOf(0).toFixed(1), lastX = xOf(n-1).toFixed(1), baseY = (H - PAD.b).toFixed(1);

  // Polyline points for the area fill (close at bottom)
  const areaPts = `${firstX},${baseY} ${pts} ${lastX},${baseY}`;

  // Crosshair dot — render at last point
  const dotX = xOf(n-1).toFixed(1), dotY = yOf(last).toFixed(1);

  return `<svg id="cot-ov-spark-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
    style="width:100%;height:72px;display:block;overflow:visible;"
    xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="cot-spark-grad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${lineCol}" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="${lineCol}" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <polygon points="${areaPts}" fill="url(#cot-spark-grad)" stroke="none"/>
    <polyline points="${pts}" fill="none" stroke="${lineCol}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${dotX}" cy="${dotY}" r="4" fill="${lineCol}" stroke="#131722" stroke-width="1.5"/>
  </svg>`;
}

function _buildSparklineChart(container, history, nWeeks) {
  // SVG sparkline is now built inline by _cotSparkline() — no LWC chart needed.
  // This function is kept as a no-op so existing callers don't error.
}

function _cotTrendLabel(history) {
  const n = Math.min(history.length,4);
  if (n<2) return '—';
  const recent = history.slice(-n).map(h=>h.levNet??((h.levLong||0)-(h.levShort||0)));
  let up=0,dn=0;
  for(let i=1;i<recent.length;i++){if(recent[i]>recent[i-1])up++;else if(recent[i]<recent[i-1])dn++;}
  if (up===n-1) return 'Accumulating · '+(n-1)+' consecutive '+(n-1===1?'week':'weeks');
  if (dn===n-1) return 'Distributing · '+(n-1)+' consecutive '+(n-1===1?'week':'weeks');
  return 'Mixed';
}

function _cotRangeCard(history, current) {
  const vals = history.map(h=>h.levNet??((h.levLong||0)-(h.levShort||0))).filter(v=>v!=null);
  if (vals.length<2) return '<div style="font-size:9px;color:#6e7681">Insufficient data</div>';
  const hi=Math.max(...vals),lo=Math.min(...vals);
  const pct = hi!==lo?Math.round((current-lo)/(hi-lo)*100):50;
  const bar=`<div style="margin:10px 0 8px;height:6px;background:rgba(255,255,255,.06);border-radius:3px;position:relative;">
    <div style="position:absolute;left:0;top:0;height:100%;width:${pct}%;background:var(--up,#26a69a);border-radius:3px;"></div>
    <div style="position:absolute;top:-4px;left:calc(${pct}% - 5px);width:10px;height:10px;border-radius:50%;background:#e6edf3;border:2px solid #161b22;box-shadow:0 0 0 1px rgba(255,255,255,.2)"></div>
  </div>`;
  const rows=[{label:vals.length+'w High',val:hi,cls:'cu'},{label:'Current',val:current,cls:_cotCls(current)},{label:vals.length+'w Low',val:lo,cls:'cd'}];
  return bar+rows.map(r=>`<div style="display:flex;justify-content:space-between;align-items:baseline;font-family:${_monoF};padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">
    <span style="font-size:11px;color:#6e7681">${r.label}</span>
    <span style="font-size:14px;font-weight:600" class="${r.cls}">${_cotFmt(r.val)}</span>
  </div>`).join('');
}

function _cotSignalSummary(net, amNet, ddNet, aligned, isCrowded) {
  const signals=[];
  if(net>0) signals.push({col:'#26a69a',text:'LF net long — bullish signal'});
  else if(net<0) signals.push({col:'#ef5350',text:'LF net short — bearish signal'});
  else signals.push({col:'#8b949e',text:'LF neutral'});
  if(amNet!=null){
    if(aligned) signals.push({col:'#26a69a',text:'LF/AM aligned — reinforced'});
    else signals.push({col:'#ff9800',text:'LF/AM diverging — exercise caution'});
  }
  if(isCrowded) signals.push({col:'#ff9800',text:'Crowded trade (z >= 1.5σ)'});
  else signals.push({col:'#26a69a',text:'Not crowded (z < 1.5σ)'});
  if(ddNet!=null){
    const contra=(net>0&&ddNet<0)||(net<0&&ddNet>0);
    if(contra) signals.push({col:'#ff9800',text:'Dealers contra-positioned'});
    else signals.push({col:'#8b949e',text:'Dealers aligned with LF'});
  }
  return signals.map(s=>`<div style="display:flex;align-items:center;gap:8px;font-size:11px;font-family:${_monoF};padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);">
    <span class="cot-sig-dot" style="background:${s.col}"></span>
    <span style="color:#8b949e">${s.text}</span>
  </div>`).join('');
}

// ── LWC chart instances ───────────────────────────────────────────────────────
const _cotLwCharts=[];
function _destroyCOTCharts(){
  _cotLwCharts.forEach(c=>{try{c.remove();}catch(_){}});
  _cotLwCharts.length=0;
}

function _lwOpts(W,H){
  return {
    width:W,height:H,
    layout:{background:{type:'solid',color:'#131722'},textColor:'#6e7681',fontFamily:_monoF,fontSize:10,attributionLogo:false},
    grid:{vertLines:{color:'rgba(255,255,255,0.04)'},horzLines:{color:'rgba(255,255,255,0.04)'}},
    crosshair:{
      mode:window.LightweightCharts?.CrosshairMode?.Normal??1,
      vertLine:{color:'rgba(255,255,255,0.2)',style:2,labelVisible:false},
      horzLine:{color:'rgba(255,255,255,0.12)',style:2,labelVisible:true},
    },
    rightPriceScale:{borderVisible:false,scaleMargins:{top:0.12,bottom:0.08}},
    timeScale:{borderVisible:false},
    handleScroll:{mouseWheel:true,pressedMouseMove:true},
    handleScale:{mouseWheel:true,pinch:true},
    localization:{priceFormatter:v=>v!=null?Math.round(v).toLocaleString():'—'},
  };
}

function _mkTooltip(container,lwChart,getSeries,fmtFn){
  const tip=document.createElement('div');
  tip.className='cot-lw-tooltip';
  container.style.position='relative';
  container.appendChild(tip);
  const TW=200,TM=12;
  lwChart.subscribeCrosshairMove(param=>{
    if(!param?.point||!param.seriesData){tip.style.display='none';return;}
    const html=fmtFn(param);
    if(!html){tip.style.display='none';return;}
    tip.innerHTML=html;tip.style.display='block';
    const cW=container.offsetWidth,cx=param.point.x,cy=param.point.y,th=tip.offsetHeight||50;
    const tx=(cx+TM+TW<=cW-4)?cx+TM:cx-TM-TW;
    const ty=(cy-th-TM>=4)?cy-th-TM:cy+TM;
    tip.style.left=Math.max(0,tx)+'px';tip.style.top=Math.max(0,ty)+'px';
  });
}

function _lwResize(container,lwChart){
  const apply=()=>{
    requestAnimationFrame(()=>{
      const rect=container.getBoundingClientRect();
      const h=Math.round(rect.height)||container.offsetHeight||container.parentElement?.getBoundingClientRect().height||240;
      const w=Math.round(rect.width)||container.offsetWidth||600;
      if(lwChart&&w>0&&h>10)lwChart.applyOptions({width:w,height:h});
    });
  };
  if(window.ResizeObserver){const ro=new ResizeObserver(()=>apply());ro.observe(container);container._lwRo=ro;}
  window.addEventListener('resize',apply);container._lwResize=apply;
  setTimeout(apply,60);setTimeout(apply,200);setTimeout(apply,500);
  return apply;
}

// ── Chart builders ────────────────────────────────────────────────────────────
function _buildNetChart(container,dates,netData,ccy){
  const LWC=window.LightweightCharts;if(!LWC||!container)return null;
  const W=container.offsetWidth||600,H=container.offsetHeight||container.parentElement?.offsetHeight||280;
  const chart=LWC.createChart(container,_lwOpts(W,H));
  _cotLwCharts.push(chart);
  const hist=chart.addSeries(LWC.HistogramSeries,{color:'#4f7fff',priceLineVisible:false,lastValueVisible:true,base:0});
  hist.setData(dates.map((d,i)=>({time:d,value:netData[i]??0,color:(netData[i]??0)>=0?'rgba(38,166,154,0.82)':'rgba(239,83,80,0.82)'})));
  chart.timeScale().fitContent();_lwResize(container,chart);
  _mkTooltip(container,chart,()=>hist,param=>{
    const v=param.seriesData.get(hist);if(!v)return null;
    const mon=typeof param.time==='string'?param.time.slice(0,7):'';
    const col=v.value>=0?'#26a69a':'#ef5350';
    return `<div style="font-size:9px;color:#6e7681;margin-bottom:4px;">${mon}</div><div>${ccy} LF Net &nbsp;<span style="color:${col};font-weight:700">${_cotFmt(v.value)}</span></div>`;
  });
  return chart;
}

function _buildSplitChart(container,dates,lngData,shrtData,ccy){
  const LWC=window.LightweightCharts;if(!LWC||!container)return null;
  const W=container.offsetWidth||600,H=container.offsetHeight||container.parentElement?.offsetHeight||280;
  const chart=LWC.createChart(container,_lwOpts(W,H));_cotLwCharts.push(chart);
  const lS=chart.addSeries(LWC.AreaSeries,{lineColor:'#26a69a',topColor:'rgba(38,166,154,0.15)',bottomColor:'rgba(38,166,154,0.01)',lineWidth:2,priceLineVisible:false,lastValueVisible:true,crosshairMarkerRadius:4});
  const sS=chart.addSeries(LWC.AreaSeries,{lineColor:'#ef5350',topColor:'rgba(239,83,80,0.15)',bottomColor:'rgba(239,83,80,0.01)',lineWidth:2,priceLineVisible:false,lastValueVisible:true,crosshairMarkerRadius:4});
  lS.setData(dates.map((d,i)=>({time:d,value:lngData[i]??0})));
  sS.setData(dates.map((d,i)=>({time:d,value:shrtData[i]??0})));
  chart.timeScale().fitContent();_lwResize(container,chart);
  _mkTooltip(container,chart,()=>lS,param=>{
    const lv=param.seriesData.get(lS),sv=param.seriesData.get(sS);if(!lv)return null;
    const mon=typeof param.time==='string'?param.time.slice(0,7):'';
    return `<div style="font-size:9px;color:#6e7681;margin-bottom:4px;">${mon}</div>`+
      `<div>Long &nbsp;<span style="color:#26a69a;font-weight:700">${lv.value!=null?Math.round(lv.value).toLocaleString():'—'}</span></div>`+
      (sv?`<div>Short<span style="color:#ef5350;font-weight:700"> ${sv.value!=null?Math.round(sv.value).toLocaleString():'—'}</span></div>`:'');
  });
  return chart;
}

function _buildParticipantsChart(container,dates,netData,amData,ddData,ccy){
  const LWC=window.LightweightCharts;if(!LWC||!container)return null;
  const W=container.offsetWidth||600,H=container.offsetHeight||280;
  const chart=LWC.createChart(container,_lwOpts(W,H));_cotLwCharts.push(chart);
  const lfS=chart.addSeries(LWC.LineSeries,{color:'#4f7fff',lineWidth:2,priceLineVisible:false,lastValueVisible:true,crosshairMarkerRadius:4});
  lfS.setData(dates.map((d,i)=>({time:d,value:netData[i]??null})).filter(p=>p.value!=null));
  let amS=null,ddS=null;
  if(amData.some(v=>v!=null)){
    amS=chart.addSeries(LWC.LineSeries,{color:'#ff9800',lineWidth:2,priceLineVisible:false,lastValueVisible:true,crosshairMarkerRadius:4});
    amS.setData(dates.map((d,i)=>({time:d,value:amData[i]})).filter(p=>p.value!=null));
  }
  if(ddData.some(v=>v!=null)){
    ddS=chart.addSeries(LWC.LineSeries,{color:'#ef5350',lineWidth:2,priceLineVisible:false,lastValueVisible:true,crosshairMarkerRadius:4});
    ddS.setData(dates.map((d,i)=>({time:d,value:ddData[i]})).filter(p=>p.value!=null));
  }
  chart.timeScale().fitContent();_lwResize(container,chart);
  const legendEl=container.parentElement?.querySelector('#cot-part-legend');
  if(legendEl){
    legendEl.innerHTML=[['Leveraged Funds','#4f7fff'],['Asset Managers','#ff9800'],['Dealers','#ef5350']].map(([lbl,col])=>
      `<span style="display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:18px;height:2px;background:${col};border-radius:1px"></span><span>${lbl}</span></span>`
    ).join('');
  }
  _mkTooltip(container,chart,()=>lfS,param=>{
    const lf=param.seriesData.get(lfS);if(!lf)return null;
    const mon=typeof param.time==='string'?param.time.slice(0,7):'';
    let html=`<div style="font-size:9px;color:#6e7681;margin-bottom:4px;">${mon}</div>`;
    html+=`<div style="color:#4f7fff">LF &nbsp;&nbsp;${_cotFmt(lf.value)}</div>`;
    if(amS){const av=param.seriesData.get(amS);if(av)html+=`<div style="color:#ff9800">AM &nbsp;&nbsp;${_cotFmt(av.value)}</div>`;}
    if(ddS){const dv=param.seriesData.get(ddS);if(dv)html+=`<div style="color:#ef5350">DD &nbsp;&nbsp;${_cotFmt(dv.value)}</div>`;}
    return html;
  });
  return chart;
}

// ── Main open function ────────────────────────────────────────────────────────
// Ensure LightweightCharts is loaded (mirrors dashboard.js loader — idempotent)
let _cotLwLibPromise = null;
function _cotEnsureLWLib() {
  if (window.LightweightCharts) return Promise.resolve();
  if (_cotLwLibPromise) return _cotLwLibPromise;
  _cotLwLibPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
    s.onload  = resolve;
    s.onerror = () => { _cotLwLibPromise = null; reject(new Error('LW lib load failed')); };
    document.head.appendChild(s);
  });
  return _cotLwLibPromise;
}

function openCOTModal(ccy,data){
  closeCOTModal();
  const history=Array.isArray(data.history)?[...data.history]:[];
  const net=data.netPosition||0,long_=data.longPositions||0,short_=data.shortPositions||0;
  const total=long_+short_,lPct=total>0?Math.round(long_/total*100):50;
  const amNet=data.assetManagerNet,ddNet=data.dealerNet,weekEnd=data.weekEnding||'',nWks=history.length;
  const zScore=_calcZ(history),pctHist=_calcPct(history),zInfo=_posLabel(zScore),isCrowded=Math.abs(zScore||0)>=1.5;
  let wow=null;
  if(history.length>=2){const prev=history[history.length-2];wow=net-(prev.levNet??((prev.levLong||0)-(prev.levShort||0)));}
  const netPctOI=total>0?(net/total*100):null;
  const netPctStr=netPctOI!=null?(netPctOI>0?'+':'')+netPctOI.toFixed(1)+'%':'—';
  const zStr=zScore!=null?(zScore>0?'+':'')+zScore.toFixed(2):'—';
  const pStr=pctHist!=null?pctHist+'%':'—';
  const zCol=isCrowded?'#ff9800':'#e6edf3';
  const lfDir=Math.sign(net),amDir=amNet!=null?Math.sign(amNet):0,aligned=lfDir!==0&&amDir!==0&&lfDir===amDir;
  const gaugeLeft=zScore!=null?((Math.max(-3,Math.min(3,zScore))+3)/6*90+5).toFixed(1)+'%':'50%';
  const dates=history.map(h=>{const d=h.weekEnding||'';return d.length===10?d:d.slice(0,10);});
  const netData=history.map(h=>h.levNet??((h.levLong||0)-(h.levShort||0)));
  const lngData=history.map(h=>h.levLong??null);
  const shrtData=history.map(h=>h.levShort??null);
  const amData=history.map(h=>h.assetManagerNet??null);
  const ddData=history.map(h=>h.dealerNet??null);

  const bd=document.createElement('div');
  bd.id='cot-bd';
  bd.innerHTML=`
<div id="cot-modal">
  <div id="cot-m-hd">
    <div>
      <div id="cot-m-title">CFTC Positioning · ${ccy} · Leveraged Funds</div>
      <div id="cot-m-sub">week ending ${weekEnd} · ${nWks}w history · CFTC TFF Disaggregated · Options+Futures Combined</div>
    </div>
    <button id="cot-m-close" onclick="closeCOTModal()" aria-label="Close">✕</button>
  </div>
  <div id="cot-m-metrics">
    <div class="cot-mm"><div class="cot-mm-lbl">Net LF</div><div class="cot-mm-val ${_cotCls(net)}">${_cotFmt(net)}</div><div class="cot-mm-sub">contracts</div></div>
    <div class="cot-mm"><div class="cot-mm-lbl">Long %</div><div class="cot-mm-val ${_cotCls(lPct-50)}">${lPct}%</div><div class="cot-mm-sub">of own OI</div></div>
    <div class="cot-mm"><div class="cot-mm-lbl">WoW Delta</div><div class="cot-mm-val ${_cotCls(wow)}">${_cotFmt(wow)}</div><div class="cot-mm-sub">weekly change</div></div>
    <div class="cot-mm"><div class="cot-mm-lbl">Net%OI</div><div class="cot-mm-val ${_cotCls(netPctOI)}">${netPctStr}</div><div class="cot-mm-sub">normalised</div></div>
    <div class="cot-mm"><div class="cot-mm-lbl">Z-Score</div><div class="cot-mm-val" style="color:${zCol}">${zStr}</div><div class="cot-mm-sub">${pStr} pctile</div></div>
    <div class="cot-mm"><div class="cot-mm-lbl">Positioning</div><div class="cot-mm-val" style="color:${zInfo.col};font-size:11px">${zInfo.txt}</div><div class="cot-mm-sub">${isCrowded?'crowded':'not crowded'}</div></div>
  </div>
  <div id="cot-m-tabs" role="tablist" aria-label="COT chart views">
    <div class="cot-tab on" data-tab="overview"    onclick="cotTab(this,'overview')"    role="tab" aria-selected="true">Overview</div>
    <div class="cot-tab"    data-tab="net"          onclick="cotTab(this,'net')"          role="tab" aria-selected="false">Net Position</div>
    <div class="cot-tab"    data-tab="split"        onclick="cotTab(this,'split')"        role="tab" aria-selected="false">Long / Short</div>
    <div class="cot-tab"    data-tab="participants" onclick="cotTab(this,'participants')" role="tab" aria-selected="false">Participants</div>
    <div class="cot-tab"    data-tab="history"      onclick="cotTab(this,'history')"      role="tab" aria-selected="false">History</div>
  </div>
  <div id="cot-m-body" class="cot-body--overview">
    <div id="p-overview" class="cot-panel on">

      <!-- TOP ROW: Positioning Gauge + L/S Split side by side -->
      <div class="cot-ov-top-row">
        <div class="cot-ov-half">
          <div class="cot-ct">POSITIONING · Z-SCORE</div>
          <div class="cot-ov-bignum" style="color:${zCol}">${zStr}σ <span style="font-size:12px;color:var(--text2);font-weight:400">· ${pStr} pctile</span></div>
          <div class="cot-ov-sub" style="margin-bottom:8px">${zInfo.txt} · ${nWks}w window</div>
          <div class="cot-gauge-track"><div class="cot-gauge-fill"></div><div id="cot-pin" class="cot-gauge-pin" style="left:50%"></div></div>
          <div class="cot-gauge-lbls"><span>Extreme Short</span><span>Neutral</span><span>Extreme Long</span></div>
        </div>
        <div class="cot-ov-half">
          <div class="cot-ct">LONG / SHORT SPLIT</div>
          <div class="cot-ls-row">
            <div><div class="cot-ls-num cu">${long_.toLocaleString()}</div><div class="cot-ov-sub">Longs</div></div>
            <div class="cot-ls-vs">vs</div>
            <div style="text-align:right"><div class="cot-ls-num cd">${short_.toLocaleString()}</div><div class="cot-ov-sub">Shorts</div></div>
          </div>
          <div class="cot-ls-bar"><div class="cot-ls-bar-fill" style="width:100%;background:linear-gradient(90deg,#26a69a ${lPct}%,#ef5350 ${lPct}%)"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:9px;font-family:${_monoF};color:#6e7681"><span>${lPct}% Long</span><span>${100-lPct}% Short</span></div>
        </div>
      </div>

      <!-- SECTION: KEY METRICS -->
      <div class="cot-ov-sec">
        <span class="cot-ov-sec-lbl">Key Metrics</span>
      </div>
      <div class="cot-kfv">
        <span class="cot-kfv-key">Net as % of Open Interest</span>
        <span class="cot-kfv-val ${_cotCls(netPctOI)}">${netPctStr}</span>
        ${Math.abs(netPctOI||0)>15?'<span class="cot-kfv-badge cot-badge-w">ELEVATED</span>':''}
      </div>
      <div class="cot-kfv">
        <span class="cot-kfv-key">Week-on-Week Change</span>
        <span class="cot-kfv-val ${_cotCls(wow)}">${_cotFmt(wow)}</span>
        ${wow!=null&&wow>0?'<span class="cot-kfv-badge cot-badge-l">BUYING</span>':wow!=null&&wow<0?'<span class="cot-kfv-badge cot-badge-s">SELLING</span>':''}
      </div>
      <div class="cot-kfv">
        <span class="cot-kfv-key">Crowd Alignment (LF + AM)</span>
        <span class="cot-kfv-val ${aligned?_cotCls(net):'cn'}">${aligned?(net>0?'Both Long':'Both Short'):'Diverging'}</span>
        ${isCrowded?'<span class="cot-kfv-badge cot-badge-w">CROWDED</span>':aligned?'<span class="cot-kfv-badge cot-badge-l">ALIGNED</span>':'<span class="cot-kfv-badge cot-badge-n">MIXED</span>'}
      </div>
      <div class="cot-kfv">
        <span class="cot-kfv-key">Trend Pattern (last 4w)</span>
        <span class="cot-kfv-val" style="color:var(--text)">${_cotTrendLabel(history).split(' · ')[0]}</span>
      </div>

      <!-- SECTION: PARTICIPANTS -->
      <div class="cot-ov-sec">
        <span class="cot-ov-sec-lbl">Participants · Current Week</span>
        <span class="cot-ov-sec-note">Net contracts by category</span>
      </div>
      ${(()=>{
        const maxAbs = Math.max(Math.abs(net), Math.abs(amNet||0), Math.abs(ddNet||0), 1);
        const pRow = (label, val) => {
          if (val == null) return '';
          const pct = Math.round(Math.abs(val) / maxAbs * 100);
          const col = val >= 0 ? '#26a69a' : '#ef5350';
          const dir = val > 0 ? 'LONG' : val < 0 ? 'SHORT' : 'FLAT';
          const badgeCls = val > 0 ? 'cot-badge-l' : val < 0 ? 'cot-badge-s' : 'cot-badge-n';
          return `<div class="cot-kfv">
            <span class="cot-kfv-key">${label}</span>
            <div class="cot-kfv-bar"><div class="cot-kfv-bar-fill" style="width:${pct}%;background:${col}"></div></div>
            <span class="cot-kfv-val ${_cotCls(val)}">${_cotFmt(val)}</span>
            <span class="cot-kfv-badge ${badgeCls}">${dir}</span>
          </div>`;
        };
        return pRow('Leveraged Funds', net) +
               pRow('Asset Managers', amNet) +
               pRow('Dealers / Intermediaries', ddNet);
      })()}

      <!-- SECTION: 52-WEEK RANGE -->
      <div class="cot-ov-sec">
        <span class="cot-ov-sec-lbl">52-Week Range</span>
      </div>
      ${(()=>{
        const vals = history.map(h=>h.levNet??((h.levLong||0)-(h.levShort||0))).filter(v=>v!=null);
        if (vals.length < 2) return '<div class="cot-kfv"><span class="cot-kfv-key" style="color:var(--text3)">Insufficient data</span></div>';
        const hi = Math.max(...vals), lo = Math.min(...vals);
        const pct = hi !== lo ? Math.round((net - lo) / (hi - lo) * 100) : 50;
        const rangeBar = `<div style="margin:6px 14px 2px;height:5px;background:rgba(255,255,255,.06);border-radius:3px;position:relative;">
          <div style="position:absolute;left:0;top:0;height:100%;width:${pct}%;background:var(--up,#26a69a);border-radius:3px;"></div>
          <div style="position:absolute;top:-3px;left:calc(${pct}% - 4px);width:8px;height:8px;border-radius:50%;background:var(--text,#e6edf3);border:2px solid var(--bg2,#161b22);box-shadow:0 0 0 1px rgba(255,255,255,.2)"></div>
        </div>`;
        return rangeBar +
          `<div class="cot-kfv"><span class="cot-kfv-key">${vals.length}w High</span><span class="cot-kfv-val cu">${_cotFmt(hi)}</span></div>` +
          `<div class="cot-kfv"><span class="cot-kfv-key">Current</span><span class="cot-kfv-val ${_cotCls(net)}">${_cotFmt(net)}</span><span class="cot-kfv-badge cot-badge-n">${pct}th PCTILE</span></div>` +
          `<div class="cot-kfv"><span class="cot-kfv-key">${vals.length}w Low</span><span class="cot-kfv-val cd">${_cotFmt(lo)}</span></div>`;
      })()}

      <!-- SECTION: 12-WEEK TREND -->
      <div class="cot-ov-sec">
        <span class="cot-ov-sec-lbl">12-Week Net Trend</span>
        <span class="cot-ov-sec-note">Leveraged Funds · weekly snapshot</span>
      </div>
      <div class="cot-ov-spark-row">
        <div class="cot-ov-spark-top">
          <span class="cot-ct" style="margin-bottom:0">Net contracts — last 12 weeks</span>
          <span class="cot-ov-spark-trend ${_cotCls(net)}">${_cotTrendLabel(history)}</span>
        </div>
        ${_cotSparkline(history,12)}
        <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:8.5px;color:var(--text3,#4e5c70);font-family:${_monoF};padding:0 14px"><span>12w ago</span><span>Now · ${_cotFmt(net)}</span></div>
      </div>

    </div>
    <div id="p-net" class="cot-panel">
      <div class="cot-cw"><div class="cot-ct">NET POSITION · LEVERAGED FUNDS · WEEKLY CONTRACTS</div><div class="cot-chart-area"><div class="cot-lw-wrap" id="cot-lw-net"></div></div></div>
    </div>
    <div id="p-split" class="cot-panel">
      <div class="cot-cw"><div class="cot-ct">LONGS VS SHORTS · LEVERAGED FUNDS · CONTRACTS</div><div class="cot-chart-area"><div class="cot-lw-wrap" id="cot-lw-split"></div></div></div>
    </div>
    <div id="p-participants" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">LF vs AM vs DEALER · NET BY CATEGORY</div>
        <div id="cot-part-legend" style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:8px;font-size:10px;font-family:${_monoF};color:#8b949e"></div>
        <div class="cot-chart-area"><div class="cot-lw-wrap" id="cot-lw-part"></div></div>
      </div>
      <div class="cot-cw"><div style="font-size:10px;color:#6e7681;font-family:${_monoF};line-height:1.7">
        <strong style="color:#8b949e">LF (Leveraged Funds):</strong> Hedge funds and CTAs. Primary speculative momentum signal.<br>
        <strong style="color:#8b949e">AM (Asset Managers):</strong> Mutual funds and pensions. Slow trend-followers. Confluence with LF = stronger signal.<br>
        <strong style="color:#8b949e">DD (Dealers):</strong> Market-makers. Typically contra-positioned to speculators. Useful contrarian signal.
      </div></div>
    </div>
    <div id="p-history" class="cot-panel">
      <div class="cot-cw">
        <div class="cot-ct">WEEKLY HISTORY · ${nWks} WEEKS</div>
        <div style="overflow-x:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.12) transparent">
          <table class="cot-tbl"><thead><tr>
            <th>Week</th><th>Net LF</th><th>WoW Δ</th><th>Longs</th><th>Shorts</th><th>Long%</th><th>Net%OI</th><th>AM Net</th><th>Dealer</th>
          </tr></thead><tbody id="cot-hist-body"></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>`;

  document.body.appendChild(bd);
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    const pin=document.getElementById('cot-pin');if(pin)pin.style.left=gaugeLeft;
    // Sparkline is now inline SVG — no async build needed
  }));

  const tbody=document.getElementById('cot-hist-body');
  if(tbody){
    const rev=[...history].reverse();
    tbody.innerHTML=rev.map((h,i)=>{
      const hNet=h.levNet??((h.levLong||0)-(h.levShort||0));
      const hL=h.levLong,hS=h.levShort,hTot=(hL||0)+(hS||0);
      const hLP=hTot>0?Math.round(hL/hTot*100):null;
      const hPctOI=hTot>0?(hNet/hTot*100):null;
      const prevH=rev[i+1];
      const hWow=prevH?hNet-(prevH.levNet??((prevH.levLong||0)-(prevH.levShort||0))):null;
      return`<tr style="${i===0?'background:rgba(255,255,255,.04)':''}">
        <td>${h.weekEnding}${i===0?' <span style="color:var(--up,#26a69a);font-size:9px">now</span>':''}</td>
        <td class="${_cotCls(hNet)}">${_cotFmt(hNet)}</td>
        <td class="${_cotCls(hWow)}">${hWow!=null?_cotFmt(hWow):'—'}</td>
        <td style="color:var(--up,#26a69a)">${hL!=null?hL.toLocaleString():'—'}</td>
        <td style="color:var(--down,#ef5350)">${hS!=null?hS.toLocaleString():'—'}</td>
        <td class="${_cotCls(hLP!=null?hLP-50:null)}">${hLP!=null?hLP+'%':'—'}</td>
        <td class="${_cotCls(hPctOI)}">${hPctOI!=null?(hPctOI>0?'+':'')+hPctOI.toFixed(1)+'%':'—'}</td>
        <td class="${_cotCls(h.assetManagerNet)}">${h.assetManagerNet!=null?_cotFmt(h.assetManagerNet):'—'}</td>
        <td class="${_cotCls(h.dealerNet)}">${h.dealerNet!=null?_cotFmt(h.dealerNet):'—'}</td>
      </tr>`;
    }).join('');
  }

  bd.addEventListener('click',e=>{if(e.target===bd)closeCOTModal();});
  const esc=e=>{if(e.key==='Escape')closeCOTModal();};
  document.addEventListener('keydown',esc);bd._esc=esc;
  bd._cotData={dates,netData,lngData,shrtData,amData,ddData,ccy,history};
}

function cotTab(el,tabId){
  document.querySelectorAll('.cot-tab').forEach(t=>{t.classList.remove('on');t.setAttribute('aria-selected','false');});
  document.querySelectorAll('.cot-panel').forEach(p=>p.classList.remove('on'));
  el.classList.add('on');el.setAttribute('aria-selected','true');
  const panel=document.getElementById('p-'+tabId);if(panel)panel.classList.add('on');
  const body=document.getElementById('cot-m-body');
  if(body){body.classList.toggle('cot-body--chart',tabId==='net'||tabId==='split');body.classList.toggle('cot-body--overview',tabId==='overview');}
  const bd=document.getElementById('cot-bd');if(!bd?._cotData)return;
  const d=bd._cotData;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    if(tabId==='net'){const w=document.getElementById('cot-lw-net');if(w&&!w._built){w._built=true;_buildNetChart(w,d.dates,d.netData,d.ccy);}else if(w&&w._lwResize)w._lwResize();}
    if(tabId==='split'){const w=document.getElementById('cot-lw-split');if(w&&!w._built){w._built=true;_buildSplitChart(w,d.dates,d.lngData,d.shrtData,d.ccy);}else if(w&&w._lwResize)w._lwResize();}
    if(tabId==='participants'){const w=document.getElementById('cot-lw-part');if(w&&!w._built){w._built=true;_buildParticipantsChart(w,d.dates,d.netData,d.amData,d.ddData,d.ccy);}else if(w&&w._lwResize)w._lwResize();}
    if(tabId==='overview'){ /* sparkline is inline SVG — no build needed */ }
    setTimeout(()=>{['cot-lw-net','cot-lw-split','cot-lw-part'].forEach(id=>{const w=document.getElementById(id);if(w&&w._lwResize)w._lwResize();});},120);
  }));
}

function closeCOTModal(){
  const bd=document.getElementById('cot-bd');
  if(bd){
    if(bd._esc)document.removeEventListener('keydown',bd._esc);
    document.querySelectorAll('.cot-lw-wrap').forEach(w=>{if(w._lwResize)window.removeEventListener('resize',w._lwResize);if(w._lwRo)w._lwRo.disconnect();});
    bd.remove();
  }
  _destroyCOTCharts();
}

window.openCOTModal=openCOTModal;window.closeCOTModal=closeCOTModal;window.cotTab=cotTab;
