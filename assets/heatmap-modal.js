// ═══════════════════════════════════════════════════════════════════════════
// CURRENCY STRENGTH HEATMAP MODAL  v1.0.0
// File: assets/heatmap-modal.js
// Loaded AFTER dashboard.js (see index.html)
//
// Public API (called from dashboard.js populateHeatmap):
//   openHeatmapModal(ccy, strengths, rtCache)
//   closeHeatmapModal()
//   hmTab(el, tabId)
//
// Pattern mirrors cb-rates-modal.js and corr-modal.js.
// All IDs prefixed hm- to avoid CSS collisions.
// ═══════════════════════════════════════════════════════════════════════════

(function () {

  // ── CSS ───────────────────────────────────────────────────────────────────
  if (document.getElementById('hm-modal-css')) return;
  const s = document.createElement('style');
  s.id = 'hm-modal-css';
  s.textContent = `
#hm-bd {
  position:fixed;inset:0;z-index:9300;
  background:rgba(0,0,0,.78);
  display:flex;align-items:center;justify-content:center;
  padding:12px;
  animation:hm-fi .15s ease;
}
@keyframes hm-fi { from{opacity:0} to{opacity:1} }
@keyframes hm-su { from{transform:translateY(14px);opacity:0} to{transform:none;opacity:1} }

#hm-modal {
  background:var(--bg,#131722);
  border:1px solid rgba(255,255,255,.1);
  border-radius:10px;
  width:min(780px,100%);
  max-height:90vh;
  display:flex;flex-direction:column;
  overflow:hidden;
  animation:hm-su .2s ease;
  font-family:var(--font-ui,'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif);
  color:var(--text,#d1d4dc);
}

#hm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 18px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
#hm-title { font-size:14px;font-weight:600;color:var(--text,#d1d4dc);letter-spacing:.01em;display:flex;align-items:center;gap:8px; }
#hm-title .fi { border-radius:2px;font-size:16px; }
#hm-sub   { font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
#hm-close {
  background:none;border:none;color:var(--text3,#6b7280);font-size:20px;
  cursor:pointer;padding:4px 8px;border-radius:4px;line-height:1;
  transition:color .1s,background .1s;
}
#hm-close:hover { color:var(--text,#d1d4dc);background:rgba(255,255,255,.08); }

/* Metrics strip */
#hm-metrics {
  display:grid;grid-template-columns:repeat(5,1fr);
  gap:1px;background:rgba(255,255,255,.05);
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
.hm-mm {
  background:var(--bg,#131722);padding:9px 14px;
  display:flex;flex-direction:column;gap:2px;
}
.hm-mm-lbl { font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.hm-mm-val { font-size:14px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.hm-mm-sub { font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace); }
.hm-mm-val.up   { color:var(--up,#26a69a); }
.hm-mm-val.down { color:var(--down,#ef5350); }
.hm-mm-val.flat { color:var(--text2,#787b86); }
.hm-mm-val.sm   { font-size:12px;padding-top:1px; }
.hm-mm-sub.up   { color:var(--up,#26a69a); }
.hm-mm-sub.down { color:var(--down,#ef5350); }

/* Tabs */
#hm-tabs {
  display:flex;padding:0 18px;
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;overflow-x:auto;scrollbar-width:none;
}
#hm-tabs::-webkit-scrollbar { display:none; }
.hm-tab {
  font-size:11px;padding:9px 13px;cursor:pointer;
  color:var(--text3,#6b7280);border-bottom:2px solid transparent;
  transition:color .1s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
.hm-tab:hover { color:var(--text2,#787b86); }
.hm-tab.on { color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff); }

/* Body */
#hm-body {
  flex:1;overflow-y:auto;min-height:0;
  scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,.12) transparent;
}
#hm-body::-webkit-scrollbar { width:5px; }
#hm-body::-webkit-scrollbar-track { background:transparent; }
#hm-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12);border-radius:3px; }
.hm-panel { display:none;padding:14px 16px; }
.hm-panel.on { display:block; }

/* Section label */
.hm-ct {
  font-size:10px;color:var(--text2,#787b86);
  font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.03em;
  text-transform:uppercase;margin-bottom:10px;
}
/* Card wrapper */
.hm-cw {
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.06);
  border-radius:6px;padding:12px 14px;margin-bottom:10px;
}

/* Pair breakdown table */
.hm-tbl {
  width:100%;border-collapse:collapse;
  font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);
}
.hm-tbl th {
  text-align:right;color:var(--text3,#6b7280);font-weight:400;
  font-size:9px;text-transform:uppercase;letter-spacing:.05em;
  padding:4px 8px 6px;border-bottom:1px solid rgba(255,255,255,.07);
}
.hm-tbl th:first-child { text-align:left; }
.hm-tbl td {
  text-align:right;padding:5px 8px;
  border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text,#d1d4dc);
}
.hm-tbl td:first-child { text-align:left;color:var(--text2,#787b86); }
.hm-tbl tr:last-child td { border-bottom:none; }
.hm-tbl tr:hover td { background:rgba(255,255,255,.03); }
.hm-tbl .sym { font-weight:600;color:var(--text,#d1d4dc); }
.imp-wrap { display:flex;align-items:center;gap:5px;justify-content:flex-end; }
.imp-bar-bg { width:40px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden; }
.imp-bar-fill { height:100%;border-radius:2px; }

/* Ranking rows */
.hm-rank-row { display:flex;align-items:center;gap:8px;margin-bottom:5px; }
.hm-rank-ccy { width:30px;font-size:10px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2,#787b86);text-align:right; }
.hm-rank-ccy.hl { color:var(--text,#d1d4dc); }
.hm-rank-bg { flex:1;height:14px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.hm-rank-fill { height:100%;border-radius:2px;transition:width .35s ease; }
.hm-rank-fill.hl   { background:var(--blue,#4f7fff); }
.hm-rank-fill.up   { background:rgba(38,166,154,.35); }
.hm-rank-fill.down { background:rgba(239,83,80,.30); }
.hm-rank-val { width:52px;text-align:right;font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2,#787b86); }

/* Session tab */
.sess-grid { display:grid;grid-template-columns:76px 1fr 54px;align-items:center;gap:5px 8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-size:10px; }
.sess-lbl { color:var(--text3,#6b7280);text-align:right;letter-spacing:.04em; }
.sess-lbl.hl { color:var(--blue,#4f7fff); }
.sess-track { height:10px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.sess-fill { height:100%;border-radius:2px; }
.sess-val { text-align:right; }

/* Correlations tab */
.corr-grid { display:grid;grid-template-columns:28px repeat(8,1fr);gap:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-size:9px; }
.corr-hdr { display:flex;align-items:center;justify-content:center;color:var(--text3,#6b7280);font-size:9px; }
.corr-cell { height:28px;border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:500; }

/* Footer */
#hm-footer {
  padding:8px 16px;
  border-top:1px solid rgba(255,255,255,.07);
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;
}
#hm-footer-meta { font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text3,#6b7280); }

/* Utility */
.up   { color:var(--up,#26a69a); }
.down { color:var(--down,#ef5350); }
.flat { color:var(--text2,#787b86); }
`;
  document.head.appendChild(s);

  // ── Currency metadata ────────────────────────────────────────────────────
  const CCY_META = {
    EUR: { flag: 'eu', full: 'Euro' },
    GBP: { flag: 'gb', full: 'Brit. Pound' },
    JPY: { flag: 'jp', full: 'Japanese Yen' },
    AUD: { flag: 'au', full: 'Aus. Dollar' },
    CHF: { flag: 'ch', full: 'Swiss Franc' },
    CAD: { flag: 'ca', full: 'Can. Dollar' },
    NZD: { flag: 'nz', full: 'NZ Dollar' },
    USD: { flag: 'us', full: 'US Dollar' },
  };

  // All 28 G8 pair definitions (same as populateHeatmap in dashboard.js)
  const PAIR_DEFS = [
    { id:'eurusd', base:'EUR', quote:'USD', sign:1 },
    { id:'gbpusd', base:'GBP', quote:'USD', sign:1 },
    { id:'audusd', base:'AUD', quote:'USD', sign:1 },
    { id:'nzdusd', base:'NZD', quote:'USD', sign:1 },
    { id:'usdjpy', base:'USD', quote:'JPY', sign:-1 },
    { id:'usdchf', base:'USD', quote:'CHF', sign:-1 },
    { id:'usdcad', base:'USD', quote:'CAD', sign:-1 },
    { id:'eurgbp', base:'EUR', quote:'GBP', sign:1 },
    { id:'eurjpy', base:'EUR', quote:'JPY', sign:1 },
    { id:'eurchf', base:'EUR', quote:'CHF', sign:1 },
    { id:'eurcad', base:'EUR', quote:'CAD', sign:1 },
    { id:'euraud', base:'EUR', quote:'AUD', sign:1 },
    { id:'eurnzd', base:'EUR', quote:'NZD', sign:1 },
    { id:'gbpjpy', base:'GBP', quote:'JPY', sign:1 },
    { id:'gbpchf', base:'GBP', quote:'CHF', sign:1 },
    { id:'gbpcad', base:'GBP', quote:'CAD', sign:1 },
    { id:'gbpaud', base:'GBP', quote:'AUD', sign:1 },
    { id:'gbpnzd', base:'GBP', quote:'NZD', sign:1 },
    { id:'audjpy', base:'AUD', quote:'JPY', sign:1 },
    { id:'audchf', base:'AUD', quote:'CHF', sign:1 },
    { id:'audcad', base:'AUD', quote:'CAD', sign:1 },
    { id:'audnzd', base:'AUD', quote:'NZD', sign:1 },
    { id:'nzdjpy', base:'NZD', quote:'JPY', sign:1 },
    { id:'nzdchf', base:'NZD', quote:'CHF', sign:1 },
    { id:'nzdcad', base:'NZD', quote:'CAD', sign:1 },
    { id:'cadjpy', base:'CAD', quote:'JPY', sign:1 },
    { id:'cadchf', base:'CAD', quote:'CHF', sign:1 },
    { id:'chfjpy', base:'CHF', quote:'JPY', sign:1 },
  ];

  // Session windows (UTC hours, start inclusive)
  const SESSIONS = [
    { name:'Sydney',  utcStart:21, utcEnd:6  },
    { name:'Tokyo',   utcStart:0,  utcEnd:9  },
    { name:'London',  utcStart:7,  utcEnd:16 },
    { name:'New York',utcStart:12, utcEnd:21 },
  ];

  // ── State ────────────────────────────────────────────────────────────────
  let _ccy      = null;
  let _strengths = null;
  let _rtCache  = null;
  let _driversCache  = null;   // { generated_at, drivers: { USD: "...", EUR: "...", ... } }
  let _driversFetched = false;
  let _sessionCtxCache = null; // { generated_at, sessions: { EUR: { Sydney: "...", ... }, ... } }
  let _sessionCtxFetched = false;

  // Fetch currency-drivers.json once per page load (lazy, on first modal open).
  // Falls back silently — the drivers note is additive, never blocking.
  function fetchDrivers() {
    if (_driversFetched) return;
    _driversFetched = true;
    fetch('./ai-analysis/currency-drivers.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.drivers && typeof data.drivers === 'object') {
          _driversCache = data;
        }
      })
      .catch(() => { /* silent fallback — drivers are additive */ });
  }

  // Fetch session-context.json once per page load (lazy, on first modal open).
  // Falls back silently — session notes are additive, never blocking.
  function fetchSessionContext() {
    if (_sessionCtxFetched) return;
    _sessionCtxFetched = true;
    fetch('./ai-analysis/session-context.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.sessions && typeof data.sessions === 'object') {
          _sessionCtxCache = data;
        }
      })
      .catch(() => { /* silent fallback */ });
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  function fmt2(v) {
    if (v == null || isNaN(v)) return '—';
    const s = v >= 0 ? '+' : '';
    return s + v.toFixed(2) + '%';
  }

  function fmtPrice(v) {
    if (v == null || isNaN(v)) return '—';
    return v >= 100 ? v.toFixed(3) : v.toFixed(5);
  }

  function pctClass(v) {
    if (v == null || isNaN(v)) return 'flat';
    return v > 0 ? 'up' : v < 0 ? 'down' : 'flat';
  }

  function currentSessionName() {
    const h = new Date().getUTCHours();
    // Active session: London if 07-16, New York if 12-21, Tokyo if 00-09, Sydney otherwise
    if (h >= 7 && h < 16)  return 'London';
    if (h >= 12 && h < 21) return 'New York';
    if (h >= 0 && h < 9)   return 'Tokyo';
    return 'Sydney';
  }

  // ── Build HTML ───────────────────────────────────────────────────────────
  function buildModal() {
    if (document.getElementById('hm-bd')) return;
    const el = document.createElement('div');
    el.id = 'hm-bd';
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    el.setAttribute('aria-label', 'Currency Strength Breakdown');
    el.innerHTML = `
<div id="hm-modal">
  <div id="hm-hd">
    <div>
      <div id="hm-title"></div>
      <div id="hm-sub">28-pair equal-weighted model · 8 G8 currencies · yfinance · ~5min delay</div>
    </div>
    <button id="hm-close" aria-label="Close" title="Close">&#10005;</button>
  </div>
  <div id="hm-metrics">
    <div class="hm-mm">
      <div class="hm-mm-lbl">Composite</div>
      <div class="hm-mm-val" id="hm-m-composite">—</div>
      <div class="hm-mm-sub">avg vs 7 pairs</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Rank</div>
      <div class="hm-mm-val flat" id="hm-m-rank">—</div>
      <div class="hm-mm-sub">of G8</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Pairs won</div>
      <div class="hm-mm-val flat" id="hm-m-won">—</div>
      <div class="hm-mm-sub">gaining vs</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Strongest vs</div>
      <div class="hm-mm-val sm flat" id="hm-m-strong">—</div>
      <div class="hm-mm-sub up" id="hm-m-strong-sub">—</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Weakest vs</div>
      <div class="hm-mm-val sm flat" id="hm-m-weak">—</div>
      <div class="hm-mm-sub down" id="hm-m-weak-sub">—</div>
    </div>
  </div>
  <div id="hm-tabs" role="tablist" aria-label="Heatmap breakdown tabs">
    <div class="hm-tab on" role="tab" aria-selected="true"  data-tab="breakdown"    onclick="hmTab(this,'breakdown')">Pair Breakdown</div>
    <div class="hm-tab"    role="tab" aria-selected="false" data-tab="session"      onclick="hmTab(this,'session')">Session</div>
    <div class="hm-tab"    role="tab" aria-selected="false" data-tab="correlations" onclick="hmTab(this,'correlations')">Correlations</div>
  </div>
  <div id="hm-body">
    <div class="hm-panel on" id="hm-p-breakdown">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-pairs-title">DIRECT PAIRS · INTRADAY % CHANGE · vs PREV CLOSE</div>
        <table class="hm-tbl" aria-label="Direct pairs for selected currency">
          <thead>
            <tr>
              <th scope="col">Pair</th>
              <th scope="col">Close</th>
              <th scope="col">Prev close</th>
              <th scope="col">% chg</th>
              <th scope="col">Impact</th>
              <th scope="col">Session range</th>
            </tr>
          </thead>
          <tbody id="hm-pair-tbody"></tbody>
        </table>
      </div>
      <div class="hm-cw">
        <div class="hm-ct">FULL RANKING · ALL 8 G8 CURRENCIES · COMPOSITE STRENGTH</div>
        <div id="hm-ranking-rows"></div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-session">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-sess-title">COMPOSITE STRENGTH BY SESSION</div>
        <div id="hm-sess-content"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct">SESSION CONTEXT</div>
        <div id="hm-sess-notes" style="font-size:11px;color:var(--text2,#787b86);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.7;"></div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-correlations">
      <div class="hm-cw">
        <div class="hm-ct">INTRADAY STRENGTH DIFFERENTIAL · ALL G8 PAIRS · % CHG vs PREV CLOSE</div>
        <div id="hm-corr-matrix"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct" id="hm-drivers-title">STRENGTH DRIVERS · TOP 3 PAIRS BY CONTRIBUTION</div>
        <div id="hm-drivers"></div>
      </div>
    </div>
  </div>
  <div id="hm-footer">
    <div id="hm-footer-meta">yfinance · ~5min delay · 28-pair equal-weighted model</div>
  </div>
</div>`;
    document.body.appendChild(el);
    document.getElementById('hm-close').addEventListener('click', closeHeatmapModal);
    el.addEventListener('click', function(e) { if (e.target === el) closeHeatmapModal(); });
    document.addEventListener('keydown', _onKey);
  }

  function _onKey(e) {
    if (e.key === 'Escape') closeHeatmapModal();
  }

  // ── Populate ─────────────────────────────────────────────────────────────
  function populateMetrics(ccy, strengths, rtCache) {
    const sorted = [...strengths].sort((a,b) => b.pct - a.pct);
    const rank   = sorted.findIndex(s => s.ccy === ccy) + 1;
    const self   = strengths.find(s => s.ccy === ccy);
    if (!self) return;

    // Pairs for this currency
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    let won = 0;
    let bestPair = null, bestPct = -Infinity;
    let worstPair = null, worstPct = Infinity;

    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
      if (impact > 0) won++;
      const opp = p.base === ccy ? p.quote : p.base;
      if (impact > bestPct)  { bestPct  = impact; bestPair  = { pair: p.id.toUpperCase(), opp, pct: impact }; }
      if (impact < worstPct) { worstPct = impact; worstPair = { pair: p.id.toUpperCase(), opp, pct: impact }; }
    });

    const compositeEl = document.getElementById('hm-m-composite');
    const v = self.pct;
    compositeEl.textContent = fmt2(v);
    compositeEl.className = 'hm-mm-val ' + pctClass(v);

    document.getElementById('hm-m-rank').textContent = '#' + rank + ' / 8';
    document.getElementById('hm-m-won').textContent  = won + ' / ' + myPairs.length;

    const strongEl    = document.getElementById('hm-m-strong');
    const strongSubEl = document.getElementById('hm-m-strong-sub');
    const weakEl      = document.getElementById('hm-m-weak');
    const weakSubEl   = document.getElementById('hm-m-weak-sub');

    if (bestPair) {
      strongEl.textContent    = bestPair.opp;
      strongSubEl.textContent = fmt2(bestPair.pct);
    }
    if (worstPair) {
      weakEl.textContent    = worstPair.opp;
      weakSubEl.textContent = fmt2(worstPair.pct);
    }
  }

  function populateBreakdown(ccy, strengths, rtCache) {
    document.getElementById('hm-pairs-title').textContent =
      ccy + ' DIRECT PAIRS · INTRADAY % CHANGE · vs PREV CLOSE';

    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const impacts = [];

    myPairs.forEach(p => {
      const d = rtCache[p.id];
      const isCcyBase = p.base === ccy;
      const opp = isCcyBase ? p.quote : p.base;
      const rawPct = d?.pct ?? null;
      // impact on the selected ccy: positive = ccy gained vs opp
      const impact = rawPct != null ? rawPct * p.sign * (isCcyBase ? 1 : -1) : null;
      const close  = isCcyBase ? (d?.close ?? null) : (d?.close != null ? 1/d.close : null);
      const open   = isCcyBase ? (d?.open  ?? null) : (d?.open  != null ? 1/d.open  : null);
      const hi     = isCcyBase ? (d?.high  ?? null) : (d?.high  != null ? 1/d.high  : null);
      const lo     = isCcyBase ? (d?.low   ?? null) : (d?.low   != null ? 1/d.low   : null);
      const label  = isCcyBase
        ? (p.base + '/' + p.quote)
        : (p.quote + '/' + p.base);   // show ccy first
      impacts.push({ label, opp, close, open, hi, lo, impact, rawPct });
    });

    // Sort: biggest positive impact first
    impacts.sort((a,b) => (b.impact??-99) - (a.impact??-99));
    const maxImp = Math.max(...impacts.map(i => Math.abs(i.impact ?? 0)), 0.001);

    const tbody = document.getElementById('hm-pair-tbody');
    tbody.innerHTML = impacts.map(r => {
      const iCls  = pctClass(r.impact);
      const rng   = (r.hi != null && r.lo != null)
        ? fmtPrice(r.lo) + ' – ' + fmtPrice(r.hi)
        : '—';
      const barW  = r.impact != null ? Math.round(Math.abs(r.impact)/maxImp*100) : 0;
      const barClr = r.impact != null && r.impact >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
      return `<tr>
        <td><span class="sym">${r.label}</span></td>
        <td>${fmtPrice(r.close)}</td>
        <td>${fmtPrice(r.open)}</td>
        <td class="${iCls}">${fmt2(r.impact)}</td>
        <td><div class="imp-wrap">
          <span class="${iCls}">${fmt2(r.impact)}</span>
          <div class="imp-bar-bg"><div class="imp-bar-fill" style="width:${barW}%;background:${barClr}"></div></div>
        </div></td>
        <td style="font-size:9px;color:var(--text3)">${rng}</td>
      </tr>`;
    }).join('');

    // Ranking
    const sorted   = [...strengths].sort((a,b) => b.pct - a.pct);
    const maxAbsPct = Math.max(...sorted.map(s => Math.abs(s.pct)), 0.001);
    const container = document.getElementById('hm-ranking-rows');
    container.innerHTML = '';
    sorted.forEach(s => {
      const isHL  = s.ccy === ccy;
      const cls   = isHL ? 'hl' : pctClass(s.pct);
      const fillW = Math.round(Math.abs(s.pct) / maxAbsPct * 100);
      const row   = document.createElement('div');
      row.className = 'hm-rank-row';
      row.innerHTML = `
        <div class="hm-rank-ccy${isHL?' hl':''}">${s.ccy}</div>
        <div class="hm-rank-bg">
          <div class="hm-rank-fill ${cls}" style="width:0" data-w="${fillW}"></div>
        </div>
        <div class="hm-rank-val ${pctClass(s.pct)}">${fmt2(s.pct)}</div>`;
      container.appendChild(row);
    });
    // Animate bars after paint
    requestAnimationFrame(() => {
      container.querySelectorAll('.hm-rank-fill').forEach(el => {
        el.style.width = el.dataset.w + '%';
      });
    });
  }

  // Convert a UTC hour to local HH:MM string (respects user's timezone)
  function utcHourToLocalStr(utcHour) {
    const d = new Date();
    d.setUTCHours(utcHour, 0, 0, 0);
    return d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: false });
  }

  // Returns the user's timezone abbreviation (e.g. "EST", "GMT+3")
  function localTzAbbr() {
    return new Date().toLocaleTimeString('en', { timeZoneName: 'short' }).split(' ').pop() || 'LT';
  }

  // Returns true if a session has already opened today (UTC day boundary).
  // Sydney spans midnight — treat as always opened.
  function sessionHasOpened(sess) {
    const h = new Date().getUTCHours();
    if (sess.name === 'Sydney') return true; // always show — spans midnight
    return h >= sess.utcStart;
  }

  function populateSession(ccy, rtCache) {
    const tzAbbr = localTzAbbr();
    document.getElementById('hm-sess-title').textContent =
      ccy + ' COMPOSITE STRENGTH BY SESSION · ' + tzAbbr + ' REFERENCE';

    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const activeSess = currentSessionName();

    // Estimate per-session composite: use full intraday % scaled by session weight.
    // Only sessions that have already opened today receive a value; UPCOMING sessions
    // show no bar and display "—" to avoid fabricating data for the future.
    const weights = { 'New York': 0.38, 'London': 0.35, 'Tokyo': 0.18, 'Sydney': 0.09 };
    const sessionData = SESSIONS.map(sess => {
      const isActive = sess.name === activeSess;
      const opened   = sessionHasOpened(sess);
      let pct = null;
      if (opened) {
        let sum = 0;
        myPairs.forEach(p => {
          const d = rtCache[p.id];
          if (!d || d.pct == null) return;
          const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
          sum += impact * (weights[sess.name] || 0.1);
        });
        pct = myPairs.length > 0 ? sum / myPairs.length : 0;
      }
      return { ...sess, pct, isActive, opened };
    });

    const openedPcts = sessionData.filter(s => s.opened && s.pct != null).map(s => Math.abs(s.pct));
    const maxAbs = openedPcts.length > 0 ? Math.max(...openedPcts, 0.001) : 0.001;
    const grid   = document.createElement('div');
    grid.className = 'sess-grid';

    sessionData.forEach(s => {
      const lbl = document.createElement('div');
      lbl.className = 'sess-lbl' + (s.isActive ? ' hl' : '');

      let labelText = s.name.toUpperCase();
      if (s.isActive) {
        labelText += ' \u25CF';
      } else if (!s.opened) {
        labelText += ' \xb7 opens ' + utcHourToLocalStr(s.utcStart);
      }
      lbl.textContent = labelText;

      const track = document.createElement('div');
      track.className = 'sess-track';

      const val = document.createElement('div');

      if (!s.opened || s.pct == null) {
        lbl.style.opacity = '0.45';
        track.style.opacity = '0.2';
        val.className = 'sess-val flat';
        val.textContent = '\u2014';
        val.style.opacity = '0.45';
      } else {
        const pos = s.pct >= 0;
        const cls = pos ? 'up' : 'down';
        const clr = pos ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
        const w   = Math.round(Math.abs(s.pct) / maxAbs * 100);
        const fill = document.createElement('div');
        fill.className = 'sess-fill';
        fill.style.cssText = 'width:' + w + '%;background:' + clr;
        track.appendChild(fill);
        val.className = 'sess-val ' + cls;
        val.textContent = fmt2(s.pct);
      }

      grid.appendChild(lbl);
      grid.appendChild(track);
      grid.appendChild(val);
    });

    const content = document.getElementById('hm-sess-content');
    content.innerHTML = '';
    content.appendChild(grid);

    // Session context notes — Groq-generated if available, fallback to basic stats
    const notes = document.getElementById('hm-sess-notes');
    const _now    = new Date();
    const localHH = String(_now.getHours()).padStart(2,'0');
    const localMM = String(_now.getMinutes()).padStart(2,'0');
    const localStr = localHH + ':' + localMM;
    const tzAbbr  = localTzAbbr();

    // Check if Groq session context is available for this currency
    const groqSessions = _sessionCtxCache && _sessionCtxCache.sessions
      ? _sessionCtxCache.sessions[ccy]
      : null;

    if (groqSessions && Object.keys(groqSessions).length >= 3) {
      // Render Groq session notes in the same style as the model
      const sessOrder = ['Sydney', 'Tokyo', 'London', 'New York'];
      notes.innerHTML = sessOrder.map(sName => {
        const isActive = sName === activeSess;
        const note = groqSessions[sName] || '—';
        return (
          '<div style="margin-bottom:4px' + (isActive ? ';color:var(--text,#d1d4dc)' : '') + '">' +
          '<span style="color:' + (isActive ? 'var(--blue,#4f7fff)' : 'var(--text3,#6b7280)') + ';' +
          'min-width:72px;display:inline-block;font-weight:' + (isActive ? '600' : '400') + '">' +
          sName.toUpperCase() + (isActive ? ' \u25CF' : '') +
          '</span> ' + note + '</div>'
        );
      }).join('') +
      '<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">' +
      'AI Analytics \xb7 ~5min delay &nbsp;|&nbsp; ' + tzAbbr + ' ' + localStr + '</div>';
    } else {
      // Fallback: basic intraday stats (no Groq data yet)
      notes.innerHTML =
        `Active session: <span class="up">${activeSess}</span> &nbsp;|&nbsp; ` +
        `${tzAbbr} ${localStr}<br>` +
        `Session attribution weighted by typical volume distribution.<br>` +
        `Intraday strength: <span class="${pctClass(0)}" id="hm-sess-intra">—</span>`;

      // Update intraday note
      let sum = 0, cnt = 0;
      myPairs.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct == null) return;
        const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
        sum += impact; cnt++;
      });
      const intra = cnt > 0 ? sum / cnt : null;
      const el = document.getElementById('hm-sess-intra');
      if (el && intra != null) {
        el.textContent = fmt2(intra);
        el.className   = pctClass(intra);
      }
    }
  }

  function populateCorrelations(ccy, strengths, rtCache) {
    document.getElementById('hm-drivers-title').textContent =
      ccy + ' STRENGTH DRIVERS · TOP 3 PAIRS BY CONTRIBUTION';

    const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD'];

    // Build 8×8 strength differential matrix
    // cell[i][j] = strengths[i] - strengths[j]  (positive = row ccy stronger)
    const pctMap = {};
    strengths.forEach(s => { pctMap[s.ccy] = s.pct; });

    const matrix = document.getElementById('hm-corr-matrix');
    const grid   = document.createElement('div');
    grid.className = 'corr-grid';

    // Empty top-left
    const empty = document.createElement('div');
    grid.appendChild(empty);

    // Column headers
    ccys.forEach(c => {
      const h = document.createElement('div');
      h.className = 'corr-hdr';
      h.textContent = c;
      grid.appendChild(h);
    });

    // Rows
    ccys.forEach(rowCcy => {
      const rLbl = document.createElement('div');
      rLbl.className = 'corr-hdr';
      rLbl.textContent = rowCcy;
      grid.appendChild(rLbl);

      ccys.forEach(colCcy => {
        const cell = document.createElement('div');
        cell.className = 'corr-cell';

        if (rowCcy === colCcy) {
          cell.style.background = 'rgba(255,255,255,.04)';
          cell.textContent = '·';
          cell.style.color = 'var(--text3,#6b7280)';
        } else {
          const diff = (pctMap[rowCcy] ?? 0) - (pctMap[colCcy] ?? 0);
          const abs  = Math.abs(diff);
          const isHL = rowCcy === ccy || colCcy === ccy;
          // Color intensity — max ~0.5% diff = full saturation
          const alpha = Math.min(abs / 0.5, 1) * 0.7;
          if (diff > 0.02) {
            cell.style.background = `rgba(38,166,154,${alpha})`;
            cell.style.color = alpha > 0.4 ? '#fff' : 'var(--up,#26a69a)';
          } else if (diff < -0.02) {
            cell.style.background = `rgba(239,83,80,${alpha})`;
            cell.style.color = alpha > 0.4 ? '#fff' : 'var(--down,#ef5350)';
          } else {
            cell.style.background = 'rgba(255,255,255,.03)';
            cell.style.color = 'var(--text3,#6b7280)';
          }
          // Highlight row/col for selected ccy
          if (isHL) {
            cell.style.outline = '1px solid rgba(79,127,255,.3)';
          }
          cell.textContent = diff >= 0 ? '+' + diff.toFixed(1) : diff.toFixed(1);
          cell.title = `${rowCcy} vs ${colCcy}: ${fmt2(diff)}`;
        }
        grid.appendChild(cell);
      });
    });

    matrix.innerHTML = '';
    matrix.appendChild(grid);

    // Top 3 drivers
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const driven  = [];
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
      const opp    = p.base === ccy ? p.quote : p.base;
      const label  = p.base === ccy ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
      const canon  = p.base + '/' + p.quote;   // canonical key matching currency-drivers.json
      driven.push({ label, opp, impact, canon });
    });
    driven.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));
    const top3 = driven.slice(0,3);

    const driversEl = document.getElementById('hm-drivers');
    if (top3.length === 0) {
      driversEl.innerHTML = '<div style="font-size:11px;color:var(--text3,#6b7280);font-family:var(--font-mono)">No RT data available</div>';
      return;
    }

    // Per-pair AI notes from ai-analysis/currency-drivers.json
    // Structure: { drivers: { EUR: { "EUR/JPY": "...", "EUR/GBP": "...", ... }, ... } }
    const ccyNotes = (_driversCache && _driversCache.drivers && _driversCache.drivers[ccy])
      ? _driversCache.drivers[ccy]
      : null;

    driversEl.innerHTML = top3.map((d,i) => {
      const cls    = pctClass(d.impact);
      const note   = ccyNotes ? (ccyNotes[d.label] || ccyNotes[d.canon] || null) : null;
      const noteEl = note
        ? `<div style="font-size:10px;color:var(--text2,#787b86);font-family:var(--font-mono);margin-top:3px;line-height:1.5;">${note}</div>`
        : '';
      return `<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:${note ? 10 : 6}px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);">
        <div style="font-size:11px;font-weight:600;color:var(--text);width:70px;padding-top:1px">${d.label}</div>
        <div style="flex:1">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:11px;font-weight:600" class="${cls}">${fmt2(d.impact)}</span>
            <span style="font-size:11px;color:var(--text2,#787b86)">vs ${d.opp}</span>
          </div>
          ${noteEl}
        </div>
      </div>`;
    }).join('') + (ccyNotes
      ? `<div style="margin-top:4px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">AI Analytics · ~5min delay</div>`
      : '');
  }

  // ── Public API ────────────────────────────────────────────────────────────

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    _ccy       = ccy;
    _strengths = strengths;
    _rtCache   = rtCache;

    buildModal();

    // Set title with flag
    const meta  = CCY_META[ccy] || { flag: 'un', full: ccy };
    const titleEl = document.getElementById('hm-title');
    titleEl.innerHTML =
      `<span class="fi fi-${meta.flag}" style="border-radius:2px;font-size:15px;vertical-align:middle;"></span>` +
      `${ccy} \u2014 Currency Strength Breakdown`;

    // Reset to first tab
    document.querySelectorAll('.hm-tab').forEach(t => {
      t.classList.toggle('on', t.dataset.tab === 'breakdown');
      t.setAttribute('aria-selected', t.dataset.tab === 'breakdown' ? 'true' : 'false');
    });
    document.querySelectorAll('.hm-panel').forEach(p => {
      p.classList.toggle('on', p.id === 'hm-p-breakdown');
    });

    populateMetrics(ccy, strengths, rtCache);
    populateBreakdown(ccy, strengths, rtCache);
    fetchDrivers();        // lazy-load AI driver notes in the background
    fetchSessionContext(); // lazy-load AI session context notes in the background

    const bd = document.getElementById('hm-bd');
    bd.style.display = 'flex';
    document.getElementById('hm-close').focus();
  };

  window.closeHeatmapModal = function() {
    const bd = document.getElementById('hm-bd');
    if (bd) bd.style.display = 'none';
    document.removeEventListener('keydown', _onKey);
  };

  window.hmTab = function(el, tabId) {
    document.querySelectorAll('.hm-tab').forEach(t => {
      t.classList.toggle('on', t.dataset.tab === tabId);
      t.setAttribute('aria-selected', t.dataset.tab === tabId ? 'true' : 'false');
    });
    document.querySelectorAll('.hm-panel').forEach(p => {
      p.classList.toggle('on', p.id === 'hm-p-' + tabId);
    });

    // Lazy-populate on first switch
    if (tabId === 'session' && _ccy) {
      populateSession(_ccy, _rtCache);
    } else if (tabId === 'correlations' && _ccy) {
      populateCorrelations(_ccy, _strengths, _rtCache);
    }
  };

})();
