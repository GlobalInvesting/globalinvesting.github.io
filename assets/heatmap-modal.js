// CURRENCY STRENGTH HEATMAP MODAL  v2.2 — audit fixes: tooltipEl bug, keyframes, tab a11y, labels
// CURRENCY STRENGTH HEATMAP MODAL  v1.1.0
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
  if (document.getElementById('hm-modal2-css')) return;
  const s = document.createElement('style');
  s.id = 'hm-modal2-css';
  s.textContent = `
/* ── Heatmap Modal — cohesive with Real Carry Modal ── */

#hm-bd {
  display:block!important;
}
@keyframes hm-fadein  { from{opacity:0}                              to{opacity:1} }
@keyframes hm-slidein { from{transform:translateY(-8px);opacity:0}  to{transform:none;opacity:1} }

#hm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}

#hm-modal::before {
  display:none;
}

/* ── Header ── */
#hm-hd {
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px 9px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;
  background:var(--bg2);
}
#hm-hd-left { display:flex;flex-direction:column;gap:2px; }


#hm-title-row { display:flex;align-items:center;gap:7px; }
#hm-title { font-size:14px;font-weight:600;color:var(--text);letter-spacing:-.01em;line-height:1.2;font-family:var(--font-ui,'Inter',-apple-system,sans-serif); }
#hm-title .fi { border-radius:2px;font-size:16px; }
#hm-sub { font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);letter-spacing:.02em;margin-top:1px; }
#hm-close {
  background:none;border:none;color:var(--text3,#4e5c70);font-size:16px;
  cursor:pointer;padding:3px 6px;border-radius:3px;line-height:1;
  transition:color .1s,background .1s;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);
}
#hm-close:hover { color:var(--text);background:var(--bg3); }

/* ── Metrics strip ── */
#hm-metrics {
  display:grid;grid-template-columns:repeat(6,1fr);
  border-bottom:1px solid var(--border2);
  flex-shrink:0;
  background:var(--bg);
}
.hm-mm {
  padding:9px 14px;
  border-right:1px solid var(--border2);
  display:flex;flex-direction:column;gap:1px;
}
.hm-mm:last-child { border-right:none; }
.hm-mm-lbl {
  font-size:9px;font-family:var(--font-mono,monospace);font-weight:600;
  color:var(--text2);text-transform:uppercase;letter-spacing:.09em;
}
.hm-mm-val {
  font-size:15px;font-weight:600;font-family:var(--font-mono,monospace);
  color:var(--text);line-height:1;margin-top:2px;
}
.hm-mm-val.sm   { font-size:12px; }
.hm-mm-val.up   { color:var(--up); }
.hm-mm-val.down { color:var(--down); }
.hm-mm-val.flat { color:var(--text2); }
.hm-mm-sub {
  font-size:9px;font-family:var(--font-mono,monospace);
  color:var(--text2);margin-top:1px;
}
.hm-mm-sub.up   { color:var(--up); }
.hm-mm-sub.down { color:var(--down); }

/* ── Tabs ── */
#hm-tabs {
  display:flex;padding:0 14px;
  border-bottom:1px solid var(--border,#252d3d);
  flex-shrink:0;background:var(--bg2);
  overflow-x:auto;scrollbar-width:none;
}
#hm-tabs::-webkit-scrollbar { display:none; }
.hm-tab {
  font-size:11px;font-weight:500;
  padding:9px 14px;cursor:pointer;
  color:var(--text2);
  border-bottom:2px solid transparent;
  transition:color .12s;white-space:nowrap;user-select:none;
  font-family:var(--font-ui,sans-serif);
  /* button reset — keeps visual identical to former div */
  background:none;border-top:none;border-left:none;border-right:none;outline:none;
}
.hm-tab:focus-visible { outline:2px solid var(--blue);outline-offset:-2px;border-radius:2px; }
.hm-tab:hover { color:var(--text2); }
.hm-tab.on { color:var(--text);border-bottom-color:var(--blue); }

/* ── Body ── */
#hm-body {
  flex:1;min-height:0;
  overflow-y:auto;
  padding:0;
  background:var(--bg);
  scrollbar-width:thin;
  scrollbar-color:var(--border2,#2e3a50) transparent;
}
#hm-body::-webkit-scrollbar { width:3px!important; }
#hm-body::-webkit-scrollbar-track { background:transparent; }
#hm-body::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }
#hm-body::-webkit-scrollbar-thumb:hover { background:var(--text2); }
.hm-panel { display:none;padding:0; }
.hm-panel.on { display:flex;flex:1;flex-direction:column;min-height:0; }

/* ── Card wrapper ── */
.hm-cw {
  background:var(--bg);
  border:none;
  border-radius:0;
  padding:14px;
  margin-bottom:0;
  border-bottom:1px solid var(--border,#252d3d);
  overflow-x:auto;
  scrollbar-width:thin;
  scrollbar-color:var(--border2,#2e3a50) transparent;
}
.hm-cw:last-child { border-bottom:none; }
.hm-cw::-webkit-scrollbar { height:3px; }
.hm-cw::-webkit-scrollbar-thumb { background:var(--border2,#2e3a50);border-radius:2px; }

/* Section label */
.hm-ct {
  font-size:8.5px;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text3,#4e5c70);
  letter-spacing:.07em;margin-bottom:10px;
  font-weight:600;
  text-transform:uppercase;
}

/* ── Pair breakdown table ── */
.hm-tbl {
  width:100%;border-collapse:collapse;
  font-size:11.5px;font-family:var(--font-mono,monospace);
}
.hm-tbl thead th {
  text-align:right;color:var(--text2);font-weight:500;
  font-size:9px;text-transform:uppercase;letter-spacing:.08em;
  padding:7px 10px;
  border-bottom:1px solid var(--border2);
  white-space:nowrap;
}
.hm-tbl thead th:first-child { text-align:left; }
.hm-tbl th { text-align:right;color:var(--text2);font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:7px 10px;border-bottom:1px solid var(--border2);white-space:nowrap; }
.hm-tbl th:first-child { text-align:left; }
.hm-tbl tbody tr { transition:background .08s; }
.hm-tbl tbody tr:nth-child(even) td { background:rgba(255,255,255,.015); }
.hm-tbl tbody tr:hover td { background:rgba(88,166,255,.05); }
.hm-tbl td {
  text-align:right;padding:7px 10px;
  border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text);vertical-align:middle;white-space:nowrap;
}
.hm-tbl td:first-child { text-align:left; }
.hm-tbl tr:last-child td { border-bottom:none; }
.hm-tbl td.up   { color:var(--up); }
.hm-tbl td.down { color:var(--down); }
.hm-tbl td.flat { color:var(--text2); }
.hm-tbl .sym,.hm-tbl .hm-sym { font-weight:600;color:var(--text); }
.imp-wrap { display:flex;align-items:center;gap:6px;justify-content:flex-end; }
.imp-bar-bg { width:36px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden; }
.imp-bar-fill { height:100%;border-radius:2px; }

/* ── Color classes ── */
.up   { color:var(--up); }
.down,.dn { color:var(--down); }
.flat { color:var(--text2); }

/* ── Ranking bars ── */
.hm-rank-row { display:flex;align-items:center;gap:8px;margin-bottom:5px; }
.hm-rank-ccy {
  width:34px;font-size:10px;font-weight:600;
  font-family:var(--font-mono,monospace);color:var(--text2);text-align:right;
}
.hm-rank-ccy.hl { color:var(--text); }
.hm-rank-bg { flex:1;height:14px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.hm-rank-fill { height:100%;border-radius:2px;transition:width .35s ease; }
.hm-rank-fill.no-transition { transition:none; }
.hm-rank-fill.hl   { background:var(--blue); }
.hm-rank-fill.up   { background:rgba(38,166,154,.35); }
.hm-rank-fill.down { background:rgba(239,83,80,.30); }
.hm-rank-fill.flat { background:rgba(139,148,158,.20); }
.hm-rank-val { width:56px;text-align:right;font-size:10px;font-family:var(--font-mono,monospace);color:var(--text2); }
.hm-rank-sublbl { font-size:8.5px;font-family:var(--font-mono,monospace);color:var(--text2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px; }

/* ── Session tab ── */
.sess-grid { display:grid;grid-template-columns:80px 1fr 60px;align-items:center;gap:5px 8px;font-family:var(--font-mono,monospace);font-size:10px; }
.sess-lbl { color:var(--text2);text-align:right;letter-spacing:.04em; }
.sess-lbl.hl { color:var(--blue); }
.sess-track { height:10px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden; }
.sess-fill { height:100%;border-radius:2px; }
.sess-val { text-align:right; }

/* State chips */
.state-chip {
  display:inline-flex;align-items:center;
  font-size:8px;font-family:var(--font-mono,monospace);font-weight:700;
  padding:1px 5px;border-radius:3px;letter-spacing:.06em;
  vertical-align:middle;margin-left:5px;
}
.state-live     { background:rgba(56,139,253,.15);color:var(--blue);border:1px solid rgba(56,139,253,.25); }
.state-closed   { background:transparent;color:var(--text2);border:1px solid var(--border2); }
.state-upcoming { background:rgba(210,153,34,.10);color:#d29922;border:1px solid rgba(210,153,34,.22); }

.sess-note { margin-bottom:7px; }
.sess-note-hdr { display:flex;align-items:center;gap:6px;margin-bottom:3px; }
.sess-note-name { font-size:10px;font-family:var(--font-mono,monospace);font-weight:600;letter-spacing:.04em;color:var(--text); }
.sess-note-body { font-size:10.5px;font-family:var(--font-mono,monospace);color:var(--text2);line-height:1.6;padding-left:2px; }

/* ── Rel. Strength tab (Strength Differential matrix) ── */
/* ── Strength matrix — rcm-matrix aesthetic ── */
.corr-wrap { overflow:auto;flex:1;min-height:0;scrollbar-width:thin;scrollbar-color:#444c56 transparent; }
.corr-wrap::-webkit-scrollbar { width:4px;height:4px; }
.corr-wrap::-webkit-scrollbar-track { background:transparent; }
.corr-wrap::-webkit-scrollbar-thumb { background:#444c56;border-radius:2px; }
.corr-wrap::-webkit-scrollbar-thumb:hover { background:var(--text2); }
.corr-matrix { border-collapse:collapse;font-size:10.5px;font-family:var(--font-mono,monospace);width:100%;table-layout:fixed; }
.corr-matrix th { font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:6px 4px;color:var(--text2);text-align:center;white-space:nowrap;background:var(--bg2);position:sticky;top:0;z-index:2;border-bottom:1px solid var(--border2); }
.corr-matrix th.row-head { text-align:left;width:72px;padding:6px 8px; }
.corr-matrix th.focal { color:var(--blue); }
.corr-matrix td { padding:6px 4px;text-align:center;border:1px solid rgba(255,255,255,.04);font-size:10.5px;transition:filter .1s; }
.corr-matrix td:hover { filter:brightness(1.4); }
.corr-matrix td.diag { background:#2d333b;color:var(--text2);font-size:10px;font-weight:600; }
.corr-matrix td.row-head { text-align:left;color:var(--text2);font-weight:700;font-size:10.5px;background:var(--bg2);border:none;position:sticky;left:0;z-index:1; }
.corr-matrix td.row-head.focal { color:var(--blue); }
.corr-matrix td.empty { background:transparent;border:none; }
.corr-matrix td.comp-col { border-left:2px solid rgba(255,255,255,.10); }
.corr-matrix tr.comp-row td { border-top:2px solid rgba(255,255,255,.10); }
/* cell shading — terminal palette */
.corr-cell-pos-hi { background:rgba(38,166,154,.25);color:var(--up);font-weight:700; }
.corr-cell-pos    { background:rgba(38,166,154,.10);color:var(--up); }
.corr-cell-neg-hi { background:rgba(239,83,80,.25);color:var(--down);font-weight:700; }
.corr-cell-neg    { background:rgba(239,83,80,.10);color:var(--down); }
.corr-cell-flat   { color:var(--text2); }
.corr-cell-focal  { outline:1px solid rgba(56,139,253,.35); }
/* legend — mirrors rcm-matrix-legend */
.corr-legend { display:flex;gap:16px;flex-wrap:wrap;font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);margin-top:10px;padding-top:10px;border-top:1px solid var(--border2);align-items:center;flex-shrink:0; }

/* ── Top-3 drivers ── */
.driver-row { display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;font-family:var(--font-mono,monospace); }
.driver-pair { font-size:11px;font-weight:600;color:var(--text);width:72px;padding-top:1px;flex-shrink:0; }
.driver-body { flex:1; }
.driver-top  { display:flex;align-items:center;gap:8px; }
.driver-pct  { font-size:11px;font-weight:600; }
.driver-vs   { font-size:11px;color:var(--text2); }
.driver-note { font-size:10px;color:var(--text2);margin-top:3px;line-height:1.5; }

/* ── CSI chart ── */
#hm-csi-period,.csi-controls { display:flex;gap:3px;margin-bottom:10px; }
#hm-csi-wrap,.csi-wrap {
  position:relative;height:280px;
  background:var(--bg);border-radius:4px;overflow:hidden;
  margin-bottom:10px;
}
#hm-csi-chart,.csi-canvas-placeholder { width:100%;height:100%; }
.hm-csi-pbtn,.csi-pbtn {
  font-size:10px;padding:3px 9px;border-radius:3px;
  border:1px solid var(--border2);
  background:none;color:var(--text2);cursor:pointer;
  font-family:var(--font-mono,monospace);
  transition:background .1s,color .1s,border-color .1s;
}
.hm-csi-pbtn:hover,.csi-pbtn:hover { background:rgba(255,255,255,.05);color:var(--text); }
.hm-csi-pbtn.on,.csi-pbtn.on {
  background:rgba(56,139,253,.15);
  border-color:rgba(56,139,253,.35);
  color:var(--blue);
}
#hm-csi-legend,.csi-legend {
  display:flex;flex-wrap:wrap;gap:4px 10px;margin-top:8px;
  font-size:9px;font-family:var(--font-mono,monospace);
}
.hm-csi-leg,.csi-leg {
  display:flex;align-items:center;gap:4px;cursor:pointer;
  padding:2px 4px;border-radius:2px;transition:background .1s;
}
.hm-csi-leg:hover,.csi-leg:hover { background:rgba(255,255,255,.06); }
.hm-csi-leg-dot,.csi-leg-dot { width:8px;height:2px;border-radius:1px;flex-shrink:0; }
.hm-csi-leg-lbl,.csi-leg-lbl { color:var(--text2);letter-spacing:.04em; }
.hm-csi-leg-val,.csi-leg-val { color:var(--text);font-weight:600;min-width:42px;text-align:right; }
#hm-csi-tooltip {
  position:absolute;pointer-events:none;z-index:10;
  background:rgba(13,17,23,.95);border:1px solid var(--border2);
  border-radius:4px;padding:7px 10px;font-size:9px;
  font-family:var(--font-mono,monospace);
  min-width:130px;display:none;
}
.hm-csi-tt-date { color:var(--text2);margin-bottom:5px;font-size:9px;letter-spacing:.04em; }
.hm-csi-tt-row  { display:flex;justify-content:space-between;gap:12px;margin-bottom:2px; }
.hm-csi-tt-ccy  { color:var(--text2); }
.hm-csi-tt-val  { font-weight:600; }
#hm-csi-loading {
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:10px;color:var(--text2);
  font-family:var(--font-mono,monospace);
  letter-spacing:.04em;background:var(--bg);
}

/* ── Source note ── */
.hm-src-note {
  font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);
  margin-top:10px;padding-top:9px;
  border-top:1px solid var(--border2);
  line-height:1.6;
}

/* ── Footer ── */
#hm-footer {
  padding:8px 18px;
  border-top:1px solid var(--border2);
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;background:var(--bg2);
}
#hm-footer-meta { font-size:9px;font-family:var(--font-mono,monospace);color:var(--text2);letter-spacing:.03em; }

/* ── Mobile ── */
@media (max-width:640px) {
  #hm-modal {
  width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;
  border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;
  background:var(--bg)!important;position:static!important;
  font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);
  display:flex;flex-direction:column;
}
  #hm-metrics { grid-template-columns:repeat(3,1fr); }
  .hm-mm { border-bottom:1px solid var(--border2); }
  .hm-mm:nth-child(3),.hm-mm:nth-child(6) { border-right:none; }
  #hm-body { padding:10px; }
  .hm-tbl .col-rng,.hm-tbl .col-prev { display:none; }
}
@media (max-width:520px) {
  .hm-panel { padding:0; }
  .hm-cw { padding:10px; }
  .hm-tbl th,.hm-tbl td { padding:4px 5px; }
  .hm-tbl { font-size:10px; }
  .hm-tbl .col-prev-close { display:none; }
  .imp-bar-bg { width:28px; }
}
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

  // All 28 8 major currencies pair definitions (same as populateHeatmap in dashboard.js)
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
  let _sessionCtxIsWeekend = false; // true when session-context.json was generated in closed-market mode

  // CSI state
  let _csiData       = null;  // { dates: [...], series: { EUR: [...], GBP: [...], ... } }
  let _csiChart      = null;  // LWC chart instance
  let _csiPeriodDays = 63;    // default 3M
  let _csiSeriesMap  = {};    // { EUR: LineSeries, ... } — kept for highlight toggling
  let _csiHighlightCcy = null; // currently highlighted ccy (null = use modal focal ccy)
  let _csiInited     = false;

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
  // On weekends, the file contains AI-generated recap notes
  // generate_session_context_closed() — same schema, market_closed:true flag added.
  function fetchSessionContext() {
    if (_sessionCtxFetched) return;
    _sessionCtxFetched = true;
    fetch('./ai-analysis/session-context.json?_=' + Date.now())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && data.sessions && typeof data.sessions === 'object') {
          _sessionCtxCache = data;
          // market_closed flag in JSON tells us notes are weekend recaps —
          // used by populateSession() to apply correct framing regardless of
          // whether the client clock says weekend (handles edge cases at open/close).
          _sessionCtxIsWeekend = !!data.market_closed;
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

  // Returns true when the FX market is closed for the weekend.
  // FX convention (industry standard):
  //   Closes:  Friday    21:00 UTC  (New York close)
  //   Opens:   Sunday    21:00 UTC  (Sydney open)
  // UTC day: 0=Sun, 1=Mon, …, 5=Fri, 6=Sat
  function isMarketWeekend() {
    const now  = new Date();
    const day  = now.getUTCDay();
    const hour = now.getUTCHours();
    return (
      day === 6 ||                    // All of Saturday
      (day === 5 && hour >= 21) ||    // Friday from 21:00 UTC onward
      (day === 0 && hour < 21)        // Sunday before 21:00 UTC
    );
  }

  // Returns a Set of all session names that are currently active (handles overlaps).
  // Returns an empty Set during the FX weekend (Fri 21:00 – Sun 21:00 UTC).
  function getActiveSessions() {
    if (isMarketWeekend()) return new Set();
    const h = new Date().getUTCHours();
    const active = new Set();
    // Sydney: 21:00–06:00 UTC (crosses midnight)
    if (h >= 21 || h < 6)  active.add('Sydney');
    // Tokyo: 00:00–09:00 UTC
    if (h >= 0  && h < 9)  active.add('Tokyo');
    // London: 07:00–16:00 UTC
    if (h >= 7  && h < 16) active.add('London');
    // New York: 12:00–21:00 UTC
    if (h >= 12 && h < 21) active.add('New York');
    return active;
  }

  // Legacy helper — returns the single "primary" active session for fallback text.
  function currentSessionName() {
    const active = getActiveSessions();
    for (const s of ['London', 'New York', 'Tokyo', 'Sydney']) {
      if (active.has(s)) return s;
    }
    return 'London';
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
    <div id="hm-hd-left">
      <div id="hm-title-row">
        <div id="hm-title"></div>
      </div>
      <div id="hm-sub">28-pair equal-weighted model · 8 major currencies · yfinance · ~5min delay</div>
    </div>
    <button id="hm-close" aria-label="Close" title="Close">&#10005;</button>
  </div>
  <div id="hm-metrics">
    <div class="hm-mm">
      <div class="hm-mm-lbl">Composite</div>
      <div class="hm-mm-val" id="hm-m-composite">—</div>
      <div class="hm-mm-sub" id="hm-m-comp-sub">avg vs 7 pairs</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">1W Strength</div>
      <div class="hm-mm-val" id="hm-m-1w">—</div>
      <div class="hm-mm-sub" id="hm-m-1w-sub">vs prior Fri</div>
    </div>
    <div class="hm-mm">
      <div class="hm-mm-lbl">Rank</div>
      <div class="hm-mm-val flat" id="hm-m-rank">—</div>
      <div class="hm-mm-sub">of 8 major currencies</div>
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
    <button class="hm-tab on" role="tab" aria-selected="true"  data-tab="breakdown"    onclick="hmTab(this,'breakdown')">Pair Breakdown</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="session"      onclick="hmTab(this,'session')">Session</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="correlations" onclick="hmTab(this,'correlations')">Rel. Strength</button>
    <button class="hm-tab"    role="tab" aria-selected="false" data-tab="csi"          onclick="hmTab(this,'csi')">CSI</button>
  </div>
  <div id="hm-body">
    <div class="hm-panel on" id="hm-p-breakdown">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-pairs-title">DIRECT PAIRS · DAY % &amp; 1W % · vs PREV CLOSE / PREV FRIDAY</div>
        <table class="hm-tbl" aria-label="Direct pairs for selected currency">
          <thead>
            <tr>
              <th scope="col">Pair</th>
              <th scope="col">Close</th>
              <th scope="col" class="col-prev-close">Prev close</th>
              <th scope="col">Day %</th>
              <th scope="col">1W %</th>
              <th scope="col" title="Relative contribution vs peers — bar width = magnitude">Contribution</th>
              <th scope="col">Session range</th>
            </tr>
          </thead>
          <tbody id="hm-pair-tbody"></tbody>
        </table>
      </div>
      <div class="hm-cw">
        <div class="hm-ct">FULL RANKING · ALL 8 MAJOR CURRENCIES · COMPOSITE STRENGTH</div>
        <div style="display:flex;gap:16px;">
          <div style="flex:1;">
            <div class="hm-rank-sublbl">Day %</div>
            <div id="hm-ranking-rows"></div>
          </div>
          <div style="flex:1;">
            <div class="hm-rank-sublbl">1W % · vs prior Fri</div>
            <div id="hm-ranking-1w-rows"></div>
          </div>
        </div>
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
      <div class="hm-cw" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
        <div class="hm-ct">RELATIVE STRENGTH DIFFERENTIAL · ALL 8 MAJOR · % COMPOSITE vs PREV CLOSE</div>
        <div id="hm-corr-matrix" style="flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct" id="hm-drivers-title">STRENGTH DRIVERS · TOP 3 PAIRS BY CONTRIBUTION</div>
        <div id="hm-drivers"></div>
      </div>
    </div>
    <div class="hm-panel" id="hm-p-csi">
      <div class="hm-cw">
        <div class="hm-ct" id="hm-csi-title">CURRENCY STRENGTH INDEX · ACCUMULATED % RETURN · DAILY OHLC</div>
        <div id="hm-csi-period">
          <button class="hm-csi-pbtn" data-days="21"  onclick="csiPeriod(this,21)">1M</button>
          <button class="hm-csi-pbtn on" data-days="63"  onclick="csiPeriod(this,63)">3M</button>
          <button class="hm-csi-pbtn" data-days="126" onclick="csiPeriod(this,126)">6M</button>
          <button class="hm-csi-pbtn" data-days="252" onclick="csiPeriod(this,252)">1Y</button>
          <button class="hm-csi-pbtn" data-days="0"   onclick="csiPeriod(this,0)">All</button>
        </div>
        <div id="hm-csi-wrap">
          <div id="hm-csi-loading">Loading OHLC data…</div>
          <div id="hm-csi-chart"></div>
          <div id="hm-csi-tooltip"></div>
        </div>
        <div id="hm-csi-legend"></div>
      </div>
      <div class="hm-cw">
        <div class="hm-ct" id="hm-csi-stats-title">CSI SNAPSHOT · CURRENT PERIOD</div>
        <div id="hm-csi-stats"></div>
      </div>
    </div>
  </div>
  <div id="hm-footer">
    <div id="hm-footer-meta">yfinance · ~5min delay · 28-pair equal-weighted model</div>
  </div>
</div>`;
    document.body.appendChild(el);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{ el.scrollIntoView({behavior:'smooth',block:'start'}); }));
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

    const compositeEl    = document.getElementById('hm-m-composite');
    const compositeSubEl = document.getElementById('hm-m-comp-sub');
    const v = self.pct;
    compositeEl.textContent = fmt2(v);
    compositeEl.className   = 'hm-mm-val ' + pctClass(v);

    // Count how many pairs actually contributed to this currency's composite
    const compPairCnt = myPairs.filter(p => {
      const d = rtCache[p.id];
      return d && d.pct != null;
    }).length;
    if (compositeSubEl) {
      compositeSubEl.textContent = compPairCnt + ' pair' + (compPairCnt !== 1 ? 's' : '') + ' · intraday';
    }

    // 1W composite — same equal-weighted model but using pct1w per pair
    let w1sum = 0, w1n = 0;
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct1w == null) return;
      const impact1w = d.pct1w * p.sign * (p.base === ccy ? 1 : -1);
      w1sum += impact1w;
      w1n++;
    });
    const w1El    = document.getElementById('hm-m-1w');
    const w1SubEl = document.getElementById('hm-m-1w-sub');
    if (w1n > 0) {
      const w1avg = w1sum / w1n;
      w1El.textContent    = fmt2(w1avg);
      w1El.className      = 'hm-mm-val ' + pctClass(w1avg);
      w1SubEl.textContent = w1n + ' pairs · vs prior Fri';
    } else {
      w1El.textContent    = '—';
      w1El.className      = 'hm-mm-val flat';
      w1SubEl.textContent = 'no data';
    }

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

  // _skipAnim: true when called from _updateBreakdownRT on sort-order changes
  //            (modal already open — bars should appear at target width, not animate from 0)
  function populateBreakdown(ccy, strengths, rtCache, _skipAnim) {
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
      // 1W impact — same sign convention as intraday
      const raw1w  = d?.pct1w ?? null;
      const imp1w  = raw1w != null ? raw1w * p.sign * (isCcyBase ? 1 : -1) : null;
      const close  = isCcyBase ? (d?.close ?? null) : (d?.close != null ? 1/d.close : null);
      const open   = isCcyBase ? (d?.open  ?? null) : (d?.open  != null ? 1/d.open  : null);
      const hi     = isCcyBase ? (d?.high  ?? null) : (d?.high  != null ? 1/d.high  : null);
      const lo     = isCcyBase ? (d?.low   ?? null) : (d?.low   != null ? 1/d.low   : null);
      const label  = isCcyBase
        ? (p.base + '/' + p.quote)
        : (p.quote + '/' + p.base);   // show ccy first
      impacts.push({ label, opp, close, open, hi, lo, impact, rawPct, imp1w });
    });

    // Sort: biggest positive impact first
    impacts.sort((a,b) => (b.impact??-99) - (a.impact??-99));
    const maxImp = Math.max(...impacts.map(i => Math.abs(i.impact ?? 0)), 0.001);

    const tbody = document.getElementById('hm-pair-tbody');
    tbody.innerHTML = impacts.map((r, _i) => {
      const iCls  = pctClass(r.impact);
      const rng   = (r.hi != null && r.lo != null)
        ? fmtPrice(r.lo) + ' – ' + fmtPrice(r.hi)
        : '—';
      const barW  = r.impact != null ? Math.round(Math.abs(r.impact)/maxImp*100) : 0;
      const barClr = r.impact != null && r.impact >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
      return `<tr data-pair="${r.label}">
        <td><span class="sym">${r.label}</span></td>
        <td data-cell="close">${fmtPrice(r.close)}</td>
        <td class="col-prev-close" data-cell="open">${fmtPrice(r.open)}</td>
        <td class="${iCls}" data-cell="impact">${fmt2(r.impact)}</td>
        <td class="${pctClass(r.imp1w)}" data-cell="imp1w">${r.imp1w != null ? fmt2(r.imp1w) : '—'}</td>
        <td><div class="imp-wrap" title="${fmt2(r.impact)} vs peers">
          <div class="imp-bar-bg"><div class="imp-bar-fill" data-cell="bar" style="width:${barW}%;background:${barClr}"></div></div>
        </div></td>
        <td style="font-size:9px;color:var(--text3)" data-cell="rng">${rng}</td>
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
      row.dataset.rankCcy = s.ccy;
      // On RT rebuilds (_skipAnim) set final width directly — no 0→target animation that causes flash
      const initW = _skipAnim ? fillW + '%' : '0';
      row.innerHTML = `
        <div class="hm-rank-ccy${isHL?' hl':''}">${s.ccy}</div>
        <div class="hm-rank-bg">
          <div class="hm-rank-fill ${cls}" style="width:${initW}" data-w="${fillW}"></div>
        </div>
        <div class="hm-rank-val ${pctClass(s.pct)}" data-rank-val>${fmt2(s.pct)}</div>`;
      container.appendChild(row);
    });
    // Only run the entry animation on first open (not on RT sort-order rebuilds)
    if (!_skipAnim) {
      requestAnimationFrame(() => {
        container.querySelectorAll('.hm-rank-fill').forEach(el => {
          el.style.width = el.dataset.w + '%';
        });
      });
    }

    // 1W Ranking — compute 8 major currencies composite weekly strength from pct1w across all 28 pairs
    const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];
    const w1map = {};
    ccys.forEach(c => { w1map[c] = { sum: 0, n: 0 }; });
    PAIR_DEFS.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct1w == null) return;
      const v = d.pct1w * p.sign;   // positive = base strengthened vs quote
      w1map[p.base].sum += v;  w1map[p.base].n++;
      w1map[p.quote].sum -= v; w1map[p.quote].n++;
    });
    const w1strengths = ccys
      .map(c => ({ ccy: c, pct: w1map[c].n > 0 ? w1map[c].sum / w1map[c].n : null }))
      .filter(s => s.pct != null)
      .sort((a, b) => b.pct - a.pct);

    const cont1w = document.getElementById('hm-ranking-1w-rows');
    cont1w.innerHTML = '';
    if (w1strengths.length > 0) {
      const maxAbs1w = Math.max(...w1strengths.map(s => Math.abs(s.pct)), 0.001);
      w1strengths.forEach(s => {
        const isHL  = s.ccy === ccy;
        const cls   = isHL ? 'hl' : pctClass(s.pct);
        const fillW = Math.round(Math.abs(s.pct) / maxAbs1w * 100);
        const row   = document.createElement('div');
        row.className = 'hm-rank-row';
        row.dataset.rankCcy = s.ccy;
        const initW = _skipAnim ? fillW + '%' : '0';
        row.innerHTML = `
          <div class="hm-rank-ccy${isHL?' hl':''}">${s.ccy}</div>
          <div class="hm-rank-bg">
            <div class="hm-rank-fill ${cls}" style="width:${initW}" data-w="${fillW}"></div>
          </div>
          <div class="hm-rank-val ${pctClass(s.pct)}" data-rank-val>${fmt2(s.pct)}</div>`;
        cont1w.appendChild(row);
      });
      if (!_skipAnim) {
        requestAnimationFrame(() => {
          cont1w.querySelectorAll('.hm-rank-fill').forEach(el => {
            el.style.width = el.dataset.w + '%';
          });
        });
      }
    } else {
      cont1w.innerHTML = '<div style="font-size:10px;color:var(--text3);padding:6px 0">No 1W data available</div>';
    }
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

  // Replaces all "HH:MM UTC" patterns in a Groq-generated note with the user's
  // local equivalent. E.g. "PMI at 12:00 UTC" becomes "PMI at 09:00 GMT-3".
  function convertUtcTimesInNote(text) {
    if (!text) return text;
    const tzAbbr = localTzAbbr();
    return text.replace(/\b(\d{1,2}):(\d{2})\s*UTC\b/g, function(_, hh, mm) {
      const d = new Date();
      d.setUTCHours(parseInt(hh, 10), parseInt(mm, 10), 0, 0);
      const local = d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: false });
      return local + ' ' + tzAbbr;
    });
  }

  // Returns the temporal state of a session at the current UTC hour:
  //   'active'   — session is currently open
  //   'past'     — session opened and closed earlier today (result is real)
  //   'upcoming' — session has not yet opened today
  // Industry convention (Bloomberg FXGO, Refinitiv Eikon):
  //   Bars and values are shown for active and past sessions only.
  //   Upcoming sessions show a placeholder track — no fabricated data.
  function getBarSessionState(sess) {
    if (isMarketWeekend()) return 'past'; // weekend: all bars show last-close (dimmed)
    const h = new Date().getUTCHours();
    const isActive = getActiveSessions().has(sess.name);
    if (isActive) return 'active';
    // Sydney crosses midnight: closed when 06:00 <= h < 21:00
    if (sess.name === 'Sydney') return (h >= 6 && h < 21) ? 'past' : 'upcoming';
    // All other sessions: past if current hour is past their close, upcoming if before their open
    return h >= sess.utcEnd ? 'past' : 'upcoming';
  }

  function populateSession(ccy, rtCache) {
    const tzAbbr   = localTzAbbr();
    const weekend  = isMarketWeekend();
    document.getElementById('hm-sess-title').textContent =
      ccy + ' INTRADAY COMPOSITE · SESSION WINDOW STATUS · ' + tzAbbr;

    const myPairs      = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const activeSessions = getActiveSessions();   // empty Set on weekends
    const activeSess   = currentSessionName();    // legacy fallback text

    // Compute the single intraday composite once — this is the day % vs prev close
    // weighted equally across all direct pairs. Bloomberg convention: when session-specific
    // OHLC is not available, show the full-day composite alongside session window status
    // rather than fabricating per-session values from volume weights.
    let compositeSum = 0, compositeCnt = 0;
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      if (!d || d.pct == null) return;
      compositeSum += d.pct * p.sign * (p.base === ccy ? 1 : -1);
      compositeCnt++;
    });
    const dayComposite = compositeCnt > 0 ? compositeSum / compositeCnt : null;

    // Volume share labels — used as context markers, not bar values
    const volShare = { 'New York': '38%', 'London': '35%', 'Tokyo': '18%', 'Sydney': '9%' };

    // Session bar data: active and past sessions show the day composite (honest label).
    // Upcoming sessions show no bar — Bloomberg/Eikon do not fabricate forward values.
    const sessionData = SESSIONS.map(sess => {
      const barState = getBarSessionState(sess);
      const showBar  = barState === 'active' || barState === 'past';
      // All shown sessions display the same day composite — this is transparent about
      // data availability. The session window context is conveyed by the state indicator
      // and AI notes, not by fabricated per-session performance figures.
      const pct = showBar ? dayComposite : null;
      return { ...sess, pct, barState, isActive: barState === 'active' };
    });

    const grid = document.createElement('div');
    grid.className = 'sess-grid';

    // Bar color: --blue (terminal design system) — active at full opacity, past dimmed.
    // This matches Proposal A: bars represent session window status, not directional sign.
    const compositePos = dayComposite != null && dayComposite >= 0;
    const barClr = getComputedStyle(document.documentElement).getPropertyValue('--blue').trim() || '#4f7fff';

    sessionData.forEach(s => {
      const lbl = document.createElement('div');
      lbl.className = 'sess-lbl';

      let labelText = s.name.toUpperCase();
      if (s.barState === 'active') {
        labelText += ' \u25CF';        // ● active
      } else if (s.barState === 'upcoming' && !weekend) {
        labelText += ' \u25CB';        // ○ upcoming
      }
      lbl.textContent = labelText;

      const track = document.createElement('div');
      track.className = 'sess-track';

      const val = document.createElement('div');

      if (s.barState === 'upcoming' || s.pct == null) {
        // Upcoming: empty track, no value — Bloomberg/Eikon show no forward bar
        lbl.style.cssText = 'opacity:.35;color:var(--orange,#f6941c)';
        track.style.opacity = '0.08';
        val.className = 'sess-val flat';
        val.style.cssText = 'opacity:.35;font-size:9px;color:var(--text3,#6b7280)';
        val.textContent = utcHourToLocalStr(s.utcStart);  // show open time as hint
      } else {
        const fill = document.createElement('div');
        fill.className = 'sess-fill';
        // Active: full-width bar at 75% opacity (live session — result in progress)
        // Past: dimmed bar — closed session result (Bloomberg convention)
        // Weekend: all bars dimmed (last-close convention)
        const isActive = s.barState === 'active' && !weekend;
        const dimBar   = !isActive;
        fill.style.cssText = 'width:100%;background:' + barClr +
          (dimBar ? ';opacity:.30' : ';opacity:.70');
        track.appendChild(fill);
        val.className = 'sess-val ' + (compositePos ? 'up' : 'down');
        val.textContent = fmt2(s.pct);
        if (dimBar) {
          val.style.opacity = '0.55';
          lbl.style.opacity = '0.55';
        }
      }

      grid.appendChild(lbl);
      grid.appendChild(track);
      grid.appendChild(val);
    });

    // Data note: explain the bars represent full-day composite (institutional transparency)
    const dataNote = document.createElement('div');
    dataNote.style.cssText = 'font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono,\'JetBrains Mono\',\'Courier New\',monospace);letter-spacing:.02em;margin-top:6px;opacity:.7';
    dataNote.textContent = 'Day % vs prev close \xb7 session-specific OHLC not available';

    // Weekend: no banner, no status line — dimmed bars only (Bloomberg/Eikon convention)
    const content = document.getElementById('hm-sess-content');
    content.innerHTML = '';
    content.appendChild(grid);
    content.appendChild(dataNote);

    // Session context notes — suspended on weekends; otherwise show Groq or fallback
    const notes = document.getElementById('hm-sess-notes');
    const _now    = new Date();
    const localHH = String(_now.getHours()).padStart(2,'0');
    const localMM = String(_now.getMinutes()).padStart(2,'0');
    const localStr = localHH + ':' + localMM;

    if (weekend) {
      // Weekend: show recap notes from session-context.json (generated by
      // generate_session_context_closed() — Bloomberg weekend desk note style).
      // If Groq notes are available, display them with "WEEKLY RECAP" framing.
      // Each note maps to: Sydney=Friday close, Tokyo=weekly range,
      // London=main catalyst, New York=Monday outlook.
      const groqSessions = _sessionCtxCache && _sessionCtxCache.sessions
        ? _sessionCtxCache.sessions[ccy]
        : null;

      if (groqSessions && Object.keys(groqSessions).length >= 3) {
        const weekendLabels = {
          'Sydney':   'FRI CLOSE',
          'Tokyo':    'WEEKLY',
          'London':   'CATALYST',
          'New York': 'MON OPEN',
        };
        const sessOrder = ['Sydney', 'Tokyo', 'London', 'New York'];
        notes.innerHTML = sessOrder.map(sName => {
          const note   = convertUtcTimesInNote(groqSessions[sName] || '—');
          const wLabel = weekendLabels[sName] || sName.toUpperCase();
          return (
            '<div style="margin-bottom:5px">' +
            '<span style="color:var(--text3,#6b7280);min-width:72px;display:inline-block;' +
            'font-size:9px;letter-spacing:.04em;font-weight:600">' +
            wLabel + '</span> ' +
            '<span style="color:var(--text2,#787b86)">' + note + '</span>' +
            '</div>'
          );
        }).join('') +
        '<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);' +
        'font-family:var(--font-mono);letter-spacing:.03em;">' +
        'AI Analytics \xb7 Weekly recap \xb7 Resumes at Sunday 21:00 UTC</div>';
      } else {
        // Groq notes not yet available for weekend (first run after Friday close)
        notes.innerHTML =
          '<div style="font-size:10px;color:var(--text3,#6b7280);' +
          'font-family:var(--font-mono,\'JetBrains Mono\',\'Courier New\',monospace);line-height:1.6;">' +
          'Weekly recap generating\u2026 Check back shortly.' +
          '<br>Session context resumes at Sunday 21:00 UTC (Sydney open).' +
          '</div>';
      }
      return;
    }

    // Weekday: check if Groq session context is available for this currency
    const groqSessions = _sessionCtxCache && _sessionCtxCache.sessions
      ? _sessionCtxCache.sessions[ccy]
      : null;

    if (groqSessions && Object.keys(groqSessions).length >= 3) {
      // Render session notes using getBarSessionState for consistent classification
      // with the bar section above (same UTC boundary logic, single source of truth).
      // Industry convention (Bloomberg FXGO, Refinitiv Eikon):
      //   active   — blue label + ● + full-brightness AI note (live session)
      //   past     — gray label + AI note dimmed + CLOSED badge (result, historical fact)
      //   upcoming — amber label + ○ + "opens HH:MM" placeholder, no AI note
      //              AI-generated text for a future session is an outlook written at
      //              06:00 UTC; showing it at full brightness before the session opens
      //              makes a forward projection read as an accomplished result.
      // Session notes: state-first layout (Bloomberg convention)
      //   LIVE chip     — blue, session currently open
      //   CLOSED chip   — muted gray on its own line above note (result is historical fact)
      //   UPCOMING chip — amber, session not yet open; AI note suppressed to prevent
      //                   outlook text from reading as accomplished result
      const sessOrder = ['Sydney', 'Tokyo', 'London', 'New York'];
      notes.innerHTML = sessOrder.map(sName => {
        const sess  = SESSIONS.find(s => s.name === sName);
        const state = getBarSessionState(sess);
        const aiNote = convertUtcTimesInNote(groqSessions[sName] || '\u2014');

        const labelColor = state === 'active'   ? 'var(--blue,#4f7fff)'
                         : state === 'past'      ? 'var(--text3,#6b7280)'
                         :                         'var(--orange,#f6941c)';
        const textColor  = state === 'active'   ? 'var(--text,#d1d4dc)'
                         : state === 'past'      ? 'var(--text3,#6b7280)'
                         :                         'var(--text3,#6b7280)';
        const labelDot   = state === 'active'   ? ' \u25CF'    // ●
                         : state === 'upcoming' ? ' \u25CB'    // ○
                         :                        '';

        // State chip on the header line: unambiguous before reading the note text
        const stateChip  = state === 'active'
          ? '<span style="font-size:8px;background:rgba(79,127,255,.15);color:var(--blue,#4f7fff);border-radius:2px;padding:1px 4px;letter-spacing:.07em;font-weight:700;margin-left:6px;vertical-align:middle">LIVE</span>'
          : state === 'past'
          ? '<span style="font-size:8px;color:var(--text3,#6b7280);letter-spacing:.07em;opacity:.6;margin-left:6px;vertical-align:middle">CLOSED</span>'
          : '<span style="font-size:8px;background:rgba(246,148,28,.10);color:var(--orange,#f6941c);border-radius:2px;padding:1px 4px;letter-spacing:.07em;opacity:.8;margin-left:6px;vertical-align:middle">UPCOMING</span>';

        // Upcoming: replace AI outlook with open time — AI note suppressed
        // (generated at 06:00 UTC; showing it before open reads as accomplished fact)
        const displayNote = state === 'upcoming'
          ? '<span style="color:var(--text3,#6b7280);font-style:italic">Opens ' + utcHourToLocalStr(sess.utcStart) + ' \u2014 context generated daily at 06:00 UTC</span>'
          : aiNote;

        return (
          '<div style="margin-bottom:7px">' +
          '<div style="margin-bottom:2px">' +
          '<span style="color:' + labelColor + ';font-weight:' + (state === 'active' ? '700' : '500') + ';letter-spacing:.04em;font-size:10px">' +
          sName.toUpperCase() + labelDot + '</span>' + stateChip +
          '</div>' +
          '<div style="color:' + textColor + ';padding-left:2px;font-size:11px;' + (state === 'upcoming' ? 'opacity:.6' : '') + '">' +
          displayNote +
          '</div>' +
          '</div>'
        );
      }).join('') +
      '<div style="margin-top:6px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;border-top:1px solid rgba(255,255,255,.05);padding-top:6px">' +
      'AI Analytics \xb7 ' + (_sessionCtxCache && _sessionCtxCache.generated_at
        ? (() => {
            const d = new Date(_sessionCtxCache.generated_at);
            const hh = String(d.getUTCHours()).padStart(2,'0');
            const mm = String(d.getUTCMinutes()).padStart(2,'0');
            return 'Updated ' + hh + ':' + mm + ' UTC';
          })()
        : '~2h refresh') +
      ' &nbsp;|&nbsp; ' + tzAbbr + ' ' + localStr + '</div>';
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

    // ── Helper: classify diff into CSS class (mirrors rcm-matrix logic, terminal palette) ──
    function corrCellClass(diff) {
      if (diff == null) return 'corr-cell-flat';
      if (diff >=  0.40) return 'corr-cell-pos-hi';
      if (diff >=  0.06) return 'corr-cell-pos';
      if (diff <= -0.40) return 'corr-cell-neg-hi';
      if (diff <= -0.06) return 'corr-cell-neg';
      return 'corr-cell-flat';
    }
    function corrFmt(v) {
      if (v == null) return '—';
      if (Math.abs(v) < 0.005) return '0';
      return (v > 0 ? '+' : '') + v.toFixed(2);
    }

    // ── Build <table> identical in structure to rcm-matrix ───────────────
    const matrix = document.getElementById('hm-corr-matrix');
    const wrap   = document.createElement('div');
    wrap.className = 'corr-wrap';

    // Header row
    const headerCells = `<th class="row-head" scope="col" title="Row − Column = strength differential. Positive = row currency outperforms column currency today.">Δ Strength (row − col)</th>` +
      ccys.map(c => `<th scope="col"${c === ccy ? ' class="focal"' : ''}>${c}</th>`).join('') +
      `<th scope="col" class="focal" title="Equal-weighted composite — avg % vs all 7 major currency peers">Comp.</th>`;

    // Data rows
    const bodyRows = ccys.map(rowCcy => {
      const isFocalRow = rowCcy === ccy;
      const cells = ccys.map(colCcy => {
        if (rowCcy === colCcy) {
          // Diagonal: absolute composite strength
          const abs = pctMap[rowCcy] ?? 0;
          return `<td class="diag" data-diag="${rowCcy}" title="${rowCcy} composite: ${corrFmt(abs)}">${corrFmt(abs)}</td>`;
        }
        const diff = (pctMap[rowCcy] ?? 0) - (pctMap[colCcy] ?? 0);
        const cls  = corrCellClass(diff);
        const focalCls = (isFocalRow || colCcy === ccy) ? ' corr-cell-focal' : '';
        return `<td class="${cls}${focalCls}" data-r="${rowCcy}" data-c="${colCcy}" title="${rowCcy} vs ${colCcy}: ${corrFmt(diff)}">${corrFmt(diff)}</td>`;
      }).join('');

      // Comp. column (row composite)
      const rowComp = pctMap[rowCcy] ?? 0;
      const compCls = corrCellClass(rowComp);
      const compFocalCls = isFocalRow ? ' corr-cell-focal' : '';
      const compCell = `<td class="${compCls} comp-col${compFocalCls}" data-comp-row="${rowCcy}" style="font-weight:700" title="${rowCcy} composite vs major currency peers: ${corrFmt(rowComp)}">${corrFmt(rowComp)}</td>`;

      return `<tr><td class="row-head${isFocalRow ? ' focal' : ''}">${rowCcy}</td>${cells}${compCell}</tr>`;
    }).join('');

    // Footer row (column composites)
    const footCells = ccys.map(colCcy => {
      const cv  = pctMap[colCcy] ?? 0;
      const cls = corrCellClass(cv);
      const focalCls = colCcy === ccy ? ' corr-cell-focal' : '';
      return `<td class="${cls}${focalCls}" data-comp-col="${colCcy}" style="font-weight:700" title="${colCcy} composite vs major currency peers: ${corrFmt(cv)}">${corrFmt(cv)}</td>`;
    }).join('');
    const footRow = `<tr class="comp-row"><td class="row-head focal" style="font-size:9px">Comp.</td>${footCells}<td class="diag" style="font-size:9px">—</td></tr>`;

    const legend = `<div class="corr-legend">
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.25);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong outperformance (Δ ≥ +0.40%)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.10);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Mild outperformance (Δ ≥ +0.06%)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.10);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Mild underperformance</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.25);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong underperformance</span>
      <span style="color:var(--text3,#4e5c70)">Diagonal = intraday composite · Values = equal-weighted Δ%, not Pearson correlations</span>
    </div>`;

    wrap.innerHTML = `<table class="corr-matrix" aria-label="Intraday strength differential matrix 8 major currencies">
      <thead><tr>${headerCells}</tr></thead>
      <tbody>${bodyRows}${footRow}</tbody>
    </table>`;

    matrix.innerHTML = '';
    matrix.appendChild(wrap);
    matrix.insertAdjacentHTML('beforeend', legend);

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
            <span style="font-size:11px;font-weight:600" class="${cls}" data-driver-idx="${i}">${fmt2(d.impact)}</span>
            <span style="font-size:11px;color:var(--text2,#787b86)">vs ${d.opp}</span>
          </div>
          ${noteEl}
        </div>
      </div>`;
    }).join('') + (ccyNotes
      ? `<div style="margin-top:4px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">AI Analytics · ~5min delay</div>`
      : '');
  }

  // ── CSI (Currency Strength Index) ────────────────────────────────────────
  // Bloomberg WCRS convention: normalized cumulative log-return from period start.
  // All 8 series start at 0bp on day 0 — divergence represents relative performance.

  // Colour palette — 8 distinct, accessible colours matching terminal design language
  const CSI_COLORS = {
    EUR: '#4f7fff',  // --blue
    GBP: '#26a69a',  // --up (teal)
    JPY: '#ef5350',  // --down (red)
    AUD: '#f6941c',  // --orange
    CAD: '#a78bfa',  // purple
    CHF: '#34d399',  // emerald
    NZD: '#fb923c',  // amber
    USD: '#94a3b8',  // slate (USD neutral)
  };

  const CCY_ORDER = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];

  // Pairs sign convention for deriving ccy strength from OHLC:
  // +1 = pair close goes up → base strengthens; -1 = inverse
  const PAIR_SIGN = {};
  PAIR_DEFS.forEach(p => { PAIR_SIGN[p.id] = p.sign; });

  // Load all 28 OHLC files in parallel, compute per-currency daily log-returns
  // and accumulate into the CSI series.
  async function _loadCSIData() {
    const pairIds = PAIR_DEFS.map(p => p.id);
    const fetches = pairIds.map(id =>
      fetch('./ohlc-data/' + id + '.json')
        .then(r => r.ok ? r.json() : [])
        .catch(() => [])
    );
    const allOHLC = await Promise.all(fetches);

    // Build a date-keyed map of log-returns for each pair
    // pairRet[id][date] = log(close/prevClose) * sign * (base=ccy ? +1 : -1)
    const pairRet = {};
    const allDates = new Set();

    pairIds.forEach((id, i) => {
      const bars = allOHLC[i];
      const p    = PAIR_DEFS[i];
      pairRet[id] = {};
      for (let j = 1; j < bars.length; j++) {
        const date = bars[j].time;
        const ret  = Math.log(bars[j].close / bars[j - 1].close);
        pairRet[id][date] = ret * p.sign;  // positive = base ccy gained vs quote
        allDates.add(date);
      }
    });

    // Sort dates
    const dates = [...allDates].sort();

    // For each ccy and each date: average log-return across its 7 pairs
    // (sign-corrected so positive always = this ccy strengthened)
    const ccyDailyRet = {};
    CCY_ORDER.forEach(ccy => { ccyDailyRet[ccy] = {}; });

    dates.forEach(date => {
      PAIR_DEFS.forEach(p => {
        const ret = pairRet[p.id][date];
        if (ret == null || isNaN(ret)) return;
        // base ccy gets +ret, quote ccy gets -ret
        if (ccyDailyRet[p.base]) {
          ccyDailyRet[p.base][date] = (ccyDailyRet[p.base][date] || 0) + ret;
        }
        if (ccyDailyRet[p.quote]) {
          ccyDailyRet[p.quote][date] = (ccyDailyRet[p.quote][date] || 0) - ret;
        }
      });
    });

    // Normalize by number of pairs each ccy participates in (always 7 for 8 major currencies)
    // then accumulate to get the CSI series
    const series = {};
    CCY_ORDER.forEach(ccy => {
      const pairsForCcy = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy).length;
      let cum = 0;
      series[ccy] = dates.map(date => {
        const sum = ccyDailyRet[ccy][date];
        if (sum != null) cum += sum / pairsForCcy;
        // Convert to % (×100) for display
        return { time: date, value: parseFloat((cum * 100).toFixed(4)) };
      });
    });

    return { dates, series };
  }

  // Render or update the LWC chart with the current period
  function _renderCSIChart(ccy) {
    const LWC = window.LightweightCharts;
    if (!LWC || !_csiData) return;

    const wrap      = document.getElementById('hm-csi-wrap');
    const chartEl   = document.getElementById('hm-csi-chart');
    const tooltipEl = document.getElementById('hm-csi-tooltip');
    if (!wrap || !chartEl) return;

    // Determine date slice
    const allDates  = _csiData.dates;
    let startIdx    = 0;
    if (_csiPeriodDays > 0) {
      startIdx = Math.max(0, allDates.length - _csiPeriodDays);
    }
    const cutoffDate = allDates[startIdx];

    // Destroy old chart if it exists
    if (_csiChart) {
      try { _csiChart.remove(); } catch(e) {}
      _csiChart = null;
      chartEl.innerHTML = '';
    }
    _csiSeriesMap = {};
    _csiHighlightCcy = null;

    const _csiBg    = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()    || '#131722';
    const _csiText2 = getComputedStyle(document.documentElement).getPropertyValue('--text2').trim() || '#9096a0';
    const _csiBlue  = getComputedStyle(document.documentElement).getPropertyValue('--blue').trim()  || '#4f7fff';

    _csiChart = LWC.createChart(chartEl, {
      layout: {
        background: { color: _csiBg },
        textColor: _csiText2,
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,.04)' },
        horzLines: { color: 'rgba(255,255,255,.04)' },
      },
      crosshair: {
        mode: LWC.CrosshairMode?.Normal ?? 1,
        vertLine: { color: 'rgba(255,255,255,.25)', style: 2, labelVisible: true },
        horzLine: { color: 'rgba(255,255,255,.15)', style: 2, labelVisible: true },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,.08)',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,.08)',
        timeVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      width: wrap.offsetWidth,
      height: 280,
    });

    CCY_ORDER.forEach(c => {
      const isFocus = c === ccy;
      const allPts = _csiData.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      const baseVal  = sliceIdx >= 0 ? allPts[sliceIdx].value : 0;
      const raw = (sliceIdx >= 0 ? allPts.slice(sliceIdx) : allPts)
        .map(pt => ({ time: pt.time, value: parseFloat((pt.value - baseVal).toFixed(4)) }));
      const ls = _csiChart.addSeries(LWC.LineSeries, {
        color: CSI_COLORS[c],
        lineWidth: isFocus ? 2.5 : 1,
        lineStyle: 0,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: isFocus,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: _csiBg,
        crosshairMarkerBackgroundColor: CSI_COLORS[c],
      });
      ls.setData(raw);
      if (!isFocus) ls.applyOptions({ lineWidth: 1, color: CSI_COLORS[c] + 'aa' });
      _csiSeriesMap[c] = ls;
    });

    // Zero baseline
    const firstSeries = _csiSeriesMap[CCY_ORDER[0]];
    if (firstSeries) {
      firstSeries.createPriceLine({
        price: 0,
        color: 'rgba(255,255,255,.20)',
        lineWidth: 1,
        lineStyle: 1,
        axisLabelVisible: false,
        title: '',
      });
    }

    // Bloomberg-style multi-series crosshair tooltip
    _csiChart.subscribeCrosshairMove(param => {
      if (!param || !param.time || !tooltipEl) {
        if (tooltipEl) tooltipEl.style.display = 'none';
        return;
      }
      const rows = CCY_ORDER.map(c => {
        const v = param.seriesData.get(_csiSeriesMap[c]);
        return { ccy: c, val: v ? v.value : null };
      }).filter(r => r.val != null).sort((a, b) => b.val - a.val);

      if (!rows.length) { tooltipEl.style.display = 'none'; return; }

      tooltipEl.innerHTML =
        '<div class="hm-csi-tt-date">' + param.time + '</div>' +
        rows.map(r => {
          const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
          const dot = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:' + CSI_COLORS[r.ccy] + ';margin-right:5px;"></span>';
          return '<div class="hm-csi-tt-row">' +
            '<span class="hm-csi-tt-ccy">' + dot + r.ccy + '</span>' +
            '<span class="hm-csi-tt-val ' + cls + '">' +
              (r.val >= 0 ? '+' : '') + r.val.toFixed(2) + '%' +
            '</span></div>';
        }).join('');
      tooltipEl.style.display = 'block';

      // Position: keep within wrap bounds
      const wrapRect = wrap.getBoundingClientRect();
      const x = param.point ? param.point.x : 0;
      const y = param.point ? param.point.y : 0;
      const ttW = 140, ttH = 20 + rows.length * 18;
      const left = (x + ttW + 20 > wrap.offsetWidth) ? (x - ttW - 10) : (x + 16);
      const top  = Math.max(0, Math.min(y - ttH / 2, wrap.offsetHeight - ttH));
      tooltipEl.style.left = left + 'px';
      tooltipEl.style.top  = top  + 'px';
    });

    // Update legend with final values
    _updateCSILegend(ccy, cutoffDate);
  }

  function _updateCSILegend(ccy, cutoffDate) {
    const legendEl = document.getElementById('hm-csi-legend');
    if (!legendEl || !_csiData) return;

    // Get final value for each ccy in the current period — rebased to 0 at period start
    const vals = CCY_ORDER.map(c => {
      const allPts   = _csiData.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      if (sliceIdx < 0) return { ccy: c, val: null, change: null };
      const baseVal  = allPts[sliceIdx].value;
      const filtered = allPts.slice(sliceIdx).map(pt => pt.value - baseVal);
      const last  = filtered.length ? filtered[filtered.length - 1] : null;
      const first = 0; // always 0 after rebase
      return { ccy: c, val: last != null ? parseFloat(last.toFixed(4)) : null, change: last };
    }).sort((a, b) => (b.val ?? -99) - (a.val ?? -99));

    legendEl.innerHTML = vals.map(r => {
      const isFocus = r.ccy === ccy;
      const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
      const valStr = r.val != null ? (r.val >= 0 ? '+' : '') + r.val.toFixed(2) + '%' : '—';
      return '<div class="hm-csi-leg" onclick="csiHighlight(\'' + r.ccy + '\')" style="cursor:pointer" title="Click to highlight ' + r.ccy + '">' +
        '<div class="hm-csi-leg-dot" style="background:' + CSI_COLORS[r.ccy] + ';' +
          (isFocus ? 'height:3px;' : '') + '"></div>' +
        '<span class="hm-csi-leg-lbl" style="' + (isFocus ? 'color:var(--text,#d1d4dc);font-weight:600;' : '') + '">' +
          r.ccy + '</span>' +
        '<span class="hm-csi-leg-val ' + cls + '">' + valStr + '</span>' +
      '</div>';
    }).join('');
  }

  function _renderCSIStats(ccy) {
    const statsEl = document.getElementById('hm-csi-stats');
    const titleEl = document.getElementById('hm-csi-stats-title');
    if (!statsEl || !_csiData) return;

    const allDates = _csiData.dates;
    let startIdx   = 0;
    if (_csiPeriodDays > 0) startIdx = Math.max(0, allDates.length - _csiPeriodDays);
    const cutoffDate = allDates[startIdx];

    const rows = CCY_ORDER.map(c => {
      const allPts   = _csiData.series[c];
      const sliceIdx = allPts.findIndex(pt => pt.time >= cutoffDate);
      if (sliceIdx < 0) return { ccy: c, val: null, min: null, max: null, range: null };
      const baseVal = allPts[sliceIdx].value;
      const vals = allPts.slice(sliceIdx).map(pt => parseFloat((pt.value - baseVal).toFixed(4)));
      return {
        ccy: c,
        val: vals[vals.length - 1],
        min: Math.min(...vals),
        max: Math.max(...vals),
        range: Math.max(...vals) - Math.min(...vals),
      };
    }).sort((a, b) => (b.val ?? -99) - (a.val ?? -99));

    const periodLabel = _csiPeriodDays === 21 ? '1M' : _csiPeriodDays === 63 ? '3M' :
                        _csiPeriodDays === 126 ? '6M' : _csiPeriodDays === 252 ? '1Y' : 'All';
    if (titleEl) titleEl.textContent = 'CSI SNAPSHOT · ' + periodLabel + ' · ACCUMULATED RETURN';

    statsEl.innerHTML = '<table class="hm-tbl" aria-label="CSI period statistics">' +
      '<thead><tr>' +
      '<th scope="col">Currency</th>' +
      '<th scope="col">Accum. Return</th>' +
      '<th scope="col">Drawdown (low)</th>' +
      '<th scope="col">Peak (high)</th>' +
      '<th scope="col">Peak-to-Trough</th>' +
      '</tr></thead><tbody>' +
      rows.map(r => {
        const isFocus = r.ccy === ccy;
        const cls = r.val > 0 ? 'up' : r.val < 0 ? 'down' : 'flat';
        const fmt = v => v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '—';
        return '<tr style="' + (isFocus ? 'background:rgba(79,127,255,.07);' : '') + '">' +
          '<td><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:' +
            CSI_COLORS[r.ccy] + ';margin-right:6px;vertical-align:middle;"></span>' +
            '<span class="sym" style="' + (isFocus ? 'color:var(--blue,#4f7fff);' : '') + '">' + r.ccy + '</span></td>' +
          '<td class="' + cls + '">' + fmt(r.val) + '</td>' +
          '<td class="' + (r.min != null && r.min < 0 ? 'down' : r.min != null && r.min > 0 ? 'up' : 'flat') + '">' + fmt(r.min) + '</td>' +
          '<td class="' + (r.max != null && r.max > 0 ? 'up' : r.max != null && r.max < 0 ? 'down' : 'flat') + '">' + fmt(r.max) + '</td>' +
          '<td style="color:var(--text2,#787b86)">' + (r.range != null ? r.range.toFixed(2) + '%' : '—') + '</td>' +
        '</tr>';
      }).join('') +
      '</tbody></table>' +
      '<div style="margin-top:8px;font-size:9px;color:var(--text3,#6b7280);font-family:var(--font-mono);letter-spacing:.03em;">' +
      'ohlc-data · yfinance · 28-pair equal-weighted CSI · Accum. Return = total from period start · Drawdown/Peak = lowest/highest CSI value within period</div>';
  }

  async function populateCSI(ccy) {
    const loadingEl = document.getElementById('hm-csi-loading');

    // Only fetch OHLC data once per page session
    if (!_csiData) {
      if (loadingEl) loadingEl.style.display = 'flex';

      // Ensure LWC is loaded (reuse dashboard.js loadLWC pattern)
      if (!window.LightweightCharts) {
        await new Promise((res, rej) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
      }

      try {
        _csiData = await _loadCSIData();
      } catch(e) {
        if (loadingEl) loadingEl.textContent = 'Failed to load OHLC data';
        return;
      }
    }

    if (loadingEl) loadingEl.style.display = 'none';

    const allDates  = _csiData.dates;
    let startIdx    = 0;
    if (_csiPeriodDays > 0) startIdx = Math.max(0, allDates.length - _csiPeriodDays);
    const cutoffDate = allDates[startIdx];

    _renderCSIChart(ccy);
    _renderCSIStats(ccy);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    _ccy       = ccy;
    _strengths = strengths;
    _rtCache   = rtCache;

    buildModal();

    // Set title with flag — Proposal A: flag in hm-title-row, text in hm-title
    const meta  = CCY_META[ccy] || { flag: 'un', full: ccy };
    const titleEl = document.getElementById('hm-title');
    const titleRow = document.getElementById('hm-title-row');
    titleEl.textContent = `${ccy} — ${meta.full} Strength`;
    // Insert flag before title text if not already present
    if (!titleRow.querySelector('.fi')) {
      const flagSpan = document.createElement('span');
      flagSpan.className = `fi fi-${meta.flag}`;
      flagSpan.style.cssText = 'border-radius:2px;font-size:15px;vertical-align:middle;flex-shrink:0;';
      titleRow.insertBefore(flagSpan, titleEl);
    } else {
      const flagSpan = titleRow.querySelector('.fi');
      flagSpan.className = `fi fi-${meta.flag}`;
    }

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

    // Update source labels to reflect active data source (Finnhub live vs yfinance)
    _updateModalSourceLabels();

    const bd = document.getElementById('hm-bd');
    bd.style.display = 'flex';
    document.getElementById('hm-close').focus();
  };

  window.csiPeriod = function(btn, days) {
    _csiPeriodDays = days;
    document.querySelectorAll('.hm-csi-pbtn').forEach(b => b.classList.toggle('on', b === btn));
    if (_csiData && _ccy) {
      _renderCSIChart(_ccy);
      _renderCSIStats(_ccy);
    }
  };

  // Toggle highlight on a CSI series — clicking active focal ccy resets to modal ccy
  window.csiHighlight = function(clickedCcy) {
    if (!_csiChart || !_csiSeriesMap[clickedCcy]) return;
    const newFocus = (_csiHighlightCcy === clickedCcy) ? _ccy : clickedCcy;
    _csiHighlightCcy = (newFocus === _ccy) ? null : newFocus;
    CCY_ORDER.forEach(c => {
      const isFocus = c === newFocus;
      _csiSeriesMap[c].applyOptions({
        lineWidth: isFocus ? 2.5 : 1,
        color: isFocus ? CSI_COLORS[c] : CSI_COLORS[c] + 'aa',
        crosshairMarkerVisible: isFocus,
      });
    });
    // Re-render legend to update bold/active state
    const allDates = _csiData.dates;
    let startIdx = 0;
    if (_csiPeriodDays > 0) startIdx = Math.max(0, allDates.length - _csiPeriodDays);
    _updateCSILegend(newFocus, allDates[startIdx]);
  };

  window.closeHeatmapModal = function() {
    const bd = document.getElementById('hm-bd');
    if (bd) bd.style.display = 'none';
    document.removeEventListener('keydown', _onKey);
    // Destroy chart so it re-renders at correct size on next open
    if (_csiChart) { try { _csiChart.remove(); } catch(e) {} _csiChart = null; }
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
    } else if (tabId === 'csi' && _ccy) {
      populateCSI(_ccy);
    }
  };

  // ── Live source label — updates hm-sub and hm-footer-meta to reflect active source ──
  function _updateModalSourceLabels() {
    const hasFh = window.STOOQ_RT_CACHE
      ? Object.values(window.STOOQ_RT_CACHE).some(e => e?.fromFinnhub)
      : false;
    const srcLabel = hasFh
      ? 'Finnhub \u00b7 live \u00b7 28-pair equal-weighted \u00b7 8 major currencies'
      : '28-pair equal-weighted model \u00b7 8 major currencies \u00b7 yfinance \u00b7 ~5min delay';
    const footerLabel = hasFh
      ? 'Finnhub \u00b7 live \u00b7 28-pair equal-weighted model'
      : 'yfinance \u00b7 ~5min delay \u00b7 28-pair equal-weighted model';
    const subEl    = document.getElementById('hm-sub');
    const footerEl = document.getElementById('hm-footer-meta');
    if (subEl)    subEl.textContent    = srcLabel;
    if (footerEl) footerEl.textContent = footerLabel;
  }

  // ── _updateBreakdownRT — flash-free in-place update for the Breakdown tab ──────────
  // Called by _hmRefreshIfOpen on every Finnhub tick instead of full populateBreakdown().
  // Updates only textContent/className/style on already-rendered DOM nodes.
  // Falls back to full populateBreakdown() if the DOM structure is stale (e.g. ccy changed).
  function _updateBreakdownRT(ccy, strengths, rtCache) {
    const tbody = document.getElementById('hm-pair-tbody');
    if (!tbody || tbody.children.length === 0) {
      populateMetrics(ccy, strengths, rtCache);
      populateBreakdown(ccy, strengths, rtCache, true); // _skipAnim: modal already open
      return;
    }

    // Re-compute impacts (same logic as populateBreakdown but no DOM rebuild)
    const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
    const impacts = [];
    myPairs.forEach(p => {
      const d = rtCache[p.id];
      const isCcyBase = p.base === ccy;
      const opp = isCcyBase ? p.quote : p.base;
      const rawPct = d?.pct ?? null;
      const impact = rawPct != null ? rawPct * p.sign * (isCcyBase ? 1 : -1) : null;
      const raw1w  = d?.pct1w ?? null;
      const imp1w  = raw1w != null ? raw1w * p.sign * (isCcyBase ? 1 : -1) : null;
      const close  = isCcyBase ? (d?.close ?? null) : (d?.close != null ? 1/d.close : null);
      const open   = isCcyBase ? (d?.open  ?? null) : (d?.open  != null ? 1/d.open  : null);
      const hi     = isCcyBase ? (d?.high  ?? null) : (d?.high  != null ? 1/d.high  : null);
      const lo     = isCcyBase ? (d?.low   ?? null) : (d?.low   != null ? 1/d.low   : null);
      const label  = isCcyBase ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
      impacts.push({ label, opp, close, open, hi, lo, impact, rawPct, imp1w });
    });
    impacts.sort((a,b) => (b.impact??-99) - (a.impact??-99));
    const maxImp = Math.max(...impacts.map(i => Math.abs(i.impact ?? 0)), 0.001);

    // Check if sort order changed — if so, fall back to full render (with _skipAnim to avoid flash)
    const rows = Array.from(tbody.querySelectorAll('tr[data-pair]'));
    const currentOrder = rows.map(r => r.dataset.pair);
    const newOrder = impacts.map(r => r.label);
    if (currentOrder.join(',') !== newOrder.join(',')) {
      populateMetrics(ccy, strengths, rtCache);
      populateBreakdown(ccy, strengths, rtCache, true); // _skipAnim: modal already open
      return;
    }

    // In-place update only — no innerHTML touches
    impacts.forEach(r => {
      const row = tbody.querySelector(`tr[data-pair="${r.label}"]`);
      if (!row) return;
      const iCls   = pctClass(r.impact);
      const barW   = r.impact != null ? Math.round(Math.abs(r.impact)/maxImp*100) : 0;
      const barClr = r.impact != null && r.impact >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
      const rng    = (r.hi != null && r.lo != null) ? fmtPrice(r.lo) + ' – ' + fmtPrice(r.hi) : '—';

      const closeCell  = row.querySelector('[data-cell="close"]');
      const openCell   = row.querySelector('[data-cell="open"]');
      const impactCell = row.querySelector('[data-cell="impact"]');
      const imp1wCell  = row.querySelector('[data-cell="imp1w"]');
      const barFill    = row.querySelector('[data-cell="bar"]');
      const rngCell    = row.querySelector('[data-cell="rng"]');

      if (closeCell)  closeCell.textContent  = fmtPrice(r.close);
      if (openCell)   openCell.textContent   = fmtPrice(r.open);
      if (impactCell) { impactCell.textContent = fmt2(r.impact); impactCell.className = iCls; }
      if (imp1wCell)  { imp1wCell.textContent  = r.imp1w != null ? fmt2(r.imp1w) : '—'; imp1wCell.className = pctClass(r.imp1w); }
      if (barFill)    { barFill.style.width = barW + '%'; barFill.style.background = barClr; }
      if (rngCell)    rngCell.textContent   = rng;
    });

    // Update metrics header (already in-place — populateMetrics uses textContent throughout)
    populateMetrics(ccy, strengths, rtCache);

    // Update day% ranking in-place
    const container = document.getElementById('hm-ranking-rows');
    if (container) {
      const sorted    = [...strengths].sort((a,b) => b.pct - a.pct);
      const maxAbsPct = Math.max(...sorted.map(s => Math.abs(s.pct)), 0.001);
      sorted.forEach(s => {
        const rankRow = container.querySelector(`[data-rank-ccy="${s.ccy}"]`);
        if (!rankRow) return;
        const fillEl = rankRow.querySelector('.hm-rank-fill');
        const valEl  = rankRow.querySelector('[data-rank-val]');
        const fillW  = Math.round(Math.abs(s.pct) / maxAbsPct * 100);
        const cls    = 'hm-rank-fill ' + ((s.ccy === ccy) ? 'hl' : pctClass(s.pct));
        const newW   = fillW + '%';
        if (fillEl) {
          if (fillEl.style.width !== newW)    fillEl.style.width = newW;
          if (fillEl.className  !== cls)      fillEl.className   = cls;
        }
        if (valEl) {
          const newTxt = fmt2(s.pct);
          const newCls = 'hm-rank-val ' + pctClass(s.pct);
          if (valEl.textContent !== newTxt) valEl.textContent = newTxt;
          if (valEl.className   !== newCls) valEl.className   = newCls;
        }
      });
    }

    // Update 1W ranking in-place
    const cont1w = document.getElementById('hm-ranking-1w-rows');
    if (cont1w && cont1w.querySelector('[data-rank-ccy]')) {
      const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];
      const w1map = {};
      ccys.forEach(c => { w1map[c] = { sum: 0, n: 0 }; });
      PAIR_DEFS.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct1w == null) return;
        const v = d.pct1w * p.sign;
        w1map[p.base].sum += v; w1map[p.base].n++;
        w1map[p.quote].sum -= v; w1map[p.quote].n++;
      });
      const w1strengths = ccys.map(c => ({ ccy: c, pct: w1map[c].n > 0 ? w1map[c].sum / w1map[c].n : null })).filter(s => s.pct != null);
      const maxAbs1w = Math.max(...w1strengths.map(s => Math.abs(s.pct)), 0.001);
      w1strengths.forEach(s => {
        const rankRow = cont1w.querySelector(`[data-rank-ccy="${s.ccy}"]`);
        if (!rankRow) return;
        const fillEl = rankRow.querySelector('.hm-rank-fill');
        const valEl  = rankRow.querySelector('[data-rank-val]');
        const fillW  = Math.round(Math.abs(s.pct) / maxAbs1w * 100);
        const cls    = 'hm-rank-fill ' + ((s.ccy === ccy) ? 'hl' : pctClass(s.pct));
        const newW   = fillW + '%';
        if (fillEl) {
          if (fillEl.style.width !== newW)    fillEl.style.width = newW;
          if (fillEl.className  !== cls)      fillEl.className   = cls;
        }
        if (valEl) {
          const newTxt = fmt2(s.pct);
          const newCls = 'hm-rank-val ' + pctClass(s.pct);
          if (valEl.textContent !== newTxt) valEl.textContent = newTxt;
          if (valEl.className   !== newCls) valEl.className   = newCls;
        }
      });
    }
  }

  // ── _updateCorrelationsRT — flash-free in-place update for the Correlations tab ──
  // Updates only cell values/classes in the already-rendered corr matrix.
  // Falls back to full populateCorrelations() if the table is missing (shouldn't happen).
  function _updateCorrelationsRT(ccy, strengths, rtCache) {
    const matrix = document.getElementById('hm-corr-matrix');
    if (!matrix || !matrix.querySelector('[data-r]')) {
      populateCorrelations(ccy, strengths, rtCache);
      return;
    }

    // Re-compute pctMap
    const ccys = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD','USD'];
    const pctMap = {};
    ccys.forEach(c => { pctMap[c] = null; });
    strengths.forEach(s => { pctMap[s.ccy] = s.pct; });

    function corrFmt(v) {
      if (v == null) return '—';
      if (Math.abs(v) < 0.005) return '0';
      return (v > 0 ? '+' : '') + v.toFixed(2);
    }
    function corrCellClass(diff) {
      if (diff == null) return 'corr-cell-flat';
      if (diff >=  0.40) return 'corr-cell-pos-hi';
      if (diff >=  0.06) return 'corr-cell-pos';
      if (diff <= -0.40) return 'corr-cell-neg-hi';
      if (diff <= -0.06) return 'corr-cell-neg';
      return 'corr-cell-flat';
    }

    // Update body cells
    matrix.querySelectorAll('td[data-r][data-c]').forEach(td => {
      const r = td.dataset.r, c = td.dataset.c;
      const diff = (pctMap[r] ?? 0) - (pctMap[c] ?? 0);
      const focalCls = (r === ccy || c === ccy) ? ' corr-cell-focal' : '';
      td.className = corrCellClass(diff) + focalCls;
      td.textContent = corrFmt(diff);
      td.title = `${r} vs ${c}: ${corrFmt(diff)}`;
    });

    // Update diagonal cells
    matrix.querySelectorAll('td[data-diag]').forEach(td => {
      const r = td.dataset.diag;
      const abs = pctMap[r] ?? 0;
      td.textContent = corrFmt(abs);
      td.title = `${r} composite: ${corrFmt(abs)}`;
    });

    // Update Comp. column cells (row composites)
    matrix.querySelectorAll('td[data-comp-row]').forEach(td => {
      const r = td.dataset.compRow;
      const v = pctMap[r] ?? 0;
      const focalCls = r === ccy ? ' corr-cell-focal' : '';
      td.className = corrCellClass(v) + ' comp-col' + focalCls;
      td.textContent = corrFmt(v);
      td.title = `${r} composite vs major currency peers: ${corrFmt(v)}`;
    });

    // Update footer cells (column composites)
    matrix.querySelectorAll('td[data-comp-col]').forEach(td => {
      const c = td.dataset.compCol;
      const v = pctMap[c] ?? 0;
      const focalCls = c === ccy ? ' corr-cell-focal' : '';
      td.className = corrCellClass(v) + focalCls;
      td.textContent = corrFmt(v);
      td.title = `${c} composite vs major currency peers: ${corrFmt(v)}`;
    });

    // Update top-3 drivers in-place (just the pct values, no layout change)
    const driversEl = document.getElementById('hm-drivers');
    if (driversEl) {
      const myPairs = PAIR_DEFS.filter(p => p.base === ccy || p.quote === ccy);
      const driven  = [];
      myPairs.forEach(p => {
        const d = rtCache[p.id];
        if (!d || d.pct == null) return;
        const impact = d.pct * p.sign * (p.base === ccy ? 1 : -1);
        const opp    = p.base === ccy ? p.quote : p.base;
        const label  = p.base === ccy ? (p.base+'/'+p.quote) : (p.quote+'/'+p.base);
        driven.push({ label, opp, impact });
      });
      driven.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));
      driven.slice(0,3).forEach((d,i) => {
        const pctEl = driversEl.querySelector(`[data-driver-idx="${i}"]`);
        if (pctEl) {
          pctEl.textContent = fmt2(d.impact);
          pctEl.className   = pctClass(d.impact);
        }
      });
    }
  }

    // ── _hmRefreshIfOpen — called by dashboard.js populateHeatmap() on every RT update ──
  // Refreshes whichever tab is currently active without closing/reopening the modal.
  // Only runs when the modal is actually visible — no-op otherwise.
  // This is the mechanism that makes the modal update in real time from Finnhub ticks.
  window._hmRefreshIfOpen = function(newStrengths, newRtCache) {
    const bd = document.getElementById('hm-bd');
    if (!bd || bd.style.display === 'none' || !_ccy) return;

    // Update stored references so tab switches also get fresh data
    _strengths = newStrengths;
    _rtCache   = newRtCache;

    // Update source labels in header and footer
    _updateModalSourceLabels();

    // Refresh the active tab with flash-free in-place updates
    const activeTab = document.querySelector('.hm-tab.on');
    if (!activeTab) return;
    const tabId = activeTab.dataset.tab;

    if (tabId === 'breakdown') {
      // In-place update — no innerHTML rebuild, no flash
      _updateBreakdownRT(_ccy, _strengths, _rtCache);
    } else if (tabId === 'session') {
      // Session data is from session_high/low (changes slowly) — full render acceptable
      populateSession(_ccy, _rtCache);
    } else if (tabId === 'correlations') {
      // In-place update — only cell values/classes, no table rebuild
      _updateCorrelationsRT(_ccy, _strengths, _rtCache);
    }
    // CSI tab uses historical OHLC data only — no RT refresh needed
  };

})();
