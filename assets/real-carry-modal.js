// ═══════════════════════════════════════════════════════════════════════════
// REAL RATE CARRY MODAL  v1.4
// File: assets/real-carry-modal.js
//
// Architecture:
//   Tab 1 — Rates Breakdown   : 8-row table — Nominal | Infl.Exp | Real Rate | OIS Bias
//   Tab 2 — Real Rate Matrix  : 8×8 color-coded differential matrix (lower triangle)
//   Tab 3 — Pair Detail       : per-pair deep dive opened from carry ranking click
//
// Data sources:
//   Nominal rates  : ./rates/{CCY}.json  (daily batch)
//   Infl.Exp.      : FRED CSV (T5YIE USD, T5YIFR EUR) → ./extended-data/{CCY}.json fallback
//   OIS Bias       : ./meetings-data/meetings.json
//
// Real rate = Nominal CB rate − Inflation Expectation (breakeven / CPI proxy)
// This is the standard used by Bloomberg FXFR, Refinitiv FX carry screens,
// and institutional macro PM morning packets.
//
// Inflation expectation sources (in priority order):
//   USD: FRED T5YIE   — 5Y breakeven inflation, market-derived, daily
//   EUR: FRED T5YIFR  — EUR 5Y5Y inflation swap rate, daily
//   GBP: extended-data/GBP.json — CPI YoY (IMF SDMX 3.0 api.imf.org → index-to-YoY, weekly batch)
//   JPY: extended-data/JPY.json — CPI YoY (IMF SDMX 3.0 api.imf.org → index-to-YoY, weekly batch)
//   AUD: extended-data/AUD.json — CPI YoY (IMF SDMX 3.0 api.imf.org → index-to-YoY, weekly batch)
//   CAD: extended-data/CAD.json — CPI YoY (IMF SDMX 3.0 api.imf.org → index-to-YoY, weekly batch)
//   CHF: extended-data/CHF.json — CPI YoY (IMF SDMX 3.0 api.imf.org → index-to-YoY, weekly batch)
//   NZD: extended-data/NZD.json — CPI YoY (OECD Data Explorer fallback → index-to-YoY, weekly batch)
//
// Note: extended-data/{CCY}.json is written weekly by update-inflation-expectations.yml
// (runs Wednesdays 07:00 UTC). G6 data sourced primarily from IMF SDMX 3.0 CPI dataset
// (api.imf.org/external/sdmx/3.0, dataflow IMF.STA/CPI, monthly index → 12-month YoY).
// USD/EUR use live FRED breakevens at open.
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('rcm-modal-css')) return;

  // ── CSS ─────────────────────────────────────────────────────────────────
  const s = document.createElement('style');
  s.id = 'rcm-modal-css';
  s.textContent = `
/* ── Animations ── */
@keyframes rcm-fi{from{opacity:0}to{opacity:1}}
@keyframes rcm-su{from{transform:translateY(12px);opacity:0}to{transform:none;opacity:1}}
@keyframes rcm-pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* ── Backdrop ── */
#rcm-bd{position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.85);display:flex;align-items:center;justify-content:center;padding:16px;animation:rcm-fi .15s ease;}
/* ── Modal shell ── */
#rcm-modal{background:#161b22;border:1px solid #30363d;border-radius:8px;width:min(940px,100%);height:min(640px,92vh);display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.04);animation:rcm-su .18s cubic-bezier(.16,1,.3,1);font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:#e6edf3;position:relative;}
#rcm-modal::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1f6feb 0%,#58a6ff 50%,#3fb950 100%);border-radius:8px 8px 0 0;}
/* ── Header ── */
#rcm-hd{display:flex;align-items:center;justify-content:space-between;padding:14px 18px 12px;border-bottom:1px solid #30363d;flex-shrink:0;background:#161b22;}
#rcm-hd-left{display:flex;flex-direction:column;gap:3px;}
.rcm-badge{display:inline-flex;align-items:center;gap:5px;font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#388bfd;margin-bottom:2px;}
.rcm-badge::before{content:'';width:6px;height:6px;border-radius:50%;background:#388bfd;flex-shrink:0;}
#rcm-title{font-size:14px;font-weight:600;color:#e6edf3;letter-spacing:-.01em;}
#rcm-sub{font-size:10px;color:#6e7681;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.02em;}
#rcm-close{background:none;border:none;color:#6e7681;font-size:18px;cursor:pointer;padding:5px 7px;border-radius:5px;line-height:1;transition:color .1s,background .1s;font-family:inherit;}
#rcm-close:hover{color:#e6edf3;background:#21262d;}
/* ── Metrics strip ── */
#rcm-metrics{display:flex;border-bottom:1px solid #30363d;flex-shrink:0;background:#0d1117;}
.rcm-mm{flex:1;padding:9px 16px;border-right:1px solid #30363d;display:flex;flex-direction:column;gap:1px;}
.rcm-mm:last-child{border-right:none;}
.rcm-mm-lbl{font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:600;color:#6e7681;text-transform:uppercase;letter-spacing:.09em;}
.rcm-mm-val{font-size:16px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#e6edf3;line-height:1;margin-top:2px;}
.rcm-mm-sub{font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#6e7681;margin-top:1px;}
/* ── Tabs ── */
#rcm-tabs{display:flex;padding:0 18px;border-bottom:1px solid #30363d;flex-shrink:0;background:#161b22;overflow-x:auto;scrollbar-width:none;}
#rcm-tabs::-webkit-scrollbar{display:none;}
.rcm-tab{font-size:11px;font-weight:500;padding:9px 14px;cursor:pointer;color:#6e7681;border-bottom:2px solid transparent;transition:color .12s;white-space:nowrap;user-select:none;}
.rcm-tab:hover{color:#8b949e;}
.rcm-tab.on{color:#e6edf3;border-bottom-color:#388bfd;}
/* ── Body ── */
#rcm-body{flex:1;min-height:0;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;background:#0d1117;scrollbar-width:thin;scrollbar-color:#444c56 transparent;}
#rcm-body::-webkit-scrollbar{width:4px;}
#rcm-body::-webkit-scrollbar-track{background:transparent;}
#rcm-body::-webkit-scrollbar-thumb{background:#444c56;border-radius:2px;}
#rcm-body::-webkit-scrollbar-thumb:hover{background:#6e7681;}
.rcm-panel{display:none;}
.rcm-panel.on{display:flex;flex:1;flex-direction:column;min-height:0;}
/* ── Card wrapper ── */
.rcm-cw{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 14px;margin-bottom:10px;}
.rcm-cw:last-child{margin-bottom:0;}
.rcm-ct{font-size:9.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#6e7681;letter-spacing:.04em;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #30363d;text-transform:uppercase;}
/* ── Table ── */
.rcm-tbl{width:100%;border-collapse:collapse;font-size:11.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-tbl thead th{text-align:right;color:#6e7681;font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:0 10px 7px;border-bottom:1px solid #30363d;}
.rcm-tbl thead th:first-child{text-align:left;}
.rcm-tbl tbody tr{transition:background .08s;}
.rcm-tbl tbody tr:nth-child(even) td{background:rgba(255,255,255,.015);}
.rcm-tbl tbody tr:hover td{background:rgba(88,166,255,.05);}
.rcm-tbl td{text-align:right;padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle;}
.rcm-tbl td:first-child{text-align:left;}
.rcm-tbl tr:last-child td{border-bottom:none;}
/* #1 badge on best row */
.rcm-tbl .rcm-best td:first-child::before{content:'#1';display:inline-block;margin-right:7px;font-size:8px;font-weight:700;background:#3fb950;color:#0d1117;padding:1px 4px;border-radius:3px;vertical-align:middle;}
/* ── Real rate coloring ── */
.rr-pos2{color:#3fb950;font-weight:700;}
.rr-pos1{color:#26a641;}
.rr-neg1{color:#e07070;}
.rr-neg2{color:#f85149;font-weight:700;}
.rr-flat{color:#6e7681;}
/* ── OIS bias chip ── */
.rcm-bias{display:inline-flex;align-items:center;gap:3px;font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;padding:2px 6px;border-radius:3px;letter-spacing:.04em;white-space:nowrap;}
.rcm-bias-hike{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid rgba(63,185,80,.25);}
.rcm-bias-cut{background:rgba(248,81,73,.15);color:#f85149;border:1px solid rgba(248,81,73,.25);}
.rcm-bias-hold{background:rgba(139,148,158,.10);color:#8b949e;border:1px solid rgba(139,148,158,.18);}
/* ── Live dot ── */
.rcm-live-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:#3fb950;margin-left:3px;vertical-align:middle;animation:rcm-pulse 2s ease-in-out infinite;}
/* ── Real Rate Matrix ── */
#rcm-matrix-wrap{overflow:auto;flex:1;min-height:0;scrollbar-width:thin;scrollbar-color:#444c56 transparent;}
#rcm-matrix-wrap::-webkit-scrollbar{width:4px;height:4px;}
#rcm-matrix-wrap::-webkit-scrollbar-track{background:transparent;}
#rcm-matrix-wrap::-webkit-scrollbar-thumb{background:#444c56;border-radius:2px;}
#rcm-matrix-wrap::-webkit-scrollbar-thumb:hover{background:#6e7681;}
.rcm-matrix{border-collapse:collapse;font-size:10.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);width:100%;}
.rcm-matrix th{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:6px 8px;color:#6e7681;text-align:center;white-space:nowrap;background:#161b22;position:sticky;top:0;z-index:2;border-bottom:1px solid #30363d;}
.rcm-matrix th.row-head{text-align:left;min-width:46px;}
.rcm-matrix td{padding:5px 8px;text-align:center;border:1px solid rgba(255,255,255,.04);font-size:10.5px;min-width:58px;transition:filter .1s;}
.rcm-matrix td:hover{filter:brightness(1.4);}
.rcm-matrix td.diag{background:#2d333b;color:#8b949e;font-size:10px;font-weight:600;}
.rcm-matrix td.row-head{text-align:left;color:#8b949e;font-weight:700;font-size:10.5px;background:#161b22;border:none;position:sticky;left:0;z-index:1;}
.rcm-matrix td.empty{background:transparent;border:none;}
/* matrix cell shading */
.rcm-cell-pos-hi{background:rgba(63,185,80,.25);color:#3fb950;font-weight:700;}
.rcm-cell-pos{background:rgba(63,185,80,.10);color:#3fb950;}
.rcm-cell-neg-hi{background:rgba(248,81,73,.25);color:#f85149;font-weight:700;}
.rcm-cell-neg{background:rgba(248,81,73,.10);color:#f85149;}
.rcm-cell-flat{color:#6e7681;}
/* matrix legend */
.rcm-matrix-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#6e7681;margin-top:10px;padding-top:10px;border-top:1px solid #30363d;align-items:center;}
.rcm-legend-sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:5px;}
/* ── Pair detail ── */
.rcm-pd-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;}
.rcm-pd-kv{display:flex;flex-direction:column;gap:3px;}
.rcm-pd-lbl{font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#6e7681;}
.rcm-pd-val{font-size:15px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1;}
.rcm-pd-sub{font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#6e7681;margin-top:1px;}
.rcm-sustain{border-radius:5px;padding:9px 12px;font-size:10.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.5;margin-top:4px;}
.rcm-sustain-ok{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.22);color:#3fb950;}
.rcm-sustain-warn{background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.22);color:#d29922;}
.rcm-sustain-bad{background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.22);color:#f85149;}
.rcm-src-note{font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#6e7681;margin-top:10px;padding-top:9px;border-top:1px solid #30363d;line-height:1.6;}
.rcm-loading{display:flex;align-items:center;justify-content:center;flex:1;color:#6e7681;font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.06em;}
@media(max-width:640px){
  #rcm-bd{padding:0;align-items:flex-end;}
  #rcm-modal{width:100%;height:94vh;border-radius:12px 12px 0 0;}
  #rcm-metrics{flex-wrap:wrap;}
  .rcm-mm{flex:1 1 50%;border-right:none;border-bottom:1px solid #30363d;}
  .rcm-mm-val{font-size:12px;}
  .rcm-pd-grid{grid-template-columns:1fr 1fr;}
  #rcm-body{padding:10px;}
}
`;
  document.head.appendChild(s);
})();

// ── Constants ───────────────────────────────────────────────────────────────
const _RCM_G8 = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'];
const _RCM_CB = { USD: 'Fed', EUR: 'ECB', GBP: 'BoE', JPY: 'BoJ', AUD: 'RBA', CAD: 'BoC', CHF: 'SNB', NZD: 'RBNZ' };

// Inflation expectation source labels — shown in the source column for transparency
const _RCM_IE_SRC = {
  USD: 'FRED T5YIE · 5Y breakeven',
  EUR: 'FRED T5YIFR · EUR 5Y5Y swap',
  GBP: 'CPI YoY · IMF SDMX',
  JPY: 'CPI YoY · IMF SDMX',
  AUD: 'CPI YoY · IMF SDMX',
  CAD: 'CPI YoY · IMF SDMX',
  CHF: 'CPI YoY · IMF SDMX',
  NZD: 'CPI YoY · OECD Explorer',
};

// ── State ───────────────────────────────────────────────────────────────────
let _rcmData = null;           // cached computed data
let _rcmFetchPromise = null;   // in-flight promise — prevents duplicate concurrent fetches
let _rcmFetchedAt = 0;         // timestamp of last successful fetch (ms)
const _RCM_TTL = 15 * 60 * 1000; // 15-minute TTL (matches intraday-data cadence)
let _rcmActiveTab = 'breakdown';
let _rcmActivePair = null;

// ── FRED CSV fetch — non-blocking live enhancement ───────────────────────────
// FRED provides public CSVs at: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
// Used to UPGRADE extended-data values with live market-implied breakevens for USD/EUR.
// Called with a short timeout — never blocks the primary render path.
// NOTE: cache:'default' intentionally allows the browser HTTP cache (typically 5-30min
// for FRED responses) so repeated opens within a session avoid a redundant full download.
async function _rcmFredLatest(seriesId) {
  try {
    const url = `https://fred.stlouisfed.org/graph/fredgraph.csv?id=${seriesId}`;
    const r = await fetch(url); // cache:'default' — reuses browser HTTP cache when fresh
    if (!r.ok) return { val: null, date: null };
    const text = await r.text();
    const lines = text.trim().split('\n').filter(l => !l.startsWith('DATE'));
    for (let i = lines.length - 1; i >= 0; i--) {
      const [date, val] = lines[i].split(',');
      if (val && val.trim() !== '.' && val.trim() !== '') {
        return { val: parseFloat(val), date: date.trim() };
      }
    }
    return { val: null, date: null };
  } catch {
    return { val: null, date: null };
  }
}

// ── Data assembly ─────────────────────────────────────────────────────────────
// Performance design (v1.5):
//
//   PHASE 1 — same-origin parallel fetch (all 8 rates + all 8 extended-data +
//             meetings.json + quotes.json in one Promise.all).
//             Same-origin requests share HTTP/2 multiplexing → typically 40-80ms total.
//             Modal renders with full data as soon as Phase 1 completes.
//
//   PHASE 2 — FRED live upgrade (non-blocking, fire-and-forget).
//             Runs concurrently with Phase 1 and races to finish.
//             If FRED responds before Phase 1 completes → live values used in first render.
//             If FRED is slow → extended-data values used in first render (no spinner gap),
//             then a silent background patch replaces USD/EUR infl.exp with live values
//             and re-renders the metrics bar + table without the user noticing a delay.
//
//   CACHE — timestamp-based TTL replaces the always-invalidate 15-min interval.
//           _rcmData persists across modal opens for up to 15 min. The interval only
//           triggers a background re-fetch when the cache is actually stale — never
//           forces a cold start on the next user-initiated open.
//
//   IN-FLIGHT DEDUP — _rcmFetchPromise stores the active Promise so that rapid
//           double-clicks or tab switches share one fetch instead of spawning
//           two parallel fetches that race to overwrite _rcmData.

async function _rcmFetchData() {
  // Dedup: return the in-flight promise if already fetching
  if (_rcmFetchPromise) return _rcmFetchPromise;

  _rcmFetchPromise = (async () => {
    try {
      // ── PHASE 1: all same-origin fetches in parallel ───────────────────────
      const extKeys = _RCM_G8; // 8 extended-data files
      const [rateResults, extResults, meetingsRes, quotesRes] = await Promise.all([
        // 8 × rates/*.json
        Promise.all(_RCM_G8.map(ccy =>
          fetch(`./rates/${ccy}.json`).then(r => r.ok ? r.json() : null).catch(() => null)
        )),
        // 8 × extended-data/*.json
        Promise.all(extKeys.map(ccy =>
          fetch(`./extended-data/${ccy}.json`).then(r => r.ok ? r.json() : null).catch(() => null)
        )),
        // meetings.json
        fetch('./meetings-data/meetings.json').then(r => r.ok ? r.json() : null).catch(() => null),
        // quotes.json (HV30)
        fetch('./intraday-data/quotes.json').then(r => r.ok ? r.json() : null).catch(() => null),
      ]);

      // Parse nominal rates
      const nominalRates = {};
      _RCM_G8.forEach((ccy, i) => {
        const d = rateResults[i];
        if (d?.observations?.[0]?.value != null) {
          nominalRates[ccy] = { rate: parseFloat(d.observations[0].value), date: d.observations[0].date };
        }
      });

      // Parse inflation expectations from extended-data (all 8 currencies including USD/EUR)
      // USD/EUR will be upgraded with live FRED values in Phase 2 if available.
      const inflExp = {};
      extKeys.forEach((ccy, i) => {
        const d = extResults[i];
        const ie = d?.data?.inflationExpectations;
        const ieDate = d?.dates?.inflationExpectations;
        if (ie != null) {
          inflExp[ccy] = { val: ie, date: ieDate || null, live: false };
        }
      });

      // Parse OIS bias
      const biasMap = {};
      if (meetingsRes?.meetings) {
        for (const [ccy, m] of Object.entries(meetingsRes.meetings)) {
          biasMap[ccy] = {
            bias: m.bias || 'hold',
            hikeProb: m.hikeProb ?? null,
            cutProb: m.cutProb ?? null,
            method: m.biasMethod || '',
          };
        }
      }

      // Parse HV30
      const hv30Map = {};
      if (quotesRes?.hv30) Object.assign(hv30Map, quotesRes.hv30);

      // Compute real rates from Phase 1 data (extended-data infl.exp)
      const realRates = {};
      for (const ccy of _RCM_G8) {
        const nom = nominalRates[ccy]?.rate;
        const ie  = inflExp[ccy]?.val;
        realRates[ccy] = (nom != null && ie != null) ? parseFloat((nom - ie).toFixed(3)) : null;
      }

      // Commit Phase 1 result — modal can render immediately from here
      _rcmData = { nominalRates, inflExp, biasMap, hv30Map, realRates };
      _rcmFetchedAt = Date.now();

      // ── PHASE 2: FRED live upgrade (non-blocking) ─────────────────────────
      // Fire-and-forget. If FRED responds, silently upgrade USD/EUR infl.exp
      // and re-render the metrics bar + breakdown table. No spinner shown.
      Promise.all([
        _rcmFredLatest('T5YIE'),
        _rcmFredLatest('T5YIFR'),
      ]).then(([fredUSD, fredEUR]) => {
        if (!_rcmData) return; // cache was invalidated while FRED was in-flight
        let upgraded = false;
        if (fredUSD.val != null) {
          _rcmData.inflExp['USD'] = { val: fredUSD.val, date: fredUSD.date, live: true };
          upgraded = true;
        }
        if (fredEUR.val != null) {
          _rcmData.inflExp['EUR'] = { val: fredEUR.val, date: fredEUR.date, live: true };
          upgraded = true;
        }
        if (!upgraded) return;
        // Recompute real rates for USD and EUR with live infl.exp
        for (const ccy of ['USD', 'EUR']) {
          const nom = _rcmData.nominalRates[ccy]?.rate;
          const ie  = _rcmData.inflExp[ccy]?.val;
          _rcmData.realRates[ccy] = (nom != null && ie != null) ? parseFloat((nom - ie).toFixed(3)) : null;
        }
        // Re-render only if the modal is currently open (avoid invisible background work)
        const bd = document.getElementById('rcm-bd');
        if (bd && bd.style.display !== 'none') _rcmRender();
      }).catch(() => {}); // FRED failure is silent — extended-data values remain

    } finally {
      _rcmFetchPromise = null; // clear in-flight lock regardless of outcome
    }
  })();

  return _rcmFetchPromise;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function _rcmRrClass(rr) {
  if (rr == null) return 'rr-flat';
  if (rr >= 1.0)  return 'rr-pos2';
  if (rr >= 0.0)  return 'rr-pos1';
  if (rr >= -1.0) return 'rr-neg1';
  return 'rr-neg2';
}

function _rcmRrFmt(rr) {
  if (rr == null) return '—';
  return (rr >= 0 ? '+' : '') + rr.toFixed(2) + '%';
}

function _rcmBiasChip(biasObj) {
  if (!biasObj) return '<span class="rcm-bias rcm-bias-hold">Hold</span>';
  const { bias, hikeProb, cutProb } = biasObj;
  let pct = '';
  if (bias === 'hike' && hikeProb != null && hikeProb > 0) pct = ` ${hikeProb}%`;
  if (bias === 'cut'  && cutProb  != null && cutProb  > 0) pct = ` ${cutProb}%`;
  const label = bias === 'hike' ? 'Hike' : bias === 'cut' ? 'Cut' : 'Hold';
  const cls   = bias === 'hike' ? 'rcm-bias-hike' : bias === 'cut' ? 'rcm-bias-cut' : 'rcm-bias-hold';
  return `<span class="rcm-bias ${cls}">${label}${pct}</span>`;
}

function _rcmDateAge(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    const days = Math.round((Date.now() - d) / 86400000);
    if (days < 2)  return 'today';
    if (days < 7)  return `${days}d ago`;
    if (days < 60) return `${Math.round(days / 7)}w ago`;
    return `${Math.round(days / 30)}mo ago`;
  } catch { return ''; }
}

// ── Tab 1: Rates Breakdown ───────────────────────────────────────────────────
function _rcmRenderBreakdown() {
  const d = _rcmData;
  if (!d) return '<div class="rcm-loading">Loading...</div>';

  // Sort currencies by real rate descending (carry attractiveness)
  const sorted = [..._RCM_G8].sort((a, b) => {
    const ra = d.realRates[a] ?? -99;
    const rb = d.realRates[b] ?? -99;
    return rb - ra;
  });

  const rows = sorted.map((ccy, idx) => {
    const nom  = d.nominalRates[ccy]?.rate;
    const ie   = d.inflExp[ccy];
    const rr   = d.realRates[ccy];
    const bias = d.biasMap[ccy];
    const rrCls = _rcmRrClass(rr);
    const nomFmt = nom != null ? nom.toFixed(2) + '%' : '—';
    const ieFmt  = ie  ? ie.val.toFixed(2) + '%' : '—';
    const rrFmt  = _rcmRrFmt(rr);
    const isLive  = ie?.live ? '<span title="Market-implied (FRED breakeven)" style="color:var(--up,#26a69a);font-size:8px;">&#x25CF;</span>' : '';
    const dateAge = _rcmDateAge(ie?.date);
    const srcLabel = _RCM_IE_SRC[ccy] || '';
    const srcTitle = `${srcLabel}${ie?.date ? ' · ' + ie.date : ''}`;
    const isTR = idx === 0 ? ' class="rcm-best"' : '';

    return `<tr${isTR} title="${ccy} — Real rate = ${nomFmt} nominal − ${ieFmt} infl.exp = ${rrFmt}">
      <td style="font-weight:700;color:var(--text,#d1d4dc);">${_RCM_CB[ccy]} (${ccy})</td>
      <td>${nomFmt}</td>
      <td title="${srcTitle}">${ieFmt} ${isLive}</td>
      <td title="${srcTitle}">${dateAge}</td>
      <td class="${rrCls}" style="font-weight:600;">${rrFmt}</td>
      <td>${_rcmBiasChip(bias)}</td>
    </tr>`;
  }).join('');

  const liveNote = 'USD/EUR: FRED market-implied breakeven (live, daily). GBP/JPY/AUD/CAD/CHF: CPI YoY (IMF SDMX 3.0, weekly batch — updated Wednesdays). NZD: OECD Data Explorer (weekly batch). ' +
    'CPI YoY and 5Y breakeven are different methodologies — cross-currency real rate comparisons carry wider uncertainty for non-USD/EUR legs. ' +
    'Data age column shows observation date — treat figures older than 6 months as indicative only.';

  return `<div class="rcm-cw" style="flex:1;min-height:0;overflow:auto;">
    <div class="rcm-ct">Real Rate Carry Ranking — G8 Central Banks · sorted by real rate descending</div>
    <table class="rcm-tbl" aria-label="Real rate carry ranking by currency">
      <thead>
        <tr>
          <th scope="col" style="text-align:left;">Central Bank</th>
          <th scope="col">Nominal</th>
          <th scope="col">Infl. Exp.</th>
          <th scope="col">Data Age</th>
          <th scope="col">Real Rate</th>
          <th scope="col">OIS Bias</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="rcm-src-note">${liveNote}</div>
  </div>`;
}

// ── Tab 2: Real Rate Matrix ──────────────────────────────────────────────────
// Upper triangle: real rate of ROW minus real rate of COLUMN (long row, short col)
// Positive = row currency has higher real rate (carry advantage long row / short col)
// Diagonal: absolute real rate of the currency
function _rcmRenderMatrix() {
  const d = _rcmData;
  if (!d) return '<div class="rcm-loading">Loading...</div>';

  const G8 = _RCM_G8;

  function cellClass(diff) {
    if (diff == null) return 'rcm-cell-flat';
    if (diff >= 1.5)  return 'rcm-cell-pos-hi';
    if (diff >= 0.15) return 'rcm-cell-pos';
    if (diff <= -1.5) return 'rcm-cell-neg-hi';
    if (diff <= -0.15)return 'rcm-cell-neg';
    return 'rcm-cell-flat';
  }

  function cellFmt(diff) {
    if (diff == null) return '—';
    if (Math.abs(diff) < 0.01) return '0';
    return (diff > 0 ? '+' : '') + diff.toFixed(2) + '%';
  }

  const header = `<tr><th class="row-head" scope="col">Long ↓ / Short →</th>${G8.map(c => `<th scope="col">${c}</th>`).join('')}</tr>`;

  const rows = G8.map(rowCcy => {
    const cells = G8.map(colCcy => {
      if (rowCcy === colCcy) {
        // Diagonal: absolute real rate
        const rr = d.realRates[rowCcy];
        const fmt = rr != null ? (rr >= 0 ? '+' : '') + rr.toFixed(2) + '%' : '—';
        return `<td class="diag" title="${rowCcy} real rate: ${fmt}">${fmt}</td>`;
      }
      const rrRow = d.realRates[rowCcy];
      const rrCol = d.realRates[colCcy];
      const diff  = (rrRow != null && rrCol != null) ? parseFloat((rrRow - rrCol).toFixed(3)) : null;
      const cls   = cellClass(diff);
      const fmt   = cellFmt(diff);
      const tip   = diff != null
        ? `Long ${rowCcy} (${_rcmRrFmt(rrRow)}) / Short ${colCcy} (${_rcmRrFmt(rrCol)}) — real spread: ${cellFmt(diff)}`
        : `${rowCcy}/${colCcy} — insufficient data`;
      return `<td class="${cls}" title="${tip}">${fmt}</td>`;
    }).join('');
    return `<tr><td class="row-head">${rowCcy}</td>${cells}</tr>`;
  }).join('');

  const legend = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text3,#6b7280);margin-top:8px;">
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.28);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong real carry (&ge;1.5%)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(38,166,154,.14);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Positive real spread</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.14);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Negative real spread</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:rgba(239,83,80,.28);border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Strong real drag (&le;-1.5%)</span>
      <span>Diagonal = absolute real rate</span>
    </div>`;

  return `<div class="rcm-cw" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
    <div class="rcm-ct">Real Rate Differential Matrix · Long row / Short column · Cell = Row real rate − Column real rate</div>
    <div id="rcm-matrix-wrap" style="overflow:auto;flex:1;">
      <table class="rcm-matrix" aria-label="Real rate differential matrix G8 currencies">
        <thead>${header}</thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${legend}
    <div class="rcm-src-note">Real rate = Nominal CB rate − Inflation Expectation. Positive cell = long row currency earns higher real carry vs short column currency.</div>
  </div>`;
}

// ── Tab 3: Pair Detail ───────────────────────────────────────────────────────
function _rcmRenderPairDetail(longCcy, shortCcy) {
  const d = _rcmData;
  if (!d || !longCcy || !shortCcy) {
    return `<div class="rcm-loading">Select a pair from the Carry Ranking to view detail.</div>`;
  }

  const nomL = d.nominalRates[longCcy]?.rate;
  const nomS = d.nominalRates[shortCcy]?.rate;
  const ieL  = d.inflExp[longCcy]?.val;
  const ieS  = d.inflExp[shortCcy]?.val;
  const rrL  = d.realRates[longCcy];
  const rrS  = d.realRates[shortCcy];
  const biasL = d.biasMap[longCcy];
  const biasS = d.biasMap[shortCcy];

  const nomSpread  = (nomL != null && nomS != null) ? nomL - nomS : null;
  const realSpread = (rrL  != null && rrS  != null) ? rrL  - rrS  : null;

  // HV30 for the pair
  // Uses FX market convention lookup table matching HV30_FX_PAIRS in fetch_intraday_quotes.py.
  // Pure alphabetical is WRONG for crosses: EUR/AUD key is 'euraud' not 'audeur',
  // GBP/CHF is 'gbpchf' not 'chfgbp', NZD/JPY is 'nzdjpy' not 'jpynzd'.
  function pairId(a, b) {
    const HV30_PAIRS = new Set([
      'eurusd','gbpusd','usdjpy','audusd','usdchf','usdcad','nzdusd',
      'eurgbp','eurjpy','eurchf','eurcad','euraud',
      'gbpjpy','gbpchf','gbpcad',
      'audjpy','audnzd','audchf',
      'cadjpy','chfjpy','nzdjpy',
      'eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf',
    ]);
    const c1 = (a + b).toLowerCase();
    const c2 = (b + a).toLowerCase();
    if (HV30_PAIRS.has(c1)) return c1;
    if (HV30_PAIRS.has(c2)) return c2;
    return a < b ? c1 : c2;
  }
  const pid  = pairId(longCcy, shortCcy);
  const hv30 = d.hv30Map[pid] ?? null;
  const nomCarryVol  = (hv30 && nomSpread  != null) ? (Math.abs(nomSpread)  / hv30).toFixed(2) : null;
  const realCarryVol = (hv30 && realSpread != null) ? (Math.abs(realSpread) / hv30).toFixed(2) : null;

  // Sustainability assessment
  let sustainCls = 'rcm-sustain-ok', sustainText = '';
  if (rrL != null && rrS != null) {
    if (rrL > 0 && rrS < 0) {
      sustainText = `Sustainable — long leg has positive real rate (+${rrL.toFixed(2)}%), short leg negative (${rrS.toFixed(2)}%). Carry unlikely to erode via inflation differential.`;
      sustainCls  = 'rcm-sustain-ok';
    } else if (rrL < 0) {
      sustainText = `Carry trap risk — long leg real rate is negative (${rrL.toFixed(2)}%). Nominal carry may be eroded by inflation; real return to holder is negative.`;
      sustainCls  = 'rcm-sustain-bad';
    } else if (rrL > 0 && rrS > 0) {
      if (rrL > rrS) {
        sustainText = `Moderate — both legs have positive real rates. Spread ${_rcmRrFmt(realSpread)} real vs ${nomSpread != null ? (nomSpread >= 0 ? '+' : '') + nomSpread.toFixed(2) + '%' : '—'} nominal. Watch for inflation convergence.`;
        sustainCls  = 'rcm-sustain-warn';
      } else {
        sustainText = `Negative real spread — short leg has higher real rate. Nominal carry favors long ${longCcy}, but real carry favors long ${shortCcy}.`;
        sustainCls  = 'rcm-sustain-bad';
      }
    } else {
      sustainText = 'Insufficient real rate data for sustainability assessment.';
      sustainCls  = 'rcm-sustain-warn';
    }
  }

  function fmt(v, suffix) { return v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + (suffix || '') : '—'; }

  return `
  <div class="rcm-cw">
    <div class="rcm-ct">${longCcy}/${shortCcy} — Real Rate Carry Analysis · Long ${longCcy} / Short ${shortCcy}</div>
    <div class="rcm-pd-grid">
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Nominal carry</div>
        <div class="rcm-pd-val ${_rcmRrClass(nomSpread)}">${fmt(nomSpread, '%')}</div>
        <div class="rcm-pd-sub">${_RCM_CB[longCcy]} ${nomL != null ? nomL.toFixed(2) + '%' : '—'} − ${_RCM_CB[shortCcy]} ${nomS != null ? nomS.toFixed(2) + '%' : '—'}</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Real carry</div>
        <div class="rcm-pd-val ${_rcmRrClass(realSpread)}">${fmt(realSpread, '%')}</div>
        <div class="rcm-pd-sub">After inflation expectations</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Long real rate (${longCcy})</div>
        <div class="rcm-pd-val ${_rcmRrClass(rrL)}">${_rcmRrFmt(rrL)}</div>
        <div class="rcm-pd-sub">${nomL != null ? nomL.toFixed(2) + '%' : '—'} nom − ${ieL != null ? ieL.toFixed(2) + '%' : '—'} infl.exp</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Short real rate (${shortCcy})</div>
        <div class="rcm-pd-val ${_rcmRrClass(rrS)}">${_rcmRrFmt(rrS)}</div>
        <div class="rcm-pd-sub">${nomS != null ? nomS.toFixed(2) + '%' : '—'} nom − ${ieS != null ? ieS.toFixed(2) + '%' : '—'} infl.exp</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Nominal carry/vol</div>
        <div class="rcm-pd-val">${nomCarryVol ?? '—'}</div>
        <div class="rcm-pd-sub">HV30 ${hv30 ? hv30.toFixed(1) + '%' : 'n/a'}</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">Real carry/vol</div>
        <div class="rcm-pd-val">${realCarryVol ?? '—'}</div>
        <div class="rcm-pd-sub">Real spread / HV30</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">OIS bias (${longCcy})</div>
        <div class="rcm-pd-val" style="font-size:12px;">${_rcmBiasChip(biasL)}</div>
        <div class="rcm-pd-sub">${biasL?.method || '—'}</div>
      </div>
      <div class="rcm-pd-kv"><div class="rcm-pd-lbl">OIS bias (${shortCcy})</div>
        <div class="rcm-pd-val" style="font-size:12px;">${_rcmBiasChip(biasS)}</div>
        <div class="rcm-pd-sub">${biasS?.method || '—'}</div>
      </div>
    </div>
    ${sustainText ? `<div class="rcm-sustain ${sustainCls}">${sustainText}</div>` : ''}
    <div class="rcm-src-note">
      Infl. Exp. source — ${longCcy}: ${_RCM_IE_SRC[longCcy] || '—'} · ${shortCcy}: ${_RCM_IE_SRC[shortCcy] || '—'}<br>
      Real carry/vol = |real spread| / HV30 (30-day realised vol, annualised)
    </div>
  </div>`;
}

// ── Modal render ─────────────────────────────────────────────────────────────
function _rcmRender() {
  const modal = document.getElementById('rcm-body');
  if (!modal) return;

  let content = '';
  if (_rcmActiveTab === 'breakdown') {
    content = _rcmRenderBreakdown();
  } else if (_rcmActiveTab === 'matrix') {
    content = _rcmRenderMatrix();
  } else if (_rcmActiveTab === 'detail') {
    const [l, s] = (_rcmActivePair || '/').split('/');
    content = _rcmRenderPairDetail(l, s);
  }

  modal.innerHTML = `<div class="rcm-panel on" style="overflow:auto;">${content}</div>`;

  // Update summary metrics bar
  _rcmUpdateMetrics();

  // Update tabs
  document.querySelectorAll('.rcm-tab').forEach(t => {
    t.classList.toggle('on', t.dataset.tab === _rcmActiveTab);
  });
}

function _rcmUpdateMetrics() {
  const d = _rcmData;
  if (!d) return;

  // Best real rate carry
  const sorted = [..._RCM_G8].filter(c => d.realRates[c] != null).sort((a, b) => d.realRates[b] - d.realRates[a]);
  const best  = sorted[0];
  const worst = sorted[sorted.length - 1];

  const elBest = document.getElementById('rcm-mm-best');
  const elWorst = document.getElementById('rcm-mm-worst');
  const elSpread = document.getElementById('rcm-mm-spread');
  const elSrc = document.getElementById('rcm-mm-src');

  if (elBest && best) {
    elBest.querySelector('.rcm-mm-val').textContent = _rcmRrFmt(d.realRates[best]);
    elBest.querySelector('.rcm-mm-lbl').textContent = `Best real (${best})`;
    elBest.querySelector('.rcm-mm-sub').textContent = _RCM_CB[best];
  }
  if (elWorst && worst) {
    elWorst.querySelector('.rcm-mm-val').textContent = _rcmRrFmt(d.realRates[worst]);
    elWorst.querySelector('.rcm-mm-lbl').textContent = `Worst real (${worst})`;
    elWorst.querySelector('.rcm-mm-sub').textContent = _RCM_CB[worst];
  }
  if (elSpread && best && worst && d.realRates[best] != null && d.realRates[worst] != null) {
    const spread = d.realRates[best] - d.realRates[worst];
    elSpread.querySelector('.rcm-mm-val').textContent = '+' + spread.toFixed(2) + '%';
    elSpread.querySelector('.rcm-mm-sub').textContent = `${best}−${worst} spread`;
  }
  if (elSrc) {
    const usdFresh = d.inflExp['USD']?.live;
    elSrc.querySelector('.rcm-mm-val').textContent = usdFresh ? 'Live' : 'Batch';
    elSrc.querySelector('.rcm-mm-sub').textContent = usdFresh ? 'USD/EUR: FRED live' : 'extended-data batch';
  }
}

// ── Public API ───────────────────────────────────────────────────────────────
// Called from fetchCarryRanking() when user clicks a row
async function openRealCarryModal(longCcy, shortCcy) {
  // Build DOM if not present
  if (!document.getElementById('rcm-bd')) {
    _rcmBuildDOM();
  }

  // If a pair is provided, switch to detail tab
  if (longCcy && shortCcy) {
    _rcmActivePair = `${longCcy}/${shortCcy}`;
    _rcmActiveTab  = 'detail';
  } else {
    _rcmActiveTab = 'breakdown';
  }

  const bd = document.getElementById('rcm-bd');
  bd.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  if (_rcmData) {
    // Cache hit — render immediately, no spinner shown
    _rcmRender();
  } else {
    // Cold start — show loading state and await Phase 1
    const body = document.getElementById('rcm-body');
    if (body) body.innerHTML = '<div class="rcm-loading">Fetching rates & inflation data...</div>';
    await _rcmFetchData();
    _rcmRender();
  }
}

function _rcmBuildDOM() {
  const bd = document.createElement('div');
  bd.id = 'rcm-bd';
  bd.style.display = 'none';
  bd.setAttribute('role', 'dialog');
  bd.setAttribute('aria-modal', 'true');
  bd.setAttribute('aria-label', 'Real Rate Carry Analysis');

  bd.innerHTML = `
  <div id="rcm-modal" role="document">
    <div id="rcm-hd">
      <div id="rcm-hd-left">
        <span class="rcm-badge">Real Rate Carry</span>
        <div id="rcm-title">Real Rate Carry Analysis</div>
        <div id="rcm-sub">Nominal CB rate &minus; Inflation Expectation (breakeven / CPI proxy) &middot; G8</div>
      </div>
      <button id="rcm-close" aria-label="Close real rate carry modal">&times;</button>
    </div>
    <div id="rcm-metrics">
      <div class="rcm-mm" id="rcm-mm-best">
        <div class="rcm-mm-lbl">Best real (—)</div>
        <div class="rcm-mm-val rr-pos1">—</div>
        <div class="rcm-mm-sub">—</div>
      </div>
      <div class="rcm-mm" id="rcm-mm-worst">
        <div class="rcm-mm-lbl">Worst real (—)</div>
        <div class="rcm-mm-val rr-neg1">—</div>
        <div class="rcm-mm-sub">—</div>
      </div>
      <div class="rcm-mm" id="rcm-mm-spread">
        <div class="rcm-mm-lbl">Max Real Spread</div>
        <div class="rcm-mm-val">—</div>
        <div class="rcm-mm-sub">—</div>
      </div>
      <div class="rcm-mm" id="rcm-mm-src">
        <div class="rcm-mm-lbl">Data Source</div>
        <div class="rcm-mm-val" style="font-size:12px;">—</div>
        <div class="rcm-mm-sub">—</div>
      </div>
    </div>
    <div id="rcm-tabs" role="tablist" aria-label="Real carry analysis tabs">
      <div class="rcm-tab on" role="tab" aria-selected="true"  data-tab="breakdown">Rates Breakdown</div>
      <div class="rcm-tab"    role="tab" aria-selected="false" data-tab="matrix">Real Rate Matrix</div>
      <div class="rcm-tab"    role="tab" aria-selected="false" data-tab="detail">Pair Detail</div>
    </div>
    <div id="rcm-body"></div>
  </div>`;

  document.body.appendChild(bd);

  // Close handlers
  document.getElementById('rcm-close').addEventListener('click', _rcmClose);
  bd.addEventListener('click', e => { if (e.target === bd) _rcmClose(); });
  document.addEventListener('keydown', _rcmEsc, { capture: true });

  // Tab switching
  bd.querySelectorAll('.rcm-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      _rcmActiveTab = tab.dataset.tab;
      bd.querySelectorAll('.rcm-tab').forEach(t => {
        t.classList.toggle('on', t === tab);
        t.setAttribute('aria-selected', String(t === tab));
      });
      _rcmRender();
    });
  });
}

function _rcmClose() {
  const bd = document.getElementById('rcm-bd');
  if (bd) bd.style.display = 'none';
  document.body.style.overflow = '';
}

function _rcmEsc(e) {
  if (e.key === 'Escape') {
    const bd = document.getElementById('rcm-bd');
    if (bd && bd.style.display !== 'none') _rcmClose();
  }
}

// ── Background refresh — TTL-based, never forces a cold start on next open ───
// Checks every 15 minutes. If the cache is stale (older than TTL):
//   - Modal is open: refresh data and re-render silently.
//   - Modal is closed: fetch in the background so the next open is instant.
// The cache is NOT blindly nulled — it stays valid until Phase 1 of the new
// fetch completes, so a user opening the modal mid-refresh still gets
// the previous data immediately rather than seeing the loading spinner.
setInterval(async () => {
  const isStale = (Date.now() - _rcmFetchedAt) >= _RCM_TTL;
  if (!isStale) return;

  const bd = document.getElementById('rcm-bd');
  const isOpen = bd && bd.style.display !== 'none';

  // Fetch fresh data — _rcmData stays intact until new data is committed
  await _rcmFetchData();

  if (isOpen) _rcmRender(); // silently update if modal is visible
}, 60 * 1000); // check every 60s; actual refresh only fires when TTL is exceeded
