// ═══════════════════════════════════════════════════════════════════════════
// REAL RATE CARRY MODAL  v1.2
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
//   GBP: extended-data/GBP.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//   JPY: extended-data/JPY.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//   AUD: extended-data/AUD.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//   CAD: extended-data/CAD.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//   CHF: extended-data/CHF.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//   NZD: extended-data/NZD.json — CPI YoY (ForexFactory calendar actuals, weekly batch)
//
// Note: extended-data/{CCY}.json is written weekly by update-inflation-expectations.yml
// (runs Mondays 06:00 UTC). G6 data comes from economic-data/ calendar actuals
// (update_pmi_from_calendar.py). USD/EUR use live FRED breakevens at open.
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('rcm-modal-css')) return;

  // ── CSS ─────────────────────────────────────────────────────────────────
  const s = document.createElement('style');
  s.id = 'rcm-modal-css';
  s.textContent = `
#rcm-bd{position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.80);display:flex;align-items:center;justify-content:center;padding:12px;animation:rcm-fi .15s ease;}
@keyframes rcm-fi{from{opacity:0}to{opacity:1}}
@keyframes rcm-su{from{transform:translateY(14px);opacity:0}to{transform:none;opacity:1}}
#rcm-modal{background:var(--bg,#131722);border:1px solid rgba(255,255,255,.1);border-radius:10px;width:min(900px,100%);height:min(620px,92vh);display:flex;flex-direction:column;overflow:hidden;animation:rcm-su .2s ease;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text,#d1d4dc);}
#rcm-hd{display:flex;align-items:flex-start;justify-content:space-between;padding:13px 18px 11px;border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;}
#rcm-hd-left{display:flex;flex-direction:column;gap:2px;}
#rcm-title{font-size:13px;font-weight:600;color:var(--text,#d1d4dc);letter-spacing:.01em;}
#rcm-sub{font-size:9.5px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.02em;}
#rcm-close{background:none;border:none;color:var(--text3,#6b7280);font-size:20px;cursor:pointer;padding:4px 8px;border-radius:4px;line-height:1;transition:color .1s,background .1s;flex-shrink:0;}
#rcm-close:hover{color:var(--text,#d1d4dc);background:rgba(255,255,255,.08);}
#rcm-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.05);border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;}
.rcm-mm{background:var(--bg,#131722);padding:8px 14px;display:flex;flex-direction:column;gap:1px;}
.rcm-mm-lbl{font-size:8.5px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.07em;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-mm-val{font-size:14px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-mm-sub{font-size:8.5px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#rcm-tabs{display:flex;padding:0 16px;border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;overflow-x:auto;scrollbar-width:none;}
#rcm-tabs::-webkit-scrollbar{display:none;}
.rcm-tab{font-size:10.5px;padding:8px 12px;cursor:pointer;color:var(--text3,#6b7280);border-bottom:2px solid transparent;transition:color .15s;white-space:nowrap;user-select:none;}
.rcm-tab:hover{color:var(--text2,#9096a0);}
.rcm-tab.on{color:var(--text,#d1d4dc);border-bottom-color:var(--blue,#4f7fff);}
#rcm-body{flex:1;min-height:0;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;}
#rcm-body::-webkit-scrollbar{width:4px;}
#rcm-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,.10);border-radius:2px;}
.rcm-panel{display:none;}
.rcm-panel.on{display:flex;flex:1;flex-direction:column;min-height:0;}
.rcm-cw{background:#1e222d;border:1px solid rgba(255,255,255,.06);border-radius:6px;padding:11px 14px;margin-bottom:10px;}
.rcm-cw:last-child{margin-bottom:0;}
.rcm-ct{font-size:9.5px;color:var(--text2,#9096a0);margin-bottom:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.03em;}
/* ── Table ── */
.rcm-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-tbl th{text-align:right;color:var(--text3,#6b7280);font-weight:400;font-size:8.5px;text-transform:uppercase;letter-spacing:.06em;padding:5px 8px 5px;border-bottom:1px solid rgba(255,255,255,.08);}
.rcm-tbl th:first-child{text-align:left;}
.rcm-tbl td{text-align:right;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.04);}
.rcm-tbl td:first-child{text-align:left;color:var(--text2,#9096a0);}
.rcm-tbl tr:last-child td{border-bottom:none;}
.rcm-tbl tr:hover td{background:rgba(255,255,255,.025);}
.rcm-tbl .rcm-best td:first-child::after{content:' ★';font-size:8px;color:var(--up,#26a69a);}
/* ── Real rate coloring ── */
.rr-pos2{color:var(--up,#26a69a);font-weight:600;}
.rr-pos1{color:#5cb85c;}
.rr-neg1{color:#e07070;}
.rr-neg2{color:var(--down,#ef5350);font-weight:600;}
.rr-flat{color:var(--text2,#9096a0);}
/* ── OIS bias chip ── */
.rcm-bias{display:inline-block;font-size:8.5px;padding:1px 5px;border-radius:3px;font-weight:600;letter-spacing:.03em;white-space:nowrap;}
.rcm-bias-hike{background:rgba(38,166,154,.18);color:var(--up,#26a69a);}
.rcm-bias-cut{background:rgba(239,83,80,.18);color:var(--down,#ef5350);}
.rcm-bias-hold{background:rgba(255,255,255,.08);color:var(--text2,#9096a0);}
/* ── Real Rate Matrix ── */
#rcm-matrix-wrap{overflow:auto;flex:1;}
.rcm-matrix{border-collapse:collapse;font-size:10.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);width:100%;}
.rcm-matrix th{font-size:8.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;padding:5px 6px;color:var(--text3,#6b7280);text-align:center;white-space:nowrap;}
.rcm-matrix th.row-head{text-align:left;min-width:36px;}
.rcm-matrix td{padding:4px 6px;text-align:center;border:1px solid rgba(255,255,255,.04);font-size:10px;min-width:52px;}
.rcm-matrix td.diag{background:rgba(255,255,255,.03);color:var(--text3,#6b7280);font-size:9px;}
.rcm-matrix td.row-head{text-align:left;color:var(--text2,#9096a0);font-weight:600;font-size:10px;background:transparent;border:none;}
.rcm-matrix td.empty{background:transparent;border:none;}
/* matrix cell shading — intensity via inline style opacity on a bg span */
.rcm-cell-pos{background:rgba(38,166,154,.14);color:var(--up,#26a69a);}
.rcm-cell-pos-hi{background:rgba(38,166,154,.28);color:#2fd4c4;font-weight:700;}
.rcm-cell-neg{background:rgba(239,83,80,.14);color:var(--down,#ef5350);}
.rcm-cell-neg-hi{background:rgba(239,83,80,.28);color:#f47b79;font-weight:700;}
.rcm-cell-flat{color:var(--text3,#6b7280);}
/* ── Pair detail ── */
.rcm-pd-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.rcm-pd-kv{display:flex;flex-direction:column;gap:2px;}
.rcm-pd-lbl{font-size:8.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-pd-val{font-size:14px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-pd-sub{font-size:8.5px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-sustain{border-radius:4px;padding:8px 12px;margin-top:8px;font-size:10.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.rcm-sustain-ok{background:rgba(38,166,154,.10);border:1px solid rgba(38,166,154,.25);color:var(--up,#26a69a);}
.rcm-sustain-warn{background:rgba(246,148,28,.10);border:1px solid rgba(246,148,28,.25);color:var(--orange,#f6941c);}
.rcm-sustain-bad{background:rgba(239,83,80,.10);border:1px solid rgba(239,83,80,.25);color:var(--down,#ef5350);}
.rcm-src-note{font-size:8.5px;color:var(--text3,#6b7280);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06);}
.rcm-loading{display:flex;align-items:center;justify-content:center;flex:1;color:var(--text3,#6b7280);font-size:11px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.05em;}
@media(max-width:640px){
  #rcm-bd{padding:0;align-items:flex-end;}
  #rcm-modal{width:100%;height:94vh;border-radius:12px 12px 0 0;}
  #rcm-metrics{grid-template-columns:repeat(2,1fr);}
  .rcm-mm{padding:6px 10px;}.rcm-mm-val{font-size:12px;}
  .rcm-pd-grid{grid-template-columns:1fr;}
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
  GBP: 'CPI YoY · calendar actuals',
  JPY: 'CPI YoY · calendar actuals',
  AUD: 'CPI YoY · calendar actuals',
  CAD: 'CPI YoY · calendar actuals',
  CHF: 'CPI YoY · calendar actuals',
  NZD: 'CPI YoY · calendar actuals',
};

// ── State ───────────────────────────────────────────────────────────────────
let _rcmData = null;      // cached computed data
let _rcmFetching = false;
let _rcmActiveTab = 'breakdown';
let _rcmActivePair = null;

// ── FRED CSV fetch — no API key required ────────────────────────────────────
// FRED provides public CSVs at: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
// Returns latest observation value, or null on failure.
async function _rcmFredLatest(seriesId) {
  try {
    const url = `https://fred.stlouisfed.org/graph/fredgraph.csv?id=${seriesId}`;
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) return { val: null, date: null };
    const text = await r.text();
    const lines = text.trim().split('\n').filter(l => !l.startsWith('DATE'));
    // Find last non-empty, non-'.' value
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

// ── Data assembly ────────────────────────────────────────────────────────────
async function _rcmFetchData() {
  if (_rcmFetching) return;
  _rcmFetching = true;

  try {
    // 1. Nominal CB rates
    const nominalRates = {};
    await Promise.all(_RCM_G8.map(async ccy => {
      try {
        const r = await fetch(`./rates/${ccy}.json`);
        if (!r.ok) return;
        const d = await r.json();
        if (d.observations?.[0]?.value != null) {
          nominalRates[ccy] = { rate: parseFloat(d.observations[0].value), date: d.observations[0].date };
        }
      } catch {}
    }));

    // 2. Inflation expectations
    //    USD: FRED T5YIE (5Y breakeven — market-implied, daily)
    //    EUR: FRED T5YIFR (EUR 5Y5Y inflation swap — market-implied, daily)
    //    Rest: extended-data/*.json (weekly batch — ForexFactory calendar CPI YoY actuals)
    const inflExp = {};

    const [fredUSD, fredEUR] = await Promise.all([
      _rcmFredLatest('T5YIE'),
      _rcmFredLatest('T5YIFR'),
    ]);

    if (fredUSD.val != null) {
      inflExp['USD'] = { val: fredUSD.val, date: fredUSD.date, live: true };
    }
    if (fredEUR.val != null) {
      inflExp['EUR'] = { val: fredEUR.val, date: fredEUR.date, live: true };
    }

    // Fallback + remaining currencies from extended-data
    await Promise.all(_RCM_G8.map(async ccy => {
      if (inflExp[ccy] != null) return; // already populated from FRED
      try {
        const r = await fetch(`./extended-data/${ccy}.json`);
        if (!r.ok) return;
        const d = await r.json();
        const ie = d.data?.inflationExpectations;
        const ieDate = d.dates?.inflationExpectations;
        if (ie != null) {
          inflExp[ccy] = { val: ie, date: ieDate || null, live: false };
        }
      } catch {}
    }));

    // 3. OIS bias from meetings.json
    const biasMap = {};
    try {
      const r = await fetch('./meetings-data/meetings.json');
      if (r.ok) {
        const d = await r.json();
        for (const [ccy, m] of Object.entries(d.meetings || {})) {
          biasMap[ccy] = {
            bias: m.bias || 'hold',
            hikeProb: m.hikeProb ?? null,
            cutProb: m.cutProb ?? null,
            method: m.biasMethod || '',
          };
        }
      }
    } catch {}

    // 4. HV30 from intraday cache (for pair detail carry-to-vol)
    const hv30Map = {};
    try {
      const intra = await fetch('./intraday-data/quotes.json').then(r => r.json()).catch(() => null);
      if (intra?.hv30) Object.assign(hv30Map, intra.hv30);
    } catch {}

    // 5. Compute real rates
    const realRates = {};
    for (const ccy of _RCM_G8) {
      const nom = nominalRates[ccy]?.rate;
      const ie  = inflExp[ccy]?.val;
      realRates[ccy] = (nom != null && ie != null) ? parseFloat((nom - ie).toFixed(3)) : null;
    }

    _rcmData = { nominalRates, inflExp, biasMap, hv30Map, realRates };
  } finally {
    _rcmFetching = false;
  }
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

  const liveNote = 'USD/EUR: FRED market-implied breakeven (live, daily). GBP/JPY/AUD/CAD/CHF/NZD: CPI YoY actuals (ForexFactory calendar, weekly batch — updated Mondays). ' +
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

  // Show loading state
  const body = document.getElementById('rcm-body');
  if (body) body.innerHTML = '<div class="rcm-loading">Fetching rates & inflation data...</div>';

  // Fetch data (uses cache if available)
  if (!_rcmData) {
    await _rcmFetchData();
  }

  _rcmRender();
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

// ── Refresh cached data every 15 minutes (matches intraday-data cadence) ────
setInterval(async () => {
  const bd = document.getElementById('rcm-bd');
  if (bd && bd.style.display !== 'none') {
    // Refresh live if modal is open
    _rcmData = null;
    await _rcmFetchData();
    _rcmRender();
  } else {
    // Invalidate cache so next open gets fresh data
    _rcmData = null;
  }
}, 15 * 60 * 1000);
