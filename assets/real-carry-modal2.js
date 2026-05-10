// REAL RATE CARRY MODAL  v3.0 — tabless, maximum density
// File: assets/real-carry-modal2.js
//
// Layout (single scrollable body, no tabs required):
//   ┌──────────────────────────────────────────────────────────┐
//   │ HEADER  badge · title · sub · close                      │
//   ├──────────────────────────────────────────────────────────┤
//   │ METRICS STRIP  5 KPIs                                    │
//   ├─────────────────────────┬────────────────────────────────┤
//   │ CARRY RANKING TABLE     │ PAIR DETAIL (right col)        │
//   │ (click row = long leg)  │   – 3 big KPIs                 │
//   │ (click second = short)  │   – G8 real rate bars          │
//   │                         │   – Vol / OIS / Sustainability │
//   ├─────────────────────────┴────────────────────────────────┤
//   │ REAL RATE MATRIX  (compact, full-width)                  │
//   └──────────────────────────────────────────────────────────┘
//
// Tabs only shown on narrow screens (< 640px).
//
// Data sources:
//   Nominal rates  : ./rates/{CCY}.json
//   Infl.Exp.      : FRED CSV + ./extended-data/{CCY}.json fallback
//   OIS Bias       : ./meetings-data/meetings.json
//   HV30           : ./intraday-data/quotes.json
// ═══════════════════════════════════════════════════════════════════════════

(function () {
  if (document.getElementById('rcm-modal2-css')) return;
  const s = document.createElement('style');
  s.id = 'rcm-modal2-css';
  s.textContent = `
@keyframes rcm-pulse{0%,100%{opacity:1}50%{opacity:.4}}
#rcm-bd{display:block!important;}
#rcm-modal{width:100%!important;max-width:none!important;height:auto!important;max-height:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;position:static!important;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text);display:flex;flex-direction:column;min-height:0;}
#rcm-modal::before{display:none;}
#rcm-hd{display:flex;align-items:center;justify-content:space-between;padding:10px 14px 9px;border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg2);}
#rcm-hd-left{display:flex;flex-direction:column;gap:2px;}
.rcm-badge{display:inline-flex;align-items:center;gap:5px;font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:1px;}
.rcm-badge::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--blue);flex-shrink:0;}
#rcm-title{font-size:12px;font-weight:700;color:var(--text);letter-spacing:-.01em;}
#rcm-sub{font-size:9px;color:var(--text2);font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.02em;}
#rcm-close{background:none;border:none;color:var(--text2);font-size:16px;cursor:pointer;padding:4px 7px;border-radius:4px;line-height:1;transition:color .1s,background .1s;}
#rcm-close:hover{color:var(--text);background:var(--bg3);}
#rcm-metrics{display:grid;grid-template-columns:repeat(5,1fr);border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg);}
.rcm-mm{padding:7px 12px;border-right:1px solid var(--border2);display:flex;flex-direction:column;gap:1px;}
.rcm-mm:last-child{border-right:none;}
.rcm-mm-lbl{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.09em;}
.rcm-mm-val{font-size:16px;font-weight:700;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text);line-height:1;margin-top:2px;}
.rcm-mm-sub{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);margin-top:1px;}
#rcm-body{flex:1;min-height:0;overflow:hidden;display:flex;flex-direction:column;background:var(--bg);}
#rcm-main-row{display:flex;flex:1;min-height:0;border-bottom:1px solid var(--border2);overflow:hidden;}
#rcm-left{flex:0 0 52%;min-width:180px;border-right:1px solid var(--border2);overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;display:flex;flex-direction:column;}
#rcm-left::-webkit-scrollbar{width:3px;}
#rcm-left::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
#rcm-right{flex:1;min-width:0;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;display:flex;flex-direction:column;}
#rcm-right::-webkit-scrollbar{width:3px;}
#rcm-right::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.rcm-sec-hd{font-size:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--text2);padding:6px 12px 5px;border-bottom:1px solid var(--border2);background:var(--bg2);flex-shrink:0;display:flex;align-items:center;justify-content:space-between;}
#rcm-ranking-tbl{width:100%;border-collapse:collapse;font-size:10.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#rcm-ranking-tbl thead th{text-align:right;color:var(--text2);font-weight:600;font-size:8px;text-transform:uppercase;letter-spacing:.08em;padding:5px 10px;border-bottom:1px solid var(--border2);background:var(--bg2);white-space:nowrap;}
#rcm-ranking-tbl thead th:first-child{text-align:left;}
#rcm-ranking-tbl tbody tr{cursor:pointer;transition:background .08s;}
#rcm-ranking-tbl tbody tr:hover td{background:rgba(88,166,255,.06);}
#rcm-ranking-tbl tbody tr.rcm-row-active td{background:rgba(79,127,255,.10)!important;}
#rcm-ranking-tbl tbody tr.rcm-row-long td:first-child{border-left:2px solid var(--up,#26a69a);padding-left:8px;}
#rcm-ranking-tbl tbody tr.rcm-row-short td:first-child{border-left:2px solid var(--down,#ef5350);padding-left:8px;}
#rcm-ranking-tbl td{text-align:right;padding:5px 10px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle;}
#rcm-ranking-tbl td:first-child{text-align:left;font-weight:700;}
#rcm-ranking-tbl tr:last-child td{border-bottom:none;}
.rcm-rank-num{display:inline-block;width:16px;font-size:8.5px;color:var(--text2);font-weight:400;margin-right:3px;}
.rcm-rank-num.is-1{color:var(--up,#26a69a);font-weight:700;}
.rr-pos2{color:#26a69a;font-weight:700;}
.rr-pos1{color:#5bc8a0;}
.rr-neg1{color:#e07070;}
.rr-neg2{color:#ef5350;font-weight:700;}
.rr-flat{color:var(--text2);}
.rcm-bias{display:inline-flex;align-items:center;font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;padding:1px 5px;border-radius:2px;letter-spacing:.03em;white-space:nowrap;}
.rcm-bias-hike{background:rgba(38,166,154,.15);color:#26a69a;border:1px solid rgba(38,166,154,.25);}
.rcm-bias-cut{background:rgba(239,83,80,.15);color:#ef5350;border:1px solid rgba(239,83,80,.25);}
.rcm-bias-hold{background:rgba(139,148,158,.10);color:var(--text2);border:1px solid rgba(139,148,158,.18);}
.rcm-live-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:#26a69a;margin-left:2px;vertical-align:middle;animation:rcm-pulse 2s ease-in-out infinite;}
.rcm-pd-empty{display:flex;align-items:center;justify-content:center;flex:1;flex-direction:column;gap:8px;color:var(--text2);font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);text-align:center;padding:24px 16px;}
.rcm-pd-empty-icon{font-size:28px;opacity:.2;}
.rcm-pd-kpi-row{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid var(--border2);flex-shrink:0;}
.rcm-pd-kpi{padding:9px 12px;border-right:1px solid var(--border2);}
.rcm-pd-kpi:last-child{border-right:none;}
.rcm-pd-kpi-lbl{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--text2);margin-bottom:2px;}
.rcm-pd-kpi-val{font-size:17px;font-weight:700;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1;}
.rcm-pd-kpi-sub{font-size:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);margin-top:2px;}
.rcm-rb-section{padding:8px 12px 6px;border-bottom:1px solid var(--border2);flex-shrink:0;}
.rcm-rb-hd{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:6px;}
.rcm-rb-row{display:grid;grid-template-columns:28px 1fr 44px;align-items:center;gap:7px;margin-bottom:4px;}
.rcm-rb-row:last-child{margin-bottom:0;}
.rcm-rb-label{font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;color:var(--text2);}
.rcm-rb-track{position:relative;height:5px;background:rgba(255,255,255,.05);border-radius:2px;}
.rcm-rb-zero{position:absolute;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.18);}
.rcm-rb-fill{position:absolute;top:0;height:100%;border-radius:2px;transition:width .3s;}
.rcm-rb-val{font-size:8.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:700;text-align:right;}
.rcm-aux-row{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid var(--border2);flex-shrink:0;}
.rcm-aux-cell{padding:7px 12px;border-right:1px solid var(--border2);}
.rcm-aux-cell:last-child{border-right:none;}
.rcm-aux-lbl{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:2px;}
.rcm-aux-val{font-size:12px;font-weight:700;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text);line-height:1;}
.rcm-aux-sub{font-size:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);margin-top:1px;}
.rcm-ois-row{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--border2);flex-shrink:0;}
.rcm-ois-cell{padding:7px 12px;border-right:1px solid var(--border2);}
.rcm-ois-cell:last-child{border-right:none;}
.rcm-ois-lbl{font-size:7.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:3px;}
.rcm-sustain{padding:8px 12px;font-size:9px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);line-height:1.55;border-bottom:1px solid var(--border2);flex-shrink:0;}
.rcm-sustain-ok{background:rgba(38,166,154,.05);border-left:3px solid rgba(38,166,154,.55);}
.rcm-sustain-warn{background:rgba(210,153,34,.05);border-left:3px solid rgba(210,153,34,.55);}
.rcm-sustain-bad{background:rgba(239,83,80,.05);border-left:3px solid rgba(239,83,80,.55);}
.rcm-src-note{font-size:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);line-height:1.5;padding:6px 12px;border-top:1px solid var(--border2);background:var(--bg2);}
.rcm-loading{display:flex;align-items:center;justify-content:center;min-height:60px;flex:1;color:var(--text2);font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);letter-spacing:.06em;}
#rcm-matrix-section{flex-shrink:0;}
#rcm-matrix-inner{overflow-x:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;}
#rcm-matrix-inner::-webkit-scrollbar{height:3px;}
#rcm-matrix-inner::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.rcm-matrix{border-collapse:collapse;font-size:9.5px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);width:100%;}
.rcm-matrix th{font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:5px 8px;color:var(--text2);text-align:center;white-space:nowrap;background:var(--bg2);border-bottom:1px solid var(--border2);}
.rcm-matrix th.row-head{text-align:left;min-width:40px;}
.rcm-matrix td{padding:4px 8px;text-align:center;border:1px solid rgba(255,255,255,.04);min-width:52px;}
.rcm-matrix td.diag{background:#2d333b;color:var(--text2);font-size:9px;font-weight:600;}
.rcm-matrix td.row-head{text-align:left;color:var(--text2);font-weight:700;font-size:9px;background:var(--bg2);border:none;position:sticky;left:0;z-index:1;}
.rcm-cell-pos-hi{background:rgba(38,166,154,.25);color:#26a69a;font-weight:700;}
.rcm-cell-pos{background:rgba(38,166,154,.10);color:#26a69a;}
.rcm-cell-neg-hi{background:rgba(239,83,80,.25);color:#ef5350;font-weight:700;}
.rcm-cell-neg{background:rgba(239,83,80,.10);color:#ef5350;}
.rcm-cell-flat{color:var(--text2);}
.rcm-matrix-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:8px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:var(--text2);padding:5px 12px;border-top:1px solid var(--border2);background:var(--bg2);}
@media(max-width:640px){
  #rcm-metrics{grid-template-columns:repeat(3,1fr);}
  .rcm-mm-val{font-size:13px;}
  #rcm-main-row{flex-direction:column;overflow:visible;}
  #rcm-left{flex:none;width:100%;border-right:none;border-bottom:1px solid var(--border2);max-height:240px;}
  #rcm-right{flex:none;min-height:200px;}
  .rcm-pd-kpi-row{grid-template-columns:1fr 1fr;}
  .rcm-aux-row{grid-template-columns:1fr 1fr;}
}
`;
  document.head.appendChild(s);
})();

const _RCM_G8 = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const _RCM_CB = {USD:'Fed',EUR:'ECB',GBP:'BoE',JPY:'BoJ',AUD:'RBA',CAD:'BoC',CHF:'SNB',NZD:'RBNZ'};
const _RCM_IE_SRC = {
  USD:'FRED T5YIE · 5Y breakeven', EUR:'FRED T5YIFR · EUR 5Y5Y swap',
  GBP:'CPI YoY · IMF SDMX', JPY:'CPI YoY · IMF SDMX', AUD:'CPI YoY · IMF SDMX',
  CAD:'CPI YoY · IMF SDMX', CHF:'CPI YoY · IMF SDMX', NZD:'CPI YoY · OECD Explorer',
};

let _rcmData = null, _rcmFetchPromise = null, _rcmFetchedAt = 0;
const _RCM_TTL = 15 * 60 * 1000;
let _rcmActivePair = null;   // "LONG/SHORT"
let _rcmClickState = null;   // first-click ccy awaiting second click

async function _rcmFredLatest(seriesId) {
  try {
    const r = await fetch(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${seriesId}`);
    if (!r.ok) return { val: null, date: null };
    const lines = (await r.text()).trim().split('\n').filter(l => !l.startsWith('DATE'));
    for (let i = lines.length - 1; i >= 0; i--) {
      const [date, val] = lines[i].split(',');
      if (val && val.trim() !== '.' && val.trim() !== '') return { val: parseFloat(val), date: date.trim() };
    }
    return { val: null, date: null };
  } catch { return { val: null, date: null }; }
}

async function _rcmFetchData() {
  if (_rcmFetchPromise) return _rcmFetchPromise;
  _rcmFetchPromise = (async () => {
    try {
      const [rateResults, extResults, meetingsRes, quotesRes] = await Promise.all([
        Promise.all(_RCM_G8.map(c => fetch(`./rates/${c}.json`).then(r => r.ok ? r.json() : null).catch(() => null))),
        Promise.all(_RCM_G8.map(c => fetch(`./extended-data/${c}.json`).then(r => r.ok ? r.json() : null).catch(() => null))),
        fetch('./meetings-data/meetings.json').then(r => r.ok ? r.json() : null).catch(() => null),
        fetch('./intraday-data/quotes.json').then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      const nominalRates = {};
      _RCM_G8.forEach((c, i) => { const d = rateResults[i]; if (d?.observations?.[0]?.value != null) nominalRates[c] = { rate: parseFloat(d.observations[0].value), date: d.observations[0].date }; });
      const inflExp = {};
      _RCM_G8.forEach((c, i) => { const d = extResults[i]; const ie = d?.data?.inflationExpectations; if (ie != null) inflExp[c] = { val: ie, date: d?.dates?.inflationExpectations || null, live: false }; });
      const biasMap = {};
      if (meetingsRes?.meetings) for (const [c, m] of Object.entries(meetingsRes.meetings)) biasMap[c] = { bias: m.bias || 'hold', hikeProb: m.hikeProb ?? null, cutProb: m.cutProb ?? null, method: m.biasMethod || '' };
      const hv30Map = {};
      if (quotesRes?.hv30) Object.assign(hv30Map, quotesRes.hv30);
      const realRates = {};
      for (const c of _RCM_G8) { const nom = nominalRates[c]?.rate, ie = inflExp[c]?.val; realRates[c] = (nom != null && ie != null) ? parseFloat((nom - ie).toFixed(3)) : null; }
      _rcmData = { nominalRates, inflExp, biasMap, hv30Map, realRates };
      _rcmFetchedAt = Date.now();
      Promise.all([_rcmFredLatest('T5YIE'), _rcmFredLatest('T5YIFR')]).then(([fredUSD, fredEUR]) => {
        if (!_rcmData) return;
        let up = false;
        if (fredUSD.val != null) { _rcmData.inflExp['USD'] = { val: fredUSD.val, date: fredUSD.date, live: true }; up = true; }
        if (fredEUR.val != null) { _rcmData.inflExp['EUR'] = { val: fredEUR.val, date: fredEUR.date, live: true }; up = true; }
        if (!up) return;
        for (const c of ['USD', 'EUR']) { const nom = _rcmData.nominalRates[c]?.rate, ie = _rcmData.inflExp[c]?.val; _rcmData.realRates[c] = (nom != null && ie != null) ? parseFloat((nom - ie).toFixed(3)) : null; }
        const bd = document.getElementById('rcm-bd');
        if (bd && bd.style.display !== 'none') _rcmRenderAll();
      }).catch(() => {});
    } finally { _rcmFetchPromise = null; }
  })();
  return _rcmFetchPromise;
}

function _rcmRrClass(rr) {
  if (rr == null) return 'rr-flat';
  if (rr >= 1.0)  return 'rr-pos2';
  if (rr >= 0.0)  return 'rr-pos1';
  if (rr >= -1.0) return 'rr-neg1';
  return 'rr-neg2';
}
function _rcmRrFmt(rr) { return rr == null ? '—' : (rr >= 0 ? '+' : '') + rr.toFixed(2) + '%'; }
function _rcmBiasChip(b) {
  if (!b) return '<span class="rcm-bias rcm-bias-hold">Hold</span>';
  const pct = b.bias === 'hike' && b.hikeProb > 0 ? ` ${b.hikeProb}%` : b.bias === 'cut' && b.cutProb > 0 ? ` ${b.cutProb}%` : '';
  const lbl = b.bias === 'hike' ? 'Hike' : b.bias === 'cut' ? 'Cut' : 'Hold';
  const cls = b.bias === 'hike' ? 'rcm-bias-hike' : b.bias === 'cut' ? 'rcm-bias-cut' : 'rcm-bias-hold';
  return `<span class="rcm-bias ${cls}">${lbl}${pct}</span>`;
}
function _rcmPairId(a, b) {
  const S = new Set(['eurusd','gbpusd','usdjpy','audusd','usdchf','usdcad','nzdusd','eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy','eurnzd','gbpaud','gbpnzd','audcad','cadchf','nzdcad','nzdchf']);
  const c1 = (a+b).toLowerCase(), c2 = (b+a).toLowerCase();
  return S.has(c1) ? c1 : S.has(c2) ? c2 : a < b ? c1 : c2;
}

// ── Render: Ranking (left) ────────────────────────────────────────────────────
function _rcmRenderRanking() {
  const d = _rcmData;
  if (!d) return '<div class="rcm-loading">Loading...</div>';
  const [longLeg, shortLeg] = (_rcmActivePair || '/').split('/');
  const sorted = [..._RCM_G8].sort((a, b) => (d.realRates[b] ?? -99) - (d.realRates[a] ?? -99));
  const rows = sorted.map((ccy, idx) => {
    const nom = d.nominalRates[ccy]?.rate, ie = d.inflExp[ccy], rr = d.realRates[ccy], bias = d.biasMap[ccy];
    const isLong = ccy === longLeg, isShort = ccy === shortLeg;
    const rowCls = [isLong ? 'rcm-row-active rcm-row-long' : '', isShort ? 'rcm-row-active rcm-row-short' : '', _rcmClickState === ccy && !isLong ? 'rcm-row-active' : ''].filter(Boolean).join(' ');
    const liveDot = ie?.live ? '<span class="rcm-live-dot" title="FRED live"></span>' : '';
    const rankCls = idx === 0 ? ' is-1' : '';
    return `<tr class="${rowCls}" data-ccy="${ccy}">
      <td><span class="rcm-rank-num${rankCls}">${idx+1}</span><strong>${_RCM_CB[ccy]}</strong> <span style="color:var(--text2);font-weight:400;">${ccy}</span></td>
      <td>${nom != null ? nom.toFixed(2)+'%' : '—'}</td>
      <td>${ie ? ie.val.toFixed(2)+'%' : '—'}${liveDot}</td>
      <td class="${_rcmRrClass(rr)}" style="font-weight:600;">${_rcmRrFmt(rr)}</td>
      <td>${_rcmBiasChip(bias)}</td>
    </tr>`;
  }).join('');
  return `
  <div class="rcm-sec-hd"><span>Carry Ranking · G8</span><span style="font-weight:400;letter-spacing:.02em;font-size:7.5px;">click row = long · click 2nd = short</span></div>
  <table id="rcm-ranking-tbl" aria-label="Real rate carry ranking">
    <thead><tr>
      <th style="text-align:left;">Central Bank</th><th>Nominal</th><th>Infl.Exp.</th><th>Real Rate</th><th>OIS Bias</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <div class="rcm-src-note">Real rate = Nominal CB rate − Inflation Expectation. USD/EUR: FRED 5Y breakeven (live). Others: CPI YoY (IMF/OECD, weekly).</div>`;
}

// ── Render: Pair detail (right) ───────────────────────────────────────────────
function _rcmRenderPairDetail(longCcy, shortCcy) {
  const d = _rcmData;
  if (!d || !longCcy || !shortCcy) {
    const msg = _rcmClickState
      ? `<strong>${_rcmClickState}</strong> set as long — now click a short leg`
      : 'Click a row to set long leg, then a second row for short leg';
    return `<div class="rcm-pd-empty"><div class="rcm-pd-empty-icon">↔</div><div style="line-height:1.6;">${msg}</div></div>`;
  }

  const nomL = d.nominalRates[longCcy]?.rate, nomS = d.nominalRates[shortCcy]?.rate;
  const ieL  = d.inflExp[longCcy]?.val, ieS = d.inflExp[shortCcy]?.val;
  const rrL  = d.realRates[longCcy], rrS = d.realRates[shortCcy];
  const biasL = d.biasMap[longCcy], biasS = d.biasMap[shortCcy];
  const nomSpread  = nomL != null && nomS != null ? nomL - nomS : null;
  const realSpread = rrL  != null && rrS  != null ? rrL  - rrS  : null;
  const pid  = _rcmPairId(longCcy, shortCcy);
  const hv30 = d.hv30Map[pid] ?? null;
  const nomCarryVol  = hv30 && nomSpread  != null ? (Math.abs(nomSpread)  / hv30).toFixed(2) : null;
  const realCarryVol = hv30 && realSpread != null ? (Math.abs(realSpread) / hv30).toFixed(2) : null;

  let sCls = 'rcm-sustain-warn', sTxt = '';
  if (rrL != null && rrS != null) {
    if (rrL > 0 && rrS < 0) { sCls = 'rcm-sustain-ok'; sTxt = `<strong>Sustainable</strong> — long ${longCcy} real positive (${_rcmRrFmt(rrL)}), short ${shortCcy} negative (${_rcmRrFmt(rrS)}). Carry unlikely to erode via inflation differential.`; }
    else if (rrL < 0) { sCls = 'rcm-sustain-bad'; sTxt = `<strong>Carry trap risk</strong> — long leg real rate negative (${_rcmRrFmt(rrL)}). Nominal carry may be eroded by inflation.`; }
    else if (rrL > 0 && rrS > 0) {
      if (rrL > rrS) { sCls = 'rcm-sustain-warn'; sTxt = `<strong>Moderate</strong> — both legs positive real. Real spread ${_rcmRrFmt(realSpread)}. Watch for inflation convergence.`; }
      else { sCls = 'rcm-sustain-bad'; sTxt = `<strong>Negative real spread</strong> — short leg real rate higher. Nominal carry favors long ${longCcy} but real carry favors long ${shortCcy}.`; }
    } else { sTxt = 'Insufficient real rate data for sustainability assessment.'; }
  }

  function fmt(v, sfx) { return v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + (sfx || '') : '—'; }
  const G8s = [..._RCM_G8].filter(c => d.realRates[c] != null).sort((a, b) => d.realRates[b] - d.realRates[a]);
  const maxA = Math.max(...G8s.map(c => Math.abs(d.realRates[c] ?? 0)), 0.01);
  const bars = G8s.map(ccy => {
    const rr = d.realRates[ccy], isL = ccy === longCcy, isS = ccy === shortCcy;
    const pct = Math.abs(rr) / maxA * 44;
    const bc  = rr >= 0 ? 'var(--up,#26a69a)' : 'var(--down,#ef5350)';
    const ls  = isL ? 'color:var(--up,#26a69a);font-weight:700;' : isS ? 'color:var(--down,#ef5350);font-weight:700;' : '';
    const fill = rr >= 0 ? `left:50%;width:${pct}%;background:${bc};` : `right:50%;width:${pct}%;background:${bc};`;
    const sfx  = isL ? ' ▲' : isS ? ' ▼' : '';
    return `<div class="rcm-rb-row">
      <div class="rcm-rb-label" style="${ls}">${ccy}${sfx}</div>
      <div class="rcm-rb-track"><div class="rcm-rb-zero" style="left:50%"></div><div class="rcm-rb-fill" style="${fill}"></div></div>
      <div class="rcm-rb-val ${_rcmRrClass(rr)}">${_rcmRrFmt(rr)}</div>
    </div>`;
  }).join('');

  return `
  <div class="rcm-sec-hd">
    <span>Pair Detail</span>
    <span style="font-size:11px;font-weight:700;color:var(--text);letter-spacing:-.01em;">${longCcy} <span style="color:var(--up,#26a69a);">▲</span> / ${shortCcy} <span style="color:var(--down,#ef5350);">▼</span></span>
  </div>
  <div class="rcm-pd-kpi-row">
    <div class="rcm-pd-kpi">
      <div class="rcm-pd-kpi-lbl">Nominal carry</div>
      <div class="rcm-pd-kpi-val ${_rcmRrClass(nomSpread)}">${fmt(nomSpread,'%')}</div>
      <div class="rcm-pd-kpi-sub">${nomL != null ? nomL.toFixed(2)+'%' : '—'} − ${nomS != null ? nomS.toFixed(2)+'%' : '—'}</div>
    </div>
    <div class="rcm-pd-kpi">
      <div class="rcm-pd-kpi-lbl">Real carry</div>
      <div class="rcm-pd-kpi-val ${_rcmRrClass(realSpread)}">${fmt(realSpread,'%')}</div>
      <div class="rcm-pd-kpi-sub">After infl. expectations</div>
    </div>
    <div class="rcm-pd-kpi">
      <div class="rcm-pd-kpi-lbl">Long real (${longCcy})</div>
      <div class="rcm-pd-kpi-val ${_rcmRrClass(rrL)}">${_rcmRrFmt(rrL)}</div>
      <div class="rcm-pd-kpi-sub">${nomL != null ? nomL.toFixed(2)+'%' : '—'} nom − ${ieL != null ? ieL.toFixed(2)+'%' : '—'} IE</div>
    </div>
  </div>
  <div class="rcm-rb-section">
    <div class="rcm-rb-hd">Real rate positioning G8 — ▲ long / ▼ short highlighted</div>
    ${bars}
  </div>
  <div class="rcm-aux-row">
    <div class="rcm-aux-cell">
      <div class="rcm-aux-lbl">Short real (${shortCcy})</div>
      <div class="rcm-aux-val ${_rcmRrClass(rrS)}">${_rcmRrFmt(rrS)}</div>
      <div class="rcm-aux-sub">${nomS != null ? nomS.toFixed(2)+'%' : '—'} nom − ${ieS != null ? ieS.toFixed(2)+'%' : '—'} IE</div>
    </div>
    <div class="rcm-aux-cell">
      <div class="rcm-aux-lbl">Nom carry/vol</div>
      <div class="rcm-aux-val">${nomCarryVol ?? '—'}</div>
      <div class="rcm-aux-sub">HV30 ${hv30 ? hv30.toFixed(1)+'%' : 'n/a'}</div>
    </div>
    <div class="rcm-aux-cell">
      <div class="rcm-aux-lbl">Real carry/vol</div>
      <div class="rcm-aux-val">${realCarryVol ?? '—'}</div>
      <div class="rcm-aux-sub">Real spread / HV30</div>
    </div>
  </div>
  <div class="rcm-ois-row">
    <div class="rcm-ois-cell">
      <div class="rcm-ois-lbl">OIS bias (${longCcy})</div>
      ${_rcmBiasChip(biasL)}
      <div style="font-size:8px;font-family:var(--font-mono);color:var(--text2);margin-top:2px;">${biasL?.method || '—'}</div>
    </div>
    <div class="rcm-ois-cell">
      <div class="rcm-ois-lbl">OIS bias (${shortCcy})</div>
      ${_rcmBiasChip(biasS)}
      <div style="font-size:8px;font-family:var(--font-mono);color:var(--text2);margin-top:2px;">${biasS?.method || '—'}</div>
    </div>
  </div>
  ${sTxt ? `<div class="rcm-sustain ${sCls}">${sTxt}</div>` : ''}
  <div class="rcm-src-note">
    IE source — ${longCcy}: ${_RCM_IE_SRC[longCcy] || '—'} · ${shortCcy}: ${_RCM_IE_SRC[shortCcy] || '—'}<br>
    Carry/vol = |spread| / HV30 (30d realised vol, annualised)
  </div>`;
}

// ── Render: Matrix (full-width bottom) ───────────────────────────────────────
function _rcmRenderMatrix() {
  const d = _rcmData;
  if (!d) return '<div class="rcm-loading">Loading matrix...</div>';
  function cc(diff) {
    if (diff == null) return 'rcm-cell-flat';
    if (diff >= 1.5)  return 'rcm-cell-pos-hi';
    if (diff >= 0.15) return 'rcm-cell-pos';
    if (diff <= -1.5) return 'rcm-cell-neg-hi';
    if (diff <= -0.15)return 'rcm-cell-neg';
    return 'rcm-cell-flat';
  }
  function cf(v) { return v == null ? '—' : Math.abs(v) < 0.01 ? '0' : (v > 0 ? '+' : '') + v.toFixed(2) + '%'; }
  const G8 = _RCM_G8;
  const hdr = `<tr><th class="row-head">L↓/S→</th>${G8.map(c => `<th>${c}</th>`).join('')}</tr>`;
  const rows = G8.map(r => {
    const cells = G8.map(c => {
      if (r === c) { const rr = d.realRates[r]; const f = rr != null ? (rr>=0?'+':'')+rr.toFixed(2)+'%':'—'; return `<td class="diag" title="${r} real: ${f}">${f}</td>`; }
      const diff = d.realRates[r] != null && d.realRates[c] != null ? parseFloat((d.realRates[r] - d.realRates[c]).toFixed(3)) : null;
      return `<td class="${cc(diff)}" title="Long ${r} / Short ${c}: ${cf(diff)}">${cf(diff)}</td>`;
    }).join('');
    return `<tr><td class="row-head">${r}</td>${cells}</tr>`;
  }).join('');
  return `
  <div class="rcm-sec-hd"><span>Real Rate Differential Matrix · Long row / Short column · Cell = Row − Column · Diagonal = absolute real rate</span></div>
  <div id="rcm-matrix-inner"><table class="rcm-matrix" aria-label="Real rate differential matrix"><thead>${hdr}</thead><tbody>${rows}</tbody></table></div>
  <div class="rcm-matrix-legend">
    <span><span style="display:inline-block;width:9px;height:9px;background:rgba(38,166,154,.28);border-radius:1px;vertical-align:middle;margin-right:3px;"></span>Strong carry ≥1.5%</span>
    <span><span style="display:inline-block;width:9px;height:9px;background:rgba(38,166,154,.14);border-radius:1px;vertical-align:middle;margin-right:3px;"></span>+ve spread</span>
    <span><span style="display:inline-block;width:9px;height:9px;background:rgba(239,83,80,.14);border-radius:1px;vertical-align:middle;margin-right:3px;"></span>−ve spread</span>
    <span><span style="display:inline-block;width:9px;height:9px;background:rgba(239,83,80,.28);border-radius:1px;vertical-align:middle;margin-right:3px;"></span>Strong drag ≤−1.5%</span>
    <span style="margin-left:auto;">Diagonal = absolute real rate</span>
  </div>`;
}

// ── Full render ───────────────────────────────────────────────────────────────
function _rcmRenderAll() {
  _rcmUpdateMetrics();

  const leftEl = document.getElementById('rcm-left');
  if (leftEl) {
    leftEl.innerHTML = _rcmData ? _rcmRenderRanking() : '<div class="rcm-loading">Loading...</div>';
    leftEl.querySelectorAll('#rcm-ranking-tbl tbody tr').forEach(tr => {
      tr.addEventListener('click', () => _rcmHandleRowClick(tr.dataset.ccy));
    });
  }

  const rightEl = document.getElementById('rcm-right');
  if (rightEl) {
    const [l, s] = (_rcmActivePair || '/').split('/');
    rightEl.innerHTML = _rcmRenderPairDetail(l || null, s || null);
  }

  const matEl = document.getElementById('rcm-matrix-section');
  if (matEl) matEl.innerHTML = _rcmData ? _rcmRenderMatrix() : '<div class="rcm-loading">Loading...</div>';
}

function _rcmHandleRowClick(ccy) {
  if (!_rcmClickState) {
    // First click — set long leg
    _rcmClickState = ccy;
    _rcmActivePair = null;
    // Update right panel immediately with helpful prompt
    const rightEl = document.getElementById('rcm-right');
    if (rightEl) rightEl.innerHTML = _rcmRenderPairDetail(null, null);
    // Refresh ranking to show active highlight
    const leftEl = document.getElementById('rcm-left');
    if (leftEl) {
      leftEl.innerHTML = _rcmRenderRanking();
      leftEl.querySelectorAll('#rcm-ranking-tbl tbody tr').forEach(tr => {
        tr.addEventListener('click', () => _rcmHandleRowClick(tr.dataset.ccy));
      });
    }
  } else if (ccy === _rcmClickState) {
    // Clicked same row — deselect
    _rcmClickState = null;
    _rcmActivePair = null;
    _rcmRenderAll();
  } else {
    // Second click — set short leg and open detail
    _rcmActivePair = `${_rcmClickState}/${ccy}`;
    _rcmClickState = null;
    _rcmRenderAll();
  }
}

function _rcmUpdateMetrics() {
  const d = _rcmData;
  if (!d) return;
  const sorted = [..._RCM_G8].filter(c => d.realRates[c] != null).sort((a, b) => d.realRates[b] - d.realRates[a]);
  const best = sorted[0], worst = sorted[sorted.length - 1];
  const eB = document.getElementById('rcm-mm-best'), eW = document.getElementById('rcm-mm-worst');
  const eS = document.getElementById('rcm-mm-spread'), eSrc = document.getElementById('rcm-mm-src'), eP = document.getElementById('rcm-mm-positive');
  if (eB && best) { eB.querySelector('.rcm-mm-val').textContent = _rcmRrFmt(d.realRates[best]); eB.querySelector('.rcm-mm-lbl').textContent = `Best real (${best})`; eB.querySelector('.rcm-mm-sub').textContent = _RCM_CB[best]; }
  if (eW && worst) { eW.querySelector('.rcm-mm-val').textContent = _rcmRrFmt(d.realRates[worst]); eW.querySelector('.rcm-mm-lbl').textContent = `Worst real (${worst})`; eW.querySelector('.rcm-mm-sub').textContent = _RCM_CB[worst]; }
  if (eS && best && worst) { eS.querySelector('.rcm-mm-val').textContent = '+' + (d.realRates[best] - d.realRates[worst]).toFixed(2) + '%'; eS.querySelector('.rcm-mm-sub').textContent = `${best}−${worst}`; }
  if (eSrc) { const lv = d.inflExp['USD']?.live; eSrc.querySelector('.rcm-mm-val').textContent = lv ? 'Live' : 'Batch'; eSrc.querySelector('.rcm-mm-sub').textContent = lv ? 'USD/EUR FRED' : 'extended-data'; }
  if (eP) { const pos = _RCM_G8.filter(c => (d.realRates[c] ?? -99) > 0).length; eP.querySelector('.rcm-mm-val').textContent = `${pos} / ${_RCM_G8.length}`; eP.querySelector('.rcm-mm-sub').textContent = 'positive real rate'; }
}

// ── Public API ────────────────────────────────────────────────────────────────
async function openRealCarryModal(longCcy, shortCcy) {
  if (!document.getElementById('rcm-bd')) _rcmBuildDOM();

  if (longCcy && shortCcy) { _rcmActivePair = `${longCcy}/${shortCcy}`; _rcmClickState = null; }

  const bd = document.getElementById('rcm-bd');
  bd.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  if (_rcmData) {
    _rcmRenderAll();
  } else {
    const leftEl = document.getElementById('rcm-left');
    if (leftEl) leftEl.innerHTML = '<div class="rcm-loading">Fetching rates &amp; inflation data...</div>';
    await _rcmFetchData();
    _rcmRenderAll();
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
        <div id="rcm-sub">Nominal CB rate &minus; Inflation Expectation (breakeven / CPI proxy) &middot; G8 &middot; click ranking row to build pair</div>
      </div>
      <button id="rcm-close" aria-label="Close">&times;</button>
    </div>
    <div id="rcm-metrics">
      <div class="rcm-mm" id="rcm-mm-best"><div class="rcm-mm-lbl">Best real (—)</div><div class="rcm-mm-val rr-pos1">—</div><div class="rcm-mm-sub">—</div></div>
      <div class="rcm-mm" id="rcm-mm-worst"><div class="rcm-mm-lbl">Worst real (—)</div><div class="rcm-mm-val rr-neg1">—</div><div class="rcm-mm-sub">—</div></div>
      <div class="rcm-mm" id="rcm-mm-spread"><div class="rcm-mm-lbl">Max real spread</div><div class="rcm-mm-val">—</div><div class="rcm-mm-sub">—</div></div>
      <div class="rcm-mm" id="rcm-mm-positive"><div class="rcm-mm-lbl">Positive real</div><div class="rcm-mm-val" style="color:var(--text2);">—</div><div class="rcm-mm-sub">—</div></div>
      <div class="rcm-mm" id="rcm-mm-src"><div class="rcm-mm-lbl">Data source</div><div class="rcm-mm-val" style="font-size:11px;margin-top:3px;">—</div><div class="rcm-mm-sub">—</div></div>
    </div>
    <div id="rcm-body">
      <div id="rcm-main-row">
        <div id="rcm-left"><div class="rcm-loading">Loading...</div></div>
        <div id="rcm-right"><div class="rcm-pd-empty"><div class="rcm-pd-empty-icon">↔</div><div>Click a row to set long leg</div><div style="font-size:8.5px;margin-top:3px;color:var(--text2);">Click second row to set short leg</div></div></div>
      </div>
      <div id="rcm-matrix-section"><div class="rcm-loading">Loading matrix...</div></div>
    </div>
  </div>`;
  document.body.appendChild(bd);
  document.getElementById('rcm-close').addEventListener('click', _rcmClose);
  bd.addEventListener('click', e => { if (e.target === bd) _rcmClose(); });
  document.addEventListener('keydown', _rcmEsc, { capture: true });
}

function _rcmClose() { const bd = document.getElementById('rcm-bd'); if (bd) bd.style.display = 'none'; document.body.style.overflow = ''; _rcmClickState = null; }
function _rcmEsc(e) { if (e.key === 'Escape') { const bd = document.getElementById('rcm-bd'); if (bd && bd.style.display !== 'none') _rcmClose(); } }

setInterval(async () => {
  if ((Date.now() - _rcmFetchedAt) < _RCM_TTL) return;
  const bd = document.getElementById('rcm-bd'), isOpen = bd && bd.style.display !== 'none';
  await _rcmFetchData();
  if (isOpen) _rcmRenderAll();
}, 60 * 1000);

window.openRealCarryModal = openRealCarryModal;
