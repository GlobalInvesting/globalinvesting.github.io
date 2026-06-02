/**
 * dashboard2.js — Real-data layer for index2.html (Bloomberg FXIP mock)
 *
 * Strategy: the mock's inline JS owns all rendering (buildRates, buildFwd, etc.)
 * This file fetches live JSON data, patches the in-memory data arrays,
 * then re-calls the existing render functions. No inline JS is modified.
 *
 * Data sources:
 *   intraday-data/quotes.json   — FX spots, % changes, cross-asset
 *   extended-data/[CCY].json    — 10Y yields, CB rates
 *   meetings-data/meetings.json — Next meeting dates, OIS-implied bias
 *   ohlc-data/dxy.json          — DXY OHLC for chart
 *   ohlc-data/[pair].json       — Pair OHLC for country view chart
 *
 * No Math.random(). All values deterministic. Per GUIDELINES v8.2.45.
 */

'use strict';

/* ─── rateRows index by mock currency code ──────────────────────────── */
const RATE_IDX = { USD:0, EUR:1, JPY:2, GBP:3, CAD:4, AUD:5, NZD:6, CHF:7, NOK:8, SEK:9 };

/* ─── Currency → extended-data file ─────────────────────────────────── */
const EXT_CCY = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];

/* ─── Map sidebar key → ohlc-data filename ───────────────────────────── */
const OHLC_FILE = {
  eu:'eurusd', jp:'usdjpy', uk:'gbpusd', ca:'usdcad',
  au:'audusd', nz:'nzdusd', ch:'usdchf',
};

/* Cache for ohlc fetches */
const ohlcCache = {};

/* ================================================================
   HELPERS
   ================================================================ */

function bpStr(spread) {
  if (spread == null || isNaN(spread)) return '--';
  const bp = Math.round(spread * 100);
  return (bp > 0 ? '+' : '') + bp + 'bp';
}

/* ================================================================
   DXY CHART — rebuild SVG from real closes
   ================================================================ */

function redrawDXYChart(data) {
  const slice = data.slice(-90);
  if (slice.length < 2) return;

  const vbW = 600, vbH = 65, padT = 5, padB = 12;
  const chartH = vbH - padT - padB;
  const vals = slice.map(d => d.close);
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 1;
  const last = vals[vals.length - 1];
  const first = vals[0];
  const isUp = last >= first;
  const color = isUp ? '#00cc00' : '#ff3333';

  const pts = slice.map((d, i) => {
    const x = +((i / (slice.length - 1)) * vbW).toFixed(1);
    const y = +(padT + chartH - ((d.close - mn) / range) * chartH).toFixed(1);
    return [x, y];
  });

  const linePts = pts.map(([x,y]) => `${x},${y}`).join(' L');
  const areaD   = `M${linePts} L${vbW},${vbH-padB} L0,${vbH-padB}Z`;
  const lineD   = `M${linePts}`;

  function fmtDate(iso) {
    const d = new Date(iso);
    return (d.getUTCMonth()+1) + '/' + d.getUTCDate();
  }
  const idxs = [0, Math.floor(slice.length*0.25), Math.floor(slice.length*0.5),
                Math.floor(slice.length*0.75), slice.length-1];

  // Price grid labels (4 levels)
  let gridLines = '';
  [0.2, 0.45, 0.7, 0.95].forEach(pct => {
    const v = mn + range * pct;
    const y = +(padT + chartH - pct * chartH).toFixed(1);
    gridLines += `<text x="5" y="${y}" fill="#3a6050" font-size="7" font-family="Consolas">${v.toFixed(2)}</text>`;
  });

  const svgInner = `
<defs>
  <linearGradient id="dg_dxy" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="${color}" stop-opacity=".18"/>
    <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
  </linearGradient>
</defs>
${gridLines}
<path d="${areaD}" fill="url(#dg_dxy)"/>
<path d="${lineD}" fill="none" stroke="${color}" stroke-width="1"/>
${idxs.map(i => `<text x="${pts[i][0]}" y="${vbH}" fill="#445870" font-size="6" font-family="Consolas">${fmtDate(slice[i].time)}</text>`).join('')}`;

  const chcSvg = document.querySelector('#v-all .chc svg');
  if (chcSvg) {
    chcSvg.innerHTML = svgInner;
  }

  // Update legend
  const leg  = document.querySelector('#v-all .chc .cleg span');
  const legSq = document.querySelector('#v-all .chc .csq');
  if (leg)  { leg.textContent = `DXY Index ${last.toFixed(3)}`; leg.style.color = color; }
  if (legSq){ legSq.style.background = color; }
}

/* ================================================================
   COUNTRY CHART — rebuild SVG from real pair OHLC
   ================================================================ */

async function fetchOhlc(pairFile) {
  if (ohlcCache[pairFile] !== undefined) return ohlcCache[pairFile];
  try {
    const res = await fetch(`ohlc-data/${pairFile}.json`);
    ohlcCache[pairFile] = await res.json();
  } catch { ohlcCache[pairFile] = null; }
  return ohlcCache[pairFile];
}

function redrawCtyChart(key, data) {
  if (!data || data.length < 2) return;
  const slice = data.slice(-90);
  const vals  = slice.map(d => d.close);
  const mn    = Math.min(...vals);
  const mx    = Math.max(...vals);
  const range = mx - mn || 1;
  const last  = vals[vals.length-1];
  const isUp  = last >= vals[0];
  const color = isUp ? '#00cc00' : '#ff3333';
  const vbW = 400, vbH = 55, padT = 3, padB = 7;
  const chartH = vbH - padT - padB;

  const pts = slice.map((d, i) => {
    const x = +((i/(slice.length-1))*vbW).toFixed(1);
    const y = +(padT + chartH - ((d.close-mn)/range)*chartH).toFixed(1);
    return [x, y];
  });
  const linePts = pts.map(([x,y]) => `${x},${y}`).join(' L');
  const areaD   = `M${linePts} L${vbW},${vbH-padB} L0,${vbH-padB}Z`;
  const lineD   = `M${linePts}`;

  function fmtD(iso) { const d = new Date(iso); return (d.getUTCMonth()+1)+'/'+d.getUTCDate(); }
  const li = [0, Math.floor(slice.length*0.33), Math.floor(slice.length*0.66), slice.length-1];

  const svgInner = `
<defs>
  <linearGradient id="sg_cty" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="${color}" stop-opacity=".2"/>
    <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
  </linearGradient>
</defs>
<path id="ct-cf" d="${areaD}" fill="url(#sg_cty)"/>
<path id="ct-cl" d="${lineD}" fill="none" stroke="${color}" stroke-width="1.2"/>
${li.map(i => `<text x="${pts[i][0]}" y="${vbH}" fill="#445870" font-size="5.5" font-family="Consolas">${fmtD(slice[i].time)}</text>`).join('')}`;

  const chcSvg = document.querySelector('#v-cty .chc svg');
  if (chcSvg) chcSvg.innerHTML = svgInner;

  // Update legend color
  const csq  = document.getElementById('ct-csq');
  const clbl = document.getElementById('ct-clbl');
  if (csq)  csq.style.background = color;
  if (clbl) { clbl.style.color = color; }
}

/* ================================================================
   PATCH RATE ROWS
   ================================================================ */

function patchRateRows(quotes, extData) {
  const usd10y = extData.USD?.bond10y ?? 4.475;

  const cfg = {
    EUR:{ q:'eurusd', eqSym:'stoxx' },
    JPY:{ q:'usdjpy', eqSym:'nikkei' },
    GBP:{ q:'gbpusd', eqSym:'ftse'  },
    CAD:{ q:'usdcad', eqSym:null    },
    AUD:{ q:'audusd', eqSym:'asx'   },
    NZD:{ q:'nzdusd', eqSym:null    },
    CHF:{ q:'usdchf', eqSym:null    },
  };

  for (const [ccy, c] of Object.entries(cfg)) {
    const idx = RATE_IDX[ccy];
    if (idx == null) continue;
    const q = quotes[c.q];
    if (!q) continue;

    const row = rateRows[idx];
    const close = q.close;
    const prev  = q.prev_close ?? (close - (q.chg ?? 0));

    // Spot — keep decimal precision matching pair
    if (ccy === 'JPY') {
      row.sp = close.toFixed(2);
    } else if (ccy === 'CAD' || ccy === 'CHF') {
      row.sp = close.toFixed(4);
    } else {
      row.sp = close.toFixed(4);
    }

    // % change → convert to absolute pip change for the mock's `pc` field
    // mock uses raw number that it formats to 2dp; use pct directly for coloring
    row.pc = parseFloat((q.pct ?? 0).toFixed(2));

    // 10Y yield from extended-data
    const y10  = extData[ccy]?.bond10y;
    if (y10 != null) {
      row.yld = parseFloat(y10).toFixed(3);
      row.ysp = bpStr(y10 - usd10y);
    }

    // Equity from quotes
    if (c.eqSym && quotes[c.eqSym]) {
      const eq = quotes[c.eqSym];
      row.idx = eq.close.toLocaleString('en-US', { maximumFractionDigits: 0 });
      row.ic  = parseFloat((eq.pct ?? 0).toFixed(2));
    }
  }

  // USD row: SPX equity
  const spx = quotes.spx;
  if (spx) {
    rateRows[0].idx = spx.close.toLocaleString('en-US', { maximumFractionDigits: 0 });
    rateRows[0].ic  = parseFloat((spx.pct ?? 0).toFixed(2));
  }
}

/* ================================================================
   PATCH MEETINGS / KEY DATA
   ================================================================ */

function patchMeetings(meetingsJson) {
  const m = meetingsJson?.meetings ?? {};
  const map = {
    us:'USD', eu:'EUR', jp:'JPY', uk:'GBP',
    ca:'CAD', au:'AUD', nz:'NZD', ch:'CHF',
  };
  for (const [sideKey, cur] of Object.entries(map)) {
    const mtg = m[cur];
    if (!mtg?.nextMeeting) continue;

    // Update kdMap "Next Decision" entry
    (kdMap[sideKey] ?? []).forEach(item => {
      if (item.k === 'Next Decision' || item.k === 'Next MAS') {
        item.v = mtg.nextMeeting;
      }
    });
    // Update ecoCB
    if (ecoCB[sideKey]) ecoCB[sideKey].next = mtg.nextMeeting;
  }
}

/* ================================================================
   PATCH ECO ROWS
   ================================================================ */

function patchEcoRows(extData) {
  const cyMap = {
    'U.S.':'USD','Euro':'EUR','Japan':'JPY','U.K.':'GBP',
    'Canada':'CAD','Australia':'AUD','N.Z.':'NZD','Switz.':'CHF',
  };
  for (const row of ecoRows) {
    const ccy = cyMap[row.cy];
    if (!ccy) continue;
    const ext = extData[ccy];
    if (!ext) continue;
    if (ext.bond10y != null) row.y = parseFloat(parseFloat(ext.bond10y).toFixed(3));
  }
}

/* ================================================================
   PATCH ctyMeta spot/change for selected currency view
   ================================================================ */

function patchCtyMeta(quotes) {
  const qMap = {
    eu:  [['EUR/USD',  'eurusd',  false]],
    jp:  [['USD/JPY',  'usdjpy',  false]],
    uk:  [['GBP/USD',  'gbpusd',  false]],
    ca:  [['USD/CAD',  'usdcad',  false]],
    au:  [['AUD/USD',  'audusd',  false]],
    nz:  [['NZD/USD',  'nzdusd',  false]],
    ch:  [['USD/CHF',  'usdchf',  false]],
  };
  for (const [key, pairs] of Object.entries(qMap)) {
    const meta = ctyMeta[key];
    if (!meta) continue;
    const [, qKey] = pairs[0];
    const q = quotes[qKey];
    if (!q) continue;
    // Update primary spot and change
    meta.sp = q.close.toFixed(4);
    meta.vs[0] = q.close;
    meta.cs[0] = parseFloat((q.chg ?? 0).toFixed(4));
  }

  // Also patch us: DXY
  const dxy = quotes.dxy;
  if (dxy && ctyMeta.us) {
    ctyMeta.us.sp = dxy.close.toFixed(4);
    ctyMeta.us.vs[0] = dxy.close;
    ctyMeta.us.cs[0] = parseFloat((dxy.chg ?? 0).toFixed(4));
  }
}

/* ================================================================
   OVERRIDE showCty to patch real chart data
   ================================================================ */

// Save reference to original mock function before this script runs
// (The mock's inline <script> runs synchronously before this file;
//  showCty is defined on the global scope)
const _origShowCty = window.showCty;

window.showCty = function(key) {
  // First run the original mock rendering
  if (typeof _origShowCty === 'function') _origShowCty(key);

  // Then overlay with real OHLC chart
  const pairFile = OHLC_FILE[key];
  if (!pairFile) return;
  fetchOhlc(pairFile).then(data => {
    if (data) redrawCtyChart(key, data);
  });
};

/* ================================================================
   BOOT
   ================================================================ */

async function loadRealData() {
  try {
    const extFetches = EXT_CCY.map(c =>
      fetch(`extended-data/${c}.json`).then(r => r.json()).catch(() => null)
    );

    const [quotesJson, meetingsJson, dxyData, ...extResults] = await Promise.all([
      fetch('intraday-data/quotes.json').then(r => r.json()).catch(() => null),
      fetch('meetings-data/meetings.json').then(r => r.json()).catch(() => null),
      fetch('ohlc-data/dxy.json').then(r => r.json()).catch(() => null),
      ...extFetches,
    ]);

    const quotes  = quotesJson?.quotes ?? {};
    const extData = {};
    EXT_CCY.forEach((c, i) => { extData[c] = extResults[i]?.data ?? null; });

    // Patch and re-render
    if (Object.keys(quotes).length) {
      patchRateRows(quotes, extData);
      patchCtyMeta(quotes);
      buildRates();
    }

    if (meetingsJson) {
      patchMeetings(meetingsJson);
    }

    patchEcoRows(extData);
    buildEco();

    // DXY chart
    if (Array.isArray(dxyData) && dxyData.length) {
      redrawDXYChart(dxyData);
    }

    // If a country view is currently showing, refresh its chart
    if (typeof curKey === 'string' && curKey !== 'all' && OHLC_FILE[curKey]) {
      fetchOhlc(OHLC_FILE[curKey]).then(data => {
        if (data) redrawCtyChart(curKey, data);
      });
    }

    // Timestamp in cmd bar
    const now = new Date();
    const hh = String(now.getUTCHours()).padStart(2,'0');
    const mm = String(now.getUTCMinutes()).padStart(2,'0');
    const ct2 = document.getElementById('ct2');
    if (ct2) {
      ct2.innerHTML = `<span class="kw">&lt;HELP&gt;</span> for explanation, `+
        `<span class="kw">&lt;MENU&gt;</span> for similar functions. `+
        `<span style="color:#3a6050">&#9632; Live data ${hh}:${mm} UTC</span>`;
    }

  } catch (err) {
    console.warn('[dashboard2] loadRealData error:', err);
  }
}

// Run on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    loadRealData();
    setInterval(loadRealData, 5 * 60 * 1000);
  });
} else {
  // document already parsed — fire immediately
  loadRealData();
  setInterval(loadRealData, 5 * 60 * 1000);
}
