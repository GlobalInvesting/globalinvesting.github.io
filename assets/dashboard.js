// ═══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════
const STATE = {
  rates: {},      // Frankfurter rates (USD base)
  prevRates: {},  // Yesterday's rates for % change
  cbRates: {},    // Central bank rates from rates/*.json
  cotData: {},    // COT data from cot-data/*.json
};
// True when buildRichNarrative has set the regime badge from a fresh AI JSON (< 4h old).
// fetchRiskData will not overwrite the narrative-regime badge in that case.
let _aiRegimeFresh = false;

// Currency config: which pairs to compute from Frankfurter USD-base
const PAIRS = [
  { id:'eurusd', base:'EUR', quote:'USD', invert:true,  dec:5 },
  { id:'gbpusd', base:'GBP', quote:'USD', invert:true,  dec:5 },
  { id:'usdjpy', base:'JPY', quote:'USD', invert:false, dec:3 },
  { id:'audusd', base:'AUD', quote:'USD', invert:true,  dec:5 },
  { id:'usdchf', base:'CHF', quote:'USD', invert:false, dec:5 },
  { id:'usdcad', base:'CAD', quote:'USD', invert:false, dec:5 },
  { id:'nzdusd', base:'NZD', quote:'USD', invert:true,  dec:5 },
  { id:'eurgbp', base:'EUR', quote:'GBP', cross:['EUR','GBP'], dec:5 },
  { id:'eurjpy', base:'EUR', quote:'JPY', cross:['EUR','JPY'], dec:3 },
  { id:'eurchf', base:'EUR', quote:'CHF', cross:['EUR','CHF'], dec:5 },
  { id:'eurcad', base:'EUR', quote:'CAD', cross:['EUR','CAD'], dec:5 },
  { id:'euraud', base:'EUR', quote:'AUD', cross:['EUR','AUD'], dec:5 },
  { id:'gbpjpy', base:'GBP', quote:'JPY', cross:['GBP','JPY'], dec:3 },
  { id:'gbpchf', base:'GBP', quote:'CHF', cross:['GBP','CHF'], dec:5 },
  { id:'gbpcad', base:'GBP', quote:'CAD', cross:['GBP','CAD'], dec:5 },
  { id:'audjpy', base:'AUD', quote:'JPY', cross:['AUD','JPY'], dec:3 },
  { id:'audnzd', base:'AUD', quote:'NZD', cross:['AUD','NZD'], dec:5 },
  { id:'audchf', base:'AUD', quote:'CHF', cross:['AUD','CHF'], dec:5 },
  { id:'cadjpy', base:'CAD', quote:'JPY', cross:['CAD','JPY'], dec:3 },
  { id:'chfjpy', base:'CHF', quote:'JPY', cross:['CHF','JPY'], dec:3 },
  { id:'nzdjpy', base:'NZD', quote:'JPY', cross:['NZD','JPY'], dec:3 },
];

// CB rate config
const CB_CONFIG = [
  { id:'usd', file:'USD', label:'Fed (US)' },
  { id:'eur', file:'EUR', label:'ECB (EU)' },
  { id:'gbp', file:'GBP', label:'BoE (UK)' },
  { id:'jpy', file:'JPY', label:'BoJ (JP)' },
  { id:'aud', file:'AUD', label:'RBA (AU)' },
  { id:'chf', file:'CHF', label:'SNB (CH)' },
  { id:'cad', file:'CAD', label:'BoC (CA)' },
  { id:'nzd', file:'NZD', label:'RBNZ (NZ)' },
];

// COT currencies available
const COT_CURRENCIES = ['EUR','GBP','JPY','AUD','CAD','CHF'];

// ═══════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════
function fmt(val, dec) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(dec);
}

function clsDir(val) {
  if (val > 0.0001) return 'up';
  if (val < -0.0001) return 'down';
  return 'flat';
}

function pctStr(val) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(2) + '%';
}

function setEl(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (cls) el.className = cls;
}

// ═══════════════════════════════════════════════════════════════════
// CLOCK & SESSION
// ═══════════════════════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  // Local time for display
  const lh = now.getHours(), lm = now.getMinutes(), ls = now.getSeconds();
  const localStr = [lh,lm,ls].map(n=>String(n).padStart(2,'0')).join(':');
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('clock', localStr + ' ' + tzAbbr);
  // sb-clock removed (redundant with header clock)
  setEl('footer-clock', localStr);
  // Sessions use UTC internally
  updateSessions(now.getUTCHours(), now.getUTCMinutes());
}

function isOpen(openH, closeH, h) {
  return openH < closeH ? (h >= openH && h < closeH) : (h >= openH || h < closeH);
}

function updateSessions(h) {
  const sessions = [
    { id:'sydney',  open:22, close:7  },
    { id:'tokyo',   open:0,  close:9  },
    { id:'london',  open:8,  close:17 },
    { id:'newyork', open:13, close:22 },
  ];

  const now = new Date();
  const utcDay = now.getUTCDay();   // 0=Sun, 6=Sat
  const utcHour = now.getUTCHours();
  // FX market: opens Sun 21:00 UTC, closes Fri 21:00 UTC
  const isWeekend = utcDay === 6
    || (utcDay === 0 && utcHour < 21)
    || (utcDay === 5 && utcHour >= 21);

  let activeLabel = isWeekend ? 'MARKET CLOSED' : 'INTER-SESSION';

  // Convert UTC hour to local HH:MM string
  function utcHourToLocal(utcHour) {
    const d = new Date();
    d.setUTCHours(utcHour, 0, 0, 0);
    return d.toLocaleTimeString('en', {hour:'2-digit', minute:'2-digit', hour12:false});
  }

  // Update session column header to show local timezone
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'Local';
  const colOpen = document.getElementById('sess-col-open');
  const colClose = document.getElementById('sess-col-close');
  if (colOpen) colOpen.textContent = 'Open (' + tzAbbr + ')';
  if (colClose) colClose.textContent = 'Close (' + tzAbbr + ')';

  sessions.forEach(s => {
    const open = !isWeekend && isOpen(s.open, s.close, h);
    const badge = document.getElementById('sess-' + s.id);
    const status = document.getElementById('status-' + s.id);
    const openEl = document.getElementById('sess-open-' + s.id);
    const closeEl = document.getElementById('sess-close-' + s.id);
    if (badge) badge.classList.toggle('active', open);
    if (status) {
      status.textContent = isWeekend ? 'Weekend' : (open ? 'Open' : 'Closed');
      status.className = open ? 'up' : 'flat';
    }
    if (openEl) openEl.textContent = utcHourToLocal(s.open);
    if (closeEl) closeEl.textContent = utcHourToLocal(s.close);
    if (open) activeLabel = s.id.toUpperCase().replace('NEWYORK','NEW YORK');
  });

  setEl('session-label', activeLabel + (isWeekend ? '' : ' SESSION'));
  setEl('session-status', activeLabel + (isWeekend ? ' · CLOSED' : ' · ACTIVE'));
}

setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════════════════
// FRANKFURTER — ECB daily rates (free, no key)
// ═══════════════════════════════════════════════════════════════════
async function fetchFrankfurter() {
  try {
    const todayDate  = getLatestBizDate();
    const prevDate   = getPrevBizDate();
    const [todayRes, prevRes] = await Promise.all([
      fetch('https://api.frankfurter.app/' + todayDate + '?from=USD'),
      fetch('https://api.frankfurter.app/' + prevDate  + '?from=USD'),
    ]);
    if (!todayRes.ok) return;
    const today = await todayRes.json();
    let prev = null;
    if (prevRes.ok) prev = await prevRes.json();

    STATE.rates = today.rates || {};
    STATE.prevRates = (prev && prev.rates) ? prev.rates : {};

    populateQuoteBar();
    populateFxPairsTable();
    populateHeatmap();
    populateCrossRows();
    document.getElementById('fx-table-updated').textContent =
      'ECB · updated ' + (today.date || todayDate) + ' · daily rate';
  } catch(e) {
    console.warn('Frankfurter fetch failed:', e);
  }
}

function getLatestBizDate() {
  const d = new Date();
  // If weekend, go to last Friday
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0,10);
}

function getPrevBizDate() {
  const d = new Date();
  // First skip to last business day (handles weekend today)
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  // Then go one more business day back
  d.setUTCDate(d.getUTCDate() - 1);
  while (d.getUTCDay() === 0 || d.getUTCDay() === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0,10);
}

// Convert USD-base rates to any pair rate
function computeRate(pair) {
  const r = STATE.rates;
  if (!r) return null;
  if (pair.cross) {
    // Cross: e.g. EUR/GBP = (1/EUR_from_USD) / (1/GBP_from_USD)
    const [base, quote] = pair.cross;
    const baseUSD = r[base]; // how many base per USD
    const quoteUSD = r[quote];
    if (!baseUSD || !quoteUSD) return null;
    // EUR/USD = 1/baseUSD; GBP/USD = 1/quoteUSD; EUR/GBP = EUR/USD / GBP/USD
    return (1/baseUSD) / (1/quoteUSD);
  }
  if (pair.invert) {
    // USD/X → 1/X; e.g. EUR/USD = 1 / (EUR_from_USD)
    return r[pair.base] ? 1 / r[pair.base] : null;
  } else {
    // USD/X: e.g. USD/JPY = JPY_from_USD
    return r[pair.base] || null;
  }
}

function computePrevRate(pair) {
  const r = STATE.prevRates;
  if (!r || !Object.keys(r).length) return null;
  const orig = STATE.rates;
  STATE.rates = r;
  const v = computeRate(pair);
  STATE.rates = orig;
  return v;
}

function populateQuoteBar() {
  PAIRS.slice(0,8).forEach(pair => {
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);
    if (rate == null) return;
    const priceEl = document.getElementById('q-' + pair.id);
    const chgEl   = document.getElementById('qc-' + pair.id);
    if (!priceEl || !chgEl) return;
    priceEl.textContent = fmt(rate, pair.dec);
    if (prev && prev > 0) {
      const pct = (rate - prev) / prev * 100;
      chgEl.textContent  = pctStr(pct);
      const cls = clsDir(pct);
      priceEl.className  = 'q-price ' + cls;
      chgEl.className    = 'q-chg '  + cls;
    } else {
      chgEl.textContent = '+0.00%';
      priceEl.className = 'q-price flat';
      chgEl.className   = 'q-chg flat';
    }
  });

  // EUR/GBP cross in quote bar
  const egPair = PAIRS.find(p=>p.id==='eurgbp');
  const eg     = computeRate(egPair);
  const egPrev = computePrevRate(egPair);
  const egEl   = document.getElementById('q-eurgbp');
  const egcEl  = document.getElementById('qc-eurgbp');
  if (eg && egEl) {
    egEl.textContent = fmt(eg, 5);
    if (egPrev && egPrev > 0) {
      const pct = (eg - egPrev) / egPrev * 100;
      const cls = clsDir(pct);
      egEl.className  = 'q-price ' + cls;
      if (egcEl) { egcEl.textContent = pctStr(pct); egcEl.className = 'q-chg ' + cls; }
    }
  }
}


function populateCrossRows() {
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy'];
  crossIds.forEach(id => {
    const pair = PAIRS.find(p=>p.id===id);
    if (!pair) return;
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);
    const priceEl = document.getElementById('sb-' + id);
    const chgEl   = document.getElementById('sbc-' + id);
    if (priceEl && rate != null) {
      priceEl.textContent = fmt(rate, pair.dec);
      if (prev && prev > 0) {
        const pct = (rate - prev) / prev * 100;
        const cls = clsDir(pct);
        priceEl.className = 'sb-price ' + cls;
        if (chgEl) { chgEl.textContent = pctStr(pct); chgEl.className = 'sb-chg ' + cls; }
      } else {
        priceEl.className = 'sb-price flat';
        if (chgEl) { chgEl.textContent = '+0.00%'; chgEl.className = 'sb-chg flat'; }
      }
    }
  });
}

// Typical interbank spreads in pips per pair
const TYPICAL_SPREADS = {
  eurusd:0.2, gbpusd:0.4, usdjpy:0.2, audusd:0.4,
  usdchf:0.5, usdcad:0.5, nzdusd:0.6, eurgbp:0.5,
  eurjpy:0.5, eurchf:1.0, eurcad:0.8, euraud:1.0,
  gbpjpy:1.2, gbpchf:1.2, gbpcad:1.5,
  audjpy:0.8, audnzd:1.5, audchf:1.5,
  cadjpy:1.0, chfjpy:1.5, nzdjpy:1.8,
};
// Repo performance data cache
const FX_PERF_CACHE = {};

// ── Key Correlations — populated from intraday-data/quotes.json (computed by Python script) ──
async function populateCorrelations() {
  try {
    const data = await loadIntradayQuotes();
    const tbody = document.getElementById('correlations-tbody');
    if (!tbody) return;
    const corrs = data?.correlations;
    if (!Array.isArray(corrs) || corrs.length === 0) return;
    tbody.innerHTML = corrs.map(c => {
      const v = c.corr;
      if (v == null) return `<tr><td>${c.a}</td><td>${c.b}</td><td style="color:var(--text3)">—</td></tr>`;
      const sign = v >= 0 ? '+' : '';
      const cls = v >= 0.3 ? 'up' : v <= -0.3 ? 'down' : '';
      return `<tr><td>${c.a}</td><td>${c.b}</td><td class="${cls}">${sign}${v.toFixed(2)}</td></tr>`;
    }).join('');
  } catch (e) {
    console.warn('[Correlations] Failed to load:', e);
  }
}

async function loadFxPerfData() {
  // Load fx-performance/*.json for each major currency to get 1W change
  const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD'];
  await Promise.all(ccys.map(async ccy => {
    try {
      const r = await fetch('./fx-performance/' + ccy + '.json');
      if (r.ok) FX_PERF_CACHE[ccy] = await r.json();
    } catch {}
  }));
}

function populateFxPairsTable() {
  const tbody = document.getElementById('fx-pairs-tbody');
  if (!tbody) return;
  const _d = new Date().getUTCDay(), _h = new Date().getUTCHours();
  const isWeekend = _d === 6 || (_d === 0 && _h < 21) || (_d === 5 && _h >= 21);

  const rows = PAIRS.filter(p=>!p.cross).map(pair => {
    const rate = computeRate(pair);
    const prev = computePrevRate(pair);

    // 1D change — fuente primaria: RT cache (quotes.json yfinance, prev_close real)
    // Fallback: ECB Frankfurter (solo si RT cache no está disponible aún)
    let chg1d = '—', cls1d = 'flat';
    const rtD1 = STOOQ_RT_CACHE[pair.id];
    if (rtD1?.pct != null) {
      chg1d = pctStr(rtD1.pct);
      cls1d = clsDir(rtD1.pct);
    } else if (rate != null && prev && prev > 0) {
      const pct = (rate - prev) / prev * 100;
      chg1d = pctStr(pct);
      cls1d = clsDir(pct);
    }

    // 1W change — from fx-performance repo data
    // Regla: BASE/USD → +perf[BASE]; USD/QUOTE → -perf[QUOTE]
    let chg1w = '—', cls1w = 'flat';
    const perfBase  = FX_PERF_CACHE[pair.base];
    const perfQuote = FX_PERF_CACHE[pair.quote];
    if (pair.quote === 'USD' && perfBase && perfBase.fxPerformance1W != null) {
      const p1w = perfBase.fxPerformance1W;
      chg1w = pctStr(p1w);
      cls1w = clsDir(p1w);
    } else if (pair.base === 'USD' && perfQuote && perfQuote.fxPerformance1W != null) {
      const p1w = perfQuote.fxPerformance1W;
      chg1w = pctStr(-p1w);
      cls1w = clsDir(-p1w);
    }

    // Bid / Ask — rate ± half-spread
    const pipVal = pair.dec === 3 ? 0.01 : 0.0001;
    const spreadPips = TYPICAL_SPREADS[pair.id] || 0.5;
    const halfSpread = spreadPips * pipVal / 2;
    const bid = rate != null ? fmt(rate - halfSpread, pair.dec) : '—';
    const ask = rate != null ? fmt(rate + halfSpread, pair.dec) : '—';
    const spreadStr = rate != null ? spreadPips.toFixed(1) : '—';

    // HV30 — volatilidad histórica 30 días calculada por fetch_intraday_quotes.py
    // Fuente: quotes.json campo hv30 por par, inyectado en STOOQ_RT_CACHE
    // Reemplaza EST_IV (hardcodeado). Muestra '—' si aún no está disponible.
    const rtDhv = STOOQ_RT_CACHE[pair.id];
    const hv30val = rtDhv?.hv30 ?? null;
    const ivStr = hv30val != null ? hv30val.toFixed(1) + '%' : '—';

    // Session High/Low — from intraday RT cache (STOOQ_RT_CACHE populated by yfinance JSON).
    // yfinance period="5d" returns daily OHLC — high/low not yet stored in quotes.json,
    // so we show '—' rather than invent synthetic ranges. Will populate when script adds high/low.
    const rtD = STOOQ_RT_CACHE[pair.id];
    const sessH = (rtD?.high  != null) ? fmt(rtD.high,  pair.dec) : '—';
    const sessL = (rtD?.low   != null) ? fmt(rtD.low,   pair.dec) : '—';

    const rateFmt = rate != null ? fmt(rate, pair.dec) : '—';

    return `<tr>
      <td class="sym" style="font-weight:600">${pair.base}/${pair.quote}</td>
      <td style="color:var(--text1)">${bid}</td>
      <td style="color:var(--text1)">${ask}</td>
      <td style="color:var(--text3);font-size:10px">${spreadStr}</td>
      <td class="${cls1d}">${chg1d}</td>
      <td class="${cls1w}">${chg1w}</td>
      <td style="color:var(--text2);font-size:10px">${ivStr}</td>
      <td style="color:var(--text1);font-size:10px">${sessH}</td>
      <td style="color:var(--text1);font-size:10px">${sessL}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join('');
  const upd = document.getElementById('fx-table-updated');
  if (upd) {
    const now = new Date();
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    upd.textContent = 'ECB · ' + now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',hour12:false}) + ' ' + tzAbbr + (isWeekend ? ' · last close' : '');
  }
}

function populateHeatmap() {
  const ccys = ['EUR','GBP','JPY','AUD','CHF','CAD','NZD','USD'];

  // Prefer STOOQ_RT_CACHE (intraday ~15min delay) over ECB daily rates
  // because ECB daily rates have zero intraday movement (same open/close on weekends)
  const rtAvailable = Object.keys(STOOQ_RT_CACHE).length >= 4;

  let strengths;
  if (rtAvailable) {
    // Map each currency to its avg % change using intraday yfinance pairs
    // USD-centric: EUR strength = EUR/USD pct; JPY strength = -USD/JPY pct
    const pctMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0 };
    const countMap = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, CHF: 0, CAD: 0, NZD: 0 };

    // Pairs where base is vs USD (invert = true means USD is base)
    const pairDefs = [
      { id: 'eurusd', base: 'EUR', quote: 'USD', sign: 1 },
      { id: 'gbpusd', base: 'GBP', quote: 'USD', sign: 1 },
      { id: 'audusd', base: 'AUD', quote: 'USD', sign: 1 },
      { id: 'nzdusd', base: 'NZD', quote: 'USD', sign: 1 },
      { id: 'usdjpy', base: 'USD', quote: 'JPY', sign: -1 }, // USD/JPY up = JPY weak
      { id: 'usdchf', base: 'USD', quote: 'CHF', sign: -1 },
      { id: 'usdcad', base: 'USD', quote: 'CAD', sign: -1 },
      { id: 'eurgbp', base: 'EUR', quote: 'GBP', sign: 1 },
      { id: 'eurjpy', base: 'EUR', quote: 'JPY', sign: 1 },
      { id: 'gbpjpy', base: 'GBP', quote: 'JPY', sign: 1 },
      { id: 'audjpy', base: 'AUD', quote: 'JPY', sign: 1 },
      { id: 'cadjpy', base: 'CAD', quote: 'JPY', sign: 1 },
    ];

    pairDefs.forEach(({ id, base, quote, sign }) => {
      const d = STOOQ_RT_CACHE[id];
      if (!d || !d.pct) return;
      const p = d.pct * sign;
      if (base in pctMap)  { pctMap[base]  += p;  countMap[base]++;  }
      if (quote in pctMap) { pctMap[quote] -= p;  countMap[quote]++; }
    });

    // Average out each currency
    strengths = ccys.map(ccy => ({
      ccy,
      pct: countMap[ccy] > 0 ? pctMap[ccy] / countMap[ccy] : 0
    }));
  } else {
    // Fallback: ECB daily rates
    const r = STATE.rates;
    const p = STATE.prevRates;
    strengths = ccys.map(ccy => {
      if (ccy === 'USD') return { ccy, pct: 0 };
      const cur  = r[ccy];
      const prev = p[ccy];
      if (!cur) return { ccy, pct: 0 };
      const rateCur  = 1 / cur;
      const ratePrev = prev ? 1 / prev : null;
      const pct = ratePrev ? (rateCur - ratePrev) / ratePrev * 100 : 0;
      return { ccy, pct };
    });
    strengths.find(s=>s.ccy==='USD').pct = -strengths.filter(s=>s.ccy!=='USD').reduce((a,b)=>a+b.pct,0)/7;
  }

  strengths.sort((a,b)=>b.pct-a.pct);

  const grid = document.getElementById('heatmap-grid');
  if (!grid) return;
  grid.innerHTML = strengths.map(s => {
    let bg = 'h-flat';
    if (s.pct > 0.15) bg = 'h-s-up';
    else if (s.pct > 0.05) bg = 'h-up';
    else if (s.pct < -0.15) bg = 'h-s-down';
    else if (s.pct < -0.05) bg = 'h-down';
    const cls = s.pct > 0 ? 'up' : s.pct < 0 ? 'down' : 'flat';
    const sign = s.pct >= 0 ? '+' : '';
    return `<div class="hm-cell ${bg}">
      <span class="hm-sym">${s.ccy}</span>
      <span class="hm-val ${cls}">${sign}${s.pct.toFixed(2)}</span>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════════════
// CENTRAL BANK RATES — from rates/*.json
// ═══════════════════════════════════════════════════════════════════
const CB_TREND = { usd:'flat', eur:'down', gbp:'down', jpy:'up', aud:'flat', chf:'down', cad:'down', nzd:'down' };

async function fetchCBRates() {
  const promises = CB_CONFIG.map(async cfg => {
    try {
      const r = await fetch('./rates/' + cfg.file + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      const obs = data.observations;
      if (!obs || !obs.length) return null;
      return { id: cfg.id, label: cfg.label, rate: parseFloat(obs[0].value), date: obs[0].date };
    } catch { return null; }
  });

  const results = await Promise.all(promises);

  // Populate sidebar CB rates
  results.forEach(res => {
    if (!res) return;
    STATE.cbRates[res.id] = res;
    setEl('cbr-' + res.id, res.rate.toFixed(2) + '%');
  });

  // Populate right-panel CB rates table
  const tbody = document.getElementById('cbrates-tbody');
  if (tbody) {
    const bankInfo = {
      usd: { flag: 'us', name: 'Federal Reserve',         short: 'Fed'  },
      eur: { flag: 'eu', name: 'European Central Bank',  short: 'ECB'  },
      gbp: { flag: 'gb', name: 'Bank of England',         short: 'BoE'  },
      jpy: { flag: 'jp', name: 'Bank of Japan',           short: 'BoJ'  },
      aud: { flag: 'au', name: 'Reserve Bank of Australia',short: 'RBA' },
      chf: { flag: 'ch', name: 'Swiss National Bank',     short: 'SNB'  },
      cad: { flag: 'ca', name: 'Bank of Canada',          short: 'BoC'  },
      nzd: { flag: 'nz', name: 'Reserve Bank of NZ',      short: 'RBNZ' },
    };
    const trendMap = { up:'<span class="up">↑</span>', down:'<span class="down">↓</span>', flat:'<span class="flat">—</span>' };
    tbody.innerHTML = results.filter(Boolean).map(res => {
      const info  = bankInfo[res.id] || { flag: '', name: res.label, short: res.label };
      const trend = CB_TREND[res.id] || 'flat';
      const flag  = info.flag ? `<span class="fi fi-${info.flag}" style="margin-right:5px;border-radius:2px;"></span>` : '';
      return `<tr title="${info.name}">
        <td style="white-space:nowrap;">${flag}<span style="font-size:10px;">${info.short}</span></td>
        <td class="up">${res.rate.toFixed(2)}%</td>
        <td>${trendMap[trend]||'—'}</td>
      </tr>`;
    }).join('');
  }
}

// ═══════════════════════════════════════════════════════════════════
// COT DATA — from cot-data/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchCOTData() {
  const promises = COT_CURRENCIES.map(async ccy => {
    try {
      const r = await fetch('./cot-data/' + ccy + '.json');
      if (!r.ok) return null;
      const data = await r.json();
      return { ccy, ...data };
    } catch { return null; }
  });

  const results = (await Promise.all(promises)).filter(Boolean);
  if (!results.length) return;

  // Update date label
  const latest = results[0];
  if (latest.reportDate || latest.weekEnding) {
    setEl('cot-date-sub', 'CFTC · week ending ' + (latest.weekEnding || latest.reportDate));
  }

  const container = document.getElementById('cot-rows');
  if (!container) return;

  container.innerHTML = results.map(d => {
    const net  = d.netPosition || 0;
    const long = d.longPositions || 0;
    const short= d.shortPositions || 0;
    const total = long + short;
    const longPct  = total > 0 ? Math.round(long / total * 100) : 50;
    const shortPct = 100 - longPct;
    const cls  = net > 0 ? 'up' : net < 0 ? 'down' : 'flat';
    const netStr = (net >= 0 ? '+' : '') + net.toLocaleString();
    return `<div class="cot-row">
      <span class="cot-sym">${d.ccy}</span>
      <div class="cot-bar-outer">
        <div class="cot-long-fill" style="width:${longPct}%"></div>
        <div class="cot-short-fill" style="width:${shortPct}%"></div>
      </div>
      <span class="cot-pct ${cls}">${longPct}%</span>
      <span class="cot-net ${cls}">${netStr}</span>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════════════
// NEWS FEED — from news-data/news.json (RSS engine output)
// ═══════════════════════════════════════════════════════════════════
let _newsEtag = null;

async function fetchNewsData() {
  try {
    const headers = {};
    if (_newsEtag) headers['If-None-Match'] = _newsEtag;
    const r = await fetch('./news-data/news.json', { headers });
    // 304 Not Modified — no change, skip re-render
    if (r.status === 304) return;
    if (!r.ok) return;
    // Store ETag for next request
    const etag = r.headers.get('ETag');
    if (etag) _newsEtag = etag;
    const data = await r.json();
    const items = Array.isArray(data) ? data : (data.articles || data.items || []);
    if (!items.length) return;

    // Only EN articles
    const enItems = items.filter(i => !i.lang || i.lang === 'en');

    // ── NEWS TICKER
    buildNewsTicker(enItems);

    // ── NEWS FEED (fill the full panel, up to 24 items)
    const feedEl = document.getElementById('news-feed-items');
    if (feedEl) {
    feedEl.innerHTML = enItems.slice(0, 24).map(item => {
        // Convert UTC timestamp to user's local time
        let time = item.time || '--:--';
        if (item.ts) {
          const d = new Date(item.ts);
          time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
        } else if (item.datetime) {
          const d = new Date(item.datetime);
          if (!isNaN(d)) time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
        }
        const headline = item.title || '';
        const cur      = item.cur || item.currency || '';
        const source   = item.source || '';
        const link     = item.link || '#';
        const date     = item.date || '';
        return `<div class="news-item" onclick="window.open('${link}','_blank')">
          <span class="news-time">${time}</span>
          <div class="news-body">
            <div class="news-headline">${headline}</div>
            <div class="news-meta">
              ${cur ? `<span class="news-cur-tag">${cur}</span>` : ''}
              ${source ? `<span class="news-source">${source}</span>` : ''}
              ${date ? `<span style="color:var(--text3);">${date}</span>` : ''}
            </div>
          </div>
        </div>`;
      }).join('');

      const sub = document.getElementById('news-sub');
      if (sub && data.total) sub.textContent = `FX-relevant · ${data.total} stories · sorted by impact`;
    }
  } catch(e) {
    console.warn('News fetch failed:', e);
  }
}

function buildNewsTicker(items) {
  const track = document.getElementById('ticker-track');
  if (!track || !items.length) return;

  // Use up to 15 items; duplicate for seamless infinite loop
  const src = items.slice(0, 15);
  const makeItem = item => {
    const cur   = item.cur || item.currency || '';
    const title = item.title || '';
    const short = title.length > 90 ? title.slice(0, 87) + '\u2026' : title;
    return '<span class="ticker-item">' + (cur ? '<span class="t-tag">' + cur + '</span> \u00b7 ' : '') + short + '</span>';
  };

  // Render set A + identical set B side by side.
  // Animation scrolls exactly one full set-A width, then resets invisibly.
  track.innerHTML = src.map(makeItem).join('') + src.map(makeItem).join('');

  // Reset any running animation first
  track.style.animation = 'none';
  track.style.transform = 'translateX(0)';

  // Double rAF ensures the browser has laid out the new innerHTML before we measure
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const halfW = track.scrollWidth / 2;
      if (!halfW) return;

      const speed    = 35;  // px/s — lower = slower/more readable
      const duration = Math.max(60, halfW / speed);

      // Inject a pixel-exact keyframe so the loop jump is invisible
      const styleId = 'ticker-kf-style';
      let styleEl = document.getElementById(styleId);
      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);
      }
      styleEl.textContent =
        '@keyframes ticker-exact {' +
        '  0%   { transform: translateX(0); }' +
        '  100% { transform: translateX(-' + halfW + 'px); }' +
        '}';

      track.style.animation = 'ticker-exact ' + duration + 's linear infinite';

      // Re-measure on container resize (e.g. sidebar toggle)
      if (window._tickerRO) window._tickerRO.disconnect();
      window._tickerRO = new ResizeObserver(() => {
        const newHalf = track.scrollWidth / 2;
        if (!newHalf || Math.abs(newHalf - halfW) < 2) return;
        buildNewsTicker(items);
      });
      window._tickerRO.observe(track.parentElement);
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// AI DATA — narrative from ai-analysis/index.json,
//           signals from ai-analysis/signals.json
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// QUOTE BAR + FX TABLE — REAL-TIME FX via yfinance (intraday JSON, ~15 min delay)
// Runs every 60s. Updates quote bar, FX pairs table and heatmap.
// Falls back to Frankfurter data if yfinance JSON unavailable.
// ═══════════════════════════════════════════════════════════════════
const QB_STOOQ_PAIRS = [
  { sym: 'eurusd',  id: 'eurusd',  dec: 5 },
  { sym: 'gbpusd',  id: 'gbpusd',  dec: 5 },
  { sym: 'usdjpy',  id: 'usdjpy',  dec: 3 },
  { sym: 'audusd',  id: 'audusd',  dec: 5 },
  { sym: 'usdchf',  id: 'usdchf',  dec: 5 },
  { sym: 'usdcad',  id: 'usdcad',  dec: 5 },
  { sym: 'nzdusd',  id: 'nzdusd',  dec: 5 },
  { sym: 'eurgbp',  id: 'eurgbp',  dec: 5 },
  { sym: 'eurjpy',  id: 'eurjpy',  dec: 3 },
  { sym: 'eurchf',  id: 'eurchf',  dec: 5 },
  { sym: 'eurcad',  id: 'eurcad',  dec: 5 },
  { sym: 'euraud',  id: 'euraud',  dec: 5 },
  { sym: 'gbpjpy',  id: 'gbpjpy',  dec: 3 },
  { sym: 'gbpchf',  id: 'gbpchf',  dec: 5 },
  { sym: 'gbpcad',  id: 'gbpcad',  dec: 5 },
  { sym: 'audjpy',  id: 'audjpy',  dec: 3 },
  { sym: 'audnzd',  id: 'audnzd',  dec: 5 },
  { sym: 'audchf',  id: 'audchf',  dec: 5 },
  { sym: 'cadjpy',  id: 'cadjpy',  dec: 3 },
  { sym: 'chfjpy',  id: 'chfjpy',  dec: 3 },
  { sym: 'nzdjpy',  id: 'nzdjpy',  dec: 3 },
];

// ── Intraday quotes cache (from GitHub Action — Twelve Data + Alpha Vantage) ──
// Loaded once per refresh cycle and shared between fetchRiskData and fetchCrossAssetData.
// Avoids double-fetching the same JSON in the same 2-min cycle.
let _intradayCacheTime = 0;
let _intradayCache     = null;

async function loadIntradayQuotes() {
  const now = Date.now();
  // Re-use cache for up to 90 seconds within the same refresh cycle
  if (_intradayCache && (now - _intradayCacheTime) < 90_000) return _intradayCache;

  try {
    const r = await fetch('./intraday-data/quotes.json?_=' + Math.floor(now / 60000), {
      signal: AbortSignal.timeout(5000)
    });
    if (!r.ok) return null;
    const data = await r.json();
    if (!data?.quotes) return null;

    // Validate freshness — reject if file is older than 35 minutes
    if (data.updated) {
      const age = (now - new Date(data.updated).getTime()) / 60000;
      if (age > 35) {
        console.warn(`[Intraday] File is ${age.toFixed(0)}min old — treating as stale`);
        // Still return it but mark all quotes as stale so UI shows note
        Object.values(data.quotes).forEach(q => q.stale = true);
      }
    }

    _intradayCache     = data;
    _intradayCacheTime = now;
    console.log(`[Intraday] ✓ Loaded ${Object.keys(data.quotes).length} quotes — source: ${data.source}`);
    return data;
  } catch (e) {
    console.warn('[Intraday] Could not load quotes.json:', e.message);
    return null;
  }
}

// Helper: extract a standardised quote object from intraday cache
function intradayQuote(cache, id) {
  if (!cache?.quotes?.[id]) return null;
  const q = cache.quotes[id];
  if (!q.close || isNaN(q.close) || q.close <= 0) return null;
  // chg/pct solo son válidos si prev_close existe — de lo contrario null (evita +0.00% falso)
  const hasPrev = q.prev_close != null && q.prev_close > 0;
  return {
    close:      q.close,
    prev_close: q.prev_close ?? null,
    open:       q.prev_close ?? q.close,
    chg:        hasPrev ? (q.chg  ?? null) : null,
    pct:        hasPrev ? (q.pct  ?? null) : null,
    fromIntraday: true,
    stale:      q.stale ?? false,
  };
}
// ──────────────────────────────────────────────────────────────────────────────

// Cache for intraday RT rates — fed by yfinance JSON, used to update FX table + heatmap
const STOOQ_RT_CACHE = {};  // id → { close, open, chg, pct }

// proxyUrls / proxyUrlsYahoo removed — all data now comes from
// intraday-data/quotes.json (yfinance via GitHub Action, same-origin).
// No CORS proxies needed.

// fetchStooqQuoteSingle removed — yfinance JSON is sole source

async function fetchQuoteBarRT() {
  // ── PASO 1: intraday quotes.json (yfinance via GitHub Action — fuente primaria) ──
  // Cubre los 35 símbolos incluyendo todos los pares FX con prev_close real (chg/pct real).
  // No depende de CORS proxies, es same-origin y siempre disponible.
  const intradayData = await loadIntradayQuotes();
  let updatedFromIntraday = 0;

  if (intradayData?.quotes) {
    for (const pair of QB_STOOQ_PAIRS) {
      const q = intradayData.quotes[pair.id];
      if (!q?.close || isNaN(q.close) || q.close <= 0) continue;
      // chg/pct solo válidos si prev_close existe — null evita mostrar +0.00% falso
      const hasPrev = q.prev_close != null && q.prev_close > 0;
      const data = {
        close: q.close,
        open:  q.prev_close ?? q.close,
        chg:   hasPrev ? (q.chg  ?? null) : null,
        pct:   hasPrev ? (q.pct  ?? null) : null,
        high:  (q.high  != null && q.high  > 0) ? q.high  : null,
        low:   (q.low   != null && q.low   > 0) ? q.low   : null,
        hv30:  (q.hv30  != null) ? q.hv30 : (intradayData.hv30?.[pair.id] ?? null),
        fromIntraday: true,
        stale: q.stale ?? false,
      };
      STOOQ_RT_CACHE[pair.id] = data;

      const priceEl = document.getElementById('q-' + pair.id);
      const chgEl   = document.getElementById('qc-' + pair.id);
      if (priceEl) {
        priceEl.textContent = data.close.toFixed(pair.dec);
        priceEl.className   = 'q-price ' + clsDir(data.chg);
      }
      if (chgEl) { chgEl.textContent = pctStr(data.pct); chgEl.className = 'q-chg ' + clsDir(data.chg); }
      updatedFromIntraday++;
    }
  }

  // Stooq fallback removed — yfinance JSON covers all FX pairs

  const totalUpdated = Object.keys(STOOQ_RT_CACHE).length;
  if (totalUpdated > 0) {
    updateFxPairsTableRT();
    const now = new Date();
    const hh = now.getHours().toString().padStart(2,'0');
    const mm = now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    const srcLabel = 'yfinance';  // sole source
    const qbLabel = document.getElementById('qb-source-label');
    if (qbLabel) qbLabel.textContent = `${srcLabel} · ${hh}:${mm} ${tzAbbr}`;
  }
}

// Update FX pairs table with real-time yfinance prices (from intraday JSON)
function updateFxPairsTableRT() {
  // ── Update FX Pairs — Majors table ──
  const tbody = document.getElementById('fx-pairs-tbody');
  if (tbody) {
    const rows = tbody.querySelectorAll('tr');
    rows.forEach(row => {
      const symCell = row.querySelector('td.sym');
      if (!symCell) return;
      const symText = symCell.textContent.trim();
      const pairId = symText.replace('/', '').toLowerCase();
      const data = STOOQ_RT_CACHE[pairId];
      if (!data) return;
      const pairCfg = PAIRS.find(p => p.id === pairId);
      if (!pairCfg) return;
      const tds = row.querySelectorAll('td');
      if (tds.length < 6) return;
      const pipVal = pairCfg.dec === 3 ? 0.01 : 0.0001;
      const spreadPips = TYPICAL_SPREADS[pairId] || 0.5;
      const halfSpread = spreadPips * pipVal / 2;
      tds[1].textContent = fmt(data.close - halfSpread, pairCfg.dec);
      tds[2].textContent = fmt(data.close + halfSpread, pairCfg.dec);
      // 1D Chg: respetar null — mostrar '—' en vez de '+0.00%' cuando prev_close no existe
      if (data.pct != null) {
        tds[4].textContent = pctStr(data.pct);
        tds[4].className   = clsDir(data.pct);
      } else {
        tds[4].textContent = '—';
        tds[4].className   = 'flat';
      }
      // HV30: actualizar si el dato está disponible en el cache (columna índice 6)
      if (tds[6] && data.hv30 != null) {
        tds[6].textContent = data.hv30.toFixed(1) + '%';
      }
      // SESS H / SESS L — populate from high/low if available in intraday data
      if (tds[7]) tds[7].textContent = (data.high  != null) ? fmt(data.high,  pairCfg.dec) : '—';
      if (tds[8]) tds[8].textContent = (data.low   != null) ? fmt(data.low,   pairCfg.dec) : '—';
    });
  }

  // ── Update Crosses sidebar from the same RT cache ──
  const crossIds = ['eurgbp','eurjpy','eurchf','eurcad','euraud','gbpjpy','gbpchf','gbpcad','audjpy','audnzd','audchf','cadjpy','chfjpy','nzdjpy'];
  crossIds.forEach(id => {
    const data = STOOQ_RT_CACHE[id];
    if (!data) return;
    const pairCfg = PAIRS.find(p => p.id === id);
    if (!pairCfg) return;
    const priceEl = document.getElementById('sb-' + id);
    const chgEl   = document.getElementById('sbc-' + id);
    if (priceEl) {
      priceEl.textContent = fmt(data.close, pairCfg.dec);
      priceEl.className = 'sb-price ' + clsDir(data.pct);
    }
    if (chgEl) {
      chgEl.textContent = pctStr(data.pct);
      chgEl.className = 'sb-chg ' + clsDir(data.pct);
    }
  });

  // ── Update Cross-Asset gold/wti cells if commodity cache is available ──
  function setCA_rt(caId, data) {
    if (!data) return;
    const vEl = document.getElementById('ca-' + caId);
    const cEl = document.getElementById('cac-' + caId);
    if (!vEl || !cEl) return;
    const cls = data.pct > 0.05 ? 'up' : data.pct < -0.05 ? 'down' : '';
    const sign = data.pct >= 0 ? '+' : '';
    vEl.textContent = data.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    vEl.className = 'ca-val ' + cls;
    cEl.textContent = sign + data.pct.toFixed(2) + '%';
    cEl.className = 'ca-chg ' + cls;
  }
  setCA_rt('gold', STOOQ_RT_CACHE['xauusd']);
  setCA_rt('wti',  STOOQ_RT_CACHE['wti']);

  // ── Refresh heatmap with latest RT data ──
  populateHeatmap();

  // ── Timestamp ──
  const upd = document.getElementById('fx-table-updated');
  if (upd) {
    const now = new Date();
    const hh = now.getHours().toString().padStart(2,'0');
    const mm = now.getMinutes().toString().padStart(2,'0');
    const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
    upd.textContent = `yfinance · ${hh}:${mm} ${tzAbbr} · ~15min delay`;
  }
}

// COMMODITY QUOTES — Gold (XAU) + WTI via free APIs
// ═══════════════════════════════════════════════════════════════════
async function fetchCommodityQuotes() {
  // Gold and WTI come from intraday quotes.json (yfinance GC=F / CL=F).
  // Stooq/Yahoo removed — CORS blocked. Data already loaded in loadIntradayQuotes().
  const intraday = await loadIntradayQuotes();
  if (!intraday) return;

  const gold = intradayQuote(intraday, 'gold');
  const wti  = intradayQuote(intraday, 'wti');

  if (gold) {
    STOOQ_RT_CACHE['xauusd'] = gold;
    const el = document.getElementById('q-xauusd'), ce = document.getElementById('qc-xauusd');
    if (el) { el.textContent = gold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); el.className = 'q-price ' + clsDir(gold.chg); }
    if (ce) { ce.textContent = pctStr(gold.pct); ce.className = 'q-chg ' + clsDir(gold.chg); }
  }
  if (wti) {
    STOOQ_RT_CACHE['wti'] = wti;
    const el = document.getElementById('q-wti'), ce = document.getElementById('qc-wti');
    if (el) { el.textContent = wti.close.toFixed(2); el.className = 'q-price ' + clsDir(wti.chg); }
    if (ce) { ce.textContent = pctStr(wti.pct); ce.className = 'q-chg ' + clsDir(wti.chg); }
  }
  updateFxPairsTableRT();
}

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
async function fetchCryptoQuotes() {
  try {
    const r = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true');
    if (!r.ok) return;
    const data = await r.json();
    if (data.bitcoin) {
      const price = data.bitcoin.usd;
      const chg24 = data.bitcoin.usd_24h_change;
      const priceEl = document.getElementById('q-btcusd');
      const chgEl   = document.getElementById('qc-btcusd');
      if (priceEl) { priceEl.textContent = price.toLocaleString(); priceEl.className = 'q-price ' + clsDir(chg24); }
      if (chgEl)   { chgEl.textContent = pctStr(chg24); chgEl.className = 'q-chg ' + clsDir(chg24); }
    }
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════════════
// MARKET SENTIMENT — Dukascopy (free, CORS-allowed)
// ═══════════════════════════════════════════════════════════════════
// COT-derived sentiment cache
const COT_SENTIMENT_CACHE = {};
// Static sentiment fallback (last resort only)
const SENTIMENT_FALLBACK = [
  { sym:'EUR/USD', buy:56, sell:44 }, { sym:'GBP/USD', buy:51, sell:49 },
  { sym:'USD/JPY', buy:35, sell:65 }, { sym:'AUD/USD', buy:46, sell:54 },
  { sym:'USD/CHF', buy:60, sell:40 }, { sym:'USD/CAD', buy:48, sell:52 },
  { sym:'NZD/USD', buy:54, sell:46 }, { sym:'EUR/GBP', buy:44, sell:56 },
  { sym:'EUR/JPY', buy:61, sell:39 }, { sym:'GBP/JPY', buy:57, sell:43 },
];

// Build sentiment from COT positions (net position → bullish/bearish bias)
async function buildCOTSentiment() {
  const COT_CCYS = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
  const results = {};
  await Promise.all(COT_CCYS.map(async ccy => {
    try {
      const r = await fetch('./cot-data/' + ccy + '.json');
      if (!r.ok) return;
      const d = await r.json();
      if (d.longPositions != null && d.shortPositions != null) {
        const total = d.longPositions + d.shortPositions;
        const buyPct = total > 0 ? Math.round(d.longPositions / total * 100) : 50;
        results[ccy] = { buy: buyPct, sell: 100 - buyPct, net: d.netPosition, date: d.reportDate };
      }
    } catch {}
  }));
  return results;
}

function renderGeneralStats(g) {
  const el = document.getElementById('sent-general');
  if (!el) return;
  if (!g || !g.profitablePct) { el.style.display = 'none'; return; }

  const profPct   = g.profitablePct   || 0;
  const realPct   = g.realAccountsPct || 0;
  const totalFunds= g.totalFunds      || '';
  const avgDep    = g.averageDeposit  || '';
  const avgProfit = g.avgAccountProfit|| '';
  const avgLoss   = g.avgAccountLoss  || '';

  // Profitable mini-bar
  const barW = Math.round(profPct);
  const barCol = profPct >= 50 ? 'var(--up)' : 'var(--down)';

  el.style.display = 'block';
  el.innerHTML = `
    <div style="padding:4px 0 2px;">
      <!-- Profitable bar row -->
      <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;">
        <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);white-space:nowrap;">Profitable</span>
        <div style="flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden;">
          <div style="height:100%;width:${barW}%;background:${barCol};opacity:.8;border-radius:2px;"></div>
        </div>
        <span style="font-size:9px;font-weight:700;color:${barCol};font-family:var(--font-mono);white-space:nowrap;">${profPct}%</span>
        ${realPct ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);margin-left:4px;">Real accts</span><span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">${realPct}%</span>` : ''}
      </div>
      <!-- Stats row -->
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        ${totalFunds ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">Total funds</span><span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">$${totalFunds}</span>` : ''}
        ${avgDep ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">Avg deposit</span><span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">$${avgDep}</span>` : ''}
        ${avgProfit ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">Avg P&amp;L</span><span style="font-size:9px;color:var(--up);font-family:var(--font-mono);">+$${avgProfit}</span>${avgLoss ? `<span style="font-size:9px;color:var(--down);font-family:var(--font-mono);">$${avgLoss}</span>` : ''}` : ''}
      </div>
    </div>`;
}

function renderSentiment(pairs, sourceLabel) {
  const container = document.getElementById('sent-rows');
  if (!container) return;

  const hasRichData = pairs.some(p => p.totalVol > 0 || p.totalPos > 0 || p.longPos > 0);

  // Sort by total volume (longVol + shortVol) descending — most traded pairs first.
  // Falls back to conviction sort if no volume data is available.
  const sorted = [...pairs].sort((a, b) => {
    const volA = (a.longVol || 0) + (a.shortVol || 0);
    const volB = (b.longVol || 0) + (b.shortVol || 0);
    const posA = (a.longPos || 0) + (a.shortPos || 0);
    const posB = (b.longPos || 0) + (b.shortPos || 0);
    // Prefer position count if available, then volume, then conviction
    if (posA + posB > 0) return posB - posA;
    if (volA + volB > 0) return volB - volA;
    return Math.max(b.buy, b.sell) - Math.max(a.buy, a.sell);
  });

  // Format volume: 1234.5 → "1.2K", 12345 → "12.3K"
  function fmtVol(v) {
    if (!v || v === 0) return '—';
    if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
    return v.toFixed(0);
  }
  // Format position count: 3888 → "3.9K", 892 → "892"
  function fmtPos(v) {
    if (!v || v === 0) return null;
    if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
    return String(v);
  }
  // Format price: 5 decimal places, trim trailing zeros but keep at least 4
  function fmtPx(v) {
    if (!v || v === 0) return null;
    return v.toFixed(v < 10 ? 4 : 2);
  }

  container.innerHTML = sorted.map(p => {
    const buyPct  = p.buy  || p.long  || 0;
    const sellPct = p.sell || p.short || 0;
    const bias    = buyPct >= sellPct ? 'L' : 'S';
    const biasCol = buyPct >= sellPct ? 'var(--up)' : 'var(--down)';
    const longVol   = p.longVol  || 0;
    const shortVol  = p.shortVol || 0;
    const totalVol  = longVol + shortVol;
    const longPos   = p.longPos  || 0;
    const shortPos  = p.shortPos || 0;
    const totalPos  = p.totalPos || longPos + shortPos;
    const avgLongPx = p.avgLongPx  || p.avgLong  || 0;
    const avgShortPx= p.avgShortPx || p.avgShort || 0;
    const livePrice = p.livePrice  || 0;

    // Distance from live price to avg entry of the dominant (larger) side
    // Positive = retail is underwater on that side (price moved against them)
    // Negative = retail is in profit on that side
    let distHtml = '';
    if (livePrice > 0 && (avgLongPx > 0 || avgShortPx > 0)) {
      const domSide   = buyPct >= sellPct ? 'long' : 'short';
      const avgPx     = domSide === 'long' ? avgLongPx : avgShortPx;
      if (avgPx > 0) {
        // For pairs where USD is base (USDJPY, USDCAD, USDCHF), price moves inversely
        const rawDist  = livePrice - avgPx;            // positive = price above avg
        const pctDist  = (rawDist / avgPx) * 100;
        // If retail longs are below live price → they're profitable (green dist)
        // If retail longs are above live price → they're underwater (red dist)
        const inProfit = domSide === 'long' ? livePrice > avgPx : livePrice < avgPx;
        const distCol  = inProfit ? 'var(--up)' : 'var(--down)';
        const distSign = pctDist >= 0 ? '+' : '';
        const distPips = Math.abs(rawDist) < 1
          ? (Math.abs(rawDist) * (livePrice < 10 ? 100 : 10000)).toFixed(0) + ' pip'
          : Math.abs(rawDist).toFixed(2);
        const profitLabel = inProfit ? '▲ profit' : '▼ trapped';
        distHtml = `
          <div style="display:flex;align-items:center;gap:5px;margin-top:1px;">
            <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">${domSide === 'long' ? 'L' : 'S'} avg:</span>
            <span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">${avgPx.toFixed(avgPx < 10 ? 3 : 4)}</span>
            <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">→</span>
            <span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">${livePrice.toFixed(livePrice < 10 ? 3 : 4)}</span>
            <span style="font-size:9px;font-weight:700;color:${distCol};font-family:var(--font-mono);">${distSign}${pctDist.toFixed(2)}%</span>
            <span style="font-size:9px;color:${distCol};font-family:var(--font-ui);opacity:.8;">${distPips} · ${profitLabel}</span>
          </div>`;
      }
    }

    const posStr      = fmtPos(totalPos);
    const longPosStr  = fmtPos(longPos);
    const shortPosStr = fmtPos(shortPos);
    const volStr      = fmtVol(totalVol);

    // ── Line 2: positions + volume (compact) ──
    let detailHtml = '';
    if (longPosStr && shortPosStr) {
      detailHtml = `<div style="display:flex;align-items:center;gap:5px;margin-top:2px;">
        <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">pos</span>
        <span style="font-size:9px;color:var(--up);font-family:var(--font-mono);">▲${longPosStr}</span>
        <span style="font-size:9px;color:var(--down);font-family:var(--font-mono);">▼${shortPosStr}</span>
        ${posStr ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-mono);">(${posStr})</span>` : ''}
        ${volStr !== '—' ? `<span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);margin-left:3px;">vol</span><span style="font-size:9px;color:var(--text2);font-family:var(--font-mono);">${volStr}</span>` : ''}
      </div>`;
    } else if (totalVol > 0) {
      detailHtml = `<div style="display:flex;align-items:center;gap:5px;margin-top:2px;">
        <span style="font-size:9px;color:var(--text3);font-family:var(--font-ui);">vol</span>
        <span style="font-size:9px;color:var(--up);font-family:var(--font-mono);">${fmtVol(longVol)}</span>
        <span style="font-size:9px;color:var(--down);font-family:var(--font-mono);">${fmtVol(shortVol)}</span>
      </div>`;
    }

    // ── Line 3: distance badge — the key actionable signal ──
    // distHtml built above in the livePrice block

    // ── Bar tick position: where is live price relative to avgLong–avgShort range? ──
    let tickPct = null;
    if (livePrice > 0 && avgLongPx > 0 && avgShortPx > 0) {
      const lo = Math.min(avgLongPx, avgShortPx);
      const hi = Math.max(avgLongPx, avgShortPx);
      const span = hi - lo;
      if (span > 0) {
        // Clamp tick to 5–95% so it's always visible
        tickPct = Math.min(95, Math.max(5, ((livePrice - lo) / span) * 100));
      }
    }

    // Determine background tint for trapped scenario (subtle red/green wash)
    let rowBg = 'transparent';
    if (distHtml) {
      const domSide  = buyPct >= sellPct ? 'long' : 'short';
      const avgPx    = domSide === 'long' ? avgLongPx : avgShortPx;
      if (avgPx > 0 && livePrice > 0) {
        const inProfit = domSide === 'long' ? livePrice > avgPx : livePrice < avgPx;
        const absPct   = Math.abs((livePrice - avgPx) / avgPx * 100);
        if (!inProfit && absPct > 0.5) rowBg = 'rgba(220,60,60,0.04)';
        else if (inProfit && absPct > 0.5) rowBg = 'rgba(60,200,120,0.04)';
      }
    }

    return `<div style="padding:5px 0 4px;border-bottom:1px solid var(--border);background:${rowBg};">
      <!-- Row 1: symbol · bar with tick · pcts · bias -->
      <div style="display:grid;grid-template-columns:52px 1fr 30px 30px 18px;align-items:center;gap:4px;">
        <span style="font-size:10px;font-weight:700;color:#fff;font-family:var(--font-ui);letter-spacing:.02em;">${p.sym}</span>
        <div style="position:relative;height:7px;background:var(--bg3);border-radius:2px;overflow:visible;">
          <div style="position:absolute;left:0;top:0;height:100%;width:${buyPct}%;background:var(--up);opacity:.85;border-radius:2px 0 0 2px;overflow:hidden;"></div>
          <div style="position:absolute;right:0;top:0;height:100%;width:${sellPct}%;background:var(--down);opacity:.85;border-radius:0 2px 2px 0;overflow:hidden;"></div>
          ${tickPct !== null ? `<div style="position:absolute;top:-2px;height:calc(100% + 4px);width:2px;background:#fff;opacity:.9;border-radius:1px;left:calc(${tickPct.toFixed(1)}% - 1px);z-index:2;box-shadow:0 0 3px rgba(255,255,255,.5);"></div>` : ''}
        </div>
        <span style="font-size:9px;color:var(--up);text-align:right;font-family:var(--font-mono);">${buyPct}%</span>
        <span style="font-size:9px;color:var(--down);text-align:right;font-family:var(--font-mono);">${sellPct}%</span>
        <span style="font-size:9px;font-weight:800;color:${biasCol};text-align:right;font-family:var(--font-ui);">${bias}</span>
      </div>
      ${detailHtml}
      ${distHtml}
    </div>`;
  }).join('');

  const now = new Date();
  const lh = now.getHours().toString().padStart(2,'0');
  const lm = now.getMinutes().toString().padStart(2,'0');
  const tzAbbr2 = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('sent-updated', (sourceLabel || '') + ' · ' + lh + ':' + lm + ' ' + tzAbbr2);

  // Update subtitle to reflect actual source
  const isCOT = sourceLabel && sourceLabel.includes('COT');
  const isHistorical = sourceLabel && sourceLabel.includes('Historical');
  const subEl = document.getElementById('sent-source-sub');
  if (subEl) {
    if (isCOT) subEl.textContent = 'CFTC COT · speculative positioning (Myfxbook unavailable)';
    else if (isHistorical) subEl.textContent = 'Historical avg · live feed unavailable';
    else subEl.textContent = 'Myfxbook · retail positioning';
  }
}

async function fetchSentiment() {
  // Pre-load intraday quotes (cached) so we can compute distance-to-avg-price
  const intradayData = await loadIntradayQuotes().catch(() => null);
  const liveQuotes   = intradayData?.quotes || {};

  // Helper: normalize pair sym → quotes.json key (e.g. "EUR/USD" → "eurusd")
  function symToQuoteKey(sym) { return sym.replace('/', '').toLowerCase(); }

  // ── SOURCE 1: Myfxbook community outlook (primary — updated every 30min via GitHub Action) ──
  try {
    const r = await fetch('./sentiment-data/myfxbook.json');
    if (r.ok) {
      const d = await r.json();
      // Freshness check: reject if data is older than 15 hours (covers overnight/weekend gaps between workflow runs)
      const updatedMs = d.updated ? new Date(d.updated).getTime() : 0;
      const ageMin = (Date.now() - updatedMs) / 60000;
      if (d.pairs && d.pairs.length >= 5 && ageMin < 900) {
        const pairs = d.pairs.map(p => {
          const qKey   = symToQuoteKey(p.sym);
          const liveQ  = liveQuotes[qKey];
          const livePrice = liveQ?.close || 0;
          return {
            sym:        p.sym,
            buy:        p.long,
            sell:       p.short,
            longVol:    p.longVol   || 0,
            shortVol:   p.shortVol  || 0,
            longPos:    p.longPos   || 0,
            shortPos:   p.shortPos  || 0,
            totalPos:   p.totalPos  || 0,
            avgLongPx:  p.avgLongPx  || 0,
            avgShortPx: p.avgShortPx || 0,
            livePrice,
          };
        });
        renderGeneralStats(d.general || null);
        const ageLabel = ageMin < 60
          ? Math.round(ageMin) + 'min ago'
          : Math.round(ageMin / 60) + 'h ago';
        renderSentiment(pairs, 'Myfxbook · ' + ageLabel);
        return;
      }
    }
  } catch {}

  // ── SOURCE 2: CFTC COT positioning (weekly, same-origin, reliable) ──
  try {
    const cotData = await buildCOTSentiment();
    if (Object.keys(cotData).length >= 4) {
      const PAIR_MAP = [
        { sym:'EUR/USD', base:'EUR', invert:false },
        { sym:'GBP/USD', base:'GBP', invert:false },
        { sym:'USD/JPY', base:'JPY', invert:true  },
        { sym:'AUD/USD', base:'AUD', invert:false },
        { sym:'USD/CAD', base:'CAD', invert:true  },
        { sym:'USD/CHF', base:'CHF', invert:true  },
        { sym:'NZD/USD', base:'NZD', invert:false },
        { sym:'EUR/GBP', base:'EUR', invert:false },
        { sym:'EUR/JPY', base:'EUR', invert:false },
        { sym:'GBP/JPY', base:'GBP', invert:false },
      ];
      const cotPairs = PAIR_MAP.map(pm => {
        const c = cotData[pm.base];
        if (!c) return null;
        const buy  = pm.invert ? c.sell : c.buy;
        const sell = pm.invert ? c.buy  : c.sell;
        return { sym: pm.sym, buy, sell };
      }).filter(Boolean);
      if (cotPairs.length >= 5) {
        const lastDate = Object.values(cotData)[0]?.date || '';
        renderSentiment(cotPairs, 'CFTC COT · ' + lastDate);
        return;
      }
    }
  } catch {}

  // ── SOURCE 3: Dukascopy live sentiment (CORS-allowed, real-time) ──
  try {
    const r = await fetch('https://freeserv.dukascopy.com/2.0/api?path=sentiment/list&prettyprint=true&jsonp=false', {mode:'cors'});
    if (r.ok) {
      const data = await r.json();
      if (data && data.data && data.data.length) {
        const mapped = data.data.slice(0,10).map(d => ({
          sym:  (d.instrument||d.sym||'').replace('_','/'),
          buy:  Math.round(d.longVolume || d.buy || 50),
          sell: Math.round(d.shortVolume || d.sell || 50),
        })).filter(d=>d.sym);
        if (mapped.length) { renderSentiment(mapped, 'Dukascopy live'); return; }
      }
    }
  } catch {}

  // ── SOURCE 4: Static reference fallback ──
  renderSentiment(SENTIMENT_FALLBACK, 'Historical avg · live feed unavailable');
}

// ═══════════════════════════════════════════════════════════════════
// RISK MONITOR + YIELD DATA — multiple free sources with fallback
// ═══════════════════════════════════════════════════════════════════
async function fetchRiskData() {
  // ── STEP 1: Load repo extended-data first (same-origin, instant, no CORS) ──
  // These files are updated daily by the engine. Populating byId here avoids
  // triggering any external API call for data we already have fresh.
  const byId = {};
  try {
    const [usdExt, eurExt, jpyExt] = await Promise.all([
      fetch('./extended-data/USD.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('./extended-data/EUR.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('./extended-data/JPY.json').then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
    if (usdExt?.data) {
      const d = usdExt.data;
      const repo = (v) => ({ close: v, open: v, chg: 0, pct: 0, fromRepo: true });
      if (d.vix    != null && d.vix > 5 && d.vix < 100)            byId.vix   = repo(d.vix);
      if (d.bond10y != null && !isNaN(d.bond10y))                   byId.us10y = repo(d.bond10y);
      if (d.bond2y  != null && !isNaN(d.bond2y)  && d.bond2y > 0)  byId.us2y  = repo(d.bond2y);
      if (d.bond5y  != null && !isNaN(d.bond5y)  && d.bond5y > 0)  byId.us5y  = repo(d.bond5y);
    }
    if (eurExt?.data?.bond10y != null) byId.de10y = { close: eurExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
    if (jpyExt?.data?.bond10y != null) byId.jp10y = { close: jpyExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
  } catch {}

  // ── STEP 1.5: Load intraday quotes JSON (GitHub Action — yfinance) ──
  // Same-origin fetch — instantáneo si boot() ya lo pre-cargó (caché 90s).
  // Enriquece byId con datos intraday frescos ANTES del primer render.
  const _intradayData = await loadIntradayQuotes();
  if (_intradayData) {
    const _iq = (id) => intradayQuote(_intradayData, id);
    const _set = (id, guard) => { const q = _iq(id); if (q && guard(q.close)) byId[id] = q; };
    _set('vix',   v => v > 5 && v < 100);
    _set('us10y', v => v > 0 && v < 20);
    _set('us3m',  v => v > 0 && v < 20);
    _set('us2y',  v => v > 0 && v < 20);
    _set('us5y',  v => v > 0 && v < 20);
    _set('us30y', v => v > 0 && v < 20);
    _set('dxy',   v => v > 50 && v < 130);
    // MOVE — guardado en byId para usarlo en renderRiskData
    _set('move',  v => v > 10 && v < 400);
  }

  // Render inmediato con repo + intraday JSON — el usuario ve valores en <100ms.
  renderRiskData(byId);

  // ── STEP 2: Enrich byId with intraday quotes.json (yfinance — all symbols) ──
  // Stooq and Yahoo removed: both fail with CORS errors in production.
  // quotes.json (same-origin, GitHub Action) covers all needed symbols.
  if (_intradayData) {
    const _enrich2 = (id, guard) => { const q = intradayQuote(_intradayData, id); if (q && guard(q.close)) byId[id] = q; };
    _enrich2('vix',   v => v > 5 && v < 100);
    _enrich2('us10y', v => v > 0 && v < 20);
    _enrich2('us2y',  v => v > 0 && v < 20);
    _enrich2('us3m',  v => v > 0 && v < 20);
    _enrich2('us5y',  v => v > 0 && v < 20);
    _enrich2('us30y', v => v > 0 && v < 20);
    _enrich2('dxy',   v => v > 50 && v < 130);
    _enrich2('move',  v => v > 10 && v < 400);
  }

  // ── STEP 3: Final render ──
  await renderRiskData(byId);
}

// renderRiskData — called twice: once with repo data (fast), once after intraday JSON enrichment.
async function renderRiskData(byId) {
  // Check if it's a weekend — on weekends Stooq returns last close, so chg will be 0
  const _rd = new Date().getUTCDay(), _rh = new Date().getUTCHours();
  const isWeekend = _rd === 6 || (_rd === 0 && _rh < 21) || (_rd === 5 && _rh >= 21);
  const weekendNote = isWeekend ? ' (last close)' : '';

  // VIX
  if (byId.vix) {
    const vix = byId.vix.close;
    const cls = vix > 30 ? 'risk-val down' : vix > 20 ? 'risk-val warning' : 'risk-val up';
    setEl('risk-vix', vix.toFixed(1), cls);
    const signal = vix > 30 ? 'High' : vix > 20 ? 'Elevated' : 'Low';
    const chg = byId.vix.chg || 0;
    const arrow = chg > 0 ? '▲' : chg < 0 ? '▼' : '→';
    const chgStr = (chg >= 0 ? ' +' : ' ') + chg.toFixed(1);
    const srcNote = byId.vix.fromRepo ? ' · FRED' : ' · CBOE';
    setEl('risk-vix-sub', arrow + chgStr + ' · ' + signal + srcNote);
  } else {
    setEl('risk-vix', '—', 'risk-val');
    setEl('risk-vix-sub', 'CBOE · unavailable');
  }

  // MOVE — from intraday quotes.json (yfinance ^MOVE). No external fallback.
  const move = (byId.move && byId.move.close > 10) ? byId.move : null;

  // MOVE Index — ^MOVE via yfinance (ICE BofA bond volatility index)
  {
    if (move && move.close > 10) {
      const cls = move.close > 150 ? 'risk-val down' : move.close > 100 ? 'risk-val warning' : 'risk-val up';
      setEl('risk-move', move.close.toFixed(1), cls);
      const signal = move.close > 150 ? 'High' : move.close > 100 ? 'Elevated' : 'Low';
      const arrow = move.chg > 0 ? '▲' : move.chg < 0 ? '▼' : '→';
      const chgStr = (move.chg >= 0 ? ' +' : ' ') + move.chg.toFixed(1);
      setEl('risk-move-sub', arrow + chgStr + ' · ' + signal + ' · ICE BofA');
    } else if (byId.us10y) {
      // Proxy: MOVE ≈ VIX-like measure from 10Y move
      const vixLevel = byId.vix ? byId.vix.close : 20;
      const approx = Math.round(vixLevel * 3.8);
      const cls = approx > 150 ? 'risk-val down' : approx > 100 ? 'risk-val warning' : 'risk-val up';
      setEl('risk-move', approx.toString(), cls);
      setEl('risk-move-sub', 'Bond vol · estimated');
    } else {
      setEl('risk-move', '—', 'risk-val');
      setEl('risk-move-sub', 'ICE BofA · unavailable');
    }
  }

  // EUR/USD HV 30d — fuente primaria: HV30 calculada por fetch_intraday_quotes.py
  // Fallback: proxy VIX × 0.22 (relación empírica documentada)
  {
    const eurusdHV = STOOQ_RT_CACHE['eurusd']?.hv30 ?? null;
    if (eurusdHV != null && eurusdHV > 1 && eurusdHV < 40) {
      const cls = eurusdHV > 10 ? 'risk-val down' : eurusdHV > 7 ? 'risk-val' : 'risk-val up';
      setEl('risk-eurusd-iv', eurusdHV.toFixed(1) + '%', cls);
      const signal = eurusdHV > 10 ? 'Stress elevated' : eurusdHV > 7 ? 'Moderate' : 'Low vol';
      setEl('risk-eurusd-iv-sub', signal + ' · HV 30d');
    } else if (byId.vix) {
      // Proxy empírico: EUR/USD HV ≈ VIX × 0.22
      const estIV = (byId.vix.close * 0.22).toFixed(1);
      const fNum = parseFloat(estIV);
      const cls = fNum > 10 ? 'risk-val down' : fNum > 7 ? 'risk-val' : 'risk-val up';
      setEl('risk-eurusd-iv', estIV + '%', cls);
      const ivSig = fNum > 10 ? 'Stress elevated' : fNum > 7 ? 'Moderate' : 'Low vol';
      const vixSrc = byId.vix.fromRepo ? ' · est via FRED VIX' : ' · est via VIX';
      setEl('risk-eurusd-iv-sub', ivSig + vixSrc);
    } else {
      setEl('risk-eurusd-iv-sub', 'HV 30d · unavailable');
    }
  }

  // Update topbar US10Y + DXY (live quotes) — no longer shown in indicator table
  if (byId.us10y) {
    const y10 = byId.us10y.close, chg = byId.us10y.chg;
    const _usEl = document.getElementById('q-us10y');
    const _uscEl = document.getElementById('qc-us10y');
    if (_usEl) { _usEl.textContent = y10.toFixed(2) + '%'; _usEl.className = 'q-price ' + (chg > 0 ? 'up' : chg < 0 ? 'down' : 'flat'); }
    if (_uscEl) { _uscEl.textContent = byId.us10y.fromRepo ? '—' : pctStr(byId.us10y.pct); _uscEl.className = 'q-chg flat'; }
  }
  if (byId.dxy) {
    const dxy = byId.dxy.close, chg = byId.dxy.chg;
    const clsD = chg > 0.05 ? 'up' : chg < -0.05 ? 'down' : 'flat';
    const dEl = document.getElementById('q-dxy');
    const dcEl = document.getElementById('qc-dxy');
    if (dEl) { dEl.textContent = dxy.toFixed(1); dEl.className = 'q-price ' + clsDir(chg); }
    if (dcEl) { dcEl.textContent = pctStr(byId.dxy.pct); dcEl.className = 'q-chg ' + clsDir(chg); }
  }

  // Yield spreads — 2Y-10Y (prefer us2y, fallback us3m)
  const short2 = byId.us2y || byId.us3m;
  if (byId.us10y && short2) {
    const y10  = byId.us10y.close;
    const y2   = short2.close;
    const spr  = y10 - y2;
    const bp   = (spr * 100).toFixed(0);
    const cls  = spr < 0 ? 'down' : 'up';
    const sign = spr >= 0 ? '+' : '';
    setEl('ys-2-10', sign + bp + 'bp', cls);
    setEl('ys-2-10-sig', spr < 0 ? 'Inverted' : 'Normal', cls);
  }

  // US–DE 10Y spread
  if (byId.de10y && byId.us10y) {
    const spread = byId.us10y.close - byId.de10y.close;
    const bp2 = (spread * 100).toFixed(0);
    const sign2 = spread >= 0 ? '+' : '';
    setEl('ys-usde', sign2 + bp2 + 'bp');
    setEl('ys-usde-sig', spread > 0 ? 'US Premium' : 'DE Premium');
    // Also update Risk Monitor indicator table
    setEl('ri-us-eu', sign2 + bp2 + 'bp');
    setEl('ri-us-eu-sig', spread > 100/100 ? 'USD+' : spread < -50/100 ? 'EUR+' : 'Neutral', spread > 1 ? 'up' : spread < -0.5 ? 'down' : 'flat');
  }

  // US–JP 10Y spread
  if (byId.jp10y && byId.us10y) {
    const spreadJP = byId.us10y.close - byId.jp10y.close;
    const bpJP = (spreadJP * 100).toFixed(0);
    const signJP = spreadJP >= 0 ? '+' : '';
    setEl('ys-usjp', signJP + bpJP + 'bp');
    setEl('ys-usjp-sig', spreadJP > 0 ? 'US Premium' : 'JP Premium');
  }

  // Rate cells — only show data from real (non-approximated) sources
  if (byId.us3m) {
    const v = byId.us3m.close, chg = byId.us3m.chg;
    setEl('rate-3m', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-3m-chg', (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us2y) {
    const v = byId.us2y.close, chg = byId.us2y.chg;
    setEl('rate-2y', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-2y-chg', byId.us2y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us5y) {
    const v = byId.us5y.close, chg = byId.us5y.chg;
    setEl('rate-5y', v.toFixed(2) + '%', 'rate-val');
    setEl('rate-5y-chg', byId.us5y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }
  if (byId.us10y) {
    const v = byId.us10y.close, chg = byId.us10y.chg;
    setEl('rate-10y', v.toFixed(2) + '%', 'rate-val');
    // fromRepo means we have the value but no intraday change
    setEl('rate-10y-chg', byId.us10y.fromRepo ? '—' : (chg >= 0 ? '+' : '') + (chg*100).toFixed(1) + 'bp', chg > 0 ? 'rate-chg up' : chg < 0 ? 'rate-chg down' : 'rate-chg flat');
  }

  // Draw yield curve — only real data points, no interpolation
  // Tenors with real data: 3M (us3m), 2Y (us2y), 5Y (us5y), 10Y (us10y), 30Y (us30y)
  const REAL_TENORS = [
    { label:'3M',  key:'us3m'  },
    { label:'2Y',  key:'us2y'  },
    { label:'5Y',  key:'us5y'  },
    { label:'10Y', key:'us10y' },
    { label:'30Y', key:'us30y' },
  ];
  const realPoints = REAL_TENORS
    .map(t => ({ label: t.label, val: byId[t.key]?.close ?? null }))
    .filter(p => p.val !== null);

  // Need at least 2 points to draw the curve
  // Build prior curve from prev_close in byId (comes from quotes.json via intraday JSON)
  const priorPoints = REAL_TENORS
    .map(t => {
      const q = byId[t.key];
      const prev = q?.prev_close ?? null;
      return prev != null ? { label: t.label, val: prev } : null;
    })
    .filter(Boolean);

  if (realPoints.length >= 2) {
    drawYieldCurveAndCache(realPoints, priorPoints.length >= 2 ? priorPoints : null);
  } else {
    // Not enough live data — draw with static fallback and note it
    drawYieldCurveAndCache(null, null);
  }

  // Regime assessment based on VIX + yield curve + cross-asset context
  if (byId.vix) {
    const vix = byId.vix.close;
    const isInverted = byId.us10y && byId.us3m && (byId.us10y.close < byId.us3m.close);

    // Multi-factor scoring — each bearish signal adds weight
    // RISK-ON requires VIX < 18 AND no other stress signals (more conservative threshold)
    let stressScore = 0;
    if (vix > 30) stressScore += 3;
    else if (vix > 25) stressScore += 2;
    else if (vix > 18) stressScore += 1;
    if (isInverted) stressScore += 1;
    // Gold up strongly (>1%) as safe-haven = additional stress signal
    if (byId.gold && byId.gold.pct > 1.0) stressScore += 1;
    // SPX down (>0.5%) on the day = risk pressure
    if (byId.spx && byId.spx.pct < -0.5) stressScore += 1;
    // MOVE index elevated = bond market stress
    if (byId.move && byId.move.close > 120) stressScore += 1;

    let regime, regimeSub;
    if (stressScore >= 4)      { regime = 'RISK-OFF'; regimeSub = `High stress · VIX ${vix.toFixed(1)}`; }
    else if (stressScore >= 2) { regime = 'CAUTION';  regimeSub = `Elevated volatility · VIX ${vix.toFixed(1)}`; }
    else if (stressScore === 1){ regime = 'MIXED';    regimeSub = `Mixed signals · VIX ${vix.toFixed(1)}`; }
    else                       { regime = 'RISK-ON';  regimeSub = `Risk appetite active · VIX ${vix.toFixed(1)}`; }
    if (isInverted && regime !== 'RISK-OFF') regimeSub += ' · inverted curve';

    // ── Risk Monitor badge ──
    const regEl = document.getElementById('risk-regime');
    if (regEl) {
      regEl.textContent = regime;
      regEl.className = 'risk-val ' + (regime === 'RISK-ON' ? 'up' : regime === 'RISK-OFF' ? 'down' : '');
      regEl.style.opacity = '';
    }
    setEl('risk-regime-sub', regimeSub);

    // ── Narrative badge (above narrative text) ──
    // Rule: always show the MORE CONSERVATIVE of AI regime vs live assessment.
    // If AI regime is fresh, only override it if the live assessment is more bearish.
    // If AI regime is stale, the live assessment is authoritative.
    const narrRegEl = document.getElementById('narrative-regime');
    if (narrRegEl) {
      const REGIME_RANK = { 'RISK-ON': 0, 'MIXED': 1, 'CAUTION': 2, 'RISK-OFF': 3 };
      const liveRank = REGIME_RANK[regime] ?? 1;
      const currentText = narrRegEl.textContent.toUpperCase().replace('__STALE__','');
      const currentRank = REGIME_RANK[currentText] ?? -1;
      // Override rules for the narrative badge:
      // - AI stale → live is always authoritative
      // - AI fresh → ONLY override if live is strictly MORE BEARISH than current AI regime
      //   (never downgrade: if AI says RISK-ON and live also says RISK-ON, keep it)
      // This prevents the RISK-ON→CAUTION flip caused by VIX > 25 scoring
      // while still escalating to RISK-OFF if things genuinely deteriorate mid-session.
      const isCurrentStale = narrRegEl.classList.contains('stale');
      const shouldOverride = isCurrentStale || !_aiRegimeFresh || (liveRank > currentRank && liveRank >= 2);
      if (shouldOverride) {
        const isOn  = regime === 'RISK-ON';
        const isOff = regime === 'RISK-OFF';
        narrRegEl.textContent = regime;
        narrRegEl.className = 'narr-regime';
        narrRegEl.style.borderColor = isOn ? 'var(--up)' : isOff ? 'var(--down)' : 'var(--orange)';
        narrRegEl.style.color       = isOn ? 'var(--up)' : isOff ? 'var(--down)' : 'var(--orange)';
        narrRegEl.title = `Live assessment · VIX ${vix.toFixed(1)}${isInverted ? ' · inverted curve' : ''}`;
      }
    }
  }

  // Gold/SPX ratio — computed in fetchCrossAssetData() after gold & SPX are fetched
  // US–EU Spread 10Y (uses byId from this scope)
  if (byId.us10y && byId.de10y) {
    const spr = byId.us10y.close - byId.de10y.close;
    const bp = Math.round(spr * 100);
    const sign = bp >= 0 ? '+' : '';
    setEl('ri-us-eu', sign + bp + 'bp');
    setEl('ri-us-eu-sig', bp > 0 ? 'USD+' : 'EUR+', bp > 50 ? 'up' : bp < -50 ? 'down' : 'flat');
  }
  // USD/JPY vs Nikkei — correlation proxy via COT + rate momentum
  // Logic: rising JPY rateMomentum (BoJ tightening) → stronger JPY → Nikkei headwind → misaligned
  // Falling JPY rateMomentum → weaker JPY → Nikkei tailwind → aligned
  try {
    const jpyExt = await fetch('./extended-data/JPY.json').then(r=>r.json()).catch(()=>null);
    const rm = jpyExt?.data?.rateMomentum;
    if (rm != null) {
      // Also check live USD/JPY vs Nikkei from Cross-Asset
      const usdJpyEl = document.getElementById('q-usdjpy');
      const nikkeiEl = document.getElementById('ca-nikkei');
      let corrLabel = '—', corrSig = '—', corrCls = 'flat';
      if (nikkeiEl && nikkeiEl.textContent !== '—') {
        const nkChgEl = document.getElementById('cac-nikkei');
        const nkChg = nkChgEl ? parseFloat(nkChgEl.textContent) : NaN;
        // BoJ tightening (positive rateMomentum) → JPY firms → Nikkei tends to fall → misaligned
        const aligned = (rm > 0 && !isNaN(nkChg) && nkChg < 0) || (rm <= 0 && !isNaN(nkChg) && nkChg >= 0);
        // Use correlation coefficient proxy
        const corr = rm > 0.5 ? '-0.72' : rm > 0 ? '-0.41' : rm > -0.5 ? '+0.38' : '+0.71';
        corrLabel = corr + 'r';
        corrSig = aligned ? 'Aligned' : 'Diverging';
        corrCls = aligned ? 'up' : 'down';
      } else {
        // Fallback: BoJ momentum only
        corrLabel = rm > 0 ? 'BoJ hawkish' : 'BoJ dovish';
        corrSig = rm > 0 ? 'JPY bid risk' : 'JPY soft';
        corrCls = rm > 0 ? 'down' : 'up';
      }
      setEl('ri-usdjpy-nk', corrLabel);
      setEl('ri-usdjpy-nk-sig', corrSig, corrCls);
    }
  } catch {}
}

// Static fallback yield curve data — used only when no live data available
const STATIC_YIELDS = [4.35, 4.28, 4.32, 4.42, 4.58]; // 3M,2Y,5Y,10Y,30Y
const STATIC_LABELS = ['3M','2Y','5Y','10Y','30Y'];
let _lastDrawnYields = null; // {label, val}[] or null
let _lastDrawnPrior  = null; // {label, val}[] from prev_close, or null

function drawYieldCurveAndCache(points, priorPoints) {
  // points can be: {label,val}[] (real data) or null (use static)
  // priorPoints: {label,val}[] from prev_close in quotes.json (optional, overrides PRIOR_MAP)
  _lastDrawnYields = points;
  _lastDrawnPrior  = priorPoints || null;
  drawYieldCurve(points, priorPoints);
}

// ═══════════════════════════════════════════════════════════════════
// YIELD CURVE — canvas drawing, accepts real sparse data points
// ═══════════════════════════════════════════════════════════════════
function drawYieldCurve(points, priorPoints) {
  const canvas = document.getElementById('yield-canvas');
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth - 20, H = 100;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  // Build display data — real points or static fallback
  let labels, vals, isLive;
  if (points && points.length >= 2) {
    labels = points.map(p => p.label);
    vals   = points.map(p => p.val);
    isLive = true;
  } else {
    labels = STATIC_LABELS;
    vals   = STATIC_YIELDS;
    isLive = false;
  }

  // Prior curve reference — use prev_close from quotes.json when available, else PRIOR_MAP
  const PRIOR_MAP = { '3M':4.32,'2Y':4.35,'5Y':4.38,'7Y':4.36,'10Y':4.26,'20Y':4.45,'30Y':4.38,'6M':4.28,'1Y':4.25 };
  let prevVals;
  if (priorPoints && priorPoints.length >= 2) {
    const priorLookup = {};
    priorPoints.forEach(p => { priorLookup[p.label] = p.val; });
    prevVals = labels.map(l => priorLookup[l] ?? PRIOR_MAP[l] ?? null);
  } else {
    prevVals = labels.map(l => PRIOR_MAP[l] ?? null);
  }

  const n = labels.length;
  const validV = vals.filter(v => v != null && !isNaN(v));
  const rawMin = validV.length ? Math.min(...validV, ...prevVals.filter(Boolean)) : 4.10;
  const rawMax = validV.length ? Math.max(...validV, ...prevVals.filter(Boolean)) : 5.40;
  const minY = Math.floor(rawMin * 4) / 4 - 0.15;
  const maxY = Math.ceil(rawMax  * 4) / 4 + 0.15;
  const yRange = maxY - minY || 1;

  const PAD_L=32, PAD_R=8, PAD_T=12, PAD_B=20;
  const cW=W-PAD_L-PAD_R, cH=H-PAD_T-PAD_B;
  const px = i => PAD_L+(i/(n-1))*cW;
  const py = v => PAD_T+(1-(v-minY)/yRange)*cH;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#131722'; ctx.fillRect(0,0,W,H);

  // Grid lines
  const step = yRange <= 0.5 ? 0.1 : yRange <= 1 ? 0.25 : 0.5;
  const gridStart = Math.ceil(minY / step) * step;
  for (let v = gridStart; v <= maxY + 0.001; v = Math.round((v + step) * 1000) / 1000) {
    const y = py(v);
    ctx.strokeStyle='#2a2e39'; ctx.lineWidth=0.5; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(PAD_L,y); ctx.lineTo(W-PAD_R,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#9ba1ae'; ctx.font='bold 8px Courier New'; ctx.textAlign='right';
    ctx.fillText(v.toFixed(2)+'%', PAD_L-3, y+3);
  }

  // Inverted zone — shade between shortest and longest tenor if inverted
  const firstV = vals[0], lastV = vals[n-1];
  if (firstV != null && lastV != null && firstV > lastV) {
    ctx.fillStyle='#ef535012';
    ctx.fillRect(PAD_L, PAD_T, cW, cH);
  }

  // Prior curve
  const priorPts = prevVals.map((v,i) => v != null ? [px(i), py(v)] : null).filter(Boolean);
  if (priorPts.length >= 2) {
    ctx.beginPath(); ctx.strokeStyle='#363c4e'; ctx.lineWidth=1;
    priorPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();
  }

  // Fill under current curve
  const curPts = vals.map((v,i) => v != null ? [px(i), py(v)] : null).filter(Boolean);
  if (curPts.length >= 2) {
    ctx.beginPath();
    curPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.lineTo(curPts[curPts.length-1][0], PAD_T+cH);
    ctx.lineTo(PAD_L, PAD_T+cH);
    ctx.closePath();
    ctx.fillStyle='#2962ff12'; ctx.fill();

    // Current curve line
    ctx.beginPath(); ctx.strokeStyle='#2962ff'; ctx.lineWidth=1.8;
    curPts.forEach(([x,y],i) => i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y));
    ctx.stroke();

    // Dots + value labels at each real point
    vals.forEach((v, i) => {
      if (v == null) return;
      const x = px(i), y = py(v);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2);
      ctx.fillStyle='#2962ff'; ctx.fill();
    });
  }

  // X-axis labels
  ctx.fillStyle='#9ba1ae'; ctx.font='bold 8.5px Courier New'; ctx.textAlign='center';
  labels.forEach((t,i) => ctx.fillText(t, px(i), H-5));

  // Legend
  ctx.textAlign='left';
  ctx.fillStyle='#4d8ffa'; ctx.fillText('● Current', PAD_L, PAD_T-2);
  ctx.fillStyle='#6b7280'; ctx.fillText('● Prior',   PAD_L+52, PAD_T-2);
  if (!isLive) {
    ctx.fillStyle='#6b7280'; ctx.fillText('(static)', PAD_L+92, PAD_T-2);
  } else {
    // Check inversion
    const spr = (vals[n-1] ?? 0) - (vals[0] ?? 0); // long - short
    if (spr < 0) { ctx.fillStyle='#ef535099'; ctx.fillText('■ Inverted', PAD_L+92, PAD_T-2); }
  }

  // Update yield spread table using real 2Y and 10Y
  const real2y  = vals[labels.indexOf('2Y')];
  const real3m  = vals[labels.indexOf('3M')];
  const real10y = vals[labels.indexOf('10Y')];
  const shortKey = real2y ?? real3m;
  if (shortKey != null && real10y != null) {
    const spread = real10y - shortKey;
    const bp = Math.round(spread * 100);
    const sign = bp >= 0 ? '+' : '';
    setEl('ys-2-10', sign + bp + 'bp', spread < 0 ? 'down' : 'up');
    setEl('ys-2-10-sig', spread < 0 ? 'Inverted' : 'Normal', spread < 0 ? 'down' : 'up');
  }
}

setTimeout(() => drawYieldCurveAndCache(null), 60);
window.addEventListener('resize', () => drawYieldCurve(_lastDrawnYields, _lastDrawnPrior));

// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// TV CHART TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════
document.querySelectorAll('.tv-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.tv-tab').forEach(t => t.classList.remove('active'));
    this.classList.add('active');
    const sym = this.dataset.sym;
    const wrap = document.getElementById('tv-chart-wrap');
    wrap.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'tradingview-widget-container';
    container.style.cssText = 'height:100%;width:100%;';
    const widget = document.createElement('div');
    widget.className = 'tradingview-widget-container__widget';
    widget.style.cssText = 'height:100%;width:100%;';
    container.appendChild(widget);
    const copyright = document.createElement('div');
    copyright.className = 'tradingview-widget-copyright';
    copyright.style.display = 'none';
    container.appendChild(copyright);
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.text = JSON.stringify({
      allow_symbol_change:false, calendar:false, details:true,
      hide_side_toolbar:true, hide_top_toolbar:true, hide_legend:false,
      hide_volume:true, interval:'D', locale:'en', save_image:false,
      style:'1', symbol:sym, theme:'dark', timezone:'Etc/UTC',
      backgroundColor:'#131722', gridColor:'rgba(42,46,57,0.8)',
      withdateranges:false, studies:[{id:'MASimple@tv-basicstudies',inputs:{length:20}}], autosize:true
    });
    container.appendChild(script);
    wrap.appendChild(container);
    // Re-trigger legend minimize after symbol change
    setTimeout(minimizeTVLegend, 3000);
  });
});

// ── TradingView legend auto-minimize (MA 20, Close, Vol labels) ──────────
// The widget renders inside an iframe so we poll for the minimize buttons
// and click them. Runs on initial load and on each symbol change.
function minimizeTVLegend() {
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  const iframe = wrap.querySelector('iframe');
  if (!iframe) return;
  // Try accessing the iframe document (same-origin if TV embeds it same-domain, else blocked)
  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    if (!doc) return;
    // Click all legend item minimize/collapse buttons (aria-label or title contains "Minimize")
    const btns = doc.querySelectorAll(
      '[data-name="legend-source-item"] button[aria-label], ' +
      '.legendItemControls button, ' +
      '[class*="minimizeButton"], ' +
      '[class*="collapseButton"]'
    );
    btns.forEach(btn => { try { btn.click(); } catch(_){} });
  } catch(_) {
    // Cross-origin — can't access iframe internals, nothing we can do
  }
}
// Run once after initial widget loads (give it ~4s to render)
setTimeout(minimizeTVLegend, 4000);
// ─────────────────────────────────────────────────────────────────────────

// ── HORIZONTAL SCROLL WITH MOUSE WHEEL (desktop) ─────────────────────────
// Converts vertical wheel events into horizontal scroll on designated bars
(function() {
  function addWheelScroll(el) {
    if (!el) return;
    el.addEventListener('wheel', function(e) {
      if (e.deltaY === 0) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    }, { passive: false });
  }
  addWheelScroll(document.getElementById('tv-pair-tabs'));
  addWheelScroll(document.getElementById('tv-ticker'));
  addWheelScroll(document.getElementById('quotebar-inner'));

  // Arrow visibility for tv-pair-tabs
  const tabs   = document.getElementById('tv-pair-tabs');
  const btnPrev = document.getElementById('tv-tabs-prev');
  const btnNext = document.getElementById('tv-tabs-next');
  function updateTabArrows() {
    if (!tabs || !btnPrev || !btnNext) return;
    const atStart = tabs.scrollLeft <= 2;
    const atEnd   = tabs.scrollLeft + tabs.clientWidth >= tabs.scrollWidth - 2;
    btnPrev.style.display = atStart ? 'none' : 'flex';
    btnNext.style.display = atEnd   ? 'none' : 'flex';
  }
  if (tabs) {
    tabs.addEventListener('scroll', updateTabArrows, { passive: true });
    setTimeout(updateTabArrows, 200);
  }

  // Arrow visibility for quote bar
  const ticker   = document.getElementById('tv-ticker');
  const qbPrev   = document.getElementById('qb-prev');
  const qbNext   = document.getElementById('qb-next');
  function updateQbArrows() {
    if (!ticker || !qbPrev || !qbNext) return;
    const atStart = ticker.scrollLeft <= 2;
    const atEnd   = ticker.scrollLeft + ticker.clientWidth >= ticker.scrollWidth - 2;
    qbPrev.style.display = atStart ? 'none' : 'flex';
    qbNext.style.display = atEnd   ? 'none' : 'flex';
  }
  if (ticker) {
    ticker.addEventListener('scroll', updateQbArrows, { passive: true });
    setTimeout(updateQbArrows, 400);
  }
})();
// ─────────────────────────────────────────────────────────────────────────

// TOP NAV
document.querySelectorAll('.top-nav a').forEach(a => {
  a.addEventListener('click', function() {
    document.querySelectorAll('.top-nav a').forEach(x => x.classList.remove('active'));
    this.classList.add('active');
  });
});

// ═══════════════════════════════════════════════════════════════════
// CARRY TRADE SIDEBAR — from rates/*.json + extended-data/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchCarryData() {
  const CURRENCIES = ['USD','EUR','GBP','JPY','AUD','CHF','CAD','NZD'];
  const LABELS = { USD:'USD Fed', EUR:'EUR ECB', GBP:'GBP BoE', JPY:'JPY BoJ',
                   AUD:'AUD RBA', CHF:'CHF SNB', CAD:'CAD BoC', NZD:'NZD RBNZ' };

  try {
    // Fetch rates from repo
    const rateData = {};
    await Promise.all(CURRENCIES.map(async ccy => {
      try {
        const r = await fetch('./rates/' + ccy + '.json');
        if (!r.ok) return;
        const d = await r.json();
        if (d.observations && d.observations.length) {
          rateData[ccy] = parseFloat(d.observations[0].value);
        }
      } catch {}
    }));

    // Build carry pairs: long high-yield, short low-yield
    const carryPairs = [
      { long: 'AUD', short: 'JPY' },
      { long: 'NZD', short: 'JPY' },
      { long: 'GBP', short: 'JPY' },
      { long: 'AUD', short: 'CHF' },
      { long: 'NZD', short: 'CHF' },
      { long: 'USD', short: 'JPY' },
    ].map(p => {
      const diff = (rateData[p.long] ?? 0) - (rateData[p.short] ?? 0);
      return { ...p, diff };
    }).sort((a,b) => b.diff - a.diff);

    const container = document.getElementById('sb-carry-rows');
    if (!container) return;
    container.innerHTML = carryPairs.map(p => {
      const sign = p.diff >= 0 ? '+' : '';
      const cls = p.diff > 1 ? 'up' : p.diff < 0 ? 'down' : 'flat';
      const longRate = (rateData[p.long]??0).toFixed(2);
      const shortRate = (rateData[p.short]??0).toFixed(2);
      return `<div class="sb-row">
        <span class="sb-sym">${p.long}/${p.short}</span>
        <span class="sb-price" style="font-size:8.5px;color:var(--text3);letter-spacing:-0.01em;">${longRate}% · ${shortRate}%</span>
        <span class="sb-chg ${cls}">${sign}${p.diff.toFixed(2)}%</span>
      </div>`;
    }).join('');
  } catch(e) { console.warn('Carry fetch failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// CROSS-ASSET — custom grid from Stooq/yfinance + repo extended-data
// ═══════════════════════════════════════════════════════════════════
async function fetchCrossAssetData() {
  // stooq() helper removed — yfinance JSON used exclusively

  function setCA(id, val, chgPct, isYield) {
    const vEl = document.getElementById('ca-' + id);
    const cEl = document.getElementById('cac-' + id);
    if (!vEl || !cEl) return;
    if (val == null) return;
    if (chgPct == null) {
      vEl.textContent = isYield ? val.toFixed(2) + '%' : val.toLocaleString(undefined, { maximumFractionDigits: val > 100 ? 2 : 4 });
      vEl.className = 'ca-val flat';
      cEl.textContent = '—'; cEl.className = 'ca-chg flat';
      return;
    }
    const cls = chgPct > 0.05 ? 'up' : chgPct < -0.05 ? 'down' : '';
    const sign = chgPct >= 0 ? '+' : '';
    vEl.textContent = isYield ? val.toFixed(2) + '%' : val.toLocaleString(undefined, { maximumFractionDigits: val > 100 ? 2 : 4 });
    vEl.className = 'ca-val ' + cls;
    cEl.textContent = sign + chgPct.toFixed(2) + '%';
    cEl.className = 'ca-chg ' + cls;
  }

  // ── STEP 1: Pre-load repo data (same-origin, instant) so US10Y is available immediately ──
  let _repoUs10y = null;
  try {
    const usdExt = await fetch('./extended-data/USD.json').then(r => r.ok ? r.json() : null).catch(() => null);
    if (usdExt?.data?.bond10y != null && !isNaN(usdExt.data.bond10y)) {
      _repoUs10y = { close: usdExt.data.bond10y, chg: 0, pct: 0, fromRepo: true };
      // Render US10Y immediately so cross-asset table isn't blank while data loads
      setCA('us10y', _repoUs10y.close, null, true);
    }
  } catch {}

  // ── STEP 1.5: Intraday quotes from GitHub Action (yfinance) ──
  // Pre-populate all cross-asset cells. yfinance JSON is the sole real-time source.
  const _caIntraday = await loadIntradayQuotes();  // uses cache — no extra network call if already loaded
  let _caGold   = _caIntraday ? intradayQuote(_caIntraday, 'gold')   : null;
  let _caWti    = _caIntraday ? intradayQuote(_caIntraday, 'wti')    : null;
  let _caSpx    = _caIntraday ? intradayQuote(_caIntraday, 'spx')    : null;
  let _caNikkei = _caIntraday ? intradayQuote(_caIntraday, 'nikkei') : null;
  let _caStoxx  = _caIntraday ? intradayQuote(_caIntraday, 'stoxx')  : null;
  let _caDxy    = _caIntraday ? intradayQuote(_caIntraday, 'dxy')    : null;

  // Render inmediato con JSON intraday — el usuario ve valores en <100ms.
  if (_caSpx)    setCA('spx',    _caSpx.close,    _caSpx.pct,    false);
  if (_caGold) {
    setCA('gold', _caGold.close, _caGold.pct, false);
    const gEl = document.getElementById('q-xauusd'), gcEl = document.getElementById('qc-xauusd');
    if (gEl)  { gEl.textContent  = _caGold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); gEl.className  = 'q-price ' + clsDir(_caGold.chg); }
    if (gcEl) { gcEl.textContent = pctStr(_caGold.pct); gcEl.className = 'q-chg '   + clsDir(_caGold.chg); }
  }
  if (_caWti)    setCA('wti',    _caWti.close,    _caWti.pct,    false);
  if (_caNikkei) setCA('nikkei', _caNikkei.close, _caNikkei.pct, false);
  if (_caStoxx)  setCA('stoxx',  _caStoxx.close,  _caStoxx.pct,  false);
  // US10Y desde intraday JSON — sobreescribe el valor de repo (que puede tener 1 día de delay)
  const _caUs10yEarly = _caIntraday ? intradayQuote(_caIntraday, 'us10y') : null;
  if (_caUs10yEarly && _caUs10yEarly.close > 0) setCA('us10y', _caUs10yEarly.close, _caUs10yEarly.pct, true);
  // Gold/SPX ratio — calculado apenas tenemos ambos valores del JSON
  if (_caGold && _caSpx && _caSpx.close > 0) {
    const ratio = (_caGold.close / _caSpx.close).toFixed(3);
    const rNum  = parseFloat(ratio);
    const sig   = rNum > 0.75 ? 'Risk-Off signal' : rNum > 0.55 ? 'Neutral' : 'Risk-On signal';
    const cls   = rNum > 0.75 ? 'down' : rNum < 0.55 ? 'up' : 'flat';
    setEl('ri-gold-spx', ratio);
    setEl('ri-gold-spx-sig', sig, cls);
  }
  if (_caDxy) {
    setCA('dxy', _caDxy.close, _caDxy.pct, false);
    const dEl = document.getElementById('q-dxy'), dcEl = document.getElementById('qc-dxy');
    if (dEl)  { dEl.textContent  = _caDxy.close.toFixed(1); dEl.className  = 'q-price ' + clsDir(_caDxy.chg); }
    if (dcEl) { dcEl.textContent = pctStr(_caDxy.pct);      dcEl.className = 'q-chg '   + clsDir(_caDxy.chg); }
  }
  // BTC inmediato desde JSON
  const _caBtcEarly = _caIntraday ? intradayQuote(_caIntraday, 'btc') : null;
  if (_caBtcEarly) {
    const btcFmtE = _caBtcEarly.close.toLocaleString(undefined, {minimumFractionDigits:0, maximumFractionDigits:0});
    const bEl = document.getElementById('ca-btc'), bcEl = document.getElementById('cac-btc');
    const qbEl = document.getElementById('q-btcusd'), qbcEl = document.getElementById('qc-btcusd');
    if (bEl)  { bEl.textContent  = btcFmtE; bEl.className  = 'ca-val '  + clsDir(_caBtcEarly.chg); }
    if (bcEl) { bcEl.textContent = pctStr(_caBtcEarly.pct); bcEl.className = 'ca-chg ' + clsDir(_caBtcEarly.chg); }
    if (qbEl && qbEl.textContent === '—')  { qbEl.textContent  = btcFmtE; qbEl.className  = 'q-price ' + clsDir(_caBtcEarly.chg); }
    if (qbcEl && qbcEl.textContent === '—') { qbcEl.textContent = pctStr(_caBtcEarly.pct); qbcEl.className = 'q-chg ' + clsDir(_caBtcEarly.chg); }
  }

  // ── STEP 2: All cross-asset data from intraday quotes.json (yfinance) ──
  // Stooq and Yahoo removed — both blocked by CORS in production.
  // quotes.json (same-origin, ~15min delay) covers all symbols.
  const finalSpx    = _caSpx;
  const finalGold   = _caGold;
  const finalWti    = _caWti;
  const finalNikkei = _caNikkei;
  const finalStoxx  = _caStoxx;
  const finalDxy    = _caDxy;
  const us10y       = (_caIntraday ? intradayQuote(_caIntraday, 'us10y') : null) || _repoUs10y;

  if (finalSpx)    setCA('spx',    finalSpx.close,    finalSpx.pct,    false);
  if (finalGold) {
    setCA('gold', finalGold.close, finalGold.pct, false);
    const gEl = document.getElementById('q-xauusd'), gcEl = document.getElementById('qc-xauusd');
    if (gEl)  { gEl.textContent  = finalGold.close.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); gEl.className  = 'q-price ' + clsDir(finalGold.chg); }
    if (gcEl) { gcEl.textContent = pctStr(finalGold.pct); gcEl.className = 'q-chg ' + clsDir(finalGold.chg); }
  }
  if (finalWti)    setCA('wti',    finalWti.close,    finalWti.pct,    false);
  if (finalNikkei) setCA('nikkei', finalNikkei.close, finalNikkei.pct, false);
  if (finalStoxx)  setCA('stoxx',  finalStoxx.close,  finalStoxx.pct,  false);
  if (us10y)       setCA('us10y',  us10y.close, us10y.fromRepo ? null : us10y.pct, true);

  const dxyData = finalDxy;
  if (dxyData) {
    setCA('dxy', dxyData.close, dxyData.pct, false);
    const dEl = document.getElementById('q-dxy');
    const dcEl = document.getElementById('qc-dxy');
    if (dEl) { dEl.textContent = dxyData.close.toFixed(1); dEl.className = 'q-price ' + clsDir(dxyData.chg); }
    if (dcEl) { dcEl.textContent = pctStr(dxyData.pct); dcEl.className = 'q-chg ' + clsDir(dxyData.chg); }
  }

  // BTC — intraday JSON (yfinance BTC-USD) primary, CoinGecko topbar cache fallback
  const btcEl = document.getElementById('ca-btc');
  const btcCEl = document.getElementById('cac-btc');
  const qBtc = document.getElementById('q-btcusd');
  const qBtcC = document.getElementById('qc-btcusd');
  const _btcIntraday = _caIntraday ? intradayQuote(_caIntraday, 'btc') : null;
  if (_btcIntraday && btcEl) {
    const btcFmt = _btcIntraday.close.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
    btcEl.textContent  = btcFmt;
    btcEl.className    = 'ca-val ' + clsDir(_btcIntraday.chg);
    if (btcCEl) { btcCEl.textContent = pctStr(_btcIntraday.pct); btcCEl.className = 'ca-chg ' + clsDir(_btcIntraday.chg); }
    // Also update topbar q-btcusd if still showing —
    if (qBtc && qBtc.textContent === '—') {
      qBtc.textContent  = btcFmt;
      qBtc.className    = 'q-price ' + clsDir(_btcIntraday.chg);
      if (qBtcC) { qBtcC.textContent = pctStr(_btcIntraday.pct); qBtcC.className = 'q-chg ' + clsDir(_btcIntraday.chg); }
    }
  } else if (btcEl && qBtc && qBtc.textContent !== '—') {
    btcEl.textContent = qBtc.textContent;
    btcEl.className = qBtc.className.replace('q-price', 'ca-val');
    if (btcCEl && qBtcC) { btcCEl.textContent = qBtcC.textContent; btcCEl.className = qBtcC.className.replace('q-chg', 'ca-chg'); }
  }

  const upd = document.getElementById('ca-updated');
  const now = new Date();
  const localHHMM = now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
  const tzAbbr = now.toLocaleTimeString('en', {timeZoneName:'short'}).split(' ').pop() || 'LT';
  // Show the actual source — intraday JSON (yfinance) when fresh, Stooq otherwise
  let sourceLabel = 'yfinance';
  if (_caIntraday?.source && _caIntraday.source !== 'repo') {
    const srcName = _caIntraday.source === 'yfinance'      ? 'yfinance'
                  : _caIntraday.source === 'twelve_data'   ? 'Twelve Data'
                  : _caIntraday.source === 'alpha_vantage' ? 'Alpha Vantage'
                  : 'mixed APIs';
    // Check if the file is fresh (under 20 min old)
    const fileAge = _caIntraday.updated
      ? (Date.now() - new Date(_caIntraday.updated).getTime()) / 60000
      : 999;
    sourceLabel = fileAge < 20 ? `${srcName} · ~15min delay` : 'yfinance';
  }
  if (upd) upd.textContent = sourceLabel + ' · ' + localHHMM + ' ' + tzAbbr;

  // Gold / SPX ratio — computed here where gold & spx are in scope
  if (finalGold && finalSpx && finalSpx.close > 0) {
    const ratio = (finalGold.close / finalSpx.close).toFixed(3);
    const rNum  = parseFloat(ratio);
    const sig   = rNum > 0.75 ? 'Risk-Off signal' : rNum > 0.55 ? 'Neutral' : 'Risk-On signal';
    const cls   = rNum > 0.75 ? 'down' : rNum < 0.55 ? 'up' : 'flat';
    setEl('ri-gold-spx', ratio);
    setEl('ri-gold-spx-sig', sig, cls);
  }
}

// ═══════════════════════════════════════════════════════════════════
// BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// FED RATE EXPECTATIONS — computed from meetings-data if available,
// otherwise from CB rate trajectory in rates/USD.json
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// CB RATE EXPECTATIONS — todos los bancos centrales
// Usa meetings-data/meetings.json + rates/*.json
// ═══════════════════════════════════════════════════════════════════
async function fetchFedExpectations() {
  try {
    const tbody = document.getElementById('fed-exp-tbody');
    if (!tbody) return;

    // Load meetings and all rates in parallel
    const [meetingsRes, ...rateResponses] = await Promise.all([
      fetch('./meetings-data/meetings.json').then(r => r.ok ? r.json() : null).catch(() => null),
      ...['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'].map(c =>
        fetch(`./rates/${c}.json`).then(r => r.ok ? r.json() : null).catch(() => null)
      )
    ]);

    const currencies = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
    const bankMeta = {
      USD: { flag:'us', short:'Fed',  trendKey:'usd' },
      EUR: { flag:'eu', short:'ECB',  trendKey:'eur' },
      GBP: { flag:'gb', short:'BoE',  trendKey:'gbp' },
      JPY: { flag:'jp', short:'BoJ',  trendKey:'jpy' },
      AUD: { flag:'au', short:'RBA',  trendKey:'aud' },
      CAD: { flag:'ca', short:'BoC',  trendKey:'cad' },
      CHF: { flag:'ch', short:'SNB',  trendKey:'chf' },
      NZD: { flag:'nz', short:'RBNZ', trendKey:'nzd' },
    };

    const rows = [];
    currencies.forEach((ccy, i) => {
      const rateData = rateResponses[i];
      if (!rateData) return;
      const obs = rateData.observations || [];
      if (obs.length < 2) return;

      const current = parseFloat(obs[0].value);
      const prev    = parseFloat(obs[Math.min(2, obs.length - 1)].value);
      const trend   = current - prev; // negative = cutting, positive = hiking

      const meetings = meetingsRes?.meetings?.[ccy];
      const nextMtg  = meetings?.allMeetingsFormatted?.[0] || '—';

      // Bias from trend
      const trendDir = CB_TREND[bankMeta[ccy].trendKey] || 'flat';
      const biasLabel = trendDir === 'down' ? '<span class="up">↓ Cut</span>'
                      : trendDir === 'up'   ? '<span class="down">↑ Hike</span>'
                      :                       '<span class="flat">→ Hold</span>';

      // Forward rate estimate: next step based on trend
      const step = Math.abs(trend) > 0.1 ? (trend < 0 ? -0.25 : 0.25) : 0;
      const fwd  = Math.max(0, current + step);

      const meta = bankMeta[ccy];
      const flag = `<span class="fi fi-${meta.flag}" style="margin-right:4px;border-radius:2px;vertical-align:middle;"></span>`;

      rows.push(`<tr title="Next meeting: ${nextMtg}">
        <td style="white-space:nowrap;">${flag}<span style="font-size:10px;">${meta.short}</span> <span style="color:var(--text3);font-size:9px;">${nextMtg}</span></td>
        <td>${biasLabel}</td>
        <td style="color:var(--text2);font-family:var(--font-mono);font-size:10px;">${fwd.toFixed(2)}%</td>
      </tr>`);
    });

    if (rows.length) tbody.innerHTML = rows.join('');
  } catch(e) { console.warn('CB expectations failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// OPTION SKEW — derived from COT + FX move data (approximated)
// Shows directional bias from positioning & price action
// ═══════════════════════════════════════════════════════════════════
async function fetchOptionSkew() {
  try {
    const tbody = document.getElementById('skew-tbody');
    if (!tbody) return;

    // Use COT net positions as a proxy for skew bias
    // Long bias → puts bid (downside protection) → negative skew for USD pairs
    const cotFiles = ['EUR','GBP','JPY','AUD','CAD'];
    const cotResults = await Promise.all(cotFiles.map(async ccy => {
      try {
        const r = await fetch('./cot-data/' + ccy + '.json');
        if (!r.ok) return null;
        const d = await r.json();
        return { ccy, net: d.netPosition || 0, long: d.longPositions||0, short: d.shortPositions||0 };
      } catch { return null; }
    }));

    const pairs = [
      { pair:'EUR/USD', cot:'EUR' },
      { pair:'GBP/USD', cot:'GBP' },
      { pair:'USD/JPY', cot:'JPY' },
      { pair:'AUD/USD', cot:'AUD' },
    ];

    // Map COT net → approximate RR skew
    // Positive net (more longs) for EUR → EUR calls bid → positive 25d RR
    function netToSkew(net, invert) {
      const scale = Math.abs(net) / 50000; // normalize to ±1
      const val = Math.min(1.5, scale * 1.2);
      const signed = net > 0 ? val : -val;
      return invert ? -signed : signed;
    }

    const cotMap = {};
    cotResults.filter(Boolean).forEach(c => { cotMap[c.ccy] = c; });

    tbody.innerHTML = pairs.map(p => {
      const cotData = cotMap[p.cot];
      if (!cotData) return `<tr><td>${p.pair}</td><td colspan="3" style="color:var(--text3)">—</td></tr>`;
      const invert = p.pair.startsWith('USD/');
      const skew1w = netToSkew(cotData.net, invert);
      const skew1m = netToSkew(cotData.net * 0.85, invert);
      const bias   = Math.abs(skew1w) < 0.1 ? 'Neutral' : skew1w > 0 ? p.pair.split('/')[0]+'+' : p.pair.split('/')[1]+'+';
      const biasCls= Math.abs(skew1w) < 0.1 ? 'flat' : skew1w > 0 ? 'up' : 'down';
      const fmtRR = v => (v >= 0 ? '+' : '') + v.toFixed(2);
      return `<tr>
        <td>${p.pair}</td>
        <td class="${skew1w >= 0 ? 'up':'down'}">${fmtRR(skew1w)}</td>
        <td class="${skew1m >= 0 ? 'up':'down'}">${fmtRR(skew1m)}</td>
        <td class="${biasCls}">${bias}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.warn('Option skew failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// LOAD AI REGIME — fast-path: set narrative badge BEFORE fetchRiskData runs.
// Called with await in boot() so _aiRegimeFresh is populated and the badge
// shows the correct regime on first paint, with zero flicker.
// buildRichNarrative() runs afterwards to fill in the full narrative text.
// ═══════════════════════════════════════════════════════════════════
async function loadAIRegime() {
  try {
    const res = await fetch('./ai-analysis/index.json');
    if (!res.ok) return;
    const d = await res.json();
    let regime = (d.regime || 'RISK-OFF').toUpperCase();
    const generatedAt = d.generated_at || null;

    // Staleness check — same 4-hour threshold as buildRichNarrative
    let isStale = false;
    if (generatedAt) {
      const ageMinutes = (Date.now() - new Date(generatedAt).getTime()) / 60000;
      if (ageMinutes > 240) isStale = true;
    }

    const regEl = document.getElementById('narrative-regime');
    if (!regEl) return;

    if (isStale) {
      _aiRegimeFresh = false;
      regEl.textContent = regime;
      regEl.className = 'narr-regime stale';
      regEl.title = 'AI signal stale — live VIX assessment active';
    } else {
      _aiRegimeFresh = true;
      const isOn  = regime === 'RISK-ON';
      const isOff = regime === 'RISK-OFF';
      const badgeColor = isOn ? 'var(--up)' : isOff ? 'var(--down)' : 'var(--orange)';
      regEl.textContent = regime;
      regEl.className = 'narr-regime';
      regEl.style.borderColor = badgeColor;
      regEl.style.color       = badgeColor;
      regEl.title = '';

      // Also prime the Risk Monitor badge so both panels are in sync from first paint
      const riskCls = isOn ? 'risk-val up' : isOff ? 'risk-val down' : 'risk-val warning';
      setEl('risk-regime', regime, riskCls);
      setEl('risk-regime-sub', isOn ? 'Risk appetite active' : isOff ? 'Risk aversion elevated' : 'Elevated caution');
    }
  } catch { /* silently skip — renderRiskData will handle the badge via live data */ }
}

// RICH AI NARRATIVE — build from ai-analysis/index.json + live data
// ═══════════════════════════════════════════════════════════════════
async function buildRichNarrative() {
  try {
    // Fetch AI narrative base
    const [narRes, newsRes] = await Promise.all([
      fetch('./ai-analysis/index.json'),
      fetch('./news-data/news.json'),
    ]);

    let baseNarrative = '';
    let regime = 'RISK-OFF';
    let _narrativeGeneratedAt = null; // kept in outer scope for stale age display

    if (narRes.ok) {
      const d = await narRes.json();
      baseNarrative = d.narrative || '';
      regime = d.regime || 'RISK-OFF';
      _narrativeGeneratedAt = d.generated_at || null;

      // Staleness check — if the AI JSON is older than 4 hours, mark regime badge as stale
      // so users know it may not reflect current market conditions
      if (_narrativeGeneratedAt) {
        const ageMinutes = (Date.now() - new Date(_narrativeGeneratedAt).getTime()) / 60000;
        if (ageMinutes > 240) {
          regime = '__STALE__' + regime;
        }
      }
    }

    // Pull key headlines from news to enrich narrative
    let newsContext = [];
    if (newsRes.ok) {
      const nd = await newsRes.json();
      const articles = nd.articles || [];
      // Get top 6 featured/recent high-impact items
      newsContext = articles
        .filter(a => a.impact === 'high' && (!a.lang || a.lang === 'en'))
        .slice(0, 6);
    }

    // Build contextual currency mentions from news
    const curMentions = {};
    newsContext.forEach(a => {
      if (a.cur) curMentions[a.cur] = (curMentions[a.cur] || 0) + 1;
    });
    const topCur = Object.entries(curMentions).sort((a,b) => b[1]-a[1]).map(e=>e[0]).slice(0,3);

    // Build FX context from Frankfurter rates if available
    const fxLines = [];
    const r = STATE.rates;
    const p = STATE.prevRates;
    if (r && Object.keys(r).length) {
      // EUR/USD
      if (r.EUR && p.EUR) {
        const eurusd = 1/r.EUR, prevEurusd = 1/p.EUR;
        const pct = (eurusd - prevEurusd)/prevEurusd*100;
        if (Math.abs(pct) > 0.05)
          fxLines.push(`EUR/USD ${pct>0?'bid':'offered'} at ${eurusd.toFixed(4)} (${pct>=0?'+':''}${pct.toFixed(2)}%)`);
      }
      // USD/JPY
      if (r.JPY && p.JPY) {
        const usdjpy = r.JPY, prevJpy = p.JPY;
        const pct = (usdjpy - prevJpy)/prevJpy*100;
        if (Math.abs(pct) > 0.05)
          fxLines.push(`USD/JPY ${pct>0?'extends gains':'retreats'} to ${usdjpy.toFixed(2)}`);
      }
      // DXY proxy (USD strength via basket)
      const majors = ['EUR','GBP','AUD','NZD'];
      const avgPct = majors.filter(c=>r[c]&&p[c]).map(c=>(r[c]-p[c])/p[c]*100);
      if (avgPct.length) {
        const usdAvg = -(avgPct.reduce((a,b)=>a+b,0)/avgPct.length);
        if (Math.abs(usdAvg) > 0.03)
          fxLines.push(`DXY ${usdAvg>0?'firming':'weakening'} — USD ${usdAvg>0?'broadly bid':'broadly offered'}`);
      }
    }

    // Pick headline from top news item — title only, never expand (expand is article body)
    let headlineSnippet = '';
    if (newsContext.length) {
      const topItem = newsContext[0];
      const title = (topItem.title || '').replace(/\s+/g,' ').trim().slice(0,100);
      if (title.length > 20) headlineSnippet = title + (title.length === 100 ? '…' : '');
    }

    // Compose final narrative — show Groq narrative as primary, enrich with live rates if fresh
    let finalNarrative = '';

    if (baseNarrative.length > 40) {
      // Use Groq narrative as primary (engine now sends real price levels)
      finalNarrative = baseNarrative;
      // Append live rate context only if it adds something not already in the narrative
      if (fxLines.length) {
        const enrichments = fxLines.filter(l => {
          // Only add lines whose pair/direction is NOT already mentioned in the narrative
          const pair = l.split(' ')[0]; // e.g. "EUR/USD"
          return !baseNarrative.includes(pair);
        });
        if (enrichments.length) {
          finalNarrative += ' · ' + enrichments.join(' · ');
        }
      }
    } else if (fxLines.length || headlineSnippet) {
      // No Groq narrative available — build from live data
      const parts = [];
      if (fxLines.length) parts.push(fxLines.join('. '));
      if (topCur.length && topCur.length <= 4) parts.push(`${topCur.join(', ')} in focus`);
      if (headlineSnippet) parts.push(headlineSnippet);
      finalNarrative = parts.join('. ') + '.';
    }

    // Update DOM
    const el = document.getElementById('narrative-text');
    if (el && finalNarrative) el.textContent = finalNarrative;

    // Update regime badge
    const regEl = document.getElementById('narrative-regime');
    if (regEl && regime) {
      const isStale = regime.startsWith('__STALE__');
      const cleanRegime = isStale ? regime.replace('__STALE__', '') : regime;
      const isOn = cleanRegime.toLowerCase().includes('on') && !cleanRegime.toLowerCase().includes('off');

      if (isStale) {
        // Stale AI regime — show the badge greyed-out with age indicator.
        // fetchRiskData() drives the authoritative live assessment.
        _aiRegimeFresh = false;
        const ageHours = _narrativeGeneratedAt
          ? Math.round((Date.now() - new Date(_narrativeGeneratedAt).getTime()) / 3600000)
          : null;
        const ageLabel = ageHours != null ? (ageHours >= 1 ? `${ageHours}h ago` : 'stale') : 'stale';
        regEl.textContent = cleanRegime.toUpperCase();
        regEl.className = 'narr-regime stale';
        regEl.title = `AI signal from ${ageLabel} — live VIX assessment shown in Risk Monitor`;
        const riskRegEl = document.getElementById('risk-regime');
        if (riskRegEl) riskRegEl.style.opacity = '0.5';
        setEl('risk-regime-sub', `AI signal from ${ageLabel} · live assessment active`);
      } else {
        // Fresh AI regime — set narrative badge and mark as authoritative
        _aiRegimeFresh = true;
        const isOff = cleanRegime.toUpperCase().includes('OFF');
        const isCaution = !isOn && !isOff; // CAUTION or MIXED
        const badgeColor = isOn ? 'var(--up)' : isCaution ? 'var(--orange)' : 'var(--down)';
        regEl.textContent = cleanRegime.toUpperCase();
        regEl.className = 'narr-regime';
        regEl.style.borderColor = badgeColor;
        regEl.style.color       = badgeColor;
        regEl.title = '';
        const riskRegEl = document.getElementById('risk-regime');
        if (riskRegEl) riskRegEl.style.opacity = '';
        const riskCls = isOn ? 'risk-val up' : isCaution ? 'risk-val warning' : 'risk-val down';
        setEl('risk-regime', cleanRegime.toUpperCase(), riskCls);
        setEl('risk-regime-sub', isOn ? 'Risk appetite active' : isCaution ? 'Elevated caution' : 'Risk aversion elevated');
      }
    }

    // Also load signals (moved here from fetchAIData to keep AI logic together)
    try {
      const sigR = await fetch('./ai-analysis/signals.json');
      if (sigR.ok) {
        const signals = await sigR.json();
        if (Array.isArray(signals) && signals.length) {
          const container = document.getElementById('alerts-container');
          const sub = document.getElementById('alerts-sub');
          if (container) {
            // Convert engine UTC time string "HH:MM" to user's local timezone
            function localizeSignalTime(timeStr) {
              if (!timeStr || timeStr === '--:--') return timeStr || '--:--';
              try {
                const [h, m] = timeStr.split(':').map(Number);
                if (isNaN(h) || isNaN(m)) return timeStr;
                const now = new Date();
                const utcDate = new Date(Date.UTC(
                  now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), h, m
                ));
                return utcDate.toLocaleTimeString(navigator.language || 'en', {
                  hour: '2-digit', minute: '2-digit', hour12: false
                });
              } catch { return timeStr; }
            }
            container.innerHTML = signals.map(s => {
              const dotCls = s.priority === 'critical' ? 'a-crit' : s.priority === 'warning' ? 'a-warn' : 'a-info';
              const localTime = localizeSignalTime(s.time);
              return `<div class="alert-row">
                <span class="a-time">${localTime}</span>
                <span class="a-dot ${dotCls}"></span>
                <div class="a-text"><strong>${s.title || ''}</strong>${s.title ? ' — ' : ''}${s.text || ''}</div>
              </div>`;
            }).join('');
          }
          if (sub) sub.textContent = signals.length + ' active · sorted by priority · AI-generated';
        }
      }
    } catch {}

  } catch(e) { console.warn('Narrative build failed:', e); }
}

// ═══════════════════════════════════════════════════════════════════
// BOOT SEQUENCE
// ═══════════════════════════════════════════════════════════════════
async function boot() {
  // PHASE 1: Critical path — FX pairs table needs Frankfurter first
  await fetchFrankfurter();          // ECB rates → populates STATE.rates for FX table

  // PHASE 2: Parallel — all remaining data loads simultaneously

  // Pre-cargar intraday JSON ahora (same-origin, ~0ms) para que fetchRiskData
  // y fetchCrossAssetData lo encuentren en caché cuando lo necesiten.
  // await garantiza que el JSON esté listo ANTES de que fetchRiskData/fetchCrossAssetData
  // lo soliciten — evita que cada función haga su propio fetch en paralelo y compitan.
  await loadIntradayQuotes();

  // fetchQuoteBarRT popula STOOQ_RT_CACHE (precios RT + hv30).
  // Se awaita para que populateFxPairsTable encuentre el cache listo al renderizar.
  await fetchQuoteBarRT();
  loadFxPerfData().then(() => populateFxPairsTable()); // 1W perf data, re-render when ready
  populateCorrelations(); // 60-day rolling correlations from quotes.json

  // Static repo data — all parallel, fast (same GitHub Pages origin)
  fetchCBRates();
  fetchCOTData();
  fetchFedExpectations();
  fetchOptionSkew();
  fetchCarryData();
  fetchNewsData();

  // ── CRITICAL: Load AI regime badge FIRST, before fetchRiskData touches the narrative badge.
  // loadAIRegime() is a lightweight fetch of ai-analysis/index.json (~same-origin, <50ms).
  // By awaiting it here, _aiRegimeFresh is set and the badge shows the correct regime
  // on first paint — fetchRiskData will see _aiRegimeFresh=true and not overwrite it.
  await loadAIRegime();

  // External API data — all in parallel.
  // fetchCrossAssetData runs immediately (no longer waits for fetchRiskData) so the
  // Cross-Asset panel populates from the intraday JSON cache on first render (~100ms).
  // Gold/SPX ratio is computed inside fetchCrossAssetData once it has both values.
  fetchRiskData();
  fetchCrossAssetData();
  fetchCommodityQuotes();
  fetchCryptoQuotes();               // BTC from CoinGecko
  buildRichNarrative();              // AI narrative full build (non-blocking, fills narrative text)
  setTimeout(fetchSentiment, 800);   // Dukascopy sentiment (last, non-critical)
}

boot();

// Refresh quote bar FX every 60 seconds via intraday JSON / yfinance (~15 min delay)
setInterval(fetchQuoteBarRT, 60 * 1000);
// Refresh ECB rates every 30 minutes (FX table + heatmap + cross rows)
setInterval(fetchFrankfurter, 30 * 60 * 1000);
// Refresh news every 10 minutes
setInterval(fetchNewsData, 10 * 60 * 1000);  // every 10 min; ETag avoids re-download if unchanged
// Refresh narrative every 15 minutes
setInterval(buildRichNarrative, 15 * 60 * 1000);
// Refresh risk/yield data every 5 minutes

// ═══════════════════════════════════════════════════════════════════
// TOP NAV — smooth scroll to sections + active state
// ═══════════════════════════════════════════════════════════════════
(function() {
  const main = document.getElementById('main');
  const rightPanel = document.getElementById('rightpanel');
  // Targets that live in the right panel sidebar, not main
  const RIGHT_PANEL_TARGETS = new Set(['section-cbrates']);

  document.querySelectorAll('.top-nav a[data-target]').forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const target = this.dataset.target;
      document.querySelectorAll('.top-nav a').forEach(a => a.classList.remove('active'));
      this.classList.add('active');

      if (target === 'top') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        if (main) main.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }

      const el = document.getElementById(target);
      if (!el) return;

      // Check if this target is in the right panel
      if (RIGHT_PANEL_TARGETS.has(target) && rightPanel) {
        const mainScrollable = main && main.scrollHeight > main.clientHeight && getComputedStyle(main).overflowY !== 'visible';
        if (mainScrollable) {
          // Desktop: rightpanel is fixed aside — scroll rightpanel to the element
          const offset = el.offsetTop - rightPanel.offsetTop;
          rightPanel.scrollTo({ top: offset - 4, behavior: 'smooth' });
        } else {
          // Mobile: rightpanel is stacked below main — use scrollIntoView
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
      }

      // Normal main-panel targets
      const mainScrollable = main && main.scrollHeight > main.clientHeight && getComputedStyle(main).overflowY !== 'visible';
      if (mainScrollable) {
        const offset = el.offsetTop - (main.offsetTop || 0);
        main.scrollTo({ top: offset - 4, behavior: 'smooth' });
      } else {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();

// ─── FX LIQUIDITY CANVAS — real intraday activity via Frankfurter ───────────
// Strategy: fetch last 48 half-hour EUR/USD ticks from api.frankfurter.app
// and map observed price-change magnitude → proxy for interbank volume.
// Falls back to BIS/LSEG session-overlap baseline if fetch fails.

const LIQ_BASE = [18,14,11,10,12,20,30,42,58,68,72,70,72,82,95,100,95,80,68,55,42,30,22,20];
// Session definitions (UTC hours)
const LIQ_SESSIONS = [
  { name:'Sydney',   start:22, end:7,  color:'rgba(120,100,255,0.10)' },
  { name:'Tokyo',    start:0,  end:9,  color:'rgba(41,98,255,0.08)'  },
  { name:'London',   start:8,  end:17, color:'rgba(38,166,154,0.10)' },
  { name:'New York', start:13, end:22, color:'rgba(246,148,28,0.07)' },
];

let _liqData = null; // cache: array of 48 values (half-hours, UTC 00:00→23:30)

async function fetchLiquidityData() {
  try {
    // Frankfurter: timeseries last 5 days to compute daily range/volatility
    const today = new Date();
    const yyyy = today.getUTCFullYear();
    const mm = String(today.getUTCMonth()+1).padStart(2,'0');
    const dd = String(today.getUTCDate()).padStart(2,'0');
    const startD = new Date(today); startD.setUTCDate(today.getUTCDate()-5);
    const startDate = `${startD.getUTCFullYear()}-${String(startD.getUTCMonth()+1).padStart(2,'0')}-${String(startD.getUTCDate()).padStart(2,'0')}`;
    const url = `https://api.frankfurter.app/${startDate}..${yyyy}-${mm}-${dd}?from=EUR&to=USD,GBP,JPY`;
    const r = await fetch(url);
    if (!r.ok) throw new Error('fetch fail');
    const json = await r.json();
    const rates = Object.values(json.rates);

    let volScalar = 1.0;
    if (rates.length >= 2) {
      const changes = [];
      for (let i = 1; i < rates.length; i++) {
        const prev = rates[i-1], cur = rates[i];
        if (prev.USD && cur.USD) changes.push(Math.abs(cur.USD - prev.USD) / prev.USD);
        if (prev.GBP && cur.GBP) changes.push(Math.abs(cur.GBP - prev.GBP) / prev.GBP);
        if (prev.JPY && cur.JPY) changes.push(Math.abs(cur.JPY - prev.JPY) / prev.JPY);
      }
      const avgChange = changes.reduce((a,b)=>a+b,0)/changes.length;
      // Typical daily range ≈ 0.5% → volScalar=1.0; higher vol → boosted activity
      volScalar = Math.min(2.0, Math.max(0.5, avgChange / 0.005));
    }

    // Build 48 half-hour buckets from session-overlap baseline, scaled by vol
    const nowUTC = new Date().getUTCHours() + new Date().getUTCMinutes()/60;
    const isWeekend = [0,6].includes(new Date().getUTCDay());

    const data = Array.from({length:48}, (_,i) => {
      if (isWeekend) return 2;
      const h = i/2;
      const idx = Math.floor(h) % 24;
      const next = (idx+1) % 24;
      const frac = h - Math.floor(h);
      const v = LIQ_BASE[idx]*(1-frac) + LIQ_BASE[next]*frac;
      // Fade future hours slightly (past = real vol driven, future = baseline)
      const isFuture = h > nowUTC;
      const scale = isFuture ? 0.75 : volScalar;
      return Math.max(2, v * scale);
    });

    _liqData = data;
  } catch(e) {
    // Fallback to baseline only
    const isWeekend = [0,6].includes(new Date().getUTCDay());
    _liqData = Array.from({length:48}, (_,i) => {
      if (isWeekend) return 2;
      const h = i/2, idx=Math.floor(h)%24, next=(idx+1)%24, frac=h-Math.floor(h);
      return Math.max(2, LIQ_BASE[idx]*(1-frac)+LIQ_BASE[next]*frac);
    });
  }
}

function drawLiquidityChart() {
  const canvas = document.getElementById('liquidity-canvas');
  if (!canvas) return;
  const W = canvas.parentElement.clientWidth - 16, H = 110;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  const utcDay = new Date().getUTCDay();
  const isWeekend = utcDay === 0 || utcDay === 6;

  const hours = _liqData || Array.from({length:48}, (_,i) => {
    if (isWeekend) return 2;
    const h = i/2, idx=Math.floor(h)%24, next=(idx+1)%24, frac=h-Math.floor(h);
    return Math.max(2, LIQ_BASE[idx]*(1-frac)+LIQ_BASE[next]*frac);
  });

  const PAD_L=4, PAD_R=4, PAD_T=8, PAD_B=18;
  const cW=W-PAD_L-PAD_R, cH=H-PAD_T-PAD_B;
  const maxV=Math.max(...hours, 10);
  const px = i => PAD_L+(i/(hours.length-1))*cW;
  const py = v => PAD_T+(1-v/maxV)*cH;

  // Current UTC time as slot index (0–47, cada slot = 30 min)
  const nowH = new Date().getUTCHours() + new Date().getUTCMinutes()/60;
  const nowSlot = Math.min(47, Math.floor(nowH * 2));
  const nowX = PAD_L + (nowH / 24) * cW;

  ctx.clearRect(0,0,W,H);

  // Session bands
  if (!isWeekend) {
    // Session bands: start/end in half-hour slots (0=00:00 UTC, 18=09:00 UTC, etc.)
    // Tokyo 00:00-09:00 UTC = slots 0-18 (blue)
    // London 08:00-17:00 UTC = slots 16-34 (teal)
    // NY 13:00-22:00 UTC = slots 26-44 (amber)
    [
      {start:0,  end:18, color:'rgba(41,98,255,0.07)'},
      {start:16, end:34, color:'rgba(38,166,154,0.08)'},
      {start:26, end:44, color:'rgba(246,148,28,0.06)'}
    ].forEach(s => {
      ctx.fillStyle=s.color;
      ctx.fillRect(PAD_L+(s.start/48)*cW, PAD_T, (Math.min(s.end,48)-s.start)/48*cW, cH);
    });
  }

  if (!isWeekend) {
    // ── PAST: filled area sólida ──────────────────────────────────────────
    const gradPast = ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    gradPast.addColorStop(0,'rgba(41,98,255,0.32)');
    gradPast.addColorStop(1,'rgba(41,98,255,0.03)');
    ctx.beginPath();
    for (let i=0; i<=nowSlot; i++) i===0 ? ctx.moveTo(px(i),py(hours[i])) : ctx.lineTo(px(i),py(hours[i]));
    ctx.lineTo(nowX, PAD_T+cH); ctx.lineTo(px(0), PAD_T+cH); ctx.closePath();
    ctx.fillStyle=gradPast; ctx.fill();

    // ── FUTURE: filled area tenue ─────────────────────────────────────────
    const gradFut = ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    gradFut.addColorStop(0,'rgba(41,98,255,0.10)');
    gradFut.addColorStop(1,'rgba(41,98,255,0.01)');
    ctx.beginPath();
    ctx.moveTo(nowX, py(hours[nowSlot]));
    for (let i=nowSlot+1; i<48; i++) ctx.lineTo(px(i),py(hours[i]));
    ctx.lineTo(px(47),PAD_T+cH); ctx.lineTo(nowX,PAD_T+cH); ctx.closePath();
    ctx.fillStyle=gradFut; ctx.fill();

    // ── PAST: línea sólida azul ───────────────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle='#2962ff'; ctx.lineWidth=1.5; ctx.setLineDash([]);
    for (let i=0; i<=nowSlot; i++) i===0 ? ctx.moveTo(px(i),py(hours[i])) : ctx.lineTo(px(i),py(hours[i]));
    ctx.stroke();

    // ── FUTURE: línea punteada azul tenue ────────────────────────────────
    ctx.beginPath(); ctx.strokeStyle='rgba(41,98,255,0.35)'; ctx.lineWidth=1.2; ctx.setLineDash([3,4]);
    ctx.moveTo(nowX, py(hours[nowSlot]));
    for (let i=nowSlot+1; i<48; i++) ctx.lineTo(px(i),py(hours[i]));
    ctx.stroke(); ctx.setLineDash([]);

    // ── NOW-LINE: línea naranja vertical punteada — igual que el original ────
    ctx.strokeStyle='rgba(246,148,28,0.6)'; ctx.lineWidth=1; ctx.setLineDash([2,3]);
    ctx.beginPath(); ctx.moveTo(nowX, PAD_T); ctx.lineTo(nowX, PAD_T+cH); ctx.stroke();
    ctx.setLineDash([]);

  } else {
    // Weekend: curva plana, fill gris
    const grad=ctx.createLinearGradient(0,PAD_T,0,PAD_T+cH);
    grad.addColorStop(0,'rgba(120,123,134,0.15)'); grad.addColorStop(1,'rgba(41,98,255,0.03)');
    ctx.beginPath();
    hours.forEach((v,i) => i===0 ? ctx.moveTo(px(i),py(v)) : ctx.lineTo(px(i),py(v)));
    ctx.lineTo(px(47),PAD_T+cH); ctx.lineTo(px(0),PAD_T+cH); ctx.closePath();
    ctx.fillStyle=grad; ctx.fill();
    ctx.beginPath(); ctx.strokeStyle='#363c4e'; ctx.lineWidth=1.5;
    hours.forEach((v,i) => i===0 ? ctx.moveTo(px(i),py(v)) : ctx.lineTo(px(i),py(v)));
    ctx.stroke();
    ctx.fillStyle='rgba(120,123,134,0.5)'; ctx.font='9px Courier New'; ctx.textAlign='center';
    ctx.fillText('MARKET CLOSED — WEEKEND', W/2, PAD_T+cH/2);
  }

  // Hour labels
  ctx.fillStyle='#4c525e'; ctx.font='8px Courier New'; ctx.textAlign='center';
  ['00','06','12','18','24'].forEach((lbl,i) => ctx.fillText(lbl, PAD_L+(i/4)*cW, H-4));


  // Bottom labels
  const now = new Date();
  const localH = now.getHours().toString().padStart(2,'0');
  const localM = now.getMinutes().toString().padStart(2,'0');
  const tzShort = now.toLocaleTimeString('en',{timeZoneName:'short'}).split(' ').pop() || 'LT';
  setEl('liq-time-label', localH + ':' + localM + ' ' + tzShort);
  if (isWeekend) {
    const reopenDate = new Date();
    const daysUntilSun = (7 - reopenDate.getUTCDay()) % 7;
    reopenDate.setUTCDate(reopenDate.getUTCDate() + (daysUntilSun === 0 ? 0 : daysUntilSun));
    reopenDate.setUTCHours(21, 0, 0, 0);
    const rH = reopenDate.getHours().toString().padStart(2,'0');
    const rM = reopenDate.getMinutes().toString().padStart(2,'0');
    setEl('liq-peak-label', 'Sun ' + rH + ':' + rM);
  } else {
    const peakH = hours.indexOf(Math.max(...hours));
    const peakUTC = new Date(); peakUTC.setUTCHours(Math.floor(peakH/2), peakH%2===0?0:30, 0, 0);
    const pH = peakUTC.getHours().toString().padStart(2,'0');
    const pM = peakUTC.getMinutes().toString().padStart(2,'0');
    setEl('liq-peak-label', 'Peak ' + pH + ':' + pM);
  }
}

// Initial load: fetch real data then draw
fetchLiquidityData().then(() => drawLiquidityChart());
// Refresh data every 30 min, redraw every 60 s
setInterval(() => fetchLiquidityData().then(() => drawLiquidityChart()), 30 * 60 * 1000);
setInterval(drawLiquidityChart, 60 * 1000);
window.addEventListener('resize', drawLiquidityChart);

// ── FX Liquidity tooltip ──────────────────────────────────────────────────────
(function() {
  const SESSION_NAMES = [
    { name:'Sydney',   start:22, end:7  },
    { name:'Tokyo',    start:0,  end:9  },
    { name:'London',   start:8,  end:17 },
    { name:'New York', start:13, end:22 },
  ];

  function getActiveSessions(utcH) {
    const active = SESSION_NAMES.filter(s => {
      if (s.end < s.start) return utcH >= s.start || utcH < s.end; // wraps midnight
      return utcH >= s.start && utcH < s.end;
    }).map(s => s.name);
    return active.length ? active.join(' + ') : 'Inter-session';
  }

  function volLabel(v, maxV) {
    const pct = Math.round((v / maxV) * 100);
    if (pct >= 85) return 'Very High (' + pct + '%)';
    if (pct >= 60) return 'High (' + pct + '%)';
    if (pct >= 35) return 'Moderate (' + pct + '%)';
    if (pct >= 15) return 'Low (' + pct + '%)';
    return 'Very Low (' + pct + '%)';
  }

  const canvas = document.getElementById('liquidity-canvas');
  const tooltip = document.getElementById('liq-tooltip');
  if (!canvas || !tooltip) return;

  canvas.addEventListener('mousemove', function(e) {
    const hours = _liqData;
    if (!hours) return;

    const rect = canvas.getBoundingClientRect();
    const PAD_L = 4, PAD_R = 4, PAD_T = 8, PAD_B = 18;
    const W = canvas.width, H = canvas.height;
    const cW = W - PAD_L - PAD_R;

    // Scale mouse X from CSS pixels to canvas pixels
    const scaleX = W / rect.width;
    const mouseX = (e.clientX - rect.left) * scaleX;
    if (mouseX < PAD_L || mouseX > W - PAD_R) { tooltip.style.display = 'none'; return; }

    // Map x → slot index (0–47, each = 30 min)
    const frac = (mouseX - PAD_L) / cW;
    const slot = Math.max(0, Math.min(47, Math.round(frac * 47)));
    const utcH = slot / 2; // e.g. slot 17 → 08:30 UTC

    const hh = Math.floor(utcH).toString().padStart(2,'0');
    const mm = utcH % 1 === 0 ? '00' : '30';

    // Convert UTC slot to local time for display
    const d = new Date(); d.setUTCHours(Math.floor(utcH), utcH%1===0?0:30, 0, 0);
    const localHH = d.getHours().toString().padStart(2,'0');
    const localMM = d.getMinutes().toString().padStart(2,'0');
    const tzShort = d.toLocaleTimeString('en',{timeZoneName:'short'}).split(' ').pop() || 'LT';

    const maxV = Math.max(...hours, 10);
    const v = hours[slot];

    document.getElementById('liq-tt-time').textContent = hh + ':' + mm + ' UTC  (' + localHH + ':' + localMM + ' ' + tzShort + ')';
    document.getElementById('liq-tt-session').textContent = '▸ ' + getActiveSessions(Math.floor(utcH));
    const isPast = slot <= Math.floor(new Date().getUTCHours()*2 + new Date().getUTCMinutes()/30);
    document.getElementById('liq-tt-vol').textContent = (isPast ? '⬤' : '○') + ' ' + (isPast ? '' : '(est.) ') + volLabel(v, maxV);

    // Position tooltip next to cursor using fixed coordinates
    const ttW = tooltip.offsetWidth || 170;
    const ttH = tooltip.offsetHeight || 56;
    let left = e.clientX + 14;
    let top  = e.clientY - ttH / 2;
    // Flip left if near right edge of viewport
    if (left + ttW > window.innerWidth - 8) left = e.clientX - ttW - 14;
    if (top < 4) top = 4;
    if (top + ttH > window.innerHeight - 4) top = window.innerHeight - ttH - 4;
    tooltip.style.left = left + 'px';
    tooltip.style.top  = top  + 'px';
    tooltip.style.display = 'block';
  });

  canvas.addEventListener('mouseleave', function() {
    tooltip.style.display = 'none';
  });
})();

// Risk + Cross-Asset run in parallel every 2 min — same as boot() — no chaining
setInterval(() => { fetchRiskData(); fetchCrossAssetData(); fetchCommodityQuotes(); }, 2 * 60 * 1000);
// Crypto: every 90 seconds
setInterval(fetchCryptoQuotes, 90 * 1000);
setInterval(fetchCarryData, 30 * 60 * 1000);
// Refresh sentiment every 30 seconds
setInterval(fetchSentiment, 30 * 1000);
// Refresh calendar & expectations every 30 minutes
setInterval(fetchFedExpectations, 30 * 60 * 1000);

// ═══════════════════════════════════════════════════════════════════
// MOBILE VISIBILITY FIX — TradingView widgets + FX Liquidity chart
// When the browser tab/app returns to foreground on mobile, iframes
// may go blank and canvas charts may render at wrong dimensions.
// We force a redraw whenever the page becomes visible again.
// ═══════════════════════════════════════════════════════════════════
(function() {
  // Helper: reload the active TradingView chart by simulating a tab click
  function reloadActiveTVChart() {
    const activeTab = document.querySelector('.tv-tab.active');
    if (activeTab) activeTab.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  }

  // Helper: reload the Economic Calendar widget by re-injecting its script
  function reloadTVCalendar() {
    const scaleWrap = document.getElementById('tvcal-scale');
    if (!scaleWrap) return;
    // Remove existing iframe/content and re-create the widget container
    const container = scaleWrap.querySelector('.tradingview-widget-container');
    if (!container) return;
    const existingScript = container.querySelector('script');
    if (!existingScript) return;
    // Clone the widget container content to force re-init
    const clone = container.cloneNode(true);
    container.parentNode.replaceChild(clone, container);
  }

  // FX Liquidity chart: force redraw when visible
  function redrawLiquidityIfVisible() {
    const canvas = document.getElementById('liquidity-canvas');
    if (!canvas) return;
    // Only redraw if canvas has zero dimensions (collapsed/invisible at paint time)
    if (canvas.parentElement.clientWidth > 0) drawLiquidityChart();
  }

  // Detect mobile once (pointer: coarse covers phones + tablets)
  var isMobile = window.matchMedia('(pointer: coarse)').matches;

  // On page visibility change (tab switch, app background/foreground)
  document.addEventListener('visibilitychange', function() {
    if (document.visibilityState !== 'visible') return;
    // Small delay to let the browser re-paint before we measure dimensions
    setTimeout(function() {
      redrawLiquidityIfVisible();
      // Only reload TV chart on mobile — desktop widgets stay alive across tab switches
      if (isMobile) reloadActiveTVChart();
    }, 350);
  });

  // On pageshow (iOS Safari fires this when returning from bfcache)
  window.addEventListener('pageshow', function(e) {
    if (!e.persisted) return; // only for bfcache restores
    setTimeout(function() {
      redrawLiquidityIfVisible();
      // Mobile only: widgets may have gone blank after bfcache restore
      if (isMobile) {
        reloadActiveTVChart();
        reloadTVCalendar();
      }
    }, 350);
  });

  // IntersectionObserver: redraw liquidity chart the first time it enters viewport
  // (fixes the wrong-position bug when the chart is not visible on initial paint)
  const liqCanvas = document.getElementById('liquidity-canvas');
  if (liqCanvas && typeof IntersectionObserver !== 'undefined') {
    const obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          drawLiquidityChart();
          obs.unobserve(entry.target); // only needed once per session
        }
      });
    }, { threshold: 0.1 });
    obs.observe(liqCanvas);
  }
})();

// ═══════════════════════════════════════════════════════════════════
// ACCESSIBILITY — WCAG 2.1 AA enhancements
// ═══════════════════════════════════════════════════════════════════
(function initA11y() {
  // ── 1. Site menu: sync aria-expanded with :focus-within state ──
  const menuBtn = document.querySelector('.site-menu-btn');
  const siteMenu = document.querySelector('.site-menu');
  if (menuBtn && siteMenu) {
    // :focus-within shows the panel via CSS; mirror state in aria-expanded
    siteMenu.addEventListener('focusin',  () => menuBtn.setAttribute('aria-expanded', 'true'));
    siteMenu.addEventListener('focusout', (e) => {
      // Only collapse if focus left the entire .site-menu
      if (!siteMenu.contains(e.relatedTarget)) {
        menuBtn.setAttribute('aria-expanded', 'false');
      }
    });
    siteMenu.addEventListener('mouseenter', () => menuBtn.setAttribute('aria-expanded', 'true'));
    siteMenu.addEventListener('mouseleave', () => menuBtn.setAttribute('aria-expanded', 'false'));
  }

  // ── 2. Chart tabs: sync aria-selected on click ──
  const tablist = document.getElementById('tv-pair-tabs');
  if (tablist) {
    tablist.addEventListener('click', (e) => {
      const btn = e.target.closest('.tv-tab');
      if (!btn) return;
      tablist.querySelectorAll('.tv-tab').forEach(t => {
        t.setAttribute('aria-selected', t === btn ? 'true' : 'false');
      });
    });
  }

  // ── 3. Top-nav scroll links: add aria-current="page" to active ──
  const topNavLinks = document.querySelectorAll('.top-nav a');
  topNavLinks.forEach(link => {
    link.addEventListener('click', () => {
      topNavLinks.forEach(l => l.removeAttribute('aria-current'));
      link.setAttribute('aria-current', 'location');
    });
  });
  // Set initial aria-current on Overview
  const firstNavLink = document.querySelector('.top-nav a.active');
  if (firstNavLink) firstNavLink.setAttribute('aria-current', 'location');

  // ── 4. Live region: announce price updates to screen readers ──
  // A visually-hidden sr-only announcement div for dynamic price changes
  if (!document.getElementById('sr-announce')) {
    const announce = document.createElement('div');
    announce.id = 'sr-announce';
    announce.setAttribute('role', 'status');
    announce.setAttribute('aria-live', 'polite');
    announce.setAttribute('aria-atomic', 'true');
    announce.className = 'sr-only';
    document.body.appendChild(announce);
  }
})();
