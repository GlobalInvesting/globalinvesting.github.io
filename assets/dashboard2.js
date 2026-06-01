/* ═══════════════════════════════════════════════════════════════
   Global Investing FX Terminal v2 — dashboard2.js
   Bloomberg-style data integration
   ═══════════════════════════════════════════════════════════════ */
'use strict';

/* ── Constants ─────────────────────────────────────────────── */
const BASE = '';  // same-origin

const FX_PAIRS_G10 = [
  'eurusd','gbpusd','usdjpy','usdchf','audusd','usdcad','nzdusd',
  'eurgbp','eurjpy','eurchf','gbpjpy','gbpchf','audjpy','audnzd'
];
const FX_PAIRS_CROSS = [
  'eurcad','euraud','eurnzd','gbpcad','gbpaud','gbpnzd',
  'audchf','audcad','cadjpy','chfjpy','nzdjpy','nzdcad','nzdchf','cadchf'
];

const FX_LABELS = {
  eurusd:'EUR/USD', gbpusd:'GBP/USD', usdjpy:'USD/JPY', usdchf:'USD/CHF',
  audusd:'AUD/USD', usdcad:'USD/CAD', nzdusd:'NZD/USD', eurgbp:'EUR/GBP',
  eurjpy:'EUR/JPY', eurchf:'EUR/CHF', gbpjpy:'GBP/JPY', gbpchf:'GBP/CHF',
  audjpy:'AUD/JPY', audnzd:'AUD/NZD', eurcad:'EUR/CAD', euraud:'EUR/AUD',
  eurnzd:'EUR/NZD', gbpcad:'GBP/CAD', gbpaud:'GBP/AUD', gbpnzd:'GBP/NZD',
  audchf:'AUD/CHF', audcad:'AUD/CAD', cadjpy:'CAD/JPY', chfjpy:'CHF/JPY',
  nzdjpy:'NZD/JPY', nzdcad:'NZD/CAD', nzdchf:'NZD/CHF', cadchf:'CAD/CHF'
};

const CB_CURRENCIES = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const CB_NAMES = {
  USD:'Federal Reserve', EUR:'European Central Bank', GBP:'Bank of England',
  JPY:'Bank of Japan', AUD:'Reserve Bank of Australia', CAD:'Bank of Canada',
  CHF:'Swiss National Bank', NZD:'Reserve Bank of New Zealand'
};
const CB_SHORT = {
  USD:'Fed', EUR:'ECB', GBP:'BoE', JPY:'BoJ', AUD:'RBA', CAD:'BoC', CHF:'SNB', NZD:'RBNZ'
};
const COT_CURRENCIES = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const CROSS_ASSETS = [
  {key:'vix',   label:'VIX',     fmt:'dec2'},
  {key:'spx',   label:'S&P 500', fmt:'dec0'},
  {key:'dxy',   label:'DXY',     fmt:'dec3'},
  {key:'gold',  label:'Gold',    fmt:'dec0'},
  {key:'wti',   label:'WTI',     fmt:'dec2'},
  {key:'brent', label:'Brent',   fmt:'dec2'},
  {key:'us10y', label:'US10Y',   fmt:'dec3', suffix:'%'},
  {key:'us2y',  label:'US2Y',    fmt:'dec3', suffix:'%'},
  {key:'us3m',  label:'US3M',    fmt:'dec3', suffix:'%'},
  {key:'move',  label:'MOVE',    fmt:'dec1'},
  {key:'btc',   label:'BTC',     fmt:'dec0'},
  {key:'nasdaq',label:'NASDAQ',  fmt:'dec0'},
];

/* ── State ─────────────────────────────────────────────────── */
let _quotes   = {};
let _meetings = {};
let _rates    = {};
let _cot      = {};
let _rr       = {};
let _selectedPair = 'eurusd';
let _fxTab    = 'g10';
let _lowerTab = 'crossasset';

/* ── Formatting helpers ────────────────────────────────────── */
function fmt(v, type, suffix='') {
  if (v == null || isNaN(v)) return '—';
  if (type === 'dec0') return v.toLocaleString('en-US', {minimumFractionDigits:0, maximumFractionDigits:0}) + suffix;
  if (type === 'dec1') return v.toFixed(1) + suffix;
  if (type === 'dec2') return v.toFixed(2) + suffix;
  if (type === 'dec3') return v.toFixed(3) + suffix;
  if (type === 'dec4') return v.toFixed(4) + suffix;
  if (type === 'pct')  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
  if (type === 'pct3') return (v >= 0 ? '+' : '') + v.toFixed(3) + '%';
  return v + suffix;
}

function fmtPair(pair, price) {
  if (price == null) return '—';
  const jpyPairs = ['usdjpy','eurjpy','gbpjpy','audjpy','cadjpy','chfjpy','nzdjpy'];
  const p = pair.toLowerCase();
  if (jpyPairs.includes(p)) return price.toFixed(3);
  return price.toFixed(5);
}

function colorCls(v) {
  if (v == null) return 'neu';
  return v > 0 ? 'up' : v < 0 ? 'dn' : 'neu';
}

function pctSign(v) {
  if (v == null || isNaN(v)) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(3) + '%';
}

/* ── Clock ─────────────────────────────────────────────────── */
function tickClock() {
  const el = document.getElementById('tb-clock');
  if (!el) return;
  const now = new Date();
  const hh = String(now.getUTCHours()).padStart(2,'0');
  const mm = String(now.getUTCMinutes()).padStart(2,'0');
  const ss = String(now.getUTCSeconds()).padStart(2,'0');
  el.textContent = `${hh}:${mm}:${ss} UTC`;
}
setInterval(tickClock, 1000);
tickClock();

/* ── Data fetching ─────────────────────────────────────────── */
async function loadQuotes() {
  try {
    const r = await fetch(`${BASE}/intraday-data/quotes.json`);
    if (!r.ok) return;
    const d = await r.json();
    _quotes = d.quotes || {};
  } catch(e) { console.warn('quotes fetch failed', e); }
}

async function loadMeetings() {
  try {
    const r = await fetch(`${BASE}/meetings-data/meetings.json`);
    if (!r.ok) return;
    const d = await r.json();
    _meetings = d.meetings || {};
  } catch(e) { console.warn('meetings fetch failed', e); }
}

async function loadRates() {
  try {
    const fetches = CB_CURRENCIES.map(c =>
      fetch(`${BASE}/rates/${c}.json`)
        .then(r => r.ok ? r.json() : null)
        .then(d => ({ c, d }))
        .catch(() => ({ c, d: null }))
    );
    const results = await Promise.all(fetches);
    for (const {c, d} of results) {
      if (d && d.observations && d.observations.length) {
        _rates[c] = parseFloat(d.observations[0].value);
      }
    }
  } catch(e) { console.warn('rates fetch failed', e); }
}

async function loadCOT() {
  try {
    const fetches = COT_CURRENCIES.map(c =>
      fetch(`${BASE}/cot-data/${c}.json`)
        .then(r => r.ok ? r.json() : null)
        .then(d => ({ c, d }))
        .catch(() => ({ c, d: null }))
    );
    const results = await Promise.all(fetches);
    for (const {c, d} of results) {
      if (d) _cot[c] = d;
    }
  } catch(e) { console.warn('COT fetch failed', e); }
}

async function loadRR() {
  try {
    const r = await fetch(`${BASE}/rr-data/rr.json`);
    if (!r.ok) return;
    const d = await r.json();
    _rr = d.pairs || {};
  } catch(e) { console.warn('RR fetch failed', e); }
}

/* ── Render Quote Bar ──────────────────────────────────────── */
function renderQuotebar() {
  const bar = document.getElementById('quotebar');
  if (!bar) return;
  const items = [
    {key:'eurusd', label:'EUR/USD'},
    {key:'gbpusd', label:'GBP/USD'},
    {key:'usdjpy', label:'USD/JPY'},
    {key:'audusd', label:'AUD/USD'},
    {key:'usdcad', label:'USD/CAD'},
    {key:'dxy',    label:'DXY'},
    {key:'gold',   label:'GOLD'},
    {key:'vix',    label:'VIX'},
    {key:'spx',    label:'S&P 500'},
    {key:'us10y',  label:'US10Y'},
  ];
  bar.innerHTML = items.map(({key, label}) => {
    const q = _quotes[key];
    if (!q) return `<div class="qb-item"><span class="qb-sym">${label}</span><span class="qb-price dim">—</span></div>`;
    const price = key === 'eurusd' || key === 'gbpusd' || key === 'audusd' || key === 'usdcad' || key === 'nzdusd'
      ? (q.close || 0).toFixed(5)
      : key === 'usdjpy' ? (q.close || 0).toFixed(3)
      : key === 'dxy' || key === 'vix' ? (q.close || 0).toFixed(3)
      : key === 'gold' ? (q.close || 0).toFixed(0)
      : key === 'spx' ? (q.close || 0).toFixed(2)
      : key === 'us10y' ? (q.close || 0).toFixed(3) + '%'
      : (q.close || 0).toFixed(2);
    const pct = q.pct || 0;
    const cls = pct > 0 ? 'up' : pct < 0 ? 'dn' : 'neu';
    const sign = pct > 0 ? '+' : '';
    return `<div class="qb-item">
      <span class="qb-sym">${label}</span>
      <span class="qb-price">${price}</span>
      <span class="qb-chg ${cls}">${sign}${pct.toFixed(2)}%</span>
    </div>`;
  }).join('');
}

/* ── Render FX Watchlist ───────────────────────────────────── */
function renderFXWatchlist() {
  const tbody = document.getElementById('fx-tbody');
  if (!tbody) return;
  const pairs = _fxTab === 'g10' ? FX_PAIRS_G10 : FX_PAIRS_CROSS;
  tbody.innerHTML = pairs.map(pair => {
    const q = _quotes[pair];
    const label = FX_LABELS[pair] || pair.toUpperCase();
    const price = q ? fmtPair(pair, q.close) : '—';
    const bid   = q && q.bid ? fmtPair(pair, q.bid) : '—';
    const ask   = q && q.ask ? fmtPair(pair, q.ask) : '—';
    const pct   = q ? (q.pct || 0) : null;
    const cls   = pct !== null ? (pct > 0 ? 'up' : pct < 0 ? 'dn' : 'neu') : 'neu';
    const hv30  = q && q.hv30 != null ? q.hv30.toFixed(1) + '%' : '—';
    const sel   = pair === _selectedPair ? 'selected' : '';
    return `<tr class="${sel}" onclick="selectPair('${pair}')">
      <td class="sym">${label}</td>
      <td class="r bold">${price}</td>
      <td class="r dim hide-sm">${bid}</td>
      <td class="r dim hide-sm">${ask}</td>
      <td class="r ${cls}">${pct !== null ? (pct >= 0 ? '+' : '') + pct.toFixed(3) + '%' : '—'}</td>
      <td class="r dim hide-sm">${hv30}</td>
    </tr>`;
  }).join('');
}

/* ── Render Cross-Asset Grid ───────────────────────────────── */
function renderCrossAsset() {
  const el = document.getElementById('ca-grid');
  if (!el) return;
  el.innerHTML = CROSS_ASSETS.map(({key, label, fmt: fmtType, suffix}) => {
    const q = _quotes[key];
    if (!q) return `<div class="ca-cell"><div class="ca-sym">${label}</div><div class="ca-price dim">—</div></div>`;
    const price = fmt(q.close, fmtType, suffix || '');
    const pct   = q.pct || 0;
    const cls   = pct > 0 ? 'up' : pct < 0 ? 'dn' : 'neu';
    const sign  = pct >= 0 ? '+' : '';
    return `<div class="ca-cell" title="${label}">
      <div class="ca-sym">${label}</div>
      <div class="ca-price">${price}</div>
      <div class="ca-chg ${cls}">${sign}${pct.toFixed(2)}%</div>
    </div>`;
  }).join('');
}

/* ── Render CB Rates ───────────────────────────────────────── */
function renderCBRates() {
  const html = CB_CURRENCIES.map(c => {
    const rate   = _rates[c];
    const m      = _meetings[c] || {};
    const bias   = m.bias || 'hold';
    const next   = m.nextMeeting || '—';
    const fwd    = m.fwdRate != null ? m.fwdRate.toFixed(2) + '%' : '—';
    const rateStr = rate != null ? rate.toFixed(2) + '%' : '—';
    const biasCls = bias === 'cut' ? 'bias-cut' : bias === 'hike' ? 'bias-hike' : 'bias-hold';
    const biasLabel = bias === 'cut' ? '↓ Cut' : bias === 'hike' ? '↑ Hike' : '→ Hold';
    return `<tr>
      <td class="sym">${CB_SHORT[c]}</td>
      <td class="r bold">${rateStr}</td>
      <td class="r"><span class="bias-pill ${biasCls}">${biasLabel}</span></td>
      <td class="r dim hide-sm">${next}</td>
      <td class="r dim hide-sm">${fwd}</td>
    </tr>`;
  }).join('');
  // populate both tbodies (right panel + lower tab)
  ['cbrates-tbody', 'cbrates-lower-tbody'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  });
}

/* ── Render COT Table ──────────────────────────────────────── */
function renderCOT() {
  const tbody = document.getElementById('cot-tbody');
  if (!tbody) return;
  tbody.innerHTML = COT_CURRENCIES.map(c => {
    const d = _cot[c] || {};
    const net  = d.netPosition;
    const longs = d.longPositions;
    const shorts = d.shortPositions;
    const week = d.weekEnding || '—';
    const netStr = net != null ? (net >= 0 ? '+' : '') + net.toLocaleString('en-US') : '—';
    const cls    = net != null ? (net > 0 ? 'up' : net < 0 ? 'dn' : 'neu') : 'neu';
    // bar: net as % of total OI (approx)
    const total = (longs || 0) + (shorts || 0);
    const netPct = total > 0 ? net / total : 0;
    const barW   = Math.min(Math.abs(netPct) * 100, 50);
    const barColor = net > 0 ? 'var(--up)' : 'var(--down)';
    // WoW from history
    let wow = '—';
    if (d.history && d.history.length >= 2) {
      const hist = d.history;
      const last = hist[hist.length - 1];
      const prev = hist[hist.length - 2];
      if (last && prev && last.levNet != null && prev.levNet != null) {
        const diff = last.levNet - prev.levNet;
        wow = `<span class="${diff >= 0 ? 'up' : 'dn'}">${diff >= 0 ? '+' : ''}${diff.toLocaleString('en-US')}</span>`;
      }
    }
    const rr25 = _rr[c + 'USD'] || _rr['USD' + c] || _rr['EUR' + (c === 'EUR' ? 'USD' : c)];
    const rrStr = rr25 && rr25.rr25d != null ? (rr25.rr25d >= 0 ? '+' : '') + rr25.rr25d.toFixed(2) : '—';
    const rrCls = rr25 && rr25.rr25d != null ? (rr25.rr25d > 0 ? 'up' : rr25.rr25d < 0 ? 'dn' : 'neu') : 'neu';
    return `<tr>
      <td class="sym">${c}</td>
      <td class="r ${cls} bold">${netStr}</td>
      <td>
        <div class="cot-bar-wrap">
          <div class="cot-bar-bg"><div class="cot-bar-fill" style="width:${barW}%;background:${barColor};margin-left:${net < 0 ? 0 : 50 - barW}%"></div></div>
        </div>
      </td>
      <td class="r dim hide-sm">${wow}</td>
      <td class="r ${rrCls} hide-sm">${rrStr}</td>
      <td class="dim hide-sm">${week}</td>
    </tr>`;
  }).join('');
}

/* ── Render Yield Curve ────────────────────────────────────── */
function renderYieldCurve() {
  const canvas = document.getElementById('yc-canvas');
  if (!canvas) return;
  const tenors = [
    {key:'us3m',  label:'3M', months:3},
    {key:'us2y',  label:'2Y', months:24},
    {key:'us5y',  label:'5Y', months:60},
    {key:'us10y', label:'10Y',months:120},
    {key:'us30y', label:'30Y',months:360},
  ];
  const rates = tenors.map(t => {
    const q = _quotes[t.key];
    return {label: t.label, value: q ? q.close : null, months: t.months};
  });
  const valid = rates.filter(r => r.value != null);
  if (valid.length < 2) return;
  const W = canvas.offsetWidth || 400;
  const H = canvas.offsetHeight || 160;
  const pad = {t:15, r:15, b:28, l:38};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const minY = Math.min(...valid.map(r => r.value)) * 0.97;
  const maxY = Math.max(...valid.map(r => r.value)) * 1.03;
  const scaleX = m => pad.l + (m / 360) * cw;
  const scaleY = v => pad.t + (1 - (v - minY)/(maxY - minY)) * ch;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  // grid
  ctx.strokeStyle = '#1a1e26'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ch / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }
  // gradient fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
  grad.addColorStop(0, 'rgba(59,130,246,0.25)');
  grad.addColorStop(1, 'rgba(59,130,246,0)');
  ctx.beginPath();
  ctx.moveTo(scaleX(valid[0].months), scaleY(valid[0].value));
  for (let i = 1; i < valid.length; i++) {
    ctx.lineTo(scaleX(valid[i].months), scaleY(valid[i].value));
  }
  ctx.lineTo(scaleX(valid[valid.length-1].months), H - pad.b);
  ctx.lineTo(scaleX(valid[0].months), H - pad.b);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  // line
  ctx.beginPath();
  ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5;
  ctx.moveTo(scaleX(valid[0].months), scaleY(valid[0].value));
  for (let i = 1; i < valid.length; i++) {
    ctx.lineTo(scaleX(valid[i].months), scaleY(valid[i].value));
  }
  ctx.stroke();
  // dots + labels
  ctx.fillStyle = '#3b82f6';
  ctx.font = '8px JetBrains Mono, Consolas, monospace';
  valid.forEach(r => {
    const x = scaleX(r.months);
    const y = scaleY(r.value);
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = '#8b95a6';
    ctx.textAlign = 'center';
    ctx.fillText(r.label, x, H - pad.b + 10);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(r.value.toFixed(2) + '%', x, y - 6);
    ctx.fillStyle = '#3b82f6';
  });
  // Y axis labels
  ctx.fillStyle = '#5a6275'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minY + (maxY - minY) * (1 - i/4);
    const y = pad.t + (ch / 4) * i;
    ctx.fillText(v.toFixed(2), pad.l - 3, y + 3);
  }
  // Spread annotation
  const s3m = _quotes['us3m'] && _quotes['us3m'].close;
  const s10y = _quotes['us10y'] && _quotes['us10y'].close;
  if (s3m && s10y) {
    const spread = (s10y - s3m).toFixed(0);
    const cls = s10y > s3m ? '#22c55e' : '#ef4444';
    ctx.fillStyle = cls;
    ctx.textAlign = 'right';
    ctx.font = '8px JetBrains Mono, Consolas, monospace';
    ctx.fillText(`10Y-3M: ${spread >= 0 ? '+' : ''}${spread}bp`, W - pad.r, pad.t + 10);
  }
}

/* ── Select pair & update chart ────────────────────────────── */
function selectPair(pair) {
  _selectedPair = pair;
  renderFXWatchlist();
  updateChartLabel();
  loadTVChart(pair);
}

function updateChartLabel() {
  const q = _quotes[_selectedPair];
  const lbl = FX_LABELS[_selectedPair] || _selectedPair.toUpperCase();
  const el = document.getElementById('chart-pair-label');
  const pel = document.getElementById('chart-price-label');
  const cel = document.getElementById('chart-chg-label');
  if (el) el.textContent = lbl;
  if (q && pel) {
    pel.textContent = fmtPair(_selectedPair, q.close);
    const pct = q.pct || 0;
    if (cel) {
      cel.className = 'chart-chg-label ' + (pct > 0 ? 'up' : pct < 0 ? 'dn' : 'neu');
      cel.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(3) + '%';
    }
  }
}

/* ── TradingView Chart ─────────────────────────────────────── */
let _tvInterval = '60';
function loadTVChart(pair) {
  const wrap = document.getElementById('tv-chart-wrap');
  if (!wrap) return;
  const sym = pair.toUpperCase();
  const script = document.createElement('script');
  wrap.innerHTML = '<div class="tradingview-widget-container" style="height:100%;width:100%"><div id="tv-widget" style="height:100%;"></div></div>';
  const s = document.createElement('script');
  s.src = 'https://s3.tradingview.com/tv.js';
  s.onload = function() {
    // eslint-disable-next-line no-undef
    new TradingView.widget({
      container_id: 'tv-widget',
      autosize: true,
      symbol: 'FX:' + sym,
      interval: _tvInterval,
      timezone: 'UTC',
      theme: 'dark',
      style: '1',
      locale: 'en',
      toolbar_bg: '#000000',
      enable_publishing: false,
      hide_top_toolbar: false,
      hide_legend: false,
      save_image: false,
      backgroundColor: '#000000',
      gridColor: 'rgba(26,30,38,0.5)',
      hide_side_toolbar: false,
    });
  };
  wrap.appendChild(s);
}

/* ── FX Tab switching ──────────────────────────────────────── */
function setFXTab(tab) {
  _fxTab = tab;
  document.querySelectorAll('.fx-tab').forEach(t => t.classList.remove('active'));
  const el = document.getElementById('fxtab-' + tab);
  if (el) el.classList.add('active');
  renderFXWatchlist();
}

/* ── Lower panel tab switching ─────────────────────────────── */
function setLowerTab(tab) {
  _lowerTab = tab;
  document.querySelectorAll('.lower-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.lower-content').forEach(t => t.classList.remove('active'));
  const btn = document.getElementById('ltab-' + tab);
  if (btn) btn.classList.add('active');
  const content = document.getElementById('lcontent-' + tab);
  if (content) content.classList.add('active');
  if (tab === 'yieldcurve') {
    requestAnimationFrame(renderYieldCurve);
  }
}

/* ── Risk Score ─────────────────────────────────────────────── */
function computeRiskScore() {
  let score = 0;
  const vix = _quotes['vix'] && _quotes['vix'].close;
  const gold = _quotes['gold'] && _quotes['gold'].pct;
  const spx  = _quotes['spx']  && _quotes['spx'].pct;
  const move = _quotes['move'] && _quotes['move'].close;
  const us10y = _quotes['us10y'] && _quotes['us10y'].close;
  const us3m  = _quotes['us3m']  && _quotes['us3m'].close;
  if (vix > 30) score += 3;
  else if (vix > 25) score += 2;
  else if (vix > 18) score += 1;
  if (us10y && us3m && us10y < us3m) score += 1;
  if (gold > 2.0) score += 1;
  if (spx < -1.5) score += 1;
  if (move > 100) score += 1;
  if (score >= 4) return {label:'RISK-OFF', score};
  if (score >= 2) return {label:'CAUTION', score};
  if (score >= 1) return {label:'MIXED', score};
  return {label:'RISK-ON', score};
}

function renderRiskBadge() {
  const el = document.getElementById('risk-regime');
  if (!el) return;
  const {label, score} = computeRiskScore();
  el.className = 'regime-badge regime-' + label.replace(' ','-');
  el.textContent = label + ' ·' + score;
}

function renderRiskRows() {
  const wrap = document.getElementById('risk-rows');
  if (!wrap) return;
  const items = [
    {label:'VIX',        key:'vix',   suf:''},
    {label:'MOVE Index', key:'move',  suf:''},
    {label:'Gold 1D%',   key:'gold',  pct:true},
    {label:'S&P 500 1D%',key:'spx',   pct:true},
    {label:'DXY',        key:'dxy',   suf:''},
    {label:'BTC',        key:'btc',   suf:''},
  ];
  wrap.innerHTML = items.map(({label, key, pct: isPct, suf}) => {
    const q = _quotes[key];
    if (!q) return `<div class="risk-row"><span class="risk-label">${label}</span><span class="risk-val dim">—</span></div>`;
    const v = isPct ? (q.pct || 0) : q.close;
    const cls = isPct ? colorCls(v) : 'neu';
    const vStr = isPct ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : v.toLocaleString('en-US', {maximumFractionDigits:2}) + suf;
    return `<div class="risk-row"><span class="risk-label">${label}</span><span class="risk-val ${cls}">${vStr}</span></div>`;
  }).join('');
}

/* ── RR Table (in right panel) ─────────────────────────────── */
function renderRRTable() {
  const tbody = document.getElementById('rr-tbody');
  if (!tbody) return;
  const pairs = ['EURUSD','USDJPY','GBPUSD','AUDUSD','USDCAD','USDCHF','EURJPY'];
  tbody.innerHTML = pairs.map(p => {
    const d = _rr[p];
    if (!d) return `<tr><td class="sym">${p.slice(0,3)}/${p.slice(3)}</td><td class="r dim" colspan="2">—</td></tr>`;
    const rr = d.rr25d;
    const cls = rr > 0 ? 'up' : rr < 0 ? 'dn' : 'neu';
    const str = (rr >= 0 ? '+' : '') + rr.toFixed(2);
    return `<tr>
      <td class="sym">${p.slice(0,3)}/${p.slice(3)}</td>
      <td class="r ${cls} bold">${str}</td>
      <td class="r dim">${d.tenor || '1M'}</td>
    </tr>`;
  }).join('');
}

/* ── Full render ───────────────────────────────────────────── */
function renderAll() {
  renderQuotebar();
  renderFXWatchlist();
  renderCrossAsset();
  renderCBRates();
  renderCOT();
  renderRiskBadge();
  renderRiskRows();
  renderRRTable();
  updateChartLabel();
  if (_lowerTab === 'yieldcurve') renderYieldCurve();
}

/* ── Boot ───────────────────────────────────────────────────── */
async function boot() {
  // Load all data in parallel
  await Promise.all([loadQuotes(), loadMeetings(), loadRates(), loadCOT(), loadRR()]);
  renderAll();
  // Refresh every 5 minutes
  setInterval(async () => {
    await loadQuotes();
    renderAll();
  }, 5 * 60 * 1000);
  // Slow refresh for static data
  setInterval(async () => {
    await Promise.all([loadMeetings(), loadRates(), loadCOT(), loadRR()]);
    renderAll();
  }, 60 * 60 * 1000);
  // Resize observer for yield curve canvas
  const canvas = document.getElementById('yc-canvas');
  if (canvas) {
    const ro = new ResizeObserver(() => {
      if (_lowerTab === 'yieldcurve') renderYieldCurve();
    });
    ro.observe(canvas);
  }
}

document.addEventListener('DOMContentLoaded', boot);

// Expose globals for inline event handlers and cross-script access
window.selectPair  = selectPair;
window.setFXTab    = setFXTab;
window.setLowerTab = setLowerTab;
window.loadTVChart = loadTVChart;
window.renderAll   = renderAll;
window.renderRRTable = renderRRTable;

// Expose state references
Object.defineProperty(window, '_quotes',   { get: () => _quotes });
Object.defineProperty(window, '_rr',       { get: () => _rr });
Object.defineProperty(window, '_meetings', { get: () => _meetings });
Object.defineProperty(window, '_rates',    { get: () => _rates });
Object.defineProperty(window, '_cot',      { get: () => _cot });
Object.defineProperty(window, '_selectedPair', {
  get: () => _selectedPair,
  set: v => { _selectedPair = v; }
});
