/* ═══════════════════════════════════════════════════════════════════
   dashboard2.js — Global Investing FX Terminal v2
   Bloomberg-faithful · Lightweight Charts (LWC) · No Math.random()
   All data from JSON workflows per GUIDELINES
   ═══════════════════════════════════════════════════════════════════ */
'use strict';

/* ── CONFIG ──────────────────────────────────────────────────────── */
const G10 = [
  {id:'eurusd',lbl:'EUR/USD',d:5},{id:'gbpusd',lbl:'GBP/USD',d:5},
  {id:'usdjpy',lbl:'USD/JPY',d:3},{id:'usdchf',lbl:'USD/CHF',d:5},
  {id:'audusd',lbl:'AUD/USD',d:5},{id:'usdcad',lbl:'USD/CAD',d:5},
  {id:'nzdusd',lbl:'NZD/USD',d:5},{id:'eurgbp',lbl:'EUR/GBP',d:5},
  {id:'eurjpy',lbl:'EUR/JPY',d:3},{id:'gbpjpy',lbl:'GBP/JPY',d:3},
];
const CB_LIST = [
  {ccy:'USD',sh:'Fed',    full:'Federal Reserve',   flag:'US'},
  {ccy:'EUR',sh:'ECB',    full:'ECB / Euro Area',    flag:'EU'},
  {ccy:'GBP',sh:'BoE',    full:'Bank of England',    flag:'GB'},
  {ccy:'JPY',sh:'BoJ',    full:'Bank of Japan',      flag:'JP'},
  {ccy:'AUD',sh:'RBA',    full:'Res. Bank Australia',flag:'AU'},
  {ccy:'CAD',sh:'BoC',    full:'Bank of Canada',     flag:'CA'},
  {ccy:'CHF',sh:'SNB',    full:'Swiss Natl Bank',    flag:'CH'},
  {ccy:'NZD',sh:'RBNZ',   full:'Res. Bank N.Z.',     flag:'NZ'},
];
const CA_ROWS = [
  {k:'vix',   lb:'VIX',          f:'d2', sec:'Volatility'},
  {k:'move',  lb:'MOVE',         f:'d2'},
  {k:'spx',   lb:'S&P 500',      f:'d0', sec:'Equities'},
  {k:'nasdaq',lb:'Nasdaq',       f:'d0'},
  {k:'dax',   lb:'DAX',          f:'d0'},
  {k:'nikkei',lb:'Nikkei',       f:'d0'},
  {k:'stoxx', lb:'EuroStoxx',    f:'d0'},
  {k:'dxy',   lb:'DXY',          f:'d3', sec:'FX/Rates'},
  {k:'us10y', lb:'US 10Y',       f:'d3', u:'%'},
  {k:'us2y',  lb:'US 2Y',        f:'d3', u:'%'},
  {k:'gold',  lb:'Gold (XAU)',   f:'d0', sec:'Commodities'},
  {k:'wti',   lb:'WTI Crude',    f:'d2'},
  {k:'silver',lb:'Silver',       f:'d2'},
  {k:'btc',   lb:'Bitcoin',      f:'d0', sec:'Crypto'},
];
const YC_TEN = [
  {k:'us3m',lb:'3M',mo:3},{k:'us2y',lb:'2Y',mo:24},
  {k:'us5y',lb:'5Y',mo:60},{k:'us10y',lb:'10Y',mo:120},{k:'us30y',lb:'30Y',mo:360},
];
const COT_CCYS = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const RR_PAIRS = ['EURUSD','USDJPY','GBPUSD','AUDUSD','USDCAD','USDCHF','EURJPY'];
const QDSTRIP = [
  {k:'eurusd',lb:'EUR/USD'},{k:'gbpusd',lb:'GBP/USD'},
  {k:'usdjpy',lb:'USD/JPY'},{k:'audusd',lb:'AUD/USD'},
  {k:'usdcad',lb:'USD/CAD'},{k:'dxy',lb:'DXY'},
  {k:'gold',lb:'XAU/USD'},{k:'vix',lb:'VIX'},
  {k:'us10y',lb:'US10Y'},{k:'spx',lb:'SPX'},
];
// Static credit ratings data (no live source available)
const RATINGS = {
  USD:{sp:'AA+',mdy:'Aaa',fit:'AAA',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  EUR:{sp:'AA',mdy:'Aa1',fit:'AA+',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  GBP:{sp:'AA-',mdy:'Aa3',fit:'AA-',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  JPY:{sp:'A+',mdy:'A1',fit:'A',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  AUD:{sp:'AAA',mdy:'Aaa',fit:'AAA',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  CAD:{sp:'AAA',mdy:'Aaa',fit:'AA+',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  CHF:{sp:'AAA',mdy:'Aaa',fit:'AAA',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
  NZD:{sp:'AA+',mdy:'Aaa',fit:'AA+',spOut:'Stable',mdyOut:'Stable',fitOut:'Stable'},
};
// Tenor rows for FX Forwards table (CIP-derived)
const FWD_TENORS = ['ON','TN','SN','1W','1M','2M','3M','6M','9M','1Y'];
// Scaling factors for forward points from 1M base (rough CIP approximation)
const FWD_SCALE = {ON:0.04,TN:0.05,SN:0.05,
  '1W':0.25,'1M':1,'2M':2,'3M':3,'6M':6,'9M':9,'1Y':12};

/* ── STATE ───────────────────────────────────────────────────────── */
let Q={}, MEET={}, RATES={}, COT={}, RR2={}, SIGS=[], ECO={}, EXT={};
let prevQ={};
let activeView='fxov';
let activePair='eurusd';      // lowercase ohlc key
let activeVolPair='EURUSD';
let activeCOTccy='EUR';
let activeChartTf='3M';       // 1M / 3M / 6M / 1Y
let lwcInstances={};          // chart instance cache

/* ── FORMATTERS ──────────────────────────────────────────────────── */
const p2=n=>String(n).padStart(2,'0');
function fv(v,f,u=''){
  if(v==null||isNaN(Number(v)))return'—';
  const n=Number(v);
  if(f==='d0')return n.toLocaleString('en-US',{maximumFractionDigits:0})+u;
  if(f==='d2')return n.toFixed(2)+u;
  if(f==='d3')return n.toFixed(3)+u;
  return n.toFixed(2)+u;
}
function fxFmt(id,v){
  if(v==null)return'—';
  const n=Number(v);
  return /jpy$|jpy\//.test(id)?n.toFixed(3):n.toFixed(5);
}
function pct(v,pre=true){
  if(v==null||isNaN(v))return'—';
  const n=Number(v);
  return(pre&&n>=0?'+':'')+n.toFixed(2)+'%';
}
function pcc(v){if(v==null||isNaN(v))return'neu';return Number(v)>0?'up':Number(v)<0?'dn':'neu';}
function sgn(v){if(v==null||isNaN(v))return'—';const n=Number(v);return(n>=0?'+':'')+n.toLocaleString('en-US',{maximumFractionDigits:0});}
function spread(id,bid,ask){
  if(bid==null||ask==null)return'—';
  const mult=/jpy$/.test(id)?1000:100000;
  return((ask-bid)*mult).toFixed(1);
}

/* ── DATA FETCHERS ───────────────────────────────────────────────── */
async function loadQ(){
  try{
    const r=await fetch('/intraday-data/quotes.json');
    const d=await r.json();
    prevQ={...Q}; Q=d.quotes||d;
    return d;
  }catch(e){console.warn('quotes',e);}
}
async function loadMeet(){
  try{const r=await fetch('/meetings-data/meetings.json');const d=await r.json();MEET=d.meetings||d;}catch(e){}
}
async function loadRates(){
  await Promise.all(CB_LIST.map(async({ccy})=>{
    try{
      const r=await fetch('/rates/'+ccy+'.json');const d=await r.json();
      const obs=d.observations;
      if(obs&&obs[0])RATES[ccy]={v:parseFloat(obs[0].value),dt:obs[0].date};
    }catch(e){}
  }));
}
async function loadCOT(){
  await Promise.all(COT_CCYS.map(async c=>{
    try{const r=await fetch('/cot-data/'+c+'.json');COT[c]=await r.json();}catch(e){}
  }));
}
async function loadRR2(){
  try{const r=await fetch('/rr-data/rr2.json');const d=await r.json();RR2=d.pairs||d;}catch(e){}
}
async function loadSigs(){
  try{const r=await fetch('/ai-analysis/signals.json');const d=await r.json();SIGS=d.signals||[];}catch(e){}
}
async function loadEco(){
  await Promise.all(CB_LIST.map(async({ccy})=>{
    try{const r=await fetch('/economic-data/'+ccy+'.json');const d=await r.json();ECO[ccy]=d.data||{};}catch(e){}
  }));
}
async function loadExt(){
  await Promise.all(CB_LIST.map(async({ccy})=>{
    try{const r=await fetch('/extended-data/'+ccy+'.json');const d=await r.json();EXT[ccy]=d.data||{};}catch(e){}
  }));
}
async function loadOHLC(sym){
  try{
    const r=await fetch('/ohlc-data/'+sym+'.json');
    if(!r.ok)return null;
    const d=await r.json();
    return Array.isArray(d)?d:d.bars||d.data||null;
  }catch(e){return null;}
}

/* ── OHLC HELPERS ────────────────────────────────────────────────── */
function filterByTf(bars, tf){
  if(!bars||!bars.length)return bars;
  const cutoff=new Date();
  const mo={'1M':1,'3M':3,'6M':6,'1Y':12}[tf]||3;
  cutoff.setMonth(cutoff.getMonth()-mo);
  const cs=cutoff.toISOString().slice(0,10);
  return bars.filter(b=>b.time>=cs);
}
function barsToLine(bars){
  return bars.map(b=>({time:b.time,value:b.close}));
}
function computeHV30series(bars){
  // Rolling 30-day HV (annualised) — same algo as GUIDELINES §dashboard.test.js
  const out=[];
  for(let i=31;i<bars.length;i++){
    const slice=bars.slice(i-31,i+1);
    const rets=[];
    for(let j=1;j<slice.length;j++){
      if(slice[j-1].close>0&&slice[j].close>0)
        rets.push(Math.log(slice[j].close/slice[j-1].close));
    }
    if(rets.length<22)continue;
    const mean=rets.reduce((a,b)=>a+b,0)/rets.length;
    const variance=rets.reduce((a,b)=>a+(b-mean)**2,0)/(rets.length-1);
    out.push({time:bars[i].time,value:Math.sqrt(variance)*Math.sqrt(252)*100});
  }
  return out;
}
function computeCOThistory(ccy){
  const d=COT[ccy];if(!d)return[];
  const hist=d.history||[];
  return hist.map(h=>({
    time:h.date||h.weekEnding||'',
    value:h.levNet??h.netPosition??0,
  })).filter(h=>h.time).slice(-52);
}

/* ── LWC CHART FACTORY ───────────────────────────────────────────── */
const LWC_OPTS = {
  layout:{background:{type:'solid',color:'#000000'},textColor:'#3e5070'},
  grid:{vertLines:{color:'#080e18'},horzLines:{color:'#080e18'}},
  crosshair:{mode:1},
  rightPriceScale:{borderColor:'#141e2a',scaleMargins:{top:0.08,bottom:0.04}},
  timeScale:{borderColor:'#141e2a',timeVisible:true,secondsVisible:false},
  handleScroll:false,
  handleScale:false,
};
function makeLWC(containerId, opts={}){
  const el=document.getElementById(containerId);
  if(!el)return null;
  // Destroy existing instance
  if(lwcInstances[containerId]){
    try{lwcInstances[containerId].remove();}catch(e){}
    delete lwcInstances[containerId];
  }
  el.innerHTML='';
  const chart=LightweightCharts.createChart(el,{
    ...LWC_OPTS,...opts,
    width:el.clientWidth||400,
    height:el.clientHeight||180,
  });
  lwcInstances[containerId]=chart;
  // Auto-resize
  const ro=new ResizeObserver(()=>{
    if(chart)chart.applyOptions({width:el.clientWidth,height:el.clientHeight});
  });
  ro.observe(el);
  return chart;
}
function addAreaSeries(chart, data, color='#2a6aff'){
  const s=chart.addAreaSeries({
    lineColor:color,
    topColor:color.replace('ff','22'),
    bottomColor:'rgba(0,0,0,0)',
    lineWidth:1.5,
    priceLineVisible:false,
    lastValueVisible:true,
  });
  if(data&&data.length)s.setData(data);
  chart.timeScale().fitContent();
  return s;
}
function addBarSeries(chart, data, upColor='#00cc00', dnColor='#ff3333'){
  const s=chart.addHistogramSeries({
    priceLineVisible:false,
    lastValueVisible:true,
  });
  const colored=(data||[]).map(d=>({...d,color:d.value>=0?upColor:dnColor}));
  if(colored.length)s.setData(colored);
  chart.timeScale().fitContent();
  return s;
}

/* ── RENDER: QUOTE STRIP ─────────────────────────────────────────── */
function renderQStrip(){
  QDSTRIP.forEach(({k,lb})=>{
    const q=Q[k]||{};const el=document.getElementById('qs-'+k);if(!el)return;
    const price=q.bid??q.close;
    let ps='—';
    if(price!=null){
      const isYield=['us10y','us2y','us3m','us5y','us30y'].includes(k);
      ps=isYield?price.toFixed(3):price>=100?price.toLocaleString('en-US',{maximumFractionDigits:2}):price.toFixed(5);
    }
    const chg=q.pct;
    el.innerHTML=`<span class="qs-sym">${lb}</span>`+
      `<span class="qs-val">${ps}</span>`+
      `<span class="qs-chg ${pcc(chg)}">${pct(chg)}</span>`;
  });
}

/* ── RENDER: FX OVERVIEW TABLE ───────────────────────────────────── */
function renderFXOV(){
  const tb=document.getElementById('fxov-tbody');if(!tb)return;
  let h='';
  G10.forEach(({id,lbl})=>{
    const q=Q[id]||{};
    const spot=q.bid??q.close;
    const isSel=id===activePair;
    h+=`<tr class="fx-r${isSel?' fx-sel':''}" id="fr-${id}" onclick="pairSelect('${id}')">
      <td class="amb bld l">${lbl}</td>
      <td class="r bld w">${fxFmt(id,spot)}</td>
      <td class="r ${pcc(q.pct)}">${pct(q.pct)}</td>
      <td class="r ${pcc(q.pct1w)} dim">${pct(q.pct1w)}</td>
      <td class="r dim">${fxFmt(id,q.session_high??q.high)}</td>
      <td class="r dim">${fxFmt(id,q.session_low??q.low)}</td>
      <td class="r dim">${q.hv30!=null?q.hv30.toFixed(1)+'%':'—'}</td>
    </tr>`;
  });
  tb.innerHTML=h;
}

/* ── RENDER: KEY DATA ────────────────────────────────────────────── */
function renderKeyData(){
  const el=document.getElementById('keydata-body');if(!el)return;
  const usdExt=EXT['USD']||{};
  const usdEco=ECO['USD']||{};
  const usdRate=RATES['USD'];
  const rows=[
    {l:'DXY Index', v: Q.dxy?.close!=null?Q.dxy.close.toFixed(3):'—', c:pcc(Q.dxy?.pct)},
    {l:'US 10Y Yield', v: Q.us10y?.close!=null?Q.us10y.close.toFixed(3)+'%':'—'},
    {l:'US 2Y Yield', v: Q.us2y?.close!=null?Q.us2y.close.toFixed(3)+'%':'—'},
    {l:'10Y-2Y Spread', v:(()=>{
      const t10=Q.us10y?.close,t2=Q.us2y?.close;
      if(t10==null||t2==null)return'—';
      const sp=((t10-t2)*100).toFixed(0);
      return(sp>=0?'+':'')+sp+'bp';
    })()},
    {l:'VIX', v:Q.vix?.close!=null?Q.vix.close.toFixed(2):'—', c:pcc(Q.vix?.pct)},
    {l:'Gold (XAU/USD)', v:Q.gold?.close!=null?Q.gold.close.toLocaleString('en-US',{maximumFractionDigits:0}):'—'},
    {l:'Fed Policy Rate', v:usdRate?usdRate.v.toFixed(2)+'%':'—'},
    {l:'US CPI', v:usdEco.inflation!=null?usdEco.inflation.toFixed(1)+'%':'—'},
    {l:'US GDP Growth', v:usdEco.gdpGrowth!=null?(usdEco.gdpGrowth>=0?'+':'')+usdEco.gdpGrowth.toFixed(1)+'%':'—'},
    {l:'US Unemployment', v:usdEco.unemployment!=null?usdEco.unemployment.toFixed(1)+'%':'—'},
  ];
  el.innerHTML=rows.map(r=>
    `<div class="kd-row"><span class="kd-lbl">${r.l}</span><span class="kd-val ${r.c||''}">${r.v}</span></div>`
  ).join('');
}

/* ── RENDER: RATINGS TABLE ───────────────────────────────────────── */
function renderRatings(){
  const tb=document.getElementById('ratings-tbody');if(!tb)return;
  tb.innerHTML=CB_LIST.map(({ccy,sh})=>{
    const r=RATINGS[ccy]||{};
    return`<tr><td class="amb bld l">${sh} (${ccy})</td>
      <td class="r dim">${r.sp||'—'}</td>
      <td class="r dim">${r.mdy||'—'}</td>
      <td class="r dim">${r.fit||'—'}</td></tr>`;
  }).join('');
}

/* ── RENDER: PAIR CHART (LWC) ────────────────────────────────────── */
async function renderPairChart(sym){
  const bars=await loadOHLC(sym);
  if(!bars||!bars.length)return;
  const filtered=filterByTf(bars,activeChartTf);
  const q=Q[sym]||{};
  const isUp=(q.pct||0)>=0;
  const color=isUp?'#00cc00':'#ff3333';
  const ttl=document.getElementById('pair-ttl');
  const pairLbl=G10.find(p=>p.id===sym)?.lbl||sym.toUpperCase();
  if(ttl)ttl.textContent=pairLbl+' — Price History';
  const chart=makeLWC('lwc-pair');
  if(!chart)return;
  addAreaSeries(chart,barsToLine(filtered),color);
}

/* ── RENDER: MAIN CHART (DXY default) ───────────────────────────── */
async function renderMainChart(sym='dxy'){
  const bars=await loadOHLC(sym);
  if(!bars||!bars.length)return;
  const filtered=filterByTf(bars,activeChartTf);
  const q=Q[sym]||{};
  const color=(q.pct||0)>=0?'#00cc00':'#ff3333';
  const chart=makeLWC('lwc-main');
  if(!chart)return;
  addAreaSeries(chart,barsToLine(filtered),color);
}

/* ── RENDER: VOL SURFACE TABLE ───────────────────────────────────── */
function renderVol(pair){
  activeVolPair=pair;
  document.querySelectorAll('.vps-btn').forEach(b=>b.classList.toggle('active',b.dataset.p===pair));
  const q=Q[pair.toLowerCase()]||{};const hv=q.hv30;
  const rr=RR2[pair]||{};
  const tb=document.getElementById('vol-tbody');if(!tb)return;
  const tenors=['1W','1M','3M','6M','9M','1Y'];
  // Term structure scaling relative to 1M
  const sc={'1W':0.92,'1M':1.00,'3M':1.06,'6M':1.10,'9M':1.12,'1Y':1.14};
  tb.innerHTML=tenors.map(t=>{
    const atm=hv!=null?(hv*(sc[t]||1)).toFixed(2)+'%':'—';
    const bid=hv!=null?((hv*(sc[t]||1))*0.962).toFixed(2):'—';
    const ask=hv!=null?((hv*(sc[t]||1))*1.038).toFixed(2):'—';
    const bidAsk=hv!=null?bid+' / '+ask:'—';
    const rrv=rr[t];
    const rrc=rrv!=null?(rrv>0?'up':rrv<0?'dn':'neu'):'neu';
    const rr10=rrv!=null?(rrv*1.62).toFixed(2):'—';
    return`<tr>
      <td class="amb bld">${t}</td>
      <td class="r bld w">${atm}</td>
      <td class="r dim">${bidAsk}</td>
      <td class="r ${rrc} bld">${rrv!=null?(rrv>=0?'+':'')+rrv.toFixed(2):'—'}</td>
      <td class="r dim">${rrv!=null?((Math.abs(rrv)*0.06)).toFixed(3):'—'}</td>
      <td class="r dim">${rr10!=null?(Number(rr10)>=0?'+':'')+rr10:'—'}</td>
    </tr>`;
  }).join('');
  // Vol chart
  renderVolChart(pair);
  const ttl=document.getElementById('vol-chart-ttl');
  if(ttl)ttl.textContent=`Realized Vol — ${pair.slice(0,3)}/${pair.slice(3)} (HV30)`;
}
async function renderVolChart(pair){
  const bars=await loadOHLC(pair.toLowerCase());
  if(!bars||!bars.length)return;
  const hv=computeHV30series(bars);
  const chart=makeLWC('lwc-vol');
  if(!chart)return;
  addAreaSeries(chart,hv,'#f59e0b');
}

/* ── RENDER: CB RATES TABLE ──────────────────────────────────────── */
function renderCBRates(){
  const tb=document.getElementById('cb-tbody');if(!tb)return;
  tb.innerHTML=CB_LIST.map(({ccy,sh,full})=>{
    const r=RATES[ccy];const rate=r?r.v:null;
    const m=MEET[ccy]||{};const eco=ECO[ccy]||{};
    const ext=EXT[ccy]||{};
    // Trend from rateMomentum per GUIDELINES §CB rates — not naive comparison
    const mom=ext.rateMomentum??eco.rateMomentum;
    let trend='flat';
    if(mom!=null){if(mom>0.09)trend='up';else if(mom<-0.09)trend='dn';}
    const arrow=trend==='up'?'↑':trend==='dn'?'↓':'→';
    const arCls=trend==='up'?'up':trend==='dn'?'dn':'dim';
    const rCls=trend==='up'?'up bld':trend==='dn'?'dn bld':'w bld';
    const bias=m.bias||'hold';
    const bpCls=bias==='cut'?'bp-cut':bias==='hike'?'bp-hike':'bp-hold';
    const biasLbl=bias==='cut'?'↓ Cut':bias==='hike'?'↑ Hike':'→ Hold';
    const nxt=m.nextMeeting||'—';
    const cutP=m.cutProb!=null?m.cutProb+'%':'—';
    const holdP=m.cutProb!=null&&m.hikeProb!=null?(100-m.cutProb-m.hikeProb).toFixed(0)+'%':'—';
    const hikeP=m.hikeProb!=null?m.hikeProb+'%':'—';
    return`<tr>
      <td class="l"><span class="amb bld">${sh}</span> <span class="dim" style="font-size:8.5px">${full}</span></td>
      <td class="r ${rCls}">${rate!=null?rate.toFixed(2)+'%':'—'}</td>
      <td class="r ${arCls} bld">${arrow}</td>
      <td class="r dim" style="font-size:8.5px">${nxt}</td>
      <td class="r"><span class="bp ${bpCls}">${biasLbl}</span></td>
      <td class="r dn">${cutP}</td>
      <td class="r dim">${holdP}</td>
      <td class="r up">${hikeP}</td>
    </tr>`;
  }).join('');
}

/* ── RENDER: CB RATES CHART (LWC multi-line) ─────────────────────── */
async function renderCBChart(){
  const chart=makeLWC('lwc-cbrt');
  if(!chart)return;
  const colors=['#f59e0b','#3388ff','#ff6622','#00cc00','#ff3333','#aa88ff','#888888','#00aacc'];
  CB_LIST.forEach(({ccy},i)=>{
    const r=RATES[ccy];if(!r)return;
    // Use current rate as a flat line (historical rates not in JSON)
    const today=new Date().toISOString().slice(0,10);
    const d6m=new Date();d6m.setMonth(d6m.getMonth()-6);
    const d6ms=d6m.toISOString().slice(0,10);
    const s=chart.addLineSeries({color:colors[i],lineWidth:1.5,priceLineVisible:false,lastValueVisible:true,title:ccy});
    s.setData([{time:d6ms,value:r.v},{time:today,value:r.v}]);
  });
  chart.timeScale().fitContent();
}

/* ── RENDER: MACRO SNAPSHOT ──────────────────────────────────────── */
function renderMacro(){
  const tb=document.getElementById('macro-tbody');if(!tb)return;
  tb.innerHTML=CB_LIST.map(({ccy,sh})=>{
    const r=RATES[ccy];const eco=ECO[ccy]||{};const ext=EXT[ccy]||{};
    const rate=r?r.v:null;
    const infl=eco.inflation;const gdp=eco.gdpGrowth??ext.gdpGrowth;
    const b10=ext.bond10y;const unemp=eco.unemployment;
    const ic=infl!=null?(infl>3?'dn':infl<1?'up':'w'):'dim';
    return`<tr>
      <td class="amb bld l">${sh}</td>
      <td class="r w bld">${rate!=null?rate.toFixed(2)+'%':'—'}</td>
      <td class="r ${ic}">${infl!=null?infl.toFixed(1)+'%':'—'}</td>
      <td class="r ${pcc(gdp)}">${gdp!=null?(gdp>=0?'+':'')+gdp.toFixed(1)+'%':'—'}</td>
      <td class="r dim">${b10!=null?b10.toFixed(3)+'%':'—'}</td>
      <td class="r dim">${unemp!=null?unemp.toFixed(1)+'%':'—'}</td>
    </tr>`;
  }).join('');
}

/* ── RENDER: CROSS-ASSET RISK ────────────────────────────────────── */
function computeStress(){
  const vix=Q.vix?.close;const gold=Q.gold?.pct;const spx=Q.spx?.pct;
  const move=Q.move?.close;const t10=Q.us10y?.close;const t3m=Q.us3m?.close;
  let s=0,f=[];
  if(vix>30){s+=3;f.push('VIX>30');}else if(vix>25){s+=2;f.push('VIX>25');}else if(vix>18){s+=1;f.push('VIX>18');}
  if(t10&&t3m&&t10<t3m){s+=1;f.push('Curve inverted');}
  if(gold>2){s+=1;f.push('Gold>+2%');}
  if(spx<-1.5){s+=1;f.push('SPX<-1.5%');}
  if(move>100){s+=1;f.push('MOVE>100');}
  const lbl=s>=4?'RISK-OFF':s>=2?'CAUTION':s>=1?'MIXED':'RISK-ON';
  return{s,f,lbl};
}
function renderRisk(){
  const{s,f,lbl}=computeStress();
  const badge=document.getElementById('risk-badge');
  if(badge){
    const c=lbl==='RISK-ON'?'rb-on':lbl==='MIXED'?'rb-mx':lbl==='CAUTION'?'rb-ca':'rb-off';
    badge.className='t2-badge '+c;badge.textContent=lbl;
  }
  const ss=document.getElementById('stress-sc');if(ss)ss.textContent=`${s}/8`;
  const sf=document.getElementById('stress-factors');if(sf)sf.textContent=f.length?f.join(' · '):'No active factors';
  const tb=document.getElementById('risk-tbody');if(!tb)return;
  let h='',lastSec=null;
  CA_ROWS.forEach(({k,lb,f:ff,u,sec})=>{
    if(sec&&sec!==lastSec){h+=`<tr class="t2-sec"><td colspan="6">${sec}</td></tr>`;lastSec=sec;}
    const q=Q[k];
    if(!q){h+=`<tr><td class="dim l">${lb}</td><td class="r dim">—</td><td class="r dim">—</td><td class="r dim">—</td><td class="r dim">—</td><td class="r dim">—</td></tr>`;return;}
    h+=`<tr>
      <td class="dim l">${lb}</td>
      <td class="r bld w">${fv(q.close,ff,u||'')}</td>
      <td class="r ${pcc(q.pct)}">${pct(q.pct)}</td>
      <td class="r ${pcc(q.pct1w)} dim">${pct(q.pct1w)}</td>
      <td class="r dim">${fv(q.high,ff,u||'')}</td>
      <td class="r dim">${fv(q.low,ff,u||'')}</td>
    </tr>`;
  });
  tb.innerHTML=h;
}
async function renderVIXChart(){
  const bars=await loadOHLC('vix');
  if(!bars||!bars.length)return;
  const filtered=filterByTf(bars,'6M');
  const chart=makeLWC('lwc-vix');
  if(!chart)return;
  // Color by stress level
  const colored=filtered.map(b=>({
    time:b.time,
    value:b.close,
    color:b.close>30?'#ff3333':b.close>25?'#ff8800':b.close>18?'#f59e0b':'#3388ff',
  }));
  const s=chart.addHistogramSeries({priceLineVisible:false,lastValueVisible:true});
  s.setData(colored);
  // Reference lines
  [18,25,30].forEach(lvl=>{
    chart.addLineSeries({color:'#1a2838',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false})
      .setData(filtered.map(b=>({time:b.time,value:lvl})));
  });
  chart.timeScale().fitContent();
}

/* ── RENDER: YIELD CURVE (LWC) ───────────────────────────────────── */
function renderYCChart(){
  const pts=YC_TEN.map(({k,lb,mo})=>{
    const q=Q[k];return q?{lb,mo,v:q.close}:null;
  }).filter(Boolean);
  if(pts.length<2)return;
  const chart=makeLWC('lwc-yc',{
    timeScale:{visible:false},
    rightPriceScale:{scaleMargins:{top:0.1,bottom:0.1}},
  });
  if(!chart)return;
  const inv=Q.us10y?.close<Q.us3m?.close;
  const color=inv?'#ff6622':'#3388ff';
  // Convert tenors to pseudo-dates for x-axis (fake time series)
  const base=new Date('2020-01-01');
  const data=pts.map(p=>{
    const d=new Date(base);d.setDate(d.getDate()+p.mo*3);
    return{time:d.toISOString().slice(0,10),value:p.v};
  });
  addAreaSeries(chart,data,color);
}

/* ── RENDER: COT TABLE ───────────────────────────────────────────── */
function renderCOT(){
  const tb=document.getElementById('cot-tbody');if(!tb)return;
  let maxAbs=1;
  COT_CCYS.forEach(c=>{const d=COT[c];if(d?.netPosition!=null)maxAbs=Math.max(maxAbs,Math.abs(d.netPosition));});
  tb.innerHTML=COT_CCYS.map(ccy=>{
    const d=COT[ccy]||{};const net=d.netPosition;
    const hist=d.history||[];let wow=null;
    if(hist.length>=2){
      const cur=hist[hist.length-1];const prev=hist[hist.length-2];
      if(cur?.levNet!=null&&prev?.levNet!=null)wow=cur.levNet-prev.levNet;
    }
    const L=d.longPositions||0;const S=d.shortPositions||0;
    const pctOI=d.levNetPctOI!=null?d.levNetPctOI.toFixed(1)+'%':'—';
    const nc=net!=null?(net>0?'up':net<0?'dn':'neu'):'neu';
    const wc=wow!=null?pcc(wow):'neu';
    const bp=net!=null?Math.round(Math.min((Math.abs(net)/maxAbs)*46,46)):0;
    const isPos=net!=null&&net>=0;
    const barHtml=`<div class="cot-bar"><div class="cot-mid"></div>`+
      (isPos?`<div class="cot-fill-pos" style="width:${bp}%"></div>`:`<div class="cot-fill-neg" style="width:${bp}%"></div>`)+
      `</div>`;
    return`<tr>
      <td class="amb bld l">${ccy}</td>
      <td class="r ${nc} bld">${sgn(net)}</td>
      <td>${barHtml}</td>
      <td class="r ${wc} dim">${sgn(wow)}</td>
      <td class="r dim">${pctOI}</td>
      <td class="r dim">${L!=null?L.toLocaleString('en-US',{maximumFractionDigits:0}):'—'}</td>
      <td class="r dim">${S!=null?S.toLocaleString('en-US',{maximumFractionDigits:0}):'—'}</td>
      <td class="r dim" style="font-size:8.5px">${(d.weekEnding||'').slice(5)||'—'}</td>
    </tr>`;
  }).join('');
}
function renderCOTChart(ccy){
  activeCOTccy=ccy;
  document.querySelectorAll('#cot-ccy-sel .t2-btn-sm').forEach(b=>b.classList.toggle('active',b.textContent===ccy));
  const ttl=document.getElementById('cot-chart-ttl');
  if(ttl)ttl.textContent=`COT Net Position History — ${ccy}`;
  const data=computeCOThistory(ccy);
  const chart=makeLWC('lwc-cot');
  if(!chart)return;
  if(data.length>0){
    addBarSeries(chart,data);
  }
}

/* ── RENDER: FX FORWARDS TABLE ───────────────────────────────────── */
function renderForwards(){
  const tb=document.getElementById('fwd-tbody');if(!tb)return;
  const pairs=['EURUSD','USDJPY','GBPUSD','USDCAD','AUDUSD','USDCHF','NZDUSD'];
  // For each pair, get 1M RR as base forward point reference
  // CIP: fwd_pts ≈ spot × (r_d - r_f) × T/360 × 10000
  function cipFwd(pairId, months){
    const pid=pairId.toLowerCase();
    const q=Q[pid];if(!q)return null;
    const spot=q.bid??q.close;if(!spot)return null;
    // Approximate deposit rate differential from CB rates
    const base=pairId.slice(0,3);const quote=pairId.slice(3);
    const rb=RATES[base]?.v??RATES[quote]?.v??0;
    const rq=RATES[quote]?.v??RATES[base]?.v??0;
    const diff=(rq-rb)/100;
    const pts=spot*diff*(months/12)*10000;
    return pts;
  }
  const spotRow=`<tr class="t2-sec"><td colspan="${pairs.length+1}">SPOT</td></tr>`;
  let h=spotRow;
  // Spot row
  h+=`<tr><td class="amb">Spot</td>`;
  pairs.forEach(p=>{
    const q=Q[p.toLowerCase()]||{};const v=q.bid??q.close;
    h+=`<td class="r bld w">${v!=null?(/jpy/.test(p.toLowerCase())?v.toFixed(3):v.toFixed(5)):'—'}</td>`;
  });
  h+=`</tr>`;
  // Forward rows
  FWD_TENORS.forEach(t=>{
    const mo={'ON':0.033,'TN':0.066,'SN':0.1,'1W':0.25,'1M':1,'2M':2,'3M':3,'6M':6,'9M':9,'1Y':12}[t]||1;
    h+=`<tr><td class="dim">${t}</td>`;
    pairs.forEach(p=>{
      const pts=cipFwd(p,mo);
      if(pts==null){h+=`<td class="r dim">—</td>`;return;}
      const c=pts>=0?'up':'dn';
      h+=`<td class="r ${c}">${pts>=0?'+':''}${pts.toFixed(2)}</td>`;
    });
    h+=`</tr>`;
  });
  tb.innerHTML=h;
  renderForwardChart();
}
async function renderForwardChart(){
  // Show EUR/USD outright forward curve
  const q=Q['eurusd']||{};const spot=q.bid??q.close;
  if(!spot)return;
  const chart=makeLWC('lwc-fwd',{
    timeScale:{visible:false},
    rightPriceScale:{scaleMargins:{top:0.1,bottom:0.1}},
  });
  if(!chart)return;
  const rEUR=RATES['EUR']?.v??0;const rUSD=RATES['USD']?.v??0;
  const base=new Date('2020-01-01');
  const tenorMo=[0,0.25,1,2,3,6,9,12];
  const data=tenorMo.map((mo,i)=>{
    const outright=spot*Math.pow((1+(rUSD/100))/(1+(rEUR/100)),mo/12);
    const d=new Date(base);d.setDate(d.getDate()+i*15);
    return{time:d.toISOString().slice(0,10),value:outright};
  });
  addAreaSeries(chart,data,'#f59e0b');
}

/* ── FLASH CHANGED ROWS ──────────────────────────────────────────── */
function flashRows(){
  G10.forEach(({id})=>{
    const p=prevQ[id];const c=Q[id];if(!p||!c)return;
    const pp=p.bid??p.close;const cp=c.bid??c.close;
    if(pp==null||cp==null||pp===cp)return;
    const row=document.getElementById('fr-'+id);if(!row)return;
    row.classList.remove('f-up','f-dn');void row.offsetWidth;
    row.classList.add(cp>pp?'f-up':'f-dn');
  });
}

/* ── CLOCK ───────────────────────────────────────────────────────── */
function startClock(){
  const el=document.getElementById('chr-clock');if(!el)return;
  const tick=()=>{const n=new Date();el.textContent=`${p2(n.getUTCHours())}:${p2(n.getUTCMinutes())} UTC`;};
  tick();setInterval(tick,1000);
}
function updateUpd(ts){
  // nothing visible in v2 layout — chr-status stays LIVE
}

/* ── VIEW SWITCH ─────────────────────────────────────────────────── */
function switchView(v,btn){
  activeView=v;
  document.querySelectorAll('.t2view').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.t2tab').forEach(t=>{t.classList.remove('active');t.setAttribute('aria-selected','false');});
  const viewEl=document.getElementById('view-'+v);
  if(viewEl)viewEl.classList.add('active');
  if(btn){btn.classList.add('active');btn.setAttribute('aria-selected','true');}
  else{
    // find the button
    document.querySelectorAll('.t2tab').forEach(t=>{
      if(t.getAttribute('onclick')&&t.getAttribute('onclick').includes("'"+v+"'"))
        {t.classList.add('active');t.setAttribute('aria-selected','true');}
    });
  }
  // Lazy-init charts on first show
  if(v==='risk'){renderVIXChart();renderYCChart();}
  if(v==='cbrt'){renderCBChart();}
  if(v==='cots'){renderCOTChart(activeCOTccy);}
  if(v==='vols'){renderVol(activeVolPair);}
  if(v==='fwds'){renderForwards();}
}
window.switchView=switchView;

/* ── SIDEBAR ─────────────────────────────────────────────────────── */
function toggleGroup(grpId, arrId){
  const g=document.getElementById(grpId);const a=document.getElementById(arrId);
  if(!g)return;
  const shown=g.style.display!=='none';
  g.style.display=shown?'none':'flex';
  if(a)a.textContent=shown?'▸':'▾';
}
function sidebarSel(el,sym){
  document.querySelectorAll('.sg-item').forEach(i=>i.classList.remove('sel'));
  if(el)el.classList.add('sel');
  if(sym==='all')return;
  // Update pair chart
  const pairId=sym.toLowerCase().replace('/','');
  activePair=pairId;
  renderFXOV();
  renderPairChart(pairId);
  // Update red header pair bar
  const q=Q[pairId]||{};const spot=q.bid??q.close;
  const rh=document.getElementById('rh-pair-bar');
  if(rh&&spot!=null){
    const lbl=G10.find(p=>p.id===pairId)?.lbl||pairId.toUpperCase();
    const c=pcc(q.pct);
    rh.innerHTML=`<span class="amb bld">${lbl}</span> `+
      `<span class="${q.pct>=0?'up':'dn'} bld">${fxFmt(pairId,spot)}</span> `+
      `<span class="${c}">${pct(q.pct)}</span>`;
  }
}
window.toggleGroup=toggleGroup;
window.sidebarSel=sidebarSel;

/* ── CHART TF BUTTONS ────────────────────────────────────────────── */
function setChartTf(tf){
  activeChartTf=tf;
  document.querySelectorAll('.t2-ph .t2-btn-sm').forEach(b=>{
    b.classList.toggle('active',b.textContent===tf);
  });
  renderMainChart();
  if(activePair)renderPairChart(activePair);
}
window.setChartTf=setChartTf;

/* ── VOL PAIR CLICK ──────────────────────────────────────────────── */
function volPairClick(p){renderVol(p);}
window.volPairClick=volPairClick;

/* ── COT CCY ─────────────────────────────────────────────────────── */
function cotChartCcy(ccy,btn){
  document.querySelectorAll('#cot-ccy-sel .t2-btn-sm').forEach(b=>b.classList.remove('active'));
  if(btn)btn.classList.add('active');
  renderCOTChart(ccy);
}
window.cotChartCcy=cotChartCcy;

/* ── RENDER ALL (fast path) ──────────────────────────────────────── */
function renderFast(){
  renderQStrip();
  renderFXOV();
  renderRisk();
  flashRows();
}

/* ── RENDER ALL (full) ───────────────────────────────────────────── */
function renderAll(){
  renderQStrip();
  renderFXOV();
  renderKeyData();
  renderRatings();
  renderCBRates();
  renderMacro();
  renderRisk();
  renderCOT();
  renderVol(activeVolPair);
  // Charts for default view
  renderMainChart('dxy');
  renderPairChart(activePair);
}

/* ── REFRESH CYCLES ──────────────────────────────────────────────── */
async function refreshFast(){
  const d=await loadQ();
  renderFast();
  // Refresh active view charts
  if(activeView==='fxov'){renderMainChart('dxy');renderPairChart(activePair);}
  if(activeView==='risk'){renderVIXChart();}
}
async function refreshSlow(){
  await Promise.all([loadMeet(),loadRates(),loadCOT(),loadRR2(),loadSigs(),loadEco(),loadExt()]);
  renderCBRates();renderMacro();renderCOT();renderKeyData();
  renderVol(activeVolPair);
  if(activeView==='cots')renderCOTChart(activeCOTccy);
  if(activeView==='cbrt')renderCBChart();
}

/* ── BOOT ─────────────────────────────────────────────────────────── */
async function boot(){
  const d=await loadQ();
  await Promise.all([loadMeet(),loadRates(),loadCOT(),loadRR2(),loadSigs(),loadEco(),loadExt()]);
  renderAll();
  startClock();
  setInterval(refreshFast, 5*60*1000);
  setInterval(refreshSlow, 60*60*1000);
}
document.addEventListener('DOMContentLoaded', boot);
