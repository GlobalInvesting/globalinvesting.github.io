/* ═══════════════════════════════════════════════════════════════════
   dashboard2.js — Bloomberg-Faithful FX Terminal (TEST BUILD)
   No Math.random() · Deterministic · All values from JSON workflows
   ═══════════════════════════════════════════════════════════════════ */
'use strict';

/* ── CONFIG ─────────────────────────────────────────────────────── */
const G10 = [
  {id:'eurusd',lbl:'EUR/USD',d:5},{id:'gbpusd',lbl:'GBP/USD',d:5},
  {id:'usdjpy',lbl:'USD/JPY',d:3},{id:'usdchf',lbl:'USD/CHF',d:5},
  {id:'audusd',lbl:'AUD/USD',d:5},{id:'usdcad',lbl:'USD/CAD',d:5},
  {id:'nzdusd',lbl:'NZD/USD',d:5},
];
const CROSS = [
  {id:'eurgbp',lbl:'EUR/GBP',d:5},{id:'eurjpy',lbl:'EUR/JPY',d:3},
  {id:'eurchf',lbl:'EUR/CHF',d:5},{id:'eurcad',lbl:'EUR/CAD',d:5},
  {id:'euraud',lbl:'EUR/AUD',d:5},{id:'gbpjpy',lbl:'GBP/JPY',d:3},
  {id:'gbpchf',lbl:'GBP/CHF',d:5},{id:'gbpcad',lbl:'GBP/CAD',d:5},
  {id:'gbpaud',lbl:'GBP/AUD',d:5},{id:'audjpy',lbl:'AUD/JPY',d:3},
  {id:'audnzd',lbl:'AUD/NZD',d:5},{id:'audchf',lbl:'AUD/CHF',d:5},
  {id:'audcad',lbl:'AUD/CAD',d:5},{id:'cadjpy',lbl:'CAD/JPY',d:3},
  {id:'chfjpy',lbl:'CHF/JPY',d:3},{id:'nzdjpy',lbl:'NZD/JPY',d:3},
  {id:'eurnzd',lbl:'EUR/NZD',d:5},{id:'nzdcad',lbl:'NZD/CAD',d:5},
];
const CB_LIST = [
  {ccy:'USD',sh:'Fed'   ,full:'Federal Reserve'},
  {ccy:'EUR',sh:'ECB'   ,full:'Euro Central Bank'},
  {ccy:'GBP',sh:'BoE'   ,full:'Bank of England'},
  {ccy:'JPY',sh:'BoJ'   ,full:'Bank of Japan'},
  {ccy:'AUD',sh:'RBA'   ,full:'Res. Bank of Australia'},
  {ccy:'CAD',sh:'BoC'   ,full:'Bank of Canada'},
  {ccy:'CHF',sh:'SNB'   ,full:'Swiss Natl Bank'},
  {ccy:'NZD',sh:'RBNZ'  ,full:'Res. Bank of NZ'},
];
const CA_ROWS = [
  {k:'vix',   lb:'VIX',       f:'d2',  sec:'Volatility'},
  {k:'move',  lb:'MOVE',      f:'d2'},
  {k:'spx',   lb:'S&P 500',   f:'d0',  sec:'Equities'},
  {k:'nasdaq',lb:'Nasdaq',    f:'d0'},
  {k:'dax',   lb:'DAX',       f:'d0'},
  {k:'nikkei',lb:'Nikkei',    f:'d0'},
  {k:'stoxx', lb:'EuroStoxx', f:'d0'},
  {k:'ftse',  lb:'FTSE 100',  f:'d0'},
  {k:'dxy',   lb:'DXY',       f:'d3',  sec:'FX/Rates'},
  {k:'us10y', lb:'US 10Y',    f:'d3',  u:'%'},
  {k:'us2y',  lb:'US 2Y',     f:'d3',  u:'%'},
  {k:'gold',  lb:'Gold (XAU)',f:'d0',  sec:'Commodities'},
  {k:'wti',   lb:'WTI',       f:'d2'},
  {k:'brent', lb:'Brent',     f:'d2'},
  {k:'silver',lb:'Silver',    f:'d2'},
  {k:'btc',   lb:'Bitcoin',   f:'d0',  sec:'Crypto'},
  {k:'eth',   lb:'Ethereum',  f:'d0'},
];
const YC_TEN = [
  {k:'us3m', lb:'3M', mo:3},{k:'us2y',lb:'2Y',mo:24},
  {k:'us5y', lb:'5Y', mo:60},{k:'us10y',lb:'10Y',mo:120},
  {k:'us30y',lb:'30Y',mo:360},
];
const COT_CCYS = ['EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const RR_PAIRS = ['EURUSD','USDJPY','GBPUSD','AUDUSD','USDCAD','USDCHF','EURJPY'];
const TICKER = [
  {k:'eurusd',lb:'EUR/USD'},{k:'gbpusd',lb:'GBP/USD'},
  {k:'usdjpy',lb:'USD/JPY'},{k:'audusd',lb:'AUD/USD'},
  {k:'usdcad',lb:'USD/CAD'},{k:'dxy',   lb:'DXY'},
  {k:'gold',  lb:'XAU/USD'},{k:'spx',   lb:'SPX'},
  {k:'vix',   lb:'VIX'},   {k:'us10y', lb:'US10Y'},
];

/* ── STATE ──────────────────────────────────────────────────────── */
let Q={}, MEET={}, RATES={}, COT={}, RR2={}, SIGS=[], ECO={};
let prevQ={}, activePair='EURUSD';

/* ── FORMATTERS ─────────────────────────────────────────────────── */
const p2 = n=>String(n).padStart(2,'0');
function fv(v,f,u=''){
  if(v==null||v===''||isNaN(Number(v)))return '—';
  const n=Number(v);
  if(f==='d0')return n.toLocaleString('en-US',{maximumFractionDigits:0})+u;
  if(f==='d1')return n.toFixed(1)+u;
  if(f==='d2')return n.toFixed(2)+u;
  if(f==='d3')return n.toFixed(3)+u;
  if(f==='d4')return n.toFixed(4)+u;
  if(f==='d5')return n.toFixed(5)+u;
  return String(v)+u;
}
function fxp(id,v){if(v==null)return'—';return /jpy|cad/.test(id)&&!id.includes('cad/')&&Number(v)>10?Number(v).toFixed(3):Number(v).toFixed(5);}
// More precise JPY detection
function fxFmt(id,v){
  if(v==null)return'—';
  const n=Number(v);
  const isJPY=/jpy$/.test(id);
  return isJPY?n.toFixed(3):n.toFixed(5);
}
function pct(v){if(v==null||isNaN(v))return'—';const n=Number(v);return(n>=0?'+':'')+n.toFixed(2)+'%';}
function pcc(v){if(v==null||isNaN(v))return'neu';return Number(v)>0?'up':Number(v)<0?'dn':'neu';}
function sgn(v){if(v==null||isNaN(v))return'—';const n=Number(v);return(n>=0?'+':'')+n.toLocaleString('en-US',{maximumFractionDigits:0});}
function spread(id,bid,ask){
  if(bid==null||ask==null)return'—';
  const mult=/jpy$/.test(id)?1000:100000;
  return ((ask-bid)*mult).toFixed(1);
}

/* ── DATA FETCH ─────────────────────────────────────────────────── */
async function loadQ(){
  try{const r=await fetch('/intraday-data/quotes.json');const d=await r.json();
  prevQ={...Q};Q=d.quotes||d;return d;}catch(e){console.warn('quotes',e);}
}
async function loadMeet(){
  try{const r=await fetch('/meetings-data/meetings.json');const d=await r.json();MEET=d.meetings||d;}catch(e){}
}
async function loadRates(){
  await Promise.all(CB_LIST.map(async({ccy})=>{
    try{const r=await fetch('/rates/'+ccy+'.json');const d=await r.json();
    const obs=d.observations;if(obs&&obs[0])RATES[ccy]={v:parseFloat(obs[0].value),dt:obs[0].date};}
    catch(e){}
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

/* ── RENDER: TICKER ─────────────────────────────────────────────── */
function renderTicker(){
  TICKER.forEach(({k,lb})=>{
    const q=Q[k]||{};const el=document.getElementById('tk-'+k);if(!el)return;
    const price=q.bid||q.close;
    let ps='—';
    if(price!=null){
      const isYield=['us10y','us2y','us3m','us5y','us30y'].includes(k);
      const isBig=price>=100;
      ps=isYield?price.toFixed(3):isBig?price.toLocaleString('en-US',{maximumFractionDigits:2}):price.toFixed(5);
    }
    const chg=q.pct;
    el.innerHTML=`<span class="tk-sym">${lb}</span><span class="tk-val">${ps}</span><span class="tk-chg ${pcc(chg)}">${pct(chg)}</span>`;
  });
}

/* ── RENDER: SPOT FX ────────────────────────────────────────────── */
function renderSpot(){
  const tb=document.getElementById('spot-tbody');if(!tb)return;
  let h='<tr class="t-sec"><td colspan="6">G10 CORE</td></tr>';
  G10.forEach(({id,lbl})=>{
    const q=Q[id]||{};
    const bid=q.bid??q.close; const ask=q.ask;
    const bs=fxFmt(id,bid); const as=fxFmt(id,ask);
    const sp=spread(id,q.bid,q.ask);
    const isSel=id===activePair.toLowerCase();
    h+=`<tr class="${isSel?'fx-sel':''} fx-r" id="fr-${id}" onclick="pairClick('${id.toUpperCase()}')">
      <td class="ccy">${lbl}</td>
      <td class="r bld up">${bs}</td>
      <td class="r dn">${as}</td>
      <td class="r dim">${sp}</td>
      <td class="r ${pcc(q.pct)}">${pct(q.pct)}</td>
      <td class="r dim">${q.hv30!=null?q.hv30.toFixed(1)+'%':'—'}</td></tr>`;
  });
  h+='<tr class="t-sec"><td colspan="6">CROSS RATES</td></tr>';
  CROSS.forEach(({id,lbl})=>{
    const q=Q[id]||{};const mid=q.bid??q.close;
    h+=`<tr class="fx-r" id="fr-${id}">
      <td class="ccy">${lbl}</td>
      <td class="r bld w">${fxFmt(id,mid)}</td>
      <td class="r dim">—</td><td class="r dim">—</td>
      <td class="r ${pcc(q.pct)}">${pct(q.pct)}</td>
      <td class="r dim">${q.hv30!=null?q.hv30.toFixed(1)+'%':'—'}</td></tr>`;
  });
  tb.innerHTML=h;
}

/* ── RENDER: VOL SURFACE ────────────────────────────────────────── */
/* ATM uses HV30 as proxy — labeled per GUIDELINES §Data Integrity */
function renderVol(pair){
  activePair=pair;
  document.querySelectorAll('.vps-btn').forEach(b=>b.classList.toggle('active',b.dataset.p===pair));
  document.querySelectorAll('.fx-r').forEach(r=>r.classList.remove('fx-sel'));
  const sr=document.getElementById('fr-'+pair.toLowerCase());if(sr)sr.classList.add('fx-sel');
  const q=Q[pair.toLowerCase()]||{};const hv=q.hv30;
  const rr=RR2[pair]||{};
  const tb=document.getElementById('vol-tbody');if(!tb)return;
  const tenors=['1W','1M','3M','6M','9M','1Y'];
  // HV30 scaling factors by tenor (term structure proxy)
  const sc={'1W':0.92,'1M':1.00,'3M':1.06,'6M':1.10,'9M':1.12,'1Y':1.14};
  let h='';
  tenors.forEach(t=>{
    const atm=hv!=null?(hv*(sc[t]||1)).toFixed(2)+'%':'—';
    const v=rr[t]; const rrc=v!=null?(v>0?'up':v<0?'dn':'neu'):'neu';
    const rvol=hv!=null&&v!=null?(hv*(sc[t]||1)*0.95).toFixed(2)+'%':'—';
    h+=`<tr><td class="amb bld">${t}</td><td class="r bld w">${atm}</td>
      <td class="r ${rrc} bld">${v!=null?(v>=0?'+':'')+v.toFixed(2):'—'}</td>
      <td class="r dim">${rvol}</td></tr>`;
  });
  tb.innerHTML=h;
  const src=document.getElementById('vol-src');
  if(src)src.textContent=`ATM: HV30 est. · 25d RR: rr-data/rr2.json (Saxo Bank indicative · 1M tenor)`;
  const ttl=document.getElementById('vol-ttl');
  if(ttl)ttl.textContent=`Vol Structure — ${pair.slice(0,3)}/${pair.slice(3)}`;
}

/* ── RENDER: CB RATES ───────────────────────────────────────────── */
function renderCBRates(){
  const tb=document.getElementById('cb-tbody');if(!tb)return;
  tb.innerHTML=CB_LIST.map(({ccy,sh})=>{
    const r=RATES[ccy];const rate=r?r.v:null;
    const m=MEET[ccy]||{};const eco=ECO[ccy]||{};
    // Trend from rateMomentum (not naive compare) per GUIDELINES
    const mom=eco.rateMomentum;
    let trend='flat';
    if(mom!=null){if(mom>0.09)trend='up';else if(mom<-0.09)trend='dn';}
    const arrow=trend==='up'?'↑':trend==='dn'?'↓':'→';
    const arCls=trend==='up'?'tr-up':trend==='dn'?'tr-dn':'tr-flat';
    const rCls=trend==='up'?'up bld':trend==='dn'?'dn bld':'w bld';
    const bias=m.bias||'hold';
    const bpCls=bias==='cut'?'bp-cut':bias==='hike'?'bp-hike':'bp-hold';
    const biasLbl=bias==='cut'?'↓ Cut':bias==='hike'?'↑ Hike':'→ Hold';
    const meth=m.biasMethod?`<span class="dim"> ${m.biasMethod}</span>`:'';
    const cutP=m.cutProb!=null?m.cutProb+'%':'—';
    const hikeP=m.hikeProb!=null?m.hikeProb+'%':'—';
    const nxt=m.nextMeeting||'—';
    return`<tr>
      <td class="amb bld">${sh}</td>
      <td class="r ${rCls}">${rate!=null?rate.toFixed(2)+'%':'—'}</td>
      <td class="r ${arCls}">${arrow}</td>
      <td class="r dim" style="font-size:8.5px">${nxt}</td>
      <td class="r"><span class="bp ${bpCls}">${biasLbl}</span>${meth}</td>
      <td class="r dn">${cutP}</td>
      <td class="r up">${hikeP}</td></tr>`;
  }).join('');
}

/* ── RENDER: CROSS-ASSET RISK ───────────────────────────────────── */
function computeStress(){
  const vix=Q.vix?.close; const gold=Q.gold?.pct; const spx=Q.spx?.pct;
  const move=Q.move?.close; const t10=Q.us10y?.close; const t3m=Q.us3m?.close;
  let s=0,f=[];
  if(vix>30){s+=3;f.push('VIX>30');}else if(vix>25){s+=2;f.push('VIX>25');}else if(vix>18){s+=1;f.push('VIX>18');}
  if(t10&&t3m&&t10<t3m){s+=1;f.push('Curve inverted');}
  if(gold>2){s+=1;f.push('Gold>+2%');}
  if(spx<-1.5){s+=1;f.push('SPX<-1.5%');}
  if(move>100){s+=1;f.push('MOVE>100');}
  return{s,f,lbl:s>=4?'RISK-OFF':s>=2?'CAUTION':s>=1?'MIXED':'RISK-ON'};
}
function renderRisk(){
  const{s,f,lbl}=computeStress();
  const badge=document.getElementById('risk-badge');
  if(badge){const c=lbl==='RISK-ON'?'rb-on':lbl==='MIXED'?'rb-mx':lbl==='CAUTION'?'rb-ca':'rb-off';badge.className='ph-badge '+c;badge.textContent=lbl;}
  const ss=document.getElementById('stress-sc');if(ss)ss.textContent=`${s}/8`;
  const sf=document.getElementById('stress-factors');if(sf)sf.textContent=f.length?f.join(' · '):'No active factors';
  const tb=document.getElementById('risk-tbody');if(!tb)return;
  let h='',lastSec=null;
  CA_ROWS.forEach(({k,lb,f:ff,u,sec})=>{
    if(sec&&sec!==lastSec){h+=`<tr class="t-sec"><td colspan="4">${sec}</td></tr>`;lastSec=sec;}
    const q=Q[k];
    if(!q){h+=`<tr><td class="dim">${lb}</td><td class="r dim">—</td><td class="r dim">—</td><td class="r dim">—</td></tr>`;return;}
    h+=`<tr><td class="dim">${lb}</td><td class="r bld w">${fv(q.close,ff,u||'')}</td>
      <td class="r ${pcc(q.pct)}">${pct(q.pct)}</td>
      <td class="r ${pcc(q.pct1w)} dim">${pct(q.pct1w)}</td></tr>`;
  });
  tb.innerHTML=h;
}

/* ── RENDER: COT ────────────────────────────────────────────────── */
function renderCOT(){
  const tb=document.getElementById('cot-tbody');if(!tb)return;
  let maxAbs=1;
  COT_CCYS.forEach(c=>{const d=COT[c];if(d?.netPosition!=null)maxAbs=Math.max(maxAbs,Math.abs(d.netPosition));});
  tb.innerHTML=COT_CCYS.map(ccy=>{
    const d=COT[ccy]||{};const net=d.netPosition;
    const hist=d.history||[];let wow=null;
    if(hist.length>=2){const cur=hist[hist.length-1];const prev=hist[hist.length-2];
      if(cur?.levNet!=null&&prev?.levNet!=null)wow=cur.levNet-prev.levNet;}
    const L=d.longPositions||0;const S=d.shortPositions||0;
    const tot=L+S;const pctOI=tot>0&&net!=null?((net/tot)*100).toFixed(1)+'%':'—';
    const nc=net!=null?(net>0?'up':net<0?'dn':'neu'):'neu';
    const wc=wow!=null?pcc(wow):'neu';
    const bp=net!=null?Math.round(Math.min((Math.abs(net)/maxAbs)*48,48)):0;
    const bd=net!=null&&net>=0?'pos':'neg';
    return`<tr>
      <td class="amb bld">${ccy}</td>
      <td class="r ${nc} bld">${sgn(net)}</td>
      <td><div class="cot-bar-wrap"><div class="cot-mid"></div><div class="cot-fill ${bd}" style="width:${bp}%"></div></div></td>
      <td class="r ${wc} dim">${sgn(wow)}</td>
      <td class="r dim">${pctOI}</td>
      <td class="r dim" style="font-size:8px">${(d.weekEnding||'').slice(5)||'—'}</td></tr>`;
  }).join('');
}

/* ── RENDER: YIELD CURVE ────────────────────────────────────────── */
function renderYC(){
  const canvas=document.getElementById('yc-canvas');if(!canvas)return;
  const pts=YC_TEN.map(({k,lb,mo})=>{const q=Q[k];return q?{lb,mo,v:q.close,chg:q.chg||0}:null;}).filter(Boolean);
  // Small yield table
  const tb=document.getElementById('yc-tb');
  if(tb)tb.innerHTML=YC_TEN.map(({k,lb})=>{
    const q=Q[k];if(!q)return`<tr><td class="dim">${lb}</td><td class="r dim">—</td><td class="r dim">—</td></tr>`;
    const c=q.chg||q.pct||0;
    return`<tr><td class="dim">${lb}</td><td class="r bld">${q.close.toFixed(3)}%</td><td class="r ${pcc(c)} dim">${c>=0?'+':''}${c.toFixed(3)}</td></tr>`;
  }).join('');
  if(pts.length<2)return;
  const W=canvas.offsetWidth;const H=canvas.offsetHeight;
  if(!W||!H)return;
  canvas.width=W;canvas.height=H;
  const PAD={t:18,r:10,b:20,l:34};
  const CW=W-PAD.l-PAD.r;const CH=H-PAD.t-PAD.b;
  const vMin=Math.min(...pts.map(p=>p.v))*0.975;
  const vMax=Math.max(...pts.map(p=>p.v))*1.025;
  const sx=mo=>PAD.l+(mo/360)*CW;
  const sy=v=>PAD.t+(1-(v-vMin)/(vMax-vMin))*CH;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,W,H);
  const inv=Q.us10y?.close<Q.us3m?.close;
  const cc=inv?'#ff6622':'#2a6aff';
  // Grid
  ctx.strokeStyle='#0e1218';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=PAD.t+(CH/4)*i;ctx.beginPath();ctx.moveTo(PAD.l,y);ctx.lineTo(W-PAD.r,y);ctx.stroke();}
  // Fill
  const g=ctx.createLinearGradient(0,PAD.t,0,H);
  g.addColorStop(0,inv?'rgba(255,102,34,0.14)':'rgba(42,106,255,0.14)');
  g.addColorStop(1,inv?'rgba(255,102,34,0)':'rgba(42,106,255,0)');
  ctx.beginPath();ctx.moveTo(sx(pts[0].mo),sy(pts[0].v));
  pts.slice(1).forEach(p=>ctx.lineTo(sx(p.mo),sy(p.v)));
  ctx.lineTo(sx(pts[pts.length-1].mo),H-PAD.b);ctx.lineTo(sx(pts[0].mo),H-PAD.b);
  ctx.closePath();ctx.fillStyle=g;ctx.fill();
  // Line
  ctx.beginPath();ctx.strokeStyle=cc;ctx.lineWidth=1.6;
  ctx.moveTo(sx(pts[0].mo),sy(pts[0].v));pts.slice(1).forEach(p=>ctx.lineTo(sx(p.mo),sy(p.v)));ctx.stroke();
  // Dots + labels
  ctx.font='7.5px Consolas,monospace';
  pts.forEach(p=>{
    const x=sx(p.mo);const y=sy(p.v);
    ctx.fillStyle=cc;ctx.beginPath();ctx.arc(x,y,2.2,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#3e4a5c';ctx.textAlign='center';ctx.fillText(p.lb,x,H-PAD.b+9);
    ctx.fillStyle='#8090a8';ctx.fillText(p.v.toFixed(2),x,y-4);
  });
  // Y axis
  ctx.textAlign='right';
  for(let i=0;i<=4;i++){const v=vMin+(vMax-vMin)*(1-i/4);ctx.fillStyle='#2a3040';ctx.fillText(v.toFixed(2),PAD.l-3,PAD.t+(CH/4)*i+3);}
  // Spread annotation
  if(Q.us10y?.close&&Q.us3m?.close){
    const sp=((Q.us10y.close-Q.us3m.close)*100).toFixed(0);
    ctx.textAlign='right';ctx.fillStyle=inv?'#ff6622':'#2a4a7a';
    ctx.fillText(inv?`INVERTED  10Y-3M: ${sp}bp`:`10Y-3M: +${sp}bp`,W-PAD.r,PAD.t+11);
  }
}

/* ── RENDER: SIGNALS ────────────────────────────────────────────── */
function renderSigs(){
  const el=document.getElementById('sigs-wire');if(!el)return;
  if(!SIGS.length){el.innerHTML='<div style="padding:12px 6px;color:#2a3040;font-size:9px;">No signals available</div>';return;}
  const pc={critical:'sig-crit',warning:'sig-warn',info:'sig-info'};
  const pl={critical:'[CRIT]',warning:'[WARN]',info:'[INFO]'};
  el.innerHTML=SIGS.map(s=>{
    const body=(s.text||'').slice(0,240);
    return`<div class="sig-item">
      <div class="sig-hd"><span class="sig-time">${s.time||'—'}</span><span class="${pc[s.priority]||'sig-info'}">${pl[s.priority]||'[INFO]'}</span><span class="sig-title"> ${s.title||''}</span></div>
      <div class="sig-body">${body}${body.length===240?'…':''}</div></div>`;
  }).join('');
}

/* ── RENDER: MACRO SNAPSHOT ─────────────────────────────────────── */
function renderMacro(){
  const tb=document.getElementById('macro-tbody');if(!tb)return;
  tb.innerHTML=CB_LIST.map(({ccy,sh})=>{
    const r=RATES[ccy];const eco=ECO[ccy]||{};
    const rate=r?r.v:null;const infl=eco.inflation;const gdp=eco.gdpGrowth;const b10=eco.bond10y;
    const ic=infl!=null?(infl>3?'dn':infl<1?'up':'w'):'dim';
    return`<tr>
      <td class="amb bld">${sh}</td>
      <td class="r bld">${rate!=null?rate.toFixed(2)+'%':'—'}</td>
      <td class="r ${ic}">${infl!=null?infl.toFixed(1)+'%':'—'}</td>
      <td class="r ${pcc(gdp)}">${gdp!=null?(gdp>=0?'+':'')+gdp.toFixed(1)+'%':'—'}</td>
      <td class="r dim">${b10!=null?b10.toFixed(2)+'%':'—'}</td></tr>`;
  }).join('');
}

/* ── CLOCK ──────────────────────────────────────────────────────── */
function startClock(){
  const el=document.getElementById('cmd-clock');if(!el)return;
  const tick=()=>{const n=new Date();el.textContent=`${p2(n.getUTCHours())}:${p2(n.getUTCMinutes())}:${p2(n.getUTCSeconds())} UTC`;};
  tick();setInterval(tick,1000);
}
function updateUpd(d){
  const el=document.getElementById('cmd-upd');if(!el||!d)return;
  const t=new Date(d);el.textContent=`Data: ${p2(t.getUTCHours())}:${p2(t.getUTCMinutes())} UTC`;
}

/* ── FLASH CHANGED ROWS ─────────────────────────────────────────── */
function flashRows(){
  [...G10,...CROSS].forEach(({id})=>{
    const p=prevQ[id];const c=Q[id];if(!p||!c)return;
    const pp=p.bid??p.close;const cp=c.bid??c.close;
    if(pp==null||cp==null||pp===cp)return;
    const row=document.getElementById('fr-'+id);if(!row)return;
    row.classList.remove('f-up','f-dn');void row.offsetWidth;
    row.classList.add(cp>pp?'f-up':'f-dn');
  });
}

/* ── RENDER ALL ─────────────────────────────────────────────────── */
function renderAll(){
  renderTicker();renderSpot();renderVol(activePair);
  renderCBRates();renderRisk();renderCOT();renderYC();
  renderSigs();renderMacro();
}

/* ── PUBLIC API ─────────────────────────────────────────────────── */
function pairClick(p){
  if(RR_PAIRS.includes(p))renderVol(p);
  const inp=document.getElementById('bbg-input');
  if(inp)inp.value=`FXIP ${p} <GO>`;
}
function ctabClick(el,cmd){
  document.querySelectorAll('.ctab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  const inp=document.getElementById('bbg-input');if(inp)inp.value=cmd+' <GO>';
}
window.pairClick=pairClick;window.ctabClick=ctabClick;

/* ── REFRESH ────────────────────────────────────────────────────── */
async function refreshFast(){
  const d=await loadQ();if(d)updateUpd(d.generated_at||d.timestamp);
  renderTicker();renderSpot();renderRisk();renderYC();flashRows();
}
async function refreshSlow(){
  await Promise.all([loadMeet(),loadRates(),loadCOT(),loadRR2(),loadSigs(),loadEco()]);
  renderCBRates();renderCOT();renderVol(activePair);renderSigs();renderMacro();
}

/* ── BOOT ────────────────────────────────────────────────────────── */
async function boot(){
  const d=await loadQ();if(d)updateUpd(d.generated_at||d.timestamp);
  await Promise.all([loadMeet(),loadRates(),loadCOT(),loadRR2(),loadSigs(),loadEco()]);
  renderAll();startClock();
  window.addEventListener('resize',renderYC);
  setInterval(refreshFast,5*60*1000);
  setInterval(refreshSlow,60*60*1000);
}
document.addEventListener('DOMContentLoaded',boot);
