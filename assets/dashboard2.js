/**
 * dashboard2.js — Real-data + LWC chart layer for index2.html
 * Patches mock data arrays with live JSON; renders LWC area charts.
 * GUIDELINES v8.2.45 — no Math.random(), all values deterministic.
 */
'use strict';

/* ── Config ────────────────────────────────────────────────────────── */
const RATE_IDX = { USD:0,EUR:1,JPY:2,GBP:3,CAD:4,AUD:5,NZD:6,CHF:7,NOK:8,SEK:9 };
const EXT_CCY  = ['USD','EUR','GBP','JPY','AUD','CAD','CHF','NZD'];
const OHLC_FILE = {
  eu:'eurusd', jp:'usdjpy', uk:'gbpusd', ca:'usdcad',
  au:'audusd', nz:'nzdusd', ch:'usdchf',
};
const ohlcCache = {};

/* ── LWC theme — deep Bloomberg dark palette ────────────────────────── */
const LWC_THEME = {
  layout:{
    background:{ type:'solid', color:'#010206' },
    textColor:'#7890b0',
    fontSize:9,
    fontFamily:"'IBM Plex Mono',Consolas,monospace",
  },
  grid:{
    vertLines:{ color:'#070d18', style:0 },
    horzLines:{ color:'#070d18', style:0 },
  },
  crosshair:{
    mode:1,
    vertLine:{ color:'#1e3a52', labelBackgroundColor:'#040a10' },
    horzLine:{ color:'#1e3a52', labelBackgroundColor:'#040a10' },
  },
  timeScale:{
    borderColor:'#101e2c',
    timeVisible:true,
    secondsVisible:false,
    fixLeftEdge:true,
    fixRightEdge:true,
  },
  rightPriceScale:{
    borderColor:'#101e2c',
    scaleMargins:{ top:0.08, bottom:0.08 },
    textColor:'#5a7898',
  },
  handleScroll:{ mouseWheel:true, pressedMouseMove:true },
  handleScale:{ mouseWheel:true, pinch:true },
};

/* ── Chart instances ────────────────────────────────────────────────── */
let _dxyChart=null, _dxySeries=null;
let _ctyChart=null, _ctySeries=null;
let _curCtyKey=null;

/* ── Helpers ────────────────────────────────────────────────────────── */
function hexAlpha(hex,a){
  const r=parseInt(hex.slice(1,3),16);
  const g=parseInt(hex.slice(3,5),16);
  const b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}
function lwcColor(last,first){ return last>=first?'#00dd00':'#ff2424'; }
function toLineData(arr){ return arr.map(d=>({time:d.time,value:d.close})); }

/* ── Create / recreate an area chart ───────────────────────────────── */
function makeAreaChart(containerId, color, existingChart){
  const el=document.getElementById(containerId);
  if(!el||typeof LightweightCharts==='undefined') return null;
  if(existingChart){ try{ existingChart.remove(); }catch{} }
  el.innerHTML='';

  const w=el.offsetWidth||el.parentElement?.offsetWidth||460;
  const h=el.offsetHeight||el.parentElement?.offsetHeight||100;

  const chart=LightweightCharts.createChart(el,{...LWC_THEME,width:w,height:Math.max(h,60)});

  const series=chart.addAreaSeries({
    lineColor:color,
    topColor:hexAlpha(color,0.18),
    bottomColor:hexAlpha(color,0.00),
    lineWidth:1,
    priceLineVisible:false,
    lastValueVisible:false,
    crosshairMarkerVisible:true,
    crosshairMarkerRadius:3,
    crosshairMarkerBorderColor:color,
    crosshairMarkerBackgroundColor:'#040a10',
  });

  // Auto-resize on container size change
  const ro=new ResizeObserver(()=>{
    const nw=el.offsetWidth, nh=el.offsetHeight;
    if(nw>10&&nh>10) chart.resize(nw,nh);
  });
  ro.observe(el);

  return {chart,series};
}

/* ── DXY Chart ──────────────────────────────────────────────────────── */
function initDXYChart(ohlcData){
  const el=document.getElementById('lwc-dxy');
  if(!el||typeof LightweightCharts==='undefined') return;
  const slice=ohlcData.slice(-180);
  if(slice.length<2) return;

  const color=lwcColor(slice[slice.length-1].close,slice[0].close);
  const res=makeAreaChart('lwc-dxy',color,_dxyChart);
  if(!res) return;
  _dxyChart=res.chart; _dxySeries=res.series;
  _dxySeries.setData(toLineData(slice));
  _dxyChart.timeScale().fitContent();

  // Update legend
  const lbl=document.getElementById('dxy-lbl');
  const sq=document.getElementById('dxy-csq');
  const last=slice[slice.length-1].close;
  if(lbl){ lbl.textContent=`DXY Index ${last.toFixed(3)}`; lbl.style.color=color; }
  if(sq)  sq.style.background=color;

  _dxyChart.subscribeCrosshairMove(p=>{
    if(!p.point||!p.seriesData) return;
    const v=p.seriesData.get(_dxySeries);
    if(v&&lbl) lbl.textContent=`DXY Index ${v.value.toFixed(3)}`;
  });
}

/* ── Country Chart ──────────────────────────────────────────────────── */
async function fetchOhlc(pairFile){
  if(ohlcCache[pairFile]!==undefined) return ohlcCache[pairFile];
  try{
    const r=await fetch(`ohlc-data/${pairFile}.json`);
    ohlcCache[pairFile]=await r.json();
  }catch{ ohlcCache[pairFile]=null; }
  return ohlcCache[pairFile];
}

function initCtyChart(key,data){
  const el=document.getElementById('lwc-cty');
  if(!el||typeof LightweightCharts==='undefined'||!data) return;
  const slice=data.slice(-180);
  if(slice.length<2) return;

  const last=slice[slice.length-1].close;
  const color=lwcColor(last,slice[0].close);
  const res=makeAreaChart('lwc-cty',color,_ctyChart);
  if(!res) return;
  _ctyChart=res.chart; _ctySeries=res.series;
  _ctySeries.setData(toLineData(slice));
  _ctyChart.timeScale().fitContent();

  const sq=document.getElementById('ct-csq');
  const lbl=document.getElementById('ct-clbl');
  if(sq) sq.style.background=color;
  if(lbl){ lbl.style.color=color; }

  _ctyChart.subscribeCrosshairMove(p=>{
    if(!p.point||!p.seriesData) return;
    const v=p.seriesData.get(_ctySeries);
    if(v&&lbl) lbl.textContent=v.value.toFixed(4);
  });
}

/* ── Override showCty to attach real OHLC ──────────────────────────── */
const _origShowCty=window.showCty;
window.showCty=function(key){
  if(typeof _origShowCty==='function') _origShowCty(key);
  _curCtyKey=key;
  const pairFile=OHLC_FILE[key];
  if(!pairFile) return;
  // Small delay to ensure DOM is fully laid out
  requestAnimationFrame(()=>{
    fetchOhlc(pairFile).then(data=>{
      if(data) initCtyChart(key,data);
    });
  });
};

/* ── Data patchers ──────────────────────────────────────────────────── */
function bpStr(spd){
  if(spd==null||isNaN(spd)) return '--';
  const bp=Math.round(spd*100);
  return(bp>0?'+':'')+bp+'bp';
}

function patchRateRows(quotes,extData){
  const usd10y=extData.USD?.bond10y??4.475;
  const cfg={
    EUR:{q:'eurusd',eqSym:'stoxx'},
    JPY:{q:'usdjpy',eqSym:'nikkei'},
    GBP:{q:'gbpusd',eqSym:'ftse'},
    CAD:{q:'usdcad',eqSym:null},
    AUD:{q:'audusd',eqSym:'asx'},
    NZD:{q:'nzdusd',eqSym:null},
    CHF:{q:'usdchf',eqSym:null},
  };
  for(const [ccy,c] of Object.entries(cfg)){
    const idx=RATE_IDX[ccy]; if(idx==null) continue;
    const q=quotes[c.q];   if(!q) continue;
    const row=rateRows[idx];
    const dp=(ccy==='JPY')?2:4;
    row.sp=q.close.toFixed(dp);
    row.pc=parseFloat((q.pct??0).toFixed(2));
    const y10=extData[ccy]?.bond10y;
    if(y10!=null){ row.yld=parseFloat(y10).toFixed(3); row.ysp=bpStr(y10-usd10y); }
    if(c.eqSym&&quotes[c.eqSym]){
      const eq=quotes[c.eqSym];
      row.idx=eq.close.toLocaleString('en-US',{maximumFractionDigits:0});
      row.ic=parseFloat((eq.pct??0).toFixed(2));
    }
  }
  const spx=quotes.spx;
  if(spx){ rateRows[0].idx=spx.close.toLocaleString('en-US',{maximumFractionDigits:0}); rateRows[0].ic=parseFloat((spx.pct??0).toFixed(2)); }
}

function patchCtyMeta(quotes){
  const map={eu:'eurusd',jp:'usdjpy',uk:'gbpusd',ca:'usdcad',au:'audusd',nz:'nzdusd',ch:'usdchf'};
  for(const [key,qk] of Object.entries(map)){
    const q=quotes[qk]; if(!q||!ctyMeta[key]) continue;
    const dp=(key==='jp')?2:4;
    ctyMeta[key].sp=q.close.toFixed(dp);
    ctyMeta[key].vs[0]=q.close;
    ctyMeta[key].cs[0]=parseFloat((q.chg??0).toFixed(dp));
  }
  const dxy=quotes.dxy;
  if(dxy&&ctyMeta.us){ ctyMeta.us.sp=dxy.close.toFixed(4); ctyMeta.us.vs[0]=dxy.close; ctyMeta.us.cs[0]=parseFloat((dxy.chg??0).toFixed(4)); }
}

function patchMeetings(mtgJson){
  const m=mtgJson?.meetings??{};
  const map={us:'USD',eu:'EUR',jp:'JPY',uk:'GBP',ca:'CAD',au:'AUD',nz:'NZD',ch:'CHF'};
  for(const [sk,cur] of Object.entries(map)){
    const mtg=m[cur]; if(!mtg?.nextMeeting) continue;
    (kdMap[sk]??[]).forEach(item=>{ if(item.k==='Next Decision'||item.k==='Next MAS') item.v=mtg.nextMeeting; });
    if(ecoCB[sk]) ecoCB[sk].next=mtg.nextMeeting;
  }
}

function patchEcoRows(extData){
  const cy={'U.S.':'USD','Euro':'EUR','Japan':'JPY','U.K.':'GBP','Canada':'CAD','Australia':'AUD','N.Z.':'NZD','Switz.':'CHF'};
  for(const row of ecoRows){
    const ext=extData[cy[row.cy]]; if(!ext) continue;
    if(ext.bond10y!=null) row.y=parseFloat(parseFloat(ext.bond10y).toFixed(3));
  }
}

/* ── Main load ──────────────────────────────────────────────────────── */
async function loadRealData(){
  try{
    const extFetches=EXT_CCY.map(c=>fetch(`extended-data/${c}.json`).then(r=>r.json()).catch(()=>null));
    const [quotesJson,mtgJson,dxyOhlc,...extResults]=await Promise.all([
      fetch('intraday-data/quotes.json').then(r=>r.json()).catch(()=>null),
      fetch('meetings-data/meetings.json').then(r=>r.json()).catch(()=>null),
      fetch('ohlc-data/dxy.json').then(r=>r.json()).catch(()=>null),
      ...extFetches,
    ]);

    const quotes=quotesJson?.quotes??{};
    const extData={};
    EXT_CCY.forEach((c,i)=>{ extData[c]=extResults[i]?.data??null; });

    if(Object.keys(quotes).length){
      patchRateRows(quotes,extData);
      patchCtyMeta(quotes);
      buildRates();
    }
    if(mtgJson) patchMeetings(mtgJson);
    patchEcoRows(extData);
    buildEco();

    // DXY LWC chart — wait for layout
    if(Array.isArray(dxyOhlc)&&dxyOhlc.length){
      requestAnimationFrame(()=>{ requestAnimationFrame(()=>initDXYChart(dxyOhlc)); });
    }

    // Re-draw country chart if one is open
    if(_curCtyKey&&OHLC_FILE[_curCtyKey]){
      fetchOhlc(OHLC_FILE[_curCtyKey]).then(d=>{ if(d) initCtyChart(_curCtyKey,d); });
    }

    // Timestamp
    const now=new Date();
    const hh=String(now.getUTCHours()).padStart(2,'0');
    const mm=String(now.getUTCMinutes()).padStart(2,'0');
    const ct2=document.getElementById('ct2');
    if(ct2) ct2.innerHTML=`<span class="kw">&lt;HELP&gt;</span> for explanation, <span class="kw">&lt;MENU&gt;</span> for similar functions. <span style="color:#2a7048">&#9632; Live ${hh}:${mm} UTC</span>`;

  }catch(err){ console.warn('[dashboard2]',err); }
}

/* ── Boot ───────────────────────────────────────────────────────────── */
const _boot=()=>{ loadRealData(); setInterval(loadRealData,5*60*1000); };
document.readyState==='loading'?document.addEventListener('DOMContentLoaded',_boot):_boot();
