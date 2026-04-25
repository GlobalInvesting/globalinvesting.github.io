// ═══════════════════════════════════════════════════════════════════════════
// YIELD CURVE MODAL  v2.0 — LightweightCharts v5 createYieldCurveChart
// File: assets/yc-modal.js
// ═══════════════════════════════════════════════════════════════════════════

(function(){
  if(document.getElementById('ycm-css'))return;
  const s=document.createElement('style');s.id='ycm-css';
  s.textContent=`
#ycm-bd{position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.78);display:flex;align-items:center;justify-content:center;padding:12px;animation:ycm-fi .15s ease;}
@keyframes ycm-fi{from{opacity:0}to{opacity:1}}
@keyframes ycm-su{from{transform:translateY(14px);opacity:0}to{transform:none;opacity:1}}
#ycm-modal{background:var(--bg,#131722);border:1px solid rgba(255,255,255,.1);border-radius:10px;width:min(800px,100%);height:min(560px,90vh);display:flex;flex-direction:column;overflow:hidden;animation:ycm-su .2s ease;font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:var(--text,#d1d4dc);}
#ycm-hd{display:flex;align-items:center;justify-content:space-between;padding:13px 18px 11px;border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;}
#ycm-title{font-size:14px;font-weight:600;letter-spacing:.01em;}
#ycm-sub{font-size:10px;color:var(--text3,#6b7280);margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#ycm-close{background:none;border:none;color:var(--text3,#6b7280);font-size:20px;cursor:pointer;padding:0 4px;line-height:1;}
#ycm-close:hover{color:var(--text,#d1d4dc);}
#ycm-strip{display:flex;border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;overflow-x:auto;scrollbar-width:none;}
#ycm-strip::-webkit-scrollbar{display:none;}
.ycm-metric{flex:1;min-width:70px;padding:9px 12px;border-right:1px solid rgba(255,255,255,.06);text-align:center;}
.ycm-metric:last-child{border-right:none;}
.ycm-m-lbl{font-size:9px;color:var(--text3,#6b7280);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;}
.ycm-m-val{font-size:15px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
.ycm-m-chg{font-size:9px;margin-top:2px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#ycm-chart-wrap{flex:1;position:relative;padding:12px 14px 8px;display:flex;flex-direction:column;min-height:0;}
#ycm-legend{display:flex;gap:16px;margin-bottom:8px;flex-shrink:0;align-items:center;}
.ycm-leg-item{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2,#9ca3af);}
.ycm-leg-line{width:20px;height:2px;border-radius:1px;flex-shrink:0;}
#ycm-lw-wrap{flex:1;position:relative;min-height:160px;overflow:hidden;}
#ycm-tooltip{position:absolute;display:none;pointer-events:none;background:#1e222d;border:1px solid #363c4e;border-radius:4px;padding:7px 11px;font-size:11px;line-height:1.55;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);color:#d1d4dc;z-index:50;box-shadow:0 4px 12px rgba(0,0,0,.6);white-space:nowrap;}
#ycm-shape{margin-left:auto;font-size:10px;font-weight:600;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#ycm-table-wrap{flex-shrink:0;border-top:1px solid rgba(255,255,255,.07);overflow-x:auto;}
#ycm-table{width:100%;border-collapse:collapse;font-size:10px;font-family:var(--font-mono,'JetBrains Mono','Courier New',monospace);}
#ycm-table th{padding:5px 10px;text-align:right;color:var(--text3,#6b7280);font-weight:400;font-size:9px;text-transform:uppercase;letter-spacing:.05em;}
#ycm-table th:first-child{text-align:left;}
#ycm-table td{padding:5px 10px;text-align:right;border-top:1px solid rgba(255,255,255,.04);}
#ycm-table td:first-child{text-align:left;color:var(--text2,#9ca3af);}
.ycu{color:var(--down,#ef5350);}.ycd{color:var(--up,#26a69a);}
@media(max-width:600px){
  #ycm-modal{border-radius:12px 12px 0 0;position:fixed;bottom:0;left:0;right:0;width:100%;height:88vh;}
  #ycm-bd{align-items:flex-end;padding:0;}
  .ycm-metric{min-width:55px;padding:7px 8px;}.ycm-m-val{font-size:12px;}
}
`;
  document.head.appendChild(s);
})();

const _TENOR_MONTHS={'1M':1,'2M':2,'3M':3,'6M':6,'1Y':12,'2Y':24,'3Y':36,'5Y':60,'7Y':84,'10Y':120,'20Y':240,'30Y':360};
const _MONTHS_TENOR={};
Object.entries(_TENOR_MONTHS).forEach(([l,m])=>{_MONTHS_TENOR[m]=l;});
let _ycLwChart=null;

function _ycChg(chg){
  if(chg==null||isNaN(chg))return{txt:'—',cls:''};
  const sign=chg>0?'+':'';
  return{txt:sign+(chg*100).toFixed(1)+'bp',cls:chg>0.001?'ycu':chg<-0.001?'ycd':''};
}
function _ycShape(tenors){
  const get=label=>tenors.find(t=>t.label===label)?.close;
  const t3m=get('3M'),t2y=get('2Y'),t10y=get('10Y');
  if(t3m==null||t10y==null)return null;
  const s10_3m=t10y-t3m,s10_2y=t2y!=null?t10y-t2y:null;
  if(s10_3m<-0.05&&(s10_2y==null||s10_2y<-0.05))return'Inverted';
  if(s10_3m>0.5)return'Steep';
  if(s10_3m>0.05)return'Normal';
  return'Flat';
}

function openYCModal(tenorData){
  closeYCModal();
  if(!Array.isArray(tenorData)||tenorData.length===0)return;
  const shape=_ycShape(tenorData);
  const shapeCols={Inverted:'#ef4444',Flat:'#f59e0b',Normal:'#22c55e',Steep:'#4f7fff'};
  const shapeCol=shapeCols[shape]||'var(--text3)';
  const metricsHtml=tenorData.map(t=>{
    const{txt,cls}=_ycChg(t.chg);
    const valCls=t.chg>0.001?'ycu':t.chg<-0.001?'ycd':'';
    return`<div class="ycm-metric"><div class="ycm-m-lbl">${t.label}</div><div class="ycm-m-val ${valCls}">${t.close!=null?t.close.toFixed(2)+'%':'—'}</div><div class="ycm-m-chg ${cls}">${txt}</div></div>`;
  }).join('');
  const tableRows=tenorData.map(t=>{
    const{txt,cls}=_ycChg(t.chg);
    return`<tr><td>${t.label}</td><td>${t.close!=null?t.close.toFixed(3)+'%':'—'}</td><td>${t.prev_close!=null?t.prev_close.toFixed(3)+'%':'—'}</td><td class="${cls}">${txt}</td></tr>`;
  }).join('');
  const bd=document.createElement('div');bd.id='ycm-bd';
  bd.setAttribute('role','dialog');bd.setAttribute('aria-modal','true');
  bd.innerHTML=`<div id="ycm-modal">
    <div id="ycm-hd">
      <div><div id="ycm-title">US Treasury Yield Curve</div><div id="ycm-sub">FRED · today vs prior close · basis points change</div></div>
      <button id="ycm-close" onclick="closeYCModal()" aria-label="Close">×</button>
    </div>
    <div id="ycm-strip">${metricsHtml}</div>
    <div id="ycm-chart-wrap">
      <div id="ycm-legend">
        <div class="ycm-leg-item"><div class="ycm-leg-line" style="background:#4f7fff;"></div>Today</div>
        <div class="ycm-leg-item"><div class="ycm-leg-line" style="background:repeating-linear-gradient(90deg,#6b7280 0,#6b7280 4px,transparent 4px,transparent 8px);"></div>Prior close</div>
        ${shape?`<div id="ycm-shape" style="color:${shapeCol}">${shape}</div>`:''}
      </div>
      <div id="ycm-lw-wrap"><div id="ycm-tooltip"></div></div>
    </div>
    <div id="ycm-table-wrap">
      <table id="ycm-table">
        <thead><tr><th>Tenor</th><th>Today</th><th>Prior close</th><th>Change (bp)</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
  </div>`;
  document.body.appendChild(bd);
  bd.addEventListener('click',e=>{if(e.target===bd)closeYCModal();});
  document.addEventListener('keydown',_ycKeydown);
  // Delay draw until after modal animation (.2s) so container has final dimensions on mobile
  // Double-draw: first at 220ms, then a corrective fitContent at 420ms for mobile layout stragglers
  setTimeout(()=>_ycDraw(tenorData), 220);
  setTimeout(()=>{if(_ycLwChart){_ycLwChart.timeScale().fitContent();}}, 420);
}

const _LWC_CDN='https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.7/dist/lightweight-charts.standalone.production.js';
let _ycLwcPromise=null;
function _ensureYcLwc(){
  if(window.LightweightCharts)return Promise.resolve();
  if(_ycLwcPromise)return _ycLwcPromise;
  _ycLwcPromise=new Promise((res,rej)=>{
    const s=document.createElement('script');
    s.src=_LWC_CDN;
    s.onload=()=>{_ycLwcPromise=null;res();};
    s.onerror=()=>{_ycLwcPromise=null;rej(new Error('YC: LWC load failed'));};
    document.head.appendChild(s);
  });
  return _ycLwcPromise;
}

function _ycDraw(tenorData){
  const container=document.getElementById('ycm-lw-wrap');if(!container)return;
  _ensureYcLwc().then(()=>{
    const LWC=window.LightweightCharts;if(!LWC)return;
    if(_ycLwChart){try{_ycLwChart.remove();}catch(_){}  _ycLwChart=null;}
    const toData=tenorData.map(t=>{const m=t.months??_TENOR_MONTHS[t.label];return(m!=null&&t.close!=null)?{time:m,value:t.close}:null;}).filter(Boolean);
    const prData=tenorData.map(t=>{const m=t.months??_TENOR_MONTHS[t.label];return(m!=null&&t.prev_close!=null)?{time:m,value:t.prev_close}:null;}).filter(Boolean);
    if(typeof LWC.createYieldCurveChart==='function')_ycDrawNative(container,toData,prData,tenorData);
    else _ycDrawFallback(container,toData,prData,tenorData);
  }).catch(e=>console.error(e));
}

function _ycDrawNative(container,toData,prData,tenorData){
  const LWC=window.LightweightCharts;
  const _tickLabels={};
  tenorData.forEach(t=>{const m=t.months??_TENOR_MONTHS[t.label];if(m!=null)_tickLabels[m]=t.label;});
  const w=container.offsetWidth||360, h=container.offsetHeight||220;
  const firstTenorMonth=toData[0]?.time??3;
  const lastTenorMonth=toData[toData.length-1]?.time??360;
  const dataSpanYears=Math.ceil((lastTenorMonth-firstTenorMonth)/12)+2;
  _ycLwChart=LWC.createYieldCurveChart(container,{
    width:w, height:h,
    layout:{background:{type:'solid',color:'#131722'},textColor:'#787b86',fontFamily:"'JetBrains Mono','Courier New',monospace",fontSize:10,attributionLogo:false},
    yieldCurve:{baseResolution:12,minimumTimeRange:dataSpanYears,startTimeRange:firstTenorMonth},
    grid:{vertLines:{color:'rgba(255,255,255,0.04)'},horzLines:{color:'rgba(255,255,255,0.04)'}},
    crosshair:{mode:LWC.CrosshairMode?.Magnet??1,vertLine:{color:'rgba(255,255,255,0.25)',style:LWC.LineStyle?.Dashed??1,labelVisible:false},horzLine:{color:'rgba(255,255,255,0.15)',style:LWC.LineStyle?.Dashed??1,labelVisible:true}},
    leftPriceScale:{visible:false},
    rightPriceScale:{borderVisible:false,scaleMargins:{top:0.12,bottom:0.08}},
    timeScale:{borderVisible:false,minBarSpacing:1,tickMarkFormatter:m=>_tickLabels[m]||''},
    handleScroll:false,handleScale:false,
    localization:{priceFormatter:v=>v!=null?v.toFixed(3)+'%':'—'},
  });
  // Force canvas to correct size immediately — LWC initializes canvas to leftPriceScale width (56px)
  // even when leftPriceScale is disabled, because the internal layout runs before the option takes effect.
  // applyOptions with the real container dimensions forces a re-layout at the correct size.
  _ycLwChart.applyOptions({width:w,height:h});
  let priorSeries=null;
  if(prData.length>=2){
    priorSeries=_ycLwChart.addSeries(LWC.LineSeries,{color:'rgba(107,114,128,0.55)',lineWidth:1,lineType:LWC.LineType?.Curved??2,lineStyle:LWC.LineStyle?.Dashed??1,pointMarkersVisible:true,crosshairMarkerVisible:true,crosshairMarkerRadius:3,priceLineVisible:false,lastValueVisible:false});
    priorSeries.setData(prData);
  }
  const todaySeries=_ycLwChart.addSeries(LWC.LineSeries,{color:'#4f7fff',lineWidth:2,lineType:LWC.LineType?.Curved??2,pointMarkersVisible:true,crosshairMarkerVisible:true,crosshairMarkerRadius:4,crosshairMarkerBorderColor:'#131722',crosshairMarkerBorderWidth:2,priceLineVisible:false,lastValueVisible:false});
  todaySeries.setData(toData);
  _ycLwChart.timeScale().fitContent();
  _ycLwChart.timeScale().subscribeSizeChange(()=>_ycLwChart.timeScale().fitContent());
  // ResizeObserver: keep canvas in sync when container changes (orientation, keyboard, etc.)
  if(window.ResizeObserver){
    const ro=new ResizeObserver(entries=>{
      if(!_ycLwChart)return;
      const e=entries[0],nw=Math.floor(e.contentRect.width),nh=Math.floor(e.contentRect.height);
      if(nw>0&&nh>0){_ycLwChart.applyOptions({width:nw,height:nh});_ycLwChart.timeScale().fitContent();}
    });
    ro.observe(container);container._ycRo=ro;
  }
  _ycAttachTooltip(container,_ycLwChart,todaySeries,priorSeries,tenorData,false);
}

function _ycDrawFallback(container,toData,prData,tenorData){
  const LWC=window.LightweightCharts;
  const LABELS=tenorData.map(t=>t.label);
  _ycLwChart=LWC.createChart(container,{
    autoSize:true,
    layout:{background:{type:'solid',color:'#131722'},textColor:'#787b86',fontFamily:"'JetBrains Mono','Courier New',monospace",fontSize:10,attributionLogo:false},
    grid:{vertLines:{color:'rgba(255,255,255,0.04)'},horzLines:{color:'rgba(255,255,255,0.04)'}},
    crosshair:{mode:LWC.CrosshairMode?.Magnet??1,vertLine:{color:'rgba(255,255,255,0.25)',style:LWC.LineStyle?.Dashed??1,labelVisible:false},horzLine:{color:'rgba(255,255,255,0.15)',style:LWC.LineStyle?.Dashed??1,labelVisible:true}},
    rightPriceScale:{borderVisible:false,scaleMargins:{top:0.12,bottom:0.08}},
    timeScale:{borderVisible:false,tickMarkFormatter:t=>LABELS[t-1]||String(t),fixLeftEdge:true,fixRightEdge:true,barSpacing:0},
    handleScroll:false,handleScale:false,
    localization:{priceFormatter:v=>v!=null?v.toFixed(3)+'%':'—'},
  });
  // Sequential integers → equal spacing between tenors (Bloomberg/Eikon standard)
  const toSeq=toData.map((d,i)=>({time:i+1,value:d.value}));
  const prSeq=prData.map((d,i)=>({time:i+1,value:d.value}));
  let priorSeries=null;
  if(prSeq.length>=2){
    priorSeries=_ycLwChart.addSeries(LWC.LineSeries,{color:'rgba(107,114,128,0.55)',lineWidth:1,lineStyle:LWC.LineStyle?.Dashed??1,pointMarkersVisible:true,crosshairMarkerVisible:true,crosshairMarkerRadius:3,priceLineVisible:false,lastValueVisible:false});
    priorSeries.setData(prSeq);
  }
  const todaySeries=_ycLwChart.addSeries(LWC.LineSeries,{color:'#4f7fff',lineWidth:2,pointMarkersVisible:true,crosshairMarkerVisible:true,crosshairMarkerRadius:4,crosshairMarkerBorderColor:'#131722',crosshairMarkerBorderWidth:2,priceLineVisible:false,lastValueVisible:false});
  todaySeries.setData(toSeq);
  _ycLwChart.timeScale().fitContent();
  // ResizeObserver for mobile layout shifts
  if(window.ResizeObserver){
    const ro=new ResizeObserver(entries=>{
      if(!_ycLwChart)return;
      const e=entries[0],nw=Math.floor(e.contentRect.width),nh=Math.floor(e.contentRect.height);
      if(nw>0&&nh>0){_ycLwChart.applyOptions({width:nw,height:nh});_ycLwChart.timeScale().fitContent();}
    });
    ro.observe(container);container._ycRo=ro;
  }
  const resize=()=>{if(_ycLwChart&&container.offsetWidth>0)_ycLwChart.applyOptions({width:container.offsetWidth,height:container.offsetHeight});};
  window.addEventListener('resize',resize);container._ycResize=resize;
  _ycAttachTooltip(container,_ycLwChart,todaySeries,priorSeries,tenorData,true);
}

function _ycAttachTooltip(container,chart,todaySeries,priorSeries,tenorData,isSeq){
  const tooltip=document.getElementById('ycm-tooltip');if(!tooltip)return;
  const TW=180,TM=10;
  chart.subscribeCrosshairMove(param=>{
    if(!param?.point||!param.seriesData){tooltip.style.display='none';return;}
    const tv=param.seriesData.get(todaySeries),pv=priorSeries?param.seriesData.get(priorSeries):null;
    if(!tv){tooltip.style.display='none';return;}
    const tenor=isSeq?(tenorData[(param.time??1)-1]?.label||''):(_MONTHS_TENOR[param.time]||(param.time+'M'));
    const diff=(tv.value!=null&&pv?.value!=null)?((tv.value-pv.value)*100):null;
    const diffStr=diff!=null?`<span style="color:${diff>0?'#ef5350':diff<0?'#26a69a':'#787b86'}">${diff>0?'+':''}${diff.toFixed(1)}bp</span>`:'';
    tooltip.innerHTML=`<div style="font-size:9px;color:#6b7280;letter-spacing:.05em;margin-bottom:4px;">${tenor} TREASURY</div>`+
      (tv.value!=null?`<div>Today &nbsp;&nbsp;<span style="color:#4f7fff;font-weight:700;">${tv.value.toFixed(3)}%</span> ${diffStr}</div>`:'')+
      (pv?.value!=null?`<div style="color:#6b7280;">Prior &nbsp;&nbsp;&nbsp;${pv.value.toFixed(3)}%</div>`:'');
    tooltip.style.display='block';
    const cW=container.offsetWidth,cx=param.point.x,cy=param.point.y,th=tooltip.offsetHeight||60;
    const tx=(cx+TM+TW<=cW-4)?cx+TM:cx-TM-TW;
    const ty=(cy-th-TM>=4)?cy-th-TM:cy+TM;
    tooltip.style.left=Math.max(0,tx)+'px';tooltip.style.top=Math.max(0,ty)+'px';
  });
}

function _ycKeydown(e){if(e.key==='Escape')closeYCModal();}

function closeYCModal(){
  if(_ycLwChart){
    const c=document.getElementById('ycm-lw-wrap');
    if(c?._ycResize)window.removeEventListener('resize',c._ycResize);
    if(c?._ycRo){c._ycRo.disconnect();c._ycRo=null;}
    try{_ycLwChart.remove();}catch(_){}
    _ycLwChart=null;
  }
  const bd=document.getElementById('ycm-bd');if(bd)bd.remove();
  document.removeEventListener('keydown',_ycKeydown);
}

window.openYCModal=openYCModal;window.closeYCModal=closeYCModal;
