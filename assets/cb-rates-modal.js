// ═══════════════════════════════════════════════════════════════════════════
// CB RATES MODAL  v2.0 — LightweightCharts v5 (replaces Chart.js)
// File: assets/cb-rates-modal.js
// ═══════════════════════════════════════════════════════════════════════════

(function(){
  if(document.getElementById('cbr-modal-css'))return;
  const s=document.createElement('style');s.id='cbr-modal-css';
  s.textContent=`
#cbr-bd{position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:flex;align-items:center;justify-content:center;padding:12px;animation:cbr-fi .15s ease;}
@keyframes cbr-fi{from{opacity:0}to{opacity:1}}
@keyframes cbr-su{from{transform:translateY(12px);opacity:0}to{transform:none;opacity:1}}
#cbr-modal{background:#161b22;border:1px solid #30363d;border-radius:8px;width:min(860px,100%);height:min(600px,90vh);display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.04);animation:cbr-su .18s cubic-bezier(.16,1,.3,1);font-family:var(--font-ui,'Inter',-apple-system,sans-serif);color:#e6edf3;position:relative;}
#cbr-modal::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1f6feb 0%,#58a6ff 50%,#26a69a 100%);border-radius:8px 8px 0 0;z-index:1;}
#cbr-m-hd{display:flex;align-items:center;justify-content:space-between;padding:14px 18px 12px;border-bottom:1px solid #30363d;flex-shrink:0;background:#161b22;}
#cbr-m-title{font-size:14px;font-weight:600;color:#e6edf3;letter-spacing:-.01em;}
#cbr-m-sub{font-size:10px;color:#6e7681;margin-top:2px;font-family:'IBM Plex Mono',var(--font-mono,monospace);letter-spacing:.02em;}
#cbr-m-close{background:none;border:none;color:#6e7681;font-size:18px;cursor:pointer;padding:5px 7px;border-radius:5px;line-height:1;transition:color .1s,background .1s;}
#cbr-m-close:hover{color:#e6edf3;background:#21262d;}
#cbr-m-metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:0;background:#0d1117;border-bottom:1px solid #30363d;flex-shrink:0;}
.cbr-mm{background:#0d1117;padding:9px 16px;display:flex;flex-direction:column;gap:2px;border-right:1px solid #30363d;}
.cbr-mm:last-child{border-right:none;}
.cbr-mm-lbl{font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;font-family:'IBM Plex Mono',var(--font-mono,monospace);}
.cbr-mm-val{font-size:14px;font-weight:600;font-family:'IBM Plex Mono',var(--font-mono,monospace);}
.cbr-mm-sub{font-size:9px;color:#6e7681;font-family:'IBM Plex Mono',var(--font-mono,monospace);}
#cbr-m-tabs{display:flex;padding:0 18px;border-bottom:1px solid #30363d;flex-shrink:0;overflow-x:auto;scrollbar-width:none;background:#161b22;}
#cbr-m-tabs::-webkit-scrollbar{display:none;}
.cbr-tab{font-size:11px;padding:9px 13px;cursor:pointer;color:#6e7681;border-bottom:2px solid transparent;transition:color .1s;white-space:nowrap;user-select:none;}
.cbr-tab:hover{color:#8b949e;}
.cbr-tab.on{color:#e6edf3;border-bottom-color:#388bfd;}
#cbr-m-body{flex:1;min-height:0;overflow-y:hidden;padding:14px 16px;display:flex;flex-direction:column;background:#0d1117;scrollbar-width:thin;scrollbar-color:#444c56 transparent;}
#cbr-m-body.cbr-body--history{overflow-y:auto;}
#cbr-m-body::-webkit-scrollbar{width:4px;}
#cbr-m-body::-webkit-scrollbar-track{background:transparent;}
#cbr-m-body::-webkit-scrollbar-thumb{background:#444c56;border-radius:2px;}
.cbr-panel{display:none;}
.cbr-panel.on{display:flex;flex:1;flex-direction:column;min-height:0;}
#cbr-p-decisions.on{display:block;flex:none;}
.cbr-cw{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 14px;margin-bottom:10px;display:flex;flex-direction:column;}
.cbr-cw.fill{flex:1;min-height:0;}
.cbr-ct{font-size:9.5px;color:#6e7681;margin-bottom:8px;font-family:'IBM Plex Mono',var(--font-mono,monospace);letter-spacing:.04em;padding-bottom:8px;border-bottom:1px solid #30363d;text-transform:uppercase;}
.cbr-chart-area{flex:1;min-height:0;height:100%;position:relative;}
.cbr-lw-wrap{width:100%;height:100%;min-height:200px;position:relative;}
.cbr-lw-tooltip{position:absolute;display:none;pointer-events:none;background:#161b22;border:1px solid #30363d;border-radius:4px;padding:7px 11px;font-size:11px;line-height:1.55;font-family:'IBM Plex Mono',var(--font-mono,monospace);color:#e6edf3;z-index:50;box-shadow:0 4px 16px rgba(0,0,0,.7);white-space:nowrap;}
.cbr-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:'IBM Plex Mono',var(--font-mono,monospace);}
.cbr-tbl th{text-align:right;color:#6e7681;font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:.08em;padding:5px 8px 4px;border-bottom:1px solid #30363d;}
.cbr-tbl th:first-child{text-align:left;}
.cbr-tbl td{text-align:right;padding:5px 8px;border-bottom:1px solid rgba(48,54,61,.6);}
.cbr-tbl td:first-child{text-align:left;color:#8b949e;}
.cbr-tbl tr:last-child td{border-bottom:none;}
.cbr-tbl tr:hover td{background:rgba(255,255,255,.03);}
.cbr-tbl .now-row td{background:rgba(56,139,253,.06);}
.cu{color:#26a69a;}.cd{color:#ef5350;}.cf{color:#8b949e;}
@media(max-width:600px){
  #cbr-bd{padding:0;align-items:flex-end;}
  #cbr-modal{width:100%;height:93vh;border-radius:12px 12px 0 0;border-bottom:none;}
  #cbr-m-metrics{grid-template-columns:repeat(3,1fr);}
  .cbr-mm{padding:6px 10px;}.cbr-mm-val{font-size:12px;}
  #cbr-m-tabs{padding:0 8px;}.cbr-tab{font-size:10px;padding:8px 8px;}
  #cbr-m-body{padding:8px;}.cbr-cw{padding:9px 10px;margin-bottom:8px;}
}
`;
  document.head.appendChild(s);
})();

const _cbrMonoF="'JetBrains Mono','Courier New',monospace";
let _cbrLwChart=null;
function _destroyCBRChart(){if(_cbrLwChart){try{_cbrLwChart.remove();}catch(_){}  _cbrLwChart=null;}}

function _processCBRateData(obs){
  if(!obs||!obs.length)return{chronData:[],decisions:[]};
  const chron=[...obs].reverse();
  const chronData=chron.map(o=>({time:o.date.length===7?o.date+'-01':o.date,value:parseFloat(o.value)}));
  const decisions=[];
  for(let i=1;i<chronData.length;i++){
    const delta=chronData[i].value-chronData[i-1].value;
    if(Math.abs(delta)>=0.01)decisions.push({time:chronData[i].time,delta,rate:chronData[i].value});
  }
  return{chronData,decisions};
}

function _cbrLwOptions(W,H){
  return{
    width:W,height:H,
    layout:{background:{type:'solid',color:'#161b22'},textColor:'#6e7681',fontFamily:_cbrMonoF,fontSize:10,attributionLogo:false},
    grid:{vertLines:{color:'rgba(255,255,255,0.04)'},horzLines:{color:'rgba(255,255,255,0.04)'}},
    crosshair:{mode:window.LightweightCharts?.CrosshairMode?.Normal??1,vertLine:{color:'rgba(255,255,255,0.2)',style:2,labelVisible:false},horzLine:{color:'rgba(255,255,255,0.12)',style:2,labelVisible:true}},
    rightPriceScale:{borderVisible:false,scaleMargins:{top:0.15,bottom:0.1}},
    timeScale:{borderVisible:false,timeVisible:false},
    handleScroll:{mouseWheel:true,pressedMouseMove:true},
    handleScale:{mouseWheel:true,pinch:true},
    localization:{priceFormatter:v=>v!=null?v.toFixed(2)+'%':'—'},
  };
}

function _buildDecisionOverlay(container,lwChart,decisions){
  const old=container.querySelector('.cbr-decision-svg');
  if(old){if(old._cleanup)old._cleanup();old.remove();}
  if(!decisions.length)return;
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.classList.add('cbr-decision-svg');
  svg.style.cssText='position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;overflow:visible;';
  container.appendChild(svg);

  // Bloomberg-style staggered decision markers:
  // Labels are placed at multiple vertical tiers so nearby decisions don't overlap.
  // Each label gets a thin vertical stem from its tier down to the chart line.
  // Tiers are assigned greedily: for each marker, pick the lowest tier whose last
  // placed label ended at least MIN_GAP_PX to the left of this marker's x position.
  const LABEL_W=32,LABEL_H=14,LABEL_PAD=4,MIN_GAP_PX=6;
  // Three tiers: top row, mid row, lower row (Bloomberg typically uses 2-3)
  const TIERS=[6,24,42]; // y offset from top of chart area

  function redraw(){
    const ts=lwChart.timeScale(),H=container.offsetHeight,W=container.offsetWidth;
    if(!H||!W)return;

    // Collect visible decisions with their x coordinates
    const visible=[];
    decisions.forEach(d=>{
      try{
        const x=ts.timeToCoordinate(d.time);
        if(x==null||x<-LABEL_W||x>W+LABEL_W)return;
        visible.push({...d,x});
      }catch(_){}
    });

    // Assign tiers greedily left→right
    // tierEnd[i] tracks the rightmost x used in tier i
    const tierEnd=TIERS.map(()=>-Infinity);
    const assigned=visible.map(d=>{
      // Find the first tier where the label fits without overlapping
      let tier=0;
      for(let t=0;t<TIERS.length;t++){
        if(d.x-LABEL_W/2 >= tierEnd[t]+MIN_GAP_PX){tier=t;break;}
        // If no tier fits cleanly, pick the one with oldest end (most space)
        if(t===TIERS.length-1){
          tier=tierEnd.indexOf(Math.min(...tierEnd));
        }
      }
      tierEnd[tier]=d.x+LABEL_W/2;
      return{...d,tier};
    });

    let html='';
    assigned.forEach(d=>{
      const x=d.x,col=d.delta>0?'#26a69a':'#ef5350';
      const colA=d.delta>0?'rgba(38,166,154,0.35)':'rgba(239,83,80,0.35)';
      const colB=d.delta>0?'rgba(38,166,154,0.12)':'rgba(239,83,80,0.12)';
      const sign=d.delta>0?'+':'',label=sign+Math.round(d.delta*100)+'bp';
      const ty=TIERS[d.tier];
      const labelMidY=ty+LABEL_H/2+1;
      const stemTop=ty+LABEL_H+2;

      // Vertical dashed line from bottom of label tier down to chart area
      html+=`<line x1="${x.toFixed(1)}" y1="${stemTop}" x2="${x.toFixed(1)}" y2="${(H-16).toFixed(1)}" stroke="${colA}" stroke-width="1" stroke-dasharray="2,3"/>`;
      // Pill background
      html+=`<rect x="${(x-LABEL_W/2).toFixed(1)}" y="${ty}" width="${LABEL_W}" height="${LABEL_H}" rx="3" fill="${colB}" stroke="${col}" stroke-width="0.5" stroke-opacity="0.6"/>`;
      // Label text
      html+=`<text x="${x.toFixed(1)}" y="${(ty+LABEL_H-3).toFixed(1)}" text-anchor="middle" font-size="8.5" font-family="${_cbrMonoF}" fill="${col}" font-weight="700">${label}</text>`;
    });
    svg.innerHTML=html;
  }

  redraw();
  lwChart.timeScale().subscribeVisibleTimeRangeChange(redraw);
  svg._cleanup=()=>{try{lwChart.timeScale().unsubscribeVisibleTimeRangeChange(redraw);}catch(_){}};
}

function _attachCBRTooltip(container,lwChart,mainSeries,fwdSeries,decisions){
  const tip=document.createElement('div');tip.className='cbr-lw-tooltip';container.style.position='relative';container.appendChild(tip);
  const TW=210,TM=12,decMap={};decisions.forEach(d=>{decMap[d.time]=d;});
  lwChart.subscribeCrosshairMove(param=>{
    if(!param?.point||!param.seriesData){tip.style.display='none';return;}
    const mv=param.seriesData.get(mainSeries);if(!mv){tip.style.display='none';return;}
    const timeStr=typeof param.time==='string'?param.time:'',dec=decMap[timeStr],rate=mv.value,mon=timeStr.slice(0,7);
    let html=`<div style="font-size:9px;color:#6e7681;letter-spacing:.05em;margin-bottom:4px;">${mon}</div>`;
    html+=`<div>Rate &nbsp;<span style="color:#4f7fff;font-weight:700;">${rate.toFixed(2)}%</span></div>`;
    if(dec){const col=dec.delta>0?'#26a69a':'#ef5350',sign=dec.delta>0?'+':'';html+=`<div style="margin-top:3px;color:${col};font-weight:600;">${sign}${Math.round(dec.delta*100)}bp decision</div>`;}
    if(fwdSeries){const fv=param.seriesData.get(fwdSeries);if(fv)html+=`<div style="color:#8b949e;">OIS fwd ${fv.value.toFixed(2)}%</div>`;}
    tip.innerHTML=html;tip.style.display='block';
    const cW=container.offsetWidth,cx=param.point.x,cy=param.point.y,th=tip.offsetHeight||60;
    const tx=(cx+TM+TW<=cW-4)?cx+TM:cx-TM-TW,ty=(cy-th-TM>=4)?cy-th-TM:cy+TM;
    tip.style.left=Math.max(0,tx)+'px';tip.style.top=Math.max(0,ty)+'px';
  });
}

function _buildCBRChart(data){
  const container=document.querySelector('.cbr-lw-wrap');if(!container)return;
  const LWC=window.LightweightCharts;if(!LWC){console.warn('CBR modal: LWC not loaded');return;}
  _destroyCBRChart();
  const W=container.offsetWidth||600,H=container.offsetHeight||240;
  _cbrLwChart=LWC.createChart(container,_cbrLwOptions(W,H));
  const{chronData,decisions,fwdRate,bias}=data;
  const mainSeries=_cbrLwChart.addSeries(LWC.AreaSeries,{lineColor:'#4f7fff',topColor:'rgba(79,127,255,0.18)',bottomColor:'rgba(79,127,255,0.01)',lineWidth:2,lineType:LWC.LineType?.WithSteps??1,crosshairMarkerVisible:true,crosshairMarkerRadius:4,crosshairMarkerBorderColor:'#161b22',crosshairMarkerBorderWidth:2,priceLineVisible:false,lastValueVisible:true});
  mainSeries.setData(chronData);
  let fwdSeries=null;
  if(fwdRate!=null&&chronData.length>0){
    const last=chronData[chronData.length-1],ld=new Date(last.time);
    ld.setMonth(ld.getMonth()+1);
    const fwdTime=ld.toISOString().slice(0,10);
    const fwdCol=bias==='cut'?'#ef5350':bias==='hike'?'#26a69a':'#6e7681';
    fwdSeries=_cbrLwChart.addSeries(LWC.LineSeries,{color:fwdCol,lineWidth:1,lineStyle:LWC.LineStyle?.Dashed??1,crosshairMarkerVisible:true,crosshairMarkerRadius:4,priceLineVisible:false,lastValueVisible:true});
    fwdSeries.setData([{time:last.time,value:last.value},{time:fwdTime,value:fwdRate}]);
  }
  _cbrLwChart.timeScale().fitContent();
  _buildDecisionOverlay(container,_cbrLwChart,decisions);
  _attachCBRTooltip(container,_cbrLwChart,mainSeries,fwdSeries,decisions);
  const resize=()=>{if(_cbrLwChart&&container.offsetWidth>0)_cbrLwChart.applyOptions({width:container.offsetWidth,height:container.offsetHeight});};
  window.addEventListener('resize',resize);container._cbrResize=resize;
}

async function openCBRatesModal(ccy,obs,bankInfo,meetingData){
  closeCBRatesModal();
  // If meetingData not yet available (race: CB Expectations fetch still in-flight), fetch directly
  if(!meetingData){
    try{
      const st=window._STATE_meetings;
      if(st?.meetings?.[ccy]){meetingData=st.meetings[ccy];}
      else{
        const res=await fetch('./meetings-data/meetings.json').then(r=>r.ok?r.json():null).catch(()=>null);
        if(res?.meetings?.[ccy])meetingData=res.meetings[ccy];
      }
    }catch(_){}
  }
  const{chronData,decisions}=_processCBRateData(obs);
  const rates=chronData.map(d=>d.value);
  const currentRate=rates[rates.length-1]??0,rateStart=rates[0]??0,totalChange=currentRate-rateStart;
  const nDecisions=decisions.length,nMonths=obs.length;
  const lastDec=decisions[decisions.length-1];
  // Bloomberg pause detection: if the last rate decision was >90 days ago, the
  // trend indicator resets to neutral (— On hold) regardless of prior direction.
  // This matches Bloomberg Terminal convention (corrected in GUIDELINES v7.13.45-46).
  const PAUSE_THRESHOLD_DAYS=90;
  let pauseDetected=false;
  if(lastDec){
    const lastDecMs=new Date(lastDec.time).getTime();
    const nowMs=Date.now();
    const daysSinceLast=(nowMs-lastDecMs)/(1000*60*60*24);
    if(daysSinceLast>PAUSE_THRESHOLD_DAYS)pauseDetected=true;
  }
  const trend=!lastDec||pauseDetected?'flat':lastDec.delta>0?'hiking':'cutting';
  const trendLabel=trend==='hiking'?'↑ Hiking cycle':trend==='cutting'?'↓ Cutting cycle':'— On hold';
  const trendCol=trend==='hiking'?'var(--up,#26a69a)':trend==='cutting'?'var(--down,#ef5350)':'#8b949e';
  const lastDir=lastDec?Math.sign(lastDec.delta):0;
  let cycleCum=0;
  for(let i=decisions.length-1;i>=0;i--){if(Math.sign(decisions[i].delta)===lastDir)cycleCum+=decisions[i].delta;else break;}
  const cycleStr=lastDir!==0?(lastDir>0?'+':'')+Math.round(cycleCum*100)+'bp this cycle':'—';
  const bias=meetingData?.bias??null,nextMtg=meetingData?.nextMeeting??'—';
  const biasLabel=bias==='cut'?'↓ Cut':bias==='hike'?'↑ Hike':'→ Hold';
  const biasCol=bias==='cut'?'var(--down,#ef5350)':bias==='hike'?'var(--up,#26a69a)':'#8b949e';
  let fwdRate=null,fwdDisplay='—',fwdIsEst=false;
  if(meetingData?.fwdRate!=null&&!isNaN(meetingData.fwdRate)&&meetingData.fwdRate>0){
    fwdRate=parseFloat(meetingData.fwdRate);fwdDisplay=fwdRate.toFixed(2)+'%';
  }else{
    const pCut=meetingData?.cutProb!=null?Math.min(100,Math.max(0,meetingData.cutProb)):null;
    const pHike=meetingData?.hikeProb!=null?Math.min(100,Math.max(0,meetingData.hikeProb)):null;
    if(pCut!==null||pHike!==null){
      const cut=pCut??0,hike=Math.min(pHike??0,100-cut);
      fwdRate=Math.max(0,currentRate+(hike/100)*0.25-(cut/100)*0.25);fwdDisplay=fwdRate.toFixed(2)+'%';
    }else{
      const step=bias==='cut'?-0.25:bias==='hike'?0.25:0;
      fwdRate=Math.max(0,currentRate+step);fwdDisplay='~'+fwdRate.toFixed(2)+'%';fwdIsEst=true;
    }
  }
  const bankName=bankInfo?.name||ccy,bankShort=bankInfo?.short||ccy;
  const flagHtml=bankInfo?.flag?`<span class="fi fi-${bankInfo.flag}" style="margin-right:8px;border-radius:2px;font-size:16px;vertical-align:middle;"></span>`:'';

  const bd=document.createElement('div');bd.id='cbr-bd';
  bd.innerHTML=`
<div id="cbr-modal">
  <div id="cbr-m-hd">
    <div>
      <div id="cbr-m-title">${flagHtml}${bankName} — Policy Rate History</div>
      <div id="cbr-m-sub">${nMonths} months · ${nDecisions} decisions · CB rates cache · monthly observations</div>
    </div>
    <button id="cbr-m-close" onclick="closeCBRatesModal()" aria-label="Close">✕</button>
  </div>
  <div id="cbr-m-metrics">
    <div class="cbr-mm"><div class="cbr-mm-lbl">Current Rate</div><div class="cbr-mm-val">${currentRate.toFixed(2)}%</div><div class="cbr-mm-sub">${bankShort}</div></div>
    <div class="cbr-mm"><div class="cbr-mm-lbl">Cycle</div><div class="cbr-mm-val" style="font-size:12px;color:${trendCol}">${trendLabel}</div><div class="cbr-mm-sub">${cycleStr}</div></div>
    <div class="cbr-mm"><div class="cbr-mm-lbl">Total Change</div><div class="cbr-mm-val ${totalChange>0?'cu':totalChange<0?'cd':'cf'}">${totalChange>0?'+':''}${Math.round(totalChange*100)}bp</div><div class="cbr-mm-sub">${nMonths}m period</div></div>
    <div class="cbr-mm"><div class="cbr-mm-lbl">Next Meeting</div><div class="cbr-mm-val cf" style="font-size:12px">${nextMtg}</div><div class="cbr-mm-sub" style="color:${biasCol}">${biasLabel}</div></div>
    <div class="cbr-mm" title="${fwdIsEst?'Bias-only estimate — no OIS probability data. Not a CIP-derived forward rate.':'OIS-implied forward rate (Covered Interest Parity convention)'}">
      <div class="cbr-mm-lbl">Fwd Rate</div>
      <div class="cbr-mm-val ${bias==='cut'?'cd':bias==='hike'?'cu':'cf'}">${fwdDisplay}</div>
      <div class="cbr-mm-sub">${fwdIsEst?'~ est · bias only · no OIS':'OIS implied · CIP'}</div>
    </div>
  </div>
  <div id="cbr-m-tabs" role="tablist" aria-label="CB rate chart views">
    <div class="cbr-tab on" data-tab="chart"    onclick="cbRatesTab(this,'chart')"    role="tab" aria-selected="true">Rate Chart</div>
    <div class="cbr-tab"    data-tab="decisions" onclick="cbRatesTab(this,'decisions')" role="tab" aria-selected="false">Decisions</div>
  </div>
  <div id="cbr-m-body">
    <div id="cbr-p-chart" class="cbr-panel on">
      <div class="cbr-cw fill">
        <div class="cbr-ct">POLICY RATE · MONTHLY · STEP CHART · DECISION MARKERS · OIS FORWARD</div>
        <div class="cbr-chart-area"><div class="cbr-lw-wrap"></div></div>
      </div>
    </div>
    <div id="cbr-p-decisions" class="cbr-panel">
      <div class="cbr-cw">
        <div class="cbr-ct">RATE DECISIONS · ${nDecisions} CHANGES IN ${nMonths} MONTHS</div>
        <div style="overflow-x:auto;">
          <table class="cbr-tbl">
            <thead><tr><th style="text-align:left">Date</th><th>Before</th><th>After</th><th>Change</th><th>Cumulative</th></tr></thead>
            <tbody id="cbr-decisions-body"></tbody>
          </table>
        </div>
      </div>
      <div class="cbr-cw" style="margin-top:4px;">
        <div class="cbr-ct">RATE PROFILE · ${rateStart.toFixed(2)}% → ${currentRate.toFixed(2)}%</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden;">
          <div style="background:#161b22;padding:10px 14px;"><div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_cbrMonoF}">Period High</div><div style="font-size:18px;font-weight:600;font-family:${_cbrMonoF};color:var(--up,#26a69a)">${Math.max(...rates).toFixed(2)}%</div></div>
          <div style="background:#161b22;padding:10px 14px;"><div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_cbrMonoF}">Period Low</div><div style="font-size:18px;font-weight:600;font-family:${_cbrMonoF};color:var(--down,#ef5350)">${Math.min(...rates).toFixed(2)}%</div></div>
          <div style="background:#161b22;padding:10px 14px;"><div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;font-family:${_cbrMonoF}">Decisions</div><div style="font-size:18px;font-weight:600;font-family:${_cbrMonoF};color:#e6edf3">${nDecisions}</div></div>
        </div>
      </div>
    </div>
  </div>
</div>`;

  document.body.appendChild(bd);

  const dBody=document.getElementById('cbr-decisions-body');
  if(dBody&&decisions.length){
    let running=0;
    const cumuls=decisions.map(d=>{running+=d.delta;return running;});
    const rev=[...decisions].map((d,i)=>({...d,cumul:cumuls[decisions.length-1-i]})).reverse();
    dBody.innerHTML=rev.map((d,i)=>{
      const before=(d.rate-d.delta).toFixed(2),after=d.rate.toFixed(2);
      const chgBp=Math.round(d.delta*100),cumBp=Math.round(d.cumul*100);
      return`<tr${i===0?' class="now-row"':''}><td>${d.time.slice(0,7)}${i===0?' <span style="color:var(--up,#26a69a);font-size:9px">latest</span>':''}</td><td>${before}%</td><td>${after}%</td><td class="${chgBp>0?'cu':chgBp<0?'cd':'cf'}">${chgBp>0?'+':''}${chgBp}bp</td><td class="${cumBp>0?'cu':cumBp<0?'cd':'cf'}">${cumBp>0?'+':''}${cumBp}bp</td></tr>`;
    }).join('');
  }else if(dBody){dBody.innerHTML=`<tr><td colspan="5" style="color:#6e7681;padding:12px 8px;">No rate changes in the available history.</td></tr>`;}

  bd.addEventListener('click',e=>{if(e.target===bd)closeCBRatesModal();});
  const esc=e=>{if(e.key==='Escape')closeCBRatesModal();};
  document.addEventListener('keydown',esc);bd._esc=esc;
  bd._chartData={chronData,decisions,fwdRate,bias,currentRate};
  requestAnimationFrame(()=>requestAnimationFrame(()=>_buildCBRChart(bd._chartData)));
}

function cbRatesTab(el,tabId){
  document.querySelectorAll('.cbr-tab').forEach(t=>{t.classList.remove('on');t.setAttribute('aria-selected','false');});
  document.querySelectorAll('.cbr-panel').forEach(p=>p.classList.remove('on'));
  el.classList.add('on');el.setAttribute('aria-selected','true');
  const panel=document.getElementById('cbr-p-'+tabId);if(panel)panel.classList.add('on');
  const body=document.getElementById('cbr-m-body');if(body)body.classList.toggle('cbr-body--history',tabId==='decisions');
  if(tabId==='chart'){const bd=document.getElementById('cbr-bd');if(bd?._chartData)requestAnimationFrame(()=>requestAnimationFrame(()=>_buildCBRChart(bd._chartData)));}
}

function closeCBRatesModal(){
  const bd=document.getElementById('cbr-bd');
  if(bd){if(bd._esc)document.removeEventListener('keydown',bd._esc);bd.remove();}
  const wrap=document.querySelector('.cbr-lw-wrap');
  if(wrap?._cbrResize)window.removeEventListener('resize',wrap._cbrResize);
  const svg=wrap?.querySelector('.cbr-decision-svg');if(svg?._cleanup)svg._cleanup();
  _destroyCBRChart();
}

window.openCBRatesModal=openCBRatesModal;window.closeCBRatesModal=closeCBRatesModal;window.cbRatesTab=cbRatesTab;
