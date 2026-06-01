/**
 * Global Investing FX Terminal — Bloomberg-style Panel Manager
 * panel-manager2.js  (PROTOTYPE — index2.html only)
 *
 * Implements free-floating, resizable, draggable panel windows
 * inspired by Bloomberg Terminal / OpenBB / Godel Terminal UX.
 *
 * Architecture:
 *  All original dashboard HTML panels (#section-*, #narrative, etc.)
 *  are MOVED into floating .gi-panel containers at init time —
 *  no content is duplicated. dashboard.js runs unchanged; all IDs remain.
 */

(function () {
  'use strict';

  /* ── Constants ─────────────────────────────────────────────────────── */
  var STORAGE_KEY  = 'gi_panel_layout_v2';
  var SNAP_GRID    = 8;
  var MIN_W        = 240;
  var MIN_H        = 120;
  var MOBILE_BREAK = 900;

  /* ── Panel definitions ─────────────────────────────────────────────── */
  var PANEL_DEFS = [
    { id:'gipm-chart',       title:'Price Chart',            srcId:'section-fxpairs',      dW:700,  dH:360, dX:0,    dY:0   },
    { id:'gipm-fxtable',     title:'FX Pairs Table',         srcId:'section-fxtable',      dW:1160, dH:240, dX:0,    dY:368 },
    { id:'gipm-narrative',   title:'AI Narrative',           srcId:'narrative',             dW:440,  dH:100, dX:708,  dY:0   },
    { id:'gipm-crossasset',  title:'Cross-Asset / Risk',     srcId:'_group_crossrisk',     dW:440,  dH:260, dX:708,  dY:108 },
    { id:'gipm-rates',       title:'Rates & Yield Curve',    srcId:'section-rates',        dW:580,  dH:300, dX:0,    dY:616 },
    { id:'gipm-positioning', title:'COT Positioning',        srcId:'section-positioning',  dW:580,  dH:260, dX:0,    dY:924 },
    { id:'gipm-sessions',    title:'Sessions & Liquidity',   srcId:'_group_sessions',      dW:580,  dH:220, dX:0,    dY:1192 },
    { id:'gipm-sentiment',   title:'Retail Sentiment',       srcId:'section-sentiment',    dW:440,  dH:200, dX:708,  dY:376 },
    { id:'gipm-calendar',    title:'Economic Calendar',      srcId:'section-tvcalendar',   dW:440,  dH:200, dX:708,  dY:584 },
    { id:'gipm-macro',       title:'Alerts & Signals',       srcId:'section-macro',        dW:440,  dH:200, dX:708,  dY:792 },
    { id:'gipm-econmap',     title:'Economic Map',           srcId:'section-econmap',      dW:700,  dH:460, dX:0,    dY:1420 },
    { id:'gipm-derivatives', title:'Derivatives',            srcId:'section-derivatives',  dW:740,  dH:440, dX:0,    dY:1892 },
    { id:'gipm-news',        title:'News Feed',              srcId:'section-news',         dW:440,  dH:400, dX:708,  dY:1000 },
    { id:'gipm-cbrates',     title:'CB Rates',               srcId:'_group_cbrates',       dW:300,  dH:260, dX:1160, dY:0   },
    { id:'gipm-corr',        title:'Key Correlations',       srcId:'_group_corr',          dW:300,  dH:300, dX:1160, dY:268 },
    { id:'gipm-cbexp',       title:'CB Rate Expectations',   srcId:'section-cb-expectations', dW:300, dH:280, dX:1160, dY:576 },
    { id:'gipm-surprises',   title:'Economic Surprises',     srcId:'section-econ-surprise',dW:300,  dH:240, dX:1160, dY:864 },
    { id:'gipm-sidebar',     title:'FX Watch / Liquidity',  srcId:'_sidebar',              dW:180,  dH:600, dX:1160, dY:1112 },
  ];

  /* ── State ──────────────────────────────────────────────────────────── */
  var panels    = {};
  var zTop      = 100;
  var workspace = null;
  var taskbar   = null;

  /* ── Utils ──────────────────────────────────────────────────────────── */
  function snap(v) { return Math.round(v / SNAP_GRID) * SNAP_GRID; }
  function isMobile() { return window.innerWidth < MOBILE_BREAK; }
  function loadLayout() {
    try { var s = localStorage.getItem(STORAGE_KEY); return s ? JSON.parse(s) : {}; }
    catch (e) { return {}; }
  }
  function saveLayout() {
    var out = {};
    Object.keys(panels).forEach(function(id) {
      var p = panels[id];
      out[id] = { x:p.state.x, y:p.state.y, w:p.state.w, h:p.state.h,
                  minimized:p.state.minimized, hidden:p.state.hidden, maximized:p.state.maximized };
    });
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(out)); } catch(e) {}
  }

  /* ── Source node extraction ─────────────────────────────────────────── */
  function getSourceNode(srcId) {
    /* Regular IDs */
    if (srcId[0] !== '_') return document.getElementById(srcId);

    /* Virtual groups */
    var wrap = document.createElement('div');
    wrap.className = 'gi-panel-group-wrap';

    if (srcId === '_group_crossrisk') {
      /* section-crossasset + section-risk (they live inside a flex row wrapper) */
      var ca   = document.getElementById('section-crossasset');
      var risk = document.getElementById('section-risk');
      if (!ca) return null;
      var parent = ca.parentNode;
      parent.parentNode.insertBefore(wrap, parent);
      wrap.appendChild(parent);   /* move the entire flex row */
      return wrap;
    }

    if (srcId === '_group_sessions') {
      /* section-sessions + section-tvsessions — they sit in a flex row */
      var sess = document.getElementById('section-sessions');
      if (!sess) return null;
      var sesParent = sess.parentNode;
      sesParent.parentNode.insertBefore(wrap, sesParent);
      wrap.appendChild(sesParent);
      return wrap;
    }

    if (srcId === '_group_cbrates') {
      var head = document.getElementById('section-cbrates');
      if (!head) return null;
      var tbl = head.nextElementSibling;
      head.parentNode.insertBefore(wrap, head);
      wrap.appendChild(head);
      if (tbl && tbl.tagName === 'TABLE') wrap.appendChild(tbl);
      return wrap;
    }

    if (srcId === '_group_corr') {
      /* The sb-head "Key Correlations" + table + footer note */
      var corrTbl = document.querySelector('table[aria-label="Key FX correlations"]');
      if (!corrTbl) return null;
      var corrHead = corrTbl.previousElementSibling;
      var corrNote = corrTbl.nextElementSibling;
      corrTbl.parentNode.insertBefore(wrap, corrHead || corrTbl);
      if (corrHead && corrHead.classList && corrHead.classList.contains('sb-head')) wrap.appendChild(corrHead);
      wrap.appendChild(corrTbl);
      if (corrNote && corrNote.nodeType === 1 && corrNote.style && corrNote.getAttribute('style')) wrap.appendChild(corrNote);
      return wrap;
    }

    if (srcId === '_sidebar') {
      var sb = document.getElementById('sidebar');
      return sb;
    }

    return null;
  }

  /* ── Workspace ──────────────────────────────────────────────────────── */
  function buildScaffold() {
    /* Workspace */
    workspace = document.createElement('div');
    workspace.id        = 'gi-workspace';
    workspace.className = 'gi-workspace';

    /* Adjust top to sit below topbar + quotebar + ticker */
    function setWorkspaceTop() {
      var topbar  = document.getElementById('topbar');
      var ticker  = document.getElementById('tv-ticker-wrap');
      var newsTkr = document.getElementById('news-ticker');
      var h = 0;
      if (topbar)  h += topbar.offsetHeight;
      if (ticker)  h += ticker.offsetHeight;
      if (newsTkr) h += newsTkr.offsetHeight;
      workspace.style.top = h + 'px';
    }
    setWorkspaceTop();
    window.addEventListener('resize', setWorkspaceTop);

    /* Taskbar */
    taskbar = document.createElement('div');
    taskbar.id        = 'gi-taskbar';
    taskbar.className = 'gi-taskbar';

    var resetBtn = document.createElement('button');
    resetBtn.className   = 'gi-taskbar-reset';
    resetBtn.textContent = 'Reset Layout';
    resetBtn.title       = 'Restore all panels to default positions';
    resetBtn.addEventListener('click', resetAllPanels);
    taskbar.appendChild(resetBtn);

    document.body.appendChild(workspace);
    document.body.appendChild(taskbar);

    /* "Panels" button in topbar */
    var topbarRight = document.querySelector('.topbar-right');
    if (topbarRight) {
      var btn = document.createElement('button');
      btn.id        = 'gi-panels-btn';
      btn.className = 'gi-panels-btn';
      btn.title     = 'Toggle panel visibility (Bloomberg panel mode)';
      btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none">'
        + '<rect x="0" y="0" width="5" height="5" rx=".8" fill="currentColor"/>'
        + '<rect x="7" y="0" width="5" height="5" rx=".8" fill="currentColor"/>'
        + '<rect x="0" y="7" width="5" height="5" rx=".8" fill="currentColor"/>'
        + '<rect x="7" y="7" width="5" height="5" rx=".8" fill="currentColor"/>'
        + '</svg>&nbsp;Panels';
      btn.addEventListener('click', function(e) { e.stopPropagation(); togglePanelDropdown(); });
      topbarRight.insertBefore(btn, topbarRight.firstChild);
    }

    /* Prototype badge */
    var badge = document.getElementById('gi-proto-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'gi-proto-badge';
      badge.textContent = 'PROTOTYPE · PANEL MODE';
      document.body.appendChild(badge);
    }
  }

  /* ── Panel dropdown ─────────────────────────────────────────────────── */
  function togglePanelDropdown() {
    var ex = document.getElementById('gi-pd');
    if (ex) { ex.remove(); return; }

    var dd = document.createElement('div');
    dd.id        = 'gi-pd';
    dd.className = 'gi-panels-dropdown';

    var hdr = document.createElement('div');
    hdr.className   = 'gi-pd-header';
    hdr.textContent = 'PANELS';
    dd.appendChild(hdr);

    PANEL_DEFS.forEach(function(def) {
      var p = panels[def.id];
      if (!p) return;
      var row = document.createElement('div');
      row.className = 'gi-pd-row';

      var chk = document.createElement('span');
      chk.className = 'gi-pd-check';
      chk.innerHTML = p.state.hidden ? '' : '&#10003;';

      var lbl = document.createElement('span');
      lbl.className   = 'gi-pd-label';
      lbl.textContent = def.title;

      row.appendChild(chk);
      row.appendChild(lbl);
      row.addEventListener('click', function() {
        if (p.state.hidden) showPanel(def.id); else hidePanel(def.id);
        chk.innerHTML = p.state.hidden ? '' : '&#10003;';
      });
      dd.appendChild(row);
    });

    var showAll = document.createElement('button');
    showAll.className   = 'gi-pd-showall';
    showAll.textContent = 'Show All Panels';
    showAll.addEventListener('click', function() {
      PANEL_DEFS.forEach(function(d){ showPanel(d.id); });
      dd.remove();
    });
    dd.appendChild(showAll);

    var refBtn = document.getElementById('gi-panels-btn');
    var r = refBtn.getBoundingClientRect();
    dd.style.top   = (r.bottom + 4) + 'px';
    dd.style.right = (window.innerWidth - r.right) + 'px';
    document.body.appendChild(dd);

    setTimeout(function() {
      document.addEventListener('click', function h(e) {
        if (!dd.contains(e.target) && e.target !== refBtn) { dd.remove(); document.removeEventListener('click', h); }
      });
    }, 10);
  }

  /* ── Create panel window ────────────────────────────────────────────── */
  function createPanel(def, savedState) {
    var state = Object.assign({
      x: def.dX, y: def.dY, w: def.dW, h: def.dH,
      minimized: false, hidden: false, maximized: false,
    }, savedState || {});

    var el = document.createElement('div');
    el.id        = def.id;
    el.className = 'gi-panel';
    el.setAttribute('role', 'region');
    el.setAttribute('aria-label', def.title);
    el.style.cssText = 'left:'+state.x+'px;top:'+state.y+'px;width:'+state.w+'px;height:'+state.h+'px;z-index:'+(++zTop)+';';

    /* ── Titlebar ── */
    var tb = document.createElement('div');
    tb.className = 'gi-panel-titlebar';

    var accent = document.createElement('div');
    accent.className = 'gi-panel-accent';
    tb.appendChild(accent);

    var titleSpan = document.createElement('span');
    titleSpan.className   = 'gi-panel-title';
    titleSpan.textContent = def.title;
    tb.appendChild(titleSpan);

    var controls = document.createElement('div');
    controls.className = 'gi-panel-controls';

    function mkBtn(cls, html, tip, fn) {
      var b = document.createElement('button');
      b.className = 'gi-panel-btn ' + cls;
      b.innerHTML = html;
      b.title     = tip;
      b.setAttribute('aria-label', tip);
      b.addEventListener('click', function(e){ e.stopPropagation(); fn(); });
      return b;
    }

    controls.appendChild(mkBtn('gi-panel-min',   '&#8722;', 'Minimize', function(){ minimizePanel(def.id); }));
    controls.appendChild(mkBtn('gi-panel-max',   '&#9633;', 'Maximize', function(){ toggleMaximize(def.id); }));
    controls.appendChild(mkBtn('gi-panel-close', '&#215;',  'Hide',     function(){ hidePanel(def.id); }));
    tb.appendChild(controls);
    el.appendChild(tb);

    /* ── Content ── */
    var content = document.createElement('div');
    content.className = 'gi-panel-content';

    var srcNode = getSourceNode(def.srcId);
    if (srcNode) {
      content.appendChild(srcNode);
    } else {
      content.innerHTML = '<div style="padding:14px;color:#5a6070;font-size:10px;font-family:var(--font-ui);">' + def.title + ' — loading…</div>';
    }
    el.appendChild(content);

    /* ── Resize handles ── */
    ['n','ne','e','se','s','sw','w','nw'].forEach(function(dir) {
      var rh = document.createElement('div');
      rh.className     = 'gi-resize-handle gi-rh-' + dir;
      rh.dataset.dir   = dir;
      bindResize(el, rh, def.id);
      el.appendChild(rh);
    });

    /* ── Interactions ── */
    bindDrag(el, tb, def.id);
    tb.addEventListener('dblclick', function() { toggleMaximize(def.id); });
    el.addEventListener('mousedown', function() { focusPanel(def.id); }, true);

    workspace.appendChild(el);

    /* Apply persisted state */
    if (state.hidden)    el.style.display = 'none';
    if (state.minimized) doMinimize(def.id, el, def.title);
    if (state.maximized) doMaximize(def.id, el, state);

    panels[def.id] = { el:el, state:state, def:def };
  }

  /* ── Focus ──────────────────────────────────────────────────────────── */
  function focusPanel(id) {
    if (panels[id]) panels[id].el.style.zIndex = ++zTop;
  }

  /* ── Show/Hide ──────────────────────────────────────────────────────── */
  function hidePanel(id) {
    var p = panels[id]; if (!p) return;
    p.state.hidden = true;
    p.el.style.display = 'none';
    saveLayout();
  }
  function showPanel(id) {
    var p = panels[id]; if (!p) return;
    p.state.hidden    = false;
    p.state.minimized = false;
    p.el.style.display = '';
    p.el.classList.remove('gi-panel-minimized');
    var tb = document.getElementById('gi-task-' + id);
    if (tb) tb.remove();
    focusPanel(id);
    saveLayout();
  }

  /* ── Minimize ───────────────────────────────────────────────────────── */
  function doMinimize(id, el, title) {
    el.classList.add('gi-panel-minimized');
    if (!document.getElementById('gi-task-' + id)) {
      var btn = document.createElement('button');
      btn.id          = 'gi-task-' + id;
      btn.className   = 'gi-taskbar-btn';
      btn.textContent = title;
      btn.title       = 'Restore ' + title;
      btn.addEventListener('click', function() { showPanel(id); });
      taskbar.insertBefore(btn, taskbar.querySelector('.gi-taskbar-reset'));
    }
  }
  function minimizePanel(id) {
    var p = panels[id]; if (!p || p.state.minimized) return;
    p.state.minimized = true;
    doMinimize(id, p.el, p.def.title);
    saveLayout();
  }

  /* ── Maximize ───────────────────────────────────────────────────────── */
  function doMaximize(id, el, state) {
    state._pre = { x:state.x, y:state.y, w:state.w, h:state.h };
    el.classList.add('gi-panel-maximized');
    el.style.left = '0'; el.style.top  = '0';
    el.style.width = '100%'; el.style.height = '100%';
    el.style.zIndex = ++zTop;
  }
  function toggleMaximize(id) {
    var p = panels[id]; if (!p) return;
    if (p.state.maximized) {
      p.state.maximized = false;
      p.el.classList.remove('gi-panel-maximized');
      if (p.state._pre) {
        var pr = p.state._pre;
        p.el.style.left = pr.x+'px'; p.el.style.top  = pr.y+'px';
        p.el.style.width = pr.w+'px'; p.el.style.height = pr.h+'px';
        Object.assign(p.state, pr);
      }
    } else {
      p.state.maximized = true;
      doMaximize(id, p.el, p.state);
    }
    saveLayout();
    setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 50);
  }

  /* ── Reset all ──────────────────────────────────────────────────────── */
  function resetAllPanels() {
    try { localStorage.removeItem(STORAGE_KEY); } catch(e) {}
    PANEL_DEFS.forEach(function(def) {
      var p = panels[def.id]; if (!p) return;
      Object.assign(p.state, { x:def.dX, y:def.dY, w:def.dW, h:def.dH,
        minimized:false, hidden:false, maximized:false });
      p.el.style.cssText = 'left:'+def.dX+'px;top:'+def.dY+'px;width:'+def.dW+'px;height:'+def.dH+'px;z-index:'+(++zTop)+';';
      p.el.classList.remove('gi-panel-minimized','gi-panel-maximized');
      var tb = document.getElementById('gi-task-' + def.id);
      if (tb) tb.remove();
    });
  }

  /* ── Drag ────────────────────────────────────────────────────────────── */
  function bindDrag(el, handle, id) {
    function startDrag(clientX, clientY) {
      var p = panels[id];
      if (!p || p.state.maximized) return;
      focusPanel(id);
      var sx = clientX, sy = clientY;
      var sl = p.state.x, st = p.state.y;
      el.classList.add('gi-panel-dragging');

      function onMove(e2) {
        var cx = e2.touches ? e2.touches[0].clientX : e2.clientX;
        var cy = e2.touches ? e2.touches[0].clientY : e2.clientY;
        var nx = Math.max(0, snap(sl + cx - sx));
        var ny = Math.max(0, snap(st + cy - sy));
        p.state.x = nx; p.state.y = ny;
        el.style.left = nx+'px'; el.style.top = ny+'px';
      }
      function onEnd() {
        el.classList.remove('gi-panel-dragging');
        saveLayout();
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onEnd);
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend',  onEnd);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onEnd);
      document.addEventListener('touchmove', onMove, { passive:true });
      document.addEventListener('touchend',  onEnd);
    }

    handle.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      if (e.target.closest('.gi-panel-controls')) return;
      e.preventDefault();
      startDrag(e.clientX, e.clientY);
    });
    handle.addEventListener('touchstart', function(e) {
      if (e.target.closest('.gi-panel-controls')) return;
      startDrag(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive:true });
  }

  /* ── Resize ──────────────────────────────────────────────────────────── */
  function bindResize(el, handle, id) {
    handle.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      var p = panels[id];
      if (!p || p.state.maximized) return;
      e.preventDefault(); e.stopPropagation();
      focusPanel(id);
      var dir = handle.dataset.dir;
      var sx = e.clientX, sy = e.clientY;
      var sl = p.state.x, st = p.state.y, sw = p.state.w, sh = p.state.h;
      document.body.style.userSelect = 'none';

      function onMove(ev) {
        var dx = ev.clientX - sx, dy = ev.clientY - sy;
        var nx=sl, ny=st, nw=sw, nh=sh;
        if (dir.includes('e'))  nw = Math.max(MIN_W, snap(sw + dx));
        if (dir.includes('s'))  nh = Math.max(MIN_H, snap(sh + dy));
        if (dir.includes('w')) { nw = Math.max(MIN_W, snap(sw - dx)); nx = snap(sl + sw - nw); }
        if (dir.includes('n')) { nh = Math.max(MIN_H, snap(sh - dy)); ny = snap(st + sh - nh); }
        p.state.x=nx; p.state.y=ny; p.state.w=nw; p.state.h=nh;
        el.style.cssText = 'left:'+nx+'px;top:'+ny+'px;width:'+nw+'px;height:'+nh+'px;z-index:'+el.style.zIndex+';';
      }
      function onUp() {
        document.body.style.userSelect = '';
        saveLayout();
        window.dispatchEvent(new Event('resize'));
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  }

  /* ── Init ────────────────────────────────────────────────────────────── */
  function init() {
    if (isMobile()) return;

    var origLayout = document.getElementById('layout');
    if (!origLayout) return;

    buildScaffold();

    var saved = loadLayout();

    PANEL_DEFS.forEach(function(def) {
      createPanel(def, saved[def.id]);
    });

    /* Hide original grid layout (now mostly emptied) */
    origLayout.style.visibility = 'hidden';
    origLayout.style.pointerEvents = 'none';
    origLayout.style.position = 'absolute';
    origLayout.style.left = '-9999px';

    /* Keyboard shortcuts: Alt+1-9 to focus panels */
    document.addEventListener('keydown', function(e) {
      if (!e.altKey) return;
      var idx = parseInt(e.key, 10);
      if (!isNaN(idx) && idx >= 1 && idx <= PANEL_DEFS.length) {
        var pid = PANEL_DEFS[idx-1].id;
        showPanel(pid);
        focusPanel(pid);
        e.preventDefault();
      }
    });

    /* Reflow charts after panels settle */
    setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 400);
    setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 1200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 0);
  }

})();
