/**
 * Global Investing FX Terminal — Layout Resize Handles + Panel Collapse
 * v7.72.0
 *
 * 1. RESIZE HORIZONTAL — handles arrastrables entre las 3 columnas del grid
 * 2. RESIZE VERTICAL   — handle en el borde inferior del chart (#tv-chart-wrap)
 * 3. PANEL COLLAPSE    — botón ▾/▸ en cada panel-head; colapsa hijos directos
 *
 * Storage:
 *   gi_layout_sidebar_w   gi_layout_right_w   gi_layout_chart_h
 *   gi_panel_collapsed_<id>
 */
(function () {
  'use strict';

  var SK_SIDEBAR  = 'gi_layout_sidebar_w';
  var SK_RIGHT    = 'gi_layout_right_w';
  var SK_CHART_H  = 'gi_layout_chart_h';
  var SK_COL      = 'gi_panel_collapsed_';

  var DEF_SIDEBAR = 180, MIN_SIDEBAR = 120, MAX_SIDEBAR = 320;
  var DEF_RIGHT   = 220, MIN_RIGHT   = 150, MAX_RIGHT   = 400;
  var DEF_CHART_H = 290, MIN_CHART_H = 120, MAX_CHART_H = 600;
  var MIN_MAIN    = 320;
  var MOBILE      = 900;

  /* Panels to collapse.
   * contentSel: selectors of DIRECT children to hide (comma-separated).
   * If '*', all direct children except panel-head are hidden.             */
  var PANELS = [
    { id: 'narrative',           headSel: '.narr-label',  contentSel: '.narr-text,.narr-meta',                       label: 'Narrative' },
    { id: 'section-fxpairs',     headSel: '.panel-head',  contentSel: '#lw-range-bar,#lw-chart-header,#tv-chart-wrap', label: 'Price Chart' },
    { id: 'section-fxtable',     headSel: '.panel-head',  contentSel: '*',                                            label: 'FX Pairs' },
    { id: 'split-upper-heatmap', headSel: '.panel-head',  contentSel: '#heatmap-grid',                                label: 'Currency Heatmap' },
    { id: 'section-rates',       headSel: '.panel-head',  contentSel: '#rates-grid-wrap,.yield-canvas-wrap,*',        label: 'Rates & Yield Curve' },
    { id: 'section-positioning', headSel: '.panel-head',  contentSel: '*',                                            label: 'COT Positioning' },
    { id: 'section-macro',       headSel: '.panel-head',  contentSel: '#alerts-container',                            label: 'Alerts & Signals' },
  ];

  /* ─── Helpers ─────────────────────────────────────────────────────────── */
  function isMobile() { return window.innerWidth <= MOBILE; }
  function load(k, d) { try { var v = localStorage.getItem(k); return v !== null ? parseInt(v, 10) : d; } catch { return d; } }
  function loadB(k)   { try { return localStorage.getItem(k) === '1'; } catch { return false; } }
  function save(k, v) { try { localStorage.setItem(k, String(v)); } catch {} }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function toast(msg) {
    var t = document.getElementById('gi-toast');
    if (!t) { t = document.createElement('div'); t.id = 'gi-toast'; document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add('visible');
    clearTimeout(t._t);
    t._t = setTimeout(function () { t.classList.remove('visible'); }, 1800);
  }

  /* ─── Content elements for a panel ───────────────────────────────────── */
  function getContent(panel, headEl, cfg) {
    if (cfg.contentSel === '*') {
      // All direct children except the head
      return Array.from(panel.children).filter(function (c) { return c !== headEl; });
    }
    var found = [];
    var seen  = new Set();
    cfg.contentSel.split(',').forEach(function (sel) {
      var el = panel.querySelector(sel.trim());
      if (el && !seen.has(el)) { seen.add(el); found.push(el); }
    });
    // Fallback if nothing matched
    if (found.length === 0) {
      found = Array.from(panel.children).filter(function (c) { return c !== headEl; });
    }
    return found;
  }

  /* ─── Animate collapse / expand ───────────────────────────────────────── */
  function collapse(els, instant) {
    els.forEach(function (el) {
      if (instant) { el.style.display = 'none'; return; }
      el.style.overflow   = 'hidden';
      el.style.maxHeight  = el.scrollHeight + 'px';
      el.style.opacity    = '1';
      el.offsetHeight; // reflow
      el.style.transition = 'max-height .22s ease, opacity .18s ease';
      el.style.maxHeight  = '0';
      el.style.opacity    = '0';
      setTimeout(function () {
        el.style.display = 'none';
        el.style.cssText = el.style.cssText.replace(/transition[^;]+;?/g, '');
        el.style.maxHeight = '';
        el.style.overflow  = '';
        el.style.opacity   = '';
      }, 230);
    });
  }

  function expand(els) {
    els.forEach(function (el) {
      el.style.display = '';
      var target = el.scrollHeight;
      el.style.overflow   = 'hidden';
      el.style.maxHeight  = '0';
      el.style.opacity    = '0';
      el.offsetHeight;
      el.style.transition = 'max-height .22s ease, opacity .18s ease';
      el.style.maxHeight  = target + 'px';
      el.style.opacity    = '1';
      setTimeout(function () {
        el.style.cssText = el.style.cssText.replace(/transition[^;]+;?/g, '');
        el.style.maxHeight = '';
        el.style.overflow  = '';
        el.style.opacity   = '';
        window.dispatchEvent(new Event('resize'));
      }, 230);
    });
  }

  /* ─── 3. Panel Collapse ───────────────────────────────────────────────── */
  function initCollapse() {
    PANELS.forEach(function (cfg) {
      var panel = document.getElementById(cfg.id);
      if (!panel) return;
      var head = panel.querySelector(cfg.headSel);
      if (!head) return;

      // Make head flex so button sits at far right
      head.style.display    = 'flex';
      head.style.alignItems = 'center';

      var btn = document.createElement('button');
      btn.className = 'gi-collapse-btn';
      btn.title     = 'Collapse / expand · Double-click header to collapse all';
      btn.setAttribute('aria-label', 'Collapse ' + cfg.label);

      var isCollapsed = loadB(SK_COL + cfg.id);
      btn.innerHTML = isCollapsed ? '&#9656;' : '&#9662;'; // ▶ or ▾
      head.appendChild(btn);

      var content = getContent(panel, head, cfg);

      if (isCollapsed) {
        collapse(content, true);
        panel.classList.add('gi-collapsed');
      }

      function doCollapse() {
        panel.classList.add('gi-collapsed');
        collapse(content, false);
        btn.innerHTML = '&#9656;';
        btn.setAttribute('aria-label', 'Expand ' + cfg.label);
        save(SK_COL + cfg.id, '1');
      }
      function doExpand() {
        panel.classList.remove('gi-collapsed');
        expand(content);
        btn.innerHTML = '&#9662;';
        btn.setAttribute('aria-label', 'Collapse ' + cfg.label);
        save(SK_COL + cfg.id, '0');
      }

      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        panel.classList.contains('gi-collapsed') ? doExpand() : doCollapse();
      });

      // Store refs for "collapse all"
      panel._giCollapse = doCollapse;
      panel._giExpand   = doExpand;
      panel._giIsCollapsed = function () { return panel.classList.contains('gi-collapsed'); };

      // Double-click on header = toggle all
      head.addEventListener('dblclick', function (e) {
        if (e.target === btn) return;
        toggleAll();
      });
    });
  }

  var _allCollapsed = false;
  function toggleAll() {
    _allCollapsed = !_allCollapsed;
    PANELS.forEach(function (cfg) {
      var panel = document.getElementById(cfg.id);
      if (!panel) return;
      if (_allCollapsed && panel._giCollapse && !panel._giIsCollapsed()) panel._giCollapse();
      if (!_allCollapsed && panel._giExpand && panel._giIsCollapsed()) panel._giExpand();
    });
    toast(_allCollapsed ? 'All panels collapsed' : 'All panels expanded');
  }
  window.giToggleAllPanels = toggleAll;

  /* ─── 1. Horizontal resize ────────────────────────────────────────────── */
  function initHorizontal() {
    var layout     = document.getElementById('layout');
    var main       = document.getElementById('main');
    var rightpanel = document.getElementById('rightpanel');
    var sidebar    = document.getElementById('sidebar');
    if (!layout || !main || !rightpanel || !sidebar) return;

    var sw = clamp(load(SK_SIDEBAR, DEF_SIDEBAR), MIN_SIDEBAR, MAX_SIDEBAR);
    var rw = clamp(load(SK_RIGHT,   DEF_RIGHT),   MIN_RIGHT,   MAX_RIGHT);

    function tmpl(s, r) {
      return layout.classList.contains('layout--no-right')
        ? s + 'px 4px 1fr 4px 0'
        : s + 'px 4px 1fr 4px ' + r + 'px';
    }

    var hL = mkHHandle('gi-hh-left',  'Resize sidebar');
    var hR = mkHHandle('gi-hh-right', 'Resize right panel');
    layout.insertBefore(hL, main);
    layout.insertBefore(hR, rightpanel);
    layout.style.gridTemplateColumns = tmpl(sw, rw);

    function drag(handle, calcFn) {
      handle.addEventListener('mousedown', function (e) {
        e.preventDefault();
        var cols   = layout.style.gridTemplateColumns.split(' ');
        var cSW    = parseInt(cols[0], 10) || sw;
        var cRW    = parseInt(cols[4], 10) || rw;
        var totalW = layout.offsetWidth;
        var startX = e.clientX;
        document.body.classList.add('gi-resizing');

        function mv(ev) {
          var r = calcFn(ev.clientX - startX, cSW, cRW, totalW);
          layout.style.gridTemplateColumns = tmpl(r.s, r.r);
        }
        function up(ev) {
          var r = calcFn(ev.clientX - startX, cSW, cRW, totalW);
          sw = r.s; rw = r.r;
          save(SK_SIDEBAR, sw); save(SK_RIGHT, rw);
          document.body.classList.remove('gi-resizing');
          window.dispatchEvent(new Event('resize'));
          document.removeEventListener('mousemove', mv);
          document.removeEventListener('mouseup', up);
        }
        document.addEventListener('mousemove', mv);
        document.addEventListener('mouseup', up);
      });
    }

    drag(hL, function (dx, cSW, cRW, W) {
      var s = clamp(cSW + dx, MIN_SIDEBAR, MAX_SIDEBAR);
      if (W - s - 8 - cRW < MIN_MAIN) s = clamp(W - MIN_MAIN - 8 - cRW, MIN_SIDEBAR, MAX_SIDEBAR);
      return { s: s, r: cRW };
    });
    drag(hR, function (dx, cSW, cRW, W) {
      var r = clamp(cRW - dx, MIN_RIGHT, MAX_RIGHT);
      if (W - cSW - 8 - r < MIN_MAIN) r = clamp(W - MIN_MAIN - 8 - cSW, MIN_RIGHT, MAX_RIGHT);
      return { s: cSW, r: r };
    });

    [hL, hR].forEach(function (h) {
      h.addEventListener('dblclick', function () {
        sw = DEF_SIDEBAR; rw = DEF_RIGHT;
        save(SK_SIDEBAR, sw); save(SK_RIGHT, rw);
        layout.style.gridTemplateColumns = tmpl(sw, rw);
        window.dispatchEvent(new Event('resize'));
        toast('Column widths reset');
      });
    });

    new MutationObserver(function () {
      layout.style.gridTemplateColumns = tmpl(sw, rw);
    }).observe(layout, { attributes: true, attributeFilter: ['class'] });

    window.addEventListener('resize', function () {
      var m = isMobile();
      hL.style.display = hR.style.display = m ? 'none' : '';
      if (m) layout.style.gridTemplateColumns = '';
    });
  }

  function mkHHandle(id, label) {
    var h = document.createElement('div');
    h.id = id; h.className = 'gi-hh';
    h.title = 'Drag to resize · Double-click to reset';
    h.setAttribute('role', 'separator');
    h.setAttribute('aria-label', label);
    return h;
  }

  /* ─── 2. Vertical resize (chart height) ───────────────────────────────── */
  function initVertical() {
    var chartWrap = document.getElementById('tv-chart-wrap');
    if (!chartWrap) return;

    var h = clamp(load(SK_CHART_H, DEF_CHART_H), MIN_CHART_H, MAX_CHART_H);
    setChartH(chartWrap, h);

    var handle = document.createElement('div');
    handle.id = 'gi-vh-chart';
    handle.className = 'gi-vh';
    handle.title = 'Drag to resize chart · Double-click to reset';
    handle.setAttribute('role', 'separator');
    handle.setAttribute('aria-label', 'Resize chart height');
    chartWrap.parentNode.insertBefore(handle, chartWrap.nextSibling);

    handle.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var startY = e.clientY, startH = chartWrap.offsetHeight;
      document.body.classList.add('gi-resizing-v');
      function mv(ev) { h = clamp(startH + ev.clientY - startY, MIN_CHART_H, MAX_CHART_H); setChartH(chartWrap, h); }
      function up()   { save(SK_CHART_H, h); document.body.classList.remove('gi-resizing-v'); window.dispatchEvent(new Event('resize')); document.removeEventListener('mousemove', mv); document.removeEventListener('mouseup', up); }
      document.addEventListener('mousemove', mv);
      document.addEventListener('mouseup', up);
    });

    handle.addEventListener('dblclick', function () {
      h = DEF_CHART_H; save(SK_CHART_H, h); setChartH(chartWrap, h);
      window.dispatchEvent(new Event('resize'));
      toast('Chart height reset');
    });
  }

  function setChartH(el, h) { el.style.height = el.style.minHeight = h + 'px'; }

  /* ─── Init ────────────────────────────────────────────────────────────── */
  function init() {
    if (!isMobile()) { initHorizontal(); initVertical(); }
    initCollapse();
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();

})();
