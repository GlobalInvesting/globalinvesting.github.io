/**
 * Global Investing FX Terminal — Layout Resize Handles + Panel Collapse
 * v7.72.1 — fix Rates collapse bug + expanded panel coverage
 *
 * 1. RESIZE HORIZONTAL — handles entre las 3 columnas del grid
 * 2. RESIZE VERTICAL   — handle en el borde inferior del chart
 * 3. PANEL COLLAPSE    — botón ▾/▸ en cada panel-head
 *
 * Fix v7.72.1: getContent() SIEMPRE excluye el panel-head — resuelve el bug
 * donde Rates colapsaba el botón junto con el contenido y no se podía expandir.
 * Agregados: section-crossasset, section-risk, section-sessions.
 *
 * Storage keys:
 *   gi_layout_sidebar_w / gi_layout_right_w / gi_layout_chart_h
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

  /*
   * Panel definitions.
   * headSel: selector for the header element (gets the collapse button).
   * All direct children EXCEPT the head are collapsed — no contentSel needed.
   * This is the key fix: never risk hiding the head itself.
   */
  var PANELS = [
    { id: 'narrative',           headSel: '.narr-label',  label: 'Narrative' },
    { id: 'section-fxpairs',     headSel: '.panel-head',  label: 'Price Chart' },
    { id: 'section-fxtable',     headSel: '.panel-head',  label: 'FX Pairs' },
    { id: 'split-upper-heatmap', headSel: '.panel-head',  label: 'Currency Heatmap' },
    { id: 'section-crossasset',  headSel: '.panel-head',  label: 'Cross-Asset' },
    { id: 'section-risk',        headSel: '.panel-head',  label: 'Risk Monitor' },
    { id: 'section-rates',       headSel: '.panel-head',  label: 'Rates & Yield Curve' },
    { id: 'section-positioning', headSel: '.panel-head',  label: 'COT Positioning' },
    { id: 'section-sessions',    headSel: '.panel-head',  label: 'Market Sessions' },
    { id: 'section-macro',       headSel: '.panel-head',  label: 'Alerts & Signals' },
  ];

  /* ─── Helpers ──────────────────────────────────────────────────────────── */
  function isMobile() { return window.innerWidth <= MOBILE; }
  function load(k, d)  { try { var v = localStorage.getItem(k); return v !== null ? parseInt(v, 10) : d; } catch(e) { return d; } }
  function loadB(k)    { try { return localStorage.getItem(k) === '1'; } catch(e) { return false; } }
  function save(k, v)  { try { localStorage.setItem(k, String(v)); } catch(e) {} }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function toast(msg) {
    var t = document.getElementById('gi-toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'gi-toast';
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add('visible');
    clearTimeout(t._t);
    t._t = setTimeout(function () { t.classList.remove('visible'); }, 1800);
  }

  /*
   * Safe content getter: all direct children of panel EXCEPT the head element.
   * This guarantees the collapse button (inside head) is never hidden,
   * fixing the Rates & Yield Curve disappearing-button bug.
   */
  function getContent(panel, headEl) {
    return Array.from(panel.children).filter(function (c) { return c !== headEl; });
  }

  /* ─── Animate collapse / expand ───────────────────────────────────────── */
  function collapse(els, instant) {
    els.forEach(function (el) {
      if (instant) { el.style.display = 'none'; return; }
      var h = el.scrollHeight;
      el.style.overflow  = 'hidden';
      el.style.maxHeight = h + 'px';
      el.style.opacity   = '1';
      el.offsetHeight; // force reflow
      el.style.transition = 'max-height .22s ease, opacity .18s ease';
      el.style.maxHeight  = '0';
      el.style.opacity    = '0';
      setTimeout(function () {
        el.style.display    = 'none';
        el.style.transition = '';
        el.style.maxHeight  = '';
        el.style.overflow   = '';
        el.style.opacity    = '';
      }, 230);
    });
  }

  function expand(els) {
    els.forEach(function (el) {
      el.style.display = '';
      el.offsetHeight;
      var target = el.scrollHeight;
      el.style.overflow  = 'hidden';
      el.style.maxHeight = '0';
      el.style.opacity   = '0';
      el.offsetHeight;
      el.style.transition = 'max-height .22s ease, opacity .18s ease';
      el.style.maxHeight  = target + 'px';
      el.style.opacity    = '1';
      setTimeout(function () {
        el.style.transition = '';
        el.style.maxHeight  = '';
        el.style.overflow   = '';
        el.style.opacity    = '';
        window.dispatchEvent(new Event('resize'));
      }, 230);
    });
  }

  /* ─── 3. Panel Collapse ────────────────────────────────────────────────── */
  function initCollapse() {
    PANELS.forEach(function (cfg) {
      var panel = document.getElementById(cfg.id);
      if (!panel) return;
      var head = panel.querySelector(cfg.headSel);
      if (!head) return;

      head.style.display    = 'flex';
      head.style.alignItems = 'center';

      var btn = document.createElement('button');
      btn.className = 'gi-collapse-btn';
      btn.title     = 'Collapse / expand · Double-click header to toggle all';
      btn.setAttribute('aria-label', 'Collapse ' + cfg.label);

      var isCollapsed = loadB(SK_COL + cfg.id);
      btn.innerHTML = isCollapsed ? '&#9656;' : '&#9662;';
      head.appendChild(btn);

      var content = getContent(panel, head);

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

      // Expose on element for toggleAll
      panel._giCollapse    = doCollapse;
      panel._giExpand      = doExpand;
      panel._giIsCollapsed = function () { return panel.classList.contains('gi-collapsed'); };

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
      if (!_allCollapsed && panel._giExpand   &&  panel._giIsCollapsed()) panel._giExpand();
    });
    toast(_allCollapsed ? 'All panels collapsed' : 'All panels expanded');
  }
  window.giToggleAllPanels = toggleAll;

  /* ─── 1. Horizontal resize ─────────────────────────────────────────────── */
  function initHorizontal() {
    var layout     = document.getElementById('layout');
    var main       = document.getElementById('main');
    var rightpanel = document.getElementById('rightpanel');
    if (!layout || !main || !rightpanel) return;

    var sw = clamp(load(SK_SIDEBAR, DEF_SIDEBAR), MIN_SIDEBAR, MAX_SIDEBAR);
    var rw = clamp(load(SK_RIGHT,   DEF_RIGHT),   MIN_RIGHT,   MAX_RIGHT);

    function tmpl(s, r) {
      return layout.classList.contains('layout--no-right')
        ? s + 'px 4px 1fr 4px 0'
        : s + 'px 4px 1fr 4px ' + r + 'px';
    }

    var hL = mkHH('gi-hh-left',  'Resize sidebar');
    var hR = mkHH('gi-hh-right', 'Resize right panel');
    layout.insertBefore(hL, main);
    layout.insertBefore(hR, rightpanel);
    layout.style.gridTemplateColumns = tmpl(sw, rw);

    function makeDrag(handle, calcFn) {
      handle.addEventListener('mousedown', function (e) {
        e.preventDefault();
        var cols = layout.style.gridTemplateColumns.split(' ');
        var cSW  = parseInt(cols[0], 10) || sw;
        var cRW  = parseInt(cols[4], 10) || rw;
        var W    = layout.offsetWidth;
        var x0   = e.clientX;
        document.body.classList.add('gi-resizing');
        function mv(ev) {
          var r = calcFn(ev.clientX - x0, cSW, cRW, W);
          layout.style.gridTemplateColumns = tmpl(r.s, r.r);
        }
        function up(ev) {
          var r = calcFn(ev.clientX - x0, cSW, cRW, W);
          sw = r.s; rw = r.r;
          save(SK_SIDEBAR, sw); save(SK_RIGHT, rw);
          document.body.classList.remove('gi-resizing');
          window.dispatchEvent(new Event('resize'));
          document.removeEventListener('mousemove', mv);
          document.removeEventListener('mouseup',   up);
        }
        document.addEventListener('mousemove', mv);
        document.addEventListener('mouseup',   up);
      });
    }

    makeDrag(hL, function (dx, cSW, cRW, W) {
      var s = clamp(cSW + dx, MIN_SIDEBAR, MAX_SIDEBAR);
      if (W - s - 8 - cRW < MIN_MAIN) s = clamp(W - MIN_MAIN - 8 - cRW, MIN_SIDEBAR, MAX_SIDEBAR);
      return { s: s, r: cRW };
    });
    makeDrag(hR, function (dx, cSW, cRW, W) {
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

  function mkHH(id, label) {
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
    setH(chartWrap, h);

    var handle = document.createElement('div');
    handle.id = 'gi-vh-chart';
    handle.className = 'gi-vh';
    handle.title = 'Drag to resize chart · Double-click to reset';
    handle.setAttribute('role', 'separator');
    handle.setAttribute('aria-label', 'Resize chart height');
    chartWrap.parentNode.insertBefore(handle, chartWrap.nextSibling);

    handle.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var y0 = e.clientY, h0 = chartWrap.offsetHeight;
      document.body.classList.add('gi-resizing-v');
      function mv(ev) {
        h = clamp(h0 + ev.clientY - y0, MIN_CHART_H, MAX_CHART_H);
        setH(chartWrap, h);
      }
      function up() {
        save(SK_CHART_H, h);
        document.body.classList.remove('gi-resizing-v');
        window.dispatchEvent(new Event('resize'));
        document.removeEventListener('mousemove', mv);
        document.removeEventListener('mouseup',   up);
      }
      document.addEventListener('mousemove', mv);
      document.addEventListener('mouseup',   up);
    });

    handle.addEventListener('dblclick', function () {
      h = DEF_CHART_H;
      save(SK_CHART_H, h);
      setH(chartWrap, h);
      window.dispatchEvent(new Event('resize'));
      toast('Chart height reset');
    });
  }

  function setH(el, h) { el.style.height = el.style.minHeight = h + 'px'; }

  /* ─── Init ─────────────────────────────────────────────────────────────── */
  function init() {
    if (!isMobile()) { initHorizontal(); initVertical(); }
    initCollapse();
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();

})();
