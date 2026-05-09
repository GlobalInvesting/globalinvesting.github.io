/**
 * Global Investing FX Terminal — Layout Resize Handles + Panel Collapse
 * v7.72.0
 *
 * Funcionalidades:
 *
 * 1. RESIZE HORIZONTAL — 3 columnas del grid (#layout)
 *    Handles de 4px entre sidebar↔main y main↔rightpanel.
 *    Anchos persistidos en localStorage.
 *    Double-click → reset a valores por defecto.
 *
 * 2. RESIZE VERTICAL — altura del chart (#tv-chart-wrap)
 *    Handle de 4px en el borde inferior del chart panel.
 *    Arrastrá para dar más/menos espacio al chart vs el resto.
 *    Altura persistida en localStorage.
 *    Double-click → reset a 290px por defecto.
 *
 * 3. PANEL COLLAPSE — botón ▾/▸ en cada panel-head
 *    Colapsa el contenido del panel manteniendo el panel-head visible.
 *    Estado persistido en localStorage por panel ID.
 *    Solo en desktop (> 900px).
 *
 * Storage keys:
 *   gi_layout_sidebar_w   — ancho sidebar px
 *   gi_layout_right_w     — ancho rightpanel px
 *   gi_layout_chart_h     — altura chart px
 *   gi_panel_collapsed_*  — estado colapsado por panel ID
 */

(function () {
  'use strict';

  /* ─── Config ─────────────────────────────────────────────────────────── */
  var SK_SIDEBAR   = 'gi_layout_sidebar_w';
  var SK_RIGHT     = 'gi_layout_right_w';
  var SK_CHART_H   = 'gi_layout_chart_h';
  var SK_COLLAPSED = 'gi_panel_collapsed_';

  var DEF_SIDEBAR  = 180;
  var DEF_RIGHT    = 220;
  var DEF_CHART_H  = 290;

  var MIN_SIDEBAR  = 120;  MAX_SIDEBAR  = 320;
  var MIN_RIGHT    = 150;  MAX_RIGHT    = 400;
  var MIN_CHART_H  = 120;  MAX_CHART_H  = 600;
  var MIN_MAIN     = 320;

  var MOBILE_BREAK = 900;

  /* Panels that get a collapse button.
   * For panels without a standard .panel-head, we target the header manually.
   * format: { id, contentSel, label }
   *   id:         panel element ID
   *   contentSel: CSS selector for the collapsible content (relative to panel)
   *   label:      human label for aria
   */
  var COLLAPSIBLE_PANELS = [
    { id: 'narrative',          contentSel: '.narr-text, .narr-meta',   label: 'Narrative' },
    { id: 'section-fxpairs',    contentSel: '#tv-tabs-outer, #tv-chart-toolbar, #tv-chart-wrap', label: 'Price Chart' },
    { id: 'section-fxtable',    contentSel: '.fx-pairs-scroll, #fx-pairs-scroll, table',         label: 'FX Pairs' },
    { id: 'split-upper-heatmap',contentSel: '.heatmap, #heatmap-grid',  label: 'Currency Heatmap' },
    { id: 'section-rates',      contentSel: '#rates-grid-wrap, #yield-canvas-wrap, #yield-spreads-wrap, #yield-spreads-cont', label: 'Rates & Yield Curve' },
    { id: 'section-positioning',contentSel: '#cot-rows, .cot-scroll',   label: 'COT Positioning' },
    { id: 'section-macro',      contentSel: '#alerts-container',        label: 'Alerts & Signals' },
  ];

  /* ─── Helpers ─────────────────────────────────────────────────────────── */
  function isMobile() { return window.innerWidth <= MOBILE_BREAK; }

  function load(key, def) {
    try { var v = localStorage.getItem(key); return v !== null ? parseInt(v, 10) : def; }
    catch { return def; }
  }
  function loadBool(key) {
    try { return localStorage.getItem(key) === '1'; }
    catch { return false; }
  }
  function save(key, val) {
    try { localStorage.setItem(key, String(val)); } catch {}
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function showToast(msg) {
    var t = document.getElementById('gi-layout-toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'gi-layout-toast';
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add('visible');
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { t.classList.remove('visible'); }, 1800);
  }

  /* ─── 1. HORIZONTAL RESIZE ──────────────────────────────────────────── */
  function initHorizontalResize() {
    var layout     = document.getElementById('layout');
    var main       = document.getElementById('main');
    var rightpanel = document.getElementById('rightpanel');
    if (!layout || !main || !rightpanel) return;

    var sw = clamp(load(SK_SIDEBAR, DEF_SIDEBAR), MIN_SIDEBAR, MAX_SIDEBAR);
    var rw = clamp(load(SK_RIGHT,   DEF_RIGHT),   MIN_RIGHT,   MAX_RIGHT);

    function buildTemplate(sw, rw) {
      var noRight = layout.classList.contains('layout--no-right');
      return noRight
        ? sw + 'px 4px 1fr 4px 0'
        : sw + 'px 4px 1fr 4px ' + rw + 'px';
    }

    // Create handles
    var hLeft  = makeHandle('gi-h-left',  'sidebar', 'Resize sidebar');
    var hRight = makeHandle('gi-h-right', 'right',   'Resize right panel');

    var sidebar = document.getElementById('sidebar');
    layout.insertBefore(hLeft,  main);
    layout.insertBefore(hRight, rightpanel);
    layout.style.gridTemplateColumns = buildTemplate(sw, rw);

    // Drag — left handle (sidebar resize)
    bindHDrag(hLeft, layout, function (dx, startSW, startRW, totalW) {
      var newSW = clamp(startSW + dx, MIN_SIDEBAR, MAX_SIDEBAR);
      if (totalW - newSW - 8 - startRW < MIN_MAIN) {
        newSW = clamp(totalW - MIN_MAIN - 8 - startRW, MIN_SIDEBAR, MAX_SIDEBAR);
      }
      return { sw: newSW, rw: startRW };
    }, buildTemplate, function (s, r) { sw = s; rw = r; save(SK_SIDEBAR, s); save(SK_RIGHT, r); });

    // Drag — right handle (rightpanel resize)
    bindHDrag(hRight, layout, function (dx, startSW, startRW, totalW) {
      var newRW = clamp(startRW - dx, MIN_RIGHT, MAX_RIGHT);
      if (totalW - startSW - 8 - newRW < MIN_MAIN) {
        newRW = clamp(totalW - MIN_MAIN - 8 - startSW, MIN_RIGHT, MAX_RIGHT);
      }
      return { sw: startSW, rw: newRW };
    }, buildTemplate, function (s, r) { sw = s; rw = r; save(SK_SIDEBAR, s); save(SK_RIGHT, r); });

    // Double-click reset
    [hLeft, hRight].forEach(function (h) {
      h.addEventListener('dblclick', function () {
        sw = DEF_SIDEBAR; rw = DEF_RIGHT;
        save(SK_SIDEBAR, sw); save(SK_RIGHT, rw);
        layout.style.gridTemplateColumns = buildTemplate(sw, rw);
        window.dispatchEvent(new Event('resize'));
        showToast('Layout reset to default');
      });
    });

    // Watch layout--no-right class toggle (rightpanel show/hide)
    new MutationObserver(function () {
      layout.style.gridTemplateColumns = buildTemplate(sw, rw);
    }).observe(layout, { attributes: true, attributeFilter: ['class'] });

    // Mobile: remove handles
    window.addEventListener('resize', function () {
      var mobile = isMobile();
      hLeft.style.display  = mobile ? 'none' : '';
      hRight.style.display = mobile ? 'none' : '';
      if (mobile) layout.style.gridTemplateColumns = '';
    });
  }

  function makeHandle(id, which, label) {
    var h = document.createElement('div');
    h.id = id;
    h.className = 'gi-layout-handle gi-layout-handle--h';
    h.dataset.which = which;
    h.title = 'Drag to resize · Double-click to reset';
    h.setAttribute('role', 'separator');
    h.setAttribute('aria-label', label);
    return h;
  }

  function bindHDrag(handle, layout, calcFn, buildTemplate, onCommit) {
    handle.addEventListener('mousedown', function (e) {
      if (isMobile()) return;
      e.preventDefault();
      var cols   = layout.style.gridTemplateColumns.split(' ');
      var startSW = parseInt(cols[0], 10) || DEF_SIDEBAR;
      var startRW = parseInt(cols[4], 10) || DEF_RIGHT;
      var totalW  = layout.offsetWidth;
      var startX  = e.clientX;

      document.body.classList.add('gi-resizing');

      function onMove(ev) {
        var result = calcFn(ev.clientX - startX, startSW, startRW, totalW);
        layout.style.gridTemplateColumns = buildTemplate(result.sw, result.rw);
      }
      function onUp(ev) {
        var result = calcFn(ev.clientX - startX, startSW, startRW, totalW);
        document.body.classList.remove('gi-resizing');
        onCommit(result.sw, result.rw);
        window.dispatchEvent(new Event('resize'));
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  /* ─── 2. VERTICAL RESIZE (chart height) ─────────────────────────────── */
  function initVerticalResize() {
    var chartWrap = document.getElementById('tv-chart-wrap');
    if (!chartWrap) return;

    var h = clamp(load(SK_CHART_H, DEF_CHART_H), MIN_CHART_H, MAX_CHART_H);
    applyChartH(chartWrap, h);

    var handle = document.createElement('div');
    handle.id        = 'gi-h-chart';
    handle.className = 'gi-layout-handle gi-layout-handle--v';
    handle.title     = 'Drag to resize chart · Double-click to reset';
    handle.setAttribute('role', 'separator');
    handle.setAttribute('aria-label', 'Resize chart height');
    // Insert after chartWrap
    chartWrap.parentNode.insertBefore(handle, chartWrap.nextSibling);

    handle.addEventListener('mousedown', function (e) {
      if (isMobile()) return;
      e.preventDefault();
      var startY  = e.clientY;
      var startH  = chartWrap.offsetHeight;
      document.body.classList.add('gi-resizing-v');

      function onMove(ev) {
        h = clamp(startH + (ev.clientY - startY), MIN_CHART_H, MAX_CHART_H);
        applyChartH(chartWrap, h);
      }
      function onUp() {
        save(SK_CHART_H, h);
        window.dispatchEvent(new Event('resize'));
        document.body.classList.remove('gi-resizing-v');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });

    handle.addEventListener('dblclick', function () {
      h = DEF_CHART_H;
      save(SK_CHART_H, h);
      applyChartH(chartWrap, h);
      window.dispatchEvent(new Event('resize'));
      showToast('Chart height reset');
    });
  }

  function applyChartH(el, h) {
    el.style.height    = h + 'px';
    el.style.minHeight = h + 'px';
  }

  /* ─── 3. PANEL COLLAPSE ─────────────────────────────────────────────── */
  function initPanelCollapse() {
    COLLAPSIBLE_PANELS.forEach(function (cfg) {
      var panel = document.getElementById(cfg.id);
      if (!panel) return;

      // Find the panel-head (or narr-label for narrative)
      var head = panel.querySelector('.panel-head') || panel.querySelector('.narr-label');
      if (!head) return;

      var collapsed = loadBool(SK_COLLAPSED + cfg.id);

      // Build collapse button
      var btn = document.createElement('button');
      btn.className = 'gi-collapse-btn';
      btn.setAttribute('aria-label', (collapsed ? 'Expand ' : 'Collapse ') + cfg.label);
      btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      btn.setAttribute('title', 'Collapse / expand panel · Double-click to collapse all');
      btn.innerHTML = collapsed ? '&#9656;' : '&#9662;';  // ▶ / ▾
      btn.dataset.panelId = cfg.id;

      // Insert at far right of panel-head
      head.style.position = 'relative';
      head.appendChild(btn);

      // Find content elements
      var contentEls = getContentEls(panel, head, cfg);

      // Apply initial state
      if (collapsed) collapsePanel(panel, contentEls, btn, true);

      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var isCollapsed = panel.classList.contains('gi-collapsed');
        if (isCollapsed) {
          expandPanel(panel, contentEls, btn, cfg);
        } else {
          collapsePanel(panel, contentEls, btn, false);
        }
        save(SK_COLLAPSED + cfg.id, panel.classList.contains('gi-collapsed') ? '1' : '0');
      });

      // Double-click on panel-head: collapse all
      head.addEventListener('dblclick', function (e) {
        if (e.target === btn) return; // handled by btn click
        toggleAllPanels();
      });
    });
  }

  function getContentEls(panel, head, cfg) {
    // Try selector first
    var els = [];
    if (cfg.contentSel) {
      cfg.contentSel.split(',').forEach(function (sel) {
        var found = panel.querySelector(sel.trim());
        if (found) els.push(found);
      });
    }
    // Fallback: all direct children except the head
    if (els.length === 0) {
      Array.from(panel.children).forEach(function (child) {
        if (child !== head && !child.classList.contains('panel-head') && !child.classList.contains('narr-label')) {
          els.push(child);
        }
      });
    }
    return els;
  }

  function collapsePanel(panel, contentEls, btn, instant) {
    panel.classList.add('gi-collapsed');
    contentEls.forEach(function (el) {
      if (instant) {
        el.style.display = 'none';
      } else {
        el.style.overflow  = 'hidden';
        el.style.maxHeight = el.scrollHeight + 'px';
        // Force reflow
        el.offsetHeight; // eslint-disable-line no-unused-expressions
        el.style.transition = 'max-height 0.22s ease, opacity 0.18s ease';
        el.style.maxHeight  = '0';
        el.style.opacity    = '0';
        setTimeout(function () { el.style.display = 'none'; el.style.transition = ''; el.style.maxHeight = ''; el.style.opacity = ''; }, 230);
      }
    });
    btn.innerHTML = '&#9656;';  // ▶
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-label', btn.getAttribute('aria-label').replace('Collapse', 'Expand'));
  }

  function expandPanel(panel, contentEls, btn, cfg) {
    panel.classList.remove('gi-collapsed');
    contentEls.forEach(function (el) {
      el.style.display = '';
      var target = el.scrollHeight;
      el.style.overflow   = 'hidden';
      el.style.maxHeight  = '0';
      el.style.opacity    = '0';
      el.offsetHeight; // force reflow
      el.style.transition = 'max-height 0.22s ease, opacity 0.18s ease';
      el.style.maxHeight  = target + 'px';
      el.style.opacity    = '1';
      setTimeout(function () { el.style.transition = ''; el.style.maxHeight = ''; el.style.overflow = ''; el.style.opacity = ''; window.dispatchEvent(new Event('resize')); }, 230);
    });
    btn.innerHTML = '&#9662;';  // ▾
    btn.setAttribute('aria-expanded', 'true');
    btn.setAttribute('aria-label', btn.getAttribute('aria-label').replace('Expand', 'Collapse'));
  }

  var _allCollapsed = false;
  function toggleAllPanels() {
    _allCollapsed = !_allCollapsed;
    COLLAPSIBLE_PANELS.forEach(function (cfg) {
      var panel = document.getElementById(cfg.id);
      if (!panel) return;
      var btn = panel.querySelector('.gi-collapse-btn');
      var head = panel.querySelector('.panel-head') || panel.querySelector('.narr-label');
      if (!btn || !head) return;
      var contentEls = getContentEls(panel, head, cfg);
      if (_allCollapsed) {
        collapsePanel(panel, contentEls, btn, false);
      } else {
        expandPanel(panel, contentEls, btn, cfg);
      }
      save(SK_COLLAPSED + cfg.id, _allCollapsed ? '1' : '0');
    });
    showToast(_allCollapsed ? 'All panels collapsed' : 'All panels expanded');
  }

  /* ─── Init ───────────────────────────────────────────────────────────── */
  function init() {
    if (!isMobile()) {
      initHorizontalResize();
      initVerticalResize();
    }
    initPanelCollapse(); // collapse buttons visible on all viewports
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Public API
  window.giToggleAllPanels = toggleAllPanels;

})();
