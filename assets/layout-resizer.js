
(function () {
  'use strict';

  var STORAGE_SIDEBAR = 'gi_layout_sidebar_w';
  var STORAGE_RIGHT   = 'gi_layout_right_w';

  var DEFAULT_SIDEBAR = 180;
  var DEFAULT_RIGHT   = 220;
  var MIN_SIDEBAR     = 120;
  var MAX_SIDEBAR     = 320;
  var MIN_RIGHT       = 150;
  var MAX_RIGHT       = 400;
  var MIN_MAIN        = 320;   
  var MOBILE_BREAK    = 900;   

  function load(key, def) {
    try {
      var v = localStorage.getItem(key);
      return v !== null ? parseInt(v, 10) : def;
    } catch { return def; }
  }
  function save(key, val) {
    try { localStorage.setItem(key, val); } catch { }
  }

  function isMobile() { return window.innerWidth <= MOBILE_BREAK; }

  function applyLayout(sidebarW, rightW) {
    var layout = document.getElementById('layout');
    if (!layout) return;
    var noRight = layout.classList.contains('layout--no-right');
    if (noRight) {
      layout.style.gridTemplateColumns = sidebarW + 'px 1fr 0';
    } else {
      layout.style.gridTemplateColumns = sidebarW + 'px 1fr ' + rightW + 'px';
    }
  }

  function createHandle(id, which) {
    var h = document.createElement('div');
    h.id        = id;
    h.className = 'gi-layout-handle';
    h.dataset.which = which;   
    h.setAttribute('title', 'Drag to resize · Double-click to reset');
    h.setAttribute('role', 'separator');
    h.setAttribute('aria-label', which === 'sidebar' ? 'Resize sidebar' : 'Resize right panel');
    return h;
  }

  function injectHandles() {
    var layout     = document.getElementById('layout');
    var sidebar    = document.getElementById('sidebar');
    var main       = document.getElementById('main');
    var rightpanel = document.getElementById('rightpanel');
    if (!layout || !sidebar || !main || !rightpanel) return;

    var hLeft = createHandle('gi-handle-left', 'sidebar');
    layout.insertBefore(hLeft, main);

    var hRight = createHandle('gi-handle-right', 'right');
    layout.insertBefore(hRight, rightpanel);

    var noRight = layout.classList.contains('layout--no-right');
    var sw = load(STORAGE_SIDEBAR, DEFAULT_SIDEBAR);
    var rw = load(STORAGE_RIGHT,   DEFAULT_RIGHT);

    sw = Math.max(MIN_SIDEBAR, Math.min(MAX_SIDEBAR, sw));
    rw = Math.max(MIN_RIGHT,   Math.min(MAX_RIGHT,   rw));

    function buildTemplate(sidebarW, rightW) {
      var noRight = document.getElementById('layout').classList.contains('layout--no-right');
      return noRight
        ? sidebarW + 'px 4px 1fr 4px 0'
        : sidebarW + 'px 4px 1fr 4px ' + rightW + 'px';
    }

    layout.style.gridTemplateColumns = buildTemplate(sw, rw);

    bindDrag(hLeft,  'sidebar', sw, rw, buildTemplate);
    bindDrag(hRight, 'right',   sw, rw, buildTemplate);

    [hLeft, hRight].forEach(function(h) {
      h.addEventListener('dblclick', function() {
        sw = DEFAULT_SIDEBAR;
        rw = DEFAULT_RIGHT;
        save(STORAGE_SIDEBAR, sw);
        save(STORAGE_RIGHT,   rw);
        layout.style.gridTemplateColumns = buildTemplate(sw, rw);
        showToast('Layout reset to default');
      });
    });
  }

  function bindDrag(handleEl, which, initialSW, initialRW, buildTemplate) {
    var layout = document.getElementById('layout');
    var sw = initialSW;
    var rw = initialRW;

    handleEl.addEventListener('mousedown', function(e) {
      if (isMobile()) return;
      e.preventDefault();

      var cols = layout.style.gridTemplateColumns.split(' ');
      sw = parseInt(cols[0], 10) || DEFAULT_SIDEBAR;
      rw = parseInt(cols[4], 10) || DEFAULT_RIGHT;

      var startX   = e.clientX;
      var startSW  = sw;
      var startRW  = rw;
      var totalW   = layout.offsetWidth;

      document.body.classList.add('gi-resizing');

      function onMove(ev) {
        var dx = ev.clientX - startX;
        if (which === 'sidebar') {
          sw = Math.max(MIN_SIDEBAR, Math.min(MAX_SIDEBAR, startSW + dx));
          var mainAvail = totalW - sw - 8 - rw;
          if (mainAvail < MIN_MAIN) {
            sw = totalW - MIN_MAIN - 8 - rw;
            sw = Math.max(MIN_SIDEBAR, sw);
          }
        } else {
          rw = Math.max(MIN_RIGHT, Math.min(MAX_RIGHT, startRW - dx));
          var mainAvail2 = totalW - sw - 8 - rw;
          if (mainAvail2 < MIN_MAIN) {
            rw = totalW - MIN_MAIN - 8 - sw;
            rw = Math.max(MIN_RIGHT, rw);
          }
        }
        layout.style.gridTemplateColumns = buildTemplate(sw, rw);
      }

      function onUp() {
        document.body.classList.remove('gi-resizing');
        save(STORAGE_SIDEBAR, sw);
        save(STORAGE_RIGHT,   rw);
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
        window.dispatchEvent(new Event('resize'));
      }

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  }

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
    t._timer = setTimeout(function() { t.classList.remove('visible'); }, 1800);
  }

  function watchLayoutClass() {
    var layout = document.getElementById('layout');
    if (!layout) return;
    var observer = new MutationObserver(function() {
      if (isMobile()) return;
      var cols = layout.style.gridTemplateColumns.split(' ');
      var sw = parseInt(cols[0], 10) || load(STORAGE_SIDEBAR, DEFAULT_SIDEBAR);
      var rw = load(STORAGE_RIGHT, DEFAULT_RIGHT);
      var noRight = layout.classList.contains('layout--no-right');
      layout.style.gridTemplateColumns = noRight
        ? sw + 'px 4px 1fr 4px 0'
        : sw + 'px 4px 1fr 4px ' + rw + 'px';
    });
    observer.observe(layout, { attributes: true, attributeFilter: ['class'] });
  }

  function init() {
    if (isMobile()) return;
    injectHandles();
    watchLayoutClass();

    window.addEventListener('resize', function() {
      var layout = document.getElementById('layout');
      if (!layout) return;
      var hL = document.getElementById('gi-handle-left');
      var hR = document.getElementById('gi-handle-right');
      if (!hL || !hR) return;
      if (isMobile()) {
        hL.style.display = 'none';
        hR.style.display = 'none';
        layout.style.gridTemplateColumns = '';
      } else {
        hL.style.display = '';
        hR.style.display = '';
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
