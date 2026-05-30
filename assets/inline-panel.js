// ═══════════════════════════════════════════════════════════════════
// INLINE PANEL SYSTEM  v1.3.1
// File: assets/inline-panel.js
//
//   LEFT center  (#split-upper):  Carry Trade · Heatmap
//   RIGHT center (#split-lower):  Correlations · CB Rates · COT
//
// v1.3.0: Fix height:0 bug. Instead of prepending with height:100%
//         (which resolves to 0 in a scrollable flex container), we
//         now HIDE existing children and insert the panel as the sole
//         visible child, sized with min-height to fill the container.
// ═══════════════════════════════════════════════════════════════════

(function () {
  'use strict';

  var LS_KEY  = 'gi_split_layout';
  var IP_ATTR = 'data-inline-panel';
  var HIDDEN_ATTR = 'data-ip-hidden';

  // ── Ensure split-layout is active ─────────────────────────────────
  function _ensureSplit() {
    var main  = document.getElementById('main');
    var upper = document.getElementById('split-upper');
    var lower = document.getElementById('split-lower');
    if (!main || !upper || !lower) return null;

    if (!main.classList.contains('split-layout')) {
      var btn    = document.getElementById('split-layout-btn');
      var handle = document.getElementById('split-drag-handle');

      main.classList.add('split-layout');
      if (btn)    { btn.classList.add('active'); btn.setAttribute('aria-pressed','true'); }
      if (handle) handle.style.display = '';
      upper.style.width = '55%';
      upper.style.flex  = 'none';

      try { localStorage.setItem(LS_KEY, JSON.stringify({ active: true, leftPct: 55 })); } catch(e) {}
    }
    return { upper: upper, lower: lower };
  }

  // ── Hide existing children of target, return restore function ─────
  function _hideChildren(target) {
    var hidden = [];
    Array.from(target.children).forEach(function(child) {
      if (!child.hasAttribute(IP_ATTR)) {
        hidden.push({ el: child, display: child.style.display });
        child.style.display = 'none';
        child.setAttribute(HIDDEN_ATTR, '1');
      }
    });
    return function restore() {
      hidden.forEach(function(item) {
        item.el.style.display = item.display;
        item.el.removeAttribute(HIDDEN_ATTR);
      });
    };
  }

  // ── Create inline shell ────────────────────────────────────────────
  function _makeShell(target, title, onClose) {
    // Remove any existing inline panel in this target first
    var old = target.querySelector('[' + IP_ATTR + ']');
    if (old) old.remove();
    // Restore any previously hidden children
    Array.from(target.querySelectorAll('[' + HIDDEN_ATTR + ']')).forEach(function(el) {
      el.style.display = '';
      el.removeAttribute(HIDDEN_ATTR);
    });

    // Hide existing children so the panel can take full height
    var restoreChildren = _hideChildren(target);

    var wrap = document.createElement('div');
    wrap.setAttribute(IP_ATTR, '1');
    // Use min-height instead of height:100% — works correctly in scrollable flex containers
    wrap.style.cssText = 'display:flex;flex-direction:column;min-height:calc(100vh - 100px);background:var(--bg);overflow:hidden;';

    // Header bar
    var hd = document.createElement('div');
    hd.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 10px 4px;border-bottom:1px solid var(--border2);flex-shrink:0;background:var(--bg2);';

    var titleEl = document.createElement('span');
    titleEl.style.cssText = 'font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);font-family:var(--font-mono);';
    titleEl.textContent = title;

    var closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.setAttribute('aria-label', 'Close panel');
    closeBtn.style.cssText = 'background:none;border:none;color:var(--text3);cursor:pointer;font-size:11px;line-height:1;padding:2px 5px;border-radius:3px;transition:color .1s;';
    closeBtn.onmouseenter = function() { closeBtn.style.color = 'var(--text)'; };
    closeBtn.onmouseleave = function() { closeBtn.style.color = 'var(--text3)'; };
    closeBtn.onclick = function() {
      wrap.remove();
      restoreChildren();
      if (typeof onClose === 'function') onClose();
    };

    hd.appendChild(titleEl);
    hd.appendChild(closeBtn);

    var body = document.createElement('div');
    body.style.cssText = 'flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;display:flex;flex-direction:column;';

    wrap.appendChild(hd);
    wrap.appendChild(body);
    target.prepend(wrap);
    target.scrollTop = 0;

    return body;
  }

  // ── Synchronously transplant modal into inline body ────────────────
  function _transplant(body, bdId, modalId, closeId, modalExtraCSS) {
    var bd    = document.getElementById(bdId);
    var modal = document.getElementById(modalId);
    if (!bd || !modal) return false;

    bd.style.cssText = 'display:block!important;position:static!important;background:none!important;padding:0!important;animation:none!important;z-index:auto!important;';

    var base = 'width:100%!important;max-width:none!important;height:100%!important;max-height:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;position:static!important;';
    modal.style.cssText = base + (modalExtraCSS || '');

    if (closeId) {
      var oc = document.getElementById(closeId);
      if (oc) oc.style.display = 'none';
    }

    body.appendChild(bd);
    return true;
  }

  // ── Restore modal back to document.body on panel close ────────────
  function _restore(bdId, closeId, bdOrigCSS, modalId) {
    var bd = document.getElementById(bdId);
    if (!bd) return;
    if (bd.parentNode !== document.body) document.body.appendChild(bd);
    bd.style.cssText = bdOrigCSS || 'display:none;';
    if (closeId) {
      var oc = document.getElementById(closeId);
      if (oc) oc.style.display = '';
    }
    if (modalId) {
      var modal = document.getElementById(modalId);
      if (modal) modal.style.cssText = '';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Real Carry Modal → #split-upper (LEFT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenRCM = window.openRealCarryModal;

  window.openRealCarryModal = function(longCcy, shortCcy) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenRCM && _origOpenRCM(longCcy, shortCcy); return; }

    var label = (longCcy && shortCcy) ? longCcy + '/' + shortCcy : '8 major currencies';
    var body  = _makeShell(panels.upper, 'Real Rate Carry · ' + label, function() {
      _restore('rcm-bd', 'rcm-close',
        'position:fixed;inset:0;z-index:9200;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:16px;',
        'rcm-modal');
    });

    var bdPre = document.getElementById('rcm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenRCM && _origOpenRCM(longCcy, shortCcy);

    if (!_transplant(body, 'rcm-bd', 'rcm-modal', 'rcm-close',
        'display:flex!important;flex-direction:column!important;overflow:hidden!important;')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Carry data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Heatmap Modal → #split-upper (LEFT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenHM = window.openHeatmapModal;

  window.openHeatmapModal = function(ccy, strengths, rtCache) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenHM && _origOpenHM(ccy, strengths, rtCache); return; }

    var body = _makeShell(panels.upper, 'Currency Strength · ' + (ccy || ''), function() {
      var hmBd = document.getElementById('hm-bd');
      if (hmBd) hmBd.remove();
    });

    var bdPre = document.getElementById('hm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenHM && _origOpenHM(ccy, strengths, rtCache);

    if (!_transplant(body, 'hm-bd', 'hm-modal', 'hm-close',
        'display:flex!important;flex-direction:column!important;overflow:hidden!important;')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Heatmap unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Correlation Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCorr = window.openCorrModal;

  window.openCorrModal = function(corrObj) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCorr && _origOpenCorr(corrObj); return; }

    var a = corrObj ? corrObj.a : '';
    var b = corrObj ? corrObj.b : '';
    var body = _makeShell(panels.lower, 'Correlations · ' + a + ' / ' + b, function() {
      if (typeof window.closeCorrModal === 'function') window.closeCorrModal();
    });

    var bdPre = document.getElementById('cm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCorr && _origOpenCorr(corrObj);

    var cmBd    = document.getElementById('cm-bd');
    var cmModal = document.getElementById('cm-modal');
    if (!cmBd || !cmModal) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Correlation data unavailable.</div>';
      return;
    }

    cmBd.style.cssText    = 'display:block!important;position:static!important;background:none!important;padding:0!important;animation:none!important;z-index:auto!important;';
    cmModal.style.cssText = 'width:100%!important;max-width:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;height:auto!important;max-height:none!important;position:static!important;';

    var oc = document.getElementById('cm-close');
    if (oc) oc.style.display = 'none';

    body.appendChild(cmBd);
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: CB Rates Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCBR = window.openCBRatesModal;

  window.openCBRatesModal = function(ccy, obs, bankInfo, meetingData) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCBR && _origOpenCBR(ccy, obs, bankInfo, meetingData); return; }

    var body = _makeShell(panels.lower, 'CB Rates · ' + (ccy || ''), function() {
      _restore('cbr-bd', 'cbr-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;',
        'cbr-modal');
    });

    var bdPre = document.getElementById('cbr-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCBR && _origOpenCBR(ccy, obs, bankInfo, meetingData);

    if (!_transplant(body, 'cbr-bd', 'cbr-modal', 'cbr-m-close', '')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">CB rate data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: COT Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenCOT = window.openCOTModal;

  window.openCOTModal = function(ccy, data) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenCOT && _origOpenCOT(ccy, data); return; }

    var body = _makeShell(panels.lower, 'COT Positioning · ' + (ccy || ''), function() {
      _restore('cot-bd', 'cot-m-close',
        'position:fixed;inset:0;z-index:9100;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;padding:12px;',
        'cot-modal');
    });

    var bdPre = document.getElementById('cot-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenCOT && _origOpenCOT(ccy, data);

    if (!_transplant(body, 'cot-bd', 'cot-modal', 'cot-m-close', '')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">COT data unavailable.</div>';
    }
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: YC Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenYC = window.openYCModal;

  window.openYCModal = function(tenorData) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenYC && _origOpenYC(tenorData); return; }

    var body = _makeShell(panels.lower, 'Yield Curve · US Treasury', function() {
      if (typeof window.closeYCModal === 'function') window.closeYCModal();
    });

    var bdPre = document.getElementById('ycm-bd');
    if (bdPre) { bdPre.style.display = 'none'; }

    _origOpenYC && _origOpenYC(tenorData);

    var ycBd    = document.getElementById('ycm-bd');
    var ycModal = document.getElementById('ycm-modal');
    if (!ycBd || !ycModal) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Yield curve data unavailable.</div>';
      return;
    }

    ycBd.style.cssText    = 'display:block!important;position:static!important;background:none!important;padding:0!important;z-index:auto!important;border:none!important;';
    ycModal.style.cssText = 'width:100%!important;max-width:none!important;border-radius:0!important;border:none!important;box-shadow:none!important;animation:none!important;background:var(--bg)!important;position:static!important;height:auto!important;max-height:none!important;display:flex!important;flex-direction:column!important;';

    var oc = document.getElementById('ycm-close');
    if (oc) oc.style.display = 'none';

    body.appendChild(ycBd);
    document.body.style.overflow = '';
  };

  // ═══════════════════════════════════════════════════════════════════
  // INTERCEPT: Economic Surprises Modal → #split-lower (RIGHT)
  // ═══════════════════════════════════════════════════════════════════
  var _origOpenESM = window.openEconSurprisesModal;

  window.openEconSurprisesModal = function(ccy) {
    var panels = _ensureSplit();
    if (!panels) { _origOpenESM && _origOpenESM(ccy); return; }

    var body = _makeShell(panels.lower, 'Economic Surprises \u00b7 ' + (ccy || '8 major currencies'), function() {
      if (typeof window.closeESModal === 'function') window.closeESModal();
    });

    // Override shell body from overflow-y:auto → overflow:hidden.
    // Without this, body is a scroll container and esm-modal's height:100%
    // resolves to auto (circular) — the flex:1;min-height:0 chain collapses,
    // #esm-events-wrap never scrolls, and position:sticky floats over the chart.
    body.style.overflowY = 'hidden';

    var bdPre = document.getElementById('esm-bd');
    if (bdPre) bdPre.style.display = 'none';

    _origOpenESM && _origOpenESM(ccy);

    if (!_transplant(body, 'esm-bd', 'esm-modal', 'esm-close',
        'display:flex!important;flex-direction:column!important;overflow:hidden!important;')) {
      body.innerHTML = '<div style="padding:12px;font-size:11px;color:var(--text3);">Economic Surprises data unavailable.</div>';
    }
    document.body.style.overflow = '';

    // Bug 1 (mobile): split-lower has overflow:visible on mobile so the panel
    // lands below the fold. Scroll it into view after the transplant.
    requestAnimationFrame(function() {
      var wrap = panels.lower.querySelector('[data-inline-panel]');
      if (wrap) wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };

  window._showInlinePanel   = _makeShell;
  window._ensureInlineSplit = _ensureSplit;

})();
